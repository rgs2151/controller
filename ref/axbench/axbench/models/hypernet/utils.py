import torch
from axbench.utils.model_utils import get_prefix_length, get_suffix_length
from axbench.utils.constants import EMPTY_CONCEPT
import pandas as pd

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def prepare_df_combined(
        original_df, negative_df, tokenizer, 
        binarize, train_on_negative, is_chat_model, output_length, model_name, max_num_of_examples=None,
        replace_negative_description=True, negative_example_ratio=None):
    suffix_length, suffix_str = get_suffix_length(tokenizer)
    # print(f"Suffix length for {model_name}: {suffix_length}, Suffix string: {suffix_str}")
    # assign input and output containing concept with 1, otherwise 0
    positive_df = original_df[original_df["category"] == "positive"]
    
    # assert len(positive_df) > len(negative_df), "Positive examples should be more than negative examples."
    
    if negative_example_ratio is not None:
        positive_example_per_concept = len(positive_df) // len(positive_df["output_concept"].unique())
        negative_example_per_concept = positive_example_per_concept * negative_example_ratio
        
        assert negative_example_per_concept < len(negative_df), "Negative examples should be more than positive examples."
        
        negative_dfs = []
        for concept_id in positive_df["concept_id"].unique():
            concept_description = positive_df[positive_df["concept_id"] == concept_id]["output_concept"].values[0]
            
            # Randomly sample negative examples for each concept from negative_df
            # and replace the description with the positive one.
            concept_negative_df = negative_df[negative_df["output_concept"] == EMPTY_CONCEPT].sample(
                n=negative_example_per_concept
            )
            concept_negative_df = concept_negative_df.copy()
            
            if replace_negative_description:
                concept_negative_df["output_concept"] = concept_description
                concept_negative_df["concept_id"] = concept_id

            negative_dfs.append(concept_negative_df)
            
        negative_df = pd.concat(negative_dfs, axis=0, ignore_index=True)
        
        # shuffle the negative examples
        negative_df = negative_df.sample(frac=1).reset_index(drop=True)

    if binarize:
        if is_chat_model:
            if model_name == "meta-llama/Llama-3.1-8B-Instruct":
                def apply_chat_template(row):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."}, 
                        {"role": "user", "content": row["input"]},
                        {"role": "assistant", "content": row["output"]}
                    ]
                    nobos = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True)[1:-suffix_length]
                    return tokenizer.decode(nobos)
                positive_df = positive_df.copy()
                negative_df = negative_df.copy()
                positive_df['combined'] = positive_df.apply(apply_chat_template, axis=1)
                negative_df['combined'] = negative_df.apply(apply_chat_template, axis=1)
            else:
                def apply_chat_template(row):
                    messages = [
                        {"role": "user", "content": row["input"]},
                        {"role": "assistant", "content": row["output"]}
                    ]
                    nobos = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True)[1:-suffix_length]
                    return tokenizer.decode(nobos)
                positive_df = positive_df.copy()
                negative_df = negative_df.copy()
                positive_df['combined'] = positive_df.apply(apply_chat_template, axis=1)
                negative_df['combined'] = negative_df.apply(apply_chat_template, axis=1)
        else:
            positive_df = positive_df.copy()
            negative_df = negative_df.copy()
            positive_df['combined'] = positive_df['input'] + positive_df['output']
            negative_df['combined'] = negative_df['input'] + negative_df['output']
        positive_df = pd.DataFrame(positive_df[['combined']]).rename(columns={'combined': 'input'})
        negative_df = pd.DataFrame(negative_df[['combined']]).rename(columns={'combined': 'input'})
        positive_df["labels"] = 1
        negative_df["labels"] = 0
        return pd.concat([positive_df, negative_df], axis=0)
    else:
        # if not binarizing, we need to apply the chat template to the input. It becomes a standard instruction tuning task.
        if train_on_negative:
            all_df = pd.concat([positive_df, negative_df], axis=0)
        else:
            all_df = positive_df
        if is_chat_model:
            if model_name == "meta-llama/Llama-3.1-8B-Instruct":
                def apply_chat_template(row):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."}, 
                        {"role": "user", "content": row["input"]},
                    ]
                    nobos = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[1:]
                    return tokenizer.decode(nobos)
                all_df['input'] = all_df.apply(apply_chat_template, axis=1)

                # handling output separately.
                def apply_chat_template_for_output(row):
                    if len(tokenizer.tokenize(row["output"])) < output_length:
                        return row["output"] + suffix_str
                    else:
                        return row["output"]
                all_df['output'] = all_df.apply(apply_chat_template_for_output, axis=1)
            else:
                def apply_chat_template(row):
                    messages = [{"role": "user", "content": row["input"]}]
                    nobos = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[1:]
                    return tokenizer.decode(nobos)
                all_df['input'] = all_df.apply(apply_chat_template, axis=1)
        return all_df # do nothing, the task will be standard instruction tuning.