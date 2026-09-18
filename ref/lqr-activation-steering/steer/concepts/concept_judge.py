# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import functools
import logging
import os
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import json

from steer.truthfulness.tqa_data_script import load_model

from nltk.corpus import wordnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(wordnet.synsets("game")[0].definition())

# Already run in parallel inside DataLoader

wordnet_definitions = {}
system_answers = ["Yes", "No"]
concepts = ["football","circus","balloon","dog","church"]
for c in concepts:
    wordnet_definitions[c] = wordnet.synsets(c)[0].definition()


def build_prompt(concept):
    if concept not in wordnet_definitions:
        raise ValueError(f"An invalid concept was provided: {concept}")
        return None
    definition = wordnet_definitions[concept]
    prompt = (f"You are a chatbot who answers whether the provided sentence "
            f"is referring to {concept} defined as {definition.lower()}. Note that the sentence might not "
            f"contain the word {concept}, but may just be "
            f"referencing concept as defined."
    )
    return prompt

def compute_topk_softmax(
    logits: torch.Tensor, protected_tokens: t.List[int], topk: int = 10
) -> t.List[float]:
    """
    Computes the softmax'd values for `protected_tokens` "against" the topk tokens predicted.
    If `protected_toke    ns` are not in the topk, they are included. So the final set of tokens
    used for softmax can have a size between [topk, topk+len(protected_tokens)].

    :param logits: Logits tensor, of size (num_tokens, vocabulary). Only the first token will be used!
    :param protected_tokens: List of indices to the protected tokens.
    :param topk: Number of top tokens to use for softmax.

    :return:
    """
    device = logits.device
    # Get the indices to the top-l tokens according to logits.
    top_logits_idx = (
        (torch.argsort(logits[0], dim=0, descending=True)[:topk])
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
    # Make a set
    top_logits_idx = set(top_logits_idx)
    # Add protected tokens if needed
    top_logits_idx.update(set(protected_tokens))
    # Back to list
    top_logits_idx = list(top_logits_idx)
    # And back to tensor
    top_logits_idx_torch = torch.tensor(top_logits_idx).to(device)
    # Get the values corresponding to the top-k tokens
    top_logits = logits[0, top_logits_idx_torch]
    # Compute softmax
    top_logits_sm = torch.softmax(top_logits, 0)
    # Take the indices in `top_logits_idx` values corresponding to the protected tokens
    answer_idx_in_top = [top_logits_idx.index(pt) for pt in protected_tokens]
    # Get the softmaxed values corresponding to the protected tokens
    logits_answers_sm = [top_logits_sm[idx].item() for idx in answer_idx_in_top]
    return logits_answers_sm


def build_llm_input(
    mode: str,
    system_question: str,
    sentence: str,
    sentence2: t.Optional[str] = None,
    prompt: t.Optional[str] = None,
    prepend: t.Optional[t.List[str]] = None,
) -> t.List[t.Dict]:
    messages = []
    s1 = sentence if prepend is None else f"{prepend[0]} {sentence}"
    s2 = sentence2 or None
    if prepend is not None and s2 is not None:
        s2 = f"{prepend[1]} {s2}"

    if mode == "sentence":
        messages = [
            {
                "role": "system",
                "content": system_question,
            },
            {"role": "user", "content": s1},
        ]
    elif mode == "2sentences":
        messages = [
            {
                "role": "system",
                "content": system_question,
            },
            {
                "role": "user",
                "content": f"{s1}\n{s2}",
            },
        ]
    elif mode == "sentence_prompt":
        messages = [
            {
                "role": "system",
                "content": system_question,
            },
            {
                "role": "user",
                "content": f"Prompt: {prompt}\nContinuation: {s1}",
            },
        ]
    elif mode == "2sentences_prompt":
        messages = [
            {
                "role": "system",
                "content": system_question,
            },
            {
                "role": "user",
                "content": f"Prompt: {prompt}\n{s1}\n{s2}",
            },
        ]
    return messages


def read_csv(data_path: Path) -> pd.DataFrame:
    # Trying , and ; as delimiters.
    try:
        df = pd.read_csv(data_path)
    except:
        try:
            df = pd.read_csv(data_path, delimiter=";", index_col=0)
        except Exception as exc:
            raise RuntimeError(exc)
    # Hack for user study csvs, remove NaN in the "id" column (there are explanation cells).
    # if id in df.columns:
    #     df = df[~df.id.isna()]
    return df

def get_responses(path):
    with open(path, 'r') as f:
        dat = json.load(f)
    return dat

def evaluate(data_path, concept) -> None:
    # Setup device and distributed learning
    model, tokenizer = load_model(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        quant=True
    )

    terminators = []
    if hasattr(tokenizer, "eos_id"):
        terminators += [
            tokenizer.eos_id,
        ]
    if hasattr(tokenizer, "special_tokens"):
        terminators += [
            tokenizer.special_tokens["<|eot_id|>"],
            tokenizer.special_tokens["<|end_of_text|>"],
        ]
    print(f"Terminators: {terminators}")

    generator = functools.partial(
        model.generate,
        eos_token_id=terminators,
        do_sample=False,
        # temperature=0.6,
        # top_p=0.9,
        # output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        # suppress_tokens=terminators,
    )

    answers_token_ids = [tokenizer.encode(a)[0] for a in system_answers]
    print(f"Answer tokens: {answers_token_ids}")

    mode = "sentence"
    df = read_csv(data_path)
    print(df)

    l = 2

    sentences = df.loc[:,"steered"].tolist()
    sentences2, prompts = None, None

    answers_str = ", ".join(system_answers)

    system_prompt = [build_prompt(concept)]

    system_prompts = [
        (
            sp.strip()
            + f"\nYou can only output one of these answers: ({answers_str}). Do not provide explanations, elaborations, or any other information, just ({answers_str})."
        )
        for sp in system_prompt
    ]



    results_per_system_prompt = []
    for sp, system_question in enumerate(system_prompts):
        # logger.info(f"SYSTEM_QUESTION:\n{system_question}")

        results = []
        # logger.info(f"Running on {len(sentences)} sentences.")
        for i, sentence in enumerate(tqdm.tqdm(sentences, desc="Answering")):
            sentence2 = sentences2[i] if sentences2 is not None else None
            prompt = prompts[i] if prompts is not None else None
            messages = build_llm_input(
                mode=mode,
                system_question=system_question,
                sentence=sentence,
                sentence2=sentence2,
                prompt=prompt,
                prepend=None,
            )
            # print(messages)
            # if i == 0:
            #     logger.info(mode)
            #     logger.info(messages)

            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)

            # Generate!
            with torch.no_grad():
                outputs = generator(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=1,
                )

            response = outputs["sequences"][0, input_ids.shape[-1] :]
            decoded = tokenizer.decode(response.tolist())
            decoded = decoded.replace("<|eot_id|>", "")

            # outputs["logits"] is a tuple of N tokens being generated.
            # Each element is a [1, vocab] shaped array.
            logits_answers = [
                outputs["logits"][0][0, a_id].item() for a_id in answers_token_ids
            ]

            # Also compute softmax of logits among top K tokens
            topk = 10
            logits_answers_sm = compute_topk_softmax(
                logits=outputs["logits"][0],
                protected_tokens=answers_token_ids,
                topk=topk,
            )

            # Collect results
            results.append(
                [
                    decoded,
                ]
                + logits_answers
                + logits_answers_sm
            )
        q_str = f"q{sp}"
        results_df = pd.DataFrame(
            data=results,
            columns=[  # cfg.columns +
                f"{concept}_{q_str}_llm_answer",
            ]
            + [f"{concept}_{q_str}_{sa}" for sa in system_answers]
            + [f"{concept}_{q_str}_{a}" + f"_sm_{topk}" for a in system_answers],
        )

        if len(system_answers) == 2:
            c0 = results_df[f"{concept}_{q_str}_{system_answers[0]}"]
            c1 = results_df[f"{concept}_{q_str}_{system_answers[1]}"]
            diff = np.array(c1 > c0).astype(np.int32)
            results_df[f"{concept}_{q_str}_llm_answer_logit"] = diff

        results_df[f"{concept}_{q_str}_llm_answer_num"] = results_df[f"{concept}_{q_str}_llm_answer"].apply(
            lambda x: system_answers.index(x) if x in system_answers else 1
        )
        results_per_system_prompt.append(results_df)

    # All system prompts answered
    dfs_out = (
        [
            df,
        ]
        # if df2 is None
        # else [df, df2]
    )

    results_final = pd.concat(
        dfs_out + results_per_system_prompt,
        axis=1,
    )

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        output_path = data_path
        if output_path is not None:
            # filename = Path(output_path) / "0_vague_shot_eval.csv"
            # results_final.to_csv(filename)
            results_final.to_csv(output_path)
            print(f"Saved results in {output_path}")
        # if cfg.wandb.mode != "disabled":
            # utils.log_wandb(evaluate_0shot=results_final)


def main() -> None:
    concepts = ["football","circus","balloon","dog","church"]
    for c in concepts:
        wordnet_definitions[c] = wordnet.synsets(c)[0].definition()

    print(wordnet_definitions)

    evaluate("concepts/gemma-2-2b-vague-out.csv", "dog")


if __name__ == "__main__":
    main()