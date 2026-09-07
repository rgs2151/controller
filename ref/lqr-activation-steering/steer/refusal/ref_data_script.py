import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from steer.data_handling import ContrastiveBuilder
import pandas as pd
import requests
import io
import argparse

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"device: {device}")


def load_model(model_name, quant=False):

    if quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,          # or load_in_8bit=True
            # load_in_8bit=True,
            bnb_4bit_compute_dtype=th.float16,
            bnb_4bit_quant_type="nf4",  # best for LLMs
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, dtype=th.float32, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
        
def get_refused_prompts():
    dataset_name = "walledai/AdvBench"
    dataset = load_dataset(dataset_name)

    return dataset['train'][:]["prompt"]


def get_safe_prompts():
    dataset = load_dataset("tatsu-lab/alpaca")
    return dataset['train'][:]["instruction"]


def get_compliant_prompts(tokenizer):
    dataset = load_dataset("tatsu-lab/alpaca")
    safe_prompts = dataset['train'][:]

    formatted_safe_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in safe_prompts["instruction"]]

    comp_prompts = []
    for i, resp in enumerate(safe_prompts["output"]):
        comp_prompts.append(formatted_safe_prompts[i] + resp)

    return comp_prompts

def create_prompts_with_completions(model, tokenizer, prompts, num_prompts=416, batch_sz=10, filename="output"):
    formatted_harmful_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in prompts]
    print(f"prompt: {formatted_harmful_prompts[0]}")

    outputs = []
    for i in range(0, num_prompts, batch_sz):
        batch = formatted_harmful_prompts[i:i+batch_sz]
        inputs = tokenizer(
                formatted_harmful_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id,
                    # **model_generation_kwargs, #
                )
        output_str = tokenizer.batch_decode(output.sequences, skip_special_tokens=False)

        outputs.extend(output_str)
    print(outputs[:3])

    completions = []
    for i, s in enumerate(formatted_harmful_prompts):
        completions.append(outputs[i][len(s):].strip())

    print(completions[:3])

    df = pd.DataFrame({
        'prompts': prompts,
        'completions': completions
    })
    df.to_csv(f'ref_data/{filename}.csv', index=False)
    return outputs


def retrieve_harmful_instructions(tokenizer):
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()
    targets = dataset["target"].tolist()

    formatted_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in instructions]

    comp_prompts = []
    for i, resp in enumerate(targets):
        comp_prompts.append(formatted_prompts[i] + resp)

    return comp_prompts

from sklearn.model_selection import train_test_split
def get_harmful_instructions_test_():
    """Load harmful instructions from AdvBench dataset."""
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

def get_harmless_instructions():
    """Load harmless instructions from Alpaca dataset."""
    dataset = load_dataset("tatsu-lab/alpaca")
    instructions = [
        item["instruction"] for item in dataset["train"] if item["input"].strip() == ""
    ]
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train[:512], test[:128]

def main():

    models = {
            "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
            "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
            "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
            "gemma9b": "google/gemma-2-9b-it",
            "gemma2b": "google/gemma-2-2b-it",
            "llama3b": "meta-llama/Llama-3.2-3B-Instruct"
        }

    model_keys = {  
            "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
            "google/gemma-2-9b-it": "gemma-2-9b-it",
            "google/gemma-2-2b-it": "gemma-2-2b-it",
            "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct"
        }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["qwen3b", "llama8b", "gemma2b", "gemma9b", "llama3b", "qwen14b"],
        default="llama8b",
    )

    args = parser.parse_args()

    if args.model in models:
        model_name = models[args.model]
        print(f"Running model: {model_name}")
    else:
        raise ValueError(f"Specify a model: {models}")


    model, tokenizer = load_model(model_name, quant=True)

    key = model_keys[model_name]

    data_handler = ContrastiveBuilder(model, tokenizer)

    harmful_prompts, _ = get_harmful_instructions_test_()
    safe_prompts, _ = get_harmless_instructions()

    formatted_harmful_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in harmful_prompts]

    formatted_safe_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in safe_prompts]

    filename = key + '-ref'
    data_handler.collect_data_batch(formatted_harmful_prompts, 200, filename)
    print("done with", filename)

    filename = key + '-nonref'
    data_handler.collect_data_batch(formatted_safe_prompts, 200, filename)
    print("done with ", filename)

    filename = key + '-nonref_jac'
    data_handler.collect_jacobians(formatted_safe_prompts, 50, filename, max_ctx=24)
    print("done with jac")

if __name__ == "__main__":
    main()
