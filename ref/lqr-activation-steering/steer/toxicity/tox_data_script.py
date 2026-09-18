import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time
from steer.data_handling import ContrastiveBuilder
import gc
import argparse

device = th.device("cuda" if th.cuda.is_available() else "cpu")
dataset_name = "allenai/real-toxicity-prompts"
dataset = load_dataset(dataset_name)


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
        
def get_tox_prompts(lb, ub):
    assert(lb >= 0 and lb <= ub)
    assert(ub <= 1)

    data = dataset["train"]
    prompts = [
                item["text"]
                for item in data["prompt"]
                if item["toxicity"] is not None and item["toxicity"] <= ub and item["toxicity"] >= lb
            ]
    return prompts

def get_alternate_prompts():
    ds = load_dataset("tcapelle/jigsaw-toxic-comment-classification-challenge")['test']
    prompts = [p['comment_text'] for p in ds]
    return prompts

def main():

    model_keys = {  
        "google/gemma-2-2b": "gemma-2-2b",
        "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
        "Qwen/Qwen2.5-14B": "Qwen2.5-14B",
        "Qwen/Qwen2.5-32B": "Qwen2.5-32B",
        "llama1b": "Llama-3.2-1B"
    }

    models = {
        "gemma2b": "google/gemma-2-2b",
        "llama8b": "meta-llama/Meta-Llama-3-8B",
        "qwen3b": "Qwen/Qwen2.5-3B",
        "qwen14b": "Qwen/Qwen2.5-14B",
        "qwen32b": "Qwen/Qwen2.5-32B",
        "llama1b": "meta-llama/Llama-3.2-1B"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gemma2b", "qwen3b", "llama8b", "qwen14b", "llama1b", "qwen32b"],
        default=None,
    )

    args = parser.parse_args()

    if args.model in models:
        model_name = models[args.model]
        print(f"Running model: {model_name}")
    else:
        raise ValueError(f"Specify a model: {models}")


    model, tokenizer = load_model(model_name, quant=True)

    key = model_keys[model_name]

    toxic_prompts = get_tox_prompts(0.8, 1)
    nontoxic_prompts = get_tox_prompts(0, 0.1)

    data_handler = ContrastiveBuilder(model, tokenizer)

    filename = key + '-tox'
    data_handler.collect_data_batch(toxic_prompts, 200, filename)
    print("done with", filename)

    filename = key + '-nontox'
    data_handler.collect_data_batch(nontoxic_prompts, 200, filename)
    print("done with ", filename)

    filename = key + '-nontox_jac'
    data_handler.collect_jacobians(nontoxic_prompts, 50, filename, max_ctx=24)
    print("done with jac")

if __name__ == "__main__":
    print(f"device: {device}")
    main()
