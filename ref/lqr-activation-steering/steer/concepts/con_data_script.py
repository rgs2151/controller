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
import csv
import argparse
from pathlib import Path

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def load_model(model_name, quant=False):

    if quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,          # or load_in_8bit=True
            # load_in_8bit=True,
            bnb_4bit_compute_dtype=th.float32,
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
        
def get_target_and_other_sentences(csv_path, target):
    sentences = []  # Target concept sentences
    others = {}     # Other concepts: {concept_name: [sentences]}

    # Read CSV and separate sentences
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            concept_name = row["concept"]
            sentence = row["sentence"]

            if concept_name == f"{target}.NOUN":
                sentences.append(sentence)
            else:
                # Initialize list if not exists
                if concept_name not in others:
                    others[concept_name] = []
                others[concept_name].append(sentence)

    # Find the minimum length among other concepts
    min_len = min(len(sents) for sents in others.values())

    # Build the alternating list
    other_sentences = []
    for i in range(min_len):
        for concept in others:
            other_sentences.append(others[concept][i])

    return sentences, other_sentences

def main():
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
        choices=["gemma2b", "qwen3b", "llama8b", "qwen14b", "llama1b", "qwen32b",],
        default="gemma2b",
    )

    args = parser.parse_args()

    if args.model in models:
        model_name = models[args.model]
        print(f"Running model: {model_name}")
    else:
        raise ValueError("vro...")

    model_keys = {  
        "google/gemma-2-2b": "gemma-2-2b",
        "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
        "Qwen/Qwen2.5-14B": "Qwen2.5-14B",
        "meta-llama/Llama-3.2-1B": "Llama-3.2-1B"
    }

    concepts = [
        "football",
        # "circus",
        "church",
        "dog",
        # "balloon"
    ]

    model, tokenizer = load_model(model_name, quant=True)

    data_handler = ContrastiveBuilder(model, tokenizer)

    key = model_keys[model_name]


    for concept in concepts:
        print(f"running {concept}")
        sen, other = get_target_and_other_sentences(Path(__file__).resolve().parent / 'filtered_sentences.csv', concept)

        for i in range(10):
            print(sen[i])
            print(f"other: {other[i]}\n")

        
        
        filename = key + '-' + concept
        data_handler.collect_data_batch(sen, 200, filename)
        print("done with ", filename)

        filename = key + '-non' + concept
        data_handler.collect_data_batch(other, 200, filename)
        print("done with ", filename)

        filename = key + '-' + concept + '_jac'
        data_handler.collect_jacobians(sen, 25, filename, max_ctx=24)
        print("done with jac")

if __name__ == "__main__":
    print(f"device: {device}")
    main()

