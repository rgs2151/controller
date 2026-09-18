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
import argparse

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"device: {device}")


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
            model_name, 
            quantization_config=quant_config, 
            dtype=th.float32, 
            device_map="auto",
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def get_all_questions_no_it():
    gen = load_dataset("truthfulqa/truthful_qa", "generation")
    ds_gen = gen["validation"]
    shuffled = ds_gen.shuffle(seed=None)  

    # print(shuffled[0])
    questions = [shuffled[i]["question"] for i in range(len(shuffled))]
    for i in range(len(questions)):
        questions[i] = "Q: " + questions[i] + " A:"
    return questions


def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

def format_truthfulqa_end_q(question, choice, rand_question): 
    return f"Q: {question} A: {choice} Q: {rand_question}"


def tqa_prompts(): 
    mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    dataset = mc["validation"]
    dataset = dataset.shuffle(seed=None)
    true_prompts = []
    false_prompts = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)): 
            choice = choices[j]
            prompt = format_truthfulqa(question, choice)
            if labels[j] == 1:
                true_prompts.append(prompt)
            else:
                false_prompts.append(prompt)

    return true_prompts, false_prompts


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

    t_prompts, f_prompts = tqa_prompts()

    data_handler = ContrastiveBuilder(model, tokenizer)

    filename = key + '-truetest'
    data_handler.collect_data_batch(t_prompts, 12, filename)
    print("done with", filename)

    filename = key + '-falsetest'
    data_handler.collect_data_batch(f_prompts, 12, filename)
    print("done with ", filename)

    filename = key + '-true_jactest'
    data_handler.collect_jacobians(t_prompts, 1, filename, max_ctx=24)
    print("done with jac")

if __name__ == "__main__":
    main()
