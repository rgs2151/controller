import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
import pickle
from steer.steering import LQRSteering
from datasets import load_dataset
import random
import time
from evaluate import load
import steer.toxicity.tox_data_script as utils
import json
import yaml
from pathlib import Path

from steer.config import config
PATH = Path(__file__).resolve().parent / config["environment"]["tox_data_path"]
PICKLE_JAR = Path(__file__).resolve().parent / config["environment"]["pickle_jar"]




device = th.device("cuda" if th.cuda.is_available() else "cpu")

classifier = pipeline(
        "text-classification",
        's-nlp/roberta_toxicity_classifier',
        device=th.device(device),
    )

def toxicity_score(text):
    # probability of class 1 (toxic)
    got = classifier(text)[0]
    return got["score"], got["label"]

def calculate_dist_n(texts_list, n=1):
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Collect all tokens
    tokens = []
    for text_item in texts_list:
        tokens.extend(text_item.lower().split())

    if len(tokens) < n:
        return 0.0

    # Build n-grams
    ngrams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]

    unique_ngrams = set(ngrams)

    return len(unique_ngrams) / len(ngrams)


def load_file(filename):
    try:
        with open(PICKLE_JAR / (filename + ".pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def run_trials(model, tokenizer, toxic_prompts, num_trials, A, X_contr, l_list=[1], q_list=[0.1], r_list=[10], qf_list=[0.1], BATCH_SZ=10):
    samples = random.sample(toxic_prompts, num_trials)
    
    do_sample = True
    temp = 1

    # headers: 
    print("lambda,q,r,qf,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered")

    start_time = time.perf_counter()

    inputs = tokenizer(
            samples, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    un_completions = []
    k=100
    with th.no_grad():
        output_un = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=k,
                        return_dict_in_generate=True,
                        do_sample=do_sample,
                        top_p=0.3,
                        repetition_penalty=1.2,
                        temperature=temp,
                        use_cache=False,
                        pad_token_id=tokenizer.eos_token_id,
                        # **model_generation_kwargs, #
                    )

    output_str = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)
    postbase_time = time.perf_counter()

    for i, inp in enumerate(samples):
        un_completions.append(output_str[i][len(inp):].strip())
    un_preds = classifier(un_completions)

    dist_1_base = calculate_dist_n(un_completions, n=1)
    dist_2_base = calculate_dist_n(un_completions, n=2)
    dist_3_base = calculate_dist_n(un_completions, n=3)
    
    sweep = []
    for q in q_list:
        for r in r_list:
            for qf in qf_list:
                # if q == 0.1 and r == 0.1:
                    # continue
                steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr)
                post_init_time = time.perf_counter()

                for l in l_list:
                    track_completions = []
                    contr_completions = []
                    counter = 0

                    contr_out = steer_contr.track_setpoint(samples, k, lmbda=l, do_sample=do_sample, temp = temp)
   
                    for i, inp in enumerate(samples):
                        contr_completions.append(contr_out[i][len(inp):].strip())

                    end_time = time.perf_counter()





                # print(f"contr completions: {contr_completions}")


                    contr_preds = classifier(contr_completions)
                    
                    # print(contr_preds)
                    # print(un_preds)
                    dist_1_steered = calculate_dist_n(contr_completions, n=1)
                    dist_2_steered = calculate_dist_n(contr_completions, n=2)
                    dist_3_steered = calculate_dist_n(contr_completions, n=3)

                    dist_1_base = calculate_dist_n(un_completions, n=1)
                    dist_2_base = calculate_dist_n(un_completions, n=2)
                    dist_3_base = calculate_dist_n(un_completions, n=3)
                    n = 0
                    m = 0

                    num_tox_contr = 0
                    num_tox_un = 0
                    for i in range(len(contr_preds)):
                        if contr_preds[i]["label"] == "neutral" and un_preds[i]["label"] == "toxic":
                            n=n+1
                        if contr_preds[i]["label"] == "toxic" and un_preds[i]["label"] == "neutral":
                            m=m+1
                            # print(f"\ntoxified: {samples[i]}")
                            # print(f"steered: {contr_completions[i]}")
                            # print(f"toxicity score: {contr_preds[i]['score']}\n")
                            # print(f"unsteered: {un_completions[i]}\n")

                        if contr_preds[i]["label"] == "toxic":
                            num_tox_contr += 1
                        if un_preds[i]["label"] == "toxic":
                            num_tox_un += 1


                    data = {
                        "lambda": l,
                        "Q": q,
                        "R": r,
                        "Qf": qf,
                        "prompts": samples,
                        "unsteered output": output_str, 
                        "steered output": contr_out,
                        "num safeified": n,
                        "num unsafeified": m,
                        "num tox unsteered": num_tox_un,
                        "num tox steered": num_tox_contr
                        }
                    sweep.append(data)
                    print(l,q,r,qf,n,m,num_tox_un,num_tox_contr,dist_1_base,dist_2_base,dist_3_base,dist_1_steered,dist_2_steered,dist_3_steered, sep=",")

                del steer_contr

    # steer_contr.plot_unorms(f"lqr_unorms")
    end_time = time.perf_counter()
    print(f"runtime: {end_time - start_time}")
    return sweep    


def main():
    model_name = "google/gemma-2-2b"

    filename = "gemma-2-9b-tox-prelim"
    model, tokenizer = utils.load_model(model_name, quant=True)

    print(f"Running test_toxicity.py: {model_name}")
    print(f"model dtype: {model.dtype}")

    tox = load_file("gemma-2-2b-tox")
    nontox = load_file("gemma-2-2b-nontox")
    jac = load_file("gemma-2-2b-nontox_jac")

    X = nontox["X"]
    A = jac["A"]
    X_tox = tox["X"]
    X_contr = X - X_tox

    toxic_prompts = utils.get_tox_prompts(0.0, 1)

    # l_list = [2, 2.5, 3]
    l_list = [3.5]
    q_list = [0.1]
    r_list = [10]
    qf_list = [1]

    sweeps = []

    for i in range(5):
        print(f"running test_toxicity.py: {model_name}")
        num_trials = 1000
        s = run_trials(
            model, 
            tokenizer, 
            toxic_prompts, 
            num_trials, 
            A, 
            X_contr, 
            l_list, 
            q_list, 
            r_list, 
            qf_list
        )
        sweeps.extend(s)
    

# Open the file in write mode and dump the data
    with open(PATH / (filename + ".txt"), 'w', encoding='utf-8') as json_file:
        json.dump(sweeps, json_file, indent=4)


if __name__ == "__main__":
    main()
