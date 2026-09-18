import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
import pickle
from steer.PIDsteering import PIDSteering
from datasets import load_dataset
import random
import time
from evaluate import load
import steer.toxicity.tox_data_script as utils
import yaml
import json
from pathlib import Path

from steer.config import config
PATH = Path(__file__).resolve().parent / config["environment"]["tox_data_path"]
PICKLE_JAR = Path(__file__).resolve().parent / config["environment"]["pickle_jar"]

# ppl = load("perplexity", module_type="metric")


device = th.device("cuda" if th.cuda.is_available() else "cpu")

classifier = pipeline(
        "text-classification",
        's-nlp/roberta_toxicity_classifier',
        # device=th.device(device),
        device=th.device("cpu"),
    )

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
    with open((PICKLE_JAR / filename).with_suffix(".pkl"), "rb") as f:
        loaded_tensors = pickle.load(f)
    return loaded_tensors

def run_trials(model, tokenizer, toxic_prompts, num_trials, X_contr, kp, ki, kd, l_list=[1], batch_size = 25, filename="json_out"):
    samples = random.sample(toxic_prompts, num_trials)
    
    do_sample = True
    temp = 1
    k=100
    # headers: 
    print("lambda,kp,ki,kd,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered, ppl_steered, ppl_base")

    start_time = time.perf_counter()

    
    output_str = []
    data_list = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        un_completions = []
        start_time = time.perf_counter()
        # k=50
        inputs = tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

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

        output = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)
        output_str.extend(output)
    postbase_time = time.perf_counter()
    print(f"len of output_str: {len(output_str)}")
    print(f"len of samples: {len(samples)}")

    del output_un
    output_un = None



    for i, inp in enumerate(samples):
        un_completions.append(output_str[i][len(inp):].strip())
    un_preds = classifier(un_completions)
    num_tox_un = 0
    for i in range(len(un_preds)):
        if un_preds[i]["label"] == "toxic":
            num_tox_un += 1
        
    dist_1_base = calculate_dist_n(un_completions, n=1)
    dist_2_base = calculate_dist_n(un_completions, n=2)
    dist_3_base = calculate_dist_n(un_completions, n=3)
    
    data = {"unsteered output": output_str, 
            "num tox": num_tox_un,
            "dist 1": dist_1_base,
            "dist 2": dist_2_base,
            "dist 3": dist_3_base,
            }
    data_list.append(data)

    sweep_data_list = []

    steer_contr = PIDSteering(model, tokenizer, kp=kp,ki=ki, kd=kd, contrastive_vecs=X_contr)
    post_init_time = time.perf_counter()
    
    for l in l_list:
        contr_completions = []


        contr_out = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            contr = steer_contr.track_setpoint(batch, k, lmbda=l, do_sample=do_sample)
            contr_out.extend(contr)
            
        for i, inp in enumerate(samples):
            contr_completions.append(contr_out[i][len(inp):].strip())
        # print(f"PROMPT: {samples}")
        # print(f"\nSteered output: {contr_out}")
        # print(f"\nUnsteered output: {output_str}")
        end_time = time.perf_counter()
        contr_preds = classifier(contr_completions)
        
        dist_1_steered = calculate_dist_n(contr_completions, n=1)
        dist_2_steered = calculate_dist_n(contr_completions, n=2)
        dist_3_steered = calculate_dist_n(contr_completions, n=3)

        # dist_1_base = calculate_dist_n(un_completions, n=1)
        # dist_2_base = calculate_dist_n(un_completions, n=2)
        # dist_3_base = calculate_dist_n(un_completions, n=3)
        n = 0
        m = 0

        num_tox_contr = 0
        for i in range(len(contr_preds)):
            if contr_preds[i]["label"] == "neutral" and un_preds[i]["label"] == "toxic":
                n=n+1
            if contr_preds[i]["label"] == "toxic" and un_preds[i]["label"] == "neutral":
                m=m+1

            if contr_preds[i]["label"] == "toxic":
                num_tox_contr += 1

        sweep_data = {
            "lambda": l,
            "Kp": kp,
            "Ki": ki, 
            "Kd": kd,
            "prompts": samples,
            "unsteered output": output_str, 
            "steered output": contr_out,
            "num safeified": n,
            "num unsafeified": m,
            "num tox unsteered": num_tox_un,
            "num tox steered": num_tox_contr,
            "dist 1 base": dist_1_base,
            "dist 2 base": dist_2_base,
            "dist 3 base": dist_3_base,
            "dist 1 steered": dist_1_steered,
            "dist 2 steered": dist_2_steered,
            "dist 3 steered": dist_3_steered,
        }

        sweep_data_list.append(sweep_data)


        # steered_results = ppl.compute(predictions=contr_out, model_id='gpt2-xl')
        # unsteered_results = ppl.compute(predictions=output_str, model_id='gpt2-xl')
        
        # ppl_steered = steered_results['mean_perplexity']
        ppl_steered = 0
        ppl_unsteered = 0
        # ppl_unsteered = unsteered_results['mean_perplexity']

        print(l,kp,ki,kd,n,m,num_tox_un,num_tox_contr,dist_1_base,dist_2_base,dist_3_base,dist_1_steered,dist_2_steered,dist_3_steered,ppl_steered,ppl_unsteered, sep=",")
    return sweep_data_list



def run_trials_actadd(model, tokenizer, toxic_prompts, num_trials, X_contr, kp, ki, kd, l_list=[1], layer_inds=[5],  batch_size = 25, filename="json_out"):
    samples = random.sample(toxic_prompts, num_trials)
    
    do_sample = True
    temp = 1
    k=100
    # headers: 
    print("lambda,kp,ki,kd,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered, ppl_steered, ppl_base")

    start_time = time.perf_counter()

    
    output_str = []
    data_list = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        un_completions = []
        start_time = time.perf_counter()
        # k=50
        inputs = tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

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

        output = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)
        output_str.extend(output)
    postbase_time = time.perf_counter()
    print(f"len of output_str: {len(output_str)}")
    print(f"len of samples: {len(samples)}")

    del output_un
    output_un = None



    for i, inp in enumerate(samples):
        un_completions.append(output_str[i][len(inp):].strip())
    un_preds = classifier(un_completions)
    num_tox_un = 0
    for i in range(len(un_preds)):
        if un_preds[i]["label"] == "toxic":
            num_tox_un += 1
        
    dist_1_base = calculate_dist_n(un_completions, n=1)
    dist_2_base = calculate_dist_n(un_completions, n=2)
    dist_3_base = calculate_dist_n(un_completions, n=3)
    
    data = {"unsteered output": output_str, 
            "num tox": num_tox_un,
            "dist 1": dist_1_base,
            "dist 2": dist_2_base,
            "dist 3": dist_3_base,
            }
    data_list.append(data)

    sweep_data_list = []

    steer_contr = PIDSteering(model, tokenizer, kp=kp,ki=ki, kd=kd, contrastive_vecs=X_contr)
    post_init_time = time.perf_counter()
    
    for l in l_list:
        contr_completions = []


        contr_out = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            contr = steer_contr.track_setpoint_actadd(batch, k, lmbda=l, layer_inds=layer_inds, do_sample=do_sample)
            contr_out.extend(contr)
            
        for i, inp in enumerate(samples):
            contr_completions.append(contr_out[i][len(inp):].strip())
        # print(f"PROMPT: {samples}")
        # print(f"\nSteered output: {contr_out}")
        # print(f"\nUnsteered output: {output_str}")
        end_time = time.perf_counter()
        contr_preds = classifier(contr_completions)
        
        dist_1_steered = calculate_dist_n(contr_completions, n=1)
        dist_2_steered = calculate_dist_n(contr_completions, n=2)
        dist_3_steered = calculate_dist_n(contr_completions, n=3)

        # dist_1_base = calculate_dist_n(un_completions, n=1)
        # dist_2_base = calculate_dist_n(un_completions, n=2)
        # dist_3_base = calculate_dist_n(un_completions, n=3)
        n = 0
        m = 0

        num_tox_contr = 0
        for i in range(len(contr_preds)):
            if contr_preds[i]["label"] == "neutral" and un_preds[i]["label"] == "toxic":
                n=n+1
            if contr_preds[i]["label"] == "toxic" and un_preds[i]["label"] == "neutral":
                m=m+1

            if contr_preds[i]["label"] == "toxic":
                num_tox_contr += 1

        sweep_data = {
            "lambda": l,
            "Kp": kp,
            "Ki": ki, 
            "Kd": kd,
            "prompts": samples,
            "unsteered output": output_str, 
            "steered output": contr_out,
            "num safeified": n,
            "num unsafeified": m,
            "num tox unsteered": num_tox_un,
            "num tox steered": num_tox_contr,
            "dist 1 base": dist_1_base,
            "dist 2 base": dist_2_base,
            "dist 3 base": dist_3_base,
            "dist 1 steered": dist_1_steered,
            "dist 2 steered": dist_2_steered,
            "dist 3 steered": dist_3_steered,
        }

        sweep_data_list.append(sweep_data)


        # steered_results = ppl.compute(predictions=contr_out, model_id='gpt2-xl')
        # unsteered_results = ppl.compute(predictions=output_str, model_id='gpt2-xl')
        
        # ppl_steered = steered_results['mean_perplexity']
        ppl_steered = 0
        ppl_unsteered = 0
        # ppl_unsteered = unsteered_results['mean_perplexity']

        print(l,kp,ki,kd,n,m,num_tox_un,num_tox_contr,dist_1_base,dist_2_base,dist_3_base,dist_1_steered,dist_2_steered,dist_3_steered,ppl_steered,ppl_unsteered, sep=",")
    return sweep_data_list
    # data_list.append({"sweeps": sweep_data_list})

    # file_path = PATH + filename + ".txt"
    # with open(file_path, 'w') as file:
        # json.dump(data_list, file, indent=4)
    

    # end_time = time.perf_counter()
    # print(f"runtime: {end_time - start_time}")

def main():
    model_name = "Qwen/Qwen2.5-32B"
    # model_name = "google/gemma-2-2b"
    model, tokenizer = utils.load_model(model_name, quant=True)
    filename = "qwen32_pid_sweep"

    tox = load_file("Qwen2.5-32B-tox")
    nontox = load_file("Qwen2.5-32B-nontox")

    X = nontox["X"]
    X_tox = tox["X"]
    X_contr = X - X_tox

    print(th.norm(X_contr, dim=0))
    toxic_prompts = utils.get_tox_prompts(0.8, 1)

    l_list = [1, 1.5]
    # l_list = [1]
    # kp = 0.7
    # ki = 0.0
    # kd = 0.0
    kp_list = [0.5, 0.7]
    ki_list = [0.1, 0.01]
    kd_list = [0.0, 0.01]
    sweeps = []
    for i in range(1):
        print(f"running PID_test_toxicity.py: {model_name}")
        for kp in kp_list:
            for ki in ki_list:
                for kd in kd_list:
                    num_trials = 100
                    s = run_trials(
                        model, 
                        tokenizer, 
                        toxic_prompts, 
                        num_trials, 
                        X_contr, 
                        kp, 
                        ki, 
                        kd,
                        l_list, 
                        filename=filename
                    )
                    sweeps.extend(s)
    with open(PATH / (filename + ".txt"), 'w', encoding='utf-8') as json_file:
        json.dump(sweeps, json_file, indent=4)


if __name__ == "__main__":
    main()
