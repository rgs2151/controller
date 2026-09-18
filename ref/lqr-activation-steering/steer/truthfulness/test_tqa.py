import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
import pickle
from steer.steering import LQRSteering
from steer.PIDsteering import PIDSteering
from datasets import load_dataset
import random
import time
import steer.truthfulness.tqa_data_script as utils
import json
import yaml
import argparse
from pathlib import Path
# with open('config/config.yaml', 'r') as f:
    # config_data = yaml.safe_load(f)
# PICKLE_JAR = config_data["environment"]["pickle_jar"]
# PATH = config_data["environment"]["tqa_data_path"]

from steer.config import config
PATH = Path(__file__).resolve().parent / config["environment"]["tqa_data_path"]
PICKLE_JAR = Path(__file__).resolve().parent / config["environment"]["pickle_jar"]

device = th.device("cuda" if th.cuda.is_available() else "cpu")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,          # or load_in_8bit=True
    bnb_4bit_compute_dtype=th.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

info_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B",quantization_config=quant_config, dtype=th.float32, device_map="auto", trust_remote_code=True)
info_tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B", padding_side="left")
info_tokenizer.pad_token = info_tokenizer.eos_token
info_tokenizer.pad_token_id = info_tokenizer.eos_token_id

truth_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B",quantization_config=quant_config, dtype=th.float32, device_map="auto", trust_remote_code=True)
truth_tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B", padding_side="left")
truth_tokenizer.pad_token = truth_tokenizer.eos_token
truth_tokenizer.pad_token_id = truth_tokenizer.eos_token_id

llama_it_format = {"begin": "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\n", 
                   "end": "assistant\n\n"}

qwen_it_format = {"begin": "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nuser\n",
                 "end": "\nassistant\n"}

gemma_it_format = {"begin": "user\n",
                 "end": "\nmodel\n"}

no_it_format = {"begin": "Q: ",
                 "end": "A:"}

def load_file(filename):
    try:
        with open(PICKLE_JAR / (filename + ".pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def info_pipeline(prompts, tokenizer, BATCH_SIZE):
    pred_info_label = []
    for start in range(0, len(prompts), BATCH_SIZE):
        # print(f"START: {start}")
        batch = prompts[start:start+BATCH_SIZE]
        inputs = info_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = info_judge.generate(
            **inputs,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = info_tokenizer.batch_decode(
                        outputs.sequences,
                        skip_special_tokens=True
                    )
        labels = []
        for prompt, decoded_text in zip(batch, decoded):
            labels.append(decoded_text[len(prompt):].strip())
        # labels = [d[len(prompts[i]):].strip() for i, d in enumerate(decoded)]
        pred_info_label = pred_info_label + labels
    
    return pred_info_label

def truth_pipeline(prompts, tokenizer, BATCH_SIZE):
    pred_truth_label = []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start+BATCH_SIZE]
        inputs = truth_tokenizer(
            batch,
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
        ).to(device)
        
        outputs = truth_judge.generate(
            **inputs,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        decoded = truth_tokenizer.batch_decode(
                        outputs.sequences,
                        skip_special_tokens=True
                    )
        labels = []
        for prompt, decoded_text in zip(batch, decoded):
            labels.append(decoded_text[len(prompt):].strip())
        # labels = [d[len(prompts[i]):].strip() for i, d in enumerate(decoded)]
        pred_truth_label = pred_truth_label + labels
        # print(f"in loop: {pred_truth_label}")
    # print(f"at return: {pred_truth_label}")
    return pred_truth_label

def build_class_prompts(it_format, outputs, spec):
    prompts = []
    begin = it_format["begin"]
    end = it_format["end"]
    for s in outputs:
        start = s.index(begin) + len(begin)
        end_ind = s.index(end, start)
        
        prompts.append("Q: " + s[start:end_ind] + "\nA: " + s[end_ind + len(end):] + "\n" + spec + ":")
    return prompts

def get_t_i_scores(tokenizer, outputs, it_format):
    t_prompt = build_class_prompts(it_format, outputs, "True")
    # print(f"T PROMPTS: \n {t_prompt}")
    # print(f"T PROMPTS len: \n {len(t_prompt)}")
    truth_label = truth_pipeline(t_prompt, tokenizer, 10)
    t_classifications = [float(pred.lower() == "yes") for pred in truth_label]
    # print("t classifications: ", t_classifications)
    t_evaluation_score = th.mean(th.tensor(t_classifications)).item()

    h_prompt = build_class_prompts(it_format, outputs, "Helpful")
    help_label = info_pipeline(h_prompt, tokenizer, 10)
    h_classifications = [float(pred.lower() == "yes") for pred in help_label]
    # print("h classifications: ", h_classifications)
    h_evaluation_score = th.mean(th.tensor(h_classifications)).item()
    return t_evaluation_score, h_evaluation_score


def run_base_model(model, tokenizer, prompts, it_format, num_trials, k=50, do_sample=True, filename="json_out", batch_size=100):
    samples = random.sample(prompts, num_trials)
    # do_sample = False
    # print("lambda,q,r,qf,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered, ppl_base, ppl_steered")

    output_str = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
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

        data_list = []
        with th.no_grad():
            output_un = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=k,
                            return_dict_in_generate=True,
                            do_sample=do_sample,
                            top_p=0.3,
                            repetition_penalty=1.2,
                            temperature=1,
                            use_cache=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )

        output = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)
        output_str.extend(output)
        postbase_time = time.perf_counter()

    t, i = get_t_i_scores(tokenizer, output_str, it_format)
    print(output_str[0:10])
    data = {
            "unsteered t": t,
            "unsteered i": i,
            "unsteered output": output_str,
            "prompts": samples
            }
    # data_list.append(data)
    return data
    file_path = PATH + filename + ".txt"
    with open(file_path, 'w') as file:
        json.dump(data_list, file, indent=4)


def run_trials(model, tokenizer, prompts, it_format, num_trials, A, X_contr, l_list=[1], q_list=[0.1], r_list=[10], qf_list=[0.1], k=50, do_sample=True, filename="json_out", batch_size=500):
    samples = random.sample(prompts, num_trials)
    # do_sample = False
    # print("lambda,q,r,qf,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered, ppl_base, ppl_steered")

    output_str = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
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

        data_list = []
        with th.no_grad():
            output_un = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=k,
                            return_dict_in_generate=True,
                            do_sample=do_sample,
                            top_p=0.3,
                            repetition_penalty=1.2,
                            temperature=1,
                            use_cache=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )

        output = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)
        output_str.extend(output)
        postbase_time = time.perf_counter()

    t, i = get_t_i_scores(tokenizer, output_str, it_format)
    print(output_str[0:10])
    data = {
            "unsteered t": t,
            "unsteered i": i,
            "unsteered output": output_str,
            "prompts": samples
            }
    data_list.append(data)
    print(f"BASE MODEL: t: {t}, i: {i}")

    sweep_data_list = []
    # # print(f"UNSTEERED:\n {output_str}\n\n")
    for q in q_list:
        for r in r_list:
            for qf in qf_list:
                steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr, preserve_mem=True)
                temp_data = []
                for l in l_list:
                    contr_completions = []
                    un_completions = []
                    
                    contr_out = []
                    for i in range(0, len(samples), batch_size):
                        batch = samples[i:i+batch_size]
                        contr = steer_contr.track_setpoint(batch, k, lmbda=l, do_sample=do_sample)
                        contr_out.extend(contr)
                    # print(f"Q = {q}, R = {r}, Qf = {qf}")
                    # print(f"STEERED:\n {contr_out}")
                    # for i, inp in enumerate(samples):
                    #     # print("inp:",inp)
                    #     un_completions.append(output_str[i][len(inp):].strip())
                    #     contr_completions.append(contr_out[i][len(inp):].strip())
                        
                        
                    # print(f"unsteered completions: {un_completions}")
                    t, i = get_t_i_scores(tokenizer, contr_out, it_format)


                    # count_steered = sum(any(ss in comp for ss in refusal_ss) for comp in contr_out)
                    # count_steered_non = sum(all(ss not in comp for ss in refusal_ss) for comp in contr_out)
                    
                    sweep_data = {
                        "lambda": l,
                        "Q": q,
                        "R": r, 
                        "Qf": qf,
                        "steered t": t,
                        "steered i": i,
                        "steered output": contr_out,

                    }
                    sweep_data_list.append(sweep_data)
                    temp_data.append(sweep_data)
                    print(sweep_data)
                    # print(f"count steered: {count_steered}")
                    # print(f"count unsteered: {count_unsteered}")
                    # print(f"count steered non: {count_steered_non}")
                    # print(f"count unsteered non: {count_unsteered_non}")
                # print(f"Done with q: {q}, r: {r}, qf: {qf}")
                # del steer_contr
                # file_path = PATH + filename + f"q-{q}r-{r}-qf-{qf}.txt"
                # with open(file_path, 'w') as file:
                #     json.dump(temp_data, file, indent=4)


    data_list.append({"sweeps": sweep_data_list})
    
    end_time = time.perf_counter()
    print(f"runtime: {end_time - start_time}")
    return data_list


def run_trialsPID(model, tokenizer, prompts, it_format, num_trials, X_contr, kp, ki, kd, l_list=[1], k=50, do_sample=True, filename="json_out", batch_size=800):
    samples = random.sample(prompts, num_trials)
    # do_sample = False
    # print("lambda,q,r,qf,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered, ppl_base, ppl_steered")

    output_str = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
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

        data_list = []
        with th.no_grad():
            output_un = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=k,
                            return_dict_in_generate=True,
                            do_sample=do_sample,
                            top_p=0.3,
                            repetition_penalty=1.2,
                            temperature=1,
                            use_cache=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )

        output = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)
        output_str.extend(output)
        postbase_time = time.perf_counter()

    t, i = get_t_i_scores(tokenizer, output_str, it_format)
    print(output_str[0:10])
    data = {
            "unsteered t": t,
            "unsteered i": i,
            "unsteered output": output_str,
            "prompts": samples
            }
    data_list.append(data)
    print(f"BASE MODEL: t: {t}, i: {i}")

    sweep_data_list = []
    # # print(f"UNSTEERED:\n {output_str}\n\n")
    steer_contr = PIDSteering(model, tokenizer, kp=kp,ki=ki, kd=kd, contrastive_vecs=X_contr)
    temp_data = []
    for l in l_list:
        contr_completions = []
        un_completions = []
        
        contr_out = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            contr = steer_contr.track_setpoint(batch, k, lmbda=l, do_sample=do_sample)
            contr_out.extend(contr)
        # print(f"Q = {q}, R = {r}, Qf = {qf}")
        # print(f"STEERED:\n {contr_out}")
        # for i, inp in enumerate(samples):
        #     # print("inp:",inp)
        #     un_completions.append(output_str[i][len(inp):].strip())
        #     contr_completions.append(contr_out[i][len(inp):].strip())
            
            
        # print(f"unsteered completions: {un_completions}")
        t, i = get_t_i_scores(tokenizer, contr_out, it_format)


        # count_steered = sum(any(ss in comp for ss in refusal_ss) for comp in contr_out)
        # count_steered_non = sum(all(ss not in comp for ss in refusal_ss) for comp in contr_out)
        
        sweep_data = {
            "lambda": l,
            "Kp": kp,
            "Ki": ki, 
            "Kd": kd,
            "steered t": t,
            "steered i": i,
            "steered output": contr_out,

        }
        sweep_data_list.append(sweep_data)
        temp_data.append(sweep_data)
        print(sweep_data)
        # print(f"count steered: {count_steered}")
        # print(f"count unsteered: {count_unsteered}")
        # print(f"count steered non: {count_steered_non}")
        # print(f"count unsteered non: {count_unsteered_non}")
    # print(f"Done with q: {q}, r: {r}, qf: {qf}")
    # del steer_contr
    # file_path = PATH + filename + f"q-{q}r-{r}-qf-{qf}.txt"
    # with open(file_path, 'w') as file:
    #     json.dump(temp_data, file, indent=4)


    data_list.append({"sweeps": sweep_data_list})
    # file_path = PATH + filename + ".txt"
    # with open(file_path, 'w') as file:
    #     json.dump(data_list, file, indent=4)
    
    # end_time = time.perf_counter()
    # print(f"runtime: {end_time - start_time}")
    return data_list

def main():


    models = {
        "gemma2b": "google/gemma-2-2b",
        "gemma9b": "google/gemma-2-9b",
        "llama8b": "meta-llama/Meta-Llama-3-8B",
        "qwen3b": "Qwen/Qwen2.5-3B",
        "qwen14b": "Qwen/Qwen2.5-14B",
        "llama1b": "meta-llama/Llama-3.2-1B"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gemma2b", "gemma9b", "qwen3b", "llama8b", "qwen14b", "llama1b"],
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
        "google/gemma-2-9b": "gemma-2-9b",
        "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
        "Qwen/Qwen2.5-14B": "Qwen2.5-14B",
        "meta-llama/Llama-3.2-1B": "Llama-3.2-1B"
    }

    # prompts = utils.get_refused_prompts()
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "google/gemma-2-2B"
    # model_name = "Qwen/Qwen2.5-3B"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "meta-llama/Meta-Llama-3.1-8B"
    # model_name = "Qwen/Qwen2.5-14B-Instruct"

    # output_filename = "Llama-3.1-8B-Instruct-sweep"

    key = model_keys[model_name]
    output_filename = key + "-resweep"
    # it_format = qwen_it_format

    print("Running test_tqa.py:", model_name)

    it_format = no_it_format

    model, tokenizer = utils.load_model(model_name, quant=True)
    # prompts = utils.get_questions(tokenizer)
    # prompts = utils.get_questions_no_it()
    prompts = utils.get_all_questions_no_it()
    
    print(prompts[0])
    print(len(prompts))


    true = load_file(key + "-true")
    false = load_file(key + "-false")
    jac = load_file(key + "-true_jac")

    X = true["X"]
    X_f = false["X"]
    A = jac["A"]
    print(f"X device {X.device}")

    print(f"X shape: {X.shape}")
    print(f"X_ref shape: {X_f.shape}")
    # print(f"A shape: {A.shape}")

    X_contr = X - X_f
    del X
    del X_f
    l_list = [2.5, 3, 3.5]
    # l_list = [3, 3.5, 4]
    # l_list = [1]

    q_list = [0.1, 0.3, 1]
    r_list = [1,3,5]
    qf_list = [0.1, 0.3, 1]
    # kp = 0.5
    # ki = 0.5
    # kd = 0.01
    num_trials = 437
    out = run_trials(
        model, 
        tokenizer, 
        prompts, 
        it_format,
        num_trials, 
        A, 
        X_contr, 
        l_list, 
        q_list, 
        r_list, 
        qf_list,
        filename=output_filename
    )
    file_path = PATH + output_filename + ".txt"
    with open(file_path, 'w') as file:
        json.dump(out, file, indent=4)

    # run_trialsPID(
    #             model, 
    #             tokenizer, 
    #             prompts, 
    #             no_it_format,
    #             num_trials, 
    #             X_contr, 
    #             kp, 
    #             ki, 
    #             kd,
    #             l_list, 
    #             filename=output_filename
    #         )

# def run_trials(model, tokenizer, prompts, it_format, num_trials, A, X_contr, l_list=[1], q_list=[0.1], r_list=[10], qf_list=[0.1], k=50, do_sample=False, filename="json_out"):



if __name__ == "__main__":
    main()

