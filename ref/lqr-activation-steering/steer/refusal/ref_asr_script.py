import steer.refusal.test_ref as tref
import torch as th
import json
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import io
import argparse

def get_harmful_instructions():
    """Load harmful instructions from AdvBench dataset."""
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_best_dir(X_contr):
    print(f"contrrrrrrr: {X_contr[0]}")
    candidates_normalized = [
        v / v.norm() if not (v.norm() == 0) else v for v in X_contr 
    ]
    print(f"normalized: {candidates_normalized[0]}")
    candidates_stack = th.stack(
        candidates_normalized
    )
    print(f"stack: {candidates_stack[0]}")

    # Compute pairwise cosine similarities
    pairwise_cosine = candidates_stack @ candidates_stack.T
    mean_cosine = pairwise_cosine.mean(dim=-1)

    # Find layer with highest mean cosine similarity
    max_idx = mean_cosine.argmax().item()
    # print(f"selected: {})

    # Log layer selection info
    print(f"\n  Max sim layer selection:")
    for i, key in enumerate(X_contr):
        # layer_num = int(key.split("_")[1])
        marker = " ← SELECTED" if i == max_idx else ""
        print(
            f"    Layer {i}: cosine={mean_cosine[i].item():.4f}{marker}"
        )
    all_best = [
        X_contr[max_idx] for x in X_contr
    ]
    # print(all_best)
    best_stack = th.stack(
        all_best
    )
    return best_stack


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

    parser.add_argument(
        "--setpoint",
        choices=["lfs", "ang"],
        default="lfs",
    )

    parser.add_argument(
        "--steering",
        choices=["lqr", "pid"],
        default="lqr",
    )

    parser.add_argument(
        "--asr-only",
        choices=["1", "0"],
        default="0",
    )

    args = parser.parse_args()

    if args.model in models:
        model_name = models[args.model]
        print(f"Running model: {model_name}")
    else:
        raise ValueError("vro...")
    l_list = [1]
    q_list = [0.2]
    r_list = [7]
    qf_list = [0.2]


    kp_list = [0.05, 0.1]
    ki_list = [0.0, 0.025]
    kd_list = [0.0, 0.025]


    m = model_name.split("/")[-1]
    output_filename = f"{m}_{args.setpoint}_{args.steering}steering"

    key = model_keys[model_name]
    

    ref = tref.load_file(key + "-ref")
    nonref = tref.load_file(key + "-nonref")
    jac = tref.load_file(key + "-nonref_jac")


    model, tokenizer = tref.utils.load_model(model_name, quant=True)
    _, harmful_prompts = get_harmful_instructions()
    formatted_harmful_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p + "\n\n"}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in harmful_prompts]
    print(formatted_harmful_prompts[0])

    X = nonref["X"]
    X_ref = ref["X"]
    A = jac["A"]
    print(f"X device {X.device}")

    print(f"X shape: {X.shape}")
    print(f"X_ref shape: {X_ref.shape}")
    print(f"A shape: {A.shape}")


    
    k=512
    

    num_trials = 10
    

    if not args.asr_only == '1':
        if args.setpoint == 'lfs':
            X_contr = X - X_ref
            del X
            del X_ref

            if args.steering == 'lqr':
                tref.run_trials_lfs(
                    model, 
                    tokenizer, 
                    formatted_harmful_prompts, 
                    num_trials, 
                    A, 
                    X_contr, 
                    l_list, 
                    q_list, 
                    r_list, 
                    qf_list,
                    k=k,
                    do_sample=False,
                    all_tokens=True, # A-LQR+, set to false for standard A-LQR
                    filename=output_filename,
                    batch_size=50
                )
            elif args.steering == 'pid':
                print("In PID")
                tref.run_trials_pid(
                    model, 
                    tokenizer, 
                    formatted_harmful_prompts, 
                    num_trials, 
                    A, 
                    X_contr, 
                    l_list, 
                    kp_list, 
                    ki_list, 
                    kd_list,
                    k=k,
                    do_sample=False,
                    filename=output_filename,
                    batch_size=50
                )
            else:
                raise ValueError(f"Steering specification invalid: {args.steering}")


        elif args.setpoint == 'ang':
            X_contr = X_ref - X
            del X
            del X_ref
            # X_contr = get_best_dir(X_contr)
            tref.run_trials_ang(
                model, 
                tokenizer, 
                formatted_harmful_prompts, 
                num_trials, 
                A, 
                X_contr, 
                [180], 
                q_list, 
                r_list, 
                qf_list,
                k=k,
                do_sample=False,
                filename=output_filename,
                batch_size=50
            )
        else:
            raise ValueError("bubby")


    print("__________________________________________\nFinished Initial Sweep\n__________________________________________")

    
    import steer.refusal.asr as asr

    formats = {
        "llama3b": asr.llama_it_format,
        "llama8b": asr.llama_it_format,
        "qwen3b": asr.qwen_it_format,
        "qwen14b": asr.qwen_it_format,
        "gemma9b": asr.gemma_it_format,
        "gemma2b": asr.gemma_it_format,
    }

    it_format = formats[args.model]



    data_file_path = asr.PATH / (output_filename + ".txt")
    with open(data_file_path, 'r') as file:
        data = json.load(file)

    if args.setpoint == 'lfs':
        if args.steering == 'lqr':
            if "unsteered ASR" not in data:
                print("UNSTEERED")
                inps = asr.get_classifier_inputs(it_format, data["unsteered output"])
                u_score = asr.harmbench_judge(inps,batch_size=4)
                data["unsteered ASR"] = u_score
            for d in data["sweeps"]:
                q = d["Q"]
                r = d["R"]
                qf = d["Qf"]
                if "Steered ASR" not in d:
                    print("STEERED")
                    l = d["lambda"]
                    print(f"lambda: {l}, q: {q}, r: {r}, qf: {qf}")
                    inps = asr.get_classifier_inputs(it_format, d["steered output"])
                    s_score = asr.harmbench_judge(inps,batch_size=4)
                    d["Steered ASR"] = s_score
            
        else:
            for d in data["sweeps"]:
                kp = d["Kp"]
                ki = d["Ki"]
                kd = d["Kd"]
                l = d["lambda"]
                if "Steered ASR" not in d:
                    print(f"lambda: {l}, kp: {kp}, ki: {ki}, kd: {kd}")
                    inps = asr.get_classifier_inputs(it_format, d["steered output"])
                    s_score = asr.harmbench_judge(inps,batch_size=4)
                    d["Steered ASR"] = s_score

    else:
        if "unsteered ASR" not in data:
            print("UNSTEERED")
            inps = asr.get_classifier_inputs(it_format, data["unsteered output"])
            u_score = asr.harmbench_judge(inps,batch_size=4)
            data["unsteered ASR"] = u_score

        for d in data["sweeps"]:
            q = d["Q"]
            r = d["R"]
            qf = d["Qf"]
            a_sweep = d["angle sweep"]
            for a in a_sweep:
                if "Steered ASR" not in a:
                    angle = a["angle"]
                    print(f"angle: {angle}, q: {q}, r: {r}, qf: {qf}")
                    inps = asr.get_classifier_inputs(it_format, a["steered output"])
                    s_score = asr.harmbench_judge(inps,batch_size=4)
                    a["Steered ASR"] = s_score



    with open(data_file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    main()