import torch as th
import pickle
from steer.steering import LQRSteering
import time
import steer.concepts.con_data_script as utils
import json
import csv
import yaml
import os
from pathlib import Path

from steer.config import config
PATH = Path(__file__).resolve().parent / config["environment"]["con_data_path"]
PICKLE_JAR = Path(__file__).resolve().parent / config["environment"]["pickle_jar"]

device = th.device("cuda" if th.cuda.is_available() else "cpu")

def load_file(filename):
    try:
        with open(PICKLE_JAR / (filename + ".pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def run_trials(model, tokenizer, num_trials, A, X_contr, l_list=[1], q_list=[0.1], r_list=[1], qf_list=[0.1], k=100, do_sample=True, filename="json_out", batch_size=10):
    print(A.device)
    print(X_contr.device)

    samples = []
    for i in range(num_trials):
        samples.append("Once upon a time")

    print(samples[0:3])

    data_list = []
    for q in q_list:
        for r in r_list:
            for qf in qf_list:
                steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr, preserve_mem=True)
                temp_data = []
                for l in l_list:
                    contr_out = []
                    for i in range(0, len(samples), batch_size):
                        batch = samples[i:i+batch_size]
                        contr = steer_contr.track_setpoint(batch, k, lmbda=l, do_sample=do_sample)
                        contr_out.extend(contr)
                    print(contr_out)

                    for i in range(len(contr_out)):
                        data_list.append({
                            "lambda": l,
                            "steered": contr_out[i],
                        })
    file_path = PATH / (filename + ".csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["lambda", "steered", "unsteered"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_list)

def test_mutlisteer(model, tokenizer, num_trials, A, X_contr, X_contr_alt, l_list=[1], q_list=[0.1], r_list=[1], qf_list=[0.1], k=100, do_sample=True, filename="json_out", batch_size=100):
    samples = []
    for i in range(num_trials):
        samples.append("Once upon a time")

    print(samples[0:3])

    data_list = []
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
                        contr = steer_contr.multisteer(batch, k, alt_contr=X_contr_alt, lmbda=l, alt_lmbda=1.5, do_sample=do_sample)
                        contr_out.extend(contr)
                    # contr = steer_contr.track_setpoint(prompt, k, lmbda=l, do_sample=do_sample)
                    # contr_out.extend(contr)

                    print(contr)

                    for i in range(len(contr_out)):
                        data_list.append({
                            "lambda": l,
                            "steered": contr_out[i],
                        })

    file_path = PATH / (filename + ".csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["lambda", "steered", "unsteered"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_list)


def main():
    # prompts = utils.get_refused_prompts()
    model_name = "google/gemma-2-2b"
    # model_name = "meta-llama/Meta-Llama-3.1-8B"

    print("Running test_con.py:", model_name)

    model, tokenizer = utils.load_model(model_name, quant=True)
    prompt = "Once upon a time"
    


    dog = load_file("gemma-2-2b-football")
    notdog = load_file("gemma-2-2b-nonfootball")

    church = load_file("gemma-2-2b-balloon")
    notchurch = load_file("gemma-2-2b-balloon")

    jac = load_file("gemma-2-2b-football_jac")
    

    X = dog["X"]
    X_f = notdog["X"]
    A = jac["A"]


    X_c = church['X']
    X_nc = notchurch['X']

    print(f"X device {X.device}")

    print(f"X shape: {X.shape}")
    print(f"X_ref shape: {X_f.shape}")
    # print(f"A shape: {A.shape}")

    X_contr = X - X_f
    del X
    del X_f

    l_list = [1.5]

    kp = 0.5
    ki = 0.5
    kd = 0.01

    q_list = [1]
    r_list = [1]
    qf_list = [0.1]

    A_lr = th.zeros_like(A, device=device)
    for i, At in enumerate(A):
        rank = th.linalg.matrix_rank(At)
        print(f"layer {i} true rank: {rank}")
        U, S, Vh = th.linalg.svd(At, full_matrices=False)
        tol = 1.2  # or relative threshold
        rank = (S > tol).sum()
        print(f"reduced rank: {rank}")
        # tol = S.max() * 1e-6
        mask = S > tol
        U_k = U[:, mask]
        S_k = S[mask]
        Vh_k = Vh[mask, :]

        A_lr[i] = (U_k * S_k) @ Vh_k

    # num_trials = 3
    output_filename = "gemma-2-2b-lowrank"
    num_trials = 500
    # num_trials = 15
    run_trials(
        model, 
        tokenizer, 
        num_trials,
        A_lr, 
        X_contr, 
        l_list, 
        k=300,
        q_list=q_list, 
        r_list=r_list, 
        qf_list=qf_list,
        filename=output_filename
    )

    # test_mutlisteer(
    #     model, 
    #     tokenizer, 
    #     num_trials,
    #     A, 
    #     X_contr=X_contr_church,
    #     X_contr_alt=X_contr_dog,
    #     l_list=l_list, 
    #     k=300,
    #     # q_list, 
    #     # r_list, 
    #     # qf_list,
    #     filename=output_filename
    # )


if __name__ == "__main__":
    main()

