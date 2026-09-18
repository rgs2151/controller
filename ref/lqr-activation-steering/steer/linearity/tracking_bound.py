import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time
from steer.steering import LQRSteering
from steer.data_handling import ContrastiveBuilder
import yaml
import random
import json
# from linearization import compute_lin_err as OLcompute_lin_err
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from steer.config import config
PICKLE_JAR = Path(__file__).resolve().parent / config["environment"]["pickle_jar"]

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
        

def get_safe_prompts():
    dataset = load_dataset("tatsu-lab/alpaca")
    return dataset['train'][:]["instruction"]

def load_file(filename):
    try:
        with open(PICKLE_JAR / (filename + ".pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def compute_lin_err(nom_acts, jacs, acts, var, cov, P):
    l1norm_byvar_list = []
    avg_l1norm_byvar_list = []
    l2norm_byvar_list = []
    l2norm_list = []
    linfty_by_var_list = []
    cos_sim_list = []
    err_list = []
    Pnorm_list = []
    for i in range(nom_acts.shape[0]-1):
        delta_xt = nom_acts[i] - acts[i]
        Adelta_xt = jacs[i] @ delta_xt
        delta_xtp1 = nom_acts[i+1] - acts[i+1]

        err = Adelta_xt - delta_xtp1

        l1byvar = th.sum(th.abs(err / var[i]))

        l1norm_byvar_list.append((l1byvar).item())
        avg_l1norm_byvar_list.append((l1byvar / err.shape[-1]).item())
        l2norm_byvar_list.append(th.norm(err / var[i]).item())
        cos_sim_list.append((th.dot(Adelta_xt, delta_xtp1)/(th.norm(Adelta_xt)*th.norm(delta_xtp1))).item()) # cosine similarity
        l2norm_list.append(th.norm(err).item())
        linfty_by_var_list.append(th.max(th.abs(err)).item())
        Pnorm_list.append(th.sqrt(err.T @ P[i+1] @ err).item())
        # err_list.append(err.tolist())

    
    data = {
        "l1_by_var": l1norm_byvar_list,
        "avg_l1_by_var": avg_l1norm_byvar_list,
        "l2_by_var": l2norm_byvar_list,
        "l2": l2norm_list,
        "linfty_by_var_list": linfty_by_var_list,
        "cos_sim_list": cos_sim_list,
        "Pnorm_list": Pnorm_list
        # "err_list": err_list
            }
    return data


def helper(X1, X2):
    print(X1.shape)
    print(X2.shape)
    # for i in range(X1.shape[0]):

def lin_err_CL(nom_acts, jacs, acts, U, var, diffs, cov, P):
    l1norm_byvar_list = []
    avg_l1norm_bydiffs_list = []
    avg_l1norm_byvar_list = []
    l2norm_byvar_list = []
    l2norm_list = []
    linfty_by_var_list = []
    cos_sim_list = []
    # err_list = []
    l1_list = []
    ratio_list = []
    err_list = []
    
    Pnorm_list = []

    for i in range(nom_acts.shape[0]-1):
        delta_xt_cl = acts[i] - nom_acts[i]
        delta_xt_cl = jacs[i] @ delta_xt_cl + U[i]

        true_delta = acts[i+1] - nom_acts[i+1]
        err = delta_xt_cl - true_delta
        
        
        std = th.sqrt(var[i])

        ratio_list.append((th.norm(err) / th.norm(acts[i] - nom_acts[i])).item())
        # ratio_list.append((th.norm(err) / th.norm(true_delta)).item())


        l1 = th.sum(th.abs(err))
        l1_list.append(l1)

        l1byvar = th.sum(th.abs(err / std))
        l1bydiff = th.sum(th.abs(err / diffs[i+1]))

        l1norm_byvar_list.append((l1byvar).item())
        avg_l1norm_byvar_list.append((l1byvar / err.shape[-1]).item())
        avg_l1norm_bydiffs_list.append((l1bydiff / err.shape[-1]).item())

        l2norm_byvar_list.append(th.norm(err / std).item())
        cos_sim_list.append((th.dot(delta_xt_cl, true_delta)/(th.norm(delta_xt_cl)*th.norm(true_delta))).item()) # cosine similarity
        l2norm_list.append(th.norm(err).item())
        err_list.append(err.tolist())


        errbydiff = err / diffs[i+1]
        Pnorm_list.append(th.sqrt(err.T @ P[i+1] @ err).item())

        linfty_by_var_list.append(th.max(th.abs(err)/ std).item())

    data = {
        "l1": l1_list,
        "l1_by_var": l1norm_byvar_list,
        "avg_l1_by_var": avg_l1norm_byvar_list,
        "avg_l1_by_diffs": avg_l1norm_bydiffs_list,
        "ratio": ratio_list,
        "l2_by_var": l2norm_byvar_list,
        "l2": l2norm_list,
        "linfty_by_var_list": linfty_by_var_list,
        "cos_sim_list": cos_sim_list,
        "err_list": err_list,
        "Pnorm_list": Pnorm_list
            }
    return data


def solve_backwards_lyapunov(A):
    N = A.shape[0]
    n = A.shape[1]

    Q = th.eye(n, device=device)
    P_T = th.eye(n, device=device)
    
    P = th.zeros((N+1, n, n),device=device)
    P[N] = P_T

    A_gpu = A.to(device)
    for k in reversed(range(N)):
        P[k] = A_gpu[k].T @ P[k+1] @ A_gpu[k] + Q

    return P.detach().cpu()

def op_norm(A, Pi, Pip1):
    L = th.linalg.cholesky(Pi)

    M = A.T @ Pip1 @ A

    X = th.cholesky_solve(M,L)
    lambda_max = th.linalg.eigvalsh(X).max()

    return th.sqrt(lambda_max).item()


def is_contracting(A, Pi, Pip1, gamma=1.0):
    lhs = A.T @ Pip1 @ A
    rhs = gamma**2 * Pi
    eigs = th.linalg.eigvalsh(lhs-rhs)
    return eigs.max() <= 0

def bound_at_k(k, delta_x0, errs, A_cl, P):
    phi_k_0 = th.eye(A_cl.shape[-1], device=device)

    sigma = 0


    for i in range(k-1):
        phi_k_0 = A_cl[i] @ phi_k_0


    t1 = op_norm(phi_k_0, P[0], P[k]) * th.sqrt(delta_x0.T @ P[0] @ delta_x0)


    for i in range(k):
        phi_k_i = th.eye(A_cl.shape[-1], device=device)
        for j in range(i+1, k):
            phi_k_i = A_cl[j] @ phi_k_i
        
        sigma += op_norm(phi_k_i, P[i+1], P[k])*errs[i]
    

    bound = t1 + sigma

    return bound

def P_norm(x, P):
    return th.sqrt(x.T @ P @ x)

def compute_closed_loop_lin_err_P(
    nom_states,
    states,
    A_list,
    B_list,
    K_list,
    P_list,
    eps=1e-8
):
    r_Pnorm_list = []
    dz_Pnorm_list = []
    L_hat_list = []

    for k in range(nom_states.shape[0] - 1):
        bar_z_k = nom_states[k]
        bar_z_kp1 = nom_states[k + 1]

        z_k = states[k]
        z_kp1 = states[k + 1]

        delta_z_k = z_k - bar_z_k
        delta_z_kp1 = z_kp1 - bar_z_kp1

        dz_norm = P_norm(delta_z_k, P_list[k])

        if dz_norm < eps:
            r_Pnorm_list.append(0.0)
            dz_Pnorm_list.append(0.0)
            L_hat_list.append(0.0)
            continue

        Ahat_k = A_list[k] - B_list[k] @ K_list[k]

        r_k = delta_z_kp1 - Ahat_k @ delta_z_k
        r_norm = P_norm(r_k, P_list[k])

        L_hat = 2.0 * r_norm / (dz_norm ** 2)

        r_Pnorm_list.append(r_norm.item())
        dz_Pnorm_list.append(dz_norm.item())
        L_hat_list.append(L_hat.item())

    return {
        "r_Pnorm": r_Pnorm_list,
        "dz_Pnorm": dz_Pnorm_list,
        "L_hat": L_hat_list,
    }


def get_tox_prompts(lb, ub):
    assert(lb >= 0 and lb <= ub)
    assert(ub <= 1)
    dataset_name = "allenai/real-toxicity-prompts"
    dataset = load_dataset(dataset_name)
    data = dataset["train"]
    prompts = [
                item["text"]
                for item in data["prompt"]
                if item["toxicity"] is not None and item["toxicity"] <= ub and item["toxicity"] >= lb
            ]
    return prompts

def load_mtnt():
    prompts = []
    with open('MTNT/test/test.ja-en.ja', 'r') as file:
        for i, line in enumerate(file):
            prompts.append(line.strip())
    
    return prompts

def load_code():
    ds = load_dataset("openai/openai_humaneval")['test']

    code_prompts = []
    for i in range(100):
        code_prompts.append(ds[i]['prompt'])

    return code_prompts

def load_adversarial():
    ds = load_dataset("allenai/wildjailbreak", 'eval')['train']

    e = []
    for p in ds:
        if p['data_type'] == "adversarial_harmful":
            e.append(p['adversarial'])
    return e


# model_name = "google/gemma-2-2b"
# model_name = "Qwen/Qwen2.5-3B-Instruct"
model_name = "Qwen/Qwen2.5-3B"
# model_name = "meta-llama/Meta-Llama-3-8B"


model, tokenizer = load_model(model_name, quant=True)


ds = load_dataset("llm-aes/writing-prompts")["train"]
# prompts = get_safe_prompts()
rand_prompts = [d['prompt'][-32:] for d in ds]
# random.shuffle(rand_prompts)
# print(rand_prompts[:10])
def get_tox_prompts(lb, ub):
    dataset_name = "allenai/real-toxicity-prompts"
    dataset = load_dataset(dataset_name)
    assert(lb >= 0 and lb <= ub)
    assert(ub <= 1)

    data = dataset["train"]
    prompts = [
                item["text"]
                for item in data["prompt"]
                if item["toxicity"] is not None and item["toxicity"] <= ub and item["toxicity"] >= lb
            ]
    return prompts

nontox = get_tox_prompts(0,0.1)
tox = get_tox_prompts(0.8,1)


noms = []

steer = LQRSteering(model, tokenizer, 10, 10, 10)

p = 1
i = 0
mtnt = load_mtnt()
code = load_code()
adv = load_adversarial()
a = len(nontox)
b = len(tox)
c = len(mtnt)
d = len(code)
e = len(adv)
print(a)
print(b)


o = th.min(th.tensor([a, b, c, d, e])).item()
print(o)
while i < 5:    
    ind = random.randint(0, o-2)
    out = steer.track_tokens(adv[ind], adv[ind+1], k=1)
    X = steer.X[0][:,-1,:].detach().cpu()
    X_cl = steer.X_cl[0][:,-1,:].detach().cpu()
    if th.norm(X[0] - X_cl[0]).item() < 0.01:
        print("yeehaw")
        # i = i
        p = p+7
        continue
    else:
        i = i+1
    print(f"diff: {th.norm(X[0] - X_cl[0]).item()}")

    A = steer.A.detach().cpu()
    K = steer.K.detach().cpu()
    U = steer.U.detach().cpu()
    dat = {
        "X_nom": X,
        "A_nom": A,
        "X_cl": X_cl,
        "K": K,
        "U": U,
    }
    noms.append(dat)
    print(f"Steered out: {out}")


goof = noms[0]
P_cpu = solve_backwards_lyapunov(goof["A_nom"]-goof["K"])
B = th.eye(goof["A_nom"].shape[-1])
B = B.repeat(goof["A_nom"].shape[0], 1, 1)
out = compute_closed_loop_lin_err_P(goof["X_nom"], goof["X_cl"], goof["A_nom"], B, goof["K"], P_cpu)

P = P_cpu
data_handler = ContrastiveBuilder(model, tokenizer)

acts_new = data_handler.collect_activations(prompts=rand_prompts, num_samples=15)

cov_mat = [th.cov(acts_new[:,i,:].T) for i in range(1, acts_new.shape[1])]
variances = [th.diag(th.cov(acts_new[:,i,:].T)) for i in range(1, acts_new.shape[1])]

acts_max = acts_new.max(dim=0).values
acts_min = acts_new.min(dim=0).values
acts_diff = acts_max - acts_min

x_col = acts_diff.unsqueeze(1)

xP = th.bmm(x_col, P)

x_Pnorm = th.sqrt(
    th.bmm(xP, x_col.transpose(1, 2)).squeeze()
)


def get_global_L(noms, P):
    L = None

    mat_shape = noms[0]["A_nom"].shape
    B = th.eye(mat_shape[-1])
    B = B.repeat(mat_shape[0], 1, 1)
    for rollout_idx in range(len(noms)):
        dat = noms[rollout_idx]
        X_nom = dat["X_nom"].to(device)
        X_cl = dat["X_cl"].to(device)

        # === NEW: closed-loop remainder computation ===
        rem_data = compute_closed_loop_lin_err_P(
            nom_states=X_nom.detach().cpu(),
            states=X_cl.detach().cpu(),
            A_list=dat["A_nom"],      # same A used to build CL_mats
            B_list=B,
            K_list=dat["K"],
            P_list=P,
        )
        if L is None:
            L = rem_data["L_hat"]
        else:
            L_curr = rem_data["L_hat"]
            for i, l in enumerate(L):
                if L_curr[i] > l:
                    L[i] = L_curr[i]
    return L


# NEW
data_handler = ContrastiveBuilder(model, tokenizer)

acts_new = data_handler.collect_activations(prompts=rand_prompts, num_samples=10)


act_mean = acts_new.mean(0).to(device)
P = P_cpu.to(device)

global_L = get_global_L(noms, P_cpu)

print(global_L)


x_col = act_mean.unsqueeze(1)

xP = th.bmm(x_col, P)

x_Pnorm = th.sqrt(
    th.bmm(xP, x_col.transpose(1, 2)).squeeze()
)


x_Pnorm = x_Pnorm.to(device)
print(x_Pnorm.shape)
X_nom = dat["X_nom"].to(device)

all_bounds = []
all_errs = []

for rollout_idx in range(len(noms)):
    dat = noms[rollout_idx]
    X_cl = dat["X_cl"].to(device)
    X_nom = dat["X_nom"].to(device)

    # === NEW: closed-loop remainder computation ===
    rem_data = compute_closed_loop_lin_err_P(
        nom_states=X_nom.detach().cpu(),
        states=X_cl.detach().cpu(),
        A_list=dat["A_nom"],      # same A used to build CL_mats
        B_list=B,
        K_list=dat["K"],
        P_list=P_cpu,
    )

    # This is ||r_k||_P
    r_Pnorm = th.tensor(rem_data["r_Pnorm"], device=device)

    # This is empirical L_k = 2||r_k|| / ||δz_k||^2
    # L_hat = th.tensor(rem_data["L_hat"], device=device)
    L_hat = th.tensor(global_L, device=device)

    err0 = X_cl[0] - X_nom[0]
    err0P = th.sqrt(err0.T @ P[0] @ err0).item()

    # bounds = [err0P / Pvar[0]]
    bounds = [(err0P / x_Pnorm[0]).detach().cpu().item()]
    true_errs = [bounds[0]]

    print(err0P)

    CL_mats = (dat["A_nom"] - dat["K"]).to(device)
    for k in range(1, CL_mats.shape[0] + 1):
        print(f"layer {k}")

        # induced P-norm of closed-loop linearization
        print(f"op norm: {op_norm(CL_mats[k-1], P[k-1], P[k])}")

        # === NEW: quadratic remainder contribution ===
        # theorem uses (L_i / 2) * ||δz_i||^2
        quad_errors = 0.5 * L_hat[:k] * (
            th.tensor(rem_data["dz_Pnorm"][:k], device=device) ** 2
        )

        bound = bound_at_k(
            k,
            err0,
            quad_errors,   # <-- THIS is the key change
            CL_mats,
            P
        )

        print(f"bound: {bound}")

        # true error
        err = X_cl[k] - X_nom[k]
        errP = th.sqrt(err.T @ P[k] @ err).item()
        print(f"true error: {errP}")

        # bounds.append(bound.cpu().item() / Pvar[k])
        # true_errs.append(errP / Pvar[k])

        # print(f"normalization factor: {Pvar[k]}")
        bounds.append((bound.cpu().item() / x_Pnorm[k]).cpu().item())
        true_errs.append((errP / x_Pnorm[k]).cpu().item())

        print(f"normalization factor: {x_Pnorm[k]}")
    all_bounds.append(bounds)
    all_errs.append(true_errs)

bounds = np.array(all_bounds)
errs = np.array(all_errs)


np.savez(
    "llama1b_lin_data_CANDIDATE.npz",
    all_bounds=bounds,
    all_errs=errs,
)

print(model_name)
test = acts_new[0] - noms[0]["X_nom"]
test_nm_ = th.norm(test, dim=-1)

# test_nm = [test_nm_[i].item() / x_Pnorm[i] for i in range(len(Pvar))]
test_nm = [test_nm_[i].item() / x_Pnorm[i].item() for i in range(len(x_Pnorm))]

# print(test_nm.shape)
print(len(test_nm))


# data = np.load("gemma_lin_data.npz")

# bounds = data["all_bounds"]
# errs = data["all_errs"]
print(errs)

print("bounds shape", bounds.shape)
print("errs shape", errs.shape)

# Time axis
x = np.arange(bounds.shape[1])

# --- Select the bound with the largest initial value ---
# initial value = value at t = 0
idx = np.argmax(bounds[:, 0])
worst_bound = bounds[idx]

# --- Compute min/max envelope of errors ---
err_min = errs.min(axis=0)
err_max = errs.max(axis=0)

plt.figure()

# Error envelope
plt.fill_between(
    x,
    err_min,
    err_max,
    color="red",
    alpha=0.3,
    label="Tracking Error"
)

# Outline the envelope
plt.plot(x, err_min, color="red", linewidth=1, label="lower bound")
plt.plot(x, err_max, color="blue", linewidth=1, label="upper bound")

# Worst-case bound
plt.plot(
    x,
    worst_bound,
    color="black",
    linewidth=2,
    label="Worst-case bound"
)

plt.xlabel("Model Layer")
plt.ylabel("Normalized Error (Lyapunov Norm)")
plt.title("Worst-Case Bound vs Error Envelope")

plt.savefig("qwen3b-err-bound_range_adv.pdf")
plt.show()