import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
from datasets import load_dataset
import pickle
from steer.steering import LQRSteering
from steer.data_handling import ContrastiveBuilder
import yaml
import random
import json
import matplotlib.pyplot as plt
import numpy as np
# import lqr.toxicity.tox_data_script as utils
import csv
from pathlib import Path
import argparse

from scipy.linalg import subspace_angles

from pathlib import Path

from steer.config import config
PICKLE_JAR = Path(__file__).resolve().parent / config["environment"]["pickle_jar"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def load_model(model_name, quant=False):

    if quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,          # or load_in_8bit=True
            # load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",  # best for LLMs
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, dtype=torch.float32, device_map="auto")
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

def plot_singular_vals(s_lists, filename):
    # Ensure output directory exists
    print("in plot function")
    out_dir = f"jacobian_alignment/{filename}-singular-values"
    os.makedirs(out_dir, exist_ok=True)

    # Max singular value per layer (for normalization)
    s_m = [max(max(s) for s in s_sub) for s_sub in s_lists]

    for i, s_sub in enumerate(s_lists):
        fig, ax = plt.subplots(figsize=(5, 4))

        for s in s_sub:
            s = np.array(s) / s_m[i]
            ax.semilogy(s)

        ax.grid(True)
        ax.set_title(f"Layer {i}")
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Normalized singular value")

        fig.tight_layout()
        fig.savefig(f"{out_dir}/spectrum-layer-{i}.png")
        plt.close(fig)


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

pth = "./mat_subspace_couple/"
def save_coupling_figure_and_data(
    sim_mats,
    singular_vals,
    n,
    out_dir,
    prefix="coupling",
    cmap="cividis"
):
    out_dir = Path(pth + out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save raw tensors (best for reproducibility) ---
    torch.save(
        {
            "sim_mats": sim_mats,
            "singular_vals": singular_vals,
            "n": n,
        },
        out_dir / f"{prefix}_data_n{n}.pt",
    )


    t = len(sim_mats)

    fig, axes = plt.subplots(1, t, figsize=(3*t, 3))

    if t == 1:
        axes = [axes]

    vmin = 0
    vmax = 1

    mats = []
    for ax, arr in zip(axes, sim_mats):
        m = ax.matshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis("off")
        mats.append(m)

    # Single colorbar for all plots
    cbar = fig.colorbar(mats[0], ax=axes, fraction=0.046, pad=0.04)

    fig.savefig(out_dir / f"{prefix}_n{n}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{prefix}_n{n}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # plot singular values
    s_m = [max(max(s) for s in s_sub) for s_sub in singular_vals]

    for i, s_sub in enumerate(singular_vals):
        fig, ax = plt.subplots(figsize=(5, 4))

        for s in s_sub:
            s = np.array(s) / s_m[i]
            ax.semilogy(s)

        ax.grid(True)
        # ax.set_title(f"Layer {i}")
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Normalized singular value")

        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_n{n}_sings.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{prefix}_n{n}_sings.pdf", bbox_inches="tight")
        plt.close(fig)

def spectral_subspace_similarity(U_W, s_W, U_X, s_X, eps=1e-12):
        """
        Spectrally-informed similarity between top-k subspaces.

        Parameters
        ----------
        U_W : (n, k) ndarray
            Top-k left singular vectors of W
        s_W : (k,) ndarray
            Corresponding singular values of W
        U_X : (n, k) ndarray
            Top-k left singular vectors of X
        s_X : (k,) ndarray
            Corresponding singular values of X
        eps : float
            Numerical stability constant

        Returns
        -------
        sim : float in [0, 1]
            Normalized spectral subspace similarity
        """

        C = U_W.T @ U_X

        A = (s_W[:, None] * C) * s_X[None, :]

        singvals = np.linalg.svd(A, compute_uv=False)
        nuclear = singvals.sum()

        norm_W = np.linalg.norm(s_W)
        norm_X = np.linalg.norm(s_X)

        sim = nuclear / (norm_W * norm_X + eps)
        return sim


LATENT_DIM = 0

def topk_svd(mat, k):
    """
    mat: torch.Tensor [out_dim, in_dim]
    returns: U_k, s_k, Vh_k
    """
    # torch.linalg.svd is often faster than numpy and avoids host copies
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    return U[:, :k], S[:k], Vh[:k, :]


def alignment_plots(model, tokenizer, prompts, key, filename, num_plots=3, num_mats=10, batch_sz=15):

    similarities = []
    data_handler = ContrastiveBuilder(model, tokenizer)


    LATENT_DIM = None
    k_max = None

    s_list_complete = []   # [layer][matrix][k]
    U_list_complete = []   # optional

    all_s = []

    for i in range(0, num_mats, batch_sz):

        nmats = min(batch_sz, num_mats-i)
        acts, As = data_handler.collect_acts_and_jacs(
            prompts=prompts,
            num_samples=nmats,
            filename=filename + "-acts-jacs",
            max_ctx=16
        )

        del acts  # free ASAP

        As = As.detach()  # keep on device for SVD

        if LATENT_DIM is None:
            LATENT_DIM = As.shape[-1]
            k_max = LATENT_DIM // 20
            print(f"k max: {k_max}")

            num_layers = As.shape[1]
            skip = num_layers // num_plots
            

            # initialize per-layer storage
            s_list_complete = [[] for _ in range(0,num_layers, skip)]
            all_s = [[] for _ in range(0,num_layers, skip)]
            U_list_complete = [[] for _ in range(0,num_layers, skip)]
            V_list_complete = [[] for _ in range(0,num_layers, skip)]

        num_matrices = As.shape[0]
        num_layers = len(As[0])  # assuming all matrices have the same number of layers
        for l, layer in enumerate(range(0, num_layers, skip)):
            print(f"Batch {i}, Layer {layer}")

            for m in range(num_matrices):
                J = As[m, layer]  # single Jacobian

                # determine k dynamically (matches your logic)
                k = min(J.shape[-1], k_max)

                U_k, s_k, Vh_k = topk_svd(J, k)
                s_all_k = np.linalg.svd(J, compute_uv=False)

                all_s[l].append(s_all_k)
                s_list_complete[l].append(s_k)
                U_list_complete[l].append(U_k)


    for i in range(0, len(s_list_complete)):
    # for i in range(1):
        print(f"Layer {i}")
        
        # Convert all tensors to numpy
        

        # Full SVDs
        
        # w_list = [s / sum(s) for s in s_list]
        s_list = s_list_complete[i]
        w_list = [s / s.sum() for s in s_list]
        U_list = U_list_complete[i]

        nmats = len(w_list)
        sim_mat = np.zeros((nmats, nmats))

        # print(len(mats))
        print(len(U_list))
        print(len(w_list))
        print(sim_mat.shape)
        for j in range(nmats):
            for r in range(nmats):
                print(j, r)
                print(U_list[j].shape)
                angles = subspace_angles(U_list[j], U_list[r])
                score = spectral_subspace_similarity(U_list[j], s_list[j], U_list[r], s_list[r])
                # score = weighted_cos(angles, w_list[j], w_list[r])

                sim_mat[j,r] = score
        similarities.append(sim_mat)

        # print("\n")

    out_file = key + '/' + 'BIG_' + filename 
    save_coupling_figure_and_data(similarities, all_s, k, out_file)
    del As


def load_adversarial():
    ds = load_dataset("allenai/wildjailbreak", 'eval')['train']

    e = []
    for p in ds:
        if p['data_type'] == "adversarial_harmful":
            e.append(p['adversarial'])
    return e

def alignment_plots_multi(model, tokenizer, base_prompts, other_prompts, key, filename, num_plots=3, num_mats=10, batch_sz=15):

    similarities = []

    data_handler = ContrastiveBuilder(model, tokenizer)


    LATENT_DIM = None
    k_max = None

    s_list_complete = []   # [layer][matrix][k]
    U_list_complete = []   # optional

    all_s = []

    for i in range(0, num_mats, batch_sz):
        if i < num_mats / 2:
            prompts=base_prompts
        else:
            prompts=other_prompts
        nmats = min(batch_sz, num_mats-i)
        acts, As = data_handler.collect_acts_and_jacs(
            prompts=prompts,
            num_samples=nmats,
            filename=filename + "-acts-jacs",
            max_ctx=16
        )

        del acts  # free ASAP

        As = As.detach()  # keep on device for SVD

        if LATENT_DIM is None:
            LATENT_DIM = As.shape[-1]
            k_max = LATENT_DIM // 20
            print(f"k max: {k_max}")

            num_layers = As.shape[1]
            skip = num_layers // num_plots
            

            s_list_complete = [[] for _ in range(0,num_layers, skip)]
            all_s = [[] for _ in range(0,num_layers, skip)]
            U_list_complete = [[] for _ in range(0,num_layers, skip)]
            V_list_complete = [[] for _ in range(0,num_layers, skip)]

        num_matrices = As.shape[0]
        num_layers = len(As[0])  # assuming all matrices have the same number of layers

        for l, layer in enumerate(range(0, num_layers, skip)):
        # for l, layer in enumerate(indices):
            print(f"Batch {i}, Layer {layer}")

            for m in range(num_matrices):
                J = As[m, layer]  # single Jacobian

                k = min(J.shape[-1], k_max)

                U_k, s_k, Vh_k = topk_svd(J, k)
                s_all_k = np.linalg.svd(J, compute_uv=False)

                all_s[l].append(s_all_k)
                s_list_complete[l].append(s_k)
                U_list_complete[l].append(U_k)


    for i in range(0, len(s_list_complete)):
        print(f"Layer {i}")
        
        s_list = s_list_complete[i]
        w_list = [s / s.sum() for s in s_list]
        U_list = U_list_complete[i]

        nmats = len(w_list)
        sim_mat = np.zeros((nmats, nmats))

        # print(len(mats))
        print(len(U_list))
        print(len(w_list))
        print(sim_mat.shape)
        for j in range(nmats):
            for r in range(nmats):
                print(j, r)
                print(U_list[j].shape)
                angles = subspace_angles(U_list[j], U_list[r])
                score = spectral_subspace_similarity(U_list[j], s_list[j], U_list[r], s_list[r])

                sim_mat[j,r] = score
        similarities.append(sim_mat)

        # print("\n")

    out_file = key + '/' + 'BIG_' + filename 
    save_coupling_figure_and_data(similarities, all_s, k, out_file)
    del As

def random_mat(n, k, rng=None):
   """
   Generate an n x k matrix
   """
   if rng is None:
       rng = np.random.default_rng()


   X = np.random.randn(n, k)
   X = X / np.linalg.norm(X, axis=0, keepdims=True)
   return X

def compute_random(d,k,num_mats):

    s_ref = np.ones(k)
    w_energy = np.array([s /  sum(s_ref) for s in s_ref])

    num_trials = num_mats
    S_rand_energy = np.zeros((num_trials, num_trials))
    n = d
    M1 = [random_mat(n, k) for i in range(num_trials)]
    for r in range(num_trials):
        for p in range(num_trials):
            score = spectral_subspace_similarity(M1[r], s_ref, M1[p], s_ref)
            S_rand_energy[r, p] = score


    return S_rand_energy

def from_file(fp):
    ckpt = torch.load(fp, weights_only=False)
    sim_mats = ckpt["sim_mats"]

    for mat in sim_mats:
        print(np.mean(mat))

def load_mtnt():
    prompts = []
    with open('MTNT/test/test.ja-en.ja', 'r') as file:
        for i, line in enumerate(file):
            prompts.append(line.strip())
    
    return prompts

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


def main():

    models = {
        "gemma2b": "google/gemma-2-2b",
        "llama8b": "meta-llama/Meta-Llama-3-8B",
        "qwen3b": "Qwen/Qwen2.5-3B",
        "qwen32b": "Qwen/Qwen2.5-32B",
        "qwen14b": "Qwen/Qwen2.5-14B",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gemma2b", "qwen3b", "llama8b", "qwen32b", "qwen14b"],
        default="gemma2b",
    )

    parser.add_argument(
        "--task",
        choices=["concepts", "law", "code", "multi", "adv"],
        default="concepts",
    )

    parser.add_argument(
        "--bsz",
        type=int,
        default=15
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
        "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
        "Qwen/Qwen2.5-14B": "Qwen2.5-14B",
        "Qwen/Qwen2.5-32B": "Qwen2.5-32B"
    }


    concepts = [
        # "shelter", 
        # "carbon", 
        "cloud",
        # "speculation", 
        # "football", 
        # "balloon", 
        # "foreigner", 
        # "circus", 
        # "friend",
        # "radio", 
        # "basement", 
        # "flower", 
        # "book", 
        # "cloud", 
        # "church", 
        # "assistance", 
        # "spirit", 
        # "obstacle", 
        # "baby", 
        # "dog"
    ]

    print("RUNNING NUCLEAR")
    # for model_name in models:
    model, tokenizer = load_model(model_name, quant=True)
    print("MODEL", model_name)

    batch_sz = args.bsz

    if args.task == 'concepts':
        for concept in concepts:
            print(f"CONCEPT: {concept}")
            key = model_keys[model_name]
            con_prompts, other_prompts = get_target_and_other_sentences("concepts/filtered_sentences.csv", concept)
            filename = concept + "_ordered"
            alignment_plots(model, tokenizer, con_prompts, key, filename, 3, num_mats=50, batch_sz=batch_sz)

            print(f"LATENT: {LATENT_DIM}")

    elif args.task == 'law':
        print(f"LAW:")
        key = model_keys[model_name]
        ds = load_dataset("AdaptLLM/law-tasks", "CaseHOLD")['test']
        law_prompts = []
        for i in range(100):
            law_prompts.append(ds[i]['input_options'][0])
        filename = "law_prompts"

        alignment_plots(model, tokenizer, law_prompts, key, filename, 3, num_mats=50, batch_sz=batch_sz)

        print(f"LATENT: {LATENT_DIM}")

    elif args.task == 'code':
        print(f"CODE:")
        key = model_keys[model_name]
        ds = load_dataset("openai/openai_humaneval")['test']
        code_prompts = []
        for i in range(100):
            code_prompts.append(ds[i]['prompt'])
        filename = "code_prompts"

        alignment_plots(model, tokenizer, code_prompts, key, filename, 3, num_mats=50, batch_sz=batch_sz)

        print(f"LATENT: {LATENT_DIM}")

    elif args.task == 'multi':
        print(f"MULTI:")
        print(f"MTNT+CODE")
        key = model_keys[model_name]
        ds = load_dataset("openai/openai_humaneval")['test']

        mtnt = load_mtnt()


        code_prompts = []
        for i in range(100):
            code_prompts.append(ds[i]['prompt'])
        filename = "MTNT+CODE_prompts"

        alignment_plots_multi(model, tokenizer, code_prompts, mtnt, key, filename, 3, num_mats=50, batch_sz=batch_sz)

        print(f"LATENT: {LATENT_DIM}")

    elif args.task == 'adv':
        print(f"MULTI:")
        print(f"adv+cloud")
        key = model_keys[model_name]
        ds = load_dataset("openai/openai_humaneval")['test']
        con_prompts, other_prompts = get_target_and_other_sentences("concepts/filtered_sentences.csv", "cloud")

        # mtnt = load_mtnt()
        adv = load_adversarial()


        code_prompts = []
        for i in range(100):
            code_prompts.append(ds[i]['prompt'])
        filename = "adv+cloud"

        alignment_plots_multi(model, tokenizer, code_prompts, adv, key, filename, 3, num_mats=50, batch_sz=batch_sz)

        print(f"LATENT: {LATENT_DIM}")




if __name__ == "__main__":
    main()