import torch as th
import pickle
from steer.steering import LQRSteering
from steer.PIDsteering import PIDSteering
from datasets import load_dataset
import steer.toxicity.tox_data_script as utils
import yaml
from timeit import default_timer as timer
import argparse

device = th.device("cuda" if th.cuda.is_available() else "cpu")

from steer.config import config
PATH = Path(__file__).resolve().parent / config["environment"]["tox_data_path"] ["environment"]["tox_data_path"]
PICKLE_JAR = config_data["environment"]["pickle_jar"]

def load_file(filename):
    try:
        with open(PICKLE_JAR + filename + ".pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


models = {
        "gemma2b": "google/gemma-2-2b",
        "llama8b": "meta-llama/Meta-Llama-3-8B",
        "llama1b": "meta-llama/Llama-3.2-1B",
        "qwen3b": "Qwen/Qwen2.5-3B",
        "qwen14b": "Qwen/Qwen2.5-14B",
        "gemma9b": "google/gemma-2-9b"
    }

model_keys = {  
        "google/gemma-2-2b": "gemma-2-2b",
        "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
        "Qwen/Qwen2.5-14B": "Qwen2.5-14B",
        "google/gemma-2-9b": "gemma-2-9b"
    }

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["llama1b", "gemma2b", "qwen3b", "llama8b", "gemma9b", "qwen14b"],
    default="gemma2b",
)

parser.add_argument(
    "--method",
    choices=["lqr", "pid"],
    default="lqr",
)

args = parser.parse_args()

if args.model in models:
    model_name = models[args.model]
    print(f"Running model: {model_name}")
else:
    raise ValueError("vro...")


model, tokenizer = utils.load_model(model_name, quant=True)

key = model_keys[model_name]

print(f"Running test_toxicity.py: {model_name}")
print(f"model dtype: {model.dtype}")
tox = load_file(key + "-tox")
nontox = load_file(key + "-nontox")
jac = load_file(key + "-nontox_jac")

X = nontox["X"]
A = jac["A"]
X_tox = tox["X"]
X_contr = X - X_tox

samples = utils.get_tox_prompts(0.9, 1)[:1]

l = 1
if args.method == 'lqr':
    q = 0.1
    r = 1
    qf = 0.1
elif args.method == 'pid':
    kp = 0.5
    ki = 0.5
    kd = 0.1
else:
    raise ValueError('dawg')

do_sample=True
temp=1

num_trials = 100
k=100

total_time = 0

times = []

for _ in range(num_trials):
    start = timer()

    inputs = tokenizer(
            samples, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    un_completions = []
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
    end = timer()
    total_time += (end-start)
    times.append(end-start)

print("========================= BASE MODEL =========================")
print(f"Average runtime over {num_trials} trials: {total_time / num_trials}")
times_th = th.tensor(times)

print(f"th mean: {th.mean(times_th)}, std: {th.std(times_th)}")
print("===============================================================")


total_init_time = 0
total_runtime = 0

runtimes = []
init_times = []

for _ in range(num_trials):
    start = timer()
    if args.method == 'pid':
        steer_contr = PIDSteering(model, tokenizer, kp=kp,ki=ki, kd=kd, contrastive_vecs=X_contr)
    else:
        steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr)
    post_init = timer()
    contr_out = steer_contr.track_setpoint(samples, k, lmbda=l, do_sample=do_sample, temp = temp)
    end = timer()

    total_init_time += (post_init - start)
    total_runtime += (end - post_init)
    del steer_contr
    runtimes.append(end-post_init)
    init_times.append(post_init - start)
    
print("========================= STEERED MODEL =========================")

print(f"Average init time over {num_trials} trials: {total_init_time / num_trials}")
print(f"Average runtime over {num_trials} trials: {total_runtime / num_trials}")

runtimes_th = th.tensor(runtimes)
print(f"th mean: {th.mean(runtimes_th)}, std: {th.std(runtimes_th)}")

init_th = th.tensor(init_times)
print(f"INIT th mean: {th.mean(init_th)}, std: {th.std(init_th)}")
print("===================================================================")

