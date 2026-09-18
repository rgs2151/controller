import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import steer.lqr_utils as lqr
from functools import partial
import pickle
from steer.steering import LQRSteering


device = th.device("cuda" if th.cuda.is_available() else "cpu")


# load model from huggingface
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "google/gemma-2-2b"

model = AutoModelForCausalLM.from_pretrained(
    model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

steer = LQRSteering(model, tokenizer)

prompt = "something: "

num_tokens = 1
X, A, output = steer.complete_rollout(prompt=prompt, k=num_tokens)

print(f"X: {X}")
print(f"A: {A}")
