import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
import pickle
from steer.steering import LQRSteering
import os   
os.environ["PYTORCH_SDP_DISABLE_FLASH"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"] = "1"
os.environ["PYTORCH_SDP_DISABLE_HEURISTIC"] = "1"

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# load model from huggingface
# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "google/gemma-2-2b"
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "Qwen/Qwen2.5-3B"

quant_config = BitsAndBytesConfig( # QUANTIZATION
    load_in_4bit=True,          # or load_in_8bit=True
    bnb_4bit_compute_dtype=th.float16,
    bnb_4bit_quant_type="nf4",  # best for LLMs
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quant_config, dtype=th.float32, device_map="auto")# QUANTIZED


tokenizer = AutoTokenizer.from_pretrained(model_name)


print(f"model name: {model_name}")
print(f"model blocks: {len(model.model.layers)}")
print(f"model device: {model.device}")
print(f"latent dim: {model.model.embed_tokens.embedding_dim}")


nom = "What happens if you swallow a watermellon seed? A. you die. B. you get superpowers. C. diarrhea. Your answer: "
inp = "Something that does not have to do with watermelons "
steer = LQRSteering(model, tokenizer, q=100, r=0.1, qf=1000)
k=50
first_track = steer.track_tokens(nom, inp, k=k)
print(f"Steered out: {first_track}")


inputs = tokenizer(inp, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

with th.no_grad():
    output_un = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=k,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

output_str = tokenizer.decode(output_un.sequences[0], skip_special_tokens=True)
print(f"unsteered: {output_str}")