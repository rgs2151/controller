import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time
from steer.data_handling import ContrastiveBuilder
import csv
import steer.concepts.con_data_script as utils
from pathlib import Path
import steer.concepts.test_con as test
from steer.concepts.concept_judge import evaluate
from steer.concepts.concept_score import get_concept_score
from steer.benchmarks.ppl_from_file import get_ppl_from_csv

import yaml

from steer.config import config
PATH = Path(__file__).resolve().parent / config["environment"]["con_data_path"]
PICKLE_JAR = Path(__file__).resolve().parent / config["environment"]["pickle_jar"]

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def collect_data(model, tokenizer, key, concept):
    data_handler = ContrastiveBuilder(model, tokenizer)
    sen, other = utils.get_target_and_other_sentences('concepts/filtered_sentences.csv', concept)
    
    filename = key + '-' + concept

    path = Path(PICKLE_JAR + filename + '.pkl')
    if path.exists():
        print(f"{filename} already exists")
    else:
        data_handler.collect_data_batch(sen, 200, filename)
        print("done with ", filename)

    filename = key + '-' + 'non' + concept
    path = Path(PICKLE_JAR + filename + '.pkl')
    if path.exists():
        print(f"{filename} already exists")
    else:
        data_handler.collect_data_batch(other, 200, filename)
        print("done with ", filename)

    filename = key + '-' + concept + '_jac'
    path = Path(PICKLE_JAR + filename + '.pkl')
    if path.exists():
        print(f"{filename} already exists")
    else:
        data_handler.collect_jacobians(sen, 50, filename, max_ctx=32)
        print("done with jac")
    
def load_files(key, concept):
    filename = key + '-' + concept
    con = test.load_file(filename)
    
    filename = key + '-non' + concept
    noncon = test.load_file(filename)

    filename = key + '-' + concept + '_jac'
    jac = test.load_file(filename)

    return con["X"]-noncon["X"], jac["A"]


def main():
    models = [
        "google/gemma-2-2b",
        # "meta-llama/Meta-Llama-3-8B",
        # "Qwen/Qwen2.5-3B",
    ]


    model_keys = {  
        "google/gemma-2-2b": "gemma-2-2b",
        "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B"

    }
    # l_list = [0.5, 1.5, 2.5]
    l_list = [1.5]


    for model_name in models:
        model, tokenizer = utils.load_model(model_name, quant=True)
        print(f"MODEL: {model_name}")
        key = model_keys[model_name]
        
        
        concept = "football"
        alt_concept = "church"


        X_contr, A = load_files(key, concept)
        

        church = test.load_file("gemma-2-2b-"+alt_concept)["X"]
        notchurch = test.load_file("gemma-2-2b-non"+alt_concept)["X"]
        X_contr_alt = church-notchurch
        num_trials = 10
        output_filename = key + '/' + key + concept + "_" + alt_concept + 'out'
        path = PATH / (output_filename + '.csv')
        test.test_mutlisteer(
            model, 
            tokenizer, 
            num_trials,
            A, 
            X_contr=X_contr,
            X_contr_alt=X_contr_alt,
            l_list=l_list, 
            k=100,
            # q_list, 
            # r_list, 
            # qf_list,
            filename=output_filename
        )
    

        evaluate(
            str(path),
            concept
        )

        get_concept_score(
            str(path), 
            key, 
            concept,
            l_list
        )

        evaluate(
            str(path),
            alt_concept
        )

        get_concept_score(
            str(path), 
            key, 
            alt_concept,
            l_list
        )



if __name__ == "__main__":
    print(f"device: {device}")
    main()

