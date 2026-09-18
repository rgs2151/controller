from steer.toxicity.test_toxicityPID import PATH, load_file, run_trials
from steer.benchmarks.ppl_from_file import get_ppl_from_file
from steer.benchmarks.testMMLU import test_mmlu_PID
import steer.toxicity.tox_data_script as utils
import json
import os.path


def generation(models, params):
    toxic_prompts = utils.get_tox_prompts(0.0, 1)
    for model_name in models:
        print(f"Running Generation: {model_name}")
        model, tokenizer = utils.load_model(model_name, quant=True)
        l_list = params[model_name]['l_list']
        pid_p = params[model_name]['PID_params']
        kp = pid_p[0]
        ki = pid_p[1]
        kd = pid_p[2]
        print(f"kp:{kp}, ki:{ki}, kd:{kd}")

        f_pfix = params[model_name]['filename_pfix']
        tox = load_file(f_pfix + "-tox")
        nontox = load_file(f_pfix + "-nontox")
        # jac = load_file(f_pfix + "-nontox_jac")

        if tox is None or nontox is None:
            print(f"Skipping model -- contrastive vectors or jacobians not found")
            continue
        
        X = nontox["X"]
        X_tox = tox["X"]
        X_contr = X - X_tox
        sweeps = []
        output_filename = "PID" + f_pfix + "_tox_eval"
        for i in range(1):
            num_trials = 10
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
                filename=output_filename
            )
            sweeps.extend(s)

        # data = {"prompts": toxic_prompts, "sweeps": sweeps}
        with open((PATH / output_filename).with_suffix(".txt"), 'w', encoding='utf-8') as json_file:
            json.dump(sweeps, json_file, indent=4)

        print(f"Finish generation: {model_name}, output to {f_pfix}_tox_eval.txt")
        print("___________________________________________")

def ppl(models, params):
    for model_name in models:
        print(f"Running PPL: {model_name}")
        f_pfix = params[model_name]['filename_pfix']
        w = get_ppl_from_file("PID" + f_pfix + "_tox_eval",path=PATH)
        if w:
            print(f"Finish PPL: {model_name}")
        else:
            print(f"File not found for PPL: {f_pfix}_tox_eval")
    print("___________________________________________")
    
    
def mmlu(models, params):
    mmlu_filename = "MMLU_trials_PID"
    try:
        with open((PATH / mmlu_filename).with_suffix(".txt"), "r") as f:
            mmlu_data = json.load(f)
    except FileNotFoundError:
        mmlu_data = {}

    for model_name in models:
        print(f"Running MMLU: {model_name}")
        model, tokenizer = utils.load_model(model_name, quant=True)
        l_list = params[model_name]['l_list']
        pid_p = params[model_name]['PID_params']
        kp = pid_p[0]
        ki = pid_p[1]
        kd = pid_p[2]
        instruct = params[model_name]["instruct"]
        print(f"kp:{kp}, ki:{ki}, kd:{kd}")

        f_pfix = params[model_name]['filename_pfix']
        tox = load_file(f_pfix + "-tox")
        nontox = load_file(f_pfix + "-nontox")

        if tox is None or nontox is None:
            print(f"Skipping model MMLU -- contrastive vectors or jacobians not found")
            continue

        X = nontox["X"]
        X_tox = tox["X"]
        X_contr = X - X_tox
        sweeps = []
        for i in range(5):
            out = test_mmlu_PID(model, tokenizer, X_contr, kp, ki, kd, lambda_list=l_list, N_PROMPTS=10, N_LOOP=1, BATCH_SIZE=4, N_SHOTS = 5, INSTRUCT=instruct)
            sweeps.append(out)
        mmlu_data[model_name] = sweeps
        print(f"Finished MMLU model {model_name}")
    with open((PATH / mmlu_filename).with_suffix(".txt"), 'w') as file:
        json.dump(mmlu_data, file, indent=4)


def main():
    models = [
        'google/gemma-2-2b',
        # 'google/gemma-2-9b',
        # "meta-llama/Meta-Llama-3-8B",
        # "Qwen/Qwen2.5-14B",
        # "Qwen/Qwen2.5-3B",
        # "meta-llama/Llama-3.2-1B"
        # "Qwen/Qwen2.5-32B"
    ]

    params = {
        'google/gemma-2-2b': {'filename_pfix': 'gemma-2-2b', 
                              'l_list': [0.5, 1], 
                              'PID_params': [0.7,0.01,0.1],
                              'instruct': False},
        "meta-llama/Meta-Llama-3-8B": {'filename_pfix': 'Llama-3-8B', 
                              'l_list': [0.5, 1, 1.5], 
                              'PID_params': [0.1,0.1,0.0],
                              'instruct': False},
        'google/gemma-2-9b': {'filename_pfix': 'gemma-2-9b', 
                              'l_list': [0.5, 1], 
                              'PID_params': [0.7,0.05,0.0],
                              'instruct': False},
        "Qwen/Qwen2.5-14B": {'filename_pfix': 'Qwen2.5-14B', 
                              'l_list': [0.5, 1], 
                              'PID_params': [0.5,0.01,0.01],
                              'instruct': False},
        "Qwen/Qwen2.5-3B": {'filename_pfix': 'Qwen2.5-3B', 
                              'l_list': [0.5, 1], 
                              'PID_params': [0.3,0.1,0],
                              'instruct': False},
        "meta-llama/Llama-3.2-1B": {'filename_pfix': 'Llama-3.2-1B', 
                              'l_list': [1], 
                              'PID_params': [0.5,0.5,0.1],
                              'instruct': False},
        "Qwen/Qwen2.5-32B": {'filename_pfix': 'Qwen2.5-32B', 
                              'l_list': [1, 1.5], 
                              'PID_params': [0.7,0.1,0],
                              'instruct': False},
    }

    # Generate outputs, measure toxicity, and measure Dist 1,2,3
    # generation(models, params)
    print("===========================")
    print("Done with all generations")
    print("===========================")

    # Measure PPL of the generations
    # ppl(models, params)
    print("===========================")
    print("finish all PPL")
    print("===========================")

    # Get MMLU performance
    mmlu(models, params)
    print("===========================")
    print("finish all MMLU")
    print("===========================")


if __name__ == "__main__":
    main()

    