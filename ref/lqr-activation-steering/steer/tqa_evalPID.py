from steer.truthfulness.test_tqa import PATH, load_file, run_trialsPID, no_it_format
from steer.benchmarks.testMMLU import test_mmlu_PID, SUBJECTS
import steer.truthfulness.tqa_data_script as utils
import json
import os.path
from datasets import load_dataset

def generation(models, params):
    prompts = utils.get_all_questions_no_it()
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
        false = load_file(f_pfix + "-false")
        true = load_file(f_pfix + "-true")


        if true is None or false is None:
            print(f"Skipping model -- contrastive vectors or jacobians not found")
            continue
        
        X = true["X"]
        X_f = false["X"]
        X_contr = X - X_f

        output_filename= "SWEEP_PID"+f_pfix + "_tqa_eval"
        sweeps = []
        for i in range(1):
            num_trials = 10
            s = run_trialsPID(
                model, 
                tokenizer, 
                prompts, 
                no_it_format,
                num_trials, 
                X_contr, 
                kp, 
                ki, 
                kd,
                l_list, 
                filename=output_filename,
                batch_size=200
            )
            sweeps.extend(s)
        with open((PATH / output_filename).with_suffix(".txt"), 'w', encoding='utf-8') as json_file:
            json.dump(sweeps, json_file, indent=4)
        
        del model
        model=None
        del tokenizer
        tokenizer=None
        print(f"Finish generation: {model_name}, output to PID{f_pfix}_tqa_eval.txt")
        print("___________________________________________")


    
def mmlu(models, params):
    mmlu_filename = "MMLU_trials_TQA_PID"
    try:
        with open((PATH / mmlu_filename).with_suffix(".txt"), "r") as f:
            mmlu_data = json.load(f)
    except FileNotFoundError:
        mmlu_data = {}

    subject_datasets = {
                sub: load_dataset("cais/mmlu", sub)
                for sub in SUBJECTS
            }

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
        false = load_file(f_pfix + "-false")
        true = load_file(f_pfix + "-true")


        if true is None or false is None:
            print(f"Skipping model -- contrastive vectors or jacobians not found")
            continue

        X = true["X"]
        X_f = false["X"]
        X_contr = X - X_f


        sweeps = []
        for i in range(1):
            out = test_mmlu_PID(model, tokenizer, X_contr, kp, ki, kd, lambda_list=l_list, N_PROMPTS=10, N_LOOP=1, BATCH_SIZE=4, N_SHOTS = 5, INSTRUCT=instruct, dataset=subject_datasets)
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
        # "Qwen/Qwen2.5-3B",
        # "meta-llama/Llama-3.2-1B",
        # "Qwen/Qwen2.5-32B",
    ]

    params = {
        'google/gemma-2-2b': {'filename_pfix': 'gemma-2-2b', 
                            #   'l_list': [1.5, 2, 2.5], 
                              'l_list': [0.5, 1, 1.5], 
                              'PID_params': [0.7,0.01,0.1],
                              'instruct': False},
        "meta-llama/Meta-Llama-3-8B": {'filename_pfix': 'Llama-3-8B', 
                            #   'l_list': [1.5, 2, 2.5, 3], 
                              'l_list': [0.5, 1, 1.5], 
                              'PID_params': [0.1,0.1,0.0],
                              'instruct': False},
        'google/gemma-2-9b': {'filename_pfix': 'gemma-2-9b', 
                              'l_list': [0.5, 1, 1.5], 
                            #   'l_list': [1.5, 2, 2.5, 3], 
                              'PID_params': [0.7,0.05,0.0],
                              'instruct': False},
        "Qwen/Qwen2.5-14B": {'filename_pfix': 'Qwen2.5-14B', 
                            #   'l_list': [1.5, 2, 2.5, 3], 
                              'l_list': [2], 
                              'PID_params': [0.5,0.01,0.01],
                              'instruct': False},
        "Qwen/Qwen2.5-3B": {'filename_pfix': 'Qwen2.5-3B', 
                            #   'l_list': [1.5, 2, 2.5, 3], 
                              'l_list': [0.5, 1, 1.5], 
                              'PID_params': [0.3,0.1,0],
                              'instruct': False},
        "meta-llama/Llama-3.2-1B": {'filename_pfix': 'Llama-3.2-1B', 
                            #   'l_list': [1.5, 2, 2.5, 3], 
                              'l_list': [0.5, 1, 1.5], 
                              'PID_params': [0.5,0.5,0.1],
                              'instruct': False},
        "Qwen/Qwen2.5-32B": {'filename_pfix': 'Qwen2.5-32B', 
                            #   'l_list': [1.5, 2, 2.5, 3], 
                              'l_list': [1.5], 
                              'PID_params': [0.7,0.1,0],
                              'instruct': False},
    }

    # Generate outputs, measure t, i
    # generation(models, params)
    print("Done with all generations")

    # Get MMLU performance
    mmlu(models, params)
    print("finish all MMLU")


if __name__ == "__main__":
    main()

    