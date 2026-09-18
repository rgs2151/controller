import torch as th
from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline, BitsAndBytesConfig
import steer.lqr_utils as lqr
from functools import partial
import pickle
from steer.steering import LQRSteering
from steer.PIDsteering import PIDSteering
from datasets import load_dataset
import random
import time
from steer.config import config
from pathlib import Path
from steer.toxicity.tox_data_script import load_model


device = th.device("cuda" if th.cuda.is_available() else "cpu")

def load_file(filename):
    with open("../../scratch/"+filename+".pkl", "rb") as f:
        loaded_tensors = pickle.load(f)
    return loaded_tensors




LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

# List of all MMLU subjects (configs) in cais/mmlu
# SUBJECTS = [
    # 'abstract_algebra']
SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
    'college_mathematics', 'college_medicine', 'college_physics', 'computer_security',
    'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
    'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
    'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
    'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies',
    'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
    'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
    'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

def format_example(example):
    answer_letter = LETTER_MAP[example["answer"]]
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in zip(["A", "B", "C", "D"], example["choices"])
    )
    return f"Question: {example['question']}\n{choices}\nAnswer: {answer_letter}\n\n"


def format_query(example, instruct=False):
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in zip(["A", "B", "C", "D"], example["choices"])
    )
    if not instruct:
        return f"Question: {example['question']}\n{choices}\nAnswer:"
    else:
        return f"Answer the multiple-choice question: {example['question']}\n{choices}\n Answer with only a single letter."


def build_5shot_prompt(dev_set, test_example, n_shots=5, instruct=False):
    exemplars = random.sample(list(dev_set), n_shots)
    prompt = ""
    for ex in exemplars:
        prompt += format_example(ex)
    prompt += format_query(test_example, instruct)
    # Store the correct answer letter for the test example
    correct_answer = LETTER_MAP[test_example["answer"]]
    return prompt, correct_answer


# N_PROMPTS = 10      # prompts per batch
# N_LOOP = 100         # number of batches
# BATCH_SIZE = 4      # how many to send to GPU at once
# N_SHOTS = 5 # 5 default, 0 for instruct
# do_sample = False
# temp = 0.7
# k=1
# lambda_list = [0.5, 1, 1.5, 2, 2.5]
# lambda_list = [0.5]

def test_mmlu(model, tokenizer, X_contr, A, lambda_list=[1], q=0.1, r=1, qf=0.1, INSTRUCT=False, N_PROMPTS = 10, N_LOOP = 100, BATCH_SIZE = 4, N_SHOTS = 5, do_sample = False, temp = 0.7, k=1):
    steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr)
    
    if INSTRUCT:
        N_SHOTS = 0

    data_list = []
    for l in lambda_list:
        results = []
        output_str = []

        results_steer = []
        output_str_steer = []

        # -------------------------------
        # Pre-load all subjects ONCE
        # -------------------------------
        subject_datasets = {
            sub: load_dataset("cais/mmlu", sub)
            for sub in SUBJECTS
        }

        for _ in range(N_LOOP):

            prompts_with_answers = []
            samples = []

            # -------------------------------
            # Build 10 prompts (CPU only)
            # -------------------------------
            for i in range(N_PROMPTS):
                subject = random.choice(SUBJECTS)
                ds = subject_datasets[subject]
                dev, test = ds["dev"], ds["test"]

                if len(dev) < N_SHOTS or len(test) == 0:
                    continue

                test_example = random.choice(test)
                prompt, correct_answer = build_5shot_prompt(dev, test_example, N_SHOTS, INSTRUCT)

                # print(f"prompt: {prompt}")

                prompts_with_answers.append({
                    "subject": subject,
                    "prompt": prompt,
                    "answer": correct_answer
                })
                if INSTRUCT:
                    samples.append(tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True
                        )
                    )
                else:
                    samples.append(prompt)

            # -------------------------------
            # GPU-efficient batching
            # -------------------------------
            batch_outputs = []
            batch_outputs_steer = []

            # print(f"samples: {samples}")
            for start in range(0, len(samples), BATCH_SIZE):
                
                batch = samples[start:start+BATCH_SIZE]

                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)


                output_un = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=k,
                    do_sample=do_sample,
                    temperature=temp,
                    use_cache=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                decoded = tokenizer.batch_decode(
                    output_un.sequences,
                    skip_special_tokens=True
                )

                batch_outputs.extend(decoded)


                contr_out = steer_contr.track_setpoint(batch, k, lmbda=l, do_sample=do_sample, temp = temp)
                batch_outputs_steer.extend(contr_out)


                # Important: free GPU memory of this batch
                del inputs
                del output_un
                th.cuda.empty_cache()

            # Store results
            output_str.extend(batch_outputs)
            output_str_steer.extend(batch_outputs_steer)

            # -------------------------------
            # Compare answers
            # -------------------------------
            # print("base model")
            for item, model_output in zip(prompts_with_answers, batch_outputs):
                model_choice = model_output.strip()[-1].upper()
                correct_choice = item["answer"]
                # print(f"model choice: {model_choice}")
                # print(f"correct choice: {correct_choice}")
                results.append(model_choice == correct_choice)

            # print("")
            # print("steered model")
            for item, model_output in zip(prompts_with_answers, batch_outputs_steer):
                model_choice = model_output.strip()[-1].upper()
                correct_choice = item["answer"]
                # print(f"model choice: {model_choice}")
                # print(f"correct choice: {correct_choice}")
                results_steer.append(model_choice == correct_choice)

        # Final accuracy
        print(f"LAMBDA: {l}")
        accuracy = sum(results) / len(results)
        print(f"Final accuracy: {accuracy*100}%")

        accuracy_steer = sum(results_steer) / len(results_steer)
        print(f"Final accuracy (steered): {accuracy_steer*100}%")

        data = {"lambda": l, "unsteered accuracy": accuracy, "steered accuracy": accuracy_steer}
        data_list.append(data)
    return data_list


def test_mmlu_PID(model, tokenizer, X_contr, kp, ki, kd, lambda_list=[1], INSTRUCT=False, N_PROMPTS = 10, N_LOOP = 100, BATCH_SIZE = 4, N_SHOTS = 5, do_sample = False, temp = 0.7, k=1, dataset=None):
    steer_contr = PIDSteering(model, tokenizer, kp=kp,ki=ki, kd=kd, contrastive_vecs=X_contr)

    if INSTRUCT:
        N_SHOTS = 0

    data_list = []
    for l in lambda_list:
        results = []
        output_str = []

        results_steer = []
        output_str_steer = []

        # -------------------------------
        # Pre-load all subjects ONCE
        # -------------------------------
        

        print("pre load")
        if dataset is None:
            subject_datasets = {
                sub: load_dataset("cais/mmlu", sub)
                for sub in SUBJECTS
            }
        else:
            subject_datasets = dataset
        print("post load")

        for _ in range(N_LOOP):

            prompts_with_answers = []
            samples = []

            # -------------------------------
            # Build 10 prompts (CPU only)
            # -------------------------------
            for i in range(N_PROMPTS):
                subject = random.choice(SUBJECTS)
                ds = subject_datasets[subject]
                dev, test = ds["dev"], ds["test"]

                if len(dev) < N_SHOTS or len(test) == 0:
                    continue

                test_example = random.choice(test)
                prompt, correct_answer = build_5shot_prompt(dev, test_example, N_SHOTS, INSTRUCT)

                # print(f"prompt: {prompt}")

                prompts_with_answers.append({
                    "subject": subject,
                    "prompt": prompt,
                    "answer": correct_answer
                })
                if INSTRUCT:
                    samples.append(tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True
                        )
                    )
                else:
                    samples.append(prompt)

            # -------------------------------
            # GPU-efficient batching
            # -------------------------------
            batch_outputs = []
            batch_outputs_steer = []

            # print(f"samples: {samples}")
            for start in range(0, len(samples), BATCH_SIZE):
                
                batch = samples[start:start+BATCH_SIZE]

                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)


                output_un = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=k,
                    do_sample=do_sample,
                    temperature=temp,
                    use_cache=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                decoded = tokenizer.batch_decode(
                    output_un.sequences,
                    skip_special_tokens=True
                )

                batch_outputs.extend(decoded)


                contr_out = steer_contr.track_setpoint(batch, k, lmbda=l, do_sample=do_sample, temp = temp)
                batch_outputs_steer.extend(contr_out)


                # Important: free GPU memory of this batch
                del inputs
                del output_un
                th.cuda.empty_cache()

            # Store results
            output_str.extend(batch_outputs)
            output_str_steer.extend(batch_outputs_steer)

            # -------------------------------
            # Compare answers
            # -------------------------------
            # print("base model")
            for item, model_output in zip(prompts_with_answers, batch_outputs):
                model_choice = model_output.strip()[-1].upper()
                correct_choice = item["answer"]
                # print(f"model choice: {model_choice}")
                # print(f"correct choice: {correct_choice}")
                results.append(model_choice == correct_choice)

            # print("")
            # print("steered model")
            for item, model_output in zip(prompts_with_answers, batch_outputs_steer):
                model_choice = model_output.strip()[-1].upper()
                correct_choice = item["answer"]
                # print(f"model choice: {model_choice}")
                # print(f"correct choice: {correct_choice}")
                results_steer.append(model_choice == correct_choice)

        # Final accuracy
        print(f"LAMBDA: {l}")
        accuracy = sum(results) / len(results)
        print(f"Final accuracy: {accuracy*100}%")

        accuracy_steer = sum(results_steer) / len(results_steer)
        print(f"Final accuracy (steered): {accuracy_steer*100}%")

        data = {"lambda": l, "unsteered accuracy": accuracy, "steered accuracy": accuracy_steer}
        data_list.append(data)
    return data_list



def main():
    model_name = "google/gemma-2-2b"

    INSTRUCT = True
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "Qwen/Qwen2.5-3B-Instruct"
    model, tokenizer = load_model(model_name, quant=True)
    # ref = load_file("llama-3.1-8B-it-ref")
    # nonref = load_file("llama-3.1-8B-it-nonref")
    # jac = load_file("llama-3.1-8B-it-nonref_jac")

    ref = load_file("gemma-2-2b-true")
    nonref = load_file("gemma-2-2b-false")
    # jac = load_file("Qwen2.5-3B-Instruct-nonref_jac")
    X = nonref["X"]
    X_ref = ref["X"]
    # A = jac["A"]

    X_contr = X - X_ref
    # assert X_contr.shape[-1] == A.shape[-1], "Assert Error: X and A shapes do not align"
    # assert X_contr.shape[-1] == model.model.embed_tokens.embedding_dim, "Assert Error: X shape does not match model embedding dimension"
    # steer_contr = LQRSteering(model, tokenizer, q=0.1,r=1,qf=0.1, A=A, contrastive_vecs=X_contr)
    
    
    del X
    del X_ref
    lambda_list = [0.75]


    test_mmlu_PID(model, tokenizer, X_contr, kp=0.7, ki=0.01, kd=0, lambda_list=[1], INSTRUCT=False, N_PROMPTS = 10, N_LOOP = 100, BATCH_SIZE = 4, N_SHOTS = 5, do_sample = False, temp = 0.7, k=1)
    # test_mmlu(model, tokenizer, X_contr, A, lambda_list=[1], q=0.1, r=1, qf=0.1, INSTRUCT=True)

if __name__ == "__main__":
    main()
