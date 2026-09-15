# Truthfulness

## Goal

Steer models away from answers based on common misconceptions.

## Design

- **Base dataset:** English TruthfulQA.
- **Controller setup:** build and select controllers using only the English TruthfulQA data reserved for those stages.
- **In-distribution evaluation:** answer all 817 English TruthfulQA questions in five seeded repetitions.
- **Language transfer:** answer Spanish translations of the same 817 questions in five seeded repetitions, with answers requested in English. Reuse the English controller without refitting or recalibration.
- **Capability check:** answer one fixed random sample of 200 MMLU test questions. Each question includes five solved examples and requires one A/B/C/D answer. Every method receives the same questions.

## Models

- Gemma-2-2B
- Llama-3-8B
- Qwen-2.5-14B

Spanish is out of distribution relative to the English controller setup. This does not mean the models have never seen Spanish during pretraining.

## Methods

- Original
- ITI
- ActAdd
- Mean-AcT
- Linear-AcT
- PID-AcT
- ODESteer
- S-PID
- A-LQR
- H∞

## Measures

- **True:** whether the answer is judged truthful.
- **Informative:** whether the answer provides useful information.
- **Instruction relevance:** whether the answer addresses the question.
- **Fluency:** whether the answer is well written.
- **MMLU accuracy:** percentage of the 200 questions answered with the correct A/B/C/D choice.

## Main comparison

Does a controller built on English TruthfulQA improve truthfulness on English, retain its effect after the input language changes to Spanish, and preserve general MMLU performance?
