# Network-size residual exploration

This experiment compares residual dynamics across model scale and an explicit adversarial OOD condition using a uniform hidden-state residual metric.

## Goal

Quantify how the mean layerwise residual magnitude changes as model size increases, and whether adversarial prompts amplify residuals relative to within-distribution prompts.

## Models to try

- distilgpt2
- gpt2
- gpt2-medium
- gpt2-large
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- optional: Qwen/Qwen2.5-14B-Instruct as an H100-only extension

## Outputs

- `cache/network_size_residual_summary_adversarial.json`
- `plots/network_size_residual_trend_adversarial.png`
- `plots/network_size_residual_matrix_adversarial.png`
- `plots/network_size_residual_scaling_paper_adversarial.png`

## OOD condition

The default OOD family is `Adversarial`, not an average over multiple OOD families. You can override this with `--ood_family` if needed.

## Prompt counts

- Within-distribution: 50 prompts per run
- OOD family: 50 prompts per run
- Current top-3 OOD comparison: Adversarial, Long context, Spanish

These 50 prompts are generated from smaller handcrafted base sets with controlled prefix/suffix variation. They are enough for a stable exploratory figure, but they are weaker than 50 independently sourced benchmark examples.

## Recommended high-residual setup

If the goal is to make OOD residuals visibly larger under the current last-token residual metric, use the following protocol:

- set `--max_length 128` or higher so the disruptive part of the OOD prompt is not truncated away
- prefer OOD prompts whose strongest distribution shift happens near the final clause, because the residual uses the last-token hidden-state trajectory
- use `Spanish` or `Long context` for the cleanest cross-model increase in the current 9-model sweep
- use `Adversarial` with a longer context budget when you want a stronger small-model mid/late-layer effect

Empirically, with the corrected longer-context setup, the average cross-model amplification ranks as:

1. `Spanish`
2. `Long context`
3. `Adversarial`

On a focused Qwen 0.5B pilot with `max_length=128`, `Adversarial` becomes much stronger in mid and late layers, which means prompt truncation was suppressing part of the earlier effect.

## Paper sentence

Across model scales, adversarial prompts induce a persistent residual increase in mid-to-late layers, while the dependence on parameter count remains non-monotonic across GPT-2 and Qwen families.
