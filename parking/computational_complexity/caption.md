# computational_complexity

## Caption

Computational cost of A-LQR and H-infinity steering for Llama-3.2-1B-Instruct on one NVIDIA GeForce RTX 5090. Offline synthesis reports medians and interquartile ranges across 20 cache-cleared trials using the deployed full-state A-LQR and reduced-state H-infinity configurations. The gamma-star panel shows the production feasibility bracket contracting over 24 bisection iterations. Deployment panels report median and interquartile range across seven trials of the same 16 HarmBench prompts with exactly 50 generated tokens per prompt and KV caching disabled. H-infinity synthesis completes in 0.125 s; its median decoding latency is 0.050 ms/token above A-LQR, while peak allocated VRAM is identical at the reported precision.

## Panel Notes

- A: Median offline synthesis seconds for deployed A-LQR and H-infinity on a logarithmic axis.
- B: Width of the feasible gamma bracket after each H-infinity bisection iteration.
- C: End-to-end generation latency in milliseconds per generated token for Original, A-LQR, and H-infinity.
- D: Peak PyTorch-allocated GPU memory for the same deployment trials.

## Checks

- Visual encodings checked against: `computational_complexity.py`, `plots/computational_complexity.png`, `cache/synthesis_trials.csv`, and `cache/inference_trials.csv`.
- Statistics checked against: `cache/summary.json`; descriptive medians and interquartile ranges only.
- Remaining uncertainty: the measurements characterize one model, behavior, GPU, batch size, and cache-off decoding condition; they are not a hardware-independent equivalence claim.
