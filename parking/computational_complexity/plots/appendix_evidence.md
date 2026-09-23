# Computational Complexity Appendix Evidence

## Experimental scope

- Model: Llama-3.2-1B-Instruct at revision `9213176726f574b556790deb65791e0c5aa438b6`.
- Behavior/controller source: HarmBench harmful-request refusal.
- Hardware: NVIDIA GeForce RTX 5090 with 33.67 GB device memory.
- Evaluated-model KV cache: off.
- Controller horizon: 16 layers.
- Production dimensions: A-LQR $d=2048$; H-infinity $r_x=8$.

## M.1 Asymptotic synthesis complexity

- A-LQR performs one backward Riccati sweep: $O(L d^3)$ in the deployed full-state implementation.
- H-infinity performs feasibility sweeps plus one final gain collection: $O(N_{iter} L r_x^3)$ in the reduced-state implementation.
- Deployment applies one feedback update per controlled layer. The measured end-to-end model latency below includes all projections, hooks, and controller operations.

## M.2 Empirical synthesis runtime

- A-LQR median synthesis time: 0.201472 s across 20 cache-cleared trials.
- H-infinity median synthesis time: 0.124515 s across 20 cache-cleared trials.
- At the same reduced state dimension, H-infinity synthesis is 40.52 times the A-LQR controller-core time; its absolute median remains 0.124515 s.
- Production $\gamma^\star$: 0.13007522.
- Production bisection iterations: 24.
- Across 100 seeded bootstrap refits of the 50-prompt residual set, mean iterations were 24.00, maximum iterations were 24, and all fits converged: True.
- This pipeline synthesizes one prompt-aggregated control problem. Bisection iterations are therefore not a prompt-level random variable; bootstrap refits quantify calibration-set sensitivity without mislabeling repeated deterministic runs as independent prompts.

## M.3 Inference-time deployment

- Original median latency: 5.347 ms/token.
- A-LQR median latency: 5.409 ms/token (+1.16% versus Original).
- H-infinity median latency: 5.459 ms/token (+2.09% versus Original).
- A-LQR median peak allocated VRAM: 2.882 GB.
- H-infinity median peak allocated VRAM: 2.882 GB.
- Each trial uses the same 16 prompts and forces exactly 50 generated tokens per prompt.

## Reproducibility record

- Python: 3.11.16.
- PyTorch: 2.9.1+cu128.
- CUDA runtime: 12.8.
- NVIDIA: NVIDIA GeForce RTX 5090, 595.84, 32607 MiB.
- Git commit: `67a3f23ee1931e441a852c6a25ece1ffdd2e6748`.
