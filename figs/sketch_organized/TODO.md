# Robust-steerability paper TODO and panel coverage

## Implementation work

1. Connect the complete H∞ controller to the experiment pipeline. — **Done**
2. Optimize H∞ for memory and runtime. — **Done**

## Experiments

1. OOD and adversarial A-LQR failure exploration. — **Done**
2. Residual analysis across models and OOD conditions. — **Done**
3. Original/A-LQR/S-PID/H∞ benchmark. — **Current 50-prompt suite done; paper-final scale remains to be fixed**
4. Gain sweep to stop A-LQR/H∞ oversteering. — **Missing**
5. Test whether \(S_{\mathrm{rob}}=1/\gamma^\star\) predicts OOD steering failure. — **Missing**
6. Test whether H∞ helps most where \(S_{\mathrm{rob}}\) predicts fragility. — **Real toxicity-only precursor shown; final experiment missing**
7. Calibration-size and disturbance-geometry ablations. — **Missing, lower priority**

## Figure coverage

- Figure 1: complete conceptual framing.
- Figure 2: real existing dynamics, residual, OOD-failure, and model-scale results.
- Figure 3: calibration diagram retained; prediction panels remain placeholders.
- Figure 4: real truthfulness, ID toxicity, and MMLU utility; intervention energy remains a placeholder.
- Figure 5: real current OOD benchmark; crossover panels are explicitly preliminary.
- Figure 6: long-context length sweep remains a placeholder.
- Figure S1: real H∞ parity, attenuation, speed, and memory diagnostics.
- Figure S2: real actual-versus-linearized trajectory in panel B; other nonlinear-excursion panels remain placeholders.
- Figure S3: gain and ablation panels remain placeholders.
