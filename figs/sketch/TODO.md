# Robust-steerability paper TODO

## Implementation work

1. Connect the complete H∞ controller to the experiment pipeline. — **Partial**
2. Optimize H∞ for memory and runtime. — **Missing**

## Experiments

1. OOD and adversarial A-LQR failure exploration. — **Done**
2. Residual analysis across models and OOD conditions. — **Done**
3. Full Original/A-LQR/S-PID/H∞ benchmark. — **Partial: several full and smoke runs exist**
4. Gain sweep to stop A-LQR/H∞ oversteering. — **Missing**
5. Test whether \(S_{\mathrm{rob}}=1/\gamma^\star\) predicts OOD steering failure. — **Missing**
6. Test whether H∞ helps most where \(S_{\mathrm{rob}}\) predicts fragility. — **Loose version done; final experiment missing**
7. Calibration-size and disturbance-geometry ablations. — **Missing, lower priority**
