# Source-faithful comparison methods

This package is the only implementation path for A-LQR-paper comparisons made
after the historical `paper_benchmark_50` unit. It deliberately does not
contain H-infinity.

The operators, calibration sizes, intervention sites, generation settings,
and sweep grids are frozen from:

- A-LQR core: `trustworthyrobotics/lqr-activation-steering` at
  `c2e0c8450797e0dc234b2c53475267a2cfb2457a`.
- A-LQR baseline adapters: the same repository at
  `11b49bad15c02f0a23d9481980d4b88f2d6313a5`, the final parent before the
  adapters were removed from the public tree.
- ITI: `likenneth/honest_llama` at `2c6b217...`.
- Mean/Linear-AcT: `apple/ml-act` at `d2c3560...`.
- PID-AcT: `dungnvnus/pid-steering` at `d705c44...`.
- ODESteer: `ZhaoHongjue/odesteer` at `8a3c481...`.

`protocol_manifest()` requires a concrete model revision and records all of
these source revisions. The evaluation sample count is independent of every
calibration and sweep setting. The pilot is five repetitions of 50 examples;
changing 50 to the full count later must not change the fitted data sizes,
sweep grid, intervention site, generation settings, or model-loading protocol.

The high-level calibration functions enforce the source fit sizes before any
model work begins: 200/200 plus 50 Jacobian prompts for toxicity A-LQR/S-PID,
12/12 plus one Jacobian prompt for truthfulness A-LQR/S-PID, 100/100 for
ActAdd, 80/80 for ITI, 200/200 for AcT, and 5000/5000 (toxicity) or 1000/1000
(truthfulness) for ODESteer. Every sweep candidate is retained.

Unsupported checkpoints fail. They are not assigned borrowed layers, gains,
or strengths. A new checkpoint needs a newly run, recorded source sweep.
