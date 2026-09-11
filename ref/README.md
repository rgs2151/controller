# References

Reference material records provenance and supports comparison with prior implementations. It is not imported by the `robust_steerability` package at runtime.

Keep reference material read-only. Move active implementation into the installable package or an owned compact unit.

- `lqr-activation-steering/`: pinned upstream implementation of Activation-LQR and S-PID.
- `2604.19018v1.pdf`: corresponding paper snapshot.
- `papers/iti_2306.03341.pdf`: [ITI paper](https://arxiv.org/abs/2306.03341) (SHA-256
  `19e55fa96a0b688044c6f75b41a7b39ca672b5a5193054f22302f6bfb5684834`).
- `papers/activation_addition_2308.10248.pdf`: [Activation Addition paper](https://arxiv.org/abs/2308.10248)
  (SHA-256 `dd3db9f8ed1c327d01343cb49aa29983368cf4ad241a413ce5d10e4ba1796b71`).
- `papers/activation_transport_2410.23054.pdf`: [Activation Transport paper](https://arxiv.org/abs/2410.23054)
  (SHA-256 `8b0b1ab0164e1db36ac5f23eb088424e0b479f260081e02bc10929d9671671de`).
- `papers/pid_activation_steering_2510.04309.pdf`: [PID activation-steering paper](https://arxiv.org/abs/2510.04309)
  (SHA-256 `f2ebc712cd8b441c75d630a5a46c30614dea8f59602f4b82271384d5cb59d70c`).
- `papers/odesteer_2602.17560.pdf`: [ODESteer paper](https://arxiv.org/abs/2602.17560) (SHA-256
  `098c86245f4a8c5f0b3ab8a0278c951fcd529c1b39ed6c10a9147c1dab5961af`).
- `NON_HINFINITY_PROTOCOL.md`: audited calibration, selection, evaluation, and
  source-revision record for every non-H-infinity comparison method.
- `h_infinity.py`: Hannah's final finite-horizon H-infinity implementation, preserved unchanged as the numerical oracle for `parking/h_infinity_optimization/` (SHA-256 `293c7f45a54c6ffb6caf2a13cec9b0e09396fd4921973bccc019ba09e19f96f0`).
- `erfan_applied_controller/`: Erfan's untouched September 2026 multimodel
  runner, result tables, logs, and six summary figures. Its H-infinity rows use
  the superseded controller and identity disturbance placeholder, so this
  folder is provenance only and must not be imported by active analyses.

The pinned upstream checkout does not currently contain a license file. Treat its source as a behavioral and mathematical reference unless reuse permission or an explicit license is confirmed. Independently implemented project code should retain paper and repository attribution where appropriate.
