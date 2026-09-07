# References

Reference material records provenance and supports comparison with prior implementations. It is not imported by the `robust_steerability` package at runtime.

Keep reference material read-only. Move active implementation into the installable package or an owned compact unit.

- `lqr-activation-steering/`: pinned upstream implementation of Activation-LQR and S-PID.
- `2604.19018v1.pdf`: corresponding paper snapshot.
- `h_infinity.py`: Hannah's final finite-horizon H-infinity implementation, preserved unchanged as the numerical oracle for `parking/h_infinity_optimization/` (SHA-256 `293c7f45a54c6ffb6caf2a13cec9b0e09396fd4921973bccc019ba09e19f96f0`).

The pinned upstream checkout does not currently contain a license file. Treat its source as a behavioral and mathematical reference unless reuse permission or an explicit license is confirmed. Independently implemented project code should retain paper and repository attribution where appropriate.
