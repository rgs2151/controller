# Gamma Star collaborator bundle

This bundle contains only final-paper benchmark/model pairs. It excludes toxicity, Spanish L-CiteEval, L-CiteEval Small, superseded calibrations, and unreported models.

- `final_performance_id_ood.csv`: all methods and metrics from the final tables, with ID/OOD labels.
- `final_hinf_srob_analysis.csv`: H-infinity primary outcomes joined to final gamma_star/S_rob, the best reported comparator, and the signed H-infinity differential. Repeated conditions from one model share the same controller and are not independent gamma estimates.
- `final_selected_calibrations.csv`: one explicit final H-infinity calibration for each of 12 benchmark/model pairs.
- `final_controller_grid.csv`: only controller-grid configurations belonging to those 12 final calibrations.
- `derive_srob.py`: standalone standard-library script that recomputes `S_rob = 1 / gamma_star` from the final selections.

MGSM has no reported ID evaluation in the final table, so its rows are OOD only. HarmBench Direct is ID and the five jailbreak templates are OOD. Truthfulness English is ID and Spanish is OOD. L-CiteEval 8K is ID and 16K is OOD.

Definitions: `S_rob = 1 / gamma_star`. `hinf_differential = hinf_reliability - best_comparator_reliability`, with HarmBench first transformed to safe-response rate (`100 - ASR`) so positive is always better for H-infinity.

Run `python3 derive_srob.py` inside the extracted directory to create `derived_srob.csv` independently.
