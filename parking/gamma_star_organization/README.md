# Gamma Star Organization

## Purpose

This compact analysis unit collects the scalar H-infinity synthesis metadata needed to study `gamma_star` and robust steerability without copying model weights, Jacobians, gains, hidden states, prompts, generations, or judge payloads.

The paper-facing figures are intentionally narrower than the archival inventory. **Permanent unit rule:** figures include only HarmBench, Truthfulness, and MGSM. All L-CiteEval variants are excluded unless a future request explicitly opts them back in. Archival controller tables may retain L-CiteEval rows so source calculations are not destroyed.

## Final figure registry

`plots/final_report_points.csv` is the sole source for `plots/srob_ood_reliability.{pdf,png}`. Its calibration binding is explicit in `FINAL_CALIBRATIONS` inside `gamma_star_organization.py`; no run is selected heuristically.

The 34 plotted observations are:

- Truthfulness: 4 final models x Spanish OOD True percentage = 4 points.
- HarmBench: 3 final models x 5 adversarial templates = 15 points. Reliability is `100 - ASR`.
- MGSM: 3 final models x 5 reported transfer languages = 15 points.

The figure is a point inventory only. It contains no best-fit line, Spearman coefficient, sample-count annotation, ambiguity marker, or inferential claim. Color and marker identify the benchmark. Tiny horizontal offsets are applied only when points coincide exactly; the CSV retains the unmodified values.

## Compact outputs

- `plots/final_controller_grid.csv`: controller configurations only from the 12 explicitly registered final benchmark/model calibrations.
- `plots/final_selected_calibrations.csv`: exactly one selected H-infinity calibration for each of the 12 final benchmark/model pairs.
- `plots/final_report_points.csv`: explicit final-paper subset used by the scatter.
- `plots/manifest.json`: schema definitions, counts, output paths, and extraction warnings.
- `plots/srob_ood_reliability.pdf` and `.png`: vector and raster renderings of the final-only point plot.

## Variables

- `gamma_star`: minimum feasible H-infinity attenuation boundary saved by synthesis.
- `s_rob`: `1 / gamma_star`, in the saved controller coordinates.
- `ood_reliability`: benchmark-native held-out result oriented so higher is better.
- `calibration_id`: explicit calibration attempt bound to the final table row.
- `performance_source` and `selection_source`: exact provenance for the reported outcome and controller selection.

Raw `gamma_star` and `s_rob` can depend on coordinates, costs, normalization, model scale, and calibration protocol. This unit preserves those fields; it does not claim that raw values are automatically commensurate across every model and benchmark.

The output directory deliberately contains no broad exploratory CSV. Every generated CSV is safe for the final-results collaboration bundle. Regeneration also removes the deprecated broad files (`controller_grid.csv`, `selected_calibrations.csv`, `performance_long.csv`, and `analysis_candidates.csv`) if they are present.

## Reproduction

From the repository root:

```bash
python3 code/parking/gamma_star_organization/gamma_star_organization.py
```

Use `--check` to validate sources and print counts without writing outputs. The extractor uses the Python standard library for metadata extraction; the renderer uses Matplotlib with Arial typography.

To build the final-only collaborator archive after regenerating the unit:

```bash
python3 code/parking/gamma_star_organization/build_collaborator_bundle.py
```

The resulting `exports/gamma_star_collaborator_bundle.zip` contains both ID and OOD performance, the H-infinity differential/SRob analysis table, the 12 final calibration selections, the matching controller grids, a data dictionary, and a checksum manifest.

Render the controller-level structural figures with:

```bash
/home/dev/miniconda3/bin/python3.13 code/parking/gamma_star_organization/render_task_structure.py
```

- `plots/srob_parameter_size_truthfulness_main.{pdf,png}` is the main five-model Truthfulness scaling figure. Both axes are logarithmic; Qwen-2.5-32B is excluded; and no panel-title statistics are shown. Its exact source rows and log-log fit metadata are stored in matching `.csv` and `_fit.json` files.
- `plots/srob_parameter_size_truthfulness_axis_diagnostic_1x4.{pdf,png}` is the retained exploratory coordinate-system diagnostic. It shows log-log, log-linear, linear-log, and linear-linear views of the same five models, with coordinate-specific fits plus Kendall/Mann–Kendall trend statistics. Its four fit records are stored in the matching `_fit.json` file.
- `plots/srob_parameter_size.{pdf,png}` is the cross-benchmark parameter plot colored by task. Both axes are logarithmic; light dashed links identify the same model evaluated in two tasks.
- `plots/srob_task_hierarchical.{pdf,png}` is the clustering heat map, using Ward clustering on one-dimensional `log10(S_rob)` and an ordered pairwise-distance matrix.

## Final performance sources

- `figs/bench_table/truthfulness/truthfulqa_spanish.md`
- `figs/bench_table/harmful/harmbench_full.md`
- `figs/bench_table/mgsm/mgsm_full.md`

## Controller sources

- `benchmarks/*/cache/*/calibrations/h_infinity/**/selection.json`
- `benchmarks/*/cache/*/calibrations/h_infinity/**/grid/controllers/*.pt`
- `benchmarks/*/cache/*/calibrations/h_infinity/**/controller_diagnostics/runs/*/score.json`
