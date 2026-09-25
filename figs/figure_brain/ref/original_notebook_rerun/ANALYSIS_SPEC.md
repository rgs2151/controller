# Matched P12–GPT-2 empirical recreation

This directory reruns only Erfan Zabeh's matched empirical analysis used in the
brain/LLM figure. It does not import results from any other exploratory unit.

Canonical sources:

- `source/final_figure_brain.ipynb`: exact copy of `NeuroAnalysis/final_figure.ipynb`.
- `source/final_figure_gpt2.ipynb`: exact copy of `NeuroAnalysis/final_figure_llm_fitted.ipynb`.
- `source/network_size_residual_explore.py`: exact historical prompt-bank generator
  from Git commit `8cd7a02f2057fe2af5c19aae958f9771d9c1a8ce`.

The executable copies in this directory preserve the numerical definitions of
those notebooks. Their only infrastructure changes are local paths and durable
intermediate caches.

Outputs are separated as follows:

- `cache/brain/sessions`: decoded P12 MAT sessions.
- `cache/brain/preprocessed`: theta tokens, trial metadata, channel metadata,
  frozen splits and normalization inputs.
- `cache/gpt2/prompt_banks.json`: exact 50/50/50 prompt banks.
- `cache/gpt2/trajectories.pt`: all GPT-2 Small token trajectories required by
  the analysis (all 12 layers for ID and layer 6 for both OOD conditions).
- `cache/*/fits`: fitted ridge/MLP parameters and per-example residual arrays.
- `results`: regenerated numerical tables.
- `plots`: regenerated standalone empirical panels.
- `manifest.json`: provenance, checksums, package versions and run status.

Run `python run.py --all`. Existing valid caches are reused; pass `--force` to
recompute them.
