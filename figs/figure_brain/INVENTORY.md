# Figure Brain source inventory

| Reference panel | Content | Repository source | Reproduction status |
|---|---|---|---|
| A | Electrode coverage and example signals | `NeuroAnalysis/brain_local_linearization.ipynb`; existing `electrode_map_P12.pdf/png` | Existing vector/raster export is staged as `brain_electrode_map_existing_export`. Exact recomputation is blocked because ignored `NeuroAnalysis/cache/*.npz` and raw `.mat` inputs are absent locally. |
| B | MSIT task schematic | `NeuroAnalysis/StimPaper.pdf`, Figure 1 | No local plotting code. Can be cropped from the cited paper as user-approved task artwork. |
| C | Cortical information flow and local-linear equation | Reference composition and analysis specification | Reconstructed as `brain_information_flow_schematic` without empirical data. |
| D-left | Brain conflict-shift error reduction | `final_panelD_error_shrinkage_LVF_theta.csv` | Fully reproduced as `brain_error_shrinkage_conflict`. |
| D-right | Brain stimulation-context error reduction | Same CSV | Fully reproduced as `brain_error_shrinkage_stim_context`. |
| E | Brain context-length analysis | `final_panelE_context_length_LVF_theta.csv` | Source exists but intentionally excluded from this figure per user instruction. |
| F-left | Brain residual size across leads | `final_panelF_state_size_theta.csv` | Fully reproduced as `brain_state_size_across_leads`. |
| F-right | Brain residual size across LVF channel subsets | Same CSV | Fully reproduced as `brain_state_size_lvf_subsets`. |
| G | GPT architecture and trajectory schematic | No source artwork or generating code found | Not currently reproducible; requires manual redraw or original collaborator asset. |
| H | SST-2 task/shift schematic | No source artwork or generating code found | Not currently reproducible; may be supplied as task artwork or redrawn later. |
| I-left | GPT-2 Spanish-shift error reduction | `final_llmfit_panelD_error_shrinkage.csv` | Fully reproduced as `llm_error_shrinkage_spanish`. |
| I-right | GPT-2 long-context error reduction | Same CSV | Fully reproduced as `llm_error_shrinkage_long_context`. |
| J | GPT-2 context-length analysis | `final_llmfit_panelE_context_length.csv` | Source exists but intentionally excluded from this figure per user instruction. |
| K-left | GPT-2 residual size across layers | `final_llmfit_panelF_state_size.csv` | Fully reproduced as `llm_state_size_across_layers`. |
| K-right | GPT-2 residual size across hidden-dimension subsets | Same CSV | Fully reproduced as `llm_state_size_hidden_dimensions`. |

## Audit conclusions

- All eight requested quantitative subplots are reproducible from tracked final-result CSVs without rerunning neural or language-model inference.
- The surviving panel-A export is copied byte-for-byte into this unit and is never misrepresented as a fresh recomputation.
- The new unit reproduces plots from frozen summary data; it does not silently recompute or alter the original fitted models.
- The empirical neural pipeline itself is not currently rerunnable because its ignored session caches/raw data are absent from this checkout.
- Panels E and J remain available in `NeuroAnalysis/results/` but are deliberately excluded from this figure.
- Panels B, G, and H are artwork/schematic problems rather than missing numerical analyses.
