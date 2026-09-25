# Figure Brain

This is one analysis unit. It contains one retained exploratory comparison and
one reference archive of the collaborator's original notebook reruns.

## Retained analysis

`plot.py` recreates `plots/residual_metric_behavior_grid_4x4.{png,pdf}`. The
four rows compare human and GPT-2 dynamics under distribution shift:

1. human aggregate correction magnitude and linear residual across horizons;
2. human trial-level residual measures versus response time;
3. GPT-2 aggregate correction magnitude and linear residual across horizons;
4. GPT-2 prompt-level residual measures versus correct-label NLL.

Gray is ID (`#8A8F94`) and red is OOD (`#8B1E1E`). Numerical inputs used by
the retained figure are in `results/`.

Run:

```bash
python plot.py
```

## Cross-subject conflict screen

`all_subject_conflict_screen.py` applies the same no-stimulation conflict-shift
linearization to every participant. For each participant it deterministically
selects the no-stimulation session with the most trials (latest session breaks
ties), caches theta tokens for its complete clean montage, and evaluates every
lead containing at least five clean bipolar channels. The displayed lead is the
one with the largest mean proportional OOD excess, `(OOD−ID)/ID`, across the
predefined 20, 40, 100, 200, 300, 400, 500, and 700 ms horizons; this prevents
high-amplitude or poorly scaled leads from winning
solely because their absolute z scale is large. This is an explicitly
exploratory, outcome-selected screen. The ridge dynamics and
baseline normalization are fitted only on low-conflict trials. Held-out
low-conflict trials are ID and every high-conflict trial is OOD. The script
writes both the complete all-lead table and the selected subject ranking to
`results/`, then generates the requested subject grid with independent axes.
Both the full-montage theta states and the complete fitted all-lead result table
are cached. Missing horizons are fitted incrementally and appended, while
cosmetic replots require neither raw-data preprocessing nor model refitting.
Twenty participants have no-stimulation neural recordings; P13 is absent from
this screen because the Zenodo release provides only `P13_Stim1`. As a numerical
regression check, every P12 bar exactly equals the original notebook output.

```bash
python all_subject_conflict_screen.py
```

`stimulation_context_illustration.py` builds the exploratory P10/P17 panel:
P10 no-stimulation conflict shift, P10 open-loop stimulation-context shift,
P17 closed-loop stimulation-context shift, and participant-specific response
accuracy. For each stimulation-context panel, dynamics and normalization are
fitted only in the participant's pure no-stimulation session. ID is held-out
data from that session; OOD is artifact-free, non-stimulated trials recorded
inside the corresponding stimulation session. This isolates contextual or
carryover changes in the dynamics without allowing electrical artifacts to
define prediction error. Neural axes remain raw linear residual RMS in baseline
z units. Full theta states and fitted horizon results are cached; subsequent
visual edits only replot the saved result tables.

```bash
python stimulation_context_illustration.py
```

## Reference notebook rerun

`ref/original_notebook_rerun/` is a provenance archive, not a nested analysis
unit. It contains the original brain and GPT-2 notebook snapshots, their
executable/completed copies, the exact caches used by that rerun, and the six
canonical numerical panels and result tables they produced.

## Cleanup boundary

Alternative behavior transformations, expanded-horizon variants, the failed
composite recreation, P11 exploratory output, duplicate plots, and Python byte
code are intentionally excluded. The complete pre-cleanup state is preserved
outside the unit at:

`/home/dev/controller/exports/figure_brain_checkpoints/figure_brain_pre_cleanup_2026-09-25`
