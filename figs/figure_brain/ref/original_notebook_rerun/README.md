# Original notebook rerun reference

This folder preserves the exact rerun of the collaborator's original P12 and
GPT-2 Small notebooks. It is supporting provenance for `Figure Brain`; it is
not a separate analysis unit.

- `source/`: untouched canonical notebook snapshots and historical prompt code.
- `executed/`: localized executable notebooks and completed executions.
- `cache/`: decoded P12 sessions, P12 theta tokens, exact GPT-2 prompt banks,
  and cached trajectories.
- `results/`: the six canonical CSV result tables.
- `plots/`: the six canonical PDF/PNG panel pairs.
- `run.py`: cache-aware notebook rerunner.
- `manifest.json`: source hashes, raw-data hashes, versions, and provenance.
