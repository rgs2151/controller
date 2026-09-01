# Robust Steerability

Research code for measuring and controlling robust steerability in language-model representation dynamics under distribution shift.

## Setup

Create the environment from the repo root:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate robust-steerability
```

Install the package in editable mode if the environment was created without the pip step:

```bash
python -m pip install -e .
```

## Running Units

Compact units live in `parking/` while they are being explored and in `figs/` after graduation.

Run the first unit directly from the repo root once its documented input data are available:

```bash
python parking/residual_geometry/residual_geometry.py
```

Each unit owns its own `cache/` and `plots/` folders. Existing caches are reused by default. To recompute a unit, delete that unit's relevant cache or run the unit with `--recompute` when supported.

## Data

Place organized analysis-ready data under:

```text
data/representation_dynamics/
```

See `data/README.md` for the required arrays, record hierarchy, variable meanings, split rules, alignment rules, and current inventory.

## Project Docs

- `AGENTS.md`: working rules for Codex agents.
- `ORGANIZATION.md`: folder layout, compact-unit pattern, cache rules, and README template.
- `STYLE.md`: figure styling standards.
- `DECISIONS.md`: project-wide scientific and analytical decisions.
- `skills/`: repo-local Rudra workflow skills.
