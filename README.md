# BeaconXAI

BEACON-XAI is a research codebase for **budgeted black-box counter-evidence analysis**.
It focuses on low-query settings (`Q=8..16`) where we need a short ranked list of suspicious input components rather than a full attribution map.

## Scope

- Core method: BEACON (budgeted counter-evidence shortlisting)
- Target domain: multichannel time-series
- Models: black-box compatible (e.g., ExtraTrees), plus differentiable models (CNN) for gradient baselines
- Evaluation: component-level localization and shortlisting

## Repository Structure

- `beaconxai/` — core library (audit, partitioning, neutralization, scoring)
- `scripts/` — runnable experiments and preprocessing scripts
- `tests/` — smoke/unit checks
- `data/` — local datasets (gitignored)
- `outputs*/` — local artifacts/results (gitignored)

## Environment

Use a local virtual environment (example `.venv`):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For recent experiments, additional packages are used:

```bash
.venv/bin/pip install xgboost shap lime matplotlib
```

## Key Scripts (Current)

### Component-level time-series benchmark

- `scripts/run_component_conflict_benchmark.py`

Main use: evaluate BEACON / uniform / random / leave-one-group-out on
`sensor_group × time_bin` components with strict budget.

Example:

```bash
.venv/bin/python scripts/run_component_conflict_benchmark.py \
  --npz-path data/pamap2_acc9_w200s100_p095.npz \
  --dataset-name pamap2 --model extratrees \
  --q-values 8,16 --group-mode pamap3 --time-bins 12 \
  --neutralizer interp --partition-mode sensor_group_time \
  --out outputs_composite/pamap2_component.csv \
  --per-sample-out outputs_composite/pamap2_component_per_sample.csv
```

### Adapted time-series attribution baselines (PAMAP2)

- `scripts/run_pamap2_tsxai_baselines.py`

Compares BEACON (Q=16) against:
- RISE-style random masking (`64/128/256/512` calls)
- KernelSHAP-over-components (`128/256/512` calls)
- plus uniform/random references

Example:

```bash
.venv/bin/python scripts/run_pamap2_tsxai_baselines.py \
  --out-summary outputs_composite/pamap2_tsxai_baselines_summary.csv \
  --out-per-sample outputs_composite/pamap2_tsxai_baselines_per_sample.csv
```

### Tabular pilot (appendix-style)

- `scripts/run_tabular_conflict_benchmark.py`

Runs Adult conflict injection with RF/XGBoost and compares:
BEACON / uniform / random / LIME / KernelSHAP.

## Current Paper-Oriented Artifacts

Local (gitignored) article tables are generated under `outputs_composite/`, e.g.:

- `component_main_table_article.csv`
- `pamap2_sensitivity_table_article.csv`
- `pamap2_neutralizer_ablation_article.csv`
- `pamap2_tsxai_comparison_table_article.csv`

Draft text files (also local) are maintained in `outputs_composite/` during writing.

## Reproducibility Notes

- Many scripts use fixed `--seed 42` by default.
- All budget comparisons should be interpreted together with **model call counts**.
- In low-dimensional settings (e.g., Adult with 14 features), uniform occlusion can be near-exhaustive under moderate budgets.

## Testing / Sanity

Quick compile check:

```bash
.venv/bin/python -m py_compile beaconxai/*.py scripts/*.py tests/*.py
```

## License

No explicit license file is included yet. Add one before public release.
