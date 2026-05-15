# BeaconXAI

BEACON-XAI is a research codebase for **budgeted counter-evidence shortlisting** in low-query black-box settings.
The main target is multichannel time series, with additional tabular pilots.

## What This Repo Contains

- `beaconxai/` — core method implementation (partitioning, neutralization, audit logic)
- `scripts/` — experiments and table/figure generation
- `tests/` — smoke/unit tests
- `requirements.txt` — minimal base deps

`data/` and `outputs*/` are local-only (gitignored).

## Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-extra.txt
```

## Reproduce Main Paper Blocks

All commands below write artifacts to `outputs_composite/`.

### 1) Component-level benchmark (PAMAP2/WISDM)

```bash
.venv/bin/python scripts/run_component_conflict_benchmark.py \
  --npz-path data/pamap2_acc9_w200s100_p095.npz \
  --dataset-name pamap2 --model extratrees \
  --q-values 8,16 --group-mode pamap3 --time-bins 12 \
  --neutralizer interp --partition-mode sensor_group_time \
  --out outputs_composite/component_localization_results_table.csv \
  --per-sample-out outputs_composite/component_localization_per_sample.csv
```

### 2) Adapted TS-XAI baselines (RISE-style, KernelSHAP-components)

```bash
.venv/bin/python scripts/run_pamap2_tsxai_baselines.py \
  --out-summary outputs_composite/pamap2_tsxai_baselines_summary.csv \
  --out-per-sample outputs_composite/pamap2_tsxai_baselines_per_sample.csv
```

### 3) Tabular pilot (Adult, RF/XGB)

```bash
.venv/bin/python scripts/run_tabular_conflict_benchmark.py \
  --out-dir outputs_composite
```

### 4) Practical HAR portability/fault blocks

```bash
.venv/bin/python scripts/run_har_sensor_fault_benchmark.py \
  --out-summary outputs_composite/har_sensor_fault_localization_table.csv \
  --out-per-sample outputs_composite/har_sensor_fault_localization_per_sample.csv

.venv/bin/python scripts/run_har_hidden_conflict_benchmark.py \
  --out-summary outputs_composite/har_hidden_conflict_localization_table.csv \
  --out-per-sample outputs_composite/har_hidden_conflict_localization_per_sample.csv

.venv/bin/python scripts/measure_portability.py \
  --out outputs_composite/edge_portability_profile.csv

.venv/bin/python scripts/estimate_resource_budget.py \
  --profile-csv outputs_composite/edge_portability_profile.csv \
  --out outputs_composite/edge_resource_budget_table.csv
```

### 5) Audit panel tables

```bash
.venv/bin/python scripts/make_audit_panel_tables.py --n-boot 1000
```

### 6) Hidden-conflict significance (paired bootstrap)

```bash
.venv/bin/python scripts/compute_hidden_conflict_significance.py \
  --per-sample outputs_composite/har_hidden_conflict_localization_per_sample.csv \
  --out outputs_composite/table8_significance.csv
```

### 7) One-command paper rerun

```bash
make all
# or
.venv/bin/python scripts/run_all.py
```

## Final Aggregated Results File

Primary one-file summary for manuscript work:

- `outputs_composite/article_results_all_in_one.md`
- `outputs_composite/edge_resource_budget_table.csv`

## Reproducibility Notes

- Default random seed is typically `42` (see each script).
- Compare methods together with `model calls`, not only ranking metrics.
- In low-dimensional tabular settings, uniform occlusion can be near-exhaustive at moderate budgets.
- For portability profiling, CPU-affinity/nice constraints are supported; fixed frequency may require root/system support.
- Data preparation steps are documented in `DATA_PREPARATION.md`.

## Sanity Check

```bash
.venv/bin/python -m py_compile beaconxai/*.py scripts/*.py tests/*.py
```

## Pre-Publication Checklist

- Verify `LICENSE`, citation metadata (`CITATION.cff`) and reproducibility commands before submission.
- Add canonical citation metadata (`CITATION.cff` or BibTeX in docs).
- Ensure referenced figures/tables in manuscript are present and numbered consistently.
