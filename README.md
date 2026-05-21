# BeaconXAI

BEACON-XAI is a research codebase for **budgeted counter-evidence shortlisting** in low-query black-box settings.
The main target is multichannel time series, with additional tabular pilots.

## Release Status (Current)

This repository is prepared as a **research release** with:

- final BEACON core/audit pipeline code (`beaconxai/`, `scripts/`, `tests/`);
- reproducible experiment scripts for Part1/Part2 blocks;
- embedded policy-layer export/build path (`embedded/`) for ESP32-C3 compilation;
- local run artifacts excluded from git (outputs, logs, `.pio` caches, generated headers/binaries).

### What is tracked vs not tracked

- **Tracked:** source code, tests, docs, manuscript helper artifacts.
- **Not tracked:** datasets, `outputs*`, temporary logs, generated embedded build files.

This keeps the repo clean for review and reproducible reruns.

## Quick Release Checklist

```bash
git status
pytest -q
```

Optional build sanity:

```bash
cd embedded
pio run -e esp32c3
pio run -e esp32c3 -t size
```

If `git status` is clean after these checks, repository is release-ready.

## What This Repo Contains

- `beaconxai/` — core method implementation (partitioning, neutralization, audit logic)
- `scripts/` — experiments and table/figure generation
- `tests/` — smoke/unit tests
- `requirements.txt` — minimal base deps

`data/` and `outputs*/` are local-only (gitignored).

## Canonical Paper Artifacts (use these for claims)

Use only these files as the final manuscript source:

- `outputs_composite/table8_significance.csv`
- `outputs_composite/audit_panel_vs_scalar.csv`
- `outputs_composite/audit_policy_deltas.csv`
- `outputs_composite/audit_beacon_vs_uniform.csv`
- `outputs_composite/edge_resource_budget_table.csv`
- `outputs_composite/tinyxai_full_audit_cost.csv`
- `outputs_composite/part2_extended_v6/table2_q64_metric_ci.csv`
- `outputs_composite/edge_resource_budget_q64_profile.csv`
- `artifacts/beacon_article_insert_pack_ru.md`
- `artifacts/beacon_q1_submission_pack.md`

Everything else in `outputs_composite/` should be treated as exploratory or archived runs.

## Environment

Conda:

```bash
conda env create -f environment.yml
conda activate beaconxai
```

Pip/venv:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-extra.txt
.venv/bin/pip install -e .
```

## Reproduce Main Paper Blocks

All commands below write artifacts to `outputs_composite/`.

### Quick Reproduce Entrypoints

```bash
# Full key-table reproduce (long)
make reproduce

# Quick smoke reproduce
make reproduce-quick
```

This runs:
- `scripts/benchmark_beacon_vs_uniform.py` (Table-1 style: BEACON vs uniform, Q-sweep),
- `scripts/run_cross_dataset_benchmark.py` + `aggregate_v11_results.py` + `make_v11_summary_table.py` (Table-2 style policy comparison).

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
.venv/bin/python scripts/make_audit_panel_tables.py --n-boot 1000 --tan-bins 6
```

This step now exports alert-policy comparison with four modes:
`scalar`, `panel` (logit), `fuzzy_policy`, `tan_policy`.
It also exports bootstrap deltas for policy-vs-policy checks:
`outputs_composite/audit_policy_deltas.csv`.

### 5.1) One-command final policy/statistics rerun

```bash
./scripts/run_full_experiments.sh --bootstrap 1000 --tan-bins 6
```

This script regenerates:
- `outputs_composite/audit_panel_vs_scalar.csv`
- `outputs_composite/audit_policy_deltas.csv`
- `outputs_composite/audit_beacon_vs_uniform.csv`
- `outputs_composite/table8_significance.csv`
- `outputs_composite/edge_resource_budget_table.csv`

### 6) Hidden-conflict significance (paired bootstrap)

```bash
.venv/bin/python scripts/compute_hidden_conflict_significance.py \
  --per-sample outputs_composite/har_hidden_conflict_localization_per_sample.csv \
  --out outputs_composite/table8_significance.csv
```

### 7) Hidden-conflict detection with TAN (new)

```bash
.venv/bin/python scripts/run_har_hidden_conflict_tan.py \
  --out-summary outputs_composite/har_hidden_conflict_detection_tan_table.csv \
  --out-per-sample outputs_composite/har_hidden_conflict_detection_tan_per_sample.csv
```

### 8) Q-sweep: BEACON vs uniform (detection panel)

```bash
.venv/bin/python scripts/run_beacon_uniform_q_sweep.py \
  --q-values 16,32,64 \
  --neutralizers interp,zero,channel_mean,train_class_mean \
  --n-boot 500 \
  --out-summary outputs_composite/beacon_uniform_q_sweep.csv \
  --out-bootstrap outputs_composite/beacon_uniform_q_sweep_bootstrap.csv
```

`channel_mean -> mean` and `train_class_mean -> class_mean` are accepted aliases.

### 9) Extended Part2 pipeline (BEACON features -> TAN/fuzzy/fuzzy-gate)

```bash
.venv/bin/python scripts/run_part2_extended.py \
  --dataset data/uci_har_shifted.npz \
  --n-total 5000 \
  --q-max 16 \
  --seed 42 \
  --out outputs_composite/part2_extended
```

Main artifacts:
- `audit_features_beacon_core.csv`
- `audit_features_uniform.csv`
- `tan_sweep_results.csv`
- `tan_final_test.csv`
- `fuzzy_policy_results.csv`
- `fuzzy_final_test.csv`
- `policy_comparison.csv`
- `bootstrap_deltas.csv`
- `split_manifest.json`

### 10) Full-audit cost envelope (TinyXAI section)

```bash
.venv/bin/python scripts/estimate_full_audit_cost.py \
  --profile-csv outputs_composite/edge_portability_profile.csv \
  --resource-csv outputs_composite/edge_resource_budget_table.csv \
  --out outputs_composite/tinyxai_full_audit_cost.csv
```

### 11) One-command paper rerun

```bash
make all
# or
.venv/bin/python scripts/run_all.py
```

`make all` requires locally prepared datasets. Run `DATA_PREPARATION.md` first.
Optional new blocks:
- `--run-tan-detection` (hidden-conflict detection with TAN)
- `--run-audit-panel` (includes fuzzy-panel policy table)

### 12) v6 Q1 candidate package

This block is the strict Q1 gate: it runs Q-sweep, fuzzy_v5/soft-mix policies,
sensor anomaly localization, simple baselines, full-audit cost, and a claim registry.

Fast smoke:

```bash
.venv/bin/python scripts/run_v6_experiments.py \
  --skip-policy-grid \
  --q-list 4 \
  --anomaly-model extratrees \
  --anomaly-max-test 32 \
  --n-boot 20 \
  --base-out outputs_composite/part2_extended_v6_smoke
```

Full v6 candidate:

```bash
/home/lebedeffson/Code/deep-neuro-fuzzy/.venv/bin/python scripts/run_v6_experiments.py \
  --dataset data/uci_har_shifted.npz \
  --model cnn1d \
  --device cuda \
  --q-list 16,32,64 \
  --neutralizers interp,zero,mean,class_mean \
  --n-total 5000 \
  --anomaly-model cnn1d \
  --anomaly-max-test 512 \
  --anomaly-fault-types spike,drift,stuck_sensor,dropout \
  --n-boot 1000 \
  --base-out outputs_composite/part2_extended_v6
```

Main v6 artifacts:
- `beacon_vs_uniform_q_sweep.csv`
- `bootstrap_deltas_v6.csv`
- `sensor_anomaly_localization.csv`
- `sensor_anomaly_bootstrap.csv`
- `tinyxai_full_audit_cost.csv`
- `manuscript_claim_registry_v6.csv`

### 13) v8 Track A runner + embedded policy export

Run v8 grid (time-bins/Q/neutralizer) and aggregate strict artifacts:

```bash
.venv/bin/python scripts/run_v8_experiments.py \
  --dataset data/uci_har_shifted.npz \
  --model extratrees \
  --time-bins-list 16 \
  --q-list 16,32,64 \
  --neutralizers interp \
  --preselect-mode adaptive_v2 \
  --n-boot 1000 \
  --base-out outputs_composite/part2_extended_v8
```

Main v8 artifacts:
- `beacon_vs_uniform_q_sweep_v8.csv`
- `bootstrap_deltas_v8.csv`
- `har_component_budget_summary_v8.csv`
- `manuscript_claim_registry_v8.csv`

Improved policy training blocks:

```bash
.venv/bin/python scripts/train_tan_improved.py \
  --bundle-dir outputs_composite/part2_extended_v8/tb16_q16_interp
```

```bash
.venv/bin/python scripts/train_fuzzy_improved.py \
  --bundle-dir outputs_composite/part2_extended_v8/tb16_q16_interp
```

Hybrid sensor-fault benchmark:

```bash
.venv/bin/python scripts/run_har_sensor_fault_benchmark.py \
  --npz-path data/uci_har_shifted.npz \
  --model extratrees \
  --q 16 \
  --enable-hybrid
```

Export policy layer to C++ header (`h(a(x))` only):

```bash
.venv/bin/python scripts/export_policy_to_cpp.py \
  --bundle-dir outputs_composite/part2_extended_v8/tb16_q16_interp \
  --out-header embedded/beacon_policy.h
```

## Quick Smoke Check

```bash
make smoke
```

## Final Aggregated Results File

Primary one-file summary for manuscript work:

- `outputs_composite/article_results_all_in_one.md`
- `outputs_composite/edge_resource_budget_table.csv`

## Reproducibility Notes

- All reported experiments use seed `42` unless otherwise stated.
- Supported partition modes:
  - `time_only` (time-first 1D chunks),
  - `time_channel` (balanced 2D chunks),
  - `channel_time` (channel-first chunks),
  - `sensor_group_time` (sensor-group chunks over time),
  - `fuzzy_chunks` (alias of `sensor_group_time` for manuscript wording).
- Compare methods together with `model calls`, not only ranking metrics.
- In low-dimensional tabular settings, uniform occlusion can be near-exhaustive at moderate budgets.
- For portability profiling, CPU-affinity/nice constraints are supported; fixed frequency may require root/system support.
- Data preparation steps are documented in `DATA_PREPARATION.md`.
- End-to-end reproduction flow is documented in `REPRODUCIBILITY.md`.
- Manuscript should cite a fixed code tag (e.g., `v1.0-submission`), not moving `main`.

## Submission Artifacts

Reference result artifacts can be attached as a GitHub Release bundle:

- `v1.0-submission-artifacts.zip`

Recommended contents: `table8_significance.csv`, `edge_resource_budget_table.csv`,
final manuscript tables, and a short `reproduction_log.txt`.

## Sanity Check

```bash
.venv/bin/python -m py_compile beaconxai/*.py scripts/*.py tests/*.py
```

## Pre-Publication Checklist

- Verify `LICENSE`, citation metadata (`CITATION.cff`) and reproducibility commands before submission.
- Add canonical citation metadata (`CITATION.cff` or BibTeX in docs).
- Ensure referenced figures/tables in manuscript are present and numbered consistently.
