# BeaconXAI

Research codebase for **budgeted counter-evidence auditing** in black-box time-series models.

Repository scope: two companion studies.

- **Part 1**: component localization (counter-evidence shortlisting, low budget).
- **Part 2**: risk auditing with adaptive budget (early stopping) and edge-server feasibility.

## Status

- Code and scripts are reproducible for published tables.
- Final submission-oriented tag: `v12-final` (see tags).
- Datasets are not bundled (download/preprocess locally into `data/`).

## Key Results (Claim-Safe)

### Part 1 (Localization, PAMAP2)

Adaptive shortlist (`adaptive_v2`) improves localization over uniform occlusion in the confirmed PAMAP2 setup.

| Method | loc@1 | hit@3 |
|---|---:|---:|
| Uniform (interp) | 0.1211 | 0.2109 |
| BEACON adaptive_v2 | 0.2969 | 0.4609 |

### Part 2 (Risk audit, adaptive budget)

Early stopping reduces average additional checks from `Q=64` to about `~11` (roughly 5–6x budget reduction), but equal-budget quality gains over uniform are **not** supported in the current protocol.

| Dataset | q_mean | ΔAUROC vs equal-budget uniform | p-value |
|---|---:|---:|---:|
| UCI HAR | 10.69 | -0.0528 | 0.252 |
| PAMAP2 | 11.10 | -0.0736 | 0.200 |
| WISDM | 10.59 | -0.0481 | 0.420 |

Interpretation: strong engineering gain (latency/budget), cautious quality claim.

## Edge / Portability Summary

Measured on HAR + ExtraTrees (Ryzen 7 7840HS):

- `inference_only p50`: **7.30 ms**
- `BEACON core Q64 p50`: **436.07 ms**
- `early-stop estimate`: **81.8–93.3 ms** (~5x faster than core Q64)

Raspberry Pi 4 in this repo is **estimate-only** (not direct measurement): about `0.7–1.0 s` per audited sample under current assumptions.

## Repository Layout

- `beaconxai/` — core library (audit logic, features, utilities)
- `scripts/` — experiment runners and aggregators
- `configs/` — experiment configs (cross-dataset, v10/v11)
- `artifacts/` — manuscript insert packs and summaries
- `supplementary/` — export-ready tables used in appendix/supp

Local-only (gitignored): `data/`, `outputs_composite/`, build caches/logs.

## Setup

### Conda

```bash
conda env create -f environment.yml
conda activate beaconxai
```

### venv

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-extra.txt
.venv/bin/pip install -e .
```

## Repro Commands

### Quick

```bash
make reproduce-quick
```

### Main cross-dataset policy block (v11)

```bash
.venv/bin/python scripts/run_cross_dataset_benchmark.py \
  --config configs/experiments_v11_cross_dataset.json \
  --out-root outputs_composite/v11_cross_dataset
```

Then aggregate:

```bash
.venv/bin/python scripts/aggregate_v11_results.py
.venv/bin/python scripts/make_v11_summary_table.py
```

### BEACON vs uniform benchmark (v12)

```bash
.venv/bin/python scripts/benchmark_beacon_vs_uniform.py \
  --datasets har,pamap2,wisdm \
  --budgets 16,32,64 \
  --n-boot 2000 \
  --adaptive-v2 \
  --out-root outputs_composite/v12_beacon_vs_uniform_full
```

### Early-stop full run (HAR/PAMAP2/WISDM)

```bash
./scripts/run_early_stop_v12_full.sh
```

Produces:

- `outputs_composite/early_stop_v12_full_summary.csv`
- `artifacts/early_stop_v12_full_summary.md`

## Canonical Files for Manuscript Claims

- `artifacts/part2_earlystop_insert_pack_ru_en_v12.md`
- `artifacts/handoff_science_analysis_v12_draft_ru.md`
- `artifacts/v11_full_summary.md`
- `outputs_composite/edge_portability_profile_v12.csv`
- `outputs_composite/edge_resource_budget_table_v12.csv`
- `outputs_composite/edge_portability_earlystop_estimate_v12.csv`
- `outputs_composite/early_stop_v12_full_summary.csv`

## Notes on TAN/Fuzzy

- `logit-panel` is the practical baseline.
- `TAN` shows local improvements on specific WISDM settings only.
- `Fuzzy` (current fixed-rule setup) does not support a stable positive quality claim.

Detailed comparisons are kept in supplementary artifacts.

## Citation

If you use this code, cite the associated BEACON-XAI papers and link this repository tag.

