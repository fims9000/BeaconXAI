# BeaconXAI

Research codebase for **budgeted counter-evidence auditing** in black-box time-series models.

Repository scope: two companion studies.

- **Part 1**: component localization (counter-evidence shortlisting, low budget).
- **Part 2**: risk auditing with adaptive budget (early stopping) and edge-server feasibility.

## Research update: verified counterfactuals under shared Q8/Q16 (November 2026)

The historical `v12-final` tag and the two original studies above are unchanged. The **draft research branch** adds a third experimental track: budget-exact finite-donor counterfactual certificates for time-series classifiers.

- **Complete cross-dataset scientific status:** [counterfactual research summary](artifacts/beacon_counterfactual_research_status_2026-11-22.md). Contains the verified NATOPS, UWaveGestureLibrary, PenDigits, Epilepsy and RacketSports numbers, model accuracies, exact McNemar comparisons, failed hypotheses and TRAIN-only overhead.
- NATOPS ExtraTrees: compact two-donor certificates at Q16 **67/180 vs 51/180** against a strong forward control (paired p=0.00249). This is a conditional operator-dependent result, not universal BEACON superiority.
- UWave ExtraTrees: adaptive pair model **160/320 vs 141/320** static (p=0.0183), but **classifier accuracy only 62.2%**; additive adaptive is higher at 175/320.
- PenDigits raw MLP: additive adaptive **257/320 vs 252/320**, p=0.125; **classifier accuracy 94.77%**, so improvement is **not confirmed** on this model.
- D3/D4 transfer of previously certified masks is poor for both learned variants; **do not describe certification on D1+D2 as invariance across donors**.
- SpokenArabicDigits ExtraTrees, 92.72% accuracy, Q16: K3 (adding one distant donor) gave **80/320** fresh D4+D5 transfers versus **91/320** under K2 (paired p=0.0433, negative result). This required a **documented pre-inference interpolation amendment** for aeon's variable-length MFCC inputs; see [results](artifacts/beacon_spokenarabic_k3_transfer_results_2026-11-22.md).

All methods count one **evaluated input** to the black-box model as one online query (Q=8 or 16 **total across selected donors**), while offline TRAIN-only calibration and independent posthoc donor audits are separately reported in the experiment manifests. Validity means a directly measured target-class switch on the specified edited signal; it does not establish causal attribution, physical plausibility, global minimality or cross-subject transport.

Reproduce using the GitHub Actions workflows `uea_natops_sparse_q16.yml`, `uea_uwave_pairwise_adaptive.yml`, and `uea_pendigits_adaptive.yml` under `.github/workflows/`. Each workflow stores `per_sample.csv`, `summary.csv`, `manifest.json` and provenance. TRAIN/TEST archives are fetched and checked by matching `scripts/preprocess_uea_*.py`. The workflows are research experiments, not a production release.

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

