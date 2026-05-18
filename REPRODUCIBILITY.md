# Reproducibility

## 1) Create environment

Conda:

```bash
conda env create -f environment.yml
conda activate beaconxai
```

Pip:

```bash
python -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements-extra.txt
```

## 2) Prepare datasets

Follow `DATA_PREPARATION.md`.

## 3) Run smoke checks

```bash
make smoke
```

## 4) Run full paper pipeline

```bash
make all
```

`make all` requires locally prepared datasets.

## 5) Regenerate key manuscript artifacts

- Hidden-conflict table: `outputs_composite/har_hidden_conflict_localization_table.csv`
- Significance table: `outputs_composite/table8_significance.csv`
- TAN hidden-conflict detection table: `outputs_composite/har_hidden_conflict_detection_tan_table.csv`
- Portability profile: `outputs_composite/edge_portability_profile.csv`
- Resource budget: `outputs_composite/edge_resource_budget_table.csv`
- Unified summary: `outputs_composite/article_results_all_in_one.md`

## 6) Frozen submission version

For manuscript citation use tag:

- `v1.0-submission`

