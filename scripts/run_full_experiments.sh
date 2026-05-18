#!/usr/bin/env bash
set -euo pipefail

# Final reproducible table build for manuscript artifacts.
# Usage:
#   ./scripts/run_full_experiments.sh --bootstrap 1000 --tan-bins 6

BOOTSTRAP=1000
TAN_BINS=6
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bootstrap)
      BOOTSTRAP="$2"
      shift 2
      ;;
    --tan-bins)
      TAN_BINS="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "[run_full_experiments] python=${PYTHON_BIN} bootstrap=${BOOTSTRAP} tan_bins=${TAN_BINS}"

"${PYTHON_BIN}" scripts/make_audit_panel_tables.py \
  --n-boot "${BOOTSTRAP}" \
  --tan-bins "${TAN_BINS}" \
  --out-error-type outputs_composite/audit_error_type_specialization.csv \
  --out-incremental outputs_composite/audit_incremental_feature_value.csv \
  --out-panel-vs-scalar outputs_composite/audit_panel_vs_scalar.csv \
  --out-policy-deltas outputs_composite/audit_policy_deltas.csv \
  --out-beacon-vs-uniform outputs_composite/audit_beacon_vs_uniform.csv

"${PYTHON_BIN}" scripts/compute_hidden_conflict_significance.py \
  --per-sample outputs_composite/har_hidden_conflict_localization_per_sample.csv \
  --out outputs_composite/table8_significance.csv

"${PYTHON_BIN}" scripts/estimate_resource_budget.py \
  --profile-csv outputs_composite/edge_portability_profile.csv \
  --out outputs_composite/edge_resource_budget_table.csv

echo "[run_full_experiments] done"
