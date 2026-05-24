#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lebedeffson/Code/BeaconXAI"
PY="/home/lebedeffson/Code/venv/bin/python"
OUT_ROOT="$ROOT/outputs_composite/q1_har_divstop_q12_lam0.5_seeds"
MODEL="$ROOT/models/har_extratrees_seed1.pkl"
LOG_DIR="$OUT_ROOT/logs"
SEED_MAX="${SEED_MAX:-5}"
N_BOOT="${N_BOOT:-1000}"

mkdir -p "$LOG_DIR"

{
  echo "[start] $(date -Is) q1_har_divstop_q12_lam0.5 seed_max=${SEED_MAX} n_boot=${N_BOOT}"
  echo "[model] ${MODEL}"

  for seed in $(seq 1 "$SEED_MAX"); do
    echo "[seed ${seed}] $(date -Is)"
    "$PY" "$ROOT/scripts/run_early_stopping_equal_budget.py" \
      --dataset "$ROOT/data/uci_har_shifted.npz" \
      --n-total 600 \
      --time-bins 16 \
      --q-max 12 \
      --seed "$seed" \
      --tol 0.0 \
      --min-q 12 \
      --n-boot "$N_BOOT" \
      --baseline fixed_uniform \
      --policy-train-mode prefix_mix \
      --policy-prefix-list 5,8,10,12,16,24,32,64 \
      --order-mode diverse_importance \
      --div-lambda 0.5 \
      --model-cache "$MODEL" \
      --out "$OUT_ROOT/seed_${seed}"
  done

  "$PY" "$ROOT/scripts/aggregate_early_stop_seeds.py" \
    --input-dir "$OUT_ROOT" \
    --out-csv "$OUT_ROOT/all_seeds.csv" \
    --out-summary "$OUT_ROOT/all_seeds_summary.csv"

  echo "[finish] $(date -Is) q1_har_divstop_q12_lam0.5"
} 2>&1 | tee "$LOG_DIR/run.log"
