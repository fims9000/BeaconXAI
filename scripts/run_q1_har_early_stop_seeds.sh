#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lebedeffson/Code/BeaconXAI"
PY="/home/lebedeffson/Code/venv/bin/python"
OUT_ROOT="$ROOT/outputs_composite/q1_har_early_stop_seeds"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$LOG_DIR"

{
  echo "[start] $(date -Is) q1_har_early_stop_seeds"
  for seed in $(seq 1 10); do
    echo "[seed ${seed}] $(date -Is)"
    "$PY" "$ROOT/scripts/run_early_stopping_equal_budget.py" \
      --dataset "$ROOT/data/uci_har_shifted.npz" \
      --n-total 600 \
      --time-bins 16 \
      --q-max 64 \
      --seed "$seed" \
      --tol 0.005 \
      --min-q 10 \
      --n-boot 1000 \
      --baseline both \
      --model-cache "$ROOT/models/har_extratrees_seed${seed}.pkl" \
      --out "$OUT_ROOT/seed_${seed}"
  done

  "$PY" "$ROOT/scripts/aggregate_early_stop_seeds.py" \
    --input-dir "$OUT_ROOT" \
    --out-csv "$OUT_ROOT/all_seeds.csv" \
    --out-summary "$OUT_ROOT/all_seeds_summary.csv"
  echo "[finish] $(date -Is) q1_har_early_stop_seeds"
} 2>&1 | tee "$LOG_DIR/run.log"
