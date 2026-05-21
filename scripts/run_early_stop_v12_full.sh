#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lebedeffson/Code/BeaconXAI"
PY="/home/lebedeffson/Code/venv/bin/python"
OUT_ROOT="$ROOT/outputs_composite/early_stop_v12_full"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  local dataset="$2"
  local bins="$3"
  local qmax="$4"
  local ntotal="$5"
  local seed="$6"
  local nboot="$7"
  local tol="$8"
  local minq="$9"

  "$PY" "$ROOT/scripts/run_early_stopping_equal_budget.py" \
    --dataset "$dataset" \
    --n-total "$ntotal" \
    --time-bins "$bins" \
    --q-max "$qmax" \
    --seed "$seed" \
    --tol "$tol" \
    --min-q "$minq" \
    --n-boot "$nboot" \
    --out "$OUT_ROOT/$name"
}

{
  echo "[start] $(date -Is) early_stop_v12_full"
  run_one "har" "$ROOT/data/uci_har_shifted.npz" 16 64 600 42 1000 0.005 10
  echo "[done] $(date -Is) har"
  run_one "pamap2" "$ROOT/data/pamap2_acc9_w200s100_p095.npz" 12 64 600 42 1000 0.005 10
  echo "[done] $(date -Is) pamap2"
  run_one "wisdm" "$ROOT/data/wisdm_phone_accel_gyro_w200s100_p90_windowrand42.npz" 12 64 600 42 1000 0.005 10
  echo "[done] $(date -Is) wisdm"
  echo "[finish] $(date -Is) early_stop_v12_full"
} 2>&1 | tee "$LOG_DIR/run.log"
