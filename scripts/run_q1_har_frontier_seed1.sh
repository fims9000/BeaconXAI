#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lebedeffson/Code/BeaconXAI"
PY="/home/lebedeffson/Code/venv/bin/python"
OUT_ROOT="$ROOT/outputs_composite/q1_har_frontier_seed1"
MODEL="$ROOT/models/har_extratrees_seed1.pkl"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  local qmax="$2"
  local minq="$3"
  local tol="$4"

  "$PY" "$ROOT/scripts/run_early_stopping_equal_budget.py" \
    --dataset "$ROOT/data/uci_har_shifted.npz" \
    --n-total 600 \
    --time-bins 16 \
    --q-max "$qmax" \
    --seed 1 \
    --tol "$tol" \
    --min-q "$minq" \
    --n-boot 500 \
    --baseline both \
    --policy-train-mode prefix_mix \
    --policy-prefix-list 5,8,10,12,16,24,32,64 \
    --model-cache "$MODEL" \
    --out "$OUT_ROOT/$name"
}

{
  echo "[start] $(date -Is) q1_har_frontier_seed1"

  for q in 8 10 12 16 24 32 64; do
    echo "[fixed-q q=${q}] $(date -Is)"
    run_one "fixed_q${q}" "$q" "$q" 0.0
  done

  for minq in 5 10 15; do
    for tol in 0.001 0.005 0.01; do
      name="early_minq${minq}_tol${tol}"
      echo "[early min_q=${minq} tol=${tol}] $(date -Is)"
      run_one "$name" 64 "$minq" "$tol"
    done
  done

  echo "[finish] $(date -Is) q1_har_frontier_seed1"
} 2>&1 | tee "$LOG_DIR/run.log"
