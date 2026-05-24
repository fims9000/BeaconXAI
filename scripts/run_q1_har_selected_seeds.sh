#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lebedeffson/Code/BeaconXAI"
PY="/home/lebedeffson/Code/venv/bin/python"
OUT_ROOT="$ROOT/outputs_composite/q1_har_selected_seeds"
MODEL="$ROOT/models/har_extratrees_seed1.pkl"
LOG_DIR="$OUT_ROOT/logs"
SEED_MAX="${SEED_MAX:-10}"
N_BOOT="${N_BOOT:-1000}"

mkdir -p "$LOG_DIR"

run_one() {
  local out_dir="$1"
  local seed="$2"
  local qmax="$3"
  local minq="$4"
  local tol="$5"

  "$PY" "$ROOT/scripts/run_early_stopping_equal_budget.py" \
    --dataset "$ROOT/data/uci_har_shifted.npz" \
    --n-total 600 \
    --time-bins 16 \
    --q-max "$qmax" \
    --seed "$seed" \
    --tol "$tol" \
    --min-q "$minq" \
    --n-boot "$N_BOOT" \
    --baseline both \
    --policy-train-mode prefix_mix \
    --policy-prefix-list 5,8,10,12,16,24,32,64 \
    --model-cache "$MODEL" \
    --out "$out_dir/seed_${seed}"
}

aggregate_one() {
  local mode_dir="$1"
  "$PY" "$ROOT/scripts/aggregate_early_stop_seeds.py" \
    --input-dir "$mode_dir" \
    --out-csv "$mode_dir/all_seeds.csv" \
    --out-summary "$mode_dir/all_seeds_summary.csv"
}

{
  echo "[start] $(date -Is) q1_har_selected_seeds seed_max=${SEED_MAX} n_boot=${N_BOOT}"
  echo "[model] ${MODEL}"

  for seed in $(seq 1 "$SEED_MAX"); do
    echo "[fixed_q12 seed=${seed}] $(date -Is)"
    run_one "$OUT_ROOT/fixed_q12" "$seed" 12 12 0.0

    echo "[early_minq10_tol0.001 seed=${seed}] $(date -Is)"
    run_one "$OUT_ROOT/early_minq10_tol0.001" "$seed" 64 10 0.001
  done

  echo "[aggregate fixed_q12] $(date -Is)"
  aggregate_one "$OUT_ROOT/fixed_q12"

  echo "[aggregate early_minq10_tol0.001] $(date -Is)"
  aggregate_one "$OUT_ROOT/early_minq10_tol0.001"

  echo "[finish] $(date -Is) q1_har_selected_seeds"
} 2>&1 | tee "$LOG_DIR/run.log"
