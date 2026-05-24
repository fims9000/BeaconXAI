#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lebedeffson/Code/BeaconXAI"
PY="/home/lebedeffson/Code/venv/bin/python"
OUT_ROOT="$ROOT/outputs_composite/part1_har_adaptive_v2_q16_seeds"
SEED_MAX="${SEED_MAX:-30}"
BOOT="${BOOT:-1000}"

mkdir -p "$OUT_ROOT/logs"

{
  echo "[start] $(date -Is) part1_har_adaptive_v2_q16 seed_max=${SEED_MAX} boot=${BOOT}"
  for seed in $(seq 1 "$SEED_MAX"); do
    out_dir="$OUT_ROOT/seed_${seed}"
    mkdir -p "$out_dir"
    if [ -f "$out_dir/results.csv" ]; then
      echo "[skip] seed=${seed} exists"
      continue
    fi
    echo "[run] seed=${seed} $(date -Is)"
    "$PY" "$ROOT/scripts/run_adaptive_v2_localization.py" \
      --npz-path "$ROOT/data/uci_har_shifted.npz" \
      --dataset-name har \
      --model extratrees \
      --seed "$seed" \
      --max-test 600 \
      --conflict-ratio 0.5 \
      --q-values 16 \
      --neutralizer interp \
      --group-mode per_channel \
      --time-bins 16 \
      --bootstrap "$BOOT" \
      --out-results "$out_dir/results.csv" \
      --out-bootstrap "$out_dir/bootstrap.csv" \
      --out-claims "$out_dir/claims.csv"
  done
  echo "[finish] $(date -Is) part1_har_adaptive_v2_q16"
} 2>&1 | tee "$OUT_ROOT/logs/run.log"
