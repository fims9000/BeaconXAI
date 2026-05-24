#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lebedeffson/Code/BeaconXAI"
PY="/home/lebedeffson/Code/venv/bin/python"
OUT_ROOT="$ROOT/outputs_composite/q1_pamap2_logit_q12_seeds"
MODEL="$ROOT/models/pamap2_extratrees_seed1.pkl"
SEED_MAX="${SEED_MAX:-10}"
N_BOOT="${N_BOOT:-1000}"

mkdir -p "$OUT_ROOT/logs"

{
  echo "[start] $(date -Is) q1_pamap2_logit_q12 seed_max=${SEED_MAX} n_boot=${N_BOOT}"
  echo "[model] ${MODEL}"
  for seed in $(seq 1 "$SEED_MAX"); do
    out="$OUT_ROOT/seed_${seed}"
    if [ -f "$out/early_stop_vs_uniform_equal_budget.csv" ]; then
      echo "[skip] seed=${seed} exists"
      continue
    fi
    echo "[run] seed=${seed} $(date -Is)"
    "$PY" "$ROOT/scripts/run_early_stopping_equal_budget.py" \
      --dataset "$ROOT/data/pamap2_acc9_w200s100_p095.npz" \
      --n-total 600 \
      --time-bins 12 \
      --q-max 12 \
      --seed "$seed" \
      --tol 0.0 \
      --min-q 12 \
      --n-boot "$N_BOOT" \
      --baseline fixed_uniform \
      --policy-train-mode prefix_mix \
      --policy-prefix-list 5,8,10,12,16,24,32,64 \
      --policy-model logit \
      --order-mode adaptive \
      --model-cache "$MODEL" \
      --out "$out"
  done
  "$PY" "$ROOT/scripts/aggregate_early_stop_seeds.py" \
    --input-dir "$OUT_ROOT" \
    --out-csv "$OUT_ROOT/all_seeds.csv" \
    --out-summary "$OUT_ROOT/all_seeds_summary.csv"
  echo "[finish] $(date -Is) q1_pamap2_logit_q12"
} 2>&1 | tee "$OUT_ROOT/logs/run.log"
