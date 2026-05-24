#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lebedeffson/Code/BeaconXAI"
PY="/home/lebedeffson/Code/venv/bin/python"

run_dataset() {
  local name="$1"
  local dataset="$2"
  local bins="$3"
  local model="$4"
  local out_root="$ROOT/outputs_composite/q1_${name}_frontier_short_seed1"
  local log_dir="$out_root/logs"
  mkdir -p "$log_dir"

  {
    echo "[start] $(date -Is) ${name}"
    for q in 10 12 16 24 32 64; do
      echo "[${name} fixed-q q=${q}] $(date -Is)"
      "$PY" "$ROOT/scripts/run_early_stopping_equal_budget.py" \
        --dataset "$dataset" \
        --n-total 600 \
        --time-bins "$bins" \
        --q-max "$q" \
        --seed 1 \
        --tol 0.0 \
        --min-q "$q" \
        --n-boot 500 \
        --baseline both \
        --policy-train-mode prefix_mix \
        --policy-prefix-list 5,8,10,12,16,24,32,64 \
        --model-cache "$ROOT/models/${model}_seed1.pkl" \
        --out "$out_root/fixed_q${q}"
    done

    echo "[${name} early min_q=10 tol=0.005] $(date -Is)"
    "$PY" "$ROOT/scripts/run_early_stopping_equal_budget.py" \
      --dataset "$dataset" \
      --n-total 600 \
      --time-bins "$bins" \
      --q-max 64 \
      --seed 1 \
      --tol 0.005 \
      --min-q 10 \
      --n-boot 500 \
      --baseline both \
      --policy-train-mode prefix_mix \
      --policy-prefix-list 5,8,10,12,16,24,32,64 \
      --model-cache "$ROOT/models/${model}_seed1.pkl" \
      --out "$out_root/early_minq10_tol0.005"
    echo "[finish] $(date -Is) ${name}"
  } 2>&1 | tee "$log_dir/run.log"
}

run_dataset "pamap2" "$ROOT/data/pamap2_acc9_w200s100_p095.npz" 12 "pamap2_extratrees"
run_dataset "wisdm" "$ROOT/data/wisdm_phone_accel_gyro_w200s100_p90_windowrand42.npz" 12 "wisdm_extratrees"
