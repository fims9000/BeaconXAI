#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lebedeffson/Code/BeaconXAI"
PY="/home/lebedeffson/Code/venv/bin/python"
SEED_MAX="${SEED_MAX:-5}"
N_BOOT="${N_BOOT:-1000}"

run_dataset() {
  local name="$1"
  local dataset="$2"
  local bins="$3"
  local model="$4"
  local out_root="$ROOT/outputs_composite/q1_${name}_divstop_q12_lam0.5_short"
  local log_dir="$out_root/logs"
  mkdir -p "$log_dir"

  {
    echo "[start] $(date -Is) dataset=${name} seed_max=${SEED_MAX} n_boot=${N_BOOT}"
    for seed in $(seq 1 "$SEED_MAX"); do
      local out="$out_root/seed_${seed}"
      if [ -f "$out/early_stop_vs_uniform_equal_budget.csv" ]; then
        echo "[skip] ${name} seed=${seed} exists"
        continue
      fi
      echo "[run] ${name} seed=${seed} $(date -Is)"
      "$PY" "$ROOT/scripts/run_early_stopping_equal_budget.py" \
        --dataset "$dataset" \
        --n-total 600 \
        --time-bins "$bins" \
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
        --model-cache "$ROOT/models/${model}_seed1.pkl" \
        --out "$out"
    done
    "$PY" "$ROOT/scripts/aggregate_early_stop_seeds.py" \
      --input-dir "$out_root" \
      --out-csv "$out_root/all_seeds.csv" \
      --out-summary "$out_root/all_seeds_summary.csv"
    echo "[finish] $(date -Is) dataset=${name}"
  } 2>&1 | tee "$log_dir/run.log"
}

run_dataset "pamap2" "$ROOT/data/pamap2_acc9_w200s100_p095.npz" 12 "pamap2_extratrees"
run_dataset "wisdm" "$ROOT/data/wisdm_phone_accel_gyro_w200s100_p90_windowrand42.npz" 12 "wisdm_extratrees"
