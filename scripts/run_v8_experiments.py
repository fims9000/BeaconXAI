#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from beaconxai.datasets import load_npz_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BEACON v8 policy grid and aggregate manuscript artifacts")
    p.add_argument("--dataset", default="data/uci_har_shifted.npz")
    p.add_argument("--model", default="extratrees", choices=["extratrees", "histgbt", "cnn1d"])
    p.add_argument("--n-total", type=int, default=3000)
    p.add_argument("--time-bins-list", default="16")
    p.add_argument("--q-list", default="16,32,64")
    p.add_argument("--neutralizers", default="interp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--preselect-mode", choices=["none", "adaptive_v2"], default="none")
    p.add_argument("--base-out", default="outputs_composite/part2_extended_v8")
    p.add_argument("--skip-feature-run", action="store_true")
    p.add_argument("--skip-policy-eval", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _iter_grid(time_bins: list[int], q_values: list[int], neutralizers: list[str]):
    for tb in time_bins:
        for q in q_values:
            for mode in neutralizers:
                yield tb, q, mode


def _bundle_name(tb: int, q: int, mode: str) -> str:
    return f"tb{tb}_q{q}_{mode}"


def _estimate_n_components(npz_path: str, time_bins: int) -> int:
    _x_train, _y_train, x_test, _y_test = load_npz_dataset(npz_path)
    n_channels = int(x_test.shape[2])
    return int(n_channels * time_bins)


def main() -> None:
    args = parse_args()
    time_bins = [int(v.strip()) for v in args.time_bins_list.split(",") if v.strip()]
    q_values = [int(v.strip()) for v in args.q_list.split(",") if v.strip()]
    neutralizers = [v.strip() for v in args.neutralizers.split(",") if v.strip()]

    base = Path(args.base_out)
    base.mkdir(parents=True, exist_ok=True)

    for tb, q, mode in _iter_grid(time_bins, q_values, neutralizers):
        bdir = base / _bundle_name(tb, q, mode)
        bdir.mkdir(parents=True, exist_ok=True)

        if not args.skip_feature_run:
            cmd = [
                sys.executable,
                "scripts/run_part2_extended.py",
                "--dataset",
                args.dataset,
                "--model",
                args.model,
                "--n-total",
                str(args.n_total),
                "--q-max",
                str(q),
                "--time-bins",
                str(tb),
                "--neutralizer-mode",
                mode,
                "--seed",
                str(args.seed),
                "--features-only",
                "--out",
                str(bdir),
            ]
            if args.preselect_mode == "adaptive_v2":
                cmd.append("--adaptive-v2-preselect")
            _run(
                cmd
            )

        if not args.skip_policy_eval:
            beacon_file = "audit_features_beacon_core.csv"
            if args.preselect_mode == "adaptive_v2":
                beacon_file = "audit_features_adaptive_v2.csv"
            _run(
                [
                    sys.executable,
                    "scripts/evaluate_v6_policies.py",
                    "--bundle-dir",
                    str(bdir),
                    "--beacon-file",
                    beacon_file,
                    "--uniform-file",
                    "audit_features_uniform.csv",
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--n-boot",
                    str(args.n_boot),
                    "--out",
                    "v8_policy_eval.csv",
                    "--out-bootstrap",
                    "v8_bootstrap_deltas.csv",
                    "--out-cost",
                    "v8_tinyxai_full_audit_cost.csv",
                ]
            )

    eval_paths = sorted(base.glob("tb*/v8_policy_eval.csv"))
    boot_paths = sorted(base.glob("tb*/v8_bootstrap_deltas.csv"))
    cost_paths = sorted(base.glob("tb*/v8_tinyxai_full_audit_cost.csv"))
    manifest_paths = sorted(base.glob("tb*/split_manifest.json"))

    if eval_paths:
        pd.concat([pd.read_csv(p) for p in eval_paths], ignore_index=True).to_csv(
            base / "beacon_vs_uniform_q_sweep_v8.csv", index=False
        )
    if boot_paths:
        pd.concat([pd.read_csv(p) for p in boot_paths], ignore_index=True).to_csv(
            base / "bootstrap_deltas_v8.csv", index=False
        )
    if cost_paths:
        pd.concat([pd.read_csv(p) for p in cost_paths], ignore_index=True).to_csv(
            base / "tinyxai_full_audit_cost_v8.csv", index=False
        )

    budget_rows: list[dict[str, object]] = []
    for mp in manifest_paths:
        bname = mp.parent.name
        parts = bname.split("_")
        tb = int(parts[0].replace("tb", ""))
        q = int(parts[1].replace("q", ""))
        mode = parts[2]
        with mp.open("r", encoding="utf-8") as f:
            man = json.load(f)
        n_components = _estimate_n_components(args.dataset, tb)
        budget_rows.append(
            {
                "bundle": bname,
                "preselect_mode": args.preselect_mode,
                "dataset": str(man.get("dataset", args.dataset)),
                "model": str(man.get("model", args.model)),
                "time_bins": tb,
                "neutralizer_mode": mode,
                "q_max": q,
                "n_components": n_components,
                "q_over_m": float(q / max(1, n_components)),
                "train_n": int(len(man.get("train_ids", []))),
                "val_n": int(len(man.get("val_ids", []))),
                "test_n": int(len(man.get("test_ids", []))),
            }
        )
    if budget_rows:
        pd.DataFrame(budget_rows).sort_values(["time_bins", "q_max", "neutralizer_mode"]).to_csv(
            base / "har_component_budget_summary_v8.csv", index=False
        )

    claims = []
    if boot_paths:
        bdf = pd.concat([pd.read_csv(p) for p in boot_paths], ignore_index=True)
        det = bdf[
            (bdf["comparison"] == "logit_beacon_vs_uniform")
            & (bdf["metric"].isin(["delta_auroc", "delta_auprc", "delta_f1_10", "delta_f1_20"]))
        ].copy()
        for _, r in det.iterrows():
            claims.append(
                {
                    "block": "binary_detection",
                    "bundle": str(r["bundle"]),
                    "preselect_mode": args.preselect_mode,
                    "comparison": str(r["comparison"]),
                    "metric": str(r["metric"]),
                    "delta": float(r["delta"]),
                    "ci_low": float(r["ci_low"]),
                    "ci_high": float(r["ci_high"]),
                    "p_value": float(r["p_value"]),
                    "q1_signal": int(float(r["ci_low"]) > 0.0 and float(r["p_value"]) < 0.05),
                }
            )
    if claims:
        pd.DataFrame(claims).to_csv(base / "manuscript_claim_registry_v8.csv", index=False)

    print(f"saved grid outputs to: {base}", flush=True)


if __name__ == "__main__":
    main()
