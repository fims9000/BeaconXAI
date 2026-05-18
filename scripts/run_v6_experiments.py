#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BEACON v6 grid (Q x neutralizer) and evaluate policies")
    p.add_argument("--dataset", default="data/uci_har_shifted.npz")
    p.add_argument("--model", default="extratrees", choices=["extratrees", "histgbt", "cnn1d"])
    p.add_argument("--n-total", type=int, default=3000)
    p.add_argument("--q-list", default="16,32,64")
    p.add_argument("--neutralizers", default="interp,zero,channel_mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-out", default="outputs_composite/part2_extended_v6")
    p.add_argument("--skip-feature-run", action="store_true")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    q_list = [int(v.strip()) for v in args.q_list.split(",") if v.strip()]
    modes = [v.strip() for v in args.neutralizers.split(",") if v.strip()]

    base = Path(args.base_out)
    base.mkdir(parents=True, exist_ok=True)

    for q in q_list:
        for mode in modes:
            bdir = base / f"q{q}_{mode}"
            bdir.mkdir(parents=True, exist_ok=True)

            if not args.skip_feature_run:
                _run(
                    [
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
                        "--neutralizer-mode",
                        mode,
                        "--seed",
                        str(args.seed),
                        "--features-only",
                        "--out",
                        str(bdir),
                    ]
                )

            _run(
                [
                    sys.executable,
                    "scripts/evaluate_v6_policies.py",
                    "--bundle-dir",
                    str(bdir),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                ]
            )

    # aggregate
    rows = []
    brows = []
    costs = []
    for q in q_list:
        for mode in modes:
            bdir = base / f"q{q}_{mode}"
            p = bdir / "v6_policy_eval.csv"
            pb = bdir / "v6_bootstrap_deltas.csv"
            pc = bdir / "tinyxai_full_audit_cost.csv"
            if p.exists():
                rows.append(p)
            if pb.exists():
                brows.append(pb)
            if pc.exists():
                costs.append(pc)

    import pandas as pd

    if rows:
        pd.concat([pd.read_csv(p) for p in rows], ignore_index=True).to_csv(base / "beacon_vs_uniform_q_sweep.csv", index=False)
    if brows:
        pd.concat([pd.read_csv(p) for p in brows], ignore_index=True).to_csv(base / "bootstrap_deltas_v6.csv", index=False)
    if costs:
        pd.concat([pd.read_csv(p) for p in costs], ignore_index=True).to_csv(base / "tinyxai_full_audit_cost.csv", index=False)

    print(f"saved grid outputs to: {base}")


if __name__ == "__main__":
    main()