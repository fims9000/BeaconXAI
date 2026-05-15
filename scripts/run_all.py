#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repro entrypoint for final paper artifacts")
    p.add_argument("--python", default=sys.executable, help="Python interpreter path")
    p.add_argument("--skip-hidden", action="store_true")
    p.add_argument("--skip-portability", action="store_true")
    p.add_argument("--skip-significance", action="store_true")
    p.add_argument("--skip-resource-budget", action="store_true")
    p.add_argument("--n-profile", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = args.python

    Path("outputs_composite").mkdir(parents=True, exist_ok=True)

    if not args.skip_hidden:
        run(
            [
                py,
                "scripts/run_har_hidden_conflict_benchmark.py",
                "--out-summary",
                "outputs_composite/har_hidden_conflict_localization_table.csv",
                "--out-per-sample",
                "outputs_composite/har_hidden_conflict_localization_per_sample.csv",
            ]
        )

    if not args.skip_significance:
        run(
            [
                py,
                "scripts/compute_hidden_conflict_significance.py",
                "--per-sample",
                "outputs_composite/har_hidden_conflict_localization_per_sample.csv",
                "--out",
                "outputs_composite/table8_significance.csv",
            ]
        )

    if not args.skip_portability:
        run(
            [
                py,
                "scripts/measure_portability.py",
                "--n-profile",
                str(args.n_profile),
                "--warmup",
                str(args.warmup),
                "--out",
                "outputs_composite/edge_portability_profile.csv",
            ]
        )

    if not args.skip_resource_budget:
        run(
            [
                py,
                "scripts/estimate_resource_budget.py",
                "--profile-csv",
                "outputs_composite/edge_portability_profile.csv",
                "--out",
                "outputs_composite/edge_resource_budget_table.csv",
            ]
        )

    print("Done.")
    print("Artifacts:")
    print("- outputs_composite/har_hidden_conflict_localization_table.csv")
    print("- outputs_composite/table8_significance.csv")
    print("- outputs_composite/edge_portability_profile.csv")
    print("- outputs_composite/edge_resource_budget_table.csv")


if __name__ == "__main__":
    main()
