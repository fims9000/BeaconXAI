#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_table1(py: str, quick: bool) -> None:
    n_boot = "100" if quick else "2000"
    n_total = "300" if quick else "600"
    cmd = [
        py,
        "scripts/benchmark_beacon_vs_uniform.py",
        "--datasets",
        "har,pamap2,wisdm",
        "--budgets",
        "16,32,64",
        "--n-boot",
        n_boot,
        "--n-total",
        n_total,
        "--adaptive-v2",
        "--out-root",
        "outputs_composite/v12_beacon_vs_uniform_repro",
    ]
    _run(cmd)


def run_table2(py: str, quick: bool) -> None:
    n_boot = "200" if quick else "5000"
    out_root = "outputs_composite/v11_cross_dataset"
    cmd = [
        py,
        "scripts/run_cross_dataset_benchmark.py",
        "--config",
        "configs/experiments_v11_cross_dataset.json",
        "--out-root",
        out_root,
    ]
    _run(cmd)
    _run([py, "scripts/aggregate_v11_results.py", "--glob", "outputs_composite/v11_*", "--out", "outputs_composite/v11_full_summary.csv"])
    _run([py, "scripts/make_v11_summary_table.py", "--input-csv", "outputs_composite/v11_full_summary.csv", "--output", "artifacts/v11_full_summary.md", "--claim-md", "artifacts/claim_registry_v11.md"])
    # Note: n_boot is controlled by config; keep command deterministic.
    print(f"[info] table2 used config n_boot (quick={quick}, requested={n_boot})")


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce key paper tables")
    p.add_argument("--quick", action="store_true", help="Fast smoke mode")
    p.add_argument("--table", choices=["table1", "table2", "all"], default="all")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    if args.table in ("table1", "all"):
        run_table1(args.python, args.quick)
    if args.table in ("table2", "all"):
        run_table2(args.python, args.quick)


if __name__ == "__main__":
    main()
