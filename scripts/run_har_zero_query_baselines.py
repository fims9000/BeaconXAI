#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract zero-query HAR baselines table")
    p.add_argument("--in-summary", default="outputs_composite/har_sensor_fault_localization_table.csv")
    p.add_argument("--out", default="outputs_composite/har_zero_query_baselines_table.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.in_summary)
    keep = ["random", "amplitude_heuristic", "energy_heuristic", "variance_heuristic"]
    out = df[df["method"].isin(keep)].copy()
    out = out[["dataset", "model", "q_max", "time_bins", "n_components", "method", "calls", "loc@1", "hit@3", "hit@5", "mrr"]]
    out = out.sort_values(["calls", "method"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
