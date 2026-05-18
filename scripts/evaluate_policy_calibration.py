#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from beaconxai.calibration import brier_score, calibration_curve_bins, calibration_slope, expected_calibration_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate calibration of policy scores")
    p.add_argument("--scores", default="outputs_composite/part2_extended_v2/policy_scores_long.csv")
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/policy_calibration.csv")
    p.add_argument("--out-bins", default="outputs_composite/part2_extended_v2/policy_calibration_bins.csv")
    p.add_argument("--n-bins", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.scores)
    rows = []
    bins_rows = []
    for name, g in df.groupby("policy"):
        y = g["y_true"].to_numpy(dtype=int)
        s = g["score"].to_numpy(dtype=float)
        rows.append(
            {
                "policy": name,
                "n": int(len(g)),
                "ece": float(expected_calibration_error(y, s, n_bins=args.n_bins)),
                "brier": float(brier_score(y, s)),
                "calibration_slope": float(calibration_slope(y, s)),
            }
        )
        for b in calibration_curve_bins(y, s, n_bins=args.n_bins):
            bins_rows.append({"policy": name, **b})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("ece").to_csv(out, index=False)
    pd.DataFrame(bins_rows).to_csv(args.out_bins, index=False)
    print(f"saved: {out}")
    print(f"saved: {args.out_bins}")


if __name__ == "__main__":
    main()
