#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate TAN/Fuzzy deltas from one v11 output root")
    p.add_argument("--in", dest="in_dir", required=True, help="Input root, e.g. outputs_composite/v11_uwave_only")
    p.add_argument("--out", required=True, help="Output csv path")
    return p.parse_args()


def extract_metric(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    row = df[df["metric"] == metric]
    if row.empty:
        return float("nan"), float("nan")
    return float(row.iloc[0]["delta"]), float(row.iloc[0]["p_value"])


def main() -> None:
    args = parse_args()
    root = Path(args.in_dir)
    rows: list[dict[str, object]] = []

    for tan_boot in sorted(root.glob("*/tan_improved_bootstrap.csv")):
        bundle = tan_boot.parent.name
        fuzzy_boot = tan_boot.parent / "fuzzy_improved_bootstrap.csv"
        if not fuzzy_boot.exists():
            continue

        d_tan = pd.read_csv(tan_boot)
        d_fz = pd.read_csv(fuzzy_boot)

        tan_d_auroc, tan_p_auroc = extract_metric(d_tan, "delta_auroc")
        tan_d_f1, tan_p_f1 = extract_metric(d_tan, "delta_f1_10")
        fz_d_auroc, fz_p_auroc = extract_metric(d_fz, "delta_auroc")
        fz_d_f1, fz_p_f1 = extract_metric(d_fz, "delta_f1_10")

        rows.append(
            {
                "bundle": bundle,
                "tan_d_auroc": tan_d_auroc,
                "tan_p_auroc": tan_p_auroc,
                "tan_d_f1_10": tan_d_f1,
                "tan_p_f1_10": tan_p_f1,
                "fuzzy_d_auroc": fz_d_auroc,
                "fuzzy_p_auroc": fz_p_auroc,
                "fuzzy_d_f1_10": fz_d_f1,
                "fuzzy_p_f1_10": fz_p_f1,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("bundle").to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
