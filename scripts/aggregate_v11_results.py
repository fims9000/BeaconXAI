#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


_BUNDLE_RE = re.compile(r"^(?P<dataset>.+?)_tb(?P<tb>\d+)_q(?P<q>\d+)_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate all v11 result roots into one CSV")
    p.add_argument("--glob", default="outputs_composite/v11_*", help="Glob for v11 roots")
    p.add_argument("--out", default="outputs_composite/v11_full_summary.csv")
    return p.parse_args()


def _extract(df: pd.DataFrame, metric: str) -> tuple[float, float, float, float]:
    row = df[df["metric"] == metric]
    if row.empty:
        return (float("nan"),) * 4
    r = row.iloc[0]
    return float(r["delta"]), float(r["p_value"]), float(r["ci_low"]), float(r["ci_high"])


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    roots = sorted(Path(".").glob(args.glob))

    for root in roots:
        for tan_boot in sorted(root.glob("*/tan_improved_bootstrap.csv")):
            bundle = tan_boot.parent.name
            fuzzy_boot = tan_boot.parent / "fuzzy_improved_bootstrap.csv"
            if not fuzzy_boot.exists():
                continue

            m = _BUNDLE_RE.match(bundle)
            dataset = m.group("dataset") if m else "unknown"
            tb = int(m.group("tb")) if m else -1
            q = int(m.group("q")) if m else -1

            d_tan = pd.read_csv(tan_boot)
            d_fz = pd.read_csv(fuzzy_boot)

            tan_d_auroc, tan_p_auroc, tan_ci_l_auroc, tan_ci_h_auroc = _extract(d_tan, "delta_auroc")
            tan_d_f1, tan_p_f1, tan_ci_l_f1, tan_ci_h_f1 = _extract(d_tan, "delta_f1_10")
            fz_d_auroc, fz_p_auroc, fz_ci_l_auroc, fz_ci_h_auroc = _extract(d_fz, "delta_auroc")
            fz_d_f1, fz_p_f1, fz_ci_l_f1, fz_ci_h_f1 = _extract(d_fz, "delta_f1_10")

            rows.append(
                {
                    "root": root.name,
                    "dataset": dataset,
                    "bundle": bundle,
                    "time_bins": tb,
                    "q": q,
                    "tan_d_auroc": tan_d_auroc,
                    "tan_p_auroc": tan_p_auroc,
                    "tan_ci_low_auroc": tan_ci_l_auroc,
                    "tan_ci_high_auroc": tan_ci_h_auroc,
                    "tan_d_f1_10": tan_d_f1,
                    "tan_p_f1_10": tan_p_f1,
                    "tan_ci_low_f1_10": tan_ci_l_f1,
                    "tan_ci_high_f1_10": tan_ci_h_f1,
                    "fuzzy_d_auroc": fz_d_auroc,
                    "fuzzy_p_auroc": fz_p_auroc,
                    "fuzzy_ci_low_auroc": fz_ci_l_auroc,
                    "fuzzy_ci_high_auroc": fz_ci_h_auroc,
                    "fuzzy_d_f1_10": fz_d_f1,
                    "fuzzy_p_f1_10": fz_p_f1,
                    "fuzzy_ci_low_f1_10": fz_ci_l_f1,
                    "fuzzy_ci_high_f1_10": fz_ci_h_f1,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "time_bins", "q", "bundle"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
