#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLS = [
    "delta_auroc_vs_uniform_early_stop",
    "delta_auprc_vs_uniform_early_stop",
    "delta_f1_10_vs_uniform_early_stop",
    "q_mean_early",
    "q_mean_uniform_early_stop",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate early-stop runs over seed_* directories.")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-summary", default="")
    return p.parse_args()


def _seed_from_name(path: Path) -> int:
    text = path.name.replace("seed_", "")
    try:
        return int(text)
    except ValueError:
        return -1


def _ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    rows = []
    for run_dir in sorted(root.glob("seed_*"), key=_seed_from_name):
        csv_path = run_dir / "early_stop_vs_uniform_equal_budget.csv"
        if not csv_path.exists():
            continue
        row = pd.read_csv(csv_path).iloc[0].to_dict()
        row["seed"] = _seed_from_name(run_dir)
        row["run_dir"] = str(run_dir)
        rows.append(row)

    if not rows:
        raise SystemExit(f"No seed_* results found in {root}")

    df = pd.DataFrame(rows).sort_values("seed")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary_rows = []
    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        lo, hi = _ci(vals)
        summary_rows.append(
            {
                "metric": col,
                "n": int(np.isfinite(vals).sum()),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci_low_seed_quantile": lo,
                "ci_high_seed_quantile": hi,
                "positive_fraction": float(np.mean(vals > 0.0)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }
        )

    out_summary = Path(args.out_summary) if args.out_summary else out_csv.with_name(out_csv.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print(f"saved: {out_csv}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()
