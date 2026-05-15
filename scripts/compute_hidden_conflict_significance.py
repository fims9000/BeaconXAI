#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    obs = float(np.mean(a) - np.mean(b))
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(a[idx]) - np.mean(b[idx]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2.0 * min(float(np.mean(boots <= 0.0)), float(np.mean(boots >= 0.0)))
    return obs, float(lo), float(hi), float(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired bootstrap significance for Hidden Conflict table")
    p.add_argument("--per-sample", default="outputs_composite/har_hidden_conflict_localization_per_sample.csv")
    p.add_argument("--method-a", default="beacon_adaptive")
    p.add_argument("--method-b", default="uniform_occlusion")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs_composite/table8_significance.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.per_sample)

    need_cols = {"sample_index_eval", "method", "is_correct", "hit3", "hit5", "rank_true"}
    miss = need_cols.difference(df.columns)
    if miss:
        raise ValueError(f"missing columns: {sorted(miss)}")

    a = df[df["method"] == args.method_a].set_index("sample_index_eval")
    b = df[df["method"] == args.method_b].set_index("sample_index_eval")
    common = sorted(set(a.index).intersection(set(b.index)))
    if not common:
        raise RuntimeError("No overlapping sample_index_eval between methods")

    a = a.loc[common]
    b = b.loc[common]

    metrics = {
        "loc@1": (a["is_correct"].to_numpy(dtype=np.float64), b["is_correct"].to_numpy(dtype=np.float64)),
        "hit@3": (a["hit3"].to_numpy(dtype=np.float64), b["hit3"].to_numpy(dtype=np.float64)),
        "hit@5": (a["hit5"].to_numpy(dtype=np.float64), b["hit5"].to_numpy(dtype=np.float64)),
        "mrr": (1.0 / a["rank_true"].to_numpy(dtype=np.float64), 1.0 / b["rank_true"].to_numpy(dtype=np.float64)),
        "mean_rank": (-a["rank_true"].to_numpy(dtype=np.float64), -b["rank_true"].to_numpy(dtype=np.float64)),
    }

    rows: list[dict[str, float | str | int]] = []
    for name, (xa, xb) in metrics.items():
        d, lo, hi, p = paired_bootstrap_delta(xa, xb, n_boot=args.n_bootstrap, seed=args.seed + len(name) * 101)
        rows.append(
            {
                "metric": name,
                "method_a": args.method_a,
                "method_b": args.method_b,
                "n_eval": int(len(xa)),
                "mean_a": float(np.mean(xa)),
                "mean_b": float(np.mean(xb)),
                "delta_a_minus_b": d,
                "ci_low": lo,
                "ci_high": hi,
                "p_value_two_sided": p,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
