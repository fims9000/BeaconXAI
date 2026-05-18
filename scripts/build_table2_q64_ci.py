#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _eval_f1_at_budget(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-s)[:k]
    yhat = np.zeros(n, dtype=np.int64)
    yhat[idx] = 1
    tp = int(np.sum((yhat == 1) & (y == 1)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return float(2.0 * p * r / max(1e-12, p + r))


def _bootstrap_ci(y: np.ndarray, s: np.ndarray, fn, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        ss = s[idx]
        try:
            v = fn(yy, ss)
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(float(v))
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Table2 absolute metric CIs for interp Q=64.")
    p.add_argument("--per-sample-csv", required=True)
    p.add_argument("--neutralizer", default="interp")
    p.add_argument("--q-max", type=int, default=64)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs_composite/part2_extended_v6/table2_q64_metric_ci.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.per_sample_csv)
    sub = df[(df["neutralizer_input"] == args.neutralizer) & (df["q_max"] == args.q_max)].copy()
    y = sub["label"].to_numpy(dtype=np.int64)
    sb = sub["score_beacon_panel"].to_numpy(dtype=float)
    su = sub["score_uniform_panel"].to_numpy(dtype=float)

    metrics = {
        "AUROC": lambda yy, ss: float(roc_auc_score(yy, ss)) if len(np.unique(yy)) >= 2 else float("nan"),
        "AUPRC": lambda yy, ss: float(average_precision_score(yy, ss)),
        "F1@10": lambda yy, ss: _eval_f1_at_budget(yy, ss, 0.10),
        "F1@20": lambda yy, ss: _eval_f1_at_budget(yy, ss, 0.20),
    }

    rows = []
    for method, score in [("beacon_panel", sb), ("uniform_panel", su)]:
        for mname, fn in metrics.items():
            val = fn(y, score)
            lo, hi = _bootstrap_ci(y, score, fn, n_boot=args.n_boot, seed=args.seed + abs(hash((method, mname))) % 100000)
            rows.append(
                {
                    "setting": f"{args.neutralizer}_q{args.q_max}",
                    "method": method,
                    "metric": mname,
                    "value": float(val),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_samples": int(len(y)),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

