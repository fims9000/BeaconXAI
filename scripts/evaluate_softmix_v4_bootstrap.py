#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from beaconxai.fuzzy_policy_v2 import eval_at_budget


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap comparison: soft_mix_v4 vs logit")
    p.add_argument("--beacon-features", default="outputs_composite/part2_extended_v2/audit_features_beacon_core.csv")
    p.add_argument("--split-manifest", default="outputs_composite/part2_extended_v2/split_manifest.json")
    p.add_argument("--fuzzy-v4-results", default="outputs_composite/part2_extended_v2/fuzzy_v4_results.csv")
    p.add_argument("--fuzzy-v4-lambda", default="outputs_composite/part2_extended_v2/fuzzy_v4_lambda_profile.csv")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/fuzzy_v4_bootstrap_deltas.csv")
    return p.parse_args()


def _bootstrap_delta_metric(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        try:
            da = metric_fn(yy, a[idx])
            db = metric_fn(yy, b[idx])
        except Exception:
            continue
        if np.isfinite(da) and np.isfinite(db):
            vals.append(float(da - db))
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    p = 2.0 * min(float(np.mean(arr < 0.0)), float(np.mean(arr > 0.0)))
    p = float(min(1.0, max(0.0, p)))
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(p)


def _f1_budget_metric(frac: float):
    def fn(y: np.ndarray, score: np.ndarray) -> float:
        _p, _r, f1 = eval_at_budget(y, score, frac)
        return float(f1)

    return fn


def _precision_budget_metric(frac: float):
    def fn(y: np.ndarray, score: np.ndarray) -> float:
        p, _r, _f1 = eval_at_budget(y, score, frac)
        return float(p)

    return fn


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.beacon_features).set_index("sample_id").sort_index()
    with Path(args.split_manifest).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    te = np.asarray(manifest["test_ids"], dtype=np.int64)

    y_all = df["is_hidden_conflict"].to_numpy(dtype=np.int64)

    lam = pd.read_csv(args.fuzzy_v4_lambda)
    lam_test = lam[lam["split"] == "test"].copy()
    lam_test = lam_test.sort_values("sample_id")

    # Align y by sample_id in lambda profile (same id space as split ids)
    sid = lam_test["sample_id"].to_numpy(dtype=np.int64)
    y = y_all[sid]

    s_f = lam_test["score_fuzzy"].to_numpy(dtype=float)
    s_l = lam_test["score_logit"].to_numpy(dtype=float)
    s_ad = lam_test["lambda"].to_numpy(dtype=float) * s_f + (1.0 - lam_test["lambda"].to_numpy(dtype=float)) * s_l

    fv4 = pd.read_csv(args.fuzzy_v4_results)
    row_fixed = fv4[(fv4["policy"] == "soft_mix_fixed_v4") & (fv4["budget"] == 0.10)].iloc[0]
    lam_fixed = float(row_fixed["lambda_fixed"])
    s_fx = lam_fixed * s_f + (1.0 - lam_fixed) * s_l

    comparisons = {
        "soft_mix_fixed_v4_vs_logit": (s_fx, s_l),
        "soft_mix_adaptive_v4_vs_logit": (s_ad, s_l),
        "soft_mix_adaptive_v4_vs_fixed_v4": (s_ad, s_fx),
    }

    metrics = {
        "delta_auroc": lambda yy, ss: float(roc_auc_score(yy, ss)) if len(np.unique(yy)) >= 2 else float("nan"),
        "delta_auprc": lambda yy, ss: float(average_precision_score(yy, ss)),
        "delta_f1_10": _f1_budget_metric(0.10),
        "delta_f1_20": _f1_budget_metric(0.20),
        "delta_precision_10": _precision_budget_metric(0.10),
    }

    rows = []
    for cname, (sa, sb) in comparisons.items():
        for mname, mfn in metrics.items():
            d, lo, hi, p = _bootstrap_delta_metric(y, sa, sb, mfn, n_boot=args.n_boot, seed=args.seed + hash((cname, mname)) % 100000)
            rows.append(
                {
                    "comparison": cname,
                    "metric": mname,
                    "delta": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p,
                    "n_test": int(len(y)),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
