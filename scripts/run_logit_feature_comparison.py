#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beaconxai.calibration import brier_score, calibration_slope, expected_calibration_error
from beaconxai.tan_policy import bootstrap_delta_auroc


def _best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    qs = np.linspace(0.05, 0.95, 181)
    thrs = np.quantile(s, qs)
    best_f1 = -1.0
    best_t = float(np.median(s))
    for t in thrs:
        pred = (s >= t).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def _metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) >= 2:
        auroc = float(roc_auc_score(y, prob))
        auprc = float(average_precision_score(y, prob))
    else:
        auroc = float("nan")
        auprc = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "ece": float(expected_calibration_error(y, prob, n_bins=10)),
        "brier": float(brier_score(y, prob)),
        "calibration_slope": float(calibration_slope(y, prob)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Logit BEACON vs Uniform feature comparison")
    p.add_argument("--beacon-features", default="outputs_composite/part2_extended_v2/audit_features_beacon_core.csv")
    p.add_argument("--uniform-features", default="outputs_composite/part2_extended_v2/audit_features_uniform.csv")
    p.add_argument("--split-manifest", default="outputs_composite/part2_extended_v2/split_manifest.json")
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/logit_beacon_vs_uniform.csv")
    p.add_argument("--out-bootstrap", default="outputs_composite/part2_extended_v2/logit_beacon_vs_uniform_bootstrap.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df_b = pd.read_csv(args.beacon_features)
    df_u = pd.read_csv(args.uniform_features)

    with Path(args.split_manifest).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    tr = np.asarray(manifest["train_ids"], dtype=np.int64)
    va = np.asarray(manifest["val_ids"], dtype=np.int64)
    te = np.asarray(manifest["test_ids"], dtype=np.int64)

    cols = [
        "m_neg",
        "M_B_minus",
        "M_B_plus",
        "r_B_minus",
        "CE_B",
        "rho_B_cost",
        "frag_drop",
        "top1_delta",
        "top3_sum_delta",
        "top3_mean_delta",
        "top3_conflict_count",
        "delta_entropy",
    ]

    # Backward compatibility for older feature dumps.
    if "delta_entropy" not in df_b.columns and "rank_entropy" in df_b.columns:
        df_b = df_b.copy()
        df_b["delta_entropy"] = df_b["rank_entropy"]
    if "delta_entropy" not in df_u.columns and "rank_entropy" in df_u.columns:
        df_u = df_u.copy()
        df_u["delta_entropy"] = df_u["rank_entropy"]

    df_b = df_b.set_index("sample_id").sort_index()
    df_u = df_u.set_index("sample_id").sort_index()

    y = df_b.loc[:, "is_hidden_conflict"].to_numpy(dtype=np.int64)
    xb = df_b.loc[:, cols].to_numpy(dtype=float)
    xu = df_u.loc[:, cols].to_numpy(dtype=float)

    clf_b = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    clf_u = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    clf_b.fit(xb[tr], y[tr])
    clf_u.fit(xu[tr], y[tr])

    p_b_val = clf_b.predict_proba(xb[va])[:, 1]
    p_u_val = clf_u.predict_proba(xu[va])[:, 1]
    t_b = _best_f1_threshold(y[va], p_b_val)
    t_u = _best_f1_threshold(y[va], p_u_val)

    p_b = clf_b.predict_proba(xb[te])[:, 1]
    p_u = clf_u.predict_proba(xu[te])[:, 1]
    pred_b = (p_b >= t_b).astype(np.int64)
    pred_u = (p_u >= t_u).astype(np.int64)

    mb = _metrics(y[te], p_b, pred_b)
    mu = _metrics(y[te], p_u, pred_u)

    rows = [
        {
            "features": "logit_beacon_features",
            "budget": 0.10,
            **mb,
            "threshold": t_b,
            "n_test": int(len(te)),
        },
        {
            "features": "logit_uniform_features",
            "budget": 0.10,
            **mu,
            "threshold": t_u,
            "n_test": int(len(te)),
        },
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    d_mean, d_lo, d_hi, pval = bootstrap_delta_auroc(y[te], p_b, p_u, n_boot=5000, seed=args.seed + 99)
    b_rows = [
        {
            "comparison": "logit_beacon_vs_logit_uniform",
            "delta_auroc": d_mean,
            "ci_low": d_lo,
            "ci_high": d_hi,
            "p_value": pval,
            "beacon_auroc": mb["auroc"],
            "uniform_auroc": mu["auroc"],
            "beacon_auprc": mb["auprc"],
            "uniform_auprc": mu["auprc"],
            "beacon_f1": mb["f1"],
            "uniform_f1": mu["f1"],
        }
    ]
    pd.DataFrame(b_rows).to_csv(args.out_bootstrap, index=False)
    print(f"saved: {out}")
    print(f"saved: {args.out_bootstrap}")


if __name__ == "__main__":
    main()
