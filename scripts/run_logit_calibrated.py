#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PANEL_COLS = [
    "m_neg",
    "M_B_minus",
    "r_B_minus",
    "CE_B",
    "rho_B_cost",
    "frag_drop",
    "top1_delta",
    "top3_sum_delta",
    "top3_conflict_count",
    "margin_entropy",
    "mean_conflict",
    "var_conflict_proxy",
    "frac_conflict_top3",
    "fragility_gap",
    "ce_density",
    "var_conflict",
    "conflict_connectivity",
    "delta_frag_proxy",
    "r_cf",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate calibrated logistic panel on existing bundle")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--out-results", default="logit_calibrated_results.csv")
    p.add_argument("--out-bootstrap", default="logit_calibrated_bootstrap.csv")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cost-fn", type=float, default=5.0)
    p.add_argument("--cost-fp", type=float, default=1.0)
    return p.parse_args()


def _f1_at_frac(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-s)[:k]
    pred = np.zeros(n, dtype=np.int64)
    pred[idx] = 1
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    if tp == 0:
        return 0.0
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return float(2 * p * r / max(p + r, 1e-12))


def _ece(y: np.ndarray, s: np.ndarray, n_bins: int = 15) -> float:
    y = y.astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(s, bins) - 1
    e = 0.0
    n = len(y)
    for b in range(n_bins):
        m = ids == b
        if not np.any(m):
            continue
        conf = float(np.mean(s[m]))
        acc = float(np.mean(y[m]))
        e += (np.sum(m) / n) * abs(acc - conf)
    return float(e)


def _expected_cost(y: np.ndarray, s: np.ndarray, th: float, c_fn: float, c_fp: float) -> float:
    pred = (s >= th).astype(np.int64)
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return float(c_fn * fn + c_fp * fp)


def _best_cost_threshold(y: np.ndarray, s: np.ndarray, c_fn: float, c_fp: float) -> tuple[float, float]:
    ths = np.linspace(0.0, 1.0, 201)
    costs = [_expected_cost(y, s, th, c_fn, c_fp) for th in ths]
    i = int(np.argmin(costs))
    return float(ths[i]), float(costs[i])


def _bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d[i] = float(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    mean = float(np.mean(d))
    lo = float(np.quantile(d, 0.025))
    hi = float(np.quantile(d, 0.975))
    p = float(min(1.0, 2.0 * min(np.mean(d <= 0.0), np.mean(d >= 0.0))))
    return mean, lo, hi, p


def main() -> None:
    args = parse_args()
    bdir = Path(args.bundle_dir)

    df = pd.read_csv(bdir / "audit_features_beacon_core.csv").sort_values("sample_id")
    split = json.loads((bdir / "split_manifest.json").read_text(encoding="utf-8"))
    tr = np.asarray(split["train_ids"], dtype=np.int64)
    va = np.asarray(split["val_ids"], dtype=np.int64)
    te = np.asarray(split["test_ids"], dtype=np.int64)

    cols = [c for c in PANEL_COLS if c in df.columns]
    X = df[cols].to_numpy(dtype=float)
    y = df["is_hidden_conflict"].to_numpy(dtype=np.int64)

    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    base.fit(X[tr], y[tr])
    s_base_val = base.predict_proba(X[va])[:, 1]
    s_base = base.predict_proba(X[te])[:, 1]

    calib = CalibratedClassifierCV(base, method=args.method, cv="prefit")
    calib.fit(X[va], y[va])
    s_cal = calib.predict_proba(X[te])[:, 1]

    th_base, c_base = _best_cost_threshold(y[va], s_base_val, args.cost_fn, args.cost_fp)
    th_cal, c_cal = _best_cost_threshold(y[va], calib.predict_proba(X[va])[:, 1], args.cost_fn, args.cost_fp)
    cost_test_base = _expected_cost(y[te], s_base, th_base, args.cost_fn, args.cost_fp)
    cost_test_cal = _expected_cost(y[te], s_cal, th_cal, args.cost_fn, args.cost_fp)

    res = pd.DataFrame(
        [
            {
                "policy": "logit_panel_raw",
                "auroc": float(roc_auc_score(y[te], s_base)),
                "auprc": float(average_precision_score(y[te], s_base)),
                "f1_10": float(_f1_at_frac(y[te], s_base, 0.10)),
                "f1_20": float(_f1_at_frac(y[te], s_base, 0.20)),
                "ece": float(_ece(y[te], s_base)),
                "cost_threshold": th_base,
                "expected_cost_test": cost_test_base,
            },
            {
                "policy": f"logit_panel_calibrated_{args.method}",
                "auroc": float(roc_auc_score(y[te], s_cal)),
                "auprc": float(average_precision_score(y[te], s_cal)),
                "f1_10": float(_f1_at_frac(y[te], s_cal, 0.10)),
                "f1_20": float(_f1_at_frac(y[te], s_cal, 0.20)),
                "ece": float(_ece(y[te], s_cal)),
                "cost_threshold": th_cal,
                "expected_cost_test": cost_test_cal,
            },
        ]
    )
    res.to_csv(bdir / args.out_results, index=False)

    rows = []
    metrics = [
        ("delta_auroc", roc_auc_score),
        ("delta_auprc", average_precision_score),
        ("delta_f1_10", lambda yt, st: _f1_at_frac(yt, st, 0.10)),
        ("delta_f1_20", lambda yt, st: _f1_at_frac(yt, st, 0.20)),
        ("delta_ece", _ece),
    ]
    for i, (mname, fn) in enumerate(metrics):
        d, lo, hi, p = _bootstrap_delta(y[te], s_cal, s_base, fn, args.n_boot, args.seed + 100 + i)
        rows.append(
            {
                "bundle": bdir.name,
                "comparison": f"logit_calibrated_{args.method}_vs_raw",
                "metric": mname,
                "delta": d,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
            }
        )
    # Cost delta (calibrated better -> negative delta)
    rows.append(
        {
            "bundle": bdir.name,
            "comparison": f"logit_calibrated_{args.method}_vs_raw",
            "metric": "delta_expected_cost",
            "delta": float(cost_test_cal - cost_test_base),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
    )
    pd.DataFrame(rows).to_csv(bdir / args.out_bootstrap, index=False)
    print(f"saved: {bdir / args.out_results}")
    print(f"saved: {bdir / args.out_bootstrap}")


if __name__ == "__main__":
    main()
