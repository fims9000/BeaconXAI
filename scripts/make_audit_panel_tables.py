#!/usr/bin/env python3
"""Build audit-panel analysis tables from per-sample risk/local metrics logs.

Outputs:
- outputs_composite/audit_error_type_specialization.csv
- outputs_composite/audit_incremental_feature_value.csv
- outputs_composite/audit_panel_vs_scalar.csv

Assumptions (explicit, for draft stage):
- `m_neg` is taken from `negative_margin` risk rows (higher -> riskier).
- `M_B_minus` is proxied by `counter_evidence_gain`.
- `CE_B` is proxied by `necessity`.
- `r_B_minus` is proxied by `rho_b`.
- `rho_B_cost` is taken directly from local metrics.
- `frag_drop` is a soft fragility proxy `max(0, -sufficiency_margin)`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RNG = np.random.default_rng(42)


def _trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    if b <= a:
        b = a + 1e-12
    if c <= b:
        c = b + 1e-12
    y = np.zeros_like(x, dtype=float)
    left = (x >= a) & (x <= b)
    right = (x >= b) & (x <= c)
    y[left] = (x[left] - a) / (b - a)
    y[right] = (c - x[right]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def _trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    if b < a:
        b = a
    if c < b:
        c = b
    if d < c:
        d = c
    y = np.zeros_like(x, dtype=float)
    if b > a:
        up = (x >= a) & (x < b)
        y[up] = (x[up] - a) / (b - a)
    core = (x >= b) & (x <= c)
    y[core] = 1.0
    if d > c:
        down = (x > c) & (x <= d)
        y[down] = (d - x[down]) / (d - c)
    return np.clip(y, 0.0, 1.0)


def _fuzzy_memberships(values: np.ndarray) -> dict[str, np.ndarray]:
    q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, 1e-6)
    lo = vmin - 0.05 * span
    hi = vmax + 0.05 * span
    return {
        "low": _trapmf(values, lo, lo, q25, q50),
        "med": _trimf(values, q10, q50, q90),
        "high": _trapmf(values, q50, q75, hi, hi),
    }


def make_fuzzy_panel_score(df: pd.DataFrame) -> np.ndarray:
    # Conflict/fragility are aggregated panel axes; margin is direct scalar uncertainty.
    conf = (
        _z(df["M_B_minus"].to_numpy(dtype=float))
        + _z(df["CE_B"].to_numpy(dtype=float))
        + _z(df["r_B_minus"].to_numpy(dtype=float))
    ) / 3.0
    frag = (_z(df["rho_B_cost"].to_numpy(dtype=float)) + _z(df["frag_drop"].to_numpy(dtype=float))) / 2.0
    mneg = df["m_neg"].to_numpy(dtype=float)

    mu_conf = _fuzzy_memberships(conf)
    mu_frag = _fuzzy_memberships(frag)
    mu_margin = _fuzzy_memberships(mneg)

    u = np.linspace(0.0, 1.0, 201)
    risk_terms = {
        "low": _trapmf(u, 0.0, 0.0, 0.20, 0.40),
        "med": _trimf(u, 0.25, 0.50, 0.72),
        "high": _trimf(u, 0.60, 0.78, 0.92),
        "critical": _trapmf(u, 0.85, 0.95, 1.0, 1.0),
    }

    out = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        # Rules reflect the paper narrative: conflict x fragility x margin interactions.
        r_med_1 = min(mu_conf["high"][i], mu_frag["low"][i])  # check sensor conflict
        r_high_1 = min(mu_conf["low"][i], mu_frag["high"][i])  # repeat measurement
        r_crit = min(mu_conf["high"][i], mu_frag["high"][i], mu_margin["high"][i])  # escalate
        r_high_2 = min(mu_conf["med"][i], mu_frag["med"][i], mu_margin["high"][i])
        r_low = min(mu_conf["low"][i], mu_frag["low"][i], mu_margin["low"][i])  # accept
        r_high_3 = min(mu_conf["high"][i], mu_frag["low"][i], mu_margin["high"][i])
        r_med_2 = min(mu_conf["med"][i], mu_margin["low"][i])

        agg = np.zeros_like(u, dtype=float)
        agg = np.maximum(agg, np.minimum(r_low, risk_terms["low"]))
        agg = np.maximum(agg, np.minimum(max(r_med_1, r_med_2), risk_terms["med"]))
        agg = np.maximum(agg, np.minimum(max(r_high_1, r_high_2, r_high_3), risk_terms["high"]))
        agg = np.maximum(agg, np.minimum(r_crit, risk_terms["critical"]))

        den = float(np.sum(agg))
        out[i] = float(np.sum(u * agg) / den) if den > 1e-12 else 0.0
    return out


def p_at_fraction(y_true: np.ndarray, score: np.ndarray, frac: float = 0.10) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-score)[:k]
    return float(np.mean(y_true[idx]))


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    n = len(y_true)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        v = metric_fn(y_true[idx], score[idx])
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan"), float("nan")
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi


def cv_logit_scores(X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    pred = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs"),
        )
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
    return pred


def build_panel_df(risk_rows: Path, local_metrics: Path, q: int, method: str) -> pd.DataFrame:
    rr = pd.read_csv(risk_rows)
    lm = pd.read_csv(local_metrics)

    base = rr[(rr["method"] == "negative_margin") & (rr["q_max"] == 0)][
        ["sample_id", "is_error", "risk_score"]
    ].rename(columns={"risk_score": "m_neg"})

    m = lm[(lm["method"] == method) & (lm["q_max"] == q)][
        [
            "sample_id",
            "sufficiency_margin",
            "necessity",
            "counter_evidence_gain",
            "rho_b",
            "rho_b_cost",
            "censored",
        ]
    ].copy()

    m["frag_drop"] = np.maximum(0.0, -m["sufficiency_margin"].astype(float))
    m = m.rename(
        columns={
            "counter_evidence_gain": "M_B_minus",
            "necessity": "CE_B",
            "rho_b": "r_B_minus",
            "rho_b_cost": "rho_B_cost",
        }
    )

    df = base.merge(m, on="sample_id", how="inner")
    df = df.dropna(
        subset=["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop", "is_error"]
    ).copy()
    df["is_error"] = df["is_error"].astype(int)
    return df


def define_error_types(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    err = df[df["is_error"] == 1]
    ok = df[df["is_error"] == 0]

    q_conf = float(err["m_neg"].quantile(0.25))
    q_lowc = float(err["m_neg"].quantile(0.75))
    q_ce = float(err["CE_B"].median())
    q_mass = float(err["M_B_minus"].median())
    q_rho_ok = float(ok["rho_B_cost"].quantile(0.25))
    q_frag_ok = float(ok["frag_drop"].median())

    tA = (df["is_error"] == 1) & (df["m_neg"] <= q_conf) & (df["CE_B"] >= q_ce)
    tB = (df["is_error"] == 1) & (df["m_neg"] >= q_lowc) & (df["M_B_minus"] >= q_mass)
    tC = (df["is_error"] == 0) & (df["rho_B_cost"] <= q_rho_ok) & (df["frag_drop"] >= q_frag_ok)
    tD = df["censored"].astype(int) == 1

    return {
        "A": tA.to_numpy().astype(int),
        "B": tB.to_numpy().astype(int),
        "C": tC.to_numpy().astype(int),
        "D": tD.to_numpy().astype(int),
    }


def make_error_type_table(df: pd.DataFrame, out_csv: Path, n_boot: int) -> None:
    features = ["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]
    types = define_error_types(df)
    rows = []

    for tname, y in types.items():
        n_pos = int(y.sum())
        for feat in features:
            s = df[feat].to_numpy(dtype=float)
            auc = safe_auc(y, s)
            p10 = p_at_fraction(y, s, frac=0.10)
            ci_auc = bootstrap_ci(safe_auc, y, s, n_boot=n_boot)
            ci_p10 = bootstrap_ci(lambda yy, ss: p_at_fraction(yy, ss, 0.10), y, s, n_boot=n_boot)
            rows.append(
                {
                    "dataset": "har",
                    "model": "cnn1d",
                    "q_max": 16,
                    "error_type": tname,
                    "n_samples": n_pos,
                    "feature": feat,
                    "auroc": auc,
                    "p_at_10": p10,
                    "ci_auroc_low": ci_auc[0],
                    "ci_auroc_high": ci_auc[1],
                    "ci_p10_low": ci_p10[0],
                    "ci_p10_high": ci_p10[1],
                }
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def delta_ci(y: np.ndarray, s_prev: np.ndarray, s_curr: np.ndarray, n_boot: int = 1000) -> Tuple[float, float, float, float]:
    d_auc_vals = []
    d_p10_vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yy = y[idx]
        a_prev = safe_auc(yy, s_prev[idx])
        a_curr = safe_auc(yy, s_curr[idx])
        p_prev = p_at_fraction(yy, s_prev[idx], 0.10)
        p_curr = p_at_fraction(yy, s_curr[idx], 0.10)
        if np.isfinite(a_prev) and np.isfinite(a_curr):
            d_auc_vals.append(a_curr - a_prev)
        if np.isfinite(p_prev) and np.isfinite(p_curr):
            d_p10_vals.append(p_curr - p_prev)

    if d_auc_vals:
        auc_lo, auc_hi = float(np.quantile(d_auc_vals, 0.025)), float(np.quantile(d_auc_vals, 0.975))
    else:
        auc_lo = auc_hi = float("nan")
    if d_p10_vals:
        p_lo, p_hi = float(np.quantile(d_p10_vals, 0.025)), float(np.quantile(d_p10_vals, 0.975))
    else:
        p_lo = p_hi = float("nan")
    return auc_lo, auc_hi, p_lo, p_hi


def make_incremental_table(df: pd.DataFrame, out_csv: Path, n_boot: int) -> None:
    y = df["is_error"].to_numpy(dtype=int)
    subset_defs = [
        ("base", ["m_neg"]),
        ("base_plus_conflict", ["m_neg", "M_B_minus", "CE_B", "r_B_minus"]),
        (
            "base_plus_conflict_plus_fragility",
            ["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost"],
        ),
        (
            "full_panel",
            ["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"],
        ),
    ]

    preds: Dict[str, np.ndarray] = {}
    rows = []
    prev_name = None

    for name, feats in subset_defs:
        X = df[feats].to_numpy(dtype=float)
        pred = cv_logit_scores(X, y)
        preds[name] = pred

        auc = safe_auc(y, pred)
        p10 = p_at_fraction(y, pred, 0.10)

        if prev_name is None:
            d_auc = d_p10 = 0.0
            ci = (0.0, 0.0, 0.0, 0.0)
        else:
            d_auc = auc - safe_auc(y, preds[prev_name])
            d_p10 = p10 - p_at_fraction(y, preds[prev_name], 0.10)
            ci = delta_ci(y, preds[prev_name], pred, n_boot=n_boot)

        rows.append(
            {
                "dataset": "har",
                "model": "cnn1d",
                "q_max": 16,
                "subset_name": name,
                "features": "|".join(feats),
                "n_samples": len(y),
                "auroc": auc,
                "p_at_10": p10,
                "delta_auroc_vs_prev": d_auc,
                "delta_p10_vs_prev": d_p10,
                "ci_delta_auroc_low": ci[0],
                "ci_delta_auroc_high": ci[1],
                "ci_delta_p10_low": ci[2],
                "ci_delta_p10_high": ci[3],
            }
        )
        prev_name = name

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _z(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    if s == 0:
        return np.zeros_like(x, dtype=float)
    return (x - np.mean(x)) / s


def _panel_score(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = (_z(df["M_B_minus"].to_numpy()) + _z(df["CE_B"].to_numpy()) + _z(df["r_B_minus"].to_numpy())) / 3.0
    frag = (_z(df["rho_B_cost"].to_numpy()) + _z(df["frag_drop"].to_numpy())) / 2.0
    score = 0.5 * conf + 0.5 * frag
    return conf, frag, score


def evaluate_alert_policy(y: np.ndarray, score: np.ndarray, frac: float) -> Tuple[float, float, float]:
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-score)[:k]
    tp = int(y[idx].sum())
    fp = k - tp
    fn = int(y.sum()) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def make_panel_vs_scalar_table(df: pd.DataFrame, out_csv: Path, n_boot: int) -> None:
    y = df["is_error"].to_numpy(dtype=int)
    mneg = df["m_neg"].to_numpy(dtype=float)
    panel_feats = ["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]
    panel_score = cv_logit_scores(df[panel_feats].to_numpy(dtype=float), y)
    fuzzy_score = make_fuzzy_panel_score(df)

    budgets = [0.10, 0.20]
    rows = []

    for b in budgets:
        for policy, score, params in [
            ("scalar", mneg, "threshold=top_budget_by_m_neg"),
            ("panel", panel_score, "logit(conflict+fragility)_top_budget"),
            ("fuzzy_panel", fuzzy_score, "mamdani(conflict,fragility,margin)_top_budget"),
        ]:
            prec, rec, f1 = evaluate_alert_policy(y, score, b)
            auprc = float(average_precision_score(y, score))
            ci_prec = bootstrap_ci(lambda yy, ss: evaluate_alert_policy(yy, ss, b)[0], y, score, n_boot=n_boot)
            ci_rec = bootstrap_ci(lambda yy, ss: evaluate_alert_policy(yy, ss, b)[1], y, score, n_boot=n_boot)

            rows.append(
                {
                    "dataset": "har",
                    "model": "cnn1d",
                    "q_max": 16,
                    "policy": policy,
                    "policy_params": params,
                    "alert_budget": b,
                    "n_samples": len(y),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auprc": auprc,
                    "ci_precision_low": ci_prec[0],
                    "ci_precision_high": ci_prec[1],
                    "ci_recall_low": ci_rec[0],
                    "ci_recall_high": ci_rec[1],
                }
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--risk-rows", default="outputs_composite/har_main_beacon_cnn_fast/risk_rows.csv")
    p.add_argument("--local-metrics", default="outputs_composite/har_main_beacon_cnn_fast/local_metrics.csv")
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--local-method", default="beacon_refine")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out-error-type", default="outputs_composite/audit_error_type_specialization.csv")
    p.add_argument("--out-incremental", default="outputs_composite/audit_incremental_feature_value.csv")
    p.add_argument("--out-panel-vs-scalar", default="outputs_composite/audit_panel_vs_scalar.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = build_panel_df(Path(args.risk_rows), Path(args.local_metrics), q=args.q_max, method=args.local_method)

    make_error_type_table(df, Path(args.out_error_type), n_boot=args.n_boot)
    make_incremental_table(df, Path(args.out_incremental), n_boot=args.n_boot)
    make_panel_vs_scalar_table(df, Path(args.out_panel_vs_scalar), n_boot=args.n_boot)

    print(f"Built audit tables from n={len(df)} samples")
    print(f"- {args.out_error_type}")
    print(f"- {args.out_incremental}")
    print(f"- {args.out_panel_vs_scalar}")


if __name__ == "__main__":
    main()
