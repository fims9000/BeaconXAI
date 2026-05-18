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


def bootstrap_metric_delta(
    metric_fn,
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 1000,
) -> Tuple[float, float, float]:
    n = len(y_true)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        va = metric_fn(y_true[idx], score_a[idx])
        vb = metric_fn(y_true[idx], score_b[idx])
        if np.isfinite(va) and np.isfinite(vb):
            vals.append(float(va - vb))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    lo = float(np.quantile(arr, 0.025))
    hi = float(np.quantile(arr, 0.975))
    # Two-sided bootstrap sign p-value around zero.
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    return lo, hi, min(1.0, max(0.0, p))


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


def _fit_quantile_bins(X: np.ndarray, n_bins: int = 3) -> list[np.ndarray]:
    # Internal edges only (without -inf/+inf).
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges: list[np.ndarray] = []
    for j in range(X.shape[1]):
        e = np.quantile(X[:, j], qs)
        e = np.unique(e.astype(float))
        edges.append(e)
    return edges


def _apply_quantile_bins(X: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(X, dtype=np.int64)
    for j, e in enumerate(edges):
        if e.size == 0:
            out[:, j] = 0
        else:
            out[:, j] = np.digitize(X[:, j], e, right=False).astype(np.int64)
    return out


def _fit_tan_binary(X_disc: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    # Tree-Augmented Naive Bayes for discrete features.
    n, d = X_disc.shape
    classes = np.array([0, 1], dtype=np.int64)
    n_classes = 2
    feat_states = np.array([int(X_disc[:, j].max()) + 1 for j in range(d)], dtype=np.int64)

    # P(Y)
    cnt_y = np.array([(y == c).sum() for c in classes], dtype=np.float64)
    p_y = (cnt_y + alpha) / (cnt_y.sum() + alpha * n_classes)

    # Conditional mutual information I(Xi;Xj|Y)
    cmi = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        si = feat_states[i]
        for j in range(i + 1, d):
            sj = feat_states[j]
            v = 0.0
            for c in classes:
                mask = y == c
                nc = int(mask.sum())
                if nc == 0:
                    continue
                p_c = p_y[c]
                xi = X_disc[mask, i]
                xj = X_disc[mask, j]
                cnt_ij = np.zeros((si, sj), dtype=np.float64)
                for a, b in zip(xi, xj):
                    cnt_ij[a, b] += 1.0
                p_ij = (cnt_ij + alpha) / (nc + alpha * si * sj)
                cnt_i = cnt_ij.sum(axis=1)
                cnt_j = cnt_ij.sum(axis=0)
                p_i = (cnt_i + alpha) / (nc + alpha * si)
                p_j = (cnt_j + alpha) / (nc + alpha * sj)
                ratio = p_ij / np.maximum(p_i[:, None] * p_j[None, :], 1e-12)
                v += p_c * float(np.sum(p_ij * np.log(np.maximum(ratio, 1e-12))))
            cmi[i, j] = cmi[j, i] = v

    # Maximum spanning tree (Prim), root=0
    parent = np.full(d, -1, dtype=np.int64)
    used = np.zeros(d, dtype=bool)
    used[0] = True
    best_w = np.full(d, -np.inf, dtype=np.float64)
    best_p = np.full(d, -1, dtype=np.int64)
    best_w[0] = 0.0
    for _ in range(d - 1):
        u = -1
        mx = -np.inf
        for v in range(d):
            if not used[v] and best_w[v] > mx:
                mx = best_w[v]
                u = v
        if u < 0:
            break
        used[u] = True
        parent[u] = best_p[u]
        for v in range(d):
            if not used[v] and cmi[u, v] > best_w[v]:
                best_w[v] = cmi[u, v]
                best_p[v] = u

    # CPTs
    root = 0
    root_states = feat_states[root]
    p_root_y = np.zeros((n_classes, root_states), dtype=np.float64)
    for c in classes:
        mask = y == c
        nc = int(mask.sum())
        cnt = np.bincount(X_disc[mask, root], minlength=root_states).astype(np.float64)
        p_root_y[c] = (cnt + alpha) / (nc + alpha * root_states)

    p_x_parent_y: dict[int, np.ndarray] = {}
    for i in range(1, d):
        p = parent[i]
        if p < 0:
            p = root
            parent[i] = root
        si = feat_states[i]
        sp = feat_states[p]
        t = np.zeros((n_classes, sp, si), dtype=np.float64)
        for c in classes:
            mask = y == c
            xp = X_disc[mask, p]
            xi = X_disc[mask, i]
            cnt = np.zeros((sp, si), dtype=np.float64)
            for a, b in zip(xp, xi):
                cnt[a, b] += 1.0
            denom = cnt.sum(axis=1, keepdims=True)
            t[c] = (cnt + alpha) / np.maximum(denom + alpha * si, 1e-12)
        p_x_parent_y[i] = t

    return {
        "p_y": p_y,
        "parent": parent,
        "root": root,
        "p_root_y": p_root_y,
        "p_x_parent_y": p_x_parent_y,
    }


def _predict_tan_proba(model: dict, X_disc: np.ndarray) -> np.ndarray:
    n, d = X_disc.shape
    logp = np.zeros((n, 2), dtype=np.float64)
    p_y = model["p_y"]
    parent = model["parent"]
    root = int(model["root"])
    p_root_y = model["p_root_y"]
    p_x_parent_y = model["p_x_parent_y"]

    for c in (0, 1):
        lp = np.full(n, np.log(np.maximum(p_y[c], 1e-12)), dtype=np.float64)
        xr = X_disc[:, root]
        lp += np.log(np.maximum(p_root_y[c, xr], 1e-12))
        for i in range(d):
            if i == root:
                continue
            p = int(parent[i])
            xi = X_disc[:, i]
            xp = X_disc[:, p]
            lp += np.log(np.maximum(p_x_parent_y[i][c, xp, xi], 1e-12))
        logp[:, c] = lp

    mx = np.max(logp, axis=1, keepdims=True)
    ex = np.exp(logp - mx)
    pr = ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)
    return pr[:, 1]


def cv_tan_scores(X: np.ndarray, y: np.ndarray, seed: int = 42, n_bins: int = 3) -> np.ndarray:
    pred = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        edges = _fit_quantile_bins(X[tr], n_bins=n_bins)
        xtr = _apply_quantile_bins(X[tr], edges)
        xte = _apply_quantile_bins(X[te], edges)
        model = _fit_tan_binary(xtr, y[tr])
        pred[te] = _predict_tan_proba(model, xte)
    return pred


def _membership_stats(x: np.ndarray) -> tuple[float, float, float]:
    q10, q50, q90 = np.quantile(x, [0.10, 0.50, 0.90]).astype(float)
    if q50 <= q10:
        q50 = q10 + 1e-6
    if q90 <= q50:
        q90 = q50 + 1e-6
    return q10, q50, q90


def _mu_low(v: np.ndarray, q10: float, q50: float) -> np.ndarray:
    return np.clip((q50 - v) / max(q50 - q10, 1e-12), 0.0, 1.0)


def _mu_high(v: np.ndarray, q50: float, q90: float) -> np.ndarray:
    return np.clip((v - q50) / max(q90 - q50, 1e-12), 0.0, 1.0)


def _fuzzy_train_params(X: np.ndarray, feat_names: list[str]) -> dict:
    stats = {}
    for j, name in enumerate(feat_names):
        stats[name] = _membership_stats(X[:, j])
    return {"stats": stats}


def _fuzzy_score(X: np.ndarray, feat_names: list[str], params: dict) -> np.ndarray:
    # Sugeno-style compact fuzzy policy.
    idx = {n: i for i, n in enumerate(feat_names)}
    s = params["stats"]
    eps = 1e-12

    def low(name: str) -> np.ndarray:
        q10, q50, _ = s[name]
        return _mu_low(X[:, idx[name]], q10, q50)

    def high(name: str) -> np.ndarray:
        _, q50, q90 = s[name]
        return _mu_high(X[:, idx[name]], q50, q90)

    support_high = np.maximum.reduce([high("M_B_minus"), high("CE_B"), high("r_B_minus")])
    support_low = np.maximum.reduce([low("M_B_minus"), low("CE_B"), low("r_B_minus")])
    frag_high = high("rho_B_cost")
    margin_high = high("m_neg")
    margin_low = low("m_neg")

    # Rules:
    w1 = np.minimum(margin_high, frag_high)            # fragile + small margin
    w2 = np.minimum(margin_high, support_low)          # margin + weak support/counter evidence
    w3 = np.minimum(frag_high, support_low)            # fragile + weak support
    w4 = margin_high                                   # margin-only risk
    w5 = np.maximum(support_high, margin_low)          # safe zone

    if params.get("calibrate", False):
        # Return rule activations as feature matrix [n, 5].
        return np.column_stack([w1, w2, w3, w4, w5]).astype(np.float64)

    num = 0.95 * w1 + 0.85 * w2 + 0.80 * w3 + 0.65 * w4 + 0.10 * w5
    den = w1 + w2 + w3 + w4 + w5 + eps
    return num / den


def cv_fuzzy_scores(X: np.ndarray, y: np.ndarray, feat_names: list[str], seed: int = 42) -> np.ndarray:
    pred = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        params = _fuzzy_train_params(X[tr], feat_names)
        # Calibrated fuzzy: learn mapping from rule activations to error risk.
        params_cal = dict(params)
        params_cal["calibrate"] = True
        r_tr = _fuzzy_score(X[tr], feat_names, params_cal)
        r_te = _fuzzy_score(X[te], feat_names, params_cal)
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
            max_iter=2000,
        )
        clf.fit(r_tr, y[tr])
        pred[te] = clf.predict_proba(r_te)[:, 1]
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


def _rank_prob(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    if len(x) <= 1:
        return np.full(len(x), 0.5, dtype=np.float64)
    return ranks / (len(x) - 1)


def ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(y_true, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        ece += (np.sum(m) / max(n, 1)) * abs(acc - conf)
    return float(ece)


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


def make_panel_vs_scalar_table(
    df: pd.DataFrame,
    out_csv: Path,
    n_boot: int,
    tan_bins: int = 6,
    out_policy_deltas: Path | None = None,
) -> None:
    y = df["is_error"].to_numpy(dtype=int)
    mneg = df["m_neg"].to_numpy(dtype=float)
    panel_feats = ["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]
    X_panel = df[panel_feats].to_numpy(dtype=float)
    panel_score = cv_logit_scores(X_panel, y)
    fuzzy_score = cv_fuzzy_scores(X_panel, y, panel_feats)
    tan_score = cv_tan_scores(X_panel, y, n_bins=max(2, int(tan_bins)))

    policy_scores = {
        "scalar": mneg,
        "panel": panel_score,
        "fuzzy_policy": fuzzy_score,
        "tan_policy": tan_score,
    }
    policy_params = {
        "scalar": "threshold=top_budget_by_m_neg",
        "panel": "logit(conflict+fragility)_top_budget",
        "fuzzy_policy": "sugeno_rules(conflict,fragility,margin)",
        "tan_policy": f"tree_augmented_nb(quantile_bins={max(2, int(tan_bins))})",
    }

    budgets = [0.10, 0.20]
    rows = []
    delta_rows = []

    for b in budgets:
        f1_by_policy: Dict[str, float] = {}
        for policy, score in policy_scores.items():
            prec, rec, f1 = evaluate_alert_policy(y, score, b)
            auprc = float(average_precision_score(y, score))
            if policy == "scalar":
                score_prob = _rank_prob(score)
            else:
                score_prob = np.clip(score, 0.0, 1.0)
            ece = ece_binary(y, score_prob, n_bins=10)
            ci_prec = bootstrap_ci(lambda yy, ss: evaluate_alert_policy(yy, ss, b)[0], y, score, n_boot=n_boot)
            ci_rec = bootstrap_ci(lambda yy, ss: evaluate_alert_policy(yy, ss, b)[1], y, score, n_boot=n_boot)
            ci_f1 = bootstrap_ci(lambda yy, ss: evaluate_alert_policy(yy, ss, b)[2], y, score, n_boot=n_boot)
            f1_by_policy[policy] = float(f1)

            rows.append(
                {
                    "dataset": "har",
                    "model": "cnn1d",
                    "q_max": 16,
                    "policy": policy,
                    "policy_params": policy_params[policy],
                    "alert_budget": b,
                    "n_samples": len(y),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auprc": auprc,
                    "ece": ece,
                    "ci_precision_low": ci_prec[0],
                    "ci_precision_high": ci_prec[1],
                    "ci_recall_low": ci_rec[0],
                    "ci_recall_high": ci_rec[1],
                    "ci_f1_low": ci_f1[0],
                    "ci_f1_high": ci_f1[1],
                }
            )

        # Deltas for article claims.
        panel_s = policy_scores["panel"]
        scalar_s = policy_scores["scalar"]
        for policy, score in policy_scores.items():
            if policy == "panel":
                continue
            d_lo, d_hi, p = bootstrap_metric_delta(
                lambda yy, ss: evaluate_alert_policy(yy, ss, b)[2],
                y,
                score,
                panel_s,
                n_boot=n_boot,
            )
            ds_lo, ds_hi, ps = bootstrap_metric_delta(
                lambda yy, ss: evaluate_alert_policy(yy, ss, b)[2],
                y,
                score,
                scalar_s,
                n_boot=n_boot,
            )
            delta_rows.append(
                {
                    "dataset": "har",
                    "model": "cnn1d",
                    "q_max": 16,
                    "alert_budget": b,
                    "policy": policy,
                    "f1": f1_by_policy[policy],
                    "panel_f1": f1_by_policy["panel"],
                    "scalar_f1": f1_by_policy["scalar"],
                    "delta_f1_vs_panel": f1_by_policy[policy] - f1_by_policy["panel"],
                    "ci_delta_f1_vs_panel_low": d_lo,
                    "ci_delta_f1_vs_panel_high": d_hi,
                    "p_value_vs_panel": p,
                    "delta_f1_vs_scalar": f1_by_policy[policy] - f1_by_policy["scalar"],
                    "ci_delta_f1_vs_scalar_low": ds_lo,
                    "ci_delta_f1_vs_scalar_high": ds_hi,
                    "p_value_vs_scalar": ps,
                }
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if out_policy_deltas is not None:
        pd.DataFrame(delta_rows).to_csv(out_policy_deltas, index=False)


def make_beacon_vs_uniform_table(
    risk_rows: Path,
    local_metrics: Path,
    q: int,
    out_csv: Path,
    n_boot: int = 1000,
) -> None:
    df_b = build_panel_df(risk_rows, local_metrics, q=q, method="beacon_refine")
    df_u = build_panel_df(risk_rows, local_metrics, q=q, method="uniform_refinement")
    key = ["sample_id", "is_error", "m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]
    db = df_b[key].rename(columns={c: f"{c}_b" for c in key if c != "sample_id"})
    du = df_u[key].rename(columns={c: f"{c}_u" for c in key if c != "sample_id"})
    m = db.merge(du, on="sample_id", how="inner")
    y = m["is_error_b"].to_numpy(dtype=int)

    feats_b = np.column_stack([m[f"{c}_b"].to_numpy(dtype=float) for c in ["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]])
    feats_u = np.column_stack([m[f"{c}_u"].to_numpy(dtype=float) for c in ["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]])
    s_b = cv_logit_scores(feats_b, y)
    s_u = cv_logit_scores(feats_u, y)

    rows = []
    for name, s in [("beacon_panel", s_b), ("uniform_panel", s_u)]:
        p10, r10, f10 = evaluate_alert_policy(y, s, 0.10)
        p20, r20, f20 = evaluate_alert_policy(y, s, 0.20)
        rows.append(
            {
                "method": name,
                "n_samples": len(y),
                "auroc": safe_auc(y, s),
                "auprc": float(average_precision_score(y, s)),
                "f1_at_10": f10,
                "f1_at_20": f20,
                "ece": ece_binary(y, s, n_bins=10),
            }
        )

    d_auc = bootstrap_metric_delta(safe_auc, y, s_b, s_u, n_boot=n_boot)
    d_auprc = bootstrap_metric_delta(lambda yy, ss: float(average_precision_score(yy, ss)), y, s_b, s_u, n_boot=n_boot)
    d_f10 = bootstrap_metric_delta(lambda yy, ss: evaluate_alert_policy(yy, ss, 0.10)[2], y, s_b, s_u, n_boot=n_boot)
    d_f20 = bootstrap_metric_delta(lambda yy, ss: evaluate_alert_policy(yy, ss, 0.20)[2], y, s_b, s_u, n_boot=n_boot)
    rows.append(
        {
            "method": "delta_beacon_minus_uniform",
            "n_samples": len(y),
            "auroc": float(safe_auc(y, s_b) - safe_auc(y, s_u)),
            "auprc": float(average_precision_score(y, s_b) - average_precision_score(y, s_u)),
            "f1_at_10": float(evaluate_alert_policy(y, s_b, 0.10)[2] - evaluate_alert_policy(y, s_u, 0.10)[2]),
            "f1_at_20": float(evaluate_alert_policy(y, s_b, 0.20)[2] - evaluate_alert_policy(y, s_u, 0.20)[2]),
            "ece": float(ece_binary(y, s_b, n_bins=10) - ece_binary(y, s_u, n_bins=10)),
            "ci_auroc_low": d_auc[0],
            "ci_auroc_high": d_auc[1],
            "p_auroc": d_auc[2],
            "ci_auprc_low": d_auprc[0],
            "ci_auprc_high": d_auprc[1],
            "p_auprc": d_auprc[2],
            "ci_f1_10_low": d_f10[0],
            "ci_f1_10_high": d_f10[1],
            "p_f1_10": d_f10[2],
            "ci_f1_20_low": d_f20[0],
            "ci_f1_20_high": d_f20[1],
            "p_f1_20": d_f20[2],
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
    p.add_argument("--out-policy-deltas", default="outputs_composite/audit_policy_deltas.csv")
    p.add_argument("--out-beacon-vs-uniform", default="outputs_composite/audit_beacon_vs_uniform.csv")
    p.add_argument("--tan-bins", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = build_panel_df(Path(args.risk_rows), Path(args.local_metrics), q=args.q_max, method=args.local_method)

    make_error_type_table(df, Path(args.out_error_type), n_boot=args.n_boot)
    make_incremental_table(df, Path(args.out_incremental), n_boot=args.n_boot)
    make_panel_vs_scalar_table(
        df,
        Path(args.out_panel_vs_scalar),
        n_boot=args.n_boot,
        tan_bins=args.tan_bins,
        out_policy_deltas=Path(args.out_policy_deltas),
    )
    make_beacon_vs_uniform_table(
        Path(args.risk_rows),
        Path(args.local_metrics),
        q=args.q_max,
        out_csv=Path(args.out_beacon_vs_uniform),
        n_boot=args.n_boot,
    )

    print(f"Built audit tables from n={len(df)} samples")
    print(f"- {args.out_error_type}")
    print(f"- {args.out_incremental}")
    print(f"- {args.out_panel_vs_scalar}")
    print(f"- {args.out_policy_deltas}")
    print(f"- {args.out_beacon_vs_uniform}")


if __name__ == "__main__":
    main()
