#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.make_audit_panel_tables import build_panel_df
from scripts.run_har_hidden_conflict_tan import TANModel


def _z(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    if s < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - float(np.mean(x))) / s


def _stratified_split(y: np.ndarray, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(round(n * train_frac)))
        n_va = max(1, int(round(n * val_frac)))
        n_te = max(1, n - n_tr - n_va)
        if n_tr + n_va + n_te > n:
            n_te = max(1, n - n_tr - n_va)
        tr.append(idx[:n_tr])
        va.append(idx[n_tr : n_tr + n_va])
        te.append(idx[n_tr + n_va :])
    tr = np.concatenate(tr)
    va = np.concatenate(va)
    te = np.concatenate(te)
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


def _trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if b > a:
        m = (x >= a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    m = (x >= b) & (x <= c)
    y[m] = 1.0
    if d > c:
        m = (x > c) & (x <= d)
        y[m] = (d - x[m]) / (d - c)
    return np.clip(y, 0.0, 1.0)


def _trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if b > a:
        m = (x >= a) & (x <= b)
        y[m] = (x[m] - a) / (b - a)
    if c > b:
        m = (x >= b) & (x <= c)
        y[m] = (c - x[m]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def _membership_params(train_vals: np.ndarray, scheme: str):
    qmap = {
        "q33_66": (0.33, 0.50, 0.66),
        "q25_75": (0.25, 0.50, 0.75),
        "q20_80": (0.20, 0.50, 0.80),
    }
    ql, qm, qh = qmap[scheme]
    lo = float(np.min(train_vals))
    hi = float(np.max(train_vals))
    span = max(hi - lo, 1e-6)
    lo -= 0.05 * span
    hi += 0.05 * span
    vql, vqm, vqh = np.quantile(train_vals, [ql, qm, qh])
    return lo, float(vql), float(vqm), float(vqh), hi


def _memberships(vals: np.ndarray, params):
    lo, vql, vqm, vqh, hi = params
    return {
        "low": _trapmf(vals, lo, lo, vql, vqm),
        "med": _trimf(vals, vql, vqm, vqh),
        "high": _trapmf(vals, vqm, vqh, hi, hi),
    }


def _fuzzy_sugeno_score(conf: np.ndarray, frag: np.ndarray, margin: np.ndarray, params, rule_set: str, high_w: float, med_w: float, margin_w: float) -> np.ndarray:
    mu_c = _memberships(conf, params["conf"]) 
    mu_f = _memberships(frag, params["frag"]) 
    mu_m = _memberships(margin, params["margin"]) 
    n = len(conf)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        rules = []
        # Base 5 rules
        rules.append((min(mu_c["high"][i], mu_m["high"][i]) * margin_w, 1.0 * high_w))
        rules.append((min(mu_c["high"][i], mu_f["high"][i]), 1.0 * high_w))
        rules.append((min(mu_f["high"][i], mu_m["high"][i]) * margin_w, 1.0 * high_w))
        rules.append((min(mu_c["med"][i], mu_f["high"][i]), 0.5 * med_w))
        rules.append((min(mu_c["low"][i], mu_f["low"][i], mu_m["low"][i]), 0.0))
        if rule_set == "extended7":
            rules.append((min(mu_c["high"][i], mu_f["low"][i]), 0.5 * med_w))
            rules.append((min(mu_m["high"][i], mu_c["med"][i]) * margin_w, 0.5 * med_w))
        ws = np.array([r[0] for r in rules], dtype=float)
        zs = np.array([r[1] for r in rules], dtype=float)
        den = float(np.sum(ws))
        out[i] = float(np.sum(ws * zs) / den) if den > 1e-12 else 0.0
    return out


def _eval_budget(y: np.ndarray, score: np.ndarray, frac: float):
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-score)[:k]
    tp = int(np.sum(y[idx] == 1))
    fp = k - tp
    fn = int(np.sum(y == 1)) - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return float(p), float(r), float(f1)


def _gate_score(fuzzy_score: np.ndarray, backend_score: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    out = backend_score.copy()
    out[fuzzy_score <= tau_low] = 0.0
    out[fuzzy_score >= tau_high] = 1.0
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrated fuzzy and fuzzy-gate risk policies")
    p.add_argument("--risk-rows", default="outputs_composite/har_main_beacon_extratrees_fast/risk_rows.csv")
    p.add_argument("--local-metrics", default="outputs_composite/har_main_beacon_extratrees_fast/local_metrics.csv")
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--local-method", default="beacon_refine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--tan-bins", type=int, default=5)
    p.add_argument("--tan-alpha", type=float, default=1.0)
    p.add_argument("--out-summary", default="outputs_composite/fuzzy_gate_summary.csv")
    p.add_argument("--out-grid", default="outputs_composite/fuzzy_gate_grid.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = build_panel_df(Path(args.risk_rows), Path(args.local_metrics), q=args.q_max, method=args.local_method)
    y = df["is_error"].to_numpy(dtype=np.int64)

    # Three aggregated axes.
    conf = (_z(df["M_B_minus"].to_numpy(float)) + _z(df["CE_B"].to_numpy(float)) + _z(df["r_B_minus"].to_numpy(float))) / 3.0
    frag = (_z(df["frag_drop"].to_numpy(float)) + _z(1.0 - df["rho_B_cost"].to_numpy(float))) / 2.0
    margin = _z(df["m_neg"].to_numpy(float))

    X_panel = df[["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]].to_numpy(float)
    X_tan = np.stack([margin, conf, frag], axis=1)
    X_scalar = df[["m_neg"]].to_numpy(float)

    tr, va, te = _stratified_split(y, args.train_frac, args.val_frac, args.seed)

    # Baselines fitted on train only.
    logit_panel = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    logit_panel.fit(X_panel[tr], y[tr])
    sc_panel_val = logit_panel.predict_proba(X_panel[va])[:, 1]
    sc_panel_test = logit_panel.predict_proba(X_panel[te])[:, 1]

    tan = TANModel(n_bins=args.tan_bins, alpha=args.tan_alpha).fit(X_tan[tr], y[tr])
    sc_tan_val = tan.predict_proba(X_tan[va])[:, 1]
    sc_tan_test = tan.predict_proba(X_tan[te])[:, 1]

    sc_scalar_val = X_scalar[va, 0]
    sc_scalar_test = X_scalar[te, 0]

    schemes = ["q33_66", "q25_75", "q20_80"]
    rule_sets = ["base5", "extended7"]
    high_ws = [1.0, 1.5, 2.0]
    med_ws = [0.5, 0.75, 1.0]
    margin_ws = [0.5, 1.0, 1.5]

    params_by_scheme = {}
    for scheme in schemes:
        params_by_scheme[scheme] = {
            "conf": _membership_params(conf[tr], scheme),
            "frag": _membership_params(frag[tr], scheme),
            "margin": _membership_params(margin[tr], scheme),
        }

    grid_rows = []
    best = None
    best_score = -1.0

    for scheme in schemes:
        for rs in rule_sets:
            for hw in high_ws:
                for mw in med_ws:
                    for marw in margin_ws:
                        f_val = _fuzzy_sugeno_score(conf[va], frag[va], margin[va], params_by_scheme[scheme], rs, hw, mw, marw)
                        f_test = _fuzzy_sugeno_score(conf[te], frag[te], margin[te], params_by_scheme[scheme], rs, hw, mw, marw)

                        p10, r10, f1_10 = _eval_budget(y[va], f_val, 0.10)
                        p20, r20, f1_20 = _eval_budget(y[va], f_val, 0.20)
                        auprc = float(average_precision_score(y[va], f_val))

                        # Gate thresholds from validation distribution.
                        tau_low = float(np.quantile(f_val, 0.30))
                        tau_high = float(np.quantile(f_val, 0.70))
                        g_logit_val = _gate_score(f_val, sc_panel_val, tau_low, tau_high)
                        gp10, gr10, gf1_10 = _eval_budget(y[va], g_logit_val, 0.10)
                        _, _, gf1_20 = _eval_budget(y[va], g_logit_val, 0.20)
                        g_auprc = float(average_precision_score(y[va], g_logit_val))

                        target = 0.5 * gf1_10 + 0.25 * gp10 + 0.25 * gf1_20
                        row = {
                            "scheme": scheme,
                            "rule_set": rs,
                            "high_weight": hw,
                            "medium_weight": mw,
                            "margin_weight": marw,
                            "val_fuzzy_f1_10": f1_10,
                            "val_fuzzy_precision_10": p10,
                            "val_fuzzy_recall_10": r10,
                            "val_fuzzy_f1_20": f1_20,
                            "val_fuzzy_auprc": auprc,
                            "val_gate_logit_f1_10": gf1_10,
                            "val_gate_logit_precision_10": gp10,
                            "val_gate_logit_recall_10": gr10,
                            "val_gate_logit_f1_20": gf1_20,
                            "val_gate_logit_auprc": g_auprc,
                            "val_target": target,
                        }
                        grid_rows.append(row)
                        if target > best_score:
                            best_score = target
                            best = (scheme, rs, hw, mw, marw)

    if best is None:
        raise RuntimeError("No fuzzy configuration evaluated")

    scheme, rs, hw, mw, marw = best
    params = params_by_scheme[scheme]
    f_test = _fuzzy_sugeno_score(conf[te], frag[te], margin[te], params, rs, hw, mw, marw)
    f_val = _fuzzy_sugeno_score(conf[va], frag[va], margin[va], params, rs, hw, mw, marw)

    tau_low = float(np.quantile(f_val, 0.30))
    tau_high = float(np.quantile(f_val, 0.70))
    g_logit_test = _gate_score(f_test, sc_panel_test, tau_low, tau_high)
    g_tan_test = _gate_score(f_test, sc_tan_test, tau_low, tau_high)

    policies = {
        "scalar": sc_scalar_test,
        "logit_panel": sc_panel_test,
        "fuzzy_only": f_test,
        "fuzzy_gate_logit": g_logit_test,
        "fuzzy_gate_tan": g_tan_test,
    }

    summary_rows = []
    for name, s in policies.items():
        p10, r10, f1_10 = _eval_budget(y[te], s, 0.10)
        p20, r20, f1_20 = _eval_budget(y[te], s, 0.20)
        summary_rows.append(
            {
                "policy": name,
                "test_precision_10": p10,
                "test_recall_10": r10,
                "test_f1_10": f1_10,
                "test_precision_20": p20,
                "test_recall_20": r20,
                "test_f1_20": f1_20,
                "test_auprc": float(average_precision_score(y[te], s)),
                "n_test": int(len(te)),
                "best_scheme": scheme,
                "best_rule_set": rs,
                "best_high_weight": hw,
                "best_medium_weight": mw,
                "best_margin_weight": marw,
            }
        )

    out_grid = Path(args.out_grid)
    out_grid.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_rows).sort_values("val_target", ascending=False).to_csv(out_grid, index=False)

    out_summary = Path(args.out_summary)
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        wr.writeheader()
        wr.writerows(summary_rows)

    print(f"n_total={len(df)} train={len(tr)} val={len(va)} test={len(te)}")
    print(
        f"best fuzzy config: scheme={scheme}, rules={rs}, high_w={hw}, med_w={mw}, margin_w={marw}, val_target={best_score:.4f}"
    )
    print(f"saved: {out_grid}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()
