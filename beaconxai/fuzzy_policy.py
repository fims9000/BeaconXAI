from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    if s < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - float(np.mean(x))) / s


def make_fuzzy_inputs(df):
    m_minus = df["M_B_minus"].to_numpy(dtype=float)
    r_minus = df["r_B_minus"].to_numpy(dtype=float)
    ce_b = df["CE_B"].to_numpy(dtype=float)
    frag_drop = df["frag_drop"].to_numpy(dtype=float)
    rho_cost = df["rho_B_cost"].to_numpy(dtype=float)
    m_neg = df["m_neg"].to_numpy(dtype=float)

    conflict = (zscore(m_minus) + zscore(r_minus) + zscore(ce_b)) / 3.0
    fragility = (zscore(frag_drop) + zscore(1.0 - rho_cost)) / 2.0
    margin = zscore(m_neg)
    return conflict, fragility, margin


def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
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


def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if b > a:
        m = (x >= a) & (x <= b)
        y[m] = (x[m] - a) / (b - a)
    if c > b:
        m = (x >= b) & (x <= c)
        y[m] = (c - x[m]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def membership_params(train_vals: np.ndarray, scheme: str):
    qmap = {
        "q25_75": (0.25, 0.50, 0.75),
        "q33_66": (0.33, 0.50, 0.66),
    }
    ql, qm, qh = qmap[scheme]
    vql, vqm, vqh = np.quantile(train_vals, [ql, qm, qh])
    lo = float(np.min(train_vals))
    hi = float(np.max(train_vals))
    span = max(hi - lo, 1e-6)
    lo -= 0.05 * span
    hi += 0.05 * span
    return lo, float(vql), float(vqm), float(vqh), hi


def memberships(vals: np.ndarray, params):
    lo, vql, vqm, vqh, hi = params
    return {
        "low": trapmf(vals, lo, lo, vql, vqm),
        "med": trimf(vals, vql, vqm, vqh),
        "high": trapmf(vals, vqm, vqh, hi, hi),
    }


@dataclass
class FuzzyConfig:
    membership_scheme: str
    inference: str
    rule_set: str
    high_weight: float
    medium_weight: float
    margin_weight: float


def fuzzy_score(
    conflict: np.ndarray,
    fragility: np.ndarray,
    margin: np.ndarray,
    params: dict,
    cfg: FuzzyConfig,
) -> np.ndarray:
    mc = memberships(conflict, params["conflict"])
    mf = memberships(fragility, params["fragility"])
    mm = memberships(margin, params["margin"])

    out = np.zeros(len(conflict), dtype=float)
    for i in range(len(conflict)):
        rules = [
            (min(mc["high"][i], mm["high"][i]) * cfg.margin_weight, 1.0 * cfg.high_weight),
            (min(mc["high"][i], mf["high"][i]), 1.0 * cfg.high_weight),
            (min(mf["high"][i], mm["high"][i]) * cfg.margin_weight, 1.0 * cfg.high_weight),
            (min(mc["med"][i], mf["high"][i]), 0.5 * cfg.medium_weight),
            (min(mc["low"][i], mf["low"][i], mm["low"][i]), 0.0),
        ]
        if cfg.rule_set == "extended7":
            rules.extend(
                [
                    (min(mc["high"][i], mf["low"][i]), 0.75 * cfg.medium_weight),
                    (min(mm["high"][i], mc["med"][i]) * cfg.margin_weight, 0.75 * cfg.medium_weight),
                ]
            )

        if cfg.inference == "sugeno":
            ws = np.array([r[0] for r in rules], dtype=float)
            zs = np.array([r[1] for r in rules], dtype=float)
            den = float(np.sum(ws))
            out[i] = float(np.sum(ws * zs) / den) if den > 1e-12 else 0.0
        else:
            # Lightweight Mamdani surrogate for ranking score.
            out[i] = float(max(r[0] * r[1] for r in rules))
    return out


def gate_score(fuzzy: np.ndarray, backend: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    out = backend.copy()
    out[fuzzy <= tau_low] = 0.0
    out[fuzzy >= tau_high] = 1.0
    return out


def eval_at_budget(y: np.ndarray, score: np.ndarray, frac: float):
    k = max(1, int(np.ceil(frac * len(y))))
    idx = np.argsort(-score)[:k]
    tp = int(np.sum(y[idx] == 1))
    fp = k - tp
    fn = int(np.sum(y == 1)) - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return float(p), float(r), float(f1)
