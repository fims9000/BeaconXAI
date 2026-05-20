from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1e-8
R_CF_MAX = 10.0


def _extra_conflict_features(
    deltas: np.ndarray,
    margin: float,
    rho_b_cost: float,
) -> dict[str, float]:
    d = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return {
            "var_conflict": 0.0,
            "conflict_connectivity": 0.0,
            "delta_frag_proxy": 0.0,
            "r_cf": 0.0,
        }

    # Conflict/support convention:
    # d < 0 -> conflict (neutralization increases model margin),
    # d > 0 -> support.
    conf_mag = -d[d < 0.0]
    sup_mag = d[d > 0.0]
    var_conflict = float(np.var(conf_mag)) if conf_mag.size > 1 else 0.0

    conf_mask = d < 0.0
    conf_cnt = int(np.sum(conf_mask))
    if conf_cnt <= 1:
        conflict_connectivity = 0.0
    else:
        edge_hits = float(np.sum(conf_mask[1:] & conf_mask[:-1]))
        conflict_connectivity = float(edge_hits / max(conf_cnt - 1, 1))

    max_support = float(np.max(sup_mag)) if sup_mag.size > 0 else 0.0
    full_frag_proxy = float(abs(float(margin)) / (max_support + EPS))
    delta_frag_proxy = float(full_frag_proxy - float(rho_b_cost))

    m_minus = float(np.sum(conf_mag))
    r_cf = float(m_minus / (float(rho_b_cost) + EPS))
    r_cf = float(np.clip(r_cf, 0.0, R_CF_MAX))
    return {
        "var_conflict": var_conflict,
        "conflict_connectivity": conflict_connectivity,
        "delta_frag_proxy": delta_frag_proxy,
        "r_cf": r_cf,
    }


def summarize_deltas(deltas: np.ndarray, top_k: int = 3) -> dict[str, float]:
    d = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return {
            "top1_delta": 0.0,
            "top3_sum_delta": 0.0,
            "top3_mean_delta": 0.0,
            "top3_conflict_count": 0.0,
            "delta_entropy": 0.0,
            "rank_entropy": 0.0,
        }

    order = np.argsort(-np.abs(d))
    k = int(min(max(1, top_k), d.size))
    top = d[order[:k]]
    top_abs = np.abs(top)

    abs_all = np.abs(d)
    s = float(np.sum(abs_all))
    if s > EPS:
        p = abs_all / s
        ent = float(-np.sum(p * np.log(np.maximum(p, EPS))) / np.log(max(2, len(p))))
    else:
        ent = 0.0

    out = {
        "top1_delta": float(top_abs[0]) if len(top_abs) > 0 else 0.0,
        "top3_sum_delta": float(np.sum(top_abs)),
        "top3_mean_delta": float(np.mean(top_abs)),
        "top3_conflict_count": float(np.sum(top < 0.0)),
        "delta_entropy": ent,
        "rank_entropy": ent,
    }
    return out


def margin_entropy_from_margin(margin: float) -> float:
    # Binary uncertainty proxy from margin.
    # Large positive margin => low entropy, near-zero margin => high entropy.
    p = 1.0 / (1.0 + np.exp(-float(margin)))
    p = float(np.clip(p, EPS, 1.0 - EPS))
    ent = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return float(ent)


def extract_audit_vector(
    beacon_result: Any | None,
    margin: float,
    q_max: int,
    sample_id: int,
    label: int,
    is_hidden_conflict: int,
    method: str,
    seed: int,
    deltas: np.ndarray | None = None,
    rho_b_cost: float | None = None,
    frag_drop: float | None = None,
    margin_entropy: float | None = None,
) -> dict[str, float | int | str]:
    if deltas is None:
        deltas_arr = np.array([], dtype=np.float64)
    else:
        deltas_arr = np.asarray(deltas, dtype=np.float64).reshape(-1)

    if beacon_result is not None:
        m_minus = float(getattr(beacon_result, "counter_mass", 0.0))
        m_plus = float(getattr(beacon_result, "support_mass", 0.0))
        ce_b = float(getattr(beacon_result, "counter_evidence_gain", 0.0))
        rho_cost_v = float(getattr(beacon_result, "rho_b_cost", 1.0)) if rho_b_cost is None else float(rho_b_cost)
        m0_b = float(getattr(beacon_result, "m0", margin))
        mlast_b = float(getattr(beacon_result, "m_last", margin))
        frag_drop_v = float(max(0.0, (m0_b - mlast_b) / (abs(m0_b) + EPS)))
        if frag_drop is not None:
            frag_drop_v = float(frag_drop)
    else:
        m_plus = float(np.sum(np.maximum(deltas_arr, 0.0)))
        m_minus = float(np.sum(np.maximum(-deltas_arr, 0.0)))
        ce_b = float(m_minus / max(1.0, float(q_max)))
        rho_cost_v = float(1.0 if rho_b_cost is None else rho_b_cost)
        frag_drop_v = float(max(0.0, 0.0 if frag_drop is None else frag_drop))
    frag_drop_v = float(min(1.0, max(0.0, frag_drop_v)))

    r_minus = float(m_minus / max(m_minus + m_plus, EPS))
    m_neg = float(-margin)
    top = summarize_deltas(deltas_arr)
    m_ent = float(margin_entropy_from_margin(margin) if margin_entropy is None else margin_entropy)
    extra = _extra_conflict_features(deltas_arr, margin=float(margin), rho_b_cost=float(rho_cost_v))

    return {
        "sample_id": int(sample_id),
        "label": int(label),
        "is_hidden_conflict": int(is_hidden_conflict),
        "m_neg": m_neg,
        "M_B_minus": m_minus,
        "M_B_plus": m_plus,
        "r_B_minus": r_minus,
        "CE_B": ce_b,
        "rho_B_cost": float(rho_cost_v),
        "frag_drop": float(frag_drop_v),
        "top1_delta": top["top1_delta"],
        "top3_sum_delta": top["top3_sum_delta"],
        "top3_mean_delta": top["top3_mean_delta"],
        "top3_conflict_count": top["top3_conflict_count"],
        "delta_entropy": top["delta_entropy"],
        "rank_entropy": top["rank_entropy"],
        "margin_entropy": m_ent,
        "var_conflict": extra["var_conflict"],
        "conflict_connectivity": extra["conflict_connectivity"],
        "delta_frag_proxy": extra["delta_frag_proxy"],
        "r_cf": extra["r_cf"],
        "method": str(method),
        "q_max": int(q_max),
        "seed": int(seed),
    }
