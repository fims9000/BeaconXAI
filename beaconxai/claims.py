from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from .baselines import run_beacon_refine
from .neutralization import Neutralizer
from .types import BeaconConfig, Component, LocalMetricRow, RiskEvalRow


@dataclass
class ClaimReport:
    h1_pass: bool
    h2_pass: bool
    h3_pass: bool
    h4_pass: bool
    h5_pass: bool
    details: dict


def summarize_auroc(rows: Sequence[RiskEvalRow]) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    methods = sorted({r.method for r in rows})
    q_values = sorted({r.q_max for r in rows})
    for m in methods:
        for q in q_values:
            cur = [r for r in rows if r.method == m and r.q_max == q]
            if not cur:
                continue
            y = np.array([r.is_error for r in cur], dtype=np.int64)
            s = np.array([r.risk_score for r in cur], dtype=np.float64)
            out[(m, q)] = _auc(y, s)
    return out


def counter_evidence_controls(
    x_test: np.ndarray,
    y_test: np.ndarray,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    neutralizer: Neutralizer,
    cfg: BeaconConfig,
    max_samples: int = 256,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = min(max_samples, x_test.shape[0])

    gains_sminus = []
    gains_random = []
    gains_lowabs = []
    gains_topabs = []
    fix_sminus = []
    fix_random = []
    fix_lowabs = []
    fix_topabs = []
    frag_flags = []

    for i in range(n):
        x = x_test[i]
        r = run_beacon_refine(logits_fn, x, neutralizer, cfg)

        leaf_meta = r.metadata.get("leaf_components", [])
        leaf_deltas = r.metadata.get("leaf_deltas", [])
        if not leaf_meta or not leaf_deltas:
            continue

        leaves = [
            Component(cid=t[0], t0=t[1], t1=t[2], c0=t[3], c1=t[4])
            for t in leaf_meta
        ]
        deltas = np.array(leaf_deltas, dtype=np.float64)

        k = len(r.s_minus)
        if k <= 0 or len(leaves) < k:
            continue

        y_hat = r.y_hat
        m0 = r.m0
        frag_flags.append(int(r.rho_b_cost <= 0.25 or r.rho_b <= 2))
        pred0 = y_hat
        true_y = int(y_test[i])

        def margin(z: np.ndarray) -> float:
            lg = logits_fn(z)
            return float(lg[y_hat] - np.max(np.delete(lg, y_hat)))

        sm_comp = [s.component for s in r.s_minus]
        g_sminus = margin(neutralizer(x, sm_comp)) - m0
        gains_sminus.append(g_sminus)
        pred_sminus = int(np.argmax(logits_fn(neutralizer(x, sm_comp))))
        fix_sminus.append(int(pred0 != true_y and pred_sminus == true_y))

        idx_rand = rng.choice(len(leaves), size=k, replace=False)
        rand_comp = [leaves[j] for j in idx_rand]
        g_rand = margin(neutralizer(x, rand_comp)) - m0
        gains_random.append(g_rand)
        pred_rand = int(np.argmax(logits_fn(neutralizer(x, rand_comp))))
        fix_random.append(int(pred0 != true_y and pred_rand == true_y))

        idx_low = np.argsort(np.abs(deltas))[:k]
        low_comp = [leaves[j] for j in idx_low]
        g_low = margin(neutralizer(x, low_comp)) - m0
        gains_lowabs.append(g_low)
        pred_low = int(np.argmax(logits_fn(neutralizer(x, low_comp))))
        fix_lowabs.append(int(pred0 != true_y and pred_low == true_y))

        idx_top = np.argsort(-np.abs(deltas))[:k]
        top_comp = [leaves[j] for j in idx_top]
        g_top = margin(neutralizer(x, top_comp)) - m0
        gains_topabs.append(g_top)
        pred_top = int(np.argmax(logits_fn(neutralizer(x, top_comp))))
        fix_topabs.append(int(pred0 != true_y and pred_top == true_y))

    def mean_or_nan(v: list[float]) -> float:
        return float(np.mean(v)) if v else float("nan")

    def rate_or_nan(v: list[int]) -> float:
        return float(np.mean(v)) if v else float("nan")

    frag_idx = np.array(frag_flags, dtype=np.int64)
    fix_s = np.array(fix_sminus, dtype=np.int64)
    fix_r = np.array(fix_random, dtype=np.int64)
    fix_l = np.array(fix_lowabs, dtype=np.int64)
    fix_t = np.array(fix_topabs, dtype=np.int64)

    def frag_rate(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        if frag_idx.size != arr.size:
            return float("nan")
        mask = frag_idx == 1
        if mask.sum() == 0:
            return float("nan")
        return float(arr[mask].mean())

    return {
        "ce_gain_sminus": mean_or_nan(gains_sminus),
        "ce_gain_random": mean_or_nan(gains_random),
        "ce_gain_low_abs": mean_or_nan(gains_lowabs),
        "ce_gain_top_abs": mean_or_nan(gains_topabs),
        "ce_fix_rate_sminus": rate_or_nan(fix_sminus),
        "ce_fix_rate_random": rate_or_nan(fix_random),
        "ce_fix_rate_low_abs": rate_or_nan(fix_lowabs),
        "ce_fix_rate_top_abs": rate_or_nan(fix_topabs),
        "ce_fix_rate_sminus_fragile": frag_rate(fix_s),
        "ce_fix_rate_random_fragile": frag_rate(fix_r),
        "ce_fix_rate_low_abs_fragile": frag_rate(fix_l),
        "ce_fix_rate_top_abs_fragile": frag_rate(fix_t),
        "n_used": float(len(gains_sminus)),
    }


def evaluate_claims(
    rows_k0_8: Sequence[RiskEvalRow],
    rows_k0_16: Sequence[RiskEvalRow] | None,
    ce_controls: dict[str, float],
) -> ClaimReport:
    a8 = summarize_auroc(rows_k0_8)
    a16 = summarize_auroc(rows_k0_16) if rows_k0_16 else {}

    q_small = [8, 16, 32]

    # H1: BEACON-refine better than confidence/entropy/negative_margin for Q<=32
    h1_checks = []
    for q in q_small:
        br = a8.get(("beacon_refine", q), float("nan"))
        for b in ["confidence", "entropy", "negative_margin"]:
            bb = a8.get((b, 0), float("nan"))
            if np.isfinite(br) and np.isfinite(bb):
                h1_checks.append(br > bb)
        for b in ["saliency_topk", "ig_topk"]:
            bb = a8.get((b, q), float("nan"))
            if np.isfinite(br) and np.isfinite(bb):
                h1_checks.append(br > bb)
    h1_pass = bool(h1_checks and all(h1_checks))

    # H2: refine better than flat on all tested Q
    h2_checks = []
    for q in sorted({q for (_, q) in a8.keys() if q > 0}):
        br = a8.get(("beacon_refine", q), float("nan"))
        bf = a8.get(("beacon_flat", q), float("nan"))
        if np.isfinite(br) and np.isfinite(bf):
            h2_checks.append(br > bf)
    h2_pass = bool(h2_checks and all(h2_checks))

    # H3: S- gain better than random and low-abs controls
    sminus = ce_controls.get("ce_gain_sminus", float("nan"))
    random = ce_controls.get("ce_gain_random", float("nan"))
    lowabs = ce_controls.get("ce_gain_low_abs", float("nan"))
    fix_s = ce_controls.get("ce_fix_rate_sminus_fragile", float("nan"))
    fix_r = ce_controls.get("ce_fix_rate_random_fragile", float("nan"))
    fix_l = ce_controls.get("ce_fix_rate_low_abs_fragile", float("nan"))
    h3_pass = bool(
        np.isfinite(sminus)
        and np.isfinite(random)
        and np.isfinite(lowabs)
        and sminus > random
        and sminus > lowabs
        and np.isfinite(fix_s)
        and np.isfinite(fix_r)
        and np.isfinite(fix_l)
        and fix_s >= fix_r
        and fix_s >= fix_l
    )

    # H4: stability K0=8 vs K0=16 for beacon_refine AUROC-vs-Q
    diffs = []
    if a16:
        for q in [8, 16, 32, 64]:
            v8 = a8.get(("beacon_refine", q), float("nan"))
            v16 = a16.get(("beacon_refine", q), float("nan"))
            if np.isfinite(v8) and np.isfinite(v16):
                diffs.append(abs(v8 - v16))
    h4_pass = bool(diffs and max(diffs) <= 0.05)

    # H5: BEACON better than budgeted Shapley-like at small Q
    h5_checks = []
    for q in q_small:
        br = a8.get(("beacon_refine", q), float("nan"))
        sh = a8.get(("budgeted_shapley_like", q), float("nan"))
        if np.isfinite(br) and np.isfinite(sh):
            h5_checks.append(br > sh)
    h5_pass = bool(h5_checks and all(h5_checks))

    details = {
        "auroc_k0_8": {f"{m}@Q{q}": v for (m, q), v in a8.items()},
        "auroc_k0_16": {f"{m}@Q{q}": v for (m, q), v in a16.items()},
        "ce_controls": ce_controls,
        "h4_evaluable": bool(a16),
    }

    return ClaimReport(
        h1_pass=h1_pass,
        h2_pass=h2_pass,
        h3_pass=h3_pass,
        h4_pass=h4_pass,
        h5_pass=h5_pass,
        details=details,
    )


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)
