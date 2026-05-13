from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Sequence

import numpy as np

from .baselines import (
    base_scores,
    run_full_occlusion_risk,
    run_ig_topk_risk,
    run_beacon_flat,
    run_beacon_refine,
    run_saliency_topk_risk,
    run_shapley_like_risk,
    run_simple_counterfactual_risk,
    run_uniform_refinement,
)
from .neutralization import Neutralizer
from .types import BeaconConfig, LocalMetricRow, RiskEvalRow


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = np.sum(y_true == 1)
    if pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(0.0, r - prev_recall)
        prev_recall = r
    return float(ap)


def _rank_norm(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    if len(x) <= 1:
        return np.zeros_like(ranks)
    return ranks / (len(x) - 1)


def evaluate_error_risk(
    x_test: np.ndarray,
    y_test: np.ndarray,
    predict_fn: Callable[[np.ndarray], int],
    logits_fn: Callable[[np.ndarray], np.ndarray],
    neutralizer: Neutralizer,
    base_cfg: BeaconConfig,
    q_values: Sequence[int],
    margin_gradient_fn=None,
    composite_weights: dict[str, float] | None = None,
    methods: set[str] | None = None,
) -> tuple[list[RiskEvalRow], list[LocalMetricRow], list[dict[str, float]]]:
    rows: list[RiskEvalRow] = []
    local_rows: list[LocalMetricRow] = []
    if methods is None:
        methods = {
            "confidence",
            "entropy",
            "negative_margin",
            "beacon_refine",
            "beacon_flat",
            "uniform_refinement",
            "budgeted_shapley_like",
            "saliency_topk",
            "ig_topk",
            "simple_counterfactual",
            "full_occlusion",
            "beacon_composite",
        }

    for i in range(x_test.shape[0]):
        x = x_test[i]
        pred = predict_fn(x)
        is_error = int(pred != int(y_test[i]))

        bs = base_scores(logits_fn, x)
        if "confidence" in methods:
            rows.append(RiskEvalRow(i, is_error, 0, "confidence", 1.0 - bs.confidence, 0, 0))
        if "entropy" in methods:
            rows.append(RiskEvalRow(i, is_error, 0, "entropy", bs.entropy, 0, 0))
        if "negative_margin" in methods:
            rows.append(RiskEvalRow(i, is_error, 0, "negative_margin", -bs.margin, 0, 0))

        for q in q_values:
            # For q <= k0 there is no refinement/fragility budget left, BEACON
            # risk degenerates to censored edge-cases; skip such settings.
            if q <= base_cfg.k0:
                continue
            cfg = replace(base_cfg, q_max=int(q))

            if "beacon_refine" in methods or "beacon_composite" in methods:
                r_ref = run_beacon_refine(logits_fn, x, neutralizer, cfg)
                if "beacon_refine" in methods:
                    rows.append(RiskEvalRow(i, is_error, q, "beacon_refine", r_ref.risk_b, r_ref.q_used, int(r_ref.censored)))
                local_rows.append(
                    LocalMetricRow(
                        sample_id=i,
                        q_max=q,
                        method="beacon_refine",
                        sufficiency_margin=r_ref.sufficiency_margin,
                        sufficiency_kept_class=int(r_ref.sufficiency_kept_class),
                        necessity=r_ref.necessity,
                        counter_evidence_gain=r_ref.counter_evidence_gain,
                        rho_b=r_ref.rho_b,
                        rho_b_cost=r_ref.rho_b_cost,
                        censored=int(r_ref.censored),
                    )
                )

            if "beacon_flat" in methods:
                r_flat = run_beacon_flat(logits_fn, x, neutralizer, cfg)
                rows.append(RiskEvalRow(i, is_error, q, "beacon_flat", r_flat.risk_b, r_flat.q_used, int(r_flat.censored)))
                local_rows.append(
                    LocalMetricRow(
                        sample_id=i,
                        q_max=q,
                        method="beacon_flat",
                        sufficiency_margin=r_flat.sufficiency_margin,
                        sufficiency_kept_class=int(r_flat.sufficiency_kept_class),
                        necessity=r_flat.necessity,
                        counter_evidence_gain=r_flat.counter_evidence_gain,
                        rho_b=r_flat.rho_b,
                        rho_b_cost=r_flat.rho_b_cost,
                        censored=int(r_flat.censored),
                    )
                )

            if "uniform_refinement" in methods:
                r_uni = run_uniform_refinement(logits_fn, x, neutralizer, cfg)
                rows.append(RiskEvalRow(i, is_error, q, "uniform_refinement", r_uni.risk_b, r_uni.q_used, int(r_uni.censored)))
                local_rows.append(
                    LocalMetricRow(
                        sample_id=i,
                        q_max=q,
                        method="uniform_refinement",
                        sufficiency_margin=r_uni.sufficiency_margin,
                        sufficiency_kept_class=int(r_uni.sufficiency_kept_class),
                        necessity=r_uni.necessity,
                        counter_evidence_gain=r_uni.counter_evidence_gain,
                        rho_b=r_uni.rho_b,
                        rho_b_cost=r_uni.rho_b_cost,
                        censored=int(r_uni.censored),
                    )
                )

            if "budgeted_shapley_like" in methods:
                shap_risk, shap_q = run_shapley_like_risk(logits_fn, x, neutralizer, cfg, seed=42 + i + q)
                rows.append(RiskEvalRow(i, is_error, q, "budgeted_shapley_like", shap_risk, shap_q, 0))

            if "saliency_topk" in methods:
                sal_risk, sal_q = run_saliency_topk_risk(
                    logits_fn, x, neutralizer, cfg, margin_gradient_fn=margin_gradient_fn
                )
                rows.append(RiskEvalRow(i, is_error, q, "saliency_topk", sal_risk, sal_q, 0))

            if "ig_topk" in methods:
                ig_risk, ig_q = run_ig_topk_risk(
                    logits_fn, x, neutralizer, cfg, margin_gradient_fn=margin_gradient_fn, steps=8
                )
                rows.append(RiskEvalRow(i, is_error, q, "ig_topk", ig_risk, ig_q, 0))

            if "simple_counterfactual" in methods:
                cf_risk, cf_q = run_simple_counterfactual_risk(logits_fn, x, neutralizer, cfg)
                rows.append(RiskEvalRow(i, is_error, q, "simple_counterfactual", cf_risk, cf_q, 0))

            if "full_occlusion" in methods:
                full = run_full_occlusion_risk(logits_fn, x, neutralizer, cfg)
                if full is not None:
                    full_risk, full_q = full
                    rows.append(RiskEvalRow(i, is_error, q, "full_occlusion", full_risk, full_q, 0))

    if composite_weights is not None and "beacon_composite" in methods:
        _add_composite_rows(rows, local_rows, q_values, composite_weights)

    metrics: list[dict[str, float]] = []
    methods = sorted({r.method for r in rows})
    q_grid = sorted({r.q_max for r in rows})

    for method in methods:
        for q in q_grid:
            cur = [r for r in rows if r.method == method and r.q_max == q]
            if not cur:
                continue
            y = np.array([r.is_error for r in cur], dtype=np.int64)
            s = np.array([r.risk_score for r in cur], dtype=np.float64)

            metrics.append(
                {
                    "method": method,
                    "q_max": float(q),
                    "auroc": _safe_auc(y, s),
                    "auprc": _safe_auprc(y, s),
                    "mean_q_used": float(np.mean([r.q_used for r in cur])),
                    "censored_rate": float(np.mean([r.censored for r in cur])),
                }
            )

    return rows, local_rows, metrics


def _add_composite_rows(
    rows: list[RiskEvalRow],
    local_rows: list[LocalMetricRow],
    q_values: Sequence[int],
    weights: dict[str, float],
) -> None:
    conf_map = {r.sample_id: r.risk_score for r in rows if r.method == "confidence" and r.q_max == 0}
    neg_margin_map = {r.sample_id: r.risk_score for r in rows if r.method == "negative_margin" and r.q_max == 0}
    y_map = {r.sample_id: r.is_error for r in rows if r.method == "confidence" and r.q_max == 0}

    w_beacon = float(weights.get("beacon", 1.0))
    w_conf = float(weights.get("conf", 1.0))
    w_neg_margin = float(weights.get("neg_margin", 0.0))
    w_rho = float(weights.get("rho", 1.0))
    w_nec = float(weights.get("nec", 0.2))
    w_ce = float(weights.get("ce", 0.2))
    w_suff = float(weights.get("suff_bad", 0.5))
    w_cens = float(weights.get("censored", 0.1))

    for q in q_values:
        br = [r for r in rows if r.method == "beacon_refine" and r.q_max == q]
        lm = [r for r in local_rows if r.method == "beacon_refine" and r.q_max == q]
        if not br or not lm:
            continue

        br_map = {r.sample_id: r for r in br}
        lm_map = {r.sample_id: r for r in lm}
        ids = sorted(set(br_map).intersection(lm_map).intersection(conf_map).intersection(neg_margin_map))
        if not ids:
            continue

        risk_beacon = np.array([br_map[i].risk_score for i in ids], dtype=np.float64)
        risk_conf = np.array([conf_map[i] for i in ids], dtype=np.float64)
        risk_neg_margin = np.array([neg_margin_map[i] for i in ids], dtype=np.float64)
        rho = np.array([lm_map[i].rho_b_cost for i in ids], dtype=np.float64)
        nec = np.array([lm_map[i].necessity for i in ids], dtype=np.float64)
        ce = np.array([lm_map[i].counter_evidence_gain for i in ids], dtype=np.float64)
        suff_bad = np.array([-lm_map[i].sufficiency_margin for i in ids], dtype=np.float64)
        cens = np.array(
            [max(br_map[i].censored, lm_map[i].censored) for i in ids],
            dtype=np.float64,
        )

        x_beacon = _rank_norm(risk_beacon)
        x_conf = _rank_norm(risk_conf)
        x_neg_margin = _rank_norm(risk_neg_margin)
        x_rho = _rank_norm(rho)
        x_nec = _rank_norm(nec)
        x_ce = _rank_norm(ce)
        x_suff = _rank_norm(suff_bad)

        score = (
            w_beacon * x_beacon
            + w_conf * x_conf
            + w_neg_margin * x_neg_margin
            + w_rho * x_rho
            + w_nec * x_nec
            + w_ce * x_ce
            + w_suff * x_suff
            + w_cens * cens
        )

        max_q = np.array([br_map[i].q_used for i in ids], dtype=np.float64)
        cens_rate = np.array([br_map[i].censored for i in ids], dtype=np.float64)
        for sid, s, q_used, c in zip(ids, score, max_q, cens_rate):
            rows.append(
                RiskEvalRow(
                    sample_id=int(sid),
                    is_error=int(y_map[sid]),
                    q_max=int(q),
                    method="beacon_composite",
                    risk_score=float(s),
                    q_used=int(q_used),
                    censored=int(c),
                )
            )
