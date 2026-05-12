from __future__ import annotations

import numpy as np

from .neutralization import Neutralizer
from .partition import components_cost, make_initial_partition, make_initial_partition_time
from .types import Component
from .core import BeaconAudit
from .types import BaseScores, BeaconConfig, LogitFn


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / np.sum(e)


def base_scores(model_logits: LogitFn, x: np.ndarray) -> BaseScores:
    logits = model_logits(x)
    probs = softmax(logits)
    y = int(np.argmax(logits))
    confidence = float(probs[y])
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    margin = float(logits[y] - np.max(np.delete(logits, y)))
    return BaseScores(confidence=confidence, entropy=entropy, margin=margin)


def run_beacon_refine(model_logits: LogitFn, x: np.ndarray, neutralizer: Neutralizer, cfg: BeaconConfig):
    return BeaconAudit(model_logits, neutralizer, cfg).audit(x)


def run_beacon_flat(model_logits: LogitFn, x: np.ndarray, neutralizer: Neutralizer, cfg: BeaconConfig):
    flat_cfg = BeaconConfig(**{**cfg.__dict__, "refinement_policy": "none"})
    return BeaconAudit(model_logits, neutralizer, flat_cfg).audit(x)


def run_uniform_refinement(model_logits: LogitFn, x: np.ndarray, neutralizer: Neutralizer, cfg: BeaconConfig):
    uniform_cfg = BeaconConfig(**{**cfg.__dict__, "refinement_policy": "uniform"})
    return BeaconAudit(model_logits, neutralizer, uniform_cfg).audit(x)


def run_shapley_like_risk(
    model_logits: LogitFn,
    x: np.ndarray,
    neutralizer: Neutralizer,
    cfg: BeaconConfig,
    seed: int = 42,
) -> tuple[float, int]:
    """
    Budgeted Shapley-like approximation.
    Returns (risk_score, q_used).
    """
    rng = np.random.default_rng(seed)
    p0 = _make_partition_from_cfg(x, cfg)
    if not p0:
        return 0.5, 0

    logits0 = model_logits(x)
    y_hat = int(np.argmax(logits0))
    m0 = _margin_from_logits(logits0, y_hat)
    total_points = x.shape[0] * x.shape[1]

    contrib_sum = {c.cid: 0.0 for c in p0}
    contrib_cnt = {c.cid: 0 for c in p0}
    by_id = {c.cid: c for c in p0}

    q_used = 0
    cids = [c.cid for c in p0]

    while q_used + 2 <= cfg.q_max:
        g_id = cids[int(rng.integers(0, len(cids)))]
        others = [cid for cid in cids if cid != g_id]
        if others:
            mask = rng.random(len(others)) < 0.5
            a_ids = [cid for cid, keep in zip(others, mask) if keep]
        else:
            a_ids = []

        a = [by_id[cid] for cid in a_ids]
        a_plus = a + [by_id[g_id]]

        m_without = _margin(model_logits, neutralizer(x, a), y_hat)
        m_with = _margin(model_logits, neutralizer(x, a_plus), y_hat)
        marginal = m_without - m_with

        contrib_sum[g_id] += marginal
        contrib_cnt[g_id] += 1
        q_used += 2

    est = []
    for cid in cids:
        if contrib_cnt[cid] > 0:
            est.append((by_id[cid], contrib_sum[cid] / contrib_cnt[cid]))
        else:
            est.append((by_id[cid], 0.0))

    support = [(c, v) for c, v in est if v > 0]
    support.sort(key=lambda z: z[1], reverse=True)

    if not support:
        return 0.5, q_used

    # Additive proxy (no extra queries left for true cumulative recomputation).
    accum = 0.0
    used_components: list[Component] = []
    rho_cost = 1.0
    for comp, val in support:
        accum += val
        used_components.append(comp)
        if accum >= max(m0, 0.0):
            rho_cost = components_cost(used_components, total_points)
            break
    else:
        rho_cost = components_cost(used_components, total_points)

    risk = 1.0 / (1.0 + rho_cost)
    return float(risk), q_used


def run_saliency_topk_risk(
    model_logits: LogitFn,
    x: np.ndarray,
    neutralizer: Neutralizer,
    cfg: BeaconConfig,
    margin_gradient_fn=None,
) -> tuple[float, int]:
    p0 = _make_partition_from_cfg(x, cfg)
    if not p0:
        return 0.5, 0

    logits0 = model_logits(x)
    y_hat = int(np.argmax(logits0))
    q_used = 0

    if margin_gradient_fn is not None:
        grad = margin_gradient_fn(x, y_hat)
        q_used += 1  # gradient-equivalent query cost
        attr = grad * x
        scores = [(c, float(attr[c.t0 : c.t1, c.c0 : c.c1].sum())) for c in p0]
    else:
        # Finite-difference fallback
        h = 1e-2
        scores = []
        for c in p0:
            if q_used + 2 > cfg.q_max:
                break
            xp = x.copy()
            xm = x.copy()
            xp[c.t0 : c.t1, c.c0 : c.c1] += h
            xm[c.t0 : c.t1, c.c0 : c.c1] -= h
            mp = _margin(model_logits, xp, y_hat)
            mm = _margin(model_logits, xm, y_hat)
            q_used += 2
            scores.append((c, (mp - mm) / (2.0 * h)))

    return _risk_from_scores(model_logits, x, neutralizer, y_hat, scores, cfg.q_max, q_used)


def run_ig_topk_risk(
    model_logits: LogitFn,
    x: np.ndarray,
    neutralizer: Neutralizer,
    cfg: BeaconConfig,
    margin_gradient_fn=None,
    steps: int = 8,
) -> tuple[float, int]:
    p0 = _make_partition_from_cfg(x, cfg)
    if not p0:
        return 0.5, 0

    logits0 = model_logits(x)
    y_hat = int(np.argmax(logits0))
    q_used = 0

    x0 = np.zeros_like(x)
    dx = x - x0

    if margin_gradient_fn is not None:
        steps_eff = max(1, min(steps, cfg.q_max))
        grads = []
        for s in range(1, steps_eff + 1):
            alpha = s / steps_eff
            xs = x0 + alpha * dx
            grads.append(margin_gradient_fn(xs, y_hat))
            q_used += 1  # gradient-equivalent
        avg_grad = np.mean(np.stack(grads, axis=0), axis=0)
        ig = dx * avg_grad
        scores = [(c, float(ig[c.t0 : c.t1, c.c0 : c.c1].sum())) for c in p0]
    else:
        # Fallback: path-occlusion proxy
        steps_eff = max(1, min(steps, max(1, cfg.q_max // max(1, len(p0)))))
        scores = []
        for c in p0:
            vals = []
            for s in range(1, steps_eff + 1):
                if q_used + 1 > cfg.q_max:
                    break
                alpha = s / steps_eff
                xs = x0 + alpha * dx
                vals.append(_margin(model_logits, xs, y_hat) - _margin(model_logits, neutralizer(xs, [c]), y_hat))
                q_used += 1
            scores.append((c, float(np.mean(vals)) if vals else 0.0))

    return _risk_from_scores(model_logits, x, neutralizer, y_hat, scores, cfg.q_max, q_used)


def run_full_occlusion_risk(
    model_logits: LogitFn,
    x: np.ndarray,
    neutralizer: Neutralizer,
    cfg: BeaconConfig,
    max_components: int = 10000,
) -> tuple[float, int] | None:
    t, d = x.shape
    total = t * d
    if total > max_components:
        return None

    logits0 = model_logits(x)
    y_hat = int(np.argmax(logits0))

    components = []
    for ti in range(t):
        for di in range(d):
            components.append(Component(cid=f"u_{ti}_{di}", t0=ti, t1=ti + 1, c0=di, c1=di + 1))

    # one-occlusion scan
    scores = []
    q_used = 0
    m0 = _margin_from_logits(logits0, y_hat)
    for c in components:
        if q_used + 1 > cfg.q_max:
            break
        m = _margin(model_logits, neutralizer(x, [c]), y_hat)
        q_used += 1
        scores.append((c, m0 - m))

    return _risk_from_scores(model_logits, x, neutralizer, y_hat, scores, cfg.q_max, q_used)


def run_simple_counterfactual_risk(
    model_logits: LogitFn,
    x: np.ndarray,
    neutralizer: Neutralizer,
    cfg: BeaconConfig,
) -> tuple[float, int]:
    p0 = _make_partition_from_cfg(x, cfg)
    if not p0:
        return 0.5, 0

    logits0 = model_logits(x)
    y_hat = int(np.argmax(logits0))
    m0 = _margin_from_logits(logits0, y_hat)
    total_points = x.shape[0] * x.shape[1]

    # one-step ranking by strongest class-destabilizing effect
    ranked = []
    q_used = 0
    for c in p0:
        if q_used + 1 > cfg.q_max:
            break
        m = _margin(model_logits, neutralizer(x, [c]), y_hat)
        ranked.append((c, m0 - m))
        q_used += 1
    ranked.sort(key=lambda z: z[1], reverse=True)

    selected: list[Component] = []
    for c, _ in ranked:
        if q_used + 1 > cfg.q_max:
            break
        selected.append(c)
        m = _margin(model_logits, neutralizer(x, selected), y_hat)
        q_used += 1
        if m <= 0:
            cost = components_cost(selected, total_points)
            return float(1.0 / (1.0 + cost)), q_used

    if not selected:
        return 0.5, q_used
    cost = components_cost(selected, total_points)
    return float(1.0 / (1.0 + cost)), q_used


def _risk_from_scores(
    model_logits: LogitFn,
    x: np.ndarray,
    neutralizer: Neutralizer,
    y_hat: int,
    scores: list[tuple[Component, float]],
    q_max: int,
    q_used: int,
) -> tuple[float, int]:
    total_points = x.shape[0] * x.shape[1]
    support = [(c, s) for c, s in scores if s > 0]
    support.sort(key=lambda z: z[1], reverse=True)
    if not support:
        return 0.5, q_used

    selected: list[Component] = []
    checked_cost = 1.0
    for c, _ in support:
        if q_used + 1 > q_max:
            break
        selected.append(c)
        m = _margin(model_logits, neutralizer(x, selected), y_hat)
        q_used += 1
        checked_cost = components_cost(selected, total_points)
        if m <= 0:
            return float(1.0 / (1.0 + checked_cost)), q_used
    return float(1.0 / (1.0 + checked_cost)), q_used


def _make_partition_from_cfg(x: np.ndarray, cfg: BeaconConfig) -> list[Component]:
    if cfg.partition_mode == "time_only":
        return make_initial_partition_time(x.shape[0], x.shape[1], cfg.k0)
    return make_initial_partition(x.shape[0], x.shape[1], cfg.k0)


def _margin(model_logits: LogitFn, x: np.ndarray, y_hat: int) -> float:
    return _margin_from_logits(model_logits(x), y_hat)


def _margin_from_logits(logits: np.ndarray, y_hat: int) -> float:
    ref = float(logits[y_hat])
    alt = float(np.max(np.delete(logits, y_hat)))
    return ref - alt
