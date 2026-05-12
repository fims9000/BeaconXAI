from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from .core import BeaconAudit
from .neutralization import Neutralizer
from .partition import components_cost
from .types import BeaconConfig, Component


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    den = np.sqrt((vx * vx).sum() * (vy * vy).sum())
    if den <= 0:
        return float("nan")
    return float((vx * vy).sum() / den)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    return ranks


def rho_exact_or_beam_cost(
    model_logits: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    y_hat: int,
    support: list[Component],
    neutralizer: Neutralizer,
    max_exact_support: int = 12,
    beam_width: int = 128,
    max_k: int = 8,
) -> float:
    total_points = x.shape[0] * x.shape[1]

    def margin(z: np.ndarray) -> float:
        lg = model_logits(z)
        return float(lg[y_hat] - np.max(np.delete(lg, y_hat)))

    if not support:
        return 1.0

    n = len(support)
    if n <= max_exact_support:
        best = None
        for k in range(1, n + 1):
            for idxs in combinations(range(n), k):
                comps = [support[i] for i in idxs]
                m = margin(neutralizer(x, comps))
                if m <= 0:
                    c = components_cost(comps, total_points)
                    if best is None or c < best:
                        best = c
            if best is not None:
                break
        return float(best if best is not None else 1.0)

    # Beam search for larger supports
    states = [([], 0.0)]
    best = None
    for _k in range(1, max_k + 1):
        cand = []
        for used, _ in states:
            used_set = set(used)
            for i in range(n):
                if i in used_set:
                    continue
                nxt = used + [i]
                comps = [support[j] for j in nxt]
                m = margin(neutralizer(x, comps))
                c = components_cost(comps, total_points)
                if m <= 0:
                    if best is None or c < best:
                        best = c
                cand.append((nxt, c, m))

        cand.sort(key=lambda t: (t[1], t[2]))
        states = [(c[0], c[1]) for c in cand[:beam_width]]
        if best is not None:
            break

    return float(best if best is not None else 1.0)


def run_rho_sanity(
    x_test: np.ndarray,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    neutralizer: Neutralizer,
    cfg: BeaconConfig,
    n_samples: int = 128,
) -> dict[str, float]:
    audit = BeaconAudit(logits_fn, neutralizer, cfg)

    rho_b = []
    rho_exact = []

    m = min(n_samples, x_test.shape[0])
    for i in range(m):
        x = x_test[i]
        r = audit.audit(x)
        leaf_meta = r.metadata.get("leaf_components", [])
        leaf_deltas = r.metadata.get("leaf_deltas", [])
        if not leaf_meta or not leaf_deltas:
            continue

        leaves = [
            Component(cid=t[0], t0=t[1], t1=t[2], c0=t[3], c1=t[4])
            for t in leaf_meta
        ]
        deltas = np.array(leaf_deltas, dtype=np.float64)
        support = [leaf for leaf, d in zip(leaves, deltas) if d > 0]

        rho_b.append(float(r.rho_b_cost))
        rho_exact.append(
            rho_exact_or_beam_cost(
                model_logits=logits_fn,
                x=x,
                y_hat=r.y_hat,
                support=support,
                neutralizer=neutralizer,
            )
        )

    if not rho_b:
        return {"n": 0.0, "spearman": float("nan"), "auroc_rho_b": float("nan"), "auroc_rho_exact": float("nan")}

    rb = np.array(rho_b, dtype=np.float64)
    re = np.array(rho_exact, dtype=np.float64)

    return {
        "n": float(len(rb)),
        "spearman": spearman_corr(rb, re),
        "mean_abs_diff": float(np.mean(np.abs(rb - re))),
    }
