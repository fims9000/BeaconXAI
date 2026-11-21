"""Within-budget BEACON rank fusion: measured perturbations + cheap TRAIN prior.

Only Q actually perturbed components can be assigned a rank. This is a
fixed-weight research comparator, not an established improved XAI method.
"""
from __future__ import annotations
import numpy as np


def fuse_tested_ranks(cheap_scores: np.ndarray, measured_order: list[int],
                      prior_weight: float = 0.5) -> list[int]:
    """Combine candidate-relative ranks by weighted Borda aggregation.

    The prior ranks use only candidate feature scores; measured ranks come
    from Q actual model evaluations. No unqueried component is admitted.
    Weight 0 reproduces measured order; 1 reproduces prior order restricted
    to Q queried candidates. Both rankings are on identical Q candidates.
    """
    cheap = np.asarray(cheap_scores, dtype=float)
    ids = [int(i) for i in measured_order]
    if (cheap.ndim != 1 or not np.isfinite(cheap).all()
        or not ids or len(set(ids)) != len(ids)
        or any(i < 0 or i >= len(cheap) for i in ids)
        or not np.isfinite(prior_weight) or not 0 <= prior_weight <= 1):
        raise ValueError("invalid prior, candidate order or fusion weight")
    if prior_weight == 0:
        return ids.copy()
    prior_order = sorted(ids, key=lambda i: (-cheap[i], i))
    if prior_weight == 1:
        return prior_order
    pos_prior = {j: i for i, j in enumerate(prior_order)}
    pos_model = {j: i for i, j in enumerate(ids)}
    return sorted(ids, key=lambda j: (
        prior_weight * pos_prior[j] + (1-prior_weight) * pos_model[j],
        pos_prior[j], pos_model[j], j))
