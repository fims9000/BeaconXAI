"""Pre-registered cheap-score shortlist controls for PAMAP time-series patches.

All candidates use the input's cheap TRAIN-class prior and the known
sensor/time partition. Neither injected patch ID nor held-out label enters
selection. Quotas are fixed before evaluating outer test subjects.
"""
from __future__ import annotations
import numpy as np


def stratified_top_q(scores: np.ndarray, components, q: int,
                     by: str = "group") -> list[int]:
    """Top-Q candidates with an explicit per-stratum representation floor.

    For Q >= number of strata: take floor(Q/strata) highest-prior
    components from each sensor group or time bin, then fill remaining
    budget globally by prior. If Q < strata, fall back to global top Q.
    Every selected component is evaluated; none is ranked without a call.
    """
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or len(a) != len(components) or not np.isfinite(a).all():
        raise ValueError("scores must be finite, one per component")
    if not (1 <= q <= len(components)) or by not in ("group", "time"):
        raise ValueError("invalid q or stratification")
    labels = [int(c.group if by == "group" else c.time_bin) for c in components]
    groups = sorted(set(labels))
    if not groups:
        raise ValueError("empty components")
    order = sorted(range(len(a)), key=lambda i: (-a[i], i))
    quota = q // len(groups)
    selected = set()
    if quota:
        for group in groups:
            ids = [i for i in order if labels[i] == group]
            selected.update(ids[:quota])
    return ([i for i in order if i in selected]
            + [i for i in order if i not in selected][:q - len(selected)])
