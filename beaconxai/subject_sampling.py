"""Subject-aware sampling and synthetic injection for OFFLINE XAI benchmarks.

Ground-truth class labels are used to construct a known artificial change;
model selection, Q0/Q16 ranking and the query gate must never receive them.
"""
from __future__ import annotations

import numpy as np


def balanced_subject_sample(subjects: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Sample in round-robin subject order; retain all rare-subject windows."""
    s = np.asarray(subjects)
    if s.ndim != 1 or not len(s):
        raise ValueError("subjects must be a nonempty vector")
    groups = np.unique(s)
    if limit <= 0 or limit >= len(s):
        return np.arange(len(s), dtype=int)
    if limit < len(groups):
        raise ValueError("sample limit too small to represent every subject")
    rng = np.random.default_rng(seed)
    buckets = [rng.permutation(np.flatnonzero(s == group)) for group in groups]
    offsets = np.zeros(len(groups), dtype=int)
    indices = []
    while len(indices) < limit:
        for k, bucket in enumerate(buckets):
            if offsets[k] < len(bucket):
                indices.append(int(bucket[offsets[k]]))
                offsets[k] += 1
                if len(indices) == limit:
                    break
    return np.array(sorted(indices), dtype=int)


def inject_subject_balanced(x: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                            components, fraction: float, seed: int):
    """Insert one known other-class fragment in every represented subject.

    Exactly max(round(fraction*N), number_of_subjects) examples are injected,
    up to N; allocation is approximately proportional to subject size while
    guaranteeing >=1 per subject. Donors come ONLY from the evaluation pool
    and are selected using labels for synthetic benchmark construction, not
    the explanation algorithm. The original array is left untouched.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    s = np.asarray(subjects)
    if x.ndim != 3 or y.shape != (len(x),) or s.shape != (len(x),) or not len(x):
        raise ValueError("expected X[N,T,C], y[N] and subjects[N]")
    if not np.isfinite(x).all() or not np.isfinite(fraction) or not (0 < fraction <= 1):
        raise ValueError("invalid input values or injection fraction")
    if not components or np.unique(y).size < 2:
        raise ValueError("need components and at least two evaluation classes")
    groups, counts = np.unique(s, return_counts=True)
    target = min(len(x), max(len(groups), int(round(float(fraction) * len(x)))))
    allocation = np.ones(len(groups), dtype=int)
    desired = target * counts / len(x)
    for _ in range(target - len(groups)):
        possible = allocation < counts
        deficits = np.where(possible, desired - allocation, -np.inf)
        allocation[int(np.argmax(deficits))] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for group, n in zip(groups, allocation):
        pool = np.flatnonzero(s == group)
        selected.extend(rng.choice(pool, size=int(n), replace=False).tolist())
    edited = x.copy()
    injected = np.zeros(len(x), dtype=np.int64)
    truth = np.full(len(x), -1, dtype=np.int64)
    class_donors = {lab: np.flatnonzero(y != lab) for lab in np.unique(y)}
    for i in selected:
        pool = class_donors[y[i]]
        j = int(rng.choice(pool))
        component_id = int(rng.integers(0, len(components)))
        c = components[component_id]
        edited[i, c.t0:c.t1, c.c0:c.c1] = x[j, c.t0:c.t1, c.c0:c.c1]
        injected[i] = 1
        truth[i] = component_id
    return edited, injected, truth
