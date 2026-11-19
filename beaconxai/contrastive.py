"""Budgeted contrastive time-series auditing: experimental, model-agnostic.

The model is fixed. All class profiles are fitted on training data only.
Each evaluated perturbation counts as one extra model inference, even if batched.
This module does NOT claim to recover causal or physical class boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np


@dataclass(frozen=True)
class Component:
    group: int
    time_bin: int
    t0: int
    t1: int
    c0: int
    c1: int


@dataclass
class AuditResult:
    source_class: int
    rival_class: int
    original_gap: float
    component_ids: list[int]
    signed_gap_drops: list[float]
    final_gaps: list[float]
    target_flips: list[bool]
    calls: int
    prior: np.ndarray
    scores: np.ndarray

    def ranked(self) -> list[int]:
        """Only tested components are ranked (unqueried components score -inf)."""
        return list(np.argsort(-self.scores, kind="stable")[:self.calls].astype(int))


def make_components(shape: tuple[int, int], time_bins: int, group_mode: str = "pamap3") -> list[Component]:
    t_len, channels = map(int, shape)
    if not (1 <= time_bins <= t_len):
        raise ValueError("time_bins must be between 1 and sample length")
    if group_mode == "pamap3":
        if channels != 9:
            raise ValueError("pamap3 requires nine channel features (three 3-axis sensors)")
        c_slices = [(0, 3), (3, 6), (6, 9)]
    elif group_mode == "per_channel":
        c_slices = [(j, j + 1) for j in range(channels)]
    else:
        raise ValueError("unknown group_mode")
    edges = np.linspace(0, t_len, time_bins + 1, dtype=int)
    return [Component(g, b, int(edges[b]), int(edges[b+1]), c0, c1)
            for g, (c0, c1) in enumerate(c_slices)
            for b in range(time_bins)]


def fit_class_profiles(x_train: np.ndarray, y_train: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Class-conditional mean and dispersion; training split only."""
    if x_train.ndim != 3 or y_train.ndim != 1 or len(x_train) != len(y_train):
        raise ValueError("expected X [n,time,channels] and y [n]")
    result = {}
    for lab in np.unique(y_train):
        samples = x_train[y_train == lab].astype(float)
        result[int(lab)] = (samples.mean(axis=0), samples.var(axis=0))
    return result


def contrastive_prior(x: np.ndarray, a: int, b: int,
                       profiles: Mapping[int, tuple[np.ndarray, np.ndarray]],
                       components: list[Component], var_floor: float = 0.5) -> np.ndarray:
    """Cheap, *heuristic* discrimination proxy, not a verified model attribution.

    Signed evidence >0: locally resembles class a more than class b under
    diagonal class-conditioned squared distance. Magnitude ranks probes.
    """
    mu_a, va = profiles[a]
    mu_b, vb = profiles[b]
    if mu_a.shape != x.shape or mu_b.shape != x.shape:
        raise ValueError("class profiles and sample shape differ")
    out = []
    for c in components:
        s = np.s_[c.t0:c.t1, c.c0:c.c1]
        z = x[s]
        scale = np.maximum((va[s] + vb[s]) / 2, var_floor)
        score = np.mean(((z - mu_b[s])**2 - (z - mu_a[s])**2) / scale)
        out.append(float(score))
    return np.array(out, dtype=float)


def neutralize_interp(x: np.ndarray, c: Component) -> np.ndarray:
    """Matches the repository's interpolation boundary behavior for fair comparisons."""
    y = x.copy()
    if c.t0 > 0 and c.t1 < len(x):
        for k in range(c.c0, c.c1):
            y[c.t0:c.t1, k] = np.linspace(y[c.t0-1, k], y[c.t1, k], c.t1-c.t0, endpoint=False)
    else:
        y[c.t0:c.t1, c.c0:c.c1] = 0.0
    return y


def margin(logits: np.ndarray, a: int, b: int) -> float:
    """Fixed A-vs-B log-score gap. Never change competitor after intervention."""
    return float(logits[a] - logits[b])


def _order_from_prior(prior: np.ndarray, budget: int, seed: int,
                      explore_fraction: float = 0.25) -> list[int]:
    """Exploit high magnitude contrast, reserve some probes for exploration.

    Selection uses only X and *training* profiles; zero model queries.
    Avoid an evaluation-ground-truth leak: indices of injected changes are not used.
    """
    n = len(prior)
    if not (0 <= explore_fraction <= 1):
        raise ValueError("explore_fraction must be in [0, 1]")
    if not (1 <= budget <= n):
        raise ValueError("budget must be between 1 and number of components")
    n_explore = min(budget, int(round(budget * explore_fraction)))
    n_exploit = budget - n_explore
    ranked = np.argsort(-np.abs(prior), kind="stable")
    exploit = ranked[:n_exploit].astype(int).tolist()
    remainder = ranked[n_exploit:]
    rng = np.random.default_rng(seed)
    explore = rng.choice(remainder, size=n_explore, replace=False).astype(int).tolist() if n_explore else []
    return exploit + explore


def selected_audit(x: np.ndarray, logits_fn: Callable[[np.ndarray], np.ndarray],
                   profiles: Mapping[int, tuple[np.ndarray, np.ndarray]],
                   components: list[Component], budget: int, seed: int,
                   selection: str = "contrastive", explore_fraction: float = 0.25,
                   fixed_source: int | None = None, fixed_rival: int | None = None,
                   logits0: np.ndarray | None = None) -> AuditResult:
    """Do the same single-component perturbation for contrastive and uniform strategies.

    logits0 is shared with other methods if supplied; baseline query is outside
    the *additional* call budget for every method. Source and rival are pinned.
    """
    if x.ndim != 2:
        raise ValueError("x must be [time,channels]")
    baseline = np.asarray(logits_fn(x) if logits0 is None else logits0, dtype=float)
    a = int(np.argmax(baseline) if fixed_source is None else fixed_source)
    alternatives = baseline.copy()
    alternatives[a] = -np.inf
    b = int(np.argmax(alternatives) if fixed_rival is None else fixed_rival)
    if a == b:
        raise ValueError("source and rival must differ")
    prior = contrastive_prior(x, a, b, profiles, components)
    if selection == "contrastive":
        order = _order_from_prior(prior, budget, seed, explore_fraction)
    elif selection == "uniform":
        order = np.random.default_rng(seed).choice(len(components), size=budget, replace=False).astype(int).tolist()
    elif selection == "prior_only":
        # Deliberate zero-query control: use a separate API for it below.
        raise ValueError("use prior_ranking for a zero-query baseline")
    else:
        raise ValueError("selection must be 'contrastive' or 'uniform'")
    initial_gap = margin(baseline, a, b)
    drops, gaps, flips = [], [], []
    scores = np.full(len(components), -np.inf, dtype=float)
    for idx in order:
        modified = neutralize_interp(x, components[idx])
        new_logits = np.asarray(logits_fn(modified), dtype=float)
        g = margin(new_logits, a, b)
        drop = initial_gap - g
        drops.append(drop)
        gaps.append(g)
        flips.append(int(np.argmax(new_logits)) == b)
        scores[idx] = abs(drop)  # localization: magnitude, not causality
    return AuditResult(a, b, initial_gap, order, drops, gaps, flips, len(order), prior, scores)


def prior_ranking(prior: np.ndarray) -> list[int]:
    """Zero-query contrastive profile baseline; do not call this BEACON."""
    return np.argsort(-np.abs(prior), kind="stable").astype(int).tolist()


def summarize_hotspots(audits: list[AuditResult], components: list[Component],
                       min_count: int = 3) -> list[dict]:
    """Descriptive aggregation (not yet validated decision rules or class boundaries).

    Only feed TRAINING-SPLIT audits to this function when building prototypes.
    """
    groups: dict[tuple[int, int, int], list[float]] = {}
    for audit in audits:
        for idx in audit.ranked()[:3]:
            signed = audit.signed_gap_drops[audit.component_ids.index(idx)]
            key = (audit.source_class, audit.rival_class, idx)
            groups.setdefault(key, []).append(float(signed))
    rows = []
    for (a, b, idx), vals in groups.items():
        if len(vals) < min_count:
            continue
        c = components[idx]
        rows.append({"source": a, "rival": b, "component": idx,
                     "group": c.group, "time_bin": c.time_bin,
                     "count": len(vals), "mean_gap_drop": float(np.mean(vals)),
                     "positive_fraction": float(np.mean(np.array(vals) > 0))})
    return sorted(rows, key=lambda r: (-r["count"], -abs(r["mean_gap_drop"])))


@dataclass
class BoundaryProbe:
    """Observed targeted A->B changes among Q one-component interventions.

    These are *local probes*, not a mathematically recovered global boundary.
    """
    source_class: int
    target_class: int
    baseline_gap: float
    checked: list[int]
    gap_after: list[float]
    predicted_after: list[int]
    best_component: int
    best_target_gap_reduction: float
    flip_to_target: bool
    extra_model_calls: int


def targeted_prototype_probe(x: np.ndarray, logits_fn: Callable[[np.ndarray], np.ndarray],
                             profiles: Mapping[int, tuple[np.ndarray, np.ndarray]],
                             components: list[Component], budget: int, seed: int,
                             selection: str = "contrastive", logits0: np.ndarray | None = None,
                             taper_points: int = 2, fixed_target: int | None = None) -> BoundaryProbe:
    """Replace one component with its TRAINING-CLASS B mean and evaluate class switch.

    Both contrastive and uniform use the IDENTICAL intervention operator.
    A patch from a mean prototype is not guaranteed to be a physically realizable
    time series. Actual class-transition validity requires external validation.
    """
    baseline = np.asarray(logits_fn(x) if logits0 is None else logits0, dtype=float)
    a = int(np.argmax(baseline))
    alts = baseline.copy(); alts[a] = -np.inf
    b = int(np.argmax(alts) if fixed_target is None else fixed_target)
    if b == a:
        raise ValueError("target class must differ from predicted class")
    if b not in profiles:
        raise ValueError(f"no training profile available for class {b}")
    prior = contrastive_prior(x, a, b, profiles, components)
    if selection == "contrastive":
        checked = _order_from_prior(prior, budget, seed)
    elif selection == "uniform":
        checked = np.random.default_rng(seed).choice(len(components),budget,replace=False).astype(int).tolist()
    else:
        raise ValueError(selection)
    gap0 = margin(baseline,a,b)
    gaps, preds = [], []
    for idx in checked:
        c = components[idx]
        edited = x.copy()
        s = np.s_[c.t0:c.t1,c.c0:c.c1]
        target_patch = profiles[b][0][s]
        length = c.t1-c.t0
        taper = min(max(taper_points,0),length//2)
        weights = np.ones(length,dtype=float)
        if taper:
            weights[:taper] = np.linspace(0,1,taper+2)[1:-1]
            weights[-taper:] = np.linspace(1,0,taper+2)[1:-1]
        edited[s] = x[s]*(1-weights[:,None]) + target_patch*weights[:,None]
        new_logits = np.asarray(logits_fn(edited),dtype=float)
        gaps.append(margin(new_logits,a,b))
        preds.append(int(np.argmax(new_logits)))
    j=int(np.argmin(gaps))
    return BoundaryProbe(a,b,gap0,checked,gaps,preds,checked[j],
                         float(gap0-gaps[j]),bool(any(p==b for p in preds)),len(checked))
