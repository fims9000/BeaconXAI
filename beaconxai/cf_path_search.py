"""Budgeted targeted counterfactual paths for black-box time-series models.

This module cannot guarantee global or physical minimality. A model query is
one evaluated input, even when evaluations are batched. Returned success is
always a model-confirmed switch to the fixed target class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

from .verified_counterfactual import splice, target_margin


@dataclass(frozen=True)
class PathProbe:
    components: tuple[int, ...]
    predicted_class: int
    target_margin: float
    targeted_flip: bool
    relative_l2: float
    stage: str


@dataclass(frozen=True)
class PathSearchResult:
    status: str
    source_class: int
    target_class: int
    verified_edit: tuple[int, ...] | None
    calls: int
    budget: int
    probes: tuple[PathProbe, ...]
    local_minimality_checked: bool

    @property
    def is_valid(self) -> bool:
        return self.status == "verified_target_flip"


def search_counterfactual_path(
    x: np.ndarray,
    baseline: np.ndarray,
    donor: np.ndarray,
    components: Sequence,
    cheap_rank: Sequence[int],
    predict_many: Callable,
    *,
    target_class: int,
    budget: int,
    max_changed_components: int = 8,
    seed_singletons: int = 3,
    taper: int = 2,
) -> PathSearchResult:
    """Use several singleton probes, then grow one verified-target path.

    Stage 1 queries up to seed_singletons in the cheap ranking, and chooses
    the highest fixed target margin. Stage 2 adds further components in the
    predetermined cheap ranking. A target flip is checked against every class.
    After finding a flip, unused model queries may verify and prune 1-removal
    subsets; absence of these audits is explicitly marked unverified.

    All decisions use only input x, frozen scores, actual model responses and
    TRAIN donor; labels and full-Q oracle never enter the search.
    """
    x, donor = np.asarray(x), np.asarray(donor)
    base = np.asarray(baseline, dtype=float)
    order = tuple(int(i) for i in cheap_rank)
    if (
        x.ndim != 2 or donor.shape != x.shape or
        not np.isfinite(x).all() or not np.isfinite(donor).all() or
        base.ndim != 1 or len(base) < 2 or not np.isfinite(base).all()
    ):
        raise ValueError("invalid input, donor or cached scores")
    source = int(np.argmax(base))
    if not (0 <= target_class < len(base)) or source == target_class:
        raise ValueError("target must differ from source")
    if not components or set(order) != set(range(len(components))) or len(order) != len(components):
        raise ValueError("ranking must include every component once")
    if not (budget >= 1 and 1 <= seed_singletons <= budget and
            1 <= max_changed_components):
        raise ValueError("invalid budget or group-size limits")
    max_changed_components = min(max_changed_components, len(components))
    probes: list[PathProbe] = []
    scores: dict[tuple[int, ...], PathProbe] = {}
    dist_scale = max(float(np.linalg.norm(x)), float(np.linalg.norm(donor)), 1e-8)

    def measure(selected: tuple[int, ...], stage: str) -> PathProbe:
        key = tuple(sorted(selected))
        if key in scores:
            return scores[key]
        if len(probes) >= budget:
            raise RuntimeError("budget exhausted")
        changed = splice(x, donor, components, key, taper=taper)
        output = np.asarray(predict_many(changed[None]), dtype=float)
        if output.shape != (1, len(base)) or not np.isfinite(output).all():
            raise ValueError("model must return finite [1,classes] logits")
        cls = int(np.argmax(output[0]))
        p = PathProbe(key, cls, target_margin(output[0], target_class),
                      cls == target_class, float(np.linalg.norm(changed-x) / dist_scale), stage)
        scores[key] = p
        probes.append(p)
        return p

    def done(s: tuple[int, ...] | None, minimal: bool = False) -> PathSearchResult:
        return PathSearchResult(
            "verified_target_flip" if s is not None else "not_found_within_budget",
            source, target_class, s, len(probes), budget, tuple(probes), minimal,
        )

    def minimize(selected: tuple[int, ...]) -> tuple[tuple[int, ...], bool]:
        """Verify inclusion-minimality locally only if all deletions checked."""
        if len(selected) <= 1:
            return selected, True
        current = selected
        while len(current) > 1:
            reduced = False
            for j in current:
                subset = tuple(sorted(v for v in current if v != j))
                if subset not in scores and len(probes) >= budget:
                    return current, False
                p = measure(subset, "prune")
                if p.targeted_flip:
                    current = subset
                    reduced = True
                    break
            if not reduced:
                return current, True
        return current, True

    seed = min(seed_singletons, budget, len(order))
    singles = []
    for idx in order[:seed]:
        p = measure((idx,), "singleton")
        singles.append(p)
        if p.targeted_flip:
            return done((idx,), True)
    if max_changed_components == 1 or len(probes) >= budget:
        return done(None)
    current = max(singles, key=lambda p: p.target_margin).components
    seed_order = [p.components[0] for p in sorted(singles, key=lambda p: -p.target_margin)]
    additions = [j for j in seed_order if j not in current] + [j for j in order[seed:]]
    for idx in additions:
        if len(probes) >= budget or len(current) >= max_changed_components:
            break
        current = tuple(sorted((*current, idx)))
        p = measure(current, "grow")
        if p.targeted_flip:
            best, minimal = minimize(current)
            return done(best, minimal)
    return done(None)
