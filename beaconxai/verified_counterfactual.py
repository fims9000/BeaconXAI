"""Verified budget-limited contrastive examples for a fixed black-box model.

One model evaluation means one evaluated perturbed input, not one batch call.
The original output is cached. The donor is an actual TRAINING window from
the target class. Splicing real fragments does not prove physical realism.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence
import numpy as np


def splice(x: np.ndarray, donor: np.ndarray, components: Sequence,
           selected: Sequence[int], taper: int = 2) -> np.ndarray:
    """Combine disjoint tapered patches from ONE donor window."""
    x, donor = np.asarray(x), np.asarray(donor)
    if x.ndim != 2 or donor.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(donor).all():
        raise ValueError("finite same-shaped [T,C] source and donor required")
    if taper < 0 or not isinstance(taper, int):
        raise ValueError("taper must be a nonnegative integer")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected components must be nonempty and distinct")
    out = x.copy()
    changed = np.zeros(x.shape, dtype=bool)
    for idx in selected:
        if idx < 0 or idx >= len(components):
            raise ValueError("component index out of range")
        candidate = components[idx]
        # A logical candidate may edit multiple physically related channels.
        # It still consumes exactly ONE model evaluation after this splice.
        for c in getattr(candidate, "regions", (candidate,)):
            if not (0 <= c.t0 < c.t1 <= len(x) and 0 <= c.c0 < c.c1 <= x.shape[1]):
                raise ValueError("invalid component boundaries")
            sl = np.s_[c.t0:c.t1, c.c0:c.c1]
            if np.any(changed[sl]):
                raise ValueError("overlapping components cannot be combined")
            changed[sl] = True
            n = c.t1 - c.t0
            w = np.ones(n, dtype=float)
            edge = min(taper, n // 2)
            if edge:
                w[:edge] = np.linspace(0, 1, edge + 2)[1:-1]
                w[-edge:] = np.linspace(1, 0, edge + 2)[1:-1]
            out[sl] = x[sl] * (1 - w[:, None]) + donor[sl] * w[:, None]
    return out


def target_margin(scores: np.ndarray, target: int) -> float:
    v = np.asarray(scores, dtype=float)
    if v.ndim != 1 or v.size < 2 or not np.isfinite(v).all() or not 0 <= target < v.size:
        raise ValueError("invalid class scores or target")
    return float(v[target] - np.max(np.delete(v, target)))


@dataclass(frozen=True)
class TestedEdit:
    components: tuple[int, ...]
    predicted_class: int
    target_margin: float
    target_pair_gain: float
    flip_to_target: bool
    relative_l2: float
    interaction_pair: float | None


@dataclass(frozen=True)
class CounterfactualResult:
    source_class: int
    target_class: int
    status: str
    verified_edit: tuple[int, ...] | None
    evaluations: int
    budget: int
    queries: tuple[TestedEdit, ...]
    reference_donor_only: bool = True

    @property
    def is_valid(self) -> bool:
        return self.status == "verified_target_flip"


def find_verified_counterfactual(
    x: np.ndarray, baseline_scores: np.ndarray, donor: np.ndarray,
    components: Sequence, cheap_rank: Sequence[int],
    predict_many: Callable[[np.ndarray], np.ndarray],
    *, target_class: int, budget: int,
    max_changed_components: int = 2, single_fraction: float = 0.5,
    taper: int = 2,
) -> CounterfactualResult:
    """Search singleton edits, then interaction-aware pairs.

    Final validity is checked with actual classifier outputs. No reference
    Q36, test labels, global minimality, or physical realism is assumed.
    """
    base = np.asarray(baseline_scores, dtype=float)
    x = np.asarray(x)
    if base.ndim != 1 or base.size < 2 or not np.isfinite(base).all():
        raise ValueError("invalid cached model output")
    source = int(np.argmax(base))
    if target_class < 0 or target_class >= len(base) or target_class == source:
        raise ValueError("target must differ from original predicted class")
    if not components or budget < 1 or max_changed_components not in (1, 2):
        raise ValueError("invalid search budget or edit limit")
    if not 0 < single_fraction <= 1 or not np.isfinite(single_fraction):
        raise ValueError("invalid single-search fraction")
    rank = tuple(int(j) for j in cheap_rank)
    if len(rank) != len(components) or set(rank) != set(range(len(components))):
        raise ValueError("cheap ranking must be a permutation of all components")
    if np.asarray(donor).shape != x.shape or not np.isfinite(donor).all():
        raise ValueError("invalid training donor")

    first_count = (min(len(rank), budget) if max_changed_components == 1 else
                   min(len(rank), budget, max(1, int(np.ceil(budget * single_fraction)))))
    queries: list[TestedEdit] = []
    pair_gains: dict[int, float] = {}
    base_pair = float(base[target_class] - base[source])

    def probe(ids: tuple[int, ...]) -> TestedEdit:
        edited = splice(x, donor, components, ids, taper=taper)
        outputs = np.asarray(predict_many(edited[None]), dtype=float)
        if outputs.shape != (1, len(base)) or not np.isfinite(outputs).all():
            raise ValueError("model must return finite [1,classes] scores")
        s = outputs[0]
        rec = TestedEdit(
            components=ids, predicted_class=int(np.argmax(s)),
            target_margin=target_margin(s, target_class),
            target_pair_gain=float(s[target_class] - s[source] - base_pair),
            flip_to_target=bool(int(np.argmax(s)) == target_class),
            relative_l2=float(np.linalg.norm(edited - x) /
                              (max(np.linalg.norm(x), np.linalg.norm(donor)) + 1e-8)),
            interaction_pair=None,
        )
        queries.append(rec)
        return rec

    def result(edit: tuple[int, ...] | None) -> CounterfactualResult:
        return CounterfactualResult(
            source, target_class,
            "verified_target_flip" if edit is not None else "not_found_within_budget",
            edit, len(queries), budget, tuple(queries),
        )

    for j in rank[:first_count]:
        rec = probe((j,))
        pair_gains[j] = rec.target_pair_gain
        if rec.flip_to_target:
            return result((j,))
    if max_changed_components == 1 or len(queries) >= budget or first_count < 2:
        return result(None)

    pairs = list(combinations(rank[:first_count], 2))
    pairs.sort(key=lambda ij: (-pair_gains[ij[0]] - pair_gains[ij[1]],
                               rank.index(ij[0]) + rank.index(ij[1]), ij))
    for left, right in pairs:
        if len(queries) >= budget:
            break
        rec = probe((left, right))
        interaction = rec.target_pair_gain - pair_gains[left] - pair_gains[right]
        queries[-1] = TestedEdit(rec.components, rec.predicted_class,
                                 rec.target_margin, rec.target_pair_gain,
                                 rec.flip_to_target, rec.relative_l2, float(interaction))
        if rec.flip_to_target:
            return result((left, right))
    return result(None)


def donor_stability_audit(
    x: np.ndarray, base: np.ndarray, target: int, selected: Sequence[int],
    other_training_donors: Sequence[np.ndarray], components: Sequence,
    predict_many: Callable[[np.ndarray], np.ndarray], *, budget_remaining: int,
    taper: int = 2,
) -> dict:
    """Optional measured donor checks; never a coverage guarantee."""
    if budget_remaining < 0:
        raise ValueError("negative remaining budget")
    if not selected:
        return {"tested": 0, "target_flips": 0, "flip_fraction": None}
    base = np.asarray(base, dtype=float)
    if target == int(np.argmax(base)):
        raise ValueError("target must differ from baseline prediction")
    tested = flips = 0
    for d in other_training_donors[:budget_remaining]:
        edited = splice(x, d, components, selected, taper=taper)
        score = np.asarray(predict_many(edited[None]), dtype=float)
        if score.shape != (1, len(base)) or not np.isfinite(score).all():
            raise ValueError("invalid donor audit response")
        tested += 1
        flips += int(int(np.argmax(score[0])) == target)
    return {"tested": tested, "target_flips": flips,
            "flip_fraction": flips / tested if tested else None}
