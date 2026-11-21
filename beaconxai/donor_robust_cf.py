"""Donor-robust target counterfactuals for a fixed black-box classifier.

An edit is certified only if the *same* component set switches the prediction
to the *same preselected target* for all supplied TRAIN-only donors, measured
with actual model outputs within one exact Q budget. This finite-donor
certificate is not statistical population coverage or physical causality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

from .verified_counterfactual import splice, target_margin


@dataclass(frozen=True)
class DonorProbe:
    components: tuple[int, ...]
    donor_index: int
    predicted_class: int
    target_margin: float
    targeted_flip: bool
    stage: str


@dataclass(frozen=True)
class RobustCounterfactual:
    status: str
    source_class: int
    target_class: int
    certified_edit: tuple[int, ...] | None
    queries: int
    budget: int
    n_donors: int
    probes: tuple[DonorProbe, ...]
    local_minimality_checked: bool
    ever_found_single_donor_flip: bool

    @property
    def is_certified(self) -> bool:
        return self.status == "verified_all_selected_donors"


def search_robust_counterfactual(
    x: np.ndarray,
    baseline_scores: np.ndarray,
    donors: Sequence[np.ndarray],
    components: Sequence,
    cheap_rank: Sequence[int],
    predict_many: Callable[[np.ndarray], np.ndarray],
    *,
    target_class: int,
    budget: int,
    max_changed_components: int = 8,
    seed_singletons: int = 3,
    taper: int = 2,
    prune: bool = True,
) -> RobustCounterfactual:
    """Greedy repair after a failed donor audit, with exact model-call counting.

    Evaluate a few singleton edits on donor 0. If a candidate switches on
    donor 0, audit the identical component set on other TRAIN donors. If a
    donor fails, continue growing the edit, testing the last failing donor
    first. Never reuse a successful result for a *different* edit set.
    After all donors switch, optionally remove components; local minimality
    is claimed only after all single deletions are actually disproved.

    The cheap ranking is fixed before any model calls. Donor checks, pruning
    and search all consume the SAME Q budget. No holdout labels are used.
    """
    x = np.asarray(x, dtype=float)
    base = np.asarray(baseline_scores, dtype=float)
    pool = tuple(np.asarray(d, dtype=float) for d in donors)
    rank = tuple(int(v) for v in cheap_rank)
    if (x.ndim != 2 or not np.isfinite(x).all() or
            base.ndim != 1 or len(base) < 2 or not np.isfinite(base).all()):
        raise ValueError("finite [T,C] source and [classes] baseline required")
    if len(pool) < 2 or any(d.shape != x.shape or not np.isfinite(d).all() for d in pool):
        raise ValueError("at least two finite, same-shaped training donors required")
    if not components or len(rank) != len(components) or set(rank) != set(range(len(components))):
        raise ValueError("ranking must be a permutation of all components")
    source = int(np.argmax(base))
    if target_class < 0 or target_class >= len(base) or target_class == source:
        raise ValueError("target must differ from original predicted class")
    if (budget < 1 or not 1 <= seed_singletons <= budget or
            max_changed_components < 1 or taper < 0):
        raise ValueError("invalid query budget, seed count or group size")

    max_changed_components = min(max_changed_components, len(components))
    probes: list[DonorProbe] = []
    cache: dict[tuple[tuple[int, ...], int], DonorProbe] = {}
    ever_flip = False
    exhausted = False

    def measure(selected: tuple[int, ...], donor_index: int, stage: str) -> DonorProbe | None:
        nonlocal ever_flip, exhausted
        key = tuple(sorted(selected))
        ck = (key, donor_index)
        if ck in cache:
            return cache[ck]
        if len(probes) >= budget:
            exhausted = True
            return None
        edited = splice(x, pool[donor_index], components, key, taper=taper)
        out = np.asarray(predict_many(edited[None]), dtype=float)
        if out.shape != (1, len(base)) or not np.isfinite(out).all():
            raise ValueError("model must return finite [1,classes] scores")
        predicted = int(np.argmax(out[0]))
        flip = predicted == target_class
        ever_flip = ever_flip or flip
        p = DonorProbe(key, donor_index, predicted,
                       target_margin(out[0], target_class), flip, stage)
        cache[ck] = p
        probes.append(p)
        return p

    def check_all(selected: tuple[int, ...], first: int, stage: str):
        """Return (certified, failed_donor); None means Q exhausted."""
        order = (first,) + tuple(i for i in range(len(pool)) if i != first)
        for donor_index in order:
            p = measure(selected, donor_index, stage)
            if p is None:
                return None, donor_index
            if not p.targeted_flip:
                return False, donor_index
        return True, None

    def minimize(selected: tuple[int, ...]):
        """Try removal. Disproving universal validity needs one failing donor."""
        current = selected
        while len(current) > 1:
            reduced = False
            for j in current:
                subset = tuple(v for v in current if v != j)
                valid, _ = check_all(subset, 0, "prune")
                if valid is None:
                    return current, False
                if valid:
                    current = subset
                    reduced = True
                    break
            if not reduced:
                return current, True
        return current, True

    def result(selected: tuple[int, ...] | None, minimal: bool = False):
        status = ("verified_all_selected_donors" if selected is not None
                  else "budget_exhausted_without_certificate" if exhausted
                  else "not_found_within_budget")
        return RobustCounterfactual(
            status, source, target_class, selected, len(probes),
            budget, len(pool), tuple(probes), minimal if selected else False,
            ever_flip,
        )

    seed_count = min(seed_singletons, len(rank), budget)
    singles: list[DonorProbe] = []
    last_failed = 0
    for j in rank[:seed_count]:
        p = measure((j,), 0, "singleton")
        if p is None:
            return result(None)
        singles.append(p)
        if p.targeted_flip:
            valid, failed = check_all((j,), 0, "audit_single")
            if valid is None:
                return result(None)
            if valid:
                return result((j,), True)  # original x predicted source != target
            last_failed = failed
    if max_changed_components == 1 or len(probes) >= budget:
        return result(None)

    # Same seed ordering as the reference single-donor path. Repair grows the
    # selected edit instead of stopping at the first non-robust switch.
    strongest = sorted(singles, key=lambda p: -p.target_margin)
    current = strongest[0].components
    remaining = [p.components[0] for p in strongest if p.components[0] not in current]
    remaining += [j for j in rank[seed_count:] if j not in current]
    for j in remaining:
        if len(probes) >= budget or len(current) >= max_changed_components:
            break
        current = tuple(sorted((*current, j)))
        valid, failed = check_all(current, last_failed, "repair")
        if valid is None:
            return result(None)
        if valid:
            if prune:
                small, minimal = minimize(current)
                return result(small, minimal)
            return result(current, len(current) == 1)
        last_failed = failed
    return result(None)



def search_backward_robust_counterfactual(
    x: np.ndarray,
    baseline_scores: np.ndarray,
    donors: Sequence[np.ndarray],
    components: Sequence,
    removal_order: Sequence[int],
    predict_many: Callable[[np.ndarray], np.ndarray],
    *,
    target_class: int,
    budget: int = 16,
    maximum_modified_fraction: float | None = None,
    taper: int = 2,
) -> RobustCounterfactual:
    """Budget-exact coarse-to-fine pruning of one mask across all TRAIN donors.

    The full donor-patched window must first be measured on EVERY donor.
    Subsequent grouped removals are accepted only after measuring the SAME
    reduced mask on EVERY donor; a failure on one donor is sufficient to
    reject the removal. All model examples, not Python batch calls, count.

    A certificate refers to exactly the donor instances provided. A
    singleton-deletion minimality flag is true ONLY if the final mask has
    been checked against every individual removal, either with a measured
    failure witness or (for empty masks) with the cached original output.
    max_modified_fraction, if set, is a *hard return constraint* on every
    donor, never an inferred guarantee.
    """
    source = np.asarray(x, dtype=float)
    base = np.asarray(baseline_scores, dtype=float)
    pool = tuple(np.asarray(d, dtype=float) for d in donors)
    order = tuple(int(j) for j in removal_order)
    if (source.ndim != 2 or not np.isfinite(source).all()
        or base.ndim != 1 or len(base) < 2 or not np.isfinite(base).all()
        or len(pool) < 2 or any(d.shape != source.shape or not np.isfinite(d).all()
                                for d in pool)
        or not components or len(order) != len(components)
        or set(order) != set(range(len(components)))):
        raise ValueError("invalid source, donor bank, scores or removal order")
    source_class = int(np.argmax(base))
    if not 0 <= target_class < len(base) or target_class == source_class:
        raise ValueError("target must differ from predicted source class")
    if not isinstance(budget, int) or budget < len(pool):
        raise ValueError("budget must fit initial checks for all donors")
    if not isinstance(taper, int) or taper < 0:
        raise ValueError("invalid taper")
    if maximum_modified_fraction is not None and not (
        0 < maximum_modified_fraction <= 1
        and np.isfinite(maximum_modified_fraction)
    ):
        raise ValueError("invalid edit support cap")

    probes: list[DonorProbe] = []
    cache: dict[tuple[tuple[int, ...], int], DonorProbe] = {}
    ever_flip = False
    exhausted = False

    def check(mask: tuple[int, ...], donor_index: int, stage: str) -> DonorProbe | None:
        nonlocal ever_flip, exhausted
        key = tuple(sorted(mask))
        ck = (key, donor_index)
        if ck in cache:
            return cache[ck]
        if len(probes) >= budget:
            exhausted = True
            return None
        edited = splice(source, pool[donor_index], components, key, taper=taper)
        scores = np.asarray(predict_many(edited[None]), dtype=float)
        if scores.shape != (1, len(base)) or not np.isfinite(scores).all():
            raise ValueError("classifier must return finite [1,classes] scores")
        predicted = int(np.argmax(scores[0]))
        flip = predicted == target_class
        probe = DonorProbe(key, donor_index, predicted,
                           target_margin(scores[0], target_class), flip, stage)
        probes.append(probe)
        cache[ck] = probe
        ever_flip |= flip
        return probe

    def evaluate_all(mask: tuple[int, ...], stage: str) -> bool | None:
        # Empty mask is exactly x; cached source prediction proves a failure.
        if not mask:
            return False
        for donor_index in range(len(pool)):
            p = check(mask, donor_index, stage)
            if p is None:
                return None
            if not p.targeted_flip:
                return False
        return True

    def compact(mask: tuple[int, ...]) -> bool:
        if maximum_modified_fraction is None:
            return True
        for donor in pool:
            edited = splice(source, donor, components, mask, taper=taper)
            if float(np.mean(np.abs(edited - source) > 1e-8)) > maximum_modified_fraction:
                return False
        return True

    def done(mask: tuple[int, ...] | None, minimal: bool = False) -> RobustCounterfactual:
        qualified = mask is not None and compact(mask)
        result_mask = mask if qualified else None
        return RobustCounterfactual(
            "verified_all_selected_donors" if qualified
            else ("budget_exhausted_without_certificate" if exhausted
                  else "not_found_within_budget"),
            source_class, target_class, result_mask,
            len(probes), budget, len(pool), tuple(probes),
            bool(minimal and qualified), ever_flip,
        )

    current = tuple(range(len(components)))
    initial = evaluate_all(current, "full_donor_seed")
    if initial is not True:
        return done(None)

    # A verified full mask is retained if finer changes cannot be confirmed.
    chunk = max(1, len(current) // 2)
    while chunk >= 1:
        while True:
            active = [j for j in order if j in current]
            removed = False
            for pos in range(0, len(active), chunk):
                dropped = set(active[pos:pos+chunk])
                reduced = tuple(j for j in current if j not in dropped)
                verdict = evaluate_all(reduced, "chunk_prune" if chunk > 1 else "singleton_prune")
                if verdict is None:
                    return done(current)
                if verdict:
                    current = reduced
                    removed = True
                    break
            if not removed:
                if chunk == 1:
                    return done(current, minimal=True)
                break
        if chunk == 1:
            break
        chunk = max(1, chunk // 2)
    return done(current, minimal=len(current) == 1)
