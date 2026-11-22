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



def search_backward_counterfactual(
    x: np.ndarray,
    baseline: np.ndarray,
    donor: np.ndarray,
    components: Sequence,
    removal_order: Sequence[int],
    predict_many: Callable,
    *,
    target_class: int,
    budget: int,
    taper: int = 2,
) -> PathSearchResult:
    """Prune a measured full-donor patch to a smaller verified counterfactual.

    Reverse-search control for a partition of disjoint temporal regions.
    A full patch is tested first; absent a measured target switch, abstain.
    Subsequent removals are accepted ONLY after a direct model evaluation.
    This works with any fixed candidate set, has no test-label inputs,
    and never claims global minimum intervention size.

    A local 1-deletion minimality flag is issued only when the *final*
    selected set has been checked against every single-component deletion
    after the last successful deletion. Q includes the initial full patch,
    all attempted deletions and any local-minimality confirmation calls.
    """
    source=np.asarray(x,dtype=float)
    donor=np.asarray(donor,dtype=float)
    base=np.asarray(baseline,dtype=float)
    ordered=tuple(int(i) for i in removal_order)
    if (source.ndim!=2 or donor.shape!=source.shape or
        not np.isfinite(source).all() or not np.isfinite(donor).all() or
        base.ndim!=1 or len(base)<2 or not np.isfinite(base).all()):
        raise ValueError("finite matching [T,C] arrays and class scores required")
    if (target_class<0 or target_class>=len(base) or
        target_class==int(np.argmax(base)) or budget<1):
        raise ValueError("valid fixed target distinct from source and positive Q required")
    if (not components or len(ordered)!=len(components)
        or set(ordered)!=set(range(len(components)))):
        raise ValueError("removal order must be a component permutation")
    if not isinstance(taper,int) or taper<0:
        raise ValueError("nonnegative integer taper required")
    probes: list[PathProbe]=[]
    measured:dict[tuple[int,...],PathProbe]={}
    scale=max(float(np.linalg.norm(source)),float(np.linalg.norm(donor)),1e-8)

    def check(selection:tuple[int,...],stage:str) -> PathProbe|None:
        key=tuple(sorted(selection))
        if not key:
            return None  # original baseline already known not to equal target
        if key in measured:
            return measured[key]
        if len(probes)>=budget:
            return None
        edit=splice(source,donor,components,key,taper=taper)
        output=np.asarray(predict_many(edit[None]),dtype=float)
        if output.shape!=(1,len(base)) or not np.isfinite(output).all():
            raise ValueError("classifier must return [1,classes] finite scores")
        winner=int(np.argmax(output[0]))
        result=PathProbe(
            key,winner,target_margin(output[0],target_class),
            winner==target_class,
            float(np.linalg.norm(edit-source)/scale),stage
        )
        measured[key]=result
        probes.append(result)
        return result

    source_class=int(np.argmax(base))
    current=tuple(range(len(components)))
    initial=check(current,"full_donor_seed")
    if initial is None or not initial.targeted_flip:
        return PathSearchResult(
            "not_found_within_budget",source_class,target_class,None,
            len(probes),budget,tuple(probes),False
        )

    # Try removing less-promising parts first. A successful removal changes
    # the candidate, so all prior failed deletions must be rechecked if
    # claiming local minimality for the final set.
    while True:
        removed=False
        completed_pass=True
        for j in ordered:
            if j not in current:
                continue
            subset=tuple(k for k in current if k!=j)
            if not subset:
                continue  # original non-target from cached baseline
            probe=check(subset,"remove")
            if probe is None:
                completed_pass=False
                break
            if probe.targeted_flip:
                current=subset
                removed=True
                break
        if not removed:
            return PathSearchResult(
                "verified_target_flip",source_class,target_class,current,
                len(probes),budget,tuple(probes),completed_pass
            )
        if len(probes)>=budget:
            return PathSearchResult(
                "verified_target_flip",source_class,target_class,current,
                len(probes),budget,tuple(probes),False
            )
