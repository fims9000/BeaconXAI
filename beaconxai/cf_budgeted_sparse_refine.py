"""Budgeted, model-verified sparse refinement of a valid UCI HAR counterfactual.

Input candidate is ALREADY verified for a fixed target on a fixed classifier.
Try ascending group-support caps using exact train-normalized, raw-unit
gravity-preserving knapsack projections, measuring the classifier once per
candidate. Return the first actually verified compact replacement or retain
the already verified original. Every refinement measurement costs one model
evaluation. No claims about global minimality or physical realizability.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .cf_group_sparse import project_uci_har_group_sparse


@dataclass(frozen=True)
class SparseRefinement:
    x: np.ndarray
    status: str
    attempted_fractions: tuple[float, ...]
    accepted_fraction_cap: float | None
    model_evaluations: int
    original_retained: bool
    edited_fraction: float


def refine_verified_cf_sparse(
    original: np.ndarray,
    verified_candidate: np.ndarray,
    train_mean: np.ndarray,
    train_scale: np.ndarray,
    predict_many: Callable[[np.ndarray], np.ndarray],
    *,
    target_class: int,
    budget_remaining: int,
    fraction_caps: Sequence[float] = (.2,.35,.5,.65),
    bins: int = 12,
) -> SparseRefinement:
    """A sequential support-cap search within *remaining* black-box calls.

    Caller must ensure verified_candidate was measured as class target before
    invocation. This function never infers validity from statistical scores.
    """
    x=np.asarray(original,dtype=float)
    candidate=np.asarray(verified_candidate,dtype=float)
    if (x.ndim!=2 or x.shape[1]!=9 or candidate.shape!=x.shape or
        not np.isfinite(x).all() or not np.isfinite(candidate).all() or
        not isinstance(budget_remaining,int) or budget_remaining<0):
        raise ValueError("finite UCI HAR windows and nonnegative budget required")
    caps=tuple(float(v) for v in fraction_caps)
    if (not caps or any(not np.isfinite(v) or v<=0 or v>1 for v in caps)
        or list(caps)!=sorted(set(caps))):
        raise ValueError("fraction caps must be strictly increasing, unique and in (0,1]")
    if target_class<0:
        raise ValueError("invalid target")
    tried=[]
    count=0
    for cap in caps:
        if count>=budget_remaining:
            break
        reduced=project_uci_har_group_sparse(
            x,candidate,train_mean,train_scale,
            maximum_modified_fraction=cap,bins=bins
        )
        scored=np.asarray(predict_many(reduced.x[None]),dtype=float)
        if scored.ndim!=2 or scored.shape[0]!=1 or not np.isfinite(scored).all():
            raise ValueError("classifier must return finite [1,classes] scores")
        if target_class>=scored.shape[1]:
            raise ValueError("target out of range")
        count+=1
        tried.append(cap)
        if int(np.argmax(scored[0]))==target_class:
            return SparseRefinement(
                reduced.x.copy(),"verified_sparse",tuple(tried),
                cap,count,False,reduced.modified_fraction,
            )
    return SparseRefinement(
        candidate.copy(),"fallback_original_verified",
        tuple(tried),None,count,True,
        float(np.mean(np.abs(candidate-x)>1e-8)),
    )
