"""Donor-independent counterfactual audit and predeclared paired statistics.

Pure helpers: no test labels, no black-box queries, no leakage from
other-donor results into search. Statistical inference is on paired windows.
"""
from __future__ import annotations
import math
import numpy as np


def rank_confirmed_donors(
    source: np.ndarray, donor_windows: np.ndarray,
    model_labels: np.ndarray, target: int,
    original_train_indices: np.ndarray,
) -> tuple[int, ...]:
    x=np.asarray(source,dtype=float)
    donor=np.asarray(donor_windows,dtype=float)
    labels=np.asarray(model_labels)
    ids=np.asarray(original_train_indices)
    if (x.ndim!=2 or donor.ndim!=3 or donor.shape[1:]!=x.shape
        or labels.shape!=(len(donor),) or ids.shape!=(len(donor),)
        or not np.isfinite(x).all() or not np.isfinite(donor).all()
        or len(set(map(int,ids)))!=len(ids)):
        raise ValueError("inconsistent TRAIN-only donor pool or source")
    qualified=np.flatnonzero(labels==target)
    distance=np.mean((donor-x[None])**2,axis=(1,2))
    return tuple(sorted(
        map(int,qualified),
        key=lambda j:(float(distance[j]),int(ids[j]))
    ))


def joint_quality(
    measured_target_flip: bool, modified_fraction: float | None,
    locally_minimal_checked: bool, maximum_fraction: float = .35
) -> bool:
    if not 0<=maximum_fraction<=1:
        raise ValueError("invalid predeclared support limit")
    return bool(
        measured_target_flip and locally_minimal_checked
        and modified_fraction is not None
        and np.isfinite(modified_fraction)
        and modified_fraction<=maximum_fraction
    )


def component_jaccard(a, b) -> float:
    first=set(map(int,a))
    second=set(map(int,b))
    if not first and not second:
        return 1.
    return float(len(first&second)/len(first|second))


def exact_two_sided_mcnemar(backward_only:int,forward_only:int) -> float:
    b=int(backward_only)
    c=int(forward_only)
    if b<0 or c<0 or b!=backward_only or c!=forward_only:
        raise ValueError("nonnegative discordant counts required")
    n=b+c
    if not n:
        return 1.
    k=min(b,c)
    return min(1.,2.*sum(math.comb(n,j) for j in range(k+1))/(2**n))
