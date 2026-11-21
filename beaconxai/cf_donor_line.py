"""Simple real-donor line-search baseline under strict black-box query budgets.

Not an attribution or physical-causality method. It interpolates ALL
channels between the source and one preselected TRAIN donor. Strong
comparison for target-flip query efficiency but typically poor sparsity.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LineCounterfactual:
    status: str
    edited: np.ndarray | None
    alpha: float | None
    queries: int
    budget: int
    tested_alphas: tuple[float, ...]

    @property
    def is_verified(self) -> bool:
        return self.status=="verified_target_flip"


def search_donor_segment(
    x: np.ndarray, donor: np.ndarray, base_scores: np.ndarray,
    predict_many: Callable[[np.ndarray], np.ndarray],
    *, target_class: int, budget: int,
) -> LineCounterfactual:
    x=np.asarray(x,dtype=float)
    donor=np.asarray(donor,dtype=float)
    base=np.asarray(base_scores,dtype=float)
    if (x.ndim!=2 or donor.shape!=x.shape or
        not np.isfinite(x).all() or not np.isfinite(donor).all()
        or base.ndim!=1 or len(base)<2 or not np.isfinite(base).all()):
        raise ValueError("finite matching [T,C] source/donor and class scores required")
    if (not 0<=target_class<len(base) or
        int(np.argmax(base))==target_class or budget<1):
        raise ValueError("invalid fixed target or query budget")
    tested=[]
    def measure(alpha):
        tested.append(float(alpha))
        edited=x+alpha*(donor-x)
        out=np.asarray(predict_many(edited[None]),dtype=float)
        if out.shape!=(1,len(base)) or not np.isfinite(out).all():
            raise ValueError("model must return [1,classes] finite scores")
        return int(np.argmax(out[0]))==target_class
    if not measure(1.):
        return LineCounterfactual(
            "no_target_in_donor",None,None,len(tested),budget,tuple(tested))
    lo,hi=0.,1.
    for _ in range(budget-1):
        mid=(lo+hi)/2
        if measure(mid):
            hi=mid
        else:
            lo=mid
    return LineCounterfactual(
        "verified_target_flip",x+hi*(donor-x),hi,
        len(tested),budget,tuple(tested),
    )
