"""Coupled UCI HAR acceleration edit candidates.

UCI HAR records body_acc, body_gyro and total_acc. Editing total_acc
alone can make the derived gravity residual inconsistent. This control
replaces body_acc and total_acc in the same time bin, or a gyro fragment.
The operator still does not guarantee physical realism.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class PatchGroup:
    regions: tuple
    label: str
    time_bin: int


def uci_har_coupled_groups(components: Sequence) -> list[PatchGroup]:
    c = {(int(x.group), int(x.time_bin)): x for x in components}
    time_bins = sorted({int(x.time_bin) for x in components})
    if (len(c) != 3 * len(time_bins) or time_bins != list(range(len(time_bins)))
        or set(int(x.group) for x in components) != {0, 1, 2}):
        raise ValueError("requires a complete 3-group UCI HAR time grid")
    for t in time_bins:
        a, g, total = c[(0,t)], c[(1,t)], c[(2,t)]
        if ((a.t0,a.t1)!=(total.t0,total.t1) or
            (a.c0,a.c1)!=(0,3) or (g.c0,g.c1)!=(3,6) or
            (total.c0,total.c1)!=(6,9)):
            raise ValueError("requires body_acc, body_gyro, total_acc channels")
    return ([PatchGroup((c[(0,t)],c[(2,t)]), "body_and_total_acc",t) for t in time_bins]
            + [PatchGroup((c[(1,t)],),"gyro",t) for t in time_bins])


def coupled_priority(prior: np.ndarray, original_components: Sequence) -> np.ndarray:
    p=np.asarray(prior,dtype=float)
    if len(p)!=len(original_components) or not np.isfinite(p).all():
        raise ValueError("invalid original component prior")
    groups=uci_har_coupled_groups(original_components)
    idx={id(c):i for i,c in enumerate(original_components)}
    return np.array([max(abs(p[idx[id(rc)]]) for rc in g.regions) for g in groups],dtype=float)
