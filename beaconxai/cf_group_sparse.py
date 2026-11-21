"""Exact finite-bin group-sparse projection under UCI HAR gravity invariant.

Given a proposed CF y, first find nearest raw-L2 acceleration projection
z with total_acc' - body_acc' == original residual at every timestamp.
Then choose a disjoint collection of time-bin acceleration/gyro groups to
keep from z, reverting all others to original x. For a bounded cell cost,
a 0/1 knapsack maximizes retained raw-L2 energy (equivalently minimizes
squared raw-L2 distance to z) among this predefined set of group masks.

This is NOT global counterfactual minimality, biomechanical realism or a
query-free validity certificate; candidate flips must be tested by the model.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .cf_physical_projection import project_uci_har_preserve_gravity


@dataclass(frozen=True)
class GroupSparseProjection:
    x: np.ndarray
    groups: tuple[int, ...]
    maximum_modified_fraction: float
    modified_fraction: float
    cell_cost_bound: int
    selected_cell_cost: int
    energy_retained: float
    full_candidate_energy: float


def project_uci_har_group_sparse(original, proposed, train_mean, train_scale,
                                  *, maximum_modified_fraction: float,
                                  bins: int = 12) -> GroupSparseProjection:
    src=np.asarray(original,dtype=float)
    mu=np.asarray(train_mean,dtype=float).reshape(-1)
    sd=np.asarray(train_scale,dtype=float).reshape(-1)
    if (src.ndim!=2 or src.shape[1]!=9 or
        not 0<maximum_modified_fraction<=1 or not np.isfinite(maximum_modified_fraction)
        or not 1<=bins<=src.shape[0]):
        raise ValueError("invalid UCI HAR input shape, bins or sparsity budget")
    feasible=project_uci_har_preserve_gravity(src,proposed,mu,sd)
    raw_delta=(feasible-src)*sd
    edges=np.linspace(0,len(src),bins+1,dtype=int)
    groups=[]
    for channel_start,channel_end in ((0,3),(3,6)):
        # acceleration is a 6-channel indivisible group across body and total
        # (gyro is an independent 3-channel group).
        for t0,t1 in zip(edges[:-1],edges[1:]):
            if channel_start==0:
                channels=(slice(0,3),slice(6,9))
                n_cells=6*(t1-t0)
            else:
                channels=(slice(3,6),)
                n_cells=3*(t1-t0)
            value=sum(float(np.sum(raw_delta[t0:t1,c]**2)) for c in channels)
            groups.append((int(t0),int(t1),channels,int(n_cells),value))
    capacity=int(np.floor(maximum_modified_fraction*src.size+1e-12))
    dp=np.full(capacity+1,-np.inf,dtype=float)
    masks=[0]*(capacity+1)
    dp[0]=0.
    for i,(t0,t1,channels,cost,value) in enumerate(groups):
        if value<=1e-18 or cost>capacity:
            continue
        for used in range(capacity,cost-1,-1):
            prev=dp[used-cost]
            if np.isfinite(prev) and prev+value>dp[used]+1e-12:
                dp[used]=prev+value
                masks[used]=masks[used-cost]|(1<<i)
    # smallest cell cost wins ties for compactness
    used=int(np.argmax(dp))
    bits=masks[used]
    selected=tuple(i for i in range(len(groups)) if bits & (1<<i))
    result=src.copy()
    for j in selected:
        t0,t1,channels,_,_=groups[j]
        for ch in channels:
            result[t0:t1,ch]=feasible[t0:t1,ch]
    modified=float(np.mean(np.abs(result-src)>1e-8))
    if modified>maximum_modified_fraction+1e-12:
        raise AssertionError("sparsity constraint violated")
    # Exact acceleration invariant is guaranteed by keeping coupled channels
    # together or reverting both to the original.
    residual_before=src[:,6:9]*sd[6:9]+mu[6:9]-src[:,:3]*sd[:3]-mu[:3]
    residual_after=result[:,6:9]*sd[6:9]+mu[6:9]-result[:,:3]*sd[:3]-mu[:3]
    if not np.allclose(residual_before,residual_after,rtol=1e-10,atol=1e-10):
        raise AssertionError("gravity residual not preserved")
    return GroupSparseProjection(
        x=result,groups=selected,
        maximum_modified_fraction=float(maximum_modified_fraction),
        modified_fraction=modified,cell_cost_bound=capacity,
        selected_cell_cost=sum(groups[j][3] for j in selected),
        energy_retained=float(dp[used]),
        full_candidate_energy=float(sum(g[4] for g in groups)),
    )
