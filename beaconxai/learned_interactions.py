"""Learned pair interactions + online-updated Q-budgeted time-series CF policy.

TRAIN-only calibration may evaluate the frozen classifier many times and MUST
be accounted for separately. The online controller issues <=Q actual model
evaluations shared by all donors; no surrogate prediction is a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from .donor_robust_cf import (
    DonorProbe, RobustCounterfactual, _disjoint_support_counts,
)
from .verified_counterfactual import splice, target_margin


def feature_vector(gains: np.ndarray, mask: Sequence[int], *, pairwise: bool) -> np.ndarray:
    g = np.asarray(gains, dtype=float)
    if g.ndim != 1 or len(g) < 2 or not np.isfinite(g).all():
        raise ValueError("expected finite component gains")
    ix = tuple(sorted(map(int, mask)))
    if len(ix) != len(set(ix)) or any(j < 0 or j >= len(g) for j in ix):
        raise ValueError("mask indices invalid")
    membership = np.zeros(len(g), dtype=float)
    membership[list(ix)] = 1.
    # Dimensionless and bounded; TRAIN-prior scale may vary greatly by class.
    signed = float(np.sum(g[list(ix)]) / max(np.sum(np.abs(g)), 1e-12))
    base = np.r_[signed, membership]
    if not pairwise:
        return base
    pairs = np.array([membership[i] * membership[j]
                      for i in range(len(g))
                      for j in range(i+1, len(g))], dtype=float)
    return np.r_[base, pairs]


@dataclass(frozen=True)
class PairwisePrior:
    n_components: int
    pairwise: bool
    mean: np.ndarray
    scale: np.ndarray
    theta: np.ndarray
    training_rows: int
    alpha: float
    calibration_mae: float

    @property
    def pair_coefficient_norm(self) -> float:
        return (float(np.linalg.norm(self.theta[2+self.n_components:]))
                if self.pairwise else 0.)

    def vector(self, gains: np.ndarray, mask: Sequence[int]) -> np.ndarray:
        raw = feature_vector(gains, mask, pairwise=self.pairwise)
        if len(raw) != len(self.mean):
            raise ValueError("feature map geometry mismatch")
        return np.r_[1., (raw - self.mean) / self.scale]


def fit_interaction_prior(
    rows: Sequence[tuple[np.ndarray, Sequence[int], float]],
    *,
    n_components: int,
    pairwise: bool,
    alpha: float = 10.,
) -> PairwisePrior:
    """Fit from genuine, separately metered TRAIN/CALIBRATION model effects."""
    if n_components < 2 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid ridge configuration")
    if len(rows) < n_components * 2:
        raise ValueError("insufficient TRAIN-only calibration observations")
    x = np.stack([feature_vector(r[0], r[1], pairwise=pairwise) for r in rows])
    y = np.asarray([r[2] for r in rows], dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("finite measured target margin deltas required")
    if x.shape[1] != 1+n_components+(
        n_components*(n_components-1)//2 if pairwise else 0
    ):
        raise ValueError("training component count disagrees with supplied gains")
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-12] = 1.
    z = (x-means)/scales
    model = Ridge(alpha=alpha,fit_intercept=True)
    model.fit(z,y)
    theta=np.r_[float(model.intercept_),np.asarray(model.coef_,dtype=float)]
    residual=np.asarray(model.predict(z))-y
    return PairwisePrior(
        n_components,pairwise,means,scales,theta,
        len(y),float(alpha),float(np.mean(np.abs(residual)))
    )


@dataclass(frozen=True)
class AdaptiveSearch:
    certificate: RobustCounterfactual
    posterior_updates: int
    online_rerankings: int
    candidate_masks: int

    @property
    def queries(self):
        return self.certificate.queries

    @property
    def certified_edit(self):
        return self.certificate.certified_edit

    @property
    def is_certified(self):
        return self.certificate.is_certified


def search_learned_adaptive(
    x: np.ndarray,
    baseline_scores: np.ndarray,
    donors: Sequence[np.ndarray],
    components: Sequence,
    donor_gains: np.ndarray,
    prior: PairwisePrior,
    predict_many: Callable[[np.ndarray], np.ndarray],
    *,
    target_class: int,
    budget: int = 16,
    maximum_modified_fraction: float = .35,
    taper: int = 2,
    adaptive: bool = True,
    posterior_variance: float = .25,
    observation_noise: float = 1.,
    exploration: float = .25,
    support_penalty: float = .1,
) -> AdaptiveSearch:
    """Find a *measured* 2-donor compact mask, with cheap posterior updates.

    Initial Ridge is fit on TRAIN/CALIBRATION only. Each online call makes a
    local Bayesian linear correction to its residual in the frozen model.
    Cached f(x) is not counted. No edits to the actual classifier, no labels,
    no additional verification calls outside Q, no donor population guarantee.
    """
    source=np.asarray(x,dtype=float)
    base=np.asarray(baseline_scores,dtype=float)
    pool=tuple(np.asarray(d,dtype=float) for d in donors)
    gains=np.asarray(donor_gains,dtype=float)
    m=len(components)
    if (source.ndim!=2 or not np.isfinite(source).all()
        or base.ndim!=1 or len(base)<2 or not np.isfinite(base).all()
        or len(pool)<2 or any(d.shape!=source.shape or not np.isfinite(d).all()
                             for d in pool)
        or gains.shape!=(len(pool),m) or not np.isfinite(gains).all()
        or prior.n_components!=m or not 2<=m<=16):
        raise ValueError("invalid source, frozen donor geometry or TRAIN prior")
    if not 0<=target_class<len(base) or target_class==int(np.argmax(base)):
        raise ValueError("invalid fixed target class")
    if not isinstance(budget,int) or budget<len(pool):
        raise ValueError("shared budget too small to verify all donors")
    if not isinstance(taper,int) or taper<0:
        raise ValueError("taper must be a nonnegative integer")
    for v in (maximum_modified_fraction,posterior_variance,observation_noise):
        if not np.isfinite(v) or v<=0:
            raise ValueError("positive finite support and posterior parameters required")
    if maximum_modified_fraction>1 or not np.isfinite(exploration) or not np.isfinite(support_penalty):
        raise ValueError("invalid support and scoring settings")

    support_counts=_disjoint_support_counts(source,pool,components,taper)
    masks=[]
    supports=[]
    features=[]
    for k in range(1,m+1):
        for mask in combinations(range(m),k):
            if support_counts is not None:
                fractions=np.sum(support_counts[:,list(mask)],axis=1)/source.size
            else:
                fractions=np.asarray([
                    np.mean(np.abs(splice(source,d,components,mask,taper=taper)-source)>1e-8)
                    for d in pool
                ],dtype=float)
            if np.any(fractions>maximum_modified_fraction):
                continue
            masks.append(mask)
            supports.append(float(np.max(fractions)))
            features.append(np.stack([
                prior.vector(gains[didx],mask) for didx in range(len(pool))
            ]))
    n=len(masks)
    if not n:
        cf=RobustCounterfactual(
            "no_compact_masks",int(np.argmax(base)),target_class,
            None,0,budget,len(pool),tuple(),False,False
        )
        return AdaptiveSearch(cf,0,0,0)
    phi=np.asarray(features,dtype=float)
    if phi.ndim!=3 or not np.isfinite(phi).all():
        raise ValueError("invalid candidate feature matrix")
    support=np.asarray(supports,dtype=float)
    theta=prior.theta.copy()
    covariance=np.eye(len(theta),dtype=float)*posterior_variance
    probes:list[DonorProbe]=[]
    cache:dict[tuple[tuple[int,...],int],DonorProbe]={}
    posterior_updates=0
    online_rerankings=0
    ever_flip=False
    base_margin=target_margin(base,target_class)
    tried=set()
    exhausted=False

    def prediction(feats):
        predicted=base_margin+np.einsum("...j,j->...",feats,theta)
        uncertainty=np.sqrt(np.maximum(
            np.einsum("...j,jk,...k->...",feats,covariance,feats),0.
        ))
        return predicted,uncertainty

    def check(mask_index:int,didx:int,stage:str) -> DonorProbe|None:
        nonlocal posterior_updates,ever_flip,exhausted
        mask=masks[mask_index]
        key=(mask,didx)
        if key in cache:
            return cache[key]
        if len(probes)>=budget:
            exhausted=True
            return None
        edit=splice(source,pool[didx],components,mask,taper=taper)
        outputs=np.asarray(predict_many(edit[None]),dtype=float)
        if outputs.shape!=(1,len(base)) or not np.isfinite(outputs).all():
            raise ValueError("classifier must return finite [1,classes] scores")
        score=outputs[0]
        correct=int(np.argmax(score))==target_class
        measured=target_margin(score,target_class)
        p=DonorProbe(mask,didx,int(np.argmax(score)),measured,correct,stage)
        probes.append(p)
        cache[key]=p
        ever_flip=ever_flip or correct
        if adaptive:
            v=phi[mask_index,didx]
            cov_v=covariance@v
            denom=observation_noise+float(v@cov_v)
            gain=cov_v/denom
            theta[:]+=gain*((measured-base_margin)-float(v@theta))
            covariance[:]-=np.outer(gain,cov_v)
            posterior_updates+=1
        return p

    def check_all(mask_index:int,stage:str)->bool|None:
        values,_=prediction(phi[mask_index])
        ordering=np.argsort(values,kind="stable")
        for didx in ordering:
            p=check(mask_index,int(didx),stage)
            if p is None:
                return None
            if not p.targeted_flip:
                return False
        return True

    def result(selected:tuple[int,...]|None,minimal:bool=False)->AdaptiveSearch:
        c=RobustCounterfactual(
            "verified_all_selected_donors" if selected is not None
            else ("budget_exhausted_without_certificate" if exhausted
                  else "no_compact_measured_certificate"),
            int(np.argmax(base)),target_class,selected,len(probes),budget,
            len(pool),tuple(probes),bool(selected is not None and minimal),
            ever_flip,
        )
        return AdaptiveSearch(c,posterior_updates,online_rerankings,n)

    while len(probes)+len(pool)<=budget:
        pred,var=prediction(phi)
        acquisition=(
            np.min(pred+exploration*var,axis=1)
            -support_penalty*support
        )
        available=[j for j in range(n) if j not in tried]
        if not available:
            break
        j=min(available,key=lambda k:(-float(acquisition[k]),-len(masks[k]),masks[k]))
        tried.add(j)
        online_rerankings+=1
        outcome=check_all(j,"adaptive_candidate" if adaptive else "static_candidate")
        if outcome is None:
            break
        if outcome is not True:
            continue
        current=j
        # Conservative deletion scan. If any accepted deletion changes the
        # final mask, restart the scan; otherwise count all rejected deletions.
        while True:
            present=masks[current]
            if len(present)<=1:
                return result(present,minimal=True)
            completed=True
            shortened=False
            for removed in present:
                subset=tuple(v for v in present if v!=removed)
                ix=masks.index(subset)  # all submasks of a compact mask are compact
                outcome=check_all(ix,"local_prune")
                if outcome is None:
                    completed=False
                    break
                if outcome:
                    current=ix
                    shortened=True
                    break
            if not shortened:
                return result(masks[current],minimal=completed)
            if len(probes)>=budget:
                return result(masks[current],minimal=False)
    exhausted=len(probes)>=budget or bool(tried)
    return result(None)
