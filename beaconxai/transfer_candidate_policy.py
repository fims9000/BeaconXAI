"""TRAIN-fitted candidate priority for donor-transfer-aware shared-Q16 search.

No model query occurs here: feature vectors use source, two ONLINE donors,
TRAIN-only class-profile gains and a CACHED original model prediction only.
TRAIN label for fitting is a separately measured four-donor joint target flip.
The learned probability is a ranking approximation, never a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .donor_robust_cf import DonorProbe
from .verified_counterfactual import splice, target_margin


BASE_FEATURES = (
    "n_selected_fraction",
    *(f"component_{i}" for i in range(10)),
    "support_mean_online", "support_max_online",
    "prior_gain_min", "prior_gain_mean", "prior_gain_spread",
    "component_gain_min", "component_gain_max", "component_gain_mean",
    "source_d1_mse", "source_d2_mse", "d1_d2_mse",
    "original_target_margin",
    "time_mask_center", "adjacent_selected_pair_fraction",
)
N_FEATURES = len(BASE_FEATURES)


def fixed_training_masks(
    generator: np.random.Generator,
    *,
    n_components: int = 10,
) -> tuple[tuple[int, ...], ...]:
    """Exactly 10 singles, 5 sampled pairs, 5 sampled triples per window."""
    if n_components != 10:
        raise ValueError("frozen component count must equal 10")
    pairs=list(combinations(range(n_components), 2))
    triples=list(combinations(range(n_components), 3))
    ix2=generator.choice(len(pairs),size=5,replace=False)
    ix3=generator.choice(len(triples),size=5,replace=False)
    masks=tuple((i,) for i in range(10))
    masks+=tuple(pairs[int(i)] for i in ix2)
    masks+=tuple(triples[int(i)] for i in ix3)
    if len(masks)!=20 or len(set(masks))!=20:
        raise AssertionError("frozen TRAIN candidate set changed")
    return masks


def candidate_features(
    source: np.ndarray,
    online_donors: Sequence[np.ndarray],
    components: Sequence,
    gains: np.ndarray,
    original_scores: np.ndarray,
    target: int,
    mask: Sequence[int],
    *,
    taper: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cheap [25] features and *exact* support on D1/D2 only."""
    x=np.asarray(source,dtype=float)
    pool=np.asarray(online_donors,dtype=float)
    prior=np.asarray(gains,dtype=float)
    base=np.asarray(original_scores,dtype=float)
    m=len(components)
    chosen=tuple(sorted(map(int,mask)))
    if (m!=10 or x.ndim!=2 or pool.shape!=(2,*x.shape)
        or prior.shape!=(2,m) or base.ndim!=1 or len(base)<2
        or not np.isfinite(x).all() or not np.isfinite(pool).all()
        or not np.isfinite(prior).all() or not np.isfinite(base).all()
        or not 0<=target<len(base) or target==int(np.argmax(base))
        or len(chosen)==0 or len(chosen)!=len(set(chosen))
        or min(chosen)<0 or max(chosen)>=m or taper<0):
        raise ValueError("invalid candidate, geometry, TRAIN gains or target")
    selected=np.zeros(m,dtype=float)
    selected[list(chosen)]=1.
    patched=np.stack([
        splice(x,d,components,chosen,taper=taper)
        for d in pool
    ])
    support=np.mean(np.abs(patched-x[None])>1e-8,axis=(1,2))
    sums=np.sum(prior[:,list(chosen)],axis=1)
    elements=prior[:,list(chosen)].reshape(-1)
    features=np.r_[
        len(chosen)/m,selected,
        np.mean(support),np.max(support),
        np.min(sums),np.mean(sums),np.ptp(sums),
        np.min(elements),np.max(elements),np.mean(elements),
        np.mean((x-pool[0])**2),
        np.mean((x-pool[1])**2),
        np.mean((pool[0]-pool[1])**2),
        target_margin(base,target),
        np.mean(chosen)/(m-1),
        sum(selected[i]*selected[i+1] for i in range(m-1))/(m-1),
    ]
    if features.shape!=(N_FEATURES,) or not np.isfinite(features).all():
        raise AssertionError("TRAIN-only candidate feature dimensions invalid")
    return features.astype(float),support.astype(float)


@dataclass(frozen=True)
class CandidateTransferPrior:
    scaler: StandardScaler | None
    model: LogisticRegression | None
    constant: float | None
    n_fit: int
    joint_successes: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        array=np.asarray(features,dtype=float)
        if array.ndim!=2 or array.shape[1]!=N_FEATURES or not np.isfinite(array).all():
            raise ValueError("expected finite candidate features [N,25]")
        if self.model is None:
            probability=np.full(len(array),float(self.constant),dtype=float)
        else:
            probability=self.model.predict_proba(
                self.scaler.transform(array)
            )[:,1]
        return np.clip(probability,1e-6,1-1e-6)


def fit_candidate_transfer_prior(
    features: np.ndarray,
    joint_success: np.ndarray,
) -> CandidateTransferPrior:
    X=np.asarray(features,dtype=float)
    y=np.asarray(joint_success)
    if (X.ndim!=2 or X.shape[1]!=N_FEATURES or not np.isfinite(X).all()
        or y.shape!=(len(X),) or not np.isin(y,(0,1)).all()):
        raise ValueError("TRAIN-only candidate examples and binary measured labels required")
    n=len(y)
    pos=int(np.sum(y))
    if not n or len(np.unique(y))<2:
        return CandidateTransferPrior(None,None,(pos+1)/(n+2),n,pos)
    scaler=StandardScaler()
    z=scaler.fit_transform(X)
    model=LogisticRegression(
        C=1.,class_weight="balanced",solver="liblinear",
        random_state=57,max_iter=600,
    ).fit(z,y.astype(int))
    return CandidateTransferPrior(scaler,model,None,n,pos)


class TransferAwareRanking:
    """Query-free callback compatible with the existing sparse core.

    Fixed logistic scoring is computed once per TEST window. Only the
    chronological *actually measured* D1/D2 probes can adapt subsequent
    candidate order. No D3/D4 data are passed into this object.
    """
    def __init__(
        self,
        source: np.ndarray,
        donors: Sequence[np.ndarray],
        components: Sequence,
        gains: np.ndarray,
        original_scores: np.ndarray,
        target: int,
        prior: CandidateTransferPrior,
        *,
        taper: int = 2,
        adaptive: bool = False,
    ):
        self.source=source
        self.donors=tuple(donors)
        self.components=components
        self.gains=gains
        self.original_scores=original_scores
        self.target=target
        self.prior=prior
        self.taper=taper
        self.adaptive=bool(adaptive)
        self.call_count=0
        self.max_feedback=0
        self._cached_masks=None
        self._base_logits=None
        self._membership=None

    def __call__(self,masks,old_scores,support,probes):
        self.call_count+=1
        n=len(masks)
        if (np.asarray(old_scores).shape!=(n,)
            or np.asarray(support).shape!=(n,)):
            raise ValueError("policy mask meta geometry mismatch")
        if self._cached_masks is None:
            self._cached_masks=tuple(masks)
            if n:
                features=np.stack([
                    candidate_features(
                        self.source,self.donors,self.components,
                        self.gains,self.original_scores,self.target,mask,
                        taper=self.taper,
                    )[0] for mask in masks
                ])
                p=self.prior.predict(features)
                self._base_logits=np.log(p/(1-p))
                self._membership=np.stack([
                    np.asarray([int(j in mask) for j in range(10)])
                    for mask in masks
                ])
            else:
                self._base_logits=np.empty(0)
                self._membership=np.empty((0,10),dtype=int)
        elif tuple(masks)!=self._cached_masks:
            raise ValueError("candidate set cannot change after first ranking")
        scores=self._base_logits.copy()
        if self.adaptive and probes and n:
            counted=0
            for evidence in probes:
                if evidence.donor_index not in (0,1):
                    raise ValueError("unseen donor probe leaked into adaptive policy")
                sample=np.asarray([
                    int(j in evidence.components) for j in range(10)
                ])
                hamming=np.sum(self._membership!=sample[None],axis=1)
                sign=1. if evidence.targeted_flip else -1.
                scores+=2.*sign*np.exp(-hamming/1.5)/len(probes)
                counted+=1
            self.max_feedback=max(self.max_feedback,counted)
        return scores
