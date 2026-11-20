"""BEACON: cheap surrogate and subject-block conformal top-3 certificate.

Scientific question: can train-only cheap feature-to-response learning reduce
actual perturbation calls while returning an explicit VERIFIED/ABSTAIN flag?

Let D_g(x)=|m(x)-m(T_g(x))| for the moving top-two log-score margin,
phi_g(x) cheap to compute. Fit log(1+D_g) ~ ridge(phi) on surrogate-fit
subjects. On distinct calibration subjects fit the one-sided score
R_s=max_{x in sampled windows of s,g} (D_g-h(phi_g))/scale_group[g].
The upper bound U_g=h(phi_g)+q_{1-alpha}(R_s)*scale_group[g] is
simultaneous over all components and sampled windows for a NEW
EXCHANGEABLE SUBJECT, not distribution-free unconditionally. In particular,
the official UCI HAR split is not necessarily exchangeable; finite-sample
math states an explicit assumption and empirical held-out coverage is checked.

After measuring at least 3 candidates, if the third-largest observed D_g
is strictly greater than ALL unmeasured U_g, the measured top 3 equal the
full-Q=M top 3 on the simultaneous-bound event. Otherwise ABSTAIN.
This is perturbation-response fidelity ONLY, not human/causal importance.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .sensitivity_reference import moving_margin_effects, rank_effects


def cheap_feature_matrix(
    x: np.ndarray, class_mean: np.ndarray,
    baseline: np.ndarray, components: Sequence,
) -> np.ndarray:
    """All features are computed BEFORE any counterfactual model evaluation."""
    x=np.asarray(x,dtype=float)
    mu=np.asarray(class_mean,dtype=float)
    z=np.asarray(baseline,dtype=float)
    if (x.ndim!=2 or x.shape!=mu.shape or z.ndim!=1 or len(z)<2
        or not components or not np.isfinite(x).all()
        or not np.isfinite(mu).all() or not np.isfinite(z).all()):
        raise ValueError("invalid cheap feature input")
    energy=[];variances=[];profile=[];means=[];slopes=[]
    for c in components:
        p=x[c.t0:c.t1,c.c0:c.c1]
        r=mu[c.t0:c.t1,c.c0:c.c1]
        if (not len(p) or not p.shape[1] or c.t1>len(x)
            or c.c1>x.shape[1] or c.t0<0 or c.c0<0):
            raise ValueError("invalid component geometry")
        energy.append(float(np.mean(p*p)))
        variances.append(float(np.var(p)))
        profile.append(float(np.mean(np.abs(p-r))))
        means.append(float(abs(np.mean(p)-np.mean(r))))
        slopes.append(float(np.mean(np.abs(np.diff(p,axis=0))))
                      if len(p)>1 else 0.)
    raw=np.stack((energy,variances,profile,means),axis=1)
    # Original four-term BEACON z-score prior, retaining each constituent.
    scales=np.std(raw,axis=0)
    norm=(raw-np.mean(raw,axis=0))/np.maximum(scales,1e-12)
    norm[:,scales<=1e-12]=0.
    original_prior=np.sum(norm,axis=1)
    m=np.sort(z)
    baseline_margin=float(m[-1]-m[-2])
    n=len(components)
    out=np.zeros((n,17),dtype=float)
    out[:,:4]=norm
    out[:,4]=original_prior
    out[:,5:9]=np.log1p(np.maximum(raw,0.))
    out[:,9]=np.log1p(np.maximum(slopes,0.))
    out[:,10]=np.log1p(max(baseline_margin,0.))
    for j,c in enumerate(components):
        out[j,11]=float(c.time_bin)/max(
            1,max(k.time_bin for k in components))
        out[j,12]=float(c.group)/max(
            1,max(k.group for k in components))
        out[j,13]=float(c.t1-c.t0)/len(x)
        out[j,14]=float(c.c1-c.c0)/x.shape[1]
        out[j,15]=1. if c.group==0 else 0.
        out[j,16]=1. if c.group==1 else 0.
    if not np.isfinite(out).all():
        raise ValueError("nonfinite cheap features")
    return out


def full_model_responses(x: np.ndarray, baseline: np.ndarray,
                         components: Sequence, predict_many) -> np.ndarray:
    """Exactly M additional model evaluations for offline training/assessment."""
    from .contrastive import neutralize_interp
    probes=np.stack([neutralize_interp(x,c) for c in components])
    scores=np.asarray(predict_many(probes),dtype=float)
    if scores.shape!=(len(components),len(baseline)):
        raise ValueError("incorrect number of model probes")
    return moving_margin_effects(baseline,scores)


@dataclass
class ResponseSurrogate:
    feature_scaler: StandardScaler
    ridge: Ridge
    group_scales: np.ndarray
    features: int
    fit_window_count: int
    trained_effect_count: int

    def predict(self, features: np.ndarray, components: Sequence) -> tuple[np.ndarray,np.ndarray]:
        z=np.asarray(features,dtype=float)
        if z.ndim!=2 or z.shape!=(len(components),self.features):
            raise ValueError("surrogate feature geometry mismatch")
        pred=np.maximum(0.,np.expm1(np.clip(
            self.ridge.predict(self.feature_scaler.transform(z)),-10.,10.)))
        scale=np.asarray([self.group_scales[int(c.group)] for c in components])
        if not np.isfinite(pred).all() or not np.isfinite(scale).all():
            raise ValueError("nonfinite surrogate")
        return pred,scale


def fit_response_surrogate(
    features: Sequence[np.ndarray],
    effects: Sequence[np.ndarray],
    components: Sequence, ridge_alpha: float = 100.0,
) -> ResponseSurrogate:
    if not features or len(features)!=len(effects) or ridge_alpha<=0:
        raise ValueError("invalid surrogate fit examples or ridge penalty")
    n=len(components)
    x=np.concatenate([np.asarray(f,dtype=float) for f in features])
    y=np.concatenate([np.asarray(e,dtype=float) for e in effects])
    if (x.ndim!=2 or x.shape[0]!=y.size
        or any(np.shape(f)!=(n,x.shape[1]) for f in features)
        or any(np.shape(v)!=(n,) for v in effects)
        or not np.isfinite(x).all() or not np.isfinite(y).all()
        or np.any(y<0)):
        raise ValueError("invalid surrogate fit data")
    transformer=StandardScaler().fit(x)
    model=Ridge(alpha=ridge_alpha).fit(transformer.transform(x),np.log1p(y))
    pred=np.maximum(0.,np.expm1(np.clip(model.predict(
        transformer.transform(x)),-10.,10.)))
    groups=np.asarray([int(c.group) for c in components])
    if np.any(groups<0):
        raise ValueError("invalid component group identity")
    gids=np.tile(groups,len(features))
    # Heteroscedastic scale is learned on fit-subjects ONLY.
    residuals=np.abs(y-pred)
    scale=np.asarray([
        max(float(np.quantile(residuals[gids==g],.75)),1e-3)
        for g in range(int(groups.max())+1)
    ])
    return ResponseSurrogate(transformer,model,scale,x.shape[1],
                             len(features),len(y))


@dataclass(frozen=True)
class SubjectBound:
    alpha: float
    calibration_subjects: int
    cutoff_rank: int
    quantile: float
    minimum_windows_per_subject: int
    maximum_windows_per_subject: int

    def upper(self, predicted: np.ndarray, scale: np.ndarray) -> np.ndarray:
        p=np.asarray(predicted,dtype=float)
        s=np.asarray(scale,dtype=float)
        if (p.ndim!=1 or s.shape!=p.shape or not np.isfinite(p).all()
            or not np.isfinite(s).all() or np.any(s<=0)):
            raise ValueError("invalid bound geometry")
        if np.isinf(self.quantile):
            return np.full_like(p,np.inf)
        return np.maximum(0.,p+self.quantile*s)


def calibrate_subject_bound(
    records: Sequence[dict], alpha: float = .2,
    expected_windows_per_subject: int = 6,
) -> tuple[SubjectBound,list[dict]]:
    """One score PER SUBJECT = largest normalized underprediction over windows, M.

    Subject-level block calibration avoids treating overlapping time windows
    as iid data. For exchangeable subject-blocks the conformal k-th order
    statistic gives marginal probability >=1-alpha of bounding ALL sampled
    windows/components for a new block. When k>S, the honest upper bound is
    infinite (no automatic certificate).
    """
    if not (0<alpha<1) or expected_windows_per_subject<1:
        raise ValueError("invalid conformal calibration level")
    if not records:
        raise ValueError("empty subject calibration")
    groups={}
    for r in records:
        subject=int(r["subject"])
        true=np.asarray(r["effects"],dtype=float)
        pred=np.asarray(r["predicted"],dtype=float)
        scale=np.asarray(r["scale"],dtype=float)
        if (true.ndim!=1 or len(true)==0 or pred.shape!=true.shape
            or scale.shape!=true.shape or not np.isfinite(true).all()
            or not np.isfinite(pred).all() or not np.isfinite(scale).all()
            or np.any(true<0) or np.any(scale<=0)):
            raise ValueError("invalid calibration effect/scale arrays")
        groups.setdefault(subject,[]).append(
            float(np.max((true-pred)/scale)))
    counts=[len(group) for group in groups.values()]
    if not all(v==expected_windows_per_subject for v in counts):
        raise ValueError("every calibration subject needs the same window count")
    scores=np.sort([max(v) for v in groups.values()])
    S=len(scores)
    k=ceil((S+1)*(1-alpha))
    cutoff=float(scores[k-1]) if k<=S else np.inf
    bound=SubjectBound(alpha,S,k,cutoff,min(counts),max(counts))
    return bound,[{"subject":s,"n_windows":len(v),
                   "max_one_sided_standardized_residual":max(v)}
                  for s,v in sorted(groups.items())]


@dataclass
class CertifiedExplanation:
    ranked: list[int]
    selected: list[int]
    measured: list[float]
    extra_model_evaluations: int
    forward_batches: int
    certified: bool
    third_effect: float | None
    maximum_unqueried_upper_bound: float
    reason: str


def affordable_certificate(
    selected: Sequence[int],measured: Sequence[float],
    upper_bound: np.ndarray, number_of_components: int,
    tolerance: float = 1e-8,
) -> tuple[bool,float|None,float]:
    chosen=list(map(int,selected))
    y=np.asarray(measured,dtype=float)
    upper=np.asarray(upper_bound,dtype=float)
    if (upper.shape!=(number_of_components,)
        or y.shape!=(len(chosen),)
        or len(set(chosen))!=len(chosen)
        or any(g<0 or g>=number_of_components for g in chosen)
        or not np.isfinite(y).all() or np.any(y<0)
        or np.any(np.isnan(upper)) or np.any(upper<0)
        or tolerance<0):
        raise ValueError("invalid measured evidence or conformal envelope")
    if len(chosen)<3:
        return False,None,float(np.max(upper))
    third=float(np.sort(y)[-3])
    missing=np.ones(number_of_components,dtype=bool)
    missing[chosen]=False
    maximum=float(np.max(upper[missing])) if missing.any() else float("-inf")
    return bool(third>maximum+tolerance),third,maximum


def run_certified_explanation(
    x: np.ndarray, baseline: np.ndarray, components: Sequence,
    predicted_effect: np.ndarray, bounds: SubjectBound,
    effect_scale: np.ndarray, predict_many: Callable,
    max_queries: int = 16, step: int = 4,
) -> CertifiedExplanation:
    """Queries highest upper-bound first; returns a top3 ONLY IF certified.

    The Q=M responses are not available here. The *flag* is determined by
    measured responses and train-calibrated envelopes alone.
    """
    if (step<1 or max_queries<3 or max_queries>len(components)
        or not np.isfinite(x).all() or not np.isfinite(baseline).all()):
        raise ValueError("invalid online query budget")
    upper=bounds.upper(predicted_effect,effect_scale)
    ids=np.argsort(-upper,kind="stable").astype(int).tolist()
    selected=[];effects=[];batches=0
    valid=False;third=None;unqueried=np.inf
    for pos in range(0,max_queries,step):
        chunk=ids[pos:min(pos+step,max_queries)]
        probes=np.stack([__neutralize(x,components[i]) for i in chunk])
        outputs=np.asarray(predict_many(probes))
        if outputs.shape!=(len(chunk),len(baseline)):
            raise ValueError("model returned wrong number of new observations")
        score=moving_margin_effects(baseline,outputs)
        selected.extend(chunk);effects.extend(score.tolist());batches+=1
        valid,third,unqueried=affordable_certificate(
            selected,effects,upper,len(components))
        if valid:
            break
    rank=[selected[j] for j in np.argsort(-np.array(effects),kind="stable")]
    return CertifiedExplanation(rank,selected,effects,len(selected),batches,
        valid,third,unqueried,
        "bounded_top3" if valid else "abstain_unqueried_ambiguous")


def __neutralize(x,component):
    from .contrastive import neutralize_interp
    return neutralize_interp(x,component)
