"""Budgeted BEACON: residual-covariance inference for UNQUERIED components.

The estimator is an ordinary covariance-regularized conditional linear
predictor; it is NOT a Shapley value, Gaussian-process guarantee, or
calibrated causal attribution.

Fit strictly on TRAIN surrogate-subject full-Q=M effects, once per frozen
classifier. On a new window, compute all 36 cheap features, measure 8 true
responses, condition predictions on the 8 actual outcomes, select 8 new
components using conditional mean+uncertainty, measure those 8, and return
an inferred global top-3 while retaining the list of 16 measured responses.

Conditional means on unmeasured G given measured S:
  mu_G|S = mu_G + K[G,S] (K[S,S]+sigma^2 I)^-1 (y_S-mu_S)
Conditional uncertainty is treated solely as a HEURISTIC for next probes:
  V_G|S = diag(K_GG - K_GS (K_SS+sigma^2 I)^-1 K_SG).
No claim about coverage, exact top-3, or unseen physical signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .conformal_evidence import (
    ResponseSurrogate, fit_response_surrogate,
    cheap_feature_matrix,
)
from .contrastive import neutralize_interp
from .directed_evidence import signed_fragment_evidence


@dataclass
class ResidualConditional:
    surrogate: ResponseSurrogate
    covariance: np.ndarray
    ridge_noise: float
    fitted_windows: int
    measured_effects_for_training: int

    def conditional(
        self, feature_matrix: np.ndarray, components: Sequence,
        observed_ids: Sequence[int], observed_values: Sequence[float],
    ) -> tuple[np.ndarray,np.ndarray]:
        """Return nonnegative conditional response and diagonal uncertainty.

        This method is a deterministic linear adjustment, NOT an
        exchangeability/coverage confidence bound. Exact measured values
        overwrite predictions for those IDs.
        """
        base,_=self.surrogate.predict(feature_matrix,components)
        selected=np.asarray(observed_ids,dtype=int)
        actual=np.asarray(observed_values,dtype=float)
        m=len(base)
        if (self.covariance.shape!=(m,m) or selected.ndim!=1
            or actual.shape!=selected.shape
            or len(set(selected.tolist()))!=len(selected)
            or np.any(selected<0) or np.any(selected>=m)
            or not np.isfinite(actual).all() or np.any(actual<0)):
            raise ValueError("invalid observed signed-model effects")
        k=self.covariance
        var=np.diag(k).copy()
        adjusted=base.copy()
        if len(selected):
            A=k[np.ix_(selected,selected)]+self.ridge_noise*np.eye(len(selected))
            cross=k[:,selected]
            try:
                updates=np.linalg.solve(A,actual-base[selected])
                shrink=np.linalg.solve(A,cross.T)
            except np.linalg.LinAlgError as ex:
                raise ValueError("residual covariance is singular") from ex
            adjusted=base+cross@updates
            var-=np.sum(cross*shrink.T,axis=1)
            adjusted[selected]=actual
            var[selected]=0.
        return np.maximum(adjusted,0.),np.maximum(var,0.)


def fit_residual_conditional(
    features: Sequence[np.ndarray],observed_effects: Sequence[np.ndarray],
    components: Sequence,
    ridge_alpha:float=100., temporal_width:float=2.0,
    empirical_weight:float=.65, ridge_noise:float=.05,
) -> ResidualConditional:
    """PSD mixture of empirical centered and smooth geometry covariances.

    The simple fixed smooth kernel encodes sensor/time similarity, and
    regularization prevents interpolation of sampling noise as truth.
    Hyperparameters are chosen before OUTER held-out evaluation.
    """
    if (not features or len(features)!=len(observed_effects)
        or temporal_width<=0 or not 0<=empirical_weight<=1
        or ridge_noise<=0 or not np.isfinite(temporal_width)
        or not np.isfinite(ridge_noise)):
        raise ValueError("invalid frozen residual covariance parameters")
    surrogate=fit_response_surrogate(
        features,observed_effects,components,ridge_alpha=ridge_alpha)
    f=np.asarray(features,dtype=float)
    y=np.asarray(observed_effects,dtype=float)
    if y.ndim!=2 or y.shape!=(len(f),len(components)):
        raise ValueError("invalid training residual geometry")
    predictions=np.stack([surrogate.predict(z,components)[0] for z in f])
    residual=y-predictions
    if len(y)>1:
        centered=residual-residual.mean(axis=0)
        empirical=centered.T@centered/(len(y)-1)
    else:
        empirical=np.zeros((len(components),len(components)))
    variances=np.maximum(np.diag(empirical),1e-4)
    time=np.array([c.time_bin for c in components],dtype=float)
    group=np.array([c.group for c in components],dtype=int)
    distance=(time[:,None]-time[None,:])/temporal_width
    geometry=np.exp(-.5*distance**2)*(
        .15+.85*(group[:,None]==group[None,:]))
    smooth=np.sqrt(np.outer(variances,variances))*geometry
    covariance=empirical_weight*empirical+(1-empirical_weight)*smooth
    # Preserve positivity after mixed-precision model scores.
    covariance=(covariance+covariance.T)*.5
    eigmin=float(np.linalg.eigvalsh(covariance)[0])
    if eigmin<-1e-7:
        raise ValueError("residual kernel must be positive semidefinite")
    covariance+=np.eye(len(components))*1e-6
    return ResidualConditional(surrogate,covariance,ridge_noise,
                               len(y),int(y.size))


@dataclass
class ActiveCounterExplanation:
    candidate_ids: list[int]
    model_measured_positive: list[float]
    measured_only_ranking: list[int]
    conditional_ranking_all_components: list[int]
    inferred_effects_all_components: np.ndarray
    uncertainty_all_components: np.ndarray
    extra_model_evaluations: int
    forward_batches: int
    first_stage_ids: list[int]
    second_stage_ids: list[int]


def run_active_counterevidence(
    x: np.ndarray, baseline_scores: np.ndarray, components: Sequence,
    conditional: ResidualConditional, predict_many: Callable,
    first_stage_score: np.ndarray,
    q_first:int=8,q_second:int=8,
    uncertainty_bonus:float=.5,
) -> ActiveCounterExplanation:
    """True 8+8 measured calls, no Q36 oracle or held-out test labels.

    The final top-3 MAY contain unmeasured fragments: those are labelled
    INFERRED and never presented as verified model perturbations.
    """
    base=np.asarray(baseline_scores,dtype=float)
    cheap=np.asarray(first_stage_score,dtype=float)
    m=len(components)
    if (x.ndim!=2 or base.ndim!=1 or len(base)<2
        or cheap.shape!=(m,) or q_first<3 or q_second<1
        or q_first+q_second>=m or uncertainty_bonus<0
        or not np.isfinite(x).all() or not np.isfinite(base).all()
        or not np.isfinite(cheap).all()
        or not np.isfinite(uncertainty_bonus)):
        raise ValueError("invalid active BEACON Q8/Q16 inputs")
    source=int(np.argmax(base))
    features=cheap_feature_matrix(
        x,conditional.class_profile,base,components) if hasattr(
            conditional,'class_profile') else None
    # The caller must pass feature_matrix already computed from
    # training-only mean of ORIGINAL predicted class; avoid silently
    # guessing the class mean inside this budgeted algorithm.
    if features is None:
        raise ValueError("provide feature_matrix to online active audit")
    return _run_active_with_features(x,base,components,conditional,
                                     predict_many,cheap,features,
                                     q_first,q_second,uncertainty_bonus)


def active_counterevidence_from_features(
    x: np.ndarray, baseline_scores: np.ndarray, components: Sequence,
    cheap_features: np.ndarray, conditional: ResidualConditional,
    predict_many: Callable,
    first_stage_score: np.ndarray,
    q_first:int=8,q_second:int=8,
    uncertainty_bonus:float=.5,
) -> ActiveCounterExplanation:
    """Practical API, exposing exactly the cheap observations used online."""
    base=np.asarray(baseline_scores,dtype=float)
    cheap=np.asarray(first_stage_score,dtype=float)
    f=np.asarray(cheap_features,dtype=float)
    m=len(components)
    if (x.ndim!=2 or base.ndim!=1 or len(base)<2
        or cheap.shape!=(m,) or f.shape!=(m,conditional.surrogate.features)
        or q_first<3 or q_second<1 or q_first+q_second>=m
        or not np.isfinite(x).all() or not np.isfinite(base).all()
        or not np.isfinite(cheap).all() or not np.isfinite(f).all()
        or uncertainty_bonus<0 or not np.isfinite(uncertainty_bonus)):
        raise ValueError("invalid real Q8+8 inputs")
    return _run_active_with_features(x,base,components,conditional,
                                     predict_many,cheap,f,
                                     q_first,q_second,uncertainty_bonus)


def _run_active_with_features(
    x,base,components,conditional,predict_many,initial_score,features,
    q_first,q_second,uncertainty_bonus,
) -> ActiveCounterExplanation:
    m=len(components)
    first=np.argsort(-initial_score,kind="stable")[:q_first].tolist()

    def query(ids):
        patched=np.stack([neutralize_interp(x,components[j]) for j in ids])
        logits=np.asarray(predict_many(patched),dtype=float)
        if logits.shape!=(len(ids),len(base)):
            raise ValueError("model returned incorrect number of evaluations")
        _,against,_=signed_fragment_evidence(base,logits)
        return against.astype(float).tolist()

    first_responses=query(first)
    posterior,variance=conditional.conditional(
        features,components,first,first_responses)
    score=posterior+uncertainty_bonus*np.sqrt(variance)
    score[first]=-np.inf
    second=np.argsort(-score,kind="stable")[:q_second].tolist()
    if len(second)!=q_second or set(first)&set(second):
        raise AssertionError("failed to select new second-stage components")
    second_responses=query(second)
    all_ids=first+second
    measured=first_responses+second_responses
    posterior,var=conditional.conditional(features,components,all_ids,measured)
    # Two valid outputs: (1) rank ONLY actually measured fragments;
    # (2) top3 among ALL components, extrapolating unqueried responses.
    measured_rank=[all_ids[j] for j in np.argsort(-np.asarray(measured),
                                                  kind="stable")]
    inferred_rank=np.argsort(-posterior,kind="stable").astype(int).tolist()
    return ActiveCounterExplanation(
        candidate_ids=all_ids,model_measured_positive=measured,
        measured_only_ranking=measured_rank,
        conditional_ranking_all_components=inferred_rank,
        inferred_effects_all_components=posterior,
        uncertainty_all_components=var,
        extra_model_evaluations=len(all_ids),forward_batches=2,
        first_stage_ids=first,second_stage_ids=second,
    )
