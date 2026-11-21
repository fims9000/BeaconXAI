"""Signed evidence for the ORIGINAL predicted class, not moving top-two.

Important correction to BEACON's original absolute moving-margin proxy:
when an intervention changes the winning class, recomputing the top-two
margin with the new winner can conceal a decisive class reversal.

For a=argmax f(x), hold a fixed under every intervention T_g:
m_a(z)=l_a(z)-max_{c!=a}l_c(z)
C_g=m_a(T_g(x))-m_a(x)
C_g>0 => removing fragment STRENGTHENS original class a; fragment
         opposed the original class (local counter-evidence).
C_g<0 => removing fragment WEAKENS a; fragment supported a.
These are operational, perturbation-relative signs. Not causal proof.

This module deliberately does not assert Shapley-additivity, interaction
faithfulness, physical realism, or ground-truth semantic explanation.
"""
from __future__ import annotations
import numpy as np


def fixed_source_margin(scores: np.ndarray, source: int) -> np.ndarray:
    """[classes] -> scalar, [probes, classes] -> [probes]."""
    arr=np.asarray(scores,dtype=float)
    if (arr.ndim not in (1,2) or arr.shape[-1]<2
        or source<0 or source>=arr.shape[-1]
        or not np.isfinite(arr).all()):
        raise ValueError("invalid class scores or source index")
    rival=np.concatenate((arr[..., :source],arr[..., source+1:]),axis=-1)
    margin=arr[...,source]-np.max(rival,axis=-1)
    return margin


def signed_fragment_evidence(
    baseline_scores: np.ndarray, perturbed_scores: np.ndarray,
    source: int | None = None,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return signed C, positive counter evidence, positive supporting evidence.

    Source defaults to the ORIGINAL prediction, never reselected after
    changing the input. All outputs use the SAME fixed source class.
    """
    base=np.asarray(baseline_scores,dtype=float)
    p=np.asarray(perturbed_scores,dtype=float)
    if base.ndim!=1 or p.ndim!=2 or p.shape[1]!=len(base):
        raise ValueError("invalid baseline/perturbation logits shape")
    a=int(np.argmax(base)) if source is None else int(source)
    m0=float(fixed_source_margin(base,a))
    changed=np.asarray(fixed_source_margin(p,a),dtype=float)-m0
    if not np.isfinite(changed).all():
        raise ValueError("nonfinite signed evidence")
    return changed,np.maximum(changed,0.),np.maximum(-changed,0.)


def positive_reference_ranking(values: np.ndarray) -> tuple[list[int],int]:
    """A top3 for counter-evidence is meaningful only if >=3 positives exist.

    A zero-only reference must not be presented as three contradictions.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<3 or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError("expected finite nonnegative counter-evidence")
    rank=np.argsort(-x,kind="stable").astype(int).tolist()
    return rank,int(np.count_nonzero(x>1e-8))
