"""Black-box lower bound: 16 coalition queries cannot identify 36 Shapley values.

Even in the EASIER additive game v(S)=sum_{g in S} theta_g,
an unrestricted 36-dimensional theta vector cannot be identified from
16 scalar evaluations. The observed subset-value design matrix has rank
at most 16, leaving a nullspace of dimension >=20. Two distinct additive
games can therefore agree on every queried coalition but have different
Shapley vectors. This is an identifiability statement, not a performance
benchmark of canonical KernelSHAP (which cannot be claimed run at Q16).
"""
import numpy as np


def test_under_budgeted_shapley_nonidentifiability_even_for_additive_game():
    rng=np.random.default_rng(57)
    m=36
    q=16
    masks=rng.binomial(1,.5,size=(q,m)).astype(float)
    U,S,Vt=np.linalg.svd(masks,full_matrices=True)
    rank=int(np.sum(S>1e-10))
    assert rank<=q
    assert m-rank>=20
    hidden=Vt[-1]
    assert np.linalg.norm(hidden)>0.9999999
    assert np.linalg.norm(masks@hidden)<1e-10
    theta=rng.normal(size=m)
    second=theta+hidden
    # All actual observed v(S)=mask dot theta exactly coincide.
    assert np.allclose(masks@theta,masks@second,atol=1e-10)
    # But in an additive game phi_g=theta_g: explanations differ.
    assert np.linalg.norm(theta-second)>0.999
    # In particular, at least one component's importance differs
    # although no one of 16 queries can distinguish these two games.
    assert np.max(np.abs(theta-second))>0.05
