"""Orthogonal acceleration-channel projection for UCI HAR counterfactuals.

Given standardized source x and proposed edit y, solve at each timestamp:
  min_{b',t'} ||b'-b_y||_2^2 + ||t'-t_y||_2^2,
  subject to t'-b' = t_x-b_x,
in *unnormalized* acceleration units. Gyroscope channels remain unchanged.

This enforces one affine identity only. It does not enforce gait dynamics,
sensor feasibility, causal intervention or membership in real data manifold.
"""
from __future__ import annotations

import numpy as np


def project_uci_har_preserve_gravity(
    original: np.ndarray, proposed: np.ndarray,
    train_mean: np.ndarray, train_scale: np.ndarray,
) -> np.ndarray:
    x=np.asarray(original,dtype=float)
    y=np.asarray(proposed,dtype=float)
    mu=np.asarray(train_mean,dtype=float).reshape(-1)
    sd=np.asarray(train_scale,dtype=float).reshape(-1)
    if (x.ndim!=2 or x.shape[1]!=9 or y.shape!=x.shape or
            mu.shape!=(9,) or sd.shape!=(9,) or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            not np.isfinite(mu).all() or not np.isfinite(sd).all() or
            np.any(sd<=0)):
        raise ValueError("requires finite [T,9] arrays and positive train scales")

    src=x*sd+mu
    want=y*sd+mu
    residual=src[:,6:9]-src[:,:3]
    body=.5*(want[:,:3]+want[:,6:9]-residual)
    total=body+residual
    out=want.copy()
    out[:,:3]=body
    out[:,6:9]=total
    z=(out-mu)/sd
    if not np.isfinite(z).all():
        raise ValueError("projection generated nonfinite normalized values")
    return z
