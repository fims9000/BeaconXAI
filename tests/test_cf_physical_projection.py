"""Projection is an exact minimum-distance affine constraint, not realism proof."""
import numpy as np
import pytest

from beaconxai.cf_physical_projection import project_uci_har_preserve_gravity


def test_restores_acceleration_residual_at_every_time_point():
    rng=np.random.default_rng(24)
    mu=np.array([.3,.1,.2,0.,0.,0.,.9,.8,.7]).reshape(1,1,9)
    sd=np.array([.3,.4,.5,1.,2.,3.,.7,.8,.9]).reshape(1,1,9)
    x=rng.normal(size=(128,9))
    y=rng.normal(size=(128,9))
    z=project_uci_har_preserve_gravity(x,y,mu,sd)
    raw=lambda a:a*sd.reshape(-1)+mu.reshape(-1)
    np.testing.assert_allclose(raw(z)[:,6:9]-raw(z)[:,:3],
                               raw(x)[:,6:9]-raw(x)[:,:3],atol=1e-12)
    np.testing.assert_allclose(z[:,3:6],y[:,3:6],atol=1e-12)


def test_projection_is_closer_than_other_feasible_shift():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(20,9))
    y=rng.normal(size=(20,9))
    mu=np.zeros((9,))
    sd=np.ones((9,))
    z=project_uci_har_preserve_gravity(x,y,mu,sd)
    shifted=z.copy()
    shifted[:,:3]+=2.0
    shifted[:,6:9]+=2.0
    assert np.linalg.norm(z-y)<np.linalg.norm(shifted-y)


def test_idempotent_and_unnecessary_if_already_valid():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(20,9))
    sd=np.arange(1,10,dtype=float)
    mu=np.arange(9)*.2
    np.testing.assert_allclose(
        project_uci_har_preserve_gravity(x,x,mu,sd),x,atol=1e-12)
    z=project_uci_har_preserve_gravity(x,x+.3,mu,sd)
    np.testing.assert_allclose(
        project_uci_har_preserve_gravity(x,z,mu,sd),z,atol=1e-12)


def test_invalid_normalizer_or_signal_rejected():
    x=np.zeros((3,9))
    for kwargs in (
        (x,np.zeros((4,9)),np.zeros(9),np.ones(9)),
        (x,x,np.zeros(9),np.zeros(9)),
        (x,x,np.full(9,np.nan),np.ones(9)),
    ):
        with pytest.raises(ValueError):
            project_uci_har_preserve_gravity(*kwargs)
