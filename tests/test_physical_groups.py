"""UCI HAR body/total acceleration coupling controls for counterfactual patches."""
import numpy as np
import pytest
from beaconxai.verified_counterfactual import splice
from beaconxai.physical_groups import uci_har_coupled_groups, coupled_priority

class C:
    def __init__(self, group, t0, c0):
        self.group = group
        self.time_bin = t0
        self.t0 = t0 * 4
        self.t1 = t0 * 4 + 4
        self.c0 = c0
        self.c1 = c0 + 3

COMP = [C(g, t, g * 3) for g in range(3) for t in range(2)]

def test_accelerations_together_gyro_separately():
    groups = uci_har_coupled_groups(COMP)
    assert len(groups) == 4
    x = np.zeros((8, 9))
    donor = np.ones_like(x)
    edited = splice(x, donor, groups, [0], taper=0)
    assert np.all(edited[:4, 0:3] == 1)
    assert np.all(edited[:4, 6:9] == 1)
    assert np.all(edited[:, 3:6] == 0)
    assert np.all(edited[4:] == 0)
    gyro = splice(x, donor, groups, [2], taper=0)
    assert np.all(gyro[:4, 3:6] == 1)
    assert np.all(gyro[:, 6:9] == 0)

def test_one_coupled_probe_one_model_input():
    groups = uci_har_coupled_groups(COMP)
    x = np.zeros((8,9)); donor = np.ones_like(x)
    def classifier(xs):
        return np.stack([np.zeros(len(xs)), xs[:,:4,0].mean(axis=1)+xs[:,:4,6].mean(axis=1)-1],axis=1)
    assert np.argmax(classifier(splice(x, donor, groups, [0], taper=0)[None])[0]) == 1

def test_priority_and_overlapping_regions_are_rejected():
    groups = uci_har_coupled_groups(COMP)
    assert coupled_priority(np.array([1.,2.,3.,4.,5.,6.]), COMP).tolist() == [5.,6.,3.,4.]
    x = np.zeros((8,9)); donor = np.ones_like(x)
    with pytest.raises(ValueError, match="overlapping"):
        splice(x, donor, [groups[0], groups[0]], [0,1])
    with pytest.raises(ValueError):
        uci_har_coupled_groups(COMP[:-1])


def test_gravity_residual_proxy_detects_inconsistent_single_channel_splice():
    from scripts.run_verified_counterfactual_benchmark import gravity_residual_roughness
    x = np.zeros((8, 9))
    d = np.ones_like(x)
    coupled = uci_har_coupled_groups(COMP)
    paired = splice(x, d, coupled, [0], taper=0)
    independent = splice(x, d, COMP, [4], taper=0)
    mean = np.zeros((1, 1, 9))
    sigma = np.ones((1, 1, 9))
    assert gravity_residual_roughness(x, mean, sigma) == 0
    assert gravity_residual_roughness(paired, mean, sigma) == 0
    assert gravity_residual_roughness(independent, mean, sigma) > 0


def test_gravity_preserving_component_in_original_units():
    mu=np.array([.2,.3,.4,0.,0.,0.,.5,.6,.7]).reshape(1,1,9)
    sd=np.array([2.,2.5,3.,1.,1.,1.,.4,.5,.6]).reshape(1,1,9)
    coupled=uci_har_coupled_groups(COMP,gravity_preserving=True,mean=mu,scale=sd)
    rng=np.random.default_rng(7)
    x=rng.normal(size=(8,9))
    donor=rng.normal(size=(8,9))
    edited=splice(x,donor,coupled,[0,3],taper=2)
    raw=lambda a:a*sd.reshape(-1)+mu.reshape(-1)
    before=raw(x)[:,6:9]-raw(x)[:,:3]
    after=raw(edited)[:,6:9]-raw(edited)[:,:3]
    np.testing.assert_allclose(before,after,atol=1e-12)
    assert not np.array_equal(edited[:,0:3],x[:,0:3])
    assert not np.array_equal(edited[:,3:6],x[:,3:6])
    with pytest.raises(ValueError):
        uci_har_coupled_groups(COMP,gravity_preserving=True)
    with pytest.raises(ValueError):
        uci_har_coupled_groups(COMP,gravity_preserving=True,
                               mean=np.full((1,1,9), np.nan),
                               scale=sd)
