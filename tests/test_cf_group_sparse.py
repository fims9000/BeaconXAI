"""Group-sparse UCI HAR projection is an exact finite-bin knapsack."""
from itertools import combinations
import numpy as np
import pytest

from beaconxai.cf_group_sparse import project_uci_har_group_sparse
from beaconxai.cf_physical_projection import project_uci_har_preserve_gravity


def test_sparse_projection_keeps_acceleration_channels_coupled():
    x=np.zeros((12,9),dtype=float)
    y=x.copy()
    y[:4,:3]=4.
    y[:4,6:9]=4.
    y[4:8,3:6]=.1
    z=project_uci_har_group_sparse(
        x,y,np.zeros(9),np.ones(9),
        maximum_modified_fraction=.25,bins=3)
    assert z.selected_cell_cost<=int(.25*x.size)
    assert z.modified_fraction<=.25
    assert z.groups==(0,)
    np.testing.assert_array_equal(z.x[:4,0:3],4)
    np.testing.assert_array_equal(z.x[:4,6:9],4)
    np.testing.assert_array_equal(z.x[4:],0)


def test_exact_best_group_combination_against_brute_force():
    rng=np.random.default_rng(6)
    x=rng.normal(size=(10,9))
    y=rng.normal(size=(10,9))
    mean=np.zeros(9)
    scale=np.ones(9)
    target=project_uci_har_preserve_gravity(x,y,mean,scale)
    fraction=.55
    res=project_uci_har_group_sparse(
        x,y,mean,scale,maximum_modified_fraction=fraction,bins=2)
    capacity=int(np.floor(fraction*x.size))
    groups=[]
    for start, end in ((0,3),(3,6)):
        for t0,t1 in ((0,5),(5,10)):
            if start==0:
                idx=np.r_[np.arange(0,3),np.arange(6,9)]
            else:
                idx=np.arange(3,6)
            cost=len(idx)*(t1-t0)
            value=float(np.sum((target[t0:t1][:,idx]-x[t0:t1][:,idx])**2))
            groups.append((cost,value))
    best=max(sum(groups[j][1] for j in chosen)
             for n in range(5)
             for chosen in combinations(range(4),n)
             if sum(groups[j][0] for j in chosen)<=capacity)
    assert res.energy_retained==pytest.approx(best)
    assert res.selected_cell_cost<=capacity


def test_preserves_raw_unit_gravity_with_unequal_scales():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(128,9))
    y=rng.normal(size=(128,9))
    mu=np.arange(9)*.2
    sd=np.arange(1,10,dtype=float)
    result=project_uci_har_group_sparse(
        x,y,mu,sd,maximum_modified_fraction=.2,bins=12)
    raw=lambda z:z*sd+mu
    np.testing.assert_allclose(
        raw(result.x)[:,6:9]-raw(result.x)[:,:3],
        raw(x)[:,6:9]-raw(x)[:,:3],atol=1e-10)
    assert result.modified_fraction<=.2
    assert result.energy_retained<=result.full_candidate_energy+1e-8


def test_empty_budget_and_invalid_bounds():
    x=np.zeros((12,9))
    y=np.ones((12,9))
    z=project_uci_har_group_sparse(
        x,y,np.zeros(9),np.ones(9),maximum_modified_fraction=.01,bins=3)
    assert z.groups==() and z.modified_fraction==0
    with pytest.raises(ValueError):
        project_uci_har_group_sparse(x,y,np.zeros(9),np.zeros(9),
                                      maximum_modified_fraction=.5)
    with pytest.raises(ValueError):
        project_uci_har_group_sparse(x,y,np.zeros(9),np.ones(9),
                                      maximum_modified_fraction=1.2)
