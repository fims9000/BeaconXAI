"""Sparse refinement never invents target flips or exceeds remaining Q."""
import numpy as np
import pytest

from beaconxai.cf_budgeted_sparse_refine import refine_verified_cf_sparse


def test_finds_smallest_tested_success_and_counts_all_calls():
    x=np.zeros((12,9))
    valid=x.copy()
    valid[:4,:3]=1
    valid[:4,6:9]=1
    count=[0]
    def model(z):
        count[0]+=len(z)
        margin=z[:,:,:3].sum(axis=(1,2))-6.
        return np.column_stack((np.zeros(len(z)),margin))
    out=refine_verified_cf_sparse(
        x,valid,np.zeros(9),np.ones(9),model,
        target_class=1,budget_remaining=3,
        fraction_caps=(.2,.35,.5),bins=3)
    assert out.status=="verified_sparse"
    assert not out.original_retained
    assert out.accepted_fraction_cap==.35
    assert out.model_evaluations==count[0]==2
    assert out.attempted_fractions==(.2,.35)
    assert out.edited_fraction<=.35


def test_abstains_from_unconfirmed_reduction_and_preserves_verified_input():
    x=np.zeros((12,9))
    y=np.ones((12,9))
    def always_source(z):
        return np.column_stack((np.ones(len(z)),np.zeros(len(z))))
    out=refine_verified_cf_sparse(
        x,y,np.zeros(9),np.ones(9),always_source,
        target_class=1,budget_remaining=2,
        fraction_caps=(.2,.5,.8),bins=3)
    assert out.status=="fallback_original_verified"
    assert out.original_retained
    assert out.model_evaluations==2
    assert out.attempted_fractions==(.2,.5)
    np.testing.assert_array_equal(out.x,y)


def test_zero_remaining_budget_does_not_access_black_box():
    x=np.zeros((12,9))
    y=np.ones((12,9))
    def forbidden(_):
        raise AssertionError("model called")
    out=refine_verified_cf_sparse(
        x,y,np.zeros(9),np.ones(9),forbidden,
        target_class=1,budget_remaining=0,bins=3)
    assert out.model_evaluations==0
    assert out.original_retained and not out.attempted_fractions


def test_invalid_fraction_grid_and_nonfinite_model_result():
    x=np.zeros((12,9))
    with pytest.raises(ValueError):
        refine_verified_cf_sparse(
            x,x,np.zeros(9),np.ones(9),lambda z:z,
            target_class=1,budget_remaining=3,
            fraction_caps=(.5,.2))
    with pytest.raises(ValueError):
        refine_verified_cf_sparse(
            x,x,np.zeros(9),np.ones(9),
            lambda z:np.full((len(z),2),np.nan),
            target_class=1,budget_remaining=1,bins=3)
