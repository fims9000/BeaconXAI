"""Official UCR archive invariants and one-shared-Q transfer-aware evaluation."""
import numpy as np
import pytest

from beaconxai.contrastive import make_components
from beaconxai.transfer_candidate_policy import (
    N_FEATURES,fit_candidate_transfer_prior,
)
from scripts.preprocess_ucr_active_transfer import prepare_ucr_transfer
from scripts.run_ucr_active_transfer import (
    Q,N_SELECTED,METHODS,paired_primary,stratified_threeway,
    evaluate_one,measure_training_candidates,
)


@pytest.mark.parametrize("name,train_n,test_n,length,classes",[
    ("Wafer",1000,6164,152,2),
    ("TwoPatterns",1000,4000,128,4),
])
def test_unchanged_official_ucr_splits(tmp_path,name,train_n,test_n,length,classes):
    xtr=np.zeros((train_n,1,length),dtype=np.float32)
    xte=np.zeros((test_n,1,length),dtype=np.float32)
    tr=np.arange(train_n)%classes
    te=np.arange(test_n)%classes
    output=tmp_path/f"{name}.npz"
    meta=prepare_ucr_transfer(name,xtr,tr,xte,te,output)
    assert meta["original_device_or_subject_disjoint_verified"] is False
    with np.load(output) as z:
        assert z["x_train"].shape==(train_n,length,1)
        assert z["x_test"].shape==(test_n,length,1)
        fit,risk,gate=stratified_threeway(z["y_train"],classes)
        assert not(set(fit)&set(risk))
        assert not(set(fit)&set(gate))
        assert not(set(risk)&set(gate))
        assert sorted(fit.tolist()+risk.tolist()+gate.tolist())==list(range(1000))


def test_archive_geometry_rejection_before_model_predictions(tmp_path):
    with pytest.raises(ValueError,match="official"):
        prepare_ucr_transfer(
            "Wafer",np.zeros((999,1,152)),np.zeros(999),
            np.zeros((6164,1,152)),np.zeros(6164),tmp_path/"bad.npz",
        )


def test_offline_training_labels_pay_exact_measured_input_calls():
    x=np.zeros((4,50,1))
    base=np.tile(np.array([0.,-.2]),(4,1))
    groups=make_components((50,1),10,"all_channels")
    from types import SimpleNamespace
    b=SimpleNamespace(
        donors={1:np.stack([
            np.full((50,1),v) for v in (1.,.9,.8,.7,.6)
        ])},
        selected_train_indices={1:np.arange(5)},
    )
    screened={1:np.ones(5,dtype=int)}
    profiles={
        0:(np.zeros((50,1)),np.ones((50,1))),
        1:(np.ones((50,1)),np.ones((50,1))),
    }
    calls={"examples":0}
    def fake(batch):
        calls["examples"]+=len(batch)
        p=batch.mean(axis=(1,2))
        return np.column_stack([np.zeros(len(batch)),p-.16])
    features,y,report=measure_training_candidates(
        x,base,b,screened,groups,profiles,fake,calls,name="riskfit",
    )
    assert len(features)==len(y)==4*20
    assert features.shape[1]==N_FEATURES
    assert report["offline_model_input_evaluations"]==calls["examples"]
    assert report["candidate_features_see_D1_D2_only"]
    assert (y>=0).all() and (y<=1).all()


@pytest.mark.parametrize("method",METHODS)
def test_test_one_certified_only_after_direct_model_calls_with_posthoc_cost(method):
    x=np.zeros((50,1))
    groups=make_components((50,1),10,"all_channels")
    donors=tuple(np.full_like(x,v) for v in (1.,.9,.8,.7))
    profile={
        0:(np.zeros_like(x),np.ones_like(x)),
        1:(np.ones_like(x),np.ones_like(x)),
    }
    prior=fit_candidate_transfer_prior(
        np.zeros((10,N_FEATURES)),np.ones(10,dtype=int)
    )
    c={"examples":0}
    def fake(batch):
        c["examples"]+=len(batch)
        p=batch.mean(axis=(1,2))
        return np.column_stack([np.zeros(len(batch)),p-.07])
    r=evaluate_one(
        x,np.array([0.,-.07]),donors,groups,profile,prior,fake,c,
        method=method,
    )
    assert r["online_Q"]<=Q
    assert r["online_Q"]+r["posthoc_D3D4_queries"]==c["examples"]
    if r["online_certified"]:
        assert r["selected"] is not None
        assert r["online_max_support"]<=.35
        assert r["posthoc_D3D4_queries"]==2
    else:
        assert r["selected"] is None and r["posthoc_D3D4_queries"]==0


def test_missing_four_donors_stays_in_full_denominator():
    groups=make_components((50,1),10,"all_channels")
    profiles={0:(np.zeros((50,1)),np.ones((50,1))),
              1:(np.ones((50,1)),np.ones((50,1)))}
    prior=fit_candidate_transfer_prior(
        np.empty((0,N_FEATURES)),np.empty((0,),dtype=int)
    )
    r=evaluate_one(
        np.zeros((50,1)),np.array([0.,-.2]),None,
        groups,profiles,prior,lambda a:None,{"examples":0},
        method="learned_online",
    )
    assert r["online_Q"]==0 and r["fresh_D3D4_transfer"]==0
    assert not r["online_certified"]


def test_primary_is_paired_full_320_with_exact_mcnemar():
    rows=[]
    for i in range(N_SELECTED):
        for method in METHODS:
            v=int(i<12 if method=="learned_online" else
                  i<8 if method=="static_sparse" else i<5)
            rows.append({
                "sample":i,"method":method,
                "fresh_D3D4_transfer":v,
            })
    r=paired_primary(rows)
    assert r["both"]==8 and r["learned_only"]==4
    assert r["static_only"]==0 and r["neither"]==308
    assert r["learned_fresh"]==12 and r["static_fresh"]==8
    assert r["exact_two_sided_mcnemar"]==.125
