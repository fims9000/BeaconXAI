"""Strict CharacterTrajectories split, donor isolation and shared Q16 accounting."""
import numpy as np
import pytest

from beaconxai.contrastive import make_components
from scripts.preprocess_uea_charactertrajectories import (
    prepare_character_trajectories,
)
from scripts.run_uea_charactertrajectories_transfer_gate import (
    Q, stratified_threeway, explain_one, accepted_features,
    summarize_selection, N_TEST,
)


def test_official_fixed_archive_train_and_test_are_kept_distinct(tmp_path):
    train=np.zeros((1422,3,182),dtype=np.float32)
    test=np.zeros((1436,3,182),dtype=np.float32)
    ytrain=np.arange(1422)%20
    ytest=np.arange(1436)%20
    path=tmp_path/"character.npz"
    record=prepare_character_trajectories(train,ytrain,test,ytest,path)
    assert record["source_unequal_length"] is True
    with np.load(path) as z:
        assert z["x_train"].shape==(1422,182,3)
        assert z["x_test"].shape==(1436,182,3)
        fit,risk,gate=stratified_threeway(z["y_train"])
        assert len(fit)+len(risk)+len(gate)==1422
        assert not set(fit)&set(risk)
        assert not set(fit)&set(gate)
        assert not set(risk)&set(gate)
        assert sorted(fit.tolist()+risk.tolist()+gate.tolist())==list(range(1422))


def test_archive_wrong_shape_and_class_count_rejected(tmp_path):
    train=np.zeros((1422,3,182))
    test=np.zeros((1436,3,182))
    ytr=np.arange(1422)%20
    yte=np.arange(1436)%20
    with pytest.raises(ValueError,match="1422/1436"):
        prepare_character_trajectories(
            train[:-1],ytr[:-1],test,yte,tmp_path/"invalid.npz",
        )
    with pytest.raises(ValueError,match="twenty"):
        prepare_character_trajectories(
            train,np.zeros(len(train)),test,yte,tmp_path/"invalid.npz",
        )


def test_k2_certificate_and_d3_d4_audit_have_separate_real_model_input_cost():
    x=np.zeros((182,3))
    candidates=np.stack([
        np.full_like(x,.8),np.full_like(x,1.),
        np.full_like(x,1.2),np.full_like(x,1.4),
        np.full_like(x,1.6),
    ])
    from beaconxai.contrastive import fit_class_profiles
    tr=np.r_[np.zeros((5,182,3)),candidates]
    yy=np.r_[np.zeros(5,dtype=int),np.ones(5,dtype=int)]
    profiles=fit_class_profiles(tr,yy)
    groups=make_components((182,3),10,"all_channels")
    from types import SimpleNamespace
    bank=SimpleNamespace(
        donors={1:candidates},
        selected_train_indices={1:np.arange(5)},
    )
    screened={1:np.ones(5,dtype=int)}
    counter={"examples":0}
    def model(batch):
        counter["examples"]+=len(batch)
        mean=np.mean(batch,axis=(1,2))
        return np.column_stack([np.zeros(len(batch)),mean-.07])
    result=explain_one(
        x,np.array([0.,-.07]),bank,screened,groups,profiles,model,counter,
        sample=0,original_index=100,stage="test",
    )
    assert result["online_queries"]<=Q
    assert result["online_queries"]+result["posthoc_queries"]==counter["examples"]
    if result["online_certified"]:
        assert result["posthoc_queries"]==2
        assert result["features"].shape==(9,)
        assert result["fraction_max"]<=.35
        assert result["selected"]
    else:
        assert result["posthoc_queries"]==0
        assert result["features"] is None


def test_unseen_donors_do_not_affect_mask_selection_or_online_q():
    x=np.zeros((182,3))
    groups=make_components((182,3),10,"all_channels")
    profiles={
        0:(np.zeros_like(x),np.ones_like(x)),
        1:(np.ones_like(x),np.ones_like(x)),
    }
    from types import SimpleNamespace
    results=[]
    for out_of_search in [(.9,1.1),(.01,.02)]:
        donors=np.stack([
            np.ones_like(x),1.05*np.ones_like(x),
            *(np.full_like(x,v) for v in out_of_search),
        ])
        bank=SimpleNamespace(
            donors={1:donors},
            selected_train_indices={1:np.arange(4)},
        )
        screened={1:np.ones(4,dtype=int)}
        counter={"examples":0}
        def model(batch):
            counter["examples"]+=len(batch)
            return np.column_stack([
                np.zeros(len(batch)),
                np.mean(batch,axis=(1,2))-.06
            ])
        r=explain_one(
            x,np.array([0.,-.06]),bank,screened,groups,profiles,model,counter,
            sample=0,original_index=0,stage="test",
        )
        assert r["online_queries"]+r["posthoc_queries"]==counter["examples"]
        results.append(r)
    assert results[0]["selected"]==results[1]["selected"]
    assert results[0]["online_queries"]==results[1]["online_queries"]


def test_all_failures_in_denominator_and_risk_features_only_for_certificates():
    from scripts.run_uea_charactertrajectories_transfer_gate import empty_record
    rows=[empty_record(i,i,0,1,"test","none") for i in range(N_TEST)]
    features,labels,support=accepted_features(rows)
    assert features.shape==(0,9) and labels.shape==(0,)
    for r in rows:
        r["base_issue"]=r["risk_issue"]=r["support_issue"]=0
    stats=summarize_selection(rows,"risk_issue")
    assert stats["N"]==320 and stats["issued"]==0
    assert stats["abstentions"]==320
    assert stats["failure_95pct_CP_interval"]==[None,None]
