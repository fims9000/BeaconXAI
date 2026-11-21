"""Donor-role isolation, shared Q16, independent D4/D5 verification and audit."""
import numpy as np
import pytest

from beaconxai.contrastive import fit_class_profiles,make_components
from scripts.preprocess_uea_spokenarabic import prepare_spokenarabic,_resample_multichannel
from scripts.run_uea_spokenarabic_k3_transfer import (
    METHODS,Q,assign_five_donor_roles,
    evaluate_method,paired_primary,summarize,no_donor_result,
)


def test_spokenarabic_exact_uea_geometry_and_source_provenance(tmp_path):
    tr=np.zeros((6599,13,93),dtype=np.float32)
    te=np.zeros((2199,13,93),dtype=np.float32)
    ytr=np.arange(6599)%10
    yte=np.arange(2199)%10
    path=tmp_path/"spoken.npz"
    result=prepare_spokenarabic(tr,ytr,te,yte,path)
    assert result["speaker_disjoint_verified"] is False
    with np.load(path) as z:
        assert z["x_train"].shape==(6599,93,13)
        assert z["x_test"].shape==(2199,93,13)
        assert "TRAIN 6599 TEST 2199" in str(z["dataset_protocol"])


def test_preprocessor_rejects_wrong_archive_geometry(tmp_path):
    with pytest.raises(ValueError,match="6599/2199"):
        prepare_spokenarabic(
            np.zeros((3,13,93)),[0,1,2],
            np.zeros((2,13,93)),[0,1],
            tmp_path/"invalid.npz",
        )


def test_five_roles_nearest_two_farthest_third_two_unseen():
    source=np.zeros((93,13))
    bank=np.asarray([np.full_like(source,float(i+1))
                     for i in range(8)])
    labels=np.ones(8,dtype=int)
    ids=np.arange(100,108)
    roles=assign_five_donor_roles(source,bank,labels,1,ids)
    assert roles==(0,1,7,2,3)
    assert len(set(roles))==5
    unavailable=assign_five_donor_roles(
        source,bank,np.array([1,1,1,1,0,0,0,0]),1,ids
    )
    assert unavailable is None


@pytest.mark.parametrize("method",METHODS)
def test_three_donors_must_share_one_q16_and_audit_is_separate(method):
    x=np.zeros((93,13))
    groups=make_components(x.shape,10,"all_channels")
    donors=(np.ones_like(x),.9*np.ones_like(x),.7*np.ones_like(x))
    unseen=(.85*np.ones_like(x),.75*np.ones_like(x))
    profiles={
        0:(np.zeros_like(x),np.ones_like(x)),
        1:(np.ones_like(x),np.ones_like(x)),
    }
    ledger={"examples":0}
    def model(batch):
        ledger["examples"]+=len(batch)
        mean=np.mean(batch,axis=(1,2))
        return np.column_stack([np.zeros(len(batch)),mean-.12])
    result=evaluate_method(
        x,np.array([0.,-.12]),donors,unseen,
        groups,profiles,model,ledger,method,
    )
    assert result["model_queries"]<=Q
    assert result["model_queries"]+result["posthoc_queries"]==ledger["examples"]
    if result["online_verified"]:
        assert result["selected"]
        assert result["changed_fraction_max_online"]<=.35
        assert result["posthoc_queries"]==2
    else:
        assert result["selected"] is None
        assert result["posthoc_queries"]==0
        assert result["transfer_D4_D5"]==0


def test_unseen_donors_never_change_selected_online_mask():
    x=np.zeros((93,13))
    groups=make_components(x.shape,10,"all_channels")
    donors=(np.ones_like(x),.95*np.ones_like(x),.9*np.ones_like(x))
    profiles={
        0:(np.zeros_like(x),np.ones_like(x)),
        1:(np.ones_like(x),np.ones_like(x)),
    }
    def predictor(batch):
        mean=np.mean(batch,axis=(1,2))
        return np.column_stack((np.zeros(len(batch)),mean-.09))
    outputs=[]
    for unseen in [
        (.8*np.ones_like(x),.75*np.ones_like(x)),
        (.01*np.ones_like(x),.02*np.ones_like(x)),
    ]:
        counts={"examples":0}
        def counted(batch):
            counts["examples"]+=len(batch)
            return predictor(batch)
        r=evaluate_method(
            x,np.array([0.,-.09]),donors,unseen,
            groups,profiles,counted,counts,"K3_stress",
        )
        assert r["model_queries"]+r["posthoc_queries"]==counts["examples"]
        outputs.append(r)
    assert outputs[0]["selected"]==outputs[1]["selected"]
    assert outputs[0]["model_queries"]==outputs[1]["model_queries"]
    assert outputs[0]["posthoc_queries"]==outputs[1]["posthoc_queries"]


def test_paired_primary_and_summary_keep_failures_in_320_denominator():
    records=[]
    for i in range(320):
        for method in METHODS:
            transferred=1 if (i<10 if method=="K3_stress" else i<5) else 0
            records.append({
                "sample":i,"method":method,
                "online_verified":transferred,
                "transfer_D4_D5":transferred,
                "model_queries":7,
                "minimal_on_online_donors":0,
                "posthoc_queries":2*transferred,
                "search_wall_seconds":.01,
            })
    p=paired_primary(records)
    assert p["n_test"]==320
    assert p["K3_fresh"]==10 and p["K2_fresh"]==5
    assert p["K3_only"]==5 and p["K2_only"]==0
    assert p["mcnemar_p_exact_two_sided"]==.0625
    assert len(summarize(records))==2


def test_unavailable_five_donors_is_a_zero_not_an_exclusion():
    result=no_donor_result()
    assert result["transfer_D4_D5"]==0
    assert result["online_verified"]==0
    assert result["model_queries"]==0



def test_variable_length_normalized_interpolation_preserves_endpoints_and_length():
    samples=[
        [np.linspace(float(c),float(c+1),31+c) for c in range(13)],
        [np.linspace(float(c),float(c+1),93) for c in range(13)],
    ]
    x,lengths=_resample_multichannel(samples,2)
    assert x.shape==(2,13,93)
    assert lengths.shape==(2,13)
    assert int(lengths[0,0])==31 and int(lengths[0,12])==43
    assert np.all(lengths[1]==93)
    np.testing.assert_allclose(x[:,:,0],np.arange(13)[None],rtol=0,atol=1e-6)
    np.testing.assert_allclose(x[:,:,-1],(np.arange(13)+1)[None],rtol=0,atol=1e-6)
    np.testing.assert_allclose(x[0],x[1],rtol=0,atol=2e-6)


def test_variable_length_invalid_channel_geometry_is_rejected():
    wrong=[[np.zeros(4)]*12]
    with pytest.raises(ValueError,match="13 MFCC"):
        _resample_multichannel(wrong,1)
    with pytest.raises(ValueError,match=">=2 finite"):
        _resample_multichannel([[np.array([np.nan,1.])]*13],1)
