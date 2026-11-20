"""Original adaptive_v2 FP32 profile export and binary parity receipt tests."""
import struct
import numpy as np
import pytest

from beaconxai.cpp_adaptive_v2_export import (
    render_class_means_header, save_baseline_q16_fixture,
    render_staged_upgrade_policy_header, save_staged_upgrade_fixture
)


def _valid_inputs(n=2,m=36,q=16):
    x=np.zeros((n,128,9),dtype=np.float32)
    cls=np.zeros(n,dtype=int)
    prior=np.tile(np.arange(m,dtype=float),(n,1))
    selected=np.tile(np.arange(q),(n,1))
    ranking=np.tile(np.arange(q-1,-1,-1),(n,1))
    gap=np.ones(n)
    separation=np.ones(n)*0.1
    return x,cls,prior,selected,ranking,gap,separation


def test_profile_header_contains_flat_f32_class_means():
    data=np.arange(3*4*9,dtype=np.float64).reshape(3,4,9)/10
    header=render_class_means_header(data,(4,9))
    assert "constexpr size_t kClasses = 3;" in header
    assert "constexpr size_t kWindow = 36;" in header
    assert "kClassMeans[108]" in header
    assert "namespace beacon_exported_profiles" in header
    assert "float" in header


def test_fixture_preserves_all_original_prior_and_q_ids(tmp_path):
    src=_valid_inputs()
    fn=tmp_path/"baseline.bin"
    save_baseline_q16_fixture(fn,*src)
    payload=fn.read_bytes()
    assert struct.unpack("<4I",payload[:16])==(2,1152,36,16)
    per=1152*4+4+8+8+36*8+16*2+16*2
    assert len(payload)==16+2*per
    pos=16+1152*4
    assert struct.unpack_from("<I",payload,pos)[0]==0
    assert struct.unpack_from("<d",payload,pos+4)[0]==1
    assert list(struct.unpack_from("<16H",payload,pos+4+16+36*8))==list(range(16))


def test_reject_duplicate_component_and_rank_identity(tmp_path):
    payload=list(_valid_inputs())
    payload[3][0,1]=payload[3][0,0]
    with pytest.raises(ValueError):
        save_baseline_q16_fixture(tmp_path/"bad.bin",*payload)
    payload=list(_valid_inputs())
    payload[4][0,1]=22
    with pytest.raises(ValueError):
        save_baseline_q16_fixture(tmp_path/"bad2.bin",*payload)


def test_reject_nonfinite_or_wrong_geometry(tmp_path):
    payload=list(_valid_inputs())
    payload[2][0,4]=np.nan
    with pytest.raises(ValueError):
        save_baseline_q16_fixture(tmp_path/"bad.bin",*payload)
    with pytest.raises(ValueError):
        render_class_means_header(np.zeros((3,4,8)),(4,9))


def test_staged_threshold_header_and_mixed_query_fixture(tmp_path):
    header=render_staged_upgrade_policy_header(.0138252582)
    assert "kQ16GapThreshold" in header
    assert ".013825" in header
    raw=np.zeros((2,128,9),dtype=np.float32)
    fn=tmp_path/"staged.bin"
    save_staged_upgrade_fixture(
        fn,raw,np.array([1,3]),np.array([16,32]),
        np.array([.01,.02]),np.array([.00001,.1]),
        [list(range(16)),list(range(32))],
        [list(range(15,-1,-1)),list(range(31,-1,-1))])
    content=fn.read_bytes()
    assert struct.unpack_from("<4I",content,0)==(2,1152,36,32)
    sample_size=1152*4+4+4+8+8+32*2+32*2
    assert len(content)==16+2*sample_size
    cls,q=struct.unpack_from("<II",content,16+1152*4)
    assert (cls,q)==(1,16)
    selected=struct.unpack_from("<32H",content,16+1152*4+24)
    assert list(selected[:16])==list(range(16))
    assert all(x==65535 for x in selected[16:])


def test_staged_fixture_rejects_wrong_budget_and_duplicate_id(tmp_path):
    raw=np.zeros((1,24,9),dtype=np.float32)
    with pytest.raises(ValueError,match="invalid staged"):
        save_staged_upgrade_fixture(
            tmp_path/"bad.bin",raw,np.array([0]),np.array([17]),
            np.array([.01]),np.array([.01]),
            [list(range(17))],[list(range(17))])
    with pytest.raises(ValueError,match="invalid staged selected"):
        save_staged_upgrade_fixture(
            tmp_path/"bad2.bin",raw,np.array([0]),np.array([16]),
            np.array([.01]),np.array([.01]),
            [list(range(15))+[14]],[list(range(16))])
    with pytest.raises(ValueError):
        render_staged_upgrade_policy_header(-.1)
