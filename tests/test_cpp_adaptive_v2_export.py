"""Original adaptive_v2 FP32 profile export and binary parity receipt tests."""
import struct
import numpy as np
import pytest

from beaconxai.cpp_adaptive_v2_export import (
    render_class_means_header, save_baseline_q16_fixture
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
