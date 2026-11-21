"""FP32 C++ exporter: geometry, finite values and binary parity format."""
import struct

import numpy as np
import pytest

from beaconxai.cpp_mlp_export import render_raw_mlp_header,save_parity_fixture


def _small_model():
    w1=np.array([[.2,-.1]]*12,dtype=float)
    b1=np.array([.0,.5])
    w2=np.array([[.3,.4,.6],[.7,-.2,.1]])
    b2=np.array([.0,.0,.1])
    return [w1,w2],[b1,b2]


def test_header_is_fp32_and_contains_consistent_shapes():
    w,b=_small_model()
    text=render_raw_mlp_header(w,b,np.zeros(3),np.ones(3),
                               sample_shape=(4,3))
    assert "constexpr size_t kInputDim = 12;" in text
    assert "constexpr size_t kHiddenDim = 2;" in text
    assert "constexpr size_t kOutputDim = 3;" in text
    assert "kW1[24]" in text
    assert "kB2[3]" in text
    assert "0.2" in text
    assert "f" in text
    assert "static const float" in text


def test_binary_fixture_exact_bytes_and_labels(tmp_path):
    x=np.arange(2*4*3,dtype=np.float32).reshape(2,4,3)
    y=np.array([.5,1.25])
    fn=tmp_path/"fixtures.bin"
    save_parity_fixture(fn,x,y)
    payload=fn.read_bytes()
    assert struct.unpack("<II",payload[:8])==(2,12)
    assert len(payload)==8+2*(12+1)*4
    assert struct.unpack("<f",payload[8+12*4:8+13*4])[0]==pytest.approx(.5)


@pytest.mark.parametrize("bad",["shape","nonfinite","negative_sigma"])
def test_bad_model_or_training_normalization_rejected(bad):
    w,b=_small_model()
    mu=np.zeros(3);std=np.ones(3)
    if bad=="shape":
        w[0]=w[0][:5]
    elif bad=="nonfinite":
        w[1][0,1]=np.inf
    else:
        std[0]=0.
    with pytest.raises(ValueError):
        render_raw_mlp_header(w,b,mu,std,sample_shape=(4,3))


def test_bad_parity_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_parity_fixture(tmp_path/"out.bin",np.zeros((2,3,9)),
                            np.array([1.,np.nan]))
