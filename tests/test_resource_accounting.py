"""Numerically exact array-byte projections, not physical MCU measurements."""
import numpy as np
import pytest

from beaconxai.resource_accounting import mlp_resource_projection


def _mlp(dtype=np.float64):
    return [
        np.zeros((128*9,32),dtype=dtype),np.zeros((32,6),dtype=dtype)
    ],[
        np.zeros(32,dtype=dtype),np.zeros(6,dtype=dtype)
    ]


def test_uci_har_mlp_fp32_and_unverified_int8_projection():
    weights,biases=_mlp()
    r=mlp_resource_projection(weights,biases,sample_shape=(128,9),
                               n_classes=6,n_components=36,
                               query_budget=16,stored_profiles=2)
    assert r["weights_count"]==37056
    assert r["bias_count"]==38
    assert r["sklearn_in_memory_parameter_array_bytes"]==296752
    assert r["model_parameters_projected_float32_bytes"]==148376
    assert r["model_parameters_projected_int8_weight_fp32_bias_bytes"]==37208
    assert r["training_profiles_float32_bytes"]==55296
    assert r["standardizer_float32_bytes"]==72
    assert r["total_projected_f32_persistent_tensors_bytes"]==203744
    assert r["input_float32_bytes"]==4608
    assert r["q_batched_edited_windows_float32_bytes"]==73728
    assert r["algorithmic_streaming_explicit_buffer_floor_bytes"]==9544
    assert r["algorithmic_q_batch_explicit_buffer_floor_bytes"]==79048
    assert r["not_a_compiled_mcu_resource_measurement"] is True


def test_batch_size_affects_scratch_floor_not_model_size():
    weights,biases=_mlp(dtype=np.float32)
    low=mlp_resource_projection(weights,biases,sample_shape=(128,9),
                                n_classes=6,n_components=36,query_budget=8)
    high=mlp_resource_projection(weights,biases,sample_shape=(128,9),
                                 n_classes=6,n_components=36,query_budget=32)
    assert low["model_parameters_projected_float32_bytes"]==high["model_parameters_projected_float32_bytes"]
    assert high["algorithmic_q_batch_explicit_buffer_floor_bytes"] > low["algorithmic_q_batch_explicit_buffer_floor_bytes"]
    assert high["algorithmic_streaming_explicit_buffer_floor_bytes"] >= low["algorithmic_streaming_explicit_buffer_floor_bytes"]


@pytest.mark.parametrize("bad", [
    {"query_budget":0},
    {"query_budget":37},
    {"sample_shape":(0,9)},
    {"n_classes":1}
])
def test_reject_invalid_configuration(bad):
    weights,biases=_mlp()
    args=dict(sample_shape=(128,9),n_classes=6,n_components=36,query_budget=16)
    args.update(bad)
    with pytest.raises(ValueError):
        mlp_resource_projection(weights,biases,**args)


def test_reject_incorrect_output_head_and_nan_weights():
    weights,biases=_mlp()
    with pytest.raises(ValueError,match="Output layer"):
        mlp_resource_projection(weights,biases,sample_shape=(128,9),
                                n_classes=5,n_components=36,query_budget=16)
    weights[0][0,0]=np.nan
    with pytest.raises(ValueError,match="numeric"):
        mlp_resource_projection(weights,biases,sample_shape=(128,9),
                                n_classes=6,n_components=36,query_budget=16)
