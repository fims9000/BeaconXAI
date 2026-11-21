"""Explicit memory *projections* for a raw-window MLP BEACON audit.

These are byte counts for model tensor arrays and the explicitly modelled
algorithmic buffers. They are NOT MCU peak RAM, compiler map output,
quantized-model verification, flash occupancy, timing or energy measurements.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def mlp_resource_projection(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    *,
    sample_shape: tuple[int, int],
    n_classes: int,
    n_components: int,
    query_budget: int,
    stored_profiles: int = 2,
) -> dict:
    """Separate persistent flash candidates from streaming and batching RAM.

    stored_profiles=2 means class means for candidate selection and medoids
    for rival-class replacement; this does NOT include a full training bank.
    """
    if not weights or len(weights) != len(biases):
        raise ValueError("Expected nonempty matching MLP tensor lists")
    if (len(sample_shape) != 2 or any(int(n) <= 0 for n in sample_shape)
        or n_classes < 2 or n_components < 3 or query_budget < 1
        or query_budget > n_components or stored_profiles < 1):
        raise ValueError("Invalid sample shape, classes, Q or profile count")
    T,C=map(int,sample_shape)
    prior_width=T*C
    prev=prior_width
    peak_activation=0
    weight_items=0
    bias_items=0
    actual_bytes=0
    for w,b in zip(weights,biases):
        w=np.asarray(w)
        b=np.asarray(b)
        if (w.ndim != 2 or b.ndim != 1 or w.shape != (prev,len(b))
            or not np.issubdtype(w.dtype,np.floating)
            or not np.issubdtype(b.dtype,np.floating)
            or not np.isfinite(w).all() or not np.isfinite(b).all()):
            raise ValueError("Invalid MLP geometry or numeric arrays")
        weight_items+=w.size
        bias_items+=b.size
        actual_bytes+=w.nbytes+b.nbytes
        peak_activation=max(peak_activation,b.size)
        prev=len(b)
    if prev not in (1,n_classes):
        raise ValueError("Output layer does not match class count")
    sample_f32_bytes=prior_width*4
    scores_f32_bytes=n_classes*4
    selected_uint16_bytes=query_budget*2 if n_components <= 65535 else query_budget*4
    input_f32_bytes=sample_f32_bytes
    edited_f32_bytes=sample_f32_bytes
    prior_f32_bytes=n_components*4
    model_output_and_hidden_f32_bytes=(n_classes+peak_activation)*4
    streaming_explicit_bytes=(
        input_f32_bytes+edited_f32_bytes+prior_f32_bytes
        +selected_uint16_bytes+model_output_and_hidden_f32_bytes
    )
    # Actual runtime activation/liveness can be higher; this is a *floor*
    # for explicit buffers. Batched Python creates Q full perturbed copies.
    batched_explicit_bytes=(
        input_f32_bytes+query_budget*sample_f32_bytes
        +prior_f32_bytes+selected_uint16_bytes
        +query_budget*scores_f32_bytes+model_output_and_hidden_f32_bytes
    )
    profile_f32_bytes=n_classes*T*C*4*stored_profiles
    standardizer_f32_bytes=2*C*4
    f32_param_bytes=4*(weight_items+bias_items)
    int8_weights_f32_bias_bytes=weight_items+4*bias_items
    out={
        "sample_time_steps":T,
        "sample_channels":C,
        "n_classes":n_classes,
        "n_components":n_components,
        "additional_model_examples":query_budget,
        "layers":len(weights),
        "weights_count":weight_items,
        "bias_count":bias_items,
        "sklearn_in_memory_parameter_array_bytes":actual_bytes,
        "model_parameters_projected_float32_bytes":f32_param_bytes,
        "model_parameters_projected_int8_weight_fp32_bias_bytes":int8_weights_f32_bias_bytes,
        "int8_projection_has_unaccounted_scales_arenas_and_accuracy":True,
        "training_profiles_float32_bytes":profile_f32_bytes,
        "standardizer_float32_bytes":standardizer_f32_bytes,
        "total_projected_f32_persistent_tensors_bytes":(
            f32_param_bytes+profile_f32_bytes+standardizer_f32_bytes),
        "input_float32_bytes":input_f32_bytes,
        "single_edited_window_float32_bytes":edited_f32_bytes,
        "q_batched_edited_windows_float32_bytes":query_budget*edited_f32_bytes,
        "algorithmic_streaming_explicit_buffer_floor_bytes":streaming_explicit_bytes,
        "algorithmic_q_batch_explicit_buffer_floor_bytes":batched_explicit_bytes,
        "not_included_in_peak_ram":["ML framework/tensor arena","call stack","DMA",
            "allocator and Python objects","intermediate layer liveness",
            "additional temporary arrays","full donor bank"],
        "not_a_compiled_mcu_resource_measurement":True,
    }
    return out
