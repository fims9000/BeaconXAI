"""Export a one-hidden-layer raw-window sklearn MLP to portable FP32 C++.

Only compiler-ready tensor arrays, shape metadata and training normalization
are produced. This is not an int8 export, board deployment or assurance of
numeric parity; the generated files must pass independent C++ parity tests.
"""
from __future__ import annotations

import struct
from pathlib import Path
import numpy as np


def _c_float(value: float) -> str:
    v=float(np.float32(value))
    if not np.isfinite(v):
        raise ValueError("MLP has non-finite parameters")
    s=format(v,".9g")
    if "." not in s and "e" not in s and "E" not in s:
        s+=".0"
    return s+"f"


def _tensor_declaration(name: str, array: np.ndarray) -> str:
    arr=np.asarray(array,dtype=np.float32).ravel(order="C")
    chunks=[
        "  "+", ".join(_c_float(x) for x in arr[i:i+8])
        for i in range(0,len(arr),8)
    ]
    return ("static const float "+name+"["+str(len(arr))+"] = {\n"+
            ",\n".join(chunks)+"\n};\n")


def render_raw_mlp_header(
    weights: list[np.ndarray], biases: list[np.ndarray],
    mu: np.ndarray, sigma: np.ndarray,
    *, sample_shape: tuple[int,int]
) -> str:
    """One hidden ReLU layer, one multi-class softmax output."""
    if len(weights)!=2 or len(biases)!=2 or len(sample_shape)!=2:
        raise ValueError("Expected one hidden layer, two weights and biases")
    time_steps,channels=map(int,sample_shape)
    if min(time_steps,channels)<1:
        raise ValueError("invalid sample geometry")
    w1,w2=[np.asarray(w) for w in weights]
    b1,b2=[np.asarray(b) for b in biases]
    if (w1.ndim!=2 or w2.ndim!=2 or b1.ndim!=1 or b2.ndim!=1
        or w1.shape!=(time_steps*channels,len(b1))
        or w2.shape!=(len(b1),len(b2)) or len(b2)<2):
        raise ValueError("invalid MLP layer dimensions")
    channel_mu=np.asarray(mu,dtype=float).reshape(-1)
    channel_sigma=np.asarray(sigma,dtype=float).reshape(-1)
    if (len(channel_mu)!=channels or len(channel_sigma)!=channels
        or np.any(channel_sigma<=0)
        or not np.isfinite(channel_mu).all()
        or not np.isfinite(channel_sigma).all()):
        raise ValueError("invalid training-only channel standardization")
    text=[
        "#ifndef BEACON_EXPORTED_RAW_MLP_HPP",
        "#define BEACON_EXPORTED_RAW_MLP_HPP",
        "#include <cstddef>",
        "// Generated float32 parameters, no post-export quantization/calibration.",
        "// Stored input is raw [T,C]; normalize with kMu and kSigma before forward.",
        "namespace beacon_exported_mlp {",
        f"constexpr size_t kTimeSteps = {time_steps};",
        f"constexpr size_t kChannels = {channels};",
        f"constexpr size_t kInputDim = {time_steps*channels};",
        f"constexpr size_t kHiddenDim = {len(b1)};",
        f"constexpr size_t kOutputDim = {len(b2)};",
        _tensor_declaration("kW1",w1),
        _tensor_declaration("kB1",b1),
        _tensor_declaration("kW2",w2),
        _tensor_declaration("kB2",b2),
        _tensor_declaration("kMu",channel_mu),
        _tensor_declaration("kSigma",channel_sigma),
        "} // namespace beacon_exported_mlp",
        "#endif // BEACON_EXPORTED_RAW_MLP_HPP",
    ]
    return "\n".join(text)+"\n"


def save_parity_fixture(
    path: str | Path,
    standardized_windows: np.ndarray,
    expected_top2_log_margins: np.ndarray
) -> None:
    """Little-endian binary (uint32 N,D, N*[D float32 + margin float32])."""
    x=np.asarray(standardized_windows,dtype=np.float32)
    margins=np.asarray(expected_top2_log_margins,dtype=np.float32)
    if (x.ndim!=3 or margins.shape!=(len(x),) or not len(x)
        or not np.isfinite(x).all() or not np.isfinite(margins).all()):
        raise ValueError("invalid finite parity windows/margins")
    n,d=len(x),int(np.prod(x.shape[1:]))
    out=Path(path)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("wb") as fp:
        fp.write(struct.pack("<II",n,d))
        for row,expected in zip(x.reshape(n,d),margins):
            fp.write(row.astype("<f4",copy=False).tobytes())
            fp.write(struct.pack("<f",float(expected)))
