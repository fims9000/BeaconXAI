"""Export original adaptive_v2 class-mean profiles and independent parity vectors.

C++ recomputes the original four-term z-score prior, chooses the Q set,
neutralizes interpolation patches and queries the *trained* raw MLP. The
reference is made by original Python _component_cheap_scores, best_q,
evaluate_selected. NumPy argpartition ordering of selected elements is not
specified: compare selected SET, and rank separately after model probes.
"""
from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from beaconxai.cpp_mlp_export import _tensor_declaration, _c_float


def render_class_means_header(
    class_means: np.ndarray, sample_shape: tuple[int,int]
) -> str:
    means=np.asarray(class_means,dtype=np.float64)
    if (means.ndim!=3 or means.shape[1:]!=sample_shape
        or means.shape[0]<2 or not np.isfinite(means).all()):
        raise ValueError("expected finite class means [classes,time,channels]")
    lines=[
        "#ifndef BEACON_EXPORTED_CLASS_MEANS_HPP",
        "#define BEACON_EXPORTED_CLASS_MEANS_HPP",
        "#include <cstddef>",
        "// Training-only class means; exported as FP32 for embedded testing.",
        "namespace beacon_exported_profiles {",
        "constexpr size_t kClasses = "+str(means.shape[0])+";",
        "constexpr size_t kWindow = "+str(means.shape[1]*means.shape[2])+";",
        _tensor_declaration("kClassMeans",means.astype(np.float32)),
        "} // namespace beacon_exported_profiles",
        "#endif // BEACON_EXPORTED_CLASS_MEANS_HPP",
    ]
    return "\n".join(lines)+"\n"


def save_baseline_q16_fixture(
    path: str|Path,
    raw_windows: np.ndarray,
    class_ids: np.ndarray,
    cheap_priors: np.ndarray,
    selected_sorted: np.ndarray,
    ranked: np.ndarray,
    cutoff_gaps: np.ndarray,
    effect_separations: np.ndarray,
) -> None:
    """Binary format (little-endian): header uint32 N,D,M,Q, then each:

    float32[D] RAW input; uint32 predicted class; float64 cutoff gap and
    min pairwise model-effect difference; float64[M] prior; uint16[Q]
    selected IDs in ascending component-ID order; uint16[Q] ranked IDs.
    """
    x=np.asarray(raw_windows,dtype=np.float32)
    classes=np.asarray(class_ids,dtype=int)
    prior=np.asarray(cheap_priors,dtype=np.float64)
    chosen=np.asarray(selected_sorted,dtype=int)
    rank=np.asarray(ranked,dtype=int)
    cutoff=np.asarray(cutoff_gaps,dtype=np.float64)
    effect_sep=np.asarray(effect_separations,dtype=np.float64)
    if x.ndim!=3 or not len(x):
        raise ValueError("expected nonempty raw windows [n,t,c]")
    n,d=len(x),int(np.prod(x.shape[1:]))
    if (classes.shape!=(n,) or prior.ndim!=2 or prior.shape[0]!=n or
        chosen.ndim!=2 or chosen.shape[0]!=n or rank.shape!=chosen.shape or
        cutoff.shape!=(n,) or effect_sep.shape!=(n,) or
        not (3<=chosen.shape[1]<prior.shape[1]<=65535) or
        np.any(classes<0) or
        np.any(chosen<0) or np.any(chosen>=prior.shape[1]) or
        np.any(rank<0) or np.any(rank>=prior.shape[1]) or
        not np.isfinite(x).all() or not np.isfinite(prior).all() or
        not np.isfinite(cutoff).all() or not np.isfinite(effect_sep).all()):
        raise ValueError("malformed baseline Q16 parity inputs")
    q=chosen.shape[1];m=prior.shape[1]
    for ids,r in zip(chosen,rank):
        if len(np.unique(ids))!=q or len(np.unique(r))!=q or set(ids)!=set(r):
            raise ValueError("selected and ranked IDs must be unique and identical sets")
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as fp:
        fp.write(struct.pack("<4I",n,d,m,q))
        for raw,cls,values,ids,ordering,gap,separation in zip(
            x.reshape(n,d),classes,prior,chosen,rank,cutoff,effect_sep
        ):
            fp.write(raw.astype("<f4",copy=False).tobytes())
            fp.write(struct.pack("<Idd",int(cls),float(gap),float(separation)))
            fp.write(values.astype("<f8",copy=False).tobytes())
            fp.write(ids.astype("<u2",copy=False).tobytes())
            fp.write(ordering.astype("<u2",copy=False).tobytes())


def render_board_reference_window_header(
    raw_window: np.ndarray,
    predicted_class: int,
    selected_ids: np.ndarray,
    ranked_ids: np.ndarray,
) -> str:
    """One frozen UCI HAR test window for future *on-device* self-test.

    Do not use the reference identity to choose BEACON probes. The header is
    generated only after independent model fitting and Python evaluation.
    """
    raw=np.asarray(raw_window,dtype=np.float32)
    selected=np.asarray(selected_ids,dtype=int)
    ranked=np.asarray(ranked_ids,dtype=int)
    if (raw.ndim!=2 or not np.isfinite(raw).all()
        or predicted_class<0 or selected.ndim!=1 or len(selected)!=16
        or ranked.shape!=selected.shape or
        len(set(selected.tolist()))!=16 or set(selected)!=set(ranked)
        or np.any(selected<0) or np.any(ranked<0)):
        raise ValueError("invalid frozen UCI HAR board reference")
    lines=[
        "#ifndef BEACON_EXPORTED_BOARD_REFERENCE_HPP",
        "#define BEACON_EXPORTED_BOARD_REFERENCE_HPP",
        "#include <cstdint>",
        "// Frozen external held-out window: for board self-check only.",
        "namespace beacon_reference {",
        "constexpr unsigned kExpectedClass = "+str(int(predicted_class))+";",
        _tensor_declaration("kRawInput",raw),
        "static const uint16_t kExpectedSelected[16] = {"+
            ",".join(str(int(x)) for x in sorted(selected))+"};",
        "static const uint16_t kExpectedRanked[16] = {"+
            ",".join(str(int(x)) for x in ranked)+"};",
        "} // namespace beacon_reference",
        "#endif // BEACON_EXPORTED_BOARD_REFERENCE_HPP",
    ]
    return "\n".join(lines)+"\n"


def render_staged_upgrade_policy_header(train_threshold: float) -> str:
    """Compile-time threshold fitted exclusively on inner training subjects."""
    if not np.isfinite(train_threshold) or train_threshold<0:
        raise ValueError("invalid trained Q16 relative gap threshold")
    return (
        "#ifndef BEACON_EXPORTED_STAGED_POLICY_HPP\n"
        "#define BEACON_EXPORTED_STAGED_POLICY_HPP\n"
        "// Training-only fixed cutoff, not causal confidence calibration.\n"
        "namespace beacon_exported_staged {\n"
        "constexpr float kQ16GapThreshold = "+_c_float(train_threshold)+";\n"
        "}\n"
        "#endif\n"
    )


def save_staged_upgrade_fixture(
    path: str | Path, raw_windows: np.ndarray,
    predicted_classes: np.ndarray, budgets: np.ndarray,
    q16_observed_gaps: np.ndarray,
    min_final_effect_separations: np.ndarray,
    selected_id_lists: list[list[int]],
    ranked_id_lists: list[list[int]],
) -> None:
    """Little endian uint32 N,D,M,Qmax then N records:

    float32[D] RAW sample; uint32 predicted source; uint32 true online Q;
    float64 observed gap16; float64 min measured pairwise-effect separation;
    uint16[32] selected IDs (ascending, unused=65535);
    uint16[32] ranked IDs (unused=65535).
    """
    x=np.asarray(raw_windows,dtype=np.float32)
    cls=np.asarray(predicted_classes,dtype=int)
    q=np.asarray(budgets,dtype=int)
    gap=np.asarray(q16_observed_gaps,dtype=float)
    separation=np.asarray(min_final_effect_separations,dtype=float)
    if (x.ndim!=3 or not len(x) or cls.shape!=(len(x),)
        or q.shape!=(len(x),) or gap.shape!=(len(x),)
        or separation.shape!=(len(x),) or
        not np.isfinite(x).all() or not np.isfinite(gap).all()
        or not np.isfinite(separation).all()
        or np.any(cls<0) or np.any((q!=16)&(q!=32))
        or len(selected_id_lists)!=len(x) or len(ranked_id_lists)!=len(x)):
        raise ValueError("invalid staged budget parity fixture")
    # Explicit check, not a chained bitwise comparison accidentally
    if any(int(z) not in (16,32) for z in q):
        raise ValueError("Q must be 16 or 32")
    n,d=len(x),int(np.prod(x.shape[1:]))
    file=Path(path)
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open("wb") as fp:
        fp.write(struct.pack("<4I",n,d,36,32))
        for raw,source,budget,g,sep,chosen,rank in zip(
            x.reshape(n,d),cls,q,gap,separation,selected_id_lists,ranked_id_lists):
            if (len(chosen)!=budget or len(rank)!=budget
                or len(set(chosen))!=budget or len(set(rank))!=budget
                or set(chosen)!=set(rank)
                or any(k<0 or k>=36 for k in list(chosen)+list(rank))):
                raise ValueError("invalid staged selected/ranked component IDs")
            selected=sorted(chosen)+[65535]*(32-int(budget))
            ordering=list(rank)+[65535]*(32-int(budget))
            fp.write(raw.astype("<f4",copy=False).tobytes())
            fp.write(struct.pack("<IIdd",int(source),int(budget),float(g),float(sep)))
            fp.write(np.asarray(selected,dtype="<u2").tobytes())
            fp.write(np.asarray(ordering,dtype="<u2").tobytes())
