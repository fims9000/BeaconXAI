"""Full-budget model-sensitivity reference for BEACON validation.

This reference is NOT causal ground truth, human annotation or proof of
semantic explanation correctness. It uses the same model and intervention
operator, testing how Q0/Q16 rankings approximate a full Q=M scan.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def moving_margin_effects(baseline: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """Abs change in the top-two model score margin after each intervention."""
    base = np.asarray(baseline, dtype=float)
    outputs = np.asarray(perturbed, dtype=float)
    if (base.ndim != 1 or base.size < 2 or outputs.ndim != 2
        or outputs.shape[1] != base.size or outputs.shape[0] < 1
        or not np.isfinite(base).all() or not np.isfinite(outputs).all()):
        raise ValueError("invalid scores or perturbation shape")
    sb = np.sort(base)
    so = np.sort(outputs, axis=1)
    return np.abs(float(sb[-1] - sb[-2]) - (so[:, -1] - so[:, -2]))


def rank_effects(scores: np.ndarray, ids: list[int]) -> list[int]:
    arr = np.asarray(scores, dtype=float)
    if (arr.ndim != 1 or len(arr) != len(ids) or not ids
        or len(set(ids)) != len(ids) or not np.isfinite(arr).all()):
        raise ValueError("invalid ranked scores or component identities")
    return [ids[k] for k in np.argsort(-arr, kind="stable")]


def reference_metrics(reference_ranking: list[int], trial_ranking: list[int],
                      number_of_components: int, q: int) -> dict:
    """Top-3 overlap and Q-shortlist coverage versus the full sensitivity scan."""
    if (number_of_components < 3 or q < 0 or q > number_of_components
        or len(reference_ranking) != number_of_components
        or len(set(reference_ranking)) != number_of_components
        or len(set(trial_ranking)) != len(trial_ranking)
        or any(k not in reference_ranking for k in trial_ranking)
        or (q and len(trial_ranking) != q)
        or (not q and len(trial_ranking) != number_of_components)):
        raise ValueError("invalid reference or candidate ranks")
    top = set(reference_ranking[:3])
    q3 = set(trial_ranking[:3])
    out = {
        "top3_overlap": len(top & q3)/3,
        "oracle_top1_in_top3": int(reference_ranking[0] in q3),
        "oracle_top3_exact": int(reference_ranking[:3] == trial_ranking[:3]),
    }
    out["top3_in_shortlist"] = len(top & set(trial_ranking))/3 if q else None
    return out


def subject_macro_summary(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Descriptive subject-macro results; no independent-window p-values."""
    if not rows:
        raise ValueError("empty reference evaluation")
    by_method_subject = defaultdict(list)
    for r in rows:
        by_method_subject[(r["method"], int(r["subject"]))].append(r)
    per_subject = []
    for (method, subject), group in sorted(by_method_subject.items()):
        enough = [r for r in group if r["reference_max_shift"] >= .01]
        stable = [r for r in group if r["reference_top3_gap"] > 1e-6]
        q_values = {int(r["extra_model_evaluations"]) for r in group}
        if len(q_values) != 1:
            raise ValueError("inconsistent Q inside method/subject")
        per_subject.append({
            "method": method, "subject": subject,
            "n_windows": len({r["sample"] for r in group}),
            "extra_model_evaluations": q_values.pop(),
            "top3_overlap": float(np.mean([r["top3_overlap"] for r in group])),
            "oracle_top1_in_top3": float(np.mean([r["oracle_top1_in_top3"] for r in group])),
            "top3_in_shortlist": (float(np.mean([r["top3_in_shortlist"] for r in group]))
                                    if group[0]["top3_in_shortlist"] is not None else None),
            "overlap_on_reference_effect_ge_001": (float(np.mean([r["top3_overlap"] for r in enough]))
                                                    if enough else None),
            "n_reference_effect_ge_001": len(enough),
            "overlap_on_reference_gap_gt_1e6": (float(np.mean([r["top3_overlap"] for r in stable]))
                                                  if stable else None),
            "n_stable_reference_top3": len(stable),
            "mean_reference_max_shift": float(np.mean([r["reference_max_shift"] for r in group])),
        })
    result = []
    for method in sorted({r["method"] for r in per_subject}):
        group = [r for r in per_subject if r["method"] == method]
        result.append({
            "method": method, "n_subjects": len(group),
            "n_windows": sum(r["n_windows"] for r in group),
            "extra_model_evaluations": group[0]["extra_model_evaluations"],
            "subject_macro_top3_overlap": float(np.mean([r["top3_overlap"] for r in group])),
            "subject_macro_oracle_top1_in_top3": float(np.mean([r["oracle_top1_in_top3"] for r in group])),
            "subject_macro_top3_in_shortlist": (
                float(np.mean([r["top3_in_shortlist"] for r in group]))
                if all(r["top3_in_shortlist"] is not None for r in group) else None
            ),
        })
    return result, per_subject
