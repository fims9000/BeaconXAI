"""Offline diagnostics for synthetic patch localization, not an explanation model.

The diagnostics use clean and patched classifier predictions plus the known
artificial patch location. NEVER use them in candidate selection, gate fitting,
or model inference for the held-out sample. They diagnose whether the benchmark
is asking an explainer to find a patch that affected the classifier at all.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def _probs(scores: np.ndarray) -> np.ndarray:
    # Works for logits, log-probabilities and any finite classifier scores.
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def patch_effects(
    clean: np.ndarray,
    patched: np.ndarray,
    clean_scores: np.ndarray,
    patched_scores: np.ndarray,
    component_ids: np.ndarray,
    components: Sequence,
) -> list[dict]:
    """One record per patched window; one clean-model call per window.

    TV is the total variation distance between predicted class distributions,
    from 0 to 1. Thresholds 0.05 and 0.10 are exploratory descriptors,
    NOT claims that an explanation is clinically/causally meaningful.
    """
    a = np.asarray(clean, dtype=float)
    b = np.asarray(patched, dtype=float)
    z0 = np.asarray(clean_scores, dtype=float)
    z1 = np.asarray(patched_scores, dtype=float)
    gold = np.asarray(component_ids, dtype=int)
    if (a.ndim != 3 or a.shape != b.shape or len(a) < 1
        or z0.ndim != 2 or z0.shape != z1.shape or z0.shape[0] != len(a)
        or z0.shape[1] < 2 or gold.shape != (len(a),)
        or not components or np.any((gold < 0) | (gold >= len(components)))
        or not all(np.isfinite(v).all() for v in (a, b, z0, z1))):
        raise ValueError("invalid clean/patched signals, model scores, or patch indices")
    p0, p1 = _probs(z0), _probs(z1)
    tv = .5 * np.abs(p1 - p0).sum(axis=1)
    before, after = np.argmax(p0, axis=1), np.argmax(p1, axis=1)
    records: list[dict] = []
    for i, k in enumerate(gold):
        c = components[int(k)]
        delta = b[i, c.t0:c.t1, c.c0:c.c1] - a[i, c.t0:c.t1, c.c0:c.c1]
        if delta.size == 0:
            raise ValueError("empty patch component")
        records.append({
            "sample": int(i), "true_component": int(k),
            "true_group": int(c.group), "true_time_bin": int(c.time_bin),
            "reference_predicted_class": int(before[i]),
            "injected_predicted_class": int(after[i]),
            "patch_changed_predicted_class": int(before[i] != after[i]),
            "patch_model_tv": float(tv[i]),
            "patch_model_tv_ge_005": int(tv[i] >= .05),
            "patch_model_tv_ge_010": int(tv[i] >= .10),
            "patch_rms": float(np.sqrt(np.mean(delta**2))),
        })
    return records


def summarize_patch_effects(effect_rows: Sequence[dict], subjects: Sequence[int]) -> list[dict]:
    """Aggregate one row per sample; no repeat-based pseudo-replication."""
    if len(effect_rows) != len(subjects) or not effect_rows:
        raise ValueError("invalid effect rows or subject vector")
    groups = defaultdict(list)
    for row, subject in zip(effect_rows, subjects):
        groups[int(subject)].append(row)
    result = []
    for subject, rows in sorted(groups.items()):
        result.append({
            "subject": subject, "windows": len(rows),
            "prediction_change_rate": float(np.mean([r["patch_changed_predicted_class"] for r in rows])),
            "tv_ge_005_rate": float(np.mean([r["patch_model_tv_ge_005"] for r in rows])),
            "tv_ge_010_rate": float(np.mean([r["patch_model_tv_ge_010"] for r in rows])),
            "tv_mean": float(np.mean([r["patch_model_tv"] for r in rows])),
            "patch_rms_mean": float(np.mean([r["patch_rms"] for r in rows])),
        })
    return result


def summarize_method_effects(rows: Sequence[dict]) -> list[dict]:
    """Descriptive subject-level hit@3 conditional on model impact.

    For methods with direct Q component probes, 'candidate_covered' reflects
    whether the known synthetic patch was in the Q probed components.
    Q0 and coalition/mask methods have no direct shortlist coverage measure.
    """
    if not rows:
        raise ValueError("empty method rows")
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["subject"]))].append(row)
    output = []
    for (method, subject), group in sorted(groups.items()):
        effect = [r for r in group if int(r["patch_model_tv_ge_005"])]
        no_effect = [r for r in group if not int(r["patch_model_tv_ge_005"])]
        changed = [r for r in group if int(r["patch_changed_predicted_class"])]
        cov = [r for r in group if r.get("candidate_covered") not in (None, "")]
        output.append({
            "method": method, "subject": subject,
            "windows": len({int(r["sample"]) for r in group}),
            "repeats": len({int(r["trial"]) for r in group}),
            "hit3": float(np.mean([r["hit3"] for r in group])),
            "hit3_tv_ge_005": float(np.mean([r["hit3"] for r in effect])) if effect else None,
            "n_tv_ge_005_windows": len({int(r["sample"]) for r in effect}),
            "hit3_tv_lt_005": float(np.mean([r["hit3"] for r in no_effect])) if no_effect else None,
            "n_tv_lt_005_windows": len({int(r["sample"]) for r in no_effect}),
            "hit3_prediction_switch": float(np.mean([r["hit3"] for r in changed])) if changed else None,
            "n_prediction_switch_windows": len({int(r["sample"]) for r in changed}),
            "direct_shortlist_coverage": float(np.mean([int(r["candidate_covered"]) for r in cov])) if cov else None,
            "hit3_given_shortlisted": (
                float(np.mean([r["hit3"] for r in cov if int(r["candidate_covered"])]))
                if any(int(r["candidate_covered"]) for r in cov) else None
            ),
        })
    return output
