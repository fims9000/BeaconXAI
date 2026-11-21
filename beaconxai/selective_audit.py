"""Executable selective BEACON audit; no offline oracle or test label input.

The only inputs deciding the budget are the original model scores and the
cheap component prior built from TRAIN data. When the gate returns Q0, no
perturbed example is sent to the classifier. One batched Q invocation still
counts Q individual model evaluations, not one evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .contrastive import neutralize_interp
from .query_gate import QueryGate, cheap_gate_features

@dataclass(frozen=True)
class SelectiveAuditResult:
    ranking: list[int]
    selected_components: list[int]
    query_features: dict[str, float]
    predicted_class: int
    extra_model_evaluations: int
    baseline_model_evaluations: int
    audit_wall_ms: float


def run_selective_audit(x: np.ndarray, components, cheap_scores: np.ndarray,
                        predict_many, gate: QueryGate,
                        baseline_scores: np.ndarray | None = None) -> SelectiveAuditResult:
    """Run Q0 or Q=gate.budget, preserving original moving-margin semantics."""
    x = np.asarray(x)
    cheap = np.asarray(cheap_scores, dtype=float)
    n = len(components)
    if x.ndim != 2 or not np.isfinite(x).all() or n < 2 or cheap.shape != (n,):
        raise ValueError('invalid sample, components or prior length')
    if not np.isfinite(cheap).all() or not 1 <= gate.budget <= n:
        raise ValueError('invalid prior or model query budget')
    if baseline_scores is None:
        base_out = np.asarray(predict_many(x[None]), dtype=float)
        if base_out.ndim != 2 or base_out.shape[0] != 1:
            raise ValueError('baseline function must return [1,C]')
        baseline = base_out[0]
        baseline_calls = 1
    else:
        baseline = np.asarray(baseline_scores, dtype=float)
        baseline_calls = 0
    if baseline.ndim != 1 or len(baseline) < 2 or not np.isfinite(baseline).all():
        raise ValueError('invalid cached baseline score vector')
    features = cheap_gate_features(baseline, cheap)
    if not gate.should_query(features):
        return SelectiveAuditResult(
            ranking=np.argsort(-cheap, kind='stable').astype(int).tolist(),
            selected_components=[], query_features=features,
            predicted_class=int(np.argmax(baseline)),
            extra_model_evaluations=0,
            baseline_model_evaluations=baseline_calls, audit_wall_ms=0.0)
    q = gate.budget
    selected = (np.argpartition(-cheap, kth=q-1)[:q] if q < n
                else np.arange(n)).astype(int).tolist()
    perturbations = np.stack([neutralize_interp(x, components[i]) for i in selected])
    start = time.perf_counter()
    outcomes = np.asarray(predict_many(perturbations), dtype=float)
    duration_ms = (time.perf_counter() - start) * 1000.0
    if outcomes.shape != (q, len(baseline)) or not np.isfinite(outcomes).all():
        raise ValueError('perturbation model response must be finite [Q,C]')
    base_sorted = np.sort(baseline)
    new_sorted = np.sort(outcomes, axis=1)
    margin_before = float(base_sorted[-1] - base_sorted[-2])
    margin_after = new_sorted[:, -1] - new_sorted[:, -2]
    change = np.abs(margin_before - margin_after)
    measured = np.argsort(-change, kind='stable')
    rank = [selected[i] for i in measured]
    return SelectiveAuditResult(
        ranking=rank, selected_components=selected, query_features=features,
        predicted_class=int(np.argmax(baseline)),
        extra_model_evaluations=q, baseline_model_evaluations=baseline_calls,
        audit_wall_ms=duration_ms)
