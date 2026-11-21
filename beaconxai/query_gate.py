"""Train-calibrated Q0/Q16 decision for the original BEACON adaptive_v2.

The gate reads only the cached original model output and zero-query
class-profile scores. Injected-component labels and results of the Q16
audit are used to FIT the gate on inner held-out-by-subject subjects,
never to decide whether to query an outer-test example.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import numpy as np

FEATURES = ("margin", "prior_gap")

@dataclass(frozen=True)
class QueryGate:
    mode: str
    feature: str = ""
    direction: str = "le"
    threshold: float = 0.0
    budget: int = 16

    def __post_init__(self) -> None:
        if self.mode not in ("q0", "q16", "threshold") or self.budget < 1:
            raise ValueError("invalid gate mode or query budget")
        if self.mode == "threshold" and (
            self.feature not in FEATURES or self.direction not in ("le", "ge")
            or not np.isfinite(self.threshold)
        ):
            raise ValueError("invalid threshold gate")

    def should_query(self, features: Mapping[str, float]) -> bool:
        if self.mode == "q0":
            return False
        if self.mode == "q16":
            return True
        z = float(features[self.feature])
        if not np.isfinite(z):
            raise ValueError("nonfinite gate feature")
        return z <= self.threshold if self.direction == "le" else z >= self.threshold

    def to_dict(self) -> dict:
        return asdict(self)

def cheap_gate_features(baseline: np.ndarray, cheap_scores: np.ndarray) -> dict[str, float]:
    """No test label or perturbed-model response enters these features."""
    z = np.asarray(baseline, dtype=float)
    prior = np.asarray(cheap_scores, dtype=float)
    if z.ndim != 1 or z.size < 2 or prior.ndim != 1 or prior.size < 2:
        raise ValueError("expected model vector and >=2 component scores")
    if not np.isfinite(z).all() or not np.isfinite(prior).all():
        raise ValueError("nonfinite model or prior scores")
    p = np.sort(z)
    s = np.sort(prior)
    denominator = max(float(s[-1] - s[0]), 1e-12)
    return {"margin": float(p[-1] - p[-2]),
            "prior_gap": float((s[-1] - s[-2]) / denominator)}

def _validate(rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("empty calibration/evaluation rows")
    for row in rows:
        if "subject" not in row or any(f not in row for f in FEATURES):
            raise ValueError("missing subject or inference-time features")
        for key in ("hit3_q0", "hit3_q16"):
            val = float(row[key])
            if not np.isfinite(val) or not 0 <= val <= 1:
                raise ValueError("invalid hit@3 outcome")
        if not np.isfinite(float(row["margin"])) or not np.isfinite(float(row["prior_gap"])):
            raise ValueError("nonfinite gate feature")

def evaluate_gate(rows: Sequence[dict], gate: QueryGate,
                  cost_per_full_budget: float = 0.05) -> tuple[dict, list[dict]]:
    """Subject-macro quality and SIMULATED deployment query budget.

    Offline evaluation computes Q16 for every example to enable paired
    comparisons; deployed query counts only charge Q16 if the gate uses it.
    This is not a measurement of hardware timing or energy.
    """
    if not np.isfinite(cost_per_full_budget) or cost_per_full_budget < 0:
        raise ValueError("invalid model-call cost penalty")
    _validate(rows)
    per_subject = []
    for subject in sorted({int(row["subject"]) for row in rows}):
        sr = [r for r in rows if int(r["subject"]) == subject]
        query = np.array([gate.should_query(r) for r in sr], dtype=bool)
        q0 = np.array([float(r["hit3_q0"]) for r in sr])
        q16 = np.array([float(r["hit3_q16"]) for r in sr])
        hit = np.where(query, q16, q0)
        per_subject.append({"subject": subject, "n": len(sr),
                            "hit3_gate": float(hit.mean()),
                            "hit3_q0": float(q0.mean()),
                            "hit3_q16": float(q16.mean()),
                            "query_rate": float(query.mean()),
                            "mean_deployed_queries": float(gate.budget * query.mean()),
                            "utility": float(hit.mean() - cost_per_full_budget * query.mean())})
    macro = lambda key: float(np.mean([s[key] for s in per_subject]))
    summary = {"n_samples": len(rows), "n_subjects": len(per_subject),
               "subject_macro_hit3_gate": macro("hit3_gate"),
               "subject_macro_hit3_q0": macro("hit3_q0"),
               "subject_macro_hit3_q16": macro("hit3_q16"),
               "subject_macro_query_rate": macro("query_rate"),
               "subject_macro_mean_queries": macro("mean_deployed_queries"),
               "subject_macro_utility": macro("utility"),
               "observed_query_count_simulated": gate.budget * sum(gate.should_query(r) for r in rows),
               "offline_evaluations_for_paired_q16": gate.budget * len(rows),
               "cost_per_full_budget": float(cost_per_full_budget)}
    return summary, per_subject

def fit_query_gate(calibration_rows: Sequence[dict], budget: int = 16,
                   cost_per_full_budget: float = 0.05) -> tuple[QueryGate, dict]:
    """Select simple Q0/Q16 policy by inner-subject macro utility only.

    The candidate family is fixed before seeing held-out outer results.
    A tie prefers fewer model calls and a simpler rule. No population
    calibration guarantee is claimed.
    """
    _validate(calibration_rows)
    if len({int(r["subject"]) for r in calibration_rows}) < 2:
        raise ValueError("calibration requires at least two subjects")
    candidates = [QueryGate("q0", budget=budget), QueryGate("q16", budget=budget)]
    quantiles = (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
    for feature in FEATURES:
        values = np.array([float(r[feature]) for r in calibration_rows])
        thresholds = sorted(set(float(v) for v in np.quantile(values, quantiles)))
        for direction in ("le", "ge"):
            for threshold in thresholds:
                candidates.append(QueryGate("threshold", feature, direction, threshold, budget))
    best_gate = candidates[0]
    best_summary, _ = evaluate_gate(calibration_rows, best_gate, cost_per_full_budget)
    def score(s: dict, rule: QueryGate) -> tuple:
        return (round(s["subject_macro_utility"], 12),
                -round(s["subject_macro_query_rate"], 12),
                int(rule.mode != "threshold"))
    for gate in candidates[1:]:
        summary, _ = evaluate_gate(calibration_rows, gate, cost_per_full_budget)
        if score(summary, gate) > score(best_summary, best_gate):
            best_gate, best_summary = gate, summary
    return best_gate, {"calibration": best_summary, "searched_policies": len(candidates),
                       "selection_metric": "subject_macro_hit3 - lambda * subject_macro_Q16_rate",
                       "cost_per_full_budget": cost_per_full_budget}
