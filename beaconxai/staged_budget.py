"""Train-calibrated staged Q8/Q16/Q32 BEACON audit on natural signals.

The online path observes only an original model prediction, class-mean
training profiles and model responses to already queried interpolations.
It never sees the offline Q36 reference or held-out test labels.

Fixed policies are included so calibration can reject an ineffective
adaptive rule instead of forcing it into use.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .contrastive import neutralize_interp
from .sensitivity_reference import moving_margin_effects, rank_effects

STAGES = (8, 16, 32)
THRESHOLDS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 1.01)


@dataclass(frozen=True)
class StagedPolicy:
    mode: str
    threshold_q8: float = 1.01
    threshold_q16: float = 1.01

    def __post_init__(self):
        if self.mode not in ("fixed8", "fixed16", "fixed32", "adaptive"):
            raise ValueError("unknown staged budget policy")
        if (not np.isfinite(self.threshold_q8)
            or not np.isfinite(self.threshold_q16)
            or self.threshold_q8 < 0 or self.threshold_q16 < 0):
            raise ValueError("invalid staged threshold")


@dataclass(frozen=True)
class StagedResult:
    ranking: list[int]
    selected: list[int]
    queries: int
    stage_features: dict[int, float]
    actual_forward_batches: int


def measured_top3_gap(effects: np.ndarray) -> float:
    """Scale-invariant third-to-fourth gap among *already measured* effects."""
    a = np.asarray(effects, dtype=float)
    if a.ndim != 1 or len(a) < 4 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("need at least four finite nonnegative observed effects")
    s = np.sort(a)
    return float(max(0.0, s[-3] - s[-4]) / max(float(s[-1]), 1e-12))


def choose_budget(policy: StagedPolicy, gap8: float, gap16: float) -> int:
    if not all(np.isfinite(x) and x >= 0 for x in (gap8, gap16)):
        raise ValueError("invalid observed gap feature")
    if policy.mode.startswith("fixed"):
        return int(policy.mode[5:])
    if gap8 >= policy.threshold_q8:
        return 8
    if gap16 >= policy.threshold_q16:
        return 16
    return 32


def run_staged_audit(
    x: np.ndarray, components: Sequence, cheap_prior: np.ndarray,
    baseline: np.ndarray, predict_many: Callable[[np.ndarray], np.ndarray],
    policy: StagedPolicy,
) -> StagedResult:
    """Query in true 8+8+16 stages; skip all unused later model calls."""
    z = np.asarray(x)
    prior = np.asarray(cheap_prior, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if (z.ndim != 2 or len(components) < 32
        or prior.shape != (len(components),) or base.ndim != 1
        or len(base) < 2 or not np.isfinite(z).all()
        or not np.isfinite(prior).all() or not np.isfinite(base).all()):
        raise ValueError("invalid staged model input or original predictions")
    ordered_ids = np.argsort(-prior, kind="stable")[:32].astype(int).tolist()
    measured = []
    features = {}
    n_batches = 0
    end = 0
    for q in STAGES:
        next_ids = ordered_ids[end:q]
        patched = np.stack([neutralize_interp(z, components[j]) for j in next_ids])
        outputs = np.asarray(predict_many(patched))
        if outputs.shape != (len(next_ids), len(base)) or not np.isfinite(outputs).all():
            raise ValueError("predict_many must return finite [new probes, classes]")
        measured.extend(moving_margin_effects(base, outputs).tolist())
        end = q
        n_batches += 1
        if q in (8, 16):
            features[q] = measured_top3_gap(np.array(measured))
        if (policy.mode == "fixed8" and q == 8
            or policy.mode == "fixed16" and q == 16
            or policy.mode == "adaptive" and
                (q == 8 and features[8] >= policy.threshold_q8
                 or q == 16 and features[16] >= policy.threshold_q16)
            or q == 32):
            break
    return StagedResult(
        ranking=rank_effects(np.asarray(measured), ordered_ids[:end]),
        selected=ordered_ids[:end], queries=end,
        stage_features=features, actual_forward_batches=n_batches,
    )


def _validate_calibration_rows(rows: Sequence[dict]) -> None:
    if not rows or len({int(r["subject"]) for r in rows}) < 2:
        raise ValueError("calibration needs at least two train-only subjects")
    for r in rows:
        vals = [float(r[x]) for x in ("gap8", "gap16", "overlap8",
                                      "overlap16", "overlap32")]
        if not all(np.isfinite(z) for z in vals):
            raise ValueError("nonfinite calibration features/outcomes")
        if vals[0] < 0 or vals[1] < 0 or any(not 0 <= x <= 1 for x in vals[2:]):
            raise ValueError("invalid calibration quality or observed gap")


def evaluate_policy_offline(
    rows: Sequence[dict], policy: StagedPolicy, penalty: float,
) -> dict:
    """Inner-validation-only utility; outer-test Q36 must never fit policy."""
    _validate_calibration_rows(rows)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("invalid per-Q32 cost penalty")
    subjects = sorted({int(r["subject"]) for r in rows})
    subject_reports = []
    for subject in subjects:
        subset = [r for r in rows if int(r["subject"]) == subject]
        budgets = [choose_budget(policy,float(r["gap8"]),float(r["gap16"]))
                   for r in subset]
        overlap = [float(r["overlap"+str(q)]) for r,q in zip(subset,budgets)]
        subject_reports.append({
            "subject": subject, "n": len(subset),
            "mean_overlap": float(np.mean(overlap)),
            "mean_q": float(np.mean(budgets)),
            "utility": float(np.mean(
                [v-penalty*q/32.0 for v,q in zip(overlap,budgets)])),
        })
    return {
        "n_subjects": len(subject_reports),
        "n_windows": len(rows),
        "macro_overlap": float(np.mean([r["mean_overlap"] for r in subject_reports])),
        "macro_q": float(np.mean([r["mean_q"] for r in subject_reports])),
        "macro_utility": float(np.mean([r["utility"] for r in subject_reports])),
    }


def fit_staged_policy(rows: Sequence[dict], penalty: float = 0.75,
                      required_gain: float = 0.01) -> tuple[StagedPolicy, dict]:
    """Pick adaptive rule only if it beats the best fixed Q on inner subjects.

    Frozen candidate thresholds and a 1-point utility improvement threshold
    prevent post-hoc cherry-picking on official UCI HAR held-out subjects.
    """
    _validate_calibration_rows(rows)
    if required_gain < 0 or not np.isfinite(required_gain):
        raise ValueError("invalid minimum improvement")
    fixed = [StagedPolicy("fixed8"),StagedPolicy("fixed16"),StagedPolicy("fixed32")]
    fixed_results = [(p, evaluate_policy_offline(rows,p,penalty)) for p in fixed]
    # Tie: fewer queries, then deterministic lower-Q policy.
    winner, base = max(fixed_results,key=lambda p:(
        round(p[1]["macro_utility"],10), -p[1]["macro_q"]))
    best_adaptive, best_summary = None,None
    for t8 in THRESHOLDS:
        for t16 in THRESHOLDS:
            policy = StagedPolicy("adaptive",t8,t16)
            report = evaluate_policy_offline(rows,policy,penalty)
            if (best_adaptive is None or
                (round(report["macro_utility"],10),-report["macro_q"]) >
                (round(best_summary["macro_utility"],10),-best_summary["macro_q"])):
                best_adaptive,best_summary=policy,report
    promote = (best_summary["macro_utility"] >
               base["macro_utility"] + required_gain)
    selected = best_adaptive if promote else winner
    chosen = best_summary if promote else base
    return selected,{
        "calibration_subjects":sorted({int(r["subject"]) for r in rows}),
        "calibration_windows":len(rows),
        "penalty_per_32_extra_queries":float(penalty),
        "fixed_q_controls":[{"policy":p.mode,**d} for p,d in fixed_results],
        "best_fixed_policy":winner.mode,
        "best_fixed_utility":base["macro_utility"],
        "best_adaptive_threshold_q8":best_adaptive.threshold_q8,
        "best_adaptive_threshold_q16":best_adaptive.threshold_q16,
        "best_adaptive_utility":best_summary["macro_utility"],
        "required_adaptive_utility_gain":required_gain,
        "adaptive_promoted":bool(promote),
        "selected_policy":selected.mode,
        "selected_macro_utility":chosen["macro_utility"],
        "full_Q36_used_only_for_inner_calibration":True,
        "no_outer_test_labels_or_reference_used_to_fit":True,
    }
