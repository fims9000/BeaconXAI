"""Q-bounded two-operator audit with explicit disagreement/abstention.

Both operator outcomes are actually measured on the *same* top Q/2
training-prior candidates. No held-out labels, injected patch IDs, or full-Q
oracle responses are required. A consensus flag is diagnostic, NOT a
statistical coverage guarantee nor evidence of causal/physical importance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .class_counterfactual import replace_component
from .contrastive import neutralize_interp
from .sensitivity_reference import moving_margin_effects, rank_effects


@dataclass(frozen=True)
class DualOperatorResult:
    ranking: list[int]
    selected_components: list[int]
    interpolation_ranking: list[int]
    donor_ranking: list[int]
    top3_agreement: float
    reported: bool
    extra_model_evaluations: int
    baseline_model_evaluations: int


def dual_operator_audit(
    x: np.ndarray,
    components: Sequence,
    cheap_scores: np.ndarray,
    predict_many: Callable[[np.ndarray], np.ndarray],
    baseline_scores: np.ndarray,
    rival_template: np.ndarray,
    *,
    budget: int = 16,
    taper: int = 2,
    minimum_common_top3: int = 2,
) -> DualOperatorResult:
    """Split an even Q across two operators; return Borda consensus or abstain.

    The consensus rank has Q/2 items, each measured with BOTH operators,
    totaling exactly Q model examples (one batched predict_many call).
    The caller is responsible for the one cached original prediction.
    The reported flag is a fixed, pre-specified heuristic: >=2 shared
    top-three components among the two measured operator rankings.
    """
    x = np.asarray(x)
    prior = np.asarray(cheap_scores, dtype=float)
    base = np.asarray(baseline_scores, dtype=float)
    template = np.asarray(rival_template)
    n = len(components)
    if (x.ndim != 2 or template.shape != x.shape
        or prior.shape != (n,) or base.ndim != 1 or base.size < 2
        or not np.isfinite(x).all() or not np.isfinite(template).all()
        or not np.isfinite(prior).all() or not np.isfinite(base).all()):
        raise ValueError("invalid sample, model outputs, prior or rival template")
    if (not isinstance(budget, (int, np.integer)) or budget % 2
        or budget < 6 or budget // 2 > n
        or not isinstance(taper, int) or taper < 0
        or minimum_common_top3 not in (1, 2, 3)):
        raise ValueError("budget must be even with 3<=Q/2<=M; invalid policy")
    k = budget // 2
    selected = np.argsort(-prior, kind="stable")[:k].astype(int).tolist()
    left = np.stack([neutralize_interp(x, components[i]) for i in selected])
    right = np.stack([replace_component(x, components[i], template, taper=taper)
                      for i in selected])
    if left.shape != right.shape or left.shape[0] != k:
        raise AssertionError("intervention geometry mismatch")
    predictions = np.asarray(predict_many(np.concatenate((left, right), axis=0)))
    if predictions.shape != (budget, base.size) or not np.isfinite(predictions).all():
        raise ValueError("predict_many must return finite scores [Q, classes]")
    int_effect = moving_margin_effects(base, predictions[:k])
    donor_effect = moving_margin_effects(base, predictions[k:])
    interp_rank = rank_effects(int_effect, selected)
    donor_rank = rank_effects(donor_effect, selected)

    # Borda fusion uses only ranks among genuinely measured candidates;
    # stable ties resolve by the original training-prior component order.
    int_pos = {component: rank for rank, component in enumerate(interp_rank)}
    donor_pos = {component: rank for rank, component in enumerate(donor_rank)}
    combined = sorted(selected, key=lambda i: (int_pos[i] + donor_pos[i], selected.index(i)))
    common = len(set(interp_rank[:3]) & set(donor_rank[:3]))
    return DualOperatorResult(
        ranking=combined,
        selected_components=selected,
        interpolation_ranking=interp_rank,
        donor_ranking=donor_rank,
        top3_agreement=common / 3.0,
        reported=common >= minimum_common_top3,
        extra_model_evaluations=budget,
        baseline_model_evaluations=0,
    )
