from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


Array = np.ndarray
LogitFn = Callable[[Array], Array]


@dataclass(frozen=True)
class Component:
    cid: str
    t0: int
    t1: int
    c0: int
    c1: int
    depth: int = 0
    parent: Optional[str] = None

    @property
    def n_points(self) -> int:
        return (self.t1 - self.t0) * (self.c1 - self.c0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.t1 - self.t0, self.c1 - self.c0


@dataclass
class BeaconConfig:
    q_max: int
    k0: int
    l_min: int = 4
    k_pos: int = 3
    k_neg: int = 3
    q_frag_ratio: float = 0.25

    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 1.0
    tau_s: float = 0.10
    tau_m: float = 0.0
    eps: float = 1e-8
    refinement_policy: str = "priority"
    refinement_mode: str = "mixed"
    priority_mode: str = "base"
    switch_eta: float = 0.0
    budget_mode: str = "fixed"
    tau_conflict: float = 0.0
    margin_mode: str = "adaptive_all"
    audit_mode: str = "full"
    partition_mode: str = "time_only"
    risk_policy: str = "rho_only"

    stop_if_not_enough_ref_budget: bool = True
    fast_core: bool = False


@dataclass
class LeafStats:
    component: Component
    delta: float
    abs_norm: float


@dataclass
class AuditResult:
    y_hat: int
    m0: float
    s_plus: List[LeafStats]
    s_minus: List[LeafStats]
    rho_b: int
    rho_b_cost: float
    risk_b: float
    censored: bool
    m_last: float
    drop_ratio: float
    residual_ratio: float
    k_checked: int
    support_mass_removed: float
    support_mass: float
    counter_mass: float
    counter_ratio: float
    pos_count: int
    neg_count: int

    sufficiency_margin: float
    sufficiency_kept_class: bool
    necessity: float
    counter_evidence_gain: float

    q_used: int
    q_init: int
    q_ref_used: int
    q_frag_used: int

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetSplit:
    q_init: int
    q_ref: int
    q_frag: int


@dataclass
class BaseScores:
    confidence: float
    entropy: float
    margin: float


@dataclass
class RiskEvalRow:
    sample_id: int
    is_error: int
    q_max: int
    method: str
    risk_score: float
    q_used: int
    censored: int


@dataclass
class LocalMetricRow:
    sample_id: int
    q_max: int
    method: str
    sufficiency_margin: float
    sufficiency_kept_class: int
    necessity: float
    counter_evidence_gain: float
    rho_b: int
    rho_b_cost: float
    censored: int
    m0: float = 0.0
    m_last: float = 0.0
    drop_ratio: float = 0.0
    residual_ratio: float = 1.0
    k_checked: int = 0
    support_mass_removed: float = 0.0
    support_mass: float = 0.0
    counter_mass: float = 0.0
    counter_ratio: float = 0.0
    pos_count: int = 0
    neg_count: int = 0
