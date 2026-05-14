from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .neutralization import Neutralizer
from .partition import (
    components_cost,
    make_initial_partition,
    make_initial_partition_time,
    split_component,
    split_component_time,
)
from .types import AuditResult, BeaconConfig, Component, LeafStats, LogitFn


@dataclass
class _State:
    leaves: dict[str, Component]
    deltas: dict[str, float]
    switches: dict[str, int]
    history: dict[str, float]


class BeaconAudit:
    def __init__(self, model_logits: LogitFn, neutralizer: Neutralizer, config: BeaconConfig):
        self.model_logits = model_logits
        self.neutralizer = neutralizer
        self.cfg = config

    def audit(self, x: np.ndarray) -> AuditResult:
        if x.ndim != 2:
            raise ValueError("x must be 2D: [time, channels]")

        logits0 = self.model_logits(x)
        y_hat = int(np.argmax(logits0))
        competitor = self._competitor_from_logits(logits0, y_hat)
        m0 = self._margin_from_logits(logits0, y_hat, competitor)

        p0 = self._make_partition(x.shape[0], x.shape[1], self.cfg.k0)
        q_init = len(p0)
        if self.cfg.q_max < q_init:
            raise ValueError(f"invalid config: q_max={self.cfg.q_max} < K0={q_init}")

        q_remaining = self.cfg.q_max - q_init

        state = _State(leaves={c.cid: c for c in p0}, deltas={}, switches={}, history={})
        q_used = 0

        for comp in p0:
            delta, switch_flag = self._delta_switch(x, y_hat, m0, [comp], competitor)
            state.deltas[comp.cid] = delta
            state.switches[comp.cid] = int(switch_flag)
            state.history[comp.cid] = delta
            q_used += 1

        q_frag, q_ref = self._allocate_budget(q_remaining, state.deltas)
        if str(getattr(self.cfg, "audit_mode", "full")) == "counter_only":
            q_frag = 0

        q_ref_used = self._refine(x, y_hat, m0, state, q_ref, competitor)
        q_used += q_ref_used

        leaf_stats = self._leaf_stats(state)
        s_plus = sorted((ls for ls in leaf_stats if ls.delta > 0), key=lambda z: z.delta, reverse=True)[: self.cfg.k_pos]
        s_minus = sorted((ls for ls in leaf_stats if ls.delta < 0), key=lambda z: abs(z.delta), reverse=True)[: self.cfg.k_neg]
        support_mass = float(sum(max(ls.delta, 0.0) for ls in leaf_stats))
        counter_mass = float(sum(max(-ls.delta, 0.0) for ls in leaf_stats))
        pos_count = int(sum(ls.delta > 0 for ls in leaf_stats))
        neg_count = int(sum(ls.delta < 0 for ls in leaf_stats))
        counter_ratio = float(counter_mass / (support_mass + self.cfg.eps))

        positive_all = sorted((ls for ls in leaf_stats if ls.delta > 0), key=lambda z: z.delta, reverse=True)
        rho_b, rho_b_cost, censored, q_frag_used, m_last, k_checked, support_mass_removed = self._fragility(
            x, y_hat, m0, positive_all, q_frag, competitor
        )
        q_used += q_frag_used

        risk_b = self._risk_from_fragility(rho_b_cost, censored)
        drop_ratio = float((m0 - m_last) / (abs(m0) + self.cfg.eps))
        residual_ratio = float(m_last / (abs(m0) + self.cfg.eps))

        suff_margin, suff_keep = self._sufficiency(x, y_hat, competitor, state.leaves.values(), s_plus)
        necessity = self._necessity(x, y_hat, competitor, m0, s_plus)
        ce_gain = self._counter_evidence_gain(x, y_hat, competitor, m0, s_minus)

        return AuditResult(
            y_hat=y_hat,
            m0=m0,
            s_plus=s_plus,
            s_minus=s_minus,
            rho_b=rho_b,
            rho_b_cost=rho_b_cost,
            risk_b=risk_b,
            censored=censored,
            m_last=m_last,
            drop_ratio=drop_ratio,
            residual_ratio=residual_ratio,
            k_checked=k_checked,
            support_mass_removed=support_mass_removed,
            support_mass=support_mass,
            counter_mass=counter_mass,
            counter_ratio=counter_ratio,
            pos_count=pos_count,
            neg_count=neg_count,
            sufficiency_margin=suff_margin,
            sufficiency_kept_class=suff_keep,
            necessity=necessity,
            counter_evidence_gain=ce_gain,
            q_used=q_used,
            q_init=q_init,
            q_ref_used=q_ref_used,
            q_frag_used=q_frag_used,
            metadata={
                "q_ref_alloc": float(q_ref),
                "q_frag_alloc": float(q_frag),
                "budget_mode": str(getattr(self.cfg, "budget_mode", "fixed")),
                "leaf_components": [
                    (ls.component.cid, ls.component.t0, ls.component.t1, ls.component.c0, ls.component.c1)
                    for ls in leaf_stats
                ],
                "leaf_deltas": [float(ls.delta) for ls in leaf_stats],
            },
        )

    def _refine(self, x: np.ndarray, y_hat: int, m0: float, state: _State, q_ref: int, competitor: int | None) -> int:
        q_ref_used = 0
        queue: set[str] = {
            cid
            for cid, c in state.leaves.items()
            if c.n_points >= self.cfg.l_min and self._split(c) and self._queue_accepts_delta(state.deltas[cid])
        }

        while queue and q_ref_used < q_ref:
            max_abs = max(abs(state.deltas[cid]) for cid in state.leaves) + self.cfg.eps
            m0_low = float(m0 < self.cfg.tau_m)

            if self.cfg.refinement_policy == "uniform":
                target_cid = sorted(queue)[0]
            elif self.cfg.refinement_policy == "none":
                break
            else:
                target_cid = max(
                    queue,
                    key=lambda cid: self._priority(state.deltas[cid], state.switches.get(cid, 0), max_abs, m0_low),
                )

            target = state.leaves[target_cid]
            children = self._split(target)
            if not children:
                queue.remove(target_cid)
                continue

            needed = len(children)
            if q_ref_used + needed > q_ref:
                if self.cfg.stop_if_not_enough_ref_budget:
                    break
                queue.remove(target_cid)
                continue

            del state.leaves[target_cid]
            queue.remove(target_cid)

            for child in children:
                delta, switch_flag = self._delta_switch(x, y_hat, m0, [child], competitor)
                state.leaves[child.cid] = child
                state.deltas[child.cid] = delta
                state.switches[child.cid] = int(switch_flag)
                state.history[child.cid] = delta
                if child.n_points >= self.cfg.l_min and self._split(child) and self._queue_accepts_delta(delta):
                    queue.add(child.cid)

            q_ref_used += needed

        return q_ref_used

    def _fragility(
        self,
        x: np.ndarray,
        y_hat: int,
        m0: float,
        s_plus: Sequence[LeafStats],
        q_frag: int,
        competitor: int | None,
    ) -> tuple[int, float, bool, int, float, int, float]:
        if not s_plus:
            return 1, 1.0, True, 0, float(m0), 0, 0.0

        total_points = x.shape[0] * x.shape[1]
        selected: list[Component] = []
        q_frag_used = 0
        checked_cost = 0.0
        margin_last = float(m0)
        support_mass_removed = 0.0

        for idx, stat in enumerate(s_plus, start=1):
            if q_frag_used >= q_frag:
                break

            selected.append(stat.component)
            z = self.neutralizer(x, selected)
            margin = self._margin(z, y_hat, competitor)
            q_frag_used += 1
            checked_cost = components_cost(selected, total_points)
            margin_last = float(margin)
            support_mass_removed += float(max(stat.delta, 0.0))

            if margin <= 0.0:
                return idx, checked_cost, False, q_frag_used, margin_last, q_frag_used, support_mass_removed

        return (
            q_frag_used + 1,
            checked_cost if checked_cost > 0 else 1.0,
            True,
            q_frag_used,
            margin_last,
            q_frag_used,
            support_mass_removed,
        )

    def _sufficiency(
        self,
        x: np.ndarray,
        y_hat: int,
        competitor: int | None,
        leaves: Iterable[Component],
        s_plus: Sequence[LeafStats],
    ) -> tuple[float, bool]:
        keep_ids = {s.component.cid for s in s_plus}
        drop = [leaf for leaf in leaves if leaf.cid not in keep_ids]
        x_keep = self.neutralizer(x, drop)
        logits_keep = self.model_logits(x_keep)
        margin_keep = self._margin_from_logits(logits_keep, y_hat, competitor)
        return margin_keep, int(np.argmax(logits_keep)) == y_hat

    def _necessity(self, x: np.ndarray, y_hat: int, competitor: int | None, m0: float, s_plus: Sequence[LeafStats]) -> float:
        if not s_plus:
            return 0.0
        z = self.neutralizer(x, [s.component for s in s_plus])
        return m0 - self._margin(z, y_hat, competitor)

    def _counter_evidence_gain(
        self, x: np.ndarray, y_hat: int, competitor: int | None, m0: float, s_minus: Sequence[LeafStats]
    ) -> float:
        if not s_minus:
            return 0.0
        z = self.neutralizer(x, [s.component for s in s_minus])
        return self._margin(z, y_hat, competitor) - m0

    def _leaf_stats(self, state: _State) -> list[LeafStats]:
        max_abs = max(abs(state.deltas[cid]) for cid in state.leaves) + self.cfg.eps
        out: list[LeafStats] = []
        for cid, comp in state.leaves.items():
            delta = state.deltas[cid]
            out.append(LeafStats(component=comp, delta=delta, abs_norm=abs(delta) / max_abs))
        return out

    def _priority(self, delta: float, switch_flag: int, max_abs: float, m0_low: float) -> float:
        s = abs(delta) / max_abs
        mode = str(getattr(self.cfg, "refinement_mode", "mixed"))
        switch_bonus = float(getattr(self.cfg, "switch_eta", 0.0)) * float(switch_flag) if str(getattr(self.cfg, "priority_mode", "base")) == "switch" else 0.0
        if mode == "support":
            return s + self.cfg.beta * float(delta > 0 and 0 < s < self.cfg.tau_s) + switch_bonus
        if mode == "counter":
            return s + self.cfg.alpha * float(delta < 0) + switch_bonus
        return (
            s
            + self.cfg.alpha * float(delta < 0)
            + self.cfg.beta * float(delta > 0 and 0 < s < self.cfg.tau_s)
            + self.cfg.gamma * m0_low * float(delta > 0)
            + switch_bonus
        )

    def _queue_accepts_delta(self, delta: float) -> bool:
        if str(getattr(self.cfg, "audit_mode", "full")) == "counter_only":
            return bool(delta < 0)
        mode = str(getattr(self.cfg, "refinement_mode", "mixed"))
        if mode == "support":
            return bool(delta > 0)
        if mode == "counter":
            return bool(delta < 0)
        return True

    def _make_partition(self, t_steps: int, channels: int, k0: int) -> list[Component]:
        if self.cfg.partition_mode == "time_only":
            return make_initial_partition_time(t_steps, channels, k0)
        return make_initial_partition(t_steps, channels, k0)

    def _split(self, comp: Component) -> list[Component]:
        if self.cfg.partition_mode == "time_only":
            return split_component_time(comp)
        return split_component(comp)

    def _risk_from_fragility(self, rho_b_cost: float, censored: bool) -> float:
        base = 1.0 / (1.0 + rho_b_cost)
        if self.cfg.risk_policy == "rho_censored_boost":
            # Censored means we failed to break the decision within budget:
            # keep base risk, but add mild uncertainty penalty.
            return float(min(1.0, base + (0.10 if censored else 0.0)))
        return float(base)

    def _delta(self, x: np.ndarray, y_hat: int, m0: float, components: Sequence[Component], competitor: int | None = None) -> float:
        z = self.neutralizer(x, components)
        return m0 - self._margin(z, y_hat, competitor)

    def _delta_switch(
        self, x: np.ndarray, y_hat: int, m0: float, components: Sequence[Component], competitor: int | None
    ) -> tuple[float, int]:
        z = self.neutralizer(x, components)
        logits = self.model_logits(z)
        margin = self._margin_from_logits(logits, y_hat, competitor)
        delta = m0 - margin
        pred = int(np.argmax(logits))
        return float(delta), int(pred != y_hat)

    def _allocate_budget(self, q_remaining: int, deltas: Dict[str, float]) -> tuple[int, int]:
        mode = str(getattr(self.cfg, "budget_mode", "fixed"))
        if q_remaining <= 0:
            return 0, 0
        if mode == "conflict_first":
            counter_mass = float(sum(max(-d, 0.0) for d in deltas.values()))
            if counter_mass > float(getattr(self.cfg, "tau_conflict", 0.0)):
                q_ref = int(round(0.85 * q_remaining))
                q_ref = max(0, min(q_ref, q_remaining))
                q_frag = q_remaining - q_ref
                return q_frag, q_ref
        q_frag = floor(self.cfg.q_frag_ratio * q_remaining)
        q_ref = q_remaining - q_frag
        return q_frag, q_ref

    def _margin(self, x: np.ndarray, y_hat: int, competitor: int | None = None) -> float:
        logits = self.model_logits(x)
        return self._margin_from_logits(logits, y_hat, competitor)

    @staticmethod
    def _competitor_from_logits(logits: np.ndarray, y_hat: int) -> int:
        return int(np.argmax(np.where(np.arange(len(logits)) == y_hat, -np.inf, logits)))

    def _margin_from_logits(self, logits: np.ndarray, y_hat: int, competitor: int | None = None) -> float:
        ref = float(logits[y_hat])
        mode = str(getattr(self.cfg, "margin_mode", "adaptive_all"))
        if mode == "nearest_competitor":
            c = self._competitor_from_logits(logits, y_hat) if competitor is None else int(competitor)
            alt = float(logits[c])
        else:
            alt = float(np.max(np.delete(logits, y_hat)))
        return ref - alt
