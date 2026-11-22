"""Class-prototype interventions for model-level explanations of time-series predictions.

This is a *contrastive perturbation audit*, not a validated natural counterfactual:
class-wise means/medians can be off-manifold; model-score changes are not causal
effects in the data-generating process. Prototypes MUST be fitted on TRAIN only.
Every input row evaluated by the black-box model counts toward the query budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

from .contrastive import Component


@dataclass(frozen=True)
class PrototypeBank:
    mean: dict[int, np.ndarray]
    median: dict[int, np.ndarray]
    counts: dict[int, int]

    @property
    def labels(self) -> list[int]:
        return sorted(self.counts)

    @property
    def bytes_used(self) -> int:
        return sum(v.nbytes for d in (self.mean, self.median) for v in d.values())

    def get(self, label: int, kind: str) -> np.ndarray:
        if kind not in ("mean", "median"):
            raise ValueError("prototype kind must be mean or median")
        return (self.mean if kind == "mean" else self.median)[label]


def fit_prototype_bank(x_train: np.ndarray, y_train: np.ndarray,
                       min_count: int = 2) -> PrototypeBank:
    """Compute class-specific time/channel templates; no test windows allowed."""
    if x_train.ndim != 3 or y_train.ndim != 1 or len(x_train) != len(y_train):
        raise ValueError("training arrays must be [N,T,C] and [N]")
    if len(x_train) == 0 or not np.isfinite(x_train).all():
        raise ValueError("training inputs must be nonempty and finite")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    means, medians, counts = {}, {}, {}
    for label in np.unique(y_train):
        arr = np.asarray(x_train[y_train == label], dtype=np.float32)
        if len(arr) < min_count:
            continue
        means[int(label)] = np.mean(arr, axis=0, dtype=np.float64).astype(np.float32)
        medians[int(label)] = np.median(arr, axis=0).astype(np.float32)
        counts[int(label)] = int(len(arr))
    if not counts:
        raise ValueError("no training class has sufficient examples")
    return PrototypeBank(means, medians, counts)


def replace_component(x: np.ndarray, c: Component, template: np.ndarray,
                      taper: int = 2) -> np.ndarray:
    """Replace just the selected time/channel region with a class prototype.

    Optional taper linearly blends the template at the boundaries; taper=0
    applies the full prototype. Taper is not physical-realism validation.
    """
    if x.ndim != 2 or template.shape != x.shape:
        raise ValueError("sample and template must have same [T,C] shape")
    if not (0 <= c.t0 < c.t1 <= len(x) and 0 <= c.c0 < c.c1 <= x.shape[1]):
        raise ValueError("component exceeds sample")
    if not isinstance(taper, int) or taper < 0:
        raise ValueError("taper must be a nonnegative integer")
    out = x.copy()
    n = c.t1 - c.t0
    w = np.ones(n, dtype=float)
    edge = min(taper, n // 2)
    if edge:
        w[:edge] = np.linspace(0, 1, edge + 2)[1:-1]
        w[-edge:] = np.linspace(1, 0, edge + 2)[1:-1]
    sl = np.s_[c.t0:c.t1, c.c0:c.c1]
    out[sl] = x[sl] * (1 - w[:, None]) + template[sl] * w[:, None]
    return out


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=float)
    if z.ndim != 1 or not np.isfinite(z).all() or z.size < 2:
        raise ValueError("class scores must be finite, 1D and have >=2 classes")
    e = np.exp(z - np.max(z))
    return e / e.sum()


def explain_class_pair(
    x: np.ndarray,
    predict_many: Callable[[np.ndarray], np.ndarray],
    bank: PrototypeBank,
    components: Sequence[Component],
    selected: Sequence[int],
    source_class: int,
    rival_class: int,
    q_max: int = 16,
    prototype_mode: str = "both",
    baseline: np.ndarray | None = None,
    taper: int = 2,
) -> dict:
    """Intervene with each source and rival template for every selected region.

    With 'both' = 2 classes x 2 prototype kinds = 4 model evaluations per
    component. Q16 therefore permits at most 4 components, NOT 16 components.

    Returns signed deltas on FIXED pair a/b and full class scores.
    If source/rival is absent in train or scores, fail rather than fabricate.
    """
    if x.ndim != 2:
        raise ValueError("x must be [T,C]")
    if source_class == rival_class:
        raise ValueError("source and rival must differ")
    if source_class not in bank.counts or rival_class not in bank.counts:
        raise ValueError("source and rival must have training prototypes")
    if prototype_mode not in ("both", "mean", "median"):
        raise ValueError("prototype_mode must be both, mean, or median")
    kinds = ("mean", "median") if prototype_mode == "both" else (prototype_mode,)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("select at least one distinct component")
    if any(j < 0 or j >= len(components) for j in selected):
        raise ValueError("component ID out of range")
    n_queries = len(selected) * 2 * len(kinds)
    if q_max < 1 or n_queries > q_max:
        raise ValueError("selected components exceed the model-evaluation budget")
    base = np.asarray(predict_many(x[None])[0] if baseline is None else baseline, dtype=float)
    p0 = softmax(base)
    if max(source_class, rival_class) >= len(base) or min(source_class, rival_class) < 0:
        raise ValueError("class index outside score vector")
    a, b = source_class, rival_class
    interventions = []
    edited = []
    for j in selected:
        comp = components[j]
        for cls in (a, b):
            for kind in kinds:
                template = bank.get(cls, kind)
                edited.append(replace_component(x, comp, template, taper=taper))
                interventions.append((int(j), int(cls), kind))
    outputs = np.asarray(predict_many(np.stack(edited)), dtype=float)
    if outputs.shape != (n_queries, len(base)) or not np.isfinite(outputs).all():
        raise ValueError("predict_many must return finite scores [Q,classes]")
    records = []
    for (j, cls, kind), edited_x, scores in zip(interventions, edited, outputs):
        comp = components[j]
        p1 = softmax(scores)
        delta_margin = float((scores[a] - scores[b]) - (base[a] - base[b]))
        records.append({
            "component": int(j), "group": comp.group, "time_bin": comp.time_bin,
            "t0": comp.t0, "t1": comp.t1, "c0": comp.c0, "c1": comp.c1,
            "prototype_class": int(cls), "prototype_kind": kind,
            "source": a, "rival": b,
            "p_source_before": float(p0[a]), "p_rival_before": float(p0[b]),
            "p_source_after": float(p1[a]), "p_rival_after": float(p1[b]),
            "delta_p_source": float(p1[a] - p0[a]),
            "delta_p_rival": float(p1[b] - p0[b]),
            "gap_before": float(base[a] - base[b]),
            "gap_after": float(scores[a] - scores[b]),
            "delta_gap": delta_margin,
            "predicted_after": int(np.argmax(scores)),
            "flip_to_rival": bool(int(np.argmax(scores)) == b),
            "relative_patch_change_l2": float(np.linalg.norm(
                (edited_x - x)[comp.t0:comp.t1, comp.c0:comp.c1]
            ) / (np.linalg.norm(x[comp.t0:comp.t1, comp.c0:comp.c1]) + 1e-8)),
        })
    return {
        "source": a, "rival": b,
        "original_predicted": int(np.argmax(base)),
        "original_scores": base.tolist(),
        "original_probabilities": p0.tolist(),
        "selected_components": list(map(int, selected)),
        "queries_additional": n_queries,
        "baseline_queries": 0 if baseline is not None else 1,
        "records": records,
    }


def describe_intervention(r: dict, class_names: dict[int, str],
                          sensor_names: Sequence[str] = ("wrist", "chest", "ankle")) -> str:
    """State BOTH class score effects and actual new predicted class, not an inferred cause."""
    a, b, c = r["source"], r["rival"], r["prototype_class"]
    sensor = sensor_names[r["group"]] if r["group"] < len(sensor_names) else f"sensor {r['group']}"
    ppa = r["delta_p_source"] * 100
    ppb = r["delta_p_rival"] * 100
    kind = "средний" if r["prototype_kind"] == "mean" else "медианный"
    new_cls = int(r["predicted_after"])
    if new_cls == a:
        outcome = "исходный класс остался первым"
    elif new_cls == b:
        outcome = f"модель переключилась на конкурента «{class_names.get(b,str(b))}»"
    else:
        outcome = (f"модель переключилась на третий класс "
                   f"«{class_names.get(new_cls,str(new_cls))}»")
    directed = (ppa < -1e-9 and ppb > 1e-9)
    interpretation = ("обе оценки изменились в сторону конкурента" if directed else
                      "направленное смещение к конкуренту НЕ подтверждено")
    return (
        f"{sensor}, отсчёты {r['t0']}–{r['t1']}: подставили {kind} прототип "
        f"класса «{class_names.get(c,str(c))}». "
        f"Оценка «{class_names.get(a,str(a))}»: "
        f"{r['p_source_before']*100:.1f}% → {r['p_source_after']*100:.1f}% "
        f"({ppa:+.1f} п.п.); "
        f"оценка «{class_names.get(b,str(b))}»: "
        f"{r['p_rival_before']*100:.1f}% → {r['p_rival_after']*100:.1f}% "
        f"({ppb:+.1f} п.п.). "
        f"Изменение фиксированного отступа A–B: {r['delta_gap']:+.3f}; "
        f"{outcome}; {interpretation}."
    )
