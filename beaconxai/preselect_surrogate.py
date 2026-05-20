from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "energy",
    "variance",
    "profile_distance",
    "mean_deviation",
    "runner_distance",
    "class_separation",
    "time_index",
    "channel_index",
)


@dataclass
class SurrogatePack:
    model: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES


def component_features(
    x: np.ndarray,
    t_slices: list[tuple[int, int]],
    mu_yhat: np.ndarray,
    mu_runner: np.ndarray,
) -> np.ndarray:
    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    feats = np.zeros((n_components, len(FEATURE_NAMES)), dtype=np.float64)
    for c in range(n_channels):
        sep_c = float(abs(mu_yhat[c] - mu_runner[c]))
        for bi, (t0, t1) in enumerate(t_slices):
            cid = c * n_bins + bi
            v = x[t0:t1, c].astype(np.float64)
            energy = float(np.mean(v * v))
            var = float(np.var(v))
            mu = float(np.mean(v))
            dy = v - float(mu_yhat[c])
            dr = v - float(mu_runner[c])
            dist_y = float(np.sqrt(np.mean(dy * dy)))
            dist_r = float(np.sqrt(np.mean(dr * dr)))
            feats[cid] = (
                energy,
                var,
                dist_y,
                abs(mu - float(mu_yhat[c])),
                dist_r,
                sep_c,
                float((bi + 0.5) / max(1, n_bins)),
                float((c + 0.5) / max(1, n_channels)),
            )
    return feats


def full_deltas(
    x: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    neutralize_fn,
    neutralizer_mode: str,
    channel_means: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    from scripts.run_part2_extended import _component_decode, _margin

    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    lg0 = clf.logits(x)
    _y0, m0 = _margin(lg0)
    d = np.zeros(n_components, dtype=np.float64)
    for comp in range(n_components):
        c, b = _component_decode(int(comp), n_bins)
        t0, t1 = t_slices[b]
        xm = neutralize_fn(x, t0, t1, c, neutralizer_mode, channel_means=channel_means)
        _y1, m1 = _margin(clf.logits(xm))
        d[comp] = float(m0 - m1)
    return d, float(m0)


def selected_deltas(
    x: np.ndarray,
    selected: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    neutralize_fn,
    neutralizer_mode: str,
    channel_means: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    from scripts.run_part2_extended import _component_decode, _margin

    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    lg0 = clf.logits(x)
    _y0, m0 = _margin(lg0)
    d = np.zeros(n_components, dtype=np.float64)
    for comp in np.asarray(selected, dtype=np.int64):
        c, b = _component_decode(int(comp), n_bins)
        t0, t1 = t_slices[b]
        xm = neutralize_fn(x, t0, t1, c, neutralizer_mode, channel_means=channel_means)
        _y1, m1 = _margin(clf.logits(xm))
        d[int(comp)] = float(m0 - m1)
    return d, float(m0)


def preselect_by_surrogate(
    feats: np.ndarray,
    surrogate_model: Any,
    q_max: int,
    positive_only: bool = False,
) -> np.ndarray:
    pred = np.asarray(surrogate_model.predict(feats), dtype=np.float64).reshape(-1)
    # Surrogate predicts conflict score (larger => more conflictive component).
    order = np.argsort(-pred)
    q = min(int(q_max), len(order))
    if not positive_only:
        return np.asarray(order[:q], dtype=np.int64)
    pos = order[pred[order] > 0.0]
    if len(pos) >= q:
        return np.asarray(pos[:q], dtype=np.int64)
    rem = [int(i) for i in order if int(i) not in set(pos.tolist())]
    take = list(pos.tolist()) + rem[: max(0, q - len(pos))]
    return np.asarray(take[:q], dtype=np.int64)
