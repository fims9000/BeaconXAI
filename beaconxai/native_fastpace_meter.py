"""Exact model-evaluation instrumentation for real FastPACE CEM inference.

FastPACE consumes [C,T] / [N,C,T] (torch layout), while BEACON's frozen
black box uses [N,T,C]. Every evaluated *input* counts, including resets,
planner rollouts, batched candidates and final validation. No caching and no
assumption that cf_max_plan_calls means black-box evaluations.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Callable

import numpy as np

from .native_cf_budget import QueryBudgetExceeded, TrainingGaussianOutlier


def cf_fingerprint(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError("a counterfactual must be a [C,T] array")
    return sha256(arr.tobytes()).hexdigest()


class MeteredFastPACEModel:
    data_format = "torch"

    def __init__(
        self, predict_many: Callable[[np.ndarray], np.ndarray],
        n_channels: int, ts_length: int, n_classes: int,
        max_evaluations: int | None = None,
    ):
        if n_channels < 1 or ts_length < 1 or n_classes < 2:
            raise ValueError("invalid model shape or classes")
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError("max_evaluations must be positive or None")
        self.underlying = predict_many
        self.n_channels = int(n_channels)
        self.ts_length = int(ts_length)
        self.n_classes = int(n_classes)
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.batches = 0
        self.largest_batch = 0
        self.batch_sizes: list[int] = []
        self.queried: set[str] = set()
        self.target_winner: dict[str, int] = {}

    def to(self, device: str):
        # sklearn-backed BEACON classifier is CPU only, never moved.
        return self

    def _as_batch(self, x):
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        if (arr.ndim != 3 or
            arr.shape[1:] != (self.n_channels, self.ts_length) or
            len(arr) == 0 or not np.isfinite(arr).all()):
            raise ValueError("FastPACE model expects finite [N,C,T] arrays")
        return arr

    def predict(self, inputs):
        arr = self._as_batch(inputs)
        n = len(arr)
        if self.max_evaluations is not None:
            remaining = self.max_evaluations - self.evaluations
            if n > remaining:
                # Hard budget refuses the *entire* native batch, never a
                # convenient subset that would change the upstream optimizer.
                raise QueryBudgetExceeded(n, remaining)
        raw = np.asarray(self.underlying(arr.transpose(0, 2, 1)), dtype=float)
        if raw.shape != (n, self.n_classes) or not np.isfinite(raw).all():
            raise ValueError("black box must return finite [N,classes] scores")
        probs = np.exp(raw - raw.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        self.evaluations += n
        self.batches += 1
        self.batch_sizes.append(n)
        self.largest_batch = max(n, self.largest_batch)
        for orig, p in zip(arr, probs):
            key = cf_fingerprint(orig)
            self.queried.add(key)
            self.target_winner[key] = int(np.argmax(p))
        return probs

    def predict_class(self, inputs):
        return np.argmax(self.predict(inputs), axis=1)

    def was_measured_target(self, candidate, target_class: int) -> bool:
        x = np.asarray(candidate)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.shape != (self.n_channels, self.ts_length):
            return False
        return self.target_winner.get(cf_fingerprint(x)) == target_class


class FastPACEOutlier:
    """Train-only synthetic scorer, NOT the author's pretrained AE.

    The original FastPACE method uses a learned outlier model. Replacing it
    makes this a budget/protocol pilot, not a publication-grade reproduction.
    """

    def __init__(self, train_ntc: np.ndarray):
        train = np.asarray(train_ntc)
        if train.ndim != 3:
            raise ValueError("expected train [N,T,C]")
        self.reference = TrainingGaussianOutlier(train)
        self.n_channels = train.shape[2]
        self.ts_length = train.shape[1]

    def get_outlier_scores(self, inputs):
        arr = np.asarray(inputs, dtype=float)
        if arr.ndim == 2:
            arr = arr[None]
        if (arr.ndim != 3 or
            arr.shape[1:] != (self.n_channels, self.ts_length)):
            raise ValueError("FastPACE outlier scorer expects [N,C,T]")
        return self.reference.get_outlier_scores(arr.transpose(0, 2, 1))
