"""Exact online black-box budget accounting for *native* external CF methods.

Adapters only intercept their real model-wrapper calls. No inferred counts
from optimization steps, and a batch of N candidates costs N evaluations.
Returned candidates must have actually been tested or get a fresh evaluation.
This module does NOT change a third-party method's search strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


class QueryBudgetExceeded(RuntimeError):
    def __init__(self, requested: int, remaining: int):
        super().__init__(
            f"native CF requested {requested} model evaluations, "
            f"only {remaining} remain in hard budget"
        )
        self.requested = int(requested)
        self.remaining = int(remaining)


@dataclass(frozen=True)
class QueriedCandidate:
    x: np.ndarray
    scores: np.ndarray
    prediction: int


class CountedCFModel:
    """Emulates upstream .predict([N,T,C]) and .data_format='tf'.

    Only actual *additional* model evaluations count. Original x is supplied
    as previously cached by the caller and is not queried through this API.
    """
    data_format = "tf"

    def __init__(self, predict_many: Callable, shape: tuple[int, int],
                 n_classes: int, budget: int):
        if len(shape) != 2 or min(shape) <= 0 or n_classes < 2 or budget < 1:
            raise ValueError("invalid shape, class count or budget")
        self._model = predict_many
        self._shape = tuple(shape)
        self._classes = int(n_classes)
        self.budget = int(budget)
        self.evaluations = 0
        self.batches = 0
        self.queried: list[QueriedCandidate] = []

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        batch = np.asarray(inputs, dtype=float)
        if batch.ndim == 2:
            batch = batch[None, :, :]
        if batch.ndim != 3 or batch.shape[1:] != self._shape or not np.isfinite(batch).all():
            raise ValueError("native method must query finite [N,T,C] inputs")
        n = len(batch)
        if n > self.remaining:
            # Atomic rejection: no subset is silently evaluated.
            raise QueryBudgetExceeded(n, self.remaining)
        if n == 0:
            raise ValueError("empty model batch")
        raw = np.asarray(self._model(batch), dtype=float)
        if raw.shape != (n, self._classes) or not np.isfinite(raw).all():
            raise ValueError("black box returned invalid [N,classes] scores")
        self.evaluations += n
        self.batches += 1
        # Upstream evolutionary fitness requires normalized probabilities;
        # conversion preserves argmax, but not probability calibration.
        stable = raw - raw.max(axis=1, keepdims=True)
        weights = np.exp(stable)
        scores = weights / weights.sum(axis=1, keepdims=True)
        for x, y in zip(batch, scores):
            self.queried.append(
                QueriedCandidate(x.copy(), y.copy(), int(np.argmax(y)))
            )
        return scores

    def verified_result(self, proposed: np.ndarray, target: int,
                        source: np.ndarray | None = None) -> QueriedCandidate | None:
        """Select smallest actually evaluated, target-valid proposed candidate.

        No additional uncounted black-box call is allowed. If an upstream
        method returns unqueried candidates they must be evaluated via
        .predict() by the caller, using whatever online Q remains.
        """
        if not 0 <= target < self._classes:
            raise ValueError("invalid fixed target class")
        a = np.asarray(proposed, dtype=float)
        if a.ndim == 2:
            a = a[None]
        if a.ndim != 3 or a.shape[1:] != self._shape or not np.isfinite(a).all():
            raise ValueError("invalid native counterfactual shape")
        # Only choose among the actual returned candidates (no cherry-picking
        # from unseen internal populations).
        eligible = [q for q in self.queried
                    if q.prediction == target and
                    any(np.array_equal(q.x, candidate) for candidate in a)]
        if not eligible:
            return None
        center = np.asarray(source, dtype=float) if source is not None else a[0]
        if center.shape != self._shape:
            raise ValueError("source must match [T,C]")
        return min(eligible, key=lambda q: float(np.linalg.norm(q.x-center)))


class TrainingGaussianOutlier:
    """Simple train-only *replacement* for upstream learned anomaly scorer.

    Measures normalized featurewise squared deviation from TRAIN means.
    This is NOT the pretrained outlier model used in the original paper.
    """
    def __init__(self, train_windows: np.ndarray):
        z = np.asarray(train_windows, dtype=float)
        if z.ndim != 3 or not len(z) or not np.isfinite(z).all():
            raise ValueError("finite training windows required")
        self.mean = z.mean(axis=0)
        self.std = np.maximum(z.std(axis=0), 0.1)

    def get_outlier_scores(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs, dtype=float)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[1:] != self.mean.shape:
            raise ValueError("invalid outlier input")
        return np.mean(((x-self.mean)/self.std)**2, axis=(1,2))


class NoFeatureImportance:
    """Native MultiSubSpaCE's no-FI option without eager torch/TF imports."""
    def calculate_feature_importance(self, instance: np.ndarray, **kwargs):
        return np.zeros_like(np.asarray(instance), dtype=float)
