from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    p = np.clip(p, 1e-8, 1 - 1e-8)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        ece += (np.sum(m) / max(n, 1)) * abs(acc - conf)
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def calibration_curve_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        n_i = int(np.sum(m))
        if n_i == 0:
            rows.append(
                {
                    "bin": i,
                    "prob_low": float(lo),
                    "prob_high": float(hi),
                    "count": 0,
                    "mean_prob": float("nan"),
                    "empirical_pos_rate": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "bin": i,
                "prob_low": float(lo),
                "prob_high": float(hi),
                "count": n_i,
                "mean_prob": float(np.mean(p[m])),
                "empirical_pos_rate": float(np.mean(y[m])),
            }
        )
    return rows


def calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1.0 - p))
    x = np.column_stack([np.ones_like(logit), logit])
    # least squares proxy for logistic calibration slope
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(beta[1])
