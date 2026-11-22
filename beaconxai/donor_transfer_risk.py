"""TRAIN-only predictive risk and conservative selective donor transfer gates.

Risk is not a new black-box certificate. It is a prediction of whether an
already measured two-donor certificate survives *two unseen TRAIN donors*.
No third/fourth donor is inspected at inference before a decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "max_modified_fraction",
    "min_measured_target_margin",
    "spread_measured_target_margin",
    "selected_component_fraction",
    "max_source_donor_mse",
    "donor_pair_mse",
    "online_query_fraction",
    "original_target_margin",
    "local_minimality_verified",
)
CANDIDATE_QUANTILES = (.10, .25, .50, .75, 1.)


def donor_transfer_features(
    source: np.ndarray,
    donors: tuple[np.ndarray, np.ndarray],
    mask: tuple[int, ...],
    modified_fractions: tuple[float, float],
    final_target_margins: tuple[float, float],
    original_target_margin: float,
    *,
    components_count: int,
    online_queries: int,
    budget: int = 16,
    minimality_checked: bool = False,
) -> np.ndarray:
    """Nine features available after K2 verification, no D3/D4 access."""
    x = np.asarray(source, dtype=float)
    a, b = (np.asarray(d, dtype=float) for d in donors)
    f = np.asarray(modified_fractions, dtype=float)
    margins = np.asarray(final_target_margins, dtype=float)
    if (x.ndim != 2 or x.shape != a.shape or x.shape != b.shape
        or not np.isfinite(x).all() or not np.isfinite(a).all()
        or not np.isfinite(b).all() or f.shape != (2,)
        or margins.shape != (2,) or not np.isfinite(f).all()
        or not np.isfinite(margins).all()
        or np.any((f < 0) | (f > 1))
        or not np.isfinite(original_target_margin)
        or not 0 < len(mask) <= components_count
        or len(set(mask)) != len(mask)
        or min(mask) < 0 or max(mask) >= components_count
        or not 0 < online_queries <= budget):
        raise ValueError("invalid K2 evidence or input geometry")
    z = np.asarray((
        float(np.max(f)), float(np.min(margins)),
        float(np.ptp(margins)), len(mask) / components_count,
        max(float(np.mean((x-a)**2)), float(np.mean((x-b)**2))),
        float(np.mean((a-b)**2)),
        online_queries / budget, float(original_target_margin),
        float(bool(minimality_checked)),
    ), dtype=float)
    assert z.shape == (9,) and np.isfinite(z).all()
    return z


@dataclass(frozen=True)
class TransferRisk:
    scaler: StandardScaler | None
    model: LogisticRegression | None
    constant: float | None
    fit_n: int
    fit_failure_n: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = np.asarray(x, dtype=float)
        if xs.ndim != 2 or xs.shape[1] != 9 or not np.isfinite(xs).all():
            raise ValueError("risk features must be finite [N,9]")
        if self.model is None:
            return np.full(len(xs), float(self.constant), dtype=float)
        return self.model.predict_proba(self.scaler.transform(xs))[:, 1]


def fit_transfer_risk(features: np.ndarray, failures: np.ndarray) -> TransferRisk:
    x = np.asarray(features, dtype=float)
    y = np.asarray(failures)
    if (x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all()
        or y.shape != (len(x),)
        or not np.isin(y, (0, 1)).all()):
        raise ValueError("expected finite K2 features with binary TRAIN failure labels")
    n = len(x)
    f = int(np.sum(y))
    if n == 0 or len(np.unique(y)) < 2:
        # Laplace-smoothed fallback, intentionally no learned parameters.
        return TransferRisk(None, None, (f + 1) / (n + 2), n, f)
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    model = LogisticRegression(
        C=1., solver="liblinear", random_state=57, max_iter=1000,
    )
    model.fit(z, y.astype(int))
    return TransferRisk(scaler, model, None, n, f)


def upper_clopper_pearson(failures: int, n: int, alpha: float) -> float:
    """One-sided exact binomial upper bound, with no claim of exchangeability."""
    if (not 0 <= failures <= n or not isinstance(failures, int)
        or not isinstance(n, int) or n <= 0 or not 0 < alpha < 1):
        raise ValueError("finite positive binomial sample required")
    if failures == n:
        return 1.
    return float(beta.ppf(1-alpha, failures+1, n-failures))


@dataclass(frozen=True)
class FrozenGate:
    threshold: float | None
    calibration_accepted: int
    calibration_failed: int
    calibration_upper: float | None
    support_threshold: float | None
    fit_thresholds: tuple[float, ...]

    def accepts(self, risk_score: float) -> bool:
        return bool(self.threshold is not None
                    and np.isfinite(risk_score)
                    and risk_score <= self.threshold)

    def accepts_support(self, support: float) -> bool:
        return bool(self.support_threshold is not None
                    and np.isfinite(support)
                    and support <= self.support_threshold)


def calibrate_transfer_gate(
    fit_scores: np.ndarray,
    calibration_scores: np.ndarray,
    calibration_failures: np.ndarray,
    calibration_support: np.ndarray,
    *,
    alpha: float = .1,
    max_failure: float = .5,
    min_accepted: int = 5,
) -> FrozenGate:
    """Choose a threshold using ONLY distinct TRAIN gate-calibration rows.

    Five candidate cutoffs come exclusively from risk-fit score quantiles.
    Bonferroni exact bound controls the exploratory threshold search,
    conditional on exchangeable calibration observations.
    """
    fs = np.asarray(fit_scores, dtype=float)
    s = np.asarray(calibration_scores, dtype=float)
    y = np.asarray(calibration_failures)
    support = np.asarray(calibration_support, dtype=float)
    if (fs.ndim != 1 or s.ndim != 1 or y.shape != s.shape
        or support.shape != s.shape or not np.isfinite(fs).all()
        or not np.isfinite(s).all() or not np.isfinite(support).all()
        or not np.isin(y, (0, 1)).all()
        or np.any((support < 0) | (support > 1))
        or not 0 < alpha < 1 or not 0 < max_failure < 1
        or min_accepted < 1):
        raise ValueError("invalid TRAIN-only risk/calibration inputs")
    if len(fs) == 0:
        return FrozenGate(None, 0, 0, None, None, tuple())
    cutoffs = tuple(float(t) for t in np.quantile(fs, CANDIDATE_QUANTILES))
    selected = None
    for cutoff in sorted(set(cutoffs), reverse=True):
        accepts = s <= cutoff
        count = int(np.sum(accepts))
        if count < min_accepted:
            continue
        failures = int(np.sum(y[accepts]))
        upper = upper_clopper_pearson(failures, count, alpha / len(CANDIDATE_QUANTILES))
        if upper <= max_failure:
            selected = (cutoff, count, failures, upper)
            break
    if selected is None:
        return FrozenGate(None, 0, 0, None, None, cutoffs)
    cutoff, count, failures, upper = selected
    # Control matches (as closely as deterministic ties permit) the TRAIN
    # calibration gate acceptance volume; NO TEST transfer labels used.
    support_threshold = float(np.sort(support)[count-1])
    return FrozenGate(cutoff, count, failures, upper,
                      support_threshold, cutoffs)
