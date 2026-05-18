from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import KBinsDiscretizer


EPS = 1e-8


def build_fuzzy_inputs_v2(df):
    # Keep inputs compact and interpretable (3 signals).
    x_margin = df["m_neg"].to_numpy(dtype=float)
    x_conflict = df["r_B_minus"].to_numpy(dtype=float)
    x_frag = df["frag_drop"].to_numpy(dtype=float)
    return np.stack([x_margin, x_conflict, x_frag], axis=1)


def _trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if b > a:
        m = (x >= a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    m = (x >= b) & (x <= c)
    y[m] = 1.0
    if d > c:
        m = (x > c) & (x <= d)
        y[m] = (d - x[m]) / (d - c)
    return np.clip(y, 0.0, 1.0)


def _trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if b > a:
        m = (x >= a) & (x <= b)
        y[m] = (x[m] - a) / (b - a)
    if c > b:
        m = (x >= b) & (x <= c)
        y[m] = (c - x[m]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def _kmeans_membership_params(x_train: np.ndarray) -> tuple[float, float, float, float, float]:
    # As requested: kmeans-style binning via KBinsDiscretizer.
    disc = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="kmeans")
    disc.fit(x_train.reshape(-1, 1))
    edges = disc.bin_edges_[0]
    lo = float(edges[0])
    hi = float(edges[-1])
    span = max(hi - lo, 1e-6)
    lo -= 0.05 * span
    hi += 0.05 * span

    # Use bin centers inferred from train values per learned bin.
    bins = disc.transform(x_train.reshape(-1, 1)).reshape(-1).astype(int)
    centers = []
    for bi in range(3):
        vals = x_train[bins == bi]
        if len(vals) == 0:
            centers.append(float(np.mean(x_train)))
        else:
            centers.append(float(np.mean(vals)))
    centers = np.sort(np.asarray(centers, dtype=float))
    c0, c1, c2 = float(centers[0]), float(centers[1]), float(centers[2])

    # Ensure strict ordering for stable triangles.
    if c1 <= c0:
        c1 = c0 + 1e-6
    if c2 <= c1:
        c2 = c1 + 1e-6
    return lo, c0, c1, c2, hi


def _memberships_3(x: np.ndarray, params: tuple[float, float, float, float, float]) -> np.ndarray:
    lo, c0, c1, c2, hi = params
    m_low = _trapmf(x, lo, lo, c0, c1)
    m_med = _trimf(x, c0, c1, c2)
    m_high = _trapmf(x, c1, c2, hi, hi)
    return np.stack([m_low, m_med, m_high], axis=1)


def _rule_base_outputs() -> np.ndarray:
    # 27 rules for 3x3x3 with monotonic risk mapping by level sum.
    z = np.zeros(27, dtype=float)
    idx = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                z[idx] = float((a + b + c) / 6.0)
                idx += 1
    return z


def _rule_activations(m1: np.ndarray, m2: np.ndarray, m3: np.ndarray) -> np.ndarray:
    # m*: [N,3] -> activations [N,27]
    n = m1.shape[0]
    acts = np.zeros((n, 27), dtype=float)
    idx = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                acts[:, idx] = m1[:, a] * m2[:, b] * m3[:, c]
                idx += 1
    return acts


def _weighted_sugeno_score(acts: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    ww = np.clip(w.reshape(1, -1), 0.1, 10.0)
    num = np.sum(acts * ww * z.reshape(1, -1), axis=1)
    den = np.sum(acts * ww, axis=1)
    return num / np.maximum(den, EPS)


def _cross_entropy(y: np.ndarray, s: np.ndarray) -> float:
    p = np.clip(s, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class FuzzyV2Policy:
    params_margin: tuple[float, float, float, float, float]
    params_conflict: tuple[float, float, float, float, float]
    params_fragility: tuple[float, float, float, float, float]
    rule_outputs: np.ndarray
    rule_weights: np.ndarray


def fit_fuzzy_policy_v2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    reg: float = 1e-3,
    seed: int = 42,
) -> FuzzyV2Policy:
    rng = np.random.default_rng(seed)

    pm = _kmeans_membership_params(X_train[:, 0])
    pc = _kmeans_membership_params(X_train[:, 1])
    pf = _kmeans_membership_params(X_train[:, 2])

    m1 = _memberships_3(X_train[:, 0], pm)
    m2 = _memberships_3(X_train[:, 1], pc)
    m3 = _memberships_3(X_train[:, 2], pf)
    acts = _rule_activations(m1, m2, m3)
    z = _rule_base_outputs()

    def obj(w: np.ndarray) -> float:
        s = _weighted_sugeno_score(acts, z, w)
        ce = _cross_entropy(y_train.astype(float), s)
        l2 = reg * float(np.mean((w - 1.0) ** 2))
        return ce + l2

    best_w = np.ones(27, dtype=float)
    best_f = obj(best_w)
    bounds = [(0.1, 10.0)] * 27

    starts = [best_w]
    for _ in range(3):
        starts.append(np.clip(rng.lognormal(mean=0.0, sigma=0.35, size=27), 0.1, 10.0))

    for w0 in starts:
        res = minimize(obj, w0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        w = np.clip(res.x, 0.1, 10.0)
        f = obj(w)
        if f < best_f:
            best_f = f
            best_w = w

    return FuzzyV2Policy(
        params_margin=pm,
        params_conflict=pc,
        params_fragility=pf,
        rule_outputs=z,
        rule_weights=best_w,
    )


def predict_fuzzy_policy_v2(policy: FuzzyV2Policy, X: np.ndarray) -> np.ndarray:
    m1 = _memberships_3(X[:, 0], policy.params_margin)
    m2 = _memberships_3(X[:, 1], policy.params_conflict)
    m3 = _memberships_3(X[:, 2], policy.params_fragility)
    acts = _rule_activations(m1, m2, m3)
    return _weighted_sugeno_score(acts, policy.rule_outputs, policy.rule_weights)


def gate_score(fuzzy: np.ndarray, backend: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    out = backend.copy()
    out[fuzzy <= tau_low] = 0.0
    out[fuzzy >= tau_high] = 1.0
    return out


def eval_at_budget(y: np.ndarray, score: np.ndarray, frac: float):
    k = max(1, int(np.ceil(frac * len(y))))
    idx = np.argsort(-score)[:k]
    tp = int(np.sum(y[idx] == 1))
    fp = k - tp
    fn = int(np.sum(y == 1)) - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return float(p), float(r), float(f1)
