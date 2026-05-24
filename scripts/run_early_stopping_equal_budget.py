#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beaconxai.audit_features import extract_audit_vector
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from scripts.run_component_conflict_benchmark import _train_extratrees_local


PANEL_COLS = [
    "m_neg",
    "M_B_minus",
    "r_B_minus",
    "CE_B",
    "rho_B_cost",
    "frag_drop",
    "top1_delta",
    "top3_sum_delta",
    "top3_conflict_count",
    "margin_entropy",
    "mean_conflict",
    "var_conflict_proxy",
    "frac_conflict_top3",
    "fragility_gap",
    "ce_density",
    "var_conflict",
    "conflict_connectivity",
    "delta_frag_proxy",
    "r_cf",
    "q_fraction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Early stopping BEACON benchmark (equal-budget vs uniform)")
    p.add_argument("--dataset", default="data/uci_har_shifted.npz")
    p.add_argument("--n-total", type=int, default=600)
    p.add_argument("--time-bins", type=int, default=16)
    p.add_argument("--q-max", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--min-q", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--model-cache", default="", help="Optional pickle path for cached base model and standardizer.")
    p.add_argument("--force-train", action="store_true", help="Retrain and overwrite --model-cache if it exists.")
    p.add_argument(
        "--policy-train-mode",
        choices=["full", "prefix_mix"],
        default="full",
        help="Train risk policy on full q_max vectors or a mix of partial-prefix audit vectors.",
    )
    p.add_argument("--policy-prefix-list", default="10,12,16,24,32,64")
    p.add_argument(
        "--order-mode",
        choices=["adaptive", "eta_transport", "risk_importance"],
        default="adaptive",
        help="Component order: adaptive proxy, validation risk-importance prior, or ETA transport prior.",
    )
    p.add_argument(
        "--eta-ref-max",
        type=int,
        default=160,
        help="Maximum train detections used to fit eta_transport priors.",
    )
    p.add_argument(
        "--eta-grid",
        type=int,
        default=32,
        help="Quantile grid size for eta_transport 1D W2 approximation.",
    )
    p.add_argument(
        "--baseline",
        choices=["fixed_uniform", "uniform_early_stop", "both"],
        default="fixed_uniform",
        help="Baseline for BEACON early-stop: fixed equal-budget uniform, uniform with the same early-stop rule, or both.",
    )
    p.add_argument("--out", default="outputs_composite/early_stop_har")
    return p.parse_args()


def _time_slices(t_len: int, n_bins: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, t_len, n_bins + 1, dtype=int)
    out = []
    for i in range(n_bins):
        t0, t1 = int(edges[i]), int(edges[i + 1])
        if t1 <= t0:
            t1 = min(t_len, t0 + 1)
        out.append((t0, t1))
    return out


def _component_idx(ch: int, b: int, n_bins: int) -> int:
    return ch * n_bins + b


def _component_decode(comp: int, n_bins: int) -> tuple[int, int]:
    return comp // n_bins, comp % n_bins


def _margin(logits: np.ndarray) -> tuple[int, float]:
    y = int(np.argmax(logits))
    tmp = logits.copy()
    tmp[y] = -1e18
    return y, float(logits[y] - np.max(tmp))


def _neutralize_component(x: np.ndarray, t0: int, t1: int, c: int) -> np.ndarray:
    y = x.copy()
    if t0 > 0 and t1 < y.shape[0]:
        left = y[t0 - 1, c]
        right = y[t1, c]
        y[t0:t1, c] = np.linspace(left, right, t1 - t0, endpoint=False)
    else:
        y[t0:t1, c] = 0.0
    return y


def _inject_hidden_conflict(x: np.ndarray, donor: np.ndarray, c: int, t0: int, t1: int, alpha: float) -> np.ndarray:
    y = x.copy()
    src = y[t0:t1, c].copy()
    d = donor[t0:t1, c].copy()
    eps = 1e-6
    d = (d - np.mean(d)) / (np.std(d) + eps)
    d = d * (np.std(src) + eps) + np.mean(src)
    mix = (1.0 - alpha) * src + alpha * d
    mix = (mix - np.mean(mix)) / (np.std(mix) + eps)
    mix = mix * (np.std(src) + eps) + np.mean(src)
    y[t0:t1, c] = mix
    return y


def _stratified_split(y: np.ndarray, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(round(n * train_frac)))
        n_va = max(1, int(round(n * val_frac)))
        n_te = max(1, n - n_tr - n_va)
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    tr = np.concatenate(tr)
    va = np.concatenate(va)
    te = np.concatenate(te)
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = float(np.std(x))
    if s <= 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / s


def _adaptive_order(x: np.ndarray, t_slices: list[tuple[int, int]], q_max: int, channel_means: np.ndarray | None = None) -> np.ndarray:
    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    energy = np.zeros(n_components, dtype=np.float64)
    variance = np.zeros(n_components, dtype=np.float64)
    profile_dist = np.zeros(n_components, dtype=np.float64)
    mean_dev = np.zeros(n_components, dtype=np.float64)
    for c in range(n_channels):
        cm = float(channel_means[c]) if channel_means is not None else 0.0
        for bi, (t0, t1) in enumerate(t_slices):
            cid = _component_idx(c, bi, n_bins)
            v = x[t0:t1, c].astype(np.float64)
            energy[cid] = float(np.mean(v * v))
            variance[cid] = float(np.var(v))
            mean_dev[cid] = float(abs(np.mean(v) - cm))
            profile_dist[cid] = float(np.sqrt(np.mean((v - cm) ** 2)))
    score = _z(energy) + _z(variance) + _z(profile_dist) + _z(mean_dev)
    return np.argsort(-score)[: min(q_max, n_components)]


def _w2_quantile_distance(values: np.ndarray, target_q: np.ndarray, grid: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return float("inf")
    q = np.quantile(v, grid)
    diff = q - target_q
    return float(np.sqrt(np.mean(diff * diff)))


def _fit_eta_transport_profile(
    idx_ref: np.ndarray,
    x_det: np.ndarray,
    y_det: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    n_components: int,
    seed: int,
    ref_max: int,
    grid_size: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 700000)
    idx_ref = np.asarray(idx_ref, dtype=np.int64)
    if len(idx_ref) > int(ref_max):
        idx_ref = rng.choice(idx_ref, size=int(ref_max), replace=False)

    delta_mat = np.zeros((len(idx_ref), n_components), dtype=np.float64)
    for row, i in enumerate(idx_ref):
        x = x_det[int(i)]
        _yy, m0 = _margin(clf.logits(x))
        for comp in range(n_components):
            delta_mat[row, comp] = _delta_for_component(x, clf, m0, comp, t_slices)

    y_ref = y_det[idx_ref].astype(int)
    signed_effects = np.tanh(delta_mat)
    risk_vals = signed_effects[y_ref == 1].reshape(-1)
    if len(risk_vals) == 0:
        risk_vals = signed_effects.reshape(-1)
    normal_mask = y_ref == 0
    if not np.any(normal_mask):
        normal_mask = np.ones(len(y_ref), dtype=bool)

    pred_delta = 0.5 * (
        np.median(signed_effects[y_ref == 1], axis=0) if np.any(y_ref == 1) else np.median(signed_effects, axis=0)
    ) + 0.5 * np.median(signed_effects[normal_mask], axis=0)
    separation = np.abs(
        (np.median(signed_effects[y_ref == 1], axis=0) if np.any(y_ref == 1) else np.median(signed_effects, axis=0))
        - np.median(signed_effects[normal_mask], axis=0)
    )
    grid = np.linspace(0.05, 0.95, max(5, int(grid_size)))
    target_q = np.quantile(risk_vals, grid)
    cold_order = np.argsort(-separation)
    return {
        "pred_delta": np.asarray(pred_delta, dtype=np.float64),
        "target_q": np.asarray(target_q, dtype=np.float64),
        "grid": np.asarray(grid, dtype=np.float64),
        "cold_order": np.asarray(cold_order, dtype=np.int64),
    }


def _fit_risk_importance_order(
    idx_ref: np.ndarray,
    x_det: np.ndarray,
    y_det: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    n_components: int,
) -> np.ndarray:
    idx_ref = np.asarray(idx_ref, dtype=np.int64)
    delta_mat = np.zeros((len(idx_ref), n_components), dtype=np.float64)
    for row, i in enumerate(idx_ref):
        x = x_det[int(i)]
        _yy, m0 = _margin(clf.logits(x))
        for comp in range(n_components):
            delta_mat[row, comp] = _delta_for_component(x, clf, m0, comp, t_slices)

    y_ref = y_det[idx_ref].astype(int)
    if len(np.unique(y_ref)) < 2:
        importance = np.var(delta_mat, axis=0)
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=0.25,
                max_iter=3000,
                random_state=0,
            ),
        )
        model.fit(delta_mat, y_ref)
        importance = np.abs(model.named_steps["logisticregression"].coef_[0])
        if not np.any(importance > 0):
            importance = np.var(delta_mat, axis=0)
    return np.asarray(np.argsort(-importance), dtype=np.int64)


def _eta_next_component(observed: list[float], remaining: list[int], eta_profile: dict[str, np.ndarray]) -> int:
    if not observed:
        return int(eta_profile["cold_order"][0])
    pred_delta = eta_profile["pred_delta"]
    target_q = eta_profile["target_q"]
    grid = eta_profile["grid"]
    best_comp = int(remaining[0])
    best_dist = float("inf")
    base = np.asarray(observed, dtype=np.float64)
    for comp in remaining:
        vals = np.append(base, pred_delta[int(comp)])
        dist = _w2_quantile_distance(vals, target_q, grid)
        if dist < best_dist:
            best_dist = dist
            best_comp = int(comp)
    return best_comp


def _delta_for_component(x: np.ndarray, clf, m0: float, comp: int, t_slices: list[tuple[int, int]]) -> float:
    n_bins = len(t_slices)
    c, b = _component_decode(int(comp), n_bins)
    t0, t1 = t_slices[b]
    xm = _neutralize_component(x, t0, t1, c)
    _y1, m1 = _margin(clf.logits(xm))
    return float(m0 - m1)


def _build_vector(
    deltas: np.ndarray,
    margin: float,
    q_max: int,
    sid: int,
    y: int,
    seed: int,
    q_used: int | None = None,
    n_components: int | None = None,
) -> dict[str, float | int | str]:
    vec = extract_audit_vector(
        beacon_result=None,
        margin=float(margin),
        q_max=int(q_max),
        sample_id=int(sid),
        label=int(y),
        is_hidden_conflict=int(y),
        method="early_stop_partial",
        seed=int(seed),
        deltas=deltas,
        rho_b_cost=1.0,
        frag_drop=0.0,
    )
    if q_used is None:
        q_used = q_max
    if n_components is None:
        n_components = len(deltas)
    vec["q_used"] = int(q_used)
    vec["q_fraction"] = float(q_used) / float(max(1, n_components))
    return vec


def _f1_at(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-s)[:k]
    pred = np.zeros(n, dtype=np.int64)
    pred[idx] = 1
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    if tp == 0:
        return 0.0
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return float(2 * p * r / max(p + r, 1e-12))


def _bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d[i] = float(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(min(1.0, 2.0 * min(np.mean(d <= 0), np.mean(d >= 0))))


def _load_or_train_model(args: argparse.Namespace, x_train: np.ndarray, y_train: np.ndarray):
    cache = Path(args.model_cache) if args.model_cache else None
    if cache and cache.exists() and not args.force_train:
        print(f"Loading model from disk: {cache}")
        with cache.open("rb") as f:
            pack = pickle.load(f)
        return pack["clf"], pack["mu"], pack["sigma"], "loaded"

    print("Training base ExtraTrees model...")
    mu, sigma = fit_channel_standardizer(x_train)
    x_train_std = apply_standardizer(x_train, mu, sigma)
    clf = _train_extratrees_local(x_train_std, y_train, n_estimators=300, max_features=0.7, min_samples_leaf=1)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump({"clf": clf, "mu": mu, "sigma": sigma}, f)
        print(f"Saved model cache: {cache}")
    return clf, mu, sigma, "trained"


def _early_stop_scores(
    idx_list: np.ndarray,
    x_det: np.ndarray,
    y_det: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    n_components: int,
    q_max: int,
    cols: list[str],
    policy,
    seed: int,
    tol: float,
    min_q: int,
    order_fn,
    trace_prefix: str,
    eta_profile: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    scores = []
    q_used = []
    traces = []
    for i in idx_list:
        x = x_det[i]
        y = int(y_det[i])
        _yy, m0 = _margin(clf.logits(x))
        order = order_fn(int(i), x) if eta_profile is None else None
        d = np.zeros(n_components, dtype=np.float64)
        prev = None
        hist = []
        used = 0
        observed: list[float] = []
        remaining = list(range(n_components))
        for k in range(1, q_max + 1):
            if eta_profile is None:
                comp = int(order[k - 1])
            else:
                comp = _eta_next_component(observed, remaining, eta_profile)
                remaining.remove(int(comp))
            d[int(comp)] = _delta_for_component(x, clf, m0, int(comp), t_slices)
            observed.append(float(np.tanh(d[int(comp)])))
            vec = _build_vector(
                d,
                margin=m0,
                q_max=q_max,
                sid=int(i),
                y=y,
                seed=seed,
                q_used=k,
                n_components=n_components,
            )
            xx = np.asarray([[vec[c] for c in cols]], dtype=float)
            risk = float(policy.predict_proba(xx)[0, 1])
            hist.append(risk)
            used = k
            if prev is not None and k >= min_q and abs(risk - prev) < tol:
                break
            prev = risk
        scores.append(hist[-1] if hist else 0.5)
        q_used.append(used)
        traces.append(
            {
                "sample_id": int(i),
                "mode": trace_prefix,
                "q_used": int(used),
                "risk_last": float(scores[-1]),
            }
        )
    return np.asarray(scores, dtype=float), np.asarray(q_used, dtype=np.int64), traces


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    x_train_raw, y_train, x_test_raw, y_test = load_npz_dataset(args.dataset)
    clf, mu, sigma, model_source = _load_or_train_model(args, x_train_raw, y_train)
    x_train = apply_standardizer(x_train_raw, mu, sigma)
    x_test = apply_standardizer(x_test_raw, mu, sigma)

    n_channels = x_test.shape[2]
    t_slices = _time_slices(x_test.shape[1], args.time_bins)
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    q_max = min(args.q_max, n_components)

    # Build hidden-conflict package (same spirit as part2).
    target_pos = args.n_total // 2
    target_neg = args.n_total - target_pos
    idx_all = np.arange(len(x_test), dtype=np.int64)
    rng.shuffle(idx_all)

    positives = []
    pos_src = []
    pos_cls = []
    used_src = set()
    for i in idx_all:
        if len(positives) >= target_pos:
            break
        _yy, m0 = _margin(clf.logits(x_test[i]))
        yi = int(y_test[i])
        donor_pool = np.where(y_test != yi)[0]
        if len(donor_pool) == 0:
            continue
        accepted = None
        for _ in range(20):
            c = int(rng.integers(0, n_channels))
            b = int(rng.integers(0, n_bins))
            t0, t1 = t_slices[b]
            d_id = int(donor_pool[int(rng.integers(0, len(donor_pool)))])
            alpha = float(rng.uniform(0.35, 0.65))
            xc = _inject_hidden_conflict(x_test[i], x_test[d_id], c, t0, t1, alpha)
            _y1, m1 = _margin(clf.logits(xc))
            if float(m0 - m1) >= 0.05:
                accepted = xc
                break
        if accepted is None:
            continue
        positives.append(accepted)
        pos_src.append(int(i))
        pos_cls.append(int(yi))
        used_src.add(int(i))

    neg_candidates = [int(i) for i in idx_all if int(i) not in used_src]
    if len(neg_candidates) < target_neg:
        neg_candidates = [int(i) for i in idx_all]
    rng.shuffle(neg_candidates)
    neg_src = neg_candidates[:target_neg]
    negatives = [x_test[i] for i in neg_src]
    neg_cls = [int(y_test[i]) for i in neg_src]

    x_pos = np.asarray(positives, dtype=np.float32)
    x_neg = np.asarray(negatives, dtype=np.float32)
    y_pos = np.ones(len(x_pos), dtype=np.int64)
    y_neg = np.zeros(len(x_neg), dtype=np.int64)

    x_det = np.concatenate([x_pos, x_neg], axis=0)
    y_det = np.concatenate([y_pos, y_neg], axis=0)
    src_cls = np.asarray(pos_cls + neg_cls, dtype=np.int64)
    perm = rng.permutation(len(y_det))
    x_det, y_det, src_cls = x_det[perm], y_det[perm], src_cls[perm]
    tr, va, te = _stratified_split(y_det, 0.6, 0.2, args.seed)

    global_channel_means = np.mean(x_train, axis=(0, 1)).astype(np.float32)

    eta_profile = None
    risk_importance_order = None
    if args.order_mode == "eta_transport":
        print("Fitting ETA transport profile on validation detections...")
        eta_profile = _fit_eta_transport_profile(
            va,
            x_det,
            y_det,
            clf,
            t_slices,
            n_components,
            args.seed,
            args.eta_ref_max,
            args.eta_grid,
        )
    elif args.order_mode == "risk_importance":
        print("Fitting risk-importance order on validation detections...")
        risk_importance_order = _fit_risk_importance_order(
            va,
            x_det,
            y_det,
            clf,
            t_slices,
            n_components,
        )

    prefix_values = [
        int(v.strip())
        for v in args.policy_prefix_list.split(",")
        if v.strip()
    ]
    prefix_values = sorted({min(max(1, q), q_max) for q in prefix_values} | {q_max})
    if args.policy_train_mode == "full":
        prefix_values = [q_max]

    # Train policy on full q_max vectors or mixed partial-prefix vectors.
    rows_full = []
    y_policy = []
    sid_policy = []
    for i in range(len(y_det)):
        x = x_det[i]
        y = int(y_det[i])
        _yy, m0 = _margin(clf.logits(x))
        d = np.zeros(n_components, dtype=np.float64)
        prefix_set = set(prefix_values)
        observed: list[float] = []
        remaining = list(range(n_components))
        if risk_importance_order is not None:
            order = risk_importance_order[:q_max]
        elif eta_profile is None:
            order = _adaptive_order(x, t_slices, q_max, channel_means=global_channel_means)
        else:
            order = None
        for k in range(1, q_max + 1):
            if eta_profile is None:
                comp = int(order[k - 1])
            else:
                comp = _eta_next_component(observed, remaining, eta_profile)
                remaining.remove(int(comp))
            d[int(comp)] = _delta_for_component(x, clf, m0, int(comp), t_slices)
            observed.append(float(np.tanh(d[int(comp)])))
            if k in prefix_set:
                rows_full.append(
                    _build_vector(
                        d.copy(),
                        margin=m0,
                        q_max=q_max,
                        sid=i,
                        y=y,
                        seed=args.seed,
                        q_used=k,
                        n_components=n_components,
                    )
                )
                y_policy.append(y)
                sid_policy.append(i)
    df_full = pd.DataFrame(rows_full)
    cols = [c for c in PANEL_COLS if c in df_full.columns]
    Xf = df_full[cols].to_numpy(dtype=float)
    yp = y_det.astype(int)
    y_policy_arr = np.asarray(y_policy, dtype=int)
    sid_policy_arr = np.asarray(sid_policy, dtype=int)
    train_mask = np.isin(sid_policy_arr, tr)
    policy_beacon = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    policy_beacon.fit(Xf[train_mask], y_policy_arr[train_mask])

    # Early stopping inference on test split.
    if risk_importance_order is None:
        adaptive_order_fn = lambda _i, x: _adaptive_order(x, t_slices, q_max, channel_means=global_channel_means)
    else:
        adaptive_order_fn = lambda _i, _x: risk_importance_order[:q_max]
    s_e, q_used, traces = _early_stop_scores(
        te,
        x_det,
        y_det,
        clf,
        t_slices,
        n_components,
        q_max,
        cols,
        policy_beacon,
        args.seed,
        args.tol,
        args.min_q,
        adaptive_order_fn,
        "beacon_early_stop",
        eta_profile=eta_profile,
    )

    q_mean = int(max(1, round(float(np.mean(q_used)))))

    y_te = yp[te]
    metrics = {
        "auroc_early": float(roc_auc_score(y_te, s_e)),
        "auprc_early": float(average_precision_score(y_te, s_e)),
        "f1_10_early": float(_f1_at(y_te, s_e, 0.10)),
        "q_max": int(q_max),
        "q_mean_early": float(np.mean(q_used)),
        "q_std_early": float(np.std(q_used)),
        "q_equal_uniform": int(q_mean),
        "tol": float(args.tol),
        "min_q": int(args.min_q),
    }

    boot_rows = []
    comparisons: list[tuple[str, np.ndarray]] = []

    if args.baseline in ("fixed_uniform", "both"):
        # Uniform fixed-budget baseline at equal mean Q.
        rows_u_train = []
        rows_u_test = []
        for idx_list, sink in ((tr, rows_u_train), (te, rows_u_test)):
            for i in idx_list:
                x = x_det[i]
                y = int(y_det[i])
                _yy, m0 = _margin(clf.logits(x))
                d = np.zeros(n_components, dtype=np.float64)
                cand_rng = np.random.default_rng(args.seed + 100000 + int(i))
                cand = cand_rng.choice(n_components, size=min(q_mean, n_components), replace=False)
                for comp in cand:
                    d[int(comp)] = _delta_for_component(x, clf, m0, int(comp), t_slices)
                sink.append(
                    _build_vector(
                        d,
                        margin=m0,
                        q_max=q_max,
                        sid=int(i),
                        y=y,
                        seed=args.seed,
                        q_used=min(q_mean, n_components),
                        n_components=n_components,
                    )
                )
        df_ut = pd.DataFrame(rows_u_train)
        df_ute = pd.DataFrame(rows_u_test)
        Xu_tr = df_ut[cols].to_numpy(dtype=float)
        Xu_te = df_ute[cols].to_numpy(dtype=float)
        yu_tr = yp[tr]
        policy_u = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
        policy_u.fit(Xu_tr, yu_tr)
        s_u = policy_u.predict_proba(Xu_te)[:, 1]

        metrics.update(
            {
                "auroc_uniform_eqQ": float(roc_auc_score(y_te, s_u)),
                "auprc_uniform_eqQ": float(average_precision_score(y_te, s_u)),
                "f1_10_uniform_eqQ": float(_f1_at(y_te, s_u, 0.10)),
                "delta_auroc": float(roc_auc_score(y_te, s_e) - roc_auc_score(y_te, s_u)),
                "delta_auprc": float(average_precision_score(y_te, s_e) - average_precision_score(y_te, s_u)),
                "delta_f1_10": float(_f1_at(y_te, s_e, 0.10) - _f1_at(y_te, s_u, 0.10)),
            }
        )
        comparisons.append(("fixed_uniform", np.asarray(s_u, dtype=float)))

    if args.baseline in ("uniform_early_stop", "both"):
        uniform_order_fn = lambda i, _x: np.random.default_rng(args.seed + 200000 + int(i)).permutation(n_components)[:q_max]
        s_ues, q_ues, traces_uniform = _early_stop_scores(
            te,
            x_det,
            y_det,
            clf,
            t_slices,
            n_components,
            q_max,
            cols,
            policy_beacon,
            args.seed,
            args.tol,
            args.min_q,
            uniform_order_fn,
            "uniform_early_stop",
        )
        traces.extend(traces_uniform)
        metrics.update(
            {
                "auroc_uniform_early_stop": float(roc_auc_score(y_te, s_ues)),
                "auprc_uniform_early_stop": float(average_precision_score(y_te, s_ues)),
                "f1_10_uniform_early_stop": float(_f1_at(y_te, s_ues, 0.10)),
                "delta_auroc_vs_uniform_early_stop": float(roc_auc_score(y_te, s_e) - roc_auc_score(y_te, s_ues)),
                "delta_auprc_vs_uniform_early_stop": float(average_precision_score(y_te, s_e) - average_precision_score(y_te, s_ues)),
                "delta_f1_10_vs_uniform_early_stop": float(_f1_at(y_te, s_e, 0.10) - _f1_at(y_te, s_ues, 0.10)),
                "q_mean_uniform_early_stop": float(np.mean(q_ues)),
                "q_std_uniform_early_stop": float(np.std(q_ues)),
            }
        )
        comparisons.append(("uniform_early_stop", np.asarray(s_ues, dtype=float)))

    for comp_name, s_base in comparisons:
        for j, (mname, fn) in enumerate(
        [
            ("delta_auroc", roc_auc_score),
            ("delta_auprc", average_precision_score),
            ("delta_f1_10", lambda y, s: _f1_at(y, s, 0.10)),
        ]
        ):
            d, lo, hi, p = _bootstrap_delta(y_te, s_e, s_base, fn, args.n_boot, args.seed + 100 + j)
            boot_rows.append(
                {
                    "comparison": f"beacon_early_stop_vs_{comp_name}",
                    "metric": mname,
                    "delta": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p,
                }
            )

    pd.DataFrame([metrics]).to_csv(out / "early_stop_vs_uniform_equal_budget.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "early_stop_vs_uniform_equal_budget_bootstrap.csv", index=False)
    pd.DataFrame(traces).to_csv(out / "early_stop_query_trace_test.csv", index=False)
    with (out / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "n_total": int(args.n_total),
                "time_bins": int(args.time_bins),
                "q_max": int(q_max),
                "seed": int(args.seed),
                "tol": float(args.tol),
                "min_q": int(args.min_q),
                "n_boot": int(args.n_boot),
                "baseline": str(args.baseline),
                "model_cache": str(args.model_cache),
                "model_source": str(model_source),
                "policy_train_mode": str(args.policy_train_mode),
                "policy_prefix_list": [int(v) for v in prefix_values],
                "order_mode": str(args.order_mode),
                "eta_ref_max": int(args.eta_ref_max),
                "eta_grid": int(args.eta_grid),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved: {out / 'early_stop_vs_uniform_equal_budget.csv'}")
    print(f"saved: {out / 'early_stop_vs_uniform_equal_budget_bootstrap.csv'}")
    print(f"saved: {out / 'early_stop_query_trace_test.csv'}")


if __name__ == "__main__":
    main()
