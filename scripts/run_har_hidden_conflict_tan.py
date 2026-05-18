#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.core import BeaconAudit
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig
from scripts.run_component_conflict_benchmark import _train_extratrees_local, _train_histgbt_local


EPS = 1e-8


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


def _neighbors(comp: int, n_channels: int, n_bins: int) -> list[int]:
    c, b = _component_decode(comp, n_bins)
    out: list[int] = []
    if b > 0:
        out.append(_component_idx(c, b - 1, n_bins))
    if b + 1 < n_bins:
        out.append(_component_idx(c, b + 1, n_bins))
    if c > 0:
        out.append(_component_idx(c - 1, b, n_bins))
    if c + 1 < n_channels:
        out.append(_component_idx(c + 1, b, n_bins))
    return out


def _margin(logits: np.ndarray) -> tuple[int, float]:
    y = int(np.argmax(logits))
    tmp = logits.copy()
    tmp[y] = -1e18
    return y, float(logits[y] - np.max(tmp))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _build_component_refs(x_train: np.ndarray, t_slices: list[tuple[int, int]]) -> np.ndarray:
    n_channels = x_train.shape[2]
    n_bins = len(t_slices)
    ref_var = np.zeros((n_channels, n_bins), dtype=np.float64)
    n = float(max(1, x_train.shape[0]))
    for xi in x_train:
        for b, (t0, t1) in enumerate(t_slices):
            blk = xi[t0:t1, :]
            for c in range(n_channels):
                ref_var[c, b] += float(np.var(blk[:, c]))
    ref_var /= n
    return ref_var


def _corr_prefilter_scores(x: np.ndarray, t_slices: list[tuple[int, int]]) -> np.ndarray:
    _t_len, n_channels = x.shape
    n_bins = len(t_slices)
    out = np.zeros(n_channels * n_bins, dtype=np.float64)
    for bi, (t0, t1) in enumerate(t_slices):
        block = x[t0:t1, :]
        if block.shape[0] < 2:
            continue
        for c in range(n_channels):
            vc = block[:, c]
            vals = []
            for cc in range(n_channels):
                if cc == c:
                    continue
                vo = block[:, cc]
                vals.append(_safe_corr(vc, vo))
            mean_abs_corr = float(np.mean(np.abs(vals))) if vals else 0.0
            out[_component_idx(c, bi, n_bins)] = 1.0 - mean_abs_corr
    return out


def _neutralize_component(x: np.ndarray, t0: int, t1: int, c: int, mode: str) -> np.ndarray:
    y = x.copy()
    if mode in ("zero", "mean"):
        y[t0:t1, c : c + 1] = 0.0
        return y
    if t0 > 0 and t1 < y.shape[0]:
        left = y[t0 - 1, c]
        right = y[t1, c]
        y[t0:t1, c] = np.linspace(left, right, t1 - t0, endpoint=False)
    else:
        y[t0:t1, c] = 0.0
    return y


def _inject_hidden_conflict(
    x: np.ndarray,
    donor: np.ndarray,
    c: int,
    t0: int,
    t1: int,
    alpha: float,
) -> np.ndarray:
    y = x.copy()
    src = y[t0:t1, c].copy()
    d = donor[t0:t1, c].copy()
    d = (d - np.mean(d)) / (np.std(d) + 1e-6)
    d = d * (np.std(src) + 1e-6) + np.mean(src)
    mix = (1.0 - alpha) * src + alpha * d
    mix = (mix - np.mean(mix)) / (np.std(mix) + 1e-6)
    mix = mix * (np.std(src) + 1e-6) + np.mean(src)
    y[t0:t1, c] = mix
    return y


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    if len(score) == 0:
        return 0.0
    qs = np.linspace(0.05, 0.95, 91)
    thrs = np.quantile(score, qs)
    best_f1 = -1.0
    best_t = float(np.median(score))
    for t in thrs:
        pred = (score >= float(t)).astype(np.int64)
        _p, _r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


FEATURE_SET_MAP = {
    "conflict": [1, 2, 3],  # counter_mass, r_minus, ce_b
    "conflict_margin": [0, 1, 2, 3],  # +m_neg
    "conflict_fragility": [1, 2, 3, 4, 5],  # +frag/rho
    "full_compact": [0, 1, 2, 3, 4, 5],  # 6-d compact BEACON panel
}


def _parse_grid(values: str, cast_fn) -> list:
    out = []
    for v in values.split(","):
        v = v.strip()
        if not v:
            continue
        out.append(cast_fn(v))
    return out


def _bootstrap_auc_delta(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 4000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = _safe_auc(y, a) - _safe_auc(y, b)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = _safe_auc(y[idx], a[idx]) - _safe_auc(y[idx], b[idx])
        if np.isfinite(d):
            vals.append(float(d))
    if not vals:
        return float(obs), float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    return float(obs), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(p)


@dataclass
class TANModel:
    n_bins: int = 4
    alpha: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TANModel":
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("TANModel expects binary labels")
        self.class_to_idx_ = {int(c): i for i, c in enumerate(self.classes_)}

        n, d = X.shape
        self.d_ = d
        self.edges_ = []
        for j in range(d):
            v = X[:, j]
            qs = np.linspace(0.0, 1.0, self.n_bins + 1)
            edges = np.quantile(v, qs)
            edges[0] = -np.inf
            edges[-1] = np.inf
            for k in range(1, len(edges)):
                if not edges[k] > edges[k - 1]:
                    edges[k] = edges[k - 1] + 1e-9
            self.edges_.append(edges)

        Xd = self._discretize(X)
        c_idx = np.array([self.class_to_idx_[int(v)] for v in y], dtype=np.int64)

        w = np.zeros((d, d), dtype=np.float64)
        for i in range(d):
            for j in range(i + 1, d):
                v = self._conditional_mi(Xd[:, i], Xd[:, j], c_idx)
                w[i, j] = w[j, i] = v

        self.parent_ = self._max_spanning_tree_parents(w)

        n_classes = len(self.classes_)
        self.log_prior_ = np.zeros(n_classes, dtype=np.float64)
        for c in range(n_classes):
            nc = int(np.sum(c_idx == c))
            self.log_prior_[c] = np.log((nc + self.alpha) / (n + self.alpha * n_classes))

        self.root_tables_ = {}
        self.cond_tables_ = {}
        for i in range(d):
            p = self.parent_[i]
            if p < 0:
                tab = np.zeros((n_classes, self.n_bins), dtype=np.float64)
                for c in range(n_classes):
                    mask = c_idx == c
                    cnt = np.bincount(Xd[mask, i], minlength=self.n_bins).astype(np.float64)
                    tab[c] = np.log((cnt + self.alpha) / (np.sum(cnt) + self.alpha * self.n_bins))
                self.root_tables_[i] = tab
            else:
                tab = np.zeros((n_classes, self.n_bins, self.n_bins), dtype=np.float64)
                for c in range(n_classes):
                    mask_c = c_idx == c
                    for pv in range(self.n_bins):
                        mask = mask_c & (Xd[:, p] == pv)
                        cnt = np.bincount(Xd[mask, i], minlength=self.n_bins).astype(np.float64)
                        tab[c, pv] = np.log((cnt + self.alpha) / (np.sum(cnt) + self.alpha * self.n_bins))
                self.cond_tables_[i] = tab
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xd = self._discretize(X)
        n = Xd.shape[0]
        n_classes = len(self.classes_)
        out = np.zeros((n, n_classes), dtype=np.float64)

        for r in range(n):
            logp = self.log_prior_.copy()
            for i in range(self.d_):
                p = self.parent_[i]
                xi = int(Xd[r, i])
                if p < 0:
                    logp += self.root_tables_[i][:, xi]
                else:
                    xp = int(Xd[r, p])
                    logp += self.cond_tables_[i][:, xp, xi]
            m = float(np.max(logp))
            pr = np.exp(logp - m)
            pr = pr / np.sum(pr)
            out[r] = pr
        return out

    def _discretize(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        out = np.zeros((n, d), dtype=np.int64)
        for j in range(d):
            e = self.edges_[j]
            out[:, j] = np.clip(np.digitize(X[:, j], e[1:-1], right=False), 0, self.n_bins - 1)
        return out

    def _conditional_mi(self, xi: np.ndarray, xj: np.ndarray, yc: np.ndarray) -> float:
        n = len(xi)
        k = self.n_bins
        cvals = np.unique(yc)
        out = 0.0
        for c in cvals:
            mask = yc == c
            nc = int(np.sum(mask))
            if nc == 0:
                continue
            p_y = nc / n
            cnt_ij = np.zeros((k, k), dtype=np.float64)
            cnt_i = np.zeros(k, dtype=np.float64)
            cnt_j = np.zeros(k, dtype=np.float64)
            for a, b in zip(xi[mask], xj[mask]):
                cnt_ij[int(a), int(b)] += 1.0
                cnt_i[int(a)] += 1.0
                cnt_j[int(b)] += 1.0
            p_ij = (cnt_ij + self.alpha) / (nc + self.alpha * k * k)
            p_i = (cnt_i + self.alpha) / (nc + self.alpha * k)
            p_j = (cnt_j + self.alpha) / (nc + self.alpha * k)
            ratio = p_ij / np.maximum(p_i[:, None] * p_j[None, :], 1e-15)
            out += p_y * float(np.sum(p_ij * np.log(np.maximum(ratio, 1e-15))))
        return out

    def _max_spanning_tree_parents(self, w: np.ndarray) -> list[int]:
        d = w.shape[0]
        parent = [-1 for _ in range(d)]
        selected = np.zeros(d, dtype=bool)
        key = np.full(d, -np.inf, dtype=np.float64)
        key[0] = 0.0

        for _ in range(d):
            cand = np.where(~selected)[0]
            if len(cand) == 0:
                break
            u = int(cand[np.argmax(key[cand])])
            selected[u] = True
            for v in range(d):
                if selected[v] or v == u:
                    continue
                if w[u, v] > key[v]:
                    key[v] = w[u, v]
                    parent[v] = u
        return parent


def _compute_uniform_panel_features(
    x: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    q: int,
    neutralizer_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins

    lg0 = clf.logits(x)
    _y0, m0 = _margin(lg0)

    budget = min(int(q), n_components)
    cand = rng.choice(n_components, size=budget, replace=False)

    deltas = []
    single_margins = []
    for comp in cand:
        c, b = _component_decode(int(comp), n_bins)
        t0, t1 = t_slices[b]
        xm = _neutralize_component(x, t0, t1, c, neutralizer_mode)
        _y1, m1 = _margin(clf.logits(xm))
        d = float(m0 - m1)
        deltas.append((int(comp), d))
        single_margins.append(float(m1))

    if not deltas:
        return np.zeros(6, dtype=np.float64)

    dvals = np.array([d for _c, d in deltas], dtype=np.float64)
    support_mass = float(np.sum(np.maximum(dvals, 0.0)))
    counter_mass = float(np.sum(np.maximum(-dvals, 0.0)))
    r_minus = float(counter_mass / max(counter_mass + support_mass, EPS))
    ce_b = float(counter_mass / max(len(dvals), 1))

    # Fragility and rho_cost via cumulative support-removal over sampled components.
    pos_order = [c for c, d in sorted(deltas, key=lambda z: z[1], reverse=True) if d > 0.0]
    x_cur = x.copy()
    m_last = float(m0)
    k_flip = 0
    for k, comp in enumerate(pos_order, start=1):
        cc, bb = _component_decode(int(comp), n_bins)
        tt0, tt1 = t_slices[bb]
        x_cur = _neutralize_component(x_cur, tt0, tt1, cc, neutralizer_mode)
        _yc, mc = _margin(clf.logits(x_cur))
        m_last = float(mc)
        if mc <= 0.0:
            k_flip = k
            break

    frag_drop = float(max(0.0, m0 - m_last))
    if k_flip > 0:
        rho_cost = float(k_flip / max(n_components, 1))
    else:
        rho_cost = 1.0

    m_neg = float(-m0)
    return np.array([m_neg, counter_mass, r_minus, ce_b, frag_drop, rho_cost], dtype=np.float64)


def _compute_adaptive_max_effect(
    x: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    q: int,
    neutralizer_mode: str,
    phase1_ratio: float,
    early_eff: float,
) -> float:
    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins

    prior = _corr_prefilter_scores(x, t_slices)
    order = np.argsort(-prior)

    lg0 = clf.logits(x)
    _y0, m0 = _margin(lg0)
    q_total = int(max(1, q))
    q1 = min(max(1, int(round(phase1_ratio * q_total))), q_total)

    seen: set[int] = set()
    calls = 0
    best_eff = -1.0
    seed_comp = int(order[0])

    # Phase 1
    for comp in order[:q1]:
        cc, bb = _component_decode(int(comp), n_bins)
        tt0, tt1 = t_slices[bb]
        xm = _neutralize_component(x, tt0, tt1, cc, neutralizer_mode)
        _y1, m1 = _margin(clf.logits(xm))
        eff = float(abs(m1 - m0))
        if eff > best_eff:
            best_eff = eff
            seed_comp = int(comp)
        seen.add(int(comp))
        calls += 1

    # Neighborhood expansion on strong signal.
    if best_eff >= float(early_eff) and calls < q_total:
        nbs = _neighbors(seed_comp, n_channels, n_bins)
        nbs = sorted(nbs, key=lambda z: float(prior[z]), reverse=True)
        for nb in nbs:
            if calls >= q_total:
                break
            if nb in seen:
                continue
            cc, bb = _component_decode(int(nb), n_bins)
            tt0, tt1 = t_slices[bb]
            xm = _neutralize_component(x, tt0, tt1, cc, neutralizer_mode)
            _y1, m1 = _margin(clf.logits(xm))
            eff = float(abs(m1 - m0))
            if eff > best_eff:
                best_eff = eff
            seen.add(int(nb))
            calls += 1

    # Coverage fallback.
    for comp in order:
        if calls >= q_total:
            break
        cc = int(comp)
        if cc in seen:
            continue
        c, b = _component_decode(cc, n_bins)
        tt0, tt1 = t_slices[b]
        xm = _neutralize_component(x, tt0, tt1, c, neutralizer_mode)
        _y1, m1 = _margin(clf.logits(xm))
        eff = float(abs(m1 - m0))
        if eff > best_eff:
            best_eff = eff
        seen.add(cc)
        calls += 1

    return float(max(best_eff, 0.0))


def _build_detection_dataset(
    x_test: np.ndarray,
    y_test: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    n_pos_target: int,
    n_neg_target: int,
    seed: int,
    hidden_margin_drop_min: float,
    hidden_alpha_min: float,
    hidden_alpha_max: float,
    hidden_max_tries: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_channels = x_test.shape[2]
    n_bins = len(t_slices)

    all_idx = np.arange(len(x_test), dtype=np.int64)
    rng.shuffle(all_idx)

    positives = []
    used_idx = set()
    for i in all_idx:
        if len(positives) >= n_pos_target:
            break
        lg0 = clf.logits(x_test[i])
        _y0, m0 = _margin(lg0)
        yi = int(y_test[i])
        donor_pool = np.where(y_test != yi)[0]
        if len(donor_pool) == 0:
            continue

        accepted = None
        for _ in range(max(1, hidden_max_tries)):
            c = int(rng.integers(0, n_channels))
            b = int(rng.integers(0, n_bins))
            t0, t1 = t_slices[b]
            d_id = int(donor_pool[int(rng.integers(0, len(donor_pool)))])
            alpha = float(rng.uniform(hidden_alpha_min, hidden_alpha_max))
            xc = _inject_hidden_conflict(x_test[i], x_test[d_id], c, t0, t1, alpha)
            _y1, m1 = _margin(clf.logits(xc))
            drop = float(m0 - m1)
            if drop >= hidden_margin_drop_min:
                accepted = xc
                break
        if accepted is None:
            continue
        positives.append(accepted)
        used_idx.add(int(i))

    neg_candidates = [int(i) for i in all_idx if int(i) not in used_idx]
    if len(neg_candidates) < n_neg_target:
        neg_candidates = [int(i) for i in all_idx]

    rng.shuffle(neg_candidates)
    neg_idx = neg_candidates[:n_neg_target]
    negatives = [x_test[i] for i in neg_idx]

    x_pos = np.asarray(positives, dtype=np.float32)
    x_neg = np.asarray(negatives, dtype=np.float32)
    y_pos = np.ones(len(x_pos), dtype=np.int64)
    y_neg = np.zeros(len(x_neg), dtype=np.int64)

    if len(x_pos) == 0 or len(x_neg) == 0:
        raise RuntimeError("Could not build balanced hidden-conflict detection dataset")

    x_all = np.concatenate([x_pos, x_neg], axis=0)
    y_all = np.concatenate([y_pos, y_neg], axis=0)
    perm = rng.permutation(len(y_all))
    return x_all[perm], y_all[perm]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR hidden-conflict detection with TAN")
    p.add_argument("--npz-path", default="data/uci_har_shifted.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=2500)
    p.add_argument("--n-positive", type=int, default=1000)
    p.add_argument("--n-negative", type=int, default=1000)
    p.add_argument("--time-bins", type=int, default=8)
    p.add_argument("--q", type=int, default=16)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--tan-bins", type=int, default=4)
    p.add_argument("--tan-alpha", type=float, default=1.0)
    p.add_argument("--tan-bins-grid", default="")
    p.add_argument("--tan-alpha-grid", default="")
    p.add_argument(
        "--feature-sets",
        default="full_compact",
        help="Comma-separated: conflict,conflict_margin,conflict_fragility,full_compact",
    )
    p.add_argument("--model", choices=["cnn1d", "extratrees", "histgbt"], default="cnn1d")
    p.add_argument("--neutralizer", choices=["zero", "mean", "interp"], default="interp")
    p.add_argument("--cnn-epochs", type=int, default=12)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--hidden-margin-drop-min", type=float, default=0.05)
    p.add_argument("--hidden-alpha-min", type=float, default=0.35)
    p.add_argument("--hidden-alpha-max", type=float, default=0.65)
    p.add_argument("--hidden-max-tries", type=int, default=20)
    p.add_argument("--adaptive-phase1-ratio", type=float, default=0.4)
    p.add_argument("--adaptive-early-eff", type=float, default=0.12)
    p.add_argument("--out-summary", default="outputs_composite/har_hidden_conflict_detection_tan_table.csv")
    p.add_argument("--out-per-sample", default="outputs_composite/har_hidden_conflict_detection_tan_per_sample.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)
    if args.max_test > 0 and args.max_test < len(x_test):
        idx = rng.choice(len(x_test), size=args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "extratrees":
        clf = _train_extratrees_local(x_train, y_train, n_estimators=300, max_features=0.7, min_samples_leaf=1)
    elif args.model == "histgbt":
        clf = _train_histgbt_local(x_train, y_train)
    else:
        from beaconxai.models import train_1dcnn

        clf = train_1dcnn(
            x_train,
            y_train,
            epochs=args.cnn_epochs,
            batch_size=args.cnn_batch_size,
            lr=args.cnn_lr,
            label_smoothing=0.0,
            use_class_weights=True,
            tta_shifts=(0,),
        )

    t_len = x_test.shape[1]
    n_channels = x_test.shape[2]
    t_slices = _time_slices(t_len, args.time_bins)
    n_bins = len(t_slices)

    x_det, y_det = _build_detection_dataset(
        x_test=x_test,
        y_test=y_test,
        clf=clf,
        t_slices=t_slices,
        n_pos_target=args.n_positive,
        n_neg_target=args.n_negative,
        seed=args.seed + 101,
        hidden_margin_drop_min=args.hidden_margin_drop_min,
        hidden_alpha_min=args.hidden_alpha_min,
        hidden_alpha_max=args.hidden_alpha_max,
        hidden_max_tries=args.hidden_max_tries,
    )

    ref_var = _build_component_refs(x_train, t_slices)

    cfg = BeaconConfig(
        q_max=args.q,
        k0=8 if args.q >= 16 else 4,
        l_min=4,
        k_pos=3,
        k_neg=3,
        partition_mode="sensor_group_time",
        refinement_mode="mixed",
        margin_mode="adaptive_all",
        risk_policy="rho_only",
        audit_mode="full",
    )
    neutralizer = Neutralizer(mode=args.neutralizer, channel_means=np.zeros(n_channels, dtype=np.float32))
    audit = BeaconAudit(model_logits=clf.logits, neutralizer=neutralizer, config=cfg)

    n = len(y_det)
    feat_beacon = np.zeros((n, 6), dtype=np.float64)
    feat_uniform = np.zeros((n, 6), dtype=np.float64)
    score_variance = np.zeros(n, dtype=np.float64)
    score_adaptive = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x = x_det[i]

        # Variance heuristic score (zero-query baseline).
        vals = []
        for c in range(n_channels):
            for b, (t0, t1) in enumerate(t_slices):
                v = x[t0:t1, c]
                vals.append(abs(float(np.var(v)) - float(ref_var[c, b])))
        score_variance[i] = float(np.max(vals)) if vals else 0.0

        # Uniform occlusion panel-like features.
        feat_uniform[i] = _compute_uniform_panel_features(
            x=x,
            clf=clf,
            t_slices=t_slices,
            q=args.q,
            neutralizer_mode=args.neutralizer,
            rng=np.random.default_rng(args.seed + 10000 + i),
        )

        # BEACON-core panel features.
        lg0 = clf.logits(x)
        _y0, m0 = _margin(lg0)
        ar = audit.audit(x)
        m_neg = float(-m0)
        counter_mass = float(ar.counter_mass)
        support_mass = float(ar.support_mass)
        r_minus = float(counter_mass / max(counter_mass + support_mass, EPS))
        ce_b = float(ar.counter_evidence_gain)
        frag_drop = float(max(0.0, ar.m0 - ar.m_last))
        rho_cost = float(ar.rho_b_cost)
        feat_beacon[i] = np.array([m_neg, counter_mass, r_minus, ce_b, frag_drop, rho_cost], dtype=np.float64)

        # BEACON-adaptive binary baseline score.
        score_adaptive[i] = _compute_adaptive_max_effect(
            x=x,
            clf=clf,
            t_slices=t_slices,
            q=args.q,
            neutralizer_mode=args.neutralizer,
            phase1_ratio=args.adaptive_phase1_ratio,
            early_eff=args.adaptive_early_eff,
        )

    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(cv.split(feat_beacon, y_det))

    pred_prob_var = np.zeros(n, dtype=np.float64)
    pred_bin_var = np.zeros(n, dtype=np.int64)
    pred_prob_ad = np.zeros(n, dtype=np.float64)
    pred_bin_ad = np.zeros(n, dtype=np.int64)
    for tr, te in splits:
        ytr = y_det[tr]
        s_tr = score_variance[tr]
        s_te = score_variance[te]
        t_var = _best_f1_threshold(ytr, s_tr)
        pred_prob_var[te] = s_te
        pred_bin_var[te] = (s_te >= t_var).astype(np.int64)

        a_tr = score_adaptive[tr]
        a_te = score_adaptive[te]
        t_ad = _best_f1_threshold(ytr, a_tr)
        pred_prob_ad[te] = a_te
        pred_bin_ad[te] = (a_te >= t_ad).astype(np.int64)

    def _metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float, float]:
        auc = _safe_auc(y, prob)
        auprc = _safe_auprc(y, prob)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        return float(auc), float(auprc), float(f1), float(p), float(r)

    auc_v, auprc_v, f1_v, p_v, r_v = _metrics(y_det, pred_prob_var, pred_bin_var)
    auc_a, auprc_a, f1_a, p_a, r_a = _metrics(y_det, pred_prob_ad, pred_bin_ad)

    tan_bins_grid = _parse_grid(args.tan_bins_grid, int) if args.tan_bins_grid.strip() else [int(args.tan_bins)]
    tan_alpha_grid = _parse_grid(args.tan_alpha_grid, float) if args.tan_alpha_grid.strip() else [float(args.tan_alpha)]
    feature_sets = [s.strip() for s in args.feature_sets.split(",") if s.strip()]
    for fs in feature_sets:
        if fs not in FEATURE_SET_MAP:
            raise ValueError(f"Unknown feature set: {fs}")

    rows = []
    best_auc = -1.0
    best_maps: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    for fs in feature_sets:
        cols = FEATURE_SET_MAP[fs]
        xb = feat_beacon[:, cols]
        xu = feat_uniform[:, cols]
        for tb in tan_bins_grid:
            for ta in tan_alpha_grid:
                pred_prob_uni = np.zeros(n, dtype=np.float64)
                pred_bin_uni = np.zeros(n, dtype=np.int64)
                pred_prob_tan = np.zeros(n, dtype=np.float64)
                pred_bin_tan = np.zeros(n, dtype=np.int64)
                for tr, te in splits:
                    ytr = y_det[tr]
                    tan_u = TANModel(n_bins=int(tb), alpha=float(ta)).fit(xu[tr], ytr)
                    pu_tr = tan_u.predict_proba(xu[tr])[:, 1]
                    pu_te = tan_u.predict_proba(xu[te])[:, 1]
                    t_u = _best_f1_threshold(ytr, pu_tr)
                    pred_prob_uni[te] = pu_te
                    pred_bin_uni[te] = (pu_te >= t_u).astype(np.int64)

                    tan_b = TANModel(n_bins=int(tb), alpha=float(ta)).fit(xb[tr], ytr)
                    pb_tr = tan_b.predict_proba(xb[tr])[:, 1]
                    pb_te = tan_b.predict_proba(xb[te])[:, 1]
                    t_b = _best_f1_threshold(ytr, pb_tr)
                    pred_prob_tan[te] = pb_te
                    pred_bin_tan[te] = (pb_te >= t_b).astype(np.int64)

                auc_u, auprc_u, f1_u, p_u, r_u = _metrics(y_det, pred_prob_uni, pred_bin_uni)
                auc_t, auprc_t, f1_t, p_t, r_t = _metrics(y_det, pred_prob_tan, pred_bin_tan)
                d_auc, d_lo, d_hi, p_val = _bootstrap_auc_delta(
                    y_det,
                    pred_prob_tan,
                    pred_prob_uni,
                    n_boot=4000,
                    seed=args.seed + 777 + int(tb) * 17 + int(round(ta * 100)) * 31 + len(fs) * 101,
                )
                rows.append(
                    {
                        "feature_set": fs,
                        "tan_bins": int(tb),
                        "tan_alpha": float(ta),
                        "n_samples": int(n),
                        "n_positive": int(np.sum(y_det == 1)),
                        "q_max": int(args.q),
                        "auroc_tan_beacon": auc_t,
                        "auprc_tan_beacon": auprc_t,
                        "f1_tan_beacon": f1_t,
                        "precision_tan_beacon": p_t,
                        "recall_tan_beacon": r_t,
                        "auroc_uniform_tan": auc_u,
                        "auprc_uniform_tan": auprc_u,
                        "f1_uniform_tan": f1_u,
                        "precision_uniform_tan": p_u,
                        "recall_uniform_tan": r_u,
                        "auroc_variance": auc_v,
                        "auprc_variance": auprc_v,
                        "f1_variance": f1_v,
                        "auroc_beacon_adaptive": auc_a,
                        "auprc_beacon_adaptive": auprc_a,
                        "f1_beacon_adaptive": f1_a,
                        "delta_auroc_tan_vs_uniform": d_auc,
                        "delta_auroc_ci_low": d_lo,
                        "delta_auroc_ci_high": d_hi,
                        "p_value": p_val,
                    }
                )
                if np.isfinite(auc_t) and auc_t > best_auc:
                    best_auc = float(auc_t)
                    best_maps = {
                        "Variance heuristic": (pred_prob_var.copy(), pred_bin_var.copy()),
                        "Uniform occlusion + TAN": (pred_prob_uni.copy(), pred_bin_uni.copy()),
                        "BEACON-adaptive (binary)": (pred_prob_ad.copy(), pred_bin_ad.copy()),
                        "TAN + BEACON-core features": (pred_prob_tan.copy(), pred_bin_tan.copy()),
                    }

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["auroc_tan_beacon"], r["auprc_tan_beacon"]), reverse=True)
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    per_rows = []
    method_maps = best_maps if best_maps is not None else {
        "Variance heuristic": (pred_prob_var, pred_bin_var),
        "BEACON-adaptive (binary)": (pred_prob_ad, pred_bin_ad),
    }
    for name, (pr, pd) in method_maps.items():
        for i in range(n):
            per_rows.append(
                {
                    "sample_index": int(i),
                    "method": name,
                    "y_true": int(y_det[i]),
                    "score": float(pr[i]),
                    "y_pred": int(pd[i]),
                }
            )

    out_per = Path(args.out_per_sample)
    out_per.parent.mkdir(parents=True, exist_ok=True)
    with out_per.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        wr.writeheader()
        wr.writerows(per_rows)

    print(f"Detection dataset: n={n}, positives={int(np.sum(y_det == 1))}, negatives={int(np.sum(y_det == 0))}")
    if rows:
        top = rows[0]
        print(
            "best:"
            f" feature_set={top['feature_set']}, tan_bins={top['tan_bins']}, tan_alpha={top['tan_alpha']},"
            f" AUROC={top['auroc_tan_beacon']:.4f}, AUPRC={top['auprc_tan_beacon']:.4f},"
            f" dAUROC={top['delta_auroc_tan_vs_uniform']:.4f}, p={top['p_value']:.4g}"
        )
    print(f"saved: {out_summary}")
    print(f"saved: {out_per}")


if __name__ == "__main__":
    main()
