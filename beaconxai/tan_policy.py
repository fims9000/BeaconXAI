from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer


FEATURE_SETS = {
    "conflict_min": ["m_neg", "M_B_minus", "r_B_minus"],
    "conflict_ce": ["m_neg", "M_B_minus", "r_B_minus", "CE_B"],
    "full_compact": ["m_neg", "M_B_minus", "r_B_minus", "CE_B", "rho_B_cost", "frag_drop"],
    "rank_plus": [
        "m_neg",
        "M_B_minus",
        "r_B_minus",
        "CE_B",
        "top1_delta",
        "top3_sum_delta",
        "top3_conflict_count",
    ],
}


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
        Xd = X.astype(np.int64)
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
        Xd = X.astype(np.int64)
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


def metrics_binary(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) >= 2:
        auroc = float(roc_auc_score(y, prob))
        auprc = float(average_precision_score(y, prob))
    else:
        auroc = float("nan")
        auprc = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
    }


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    qs = np.linspace(0.05, 0.95, 91)
    thrs = np.quantile(score, qs)
    best_f1 = -1.0
    best_t = float(np.median(score))
    for t in thrs:
        pred = (score >= float(t)).astype(np.int64)
        f1 = metrics_binary(y, score, pred)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def fit_tan_policy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_bins: int,
    alpha: float,
    strategy: str = "quantile",
) -> dict[str, Any]:
    disc = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy=strategy,
        quantile_method="averaged_inverted_cdf" if strategy == "quantile" else "linear",
    )
    Xtr_d = disc.fit_transform(X_train)
    Xva_d = disc.transform(X_val)

    model = TANModel(n_bins=n_bins, alpha=alpha).fit(Xtr_d, y_train)
    p_tr = model.predict_proba(Xtr_d)[:, 1]
    p_va = model.predict_proba(Xva_d)[:, 1]
    thr = best_f1_threshold(y_train, p_tr)
    pred_va = (p_va >= thr).astype(np.int64)
    m = metrics_binary(y_val, p_va, pred_va)
    return {
        "discretizer": disc,
        "model": model,
        "threshold": thr,
        "val_metrics": m,
    }


def predict_proba_tan(policy: dict[str, Any], X: np.ndarray) -> np.ndarray:
    Xd = policy["discretizer"].transform(X)
    return policy["model"].predict_proba(Xd)[:, 1]


def bootstrap_delta_auroc(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        da = roc_auc_score(yy, a[idx])
        db = roc_auc_score(yy, b[idx])
        vals.append(float(da - db))
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(p)
