#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass
class CallCounter:
    fn: Callable[[np.ndarray], np.ndarray]
    calls: int = 0

    def reset(self) -> None:
        self.calls = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls += int(np.asarray(x).shape[0])
        return self.fn(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tabular conflict localization benchmark (Adult)")
    p.add_argument("--model", choices=["rf", "xgb"], default="rf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--n-eval", type=int, default=30)
    p.add_argument("--k-conflict", type=int, default=3)
    p.add_argument("--q-values", default="8,16")
    p.add_argument("--lime-samples", type=int, default=500)
    p.add_argument("--shap-samples", type=int, default=200)
    p.add_argument("--shap-bg", type=int, default=80)
    p.add_argument("--out", default="outputs_composite/tabular_conflict_adult_results.csv")
    p.add_argument("--per-sample-out", default="outputs_composite/tabular_conflict_adult_per_sample.csv")
    return p.parse_args()


def load_adult(seed: int, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None,
        na_values=" ?",
        skipinitialspace=True,
    )
    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "target",
    ]
    df.columns = cols
    df = df.dropna().reset_index(drop=True)
    y = (df["target"].astype(str).str.strip() == ">50K").astype(int).values
    x = df.drop(columns=["target"]).copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    return x_train.reset_index(drop=True), x_test.reset_index(drop=True), y_train, y_test


def encode_tabular(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, dict]]:
    xtr = x_train.copy()
    xte = x_test.copy()
    enc_meta: dict[str, dict] = {}
    feat_names = list(xtr.columns)
    for c in feat_names:
        if not is_numeric_dtype(xtr[c]):
            vals = sorted(xtr[c].astype(str).unique().tolist())
            m = {v: i for i, v in enumerate(vals)}
            xtr[c] = xtr[c].astype(str).map(m).astype(float)
            xte[c] = xte[c].astype(str).map(m).fillna(-1.0).astype(float)
            enc_meta[c] = {"kind": "cat", "map": m}
        else:
            med = float(np.nanmedian(xtr[c].values.astype(float)))
            xtr[c] = xtr[c].astype(float).fillna(med)
            xte[c] = xte[c].astype(float).fillna(med)
            enc_meta[c] = {"kind": "num", "median": med}
    return xtr.values.astype(np.float32), xte.values.astype(np.float32), feat_names, enc_meta


def train_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, seed: int):
    if model_name == "rf":
        m = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=seed,
        )
        m.fit(x_train, y_train)
        return m
    if XGBClassifier is None:
        raise RuntimeError("xgboost not installed")
    m = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=seed,
    )
    m.fit(x_train, y_train)
    return m


def restore_subset(x_conf: np.ndarray, x_clean: np.ndarray, feat_idx: list[int]) -> np.ndarray:
    z = x_conf.copy()
    z[feat_idx] = x_clean[feat_idx]
    return z


def make_conflicts(
    x_test: np.ndarray,
    y_test: np.ndarray,
    proba_fn: Callable[[np.ndarray], np.ndarray],
    k_conflict: int,
    n_eval: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    probs = proba_fn(x_test)
    pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)
    ok = np.where((pred == y_test) & (conf >= 0.70))[0]
    out = []
    for i in rng.permutation(ok):
        y_ref = int(pred[i])
        donors = np.where(y_test != y_ref)[0]
        if len(donors) == 0:
            continue
        j = int(rng.choice(donors))
        feat_idx = rng.choice(x_test.shape[1], size=k_conflict, replace=False).tolist()
        x_clean = x_test[i].copy()
        x_conf = x_clean.copy()
        x_conf[feat_idx] = x_test[j][feat_idx]
        p_clean = float(proba_fn(x_clean[None, :])[0, y_ref])
        p_conf = float(proba_fn(x_conf[None, :])[0, y_ref])
        if p_conf >= p_clean - 1e-6:
            continue
        out.append(
            {
                "idx": int(i),
                "x_clean": x_clean,
                "x_conf": x_conf,
                "y_ref": y_ref,
                "true_feats": feat_idx,
                "p_clean": p_clean,
                "p_conf": p_conf,
            }
        )
        if len(out) >= n_eval:
            break
    return out


def beacon_rank(
    x_clean: np.ndarray,
    x_conf: np.ndarray,
    y_ref: int,
    q: int,
    counter: CallCounter,
) -> tuple[np.ndarray, int]:
    n_feat = len(x_clean)
    groups: list[list[int]] = [list(range(n_feat))]
    scored: list[tuple[list[int], float]] = []
    calls = 0
    p0 = float(counter(x_conf[None, :])[0, y_ref])
    calls += 1

    while calls < q and groups:
        g = groups.pop(0)
        z = restore_subset(x_conf, x_clean, g)
        p = float(counter(z[None, :])[0, y_ref])
        calls += 1
        s = max(0.0, p - p0)
        scored.append((g, s))
        if len(g) > 1 and calls + 2 <= q:
            mid = len(g) // 2
            left, right = g[:mid], g[mid:]
            if left:
                groups.append(left)
            if right:
                groups.append(right)
        groups.sort(key=lambda gg: -len(gg))

    feat_scores = np.zeros(n_feat, dtype=np.float64)
    for g, s in scored:
        inc = s / max(1, len(g))
        for f in g:
            feat_scores[f] += inc
    return np.argsort(-feat_scores), calls


def uniform_rank(
    x_clean: np.ndarray,
    x_conf: np.ndarray,
    y_ref: int,
    q: int,
    counter: CallCounter,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    n_feat = len(x_clean)
    cand = rng.choice(n_feat, size=min(q, n_feat), replace=False)
    p0 = float(counter(x_conf[None, :])[0, y_ref])
    calls = 1
    scores = np.full(n_feat, -1e18, dtype=np.float64)
    for f in cand:
        z = x_conf.copy()
        z[f] = x_clean[f]
        p = float(counter(z[None, :])[0, y_ref])
        calls += 1
        scores[f] = max(0.0, p - p0)
    return np.argsort(-scores), calls


def random_rank(n_feat: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n_feat)


def lime_rank(
    x_train: np.ndarray,
    x_conf: np.ndarray,
    y_ref: int,
    feat_names: list[str],
    counter: CallCounter,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    explainer = LimeTabularExplainer(
        training_data=x_train,
        feature_names=feat_names,
        class_names=["y0", "y1"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )
    counter.reset()
    exp = explainer.explain_instance(
        x_conf,
        counter,
        labels=[int(y_ref)],
        num_features=len(feat_names),
        num_samples=num_samples,
    )
    w = np.zeros(len(feat_names), dtype=np.float64)
    for fid, weight in exp.local_exp[int(y_ref)]:
        w[int(fid)] = max(0.0, -float(weight))
    if float(np.sum(w)) == 0.0:
        for fid, weight in exp.local_exp[int(y_ref)]:
            w[int(fid)] = abs(float(weight))
    return np.argsort(-w), counter.calls


def shap_rank(
    x_train: np.ndarray,
    x_conf: np.ndarray,
    y_ref: int,
    counter: CallCounter,
    bg_size: int,
    nsamples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_train), size=min(bg_size, len(x_train)), replace=False)
    background = x_train[idx]
    expl = shap.KernelExplainer(counter, background)
    counter.reset()
    sv = expl.shap_values(x_conf[None, :], nsamples=nsamples)
    if isinstance(sv, list):
        arr = np.asarray(sv[int(y_ref)])[0]
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr[0, :, int(y_ref)]
        elif arr.ndim == 2:
            arr = arr[0]
    s = np.maximum(0.0, -arr.astype(np.float64))
    if float(np.sum(s)) == 0.0:
        s = np.abs(arr.astype(np.float64))
    return np.argsort(-s), counter.calls


def topk_precision(ranking: np.ndarray, true_feats: list[int], k: int) -> float:
    top = set(ranking[:k].tolist())
    gt = set(true_feats)
    return float(len(top & gt) / max(1, k))


def top1_hit(ranking: np.ndarray, true_feats: list[int]) -> float:
    return float(int(int(ranking[0]) in set(true_feats)))


def main() -> None:
    args = parse_args()
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]

    xtr_df, xte_df, ytr, yte = load_adult(seed=args.seed, test_size=args.test_size)
    xtr, xte, feat_names, _ = encode_tabular(xtr_df, xte_df)
    model = train_model(args.model, xtr, ytr, args.seed)
    acc = float(accuracy_score(yte, np.argmax(model.predict_proba(xte), axis=1)))

    proba_counter = CallCounter(model.predict_proba)
    conflicts = make_conflicts(xte, yte, model.predict_proba, args.k_conflict, args.n_eval, args.seed + 11)
    if not conflicts:
        raise RuntimeError("No valid conflicts produced; relax filters")

    rows = []
    per_rows = []

    def eval_method(name: str, run_fn: Callable[[dict], tuple[np.ndarray, int]], budget_label: str):
        vals_p, vals_h, vals_c = [], [], []
        for t, ex in enumerate(conflicts):
            rnk, calls = run_fn(ex)
            p = topk_precision(rnk, ex["true_feats"], args.k_conflict)
            h = top1_hit(rnk, ex["true_feats"])
            vals_p.append(p)
            vals_h.append(h)
            vals_c.append(float(calls))
            per_rows.append(
                {
                    "model": args.model,
                    "method": name,
                    "budget_label": budget_label,
                    "sample_id": t,
                    "topk_precision": p,
                    "top1_hit": h,
                    "model_calls": calls,
                }
            )
        rows.append(
            {
                "dataset": "adult",
                "model": args.model,
                "method": name,
                "budget_label": budget_label,
                "n_eval": len(conflicts),
                "k_conflict": args.k_conflict,
                "topk_precision_mean": float(np.mean(vals_p)),
                "top1_hit_mean": float(np.mean(vals_h)),
                "mean_model_calls": float(np.mean(vals_c)),
                "test_accuracy": acc,
            }
        )

    for q in q_values:
        eval_method(
            f"beacon_q{q}",
            lambda ex, qq=q: beacon_rank(ex["x_clean"], ex["x_conf"], ex["y_ref"], qq, proba_counter),
            f"q={q}",
        )
        eval_method(
            f"uniform_occlusion_q{q}",
            lambda ex, qq=q: uniform_rank(
                ex["x_clean"], ex["x_conf"], ex["y_ref"], qq, proba_counter, args.seed + ex["idx"] + qq
            ),
            f"q={q}",
        )

    eval_method(
        "random",
        lambda ex: (random_rank(len(feat_names), args.seed + ex["idx"]), 0),
        "none",
    )
    eval_method(
        f"lime_{args.lime_samples}",
        lambda ex: lime_rank(
            xtr, ex["x_conf"], ex["y_ref"], feat_names, proba_counter, args.lime_samples, args.seed + ex["idx"]
        ),
        f"samples={args.lime_samples}",
    )
    eval_method(
        f"shap_kernel_{args.shap_samples}",
        lambda ex: shap_rank(
            xtr,
            ex["x_conf"],
            ex["y_ref"],
            proba_counter,
            args.shap_bg,
            args.shap_samples,
            args.seed + ex["idx"],
        ),
        f"nsamples={args.shap_samples}",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {out}")

    po = Path(args.per_sample_out)
    po.parent.mkdir(parents=True, exist_ok=True)
    with po.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)
    print(f"saved: {po}")

    for r in rows:
        print(
            f"{r['method']}: topk={r['topk_precision_mean']:.3f} top1={r['top1_hit_mean']:.3f} calls={r['mean_model_calls']:.1f}"
        )


if __name__ == "__main__":
    main()
