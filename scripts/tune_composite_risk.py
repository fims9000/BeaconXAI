#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[y_true == 1]))
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def rank_norm(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    if len(x) <= 1:
        return np.zeros_like(ranks)
    return ranks / (len(x) - 1)


def build_frame(out_dir: Path, k0: int, q: int) -> pd.DataFrame:
    rr = pd.read_csv(out_dir / f"risk_rows_k0_{k0}.csv")
    lm = pd.read_csv(out_dir / f"local_metrics_k0_{k0}.csv")

    br = rr[(rr.method == "beacon_refine") & (rr.q_max == q)][["sample_id", "is_error", "risk_score", "censored"]].copy()
    br = br.rename(columns={"risk_score": "risk_beacon", "censored": "censored_br"})

    conf = rr[(rr.method == "confidence") & (rr.q_max == 0)][["sample_id", "risk_score"]].copy()
    conf = conf.rename(columns={"risk_score": "risk_conf"})

    lmq = lm[(lm.method == "beacon_refine") & (lm.q_max == q)][
        ["sample_id", "necessity", "counter_evidence_gain", "sufficiency_margin", "rho_b_cost", "censored"]
    ].copy()
    lmq = lmq.rename(columns={"censored": "censored_lm"})

    df = br.merge(conf, on="sample_id", how="left").merge(lmq, on="sample_id", how="left")
    df = df.dropna().reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = df["is_error"].to_numpy(np.int64)

    f_beacon = rank_norm(df["risk_beacon"].to_numpy(np.float64))
    f_conf = rank_norm(df["risk_conf"].to_numpy(np.float64))
    f_rho = rank_norm(df["rho_b_cost"].to_numpy(np.float64))
    f_nec = rank_norm(df["necessity"].to_numpy(np.float64))
    f_ce = rank_norm(df["counter_evidence_gain"].to_numpy(np.float64))
    f_suff_bad = rank_norm((-df["sufficiency_margin"]).to_numpy(np.float64))
    f_cens = df[["censored_br", "censored_lm"]].max(axis=1).to_numpy(np.float64)

    X = np.stack([f_beacon, f_conf, f_rho, f_nec, f_ce, f_suff_bad, f_cens], axis=1)
    names = ["beacon", "conf", "rho", "nec", "ce", "suff_bad", "censored"]
    return X, y, names


def split_idx(y: np.ndarray, seed: int, calib_frac: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))

    pos = idx[y == 1]
    neg = idx[y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    kpos = int(len(pos) * calib_frac)
    kneg = int(len(neg) * calib_frac)

    calib = np.concatenate([pos[:kpos], neg[:kneg]])
    eval_ = np.concatenate([pos[kpos:], neg[kneg:]])
    rng.shuffle(calib)
    rng.shuffle(eval_)
    return calib, eval_


def random_search(X: np.ndarray, y: np.ndarray, calib_idx: np.ndarray, n_trials: int, seed: int):
    rng = np.random.default_rng(seed)
    best = None
    best_auc = -1.0

    # include simple baseline candidate first
    candidates = [np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64)]
    for _ in range(n_trials):
        w = rng.uniform(0.0, 2.0, size=X.shape[1])
        candidates.append(w)

    for w in candidates:
        s = X @ w
        a = auc(y[calib_idx], s[calib_idx])
        if np.isfinite(a) and a > best_auc:
            best_auc = a
            best = w

    return best, float(best_auc)


def evaluate_dir(out_dir: Path, k0: int, q: int, n_trials: int, seed: int, calib_frac: float) -> dict:
    df = build_frame(out_dir, k0, q)
    X, y, names = make_features(df)
    calib_idx, eval_idx = split_idx(y, seed=seed, calib_frac=calib_frac)

    w, calib_auc = random_search(X, y, calib_idx, n_trials=n_trials, seed=seed)
    s = X @ w

    out = {
        "n": int(len(y)),
        "error_rate": float(y.mean()),
        "q": int(q),
        "k0": int(k0),
        "weights": {k: float(v) for k, v in zip(names, w)},
        "auc": {
            "beacon_full": float(auc(y, X[:, 0])),
            "conf_full": float(auc(y, X[:, 1])),
            "composite_full": float(auc(y, s)),
            "beacon_calib": float(auc(y[calib_idx], X[calib_idx, 0])),
            "conf_calib": float(auc(y[calib_idx], X[calib_idx, 1])),
            "composite_calib": float(calib_auc),
            "beacon_eval": float(auc(y[eval_idx], X[eval_idx, 0])),
            "conf_eval": float(auc(y[eval_idx], X[eval_idx, 1])),
            "composite_eval": float(auc(y[eval_idx], s[eval_idx])),
        },
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune composite BEACON risk from existing outputs")
    p.add_argument("--out-dirs", default="./outputs_full_clean_cuda,./outputs_full_shifted_cuda")
    p.add_argument("--k0", type=int, default=8)
    p.add_argument("--q", type=int, default=32)
    p.add_argument("--n-trials", type=int, default=4000)
    p.add_argument("--calib-frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="./outputs_composite/composite_tuning.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dirs = [Path(x.strip()) for x in args.out_dirs.split(",") if x.strip()]

    result = {}
    for d in dirs:
        result[str(d)] = evaluate_dir(
            out_dir=d,
            k0=args.k0,
            q=args.q,
            n_trials=args.n_trials,
            seed=args.seed,
            calib_frac=args.calib_frac,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:")
    print(out)


if __name__ == "__main__":
    main()
