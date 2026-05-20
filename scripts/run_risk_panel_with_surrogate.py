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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beaconxai.audit_features import extract_audit_vector
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.fuzzy_policy_v2 import eval_at_budget
from beaconxai.preselect_surrogate import component_features, preselect_by_surrogate, selected_deltas
from scripts.run_component_conflict_benchmark import _train_extratrees_local, _train_histgbt_local
from scripts.run_part2_extended import _inject_hidden_conflict, _margin, _neutralize_component, _stratified_split, _time_slices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-budget risk panel with surrogate preselect")
    p.add_argument("--dataset", default="data/uci_har_shifted.npz")
    p.add_argument("--surrogate-pkl", required=True)
    p.add_argument("--model", choices=["extratrees", "histgbt", "cnn1d"], default="extratrees")
    p.add_argument("--n-total", type=int, default=1500)
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--time-bins", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--hidden-margin-drop-min", type=float, default=0.05)
    p.add_argument("--hidden-alpha-min", type=float, default=0.35)
    p.add_argument("--hidden-alpha-max", type=float, default=0.65)
    p.add_argument("--hidden-max-tries", type=int, default=20)
    p.add_argument("--neutralizer-mode", choices=["interp", "zero", "mean", "channel_mean", "class_mean"], default="interp")
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out-dir", default="outputs_composite/low_budget_surrogate_q16")
    return p.parse_args()


def _f1_budget(frac: float):
    def fn(y: np.ndarray, s: np.ndarray):
        _p, _r, f1 = eval_at_budget(y, s, frac)
        return float(f1)

    return fn


def _bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        da = fn(yy, a[idx])
        db = fn(yy, b[idx])
        if np.isfinite(da) and np.isfinite(db):
            vals.append(float(da - db))
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    p = 2.0 * min(float(np.mean(arr < 0.0)), float(np.mean(arr > 0.0)))
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(min(1.0, max(0.0, p)))


def _metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    p10, r10, f10 = eval_at_budget(y, s, 0.10)
    return {
        "auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else float("nan"),
        "auprc": float(average_precision_score(y, s)),
        "f1_10": float(f10),
        "precision_10": float(p10),
        "recall_10": float(r10),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.surrogate_pkl, "rb") as f:
        surrogate_pack = pickle.load(f)
    surrogate = surrogate_pack.model if hasattr(surrogate_pack, "model") else surrogate_pack

    x_train, y_train, x_test, y_test = load_npz_dataset(args.dataset)
    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "histgbt":
        clf = _train_histgbt_local(x_train, y_train)
    elif args.model == "cnn1d":
        from beaconxai.models import train_1dcnn

        clf = train_1dcnn(
            x_train, y_train, epochs=12, batch_size=256, lr=1e-3, label_smoothing=0.0, use_class_weights=True, tta_shifts=(0,)
        )
    else:
        clf = _train_extratrees_local(x_train, y_train, n_estimators=300, max_features=0.7, min_samples_leaf=1)

    n_channels = x_test.shape[2]
    t_slices = _time_slices(x_test.shape[1], args.time_bins)
    n_components = n_channels * len(t_slices)
    global_means = np.mean(x_train, axis=(0, 1)).astype(np.float32)
    class_means = {int(c): np.mean(x_train[y_train == c], axis=(0, 1)).astype(np.float32) for c in np.unique(y_train)}

    # build hidden-conflict detection set
    target_pos = args.n_total // 2
    target_neg = args.n_total - target_pos
    idx_all = np.arange(len(x_test), dtype=np.int64)
    rng.shuffle(idx_all)
    positives, pos_src, pos_cls, used = [], [], [], set()
    for i in idx_all:
        if len(positives) >= target_pos:
            break
        lg0 = clf.logits(x_test[i])
        _yy, m0 = _margin(lg0)
        yi = int(y_test[i])
        donor_pool = np.where(y_test != yi)[0]
        if len(donor_pool) == 0:
            continue
        accepted = None
        for _ in range(max(1, args.hidden_max_tries)):
            c = int(rng.integers(0, n_channels))
            b = int(rng.integers(0, len(t_slices)))
            t0, t1 = t_slices[b]
            d_id = int(donor_pool[int(rng.integers(0, len(donor_pool)))])
            alpha = float(rng.uniform(args.hidden_alpha_min, args.hidden_alpha_max))
            xc = _inject_hidden_conflict(x_test[i], x_test[d_id], c, t0, t1, alpha)
            _y1, m1 = _margin(clf.logits(xc))
            if float(m0 - m1) >= args.hidden_margin_drop_min:
                accepted = xc
                break
        if accepted is None:
            continue
        positives.append(accepted)
        pos_src.append(int(i))
        pos_cls.append(int(yi))
        used.add(int(i))

    neg_candidates = [int(i) for i in idx_all if int(i) not in used]
    rng.shuffle(neg_candidates)
    neg_src = neg_candidates[:target_neg]
    negatives = [x_test[i] for i in neg_src]
    neg_cls = [int(y_test[i]) for i in neg_src]

    x_det = np.concatenate([np.asarray(positives, dtype=np.float32), np.asarray(negatives, dtype=np.float32)], axis=0)
    y_det = np.concatenate([np.ones(len(positives), dtype=np.int64), np.zeros(len(negatives), dtype=np.int64)], axis=0)
    src_ids = np.asarray(pos_src + neg_src, dtype=np.int64)
    src_cls = np.asarray(pos_cls + neg_cls, dtype=np.int64)
    perm = rng.permutation(len(y_det))
    x_det, y_det, src_ids, src_cls = x_det[perm], y_det[perm], src_ids[perm], src_cls[perm]
    tr, va, te = _stratified_split(y_det, args.train_frac, args.val_frac, args.seed)

    rows_u, rows_s, rows_diag = [], [], []
    for i in range(len(y_det)):
        x = x_det[i]
        y_lbl = int(y_det[i])
        lg = clf.logits(x)
        yhat = int(np.argmax(lg))
        tmp = lg.copy()
        tmp[yhat] = -1e18
        yrunner = int(np.argmax(tmp))
        if args.neutralizer_mode in ("mean", "channel_mean", "class_mean"):
            cm = class_means.get(yhat, global_means) if args.neutralizer_mode == "class_mean" else global_means
        else:
            cm = None

        feats = component_features(x, t_slices, class_means.get(yhat, global_means), class_means.get(yrunner, global_means))
        sel_s = preselect_by_surrogate(feats, surrogate, q_max=args.q_max, positive_only=args.positive_only)
        sel_u = np.asarray(
            np.random.default_rng(args.seed + 1000 + i).choice(n_components, size=min(args.q_max, n_components), replace=False),
            dtype=np.int64,
        )
        d_s, m0_s = selected_deltas(x, sel_s, clf, t_slices, _neutralize_component, args.neutralizer_mode, cm)
        d_u, m0_u = selected_deltas(x, sel_u, clf, t_slices, _neutralize_component, args.neutralizer_mode, cm)

        rows_s.append(extract_audit_vector(None, m0_s, args.q_max, i, y_lbl, y_lbl, "risk_preselect_surrogate", args.seed, deltas=d_s))
        rows_u.append(extract_audit_vector(None, m0_u, args.q_max, i, y_lbl, y_lbl, "uniform", args.seed, deltas=d_u))

        rows_diag.append(
            {
                "sample_id": int(i),
                "q_max": int(args.q_max),
                "is_hidden_conflict": int(y_lbl),
                "surrogate_conflict_mass": float(np.sum(np.clip(-d_s[sel_s], 0.0, None))),
                "uniform_conflict_mass": float(np.sum(np.clip(-d_u[sel_u], 0.0, None))),
                "surrogate_share_conflict": float(np.mean(d_s[sel_s] < 0.0)) if len(sel_s) else 0.0,
                "uniform_share_conflict": float(np.mean(d_u[sel_u] < 0.0)) if len(sel_u) else 0.0,
            }
        )

    df_s = pd.DataFrame(rows_s).set_index("sample_id").sort_index()
    df_u = pd.DataFrame(rows_u).set_index("sample_id").sort_index()
    pd.DataFrame(rows_diag).to_csv(out / "surrogate_preselect_diagnostics.csv", index=False)
    df_s.to_csv(out / "audit_features_surrogate.csv")
    df_u.to_csv(out / "audit_features_uniform.csv")

    cols = ["m_neg", "M_B_minus", "r_B_minus", "CE_B", "rho_B_cost", "frag_drop", "top1_delta", "top3_sum_delta", "top3_conflict_count", "margin_entropy"]
    Xs = df_s[cols].to_numpy(dtype=float)
    Xu = df_u[cols].to_numpy(dtype=float)
    log_s = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    log_u = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    log_s.fit(Xs[tr], y_det[tr])
    log_u.fit(Xu[tr], y_det[tr])
    ss = log_s.predict_proba(Xs[te])[:, 1]
    su = log_u.predict_proba(Xu[te])[:, 1]

    res_rows = []
    ms, mu = _metrics(y_det[te], ss), _metrics(y_det[te], su)
    for k, v in ms.items():
        res_rows.append({"method": "surrogate", "metric": k, "value": float(v)})
    for k, v in mu.items():
        res_rows.append({"method": "uniform", "metric": k, "value": float(v)})
    pd.DataFrame(res_rows).to_csv(out / "risk_panel_surrogate_results.csv", index=False)

    boot = []
    for name, fn in {
        "delta_auroc": lambda y, s: float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else float("nan"),
        "delta_auprc": lambda y, s: float(average_precision_score(y, s)),
        "delta_f1_10": _f1_budget(0.10),
    }.items():
        d, lo, hi, p = _bootstrap_delta(y_det[te], ss, su, fn, args.n_boot, args.seed + abs(hash(name)) % 100000)
        boot.append({"comparison": "surrogate_vs_uniform", "metric": name, "delta": d, "ci_low": lo, "ci_high": hi, "p_value": p, "q_max": int(args.q_max)})
    dfb = pd.DataFrame(boot)
    dfb.to_csv(out / "risk_panel_surrogate_bootstrap.csv", index=False)

    claim = []
    for _, r in dfb.iterrows():
        claim.append(
            {
                "setting": f"tb{args.time_bins}_q{args.q_max}_{args.neutralizer_mode}",
                "comparison": str(r["comparison"]),
                "metric": str(r["metric"]),
                "delta": float(r["delta"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
                "p_value": float(r["p_value"]),
                "q_over_m": float(args.q_max / max(1, n_components)),
                "supported_positive": int((float(r["ci_low"]) > 0.0) and (float(r["p_value"]) < 0.05)),
            }
        )
    pd.DataFrame(claim).to_csv(out / "risk_panel_surrogate_claim_registry.csv", index=False)

    man = {
        "seed": int(args.seed),
        "n_total": int(len(y_det)),
        "q_max": int(args.q_max),
        "n_components": int(n_components),
        "train_ids": [int(v) for v in tr.tolist()],
        "val_ids": [int(v) for v in va.tolist()],
        "test_ids": [int(v) for v in te.tolist()],
        "source_sample_ids": [int(v) for v in src_ids.tolist()],
        "source_class_ids": [int(v) for v in src_cls.tolist()],
    }
    with (out / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
