#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.audit_features import extract_audit_vector
from beaconxai.core import BeaconAudit
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.fuzzy_policy_v2 import (
    build_fuzzy_inputs_v2,
    eval_at_budget,
    fit_fuzzy_policy_v2,
    gate_score,
    predict_fuzzy_policy_v2,
)
from beaconxai.neutralization import Neutralizer
from beaconxai.tan_policy import FEATURE_SETS, bootstrap_delta_auroc, fit_tan_policy, metrics_binary, predict_proba_tan
from beaconxai.types import BeaconConfig
from scripts.run_component_conflict_benchmark import _train_extratrees_local, _train_histgbt_local


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


def _neutralize_component(
    x: np.ndarray,
    t0: int,
    t1: int,
    c: int,
    mode: str,
    channel_means: np.ndarray | None = None,
) -> np.ndarray:
    y = x.copy()
    if mode == "zero":
        y[t0:t1, c:c + 1] = 0.0
        return y
    if mode in ("mean", "class_mean", "channel_mean"):
        if channel_means is None:
            y[t0:t1, c:c + 1] = 0.0
        else:
            y[t0:t1, c:c + 1] = float(channel_means[c])
        return y
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
        if n_tr + n_va + n_te > n:
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


def _bootstrap_ci_budget(y: np.ndarray, score: np.ndarray, frac: float, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    p_vals = []
    r_vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p, r, _f = eval_at_budget(y[idx], score[idx], frac)
        p_vals.append(p)
        r_vals.append(r)
    return (
        float(np.quantile(p_vals, 0.025)),
        float(np.quantile(p_vals, 0.975)),
        float(np.quantile(r_vals, 0.025)),
        float(np.quantile(r_vals, 0.975)),
    )


def _signed_scores_from_audit(audit_res, n_channels: int, t_slices: list[tuple[int, int]]) -> np.ndarray:
    n_bins = len(t_slices)
    out = np.zeros(n_channels * n_bins, dtype=np.float64)
    leaf = audit_res.metadata.get("leaf_components", [])
    deltas = audit_res.metadata.get("leaf_deltas", [])
    for comp_tuple, d in zip(leaf, deltas):
        _cid, lt0, lt1, lc0, lc1 = comp_tuple
        area = max(1, (lt1 - lt0) * (lc1 - lc0))
        for c in range(max(0, lc0), min(n_channels, lc1)):
            for bi, (t0, t1) in enumerate(t_slices):
                ov_t = max(0, min(lt1, t1) - max(lt0, t0))
                if ov_t <= 0:
                    continue
                out[_component_idx(c, bi, n_bins)] += float(d) * float(ov_t / area)
    return out


def _compute_uniform_deltas(
    x: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    q: int,
    neutralizer_mode: str,
    rng: np.random.Generator,
    channel_means: np.ndarray | None = None,
):
    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    lg0 = clf.logits(x)
    _y0, m0 = _margin(lg0)

    deltas = np.zeros(n_components, dtype=np.float64)
    budget = min(int(q), n_components)
    cand = rng.choice(n_components, size=budget, replace=False)
    for comp in cand:
        c, b = _component_decode(int(comp), n_bins)
        t0, t1 = t_slices[b]
        xm = _neutralize_component(x, t0, t1, c, neutralizer_mode, channel_means=channel_means)
        _y1, m1 = _margin(clf.logits(xm))
        deltas[int(comp)] = float(m0 - m1)

    pos_order = [idx for idx in np.argsort(-deltas) if deltas[idx] > 0]
    x_cur = x.copy()
    m_last = float(m0)
    k_flip = 0
    for k, comp in enumerate(pos_order, start=1):
        c, b = _component_decode(int(comp), n_bins)
        t0, t1 = t_slices[b]
        x_cur = _neutralize_component(x_cur, t0, t1, c, neutralizer_mode, channel_means=channel_means)
        _yy, mm = _margin(clf.logits(x_cur))
        m_last = float(mm)
        if mm <= 0.0:
            k_flip = k
            break
    rho_cost = float(k_flip / max(n_components, 1)) if k_flip > 0 else 1.0
    frag_drop = float(max(0.0, m0 - m_last))
    return deltas, float(m0), rho_cost, frag_drop


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = float(np.std(x))
    if s <= 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / s


def _compute_adaptive_v2_deltas(
    x: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    q: int,
    neutralizer_mode: str,
    channel_means: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float]:
    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    lg0 = clf.logits(x)
    _y0, m0 = _margin(lg0)

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
    order = np.argsort(-score)
    budget = min(int(q), n_components)
    cand = order[:budget]

    deltas = np.zeros(n_components, dtype=np.float64)
    for comp in cand:
        c, b = _component_decode(int(comp), n_bins)
        t0, t1 = t_slices[b]
        xm = _neutralize_component(x, t0, t1, c, neutralizer_mode, channel_means=channel_means)
        _y1, m1 = _margin(clf.logits(xm))
        deltas[int(comp)] = float(m0 - m1)

    pos_order = [idx for idx in np.argsort(-deltas) if deltas[idx] > 0]
    x_cur = x.copy()
    m_last = float(m0)
    k_flip = 0
    for k, comp in enumerate(pos_order, start=1):
        c, b = _component_decode(int(comp), n_bins)
        t0, t1 = t_slices[b]
        x_cur = _neutralize_component(x_cur, t0, t1, c, neutralizer_mode, channel_means=channel_means)
        _yy, mm = _margin(clf.logits(x_cur))
        m_last = float(mm)
        if mm <= 0.0:
            k_flip = k
            break
    rho_cost = float(k_flip / max(n_components, 1)) if k_flip > 0 else 1.0
    frag_drop = float(max(0.0, m0 - m_last))
    return deltas, float(m0), rho_cost, frag_drop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended Part2 pipeline: BEACON features + TAN + fuzzy + gates")
    p.add_argument("--dataset", default="data/uci_har_shifted.npz")
    p.add_argument("--model", choices=["extratrees", "histgbt", "cnn1d"], default="extratrees")
    p.add_argument("--n-total", type=int, default=5000)
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--time-bins", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--hidden-margin-drop-min", type=float, default=0.05)
    p.add_argument("--hidden-alpha-min", type=float, default=0.35)
    p.add_argument("--hidden-alpha-max", type=float, default=0.65)
    p.add_argument("--hidden-max-tries", type=int, default=20)
    p.add_argument("--tan-bins", default="3,4,5,6")
    p.add_argument("--tan-alpha", default="0.1,0.5,1.0,2.0")
    p.add_argument("--neutralizer-mode", choices=["interp", "zero", "mean", "channel_mean", "class_mean"], default="interp")
    p.add_argument("--adaptive-v2-preselect", action="store_true")
    p.add_argument("--features-only", action="store_true")
    p.add_argument("--save-delta-vectors", action="store_true")
    p.add_argument("--out", default="outputs_composite/part2_extended")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    x_train, y_train, x_test, y_test = load_npz_dataset(args.dataset)
    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "histgbt":
        clf = _train_histgbt_local(x_train, y_train)
    elif args.model == "cnn1d":
        from beaconxai.models import train_1dcnn

        clf = train_1dcnn(
            x_train,
            y_train,
            epochs=12,
            batch_size=256,
            lr=1e-3,
            label_smoothing=0.0,
            use_class_weights=True,
            tta_shifts=(0,),
        )
    else:
        clf = _train_extratrees_local(x_train, y_train, n_estimators=300, max_features=0.7, min_samples_leaf=1)

    n_channels = x_test.shape[2]
    t_slices = _time_slices(x_test.shape[1], args.time_bins)
    n_bins = len(t_slices)

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
        lg0 = clf.logits(x_test[i])
        _yy, m0 = _margin(lg0)
        yi = int(y_test[i])
        donor_pool = np.where(y_test != yi)[0]
        if len(donor_pool) == 0:
            continue
        accepted = None
        for _ in range(max(1, args.hidden_max_tries)):
            c = int(rng.integers(0, n_channels))
            b = int(rng.integers(0, n_bins))
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
    src_ids = np.asarray(pos_src + neg_src, dtype=np.int64)
    src_cls = np.asarray(pos_cls + neg_cls, dtype=np.int64)

    perm = rng.permutation(len(y_det))
    x_det = x_det[perm]
    y_det = y_det[perm]
    src_ids = src_ids[perm]
    src_cls = src_cls[perm]

    tr, va, te = _stratified_split(y_det, args.train_frac, args.val_frac, args.seed)

    cfg = BeaconConfig(
        q_max=args.q_max,
        k0=8 if args.q_max >= 16 else 4,
        l_min=4,
        k_pos=3,
        k_neg=3,
        partition_mode="sensor_group_time",
        refinement_mode="mixed",
        margin_mode="adaptive_all",
        risk_policy="rho_only",
        audit_mode="full",
    )
    global_channel_means = np.mean(x_train, axis=(0, 1)).astype(np.float32)
    class_channel_means: dict[int, np.ndarray] = {}
    for cls in np.unique(y_train):
        class_channel_means[int(cls)] = np.mean(x_train[y_train == cls], axis=(0, 1)).astype(np.float32)

    if args.neutralizer_mode == "interp":
        neutralizer = Neutralizer(mode="interp", channel_means=np.zeros(n_channels, dtype=np.float32))
    elif args.neutralizer_mode == "zero":
        neutralizer = Neutralizer(mode="zero", channel_means=np.zeros(n_channels, dtype=np.float32))
    else:
        # channel_mean / class_mean are both mean-style substitutions.
        neutralizer = Neutralizer(mode="mean", channel_means=global_channel_means)
    audit = BeaconAudit(model_logits=clf.logits, neutralizer=neutralizer, config=cfg)

    rows_beacon = []
    rows_uniform = []
    rows_adapt = []
    delta_rows_beacon = []
    delta_rows_uniform = []
    delta_rows_adapt = []
    for i in range(len(y_det)):
        x = x_det[i]
        y_lbl = int(y_det[i])
        src_class = int(src_cls[i])
        lg0 = clf.logits(x)
        _yh, m0 = _margin(lg0)

        if args.neutralizer_mode == "class_mean":
            cm = class_channel_means.get(src_class, global_channel_means)
            audit_i = BeaconAudit(model_logits=clf.logits, neutralizer=Neutralizer(mode="mean", channel_means=cm), config=cfg)
            ar = audit_i.audit(x)
            uniform_means = cm
        elif args.neutralizer_mode == "channel_mean":
            ar = audit.audit(x)
            uniform_means = global_channel_means
        else:
            ar = audit.audit(x)
            uniform_means = None
        deltas_b = _signed_scores_from_audit(ar, n_channels, t_slices)
        rows_beacon.append(
            extract_audit_vector(
                beacon_result=ar,
                margin=m0,
                q_max=args.q_max,
                sample_id=i,
                label=y_lbl,
                is_hidden_conflict=y_lbl,
                method="beacon_core",
                seed=args.seed,
                deltas=deltas_b,
            )
        )
        if args.save_delta_vectors:
            row_db = {
                "sample_id": int(i),
                "label": int(y_lbl),
                "is_hidden_conflict": int(y_lbl),
                "method": "beacon_core",
                "q_max": int(args.q_max),
                "seed": int(args.seed),
            }
            for j, v in enumerate(deltas_b.tolist()):
                row_db[f"d{j:03d}"] = float(v)
            delta_rows_beacon.append(row_db)

        deltas_u, m0_u, rho_u, frag_u = _compute_uniform_deltas(
            x=x,
            clf=clf,
            t_slices=t_slices,
            q=args.q_max,
            neutralizer_mode=args.neutralizer_mode,
            rng=np.random.default_rng(args.seed + 1000 + i),
            channel_means=uniform_means,
        )
        rows_uniform.append(
            extract_audit_vector(
                beacon_result=None,
                margin=m0_u,
                q_max=args.q_max,
                sample_id=i,
                label=y_lbl,
                is_hidden_conflict=y_lbl,
                method="uniform_occlusion",
                seed=args.seed,
                deltas=deltas_u,
                rho_b_cost=rho_u,
                frag_drop=frag_u,
            )
        )
        if args.save_delta_vectors:
            row_du = {
                "sample_id": int(i),
                "label": int(y_lbl),
                "is_hidden_conflict": int(y_lbl),
                "method": "uniform_occlusion",
                "q_max": int(args.q_max),
                "seed": int(args.seed),
            }
            for j, v in enumerate(deltas_u.tolist()):
                row_du[f"d{j:03d}"] = float(v)
            delta_rows_uniform.append(row_du)

        if args.adaptive_v2_preselect:
            deltas_a, m0_a, rho_a, frag_a = _compute_adaptive_v2_deltas(
                x=x,
                clf=clf,
                t_slices=t_slices,
                q=args.q_max,
                neutralizer_mode=args.neutralizer_mode,
                channel_means=uniform_means,
            )
            rows_adapt.append(
                extract_audit_vector(
                    beacon_result=None,
                    margin=m0_a,
                    q_max=args.q_max,
                    sample_id=i,
                    label=y_lbl,
                    is_hidden_conflict=y_lbl,
                    method="adaptive_v2_preselect",
                    seed=args.seed,
                    deltas=deltas_a,
                    rho_b_cost=rho_a,
                    frag_drop=frag_a,
                )
            )
            if args.save_delta_vectors:
                row_da = {
                    "sample_id": int(i),
                    "label": int(y_lbl),
                    "is_hidden_conflict": int(y_lbl),
                    "method": "adaptive_v2_preselect",
                    "q_max": int(args.q_max),
                    "seed": int(args.seed),
                }
                for j, v in enumerate(deltas_a.tolist()):
                    row_da[f"d{j:03d}"] = float(v)
                delta_rows_adapt.append(row_da)

    df_b = pd.DataFrame(rows_beacon)
    df_u = pd.DataFrame(rows_uniform)
    df_b.to_csv(out_dir / "audit_features_beacon_core.csv", index=False)
    df_u.to_csv(out_dir / "audit_features_uniform.csv", index=False)
    if rows_adapt:
        pd.DataFrame(rows_adapt).to_csv(out_dir / "audit_features_adaptive_v2.csv", index=False)
    if args.save_delta_vectors:
        pd.DataFrame(delta_rows_beacon).to_csv(out_dir / "delta_vectors_beacon_core.csv", index=False)
        pd.DataFrame(delta_rows_uniform).to_csv(out_dir / "delta_vectors_uniform.csv", index=False)
        if delta_rows_adapt:
            pd.DataFrame(delta_rows_adapt).to_csv(out_dir / "delta_vectors_adaptive_v2.csv", index=False)

    manifest = {
        "seed": int(args.seed),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "n_total": int(len(y_det)),
        "q_max": int(args.q_max),
        "neutralizer_mode": str(args.neutralizer_mode),
        "model": str(args.model),
        "model_checkpoint": None,
        "fuzzy_version": "v2_weighted27_kmeans",
        "dataset": str(args.dataset),
        "train_ids": [int(v) for v in tr.tolist()],
        "val_ids": [int(v) for v in va.tolist()],
        "test_ids": [int(v) for v in te.tolist()],
        "class_balance": {
            "all_pos": int(np.sum(y_det == 1)),
            "all_neg": int(np.sum(y_det == 0)),
            "train_pos": int(np.sum(y_det[tr] == 1)),
            "val_pos": int(np.sum(y_det[va] == 1)),
            "test_pos": int(np.sum(y_det[te] == 1)),
        },
        "source_sample_ids": [int(v) for v in src_ids.tolist()],
        "source_class_ids": [int(v) for v in src_cls.tolist()],
    }
    with (out_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if args.features_only:
        print(f"saved (features-only): {out_dir}")
        print(f"n_total={len(y_det)} train={len(tr)} val={len(va)} test={len(te)}")
        return

    # TAN sweep.
    tan_bins = [int(v.strip()) for v in args.tan_bins.split(",") if v.strip()]
    tan_alpha = [float(v.strip()) for v in args.tan_alpha.split(",") if v.strip()]

    tan_rows = []
    for fs_name, fs_cols in FEATURE_SETS.items():
        Xb = df_b[fs_cols].to_numpy(dtype=float)
        Xu = df_u[fs_cols].to_numpy(dtype=float)
        for nb in tan_bins:
            for al in tan_alpha:
                pol_b = fit_tan_policy(Xb[tr], y_det[tr], Xb[va], y_det[va], n_bins=nb, alpha=al)
                pol_u = fit_tan_policy(Xu[tr], y_det[tr], Xu[va], y_det[va], n_bins=nb, alpha=al)

                p_b = predict_proba_tan(pol_b, Xb[te])
                p_u = predict_proba_tan(pol_u, Xu[te])
                pred_b = (p_b >= float(pol_b["threshold"])).astype(np.int64)
                pred_u = (p_u >= float(pol_u["threshold"])).astype(np.int64)
                mb = metrics_binary(y_det[te], p_b, pred_b)
                mu = metrics_binary(y_det[te], p_u, pred_u)
                d_mean, d_lo, d_hi, pval = bootstrap_delta_auroc(y_det[te], p_b, p_u, n_boot=1000, seed=args.seed + nb * 19 + int(round(al * 100)))

                tan_rows.append(
                    {
                        "feature_set": fs_name,
                        "n_bins": nb,
                        "alpha": al,
                        "val_auroc": pol_b["val_metrics"]["auroc"],
                        "val_auprc": pol_b["val_metrics"]["auprc"],
                        "val_f1": pol_b["val_metrics"]["f1"],
                        "test_auroc": mb["auroc"],
                        "test_auprc": mb["auprc"],
                        "test_f1": mb["f1"],
                        "test_precision": mb["precision"],
                        "test_recall": mb["recall"],
                        "uniform_test_auroc": mu["auroc"],
                        "uniform_test_auprc": mu["auprc"],
                        "delta_auroc_vs_uniform": d_mean,
                        "ci_low": d_lo,
                        "ci_high": d_hi,
                        "p_value": pval,
                    }
                )

    tan_df = pd.DataFrame(tan_rows).sort_values(["val_auroc", "val_auprc", "val_f1"], ascending=False)
    tan_df.to_csv(out_dir / "tan_sweep_results.csv", index=False)
    tan_df.head(1).to_csv(out_dir / "tan_final_test.csv", index=False)

    # Fuzzy v2 sweep + gates (27-rule weighted Sugeno with kmeans memberships).
    Xf = build_fuzzy_inputs_v2(df_b)
    fuzzy_grid = []
    best_cfg = None
    best_target = -1.0
    best_tau_pair = (0.30, 0.70)

    Xp = df_b[["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]].to_numpy(dtype=float)
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    logit.fit(Xp[tr], y_det[tr])
    p_logit_val = logit.predict_proba(Xp[va])[:, 1]
    p_logit_test = logit.predict_proba(Xp[te])[:, 1]

    best_tan = tan_df.iloc[0]
    fs_cols = FEATURE_SETS[str(best_tan["feature_set"])]
    pol_best = fit_tan_policy(
        df_b[fs_cols].to_numpy(dtype=float)[tr],
        y_det[tr],
        df_b[fs_cols].to_numpy(dtype=float)[va],
        y_det[va],
        n_bins=int(best_tan["n_bins"]),
        alpha=float(best_tan["alpha"]),
    )
    p_tan_val = predict_proba_tan(pol_best, df_b[fs_cols].to_numpy(dtype=float)[va])
    p_tan_test = predict_proba_tan(pol_best, df_b[fs_cols].to_numpy(dtype=float)[te])
    for reg in [1e-4, 1e-3, 1e-2]:
        pol_f = fit_fuzzy_policy_v2(Xf[tr], y_det[tr], reg=reg, seed=args.seed + int(reg * 1e6))
        fv = predict_fuzzy_policy_v2(pol_f, Xf[va])
        p10, r10, f10 = eval_at_budget(y_det[va], fv, 0.10)
        p20, r20, f20 = eval_at_budget(y_det[va], fv, 0.20)
        auprc = float(average_precision_score(y_det[va], fv))

        for ql, qh in [(0.20, 0.80), (0.25, 0.75), (0.30, 0.70)]:
            tau_low = float(np.quantile(fv, ql))
            tau_high = float(np.quantile(fv, qh))

            gv_logit = gate_score(fv, p_logit_val, tau_low, tau_high)
            gp10, gr10, gf10 = eval_at_budget(y_det[va], gv_logit, 0.10)
            _, _, gf20 = eval_at_budget(y_det[va], gv_logit, 0.20)
            g_auprc = float(average_precision_score(y_det[va], gv_logit))

            gv_tan = gate_score(fv, p_tan_val, tau_low, tau_high)
            tp10, tr10, tf10 = eval_at_budget(y_det[va], gv_tan, 0.10)
            _, _, tf20 = eval_at_budget(y_det[va], gv_tan, 0.20)
            t_auprc = float(average_precision_score(y_det[va], gv_tan))

            target = 0.5 * gf10 + 0.25 * gp10 + 0.25 * gf20
            fuzzy_grid.append(
                {
                    "membership_scheme": "kmeans_3bin",
                    "inference": "sugeno_weighted",
                    "rule_set": "full27",
                    "n_rules": 27,
                    "reg": reg,
                    "tau_low_q": ql,
                    "tau_high_q": qh,
                    "val_fuzzy_f1_10": f10,
                    "val_fuzzy_f1_20": f20,
                    "val_fuzzy_auprc": auprc,
                    "val_gate_logit_f1_10": gf10,
                    "val_gate_logit_f1_20": gf20,
                    "val_gate_logit_auprc": g_auprc,
                    "val_gate_tan_f1_10": tf10,
                    "val_gate_tan_f1_20": tf20,
                    "val_gate_tan_auprc": t_auprc,
                    "val_target": target,
                }
            )
            if target > best_target:
                best_target = target
                best_cfg = pol_f
                best_tau_pair = (ql, qh)

    fuzzy_df = pd.DataFrame(fuzzy_grid).sort_values("val_target", ascending=False)
    fuzzy_df.to_csv(out_dir / "fuzzy_sweep_results.csv", index=False)
    fuzzy_df.to_csv(out_dir / "fuzzy_policy_results.csv", index=False)

    assert best_cfg is not None
    fv = predict_fuzzy_policy_v2(best_cfg, Xf[va])
    ft = predict_fuzzy_policy_v2(best_cfg, Xf[te])
    tau_low = float(np.quantile(fv, best_tau_pair[0]))
    tau_high = float(np.quantile(fv, best_tau_pair[1]))

    g_logit = gate_score(ft, p_logit_test, tau_low, tau_high)
    g_tan = gate_score(ft, p_tan_test, tau_low, tau_high)

    scalar = df_b["m_neg"].to_numpy(dtype=float)[te]
    panel = p_logit_test

    policies = {
        "scalar": scalar,
        "logit_panel": panel,
        "fuzzy_only": ft,
        "fuzzy_gate_logit": g_logit,
        "fuzzy_gate_tan": g_tan,
    }

    pol_rows = []
    for name, s in policies.items():
        p10, r10, f10 = eval_at_budget(y_det[te], s, 0.10)
        p20, r20, f20 = eval_at_budget(y_det[te], s, 0.20)
        ci10 = _bootstrap_ci_budget(y_det[te], s, 0.10, n_boot=1000, seed=args.seed + len(name) * 7 + 10)
        ci20 = _bootstrap_ci_budget(y_det[te], s, 0.20, n_boot=1000, seed=args.seed + len(name) * 7 + 20)
        pol_rows.append(
            {
                "policy": name,
                "budget": 0.10,
                "precision": p10,
                "recall": r10,
                "f1": f10,
                "auprc": float(average_precision_score(y_det[te], s)),
                "membership_scheme": "kmeans_3bin",
                "inference": "sugeno_weighted",
                "rule_set": "full27",
                "n_rules": 27,
                "ci_precision_low": ci10[0],
                "ci_precision_high": ci10[1],
                "ci_recall_low": ci10[2],
                "ci_recall_high": ci10[3],
            }
        )
        pol_rows.append(
            {
                "policy": name,
                "budget": 0.20,
                "precision": p20,
                "recall": r20,
                "f1": f20,
                "auprc": float(average_precision_score(y_det[te], s)),
                "membership_scheme": "kmeans_3bin",
                "inference": "sugeno_weighted",
                "rule_set": "full27",
                "n_rules": 27,
                "ci_precision_low": ci20[0],
                "ci_precision_high": ci20[1],
                "ci_recall_low": ci20[2],
                "ci_recall_high": ci20[3],
            }
        )

    pol_df = pd.DataFrame(pol_rows)
    pol_df.to_csv(out_dir / "fuzzy_final_test.csv", index=False)
    pol_df.to_csv(out_dir / "policy_comparison.csv", index=False)

    # Bootstrap deltas for key comparisons.
    deltas = []
    key = {
        "tan_beacon_vs_uniform": (
            predict_proba_tan(
                fit_tan_policy(
                    df_b[fs_cols].to_numpy(dtype=float)[tr], y_det[tr], df_b[fs_cols].to_numpy(dtype=float)[va], y_det[va],
                    n_bins=int(best_tan["n_bins"]), alpha=float(best_tan["alpha"])
                ),
                df_b[fs_cols].to_numpy(dtype=float)[te],
            ),
            predict_proba_tan(
                fit_tan_policy(
                    df_u[fs_cols].to_numpy(dtype=float)[tr], y_det[tr], df_u[fs_cols].to_numpy(dtype=float)[va], y_det[va],
                    n_bins=int(best_tan["n_bins"]), alpha=float(best_tan["alpha"])
                ),
                df_u[fs_cols].to_numpy(dtype=float)[te],
            ),
        ),
        "fuzzy_gate_logit_vs_panel": (g_logit, panel),
        "fuzzy_gate_tan_vs_panel": (g_tan, panel),
    }
    for name, (sa, sb) in key.items():
        d, lo, hi, pv = bootstrap_delta_auroc(y_det[te], sa, sb, n_boot=1000, seed=args.seed + len(name) * 13)
        deltas.append({"comparison": name, "delta_auroc": d, "ci_low": lo, "ci_high": hi, "p_value": pv})

    pd.DataFrame(deltas).to_csv(out_dir / "bootstrap_deltas.csv", index=False)

    print(f"saved: {out_dir}")
    print(f"n_total={len(y_det)} train={len(tr)} val={len(va)} test={len(te)}")


if __name__ == "__main__":
    main()
