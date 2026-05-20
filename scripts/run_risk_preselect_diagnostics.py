#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.audit_features import extract_audit_vector
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.fuzzy_policy_v2 import eval_at_budget
from scripts.run_component_conflict_benchmark import _train_extratrees_local, _train_histgbt_local
from scripts.run_component_conflict_benchmark import _anfis_features
from scripts.run_part2_extended import (
    _component_decode,
    _component_idx,
    _inject_hidden_conflict,
    _margin,
    _neutralize_component,
    _stratified_split,
    _time_slices,
    _z,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Risk-preselect diagnostics + quick benchmark")
    p.add_argument("--dataset", default="data/uci_har_shifted.npz")
    p.add_argument("--model", choices=["extratrees", "histgbt", "cnn1d"], default="extratrees")
    p.add_argument("--n-total", type=int, default=600)
    p.add_argument("--q-values", default="16,32")
    p.add_argument("--q-max", type=int, default=None, help="Optional alias for single-Q run")
    p.add_argument("--time-bins", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--hidden-margin-drop-min", type=float, default=0.05)
    p.add_argument("--hidden-alpha-min", type=float, default=0.35)
    p.add_argument("--hidden-alpha-max", type=float, default=0.65)
    p.add_argument("--hidden-max-tries", type=int, default=20)
    p.add_argument("--neutralizer-mode", choices=["interp", "zero", "mean", "channel_mean", "class_mean"], default="interp")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument(
        "--methods",
        default="uniform,adaptive_v2_old,risk_preselect_profile,risk_preselect_margin,risk_preselect_combined",
        help="Comma-separated subset of methods",
    )
    p.add_argument("--out-dir", default="outputs_composite/part2_extended_v8_q16_adapt_n600")
    return p.parse_args()


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
    p = float(min(1.0, max(0.0, p)))
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), p


def _f1_budget(frac: float):
    def fn(y: np.ndarray, s: np.ndarray):
        _p, _r, f1 = eval_at_budget(y, s, frac)
        return float(f1)

    return fn


def _compute_full_deltas(
    x: np.ndarray,
    clf,
    t_slices: list[tuple[int, int]],
    neutralizer_mode: str,
    channel_means: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    # Fast path for tree/boosting locals with sklearn model under `clf.model`.
    if hasattr(clf, "model"):
        try:
            xb = np.repeat(x[None, :, :], n_components + 1, axis=0)
            for comp in range(n_components):
                c, b = _component_decode(int(comp), n_bins)
                t0, t1 = t_slices[b]
                if neutralizer_mode == "zero":
                    xb[comp + 1, t0:t1, c] = 0.0
                elif neutralizer_mode in ("mean", "class_mean", "channel_mean"):
                    cm = float(channel_means[c]) if channel_means is not None else 0.0
                    xb[comp + 1, t0:t1, c] = cm
                else:
                    if t0 > 0 and t1 < x.shape[0]:
                        left = float(xb[comp + 1, t0 - 1, c])
                        right = float(xb[comp + 1, t1, c])
                        xb[comp + 1, t0:t1, c] = np.linspace(left, right, t1 - t0, endpoint=False)
                    else:
                        xb[comp + 1, t0:t1, c] = 0.0

            feats = _anfis_features(xb)
            probs = clf.model.predict_proba(feats)
            logits = np.log(np.clip(probs, 1e-12, 1.0))
            top1 = np.max(logits, axis=1)
            if logits.shape[1] >= 2:
                top2 = np.partition(logits, -2, axis=1)[:, -2]
            else:
                top2 = np.zeros_like(top1)
            margins = top1 - top2
            m0 = float(margins[0])
            deltas = m0 - margins[1:]
            return deltas.astype(np.float64), m0
        except Exception:
            pass

    lg0 = clf.logits(x)
    _y0, m0 = _margin(lg0)
    deltas = np.zeros(n_components, dtype=np.float64)
    for comp in range(n_components):
        c, b = _component_decode(int(comp), n_bins)
        t0, t1 = t_slices[b]
        xm = _neutralize_component(x, t0, t1, c, neutralizer_mode, channel_means=channel_means)
        _y1, m1 = _margin(clf.logits(xm))
        deltas[int(comp)] = float(m0 - m1)
    return deltas, float(m0)


def _risk_scores(
    x: np.ndarray,
    t_slices: list[tuple[int, int]],
    yhat: int,
    y_runner: int,
    class_channel_means: dict[int, np.ndarray],
):
    n_channels = x.shape[1]
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    variance = np.zeros(n_components, dtype=np.float64)
    dist_yhat = np.zeros(n_components, dtype=np.float64)
    dist_run = np.zeros(n_components, dtype=np.float64)
    class_sep = np.zeros(n_components, dtype=np.float64)

    mu_y = class_channel_means[int(yhat)]
    mu_r = class_channel_means[int(y_runner)]
    for c in range(n_channels):
        sep_c = float(abs(mu_y[c] - mu_r[c]))
        for bi, (t0, t1) in enumerate(t_slices):
            cid = _component_idx(c, bi, n_bins)
            v = x[t0:t1, c].astype(np.float64)
            variance[cid] = float(np.var(v))
            dy = v - float(mu_y[c])
            dr = v - float(mu_r[c])
            dist_yhat[cid] = float(np.sqrt(np.mean(dy * dy)))
            dist_run[cid] = float(np.sqrt(np.mean(dr * dr)))
            class_sep[cid] = sep_c

    conflict = dist_yhat - dist_run
    closeness_run = -dist_run
    risk_profile = _z(dist_yhat) + _z(closeness_run)
    risk_margin = _z(np.abs(conflict) * class_sep) + _z(closeness_run)
    risk_combined = _z(dist_yhat) + _z(closeness_run) + _z(class_sep) + _z(variance)
    return risk_profile, risk_margin, risk_combined


def _adaptive_old_score(
    x: np.ndarray,
    t_slices: list[tuple[int, int]],
    channel_means: np.ndarray | None,
) -> np.ndarray:
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
    return _z(energy) + _z(variance) + _z(profile_dist) + _z(mean_dev)


def _compute_selected_from_score(
    score: np.ndarray,
    q: int,
) -> np.ndarray:
    return np.asarray(np.argsort(-score)[: min(int(q), score.size)], dtype=np.int64)


def _compute_metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    p10, r10, f10 = eval_at_budget(y, s, 0.10)
    p20, r20, f20 = eval_at_budget(y, s, 0.20)
    return {
        "auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else float("nan"),
        "auprc": float(average_precision_score(y, s)),
        "f1_10": f10,
        "f1_20": f20,
        "precision_10": p10,
        "recall_10": r10,
        "precision_20": p20,
        "recall_20": r20,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    q_values = [int(v.strip()) for v in str(args.q_values).split(",") if v.strip()]
    if args.q_max is not None:
        q_values = [int(args.q_max)]
    selected_methods = {m.strip() for m in str(args.methods).split(",") if m.strip()}

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
    n_components = n_channels * n_bins
    global_channel_means = np.mean(x_train, axis=(0, 1)).astype(np.float32)
    class_channel_means: dict[int, np.ndarray] = {}
    for cls in np.unique(y_train):
        class_channel_means[int(cls)] = np.mean(x_train[y_train == cls], axis=(0, 1)).astype(np.float32)

    target_pos = args.n_total // 2
    target_neg = args.n_total - target_pos
    idx_all = np.arange(len(x_test), dtype=np.int64)
    rng.shuffle(idx_all)
    positives, pos_src, pos_cls = [], [], []
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

    man = {
        "seed": int(args.seed),
        "n_total": int(len(y_det)),
        "time_bins": int(args.time_bins),
        "n_components": int(n_components),
        "train_ids": [int(v) for v in tr.tolist()],
        "val_ids": [int(v) for v in va.tolist()],
        "test_ids": [int(v) for v in te.tolist()],
        "source_sample_ids": [int(v) for v in src_ids.tolist()],
        "source_class_ids": [int(v) for v in src_cls.tolist()],
    }
    with (out_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    rows_diag = []
    rows_corr = []
    rows_res = []
    rows_boot = []
    methods_all = [
        "uniform",
        "adaptive_v2_old",
        "risk_preselect_profile",
        "risk_preselect_margin",
        "risk_preselect_combined",
    ]
    methods = [m for m in methods_all if m in selected_methods]
    if "uniform" not in methods:
        methods = ["uniform"] + methods
    methods = [m for m in methods if m in methods_all]
    if len(methods) < 2:
        raise ValueError("Need at least two methods, including uniform.")

    cache = []
    for i in range(len(y_det)):
        x = x_det[i]
        lg0 = clf.logits(x)
        yhat = int(np.argmax(lg0))
        tmp = lg0.copy()
        tmp[yhat] = -1e18
        y_runner = int(np.argmax(tmp))
        full_deltas, m0 = _compute_full_deltas(
            x=x,
            clf=clf,
            t_slices=t_slices,
            neutralizer_mode=args.neutralizer_mode,
            channel_means=global_channel_means if args.neutralizer_mode in ("mean", "channel_mean", "class_mean") else None,
        )
        s_adapt = _adaptive_old_score(
            x=x,
            t_slices=t_slices,
            channel_means=global_channel_means if args.neutralizer_mode in ("mean", "channel_mean", "class_mean") else None,
        )
        s_prof, s_marg, s_comb = _risk_scores(
            x=x,
            t_slices=t_slices,
            yhat=yhat,
            y_runner=y_runner,
            class_channel_means=class_channel_means,
        )
        cache.append(
            {
                "m0": m0,
                "full_deltas": full_deltas,
                "adaptive_v2_old": s_adapt,
                "risk_preselect_profile": s_prof,
                "risk_preselect_margin": s_marg,
                "risk_preselect_combined": s_comb,
            }
        )

    for q in q_values:
        feature_rows: dict[str, list[dict]] = {m: [] for m in methods}
        pool_score_abs: dict[str, list[float]] = {m: [] for m in methods}
        pool_score_delta: dict[str, list[float]] = {m: [] for m in methods}
        pool_abs_delta: dict[str, list[float]] = {m: [] for m in methods}
        pool_delta: dict[str, list[float]] = {m: [] for m in methods}

        for i in range(len(y_det)):
            y_lbl = int(y_det[i])
            full_deltas = cache[i]["full_deltas"]
            m0 = float(cache[i]["m0"])
            oracle = np.argsort(-full_deltas)[: min(q, n_components)]
            selected = {}
            selected["uniform"] = np.asarray(
                np.random.default_rng(args.seed + 1000 + q * 11 + i).choice(n_components, size=min(q, n_components), replace=False),
                dtype=np.int64,
            )
            for m in ("adaptive_v2_old", "risk_preselect_profile", "risk_preselect_margin", "risk_preselect_combined"):
                selected[m] = _compute_selected_from_score(cache[i][m], q=q)

            for m in methods:
                sel = selected[m]
                d = np.zeros(n_components, dtype=np.float64)
                d[sel] = full_deltas[sel]
                top = np.sort(d[d > 0])[::-1]
                top1 = float(top[0]) if top.size > 0 else 0.0
                top3 = float(np.sum(top[:3])) if top.size > 0 else 0.0
                overlap = float(len(set(sel.tolist()) & set(oracle.tolist())) / max(1, len(sel)))
                share_pos = float(np.mean(d[sel] > 0.0)) if len(sel) > 0 else 0.0
                share_nonpos = float(np.mean(d[sel] <= 0.0)) if len(sel) > 0 else 0.0
                rows_diag.append(
                    {
                        "sample_id": int(i),
                        "q_max": int(q),
                        "method": m,
                        "is_hidden_conflict": int(y_lbl),
                        "mean_abs_delta_selected": float(np.mean(np.abs(d[sel]))) if len(sel) > 0 else 0.0,
                        "sum_negative_evidence": float(np.sum(np.clip(d[sel], 0.0, None))),
                        "top1_negative_delta": top1,
                        "top3_negative_mass": top3,
                        "share_delta_pos": share_pos,
                        "share_delta_nonpos": share_nonpos,
                        "overlap_with_oracle_topq": overlap,
                    }
                )
                feature_rows[m].append(
                    extract_audit_vector(
                        beacon_result=None,
                        margin=m0,
                        q_max=q,
                        sample_id=i,
                        label=y_lbl,
                        is_hidden_conflict=y_lbl,
                        method=m,
                        seed=args.seed,
                        deltas=d,
                    )
                )
                if m != "uniform":
                    pool_score_abs[m].extend(cache[i][m].tolist())
                    pool_score_delta[m].extend(cache[i][m].tolist())
                    pool_abs_delta[m].extend(np.abs(full_deltas).tolist())
                    pool_delta[m].extend(full_deltas.tolist())

        for m in ("adaptive_v2_old", "risk_preselect_profile", "risk_preselect_margin", "risk_preselect_combined"):
            if m not in methods:
                continue
            sx = np.asarray(pool_score_abs[m], dtype=float)
            ax = np.asarray(pool_abs_delta[m], dtype=float)
            dx = np.asarray(pool_delta[m], dtype=float)
            px = np.asarray(pool_score_delta[m], dtype=float)
            pear_abs = float(np.corrcoef(sx, ax)[0, 1]) if sx.size > 1 else float("nan")
            pear_del = float(np.corrcoef(px, dx)[0, 1]) if px.size > 1 else float("nan")
            # rank corr via argsort-based ranks (fast, no scipy dependency)
            rs = np.argsort(np.argsort(sx))
            ra = np.argsort(np.argsort(ax))
            rd = np.argsort(np.argsort(dx))
            sp_abs = float(np.corrcoef(rs, ra)[0, 1]) if sx.size > 1 else float("nan")
            sp_del = float(np.corrcoef(rs, rd)[0, 1]) if sx.size > 1 else float("nan")
            rows_corr.append(
                {
                    "q_max": int(q),
                    "method": m,
                    "pearson_score_vs_abs_delta": pear_abs,
                    "pearson_score_vs_delta": pear_del,
                    "spearman_score_vs_abs_delta": sp_abs,
                    "spearman_score_vs_delta": sp_del,
                }
            )

        scores_te = {}
        for m in methods:
            dfm = pd.DataFrame(feature_rows[m]).set_index("sample_id").sort_index()
            X = dfm[
                [
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
                ]
            ].to_numpy(dtype=float)
            logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
            logit.fit(X[tr], y_det[tr])
            s_te = logit.predict_proba(X[te])[:, 1]
            scores_te[m] = s_te
            row = {"q_max": int(q), "method": m, "n_test": int(len(te))}
            row.update(_compute_metrics(y_det[te], s_te))
            rows_res.append(row)

        metric_fns = {
            "delta_auroc": lambda yy, ss: float(roc_auc_score(yy, ss)) if len(np.unique(yy)) >= 2 else float("nan"),
            "delta_auprc": lambda yy, ss: float(average_precision_score(yy, ss)),
            "delta_f1_10": _f1_budget(0.10),
            "delta_f1_20": _f1_budget(0.20),
            "delta_precision_10": lambda yy, ss: float(eval_at_budget(yy, ss, 0.10)[0]),
            "delta_recall_10": lambda yy, ss: float(eval_at_budget(yy, ss, 0.10)[1]),
        }
        for m in ("adaptive_v2_old", "risk_preselect_profile", "risk_preselect_margin", "risk_preselect_combined"):
            if m not in methods:
                continue
            for metric_name, fn in metric_fns.items():
                d, lo, hi, p = _bootstrap_delta(
                    y_det[te],
                    scores_te[m],
                    scores_te["uniform"],
                    fn,
                    n_boot=args.n_boot,
                    seed=args.seed + q * 101 + abs(hash((m, metric_name))) % 100000,
                )
                rows_boot.append(
                    {
                        "q_max": int(q),
                        "comparison": f"{m}_vs_uniform",
                        "metric": metric_name,
                        "delta": d,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p_value": p,
                        "n_test": int(len(te)),
                    }
                )

    pd.DataFrame(rows_diag).to_csv(out_dir / "preselect_diagnostics.csv", index=False)
    pd.DataFrame(rows_corr).to_csv(out_dir / "preselect_score_delta_correlation.csv", index=False)
    pd.DataFrame(rows_res).to_csv(out_dir / "risk_preselect_results.csv", index=False)
    df_boot = pd.DataFrame(rows_boot)
    df_boot.to_csv(out_dir / "risk_preselect_bootstrap.csv", index=False)

    # Claim registry focused on method vs uniform.
    claims = []
    q_over_m = {int(q): float(int(q) / max(1, n_components)) for q in q_values}
    for _, r in df_boot.iterrows():
        comp = str(r["comparison"])
        if not comp.endswith("_vs_uniform"):
            continue
        method = comp.replace("_vs_uniform", "")
        delta = float(r["delta"])
        lo = float(r["ci_low"])
        hi = float(r["ci_high"])
        pval = float(r["p_value"])
        qv = int(r["q_max"])
        supported = bool((lo > 0.0) and (pval < 0.05))
        usable = bool(supported and q_over_m.get(qv, 1.0) <= 0.5)
        claims.append(
            {
                "setting": f"tb{args.time_bins}_q{qv}_{args.neutralizer_mode}",
                "method": method,
                "comparison": "vs_uniform",
                "metric": str(r["metric"]),
                "delta": delta,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": pval,
                "q_over_m": q_over_m.get(qv, float("nan")),
                "supported_positive": int(supported),
                "usable_for_budget_claim": int(usable),
            }
        )
    pd.DataFrame(claims).to_csv(out_dir / "risk_preselect_claim_registry.csv", index=False)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
