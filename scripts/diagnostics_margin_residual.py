#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset, load_uci_har
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import train_1dcnn, train_extratrees_stats
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig, LocalMetricRow, RiskEvalRow


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def _precision_at_10(y: np.ndarray, s: np.ndarray) -> float:
    k = max(1, int(np.ceil(0.10 * len(y))))
    idx = np.argsort(-s)[:k]
    return float(np.mean(y[idx] == 1))


def _mean_by_class(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    e = y == 1
    c = y == 0
    me = float(np.mean(s[e])) if np.any(e) else float("nan")
    mc = float(np.mean(s[c])) if np.any(c) else float("nan")
    return me, mc


def _stratified_split_idx(y: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr_idx = []
    va_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        va_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:])
    tr = np.concatenate(tr_idx)
    va = np.concatenate(va_idx)
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def _collect(rows: list[RiskEvalRow], local_rows: list[LocalMetricRow], q: int):
    br = {r.sample_id: r for r in rows if r.method == "beacon_refine" and r.q_max == q}
    nm = {r.sample_id: r for r in rows if r.method == "negative_margin" and r.q_max == 0}
    lm = {r.sample_id: r for r in local_rows if r.method == "beacon_refine" and r.q_max == q}
    ids = sorted(set(br).intersection(nm).intersection(lm))
    if not ids:
        return None
    y = np.array([br[i].is_error for i in ids], dtype=np.int64)
    neg_margin = np.array([nm[i].risk_score for i in ids], dtype=np.float64)
    mneg = np.array([lm[i].counter_mass for i in ids], dtype=np.float64)
    ce = np.array([lm[i].counter_evidence_gain for i in ids], dtype=np.float64)
    rho = np.array([lm[i].rho_b_cost for i in ids], dtype=np.float64)
    return y, neg_margin, mneg, ce, rho


def _fit_margin_conditional_expectation(margin: np.ndarray, cmass: np.ndarray, n_bins: int = 10):
    qs = np.quantile(margin, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    means = np.zeros(n_bins, dtype=np.float64)
    for j in range(n_bins):
        m = (margin > qs[j]) & (margin <= qs[j + 1])
        means[j] = float(np.mean(cmass[m])) if np.any(m) else float(np.mean(cmass))
    return qs, means


def _predict_conditional_mean(margin: np.ndarray, qs: np.ndarray, means: np.ndarray) -> np.ndarray:
    out = np.empty_like(margin, dtype=np.float64)
    for i, v in enumerate(margin):
        j = int(np.searchsorted(qs, v, side="right") - 1)
        j = max(0, min(j, len(means) - 1))
        out[i] = means[j]
    return out


def _fit_lambda(y: np.ndarray, m: np.ndarray, z: np.ndarray) -> float:
    best = 0.0
    best_auc = -1.0
    for lam in np.linspace(0.0, 2.0, 81):
        s = m + lam * z
        a = _auc(y, s)
        if np.isfinite(a) and a > best_auc:
            best_auc = a
            best = float(lam)
    return best


def _fit_logreg_score(yv: np.ndarray, Xv: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray:
    sc = StandardScaler()
    Xv2 = sc.fit_transform(Xv)
    Xt2 = sc.transform(Xt)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
        max_iter=1000,
    )
    clf.fit(Xv2, yv)
    return clf.predict_proba(Xt2)[:, 1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Margin-residual diagnostics for BEACON conflict signal")
    p.add_argument("--dataset", choices=["har", "pamap2"], required=True)
    p.add_argument("--model", choices=["cnn1d", "extratrees"], required=True)
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-val", type=int, default=512)
    p.add_argument("--max-test", type=int, default=512)
    p.add_argument("--cnn-epochs", type=int, default=8)
    p.add_argument("--pamap-npz", default="./data/pamap2_acc9_w200s100_p095.npz")
    p.add_argument("--har-root", default="./data")
    p.add_argument("--out-prefix", default="./outputs_composite/diagnostics_margin_residual")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "har":
        x_train_full, y_train_full, x_test, y_test = load_uci_har(args.har_root)
    else:
        x_train_full, y_train_full, x_test, y_test = load_npz_dataset(args.pamap_npz)

    tr_idx, va_idx = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]

    if args.max_val > 0 and args.max_val < len(x_va):
        rng = np.random.default_rng(args.seed + 10)
        idx = rng.choice(len(x_va), size=args.max_val, replace=False)
        x_va = x_va[idx]
        y_va = y_va[idx]
    if args.max_test > 0 and args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed + 20)
        idx = rng.choice(len(x_test), size=args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_tr)
    x_tr = apply_standardizer(x_tr, mu, sigma)
    x_va = apply_standardizer(x_va, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "cnn1d":
        clf = train_1dcnn(
            x_tr,
            y_tr,
            epochs=args.cnn_epochs,
            batch_size=256,
            lr=1e-3,
            label_smoothing=0.0,
            use_class_weights=True,
            tta_shifts=(0, 64) if args.dataset == "har" else (0, 50),
        )
    else:
        clf = train_extratrees_stats(x_tr, y_tr, n_estimators=1000, max_features=0.7, min_samples_leaf=1)

    train_margins = []
    for i in range(min(len(x_tr), 2000)):
        lg = clf.logits(x_tr[i])
        y_hat = int(np.argmax(lg))
        m = float(lg[y_hat] - np.max(np.delete(lg, y_hat)))
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0

    cfg = BeaconConfig(
        q_max=args.q_max,
        k0=4 if args.q_max == 8 else 8,
        l_min=4,
        k_pos=3,
        k_neg=3,
        q_frag_ratio=0.25,
        alpha=1.0,
        beta=0.5,
        gamma=1.0,
        tau_s=0.10,
        tau_m=tau_m,
        refinement_mode="mixed",
        partition_mode="time_only",
        risk_policy="rho_only",
    )
    neutralizer = Neutralizer(mode="zero", channel_means=np.zeros(x_tr.shape[-1], dtype=np.float32))

    rows_v, local_v, _ = evaluate_error_risk(
        x_test=x_va,
        y_test=y_va,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=cfg,
        q_values=[args.q_max],
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        methods={"negative_margin", "beacon_refine"},
    )
    rows_t, local_t, _ = evaluate_error_risk(
        x_test=x_test,
        y_test=y_test,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=cfg,
        q_values=[args.q_max],
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        methods={"negative_margin", "beacon_refine"},
    )

    cv = _collect(rows_v, local_v, args.q_max)
    ct = _collect(rows_t, local_t, args.q_max)
    if cv is None or ct is None:
        raise RuntimeError("No collected rows")
    yv, mv, cmv, cev, rhov = cv
    yt, mt, cmt, cet, rhot = ct

    # residual conflict
    qs, means = _fit_margin_conditional_expectation(mv, cmv, n_bins=10)
    mu_t = _predict_conditional_mean(mt, qs, means)
    c_perp_t = cmt - mu_t
    c_rel_t = cmt / np.maximum(mu_t, 1e-8)
    mu_v = _predict_conditional_mean(mv, qs, means)
    c_perp_v = cmv - mu_v
    c_rel_v = cmv / np.maximum(mu_v, 1e-8)

    # fit lambda on val
    lam_m = _fit_lambda(yv, mv, cmv)
    lam_cp = _fit_lambda(yv, mv, c_perp_v)
    lam_cr = _fit_lambda(yv, mv, c_rel_v)

    score_margin = mt
    score_m = mt + lam_m * cmt
    score_cp = mt + lam_cp * c_perp_t
    score_cr = mt + lam_cr * c_rel_t

    # correlations
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    diag_rows = [
        {"feature": "counter_mass", "corr_with_neg_margin": _corr(cmt, mt), "auc": _auc(yt, cmt)},
        {"feature": "ce", "corr_with_neg_margin": _corr(cet, mt), "auc": _auc(yt, cet)},
        {"feature": "rho_cost", "corr_with_neg_margin": _corr(rhot, mt), "auc": _auc(yt, rhot)},
        {"feature": "C_perp", "corr_with_neg_margin": _corr(c_perp_t, mt), "auc": _auc(yt, c_perp_t)},
        {"feature": "C_rel", "corr_with_neg_margin": _corr(c_rel_t, mt), "auc": _auc(yt, c_rel_t)},
        {"feature": "corr(C_perp,counter_mass)", "corr_with_neg_margin": _corr(c_perp_t, cmt), "auc": float("nan")},
        {"feature": "corr(C_rel,counter_mass)", "corr_with_neg_margin": _corr(c_rel_t, cmt), "auc": float("nan")},
        {"feature": "corr(C_perp,error)", "corr_with_neg_margin": _corr(c_perp_t, yt.astype(np.float64)), "auc": float("nan")},
        {"feature": "AUC(-C_perp)", "corr_with_neg_margin": float("nan"), "auc": _auc(yt, -c_perp_t)},
    ]

    score_m_ce = _fit_logreg_score(yv, np.stack([mv, cev], axis=1), np.stack([mt, cet], axis=1), args.seed + 31)
    score_m_cm_ce = _fit_logreg_score(
        yv, np.stack([mv, cmv, cev], axis=1), np.stack([mt, cmt, cet], axis=1), args.seed + 41
    )
    score_m_cm_cp = _fit_logreg_score(
        yv, np.stack([mv, cmv, c_perp_v], axis=1), np.stack([mt, cmt, c_perp_t], axis=1), args.seed + 51
    )
    score_m_cm_ce_cp = _fit_logreg_score(
        yv,
        np.stack([mv, cmv, cev, c_perp_v], axis=1),
        np.stack([mt, cmt, cet, c_perp_t], axis=1),
        args.seed + 61,
    )

    model_scores = [
        ("margin", score_margin),
        ("margin+counter_mass", score_m),
        ("margin+CE", score_m_ce),
        ("margin+counter_mass+CE", score_m_cm_ce),
        ("margin+counter_mass+C_perp", score_m_cm_cp),
        ("margin+counter_mass+CE+C_perp", score_m_cm_ce_cp),
        ("margin+C_perp", score_cp),
        ("margin-C_perp", mt - lam_cp * c_perp_t),
        ("margin+C_rel", score_cr),
    ]

    eval_rows = []
    sign_rows = []
    for name, s in model_scores:
        p10 = _precision_at_10(yt, s)
        p10_asc = _precision_at_10(yt, -s)
        auc_s = _auc(yt, s)
        auc_neg = _auc(yt, -s)
        me, mc = _mean_by_class(yt, s)
        sign_rows.append(
            {
                "method": name,
                "auc_score": auc_s,
                "auc_neg_score": auc_neg,
                "auc_sum": (auc_s + auc_neg) if np.isfinite(auc_s) and np.isfinite(auc_neg) else float("nan"),
                "mean_score_error": me,
                "mean_score_correct": mc,
                "risk_direction_ok": int(me > mc) if np.isfinite(me) and np.isfinite(mc) else -1,
                "p10_desc": p10,
                "p10_asc": p10_asc,
            }
        )
        eval_rows.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "q_max": args.q_max,
                "method": name,
                "auroc": auc_s,
                "p10": p10,
                "delta_p10_vs_margin": p10 - _precision_at_10(yt, score_margin),
            }
        )

    # margin-bins test for independent signal
    bins = np.quantile(mt, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins[0] -= 1e-12
    bins[-1] += 1e-12
    bin_rows = []
    for j in range(4):
        m = (mt > bins[j]) & (mt <= bins[j + 1])
        if np.sum(m) < 20:
            continue
        bin_rows.append(
            {
                "bin": j,
                "n": int(np.sum(m)),
                "auc_counter_mass": _auc(yt[m], cmt[m]),
                "auc_C_perp": _auc(yt[m], c_perp_t[m]),
                "auc_C_rel": _auc(yt[m], c_rel_t[m]),
            }
        )

    out_diag = Path(f"{args.out_prefix}_{args.dataset}_{args.model}_diag.csv")
    out_eval = Path(f"{args.out_prefix}_{args.dataset}_{args.model}_eval.csv")
    out_bins = Path(f"{args.out_prefix}_{args.dataset}_{args.model}_bins.csv")
    out_sign = Path(f"{args.out_prefix}_{args.dataset}_{args.model}_sign_audit.csv")
    for path, rows in [(out_diag, diag_rows), (out_eval, eval_rows), (out_bins, bin_rows), (out_sign, sign_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            if rows:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
    print("Saved:")
    print(out_diag)
    print(out_eval)
    print(out_bins)
    print(out_sign)


if __name__ == "__main__":
    main()
