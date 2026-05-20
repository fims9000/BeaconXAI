#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train improved fuzzy policy (sigmoid memberships, Sugeno 0-order)")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--n-rules", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--compare-to", choices=["logit_panel", "uniform"], default="logit_panel")
    p.add_argument("--target", choices=["binary", "ce", "ordinal"], default="binary")
    p.add_argument("--ce-quantile", type=float, default=0.65)
    p.add_argument("--sample-weight", action="store_true")
    p.add_argument("--weight-alpha", type=float, default=5.0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-model", default="fuzzy_improved_model.json")
    p.add_argument("--out-results", default="fuzzy_improved_results.csv")
    p.add_argument("--out-bootstrap", default="fuzzy_improved_bootstrap.csv")
    return p.parse_args()


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "delta_entropy" not in d.columns and "rank_entropy" in d.columns:
        d["delta_entropy"] = d["rank_entropy"]
    if "margin_entropy" not in d.columns:
        m = -d["m_neg"].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-m))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        d["margin_entropy"] = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    # Extended conflict descriptors from existing audit columns.
    t3c = np.clip(d.get("top3_conflict_count", 0.0).to_numpy(dtype=float), 0.0, 3.0)
    t3s = np.maximum(d.get("top3_sum_delta", 0.0).to_numpy(dtype=float), 0.0)
    c1 = np.maximum(d.get("top1_delta", 0.0).to_numpy(dtype=float), 0.0)
    ce = np.maximum(d.get("CE_B", 0.0).to_numpy(dtype=float), 0.0)
    mb = np.maximum(d.get("M_B_minus", 0.0).to_numpy(dtype=float), 0.0)
    frag = d.get("frag_drop", 0.0).to_numpy(dtype=float)
    denom = np.maximum(t3c, 1.0)
    d["mean_conflict"] = t3s / denom
    d["var_conflict_proxy"] = np.maximum(c1 - d["mean_conflict"].to_numpy(dtype=float), 0.0)
    d["frac_conflict_top3"] = t3c / 3.0
    d["fragility_gap"] = frag - ce
    d["ce_density"] = ce / (mb + 1e-6)
    # New v9 descriptors; fallback derivation keeps old bundles runnable.
    if "var_conflict" not in d.columns:
        d["var_conflict"] = d["var_conflict_proxy"]
    if "conflict_connectivity" not in d.columns:
        d["conflict_connectivity"] = d["frac_conflict_top3"]
    if "delta_frag_proxy" not in d.columns:
        d["delta_frag_proxy"] = d["fragility_gap"]
    if "r_cf" not in d.columns:
        d["r_cf"] = mb / (np.maximum(d.get("rho_B_cost", 1.0).to_numpy(dtype=float), 1e-6))
    return d


def _build_ordinal_targets(y_ce: np.ndarray, tr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y_ce = np.maximum(np.asarray(y_ce, dtype=float), 0.0)
    tr_pos = y_ce[tr_idx][y_ce[tr_idx] > 1e-12]
    med_pos = float(np.median(tr_pos)) if tr_pos.size > 0 else 0.0
    y_ord = np.zeros_like(y_ce, dtype=np.int64)
    weak = (y_ce > 1e-12) & (y_ce <= med_pos)
    strong = y_ce > med_pos
    y_ord[weak] = 1
    y_ord[strong] = 2
    y_high = (y_ord == 2).astype(np.int64)
    return y_ord, y_high, med_pos


def _f1_10(y: np.ndarray, s: np.ndarray) -> float:
    n = len(y)
    k = max(1, int(np.ceil(0.10 * n)))
    order = np.argsort(-s)
    yp = np.zeros(n, dtype=np.int64)
    yp[order[:k]] = 1
    tp = float(np.sum((yp == 1) & (y == 1)))
    fp = float(np.sum((yp == 1) & (y == 0)))
    fn = float(np.sum((yp == 0) & (y == 1)))
    p = tp / max(1.0, tp + fp)
    r = tp / max(1.0, tp + fn)
    return 0.0 if p + r == 0 else float(2 * p * r / (p + r))


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def _bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        da = fn(y[idx], a[idx])
        db = fn(y[idx], b[idx])
        if np.isfinite(da) and np.isfinite(db):
            vals.append(float(da - db))
    arr = np.asarray(vals, dtype=float)
    p = 2.0 * min(float(np.mean(arr < 0.0)), float(np.mean(arr > 0.0)))
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(min(1.0, max(0.0, p)))


def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError(f"torch is required for improved fuzzy training: {e}")

    bdir = Path(args.bundle_dir)
    df_b = _prepare(pd.read_csv(bdir / "audit_features_beacon_core.csv")).set_index("sample_id").sort_index()
    df_u = _prepare(pd.read_csv(bdir / "audit_features_uniform.csv")).set_index("sample_id").sort_index()
    with (bdir / "split_manifest.json").open("r", encoding="utf-8") as f:
        man = json.load(f)
    tr = np.asarray(man["train_ids"], dtype=np.int64)
    va = np.asarray(man["val_ids"], dtype=np.int64)
    te = np.asarray(man["test_ids"], dtype=np.int64)

    y_eval_default = df_b["is_hidden_conflict"].to_numpy(dtype=np.int64)
    y_ce = np.maximum(df_b["CE_B"].to_numpy(dtype=float), 0.0)
    if args.target == "ce":
        q = float(np.clip(args.ce_quantile, 0.05, 0.95))
        thr_ce = float(np.quantile(y_ce[tr], q))
        y_train_target = (y_ce >= thr_ce).astype(np.int64)
        y_eval = y_eval_default
        ordinal_median_pos = float("nan")
    elif args.target == "ordinal":
        y_train_target, y_eval, ordinal_median_pos = _build_ordinal_targets(y_ce, tr)
        thr_ce = float("nan")
    else:
        thr_ce = float("nan")
        y_train_target = y_eval_default
        y_eval = y_eval_default
        ordinal_median_pos = float("nan")
    if args.sample_weight:
        sw_all = 1.0 + float(args.weight_alpha) * y_ce
    else:
        sw_all = np.ones_like(y_ce, dtype=float)

    feature_cols = ["r_B_minus", "frag_drop", "m_neg", "margin_entropy"]
    X = np.stack(
        [
            df_b["r_B_minus"].to_numpy(dtype=float),
            df_b["frag_drop"].to_numpy(dtype=float),
            (-df_b["m_neg"].to_numpy(dtype=float)),
            df_b["margin_entropy"].to_numpy(dtype=float),
        ],
        axis=1,
    ).astype(np.float32)

    # normalize on train
    mu = np.mean(X[tr], axis=0, keepdims=True)
    sd = np.std(X[tr], axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    class SigmoidSugeno(torch.nn.Module):
        def __init__(self, n_in: int, x_train_norm: np.ndarray, n_rules: int = 16, seed: int = 42):
            super().__init__()
            self.n_in = int(n_in)
            self.n_terms = 2  # low, high
            q10 = np.quantile(x_train_norm, 0.10, axis=0).astype(np.float32)
            q90 = np.quantile(x_train_norm, 0.90, axis=0).astype(np.float32)
            self.c_low = torch.nn.Parameter(torch.tensor(q10))
            self.c_high = torch.nn.Parameter(torch.tensor(q90))
            self.k_low = torch.nn.Parameter(torch.full((self.n_in,), 5.0))
            self.k_high = torch.nn.Parameter(torch.full((self.n_in,), 5.0))

            # full rule grid 2^n_in; optionally keep subset for lightweight mode
            terms = []
            total = int(self.n_terms ** self.n_in)
            for idx in range(total):
                v = idx
                term = []
                for _ in range(self.n_in):
                    term.append(v % self.n_terms)
                    v //= self.n_terms
                terms.append(term)
            terms = np.asarray(terms, dtype=np.int64)
            keep = max(1, min(int(n_rules), int(len(terms))))
            if keep < len(terms):
                rng_rules = np.random.default_rng(seed)
                pick = np.sort(rng_rules.choice(len(terms), size=keep, replace=False))
                terms = terms[pick]
            self.register_buffer("rule_terms", torch.tensor(terms, dtype=torch.long))
            self.rule_w = torch.nn.Parameter(torch.ones(keep))
            self.rule_out = torch.nn.Parameter(torch.zeros(keep))

        def _mf(self, x):
            # x [N,n_in]
            low = torch.sigmoid(-(F.softplus(self.k_low)) * (x - self.c_low))
            high = torch.sigmoid((F.softplus(self.k_high)) * (x - self.c_high))
            return torch.stack([low, high], dim=2)  # [N,n_in,2]

        def forward(self, x):
            mf = self._mf(x)
            n = mf.shape[0]
            r = self.rule_terms.shape[0]
            idx = self.rule_terms.view(1, r, self.n_in, 1).expand(n, r, self.n_in, 1)
            mf_e = mf.unsqueeze(1).expand(n, r, self.n_in, self.n_terms)
            chosen = torch.gather(mf_e, dim=3, index=idx).squeeze(-1)  # [N,R,4]
            acts = torch.prod(chosen, dim=2)
            w = torch.softmax(self.rule_w, dim=0)
            out = torch.sigmoid(self.rule_out)
            num = torch.sum(acts * w.unsqueeze(0) * out.unsqueeze(0), dim=1)
            den = torch.sum(acts * w.unsqueeze(0), dim=1) + 1e-8
            return torch.clamp(num / den, 1e-6, 1 - 1e-6)

    torch.manual_seed(args.seed)
    model = SigmoidSugeno(n_in=Xn.shape[1], x_train_norm=Xn[tr], n_rules=args.n_rules, seed=args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    Xtr = torch.tensor(Xn[tr], dtype=torch.float32)
    ytr_bin = torch.tensor(y_train_target[tr].astype(np.float32))
    Xva = torch.tensor(Xn[va], dtype=torch.float32)
    yva_bin = torch.tensor(y_train_target[va].astype(np.float32))
    Xte = torch.tensor(Xn[te], dtype=torch.float32)
    ce_scale = float(np.quantile(y_ce[tr], 0.95) + 1e-6)
    ytr_ce = torch.tensor(np.clip(y_ce[tr] / ce_scale, 0.0, 1.0).astype(np.float32))
    yva_ce = torch.tensor(np.clip(y_ce[va] / ce_scale, 0.0, 1.0).astype(np.float32))
    ytr_ord = torch.tensor((y_train_target[tr].astype(np.float32) / 2.0))
    yva_ord = torch.tensor((y_train_target[va].astype(np.float32) / 2.0))
    wtr = torch.tensor(sw_all[tr].astype(np.float32))
    wva = torch.tensor(sw_all[va].astype(np.float32))

    best = None
    best_f1 = -1.0
    best_loss = 1e12
    patience = max(5, int(args.patience))
    bad = 0
    rng = np.random.default_rng(args.seed)
    for _ in range(args.epochs):
        idx = rng.permutation(len(tr))
        for i in range(0, len(tr), args.batch_size):
            b = idx[i : i + args.batch_size]
            pred = model(Xtr[b])
            if args.target == "ce":
                lvec = (pred - ytr_ce[b]) ** 2
            elif args.target == "ordinal":
                lvec = (pred - ytr_ord[b]) ** 2
            else:
                lvec = F.binary_cross_entropy(pred, ytr_bin[b], reduction="none")
            wb = wtr[b]
            loss = torch.sum(lvec * wb) / (torch.sum(wb) + 1e-8)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        with torch.no_grad():
            pred_v = model(Xva)
            sv = pred_v.numpy().astype(float)
            if args.target == "ce":
                lvec_v = (pred_v - yva_ce) ** 2
            elif args.target == "ordinal":
                lvec_v = (pred_v - yva_ord) ** 2
            else:
                lvec_v = F.binary_cross_entropy(pred_v, yva_bin, reduction="none")
            lv = float((torch.sum(lvec_v * wva) / (torch.sum(wva) + 1e-8)).item())
            fv = _f1_10(y_eval[va], sv)
        if (fv > best_f1 + 1e-6) or (abs(fv - best_f1) <= 1e-6 and lv < best_loss - 1e-5):
            best_f1 = fv
            best_loss = lv
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best is not None:
        model.load_state_dict(best)

    with torch.no_grad():
        s = model(Xte).numpy().astype(float)
        s_val = model(Xva).numpy().astype(float)

    # Platt calibration on validation for fair probability scale.
    calib = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.seed)
    y_cal = y_eval[va] if args.target == "ordinal" else y_train_target[va]
    calib.fit(s_val.reshape(-1, 1), y_cal, sample_weight=sw_all[va])
    s = calib.predict_proba(s.reshape(-1, 1))[:, 1]

    panel_cols = [
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
    ]
    Xb_panel = df_b.loc[:, panel_cols].to_numpy(dtype=float)
    Xu_panel = df_u.loc[:, panel_cols].to_numpy(dtype=float)
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed),
    )
    if args.compare_to == "uniform":
        fit_kwargs = {}
        if args.sample_weight:
            fit_kwargs = {"logisticregression__sample_weight": sw_all[tr]}
        logit.fit(Xu_panel[tr], y_train_target[tr], **fit_kwargs)
        p_base = logit.predict_proba(Xu_panel[te])
        if args.target == "ordinal":
            cls = list(np.unique(y_train_target[tr]))
            idx_hi = cls.index(2) if 2 in cls else int(np.argmax(cls))
            s_base = p_base[:, idx_hi]
        else:
            s_base = p_base[:, 1]
        base_name = "uniform_logit_panel"
    else:
        fit_kwargs = {}
        if args.sample_weight:
            fit_kwargs = {"logisticregression__sample_weight": sw_all[tr]}
        logit.fit(Xb_panel[tr], y_train_target[tr], **fit_kwargs)
        p_base = logit.predict_proba(Xb_panel[te])
        if args.target == "ordinal":
            cls = list(np.unique(y_train_target[tr]))
            idx_hi = cls.index(2) if 2 in cls else int(np.argmax(cls))
            s_base = p_base[:, idx_hi]
        else:
            s_base = p_base[:, 1]
        base_name = "logit_panel"

    pd.DataFrame(
        [
            {
                "bundle": bdir.name,
                "policy": "fuzzy_improved_sigmoid",
                "target": args.target,
                "ce_quantile": args.ce_quantile,
                "ce_threshold_train": thr_ce,
                "ce_median_pos_train": ordinal_median_pos,
                "sample_weight": int(args.sample_weight),
                "weight_alpha": float(args.weight_alpha),
                "feature_cols": ",".join(feature_cols),
                "n_rules": args.n_rules,
                "compare_to": base_name,
                "f1_10_test": _f1_10(y_eval[te], s),
                "f1_10_baseline": _f1_10(y_eval[te], s_base),
                "auroc_test": _auroc(y_eval[te], s),
                "auprc_test": _auprc(y_eval[te], s),
                "auroc_baseline": _auroc(y_eval[te], s_base),
                "auprc_baseline": _auprc(y_eval[te], s_base),
            }
        ]
    ).to_csv(bdir / args.out_results, index=False)

    rows_boot = []
    metric_seeds = {"delta_auroc": 101, "delta_auprc": 211, "delta_f1_10": 307}
    for mname, fn in (("delta_auroc", _auroc), ("delta_auprc", _auprc), ("delta_f1_10", _f1_10)):
        d, lo, hi, p = _bootstrap_delta(y_eval[te], s, s_base, fn, args.n_boot, args.seed + metric_seeds[mname])
        rows_boot.append(
            {"bundle": bdir.name, "comparison": f"fuzzy_vs_{base_name}", "metric": mname, "delta": d, "ci_low": lo, "ci_high": hi, "p_value": p}
        )
    pd.DataFrame(rows_boot).to_csv(bdir / args.out_bootstrap, index=False)

    # Persist parameters for downstream export and reproducibility.
    payload = {
        "bundle": bdir.name,
        "policy": "fuzzy_improved_sigmoid",
        "target": args.target,
        "ce_quantile": args.ce_quantile,
        "ce_threshold_train": thr_ce,
        "ce_median_pos_train": ordinal_median_pos,
        "sample_weight": int(args.sample_weight),
        "weight_alpha": float(args.weight_alpha),
        "ce_scale_q95": ce_scale,
        "feature_cols": feature_cols,
        "n_rules": int(args.n_rules),
        "normalization": {"mu": mu.flatten().tolist(), "sd": sd.flatten().tolist()},
        "state_dict": {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()},
    }
    with (bdir / args.save_model).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"saved: {bdir / args.out_results}")
    print(f"saved: {bdir / args.out_bootstrap}")
    print(f"saved: {bdir / args.save_model}")


if __name__ == "__main__":
    main()
