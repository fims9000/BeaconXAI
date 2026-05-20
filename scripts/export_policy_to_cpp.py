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

from beaconxai.tan_policy import fit_tan_policy


FEATURE_COLS = [
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
TAN_FEATURES_DEFAULT = ["m_neg", "M_B_minus", "r_B_minus", "CE_B", "rho_B_cost", "frag_drop"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export compact policy layer to C++ header (logit/fuzzy/tan)")
    p.add_argument("--bundle-dir", required=True, help="Directory with split_manifest.json and audit_features_*.csv")
    p.add_argument("--out-header", default="embedded/beacon_policy.h")
    p.add_argument("--n-rules", type=int, default=7)
    p.add_argument("--n-terms", type=int, default=3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tan-bins", type=int, default=4)
    p.add_argument("--tan-alpha", type=float, default=1.0)
    p.add_argument("--tan-strategy", default="quantile")
    p.add_argument("--include-fuzzy", action="store_true", default=True)
    p.add_argument("--no-fuzzy", dest="include_fuzzy", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "delta_entropy" not in d.columns and "rank_entropy" in d.columns:
        d["delta_entropy"] = d["rank_entropy"]
    if "margin_entropy" not in d.columns:
        m = -d["m_neg"].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-m))
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        d["margin_entropy"] = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return d


def _arr1(a: np.ndarray) -> str:
    return "{" + ", ".join(f"{float(x):.8g}f" for x in a.reshape(-1)) + "}"


def _arr1_i(a: np.ndarray) -> str:
    return "{" + ", ".join(str(int(x)) for x in a.reshape(-1)) + "}"


def _arr2(a: np.ndarray) -> str:
    return "{" + ", ".join("{" + ", ".join(f"{float(x):.8g}f" for x in r) + "}" for r in a) + "}"


def _arr2_i(a: np.ndarray) -> str:
    return "{" + ", ".join("{" + ", ".join(str(int(x)) for x in r) + "}" for r in a) + "}"


def _arr3(a: np.ndarray) -> str:
    return "{" + ", ".join(_arr2(b) for b in a) + "}"


def _arr4(a: np.ndarray) -> str:
    return "{" + ", ".join(_arr3(b) for b in a) + "}"


def main() -> None:
    args = parse_args()
    bdir = Path(args.bundle_dir)
    out_header = Path(args.out_header)
    out_header.parent.mkdir(parents=True, exist_ok=True)

    df_b = _prepare_df(pd.read_csv(bdir / "audit_features_beacon_core.csv")).set_index("sample_id").sort_index()
    with (bdir / "split_manifest.json").open("r", encoding="utf-8") as f:
        man = json.load(f)
    tr = np.asarray(man["train_ids"], dtype=np.int64)
    va = np.asarray(man["val_ids"], dtype=np.int64)
    y = df_b["is_hidden_conflict"].to_numpy(dtype=np.int64)
    X = df_b.loc[:, FEATURE_COLS].to_numpy(dtype=np.float32)

    # logit
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    logit.fit(X[tr], y[tr])
    scaler = logit.named_steps["standardscaler"]
    lr = logit.named_steps["logisticregression"]
    logit_w = (lr.coef_.reshape(-1) / np.maximum(scaler.scale_, 1e-12)).astype(np.float64)
    logit_b = float(lr.intercept_[0] - np.sum((lr.coef_.reshape(-1) * scaler.mean_) / np.maximum(scaler.scale_, 1e-12)))

    # fuzzy v5 (optional; requires torch)
    fuzzy_ok = False
    fuzzy_centers = np.zeros((len(FEATURE_COLS), args.n_terms), dtype=np.float64)
    fuzzy_sigmas = np.ones((len(FEATURE_COLS), args.n_terms), dtype=np.float64)
    fuzzy_rule_terms = np.zeros((args.n_rules, len(FEATURE_COLS)), dtype=np.int64)
    fuzzy_rule_w = np.ones(args.n_rules, dtype=np.float64)
    fuzzy_rule_out = np.full(args.n_rules, 0.5, dtype=np.float64)
    if args.include_fuzzy:
        try:
            import torch
            from beaconxai.fuzzy_policy_v5 import fit_fuzzy_policy_v5

            pol = fit_fuzzy_policy_v5(
                X[tr],
                y[tr],
                X[va],
                y[va],
                n_terms=args.n_terms,
                n_rules=args.n_rules,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=args.seed,
                device="cpu",
            )
            mdl = pol.model
            with torch.no_grad():
                fuzzy_centers = mdl.centers.detach().cpu().numpy().astype(np.float64)
                fuzzy_sigmas = (torch.nn.functional.softplus(mdl.log_sigmas).detach().cpu().numpy() + 1e-4).astype(np.float64)
                fuzzy_rule_terms = mdl.rule_terms.detach().cpu().numpy().astype(np.int64)
                fuzzy_rule_w = (torch.nn.functional.softplus(mdl.raw_rule_w).detach().cpu().numpy() + 1e-4).astype(np.float64)
                fuzzy_rule_out = torch.sigmoid(mdl.raw_rule_out).detach().cpu().numpy().astype(np.float64)
            fuzzy_ok = True
        except Exception as e:
            print(f"[warn] fuzzy export skipped: {e}")

    # tan
    tan_cols = TAN_FEATURES_DEFAULT
    Xt = df_b.loc[:, tan_cols].to_numpy(dtype=float)
    tan = fit_tan_policy(
        Xt[tr],
        y[tr],
        Xt[va],
        y[va],
        n_bins=args.tan_bins,
        alpha=args.tan_alpha,
        strategy=args.tan_strategy,
    )
    disc = tan["discretizer"]
    tmodel = tan["model"]
    n_bins = int(args.tan_bins)
    n_tan_features = len(tan_cols)
    tan_parent = np.asarray(tmodel.parent_, dtype=np.int64)
    tan_log_prior = np.asarray(tmodel.log_prior_, dtype=np.float64)
    tan_bins_per_feature = np.asarray(getattr(disc, "n_bins_", np.full(n_tan_features, n_bins)), dtype=np.int64)
    tan_thr = np.full((n_tan_features, n_bins - 1), np.inf, dtype=np.float64)
    for i, edges in enumerate(disc.bin_edges_):
        inner = np.asarray(edges[1:-1], dtype=np.float64)
        k = min(len(inner), n_bins - 1)
        tan_thr[i, :k] = inner[:k]
    tan_root = np.zeros((n_tan_features, 2, n_bins), dtype=np.float64)
    tan_cond = np.zeros((n_tan_features, 2, n_bins, n_bins), dtype=np.float64)
    for i in range(n_tan_features):
        if i in tmodel.root_tables_:
            tan_root[i] = np.asarray(tmodel.root_tables_[i], dtype=np.float64)
        if i in tmodel.cond_tables_:
            tan_cond[i] = np.asarray(tmodel.cond_tables_[i], dtype=np.float64)

    header = f"""#pragma once
#include <math.h>
#include <stdint.h>

namespace beacon_policy {{

static inline float sigmoidf(float x) {{
  if (x >= 0.0f) {{
    float z = expf(-x);
    return 1.0f / (1.0f + z);
  }}
  float z = expf(x);
  return z / (1.0f + z);
}}

static constexpr int kInputDim = {len(FEATURE_COLS)};
static constexpr int kFuzzyTerms = {args.n_terms};
static constexpr int kFuzzyRules = {args.n_rules};
static constexpr int kFuzzyEnabled = {1 if fuzzy_ok else 0};
static constexpr int kTanDim = {n_tan_features};
static constexpr int kTanBinsMax = {n_bins};

static const float LOGIT_W[kInputDim] = {_arr1(logit_w)};
static const float LOGIT_B = {logit_b:.10g}f;

static const float FUZZY_CENTERS[kInputDim][kFuzzyTerms] = {_arr2(fuzzy_centers)};
static const float FUZZY_SIGMAS[kInputDim][kFuzzyTerms] = {_arr2(fuzzy_sigmas)};
static const int32_t FUZZY_RULE_TERMS[kFuzzyRules][kInputDim] = {_arr2_i(fuzzy_rule_terms)};
static const float FUZZY_RULE_W[kFuzzyRules] = {_arr1(fuzzy_rule_w)};
static const float FUZZY_RULE_OUT[kFuzzyRules] = {_arr1(fuzzy_rule_out)};

static const int32_t TAN_PARENT[kTanDim] = {_arr1_i(tan_parent)};
static const int32_t TAN_BINS_PER_FEATURE[kTanDim] = {_arr1_i(tan_bins_per_feature)};
static const float TAN_LOG_PRIOR[2] = {_arr1(tan_log_prior)};
static const float TAN_THRESH[kTanDim][kTanBinsMax - 1] = {_arr2(tan_thr)};
static const float TAN_ROOT[kTanDim][2][kTanBinsMax] = {_arr3(tan_root)};
static const float TAN_COND[kTanDim][2][kTanBinsMax][kTanBinsMax] = {_arr4(tan_cond)};
static const int32_t TAN_IDX[kTanDim] = {{0, 1, 2, 3, 4, 5}};

static inline float logit_panel(const float a[kInputDim]) {{
  float s = LOGIT_B;
  for (int i = 0; i < kInputDim; ++i) s += LOGIT_W[i] * a[i];
  return sigmoidf(s);
}}

static inline float fuzzy_policy(const float a[kInputDim]) {{
  if (!kFuzzyEnabled) return 0.5f;
  float acts[kFuzzyRules];
  for (int r = 0; r < kFuzzyRules; ++r) {{
    float prod = 1.0f;
    for (int i = 0; i < kInputDim; ++i) {{
      int t = FUZZY_RULE_TERMS[r][i];
      float z = (a[i] - FUZZY_CENTERS[i][t]) / FUZZY_SIGMAS[i][t];
      prod *= expf(-0.5f * z * z);
    }}
    acts[r] = prod;
  }}
  float num = 0.0f;
  float den = 0.0f;
  for (int r = 0; r < kFuzzyRules; ++r) {{
    num += acts[r] * FUZZY_RULE_W[r] * FUZZY_RULE_OUT[r];
    den += acts[r] * FUZZY_RULE_W[r];
  }}
  if (den <= 1e-8f) return 0.5f;
  float y = num / den;
  if (y < 1e-6f) y = 1e-6f;
  if (y > 1.0f - 1e-6f) y = 1.0f - 1e-6f;
  return y;
}}

static inline int tan_bin(float x, const float thr[kTanBinsMax - 1], int n_bins_i) {{
  int b = 0;
  while (b < n_bins_i - 1 && x > thr[b]) ++b;
  return b;
}}

static inline float tan_policy(const float a[kInputDim]) {{
  int xb[kTanDim];
  for (int i = 0; i < kTanDim; ++i) xb[i] = tan_bin(a[TAN_IDX[i]], TAN_THRESH[i], TAN_BINS_PER_FEATURE[i]);
  float l0 = TAN_LOG_PRIOR[0];
  float l1 = TAN_LOG_PRIOR[1];
  for (int i = 0; i < kTanDim; ++i) {{
    int bi = xb[i];
    int p = TAN_PARENT[i];
    if (p < 0) {{
      l0 += TAN_ROOT[i][0][bi];
      l1 += TAN_ROOT[i][1][bi];
    }} else {{
      int bp = xb[p];
      l0 += TAN_COND[i][0][bp][bi];
      l1 += TAN_COND[i][1][bp][bi];
    }}
  }}
  float m = (l0 > l1) ? l0 : l1;
  float e0 = expf(l0 - m);
  float e1 = expf(l1 - m);
  return e1 / (e0 + e1);
}}

}}  // namespace beacon_policy
"""

    out_header.write_text(header, encoding="utf-8")
    print(f"saved: {out_header}")


if __name__ == "__main__":
    main()
