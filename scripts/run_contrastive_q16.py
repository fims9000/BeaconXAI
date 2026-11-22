#!/usr/bin/env python3
"""Equal-Q PAMAP2 localization comparison and contrastive audit (research prototype).

Drop this file into the existing BeaconXAI/scripts/ directory and the module
into BeaconXAI/beaconxai/. Run from the repository root. Do not call data from
toy smoke tests "PAMAP2 results".

Question answered: can target-class contrastive cheap preselection improve finding
an injected component at equal 16 *additional* model evaluations?

Limits: RISE-like/KernelSHAP-like are the source repository's *mask regression*
variants, not necessarily standard unregularized estimators. They are rank-
deficient for Q=16 and M=36 and require seed stability checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beaconxai.contrastive import (
    fit_class_profiles, make_components, selected_audit, prior_ranking,
    contrastive_prior, neutralize_interp, targeted_prototype_probe,
)


def shap_kernel(m: int, s: int) -> float:
    if s == 0 or s == m:
        return 1e6
    return (m - 1) / (math.comb(m, s) * s * (m - s))


def masking_coalitions(method: str, m: int, q: int, rng: np.random.Generator):
    """Reproduces the baseline mask protocols in run_pamap2_tsxai_baselines.py."""
    a = np.zeros((q, m), dtype=np.float64)
    if method == "rise_like":
        for i in range(1, q):
            k = int(rng.integers(1, m + 1))
            a[i, rng.choice(m, size=k, replace=False)] = 1
        return a, None
    if method != "kernelshap_like":
        raise ValueError(method)
    weights = np.ones(q, dtype=np.float64)
    start = 1
    weights[0] = shap_kernel(m, 0)
    if q >= 2:
        a[1] = 1.0
        weights[1] = shap_kernel(m, m)
        start = 2
    for i in range(start, q):
        k = int(rng.integers(1, m))
        a[i, rng.choice(m, size=k, replace=False)] = 1.0
        weights[i] = shap_kernel(m, k)
    return a, weights


def ridge_coefficients(a: np.ndarray, responses: np.ndarray,
                       weights: np.ndarray | None, ridge: float) -> np.ndarray:
    """Stable ridge linear surrogate. Coefficients are not exact Shapley values."""
    x = np.column_stack((np.ones(len(a)), a))
    w = np.ones(len(a)) if weights is None else weights
    aw = x * np.sqrt(w)[:, None]
    yw = responses * np.sqrt(w)
    diag = np.sqrt(ridge) * np.eye(x.shape[1])
    diag[0, 0] = 0.0
    beta, *_ = np.linalg.lstsq(np.concatenate([aw, diag]),
                              np.concatenate([yw, np.zeros(x.shape[1])]), rcond=None)
    return beta[1:]


def mask_ranking(x: np.ndarray, model_batch_logits, components, a: int, b: int,
                 baseline_gap: float, method: str, q: int, seed: int,
                 ridge: float = 1e-3) -> list[int]:
    # Empty-mask prediction is already available as the shared baseline.
    # Create Q+1 rows, reuse row 0 from cache, and evaluate ONLY Q new masks.
    masks, weights = masking_coalitions(method, len(components), q+1, np.random.default_rng(seed))
    xb = np.repeat(x[None], q, axis=0)
    # Independent same-original interpolation patch per selected component.
    # Baseline mask is cached; the other Q masks are the only evaluations.
    patches = [neutralize_interp(x, c) for c in components]
    for row, mask in enumerate(masks[1:]):
        for idx in np.flatnonzero(mask):
            c = components[idx]
            s = np.s_[c.t0:c.t1, c.c0:c.c1]
            xb[row][s] = patches[idx][s]
    logits = np.asarray(model_batch_logits(xb), dtype=float)
    responses = np.r_[0.0, (logits[:, a] - logits[:, b]) - baseline_gap]
    coefs = ridge_coefficients(masks, responses, weights, ridge)
    return np.argsort(-np.abs(coefs), kind="stable").astype(int).tolist()


def ranking_metrics(rank: list[int], true_idx: int, n_components: int) -> dict[str, float]:
    # A component never evaluated by a shortlist must not be given an invented
    # tie-broken rank among unqueried components. Assign conservative M+1.
    r = rank.index(true_idx) + 1 if true_idx in rank else n_components + 1
    return {"hit1": int(r <= 1), "hit3": int(r <= 3), "hit5": int(r <= 5), "mrr": 1.0/r, "rank": r}


def bootstrap_mean(values: np.ndarray, seed: int, B: int = 1000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(0, len(values), (B, len(values)))], axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def toy_smoke() -> None:
    """A deliberately simple deterministic check; NEVER report as real benchmark."""
    rng = np.random.default_rng(7)
    train = rng.normal(0, .3, (80, 96, 9))
    labels = np.array([0]*40+[1]*40)
    train[:40, 32:48, :3] += 1.7
    train[40:, 32:48, :3] -= 1.7
    x = train[3].copy()
    x[32:40, :3] = train[55, 32:40, :3]
    comps = make_components(x.shape, 12, "pamap3")
    profiles = fit_class_profiles(train, labels)
    def model(s):
        # model depends on one sensor/time region, class response is explicitly known
        delta = np.mean(s[32:48, :3])
        return np.array([delta, -delta])
    baseline = model(x)
    a = int(np.argmax(baseline)); b = 1-a
    audit = selected_audit(x, model, profiles, comps, 16, 13, logits0=baseline)
    uni = selected_audit(x, model, profiles, comps, 16, 13, selection="uniform", logits0=baseline)
    assert audit.calls == uni.calls == 16
    assert len(set(audit.component_ids)) == 16
    assert np.isfinite(audit.scores[audit.component_ids]).all()
    print("SYNTHETIC SMOKE TEST ONLY -- no PAMAP2 data")
    print("Original predicted class",a,"rival",b)
    print("Contrastive selected", audit.component_ids)
    print("Uniform selected", uni.component_ids)
    print("Model queries (additional)",audit.calls)



def run_boundary(args, x_test, y_test, profiles, components, logits_one, logits_batch):
    """Separate pairwise decision-probe experiment, not corrupt-component localization.

    Keep only correctly classified held-out examples, select class B in advance
    as runner-up unless a fixed pair is given. Class profiles are TRAIN-only.
    """
    clean_logits=logits_batch(x_test)
    correct=np.flatnonzero(np.argmax(clean_logits,axis=1)==y_test)
    if args.source_class is not None:
        correct=np.array([i for i in correct if int(y_test[i])==args.source_class],dtype=int)
    if (args.source_class is None) != (args.target_class is None):
        raise ValueError("--source-class and --target-class must be set together")
    if not len(correct):
        raise ValueError("no correctly predicted examples of the requested class")
    rows=[]
    for trial in range(args.repeats):
        for i in correct:
            sample=x_test[i]; base=clean_logits[i]
            seed=args.seed+trial*1000003+int(i)*271
            for method in ("contrastive","uniform"):
                t0=time.perf_counter()
                r=targeted_prototype_probe(sample,logits_one,profiles,components,args.budget,seed,
                                            selection=method,logits0=base,fixed_target=args.target_class)
                rows.append({"sample":int(i),"trial":trial,"method":method,
                             "source":r.source_class,"target":r.target_class,
                             "baseline_gap":r.baseline_gap,
                             "best_gap_reduction":r.best_target_gap_reduction,
                             "best_component":r.best_component,
                             "target_flip_any":int(r.flip_to_target),
                             "extra_model_calls":r.extra_model_calls,
                             "latency_sec":time.perf_counter()-t0})
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    with (out/"boundary_per_sample.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary={"experiment":"targeted single-component training-prototype intervention",
             "note":"This probes the classifier; it does NOT establish a physical class boundary",
             "n_unique_test_examples":len(correct),"budget_extra":args.budget,"repeats":args.repeats,
             "results":[]}
    n=len(correct);pos={int(i):j for j,i in enumerate(correct)}
    for method in ("contrastive","uniform"):
        hits=np.zeros((args.repeats,n));drops=np.zeros((args.repeats,n))
        for r in rows:
            if r['method']!=method:continue
            idx=pos[r['sample']]
            hits[r['trial'],idx]=r['target_flip_any']
            drops[r['trial'],idx]=r['best_gap_reduction']
        mu,lo,hi=bootstrap_mean(hits.mean(axis=0),args.seed+3)
        dm,dl,dh=bootstrap_mean(drops.mean(axis=0),args.seed+13)
        summary['results'].append({"method":method,"target_flip_rate":mu,"flip_ci95":[lo,hi],
                                   "best_gap_reduction":dm,"reduction_ci95":[dl,dh]})
    (out/"boundary_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("BOUNDARY PROBES (not validated physical boundaries):")
    for row in summary['results']:
        print(row['method'],"targeted flip rate",round(row['target_flip_rate'],4),
              "mean best gap reduction",round(row['best_gap_reduction'],4))

def real_run(args):
    from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
    from scripts.run_component_conflict_benchmark import _inject_component_conflicts, _train_extratrees_local

    if not args.npz_path.exists():
        raise FileNotFoundError(f"PAMAP2 missing: {args.npz_path}. Prepare it via DATA_PREPARATION.md")
    x_train, y_train, x_test, y_test = load_npz_dataset(str(args.npz_path))
    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)
    if 0 < args.max_test < len(x_test):
        idx = np.random.default_rng(args.seed).choice(len(x_test), args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]
    classifier = _train_extratrees_local(x_train,y_train,args.trees,.7,1)
    components = make_components(x_test.shape[1:], args.time_bins, args.group_mode)
    if args.budget > len(components):
        raise ValueError("budget exceeds number of components")
    g_slices = [(c.c0,c.c1) for c in components if c.time_bin == 0]
    t_slices = [(c.t0,c.t1) for c in components if c.group == 0]
    x_eval, injected, true_comp = _inject_component_conflicts(
        x_test, y_test, g_slices, t_slices, args.conflict_ratio, args.seed + 101,
    )
    ix = np.flatnonzero(injected == 1)
    if not len(ix):
        raise ValueError("No injected samples")
    x_conf = x_eval[ix]; y_true_comp = true_comp[ix]
    profiles = fit_class_profiles(x_train, y_train)
    def logits_one(sample):
        return classifier.logits(sample)
    def logits_batch(batch):
        from scripts.run_component_conflict_benchmark import _anfis_features
        probs = classifier.model.predict_proba(_anfis_features(batch))
        return np.log(np.clip(probs, 1e-12, 1.0))
    ref_logits = logits_batch(x_conf)
    if args.mode in ("boundary", "both"):
        run_boundary(args, x_test, y_test, profiles, components, logits_one, logits_batch)
    if args.mode == "boundary":
        return
    methods = ["contrastive_q16", "uniform_q16", "prior_q0", "rise_like_q16", "kernelshap_like_q16"]
    rows=[]
    for trial in range(args.repeats):
        for i, sample in enumerate(x_conf):
            true_idx = int(y_true_comp[i])
            base = ref_logits[i]
            a = int(np.argmax(base)); bs = base.copy(); bs[a] = -np.inf; b = int(np.argmax(bs))
            g0 = float(base[a]-base[b])
            sample_seed = args.seed + trial*1000003 + i*271
            t0=time.perf_counter()
            contrast = selected_audit(sample,logits_one,profiles,components,args.budget,sample_seed,
                                      selection="contrastive", explore_fraction=args.explore_fraction,
                                      logits0=base)
            t_contrast=time.perf_counter()-t0
            t0=time.perf_counter()
            uniform = selected_audit(sample,logits_one,profiles,components,args.budget,sample_seed,
                                     selection="uniform",logits0=base)
            t_uniform=time.perf_counter()-t0
            t0=time.perf_counter()
            prior_order=prior_ranking(contrast.prior)
            t_prior=time.perf_counter()-t0
            t0=time.perf_counter()
            rise_order=mask_ranking(sample,logits_batch,components,a,b,g0,"rise_like",args.budget,sample_seed)
            t_rise=time.perf_counter()-t0
            t0=time.perf_counter()
            shap_order=mask_ranking(sample,logits_batch,components,a,b,g0,"kernelshap_like",args.budget,sample_seed)
            t_shap=time.perf_counter()-t0
            elapsed={"contrastive_q16":t_contrast,"uniform_q16":t_uniform,"prior_q0":t_prior,
                     "rise_like_q16":t_rise,"kernelshap_like_q16":t_shap}
            ranks = {
                "contrastive_q16": contrast.ranked(),
                "uniform_q16": uniform.ranked(),
                "prior_q0": prior_order,
                "rise_like_q16": rise_order,
                "kernelshap_like_q16": shap_order,
            }
            for method in methods:
                # Unqueried components are scored as misses; no arbitrary padding.
                met = ranking_metrics(ranks[method],true_idx,len(components))
                rows.append({"trial": trial,"sample":i,"method":method,"source":a,"rival":b,
                             "true_component":true_idx,"budget_extra":0 if method=="prior_q0" else args.budget,
                             "latency_sec":elapsed[method],**met})
        print(f"completed trial {trial+1}/{args.repeats} ({len(x_conf)} injected test examples)")

    args.out_dir.mkdir(parents=True,exist_ok=True)
    fper = args.out_dir/"per_sample.csv"
    with fper.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    # Cluster bootstrap by sample (mean across seeds first) avoids treating repeated runs as independent.
    summary=[]; paired=[]
    n=len(x_conf)
    for method in methods:
        rm=[r for r in rows if r["method"]==method]
        vals={key:np.zeros((args.repeats,n),float) for key in ("hit1","hit3","hit5","mrr","rank","latency_sec")}
        for row in rm:
            for key in vals: vals[key][row["trial"],row["sample"]]=row[key]
        res={"method":method,"n_unique_samples":n,"repeats":args.repeats,
             "M":len(components),"budget_extra":0 if method=="prior_q0" else args.budget}
        for key, val in vals.items():
            mu,lo,hi=bootstrap_mean(val.mean(axis=0),args.seed+12)
            res[key]=mu;res[key+"_ci_low"]=lo;res[key+"_ci_high"]=hi
        summary.append(res)
        if method=="contrastive_q16": contrast_hits=vals["hit3"].mean(axis=0)
        if method=="uniform_q16": uniform_hits=vals["hit3"].mean(axis=0)
    paired_delta = contrast_hits-uniform_hits
    delta,lo,hi = bootstrap_mean(paired_delta,args.seed+99)
    metadata={"data":"PAMAP2","source": "prepared locally, not bundled", "seed":args.seed,
              "model":"ExtraTrees on repository's _anfis_features",
              "n_test_injected":len(x_conf),"component_count":len(components),
              "group_mode":args.group_mode,"time_bins":args.time_bins,
              "budget":args.budget, "baseline_model_calls":1,
              "train_profile_storage_bytes":sum(a.nbytes+b.nbytes for a,b in profiles.values()),
              "neutralization":"interp", "experiment":"artificial injected component localization",
              "target":"fixed top-1 class vs fixed runner-up per sample",
              "comparison_warning":"RISE-like and KernelSHAP-like are 16-query ridge mask regressions, not exact Shapley",
              "paired_delta_contrastive_minus_uniform_hit3":{"mean":delta,"CI95":[lo,hi]},
              "results":summary}
    (args.out_dir/"summary.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    with (args.out_dir/"summary.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    print("DATA RESULTS:", args.out_dir)
    for r in summary: print(f"{r['method']:22s} Q={r['budget_extra']:2d} hit3={r['hit3']:.3f} [{r['hit3_ci_low']:.3f},{r['hit3_ci_high']:.3f}]")
    print(f"paired contrastive-uniform hit3 delta={delta:+.3f} CI [{lo:+.3f},{hi:+.3f}]")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="synthetic sanity only")
    p.add_argument("--npz-path",type=Path,default=Path("data/pamap2_acc9_w200s100_p095.npz"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-test",type=int,default=128)
    p.add_argument("--mode", choices=["localization","boundary","both"], default="both")
    p.add_argument("--source-class",type=int,default=None,help="optional class pair for boundary case study")
    p.add_argument("--target-class",type=int,default=None,help="optional class pair for boundary case study")
    p.add_argument("--conflict-ratio",type=float,default=.5)
    p.add_argument("--group-mode",choices=["pamap3","per_channel"],default="pamap3")
    p.add_argument("--time-bins",type=int,default=12)
    p.add_argument("--budget",type=int,default=16)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--explore-fraction",type=float,default=.25)
    p.add_argument("--out-dir",type=Path,default=Path("outputs_composite/contrastive_q16"))
    args=p.parse_args()
    if args.smoke: toy_smoke()
    else: real_run(args)


if __name__=="__main__":main()
