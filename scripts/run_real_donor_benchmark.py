#!/usr/bin/env python3
"""Matched-Q real-donor vs mean/median prototype intervention study for PAMAP2.

Single global Q per object PER method; one source class and the runner-up.
A target-class substitute is evaluated per selected component. Baseline is shared.
Uses TRAIN class labels for donors but no held-out labels for selecting interventions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

from beaconxai.contrastive import make_components
from beaconxai.real_donor import (fit_real_donor_bank, make_target_replacement,
                                  plausibility_proxies)

METHODS = ('mean', 'median', 'medoid', 'random_donor', 'context_donor', 'aligned_donor')


def select_cases(y: np.ndarray, pred: np.ndarray, n_per_class: int, n_errors: int,
                 max_samples: int, seed: int) -> np.ndarray:
    if n_per_class < 1 or n_errors < 0:
        raise ValueError('incorrect case quotas')
    rng = np.random.default_rng(seed)
    ids = []
    for cls in np.unique(pred):
        good = np.flatnonzero((pred == cls) & (pred == y))
        bad = np.flatnonzero((pred == cls) & (pred != y))
        ids.extend(rng.choice(good, min(len(good), n_per_class), replace=False).tolist())
        ids.extend(rng.choice(bad, min(len(bad), n_errors), replace=False).tolist())
    if len(ids) > max_samples > 0:
        # Stratified sampling is already complete; cap uniformly if requested.
        ids = rng.choice(ids, max_samples, replace=False).tolist()
    return np.asarray(sorted(set(ids)), dtype=int)


def evaluate_case(x, base, bank, components, selection, source, rival,
                  method, predict_many, seed=42, taper=2, max_shift=5):
    """Return one row per actual model evaluation, one shared baseline for all methods."""
    if len(selection) != len(set(selection)):
        raise ValueError('repeat component IDs')
    t_start = time.perf_counter()
    edits, meta, realism = [], [], []
    for ix, cid in enumerate(selection):
        sample, details = make_target_replacement(x, components[cid], cid, rival,
                method, bank, components, seed=seed+1009*ix,
                taper=taper,max_shift=max_shift)
        edits.append(sample)
        meta.append(details)
        realism.append(plausibility_proxies(x,sample,components[cid],bank,rival,components))
    logits = np.asarray(predict_many(np.stack(edits)),dtype=float)
    if logits.shape != (len(selection),len(base)):
        raise ValueError('model output shape changed')
    if not np.isfinite(logits).all():
        raise ValueError('nonfinite model output')
    # Existing ExtraTrees adapter returns log predicted probabilities.
    p0 = np.exp(base - np.logaddexp.reduce(base))
    p1 = np.exp(logits - np.logaddexp.reduce(logits,axis=1,keepdims=True))
    gap0 = float(base[source]-base[rival])
    items = []
    for ix, cid in enumerate(selection):
        lp = logits[ix]
        pa,pb = float(p1[ix,source]),float(p1[ix,rival])
        da,db = pa-float(p0[source]), pb-float(p0[rival])
        dgap = float((lp[source]-lp[rival])-gap0)
        out = dict(method=method, component=int(cid), source=int(source),rival=int(rival),
                   prob_source_before=float(p0[source]),prob_rival_before=float(p0[rival]),
                   delta_prob_source=da,delta_prob_rival=db,
                   delta_gap=dgap,final_pred=int(np.argmax(lp)),
                   directed=bool(da < -1e-9 and db > 1e-9),
                   flip_to_rival=bool(int(np.argmax(lp)) == rival),
                   **meta[ix],**realism[ix])
        items.append(out)
    latency=time.perf_counter()-t_start
    return items, latency


def summarize_case(rows, latency: float) -> dict:
    arr = np.asarray([x['delta_gap'] for x in rows])
    best = int(np.argmin(arr))
    return dict(queries=len(rows), directed_fraction=float(np.mean([r['directed'] for r in rows])),
                directed_any=int(any(r['directed'] for r in rows)),
                flip_any=int(any(r['flip_to_rival'] for r in rows)),
                best_gap_reduction=float(-arr[best]),
                best_component=int(rows[best]['component']),
                mean_relative_l2=float(np.mean([r['relative_l2'] for r in rows])),
                mean_target_fp_nn_dist=float(np.mean([r['target_fp_nn_dist'] for r in rows])),
                mean_seam_excess=float(np.mean([r['seam_excess'] for r in rows])),
                wall_ms=float(latency*1000))


def aggregate(records:list[dict], key_fields:tuple[str,...]):
    grouped=defaultdict(list)
    for r in records:
        grouped[tuple(r[k] for k in key_fields)].append(r)
    ans=[]
    for key, rows in sorted(grouped.items()):
        values=dict(zip(key_fields,key))
        values.update(n_examples=len(rows),
                      mean_flip_any=float(np.mean([r['flip_any'] for r in rows])),
                      mean_directed_any=float(np.mean([r['directed_any'] for r in rows])),
                      mean_directed_fraction=float(np.mean([r['directed_fraction'] for r in rows])),
                      mean_best_gap_reduction=float(np.mean([r['best_gap_reduction'] for r in rows])),
                      mean_target_fp_nn_dist=float(np.mean([r['mean_target_fp_nn_dist'] for r in rows])),
                      mean_relative_l2=float(np.mean([r['mean_relative_l2'] for r in rows])),
                      mean_seam_excess=float(np.mean([r['mean_seam_excess'] for r in rows])),
                      median_wall_ms=float(np.median([r['wall_ms'] for r in rows])))
        ans.append(values)
    return ans


def save_csv(path:Path,items):
    if not items:
        return
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]))
        w.writeheader();w.writerows(items)


def run(args):
    from beaconxai.datasets import apply_standardizer,fit_channel_standardizer,load_npz_dataset
    from scripts.run_component_conflict_benchmark import _anfis_features,_train_extratrees_local
    from scripts.run_q16_matched_ablation import class_cheap,best_q
    path=Path(args.npz_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    xt,yt,xv,yv = load_npz_dataset(path)
    with np.load(path) as f:
        if 'subject_test' not in f:
            raise ValueError('NPZ lacks subject_test')
        sid=f['subject_test']
    mu,std=fit_channel_standardizer(xt)
    xt=apply_standardizer(xt,mu,std)
    xv=apply_standardizer(xv,mu,std)
    components=make_components(xv.shape[1:],args.time_bins,'pamap3')
    if args.q<1 or args.q>len(components):
        raise ValueError('Q must be in [1,M]')
    t0=time.perf_counter()
    model=_train_extratrees_local(xt,yt,args.trees,.7,1)
    train_sec=time.perf_counter()-t0
    classes=np.asarray(model.model.classes_,dtype=int)
    n_classes=int(max(yt.max(),yv.max(),classes.max()))+1
    def predict_many(batch):
        probs=model.model.predict_proba(_anfis_features(batch))
        full=np.full((len(batch),n_classes),1e-12)
        full[:,classes]=probs
        return np.log(np.clip(full,1e-12,1))
    baseline_all=predict_many(xv)
    preds=np.argmax(baseline_all,axis=1)
    sel=select_cases(yv,preds,args.n_per_class,args.n_errors,args.max_samples,args.seed)
    bank=fit_real_donor_bank(xt,yt,components,max_donors=args.max_donors,seed=args.seed+31)
    all_rows=[];cases=[]
    for index in sel:
        x=xv[index]; base=baseline_all[index]
        source=int(preds[index])
        if source not in bank.labels:
            continue
        rival=int(np.argsort(-base)[1]);
        if rival not in bank.labels:
            continue
        ranked=class_cheap(x,bank.means[source],components)[2]
        component_ids=best_q(ranked,args.q)
        for method in METHODS:
            rows,elapsed=evaluate_case(x,base,bank,components,component_ids,
                                       source,rival,method,predict_many,
                                       seed=args.seed+int(index)*131,
                                       taper=args.taper,max_shift=args.max_shift)
            for row in rows:
                row.update(sample=int(index),subject=int(sid[index]),
                           true_class=int(yv[index]),is_correct=int(source==yv[index]),
                           time_bin=int(components[row['component']].time_bin),
                           sensor_group=int(components[row['component']].group))
                all_rows.append(row)
            cases.append(dict(sample=int(index),subject=int(sid[index]),
                              true_class=int(yv[index]),is_correct=int(source==yv[index]),
                              source=source,rival=rival,method=method,
                              **summarize_case(rows,elapsed)))
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'per_intervention.csv',all_rows)
    save_csv(out/'per_case.csv',cases)
    save_csv(out/'summary.csv',aggregate(cases,('method',)))
    save_csv(out/'per_subject.csv',aggregate(cases,('subject','method')))
    save_csv(out/'per_correctness.csv',aggregate(cases,('is_correct','method')))
    save_csv(out/'per_class.csv',aggregate(cases,('source','method')))
    info=dict(dataset='PAMAP2',npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              train_size=len(xt),test_size=len(xv),test_subjects=sorted(map(int,np.unique(sid))),
              sampled_cases=len(sel),completed_cases=len({c['sample'] for c in cases}),
              model='ExtraTrees on _anfis_features',model_train_sec=train_sec,
              methods=list(METHODS),q_per_object_per_method=args.q,
              one_shared_baseline_eval_per_object=True,
              total_extra_model_evaluations=int(sum(c['queries'] for c in cases)),
              total_shared_baseline_evaluations=len(xv),
              max_donors_per_class=args.max_donors,donor_bank_bytes=bank.bank_bytes,
              train_class_counts=bank.counts, taper=args.taper,max_shift=args.max_shift,
              class_selection='predicted top1 and runner-up, never test label',
              selected_components='original adaptive_v2 cheap ranking identical across methods',
              caution=['All metrics are exploratory model perturbation effects',
                       'Mean/median and real donor fragments can be off-manifold at patch seams',
                       'Nearest donor search requires stored training donor examples, not shown TinyML feasible',
                       'Fingerprint nearest-neighbor distance is a proxy, not a validated realism score',
                       'Overlapping windows and participant dependence prohibit naive iid confidence intervals'])
    (out/'manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf8')
    lines=['# Real-donor vs mean/median class interventions','',
           'Each method: '+str(args.q)+' additional model evaluations per test object.',
           'Class selection and component selection use no held-out labels.',
           'A directed change means p(source) decreased AND p(rival) increased.',
           'A flip means argmax(model output) is the target rival.',
           'These are NOT validated natural counterfactuals or causal explanations.','',
           '## Per-method descriptive results','',
           '| Method | N | Any switch | Any directed | Best mean gap reduction | FP-distance | seam excess |',
           '|---|---:|---:|---:|---:|---:|---:|']
    summary=aggregate(cases,('method',))
    for r in summary:
        lines.append(f"| {r['method']} | {r['n_examples']} | {r['mean_flip_any']:.1%} | "
                     f"{r['mean_directed_any']:.1%} | {r['mean_best_gap_reduction']:+.3f} | "
                     f"{r['mean_target_fp_nn_dist']:.3f} | {r['mean_seam_excess']:+.3f} |")
    lines+=['','## Interpretation','',
            'A full-class score may change while the target score does not. Check both.',
            'A nearest donor can be selected without a model query; its lookup cost and bank memory still count.',
            'Aligned donor means bounded shift that minimizes seam mismatch; it is NOT DTW.',
            'Only the runner-up target class was examined for each object; the budget is global per object.',
            'Compare per_subject.csv and per_correctness.csv before claiming reliability.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('MATCHED DONOR RESULTS',out,flush=True)
    for r in summary:
        print(f"{r['method']:18s} N={r['n_examples']:3d} any_flip={r['mean_flip_any']:.3f} "
              f"direction={r['mean_directed_any']:.3f} best_gap={r['mean_best_gap_reduction']:.3f} "
              f"bank_bytes={bank.bank_bytes}",flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz-path',type=Path,default=Path('data/pamap2_acc9_w200s100_p095.npz'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs_composite/donor_counterfactual'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--q',type=int,default=16)
    p.add_argument('--time-bins',type=int,default=12)
    p.add_argument('--n-per-class',type=int,default=5)
    p.add_argument('--n-errors',type=int,default=2)
    p.add_argument('--max-samples',type=int,default=84)
    p.add_argument('--max-donors',type=int,default=64)
    p.add_argument('--trees',type=int,default=120)
    p.add_argument('--taper',type=int,default=2)
    p.add_argument('--max-shift',type=int,default=5)
    run(p.parse_args())

if __name__=='__main__':
    cli()
