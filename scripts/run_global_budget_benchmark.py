#!/usr/bin/env python3
"""One-global-budget multi-rival study on held-out PAMAP2 subjects.

All policies see the same test cases, original prediction, class medoids,
cheap component order, and exactly Q extra model calls except stop-on-flip.
Each class candidate is a separate cumulative edited input. This is not a
physically validated counterfactual or a learned/calibrated stopping rule.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.contrastive import make_components
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.budget_policy import run_budget_policy,POLICIES
from scripts.run_real_donor_benchmark import select_cases,save_csv


def run(args):
    from beaconxai.datasets import fit_channel_standardizer,apply_standardizer,load_npz_dataset
    from scripts.run_component_conflict_benchmark import _anfis_features,_train_extratrees_local
    from scripts.run_q16_matched_ablation import class_cheap,best_q
    path=Path(args.npz_path)
    if not path.is_file():raise FileNotFoundError(path)
    xt,yt,xv,yv=load_npz_dataset(path)
    with np.load(path) as z:
        if 'subject_test' not in z:raise ValueError('missing subject_test')
        subjects=z['subject_test']
    mu,std=fit_channel_standardizer(xt)
    xt=apply_standardizer(xt,mu,std);xv=apply_standardizer(xv,mu,std)
    comps=make_components(xv.shape[1:],args.time_bins,'pamap3')
    if args.q<4 or args.q%4 or args.q>len(comps):raise ValueError('Q must be multiple of 4 and <=M')
    t0=time.perf_counter()
    model=_train_extratrees_local(xt,yt,args.trees,.7,1)
    train_sec=time.perf_counter()-t0
    cls=np.asarray(model.model.classes_,dtype=int)
    n_classes=int(max(cls.max(),yt.max(),yv.max()))+1
    def p_many(xs):
        raw=model.model.predict_proba(_anfis_features(np.asarray(xs)))
        full=np.full((len(xs),n_classes),1e-12)
        full[:,cls]=raw
        return np.log(np.clip(full,1e-12,1.0))
    baseline=p_many(xv)
    preds=np.argmax(baseline,axis=1)
    chosen=select_cases(yv,preds,args.n_per_class,args.n_errors,args.max_samples,args.seed)
    bank=fit_real_donor_bank(xt,yt,comps,max_donors=args.max_donors,seed=args.seed+11)
    actions=[];cases=[]
    for ix in chosen:
        x=xv[ix];base=baseline[ix];a=int(preds[ix])
        if a not in bank.labels:continue
        rivals=np.argsort(-base)[1:3]
        if any(int(b) not in bank.labels for b in rivals):continue
        cheap=class_cheap(x,bank.means[a],comps)[2]
        ordered=best_q(cheap,args.q)
        def p_one(edited):return p_many(edited[None])[0]
        for policy in POLICIES:
            t_start=time.perf_counter()
            rr,res=run_budget_policy(x,base,bank,comps,ordered,p_one,
                     q=args.q,policy=policy,donor_kind=args.donor_kind,
                     seed=args.seed+131*int(ix),taper=args.taper)
            elapsed=(time.perf_counter()-t_start)*1000
            assert 1<=res['queries']<=args.q
            sample_info=dict(sample=int(ix),subject=int(subjects[ix]),
                             true_class=int(yv[ix]),is_correct=int(a==yv[ix]),
                             wall_ms=elapsed,donor_kind=args.donor_kind)
            cases.append({**sample_info,**res})
            actions.extend({**sample_info,'policy':policy,**r} for r in rr)
    if not cases:raise ValueError('no valid test cases')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'per_action.csv',actions)
    save_csv(out/'per_case.csv',cases)
    def aggr(keys):
        grouped=defaultdict(list)
        for r in cases:grouped[tuple(r[k] for k in keys)].append(r)
        result=[]
        for k,rs in sorted(grouped.items()):
            result.append({**dict(zip(keys,k)),
               'n_examples':len(rs),
               'flip_any':float(np.mean([r['target_flip_any'] for r in rs])),
               'directed_any':float(np.mean([r['directed_any'] for r in rs])),
               'mean_queries':float(np.mean([r['queries'] for r in rs])),
               'abstain_rate':float(np.mean([r['status']=='abstain' for r in rs])),
               'best_gap_reduction':float(np.mean([r['best_gap_reduction'] for r in rs])),
               'mean_relative_l2':float(np.mean([r['mean_relative_l2'] for r in rs])),
               'mean_seam_excess':float(np.mean([r['mean_seam_excess'] for r in rs])),
               'median_wall_ms':float(np.median([r['wall_ms'] for r in rs]))})
        return result
    summ=aggr(('policy',))
    save_csv(out/'summary.csv',summ)
    save_csv(out/'per_subject.csv',aggr(('subject','policy')))
    save_csv(out/'per_correctness.csv',aggr(('is_correct','policy')))
    meta=dict(dataset='PAMAP2',data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              test_subjects=sorted(map(int,np.unique(subjects))),n_test_selected=len(chosen),
              n_explained=len({r['sample'] for r in cases}),
              model='ExtraTrees with _anfis_features',train_sec=train_sec,
              global_budget=args.q,donor_kind=args.donor_kind,bank_bytes=bank.bank_bytes,
              test_label_never_used_in_selection=True,
              both_targets_separate_cumulative_trajectories=True,
              model_queries_extra_total=int(sum(r['queries'] for r in cases)),
              baseline_model_calls=len(xv),
              rules={'top1_full':'all Q toward original runner-up',
                     'equal':'Q/2 to runner-up and third class',
                     'prior_3to1':'3Q/4 toward runner-up, Q/4 third',
                     'adaptive':'Q/4 to each, remaining Q/2 to higher observed contrastive reward',
                     'adaptive_stop':'same as adaptive, stop immediately on any observed target-class flip'},
              limitations=['True class never used for choosing target or fragments',
                           'Cumulative edits may become off-manifold; medoid is real but assembled input is artificial',
                           'No learned or calibrated stopping threshold',
                           'The study measures candidate model responses, not human semantic explanation fidelity',
                           'No TinyML hardware, power or memory measurement'])
    (out/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# Global query budget across two competing classes','',
           'All policies start from the same input and original model prediction.',
           'A separate cumulative sequence is maintained for each rival.',
           'Success means an actual predicted switch to that intended rival',
           'at any measured step (not necessarily the final candidate).','',
           '| Policy | N | Any target flip | Directed evidence | Mean Q | Abstention |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summ:
        lines.append(f"| {r['policy']} | {r['n_examples']} | {r['flip_any']:.1%} | "
                     f"{r['directed_any']:.1%} | {r['mean_queries']:.2f} | {r['abstain_rate']:.1%} |")
    lines+=['','This is a deliberately simple sequential bandit-like heuristic. '
            'It does NOT establish a new algorithmic contribution or calibrated reliability.',
            'Use per_subject.csv and per_correctness.csv to check generalization.',
            'More than one target is supported under ONE global Q per case, unlike the original Q16-per-pair atlas.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('GLOBAL BUDGET RESULTS',out)
    for r in summ:
        print(f"{r['policy']:20s} flip={r['flip_any']:.3f} directed={r['directed_any']:.3f} "
              f"Q={r['mean_queries']:.2f} n={r['n_examples']}",flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz-path',type=Path,default=Path('data/pamap2_acc9_w200s100_p095.npz'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs_composite/global_budget'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--q',type=int,default=16)
    p.add_argument('--time-bins',type=int,default=12)
    p.add_argument('--n-per-class',type=int,default=4)
    p.add_argument('--n-errors',type=int,default=2)
    p.add_argument('--max-samples',type=int,default=72)
    p.add_argument('--max-donors',type=int,default=32)
    p.add_argument('--donor-kind',choices=('medoid','context_donor','aligned_donor','random_donor'),default='medoid')
    p.add_argument('--taper',type=int,default=2)
    p.add_argument('--trees',type=int,default=120)
    run(p.parse_args())

if __name__=='__main__':cli()
