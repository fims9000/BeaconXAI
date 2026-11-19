#!/usr/bin/env python3
"""Held-out donor-resampling benchmark for Q16 evidence consistency.

Q16 single donor checks 16 components; Q16 replicated checks 4 components x
4 independent TRAIN donors. Evaluation uses a DIFFERENT unused training
subset, eight donors on each of 16 components =128 OFFLINE extra model calls.
The reference measures donor-perturbation stability, NOT causal truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.contrastive import make_components
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.evidence_consensus import run_consensus
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
    xt=apply_standardizer(xt,mu,std)
    xv=apply_standardizer(xv,mu,std)
    comps=make_components(xv.shape[1:],args.time_bins,'pamap3')
    t0=time.perf_counter()
    if args.model=='extratrees':
        clf=_train_extratrees_local(xt,yt,args.trees,.7,1)
        classes=clf.model.classes_.astype(int)
        def model_probability(batch):
            return clf.model.predict_proba(_anfis_features(np.asarray(batch)))
    elif args.model=='raw_mlp':
        from sklearn.neural_network import MLPClassifier
        clf=MLPClassifier(hidden_layer_sizes=(32,),alpha=.01,
                          batch_size=128,max_iter=30,early_stopping=True,
                          n_iter_no_change=6,random_state=args.seed)
        clf.fit(xt.reshape(len(xt),-1),yt)
        classes=clf.classes_.astype(int)
        def model_probability(batch):
            arr=np.asarray(batch)
            return clf.predict_proba(arr.reshape(len(arr),-1))
    else:
        raise ValueError('unknown model')
    train_seconds=time.perf_counter()-t0
    C=int(max(yt.max(),yv.max(),classes.max()))+1
    def p_many(batch):
        ps=model_probability(np.asarray(batch))
        z=np.full((len(batch),C),1e-12)
        z[:,classes]=ps
        return np.log(np.clip(z,1e-12,1))
    base=p_many(xv)
    preds=np.argmax(base,axis=1)
    samples=select_cases(yv,preds,args.n_per_class,args.n_errors,args.max_samples,args.seed)
    bank=fit_real_donor_bank(xt,yt,comps,max_donors=args.max_donors,seed=args.seed+89)
    per_comp=[];per_case=[];skipped=0
    for sample in samples:
        x=xv[sample];baseline=base[sample];a=int(preds[sample]);b=int(np.argsort(-baseline)[1])
        if a not in bank.labels or b not in bank.labels or \
           len(bank.donors[b])//2<args.replicates or \
           len(bank.donors[b])-len(bank.donors[b])//2<args.reference_donors:
            skipped+=1
            continue
        components=best_q(class_cheap(x,bank.means[a],comps)[2],16)
        rows,result=run_consensus(x,baseline,comps,components,bank,a,b,p_many,
                        seed=args.seed+int(sample)*191,epsilon=args.epsilon,
                        replicate_count=args.replicates,
                        reference_count=args.reference_donors,taper=args.taper)
        info=dict(sample=int(sample),subject=int(subjects[sample]),
                  true_class=int(yv[sample]),source=a,target=b,
                  is_correct=int(a==yv[sample]))
        per_comp.extend({**info,**r} for r in rows)
        for method in ('single','consensus'):
            vals=result[method]
            per_case.append({**info,'method':method,'Q':16,
                             'reference_queries':result['offline_reference_model_calls'],
                             **vals})
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'per_component.csv',per_comp)
    save_csv(out/'per_case.csv',per_case)
    summary=[]
    for method in ('single','consensus'):
        rows=[r for r in per_case if r['method']==method]
        if not rows:continue
        tp=sum(r['true_reference_claims'] for r in rows)
        fp=sum(r['unsupported_claims'] for r in rows)
        refs=set()
        for r in per_comp:
            if r['reference_consistent']:
                refs.add((r['sample'],r['component']))
        summary.append(dict(method=method,n_cases=len(rows),
                query_audit_per_case=16,offline_reference_per_case=args.reference_donors*16,
                claim_count=tp+fp, supported_claims=tp, unsupported_claims=fp,
                empirical_ppv=tp/(tp+fp) if tp+fp else None,
                empirical_recall=tp/len(refs) if refs else None,
                explanation_coverage=float(np.mean([not r['abstained'] for r in rows])),
                mean_claims_per_case=float(np.mean([r['claims'] for r in rows]))))
    save_csv(out/'summary.csv',summary)
    for subgroup in ('subject','is_correct'):
        result=[]
        for value in sorted(set(r[subgroup] for r in per_case)):
            for method in ('single','consensus'):
                rr=[r for r in per_case if r[subgroup]==value and r['method']==method]
                tp=sum(r['true_reference_claims'] for r in rr)
                fp=sum(r['unsupported_claims'] for r in rr)
                result.append({subgroup:value,'method':method,'n_cases':len(rr),
                               'claim_count':tp+fp,
                               'empirical_ppv':tp/(tp+fp) if tp+fp else None,
                               'explanation_coverage':float(np.mean([not r['abstained'] for r in rr])) if rr else None})
        save_csv(out/f'per_{subgroup}.csv',result)
    info=dict(dataset='PAMAP2',checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
              test_subjects=sorted(map(int,np.unique(subjects))),
              selected_cases=len(samples),evaluated_cases=len(per_case)//2,
              skipped_cases=skipped,classifier=args.model,baseline_test_accuracy=float(np.mean(preds==yv)),
              train_seconds=train_seconds,train_donor_memory_bytes=bank.bank_bytes,
              per_method_audit_budget=16,
              reference_donors_per_component=args.reference_donors,
              offline_reference_Q_per_case=args.reference_donors*16,
              total_model_evaluations=int(len(per_case)//2*(16+16+16*args.reference_donors)),
              epsilon=args.epsilon,
              reference_definition='positive directional response >=75% on 8 reserved TRAIN donors',
              confounders=['Reference is not natural or expert ground truth',
                           'Overlapping windows and two subject folds prevent general inference',
                           'Replicated top4 checks fewer unique components than single top16',
                           'Threshold not learned and not statistically calibrated',
                           'Full offline reference cost excluded from deployment Q but reported'])
    (out/'manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf8')
    lines=['# Donor-resampling evidence stability under equal Q16','',
           'Single: 16 different components x 1 donor. Consensus: 4 components x 4 donors.',
           'Reference: 8 OTHER training donors on each of all 16 components (offline Q128).',
           'A reference-positive means >=75% of those replacements independently yield',
           'p(source) decrease >epsilon and p(target) increase >epsilon.',
           'This is NOT a causal or expert validated truth criterion.','',
           '| Method | N | PPV vs reference | Coverage of explanations | Mean claims |',
           '|---|---:|---:|---:|---:|']
    for r in summary:
        ppv='NA' if r['empirical_ppv'] is None else f"{r['empirical_ppv']:.1%}"
        lines.append(f"| {r['method']} | {r['n_cases']} | {ppv} | "
                     f"{r['explanation_coverage']:.1%} | {r['mean_claims_per_case']:.2f} |")
    lines+=['','Independent donors are drawn only from TRAIN data; held-out test labels',
            'are used only to report subgroup diagnostics. Abstention means no eligible',
            'measured robust effect, not that a model prediction is definitely correct.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('DONOR-CONSENSUS Q16 BENCHMARK',out,flush=True)
    for r in summary:
        print(r,flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz-path',type=Path,default=Path('data/pamap2_acc9_w200s100_p095.npz'))
    p.add_argument('--out-dir',type=Path,default=Path('outputs_composite/evidence_consensus'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--time-bins',type=int,default=12)
    p.add_argument('--n-per-class',type=int,default=4)
    p.add_argument('--n-errors',type=int,default=2)
    p.add_argument('--max-samples',type=int,default=72)
    p.add_argument('--max-donors',type=int,default=64)
    p.add_argument('--replicates',type=int,default=4)
    p.add_argument('--reference-donors',type=int,default=8)
    p.add_argument('--epsilon',type=float,default=.01)
    p.add_argument('--taper',type=int,default=2)
    p.add_argument('--model',choices=('extratrees','raw_mlp'),default='extratrees')
    p.add_argument('--trees',type=int,default=120)
    run(p.parse_args())

if __name__=='__main__':cli()
