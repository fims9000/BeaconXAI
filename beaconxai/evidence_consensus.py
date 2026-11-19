"""Repeated-donor evidence checks with abstention under fixed 16-call audit budget.

Training donor pool is partitioned into SEARCH and OFFLINE REFERENCE subsets.
Offline reference probes are never visible to the selector or claim generator.
The reference measures donor-resampling consistency, NOT physical correctness.
"""
from __future__ import annotations
import numpy as np
from .class_counterfactual import replace_component


def _prob(logit):
    z=np.asarray(logit,float)
    t=np.exp(z-np.max(z,axis=-1,keepdims=True))
    return t/np.sum(t,axis=-1,keepdims=True)


def _measure(base, outcomes, a, b, min_delta):
    p0=_prob(np.asarray(base))
    p1=_prob(np.asarray(outcomes))
    da=p1[:,a]-p0[a]
    db=p1[:,b]-p0[b]
    return da,db,((da < -min_delta)&(db > min_delta))


def run_consensus(x, base, components, selected:list[int], bank, source:int,
                  target:int, predict_many, seed:int=42, epsilon:float=.01,
                  replicate_count:int=4, reference_count:int=8,
                  taper:int=2):
    """Generate two types of claims from 16 queries and check them offline.

    Uniform single donor: top16 components * 1 search donor = Q16.
    Replicated: top4 components * 4 independent donors = Q16.
    Reference: top16 * 8 separate unused training donors = 128 queries.
    The reference is NOT a ground-truth label of causal importance.
    """
    if replicate_count<2 or reference_count<2:
        raise ValueError('need repeat probes and offline reference donors')
    if len(selected)!=16 or len(set(selected))!=16 or any(j<0 or j>=len(components) for j in selected):
        raise ValueError('must pass 16 distinct selected components')
    if target not in bank.donors or target==source:
        raise ValueError('invalid target')
    if epsilon<0 or epsilon>=1:
        raise ValueError('epsilon outside [0,1)')
    refs=bank.donors[target]
    n=len(refs)
    n_dev=n//2
    if n_dev<replicate_count or n-n_dev<reference_count:
        raise ValueError('insufficient TRAIN donors for independent pool split')
    search_actions=[]
    consensus_actions=[]
    ref_actions=[]
    for position,component_id in enumerate(selected):
        rng=np.random.default_rng(seed+7919*component_id)
        chosen=rng.choice(n_dev,size=replicate_count,replace=False).tolist()
        control=chosen[0]
        c=components[component_id]
        search_actions.append(replace_component(x,c,refs[control],taper=taper))
        if position<4:
            for idx in chosen:
                consensus_actions.append(replace_component(x,c,refs[idx],taper=taper))
        verify=rng.choice(np.arange(n_dev,n),size=reference_count,replace=False)
        for idx in verify:
            ref_actions.append(replace_component(x,c,refs[idx],taper=taper))
    # 16 + 16 audit calls PLUS 128 offline reference calls.
    single=np.asarray(predict_many(np.stack(search_actions)),dtype=float)
    consensus=np.asarray(predict_many(np.stack(consensus_actions)),dtype=float)
    reference=np.asarray(predict_many(np.stack(ref_actions)),dtype=float)
    if single.shape!=(16,len(base)) or consensus.shape!=(16,len(base)) \
            or reference.shape!=(16*reference_count,len(base)):
        raise ValueError('incorrect model response shape')
    ds,bs,one=_measure(base,single,source,target,epsilon)
    dc,bc,group=_measure(base,consensus,source,target,epsilon)
    dr,br,ref=_measure(base,reference,source,target,epsilon)
    ref=ref.reshape(16,reference_count)
    ref_rate=ref.mean(axis=1)
    ref_consistent=(ref_rate>=.75)
    by_comp=[]
    for i,cid in enumerate(selected):
        n_positive=int(group[replicate_count*i:replicate_count*(i+1)].sum()) if i<4 else -1
        replicated=bool(i<4 and n_positive==replicate_count)
        by_comp.append(dict(component=int(cid),rank=i+1,
                            single_claim=bool(one[i]),
                            replicated_claim=replicated,
                            replicates_evaluated=(replicate_count if i<4 else 0),
                            repeated_positive_count=n_positive,
                            reference_donor_count=reference_count,
                            reference_positive_count=int(ref[i].sum()),
                            reference_positive_fraction=float(ref_rate[i]),
                            reference_consistent=bool(ref_consistent[i]),
                            single_delta_p_source=float(ds[i]),
                            single_delta_p_target=float(bs[i]),
                            median_repeat_delta_p_source=(float(np.median(dc[replicate_count*i:replicate_count*(i+1)])) if i<4 else None),
                            median_repeat_delta_p_target=(float(np.median(bc[replicate_count*i:replicate_count*(i+1)])) if i<4 else None)))
    def report(claim):
        preds=[(r[claim],r['reference_consistent']) for r in by_comp]
        tp=sum(bool(a and b) for a,b in preds)
        fp=sum(bool(a and not b) for a,b in preds)
        fn=sum(bool(not a and b) for a,b in preds)
        return {'claims':tp+fp,'true_reference_claims':tp,'unsupported_claims':fp,
                'precision':(tp/(tp+fp) if tp+fp else None),
                'recall':(tp/(tp+fn) if tp+fn else None),
                'abstained':bool(tp+fp==0)}
    return by_comp,dict(single=report('single_claim'),
                        consensus=report('replicated_claim'),
                        model_calls_single=16,model_calls_consensus=16,
                        offline_reference_model_calls=16*reference_count,
                        model_reference_definition=f'>=75% positive among {reference_count} other TRAIN donor replacements',
                        epsilon=epsilon)
