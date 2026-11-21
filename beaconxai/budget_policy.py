"""One GLOBAL model-call budget across alternative classes and cumulative edits.

Hypothesis-testing code, not a proven novel sequential optimizer. The policy
sees only x, train class donors, original output and observations so far.
It does not use held-out true class labels or a successful full-Q future.
"""
from __future__ import annotations
import numpy as np
from .real_donor import make_target_replacement, plausibility_proxies

POLICIES = ('top1_full', 'equal', 'prior_3to1', 'adaptive', 'adaptive_stop')


def shifted_scores(logits: np.ndarray) -> np.ndarray:
    z=np.asarray(logits,dtype=float)
    if z.ndim!=1 or not np.isfinite(z).all():
        raise ValueError('nonfinite score vector')
    a=np.exp(z-z.max())
    return a/a.sum()


def preplan(targets: tuple[int,int], q: int, policy: str) -> list[int]:
    b,c=targets
    if policy=='top1_full':
        return [b]*q
    if policy=='equal':
        return [v for t in range(q//2) for v in (b,c)]
    if policy=='prior_3to1':
        return [b]*(q-q//4)+[c]*(q//4)
    raise ValueError('no static preplan')


def _reward(records: list[dict], target: int) -> float:
    rr=[r for r in records if r['target']==target]
    if not rr:
        return float('-inf')
    if any(r['target_flip'] for r in rr):
        return 1e6  # an observed flip, not an unobserved future
    return max(0.0,max(-r['delta_gap'] for r in rr)) + max(
        0.0,max(r['delta_p_target'] for r in rr))


def run_budget_policy(x:np.ndarray, base:np.ndarray, bank, components,
                      selected:list[int], predict_one, q:int=16,
                      policy:str='adaptive', donor_kind:str='medoid',
                      seed:int=42, taper:int=2) -> tuple[list[dict],dict]:
    """Sequentially query a black box exactly q times unless stop-on-flip.

    Each target has its own cumulative edited input. Budget is shared across
    BOTH rival targets and all time/sensor components for one original object.
    Predict returns log-probability scores, as in repository's ExtraTrees adapter.
    """
    if policy not in POLICIES:
        raise ValueError(policy)
    if q<4 or q%4 or len(selected)<q or len(set(selected))!=len(selected):
        raise ValueError('q multiple of four, q<=number of distinct components')
    base=np.asarray(base,dtype=float)
    classes=np.argsort(-base,kind='stable')
    if len(base)<3:
        raise ValueError('need at least three classes for two-rival policy')
    a,b,c=map(int,classes[:3])
    if b not in bank.labels or c not in bank.labels:
        raise ValueError('rival absent from training bank')
    targets=(b,c)
    states={b:x.copy(),c:x.copy()}
    n_done={b:0,c:0}
    records=[]
    p0=shifted_scores(base)
    def one(target):
        position=n_done[target]
        if position>=len(selected):
            raise ValueError('repeated component budget exhausted')
        idx=selected[position]
        comp=components[idx]
        # Construct replacement from ORIGINAL x; update separate
        # cumulative target-specific state, preserving the other state.
        edited,_=make_target_replacement(x,comp,idx,target,donor_kind,bank,
                                        components,seed=seed+target*1007+idx*133,
                                        taper=taper)
        region=np.s_[comp.t0:comp.t1,comp.c0:comp.c1]
        candidate=states[target].copy()
        candidate[region]=edited[region]
        lp=np.asarray(predict_one(candidate),dtype=float)
        if lp.shape!=base.shape:
            raise ValueError('predict_one must return one full score vector')
        p=shifted_scores(lp)
        delta_gap=float((lp[a]-lp[target])-(base[a]-base[target]))
        target_flip=int(np.argmax(lp))==target
        stats=plausibility_proxies(x,candidate,comp,bank,target,components)
        row=dict(query=len(records)+1,target=target,component=idx,
                 target_step=position+1,source=a,
                 delta_p_source=float(p[a]-p0[a]),
                 delta_p_target=float(p[target]-p0[target]),
                 delta_gap=delta_gap,target_flip=bool(target_flip),
                 predicted_after=int(np.argmax(lp)),
                 directed=bool(p[a]<p0[a] and p[target]>p0[target]),
                 **stats)
        records.append(row)
        states[target]=candidate
        n_done[target]+=1
        return target_flip
    if policy in ('top1_full','equal','prior_3to1'):
        order=preplan(targets,q,policy)
        for tgt in order:
            one(tgt)
    else:
        # Prospectively obtain q/4 probes for each target. Allocate all
        # remaining q/2 calls to the target with largest observed reward.
        exploratory=q//4
        for k in range(exploratory):
            for tgt in targets:
                if one(tgt) and policy=='adaptive_stop':
                    break
            if policy=='adaptive_stop' and any(r['target_flip'] for r in records):
                break
        if not (policy=='adaptive_stop' and any(r['target_flip'] for r in records)):
            # Tie goes to original runner-up b.
            chosen=b if _reward(records,b)>=_reward(records,c) else c
            while len(records)<q:
                if one(chosen) and policy=='adaptive_stop':
                    break
    flips=[r for r in records if r['target_flip']]
    directed=[r for r in records if r['directed']]
    result=dict(policy=policy,source=a,runnerup=b,third=c,
                queries=len(records),target_flip_any=int(bool(flips)),
                first_flip_query=(flips[0]['query'] if flips else None),
                directed_any=int(bool(directed)),
                directed_fraction=len(directed)/len(records),
                best_gap_reduction=max(-r['delta_gap'] for r in records),
                max_delta_p_target=max(r['delta_p_target'] for r in records),
                mean_seam_excess=float(np.mean([r['seam_excess'] for r in records])),
                mean_relative_l2=float(np.mean([r['relative_l2'] for r in records])),
                status='target_flip' if flips else 'directed_only' if directed else 'abstain')
    return records,result
