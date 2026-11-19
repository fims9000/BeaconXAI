"""Train-only real-fragment donors for black-box time-series interventions.

The labels are used ONLY while fitting the donor bank. At inference, the
method sees model-predicted A/B classes and the unlabeled test time series.
These operations are perturbation audits, not validated causal effects.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def component_fingerprints(x: np.ndarray, components) -> np.ndarray:
    """Per time-bin/sensor-group mean and std; shape [...,M,2]."""
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('inputs must be finite [N,T,C]')
    feats = np.empty((len(x), len(components), 2), dtype=np.float32)
    for j, c in enumerate(components):
        p = x[:, c.t0:c.t1, c.c0:c.c1]
        feats[:, j, 0] = p.mean(axis=(1, 2))
        feats[:, j, 1] = p.std(axis=(1, 2))
    return feats


@dataclass
class RealDonorBank:
    means: dict[int, np.ndarray]
    medians: dict[int, np.ndarray]
    medoids: dict[int, np.ndarray]
    donors: dict[int, np.ndarray]
    fingerprints: dict[int, np.ndarray]
    counts: dict[int, int]
    selected_train_indices: dict[int, np.ndarray]

    @property
    def labels(self) -> list[int]:
        return sorted(self.means)

    @property
    def bank_bytes(self) -> int:
        arrays = list(self.means.values()) + list(self.medians.values()) + list(self.medoids.values())
        arrays += list(self.donors.values()) + list(self.fingerprints.values())
        return int(sum(a.nbytes for a in arrays))


def fit_real_donor_bank(x_train: np.ndarray, y_train: np.ndarray, components,
                        max_donors: int = 64, seed: int = 23) -> RealDonorBank:
    """Build an offline bank. A medoid is the TRAIN observation nearest its
    class mean in squared Euclidean distance (equivalent to squared-L2 medoid).

    The donor reservoir is sampled only from train; its memory must be counted.
    """
    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train)
    if x.ndim != 3 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError('invalid training data')
    if not np.isfinite(x).all() or max_donors < 1:
        raise ValueError('invalid data or max_donors')
    rng = np.random.default_rng(seed)
    means, medians, medoids, donors, fp, counts, indices = {}, {}, {}, {}, {}, {}, {}
    for label in np.unique(y):
        ix = np.flatnonzero(y == label)
        k = int(label)
        xs = x[ix]
        avg = np.mean(xs, axis=0, dtype=np.float64).astype(np.float32)
        median = np.median(xs, axis=0).astype(np.float32)
        distances = ((xs - avg[None]) ** 2).mean(axis=(1, 2))
        medoid_local = int(np.argmin(distances))
        chosen = (rng.choice(ix, size=max_donors, replace=False)
                  if len(ix) > max_donors else ix.copy())
        # Always retain the actual medoid among the banked real examples.
        medoid_global = int(ix[medoid_local])
        if medoid_global not in chosen:
            chosen = chosen.copy()
            chosen[-1] = medoid_global
        refs = x[chosen].copy()
        means[k], medians[k], medoids[k] = avg, median, xs[medoid_local].copy()
        donors[k], fp[k], counts[k], indices[k] = (
            refs, component_fingerprints(refs, components), len(ix), chosen.astype(int)
        )
    return RealDonorBank(means, medians, medoids, donors, fp, counts, indices)


def nearest_context_index(x: np.ndarray, component_id: int,
                          bank: RealDonorBank, target: int, components) -> int:
    """Closest donor in *untouched* time/sensor bins; no test label/target indices."""
    xfp = component_fingerprints(x, components)[0]
    d = np.mean((bank.fingerprints[target] - xfp[None]) ** 2, axis=2)
    d[:, component_id] = 0.0
    return int(np.argmin(np.sum(d, axis=1)))


def boundary_align_lag(x: np.ndarray, donor: np.ndarray, comp,
                       max_shift: int = 5) -> int:
    """Bounded local shift minimizing seam discrepancy, NOT full DTW.

    Search over unmodified TRAIN donor samples. Cropping within donor is kept
    in temporal order; two seam sides only, to avoid erasing target patterns.
    """
    if max_shift < 0:
        raise ValueError('max_shift must be >=0')
    t0, t1, c0, c1 = comp.t0, comp.t1, comp.c0, comp.c1
    best = (float('inf'), 0)
    for lag in range(-max_shift, max_shift + 1):
        a, b = t0 + lag, t1 + lag
        if a < 0 or b > len(donor):
            continue
        vals = []
        if t0 > 0:
            vals.append(np.mean((x[t0-1, c0:c1] - donor[a, c0:c1]) ** 2))
        if t1 < len(x):
            vals.append(np.mean((donor[b-1, c0:c1] - x[t1, c0:c1]) ** 2))
        cost = float(np.mean(vals)) if vals else 0.0
        # Deterministic tie-break toward no shift.
        key = (cost + 1e-5 * abs(lag), abs(lag), lag)
        if key < (best[0], abs(best[1]), best[1]):
            best = (key[0], lag)
    return best[1]


def make_target_replacement(x: np.ndarray, comp, component_id: int,
                            target: int, kind: str, bank: RealDonorBank,
                            components, seed: int = 0, taper: int = 2,
                            max_shift: int = 5) -> tuple[np.ndarray, dict]:
    """One query candidate, metadata never includes test label."""
    from .class_counterfactual import replace_component
    if target not in bank.means:
        raise ValueError('target class has no training donors')
    donor_ix = -1
    lag = 0
    if kind == 'mean':
        template = bank.means[target]
    elif kind == 'median':
        template = bank.medians[target]
    elif kind == 'medoid':
        template = bank.medoids[target]
    elif kind in ('random_donor', 'context_donor', 'aligned_donor'):
        if kind == 'random_donor':
            donor_ix = int(np.random.default_rng(seed).integers(len(bank.donors[target])))
        else:
            donor_ix = nearest_context_index(x, component_id, bank, target, components)
        template = bank.donors[target][donor_ix]
        if kind == 'aligned_donor':
            lag = boundary_align_lag(x, template, comp, max_shift=max_shift)
            template = template.copy()
            template[comp.t0:comp.t1, comp.c0:comp.c1] = bank.donors[target][donor_ix][
                comp.t0+lag:comp.t1+lag, comp.c0:comp.c1
            ]
    else:
        raise ValueError(f'unknown replacement: {kind}')
    edited = replace_component(x, comp, template, taper=taper)
    return edited, {'kind':kind, 'donor_index':donor_ix, 'lag':lag,
                    'train_index':(int(bank.selected_train_indices[target][donor_ix])
                                   if donor_ix >= 0 else -1)}


def plausibility_proxies(original: np.ndarray, edited: np.ndarray,
                         comp, bank: RealDonorBank, target: int, components) -> dict:
    """Transparent cheap proxies; NONE is proof of physical plausibility."""
    t0,t1,c0,c1 = comp.t0,comp.t1,comp.c0,comp.c1
    old, new = original[t0:t1,c0:c1], edited[t0:t1,c0:c1]
    delta = new-old
    relative = float(np.linalg.norm(delta) / (np.linalg.norm(old) + 1e-8))
    def seam(v):
        jumps = []
        if t0 > 0:
            jumps.append(float(np.mean(np.abs(v[t0,c0:c1]-v[t0-1,c0:c1]))))
        if t1 < len(v):
            jumps.append(float(np.mean(np.abs(v[t1,c0:c1]-v[t1-1,c0:c1]))))
        return float(np.mean(jumps)) if jumps else 0.0
    new_seam, old_seam = seam(edited),seam(original)
    # This 2-feature-per-component distance is only a distribution proxy.
    feat = component_fingerprints(edited,components)[0]
    ref = bank.fingerprints[target]
    dist = float(np.sqrt(np.min(np.mean((ref-feat[None])**2,axis=(1,2)))))
    return {'relative_l2':relative,'seam_excess':new_seam-old_seam,
            'seam_absolute':new_seam,'target_fp_nn_dist':dist}
