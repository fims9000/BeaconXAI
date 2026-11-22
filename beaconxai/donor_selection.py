"""Train-only diversity controls for finite-donor counterfactual explanations.

Farthest-first (fixed-seed k-center heuristic) selects a small, diverse
subset of available class-B donor windows without querying the classifier.
This has NO robustness guarantee unless a separate Lipschitz bound or
distributional assumption is established. All inputs are train-standardized.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np


def farthest_first_donors(
    donor_bank: Sequence[np.ndarray], first_index: int, count: int = 3
) -> tuple[int, ...]:
    """Fixed nearest anchor, then maximize distance to the closest selected donor.

    A deterministic tie is resolved by lowest bank index. The first anchor
    is chosen by the caller independently, using only train-bank geometry.
    """
    bank = tuple(np.asarray(x, dtype=float) for x in donor_bank)
    if (not bank or any(d.ndim != 2 or d.shape != bank[0].shape or
                        not np.isfinite(d).all() for d in bank)):
        raise ValueError("all donor windows must have the same finite [T,C] shape")
    if (not isinstance(first_index, (int, np.integer)) or
            not 0 <= first_index < len(bank) or
            not 1 <= count <= len(bank)):
        raise ValueError("invalid first donor or selection count")
    data = np.stack([d.reshape(-1) for d in bank])
    selected = [int(first_index)]
    remaining = set(range(len(bank))) - set(selected)
    min_sq = np.sum((data - data[first_index]) ** 2, axis=1)
    while len(selected) < count:
        candidate = min(remaining, key=lambda j: (-float(min_sq[j]), j))
        selected.append(candidate)
        remaining.remove(candidate)
        new_sq = np.sum((data - data[candidate]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, new_sq)
    return tuple(selected)


def cover_radius(donor_bank: Sequence[np.ndarray], selected: Sequence[int]) -> float:
    """Maximum L2 distance from donor bank to the nearest selected donor."""
    bank = np.stack([np.asarray(d, dtype=float).reshape(-1) for d in donor_bank])
    if (bank.ndim != 2 or not np.isfinite(bank).all() or
            not selected or len(set(selected)) != len(selected) or
            min(selected) < 0 or max(selected) >= len(bank)):
        raise ValueError("invalid donor bank or selected donor indices")
    squared = np.stack([
        np.sum((bank - bank[j]) ** 2, axis=1) for j in selected
    ])
    return float(np.sqrt(np.max(np.min(squared, axis=0))))
