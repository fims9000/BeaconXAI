"""Train-only k-center donor selection is deterministic and geometry-only."""
import numpy as np
import pytest
from beaconxai.donor_selection import farthest_first_donors, cover_radius


def test_farthest_first_starts_at_fixed_anchor_and_covers_outliers():
    donor_bank = [np.array([[v], [v]]) for v in (0., .1, 10., -10.)]
    chosen = farthest_first_donors(donor_bank, 0, 3)
    assert chosen == (0, 2, 3)
    assert cover_radius(donor_bank, chosen) < cover_radius(donor_bank, (0, 1, 2))
    assert farthest_first_donors(donor_bank, 0, 3) == chosen


def test_only_bank_derived_geometry_no_model_calls():
    bank = [np.ones((4,2))*v for v in (3., 9., -2., 2.)]
    assert farthest_first_donors(bank, 0, 1) == (0,)
    assert farthest_first_donors(bank, 0, 4) == (0,1,2,3)


def test_rejects_invalid_bank_indices_and_nan():
    with pytest.raises(ValueError):
        farthest_first_donors([np.ones((2, 2)), np.ones((3,2))], 0)
    with pytest.raises(ValueError):
        farthest_first_donors([np.ones((2, 2)), np.full((2,2),np.nan)], 0)
    with pytest.raises(ValueError):
        farthest_first_donors([np.ones((2,2))], 1)
    with pytest.raises(ValueError):
        cover_radius([np.ones((2,2)),np.zeros((2,2))], (1,1))
