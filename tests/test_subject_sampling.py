import numpy as np
import pytest
from dataclasses import dataclass
from beaconxai.subject_sampling import balanced_subject_sample, inject_subject_balanced

@dataclass
class Comp:
    t0: int
    t1: int
    c0: int
    c1: int


def test_rare_subject_remains_in_sample():
    s = np.array([8] * 126 + [9] * 2)
    ix = balanced_subject_sample(s, 64, 42)
    assert len(ix) == 64
    assert sum(s[ix] == 9) == 2
    np.testing.assert_array_equal(ix, balanced_subject_sample(s, 64, 42))
    with pytest.raises(ValueError):
        balanced_subject_sample(s, 1, 42)


def test_each_subject_gets_injected_and_true_component_known():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(64, 40, 3)).astype('float32')
    y = np.array([0, 1] * 32)
    s = np.array([8] * 62 + [9] * 2)
    c = [Comp(0, 20, 0, 3), Comp(20, 40, 0, 3)]
    before = x.copy()
    edited, mask, labels = inject_subject_balanced(x, y, s, c, .5, 123)
    assert mask.sum() == 32
    assert mask[s == 8].sum() == 31
    assert mask[s == 9].sum() == 1
    assert (labels[mask == 0] == -1).all()
    assert np.isin(labels[mask == 1], (0,1)).all()
    np.testing.assert_array_equal(x, before)
    assert np.count_nonzero(np.any(edited != x, axis=(1,2))) == 32
    e2, m2, t2 = inject_subject_balanced(x, y, s, c, .5, 123)
    np.testing.assert_array_equal(edited, e2)
    np.testing.assert_array_equal(mask, m2)
    np.testing.assert_array_equal(labels, t2)


def test_invalid_input_and_rare_one_window_subject():
    x = np.zeros((10, 20, 2))
    y = np.arange(10) % 2
    s = np.array([1] * 9 + [2])
    c = [Comp(0, 10, 0, 2)]
    _, mask, _ = inject_subject_balanced(x, y, s, c, .5, 4)
    assert mask[s == 2].sum() == 1
    with pytest.raises(ValueError):
        inject_subject_balanced(x, np.zeros(10), s, c, .5, 4)
    with pytest.raises(ValueError):
        inject_subject_balanced(x, y, s, c, 1.1, 4)
