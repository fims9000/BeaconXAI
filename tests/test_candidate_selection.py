import numpy as np
import pytest
from dataclasses import dataclass
from beaconxai.candidate_selection import stratified_top_q


@dataclass(frozen=True)
class Comp:
    group: int
    time_bin: int


def fixture():
    cs = [Comp(g, t) for g in range(3) for t in range(12)]
    a = np.array([100 - t for t in range(12)] +
                 [10 - t for t in range(12)] +
                 [0 - t for t in range(12)], float)
    return a, cs


def test_sensor_group_quota_represents_all_three_groups():
    a, cs = fixture()
    selected = stratified_top_q(a, cs, 16, "group")
    assert len(selected) == len(set(selected)) == 16
    assert [sum(cs[i].group == j for i in selected) for j in range(3)] == [6, 5, 5]
    assert selected == stratified_top_q(a, cs, 16, "group")


def test_time_bin_quota_and_small_budget_fallback():
    a, cs = fixture()
    selected = stratified_top_q(a, cs, 16, "time")
    assert len(set(selected)) == 16
    assert {cs[i].time_bin for i in selected} == set(range(12))
    assert stratified_top_q(a, cs, 8, "time") == list(range(8))


def test_invalid_scores_components_and_budget():
    a, cs = fixture()
    for data, q, by in [
        (np.array([1.]), 16, "group"), (a, 0, "group"),
        (a, 37, "group"), (a, 16, "other"),
        (np.where(np.arange(36) == 2, np.nan, a), 16, "time")
    ]:
        with pytest.raises(ValueError):
            stratified_top_q(data, cs, q, by)
