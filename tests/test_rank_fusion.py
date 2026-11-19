import numpy as np
import pytest
from beaconxai.rank_fusion import fuse_tested_ranks


def test_extreme_weights_preserve_known_rankings():
    cheap=np.array([1.,4.,0.,3.,2.])
    measured=[4,2,1]
    assert fuse_tested_ranks(cheap,measured,0)==measured
    assert fuse_tested_ranks(cheap,measured,1)==[1,4,2]


def test_borda_fusion_only_ranks_queried_candidates():
    cheap=np.array([100.,4.,1.,3.,2.])
    measured=[2,4,1,3]
    fused=fuse_tested_ranks(cheap,measured,.5)
    assert set(fused)==set(measured) and len(fused)==4
    assert 0 not in fused
    assert fused==fuse_tested_ranks(cheap,measured,.5)


def test_invalid_rank_inputs_fail():
    cheap=np.array([2.,1.,0.])
    for inputs in [([1,1],.5), ([-1,2],.5), ([0,3],.5), ([1,2],1.1), ([],.5)]:
        with pytest.raises(ValueError):
            fuse_tested_ranks(cheap,*inputs)
    with pytest.raises(ValueError):
        fuse_tested_ranks(np.array([1.,np.nan]),[0],.5)
