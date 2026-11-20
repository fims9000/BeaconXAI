import numpy as np
import pytest

from beaconxai.model_quality import heldout_model_quality


def test_per_subject_accuracy_and_macro_f1_do_not_micro_weight_subjects():
    y=np.array([0,1,1,0])
    subjects=np.array([1,1,9,9])
    # Predict [0,1,0,0].
    z=np.array([[3.,0.],[0.,3.],[2.,1.],[3.,0.]])
    quality,report=heldout_model_quality(z,y,subjects)
    assert report[0]["accuracy"]==1.
    assert report[1]["accuracy"]==.5
    assert quality["subject_macro_accuracy"]==.75
    assert report[0]["macro_f1"]==1.
    assert report[1]["macro_f1"]==pytest.approx(1/3)
    assert quality["subject_macro_f1"]==pytest.approx(2/3)
    assert quality["test_windows_micro_macro_f1"]==pytest.approx(11/15)
    assert quality["labels_used_in_explanation"] is False


def test_missing_true_class_is_penalized_in_f1():
    y=np.array([0,2,2,2])
    z=np.array([[10.,0.,0.],[10.,0.,0.],[10.,0.,0.],[10.,0.,0.]])
    quality,_=heldout_model_quality(z,y,np.array([7,7,7,7]))
    assert quality["test_windows_micro_accuracy"]==.25
    assert quality["test_windows_micro_macro_f1"]<.5


@pytest.mark.parametrize("scores,labels,subs", [
    (np.zeros((2,1)),np.array([0,1]),np.array([1,2])),
    (np.array([[1.,np.nan]]),np.array([0]),np.array([1])),
    (np.zeros((2,2)),np.array([0]),np.array([1,2])),
    (np.zeros((2,2)),np.array([-1,0]),np.array([1,2])),
])
def test_reject_invalid_inputs(scores,labels,subs):
    with pytest.raises(ValueError):
        heldout_model_quality(scores,labels,subs)
