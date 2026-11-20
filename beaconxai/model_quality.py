"""Held-out classifier sanity metrics, separate from the BEACON audit itself.

Ground-truth labels are inspected strictly in this *offline* scorer; they
must never enter candidate preselection, intervention generation or the Q-gate.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels=np.unique(np.concatenate((y_true,y_pred)))
    scores=[]
    for cls in labels:
        tp=int(np.sum((y_true==cls)&(y_pred==cls)))
        fp=int(np.sum((y_true!=cls)&(y_pred==cls)))
        fn=int(np.sum((y_true==cls)&(y_pred!=cls)))
        denom=2*tp+fp+fn
        scores.append((2*tp/denom) if denom else 0.)
    return float(np.mean(scores))


def heldout_model_quality(
    model_log_scores: np.ndarray, labels: np.ndarray,
    subjects: np.ndarray
) -> tuple[dict,list[dict]]:
    """Descriptive accuracy/F1 per participant; one row per genuine window.

    Classes include the union of predicted and true labels, so unseen class
    failures are not silently ignored. No independent-window significance
    estimates are claimed.
    """
    z=np.asarray(model_log_scores,dtype=float)
    y=np.asarray(labels)
    s=np.asarray(subjects)
    if (z.ndim!=2 or z.shape[1]<2 or z.shape[0]<1
        or y.shape!=(len(z),) or s.shape!=(len(z),)
        or not np.isfinite(z).all()
        or not np.issubdtype(y.dtype,np.integer)
        or np.any(y<0)):
        raise ValueError("invalid held-out scores, labels or subject IDs")
    predicted=np.argmax(z,axis=1)
    by_subject=defaultdict(list)
    for i,subject in enumerate(s):
        by_subject[int(subject)].append(i)
    report=[]
    for subject,indices in sorted(by_subject.items()):
        true=y[indices];chosen=predicted[indices]
        report.append({
            "subject":subject,"windows":len(indices),
            "n_true_classes":int(len(np.unique(true))),
            "n_predicted_classes":int(len(np.unique(chosen))),
            "accuracy":float(np.mean(true==chosen)),
            "macro_f1":_macro_f1(true,chosen),
        })
    quality={
        "n_test_subjects":len(report),
        "n_test_windows":len(y),
        "test_windows_micro_accuracy":float(np.mean(y==predicted)),
        "test_windows_micro_macro_f1":_macro_f1(y,predicted),
        "subject_macro_accuracy":float(np.mean([x["accuracy"] for x in report])),
        "subject_macro_f1":float(np.mean([x["macro_f1"] for x in report])),
        "labels_used_in_explanation":False,
        "evaluated_only_after_fixed_model_training":True,
    }
    return quality,report
