"""Subject-weighted, post-hoc diagnostics for the fixed-Q two-operator audit.

The full-Q36 rankings here are OFFLINE reference checks only. The report/
abstain decision was made from actual Q/2+Q/2 probes before this module sees
those references. A reported flag is not calibrated selective-risk control.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def _mean(rows: list[dict], field: str) -> float:
    return float(np.mean([float(r[field]) for r in rows]))


def summarize_dual_operator(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Return method means, subject means, and descriptive gate diagnostics."""
    if not rows:
        raise ValueError("no dual-operator evaluation rows")
    grouped = defaultdict(list)
    for row in rows:
        subject = int(row["subject"])
        method = str(row["method"])
        if not method:
            raise ValueError("method must be nonempty")
        grouped[(method,subject)].append(row)
    per_subject = []
    for (method,subject), group in sorted(grouped.items()):
        q = {int(r["extra_model_evaluations"]) for r in group}
        unique = {int(r["sample"]) for r in group}
        if len(q) != 1 or len(unique) != len(group):
            raise ValueError("inconsistent budget or repeated dependent sample")
        if any(not 0 <= float(r["worst_reference_overlap"])
               <= float(r["mean_reference_overlap"]) <= 1 for r in group):
            raise ValueError("invalid reference agreement")
        per_subject.append({
            "method":method,"subject":subject,
            "n_windows":len(group),"extra_model_evaluations":q.pop(),
            "mean_reference_overlap":_mean(group,"mean_reference_overlap"),
            "worst_reference_overlap":_mean(group,"worst_reference_overlap"),
            "interpolation_reference_overlap":_mean(group,"interpolation_reference_overlap"),
            "medoid_reference_overlap":_mean(group,"medoid_reference_overlap"),
        })

    summary = []
    for method in sorted({r["method"] for r in per_subject}):
        a = [r for r in per_subject if r["method"]==method]
        if len({r["extra_model_evaluations"] for r in a}) != 1:
            raise ValueError("budget cannot vary by participant for method")
        summary.append({
            "method":method,"n_subjects":len(a),
            "n_windows":sum(r["n_windows"] for r in a),
            "extra_model_evaluations":a[0]["extra_model_evaluations"],
            "subject_macro_mean_reference_overlap":_mean(a,"mean_reference_overlap"),
            "subject_macro_worst_reference_overlap":_mean(a,"worst_reference_overlap"),
            "subject_macro_interpolation_reference_overlap":_mean(
                a,"interpolation_reference_overlap"),
            "subject_macro_medoid_reference_overlap":_mean(a,"medoid_reference_overlap"),
        })

    dual_rows=[r for r in rows if r["method"]=="dual_operator_q16"]
    if not dual_rows:
        raise ValueError("dual_operator_q16 row missing")
    by_subject=defaultdict(list)
    for row in dual_rows:
        if row.get("reported_not_abstained") not in (0,1):
            raise ValueError("dual report/abstain must be explicit")
        if row.get("full_operator_reference_consistent") not in (0,1):
            raise ValueError("missing post-hoc full-Q36 consistency outcome")
        by_subject[int(row["subject"])].append(row)

    diagnostic=[]
    for subject,group in sorted(by_subject.items()):
        reported=[r for r in group if r["reported_not_abstained"]==1]
        ref_consistent=[r for r in group if r["full_operator_reference_consistent"]==1]
        captured=[r for r in reported if r["full_operator_reference_consistent"]==1]
        diagnostic.append({
            "subject":subject,"n_windows":len(group),
            "n_reported":len(reported),
            "n_full_reference_consistent":len(ref_consistent),
            "n_both":len(captured),
            "reported_rate":len(reported)/len(group),
            "reference_consistency_rate":len(ref_consistent)/len(group),
            "reference_consistent_given_reported":(
                len(captured)/len(reported) if reported else None),
            "reported_given_reference_consistent":(
                len(captured)/len(ref_consistent) if ref_consistent else None),
            "reported_mean_reference_overlap":(
                _mean(reported,"mean_reference_overlap") if reported else None),
        })
    def valid_mean(key: str) -> float | None:
        vals=[r[key] for r in diagnostic if r[key] is not None]
        return float(np.mean(vals)) if vals else None
    gate = {
        "policy":"report >=2 common top-3 of the 8 measured candidates, else abstain",
        "not_a_causal_or_calibrated_coverage_guarantee":True,
        "full_q36_references_used_at_inference":False,
        "n_subjects":len(diagnostic),
        "n_windows":len(dual_rows),
        "n_reported":sum(r["n_reported"] for r in diagnostic),
        "n_full_reference_consistent":sum(
            r["n_full_reference_consistent"] for r in diagnostic),
        "subject_macro_reported_rate":valid_mean("reported_rate"),
        "subject_macro_full_reference_consistency_rate":valid_mean(
            "reference_consistency_rate"),
        "subject_macro_reference_consistent_given_reported":valid_mean(
            "reference_consistent_given_reported"),
        "subject_macro_reported_given_reference_consistent":valid_mean(
            "reported_given_reference_consistent"),
        "subject_macro_reference_overlap_when_reported":valid_mean(
            "reported_mean_reference_overlap"),
        "per_subject":diagnostic,
    }
    return summary,per_subject,gate
