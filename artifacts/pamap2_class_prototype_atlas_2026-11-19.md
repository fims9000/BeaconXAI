# Prototype interventions for class explanations — PAMAP2 pilot (2026-11-19)

**Status:** code complete and evaluated on two held-out-subject pairs; exploratory, not a claim
of causal interpretation, physical class boundaries, or model superiority.

Repo: `fims9000/BeaconXAI`, research branch `feature/contrastive-q16-20261119`.
Source modules:
- `beaconxai/class_counterfactual.py` — train-only mean/median prototype bank, component intervention, signed p(class) and fixed A-vs-B score-gap changes, actual post-intervention predicted class.
- `scripts/run_class_counterfactual_atlas.py` — exports local examples, per-class and per-pair aggregates, tested-component summaries, and figures.
- `tests/test_class_counterfactual.py` — query-budget, train/test separation, sign, third-class switches, integration and plot tests.
- `.github/workflows/pamap2_class_counterfactual.yml` — reproducible experiment on two held-out subject pairs.

## Research question

For a correctly classified time series with source class A and candidate class B,
which selected time/sensor components change the model output if we replace
them with TRAIN-only mean or median templates for A or B? Can such experiments
be summarized in user-readable *local* statements and *descriptive class* maps?

We deliberately call these **prototype interventions**, not necessarily
feasible counterfactuals. A true *target-class counterfactual* would have to
reach a decision switch to B with a plausible input.

## Measurement protocol

- UCI PAMAP2, 200 samples per window, 100-sample stride, purity >=0.95.
- Subjects 2,3 and 6,7 held out in two **separate experiments**; models
  refitted on the remaining subjects, so class-level values cannot be
  pooled as outputs of one classifier.
- ExtraTrees, 120 trees, `_anfis_features`, training-only channel scaling.
- 12 PAMAP2 activity classes in the training split; 9 acceleration channels
  grouped into wrist/chest/ankle, 12 time bins: 36 components.
- Correctly predicted held-out examples only, max **2 examples per source
  class**; do not treat thousands of interventions as independent subjects.
- For each source example and each rival class, components preselected by
  the ORIGINAL `adaptive_v2` cheap combined class-profile ranking.
- Q=16 **additional model evaluations per class pair** = 4 selected
  components x 2 destination classes x 2 prototype kinds (mean+median).
  One cached original prediction per evaluated window is separate.
- Prototype values generated from training data only; two-point taper
  blends at the intervention boundaries.
- Outputs: `per_intervention.csv`, `component_summary.csv`,
  `pair_summary.csv`, `class_atlas.csv`, `report.md`, `plots/*.png`,
  `manifest.json` and `provenance.txt`.
- `delta_p_source` and `delta_p_rival` represent changes in the
  **model's output scores** converted to probability units. These need not be
  statistically calibrated probabilities.

To avoid a misleading explanation, we require **both**
`delta_p_source < 0` AND `delta_p_rival > 0` to call a directionally
supportive movement toward B. This criterion is weaker than a class switch.

## Completed result tables

| Held out | Source classes observed | Source examples | Source-vs-rival pairs | Extra model evaluations | Local/heatmap PNGs |
|---|---:|---:|---:|---:|---:|
| 2,3 | 12 / 12 | 24 | 264 | 4,224 | 24 |
| 6,7 | 11 / 12 | 22 | 242 | 3,872 | 22 |

`rope jumping` has no test examples in the held-out 6,7 split,
so no class passport was fabricated for it.

### Replacements using a rival-class template

| Dataset | Prototype | N interventions | Directional movement toward rival | Actual switch to rival | Mean delta model p(source) | Mean delta model p(rival) |
|---|---|---:|---:|---:|---:|---:|
| 2,3 | mean | 1,056 | 181 / 1,056 = **17.14%** | 4 / 1,056 | -1.74 pp | +0.41 pp |
| 2,3 | median | 1,056 | 200 / 1,056 = **18.94%** | 7 / 1,056 | -2.01 pp | +0.45 pp |
| 6,7 | mean | 968 | 226 / 968 = **23.35%** | 5 / 968 | -5.46 pp | +0.65 pp |
| 6,7 | median | 968 | 220 / 968 = **22.73%** | 5 / 968 | -5.57 pp | +0.66 pp |

Descriptive only. Interventions per test example and across target classes
are highly dependent. No p-values or confidence intervals are appropriate
without subject-level replication.

**Main finding:** replacing with the average or median signal of another
class usually does **not** produce a convincing transition toward that class.
Even where p(source) falls, p(rival) may stay flat or fall as well. Actual
switches to the intended rival are rare (<1% of these measured replacements).
No empirical advantage of mean versus median has been established.

Example: in one held-out PAMAP2 walking window, replacing an ankle
fragment with the running mean reduced p(walking) from **47.5% to
39.2%**, while p(running) remained approximately **0.83%**; the class
did not switch. A valid text explanation is "this intervention weakens
the walking score", NOT "this is evidence of running".

## Qualitative visual inspection

The user-facing output now contains two **separate** maps for a
given A-vs-B comparison, recording change in p(A) and change in p(B).
Untested cells are grey; color intensity is signed percentage-point
change, not causality. Overlay images show the actual segment and
training-class source/target mean/median templates. In examined signals,
templates often appear nearly flat relative to the original motion
signal, consistent with loss of temporal phase/alignment information.

This explains why class-average replacement may be a poor proxy for
a physically possible class transition. A next scientifically justified
baseline is a **real class medoid/donor segment**, possibly aligned by
local DTW or constrained morphing, with controls for replacement
magnitude and continuity.

## Limits on novelty and current capability

- The atlas reports **observed response of a fixed model**, not a
  human-validated physical explanation of what differentiates activities.
- Class-level aggregates average over **different rival classes**; use
  `pair_summary.csv` and `component_summary.csv` for pair-specific
  conclusions. Per class at most two correctly predicted examples were
  audited in each fold.
- The graphs identify checked bins, not a global class boundary.
- Mean/median class templates need not be on the empirical motion manifold.
- This experiment has **not** established usefulness over a no-query
  explanation, decision rules from ExtraTrees/ANFIS, TreeSHAP, or
  shapelet/counterfactual baselines.
- Prototype bank memory = **172,800 bytes** in float32 (mean+median
  for 12 classes, 200 samples, 9 channels), excluding classifier/model.
  No MCU experiments or energy measurements have been conducted.

## Reproduction

```bash
python -m pip install -e . pytest matplotlib
python -m pytest tests/test_class_counterfactual.py -q
python scripts/preprocess_pamap2.py --data-root data --window-length 200 --step 100 --min-purity 0.95 --test-subjects 2,3 --out data/pamap2_s2s3.npz
python scripts/run_class_counterfactual_atlas.py --npz-path data/pamap2_s2s3.npz --max-test 0 --max-examples-per-class 2 --time-bins 12 --budget 16 --prototype-mode both --rival-mode all --out-dir outputs_composite/class_atlas_s2s3
```

PAMAP2 NPZ sha256:
- 2,3: `2e2fc9f7893f2dd93796064e850f9c646797730ca1576e4b52a1f78d79a31f45`
- 6,7: `57e87ec51d198be3f42ce1d0b9ffeec2646a5e62bf8b5878dd8d4460898aa7da`

Successful actual dataset workflows:
- https://github.com/fims9000/BeaconXAI/actions/runs/44539225838
- https://github.com/fims9000/BeaconXAI/actions/runs/44540443260

Artifacts of the final **paired p(A)/p(B) maps** (retained 30 days):
- 2,3: https://github.com/fims9000/BeaconXAI/actions/runs/44540443260/artifacts/14538176559
- 6,7: https://github.com/fims9000/BeaconXAI/actions/runs/44540443260/artifacts/14538431441

**Next falsifiable hypothesis:** replacing with aligned *real* fragments
from a target-class example, rather than pointwise mean/median, yields
more frequent plausible A -> B directed movement under the same
matched query budget. Compare to a random donor, not just the class
mean, and evaluate perturbation realism independently.
