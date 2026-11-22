# PAMAP2 matched Q16 ablation — four held-out-subject folds
**Date:** 2026-11-19  
**Status:** completed exploratory real-data experiment; **not a universal superiority claim**  
**PR:** https://github.com/fims9000/BeaconXAI/pull/1  
**Script:** `scripts/run_q16_matched_ablation.py`  
**Tests:** `tests/test_q16_matched_ablation.py`

## 1. Scientific question

Does querying the classifier Q=16 times add information beyond the class-profile
statistics already used to select candidates? Does the new class-pair contrastive
selector improve the original `adaptive_v2` on the **same** PAMAP2 examples?

The original `adaptive_v2` here means the published cheap-feature
preselection and **moving top-2 margin** score
`abs(margin_before - margin_after)`, not the `BeaconAudit` hierarchical
core, which has a different calling protocol.

## 2. Matched protocol

- Official UCI PAMAP2, 200-sample windows, step 100, purity >= 0.95.
- Three accelerometer groups x 12 time bins = **M=36** components.
- A single ExtraTrees(120) per held-out pair, repository `_anfis_features`,
  standardized channels using training split only.
- Exactly the same **128 sampled test windows per split**, with one
  donor-class fragment artificially injected in 64 windows.
- The same perturbation operator: per-component interpolation.
- Same cached original model output for all methods.
- Every Q=16 method evaluates **16 perturbed examples**, as a 16-row
  batch. Zero-query controls evaluate none.
- Selection uses only the input, classifier prediction, and training class
  profiles; injected position and test true label are not passed to selectors.
- Three repeated random seeds per same 64 injected windows, with model
  fixed within split. Repeated seeds are **not independent extra samples**.
- Per-subject descriptive tables in retained workflow artifacts; **no
  population p-values**: windows overlap within a subject, and subject
  sizes are small/imbalanced.

## 3. Primary results: hit@3 (artificially injected component)

| Method | Extra evals | Held out 2,3 | 4,5 | 6,7 | 8,9 | Descriptive mean of 4 folds |
|---|---:|---:|---:|---:|---:|---:|
| **Original adaptive_v2 moving margin** | 16 | 43.75% | 48.44% | 35.94% | 39.06% | **41.80%** |
| adaptive_v2 pinned class pair | 16 | 42.19% | 46.88% | 40.63% | 39.06% | 42.19% |
| New contrastive selector + fixed pair | 16 | 29.17% | 25.52% | 33.33% | 42.19% | **32.55%** |
| Original adaptive_v2 cheap prior only | **0** | 37.50% | 28.12% | 35.94% | 26.56% | **32.03%** |
| New contrastive cheap prior only | **0** | 26.56% | 15.62% | 25.00% | 31.25% | 24.61% |
| Uniform occlusion | 16 | 21.35% | 23.96% | 21.35% | 22.92% | 22.40% |
| RISE-like ridge mask regression | 16 | 21.35% | 18.75% | 20.83% | 23.44% | 21.09% |
| KernelSHAP-like weighted ridge | 16 | 14.58% | 12.50% | 9.90% | 11.46% | 12.11% |
| Energy only | 0 | 31.25% | 18.75% | 21.88% | 25.00% | 24.22% |
| Variance only | 0 | 34.38% | 15.62% | 21.88% | 18.75% | 22.66% |

These are descriptive means across four folds (4 x 64 = **256 distinct
injected test windows**, three random repetitions per test window). They
are NOT confidence intervals or subject-independent validation.

**Critical per-subject imbalance:** injected test windows per participant:
2:37, 3:27, 4:28, 5:36, 6:32, 7:32, 8:62, **9:2**.
Therefore the 8,9 fold is essentially an experiment on participant 8.

The original adaptive_v2 exceeds uniform in **all four folds**. Conversely,
the new contrastive selector only exceeds the original adaptive_v2 in fold
8,9, which is mostly participant 8. The current new selector does **not**
support a positive claim versus the original algorithm.

## 4. Does Q=16 add value beyond the cheap prior?

At hit@3, original adaptive_v2 Q16 minus its own prior Q0:

- 2,3: **+6.25 percentage points**
- 4,5: **+20.31 pp**
- 6,7: **0.00 pp**
- 8,9: **+12.50 pp**
- Four-fold descriptive mean: **+9.77 pp**

For split 6,7, model re-ranking helps 24 of 192
(window x repetition) comparisons and harms another 24; gains and
losses **cancel exactly**, even though the overall Q16 hit@3 matches Q0.

For the new contrastive strategy Q16 minus its own Q0 prior:
2,3 **+2.60 pp**, 4,5 **+9.90 pp**, 6,7 **+8.33 pp**,
8,9 **+10.94 pp**.

Therefore **the classifier-query stage is useful in some folds but
is NOT automatically useful for every test population or individual window**.
This is an empirical basis for exploring a train-calibrated query/gating
policy, not proof that such gating already works.

## 5. Margin semantics: moving vs pinned

The original adaptive_v2 recomputes the top-two margin after each
perturbation. The pinned-pair ablation always compares the original
predicted class against the same original runner-up. They are
different explanation targets. Pinned scores did not consistently win
(2,3 worse; 4,5 worse; 6,7 better; 8,9 tied); the four-fold mean
difference is only **+0.39 pp**, so do **not** claim an improvement.

## 6. Approximate computational footprint

Each Q16 method processes 16 perturbed examples **in a batch**.
Within-fold, all use the same classifier and hardware, improving
timing comparability versus previous sequential-vs-batched runs.

Median observed timing ranges across GitHub runners:
- original adaptive_v2 Q16: roughly **5.76–10.94 ms**
- adaptive_v2 cheap-only Q0: roughly **0.80–1.28 ms**
- contrastive Q16: roughly **5.26–10.18 ms**

Do not pool absolute times across separate GitHub runners for latency
claims, and do not equate Q with power or deployment feasibility.
Excludes the cached baseline model prediction and offline training.
Class profile storage (float64, 12 classes, 200 x 9):
- class means only, used by adaptive_v2: **172,800 bytes** (~169 KiB)
- means + variances used by contrastive prototype: **345,600 bytes** (~338 KiB)

Those profiles are material for constrained TinyML devices. MCU RAM,
energy, and on-device latency were **not** measured.

## 7. Interpretation and necessary next experiment

1. **Positive:** original adaptive_v2 has a consistent *descriptive*
   advantage over uniform at fixed Q16 in four subject folds.
2. **Important ablation:** cheap-only prior is a strong competitor; Q16
   provides a net benefit in three folds and exactly zero in one.
3. **Negative:** new contrastive method does NOT improve the original
   adaptive_v2 overall; keep it experimental.
4. **Unproven:** high hit@3 for inserted class-donor patches does not
   prove real-world causal, natural-class, or physical interpretability.
5. **Limitations:** two test subjects per fold with strongly unequal
   contribution in 8,9; time windows overlap; all four training sets
   overlap; class profile model inputs are pre-computed in TRAIN.
6. **Next falsifiable improvement:** use a separate training/validation
   subset to fit a cost-aware decision rule for when Q0 explanation
   is sufficient and when Q16 model re-ranking is worth paying for.
   Evaluate this rule on untouched held-out subjects. Compare average
   query cost, hit@3, false omission/error cost, and edge memory.
   Do not tune a gate from these held-out fold outcomes.
7. Run at least two neutralization operators and verify that interpolation
   smoothing is not producing the entire effect.
8. For broader XAI claims add official LIMESegment, TreeSHAP and canonical
   RISE/SHAP where applicable; `rise_like_q16` and
   `kernelshap_like_q16` here are **underdetermined regularized mask
   regressions**, NOT the original implementations.

## 8. Reproduction and exact artifacts

Workflow (first two folds):
https://github.com/fims9000/BeaconXAI/actions/runs/44534973588
- 6,7: https://github.com/fims9000/BeaconXAI/actions/runs/44534973588/artifacts/14536810072
- 8,9: https://github.com/fims9000/BeaconXAI/actions/runs/44534973588/artifacts/14536585349

Workflow (additional disjoint folds):
https://github.com/fims9000/BeaconXAI/actions/runs/44535597541
- 2,3: https://github.com/fims9000/BeaconXAI/actions/runs/44535597541/artifacts/14535706483
- 4,5: https://github.com/fims9000/BeaconXAI/actions/runs/44535597541/artifacts/14536765797

Each artifact contains: `per_sample.csv`, `per_subject.csv`,
`summary.csv`, `manifest.json` (dataset SHA256), `provenance.txt`.
Artifacts expire after 30 days; copy locally for long-term retention.

```bash
python scripts/preprocess_pamap2.py --test-subjects 6,7 --out data/pamap2_s6s7.npz
python scripts/run_q16_matched_ablation.py --npz-path data/pamap2_s6s7.npz --time-bins 12 --budget 16 --max-test 128 --repeats 3 --seed 42
```

The widely presented **46.09% hit@3** comes from a **different
24-component experiment** and must not be pooled with the present
36-component Q16 study.
