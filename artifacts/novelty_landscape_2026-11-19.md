# Scientific novelty audit — BeaconXAI temporal counterfactual explanations (2026-11-19)

**Status: research map, not a claim of novelty.** Public literature checked against the
existing experimental branch `feature/contrastive-q16-20261119`. Scientific
conclusions require an independent matched benchmark and validation by experts.

## 1. Closest competitors

| Published method | Existing capability that overlaps with BeaconXAI | Why the current prototype is not novel merely by using it | Appropriate benchmark |
|---|---|---|---|
| LIMESegment (Sivill & Flach, AISTATS 2022) | Semantically meaningful time-series segmentation and realistic perturbations in a model-agnostic explainer | Segmenting/perturbing signals is established | Local fidelity, perturbation realism, model-call cost |
| Native Guide (Delaney et al., 2020) | Case-based counterfactuals from real alternative-class instances | Using a real donor for a counterfactual is established | Validity, proximity, sparsity, continuity |
| DiscoX (2024) | Finds anomalous subsequences and substitutes nearest target-class subsequences | Target-class nearest donors/discord replacements are established | Validity and plausibility under matched Q |
| FastPACE (2026) | Explicitly addresses speed and strict counterfactual validity for time-series | 'Fast counterfactual search' alone cannot be called novel | Q-to-validity curve and runtime vs official implementation |
| LASTS (Spinnato et al., TKDD 2023) | Model-agnostic saliency, factual/counterfactual subsequence rules, exemplar/counterexamples | Explanation rules+local examples are established | Rule fidelity and stability on unseen subjects |
| TSProto (Bobek & Nalepa, Information Fusion 2025) | Integrates important fragments into interpretable prototypes and rule-based trees | Aggregating fragments into class 'passports' is established | Rule fidelity, coverage, expert rating |
| INSHAPE (Lee et al., IJCAI 2026) | Instance-level shapelets plus aggregation into class-level prototype patterns | Local-to-global shapelet aggregation is established | Segment alignment, stability, generalization |
| TimeSHAP (Bento et al., KDD 2021) | Time-specific attributions with pruning/computation reduction | Reducing query cost or pruning by itself is established | Query/fidelity cost curve, including preprocessing |
| Towards Plausibility in Time-Series CF (Kostrzewa et al., 2026) | Uses soft-DTW with nearest target-class instances to improve realism | Temporal alignment and kNN-based plausibility are already researched | Reference-distribution realism and validity |

Sources:
- https://proceedings.mlr.press/v151/sivill22a.html
- https://arxiv.org/abs/2009.13211
- https://doi.org/10.1007/s10618-024-01028-9
- https://doi.org/10.1007/s10618-026-01242-7
- https://doi.org/10.1145/3624480
- https://doi.org/10.1016/j.inffus.2025.103357
- https://doi.org/10.24963/ijcai.2026/496
- https://arxiv.org/abs/2603.08349
- https://research.feedzai.com/wp-content/uploads/2022/08/Bento_TimeSHAP_KDD2021.pdf

**Note:** TreeSHAP on the engineered ExtraTrees input is a strong,
model-specific low-cost baseline. Its feature representation is not automatically
equivalent to raw temporal patches; mapping must be justified.

## 2. What the project has actually demonstrated

Earlier matched PAMAP2, 36 components, Q16, mean of four held-out folds:

- original `adaptive_v2`: 41.80% hit@3 on **synthetic donor-patch localization**;
- cheap adaptive-v2 prior Q0: 32.03%;
- new class-pair heuristic Q16: 32.55%;
- uniform selection Q16: 22.40%.

The class mean/median pilot then found very few true target-class switches
(typically <1% of individual replacements) and frequently negative source
score changes without any positive rival change.

These are two **different** tasks. Neither is validated natural/causal
explanation or a human expert study.

## 3. Testable open hypothesis: budget allocation + evidence reliability

Proposed task, not yet a method claim: given a black-box classifier,
time-series input x, output y=A, candidate competitors B_j and a **single
global budget Q for the entire object**, adaptively select the next
(class B_j, component g, replacement source z), or ABSTAIN:

    action_t = pi(x, f(x), prior_train, observations_{1:t-1})
    sum_t #model_evaluations(action_t) <= Q

The utility must be measurable on held-out subjects, e.g.:

    U = factual/contrastive evidence fidelity
        - lambda * (#model evaluations)
        - mu * perturbation-artifact penalty
        - nu * unsupported assertion rate

**Possible research contribution** (NOT established novelty): a model-agnostic,
resource-aware, *selective explanation policy* that allocates queries
between rival classes, components and physically plausible donors,
and reports a contrastive explanation only when measured evidence is
stable across multiple interventions.

Why potentially worthwhile:
- previous Q16 stage had no benefit over Q0 in one subject fold;
- mean/median target replacements rarely support a class transition;
- prior code consumed Q16 **per target class**, which is incompatible
  with a single total-Q TinyML requirement.

Novelty depends on related active-learning, bandit, sequential-design and
selective-XAI papers and must be independently checked. No first-ever
claim is justified.

## 4. Falsifiable experiments, order and safety gates

1. **Real donor baseline**: same model, same object, same selected region,
   same Q=16; compare mean, median, actual medoid, random real donor,
   nearest donor by untouched context and boundary-aligned donor.
   Reevaluate p(A), p(B), actual flip and perturbation magnitude.
   Include incorrectly predicted test examples.
2. **Separation of performance and realism**: no gain if flip success is
   traded for much larger distortion or seam discontinuity. Quantify
   both; characterize raw training memory and retrieval cost.
3. **One-global-budget competitors**: a single target B1 vs equal
   allocation to B1/B2 vs a sequential evidence-based allocation policy
   at Q=8/16/32. Compare true target switches and directive movement.
4. **Calibrate abstention on training-only subject-held-out validation**:
   never choose thresholds from the final outer held-out subject pair.
   Report coverage vs unsupported explanation rate and Q.
5. **Independent replications**: several held-out participants, at least
   one raw-input classifier, and separate perturbation-realism controls.
6. **Authoritative competitors**: official LIMESegment and Native Guide/
   DiscoX/FastPACE where code/dependencies allow, not custom 'like'
   estimators presented as canonical implementations.

**Stop criterion:** if the strategy cannot improve any meaningful quality-
cost tradeoff against cheap Q0, a fixed-Q policy or a published strong
baseline with comparable data and model, it is not a demonstrated advance.

## 5. Product-level explanation

For each explanation present:
- original predicted A, chosen rival B (and how B was selected);
- sensor/time of the **actually tested** component and actual donor provenance;
- p(A), p(B) before/after the intervention, and the final top-1 class;
- exact model calls charged to the object;
- measured evidence confidence/instability and optional ABSTAIN;
- example trace overlay and two distinct signed score-change maps.

Do not narrate 'feature characteristic of running' if only p(walking) fell.
Do not call smoothed class averages physical counterfactuals without validation.

## 6. Scope limits

Existing scripts on GitHub use engineered ExtraTrees features, not an ANFIS
classifier. No claim about TinyML hardware is possible until on-device
RAM/latency/energy are measured. Nearest donor requires stored reference
windows and likely contradicts MCU memory budgets unless an offline
compressed dictionary is designed and tested.
