# Amendment A1 — CharacterTrajectories equal-length archive has 119, not 182 time points

**Date:** 22 November 2026. **Pre-model, pre-TEST-inference technical amendment.**

The frozen preregistration expected the UEA CharacterTrajectories _eq archive to have 182 time points. This was incorrect. In a separate failing GitHub Actions preprocessor job [45024329103](https://github.com/fims9000/BeaconXAI/actions/runs/45024329103), after running all 34 focused technical tests, the data loader reported **actual unmodified shapes**:

- TRAIN (1422, 3, 119);
- TEST (1436, 3, 119).

The preprocessor raised a ValueError before any model fitting, test predictions, risk calibration or black-box interventions. The 182-length value appears in general dataset tables but the aeon 1.5.0 **_eq variant** is 119. The equal-length archive is a transformed representation of originally variable-duration pen movements.

**Amended, without looking at TEST predictions:**
- Accept exact (1422,3,119) and (1436,3,119), preserving [N,119,3] rather than [N,182,3].
- Apply the originally preregistered ten equal time bins and taper=2 **on 119 points**, rather than 182.
- Preserve splits, labels, seed 57, ExtraTrees 120, bank16/seed67, TEST subsample seed5733, online Q16 shared over D1+D2, edit cap 35%, nine risk features, logistic C=1, five fixed candidate quantiles, Bonferroni CP alpha=.02 and risk bound .5, all metrics.
- Record actual T=119 and this A1 amendment in the output manifest.

This is a **documented departure**, not a retroactively unchanged original preregistration. The pre-inference failure remains visible. The amendment does not authorize tuning after seeing model or TEST outcomes.
