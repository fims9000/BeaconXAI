# Supplementary: v11 cross-dataset policy deltas

Included files:
- `final_policy_delta_summary_cross_dataset.csv` — HAR/PAMAP2/WISDM package.
- `final_policy_delta_summary_uwave.csv` — UWave-only package.
- `final_policy_delta_summary_heartbeat.csv` — Heartbeat-only package.
- `v11_full_summary.csv` — unified row-wise table across all completed v11 bundles.

Metrics are deltas vs `logit_panel` for TAN/Fuzzy.
Primary columns: `*_d_auroc`, `*_p_auroc`, `*_d_f1_10`, `*_p_f1_10`.
