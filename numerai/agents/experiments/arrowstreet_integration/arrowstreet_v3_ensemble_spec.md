# Arrowstreet V3 Ensemble Spec (Draft)

Date: 2026-02-18
Owner: modeling/agents
Status: Draft for approval

## 1) Objective
Increase live-relevant BMC by moving from single-model tuning to a controlled ensemble layer on top of the V2 winners.

Primary success metric:
- `bmc_last_200_eras.mean`

Secondary metrics:
- `bmc_mean`
- `corr_mean`

Risk-control metrics:
- `avg_corr_with_benchmark`
- `bmc_last_200_eras.sharpe`
- rolling stability (no regime-specific collapse)

## 2) Current Baseline Context
Best current confirmed base model family:
- `d02_confirm_ranked128_twostage_w050_full`
  - `corr_mean=0.013317`
  - `bmc_mean=0.002277`
  - `bmc_last200_mean=0.001408`

Recent signal:
- Incremental gains from single-knob tuning are now small (`w=0.4 -> 0.5` gave a modest lift).
- This is a plateau pattern where blend-layer optimization usually has higher ROI.

## 3) Scope
In scope:
1. OOF-based blend optimization across existing full-data artifacts.
2. Multi-target specialist Arrowstreet base models for additional orthogonal signal.
3. Orthogonal residual head variants to improve BMC/uniqueness trade-off.
4. A strict scout -> scale -> confirm flow with explicit anti-overfit checks.

Out of scope (this round):
1. Pickle deployment/upload work.
2. Major deep-learning architecture changes.
3. Tournament-API automation changes.

## 4) V3 Workstreams

## 4.1 Workstream A: OOF Ensemble Frontier (P0)
Goal:
- Extract immediate gains using existing OOF predictions and learn robust blend weights.

Input candidates (initial):
- `c00`, `c01`, `d00`, `d01`, `d02` OOF predictions
- Existing Arrowstreet scout winners (`a02`, `b03`) if needed as diversity anchors
- Optional LGBM baseline OOF as a controlled high-corr component

Blend families:
1. Equal-weight rank average (control).
2. Non-negative ridge blend on OOF predictions.
3. Elastic-net blend with sparsity.
4. Penalized blend objective:
   - maximize `bmc_last200_mean - lambda * avg_corr_with_benchmark`

Planned implementation targets:
- New script: `numerai/agents/code/analysis/ensemble_frontier.py`
- Optional helper: `numerai/agents/code/analysis/ensemble_data.py`
- Artifacts:
  - `numerai/agents/experiments/arrowstreet_integration/ensemble_frontier_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/ensemble_frontier_summary.md`
  - `numerai/agents/experiments/arrowstreet_integration/ensemble_weights/*.json`

## 4.2 Workstream B: Multi-Target Specialist Arrowstreet (P0)
Goal:
- Add orthogonal signal by training d02-style Arrowstreet models on multiple targets, then stacking/averaging.

Approach:
1. Discover candidate targets from dataset (`target_*`) using existing helper utilities.
2. Select 3-5 targets using correlation/diversity filters.
3. Train one Arrowstreet model per selected target (same core params as `d02`).
4. Combine target-specific OOF predictions in blend layer (Workstream A).

Planned implementation targets:
- Reuse: `numerai/agents/code/signals/target_ensemble.py`
- New experiment configs under:
  - `numerai/agents/experiments/arrowstreet_integration/configs/e1x_*.py`

## 4.3 Workstream C: Orthogonal Residual Head (P1)
Goal:
- Improve BMC by reducing benchmark overlap in stage-2 correction.

Approach:
- Add optional stage-2 residual target mode in Arrowstreet:
  - `stage2_target_mode="residual_to_benchmark"`
  - target formula: `y_resid = y - beta * benchmark`
- Sweep `beta` in `{0.5, 0.75, 1.0}` with `stage2_weight` coupled around current best.

Planned implementation targets:
- `numerai/agents/code/modeling/models/arrowstreet_regressor.py`

## 4.4 Workstream D: Heterogeneous Blend Expansion (Stretch)
Goal:
- Add non-Arrowstreet bases only if A/B/C plateau.

Approach:
- Include one to two heterogeneous bases (e.g., stronger LGBM family variants) in blend frontier.
- Keep blend constraints to avoid over-allocating to high benchmark-corr models.

## 5) Evaluation and Anti-Overfit Protocol
1. Keep Numerai-aligned primary metric: `bmc_last_200_eras.mean`.
2. Use time-aware validation for blend fitting:
   - fit blend weights on older eras, evaluate on held-out later eras.
3. Report both:
   - full OOF metrics
   - last-200-era metrics
4. Add uncertainty checks:
   - block bootstrap confidence interval on `bmc_last200_mean`.
5. Do not accept a blend that improves BMC but causes an extreme benchmark-corr jump without a clear net benefit.

## 6) V3 Round Plan

## Round E0 (No retraining, fast frontier)
Use existing full-data OOF artifacts only.

Runs (blend candidates):
1. `e00_equal_weight_top3`
2. `e01_nonneg_ridge_top5`
3. `e02_elasticnet_top5`
4. `e03_penalized_low_benchmark_corr`
5. `e04_stability_weighted_blend`

Selection rule:
1. rank by `bmc_last200_mean`
2. tie-break by `bmc_mean`
3. reject unstable/high-risk profiles by benchmark-corr and rolling stability checks

## Round E1 (Scout multi-target specialists, downsampled)
Train 4-5 d02-style target-specific configs on downsampled data.

Example configs:
1. `e10_target_main`
2. `e11_target_alt1`
3. `e12_target_alt2`
4. `e13_target_alt3`
5. `e14_target_alt4`

Then blend E1 outputs with frontier methods from E0.

## Round E2 (Scale winners, full data)
Scale top 2-3 combined approaches from E0/E1.

Example scale runs:
1. `f00_best_frontier_no_new_training`
2. `f01_best_multitarget_blend`
3. `f02_best_multitarget_plus_orthogonal_head`

## Round E3 (Confirm)
Narrow confirm sweep around best blend/penalty settings on full data.

## 7) Acceptance Criteria
Proceed to production-candidate designation only if:
1. `bmc_last200_mean` improves vs `d02` by at least `+0.00005`.
2. Improvement is also visible in `bmc_mean` (not last-200-only noise).
3. No major instability in rolling eras.
4. Benchmark correlation remains in an acceptable range for the achieved BMC gain.

## 8) Expected Deliverables
1. New V3 spec (this file).
2. Blend frontier script + weight artifacts.
3. E0/E1/E2/E3 configs and result JSONs.
4. Updated `experiment.md` with:
   - methods
   - results tables
   - final recommended production candidate
   - fallback candidate

## 9) Open Questions for Approval
1. Should we allow LGBM baseline OOF into the frontier from E0, or keep first pass Arrowstreet-only?
2. For target specialists, do you want conservative (`max 3 targets`) or aggressive (`max 5 targets`) expansion?
3. Should we implement Workstream C (orthogonal residual head) immediately, or gate it behind E0/E1 results?
