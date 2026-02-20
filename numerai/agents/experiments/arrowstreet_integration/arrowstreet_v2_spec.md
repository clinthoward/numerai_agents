# Arrowstreet V2 Improvement Spec (Draft)

Date: 2026-02-17
Owner: modeling/agents
Status: Draft for approval

## 1) Objective
Improve Arrowstreet model quality (not just uniqueness) so it competes with the LGBM baseline on Numerai metrics, while keeping the architecture modular and reproducible in the current agents pipeline.

Primary success metrics:
- `bmc_last_200_eras.mean` (primary)
- `bmc_mean` (secondary)

Sanity metrics:
- `corr_mean`
- `avg_corr_with_benchmark`
- drawdown / sharpe stability

## 2) Current Baseline Context
- `v99_production` already trains multiple targets and averages predictions.
- `v99_production` already uses baskets + linkages in Arrowstreet two-stage mode.
- Current bottleneck is signal strength, not feature plumbing: Arrowstreet variants are diverse but underperforming in corr/BMC.

## 3) Scope
In scope:
1. Improve indirect feature selection quality.
2. Add stage-2 residual shrinkage control.
3. Add optional richer basket embedding mode.
4. Tighten implementation reliability/perf.
5. Run a scout round (5 configs) on downsampled data.

Out of scope (this round):
1. Major architecture replacement (deep nets, transformers, etc.).
2. Deployment/upload changes.
3. Full redesign of postprocess stack.

## 4) Proposed Model Changes

## 4.1 Indirect Feature Selection (P0)
Problem:
- `indirect_base` currently takes first `N` features by column order.

Change:
- Add `indirect_feature_selection` with options:
  - `first_n` (current behavior, default for backward compatibility)
  - `era_corr_ranked` (new)
- Add `indirect_feature_ranking_target` (default `target`)
- Add `indirect_feature_ranking_min_eras` (default `40`)

`era_corr_ranked` method:
1. Compute per-era corr(feature, y) on fit data.
2. Score feature by `abs(mean_corr) * (1 - std_corr)` (or equivalent monotonic stability score).
3. Rank descending by score and keep top `indirect_max_base_features`.

Implementation target:
- `/Users/clinthoward/Documents/numerai_agents/numerai/agents/code/modeling/models/arrowstreet_regressor.py`

## 4.2 Stage-2 Residual Weight (P0)
Problem:
- two-stage predictions are hard-added (`pred = pred1 + pred2`).

Change:
- Add `stage2_weight: float = 1.0`
- New formula in two-stage predict path:
  - `pred = pred1 + stage2_weight * pred2`

Implementation target:
- `/Users/clinthoward/Documents/numerai_agents/numerai/agents/code/modeling/models/arrowstreet_regressor.py`

## 4.3 Embedding Mode for Baskets (P1)
Problem:
- basket embeddings are group means only.

Change:
- Add `embedding_mode` to `BasketBuilder`:
  - `mean` (current default)
  - `pca` (new, optional)
- For `pca`, fit PCA on grouped feature block and use top components as embedding columns.

Implementation target:
- `/Users/clinthoward/Documents/numerai_agents/numerai/agents/code/modeling/models/arrowstreet_components.py`
- `/Users/clinthoward/Documents/numerai_agents/numerai/agents/code/modeling/models/arrowstreet_regressor.py`

## 4.4 Reliability + Perf Guardrails (P1)
1. Strict parameter handling:
- Fail on unexpected kwargs instead of silently swallowing them.

2. Skip unused feature builders:
- In `_build_feature_block`, do not compute basket/linkage internals when no corresponding output columns are requested.

3. Ridge numerical stability:
- Accumulate `xtx/xty` in float64 and cast output to configured dtype for prediction.

Implementation targets:
- `/Users/clinthoward/Documents/numerai_agents/numerai/agents/code/modeling/models/arrowstreet_regressor.py`
- `/Users/clinthoward/Documents/numerai_agents/numerai/agents/code/modeling/models/arrowstreet_components.py`

## 5) Backward Compatibility
Defaults preserve existing behavior:
- `indirect_feature_selection="first_n"`
- `stage2_weight=1.0`
- `embedding_mode="mean"`

No config breakage expected for existing variants.

## 6) Testing Plan
Unit tests (extend existing suite):
- `/Users/clinthoward/Documents/numerai_agents/numerai/agents/tests/test_arrowstreet_regressor.py`
- `/Users/clinthoward/Documents/numerai_agents/numerai/agents/tests/test_model_factory_arrowstreet.py`

Add tests for:
1. `stage2_weight` effect (shape + deterministic behavior).
2. `indirect_feature_selection="era_corr_ranked"` returns expected count and stable ordering.
3. `embedding_mode="pca"` smoke fit/predict path.
4. Unknown params raise clean `ValueError`.
5. `use_baskets/use_linkages` toggles avoid unnecessary computation paths.

Integration smoke:
- Existing Arrowstreet integration test continues to pass.

## 7) Experiment Plan (Round 1, Scout)
Dataset/protocol:
- Use downsampled data and benchmark models:
  - `numerai/v5.2/downsampled_full.parquet`
  - `numerai/v5.2/downsampled_full_benchmark_models.parquet`
- CV expanding with embargo 13, keep metrics aligned with skills.

Run 5 configs:
1. `a00_core_reference`
- Current Arrowstreet core behavior.

2. `a01_ranked64`
- `indirect_feature_selection="era_corr_ranked"`
- `indirect_max_base_features=64`

3. `a02_ranked128`
- `indirect_feature_selection="era_corr_ranked"`
- `indirect_max_base_features=128`

4. `a03_two_stage_weighted`
- `model_variant="residual_two_stage"`
- `stage2_model_type="lgbm"`
- `stage2_weight=0.4`

5. `a04_ranked64_pca`
- `indirect_feature_selection="era_corr_ranked"`
- `indirect_max_base_features=64`
- `embedding_mode="pca"`

Selection rule:
1. Rank by `bmc_last_200_eras.mean`.
2. Tie-break with `bmc_mean`.
3. Use `corr_mean` and drawdown as sanity filters.

## 8) Acceptance Criteria
Minimum criteria to proceed to scale phase:
1. At least one V2 config improves `bmc_last_200_eras.mean` vs `a00_core_reference` by >= `+0.0002`.
2. No catastrophic corr collapse (`corr_mean` not worse than baseline by >20% relative to current Arrowstreet control).
3. Tests pass and no new warnings/errors in fit/predict flow.

## 9) Scale/Follow-up (after scout)
If criteria pass:
1. Run winning config(s) on full data.
2. Run one confirmatory round for the strongest knob (feature ranking depth or stage2 weight).
3. Update experiment report with plots via `show_experiment benchmark`.

If criteria fail:
1. Drop PCA branch.
2. Focus next round on stage2 regularization/hyperparams and basket cluster-size sweeps.

## 10) Open Questions for Approval
1. Should we keep strict defaults for compatibility, or force ranked indirect selection immediately for all new variants?
2. Do you want `stage2_weight` exposed in `DEFAULT_VARIANTS` now, or only in experiment configs first?
3. For Round 1, should we include one additional config with `use_baskets=False` as an ablation sanity check?
