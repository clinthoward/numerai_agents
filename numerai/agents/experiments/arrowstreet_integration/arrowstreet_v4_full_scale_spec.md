# Arrowstreet V4 Full-Scale Spec

Date: 2026-02-18
Experiment root: `/Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration`

## Goal
Convert V3 scout winners into robust full-data improvements for Classic-style metrics, while adding a stronger specialist ensemble layer.

## Inputs from V3
- `e14_target_ralph_20` (downsampled): strongest `bmc_last200_mean`.
- `e20_target_main_orth_stage2` (downsampled): strongest benchmark-orthogonal profile (`bench_corr_last200 < 0`).
- E0 blend frontier did not beat full-data `d02` and should not be promoted as-is.

## Hypotheses
1. `target_ralph_20` advantage survives full-data CV and yields a higher `bmc_last200_mean` than `d02`.
2. Orth stage2 (`residual_to_benchmark`) provides a complementary low-benchmark-corr specialist that improves blend uniqueness.
3. A constrained specialist blend outperforms any single full-data model on `bmc_last200_mean` while keeping corr viable.

## Workstream A: Full-data specialist confirmation
Create full-data configs (same CV protocol as `d02`):
- `f00_full_target_ralph20_twostage.py`
- `f01_full_target_main_orth_twostage.py`
- `f02_full_target_ralph20_orth_twostage.py` (combo stress test)

Shared defaults:
- `v5.2/full.parquet`
- `medium` feature set
- `residual_two_stage`
- basket/linkage enabled (`basket_cluster_sizes=[16]`, linkage stats unchanged)

Orth settings for `f01/f02`:
- `stage2_target_mode='residual_to_benchmark'`
- `stage2_benchmark_col='v52_lgbm_ender20'`
- beta sweep in Workstream B

## Workstream B: Orth intensity sweep (full data)
Add a narrow beta sweep off best orth config:
- `f10_full_target_main_orth_beta075.py`
- `f11_full_target_main_orth_beta100.py`
- `f12_full_target_main_orth_beta125.py`

Purpose:
- Find best BMC/benchmark-corr tradeoff using `stage2_benchmark_beta`.

## Workstream C: Specialist ensemble frontier (full data)
Candidates for blend frontier:
- baseline full model: `d02`
- best ralph specialist from Workstream A
- best orth specialist from Workstream A/B
- optional `d01` as stabilizer

Blend families:
- equal-weight top-3 specialist blend
- non-negative ridge blend
- elastic-net blend
- benchmark-corr-penalized blend

## Acceptance criteria
Primary promotion gate:
- `bmc_last200_mean` must exceed `d02` by at least `+0.00010` on full data.

Secondary risk gates:
- corr mean not worse than `d02` by more than `-0.0015`.
- benchmark corr should be stable (single-model or blend rationale must be explicit).

## Deliverables
- New full-data configs under `configs/`.
- Results and predictions for each run under `results/` and `predictions/`.
- `v4_full_scale_summary.csv` and `.md` ranking table.
- `experiment.md` section with promotion decision and fallback ordering.

## Promotion logic
If no full-data specialist beats `d02`:
- keep `d02` production baseline,
- retain orth/specialist models as ensemble candidates for future rounds.

If a specialist or blend beats `d02` under gates:
- promote winner to next production candidate,
- keep runner-up as fallback profile.
