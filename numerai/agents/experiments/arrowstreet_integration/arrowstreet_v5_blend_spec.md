# Arrowstreet V5 Blend Spec

Date: 2026-02-19
Experiment root: `/Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration`

## Objective
Promote a corr-balanced ensemble candidate that preserves most of `d02` corr while materially improving BMC on tournament-target evaluation.

## Inputs
- Canonical full-data single-run summary:
  - `/Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration/v4_full_scale_tournament_summary.csv`
- Canonical frontier summary:
  - `/Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration/v4_full_frontier_canonical_summary.csv`
- Pareto blend sweep:
  - `/Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration/v5_pareto_blend_summary.csv`
- Candidate diagnostics:
  - `/Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration/v5_candidate_diagnostics.csv`
- Window diagnostics:
  - `/Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration/v5_window_diagnostics.csv`

## Candidate Set
- Aggressive: `e03_penalized_low_benchmark_corr`
- Corr-balanced: `g11_60e03_40d02`, `g12_50e03_50d02`, `g13_40e03_60d02`, `g14_30e03_70d02`
- Baseline: `d02_confirm_ranked128_twostage_w050_full`

## Decision Gates
Primary:
- `delta_bmc_last200_vs_d02 >= +0.0008`

Corr guardrails:
- Preferred: `delta_corr_vs_d02 >= -0.0010`
- Hard fail: `delta_corr_vs_d02 < -0.0015`

Stability:
- `last100` and `last200` BMC both above `d02`

## Current Recommendation
- Corr-floor compliant winner: `g12_50e03_50d02`
- Higher-BMC alternative: `g11_60e03_40d02`
- Aggressive profile: `e03`

## Next Implementation Tasks
1. Add corr-constrained blend mode to `ensemble_frontier.py` (explicit corr floor in optimization).
2. Add era-block holdout validation for blend selection (train/selection on earlier eras, score on final eras).
3. Emit a final promotion table with three slots:
   - `primary_balanced`
   - `secondary_high_bmc`
   - `fallback_stable`
