# V3 Status Snapshot (2026-02-18)

## Scope Completed
- E0 OOF ensemble frontier completed (`e00`-`e04`).
- Aggressive target specialist scout completed (`e10`-`e14`).
- Orthogonal stage2 residual mode completed (`e20`).

## Key Outcomes
- Best aggressive specialist: `e14_target_ralph_20`
  - `bmc_last200_mean = 0.008540`
  - Improvement vs prior downsampled leader `b03_ranked128_twostage_w040` (`0.001755`): `+0.006785`
- Best orth profile: `e20_target_main_orth_stage2`
  - `bmc_last200_mean = 0.004313`
  - Improvement vs non-orth counterpart `e10_target_main` (`0.001682`): `+0.002631`
  - `bmc_last200 avg_corr_with_benchmark = -0.044` (benchmark-orthogonal behavior)
- E0 blend frontier did not beat the best full-data single model
  - `e00_equal_weight_top3 bmc_last200_mean = 0.001382`
  - `d02_confirm_ranked128_twostage_w050_full bmc_last200_mean = 0.001408`
  - delta: `-0.000026`

## Interpretation
- Target specialists and orth stage2 materially improved scout-phase BMC.
- Plain ensemble of existing full-data heads was not enough; stronger specialist heads should be scaled first, then re-ensembled.
- Scout/downsampled wins must be validated on `v5.2/full.parquet` before production decisions.

## Next Execution Plan
- Follow V4 full-scale spec:
  - `/Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration/arrowstreet_v4_full_scale_spec.md`
- Priority order:
  1. Full-data `target_ralph_20` specialist confirmation.
  2. Full-data orth specialist confirmation + beta sweep.
  3. Full-data specialist blend frontier using confirmed winners.
