# Arrowstreet Model Evolution Audit (2026-02-17)

## Abstract
The current Arrowstreet implementation is producing materially weaker corr and BMC than the LGBM reference baseline on the same downsampled v5.2 setup. The model is unique (lower corr vs benchmark) but underpowered. The most likely path forward is to improve signal quality (feature selection + embedding quality + stage-2 control), not add more post-processing.

## Scope
Reviewed:
- `numerai/agents/code/modeling/models/arrowstreet_regressor.py`
- `numerai/agents/code/modeling/models/arrowstreet_components.py`
- Existing Arrowstreet experiment artifacts under:
  - `numerai/agents/experiments/arrowstreet_integration/results/arrowstreet_core_downsampled.json`
  - `agents/experiments/arrowstreet_integration/variant_benchmarks_downsampled/`

## Current Results Snapshot

### Pipeline result (downsampled full, CV/OOF)
| run | corr_mean | bmc_mean | avg_corr_with_benchmark |
|---|---:|---:|---:|
| small_lgbm_ender20_baseline | 0.02190 | 0.00367 | 0.55826 |
| arrowstreet_core_downsampled | 0.01070 | 0.00196 | 0.28957 |

### Variant ladder (downsampled train/validation benchmark)
| variant | corr_mean | bmc_mean | bmc_sharpe | avg_corr_with_benchmark |
|---|---:|---:|---:|---:|
| v00_lgbm_baseline | 0.02137 | 0.00207 | 0.18205 | 0.54944 |
| v10_arrowstreet_core | 0.01093 | 0.00139 | 0.13071 | 0.28463 |
| v20_two_stage | 0.01150 | 0.00142 | 0.13250 | 0.30020 |
| v30_target_ensemble | 0.01176 | 0.00078 | 0.07235 | 0.31934 |
| v40_calibrated | 0.01177 | 0.00092 | 0.09088 | 0.31490 |
| v50_neutralized | 0.01071 | 0.00088 | 0.08401 | 0.28591 |
| v99_production | 0.01071 | 0.00088 | 0.08401 | 0.28591 |

## Findings
1. Arrowstreet is diverse but not strong enough.
- Corr vs benchmark is much lower (roughly 0.29-0.32), but raw corr drops by roughly half versus baseline.
- Net effect: lower BMC despite uniqueness.

2. Post-processing stack is not rescuing model quality.
- Target ensemble, calibration, and neutralization do not recover BMC; they mostly reduce it in this ladder.

3. Core implementation likely bottlenecks:
- `indirect_base` uses the first N features by column order (`base_features[:indirect_max_base_features]`), not signal-ranked features.
- Default group construction is round-robin by column order (not learned/semantic grouping).
- Two-stage prediction always adds stage1 + stage2 with fixed weight 1.0 (no shrinkage control).
- Unknown kwargs are silently ignored (`**_`), making config mistakes hard to detect.

4. Compute path is heavier than needed in some toggles.
- Feature block generation still computes basket/linkage internals even when output feature name lists are empty.

## Recommended Improvements (Priority Order)

### P0: Improve core signal quality
1. Replace first-N indirect feature selection with scored selection.
- Select indirect features by era-wise corr stability on train folds (or MI proxy), not column order.

2. Add stage-2 residual weight.
- Add `stage2_weight` (default 1.0, sweep 0.2-1.0) so stage2 does not over-correct stage1.

3. Upgrade group embeddings.
- Add configurable embedding modes (`mean`, `pca`, `sparse_random_projection`) and compare against current mean-group embeddings.

### P1: Improve model robustness / maintainability
4. Enforce strict params.
- Fail on unknown kwargs in `ArrowstreetRegressor` to prevent silent config drift.

5. Skip unused feature builders.
- In `_build_feature_block`, bypass basket/linkage computation when corresponding feature name lists are empty.

6. Improve numerical stability in ridge accumulation.
- Accumulate `xtx/xty` in float64 and cast coefficients back to `dtype_float` for prediction.

### P2: Controlled expansion once P0 is stable
7. Multi-scale baskets and linkages.
- Sweep `basket_cluster_sizes` (e.g., `[8,16,32]`) and linkage stats (add median/quantiles).

8. Learn blend between Arrowstreet and baseline.
- Add a simple linear blend on OOF predictions to trade off uniqueness vs corr.

## Next Experiment Round (Scout, downsampled)
Run 5 configs in one round:
1. `a00_core_reference`: current Arrowstreet core.
2. `a01_ranked64`: ranked indirect feature selection, N=64.
3. `a02_ranked128`: ranked indirect feature selection, N=128.
4. `a03_two_stage_weighted`: residual two-stage with `stage2_weight=0.4`.
5. `a04_ranked64_pca`: PCA embedding mode + ranked indirect features.

Primary ranking metric: `bmc_last_200_eras.mean`.
Tie-breakers: `bmc_mean`, then `corr_mean`.

## Round 1 Results (Executed 2026-02-17)

Artifacts:
- Summary CSV: `numerai/agents/experiments/arrowstreet_integration/round1_scout_summary.csv`
- Summary table: `numerai/agents/experiments/arrowstreet_integration/round1_scout_summary.md`
- Result JSONs: `numerai/agents/experiments/arrowstreet_integration/results/`

Ranked by `bmc_last_200_eras.mean`:

| run | corr_mean | corr_sharpe | bmc_mean | bmc_sharpe | avg_corr_with_benchmark | bmc_last_200_eras.mean | bmc_last_200_eras.sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| a02_ranked128 | 0.010233 | 0.780604 | 0.002082 | 0.188962 | 0.267102 | 0.001773 | 0.164094 |
| a01_ranked64 | 0.010829 | 0.869948 | 0.002185 | 0.206156 | 0.291547 | 0.001762 | 0.167251 |
| a03_two_stage_weighted | 0.012136 | 0.947441 | 0.002295 | 0.208300 | 0.320236 | 0.001738 | 0.155827 |
| a00_core_reference | 0.010691 | 0.834393 | 0.001945 | 0.175867 | 0.289705 | 0.001388 | 0.126571 |
| a04_ranked64_pca | 0.010935 | 0.877595 | 0.002182 | 0.200861 | 0.287808 | 0.001330 | 0.124079 |

Delta vs control (`a00_core_reference`):
- `a02_ranked128`: `+0.000385` on `bmc_last_200_eras.mean`
- `a01_ranked64`: `+0.000374` on `bmc_last_200_eras.mean`
- `a03_two_stage_weighted`: `+0.000349` on `bmc_last_200_eras.mean`
- `a04_ranked64_pca`: `-0.000058` on `bmc_last_200_eras.mean`

Round-1 decision:
- Keep ranked indirect selection (`era_corr_ranked`) as the strongest and most consistent improvement.
- PCA embedding under current settings did not improve the primary metric.
- Two-stage weighted improved both corr and full BMC, but slightly trailed ranked-only on `bmc_last_200_eras.mean`; keep it as a strong co-candidate for next round.

## Round 2 Results (Executed 2026-02-17)

Round intent:
- Hold the strongest Round-1 setting (`ranked128`) and test:
1. Larger indirect set (`192`)
2. Two-stage residual weights (`0.2`, `0.4`, `0.7`)

Artifacts:
- Summary CSV: `numerai/agents/experiments/arrowstreet_integration/round2_scout_summary.csv`
- Summary table: `numerai/agents/experiments/arrowstreet_integration/round2_scout_summary.md`
- Result JSONs: `numerai/agents/experiments/arrowstreet_integration/results/`

Ranked by `bmc_last_200_eras.mean`:

| run | corr_mean | corr_sharpe | bmc_mean | bmc_sharpe | avg_corr_with_benchmark | bmc_last_200_eras.mean | bmc_last_200_eras.sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| b03_ranked128_twostage_w040 | 0.012004 | 0.932502 | 0.002396 | 0.214298 | 0.315615 | 0.001755 | 0.157373 |
| b00_ranked128_standard | 0.010192 | 0.784229 | 0.002059 | 0.187220 | 0.266991 | 0.001746 | 0.161739 |
| b04_ranked128_twostage_w070 | 0.011726 | 0.908773 | 0.002237 | 0.200596 | 0.312604 | 0.001577 | 0.141286 |
| b02_ranked128_twostage_w020 | 0.011865 | 0.931234 | 0.002263 | 0.203614 | 0.316439 | 0.001577 | 0.143095 |
| b01_ranked192_standard | 0.009326 | 0.736322 | 0.001708 | 0.158280 | 0.248020 | 0.001449 | 0.135627 |

Delta vs Round-2 control (`b00_ranked128_standard`):
- `b03_ranked128_twostage_w040`: `+0.000009` on `bmc_last_200_eras.mean`, `+0.000337` on `bmc_mean`, `+0.001812` on `corr_mean`
- `b02_ranked128_twostage_w020`: `-0.000169` on `bmc_last_200_eras.mean`
- `b04_ranked128_twostage_w070`: `-0.000169` on `bmc_last_200_eras.mean`
- `b01_ranked192_standard`: `-0.000297` on `bmc_last_200_eras.mean`

Cross-round check:
- Round-1 best `a02_ranked128` remains the best primary metric so far (`bmc_last_200_eras.mean=0.001773`).
- Round-2 best (`b03_ranked128_twostage_w040`) improves corr and full BMC materially but is slightly lower on the primary metric (`0.001755`).

Round-2 decision:
- Keep two co-leaders for scale phase:
1. `a02_ranked128` (best primary metric)
2. `b03_ranked128_twostage_w040` (best corr/full-BMC profile)
- Drop `192` indirect feature count and off-center stage2 weights (`0.2`, `0.7`) for now.

## Repro Notes
- Existing benchmark artifacts were generated under `agents/experiments/...` (repo root) while other runs are under `numerai/agents/experiments/...`.
- Keep output roots consistent in the next round to avoid split histories.

## Scale Results (Executed 2026-02-17)

Scale intent:
- Validate the two scout co-leaders on full `v5.2/full.parquet` data:
1. `c00_scale_ranked128_full` (`ranked128` standard)
2. `c01_scale_ranked128_twostage_w040_full` (`ranked128` + two-stage LGBM, `stage2_weight=0.4`)

Artifacts:
- Summary CSV: `numerai/agents/experiments/arrowstreet_integration/scale_summary.csv`
- Summary table: `numerai/agents/experiments/arrowstreet_integration/scale_summary.md`
- Result JSONs:
1. `numerai/agents/experiments/arrowstreet_integration/results/c00_scale_ranked128_full.json`
2. `numerai/agents/experiments/arrowstreet_integration/results/c01_scale_ranked128_twostage_w040_full.json`

Ranked by `bmc_last_200_eras.mean`:

| run | corr_mean | corr_sharpe | bmc_mean | bmc_sharpe | avg_corr_with_benchmark | bmc_last_200_eras.mean | bmc_last_200_eras.sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| c01_scale_ranked128_twostage_w040_full | 0.013293 | 0.990489 | 0.002249 | 0.198063 | 0.353846 | 0.001364 | 0.124432 |
| c00_scale_ranked128_full | 0.012560 | 0.937187 | 0.002179 | 0.190820 | 0.333071 | 0.001226 | 0.111450 |

Delta vs scale control (`c00_scale_ranked128_full`):
- `c01_scale_ranked128_twostage_w040_full`: `+0.000138` on `bmc_last_200_eras.mean`
- `c01_scale_ranked128_twostage_w040_full`: `+0.000070` on `bmc_mean`
- `c01_scale_ranked128_twostage_w040_full`: `+0.000733` on `corr_mean`
- `c01_scale_ranked128_twostage_w040_full`: `+0.012982` on `bmc_last_200_eras.sharpe`
- `c01_scale_ranked128_twostage_w040_full`: `+0.020775` on `avg_corr_with_benchmark`

Scale decision:
- Promote `c01_scale_ranked128_twostage_w040_full` as the current best full-data candidate in this experiment line.
- Keep `c00_scale_ranked128_full` as a lower-benchmark-correlation fallback profile.
- Next confirmatory step: run a narrow full-data weight sweep around `stage2_weight=0.4` (`0.3`, `0.4`, `0.5`) before production lock.

## Confirmatory Sweep Setup (Prepared 2026-02-18)

Prepared configs (full-data, same protocol as scale phase):
1. `numerai/agents/experiments/arrowstreet_integration/configs/d00_confirm_ranked128_twostage_w030_full.py`
2. `numerai/agents/experiments/arrowstreet_integration/configs/d01_confirm_ranked128_twostage_w040_full.py`
3. `numerai/agents/experiments/arrowstreet_integration/configs/d02_confirm_ranked128_twostage_w050_full.py`

Run commands:
```bash
/Users/clinthoward/Documents/numerai_agents/.venv/bin/python -m agents.code.modeling --config /Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration/configs/d00_confirm_ranked128_twostage_w030_full.py
/Users/clinthoward/Documents/numerai_agents/.venv/bin/python -m agents.code.modeling --config /Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration/configs/d01_confirm_ranked128_twostage_w040_full.py
/Users/clinthoward/Documents/numerai_agents/.venv/bin/python -m agents.code.modeling --config /Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration/configs/d02_confirm_ranked128_twostage_w050_full.py
```

## Confirmatory Sweep Results (Executed 2026-02-18)

Artifacts:
- Summary CSV: `numerai/agents/experiments/arrowstreet_integration/confirmatory_summary.csv`
- Summary table: `numerai/agents/experiments/arrowstreet_integration/confirmatory_summary.md`
- Result JSONs:
1. `numerai/agents/experiments/arrowstreet_integration/results/d00_confirm_ranked128_twostage_w030_full.json`
2. `numerai/agents/experiments/arrowstreet_integration/results/d01_confirm_ranked128_twostage_w040_full.json`
3. `numerai/agents/experiments/arrowstreet_integration/results/d02_confirm_ranked128_twostage_w050_full.json`

Ranked by `bmc_last_200_eras.mean`:

| run | corr_mean | corr_sharpe | bmc_mean | bmc_sharpe | bmc_avg_corr_benchmark | bmc_last200_mean | bmc_last200_sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| d02_confirm_ranked128_twostage_w050_full | 0.013317 | 0.990887 | 0.002277 | 0.200619 | 0.353630 | 0.001408 | 0.128878 |
| d01_confirm_ranked128_twostage_w040_full | 0.013296 | 0.990098 | 0.002252 | 0.198230 | 0.353793 | 0.001378 | 0.125737 |
| d00_confirm_ranked128_twostage_w030_full | 0.013271 | 0.989001 | 0.002233 | 0.196363 | 0.353805 | 0.001354 | 0.123368 |

Delta vs confirm control (`d01_confirm_ranked128_twostage_w040_full`):
- `d02_confirm_ranked128_twostage_w050_full`: `+0.000030` on `bmc_last200_mean`
- `d02_confirm_ranked128_twostage_w050_full`: `+0.000026` on `bmc_mean`
- `d02_confirm_ranked128_twostage_w050_full`: `+0.000021` on `corr_mean`
- `d02_confirm_ranked128_twostage_w050_full`: `+0.003141` on `bmc_last200_sharpe`

Repeatability check:
- `d01_confirm_ranked128_twostage_w040_full` reproduces `c01_scale_ranked128_twostage_w040_full` closely:
1. `corr_mean`: `+0.000003`
2. `bmc_mean`: `+0.000003`
3. `bmc_last200_mean`: `+0.000014`

Confirmatory decision:
- Promote `stage2_weight=0.5` (`d02_confirm_ranked128_twostage_w050_full`) as the current best setting within this model family.
- Keep `stage2_weight=0.4` (`d01`) as the nearest stable fallback.

## Ensemble Frontier Results (Executed 2026-02-18)

Artifacts:
- Summary CSV: `numerai/agents/experiments/arrowstreet_integration/ensemble_frontier_summary.csv`
- Summary table: `numerai/agents/experiments/arrowstreet_integration/ensemble_frontier_summary.md`
- Blend prediction files: `numerai/agents/experiments/arrowstreet_integration/predictions/e0*.parquet`
- Blend weights: `numerai/agents/experiments/arrowstreet_integration/ensemble_weights/`

Result:
- Best blend was `e00_equal_weight_top3` with `bmc_last200_mean=0.001382`.
- Current full-data single-model leader `d02_confirm_ranked128_twostage_w050_full` remained higher at `bmc_last200_mean=0.001408`.
- Decision: do not promote an E0 blend to production; keep frontier machinery for later rounds once stronger specialist heads are available.

## Round 3 V3 Aggressive Target + Orth Scout Results (Executed 2026-02-18)

Artifacts:
- Summary CSV: `numerai/agents/experiments/arrowstreet_integration/round3_v3_targets_summary.csv`
- Summary table: `numerai/agents/experiments/arrowstreet_integration/round3_v3_targets_summary.md`
- Result JSONs: `numerai/agents/experiments/arrowstreet_integration/results/e1*.json` and `e20_target_main_orth_stage2.json`

Ranked by `bmc_last200_mean`:

| run | data | target | stage2_mode | bmc_last200 | bmc | corr | bench_corr_last200 |
|---|---|---|---|---:|---:|---:|---:|
| e14_target_ralph_20 | v5.2/downsampled_full.parquet | target_ralph_20 | residual | 0.008540 | 0.008816 | 0.023385 | 0.347 |
| e20_target_main_orth_stage2 | v5.2/downsampled_full.parquet | target | residual_to_benchmark | 0.004313 | 0.004464 | 0.004297 | -0.044 |
| e13_target_claudia_20 | v5.2/downsampled_full.parquet | target_claudia_20 | residual | 0.002675 | 0.002857 | 0.008353 | 0.206 |
| e10_target_main | v5.2/downsampled_full.parquet | target | residual | 0.001682 | 0.002315 | 0.011926 | 0.323 |
| e12_target_jasper_20 | v5.2/downsampled_full.parquet | target_jasper_20 | residual | 0.001412 | 0.001989 | 0.009988 | 0.300 |
| e11_target_teager2b_20 | v5.2/downsampled_full.parquet | target_teager2b_20 | residual | 0.001005 | 0.001209 | 0.009369 | 0.284 |

Interpretation:
- `target_ralph_20` materially outperformed all prior downsampled runs on the primary metric.
- Orth residual mode (`stage2_target_mode=residual_to_benchmark`) produced large BMC gains and flipped benchmark correlation negative on the last-200 era window, with lower plain corr.
- These are scout/downsampled results and are not directly comparable to full-data `d02` levels without confirmatory full-data scale runs.

Round-3 decision:
- Promote two full-data confirmatory tracks:
1. target specialist: `target_ralph_20` with residual two-stage setup.
2. orth specialist: `target` with `stage2_target_mode=residual_to_benchmark`.
- Keep E0 blend framework; revisit blending after these tracks are validated on full data.

## V4 Full-Data Scale Results (Executed 2026-02-19)

Artifacts:
- Run summary (as originally produced by run queue): `numerai/agents/experiments/arrowstreet_integration/v4_full_scale_summary.csv`
- Frontier summary: `numerai/agents/experiments/arrowstreet_integration/v4_full_frontier_summary.csv`
- Run result JSONs: `numerai/agents/experiments/arrowstreet_integration/results/f*.json`

Raw V4 observation:
- `f02_full_target_ralph20_orth_twostage` and `f00_full_target_ralph20_twostage` appeared strongest when scored on each run's configured training target.

Critical evaluation note:
- Cross-target runs are not directly comparable if ranked using their own training target in results JSONs.
- For tournament-facing decisions, all candidates were re-scored on a shared canonical target (`target`) over common OOF rows.

Canonical tournament-target artifacts:
- `numerai/agents/experiments/arrowstreet_integration/v4_full_scale_tournament_summary.csv`
- `numerai/agents/experiments/arrowstreet_integration/v4_full_scale_tournament_summary.md`

Canonical ranking (by `bmc_last200_mean` on `target`):
1. `f10_full_target_main_orth_beta075`: `0.003299`
2. `f01_full_target_main_orth_twostage`: `0.003270`
3. `f11_full_target_main_orth_beta100`: `0.003256`
4. `f12_full_target_main_orth_beta125`: `0.003200`
5. `f02_full_target_ralph20_orth_twostage`: `0.002567`
6. `d02_confirm_ranked128_twostage_w050_full`: `0.001408`
7. `d01_confirm_ranked128_twostage_w040_full`: `0.001378`
8. `f00_full_target_ralph20_twostage`: `-0.000443`

Interpretation:
- Orth-target-main family is the true full-data winner on tournament-aligned BMC.
- `f00` (non-orth `target_ralph_20`) does not transfer to tournament-target BMC in the recent window.

## V4 Frontier Re-evaluation (Canonical)

Implementation correction:
- `agents/code/analysis/ensemble_frontier.py` was updated to merge runs on `(id, era)` and rank candidates using canonical in-frame metrics, avoiding cross-target merge loss and non-comparable ranking bias.

Artifacts:
- `numerai/agents/experiments/arrowstreet_integration/v4_full_frontier_canonical_summary.csv`
- `numerai/agents/experiments/arrowstreet_integration/v4_full_frontier_canonical_summary.md`

Top canonical blend:
- `e03_penalized_low_benchmark_corr`
  - `bmc_last200_mean=0.003336`
  - `corr_mean=0.010480`
  - vs `d02`: `+0.001928` BMC-last200, but corr delta `-0.002837`.

Decision:
- Keep `e03` as max-uniqueness / max-BMC profile.
- Continue to search for corr-balanced blends that preserve more of `d02` corr while keeping large BMC lift.

## V5 Corr-Balanced Blend Search (Executed 2026-02-19)

Artifacts:
- Blend sweep summary: `numerai/agents/experiments/arrowstreet_integration/v5_pareto_blend_summary.csv`
- Candidate diagnostics: `numerai/agents/experiments/arrowstreet_integration/v5_candidate_diagnostics.csv`
- Candidate prediction files:
1. `numerai/agents/experiments/arrowstreet_integration/predictions/g00_70f10_30d02.parquet`
2. `numerai/agents/experiments/arrowstreet_integration/predictions/g11_60e03_40d02.parquet`
3. `numerai/agents/experiments/arrowstreet_integration/predictions/g12_50e03_50d02.parquet`
4. `numerai/agents/experiments/arrowstreet_integration/predictions/g13_40e03_60d02.parquet`
5. `numerai/agents/experiments/arrowstreet_integration/predictions/g14_30e03_70d02.parquet`

Key trade-off findings (all vs `d02`):
- `e03` (max BMC): `delta_bmc200=+0.001928`, `delta_corr=-0.002837`
- `g11_60e03_40d02`: `delta_bmc200=+0.001527`, `delta_corr=-0.001150`
- `g12_50e03_50d02`: `delta_bmc200=+0.001348`, `delta_corr=-0.000767`
- `g13_40e03_60d02`: `delta_bmc200=+0.001117`, `delta_corr=-0.000442`
- `g14_30e03_70d02`: `delta_bmc200=+0.000864`, `delta_corr=-0.000176`

Current promotion recommendation:
- Balanced candidate: `g11_60e03_40d02` (large BMC lift with materially smaller corr concession than pure `e03`).
- Aggressive candidate: `e03_penalized_low_benchmark_corr` (maximum BMC lift, lower corr profile).
- Conservative fallback remains `d02_confirm_ranked128_twostage_w050_full`.

Next research step:
- Add explicit corr-floor constrained optimizer (targeting corr delta >= `-0.0010`) for blend search and run a confirmatory held-out era split for `g11/g12/g13`.

## V5 Window Stability Check (Executed 2026-02-19)

Artifacts:
- `numerai/agents/experiments/arrowstreet_integration/v5_window_diagnostics.csv`
- `numerai/agents/experiments/arrowstreet_integration/v5_window_diagnostics.md`

Key window metrics (`last200`):
- `d02`: `bmc_mean=0.001408`, `corr_mean=0.010057`
- `e03`: `bmc_mean=0.003336`, `corr_mean=0.008000`
- `g11_60e03_40d02`: `bmc_mean=0.002935`, `corr_mean=0.009327`
- `g12_50e03_50d02`: `bmc_mean=0.002756`, `corr_mean=0.009622`
- `g13_40e03_60d02`: `bmc_mean=0.002524`, `corr_mean=0.009850`
- `g14_30e03_70d02`: `bmc_mean=0.002271`, `corr_mean=0.010025`

Selection update:
- If corr floor is set to `delta_corr_vs_d02 >= -0.0010`, prefer `g12_50e03_50d02`.
- If prioritizing absolute BMC lift with moderate corr concession, keep `g11_60e03_40d02`.
- Keep `d02` as fallback control and `e03` as aggressive/high-uniqueness profile.

## V6 Corr-Floor Frontier + Holdout (Executed 2026-02-19)

Implementation updates:
- `numerai/agents/code/analysis/ensemble_frontier.py`
  - Added explicit corr-floor constrained blend (`e05_corr_floor_constrained_top5`).
  - Added era-block holdout diagnostics (`train_*`, `holdout_*` metrics).
  - Added automatic promotion table output (`primary_balanced`, `secondary_high_bmc`, `fallback_stable`).
- Added targeted selector:
  - `numerai/agents/code/analysis/blend_holdout_selection.py`
  - Purpose: evaluate precomputed `d02/e03/g11/g12/g13/g14` candidate set on explicit holdout eras and emit promotion slots.

Artifacts:
- Frontier holdout summary:
  - `numerai/agents/experiments/arrowstreet_integration/v6_frontier_holdout_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v6_frontier_holdout_summary.md`
- Frontier promotion table:
  - `numerai/agents/experiments/arrowstreet_integration/v6_frontier_holdout_promotion_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v6_frontier_holdout_promotion_table.md`
- Targeted holdout summary (`d02/e03/g11/g12/g13/g14`):
  - `numerai/agents/experiments/arrowstreet_integration/v6_holdout_selection_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v6_holdout_selection_summary.md`
- Targeted promotion table:
  - `numerai/agents/experiments/arrowstreet_integration/v6_holdout_selection_promotion_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v6_holdout_selection_promotion_table.md`

Key findings:
- New frontier blends from `c/d` family did not beat `d02` on holdout BMC:
  - `d02` holdout: `bmc_last200_mean=0.001238`, `corr_mean=0.010038`
  - `e05_corr_floor_constrained_top5` holdout: `bmc_last200_mean=0.001150`, `corr_mean=0.009872`
  - Result: corr floor was satisfied, but BMC fell below baseline.
- In the targeted `d02/e03/g11/g12/g13/g14` holdout check:
  - `g11_60e03_40d02`: `holdout_bmc_last200_mean=0.002809`, `holdout_corr_mean=0.009292`
    - vs `d02`: `delta_bmc200=+0.001572`, `delta_corr=-0.000746`
  - `g12_50e03_50d02`: `holdout_bmc_last200_mean=0.002617`, `holdout_corr_mean=0.009588`
    - vs `d02`: `delta_bmc200=+0.001380`, `delta_corr=-0.000450`
  - `g13_40e03_60d02`: `delta_bmc200=+0.001141`, `delta_corr=-0.000216`
  - `g14_30e03_70d02`: `delta_bmc200=+0.000884`, `delta_corr=-0.000034`
  - `e03` underperformed `d02` on this specific holdout split (`delta_bmc200=-0.000107`).

Updated promotion recommendation (from holdout promotion table):
1. `primary_balanced`: `g11_60e03_40d02`
2. `secondary_high_bmc`: `g12_50e03_50d02`
3. `fallback_stable`: `d02_confirm_ranked128_twostage_w050_full`

Interpretation:
- Corr/BMC trade-off is best navigated with blended candidates; pure `e03` is less reliable on strict final-era holdout.
- `g11` currently dominates the efficient frontier under corr floor `-0.0010` on holdout.

Next research step:
- Run multi-window holdout robustness (e.g., 3 staggered trailing era blocks) for `g11/g12/g13/g14/d02` and promote only if slot ordering is stable across windows.

## V7 Multi-Window Holdout Robustness (Executed 2026-02-19)

Method:
- Evaluated `d02/g11/g12/g13/g14` across three trailing holdout windows of 192 eras each:
1. `w0`: eras `1010-1201`
2. `w1`: eras `0818-1009`
3. `w2`: eras `0626-0817`
- Baseline for deltas: `d02_confirm_ranked128_twostage_w050_full`
- Guardrails:
1. primary corr floor: `min_delta_corr >= -0.0010`
2. hard corr floor: `min_delta_corr >= -0.0015`

Artifacts:
- Window-level summary:
  - `numerai/agents/experiments/arrowstreet_integration/v7_multiwindow_robustness_window_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v7_multiwindow_robustness_window_summary.md`
- Aggregate robustness summary:
  - `numerai/agents/experiments/arrowstreet_integration/v7_multiwindow_robustness_aggregate_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v7_multiwindow_robustness_aggregate_summary.md`
- Promotion table:
  - `numerai/agents/experiments/arrowstreet_integration/v7_multiwindow_robustness_promotion_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v7_multiwindow_robustness_promotion_table.md`

Aggregate findings (vs `d02`):
- `g11_60e03_40d02`: `mean_delta_bmc200=+0.002085`, `mean_delta_corr=-0.001320`, `min_delta_corr=-0.003196`, positive BMC windows `3/3`
- `g12_50e03_50d02`: `mean_delta_bmc200=+0.001792`, `mean_delta_corr=-0.000922`, `min_delta_corr=-0.002540`, positive BMC windows `3/3`
- `g13_40e03_60d02`: `mean_delta_bmc200=+0.001471`, `mean_delta_corr=-0.000571`, `min_delta_corr=-0.001939`, positive BMC windows `3/3`
- `g14_30e03_70d02`: `mean_delta_bmc200=+0.001135`, `mean_delta_corr=-0.000275`, `min_delta_corr=-0.001378`, positive BMC windows `3/3`

Promotion outcome (strict rules applied):
1. `secondary_high_bmc`: `g14_30e03_70d02`
2. `fallback_stable`: `d02_confirm_ranked128_twostage_w050_full`
- No `primary_balanced` slot was filled because no candidate met the strict worst-window corr floor (`-0.0010`).

Interpretation:
- Every `g*` blend is robustly positive on BMC across windows, but the higher-BMC blends (`g11/g12/g13`) show episodic corr drawdowns beyond strict guardrails.
- Under aggressive corr risk control, `g14` is the only blend that passes hard worst-window corr limits while maintaining consistent BMC lift.

Updated recommendation:
1. Keep `d02` as stable base profile.
2. Use `g14` as the current robust deployment candidate when strict corr guardrails are enforced.
3. Keep `g11` and `g12` as higher-upside alternates for controlled live testing where temporary corr concessions are acceptable.

Next research step:
- Run a constrained multi-window optimizer to solve for weights maximizing mean `bmc_last200` subject to explicit worst-window corr and corr-volatility constraints, then compare against fixed `g14`.

## V8 Constrained Multi-Window Optimizer (Executed 2026-02-19)

Implementation update:
- Added optimizer utility:
  - `numerai/agents/code/analysis/multiwindow_blend_optimizer.py`
- It performs a constrained grid search over two-run blends (`baseline` + `specialist`) and enforces:
1. worst-window corr floor
2. corr-delta volatility cap
3. positive worst-window BMC delta

Lineage correction note:
- The run name `e03_penalized_low_benchmark_corr` was reused by a later frontier pass, so it no longer represented the same specialist lineage used when `g11/g12/g13/g14` were first generated.
- Final V8 optimization was therefore run using a stable specialist head:
  - `f10_full_target_main_orth_beta075` blended with `d02_confirm_ranked128_twostage_w050_full`.

Artifacts (corrected run):
- Grid summary:
  - `numerai/agents/experiments/arrowstreet_integration/v8_multiwindow_optimizer_f10d02_grid_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v8_multiwindow_optimizer_f10d02_grid_summary.md`
- Selection table:
  - `numerai/agents/experiments/arrowstreet_integration/v8_multiwindow_optimizer_f10d02_selection_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v8_multiwindow_optimizer_f10d02_selection_table.md`
- Comparison table:
  - `numerai/agents/experiments/arrowstreet_integration/v8_multiwindow_optimizer_f10d02_comparison_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v8_multiwindow_optimizer_f10d02_comparison_table.md`
- Selected prediction artifact:
  - `numerai/agents/experiments/arrowstreet_integration/predictions/h20_multiwindow_opt_primary_balanced_f10_full_target_main_orth_beta075_w008.parquet`

Key V8 optimizer outputs:
- Best unconstrained (`100% f10`):
  - `mean_delta_bmc200=+0.003354`
  - `mean_delta_corr=-0.009191`
  - `min_delta_corr=-0.013560` (too much corr drawdown)
- `primary_balanced` (strict floor `-0.0010`, corr-vol cap `0.0010`):
  - weights: `8% f10 / 92% d02`
  - `mean_delta_bmc200=+0.000868`
  - `mean_delta_corr=-0.000390`
  - `min_delta_corr=-0.000992`
  - `std_delta_corr=0.000441`
- `secondary_high_bmc` (hard floor `-0.0015`, corr-vol cap `0.0010`):
  - weights: `10% f10 / 90% d02`
  - `mean_delta_bmc200=+0.001064`
  - `mean_delta_corr=-0.000577`
  - `min_delta_corr=-0.001347`
  - `std_delta_corr=0.000562`

Cross-check vs fixed V7 candidate:
- `g14_30e03_70d02` (same windows):
  - `mean_delta_bmc200=+0.001135`
  - `mean_delta_corr=-0.000275`
  - `min_delta_corr=-0.001378`
  - `std_delta_corr=0.000820`

Interpretation:
- V8 produces a strict corr-floor candidate (`8% f10 / 92% d02`) that did not exist in V7.
- Under hard corr guardrails, `g14` still provides slightly higher mean BMC lift than the optimizer's `10% f10` point, while the optimizer point is smoother on corr volatility.

Updated recommendation:
1. `primary_balanced` (strict corr control): `h20_multiwindow_opt_primary_balanced_f10_full_target_main_orth_beta075_w008`
2. `secondary_high_bmc` (higher upside): `g14_30e03_70d02`
3. `fallback_stable`: `d02_confirm_ranked128_twostage_w050_full`

Next research step:
- Freeze lineage IDs for specialist heads before blend optimization (avoid run-name reuse drift), then rerun V8 with a multi-specialist simplex (`f10`, `f01`, `d02`) under the same multi-window corr constraints.

## V8 Holdout Cross-Check for New H20 Candidate (Executed 2026-02-19)

Purpose:
- Validate the new V8 strict candidate (`h20 ... w008`) against the current robust blend (`g14`) using the same holdout-selector framework used in V6.

Artifacts:
- Summary:
  - `numerai/agents/experiments/arrowstreet_integration/v8_holdout_selection_h20_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v8_holdout_selection_h20_summary.md`
- Promotion table:
  - `numerai/agents/experiments/arrowstreet_integration/v8_holdout_selection_h20_promotion_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v8_holdout_selection_h20_promotion_table.md`

Holdout results:
- `g14_30e03_70d02`:
  - `holdout_bmc_last200_mean=0.002121`
  - `holdout_corr_mean=0.010005`
  - vs `d02`: `delta_bmc200=+0.000884`, `delta_corr=-0.000034`
- `h20_multiwindow_opt_primary_balanced_f10_full_target_main_orth_beta075_w008`:
  - `holdout_bmc_last200_mean=0.001882`
  - `holdout_corr_mean=0.009807`
  - vs `d02`: `delta_bmc200=+0.000644`, `delta_corr=-0.000231`

Promotion table outcome:
1. `primary_balanced`: `g14_30e03_70d02`
2. `secondary_high_bmc`: `h20_multiwindow_opt_primary_balanced_f10_full_target_main_orth_beta075_w008`
3. `fallback_stable`: `d02_confirm_ranked128_twostage_w050_full`

Interpretation:
- The V8 optimizer produced a valid strict-floor candidate, but in the final-era holdout selector `g14` remains the stronger corr/BMC trade-off.
- `h20` remains useful as an alternate smoother profile from a distinct (`f10` + `d02`) lineage.

Recommendation update (superseding prior V8 ranking):
1. `primary_balanced`: `g14_30e03_70d02`
2. `secondary_high_bmc` / alternate profile: `h20_multiwindow_opt_primary_balanced_f10_full_target_main_orth_beta075_w008`
3. `fallback_stable`: `d02_confirm_ranked128_twostage_w050_full`

## V9 Multi-Specialist Simplex (`f10`, `f01`, `d02`) (Executed 2026-02-19)

Implementation update:
- Added:
  - `numerai/agents/code/analysis/multiwindow_simplex_optimizer.py`
- Purpose:
  - Optimize blend weights over a simplex using:
1. baseline `d02_confirm_ranked128_twostage_w050_full`
2. specialists `f10_full_target_main_orth_beta075`, `f01_full_target_main_orth_twostage`
  - Constraints:
1. worst-window corr delta floor
2. corr-delta volatility cap
3. positive worst-window BMC delta

Run setup:
- Windows: three trailing windows (`192` eras each)
- Search:
1. coarse simplex step `0.05` (`231` points)
2. local refinement around top seeds (`0.02` step, radius `0.08`)
- Total evaluated points: `302`

Artifacts:
- Simplex grid:
  - `numerai/agents/experiments/arrowstreet_integration/v9_multiwindow_simplex_f10f01d02_grid_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v9_multiwindow_simplex_f10f01d02_grid_summary.md`
- Selection:
  - `numerai/agents/experiments/arrowstreet_integration/v9_multiwindow_simplex_f10f01d02_selection_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v9_multiwindow_simplex_f10f01d02_selection_table.md`
- Comparison:
  - `numerai/agents/experiments/arrowstreet_integration/v9_multiwindow_simplex_f10f01d02_comparison_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v9_multiwindow_simplex_f10f01d02_comparison_table.md`
- Saved V9 blend predictions:
  - `numerai/agents/experiments/arrowstreet_integration/predictions/h30_multiwindow_simplex_primary_balanced_d02094_f10000_f01006.parquet`
  - `numerai/agents/experiments/arrowstreet_integration/predictions/h30_multiwindow_simplex_secondary_high_bmc_d02092_f10000_f01008.parquet`

Key V9 simplex outputs:
- Best unconstrained:
  - weights: `4% d02 / 96% f01 / 0% f10`
  - `mean_delta_bmc200=+0.003423`
  - `mean_delta_corr=-0.010129` (too much corr drawdown)
- `primary_balanced`:
  - weights: `94% d02 / 6% f01 / 0% f10`
  - `mean_delta_bmc200=+0.000874`
  - `mean_delta_corr=-0.000387`
  - `min_delta_corr=-0.000980`
  - `std_delta_corr=0.000434`
- `secondary_high_bmc`:
  - weights: `92% d02 / 8% f01 / 0% f10`
  - `mean_delta_bmc200=+0.001140`
  - `mean_delta_corr=-0.000628`
  - `min_delta_corr=-0.001445`
  - `std_delta_corr=0.000596`

Observation:
- Under these constraints and windows, optimizer consistently zero-weighted `f10`; incremental value came from small `f01` additions on top of `d02`.

## V9 Holdout Cross-Check (`h30` vs `g14/h20/d02`) (Executed 2026-02-19)

Artifacts:
- Summary:
  - `numerai/agents/experiments/arrowstreet_integration/v9_holdout_selection_h30_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v9_holdout_selection_h30_summary.md`
- Promotion:
  - `numerai/agents/experiments/arrowstreet_integration/v9_holdout_selection_h30_promotion_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v9_holdout_selection_h30_promotion_table.md`

Holdout ranking (`bmc_last200_mean`):
1. `g14_30e03_70d02`: `0.002121`
2. `h30_multiwindow_simplex_secondary_high_bmc_d02092_f10000_f01008`: `0.002077`
3. `h30_multiwindow_simplex_primary_balanced_d02094_f10000_f01006`: `0.001886`
4. `h20_multiwindow_opt_primary_balanced_f10_full_target_main_orth_beta075_w008`: `0.001882`
5. `d02_confirm_ranked128_twostage_w050_full`: `0.001238`

Promotion table outcome:
1. `primary_balanced`: `g14_30e03_70d02`
2. `secondary_high_bmc`: `h30_multiwindow_simplex_secondary_high_bmc_d02092_f10000_f01008`
3. `fallback_stable`: `d02_confirm_ranked128_twostage_w050_full`

Interpretation:
- Multi-specialist simplex improved the alternate slot (`h30 secondary`) versus `h20`, but did not displace `g14` for the primary balanced slot on final holdout.

Recommendation update:
1. `primary_balanced`: `g14_30e03_70d02`
2. `secondary_high_bmc`: `h30_multiwindow_simplex_secondary_high_bmc_d02092_f10000_f01008`
3. `fallback_stable`: `d02_confirm_ranked128_twostage_w050_full`

Next research step:
- Run a constrained 4-head simplex (`d02`, `f01`, `f10`, `g14`) with turnover/churn diagnostics to test whether `g14` can be stabilized while preserving its holdout edge.

## V10 4-Head Constrained Simplex + Lineage Manifest (Executed 2026-02-19)

Implementation updates:
- Extended:
  - `numerai/agents/code/analysis/multiwindow_simplex_optimizer.py`
- Added new CLI interfaces:
1. `--lineage-manifest-path`
2. `--window-spec-path`
3. `--max-coarse-points`
4. `--max-refine-points`
- Added lineage manifest writing with immutable input hashes and run metadata.
- Added deterministic high-dimensional simplex point capping for tractable 3+ specialist searches (while preserving simplex corner points and seed neighborhoods).

V10 run command:
- Baseline: `d02_confirm_ranked128_twostage_w050_full`
- Specialists: `g14_30e03_70d02`, `f01_full_target_main_orth_twostage`, `f10_full_target_main_orth_beta075`
- Constraints:
1. strict corr floor `-0.0010`
2. hard corr floor `-0.0015`
3. corr std cap `0.0010`
- Output prefix: `v10_multiwindow_simplex_g14f01f10d02`

V10 artifacts:
- Grid summary:
  - `numerai/agents/experiments/arrowstreet_integration/v10_multiwindow_simplex_g14f01f10d02_grid_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v10_multiwindow_simplex_g14f01f10d02_grid_summary.md`
- Selection:
  - `numerai/agents/experiments/arrowstreet_integration/v10_multiwindow_simplex_g14f01f10d02_selection_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v10_multiwindow_simplex_g14f01f10d02_selection_table.md`
- Comparison:
  - `numerai/agents/experiments/arrowstreet_integration/v10_multiwindow_simplex_g14f01f10d02_comparison_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v10_multiwindow_simplex_g14f01f10d02_comparison_table.md`
- New blend predictions (`h40_*`):
  - `numerai/agents/experiments/arrowstreet_integration/predictions/h40_multiwindow_simplex_primary_balanced_d02075_g14025_f01000_f10000.parquet`
  - `numerai/agents/experiments/arrowstreet_integration/predictions/h40_multiwindow_simplex_secondary_high_bmc_d02066_g14024_f01006_f10004.parquet`
- Lineage manifest:
  - `numerai/agents/experiments/arrowstreet_integration/blend_lineage_manifest.csv`

V10 optimizer result snapshot:
- Evaluated points:
1. coarse generated: `1771`
2. coarse evaluated: `120`
3. refine evaluated: `180`
4. total evaluated: `298`
- Selected rows:
1. `primary_balanced`: `75% d02 + 25% g14 + 0% f01 + 0% f10`
   - `mean_delta_bmc200=+0.001019`
   - `mean_delta_corr=-0.000178`
   - `min_delta_corr=-0.000990`
   - `std_delta_corr=0.000619`
2. `secondary_high_bmc`: `66% d02 + 24% g14 + 6% f01 + 4% f10`
   - `mean_delta_bmc200=+0.001164`
   - `mean_delta_corr=-0.000405`
   - `min_delta_corr=-0.001478`
   - `std_delta_corr=0.000794`

## V10 Holdout Slot Selection (Executed 2026-02-19)

Artifacts:
- Summary:
  - `numerai/agents/experiments/arrowstreet_integration/v10_holdout_selection_h40_summary.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v10_holdout_selection_h40_summary.md`
- Promotion:
  - `numerai/agents/experiments/arrowstreet_integration/v10_holdout_selection_h40_promotion_table.csv`
  - `numerai/agents/experiments/arrowstreet_integration/v10_holdout_selection_h40_promotion_table.md`

Candidate set:
1. `d02_confirm_ranked128_twostage_w050_full`
2. `g14_30e03_70d02`
3. `h30_multiwindow_simplex_secondary_high_bmc_d02092_f10000_f01008`
4. `h40_multiwindow_simplex_primary_balanced_d02075_g14025_f01000_f10000`
5. `h40_multiwindow_simplex_secondary_high_bmc_d02066_g14024_f01006_f10004`

Holdout ranking (`bmc_last200_mean`):
1. `g14_30e03_70d02`: `0.0021213037`
2. `h40_multiwindow_simplex_secondary_high_bmc_d02066_g14024_f01006_f10004`: `0.0021204992`
3. `h30_multiwindow_simplex_secondary_high_bmc_d02092_f10000_f01008`: `0.0020767766`
4. `h40_multiwindow_simplex_primary_balanced_d02075_g14025_f01000_f10000`: `0.0019930039`
5. `d02_confirm_ranked128_twostage_w050_full`: `0.0012377989`

Promotion table outcome:
1. `primary_balanced`: `g14_30e03_70d02`
2. `secondary_high_bmc`: `h40_multiwindow_simplex_secondary_high_bmc_d02066_g14024_f01006_f10004`
3. `fallback_stable`: `d02_confirm_ranked128_twostage_w050_full`

Decision rule outcome:
- `h40_primary` did **not** beat `g14` on holdout `bmc_last200_mean` (`0.001993` vs `0.002121`).
- Final primary remains `g14`.
- Secondary is upgraded from prior `h30` alternate to `h40_secondary`.

## V10 Test Validation

1. `py_compile` passed for updated analysis script.
2. Manifest generation:
- each V10 run appended exactly one row per saved selected blend (`2` rows).
3. SHA stability across unchanged rerun:
- both `h40` run names appear twice in manifest with identical `input_sha256_json` and identical `weights_json`.
4. Constraint integrity:
- `primary_balanced` satisfies strict corr and volatility caps.
- `secondary_high_bmc` satisfies hard corr and volatility caps.
5. Selection consistency:
- `v10_holdout_selection_h40_promotion_table.csv` contains exactly:
`primary_balanced`, `secondary_high_bmc`, `fallback_stable`.

Recommendation update:
1. `primary_balanced`: `g14_30e03_70d02`
2. `secondary_high_bmc`: `h40_multiwindow_simplex_secondary_high_bmc_d02066_g14024_f01006_f10004`
3. `fallback_stable`: `d02_confirm_ranked128_twostage_w050_full`
