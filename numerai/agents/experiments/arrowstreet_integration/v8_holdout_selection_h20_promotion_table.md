| slot | run | holdout_corr_mean | holdout_bmc_last200_mean | delta_corr_vs_baseline | delta_bmc200_vs_baseline | rule |
| --- | --- | --- | --- | --- | --- | --- |
| primary_balanced | g14_30e03_70d02 | 0.010005 | 0.002121 | -0.000034 | 0.000884 | corr_floor_constrained |
| secondary_high_bmc | h20_multiwindow_opt_primary_balanced_f10_full_target_main_orth_beta075_w008 | 0.009807 | 0.001882 | -0.000231 | 0.000644 | max_bmc_with_hard_corr_guardrail |
| fallback_stable | d02_confirm_ranked128_twostage_w050_full | 0.010038 | 0.001238 | 0.000000 | 0.000000 | baseline_control |
