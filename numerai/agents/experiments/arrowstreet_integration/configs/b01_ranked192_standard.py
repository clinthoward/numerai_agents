CONFIG = {
    "data": {
        "data_version": "v5.2",
        "embargo_eras": 13,
        "era_col": "era",
        "feature_set": "medium",
        "target_col": "target",
        "id_col": "id",
        "full_data_path": "v5.2/downsampled_full.parquet",
        "benchmark_data_path": "v5.2/downsampled_full_benchmark_models.parquet",
    },
    "model": {
        "type": "ArrowstreetRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "ridge_alpha": 1000.0,
            "indirect_max_base_features": 192,
            "indirect_feature_selection": "era_corr_ranked",
            "indirect_feature_ranking_min_eras": 40,
            "basket_cluster_sizes": [16],
            "linkage_k": 10,
            "linkage_stats": ["mean", "std", "min", "max"],
            "model_variant": "standard",
            "stage2_model_type": "ridge",
            "dtype_float": "float32",
            "random_state": 42,
            "era_col": "era",
        },
    },
    "output": {
        "output_dir": "agents/experiments/arrowstreet_integration",
        "results_name": "b01_ranked192_standard",
    },
    "preprocessing": {
        "missing_value": 2.0,
        "nan_missing_all_twos": False,
    },
    "training": {
        "cv": {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
    },
}
