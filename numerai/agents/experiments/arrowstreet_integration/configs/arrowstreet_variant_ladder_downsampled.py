from agents.experiments.arrowstreet_integration.configs.variants import (
    VARIANT_LADDER,
    VARIANTS,
)

CONFIG = {
    "signals": {
        "data": {
            "data_version": "v5.2",
            "feature_set": "medium",
            "target_col": "target",
            "era_col": "era",
            "id_col": "id",
            "data_dir": "numerai",
            "dtype_float": "float32",
        },
        "training": {
            "max_train_eras": None,
            "downsample_stride": 4,
            "cv_n_splits": 5,
            "cv_embargo": 13,
            "cv_mode": "expanding",
            "cv_min_train_size": 0,
            "random_state": 42,
        },
        "output": {
            "output_dir": "agents/experiments/arrowstreet_integration/variant_benchmarks_downsampled",
            "artifacts_root": "agents/signals_artifacts",
        },
        "variants": VARIANT_LADDER,
        "variant_definitions": VARIANTS,
        "variant": [{"name": name, **spec} for name, spec in VARIANTS.items()],
        "benchmark": {
            "reference_variant": "v00_lgbm_baseline",
            "benchmark_model": "v52_lgbm_ender20",
            "benchmark_data_path": "numerai/v5.2/validation_benchmark_models.parquet",
            "ranking_metric": "bmc_sharpe",
        },
    },
}
