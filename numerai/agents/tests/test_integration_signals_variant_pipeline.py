from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from agents.code.signals.arrowstreet_pipeline import SignalsRunSettings, run_variant_training


HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None


class TestSignalsVariantPipelineIntegration(unittest.TestCase):
    def _build_mock_dataset(self, root: Path) -> Path:
        version_dir = root / "v5.2"
        version_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)
        feature_cols = ["feature_a", "feature_b", "feature_c", "feature_d"]

        train_eras = [f"{i:04d}" for i in range(1, 19)]
        val_eras = [f"{i:04d}" for i in range(19, 25)]
        rows_per_era = 24

        def _frame(eras: list[str], include_data_type: bool) -> pd.DataFrame:
            n = len(eras) * rows_per_era
            era = np.repeat(eras, rows_per_era)
            fa = rng.normal(size=n)
            fb = rng.normal(size=n)
            fc = rng.normal(size=n)
            fd = rng.normal(size=n)

            target = 0.45 * fa - 0.25 * fb + 0.10 * fc + rng.normal(scale=0.1, size=n)
            target_alt_a = target + rng.normal(scale=0.08, size=n)
            target_alt_b = -0.1 * target + rng.normal(scale=0.3, size=n)

            out = pd.DataFrame(
                {
                    "era": era,
                    "feature_a": fa.astype(np.float32),
                    "feature_b": fb.astype(np.float32),
                    "feature_c": fc.astype(np.float32),
                    "feature_d": fd.astype(np.float32),
                    "target": target.astype(np.float32),
                    "target_alt_a": target_alt_a.astype(np.float32),
                    "target_alt_b": target_alt_b.astype(np.float32),
                }
            )
            if include_data_type:
                out["data_type"] = "validation"
            return out

        train = _frame(train_eras, include_data_type=False)
        validation = _frame(val_eras, include_data_type=True)

        train.to_parquet(version_dir / "train.parquet", index=False)
        validation.to_parquet(version_dir / "validation.parquet", index=False)

        features_json = {
            "feature_sets": {
                "medium": feature_cols,
            }
        }
        (version_dir / "features.json").write_text(
            json.dumps(features_json), encoding="utf-8"
        )
        return version_dir

    def test_variant_runs_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = self._build_mock_dataset(tmp_path)
            output_dir = tmp_path / "outputs"
            artifacts_root = tmp_path / "artifacts"

            settings = SignalsRunSettings(
                data_version="v5.2",
                feature_set="medium",
                target_col="target",
                era_col="era",
                id_col=None,
                data_dir=tmp_path,
                dtype_float="float32",
                max_train_eras=None,
                downsample_stride=1,
                cv_n_splits=4,
                cv_embargo=0,
                cv_mode="blocked",
                cv_min_train_size=0,
                random_state=42,
                output_dir=output_dir,
                artifacts_root=artifacts_root,
            )

            variants = ["v00_lgbm_baseline", "v10_arrowstreet_core"]
            if HAS_LIGHTGBM:
                variants = [
                    "v00_lgbm_baseline",
                    "v10_arrowstreet_core",
                    "v20_two_stage",
                    "v30_target_ensemble",
                    "v40_calibrated",
                    "v50_neutralized",
                    "v99_production",
                ]

            for variant in variants:
                result = run_variant_training(
                    variant_name=variant,
                    settings=settings,
                    mode="benchmark",
                )
                oof_path = Path(result["paths"]["oof_predictions"])
                self.assertTrue(oof_path.exists())

                val_path = Path(result["paths"]["validation_predictions"])
                self.assertTrue(val_path.exists())

                artifact_dir = Path(result["paths"]["artifacts_dir"])
                self.assertTrue((artifact_dir / "manifest.json").exists())

                metric = result["metrics"]["validation_corr"]
                self.assertIsNotNone(metric)
                self.assertTrue(math.isfinite(float(metric["mean"])))
                self.assertTrue(math.isfinite(float(metric["sharpe"])))

            self.assertTrue((data_dir / "train.parquet").exists())


if __name__ == "__main__":
    unittest.main()
