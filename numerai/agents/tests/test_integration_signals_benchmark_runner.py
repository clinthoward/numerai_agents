from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from agents.code.signals import benchmark_variants


class TestSignalsBenchmarkRunnerIntegration(unittest.TestCase):
    def _build_mock_dataset(self, root: Path) -> Path:
        version_dir = root / "v5.2"
        version_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(123)
        feature_cols = ["feature_a", "feature_b", "feature_c"]

        def _make_frame(eras: list[str], with_data_type: bool) -> pd.DataFrame:
            n = len(eras) * 16
            era = np.repeat(eras, 16)
            ids = np.arange(n, dtype=np.int64) + (100000 if with_data_type else 0)
            fa = rng.normal(size=n)
            fb = rng.normal(size=n)
            fc = rng.normal(size=n)
            target = 0.4 * fa - 0.2 * fb + 0.1 * fc + rng.normal(scale=0.1, size=n)
            out = pd.DataFrame(
                {
                    "id": ids,
                    "era": era,
                    "feature_a": fa.astype(np.float32),
                    "feature_b": fb.astype(np.float32),
                    "feature_c": fc.astype(np.float32),
                    "target": target.astype(np.float32),
                    "target_alt": (target + rng.normal(scale=0.1, size=n)).astype(np.float32),
                }
            )
            if with_data_type:
                out["data_type"] = "validation"
            return out

        train = _make_frame([f"{i:04d}" for i in range(1, 13)], with_data_type=False)
        validation = _make_frame([f"{i:04d}" for i in range(13, 17)], with_data_type=True)
        validation_benchmark = validation[["id", "era"]].copy()
        validation_benchmark["v52_lgbm_ender20"] = (
            0.6 * validation["target"].to_numpy() + rng.normal(scale=0.05, size=len(validation))
        ).astype(np.float32)

        train.to_parquet(version_dir / "train.parquet", index=False)
        validation.to_parquet(version_dir / "validation.parquet", index=False)
        validation_benchmark.to_parquet(
            version_dir / "validation_benchmark_models.parquet",
            index=False,
        )
        (version_dir / "features.json").write_text(
            json.dumps({"feature_sets": {"medium": feature_cols}}), encoding="utf-8"
        )
        return version_dir

    def test_benchmark_runner_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._build_mock_dataset(tmp_path)
            out_dir = tmp_path / "runner_outputs"
            cfg_path = tmp_path / "runner_config.py"
            cfg_path.write_text(
                "\n".join(
                    [
                        "CONFIG = {",
                        "  'signals': {",
                        "    'data': {",
                        "      'data_version': 'v5.2',",
                        "      'feature_set': 'medium',",
                        "      'target_col': 'target',",
                        "      'era_col': 'era',",
                        "      'id_col': 'id',",
                        f"      'data_dir': r'{tmp_path}',",
                        "    },",
                        "    'training': {",
                        "      'max_train_eras': None,",
                        "      'downsample_stride': 1,",
                        "      'cv_n_splits': 4,",
                        "      'cv_embargo': 0,",
                        "      'cv_mode': 'blocked',",
                        "      'cv_min_train_size': 0,",
                        "      'random_state': 42,",
                        "    },",
                        "    'output': {",
                        f"      'output_dir': r'{out_dir}',",
                        f"      'artifacts_root': r'{tmp_path / 'artifacts'}',",
                        "    },",
                        "    'variants': ['v00_lgbm_baseline', 'v10_arrowstreet_core'],",
                        "    'benchmark': {'reference_variant': 'v00_lgbm_baseline'},",
                        "  }",
                        "}",
                    ]
                ),
                encoding="utf-8",
            )

            argv = [
                "benchmark_variants.py",
                "--config",
                str(cfg_path),
            ]
            with patch("sys.argv", argv):
                benchmark_variants.main()

            self.assertTrue((out_dir / "variant_summary.csv").exists())
            self.assertTrue((out_dir / "variant_deltas.csv").exists())
            self.assertTrue((out_dir / "variant_leaderboard.md").exists())

            summary = pd.read_csv(out_dir / "variant_summary.csv")
            self.assertIn("bmc_mean", summary.columns)
            self.assertIn("bmc_sharpe", summary.columns)
            self.assertTrue(summary["bmc_mean"].notna().any())


if __name__ == "__main__":
    unittest.main()
