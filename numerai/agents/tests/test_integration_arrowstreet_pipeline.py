from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

try:
    from agents.code.modeling.utils.pipeline import run_training
except Exception:  # pragma: no cover
    run_training = None  # type: ignore[assignment]


class TestArrowstreetIntegrationPipeline(unittest.TestCase):
    def test_pipeline_metrics(self) -> None:
        if run_training is None:
            self.skipTest("Modeling pipeline dependencies are missing in this environment.")
        repo_root = Path(__file__).resolve().parents[2]
        full_path = repo_root / "v5.2" / "downsampled_full.parquet"
        bench_path = repo_root / "v5.2" / "downsampled_full_benchmark_models.parquet"
        if not full_path.exists() or not bench_path.exists():
            self.skipTest(
                "Downsampled dataset files are missing; skipping Arrowstreet integration test."
            )

        config_path = (
            repo_root
            / "agents"
            / "experiments"
            / "arrowstreet_integration"
            / "configs"
            / "arrowstreet_core_downsampled.py"
        )
        _, results_path = run_training(config_path)
        with open(results_path, "r", encoding="utf-8") as fp:
            results = json.load(fp)

        corr_mean = float(results["metrics"]["corr"]["mean"])
        corr_sharpe = float(results["metrics"]["corr"]["sharpe"])
        bmc_mean = float(results["metrics"]["bmc"]["mean"])

        self.assertTrue(math.isfinite(corr_mean))
        self.assertTrue(math.isfinite(corr_sharpe))
        self.assertTrue(math.isfinite(bmc_mean))
