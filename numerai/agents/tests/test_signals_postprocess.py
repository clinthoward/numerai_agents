from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd

from agents.code.signals.postprocess import (
    IsotonicEraCalibrator,
    neutralize_predictions,
    to_submission_scores,
)


HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class TestSignalsPostprocess(unittest.TestCase):
    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for calibration test")
    def test_isotonic_calibrator(self) -> None:
        df = pd.DataFrame(
            {
                "era": ["0001"] * 5 + ["0002"] * 5,
                "target": [0.1, 0.2, 0.3, 0.8, 0.9, 0.15, 0.25, 0.35, 0.75, 0.95],
                "prediction": [0.05, 0.2, 0.4, 0.7, 0.9, 0.1, 0.22, 0.33, 0.72, 0.88],
            }
        )
        cal = IsotonicEraCalibrator()
        cal.fit(df, pred_col="prediction", target_col="target", era_col="era")
        applied = cal.apply(df[["era", "prediction"]], pred_col="prediction", era_col="era")
        self.assertEqual(len(applied), len(df))
        self.assertTrue(np.isfinite(applied).all())

    def test_neutralize_predictions_and_submission_scores(self) -> None:
        rng = np.random.default_rng(7)
        n = 100
        df = pd.DataFrame(
            {
                "era": np.repeat(["0001", "0002", "0003", "0004"], n // 4),
                "feature_a": rng.normal(size=n),
                "feature_b": rng.normal(size=n),
                "prediction": rng.normal(size=n),
            }
        )

        neutralized, used = neutralize_predictions(
            df,
            pred_col="prediction",
            feature_cols=["feature_a", "feature_b"],
            method="selective",
            proportion=0.5,
            top_n=1,
            era_col="era",
        )
        self.assertEqual(len(neutralized), n)
        self.assertTrue(np.isfinite(neutralized).all())
        self.assertGreaterEqual(len(used), 1)

        out = df[["era"]].copy()
        out["prediction"] = neutralized
        submission = to_submission_scores(out, pred_col="prediction", era_col="era")
        self.assertTrue(((submission >= 0.0) & (submission <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
