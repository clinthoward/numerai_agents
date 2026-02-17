from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from agents.code.signals.target_ensemble import (
    average_target_predictions,
    select_ensemble_targets,
)


class TestTargetEnsemble(unittest.TestCase):
    def test_select_ensemble_targets(self) -> None:
        rng = np.random.default_rng(123)
        n = 500
        base = rng.normal(size=n)

        df = pd.DataFrame(
            {
                "target": base,
                "target_alt_a": base + rng.normal(scale=0.05, size=n),
                "target_alt_b": rng.normal(size=n),
                "target_alt_dup": base + rng.normal(scale=1e-4, size=n),
            }
        )

        selected, corr = select_ensemble_targets(
            df,
            primary_target="target",
            candidate_targets=["target", "target_alt_a", "target_alt_b", "target_alt_dup"],
            correlation_threshold=0.2,
            max_models=3,
            duplicate_corr_cutoff=0.999,
        )

        self.assertEqual(selected[0], "target")
        self.assertIn("target_alt_a", selected)
        self.assertNotIn("target_alt_b", selected)
        self.assertNotIn("target_alt_dup", selected)
        self.assertGreater(corr["target_alt_a"], 0.2)

    def test_average_target_predictions(self) -> None:
        idx = pd.Index([1, 2, 3])
        preds = {
            "target": pd.Series([0.1, 0.2, 0.3], index=idx),
            "target_alt": pd.Series([0.3, 0.4, 0.5], index=idx),
        }
        avg = average_target_predictions(preds)
        self.assertTrue(np.allclose(avg.values, [0.2, 0.3, 0.4]))


if __name__ == "__main__":
    unittest.main()
