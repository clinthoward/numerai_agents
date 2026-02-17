from __future__ import annotations

import unittest

from agents.code.signals.variants import resolve_variant


class TestSignalsVariants(unittest.TestCase):
    def test_variant_inheritance(self) -> None:
        v50 = resolve_variant("v50_neutralized")
        self.assertEqual(v50.toggles.base_model_type, "ArrowstreetRegressor")
        self.assertTrue(v50.toggles.use_target_ensemble)
        self.assertTrue(v50.toggles.use_calibration)
        self.assertEqual(v50.neutralization.method, "selective")
        self.assertAlmostEqual(v50.neutralization.proportion, 0.5)

    def test_production_variant_defaults(self) -> None:
        v99 = resolve_variant("v99_production")
        self.assertEqual(v99.toggles.model_variant, "residual_two_stage")
        self.assertEqual(v99.toggles.stage2_model_type, "lgbm")
        self.assertEqual(v99.neutralization.method, "selective")


if __name__ == "__main__":
    unittest.main()
