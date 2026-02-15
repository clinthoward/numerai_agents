from __future__ import annotations

import importlib.util
import unittest

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    from agents.code.modeling.utils.model_factory import build_model
except Exception:  # pragma: no cover
    build_model = None  # type: ignore[assignment]


HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_PANDAS = pd is not None
HAS_FACTORY = build_model is not None


@unittest.skipUnless(
    HAS_SKLEARN and HAS_PANDAS and HAS_FACTORY,
    "scikit-learn, pandas and model factory imports are required for ArrowstreetRegressor tests.",
)
class TestModelFactoryArrowstreet(unittest.TestCase):
    def test_build_arrowstreet_model(self) -> None:
        model = build_model(
            "ArrowstreetRegressor",
            {
                "ridge_alpha": 10.0,
                "basket_cluster_sizes": [4],
                "linkage_k": 3,
                "random_state": 7,
            },
            {},
            feature_cols=["feature_a", "feature_b"],
        )
        X = pd.DataFrame(
            {
                "era": ["0001", "0001", "0002", "0002"],
                "feature_a": [0.1, 0.2, 0.3, 0.4],
                "feature_b": [1.0, 0.8, 0.5, 0.2],
            }
        )
        y = pd.Series([0.01, 0.03, 0.02, 0.04])
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_unknown_model_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_model("DOES_NOT_EXIST", {}, {})
