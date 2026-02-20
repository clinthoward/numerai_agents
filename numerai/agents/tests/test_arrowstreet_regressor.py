from __future__ import annotations

import importlib.util
import unittest
import warnings

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

try:
    from agents.code.modeling.models.arrowstreet_regressor import ArrowstreetRegressor
except Exception:  # pragma: no cover
    ArrowstreetRegressor = None  # type: ignore[assignment]


HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
HAS_PANDAS_NUMPY = np is not None and pd is not None
HAS_MODEL = ArrowstreetRegressor is not None


@unittest.skipUnless(
    HAS_SKLEARN and HAS_PANDAS_NUMPY and HAS_MODEL,
    "scikit-learn, numpy, pandas and model imports are required for ArrowstreetRegressor tests.",
)
class TestArrowstreetRegressor(unittest.TestCase):
    def _make_data(
        self,
        n_eras: int = 10,
        rows_per_era: int = 20,
        include_benchmark: bool = False,
    ):
        rng = np.random.default_rng(123)
        eras = np.repeat([f"{i:04d}" for i in range(1, n_eras + 1)], rows_per_era)
        f1 = rng.normal(size=eras.shape[0])
        f2 = rng.normal(size=eras.shape[0])
        f3 = rng.normal(size=eras.shape[0])
        target = 0.3 * f1 - 0.2 * f2 + 0.1 * f3 + rng.normal(scale=0.05, size=eras.shape[0])
        X = pd.DataFrame(
            {
                "era": eras,
                "feature_a": f1.astype(np.float32),
                "feature_b": f2.astype(np.float32),
                "feature_c": f3.astype(np.float32),
            }
        )
        if include_benchmark:
            X["v52_lgbm_ender20"] = (
                0.4 * f1 - 0.1 * f2 + rng.normal(scale=0.1, size=eras.shape[0])
            ).astype(np.float32)
        y = pd.Series(target.astype(np.float32), index=X.index)
        return X, y

    def test_fit_predict_shape(self) -> None:
        X, y = self._make_data()
        model = ArrowstreetRegressor(
            feature_cols=["feature_a", "feature_b", "feature_c"],
            ridge_alpha=10.0,
            basket_cluster_sizes=[4],
            linkage_k=3,
            random_state=7,
        )
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(preds.shape[0], X.shape[0])
        self.assertTrue(np.isfinite(preds).all())

    def test_predict_before_fit_raises(self) -> None:
        X, _ = self._make_data()
        model = ArrowstreetRegressor(feature_cols=["feature_a", "feature_b", "feature_c"])
        with self.assertRaises(RuntimeError):
            model.predict(X)

    def test_missing_era_raises(self) -> None:
        X, y = self._make_data()
        X_no_era = X.drop(columns=["era"])
        model = ArrowstreetRegressor(feature_cols=["feature_a", "feature_b", "feature_c"])
        with self.assertRaises(ValueError):
            model.fit(X_no_era, y)

    def test_unknown_param_raises(self) -> None:
        with self.assertRaises(ValueError):
            ArrowstreetRegressor(
                feature_cols=["feature_a", "feature_b", "feature_c"],
                unknown_param=123,
            )

    def test_deterministic_given_seed(self) -> None:
        X, y = self._make_data()
        kwargs = dict(
            feature_cols=["feature_a", "feature_b", "feature_c"],
            ridge_alpha=50.0,
            basket_cluster_sizes=[4],
            linkage_k=3,
            random_state=11,
        )
        m1 = ArrowstreetRegressor(**kwargs).fit(X, y)
        m2 = ArrowstreetRegressor(**kwargs).fit(X, y)
        p1 = m1.predict(X)
        p2 = m2.predict(X)
        self.assertTrue(np.allclose(p1, p2))

    @unittest.skipUnless(HAS_LIGHTGBM, "lightgbm is required for stage2 lgbm test.")
    def test_two_stage_lgbm_predict_has_no_feature_name_warning(self) -> None:
        X, y = self._make_data()
        model = ArrowstreetRegressor(
            feature_cols=["feature_a", "feature_b", "feature_c"],
            basket_cluster_sizes=[4],
            linkage_k=3,
            random_state=7,
            model_variant="residual_two_stage",
            stage2_model_type="lgbm",
            stage2_lgbm_params={
                "n_estimators": 25,
                "learning_rate": 0.1,
                "num_leaves": 16,
                "verbosity": -1,
                "n_jobs": 1,
                "random_state": 7,
            },
        )
        model.fit(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            preds = model.predict(X)
        self.assertEqual(preds.shape[0], X.shape[0])
        messages = [str(item.message) for item in caught]
        self.assertFalse(
            any("X does not have valid feature names" in msg for msg in messages),
            "stage2 LightGBM predict emitted feature-name warnings",
        )

    def test_stage2_weight_zero_matches_stage1_only(self) -> None:
        X, y = self._make_data()
        common_kwargs = dict(
            feature_cols=["feature_a", "feature_b", "feature_c"],
            basket_cluster_sizes=[4],
            linkage_k=3,
            random_state=7,
            model_variant="residual_two_stage",
            stage2_model_type="ridge",
        )

        weighted_zero = ArrowstreetRegressor(
            **common_kwargs,
            stage2_weight=0.0,
        ).fit(X, y)
        weighted_one = ArrowstreetRegressor(
            **common_kwargs,
            stage2_weight=1.0,
        ).fit(X, y)
        stage1_only = ArrowstreetRegressor(
            feature_cols=["feature_a", "feature_b", "feature_c"],
            basket_cluster_sizes=[4],
            linkage_k=3,
            random_state=7,
            model_variant="standard",
            use_linkages=False,
        ).fit(X, y)

        p_zero = weighted_zero.predict(X)
        p_one = weighted_one.predict(X)
        p_stage1 = stage1_only.predict(X)
        self.assertTrue(np.allclose(p_zero, p_stage1, atol=1e-6))
        self.assertFalse(np.allclose(p_zero, p_one))

    def test_indirect_feature_selection_ranked_count(self) -> None:
        X, y = self._make_data(n_eras=12, rows_per_era=16)
        model = ArrowstreetRegressor(
            feature_cols=["feature_a", "feature_b", "feature_c"],
            basket_cluster_sizes=[4],
            linkage_k=3,
            random_state=7,
            indirect_max_base_features=2,
            indirect_feature_selection="era_corr_ranked",
            indirect_feature_ranking_min_eras=2,
        )
        model.fit(X, y)
        selected = model._indirect_base or []
        self.assertEqual(len(selected), 2)
        self.assertTrue(set(selected).issubset({"feature_a", "feature_b", "feature_c"}))

    def test_embedding_mode_pca_smoke(self) -> None:
        X, y = self._make_data()
        model = ArrowstreetRegressor(
            feature_cols=["feature_a", "feature_b", "feature_c"],
            basket_cluster_sizes=[4],
            linkage_k=3,
            random_state=7,
            embedding_mode="pca",
            embedding_pca_components=2,
        )
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(preds.shape[0], X.shape[0])
        self.assertTrue(np.isfinite(preds).all())

    def test_stage2_target_mode_residual_to_benchmark_requires_column(self) -> None:
        X, y = self._make_data()
        model = ArrowstreetRegressor(
            feature_cols=["feature_a", "feature_b", "feature_c"],
            basket_cluster_sizes=[4],
            linkage_k=3,
            random_state=7,
            model_variant="residual_two_stage",
            stage2_model_type="ridge",
            stage2_target_mode="residual_to_benchmark",
        )
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_stage2_target_mode_residual_to_benchmark_smoke(self) -> None:
        X, y = self._make_data(include_benchmark=True)
        common_kwargs = dict(
            feature_cols=["feature_a", "feature_b", "feature_c"],
            basket_cluster_sizes=[4],
            linkage_k=3,
            random_state=7,
            model_variant="residual_two_stage",
            stage2_model_type="ridge",
        )
        base_model = ArrowstreetRegressor(
            **common_kwargs,
            stage2_target_mode="residual",
        ).fit(X, y)
        ortho_model = ArrowstreetRegressor(
            **common_kwargs,
            stage2_target_mode="residual_to_benchmark",
            stage2_benchmark_col="v52_lgbm_ender20",
            stage2_benchmark_beta=1.0,
        ).fit(X, y)
        p_base = base_model.predict(X)
        p_ortho = ortho_model.predict(X)
        self.assertEqual(p_base.shape[0], X.shape[0])
        self.assertEqual(p_ortho.shape[0], X.shape[0])
        self.assertFalse(np.allclose(p_base, p_ortho))
