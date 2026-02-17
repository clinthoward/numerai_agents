from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agents.code.modeling.models.arrowstreet_components import (
    BasketBuilder,
    LinkageBuilder,
    StreamingRidge,
    standardize_block,
)


def _sorted_unique_eras(eras: pd.Series) -> list[Any]:
    def _key(value: Any) -> tuple[int, str]:
        try:
            return (0, f"{int(value):012d}")
        except (TypeError, ValueError):
            return (1, str(value))

    return sorted(pd.unique(eras), key=_key)


def _partition_groups(feature_cols: list[str], n_groups: int = 6) -> dict[str, list[str]]:
    if not feature_cols:
        return {"group_0": []}
    n_groups = max(1, int(n_groups))
    buckets: list[list[str]] = [[] for _ in range(n_groups)]
    for i, col in enumerate(feature_cols):
        buckets[i % n_groups].append(col)
    return {f"group_{i}": cols for i, cols in enumerate(buckets) if cols}


class ArrowstreetRegressor:
    """Agents-native Arrowstreet core model wrapper (baskets + linkages + ridge)."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        ridge_alpha: float = 1000.0,
        indirect_max_base_features: int = 64,
        basket_cluster_sizes: list[int] | None = None,
        linkage_k: int = 10,
        linkage_stats: list[str] | None = None,
        use_baskets: bool = True,
        use_linkages: bool = True,
        model_variant: str = "standard",
        stage2_model_type: str = "ridge",
        stage2_lgbm_params: dict[str, Any] | None = None,
        dtype_float: str = "float32",
        random_state: int = 42,
        era_col: str = "era",
        group_sets: dict[str, list[str]] | None = None,
        **_: Any,
    ) -> None:
        self._feature_cols = feature_cols
        self._ridge_alpha = float(ridge_alpha)
        self._indirect_max_base_features = int(indirect_max_base_features)
        self._basket_cluster_sizes = basket_cluster_sizes or [16]
        self._linkage_k = int(linkage_k)
        self._linkage_stats = linkage_stats or ["mean", "std", "min", "max"]
        self._use_baskets = bool(use_baskets)
        self._use_linkages = bool(use_linkages)
        self._model_variant = model_variant
        self._stage2_model_type = stage2_model_type
        self._stage2_lgbm_params = stage2_lgbm_params or {}
        self._dtype_float = dtype_float
        self._random_state = int(random_state)
        self._era_col = era_col
        self._group_sets = group_sets

        self._base_features: list[str] | None = None
        self._indirect_base: list[str] | None = None
        self._basket_feature_names: list[str] | None = None
        self._linkage_feature_names: list[str] | None = None
        self._basket_builder: BasketBuilder | None = None
        self._linkage_builder: LinkageBuilder | None = None
        self._model: Any = None
        self._fitted = False

    def fit(self, X, y, **kwargs):  # noqa: ANN001
        del kwargs
        X_df = self._coerce_X(X)
        y_series = self._coerce_y(y, X_df.index)

        base_features = self._resolve_base_features(X_df)
        indirect_base = base_features[: self._indirect_max_base_features]
        group_sets = self._resolve_group_sets(base_features)
        basket_builder = BasketBuilder(
            group_sets=group_sets,
            cluster_sizes=list(self._basket_cluster_sizes),
            random_state=self._random_state,
            dtype_float=self._dtype_float,
        )
        # X_df already contains era + feature columns for this model path.
        basket_builder.fit(X_df)
        linkage_builder = LinkageBuilder(
            k=self._linkage_k,
            stats=list(self._linkage_stats),
            dtype_float=self._dtype_float,
        )
        basket_feature_names = self._build_basket_feature_names(
            basket_builder, indirect_base
        )
        linkage_feature_names = self._build_linkage_feature_names(indirect_base)

        self._base_features = base_features
        self._indirect_base = indirect_base
        self._basket_builder = basket_builder
        self._linkage_builder = linkage_builder
        self._basket_feature_names = basket_feature_names
        self._linkage_feature_names = linkage_feature_names

        if self._model_variant == "residual_two_stage":
            if not self._use_linkages:
                raise ValueError(
                    "ArrowstreetRegressor model_variant='residual_two_stage' requires use_linkages=True."
                )
            self._model = self._fit_two_stage(X_df, y_series)
        else:
            self._model = self._fit_standard(
                X_df,
                y_series,
                use_linkages=True,
            )
        self._fitted = True
        return self

    def predict(self, X):  # noqa: ANN001
        if not self._fitted:
            raise RuntimeError("ArrowstreetRegressor must be fitted before predict.")
        X_df = self._coerce_X(X)
        self._validate_feature_inputs(X_df)
        preds = pd.Series(index=X_df.index, dtype=self._dtype_float)
        for era in _sorted_unique_eras(X_df[self._era_col]):
            block = X_df[X_df[self._era_col] == era]
            if block.empty:
                continue
            preds.loc[block.index] = self._predict_block(block)
        return preds.to_numpy(dtype=self._dtype_float, copy=False)

    def _predict_block(self, block: pd.DataFrame) -> np.ndarray:
        if self._model_variant == "residual_two_stage":
            model_stage1 = self._model["stage1"]
            model_stage2 = self._model["stage2"]

            X_stage1 = self._build_feature_block(
                block,
                self._base_features or [],
                self._indirect_base or [],
                self._basket_feature_names or [],
                [],
            )
            pred1 = self._predict_model(model_stage1, standardize_block(X_stage1, self._dtype_float))

            X_stage2 = self._build_feature_block(
                block,
                [],
                self._indirect_base or [],
                [],
                self._linkage_feature_names or [],
            )
            pred2 = self._predict_model(model_stage2, standardize_block(X_stage2, self._dtype_float))
            return (pred1 + pred2).astype(np.dtype(self._dtype_float), copy=False)

        X_full = self._build_feature_block(
            block,
            self._base_features or [],
            self._indirect_base or [],
            self._basket_feature_names or [],
            self._linkage_feature_names or [],
        )
        return self._predict_model(self._model, standardize_block(X_full, self._dtype_float))

    def _fit_standard(
        self,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        *,
        use_linkages: bool,
    ):
        assert self._base_features is not None
        assert self._indirect_base is not None
        assert self._basket_feature_names is not None
        assert self._linkage_feature_names is not None

        linkage_names = self._linkage_feature_names if use_linkages else []
        n_features = (
            len(self._base_features)
            + len(self._basket_feature_names)
            + len(linkage_names)
        )
        model = StreamingRidge(
            n_features=n_features,
            alpha=self._ridge_alpha,
            dtype_float=self._dtype_float,
        )
        for era in _sorted_unique_eras(X_df[self._era_col]):
            block = X_df[X_df[self._era_col] == era]
            if block.empty:
                continue
            y_block = y_series.loc[block.index].to_numpy(
                dtype=np.dtype(self._dtype_float), copy=False
            )
            X_block = self._build_feature_block(
                block,
                self._base_features,
                self._indirect_base,
                self._basket_feature_names,
                linkage_names,
            )
            model.update(standardize_block(X_block, self._dtype_float), y_block)
        model.finalize()
        return model

    def _fit_two_stage(self, X_df: pd.DataFrame, y_series: pd.Series) -> dict[str, Any]:
        assert self._base_features is not None
        assert self._indirect_base is not None
        assert self._basket_feature_names is not None
        assert self._linkage_feature_names is not None

        stage1 = self._fit_standard(X_df, y_series, use_linkages=False)
        stage1_preds = pd.Series(index=X_df.index, dtype=self._dtype_float)
        for era in _sorted_unique_eras(X_df[self._era_col]):
            block = X_df[X_df[self._era_col] == era]
            if block.empty:
                continue
            X_block = self._build_feature_block(
                block,
                self._base_features,
                self._indirect_base,
                self._basket_feature_names,
                [],
            )
            stage1_preds.loc[block.index] = self._predict_model(
                stage1, standardize_block(X_block, self._dtype_float)
            )

        residual = y_series - stage1_preds.astype(y_series.dtype)
        stage2 = self._fit_stage2_model(X_df, residual)
        return {"stage1": stage1, "stage2": stage2}

    def _fit_stage2_model(self, X_df: pd.DataFrame, residual: pd.Series):
        assert self._indirect_base is not None
        assert self._linkage_feature_names is not None

        if self._stage2_model_type == "lgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError(
                    "lightgbm is required for ArrowstreetRegressor stage2_model_type='lgbm'. "
                    "Install with `.venv/bin/pip install lightgbm`."
                ) from exc
            X_parts: list[np.ndarray] = []
            y_parts: list[np.ndarray] = []
            for era in _sorted_unique_eras(X_df[self._era_col]):
                block = X_df[X_df[self._era_col] == era]
                if block.empty:
                    continue
                X_block = self._build_feature_block(
                    block,
                    [],
                    self._indirect_base,
                    [],
                    self._linkage_feature_names,
                )
                X_parts.append(standardize_block(X_block, self._dtype_float))
                y_parts.append(
                    residual.loc[block.index].to_numpy(
                        dtype=np.dtype(self._dtype_float), copy=False
                    )
                )
            X_train = np.vstack(X_parts)
            y_train = np.concatenate(y_parts)
            model = LGBMRegressor(**self._stage2_lgbm_params)
            model.fit(X_train, y_train)
            return model

        model = StreamingRidge(
            n_features=len(self._linkage_feature_names),
            alpha=self._ridge_alpha,
            dtype_float=self._dtype_float,
        )
        for era in _sorted_unique_eras(X_df[self._era_col]):
            block = X_df[X_df[self._era_col] == era]
            if block.empty:
                continue
            X_block = self._build_feature_block(
                block,
                [],
                self._indirect_base,
                [],
                self._linkage_feature_names,
            )
            y_block = residual.loc[block.index].to_numpy(
                dtype=np.dtype(self._dtype_float), copy=False
            )
            model.update(standardize_block(X_block, self._dtype_float), y_block)
        model.finalize()
        return model

    def _predict_model(self, model, X_block: np.ndarray) -> np.ndarray:  # noqa: ANN001
        X_for_predict: np.ndarray | pd.DataFrame = X_block
        feature_names = getattr(model, "feature_names_in_", None)
        if (
            feature_names is not None
            and X_block.ndim == 2
            and len(feature_names) == X_block.shape[1]
        ):
            X_for_predict = pd.DataFrame(X_block, columns=list(feature_names))
        preds = model.predict(X_for_predict)
        return np.asarray(preds, dtype=np.dtype(self._dtype_float)).ravel()

    def _build_feature_block(
        self,
        df_block: pd.DataFrame,
        base_features: list[str],
        indirect_base: list[str],
        basket_feature_names: list[str],
        linkage_feature_names: list[str],
    ) -> np.ndarray:
        assert self._basket_builder is not None
        assert self._linkage_builder is not None
        dtype = np.dtype(self._dtype_float)
        if df_block.empty:
            width = len(base_features) + len(basket_feature_names) + len(
                linkage_feature_names
            )
            return np.zeros((0, width), dtype=dtype)

        emb_block = self._basket_builder.transform_embeddings(df_block)
        basket_assign = self._basket_builder.assign_baskets(emb_block)

        basket_feats = self._basket_builder.compute_basket_features(
            df=df_block,
            basket_df=basket_assign,
            base_cols=indirect_base,
            era_col=self._era_col,
        ).reindex(columns=basket_feature_names, fill_value=0.0)

        linkage_feats = self._linkage_builder.compute_linkage_features(
            df=df_block,
            emb_df=emb_block,
            base_cols=indirect_base,
            era_col=self._era_col,
        ).reindex(columns=linkage_feature_names, fill_value=0.0)

        base_values = df_block[base_features].to_numpy(dtype=dtype, copy=False)
        basket_values = basket_feats.to_numpy(dtype=dtype, copy=False)
        linkage_values = linkage_feats.to_numpy(dtype=dtype, copy=False)
        return np.column_stack([base_values, basket_values, linkage_values]).astype(
            dtype, copy=False
        )

    def _resolve_base_features(self, X_df: pd.DataFrame) -> list[str]:
        if self._feature_cols:
            missing = [col for col in self._feature_cols if col not in X_df.columns]
            if missing:
                raise ValueError(
                    f"Missing required feature columns for ArrowstreetRegressor: {missing[:5]}"
                    + ("..." if len(missing) > 5 else "")
                )
            return list(self._feature_cols)
        return [
            col
            for col in X_df.columns
            if col not in {self._era_col}
            and not col.startswith("v")
            and col != "id"
        ]

    def _resolve_group_sets(self, base_features: list[str]) -> dict[str, list[str]]:
        if self._group_sets:
            return {k: list(v) for k, v in self._group_sets.items()}
        return _partition_groups(base_features, n_groups=6)

    def _build_basket_feature_names(
        self,
        basket_builder: BasketBuilder,
        indirect_base: list[str],
    ) -> list[str]:
        if not self._use_baskets:
            return []
        basket_cols = [f"basket_{name}" for name in basket_builder.clusterers.keys()]
        return [f"{basket}__{col}" for basket in basket_cols for col in indirect_base]

    def _build_linkage_feature_names(self, indirect_base: list[str]) -> list[str]:
        if not self._use_linkages:
            return []
        return [f"lnk_{stat}_{col}" for stat in self._linkage_stats for col in indirect_base]

    def _validate_feature_inputs(self, X_df: pd.DataFrame) -> None:
        if self._era_col not in X_df.columns:
            raise ValueError(
                f"ArrowstreetRegressor requires era column '{self._era_col}' in X."
            )
        assert self._base_features is not None
        missing = [col for col in self._base_features if col not in X_df.columns]
        if missing:
            raise ValueError(
                f"Missing required feature columns for prediction: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )

    def _coerce_X(self, X) -> pd.DataFrame:  # noqa: ANN001
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ArrowstreetRegressor expects pandas DataFrame inputs.")
        if self._era_col not in X.columns:
            raise ValueError(
                f"ArrowstreetRegressor requires era column '{self._era_col}' in X."
            )
        # Avoid full-frame copies; full-data CV folds can be multiple GB.
        return X

    @staticmethod
    def _coerce_y(y, index: pd.Index) -> pd.Series:  # noqa: ANN001
        if isinstance(y, pd.Series):
            if not y.index.equals(index):
                y = y.reindex(index)
            return y.astype("float32", copy=False)
        values = np.asarray(y, dtype=np.float32).ravel()
        if values.shape[0] != index.shape[0]:
            raise ValueError("y must have the same number of rows as X.")
        return pd.Series(values, index=index)
