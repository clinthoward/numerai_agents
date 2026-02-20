from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for ArrowstreetRegressor. "
        "Install with `.venv/bin/pip install scikit-learn`."
    ) from exc


def _as_dtype(dtype_float: str) -> np.dtype:
    return np.dtype(dtype_float)


def standardize_block(values: np.ndarray, dtype_float: str) -> np.ndarray:
    if values.size == 0:
        return values.astype(_as_dtype(dtype_float), copy=False)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    standardized = (values - mean) / std
    return standardized.astype(_as_dtype(dtype_float), copy=False)


class StreamingRidge:
    """Incrementally fit ridge regression from per-era blocks."""

    def __init__(self, n_features: int, alpha: float, dtype_float: str = "float32"):
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.dtype = _as_dtype(dtype_float)
        self._acc_dtype = np.dtype("float64")
        self.xtx = np.zeros((self.n_features, self.n_features), dtype=self._acc_dtype)
        self.xty = np.zeros(self.n_features, dtype=self._acc_dtype)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.fitted: bool = False

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            return
        X_block = np.asarray(X, dtype=self._acc_dtype)
        y_block = np.asarray(y, dtype=self._acc_dtype)
        if X_block.shape[0] != y_block.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        self.xtx += X_block.T @ X_block
        self.xty += X_block.T @ y_block

    def finalize(self) -> None:
        eye = np.eye(self.n_features, dtype=self._acc_dtype)
        coef = np.linalg.solve(self.xtx + self.alpha * eye, self.xty)
        self.coef_ = coef.astype(self.dtype, copy=False)
        self.intercept_ = 0.0
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or self.coef_ is None:
            raise RuntimeError("StreamingRidge must be finalized before prediction.")
        X_block = np.asarray(X, dtype=self.dtype)
        if X_block.ndim != 2 or X_block.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"StreamingRidge expected 2D array with {self.coef_.shape[0]} features, "
                f"got shape {X_block.shape}."
            )
        preds = X_block @ self.coef_ + self.intercept_
        return np.asarray(preds, dtype=self.dtype, copy=False)


@dataclass
class BasketBuilder:
    """Construct basket-style features from grouped feature embeddings."""

    group_sets: Dict[str, List[str]]
    cluster_sizes: List[int] = field(default_factory=lambda: [16])
    random_state: int = 42
    dtype_float: str = "float32"
    embedding_mode: str = "mean"
    pca_components: int | None = None

    scaler: StandardScaler | None = field(init=False, default=None)
    clusterers: Dict[str, KMeans] = field(init=False, default_factory=dict)
    embedding_cols: List[str] = field(init=False, default_factory=list)
    _raw_embedding_cols: List[str] = field(init=False, default_factory=list)
    _pca: PCA | None = field(init=False, default=None)
    _emb_train: pd.DataFrame | None = field(init=False, default=None)

    def fit(self, df_train: pd.DataFrame) -> None:
        raw_emb_df = self._compute_group_means(df_train)
        if raw_emb_df.empty:
            raise ValueError("Failed to build embeddings: no valid group features found.")
        self._raw_embedding_cols = list(raw_emb_df.columns)
        emb_df = self._project_embeddings(raw_emb_df, fit=True)
        dtype = _as_dtype(self.dtype_float)
        emb_values = emb_df.to_numpy(dtype=dtype, copy=False)
        self.scaler = StandardScaler().fit(emb_values)
        emb_scaled = self.scaler.transform(emb_values).astype(dtype, copy=False)
        self.embedding_cols = list(emb_df.columns)
        self._emb_train = pd.DataFrame(
            emb_scaled, index=df_train.index, columns=self.embedding_cols
        )
        self._fit_clusterers()

    def transform_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None or not self.embedding_cols:
            raise RuntimeError("BasketBuilder must be fitted before transform.")
        raw_emb_df = self._compute_group_means(df)
        for col in self._raw_embedding_cols:
            if col not in raw_emb_df.columns:
                raw_emb_df[col] = 0.0
        raw_emb_df = raw_emb_df[self._raw_embedding_cols]
        emb_df = self._project_embeddings(raw_emb_df, fit=False)
        emb_values = emb_df.to_numpy(dtype=_as_dtype(self.dtype_float), copy=False)
        emb_scaled = self.scaler.transform(emb_values).astype(
            _as_dtype(self.dtype_float), copy=False
        )
        return pd.DataFrame(emb_scaled, index=df.index, columns=self.embedding_cols)

    def assign_baskets(self, emb_df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=emb_df.index)
        for name, model in self.clusterers.items():
            out[f"basket_{name}"] = model.predict(emb_df.values)
        return out

    def compute_basket_features(
        self,
        df: pd.DataFrame,
        basket_df: pd.DataFrame,
        base_cols: List[str],
        era_col: str,
    ) -> pd.DataFrame:
        if basket_df.empty or not base_cols:
            return pd.DataFrame(index=df.index)

        n_rows = len(df)
        n_base = len(base_cols)
        basket_cols = list(basket_df.columns)
        if n_rows == 0:
            return pd.DataFrame(index=df.index)

        dtype = _as_dtype(self.dtype_float)
        data = np.zeros((n_rows, n_base * len(basket_cols)), dtype=dtype)

        for _, era_idx in df.groupby(era_col).groups.items():
            idx = list(era_idx)
            positions = df.index.get_indexer(idx)
            base_values = df.loc[idx, base_cols].to_numpy(dtype=dtype, copy=False)
            for basket_pos, basket_col in enumerate(basket_cols):
                labels = basket_df.loc[idx, basket_col].to_numpy()
                block = self._leave_one_out_means(base_values, labels, dtype)
                start = basket_pos * n_base
                data[positions, start : start + n_base] = block

        columns: list[str] = []
        for basket_col in basket_cols:
            for col in base_cols:
                columns.append(f"{basket_col}__{col}")
        return pd.DataFrame(data, index=df.index, columns=columns)

    def _compute_group_means(self, df: pd.DataFrame) -> pd.DataFrame:
        dtype = _as_dtype(self.dtype_float)
        emb_df = pd.DataFrame(index=df.index)
        for group, cols in self.group_sets.items():
            present = [col for col in cols if col in df.columns]
            if not present:
                continue
            emb_df[f"emb_{group}_mean"] = (
                df[present].astype(dtype).mean(axis=1).astype(dtype)
            )
        return emb_df

    def _project_embeddings(self, raw_emb_df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        dtype = _as_dtype(self.dtype_float)
        mode = str(self.embedding_mode).lower()

        if mode == "mean":
            if fit:
                self._pca = None
            return raw_emb_df.astype(dtype, copy=False)

        if mode != "pca":
            raise ValueError(f"Unsupported embedding_mode: {self.embedding_mode}")

        raw_values = raw_emb_df.to_numpy(dtype=dtype, copy=False)
        if fit:
            n_rows, n_cols = raw_values.shape
            max_components = min(n_rows, n_cols)
            if max_components < 1:
                raise ValueError("PCA embedding requires at least one row and one column.")
            if self.pca_components is None:
                n_components = min(8, max_components)
            else:
                n_components = max(1, min(int(self.pca_components), max_components))
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            projected = self._pca.fit_transform(raw_values).astype(dtype, copy=False)
            cols = [f"emb_pca_{i}" for i in range(projected.shape[1])]
            return pd.DataFrame(projected, index=raw_emb_df.index, columns=cols)

        if self._pca is None:
            raise RuntimeError("BasketBuilder PCA must be fitted before transform.")
        projected = self._pca.transform(raw_values).astype(dtype, copy=False)
        cols = [f"emb_pca_{i}" for i in range(projected.shape[1])]
        return pd.DataFrame(projected, index=raw_emb_df.index, columns=cols)

    def _fit_clusterers(self) -> None:
        if self._emb_train is None:
            raise RuntimeError("fit must be called before fitting clusterers.")
        self.clusterers.clear()
        n_rows = self._emb_train.shape[0]
        for k in self.cluster_sizes:
            k = int(k)
            if k <= 1 or k > n_rows:
                continue
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
            )
            model.fit(self._emb_train.values)
            self.clusterers[f"bkt{k}"] = model
        # Release temporary training embeddings once clusterers are fitted.
        self._emb_train = None
        if not self.clusterers:
            raise ValueError("No valid basket clusterers created. Check cluster_sizes.")

    @staticmethod
    def _leave_one_out_means(
        values: np.ndarray, labels: np.ndarray, dtype: np.dtype
    ) -> np.ndarray:
        n_rows, n_cols = values.shape
        out = np.zeros((n_rows, n_cols), dtype=dtype)
        if n_rows == 0:
            return out
        for lbl in np.unique(labels):
            mask = labels == lbl
            count = int(mask.sum())
            if count <= 1:
                continue
            group_values = values[mask]
            group_sum = group_values.sum(axis=0, dtype=dtype)
            out[mask] = (group_sum - group_values) / (count - 1)
        return out


@dataclass
class LinkageBuilder:
    """Construct nearest-neighbor linkage stats per era."""

    k: int = 10
    stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    dtype_float: str = "float32"

    def compute_linkage_features(
        self,
        df: pd.DataFrame,
        emb_df: pd.DataFrame,
        base_cols: List[str],
        era_col: str,
    ) -> pd.DataFrame:
        if emb_df.empty or not base_cols:
            return pd.DataFrame(index=df.index)

        dtype = _as_dtype(self.dtype_float)
        columns = [f"lnk_{stat}_{col}" for stat in self.stats for col in base_cols]
        out = pd.DataFrame(0.0, index=df.index, columns=columns, dtype=dtype)

        for _, era_idx in df.groupby(era_col).groups.items():
            idx = list(era_idx)
            positions = df.index.get_indexer(idx)
            X = emb_df.loc[idx].to_numpy(dtype=dtype, copy=False)
            if X.shape[0] <= 1:
                continue

            n_neighbors = min(int(self.k) + 1, X.shape[0])
            nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
            _, indices = nn.kneighbors(X)
            neighbor_indices = indices[:, 1:]
            if neighbor_indices.shape[1] == 0:
                continue

            era_values = df.loc[idx, base_cols].to_numpy(dtype=dtype, copy=False)
            neighbors = era_values[neighbor_indices]

            start_col = 0
            n_base = len(base_cols)
            for stat in self.stats:
                if stat == "mean":
                    values = neighbors.mean(axis=1)
                elif stat == "std":
                    values = neighbors.std(axis=1)
                elif stat == "min":
                    values = neighbors.min(axis=1)
                elif stat == "max":
                    values = neighbors.max(axis=1)
                else:
                    raise ValueError(f"Unknown linkage stat: {stat}")
                out.iloc[positions, start_col : start_col + n_base] = values.astype(
                    dtype, copy=False
                )
                start_col += n_base

        return out
