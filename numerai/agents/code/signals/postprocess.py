from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # pragma: no cover
    IsotonicRegression = None  # type: ignore[misc,assignment]


def era_rank(series: pd.Series) -> pd.Series:
    denom = series.count()
    if denom <= 0:
        return pd.Series(np.nan, index=series.index, dtype=np.float32)
    return ((series.rank(method="average") - 0.5) / denom).astype(np.float32)


class IsotonicEraCalibrator:
    """Isotonic calibrator fitted on era-ranked predictions."""

    def __init__(self) -> None:
        self.iso = None

    def fit(
        self,
        df_train: pd.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        era_col: str = "era",
    ) -> None:
        if IsotonicRegression is None:
            raise ImportError(
                "scikit-learn is required for isotonic calibration. "
                "Install with `.venv/bin/pip install scikit-learn`."
            )
        ranked = df_train.groupby(era_col)[pred_col].transform(era_rank)
        ranked = np.clip(ranked.to_numpy(dtype=np.float64), 0.01, 0.99)
        targets = df_train[target_col].to_numpy(dtype=np.float64)

        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.iso.fit(ranked, targets)

    def apply(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str,
        era_col: str = "era",
    ) -> pd.Series:
        if self.iso is None:
            raise RuntimeError("Calibrator must be fit before apply.")
        ranked = df.groupby(era_col)[pred_col].transform(era_rank)
        ranked = np.clip(ranked.to_numpy(dtype=np.float64), 0.01, 0.99)
        calibrated = self.iso.predict(ranked)
        return pd.Series(calibrated, index=df.index, name=pred_col)


def get_feature_exposures(
    df: pd.DataFrame,
    pred_col: str,
    feature_cols: list[str],
) -> pd.Series:
    if not feature_cols:
        return pd.Series(dtype=np.float64)

    pred = df[pred_col].to_numpy(dtype=np.float64)
    feats = df[feature_cols].to_numpy(dtype=np.float64)

    pred_centered = pred - pred.mean()
    feats_centered = feats - feats.mean(axis=0)

    pred_norm = np.linalg.norm(pred_centered)
    feats_norm = np.linalg.norm(feats_centered, axis=0)

    feats_norm = np.where(feats_norm < 1e-6, 1.0, feats_norm)
    if pred_norm < 1e-6:
        return pd.Series(0.0, index=feature_cols)

    corrs = np.dot(feats_centered.T, pred_centered) / (feats_norm * pred_norm)
    return pd.Series(corrs, index=feature_cols)


def get_riskiest_features(exposures: pd.Series, top_n: int | float) -> list[str]:
    if exposures.empty:
        return []
    abs_exp = exposures.abs().sort_values(ascending=False)

    n = top_n
    if isinstance(n, float) and 0 < n < 1.0:
        n = int(len(exposures) * n)

    n = int(n)
    if n <= 0:
        return []
    return abs_exp.head(n).index.tolist()


def neutralize_predictions(
    df: pd.DataFrame,
    *,
    pred_col: str,
    feature_cols: list[str],
    method: str,
    proportion: float,
    top_n: int | float,
    era_col: str = "era",
) -> tuple[pd.Series, list[str]]:
    if method == "none":
        return df[pred_col].astype(np.float32), []

    neutralizers = feature_cols
    if method == "selective":
        exposures = get_feature_exposures(df, pred_col, feature_cols)
        neutralizers = get_riskiest_features(exposures, top_n)
    elif method not in {"full", "proportional"}:
        raise ValueError(
            "neutralization method must be one of: none, full, selective, proportional"
        )

    if method == "full":
        neutralize_prop = 1.0
    else:
        neutralize_prop = float(proportion)

    out = df.copy()
    if not neutralizers:
        return out[pred_col].astype(np.float32), []

    for _, group in df.groupby(era_col):
        if group.empty:
            continue
        X = group[neutralizers].to_numpy(dtype=np.float64)
        X = np.column_stack([np.ones(len(X), dtype=np.float64), X])
        y = group[pred_col].to_numpy(dtype=np.float64)

        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = np.dot(X, beta)
            y_neutral = y - neutralize_prop * y_pred
        except np.linalg.LinAlgError:
            y_neutral = y

        out.loc[group.index, pred_col] = y_neutral

    out[pred_col] = out.groupby(era_col)[pred_col].transform(era_rank)
    return out[pred_col].astype(np.float32), list(neutralizers)


def to_submission_scores(df: pd.DataFrame, *, pred_col: str, era_col: str = "era") -> pd.Series:
    ranked = df.groupby(era_col)[pred_col].transform(era_rank)
    if ranked.isna().all():
        return pd.Series(0.5, index=df.index, dtype=np.float32, name="prediction")

    ranked = ranked.fillna(ranked.mean())
    min_val = float(ranked.min())
    max_val = float(ranked.max())
    if max_val - min_val < 1e-12:
        normalized = pd.Series(0.5, index=ranked.index, dtype=np.float32)
    else:
        normalized = ((ranked - min_val) / (max_val - min_val)).astype(np.float32)
    return pd.Series(normalized, index=df.index, name="prediction")
