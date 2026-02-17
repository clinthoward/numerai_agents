from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal


ModelType = Literal["LGBMRegressor", "ArrowstreetRegressor"]
ModelVariant = Literal["standard", "residual_two_stage"]
Stage2ModelType = Literal["ridge", "lgbm"]
NeutralizationMethod = Literal["none", "full", "selective", "proportional"]


DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 1200,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 32,
    "colsample_bytree": 0.1,
    "subsample": 0.5,
    "min_data_in_leaf": 10000,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

DEFAULT_ARROWSTREET_PARAMS: dict[str, Any] = {
    "ridge_alpha": 1000.0,
    "indirect_max_base_features": 64,
    "basket_cluster_sizes": [16],
    "linkage_k": 10,
    "linkage_stats": ["mean", "std", "min", "max"],
    "dtype_float": "float32",
    "random_state": 42,
}


@dataclass(frozen=True)
class VariantToggles:
    base_model_type: ModelType
    use_baskets: bool
    use_linkages: bool
    model_variant: ModelVariant
    stage2_model_type: Stage2ModelType
    use_target_ensemble: bool
    use_calibration: bool


@dataclass(frozen=True)
class TargetEnsembleConfig:
    enabled: bool
    correlation_threshold: float
    max_models: int
    duplicate_corr_cutoff: float


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool


@dataclass(frozen=True)
class NeutralizationConfig:
    method: NeutralizationMethod
    proportion: float
    top_n: int | float


@dataclass(frozen=True)
class VariantSpec:
    name: str
    parent: str | None
    description: str
    toggles: VariantToggles
    model_params: dict[str, Any]
    target_ensemble: TargetEnsembleConfig
    calibration: CalibrationConfig
    neutralization: NeutralizationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "description": self.description,
            "toggles": {
                "base_model_type": self.toggles.base_model_type,
                "use_baskets": self.toggles.use_baskets,
                "use_linkages": self.toggles.use_linkages,
                "model_variant": self.toggles.model_variant,
                "stage2_model_type": self.toggles.stage2_model_type,
                "use_target_ensemble": self.toggles.use_target_ensemble,
                "use_calibration": self.toggles.use_calibration,
            },
            "model_params": dict(self.model_params),
            "target_ensemble": {
                "enabled": self.target_ensemble.enabled,
                "correlation_threshold": self.target_ensemble.correlation_threshold,
                "max_models": self.target_ensemble.max_models,
                "duplicate_corr_cutoff": self.target_ensemble.duplicate_corr_cutoff,
            },
            "postprocess": {
                "calibration": {"enabled": self.calibration.enabled},
                "neutralization": {
                    "method": self.neutralization.method,
                    "proportion": self.neutralization.proportion,
                    "top_n": self.neutralization.top_n,
                },
            },
        }


DEFAULT_VARIANT_LADDER = [
    "v00_lgbm_baseline",
    "v10_arrowstreet_core",
    "v20_two_stage",
    "v30_target_ensemble",
    "v40_calibrated",
    "v50_neutralized",
    "v99_production",
]


DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "v00_lgbm_baseline": {
        "description": "Plain LGBM baseline on base features.",
        "toggles": {
            "base_model_type": "LGBMRegressor",
            "use_baskets": False,
            "use_linkages": False,
            "model_variant": "standard",
            "stage2_model_type": "ridge",
            "use_target_ensemble": False,
            "use_calibration": False,
        },
        "model_params": deepcopy(DEFAULT_LGBM_PARAMS),
        "target_ensemble": {
            "enabled": False,
            "correlation_threshold": 0.5,
            "max_models": 10,
            "duplicate_corr_cutoff": 0.9999,
        },
        "postprocess": {
            "calibration": {"enabled": False},
            "neutralization": {
                "method": "none",
                "proportion": 0.5,
                "top_n": 0.10,
            },
        },
    },
    "v10_arrowstreet_core": {
        "parent": "v00_lgbm_baseline",
        "description": "Arrowstreet core (single target, baskets + linkages).",
        "toggles": {
            "base_model_type": "ArrowstreetRegressor",
            "use_baskets": True,
            "use_linkages": True,
            "model_variant": "standard",
            "stage2_model_type": "ridge",
            "use_target_ensemble": False,
            "use_calibration": False,
        },
        "model_params": deepcopy(DEFAULT_ARROWSTREET_PARAMS),
    },
    "v20_two_stage": {
        "parent": "v10_arrowstreet_core",
        "description": "Enable residual two-stage Arrowstreet model.",
        "toggles": {
            "model_variant": "residual_two_stage",
            "stage2_model_type": "lgbm",
        },
        "model_params": {
            "stage2_lgbm_params": deepcopy(DEFAULT_LGBM_PARAMS),
        },
    },
    "v30_target_ensemble": {
        "parent": "v20_two_stage",
        "description": "Enable alternative target ensemble.",
        "toggles": {
            "use_target_ensemble": True,
        },
        "target_ensemble": {
            "enabled": True,
        },
    },
    "v40_calibrated": {
        "parent": "v30_target_ensemble",
        "description": "Enable era-ranked isotonic calibration.",
        "toggles": {
            "use_calibration": True,
        },
        "postprocess": {
            "calibration": {"enabled": True},
        },
    },
    "v50_neutralized": {
        "parent": "v40_calibrated",
        "description": "Enable selective feature neutralization.",
        "postprocess": {
            "neutralization": {
                "method": "selective",
                "proportion": 0.5,
                "top_n": 0.10,
            },
        },
    },
    "v99_production": {
        "parent": "v50_neutralized",
        "description": "Production profile for daily Signals scoring.",
    },
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_raw_variant(
    name: str,
    variants: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if name in cache:
        return cache[name]
    if name not in variants:
        raise KeyError(
            f"Unknown variant '{name}'. Available variants: {sorted(variants.keys())}"
        )
    node = deepcopy(variants[name])
    parent = node.get("parent")
    if parent:
        base = _resolve_raw_variant(parent, variants, cache)
        node = _deep_merge(base, node)
    cache[name] = node
    return node


def resolve_variant(
    name: str,
    variants: dict[str, dict[str, Any]] | None = None,
) -> VariantSpec:
    variants = variants or DEFAULT_VARIANTS
    resolved = _resolve_raw_variant(name, variants, cache={})

    toggles = resolved.get("toggles", {})
    target_ensemble = resolved.get("target_ensemble", {})
    postprocess = resolved.get("postprocess", {})
    calibration = postprocess.get("calibration", {})
    neutralization = postprocess.get("neutralization", {})

    return VariantSpec(
        name=name,
        parent=variants.get(name, {}).get("parent"),
        description=str(resolved.get("description", "")),
        toggles=VariantToggles(
            base_model_type=str(toggles.get("base_model_type", "ArrowstreetRegressor")),
            use_baskets=bool(toggles.get("use_baskets", True)),
            use_linkages=bool(toggles.get("use_linkages", True)),
            model_variant=str(toggles.get("model_variant", "standard")),
            stage2_model_type=str(toggles.get("stage2_model_type", "ridge")),
            use_target_ensemble=bool(toggles.get("use_target_ensemble", False)),
            use_calibration=bool(toggles.get("use_calibration", False)),
        ),
        model_params=deepcopy(resolved.get("model_params", {})),
        target_ensemble=TargetEnsembleConfig(
            enabled=bool(target_ensemble.get("enabled", False)),
            correlation_threshold=float(target_ensemble.get("correlation_threshold", 0.5)),
            max_models=int(target_ensemble.get("max_models", 10)),
            duplicate_corr_cutoff=float(target_ensemble.get("duplicate_corr_cutoff", 0.9999)),
        ),
        calibration=CalibrationConfig(enabled=bool(calibration.get("enabled", False))),
        neutralization=NeutralizationConfig(
            method=str(neutralization.get("method", "none")),
            proportion=float(neutralization.get("proportion", 1.0)),
            top_n=neutralization.get("top_n", 0.10),
        ),
    )


def list_variant_names(variants: dict[str, dict[str, Any]] | None = None) -> list[str]:
    variants = variants or DEFAULT_VARIANTS
    ordered = [name for name in DEFAULT_VARIANT_LADDER if name in variants]
    remainder = [name for name in variants.keys() if name not in ordered]
    return [*ordered, *remainder]


def build_variant_specs(
    variant_names: list[str] | None = None,
    variants: dict[str, dict[str, Any]] | None = None,
) -> list[VariantSpec]:
    variants = variants or DEFAULT_VARIANTS
    names = variant_names or list_variant_names(variants)
    return [resolve_variant(name, variants) for name in names]
