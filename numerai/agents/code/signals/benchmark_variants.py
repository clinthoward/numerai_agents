from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from agents.code.metrics import numerai_metrics
from agents.code.signals.arrowstreet_pipeline import (
    SignalsRunSettings,
    resolve_paths,
    run_variant_training,
    settings_from_dict,
)
from agents.code.signals.variants import DEFAULT_VARIANT_LADDER


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "arrowstreet_integration"
    / "configs"
    / "arrowstreet_variant_ladder_downsampled.py"
)


def _load_config(config_path: Path) -> dict:
    if config_path.suffix == ".py":
        module_vars = runpy.run_path(str(config_path))
        if "CONFIG" in module_vars:
            return module_vars["CONFIG"]
        if "config" in module_vars:
            return module_vars["config"]
        raise ValueError(
            f"Config file {config_path} must define CONFIG (or config) dict."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a variant ladder benchmark and summarize variant deltas."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--reference-variant", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["benchmark", "full_train"], default="benchmark")
    return parser.parse_args()


def _with_output_dir(settings: SignalsRunSettings, output_dir: Path | None) -> SignalsRunSettings:
    if output_dir is None:
        return settings
    return SignalsRunSettings(**{**settings.__dict__, "output_dir": output_dir})


def _to_markdown_table(summary: pd.DataFrame) -> str:
    header = (
        "| variant | corr_mean | corr_sharpe | bmc_mean | bmc_sharpe | "
        "bmc_avg_corr_bench |\n|---|---:|---:|---:|---:|---:|"
    )
    rows = []

    def _fmt(value: float) -> str:
        if pd.isna(value):
            return "nan"
        return f"{value:.6f}"

    for idx, row in summary.iterrows():
        rows.append(
            "| "
            + f"{idx} | {_fmt(row.get('mean', np.nan))} | {_fmt(row.get('sharpe', np.nan))} | "
            + f"{_fmt(row.get('bmc_mean', np.nan))} | {_fmt(row.get('bmc_sharpe', np.nan))} | "
            + f"{_fmt(row.get('bmc_avg_corr_with_benchmark', np.nan))} |"
        )
    return "\n".join([header, *rows])


def _resolve_benchmark_settings(
    *,
    settings: SignalsRunSettings,
    signals_cfg: dict,
    benchmark_cfg: dict,
) -> tuple[str, Path | None, str]:
    data_cfg = signals_cfg.get("data", {})
    benchmark_model = str(
        benchmark_cfg.get("benchmark_model")
        or data_cfg.get("benchmark_model")
        or "v52_lgbm_ender20"
    )
    benchmark_path_raw = benchmark_cfg.get("benchmark_data_path") or data_cfg.get(
        "benchmark_data_path"
    )
    ranking_metric = str(benchmark_cfg.get("ranking_metric", "bmc_sharpe"))

    if benchmark_path_raw is not None:
        return benchmark_model, Path(benchmark_path_raw).expanduser(), ranking_metric

    _, validation_path, _, _ = resolve_paths(settings)
    candidate = validation_path.with_name("validation_benchmark_models.parquet")
    if candidate.exists():
        return benchmark_model, candidate, ranking_metric
    return benchmark_model, None, ranking_metric


def _compute_validation_bmc(
    *,
    prediction_path: Path,
    settings: SignalsRunSettings,
    benchmark_model: str,
    benchmark_data_path: Path,
) -> dict[str, float]:
    summaries = numerai_metrics.summarize_prediction_file_with_bmc(
        predictions_path=prediction_path,
        pred_cols=["prediction"],
        target_col=settings.target_col,
        data_version=settings.data_version,
        benchmark_model=benchmark_model,
        benchmark_data_path=benchmark_data_path,
        era_col=settings.era_col,
        id_col=settings.id_col or "id",
    )
    bmc_row = summaries["bmc"].loc["prediction"]
    return {
        "bmc_mean": float(bmc_row["mean"]),
        "bmc_std": float(bmc_row["std"]),
        "bmc_sharpe": float(bmc_row["sharpe"]),
        "bmc_max_drawdown": float(bmc_row["max_drawdown"]),
        "bmc_avg_corr_with_benchmark": float(
            bmc_row.get("avg_corr_with_benchmark", np.nan)
        ),
    }


def _variant_definitions_from_config(signals_cfg: dict) -> dict | None:
    if isinstance(signals_cfg.get("variant_definitions"), dict):
        return signals_cfg["variant_definitions"]

    variant_node = signals_cfg.get("variant")
    if isinstance(variant_node, list):
        out = {}
        for item in variant_node:
            if not isinstance(item, dict) or "name" not in item:
                continue
            name = str(item["name"])
            body = {k: v for k, v in item.items() if k != "name"}
            out[name] = body
        return out or None

    if isinstance(variant_node, dict) and "name" in variant_node:
        name = str(variant_node["name"])
        body = {k: v for k, v in variant_node.items() if k != "name"}
        return {name: body}

    return None


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    signals_cfg = cfg.get("signals", cfg)
    benchmark_cfg = signals_cfg.get("benchmark", {})

    settings = settings_from_dict(signals_cfg)
    settings = _with_output_dir(settings, args.output_dir)
    variant_definitions = _variant_definitions_from_config(signals_cfg)
    benchmark_model, benchmark_data_path, ranking_metric = _resolve_benchmark_settings(
        settings=settings,
        signals_cfg=signals_cfg,
        benchmark_cfg=benchmark_cfg,
    )

    variants = (
        args.variants
        or signals_cfg.get("variants")
        or list(DEFAULT_VARIANT_LADDER)
    )
    reference_variant = (
        args.reference_variant
        or benchmark_cfg.get("reference_variant")
        or (variants[0] if variants else "v00_lgbm_baseline")
    )

    rows: list[dict[str, float | str]] = []
    for variant_name in variants:
        print(f"Running variant: {variant_name}")
        result = run_variant_training(
            variant_name=variant_name,
            settings=settings,
            mode=args.mode,
            variants=variant_definitions,
        )

        metric = result.get("metrics", {}).get("validation_corr")
        if metric:
            row: dict[str, float | str] = {
                "variant": variant_name,
                "mean": float(metric["mean"]),
                "std": float(metric["std"]),
                "sharpe": float(metric["sharpe"]),
                "max_drawdown": float(metric["max_drawdown"]),
                "bmc_mean": float("nan"),
                "bmc_std": float("nan"),
                "bmc_sharpe": float("nan"),
                "bmc_max_drawdown": float("nan"),
                "bmc_avg_corr_with_benchmark": float("nan"),
            }
            prediction_path = result.get("paths", {}).get("validation_predictions")
            if prediction_path and benchmark_data_path is not None:
                try:
                    row.update(
                        _compute_validation_bmc(
                            prediction_path=Path(prediction_path),
                            settings=settings,
                            benchmark_model=benchmark_model,
                            benchmark_data_path=benchmark_data_path,
                        )
                    )
                except Exception as exc:
                    print(
                        "Warning: failed to compute BMC for "
                        f"{variant_name} ({type(exc).__name__}: {exc})"
                    )
            elif prediction_path and benchmark_data_path is None:
                print(
                    "Warning: validation benchmark models parquet not found; "
                    f"skipping BMC for {variant_name}."
                )
            rows.append(
                row
            )

    if not rows:
        print("No validation metrics were produced (likely full_train mode).")
        return

    summary = pd.DataFrame(rows).set_index("variant")
    sort_metric = ranking_metric if ranking_metric in summary.columns else "sharpe"
    sort_cols = [sort_metric]
    for extra_col in ["bmc_mean", "sharpe", "mean"]:
        if extra_col in summary.columns and extra_col not in sort_cols:
            sort_cols.append(extra_col)
    summary = summary.sort_values(
        by=sort_cols,
        ascending=[False] * len(sort_cols),
        na_position="last",
    )

    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "variant_summary.csv"
    summary.to_csv(summary_path)

    if reference_variant not in summary.index:
        reference_variant = summary.index[0]

    reference_row = summary.loc[reference_variant]
    deltas = summary.copy()
    numeric_cols = list(summary.select_dtypes(include="number").columns)
    for col in numeric_cols:
        deltas[f"delta_{col}"] = deltas[col] - float(reference_row[col])

    deltas_path = output_dir / "variant_deltas.csv"
    deltas.to_csv(deltas_path)

    leaderboard_path = output_dir / "variant_leaderboard.md"
    leaderboard_path.write_text(_to_markdown_table(summary), encoding="utf-8")

    print("\n=== VARIANT SUMMARY ===")
    print(summary)
    print(f"\nRanking metric: {sort_metric}")
    print(f"\nReference variant: {reference_variant}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved deltas: {deltas_path}")
    print(f"Saved leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
