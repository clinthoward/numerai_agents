#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/clinthoward/Documents/numerai_agents"
SCRIPT_PATH="$REPO_ROOT/numerai/agents/experiments/arrowstreet_integration/build_arrowstreet_blend_from_component_pickles.py"
ARTIFACT_DIR="$REPO_ROOT/numerai/agents/experiments/arrowstreet_integration/upload_artifacts"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT"

export PYTHONPATH="numerai"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl"

# Component artifacts:
# c00 uses the existing full upload artifact from prior run.
C00_PKL="$ARTIFACT_DIR/arrowstreet_c00_full_upload.pkl"
D02_PKL="$ARTIFACT_DIR/v10_component_d02.pkl"
C01_PKL="$ARTIFACT_DIR/v10_component_c01.pkl"
D00_PKL="$ARTIFACT_DIR/v10_component_d00.pkl"
D01_PKL="$ARTIFACT_DIR/v10_component_d01.pkl"
F01_PKL="$ARTIFACT_DIR/v10_component_f01.pkl"
F10_PKL="$ARTIFACT_DIR/v10_component_f10.pkl"

for path in "$C00_PKL" "$D02_PKL" "$C01_PKL" "$D00_PKL" "$D01_PKL" "$F01_PKL" "$F10_PKL"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing required component pickle: $path" >&2
    exit 1
  fi
done

mkdir -p "$ARTIFACT_DIR/logs"

echo "[start] Building v10_live_primary_g14"
"$PYTHON_BIN" "$SCRIPT_PATH" \
  --blend-name v10_live_primary_g14 \
  --spec c00_scale_ranked128_full:0.18594015095407854 \
  --spec c01_scale_ranked128_twostage_w040_full:0.025911277949977093 \
  --spec d00_confirm_ranked128_twostage_w030_full:0.04094254553997524 \
  --spec d01_confirm_ranked128_twostage_w040_full:0.026434964348122723 \
  --spec d02_confirm_ranked128_twostage_w050_full:0.7207710612078464 \
  --component-pkl c00_scale_ranked128_full:"$C00_PKL" \
  --component-pkl c01_scale_ranked128_twostage_w040_full:"$C01_PKL" \
  --component-pkl d00_confirm_ranked128_twostage_w030_full:"$D00_PKL" \
  --component-pkl d01_confirm_ranked128_twostage_w040_full:"$D01_PKL" \
  --component-pkl d02_confirm_ranked128_twostage_w050_full:"$D02_PKL" \
  2>&1 | tee "$ARTIFACT_DIR/logs/v10_live_primary_g14.log"

echo "[start] Building v10_live_secondary_h40"
"$PYTHON_BIN" "$SCRIPT_PATH" \
  --blend-name v10_live_secondary_h40 \
  --spec c00_scale_ranked128_full:0.044625636228978846 \
  --spec c01_scale_ranked128_twostage_w040_full:0.0062187067079945025 \
  --spec d00_confirm_ranked128_twostage_w030_full:0.009826210929594058 \
  --spec d01_confirm_ranked128_twostage_w040_full:0.006344391443549454 \
  --spec d02_confirm_ranked128_twostage_w050_full:0.8329850546898832 \
  --spec f01_full_target_main_orth_twostage:0.06 \
  --spec f10_full_target_main_orth_beta075:0.04 \
  --component-pkl c00_scale_ranked128_full:"$C00_PKL" \
  --component-pkl c01_scale_ranked128_twostage_w040_full:"$C01_PKL" \
  --component-pkl d00_confirm_ranked128_twostage_w030_full:"$D00_PKL" \
  --component-pkl d01_confirm_ranked128_twostage_w040_full:"$D01_PKL" \
  --component-pkl d02_confirm_ranked128_twostage_w050_full:"$D02_PKL" \
  --component-pkl f01_full_target_main_orth_twostage:"$F01_PKL" \
  --component-pkl f10_full_target_main_orth_beta075:"$F10_PKL" \
  2>&1 | tee "$ARTIFACT_DIR/logs/v10_live_secondary_h40.log"

echo "[complete] Built final V10 live blend pickles."
