#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/clinthoward/Documents/numerai_agents"
SCRIPT_PATH="$REPO_ROOT/numerai/agents/experiments/arrowstreet_integration/build_arrowstreet_upload_pickle.py"
ARTIFACT_DIR="$REPO_ROOT/numerai/agents/experiments/arrowstreet_integration/upload_artifacts"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

export PYTHONPATH="numerai"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/mpl"
MAX_ROWS="${MAX_ROWS:-2000000}"
STAGE2_LGBM_N_JOBS="${STAGE2_LGBM_N_JOBS:-1}"

cd "$REPO_ROOT"

build_one() {
  local blend_name="$1"
  local config_name="$2"
  local output_pkl="$ARTIFACT_DIR/${blend_name}.pkl"
  local output_meta="$ARTIFACT_DIR/${blend_name}.json"
  local log_path="$ARTIFACT_DIR/logs/${blend_name}.log"

  mkdir -p "$ARTIFACT_DIR/logs"

  if [[ -s "$output_pkl" && -s "$output_meta" ]]; then
    echo "[skip] ${blend_name} already exists."
    return 0
  fi

  echo "[start] ${blend_name} <- ${config_name}"
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --blend-name "$blend_name" \
    --spec "${config_name}:1.0" \
    --stage2-lgbm-n-jobs "$STAGE2_LGBM_N_JOBS" \
    --max-rows "$MAX_ROWS" \
    2>&1 | tee "$log_path"
  echo "[done] ${blend_name}"
}

# Component set required to reconstruct:
# - primary: g14_30e03_70d02
# - secondary: h40_multiwindow_simplex_secondary_high_bmc_d02066_g14024_f01006_f10004
build_one "v10_component_d02" "d02_confirm_ranked128_twostage_w050_full"
build_one "v10_component_c00" "c00_scale_ranked128_full"
build_one "v10_component_c01" "c01_scale_ranked128_twostage_w040_full"
build_one "v10_component_d00" "d00_confirm_ranked128_twostage_w030_full"
build_one "v10_component_d01" "d01_confirm_ranked128_twostage_w040_full"
build_one "v10_component_f01" "f01_full_target_main_orth_twostage"
build_one "v10_component_f10" "f10_full_target_main_orth_beta075"

echo "[complete] All requested components built."
