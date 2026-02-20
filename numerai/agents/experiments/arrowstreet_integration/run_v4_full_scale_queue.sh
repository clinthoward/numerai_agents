#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/clinthoward/Documents/numerai_agents"
NUMERAI_ROOT="$REPO_ROOT/numerai"
EXPERIMENT_DIR="$NUMERAI_ROOT/agents/experiments/arrowstreet_integration"
CONFIG_DIR="$EXPERIMENT_DIR/configs"
RESULTS_DIR="$EXPERIMENT_DIR/results"
PYTHON="$REPO_ROOT/.venv/bin/python"
MASTER_LOG="/tmp/arrowstreet_v4_full_scale_queue.log"

mkdir -p "$RESULTS_DIR"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$MASTER_LOG"
}

run_cfg() {
  local run_name="$1"
  local cfg_file="$2"
  local cfg_path="$CONFIG_DIR/$cfg_file"
  local result_path="$RESULTS_DIR/$run_name.json"
  local run_log="/tmp/${run_name}.log"

  if [[ -f "$result_path" ]]; then
    log "SKIP $run_name (result exists)"
    return 0
  fi

  log "START $run_name"
  (
    cd "$NUMERAI_ROOT"
    "$PYTHON" -m agents.code.modeling --config "$cfg_path"
  ) >"$run_log" 2>&1
  log "DONE  $run_name"
}

wait_for_result_or_run() {
  local run_name="$1"
  local cfg_file="$2"
  local result_path="$RESULTS_DIR/$run_name.json"

  if [[ -f "$result_path" ]]; then
    log "FOUND $run_name result"
    return 0
  fi

  if ps aux | grep -F "$cfg_file" | grep -v grep >/dev/null 2>&1; then
    log "WAIT  $run_name (already running)"
    until [[ -f "$result_path" ]]; do
      if ! ps aux | grep -F "$cfg_file" | grep -v grep >/dev/null 2>&1; then
        log "MISS  $run_name (process ended before result); restarting"
        run_cfg "$run_name" "$cfg_file"
        return 0
      fi
      sleep 120
    done
    log "FOUND $run_name result after wait"
    return 0
  fi

  run_cfg "$run_name" "$cfg_file"
}

build_v4_summary() {
  log "BUILD v4_full_scale_summary"
  "$PYTHON" - <<'PY'
import csv
import json
from pathlib import Path

experiment_dir = Path("/Users/clinthoward/Documents/numerai_agents/numerai/agents/experiments/arrowstreet_integration")
results_dir = experiment_dir / "results"

runs = [
    "d02_confirm_ranked128_twostage_w050_full",
    "d01_confirm_ranked128_twostage_w040_full",
    "f00_full_target_ralph20_twostage",
    "f01_full_target_main_orth_twostage",
    "f02_full_target_ralph20_orth_twostage",
    "f10_full_target_main_orth_beta075",
    "f11_full_target_main_orth_beta100",
    "f12_full_target_main_orth_beta125",
]

rows = []
for run in runs:
    path = results_dir / f"{run}.json"
    if not path.exists():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    params = payload["model"]["params"]
    rows.append(
        {
            "run": run,
            "data_path": payload["data"].get("full_data_path"),
            "target": payload["data"].get("target"),
            "stage2_target_mode": params.get("stage2_target_mode", "residual"),
            "stage2_benchmark_beta": params.get("stage2_benchmark_beta", ""),
            "bmc_last200_mean": metrics["bmc_last_200_eras"]["mean"],
            "bmc_mean": metrics["bmc"]["mean"],
            "corr_mean": metrics["corr"]["mean"],
            "bmc_last200_corr_benchmark": metrics["bmc_last_200_eras"]["avg_corr_with_benchmark"],
        }
    )

rows.sort(key=lambda r: r["bmc_last200_mean"], reverse=True)

csv_path = experiment_dir / "v4_full_scale_summary.csv"
md_path = experiment_dir / "v4_full_scale_summary.md"

fieldnames = [
    "run",
    "data_path",
    "target",
    "stage2_target_mode",
    "stage2_benchmark_beta",
    "bmc_last200_mean",
    "bmc_mean",
    "corr_mean",
    "bmc_last200_corr_benchmark",
]

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# V4 Full Scale Summary",
    "",
    "| run | data | target | stage2_mode | beta | bmc_last200 | bmc | corr | bench_corr_last200 |",
    "|---|---|---|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {run} | {data_path} | {target} | {stage2_target_mode} | {stage2_benchmark_beta} | "
        "{bmc_last200_mean:.6f} | {bmc_mean:.6f} | {corr_mean:.6f} | "
        "{bmc_last200_corr_benchmark:.3f} |".format(**row)
    )

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {csv_path}")
print(f"Wrote {md_path}")
PY
}

run_full_frontier() {
  log "BUILD v4_full_frontier"
  (
    cd "$NUMERAI_ROOT"
    "$PYTHON" -m agents.code.analysis.ensemble_frontier \
      --experiment-dir "$EXPERIMENT_DIR" \
      --candidate-runs "d01_confirm_ranked128_twostage_w040_full,d02_confirm_ranked128_twostage_w050_full,f00_full_target_ralph20_twostage,f01_full_target_main_orth_twostage,f02_full_target_ralph20_orth_twostage,f10_full_target_main_orth_beta075,f11_full_target_main_orth_beta100,f12_full_target_main_orth_beta125" \
      --output-prefix "v4_full_frontier" \
      --include-benchmark-baseline
  ) >>"$MASTER_LOG" 2>&1
  log "DONE  v4_full_frontier"
}

log "QUEUE START"

wait_for_result_or_run "f00_full_target_ralph20_twostage" "f00_full_target_ralph20_twostage.py"
run_cfg "f01_full_target_main_orth_twostage" "f01_full_target_main_orth_twostage.py"
run_cfg "f02_full_target_ralph20_orth_twostage" "f02_full_target_ralph20_orth_twostage.py"
run_cfg "f10_full_target_main_orth_beta075" "f10_full_target_main_orth_beta075.py"
run_cfg "f11_full_target_main_orth_beta100" "f11_full_target_main_orth_beta100.py"
run_cfg "f12_full_target_main_orth_beta125" "f12_full_target_main_orth_beta125.py"

build_v4_summary
run_full_frontier

log "QUEUE COMPLETE"
