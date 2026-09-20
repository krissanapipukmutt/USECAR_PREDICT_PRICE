#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${USED_CAR_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
python_bin="${USED_CAR_PYTHON_BIN:-${project_dir}/.venv/bin/python}"
log_dir="${project_dir}/logs"
log_path="${log_dir}/train_${run_id}.log"

if [[ ! "${run_id}" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "RUN_ID must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi

mkdir -p "${log_dir}"
if [[ -e "${log_path}" ]]; then
  echo "Refusing to overwrite existing log: ${log_path}" >&2
  exit 2
fi

export USED_CAR_RUN_ID="${run_id}"
set +e
"${python_bin}" "${project_dir}/train_used_car_ols.py" 2>&1 | tee "${log_path}"
train_status=${PIPESTATUS[0]}
set -e
if grep -Eiq 'PWD=|password[[:space:]]*=|mssql\+pyodbc://' "${log_path}"; then
  echo "Potential credential material detected in log; do not commit or share it." >&2
  exit 4
fi
if [[ ${train_status} -ne 0 ]]; then
  echo "Training failed with status ${train_status}; log retained at ${log_path}." >&2
  exit "${train_status}"
fi

if ! grep -Fqx "[RUN] RUN_ID=${run_id}" "${log_path}"; then
  echo "Log RUN_ID does not match requested RUN_ID." >&2
  exit 3
fi

for prefix in OLS_REGRESSION_RESULT OLS_REGRESSION_COEFFICIENT used_car_models; do
  case "${prefix}" in
    used_car_models) suffix="joblib" ;;
    *) suffix="csv" ;;
  esac
  count=$(find "${project_dir}/output/train" -type f -name "${prefix}_${run_id}.${suffix}" | wc -l | tr -d ' ')
  if [[ "${count}" != "1" ]]; then
    echo "Expected exactly one ${prefix} artifact for RUN_ID=${run_id}; found ${count}." >&2
    exit 3
  fi
done

echo "Training artifacts and log share RUN_ID=${run_id}."
