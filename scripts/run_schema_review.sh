#!/usr/bin/env bash
set -euo pipefail
set +x

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${USED_CAR_PYTHON_BIN:-${project_dir}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi

if [[ -z "${USED_CAR_DB_PASSWORD:-}" ]]; then
  IFS= read -r -s -p "SQL Server password: " USED_CAR_DB_PASSWORD
  echo
  export USED_CAR_DB_PASSWORD
fi

if [[ -z "${USED_CAR_DB_PASSWORD}" ]]; then
  echo "Password must not be empty." >&2
  exit 2
fi

exec "${python_bin}" "${project_dir}/scripts/review_stg_schema.py" "$@"
