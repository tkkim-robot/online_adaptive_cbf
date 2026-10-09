#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/artifacts/tmp"
export TMPDIR="${TMPDIR:-$repo_dir/artifacts/tmp}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python_path="${OA_CBF_PYTHON:-$repo_dir/.venv/bin/python}"
export PATH="$(dirname -- "$python_path"):$PATH"
exec "$python_path" "$@"
