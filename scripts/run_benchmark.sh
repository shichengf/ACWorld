#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${ACWORLD_RUNNER_PYTHON:-}" ]]; then
  export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "$ACWORLD_RUNNER_PYTHON" -m cli.benchmark_runner "$@"
fi

uv_command="${ACWORLD_UV_BIN:-}"
if [[ -z "$uv_command" ]] && command -v uv >/dev/null 2>&1; then
  uv_command="$(command -v uv)"
fi

if [[ -z "$uv_command" ]]; then
  bootstrap_root="$repo_root/.acworld-bootstrap"
  bootstrap_python="${PYTHON:-python3}"
  if ! command -v "$bootstrap_python" >/dev/null 2>&1; then
    echo "Python 3 is required to bootstrap the runner." >&2
    exit 2
  fi
  if [[ ! -x "$bootstrap_root/bin/uv" ]]; then
    "$bootstrap_python" -m venv "$bootstrap_root"
    "$bootstrap_root/bin/python" -m pip install --disable-pip-version-check "uv==0.11.16"
  fi
  uv_command="$bootstrap_root/bin/uv"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.uv-cache}"
"$uv_command" sync --frozen
exec "$uv_command" run --frozen acworld-run-benchmark "$@"
