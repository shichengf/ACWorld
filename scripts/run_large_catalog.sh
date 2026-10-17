#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${ACWORLD_RUNNER_PYTHON:-}" ]]; then
  export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "$ACWORLD_RUNNER_PYTHON" -m cli.large_catalog "$@"
fi

uv_command="${ACWORLD_UV_BIN:-}"
if [[ -z "$uv_command" ]] && command -v uv >/dev/null 2>&1; then
  uv_command="$(command -v uv)"
fi
if [[ -z "$uv_command" ]]; then
  echo "Install uv or set ACWORLD_RUNNER_PYTHON to a Python 3.11+ interpreter." >&2
  exit 2
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.uv-cache}"
"$uv_command" sync --frozen
exec "$uv_command" run --frozen acworld-large-catalog "$@"
