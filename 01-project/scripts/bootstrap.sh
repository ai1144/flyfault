#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || {
  echo 'This lock targets native ARM64 macOS.' >&2; exit 1;
}
if [ -x .tooling/bin/uv ]; then
  uv_exec="$PWD/.tooling/bin/uv"
else
  uv_exec=$(command -v uv) || { echo 'Install uv with brew install uv, then rerun.' >&2; exit 1; }
fi
export UV_PYTHON_INSTALL_DIR="$PWD/.tooling/python"
export UV_CACHE_DIR="$PWD/.tooling/cache"
python_pin=3.12.13
"$uv_exec" python install "$python_pin" --no-bin
"$uv_exec" sync --locked --python "$python_pin"
.venv/bin/python -m flyfault doctor --skip-gl
