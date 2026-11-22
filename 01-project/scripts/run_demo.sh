#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec .venv/bin/python -m flyfault batch --config configs/demo.yaml "$@"
