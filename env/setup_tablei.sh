#!/usr/bin/env bash
# Fresh Linux/Python 3.12 environment; never clones over or patches another repo.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${1:-"$ROOT/.venv-tablei"}
PYTHON=${PYTHON:-python3.12}
test ! -e "$ENV_DIR" || { echo "Use a new environment directory: $ENV_DIR" >&2; exit 1; }
"$PYTHON" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$ENV_DIR/bin/python" -m pip install -r "$ROOT/env/requirements-tablei.txt"
"$ENV_DIR/bin/python" "$ROOT/reproduce.py" verify
echo "Activate: source '$ENV_DIR/bin/activate'"
echo 'No training has been started. GPU run commands are in README.md.'
