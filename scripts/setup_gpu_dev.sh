#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bootstrap_python="${VIPP_GPU_SETUP_PYTHON:-python3.12}"
python_path="$("$bootstrap_python" -c 'import sys; print(sys.executable)')"

exec "$python_path" "$script_dir/setup_gpu_dev.py" \
  --python "$python_path" "$@"
