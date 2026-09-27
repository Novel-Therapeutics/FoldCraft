#!/usr/bin/env bash
# Reproduce the validated Linux/Python 3.12 GPU runtime in a new virtualenv.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$script_dir/scripts/install_runtime.py" "$@"
