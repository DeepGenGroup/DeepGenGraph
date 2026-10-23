#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$bundle_dir/env.sh" ]]; then
    source "$bundle_dir/env.sh"
fi
exec "${PYTHON:-python3}" "$bundle_dir/run_tuning.py" "$@"
