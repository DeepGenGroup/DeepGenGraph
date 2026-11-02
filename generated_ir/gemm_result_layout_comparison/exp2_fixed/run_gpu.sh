#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../../.." && pwd)
if [[ $# == 0 ]]; then set -- 3.1 3.3; fi
for variant in "$@"; do
  case "$variant" in
    3.1|3.3) ;;
    *) echo 'Use 3.1 or 3.3' >&2; exit 2 ;;
  esac
done
for variant in "$@"; do
  directory="$experiment_dir/layout_${variant//./_}"
  mkdir -p -- "$directory/gpu"
  (
    cd -- "$directory/gpu"
    bash "$repo_root/GenHsaco.sh" "$directory/kernel.ll" 3 kernel.hsaco 2>&1 | tee build.log
    python3 "$repo_root/launchKernel.py" \
      --hsaco "$PWD/kernel.hsaco" --config "$directory/launch.json" \
      --warmup 10 --repeat 50 --json-output "$PWD/result.json" 2>&1 | tee test.log
  )
done
