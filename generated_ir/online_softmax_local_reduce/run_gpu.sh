#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
if [[ $# == 0 ]]; then
  set -- baseline max4 sum4 max_sum4
fi
for variant in "$@"; do
  case "$variant" in
    baseline|max4|sum4|max_sum4) ;;
    *) echo "Unknown variant: $variant" >&2; exit 1 ;;
  esac
done
for variant in "$@"; do
  (
    cd -- "$experiment_dir/$variant"
    bash "$repo_root/GenHsaco.sh" kernel.ll 3 kernel.hsaco 2>&1 | tee build-gpu.log
    python3 "$repo_root/launchKernel.py" \
      --hsaco "$PWD/kernel.hsaco" --config "$PWD/launch.json" \
      --warmup 10 --repeat 50 --json-output "$PWD/gpu-result.json" 2>&1 | tee gpu.log
  )
done
