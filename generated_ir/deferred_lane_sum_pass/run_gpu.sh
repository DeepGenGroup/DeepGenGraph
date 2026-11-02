#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
if [[ $# == 0 ]]; then
  set -- baseline deferred
fi
for variant in "$@"; do
  case "$variant" in
    baseline|deferred) ;;
    *) echo "Unknown variant: $variant (use baseline or deferred)" >&2; exit 1 ;;
  esac
done
for variant in "$@"; do
  ir_file="$experiment_dir/kernel.ll"
  if [[ "$variant" == baseline ]]; then ir_file="$experiment_dir/baseline.ll"; fi
  mkdir -p -- "$experiment_dir/$variant"
  (
    cd -- "$experiment_dir/$variant"
    bash "$repo_root/GenHsaco.sh" "$ir_file" 3 kernel.hsaco 2>&1 | tee build-gpu.log
    python3 "$repo_root/launchKernel.py" \
      --hsaco "$PWD/kernel.hsaco" --config "$experiment_dir/launch.json" \
      --warmup 10 --repeat 50 --json-output "$PWD/gpu-result.json" 2>&1 | tee gpu.log
  )
done
