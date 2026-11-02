#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
cd -- "$experiment_dir"
bash "$repo_root/GenHsaco.sh" kernel_dtk.ll 3 kernel.hsaco 2>&1 | tee build-gpu.log
python3 "$repo_root/launchKernel.py" \
  --hsaco "$experiment_dir/kernel.hsaco" --config "$experiment_dir/launch.json" \
  --warmup 10 --repeat 50 --json-output "$experiment_dir/gpu-result.json" 2>&1 | tee gpu.log
