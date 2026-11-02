#!/usr/bin/env bash
set -euo pipefail
artifact_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$artifact_dir/../.." && pwd)
mkdir -p -- "$artifact_dir/gpu"
cd -- "$artifact_dir/gpu"
bash "$repo_root/GenHsaco.sh" \
  "$repo_root/generated_ir/test_input_spidD_noncasual.ll" 3 kernel.hsaco \
  2>&1 | tee build-gpu.log
python3 "$repo_root/launchKernel.py" \
  --hsaco "$PWD/kernel.hsaco" --config "$artifact_dir/launch.json" \
  --non-causal --warmup 10 --repeat 50 --json-output "$PWD/gpu-result.json" \
  "$@" 2>&1 | tee gpu.log
