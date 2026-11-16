#!/usr/bin/env bash
set -euo pipefail
artifact_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$artifact_dir/../.." && pwd)
mkdir -p "$artifact_dir/runs"
run_dir=$(mktemp -d "$artifact_dir/runs/gpu-XXXXXX")
cd "$run_dir"
sha256sum "$artifact_dir/kernel.ll" > input.sha256
bash "$repo_dir/GenHsaco.sh" "$artifact_dir/kernel.ll" 3 kernel.hsaco 2>&1 | tee build.log
python3 "$repo_dir/launchKernel.py" --hsaco "$run_dir/kernel.hsaco" \
  --config "$artifact_dir/kernel.launch.json" --non-causal \
  --warmup "${ATTN_WARMUP:-20}" --repeat "${ATTN_REPEAT:-100}" \
  --json-output "$run_dir/result.json" 2>&1 | tee gpu.log
