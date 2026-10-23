#!/usr/bin/env bash
# Run on the target DTK/HIP machine; LLVM_DIR and ROCM_BC may be overridden.
set -euo pipefail
artifact_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$artifact_dir/../.." && pwd)
variant=${1:-group8_reverse_coalesce_waits}
case "$variant" in
  group8_reverse_coalesce_waits|group8_reverse_control) ;;
  *) echo "usage: bash run.sh [group8_reverse_coalesce_waits|group8_reverse_control]" >&2; exit 1 ;;
esac
mkdir -p "$artifact_dir/hsaco/$variant"
cd -- "$artifact_dir/hsaco/$variant"
GPU_ARCH=${GPU_ARCH:-gfx936} bash "$repo_dir/GenHsaco.sh" \
  "$artifact_dir/$variant.ll" 3 kernel.hsaco
python3 "$repo_dir/launchKernel.py" \
  --hsaco kernel.hsaco --config "$artifact_dir/launch.json" \
  --device 0 --warmup 50 --repeat 200 --json-output result.json
