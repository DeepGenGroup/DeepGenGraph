#!/usr/bin/env bash
# Run on the target gfx936 machine with DTK and HIP-enabled PyTorch.
# LLVM_DIR and ROCM_BC use GenHsaco.sh defaults unless overridden.
set -euo pipefail
artifact_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$artifact_dir/../.." && pwd)
# GenHsaco.sh writes intermediate files in the working directory.
mkdir -p "$artifact_dir/hsaco"
cd -- "$artifact_dir/hsaco"
GPU_ARCH=gfx936 bash "$repo_dir/GenHsaco.sh" \
  "$artifact_dir/attention.ll" 3 "$artifact_dir/attention.hsaco"
python3 "$repo_dir/launchKernel.py" \
  --hsaco "$artifact_dir/attention.hsaco" --kernel Attn_p2 \
  --grid 32,32,1 --block 256,1,1 --shared-mem 0 --k-layout bhsd \
  --device 0 --warmup 20 --repeat 100
