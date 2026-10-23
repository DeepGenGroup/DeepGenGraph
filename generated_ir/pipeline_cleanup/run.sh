#!/usr/bin/env bash
# Target gfx936 + DTK + HIP-enabled PyTorch. LLVM_DIR/ROCM_BC may be overridden.
set -euo pipefail
variant=${1:-combined}
case "$variant" in
  baseline|wait_only|loop_only|combined|loop_only_40k|nohint_unrolled|nohint_unrolled_32k) ;;
  *) echo "usage: bash run.sh [baseline|wait_only|loop_only|combined|loop_only_40k|nohint_unrolled|nohint_unrolled_32k]" >&2; exit 1 ;;
esac
artifact_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$artifact_dir/../.." && pwd)
mkdir -p "$artifact_dir/hsaco/$variant"
cd -- "$artifact_dir/hsaco/$variant"
GPU_ARCH=gfx936 bash "$repo_dir/GenHsaco.sh" \
  "$artifact_dir/$variant.ll" 3 "$artifact_dir/$variant.hsaco"
# Check the allocation in target assembly, including the capacity controls.
python3 - "$artifact_dir/$variant.hsaco.amdgcn" "$variant" <<'PY'
import re
import sys
from pathlib import Path
sizes = {"baseline": 40960, "wait_only": 40960, "loop_only": 32768,
         "combined": 32768, "loop_only_40k": 40960,
         "nohint_unrolled": 16384, "nohint_unrolled_32k": 32768}
text = Path(sys.argv[1]).read_text()
match = re.search(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", text)
if not match:
    match = re.search(r"\.group_segment_fixed_size:\s*(\d+)", text)
if not match or int(match[1]) != sizes[sys.argv[2]]:
    raise SystemExit(f"Unexpected target LDS allocation: {match[1] if match else 'missing'}, "
                     f"expected {sizes[sys.argv[2]]}")
print(f"Verified target LDS allocation: {match[1]} bytes")
PY
python3 "$repo_dir/launchKernel.py" \
  --hsaco "$artifact_dir/$variant.hsaco" --kernel Attn_p2 \
  --grid 32,32,1 --block 256,1,1 --shared-mem 0 --k-layout bhsd \
  --device 0 --warmup 20 --repeat 100
