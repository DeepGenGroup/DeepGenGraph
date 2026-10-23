#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_dir"
FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
3rd/deepgengraph/build/test/MyTest \
  3rd/deepgengraph/test/test_input_spidD.mlir 0 \
  generated_ir/auto_pipeline_candidates/attention.ll \
  1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 32 32 \
  --frag-prefetch=1 --frag-prefetch-words=4 \
  > generated_ir/auto_pipeline_candidates/attention.log 2>&1
