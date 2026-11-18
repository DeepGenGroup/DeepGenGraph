#!/bin/bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$repo_dir"
mkdir -p generated_ir/k5_independent_tiles

FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
3rd/deepgengraph/build/test/MyTest \
  3rd/deepgengraph/test/test_input_shared_d.mlir 0 \
  generated_ir/k5_independent_tiles/qk64_vo32.ll \
  1 1 3.3 approx 256 4,1 1 1 1 0 0 useKPack asm128 64 32
