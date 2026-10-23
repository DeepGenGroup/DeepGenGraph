#!/usr/bin/env bash
set -euo pipefail
output_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$output_dir/../.." && pwd)
cd -- "$repo_dir"

FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
  ./3rd/deepgengraph/build/test/MyTest \
  3rd/deepgengraph/test/test_input_spidD.mlir \
  0 generated_ir/k5_layout_matrix_kpack/attention.ll \
  1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 \
  > generated_ir/k5_layout_matrix_kpack/attention.log 2>&1

test -s "$output_dir/attention.ll"
if rg -q 'error:|LLVM ERROR|Assertion .*failed' "$output_dir/attention.log"; then
  tail -80 -- "$output_dir/attention.log" >&2
  exit 1
fi
echo "$output_dir/attention.ll"
