#!/usr/bin/env bash
# Reproduce a performance-profile pair, not the sd16 compile-regression defaults.
# The input is sd16, but QK/VO are explicitly retiled to 32/32 for this profile.
set -euo pipefail
output_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$output_dir/../.." && pwd)
cd -- "$repo_dir"
llvm_bin=${LLVM_BIN:-/data2/xsl/install/bin}
test_dir=3rd/deepgengraph/test
build_dir=3rd/deepgengraph/build/test

for mode in off on; do
  enabled=0
  if [[ "$mode" == on ]]; then enabled=1; fi
  # reorder, shared packing, layout, carrier budget, math, threads, warp grid,
  # lifetimes, uniform masks, global prefetch, mask-add, LDS reuse,
  # DMA packing, DMA export, QK tile, VO tile.
  FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
    "$build_dir/MyTest" "$test_dir/test_input_sd16.mlir" 0 \
    "$output_dir/candidate.$mode.llvm.ll" \
    1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 32 32 \
    "--frag-prefetch=$enabled" --frag-prefetch-words=4 \
    > "$output_dir/candidate.$mode.log" 2>&1
  test -s "$output_dir/candidate.$mode.llvm.ll"
  rg -q 'LowerFriskAsyncCopy: 6 direct, 0 staged, 0 fallback; staging bytes 0' \
    "$output_dir/candidate.$mode.log"
  "$build_dir/LegacyLLVMExportTest" --llvm22-text \
    "$output_dir/candidate.$mode.llvm.ll" > "$output_dir/candidate.$mode.ll"
  "$llvm_bin/opt" -passes=verify -disable-output "$output_dir/candidate.$mode.ll"
  python3 "$test_dir/check_attention_llvm_numerics.py" "$output_dir/candidate.$mode.ll" \
    > "$output_dir/numerics.$mode.log" 2>&1
done
python3 "$test_dir/check_frag_prefetch_export.py" \
  "$output_dir/candidate.off.ll" "$output_dir/candidate.on.ll" \
  > "$output_dir/pair-validation.log"
echo "Generated and numerically checked candidate.off.ll / candidate.on.ll."
echo "Target GPU correctness, allocated registers and performance still require measurement."
