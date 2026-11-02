#!/usr/bin/env bash
set -euo pipefail
artifact_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$artifact_dir/../.." && pwd)
cmake --build "$repo_root/3rd/deepgengraph/build" --target MyTest -j 2
cd -- "$artifact_dir"
env FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
  "$repo_root/3rd/deepgengraph/build/test/MyTest" \
  "$repo_root/3rd/deepgengraph/test/test_input_spidD_noncasual.mlir" \
  0 "$repo_root/generated_ir/test_input_spidD_noncasual.ll" \
  1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 32 32 \
  --frag-prefetch=0 --frag-prefetch-words=4 --retain-fragment-words=32 \
  --simplify-loop-control=1 --coalesce-waits=0 \
  --mask-add-bypass=approx --defer-lane-sum=approx \
  --local-reduce=approx --local-reduce-chains=4 \
  --local-reduce-min-inputs=16 --local-reduce-ops=all \
  --swizzle-q=off --swizzle-k=8:6 --swizzle-v=off \
  --swizzle-q-arg=-1 --swizzle-k-arg=-1 --swizzle-v-arg=-1 \
  --workgroup-group-size=4 --workgroup-reverse=1 \
  --workgroup-query-axis=x --workgroup-head-axis=y > lowering.log 2>&1
echo "LLVM IR: $repo_root/generated_ir/test_input_spidD_noncasual.ll"
