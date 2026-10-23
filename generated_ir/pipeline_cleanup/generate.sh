#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_dir"
for variant in baseline wait_only loop_only combined; do
  loop=0
  waits=0
  case "$variant" in
    wait_only) waits=1 ;;
    loop_only) loop=1 ;;
    combined) loop=1; waits=1 ;;
  esac
  FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
    3rd/deepgengraph/build/test/MyTest \
    3rd/deepgengraph/test/test_input_spidD.mlir 0 \
    "generated_ir/pipeline_cleanup/$variant.ll" \
    1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 32 32 \
    --frag-prefetch=1 --frag-prefetch-words=4 \
    "--simplify-loop-control=$loop" "--coalesce-waits=$waits" \
    > "generated_ir/pipeline_cleanup/$variant.log" 2>&1
done
