#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Q/K/V roles come from validated argument annotations. For another ABI, pass
# --swizzle-q-arg=N --swizzle-k-arg=N --swizzle-v-arg=N explicitly.
variant="${1:-k128}"
case "$variant" in
  k128) swizzle=(--swizzle-q=off --swizzle-k=on --swizzle-v=off) ;;
  qkv128) swizzle=(--swizzle-q=on --swizzle-k=on --swizzle-v=on) ;;
  off) swizzle=(--swizzle-q=off --swizzle-k=off --swizzle-v=off) ;;
  *) echo 'usage: generate.sh [k128|qkv128|off]' >&2; exit 1 ;;
esac
FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
3rd/deepgengraph/build/test/MyTest \
  generated_ir/pipeline_cleanup/nohint_unrolled.mlir 0 \
  "generated_ir/swizzle_pass/group8_reverse_${variant}.ll" \
  1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 32 32 \
  --frag-prefetch=1 --frag-prefetch-words=4 --simplify-loop-control=1 --coalesce-waits=0 \
  --workgroup-group-size=8 --workgroup-reverse=true "${swizzle[@]}" \
  > "generated_ir/swizzle_pass/generate_${variant}.log" 2>&1
