#!/usr/bin/env bash
# Best non-causal parameters from generated_ir/async_task_noncausal/command.sh,
# unchanged. The new passes are on by default, so no new flag is needed. Pass
# their flags (0|1) after this command's arguments to A/B test one of them.
set -euo pipefail
ROOT=/data2/xsl/DeepGenGraph
HERE=$ROOT/generated_ir/wave_invariant_noncausal
OUT=${OUT:-$HERE/kernel.ll}
env FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 $ROOT/3rd/deepgengraph/build/test/MyTest $ROOT/generated_ir/async_task_noncausal/input.mlir 0 "$OUT" 1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 32 32 --frag-prefetch=0 --frag-prefetch-words=4 --retain-fragment-words=32 --pv-interleave=4 --v-double-buffer=1 --k-dma=64-swizzle --k-dma-arg=1 --k-lds-prefetch=8 --simplify-loop-control=1 --coalesce-waits=0 --mask-add-bypass=approx --defer-lane-sum=approx --local-reduce=approx --local-reduce-chains=4 --local-reduce-min-inputs=16 --local-reduce-ops=all --swizzle-q=off --swizzle-k=8:6 --swizzle-v=off --swizzle-q-arg=-1 --swizzle-k-arg=-1 --swizzle-v-arg=-1 --workgroup-group-size=32 --workgroup-reverse=0 --workgroup-query-axis=x --workgroup-head-axis=y --softmax-fragments=pv --k-tail-prefetch=4 --cross-bn-k0=1 "$@"
