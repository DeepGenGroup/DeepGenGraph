cd /data2/xsl/DeepGenGraph
case_dir=generated_ir/best_noncausal_early_wave

env FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
  ./test/MyTest \
  "/home/chenguanyu/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD_noncasual.mlir" \
  0 \
  1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 32 32 \
  --frag-prefetch=0 --frag-prefetch-words=4 --retain-fragment-words=32 \
  --pv-interleave=4 --v-double-buffer=1 \
  --k-dma=64-swizzle --k-dma-arg=1 \
  --k-lds-prefetch=8 --simplify-loop-control=1 --coalesce-waits=0 \
  --mask-add-bypass=approx --defer-lane-sum=approx --local-reduce=approx \
  --local-reduce-chains=4 --local-reduce-min-inputs=16 --local-reduce-ops=all \
  --swizzle-q=off --swizzle-k=8:6 --swizzle-v=off \
  --swizzle-q-arg=-1 --swizzle-k-arg=-1 --swizzle-v-arg=-1 \
  --workgroup-group-size=32 --workgroup-reverse=0 \
  --workgroup-query-axis=x --workgroup-head-axis=y \
  --softmax-fragments=pv --k-tail-prefetch=4 --cross-bn-k0=1 \
  --lazy-softmax-base=8 --softmax-cvt-pack=rtz \
  --hoist-lds-bases=1 --hoist-lane-shuffles=1 \
  --dump-llvm-passes=1 \
  > lll.log 2>&1