env FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
  /data2/xsl/DeepGenGraph/3rd/deepgengraph/build/test/MyTest \
  --input="/data2/xsl/DeepGenGraph/generated_ir/best_early_wave_full_dim_bk32_20261120/test_input.mlir" --pipeline=0 --output="/data2/xsl/DeepGenGraph/kernel_retest.ll" \
  --reorder=1 --pack-shared=1 --wmma-layout=3.3 --math=approx \
  --threads=256 --block-warps=4,1 --register-lifetimes=1 --uniform-masks=1 \
  --prefetch-packets=1 --fuse-mask-add=0 --lds-reuse-words=0 \
  --dma-pack=useKPack --dma-export=asm128 --qk-tile=32 --vo-tile=32 \
  --frag-prefetch=0 --frag-prefetch-words=4 --retain-fragment-words=32 \
  --pv-interleave=4 --v-double-buffer=1 --k-dma=64-swizzle --k-dma-arg=1 \
  --k-lds-prefetch=8 --simplify-loop-control=1 --coalesce-waits=0 \
  --mask-add-bypass=approx --defer-lane-sum=approx --local-reduce=approx \
  --local-reduce-chains=4 --local-reduce-min-inputs=16 --local-reduce-ops=all \
  --swizzle-q=off --swizzle-k=8:6 --swizzle-v=off \
  --swizzle-q-arg=-1 --swizzle-k-arg=-1 --swizzle-v-arg=-1 \
  --workgroup-group-size=32 --workgroup-reverse=0 \
  --workgroup-query-axis=x --workgroup-head-axis=y \
  --softmax-fragments=pv --k-tail-prefetch=4 --cross-bn-k0=1 \
  --lazy-softmax-base=8 --softmax-cvt-pack=rtz --hoist-lds-bases=1 --hoist-lane-shuffles=1 --dump-llvm-passes=1 --llvm-opt=3 "$@"
