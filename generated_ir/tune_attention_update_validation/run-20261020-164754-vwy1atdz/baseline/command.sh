#!/usr/bin/env bash
set -euo pipefail
cd -- /data2/xsl/DeepGenGraph/generated_ir/tune_attention_update_validation/run-20261020-164754-vwy1atdz/baseline
env FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 /data2/xsl/DeepGenGraph/3rd/deepgengraph/build/test/MyTest /data2/xsl/DeepGenGraph/generated_ir/tune_attention_update_validation/run-20261020-164754-vwy1atdz/baseline/input.mlir 0 /data2/xsl/DeepGenGraph/generated_ir/tune_attention_update_validation/run-20261020-164754-vwy1atdz/baseline/kernel.ll 1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 32 32 --frag-prefetch=1 --frag-prefetch-words=4 --simplify-loop-control=1 --coalesce-waits=0 > /data2/xsl/DeepGenGraph/generated_ir/tune_attention_update_validation/run-20261020-164754-vwy1atdz/baseline/lowering.log 2>&1
