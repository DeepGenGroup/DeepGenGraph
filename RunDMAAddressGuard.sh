#!/usr/bin/env bash
# Retain DMA, prevent source-only immediate folding, verify codegen, then test.
# Usage: bash RunDMAAddressGuard.sh [input.ll] [optimization-level] [output-dir]
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
input_ir=$(realpath "${1:-$repo_dir/generated_ir/test_input_spidD_best_k5_hybrid_syncfix.ll}")
opt_level=${2:-3}
output_dir=${3:-$repo_dir/generated_ir/dma_address_guard}
for required in GenHsaco.sh kernelLauncher.py attentionDiagnostics.py \
    3rd/deepgengraph/test/make_attention_diagnostic_variants.py \
    3rd/deepgengraph/test/check_gemm_pv_addresses.py \
    3rd/deepgengraph/test/check_dma_assembly.py; do
    if [[ ! -f "$repo_dir/$required" ]]; then
        printf 'Missing dependency: %s/%s\n' "$repo_dir" "$required" >&2
        exit 2
    fi
done
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
python "$repo_dir/3rd/deepgengraph/test/make_attention_diagnostic_variants.py" \
    "$input_ir" --guard-dma-output "$output_dir/kernel.ll"
(cd "$output_dir" && bash "$repo_dir/GenHsaco.sh" "$output_dir/kernel.ll" \
    "$opt_level" "$output_dir/kernel.hsaco") 2>&1 | tee "$output_dir/build.log"
python "$repo_dir/3rd/deepgengraph/test/check_dma_assembly.py" \
    "$output_dir/kernel.hsaco.amdgcn" --expected-count 16 2>&1 | tee "$output_dir/address-check.log"
# Launch geometry/count are specific to this BM=128, 256-thread attention export.
python "$repo_dir/kernelLauncher.py" --hsaco "$output_dir/kernel.hsaco" \
    --kernel Attn_p2 --grid 32,32,1 --block 256,1,1 --k-layout bhsd \
    --warmup 0 --repeat 1 --diagnose-attention 2>&1 | tee "$output_dir/run.log"
