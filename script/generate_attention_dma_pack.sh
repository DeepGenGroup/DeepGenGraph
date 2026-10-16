#!/usr/bin/env bash
# Export best_k5 packed layouts with an explicitly selected DMA instruction path.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dma_export=${DMA_EXPORT:-asm128}
wmma_layout=${WMMA_LAYOUT:-3.3}
case "$dma_export" in
    split32) default_output=kpack_dma32_compat ;;
    asm128) default_output=kpack_dma128_asm ;;
    *) echo "DMA_EXPORT must be split32 or asm128" >&2; exit 1 ;;
esac
output_dir=${1:-"$repo_dir/generated_ir/$default_output"}
tool=${MYTEST:-"$repo_dir/3rd/deepgengraph/build/test/MyTest"}
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
for mode in useKPack noKPack; do
    # argv 4..17: reorder, shared packing, layout, carrier, math, threads,
    # block-warps, lifetimes, masks, prefetch, fuse, reuse, DMA pack strategy,
    # instruction export mode (split32 preserves the logical 16B layout).
    FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
        "$tool" "$repo_dir/3rd/deepgengraph/test/test_input_spidD.mlir" \
        0 "$output_dir/$mode.ll" 1 1 "$wmma_layout" 34816 approx 256 4,1 1 1 1 0 0 "$mode" "$dma_export" \
        > "$output_dir/$mode.log" 2>&1
    if [[ ! -s "$output_dir/$mode.ll" ]] || rg -q 'error:|LLVM ERROR|Assertion .*failed' "$output_dir/$mode.log"; then
        tail -80 -- "$output_dir/$mode.log" >&2
        exit 1
    fi
    if [[ "$dma_export" == split32 ]] && {
        ! rg -q 'call void @llvm\.amdgcn\.global\.load\.async\.lds\(.*i32 4, i32 0, i32 0\)' "$output_dir/$mode.ll" ||
        rg -q 'llvm\.hcu\.global\.|global\.load\.async\.lds\(.*i32 16, i32' "$output_dir/$mode.ll";
    }; then
        echo "Missing split32 DMA export; rebuild MyTest: $output_dir/$mode.ll" >&2
        exit 1
    fi
    if [[ "$dma_export" == asm128 ]] && {
        ! rg -q 'buffer_load_dwordx4 .* offen lds' "$output_dir/$mode.ll" ||
        ! rg -q 's_waitcnt vmcnt\(0\)' "$output_dir/$mode.ll" ||
        rg -q 'call void @llvm\.(hcu\.global\.|amdgcn\.(global\.load\.async\.lds|wait\.asyncmark|asyncmark))' "$output_dir/$mode.ll";
    }; then
        echo "Missing asm128 DMA/wait export; rebuild MyTest: $output_dir/$mode.ll" >&2
        exit 1
    fi
    echo "$output_dir/$mode.ll"
done
sha256sum -- "$output_dir/useKPack.ll" "$output_dir/noKPack.ll"
