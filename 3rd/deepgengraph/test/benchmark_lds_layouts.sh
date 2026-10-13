#!/usr/bin/env bash
# Run on the DTK/gfx936 host. Compile all controls with the SAME toolchain/options.
# Preserve artifacts and repeat the first control to expose timing drift.
# Optional arguments: <variant-directory> <variant-name> ...
set -euo pipefail
test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$test_dir/../../.." && pwd)
variant_dir=${1:-"$test_dir/../build/perf-lds128"}
variant_dir=$(cd -- "$variant_dir" && pwd)
if (( $# )); then shift; fi
variants=("$@")
if (( ${#variants[@]} == 0 )); then
    variants=(read64-control paired-before paired-swizzled)
fi
for variant in "${variants[@]}"; do
    test -f "$variant_dir/$variant.ll"
done
run_dir=$(mktemp -d "$variant_dir/results.XXXXXX")
echo "Results: $run_dir"
for variant in "${variants[@]}"; do
    mkdir "$run_dir/$variant"
    (
        cd "$run_dir/$variant"
        bash "$repo_dir/GenHsaco.sh" "$variant_dir/$variant.ll" 3 kernel.hsaco > compile.log 2>&1
        llvm-objdump -d --mcpu=gfx936 kernel.hsaco > disassembly.log
        llvm-readobj --notes kernel.hsaco > metadata.log
    )
done
for variant in "${variants[@]}" "${variants[0]}"; do
    log=$(mktemp "$run_dir/$variant/timing.XXXXXX.log")
    python3 "$repo_dir/kernelLauncher.py" \
        --hsaco "$run_dir/$variant/kernel.hsaco" \
        --kernel Attn_p2 --grid 32,64,1 --block 128,1,1 \
        --warmup 20 --repeat 100 | tee "$log"
done
echo "Results: $run_dir"
