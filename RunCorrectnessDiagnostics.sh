#!/usr/bin/env bash
# Four controlled cases. This diagnoses correctness; timings are not benchmarks.
# Launch dimensions below belong to this BM=128, 256-thread attention export.
# Usage: bash RunCorrectnessDiagnostics.sh [input.ll] [opt-level] [output-dir]
# LLVM_DIR, ROCM_BC and GPU_ARCH are forwarded to GenHsaco.sh unchanged.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Fail before building four HSACOs when a remote copy omitted dependencies.
for required in kernelLauncher.py attentionDiagnostics.py GenHsaco.sh \
    3rd/deepgengraph/test/make_attention_diagnostic_variants.py \
    3rd/deepgengraph/test/check_gemm_pv_addresses.py; do
    if [[ ! -f "$repo_dir/$required" ]]; then
        printf 'Missing diagnostic dependency: %s/%s\nCopy the complete diagnostic bundle before running.\n' "$repo_dir" "$required" >&2
        exit 2
    fi
done
input_ir=$(realpath "${1:-$repo_dir/generated_ir/test_input_spidD_best_k5_hybrid_syncfix.ll}")
opt_level=${2:-3}
output_dir=${3:-$repo_dir/generated_ir/correctness_diagnostics}
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
python "$repo_dir/3rd/deepgengraph/test/make_attention_diagnostic_variants.py" "$input_ir" "$output_dir"
printf 'case\tstatus\tlog\n' > "$output_dir/summary.tsv"

for variant in reference sync_copy mmac_wait sync_copy_mmac_wait; do
    case_dir="$output_dir/$variant"
    mkdir -p "$case_dir"
    variant_ir="$output_dir/$variant.ll"
    if [[ "$variant" == reference ]]; then variant_ir="$input_ir"; fi
    printf 'Building %s with O%s; log: %s/build.log\n' "$variant" "$opt_level" "$case_dir"
    # GenHsaco uses relative intermediate filenames. Isolate each invocation.
    if ! (cd "$case_dir" && bash "$repo_dir/GenHsaco.sh" "$variant_ir" "$opt_level" "$case_dir/kernel.hsaco") > "$case_dir/build.log" 2>&1; then
        printf '%s\tBUILD_FAIL\t%s/build.log\n' "$variant" "$case_dir" >> "$output_dir/summary.tsv"
        printf 'Build failed; inspect %s/build.log\n' "$case_dir"
        continue
    fi
    printf 'Testing %s; log: %s/run.log\n' "$variant" "$case_dir"
    if python "$repo_dir/kernelLauncher.py" --hsaco "$case_dir/kernel.hsaco" \
        --kernel Attn_p2 --grid 32,32,1 --block 256,1,1 --k-layout bhsd \
        --warmup 0 --repeat 1 --diagnose-attention > "$case_dir/run.log" 2>&1; then
        status=PASS
    else
        result=$?
        status="EXIT_$result"
    fi
    printf '%s\t%s\t%s/run.log\n' "$variant" "$status" "$case_dir" >> "$output_dir/summary.tsv"
done
cat "$output_dir/summary.tsv"
printf 'Diagnostic run finished. Inspect the per-case logs; no performance conclusions.\n'
