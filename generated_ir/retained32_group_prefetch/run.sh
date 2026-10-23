#!/usr/bin/env bash
# Run already exported IRs in the existing gfx936 DTK environment.
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$experiment_dir/../.." && pwd)
if [[ $# -lt 1 ]]; then
    echo 'usage: run.sh EXPORT_RUN_DIR [grid-000001 ...]' >&2
    exit 2
fi
export_dir=$(cd -- "$1" && pwd)
shift
# Baseline before, between, and after the five non-baseline candidates.
if [[ $# -eq 0 ]]; then
    set -- grid-000004 grid-000001 grid-000002 grid-000004 grid-000003 grid-000005 grid-000006 grid-000004
fi
for variant in "$@"; do
    case "$variant" in
        grid-00000[1-6]) ;;
        *) echo "invalid variant: $variant" >&2; exit 2 ;;
    esac
    test -s "$export_dir/$variant/kernel.ll"
done
mkdir -p "$experiment_dir/runs"
session_dir=$(mktemp -d "$experiment_dir/runs/retest-XXXXXX")
echo "Results: $session_dir"
iteration=0
for variant in "$@"; do
    build_dir="$session_dir/$variant"
    if [[ ! -d "$build_dir" ]]; then
        mkdir "$build_dir"
        cp -- "$export_dir/$variant/kernel.ll" "$build_dir/input.ll"
        cp -- "$export_dir/$variant/input.mlir" "$build_dir/input.mlir"
        cp -- "$export_dir/$variant/result.json" "$build_dir/export-result.json"
        python3 "$repo_dir/TuneAttention.py" --input "$build_dir/input.mlir" \
            --write-launch-config "$build_dir/input.ll"
        (
            cd -- "$build_dir"
            sha256sum -- input.ll launch.json > input.sha256
            bash "$repo_dir/GenHsaco.sh" input.ll 3 kernel.hsaco 2>&1 | tee compile.log
            sha256sum -- kernel.hsaco > hsaco.sha256
            python3 "$repo_dir/generated_ir/retained_ds_pair_schedule/summarize_assembly.py" \
                kernel.hsaco.amdgcn > assembly_counts.json
        )
    fi
    iteration=$((iteration + 1))
    (
        cd -- "$build_dir"
        python3 "$repo_dir/launchKernel.py" --hsaco kernel.hsaco --config launch.json \
            --warmup "${GROUP_PREFETCH_WARMUP:-50}" --repeat "${GROUP_PREFETCH_REPEAT:-200}" \
            --json-output "result-$iteration.json" 2>&1 | tee "benchmark-$iteration.log"
    )
done
echo "Saved: $session_dir"
