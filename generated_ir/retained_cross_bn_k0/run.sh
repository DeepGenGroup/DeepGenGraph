#!/usr/bin/env bash
set -euo pipefail
variant=${1:-k0_prefetch_pv_tail}
label=${2:-run}
case "$variant" in
  baseline|k0_late_control|k0_prefetch_pv_tail) ;;
  *) echo 'usage: run.sh [baseline|k0_late_control|k0_prefetch_pv_tail] [label]' >&2; exit 2 ;;
esac
if [[ ! "$label" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo 'label must contain only letters, numbers, dots, underscores or hyphens' >&2
  exit 2
fi
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$experiment_dir/../.." && pwd)
mkdir -p "$experiment_dir/runs"
run_dir=$(mktemp -d "$experiment_dir/runs/${variant}-${label}-XXXXXX")
echo "Variant: $variant; artifacts: $run_dir"
cp -- "$experiment_dir/$variant.ll" "$run_dir/input.ll"
cp -- "$experiment_dir/$variant.launch.json" "$run_dir/launch.json"
cd -- "$run_dir"
printf '%s\n' "$variant" > variant.txt
sha256sum -- input.ll launch.json > input.sha256
bash "$repo_dir/GenHsaco.sh" "$run_dir/input.ll" 3 kernel.hsaco 2>&1 | tee compile.log
python3 "$repo_dir/generated_ir/retained_ds_pair_schedule/summarize_assembly.py" \
  kernel.hsaco.amdgcn > assembly_counts.json
python3 "$repo_dir/launchKernel.py" --hsaco kernel.hsaco \
  --config "$run_dir/launch.json" --warmup "${K0_WARMUP:-50}" \
  --repeat "${K0_REPEAT:-200}" --json-output result.json 2>&1 | tee benchmark.log
echo "Saved: $run_dir/result.json"
