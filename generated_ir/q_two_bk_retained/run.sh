#!/usr/bin/env bash
set -euo pipefail
variant=${1:-baseline_8words}
case "$variant" in
  baseline_8words|retained_16words) ;;
  *) echo 'usage: run.sh [baseline_8words|retained_16words]' >&2; exit 2 ;;
esac
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$experiment_dir/../.." && pwd)
mkdir -p "$experiment_dir/build-$variant"
cd -- "$experiment_dir/build-$variant"
bash "$repo_dir/GenHsaco.sh" "$experiment_dir/$variant.ll" 3 kernel.hsaco
python3 "$repo_dir/generated_ir/retained_ds_pair_schedule/summarize_assembly.py" \
  kernel.hsaco.amdgcn > assembly_counts.json
python3 "$repo_dir/launchKernel.py" --hsaco kernel.hsaco \
  --config "$experiment_dir/$variant.launch.json" \
  --warmup 50 --repeat 200 --json-output result.json
