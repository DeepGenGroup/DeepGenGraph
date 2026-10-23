#!/usr/bin/env bash
set -euo pipefail
variant=${1:-baseline}
case "$variant" in
  baseline|fragment_pair|ds_pair_prefetch|ds_pair_prefetch_soft) ;;
  *) echo 'usage: run.sh [baseline|fragment_pair|ds_pair_prefetch|ds_pair_prefetch_soft]' >&2; exit 2 ;;
esac
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$experiment_dir/../.." && pwd)
mkdir -p "$experiment_dir/build-$variant"
cd -- "$experiment_dir/build-$variant"
bash "$repo_dir/GenHsaco.sh" "$experiment_dir/$variant.ll" 3 kernel.hsaco
python3 "$experiment_dir/summarize_assembly.py" kernel.hsaco.amdgcn > assembly_counts.json
python3 "$repo_dir/launchKernel.py" --hsaco kernel.hsaco \
  --config "$experiment_dir/$variant.launch.json" \
  --warmup 50 --repeat 200 --json-output result.json
