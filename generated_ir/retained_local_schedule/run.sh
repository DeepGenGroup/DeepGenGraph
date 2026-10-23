#!/usr/bin/env bash
set -euo pipefail
variant=${1:-baseline}
case "$variant" in
  baseline|wait_tail2|fragment_pair|combined) ;;
  *) echo 'usage: run.sh [baseline|wait_tail2|fragment_pair|combined]' >&2; exit 2 ;;
esac
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$experiment_dir/../.." && pwd)
mkdir -p "$experiment_dir/build-$variant"
cd -- "$experiment_dir/build-$variant"
bash "$repo_dir/GenHsaco.sh" "$experiment_dir/$variant.ll" 3 kernel.hsaco
python3 "$repo_dir/launchKernel.py" --hsaco kernel.hsaco \
  --config "$experiment_dir/$variant.launch.json" \
  --warmup 50 --repeat 200 --json-output result.json
