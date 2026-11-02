#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
if [[ $# == 0 ]]; then
  set -- control k0_prefetch_pv_tail control k0_late_control control
fi
for variant in "$@"; do
  case "$variant" in
    control|k0_late_control|k0_prefetch_pv_tail) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
  esac
done
mkdir -p -- "$experiment_dir/runs"
for variant in "$@"; do
  run_dir=$(mktemp -d "$experiment_dir/runs/$variant-XXXXXX")
  (
    cd -- "$run_dir"
    sha256sum "$experiment_dir/$variant.ll" > input.sha256
    bash "$repo_root/GenHsaco.sh" "$experiment_dir/$variant.ll" 3 kernel.hsaco 2>&1 | tee build.log
    python3 "$repo_root/launchKernel.py" --hsaco "$PWD/kernel.hsaco" \
      --config "$experiment_dir/$variant.launch.json" --non-causal \
      --warmup "${ATTN_WARMUP:-20}" --repeat "${ATTN_REPEAT:-100}" \
      --json-output "$PWD/result.json" 2>&1 | tee gpu.log
  )
done
echo "Results: $experiment_dir/runs"
