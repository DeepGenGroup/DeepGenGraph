#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
# warmup/repeat沿用基线；每个候选各自编译，并保留汇编和计时。
if [[ $# == 0 ]]; then
  set -- baseline v_tail_prefetch2 baseline v_prefetch2 v_late_dma_control v_prefetch4 baseline
fi
for variant in "$@"; do
  case "$variant" in
    baseline|v_tail_prefetch2|v_prefetch2|v_prefetch4|v_late_dma_control) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
  esac
done
mkdir -p -- "$experiment_dir/runs"
for variant in "$@"; do
  run_dir=$(mktemp -d "$experiment_dir/runs/$variant-XXXXXX")
  (
    cd -- "$run_dir"
    sha256sum "$experiment_dir/$variant.ll" > input.sha256
    bash "$repo_root/GenHsaco.sh" "$experiment_dir/$variant.ll" 3 kernel.hsaco \
      2>&1 | tee build.log
    python3 "$repo_root/launchKernel.py" --hsaco "$PWD/kernel.hsaco" \
      --config "$experiment_dir/$variant.launch.json" --non-causal \
      --warmup "${ATTN_WARMUP:-20}" --repeat "${ATTN_REPEAT:-100}" \
      --json-output "$PWD/result.json" 2>&1 | tee gpu.log
  )
done
echo "Results: $experiment_dir/runs"
