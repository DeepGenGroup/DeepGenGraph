#!/usr/bin/env bash
# 在有 gfx936 DTK/GPU 的环境运行；沿用 LLVM_DIR / ROCM_BC。
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
if [[ $# == 0 ]]; then
  set -- control prefetch4 control prefetch8 control
fi
for variant in "$@"; do
  case "$variant" in
    control|prefetch4|prefetch8) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
  esac
done
mkdir -p -- "$experiment_dir/runs"
for variant in "$@"; do
  run_dir=$(mktemp -d "$experiment_dir/runs/$variant-XXXXXX")
  (
    cd -- "$run_dir"
    sha256sum "$experiment_dir/$variant/kernel.ll" > input.sha256
    bash "$repo_root/GenHsaco.sh" "$experiment_dir/$variant/kernel.ll" 3 kernel.hsaco \
      2>&1 | tee build.log
    python3 "$repo_root/launchKernel.py" --hsaco "$PWD/kernel.hsaco" \
      --config "$experiment_dir/$variant/launch.json" --non-causal \
      --warmup "${ATTN_WARMUP:-20}" --repeat "${ATTN_REPEAT:-100}" \
      --json-output "$PWD/result.json" 2>&1 | tee gpu.log
  )
done
echo "Results: $experiment_dir/runs"
