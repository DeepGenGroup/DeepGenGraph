#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
# 默认以group4穿插对照，便于识别时钟/温度等造成的基线漂移。
# 参数可取 group4 / group8 / group16 / group32，按给出的顺序执行。
if [[ $# == 0 ]]; then
  set -- group4 group8 group4 group16 group4 group32 group4
fi
for variant in "$@"; do
  case "$variant" in
    group4|group8|group16|group32) ;;
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
