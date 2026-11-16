#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
if [[ $# == 0 ]]; then
  set -- measured group4_reverse_off group4_reverse_pk_rtz measured
fi
for variant in "$@"; do
  case "$variant" in
    measured|group4_reverse_off|group4_reverse_pk_rtz) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
  esac
done
mkdir -p -- "$experiment_dir/runs"
for variant in "$@"; do
  if [[ $variant == measured ]]; then
    input="$repo_root/generated_ir/causal_v_double_buffer_reverse/group4.ll"
    config="$repo_root/generated_ir/causal_v_double_buffer_reverse/group4.launch.json"
  else
    input="$experiment_dir/$variant.o3.ll"
    config="$experiment_dir/$variant.o3.launch.json"
  fi
  run_dir=$(mktemp -d "$experiment_dir/runs/$variant-XXXXXX")
  (
    cd -- "$run_dir"
    sha256sum "$input" > input.sha256
    bash "$repo_root/GenHsaco.sh" "$input" 3 kernel.hsaco 2>&1 | tee build.log
    python3 "$repo_root/launchKernel.py" --hsaco "$PWD/kernel.hsaco" \
      --config "$config" --causal \
      --warmup "${ATTN_WARMUP:-20}" --repeat "${ATTN_REPEAT:-100}" \
      --json-output "$PWD/result.json" 2>&1 | tee gpu.log
  )
done
echo "Results: $experiment_dir/runs"
