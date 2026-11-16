#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
if [[ $# == 0 ]]; then
  set -- measured pass_pk_rtz measured pipeline_t8_pk_rtz measured
fi
for variant in "$@"; do
  case "$variant" in
    measured|pass_pk_rtz|pipeline_t8_pk_rtz|pass_off) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
  esac
done
mkdir -p -- "$experiment_dir/runs"
batch_dir=$(mktemp -d "$experiment_dir/runs/batch-XXXXXX")
for variant in "$@"; do
  mode=pk_rtz
  if [[ $variant == measured ]]; then
    ir="$experiment_dir/../noncausal_cvt_pack/lazy_t8_pk_rtz.o3.ll"
    config="$experiment_dir/../noncausal_cvt_pack/lazy_t8_pk_rtz.launch.json"
  else
    ir="$experiment_dir/$variant.o3.ll"
    config="$experiment_dir/$variant.o3.launch.json"
    if [[ $variant == pass_off ]]; then mode=baseline; fi
  fi
  run_dir=$(mktemp -d "$batch_dir/$variant-XXXXXX")
  (
    cd -- "$run_dir"
    sha256sum "$ir" > input.sha256
    bash "$repo_root/GenHsaco.sh" "$ir" 3 kernel.hsaco 2>&1 | tee build.log
    if python3 "$experiment_dir/../noncausal_cvt_pack/check_assembly.py" kernel.hsaco.amdgcn "$mode" > assembly.json; then
      cat assembly.json
      python3 "$repo_root/launchKernel.py" --hsaco "$PWD/kernel.hsaco" --config "$config" \
        --non-causal --warmup "${ATTN_WARMUP:-20}" --repeat "${ATTN_REPEAT:-100}" \
        --json-output "$PWD/result.json" 2>&1 | tee gpu.log
    else
      cat assembly.json
      echo "SKIP $variant: scratch/spill or unexpected conversion count"
    fi
  )
done
echo "Results: $batch_dir"
