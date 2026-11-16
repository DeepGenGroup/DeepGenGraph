#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
export LLVM_DIR=${LLVM_DIR:-/workspace/deepgengraph/dtk/aillvm/bin}
export GPU_ARCH=${GPU_ARCH:-gfx936}
if [[ $# == 0 ]]; then
  set -- lazy_t8 lazy_t8_pk_rne lazy_t8 lazy_t8_sdwa_rne lazy_t8 lazy_t8_pk_rtz lazy_t8
fi
for variant in "$@"; do
  case "$variant" in
    control|lazy_t8|control_pk_rne|control_sdwa_rne|control_pk_rtz|lazy_t8_pk_rne|lazy_t8_sdwa_rne|lazy_t8_pk_rtz) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
  esac
done
feature_args=()
if [[ -n ${GPU_FEATURES:-} ]]; then feature_args=("-mattr=$GPU_FEATURES"); fi
mkdir -p -- "$experiment_dir/runs"
batch_dir=$(mktemp -d "$experiment_dir/runs/batch-XXXXXX")
"$LLVM_DIR/llc" --version > "$batch_dir/llvm_version.txt"
for variant in "$@"; do
  mode=baseline
  case "$variant" in
    *_pk_rne) mode=pk_rne ;;
    *_sdwa_rne) mode=sdwa_rne ;;
    *_pk_rtz) mode=pk_rtz ;;
  esac
  run_dir=$(mktemp -d "$batch_dir/$variant-XXXXXX")
  if [[ $mode != baseline ]]; then
    if ! "$LLVM_DIR/llc" -mtriple=amdgcn-amd-amdhsa -mcpu="$GPU_ARCH" "${feature_args[@]}" \
      -O3 -filetype=obj "$experiment_dir/probes/$mode.ll" -o "$run_dir/probe.o" > "$run_dir/probe.log" 2>&1; then
      echo "SKIP $variant: target instruction probe failed; see $run_dir/probe.log"
      continue
    fi
    if grep -qiE 'not a recognized processor|ignoring processor' "$run_dir/probe.log"; then
      echo "Unknown GPU_ARCH: $GPU_ARCH" >&2; exit 2
    fi
  fi
  (
    cd -- "$run_dir"
    sha256sum "$experiment_dir/$variant.ll" > input.sha256
    bash "$repo_root/GenHsaco.sh" "$experiment_dir/$variant.ll" 3 kernel.hsaco 2>&1 | tee build.log
    if python3 "$experiment_dir/check_assembly.py" kernel.hsaco.amdgcn "$mode" > assembly.json; then
      cat assembly.json
      python3 "$repo_root/launchKernel.py" --hsaco "$PWD/kernel.hsaco" \
        --config "$experiment_dir/$variant.launch.json" --non-causal \
        --warmup "${ATTN_WARMUP:-20}" --repeat "${ATTN_REPEAT:-100}" \
        --json-output "$PWD/result.json" 2>&1 | tee gpu.log
    else
      cat assembly.json
      echo "SKIP $variant: scratch/spill or unexpected conversion instruction count"
    fi
  )
done
echo "Results: $batch_dir"
