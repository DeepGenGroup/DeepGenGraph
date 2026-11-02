#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../.." && pwd)
llvm_bin=${LOCAL_LLVM_BIN:-/data2/xsl/install/bin}
tool="$repo/3rd/deepgengraph/build/test/LegacyLLVMExportTest"
source_ir="$here/../exp2_fixed/layout_3_1/kernel.ll"
"$tool" --fuse-mmac-accumulator "$source_ir" > "$here/fuse_without_cfg.ll" 2> "$here/fuse_without_cfg.log"
"$llvm_bin/opt" -passes=simplifycfg -S "$source_ir" -o "$here/cfg_simplified.ll"
"$tool" --fuse-mmac-accumulator "$here/cfg_simplified.ll" > "$here/fused.llvm24.ll" 2> "$here/fusion.log"
"$tool" --llvm22-text "$here/fused.llvm24.ll" > "$here/kernel.ll"
"$llvm_bin/opt" '-passes=default<O3>,verify' -S "$here/kernel.ll" -o "$here/kernel.o3.ll"
OPENBLAS_NUM_THREADS=1 python3 "$repo/3rd/deepgengraph/test/check_online_softmax.py" \
  "$here/kernel.ll" --bn 128 --group 4 --q-loads per-tile \
  --output "$here/validation.json" > "$here/validation.log" 2>&1
