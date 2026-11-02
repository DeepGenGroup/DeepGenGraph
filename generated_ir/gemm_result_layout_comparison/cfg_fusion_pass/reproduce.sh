#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../.." && pwd)
cd -- "$repo"
cmake --build 3rd/deepgengraph/build --target MyTest LegacyLLVMExportTest -j 2
python3 3rd/deepgengraph/test/check_register_math.py 3rd/deepgengraph/build/test/LegacyLLVMExportTest
python3 TuneAttention.py \
  --input generated_ir/online_softmax/test_input_1x32x4096x128-128-128-32-online.mlir \
  --sweep-base online-baseline --sweep 'layout=3.1;3.3' \
  --set mask-add-bypass=approx --set local-reduce=approx --set defer-lane-sum=approx \
  --llvm-as /data2/xsl/install/bin/llvm-as --output-dir "$here/reproduced"
