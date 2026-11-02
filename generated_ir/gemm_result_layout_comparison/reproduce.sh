#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$experiment_dir/../.." && pwd)
cd -- "$repo_root"
python3 TuneAttention.py \
  --input generated_ir/online_softmax/test_input_1x32x4096x128-128-128-32-online.mlir \
  --sweep-base online-baseline --sweep 'layout=3.1;3.3' \
  --set mask-add-bypass=approx --set local-reduce=approx --set defer-lane-sum=approx \
  --llvm-as /data2/xsl/install/bin/llvm-as \
  --output-dir "$experiment_dir/reproduced"
