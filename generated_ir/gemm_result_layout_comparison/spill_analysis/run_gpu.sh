#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../.." && pwd)
mkdir -p -- "$here/gpu"
cd -- "$here/gpu"
bash "$repo/GenHsaco.sh" "$here/kernel.ll" 3 kernel.hsaco 2>&1 | tee build.log
python3 "$repo/launchKernel.py" --hsaco "$PWD/kernel.hsaco" \
  --config "$here/launch.json" --warmup 10 --repeat 50 \
  --json-output "$PWD/result.json" 2>&1 | tee test.log
