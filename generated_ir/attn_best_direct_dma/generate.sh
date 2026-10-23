#!/usr/bin/env bash
# Current performance candidate with independent direct DMA issue overlap.
set -euo pipefail
output_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$output_dir/../.." && pwd)
cd -- "$repo_dir"
llvm_bin=${LLVM_BIN:-/data2/xsl/install/bin}
build_dir=3rd/deepgengraph/build
test_dir=3rd/deepgengraph/test

cmake --build "$build_dir" --target MyTest LegacyLLVMExportTest -j 2 \
  > "$output_dir/build.log" 2>&1
FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
  "$build_dir/test/MyTest" "$test_dir/test_input_sd16.mlir" 0 \
  "$output_dir/attention.llvm.ll" \
  1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 useKPack asm128 32 32 \
  --frag-prefetch=1 --frag-prefetch-words=4 \
  > "$output_dir/attention.log" 2>&1
"$build_dir/test/LegacyLLVMExportTest" --llvm22-text \
  "$output_dir/attention.llvm.ll" > "$output_dir/attention.ll"
"$llvm_bin/opt" -passes='verify,default<O3>,verify' -disable-output \
  "$output_dir/attention.ll" > "$output_dir/verify.log" 2>&1
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 "$test_dir/check_attention_llvm_numerics.py" "$output_dir/attention.ll" \
  > "$output_dir/numerics.log" 2>&1

python3 - "$output_dir" <<'PY'
from pathlib import Path
import hashlib
import json
import re
import sys

out = Path(sys.argv[1])
text = (out / 'attention.ll').read_text()
log = (out / 'attention.log').read_text()
numerics = (out / 'numerics.log').read_text()
events = [(i + 1, 'dma' if 'buffer_load_dwordx4' in line else 'wait')
          for i, line in enumerate(text.splitlines())
          if 'call ' in line and
          ('buffer_load_dwordx4' in line or 's_waitcnt vmcnt(0)' in line)]
counts = {
    'dma_calls': sum(kind == 'dma' for _, kind in events),
    'zero_count_waits': sum(kind == 'wait' for _, kind in events),
    'block_barriers': len(re.findall(r'call void @llvm.amdgcn.s.barrier\(', text)),
}
assert counts == {'dma_calls': 12, 'zero_count_waits': 6, 'block_barriers': 12}, counts
assert [kind for _, kind in events[:5]] == ['dma'] * 4 + ['wait'], events[:6]
assert 'LowerFriskAsyncCopy: 6 direct, 0 staged, 0 fallback; staging bytes 0' in log
assert numerics.count('allclose=True') == 4 and 'mismatch=0' in numerics
assert 'PASS mathematical LLVM execution' in numerics
report = {
    'file': str((out / 'attention.ll').relative_to(Path.cwd())),
    'sha256': hashlib.sha256((out / 'attention.ll').read_bytes()).hexdigest(),
    'input': '3rd/deepgengraph/test/test_input_sd16.mlir',
    'input_sha256': hashlib.sha256(Path('3rd/deepgengraph/test/test_input_sd16.mlir').read_bytes()).hexdigest(),
    'reproduce': 'bash generated_ir/attn_best_direct_dma/generate.sh',
    'configuration': {
        'copy': 'async', 'alignment': 16, 'layout': '3.3',
        'carrier_words': 34816, 'math': 'approx', 'threads': 256,
        'block_warps': [4, 1], 'qk_tile': 32, 'vo_tile': 32,
        'dma_pack': 'useKPack', 'dma_export': 'asm128',
        'uniform_masks': True, 'global_prefetch': True, 'fuse_mask_add': False,
        'fragment_prefetch': True, 'fragment_prefetch_words': 4,
        'independent_direct_dma_overlap': True,
    },
    'launch': {
        'kernel': 'Attn_p2', 'grid': [32, 32, 1], 'block': [256, 1, 1],
        'argument_order': ['Q', 'K', 'V', 'Out'],
        'shape': [1, 32, 4096, 128], 'dtype': 'float16',
        'layout': 'contiguous BHSD', 'static_lds_bytes': 16384,
        'dynamic_shared_memory_bytes': 0,
    },
    'static_counts': counts,
    'first_qk_events': events[:6],
    'validation': {'llvm_o3_verify': True, 'numerical_model_cases_passed': 4,
                   'target_gpu_tested': False, 'target_gpu_time_ms': None},
    'performance_reference': {
        'artifact': 'generated_ir/frag_prefetch_v2/temp.frag-prefetch.ll',
        'source': 'pmc_test.log', 'recorded_time_ms': 2.152,
        'note': 'Historical fast artifact only; the newly generated IR has not been GPU timed and is not proven fastest.',
    },
}
assert '@shm_0 = addrspace(3) global [16384 x i8]' in text
(out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'file': report['file'], 'sha256': report['sha256'],
                  'counts': counts, 'validation': report['validation']}, indent=2))
PY
