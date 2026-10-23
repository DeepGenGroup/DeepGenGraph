#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Keep the user's current baseline untouched.
source_ir=${1:-generated_ir/attn_best_async_current.ll}
output_dir=generated_ir/mask_compare_compact
export_tool=3rd/deepgengraph/build/test/LegacyLLVMExportTest
llvm_bin=${LLVM_BIN:-/data2/xsl/install/bin}
mkdir -p "$output_dir"
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT
"$export_tool" --optimize-register-lifetimes "$source_ir" \
  > "$temp_dir/raw.ll" 2> "$output_dir/export.log"
"$export_tool" --llvm22-text "$temp_dir/raw.ll" > "$output_dir/attention.ll"
"$llvm_bin/opt" -passes=verify -disable-output "$output_dir/attention.ll"
"$llvm_bin/opt" -S '-passes=default<O3>' "$output_dir/attention.ll" \
  -o "$temp_dir/o3.ll"
python - "$source_ir" "$output_dir/attention.ll" "$temp_dir/o3.ll" <<'PY'
import hashlib, json, re, sys
from pathlib import Path

def describe(path):
    data = Path(path).read_bytes()
    text = data.decode()
    counts = {}
    for count, width in re.findall(
            r'icmp (?:samesign )?\w+ (?:<(\d+) x i(\d+)>|i(?:32|64))', text):
        if width:
            counts['vector_i' + width] = counts.get('vector_i' + width, 0) + int(count)
    for width in (32, 64):
        counts['scalar_i' + str(width)] = len(re.findall(
            rf'icmp (?:samesign )?\w+ i{width}\b', text))
    mma = [line for line in text.splitlines() if 'v_mmac_' in line and ' asm ' in line]
    return {'sha256': hashlib.sha256(data).hexdigest(),
            'integer_comparisons': counts,
            'mmac_calls': len(mma),
            'mmac_sideeffect_calls': sum('asm sideeffect ' in line for line in mma),
            'local_comparison_boundaries': text.count('"=v,0,i"')}

report = {name: describe(path) for name, path in
          zip(('baseline', 'candidate', 'candidate_after_o3'), sys.argv[1:])}
report['baseline']['path'] = sys.argv[1]
report['candidate']['path'] = sys.argv[2]
report['validation'] = {
    'llvm_verify': True,
    'llvm_o3': True,
    'gfx936_codegen_or_execution': False,
    'note': 'Run the independent semantic tests and target benchmark separately.'}
Path(sys.argv[2]).with_name('validation.json').write_text(
    json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
