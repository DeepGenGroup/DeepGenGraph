#!/usr/bin/env python3
"""Generate isolated A/B exports; GPU correctness and timing are a separate step.

Defaults generate BM=BN=128, full D=128, BK hint=32, 256 threads and {4,1}.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from TuneAttention import prepare_input, target_loop_iv

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=project / 'test/test_input_shared_d.mlir')
parser.add_argument('--tool', type=Path, default=project / 'build/test/MyTest')
parser.add_argument('--output-dir', type=Path, default=project / 'build/attention_opt')
parser.add_argument('--threads', type=int, default=256)
parser.add_argument('--suite', choices=['original', 'mask-fusion'], default='original')
parser.add_argument('--block-warps', default='4,1')
args = parser.parse_args()
prepared = args.output_dir / 'prepared_input.mlir'
if prepared.resolve() == args.input.resolve():
    parser.error('prepared input snapshot must differ from --input')
args.output_dir.mkdir(parents=True, exist_ok=True)
source = args.input.read_text()
iv = target_loop_iv(source, 'auto')
prepared.write_text(prepare_input(source, {'pipeline-stages': 'none', 'loop-mark': '1'}, iv)
                    if iv else source)
variants = [
    ('step1_shared', None, 'mmac', 0, 0),
    ('step3_shared_mask', 'step1_shared', 'mmac', 1, 0),
    ('step4_exp2', 'step3_shared_mask', 'mmac-exp2', 1, 0),
    ('step4_recip', 'step3_shared_mask', 'mmac-recip', 1, 0),
    ('step4_math', 'step3_shared_mask', 'approx', 1, 0),
    ('step5_prefetch', 'step4_math', 'approx', 1, 1),
]
# Control variants disable mask/add fusion so comparisons isolate the flags.
variants = [(*variant, 0) for variant in variants]
if args.suite == 'mask-fusion':
    variants = [
        ('step6_mask_mmac', 'step3_shared_mask', 'mmac', 1, 0, 1),
        ('step6_mask_math_prefetch', 'step5_prefetch', 'approx', 1, 1, 1),
        ('step6_mask_exp2_prefetch', 'step6_control_exp2_prefetch', 'mmac-exp2', 1, 1, 1),
        ('step6_control_exp2_prefetch', 'step1_shared', 'mmac-exp2', 0, 1, 0),
    ]
manifest = {'input': str(args.input.resolve()), 'prepared_input': str(prepared.resolve()),
            'threads': args.threads,
            'block_warps': args.block_warps, 'hardware_tested': False, 'variants': []}
for name, parent, math, mask, prefetch, fuse_mask_add in variants:
    path = args.output_dir / (name + '.ll')
    log_path = args.output_dir / (name + '.log')
    command = [str(args.tool), str(prepared), '0', str(path), '1', '1', '3.3',
               math, str(args.threads), args.block_warps, '1', str(mask), str(prefetch), str(fuse_mask_add)]
    with log_path.open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    text = path.read_text()
    log = log_path.read_text()
    prefetched = re.search(r'PrefetchGlobalPackets: (\d+) packets', log)
    manifest['variants'].append({
        'name': name, 'compare_to': parent, 'file': str(path.resolve()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'command': command,
        'math_mode': math, 'uniform_masks': bool(mask),
        'fuse_mask_add': bool(fuse_mask_add),
        'prefetch': bool(prefetch), 'prefetched_packets': int(prefetched[1]) if prefetched else 0,
        'static_mmac_calls': text.count('v_mmac_'),
        'ocml_exp2_calls': text.count('call float @__ocml_exp2_f32('),
        'native_exp2_calls': text.count('call float @llvm.amdgcn.exp2.f32('),
        'fdiv_instructions': len(re.findall(r'^  %.* = fdiv ', text, re.M)),
    })
    print(name, 'generated', flush=True)
manifest_name = 'candidates.json' if args.suite == 'original' else 'mask_fusion_candidates.json'
(args.output_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + '\n')
