#!/usr/bin/env python3
"""Generate isolated A/B exports; GPU correctness and timing are a separate step.

Defaults reproduce BM=BN=128, BK=32, 256 threads and {4,1}. Budgets and input
are arguments, not assumptions made by the compiler passes.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=project / 'test/test_input_spidD.mlir')
parser.add_argument('--tool', type=Path, default=project / 'build/test/MyTest')
parser.add_argument('--output-dir', type=Path, default=project / 'build/attention_opt')
parser.add_argument('--threads', type=int, default=256)
parser.add_argument('--suite', choices=['original', 'mask-fusion'], default='original')
parser.add_argument('--block-warps', default='4,1')
parser.add_argument('--partial-budget', type=int, default=28672)
parser.add_argument('--full-budget', type=int, default=34816)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
variants = [
    ('step1_address', None, args.partial_budget, 'mmac', 0, 0),
    ('step2_fulljam', 'step1_address', args.full_budget, 'mmac', 0, 0),
    ('step3_fulljam_mask', 'step2_fulljam', args.full_budget, 'mmac', 1, 0),
    ('step4_exp2', 'step3_fulljam_mask', args.full_budget, 'mmac-exp2', 1, 0),
    ('step4_recip', 'step3_fulljam_mask', args.full_budget, 'mmac-recip', 1, 0),
    ('step4_math', 'step3_fulljam_mask', args.full_budget, 'approx', 1, 0),
    ('step5_prefetch', 'step4_math', args.full_budget, 'approx', 1, 1),
]
# Keep the original measured suite reproducible after enabling mask/add fusion
# by default in the compiler. The control intentionally uses its old lowering.
variants = [(*variant, 0) for variant in variants]
if args.suite == 'mask-fusion':
    variants = [
        ('step6_mask_mmac', 'step3_fulljam_mask', args.full_budget, 'mmac', 1, 0, 1),
        ('step6_mask_math_prefetch', 'step5_prefetch', args.full_budget, 'approx', 1, 1, 1),
        ('step6_mask_exp2_prefetch', 'step6_control_exp2_prefetch', args.full_budget, 'mmac-exp2', 1, 1, 1),
        ('step6_control_exp2_prefetch', 'step2_fulljam', args.full_budget, 'mmac-exp2', 0, 1, 0),
    ]
manifest = {'input': str(args.input.resolve()), 'threads': args.threads,
            'block_warps': args.block_warps, 'hardware_tested': False, 'variants': []}
for name, parent, budget, math, mask, prefetch, fuse_mask_add in variants:
    path = args.output_dir / (name + '.ll')
    log_path = args.output_dir / (name + '.log')
    command = [str(args.tool), str(args.input), '0', str(path), '1', '1', '3.3',
               str(budget), math, str(args.threads), args.block_warps, '1', str(mask), str(prefetch), str(fuse_mask_add)]
    with log_path.open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    text = path.read_text()
    log = log_path.read_text()
    prefetched = re.search(r'PrefetchGlobalPackets: (\d+) packets', log)
    manifest['variants'].append({
        'name': name, 'compare_to': parent, 'file': str(path.resolve()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'command': command,
        'carrier_budget': budget, 'math_mode': math, 'uniform_masks': bool(mask),
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
