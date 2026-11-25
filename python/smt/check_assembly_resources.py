#!/usr/bin/env python3
"""Check final AMDGPU metadata, not an IR estimate of physical registers.

Accepts UTF-8 and BOM-marked UTF-16 assembly. A missing resource field fails
closed. This is a post-codegen acceptance check, not a CP-SAT register model.
The VGPR budget counts registers (v0..v255 is 256, not 255); allocation rounding
can make the reported count larger than the highest live virtual-register count.
"""
import argparse
import json
from pathlib import Path
import re
import sys


def inspect_assembly(path, vgpr_budget=250, lds_budget=16384):
    if vgpr_budget <= 0 or lds_budget <= 0:
        raise ValueError('budgets must be positive')
    data = Path(path).read_bytes()
    text = data.decode('utf-16' if data[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig')
    match = re.search(r'^amdhsa\.kernels:\s*\n(.*?)(?=^amdhsa\.|^\.\.\.|\Z)', text, re.M | re.S)
    if not match:
        raise ValueError('missing amdhsa.kernels metadata')
    kernels = []
    for block in re.split(r'^  - (?=\.)', match[1], flags=re.M)[1:]:
        name = re.search(r'^    \.name:\s*(\S+)', block, re.M)
        if not name:
            raise ValueError('kernel metadata missing name')
        values = {}
        for field in ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count',
                      'private_segment_fixed_size', 'group_segment_fixed_size'):
            found = re.search(r'^    \.' + field + r':\s*(\d+)\s*$', block, re.M)
            if not found:
                raise ValueError(f'{name[1]}: missing {field}')
            values[field] = int(found[1])
        reasons = []
        if values['vgpr_count'] > vgpr_budget:
            reasons.append('VGPR budget exceeded')
        if values['group_segment_fixed_size'] > lds_budget:
            reasons.append('LDS budget exceeded')
        if values['vgpr_spill_count'] or values['sgpr_spill_count']:
            reasons.append('register spill reported')
        if values['private_segment_fixed_size']:
            reasons.append('nonzero private/scratch segment')
        kernels.append(dict(name=name[1], **values, accepted=not reasons, reasons=reasons))
    if not kernels:
        raise ValueError('no kernel resource entries')
    return dict(vgpr_budget=vgpr_budget, lds_budget_bytes=lds_budget,
                accepted=all(k['accepted'] for k in kernels), kernels=kernels,
                accounting='final codegen metadata; private memory may also include explicit allocas')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assembly', type=Path)
    parser.add_argument('--vgpr-budget', type=int, default=250)
    parser.add_argument('--lds-budget', type=int, default=16384)
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()
    try:
        report = inspect_assembly(args.assembly, args.vgpr_budget, args.lds_budget)
        rendered = json.dumps(report, indent=2) + '\n'
        if args.output:
            args.output.write_text(rendered)
        print(rendered, end='')
        return 0 if report['accepted'] else 1
    except (OSError, ValueError) as error:
        print(f'check_assembly_resources: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
