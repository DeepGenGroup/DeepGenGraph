#!/usr/bin/env python3
"""Inspect static LLVM loop bodies and DMA/wait windows; no cycle prediction.

Usage: compare_llvm_schedule.py best.ll candidate.ll -o comparison.json
Counts include all blocks in a natural loop. Conditional paths are listed per
block; prologue/epilogue copies are excluded from mainloop comparisons.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def event(line):
    if 'call' in line and 'v_mmac_' in line:
        return 'mmac'
    if 'call' in line and 'buffer_load_dwordx4' in line:
        return 'dma128'
    if 'call' in line and 'buffer_load_dwordx2' in line:
        return 'dma64'
    if 'call' in line and 's_waitcnt vmcnt(0)' in line:
        return 'wait_all'
    if 'call' in line and '@llvm.amdgcn.s.barrier(' in line:
        return 'barrier'
    if 'call' in line and '@llvm.hcu.ds.read.' in line:
        return 'matrix_read'
    if re.search(r'\bload .*ptr addrspace\(3\)', line):
        return 'lds_load'
    return None


def inspect(path):
    lines = Path(path).read_text().splitlines()
    blocks, active, name = {}, False, '$entry'
    for number, line in enumerate(lines, 1):
        if re.match(r'define amdgpu_kernel .*@Attn_p2\(', line):
            active = True
            blocks[name] = []
            continue
        if not active:
            continue
        if line == '}':
            break
        label = re.match(r'^([\w.]+):', line)
        if label:
            name = label[1]
            blocks[name] = []
        else:
            blocks[name].append((number, line))
    if not blocks:
        raise ValueError(f'no Attn_p2 kernel in {path}')
    successors = {b: set(re.findall(r'label %([\w.]+)', '\n'.join(t for _, t in body)))
                  for b, body in blocks.items()}
    predecessors = defaultdict(set)
    for b, targets in successors.items():
        for target in targets:
            predecessors[target].add(b)
    reachable, pending = set(), ['$entry']
    while pending:
        b = pending.pop()
        if b not in reachable:
            reachable.add(b)
            pending.extend(successors[b])
    dom = {b: ({b} if b == '$entry' else set(reachable)) for b in reachable}
    changed = True
    while changed:
        changed = False
        for b in reachable - {'$entry'}:
            pred = predecessors[b] & reachable
            value = {b} | set.intersection(*(dom[p] for p in pred))
            if value != dom[b]:
                dom[b], changed = value, True
    natural = defaultdict(set)
    for latch in reachable:
        for header in successors[latch] & dom[latch]:
            members, pending = {header}, [latch]
            while pending:
                b = pending.pop()
                if b not in members:
                    members.add(b)
                    pending.extend(predecessors[b])
            natural[header].update(members)
    reports = []
    for header, members in natural.items():
        counts, per_block, events = Counter(), {}, []
        for b, body in blocks.items():
            if b not in members:
                continue
            local = Counter()
            for number, line in body:
                kind = event(line)
                if kind:
                    local[kind] += 1
                    events.append(dict(block=b, line=number, kind=kind))
            counts.update(local)
            per_block[b] = dict(local)
        windows, pending_dma, first_mmac, mmac = [], [], 0, 0
        for e in events:
            if e['kind'] == 'mmac':
                mmac += 1
            elif e['kind'].startswith('dma'):
                if not pending_dma:
                    first_mmac = mmac
                pending_dma.append(e)
            elif e['kind'] == 'wait_all':
                windows.append(dict(wait_line=e['line'], dma_packets=len(pending_dma),
                                    mmac_since_first_issue=mmac-first_mmac if pending_dma else None))
                pending_dma = []
        reports.append(dict(header=header, counts=dict(counts), blocks=per_block,
                            dma_windows_in_text_order=windows,
                            dma_packets_pending_at_backedge=len(pending_dma)))
    reports.sort(key=lambda r: -r['counts'].get('mmac', 0))
    return dict(path=str(Path(path).resolve()), loops=reports,
                caveat='Static natural-loop counts, all conditional blocks included; '
                       'DMA windows follow textual block order, not a simulated execution trace.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ir', nargs='+', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()
    rendered = json.dumps([inspect(path) for path in args.ir], indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end='')
