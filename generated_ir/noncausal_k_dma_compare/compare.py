#!/usr/bin/env python3
"""Compare measured DMA64 variants after the same LLVM O3 pipeline."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'noncausal_k_dma_pass'
sys.path.insert(0, str(SOURCE))
from validate import LazyAddresses
from check_gemm_pv_addresses import split_args


class Addresses(LazyAddresses):
    def value(self, name, ty=None):
        c = re.fullmatch(r'ptrtoint \((ptr addrspace\(3\) @[\w.]+) to i(?:32|64)\)', name)
        return self.typed(c[1]) if c else super().value(name, ty)


def inspect(text):
    ops, events, hints, reads, qk, uses = Counter(), [], [], {}, [], defaultdict(list)
    mm = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('  '):
            rhs = line.strip().split(' = ', 1)[-1]
            ops[re.sub(r'^tail ', '', rhs).split()[0]] += 1
        if m := re.match(r'  (%[\w.]+) = load <4 x half>, ptr addrspace\(3\) (%[\w.]+)', line):
            reads[m[1]] = dict(line=lineno, after_mmac_count=mm, pointer=m[2])
        if 'v_mmac_' in line:
            if mm < 128:
                args = split_args(re.search(r'"\((.*)\)', line)[1])
                k = args[1].split()[-1]
                qk.append(k)
                uses[k].append(mm)
            mm += 1
        if 'call ' not in line:
            continue
        if '@llvm.amdgcn.sched.barrier' in line:
            hints.append(mm)
        for kind in ('buffer_load_dwordx2', 'buffer_load_dwordx4', 'ds.read.m32', 's_waitcnt', 's.barrier'):
            if kind in line:
                events.append([mm, kind])
    assert mm == 256 and len(qk) == 128
    return dict(instructions=sum(ops.values()), opcodes=dict(ops), memory_schedule=events,
                qk_sched_barriers=[x for x in hints if x < 128], total_sched_barriers=len(hints),
                k_reads={','.join(map(str, u)): dict(value=k, **reads[k]) for k, u in uses.items()})


def main():
    result, texts, models = {}, {}, {}
    for name in ('k_swizzle64', 'mytest_k_swizzle64'):
        source = SOURCE / (name + '.ll')
        output = HERE / (name + '.o3.ll')
        subprocess.run(['/data2/xsl/install/bin/opt', '-S', '-strip-debug',
                        '-passes=default<O3>,verify', source, '-o', output], check=True)
        texts[name] = output.read_text()
        result[name] = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            raw=inspect(source.read_text()), optimized=inspect(texts[name]))
        models[name] = Addresses(texts[name])
    a, b = [result[n]['optimized'] for n in result]
    assert a['memory_schedule'] == b['memory_schedule']
    assert set(a['k_reads']) == set(b['k_reads']) and len(a['k_reads']) == 64
    first, second = models.values()
    first.reset(); second.reset()
    changes = []
    for consumers, old in a['k_reads'].items():
        new = b['k_reads'][consumers]
        assert np.array_equal(first.value(old['pointer']), second.value(new['pointer']))
        if old['after_mmac_count'] != new['after_mmac_count']:
            changes.append(dict(consumers=consumers, fast=old, mytest=new))
    dma = {n: [split_args(re.search(r'"\((.*)\)', l)[1]) for l in t.splitlines()
               if 'buffer_load_dwordx2 ' in l] for n, t in texts.items()}
    for head in range(32):
        for tile in range(32):
            first.reset(head, 0, tile); second.reset(head, 0, tile)
            for aa, bb in zip(*dma.values()):
                for x, y in zip(map(first.typed, aa), map(second.typed, bb)):
                    assert np.array_equal(x, y)
    result['comparison'] = dict(dma_operands_equal_all_heads_key_tiles_lanes=True,
        k_load_addresses_equal_all_lanes=True, memory_event_mmac_positions_identical=True,
        differing_k_load_positions=changes, gpu_codegen_compared=False,
        median_ms=dict(k_swizzle64=1.7415840029716492, mytest_k_swizzle64=1.7828629612922668),
        mytest_slowdown_percent=(1.7828629612922668/1.7415840029716492-1)*100)
    (HERE/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS: all K DMA operands, K load addresses and hardware memory-event positions match')
    print('K loads with different MMAC positions:', len(changes))
    print('QK sched.barrier count:', len(a['qk_sched_barriers']), len(b['qk_sched_barriers']))
    for c in changes[:8]:
        print(c['consumers'], c['fast']['after_mmac_count'], c['mytest']['after_mmac_count'])


if __name__ == '__main__':
    main()
