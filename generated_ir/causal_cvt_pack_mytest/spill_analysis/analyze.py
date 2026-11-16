#!/usr/bin/env python3
"""Static analysis of the user-supplied gfx936 assembly; no kernel modifications."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def scalar_entry(text, x, y):
    """Evaluate supported scalar preheader instructions, skipping unrelated DAGs."""
    values = {}
    def get(v):
        return values[v] if v.startswith('%') else int(v)
    for line in text.splitlines():
        if re.match(r'  br ', line):
            break
        m = re.match(r'  (%[\w.]+) = (.*)', line)
        if not m:
            continue
        name, body = m.groups()
        body = body.split(', !')[0]
        try:
            if m := re.search(r'@llvm.amdgcn.workgroup.id.([xyz])\(\)', body):
                value = {'x': x, 'y': y, 'z': 0}[m[1]]
            elif m := re.fullmatch(r'(sext|zext)(?: nneg)? i(\d+) (%[\w.]+) to i\d+', body):
                value = get(m[3])
                if m[1] == 'sext' and value >= 1 << (int(m[2])-1):
                    value -= 1 << int(m[2])
            elif m := re.fullmatch(r'(shl|lshr|add|sub|mul|sdiv|and|or|xor)(?: (?:nuw|nsw|disjoint|exact))* i64 ([%\w.-]+), ([%\w.-]+)', body):
                a, b = get(m[2]), get(m[3])
                value = {'shl':lambda:a<<b, 'lshr':lambda:(a%(1<<64))>>b,
                         'add':lambda:a+b, 'sub':lambda:a-b, 'mul':lambda:a*b,
                         'sdiv':lambda:abs(a)//abs(b)*(1 if (a<0)==(b<0) else -1),
                         'and':lambda:a&b, 'or':lambda:a|b, 'xor':lambda:a^b}[m[1]]()
            elif m := re.fullmatch(r'icmp (ugt|ult|eq) i64 ([%\w.-]+), ([%\w.-]+)', body):
                a, b = get(m[2]), get(m[3])
                value = int({'ugt':a%(1<<64)>b%(1<<64), 'ult':a%(1<<64)<b%(1<<64), 'eq':a==b}[m[1]])
            elif m := re.fullmatch(r'select i1 ([%\w.-]+), i64 ([%\w.-]+), i64 ([%\w.-]+)', body):
                value = get(m[2] if get(m[1]) else m[3])
            else:
                continue
            values[name] = value
        except KeyError:
            continue
    return values


def main():
    assembly = (HERE/'assembly.utf8.txt').read_text().split('.Lfunc_end0:', 1)[0]
    lines = assembly.splitlines()
    start = next(i for i,l in enumerate(lines) if l.startswith('.LBB0_5:'))
    end = next(i for i,l in enumerate(lines) if l.startswith('.LBB0_6:'))
    slots = defaultdict(list)
    for i, line in enumerate(lines):
        if 'Folded ' not in line:
            continue
        offset = re.search(r'offset:(\d+)', line)
        slot = int(offset[1]) if offset else 0
        slots[slot].append(dict(line=i+1, operation='load' if 'Reload' in line else 'store',
                               region='loop' if start<i<end else 'entry'))
    loop = lines[start:end]
    stores = sum('Folded Spill' in l for l in loop)
    loads = sum('Folded Reload' in l for l in loop)
    entry_stores = sum('Folded Spill' in l for l in lines[:start])
    assert (len(slots), stores, loads, entry_stores) == (24, 1, 25, 23)
    source_line = None
    by_source = defaultdict(Counter)
    for line in loop:
        if m := re.match(r'\s*\.loc\s+1\s+(\d+)',line):
            source_line = int(m[1])
        if m := re.match(r'\s+([vs]_[\w.]+|ds_[\w.]+|buffer_[\w.]+)\s',line):
            by_source[source_line][m[1]] += 1

    new_path = HERE.parent/'group4_reverse_pk_rtz.o3.ll'
    old_path = ROOT/'generated_ir/causal_v_double_buffer_reverse/group4.ll'
    new, old = new_path.read_text(), old_path.read_text()
    seen = set()
    for y in range(32):
        for x in range(32):
            n, o = scalar_entry(new,x,y), scalar_entry(old,x,y)
            query, head = 31-((x+32*y)//128*4+x%4), ((x+32*y)//4)%32
            assert (n['%16'],n['%28']) == (o['%grp.n14'],o['%grp.n22']//524288) == (query,head)
            assert n['%31'] == o['%grp.n21'] == query*128
            seen.add((head,query))
    assert len(seen) == 1024
    report = dict(
        assembly_sha256=hashlib.sha256((HERE/'assembly.utf8.txt').read_bytes()).hexdigest(),
        ir_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
        resources=dict(vgpr=256,vgpr_spill_count=24,sgpr_spill_count=0,
                       private_bytes_per_thread=100,lds_bytes=16384,compiler_reported_occupancy=1),
        spill_slots=dict(sorted(slots.items())),
        classification=dict(q_fragment_offsets=list(range(8,56,4)),
                            address_offsets=[56,60,80,84,88,92],
                            mask_row_high_word_offsets=[64,68,72,76],
                            thread_id_offsets=[0],denominator_state_offsets=[4]),
        static_counts=dict(loop_spill_stores=stores,loop_spill_loads=loads,entry_spill_stores=entry_stores),
        logical_scratch_traffic=dict(
            scope='one full launch; 32 heads, 32 causal query tiles, 256 active lanes per block',
            total_bytes=(entry_stores*1024+(loads+stores)*32*sum(range(1,33)))*256*4,
            q_reload_bytes=12*4*256*32*sum(range(1,33)),
            not_measured_dram_or_L2_bytes=True),
        block_mapping=dict(blocks_checked=1024,identical_to_historical_safe_group4=True,
                           first_eight=[dict(query=31-i%4,head=i//4) for i in range(8)]),
        ir_mask=dict(compare_base_identity_copies=len(re.findall(r'^  %compare.local.base.* = .*asm sideeffect',new,re.M)),
                     high_word_predicates=len(re.findall(r'^  %compare.high.* = icmp',new,re.M)),
                     vector_score_plus_mask_adds=new.count('fadd <4 x float>')),
        loop_asm_opcodes_by_mlir_source_line=dict(by_source),
        limitations=['No new GPU timing or target compilation.',
                     'Debug source attribution is approximate after scheduling.',
                     'Per-pass contributions to spill need target A/B compilation.',
                     'Aggregate L2 hit rate cannot isolate scratch, K or V misses.'])
    (HERE/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS 1024 block mappings; 24 spill slots; logical scratch traffic',
          report['logical_scratch_traffic']['total_bytes']/2**20, 'MiB')


if __name__ == '__main__':
    main()
