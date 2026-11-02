#!/usr/bin/env python3
"""Fingerprint-scoped causal variants of the exact measured VDB artifact.

This experiment keeps its protected K prefetch/PV schedule byte-for-byte.
It is intentionally specialized to BHSD=(1,32,4096,128), BM=BN=128, wave64,
256 threads and layout 3.3; it is not a generic noncausal-to-causal compiler pass.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'noncausal_v_double_buffer_group32/v_double_buffer.ll'
SOURCE_SHA = '3b2bde804277d68be2926e1ad35a292164d4688a349bedcefa42ab605322888a'
LLVM = Path(os.environ.get('IR_VERIFY_LLVM_DIR', '/data2/xsl/install/bin'))
GROUPS = (4, 8, 16, 32)
EXTRACT = re.compile(r'  (%[\w.]+) = extractelement <4 x float> (%[\w.]+), i64 ([0-3]), !dbg !44')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def score_layout(source):
    extracts = EXTRACT.findall(source)
    assert len(extracts) == 64
    # Permuted K maps MMAC's lane/element columns to (tid & 48) + offset.
    # Q rows are (tid & 15) + ((tid >> 2) & 48) + row repeat * 64.
    # verify.py independently traces these coordinates through actual Q/K DMA,
    # LDS reads and MMAC operand layouts for all 16,384 scores.
    return [dict(name=n, vector=v, element=int(e), row=i//32*64,
                 column=(i%32)//16*64+i%16)
            for i, (n,v,e) in enumerate(extracts)]


def transform(source, group):
    replacements = [
        ('%grp.n12 = and i64 %grp.n11, 576460752303423456,',
         f'%grp.n12 = and i64 %grp.n11, {(1<<59)-group},'),
        ('%grp.n13 = and i64 %grp.n6, 31,', f'%grp.n13 = and i64 %grp.n6, {group-1},'),
        ('%grp.n15 = lshr i64 %grp.n10, 5,', f'%grp.n15 = lshr i64 %grp.n10, {group.bit_length()-1},'),
        ('  %grp.n14 = or disjoint i64 %grp.n12, %grp.n13, !dbg !36',
         '  %causal.forward = or disjoint i64 %grp.n12, %grp.n13\n'
         '  %grp.n14 = sub nuw nsw i64 31, %causal.forward, !dbg !36'),
        ('%grp.n127 = icmp samesign ult i64 %grp.n12010, 31,',
         '%grp.n127 = icmp samesign ult i64 %grp.n12010, %grp.n14,'),
    ]
    text = source
    for before, after in replacements:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    entry = [
        '  ; Causal row-minus-column thresholds, invariant across key tiles.',
        '  %causal.qbase = trunc i64 %grp.n21 to i32',
        '  %causal.lane.row = and i32 %grp.n25, 15',
        '  %causal.wave.shift = lshr i32 %grp.n25, 2',
        '  %causal.wave.row = and i32 %causal.wave.shift, 48',
        '  %causal.local.row = or disjoint i32 %causal.lane.row, %causal.wave.row',
        '  %causal.row = add nuw nsw i32 %causal.qbase, %causal.local.row',
        '  %causal.lane.col = and i32 %grp.n25, 48',
        '  %causal.limit = sub nsw i32 %causal.row, %causal.lane.col',
    ]
    for i, record in enumerate(score_layout(source)):
        original = EXTRACT.search(text, text.index(f'  {record["name"]} =')).group()
        # The original scalar name now denotes the masked value, so both local
        # max and exponentiation use -infinity for future positions.
        masked = original.replace(f'{record["name"]} =', f'%causal.raw.{i} =')
        masked += f'\n  %causal.valid.{i} = icmp sge i32 %causal.delta, {record["column"]-record["row"]}'
        masked += f'\n  {record["name"]} = select i1 %causal.valid.{i}, float %causal.raw.{i}, float 0xFFF0000000000000'
        previous = 'poison' if i%4 == 0 else f'%causal.score.{i-1}'
        masked += (f'\n  %causal.score.{i} = insertelement <4 x float> {previous}, '
                   f'float {record["name"]}, i64 {record["element"]}')
        text = text.replace(original, masked, 1)
        if i%4 == 3:
            before = f'fsub <4 x float> {record["vector"]},'
            assert text.count(before) == 1
            text = text.replace(before,f'fsub <4 x float> %causal.score.{i},')
    branch = '  br label %grp.n128, !dbg !40'
    assert text.count(branch) == 1
    text = text.replace(branch, '\n'.join(entry)+'\n'+branch)
    phi = '  %grp.n12010 = phi i64 [ 0, %4 ], [ %grp.n1146, %grp.n128 ]'
    text = text.replace(phi, phi+'\n  %causal.keytile = trunc i64 %grp.n12010 to i32\n'
                        '  %causal.keybase = shl nuw nsw i32 %causal.keytile, 7\n'
                        '  %causal.delta = sub nsw i32 %causal.limit, %causal.keybase')
    return text


def check_mapping(text, group):
    entry = text.split('  %grp.n25 =', 1)[0].splitlines()
    kv = [l for l in text.splitlines() if l.startswith(('  %grp.n129 =', '  %grp.n130 ='))]
    seen = set()
    for y in range(32):
        for x in range(32):
            values = {}
            for line in entry+kv:
                match = re.match(r'  (%[\w.]+) = (.*?)(?:, !dbg.*)?$', line)
                if not match:
                    continue
                name, expr = match.groups()
                if '@llvm.amdgcn.workgroup.id.' in expr:
                    values[name] = x if '.id.x()' in expr else y
                elif cast := re.fullmatch(r'sext i32 (%[\w.]+) to i64', expr):
                    values[name] = values[cast[1]]
                else:
                    op = re.fullmatch(r'(shl|lshr|and|or|add|sub)(?: (?:nuw|nsw|disjoint))* i64 ([%\w.]+), ([%\w.]+)', expr)
                    assert op, expr
                    a,b = [values[v] if v.startswith('%') else int(v) for v in op.groups()[1:]]
                    if op[1] == 'or' and 'disjoint' in expr:
                        assert a & b == 0
                    values[name] = {'shl':lambda:a<<b,'lshr':lambda:a>>b,'and':lambda:a&b,
                                    'or':lambda:a|b,'add':lambda:a+b,'sub':lambda:a-b}[op[1]]()
                    assert 0 <= values[name] < (1<<63)
            t = x+32*y
            query = 31-((t//(32*group))*group+t%group)
            head = (t//group)%32
            assert values['%grp.n14'] == query
            assert values['%grp.n21'] == query*128
            assert values['%grp.n22'] == values['%grp.n130'] == head*4096*128
            assert values['%grp.n24'] == (head*4096+query*128)*128
            seen.add((head,query))
    assert len(seen) == 1024
    return dict(blocks_checked=1024, reverse_bx=True, bijective=True, address_bases_checked=True)


def main():
    assert digest(SOURCE) == SOURCE_SHA, 'Measured best source changed; re-audit before generating'
    source = SOURCE.read_text()
    old = json.loads(SOURCE.with_name('manifest.json').read_text())
    config = json.loads(SOURCE.with_suffix('.launch.json').read_text())
    config['causal'] = True
    vgen = module('vgen', SOURCE.with_name('generate.py'))
    kdir = HERE.parent/'noncausal_grouped_lds_prefetch'
    kgen = module('kgen', kdir/'generate.py')
    km = json.loads((kdir/'manifest_protected.json').read_text())
    pv = SOURCE.with_name('baseline_pv_interleave4.ll').read_text()
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, shape=config['shape'],
                    tiles=dict(BM=128,BN=128,BK=32), lds_bytes=16384, v_slots=[0,1,0,1],
                    causal=True, reverse_bx=True, gpu_tested=False,
                    generation='fingerprint-scoped transform of exact measured artifact', variants={})
    for group in GROUPS:
        text = transform(source, group)
        path = HERE/f'group{group}.ll'
        path.write_text(text)
        path.with_suffix('.launch.json').write_text(json.dumps(config,indent=2)+'\n')
        mapping = check_mapping(text,group)
        schedule = vgen.audit(text,pv,True)
        subprocess.run([LLVM/'llvm-as',path,'-o','/dev/null'],check=True)
        optimized = HERE/f'group{group}.o3.ll'
        subprocess.run([LLVM/'opt','-passes=default<O3>,verify','-S',path,'-o',optimized],check=True)
        # O3 may rename arithmetic temporaries; verify memory scheduling anchors.
        opt_text = optimized.read_text()
        assert vgen.audit(opt_text,pv,True) == schedule
        for candidate in (text,opt_text):
            lines = candidate.splitlines()
            kgen.assert_groups_unbroken(lines,km['groups'])
            defs = kgen.definitions(lines)
            for record in km['moved_groups']:
                assert max(defs[n][0] for n in record['loads']) < defs[record['anchor']][0]
                assert 'sched.barrier(i32 0)' in lines[defs[record['anchor']][0]-1]
        params = {**old['parameters'], 'workgroup-group-size':str(group),
                  'workgroup-reverse':'1','v-double-buffer':'1'}
        manifest['variants'][f'group{group}'] = dict(parameters=params, sha256=digest(path),
            o3_sha256=digest(optimized), llvm_and_o3_verified=True, mapping=mapping,
            k_prefetch_preserved=True, all_mmac_operands_and_order_preserved=True, schedule=schedule)
        print(f'PASS group{group}: reverse mapping, LLVM/O3, K/PV/VDB schedules preserved',flush=True)
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
