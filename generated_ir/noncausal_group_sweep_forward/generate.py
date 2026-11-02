#!/usr/bin/env python3
"""从已验证的 protected IR 生成 group-size=4/8/16/32、reverse=0 对照。

仅修改入口的三个分组常量，保留整组 LDS 预取及全部读取组合并保护。
这是绑定输入 SHA、BHSD=(1,32,4096,128)、grid=(32,32) 的实验生成器。
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'noncausal_grouped_lds_prefetch'
SOURCE = BASE / 'group_prefetch2_protected.ll'
SOURCE_SHA = 'db8504595594927cf46482508172a082b0608c05173a2557adef77155f3fd268'
LLVM = Path('/data2/xsl/install/bin')
GROUPS = (4, 8, 16, 32)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant(source, group):
    # t = physical_x + 32*physical_y
    # query = (t//(32*G))*G + t%G; head = (t//G)%32.
    # 在本例合法 launch 域内，t%G == physical_x%G。
    # 原 IR 已将 query 化简为 ((t>>5)&~(G-1)) | (physical_x&(G-1))。
    # reverse=0：不执行 31-query；其余计算、同步、调度提示保持逐字不变。
    replacements = [
        ('%grp.n12 = and i64 %grp.n11, 576460752303423484,',
         f'%grp.n12 = and i64 %grp.n11, {(1 << 59)-group},'),
        ('%grp.n13 = and i64 %grp.n6, 3,',
         f'%grp.n13 = and i64 %grp.n6, {group-1},'),
        ('%grp.n15 = lshr i64 %grp.n10, 2,',
         f'%grp.n15 = lshr i64 %grp.n10, {group.bit_length()-1},'),
    ]
    result = source
    for before, after in replacements:
        assert result.count(before) == 1, before
        result = result.replace(before, after)
    # 除三个映射常量外，包括 debug metadata 在内的所有文本必须完全一致。
    restored = result
    for before, after in replacements:
        restored = restored.replace(after, before)
    assert restored == source
    return result


def check_mapping(text, group):
    """执行实际 IR 入口的整数表达式，穷举1024个block及Q/O、K/V基址。"""
    entry = text.split('  %grp.n25 =', 1)[0].splitlines()
    kv_base = [line for line in text.splitlines()
               if line.startswith(('  %grp.n129 =', '  %grp.n130 ='))]
    seen = set()
    for y in range(32):
        for x in range(32):
            values = {}
            for line in entry + kv_base:
                match = re.match(r'  (%[\w.]+) = (.*?)(?:, !dbg.*)?$', line)
                if not match:
                    continue
                name, expr = match.groups()
                if '@llvm.amdgcn.workgroup.id.x()' in expr:
                    values[name] = x
                elif '@llvm.amdgcn.workgroup.id.y()' in expr:
                    values[name] = y
                elif cast := re.fullmatch(r'sext i32 (%[\w.]+) to i64', expr):
                    values[name] = values[cast[1]]
                else:
                    op = re.fullmatch(r'(shl|lshr|and|or|add|sub)(?: (?:nuw|nsw|disjoint))* i64 ([%\w.]+), ([%\w.]+)', expr)
                    assert op, expr
                    operands = [values[v] if v.startswith('%') else int(v) for v in op.groups()[1:]]
                    a, b = operands
                    if op[1] == 'or' and 'disjoint' in expr:
                        assert a & b == 0
                    values[name] = {'shl': lambda: a << b, 'lshr': lambda: a >> b,
                                    'and': lambda: a & b, 'or': lambda: a | b,
                                    'add': lambda: a + b, 'sub': lambda: a - b}[op[1]]()
                    assert 0 <= values[name] < (1 << 63)
            t = x + 32*y
            query, head = (t//(32*group))*group + t%group, (t//group)%32
            assert values['%grp.n14'] == query
            assert values['%grp.n21'] == query*128  # Q/O row base
            assert values['%grp.n22'] == head*4096*128  # Q/O head base in half elements
            assert values['%grp.n24'] == (head*4096 + query*128)*128
            assert values['%grp.n130'] == head*4096*128  # K/V head base
            seen.add((head, query))
    assert seen == {(h, q) for h in range(32) for q in range(32)}
    return dict(blocks_checked=1024, bijective=True, q_o_and_k_v_bases_checked=True)


def main():
    assert sha(SOURCE) == SOURCE_SHA, 'Protected baseline changed'
    source = SOURCE.read_text()
    spec = importlib.util.spec_from_file_location('protected_generator', BASE/'generate.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    baseline_manifest = json.loads((BASE/'manifest_protected.json').read_text())
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA,
                    reverse=0, shape=[1, 32, 4096, 128], grid=[32, 32, 1],
                    block=[256, 1, 1], tiles=dict(BM=128, BN=128, BK=32),
                    gpu_tested=False, variants={})
    for group in GROUPS:
        text = variant(source, group)
        path = HERE/f'group{group}.ll'
        path.write_text(text)
        (HERE/f'group{group}.launch.json').write_bytes((BASE/'group_prefetch2_protected.launch.json').read_bytes())
        mapping = check_mapping(text, group)
        helper.assert_groups_unbroken(text.splitlines(), baseline_manifest['groups'])
        subprocess.run([LLVM/'llvm-as', path, '-o', '/dev/null'], check=True)
        optimized = HERE/f'group{group}.o3.ll'
        subprocess.run([LLVM/'opt', '-passes=default<O3>,verify', '-S', path, '-o', optimized], check=True)
        lines = optimized.read_text().splitlines()
        helper.assert_groups_unbroken(lines, baseline_manifest['groups'])
        defs = helper.definitions(lines)
        for record in baseline_manifest['moved_groups']:
            assert max(defs[v][0] for v in record['loads']) < defs[record['anchor']][0]
            assert 'sched.barrier(i32 0)' in lines[defs[record['anchor']][0]-1]
        assert text.count(helper.HINT) == 36
        assert sum(helper.READ in line for line in lines) == 80
        manifest['variants'][f'group{group}'] = dict(
            group_size=group, sha256=sha(path), mapping=mapping, llvm_verified=True,
            o3_all_32_groups_protected=True, o3_all_21_prefetch_anchors_preserved=True)
        print(f'PASS group{group}: all 1024 block mappings, LLVM/O3, all 32 read pairs and 21 prefetch anchors', flush=True)
    assert (HERE/'group4.ll').read_bytes() == SOURCE.read_bytes()
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
