#!/usr/bin/env python3
"""Trace the user's BM=BN=128/BK=32 export through mask and direct P -> PV.

Usage: check_wmma_b_permute_export.py build/wmma_layout_modes/3.3.ll
The complete QK reduction is injected as distinct logical-coordinate tags;
all subsequent mask coordinates and paired PV A/B operands are evaluated
from exported LLVM instructions, including V's physical LDS packing.
No GPU timing claims are made.
"""
import re
import sys
from pathlib import Path

from check_direct_shared_copy import Trace
from check_gemm_pv_addresses import split_args


def conditional_phis(text, defs):
    """Resolve diamond PHIs, allowing unconditional blocks on either arm."""
    blocks = dict(re.findall(r'^([\w.]+):[^\n]*\n(.*?)(?=^[\w.]+:|^\})', text, re.M | re.S))
    jumps = {label: re.findall(r'^  br label %([\w.]+)', body, re.M)
             for label, body in blocks.items()}
    def reaches(start, end):
        visited = set()
        while start not in visited:
            if start == end:
                return True
            visited.add(start)
            following = jumps.get(start, [])
            if len(following) != 1:
                return False
            start = following[0]
        return False
    choices = {}
    for name, expression in defs.items():
        if not expression.startswith(('phi [', 'phi <')):
            continue
        incoming = re.findall(r'\[ (%[\w.]+|zeroinitializer), %([\w.]+) \]', expression)
        if len(incoming) != 2:
            continue
        for cond, yes, no in re.findall(r'br i1 (%[\w.]+), label %([\w.]+), label %([\w.]+)', text):
            for a, b in [incoming, incoming[::-1]]:
                if reaches(yes, a[1]) and reaches(no, b[1]):
                    choices[name] = cond, a[0], b[0]
    return choices


def check_q_copy(text):
    """Check the producer ownership permutation against logical Q coordinates."""
    defs = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    threads = int(re.search(r'"amdgpu-flat-work-group-size"="\d+,(\d+)"', text)[1])
    tid = next(n for n, e in defs.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    args = re.findall(r'ptr addrspace\(1\) (%[\w.]+)',
                      re.search(r'define amdgpu_kernel void @Attn_p2\((.*)\) #', text)[1])
    ivs = [n for n, e in defs.items() if e.startswith('phi i64 ')]
    stores = re.findall(r'^  store <(\d+) x half> (%[\w.]+), ptr addrspace\(3\) (%[\w.]+),',
                        text[:text.index('v_mmac_')], re.M)
    for head, query, dk in [(0, 0, 0), (7, 31, 3)]:
        context = {n: 0 for n, e in defs.items() if e.startswith('phi i64 ') or
                   '@llvm.amdgcn.workgroup.id.' in e}
        context.update({arg: (arg, 0) for arg in args})
        for axis, value in [('x', head), ('y', query)]:
            context[next(n for n, e in defs.items()
                         if f'@llvm.amdgcn.workgroup.id.{axis}()' in e)] = value
        context[ivs[-1]] = dk
        memory = {}
        for thread in range(threads):
            lane = Trace(defs, {**context, tid: thread})
            for count, value, pointer in stores:
                data = lane.value(value)
                if data[0][0] != args[0]:
                    continue
                base, offset = lane.value(pointer)
                for i, source in enumerate(data):
                    address = base, offset + 2 * i
                    assert address not in memory, ('duplicate Q write', address)
                    memory[address] = source
        assert len(memory) == 128 * 32
        base = min(offset for _, offset in memory)
        origin = head * 4096 * 128 + query * 128 * 128 + dk * 32
        covered = set()
        for (_, address), (_, source) in memory.items():
            row, col = divmod(source // 2 - origin, 128)
            assert 0 <= row < 128 and 0 <= col < 32
            assert address - base == 2 * ((col // 4 * 128 + row) * 4 + col % 4)
            covered.add((row, col))
        assert len(covered) == 128 * 32
    print(f'PASS Q copy: all {threads} threads, complete unique packed/global coordinates')


def check(text, head=0, query_block=0, key_block=0, output_part=0):
    defs = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    tid = next(n for n, e in defs.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    threads = int(re.search(r'"amdgpu-flat-work-group-size"="\d+,(\d+)"', text)[1])
    assert threads in (128, 256)
    row_stride = threads // 64 * 16
    row_repeats = 128 // row_stride
    qk = [n for n, e in defs.items() if e.startswith(f'phi [{row_repeats} x <32 x float>]')]
    mask_choices = conditional_phis(text, defs)
    qk = [name for name in qk if name not in mask_choices]
    mma = [n for n, e in defs.items() if 'asm sideeffect ' in e and 'v_mmac_' in e]
    exp = [n for n, e in defs.items() if '@__ocml_exp2_f32(' in e or '@llvm.amdgcn.exp2.f32(' in e]
    mma_per_tile = row_repeats * 16
    parts = len(mma) // mma_per_tile - 1
    assert len(qk) == 1 and parts in (1, 2, 4) and len(exp) == 16384 // threads
    common = {n: 0 for n, e in defs.items() if e.startswith('phi i64 ') or
              '@llvm.amdgcn.workgroup.id.' in e}
    for axis, value in [('x', head), ('y', query_block)]:
        common[next(n for n, e in defs.items()
                    if f'@llvm.amdgcn.workgroup.id.{axis}()' in e)] = value
    ivs = [n for n, e in defs.items() if e.startswith('phi i64 ')]
    sequence_iv = ivs[-2]
    common[sequence_iv] = key_block
    if parts < 4:
        assert len(ivs) == 3
        common[ivs[0]] = output_part
    args = re.findall(r'ptr addrspace\(1\) (%[\w.]+)',
                      re.search(r'define amdgpu_kernel void @Attn_p2\((.*)\) #', text)[1])
    common.update({arg: (arg, 0) for arg in args})

    dependencies = {}
    def depends_on_qk(name):
        if name == qk[0]:
            return True
        if name not in dependencies:
            expression = defs.get(name, '')
            dependencies[name] = False
            if name in mask_choices:
                dependencies[name] = any(depends_on_qk(n) for n in mask_choices[name][1:])
            elif not expression.startswith('phi '):
                dependencies[name] = any(depends_on_qk(n)
                                         for n in re.findall(r'%[\w.]+', expression))
        return dependencies[name]

    class Lane(Trace):
        mask_only = False
        def value(self, name):
            if name in ('0.000000e+00', '0xFFF0000000000000'):
                return ('masked', name != '0.000000e+00') if self.mask_only else 0
            if name not in self.cache and name in self.definitions:
                expression = self.definitions[name]
                if name in mask_choices:
                    condition, yes, no = mask_choices[name]
                    selected = yes if self.value(condition) else no
                    if selected == 'zeroinitializer':
                        zero = ('masked', False) if self.mask_only else 0
                        self.cache[name] = [[zero for _ in range(32)] for _ in range(row_repeats)]
                    else:
                        self.cache[name] = self.value(selected)
                elif expression.startswith('icmp '):
                    pred, _, body = expression[5:].split(' ', 2)
                    a, b = map(self.value, split_args(body))
                    self.cache[name] = {'eq': a == b, 'ne': a != b,
                        'ult': a < b, 'ule': a <= b, 'ugt': a > b, 'uge': a >= b,
                        'slt': a < b, 'sle': a <= b, 'sgt': a > b, 'sge': a >= b}[pred]
                elif expression.startswith('fadd '):
                    operands = split_args(expression.split(' ', 1)[1])
                    names = [operand.split()[-1] for operand in operands]
                    # InstCombine may commute QK + mask. Follow the QK
                    # dependency for coordinates and the other side for mask.
                    side = next(i for i, operand in enumerate(names) if depends_on_qk(operand))
                    self.cache[name] = self.value(names[1 - side if self.mask_only else side])
                elif self.mask_only and re.match(r'select i1 %[\w.]+, float ', expression):
                    condition, yes, no = split_args(expression[len('select '):])
                    assert {yes, no} == {'float 0.000000e+00', 'float 0xFFF0000000000000'}
                    selected = yes if self.typed(condition) else no
                    self.cache[name] = ('masked', selected.endswith('0xFFF0000000000000'))
            return super().value(name)

    def logical_col(thread, reg):
        physical = reg // 4 * 16 + thread % 64 // 16 + reg % 4 * 4
        return (physical // 64 * 64 + physical % 4 * 16 +
                physical // 16 % 4 * 4 + physical // 4 % 4)

    peers, masks = [], []
    for thread in range(threads):
        tags = [[(r * row_stride + thread // 64 * 16 + thread % 16,
                  logical_col(thread, j)) for j in range(32)] for r in range(row_repeats)]
        peers.append(Lane(defs, {**common, tid: thread, qk[0]: tags}, peers))
        masks.append(Lane(defs, {**common, tid: thread, qk[0]: tags}, masks))
        masks[-1].mask_only = True
    for lane, mask in zip(peers, masks):
        for condition, _, _ in mask_choices.values():
            assert bool(mask.value(condition)) == (query_block > key_block)
        for name in exp:
            tag, (row, col) = lane.value(name)
            assert tag == 'exp2'
            assert mask.value(name) == ('exp2', ('masked',
                query_block * 128 + row < key_block * 128 + col)), (row, col, name)
    print('PASS exported causal mask: all 16384 permuted P coordinates')

    def reduction_index(k):
        return k // 64 * 64 + k // 4 % 4 * 16 + k // 16 % 4 * 4 + k % 4

    start = text.index('  ' + exp[0] + ' =')
    # Pairing follows the permuted logical K stride, retaining wide V packets.
    assert len(re.findall(r'load <8 x half>, ptr addrspace\(3\)', text[start:])) == row_repeats * 8 * parts
    covered = set()
    for part in range(parts):
        instructions = mma[(1 + part) * mma_per_tile:(2 + part) * mma_per_tile]
        end = text.index('  ' + instructions[0] + ' =')
        stores = re.findall(r'^  store (half|<\d+ x half>) (%[\w.]+), ptr addrspace\(3\) (%[\w.]+),',
                            text[start:end], re.M)
        memory = {}
        for lane in peers:
            for ty, value, ptr in stores:
                values = lane.value(value)
                if ty == 'half':
                    values = [values]
                assert all(tag[0] == args[2] for tag in values), 'P still written to LDS'
                base, offset = lane.value(ptr)
                for j, tag in enumerate(values):
                    address = base, offset + 2 * j
                    assert address not in memory, ('duplicate V write', address)
                    memory[address] = tag
        assert len(memory) == 128 * 32
        for thread, lane in enumerate(peers):
            for i, name in enumerate(instructions):
                a, b, _ = split_args(re.search(r'"\((.*)\)', defs[name])[1])
                row = i // 16 * row_stride + thread // 64 * 16 + thread % 16
                k = i % 8 * 16 + thread % 64 // 16 * 4
                ks = [reduction_index(k + j) for j in range(4)]
                actual = lane.typed(a)
                expected = [('f16', ('exp2', (row, kk))) for kk in ks]
                assert actual == expected, (thread, part, i, actual, expected)
                covered.update((row, kk) for kk in ks)
                col = (output_part * parts + part) * 32 + i // 8 % 2 * 16 + thread % 16
                actual_b = [memory[address] for address in lane.typed(b)]
                expected_b = [(args[2], 2 * (head * 4096 * 128 +
                    (key_block * 128 + kk) * 128 + col)) for kk in ks]
                assert actual_b == expected_b, (thread, part, i, actual_b, expected_b)
        start = text.index('  ' + instructions[-1] + ' =')
    assert covered == {(r, c) for r in range(128) for c in range(128)}
    assert int(re.search(r'@shm_\w+ = .*global \[(\d+) x i8\]', text)[1]) == 16384
    print(f'PASS no P LDS exchange: 16384 coordinates, {parts} PV consumers, paired V rows')


if __name__ == '__main__':
    text = Path(sys.argv[1]).read_text()
    check_q_copy(text)
    for context in [(0, 0, 0), (7, 31, 0), (7, 31, 30), (7, 31, 31),
                    (31, 31, 31)]:
        print('Checking head/query block/key block:', context)
        check(text, *context)
    if len(re.findall(r'phi i64 ', text)) == 3:
        print('Checking nonzero outer output slice')
        check(text, 7, 31, 30, output_part=1)
