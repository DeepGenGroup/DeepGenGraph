#!/usr/bin/env python3
"""Execute the exported mask/add slices on CPU, preserving original FP ops.

GPU QK arithmetic is replaced by arbitrary FP32 scores. All mask coordinates,
branch conditions and additions are taken from the actual candidate LLVM IR.
"""
import ctypes
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from check_wmma_b_permute_export import conditional_phis


def extract(text, function):
    defs = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    choices = conditional_phis(text, defs)
    qk = [n for n, e in defs.items() if e.startswith('phi [2 x <32 x float>]') and n not in choices]
    assert len(qk) == 1
    roots = {qk[0]: '%scores'}
    for n, e in defs.items():
        for intrinsic, arg in [('workitem.id.x', 'tid'), ('workgroup.id.x', 'head'),
                               ('workgroup.id.y', 'query'), ('workgroup.id.z', 'batch')]:
            if f'@llvm.amdgcn.{intrinsic}()' in e:
                roots[n] = '0' if arg == 'batch' else '%' + arg
    ivs = [n for n, e in defs.items() if e.startswith('phi i64 ')]
    assert len(ivs) == 2, 'full jam fixture expected'
    roots[ivs[0]], roots[ivs[1]] = '%key', '0'
    inputs = re.findall(r'call float @(?:__ocml_exp2_f32|llvm.amdgcn.exp2.f32)\(float (%[\w.]+)\)', text)
    assert len(inputs) == 64
    lines = ['  %scores = load [2 x <32 x float>], ptr %input, align 4']
    active = set()

    def visit(name):
        if name in roots:
            return roots[name]
        if name not in defs:
            return name
        assert name not in active, ('unexpected dependency cycle', name)
        active.add(name)
        expr = defs[name]
        out = '%slice' + str(len(roots))
        if name in choices:
            cond, yes, no = map(visit, choices[name])
            out = '%slice' + str(len(roots))
            # LLVM select does not accept arrays; select the vector members.
            if expr.startswith('phi <'):
                ty = re.match(r'phi (<[^>]+>)', expr)[1]
                lines.append(f'  {out} = select i1 {cond}, {ty} {yes}, {ty} {no}')
            else:
                for i in range(2):
                    lines.extend([
                        f'  {out}a{i} = extractvalue [2 x <32 x float>] {yes}, {i}',
                        f'  {out}b{i} = extractvalue [2 x <32 x float>] {no}, {i}',
                        f'  {out}s{i} = select i1 {cond}, <32 x float> {out}a{i}, <32 x float> {out}b{i}',
                        f'  {out}p{i} = insertvalue [2 x <32 x float>] '
                        f'{"poison" if i == 0 else out + "p0"}, <32 x float> {out}s{i}, {i}'])
                out += 'p1'
        elif 'asm sideeffect "", "=v,0"' in expr:
            # Tied-register identity has no arithmetic effect.
            out = visit(re.search(r'"\(\w+ (%[\w.]+)\)', expr)[1])
        else:
            assert expr.split()[0] in {'add', 'sub', 'mul', 'sdiv', 'udiv', 'srem', 'urem',
                'and', 'or', 'xor', 'shl', 'lshr', 'ashr', 'icmp', 'select', 'zext', 'sext',
                'trunc', 'fadd', 'extractvalue', 'insertvalue', 'extractelement',
                'insertelement', 'shufflevector'}, (name, expr)
            expr = re.sub(r'%[\w.]+', lambda m: visit(m[0]), expr)
            expr = re.sub(r', ![\w.]+ !\d+', '', expr)
            # Allocate after recursion to avoid overlapping SSA names.
            out = '%slice' + str(len(roots))
            lines.append(f'  {out} = {expr}')
        roots[name] = out
        active.remove(name)
        return out

    for i, value in enumerate(inputs):
        result = visit(value)
        lines.extend([f'  %dest{i} = getelementptr float, ptr %output, i32 {i}',
                      f'  store float {result}, ptr %dest{i}, align 4'])
    return (f'define void @{function}(ptr %input, ptr %output, i32 %tid, i32 %head, '
            'i32 %query, i64 %key) {\n' + '\n'.join(lines) + '\n  ret void\n}\n')


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    assert len(paths) >= 2
    ir = '\n'.join(extract(path.read_text(), f'mask{i}') for i, path in enumerate(paths))
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        path, obj, library = directory / 'slice.ll', directory / 'slice.o', directory / 'slice.so'
        path.write_text(ir)
        llvm = Path('/data2/xsl/install/bin')
        subprocess.run([str(llvm / 'opt'), '-passes=verify', '-disable-output', str(path)], check=True)
        subprocess.run([str(llvm / 'llc'), '-filetype=obj', '-relocation-model=pic',
                        str(path), '-o', str(obj)], check=True)
        subprocess.run(['cc', '-shared', str(obj), '-o', str(library)], check=True)
        lib = ctypes.CDLL(str(library))
        functions = [getattr(lib, f'mask{i}') for i in range(len(paths))]
        for fn in functions:
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                           ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64]
            fn.restype = None
        edge = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345,
                0x3f800000, 0xbf800000, 1, 0x80000001, 0x7f7fffff]
        def nan(bits):
            return bits & 0x7f800000 == 0x7f800000 and bits & 0x7fffff
        comparisons = 0
        for head, query, key in [(0, 0, 0), (7, 31, 0), (7, 31, 30), (7, 31, 31), (31, 15, 15)]:
            for tid in range(256):
                values = (ctypes.c_uint32 * 64)(*(edge[(j + tid) % len(edge)] for j in range(64)))
                expected = (ctypes.c_uint32 * 64)()
                functions[0](values, expected, tid, head, query, key)
                for index, fn in enumerate(functions[1:], 1):
                    actual = (ctypes.c_uint32 * 64)()
                    fn(values, actual, tid, head, query, key)
                    for j, (a, b) in enumerate(zip(expected, actual)):
                        assert a == b or (nan(a) and nan(b)), (paths[index], head, query, key, tid, j, hex(a), hex(b))
                        comparisons += 1
        print(f'PASS {comparisons} exported mask/add FP32 comparisons, all 256 lanes: '
              'diagonal/clear tiles, signed zero, infinities, NaNs, subnormals and finite extremes')


if __name__ == '__main__':
    main()
