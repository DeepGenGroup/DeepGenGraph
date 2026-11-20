#!/usr/bin/env python3
"""Check generic wide predicates, strict FP consumers and downstream codegen.

The CPU oracle removes only empty tied-register identities. It compares the
original and transformed LLVM functions, including high-word boundaries and
all signed/unsigned predicates; it is not a GPU timing/hazard simulation.
"""
import ctypes
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from check_register_lifetimes import normalize_metadata_ids


LLVM = Path('/data2/xsl/install/bin')


def run(args, text=None):
    result = subprocess.run(list(map(str, args)), input=text, text=True,
                            capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr[-8000:]
    return result.stdout


def function(name, predicate, count=8, opcode='or', side='rhs',
             attrs='"frisk.register_lifetimes"="1"'):
    lines = [f'define i32 @{name}(i64 %a, i64 %b, float %x, ptr %out) {attrs} {{']
    for i in range(count):
        a, b = '%a', '%b'
        if side in ('lhs', 'both'):
            lines.append(f'  %a{i} = {opcode} i64 %a, {i * 17}')
            a = f'%a{i}'
        if side in ('rhs', 'both'):
            lines.append(f'  %b{i} = {opcode} i64 %b, {i * 29}')
            b = f'%b{i}'
        lines.extend([
            f'  %c{i} = icmp {predicate} i64 {a}, {b}',
            f'  %m{i} = select i1 %c{i}, float 0.0, float 0xFFF0000000000000',
            f'  %f{i} = fadd float %x, %m{i}',
            f'  %p{i} = getelementptr float, ptr %out, i32 {i}',
            f'  store float %f{i}, ptr %p{i}',
            f'  %z{i} = zext i1 %c{i} to i32',
            f'  %s{i} = shl i32 %z{i}, {i}',
        ])
        if i:
            lines.append(f'  %r{i} = or i32 %r{i-1}, %s{i}')
        else:
            lines.append('  %r0 = or i32 %s0, 0')
    return '\n'.join(lines + [f'  ret i32 %r{count-1}', '}']) + '\n'


def body(text, name):
    return re.search(r'define [^@]*@' + name + r'\(.*?\n\}', text, re.S)[0]


def cpu_module(text):
    text = text.replace('amdgcn-amd-amdhsa', 'x86_64-unknown-linux-gnu')
    text = text.replace('amdgpu_kernel ', '')
    return re.sub(r'(?:tail )?call i32 asm sideeffect "", "=v,0(?:,i)?"'
                  r'\(i32 ([^,)]+)(?:, i32 \d+)?\)(?: #\d+)?',
                  r'add i32 \1, 0', text)


def main():
    tool = Path(sys.argv[1]).resolve()
    predicates = ('eq', 'ne', 'ult', 'ule', 'ugt', 'uge', 'slt', 'sle', 'sgt', 'sge')
    cases = [(f'{p}_{side}', p, side, 'xor' if side == 'both' else 'or')
             for p in predicates for side in ('lhs', 'rhs', 'both')]
    source = 'target triple = "amdgcn-amd-amdhsa"\n'
    for name, pred, side, opcode in cases:
        source += function(name, pred, opcode=opcode, side=side)
    for name, options in [
        ('single', dict(count=1)),
        ('carry', dict(opcode='add')),
        ('strict', dict(attrs='strictfp')),
        ('disabled', dict(attrs='noinline optnone')),
        # Already-exported IR must receive this new optimization without
        # repeating the unrelated address/math lifetime transformations.
        ('existing', dict(attrs='"frisk.register_lifetimes"="1"')),
        ('fresh', dict(attrs='')),
    ]:
        source += function(name, 'ugt', **options)
    source += function('uniform_kernel', 'ugt').replace(
        'define i32', 'define amdgpu_kernel void').replace('ret i32 %r7', 'ret void')
    output = run([tool, '--optimize-register-lifetimes', '-'], source)
    run([LLVM / 'opt', '-passes=verify', '-disable-output', '-'], output)
    repeated = run([tool, '--optimize-register-lifetimes', '-'], output)
    assert normalize_metadata_ids(repeated) == normalize_metadata_ids(output), \
        'comparison optimization is not idempotent'
    for name, _, _, _ in cases:
        section = body(output, name)
        assert 'compare.high.equal' in section, name
        assert not re.search(r'icmp \w+ i64', section), name
        assert section.count('fadd float') == 8, name
        assert 'fast ' not in section and 'nsz ' not in section, name
    for name in ('single', 'carry', 'strict', 'disabled', 'uniform_kernel'):
        assert 'compare.local' not in body(output, name), name
    assert 'compare.local' in body(output, 'existing')
    assert 'compare.local' in body(output, 'fresh')
    cpu = run([tool, '--optimize-register-lifetimes', '-'], cpu_module(source))
    assert 'compare.local' not in cpu, 'non-AMDGPU function changed'

    optimized = run([LLVM / 'opt', '-S', '-passes=default<O3>', '-'], output)
    run([LLVM / 'opt', '-passes=verify', '-disable-output', '-'], optimized)
    for name, _, _, _ in cases:
        assert 'compare.local.base' in body(optimized, name), name
        assert body(optimized, name).count('"=v,0,i"') == \
            body(output, name).count('"=v,0,i"'), name
        # InstCombine may express the SHARED high-word equality as a single
        # i64 xor/range test; per-element low-word comparisons must remain.
        assert len(re.findall(r'icmp \w+ i64', body(optimized, name))) <= 2, name
        assert len(re.findall(r'icmp \w+ i32', body(optimized, name))) >= 8, name

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        libraries = []
        for name, ir in [('before', source), ('after', output), ('o3', optimized)]:
            ll, obj, so = (path / (name + suffix) for suffix in ('.ll', '.o', '.so'))
            ll.write_text(cpu_module(ir))
            run([LLVM / 'llc', '-O3', '-filetype=obj', '-relocation-model=pic', ll, '-o', obj])
            run(['cc', '-shared', obj, '-o', so])
            libraries.append(ctypes.CDLL(str(so)))
        # Exhaust the low-byte domain, and test both sides of word/sign
        # boundaries plus arbitrary full-width values. No narrow-range premise.
        edges = [0, 1, 127, 255, 256, (1 << 31)-1, 1 << 31,
                 (1 << 32)-1, 1 << 32, (1 << 32)+1,
                 (1 << 63)-1, 1 << 63, (1 << 64)-1]
        pairs = [(a, b) for a in edges for b in edges]
        pairs += [(a, b) for a in range(256) for b in (a, a-1, a+1)]
        rng = random.Random(216)
        pairs += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(1000)]
        inputs = (0.0, -0.0, 1.25, -3.5, float('inf'), -float('inf'), float('nan'))
        count = 0
        for name, _, _, _ in cases:
            funcs = [getattr(lib, name) for lib in libraries]
            for func in funcs:
                func.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_float,
                                 ctypes.POINTER(ctypes.c_float)]
                func.restype = ctypes.c_uint32
            for a, b in pairs:
                for x in inputs:
                    results = []
                    for func in funcs:
                        array = (ctypes.c_float * 8)()
                        mask = func(a, b, x, array)
                        bits = bytes(array)
                        results.append((mask, bits))
                    assert results[0] == results[1] == results[2], (name, a, b, x, results)
                    count += 1
        print(f'PASS {count} full-width predicate/FP cases before, after and after O3; '
              'signed zero, NaN, infinities and word/sign boundaries')
        # Verify AMDGPU accepts all register constraints, separately from the
        # CPU semantic oracle. gfx900 is only a local backend legality check.
        gpu = path / 'gpu.ll'
        gpu.write_text(optimized)
        assembly = path / 'gpu.s'
        run([LLVM / 'llc', '-O3', '-mcpu=gfx900', gpu, '-o', assembly])
        assert 'v_cmp' in assembly.read_text()
    print('PASS all ten integer predicates, OR/XOR on either/both operands, '
          'fallbacks, existing exports, idempotence, O3 survival and gfx900 codegen')


if __name__ == '__main__':
    main()
