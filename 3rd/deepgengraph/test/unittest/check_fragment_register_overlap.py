#!/usr/bin/env python3
"""Register reductions overlap MMAs without disturbing a Fragment LDS plan."""
from collections import Counter
import hashlib
from pathlib import Path
import re
import subprocess
import sys

TOOL = (Path(sys.argv[1]) if len(sys.argv) > 1 else
        Path(__file__).resolve().parents[2] / 'build/test/ThreadTilingTest')
MMA = 'v_mmac_f32_16x16x16_f16 $0, $1, $2, $3'


def run(source, reorder=True, *extra):
    p = subprocess.run([str(TOOL), '--verify-each',
                        '--ir-deep-optimize=reorder-after-unroll=' + str(reorder).lower(),
                        *extra], input=source, text=True, capture_output=True, timeout=180)
    assert p.returncode == 0, p.stderr
    return p.stdout


def fixture(memory=True, plan=True, boundary='', opaque=False, attrs=''):
    marker = ', frisk.fragment_lds_prefetch = "8:4"' if plan else ''
    lines = ['module {', 'func.func private @opaque()',
             'func.func @kernel(%sm: memref<64xf16, 3>, %a: vector<4xf16>, '
             '%b: vector<4xf16>, %c: vector<4xf32>, %cond: i1) -> f32 '
             'attributes {thread_num = 64 : i32' + marker + attrs + '} {',
             '%seed = arith.constant 0xFF800000 : f32']
    for i in range(8):
        if memory:
            lines += [f'%i{i} = arith.constant {i * 4} : index',
                      f'%b{i} = vector.load %sm[%i{i}] {{test.id = "load{i}"}} '
                      ': memref<64xf16, 3>, vector<4xf16>']
        # Anchors protect reads, but must not isolate already-ready FP work.
        anchor = ', frisk.lds_prefetch_anchor' if i < 4 else ''
        instruction = MMA + ('\\0As_waitcnt lgkmcnt(0)' if opaque else '')
        lines.append(f'%m{i} = llvm.inline_asm has_side_effects '
                     f'{{test.id = "mma{i}"{anchor}}} "{instruction}", "=v,v,v,0" '
                     f'%a, {f"%b{i}" if memory else "%b"}, %c : '
                     '(vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>')
    if boundary:
        lines.append(boundary)
    prev = '%seed'
    for i in range(8):
        for lane in range(4):
            n = i * 4 + lane
            lines += [f'%x{n} = vector.extract %m{i}[{lane}] : f32 from vector<4xf32>',
                      f'%r{n} = arith.maxnumf {prev}, %x{n} '
                      f'{{frisk.try_multichain, test.id = "max{n}"}} : f32']
            prev = f'%r{n}'
    return '\n'.join(lines + [f'return {prev} : f32', '}', '}'])


def ids(text):
    return re.findall(r'test.id = "([^"]+)"', text)


def spine(text):
    return [x for x in ids(text) if x.startswith(('load', 'mma'))]


def graph(text):
    """Fingerprint every SSA definition and the return, retaining operand order."""
    values, operations, returns = {}, Counter(), []
    for line in text.splitlines():
        if 'func.func @kernel(' in line:
            for i, arg in enumerate(re.findall(r'(%\w+):', line)):
                values[arg] = f'arg{i}'
        match = re.match(r'\s*(%\w+(?:, %\w+)*) = (.*)', line)
        if match:
            rhs = re.sub(r'%\w+', lambda m: values[m[0]], match[2])
            digest = hashlib.sha256(rhs.encode()).hexdigest()
            results = match[1].split(', ')
            for i, name in enumerate(results):
                values[name] = digest if len(results) == 1 else f'{digest}:{i}'
            # Coalescing can leave dead address constants until the next run.
            if not match[2].startswith('arith.constant '):
                operations[digest] += 1
        if line.strip().startswith('return '):
            returns.append(re.sub(r'%\w+', lambda m: values[m[0]], line.strip()))
    return operations, returns


def positions(text):
    mmas, result = 0, []
    for line in text.splitlines():
        if 'llvm.inline_asm' in line and 'v_mmac_' in line:
            mmas += 1
        if 'arith.maxnumf ' in line:
            result.append(mmas)
    return result


def main():
    for memory in (False, True):
        source = fixture(memory=memory)
        before, after = run(source, False), run(source)
        assert graph(before) == graph(after), 'changed FP tree, MMA operands or memory access'
        assert spine(before) == spine(after), 'changed planned load/MMA issue slots'
        assert min(positions(before)) == 8
        assert min(positions(after)) < 4, 'ready reduction cannot cross a logical LDS anchor'
        assert len(positions(before)) == len(positions(after)) == 32
        replay = run(after)
        assert graph(replay) == graph(after)
        assert spine(replay) == spine(after) and positions(replay) == positions(after)
        # The existing multichain pass must retain the recovered overlap.
        reduced = run(source, True, '--frisk-thread-local-reduction=approximate=true')
        assert min(positions(reduced)) < 4 and len(positions(reduced)) == 32
        print('PASS planned', 'memory/MMA window' if memory else 'register-only tail',
              'first max after MMA', min(positions(after)))

    # No plan: retain the legacy no-memory-window behavior. Opt-out and CP-SAT
    # schedules also remain untouched.
    for source in (fixture(memory=False, plan=False),
                   fixture(attrs=', cpsat.preserve_schedule')):
        assert run(source) == run(source, False)

    # Actual effect/control/collective boundaries still prevent speculation.
    for boundary in (
            'gpu.barrier {test.id = "boundary"}',
            'func.call @opaque() {test.id = "boundary"} : () -> ()',
            'llvm.inline_asm has_side_effects {test.id = "boundary"} "s_nop 0", "" : () -> ()',
            'scf.if %cond { func.call @opaque() : () -> () } {test.id = "boundary"}',
            '%v, %valid = gpu.shuffle xor %seed, %lane, %width : f32'):
        source = fixture(boundary=boundary)
        if 'gpu.shuffle' in boundary:
            source = source.replace('%seed =', '%lane = arith.constant 1 : i32\n'
                                    '%width = arith.constant 64 : i32\n%seed =')
            source = source.replace('return %r31 : f32',
                                    '%out = arith.maxnumf %r31, %v : f32\nreturn %out : f32')
        before, after = run(source, False), run(source)
        assert graph(before) == graph(after)
        assert min(positions(after)) == 8, boundary
        assert spine(before) == spine(after)
    print('PASS real synchronization, collective, call and control boundaries')

    # A marker does not grant purity to inline asm with extra instructions.
    source = fixture(opaque=True)
    assert run(source) == run(source, False)
    # All arithmetic remains bit-for-bit the same SSA graph; no reassociation,
    # fast-math flags, waits, instructions or register-MMA padding are added.
    print('PASS FP/MMA operand graphs, read issue slots, replay and opt-outs')


if __name__ == '__main__':
    main()
