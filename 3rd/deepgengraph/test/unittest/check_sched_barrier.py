#!/usr/bin/env python3
"""Frisk scheduling boundaries: masks, effects, motion, lowering and LLVM O3."""
from pathlib import Path
import re
import subprocess

from check_mma_interleave import fixture as qk_fixture
from check_thread_fragment_schedules import softmax

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / 'build/test'
LLVM = Path('/data2/xsl/install/bin')


def run(source, *passes, ok=True):
    p = subprocess.run([str(BIN / 'ThreadTilingTest'), '--verify-each', *passes],
                       input=source, text=True, capture_output=True, timeout=120)
    assert (p.returncode == 0) == ok, p.stderr
    return p.stdout if ok else p.stderr


def main():
    masks = [0] + [1 << bit for bit in range(12)] + [0x006, 0x402, 0xfff]
    body = '\n'.join(f'frisk.sched_barrier {{mask = {mask} : i32, test.id = "b{i}"}}'
                     for i, mask in enumerate(masks))
    source = 'module { func.func @kernel() {\n' + body + '\nreturn\n}}'
    clean = run(source, '--canonicalize', '--cse')
    assert clean.count('frisk.sched_barrier') == len(masks)
    assert run(clean) == clean
    # Equal adjacent guards still have effects and must not be CSE'd away.
    default = 'module { func.func @kernel() { frisk.sched_barrier\nfrisk.sched_barrier\nreturn } }'
    assert run(default, '--canonicalize', '--cse').count('frisk.sched_barrier') == 2
    for value in (-1, 0x1000, 0x80000000):
        invalid = source.replace('mask = 0 : i32', f'mask = {value} : i32', 1)
        assert 'supported bits' in run(invalid, ok=False)
    run(source.replace('mask = 0 : i32', 'mask = 0 : i64', 1), ok=False)

    # The standalone lowering and the production lowering use the same contract.
    direct = run(source, '--frisk-lower-sched-barrier')
    assert 'frisk.sched_barrier' not in direct
    assert direct.count('llvm.call @llvm.amdgcn.sched.barrier') == len(masks)
    assert direct.count('llvm.func @llvm.amdgcn.sched.barrier') == 1
    assert all(f'test.id = "b{i}"' in direct for i in range(len(masks)))
    assert run(direct, '--frisk-lower-sched-barrier') == direct
    for declaration in ('llvm.func @llvm.amdgcn.sched.barrier(i64)',
                        'func.func private @llvm.amdgcn.sched.barrier(i32)',
                        'llvm.func @llvm.amdgcn.sched.barrier(%x: i32) { llvm.return }'):
        conflict = source.replace('module {', 'module { ' + declaration)
        assert 'conflicting declaration' in run(conflict, '--frisk-lower-sched-barrier', ok=False)
    existing = source.replace('module {', 'module { llvm.func @llvm.amdgcn.sched.barrier(i32)')
    assert run(existing, '--frisk-lower-sched-barrier').count('llvm.func @llvm.amdgcn.sched.barrier') == 1
    assert 'ROCm target' in run(source, '--test-async-to-llvm=cuda=true', ok=False)

    low = run(source, '--test-async-to-llvm')
    exported = subprocess.run([str(LLVM / 'mlir-translate'), '--mlir-to-llvmir'],
                              input=low, text=True, capture_output=True, check=True).stdout
    optimized = subprocess.run([str(LLVM / 'opt'), '-S', '-passes=default<O3>,verify', '-o', '-'],
                               input=exported, text=True, capture_output=True, check=True).stdout
    for ir in (exported, optimized):
        actual = [int(x) for x in re.findall(r'call void @llvm.amdgcn.sched.barrier\(i32 (\d+)\)', ir)]
        assert actual == masks, actual
        assert 'memory(inaccessiblemem: readwrite)' in ir
        assert '@llvm.amdgcn.s.barrier(' not in ir
        assert 'waitcnt' not in ir

    # Frisk schedulers cannot infer selected instruction classes from the mask.
    # Conservatively retain both zero-mask and nonzero-mask boundaries.
    for mask in (0, 0x006, 0x402, 0xfff):
        boundary = f'frisk.sched_barrier {{mask = {mask} : i32}}'
        qk = qk_fixture().replace('// boundary', boundary)
        assert run(qk, '--frisk-thread-interleave-qk=width=2') == run(qk)
        scores = softmax(boundary=boundary)
        assert run(scores, '--frisk-thread-schedule-softmax-fragments=mode=pv') == run(scores)

    print('PASS sched_barrier masks, verification, CSE retention, scheduling boundaries, '
          'symbol conflicts, target rejection, lowering and LLVM/O3 preservation')


if __name__ == '__main__':
    main()
