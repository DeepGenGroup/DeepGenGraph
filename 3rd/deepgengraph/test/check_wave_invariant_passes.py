#!/usr/bin/env python3
"""frisk-simplify-bounded-affine, frisk-hoist-wave-invariants, elideBackedgeBarrier.

Covers what each may rewrite, what it must refuse, and, for the affine pass,
brute-force equality of every rewritten map against the original.
"""
import importlib.util
from itertools import product
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / 'build/test'
OPT_TOOL = BUILD / 'ThreadTilingTest'
EXPORT_TOOL = BUILD / 'LegacyLLVMExportTest'
LLVM_OPT = Path('/data2/xsl/install/bin/opt')
spec = importlib.util.spec_from_file_location('cross_bn', HERE / 'check_cross_bn_k0.py')
cross_bn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross_bn)


def run_pass(text, *flags):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'input.mlir'
        path.write_text(text)
        result = subprocess.run([OPT_TOOL, *flags, path], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout, result.stderr


AFFINE = '''func.func @f(%s0: index, %s1: index, %other: index) {{
  affine.for %i = {low} to {high} {{
    %x = affine.apply affine_map<{map}>({dims})[{symbols}]
    "test.use"(%x) : (index) -> ()
  }}
  return
}}
'''


def evaluate(map_text, env):
    """Evaluate an affine result, with positive divisors, using Euclidean // and %."""
    python = re.sub(r'\bfloordiv\b', '//', map_text)
    python = re.sub(r'\bmod\b', '%', python)
    return eval(python, {'__builtins__': {}}, env)  # map_text comes from this test's tool output.


class BoundedAffine(unittest.TestCase):
    def simplify(self, map_text, low=0, high=32, dims='%i', symbols='%s0, %s1'):
        output, _ = run_pass(AFFINE.format(low=low, high=high, map=map_text, dims=dims, symbols=symbols),
                             '--allow-unregistered-dialect', '--frisk-simplify-bounded-affine')
        applied = re.search(r'affine\.apply (?:affine_map<)?(.*?)>?\(', output)
        if '#map' in output:
            alias = re.search(r'#map\d* = affine_map<(.*)>', output)
            return alias[1]
        return applied[1]

    def result(self, printed):
        return printed.split('->', 1)[1].strip()[1:-1]

    def test_real_kernel_map_folds_and_stays_equal(self):
        original = '(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + {r}) floordiv 4096) mod 32)'
        for r in (0, 1024, 2048, 3072):
            with self.subTest(r=r):
                source = original.format(r=r)
                printed = self.simplify(source)
                self.assertNotIn('floordiv', printed)
                self.assertNotIn('s1', self.result(printed))
                new = self.result(printed)
                old = source.split('->', 1)[1].strip()[1:-1]
                for d0, s0, s1 in product(range(32), range(-70, 70, 3), range(-70, 70, 7)):
                    env = dict(d0=d0, s0=s0, s1=s1)
                    self.assertEqual(evaluate(old, env), evaluate(new, env), env)

    def test_stays_equal_on_mixed_shapes(self):
        for source in ['(d0)[s0, s1] -> ((d0 * 2 + s0 * 8 + 1) floordiv 4)',
                       '(d0)[s0, s1] -> ((d0 + s0 * 64 + 63) floordiv 64)',
                       '(d0)[s0, s1] -> ((d0 + s0 * 16) mod 16)',
                       '(d0)[s0, s1] -> ((d0 + 64 * s1 - 8) floordiv 64)',
                       '(d0)[s0, s1] -> (((d0 + s0 * 128 + 32) floordiv 128) mod 8 + s1)']:
            with self.subTest(source=source):
                new = self.result(self.simplify(source, high=8))
                old = source.split('->', 1)[1].strip()[1:-1]
                for d0, s0, s1 in product(range(8), range(-40, 40, 3), range(-40, 40, 7)):
                    env = dict(d0=d0, s0=s0, s1=s1)
                    self.assertEqual(evaluate(old, env), evaluate(new, env), (source, new, env))

    def test_refuses_when_not_proven(self):
        for source, high in [
                # The remainder crosses a multiple of the divisor inside the loop.
                ('(d0)[s0, s1] -> ((d0 + s0 * 4096 + 4090) floordiv 4096)', 32),
                # A symbol with a nonzero remainder has no bound.
                ('(d0)[s0, s1] -> ((d0 + s0 * 3) floordiv 4096)', 32),
                # No loop operand: the LLVM stage proves addresses from this form.
                ('(d0)[s0, s1] -> ((s0 mod 16 + 16) floordiv 32)', 32)]:
            with self.subTest(source=source):
                printed = self.simplify(source, high=high)
                self.assertIn('floordiv', printed)

    def test_non_loop_operand_has_no_bound(self):
        # %other is passed in the IV's position: the same map must not fold.
        printed = self.simplify('(d0)[s0, s1] -> ((d0 + s0 * 4096 + 1024) floordiv 4096)', dims='%other')
        self.assertIn('floordiv', printed)


UNIFORM = 'llvm.icmp "slt" %iv, %bound : i64'
LOOP = '''llvm.mlir.global internal @shm() {{addr_space = 3 : i32}} : !llvm.array<64 x i8>
llvm.func @use(i32)
llvm.func @kernel(%src: !llvm.ptr<1>, %tid: i32) {{
  %zero = llvm.mlir.constant(0 : i32) : i32
  %ones = llvm.mlir.constant(-1 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %start = llvm.mlir.constant(0 : i64) : i64
  %one = llvm.mlir.constant(1 : i64) : i64
  %bound = llvm.mlir.constant(4 : i64) : i64
  %shm = llvm.mlir.addressof @shm : !llvm.ptr<3>
  %base = llvm.ptrtoint %shm : !llvm.ptr<3> to i32
  llvm.br ^header(%start : i64)
^header(%iv: i64):
  %more = {condition}
  llvm.cond_br %more, ^body, ^exit
^body:
  %leader = {leader}
  %pointer = llvm.inttoptr %leader : i32 to !llvm.ptr<3>
  rocdl.global.load.async.lds %src, %pointer, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
  %lo = rocdl.mbcnt.lo %ones, %zero : (i32, i32) -> i32
  %lane = rocdl.mbcnt.hi %ones, %lo : (i32, i32) -> i32
  %flipped = llvm.xor %lane, %sixteen : i32
  %width = llvm.sub %zero, %sixteen : i32
  %mask = llvm.and %flipped, %width : i32
  %index = llvm.shl %mask, %two : i32
  %shuffled = rocdl.ds_bpermute %index, %leader : (i32, i32) -> i32
  llvm.call @use(%shuffled) : (i32) -> ()
  %next = llvm.add %iv, %one : i64
  llvm.br ^header(%next : i64)
^exit:
  llvm.return
}}
'''


def blocks(text):
    """Split one printed function into entry, loop header/body and exit text."""
    function = text[text.index('llvm.func @kernel'):]
    parts = re.split(r'^\s*\^bb\d+.*$', function, flags=re.M)
    return parts[0], ''.join(parts[1:-1]), parts[-1]


class HoistWaveInvariants(unittest.TestCase):
    def run_loop(self, *options, condition=UNIFORM, leader='rocdl.readfirstlane %base : i32'):
        flag = '--frisk-hoist-wave-invariants' + ('=' + ' '.join(options) if options else '')
        return run_pass(LOOP.format(condition=condition, leader=leader), flag)

    def test_hoists_both(self):
        output, log = self.run_loop()
        entry, loop, _ = blocks(output)
        self.assertIn('1 LDS bases', log)
        for op in ('rocdl.readfirstlane', 'rocdl.mbcnt.lo', 'rocdl.mbcnt.hi', 'llvm.xor', 'llvm.shl'):
            self.assertIn(op, entry, op)
            self.assertNotIn(op, loop, op)
        # Data operations stay in the loop.
        for op in ('rocdl.global.load.async.lds', 'rocdl.ds_bpermute', 'llvm.call @use'):
            self.assertIn(op, loop, op)

    def test_options_select_independently(self):
        entry, loop, _ = blocks(self.run_loop('lane-shuffles=false')[0])
        self.assertIn('rocdl.readfirstlane', entry)
        self.assertIn('rocdl.mbcnt.lo', loop)
        entry, loop, _ = blocks(self.run_loop('lds-bases=false')[0])
        self.assertIn('rocdl.readfirstlane', loop)
        self.assertIn('rocdl.mbcnt.lo', entry)

    def test_refuses_divergent_loop(self):
        # A thread-dependent bound changes exec between entry and body, so the
        # first active lane could differ and readfirstlane may not move.
        output, log = self.run_loop(condition='llvm.icmp "slt" %tid, %sixteen : i32')
        _, loop, _ = blocks(output)
        self.assertIn('rocdl.readfirstlane', loop)
        self.assertIn('rocdl.mbcnt.lo', loop)
        self.assertIn('skipped a loop', log)

    def test_refuses_induction_dependent_base(self):
        leader = ('rocdl.readfirstlane %varying : i32\n'
                  '  %unused.anchor = llvm.mlir.constant(0 : i32) : i32')
        text = LOOP.format(condition=UNIFORM, leader=leader).replace(
            '^body:\n', '^body:\n  %varying = llvm.trunc %iv : i64 to i32\n')
        output, _ = run_pass(text, '--frisk-hoist-wave-invariants=lane-shuffles=false')
        _, loop, _ = blocks(output)
        self.assertIn('rocdl.readfirstlane', loop)


class ElideBackedgeBarrier(unittest.TestCase):
    def export(self, text, *flags):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.ll'
            path.write_text(text)
            result = subprocess.run([EXPORT_TOOL, '--cross-bn-k0=1', *flags, path], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            path.write_text(result.stdout)
            verify = subprocess.run([LLVM_OPT, '-passes=default<O3>,verify', '-disable-output', path],
                                    text=True, capture_output=True)
            self.assertEqual(verify.returncode, 0, verify.stderr)
        return result

    def test_removes_only_the_backedge_barrier(self):
        for threads, packets, width, tiles in [(64, 1, 8, 1), (128, 2, 8, 7), (256, 1, 16, 19)]:
            with self.subTest(threads=threads, packets=packets):
                text = cross_bn.fixture(threads, packets, width, tiles)
                kept = self.export(text)
                elided = self.export(text, '--elide-backedge-barrier=1')
                count = lambda s, what: s.count(what)
                self.assertIn('Elide backedge barrier: removed', elided.stderr)
                barrier = 'call void @llvm.amdgcn.s.barrier'
                self.assertEqual(count(elided.stdout, barrier), count(kept.stdout, barrier) - 1)
                self.assertEqual(count(elided.stdout, 'fence'), count(kept.stdout, 'fence') - 2)
                self.assertEqual(count(elided.stdout, 's_waitcnt vmcnt(0)'), count(kept.stdout, 's_waitcnt vmcnt(0)'))
                # The join block must no longer end in a barrier.
                join = elided.stdout[elided.stdout.index('k0.join:'):]
                join = join[:join.index('\n}')]
                self.assertNotIn(barrier, join)

    def test_inert_without_k0_prefetch(self):
        text = cross_bn.fixture(64, 1, 8, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.ll'
            path.write_text(text)
            off = subprocess.run([EXPORT_TOOL, '--cross-bn-k0=0', '--elide-backedge-barrier=1', path],
                                 text=True, capture_output=True)
        self.assertEqual(off.returncode, 0, off.stderr[-3000:])
        self.assertNotIn('Elide backedge barrier', off.stderr)
        self.assertEqual(off.stdout.count('call void @llvm.amdgcn.s.barrier'),
                         text.count('call void @llvm.amdgcn.s.barrier'))


if __name__ == '__main__':
    unittest.main()
