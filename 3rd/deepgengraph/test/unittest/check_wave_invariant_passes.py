#!/usr/bin/env python3
"""frisk-simplify-bounded-affine, frisk-hoist-wave-invariants.

Covers what each may rewrite, what it must refuse, and, for the affine pass,
brute-force equality of every rewritten map against the original.
"""
from itertools import product
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / 'build/test'
OPT_TOOL = BUILD / 'ThreadTilingTest'
LLVM_OPT = Path('/data2/xsl/install/bin/opt')


def run_pass(text, *flags):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'input.mlir'
        path.write_text(text)
        result = subprocess.run([OPT_TOOL, *flags, '--verify-each', path], text=True, capture_output=True)
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

    def test_planned_bridge_cfg_preserves_uniformity_check(self):
        for condition, uniform in [(UNIFORM, True),
                                   ('llvm.icmp "slt" %tid, %sixteen : i32', False)]:
            source = LOOP.format(condition=condition, leader='rocdl.readfirstlane %base : i32')
            source = source.replace('llvm.func @use(i32)',
                'llvm.func @use(i32)\nllvm.func @__frisk_async_load_lds('
                '!llvm.ptr<1>, !llvm.ptr<3>, i32, i32, i32)')
            source = source.replace(
                'rocdl.global.load.async.lds %src, %pointer, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>',
                '%bytes = llvm.mlir.constant(8 : i32) : i32\n'
                '  llvm.call @__frisk_async_load_lds(%src, %pointer, %bytes, %zero, %zero) '
                ': (!llvm.ptr<1>, !llvm.ptr<3>, i32, i32, i32) -> ()')
            output, _ = run_pass(source, '--frisk-hoist-wave-invariants')
            entry, loop, _ = blocks(output)
            self.assertEqual('rocdl.readfirstlane' in entry, uniform)
            self.assertEqual('rocdl.readfirstlane' in loop, not uniform)
            self.assertIn('llvm.call @__frisk_async_load_lds', loop)


EARLY_WAVE = '''module {
func.func @kernel(%src: memref<256xi32, 1>, %dst: memref<256xi32, 3>,
                  %out: memref<4xi32, 1>) attributes {thread_num = 128 : i32} {
  %zero = arith.constant 0 : index
  %offset = arith.constant 16 : i32
  %width = arith.constant 64 : i32
  affine.for %i = 0 to 4 {
    %task = frisk.async_copy_gm_to_shm %src to %dst {alignment = 4 : i64}
      : memref<256xi32, 1>, memref<256xi32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task : !frisk.async_task
    %v = memref.load %dst[%zero] : memref<256xi32, 3>
    %y, %valid = gpu.shuffle xor %v, %offset, %width : i32
    memref.store %y, %out[%i] : memref<4xi32, 1>
  }
  return
}
}
'''

# Mixed dialect fixture independent of DMA planning. The target pointer is an
# argument so tests can distinguish loop dependence from descriptor conversion.
STRUCTURED_WAVE = '''func.func @kernel(%src: !llvm.ptr<1>, %n: index, %pred: i1) {{
  %start = arith.constant 0 : index
  %end = arith.constant 4 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %ones = arith.constant -1 : i32
  %width = arith.constant 64 : i32
  %offset = arith.constant 16 : i32
  {prefix}
  {loop} {{
    {address}
    %leader = rocdl.readfirstlane %address : i32
    %pointer = llvm.inttoptr %leader : i32 to !llvm.ptr<3>
    rocdl.global.load.async.lds %src, %pointer, 4, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    %y, %valid = gpu.shuffle xor %leader, %offset, %width : i32
    "test.use"(%y, %valid) : (i32, i1) -> ()
  }}
  {suffix}
  return
}}
'''


class StructuredWaveInvariants(unittest.TestCase):
    def lower(self, source=EARLY_WAVE, options='', hoist=True):
        flags = ['--allow-unregistered-dialect', '--frisk-lower-async-copy',
                 '--frisk-lower-gpu-shuffle']
        if hoist:
            flags.append('--frisk-hoist-wave-invariants' + ('=' + options if options else ''))
        return run_pass(source, *flags)

    def fixture(self, loop='affine.for %i = 0 to 4',
                address='%address = arith.constant 0 : i32', prefix='', suffix=''):
        return STRUCTURED_WAVE.format(loop=loop, address=address, prefix=prefix, suffix=suffix)

    def test_partial_lowering_preserves_affine_and_other_operations(self):
        output, _ = self.lower(hoist=False)
        self.assertIn('func.func @kernel', output)
        self.assertIn('affine.for', output)
        for op in ('rocdl.readfirstlane', 'rocdl.mbcnt.lo', 'rocdl.ds_bpermute',
                   'rocdl.global.load.async.lds', 'memref.load', 'memref.store', 'gpu.barrier'):
            self.assertIn(op, output)
        self.assertNotIn('gpu.shuffle', output)

    def test_hoists_mixed_descriptor_and_lane_chains(self):
        output, log = self.lower()
        entry, loop = output.split('affine.for', 1)
        for op in ('rocdl.readfirstlane', 'rocdl.mbcnt.lo', 'rocdl.mbcnt.hi', 'llvm.shl'):
            self.assertIn(op, entry)
            self.assertNotIn(op, loop)
        for op in ('rocdl.global.load.async.lds', 'rocdl.ds_bpermute', 'memref.load', 'memref.store'):
            self.assertIn(op, loop)
        self.assertIn('(structured)', log)

    def test_planned_copy_participation_preserves_hoisting(self):
        for width in (8, 16):
            for participants in (64, 128):
                with self.subTest(width=width, participants=participants):
                    source = EARLY_WAVE.replace('256xi32', '1024xi32').replace(
                        'alignment = 4 : i64',
                        f'alignment = 16 : i64, frisk.dma_bytes = {width} : i64, '
                        f'frisk.copy_threads = {participants} : i64')
                    output, _ = self.lower(source)
                    entry, loop = output.split('affine.for', 1)
                    # All waves participate: bases can leave the loop. A
                    # partial copy retains its mask and convergent leader read.
                    full = participants == 128
                    self.assertEqual('scf.if' in loop, not full)
                    self.assertEqual('rocdl.readfirstlane' in loop, not full)
                    # The 8B bridge declaration may precede the function, but
                    # it must never carry the actual copy out of the loop.
                    self.assertIn('rocdl.readfirstlane', entry if full else loop)
                    instruction = ('llvm.call @__frisk_async_load_lds' if width == 8
                                   else 'rocdl.global.load.async.lds')
                    self.assertIn(instruction, loop)
                    self.assertNotIn(instruction, entry)

    def test_planned_bridge_refuses_varying_base(self):
        source = self.fixture(address='%address = arith.index_cast %i : index to i32')
        source = source.replace(
            'rocdl.global.load.async.lds %src, %pointer, 4, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>',
            '%bytes = arith.constant 8 : i32\n'
            '    llvm.call @__frisk_async_load_lds(%src, %pointer, %bytes, %zero, %zero) '
            ': (!llvm.ptr<1>, !llvm.ptr<3>, i32, i32, i32) -> ()')
        source = ('module {\nllvm.func @__frisk_async_load_lds(!llvm.ptr<1>, '
                  '!llvm.ptr<3>, i32, i32, i32)\n' + source + '\n}')
        entry, loop = self.lower(source)[0].split('affine.for', 1)
        self.assertNotIn('rocdl.readfirstlane', entry)
        self.assertIn('rocdl.readfirstlane', loop)

    def test_scf_uniform_dynamic_bound_and_options(self):
        source = self.fixture(loop='scf.for %i = %start to %n step %step')
        for options, bases, shuffles in [('', True, True), ('lds-bases=false', False, True),
                                         ('lane-shuffles=false', True, False)]:
            with self.subTest(options=options):
                entry, loop = self.lower(source, options)[0].split('scf.for', 1)
                for op, moved in [('rocdl.readfirstlane', bases), ('rocdl.mbcnt.lo', shuffles)]:
                    self.assertEqual(op in entry, moved)
                    self.assertEqual(op in loop, not moved)

    def test_refuses_induction_dependent_base(self):
        source = self.fixture(address='%address = arith.index_cast %i : index to i32')
        entry, loop = self.lower(source)[0].split('affine.for', 1)
        self.assertNotIn('rocdl.readfirstlane', entry)
        self.assertIn('rocdl.readfirstlane', loop)
        self.assertIn('rocdl.mbcnt.lo', entry)

    def test_guarded_dma_base_requires_uniform_predicate_and_invariance(self):
        for uniform, invariant in [(True, True), (False, True), (True, False)]:
            with self.subTest(uniform=uniform, invariant=invariant):
                address = ('%address = arith.constant 0 : i32' if invariant else
                           '%address = arith.index_cast %i : index to i32')
                predicate = ('%condition = arith.cmpi ult, %i, %end : index' if uniform else
                             '%tid = gpu.thread_id x\n'
                             '    %condition = arith.cmpi ult, %tid, %end : index')
                source = self.fixture(address=address).replace(address,
                    predicate + '\n    scf.if %condition {\n    ' + address)
                source = source.replace('"test.use"(%y, %valid) : (i32, i1) -> ()',
                    '"test.use"(%y, %valid) : (i32, i1) -> ()\n    }')
                output, _ = self.lower(source)
                entry, body = output.split('affine.for', 1)
                self.assertEqual('rocdl.readfirstlane' in entry, uniform and invariant)
                self.assertEqual('rocdl.readfirstlane' in body, not (uniform and invariant))
                self.assertIn('rocdl.global.load.async.lds', body.split('scf.if', 1)[1])
                repeated, _ = run_pass(output, '--allow-unregistered-dialect',
                                       '--frisk-hoist-wave-invariants')
                self.assertEqual(output, repeated)

    def test_refuses_loop_carried_base_and_memory_read(self):
        carried = self.fixture(
            loop='%r = scf.for %i = %start to %end step %step '
                 'iter_args(%carry = %zero) -> (i32)',
            address='%address = arith.addi %carry, %offset : i32').replace(
                '"test.use"(%y, %valid) : (i32, i1) -> ()',
                '"test.use"(%y, %valid) : (i32, i1) -> ()\n'
                '    scf.yield %address : i32')
        loaded = self.fixture(address='%address = llvm.load %src : !llvm.ptr<1> -> i32')
        for source, marker in [(carried, 'scf.for'), (loaded, 'affine.for')]:
            with self.subTest(marker=marker):
                entry, loop = self.lower(source)[0].split(marker, 1)
                self.assertNotIn('rocdl.readfirstlane', entry)
                self.assertIn('rocdl.readfirstlane', loop)

    def test_refuses_divergent_loop_and_conditional_scope(self):
        sources = [self.fixture(prefix='%tid = gpu.thread_id x',
                                loop='scf.for %i = %start to %tid step %step'),
                   self.fixture(prefix='scf.if %pred {', suffix='}')]
        for source in sources:
            with self.subTest(source=source):
                output, log = self.lower(source)
                prefix, loop = output.split('scf.', 1)
                self.assertNotIn('rocdl.readfirstlane', prefix)
                self.assertNotIn('rocdl.mbcnt.lo', prefix)
                self.assertIn('rocdl.readfirstlane', loop)
                self.assertNotIn('(structured)', log)

    def test_nested_uniform_loops_hoist_to_outer_preheader(self):
        source = self.fixture(prefix='affine.for %outer = 0 to 3 {', suffix='}')
        entry, loops = self.lower(source)[0].split('affine.for', 1)
        self.assertIn('rocdl.readfirstlane', entry)
        self.assertIn('rocdl.mbcnt.lo', entry)
        self.assertNotIn('rocdl.readfirstlane', loops)
        self.assertNotIn('rocdl.mbcnt.lo', loops)

    def test_rejects_unsafe_division_in_zero_trip_loop(self):
        for divisor in (0, -1):
            source = self.fixture(loop='affine.for %i = 0 to 0',
                address=f'%divisor = arith.constant {divisor} : i32\n'
                        '%address = arith.divsi %offset, %divisor : i32')
            entry, loop = self.lower(source)[0].split('affine.for', 1)
            self.assertNotIn('rocdl.readfirstlane', entry)
            self.assertIn('rocdl.readfirstlane', loop)

    def test_mbcnt_with_varying_operand_stays_in_loop(self):
        source = self.fixture().replace(
            '%y, %valid = gpu.shuffle xor %leader, %offset, %width : i32',
            '%varying = arith.index_cast %i : index to i32\n'
            '%y = rocdl.mbcnt.lo %ones, %varying : (i32, i32) -> i32\n'
            '%valid = arith.constant true')
        entry, loop = self.lower(source)[0].split('affine.for', 1)
        self.assertNotIn('rocdl.mbcnt.lo', entry)
        self.assertIn('rocdl.mbcnt.lo', loop)

    def test_idempotent_and_complete_llvm_pipeline(self):
        early, _ = self.lower()
        repeated, log = run_pass(early, '--frisk-hoist-wave-invariants')
        self.assertEqual(repeated, early)
        self.assertNotIn('(structured)', log)
        llvm_bin = LLVM_OPT.parent
        for sync in (False, True):
            with self.subTest(sync=sync):
                flags = ['--frisk-lower-async-copy=use-async-copy=false'] if sync else []
                lowered, _ = run_pass(EARLY_WAVE, *flags, '--test-async-to-llvm')
                self.assertNotIn('unrealized_conversion_cast', lowered)
                self.assertNotIn('affine.for', lowered)
                self.assertNotIn('gpu.shuffle', lowered)
                self.assertEqual('rocdl.global.load.async.lds' in lowered, not sync)
                translated = subprocess.run([llvm_bin / 'mlir-translate', '--mlir-to-llvmir'],
                                            input=lowered, text=True, capture_output=True)
                self.assertEqual(translated.returncode, 0, translated.stderr)
                verified = subprocess.run([LLVM_OPT, '-passes=default<O3>,verify', '-disable-output'],
                                         input=translated.stdout, text=True, capture_output=True)
                self.assertEqual(verified.returncode, 0, verified.stderr)



if __name__ == '__main__':
    unittest.main()
