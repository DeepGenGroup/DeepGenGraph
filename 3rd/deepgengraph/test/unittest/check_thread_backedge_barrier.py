#!/usr/bin/env python3
"""ThreadIR reader handoff elision: hazards, task SSA, and cyclic ownership."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from check_shared_cross_iteration import fixture as planned_fixture, run as plan, simulate

TOOL = Path(__file__).resolve().parents[1] / 'build/test/ThreadTilingTest'
PASS = '--frisk-thread-elide-backedge-barrier'


def run(source, *flags):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'input.mlir'
        path.write_text(source)
        result = subprocess.run([str(TOOL), str(path), '--allow-unregistered-dialect',
                                 '--verify-each', *flags], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        return result.stdout, result.stderr


def fixture(loop='affine', prefix='', tail='', exit_prefix='', head=True,
            drain=True, marker=True, bound='4', step=1):
    header = (f'affine.for %iv = 0 to {bound} step {step}' if loop == 'affine' else
              'scf.for %iv = %zero to %bound step %step')
    return f'''module {{
func.func @kernel(%shm: memref<64xi32, 3>, %gm: memref<64xi32, 1>, %bound: index)
attributes {{thread_num = 64 : i32{', frisk.cross_bn_k0 = "shared-plan-v1"' if marker else ''}}} {{
  %zero = arith.constant 0 : index
  %step = arith.constant {step} : index
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c64 = arith.constant 64 : i32
  %tid = gpu.thread_id x
  %done = frisk.async_task_completed : !frisk.async_task
  %result = {header} iter_args(%task = %done) -> (!frisk.async_task) {{
    {prefix}
    {'frisk.wait_buffer_ready %task : !frisk.async_task' if head else ''}
    %read = memref.load %shm[%zero] : memref<64xi32, 3>
    gpu.barrier
    {tail}
    {loop}.yield %done : !frisk.async_task
  }} {{frisk.cross_bn_k0}}
  {exit_prefix}
  {'frisk.wait_buffer_ready %result : !frisk.async_task' if drain else ''}
  return
}}
}}
'''


class ThreadBackedgeBarrier(unittest.TestCase):
    def check(self, source, allowed):
        before, _ = run(source)
        after, log = run(source, PASS)
        self.assertEqual('removed 1 per iteration' in log, allowed, log)
        self.assertEqual(after.count('gpu.barrier'), before.count('gpu.barrier') - int(allowed))
        self.assertEqual(after.count('frisk.wait_buffer_ready'), before.count('frisk.wait_buffer_ready'))
        self.assertEqual(after.count('frisk.async_copy_gm_to_shm'), before.count('frisk.async_copy_gm_to_shm'))
        again, _ = run(after, PASS)
        self.assertEqual(again, after, 'pass must be idempotent')
        return after

    def test_affine_scf_zero_single_dynamic_and_step(self):
        for loop in ('affine', 'scf'):
            for bound in ('0', '1', '4', '%bound'):
                for step in (1, 3):
                    with self.subTest(loop=loop, bound=bound, step=step):
                        self.check(fixture(loop, bound=bound, step=step), True)

    def test_register_only_exit_reduction(self):
        self.check(fixture(exit_prefix='''
          %peer, %valid = gpu.shuffle xor %c0, %c16, %c64 : i32
          %sum = arith.addi %peer, %c0 : i32
        '''), True)

    def test_memory_dma_unknown_effects_and_control_rejected(self):
        hazards = [
            '%bad = memref.load %shm[%zero] : memref<64xi32, 3>',
            'memref.store %c0, %shm[%zero] : memref<64xi32, 3>',
            'memref.store %c0, %gm[%zero] : memref<64xi32, 1>',
            '%copy = frisk.async_copy_gm_to_shm %gm to %shm {alignment = 16 : i64} : memref<64xi32, 1>, memref<64xi32, 3> -> !frisk.async_task',
            '"test.unknown"() : () -> ()',
            'scf.execute_region { scf.yield }',
        ]
        for place in ('prefix', 'tail', 'exit_prefix'):
            for hazard in hazards:
                with self.subTest(place=place, hazard=hazard):
                    self.check(fixture(**{place: hazard}), False)

    def test_requires_plan_head_uniformity_and_matching_drain(self):
        for source in (fixture(marker=False), fixture(head=False), fixture(drain=False),
                       fixture(bound='%tid'),
                       fixture().replace('frisk.wait_buffer_ready %task', 'frisk.wait_buffer_ready %done'),
                       fixture().replace('frisk.wait_buffer_ready %result', 'frisk.wait_buffer_ready %done'),
                       fixture().replace('shared-plan-v1', 'sum-anchor-v1')):
            with self.subTest(source=source):
                self.check(source, False)

    def test_readiness_barrier_is_not_a_reader_barrier(self):
        self.check(fixture().replace('    gpu.barrier',
                   '    frisk.wait_buffer_ready %done : !frisk.async_task'), False)

    def test_legacy_exporter_rejects_removed_option(self):
        result = subprocess.run([str(TOOL.with_name('LegacyLLVMExportTest')),
                                 '--elide-backedge-barrier=1', '/dev/null'],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('backedge barrier elision moved to ThreadIR', result.stderr)

    def test_real_execution_plan_and_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.mlir'
            for loop in ('affine', 'scf'):
                for trips, step, dynamic, lower in ((0,1,False,0), (1,1,False,0),
                        (4,1,False,0), (4,3,False,0), (4,1,True,0), (4,1,False,2)):
                    with self.subTest(loop=loop, trips=trips, step=step, dynamic=dynamic, lower=lower):
                        source, _ = plan(path, planned_fixture(trips, step, loop, dynamic, lower=lower))
                        after = self.check(source, True)
                        for bound in ((0, 1, 3, 4) if dynamic else (None,)):
                            simulate(after, bound)
                        lowered, _ = run(after, '--frisk-lower-async-copy')
                        self.assertNotIn('!frisk.async_task', lowered)
                        before_lowering, _ = run(source, '--frisk-lower-async-copy')
                        self.assertEqual(lowered.count('gpu.barrier'), before_lowering.count('gpu.barrier') - 1)


if __name__ == '__main__':
    unittest.main()
