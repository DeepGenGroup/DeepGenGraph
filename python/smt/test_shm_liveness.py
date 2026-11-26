"""Memory pressure must follow the emitted schedule, not guessed GPU cycles."""
from itertools import permutations
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ortools.sat.python import cp_model
from cpsat_scheduler import main, solve
from schedule_schema import baseline_schedule, make_schedule
from shm_liveness import add_shm_constraint, analyze_shm, lifetime_specs, validate_periodic_storage
from copy import deepcopy


def problem():
    # A.write, B.write, A.read, B.read: peak 8 in this order, 4 if interleaved.
    ops = [dict(id=i, name='test.op', kind='alu', read_bytes=4 if i >= 2 else 0,
                write_bytes=4 if i < 2 else 0, compute_ops=1,
                reads=[i - 2] if i >= 2 else [], writes=[i] if i < 2 else [],
                pipeline_anchor=True, opaque=False) for i in range(4)]
    return dict(schema='deepgengraph.cpsat.problem', version=1, fingerprint='a' * 64,
                loop=dict(id='test:0', pipeline_eligible=True, pipeline_reason='',
                          lower=0, upper=5, step=1),
                operations=ops, wait_semantics='wait_all', limits=dict(max_stages=2, max_slots=2),
                dependencies=[dict(src=i, dst=i + 2, kind='raw', same_stage=True) for i in (0, 1)],
                buffers=[dict(id=i, bytes=4, memory_space=3, type='memref<1xf32, 3>',
                              rotatable=False, writers=[i], readers=[i + 2], users=[i, i + 2],
                              liveness=dict(iteration_local=True, reason='test owned full write'))
                         for i in (0, 1)])


class ShmLivenessTests(unittest.TestCase):
    def test_periodic_versions_share_two_physical_slots(self):
        p = problem()
        p['storage_contract'] = 'periodic-offsets-v1'
        for b in p['buffers']:
            b.update(bytes=16, alignment=16, rotatable=True)
        for i in (0, 1):
            p['dependencies'][i]['same_stage'] = False
        p['operations'][0]['pipeline_anchor'] = False
        # B is written before the current A is read: a real overlap, while
        # A's next version can reuse its range after that read completes.
        p['dependencies'].append(dict(src=1, dst=2, kind='ssa', same_stage=True))
        result = solve(p, mode='pipeline', require_cross_iteration=True,
                       lds_budget=32, time_limit=5)
        self.assertEqual(result['storage']['pool_bytes'], 32)
        self.assertGreater(result['model']['shared_bytes_upper_bound'], 32)
        self.assertLessEqual(result['model']['shm_liveness']['peak_bytes'], 32)
        validate_periodic_storage(p, result)
        forged = deepcopy(result)
        for b in forged['storage']['buffers']:
            b['offsets'] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'overlapping version'):
            validate_periodic_storage(p, forged)
        with self.assertRaises(RuntimeError):
            solve(p, mode='pipeline', require_cross_iteration=True, lds_budget=15, time_limit=2)

    def test_compact_order_encoding_preserves_lds_peak(self):
        p = problem()
        for i in range(4, 260):
            p['operations'].append(dict(id=i, name='test.op', kind='alu', read_bytes=0,
                write_bytes=0, compute_ops=1, reads=[], writes=[], pipeline_anchor=True, opaque=False))
            p['dependencies'].append(dict(src=i-1, dst=i, kind='ssa', same_stage=True))
        result = solve(p, lds_budget=4, shm_check='constraint', time_limit=5)
        self.assertEqual(result['model']['order_encoding'], 'phase')
        self.assertEqual(analyze_shm(p, result)['peak_bytes'], 4)
        self.assertEqual(sorted(o['order'] for o in result['operations']), list(range(260)))

    def test_baseline_cli_obeys_default_and_explicit_budget(self):
        p = problem()
        for b in p['buffers']:
            b['bytes'] = 16384
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'problem.json', Path(directory) / 'schedule.json'
            source.write_text(json.dumps(p))
            command = [str(source), '-o', str(output), '--baseline']
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)
                self.assertFalse(output.exists())
                self.assertEqual(main(command + ['--lds-budget', '32768']), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result['model']['resource_limits']['shm_bytes'], 32768)

    def test_default_16k_is_hard_limit_and_user_can_override(self):
        p = problem()
        for b in p['buffers']:
            b['bytes'] = 32768
        with self.assertRaises(RuntimeError):
            solve(p, time_limit=2)
        result = solve(p, lds_budget=32768, time_limit=2)
        self.assertEqual(result['model']['resource_limits']['shm_bytes'], 32768)
        self.assertEqual(result['model']['shm_liveness']['peak_bytes'], 32768)
        for b in p['buffers']:
            b['bytes'] = 16384
        result = solve(p, time_limit=2)
        self.assertEqual(result['model']['resource_limits']['shm_bytes'], 16384)
        self.assertEqual(result['model']['shm_liveness']['peak_bytes'], 16384)
        with self.assertRaises(ValueError):
            solve(p, lds_budget=None)

    def test_peak_follows_order_and_shared_endpoints(self):
        p = problem()
        self.assertEqual(analyze_shm(p, baseline_schedule(p))['peak_bytes'], 8)
        s = make_schedule(p, [0] * 4, [0, 2, 1, 3], mode='intra', status='TEST')
        s['model'] = {'phase': [0, 100, 2, 500]}  # Idle gaps do not free live storage.
        self.assertEqual(analyze_shm(p, s)['peak_bytes'], 4)
        p['operations'][1]['reads'] = [0]
        p['operations'][2]['reads'] = []
        p['buffers'][0].update(readers=[1], users=[0, 1])
        self.assertEqual(analyze_shm(p, baseline_schedule(p))['peak_bytes'], 8)

    def test_constraint_and_postcheck_keep_minimum_ii(self):
        p = problem()
        with self.assertRaises(RuntimeError):
            solve(p, lds_budget=4, shm_check='sum', time_limit=2)
        for strategy in ('constraint', 'postcheck', 'auto'):
            with self.subTest(strategy=strategy):
                result = solve(p, lds_budget=4, shm_check=strategy, time_limit=3)
                self.assertEqual(result['model']['period'], 4)
                self.assertEqual(result['model']['shm_liveness']['peak_bytes'], 4)
                self.assertEqual(result['model']['shared_bytes_upper_bound'], 8)

    def test_postcheck_exhaustion_never_returns_over_budget_solution(self):
        with patch('cpsat_scheduler.SHM_MAX_POSTCHECK_REJECTIONS', 1):
            with self.assertRaisesRegex(RuntimeError, 'post-check rejected 1'):
                solve(problem(), lds_budget=0, shm_check='postcheck', time_limit=2)
        with patch('cpsat_scheduler.SHM_MAX_CUMULATIVE_INTERVALS', 0):
            result = solve(problem(), lds_budget=4, time_limit=3)
            self.assertEqual(result['model']['shm_check'], 'postcheck')

    def test_opaque_escape_and_unknown_size_are_conservative(self):
        p = problem()
        for b in p['buffers']:
            b.pop('liveness')  # Old JSON has no proof for these non-DMA buffers.
        with self.assertRaises(RuntimeError):
            solve(p, lds_budget=4, shm_check='constraint', time_limit=2)
        p['buffers'][0]['bytes'] = -1
        self.assertIsNone(analyze_shm(p, baseline_schedule(p))['peak_bytes'])
        with self.assertRaisesRegex(ValueError, 'unknown buffer size'):
            solve(p, lds_budget=8)

    def test_dma_wait_extends_lifetime_and_versions_remain_reserved(self):
        p = problem()
        p['buffers'] = p['buffers'][:1]
        p['buffers'][0].update(rotatable=True, readers=[], users=[0])
        for op in p['operations']:
            op.update(reads=[], writes=[])
        p['operations'][0].update(kind='dma', writes=[0], pipeline_anchor=False)
        p['operations'][2]['kind'] = 'wait'
        p['dependencies'] = [dict(src=0, dst=2, kind='completion', same_stage=False)]
        # No explicit consumer: completion still keeps the destination alive.
        result = analyze_shm(p, baseline_schedule(p))
        self.assertEqual(result['buffers'][0]['end'], 3)
        p['buffers'][0]['users'] = [0, 2]
        schedule = make_schedule(p, [0, 1, 1, 1], [3, 0, 1, 2], mode='pipeline', status='TEST')
        result = analyze_shm(p, schedule)
        self.assertEqual(result['buffers'][0]['end'], 6)
        self.assertEqual(result['peak_bytes'], 8)  # Two materialized slots stay reserved.
        self.assertTrue(result['buffers'][0]['budget_reserved'])
        p['dependencies'] = []
        with self.assertRaisesRegex(ValueError, 'completion/wait'):
            solve(p)

    def test_cumulative_matches_sweep_for_all_small_orders(self):
        p = problem()
        specs = lifetime_specs(p)
        for sequence in permutations(range(4)):
            rank = [sequence.index(i) for i in range(4)]
            if rank[0] > rank[2] or rank[1] > rank[3]:
                continue
            s = make_schedule(p, [0] * 4, rank, mode='intra', status='TEST')
            peak = analyze_shm(p, s)['peak_bytes']
            for budget in (4, 8):
                model = cp_model.CpModel()
                add_shm_constraint(model, specs, [0] * 4, rank, [1, 1],
                                   n=4, stage_limit=1, budget=budget)
                feasible = cp_model.CpSolver().solve(model) == cp_model.OPTIMAL
                self.assertEqual(feasible, peak <= budget, (rank, budget, peak))


if __name__ == '__main__':
    unittest.main()
