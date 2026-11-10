"""Memory pressure must follow the emitted schedule, not guessed GPU cycles."""
from itertools import permutations
import unittest
from unittest.mock import patch

from ortools.sat.python import cp_model
from cpsat_scheduler import solve
from schedule_schema import baseline_schedule, make_schedule
from shm_liveness import add_shm_constraint, analyze_shm, lifetime_specs


def problem():
    # A.write, B.write, A.read, B.read: peak 8 in this order, 4 if interleaved.
    ops = [dict(id=i, name='test.op', kind='alu', read_bytes=4 if i >= 2 else 0,
                write_bytes=4 if i < 2 else 0, compute_ops=1,
                reads=[i - 2] if i >= 2 else [], writes=[i] if i < 2 else [],
                pipeline_anchor=True, opaque=False) for i in range(4)]
    return dict(schema='deepgengraph.cpsat.problem', version=1, fingerprint='a' * 64,
                loop=dict(id='test:0', pipeline_eligible=True, pipeline_reason=''),
                operations=ops, wait_semantics='wait_all', limits=dict(max_stages=2, max_slots=2),
                dependencies=[dict(src=i, dst=i + 2, kind='raw', same_stage=True) for i in (0, 1)],
                buffers=[dict(id=i, bytes=4, memory_space=3, type='memref<1xf32, 3>',
                              rotatable=False, writers=[i], readers=[i + 2], users=[i, i + 2],
                              liveness=dict(iteration_local=True, reason='test owned full write'))
                         for i in (0, 1)])


class ShmLivenessTests(unittest.TestCase):
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
