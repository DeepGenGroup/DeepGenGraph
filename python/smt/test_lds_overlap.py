import unittest

from cost_model import CostModel
from cpsat_scheduler import solve
from lds_overlap import lds_overlap_rules, check_lds_overlap
from register_hint import register_hint
import test_cost_model


class LDSOverlapTests(unittest.TestCase):
    def fixture(self):
        p = test_cost_model.FragmentCostTests().fixture(6)
        p['dependencies'] = []
        values = []
        for i in range(3):
            read, consume = 2 * i, 2 * i + 1
            p['operations'][read].update(kind='shm_reg', name='frisk.read_matrix_from_shm',
                                         reads=[0], zero_issue=False)
            p['operations'][consume].update(kind='wmma', name='frisk.gemm_frag', zero_issue=False)
            p['dependencies'].append(dict(src=read, dst=consume, kind='ssa', same_stage=True))
            values.append(dict(producer=read, words=4, users=[dict(op=consume, distance=0)]))
        # No physical buffer is needed for this synthetic register-only probe;
        # the read ID is only the grouping key for the rule builder.
        p['register_liveness'] = dict(contract='fragment-origin-words-v2', supported=True,
                                     invariant_words=0, values=values)
        return p

    def test_order_requires_lookahead_and_retirement_not_virtual_idle(self):
        p = self.fixture()
        rules = lds_overlap_rules(p)
        sequential = dict(operations=[dict(id=i, order=i, stage=0) for i in range(6)])
        self.assertFalse(check_lds_overlap(rules, sequential)['within_rules'])
        # L0, L1, C0, L2, C1, C2: a real independent MMAC lies between
        # every steady-state read and its use; only two read values are live.
        order = register_hint(p, [0]*6, list(range(6)), 8, 0,
                              policy='source', order_edges=rules['edges'])
        self.assertEqual(order, [0, 2, 1, 4, 3, 5])

    def test_dependent_gather_does_not_get_an_illegal_prefetch_edge(self):
        p = self.fixture()
        p['dependencies'].append(dict(src=1, dst=2, kind='ssa', same_stage=True))
        rules = lds_overlap_rules(p)
        self.assertEqual(rules['edges'], [])
        self.assertEqual(rules['skipped_groups'], 1)

    def test_real_cp_sat_constraints_keep_register_cap(self):
        p = self.fixture()
        # Satisfy the full schema while retaining the read grouping key.
        # Use a valid test buffer fixture instead of pretending these reads
        # access an undeclared resource.
        p['buffers'] = [dict(id=0, name='lds', memory_space=3, bytes=16,
                             rotatable=False, users=[0, 2, 4], readers=[0, 2, 4], writers=[])]
        s = solve(p, costs=CostModel(shm_reg=9), vgpr_budget=8,
                  register_reserve=0, time_limit=5, lds_read_ahead=True)
        self.assertTrue(s['model']['lds_read_ahead']['check']['within_rules'])
        self.assertLessEqual(s['model']['register_liveness']['estimated_peak_words'], 8)
        positions = {o['id']: o['order'] for o in s['operations']}
        self.assertLess(positions[2], positions[1])
        self.assertLess(positions[1], positions[4])

    def test_explicit_disable_preserves_original_search_space(self):
        self.assertEqual(lds_overlap_rules(self.fixture(), enabled=False)['edges'], [])

    def test_default_allows_three_reads_while_preserving_register_budget(self):
        p = self.fixture()
        p['loop']['upper'] = 1
        p['buffers'] = [dict(id=0, memory_space=3, bytes=16, rotatable=False,
                             users=[0, 2, 4], readers=[0, 2, 4], writers=[])]
        options = dict(costs=CostModel(shm_reg=9), vgpr_budget=12,
                       register_reserve=0, time_limit=5)
        free = solve(p, **options)
        bounded = solve(p, lds_read_ahead=True, **options)
        self.assertFalse(free['model']['lds_read_ahead']['rules']['enabled'])
        self.assertEqual(free['model']['lds_read_ahead']['rules']['edges'], [])
        order = {o['id']: o['order'] for o in free['operations']}
        self.assertLess(max(order[i] for i in (0, 2, 4)), min(order[i] for i in (1, 3, 5)))
        self.assertLess(free['model']['makespan'], bounded['model']['makespan'])
        self.assertLessEqual(free['model']['register_liveness']['estimated_peak_words'], 12)
        limited = solve(p, **dict(options, vgpr_budget=8))
        self.assertLessEqual(limited['model']['register_liveness']['estimated_peak_words'], 8)


if __name__ == '__main__':
    unittest.main()
