import unittest
from cost_model import CostModel
from cpsat_scheduler import solve


class FragmentCostTests(unittest.TestCase):
    def test_fragment_instruction_count_and_mmac_result_delay(self):
        costs = CostModel()
        self.assertEqual(costs.operation(dict(kind='alu', issue_units=8)), (8, 8))
        self.assertEqual(costs.operation(dict(kind='wmma', issue_units=1)), (1, 4))
        p = self.fixture(3)
        p['operations'][0]['kind'] = 'wmma'
        s = solve(p, costs=costs, time_limit=5)
        self.assertGreaterEqual(s['model']['phase'][1] - s['model']['phase'][0],
                                4 * s['model']['time_scale'])

    def fixture(self, count=12):
        ops = [dict(id=i, name='test.view', kind='alu', zero_issue=0 < i < count-1,
                    read_bytes=0, write_bytes=0, compute_ops=0, reads=[], writes=[],
                    pipeline_anchor=True, opaque=False) for i in range(count)]
        return dict(schema='deepgengraph.cpsat.problem', version=1, fingerprint='c'*64,
                    loop=dict(id='test:0', pipeline_eligible=True, pipeline_reason='',
                              lower=0, upper=5, step=1),
                    operations=ops, buffers=[], wait_semantics='wait_all',
                    limits=dict(max_stages=2, max_slots=2),
                    dependencies=[dict(src=i, dst=i+1, same_stage=True, kind='ssa')
                                  for i in range(count-1)])

    def test_views_keep_order_without_charging_full_instruction_cost(self):
        p = self.fixture()
        s = solve(p, time_limit=5)
        self.assertLess(s['model']['period'], 3)
        self.assertEqual(s['model']['zero_issue_operations'], 10)
        self.assertEqual([o['order'] for o in s['operations']], list(range(12)))

    def test_load_latency_is_distinct_from_issue_and_views_cannot_hide_it(self):
        p = self.fixture()
        p['operations'][0]['kind'] = 'shm_reg'
        costs = CostModel(shm_issue=1, shm_reg=9)
        self.assertEqual(costs.operation(p['operations'][0]), (1, 9))
        s = solve(p, costs=costs, time_limit=5)
        model = s['model']
        self.assertGreaterEqual(model['phase'][1] - model['phase'][0],
                                9 * model['time_scale'])
        self.assertGreaterEqual(model['period'], 10)


if __name__ == '__main__':
    unittest.main()
