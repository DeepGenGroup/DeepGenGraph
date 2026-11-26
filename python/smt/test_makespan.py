"""Finite objective boundary, trip-count, and schedule tradeoff regressions."""
import unittest

from ortools.sat.python import cp_model
from cpsat_scheduler import solve
from makespan import add_makespan, resolve_trip_count
import test_cost_model


class MakespanTests(unittest.TestCase):
    def test_trip_count_rounding_upper_bounds_and_missing_information(self):
        p = dict(loop=dict(lower=3, upper=20, step=4))
        self.assertEqual(resolve_trip_count(p), (5, 'constant_bounds'))
        self.assertEqual(resolve_trip_count(p, 8), (5, 'constant_bounds'))
        with self.assertRaisesRegex(ValueError, 'smaller'):
            resolve_trip_count(p, 4)
        self.assertEqual(resolve_trip_count(dict(loop=dict(lower=7, upper=3, step=2)))[0], 0)
        p = dict(loop=dict(trip_count_upper_bound=32))
        self.assertEqual(resolve_trip_count(p), (32, 'exported_upper_bound'))
        self.assertEqual(resolve_trip_count(dict(loop={}), 16), (16, 'explicit_upper_bound'))
        with self.assertRaisesRegex(ValueError, 'finite trip count'):
            resolve_trip_count(dict(loop={}))
        for bad in (-1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_trip_count(dict(loop={}), bad)

    def test_short_loop_can_prefer_larger_ii_and_long_loop_reverses_choice(self):
        # Two legal periodic task schedules: A (II=5, span=10) and
        # B (II=6, span=3). A starts task 1 in stage 1; B uses stage 0.
        for trips, expected_ii, expected_time in ((1, 6, 3), (10, 5, 55)):
            model = cp_model.CpModel()
            choose_b = model.new_bool_var('choose_b')
            ii = model.new_int_var(5, 6, 'period')
            second = model.new_int_var(2, 9, 'second_start')
            model.add(ii == 5 + choose_b)
            model.add(second == 9 - 7 * choose_b)
            finite = add_makespan(model, [0, second], ii, [1, 1], [0, 1],
                                  trip_count=trips, horizon=20, stage_limit=2)
            model.minimize(finite['makespan'])
            solver = cp_model.CpSolver()
            self.assertEqual(solver.solve(model), cp_model.OPTIMAL)
            self.assertEqual(solver.value(ii), expected_ii)
            self.assertEqual(solver.value(finite['makespan']), expected_time)

    def test_views_do_not_set_boundaries_and_terminal_latency_is_counted(self):
        model = cp_model.CpModel()
        finite = add_makespan(model, [0, 7, 12, 30], 10, [0, 1, 6, 0], [1, 2],
                              trip_count=3, horizon=40, stage_limit=2)
        solver = cp_model.CpSolver()
        self.assertEqual(solver.solve(model), cp_model.OPTIMAL)
        self.assertEqual(solver.value(finite['first_issue']), 7)
        self.assertEqual(solver.value(finite['last_completion']), 18)
        self.assertEqual(solver.value(finite['makespan']), 31)

    def test_default_solver_objective_and_independent_completion_check(self):
        p = test_cost_model.FragmentCostTests().fixture(3)
        p['operations'][-1]['kind'] = 'wmma'
        for trips in (0, 1, 7):
            p['loop']['upper'] = trips
            result = solve(p, time_limit=5)
            m = result['model']
            self.assertEqual(m['objective'], 'makespan')
            self.assertEqual(m['lexicographic_solves'][0]['name'], 'makespan')
            self.assertEqual(m['trip_count'], trips)
            starts = [op['stage'] * m['period_ticks'] + m['phase'][op['id']]
                      for op in result['operations']]
            active = [i for i, issue in enumerate(m['issue_costs']) if issue]
            expected = 0 if not trips else max(
                starts[i] + (trips - 1) * m['period_ticks'] + m['latency'][i] for i in active
            ) - min(starts[i] for i in active)
            self.assertEqual(m['finite_horizon']['makespan_ticks'], expected)

    def test_no_real_tasks_and_overflow_guard(self):
        p = test_cost_model.FragmentCostTests().fixture(3)
        for op in p['operations']:
            op['zero_issue'] = True
        self.assertEqual(solve(p, time_limit=5)['model']['makespan'], 0)
        with self.assertRaisesRegex(ValueError, 'integer range'):
            add_makespan(cp_model.CpModel(), [0], 1, [1], [0],
                         trip_count=1 << 60, horizon=10, stage_limit=1)

    def test_unknown_n_requires_bound_but_legacy_ii_remains_available(self):
        p = test_cost_model.FragmentCostTests().fixture(2)
        p['loop'] = dict(id='test:0', pipeline_eligible=False, pipeline_reason='dynamic')
        with self.assertRaisesRegex(ValueError, 'finite trip count'):
            solve(p)
        bounded = solve(p, trip_count_upper_bound=32, time_limit=5)
        self.assertEqual(bounded['model']['trip_count'], 32)
        self.assertEqual(bounded['model']['trip_count_source'], 'explicit_upper_bound')
        old = solve(p, objective='ii', time_limit=5)
        self.assertEqual(old['model']['lexicographic_solves'][0]['name'], 'ii')
        self.assertNotIn('makespan', old['model'])


if __name__ == '__main__':
    unittest.main()
