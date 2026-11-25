import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from ortools.sat.python import cp_model
from register_liveness import add_register_constraint, analyze_registers
from register_hint import register_hint


class RegisterLivenessTests(unittest.TestCase):
    def export(self, source):
        root = Path(__file__).resolve().parents[2]
        bridge = root / '3rd/deepgengraph/build/lib/Pipeline/FriskScheduleJSONTest'
        with tempfile.TemporaryDirectory() as directory:
            path, output, ir = [Path(directory)/x for x in ('input.mlir', 'problem.json', 'base.mlir')]
            path.write_text(source)
            subprocess.run([str(bridge), str(path), '--fragment', '--loop=probe:0',
                            f'--export={output}', f'--ir={ir}'], check=True, capture_output=True)
            return json.loads(output.read_text()), ir.read_text()

    def test_static_fragment_slices_do_not_wait_for_unrelated_inserts(self):
        p, ir = self.export('''func.func @probe(%x: vector<8xf32>, %a: !frisk.fragment<vector<4xf32>, {}>, %b: !frisk.fragment<vector<4xf32>, {}>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = affine.for %i = 0 to 4 iter_args(%v = %x) -> vector<8xf32> {
    %t0 = frisk.insert_frag %a, %v[%c0] : <vector<4xf32>, {}> into vector<8xf32>
    %t1 = frisk.insert_frag %b, %t0[%c1] : <vector<4xf32>, {}> into vector<8xf32>
    %s = frisk.extract_frag %t1[%c0] : vector<8xf32> -> <vector<4xf32>, {}>
    %u = frisk.elementwise_frag "mul" %s, %a : (!frisk.fragment<vector<4xf32>, {}>, !frisk.fragment<vector<4xf32>, {}>) -> !frisk.fragment<vector<4xf32>, {}>
    %out = frisk.insert_frag %u, %t1[%c0] : <vector<4xf32>, {}> into vector<8xf32>
    affine.yield %out : vector<8xf32>
  }
  return %r : vector<8xf32>
}''')
        self.assertNotIn('frisk.extract_frag', ir)
        multiply = next(o['id'] for o in p['operations'] if o['name'] == 'frisk.elementwise_frag')
        self.assertFalse(any(e['dst'] == multiply for e in p['dependencies']))

    def test_narrow_alu_results_are_not_assumed_packed(self):
        p, _ = self.export('''func.func @probe(%x: !frisk.fragment<vector<4xf32>, {}>) -> !frisk.fragment<vector<4xf16>, {}> {
  %z = arith.constant dense<0.0> : vector<4xf16>
  %init = frisk.pack_frag %z : vector<4xf16> -> <vector<4xf16>, {}>
  %r = affine.for %i = 0 to 4 iter_args(%v = %init) -> !frisk.fragment<vector<4xf16>, {}> {
    %n = frisk.elementwise_frag "cast" %x : (!frisk.fragment<vector<4xf32>, {}>) -> !frisk.fragment<vector<4xf16>, {}>
    affine.yield %n : !frisk.fragment<vector<4xf16>, {}>
  }
  return %r : !frisk.fragment<vector<4xf16>, {}>
}''')
        self.assertEqual(p['register_liveness']['contract'], 'fragment-origin-words-v2')
        self.assertEqual(sum(v['words'] for v in p['register_liveness']['values']), 4)

    def test_partial_overlap_and_dynamic_slices_keep_dependencies(self):
        source = '''func.func @probe(%x: vector<8xf32>, %b: !frisk.fragment<vector<2xf32>, {}>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = affine.for %i = 0 to 2 iter_args(%v = %x) -> vector<8xf32> {
    %t = frisk.insert_frag %b, %v[%c1] : <vector<2xf32>, {}> into vector<8xf32>
    %s = frisk.extract_frag %t[POSITION] : vector<8xf32> -> <vector<4xf32>, {}>
    %u = frisk.elementwise_frag "mul" %s, %s : (!frisk.fragment<vector<4xf32>, {}>, !frisk.fragment<vector<4xf32>, {}>) -> !frisk.fragment<vector<4xf32>, {}>
    %out = frisk.insert_frag %u, %t[%c0] : <vector<4xf32>, {}> into vector<8xf32>
    affine.yield %out : vector<8xf32>
  }
  return %r : vector<8xf32>
}'''
        for position in ('%c0', '%i'):
            p, ir = self.export(source.replace('POSITION', position))
            self.assertIn('frisk.extract_frag', ir)
            extract = next(o['id'] for o in p['operations'] if o['name'] == 'frisk.extract_frag')
            self.assertTrue(any(e['src'] == 0 and e['dst'] == extract for e in p['dependencies']))

    def problem(self):
        return dict(operations=[{}] * 4, register_liveness=dict(
            contract='fragment-origin-words-v1', supported=True, invariant_words=3,
            values=[dict(producer=0, words=4, users=[dict(op=2, distance=0)]),
                    dict(producer=1, words=5, users=[dict(op=3, distance=0)])]))

    def feasible(self, p, stages, orders, budget, reserve=2, stride=None):
        model = cp_model.CpModel()
        add_register_constraint(model, p, stages, orders, stride=stride or len(stages),
                                stage_limit=max(stages) + 1, budget=budget, reserve=reserve)
        solver = cp_model.CpSolver()
        return solver.solve(model) in (cp_model.FEASIBLE, cp_model.OPTIMAL)

    def schedule(self, stages, orders):
        return dict(operations=[dict(id=i, stage=s, order=o)
                                for i, (s, o) in enumerate(zip(stages, orders))])

    def test_budget_forces_shorter_lifetimes(self):
        p = self.problem()
        self.assertFalse(self.feasible(p, [0]*4, [0, 1, 2, 3], 10))
        self.assertTrue(self.feasible(p, [0]*4, [0, 2, 1, 3], 10))

    def test_register_seed_consumes_before_producing_another_live_value(self):
        p = self.problem()
        p['operations'] = [dict(kind='alu') for _ in range(4)]
        p['dependencies'] = [dict(src=0, dst=2, kind='ssa'), dict(src=1, dst=3, kind='ssa')]
        order = register_hint(p, [0]*4, list(range(4)), 10, 2)
        ranks = [order.index(i) for i in range(4)]
        self.assertTrue(analyze_registers(p, self.schedule([0]*4, ranks), budget=10, reserve=2)['within_budget'])

    def test_constraint_matches_independent_sweep_across_wrap_and_phase_gaps(self):
        p = self.problem()
        p['register_liveness']['values'][1]['users'] = [dict(op=3, distance=1), dict(op=4, distance=0)]
        for stages in ([0]*4, [0, 0, 1, 1]):
            for sequence in itertools.permutations(range(4)):
                orders = [sequence.index(i) for i in range(4)]
                if orders[0] >= orders[2] or orders[1] >= orders[3]:
                    continue
                report = analyze_registers(p, self.schedule(stages, orders), reserve=2)
                peak = report['estimated_peak_words']
                self.assertTrue(self.feasible(p, stages, orders, peak))
                self.assertFalse(self.feasible(p, stages, orders, peak - 1))
                # Nonuniform timing gaps must not change the peak.
                phases = [o * o + 3 * o for o in orders]
                self.assertTrue(self.feasible(p, stages, phases, peak, stride=40))
                self.assertFalse(self.feasible(p, stages, phases, peak - 1, stride=40))

    def test_missing_or_unsupported_information_is_rejected(self):
        for p in (dict(operations=[{}]*4), self.problem()):
            if 'register_liveness' in p:
                p['register_liveness']['supported'] = False
            with self.assertRaises(ValueError):
                self.feasible(p, [0]*4, list(range(4)), 250)

    def test_export_deduplicates_views_and_maps_loop_carriers(self):
        root = Path(__file__).resolve().parents[2]
        bridge = root / '3rd/deepgengraph/build/lib/Pipeline/FriskScheduleJSONTest'
        source = '''func.func @probe(%x: vector<4xf32>) -> vector<4xf32> {
  %r = affine.for %i = 0 to 4 iter_args(%a = %x) -> vector<4xf32> {
    %v = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
    %w = vector.shape_cast %v : vector<2x2xf32> to vector<4xf32>
    %b = arith.addf %a, %w : vector<4xf32>
    affine.yield %b : vector<4xf32>
  }
  return %r : vector<4xf32>
}'''
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory)/'input.mlir', Path(directory)/'problem.json'
            path.write_text(source)
            subprocess.run([str(bridge), str(path), '--fragment', '--loop=probe:0',
                            f'--export={output}'], check=True, capture_output=True)
            p = json.loads(output.read_text())
            r = p['register_liveness']
            self.assertTrue(r['supported'])
            self.assertEqual(r['invariant_words'], 0)
            self.assertEqual(sum(v['words'] for v in r['values']), 4)
            self.assertTrue(any(u['distance'] == 1 for v in r['values'] for u in v['users']))
            n = len(p['operations'])
            # Old and new accumulator coexist at add; shape casts add no data.
            report = analyze_registers(p, self.schedule([0]*n, list(range(n))), reserve=0)
            self.assertEqual(report['data_peak_words'], 8)


if __name__ == '__main__':
    unittest.main()
