"""End-to-end JSON/solver/materializer checks, including delayed-DMA CPU execution.

Run from the repository root: python -m unittest discover -s python/smt -v
Build FriskScheduleJSONTest first. No GPU is required.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import unittest

from cpsat_scheduler import solve
from cost_model import CostModel
from schedule_schema import baseline_schedule, read_problem, validate_schedule, write_json
from shm_liveness import analyze_shm

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "3rd/deepgengraph/build/lib/Pipeline/FriskScheduleJSONTest"


def fixture(trips=5, name="probe"):
    return f"""func.func @{name}(%src: memref<16xf32, 1>) -> f32 attributes {{thread_num = 64 : i32}} {{
  %zero = arith.constant 0.0 : f32
  %two = arith.constant 2.0 : f32
  %c0 = arith.constant 0 : index
  %shared = frisk.alloc_buffer {{scope = "shared", alignment = 16}} -> memref<1xf32, 3>
  %result = affine.for %i = 0 to {trips} iter_args(%acc = %zero) -> (f32) {{
    %view = memref.subview %src[%i] [1] [1] : memref<16xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
    %task = frisk.async_copy_gm_to_shm %view to %shared : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task : !frisk.async_task
    %x = memref.load %shared[%c0] : memref<1xf32, 3>
    %product = arith.mulf %acc, %two : f32
    %sum = arith.addf %product, %x : f32
    affine.yield %sum : f32
  }}
  return %result : f32
}}
"""


def main_function():
    return """func.func @main() -> i32 {
  %src = memref.alloc() : memref<16xf32, 1>
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  affine.for %i = 0 to 16 {
    %i32 = arith.index_cast %i : index to i32
    %f = arith.sitofp %i32 : i32 to f32
    memref.store %f, %src[%i] : memref<16xf32, 1>
  }
  %a = func.call @probe(%src) : (memref<16xf32, 1>) -> f32
  %b = func.call @reference(%src) : (memref<16xf32, 1>) -> f32
  %ok = arith.cmpf oeq, %a, %b : f32
  %status = arith.select %ok, %zero, %one : i32
  return %status : i32
}
"""


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        if not BRIDGE.exists():
            self.fail("build FriskScheduleJSONTest before running the integration suite")
        self.temp = tempfile.TemporaryDirectory(prefix="cpsat-test-")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def bridge(self, source, *options, success=True):
        result = subprocess.run([str(BRIDGE), str(source), *map(str, options)],
                                capture_output=True, text=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def export(self, source=None):
        path = self.work / "input.mlir"
        path.write_text(source or fixture())
        problem = self.work / "problem.json"
        self.bridge(path, f"--export={problem}", "--loop=probe:0",
                    f"--ir={self.work / 'before.mlir'}")
        return path, read_problem(problem)

    def apply(self, source, schedule, *, success=True):
        plan, output = self.work / "solution.json", self.work / "after.mlir"
        write_json(plan, schedule)
        result = self.bridge(source, f"--import={plan}", "--loop=probe:0",
                             f"--ir={output}", success=success)
        return output.read_text() if success else result.stderr

    def test_identity_replay(self):
        source, p = self.export()
        after = self.apply(source, baseline_schedule(p))
        self.assertEqual(after, (self.work / "before.mlir").read_text())

    def test_mytest_exports_fragments_after_layout_and_honors_sync_mode(self):
        base = ROOT / 'python/smt/baselines/noncausal'
        metadata = json.loads((base / 'provenance.json').read_text())
        for copy_mode in ('default', 'sync', 'legacy'):
            with self.subTest(copy_mode=copy_mode):
                work = self.work / copy_mode
                work.mkdir()
                flags = list(metadata['mytest_arguments_after_output'])
                env = dict(os.environ, **metadata['environment'])
                env.pop('FRISK_GM_TO_SHM_MODE', None)
                if copy_mode != 'default':
                    env['FRISK_GM_TO_SHM_MODE'] = copy_mode
                    flags[11:13] = ['legacy', 'split32']  # positional DMA pack/export
                    flags = [f for f in flags if not f.startswith(('--k-dma=', '--swizzle-k='))]
                    flags += ['--k-dma=128', '--swizzle-k=off']
                command = [str(ROOT / '3rd/deepgengraph/build/test/MyTest'),
                           str(base / 'input.mlir'), '0', str(work / 'unused.ll'), *flags,
                           f'--cpsat-export={work / "problem.json"}',
                           f'--cpsat-dump-ir={work / "base.mlir"}', '--cpsat-stop-after=1']
                result = subprocess.run(command, cwd=work, env=env, capture_output=True,
                                        text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr[-6000:])
                p = read_problem(work / 'problem.json')
                stats = p['dma_summary']
                count = 8 if copy_mode == 'default' else 0
                self.assertEqual(stats['direct_issues'], count)
                self.assertEqual(stats['direct_waits'], count)
                self.assertEqual(stats['nested_issues'], 0)
                self.assertEqual(p['phase'], 'frisk-fragment-after-layout')
                text = (work / 'base.mlir').read_text()
                self.assertIn('!frisk.fragment', text)
                self.assertIn('frisk.gemm_frag', text)
                self.assertNotIn('frisk.gemm(', text)
                self.assertIn('cpsat.unroll_all', text)
                if copy_mode == 'default':
                    self.assertIn('frisk.retained_words = 32', text)
                    self.assertEqual(p['register_liveness']['invariant_words'], 32)
                self.assertNotIn('frisk.pipeline_stage', text)
                self.assertTrue(all(o['name'] != 'affine.for' for o in p['operations']))
                self.assertTrue(all(b['bytes'] <= 16384 for b in p['buffers'] if b['memory_space'] == 3))

    def test_fragment_full_unroll_and_numeric_replay(self):
        fixture_file = ROOT / '3rd/deepgengraph/test/unittest/check_typed_fragments.py'
        source = runpy.run_path(str(fixture_file))['CPU']
        source = source.replace('%k = 0 to 2', '%k = 0 to 17').replace('3.0 : f32', '18.0 : f32')
        path, problem_path = self.work / 'fragment.mlir', self.work / 'fragment.json'
        path.write_text(source)
        self.bridge(path, '--fragment', '--loop=main:2', f'--export={problem_path}')
        p = read_problem(problem_path)
        self.assertEqual(p['phase'], 'frisk-fragment-after-layout')
        self.assertFalse(any(o['name'] == 'affine.for' for o in p['operations']))
        schedule = solve(p, time_limit=5)
        solution, reordered = self.work / 'solution.json', self.work / 'reordered.mlir'
        write_json(solution, schedule)
        self.bridge(path, '--fragment', '--loop=main:2', f'--import={solution}', f'--ir={reordered}')
        tool = ROOT / '3rd/deepgengraph/build/test/ThreadTilingTest'
        llvm = Path(os.environ.get('LLVM_BIN', '/data2/xsl/install/bin'))
        def run(*args):
            result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        native, unrolled, lowered, ll = (self.work / name for name in
            ('native.mlir', 'unrolled.mlir', 'llvm.mlir', 'cpu.ll'))
        run(tool, reordered, '--convert-fragment-to-thread', '--finalize-thread-tiling',
            '--canonicalize', '-o', native)
        run(llvm / 'mlir-opt', native,
            '--pass-pipeline=builtin.module(func.func(affine-loop-unroll{unroll-factor=-1 unroll-num-reps=4}),canonicalize)',
            '-o', unrolled)
        run(llvm / 'mlir-opt', unrolled, '--lower-affine', '--convert-scf-to-cf',
            '--convert-vector-to-llvm', '--convert-math-to-llvm', '--convert-arith-to-llvm',
            '--finalize-memref-to-llvm', '--convert-func-to-llvm', '--convert-cf-to-llvm',
            '--convert-ub-to-llvm', '--reconcile-unrealized-casts', '-o', lowered)
        run(llvm / 'mlir-translate', '--mlir-to-llvmir', lowered, '-o', ll)
        run(llvm / 'lli', ll)
        # Dynamic inner loops cannot be silently retained in full-unroll mode.
        dynamic = source.replace('%k = 0 to 17', '%k = 0 to affine_map<(d0) -> (d0)>(%j)')
        path.write_text(dynamic)
        error = self.bridge(path, '--fragment', '--loop=main:2', f'--export={problem_path}', success=False)
        self.assertIn('statically bounded', error.stderr)

    def test_intra_respects_load_latency_without_versions(self):
        source, p = self.export()
        schedule = solve(p, time_limit=5)
        validate_schedule(p, schedule)
        self.assertTrue(all(op["stage"] == 0 for op in schedule["operations"]))
        self.assertTrue(all(b["slots"] == 1 for b in schedule["buffers"]))
        # Source order may already hide the LDS latency with the independent
        # multiply; forcing a permutation is not a performance requirement.
        for edge in p['dependencies']:
            if edge['kind'] == 'ssa' and p['operations'][edge['src']]['kind'] == 'shm_reg':
                self.assertGreaterEqual(schedule['model']['phase'][edge['dst']] -
                                        schedule['model']['phase'][edge['src']],
                                        schedule['model']['latency'][edge['src']])
        after = self.apply(source, schedule)
        self.assertNotIn("pipeline_slot", after)
        if [o['order'] for o in schedule['operations']] != list(range(len(p['operations']))):
            self.assertIn('cpsat.mode = "intra"', after)
        else:
            self.assertEqual(after, (self.work / 'before.mlir').read_text())

    def test_pipeline_versions_and_short_loops(self):
        # Reuse the repository's delayed-DMA runtime: the memory write happens
        # only at wait(task), so incorrect token/slot rotation changes numerics.
        sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test/unittest"))
        import check_auto_pipeline as cpu
        cpu.LLVM = Path(os.environ.get("LLVM_BIN", "/data2/xsl/install/bin"))
        for trips in (0, 1, 2, 7):
            with self.subTest(trips=trips):
                source, p = self.export("module {\n" + fixture(trips) + fixture(trips, "reference") + main_function() + "}\n")
                if trips == 0:
                    self.assertFalse(p["loop"]["pipeline_eligible"])
                    with self.assertRaises(ValueError): solve(p, mode="pipeline")
                    schedule = solve(p, time_limit=5)
                else:
                    self.assertTrue(p["loop"]["pipeline_eligible"], p["loop"])
                    schedule = solve(p, mode="pipeline", time_limit=5)
                    self.assertTrue(any(b["slots"] == 2 for b in schedule["buffers"]))
                after = self.apply(source, schedule)
                if trips:
                    self.assertIn("pipeline.slots = 2", after)
                    self.assertEqual(after.count("frisk.alloc_buffer"), 3)
                cpu.execute_cpu(after, self.work, f"numeric-{trips}")

    def test_fragment_memref_alloc_cross_iteration_numeric(self):
        sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test/unittest"))
        import check_auto_pipeline as cpu
        cpu.LLVM = Path(os.environ.get("LLVM_BIN", "/data2/xsl/install/bin"))
        source = "module {\n" + fixture(7) + fixture(7, "reference") + main_function() + "}\n"
        source = source.replace('frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>',
                                'memref.alloc() alignment = 16 : memref<1xf32, 3>')
        path, exported = self.work / 'input.mlir', self.work / 'problem.json'
        path.write_text(source)
        self.bridge(path, '--fragment', '--loop=probe:0', f'--export={exported}')
        p = read_problem(exported)
        schedule = solve(p, mode='pipeline', require_cross_iteration=True, time_limit=5)
        plan, output = self.work / 'solution.json', self.work / 'after.mlir'
        write_json(plan, schedule)
        self.bridge(path, '--fragment', '--loop=probe:0', f'--import={plan}', f'--ir={output}')
        after = output.read_text()
        self.assertIn('pipeline.slots = 2', after)
        cpu.execute_cpu(after, self.work, 'fragment-memref-pipeline')

    def test_periodic_pool_numeric_and_forged_overlap(self):
        sys.path.insert(0, str(ROOT / '3rd/deepgengraph/test/unittest'))
        import check_auto_pipeline as cpu
        cpu.LLVM = Path(os.environ.get('LLVM_BIN', '/data2/xsl/install/bin'))
        tool = ROOT / '3rd/deepgengraph/build/test/ThreadTilingTest'
        for trips, lower, step in ((1, 0, 1), (2, 0, 1), (7, 0, 1), (8, 0, 1), (4, 1, 2)):
            with self.subTest(trips=trips, lower=lower, step=step):
                upper = lower + trips * step
                source = 'module {\n' + fixture(upper) + fixture(upper, 'reference') + main_function() + '}\n'
                source = source.replace(f'affine.for %i = 0 to {upper} iter_args',
                                        f'affine.for %i = {lower} to {upper} step {step} iter_args')
                source = source.replace('frisk.alloc_buffer {scope = "shared", alignment = 16} ->',
                                        'memref.alloc() alignment = 16 :')
                source = source.replace('  %result = affine.for',
                    '  %other = memref.alloc() alignment = 16 : memref<1xf32, 3>\n  %result = affine.for')
                source = source.replace('    affine.yield %sum : f32', '''    %task_b = frisk.async_copy_gm_to_shm %view to %other : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task_b : !frisk.async_task
    %y = memref.load %other[%c0] : memref<1xf32, 3>
    %sum_b = arith.addf %sum, %y : f32
    affine.yield %sum_b : f32''')
                source = source.replace('memref<1xf32', 'memref<4xf32').replace('%src[%i] [1] [1]', '%src[%i] [4] [1]')
                if trips == 2:
                    # An invariant retained operand's staging storage dies
                    # before the periodic pool starts. It must reuse that pool
                    # without adding another 16 bytes to the physical budget.
                    source = source.replace('  %shared = memref.alloc()', '''  %prelude = memref.alloc() alignment = 16 : memref<4xf32, 3>
  %qview = memref.subview %src[0] [4] [1] : memref<16xf32, 1> to memref<4xf32, strided<[1]>, 1>
  %qtask = frisk.async_copy_gm_to_shm %qview to %prelude : memref<4xf32, strided<[1]>, 1>, memref<4xf32, 3> -> !frisk.async_task
  frisk.wait_buffer_ready %qtask : !frisk.async_task
  %initial = memref.load %prelude[%c0] : memref<4xf32, 3>
  %shared = memref.alloc()''')
                    source = source.replace('iter_args(%acc = %zero)', 'iter_args(%acc = %initial)')
                path, exported = self.work / 'pool.mlir', self.work / 'pool.json'
                path.write_text(source)
                self.bridge(path, '--fragment', '--loop=probe:0', f'--export={exported}')
                p = read_problem(exported)
                plan = solve(p, mode='pipeline', require_cross_iteration=True, lds_budget=32, time_limit=5)
                self.assertEqual(plan['storage']['pool_bytes'], 32)
                solution, output, packed = (self.work / name for name in ('pool-plan.json', 'pool-after.mlir', 'pool-packed.mlir'))
                write_json(solution, plan)
                self.bridge(path, '--fragment', '--loop=probe:0', f'--import={solution}', f'--ir={output}')
                result = subprocess.run([str(tool), str(output), '--lower-frisk-fragments', '-o', str(packed)],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr[-4000:])
                text = packed.read_text()
                self.assertIn('memref<32xi8, 3>', text)
                self.assertIn('cpsat.physical_pool', text)
                if trips == 2:
                    self.assertIn('1 prelude allocations reused', result.stderr)
                # CPU fixture is one thread; block barriers have no effect.
                text = re.sub(r'^.*gpu.barrier.*\n', '', text, flags=re.M)
                cpu.execute_cpu(text, self.work, f'pooled-{trips}')
                for b in plan['storage']['buffers']:
                    b['offsets'] = [0, 0]
                write_json(solution, plan)
                error = self.bridge(path, '--fragment', '--loop=probe:0', f'--import={solution}', success=False)
                self.assertIn('overlapping version lifetimes', error.stderr)

    def test_rejects_forged_or_stale_solutions(self):
        source, p = self.export()
        original = baseline_schedule(p)
        for name in ("fingerprint", "duplicate", "dependency", "slots", "anchor", "stage_limit"):
            with self.subTest(name=name):
                bad = deepcopy(original)
                if name == "fingerprint": bad["fingerprint"] = "0" * 64
                elif name == "duplicate": bad["operations"][1]["id"] = 0
                elif name == "dependency":
                    bad["operations"][1]["order"], bad["operations"][2]["order"] = 2, 1
                elif name == "slots": bad["buffers"][0]["slots"] = 2
                elif name == "stage_limit": bad["operations"][0]["stage"] = 2
                else:
                    bad["mode"] = "pipeline"
                    bad["operations"][-1]["stage"] = 1
                self.assertIn("invalid CP-SAT schedule", self.apply(source, bad, success=False))

    def test_lds_budget_and_no_solution(self):
        _, p = self.export()
        schedule = solve(p, mode="pipeline", lds_budget=4, time_limit=5)
        self.assertTrue(all(b["slots"] == 1 for b in schedule["buffers"]))
        with self.assertRaises(RuntimeError):
            solve(p, lds_budget=0, time_limit=2)

    def test_alias_and_unknown_region_are_conservative(self):
        source = fixture().replace("    %product =", "    func.call @opaque() : () -> ()\n    %product =")
        _, p = self.export("func.func private @opaque()\n" + source)
        self.assertFalse(p["loop"]["pipeline_eligible"])
        with self.assertRaises(ValueError): solve(p, mode="pipeline")
        schedule = solve(p, time_limit=5)
        validate_schedule(p, schedule)
        self.assertTrue(any(e["kind"] == "fence" for e in p["dependencies"]))

    def test_cost_scaling_is_advisory(self):
        _, p = self.export()
        a, b = solve(p, costs=CostModel(dma=10), time_limit=5), solve(p, costs=CostModel(dma=20), time_limit=5)
        validate_schedule(p, a); validate_schedule(p, b)
        self.assertGreaterEqual(b["model"]["period"], a["model"]["period"])

    def test_dma_requires_wait_before_access(self):
        invalid = fixture().replace(
            "    frisk.wait_buffer_ready %task : !frisk.async_task\n", "").replace(
            "    %product =", "    frisk.wait_buffer_ready %task : !frisk.async_task\n    %product =")
        path = self.work / "invalid.mlir"
        path.write_text(invalid)
        result = self.bridge(path, f"--export={self.work / 'bad.json'}", success=False)
        self.assertIn("access before DMA completion", result.stderr)
        path.write_text(fixture().replace("    frisk.wait_buffer_ready %task : !frisk.async_task\n", ""))
        self.assertIn("one local wait", self.bridge(path, "--loop=probe:0", success=False).stderr)

    def test_unknown_descriptor_alias_and_nonlocal_wait(self):
        # The selected descriptor could alias the owned shared buffer. The
        # exporter must not use fresh(shared) to declare them disjoint.
        text = fixture().replace("  %result =", "  %yes = arith.constant true\n  %result =")
        text = text.replace("    %product =", """    %alias = arith.select %yes, %shared, %shared : memref<1xf32, 3>
    memref.store %x, %shared[%c0] : memref<1xf32, 3>
    %aliased = memref.load %alias[%c0] : memref<1xf32, 3>
    %product =""")
        _, p = self.export(text)
        stores = [o['id'] for o in p['operations'] if o['name'] == 'memref.store']
        load = [o['id'] for o in p['operations'] if o['name'] == 'memref.load'][-1]
        self.assertTrue(any(e['src'] == stores[0] and e['dst'] == load and e['kind'] == 'raw'
                            for e in p['dependencies']))
        text = fixture().replace("  %result =", "  %done = frisk.async_task_completed : !frisk.async_task\n  %result =")
        text = text.replace("    %product =", "    frisk.wait_buffer_ready %done : !frisk.async_task\n    %product =")
        _, p = self.export(text)
        self.assertFalse(p['loop']['pipeline_eligible'])
        fences = [o['id'] for o in p['operations'] if o['opaque']]
        self.assertEqual(len(fences), 1)
        self.assertTrue(any(e['kind'] == 'fence' and e['src'] == fences[0] for e in p['dependencies']))

    def test_pipeline_multiple_ssa_carriers(self):
        sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test/unittest"))
        import check_auto_pipeline as cpu
        cpu.LLVM = Path(os.environ.get("LLVM_BIN", "/data2/xsl/install/bin"))

        def pair(name):
            return (fixture(7, name)
                .replace("-> f32 attributes", "-> (f32, index) attributes")
                .replace("%result = affine.for", "%result:2 = affine.for")
                .replace("iter_args(%acc = %zero) -> (f32)", "iter_args(%acc = %zero, %last = %c0) -> (f32, index)")
                .replace("affine.yield %sum : f32", "affine.yield %sum, %i : f32, index")
                .replace("return %result : f32", "return %result#0, %result#1 : f32, index"))

        main = (main_function().replace("%a = func.call", "%a:2 = func.call")
            .replace("%b = func.call", "%b:2 = func.call")
            .replace("-> f32", "-> (f32, index)")
            .replace("%ok = arith.cmpf oeq, %a, %b : f32", """%ok1 = arith.cmpf oeq, %a#0, %b#0 : f32
  %ok2 = arith.cmpi eq, %a#1, %b#1 : index
  %ok = arith.andi %ok1, %ok2 : i1"""))
        source, p = self.export("module {\n" + pair("probe") + pair("reference") + main + "}\n")
        schedule = solve(p, mode="pipeline", time_limit=5)
        self.assertTrue(any(b['slots'] == 2 for b in schedule['buffers']))
        # Emit the next tile's address/issue last. A direct yield of %i must
        # still refer to the current tile, not the final statement's mapping.
        for op, order in zip(schedule['operations'], (4, 5, 0, 1, 2, 3)):
            op['order'] = order
        validate_schedule(p, schedule)
        after = self.apply(source, schedule)
        cpu.execute_cpu(after, self.work, "multiple-ssa")

    def test_small_nested_loop_unroll_pipeline_and_cpu_recurrence(self):
        sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test/unittest"))
        import check_auto_pipeline as cpu
        cpu.LLVM = Path(os.environ.get("LLVM_BIN", "/data2/xsl/install/bin"))

        def nested(name, count):
            bounds = '5 to 37 step 2' if count == 16 else f'0 to {count}'
            return fixture(7, name).replace("    affine.yield %sum : f32", f"""    %inner = affine.for %j = {bounds} iter_args(%carry = %sum) -> (f32) {{
      %next = arith.addf %carry, %x : f32
      affine.yield %next : f32
    }}
    affine.yield %inner : f32""")

        for count in (0, 1, 2, 16, 17):
            with self.subTest(count=count):
                source, original = self.export("module {\n" + nested("probe", count) +
                                               nested("reference", count) + main_function() + "}\n")
                self.assertFalse(original['loop']['pipeline_eligible'])
                output = self.work / 'unrolled.json'
                self.bridge(source, f'--export={output}', '--loop=probe:0',
                            '--unroll-max-trip-count=16')
                problem = read_problem(output)
                self.assertEqual(problem['loop']['pipeline_eligible'], count <= 16)
                if count > 16:
                    continue
                plan = solve(problem, mode='pipeline', time_limit=5)
                self.assertTrue(any(b['slots'] == 2 for b in plan['buffers']))
                schedule = self.work / 'unrolled.schedule.json'
                after = self.work / 'unrolled.mlir'
                write_json(schedule, plan)
                self.bridge(source, f'--import={schedule}', '--loop=probe:0',
                            '--unroll-max-trip-count=16', f'--ir={after}')
                cpu.execute_cpu(after.read_text(), self.work, f'unrolled-{count}')
                # Import must apply the same preparation, not a stale original graph.
                self.assertIn('fingerprint', self.bridge(source, f'--import={schedule}',
                                                       '--loop=probe:0', success=False).stderr)

    def test_require_cross_iteration_rejects_intra_and_single_slot(self):
        _, problem = self.export()
        with self.assertRaises(ValueError):
            solve(problem, require_cross_iteration=True)
        with self.assertRaises(RuntimeError):
            solve(problem, mode='pipeline', max_slots=1, require_cross_iteration=True, time_limit=2)
        plan = solve(problem, mode='pipeline', require_cross_iteration=True, time_limit=5)
        self.assertTrue(any(b['slots'] == 2 for b in plan['buffers']))

    def test_nonversioned_scratch_allocation_cannot_advance_alone(self):
        source = fixture().replace('    %product =', '''    %scratch = frisk.alloc_buffer {scope = "shared"} -> memref<1xf32, 3>
    memref.store %x, %scratch[%c0] : memref<1xf32, 3>
    %product =''')
        _, problem = self.export(source)
        allocations = [o for o in problem['operations'] if o['kind'] == 'alloc']
        self.assertEqual(len(allocations), 1)
        self.assertTrue(allocations[0]['pipeline_anchor'])

    def test_unroll_budget_and_dynamic_bounds_remain_conservative(self):
        nested = fixture().replace('    %product =', '''    affine.for %j = 0 to 16 {
      memref.store %x, %shared[%c0] : memref<1xf32, 3>
    }
    %product =''')
        source, _ = self.export(nested)
        output = self.work / 'budget.json'
        self.bridge(source, f'--export={output}', '--loop=probe:0',
                    '--unroll-max-trip-count=16', '--unroll-max-expanded-ops=0')
        self.assertFalse(read_problem(output)['loop']['pipeline_eligible'])
        source.write_text(nested.replace('to 16 {', 'to affine_map<(d0) -> (d0)>(%i) {'))
        self.bridge(source, f'--export={output}', '--loop=probe:0', '--unroll-max-trip-count=16')
        self.assertFalse(read_problem(output)['loop']['pipeline_eligible'])

    def test_unroll_normalizes_multiple_vector_carriers_with_numeric_check(self):
        sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test/unittest"))
        import check_auto_pipeline as cpu
        cpu.LLVM = Path(os.environ.get("LLVM_BIN", "/data2/xsl/install/bin"))
        probe = '''func.func @probe(%src: memref<16xf32, 1>) -> f32 attributes {thread_num = 64 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant dense<0.0> : vector<1xf32>
  %two = arith.constant 2.0 : f32
  %shared = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
  %ta = frisk.alloc_buffer {scope = "local"} -> memref<1xf32>
  %tb = frisk.alloc_buffer {scope = "local"} -> memref<1xf32>
  %result:2 = affine.for %i = 0 to 7 iter_args(%a = %zero, %b = %zero) -> (vector<1xf32>, vector<1xf32>) {
    %view = memref.subview %src[%i] [1] [1] : memref<16xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
    %task = frisk.async_copy_gm_to_shm %view to %shared : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task : !frisk.async_task
    %x = memref.load %shared[%c0] : memref<1xf32, 3>
    %av = vector.extract %a[0] : f32 from vector<1xf32>
    %bv = vector.extract %b[0] : f32 from vector<1xf32>
    %product = arith.mulf %av, %two : f32
    %nexta = arith.addf %product, %x : f32
    %nextb = arith.addf %bv, %nexta : f32
    memref.store %nexta, %ta[%c0] : memref<1xf32>
    memref.store %nextb, %tb[%c0] : memref<1xf32>
    frisk.copy %ta to %a [affine_map<() -> (1)>] : memref<1xf32>, vector<1xf32>
    frisk.copy %tb to %b [affine_map<() -> (1)>] : memref<1xf32>, vector<1xf32>
    affine.yield %a, %b : vector<1xf32>, vector<1xf32>
  }
  %av = vector.extract %result#0[0] : f32 from vector<1xf32>
  %bv = vector.extract %result#1[0] : f32 from vector<1xf32>
  %sum = arith.addf %av, %bv : f32
  return %sum : f32
}
'''
        reference = probe.replace('@probe', '@reference')
        for name in ('a', 'b'):
            reference = reference.replace(
                f'frisk.copy %t{name} to %{name} [affine_map<() -> (1)>] : memref<1xf32>, vector<1xf32>',
                f'%v{name} = vector.load %t{name}[%c0] : memref<1xf32>, vector<1xf32>')
        reference = reference.replace('affine.yield %a, %b', 'affine.yield %va, %vb')
        for unsupported in (
            probe.replace('frisk.copy %ta to %a [affine_map<() -> (1)>]',
                          'frisk.copy %ta to %a [affine_map<() -> (0)>]'),
            probe.replace('    affine.yield %a, %b',
                          '    %late = vector.extract %a[0] : f32 from vector<1xf32>\n'
                          '    affine.yield %a, %b'),
        ):
            source, _ = self.export(unsupported)
            output = self.work / 'unsupported.json'
            self.bridge(source, f'--export={output}', '--loop=probe:0', '--unroll-max-trip-count=16')
            self.assertFalse(read_problem(output)['loop']['pipeline_eligible'])
        source, p = self.export('module {\n' + probe + reference + main_function() + '}')
        self.assertFalse(p['loop']['pipeline_eligible'])
        prepared = self.work / 'prepared.json'
        self.bridge(source, f'--export={prepared}', '--loop=probe:0', '--unroll-max-trip-count=16')
        p = read_problem(prepared)
        self.assertTrue(p['loop']['pipeline_eligible'])
        plan = solve(p, mode='pipeline', time_limit=5)
        self.assertTrue(any(b['slots'] == 2 for b in plan['buffers']))
        schedule, after = self.work / 'vector.json', self.work / 'vector.mlir'
        write_json(schedule, plan)
        self.bridge(source, f'--import={schedule}', '--loop=probe:0',
                    '--unroll-max-trip-count=16', f'--ir={after}')
        text = after.read_text()
        self.assertIn('cpsat.normalized_carriers = 2', text)
        # Lower only the full one-element register copy in this CPU fixture.
        def load(match):
            result, src = match.groups()
            zero = '%cpu_zero_' + result[1:]
            return (f'{zero} = arith.constant 0 : index\n'
                    f'{result} = vector.load {src}[{zero}] : memref<1xf32>, vector<1xf32>')
        text = re.sub(r'(%\w+) = frisk.copy_to_reg (%\w+) \[#[\w]+\] : memref<1xf32> -> vector<1xf32>',
                      load, text)
        after.write_text(cpu.cpu_ir(text))
        low, ll = self.work / 'vector.low.mlir', self.work / 'vector.ll'
        cpu.run([cpu.LLVM / 'mlir-opt', after, '--expand-strided-metadata', '--lower-affine',
                 '--convert-scf-to-cf', '--convert-vector-to-llvm', '--finalize-memref-to-llvm',
                 '--convert-arith-to-llvm', '--convert-func-to-llvm', '--convert-cf-to-llvm',
                 '--reconcile-unrealized-casts', '-o', low])
        cpu.run([cpu.LLVM / 'mlir-translate', '--mlir-to-llvmir', low, '-o', ll])
        runtime, linked = self.work / 'runtime.ll', self.work / 'linked.ll'
        runtime.write_text(cpu.TASK_RUNTIME)
        cpu.run([cpu.LLVM / 'llvm-link', ll, runtime, '-S', '-o', linked])
        cpu.run([cpu.LLVM / 'lli', linked])

    @unittest.skipUnless(os.environ.get("CPSAT_TEST_ATTENTION") == "1",
                         "set CPSAT_TEST_ATTENTION=1 for full attention LLVM lowering")
    def test_attention_solved_plans_lower_to_verified_llvm(self):
        # Re-export at the current Fragment boundary; saved Base JSON is historical.
        mytest = ROOT / "3rd/deepgengraph/build/test/MyTest"
        llvm_bin = Path(os.environ.get("LLVM_BIN", "/data2/xsl/install/bin"))
        for case in ("causal", "noncausal"):
            with self.subTest(case=case):
                base = ROOT / "python/smt/baselines" / case
                meta = json.loads((base / "provenance.json").read_text())
                work = self.work / case
                work.mkdir()
                problem_path = work / 'problem.json'
                command = [str(mytest), str(base / 'input.mlir'), '0', str(work / 'unused.ll'),
                           *meta['mytest_arguments_after_output'], f'--cpsat-export={problem_path}',
                           '--cpsat-stop-after=1']
                with (work / 'export.log').open('w') as log:
                    result = subprocess.run(command, cwd=work, env=dict(os.environ, **meta['environment']),
                                            stdout=log, stderr=subprocess.STDOUT, timeout=300)
                self.assertEqual(result.returncode, 0, (work / 'export.log').read_text()[-8000:])
                problem = read_problem(problem_path)
                plan = solve(problem, time_limit=120, workers=8, lds_budget=16384, shm_check='constraint')
                self.assertTrue(plan['model']['shm_liveness']['within_budget'])
                solution, llvm_file = work / "solution.json", work / "kernel.ll"
                write_json(solution, plan)
                cmd = [str(mytest), str(base / "input.mlir"), "0", str(llvm_file)]
                cmd += meta['mytest_arguments_after_output'] + [f"--cpsat-import={solution}"]
                with (work / "compile.log").open("w+") as log:
                    result = subprocess.run(cmd, cwd=work, env=dict(os.environ, **meta['environment']),
                                            stdout=log, stderr=subprocess.STDOUT, timeout=300)
                    log.seek(0)
                    self.assertEqual(result.returncode, 0, log.read()[-8000:])
                result = subprocess.run([str(llvm_bin / 'llvm-as'), str(llvm_file), '-o', os.devnull],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
