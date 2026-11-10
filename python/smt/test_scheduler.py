"""End-to-end JSON/solver/materializer checks, including delayed-DMA CPU execution.

Run from the repository root: python -m unittest discover -s python/smt -v
Build FriskScheduleJSONTest first. No GPU is required.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
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

    def test_mytest_exports_dma_before_fragment_and_honors_sync_mode(self):
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
                    flags[12:14] = ['legacy', 'split32']  # positional DMA pack/export
                command = [str(ROOT / '3rd/deepgengraph/build/test/MyTest'),
                           str(base / 'input.mlir'), '0', str(work / 'unused.ll'), *flags,
                           f'--cpsat-export={work / "problem.json"}',
                           f'--cpsat-dump-ir={work / "base.mlir"}', '--cpsat-stop-after=1']
                result = subprocess.run(command, cwd=work, env=env, capture_output=True,
                                        text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr[-6000:])
                p = read_problem(work / 'problem.json')
                stats = p['dma_summary']
                count = 4 if copy_mode == 'default' else 0
                self.assertEqual(stats['direct_issues'], count)
                self.assertEqual(stats['direct_waits'], count)
                self.assertEqual(stats['nested_issues'], 2 if count else 0)
                if not count: self.assertEqual(stats['synchronous_gm_shm'], 6)
                text = (work / 'base.mlir').read_text()
                self.assertNotIn('!frisk.fragment', text)
                self.assertNotIn('frisk.gemm_frag', text)
                self.assertIn('frisk.gemm(', text)

    def test_intra_reorders_without_versions(self):
        source, p = self.export()
        schedule = solve(p, time_limit=5)
        validate_schedule(p, schedule)
        self.assertTrue(all(op["stage"] == 0 for op in schedule["operations"]))
        self.assertTrue(all(b["slots"] == 1 for b in schedule["buffers"]))
        self.assertNotEqual([o["order"] for o in schedule["operations"]], list(range(len(p["operations"]))))
        after = self.apply(source, schedule)
        self.assertNotIn("pipeline_slot", after)
        self.assertIn('cpsat.mode = "intra"', after)

    def test_pipeline_versions_and_short_loops(self):
        # Reuse the repository's delayed-DMA runtime: the memory write happens
        # only at wait(task), so incorrect token/slot rotation changes numerics.
        sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test"))
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
        sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test"))
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

    @unittest.skipUnless(os.environ.get("CPSAT_TEST_ATTENTION") == "1",
                         "set CPSAT_TEST_ATTENTION=1 for full attention LLVM lowering")
    def test_attention_solved_plans_lower_to_verified_llvm(self):
        # Exercises legacy late optimizations after a different Base order,
        # including LazySoftmaxBase splitting the loop latch (a PHI regression).
        mytest = ROOT / "3rd/deepgengraph/build/test/MyTest"
        llvm_bin = Path(os.environ.get("LLVM_BIN", "/data2/xsl/install/bin"))
        for case in ("causal", "noncausal"):
            with self.subTest(case=case):
                base = ROOT / "python/smt/baselines" / case
                meta = json.loads((base / "provenance.json").read_text())
                problem = read_problem(base / "problem.json")
                budget = analyze_shm(problem, baseline_schedule(problem))['peak_bytes']
                plan = solve(problem, time_limit=15, lds_budget=budget, shm_check='constraint')
                self.assertTrue(plan['model']['shm_liveness']['within_budget'])
                work = self.work / case
                work.mkdir()
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
