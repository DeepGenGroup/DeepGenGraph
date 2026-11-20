#!/usr/bin/env python3
"""ThreadIR RTZ selection, typed packing, lifetime boundaries and LLVM lowering."""
from pathlib import Path
import re
import subprocess
import unittest

from check_thread_fragment_schedules import softmax, graph

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'build/test/ThreadTilingTest'
LEGACY = TOOL.with_name('LegacyLLVMExportTest')
MYTEST = TOOL.with_name('MyTest')
LLVM = Path('/data2/xsl/install/bin')
PASS = '--frisk-thread-softmax-cvt-pack'
ON = PASS + '=mode=rtz native-exp2=true'


def source(count=4, typed=False):
    text = softmax(count).replace(
        '-> (vector<4xf32>, f32) {',
        '-> (vector<4xf32>, f32) attributes {thread_num = 64 : i32} {')
    # An unrelated output conversion must remain RNE, even in the same kernel.
    text = text.replace('return %m', '%output = arith.truncf %seed : vector<4xf32> to vector<4xf16>\nreturn %m')
    text = text.replace('-> (vector<4xf32>, f32)', '-> (vector<4xf32>, f32, vector<4xf16>)')
    text = re.sub(r'(return %m\d+, %r\d+) : vector<4xf32>, f32',
                  r'\1, %output : vector<4xf32>, f32, vector<4xf16>', text)
    if typed:
        text = re.sub(
            r'llvm.inline_asm has_side_effects convergent \{test.id = "(\d+)"\} '
            r'"([^"]+)", "([^"]+)" ([^:]+) : '
            r'\(vector<4xf16>, vector<4xf16>, vector<4xf32>\) -> vector<4xf32>',
            r'frisk.warp_mma_rr \4 {inst_name = "\2", inst_constraints = "\3", test.id = "\1"} '
            r': vector<4xf16>, vector<4xf16>, vector<4xf32> -> vector<4xf32>', text)
    return text


def run(text, *passes, ok=True):
    p = subprocess.run([str(TOOL), '--verify-each', *passes], input=text,
                       text=True, capture_output=True, timeout=120)
    assert (p.returncode == 0) == ok, p.stderr
    return p.stdout if ok else p.stderr


def command(args, text):
    p = subprocess.run(list(map(str, args)), input=text, text=True,
                       capture_output=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return p.stdout


class SoftmaxCvtPackTest(unittest.TestCase):
    def test_only_pv_edge_and_original_dag(self):
        for typed in (False, True):
            for schedule in ((), ('--frisk-thread-schedule-softmax-fragments=mode=pv',)):
                with self.subTest(typed=typed, schedule=schedule):
                    before = run(source(typed=typed), *schedule)
                    after = run(before, ON)
                    self.assertEqual(after.count('frisk.convert_pack '), 4)
                    self.assertEqual(after.count('arith.truncf '), 1)
                    self.assertEqual(after.count('math.exp2 '), 4)
                    self.assertIn('frisk.softmax_cvt_pack = "rtz"', after)
                    # Compare all MMA/return expression DAGs after erasing only
                    # the intentional conversion change. Includes f32 sums.
                    restored = re.sub(
                        r'frisk.convert_pack (%\w+) \{rounding = "rtz"\} : vector<4xf32> -> vector<4xf16>',
                        r'arith.truncf \1 : vector<4xf32> to vector<4xf16>', after)
                    self.assertEqual(graph(before), graph(restored))
                    self.assertEqual(run(after, ON), after)

    def test_explicit_opt_in_and_native_math(self):
        text = source()
        before = run(text)
        for option in (PASS, PASS + '=mode=off native-exp2=true', PASS + '=mode=rtz'):
            self.assertEqual(run(text, option), before)
        for mode in ('rne', 'RTZ', 'rtz-more'):
            self.assertIn('mode=off|rtz', run(text, PASS + '=mode=' + mode, ok=False))

    def test_reject_unsafe_edges(self):
        text = source()
        cases = {
            'ordinary_function': text.replace('attributes {thread_num = 64 : i32}', ''),
            'strictfp': text.replace('thread_num = 64 : i32', 'thread_num = 64 : i32, strictfp'),
            'optnone': text.replace('thread_num = 64 : i32', 'thread_num = 64 : i32, llvm.optnone'),
            'passthrough_strictfp': text.replace('thread_num = 64 : i32', 'thread_num = 64 : i32, llvm.passthrough = ["strictfp"]'),
            'not_exp2': text.replace('math.exp2', 'math.sqrt'),
            'wrong_width': text.replace('4xf', '8xf'),
            'wrong_mma': text.replace('v_mmac_f32_16x16x16_f16', 'v_unknown'),
            'explicit_rounding': re.sub(r'arith.truncf (%\w+) :', r'arith.truncf \1 upward :', text),
            'b_operand': re.sub(r'%p(\d), %b,', r'%b, %p\1,', text),
            'a_and_b': re.sub(r'%p(\d), %b,', r'%p\1, %p\1,', text),
            'escaped_half': text.replace('%output =', '\n'.join(
                f'%escape{i} = vector.extract %p{i}[0] : f16 from vector<4xf16>'
                for i in range(4)) + '\n%output ='),
        }
        for name, rejected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(run(rejected, ON), run(rejected))
        conflict = text.replace('thread_num = 64 : i32',
                                'thread_num = 64 : i32, frisk.softmax_cvt_pack = "rne"')
        self.assertIn('conflicting softmax cvt pack mode', run(conflict, ON, ok=False))

    def test_mixed_safe_and_unsafe(self):
        text = source().replace('%output =',
            '%escape = vector.extract %p0[0] : f16 from vector<4xf16>\n%output =')
        result = run(text, ON)
        self.assertEqual(result.count('frisk.convert_pack '), 3)
        self.assertEqual(result.count('arith.truncf '), 2)
        renamed = re.sub(r'%([A-Za-z]\w*)', r'%renamed_\1', text)
        self.assertEqual(run(renamed, ON).count('frisk.convert_pack '), 3)

    def test_convert_pack_verifier(self):
        valid = '''module {
func.func @pack(%w: vector<4xf32>) -> vector<4xf16> {
  %p = frisk.convert_pack %w {rounding = "rtz"} : vector<4xf32> -> vector<4xf16>
  return %p : vector<4xf16>
}}'''
        run(valid)
        for invalid in (valid.replace('"rtz"', '"rne"'),
                        valid.replace('4xf', '8xf'),
                        valid.replace('4xf32', '4xf64'),
                        valid.replace('vector<4xf', 'vector<2x2xf'),
                        valid.replace('vector<4xf', 'vector<[4]xf')):
            run(invalid, ok=False)
        run(valid, '--test-async-to-llvm=cuda=true', ok=False)

    def test_lowering_pair_order_anchors_and_o3(self):
        for typed in (False, True):
            with self.subTest(typed=typed):
                thread = run(source(1, typed), ON)
                lowered = run(thread, '--thread-ir-legalize',
                              '--test-async-to-llvm=native-exp2=true')
                self.assertNotIn('frisk.convert_pack ', lowered)
                # The fixture returns registers; export with an ordinary ABI.
                lowered = lowered.replace('rocdl.kernel = true, ', '')
                ll = command([LLVM / 'mlir-translate', '--mlir-to-llvmir'], lowered)
                self.assertEqual(ll.count('asm "v_cvt_pkrtz_f16_f32'), 2)
                self.assertEqual(ll.count('asm sideeffect "", "=v,0"'), 2)
                self.assertEqual(ll.count(' = fptrunc '), 1)  # observable output stays RNE
                # Both packed words must consume adjacent lanes, in order,
                # extracted from the same input lifetime anchor.
                extracts = {name: (value, int(lane)) for name, value, lane in re.findall(
                    r'(%\w+) = extractelement <4 x float> (%\w+), i\d+ (\d+)', ll)}
                pairs = re.findall(r'asm "v_cvt_pkrtz_f16_f32[^\n]+\(float (%\w+), float (%\w+)\)', ll)
                self.assertEqual([[extracts[a][1], extracts[b][1]] for a, b in pairs], [[0, 1], [2, 3]])
                self.assertEqual(len({extracts[x][0] for pair in pairs for x in pair}), 1)
                prepared = command([LEGACY, '-'], ll)
                self.assertEqual(prepared.count('!frisk.register_lifetimes'), 2)
                optimized = command([LLVM / 'opt', '-S', '-passes=default<O3>,verify'], prepared)
                self.assertEqual(optimized.count('asm "v_cvt_pkrtz_f16_f32'), 2)
                self.assertEqual(optimized.count('asm sideeffect "", "=v,0"'), 2)

    def test_legacy_entrypoint_is_retired(self):
        for mode in ('off', 'rtz'):
            p = subprocess.run([LEGACY, '--softmax-cvt-pack=' + mode], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('moved to ThreadIR', p.stderr)
        for tool in (LEGACY, MYTEST):
            for mode in ('', '1', 'rne', 'pk_rtz', 'RTZ', 'rtz-more'):
                p = subprocess.run([tool, '--softmax-cvt-pack=' + mode], capture_output=True, text=True)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn('use --softmax-cvt-pack=off|rtz', p.stderr)


if __name__ == '__main__':
    unittest.main()
