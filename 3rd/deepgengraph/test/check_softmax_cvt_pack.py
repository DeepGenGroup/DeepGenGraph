#!/usr/bin/env python3
"""Guard the opt-in rounding change and compiler-owned PV operand boundary."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from check_softmax_fragments import fixture as softmax_fixture

TOOL = Path(__file__).resolve().parents[1]/'build/test/LegacyLLVMExportTest'
MYTEST = TOOL.with_name('MyTest')
OPT = Path('/data2/xsl/install/bin/opt')


def fixture():
    return ('source_filename = "fixture"\n' + softmax_fixture().replace(
        'define void @kernel', 'define amdgpu_kernel void @kernel').replace(
        '  ret void', '  %output.half = fptrunc <4 x float> %mma3 to <4 x half>\n'
        '  store <4 x half> %output.half, ptr %out\n  ret void'))


def canonical(text):
    return re.sub(r'^; ModuleID = .*\n', '', text)


class SoftmaxCvtPackTest(unittest.TestCase):
    def export(self, text, mode='rtz'):
        with tempfile.TemporaryDirectory(prefix='cvt-pack-') as tmp:
            p = Path(tmp)/'input.ll'
            p.write_text(text)
            result = subprocess.run([TOOL, '--softmax-cvt-pack='+mode, p],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            p.write_text(result.stdout)
            verify = subprocess.run([OPT, '-passes=default<O3>,verify', '-disable-output', p],
                                    capture_output=True, text=True)
            self.assertEqual(verify.returncode, 0, verify.stderr)
            return result

    def test_only_softmax_casts_and_existing_operands(self):
        source = fixture()
        result = self.export(source)
        self.assertIn('fragments=4, packed-pairs=8', result.stderr)
        self.assertEqual(result.stdout.count('asm "v_cvt_pkrtz_f16_f32'), 8)
        self.assertEqual(result.stdout.count(' = fptrunc '), 1) # output stays RNE
        self.assertIn('%output.half = fptrunc <4 x float> %mma3', result.stdout)
        for token in (' = fadd ', 'v_mmac_', ' = call <4 x float> asm sideeffect',
                      ' = call <4 x half> asm sideeffect', 'call void @llvm.amdgcn.s.barrier'):
            before = [line for line in source.splitlines() if token in line]
            after = [line for line in result.stdout.splitlines() if token in line]
            # LLVM re-numbers attribute groups when adding the pass marker.
            normalize = lambda lines: [re.sub(r' #\d+|!\d+', '', line) for line in lines]
            self.assertEqual(normalize(after), normalize(before))

    def test_off_and_idempotent(self):
        off = self.export(fixture(), 'off')
        self.assertNotIn('v_cvt_pkrtz', off.stdout)
        on = self.export(off.stdout)
        twice = self.export(on.stdout)
        self.assertEqual(canonical(on.stdout), canonical(twice.stdout))
        self.assertEqual(twice.stderr, '')

    def test_renaming(self):
        source = re.sub(r'%([A-Za-z]\w*)', r'%renamed_\1', fixture())
        self.assertEqual(self.export(source).stdout.count('asm "v_cvt_pkrtz'), 8)

    def test_reject_unproven_rounding_changes(self):
        source = fixture()
        cases = {
            'unmarked': source.replace(', !frisk.register_lifetimes !0', ''),
            'non_exp': source.replace('llvm.amdgcn.exp2.f32', 'other_exp'),
            'non_identity': source.replace('asm sideeffect "",', 'asm sideeffect "s_nop 0",'),
            'ordinary_function': source.replace('define amdgpu_kernel', 'define'),
            'strictfp': source.replace('ptr %out) {', 'ptr %out) strictfp {'),
            'optnone': source.replace('ptr %out) {', 'ptr %out) noinline optnone {'),
            'sideeffect_mma': source.replace('asm "v_mmac', 'asm sideeffect "v_mmac'),
            'b_operand': re.sub(r'<4 x half> %p(\d), <4 x half> %v',
                                r'<4 x half> %v, <4 x half> %p\1', source),
            'a_and_b_operand': re.sub(r'<4 x half> %p(\d), <4 x half> %v',
                                      r'<4 x half> %p\1, <4 x half> %p\1', source),
            'wrong_width': source.replace('<4 x ', '<8 x '),
            'extra_half_use': source.replace('  ret void', '\n'.join(
                f'  store <4 x half> %h{i}, ptr %out' for i in range(4))+'\n  ret void'),
            'packed_store': source.replace('  ret void', '\n'.join(
                f'  store <4 x half> %p{i}, ptr %out' for i in range(4))+'\n  ret void'),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                off, on = self.export(text, 'off'), self.export(text)
                self.assertEqual(canonical(off.stdout), canonical(on.stdout))

    def test_mixed_safe_and_unsafe_fragments(self):
        source = fixture().replace('  ret void', '  store <4 x half> %p0, ptr %out\n  ret void')
        result = self.export(source)
        self.assertEqual(result.stdout.count('asm "v_cvt_pkrtz'), 6)
        self.assertIn('%h0 = fptrunc', result.stdout)

    def test_bad_options_on_both_entrypoints(self):
        for tool in (TOOL, MYTEST):
            for option in ('', '1', 'rne', 'pk_rtz', 'RTZ', 'rtz-more'):
                with self.subTest(tool=tool.name, option=option):
                    result = subprocess.run([tool, '--softmax-cvt-pack='+option],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('use --softmax-cvt-pack=off|rtz', result.stderr)


if __name__ == '__main__':
    unittest.main()
