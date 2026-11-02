#!/usr/bin/env python3
"""Reduction-aware paired LDS completion: generic matching and safety boundaries."""
from collections import Counter, defaultdict
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / '3rd/deepgengraph/build/test/LegacyLLVMExportTest'
OPT = Path('/data2/xsl/install/bin/opt')
MMA = 'asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\\0A\\09", "=v,v,v,0"'


def fixture(count=8, spacing=128, reduction='maxnum', boundary=''):
    lines = ['declare float @llvm.maxnum.f32(float, float)',
             'declare float @llvm.minnum.f32(float, float)',
             'declare float @llvm.maximum.f32(float, float)',
             'declare float @llvm.minimum.f32(float, float)',
             'declare void @opaque()',
             'declare void @pure_unknown() memory(none)',
             'declare void @llvm.amdgcn.s.barrier()',
             'declare void @llvm.amdgcn.sched.barrier(i32)',
             'declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)',
             'define void @not_attention(ptr addrspace(3) %base, ptr %out) {', 'entry:']
    for i in range(count):
        offset = (i // 2) * 4096 + (i % 2) * spacing
        lines.append(f'  %p{i} = getelementptr i8, ptr addrspace(3) %base, i64 {offset}')
    for i in range(count):
        acc = 'zeroinitializer' if i == 0 else f'%m{i-1}'
        prev = '0.0' if i == 0 else f'%r{i-1}'
        lines += [f'  %k{i} = load <4 x half>, ptr addrspace(3) %p{i}, align 8',
                  f'  %m{i} = call <4 x float> {MMA}(<4 x half> zeroinitializer, <4 x half> %k{i}, <4 x float> {acc}) #0',
                  f'  %x{i} = extractelement <4 x float> %m{i}, i32 0']
        if reduction == 'sum':
            lines.append(f'  %r{i} = fadd float %x{i}, {prev}')
        else:
            lines.append(f'  %r{i} = call float @llvm.{reduction}.f32(float %x{i}, float {prev})')
        if boundary:
            lines.append('  ' + boundary.replace('$i', str(i)))
    lines += [f'  store float %r{count-1}, ptr %out', f'  store <4 x float> %m{count-1}, ptr %out',
              '  ret void', '}', 'attributes #0 = { convergent nounwind willreturn memory(none) }', '!0 = !{}']
    return '\n'.join(lines) + '\n'


def instructions(text):
    return [re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
            for l in text.splitlines() if l.startswith('  ') and '!frisk.k_tail_prefetch' not in l]


class PairedLDSTail(unittest.TestCase):
    def export(self, text, distance='4', ok=True):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input.ll'
            p.write_text(text)
            result = subprocess.run([TOOL, '--k-tail-prefetch=' + distance, p], text=True, capture_output=True)
            if not ok:
                self.assertNotEqual(result.returncode, 0)
                return result
            self.assertEqual(result.returncode, 0, result.stderr)
            p.write_text(result.stdout)
            subprocess.run([OPT, '-passes=default<O3>,verify', p, '-disable-output'], check=True, capture_output=True)
            return result

    def test_generic_pairing_and_exact_fp_order(self):
        for count, spacing, reduction in ((4, 64, 'maxnum'), (8, 128, 'minnum'),
                (12, 512, 'maximum'), (8, 2048, 'minimum'), (8, 128, 'sum')):
            text = fixture(count, spacing, reduction)
            for distance in ('1', '2', '4', '8'):
                with self.subTest(count=count, spacing=spacing, reduction=reduction, distance=distance):
                    before = self.export(text, '0').stdout
                    result = self.export(text, distance)
                    self.assertIn(f'pairs={count//2}', result.stderr)
                    a, b = instructions(before), instructions(result.stdout)
                    self.assertEqual(Counter(a), Counter(b))
                    self.assertEqual([l for l in a if ' = load ' not in l],
                                     [l for l in b if ' = load ' not in l])
                    self.assertLess(result.stdout.index('%k2 ='), result.stdout.index('%m1 ='))
                    pairs = defaultdict(list)
                    lines = result.stdout.splitlines()
                    for i, l in enumerate(lines):
                        if m := re.search(r'!frisk.k_tail_pair !(\d+)', l):
                            pairs[m[1]].append(i)
                    self.assertEqual(len(pairs), count//2)
                    for positions in pairs.values():
                        self.assertEqual(len(positions), 2)
                        lo, hi = positions
                        self.assertFalse(any('sched.barrier' in l for l in lines[lo:hi]))
                        names = [re.match(r'  (%\w+) =', lines[n])[1] for n in positions]
                        uses = [i for i,l in enumerate(lines) if 'v_mmac_' in l and
                                any(re.search(re.escape(n)+r'[, )]', l) for n in names)]
                        self.assertLess(hi, min(uses))
                        self.assertLessEqual(sum('v_mmac_' in l for l in lines[hi+1:min(uses)]), int(distance))

    def test_renaming_and_preexisting_main_pass(self):
        text = fixture().replace('ptr %out) {', 'ptr %out) "frisk.k_lds_prefetch"="8" {')
        text = re.sub(r'%([A-Za-z]\w*)', r'%renamed_\1', text).replace('@not_attention', '@another_gemm')
        result = self.export(text)
        self.assertIn('pairs=4', result.stderr)
        self.assertIn('"frisk.k_lds_prefetch"="8"', result.stdout)
        self.assertIn('"frisk.k_tail_prefetch"="4"', result.stdout)

    def test_boundaries(self):
        boundaries = ['call void @opaque()', 'call void @pure_unknown()',
            'call void @llvm.amdgcn.s.barrier()', 'call void @llvm.amdgcn.sched.barrier(i32 0)',
            'fence syncscope("workgroup") acquire', 'store i32 1, ptr addrspace(3) %base',
            'call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"()',
            '%wave$i = call i32 @llvm.amdgcn.ds.bpermute(i32 0, i32 0)',
            '%other$i = load volatile i32, ptr addrspace(3) %base']
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                text = fixture(boundary=boundary)
                result = self.export(text)
                self.assertIn('moved=0, anchors=0', result.stderr)
                self.assertEqual(instructions(result.stdout), instructions(self.export(text, '0').stdout))

    def test_rejections(self):
        base = fixture()
        texts = [base.replace('load <4 x half>', 'load volatile <4 x half>'),
                 base.replace('align 8', 'align 4'),
                 base.replace('ptr %out) {', 'ptr %out) strictfp {'),
                 base.replace('ptr %out) {', 'ptr %out) noinline optnone {'),
                 base.replace('asm "v_mmac', 'asm sideeffect "v_mmac'),
                 re.sub(r'<4 x half> zeroinitializer, <4 x half> (%k\d)',
                        r'<4 x half> \1, <4 x half> zeroinitializer', base),
                 re.sub(r'(load <4 x half>[^\n]+)', r'\1, !frisk.k_lds_pair !0', base),
                 base.replace('llvm.maxnum.f32', 'not_an_intrinsic')]
        for text in texts:
            out = self.export(text).stdout
            self.assertNotIn('!frisk.k_tail_pair', out)
            self.assertEqual(instructions(out), instructions(self.export(text, '0').stdout))

    def test_off_idempotence_conflict_and_invalid(self):
        text = fixture()
        self.assertNotIn('frisk.k_tail_prefetch', self.export(text, '0').stdout)
        first = self.export(text).stdout
        second = self.export(first).stdout
        self.assertEqual(instructions(first), instructions(second))
        self.assertEqual(first.count('call void @llvm.amdgcn.sched.barrier'),
                         second.count('call void @llvm.amdgcn.sched.barrier'))
        self.export(first, '2', ok=False)
        for distance in ('-1', '9', 'true', '', '4x'):
            self.export(text, distance, ok=False)

    def test_unknown_addresses_and_block_boundaries(self):
        text = fixture()
        text = text.replace('ptr %out)', 'ptr %out, ' + ', '.join(f'i64 %offset{i}' for i in range(8)) + ')')
        text = re.sub(r'(%p(\d+) = getelementptr[^\n]*), i64 \d+',
                      lambda m: m[1] + ', i64 %offset' + m[2], text)
        self.assertNotIn('!frisk.k_tail_pair', self.export(text).stdout)
        text = fixture(boundary='br label %next$i\nnext$i:')
        result = self.export(text)
        self.assertNotIn('!frisk.k_tail_pair', result.stdout)
        self.assertEqual(instructions(result.stdout), instructions(self.export(text, '0').stdout))

    def test_constrained_fp_boundary(self):
        text = fixture(boundary='%strict$i = call float @llvm.experimental.constrained.fadd.f32(float 1.0, float 2.0, metadata !"round.dynamic", metadata !"fpexcept.strict")')
        text = 'declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)\n' + text
        result = self.export(text)
        self.assertNotIn('!frisk.k_tail_pair', result.stdout)
        self.assertEqual(instructions(result.stdout), instructions(self.export(text, '0').stdout))


if __name__ == '__main__':
    unittest.main()
