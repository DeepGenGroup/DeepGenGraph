#!/usr/bin/env python3
"""Rewrite only the 16 softmax FP32->FP16 fragments without changing Passes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'noncausal_lazy_base'
PINNED = {
    'control': 'b3b554be9a1a0934add236ec0bb978319bd07fa6b609dc5a15537790b278743d',
    'lazy_t8': '2ed74244534df53df31a9ca58ed4b6509c6a391a64f994ee516f8863e6896ab8',
}
# Inputs are read from the original experiment, whose manifest pins their bytes.
MODES = {
    'pk_rne': ('v_cvt_pk_f16_f32 $0, $1, $2', '=v,v,v', 32, 'nearest-even'),
    'sdwa_rne': ('v_cvt_f16_f32 $0, $1\\0A\\09'
                 'v_cvt_f16_f32_sdwa $0, $2 dst_sel:WORD_1 '
                 'dst_unused:UNUSED_PRESERVE src0_sel:DWORD', '=&v,v,v', 64, 'nearest-even'),
    'pk_rtz': ('v_cvt_pkrtz_f16_f32 $0, $1, $2', '=v,v,v', 32, 'toward-zero'),
}
PATTERN = re.compile(r'^  (%[\w.]+) = fptrunc <4 x float> (%math\.result[\w.]*) to <4 x half>(.*)$', re.M)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_call(mode, a, b):
    asm, constraints, _, _ = MODES[mode]
    return f'call i32 asm "{asm}", "{constraints}"(float {a}, float {b})'


def rewrite(text, mode):
    originals = []
    def replace(m):
        originals.append(m[0])
        dest, source, meta = m.groups()
        stem = dest + '.cvtpack'
        lines = [f'  {stem}.f{i} = extractelement <4 x float> {source}, i32 {i}{meta}' for i in range(4)]
        for i in range(2):
            call = pack_call(mode, f'{stem}.f{2*i}', f'{stem}.f{2*i+1}')
            lines.append(f'  {stem}.p{i} = {call}{meta}')
        lines += [
            f'  {stem}.v0 = insertelement <2 x i32> poison, i32 {stem}.p0, i32 0{meta}',
            f'  {stem}.v1 = insertelement <2 x i32> {stem}.v0, i32 {stem}.p1, i32 1{meta}',
            f'  {dest} = bitcast <2 x i32> {stem}.v1 to <4 x half>{meta}',
        ]
        return '\n'.join(lines)
    result, count = PATTERN.subn(replace, text)
    assert count == 16, ('expected 16 softmax fragments', count)
    # Reconstruct the original byte for byte: every other instruction, including
    # the output conversion, denominator, anchors, MMAC, waits and DMA, is intact.
    restored = result
    for original in originals:
        dest = original.split(' = ')[0].strip()
        block = re.compile(r'^  ' + re.escape(dest) + r'\.cvtpack\.f0 = .*?^  '
                           + re.escape(dest) + r' = bitcast [^\n]*', re.M | re.S)
        restored, n = block.subn(lambda _: original, restored)
        assert n == 1
    assert restored == text
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-bin', type=Path, default=Path('/data2/xsl/install/bin'))
    args = parser.parse_args()
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    records = {}
    for base in PINNED:
        src = SOURCE / (base + '.ll')
        expected = manifest['variants'][base]['sha256']
        assert digest(src) == expected, ('source changed', src)
        assert expected == PINNED[base]
        text = src.read_text()
        for mode in ('baseline', *MODES):
            name = base if mode == 'baseline' else base + '_' + mode
            path = HERE / (name + '.ll')
            path.write_text(text if mode == 'baseline' else rewrite(text, mode))
            shutil.copyfile(SOURCE / (base + '.launch.json'), path.with_suffix('.launch.json'))
            subprocess.run([args.llvm_bin/'llvm-as', path, '-o', '/dev/null'], check=True)
            opt = path.with_suffix('.o3.ll')
            subprocess.run([args.llvm_bin/'opt', '-S', '-passes=default<O3>,verify', path, '-o', opt], check=True)
            if mode != 'baseline':
                opcode = MODES[mode][0].split()[0]
                assert opt.read_text().count('asm "' + opcode + ' ') == 32
            records[name] = dict(source=str(src.relative_to(HERE.parents[1])), source_sha256=expected,
                                 sha256=digest(path), o3_sha256=digest(opt), mode=mode,
                                 rounding='nearest-even' if mode == 'baseline' else MODES[mode][3],
                                 softmax_convert_pack_instructions=96 if mode == 'baseline' else MODES[mode][2],
                                 LLVM_and_O3_verified=True,
                                 other_IR_byte_identical_to_source=True)
    probes = HERE / 'probes'
    probes.mkdir(exist_ok=True)
    for mode in MODES:
        # Compile-only: arbitrary scalar arguments; no GPU launch.
        (probes / (mode + '.ll')).write_text(
            'target triple = "amdgcn-amd-amdhsa"\n'
            'define i32 @probe(float %a, float %b) {\n'
            f'  %p = {pack_call(mode, "%a", "%b")}\n  ret i32 %p\n}}\n')
    (HERE/'manifest.json').write_text(json.dumps(dict(variants=records, target='gfx936',
        shape=[1, 32, 4096, 128], causal=False, gpu_tested=False, target_codegen_tested=False,
        note='Instruction counts are intended softmax-only counts, not measured speedups.'), indent=2)+'\n')
    print('Generated and LLVM/O3 verified', len(records), 'variants in', HERE)


if __name__ == '__main__':
    main()
