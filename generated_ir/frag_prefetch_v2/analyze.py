#!/usr/bin/env python3
"""Reproduce this attention snapshot's LDS diagnosis, NOT a target GPU benchmark.

The compiler optimization is shape-independent. Only this diagnostic extracts
the known temp.ll QK loop body and substitutes gfx90a MFMA for HCU MMAC.
Run from the repository root after building LegacyLLVMExportTest.
"""
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TOOL = ROOT / '3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')


def run(args, source=None):
    result = subprocess.run(list(map(str, args)), input=source, text=True,
                            capture_output=True, check=True, timeout=120)
    return result.stdout, result.stderr


def counts(text):
    return {
        'ordinary_lds_vector_loads': len(re.findall(r'load <4 x half>, ptr addrspace\(3\)', text)),
        'matrix_reads': len(re.findall(r'call <4 x i32> @llvm.hcu.ds.read', text)),
        'bpermutes': len(re.findall(r'call i32 @llvm.amdgcn.ds.bpermute', text)),
        'mmac': text.count('v_mmac_f32_16x16x16_f16'),
        'schedule_hints': len(re.findall(r'call void @llvm.amdgcn.sched.barrier', text)),
    }


def qk_probe(text):
    # Snapshot-specific extraction; preserve the full QK loads/address DAG and
    # all 32 MMAs. Use one iteration and zero initial accumulators, store all
    # results so none can be DCE'd. This is not the full kernel's VGPR usage.
    prefix = text[:text.index('25:')]
    prefix = prefix[:prefix.rindex('  br label')]
    prefix += ('  %26 = add i64 %6, 0\n  %38 = add i64 0, 0\n'
               '  %39 = insertvalue [2 x <32 x float>] zeroinitializer, '
               '<32 x float> zeroinitializer, 0\n')
    outer = text[text.index('32:'):text.index('37:')].split('\n', 1)[1]
    outer = outer[:outer.rindex('  br label')]
    body = text[text.index('41:'):text.index('235:')].split('\n', 1)[1]
    body = body[:body.index('  fence syncscope("workgroup") release, !dbg !46')]
    probe = (prefix + outer + body +
             '  store [2 x <32 x float>] %233, ptr addrspace(1) %3, align 16\n'
             '  ret void\n}\n' +
             text[text.index('; Function Attrs:', text.index('235:')):])
    probe = re.sub(r'%([0-9]+)\b', r'%v\1', probe)
    return probe.replace('v_mmac_f32_16x16x16_f16', 'v_mfma_f32_16x16x16f16')


def main():
    inputs = {
        'temp': ROOT / 'temp.ll',
        'attn_best': ROOT / 'generated_ir/attn_best_async_current.ll',
        'old_on': ROOT / 'generated_ir/frag_prefetch/temp.frag-prefetch.ll',
    }
    report = {'inputs': {}, 'exports': {}, 'qk_gfx90a_probe': {}}
    for name, path in inputs.items():
        report['inputs'][name] = {'path': str(path.relative_to(ROOT)),
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ('temp', 'attn_best'):
        for mode in ('off', 'on'):
            source, log = run([TOOL, f'--frag-prefetch={int(mode == "on")}',
                               '--frag-prefetch-words=4', inputs[name]])
            raw_path = OUT / f'{name}.{mode}.llvm.ll'
            raw_path.write_text(source)
            (OUT / f'{name}.{mode}.log').write_text(log)
            compat, _ = run([TOOL, '--llvm22-text', raw_path])
            compat_path = OUT / f'{name}.{"frag-prefetch" if mode == "on" else "baseline"}.ll'
            compat_path.write_text(compat)
            optimized, _ = run([LLVM/'opt', '-S', '-passes=default<O3>', '-'], compat)
            (OUT / f'{name}.{mode}.opt.ll').write_text(optimized)
            report['exports'][f'{name}.{mode}'] = {
                'before_O3': counts(compat), 'after_O3': counts(optimized), 'diagnostic': log.strip()}
        assert report['exports'][f'{name}.on']['after_O3']['ordinary_lds_vector_loads'] == 20
        assert report['exports'][f'{name}.off']['after_O3']['ordinary_lds_vector_loads'] == 20

    old = inputs['old_on'].read_text()
    sources = {
        'off': (OUT/'temp.baseline.ll').read_text(),
        'old_on': old,
        # Ablation: fixing CSE alone leaves 16 DS reads; group preservation and
        # removal of the leading boundary are needed to recover the other six.
        'effects_only': old.replace('call void @llvm.amdgcn.sched.barrier(i32 0),',
                                   'call void @llvm.amdgcn.sched.barrier(i32 0) '
                                   'memory(inaccessiblemem: readwrite),'),
        'on': (OUT/'temp.frag-prefetch.ll').read_text(),
    }
    for name, source in sources.items():
        probe = qk_probe(source)
        optimized, _ = run([LLVM/'opt', '-S', '-passes=default<O3>', '-'], probe)
        assembly, _ = run([LLVM/'llc', '-O3', '-mtriple=amdgcn-amd-amdhsa',
                           '-mcpu=gfx90a', '-o', '-', '-'], optimized)
        for suffix, data in [('ll', probe), ('opt.ll', optimized), ('s', assembly)]:
            (OUT/f'qk.{name}.probe.{suffix}').write_text(data)
        ds = collections.Counter(re.findall(r'^\s+(ds_read\w+)\b', assembly, re.M))
        report['qk_gfx90a_probe'][name] = {
            'instruction_counts': dict(ds), 'total_ds_reads': sum(ds.values()),
            'resources': re.findall(r'; (?:NumVgprs|ScratchSize):.*', assembly)}
        assert assembly.count('v_mfma_f32_16x16x16f16') == 32
        assert '; ScratchSize: 0' in assembly
    assert [report['qk_gfx90a_probe'][s]['total_ds_reads']
            for s in ('off', 'old_on', 'effects_only', 'on')] == [10, 32, 16, 10]

    asm = (ROOT/'k_test.gcn.log').read_text(encoding='utf-16').split('.Lfunc_end0:')[0]
    location, origins = 0, collections.Counter()
    for line in asm.splitlines():
        match = re.search(r'\.loc\s+\d+\s+(\d+)\s+', line)
        if match:
            location = int(match[1])
        match = re.match(r'\s*(ds_\w+)', line)
        if match:
            origins[(location, match[1])] += 1
    report['user_gfx936_on_asm_origins'] = [
        {'mlir_line': line, 'opcode': op, 'static_count': n}
        for (line, op), n in sorted(origins.items())]
    # This dispatch: 1024 workgroups, four wave64s, 32 causal query tiles per
    # head, 32 heads. Sum of outer tile visits = 32 * (1+...+32).
    outer = 32 * sum(range(1, 33)) * 4
    inner = outer * 4
    report['dynamic_reconstruction'] = {
        'outer_wave_iterations': outer, 'qk_inner_wave_iterations': inner,
        'Q_reads': 2 * inner, 'K_reads_old_on': 30 * inner,
        'K_reads_off_inferred': 8 * inner, 'V_matrix_reads': 32 * outer,
        'row_reduction_bpermutes': 4 * outer, 'output_bpermutes': 64 * 4096,
        'old_on_total': (2 + 30) * inner + 36 * outer + 64 * 4096,
        'off_total_reconstructed': (2 + 8) * inner + 36 * outer + 64 * 4096,
        'note': 'On matches supplied ISA/profile exactly. Off K count is inferred '
                'from the supplied counter and independently reproduced by the gfx90a QK probe; '
                'matched off gfx936 ISA was not supplied.'}
    assert report['dynamic_reconstruction']['old_on_total'] == 11345920
    assert report['dynamic_reconstruction']['off_total_reconstructed'] == 5398528
    report['limitations'] = ['No gfx936 compiler/GPU available here; new full-kernel '
                             'VGPR count, bank conflicts, numerical correctness and '
                             'performance require target rerun. Probe VGPRs are NOT full-kernel VGPRs.']
    (OUT/'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS full IR O3 reuse, QK DS counts 10/32/16/10, and profile reconstruction')


if __name__ == '__main__':
    main()
