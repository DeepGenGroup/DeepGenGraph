#!/usr/bin/env python3
"""Compare the compiled LazySoftmaxBase pass against the measured hand-rewritten IR.

The measured artifact lazy_t8.ll was produced by generate.py's text rewrite and ran
at 1.5718 ms. The pass must reproduce that transformation's structure and numerics
from the same input, so this checks the pass output against both control.ll and
lazy_t8.ll instead of trusting the pass alone.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOL = ROOT / '3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')
spec = importlib.util.spec_from_file_location('lazy_verify', HERE / 'verify.py')
lazy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lazy)


def run(cmd):
    return subprocess.run(list(map(str, cmd)), text=True, capture_output=True, check=True)


def counts(text):
    """Per-region instruction signature, independent of SSA and block names.

    Blocks are classified structurally: the entry block and the block with the
    return are dropped, so what is left is the key-tile loop. Inside it the
    lazy.slow block is counted separately from the fast path.
    """
    body = text.split('define amdgpu_kernel void ')[1].split('\n}', 1)[0]
    body = body.split('\n', 1)[1]
    blocks, label = [('entry', [])], 'entry'
    for line in body.splitlines():
        header = re.match(r'^([\w.]+):', line)
        if header:
            label = header[1]
            blocks.append((label, []))
        else:
            blocks[-1][1].append(line)
    assert len(blocks) >= 4, 'unexpected block structure'
    loop = [(name, lines) for name, lines in blocks[1:]
            if not any(re.match(r'\s*ret ', l) for l in lines)]
    assert len(loop) == len(blocks) - 2, 'expected exactly one returning block'
    pattern = {'fsub_vec4': r'= fsub <4 x float>', 'fmul_vec4': r'= fmul <4 x float>',
               'fmul_vec8': r'= fmul <8 x float>', 'mmac': r'v_mmac_f32_16x16x16_f16',
               'exp2': r'@llvm\.amdgcn\.exp2\.f32|@__ocml_exp2_f32', 'phi': r' = phi ',
               'maxnum': r'@llvm\.maxnum\.', 'ballot': r'@llvm\.amdgcn\.ballot',
               'barrier': r'@llvm\.amdgcn\.s\.barrier', 'dma': r'buffer_load_dword',
               'wait': r's_waitcnt vmcnt', 'ds_read': r'@llvm\.hcu\.ds\.read\.',
               'fsub_v1': r'= fsub <1 x float>', 'select': r'= select '}
    out = {}
    for region, want_slow in (('fast', False), ('slow', True)):
        lines = [l for name, group in loop if name.startswith('lazy.slow') == want_slow
                 for l in group]
        out[region] = {k: len(re.findall(v, '\n'.join(lines))) for k, v in pattern.items()}
    assert out['fast']['mmac'] == 256, out['fast']
    return out


def main():
    control = HERE / 'control.ll'
    measured = HERE / 'lazy_t8.ll'
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name in ('control', 'lazy_t8'):
        path = HERE / (name + '.ll')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['variants'][name]['sha256']

    records = {}
    produced = {}
    for name, option in [('pass_t8', '--lazy-softmax-base=8'), ('pass_t4', '--lazy-softmax-base=4'),
                         ('pass_off', '--lazy-softmax-base=off')]:
        raw = HERE / (name + '.raw.ll')
        result = run([TOOL, option, control])
        raw.write_text(result.stdout)
        (HERE / (name + '.pass.log')).write_text(result.stderr)
        out = HERE / (name + '.ll')
        out.write_text(run([TOOL, '--llvm22-text', raw]).stdout)
        out.with_suffix('.launch.json').write_bytes((HERE / 'control.launch.json').read_bytes())
        run([LLVM / 'llvm-as', out, '-o', '/dev/null'])
        opt = out.with_suffix('.o3.ll')
        run([LLVM / 'opt', '-S', '-passes=default<O3>,verify', out, '-o', opt])
        produced[name] = out.read_text()
        applied = 'Lazy softmax base: rows=' in result.stderr
        assert applied == (option != '--lazy-softmax-base=off'), (name, result.stderr)
        records[name] = dict(option=option, applied=applied, sha256=hashlib.sha256(
            out.read_bytes()).hexdigest(), diagnostics=result.stderr.strip(),
            structure=counts(produced[name]), LLVM_and_O3_verified=True)
        print('PASS export', name, records[name]['structure']['fast'], flush=True)

    # The disabled run must leave the input untouched apart from re-printing.
    assert counts(produced['pass_off']) == counts(control.read_text()), 'off changed the IR'
    # The pass must reproduce the measured rewrite's work per region. Two counters
    # are expected to differ, both deliberately:
    #   ballot   measured emitted llvm.amdgcn.fcmp, which only becomes fcmp+ballot
    #            after O3; the pass emits that final form directly.
    #   fsub_v1  measured updated N through extractelement/fsub float/insertelement;
    #            the pass subtracts on the <1 x float> field, two instructions fewer.
    contract = ('fsub_vec4', 'fmul_vec4', 'fmul_vec8', 'mmac', 'exp2', 'phi', 'maxnum',
                'barrier', 'dma', 'wait', 'ds_read', 'select')
    want = counts(measured.read_text())
    for name in ('pass_t8', 'pass_t4'):
        got = counts(produced[name])
        for region in ('fast', 'slow'):
            kept = {k: got[region][k] for k in contract}
            expected = {k: want[region][k] for k in contract}
            assert kept == expected, (name, region, kept, expected)
        assert got['fast']['ballot'] == 1 and want['fast']['ballot'] == 0
        assert got['slow']['fsub_v1'] == 2 and want['slow']['fsub_v1'] == 0
        records[name]['structure_matches_measured'] = True
    print('PASS structure matches the measured rewrite on', len(contract), 'counters', flush=True)

    # Numerics: run the pass output through the same CPU model as verify.py.
    cases = [('random', 0, 0), ('random', 1, 31), ('max_changes', 1, 2),
             ('negative_underflow', 1, 1), ('gradual_growth', 1, 0), ('large_jump', 1, 1),
             ('one_wave_triggers', 1, 1)]
    numeric = []
    for kind, head, query in cases:
        arrays = lazy.dataset(kind, 42 + query)
        expected = lazy.previous.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        base = None
        for name in ('control', 'lazy_t8', 'pass_t8', 'pass_t4'):
            text = lazy.previous.helper.model_text((HERE / (name + '.ll')).read_text())
            model, full, actual = lazy.run_case(text, arrays, head, query)
            import numpy as np
            assert np.isfinite(actual).all(), (kind, name)
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            assert (model.vm_waits, model.barriers, model.mmac_count,
                    model.dma64_count) == (260, 293, 8192, 512)
            bound = lazy.W_BOUND['lazy_t8' if name.endswith('t8') else
                                 'lazy_t4' if name.endswith('t4') else name]
            assert model.max_w <= bound * (1 + 1e-6), (kind, name, model.max_w)
            if name == 'control':
                base = model
            assert model.global_trace.digest() == base.global_trace.digest()
            assert model.lds_values.digest() == base.lds_values.digest()
            assert (model.v_reads, model.v_dma) == (base.v_reads, base.v_dma)
            numeric.append(dict(kind=kind, variant=name, slow_path_per_wave=model.slow_counts.tolist(),
                max_abs_error=float(np.max(np.abs(actual - expected))), max_W=model.max_w))
            print('PASS numeric', kind, name, 'slow', model.slow_counts.tolist(), flush=True)
        # The pass and the measured rewrite must agree on when the slow path runs.
        by = {r['variant']: r['slow_path_per_wave'] for r in numeric[-4:]}
        assert by['pass_t8'] == by['lazy_t8'], (kind, by)
    (HERE / 'pass_validation.json').write_text(json.dumps(dict(
        exports=records, numeric=numeric, compared_against=str(measured),
        measured_median_ms=1.571839, gpu_tested=False), indent=2) + '\n')
    print('PASS pass output matches the measured rewrite on all cases', flush=True)


if __name__ == '__main__':
    main()
