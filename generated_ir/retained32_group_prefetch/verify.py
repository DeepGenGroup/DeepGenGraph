#!/usr/bin/env python3
"""Validate this six-case export using the existing retained-Q CPU model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'q_retention_checks', ROOT / 'generated_ir/q_bk_retention_sweep/verify.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--opt', default='/data2/xsl/install/bin/opt')
    args = parser.parse_args()
    directory = args.run.resolve()
    reference = ROOT / 'generated_ir/q_bk_retention_sweep/retained_32words.ll'
    old = reference.read_text()
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == 'c5627d7e797e5ee93c73ba04d7112a238ba4dbcb52a888267a4c7b30520f9915'
    baseline = json.loads((directory / 'grid-000004/result.json').read_text())
    reports = {}
    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    for index, (group, prefetch) in enumerate(
            ((g, p) for g in (4, 8, 16) for p in (0, 1)), 1):
        name = f'grid-{index:06d}'
        path = directory / name / 'kernel.ll'
        text = path.read_text()
        result = json.loads(path.with_name('result.json').read_text())
        assert result['status'] == 'verified'
        cfg = dict(baseline['parameters'], **{'workgroup-group-size': str(group), 'frag-prefetch': str(prefetch)})
        assert result['parameters'] == cfg
        assert result['retention_reports'] == [{'operand_tiles': 4, 'budget_words_per_thread': 32}]
        assert result['retained_loop_words_observed'] == [32]
        assert result['static_lds_bytes'] == 16384
        label = re.search(r'^([\w.]+):', text, re.M)[1]
        entry, loop = text.split('\n' + label + ':', 1)
        assert entry.count('buffer_load_dwordx4 ') == 8
        assert loop.count('buffer_load_dwordx4 ') == 16
        assert entry.count('= fmul <4 x half>') == 16
        assert loop.count('= fmul <4 x half>') == 0
        assert text.count('call void @llvm.amdgcn.sched.barrier(') == 8 * prefetch
        subprocess.run([args.opt, '-passes=default<O3>,verify', str(path), '-disable-output'], check=True)
        if group == 8 and prefetch == 1:
            source = ROOT / 'generated_ir/pipeline_cleanup/nohint_unrolled.mlir'
            normalized = text.replace(str(path.with_name('input.mlir')), str(source)).replace(
                f'filename: "input.mlir", directory: "{path.parent}"',
                f'filename: "{source.name}", directory: "{source.parent}"')
            assert normalized == old, 'baseline differs beyond source metadata'
        cases = []
        # All six variants: first BN. Both prefetch settings: longest BN loop.
        for query in ([0, 31] if group == 8 else [0]):
            head = 1
            forward = 31 - query
            linear = forward // group * (32 * group) + head * group + forward % group
            model = checks.SweepBlock(re.sub(r'\bexact ', '', text), *arrays,
                                      physical=(linear % 32, linear // 32))
            full = model.run().reshape(arrays[0].shape)
            actual = full[head, query*128:(query+1)*128]
            expected = checks.previous.audits.reference(*arrays, head, query)
            assert np.allclose(actual, expected, atol=.01, rtol=.01)
            written = np.zeros(full.shape, dtype=bool)
            written[head, query*128:(query+1)*128] = True
            assert np.isnan(full[~written]).all()
            assert not model.pending
            assert model.q_fragment_reads == {f: 2 for f in range(4)}
            for fragment in range(4):
                want = {(1 << 32) + 2*((head*4096+row)*128+col): 1
                        for row in range(query*128, (query+1)*128)
                        for col in range(fragment*32, (fragment+1)*32)}
                assert model.q_elements[fragment] == want
            assert model.read_calls['K'] == model.read_calls['V'] == 8*(query+1)
            assert model.hints == 8*prefetch*(query+1)
            cases.append(dict(head=head, query=query, Q_elements_each_read_once=16384,
                              reference_passed=True, writes_only_selected_tile=True,
                              delayed_DMA_and_cross_wave_LDS_audit_passed=True))
            print(f'PASS {name} group={group} prefetch={prefetch} query={query}', flush=True)
        reports[name] = dict(group=group, reverse=1, frag_prefetch=prefetch,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             static_lds_bytes=16384, retained_tiles=4,
                             llvm_O3_verified=True, cases=cases)
    (directory / 'validation.json').write_text(json.dumps(dict(
        baseline='grid-000004', baseline_source_metadata_normalized_equal=True,
        reference=str(reference), variants=reports, gpu_tested=False,
        limitation='CPU model checks selected blocks, not target GPU timing or register allocation.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
