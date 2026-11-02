#!/usr/bin/env python3
"""Exercise the MLIR -> MyTest -> tail pass path, then compare real IR outputs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('mytest_tail_validation', HERE / 'verify.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
validation = module.validation


def main():
    outdir = HERE / 'mytest_smoke'
    outdir.mkdir(exist_ok=True)
    original = json.loads((HERE.parent / 'noncausal_softmax_fragments/fresh-command.json').read_text())
    commands = {}
    for name, distance in [('control', 0), ('k_tail_prefetch4', 4)]:
        path = outdir / (name + '.ll')
        cmd = [str(path) if a == '/tmp/softmax-fragments-fresh.ll' else a for a in original]
        cmd.append(f'--k-tail-prefetch={distance}')
        commands[name] = cmd
        result = subprocess.run(cmd, cwd=outdir, text=True, capture_output=True, check=True)
        (outdir / (name + '.log')).write_text(result.stdout + result.stderr)
        if distance:
            assert 'regions=1, pairs=8, moved=7' in result.stderr
        subprocess.run(['/data2/xsl/install/bin/opt', '-passes=default<O3>,verify', path,
                        '-disable-output'], check=True)
    (outdir / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    arrays = [np.random.default_rng(seed).normal(size=(2,4096,128)).astype(np.float16)
              for seed in (101, 102, 103)]
    expected = validation.helper.kdma.vdb.reference(*arrays, 1, 31, causal=False)
    baseline = None
    records = []
    for name in commands:
        path = outdir / (name + '.ll')
        model = validation.Model(validation.helper.model_text(path.read_text()), *arrays,
                                 threads=256, physical=(31, 1))
        full = model.run().reshape(arrays[0].shape)
        actual = full[1, 31*128:32*128]
        np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
        if baseline is None:
            baseline = (full.copy(), model.read_records, model.global_trace.hexdigest(),
                        model.v_reads, model.v_dma)
        assert np.array_equal(full, baseline[0], equal_nan=True)
        assert model.read_records == baseline[1]
        assert (model.global_trace.hexdigest(), model.v_reads, model.v_dma) == baseline[2:]
        assert not model.pending
        assert (model.vm_waits, model.barriers, model.mmac_count, model.dma64_count) == (260,293,8192,512)
        records.append(dict(variant=name, sha256=validation.helper.kdma.digest(path),
            bitwise_equal_to_control=True, delayed_DMA_and_ownership_passed=True,
            max_abs_error=float(np.max(np.abs(actual-expected)))))
        print('PASS MyTest', name, flush=True)
    (outdir / 'validation.json').write_text(json.dumps(dict(cases=records, gpu_tested=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
