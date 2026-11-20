#!/usr/bin/env python3
"""Compile arbitrary static S and execute the exported LLVM with checked memory.

This validates numerical results and exact-sized buffers in a CPU wave model;
it does not simulate GPU timing, asynchronous hazards, or target machine code.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from generate_attention_ir import TEMPLATE, generate_ir
from check_attention_llvm_numerics import LLVMWorkgroupModel


class TailModel(LLVMWorkgroupModel):
    def __init__(self, *args, batch=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = batch

    def call(self, body):
        for axis, coordinate in zip('xyz', (self.block, self.head, self.batch)):
            if f'@llvm.amdgcn.workgroup.id.{axis}(' in body:
                return np.full(self.t, coordinate, dtype=np.int64)
        return super().call(body)


def check(text, s, bm, d, b, h, causal):
    rng = np.random.default_rng(42)
    q, k, v = [rng.normal(size=(b, h, s, d)).astype(np.float16) for _ in range(3)]
    cases = {(0, 0, 0), (b - 1, h - 1, (s - 1) // bm)}
    for batch, head, block in sorted(cases):
        model = TailModel(text, q, k, v, threads=128, batch=batch, head=head, block=block)
        actual = model.run().reshape(q.shape)
        start, end = block * bm, min((block + 1) * bm, s)
        key_end = end if causal else s
        score = q[batch, head, start:end].astype(np.float32) @ k[batch, head, :key_end].astype(np.float32).T / np.sqrt(d)
        if causal:
            score[np.arange(key_end)[None, :] > np.arange(start, end)[:, None]] = -np.inf
        weights = np.exp(score - score.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        expected = weights @ v[batch, head, :key_end].astype(np.float32)
        np.testing.assert_allclose(actual[batch, head, start:end], expected, atol=.01, rtol=.01)
        untouched = np.ones(q.shape, dtype=bool)
        untouched[batch, head, start:end] = False
        assert np.isnan(actual[untouched]).all(), 'tail store overwrote another tile/head/batch'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mytest', nargs='?', type=Path,
                        default=ROOT / '3rd/deepgengraph/build/test/MyTest')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='attention-tails-') as tmp:
        directory = args.output_dir.resolve() if args.output_dir else Path(tmp)
        directory.mkdir(parents=True, exist_ok=True)
        cases = [(s, 64, 32, 64, 1, 1, 'online') for s in (1, 31, 32, 33, 63, 65, 127, 129)]
        cases += [(65, 32, 64, 128, 2, 3, 'online'),
                  (32, 32, 64, 64, 1, 1, 'online'),
                  (33, 64, 32, 64, 2, 3, 'legacy'),
                  (64, 64, 32, 64, 1, 1, 'online')]
        for s, bm, bn, d, b, h, softmax in cases:
            for causal in (True, False):
                name = f's{s}-bm{bm}-bn{bn}-{softmax}-{causal}'
                source, _ = generate_ir(TEMPLATE.read_text(), bm, bn, 32, softmax,
                                        b=b, h=h, s=s, d=d, causal=causal)
                inp, out, log = [directory / (name + suffix) for suffix in ('.mlir', '.ll', '.log')]
                inp.write_text(source)
                # Request async mode as well: checked kernels must safely keep
                # synchronous copies, rather than promote them to unchecked DMA.
                tail = bool(s % bm or s % bn)
                env = {**os.environ, 'FRISK_GM_TO_SHM_MODE': 'async' if tail else 'legacy'}
                command = [str(args.mytest.resolve()), f'--input={inp}', f'--output={out}',
                    '--math=mmac', '--llvm-opt=0', '--threads=128', '--block-warps=2,1',
                    '--wmma-layout=3.2', '--fuse-mask-add=0', '--uniform-masks=0']
                with log.open('w') as stream:
                    result = subprocess.run(command, cwd=directory, env=env,
                        stdout=stream, stderr=subprocess.STDOUT, timeout=180)
                diagnostics = log.read_text()
                assert result.returncode == 0, diagnostics[-6000:]
                assert not re.search(r'error:|Assertion .*failed|LLVM ERROR', diagnostics)
                text = out.read_text()
                if tail:
                    assert 'frisk.checked_global_io_lowered' in diagnostics
                    assert 'buffer_load_dword' not in text, 'tail copied by unchecked DMA'
                check(text, s, bm, d, b, h, causal)
                print('PASS LLVM tail bounds, numerics and untouched output:', name, flush=True)


if __name__ == '__main__':
    main()
