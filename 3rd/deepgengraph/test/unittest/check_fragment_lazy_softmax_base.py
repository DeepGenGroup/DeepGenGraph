#!/usr/bin/env python3
"""Legacy attention pipeline: early lazy rebase, rejection, lowering and numerics.

Executes the exported LLVM using the repository's mathematical wave model;
this does not measure GPU performance or model independent wave timing.
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from check_attention_llvm_numerics import LLVMWorkgroupModel
from check_auto_pipeline_attention import module_after

ROOT = Path(__file__).resolve().parents[4]
BUILD = ROOT / '3rd/deepgengraph/build/test'
LLVM = Path('/data2/xsl/install/bin')
PASS = '--frisk-fragment-lazy-softmax-base'


def run(args, source=None, ok=True):
    p = subprocess.run(list(map(str, args)), input=source, text=True,
                       capture_output=True, timeout=300,
                       env={**os.environ, 'FRISK_GM_TO_SHM_MODE': 'async',
                            'FRISK_GM_TO_SHM_ALIGNMENT': '16'})
    assert (p.returncode == 0) == ok, p.stderr[-12000:]
    return p


def transform(source, *passes, ok=True):
    return run([BUILD / 'ThreadTilingTest', '--verify-each', *passes], source, ok).stdout


class CountVotes(LLVMWorkgroupModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.votes = []

    def call(self, body):
        result = super().call(body)
        if 'llvm.amdgcn.ballot.i64' in body:
            self.votes.append(bool(np.any(result)))
        return result


def main():
    vote = ('func.func @vote(%p: i1) -> i1 { '
            '%v = frisk.wave_any %p {wave_size = 64 : i64} return %v : i1 }')
    assert 'rocdl.ballot' in transform(vote, '--convert-fragment-to-thread')
    transform(vote.replace('64 : i64', '32 : i64'), ok=False)
    with tempfile.TemporaryDirectory(prefix='fragment-lazy-') as tmp:
        work = Path(tmp)
        source = work / 'input.mlir'
        run([sys.executable, ROOT / 'generate_attention_ir.py', '--b', 1, '--h', 1,
             '--s', 512, '--d', 32, '--bm', 128, '--bn', 128, '--bk', 32,
             '--non-causal', '--output', source])
        exports = {}
        fragment = None
        for name, option, extra in [
                ('off', 'off', []), ('on', '8', []),
                ('scheduled', '8', ['--defer-lane-sum=approx', '--k-lds-prefetch=2',
                                    '--softmax-fragments=pv', '--pv-interleave=4'])]:
            output = work / (name + '.ll')
            command = [BUILD / 'MyTest', source, 0, output, 1, 1, '3.3', 'approx',
                       256, '4,1', 1, 1, 1, 0, 0, 'useKPack', 'asm128', 32, 32,
                       '--lazy-softmax-base=' + option, '--llvm-opt=0', *extra]
            log = work / (name + '.log')
            with log.open('w') as stream:
                p = subprocess.run(list(map(str, command)), stdout=stream, stderr=stream,
                                   timeout=300, env={**os.environ,
                                       'FRISK_GM_TO_SHM_MODE': 'async',
                                       'FRISK_GM_TO_SHM_ALIGNMENT': '16'})
            diagnostic = log.read_text()
            assert p.returncode == 0, diagnostic[-12000:]
            exports[name] = output.read_text()
            if name == 'off':
                fragment = module_after(diagnostic, '---- after fragment-level fuse-fragment-accumulator -----')
            if name != 'off':
                assert 'Fragment lazy softmax base: rows=2' in diagnostic, diagnostic[-16000:]
                assert 'Thread lazy softmax base:' not in diagnostic
                assert 'llvm.amdgcn.ballot.i64' in exports[name]
                scheduled = module_after(diagnostic, '---- after frisk-fragment-scheduling -----')
                assert 'frisk.wave_any' in scheduled and 'rocdl.ballot' not in scheduled
                if name == 'scheduled':
                    assert 'frisk.lane_reduce_frag' in scheduled
            run([LLVM / 'opt', '-passes=default<O3>,verify', '-disable-output', output])
            print(f'PASS {name}: legacy export and LLVM/O3 verify', flush=True)

        # Exercise the standalone pass on real frontend/tiler output.
        result = transform(fragment, PASS)
        assert 'frisk.lazy_softmax_base = "fragment-v1"' in result
        assert 'frisk.wave_any' in result and 'rocdl.ballot' not in result
        assert 'frisk.lazy_analysis_origin' not in result
        assert transform(result, PASS) == result
        for label in ('gemm_mn', 'gemm_k'):
            assert result.count(f'iterLabel = "{label}"') == fragment.count(f'iterLabel = "{label}"')
        slow = result.split('scf.if', 1)[1].split('} else {', 1)[0]
        assert 'elementwise_frag "exp2"' in slow
        assert not any(op in slow for op in ('gemm_frag', 'load_frag', 'store_frag',
                                             'reduce_frag', 'wave_any', 'wait_buffer', 'sync_threads'))
        thread = transform(result, '--convert-fragment-to-thread')
        assert 'rocdl.ballot' in thread and 'frisk.wave_any' not in thread
        for options in ('threshold=0', 'threshold=-1', 'threshold=16',
                        'threshold=nan', 'threshold=inf', 'wave-size=32'):
            transform(fragment, PASS + '=' + options, ok=False)
        for threshold in (0.125, 4, 15):
            assert 'frisk.wave_any' in transform(fragment, PASS + '=threshold=' + str(threshold))
        rejects = [
            fragment.replace('thread_num = 256', 'thread_num = 32'),
            fragment.replace('thread_num = 256 : i32', 'thread_num = 256 : i32, strictfp'),
            fragment.replace('thread_num = 256 : i32',
                             'thread_num = 256 : i32, llvm.target_features = "+wavefrontsize32"'),
            fragment.replace('0xFF800000', '0.0'),
            fragment.replace('lane_stride = 16', 'lane_stride = 8'),
            fragment.replace('elementwise_frag "mul"', 'elementwise_frag "add"'),
        ]
        observed = fragment.replace('module {',
            'module {\nfunc.func private @observe(vector<2x1xf32>)', 1)
        peak = re.search(r'(%\w+) = frisk.unpack_frag %\w+ : <vector<2x1xf32>, \{\}> -> vector<2x1xf32>', observed)
        assert peak
        rejects.append(observed.replace(peak[0], peak[0] +
            f'\nfunc.call @observe({peak[1]}) : (vector<2x1xf32>) -> ()', 1))
        recurrence = re.search(r'(%\w+):3 = affine.for [^\n]*step 128 iter_args', observed)
        assert recurrence
        prefix, suffix = observed.rsplit('    return', 1)
        rejects.append(prefix +
            f'    func.call @observe({recurrence[1]}#2) : (vector<2x1xf32>) -> ()\n    return' + suffix)
        # A bias/nonlinear score producer cannot be seeded as a bare QK MMA.
        cast = re.search(r'elementwise_frag "cast" (%\w+) : '
                         r'\(!frisk.fragment<vector<1x4xf32>, \{\}>\)', fragment)
        assert cast
        rejects.append(fragment[:cast.start()] + fragment[cast.start():].replace(
            'elementwise_frag "cast"', 'elementwise_frag "exp2"', 1))
        for rejected in rejects:
            assert transform(rejected, PASS) == transform(rejected), 'rejection modified input'
        print('PASS fragment rewrite, preserved GEMM loops, idempotence and legality', flush=True)

        rng = np.random.default_rng(42)
        inputs = [[rng.normal(size=(1, 512, 32)).astype(np.float16) for _ in range(3)]]
        q = np.ones((1, 512, 32), dtype=np.float16)
        k = np.repeat(np.array([-1, -1.2, 1, 1.1], dtype=np.float16), 128)[None, :, None]
        k = np.broadcast_to(k, q.shape).copy()
        v = rng.normal(size=q.shape).astype(np.float16)
        inputs.append([q, k, v])
        for case, (q, k, v) in enumerate(inputs):
            score = q[0, :128].astype(np.float32) @ k[0].astype(np.float32).T / np.sqrt(32)
            weight = np.exp(score - score.max(axis=-1, keepdims=True))
            expected = weight @ v[0].astype(np.float32) / weight.sum(axis=-1, keepdims=True)
            baseline = None
            for name, text in exports.items():
                model = CountVotes(text, q, k, v, threads=256, head=0, block=0)
                actual = model.run().reshape(q.shape)[0, :128].astype(np.float32)
                assert np.isfinite(actual).all()
                assert np.max(np.abs(actual - expected)) < 0.015, (case, name)
                if baseline is None:
                    baseline = actual
                else:
                    assert np.max(np.abs(actual - baseline)) < 0.005, (case, name)
                    if case == 1:
                        assert any(model.votes) and not all(model.votes), model.votes
                print(f'PASS numeric case={case} {name}: max error={np.max(np.abs(actual - expected)):.6g}', flush=True)


if __name__ == '__main__':
    main()
