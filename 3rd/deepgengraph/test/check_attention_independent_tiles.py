#!/usr/bin/env python3
"""Independent split-D axis rewriting, sharing guard and actual LLVM work counts.

Runs the wave-level CPU model, not gfx936 machine code or asynchronous hazards.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from check_attention_split_d import check_numerics
from check_attention_llvm_numerics import LLVMWorkgroupModel, reference
from generate_attention_ir import TEMPLATE, generate_ir


class CountedModel(LLVMWorkgroupModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_count = 0

    def call(self, body):
        if re.search(r'@(?:llvm.amdgcn.exp2.f32|llvm.exp2.f32|__ocml_exp2_f32)\(', body):
            self.exp_count += 1
        return super().call(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mytest', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    tool = args.mytest.resolve()
    here = Path(__file__).resolve().parent
    source = (here / 'test_input_spidD.mlir').read_text()
    env = {**os.environ, 'FRISK_GM_TO_SHM_MODE': 'async', 'FRISK_GM_TO_SHM_ALIGNMENT': '16'}
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = nullcontext(args.output_dir.resolve()) if args.output_dir else tempfile.TemporaryDirectory(prefix='independent-tiles-')
    with directory as tmp:
        root = Path(tmp)
        report = {}

        def compile_case(name, text=source, tiles=(), budget=34816, ok=True):
            inp, out = root / (name + '.mlir'), root / (name + '.ll')
            inp.write_text(text)
            # A failed invocation must never be confused with an old artifact.
            out.unlink(missing_ok=True)
            command = [str(tool), str(inp), '0', str(out), '1', '1', '3.3', str(budget),
                       'approx', '256', '4,1', '1', '1', '1', '0', '0', 'useKPack', 'asm128', *map(str, tiles)]
            result = subprocess.run(command, env=env, cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=240)
            (root / (name + '.log')).write_text(result.stdout)
            if not ok:
                assert result.returncode == 1 and not out.exists(), result.stdout[-6000:]
                return result.stdout
            assert result.returncode == 0 and not re.search(r'error:|Assertion .*failed|LLVM ERROR', result.stdout), result.stdout[-6000:]
            assert 'split-D sharing: QK copies=1, softmax copies=1 per key tile' in result.stdout
            ir = out.read_text()
            subprocess.run(['/data2/xsl/install/bin/opt', '-passes=verify', '-disable-output', str(out)], check=True)
            assert 'frisk.split_d_' not in ir
            # Check the post-normalization algorithm, independently of wave lowering.
            check_numerics(result.stdout, 128 if name != 'bm64_bn64' else 64,
                           seed=5, blocks=[(0, 0, 0), (0, 1, 1)])
            report[name] = {'command': command, 'llvm_verify': 'PASS', 'shared_qk_softmax': True}
            print('PASS compile, sharing and tensor numerics:', name, flush=True)
            return ir

        baseline = compile_case('default')
        variants = {'default': baseline,
                    'qk64_vo32': compile_case('qk64_vo32', tiles=(64, 32)),
                    'qk32_vo64': compile_case('qk32_vo64', tiles=(32, 64)),
                    'auto_qk_vo32': compile_case('auto_qk_vo32', tiles=(0, 32))}
        # Initial QK and VO steps may differ too; each default is independent.
        unequal = source.replace('%cBK = arith.constant 32', '%cBK = arith.constant 16')
        start, end = unequal.index('          %qk ='), unequal.index('          %mask =')
        region = unequal[start:end].replace('tensor<128x32xf16>', 'tensor<128x16xf16>').replace('tensor<32x128xf16>', 'tensor<16x128xf16>')
        region = region.replace('block_shape = [128, 32]', 'block_shape = [128, 16]').replace('block_shape = [32, 128]', 'block_shape = [16, 128]')
        unequal = unequal[:start] + region + unequal[end:]
        compile_case('unequal_input_steps', unequal)
        assert 'QK=16 -> 32, VO=32 -> 64' in (root / 'unequal_input_steps.log').read_text()
        other_shape = generate_ir(TEMPLATE.read_text(), 64, 64, 32)[0]
        compile_case('bm64_bn64', other_shape, tiles=(64, 32))

        rng = np.random.default_rng(42)
        q, k, v = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        reference_counts = {}
        for name, ir in variants.items():
            measurements = []
            for head, block in [(0, 0), (1, 1), (1, 7)]:
                model = CountedModel(ir, q, k, v, head=head, block=block)
                got = model.run().reshape(q.shape)[head, block*128:(block+1)*128].astype(np.float32)
                want = reference(q, k, v, head, block)
                assert np.allclose(got, want, atol=.01, rtol=.01), (name, head, block, np.max(np.abs(got-want)))
                counts = model.mmac_count, model.exp_count
                if name == 'default': reference_counts[head, block] = counts
                assert counts == reference_counts[head, block], ('repeated work', name, counts, reference_counts)
                assert counts == (256 * (block+1), 64 * (block+1)), counts
                measurements.append({'head': head, 'block': block, 'mmac': counts[0],
                                     'exp2': counts[1], 'max_abs': float(np.max(np.abs(got-want)))})
            report[name]['llvm_execution'] = measurements
            print('PASS LLVM numerics and unchanged dynamic MMAC/exp2 work:', name, measurements, flush=True)

        for name, tiles, budget, message in [
            ('invalid_qk', (24, 32), 34816, 'positive multiples of 16'),
            ('invalid_vo', (64, 256), 34816, 'positive multiples of 16'),
            ('negative_tile', (-1, 32), 34816, 'must be nonnegative'),
            ('partial_sharing', (64, 32), 28672, 'would repeat QK/softmax'),
        ]:
            log = compile_case(name, tiles=tiles, budget=budget, ok=False)
            assert message in log, log[-6000:]
            if name == 'partial_sharing':
                assert 'QK copies=2, softmax copies=2' in log
            print('PASS rejected:', name, flush=True)
        (root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS independent tiles; smaller VO does not repeat QK/softmax; budget fallback rejected')


if __name__ == '__main__':
    main()
