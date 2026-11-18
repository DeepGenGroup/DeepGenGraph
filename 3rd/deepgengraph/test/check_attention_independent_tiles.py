#!/usr/bin/env python3
"""Independent block tiling, shared tensor IR and actual LLVM work counts.

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
        # This suite generates the template's query=x/head=y entry order.
        # The historical LLVM oracle defaults to query=y/head=x.
        if '@llvm.amdgcn.workgroup.id.x(' in body:
            return np.full(self.t, self.block, dtype=np.int64)
        if '@llvm.amdgcn.workgroup.id.y(' in body:
            return np.full(self.t, self.head, dtype=np.int64)
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
    source = generate_ir(TEMPLATE.read_text(), 128, 128, 16, softmax='legacy')[0]
    env = {**os.environ, 'FRISK_GM_TO_SHM_MODE': 'async', 'FRISK_GM_TO_SHM_ALIGNMENT': '16'}
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = nullcontext(args.output_dir.resolve()) if args.output_dir else tempfile.TemporaryDirectory(prefix='independent-tiles-')
    with directory as tmp:
        root = Path(tmp)
        report = {}

        def compile_case(name, text=source, tiles=(), ok=True):
            inp, out = root / (name + '.mlir'), root / (name + '.ll')
            inp.write_text(text)
            # A failed invocation must never be confused with an old artifact.
            out.unlink(missing_ok=True)
            command = [str(tool), str(inp), '0', str(out), '1', '1', '3.3',
                       'approx', '256', '4,1', '1', '1', '1', '0', '0', 'useKPack', 'asm128', *map(str, tiles),
                       '--workgroup-query-axis=x', '--workgroup-head-axis=y']
            result = subprocess.run(command, env=env, cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=240)
            (root / (name + '.log')).write_text(result.stdout)
            if not ok:
                assert result.returncode == 1 and not out.exists(), result.stdout[-6000:]
                return result.stdout
            assert result.returncode == 0 and not re.search(r'error:|Assertion .*failed|LLVM ERROR', result.stdout), result.stdout[-6000:]
            tensor = result.stdout.split('after scfForConversion ---------', 1)[1].split('----------', 1)[0]
            assert tensor.count('deepgengraph.precise_dot_op') == 2
            assert tensor.count('deepgengraph.exp2') == 1
            block = result.stdout.split('after frisk-tile-attention ---------', 1)[1].split('----------', 1)[0]
            assert block.count('frisk.exp2') == 1, 'softmax duplicated in block tiling'
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
        # Hints for the two axes are independent too.
        unequal = source.replace('frisk.attention_tiles = array<i64: 16, 16>',
                                 'frisk.attention_tiles = array<i64: 16, 32>')
        compile_case('unequal_input_hints', unequal)
        assert 'frisk.attention_tiled = array<i64: 32, 64>' in (root / 'unequal_input_hints.log').read_text()
        other_shape = generate_ir(TEMPLATE.read_text(), 64, 64, 32, softmax='legacy')[0]
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

        for name, tiles, message in [
            ('invalid_qk', (24, 32), 'positive multiples of 16'),
            ('invalid_vo', (64, 256), 'positive multiples of 16'),
            ('negative_tile', (-1, 32), 'must be nonnegative'),
        ]:
            log = compile_case(name, tiles=tiles, ok=False)
            assert message in log, log[-6000:]
            print('PASS rejected:', name, flush=True)
        (root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS independent block tiles; smaller VO does not repeat QK/softmax')


if __name__ == '__main__':
    main()
