#!/usr/bin/env python3
"""Reproduce the compiler pass OFF/ON pair; no LLVM text transformations."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / 'generated_ir/pipeline_cleanup/nohint_unrolled.mlir'
TOOL = ROOT / '3rd/deepgengraph/build/test/MyTest'
LLVM = Path('/data2/xsl/install/bin')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    env = dict(os.environ, FRISK_GM_TO_SHM_MODE='async', FRISK_GM_TO_SHM_ALIGNMENT='16')
    variants = {}
    config = json.loads((ROOT / 'generated_ir/q_bk32_retained/baseline.launch.json').read_text())
    for name, words in [('baseline', 0), ('retained_8words', 8)]:
        output = HERE / f'{name}.ll'
        cmd = [str(TOOL), str(SOURCE), '0', str(output),
               '1', '1', '3.3', '34816', 'approx', '256', '4,1',
               '1', '1', '1', '0', '0', 'useKPack', 'asm128', '32', '32',
               '--frag-prefetch=1', '--frag-prefetch-words=4',
               '--simplify-loop-control=1', '--coalesce-waits=0',
               '--workgroup-group-size=8', '--workgroup-reverse=true',
               '--swizzle-q=off', '--swizzle-k=on', '--swizzle-v=off',
               f'--retain-fragment-words={words}']
        with (HERE / f'{name}.compile.log').open('w') as log:
            subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([LLVM / 'llvm-as', output, '-o', '/dev/null'], check=True)
        subprocess.run([LLVM / 'opt', '-passes=default<O3>,verify', output, '-disable-output'], check=True)
        (HERE / f'{name}.launch.json').write_text(json.dumps(config, indent=2)+'\n')
        text = output.read_text()
        variants[name] = dict(sha256=sha(output), retain_words=words, command=cmd,
            dma_sites=text.count('buffer_load_dwordx4 '),
            mmac_sites=text.count('v_mmac_f32_16x16x16_f16 '),
            static_lds_arrays=re.findall(r'addrspace\(3\).*?\[(\d+) x i8\]', text),
            LLVM_assemble_and_O3_verify=True)
        print(name, json.dumps({k: v for k,v in variants[name].items() if k != 'command'}), flush=True)
    historical = ROOT / 'generated_ir/group8_reverse_k_gep/group8_reverse_k128_gep.ll'
    manifest = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=sha(SOURCE),
                    historical_best=str(historical.relative_to(ROOT)), historical_best_sha256=sha(historical),
                    note='Both variants are freshly lowered with identical settings except retention. The baseline is not a byte copy of the historical hand-edited LLVM winner.',
                    variants=variants, launch=config, gpu_tested=False)
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
