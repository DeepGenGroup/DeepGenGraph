#!/usr/bin/env python3
"""Export the compiler pass and compare its instruction graph with measured IR."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOL = ROOT / '3rd/deepgengraph/build/test/LegacyLLVMExportTest'
LLVM = Path('/data2/xsl/install/bin')
SOURCE = HERE.parent / 'noncausal_qk_tail_prefetch_pass/k_tail_prefetch4.ll'
MEASURED = HERE.parent / 'noncausal_cross_bn_k0_lifetimes/k0_sum_anchor.ll'


def run(args):
    return subprocess.run(list(map(str,args)), text=True, capture_output=True, check=True)


def canonical(text):
    body = text.split('define amdgpu_kernel void ',1)[1].split('\n}',1)[0]
    signature, body = body.split('\n',1)
    names = {v:f'%arg{n}' for n,v in enumerate(re.findall(r'%[\w.]+',signature))}
    for n,v in enumerate(re.findall(r'^([\w.]+):',body,re.M)):
        names['%'+v] = f'%block{n}'
    for n,v in enumerate(re.findall(r'^  (%[\w.]+) =',body,re.M)):
        names[v] = f'%value{n}'
    lines = []
    for l in body.splitlines():
        if not l.startswith('  ') or l.lstrip().startswith(';'):
            continue
        l = re.sub(r', ![\w.]+ !\d+', '', re.sub(r' #\d+', '', l.strip()))
        lines.append(re.sub(r'%[\w.]+',lambda m:names.get(m[0],m[0]),l))
    return lines


def main():
    source_manifest = json.loads((SOURCE.parent/'manifest.json').read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_manifest['variants']['k_tail_prefetch4']['sha256']
    measured_manifest = json.loads((MEASURED.parent/'manifest.json').read_text())
    assert hashlib.sha256(MEASURED.read_bytes()).hexdigest() == measured_manifest['variants']['k0_sum_anchor']['sha256']
    measured = run([TOOL, '--cross-bn-k0=0', MEASURED]).stdout
    records = {}
    for name, enabled in [('control',0),('cross_bn_k0',1)]:
        result = run([TOOL,f'--cross-bn-k0={enabled}',SOURCE])
        raw = HERE/(name+'.raw.ll'); raw.write_text(result.stdout)
        (HERE/(name+'.pass.log')).write_text(result.stderr)
        path = HERE/(name+'.ll')
        path.write_text(run([TOOL,'--llvm22-text',raw]).stdout)
        if enabled:
            actual, expected = canonical(result.stdout), canonical(measured)
            if actual != expected:
                import difflib
                (HERE/'comparison.diff').write_text('\n'.join(difflib.unified_diff(expected,actual,fromfile='measured',tofile='pass'))+'\n')
                raise AssertionError('Pass differs from measured instruction graph; inspect comparison.diff')
            assert 'packets=4, slot-bytes=8192, overlap-mmac=32, sum-anchors=64' in result.stderr
        else:
            assert canonical(result.stdout) == canonical(run([TOOL,'--cross-bn-k0=0',SOURCE]).stdout)
            path.write_bytes(SOURCE.read_bytes())
        path.with_suffix('.launch.json').write_bytes(SOURCE.with_suffix('.launch.json').read_bytes())
        run([LLVM/'llvm-as',path,'-o','/dev/null'])
        opt = path.with_suffix('.o3.ll')
        run([LLVM/'opt','-S','-passes=default<O3>,verify',path,'-o',opt])
        if enabled:
            optimized = opt.read_text()
            join = optimized.index('k0.join:')
            assert len(re.findall(r'call float asm sideeffect "", "=v,0"',optimized[:join])) == 64
        records[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            o3_sha256=hashlib.sha256(opt.read_bytes()).hexdigest(), diagnostics=result.stderr.strip(),
            identical_instruction_graph_to_measured=bool(enabled), LLVM_and_O3_verified=True)
        print('PASS generated',name,result.stderr.strip(),flush=True)
    (HERE/'manifest.json').write_text(json.dumps(dict(variants=records,source=str(SOURCE),
        measured=str(MEASURED), measured_sha256=hashlib.sha256(MEASURED.read_bytes()).hexdigest(),
        exporter_sha256=hashlib.sha256(TOOL.read_bytes()).hexdigest(),gpu_tested=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
