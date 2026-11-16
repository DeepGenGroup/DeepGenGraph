#!/usr/bin/env python3
"""Save verifiable stage IR and check the delivered kernel and math modes."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

here = Path(__file__).resolve().parent
repo = here.parents[1]
tool = repo / '3rd/deepgengraph/build/test/ThreadTilingTest'
mytest = repo / '3rd/deepgengraph/build/test/MyTest'
llvm = Path(os.environ.get('LLVM_BIN', '/data2/xsl/install/bin'))


def run(command, **kwargs):
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def stage(log, marker):
    tail = log.split(marker, 1)[1].split('\n', 1)[1]
    start = re.search(r'^(?:#\w+\s*=|module\b)', tail, re.M).start()
    module = re.search(r'^module\b', tail[start:], re.M).start() + start
    end = re.search(r'^}\s*$', tail[module:], re.M).end() + module
    return tail[start:end].rstrip() + '\n'


log = (here / 'lowering.log').read_text()
before = stage(log, 'after frisk-simplify-bounded-affine')
after = stage(log, 'after frisk-approximate-thread-math')
for name, text in [('thread.before.mlir', before), ('thread.after.mlir', after)]:
    (here / name).write_text(text)
    run([tool, here / name, '--verify-each', '-o', '/dev/null'])
assert before.count('arith.divf') == 16
assert after.count('arith.divf') == 2
assert after.count('fastmath<arcp>') == 2
assert 'Approximate thread math: 16 reciprocal divisions, 2 shared reciprocals' in log
assert 'Approximate register math:' not in log
run([tool, here / 'thread.before.mlir', '--pass-pipeline=builtin.module(cse,func.func(frisk-approximate-thread-math),canonicalize,cse)',
     '-o', here / 'thread.replayed.mlir'])
run([tool, here / 'thread.after.mlir', '-o', here / 'thread.after.normalized.mlir'])
assert (here / 'thread.replayed.mlir').read_text() == (here / 'thread.after.normalized.mlir').read_text()
run([llvm / 'opt', '-S', '-passes=default<O3>,verify', here / 'kernel.ll', '-o', here / 'kernel.o3.ll'])
run([llvm / 'opt', '-disable-output', '-passes=verify', here / 'kernel.ll'])
(here / 'kernel.o3.launch.json').write_bytes((here / 'kernel.launch.json').read_bytes())

def check_mode(mode):
    out = here / 'checks' / mode
    out.mkdir(parents=True, exist_ok=True)
    command = [mytest, here / 'input.mlir', '0', out / 'kernel.ll',
               '1', '1', '3.3', '34816', mode, '256', '4,1', '--dump-llvm-passes=1']
    with (out / 'lowering.log').open('w') as stream:
        run(command, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'FRISK_GM_TO_SHM_MODE': 'async', 'FRISK_GM_TO_SHM_ALIGNMENT': '16'})
    text = (out / 'lowering.log').read_text()
    translated = text.split('---- after llvm translateToLLVM -----', 1)[1].split(
        '---- after llvm optimizeRegisterMath -----', 1)[0]
    reciprocal = len(re.findall(r'\bfdiv arcp\b', translated))
    native = len(re.findall(r'call float @llvm\.amdgcn\.exp2\.f32\(', translated))
    ocml = len(re.findall(r'call float @__ocml_exp2_f32\(', translated))
    assert (reciprocal > 0) == (mode in ('approx', 'mmac-recip'))
    assert (native > 0 and ocml == 0) if mode in ('approx', 'mmac-exp2') else (ocml > 0 and native == 0)
    assert ('after frisk-approximate-thread-math' in text) == (mode in ('approx', 'mmac-recip'))
    assert 'Approximate register math:' not in text
    run([llvm / 'opt', '-disable-output', '-passes=verify', out / 'kernel.ll'])
    result = dict(translated_reciprocals=reciprocal, native_exp2=native, ocml_exp2=ocml)
    print('PASS mode', mode, result, flush=True)
    return mode, result


with ThreadPoolExecutor(max_workers=3) as executor:
    modes = dict(executor.map(check_mode, ('strict', 'mmac', 'mmac-exp2', 'mmac-recip', 'approx')))

report = dict(thread_divisions_before=16, thread_reciprocals_after=2,
              serialized_pass_replay_equal=True, llvm_verified=True, o3_verified=True,
              modes=modes, gpu_tested=False,
              sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                      for name in ('input.mlir', 'thread.before.mlir', 'thread.after.mlir', 'kernel.ll', 'kernel.o3.ll')})
(here / 'generation.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: thread IR replay, raw/O3 LLVM verifier and five math modes')
