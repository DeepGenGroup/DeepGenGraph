#!/usr/bin/env python3
"""Thread lane-sum legality, vector state, replay, lowering and lane execution."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np
from check_defer_lane_sum import Model

TOOL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'build/test/ThreadTilingTest'
LLVM = Path('/data2/xsl/install/bin')
PASS = '--frisk-thread-defer-lane-sum='


def run(command, source):
    p = subprocess.run(list(map(str, command)), input=source, text=True,
                       capture_output=True, timeout=180)
    assert p.returncode == 0, p.stderr
    return p.stdout


def transform(source, mode='approximate=true wave-size=64'):
    return run([TOOL, '--verify-each', PASS + mode], source)


def fixture(integer=False, packed=False, weighted=True, running_max=False,
            bits=(1, 2), wave=64, trips=7, loop='affine', initial=None,
            escape='', alpha='constant', threads=128, attrs='', width=None):
    ty = 'i32' if integer else 'f32'
    add, mul = ('addi', 'muli') if integer else ('addf', 'mulf')
    z = initial if initial is not None else ('0' if integer else '0.0')
    st = f'vector<2x1x{ty}>' if packed else ty
    lines = [f'func.func @statistics() -> {ty} attributes {{thread_num = {threads} : i32 {attrs}}} {{',
             '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index',
             f'%end = arith.constant {trips} : index',
             f'%width = arith.constant {width or wave} : i32',
             f'%zero = arith.constant {z} : {ty}',
             f'%init = arith.constant {"dense<"+z+">" if packed else z} : {st}',
             f'%factor = arith.constant {"3" if integer else "0.5"} : {ty}',
             '%tid = gpu.thread_id x', '%lane = arith.index_cast %tid : index to i32']
    for b in set(bits):
        lines += [f'%b{b} = arith.constant {b} : i32']
    bounds = f'0 to {trips}' if loop == 'affine' else '%c0 to %end step %c1'
    count = 2 if running_max else 1
    extra = ', %maximum = %zero' if running_max else ''
    result_types = st + (', ' + ty if running_max else '')
    lines += [f'%result:{count} = {loop}.for %iv = {bounds} iter_args(%state = %init{extra}) -> ({result_types}) {{',
              '%iter = arith.index_cast %iv : index to i32',
              '%seed = arith.xori %lane, %iter : i32']
    if integer:
        lines += ['%big = arith.constant 100000003 : i32', '%x = arith.addi %seed, %big : i32']
    else:
        lines += ['%x = arith.uitofp %seed : i32 to f32']
    factor = '%factor'
    if running_max:
        prev = '%x'
        for i, b in enumerate(bits):
            lines += [f'%mpeer{i}, %mv{i} = gpu.shuffle xor {prev}, %b{b}, %width : {ty}',
                      f'%m{i} = arith.maxnumf {prev}, %mpeer{i} : {ty}']
            prev = f'%m{i}'
        lines += [f'%nextmax = arith.maxnumf %maximum, {prev} : f32',
                  '%delta = arith.subf %maximum, %nextmax : f32',
                  '%alpha = math.exp2 %delta : f32']
        factor = '%alpha'
    elif alpha == 'lane': factor = '%x'
    elif alpha == 'group':
        mask = (wave - 1) ^ sum(set(bits))
        lines += [f'%mask = arith.constant {mask} : i32',
                  '%group = arith.andi %lane, %mask : i32',
                  '%groupf = arith.uitofp %group : i32 to f32',
                  '%alpha = arith.mulf %groupf, %factor : f32']
        factor = '%alpha'
    elif alpha == 'varying':
        lines += ['%fi = arith.uitofp %iter : i32 to f32',
                  '%alpha = arith.addf %fi, %factor : f32']
        factor = '%alpha'
    for n in range(2 if packed else 1):
        old = f'%old{n}' if packed else '%state'
        if packed:
            lines += [f'{old} = vector.extract %state[{n}, 0] : {ty} from {st}']
        prev = '%x'
        for i, b in enumerate(bits):
            lines += [f'%peer{n}_{i}, %v{n}_{i} = gpu.shuffle xor {prev}, %b{b}, %width : {ty}',
                      f'%sum{n}_{i} = arith.{add} {prev}, %peer{n}_{i} : {ty}']
            prev = f'%sum{n}_{i}'
        # Match actual attention's vector<1xf32> wrappers as well as scalars.
        if packed:
            lines += [f'%ov{n} = vector.broadcast {old} : {ty} to vector<1x{ty}>',
                      f'%sv{n} = vector.broadcast {prev} : {ty} to vector<1x{ty}>',
                      f'%av{n} = vector.broadcast {factor} : {ty} to vector<1x{ty}>']
            vt = f'vector<1x{ty}>'
            if weighted:
                lines += [f'%scaled{n} = arith.{mul} %ov{n}, %av{n} : {vt}']
            lines += [f'%uv{n} = arith.{add} %sv{n}, {"%scaled"+str(n) if weighted else "%ov"+str(n)} : {vt}',
                      f'%update{n} = vector.extract %uv{n}[0] : {ty} from {vt}']
        else:
            if weighted:
                lines += [f'%scaled{n} = arith.{mul} {old}, {factor} : {ty}']
            lines += [f'%update{n} = arith.{add} {prev}, {"%scaled"+str(n) if weighted else old} : {ty}']
        if escape:
            v = {'state': old, 'sum': prev, 'update': f'%update{n}'}[escape]
            # An observable store must prevent any changed value escaping.
            lines += [f'%mem{n} = memref.alloca() : memref<1x{ty}>',
                      f'memref.store {v}, %mem{n}[%c0] : memref<1x{ty}>']
    if packed:
        lines += [f'%back = vector.from_elements %update0, %update1 : {st}']
    lines += [f'{loop}.yield {"%back" if packed else "%update0"}{", %nextmax" if running_max else ""} : {result_types}', '}']
    if packed:
        lines += [f'%a = vector.extract %result#0[0, 0] : {ty} from {st}',
                  f'%b = vector.extract %result#0[1, 0] : {ty} from {st}',
                  f'%out = arith.{add} %a, %b : {ty}', f'return %out : {ty}']
    else:
        lines += [f'return %result#0 : {ty}']
    return '\n'.join(lines + ['}'])


def unchanged(source, options='approximate=true wave-size=64'):
    assert transform(source, options) == run([TOOL], source)


for kw in [dict(initial='1.0'), dict(alpha='lane'), dict(threads=96),
           dict(bits=(1, 1)), dict(bits=(3,)), dict(width=32),
           *[dict(escape=x, packed=p) for x in ('state', 'sum', 'update') for p in (False, True)]]:
    unchanged(fixture(**kw))
for attr in ('strictfp', 'llvm.optnone', 'passthrough = ["strictfp"]'):
    unchanged(fixture(attrs=', ' + attr))
unchanged(fixture(), 'wave-size=64')
unchanged(fixture(), 'approximate=true')
unchanged(fixture(loop='scf').replace('to %end step', 'to %tid step'))
unchanged(fixture().replace('%x = arith.uitofp',
    'llvm.inline_asm has_side_effects "", "~{exec}" : () -> ()\n%x = arith.uitofp'))

unchanged(fixture(attrs=', "target-features" = "+wavefrontsize32"'))
unchanged(fixture(attrs=', passthrough = [["target-features", "-wavefrontsize64"]]'))
unchanged(fixture().replace('thread_num = 128 : i32', 'test.no_launch_contract'))
opaque = fixture().replace('%x = arith.uitofp',
    'func.call @opaque() : () -> ()\n%x = arith.uitofp')
unchanged('func.func private @opaque()\n' + opaque)
loaded = fixture().replace('%scaled0 =',
    '%mem = memref.alloca() : memref<1xf32>\n'
    '%unknown = memref.load %mem[%c0] : memref<1xf32>\n%scaled0 =')
unchanged(loaded.replace('arith.mulf %state, %factor', 'arith.mulf %state, %unknown'))
# The same recurrence under a partial-wave branch cannot be moved to its exit.
divergent = fixture().replace('%result:1 =',
    '%cond = arith.cmpi eq, %lane, %b1 : i32\n'
    '%wrapped = scf.if %cond -> f32 {\n%result:1 =').replace(
    'return %result#0 : f32', 'scf.yield %result#0 : f32\n} else {\n'
    'scf.yield %zero : f32\n}\nreturn %wrapped : f32')
unchanged(divergent)
flagged = fixture(integer=True).replace('arith.muli %state, %factor :',
    'arith.muli %state, %factor overflow<nsw, nuw> :').replace(
    'arith.addi %sum0_1, %scaled0 :',
    'arith.addi %sum0_1, %scaled0 overflow<nsw, nuw> :')
assert 'overflow<' not in transform(flagged, 'wave-size=64')
for option in ('wave-size=16', 'wave-size=65'):
    p = subprocess.run([str(TOOL), PASS + option], input=fixture(), text=True,
                       capture_output=True)
    assert p.returncode != 0 and 'wave-size must be' in p.stderr

# Numerical validation executes the LLVM produced from each MLIR version.
class Execution(Model):
    def memory(self, ptr, ty, write=None):
        assert write is not None
        self.result = write


def lower(source):
    # Keep a scalar return for the CPU adapter, without assigning a kernel ABI.
    source = source.replace('thread_num =', 'test.thread_num =')
    lowered = run([TOOL, '--test-async-to-llvm'], source)
    ll = run([LLVM / 'mlir-translate', '--mlir-to-llvmir'], lowered)
    run([LLVM / 'opt', '-passes=default<O3>,verify', '-disable-output'], ll)
    ll = run([LLVM / 'opt', '-S', '-passes=instnamer'], ll)
    # Adapt the scalar return to the existing lane executor's kernel ABI.
    body = re.search(r'define[^\n]+@statistics\([^\n]*\n(.*?)\n}', ll, re.S)[1]
    entry = re.match(r'(\w+):', body)[1]
    body = re.sub(r'ret (float|i32) ([^\n]+)',
                  r'store \1 \2, ptr addrspace(1) %3\n  ret void', body)
    return ('@dummy = internal addrspace(3) global [1 x i8] zeroinitializer\n'
            'define amdgpu_kernel void @statistics(ptr addrspace(1) %0, '
            'ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) {\n'
            f'  br label %{entry}\n' + body + '\n}\n')


pairs = 0
for integer in (False, True):
    for packed in (False, True):
        for loop in ('affine', 'scf'):
            for weighted in (False, True):
                kw = dict(integer=integer, packed=packed, loop=loop, weighted=weighted)
                options = f'approximate={str(not integer).lower()} wave-size=64'
                source = fixture(**kw)
                result = transform(source, options)
                assert result.count('gpu.shuffle') == source.count('gpu.shuffle')
                assert 'gpu.shuffle' not in result.split(loop + '.yield')[0]
                assert transform(result, options) == result
                for trips in (0, 1, 2, 7, 31):
                    source = fixture(trips=trips, **kw)
                    result = transform(source, options)
                    values, counts = [], []
                    for s in (source, result):
                        model = Execution(lower(s), 64, integer)
                        model.run()
                        values.append(model.result)
                        counts.append(model.shuffles)
                    if integer: np.testing.assert_array_equal(*values)
                    else: np.testing.assert_allclose(*values, rtol=3e-6, atol=2e-4)
                    stages = 2 * (2 if packed else 1)
                    assert counts == [trips * stages, stages], counts
                    pairs += 1
for bits, wave in [((16, 32), 64), ((1, 2, 4, 8, 16), 32), ((2, 8, 32), 64)]:
    source = fixture(bits=bits, wave=wave)
    result = transform(source, f'approximate=true wave-size={wave}')
    assert 'gpu.shuffle' not in result.split('affine.yield')[0]
for packed in (False, True):
    for kw in (dict(running_max=True), dict(alpha='varying'), dict(alpha='group', bits=(16, 32))):
        source = fixture(packed=packed, **kw)
        result = transform(source)
        expected = 2 if kw.get('running_max') else 0
        assert result.split('affine.yield')[0].count('gpu.shuffle') == expected
        for s in (source, result):
            model = Execution(lower(s), 64, False)
            model.run()
            if s == source: before = model.result
            else: np.testing.assert_allclose(before, model.result, rtol=3e-6, atol=2e-4)
        pairs += 1
print(f'PASS: legality/rejection, scalar/vector states, affine/scf, wave32/64, '
      f'idempotence, LLVM verifier/O3, {pairs} numerical execution pairs')
