#!/usr/bin/env python3
"""Late ThreadIR global prefetch: budgets, SSA, control/effect boundaries, export."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

from check_thread_fragment_schedules import graph

ROOT = Path(__file__).resolve().parents[2]
PASS = '--frisk-thread-prefetch-global-packets'


def fixture(count=5, width=4, kind='vector', boundary='', attrs='', exp_type='f32'):
    ty = f'vector<{width}xf16>' if kind in ('vector', 'affine-vector') else 'f16'
    types = ', '.join([exp_type] + [ty] * count)
    lines = ['module {', 'func.func private @unknown()',
             f'func.func @kernel(%p: memref<256xf16, 1>, %sm: memref<256xf16, 3>, '
             f'%x: {exp_type}, %idx: index, %cond: i1) -> ({types}) {attrs} {{',
             '%zero = arith.constant 0 : index',
             '%half = arith.constant 0.0 : f16',
             '%one = arith.constant 1 : i32',
             f'%e = math.exp2 %x : {exp_type}', boundary]
    for i in range(count):
        lines += [f'%offset{i} = arith.constant {i * width} : index',
                  f'%addr{i} = arith.addi %idx, %offset{i} : index']
        if kind == 'vector':
            load = f'vector.load %p[%addr{i}] : memref<256xf16, 1>, {ty}'
        elif kind == 'affine-vector':
            load = f'affine.vector_load %p[%addr{i}] : memref<256xf16, 1>, {ty}'
        else:
            load = f'{kind}.load %p[%addr{i}] : memref<256xf16, 1>'
        lines.append(f'%v{i} = {load}')
    values = ', '.join(['%e'] + [f'%v{i}' for i in range(count)])
    return '\n'.join(lines + [f'return {values} : {types}', '}', '}'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, default=ROOT / 'build/test/ThreadTilingTest')
    parser.add_argument('--llvm-bin', type=Path, default=Path('/data2/xsl/install/bin'))
    parser.add_argument('--legacy-tool', type=Path, default=ROOT / 'build/test/LegacyLLVMExportTest')
    parser.add_argument('--driver', type=Path, help='Also check MyTest enable/disable and CP-SAT wiring')
    args = parser.parse_args()

    def command(argv, source, **kwargs):
        result = subprocess.run(list(map(str, argv)), input=source, text=True,
                                capture_output=True, timeout=180, **kwargs)
        assert result.returncode == 0, result.stderr
        return result.stdout

    def run(source, *passes):
        return command([args.tool, '--verify-each', *passes], source)

    cases = 0

    def check(source, expected, budget=8, check_graph=True):
        nonlocal cases
        original = run(source)
        option = PASS + f'=max-words={budget}'
        result = run(original, option)
        early = result.split('math.exp2')[0]
        loads = r'(?:vector\.load|memref\.load|affine\.load|affine\.vector_load) '
        assert len(re.findall(loads, early)) == expected, result
        assert len(re.findall(loads, result)) == len(re.findall(loads, original)), result
        if check_graph:
            assert graph(result) == graph(original), result
        assert run(result, option) == result, 'replay exceeded the packet budget'
        cases += 1
        return result

    for kind, width, words in [('vector', 4, 2), ('vector', 8, 4),
                               ('vector', 16, 0), ('memref', 1, 1),
                               ('affine', 1, 1), ('affine-vector', 4, 2)]:
        for budget in (0, 1, 3, 8):
            check(fixture(kind=kind, width=width), min(5, budget // words) if words else 0, budget)
    check(fixture(exp_type='vector<4xf32>'), 4)
    for exp_type in ('f16', 'f64', 'vector<4xf16>'):
        check(fixture(exp_type=exp_type), 0)
    check(fixture(boundary='memref.store %half, %sm[%zero] : memref<256xf16, 3>'), 4)
    lds = '%lds = frisk.lds_load %sm[%zero] {alignment = 8 : i64, byte_offset = 0 : i64} : memref<256xf16, 3> -> vector<4xf16>'
    check(fixture(boundary=lds), 4)
    check(fixture(boundary=lds.replace('alignment =', 'frisk.lds_pair = 0 : i64, alignment =')), 0)
    check(fixture(boundary='%s, %valid = gpu.shuffle xor %one, %one, %one : i32'),
          4, check_graph=False)

    for boundary in ('gpu.barrier', 'frisk.sync_threads_in_block', 'func.call @unknown() : () -> ()',
                     'frisk.sched_barrier {mask = 0 : i32}',
                     'memref.store %half, %p[%zero] : memref<256xf16, 1>',
                     'scf.if %cond { scf.yield }',
                     '%a = memref.atomic_rmw addf %half, %p[%zero] : (f16, memref<256xf16, 1>) -> f16'):
        check(fixture(boundary=boundary), 0)

    for attrs in ('strictfp', 'llvm.optnone', 'llvm.passthrough = ["strictfp"]'):
        check(fixture(attrs='attributes {' + attrs + '}'), 0)
    check(fixture().replace('module {', 'module attributes {llvm.target_triple = "nvptx64-nvidia-cuda"} {'), 0)
    no_exp = run(fixture().replace('math.exp2', 'math.sqrt'))
    assert graph(run(no_exp, PASS)) == graph(no_exp)

    dma = '%task = frisk.async_copy_gm_to_shm %p to %sm : memref<256xf16, 1>, memref<256xf16, 3> -> !frisk.async_task'
    check(fixture(boundary=dma), 0)
    wait = fixture(boundary='frisk.wait_buffer_ready %task : !frisk.async_task')
    check(wait.replace('%e = math.exp2', dma + '\n%e = math.exp2'), 0)

    # Addresses depending on exp2 or a later load cannot be speculated. A valid
    # prefix is still moved when the next packet has an unavailable address.
    source = fixture().replace('%offset0 =', '%dep = arith.fptosi %e : f32 to i64\n'
                              '%depidx = arith.index_cast %dep : i64 to index\n%offset0 =')
    check(source.replace('arith.addi %idx, %offset0', 'arith.addi %depidx, %offset0'), 0)
    check(source.replace('arith.addi %idx, %offset1', 'arith.addi %depidx, %offset1'), 1)
    source = fixture(kind='memref').replace('%offset1 =',
        '%dep = arith.fptosi %v0 : f16 to i64\n'
        '%depidx = arith.index_cast %dep : i64 to index\n%offset1 =')
    check(source.replace('arith.addi %idx, %offset1', 'arith.addi %depidx, %offset1'), 1)

    # A guarded load stays inside its guard, even when exp2 is outside it.
    guarded = '''func.func @guarded(%p: memref<256xf32, 1>, %x: f32, %i: index,
                                  %cond: i1) -> (f32, f32) {
      %e = math.exp2 %x : f32
      %v = scf.if %cond -> f32 {
        %r = memref.load %p[%i] : memref<256xf32, 1>
        scf.yield %r : f32
      } else { scf.yield %x : f32 }
      return %e, %v : f32, f32
    }'''
    result = check(guarded, 0)
    assert result.index('scf.if') < result.index('memref.load')
    nested = guarded.replace('%e = math.exp2 %x : f32', '').replace(
        '%r = memref.load', '%e = math.exp2 %x : f32\n%r = memref.load').replace(
        'scf.yield %r : f32', '%sum = arith.addf %r, %e : f32\nscf.yield %sum : f32').replace(
        'return %e, %v', 'return %x, %v')
    result = check(nested, 1)
    assert result.index('scf.if') < result.index('memref.load') < result.index('math.exp2')

    cfg = guarded.replace('%v = scf.if %cond -> f32 {', 'cf.br ^next\n^next:\n%v = scf.if %cond -> f32 {')
    check(cfg, 0)
    loop = '''func.func @loop(%p: memref<256xf32, 1>, %x: f32) -> f32 {
      %e = math.exp2 %x : f32
      affine.for %i = 0 to 256 {
        %v = affine.load %p[%i] : memref<256xf32, 1>
        %sum = arith.addf %v, %e : f32
        affine.store %sum, %p[%i] : memref<256xf32, 1>
      }
      return %e : f32
    }'''
    check(loop, 0, check_graph=False)

    # Verify the moved loads, exp2 and one-shot marker survive the real target
    # lowering/export; the compatibility LLVM pass must not consume more budget.
    scheduled = check(fixture().replace('module {',
        'module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {'), 4)
    lowered = run(scheduled, '--test-async-to-llvm=native-exp2=true')
    exported = command([args.llvm_bin / 'mlir-translate', '--mlir-to-llvmir'], lowered)
    command([args.llvm_bin / 'opt', '-passes=verify', '-disable-output'], exported)
    assert '"frisk.global_prefetch"="8"' in exported, exported
    exp_call = r'call float @llvm\.(?:amdgcn\.)?exp2\.f32'
    assert re.search(exp_call, exported), exported
    prefix = re.split(exp_call, exported)[0]
    assert len(re.findall(r' = load <4 x half>', prefix)) == 4, exported
    legacy = command([args.legacy_tool, '--prefetch-global-packets', '-'], exported)
    assert len(re.findall(r' = load <4 x half>', re.split(exp_call, legacy)[0])) == 4
    unmarked = exported.replace('"frisk.global_prefetch"="8"', '"test.prefetch_removed"="8"')
    legacy = command([args.legacy_tool, '--prefetch-global-packets', '-'], unmarked)
    assert len(re.findall(r' = load <4 x half>', re.split(exp_call, legacy)[0])) == 5
    print(f'PASS {cases} ThreadIR global packet prefetch cases, LLVM export and legacy replay')

    if args.driver:
        with tempfile.TemporaryDirectory(prefix='thread-packet-driver-') as directory:
            work = Path(directory)
            for name, options in [('enabled', ['--prefetch-packets=1']),
                                  ('disabled', ['--prefetch-packets=0']),
                                  ('cpsat', ['--prefetch-packets=1',
                                             f'--cpsat-baseline={work / "baseline.json"}'])]:
                output = work / (name + '.ll')
                log = command([args.driver.resolve(), ROOT / 'test/test_input.mlir', '0', output,
                               '--math=approx', '--llvm-opt=0', '--dump-llvm-passes=1',
                               '--dma-export=split32', *options], '', cwd=work,
                              env={**os.environ, 'FRISK_GM_TO_SHM_MODE': 'legacy'})
                marker = '---- after thread prefetchGlobalPackets -----'
                assert (marker in log) == (name == 'enabled'), name
                assert 'after llvm prefetchGlobalPackets' not in log, name
                if name == 'enabled':
                    assert log.index(marker) < log.index('---- secondLowering done -----')
                    assert '"frisk.global_prefetch"="8"' in output.read_text()
                else:
                    assert '"frisk.global_prefetch"' not in output.read_text()
                command([args.llvm_bin / 'llvm-as', output, '-o', work / (name + '.bc')], '')
                print(f'PASS MyTest {name}: ThreadIR stage and LLVM verification')


if __name__ == '__main__':
    main()
