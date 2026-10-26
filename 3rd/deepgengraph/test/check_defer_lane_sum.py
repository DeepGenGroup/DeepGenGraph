#!/usr/bin/env python3
"""Generic AMDGPU recurrence fixtures: legality, actual LLVM execution and O3."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
from check_attention_llvm_numerics import LLVMWorkgroupModel
from check_gemm_pv_addresses import split_args


def fixture(name='ordinary_statistics', bits=(1, 2), integer=False, aggregate=False,
            weighted=True, wave=64, threads=128, alpha='constant', attrs='', initial=None,
            escape='', divergent=False, wrong_address=False, lcssa=False):
    ty, add, mul = ('i32', 'add', 'mul') if integer else ('float', 'fadd', 'fmul')
    zero, one = ('0', '1') if integer else ('0.0', '1.0')
    initial = initial if initial is not None else ('zeroinitializer' if aggregate else zero)
    state_ty = f'[1 x <1 x {ty}>]' if aggregate else ty
    s = ['@dummy = internal addrspace(3) global [1 x i8] zeroinitializer',
         f'define amdgpu_kernel void @{name}(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) {attrs} !reqd_work_group_size !0 {{',
         '  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)',
         '  %lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)' if wave == 64 else '  %lane = add i32 %lo, 0',
         '  %tid = call i32 @llvm.amdgcn.workitem.id.x()',
         '  br label %header', 'header:',
         '  %iter = phi i32 [0, %4], [%next, %body]',
         f'  %state = phi {state_ty} [{initial}, %4], [%back, %body]',
         '  %go = icmp ult i32 %iter, 5' if not divergent else '  %go = icmp ult i32 %iter, %lane',
         '  br i1 %go, label %body, label %exit', 'body:']
    if aggregate:
        s += [f'  %pack = extractvalue {state_ty} %state, 0',
              f'  %old = extractelement <1 x {ty}> %pack, i32 0']
    else:
        # No copy instruction: the scalar PHI is used directly.
        pass
    old = '%old' if aggregate else '%state'
    if alpha == 'running_max':
        index = s.index('  %go = icmp ult i32 %iter, 5')
        s.insert(index, '  %maximum = phi float [0.0, %4], [%newmax, %body]')
    s += ['  %seed = xor i32 %tid, %iter',
          '  %x = add i32 %seed, 100000003' if integer else '  %x = uitofp i32 %seed to float']
    prev = '%x'
    for n, bit in enumerate(bits):
        s += [f'  %peer{n} = xor i32 %lane, {bit}',
              f'  %addr{n} = shl i32 %peer{n}, {1 if wrong_address else 2}']
        if not integer: s += [f'  %bits{n} = bitcast float {prev} to i32']
        s += [f'  %shuffle{n} = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr{n}, i32 {prev if integer else "%bits"+str(n)})']
        if not integer: s += [f'  %float{n} = bitcast i32 %shuffle{n} to float']
        s += [f'  %sum{n} = {add} {ty} {prev}, %{"shuffle" if integer else "float"}{n}']
        prev = f'%sum{n}'
    if alpha == 'running_max':
        value = '%x'
        for n, bit in enumerate(bits):
            s += [f'  %mb{n} = bitcast float {value} to i32',
                  f'  %ms{n} = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr{n}, i32 %mb{n})',
                  f'  %mf{n} = bitcast i32 %ms{n} to float',
                  f'  %mx{n} = call float @llvm.maxnum.f32(float {value}, float %mf{n})']
            value = f'%mx{n}'
        s += [f'  %newmax = call float @llvm.maxnum.f32(float %maximum, float {value})',
              '  %difference = fsub float %maximum, %newmax',
              '  %alpha = call float @llvm.exp2.f32(float %difference)']
        factor = '%alpha'
    elif alpha == 'group':
        mask = sum(set(bits))
        s += [f'  %group = and i32 %lane, {wave - 1 - mask}',
              '  %af = uitofp i32 %group to float', '  %alpha = fmul float %af, 0.015625']
        factor = '%alpha'
    elif alpha == 'varying':
        s += ['  %ai = and i32 %iter, 1', '  %af = uitofp i32 %ai to float', '  %alpha = fadd float %af, 0.5']
        factor = '%alpha'
    elif alpha == 'lane': factor = '%x'
    elif alpha == 'self': factor = old
    elif alpha == 'load':
        s += [f'  %alpha = load {ty}, ptr addrspace(1) %0']; factor = '%alpha'
    else: factor = '3' if integer else '0.5'
    if weighted: s += [f'  %scaled = {mul} {ty} {old}, {factor}']
    s += [f'  %update = {add} {ty} {prev}, {"%scaled" if weighted else old}']
    if aggregate:
        s += [f'  %packed = insertelement <1 x {ty}> poison, {ty} %update, i32 0',
              f'  %back = insertvalue {state_ty} poison, <1 x {ty}> %packed, 0']
    else:
        # Rename the recurrence operation rather than inserting an opaque copy.
        s[-1] = s[-1].replace('%update =', '%back =')
    if escape == 'state': s += [f'  store {ty} {old}, ptr addrspace(1) %3']
    if escape == 'sum': s += [f'  store {ty} {prev}, ptr addrspace(1) %3']
    if escape == 'update': s += [f'  store {ty} {"%update" if aggregate else "%back"}, ptr addrspace(1) %3']
    s += ['  %next = add i32 %iter, 1', '  br label %header', 'exit:']
    final = '%state'
    if lcssa:
        s += [f'  %lcssa = phi {state_ty} [%state, %header]']; final = '%lcssa'
    if aggregate:
        s += [f'  %outpack = extractvalue {state_ty} {final}, 0', f'  %outscalar = extractelement <1 x {ty}> %outpack, i32 0']; final = '%outscalar'
    s += [f'  %result = {add} {ty} {final}, {zero}',
          f'  %outptr = getelementptr {ty}, ptr addrspace(1) %3, i32 %tid',
          f'  store {ty} %result, ptr addrspace(1) %outptr', '  ret void', '}',
          'declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)', 'declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)',
          'declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)', 'declare i32 @llvm.amdgcn.workitem.id.x()',
          'declare float @llvm.maxnum.f32(float, float)', 'declare float @llvm.exp2.f32(float)',
          f'!0 = !{{i32 {threads}, i32 1, i32 1}}']
    return '\n'.join(s) + '\n'


class Model(LLVMWorkgroupModel):
    def __init__(self, text, wave, integer):
        data = np.zeros(128, dtype=np.float32)
        self.wave, self.shuffles, self.integer = wave, 0, integer
        super().__init__(text, data, data, data, threads=128)
    def operation(self, expr):
        if expr.startswith('uitofp i32 '):
            return (self.typed(expr[7:].split(' to ')[0]) % (1 << 32)).astype(np.float32)
        return super().operation(expr)
    def call(self, body):
        if '@llvm.amdgcn.mbcnt.' in body:
            lane = np.arange(self.t) % self.wave
            return np.minimum(lane, 32) if '.lo(' in body else lane
        if '@llvm.amdgcn.ds.bpermute(' in body:
            self.shuffles += 1
            args = list(map(self.typed, split_args(body.split('(', 1)[1].split(')', 1)[0])))
            return args[1][np.arange(self.t) // self.wave * self.wave + args[0] // 4]
        return super().call(body)
    def memory(self, ptr, ty, write=None):
        assert write is not None and ty in ('float', 'i32')
        # Numerical comparisons use the pre-store full-precision SSA result.
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tool', type=Path)
    ap.add_argument('--llvm', type=Path, default=Path('/data2/xsl/install/bin'))
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix='defer-lane-sum-') as directory:
        root = Path(directory)
        def run(text, mode='approx', wave=64, extra=()):
            src = root / 'input.ll'; src.write_text(text)
            command = [str(a.tool.resolve()), '--defer-lane-sum='+mode,
                       '--defer-lane-sum-wave-size='+str(wave), *extra, str(src)]
            result = subprocess.run(command, text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
            out = root / 'output.ll'; out.write_text(result.stdout)
            subprocess.run([str(a.llvm/'opt'), '-passes=verify', '-disable-output', str(out)], check=True, capture_output=True)
            return result.stdout, result.stderr
        def check(name, expected, kw=None, mode='approx', wave=64, edit=lambda s:s):
            kw = kw or {}
            effective_wave = wave or (32 if '+wavefrontsize32' in kw.get('attrs', '') else 64)
            source = edit(fixture(name=name, wave=effective_wave, **kw))
            transformed, diagnostic = run(source, mode, wave)
            changed = 'lane.sum.final' in transformed
            assert changed == expected, (name, diagnostic)
            # Formatting and intrinsic attributes do not count as mutations.
            off, _ = run(source, 'off', wave)
            if not expected: assert transformed == off, name
            else:
                if name == 'integer_flags':
                    assert ' nsw ' not in transformed and ' nuw ' not in transformed
                repeated, _ = run(transformed, mode, wave)
                assert repeated == transformed, ('not idempotent', name)
                before, after = transformed.split('\nexit:', 1)
                retained = len(kw.get('bits', (1, 2))) if kw.get('alpha') == 'running_max' else 0
                assert before.split('define ', 1)[1].count('@llvm.amdgcn.ds.bpermute(') == retained
                out = root/'output.ll'; out.write_text(transformed)
                subprocess.run([str(a.llvm/'opt'), '-passes=default<O3>,verify', '-disable-output', str(out)], check=True, capture_output=True)
                for trips in (0, 1, 2, 7, 31):
                    results = []
                    for ir in (source, transformed):
                        ir = ir.replace('%iter, 5', f'%iter, {trips}')
                        model = Model(ir, effective_wave, kw.get('integer', False)); model.run()
                        results.append((model.values['%result'], model.shuffles))
                    if kw.get('integer'): np.testing.assert_array_equal(results[0][0], results[1][0])
                    else: np.testing.assert_allclose(results[0][0], results[1][0], rtol=3e-6, atol=2e-4)
                    stages = len(kw.get('bits', (1, 2)))
                    assert results[0][1] == trips * (stages + retained)
                    assert results[1][1] == stages + trips * retained
            records.append(dict(name=name, transformed=changed, mode=mode))
        for integer in (False, True):
            for aggregate in (False, True):
                for weighted in (False, True):
                    check(f'ordinary_{integer}_{aggregate}_{weighted}', True,
                          dict(integer=integer, aggregate=aggregate, weighted=weighted), mode='legal' if integer else 'approx')
        for bits in ((1,), (16,32), (1,2,4,8,16,32), (2,8,32)):
            check('group_'+'_'.join(map(str,bits)), True, dict(bits=bits))
        check('wave32', True, wave=32, kw=dict(bits=(1,2,4,8,16)))
        check('group_uniform_scale', True, dict(alpha='group', bits=(16,32)))
        check('changing_scale', True, dict(alpha='varying'))
        check('ordinary_running_max_recurrence', True, dict(alpha='running_max'))
        check('scalar_lcssa', True, dict(lcssa=True))
        check('aggregate_lcssa_fallback', False, dict(lcssa=True, aggregate=True))
        check('oversized_aggregate', False, dict(aggregate=True),
              edit=lambda s:s.replace('[1 x <1 x float>]', '[4294967297 x <1 x float>]'))
        check('float_legal', False, mode='legal')
        check('off', False, mode='off')
        check('nonzero', False, dict(initial='1.0'))
        check('varying_lane_scale', False, dict(alpha='lane'))
        check('nonlinear_state_squared', False, dict(alpha='self'))
        check('unknown_load_scale', False, dict(alpha='load'))
        for escape in ('state','sum','update'):
            check('escaped_'+escape, False, dict(escape=escape))
        check('divergent', False, dict(divergent=True))
        check('wrong_address', False, dict(wrong_address=True))
        check('duplicate_masks', False, dict(bits=(1,1)))
        check('non_power_two_mask', False, dict(bits=(3,)))
        check('partial_wave', False, dict(threads=96))
        check('no_wave_contract', False, wave=0)
        check('strictfp', False, dict(attrs='strictfp'))
        check('optnone', False, dict(attrs='noinline optnone'))
        check('wave_contract_conflict', False, dict(attrs='"target-features"="+wavefrontsize32"'))
        check('ordinary_non_kernel', False, edit=lambda s:s.replace('define amdgpu_kernel', 'define'))
        check('explicit_exec_change', False, edit=lambda s:s.replace('  %lo =', '  call void @llvm.amdgcn.kill(i1 false)\n  %lo =') + '\ndeclare void @llvm.amdgcn.kill(i1)\n')
        check('opaque_asm', False, edit=lambda s:s.replace('  %lo =', '  call void asm sideeffect "", "~{exec}"()\n  %lo ='))
        check('unknown_call', False, edit=lambda s:s.replace('  %lo =', '  call void @opaque()\n  %lo =') + '\ndeclare void @opaque()\n')
        check('non_dedicated_exit', False, edit=lambda s:s.replace('  br label %header\nheader:', '  br i1 true, label %header, label %exit\nheader:').replace('%state, 0.0\n  %outptr', '0.0, 0.0\n  %outptr'))
        check('wave_from_features', True, wave=0, kw=dict(attrs='"target-features"="+wavefrontsize64"'))
        check('wave32_from_features', True, wave=0, kw=dict(attrs='"target-features"="+wavefrontsize32"', bits=(1,16)))
        check('disabled_wave_contract', False, kw=dict(attrs='"target-features"="-wavefrontsize64"'))
        check('integer_flags', True, dict(integer=True), mode='legal',
              edit=lambda s:s.replace('= add i32 %sum', '= add nsw nuw i32 %sum').replace('= mul i32', '= mul nsw nuw i32'))
        for option in ('--defer-lane-sum=bad', '--defer-lane-sum-wave-size=16', '--defer-lane-sum-wave-size=-1', '--defer-lane-sum-extra=1'):
            src=root/'input.ll'; src.write_text(fixture())
            result=subprocess.run([str(a.tool.resolve()),option,str(src)],capture_output=True)
            assert result.returncode != 0, option
    result = dict(fixtures=len(records), transformed=sum(r['transformed'] for r in records),
                  llvm_execution_pairs=5*sum(r['transformed'] for r in records), records=records)
    if a.output: a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__ == '__main__': main()
