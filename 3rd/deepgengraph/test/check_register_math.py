#!/usr/bin/env python3
"""Approximate-mode legality and scalar reciprocal CPU numerics.

MMAC and native exp2 accuracy still require DTK hardware. The strict exporter
must retain the original operations; opaque asm and multi-use chains are not
eligible for accumulator reassociation.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(args, text=None):
    p = subprocess.run(list(map(str, args)), input=text, text=True,
                       capture_output=True, timeout=180)
    assert p.returncode == 0, p.stderr
    return p.stdout


def main():
    tool = Path(sys.argv[1]).resolve()
    llvm = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')
    asm = ('asm sideeffect "s_nop 1\\0A\\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3'
           '\\0A\\09s_nop 1\\0A\\09", "=v,v,v,0"')
    source = 'declare float @__ocml_exp2_f32(float)\n'
    for name, seed, extra in [('fuse', 'zeroinitializer', False),
                              ('nonzero', '%acc', False),
                              ('multiuse', 'zeroinitializer', True)]:
        source += f'''define <4 x float> @{name}(<4 x half> %a, <4 x half> %b, <4 x float> %acc) {{
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> {seed})
  %y = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> %x)
  %r = fadd <4 x float> %y, %acc
'''
        source += ('  %extra = fadd <4 x float> %r, %x\n  ret <4 x float> %extra\n}\n'
                   if extra else '  ret <4 x float> %r\n}\n')
    source += '''define float @normalize(float %x, float %y, float %d) {
  %e = call float @__ocml_exp2_f32(float %x)
  %a = fdiv float %e, %d
  %b = fdiv float %y, %d
  %r = fadd float %a, %b
  ret float %r
}
'''
    source += f'''define <4 x float> @late_extract(<4 x half> %a, <4 x half> %b, [2 x <4 x float>] %carrier) {{
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
  %seed = extractvalue [2 x <4 x float>] %carrier, 1
  %result = fadd <4 x float> %x, %seed
  ret <4 x float> %result
}}
'''
    source += f'''define <4 x float> @late_load(<4 x half> %a, <4 x half> %b, ptr %p) {{
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
  %seed = load volatile <4 x float>, ptr %p
  %result = fadd <4 x float> %seed, %x
  ret <4 x float> %result
}}
define <4 x float> @cross_block(<4 x half> %a, <4 x half> %b, <4 x float> %acc, i1 %cond) {{
entry:
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
  br i1 %cond, label %yes, label %no
yes:
  %result = fadd <4 x float> %x, %acc
  ret <4 x float> %result
no:
  ret <4 x float> %acc
}}
define <4 x float> @guarded_store(<4 x half> %a, <4 x half> %b, <4 x float> %acc, i1 %cond, ptr %out) {{
entry:
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
  br i1 %cond, label %yes, label %no
yes:
  %result = fadd <4 x float> %x, %acc
  store volatile <4 x float> %result, ptr %out
  ret <4 x float> %result
no:
  ret <4 x float> %acc
}}
define <4 x float> @reverse_add(<4 x half> %a, <4 x half> %b, <4 x float> %acc) {{
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
  %result = fadd <4 x float> %acc, %x
  ret <4 x float> %result
}}
'''
    for name, assembly, attr in [('unknown_asm', asm.replace('v_mmac_f32_16x16x16_f16', 'v_mmac_unknown'), ''),
                                 ('strict_fp', asm, 'strictfp')]:
        source += f'''define <4 x float> @{name}(<4 x half> %a, <4 x half> %b, <4 x float> %acc) {attr} {{
  %x = call <4 x float> {assembly}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
  %r = fadd <4 x float> %x, %acc
  ret <4 x float> %r
}}
'''
    for name, attr, late in [('late_scale', '', False),
                             ('strict_scale', 'strictfp', False),
                             ('late_loaded_scale', '', True)]:
        seed = ('  %row = load volatile <4 x float>, ptr %p\n'
                if late else '  %row = extractvalue [2 x <4 x float>] %carrier, 1\n')
        source += f'''define <4 x float> @{name}(<4 x half> %a, <4 x half> %b, [2 x <4 x float>] %carrier, <4 x float> %alpha, ptr %p) {attr} {{
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
{seed}  %scaled = fmul <4 x float> %row, %alpha
  %result = fadd <4 x float> %x, %scaled
  ret <4 x float> %result
}}
'''
    source += f'''define <4 x float> @late_broadcast_scale(<4 x half> %a, <4 x half> %b, <4 x float> %carrier, float %alpha) {{
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
  %insert = insertelement <4 x float> poison, float %alpha, i32 0
  %broadcast = shufflevector <4 x float> %insert, <4 x float> poison, <4 x i32> zeroinitializer
  %scaled = fmul <4 x float> %carrier, %broadcast
  %result = fadd <4 x float> %x, %scaled
  ret <4 x float> %result
}}
'''
    # Layout 3.1 leaves MMACs and their terminal fadd in separate blocks.
    # Thread-bound simplification proves the LDS writeback guard true only
    # late in the export pipeline; CFG cleanup must precede the fusion retry.
    for name, branch in [('constant_guard', 'br i1 true, label %writeback, label %exit'),
                         ('unconditional_guard', 'br label %writeback')]:
        source += f'''define <4 x float> @{name}(<4 x half> %a, <4 x half> %b, <4 x float> %acc) {{
entry:
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
  %y = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> %x)
  {branch}
writeback:
  %result = fadd <4 x float> %y, %acc
  ret <4 x float> %result
exit:
  ret <4 x float> %acc
}}
'''
    strict = run([tool, '-'], source)
    native_only = run([tool, '--native-exp2', '-'], source)
    reciprocal_only = run([tool, '--reciprocal', '-'], source)
    assert native_only.count('fdiv float') == 2 and '@llvm.amdgcn.exp2' in native_only
    assert reciprocal_only.count('fdiv arcp') == 1 and '@llvm.amdgcn.exp2' not in reciprocal_only
    assert '@__ocml_exp2_f32' in reciprocal_only
    assert strict.count('fdiv float') == 2 and '@llvm.amdgcn.exp2' not in strict
    fusion = run([tool, '--fuse-mmac-accumulator', '-'], source)
    fused_body = lambda name: re.search(r'define [^@]*@' + name + r'\(.*?\n\}', fusion, re.S)[0]
    for name in ('fuse', 'late_extract', 'reverse_add', 'late_scale', 'late_broadcast_scale',
                 'constant_guard', 'unconditional_guard', 'cross_block'):
        assert 'fadd' not in fused_body(name) and '<4 x float> zeroinitializer' not in fused_body(name)
    # SimplifyCFG may safely speculate a bare fadd and replace the branch with
    # a select. A conditional side effect must still keep its guarded block.
    assert 'select i1 %cond' in fused_body('cross_block')
    assert 'br i1 %cond' in fused_body('guarded_store')
    for name in ('nonzero', 'multiuse', 'late_load', 'guarded_store', 'unknown_asm', 'strict_fp',
                 'strict_scale', 'late_loaded_scale'):
        assert 'fadd' in fused_body(name), name
    assert fused_body('late_load').index('call ') < fused_body('late_load').index('load volatile')
    assert fused_body('late_scale').index('fmul ') < fused_body('late_scale').index('call ')
    assert fused_body('late_loaded_scale').index('call ') < fused_body('late_loaded_scale').index('load volatile')
    assert fusion.count('v_mmac_') == source.count('v_mmac_')
    assert fused_body('normalize').count('fdiv float') == 2
    assert '@__ocml_exp2_f32' in fused_body('normalize')
    assert '@llvm.amdgcn.exp2' not in fusion and ' arcp ' not in fusion
    # LLVM's parser may reorder predecessor-use lists; these comments do not
    # describe an IR change when round-tripping a newly merged return block.
    without_preds = lambda text: re.sub(r'; preds =[^\n]*', '', text)
    assert without_preds(run([tool, '--fuse-mmac-accumulator', '-'], fusion)) == without_preds(fusion)
    print('PASS independent MMAC pass: constant/unconditional CFG cleanup, both add orders, no load motion, idempotence, exp2/div unchanged')
    approximate = run([tool, '--approx-register-math', '-'], source)
    body = lambda name: re.search(r'define [^@]*@' + name + r'\(.*?\n\}', approximate, re.S)[0]
    assert 'fadd' not in body('fuse')
    assert 'zeroinitializer' not in body('fuse')
    assert 'fadd' in body('nonzero') and 'zeroinitializer' in body('multiuse')
    assert approximate.count('v_mmac_') == source.count('v_mmac_')
    assert 'fadd' not in body('late_extract')
    assert body('late_extract').index('extractvalue') < body('late_extract').index('call ')
    assert body('normalize').count('fdiv arcp') == 1
    assert body('normalize').count('fmul float') == 2
    assert not re.search(r'\b(nnan|ninf|fast)\b', approximate)
    assert '@llvm.amdgcn.exp2.f32' in approximate
    print('PASS strict unchanged, single-use zero MMA fusion, multi-use/nonzero fallbacks, shared reciprocal')

    # Reproduce the ordering dependency, including the pass that proves the
    # thread guard true. Early CFG cleanup alone cannot expose this fusion.
    guarded = f'''declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @thread_guard(<4 x half> %a, <4 x half> %b, <4 x float> %acc, ptr addrspace(3) %out) !reqd_work_group_size !0 {{
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %x = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer)
  %y = call <4 x float> {asm}(<4 x half> %a, <4 x half> %b, <4 x float> %x)
  %leader = icmp ult i32 %tid, 256
  br i1 %leader, label %writeback, label %exit
writeback:
  %result = fadd <4 x float> %y, %acc
  store volatile <4 x float> %result, ptr addrspace(3) %out
  br label %exit
exit:
  ret void
}}
!0 = !{{i32 256, i32 1, i32 1}}
'''
    early = run([tool, '--approx-register-math', '-'], guarded)
    assert 'fadd' in run([tool, '--fuse-mmac-accumulator', '-'], early)
    bounded = run([tool, '--optimize-register-lifetimes', '-'], early)
    assert 'br i1 true' in bounded and 'fadd' in bounded
    final = run([tool, '--fuse-mmac-accumulator', '-'], bounded)
    assert 'fadd' not in final and 'br i1' not in final
    assert final.count('v_mmac_') == 2 and final.count('store volatile') == 1
    print('PASS late fusion after lifetime/thread-bound simplification')

    # Execute the actual transformed division code on the CPU. Use positive
    # normal row sums and signed numerators; native exp2 is tested separately
    # by codegen below and is deliberately absent from this CPU test.
    numerics = '''declare float @llvm.fabs.f32(float)
define float @divide(float %x, float %d) {
  %v = fdiv float %x, %d
  ret float %v
}
define i32 @main() {
'''
    checks = []
    for i, (x, d) in enumerate([(0.0, 1.0), (-3.0, 7.0), (1.0, 4096.0),
                                 (127.0, 513.0), (9.0, 0.25)]):
        numerics += f'''  %v{i} = call float @divide(float {x}, float {d})
  %r{i} = fdiv float {x}, {d}
  %delta{i} = fsub float %v{i}, %r{i}
  %abs{i} = call float @llvm.fabs.f32(float %delta{i})
  %ok{i} = fcmp olt float %abs{i}, 0x3EB0C6F7A0000000
'''
        checks.append(f'%ok{i}')
    result = checks[0]
    for i, check in enumerate(checks[1:]):
        numerics += f'  %all{i} = and i1 {result}, {check}\n'
        result = f'%all{i}'
    numerics += f'  %status = select i1 {result}, i32 0, i32 1\n  ret i32 %status\n}}\n'
    with tempfile.TemporaryDirectory(prefix='register-math-') as tmp:
        path = Path(tmp) / 'cpu.ll'
        # Execute the transformed seed wiring on CPU with an elementwise
        # MMAC stand-in. This checks SSA/data flow, not DCU rounding behavior.
        cpu = fused_body('fuse')
        cpu = re.sub(r'asm(?: sideeffect)? "[^"\n]*", "[^"\n]*"', '@mock_mmac', cpu)
        cpu = re.sub(r' #\d+', '', cpu)
        cpu += '''
define <4 x float> @mock_mmac(<4 x half> %a, <4 x half> %b, <4 x float> %c) {
  %af = fpext <4 x half> %a to <4 x float>
  %bf = fpext <4 x half> %b to <4 x float>
  %p = fmul <4 x float> %af, %bf
  %r = fadd <4 x float> %p, %c
  ret <4 x float> %r
}
define i32 @main() {
  %r = call <4 x float> @fuse(<4 x half> <half 1.0, half 2.0, half 3.0, half 4.0>, <4 x half> <half 2.0, half 3.0, half 4.0, half 5.0>, <4 x float> <float 10.0, float 20.0, float 30.0, float 40.0>)
  %ok = fcmp oeq <4 x float> %r, <float 14.0, float 32.0, float 54.0, float 80.0>
  %a = extractelement <4 x i1> %ok, i32 0
  %b = extractelement <4 x i1> %ok, i32 1
  %c = extractelement <4 x i1> %ok, i32 2
  %d = extractelement <4 x i1> %ok, i32 3
  %ab = and i1 %a, %b
  %cd = and i1 %c, %d
  %all = and i1 %ab, %cd
  %status = select i1 %all, i32 0, i32 1
  ret i32 %status
}
'''
        path.write_text(cpu)
        run([llvm / 'lli', path])
        print('PASS fused MMAC seed wiring CPU emulation')
        path.write_text(run([tool, '--approx-register-math', '-'], numerics))
        run([llvm / 'lli', path])
        path.write_text('define amdgpu_kernel void @exp(ptr addrspace(1) %out, float %x) {\n'
                        ' %e = call float @__ocml_exp2_f32(float %x)\n'
                        ' store float %e, ptr addrspace(1) %out\n ret void\n}\n'
                        'declare float @__ocml_exp2_f32(float)\n')
        path.write_text(run([tool, '--approx-register-math', path]))
        assembly = run([llvm / 'llc', '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx90a', path, '-o', '-'])
        assert 'v_exp_f32' in assembly and 's_swappc' not in assembly
    print('PASS reciprocal CPU numerics and native exp2 codegen (gfx90a structural check only)')


if __name__ == '__main__':
    main()
