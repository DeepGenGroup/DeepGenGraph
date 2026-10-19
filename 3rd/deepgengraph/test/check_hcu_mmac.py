#!/usr/bin/env python3
"""Check DTK HCU MMAC ABI, SSA chains, fallback and accumulator fusion.

Usage: check_hcu_mmac.py [ThreadTilingTest] [LLVM bin dir]
This verifies host IR export/optimization; DTK codegen and gfx936 execution
are separate checks. Enable with module attributes {frisk.use_hcu_mmac = true}.
"""
import re

from check_frisk_async_copy import LLVM, TOOL, command, lower


HALF = 'llvm.hcu.mmac.f32.16x16x16.f16'
VARIANTS = [
    ('v_mmac_f32_16x16x16_f16', HALF, '4xf16', '<4 x half>', 'i1 false'),
    ('v_mmac_f32_16x16x8_f32', 'llvm.hcu.mmac.16x16x8.f32',
     '2xf32', '<2 x float>', 'i1 false'),
    ('v_mmac_f32_16x16x4_f32', 'llvm.hcu.mmac.16x16x4.f32',
     '1xf32', 'float', 'i32 0'),
]


def fixture(variant=VARIANTS[0], ab=None, cd='1x4xf32', enabled='true'):
    mnemonic, _, shape, _, _ = variant
    ab = ab or '1x' + shape
    return f'''module attributes {{frisk.use_hcu_mmac = {enabled}}} {{
  func.func @mma(%a: vector<{ab}>, %b: vector<{ab}>, %c: vector<{cd}>,
                 %out: memref<4xf32, 1>) attributes {{thread_num = 64 : i32}} {{
    %z = arith.constant 0 : index
    %x = frisk.warp_mma_rr %a, %b, %c {{
      inst_name = "{mnemonic} $0, $1, $2, $3", inst_constraints = "=v,v,v,0"
    }} : vector<{ab}>, vector<{ab}>, vector<{cd}> -> vector<{cd}>
    %y = frisk.warp_mma_rr %b, %a, %x {{
      inst_name = "{mnemonic} $0, $1, $2, $3", inst_constraints = "=v,v,v,0"
    }} : vector<{ab}>, vector<{ab}>, vector<{cd}> -> vector<{cd}>
    %flat = vector.shape_cast %y : vector<{cd}> to vector<4xf32>
    vector.store %flat, %out[%z] : memref<4xf32, 1>, vector<4xf32>
    return
  }}
}}'''


def verify(source):
    mlir = lower(source, '--thread-ir-legalize', '--test-async-to-llvm')
    assert 'frisk.warp_mma_rr' not in mlir and 'unrealized_conversion_cast' not in mlir
    ir = command([LLVM / 'mlir-translate', '--mlir-to-llvmir'], mlir)
    command([LLVM / 'opt', '-passes=verify', '-disable-output'], ir)
    return ir


def check_calls(ir, variant):
    _, name, _, arg_type, control = variant
    calls = re.findall(r'(%[\w.]+) = (?:tail )?call <4 x float> @' + re.escape(name) +
                       r'\((.*?)\)', ir)
    assert len(calls) == 2, ir
    args = [call[1].split(', ') for call in calls]
    assert args[0][0].startswith(arg_type + ' '), args
    assert args[0][1].startswith(arg_type + ' '), args
    assert args[1][:2] == args[0][1::-1], 'A/B order must survive lowering'
    assert args[1][2] == '<4 x float> ' + calls[0][0], 'broken C dependency'
    assert args[0][3:] == args[1][3:] == [control, 'i1 false'], args
    assert len(re.findall(r'declare .*@' + re.escape(name) + r'\(', ir)) == 1
    assert 'convergent' in ir and 'willreturn' in ir and 'memory(none)' in ir
    assert 'asm sideeffect' not in ir and 's_nop' not in ir


def check_fusion():
    source = f'''declare <4 x float> @{HALF}(<4 x half>, <4 x half>, <4 x float>, i1, i1)
'''
    for name, seed, lit, extra in [
        ('fuse', 'zeroinitializer', 'false', False),
        ('nonzero', '%c', 'false', False),
        ('modified', 'zeroinitializer', 'true', False),
        ('multiuse', 'zeroinitializer', 'false', True),
    ]:
        source += f'''define <4 x float> @{name}(<4 x half> %a, <4 x half> %b, <4 x float> %c) {{
  %x = call <4 x float> @{HALF}(<4 x half> %a, <4 x half> %b, <4 x float> {seed}, i1 {lit}, i1 false)
  %y = call <4 x float> @{HALF}(<4 x half> %a, <4 x half> %b, <4 x float> %x, i1 {lit}, i1 false)
  %r = fadd <4 x float> %y, %c
'''
        source += ('  %s = fadd <4 x float> %r, %x\n  ret <4 x float> %s\n}\n'
                   if extra else '  ret <4 x float> %r\n}\n')
    tool = TOOL.with_name('LegacyLLVMExportTest')
    strict = command([tool, '-'], source)
    assert strict.count('fadd ') == 5
    fused = command([tool, '--fuse-mmac-accumulator', '-'], source)
    for name in ('fuse', 'nonzero', 'modified', 'multiuse'):
        body = re.search(r'define [^@]*@' + name + r'\(.*?\n\}', fused, re.S)[0]
        assert ('fadd ' not in body) == (name == 'fuse'), body
        if name == 'fuse':
            assert '<4 x float> %c, i1 false, i1 false)' in body
            assert 'zeroinitializer' not in body
        assert body.count('@' + HALF + '(') == 2
    print('PASS intrinsic accumulator fusion and nonzero/modifier/multi-use exclusions')


def main():
    for variant in VARIANTS:
        ir = verify(fixture(variant))
        check_calls(ir, variant)
        optimized = command([LLVM / 'opt', '-passes=default<O3>', '-S', '-o', '-'], ir)
        check_calls(optimized, variant)
        # Multidimensional fragments must become flat LLVM vectors, including
        # the scalar AB operands of x4, and then recover the original C shape.
        ab = '2x2xf16' if variant == VARIANTS[0] else None
        check_calls(verify(fixture(variant, ab=ab, cd='2x2xf32')), variant)
    print('PASS all 3 ABIs, flattening, scalar extraction, A/B/C ordering and O3 chains')

    for enabled in ('false', None):
        source = fixture(enabled='false')
        if enabled is None:
            source = source.replace(' attributes {frisk.use_hcu_mmac = false}', '')
        ir = verify(source)
        assert 'v_mmac_f32_16x16x16_f16' in ir and 's_nop' not in ir
        assert '@' + HALF not in ir
    for old, new in [('$0, $1, $2, $3', '$0, $2, $1, $3'),
                     ('=v,v,v,0', '=v,v,v,v'),
                     ('v_mmac_f32_16x16x16_f16', 'v_custom_mma')]:
        ir = verify(fixture().replace(old, new))
        assert 'asm sideeffect' in ir and '@' + HALF not in ir
    print('PASS default/explicit no-padding inline asm and conservative fallback')

    for source, message in [
        (fixture(ab='2xf16'), 'fragment types do not match HCU MMAC ABI'),
        (fixture().replace('frisk.use_hcu_mmac = true',
                           'frisk.use_hcu_mmac = "yes"'), 'must be a boolean'),
        (fixture().replace('inst_name =', 'bad_name ='),
         'requires string inst_name and inst_constraints'),
        (fixture().replace('  func.func @mma',
                           f'  llvm.func @{HALF}(f32) -> f32\n  func.func @mma'),
         'conflicting declaration'),
    ]:
        assert message in lower(source, '--thread-ir-legalize', ok=False)
    print('PASS malformed fragments, attributes and conflicting declarations')
    check_fusion()


if __name__ == '__main__':
    main()
