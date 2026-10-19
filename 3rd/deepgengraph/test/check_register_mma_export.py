#!/usr/bin/env python3
"""Precise register-MMA effects, conservative recognition and O3 load reuse."""
import re
import subprocess
import tempfile
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent/'build/test/LegacyLLVMExportTest'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def run(args, source=None):
    p = subprocess.run(list(map(str, args)), input=source, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-6000:]
    return p.stdout


def main():
    padding = r's_nop 7\0A\09' * 8
    asm = padding + r'v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09' + padding
    mma = r'v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09'
    tuned = r's_nop 1\0A\09' + mma + r's_nop 1\0A\09'
    variants = {
        'bare': mma,
        'exact': asm,
        'tuned': tuned,
        'asymmetric': r's_nop 0\0A\09' * 2 + mma + r's_nop 15\0A\09',
    }
    def fixture(name, assembly=asm, constraints='=v,v,v,0', half='half'):
        return f'''define <4 x float> @{name}(ptr addrspace(3) %p, <4 x {half}> %a, <4 x {half}> %b, <4 x float> %c) {{
  %x = load float, ptr addrspace(3) %p
  %r = call <4 x float> asm sideeffect "{assembly}", "{constraints}"(<4 x {half}> %a, <4 x {half}> %b, <4 x float> %c)
  %y = load float, ptr addrspace(3) %p
  %sum = fadd float %x, %y
  %out = insertelement <4 x float> %r, float %sum, i32 0
  ret <4 x float> %out
}}
'''
    source = '\n'.join([
        *(fixture(name, assembly) for name, assembly in variants.items()),
        fixture('already_pure', assembly=tuned).replace('asm sideeffect', 'asm'),
        fixture('unused', assembly=tuned).replace('ret <4 x float> %out', 'ret <4 x float> %c'),
        fixture('extra', assembly=asm+'s_waitcnt lgkmcnt(0)'),
        fixture('clobber', constraints='=v,v,v,0,~{memory}'),
        fixture('wrong_type', half='float'),
        fixture('one_sided_padding', assembly=mma + r's_nop 1\0A\09'),
        fixture('duplicate_mma', assembly=tuned + mma + r's_nop 1\0A\09'),
        fixture('bad_delay', assembly=tuned.replace('s_nop 1', 's_nop 16')),
        fixture('missing_delay', assembly=tuned.replace('s_nop 1', 's_nop')),
        fixture('extra_memory', assembly=tuned + 'ds_read_b32 v0, v1'),
        fixture('semicolon', assembly=tuned.replace('s_nop 1', 's_nop 1; ds_read_b32 v0, v1')),
        fixture('aligned', assembly=tuned).replace('asm sideeffect', 'asm sideeffect alignstack'),
        fixture('unwinding', assembly=tuned).replace('asm sideeffect', 'asm sideeffect unwind'),
    ])
    source += f'''define <4 x float> @chain(<4 x half> %a, <4 x half> %b, <4 x float> %c) {{
  %x = call <4 x float> asm sideeffect "{tuned}", "=v,v,v,0"(<4 x half> %a, <4 x half> %b, <4 x float> %c)
  %y = call <4 x float> asm sideeffect "{tuned}", "=v,v,v,0"(<4 x half> %a, <4 x half> %b, <4 x float> %x)
  ret <4 x float> %y
}}
'''
    recognized = set(variants) | {'already_pure', 'unused', 'chain'}
    patched = run([TOOL, '-'], source)
    groups = dict(re.findall(r'attributes #(\d+) = \{([^}]*)\}', patched))
    functions = re.split(r'(?=define )', patched)
    for function in functions:
        match = re.search(r'@([a-z_]+)\(', function)
        if not match:
            continue
        name = match[1]
        call = next(x for x in function.splitlines() if ' asm ' in x)
        attr = re.search(r'\) #(\d+)', call)
        effects = groups[attr[1]] if attr else ''
        for effect in ('memory(none)', 'convergent', 'willreturn'):
            assert (effect in effects) == (name in recognized), (name, effects)
        assert ('asm sideeffect' not in call) == (name in recognized), (name, call)
        if name in recognized:
            assert 'nounwind' in effects and 'speculatable' not in effects
    template = r'asm(?: sideeffect)?(?: alignstack)?(?: unwind)? "([^"]*)"'
    assert re.findall(template, patched) == re.findall(
        template, source), 'all instruction text must remain intact'
    assert run([TOOL, '-'], patched) == patched, 'effect restoration must be idempotent'
    print('PASS pure register-only MMAC: convergent/nounwind/willreturn/memory(none); '
          'idempotence, unknown asm and padding unchanged')
    with tempfile.TemporaryDirectory(prefix='register-mma-') as tmp:
        work = Path(tmp)
        for name, text, loads in [('baseline', fixture('exact'), 2), ('fixed', patched, 1)]:
            src = work/(name+'.ll')
            src.write_text(text)
            optimized = run([LLVM/'opt', '-passes=default<O3>', '-S', src, '-o', '-'])
            exact = optimized.split('@exact(', 1)[1].split('\n}', 1)[0]
            assert exact.count('load float') == loads, exact
            print('PASS O3', name, 'loads', loads)
            if name == 'fixed':
                for variant in variants:
                    body = optimized.split('@' + variant + '(', 1)[1].split('\n}', 1)[0]
                    assert body.count('load float') == 1, variant
                print('PASS O3 load reuse for tuned and asymmetric padding')
                dead = optimized.split('@unused(', 1)[1].split('\n}', 1)[0]
                assert ' asm ' not in dead, 'unused pure MMAC must be removable'
                chain = optimized.split('@chain(', 1)[1].split('\n}', 1)[0]
                calls = [line for line in chain.splitlines() if ' asm ' in line]
                assert len(calls) == 2 and 'sideeffect' not in chain, chain
                first = calls[0].strip().split(' = ')[0]
                assert f'<4 x float> {first})' in calls[1], chain
                print('PASS O3 unused-MMAC elimination and dependent accumulator chain')
        # This is the spelling sent to the DTK/LLVM 15 parser, not deletion of
        # the memory effect. New LLVM upgrades readnone back to memory(none).
        legacy = patched.replace('memory(none)', 'readnone')
        src = work/'legacy.ll'
        src.write_text(legacy)
        roundtrip = run([LLVM/'opt', '-S', src, '-o', '-'])
        assert 'memory(none)' in roundtrip
        import ctypes.util
        library = ctypes.util.find_library('LLVM-15')
        if library:
            from check_legacy_llvm_export import load_parser
            failed, error = load_parser(library)(legacy)
            assert not failed, error
            print('PASS LLVM 15 parser retains readnone/convergent')
        print('PASS legacy readnone roundtrip')


if __name__ == '__main__':
    main()
