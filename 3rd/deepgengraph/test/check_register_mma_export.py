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
        fixture('extra', assembly=asm+'s_waitcnt lgkmcnt(0)'),
        fixture('clobber', constraints='=v,v,v,0,~{memory}'),
        fixture('wrong_type', half='float'),
        fixture('missing_padding', assembly='v_mmac_f32_16x16x16_f16 $0, $1, $2, $3'),
        fixture('duplicate_mma', assembly=tuned + mma + r's_nop 1\0A\09'),
        fixture('bad_delay', assembly=tuned.replace('s_nop 1', 's_nop 16')),
        fixture('missing_delay', assembly=tuned.replace('s_nop 1', 's_nop')),
        fixture('extra_memory', assembly=tuned + 'ds_read_b32 v0, v1'),
        fixture('semicolon', assembly=tuned.replace('s_nop 1', 's_nop 1; ds_read_b32 v0, v1')),
    ])
    patched = run([TOOL, '-'], source)
    groups = dict(re.findall(r'attributes #(\d+) = \{([^}]*)\}', patched))
    functions = re.split(r'(?=define )', patched)
    for function in functions:
        match = re.search(r'@([a-z_]+)\(', function)
        if not match:
            continue
        name = match[1]
        call = next(x for x in function.splitlines() if 'asm sideeffect ' in x)
        attr = re.search(r'\) #(\d+)', call)
        effects = groups[attr[1]] if attr else ''
        assert ('memory(none)' in effects) == (name in variants), (name, effects)
        assert ('convergent' in effects) == (name in variants), (name, effects)
    assert re.findall(r'asm sideeffect "([^"]*)"', patched) == re.findall(
        r'asm sideeffect "([^"]*)"', source), 'all instruction text must remain intact'
    assert run([TOOL, '-'], patched) == patched, 'effect restoration must be idempotent'
    print('PASS exact register-only recognition; unknown asm and padding unchanged')
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
