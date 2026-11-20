#!/usr/bin/env python3
"""Immediate anchor contracts, conservative validation and legacy equivalence."""
from pathlib import Path
import re
import subprocess
import sys


def main():
    binary = Path(sys.argv[1]).resolve()
    llvm = Path(sys.argv[2] if len(sys.argv) > 2 else '/data2/xsl/install/bin')

    def run(tool, source, *args, ok=True):
        p = subprocess.run([str(tool), *args], input=source, text=True,
                           capture_output=True, timeout=180)
        assert (p.returncode == 0) == ok, p.stderr
        return p.stdout if ok else p.stderr

    def export(source, ok=True):
        return run(binary / 'FriskSemanticsTest', source, '-', '--llvm', ok=ok)

    def canonical(source):
        text = run(llvm / 'opt', source, '-S', '-passes=verify', '-o', '-', '-')
        slots = {slot: str(index) for index, slot in enumerate(
            re.findall(r'^!(\d+) =', text, re.M))}
        return re.sub(r'!(\d+)\b', lambda m: '!' + slots[m[1]], text)

    def fixture(typ):
        return f'''module {{ llvm.func @anchor(%x: {typ}) -> {typ} {{
          %r = llvm.inline_asm has_side_effects convergent {{frisk.register_anchor}}
            "", "=v,0" %x : ({typ}) -> {typ}
          llvm.return %r : {typ}
        }} }}'''

    def check_anchors(text, count):
        groups = dict(re.findall(r'attributes #(\d+) = \{([^}}]*)\}', text))
        anchors = [line for line in text.splitlines()
                   if 'asm sideeffect "", "=v,0"' in line]
        assert len(anchors) == count, text
        for line in anchors:
            group = re.search(r'\) #(\d+)', line)[1]
            for attr in ('memory(none)', 'nounwind', 'willreturn', 'convergent'):
                assert attr in groups[group], (line, groups[group])
            assert 'speculatable' not in groups[group]
            assert '!frisk.register_lifetimes' in line

    for typ in ('vector<4xf32>', 'vector<4xf16>'):
        source = fixture(typ)
        direct = export(source)
        check_anchors(direct, 1)
        unmarked = export(source.replace(' {frisk.register_anchor}', ''))
        assert '!frisk.register_lifetimes' not in unmarked
        assert 'memory(none)' not in unmarked
        prepared = run(binary / 'LegacyLLVMExportTest', unmarked, '--export-contracts-only', '-')
        assert canonical(direct) == canonical(prepared)
        # The old fallback has nothing left to add to a registered export.
        assert canonical(direct) == canonical(run(binary / 'LegacyLLVMExportTest',
                                                  direct, '--export-contracts-only', '-'))
        optimized = run(llvm / 'opt', direct, '-S', '-passes=default<O3>,verify', '-o', '-', '-')
        check_anchors(optimized, 1)

    source = fixture('vector<4xf32>')
    for invalid in (
            source.replace('"", "=v,0"', '"s_barrier", "=v,0"'),
            source.replace('"=v,0"', '"=v,0,~{memory}"'),
            source.replace('"=v,0"', '"=v,v"'),
            source.replace('has_side_effects ', ''),
            source.replace('has_side_effects convergent', 'has_side_effects is_align_stack convergent'),
            source.replace('vector<4xf32>', 'vector<8xf32>'),
            source.replace('vector<4xf32>', 'f32')):
        error = export(invalid, ok=False)
        assert 'frisk.register_anchor' in error and 'contract' in error, error
    bad_marker = source.replace('{frisk.register_anchor}', '{frisk.register_anchor = "bad"}')
    assert 'unit marker' in export(bad_marker, ok=False)

    # Real convert_pack producer: both boundaries receive the marker and the
    # plain LLVM export already equals the old export + fallback composition.
    pack = '''module { func.func @pack(%x: vector<4xf32>) -> vector<4xf16> {
      %r = frisk.convert_pack %x {rounding = "rtz"} : vector<4xf32> -> vector<4xf16>
      return %r : vector<4xf16>
    }}'''
    lowered = run(binary / 'ThreadTilingTest', pack, '--test-async-to-llvm', '--verify-each')
    assert lowered.count('frisk.register_anchor') == 2, lowered
    assert lowered.count('llvm.inline_asm has_side_effects') == 2, lowered
    direct = export(lowered)
    check_anchors(direct, 2)
    old = export(lowered.replace('frisk.register_anchor', 'test.unmarked_anchor'))
    prepared = run(binary / 'LegacyLLVMExportTest', old, '--export-contracts-only', '-')
    assert canonical(direct) == canonical(prepared)
    assert direct.count('asm "v_cvt_pkrtz_f16_f32') == 2
    print('PASS: f32/f16 anchor contracts at translation, preserved sideeffect, '
          'O3 survival, invalid marker rejection, real producer and legacy equivalence')


if __name__ == '__main__':
    main()
