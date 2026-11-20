#!/usr/bin/env python3
"""Check export contracts and generic cleanup independently, without a GPU."""
from pathlib import Path
import re
import subprocess
import sys


def main():
    tool = str(Path(sys.argv[1]).resolve())

    def run(option, source):
        result = subprocess.run([tool, option, '-'], input=source, text=True,
                                capture_output=True, timeout=120)
        assert result.returncode == 0, result.stderr
        return result.stdout

    source = r'''
define <4 x float> @mma(<4 x half> %a, <4 x half> %b, <4 x float> %c) {
  %r = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %a, <4 x half> %b, <4 x float> %c)
  ret <4 x float> %r
}
define <4 x float> @anchor(<4 x float> %c) {
  %r = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %c)
  ret <4 x float> %r
}
define i32 @fold(i32 %x) {
entry:
  %sum = add i32 %x, 0
  br i1 true, label %yes, label %no
yes:
  ret i32 %sum
no:
  ret i32 0
}
'''
    def body(text, name):
        return text.split('@' + name + '(', 1)[1].split('\n}', 1)[0]

    prepared = run('--export-contracts-only', source)
    assert 'asm sideeffect' not in body(prepared, 'mma')
    assert 'asm sideeffect "", "=v,0"' in body(prepared, 'anchor')
    assert '!frisk.register_lifetimes' in body(prepared, 'anchor')
    groups = dict(re.findall(r'attributes #(\d+) = \{([^}]*)\}', prepared))
    for name in ('mma', 'anchor'):
        group = re.search(r'\) #(\d+)', body(prepared, name))[1]
        for attr in ('memory(none)', 'nounwind', 'willreturn', 'convergent'):
            assert attr in groups[group], (name, groups[group])
    assert 'add i32 %x, 0' in prepared and 'br i1 true' in prepared
    # LLVM's reader preserves metadata slot numbers; rebuilding the empty
    # marker can choose a different slot without changing its contents.
    def normalize(text):
        return re.sub(r'!\d+', '!N', text)

    assert normalize(run('--export-contracts-only', prepared)) == normalize(prepared)

    cleaned = run('--cleanup-llvm-only', source)
    assert 'asm sideeffect' in body(cleaned, 'mma')
    assert '!frisk.register_lifetimes' not in cleaned
    assert 'memory(none)' not in cleaned
    assert 'add i32' not in cleaned and 'br i1 true' not in cleaned
    assert 'ret i32 %x' in cleaned

    split = run('--cleanup-llvm-only', prepared)
    legacy = run('--cleanup-register-mma', source)
    assert normalize(split) == normalize(legacy), 'compatibility entry must match explicit composition'
    print('PASS: contract-only preserves instructions/CFG; cleanup-only does not '
          'infer MMA contracts; lifetime anchor retained; legacy composition matches')


if __name__ == '__main__':
    main()
