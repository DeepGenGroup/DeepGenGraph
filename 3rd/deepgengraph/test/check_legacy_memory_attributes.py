#!/usr/bin/env python3
"""Compare the C++ legacy text exporter with legalizeLLVMText.py's rules."""
import ctypes.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from check_legacy_llvm_export import load_parser

HERE = Path(__file__).resolve().parent
TOOL = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else
        HERE.parent / 'build/test/LegacyLLVMExportTest')
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def run(args, text=None):
    result = subprocess.run(list(map(str, args)), input=text, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def legalize(text):
    return run([TOOL, '--legalize-text', '-'], text)


def main():
    effects = ['none', 'read', 'write', 'argmem: readwrite', 'argmem: read',
               'argmem: write', 'inaccessiblemem: readwrite',
               'inaccessiblemem: read', 'inaccessiblemem: write',
               'argmem: readwrite, inaccessiblemem: readwrite']
    # Check repeated occurrences and the same whitespace variations accepted
    # by the Python script, without relying on LLVM's canonical printer.
    variants = [e.replace(': ', ':' + space).replace(', ', ',' + space)
                for space in ('', ' ', '\t', '\n  ') for e in effects]
    source = '\n'.join(f'declare void @f{i}(ptr) memory({e})'
                       for i, e in enumerate(variants)) + '\n'
    with tempfile.TemporaryDirectory(prefix='legacy-memory-') as directory:
        path = Path(directory) / 'oracle.ll'
        path.write_text(source)
        run([sys.executable, HERE.parent / 'legalizeLLVMText.py', path])
        converted = legalize(source)
        assert converted == path.read_text(), 'C++ and Python mappings differ'
        assert 'memory(' not in converted
        assert legalize(converted) == converted, 'rewrites must be idempotent'
        library = ctypes.util.find_library('LLVM-15')
        assert library, 'LLVM 15 is required for the compatibility check'
        failed, error = load_parser(library)(converted)
        assert not failed, error
        # Reparse with modern LLVM and compare canonical memory effects. This
        # detects an accidentally stronger or weaker legacy attribute mapping.
        canonical = [run([LLVM / 'opt', '-S', '-o', '-', '-'], s)
                     for s in (source, converted)]
        attrs = [re.findall(r'attributes #\d+ = \{([^}]*)\}', s)
                 for s in canonical]
        assert attrs[0] == attrs[1], attrs
    print('PASS all 10 Python mappings, whitespace, LLVM 15 parsing and memory semantics')

    unmatched = '''declare void @unknown(ptr) memory(argmem: read, inaccessiblemem: write)
declare void @all_memory(ptr) memory(readwrite)
declare void @other_location(ptr) memory(other: read)
'''
    assert legalize(unmatched) == unmatched, 'unmatched effects must not be deleted'
    fallback = '''define ptr @gep(ptr captures(none) %p, i64 %i) {
  %n = add nuw nsw i64 %i, 1
  %a = getelementptr inbounds nuw i8, ptr %p, i64 %n
  %b = getelementptr nusw i8, ptr %a, i64 1
  %c = getelementptr nuw i8, ptr %b, i64 1
  ret ptr %c
}
'''
    expected = fallback.replace(' captures(none)', '').replace(
        'getelementptr inbounds nuw ', 'getelementptr inbounds ').replace(
        'getelementptr nusw ', 'getelementptr ').replace(
        'getelementptr nuw ', 'getelementptr ')
    assert legalize(fallback) == expected
    assert legalize(source + unmatched + fallback) == converted + unmatched + expected
    print('PASS unmatched effects preserved and existing GEP/captures fallback unchanged')


if __name__ == '__main__':
    main()
