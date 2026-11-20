#!/usr/bin/env python3
"""Dynamic prefix preservation through canonicalization and post-unroll folds."""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'build/test'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def run(args, source=None):
    result = subprocess.run(list(map(str, args)), input=source, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr[-6000:]
    return result.stdout


def fixture(name, shape, rows, chain=False):
    typ = 'vector<' + 'x'.join(map(str, shape)) + 'xi32>'
    body = []
    for col, row in enumerate(rows):
        prefix = [row] if len(shape) == 2 else ['1', row]
        body.append(f'%e{col} = vector.extract %v[{", ".join(prefix + [str(col)])}] '
                    f': i32 from {typ}')
    if chain:
        body.append('%z = arith.constant dense<0> : vector<1x4xi32>')
        previous = '%z'
        for col in range(4):
            body.append(f'%i{col} = vector.insert %e{col}, {previous}[0, {col}] '
                        ': i32 into vector<1x4xi32>')
            previous = f'%i{col}'
        body.append('%result = vector.shape_cast %i3 : vector<1x4xi32> to vector<4xi32>')
    else:
        body.append('%result = vector.from_elements %e0, %e1, %e2, %e3 : vector<4xi32>')
    body.append('return %result : vector<4xi32>')
    return (f'func.func private @{name}(%v: {typ}, %row: index, %other: index) '
            '-> vector<4xi32> {\n' + '\n'.join(body) + '\n}\n')


def main():
    cases = [
        ('dynamic_row', (2, 4), ['%row'] * 4, False),
        ('different_rows', (2, 4), ['%row', '%other', '%row', '%other'], False),
        ('mixed_prefix', (2, 2, 4), ['%row'] * 4, False),
        ('insert_chain', (2, 4), ['%row'] * 4, True),
        ('static_row', (2, 4), ['1'] * 4, False),
    ]
    functions = ''.join(fixture(*case) for case in cases)
    # Use public functions until after the tested passes: symbol DCE must not
    # erase an uncalled fixture, and inlining must not hide dynamic indices.
    functions = functions.replace('func.func private', 'func.func')
    canonical = run([BUILD/'ThreadTilingTest', '--canonicalize'], functions)
    unrolled = run([BUILD/'FragmentOptTest', '--ir-deep-optimize'], functions)
    chained = run([BUILD/'FragmentOptTest', '--ir-deep-optimize'], canonical)
    print('PASS direct from_elements, mixed/different prefixes, insert chain and static row verification')

    # Execute all four row/other combinations; distinguish rows and planes so
    # accidentally equating dynamic markers or dropping a prefix changes data.
    body = ['%v2 = arith.constant dense<[[0, 1, 2, 3], [10, 11, 12, 13]]> : vector<2x4xi32>',
            '%v3 = arith.constant dense<[[[0, 1, 2, 3], [10, 11, 12, 13]], '
            '[[100, 101, 102, 103], [110, 111, 112, 113]]]> : vector<2x2x4xi32>',
            '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index',
            '%initial = arith.constant true']
    previous = '%initial'
    count = 0
    for name, shape, rows, _ in cases:
        typ = 'vector<' + 'x'.join(map(str, shape)) + 'xi32>'
        for row in range(2):
            for other in range(2):
                values = []
                for col, index in enumerate(rows):
                    r = row if index == '%row' else other if index == '%other' else int(index)
                    values.append((100 if len(shape) == 3 else 0) + 10*r + col)
                n = count
                body += [f'%r{n} = func.call @{name}(%v{len(shape)}, %c{row}, %c{other}) '
                         f': ({typ}, index, index) -> vector<4xi32>',
                         f'%expected{n} = arith.constant dense<{values}> : vector<4xi32>',
                         f'%eq{n} = arith.cmpi eq, %r{n}, %expected{n} : vector<4xi32>',
                         f'%ok{n} = vector.reduction <and>, %eq{n} : vector<4xi1> into i1',
                         f'%all{n} = arith.andi {previous}, %ok{n} : i1']
                previous = f'%all{n}'
                count += 1
    body += ['%success = arith.constant 0 : i32', '%failure = arith.constant 1 : i32',
             f'%status = arith.select {previous}, %success, %failure : i32',
             'return %status : i32']
    main_ir = 'func.func @main() -> i32 {\n' + '\n'.join(body) + '\n}\n'
    with tempfile.TemporaryDirectory(prefix='vector-canonicalization-') as directory:
        for label, text in [('source', functions), ('canonical', canonical),
                            ('post_unroll', unrolled), ('combined', chained)]:
            # Strip the printer's module wrapper so main can be added.
            if text.lstrip().startswith('module {'):
                text = text[text.index('{')+1:text.rindex('}')]
            text = text.replace('func.func @', 'func.func private @') + main_ir
            # The SDK inliner has its own default canonicalizer; disable that
            # nested optimizer and use the explicitly filtered one below.
            lowered = run([LLVM/'mlir-opt', '--inline=default-pipeline=',
                           '--canonicalize=disable-patterns=FromElementsToShapeCast',
                           '--convert-vector-to-llvm', '--convert-arith-to-llvm',
                           '--convert-func-to-llvm', '--convert-ub-to-llvm',
                           '--reconcile-unrealized-casts'], text)
            llvm_ir = run([LLVM/'mlir-translate', '--mlir-to-llvmir'], lowered)
            path = Path(directory)/f'{label}.ll'
            path.write_text(llvm_ir)
            run([LLVM/'lli', path])
    print(f'PASS CPU results: {count} cases through source/canonical/post-unroll/combined pipelines')


if __name__ == '__main__':
    main()
