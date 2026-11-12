#!/usr/bin/env python3
"""ThreadIR mask bypass: vector view DAGs, multiple users, FP/SSA guards and CPU execution."""
import ctypes
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

TOOL = Path(sys.argv[1]).resolve()
LLVM = Path(os.environ.get('LLVM_BIN', '/data2/xsl/install/bin'))
V = 'vector<4xf32>'


def fixture(name, source='branch', view='direct', left=False, clear='0.0',
            attrs='', extra='', escape=False, late=False, invert=False):
    declarations = f'%zero = arith.constant dense<{clear}> : {V}\n'
    if view == 'multi':
        declarations += f'%otherScore = arith.mulf %score, %score : {V}\n'
    if source == 'branch':
        yes, no = ('%mask', '%zero') if invert else ('%zero', '%mask')
        root = f'%m = scf.if %condition -> ({V}) {{ scf.yield {yes} : {V} }} else {{ scf.yield {no} : {V} }}'
    else:
        condition = '%condition'
        if source == 'vector':
            declarations += '%predicate = vector.broadcast %condition : i1 to vector<4xi1>\n'
            declarations += '%alternating = arith.constant dense<[true, false, true, false]> : vector<4xi1>\n'
            declarations += '%conditions = arith.xori %predicate, %alternating : vector<4xi1>\n'
            condition = '%conditions'
        yes, no = ('%mask', '%zero') if invert else ('%zero', '%mask')
        types = f'vector<4xi1>, {V}' if source == 'vector' else V
        root = f'%m = arith.select {condition}, {yes}, {no} : {types}'
    value, score, result_type = '%m', '%score', V
    views = ''
    if view == 'shuffle':
        views = f'%view = vector.shuffle %m, %zero [2, 4, 1, 0] : {V}, {V}\n'
        value = '%view'
    elif view == 'extract':
        views = f'%shaped = vector.shape_cast %m : {V} to vector<2x2xf32>\n'
        views += '%view = vector.extract %shaped[1] : vector<2xf32> from vector<2x2xf32>\n'
        declarations += '%smallscore = vector.broadcast %scalar : f32 to vector<2xf32>\n'
        value, score, result_type = '%view', '%smallscore', 'vector<2xf32>'
    elif view == 'transpose':
        views = f'%shaped = vector.shape_cast %m : {V} to vector<2x2xf32>\n'
        views += '%transposed = vector.transpose %shaped, [1, 0] : vector<2x2xf32> to vector<2x2xf32>\n'
        views += '%view = vector.shape_cast %transposed : vector<2x2xf32> to vector<4xf32>\n'
        value = '%view'
    elif view == 'pack':
        for lane in range(4):
            views += f'%e{lane} = vector.extract %m[{lane}] : f32 from {V}\n'
        views += f'%view = vector.from_elements %e2, %e0, %e3, %e1 : {V}\n'
        value = '%view'
    elif view == 'slice':
        views = f'%view = vector.extract_strided_slice %m offsets = [1], sizes = [2], strides = [1] : {V} to vector<2xf32>\n'
        declarations += '%smallscore = vector.broadcast %scalar : f32 to vector<2xf32>\n'
        value, score, result_type = '%view', '%smallscore', 'vector<2xf32>'
    elif view == 'dynamic':
        views = f'%view = vector.extract %m[%index] : f32 from {V}\n'
        value, score, result_type = '%view', '%scalar', 'f32'
    elif view == 'poison':
        declarations += f'%poison = ub.poison : {V}\n'
        views = f'%view = vector.shuffle %m, %poison [0, 4, 1, 5] : {V}, {V}\n'
        value = '%view'
    if late:
        extra += f'%late = arith.mulf {score}, {score} : {result_type}\n'
        score = '%late'
    operands = (value, score) if left else (score, value)
    add = f'%sum = arith.addf {operands[0]}, {operands[1]} : {result_type}\n'
    if escape:
        # Return the mask as well: the entire candidate must remain untouched.
        ret = f'return %sum, %m : {result_type}, {V}'
        ret_type = f'({result_type}, {V})'
    elif view == 'multi':
        add += f'%other = arith.addf %m, %otherScore : {V}\n'
        ret = f'return %sum, %other : {V}, {V}'
        ret_type = f'({V}, {V})'
    else:
        ret, ret_type = f'return %sum : {result_type}', result_type
    return f'''func.func @{name}(%score: {V}, %mask: {V}, %scalar: f32,
          %condition: i1, %index: index) -> {ret_type} attributes {{{attrs}}} {{
        {declarations}
        {root}
        {views}
        {extra}
        {add}
        {ret}
    }}'''


def execute_cpu(tmp, run):
    # Run compiled output for per-lane select masks and structured masks, with
    # both zero arms and both addition orders. Expose scalar/raw-bit C ABIs.
    variants = [(source, view, lhs, invert) for source in ('branch', 'scalar', 'vector')
                for view in ('direct', 'shuffle', 'extract', 'transpose', 'slice', 'pack')
                for lhs in (False, True) for invert in (False, True)]
    functions = []
    for i, (source, view, lhs, invert) in enumerate(variants):
        text = fixture(f'case{i}', source, view, lhs, invert=invert)
        text = re.sub(r'func.func @case\d+\(.*?attributes \{\} \{',
            f'''func.func @case{i}(%bits: i32, %maskbits: i32, %condition: i1, %index: index) -> i32 {{
              %scalar = arith.bitcast %bits : i32 to f32
              %masked = arith.bitcast %maskbits : i32 to f32
              %score = vector.broadcast %scalar : f32 to {V}
              %mask = vector.broadcast %masked : f32 to {V}''', text, count=1, flags=re.S)
        result_type = 'vector<2xf32>' if view in ('extract', 'slice') else V
        text = text.replace(f'return %sum : {result_type}',
            f'%selected = vector.extract %sum[%index] : f32 from {result_type}\n'
            '%out = arith.bitcast %selected : f32 to i32\nreturn %out : i32')
        functions.append(text)
    module = 'module {\n' + '\n'.join(functions) + '\n}'
    objects = []
    for mode in ('off', 'approx'):
        source = re.sub(r'@case(\d+)', rf'@case\1_{mode}', module)
        native, log = run(source, ['--frisk-bypass-mask-adds=mode=' + mode])
        if mode == 'approx':
            assert native.count('frisk.mask_add_bypass') == len(variants), log
        path, lowered, ll, obj = [tmp / (mode + ext) for ext in ('.mlir', '.llvm.mlir', '.ll', '.o')]
        path.write_text(native)
        def command(args):
            p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=120)
            assert p.returncode == 0, p.stderr[-6000:]
        command([LLVM/'mlir-opt', path, '--lower-affine', '--convert-scf-to-cf',
                 '--convert-vector-to-llvm', '--convert-arith-to-llvm', '--convert-func-to-llvm',
                 '--convert-cf-to-llvm', '--convert-ub-to-llvm', '--reconcile-unrealized-casts', '-o', lowered])
        command([LLVM/'mlir-translate', lowered, '--mlir-to-llvmir', '-o', ll])
        command([LLVM/'opt', '-passes=default<O3>,verify', ll, '-disable-output'])
        command([LLVM/'llc', '-filetype=obj', '-relocation-model=pic', ll, '-o', obj])
        objects.append(obj)
    library = tmp/'masks.so'
    command(['cc', '-shared', *objects, '-o', library])
    lib = ctypes.CDLL(str(library))
    edge = [0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000,
            0x7fc12345, 0x7fa12345, 0x3f800000, 0xbf800000, 0x7f7fffff]
    nan = lambda v: (v & 0x7f800000) == 0x7f800000 and bool(v & 0x7fffff)
    comparisons = 0
    for i, (_, view, _, _) in enumerate(variants):
        before, after = [getattr(lib, f'case{i}_{mode}') for mode in ('off', 'approx')]
        for fn in (before, after):
            fn.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint64]
            fn.restype = ctypes.c_uint32
        for score in edge:
            for mask in edge:
                for condition in (False, True):
                    for lane in range(2 if view in ('extract', 'slice') else 4):
                        a, b = [fn(score, mask, condition, lane) for fn in (before, after)]
                        assert a == b or (nan(a) and nan(b)) or (
                            (a & 0x7fffffff) == 0 and (b & 0x7fffffff) == 0), (
                            variants[i], hex(score), hex(mask), condition, lane, hex(a), hex(b))
                        comparisons += 1
    print('PASS', comparisons, 'native CPU comparisons of scalar/vector selects, branches and lane views', flush=True)


def main():
    cases = {}
    for source in ('branch', 'scalar', 'vector'):
        for view in ('direct', 'shuffle', 'extract', 'transpose', 'slice', 'pack', 'multi'):
            for invert in (False, True):
                name = f'{source}_{view}_{int(invert)}'
                cases[name] = (fixture(name, source, view, invert=invert), True)
    # Static score views after the mask may be duplicated; score arithmetic
    # or loads may not. Preserve original branch effects when adding yields.
    score_view = fixture('score_view', view='extract')
    score_view = score_view.replace('%smallscore = vector.broadcast %scalar : f32 to vector<2xf32>\n', '')
    score_view = score_view.replace('%sum =',
        '%smallscore = vector.extract_strided_slice %score offsets = [0], sizes = [2], strides = [1] : vector<4xf32> to vector<2xf32>\n%sum =')
    cases['score_view'] = (score_view, True)
    effects = fixture('branch_effects').replace('scf.yield %zero',
        'func.call @touch() : () -> ()\nscf.yield %zero').replace('scf.yield %mask',
        'func.call @touch() : () -> ()\nscf.yield %mask')
    cases['branch_effects'] = ('func.func private @touch()\n' + effects, True)
    for element in ('f16', 'f64'):
        cases[element] = (fixture(element).replace('f32', element), True)
    for name, options in {
        'escape': dict(escape=True), 'late_score': dict(late=True),
        'barrier': dict(extra='gpu.barrier\n'), 'dynamic': dict(view='dynamic'),
        'poison': dict(view='poison'), 'nonzero': dict(clear='1.0'),
        'negative_zero': dict(clear='-0.0'), 'strict': dict(attrs='strictfp'),
        'llvm_strict': dict(attrs='llvm.strictfp'), 'optnone': dict(attrs='optnone'),
        'llvm_optnone': dict(attrs='llvm.optnone'),
        'passthrough': dict(attrs='passthrough = ["strictfp"]'),
        'denormals': dict(attrs='passthrough = [["denormal-fp-math", "preserve-sign"]]'),
    }.items():
        cases[name] = (fixture(name, **options), False)
    with tempfile.TemporaryDirectory(prefix='thread-mask-add-bypass-') as directory:
        tmp = Path(directory)
        src, out = tmp/'input.mlir', tmp/'out.mlir'
        def run(text, passes, ok=True):
            src.write_text(text)
            p = subprocess.run([str(TOOL), str(src), *passes, '--verify-each', '-o', str(out)],
                               capture_output=True, text=True, timeout=120)
            assert (p.returncode == 0) == ok, p.stderr[-8000:]
            return (out.read_text() if ok else ''), p.stderr
        for name, (source, expected) in cases.items():
            original, _ = run('module {\n' + source + '\n}', [])
            disabled, _ = run(original, ['--frisk-bypass-mask-adds=mode=off'])
            assert disabled == original, name
            result, log = run(original, ['--frisk-bypass-mask-adds'])
            assert ('frisk.mask_add_bypass' in result) == expected, (name, log, result)
            if not expected: assert result == original, (name, 'fallback changed IR')
            else:
                assert 'fastmath' not in result, name
                assert ('scf.if' in result) == (name.startswith('branch_') or name in ('score_view', 'f16', 'f64')), name
                if name == 'branch_effects':
                    assert result.count('call @touch') == 2
                if '_multi_' in name:
                    assert '2 additions bypassed' in log, log
            parsed, _ = run(result, [])
            repeated, _ = run(result, ['--frisk-bypass-mask-adds'])
            assert parsed == repeated, (name, 'not idempotent')
            print('PASS', name, flush=True)
        for mode in ('legal', 'invalid'):
            _, error = run('module {func.func @f() {return}}', ['--frisk-bypass-mask-adds=mode=' + mode], False)
            assert 'legal mode has been removed' in error
        execute_cpu(tmp, run)


if __name__ == '__main__':
    main()
