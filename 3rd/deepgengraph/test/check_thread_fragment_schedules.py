#!/usr/bin/env python3
"""Thread paired LDS/softmax schedules: SSA DAG, boundaries, replay and lowering."""
from collections import defaultdict
import hashlib
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'build/test/ThreadTilingTest'
LLVM = Path('/data2/xsl/install/bin')
MAIN = '--frisk-thread-prefetch-paired-lds=distance='
TAIL = '--frisk-thread-prefetch-paired-lds-tail=distance='
SOFT = '--frisk-thread-schedule-softmax-fragments=mode='


def run(source, *passes, ok=True):
    p = subprocess.run([str(TOOL), '--verify-each', *passes], input=source,
                       text=True, capture_output=True, timeout=120)
    assert (p.returncode == 0) == ok, p.stderr
    return p.stdout if ok else p.stderr


def mma(name, a, b, c, tag):
    return (f'%{name} = llvm.inline_asm has_side_effects convergent {{test.id = "{tag}"}} '
            f'"v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", "=v,v,v,0" {a}, {b}, {c} : '
            '(vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>')


def paired(count=8, spacing=128, tail=False, boundary='', align=8, views=False):
    lines = ['module {', 'func.func private @unknown()',
             'func.func @kernel(%sm: memref<65536xf16, 3>, %a: vector<4xf16>, '
             '%seed: vector<4xf32>, %other: index) -> (vector<4xf32>, f32) {',
             '%i = arith.constant 0 : index', '%zero = arith.constant 0.0 : f32']
    for i in range(count):
        lines += [f'%b{i} = frisk.lds_load %sm[%i] {{alignment = {align} : i64, '
                  f'byte_offset = {i*spacing} : i64}} : memref<65536xf16, 3> -> vector<4xf16>']
        b = f'%b{i}'
        if views:
            lines += [f'%v{i}_{j} = vector.extract %b{i}[{j}] : f16 from vector<4xf16>' for j in range(4)]
            lines += [f'%p{i} = vector.from_elements ' + ', '.join(f'%v{i}_{j}' for j in range(4)) + ' : vector<4xf16>']
            b = f'%p{i}'
        lines += [mma(f'm{i}', '%a', b, '%seed' if i == 0 else f'%m{i-1}', str(i))]
        if tail:
            lines += [f'%x{i} = vector.extract %m{i}[0] : f32 from vector<4xf32>',
                      f'%r{i} = arith.maxnumf %x{i}, ' + ('%zero' if i == 0 else f'%r{i-1}') + ' : f32']
        if boundary: lines += [boundary]
    lines += [f'return %m{count-1}, ' + (f'%r{count-1}' if tail else '%zero') + ' : vector<4xf32>, f32', '}', '}']
    return '\n'.join(lines)


def softmax(count=4, boundary='', wrapped_sum=False):
    lines = ['module {', 'func.func private @unknown()',
             'func.func @kernel(%s: vector<4xf32>, %max: vector<4xf32>, '
             '%b: vector<4xf16>, %seed: vector<4xf32>) -> (vector<4xf32>, f32) {',
             '%zero = arith.constant 0.0 : f32']
    for i in range(count):
        lines += [f'%s{i} = arith.subf %s, %max : vector<4xf32>',
                  f'%e{i} = math.exp2 %s{i} : vector<4xf32>',
                  f'%x{i} = vector.extract %e{i}[0] : f32 from vector<4xf32>',
                  f'%p{i} = arith.truncf %e{i} : vector<4xf32> to vector<4xf16>']
    for i in range(count):
        lines += [f'%r{i} = arith.addf ' + ('%zero' if i == 0 else f'%r{i-1}') + f', %x{i} : f32']
    result = f'%r{count-1}'
    if wrapped_sum:
        # Production lane-local recurrence: wrappers precede PV in the input.
        # They must travel with the sum DAG, or all exp2s get forced early.
        lines += [f'%sum = vector.broadcast {result} : f32 to vector<1xf32>',
                  '%base = vector.broadcast %zero : f32 to vector<1xf32>',
                  '%updated = arith.addf %sum, %base : vector<1xf32>',
                  '%view = vector.shape_cast %updated : vector<1xf32> to vector<1x1xf32>',
                  '%scalar = vector.extract %view[0, 0] : f32 from vector<1x1xf32>']
        result = '%scalar'
    if boundary: lines += [boundary]
    for i in range(count):
        lines += [mma(f'm{i}', f'%p{i}', '%b', '%seed' if i == 0 else f'%m{i-1}', str(i))]
    lines += [f'return %m{count-1}, {result} : vector<4xf32>, f32', '}', '}']
    return '\n'.join(lines)


def graph(text):
    """Fingerprints include FP flags, operand order, load addresses and MMA order."""
    values, mmas, returns = {}, [], []
    def resolve(m):
        name = m[0]
        assert name in values or name.startswith('%arg'), name
        return values.get(name, name)
    for line in text.splitlines():
        match = re.match(r'\s*(%\w+) = (.*)', line)
        if match:
            name, body = match.groups()
            body = re.sub(r',? frisk.k_(?:lds|tail)_pair = \d+ : i64', '', body)
            body = re.sub(r'%\w+', resolve, body)
            values[name] = hashlib.sha256(body.encode()).hexdigest()
            if 'v_mmac_' in body: mmas.append(values[name])
        elif re.match(r'\s*return ', line):
            returns.append(re.sub(r'%\w+', resolve, line.strip()))
    return mmas, returns


def export(source):
    low = run(source, '--test-async-to-llvm')
    p = subprocess.run([str(LLVM/'mlir-translate'), '--mlir-to-llvmir'], input=low,
                       capture_output=True, text=True, check=True)
    subprocess.run([str(LLVM/'opt'), '-passes=default<O3>,verify', '-disable-output'],
                   input=p.stdout, text=True, capture_output=True, check=True)
    return p.stdout


def main():
    cases = 0
    for tail, option in [(False, MAIN), (True, TAIL)]:
        for count in (4, 8, 12):
            for spacing in (128, 4096):
                for distance in (1, 2, 4, 8):
                    source = run(paired(count, spacing, tail))
                    result = run(source, option + str(distance))
                    assert graph(result) == graph(source)
                    pairs = defaultdict(list)
                    lines = result.splitlines()
                    for n, line in enumerate(lines):
                        if tag := re.search(r'frisk.k_(?:lds|tail)_pair = (\d+)', line):
                            pairs[tag[1]].append(n)
                    assert len(pairs) == count//2, result
                    for positions in pairs.values():
                        assert len(positions) == 2
                        assert not any('@llvm.amdgcn.sched.barrier' in line for line in lines[min(positions):max(positions)])
                        names = [re.search(r'(%\w+) =', lines[p])[1] for p in positions]
                        use = min(n for n, line in enumerate(lines) if 'v_mmac_' in line and
                                  any(re.search(re.escape(name)+r'[, ]', line) for name in names))
                        assert sum('v_mmac_' in line for line in lines[max(positions)+1:use]) <= distance
                    assert run(result, option + str(distance)) == result
                    cases += 1
        source = paired(tail=tail)
        assert run(source, option+'0') == run(source)
        for invalid in ('9', '-1', 'bad'):
            run(source, option+invalid, ok=False)
        for boundary in ('gpu.barrier', 'func.call @unknown() : () -> ()',
                         'vector.store %a, %sm[%i] : memref<65536xf16, 3>, vector<4xf16>'):
            source = paired(tail=tail, boundary=boundary)
            assert 'frisk.k_' not in run(source, option+'4')
        for source in (paired(tail=tail, align=4), paired(tail=tail, spacing=0),
                       paired(tail=tail, spacing=3)):
            assert 'frisk.k_' not in run(source, option+'4')
        protected = paired(tail=tail).replace('-> (vector<4xf32>, f32) {',
                                             '-> (vector<4xf32>, f32) attributes {strictfp} {')
        assert run(protected, option+'4') == run(protected)
        result = run(paired(tail=tail), option+'4')
        run(result, option+'2', ok=False)
        assert 'frisk.k_' in run(paired(tail=tail, views=True), option+'4')
        ll = export(result)
        assert '"frisk.k_' in ll and 'call void @llvm.amdgcn.sched.barrier' in ll
    # Tail must not revisit main pairs and must require a local reduction.
    source = run(paired(), MAIN+'4')
    assert run(source, TAIL+'4') == source
    assert 'frisk.k_tail_pair' not in run(paired(), TAIL+'4')

    for mode in ('fragment', 'pv'):
        for count in (2, 4, 8):
            source = run(softmax(count))
            result = run(source, SOFT+mode)
            assert graph(result) == graph(source)
            assert result.count('llvm.call @llvm.amdgcn.sched.barrier') == count
            assert run(result, SOFT+mode) == result
            if mode == 'pv':
                events = re.findall(r'math.exp2|v_mmac_f32', result)
                assert events == ['math.exp2', 'v_mmac_f32'] * count, events
            cases += 1
        ll = export(result)
        assert '"frisk.softmax_fragments"' in ll
        for boundary in ('func.call @unknown() : () -> ()',):
            source = softmax(boundary=boundary)
            assert run(source, SOFT+mode) == run(source)
        for attribute in ('strictfp', 'optnone'):
            source = softmax().replace('-> (vector<4xf32>, f32) {',
                                      f'-> (vector<4xf32>, f32) attributes {{{attribute}}} {{')
            assert run(source, SOFT+mode) == run(source)
        # Known synchronization may stay fixed while pure math is scheduled.
        assert 'frisk.softmax_fragments' in run(softmax(boundary='gpu.barrier'), SOFT+mode)
    run(softmax(), SOFT+'bad', ok=False)
    assert run(softmax(), SOFT+'off') == run(softmax())
    run(run(softmax(), SOFT+'pv'), SOFT+'fragment', ok=False)
    for count in (2, 4, 16):
        source = run(softmax(count, wrapped_sum=True))
        result = run(source, SOFT+'pv')
        assert graph(result) == graph(source)
        assert re.findall(r'math.exp2|v_mmac_f32', result) == ['math.exp2', 'v_mmac_f32'] * count
        assert run(result, SOFT+'pv') == result
        export(result)
        cases += 1
    legacy = ROOT / 'build/test/LegacyLLVMExportTest'
    for option in ('--k-lds-prefetch=8', '--k-tail-prefetch=4', '--softmax-fragments=pv',
                   '--k-lds-prefetch=0', '--k-tail-prefetch=0', '--softmax-fragments=off'):
        p = subprocess.run([str(legacy), option], capture_output=True, text=True)
        assert p.returncode != 0 and 'moved to ThreadIR' in p.stderr, p.stderr
        assert not p.stdout
    print(f'PASS {cases} schedule/DAG cases; boundaries, distances, protected functions, replay, conflicts and LLVM/O3 verification')


if __name__ == '__main__':
    main()
