#!/usr/bin/env python3
"""Thread PV order, SSA, effects, rejection, replay and LLVM guard lowering."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LLVM = Path('/data2/xsl/install/bin')


def fixture(chains=4, steps=3, dynamic=False, warp=False, distinct_packets=False):
    vt = 'vector<4xf32>'
    lines = ['module {', 'func.func private @unknown()',
             f'func.func @kernel(%sm: memref<4096xf16, 3>, %out: memref<4xf32>, %a: vector<4xf16>, %seed: {vt}, %idx: index) -> (' + ', '.join([vt]*chains) + ') {',
             '%i = arith.constant 0 : index']
    for c in range(chains if distinct_packets else 1):
        for k in range(steps):
            packet = f'{c}_{k}' if distinct_packets else str(k)
            lines += [f'%r{packet} = frisk.read_matrix_from_shm %sm[%i] {{m = 32 : i64, n = 16 : i64, element_bit_width = 16 : i64, offset = {(c*steps+k)*256} : i32}} : memref<4096xf16, 3> -> vector<4xi32>',
                      f'%v{packet} = vector.bitcast %r{packet} : vector<4xi32> to vector<8xf16>']
    for c in range(chains):
        for k in range(steps):
            packet = f'{c}_{k}' if distinct_packets else str(k)
            for j in range(4):
                pos = '%idx' if dynamic and c % 2 else str((c % 2)*4+j)
                lines += [f'%b{c}_{k}_{j} = vector.extract %v{packet}[{pos}] : f16 from vector<8xf16>']
            lines += [f'%b{c}_{k} = vector.from_elements ' + ', '.join(f'%b{c}_{k}_{j}' for j in range(4)) + ' : vector<4xf16>']
            prev = '%seed' if k == 0 else f'%s{c}_{k-1}'
            tag = f'test.id = "{c}:{k}"'
            if warp:
                lines += [f'%s{c}_{k} = frisk.warp_mma_rr %a, %b{c}_{k}, {prev} {{inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", inst_constraints = "=v,v,v,0", {tag}}} : vector<4xf16>, vector<4xf16>, {vt} -> {vt}']
            else:
                lines += [f'%s{c}_{k} = llvm.inline_asm has_side_effects convergent {{{tag}}} "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", "=v,v,v,0" %a, %b{c}_{k}, {prev} : (vector<4xf16>, vector<4xf16>, {vt}) -> {vt}']
            if c == 0 and k == 0:
                lines += ['gpu.barrier']
    lines += ['return ' + ', '.join(f'%s{c}_{steps-1}' for c in range(chains)) + ' : ' + ', '.join([vt]*chains), '}', '}']
    return '\n'.join(lines) + '\n'


def ids(text):
    return re.findall(r'test.id = "([\d:]+)"', text)


def result_graph(text):
    """SSA-name independent fingerprints of every MMA and returned value."""
    values, mmas, returned = {}, {}, []
    def resolve(match):
        name = match[0]
        assert name in values or name.startswith('%arg'), name
        return values.get(name, name)
    for line in text.splitlines():
        definition = re.match(r'\s*(%[\w]+) = (.*)', line)
        if definition:
            name, body = definition.groups()
            body = re.sub(r'frisk\.(?:(?:pv|qk)_interleave|mma_interleave) = \d+ : i32,? ?', '', body)
            body = re.sub(r'%\w+', resolve, body)
            digest = hashlib.sha256(body.encode()).hexdigest()
            values[name] = digest
            tag = re.search(r'test.id = "([\d:]+)"', line)
            if tag:
                mmas[tag[1]] = digest
        elif re.match(r'\s*return ', line):
            returned.append(re.sub(r'%\w+', resolve, line.strip()))
    return mmas, returned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, default=ROOT/'build/test/ThreadTilingTest')
    args = parser.parse_args()
    cases = 0

    def run(source, *passes, ok=True):
        p = subprocess.run([str(args.tool), '--verify-each', *passes], input=source,
                           capture_output=True, text=True, timeout=120)
        assert (p.returncode == 0) == ok, p.stderr
        return p.stdout if ok else p.stderr

    def transform(source, width=4):
        return run(source, f'--frisk-thread-interleave-pv=width={width}')

    def reject(source, width=4):
        assert transform(source, width) == run(source)

    for warp in (False, True):
        for dynamic in (False, True):
            source = fixture(warp=warp, dynamic=dynamic)
            graph = result_graph(run(source))
            for width in (0, 1, 2, 4):
                out = transform(source, width)
                assert result_graph(out) == graph, 'MMAC operands or output graph changed'
                w = width or 1
                expected = [f'{c}:{k}' for group in range(0, 4, w)
                            for k in range(3) for c in range(group, group+w)]
                assert ids(out) == expected, ids(out)
                assert out.count('llvm.call @llvm.amdgcn.sched.barrier') == (24 if width else 0)
                # Real barriers retain their original MMAC ordinal.
                assert len(ids(out.split('gpu.barrier')[0])) == 1
                assert transform(out, width) == out, 'not idempotent'
                # Canonicalization / CSE must retain guards and MMAC order.
                cleaned = run(out, '--canonicalize', '--cse')
                assert ids(cleaned) == expected
                assert cleaned.count('llvm.call @llvm.amdgcn.sched.barrier') == (24 if width else 0)
                cases += 1

    for chains in (4, 8, 16):
        source = fixture(chains=chains, steps=2, distinct_packets=True)
        out = transform(source)
        assert result_graph(out) == result_graph(run(source))
        assert ids(out) == [f'{c}:{k}' for group in range(0, chains, 4)
                            for k in range(2) for c in range(group, group+4)]
        assert transform(out) == out
        assert len(ids(out.split('gpu.barrier')[0])) == 1
        cases += 1
    # The availability proof must include every chain's packets, not just
    # the first chain. A late read in another output column is not movable.
    source = fixture(distinct_packets=True)
    lines = source.splitlines()
    moving = [l for l in lines if l.startswith(('%r3_2 =', '%v3_2 ='))]
    lines = [l for l in lines if l not in moving]
    at = next(i for i, l in enumerate(lines) if l.startswith('%b3_2_0'))
    lines[at:at] = moving
    reject('\n'.join(lines))

    # Two otherwise valid packet-local windows separated by fresh LDS reads:
    # the combined group must fail, while both former opportunities survive.
    lines = fixture(chains=8, steps=2).splitlines()
    packet_lines = []
    for line in lines:
        if line.startswith(('%r0 =', '%v0 =', '%r1 =', '%v1 =')):
            packet_lines.append(re.sub(r'%([rv])([01])\b',
                lambda m: '%'+m[1]+str(int(m[2])+2), line))
    at = next(i for i,l in enumerate(lines) if l.startswith('%b4_0_0'))
    lines[at:at] = packet_lines
    for i,line in enumerate(lines):
        if re.match(r'%b[4-7]_\d_\d = vector.extract',line):
            lines[i] = re.sub(r'%v([01])\b',lambda m: '%v'+str(int(m[1])+2),line)
    source = '\n'.join(lines)
    out = transform(source)
    assert result_graph(out) == result_graph(run(source))
    assert ids(out) == [f'{c}:{k}' for group in (0,4) for k in range(2) for c in range(group,group+4)]
    assert transform(out) == out
    cases += 1

    base = fixture()
    bad = {
        'unknown call': base.replace('gpu.barrier', 'gpu.barrier\nfunc.call @unknown() : () -> ()'),
        'extra load': base.replace('gpu.barrier', 'gpu.barrier\n%x = memref.load %out[%i] : memref<4xf32>'),
        'store': base.replace('gpu.barrier', 'gpu.barrier\n%x = vector.extract %seed[0] : f32 from vector<4xf32>\nmemref.store %x, %out[%i] : memref<4xf32>'),
        'cross-chain': base.replace('%a, %b1_0, %seed', '%a, %b1_0, %s0_2'),
        'mixed packets': base.replace('%b1_1_0 = vector.extract %v1', '%b1_1_0 = vector.extract %v0'),
        'wrong instruction': base.replace('v_mmac_f32_16x16x16_f16 $0, $1, $2, $3', 'v_mmac_f32_16x16x16_f16 $0, $1, $2, $3; s_barrier'),
        'memory clobber': base.replace('"=v,v,v,0"', '"=v,v,v,0,~{memory}"'),
        'too few chains': fixture(chains=1),
        'nondivisible': fixture(chains=3),
        'single step': fixture(steps=1),
    }
    lines = base.splitlines()
    moving = [l for l in lines if l.startswith(('%r2 =', '%v2 ='))]
    lines = [l for l in lines if l not in moving]
    index = next(i for i, l in enumerate(lines) if l.startswith('%b0_2_0'))
    lines[index:index] = moving
    bad['late packet'] = '\n'.join(lines)
    bad['fanout'] = base.replace('gpu.barrier', 'gpu.barrier\n%x = vector.extract %s0_0[0] : f32 from vector<4xf32>').replace(
        'vector<4xf32>) {', 'vector<4xf32>, f32) {').replace(
        'return %s0_2, %s1_2, %s2_2, %s3_2 :', 'return %s0_2, %s1_2, %s2_2, %s3_2, %x :').replace(
        'vector<4xf32>\n}', 'vector<4xf32>, f32\n}')
    for name, source in bad.items():
        try:
            reject(source)
        except AssertionError as error:
            raise AssertionError(name) from error
        cases += 1
    # A failed group must not mutate its first two chains before rejecting.
    reject(base.replace('%a, %b3_0, %seed', '%a, %b3_0, %s2_2'))
    cases += 1
    for width in (3, 8, -1):
        run(base, f'--frisk-thread-interleave-pv=width={width}', ok=False)
        cases += 1
    conflict = base.replace('module {', 'module {\nllvm.func @llvm.amdgcn.sched.barrier(i64)')
    assert 'conflicting declaration' in run(conflict, '--frisk-thread-interleave-pv=width=4', ok=False)
    cases += 1

    # Export through production lowering; verify callsite effects, barrier count
    # and retention through LLVM O3 without running GPU code.
    with tempfile.TemporaryDirectory(prefix='thread-pv-') as tmp:
        tmp = Path(tmp)
        lowered = run(transform(base), '--test-async-to-llvm')
        p = subprocess.run([str(LLVM/'mlir-translate'), '--mlir-to-llvmir'],
                           input=lowered, text=True, capture_output=True)
        assert p.returncode == 0, p.stderr
        (tmp/'kernel.ll').write_text(p.stdout)
        p = subprocess.run([str(LLVM/'opt'), '-S', '-passes=default<O3>,verify',
                            str(tmp/'kernel.ll'), '-o', str(tmp/'kernel.o3.ll')], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        optimized = (tmp/'kernel.o3.ll').read_text()
        assert 'memory(inaccessiblemem: readwrite)' in optimized
        assert optimized.count('call void @llvm.amdgcn.sched.barrier(i32 0)') == 24
        cases += 1
        # The production flag still validates widths; the removed LLVM pass
        # must not remain as a silent fallback in the legacy exporter.
        p = subprocess.run([str(args.tool.parent/'MyTest'), str(ROOT/'test/test_input_shared_d.mlir'),
                            '0', str(tmp/'unused.ll'), '--pv-interleave=3'],
                           capture_output=True, text=True, timeout=30)
        assert p.returncode and 'use --pv-interleave=0|1|2|4' in p.stderr
        p = subprocess.run([str(args.tool.parent/'LegacyLLVMExportTest'),
                            '--pv-interleave=4', str(tmp/'kernel.ll')],
                           capture_output=True, text=True, timeout=30)
        assert p.returncode and 'moved to Thread IR' in p.stderr
        cases += 2
    print(f'PASS {cases} Thread PV cases (widths, views, handoff, idempotence, rejection, LLVM/O3)')


if __name__ == '__main__':
    main()
