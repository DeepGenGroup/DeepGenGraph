#!/usr/bin/env python3
"""Complete operand lookahead, reuse, budget, boundaries and execution checks."""
import ctypes
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / 'build/test/LegacyLLVMExportTest'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')
MMA = 'v_mmac_f32_16x16x16_f16 $0, $1, $2, $3'
V = '<4 x half>'
C = '<4 x float>'


def command(args, text=None):
    p = subprocess.run(list(map(str, args)), input=text, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-5000:]
    return p.stdout


def export(source, enabled=True, words=8):
    return command([TOOL, f'--frag-prefetch={int(enabled)}', f'--frag-prefetch-words={words}', '-'], source)


def mma(n, a=None, b=None):
    return f'%r{n} = call {C} asm "{MMA}", "=v,v,v,0"({V} %{a or "a"+str(n)}, {V} %{b or "b"+str(n)}, {C} ' + ('zeroinitializer)' if n == 0 else f'%r{n-1})')


def fixture(name='test', boundary='', reuse=False, prepare=False, matrix=False):
    s = 'target triple = "amdgcn-amd-amdhsa"\n'
    s += 'declare void @opaque(ptr addrspace(3)) memory(read)\ndeclare void @llvm.amdgcn.s.barrier()\n'
    s += 'declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) convergent memory(read)\n'
    s += f'define amdgpu_kernel void @{name}(ptr addrspace(3) %p, ptr %out, i64 %index) {{\n'
    for n in range(3):
        if n == 1:
            s += boundary + '\n'
        s += f'%pa{n} = getelementptr half, ptr addrspace(3) %p, i64 {8*n}\n'
        s += f'%pb{n} = getelementptr half, ptr addrspace(3) %p, i64 {8*n+4}\n'
        if not reuse or n == 0:
            s += f'%a{n} = load {V}, ptr addrspace(3) %pa{n}, align 8\n'
        if matrix and n == 1:
            s += '%wide = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %pb1, i32 0)\n'
            s += '%half = bitcast <4 x i32> %wide to <8 x half>\n'
            s += f'%b1 = shufflevector <8 x half> %half, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>\n'
        else:
            s += f'%b{n} = load {V}, ptr addrspace(3) %pb{n}, align 8\n'
        if prepare and n == 1:
            s += f'%scaled = fmul {V} %a1, %a1\n'
        s += mma(n, 'a0' if reuse else 'scaled' if prepare and n == 1 else None) + '\n'
    return s + f'store {C} %r2, ptr %out\nret void\n}}\n'


def order(text, *names):
    return [text.index('%' + name + ' =') for name in names]


def increasing(text, *names):
    positions = order(text, *names)
    assert positions == sorted(positions), (names, text)


def instructions(text):
    # Ignore inserted hints and printer numbering of attribute groups. SSA
    # definitions retain their names; compare every other instruction verbatim.
    return sorted(re.sub(r' #\d+', '', line.strip()) for line in text.splitlines()
                  if line.startswith('  ') and '@llvm.amdgcn.sched.barrier' not in line)


def main():
    source = fixture()
    baseline, scheduled = export(source, False), export(source)
    increasing(baseline, 'r0', 'a1', 'b1', 'r1', 'a2', 'b2', 'r2')
    increasing(scheduled, 'a1', 'b1', 'r0', 'a2', 'b2', 'r1', 'r2')
    assert instructions(baseline) == instructions(scheduled)
    assert export(scheduled) == scheduled, 'idempotence'
    rejected = export(source, words=3)
    assert instructions(rejected) == instructions(baseline)
    increasing(rejected, 'r0', 'a1', 'b1', 'r1', 'a2', 'b2')
    assert 'call void @llvm.amdgcn.sched.barrier' not in rejected
    print('PASS complete A/B lookahead, exact instruction/operand preservation and idempotence')

    for boundary in [
        'call void @llvm.amdgcn.s.barrier()',
        'store half 0xH0000, ptr addrspace(3) %p',
        'fence seq_cst', 'call void @opaque(ptr addrspace(3) %p)',
        'call void asm sideeffect "s_waitcnt lgkmcnt(0)", "~{memory}"()',
        '%volatile = load volatile half, ptr addrspace(3) %p',
        '%atom = atomicrmw add ptr addrspace(3) %p, i32 1 seq_cst',
        '%global = load half, ptr %out',
        'br label %next\nnext:',
    ]:
        text = export(fixture(boundary=boundary))
        increasing(text, 'r0', 'a1', 'b1')
    print('PASS writes, waits, barriers, opaque readonly calls, volatile/atomic/global accesses and control flow')

    # Existing SSA reuse consumes no new packet words; the same load used by
    # both A and B must also be charged once, not once per operand.
    reused = export(fixture(reuse=True), words=2)
    increasing(reused, 'b1', 'r0', 'b2', 'r1')
    assert reused.count('load <4 x half>') == 4
    same = fixture().replace(f'{V} %b1,', f'{V} %a1,')
    same = same.replace(f'%b1 = load {V}, ptr addrspace(3) %pb1, align 8\n', '')
    increasing(export(same, words=2), 'a1', 'r0', 'r1')
    prepared = export(fixture(prepare=True))
    increasing(prepared, 'a1', 'b1', 'r0', 'scaled', 'r1')
    print('PASS shared A/B values, unique packet accounting and delayed FP preparation')

    # Do not split existing read groups to prefetch only part of them. That
    # destroys DS combining; prefetching all of them would deepen the window.
    grouped = fixture()
    begin, end = grouped.index('%pa2 ='), grouped.index('%r2 =')
    group = grouped[begin:end]
    grouped = grouped[:begin] + grouped[end:]
    grouped = grouped.replace('%r1 =', group + '%r1 =')
    grouped_export = export(grouped)
    increasing(grouped_export, 'r0', 'a1', 'b1', 'a2', 'b2', 'r1')
    assert 'call void @llvm.amdgcn.sched.barrier' not in grouped_export
    print('PASS existing LDS read groups remain intact, without deeper prefetch')

    # Real regression: default sched.barrier memory effects block elimination
    # of repeated B/A loads. Narrow effects must retain the hints AND allow CSE.
    repeated = fixture().replace('i64 16\n', 'i64 0\n').replace('i64 20\n', 'i64 4\n')
    for enabled in (False, True):
        text = export(repeated, enabled)
        optimized = command([LLVM/'opt', '-passes=default<O3>', '-S', '-', '-o', '-'], text)
        assert optimized.count('load <4 x half>') == 4, optimized
        assert ('call void @llvm.amdgcn.sched.barrier' in optimized) == enabled
    assert 'memory(inaccessiblemem: readwrite)' in export(repeated)
    print('PASS O3 eliminates duplicate LDS reads and retains scheduling hints')

    # The raw A/B packet fits four words, but keeping its two computed addresses
    # alive across independent vector work requires more. Verify the pressure
    # guard sees address live-outs, rather than just summing loaded data bytes.
    pressure = fixture().replace('i64 %index)', 'i64 %index, <16 x float> %wide)')
    pressure = pressure.replace('%pa1 =',
        '%junk = fadd <16 x float> %wide, %wide\n'
        '%small = shufflevector <16 x float> %junk, <16 x float> poison, '
        '<4 x i32> <i32 0, i32 1, i32 2, i32 3>\n'
        '%mix = fadd <4 x float> %r0, %small\n%pa1 =')
    pressure = pressure.replace('<4 x float> %r0)', '<4 x float> %mix)')
    for operand, offset in [('a', 16), ('b', 20)]:
        pressure = pressure.replace(
            f'%p{operand}2 = getelementptr half, ptr addrspace(3) %p, i64 {offset}',
            f'%p{operand}2 = getelementptr half, ptr addrspace(3) %p{operand}1, i64 8')
    increasing(export(pressure, words=4), 'r0', 'a1', 'b1')
    increasing(export(pressure, words=8), 'a1', 'b1', 'r0')
    print('PASS local peak budget includes live-out addresses and independent temporaries')

    matrix = export(fixture(matrix=True))
    increasing(matrix, 'a1', 'wide', 'r0', 'half', 'b1', 'r1')
    increasing(export(fixture(matrix=True), words=4), 'r0', 'a1', 'wide')
    assert matrix.count('call <4 x i32> @llvm.hcu.ds.read') == 1
    dependent = fixture().replace('%pa1 = getelementptr half, ptr addrspace(3) %p, i64 8',
        f'%cvalue = extractelement {C} %r0, i32 0\n%idx = fptosi float %cvalue to i64\n'
        '%pa1 = getelementptr half, ptr addrspace(3) %p, i64 %idx')
    increasing(export(dependent), 'r0', 'a1', 'b1')
    indirect = fixture().replace('%pa1 = getelementptr half, ptr addrspace(3) %p, i64 8',
        '%pa1 = load ptr addrspace(3), ptr addrspace(3) %p')
    increasing(export(indirect), 'r0', 'a1', 'b1')
    print('PASS matrix packets, full-group fallback, accumulator-dependent and indirect addresses')

    intrinsic = source.replace(f'asm "{MMA}", "=v,v,v,0"',
                               '@llvm.hcu.mmac.f32.16x16x16.f16')
    intrinsic = re.sub(r'(call <4 x float> @llvm.hcu.mmac[^\n]+)\)',
                       r'\1, i1 false, i1 false)', intrinsic)
    intrinsic += ('declare <4 x float> @llvm.hcu.mmac.f32.16x16x16.f16('
                  '<4 x half>, <4 x half>, <4 x float>, i1, i1) '
                  'convergent nounwind willreturn memory(none)\n')
    increasing(export(intrinsic), 'a1', 'b1', 'r0', 'a2', 'b2', 'r1')
    padded = source.replace(MMA, r's_nop 1\0A\09' + MMA + r'\0A\09s_nop 1')
    increasing(export(padded), 'a1', 'b1', 'r0')
    assert export(padded).count('s_nop 1') == 6
    opaque = source.replace(MMA, MMA + r'\0A\09s_waitcnt lgkmcnt(0)')
    assert 'call void @llvm.amdgcn.sched.barrier' not in export(opaque)
    cpu_target = source.replace('amdgcn-amd-amdhsa', 'x86_64-unknown-linux-gnu').replace('amdgpu_kernel ', '')
    assert 'call void @llvm.amdgcn.sched.barrier' not in export(cpu_target)
    optnone = source.replace('i64 %index) {', 'i64 %index) noinline optnone {')
    assert 'call void @llvm.amdgcn.sched.barrier' not in export(optnone)
    print('PASS HCU intrinsic, bare/padded MMAC, opaque assembly and target/optnone guards')

    for bad in ('--frag-prefetch=2', '--frag-prefetch-words=0', '--frag-prefetch-words=-1'):
        p = subprocess.run([TOOL, bad, '-'], input=source, capture_output=True, text=True)
        assert p.returncode != 0 and 'must be' in p.stderr
    command([LLVM/'opt', '-passes=verify', '-disable-output', '-'], scheduled)
    print('PASS LLVM verifier and option validation')

    # Upstream LLVM cannot assemble HCU MMAC. For a machine-scheduler probe
    # only, use the corresponding gfx90a MFMA with the same register signature.
    # Exercise one merged DS instruction versus two independent DS instructions
    # per packet: the backend must compute 1 versus 2 outstanding accesses.
    # This is not a gfx936 numerical or performance measurement.
    for separate_b, expected in [(False, 1), (True, 2)]:
        probe = source
        # Make different packets too far apart for a cross-packet read2. Reads
        # within a packet can still merge; different A/B bases cannot. Nearby
        # packets are now deliberately allowed to merge across the old leading
        # boundary, and their readiness is not described by a fixed count.
        for n in (1, 2):
            for offset in (0, 4):
                probe = probe.replace(f'i64 {8*n+offset}\n',
                                      f'i64 {(1 << 20)*n+offset}\n')
        if separate_b:
            probe = probe.replace('ptr %out, i64 %index',
                                  'ptr %out, i64 %index, ptr addrspace(3) %q')
            probe = re.sub(r'(%pb\d+ = getelementptr half, ptr addrspace\(3\)) %p',
                           r'\1 %q', probe)
        probe = export(probe).replace(MMA, 'v_mfma_f32_16x16x16f16 $0, $1, $2, $3')
        optimized = command([LLVM/'opt', '-passes=default<O3>', '-S', '-', '-o', '-'], probe)
        assembly = command([LLVM/'llc', '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx90a',
                            '-O3', '-', '-o', '-'], optimized)
        assert assembly.count(f's_waitcnt lgkmcnt({expected})') == 2, assembly
        assert assembly.count('v_mfma_f32_16x16x16f16') == 3
        assert '; ScratchSize: 0' in assembly
        mma_positions = [m.start() for m in re.finditer('v_mfma_f32_16x16x16f16', assembly)]
        reads = [m.start() for m in re.finditer(r'\bds_read\w* ', assembly)]
        assert sum(p < mma_positions[0] for p in reads) == 2 * expected
        assert sum(mma_positions[0] < p < mma_positions[1] for p in reads) == expected
    print('PASS gfx90a machine scheduler: merged/split LDS reads produce lgkmcnt(1)/(2), no spills')

    # Nearby packets must keep DS combining, including the current/next pair.
    # Compare emitted instructions, not just unchanged IR load counts.
    ds_counts = []
    for enabled in (False, True):
        probe = export(source, enabled).replace(MMA, 'v_mfma_f32_16x16x16f16 $0, $1, $2, $3')
        optimized = command([LLVM/'opt', '-passes=default<O3>', '-S', '-', '-o', '-'], probe)
        assembly = command([LLVM/'llc', '-mtriple=amdgcn-amd-amdhsa', '-mcpu=gfx90a',
                            '-O3', '-', '-o', '-'], optimized)
        ds_counts.append(len(re.findall(r'^\s+ds_read\w* ', assembly, re.M)))
        assert '; ScratchSize: 0' in assembly
    assert ds_counts == [3, 3], ds_counts
    print('PASS nearby packet DS combining: three read2 instructions in both off/on')

    # Execute transformed loads, addresses and arithmetic on CPU. Substitute
    # only the target-specific MMA with deterministic vector FP arithmetic, and
    # erase scheduling hints. This tests scheduling, not GPU MMA lane geometry.
    def host(text):
        text = re.sub(r'^target triple.*$', '', text, flags=re.M)
        text = text.replace('amdgpu_kernel ', '').replace(' addrspace(3)', '')
        text = re.sub(r'^declare .*$', '', text, flags=re.M)
        text = re.sub(r'^  call void @llvm.amdgcn.sched.barrier.*\n', '', text, flags=re.M)
        text = text.replace(f'asm "{MMA}", "=v,v,v,0"', '@cpu_mma')
        return text + f'''\ndefine {C} @cpu_mma({V} %a, {V} %b, {C} %c) {{
          %x = fpext {V} %a to {C}
          %y = fpext {V} %b to {C}
          %m = fmul {C} %x, %y
          %r = fadd {C} %m, %c
          ret {C} %r
        }}\n'''
    with tempfile.TemporaryDirectory(prefix='frag-prefetch-') as directory:
        directory = Path(directory)
        functions = []
        for label, text in [('off', baseline), ('on', scheduled)]:
            src, lib = directory/f'{label}.ll', directory/f'{label}.so'
            src.write_text(host(text))
            obj = directory/f'{label}.o'
            command([LLVM/'llc', '-filetype=obj', '-relocation-model=pic', src, '-o', obj])
            command(['cc', '-shared', obj, '-o', lib])
            function = ctypes.CDLL(str(lib)).test
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
            functions.append(function)
        import struct
        rng = random.Random(846)
        for _ in range(100):
            data = ctypes.create_string_buffer(b''.join(struct.pack('e', rng.uniform(-3, 3)) for _ in range(24)))
            result = [(ctypes.c_float * 4)() for _ in functions]
            for function, out in zip(functions, result):
                function(data, out, 0)
            assert bytes(result[0]) == bytes(result[1])
    print('PASS 100 randomized CPU differential executions (MMAC replaced by identical FP model)')


if __name__ == '__main__':
    main()
