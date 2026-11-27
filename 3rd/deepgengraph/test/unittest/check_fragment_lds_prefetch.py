#!/usr/bin/env python3
"""Fragment LDS read-ahead legality, replay and ThreadIR-only pair closure."""
from pathlib import Path
import re
import subprocess

TOOL = Path(__file__).resolve().parents[2] / 'build/test/ThreadTilingTest'
PASS = '--frisk-prefetch-fragment-lds='
FINALIZE = '--frisk-thread-prefetch-paired-lds=window=fragment'
INST = 'v_mmac_f32_16x16x16_f16 $0, $1, $2, $3'


def run(source, *passes, ok=True):
    p = subprocess.run([str(TOOL), '--verify-each', *passes], input=source,
                       text=True, capture_output=True, timeout=180)
    assert (p.returncode == 0) == ok, p.stderr
    return p.stdout if ok else p.stderr


def fixture(boundary='', space=3, loop=False):
    def layout(role, widths, repeats, lanes):
        return (f'{{instruction = "{INST}", role = "{role}", '
                f'thread_widths = array<i64: {widths}>, warp_repeat = array<i64: {repeats}>, '
                f'warp_layout = array<i64: {lanes}>, warp_threads = 64 : i64, '
                'thread_creg_order = array<i64: 0, 1>, warp_layout_order = array<i64: 0, 1>, '
                'warp_repeat_order = array<i64: 0, 1>}')
    lines = [f'!a = !frisk.fragment<vector<1x4xf16>, {layout("A", "1, 4", "1, 1", "16, 4")}>',
             f'!b = !frisk.fragment<vector<4x1xf16>, {layout("B", "4, 1", "1, 1", "4, 16")}>',
             f'!c = !frisk.fragment<vector<1x4xf32>, {layout("C", "1, 1", "1, 4", "16, 4")}>',
             'module { func.func private @opaque()',
             f'func.func @kernel(%sm: memref<16x128xf16, {space}>, %av: vector<1x4xf16>, '
             '%cv: vector<1x4xf32>, %idx: index, %task: !frisk.async_task) -> vector<1x4xf32> '
             'attributes {thread_num = 64 : i32} {',
             '%a = frisk.pack_frag %av : vector<1x4xf16> -> !a',
             '%seed = frisk.pack_frag %cv : vector<1x4xf32> -> !c',
             '%zero = arith.constant 0 : index']
    load_attrs = ('block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, '
                  'block_repeat = array<i64: 1, 8>, thread_widths = array<i64: 4, 1>, '
                  'warp_repeat = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, '
                  'warp_layout_order = array<i64: 0, 1>, thread_creg_order = array<i64: 0, 1>, '
                  'warp_repeat_order = array<i64: 0, 1>, warp_inst_unroll = array<i64: 1, 1>, '
                  'warp_threads = 64 : i64, ignore_dim = -1 : i64, tile_type = vector<4x8xf16>, '
                  'tile_layout = affine_map<(d0,d1,d2,d3,d4,d5,d6) -> '
                  '(d1*16+d3*16+(d0 floordiv 16)*4+d6 mod 4, '
                  'd2*16+d4*16+d5*16+d0 mod 16+d6 floordiv 4)>')
    if loop:
        lines.append('%out = affine.for %i = 0 to 8 iter_args(%acc = %seed) -> !c {')
    for i in range(1 if loop else 8):
        if not loop:
            lines.append(f'%i{i} = arith.constant {i} : index')
        index = '%i' if loop else f'%i{i}'
        lines.append(f'%b{i} = frisk.load_frag %sm[%zero, {index}] {{{load_attrs}}} : '
                     f'memref<16x128xf16, {space}> -> !b')
        acc = '%acc' if loop else ('%seed' if i == 0 else f'%m{i-1}')
        lines.append(f'%m{i} = frisk.gemm_frag %a, %b{i}, acc = {acc} '
                     f'{{inst_name = "{INST}", inst_constraints = "=v,v,v,0"}} : !a, !b, !c -> !c')
        if boundary:
            lines.append(boundary)
    if loop:
        lines.extend(['affine.yield %m0 : !c', '} {frisk.fragment_loop}'])
    lines.extend([f'%v = frisk.unpack_frag {"%out" if loop else "%m7"} : !c -> vector<1x4xf32>',
                  'return %v : vector<1x4xf32>', '} }'])
    return '\n'.join(lines)


def events(text):
    return re.findall(r'frisk\.(load_frag|gemm_frag)\b', text)


def graph(text):
    """Compare the complete SSA recurrence by load coordinate, ignoring names."""
    values = {}
    for line in text.splitlines():
        m = re.match(r'\s*(%\w+) = arith.constant (\d+) : index', line)
        if m:
            values[m[1]] = ('index', int(m[2]))
        m = re.match(r'\s*(%\w+) = frisk.load_frag %\w+\[(%\w+), (%\w+)\]', line)
        if m:
            values[m[1]] = ('read', values[m[2]], values[m[3]])
        m = re.match(r'\s*(%\w+) = frisk.gemm_frag (%\w+), (%\w+), acc = (%\w+)', line)
        if m:
            values[m[1]] = ('mma', values[m[3]], values.get(m[4], 'seed'))
    return [v for v in values.values() if v[0] == 'mma']


def main():
    source = run(fixture())
    for distance in (1, 2, 4, 8):
        result = run(source, PASS + f'distance={distance}')
        assert graph(source) == graph(result)
        assert events(result) != events(source)
        assert result.count('frisk.lds_prefetch_anchor') > 0
        assert run(result, PASS + f'distance={distance}') == result
        assert 'frisk.sched_barrier' not in result
        # Bootstrap just the next MMA's operand before MMA 0; remaining reads
        # start after MMA 0 so their address setup cannot delay startup.
        read = mma = 0
        positions = []
        for e in events(result):
            if e == 'load_frag':
                positions.append(mma)
                read += 1
            else:
                mma += 1
        assert read == mma == 8
        assert positions == [0, 0] + [max(1, j-distance) for j in range(2, 8)]

    assert run(source, PASS + 'distance=0') == source
    run(source, PASS + 'distance=9', ok=False)
    run(result, PASS + 'distance=2', ok=False)
    for boundary in ('gpu.barrier', 'func.call @opaque() : () -> ()',
                     'frisk.wait_buffer_ready %task : !frisk.async_task',
                     'memref.store %av_scalar, %sm[%zero, %zero] : memref<16x128xf16, 3>'):
        blocked = fixture(boundary).replace('%a = frisk.pack_frag',
            '%av_scalar = vector.extract %av[0, 0] : f16 from vector<1x4xf16>\n%a = frisk.pack_frag')
        blocked = run(blocked)
        result = run(blocked, PASS + 'distance=8 reduction-distance=4')
        assert events(result) == events(blocked)
        assert 'frisk.lds_prefetch_anchor' not in result
    global_reads = run(fixture(space=1), PASS + 'distance=8')
    assert events(global_reads) == events(source)
    assert 'frisk.lds_prefetch_anchor' not in global_reads
    authored = run(fixture().replace('thread_num = 64 : i32}',
                                    'thread_num = 64 : i32, pipeline.scheduled}'))
    assert run(authored, PASS + 'distance=8') == authored
    loop = run(fixture(loop=True), PASS + 'distance=4')
    assert 'affine.for' not in loop and len(events(loop)) == 16
    assert graph(loop) == graph(source)

    # Independent address setup for a later operand must not be pulled onto
    # MMA 0's startup path just because its consumer is within distance 8.
    startup = fixture().replace('%i2 = arith.constant 2 : index',
        '%two = arith.constant 2 : index\n%i2 = arith.addi %idx, %two : index')
    startup = run(startup, PASS + 'distance=8')
    mmas = [m.start() for m in re.finditer('frisk.gemm_frag', startup)]
    setup = startup.index('arith.addi')
    assert mmas[0] < setup < mmas[1]

    # A wait/barrier starts a new memory epoch, but not a new accumulation.
    # Follow the carried accumulator through pack/unpack instead of applying
    # the cold-start limit to every epoch (which repeatedly drains the window).
    carried = fixture()
    load0 = next(l for l in carried.splitlines() if '%b0 = frisk.load_frag' in l)
    mma0 = next(l for l in carried.splitlines() if '%m0 = frisk.gemm_frag' in l)
    prefix = ('%prior_index = arith.constant 7 : index\n' +
              load0.replace('%b0', '%prior_b').replace('%i0', '%prior_index') + '\n' +
              mma0.replace('%m0', '%prior_m').replace('%b0', '%prior_b') + '\n' +
              '%prior_v = frisk.unpack_frag %prior_m : !c -> vector<1x4xf32>\n'
              '%carried = frisk.pack_frag %prior_v : vector<1x4xf32> -> !c\n'
              'gpu.barrier\n')
    carried = carried.replace(mma0, mma0.replace('acc = %seed', 'acc = %carried'))
    carried = carried.replace(load0, prefix + load0)

    def read_positions(text):
        n = 0
        positions = []
        for event in events(text.split('gpu.barrier', 1)[1]):
            if event == 'load_frag':
                positions.append(n)
            else:
                n += 1
        return positions

    warm = run(carried, PASS + 'distance=8 reduction-distance=2')
    assert graph(run(carried)) == graph(warm)
    assert read_positions(warm) == [0] * 8

    # Honor the shorter reduction window BEFORE the wide sweep. The fragment
    # reduction's lane-local instructions do not exist until later lowering.
    def with_tail(text, operand):
        return text.replace('-> vector<1x4xf32> attributes',
            '-> (vector<1x4xf32>, !frisk.fragment<vector<1x1xf32>, {}>) attributes').replace(
            'return %v : vector<1x4xf32>',
            f'%rf = frisk.pack_frag {operand} : vector<1x4xf32> -> !frisk.fragment<vector<1x4xf32>, {{}}>\n'
            '%red = frisk.reduce_frag "add" %rf {dim = 1 : i64, lane_extent = 4 : i64, '
            'lane_stride = 16 : i64, warp_threads = 64 : i64, reduction_width = 4 : i64, '
            'output_fragment_shape = array<i64: 1, 1>} : '
            '!frisk.fragment<vector<1x4xf32>, {}> -> !frisk.fragment<vector<1x1xf32>, {}>\n'
            'return %v, %red : vector<1x4xf32>, !frisk.fragment<vector<1x1xf32>, {}>')

    tail = with_tail(carried, '%v')
    limited = run(tail, PASS + 'distance=8 reduction-distance=2')
    assert graph(run(tail)) == graph(limited)
    assert read_positions(limited) == [max(0, j-2) for j in range(8)]
    assert run(limited, PASS + 'distance=8 reduction-distance=2') == limited
    assert read_positions(run(tail, PASS + 'distance=8')) == [0] * 8
    unrelated = with_tail(carried, '%cv')
    assert read_positions(run(unrelated, PASS + 'distance=8 reduction-distance=2')) == [0] * 8

    # Real SSA address dependence forbids moving a load ahead of its producer.
    dependent = fixture().replace('%i1 = arith.constant 1 : index',
        '%v0 = frisk.unpack_frag %m0 : !c -> vector<1x4xf32>\n'
        '%s0 = vector.extract %v0[0, 0] : f32 from vector<1x4xf32>\n'
        '%n0 = arith.fptoui %s0 : f32 to i32\n'
        '%i1 = arith.index_cast %n0 : i32 to index')
    dependent = run(dependent, PASS + 'distance=8')
    first_mma = dependent.index('frisk.gemm_frag')
    index_value = re.search(r'(%\w+) = arith.index_cast', dependent)[1]
    dependent_load = next(line for line in dependent.splitlines()
                          if 'frisk.load_frag' in line and index_value + ']' in line)
    assert first_mma < dependent.index('arith.fptoui') < dependent.index(dependent_load)

    # A live scalar reduction stays in place; only the read passes across it.
    reduction = fixture('')
    reduction = reduction.replace('%i1 = arith.constant 1 : index',
        '%r = arith.addf %cv, %cv : vector<1x4xf32>\n%i1 = arith.constant 1 : index')
    reduction = reduction.replace('return %v :', '%sum = arith.addf %v, %r : vector<1x4xf32>\nreturn %sum :')
    narrow = run(reduction, PASS + 'distance=8')
    broad = run(reduction, PASS + 'reduction-distance=4')
    assert events(narrow)[:2] == ['load_frag', 'gemm_frag']
    assert events(broad)[:2] == ['load_frag', 'load_frag']

    for extent in (1, 4):
        collective = fixture().replace('%i1 = arith.constant 1 : index',
            '%rf = frisk.pack_frag %cv : vector<1x4xf32> -> !frisk.fragment<vector<1x4xf32>, {}>\n'
            '%red = frisk.reduce_frag "add" %rf '
            f'{{dim = 1 : i64, lane_extent = {extent} : i64, lane_stride = 16 : i64, '
            'warp_threads = 64 : i64, reduction_width = 4 : i64, '
            'output_fragment_shape = array<i64: 1, 1>} : '
            '!frisk.fragment<vector<1x4xf32>, {}> -> !frisk.fragment<vector<1x1xf32>, {}>\n'
            '%i1 = arith.constant 1 : index')
        collective = collective.replace('-> vector<1x4xf32> attributes',
            '-> (vector<1x4xf32>, !frisk.fragment<vector<1x1xf32>, {}>) attributes')
        collective = collective.replace('return %v : vector<1x4xf32>',
            'return %v, %red : vector<1x4xf32>, !frisk.fragment<vector<1x1xf32>, {}>')
        output = run(collective, PASS + 'distance=8 reduction-distance=4')
        assert events(output)[:2] == (['load_frag', 'load_frag'] if extent == 1
                                     else ['load_frag', 'gemm_frag'])

    # Lowering carries the logical anchor all the way to the MMAC inline asm.
    # Match the production context, which has already loaded the LLVM dialect.
    lowering_source = loop.replace('module {', 'module { llvm.func @llvm_context()')
    lowered = run(lowering_source, '--convert-fragment-to-thread', '--finalize-thread-tiling',
                  '--thread-ir-legalize')
    assert lowered.count('frisk.lds_prefetch_anchor') == loop.count('frisk.lds_prefetch_anchor')
    optimized = run(lowered, '--ir-deep-optimize=reorder-after-unroll=true')
    # Two bootstrap four-element reads precede MMA 0 even at distance 4. The generic late
    # scheduler must not sink their scalar payloads back across that anchor.
    assert optimized.split('llvm.inline_asm', 1)[0].count('memref.load') == 8
    finalized = run(lowered, FINALIZE)
    assert 'frisk.lds_prefetch_anchor' not in finalized
    assert 'frisk.sched_barrier' in finalized
    assert finalized.count('mask = 1028 : i32') == finalized.count('frisk.sched_barrier')
    assert run(finalized, FINALIZE) == finalized

    from check_thread_fragment_schedules import paired, graph as thread_graph
    physical = run(paired())
    assert thread_graph(run(physical, FINALIZE)) == thread_graph(physical)
    # No pair closure may move a read across an MMA in this mode.
    assert 'frisk.lds_pair' not in run(physical, FINALIZE)
    lines = paired().splitlines()
    load_lines = [line for line in lines if '= frisk.lds_load' in line]
    lines = [line for line in lines if '= frisk.lds_load' not in line]
    at = next(i for i, line in enumerate(lines) if '= llvm.inline_asm' in line)
    lines[at:at] = load_lines
    batched = run('\n'.join(lines))
    result = run(batched, FINALIZE)
    assert result.count('frisk.lds_pair') == 8
    assert thread_graph(result) == thread_graph(batched)
    assert run(result, FINALIZE) == result

    # Physical read2 closure may cross an independent MMA inside the existing
    # FragmentIR issue window. It must stop at the earlier read's first use.
    # The old single-MMA-gap restriction left six of these eight reads single.
    def outstanding(spacing=128, boundary=''):
        lines = paired(spacing=spacing).splitlines()
        reads = [line for line in lines if '= frisk.lds_load' in line]
        lines = [line for line in lines if '= frisk.lds_load' not in line]
        out = []
        for line in lines:
            if '= llvm.inline_asm' in line:
                i = int(re.search(r'test.id = "(\d+)"', line)[1])
                if i == 0:
                    out.extend(reads[:3])
                elif i < 6:
                    if boundary: out.append(boundary)
                    out.append(reads[i+2])
                if i < 6:
                    line = line.replace('{test.id', '{frisk.lds_prefetch_anchor, test.id')
            out.append(line)
        return '\n'.join(out).replace('-> (vector<4xf32>, f32) {',
            '-> (vector<4xf32>, f32) attributes {frisk.fragment_lds_prefetch = "2:0"} {')

    def without_anchors(text):
        return run(re.sub(r'frisk\.(?:lds_prefetch_anchor|scheduled_mma|register_mma),? ?', '', text))

    for spacing in (128, 4096):
        source = run(outstanding(spacing))
        result = run(source, FINALIZE)
        assert thread_graph(without_anchors(source)) == thread_graph(without_anchors(result))
        assert result.count('frisk.lds_pair') == 8
        assert result.count('frisk.sched_barrier') == 6
        assert result.count('mask = 1028 : i32') == 6
        assert 'frisk.lds_prefetch_anchor' not in result
        assert run(result, FINALIZE) == result
        lines = result.splitlines()
        for tag in range(4):
            positions = [n for n, line in enumerate(lines)
                         if f'frisk.lds_pair = {tag} :' in line]
            assert len(positions) == 2
            assert not any('frisk.sched_barrier' in line or 'v_mmac_' in line
                           for line in lines[positions[0]:positions[1]])
        # Use the production translation extension, then the legacy contract
        # completion entry point, and O3. The scheduled MMA must remain inside
        # its compiler guard bracket at every stage (a pure call can escape).
        low = run(result, '--test-async-to-llvm')
        direct = subprocess.run([str(TOOL.parent/'FriskSemanticsTest'), '-', '--llvm'],
            input=low, text=True, capture_output=True, check=True).stdout
        prepared = subprocess.run([str(TOOL.parent/'LegacyLLVMExportTest'),
            '--export-contracts-only', '-'], input=direct, text=True,
            capture_output=True, check=True).stdout
        optimized = subprocess.run(['/data2/xsl/install/bin/opt', '-S',
            '-passes=default<O3>,verify', '-o', '-'], input=prepared, text=True,
            capture_output=True, check=True).stdout
        for ir in (direct, prepared, optimized):
            assert 'memory(inaccessiblemem: readwrite)' in ir
            export_events = [line for line in ir.splitlines() if 'v_mmac_' in line or
                      'call void @llvm.amdgcn.sched.barrier' in line]
            anchored = [i for i, line in enumerate(export_events) if '!frisk.scheduled_mma' in line]
            assert len(anchored) == 3
            for i in anchored:
                assert 'asm sideeffect' in export_events[i]
                assert i > 0 and 'sched.barrier' in export_events[i-1]
                assert i+1 < len(export_events) and 'sched.barrier' in export_events[i+1]

    consumed = paired().replace('-> (vector<4xf32>, f32) {',
        '-> (vector<4xf32>, f32) attributes {frisk.fragment_lds_prefetch = "2:0"} {')
    assert 'frisk.lds_pair' not in run(consumed, FINALIZE)
    for boundary in ('gpu.barrier', 'func.call @unknown() : () -> ()',
                     'memref.store %zero, %dst[%i] : memref<1xf32, 3>'):
        source = outstanding(boundary=boundary).replace(
            '%other: index)', '%other: index, %dst: memref<1xf32, 3>)')
        result = run(source, FINALIZE)
        assert result.count('frisk.lds_pair') == 2
    print('PASS: fragment read-ahead distances, SSA, memory boundaries, loop unrolling, replay, lowering and physical pairs')


if __name__ == '__main__':
    main()
