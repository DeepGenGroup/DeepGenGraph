#!/usr/bin/env python3
"""Cross-iteration execution planning, task SSA, circular ownership and fallback."""
from pathlib import Path
import re
import subprocess
import tempfile
from check_v_double_buffer import fixture as pipeline_fixture

TOOLS = Path(__file__).resolve().parents[2] / 'build/test'


def fixture(trips=4, step=1, loop='affine', dynamic=False, stages=4, lower=0):
    text = pipeline_fixture(stages=stages)
    gm = f'memref<{stages}x512xf16, 1>'
    start = 'affine.for %iv = 0 to 4 {'
    prefix = '%one = arith.constant ' + str(step) + ' : index\n'
    prefix += f'%end = arith.constant {trips} : index\n'
    prefix += f'%start = arith.constant {lower} : index\n'
    if dynamic:
        text = text.replace('%out: memref<4xi32, 1>', '%out: memref<4xi32, 1>, %bound: index')
    hi = '%bound' if dynamic else ('%end' if loop == 'scf' else str(trips))
    header = (f'affine.for %iv = {lower} to {hi} step {step} {{' if loop == 'affine' else
              f'scf.for %iv = %start to {hi} step %one {{')
    body = f'''
%kbuf = memref.alloc() alignment = 16 : memref<512xf16, 3>
%ksrc = memref.subview %gm[%iv, 0] [1, 512] [1, 1] : {gm} to memref<512xf16, strided<[1], offset: ?>, 1>
%ktask = frisk.async_copy_gm_to_shm %ksrc to %kbuf {{alignment = 16 : i64, frisk.dma_bytes = 16 : i64}} : memref<512xf16, strided<[1], offset: ?>, 1>, memref<512xf16, 3> -> !frisk.async_task
frisk.wait_buffer_ready %ktask : !frisk.async_task
%k = memref.load %kbuf[%idx] : memref<512xf16, 3>
gpu.barrier
'''
    text = text.replace(start, prefix + header + body)
    # Global writes in the loop would require an independent alias proof.
    text = re.sub(r'vector.store %read\d+,.*\n', '', text)
    # Physical fixtures explicitly supply the provisional early contract.
    # Candidate discovery belongs to AutoPipeline, not this consumer test.
    text = text.replace('%ktask = frisk.async_copy_gm_to_shm %ksrc to %kbuf {',
                        '%ktask = frisk.async_copy_gm_to_shm %ksrc to %kbuf {frisk.cross_iteration_issue, ')
    text = re.sub(r'(%task' + str(stages - 1) + r' = frisk.async_copy_gm_to_shm [^\n]*?\{)',
                  r'\1frisk.cross_iteration_anchor, ', text)
    text = text.replace('}\nreturn', '} {frisk.cross_iteration_plan}\nreturn')
    return text


def run(path, text, enabled=True, extra=()):
    path.write_text(text)
    command = [str(TOOLS/'ThreadTilingTest'), str(path),
               '--lower-frisk-fragments=double-buffer=true cross-iteration=' + str(enabled).lower(),
               *extra, '--verify-each']
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout, result.stderr


def simulate(text, bound=None):
    """Execute the emitted fixture's structured control flow and task SSA.

    DMA remains pending until its named wait. Reads protect the whole logical
    slice until a collective handoff. Source row labels distinguish iterations.
    This deliberately models stricter completion than vmcnt(0), not ISA timing.
    """
    lines = text.splitlines()
    maps = dict(re.findall(r'(#\w+) = affine_map<.*?-> \((.*?)\)>', text))
    first = next(i for i, line in enumerate(lines) if 'func.func @kernel' in line)
    env = {'%arg2': bound} if bound is not None else {}
    capacity = int(re.search(r'memref<(\d+)xi8, 3>', text)[1])
    memory, pending, readers = [None] * capacity, {}, []
    issued, k_reads, v_reads = [], [], []
    serial = 0

    def parse(i):
        nodes = []
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('}'):
                return nodes, i
            if re.search(r'= (scf.if|affine.for|scf.for) ', line):
                body, end = parse(i + 1)
                otherwise = []
                if '= scf.if ' in line:
                    assert lines[end].strip().startswith('} else {')
                    otherwise, end = parse(end + 1)
                nodes.append((line, body, otherwise))
                i = end + 1
            else:
                nodes.append((line, [], []))
                i += 1
        return nodes, i

    def value(name, local):
        return local[name] if name.startswith('%') else int(name)

    def overlap(a, b):
        return a < b + 1024 and b < a + 1024

    def execute(nodes, local, iteration=None):
        nonlocal serial
        matrix = 0
        for line, body, otherwise in nodes:
            if m := re.match(r'(%\w+) = scf.if (%\w+)', line):
                local[m[1]] = execute(body if local[m[2]] else otherwise, local.copy(), iteration)[0]
            elif m := re.match(r'(%\w+) = (?:affine|scf).for (%\w+) = (\S+) to (\S+)(?: step (\S+))? iter_args\((%\w+) = (%\w+)\)', line):
                lo, hi = value(m[3], local), value(m[4], local)
                step = value(m[5], local) if m[5] else 1
                task = local[m[7]]
                for iv in range(lo, hi, step):
                    inner = local.copy()
                    inner[m[2]], inner[m[6]] = iv, task
                    task = execute(body, inner, iv)[0]
                local[m[1]] = task
            elif m := re.match(r'(?:scf|affine).yield (%\w+)', line):
                return [local[m[1]]]
            elif m := re.match(r'(%\w+) = arith.constant (-?\d+) : index', line):
                local[m[1]] = int(m[2])
            elif m := re.match(r'(%\w+) = gpu.thread_id', line):
                local[m[1]] = 0  # Reads below conservatively reserve the whole tile.
            elif m := re.match(r'(%\w+) = affine.(?:apply|max|min) (#\w+)(.*)', line):
                operands = re.findall(r'%\w+', m[3])
                expression = maps[m[2]]
                for n, operand in enumerate(operands):
                    expression = re.sub(r'\b[ds]' + str(n) + r'\b', str(local[operand]), expression)
                assert re.fullmatch(r'[0-9 *+()\-]+', expression), expression
                local[m[1]] = eval(expression, {'__builtins__': {}})
            elif m := re.match(r'(%\w+) = arith.(addi|subi) (%\w+), (%\w+)', line):
                a, b = local[m[3]], local[m[4]]
                local[m[1]] = a + b if m[2] == 'addi' else a - b
            elif m := re.match(r'(%\w+) = arith.cmpi (slt|ugt), (%\w+), (%\w+)', line):
                a, b = local[m[3]], local[m[4]]
                local[m[1]] = a < b if m[2] == 'slt' else a > b
            elif m := re.match(r'(%\w+) = memref.view %\w+\[(%\w+)\]', line):
                local[m[1]] = local[m[2]]
            elif m := re.match(r'(%\w+) = memref.subview %\w+\[([^,]+), 0\]', line):
                local[m[1]] = value(m[2], local)
            elif m := re.match(r'(%\w+) = frisk.async_task_completed', line):
                local[m[1]] = None
            elif m := re.match(r'(%\w+) = frisk.async_copy_gm_to_shm (%\w+) to (%\w+)', line):
                offset, source = local[m[3]], local[m[2]]
                assert 0 <= offset <= capacity - 1024
                assert not any(overlap(offset, other) for other in readers), 'overwrite before reader handoff'
                assert not any(overlap(offset, item[0]) for item in pending.values()), 'overlapping pending DMA'
                serial += 1
                local[m[1]] = serial
                pending[serial] = offset, source
                if 'frisk.cross_bn_k0' in line:
                    issued.append(source)
            elif m := re.match(r'frisk.wait_buffer_ready (%\w+)', line):
                task = local[m[1]]
                if task is not None:
                    offset, source = pending.pop(task)
                    memory[offset:offset+1024] = [source] * 1024
                readers.clear()
            elif m := re.match(r'%\w+ = (memref.load|frisk.read_matrix_from_shm) (%\w+)\[', line):
                offset = local[m[2]]
                assert not any(overlap(offset, item[0]) for item in pending.values()), 'read before completion'
                expected = 0 if iteration is None else iteration
                if m[1] == 'frisk.read_matrix_from_shm':
                    expected = matrix
                    matrix += 1
                    v_reads.append((iteration, expected))
                elif iteration is not None:
                    k_reads.append(iteration)
                assert memory[offset:offset+1024] == [expected] * 1024, 'wrong tile or physical slot'
                readers.append(offset)
            elif line == 'gpu.barrier':
                readers.clear()
            else:
                assert line == 'return' or ' = memref.alloc()' in line, line
        return []

    nodes, _ = parse(first + 1)
    execute(nodes, env)
    assert not pending, 'undrained task on loop exit'
    assert issued == k_reads, 'missing, duplicate or speculative next-iteration copy'
    assert len(v_reads) == len(k_reads) * 4


def main():
    count = 0
    with tempfile.TemporaryDirectory(prefix='shared-cross-iteration-') as directory:
        path = Path(directory)/'input.mlir'
        base = fixture()
        path.write_text(base)
        unit = subprocess.run([str(TOOLS/'SharedExecutionPlanTest'), str(path), 'cross-iteration'],
                              text=True, capture_output=True)
        assert unit.returncode == 0, unit.stdout + unit.stderr
        print(unit.stdout.strip())
        # Early lazy-softmax introduces a register-only branch before storage
        # planning. It must not block DMA motion across pure computations.
        registers = '''
%choose = arith.cmpi eq, %iv, %idx : index
%selected = scf.if %choose -> i32 {
  %yes = arith.constant 1 : i32
  scf.yield %yes : i32
} else {
  %no = arith.constant 2 : i32
  scf.yield %no : i32
}
'''
        with_registers = base.replace('%task0 =', registers + '\n%task0 =')
        path.write_text(with_registers)
        unit = subprocess.run([str(TOOLS/'SharedExecutionPlanTest'), str(path), 'cross-iteration'],
                              text=True, capture_output=True)
        assert unit.returncode == 0, unit.stdout + unit.stderr
        _, diagnostic = run(path, with_registers)
        assert 'Cross BN K0 plan: buffer=' in diagnostic, diagnostic
        with_vote = with_registers.replace('%selected =',
            '%vote = rocdl.ballot %choose : i64\n%selected =')
        _, diagnostic = run(path, with_vote)
        assert 'Cross BN K0 plan: buffer=' in diagnostic, diagnostic
        count += 1
        for loop in ('affine', 'scf'):
            for trips, step, dynamic, lower in ((0,1,False,0), (1,1,False,0), (4,1,False,0),
                                               (4,2,False,0), (4,3,False,0), (4,1,True,0),
                                               (4,1,False,2), (1,1,False,2)):
                source = fixture(trips, step, loop, dynamic, lower=lower)
                out, log = run(path, source)
                assert 'Cross BN K0 plan: buffer=' in log, log
                assert 'LDS=2048' in log
                assert out.count('scf.if') == 2, out
                assert 'iter_args' in out and '!frisk.async_task' in out
                assert 'arith.cmpi slt' in out and 'arith.cmpi ugt' in out
                assert out.count('frisk.async_copy_gm_to_shm') == source.count('frisk.async_copy_gm_to_shm') + 1
                for bound in ((0, 1, 3, 4) if dynamic else (None,)):
                    simulate(out, bound)
                again, _ = run(path, out)
                assert out == again, 'repeated execution planning changes a materialized schedule'
                for lowering in ('--frisk-lower-async-copy', '--frisk-lower-async-to-sync'):
                    lowered, _ = run(path, out, extra=(lowering,))
                    assert '!frisk.async_task' not in lowered
                if dynamic:
                    lowered, _ = run(path, out, extra=('--test-async-to-llvm',))
                    assert '!frisk.async_task' not in lowered and 'llvm.func' in lowered
                count += 1
        out, _ = run(path, base)
        scheduled = base.replace('{frisk.cross_iteration_plan}',
                                 '{frisk.cross_iteration_plan, pipeline.scheduled}')
        assert scheduled != base
        preserved, diagnostic = run(path, scheduled)
        baseline, _ = run(path, scheduled, False)
        assert preserved == baseline
        assert 'existing pipeline schedule owns task/slot lifetimes' in diagnostic, diagnostic
        last_slot = re.findall(r'frisk.read_matrix_from_shm (%\w+)\[', out)[-1]
        broken = re.sub(r'(frisk.async_copy_gm_to_shm %\w+ to )%\w+( \{[^\n]*frisk.cross_bn_k0)',
                        lambda m: m[1] + last_slot + m[2], out)
        try:
            simulate(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('ownership oracle missed a next-K/final-V slot collision')
        rejected = [
            with_registers.replace('%yes =', '%vote = rocdl.ballot %choose : i64\n%yes ='),
            with_registers.replace('%yes =', 'gpu.barrier\n%yes ='),
            with_registers.replace('%yes =',
                '%conditional_read = memref.load %kbuf[%idx] : memref<512xf16, 3>\n%yes ='),
            base.replace('frisk.cross_iteration_plan', 'test.no_cross_plan'),
            base.replace('frisk.cross_iteration_issue', 'test.no_cross_issue'),
            base.replace('frisk.cross_iteration_anchor', 'test.no_cross_anchor'),
            base.replace('frisk.cross_iteration_issue,', 'frisk.cross_iteration_issue = 1 : i64,'),
            base.replace('affine.for %iv = 0 to 4 step 1', 'affine.for %iv = 0 to %tid'),
            base.replace('%gm[%iv, 0]', '%gm[%c0, 0]'),
            re.sub(r'(%read3 = [^\n]+)\n',
                   r'\1\n%late = memref.load %kbuf[%idx] : memref<512xf16, 3>\n',
                   base.replace('%k = memref.load %kbuf[%idx] : memref<512xf16, 3>', '')),
            base.replace('%read3 =', 'memref.store %k, %warm0[%idx] : memref<512xf16, 3>\n%read3 ='),
            base.replace('func.func @kernel', 'func.func private @unknown()\nfunc.func @kernel').replace(
                '%read3 =', 'func.call @unknown() : () -> ()\n%read3 ='),
            # Last V stage shares slot 0 in an odd-length alternating stream;
            # keeping that schedule would need a third slot for next K0.
            fixture(stages=3),
        ]
        for source in rejected:
            out, log = run(path, source)
            assert 'Cross BN K0 plan: buffer=' not in log, log
            baseline, _ = run(path, source, False)
            assert out == baseline, 'rejected cross-iteration plan changed baseline IR'
            count += 1
    print(f'PASS {count} cross-iteration cases: ownership simulation, guards, steps, dynamic bounds, task lowering and rejection')


if __name__ == '__main__':
    main()
