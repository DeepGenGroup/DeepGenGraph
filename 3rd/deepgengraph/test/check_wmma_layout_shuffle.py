#!/usr/bin/env python3
"""Check planning, emitted XOR dataflow, fallbacks, and attention integration.

Usage: check_wmma_layout_shuffle.py build/test/ThreadTilingTest
The scalar interpreter executes the generated shuffle/select operands for every
lane; input values encode logical matrix coordinates, rather than uniform data.
This is a compiler regression, not a GPU numerical/performance benchmark.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
SSA = r"%[\w]+"


def run(directory, name, source, *passes):
    src, out = directory / (name + ".mlir"), directory / (name + ".out.mlir")
    src.write_text(source)
    proc = subprocess.run([str(TOOL), str(src), *passes, "--verify-each", "-o", str(out)],
                          capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, name + "\n" + proc.stderr[-6000:]
    return out.read_text(), proc.stdout


def gemm(m, n, k, threads):
    a, b, c = f"memref<{m}x{k}xf16, 3>", f"memref<{k}x{n}xf16, 3>", f"memref<{m}x{n}xf32>"
    return f'''func.func @gemm(%a: {a}, %b: {b}) -> {c} attributes {{thread_num = {threads} : i32}} {{
      %g = frisk.gemm(%a, %b) {{transA = false, transB = false}} : {a}, {b} -> {c}
      return %g : {c}
    }}'''


def execute_shuffle(ir, threads=256):
    # Take the compute region containing the layout shuffles, and identify its
    # input fragment from the actual vector.extract operands.
    start = ir.rfind('frisk.fragment "compute"', 0, ir.index("gpu.shuffle"))
    body = ir[start:ir.index("frisk.fragment_yield", start)]
    extracts = re.findall(rf"({SSA}) = vector.extract ({SSA})\[0, ([0-3])\]", body)
    inputs = extracts[-4:]
    assert [int(x[2]) for x in inputs] == list(range(4))
    assert len({x[1] for x in inputs}) == 1
    fragment = inputs[0][1]
    definitions = dict(re.findall(rf"^\s*({SSA}) = (.+)$", ir[:start], re.M))
    definitions.update(re.findall(rf"^\s*({SSA}) = (.+)$", body, re.M))
    for result, value, offset, width in re.findall(
            rf"({SSA}), {SSA} = gpu.shuffle\s+xor ({SSA}), ({SSA}), ({SSA})", body):
        definitions[result] = f"shuffle {value} {offset} {width}"
    # Values in the raw C fragment: C[lane%16, lane//16 + 4*reg].
    # Distinct wave/fragment bases catch accidental wave crossing or reuse.
    for base in (0, 100000):
        cache = {}

        def value(name, tid):
            key = name, tid
            if key in cache:
                return cache[key]
            if name == fragment:
                return [base + (tid // 64) * 256 + (tid % 16) * 16 +
                        (tid % 64) // 16 + 4 * reg for reg in range(4)]
            expr = definitions[name]
            operands = re.findall(SSA, expr)
            if expr.startswith("gpu.thread_id"):
                result = tid
            elif expr.startswith("arith.constant"):
                result = int(re.search(r"constant (-?\d+)", expr)[1])
            elif expr.startswith("arith.index_cast"):
                result = value(operands[0], tid)
            elif expr.startswith("arith.andi"):
                result = value(operands[0], tid) & value(operands[1], tid)
            elif expr.startswith("arith.cmpi ne"):
                result = value(operands[0], tid) != value(operands[1], tid)
            elif expr.startswith("arith.select"):
                result = value(operands[1] if value(operands[0], tid) else operands[2], tid)
            elif expr.startswith("vector.extract"):
                reg = int(re.search(r"\[0, (\d+)\]", expr)[1])
                result = value(operands[0], tid)[reg]
            elif expr.startswith("vector.insert"):
                reg = int(re.search(r"\[0, (\d+)\]", expr)[1])
                result = list(value(operands[1], tid))
                result[reg] = value(operands[0], tid)
            elif expr.startswith("shuffle"):
                src, offset, width = operands
                width = value(width, tid)
                peer = tid // width * width + ((tid % width) ^ value(offset, tid))
                result = value(src, peer)
            else:
                raise AssertionError((name, expr))
            cache[key] = result
            return result

        result_name = re.findall(rf"({SSA}) = vector.insert", body)[-1]
        for tid in range(threads):
            expected = [base + (tid // 64) * 256 + (tid % 16) * 16 +
                        4 * ((tid % 64) // 16) + reg for reg in range(4)]
            assert value(result_name, tid) == expected, (tid, value(result_name, tid), expected)


def planner_fixture(log):
    line = next(line for line in log.splitlines() if 'frisk.shuffle_plan' in line)
    attrs = line[line.index("{"):line.rindex("}") + 1]
    # The standalone planner must derive the plan, rather than trust the dump.
    attrs = re.sub(r"frisk.shuffle_plan = \{[^}]+\}, ", "", attrs)
    return f'''func.func @plan(%x: memref<16x16xf32>) -> memref<16x16xf32> {{
      %y = frisk.convert_layout %x {attrs} : memref<16x16xf32> -> memref<16x16xf32>
      return %y : memref<16x16xf32>
    }}'''


def main():
    with tempfile.TemporaryDirectory(prefix="wmma-layout-") as tmp:
        directory = Path(tmp)
        small, log = run(directory, "small", gemm(16, 16, 16, 64), "--convert-friskbase-to-thread")
        assert small.count("gpu.shuffle") == 4
        assert "memref.alloc" not in small and "gpu.barrier" not in small
        assert "frisk.sync_threads_in_block" not in small
        assert 'xor_masks = array<i64: 16, 32>' in log
        execute_shuffle(small)
        print("PASS emitted dataflow: 256 threads, 4 registers, distinct matrix coordinates")

        fixture = planner_fixture(log)
        planned, _ = run(directory, "plan", fixture, "--plan-wmma-layout-conversions")
        assert "frisk.shuffle_plan" in planned
        again, _ = run(directory, "plan-twice", planned, "--plan-wmma-layout-conversions")
        assert planned == again
        # Instruction K and operand dtype do not determine C's lane mapping.
        other_inst = fixture.replace("16x16x16_f16", "16x16x4_f32").replace(
            "shape = array<i64: 16, 16, 16>", "shape = array<i64: 16, 16, 4>")
        other_plan, _ = run(directory, "other-inst", other_inst,
                            "--plan-wmma-layout-conversions")
        assert "xor_masks = array<i64: 16, 32>" in other_plan
        # Feed stale plans, and ensure unsupported metadata removes them.
        for name, bad in {
            "role": planned.replace('"wmmaC"', '"wmmaA"'),
            "wave32": planned.replace("warp_threads = 64", "warp_threads = 32"),
            "type": planned.replace("xf32", "xf16"),
            "missing-inst": planned.replace("frisk.wmma_instruction", "test.unknown"),
            "cross-wave": planned.replace(
                "frisk.target_layout = {block_layout = array<i64: 1, 1>",
                "frisk.target_layout = {block_layout = array<i64: 1, 2>"),
            "wrong-order": planned.replace("warp_layout_order = array<i64: 0, 1>",
                                            "warp_layout_order = array<i64: 1, 0>"),
            "incomplete": planned.replace("16x16xf32", "15x16xf32"),
        }.items():
            rejected, _ = run(directory, name, bad, "--plan-wmma-layout-conversions")
            assert "frisk.shuffle_plan" not in rejected, name
        print("PASS standalone planning, idempotence and unsupported/stale metadata")

        # Explicitly request contiguous C, but disable shuffle lowering: this
        # exercises the same semantic conversion through the shared fallback.
        fallback, _ = run(directory, "fallback", gemm(16, 16, 16, 64),
                          "--plan-wmma-layout-conversions",
                          "--convert-friskbase-to-thread=wmma-shuffle=false")
        assert "gpu.shuffle" not in fallback and "memref.alloc" in fallback
        assert "gpu.barrier" in fallback or "frisk.sync_threads_in_block" in fallback
        for m, n, k, threads in [(64, 32, 128, 64), (64, 128, 32, 128), (64, 128, 32, 256)]:
            ir, _ = run(directory, f"tile-{threads}", gemm(m, n, k, threads),
                        "--convert-friskbase-to-thread")
            assert ir.count("gpu.shuffle") == 4 and "memref.alloc" not in ir
            twice, _ = run(directory, f"twice-{threads}", ir, "--convert-friskbase-to-thread")
            normalized, _ = run(directory, f"normalized-{threads}", ir)
            assert normalized == twice
        print("PASS shared fallback, multiple tile shapes/waves and repeated tiling")

        # A row scale commutes with C's column-lane permutation. Column scales
        # do not: retain the in-loop conversion unless their mapping is proved.
        for reverse, factor_shape in [(False, '32x1'), (True, '32x1'), (False, '1x32')]:
            lhs, rhs = ('%alpha', '%acc') if reverse else ('%acc', '%alpha')
            lhs_ty, rhs_ty = (f'memref<{factor_shape}xf32>', 'vector<32x32xf32>') if reverse else (
                'vector<32x32xf32>', f'memref<{factor_shape}xf32>')
            source = f'''func.func @scaled(%a: memref<32x32xf16, 3>, %b: memref<32x32xf16, 3>, %alpha: memref<{factor_shape}xf32>) -> vector<32x32xf32> attributes {{thread_num = 64 : i32}} {{
  %init = arith.constant dense<0.0> : vector<32x32xf32>
  %out = affine.for %i = 0 to 3 iter_args(%acc = %init) -> vector<32x32xf32> {{
    %g = frisk.gemm(%a, %b) {{transA = false, transB = false}} : memref<32x32xf16, 3>, memref<32x32xf16, 3> -> memref<32x32xf32>
    %scaled = frisk.mul {lhs}, {rhs} : {lhs_ty}, {rhs_ty} -> memref<32x32xf32>
    %next = frisk.add %scaled, %g : memref<32x32xf32>, memref<32x32xf32> -> memref<32x32xf32>
    frisk.copy %next to %acc [affine_map<() -> (2)>] : memref<32x32xf32>, vector<32x32xf32>
    affine.yield %acc : vector<32x32xf32>
  }}
  return %out : vector<32x32xf32>
}}'''
            _, log = run(directory, f'scaled-{reverse}-{factor_shape}', source,
                         '--convert-friskbase-to-thread')
            assert ('"loop_result"' in log) == (factor_shape == '32x1'), log[-6000:]
        print('PASS row-scaled loop layout sinking in both operand orders; column-scale fallback')

        attention, _ = run(directory, "attention", (HERE / "test_friskBaseDebug.mlir").read_text(),
                           "--convert-friskbase-to-thread", "--finalize-thread-tiling",
                           "--frisk-frag-ir-reorder", "--lower-frisk-fragments")
        assert 'frisk.layout_role = "wmmaC"' in attention
        assert not re.search(r"frisk\.(?:to_threadTile|from_threadTile|convert_layout)\b", attention)
        print("PASS full attention tiling, finalization and fragment scheduling")


if __name__ == "__main__":
    main()
