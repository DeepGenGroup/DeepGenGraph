#!/usr/bin/env python3
"""Compile split-D attention and execute its normalized Frisk IR on the CPU.

Usage: OPENBLAS_NUM_THREADS=1 python3 check_attention_split_d.py path/to/MyTest
Requires NumPy. This checks the actual lowered addresses, loop-carried sums,
causal mask and f16/f32 arithmetic, but does not execute GPU machine code.
"""
import argparse
import ast
import operator
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from generate_attention_ir import TEMPLATE, generate_ir

SSA = r"%[\w]+(?:#\d+)?"


def expr(text, env):
    """Evaluate only the integer arithmetic subset printed by affine maps."""
    text = re.sub(SSA, lambda m: str(int(env[m[0]])), text)
    text = re.sub(r"\b[ds]\d+\b", lambda m: str(int(env[m[0]])), text)
    text = text.replace("floordiv", "//").replace("mod", "%")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        raise AssertionError(ast.dump(node))

    return visit(ast.parse(text.strip(), mode="eval").body)


def affine_map(text, values, env):
    match = re.search(r"affine_map<(.*?) -> \((.*?)\)>", text)
    names = re.findall(r"[ds]\d+", match[1])
    local = {**env, **dict(zip(names, values))}
    return [expr(item, local) for item in match[2].split(",")]


def parse_region(lines, start=0):
    nodes = []
    i = start
    while i < len(lines) and not lines[i].startswith("}"):
        header = lines[i]
        body = alternate = None
        i += 1
        if header.endswith("{"):
            body, i = parse_region(lines, i)
            if lines[i].startswith("} else {"):
                alternate, i = parse_region(lines, i + 1)
            # Include result types printed after mask's closing brace.
            header += " " + lines[i]
            i += 1
        nodes.append((header, body, alternate))
    return nodes, i


def result_type(text):
    types = re.findall(r"(?:memref|vector)<([\dx]+)x(f16|f32)", text)
    if types:
        shape, dtype = types[-1]
        return tuple(map(int, shape.split("x"))), np.dtype(dtype.replace("f", "float"))
    return (), np.dtype("float16" if text.endswith("f16") else "float32")


def subview(array, origin, shape):
    prefix = len(origin) - len(shape)
    for axis, value in enumerate(origin):
        extent = 1 if axis < prefix else shape[axis - prefix]
        assert 0 <= value and value + extent <= array.shape[axis], (origin, shape, array.shape)
    return array[tuple(origin[:prefix]) + tuple(
        slice(value, value + size) for value, size in zip(origin[prefix:], shape))]


def execute(nodes, env, grid):
    for header, body, alternate in nodes:
        if header.startswith("^bb0"):
            continue
        assign = re.match(r"(" + SSA + r")(?:\:\d+)? = (.*)", header)
        target, op = (assign[1], assign[2]) if assign else (None, header)
        args = re.findall(SSA, op.split(" : ")[0])
        value = None
        if op.startswith("affine.for"):
            match = re.match(r"affine.for (" + SSA + r") = (.*?) to (.*?) step (\d+)", op)
            assert match, op
            iv, lower, upper, step = match.groups()
            if "affine_map" in upper:
                operands = [env[n] for n in re.findall(SSA, upper)]
                upper = affine_map(upper, operands, env)[0]
            else:
                upper = expr(upper, env)
            carriers = re.search(r"iter_args\((.*?)\)", op)
            pairs = re.findall(r"(" + SSA + r") = (" + SSA + r")", carriers[1]) if carriers else []
            values = tuple(np.copy(env[initial]) for _, initial in pairs)
            for index in range(expr(lower, env), upper, int(step)):
                local = {**env, iv: index, **dict(zip((p[0] for p in pairs), values))}
                yielded = execute(body, local, grid)
                if pairs:
                    values = yielded
            if target:
                if len(values) == 1:
                    env[target] = values[0]
                else:
                    env.update({f"{target}#{i}": v for i, v in enumerate(values)})
            continue
        if op.startswith(("affine.yield", "scf.yield", "frisk.mask_yield")):
            return tuple(env[n] for n in args)
        if op == "return":
            return
        if op.startswith("gpu."):
            axis = re.search(r"(?:block_id|thread_id)\s+([xyz])", op)[1]
            value = grid[axis] if "block_id" in op else 0
        elif op.startswith("arith.constant"):
            literal = op.split()[1]
            if literal.startswith("dense<"):
                shape, dtype = result_type(op)
                value = np.full(shape, float(literal[6:-1]), dtype=dtype)
            else:
                value = -np.inf if literal == "0xFF800000" else float(literal)
                value = int(value) if ": index" in op else np.float32(value)
        elif op.startswith("arith.addi"):
            value = env[args[0]] + env[args[1]]
        elif op.startswith("arith.muli"):
            value = env[args[0]] * env[args[1]]
        elif op.startswith("arith.cmpi ule"):
            value = env[args[0]] <= env[args[1]]
        elif op.startswith("arith.select"):
            value = np.where(env[args[0]], env[args[1]], env[args[2]])
        elif op.startswith("scf.if"):
            value = np.where(env[args[0]], execute(body, dict(env), grid)[0],
                             execute(alternate, dict(env), grid)[0])
        elif op.startswith("memref.transpose"):
            perm = re.search(r"-> \((.*?)\)", op)[1]
            value = env[args[0]].transpose(tuple(map(int, re.findall(r"d(\d+)", perm))))
        elif op.startswith(("frisk.zero", "frisk.alloc_buffer")):
            shape, dtype = result_type(op)
            value = np.zeros(shape, dtype=dtype) if op.startswith("frisk.zero") else np.full(shape, np.nan, dtype=dtype)
        elif op.startswith("frisk.buffer_view"):
            origin = re.search(r"\[(.*?)\], ranges", op)[1]
            shape, _ = result_type(op)
            value = subview(env[args[0]], [expr(x, env) for x in origin.split(",")], shape)
        elif op.startswith("frisk.copy"):
            src, dst = (env[n] for n in args[:2])
            if " to " in op and " at " in op:
                positions = [env[n] for n in args[2:]]
                dst = subview(dst, affine_map(op, positions, env), src.shape)
            np.copyto(dst, src, casting="unsafe")
            continue
        elif op.startswith("frisk.mask starts"):
            starts = re.search(r"starts = \[(.*?)\]", op)[1]
            row, col = [env[n] for n in re.findall(SSA, starts)]
            shape, _ = result_type(op)
            block_args = re.findall(SSA, body[0][0])
            local = {**env, block_args[0]: row + np.arange(shape[0])[:, None],
                     block_args[1]: col + np.arange(shape[1])[None, :]}
            value = np.broadcast_to(execute(body, local, grid)[0], shape).astype(np.float32)
        elif op.startswith("frisk.reduce"):
            axis = int(re.search(r"dim = (\d+)", op)[1])
            kind = re.search(r'kind = "(\w+)"', op)[1]
            if kind == 'add':
                value = env[args[0]].sum(axis=axis, keepdims=True, dtype=np.float32)
            elif kind == 'max':
                value = env[args[0]].max(axis=axis, keepdims=True)
            else:
                raise AssertionError(f'Unsupported reduction: {kind}')
            env[args[1]][...] = value
            continue
        elif op.startswith("frisk.gemm"):
            assert "transA = false, transB = false" in op
            value = env[args[0]].astype(np.float32) @ env[args[1]].astype(np.float32)
        elif op.startswith("frisk.exp2"):
            value = np.exp2(env[args[0]])
        elif op.startswith("frisk.cast"):
            value = np.asarray(env[args[0]]).astype(result_type(op)[1])
        elif op.startswith(("frisk.add", "frisk.sub", "frisk.max", "frisk.mul", "frisk.div")):
            fn = {"frisk.add": np.add, "frisk.sub": np.subtract, "frisk.max": np.fmax,
                  "frisk.mul": np.multiply, "frisk.div": np.divide}[op.split()[0]]
            value = fn(env[args[0]], env[args[1]]).astype(result_type(op)[1])
        else:
            raise AssertionError(f"Unsupported IR (do not silently skip): {op}")
        assert target is not None, op
        env[target] = value


def check_numerics(log, bm, seed=0, blocks=None):
    marker = "---------- after createNormalizeAttentionKLayoutPass ---------\n"
    module = log.split(marker, 1)[1].split("\n}\n", 1)[0]
    function = module.split("  func.func @Attn_p2", 1)[1]
    lines = [line.strip() for line in function.splitlines()[1:]]
    nodes, _ = parse_region(lines)
    rng = np.random.default_rng(seed)
    # Read the actual launch ABI; nonuniform values expose addressing errors.
    shape = tuple(map(int, re.search(r"memref<(\d+)x(\d+)x(\d+)x(\d+)xf16", function).groups()))
    q, k, v = [rng.normal(size=shape).astype(np.float16) for _ in range(3)]
    output = np.full(shape, np.nan, dtype=np.float16)
    if blocks is None:
        blocks = [(b, h, m) for b in range(shape[0]) for h in range(shape[1])
                  for m in range(shape[2] // bm)]
    max_error = 0.0
    for batch, head, block in blocks:
        execute(nodes, dict(zip(("%arg0", "%arg1", "%arg2", "%arg3"), (q, k, v, output))),
                {"x": head, "y": block, "z": batch})
        start, end = block * bm, (block + 1) * bm
        scores = (q[batch, head, start:end].astype(np.float32)
                  @ k[batch, head, :end].astype(np.float32).T) / np.sqrt(shape[3])
        mask = np.arange(end)[None, :] > np.arange(start, end)[:, None]
        scores[mask] = -np.inf
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        reference = (weights / weights.sum(axis=-1, keepdims=True)) @ v[batch, head, :end].astype(np.float32)
        actual = output[batch, head, start:end]
        assert np.isfinite(actual).all(), "Unwritten or nonfinite output"
        np.testing.assert_allclose(actual, reference, atol=3e-3, rtol=3e-3)
        max_error = max(max_error, float(np.max(np.abs(actual - reference))))
    return max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mytest", type=Path)
    args = parser.parse_args()
    tool = args.mytest.resolve()
    source = TEMPLATE.read_text()
    fixture = TEMPLATE.with_name("test_input_spidD.mlir")
    fixture_text = fixture.read_text()
    fixture_bm, fixture_bn, fixture_bk = (
        int(re.search(rf'%c{name} = arith.constant (\d+)', fixture_text)[1])
        for name in ('BM', 'BN', 'BK'))
    # This fixture deliberately records the historical non-stable kernel.
    assert fixture_text == generate_ir(source, fixture_bm, fixture_bn, fixture_bk, softmax="legacy")[0]
    for bk in (0, -1, 3, 129):
        try:
            generate_ir(source, 64, 32, bk)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid BK accepted: {bk}")
    small = source.replace("4096", "256").replace("1x256x32x", "2x256x3x").replace("1x32x", "2x3x")
    configs = [(64, 32, bk) for bk in (16, 32, 64, 128)] + [(32, 64, 32), (128, 32, 64)]
    with tempfile.TemporaryDirectory(prefix="attention-split-d-") as directory:
        for bm, bn, bk in configs:
            path = Path(directory) / f"attention-{bm}-{bn}-{bk}.mlir"
            ir, _ = generate_ir(small, bm, bn, bk, softmax="legacy")
            assert ir.split("  deepgengraph.kernel", 1)[0] == small.split("  deepgengraph.kernel", 1)[0]
            path.write_text(ir)
            output = path.with_suffix(".ll")
            proc = subprocess.run([str(tool), str(path), "0", str(output)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=180)
            log = proc.stdout
            assert proc.returncode == 0, log[-5000:]
            assert not re.search(r"error:|Assertion .*failed|LLVM ERROR", log), log[-5000:]
            assert "define amdgpu_kernel void @Attn_p2" in output.read_text()
            error = check_numerics(log, bm)
            print(f"PASS BM={bm} BN={bn} BK={bk}: LLVM export, lowered-IR CPU execution; max_abs={error:.6g}", flush=True)
        output = Path(directory) / "full.ll"
        proc = subprocess.run([str(tool), str(fixture), "0", str(output)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=180)
        assert proc.returncode == 0 and not re.search(r"error:|Assertion .*failed|LLVM ERROR", proc.stdout), proc.stdout[-5000:]
        last_tile = 4096 // fixture_bm - 1
        error = check_numerics(proc.stdout, fixture_bm, seed=1,
            blocks=[(0, 0, 0), (0, 13, last_tile // 2), (0, 31, last_tile)])
        print(f"PASS original 1x32x4096x128 shape: first/middle/last tiles; max_abs={error:.6g}")


if __name__ == "__main__":
    main()
