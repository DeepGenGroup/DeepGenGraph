#!/usr/bin/env python3
"""Check shared tensor attention, lowering hints and CPU attention semantics."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from generate_attention_ir import TEMPLATE, generate_ir
from check_attention_split_d import SSA, expr, parse_region


def tensor_type(op):
    shape, dtype = re.findall(r"tensor<([\dx]+)x(f16|f32)>", op)[-1]
    return tuple(map(int, shape.split("x"))), np.dtype(dtype.replace("f", "float"))


def execute_tensor(nodes, env):
    """Execute the emitted tensor operations, rejecting unsupported operations."""
    for header, body, alternate in nodes:
        if header.startswith("^bb0"):
            continue
        assignment = re.match(r"(" + SSA + r")(?:\:\d+)? = (.*)", header)
        target, op = (assignment[1], assignment[2]) if assignment else (None, header)
        args = re.findall(SSA, op.split(" : ")[0])
        if op.startswith("scf.for"):
            iv, lower, upper, step = re.match(
                r"scf.for (" + SSA + r") = (.*?) to (.*?) step (" + SSA + r")", op).groups()
            pairs = re.findall(r"(" + SSA + r") = (" + SSA + r")",
                               re.search(r"iter_args\((.*?)\)", op)[1])
            values = tuple(np.copy(env[initial]) for _, initial in pairs)
            for index in range(expr(lower, env), expr(upper, env), expr(step, env)):
                values = execute_tensor(body, {**env, iv: index,
                    **dict(zip((name for name, _ in pairs), values))})
            env.update({f"{target}#{i}": value for i, value in enumerate(values)})
            continue
        if op.startswith(("scf.yield", "deepgengraph.mask_yield")):
            return tuple(env[name] for name in args)
        if op.startswith("arith.constant"):
            literal = op.split()[1]
            if literal.startswith("dense<"):
                shape, dtype = tensor_type(op)
                scalar = -np.inf if literal[6:-1] == "0xFF800000" else float(literal[6:-1])
                value = np.full(shape, scalar, dtype=dtype)
            else:
                value = -np.inf if literal == "0xFF800000" else float(literal)
                value = int(value) if ": index" in op else np.float32(value)
        elif op.startswith(("arith.addi", "arith.muli")):
            lhs, rhs = (env[name] for name in args[:2])
            value = lhs + rhs if op.startswith("arith.addi") else lhs * rhs
        elif op.startswith("arith.cmpi ule"):
            value = env[args[0]] <= env[args[1]]
        elif op.startswith("scf.if"):
            value = np.where(env[args[0]], execute_tensor(body, dict(env))[0],
                             execute_tensor(alternate, dict(env))[0])
        elif op.startswith("deepgengraph_triton.block_ptr_of"):
            array, origin = env[args[0]], int(env[args[1]])
            # The generator's offsets refer to permuted BHSD/BHDS arguments.
            if args[0] == "%argK":
                array = array.transpose(0, 1, 3, 2)
            batch, head, row, col = np.unravel_index(origin, array.shape)
            shape, _ = tensor_type(op)
            value = array[batch, head, row:row + shape[0], col:col + shape[1]]
            assert value.shape == shape, (value.shape, shape, origin)
        elif op.startswith("deepgengraph_triton.block_load"):
            value = np.copy(env[args[0]])
        elif op.startswith("deepgengraph_triton.block_store"):
            np.copyto(env[args[0]], env[args[1]])
            continue
        elif op.startswith("deepgengraph.zero"):
            shape, dtype = tensor_type(op)
            value = np.zeros(shape, dtype=dtype)
        elif op.startswith("deepgengraph.mask"):
            row, col = (env[name] for name in args[:2])
            shape, dtype = tensor_type(op)
            block_args = re.findall(SSA, body[0][0])
            local = {**env, block_args[0]: row + np.arange(shape[0])[:, None],
                     block_args[1]: col + np.arange(shape[1])[None, :]}
            value = np.broadcast_to(execute_tensor(body, local)[0], shape).astype(dtype)
        elif op.startswith("deepgengraph.convert"):
            value = np.asarray(env[args[0]]).astype(tensor_type(op)[1])
        elif op.startswith("deepgengraph.precise_dot_op"):
            value = env[args[0]].astype(np.float32) @ env[args[1]].astype(np.float32)
        elif op.startswith("deepgengraph.reduce"):
            axis = int(re.search(r"dim = (\d+)", op)[1])
            if "op = MAX" in op:
                value = np.maximum(env[args[0]].max(axis=axis, keepdims=True), env[args[1]])
            else:
                value = env[args[0]].sum(axis=axis, keepdims=True, dtype=np.float32) + env[args[1]]
        elif op.startswith("deepgengraph.exp2"):
            value = np.exp2(env[args[0]])
        elif op.startswith(("deepgengraph.add", "deepgengraph.sub", "deepgengraph.mul", "deepgengraph.div")):
            fn = {"deepgengraph.add": np.add, "deepgengraph.sub": np.subtract,
                  "deepgengraph.mul": np.multiply, "deepgengraph.div": np.divide}[op.split()[0]]
            value = fn(env[args[0]], env[args[1]]).astype(tensor_type(op)[1])
        else:
            raise AssertionError(f"Unsupported tensor IR: {op}")
        assert target is not None, op
        env[target] = value


class SharedAttentionGeneratorTests(unittest.TestCase):
    def test_body_has_one_shared_full_d_attention_recurrence(self):
        for softmax in ("online", "legacy"):
            for causal in (True, False):
                for bk in (None, 16, 32, 64):
                    with self.subTest(softmax=softmax, causal=causal, bk=bk):
                        ir, shape = generate_ir(TEMPLATE.read_text(), 32, 16, bk,
                            softmax, b=2, h=3, s=64, d=64, causal=causal)
                        self.assertEqual(shape, (2, 3, 64, 64))
                        kernel = ir.split("  deepgengraph.kernel", 1)[1]
                        self.assertNotRegex(kernel, r"(?:affine|scf)\.for %(?:od|dk)\b")
                        self.assertEqual(len(re.findall(r"(?:affine|scf)\.for", kernel)), 1)
                        self.assertEqual(kernel.count("deepgengraph.precise_dot_op"), 2)
                        self.assertIn("iter_args(%accO = %outZero, %accL = %rowZero", kernel)
                        self.assertIn("tensor<32x64xf32>", kernel)
                        self.assertIn("block_shape = [64, 16]", kernel)
                        self.assertIn("block_shape = [16, 64]", kernel)
                        self.assertEqual(kernel.count("deepgengraph.mask starts"), int(causal))
                        self.assertIn("to %keyEnd" if causal else "to %cS", kernel)
                        self.assertEqual("%accM = %rowNegInf" in kernel, softmax == "online")
                        key_loop = kernel.index("scf.for %n")
                        for name in ("qPtr", "kPtr", "vPtr"):
                            self.assertGreater(kernel.index(f"%{name} ="), key_loop)
                        self.assertIn("%qScaled = deepgengraph.mul %qTile, %scale : (tensor<32x64xf16>, tensor<1xf16>)", kernel)
                        self.assertNotIn("block_advance", kernel)
                        if bk is None:
                            self.assertNotIn("frisk.attention_tiles", kernel)
                        else:
                            self.assertIn(f"frisk.attention_tiles = array<i64: {bk}, {bk}>", kernel)

    def test_hint_rejects_non_hardware_tiles(self):
        for bk in (0, -16, 8, 24, 80):
            with self.subTest(bk=bk), self.assertRaises(ValueError):
                generate_ir(TEMPLATE.read_text(), 32, 16, bk, d=64, s=64)

    def test_hint_changes_no_tensor_computation(self):
        outputs = [generate_ir(TEMPLATE.read_text(), 32, 16, bk, s=64, d=64)[0]
                   for bk in (None, 16, 32, 64)]
        strip_hint = lambda ir: re.sub(r"frisk.attention_tiles = array<i64: \d+, \d+>, ", "", ir)
        self.assertEqual(len(set(map(strip_hint, outputs))), 1)

    def test_generated_tensor_cpu_attention(self):
        rng = np.random.default_rng(7)
        # Unequal S/D and BN > BM expose transposed-K origin mistakes and
        # partial causal tiles hidden by the square baseline configuration.
        for s, d, bm, bn in ((64, 64, 32, 16), (128, 32, 32, 64)):
            q, k, v = [rng.normal(size=(2, 3, s, d)).astype(np.float16) for _ in range(3)]
            for softmax in ("online", "legacy"):
                for causal in (True, False):
                    with self.subTest(s=s, d=d, bm=bm, bn=bn, softmax=softmax, causal=causal):
                        ir, _ = generate_ir(TEMPLATE.read_text(), bm, bn, 16, softmax,
                            b=2, h=3, s=s, d=d, causal=causal)
                        device = ir.split("    ^bb0(", 1)[1].split("\n    } :", 1)[0]
                        nodes, _ = parse_region([line.strip() for line in device.splitlines()[1:]])
                        output = np.full_like(q, np.nan)
                        for batch, head, block in ((0, 0, 0), (1, 2, s // bm - 1)):
                            execute_tensor(nodes, {"%bz": batch, "%by": head, "%bx": block,
                                "%argQ": q, "%argK": k, "%argV": v, "%argO": output})
                            start, end = block * bm, (block + 1) * bm
                            key_end = end if causal else s
                            scores = q[batch, head, start:end].astype(np.float32) @ k[batch, head, :key_end].astype(np.float32).T / np.sqrt(d)
                            if causal:
                                scores[np.arange(key_end)[None, :] > np.arange(start, end)[:, None]] = -np.inf
                            weights = np.exp(scores - scores.max(axis=1, keepdims=True))
                            reference = (weights / weights.sum(axis=1, keepdims=True)) @ v[batch, head, :key_end].astype(np.float32)
                            np.testing.assert_allclose(output[batch, head, start:end], reference, atol=3e-3, rtol=3e-3)

    def test_cli_generates_parseable_hint(self):
        with tempfile.TemporaryDirectory(prefix="attention-shared-generator-") as directory:
            path = Path(directory) / "attention.mlir"
            command = [sys.executable, str(ROOT / "generate_attention_ir.py"),
                "--bm", "32", "--bn", "16", "--bk", "16", "--s", "64", "--d", "64",
                "--output", str(path)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), path)
            self.assertIn("frisk.attention_tiles = array<i64: 16, 16>", path.read_text())
            tool = ROOT / "3rd/deepgengraph/build/test/ThreadTilingTest"
            if tool.is_file():
                result = subprocess.run([str(tool), str(path), "--verify-each"],
                                        text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
