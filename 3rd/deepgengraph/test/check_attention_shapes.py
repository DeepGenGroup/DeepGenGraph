#!/usr/bin/env python3
"""Check configurable attention shapes, addressing, scales and CLI validation."""
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from generate_attention_ir import TEMPLATE, generate_ir


class AttentionShapeTests(unittest.TestCase):
    def test_custom_shapes(self):
        for softmax in ("online", "legacy"):
            for bk in (None, 16):
                for d in (32, 64, 128):
                    with self.subTest(softmax=softmax, bk=bk, d=d):
                        ir, shape = generate_ir(TEMPLATE.read_text(), 64, 32, bk,
                                                softmax, b=2, h=3, s=256, d=d)
                        self.assertEqual(shape, (2, 3, 256, d))
                        self.assertIn(f"tensor<2x256x3x{d}xf16>", ir)
                        self.assertIn(f"tensor<2x3x{d}x256xf16>", ir)
                        self.assertIn("tensor<2x3x256x256xf32>", ir)
                        self.assertIn("shape = [256, 256]", ir)
                        self.assertIn("grid = [2, 3, 4]", ir)
                        self.assertRegex(ir, r"res_dims = \[0\].*?unit_num = 2 : i64")
                        self.assertRegex(ir, r"res_dims = \[2\].*?unit_num = 3 : i64")
                        self.assertNotIn("4096", ir)
                        prefix, kernel = ir.split("  deepgengraph.kernel", 1)
                        scale = float(re.search(r"dense<([^>]+)>", prefix)[1])
                        self.assertAlmostEqual(scale, math.sqrt(d), delta=0.002)
                        scale = float(re.search(r"dense<([^>]+)>", kernel)[1])
                        # D=128 retains the template's historical rounded scale.
                        self.assertAlmostEqual(scale, math.log2(math.e) / math.sqrt(d), delta=2e-5)
                        self.assertIn(f"stride = [{3*d}, 1]", kernel)
                        self.assertIn(f"stride = [1, {3*d}]", kernel)
                        self.assertIn(f"shape = [{d}, 256]", kernel)
                        self.assertIn(f"shape = [256, {d}]", kernel)

    def test_batch_and_head_pointer_offsets(self):
        # Execute the generated integer address arithmetic for batch 1, head 2,
        # query tile 3. This catches a missing batch term, even if types match.
        for bk in (None, 16):
            ir, _ = generate_ir(TEMPLATE.read_text(), 64, 32, bk,
                                b=2, h=3, s=256, d=64)
            kernel = ir.split("  deepgengraph.kernel", 1)[1]
            values = {"%bz": 1, "%by": 2, "%bx": 3, "%od": 0,
                      "%dk": 0, "%n": 0}
            offsets = {}
            for line in kernel.splitlines():
                constant = re.search(r"(%\w+) = arith.constant (\d+) : index", line)
                binary = re.search(r"(%\w+) = arith.(addi|muli) (%\w+)\s*,\s*(%\w+) : index", line)
                pointer = re.search(r"block_ptr_of base = %(arg\w+), base_offset = (%\w+)", line)
                if constant:
                    values[constant[1]] = int(constant[2])
                elif binary and binary[3] in values and binary[4] in values:
                    lhs, rhs = values[binary[3]], values[binary[4]]
                    values[binary[1]] = lhs + rhs if binary[2] == "addi" else lhs * rhs
                elif pointer:
                    offsets[pointer[1]] = values[pointer[2]]
            self.assertEqual(offsets, {"argQ": 94208, "argO": 94208,
                                       "argK": 81920, "argV": 81920})

    def test_defaults_and_partial_override(self):
        source = TEMPLATE.read_text()
        fixture = TEMPLATE.with_name("test_input_spidD.mlir").read_text()
        bm, bn, bk = (int(re.search(rf"%c{name} = arith.constant (\d+)", fixture)[1])
                      for name in ("BM", "BN", "BK"))
        def algorithm(text):
            # The benchmark fixture adds comments and a scheduling hint.
            text = re.sub(r"//[^\n]*", "", text)
            text = text.replace("{frisk.pipeline_hint = 2 : i64}", "")
            return " ".join(text.split())

        self.assertEqual(algorithm(generate_ir(source, bm, bn, bk, "legacy")[0]),
                         algorithm(fixture))
        self.assertEqual(generate_ir(source, 64, 32, h=4)[1], (1, 4, 4096, 128))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(ROOT / "generate_attention_ir.py"),
                       "--bm", "64", "--bn", "32", "--b", "2", "--h", "3",
                       "--s", "256", "--d", "64", "--output-dir", directory]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = Path(result.stdout.strip())
            self.assertEqual(output.name, "test_input_2x3x256x64-64-32-online.mlir")
            self.assertIn("tensor<2x256x3x64xf16>", output.read_text())
            for flags, message in ((["--b", "0"], "B"), (["--h", "-1"], "H"),
                                   (["--s", "0"], "S"), (["--d", "0"], "D"),
                                   (["--s", "255"], "S=255"),
                                   (["--d", "48", "--bk", "32"], "D=48")):
                invalid_dir = Path(directory) / "invalid"
                result = subprocess.run(command + ["--output-dir", str(invalid_dir)] + flags,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse(invalid_dir.exists())


if __name__ == "__main__":
    unittest.main()
