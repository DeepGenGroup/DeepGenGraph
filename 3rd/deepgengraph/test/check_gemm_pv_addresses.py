#!/usr/bin/env python3
"""Check actual LLVM PV operand addresses for every thread and MMA fragment.

Usage: python3 check_gemm_pv_addresses.py [build/finalLLVMText.ll]

Loads evaluate to (buffer, offset), rather than constants: repeating the first
P fragment cannot pass this check even when the debug input is all ones.
This checks lowering, not the DCU instruction's runtime behavior.
"""
import re
import sys
from pathlib import Path


def split_args(text):
    depth = 0
    start = 0
    result = []
    for i, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    result.append(text[start:].strip())
    return result


class Addresses:
    def __init__(self, definitions, context):
        self.definitions = definitions
        self.cache = dict(context)

    def typed(self, text):
        if text.startswith(("<", "[")):
            end = text.index(">" if text[0] == "<" else "]")
            count = int(text[1:end].split()[0])
            value = text[end + 1:].strip()
            if value == "poison":
                return [None] * count
            if value == "zeroinitializer":
                return [0] * count
            if value.startswith("<"):
                return [self.typed(x) for x in split_args(value[1:-1])]
            return self.value(value)
        return self.value(text.split()[-1])

    def value(self, name):
        if name in self.cache:
            return self.cache[name]
        if name.startswith("@"):
            return (name, 0)
        if not name.startswith("%"):
            return int(name)
        expr = self.definitions[name]
        opcode, body = expr.split(" ", 1)
        if opcode in ("add", "sub", "mul", "sdiv", "srem"):
            body = re.sub(r"^(?:(?:nsw|nuw) )*i\d+ ", "", body)
            a, b = map(self.value, split_args(body))
            result = {
                "add": lambda: a + b, "sub": lambda: a - b,
                "mul": lambda: a * b, "sdiv": lambda: int(a / b),
                "srem": lambda: a - int(a / b) * b,
            }[opcode]()
        elif opcode == "icmp":
            pred, _, body = body.split(" ", 2)
            a, b = map(self.value, split_args(body))
            result = {"slt": a < b, "sge": a >= b, "eq": a == b}[pred]
        elif opcode == "select":
            cond, yes, no = split_args(body)
            result = self.typed(yes if self.typed(cond) else no)
        elif opcode in ("sext", "zext", "trunc"):
            result = self.value(body.split()[1])
        elif opcode == "getelementptr":
            _, ptr, index = split_args(body)
            buffer, offset = self.typed(ptr)
            result = (buffer, offset + self.typed(index))
        elif opcode == "load":
            result = self.typed(split_args(body)[1])
        elif opcode == "insertelement":
            vector, scalar, index = split_args(body)
            result = list(self.typed(vector))
            result[self.typed(index)] = self.typed(scalar)
        elif opcode == "shufflevector":
            a, b, indices = map(self.typed, split_args(body))
            result = [(a + b)[i] for i in indices]
        elif opcode in ("extractvalue", "extractelement"):
            vector, index = split_args(body)
            result = self.typed(vector)[self.typed(index)]
        elif opcode == "insertvalue":
            vector, scalar, index = split_args(body)
            result = list(self.typed(vector))
            result[self.typed(index)] = self.typed(scalar)
        else:
            raise AssertionError(f"Unsupported address dependency: {name} = {expr}")
        self.cache[name] = result
        return result


def check_vtile_copy(text, definitions):
    """Evaluate GM -> LDS addresses, independently of the MMA read layout."""
    tid = next(n for n, e in definitions.items()
               if "@llvm.amdgcn.workitem.id.x()" in e)
    head = next(n for n, e in definitions.items()
                if "@llvm.amdgcn.workgroup.id.x()" in e)
    stores = []
    for match in re.finditer(
        r"^  store half (%\w+), ptr addrspace\(3\) (%\w+),", text, re.M
    ):
        value, dest = match.groups()
        if "@shm_2," in definitions[dest]:
            stores.append((value, dest, match.start()))
    assert len(stores) == 4, "Expected four unrolled scalar V copy stores"

    # Discover the two induction variables from the actual address graph.
    phis, seen = set(), set()

    def visit(name):
        if name in seen or name not in definitions:
            return
        seen.add(name)
        expr = definitions[name]
        if expr.startswith("phi i64 "):
            phis.add(name)
            return
        for operand in re.findall(r"%[\w.]+", expr):
            visit(operand)

    for value, dest, _ in stores:
        visit(value)
        visit(dest)
    assert len(phis) == 2, phis
    outer, chunk = sorted(phis, key=lambda n: text.index(f"  {n} ="))
    assert re.search(rf"icmp slt i64 {re.escape(chunk)}, 8\b", text)
    vptr = "%1"  # Attn_p2's positional ABI is Q, V, K, O.
    for h in (0, 1, 31):
        for iteration in (0, 1, 63, 127):
            covered = set()
            for thread in range(128):
                for c in range(8):
                    evaluator = Addresses(definitions, {
                        tid: thread, head: h, outer: iteration, chunk: c,
                        vptr: ("V", 0),
                    })
                    for j, (value, dest, _) in enumerate(stores):
                        local = thread * 32 + c * 4 + j
                        expected = h * 4096 * 128 + iteration * 32 * 128 + local
                        assert evaluator.value(value) == ("V", expected), (
                            h, iteration, thread, c, j, "wrong GM V address")
                        assert evaluator.value(dest) == ("@shm_2", local), (
                            h, iteration, thread, c, j, "wrong LDS V address")
                        assert local not in covered, "Duplicate V copy write"
                        covered.add(local)
            assert covered == set(range(32 * 128)), "Incomplete V tile"
    after_copy = text[stores[-1][2]:]
    first_read = re.search(r"^  %\w+ = load half, ptr addrspace\(3\)", after_copy, re.M)
    assert first_read and "@llvm.amdgcn.s.barrier()" in after_copy[:first_read.start()]
    print("PASS V copy: all 4096 LDS elements exactly once; GM addresses at head/K boundaries; barrier before reads")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "build/finalLLVMText.ll"
    )
    text = path.read_text()
    definitions = dict(re.findall(r"^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$", text, re.M))
    check_vtile_copy(text, definitions)
    mma = [name for name, expr in definitions.items()
           if 'asm sideeffect "' in expr and 'v_mmac_' in expr]
    assert len(mma) == 2, "Expected QK and PV GEMMs"
    for name in mma:
        asm = definitions[name].split('asm sideeffect "', 1)[1].split('"', 1)[0]
        prefix, suffix = asm.split("v_mmac_f32_16x16x16_f16", 1)
        # Operand addresses alone cannot detect an A/B swap in the asm:
        # the column-varying V hardware probe exposed that mismatch.
        assert suffix.startswith(" $0, $1, $2, $3"), (
            "MMAC must consume {A, B, C} in the order required by the checked layouts")
        assert prefix.count("s_nop 7") == 8, "MMAC input spacing was lost"
        assert suffix.count("s_nop 7") == 8, "MMAC output spacing was lost"
    print("PASS QK/PV: A/B/C operand order and hardware-validated scheduling padding")
    pv = definitions[mma[1]]
    operands = re.search(r'"\(<4 x half> (%\w+), <4 x half> (%\w+),', pv).groups()
    before = text[:text.index(f"  {mma[1]} =")]
    mn, k = re.findall(r"^  (%\w+) = phi i64", before, re.M)[-2:]
    tid = next(name for name, expr in definitions.items() if "@llvm.amdgcn.workitem.id.x()" in expr)
    assert re.search(rf"icmp slt i64 {re.escape(k)}, 2\b", before)
    assert re.search(rf"icmp slt i64 {re.escape(mn)}, 16\b", before)
    accumulator = re.search(r", <4 x float> (%\w+)\)", pv).group(1)
    carried = re.search(r"extractvalue \[1 x <4 x float>\] (%\w+), 0",
                        definitions[accumulator]).group(1)
    carry_phi = definitions[carried]
    assert carry_phi.startswith("phi [1 x <4 x float>]")
    assert "[ zeroinitializer," in carry_phi
    updated = re.search(r"\[ (%\w+),", carry_phi).group(1)
    assert f"<4 x float> {mma[1]}, 0" in definitions[updated], "PV must feed MMA output into the next K iteration"
    # Extra materialized shared copies can change the allocation number.
    p_buffer = Addresses(definitions, {tid: 0, mn: 0, k: 0}).value(operands[0])[0][0]
    assert re.search(re.escape(p_buffer) + r" = addrspace\(3\) global \[64 x \[32 x half\]\]", text)
    for thread in range(128):
        lane = thread % 64
        for tile in range(16):
            for reduction in range(2):
                evaluator = Addresses(definitions, {tid: thread, mn: tile, k: reduction})
                a, b = map(evaluator.value, operands)
                row = tile // 8 * 32 + thread // 64 * 16 + lane % 16
                col = tile % 8 * 16 + lane % 16
                kk = reduction * 16 + lane // 16 * 4
                expected_a = [(p_buffer, row * 32 + kk + j) for j in range(4)]
                expected_b = [("@shm_2", (kk + j) * 128 + col) for j in range(4)]
                assert a == expected_a, (thread, tile, reduction, "P", a, expected_a)
                assert b == expected_b, (thread, tile, reduction, "V", b, expected_b)
    print("PASS PV: all 32768 scalar operand addresses (128 threads, 16 MN tiles, 2 K tiles)")
    print("PASS PV: two K iterations carry the previous MMA result as accumulator")

    # Give every element of the scaled Q thread vector a distinct logical
    # coordinate, then evaluate the actual vector extraction feeding QK.
    qk = definitions[mma[0]]
    q_operand = re.search(r'"\(<4 x half> (%\w+),', qk).group(1)
    before = text[:text.index(f"  {mma[0]} =")]
    mn, k = re.findall(r"^  (%\w+) = phi i64", before, re.M)[-2:]
    scaled_q = [name for name, expr in definitions.items() if expr.startswith("fmul <32 x half>")]
    assert len(scaled_q) == 2, "Q scale must retain both M tiles and all eight K tiles"
    for thread in range(128):
        lane = thread % 64
        context = {tid: thread}
        for block_m, name in enumerate(scaled_q):
            row = block_m * 32 + thread // 64 * 16 + lane % 16
            context[name] = [("Q", row, j // 4 * 16 + lane // 16 * 4 + j % 4)
                             for j in range(32)]
        for tile in range(4):
            for reduction in range(8):
                evaluator = Addresses(definitions, {**context, mn: tile, k: reduction})
                actual = evaluator.value(q_operand)
                row = tile // 2 * 32 + thread // 64 * 16 + lane % 16
                expected = [("Q", row, reduction * 16 + lane // 16 * 4 + j)
                            for j in range(4)]
                assert actual == expected, (thread, tile, reduction, actual, expected)
    print("PASS QK: all 16384 scaled-Q fragment elements retain their M/K coordinates")


if __name__ == "__main__":
    main()
