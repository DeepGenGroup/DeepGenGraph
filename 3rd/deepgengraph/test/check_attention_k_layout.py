#!/usr/bin/env python3
"""Check the example ABI conversion and final scalar K addressing (no GPU).

Usage: check_attention_k_layout.py path/to/ThreadTilingTest [attention.ll [packed]]
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def check_llvm(text, packed=False):
    # Trace actual GLOBAL -> LDS K values in the full lowered attention kernel.
    # Each global element is a distinct symbolic value; no uniform inputs.
    from check_gemm_pv_addresses import Addresses, split_args

    class Trace(Addresses):
        def value(self, name):
            if name not in self.cache and name in self.definitions:
                opcode, body = self.definitions[name].split(" ", 1)
                if opcode == "getelementptr":
                    ty, ptr, index = split_args(body)
                    base, offset = self.typed(ptr)
                    size = {"i8": 1, "half": 2, "float": 4}[ty.split()[-1]]
                    self.cache[name] = (base, offset + size * self.typed(index))
                elif opcode == "load" and body.startswith("<"):
                    ty, ptr, *_ = split_args(body)
                    assert "half>" in ty
                    base, offset = self.typed(ptr)
                    self.cache[name] = [(base, offset + 2 * i)
                                        for i in range(int(ty.split()[0][1:]))]
            return super().value(name)

    definitions = dict(re.findall(r"^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$", text, re.M))
    def builtin(name):
        return next(n for n, e in definitions.items() if "@llvm.amdgcn." + name + "()" in e)
    tid, head, tile = map(builtin, ("workitem.id.x", "workgroup.id.x", "workgroup.id.y"))
    args = re.findall(r"ptr addrspace\(1\) (%\w+)",
                      re.search(r"define amdgpu_kernel void @Attn_p2\((.*)\) #", text)[1])
    ivs = [n for n, e in definitions.items() if e.startswith("phi i64 ")]
    assert len(ivs) == 1, ivs
    first_mma = next(e for e in definitions.values() if 'asm sideeffect ' in e and 'v_mmac_' in e)
    operand = split_args(re.search(r'"\((.*)\)', first_mma)[1])[1]
    prefix = text[:text.index(first_mma)]
    stores = re.findall(r"^  store (half|<\d+ x half>) (%\w+), ptr addrspace\(3\) (%\w+)", prefix, re.M)
    if packed:
        loads = re.findall(r" = load (half|<\d+ x half>), ptr addrspace\(1\)", prefix)
        assert loads == ["<8 x half>"] * 16, loads
    for h, step in ((0, 0), (7, 3), (31, 127)):
        common = {a: (a, 0) for a in args}
        common.update({head: h, tile: 63, ivs[0]: step})
        base, origin = Trace(definitions, {**common, tid: 0}).typed(operand)[0]
        covered = set()
        for lane in range(128):
            trace = Trace(definitions, {**common, tid: lane})
            for ty, value, dest in stores:
                values = trace.value(value)
                values = [values] if ty == "half" else values
                if values[0][0] != args[1]:
                    continue
                target, offset = trace.value(dest)
                assert target == base
                for i in range(len(values)):
                    flat = (offset - origin) // 2 + i
                    if packed:
                        group = flat // (32 * 8)
                        d = group // 4 * 32 + group % 4 * 4 + flat % 8 // 4 * 16 + flat % 4
                        s = flat // 8 % 32
                        column_group = s // 8
                        s = (column_group * 8 + (s // 4 - column_group // 4 - group // 2) % 2 * 4
                             + (s - column_group) % 4)
                    else:
                        d, s = divmod(flat, 32)
                    assert 0 <= d < 128 and (d, s) not in covered
                    covered.add((d, s))
                    expected = 2 * (h * 4096 * 128 + (step * 32 + s) * 128 + d)
                    assert values[i] == (args[1], expected), (lane, d, s, values[i], expected)
        assert len(covered) == 128 * 32
    print("PASS lowered BHSD K: 12288 GLOBAL/LDS coordinates across all 128 lanes")


def fixture(key_arg=1, sequence=8, name="Attn_p2"):
    normal = f"memref<2x3x{sequence}x4xf16, 1>"
    transposed = f"memref<2x3x4x{sequence}xf16, 1>"
    types = [normal] * 4
    types[key_arg] = transposed
    args = ", ".join(f"%arg{i}: {ty}" for i, ty in enumerate(types))
    return f"""module {{
      func.func @{name}({args}) attributes {{thread_num = 64 : i32}} {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %view = frisk.buffer_view %arg{key_arg}[1, 2, 1, 1], ranges = [2, 3]
          : {transposed} -> memref<2x3xf16, 1>
        %nested = frisk.buffer_view %view[0, 1], ranges = [2, 2]
          : memref<2x3xf16, 1> -> memref<2x2xf16, 1>
        %value = memref.load %nested[%c1, %c1] : memref<2x2xf16, 1>
        memref.store %value, %arg3[%c0, %c0, %c0, %c0] : {normal}
        return
      }}
    }}"""


def check_partial_copy(tool):
    # A nonzero-origin transposed slice smaller than a block: only four lanes
    # may read, each reading eight D elements. The barrier must be unconditional.
    source = """module {
      func.func @copy_transposed(%src: memref<8x16xf16, 1>, %dst: memref<8x4xf16, 3>)
          attributes {thread_num = 64 : i32} {
        %t = memref.transpose %src (d0,d1) -> (d1,d0)
          : memref<8x16xf16, 1> to memref<16x8xf16, strided<[1,16]>, 1>
        %v = frisk.buffer_view %t[4, 2], ranges = [8, 4]
          : memref<16x8xf16, strided<[1,16]>, 1> -> memref<8x4xf16, 1>
        frisk.copy %v to %dst [affine_map<() -> (2)>]
          : memref<8x4xf16, 1>, memref<8x4xf16, 3>
        return
      }
    }"""
    with tempfile.TemporaryDirectory(prefix="column-copy-") as directory:
        output = Path(directory) / "copy.mlir"
        proc = subprocess.run([tool, "--convert-friskbase-to-thread", "--finalize-thread-tiling",
                               "-o", str(output)],
                              input=source, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        ir = output.read_text()
    assert "memref.transpose" not in ir and "frisk.buffer_view" not in ir
    assert re.search(r"vector.load %arg0\[%\w+, %c4\].*vector<8xf16>", ir), ir
    assert "arith.cmpi ult" in ir and "scf.if" in ir
    assert ir.count("memref.store") == 8
    assert ir.index("frisk.sync_threads_in_block") > ir.index("iterLabel"), ir
    print("PASS transposed partial copy: offset slice, vector loads and unconditional barrier")


def main():
    tool = sys.argv[1]

    def run(source, options="", finalize=False, error=None):
        command = [tool, "--normalize-attention-k-layout" +
                   ("=" + options if options else "")]
        if finalize:
            command.append("--finalize-thread-tiling")
        proc = subprocess.run(command, input=source, capture_output=True,
                              text=True, timeout=60)
        if error:
            assert proc.returncode != 0 and error in proc.stderr, proc.stderr
        else:
            assert proc.returncode == 0, proc.stderr
        return proc.stdout

    for key_arg in (1, 2):
        source = fixture(key_arg)
        normalized = run(source)
        assert normalized.count("memref.transpose") == 1
        assert "strided<[96, 32, 1, 4]>" in normalized
        assert run(normalized) == normalized, "normalization must be idempotent"
        lowered = run(source, finalize=True)
        assert "memref.transpose" not in lowered and "frisk.buffer_view" not in lowered
        # Origin [1,2,1,1] + nested [0,1] + local [1,1] =
        # old BHDS [1,2,2,3], hence physical BHSD [1,2,3,2].
        constants = dict(re.findall(r"(%\w+) = arith.constant (\d+) : index", lowered))
        load = re.search(r"memref.load %arg" + str(key_arg) + r"\[([^]]+)\]", lowered)
        assert load, lowered
        assert [int(constants[x.strip()]) for x in load[1].split(",")] == [1, 2, 3, 2], lowered

    other = run(fixture(name="other"))
    assert "memref.transpose" not in other and "frisk.k_layout_bhsd" not in other
    run(fixture(sequence=4), error="S == D is ambiguous")
    assert "memref.transpose" in run(fixture(sequence=4), "key-arg=1")
    run(fixture(), "key-arg=0", error="key-arg must be 1 or 2")
    run(fixture(), "key-arg=2", error="expected contiguous BHDS K")
    direct = fixture().replace("%view = frisk.buffer_view", """
        %bad = memref.load %arg1[%c0, %c0, %c0, %c0] : memref<2x3x4x8xf16, 1>
        %view = frisk.buffer_view""")
    run(direct, error="supports buffer_view users only")
    print("PASS K layout: both ABIs, nested view addressing, idempotence and rejection checks")
    check_partial_copy(tool)
    if len(sys.argv) > 2:
        check_llvm(Path(sys.argv[2]).read_text(), len(sys.argv) > 3 and sys.argv[3] == "packed")


if __name__ == "__main__":
    main()
