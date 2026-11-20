#!/usr/bin/env python3
"""Compile-check test_async, or run a linked hsaco with HIP (no PyTorch needed).

Compile: python3 check_async.py --llvm-bin /path/to/llvm/bin --legacy-lds
GPU:     python3 check_async.py --hsaco /path/to/test_async.hsaco
Build the hsaco for the GPU's exact gfx target; see test_async.mlir for commands.
"""

import argparse
import ctypes as C
import ctypes.util
from pathlib import Path
import random
import re
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent


def compile_check(args):
    def tool(name):
        return str(Path(args.llvm_bin) / name) if args.llvm_bin else name

    with tempfile.TemporaryDirectory(prefix="test-async-") as directory:
        ir = Path(directory) / "test_async.ll"
        command = [str(args.driver)]
        if args.legacy_lds:
            command.append("--legacy-lds")
        subprocess.run(command + [str(HERE / "test_async.mlir"), str(ir)], check=True)
        subprocess.run([tool("opt"), "-passes=verify", "-disable-output", str(ir)],
                       check=True)
        text = ir.read_text()
        assert "define amdgpu_kernel void @test_async" in text
        assert "addrspace(3) global [256 x float]" in text
        assert "!reqd_work_group_size" in text
        for arch in ("gfx90a", "gfx942"):
            asm = Path(directory) / f"{arch}.s"
            obj = Path(directory) / f"{arch}.o"
            command = [tool("llc"), "-O2", "-mtriple=amdgcn-amd-amdhsa",
                       f"-mcpu={arch}", str(ir)]
            subprocess.run(command + ["-o", str(asm)], check=True)
            subprocess.run(command + ["-filetype=obj", "-o", str(obj)], check=True)
            assembly = asm.read_text()
            # Ensure a direct global-to-LDS instruction survived code generation.
            dma = re.search(r"global_load_lds_dword\b|global_load_dword[^\n]*\blds\b",
                            assembly)
            assert dma, arch
            for instruction in ("s_waitcnt", "s_barrier", "ds_read_b32",
                                "global_store_dword"):
                assert instruction in assembly, (arch, instruction)
            assert ".amdhsa_group_segment_fixed_size 1024" in assembly
            barrier = assembly.index("s_barrier", dma.end())
            assert "s_waitcnt" in assembly[dma.end():barrier]
            assert assembly.index("ds_read_b32") > barrier
            print(f"PASS: LLVM verification and {arch} LDS DMA code generation")


def gpu_check(hsaco):
    hip = C.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
    ptr = C.c_void_p
    signatures = {
        "hipInit": [C.c_uint], "hipSetDevice": [C.c_int],
        "hipModuleLoad": [C.POINTER(ptr), C.c_char_p],
        "hipModuleGetFunction": [C.POINTER(ptr), ptr, C.c_char_p],
        "hipMalloc": [C.POINTER(ptr), C.c_size_t], "hipFree": [ptr],
        "hipMemcpy": [ptr, ptr, C.c_size_t, C.c_int],
        "hipModuleLaunchKernel": [ptr] + [C.c_uint] * 7 + [ptr, C.POINTER(ptr), ptr],
        "hipDeviceSynchronize": [], "hipModuleUnload": [ptr],
    }
    for name, signature in signatures.items():
        getattr(hip, name).argtypes = signature
        getattr(hip, name).restype = C.c_int

    def call(name, *values):
        status = getattr(hip, name)(*values)
        if status:
            raise RuntimeError(f"{name} failed with HIP status {status}")

    call("hipInit", 0)
    call("hipSetDevice", 0)
    module, kernel = ptr(), ptr()
    call("hipModuleLoad", C.byref(module), str(hsaco.resolve()).encode())
    try:
        call("hipModuleGetFunction", C.byref(kernel), module, b"test_async")
        rng = random.Random(0)
        # Wave/block boundaries, multiple workgroups, and partial final waves.
        for n in (0, 1, 2, 31, 63, 64, 65, 127, 255, 256, 257, 1025, 65539):
            guard = 16
            total = n + 2 * guard
            special = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345,
                       0x00000001, 0xFFFFFFFF]
            words = [special[i] if i < len(special) else rng.getrandbits(32)
                     for i in range(n)]
            expected = [0xA5A5A5A5] * guard + words + [0xA5A5A5A5] * guard
            host_a = (C.c_uint32 * total)(*expected)
            for inplace in (False, True):
                initial = expected if inplace else [0xA5A5A5A5] * total
                host_b = (C.c_uint32 * total)(*initial)
                device_a, device_b = ptr(), ptr()
                try:
                    call("hipMalloc", C.byref(device_a), C.sizeof(host_a))
                    call("hipMemcpy", device_a, host_a, C.sizeof(host_a), 1)
                    if inplace:
                        destination = device_a
                    else:
                        call("hipMalloc", C.byref(device_b), C.sizeof(host_b))
                        call("hipMemcpy", device_b, host_b, C.sizeof(host_b), 1)
                        destination = device_b
                    a = ptr(device_a.value + guard * 4)
                    b = ptr(destination.value + guard * 4)
                    count = C.c_uint64(n)
                    params = (ptr * 3)(C.addressof(a), C.addressof(b),
                                       C.addressof(count))
                    if n:
                        call("hipModuleLaunchKernel", kernel, (n + 255) // 256,
                             1, 1, 256, 1, 1, 0, None, params, None)
                        call("hipDeviceSynchronize")
                    call("hipMemcpy", host_b, destination, C.sizeof(host_b), 2)
                    assert list(host_b) == expected, (n, inplace, "copy/guard mismatch")
                    call("hipMemcpy", host_a, device_a, C.sizeof(host_a), 2)
                    assert list(host_a) == expected, (n, inplace, "input changed")
                finally:
                    if device_b.value:
                        call("hipFree", device_b)
                    if device_a.value:
                        call("hipFree", device_a)
            print(f"PASS: N={n}, bit-exact copy, in-place copy, and guards")
    finally:
        call("hipModuleUnload", module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path,
                        default=HERE.parent / "build/test/MyTestAsync")
    parser.add_argument("--llvm-bin", help="directory containing opt and llc")
    parser.add_argument("--legacy-lds", action="store_true")
    parser.add_argument("--hsaco", type=Path, help="run on GPU instead of compile-checking")
    args = parser.parse_args()
    if args.hsaco:
        gpu_check(args.hsaco)
    else:
        compile_check(args)


if __name__ == "__main__":
    main()
