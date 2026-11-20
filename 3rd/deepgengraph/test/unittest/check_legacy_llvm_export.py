#!/usr/bin/env python3
"""Check pooled SHM export with LLVM 15's parser and AMDGPU backend.

Usage: python3 check_legacy_llvm_export.py [build/test/MyTest] [libLLVM-15.so]
Uses the driver's current <input.mlir> <pipeline-enabled> command line.
"""
import ctypes
import ctypes.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def load_llvm(path):
    lib = ctypes.CDLL(path)
    ptr = ctypes.c_void_p
    lib.LLVMContextCreate.restype = ptr
    lib.LLVMCreateMemoryBufferWithMemoryRangeCopy.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    lib.LLVMCreateMemoryBufferWithMemoryRangeCopy.restype = ptr
    lib.LLVMParseIRInContext.argtypes = [ptr, ptr, ctypes.POINTER(ptr), ctypes.POINTER(ptr)]
    lib.LLVMParseIRInContext.restype = ctypes.c_int
    lib.LLVMDisposeMessage.argtypes = [ptr]
    lib.LLVMDisposeModule.argtypes = [ptr]
    lib.LLVMContextDispose.argtypes = [ptr]
    return lib


def load_parser(path):
    lib = load_llvm(path)
    ptr = ctypes.c_void_p

    def parse(text):
        data = text.encode()
        context = lib.LLVMContextCreate()
        buffer = lib.LLVMCreateMemoryBufferWithMemoryRangeCopy(data, len(data), b"shm.ll")
        module, error = ptr(), ptr()
        # LLVMParseIRInContext takes ownership of the memory buffer.
        failed = lib.LLVMParseIRInContext(context, buffer, ctypes.byref(module), ctypes.byref(error))
        message = ctypes.string_at(error).decode() if error.value else ""
        if error.value:
            lib.LLVMDisposeMessage(error)
        if module.value:
            lib.LLVMDisposeModule(module)
        lib.LLVMContextDispose(context)
        return failed, message

    return parse


def emit_assembly(input_path, output_path, library, level):
    # Run in a subprocess: LLVM's default diagnostic handler can exit on a
    # backend error, including "local memory (107520) exceeds limit (65536)".
    lib = load_llvm(library)
    ptr = ctypes.c_void_p
    for component in ("TargetInfo", "Target", "TargetMC", "AsmPrinter", "AsmParser"):
        getattr(lib, "LLVMInitializeAMDGPU" + component)()
    lib.LLVMGetTargetFromTriple.argtypes = [ctypes.c_char_p, ctypes.POINTER(ptr), ctypes.POINTER(ptr)]
    lib.LLVMCreateTargetMachine.argtypes = [ptr, ctypes.c_char_p, ctypes.c_char_p,
                                          ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.LLVMCreateTargetMachine.restype = ptr
    lib.LLVMTargetMachineEmitToMemoryBuffer.argtypes = [ptr, ptr, ctypes.c_int,
                                                     ctypes.POINTER(ptr), ctypes.POINTER(ptr)]
    lib.LLVMGetBufferStart.argtypes = [ptr]
    lib.LLVMGetBufferStart.restype = ptr
    lib.LLVMGetBufferSize.argtypes = [ptr]
    lib.LLVMGetBufferSize.restype = ctypes.c_size_t
    lib.LLVMDisposeMemoryBuffer.argtypes = [ptr]
    lib.LLVMDisposeTargetMachine.argtypes = [ptr]
    lib.LLVMCreatePassBuilderOptions.restype = ptr
    lib.LLVMDisposePassBuilderOptions.argtypes = [ptr]
    lib.LLVMRunPasses.argtypes = [ptr, ctypes.c_char_p, ptr, ptr]
    lib.LLVMRunPasses.restype = ptr
    lib.LLVMGetErrorMessage.argtypes = [ptr]
    lib.LLVMGetErrorMessage.restype = ptr
    lib.LLVMDisposeErrorMessage.argtypes = [ptr]

    context = lib.LLVMContextCreate()
    data = Path(input_path).read_bytes()
    buffer = lib.LLVMCreateMemoryBufferWithMemoryRangeCopy(data, len(data), b"shm.ll")
    module, error, target, output = ptr(), ptr(), ptr(), ptr()
    failed = lib.LLVMParseIRInContext(context, buffer, ctypes.byref(module), ctypes.byref(error))
    assert not failed, ctypes.string_at(error).decode()
    triple = b"amdgcn-amd-amdhsa"
    failed = lib.LLVMGetTargetFromTriple(triple, ctypes.byref(target), ctypes.byref(error))
    assert not failed, ctypes.string_at(error).decode()
    # gfx90a is available in upstream LLVM 15 and has the same 64 KiB LDS limit.
    machine = lib.LLVMCreateTargetMachine(target, triple, b"gfx90a", b"", level, 0, 0)
    assert machine
    options = lib.LLVMCreatePassBuilderOptions()
    pass_error = lib.LLVMRunPasses(module, f"default<O{level}>".encode(), machine, options)
    lib.LLVMDisposePassBuilderOptions(options)
    if pass_error:
        message = lib.LLVMGetErrorMessage(pass_error)
        diagnostic = ctypes.string_at(message).decode()
        lib.LLVMDisposeErrorMessage(message)
        raise AssertionError(diagnostic)
    failed = lib.LLVMTargetMachineEmitToMemoryBuffer(machine, module, 0,
                                                   ctypes.byref(error), ctypes.byref(output))
    assert not failed, ctypes.string_at(error).decode()
    Path(output_path).write_bytes(ctypes.string_at(lib.LLVMGetBufferStart(output),
                                                 lib.LLVMGetBufferSize(output)))
    lib.LLVMDisposeMemoryBuffer(output)
    lib.LLVMDisposeTargetMachine(machine)
    lib.LLVMDisposeModule(module)
    lib.LLVMContextDispose(context)


def main():
    test_dir = Path(__file__).resolve().parent
    binary = (Path(sys.argv[1]) if len(sys.argv) > 1 else
              test_dir.parent / "build/test/MyTest").resolve()
    library = sys.argv[2] if len(sys.argv) > 2 else ctypes.util.find_library("LLVM-15")
    assert library, "LLVM 15 shared library is required for this compatibility test"
    parse = load_parser(library)

    # Verify that this parser actually catches the original failure.
    regression = """
      @shm = addrspace(3) global [16384 x i8] undef
      define ptr addrspace(3) @test() {
        ret ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm, i64 8192)
      }
    """
    failed, error = parse(regression)
    assert failed and "expected '(' in constantexpr" in error, error
    assert not parse(regression.replace("inbounds nuw", "inbounds"))[0]

    with tempfile.TemporaryDirectory(prefix="legacy-shm-export-") as directory:
        work = Path(directory)
        # A loop-carried descriptor exercises cyclic PHIs, nested array fields,
        # entry-edge constants and selects that depend on the original PHI.
        fixture = """
          %D = type { ptr addrspace(3), [2 x i64] }
          @shm = addrspace(3) global [53760 x i8] undef, align 16
          define amdgpu_kernel void @phi_test(ptr addrspace(1) %out, i1 %choose, i32 %n) {
          entry:
            store volatile i8 7, ptr addrspace(3) @shm
            br label %loop
          loop:
            %d = phi %D [ { ptr addrspace(3) @shm, [2 x i64] [i64 1, i64 2] }, %entry ], [ %next, %loop ]
            %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]
            %next = select i1 %choose, %D %d, %D { ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @shm, i64 8192), [2 x i64] [i64 3, i64 4] }
            %p = extractvalue %D %next, 0
            %v = load volatile i8, ptr addrspace(3) %p
            store volatile i8 %v, ptr addrspace(1) %out
            %inc = add i32 %iv, 1
            %done = icmp uge i32 %inc, %n
            br i1 %done, label %exit, label %loop
          exit:
            ret void
          }
        """
        helper = binary.with_name("LegacyLLVMExportTest")
        proc = subprocess.run([str(helper), "-"], input=fixture, text=True,
                              capture_output=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        assert "phi %D" not in proc.stdout
        assert not re.search(r"select i1 [^,]+, %D", proc.stdout)
        fixture_path = work / "phi.ll"
        fixture_path.write_text(proc.stdout)
        proc = subprocess.run([str(binary), str(test_dir / "test_input.mlir"), "1"],
                              cwd=work, text=True, capture_output=True, timeout=120)
        log = proc.stdout + proc.stderr
        assert proc.returncode == 0, log[-6000:]
        assert not re.search(r"error:|LLVM ERROR|Assertion .*failed", log), log[-6000:]
        text = (work / "finalLLVMText.ll").read_text()
        # Ensure the test really exercises nonzero constant offsets in the pool.
        assert re.search(r"getelementptr inbounds \(?i8, ptr addrspace\(3\) @shm_\d+, i64 [1-9]", text)
        assert not re.search(r"getelementptr (?:inbounds )?(?:nuw|nusw)\b", text)
        failed, error = parse(text)
        assert not failed, error
        pool = re.findall(r"@shm_\d+ = .*addrspace\(3\) global \[(\d+) x i8\]", text)
        assert pool == ["53760"], pool
        for level in (0, 3):
            assembly_path = work / f"attention-O{level}.s"
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   "--codegen", str(work / "finalLLVMText.ll"),
                                   str(assembly_path), library, str(level)],
                                  text=True, capture_output=True, timeout=120)
            assert proc.returncode == 0, proc.stderr
            sizes = re.findall(r"\.group_segment_fixed_size:\s*(\d+)", assembly_path.read_text())
            assert sizes == pool, (level, sizes, pool)
            print(f"PASS LLVM 15 AMDGPU O{level}: LDS = {sizes[0]} bytes, no duplicate pool")
            phi_assembly = work / f"phi-O{level}.s"
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   "--codegen", str(fixture_path), str(phi_assembly),
                                   library, str(level)], text=True, capture_output=True, timeout=30)
            assert proc.returncode == 0, proc.stderr
            sizes = re.findall(r"\.group_segment_fixed_size:\s*(\d+)", phi_assembly.read_text())
            assert sizes == pool, sizes
            print(f"PASS loop-carried LDS descriptor O{level}: valid scalar PHIs and 53760-byte LDS")
    print("PASS LLVM 15: reproduced old GEP failure; pooled attention export parses successfully")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--codegen":
        emit_assembly(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        main()
