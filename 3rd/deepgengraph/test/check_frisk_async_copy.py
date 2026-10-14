#!/usr/bin/env python3
"""Frisk tile readiness, scheduling, gfx936 lowering and CPU DMA address checks.

The CPU shim emulates dword DMA's implicit lane offset using the actual exported
LLVM addresses. It is not a gfx936 ISA/performance or concurrent barrier test.
"""
import ctypes as C
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / 'build/test/ThreadTilingTest'
LLVM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin')


def command(args, source=None, ok=True):
    p = subprocess.run(list(map(str, args)), input=source, text=True,
                       capture_output=True, timeout=240)
    assert (p.returncode == 0) == ok, p.stderr[-10000:]
    return p.stdout if ok else p.stderr


def lower(source, *passes, ok=True):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'out.mlir'
        result = command([TOOL, '-', *passes, '-o', out], source, ok)
        return out.read_text() if ok else result


def fixture(shape='260', element='i32', layout='', alignment=4, threads=128):
    src = f'memref<{shape}x{element}{layout}, 1>'
    dst = f'memref<{shape}x{element}, 3>'
    return f'''module {{
  func.func @copy_kernel(%src: {src}, %dst: {dst}) attributes {{thread_num = {threads} : i32}} {{
    frisk.async_copy_gm_to_shm %src to %dst {{alignment = {alignment} : i64}} : {src}, {dst}
    frisk.wait_buffer_ready %dst : {dst}
    return
  }}
}}
'''


def llvm_ir(source, asynchronous=True):
    passes = [] if asynchronous else ['--frisk-lower-async-copy=use-async-copy=false']
    mlir = lower(source, *passes, '--test-async-to-llvm')
    assert 'frisk.' not in mlir and 'unrealized_conversion_cast' not in mlir, mlir
    return command([LLVM / 'mlir-translate', '--mlir-to-llvmir'], mlir)


SHIM = r'''
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
static __thread unsigned tid, pending, errors;
static __thread struct { const void *src; void *dst; } queue[1024];
static int collective;
static pthread_barrier_t barrier;
void test_set_tid(unsigned v) { if (pending) ++errors; tid = v; }
unsigned test_tid(void) { return tid; }
unsigned test_first(unsigned v) { return v; }
void test_dma(const void *src, void *base, unsigned size, unsigned offset, unsigned aux) {
  if (size != 4 || offset || aux || pending == 1024) { ++errors; return; }
  queue[pending].src = src;
  queue[pending++].dst = (char *)base + (tid % 64) * 4;
}
void test_mark(void) {}
void test_wait(unsigned count) {
  if (count) ++errors;
  for (unsigned i = 0; i < pending; ++i) memcpy(queue[i].dst, queue[i].src, 4);
  pending = 0;
}
void test_barrier(void) { if (collective) pthread_barrier_wait(&barrier); }
unsigned test_errors(void) { return errors + pending; }
struct worker { void (*kernel)(void *, void *); void *src, *dst; unsigned tid, errors; };
static void *run_worker(void *arg) {
  struct worker *w = arg;
  tid = w->tid;
  w->kernel(w->src, w->dst);
  w->errors = test_errors();
  return 0;
}
unsigned test_run(void (*kernel)(void *, void *), void *src, void *dst, unsigned threads) {
  pthread_t handles[1024];
  struct worker workers[1024];
  collective = 1;
  if (threads > 1024 || pthread_barrier_init(&barrier, 0, threads)) abort();
  for (unsigned i = 0; i < threads; ++i) {
    workers[i] = (struct worker){kernel, src, dst, i, 0};
    if (pthread_create(&handles[i], 0, run_worker, &workers[i])) abort();
  }
  unsigned result = 0;
  for (unsigned i = 0; i < threads; ++i) { pthread_join(handles[i], 0); result += workers[i].errors; }
  pthread_barrier_destroy(&barrier);
  collective = 0;
  return result;
}
'''


def execute_copy(ir, rows, cols, bits, strides=None, offset=0, threads=128,
                 destination_index=None):
    # AS3 is a 32-bit hardware address. mmap MAP_32BIT preserves that property
    # instead of silently changing the generated LDS pointer arithmetic.
    libc = C.CDLL(None)
    libc.mmap.restype = C.c_void_p
    libc.mmap.argtypes = [C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_int, C.c_long]
    libc.munmap.argtypes = [C.c_void_p, C.c_size_t]
    size = 65536
    address = libc.mmap(None, size, 3, 0x22 | 0x40, -1, 0)
    assert address not in (None, C.c_void_p(-1).value) and address + size < 2**32
    staging = []
    try:
        # Give target-created LDS staging globals real, low-32-bit addresses.
        # Execute all lanes concurrently when repacking crosses lane ownership.
        for match in list(re.finditer(r'^(@\w+) = (?:(?:internal|private) )?addrspace\(3\) global \[(\d+) x (i8|i16|i32|half|float)\].*$', ir, re.M)):
            length = int(match[2]) * {'i8': 1, 'i16': 2, 'i32': 4, 'half': 2, 'float': 4}[match[3]]
            assert length + 64 <= size
            base = libc.mmap(None, size, 3, 0x22 | 0x40, -1, 0)
            assert base not in (None, C.c_void_p(-1).value) and base + size < 2**32
            C.memset(base, 0xA5, size)
            staging.append((base, length))
            ir = ir.replace(match[0] + '\n', '')
            ir = re.sub(re.escape(match[1]) + r'\b', f'inttoptr (i64 {base + 32} to ptr addrspace(3))', ir)
        byte_width = bits // 8
        strides = strides or (cols, 1)
        source_elements = offset + (rows - 1) * strides[0] + (cols - 1) * strides[1] + 1
        source = bytes((i * 73 + i // 7 + 19) % 256 for i in range(source_elements * byte_width))
        src = C.create_string_buffer(source)
        C.memset(address, 0xA5, size)
        guard = 32
        expected = bytearray([0xA5] * size)
        for r in range(rows):
            for c in range(cols):
                start = (offset + r * strides[0] + c * strides[1]) * byte_width
                index = destination_index(r, c) if destination_index else r * cols + c
                dest = guard + index * byte_width
                expected[dest:dest + byte_width] = source[start:start + byte_width]
        replacements = {
            'llvm.amdgcn.workitem.id.x': 'test_tid',
            'llvm.amdgcn.readfirstlane.i32': 'test_first',
            'llvm.amdgcn.global.load.async.lds': 'test_dma',
            'llvm.amdgcn.asyncmark': 'test_mark',
            'llvm.amdgcn.wait.asyncmark': 'test_wait',
            'llvm.amdgcn.s.barrier': 'test_barrier',
        }
        for old, new in replacements.items():
            ir = ir.replace('@' + old + '(', '@' + new + '(')
        ir = re.sub(r'^target (triple|datalayout).*$','', ir, flags=re.M)
        ir = ir.replace('amdgpu_kernel ', '').replace(' immarg', '')
        ir = re.sub(r' addrspace\([13]\)', '', ir)
        assert '@llvm.amdgcn.' not in ir, ir
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'copy.ll').write_text(ir)
            (root / 'shim.c').write_text(SHIM)
            command([LLVM / 'llc', '-mtriple=x86_64-unknown-linux-gnu',
                     '-relocation-model=pic', '-filetype=obj', root / 'copy.ll', '-o', root / 'copy.o'])
            command(['gcc', '-shared', '-fPIC', '-pthread', root / 'copy.o', root / 'shim.c', '-o', root / 'copy.so'])
            lib = C.CDLL(str(root / 'copy.so'))
            lib.copy_kernel.argtypes = [C.c_void_p, C.c_void_p]
            if staging:
                lib.test_run.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint]
                assert lib.test_run(lib.copy_kernel, src, address + guard, threads) == 0
            else:
                for tid in range(threads):
                    lib.test_set_tid(tid)
                    lib.copy_kernel(src, address + guard)
            assert lib.test_errors() == 0
            assert C.string_at(address, size) == bytes(expected), 'copy or guard mismatch'
            for base, length in staging:
                assert C.string_at(base, 32) == bytes([0xA5]) * 32
                assert C.string_at(base + 32 + length, size - 32 - length) == bytes([0xA5]) * (size - 32 - length)
    finally:
        libc.munmap(address, size)
        for base, _ in staging: libc.munmap(base, size)


def main():
    source = fixture()
    result = lower(source, '--frisk-lower-async-copy')
    for op in ('rocdl.global.load.async.lds', 'rocdl.readfirstlane', 'rocdl.asyncmark',
               'rocdl.wait.asyncmark 0', 'gpu.barrier'):
        assert op in result, result
    assert result.index('rocdl.asyncmark') < result.index('rocdl.wait.asyncmark') < result.index('gpu.barrier')
    sync = lower(source, '--frisk-lower-async-copy=use-async-copy=false')
    assert 'rocdl.' not in sync and 'memref.load' in sync and 'gpu.barrier' in sync
    for n in (1, 63, 64, 65, 127, 128, 129, 260, 2048, 2176):
        ir = llvm_ir(fixture(str(n)))
        assert 'llvm.amdgcn.global.load.async.lds' in ir
        execute_copy(ir, 1, n, 32)
    full = lower(fixture('2048'), '--frisk-lower-async-copy')
    assert 'scf.for' not in full and full.count('rocdl.global.load.async.lds') == 16
    assert full.count('rocdl.asyncmark') == 1
    assert 'scf.for' in lower(fixture('2176'), '--frisk-lower-async-copy')
    for mode in (True, False):
        execute_copy(llvm_ir(fixture('3x130', 'i16'), mode), 3, 130, 16)
        execute_copy(llvm_ir(fixture('3x130', 'i16', ', strided<[134, 1], offset: 4>'), mode),
                     3, 130, 16, (134, 1), 4)
        execute_copy(llvm_ir(fixture('3x65', 'i32', ', strided<[1, 3]>'), mode),
                     3, 65, 32, (1, 3))
    # Half transpose / odd width / insufficient alignment must not use dword
    # DMA: it would put adjacent halfwords at the wrong LDS addresses.
    for src, rows, cols, strides in (
        (fixture('3x66', 'i16', ', strided<[1, 3]>'), 3, 66, (1, 3)),
        (fixture('3x65', 'i16'), 3, 65, None),
        (fixture('3x66', 'i16', alignment=2), 3, 66, None)):
        ir = llvm_ir(src)
        assert 'llvm.amdgcn.global.load.async.lds' not in ir
        execute_copy(ir, rows, cols, 16, strides)
    assert 'multiple of 64' in lower(fixture(threads=96), '--frisk-lower-async-copy', ok=False)
    assert 'positive power of two' in lower(fixture(alignment=3), ok=False)
    assert 'global (1)' in lower(source.replace('260xi32, 1', '260xi32, 3'), ok=False)

    # Promotion preserves aliases and the original completion point.
    original = '''func.func @promote(%src: memref<128xf32, 1>, %dst: memref<128xf32, 3>) {
      %alias = frisk.copy %src to %dst [affine_map<() -> (1)>] : memref<128xf32, 1>, memref<128xf32, 3> -> memref<128xf32, 3>
      frisk.wait_buffer_ready %alias : memref<128xf32, 3>
      return
    }'''
    promoted = lower(original, '--frisk-promote-gm-to-shm-async')
    assert 'frisk.copy ' not in promoted and promoted.count('frisk.wait_buffer_ready %arg1') == 2
    # Nested logical views must preserve source row strides and origin.
    view = '''func.func @view(%src: memref<4x132xf32, 1>, %dst: memref<2x128xf32, 3>) attributes {thread_num = 128 : i32} {
      %tile = frisk.buffer_view %src[1, 4], ranges = [2, 128] : memref<4x132xf32, 1> -> memref<2x128xf32, 1>
      frisk.async_copy_gm_to_shm %tile to %dst : memref<2x128xf32, 1>, memref<2x128xf32, 3>
      frisk.wait_buffer_ready %dst : memref<2x128xf32, 3>
      return
    }'''
    materialized = lower(view, '--finalize-thread-tiling')
    assert 'frisk.buffer_view' not in materialized and 'strided<[132, 1]' in materialized
    lower(materialized, '--test-async-to-llvm')
    dynamic_view = view.replace('@view(%src:', '@view(%row: index, %src:').replace('%src[1, 4]', '%src[%row, 4]')
    dynamic_final = lower(dynamic_view, '--finalize-thread-tiling', '--test-async-to-llvm')
    assert 'unrealized_conversion_cast' not in dynamic_final and 'memref.subview' not in dynamic_final
    pipeline = (HERE / 'test_frisk_async_pipeline.mlir').read_text()
    hybrid = lower(pipeline, '--convert-friskbase-to-thread=async-direct-only=true')
    # A's contiguous packed packets stay DMA; B's transposed halfword
    # packing uses the ordinary vector copy and publishes it with a barrier.
    assert hybrid.count('frisk.async_copy_gm_to_shm') == 1
    assert hybrid.count('frisk.wait_buffer_ready') == 1
    assert 'frisk.sync_threads_in_block' in hybrid
    scheduled = lower(pipeline, '--frisk-pipeline-schedule')
    assert 'pipeline.slots = 2' in scheduled
    assert 'frisk.wait_buffer_ready' in scheduled
    # Check the actual selected SSA slot, including multi-result scf.if values.
    last_ready = None
    for line in scheduled.splitlines():
        ready = re.search(r'frisk.wait_buffer_ready (%[\w#]+)', line)
        if ready:
            last_ready = ready[1]
        consumer = re.search(r'frisk.gemm\([^,]+, (%[\w#]+)\)', line)
        if consumer:
            assert consumer[1] == last_ready, (consumer[1], last_ready)
    assert 'requires wait_buffer_ready' in lower(
        pipeline.replace('frisk.wait_buffer_ready %shared : memref<32x32xf16, 3>',
                         'frisk.sync_threads_in_block'), '--frisk-pipeline-schedule', ok=False)
    assert 'requires wait_buffer_ready' in lower(
        pipeline.replace('frisk.wait_buffer_ready %shared : memref<32x32xf16, 3>',
                         'frisk.wait_buffer_ready %lhs : memref<64x32xf16, 3>'),
        '--frisk-pipeline-schedule', ok=False)
    issue = 'frisk.async_copy_gm_to_shm %tile to %shared : memref<32x32xf16, 1>, memref<32x32xf16, 3>'
    wait = 'frisk.wait_buffer_ready %shared : memref<32x32xf16, 3>'
    premature = pipeline.replace(issue + '\n      ' + wait, wait + '\n      ' + issue)
    premature = premature.replace('0, 0, 1, 1, 1, 1', '0, 0, 0, 1, 1, 1')
    assert 'requires wait_buffer_ready' in lower(premature, '--frisk-pipeline-schedule', ok=False)
    # Both modes must accept exactly the same scheduled tile/readiness IR.
    for mode in ('true', 'false'):
        lower(scheduled, '--convert-friskbase-to-thread', '--finalize-thread-tiling',
              f'--frisk-lower-async-copy=use-async-copy={mode}')
    selected_sync = lower(scheduled, '--frisk-lower-async-to-sync')
    assert 'async_copy_gm_to_shm' not in selected_sync and 'wait_buffer_ready' not in selected_sync
    assert 'frisk.explicit_wait' in selected_sync and 'frisk.sync_threads_in_block' in selected_sync
    lower(selected_sync, '--convert-friskbase-to-thread', '--finalize-thread-tiling')
    print('PASS: Frisk async/sync copies, DMA addresses and guards, LLVM export, views, pipeline readiness')


if __name__ == '__main__':
    main()
