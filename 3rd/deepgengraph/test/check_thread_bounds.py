#!/usr/bin/env python3
"""Check ThreadIR launch bounds through ROCDL, LLVM and rematerialization."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'build/test/ThreadTilingTest'
LIFETIMES = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'build/test/LegacyLLVMExportTest'
LLVM = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('/data2/xsl/install/bin')


def run(args, source):
    result = subprocess.run(list(map(str, args)), input=source, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def fixture(count):
    return f'''module {{
      func.func @bounds(%out: memref<8192xi64, 1>, %n: index)
          attributes {{thread_num = {count} : i32}} {{
        %x = gpu.thread_id x
        %y = gpu.thread_id y
        %z = gpu.thread_id z
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c7 = arith.constant 7 : index
        %ix = arith.index_cast %x : index to i64
        %iy = arith.index_cast %y : index to i64
        %iz = arith.index_cast %z : index to i64
        memref.store %ix, %out[%c0] : memref<8192xi64, 1>
        memref.store %iy, %out[%c1] : memref<8192xi64, 1>
        memref.store %iz, %out[%c2] : memref<8192xi64, 1>
        %offset = arith.muli %x, %c7 : index
        scf.for %i = %c0 to %n step %c1 {{
          %value = arith.index_cast %i : index to i64
          memref.store %value, %out[%offset] : memref<8192xi64, 1>
        }}
        return
      }}
    }}'''


for count in (1, 32, 64, 192, 256, 512, 1024):
    lowered = run([TOOL, '--verify-each', '--test-async-to-llvm'], fixture(count))
    for axis, upper in (('x', count), ('y', 1), ('z', 1)):
        assert f'rocdl.workitem.id.{axis} range <i32, 0, {upper}>' in lowered, lowered
    exported = run([LLVM / 'mlir-translate', '--mlir-to-llvmir'], lowered)
    run([LLVM / 'opt', '-passes=verify', '-disable-output'], exported)
    # Check call-site return attributes, not a declaration's generic hardware
    # bound or the separate function-level reqd_work_group_size metadata.
    for axis, upper in (('x', count), ('y', 1), ('z', 1)):
        assert re.search(rf'call range\(i32 0, {upper}\) i32 '
                         rf'@llvm.amdgcn.workitem.id.{axis}\(\)', exported), exported
    # Range consumers must not need to recover bounds from function metadata.
    without_workgroup = re.sub(r' !reqd_work_group_size !\d+', '', exported)
    optimized = run([LIFETIMES, '--optimize-register-lifetimes', '-'], without_workgroup)
    run([LLVM / 'opt', '-passes=verify', '-disable-output'], optimized)
    if count > 1:
        anchors = re.findall(r'^.*%address.thread[\w.]* = .*$', optimized, re.M)
        assert anchors, optimized
        for anchor in anchors:
            slot = re.search(r'!range !(\d+)', anchor)
            assert slot and f'!{slot[1]} = !{{i32 0, i32 {count}}}' in optimized, anchor
        o3 = run([LLVM / 'opt', '-S', '-passes=default<O3>'], optimized)
        assert 'asm sideeffect "", "=v,0"' in o3, o3
        for anchor in re.findall(r'^.*%address.thread[\w.]* = .*$', o3, re.M):
            slot = re.search(r'!range !(\d+)', anchor)
            assert slot and f'!{slot[1]} = !{{i32 0, i32 {count}}}' in o3, anchor
        # Real backend acceptance of the exported range and tied VGPR identity.
        assembly = run([LLVM / 'llc', '-mtriple=amdgcn-amd-amdhsa',
                        '-mcpu=gfx90a', '-O3', '-o', '-'], o3)
        assert 'bounds:' in assembly, assembly
    else:
        assert 'address.thread' not in optimized, optimized

print('PASS: one-dimensional launch bounds exported on x/y/z; local thread identities retain ranges through lifetime optimization and O3')

# Exercise the same query after func-to-LLVM conversion, with no launch bound
# for malformed/missing contracts. No range may be invented even for y/z.
for attr in ('', 'thread_num = 0 : i32', 'thread_num = -1 : i32',
             'thread_num = 1025 : i32', 'thread_num = "256"',
             'thread_num = 18446744073709551616 : i128'):
    attributes = 'attributes {' + attr + '}' if attr else ''
    source = f'''module {{
      llvm.func @unknown() -> i64 {attributes} {{
        %tid = gpu.thread_id y
        %id = builtin.unrealized_conversion_cast %tid : index to i64
        llvm.return %id : i64
      }}
    }}'''
    lowered = run([TOOL, '--verify-each', '--test-exp2-to-rocdl'], source)
    assert 'rocdl.workitem.id.y' in lowered and 'range <' not in lowered, lowered

print('PASS: missing, malformed and unsupported launch bounds remain unknown')
