#!/usr/bin/env python3
"""Scheduled lowering: bound address reuse without moving memory or math."""
import re
import sys
from pathlib import Path
from check_register_lifetimes import run, body, normalize_metadata_ids


def main():
    tool, thread = map(lambda p: Path(p).resolve(), sys.argv[1:3])
    source = '''target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare void @consume(float) convergent
define void @scheduled(ptr addrspace(3) %base, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idx = mul i32 %tid, 7
  %ptr = getelementptr float, ptr addrspace(3) %base, i32 %idx
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
'''
    for i in range(24):
        source += f'''  %v{i} = load volatile float, ptr addrspace(3) %ptr
  %f{i} = fadd float %v{i}, {float(i):.1f}
  call void @consume(float %f{i})
'''
    source += '''  call void @llvm.amdgcn.s.barrier()
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %n
  br i1 %more, label %loop, label %exit
exit:
  ret void
}
'''
    fixed = run([tool, '--scheduled-register-lifetimes', '-'], source)
    llvm = Path('/data2/xsl/install/bin')
    run([llvm/'opt', '-passes=verify', '-disable-output', '-'], fixed)
    def events(s):
        return re.findall(r'load volatile|fadd float|call void @consume|call void @llvm.amdgcn.s.barrier', s)
    assert events(source) == events(fixed)
    loop = body(fixed, 'scheduled').split('loop:')[1]
    loads = re.findall(r'load volatile float, ptr addrspace\(3\) (%[\w.]+)', loop)
    assert len(loads) == 24 and len(set(loads)) == 3, loads
    assert '%ptr' not in fixed
    again = run([tool, '--scheduled-register-lifetimes', '-'], fixed)
    assert normalize_metadata_ids(fixed) == normalize_metadata_ids(again), 'not idempotent'
    optimized = run([llvm/'opt', '-S', '-passes=default<O3>', '-'], fixed)
    assert optimized.count('asm sideeffect "", "=v,0,i"') >= 3

    # Merge adjacent reads, but do not hoist the next pair across FP compute.
    mlir = '''func.func @packed(%a: memref<16xf32, 3>) -> f32 attributes {cpsat.preserve_schedule} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %x = memref.load %a[%c0] : memref<16xf32, 3>
  %y = memref.load %a[%c1] : memref<16xf32, 3>
  %s = arith.addf %x, %y : f32
  %z = memref.load %a[%c2] : memref<16xf32, 3>
  %w = memref.load %a[%c3] : memref<16xf32, 3>
  %t = arith.addf %z, %w : f32
  %r = arith.addf %s, %t : f32
  return %r : f32
}
'''
    packed = run([thread, '--ir-deep-optimize', '-'], mlir)
    assert packed.count('vector.load') == 2 and 'vector<4xf32>' not in packed, packed
    assert re.search(r'vector.load.*arith.addf.*vector.load', packed, re.S)
    # Independent memory effects must still block merging even in CP-SAT mode.
    intervening = mlir.replace('  %y = memref.load',
        '  memref.store %x, %a[%c1] : memref<16xf32, 3>\n  %y = memref.load')
    packed = run([thread, '--ir-deep-optimize', '-'], intervening)
    assert packed.count('memref.load') == 2, packed
    # Execute both versions on CPU after removing only LDS address space.
    # The intervening write must change the observed sum from 10 to 9.
    for fixture, expected in ((mlir, 10.0), (intervening, 9.0)):
        fixture += f'''func.func @main() -> i32 {{
  %a = memref.alloc() : memref<16xf32, 3>
  %one = arith.constant 1 : index
  affine.for %i = 0 to 4 {{
    %v = arith.addi %i, %one : index
    %vi = arith.index_cast %v : index to i32
    %vf = arith.sitofp %vi : i32 to f32
    memref.store %vf, %a[%i] : memref<16xf32, 3>
  }}
  %r = func.call @packed(%a) : (memref<16xf32, 3>) -> f32
  %want = arith.constant {expected} : f32
  %ok = arith.cmpf oeq, %r, %want : f32
  %zero = arith.constant 0 : i32
  %one_i32 = arith.constant 1 : i32
  %status = arith.select %ok, %zero, %one_i32 : i32
  return %status : i32
}}
'''
        packed = run([thread, '--ir-deep-optimize', '-'], fixture)
        for program in (fixture, packed):
            low = run([llvm/'mlir-opt', '--lower-affine', '--convert-scf-to-cf',
                '--convert-vector-to-llvm', '--finalize-memref-to-llvm',
                '--convert-arith-to-llvm', '--convert-func-to-llvm',
                '--convert-cf-to-llvm', '--reconcile-unrealized-casts', '-'],
                program.replace(', 3>', '>'))
            code = run([llvm/'mlir-translate', '--mlir-to-llvmir', '-'], low)
            run([llvm/'lli', '-'], code)
    print('PASS: bounded address lifetimes, memory/math order, idempotence, O3 anchors, scheduled LDS packing and write hazards')


if __name__ == '__main__':
    main()
