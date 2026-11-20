#!/usr/bin/env python3
"""Whole-function pressure: MLIR CFG liveness, regions, costs and invalidation."""
from pathlib import Path
import re
import subprocess
import sys

TOOL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'build/test/ThreadTilingTest'


def run(source, *flags):
    p = subprocess.run([str(TOOL), '--mlir-disable-threading', '--verify-each', *flags],
                       input=source, text=True, capture_output=True, timeout=90)
    assert p.returncode == 0, p.stderr
    return p.stdout, p.stderr


def analyze(source):
    baseline, _ = run(source)
    actual, log = run(source, '--frisk-analyze-register-pressure=all-points=true top=3')
    assert actual == baseline, 'analysis changed IR'
    _, replay = run(source, '--frisk-analyze-register-pressure=all-points=true top=3')
    assert log == replay, 'unstable report'
    s = summary(log)
    assert s['peak-ideal-packed-data'] <= s['peak-packed-data'] <= s['peak-data'], log
    return log


def summary(log):
    return {k: int(v) for k, v in re.findall(r'([a-z-]+)=(\d+)',
            re.search(r'^  summary:.*$', log, re.M)[0])}


def points(log, op):
    return [line for line in log.splitlines() if line.startswith('  point ') and f'op={op} ' in line]


for ty, data, packed in [('f32', 2, 2), ('f64', 4, 4),
                          ('vector<4xf32>', 8, 8), ('vector<4xf16>', 8, 4)]:
    log = analyze(f'''func.func @add(%a: {ty}, %b: {ty}) -> {ty} {{
      %r = arith.addf %a, %b : {ty}
      return %r : {ty}
    }}''')
    s = summary(log)
    assert s['peak-data'] == data and s['peak-packed-data'] == packed, log
    assert s['peak-overlap'] == data * 3 // 2, log

# Optimistic packing rounds once AFTER combining independent origins, not once
# per SSA. Arithmetic output is still a fresh origin at the overlap boundary.
log = analyze('''func.func @half_add(%a: f16, %b: f16) -> f16 {
  %r = arith.addf %a, %b : f16
  return %r : f16
}''')
s = summary(log)
assert s['peak-data'] == s['peak-packed-data'] == 2, log
assert s['peak-ideal-packed-data'] == 1, log
assert s['peak-ideal-packed-data-overlap'] == 2, log

for count in [2, 3, 5]:
    args = ', '.join(f'%a{i}: f16' for i in range(count))
    operands = ', '.join(f'%a{i}' for i in range(count))
    types = ', '.join('f16' for i in range(count))
    log = analyze(f'''func.func @halves({args}) -> ({types}) {{
      return {operands} : {types}
    }}''')
    s = summary(log)
    assert s['peak-data'] == s['peak-packed-data'] == count, log
    assert s['peak-ideal-packed-data'] == (count + 1) // 2, log

# Assembly and simultaneous aliases do not add bits to the optimistic estimate.
log = analyze('''func.func @half_views(%a: f16, %b: f16) -> (f16, f16, vector<2xf16>) {
  %v = vector.from_elements %a, %b : vector<2xf16>
  return %a, %b, %v : f16, f16, vector<2xf16>
}''')
assert summary(log)['peak-ideal-packed-data'] == 1, log

# Proven representation aliases share one payload even with simultaneous users.
log = analyze('''
!f = !frisk.fragment<vector<4xf32>, {}>
func.func @views(%x: vector<4xf32>) -> (vector<4xf32>, !f, vector<2x2xf32>) {
  %f = frisk.pack_frag %x : vector<4xf32> -> !f
  %v = frisk.unpack_frag %f : !f -> vector<4xf32>
  %s = vector.shape_cast %v : vector<4xf32> to vector<2x2xf32>
  return %x, %f, %s : vector<4xf32>, !f, vector<2x2xf32>
}''')
assert summary(log)['peak-live'] == 4 and summary(log)['peak-overlap'] == 4, log

# A subvector shares its selected elements, even when the whole source is live.
log = analyze('''func.func @extract(%x: vector<2x4xf32>) -> (vector<2x4xf32>, vector<4xf32>) {
  %s = vector.extract %x[0] : vector<4xf32> from vector<2x4xf32>
  return %x, %s : vector<2x4xf32>, vector<4xf32>
}''')
assert summary(log)['peak-data'] == 8 and summary(log)['ssa-peak-data'] == 12, log

# Insert is a new SSA version, but only changed elements introduce new content.
# Retaining the original vector must keep its overwritten elements too.
for keep_old in [False, True]:
    outputs = '%x, %r' if keep_old else '%r'
    types = 'vector<4xf32>, vector<4xf32>' if keep_old else 'vector<4xf32>'
    log = analyze(f'''func.func @insert(%x: vector<4xf32>, %s: f32) -> ({types}) {{
      %r = vector.insert %s, %x[2] : f32 into vector<4xf32>
      return {outputs} : {types}
    }}''')
    line = points(log, 'vector.insert')[0]
    assert 'before=5 (data=5' in line, log
    assert f'after={5 if keep_old else 4} (data=' in line, log
    assert 'overlap=5 ' in line, log

# An old scalar remains live even if only the updated vector survives.
log = analyze('''func.func @old_element(%x: vector<4xf32>, %s: f32) -> (f32, vector<4xf32>) {
  %old = vector.extract %x[2] : f32 from vector<4xf32>
  %r = vector.insert %s, %x[2] : f32 into vector<4xf32>
  return %old, %r : f32, vector<4xf32>
}''')
assert 'after=5 (data=5' in points(log, 'vector.insert')[0], log

# Fragment indices are fragment-grid coordinates: [1,1] selects elements
# [1,2:4], not [1,1:3]. Restoring that slice must reproduce exactly x's content.
log = analyze('''
!f = !frisk.fragment<vector<1x2xf32>, {}>
func.func @fragment_roundtrip(%x: vector<2x4xf32>) -> (vector<2x4xf32>, vector<2x4xf32>) {
  %c1 = arith.constant 1 : index
  %f = frisk.extract_frag %x[%c1, %c1] : vector<2x4xf32> -> <vector<1x2xf32>, {}>
  %v = frisk.unpack_frag %f : !f -> vector<1x2xf32>
  %a = vector.extract %v[0, 0] : f32 from vector<1x2xf32>
  %r0 = vector.insert %a, %x[1, 2] : f32 into vector<2x4xf32>
  %r = frisk.insert_frag %f, %r0[%c1, %c1] : <vector<1x2xf32>, {}> into vector<2x4xf32>
  return %x, %r : vector<2x4xf32>, vector<2x4xf32>
}''')
assert summary(log)['peak-data'] == 8 and summary(log)['conservative-inserts'] == 0, log

# Check the actual selected coordinate after the original vector dies. Treating
# fragment index 1 as element offset 1 would select an old x element here.
log = analyze('''
!f = !frisk.fragment<vector<1x2xf32>, {}>
func.func @fragment_coordinate(%x: vector<2x4xf32>, %f: !f) -> (f32, !f) {
  %c1 = arith.constant 1 : index
  %r = frisk.insert_frag %f, %x[%c1, %c1] : <vector<1x2xf32>, {}> into vector<2x4xf32>
  %a = vector.extract %r[1, 3] : f32 from vector<2x4xf32>
  return %a, %f : f32, !f
}''')
assert 'before=2 (data=2' in points(log, 'func.return')[0], log

# A two-row fragment is not contiguous in a wider tile. The extracted row
# contains two old elements and two patch elements; the full patch adds only
# its other two elements, for six distinct words altogether.
log = analyze('''
!f = !frisk.fragment<vector<2x2xf32>, {}>
func.func @fragment_rows(%x: vector<4x4xf32>, %f: !f) -> (vector<4xf32>, !f) {
  %c1 = arith.constant 1 : index
  %r = frisk.insert_frag %f, %x[%c1, %c1] : <vector<2x2xf32>, {}> into vector<4x4xf32>
  %row = vector.extract %r[3] : vector<4xf32> from vector<4x4xf32>
  return %row, %f : vector<4xf32>, !f
}''')
assert 'before=6 (data=6' in points(log, 'func.return')[0], log

# Overlapping slices, bitcasts and from_elements must deduplicate shared bits.
log = analyze('''func.func @bits(%x: vector<2xi32>) -> (vector<2xi32>, vector<4xf16>, vector<2xf16>) {
  %v = vector.bitcast %x : vector<2xi32> to vector<4xf16>
  %a = vector.extract %v[0] : f16 from vector<4xf16>
  %b = vector.extract %v[1] : f16 from vector<4xf16>
  %r = vector.from_elements %a, %b : vector<2xf16>
  return %x, %v, %r : vector<2xi32>, vector<4xf16>, vector<2xf16>
}''')
assert summary(log)['peak-data'] == 2 and summary(log)['peak-packed-data'] == 2, log
for second, expected in [(1, 1), (3, 2)]:
    log = analyze(f'''func.func @partial_bits(%x: vector<1xi64>) -> (i16, i16) {{
      %v = vector.bitcast %x : vector<1xi64> to vector<4xi16>
      %a = vector.extract %v[0] : i16 from vector<4xi16>
      %b = vector.extract %v[{second}] : i16 from vector<4xi16>
      return %a, %b : i16, i16
    }}''')
    assert f'before={expected} (data={expected}, packed-data=1' in points(log, 'func.return')[0], log

# A new computation cannot be aliased with its input just because a later
# insert overwrites that input's position. Check the actual arithmetic overlap.
log = analyze('''func.func @update(%x: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %a = vector.extract %x[0] : f32 from vector<4xf32>
  %b = arith.addf %a, %a : f32
  %r = vector.insert %b, %x[0] : f32 into vector<4xf32>
  return %x, %r : vector<4xf32>, vector<4xf32>
}''')
assert 'overlap=5 ' in points(log, 'arith.addf')[0], log
assert summary(log)['peak-data'] == 5, log

# Dynamic selectors are covered by a surviving complete source. Once it dies,
# different unresolved selections are independently charged (may be disjoint).
for keep_source in [False, True]:
    outputs = '%x, %a, %b' if keep_source else '%a, %b'
    types = 'vector<4xf32>, f32, f32' if keep_source else 'f32, f32'
    log = analyze(f'''func.func @dynamic(%x: vector<4xf32>, %i: index, %j: index) -> ({types}) {{
      %a = vector.extract %x[%i] : f32 from vector<4xf32>
      %b = vector.extract %x[%j] : f32 from vector<4xf32>
      return {outputs} : {types}
    }}''')
    assert f'before={4 if keep_source else 2} (data=' in points(log, 'func.return')[0], log
    assert summary(log)['unresolved-extracts'] == 2, log

# Dynamic fragment inserts retain dest+input as a conservative content union.
# They must not invent another full destination. A subsequent extraction from
# that unresolved mapping is a subset too, including after packing/unpacking.
log = analyze('''
!f = !frisk.fragment<vector<1x2xf32>, {}>
func.func @dynamic_fragment(%x: vector<2x4xf32>, %f: !f, %i: index, %j: index)
    -> (vector<2x4xf32>, vector<2x4xf32>, !f) {
  %r = frisk.insert_frag %f, %x[%i, %j] : <vector<1x2xf32>, {}> into vector<2x4xf32>
  %g = frisk.extract_frag %r[%i, %j] : vector<2x4xf32> -> <vector<1x2xf32>, {}>
  return %x, %r, %g : vector<2x4xf32>, vector<2x4xf32>, !f
}''')
assert 'before=10 (data=10' in points(log, 'func.return')[0], log
assert summary(log)['conservative-inserts'] == 1, log

# Multiple updates of a constant vector do not charge untouched constant lanes.
log = analyze('''func.func @assemble(%a: f32, %b: f32) -> vector<4xf32> {
  %c = arith.constant dense<0.0> : vector<4xf32>
  %v = vector.insert %a, %c[0] : f32 into vector<4xf32>
  %r = vector.insert %b, %v[1] : f32 into vector<4xf32>
  return %r : vector<4xf32>
}''')
assert summary(log)['peak-data'] == 2, log
log = analyze('''func.func @splat(%a: f32) -> (f32, vector<4xf32>) {
  %r = vector.broadcast %a : f32 to vector<4xf32>
  return %a, %r : f32, vector<4xf32>
}''')
assert summary(log)['peak-data'] == 1 and summary(log)['ssa-peak-data'] == 5, log

# Global/shared memory and their descriptor diagnostics do NOT enter pressure.
# Changing the allocation size cannot change register demand.
for space in [1, 3]:
    for size in [4, 65536]:
        log = analyze(f'''func.func @buffer(%a: memref<{size}x{size}xf32, {space}>) -> memref<{size}x{size}xf32, {space}> {{
          return %a : memref<{size}x{size}xf32, {space}>
        }}''')
        s = summary(log)
        assert s['peak-live'] == s['peak-overlap'] == s['peak-data'] == 0, log
        assert s['peak-addresses'] == s['peak-ideal-packed-data'] == 0, log
        assert s['peak-descriptor-diagnostic'] == s['peak-with-descriptors'] == 14, log

# Loaded values do occupy data slots; their backing memory never does. The two
# independent f16 loads fit one word only in the explicitly optimistic model.
log = analyze('''func.func @loads(%g: memref<65536x65536xf16, 1>, %s: memref<4x4xf16, 3>) -> (f16, f16) {
  %c0 = arith.constant 0 : index
  %x = memref.load %g[%c0, %c0] : memref<65536x65536xf16, 1>
  %y = memref.load %s[%c0, %c0] : memref<4x4xf16, 3>
  return %x, %y : f16, f16
}''')
assert summary(log)['peak-live'] == summary(log)['peak-data'] == 2, log
assert summary(log)['peak-ideal-packed-data'] == 1, log
assert summary(log)['peak-addresses'] == 0, log

# Actual pointer SSA values and dynamic address arithmetic remain chargeable.
log = analyze('''func.func @addresses(%p: !llvm.ptr<1>, %i: i64) -> (!llvm.ptr<1>, !llvm.ptr<1>) {
  %q = llvm.getelementptr %p[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  return %p, %q : !llvm.ptr<1>, !llvm.ptr<1>
}''')
s = summary(log)
assert s['peak-live'] == 4 and s['peak-overlap'] == 6, log
assert s['peak-addresses'] == 4 and s['peak-data'] == 2, log
assert s['peak-descriptor-diagnostic'] == 0, log

# Constants/dependency tokens do not occupy persistent data slots.
log = analyze('''func.func @constant() -> vector<64xf32> {
  %c = arith.constant dense<0.0> : vector<64xf32>
  return %c : vector<64xf32>
}''')
assert summary(log)['peak-live'] == 0, log
log = analyze('''func.func @token(%x: !frisk.async_task) -> !frisk.async_task {
  return %x : !frisk.async_task
}''')
assert summary(log)['max-unknown-live'] == 0 and summary(log)['peak-live'] == 0, log

# Early wave lowering already exposes LLVM descriptor aggregates in ThreadIR.
# Full aggregates remain diagnostic until their fields become explicit SSA.
log = analyze('''func.func @descriptor(%x: !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>)
    -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {
  return %x : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
}''')
assert summary(log)['peak-descriptor-diagnostic'] == 14 and summary(log)['max-unknown-live'] == 0, log
assert summary(log)['peak-live'] == summary(log)['peak-ideal-packed-data'] == 0, log
log = analyze('''func.func @descriptor_fields(%x: !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>)
    -> (!llvm.ptr<1>, i64) {
  %p = llvm.extractvalue %x[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  %i = llvm.extractvalue %x[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  return %p, %i : !llvm.ptr<1>, i64
}''')
s = summary(log)
assert s['peak-live'] == 4 and s['peak-data'] == s['peak-addresses'] == 2, log
assert s['peak-descriptor-diagnostic'] == 14 and s['peak-ideal-packed-data'] == 2, log
log = analyze('''func.func @aggregate(%x: !llvm.struct<(vector<4xf16>, array<2 x f64>)>)
    -> !llvm.struct<(vector<4xf16>, array<2 x f64>)> {
  return %x : !llvm.struct<(vector<4xf16>, array<2 x f64>)>
}''')
assert summary(log)['peak-data'] == 8 and summary(log)['peak-packed-data'] == 6, log
assert summary(log)['peak-ideal-packed-data'] == 6, log
log = analyze('''func.func @mixed(%x: !llvm.struct<(f16, f16, ptr<1>)>) -> !llvm.struct<(f16, f16, ptr<1>)> {
  return %x : !llvm.struct<(f16, f16, ptr<1>)>
}''')
s = summary(log)
assert s['peak-live'] == 4 and s['peak-data'] == s['peak-addresses'] == 2, log
assert s['peak-packed-data'] == 2 and s['peak-ideal-packed-data'] == 1, log
assert s['peak-descriptor-diagnostic'] == 0, log

# Index sizes come from MLIR DataLayout, not a hardcoded pointer-width guess.
log = analyze('''module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>} {
  func.func @indices(%a: index, %b: index) -> index {
    %r = arith.addi %a, %b : index
    return %r : index
  }
}''')
assert summary(log)['peak-data'] == 2, log

# Structured loop: held is only used AFTER the loop, cap is only used near the
# START of each iteration. Both must remain live at the later multiplication.
# The seed and the loop's result must not be charged alongside the body carrier.
for loop, yield_op in [('affine.for %i = 0 to 4', 'affine.yield'),
                       ('scf.for %i = %c0 to %c4 step %c1', 'scf.yield')]:
    log = analyze(f'''func.func @loop(%seed: vector<4xf32>, %cap: vector<4xf32>, %held: vector<8xf32>)
        -> (vector<4xf32>, vector<8xf32>) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %r = {loop} iter_args(%acc = %seed) -> vector<4xf32> {{
        %a = arith.addf %acc, %cap : vector<4xf32>
        %b = arith.mulf %a, %a : vector<4xf32>
        {yield_op} %b : vector<4xf32>
      }}
      return %r, %held : vector<4xf32>, vector<8xf32>
    }}''')
    line = points(log, 'arith.mulf')[0]
    assert 'before=16 (data=16' in line and 'after=16 (data=16' in line, log
    assert summary(log)['peak-live'] == 16, log

# Nested region liveness also includes an outer carrier used after an inner loop.
log = analyze('''func.func @nested(%seed: vector<4xf32>, %held: vector<8xf32>)
    -> (vector<4xf32>, vector<8xf32>) {
  %r = affine.for %i = 0 to 4 iter_args(%outer = %seed) -> vector<4xf32> {
    %s = affine.for %j = 0 to 4 iter_args(%inner = %outer) -> vector<4xf32> {
      %a = arith.mulf %inner, %inner : vector<4xf32>
      affine.yield %a : vector<4xf32>
    }
    %t = arith.addf %s, %outer : vector<4xf32>
    affine.yield %t : vector<4xf32>
  }
  return %r, %held : vector<4xf32>, vector<8xf32>
}''')
assert 'before=16 (data=16' in points(log, 'arith.mulf')[0], log

# Region results don't exist inside a branch. Sibling temporaries never get
# summed together; a value used beyond the if still contributes inside it.
log = analyze('''func.func @branches(%c: i1, %x: vector<4xf32>, %held: vector<8xf32>)
    -> (vector<4xf32>, vector<8xf32>) {
  %r = scf.if %c -> vector<4xf32> {
    %a = arith.addf %x, %x : vector<4xf32>
    scf.yield %a : vector<4xf32>
  } else {
    %b = arith.mulf %x, %x : vector<4xf32>
    scf.yield %b : vector<4xf32>
  }
  return %r, %held : vector<4xf32>, vector<8xf32>
}''')
for op in ['arith.addf', 'arith.mulf']:
    assert 'before=12 (data=12' in points(log, op)[0], log
    assert 'after=12 (data=12' in points(log, op)[0], log

# CFG backedge: cap is referenced in the loop header and must remain live in
# the latch for the next iteration, even though the latch itself doesn't use it.
log = analyze('''func.func @cfg(%again: i1, %seed: vector<4xf32>, %cap: vector<4xf32>, %held: vector<8xf32>)
    -> (vector<4xf32>, vector<8xf32>) {
  cf.br ^header(%seed : vector<4xf32>)
^header(%acc: vector<4xf32>):
  %a = arith.addf %acc, %cap : vector<4xf32>
  cf.br ^latch(%a : vector<4xf32>)
^latch(%v: vector<4xf32>):
  %b = arith.mulf %v, %v : vector<4xf32>
  cf.cond_br %again, ^header(%b : vector<4xf32>), ^exit(%b : vector<4xf32>)
^exit(%out: vector<4xf32>):
  return %out, %held : vector<4xf32>, vector<8xf32>
}''')
assert 'before=17 (data=17' in points(log, 'arith.mulf')[0], log

for ty in ['vector<[4]xf32>', 'tensor<4xf32>']:
    log = analyze(f'func.func @unknown(%x: {ty}) -> {ty} {{ return %x : {ty} }}')
    assert summary(log)['max-unknown-live'] == 1, log
    assert f'unmodeled-type: {ty}' in log, log

# Verify PassManager dependency invalidation after an actual IR mutation.
source = '''func.func @duplicates(%a: vector<4xf32>, %b: vector<4xf32>)
    -> (vector<4xf32>, vector<4xf32>) {
  %x = arith.addf %a, %b : vector<4xf32>
  %y = arith.addf %a, %b : vector<4xf32>
  return %x, %y : vector<4xf32>, vector<4xf32>
}'''
_, log = run(source, '--pass-pipeline=builtin.module(func.func(frisk-analyze-register-pressure{stage=before},cse,frisk-analyze-register-pressure{stage=after}))')
reports = [r for r in log.split('EndRegisterPressure') if 'RegisterPressure:' in r]
assert len(reports) == 2, log
assert summary(reports[0])['peak-live'] == 12, log
assert summary(reports[1])['peak-live'] == 8, log
print('PASS: cross-SSA optimistic packing, global/shared memory excluded, descriptor diagnostics excluded, loaded data and explicit addresses counted, scalar/vector costs, static/dynamic fragment and vector slices, insert old/new coexistence, fragment coordinates, bitcast subwords, assembly, constants, DataLayout, loop captures/live-through, nested regions, branches, CFG backedges, unknowns, IR preservation, deterministic logs and analysis invalidation')
