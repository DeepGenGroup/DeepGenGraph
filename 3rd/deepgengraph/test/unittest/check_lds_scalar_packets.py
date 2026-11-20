#!/usr/bin/env python3
"""Exact shared scalar packets: reordered stores, reverse reads, and CPU semantics."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from check_shared_packing import cpu

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent/'build/test/FragmentOptTest'


def fixture(order=(2, 0, 3, 1), between='', space=3, elem='f32'):
    ty = f'memref<8x{elem}, {space}>'
    text = f'func.func @packet(%a: {ty}, %base: index) -> {elem} {{\n'
    for i in range(4):
        value = str(float(i+1)) if elem.startswith('f') else str(i+1)
        text += f'  %i{i} = affine.apply affine_map<(d0) -> (d0 + {i})>(%base)\n'
        text += f'  %v{i} = arith.constant {value} : {elem}\n'
    for j, i in enumerate(order):
        text += f'  memref.store %v{j}, %a[%i{i}] : {ty}\n'
        if j == 1:
            text += f'  {between}\n'
    for i in (3, 2, 1, 0):
        text += f'  %r{i} = memref.load %a[%i{i}] : {ty}\n'
    add = 'arith.addf' if elem.startswith('f') else 'arith.addi'
    mul = 'arith.mulf' if elem.startswith('f') else 'arith.muli'
    for i in (1, 2, 3):
        weight = str(float(2**i)) if elem.startswith('f') else str(2**i)
        text += f'  %w{i} = arith.constant {weight} : {elem}\n'
        text += f'  %weighted{i} = {mul} %r{i}, %w{i} : {elem}\n'
    text += f'  %s0 = {add} %r0, %weighted1 : {elem}\n'
    text += f'  %s1 = {add} %weighted2, %weighted3 : {elem}\n'
    text += f'  %sum = {add} %s0, %s1 : {elem}\n  return %sum : {elem}\n}}\n'
    return text


def main():
    with tempfile.TemporaryDirectory(prefix='lds-scalars-') as tmp:
        work = Path(tmp)
        for elem in ('f16', 'f32', 'i16', 'i32'):
            source = fixture(elem=elem)
            path, out = work/'input.mlir', work/'output.mlir'
            path.write_text(source)
            result = subprocess.run([str(TOOL), str(path), '--ir-deep-optimize=reorder-after-unroll=false',
                                     '--verify-each', '-o', str(out)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            text = out.read_text()
            assert re.search(r'vector.store[^\n]*vector<4x'+elem+'>', text), text
            assert text.count('memref.store') == 0
            assert text.count('memref.load') == 0
            # Every address, not just the sum, must retain its original value.
            ty = f'memref<8x{elem}, 3>'
            driver = f'func.func @main() -> i32 {{\n  %a = memref.alloc() : {ty}\n  %base = arith.constant 2 : index\n'
            driver += f'  %sum = func.call @packet(%a, %base) : ({ty}, index) -> {elem}\n'
            literal = '38.0' if elem.startswith('f') else '38'
            cmp = 'arith.cmpf oeq' if elem.startswith('f') else 'arith.cmpi eq'
            driver += f'  %expected = arith.constant {literal} : {elem}\n'
            driver += f'  %sum_ok = {cmp}, %sum, %expected : {elem}\n'
            prev = '%sum_ok'
            for j, value in enumerate((2, 4, 1, 3)):
                literal = str(float(value)) if elem.startswith('f') else str(value)
                cmp = 'arith.cmpf oeq' if elem.startswith('f') else 'arith.cmpi eq'
                driver += f'  %i{j} = arith.constant {j+2} : index\n  %v{j} = memref.load %a[%i{j}] : {ty}\n'
                driver += f'  %want{j} = arith.constant {literal} : {elem}\n  %ok{j} = {cmp}, %v{j}, %want{j} : {elem}\n'
                driver += f'  %all{j} = arith.andi {prev}, %ok{j} : i1\n'
                prev = f'%all{j}'
            driver += f'  %zero = arith.constant 0 : i32\n  %one = arith.constant 1 : i32\n  %status = arith.select {prev}, %zero, %one : i32\n'
            driver += f'  memref.dealloc %a : {ty}\n  return %status : i32\n}}\n'
            cpu(source+driver, work, elem+'-before')
            cpu(text[:text.rfind('}')]+driver+'\n}', work, elem+'-after')
            print('PASS CPU', elem, 'permuted stores, reverse reads, every coordinate')
        cases = {
            'duplicate': fixture(order=(0, 0, 2, 3)),
            'barrier': fixture(between='gpu.barrier'),
            'read_between': fixture(between='%observed = memref.load %a[%i2] : memref<8xf32, 3>\n  memref.store %observed, %a[%i3] : memref<8xf32, 3>'),
            'opaque': fixture(between='llvm.inline_asm has_side_effects "", "" : () -> ()'),
            'global': fixture(space=1),
            'dynamic_displacement': fixture().replace(
                'affine_map<(d0) -> (d0 + 3)>', 'affine_map<(d0) -> (d0 * 2 + 3)>'),
        }
        for name, source in cases.items():
            path.write_text(source)
            result = subprocess.run([str(TOOL), str(path), '--ir-deep-optimize=reorder-after-unroll=false',
                                     '--verify-each', '-o', str(out)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert not re.search(r'vector.store[^\n]*vector<4xf32>', out.read_text()), name
            print('PASS conservative boundary', name)


if __name__ == '__main__':
    main()
