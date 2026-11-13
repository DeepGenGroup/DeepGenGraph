#!/usr/bin/env python3
"""Thread V double-buffer legality, capacity, ownership and idempotence."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def fixture(threads=64, stages=3, base=0, capacity=None, rounds=1):
    size=threads*16*rounds
    capacity=capacity if capacity is not None else max(base+size,2*size)
    elements=size//2
    lines=[f'module {{ func.func @kernel(%gm: memref<{stages}x{elements}xf16, 1>, %out: memref<4xi32, 1>) attributes {{thread_num = {threads} : i32}} {{',
           '%c0 = arith.constant 0 : index', '%c1 = arith.constant 1 : index',
           f'%base = arith.constant {base} : index', f'%spare = arith.constant {size if base == 0 else 0} : index',
           f'%pool = memref.alloc() alignment = 16 {{frisk.shm_pool}} : memref<{capacity}xi8, 3>',
           f'%slot = memref.view %pool[%base][] : memref<{capacity}xi8, 3> to memref<{elements}xf16, 3>',
           f'%other = memref.view %pool[%spare][] : memref<{capacity}xi8, 3> to memref<{elements}xf16, 3>',
           '%tid = gpu.thread_id x', '%idx = affine.apply affine_map<(d0) -> (d0 * 8)>(%tid)',
           'affine.for %iv = 0 to 4 {']
    def issue(stage):
        return [f'%src{stage} = memref.subview %gm[{stage}, 0] [1, {elements}] [1, 1] : memref<{stages}x{elements}xf16, 1> to memref<{elements}xf16, strided<[1], offset: {stage*elements}>, 1>',
                f'%task{stage} = frisk.async_copy_gm_to_shm %src{stage} to %slot {{alignment = 16 : i64, frisk.dma_bytes = 16 : i64}} : memref<{elements}xf16, strided<[1], offset: {stage*elements}>, 1>, memref<{elements}xf16, 3> -> !frisk.async_task']
    lines+=issue(0)
    for i in range(stages):
        lines += [f'frisk.wait_buffer_ready %task{i} : !frisk.async_task',
                  f'%read{i} = frisk.read_matrix_from_shm %slot[%idx] {{m = 32 : i64, n = 16 : i64, element_bit_width = 16 : i64}} : memref<{elements}xf16, 3> -> vector<4xi32>']
        if i+1<stages:lines+=['gpu.barrier']+issue(i+1)
    lines+=['gpu.barrier',f'vector.store %read{stages-1}, %out[%c0] : memref<4xi32, 1>, vector<4xi32>','}','return','} }']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--optimizer',type=Path,default=Path(__file__).resolve().parents[1]/'build/test/ThreadTilingTest')
    args=parser.parse_args()
    count=0
    with tempfile.TemporaryDirectory(prefix='thread-vdb-') as tmp:
        path=Path(tmp)/'input.mlir'
        def run(text,passes=1):
            path.write_text(text)
            cmd=[str(args.optimizer),str(path),'--pass-pipeline=builtin.module(func.func('+','.join(['frisk-thread-double-buffer-v']*passes)+'))','--verify-each']
            result=subprocess.run(cmd,text=True,capture_output=True)
            assert result.returncode==0,(result.stderr,text)
            return result.stdout,result.stderr
        def check(text,accepted=True,reason=None):
            nonlocal count
            result,log=run(text)
            changed='removed-barriers=' in log
            assert changed==accepted,(accepted,log,result)
            if accepted:
                assert 'frisk.v_double_buffer' in result
                assert result.count('memref.alloc')==text.count('memref.alloc')
                assert re.findall(r'memref<(\d+)xi8, 3>',result)[0]==re.findall(r'memref<(\d+)xi8, 3>',text)[0]
                again,_=run(result)
                assert again==result,'not idempotent'
                # DMA1 must precede the first read; current ready still precedes DMA1.
                operations=[line.strip() for line in result.splitlines()]
                wait=next(i for i,line in enumerate(operations) if 'frisk.wait_buffer_ready' in line)
                dma=next(i for i,line in enumerate(operations) if i>wait and 'frisk.async_copy_gm_to_shm' in line)
                read=next(i for i,line in enumerate(operations) if 'frisk.read_matrix_from_shm' in line)
                assert wait<dma<read
            else:
                plain=subprocess.run([str(args.optimizer),str(path)],text=True,capture_output=True,check=True).stdout
                assert result==plain,'rejected candidate was mutated'
                if reason:assert reason in log,log
            count+=1
        for threads,stages,rounds in [(64,2,1),(128,3,1),(256,4,2),(512,5,1),(1024,2,1)]:
            size=threads*16*rounds
            for base in (0,size):check(fixture(threads,stages,base,rounds=rounds))
        base=fixture()
        check(fixture(capacity=1024),False,'no spare')
        check(fixture(stages=1),False,'2..32')
        check(base.replace('gpu.barrier\nvector.store','vector.store'),False,'final reader')
        check(base.replace('gpu.barrier\n%src1','%src1'),False,'handoff')
        check(base.replace('gpu.barrier\n%src1','gpu.barrier\ngpu.barrier\n%src1'),False,'ambiguous')
        check(base.replace('%task0 =', '%old = memref.load %other[%idx] : memref<512xf16, 3>\n%task0 ='),False,'live across iteration')
        check(base.replace('gpu.barrier\nvector.store','gpu.barrier\n%late = memref.load %other[%idx] : memref<512xf16, 3>\nvector.store'),False,'surviving')
        check(base.replace('d0 * 8','d0 * 8 + 512'),False,'read outside')
        check(base.replace('to %slot {alignment','to %other {alignment',1),False,'read outside')
        check(base.replace('frisk.dma_bytes = 16','frisk.dma_bytes = 4'),False,'direct DMA')
        check(base.replace('alignment = 16 : i64','alignment = 4 : i64'),False,'direct DMA')
        check(base.replace('frisk.dma_bytes = 16 : i64','frisk.dma_bytes = 16 : i64, frisk.force_sync'),False,'direct DMA')
        check(base.replace('affine.for %iv = 0 to 4','affine.for %iv = 0 to %tid'),False,'non-uniform')
        check(base.replace('func.func @kernel','func.func private @unknown()\nfunc.func @kernel').replace('%read0 =','func.call @unknown() : () -> ()\n%read0 ='),False,'unknown operation')
        check(base.replace('%read0 =','%alias_read = memref.load %other[%idx] : memref<512xf16, 3>\n%read0 ='),False,'other live LDS')
        check(base.replace('%src1 =', '%indirect = memref.load %out[%c0] : memref<4xi32, 1>\n%address = arith.index_cast %indirect : i32 to index\n%src1 =').replace('%gm[1, 0]','%gm[%address, 0]').replace('offset: 512>','offset: ?>'),False,'direct DMA')
        check(base.replace('func.func @kernel(%gm:', 'func.func @kernel(%cond: i1, %gm:').replace('%read1 = frisk.read_matrix_from_shm %slot', '%selected = arith.select %cond, %slot, %other : memref<512xf16, 3>\n%read1 = frisk.read_matrix_from_shm %selected'), False, 'unknown matrix address')
        check(base.replace('affine.for %iv = 0 to 4 {', 'affine.for %q = 0 to 2 {\n} {frisk.pipelined}\naffine.for %iv = 0 to 4 {'))
        check(base.replace('}\nreturn', '} {pipeline.scheduled}\nreturn'), False, 'unknown operation')
        # Completed overwrite proves the old spare read iteration-local.
        init='%init = frisk.async_copy_gm_to_shm %src0 to %other {alignment = 16 : i64, frisk.dma_bytes = 16 : i64} : memref<512xf16, strided<[1], offset: 0>, 1>, memref<512xf16, 3> -> !frisk.async_task\nfrisk.wait_buffer_ready %init : !frisk.async_task\n%old = memref.load %other[%idx] : memref<512xf16, 3>\n'
        check(base.replace('%task0 =',init+'%task0 ='))
        pending=init.split('frisk.wait_buffer_ready',1)[0]
        check(base.replace('%task0 =',pending+'%task0 =').replace('gpu.barrier\nvector.store','gpu.barrier\nfrisk.wait_buffer_ready %init : !frisk.async_task\nvector.store'),False,'unknown alias/effect')
        check(base.replace('%task0 =', '%escaped = memref.memory_space_cast %other : memref<512xf16, 3> to memref<512xf16>\n%old = memref.load %escaped[%idx] : memref<512xf16>\n%task0 ='), False, 'unknown alias/effect')
    print(f'PASS {count} Thread V double-buffer cases (each success also checked for idempotence)')


if __name__=='__main__':main()
