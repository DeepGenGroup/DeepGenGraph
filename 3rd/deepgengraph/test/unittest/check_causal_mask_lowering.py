#!/usr/bin/env python3
"""Range-guarded causal masks: both lowering paths and conservative fallbacks."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parent/'build/test/ThreadTilingTest'
SOURCE = (HERE/'test_friskBase.mlir').read_text()
SOURCE = re.sub(r'%25 = scf.if %24 -> \(f32\) \{.*?\} else \{.*?\n        \}',
                '%25 = arith.select %24, %cst_0, %cst : f32',SOURCE,flags=re.S)


def run(source, mode, typed=False):
    with tempfile.TemporaryDirectory(prefix='causal-mask-') as tmp:
        src, out = Path(tmp)/'in.mlir',Path(tmp)/'out.mlir'
        src.write_text(source)
        options = f'causal-mask={mode} fuse-mask-add=false uniform-masks=false'
        cmd = [TOOL,src,'--convert-friskbase-to-'+('fragment' if typed else 'thread')+'='+options]
        if typed:
            cmd += ['--convert-fragment-to-thread']
        result=subprocess.run([*cmd,'--verify-each','-o',out],text=True,capture_output=True)
        assert result.returncode==0,result.stderr[-6000:]
        return out.read_text(),result.stderr


def main():
    for typed in (False,True):
        for mode in ('off','i32','finite'):
            out,log=run(SOURCE,mode,typed)
            assert ('frisk.causal_mask_i32' in out)==(mode!='off'),(typed,mode,log)
            assert ('frisk.causal_mask_select' in out)==(mode=='finite'),(typed,mode,log)
            if mode=='i32':
                assert 'frisk.add' in out,'exact mode must retain score + mask'
            print('PASS',typed,mode,flush=True)
        lhs=SOURCE.replace('%16 = frisk.add %qk, %maskTIle','%16 = frisk.add %maskTIle, %qk')
        assert 'frisk.causal_mask_select' in run(lhs,'finite',typed)[0]
        shared=SOURCE.replace(
            '%16 = frisk.add %qk, %maskTIle : memref<64x32xf32>, memref<64x32xf32> -> memref<64x32xf32>',
            '%16 = frisk.add %qk, %maskTIle : memref<64x32xf32>, memref<64x32xf32> -> memref<64x32xf32, 3>')
        shared=shared.replace('%qkme = frisk.exp2 %16 : memref<64x32xf32>',
                              '%qkme = frisk.exp2 %16 : memref<64x32xf32, 3>')
        assert 'frisk.causal_mask_select' in run(shared,'finite',typed)[0]
        print('PASS consumer memory space',typed,flush=True)
        renamed=re.sub(r'%([A-Za-z][\w]*)',r'%renamed_\1',SOURCE)
        assert 'frisk.causal_mask_select' in run(renamed,'finite',typed)[0]
        multi=SOURCE.replace('%16 = frisk.add',
            '%extra = frisk.add %qk, %maskTIle : memref<64x32xf32>, memref<64x32xf32> -> memref<64x32xf32>\n      %16 = frisk.add')
        # Keep the second use live, so dead-code elimination cannot restore a
        # sole consumer before the matching check.
        multi=multi.replace('%qkme = frisk.exp2 %16',
            '%sum = frisk.add %extra, %16 : memref<64x32xf32>, memref<64x32xf32> -> memref<64x32xf32>\n      %qkme = frisk.exp2 %sum')
        out,_=run(multi,'finite',typed)
        assert 'frisk.causal_mask_i32' in out and 'frisk.causal_mask_select' not in out
        barrier=SOURCE.replace('%16 = frisk.add','gpu.barrier\n      %16 = frisk.add')
        out,_=run(barrier,'finite',typed)
        assert 'frisk.causal_mask_i32' in out and 'frisk.causal_mask_select' not in out
        cases={
            'signed':SOURCE.replace('cmpi ule','cmpi sle'),
            'negative_diagonal':SOURCE.replace('%c1 = arith.constant 1 : index','%c1 = arith.constant -1 : index'),
            'large_diagonal':SOURCE.replace('%c1 = arith.constant 1 : index','%c1 = arith.constant 2147483647 : index'),
            'missing_range':SOURCE.replace('gpu.block_id  y {range = 64 : index}','gpu.block_id  y'),
            'negative_start':SOURCE.replace('arith.muli %block_id_y, %c64','arith.subi %block_id_y, %c64'),
            'large_start':SOURCE.replace('%c64 = arith.constant 64 : index','%c64 = arith.constant 2147483647 : index'),
            'overflow_start':SOURCE.replace('%c64 = arith.constant 64 : index','%c64 = arith.constant 9223372036854775807 : index'),
        }
        for name,source in cases.items():
            out,log=run(source,'finite',typed)
            assert 'frisk.causal_mask_i32' not in out,(typed,name,log)
            assert 'frisk.causal_mask_select' not in out,(typed,name,log)
            print('PASS fallback',typed,name,flush=True)
        for name,replacement in [('finite_mask','-1.000000e+00'),('positive_inf','0x7F800000')]:
            source=SOURCE.replace('0xFF800000',replacement)
            assert source!=SOURCE
            out,_=run(source,'finite',typed)
            assert 'frisk.causal_mask_i32' in out and 'frisk.causal_mask_select' not in out,(typed,name)
        # Reordering is evaluated through the actual generated arithmetic, not
        # an unchecked range attribute attached to the reversed coordinate.
        reverse=SOURCE.replace('%0 = arith.muli %block_id_y, %c64 : index',
            '%last = arith.constant 63 : index\n    %rev = arith.subi %last, %block_id_y : index\n'
            '    %0 = arith.muli %rev, %c64 : index')
        assert 'frisk.causal_mask_select' in run(reverse,'finite',typed)[0]
    print('PASS all causal mask range/semantics guards',flush=True)


if __name__=='__main__':
    main()
