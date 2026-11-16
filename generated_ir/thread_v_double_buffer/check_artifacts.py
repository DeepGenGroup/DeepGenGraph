#!/usr/bin/env python3
"""Check final hashes, serialized Thread pass, idempotence and fixed capacity."""
from pathlib import Path
import subprocess
import json
import hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TOOL=ROOT/'3rd/deepgengraph/build/test/ThreadTilingTest'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    validation=json.loads((HERE/'validation.json').read_text())
    records={}
    for kind in ['causal','noncausal']:
        before=HERE/(kind+'.thread.before.mlir')
        replay=HERE/(kind+'.thread.replayed.mlir')
        twice=HERE/(kind+'.thread.twice.mlir')
        # Compare one/two runs from the same parse: reparsing large affine maps
        # itself changes their printed term order, independently of this pass.
        for count,dst in [(1,replay),(2,twice)]:
            pipeline='--pass-pipeline=builtin.module(func.func('+','.join(['frisk-thread-double-buffer-v']*count)+'))'
            subprocess.run([str(TOOL),str(before),pipeline,'--verify-each','-o',str(dst)],check=True)
        assert replay.read_bytes()==twice.read_bytes(),kind+' not idempotent'
        assert before.read_text().count('gpu.barrier')-replay.read_text().count('gpu.barrier')==3
        assert replay.read_text().count('frisk.v_double_buffer')==4
        assert replay.read_text().count('memref.alloc')==1
        info=json.loads((HERE/(kind+'.generation.json')).read_text())
        assert digest(ROOT/'3rd/deepgengraph/build/test/MyTest')==info['mytest_sha256']
        for variant,suffix in [('thread','.ll'),('thread-o3','.o3.ll')]:
            path=HERE/(kind+suffix)
            assert digest(path) in {c['sha256'] for c in validation['cases'] if c['kind']==kind and c['variant']==variant}
            assert '@shm_0 = addrspace(3) global [16384 x i8]' in path.read_text()
        records[kind]=dict(serialized_thread_pass_verified_and_idempotent=True,final_artifacts_match_numerically_validated_hashes=True,final_compiler_hash_matches=True,static_LDS_bytes=16384)
        print('PASS',kind,records[kind])
    (HERE/'final-checks.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
