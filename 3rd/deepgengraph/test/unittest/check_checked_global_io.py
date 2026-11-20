#!/usr/bin/env python3
"""Exercise checked scalar/vector IO, including partial packets and negative indices."""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from check_attention_llvm_numerics import LLVMWorkgroupModel

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / '3rd/deepgengraph/build/test/ThreadTilingTest'
TRANSLATE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data2/xsl/install/bin/mlir-translate')


def main():
    with tempfile.TemporaryDirectory(prefix='checked-global-io-') as tmp:
        for shift in (-1, 0, 1):
            source = f'''module {{
              func.func @Attn_p2(%q: memref<64xf16, 1>, %k: memref<5xf16, 1>,
                  %v: memref<64xf16, 1>, %o: memref<64xf16, 1>)
                  attributes {{thread_num = 64 : i32, frisk.checked_global_io}} {{
                %tid = gpu.thread_id x
                %four = arith.constant 4 : index
                %shift = arith.constant {shift} : index
                %base = arith.muli %tid, %four : index
                %offset = arith.addi %base, %shift : index
                %packet = vector.load %k[%offset] : memref<5xf16, 1>, vector<4xf16>
                vector.store %packet, %o[%offset] : memref<64xf16, 1>, vector<4xf16>
                return
              }}
            }}'''
            proc = subprocess.run([str(TOOL), '--verify-each', '--finalize-thread-tiling',
                '--canonicalize', '--cse', '--test-async-to-llvm'], input=source,
                text=True, capture_output=True, cwd=tmp, timeout=60)
            assert proc.returncode == 0, proc.stderr
            translated = subprocess.run([str(TRANSLATE), '--mlir-to-llvmir'], input=proc.stdout,
                text=True, capture_output=True, timeout=30)
            assert translated.returncode == 0, translated.stderr
            # The general attention model expects a shared allocation. This
            # standalone IO fixture needs none, so provide an unused one.
            text = '@shm_0 = addrspace(3) global [128 x i8] undef\n' + translated.stdout
            zeros = np.zeros(64, dtype=np.float16)
            model = LLVMWorkgroupModel(text, zeros, np.arange(5, dtype=np.float16) + 10,
                                       zeros, threads=64)
            expected = zeros.copy()
            expected[:5] = np.arange(5) + 10
            if shift > 0: expected[:shift] = np.nan
            np.testing.assert_array_equal(model.run(), expected)
            print('PASS guarded vector packets, scalar fallback and direct load/store:', shift)


if __name__ == '__main__':
    main()
