MaskAddBypassPass
=================

LLVM FunctionPass: include/deepgengraph/Conversion/ConvertToLLVM/MaskAddBypass.h
Implementation: lib/Conversion/ConvertToLLVM/MaskAddBypass.cpp

Transform an existing zero-mask branch:
  mask = phi(clear: zero, masked: values)
  result = score + extract(mask)
into:
  masked: partial = score + extract(values)
  join: result = phi(clear: score, masked: partial)

No assumptions about function names, tile dimensions, thread count, MMAC
instruction shape, or target GPU. The pass reuses the existing branch; it
does not prove new causal/padding/window mask bounds. No CFG changes, memory
operations, calls, DMA or barriers are moved. Original addition order and
operand order are retained in the masked arm. No global fast-math is enabled.

Modes (default off):
  off:    no transformation.
  legal:  LLVM simplifyFAddInst must prove score+zero equals score using the
          existing FP flags/environment (e.g. nsz or a proven score property).
  approx: additionally allow ignoring signed zero in matched mask additions.
          This is explicit numerical permission, not an inferred consequence
          of a missing strictfp attribute. StrictFP/OptNone are always skipped.
          Constrained FP is unsupported. Infinity is not assumed absent:
          -inf masks and online-softmax running max are retained.

First-version bounds:
- Two-input PHI with a constant-zero incoming value; no backedges.
- Masked predecessor has only the merge as successor.
- Consumers in the merge or a short chain of single-predecessor successor
  blocks (up to four blocks); no walking into loops or divergent control flow.
- Float scalar/fixed-vector additions reached through extractvalue,
  extractelement and shufflevector. Constant folding must prove every consumed
  clear-arm lane is zero, including poison/undef handling.
- Score dominates both incoming edges. All mask uses must belong to supported
  views/additions; escaping/mixed consumers, dynamic indexing and unsupported
  expressions cause the entire candidate to fall back. At most 256 matched
  instructions, with bounded view recursion. Repeated application is idempotent.

Standalone A/B on an existing exported IR:
  build/test/LegacyLLVMExportTest --mask-add-bypass=approx input.ll > output.ll
This entry point runs only this pass and verification, with no unrelated
export canonicalization. It accepts off and legal for controls.

MyTest named option:
  --mask-add-bypass=off|legal|approx
The approx mode requires the existing positional math mode to be approx.
The pass runs after register math/MMAC fusion, before global prefetch and
register-lifetime scheduling. Other existing presets retain off.

From repository root, a complete compiler export:
  python3 TuneAttention.py \
    --input generated_ir/online_softmax/test_input_1x32x4096x128-128-128-32-online.mlir \
    --case online-baseline --set mask-add-bypass=approx \
    --llvm-as /data2/xsl/install/bin/llvm-as \
    --output-dir generated_ir/mask_add_bypass_tuning

TuneAttention supports --sweep 'mask-add-bypass=off;legal;approx'. Invalid
approx/strict-math combinations are filtered. result.json records the pass
mode and matched groups/additions in mask_add_bypass_reports.

Build from repository root:
  cmake --build 3rd/deepgengraph/build --target LegacyLLVMExportTest MyTest -j 2
Regression:
  python3 3rd/deepgengraph/test/check_mask_add_bypass.py \
    3rd/deepgengraph/build/test/LegacyLLVMExportTest
Attention checks use test/check_online_softmax.py with --bn 128|64 --group 4.
GPU correctness/performance of newly compiled variants requires gfx936 tests.
