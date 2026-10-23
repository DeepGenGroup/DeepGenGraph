Generic integer comparison compaction candidate

Implementation: OptimizeRegisterLifetimes.cpp, compactIntegerComparisons.
The existing register-lifetime export option now includes this optimization.
It also upgrades previously exported IR without repeating address/math passes.
No attention names, shapes, head counts, launch sizes or tensor roles are used.

For repeated scalar i64 predicates whose OR/XOR offsets leave the upper word
unchanged, compare the upper words once and compare local lower words as i32.
Signed order uses signed high-word and unsigned low-word comparisons. Equality
and inequality are supported as well. The complete i64 range is preserved;
this does not assume indices fit in 32 bits. Carrying arithmetic, small groups,
uniform scalar predicates, strictfp/optnone functions and non-AMDGPU functions
are left unchanged.
Floating-point adds and selects retain their original semantics, including -0,
NaN and infinities. No fast-math assumption is added.

Register-only MMAC export now also matches the HCU intrinsic effects:
  call ... asm "s_nop ... v_mmac_... s_nop ...", "=v,v,v,0"(...) #N
  attributes #N = { convergent nounwind willreturn memory(none) }
Only the recognized padded register-only MMAC loses asm sideeffect. NOPs,
operand order and tied accumulator constraints remain unchanged. DMA,
barriers and lifetime identities retain sideeffect. Accumulator fusion accepts
both old and new MMAC spellings. Check with check_register_mma_export.py.

Empty tied-register asm identities have unique unused immediate arguments.
They require no extra register for that argument and prevent O3 from merging
the lifetime boundaries. GPU register allocation and speed must still be
measured with the actual DTK; fewer wide comparisons is not a speedup claim.

Rebuild and regenerate from the repository root:
  cmake --build 3rd/deepgengraph/build --target LegacyLLVMExportTest MyTest -j 2
  bash generated_ir/mask_compare_compact/generate.sh

Independent semantic and codegen regression:
  python 3rd/deepgengraph/test/check_integer_comparison_lifetimes.py 3rd/deepgengraph/build/test/LegacyLLVMExportTest

Complete kernel mathematical model (not async timing/hazard validation):
  python 3rd/deepgengraph/test/check_attention_llvm_numerics.py generated_ir/mask_compare_compact/attention.ll

Build and run on the gfx936 machine with the same DTK and optimization level:
  bash GenHsaco.sh generated_ir/mask_compare_compact/attention.ll 3 generated_ir/mask_compare_compact/kernel.hsaco
  python kernelLauncher.py --hsaco generated_ir/mask_compare_compact/kernel.hsaco --kernel Attn_p2 --grid 32,32,1 --block 256,1,1 --k-layout bhsd --warmup 10 --repeat 50

Compare against the unchanged attn_best_async_current.ll using the same timing
method. Inspect target VGPR/scratch/occupancy and correctness before comparing
time. The supplied 2.184 ms is the user's profiler measurement; do not equate
it directly to a new event-timing median collected under different conditions.

Local checks passed: all ten integer predicates, OR/XOR on either/both sides,
406770 full-width integer/FP input cases before/after/after O3, idempotence,
fallbacks, gfx900 constraint/codegen legality, four complete attention-model
cases, and the existing register-math regression. No local gfx936 GPU/DTK.
The existing lifetime regression passes semantic/O3 checks but its LLVM 15
parser stage rejects LLVM 24's intrinsic return range attribute; this same
limitation is recorded in k5_matrix_optimized/validation.json.
