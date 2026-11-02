Non-causal: prefetch next K0 during final PV fragment

Recommended IR: k0_prefetch_pv_tail.ll
Launch configuration: k0_prefetch_pv_tail.launch.json
Baseline: control.ll (identical to ../noncausal_qk_tail_prefetch_pass/k_tail_prefetch4.ll).
Matched control: k0_late_control.ll (same prologue, guard and next-address
construction, but K0 is issued after the final PV fragment).

Scope: gfx936, BHSD=(1,32,4096,128), BM=BN=128, BK=32, 256 threads,
non-causal, K swizzle64, Q retention32, QK tail LDS prefetch4, V double buffer.
The generator is deliberately scoped to the baseline SHA256 in manifest.json.

Schedule:
  entry: retained Q reads -> existing Q handoff -> prime K0(0)
  tile n: existing K0 wait/publication -> QK -> PV0 -> PV1 -> PV2
          -> existing V3 wait/publication (also releases V2 low-LDS readers)
          -> if n+1<32: issue K0(n+1) into low 8 KiB
          -> V3 reads high 8 KiB -> final PV's 32 MMAC
          -> existing final reader handoff -> next tile

The final iteration skips prefetch. All original hardware barriers, waits,
MMAC arithmetic, LDS reads, and non-K0 DMA remain in their original order.
Static LDS remains 16 KiB. Dynamic K DMA64=512, V DMA128=256, Q DMA128=8,
waits=260, hardware barriers=293, MMAC=8192 per workgroup. Across 31 tile
boundaries the IR exposes 992 MMAC while next K0 is pending. This is an IR
overlap window, not a measured hardware latency or speedup.

Generate and validate:
  python3 generated_ir/noncausal_cross_bn_k0/generate.py
  python3 generated_ir/noncausal_cross_bn_k0/verify.py

The generator checks LLVM assembly and O3 verification, preserves original
instruction order outside the explicitly changed K0/loop-control operations,
and checks K0 issue remains on the intended side of final PV after O3.
The verifier executes the actual IR on the CPU interpreter: random inputs,
first/last query blocks, maximum changes, a dominant final key tile, and
large positive/negative logits; it compares bitwise with control and checks
an FP64 reference. It audits ordered global accesses, LDS values and barrier
epochs, unchanged V scheduling, pending DMA, first/last K0, and issue timing.
Negative controls remove the V3 handoff, K0 wait, and last-iteration guard.
See validation.json for completed results. The CPU model does not simulate
independent wave timing or target register allocation. GPU correctness,
gfx936 assembly scheduling, resource usage, and performance need GPU testing.

On the existing DTK/GPU test environment:
  bash generated_ir/noncausal_cross_bn_k0/run_gpu.sh

Default order: control, prefetch, control, late-control, control. For a shorter
comparison:
  bash generated_ir/noncausal_cross_bn_k0/run_gpu.sh control k0_prefetch_pv_tail

The script uses GenHsaco.sh at O3 and launchKernel.py --non-causal; it writes
each build, HSACO, assembly, correctness/performance log and result JSON to a
separate runs/ directory. LLVM_DIR, ROCM_BC, GPU_ARCH can be supplied as with
GenHsaco.sh. ATTN_WARMUP defaults to 20; ATTN_REPEAT defaults to 100.
