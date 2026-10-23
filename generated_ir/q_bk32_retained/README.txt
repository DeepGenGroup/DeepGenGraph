One BK=32 Q fragment retained across BN iterations

baseline.ll: byte-identical copy of the measured best 1.779047 ms IR,
  generated_ir/group8_reverse_k_gep/group8_reverse_k128_gep.ll
q_bk32_retained.ll: retains the first Q fragment (D columns 0..31) across BN.
This experiment starts from the measured source, independently of the preceding
generic-LDS-address candidate, whose GPU performance has not been established.

Fixed configuration for both:
  BHSD=(1,32,4096,128), FP16 causal attention, contiguous BHSD Q/K/V/O
  BM=128, BN=128, BK=32; kernel=Attn_p2
  grid=(32,32,1), block=(256,1,1), dynamic shared=0
  Static LDS remains 16384 bytes; original group8 reverse mapping is preserved.
Use baseline.launch.json and q_bk32_retained.launch.json, respectively.

Change:
  Before the BN loop, copy Q[:,0:32] using the original two 128-bit DMA sites,
  wait and synchronize, read the original four <4 x half> lane fragments, and
  perform exactly the original FP16 scaling. These scaled register values are
  reused by the first QK stage on every BN iteration. Other Q fragments, K/V,
  score/PV math, mask and output normalization follow the original body.
  No extra shared allocation or per-BN retained-fragment branch is introduced.
  The original loop waits/barriers are retained for a conservative first A/B;
  one startup wait and barrier are added. Existing loop synchronization also
  prevents subsequent Q/V writes from overwriting the startup Q LDS data before
  all waves have read it. This experiment does not remove synchronization.

For N=query_tile+1:
                            baseline       retained
  Q DMA calls/block         8*N            2+6*N
  K/V DMA calls/block       16*N           16*N
  ordinary LDS loads/block  144*N          4+140*N
  FP16 Q vector scales      16*N           4+12*N
  block barriers            16*N           1+16*N
  vmcnt(0) waits            12*N           1+12*N
  MMAC calls                256*N          256*N

The retained state is four <4 x half> values per thread: 32 packed bytes,
equivalent to eight 32-bit words. This is not a promise that the final VGPR
count increases by eight; register allocation, rematerialization and scratch
must be inspected in the target gfx936 compilation.

For the whole fixed shape, logical Q requests decrease from 528 MiB to 404 MiB.
Total Q/K/V read requests decrease from 1584 MiB to 1460 MiB (7.83%). These are
logical request bytes, not HBM traffic or a predicted speedup. At q0, no repeated
read is eliminated and the added startup synchronization remains a cost.

Reproduce from the repository root:
  python generated_ir/q_bk32_retained/generate.py
  OPENBLAS_NUM_THREADS=1 python generated_ir/q_bk32_retained/validate.py

The generator is explicitly restricted to the fingerprinted source and fixed
configuration. It clones the selected fragment's actual dependency graph and
keeps the original packing/scaling, rather than introducing a new load layout.
It is an experimental IR transform, not a general compiler loop-hoisting pass.

GPU compile and benchmark in the target DTK/gfx936 environment:
  mkdir -p generated_ir/q_bk32_retained/build-baseline
  cd generated_ir/q_bk32_retained/build-baseline
  bash ../../../GenHsaco.sh ../baseline.ll 3 kernel.hsaco
  python ../../../launchKernel.py --hsaco kernel.hsaco \
    --config ../baseline.launch.json --warmup 50 --repeat 200 --json-output result.json

For the candidate, use a separate build-retained directory, ../q_bk32_retained.ll
and ../q_bk32_retained.launch.json. Keep all compiler/environment options equal.
Compare program median and final VGPR/scratch alongside actual memory bytes;
Q request removal can lower L2 hit rate without implying a performance loss.

Validation performed:
  LLVM assembly and O3 verifier;
  actual IR CPU execution with DMA delayed until waits for q0/q1/q7/q8/q31,
  two heads and two seeds; bitwise equal outputs and reference tolerance checks;
  K/V and remaining-Q global-read traces identical;
  all 4096 retained Q elements loaded once per block;
  expected dynamic DMA/LDS-load/scaling/MMAC/wait/barrier counts;
  no pending DMA at return and output writes confined to the selected tile.
See manifest.json, validation.json and validation.log.

No target GPU compilation or performance measurement has been performed.
The CPU model does not simulate independently progressing waves or register
allocation; the 1.779047 ms historical measurement applies only to the source.
