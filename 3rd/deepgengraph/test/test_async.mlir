// A -> LDS -> B, bit-exact copy of contiguous f32 tensors of identical shape.
// ABI: void test_async(const float *A, float *B, uint64_t num_elements).
// Launch: block=(256,1,1), grid=(ceil(num_elements/256),1,1); skip N=0.
// Target: gfx9 with wave64 (e.g. gfx90a/gfx942). A/B must be 4-byte aligned;
// disjoint buffers or A == B are supported, partial overlap is not.
//
// New LLVM: mlir-translate --mlir-to-llvmir test_async.mlir -o test_async.ll
// This checkout's LLVM 21 predates the explicit async intrinsics. Use:
//   MyTestAsync --legacy-lds test_async.mlir test_async.ll
// The compatibility mode uses global.load.lds with compiler-managed waits.
//   llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj \
//       test_async.ll -o test_async.o
//   ld.lld -shared test_async.o -o test_async.hsaco
//   python3 check_async.py --hsaco test_async.hsaco
//
// Generic spelling lets the compatibility driver parse the three new ops on
// LLVM 21. size/offset/aux are immediate attributes, not SSA operands.
module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.mlir.global internal @async_lds() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<256 x f32> {
    %undef = llvm.mlir.undef : !llvm.array<256 x f32>
    llvm.return %undef : !llvm.array<256 x f32>
  }

  llvm.func @test_async(%A: !llvm.ptr<1>, %B: !llvm.ptr<1>, %N: i64)
      attributes {rocdl.kernel, rocdl.flat_work_group_size = "256,256",
                  rocdl.reqd_work_group_size = array<i32: 256, 1, 1>} {
    %block_size = llvm.mlir.constant(256 : i64) : i64
    %wave_mask = llvm.mlir.constant(-64 : i32) : i32
    %tid = rocdl.workitem.id.x : i32
    %bid = rocdl.workgroup.id.x : i32
    %tid64 = llvm.zext %tid : i32 to i64
    %bid64 = llvm.zext %bid : i32 to i64
    %base = llvm.mul %bid64, %block_size : i64
    %index = llvm.add %base, %tid64 : i64
    %valid = llvm.icmp "ult" %index, %N : i64

    %lds = llvm.mlir.addressof @async_lds : !llvm.ptr<3>
    // Hardware adds 4*lane_id to the LDS address. Supply a wave-uniform base,
    // NOT the per-thread slot; readfirstlane explicitly exposes uniformity.
    %wave_start = llvm.and %tid, %wave_mask : i32
    %wave_base = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%wave_start) : (i32) -> i32
    %wave_lds = llvm.getelementptr %lds[%wave_base] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.cond_br %valid, ^copy, ^wait
  ^copy:
    %src = llvm.getelementptr %A[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    "rocdl.global.load.async.lds"(%src, %wave_lds) {size = 4 : i32, offset = 0 : i32, aux = 0 : i32} : (!llvm.ptr<1>, !llvm.ptr<3>) -> ()
    llvm.br ^wait
  ^wait:
    // All threads, including inactive tail lanes, reach the workgroup barrier.
    "rocdl.asyncmark"() : () -> ()
    "rocdl.wait.asyncmark"() {count = 0 : i16} : () -> ()
    rocdl.barrier
    llvm.cond_br %valid, ^store, ^exit
  ^store:
    %slot = llvm.getelementptr %lds[%tid] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %value = llvm.load %slot <{alignment = 4 : i64}> : !llvm.ptr<3> -> f32
    %dst = llvm.getelementptr %B[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    llvm.store %value, %dst <{alignment = 4 : i64}> : f32, !llvm.ptr<1>
    llvm.br ^exit
  ^exit:
    llvm.return
  }
}
