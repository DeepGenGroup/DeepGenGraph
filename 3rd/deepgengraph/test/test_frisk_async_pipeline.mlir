module {
  func.func @pipeline(%a: memref<64x32xf16, 1>, %src: memref<4x32x32xf16, 1>) attributes {thread_num = 128 : i32} {
    %lhs = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<64x32xf16, 3>
    %shared = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x32xf16, 3>
    %task_lhs = frisk.async_copy_gm_to_shm %a to %lhs : memref<64x32xf16, 1>, memref<64x32xf16, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task_lhs : !frisk.async_task
    %zero = arith.constant dense<0.0> : vector<64x32xf32>
    %product_init = frisk.zero : memref<64x32xf32>
    %result = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> (vector<64x32xf32>) {
      %tile = frisk.buffer_view %src[%i, 0, 0], ranges = [32, 32] : memref<4x32x32xf16, 1> -> memref<32x32xf16, 1>
      %task_shared = frisk.async_copy_gm_to_shm %tile to %shared : memref<32x32xf16, 1>, memref<32x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %task_shared : !frisk.async_task
      %product = frisk.gemm(%lhs, %shared, %product_init) {transA = false, transB = false} : memref<64x32xf16, 3>, memref<32x32xf16, 3>, memref<64x32xf32> -> memref<64x32xf32>
      %sum = frisk.add %acc, %product : vector<64x32xf32>, memref<64x32xf32> -> memref<64x32xf32>
      frisk.copy %sum to %acc [affine_map<() -> (2)>] : memref<64x32xf32>, vector<64x32xf32>
      affine.yield %acc : vector<64x32xf32>
    } {pipeline.stage = array<i64: 0, 0, 1, 1, 1, 1>, pipeline.order = array<i64: 0, 0, 1, 1, 1, 1>}
    return
  }
}
