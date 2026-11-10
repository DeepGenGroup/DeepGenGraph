// Minimal Base-IR DMA + scalar recurrence, usable with FriskScheduleJSONTest.
// Unlike the attention outer loop, this region supports pipeline v1.
func.func @probe(%src: memref<16xf32, 1>) -> f32 attributes {thread_num = 64 : i32} {
  %zero = arith.constant 0.0 : f32
  %two = arith.constant 2.0 : f32
  %c0 = arith.constant 0 : index
  %shared = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
  %result = affine.for %i = 0 to 7 iter_args(%acc = %zero) -> (f32) {
    %view = memref.subview %src[%i] [1] [1] : memref<16xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
    %task = frisk.async_copy_gm_to_shm %view to %shared : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task : !frisk.async_task
    %x = memref.load %shared[%c0] : memref<1xf32, 3>
    %product = arith.mulf %acc, %two : f32
    %sum = arith.addf %product, %x : f32
    affine.yield %sum : f32
  }
  return %result : f32
}
