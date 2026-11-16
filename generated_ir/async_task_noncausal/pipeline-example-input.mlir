func.func @probe(%srca: memref<64xf32, 1>, %srcb: memref<64xf32, 1>, %lb: index, %ub: index) -> (f32, f32) attributes {thread_num = 128 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %three = arith.constant 3.0 : f32
  %r:2 = affine.for %i = %lb to %ub step 3 iter_args(%acc = %zero, %sum = %zero) -> (f32, f32) {
    %offset = affine.apply affine_map<(d0, d1) -> (d0 - d1 + 16)>(%i, %lb)
    %a = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
    %va = memref.subview %srca[%offset] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
    %task_a = frisk.async_copy_gm_to_shm %va to %a : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task_a : !frisk.async_task
    %b = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
    %vb = memref.subview %srcb[%offset] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
    %task_b = frisk.async_copy_gm_to_shm %vb to %b : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %task_b : !frisk.async_task

    %x = memref.load %a[%c0] : memref<1xf32, 3>
    %y = memref.load %b[%c0] : memref<1xf32, 3>
    %p = arith.mulf %acc, %three : f32
    %q = arith.addf %x, %y : f32
    %next = arith.addf %p, %q : f32
    %sum_next = arith.addf %sum, %y : f32
    affine.yield %next, %sum_next : f32, f32
  } {frisk.pipeline_hint = 3 : i64}
  return %r#0, %r#1 : f32, f32
}
