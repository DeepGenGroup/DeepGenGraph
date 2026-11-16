#map = affine_map<(d0, d1) -> (d0 - d1 + 16)>
#map1 = affine_map<(d0) -> (d0 + 3)>
#map2 = affine_map<(d0) -> (d0 + 6)>
module {
  func.func @probe(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>, %arg2: index, %arg3: index) -> (f32, f32) attributes {thread_num = 128 : i32} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.000000e+00 : f32
    %0 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
    %1 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
    %2 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
    %3 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
    %4 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
    %5 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<1xf32, 3>
    %6 = arith.cmpi slt, %arg2, %arg3 : index
    %7 = arith.subi %arg3, %arg2 : index
    %8 = frisk.async_task_completed : !frisk.async_task
    %c0_1 = arith.constant 0 : index
    %9 = arith.cmpi ugt, %7, %c0_1 : index
    %10 = arith.andi %6, %9 : i1
    %11:2 = scf.if %10 -> (!frisk.async_task, !frisk.async_task) {
      %16 = affine.apply #map(%arg2, %arg2)
      %subview = memref.subview %arg0[%16] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
      %subview_2 = memref.subview %arg1[%16] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
      %17 = frisk.async_copy_gm_to_shm %subview to %0 : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
      %18 = frisk.async_copy_gm_to_shm %subview_2 to %3 : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
      scf.yield %17, %18 : !frisk.async_task, !frisk.async_task
    } else {
      scf.yield %8, %8 : !frisk.async_task, !frisk.async_task
    }
    %c3 = arith.constant 3 : index
    %12 = arith.cmpi ugt, %7, %c3 : index
    %13 = arith.andi %6, %12 : i1
    %14:2 = scf.if %13 -> (!frisk.async_task, !frisk.async_task) {
      %16 = affine.apply #map1(%arg2)
      %17 = affine.apply #map(%16, %arg2)
      %subview = memref.subview %arg0[%17] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
      %subview_2 = memref.subview %arg1[%17] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
      %18 = frisk.async_copy_gm_to_shm %subview to %1 : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
      %19 = frisk.async_copy_gm_to_shm %subview_2 to %4 : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
      scf.yield %18, %19 : !frisk.async_task, !frisk.async_task
    } else {
      scf.yield %8, %8 : !frisk.async_task, !frisk.async_task
    }
    %15:6 = affine.for %arg4 = %arg2 to %arg3 step 3 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %11#0, %arg8 = %11#1, %arg9 = %14#0, %arg10 = %14#1) -> (f32, f32, !frisk.async_task, !frisk.async_task, !frisk.async_task, !frisk.async_task) {
      %16 = arith.subi %arg4, %arg2 : index
      %c3_2 = arith.constant 3 : index
      %17 = arith.divui %16, %c3_2 : index
      %c3_3 = arith.constant 3 : index
      %18 = arith.remui %17, %c3_3 : index
      %c1 = arith.constant 1 : index
      %19 = arith.cmpi eq, %18, %c1 : index
      %20:2 = scf.if %19 -> (memref<1xf32, 3>, memref<1xf32, 3>) {
        scf.yield %1, %4 : memref<1xf32, 3>, memref<1xf32, 3>
      } else {
        scf.yield %2, %5 : memref<1xf32, 3>, memref<1xf32, 3>
      }
      %c0_4 = arith.constant 0 : index
      %21 = arith.cmpi eq, %18, %c0_4 : index
      %22:2 = scf.if %21 -> (memref<1xf32, 3>, memref<1xf32, 3>) {
        scf.yield %0, %3 : memref<1xf32, 3>, memref<1xf32, 3>
      } else {
        scf.yield %20#0, %20#1 : memref<1xf32, 3>, memref<1xf32, 3>
      }
      frisk.wait_buffer_ready %arg7 : !frisk.async_task
      frisk.wait_buffer_ready %arg8 : !frisk.async_task
      frisk.sync_threads_in_block
      %23 = arith.subi %arg3, %arg4 : index
      %c6 = arith.constant 6 : index
      %24 = arith.cmpi ugt, %23, %c6 : index
      %25:2 = scf.if %24 -> (!frisk.async_task, !frisk.async_task) {
        %33 = affine.apply #map2(%arg4)
        %c3_6 = arith.constant 3 : index
        %34 = arith.remui %17, %c3_6 : index
        %c2 = arith.constant 2 : index
        %35 = arith.addi %34, %c2 : index
        %36 = arith.remui %35, %c3_6 : index
        %c1_7 = arith.constant 1 : index
        %37 = arith.cmpi eq, %36, %c1_7 : index
        %38:2 = scf.if %37 -> (memref<1xf32, 3>, memref<1xf32, 3>) {
          scf.yield %1, %4 : memref<1xf32, 3>, memref<1xf32, 3>
        } else {
          scf.yield %2, %5 : memref<1xf32, 3>, memref<1xf32, 3>
        }
        %c0_8 = arith.constant 0 : index
        %39 = arith.cmpi eq, %36, %c0_8 : index
        %40:2 = scf.if %39 -> (memref<1xf32, 3>, memref<1xf32, 3>) {
          scf.yield %0, %3 : memref<1xf32, 3>, memref<1xf32, 3>
        } else {
          scf.yield %38#0, %38#1 : memref<1xf32, 3>, memref<1xf32, 3>
        }
        %41 = affine.apply #map(%33, %arg2)
        %subview_9 = memref.subview %arg0[%41] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
        %subview_10 = memref.subview %arg1[%41] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
        %42 = frisk.async_copy_gm_to_shm %subview_9 to %40#0 : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
        %43 = frisk.async_copy_gm_to_shm %subview_10 to %40#1 : memref<1xf32, strided<[1], offset: ?>, 1>, memref<1xf32, 3> -> !frisk.async_task
        scf.yield %42, %43 : !frisk.async_task, !frisk.async_task
      } else {
        scf.yield %8, %8 : !frisk.async_task, !frisk.async_task
      }
      %26 = affine.apply #map(%arg4, %arg2)
      %subview = memref.subview %arg0[%26] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
      %subview_5 = memref.subview %arg1[%26] [1] [1] : memref<64xf32, 1> to memref<1xf32, strided<[1], offset: ?>, 1>
      %27 = memref.load %22#0[%c0] : memref<1xf32, 3>
      %28 = memref.load %22#1[%c0] : memref<1xf32, 3>
      %29 = arith.mulf %arg5, %cst_0 : f32
      %30 = arith.addf %27, %28 : f32
      %31 = arith.addf %29, %30 : f32
      %32 = arith.addf %arg6, %28 : f32
      affine.yield %31, %32, %arg9, %arg10, %25#0, %25#1 : f32, f32, !frisk.async_task, !frisk.async_task, !frisk.async_task, !frisk.async_task
    } {pipeline.num_stages = 3 : i64, pipeline.scheduled, pipeline.slots = 3 : i64}
    return %15#0, %15#1 : f32, f32
  }
}

