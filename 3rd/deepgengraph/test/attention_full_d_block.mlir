  func.func @Attn_p2(%arg0: memref<1x1x128x128xf16, 1>, %arg1: memref<1x1x128x128xf16, 1>, %arg2: memref<1x1x128x128xf16, 1>, %arg3: memref<1x1x128x128xf16, 1>) attributes {frisk.block_warps = array<i64: 1, 1>, thread_num = 64 : i32} {
    %block_id_z = gpu.block_id z {range = 1 : index}
    %block_id_y = gpu.block_id y {range = 1 : index}
    %block_id_x = gpu.block_id x {range = 4 : index}
    %thread_id_x = gpu.thread_id x {range = 64 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
    %1 = frisk.buffer_view %arg0[0, 0, (%block_id_x * 32) mod 128, 0], ranges = [32, 128] : memref<1x1x128x128xf16, 1> -> memref<32x128xf16, 1>
    frisk.copy %1 to %0 [affine_map<() -> (2)>] : memref<32x128xf16, 1>, memref<32x128xf16, 3>
    %2 = frisk.cast %cst : f32 -> f16
    %3 = frisk.mul %0, %2 : memref<32x128xf16, 3>, f16 -> memref<32x128xf16, 3>
    %4 = frisk.zero {initIdx = 2 : index} : memref<32x128xf32>
    %5 = frisk.zero {initIdx = 3 : index} : memref<32x1xf32>
    %6 = frisk.add %5, %cst_0 : memref<32x1xf32>, f32 -> memref<32x1xf32>
    %7 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
    %8 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<32x128xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<32x1xf32>
    %9 = frisk.cast %6 : memref<32x1xf32> -> vector<32x1xf32>
    %10:3 = affine.for %arg4 = 0 to 128 step 32 iter_args(%arg5 = %cst_1, %arg6 = %cst_2, %arg7 = %9) -> (vector<32x128xf32>, vector<32x1xf32>, vector<32x1xf32>) {
      %13 = frisk.buffer_view %arg1[0, 0, (((%arg4 floordiv 32) * 32) floordiv 128) mod 128, ((%arg4 floordiv 32) * 32) mod 128], ranges = [128, 32] : memref<1x1x128x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.copy %13 to %7 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %14 = frisk.buffer_view %arg2[0, 0, ((%arg4 floordiv 32) * 32) mod 128, 0], ranges = [32, 128] : memref<1x1x128x128xf16, 1> -> memref<32x128xf16, 1>
      frisk.copy %14 to %8 [affine_map<() -> (2)>] : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      %15 = frisk.zero : memref<32x32xf32>
      %16 = frisk.gemm(%3, %7, %15) {transA = false, transB = false} : memref<32x128xf16, 3>, memref<128x32xf16, 3>, memref<32x32xf32> -> memref<32x32xf32>
      %17 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x1xf32, 3>
      %18 = frisk.cast %16 : memref<32x32xf32> -> memref<32x32xf32, 3>
      frisk.reduce %18, %17 {dim = 1 : i64, kind = "max"} : memref<32x32xf32, 3>, memref<32x1xf32, 3>
      %19 = frisk.max %17, %arg7 : memref<32x1xf32, 3>, vector<32x1xf32> -> memref<32x1xf32, 3>
      %20 = frisk.sub %16, %19 : memref<32x32xf32>, memref<32x1xf32, 3> -> memref<32x32xf32>
      %21 = frisk.sub %arg7, %19 : vector<32x1xf32>, memref<32x1xf32, 3> -> memref<32x1xf32>
      %22 = frisk.exp2 %21 : memref<32x1xf32> -> memref<32x1xf32>
      %23 = frisk.mul %arg6, %22 : vector<32x1xf32>, memref<32x1xf32> -> memref<32x1xf32>
      %24 = frisk.exp2 %20 : memref<32x32xf32> -> memref<32x32xf32>
      %25 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x1xf32, 3>
      %26 = frisk.cast %24 : memref<32x32xf32> -> memref<32x32xf32, 3>
      frisk.reduce %26, %25 {dim = 1 : i64, kind = "add"} : memref<32x32xf32, 3>, memref<32x1xf32, 3>
      %27 = frisk.add %25, %23 : memref<32x1xf32, 3>, memref<32x1xf32> -> memref<32x1xf32, 3>
      %28 = frisk.cast %24 : memref<32x32xf32> -> memref<32x32xf16, 3>
      %29 = frisk.zero : memref<32x128xf32>
      %30 = frisk.gemm(%28, %8, %29) {transA = false, transB = false} : memref<32x32xf16, 3>, memref<32x128xf16, 3>, memref<32x128xf32> -> memref<32x128xf32>
      %31 = frisk.mul %arg5, %22 : vector<32x128xf32>, memref<32x1xf32> -> memref<32x128xf32>
      %32 = frisk.add %31, %30 : memref<32x128xf32>, memref<32x128xf32> -> memref<32x128xf32>
      frisk.copy %32 to %arg5 [affine_map<() -> (2)>] : memref<32x128xf32>, vector<32x128xf32>
      frisk.copy %27 to %arg6 [affine_map<() -> (2)>] : memref<32x1xf32, 3>, vector<32x1xf32>
      frisk.copy %19 to %arg7 [affine_map<() -> (2)>] : memref<32x1xf32, 3>, vector<32x1xf32>
      affine.yield %arg5, %arg6, %arg7 : vector<32x128xf32>, vector<32x1xf32>, vector<32x1xf32>
    } {local_yield_transformed = true}
    %11 = frisk.div %10#0, %10#1 : vector<32x128xf32>, vector<32x1xf32> -> memref<32x128xf32>
    %12 = frisk.cast %11 : memref<32x128xf32> -> memref<32x128xf16, 3>
    frisk.copy %12 to %arg3 at (%block_id_x, %block_id_y) [affine_map<(d0, d1) -> (0, 0, (d0 * 32) mod 128, 0)>] : memref<32x128xf16, 3>, memref<1x1x128x128xf16, 1>
    return
  }
