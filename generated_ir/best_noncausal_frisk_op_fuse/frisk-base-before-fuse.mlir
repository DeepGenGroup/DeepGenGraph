module {
  func.func @Attn(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> {
    %cst = arith.constant dense<1.131250e+01> : tensor<1xf32>
    %0 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %1 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %2 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16>
    %3 = deepgengraph.precise_dot_op %0, %2, acc = f32 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf32>
    %4 = deepgengraph.div %3, %cst : (tensor<1x32x4096x4096xf32>, tensor<1xf32>) -> tensor<1x32x4096x4096xf32>
    %5 = deepgengraph.reduce(%4), dim = -1, op = MAX, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %6 = deepgengraph.sub %4, %5 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %7 = deepgengraph.exp %6 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %8 = deepgengraph.reduce(%7), dim = -1, op = ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %9 = deepgengraph.div %7, %8 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %10 = deepgengraph.convert %9, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
    %11 = deepgengraph.dot %10, %1 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16>
    %12 = deepgengraph.permute %11, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16>
    return %12 : tensor<1x4096x32x128xf16>
  }
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1>, %arg1: memref<1x32x128x4096xf16, 1>, %arg2: memref<1x32x4096x128xf16, 1>, %arg3: memref<1x32x4096x128xf16, 1>) attributes {frisk.block_warps = array<i64: 4, 1>, thread_num = 256 : i32} {
    %block_id_z = gpu.block_id z {range = 1 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %block_id_x = gpu.block_id x {range = 32 : index}
    %thread_id_x = gpu.thread_id x {range = 256 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = frisk.cast %cst : f32 -> f16
    %1 = frisk.zero {initIdx = 0 : index} : memref<128x32xf32>
    %2 = frisk.zero {initIdx = 1 : index} : memref<128x1xf32>
    %3 = frisk.add %2, %cst_0 : memref<128x1xf32>, f32 -> memref<128x1xf32>
    %4 = frisk.zero {initIdx = 3 : index} : memref<128x32xf32>
    %5 = frisk.zero {initIdx = 4 : index} : memref<128x32xf32>
    %6 = frisk.zero {initIdx = 5 : index} : memref<128x32xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %7 = frisk.cast %3 : memref<128x1xf32> -> vector<128x1xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %8:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_1, %arg6 = %cst_2, %arg7 = %7, %arg8 = %cst_3, %arg9 = %cst_4, %arg10 = %cst_5) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %17 = frisk.zero {initIdx = 0 : index} : memref<128x128xf32>
      %cst_6 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %18 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_6) -> (vector<128x128xf32>) {
        %55 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
        %56 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
        %57 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %block_id_y * 524288 + %block_id_x * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %block_id_y * 4096 + %block_id_x * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.copy %57 to %55 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %58 = frisk.buffer_view %arg1[0, ((%block_id_z * 16777216 + %block_id_y * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %block_id_y * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, 1> -> memref<32x128xf16, 1>
        frisk.copy %58 to %56 [affine_map<() -> (2)>] : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %59 = frisk.mul %55, %0 : memref<128x32xf16, 3>, f16 -> memref<128x32xf16, 3>
        %60 = frisk.zero : memref<128x128xf32>
        %61 = frisk.gemm(%59, %56, %60) {transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<128x128xf32> -> memref<128x128xf32>
        %62 = frisk.add %arg12, %61 : vector<128x128xf32>, memref<128x128xf32> -> memref<128x128xf32>
        frisk.copy %62 to %arg12 [affine_map<() -> (2)>] : memref<128x128xf32>, vector<128x128xf32>
        affine.yield %arg12 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %19 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %20 = frisk.cast %18 : vector<128x128xf32> -> memref<128x128xf32, 3>
      frisk.reduce %20, %19 {dim = 1 : i64, kind = "max"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %21 = frisk.max %19, %arg7 : memref<128x1xf32, 3>, vector<128x1xf32> -> memref<128x1xf32, 3>
      %22 = frisk.sub %18, %21 : vector<128x128xf32>, memref<128x1xf32, 3> -> memref<128x128xf32>
      %23 = frisk.sub %arg7, %21 : vector<128x1xf32>, memref<128x1xf32, 3> -> memref<128x1xf32>
      %24 = frisk.exp2 %23 : memref<128x1xf32> -> memref<128x1xf32>
      %25 = frisk.mul %arg6, %24 : vector<128x1xf32>, memref<128x1xf32> -> memref<128x1xf32>
      %26 = frisk.exp2 %22 : memref<128x128xf32> -> memref<128x128xf32>
      %27 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %28 = frisk.cast %26 : memref<128x128xf32> -> memref<128x128xf32, 3>
      frisk.reduce %28, %27 {dim = 1 : i64, kind = "add"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %29 = frisk.add %27, %25 : memref<128x1xf32, 3>, memref<128x1xf32> -> memref<128x1xf32, 3>
      %30 = frisk.cast %26 : memref<128x128xf32> -> memref<128x128xf16, 3>
      %31 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %32 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %block_id_y + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.copy %32 to %31 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %33 = frisk.zero : memref<128x32xf32>
      %34 = frisk.gemm(%30, %31, %33) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
      %35 = frisk.mul %arg5, %24 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %36 = frisk.add %35, %34 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %37 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %38 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %block_id_y * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.copy %38 to %37 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %39 = frisk.zero : memref<128x32xf32>
      %40 = frisk.gemm(%30, %37, %39) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
      %41 = frisk.mul %arg8, %24 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %42 = frisk.add %41, %40 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %43 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %44 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %block_id_y * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.copy %44 to %43 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %45 = frisk.zero : memref<128x32xf32>
      %46 = frisk.gemm(%30, %43, %45) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
      %47 = frisk.mul %arg9, %24 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %48 = frisk.add %47, %46 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %49 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %50 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %block_id_y * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.copy %50 to %49 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %51 = frisk.zero : memref<128x32xf32>
      %52 = frisk.gemm(%30, %49, %51) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
      %53 = frisk.mul %arg10, %24 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %54 = frisk.add %53, %52 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      frisk.copy %36 to %arg5 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %29 to %arg6 [affine_map<() -> (2)>] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %21 to %arg7 [affine_map<() -> (2)>] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %42 to %arg8 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %48 to %arg9 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %54 to %arg10 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      affine.yield %arg5, %arg6, %arg7, %arg8, %arg9, %arg10 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %9 = frisk.div %8#0, %8#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %10 = frisk.cast %9 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %10 to %arg3 at (%block_id_x, %block_id_y, %block_id_z) [affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 0)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %11 = frisk.div %8#3, %8#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %12 = frisk.cast %11 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %12 to %arg3 at (%block_id_x, %block_id_y, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 32) floordiv 524288) mod 32, (d0 * 128) mod 4096, 32)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %13 = frisk.div %8#4, %8#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %14 = frisk.cast %13 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %14 to %arg3 at (%block_id_x, %block_id_y, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 64) floordiv 524288) mod 32, (d0 * 128) mod 4096, 64)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %15 = frisk.div %8#5, %8#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %16 = frisk.cast %15 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %16 to %arg3 at (%block_id_x, %block_id_y, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 96) floordiv 524288) mod 32, (d0 * 128) mod 4096, 96)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    return
  }
}
