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
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1>, %arg1: memref<1x32x128x4096xf16, 1>, %arg2: memref<1x32x4096x128xf16, 1>, %arg3: memref<1x32x4096x128xf16, 1>) attributes {frisk.attention_tiles = array<i64: 16, 16>, frisk.block_warps = array<i64: 4, 1>, thread_num = 256 : i32} {
    %block_id_z = gpu.block_id z {range = 1 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %block_id_x = gpu.block_id x {range = 32 : index}
    %thread_id_x = gpu.thread_id x {range = 256 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = frisk.cast %cst : f32 -> f16
    %1 = frisk.zero {initIdx = 0 : index} : memref<128x128xf32>
    %2 = frisk.zero {initIdx = 1 : index} : memref<128x1xf32>
    %3 = frisk.add %2, %cst_0 : memref<128x1xf32>, f32 -> memref<128x1xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %4 = frisk.cast %3 : memref<128x1xf32> -> vector<128x1xf32>
    %5:3 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_1, %arg6 = %cst_2, %arg7 = %4) -> (vector<128x128xf32>, vector<128x1xf32>, vector<128x1xf32>) {
      %8 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x128xf16, 3>
      %9 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x128xf16, 3>
      %10 = frisk.buffer_view %arg0[0, (%block_id_z * 32 + %block_id_y + (%block_id_x * 16384) floordiv 524288) mod 32, (%block_id_x * 128) mod 4096, 0], ranges = [128, 128] : memref<1x32x4096x128xf16, 1> -> memref<128x128xf16, 1>
      frisk.copy %10 to %8 [affine_map<() -> (2)>] : memref<128x128xf16, 1>, memref<128x128xf16, 3>
      %11 = frisk.buffer_view %arg1[0, (%block_id_z * 32 + %block_id_y + %arg4 floordiv 524288) mod 32, (%arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [128, 128] : memref<1x32x128x4096xf16, 1> -> memref<128x128xf16, 1>
      frisk.copy %11 to %9 [affine_map<() -> (2)>] : memref<128x128xf16, 1>, memref<128x128xf16, 3>
      %12 = frisk.mul %8, %0 : memref<128x128xf16, 3>, f16 -> memref<128x128xf16, 3>
      %13 = frisk.zero : memref<128x128xf32>
      %14 = frisk.gemm(%12, %9, %13) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x128xf16, 3>, memref<128x128xf32> -> memref<128x128xf32>
      %15 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %16 = frisk.cast %14 : memref<128x128xf32> -> memref<128x128xf32, 3>
      frisk.reduce %16, %15 {dim = 1 : i64, kind = "max"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %17 = frisk.max %15, %arg7 : memref<128x1xf32, 3>, vector<128x1xf32> -> memref<128x1xf32, 3>
      %18 = frisk.sub %14, %17 : memref<128x128xf32>, memref<128x1xf32, 3> -> memref<128x128xf32>
      %19 = frisk.sub %arg7, %17 : vector<128x1xf32>, memref<128x1xf32, 3> -> memref<128x1xf32>
      %20 = frisk.exp2 %19 : memref<128x1xf32> -> memref<128x1xf32>
      %21 = frisk.mul %arg6, %20 : vector<128x1xf32>, memref<128x1xf32> -> memref<128x1xf32>
      %22 = frisk.exp2 %18 : memref<128x128xf32> -> memref<128x128xf32>
      %23 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %24 = frisk.cast %22 : memref<128x128xf32> -> memref<128x128xf32, 3>
      frisk.reduce %24, %23 {dim = 1 : i64, kind = "add"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %25 = frisk.add %23, %21 : memref<128x1xf32, 3>, memref<128x1xf32> -> memref<128x1xf32, 3>
      %26 = frisk.cast %22 : memref<128x128xf32> -> memref<128x128xf16, 3>
      %27 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x128xf16, 3>
      %28 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %block_id_y + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 128] : memref<1x32x4096x128xf16, 1> -> memref<128x128xf16, 1>
      frisk.copy %28 to %27 [affine_map<() -> (2)>] : memref<128x128xf16, 1>, memref<128x128xf16, 3>
      %29 = frisk.zero : memref<128x128xf32>
      %30 = frisk.gemm(%26, %27, %29) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x128xf16, 3>, memref<128x128xf32> -> memref<128x128xf32>
      %31 = frisk.mul %arg5, %20 : vector<128x128xf32>, memref<128x1xf32> -> memref<128x128xf32>
      %32 = frisk.add %31, %30 : memref<128x128xf32>, memref<128x128xf32> -> memref<128x128xf32>
      frisk.copy %32 to %arg5 [affine_map<() -> (2)>] : memref<128x128xf32>, vector<128x128xf32>
      frisk.copy %25 to %arg6 [affine_map<() -> (2)>] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %17 to %arg7 [affine_map<() -> (2)>] : memref<128x1xf32, 3>, vector<128x1xf32>
      affine.yield %arg5, %arg6, %arg7 : vector<128x128xf32>, vector<128x1xf32>, vector<128x1xf32>
    } {local_yield_transformed = true}
    %6 = frisk.div %5#0, %5#1 : vector<128x128xf32>, vector<128x1xf32> -> memref<128x128xf32>
    %7 = frisk.cast %6 : memref<128x128xf32> -> memref<128x128xf16, 3>
    frisk.copy %7 to %arg3 at (%block_id_x, %block_id_y, %block_id_z) [affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 0)>] : memref<128x128xf16, 3>, memref<1x32x4096x128xf16, 1>
    return
  }
}
