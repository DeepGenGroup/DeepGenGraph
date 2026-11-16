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
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 32, 0, 1, 0>, thread_num = 256 : i32} {
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %c32 = arith.constant 32 : index
    %0 = arith.muli %block_id_y, %c32 : index
    %1 = arith.addi %block_id_x, %0 : index
    %c1024 = arith.constant 1024 : index
    %c32_0 = arith.constant 32 : index
    %2 = arith.divui %1, %c1024 : index
    %3 = arith.muli %2, %c32_0 : index
    %c32_1 = arith.constant 32 : index
    %4 = arith.remui %1, %c1024 : index
    %5 = arith.remui %4, %c32_1 : index
    %6 = arith.addi %3, %5 : index
    %7 = arith.divui %4, %c32_1 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %thread_id_x = gpu.thread_id x {range = 256 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %8 = frisk.cast %cst : f32 -> f16
    %9 = frisk.zero {initIdx = 0 : index} : memref<128x32xf32>
    %10 = frisk.zero {initIdx = 1 : index} : memref<128x1xf32>
    %11 = frisk.add %10, %cst_2 : memref<128x1xf32>, f32 -> memref<128x1xf32>
    %12 = frisk.zero {initIdx = 3 : index} : memref<128x32xf32>
    %13 = frisk.zero {initIdx = 4 : index} : memref<128x32xf32>
    %14 = frisk.zero {initIdx = 5 : index} : memref<128x32xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %15 = frisk.cast %11 : memref<128x1xf32> -> vector<128x1xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %16:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_3, %arg6 = %cst_4, %arg7 = %15, %arg8 = %cst_5, %arg9 = %cst_6, %arg10 = %cst_7) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %25 = frisk.zero {initIdx = 0 : index} : memref<128x128xf32>
      %cst_8 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %26 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_8) -> (vector<128x128xf32>) {
        %63 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
        %64 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
        %65 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.copy %65 to %63 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %66 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.copy %66 to %64 [affine_map<() -> (2)>] : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %67 = frisk.mul %63, %8 : memref<128x32xf16, 3>, f16 -> memref<128x32xf16, 3>
        %68 = frisk.zero : memref<128x128xf32>
        %69 = frisk.gemm(%67, %64, %68) {transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<128x128xf32> -> memref<128x128xf32>
        %70 = frisk.add %arg12, %69 : vector<128x128xf32>, memref<128x128xf32> -> memref<128x128xf32>
        frisk.copy %70 to %arg12 [affine_map<() -> (2)>] : memref<128x128xf32>, vector<128x128xf32>
        affine.yield %arg12 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %27 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %28 = frisk.cast %26 : vector<128x128xf32> -> memref<128x128xf32, 3>
      frisk.reduce %28, %27 {dim = 1 : i64, kind = "max"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %29 = frisk.max %27, %arg7 : memref<128x1xf32, 3>, vector<128x1xf32> -> memref<128x1xf32, 3>
      %30 = frisk.sub %26, %29 : vector<128x128xf32>, memref<128x1xf32, 3> -> memref<128x128xf32>
      %31 = frisk.sub %arg7, %29 : vector<128x1xf32>, memref<128x1xf32, 3> -> memref<128x1xf32>
      %32 = frisk.exp2 %31 : memref<128x1xf32> -> memref<128x1xf32>
      %33 = frisk.mul %arg6, %32 : vector<128x1xf32>, memref<128x1xf32> -> memref<128x1xf32>
      %34 = frisk.exp2 %30 : memref<128x128xf32> -> memref<128x128xf32>
      %35 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %36 = frisk.cast %34 : memref<128x128xf32> -> memref<128x128xf32, 3>
      frisk.reduce %36, %35 {dim = 1 : i64, kind = "add"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %37 = frisk.add %35, %33 : memref<128x1xf32, 3>, memref<128x1xf32> -> memref<128x1xf32, 3>
      %38 = frisk.cast %34 : memref<128x128xf32> -> memref<128x128xf16, 3>
      %39 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %40 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.copy %40 to %39 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %41 = frisk.zero : memref<128x32xf32>
      %42 = frisk.gemm(%38, %39, %41) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
      %43 = frisk.mul %arg5, %32 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %44 = frisk.add %43, %42 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %45 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %46 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.copy %46 to %45 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %47 = frisk.zero : memref<128x32xf32>
      %48 = frisk.gemm(%38, %45, %47) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
      %49 = frisk.mul %arg8, %32 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %50 = frisk.add %49, %48 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %51 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %52 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.copy %52 to %51 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %53 = frisk.zero : memref<128x32xf32>
      %54 = frisk.gemm(%38, %51, %53) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
      %55 = frisk.mul %arg9, %32 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %56 = frisk.add %55, %54 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %57 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %58 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.copy %58 to %57 [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %59 = frisk.zero : memref<128x32xf32>
      %60 = frisk.gemm(%38, %57, %59) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
      %61 = frisk.mul %arg10, %32 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %62 = frisk.add %61, %60 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      frisk.copy %44 to %arg5 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %37 to %arg6 [affine_map<() -> (2)>] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %29 to %arg7 [affine_map<() -> (2)>] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %50 to %arg8 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %56 to %arg9 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %62 to %arg10 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      affine.yield %arg5, %arg6, %arg7, %arg8, %arg9, %arg10 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %17 = frisk.div %16#0, %16#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %18 = frisk.cast %17 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %18 to %arg3 at (%6, %7, %block_id_z) [affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 0)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %19 = frisk.div %16#3, %16#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %20 = frisk.cast %19 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %20 to %arg3 at (%6, %7, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 32) floordiv 524288) mod 32, (d0 * 128) mod 4096, 32)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %21 = frisk.div %16#4, %16#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %22 = frisk.cast %21 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %22 to %arg3 at (%6, %7, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 64) floordiv 524288) mod 32, (d0 * 128) mod 4096, 64)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %23 = frisk.div %16#5, %16#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %24 = frisk.cast %23 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %24 to %arg3 at (%6, %7, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 96) floordiv 524288) mod 32, (d0 * 128) mod 4096, 96)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    return
  }
}
