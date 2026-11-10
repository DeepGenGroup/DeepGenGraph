#map = affine_map<() -> (2)>
#map1 = affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 0)>
#map2 = affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 32) floordiv 524288) mod 32, (d0 * 128) mod 4096, 32)>
#map3 = affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 64) floordiv 524288) mod 32, (d0 * 128) mod 4096, 64)>
#map4 = affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 96) floordiv 524288) mod 32, (d0 * 128) mod 4096, 96)>
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
        %67 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
        %68 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
        %69 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        %70 = frisk.async_copy_gm_to_shm %69 to %67 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
        frisk.wait_buffer_ready %70 : !frisk.async_task
        %71 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        %72 = frisk.async_copy_gm_to_shm %71 to %68 {alignment = 16 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
        frisk.wait_buffer_ready %72 : !frisk.async_task
        %73 = frisk.mul %67, %8 : memref<128x32xf16, 3>, f16 -> memref<128x32xf16, 3>
        %74 = frisk.gemm(%73, %68) {frisk.a_input_layout_method = "b-permute", frisk.b_permute, frisk.contiguous_c, frisk.wmma_layout_applied = "b-permute", inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3> -> memref<128x128xf32>
        %75 = frisk.add %arg12, %74 : vector<128x128xf32>, memref<128x128xf32> -> memref<128x128xf32>
        frisk.copy %75 to %arg12 [#map] : memref<128x128xf32>, vector<128x128xf32>
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
      %41 = frisk.async_copy_gm_to_shm %40 to %39 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %41 : !frisk.async_task
      %42 = frisk.read_matrix_from_shm %39[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %43 = frisk.gemm(%38, %42) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16> -> memref<128x32xf32>
      %44 = frisk.mul %arg5, %32 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %45 = frisk.add %44, %43 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %46 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %47 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %48 = frisk.async_copy_gm_to_shm %47 to %46 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %48 : !frisk.async_task
      %49 = frisk.read_matrix_from_shm %46[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %50 = frisk.gemm(%38, %49) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16> -> memref<128x32xf32>
      %51 = frisk.mul %arg8, %32 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %52 = frisk.add %51, %50 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %53 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %54 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %55 = frisk.async_copy_gm_to_shm %54 to %53 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %55 : !frisk.async_task
      %56 = frisk.read_matrix_from_shm %53[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %57 = frisk.gemm(%38, %56) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16> -> memref<128x32xf32>
      %58 = frisk.mul %arg9, %32 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %59 = frisk.add %58, %57 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %60 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %61 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %62 = frisk.async_copy_gm_to_shm %61 to %60 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %62 : !frisk.async_task
      %63 = frisk.read_matrix_from_shm %60[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %64 = frisk.gemm(%38, %63) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16> -> memref<128x32xf32>
      %65 = frisk.mul %arg10, %32 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %66 = frisk.add %65, %64 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      frisk.copy %45 to %arg5 [#map] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %37 to %arg6 [#map] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %29 to %arg7 [#map] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %52 to %arg8 [#map] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %59 to %arg9 [#map] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %66 to %arg10 [#map] : memref<128x32xf32>, vector<128x32xf32>
      affine.yield %arg5, %arg6, %arg7, %arg8, %arg9, %arg10 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %17 = frisk.div %16#0, %16#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %18 = frisk.cast %17 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %18 to %arg3 at (%6, %7, %block_id_z) [#map1] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %19 = frisk.div %16#3, %16#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %20 = frisk.cast %19 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %20 to %arg3 at (%6, %7, %block_id_z) [#map2] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %21 = frisk.div %16#4, %16#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %22 = frisk.cast %21 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %22 to %arg3 at (%6, %7, %block_id_z) [#map3] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %23 = frisk.div %16#5, %16#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %24 = frisk.cast %23 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %24 to %arg3 at (%6, %7, %block_id_z) [#map4] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    return
  }
}

