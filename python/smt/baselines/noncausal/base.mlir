#map = affine_map<() -> (2)>
#map1 = affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 0)>
#map2 = affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 32)>
#map3 = affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 64)>
#map4 = affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 96)>
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
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {frisk.attention_tiled = array<i64: 32, 32>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 32, 0, 1, 0>, thread_num = 256 : i32} {
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
    %cst = arith.constant 0.127531052 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %8 = frisk.cast %cst : f32 -> f16
    %9 = frisk.zero {initIdx = 1 : index} : memref<128x1xf32>
    %10 = frisk.add %9, %cst_2 : memref<128x1xf32>, f32 -> memref<128x1xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %11 = frisk.cast %10 : memref<128x1xf32> -> vector<128x1xf32>
    %12:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_3, %arg6 = %cst_4, %arg7 = %cst_5, %arg8 = %cst_6, %arg9 = %cst_7, %arg10 = %11) -> (vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>) {
      %21 = frisk.zero : memref<128x128xf32>
      %22 = frisk.cast %21 : memref<128x128xf32> -> vector<128x128xf32>
      %23 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %22) -> (vector<128x128xf32>) {
        %67 = frisk.buffer_view %arg0[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, %arg11], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        %68 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
        %69 = frisk.async_copy_gm_to_shm %67 to %68 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
        frisk.wait_buffer_ready %69 : !frisk.async_task
        %70 = frisk.mul %68, %8 : memref<128x32xf16, 3>, f16 -> memref<128x32xf16, 3>
        %71 = frisk.buffer_view %transpose[0, (%block_id_z * 32 + %7 + %arg4 floordiv 524288) mod 32, (%arg4 floordiv 4096) mod 128 + %arg11, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        %72 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
        %73 = frisk.async_copy_gm_to_shm %71 to %72 {alignment = 16 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
        frisk.wait_buffer_ready %73 : !frisk.async_task
        %74 = frisk.cast %arg12 : vector<128x128xf32> -> memref<128x128xf32>
        %75 = frisk.gemm(%70, %72, %74) {frisk.a_input_layout_method = "b-permute", frisk.b_permute, frisk.contiguous_c, frisk.wmma_layout_applied = "b-permute", inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<128x128xf32> -> memref<128x128xf32>
        frisk.copy %75 to %arg12 [#map] : memref<128x128xf32>, vector<128x128xf32>
        affine.yield %arg12 : vector<128x128xf32>
      } {frisk.loopUnrollFull = true, frisk.simplify_loop, local_yield_transformed = true}
      %24 = frisk.cast %23 : vector<128x128xf32> -> memref<128x128xf32>
      %25 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %26 = frisk.cast %24 : memref<128x128xf32> -> memref<128x128xf32, 3>
      frisk.reduce %26, %25 {dim = 1 : i64, kind = "max"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %27 = frisk.max %25, %arg10 : memref<128x1xf32, 3>, vector<128x1xf32> -> memref<128x1xf32, 3>
      %28 = frisk.sub %24, %27 : memref<128x128xf32>, memref<128x1xf32, 3> -> memref<128x128xf32>
      %29 = frisk.sub %arg10, %27 : vector<128x1xf32>, memref<128x1xf32, 3> -> memref<128x1xf32>
      %30 = frisk.exp2 %29 : memref<128x1xf32> -> memref<128x1xf32>
      %31 = frisk.mul %arg9, %30 : vector<128x1xf32>, memref<128x1xf32> -> memref<128x1xf32>
      %32 = frisk.exp2 %28 : memref<128x128xf32> -> memref<128x128xf32>
      %33 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %34 = frisk.cast %32 : memref<128x128xf32> -> memref<128x128xf32, 3>
      frisk.reduce %34, %33 {dim = 1 : i64, kind = "add"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %35 = frisk.add %33, %31 : memref<128x1xf32, 3>, memref<128x1xf32> -> memref<128x1xf32, 3>
      %36 = frisk.cast %32 : memref<128x128xf32> -> memref<128x128xf16, 3>
      %37 = frisk.zero : memref<128x32xf32>
      %38 = frisk.zero : memref<128x32xf32>
      %39 = frisk.zero : memref<128x32xf32>
      %c0 = arith.constant 0 : index
      %40 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, %c0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %41 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %42 = frisk.async_copy_gm_to_shm %40 to %41 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %42 : !frisk.async_task
      %43 = frisk.read_matrix_from_shm %41[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %44 = frisk.gemm(%36, %43, %37) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16>, memref<128x32xf32> -> memref<128x32xf32>
      %c32_8 = arith.constant 32 : index
      %45 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, %c32_8], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %46 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %47 = frisk.async_copy_gm_to_shm %45 to %46 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %47 : !frisk.async_task
      %48 = frisk.read_matrix_from_shm %46[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %49 = frisk.gemm(%36, %48, %38) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16>, memref<128x32xf32> -> memref<128x32xf32>
      %c64 = arith.constant 64 : index
      %50 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, %c64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %51 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %52 = frisk.async_copy_gm_to_shm %50 to %51 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %52 : !frisk.async_task
      %53 = frisk.read_matrix_from_shm %51[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %54 = frisk.gemm(%36, %53, %39) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16>, memref<128x32xf32> -> memref<128x32xf32>
      %c96 = arith.constant 96 : index
      %55 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, %c96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %56 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %57 = frisk.async_copy_gm_to_shm %55 to %56 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %57 : !frisk.async_task
      %58 = frisk.mul %arg5, %30 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %59 = frisk.mul %arg6, %30 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %60 = frisk.mul %arg7, %30 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %61 = frisk.mul %arg8, %30 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %62 = frisk.add %58, %44 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %63 = frisk.add %59, %49 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %64 = frisk.add %60, %54 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %65 = frisk.read_matrix_from_shm %56[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %66 = frisk.gemm(%36, %65, %61) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16>, memref<128x32xf32> -> memref<128x32xf32>
      frisk.copy %62 to %arg5 [#map] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %63 to %arg6 [#map] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %64 to %arg7 [#map] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %66 to %arg8 [#map] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %35 to %arg9 [#map] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %27 to %arg10 [#map] : memref<128x1xf32, 3>, vector<128x1xf32>
      affine.yield %arg5, %arg6, %arg7, %arg8, %arg9, %arg10 : vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>
    } {local_yield_transformed = true}
    %13 = frisk.div %12#0, %12#4 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %14 = frisk.div %12#1, %12#4 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %15 = frisk.div %12#2, %12#4 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %16 = frisk.div %12#3, %12#4 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %17 = frisk.cast %13 : memref<128x32xf32> -> memref<128x32xf16, 3>
    %18 = frisk.cast %14 : memref<128x32xf32> -> memref<128x32xf16, 3>
    %19 = frisk.cast %15 : memref<128x32xf32> -> memref<128x32xf16, 3>
    %20 = frisk.cast %16 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %17 to %arg3 at (%6, %7, %block_id_z) [#map1] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    frisk.copy %18 to %arg3 at (%6, %7, %block_id_z) [#map2] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    frisk.copy %19 to %arg3 at (%6, %7, %block_id_z) [#map3] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    frisk.copy %20 to %arg3 at (%6, %7, %block_id_z) [#map4] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    return
  }
}

