#map = affine_map<()[s0] -> (s0 * 128 + 128)>
#map1 = affine_map<() -> (2)>
#map2 = affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 0)>
#map3 = affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 32)>
#map4 = affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 64)>
#map5 = affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 96)>
module {
  func.func @Attn(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> {
    %cst = arith.constant dense<1.131250e+01> : tensor<1xf32>
    %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFF800000 : f32
    %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16>
    %4 = deepgengraph.precise_dot_op %1, %3, acc = f32 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf32>
    %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf32>, tensor<1xf32>) -> tensor<1x32x4096x4096xf32>
    %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %7 = deepgengraph.reduce(%6), dim = -1, op = MAX, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %8 = deepgengraph.sub %6, %7 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %9 = deepgengraph.exp %8 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %10 = deepgengraph.reduce(%9), dim = -1, op = ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %11 = deepgengraph.div %9, %10 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %12 = deepgengraph.convert %11, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
    %13 = deepgengraph.dot %12, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16>
    %14 = deepgengraph.permute %13, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16>
    return %14 : tensor<1x4096x32x128xf16>
  }
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {frisk.attention_tiled = array<i64: 32, 32>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 4, 0, 1, 1>, thread_num = 256 : i32} {
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %c32 = arith.constant 32 : index
    %0 = arith.muli %block_id_y, %c32 : index
    %1 = arith.addi %block_id_x, %0 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %2 = arith.divui %1, %c128 : index
    %3 = arith.muli %2, %c4 : index
    %c4_0 = arith.constant 4 : index
    %4 = arith.remui %1, %c128 : index
    %5 = arith.remui %4, %c4_0 : index
    %6 = arith.addi %3, %5 : index
    %7 = arith.divui %4, %c4_0 : index
    %c31 = arith.constant 31 : index
    %8 = arith.subi %c31, %6 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c128_3 = arith.constant 128 : index
    %cst_4 = arith.constant 0xFF800000 : f32
    %9 = frisk.cast %cst : f32 -> f16
    %10 = arith.muli %8, %c128_3 : index
    %11 = frisk.zero {initIdx = 1 : index} : memref<128x1xf32>
    %12 = frisk.add %11, %cst_4 : memref<128x1xf32>, f32 -> memref<128x1xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %13 = frisk.cast %12 : memref<128x1xf32> -> vector<128x1xf32>
    %14:6 = affine.for %arg4 = 0 to #map()[%8] step 128 iter_args(%arg5 = %cst_5, %arg6 = %cst_6, %arg7 = %cst_7, %arg8 = %cst_8, %arg9 = %cst_9, %arg10 = %13) -> (vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>) {
      %23 = frisk.zero : memref<128x128xf32>
      %24 = frisk.cast %23 : memref<128x128xf32> -> vector<128x128xf32>
      %25 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %24) -> (vector<128x128xf32>) {
        %71 = frisk.buffer_view %arg0[0, (%block_id_z * 32 + %7 + (%8 * 16384) floordiv 524288) mod 32, (%8 * 128) mod 4096, %arg11], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        %72 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
        %73 = frisk.async_copy_gm_to_shm %71 to %72 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
        frisk.wait_buffer_ready %73 : !frisk.async_task
        %74 = frisk.mul %72, %9 : memref<128x32xf16, 3>, f16 -> memref<128x32xf16, 3>
        %75 = frisk.buffer_view %transpose[0, (%block_id_z * 32 + %7 + %arg4 floordiv 524288) mod 32, (%arg4 floordiv 4096) mod 128 + %arg11, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        %76 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
        %77 = frisk.async_copy_gm_to_shm %75 to %76 {alignment = 16 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
        frisk.wait_buffer_ready %77 : !frisk.async_task
        %78 = frisk.cast %arg12 : vector<128x128xf32> -> memref<128x128xf32>
        %79 = frisk.gemm(%74, %76, %78) {frisk.a_input_layout_method = "b-permute", frisk.b_permute, frisk.contiguous_c, frisk.wmma_layout_applied = "b-permute", inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<128x128xf32> -> memref<128x128xf32>
        frisk.copy %79 to %arg12 [#map1] : memref<128x128xf32>, vector<128x128xf32>
        affine.yield %arg12 : vector<128x128xf32>
      } {frisk.loopUnrollFull = true, frisk.simplify_loop, local_yield_transformed = true}
      %26 = frisk.cast %25 : vector<128x128xf32> -> memref<128x128xf32>
      %27 = frisk.mask starts = [%10, %arg4], sizes = [128, 128], type = f32 {
      ^bb0(%arg11: index, %arg12: index):
        %71 = arith.addi %arg11, %c1 : index
        %72 = arith.cmpi ule, %71, %arg12 : index
        %73 = scf.if %72 -> (f32) {
          scf.yield %cst_1 : f32
        } else {
          scf.yield %cst_2 : f32
        }
        frisk.mask_yield %73 : f32
      } : (index, index) -> memref<128x128xf32>
      %28 = frisk.add %26, %27 : memref<128x128xf32>, memref<128x128xf32> -> memref<128x128xf32, 3>
      %29 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %30 = frisk.cast %28 : memref<128x128xf32, 3> -> memref<128x128xf32, 3>
      frisk.reduce %30, %29 {dim = 1 : i64, kind = "max"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %31 = frisk.max %29, %arg10 : memref<128x1xf32, 3>, vector<128x1xf32> -> memref<128x1xf32, 3>
      %32 = frisk.sub %28, %31 : memref<128x128xf32, 3>, memref<128x1xf32, 3> -> memref<128x128xf32>
      %33 = frisk.sub %arg10, %31 : vector<128x1xf32>, memref<128x1xf32, 3> -> memref<128x1xf32>
      %34 = frisk.exp2 %33 : memref<128x1xf32> -> memref<128x1xf32>
      %35 = frisk.mul %arg9, %34 : vector<128x1xf32>, memref<128x1xf32> -> memref<128x1xf32>
      %36 = frisk.exp2 %32 : memref<128x128xf32> -> memref<128x128xf32>
      %37 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      %38 = frisk.cast %36 : memref<128x128xf32> -> memref<128x128xf32, 3>
      frisk.reduce %38, %37 {dim = 1 : i64, kind = "add"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %39 = frisk.add %37, %35 : memref<128x1xf32, 3>, memref<128x1xf32> -> memref<128x1xf32, 3>
      %40 = frisk.cast %36 : memref<128x128xf32> -> memref<128x128xf16, 3>
      %41 = frisk.zero : memref<128x32xf32>
      %42 = frisk.zero : memref<128x32xf32>
      %43 = frisk.zero : memref<128x32xf32>
      %c0 = arith.constant 0 : index
      %44 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, %c0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %45 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %46 = frisk.async_copy_gm_to_shm %44 to %45 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %46 : !frisk.async_task
      %47 = frisk.read_matrix_from_shm %45[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %48 = frisk.gemm(%40, %47, %41) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16>, memref<128x32xf32> -> memref<128x32xf32>
      %c32_10 = arith.constant 32 : index
      %49 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, %c32_10], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %50 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %51 = frisk.async_copy_gm_to_shm %49 to %50 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %51 : !frisk.async_task
      %52 = frisk.read_matrix_from_shm %50[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %53 = frisk.gemm(%40, %52, %42) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16>, memref<128x32xf32> -> memref<128x32xf32>
      %c64 = arith.constant 64 : index
      %54 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, %c64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %55 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %56 = frisk.async_copy_gm_to_shm %54 to %55 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %56 : !frisk.async_task
      %57 = frisk.read_matrix_from_shm %55[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %58 = frisk.gemm(%40, %57, %43) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16>, memref<128x32xf32> -> memref<128x32xf32>
      %c96 = arith.constant 96 : index
      %59 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, %c96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %60 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %61 = frisk.async_copy_gm_to_shm %59 to %60 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %61 : !frisk.async_task
      %62 = frisk.mul %arg5, %34 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %63 = frisk.mul %arg6, %34 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %64 = frisk.mul %arg7, %34 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %65 = frisk.mul %arg8, %34 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
      %66 = frisk.add %62, %48 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %67 = frisk.add %63, %53 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %68 = frisk.add %64, %58 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %69 = frisk.read_matrix_from_shm %60[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
      %70 = frisk.gemm(%40, %69, %65) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16>, memref<128x32xf32> -> memref<128x32xf32>
      frisk.copy %66 to %arg5 [#map1] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %67 to %arg6 [#map1] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %68 to %arg7 [#map1] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %70 to %arg8 [#map1] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %39 to %arg9 [#map1] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %31 to %arg10 [#map1] : memref<128x1xf32, 3>, vector<128x1xf32>
      affine.yield %arg5, %arg6, %arg7, %arg8, %arg9, %arg10 : vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>
    } {local_yield_transformed = true}
    %15 = frisk.div %14#0, %14#4 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %16 = frisk.div %14#1, %14#4 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %17 = frisk.div %14#2, %14#4 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %18 = frisk.div %14#3, %14#4 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %19 = frisk.cast %15 : memref<128x32xf32> -> memref<128x32xf16, 3>
    %20 = frisk.cast %16 : memref<128x32xf32> -> memref<128x32xf16, 3>
    %21 = frisk.cast %17 : memref<128x32xf32> -> memref<128x32xf16, 3>
    %22 = frisk.cast %18 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %19 to %arg3 at (%8, %7, %block_id_z) [#map2] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    frisk.copy %20 to %arg3 at (%8, %7, %block_id_z) [#map3] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    frisk.copy %21 to %arg3 at (%8, %7, %block_id_z) [#map4] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    frisk.copy %22 to %arg3 at (%8, %7, %block_id_z) [#map5] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    return
  }
}

