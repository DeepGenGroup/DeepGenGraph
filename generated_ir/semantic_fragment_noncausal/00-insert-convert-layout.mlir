func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 32, 0, 1, 0>, thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
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
  %cst_4 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
  %11 = frisk.cast %10 : memref<128x1xf32> -> vector<128x1xf32>
  %cst_5 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
  %cst_6 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
  %cst_7 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
  %12:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_3, %arg6 = %cst_4, %arg7 = %11, %arg8 = %cst_5, %arg9 = %cst_6, %arg10 = %cst_7) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
    %26 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_8) -> (vector<128x128xf32>) {
      %64 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %65 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
      %66 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %66 to %64 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      %67 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.wait_buffer_ready %64 : memref<128x32xf16, 3>
      frisk.async_copy_gm_to_shm %67 to %65 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      %68 = frisk.mul %64, %8 : memref<128x32xf16, 3>, f16 -> memref<128x32xf16, 3>
      frisk.wait_buffer_ready %65 : memref<32x128xf16, 3>
      %69 = frisk.gemm(%68, %65) {frisk.a_input_layout_method = "b-permute", frisk.b_permute, frisk.contiguous_c, frisk.wmma_layout_applied = "b-permute", inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3> -> memref<128x128xf32>
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
    frisk.async_copy_gm_to_shm %40 to %39 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %39 : memref<128x32xf16, 3>
    %41 = frisk.read_matrix_from_shm %39[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
    %42 = frisk.gemm(%38, %41) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16> -> memref<128x32xf32>
    %43 = frisk.convert_layout %32 {frisk.layout_role = "wmmaC", frisk.source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.wmma_instruction = {name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", shape = array<i64: 16, 16, 16>}, value = "lowerinfo.convert"} : memref<128x1xf32> -> memref<128x1xf32>
    %44 = frisk.mul %arg5, %43 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
    %45 = frisk.add %44, %42 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
    %46 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
    %47 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %47 to %46 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %46 : memref<128x32xf16, 3>
    %48 = frisk.read_matrix_from_shm %46[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
    %49 = frisk.gemm(%38, %48) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16> -> memref<128x32xf32>
    %50 = frisk.mul %arg8, %43 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
    %51 = frisk.add %50, %49 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
    %52 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
    %53 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %53 to %52 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %52 : memref<128x32xf16, 3>
    %54 = frisk.read_matrix_from_shm %52[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
    %55 = frisk.gemm(%38, %54) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16> -> memref<128x32xf32>
    %56 = frisk.mul %arg9, %43 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
    %57 = frisk.add %56, %55 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
    %58 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
    %59 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %59 to %58 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %58 : memref<128x32xf16, 3>
    %60 = frisk.read_matrix_from_shm %58[] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<128x32xf16, 3> -> memref<128x32xf16>
    %61 = frisk.gemm(%38, %60) {frisk.contiguous_c, inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16> -> memref<128x32xf32>
    %62 = frisk.mul %arg10, %43 : vector<128x32xf32>, memref<128x1xf32> -> memref<128x32xf32>
    %63 = frisk.add %62, %61 : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
    frisk.copy %45 to %arg5 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
    frisk.copy %37 to %arg6 [affine_map<() -> (2)>] : memref<128x1xf32, 3>, vector<128x1xf32>
    frisk.copy %29 to %arg7 [affine_map<() -> (2)>] : memref<128x1xf32, 3>, vector<128x1xf32>
    frisk.copy %51 to %arg8 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
    frisk.copy %57 to %arg9 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
    frisk.copy %63 to %arg10 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
    affine.yield %arg5, %arg6, %arg7, %arg8, %arg9, %arg10 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
  } {local_yield_transformed = true}
  %13 = frisk.convert_layout %12#5 {frisk.layout_role = "wmmaC", frisk.shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, frisk.source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.wmma_instruction = {name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", shape = array<i64: 16, 16, 16>}, value = "loop_result"} : vector<128x32xf32> -> vector<128x32xf32>
  %14 = frisk.convert_layout %12#4 {frisk.layout_role = "wmmaC", frisk.shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, frisk.source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.wmma_instruction = {name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", shape = array<i64: 16, 16, 16>}, value = "loop_result"} : vector<128x32xf32> -> vector<128x32xf32>
  %15 = frisk.convert_layout %12#3 {frisk.layout_role = "wmmaC", frisk.shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, frisk.source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.wmma_instruction = {name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", shape = array<i64: 16, 16, 16>}, value = "loop_result"} : vector<128x32xf32> -> vector<128x32xf32>
  %16 = frisk.convert_layout %12#0 {frisk.layout_role = "wmmaC", frisk.shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, frisk.source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.wmma_instruction = {name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", shape = array<i64: 16, 16, 16>}, value = "loop_result"} : vector<128x32xf32> -> vector<128x32xf32>
  %17 = frisk.convert_layout %12#1 {frisk.layout_role = "wmmaC", frisk.source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, frisk.wmma_instruction = {name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", shape = array<i64: 16, 16, 16>}, value = "lowerinfo.convert"} : vector<128x1xf32> -> vector<128x1xf32>
  %18 = frisk.div %16, %17 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
  %19 = frisk.cast %18 : memref<128x32xf32> -> memref<128x32xf16, 3>
  frisk.copy %19 to %arg3 at (%6, %7, %block_id_z) [affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 0)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
  %20 = frisk.div %15, %17 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
  %21 = frisk.cast %20 : memref<128x32xf32> -> memref<128x32xf16, 3>
  frisk.copy %21 to %arg3 at (%6, %7, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 32) floordiv 524288) mod 32, (d0 * 128) mod 4096, 32)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
  %22 = frisk.div %14, %17 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
  %23 = frisk.cast %22 : memref<128x32xf32> -> memref<128x32xf16, 3>
  frisk.copy %23 to %arg3 at (%6, %7, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 64) floordiv 524288) mod 32, (d0 * 128) mod 4096, 64)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
  %24 = frisk.div %13, %17 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
  %25 = frisk.cast %24 : memref<128x32xf32> -> memref<128x32xf16, 3>
  frisk.copy %25 to %arg3 at (%6, %7, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 96) floordiv 524288) mod 32, (d0 * 128) mod 4096, 96)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
  return
}
