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
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 32, 0, 1, 0>, thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0xFF800000> : vector<2x1xf32>
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_10 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
    %c0 = arith.constant 0 : index
    %cst_11 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_12 = arith.constant 0xFF800000 : f32
    %cst_13 = arith.constant 0.127531052 : f32
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %0 = arith.muli %block_id_y, %c32 : index
    %1 = arith.addi %block_id_x, %0 : index
    %2 = arith.divui %1, %c1024 : index
    %3 = arith.muli %2, %c32 : index
    %4 = arith.remui %1, %c1024 : index
    %5 = arith.remui %4, %c32 : index
    %6 = arith.addi %3, %5 : index
    %7 = arith.divui %4, %c32 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %8 = arith.truncf %cst_13 : f32 to f16
    %9:3 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_11, %arg6 = %cst_11, %arg7 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %28 = frisk.pack_frag %cst_10 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
      %29 = frisk.insert_frag %28, %arg5[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      %30 = frisk.extract_frag %cst[%arg4, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %31 = frisk.elementwise_frag "add" %28, %30 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %32 = frisk.insert_frag %31, %arg6[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      %33 = frisk.elementwise_frag "cast" %31 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %34 = frisk.insert_frag %33, %arg7[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %29, %32, %34 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = frisk.to_threadTile %cst_9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %11 = frisk.to_threadTile %cst_8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %12:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#2, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %28 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %29 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %28) -> (vector<2x32xf32>) {
        %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %52 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %52 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %53 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_18 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %53 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %54 = vector.broadcast %8 : f16 to vector<2x8xf16>
        %55 = frisk.load_frag %alloc_18[%c0, %c0] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> <vector<1x4xf16>, {}>
        %56:3 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %cst_6, %arg15 = %cst_6, %arg16 = %55) -> (vector<2x8xf16>, vector<2x8xf16>, !frisk.fragment<vector<1x4xf16>, {}>) {
          %62 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg13)
          %63 = affine.apply affine_map<(d0) -> (d0 - ((d0 + 1) floordiv 2) * 2 + 1)>(%arg13)
          %64 = frisk.load_frag %alloc_18[%62, %63] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> <vector<1x4xf16>, {}>
          %65 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg13)
          %66 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg13)
          %67 = frisk.insert_frag %arg16, %arg14[%65, %66] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          %68 = frisk.extract_frag %67[%65, %66] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %69 = frisk.extract_frag %54[%65, %66] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %70 = frisk.elementwise_frag "mul" %68, %69 : (!frisk.fragment<vector<1x4xf16>, {}>, !frisk.fragment<vector<1x4xf16>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
          %71 = frisk.insert_frag %70, %arg15[%65, %66] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %67, %71, %64 : vector<2x8xf16>, vector<2x8xf16>, !frisk.fragment<vector<1x4xf16>, {}>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
        %57 = frisk.extract_frag %54[%c1, %c1] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        %58 = frisk.elementwise_frag "mul" %56#2, %57 : (!frisk.fragment<vector<1x4xf16>, {}>, !frisk.fragment<vector<1x4xf16>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %59 = frisk.insert_frag %58, %56#1[%c1, %c1] : <vector<1x4xf16>, {}> into vector<2x8xf16>
        frisk.wait_buffer_ready %alloc_19 : memref<32x128xf16, 3>
        %60 = frisk.pack_frag %cst_4 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %61:2 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5, %arg15 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
          %62 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %63 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %64 = frisk.load_frag %alloc_19[%c0, %63] {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 3, 0>, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tile_type = vector<8x8xf16>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %65:2 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %60, %arg18 = %64) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, !frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %75 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg16)
            %76 = frisk.load_frag %alloc_19[%75, %63] {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 3, 0>, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tile_type = vector<8x8xf16>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %77 = frisk.extract_frag %59[%62, %arg16] : vector<2x8xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %78 = frisk.gemm_frag %77, %arg18, acc = %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            affine.yield %78, %76 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, !frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
          %66 = frisk.extract_frag %59[%62, %c1] : vector<2x8xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %67 = frisk.gemm_frag %66, %65#1, acc = %65#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %68 = frisk.unpack_frag %67 : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> vector<1x4xf32>
          %69 = frisk.pack_frag %68 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
          %70 = frisk.extract_frag %arg12[%62, %63] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %71 = frisk.elementwise_frag "add" %70, %69 {frisk.fragment_accumulate} : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %72 = frisk.insert_frag %71, %arg14[%62, %63] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          %73 = frisk.extract_frag %72[%62, %63] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %74 = frisk.insert_frag %73, %arg15[%62, %63] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %72, %74 : vector<2x32xf32>, vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
        affine.yield %61#1 : vector<2x32xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %30 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %52 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %53 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %54 = frisk.extract_frag %29[%52, %53] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %55 = frisk.elementwise_frag "cast" %54 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %56 = frisk.insert_frag %55, %arg12[%52, %53] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %56 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %31:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11, %arg13 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %52 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_12) -> (f32) {
          %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %62 = vector.extract %30[%arg11, %61] : f32 from vector<2x32xf32>
          %63 = arith.maxnumf %arg15, %62 : f32
          %64 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %65 = vector.extract %30[%arg11, %64] : f32 from vector<2x32xf32>
          %66 = arith.maxnumf %63, %65 : f32
          %67 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %68 = vector.extract %30[%arg11, %67] : f32 from vector<2x32xf32>
          %69 = arith.maxnumf %66, %68 : f32
          %70 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %71 = vector.extract %30[%arg11, %70] : f32 from vector<2x32xf32>
          %72 = arith.maxnumf %69, %71 : f32
          affine.yield %72 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %52, %c16_i32, %c64_i32 : f32
        %53 = arith.maxnumf %52, %shuffleResult : f32
        %shuffleResult_18, %valid_19 = gpu.shuffle xor %53, %c32_i32, %c64_i32 : f32
        %54 = arith.maxnumf %53, %shuffleResult_18 : f32
        %55 = vector.broadcast %54 : f32 to vector<1x1xf32>
        %56 = frisk.pack_frag %55 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %57 = frisk.insert_frag %56, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        %58 = frisk.extract_frag %arg7[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %59 = frisk.elementwise_frag "max" %56, %58 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %60 = frisk.insert_frag %59, %arg13[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %57, %60 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %32:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %52 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %53 = vector.extract %31#1[%52, 0] : f32 from vector<2x1xf32>
        %54 = vector.from_elements %53, %53, %53, %53 : vector<1x4xf32>
        %55 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %56 = frisk.pack_frag %54 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %57 = frisk.insert_frag %56, %arg12[%52, %55] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        %58 = frisk.extract_frag %29[%52, %55] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %59 = frisk.extract_frag %57[%52, %55] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %60 = frisk.elementwise_frag "sub" %58, %59 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %61 = frisk.insert_frag %60, %arg13[%52, %55] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %57, %61 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>}
      %33:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11, %arg13 = %cst_11, %arg14 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %52 = frisk.extract_frag %arg7[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %53 = frisk.extract_frag %31#1[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %54 = frisk.elementwise_frag "sub" %52, %53 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %55 = frisk.insert_frag %54, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        %56 = frisk.elementwise_frag "exp2" %54 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %57 = frisk.insert_frag %56, %arg13[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        %58 = frisk.extract_frag %arg6[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %59 = frisk.elementwise_frag "mul" %58, %56 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %60 = frisk.insert_frag %59, %arg14[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %55, %57, %60 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %34:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %52 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %53 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %54 = frisk.extract_frag %32#1[%52, %53] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %55 = frisk.elementwise_frag "exp2" %54 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %56 = frisk.insert_frag %55, %arg12[%52, %53] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        %57 = frisk.extract_frag %56[%52, %53] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %58 = frisk.elementwise_frag "cast" %57 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %59 = frisk.insert_frag %58, %arg13[%52, %53] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %56, %59 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %35:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11, %arg13 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %52 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_3) -> (f32) {
          %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %62 = vector.extract %34#1[%arg11, %61] : f32 from vector<2x32xf32>
          %63 = arith.addf %arg15, %62 : f32
          %64 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %65 = vector.extract %34#1[%arg11, %64] : f32 from vector<2x32xf32>
          %66 = arith.addf %63, %65 : f32
          %67 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %68 = vector.extract %34#1[%arg11, %67] : f32 from vector<2x32xf32>
          %69 = arith.addf %66, %68 : f32
          %70 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %71 = vector.extract %34#1[%arg11, %70] : f32 from vector<2x32xf32>
          %72 = arith.addf %69, %71 : f32
          affine.yield %72 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %52, %c16_i32, %c64_i32 : f32
        %53 = arith.addf %52, %shuffleResult : f32
        %shuffleResult_18, %valid_19 = gpu.shuffle xor %53, %c32_i32, %c64_i32 : f32
        %54 = arith.addf %53, %shuffleResult_18 : f32
        %55 = vector.broadcast %54 : f32 to vector<1x1xf32>
        %56 = frisk.pack_frag %55 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %57 = frisk.insert_frag %56, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        %58 = frisk.extract_frag %33#2[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %59 = frisk.elementwise_frag "add" %56, %58 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %60 = frisk.insert_frag %59, %arg13[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %57, %60 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %36 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %52 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %53 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %54 = frisk.extract_frag %34#0[%52, %53] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %55 = frisk.elementwise_frag "cast" %54 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %56 = frisk.insert_frag %55, %arg12[%52, %53] : <vector<1x4xf16>, {}> into vector<2x32xf16>
        affine.yield %56 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %37 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %37 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_14 = gpu.thread_id x
      %38 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %52 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_14]
        %53 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_14]
        %54 = frisk.read_matrix_from_shm %alloc[%52, %53] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %55 = vector.bitcast %54 : vector<4xi32> to vector<8xf16>
        %56 = vector.extract %55[0] : f16 from vector<8xf16>
        %57 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_14]
        %58 = vector.extract %55[%57] : f16 from vector<8xf16>
        %59 = vector.extract %55[1] : f16 from vector<8xf16>
        %60 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_14]
        %61 = vector.extract %55[%60] : f16 from vector<8xf16>
        %62 = vector.extract %55[2] : f16 from vector<8xf16>
        %63 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_14]
        %64 = vector.extract %55[%63] : f16 from vector<8xf16>
        %65 = vector.extract %55[3] : f16 from vector<8xf16>
        %66 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_14]
        %67 = vector.extract %55[%66] : f16 from vector<8xf16>
        %68 = vector.from_elements %56, %58, %59, %61, %62, %64, %65, %67 : vector<4x2xf16>
        %69 = frisk.pack_frag %68 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %70 = frisk.insert_frag %69, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %70 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %39 = frisk.pack_frag %cst_4 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %40:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %52 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %53 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %54 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %39) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %68 = frisk.extract_frag %36[%52, %arg16] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %69 = frisk.extract_frag %38[%arg16, %53] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %70 = frisk.gemm_frag %68, %69, acc = %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %70 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %55 = frisk.insert_frag %54, %arg12[%52, %53] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        %56 = vector.extract %33#1[%52, 0] : f32 from vector<2x1xf32>
        %57 = vector.from_elements %56, %56, %56, %56 : vector<1x4xf32>
        %58 = frisk.pack_frag %57 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %59 = frisk.insert_frag %58, %arg13[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %60 = frisk.extract_frag %arg5[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %61 = frisk.extract_frag %59[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %62 = frisk.elementwise_frag "mul" %60, %61 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %63 = frisk.insert_frag %62, %arg14[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %64 = frisk.extract_frag %63[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %65 = frisk.extract_frag %55[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %66 = frisk.elementwise_frag "add" %64, %65 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %67 = frisk.insert_frag %66, %arg15[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %55, %59, %63, %67 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %41 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %41 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<128x32xf16, 3>
      %42 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %52 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_14]
        %53 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_14]
        %54 = frisk.read_matrix_from_shm %alloc_15[%52, %53] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %55 = vector.bitcast %54 : vector<4xi32> to vector<8xf16>
        %56 = vector.extract %55[0] : f16 from vector<8xf16>
        %57 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_14]
        %58 = vector.extract %55[%57] : f16 from vector<8xf16>
        %59 = vector.extract %55[1] : f16 from vector<8xf16>
        %60 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_14]
        %61 = vector.extract %55[%60] : f16 from vector<8xf16>
        %62 = vector.extract %55[2] : f16 from vector<8xf16>
        %63 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_14]
        %64 = vector.extract %55[%63] : f16 from vector<8xf16>
        %65 = vector.extract %55[3] : f16 from vector<8xf16>
        %66 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_14]
        %67 = vector.extract %55[%66] : f16 from vector<8xf16>
        %68 = vector.from_elements %56, %58, %59, %61, %62, %64, %65, %67 : vector<4x2xf16>
        %69 = frisk.pack_frag %68 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %70 = frisk.insert_frag %69, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %70 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %43:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %52 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %53 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %54 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %39) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %68 = frisk.extract_frag %36[%52, %arg16] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %69 = frisk.extract_frag %42[%arg16, %53] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %70 = frisk.gemm_frag %68, %69, acc = %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %70 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %55 = frisk.insert_frag %54, %arg12[%52, %53] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        %56 = vector.extract %33#1[%52, 0] : f32 from vector<2x1xf32>
        %57 = vector.from_elements %56, %56, %56, %56 : vector<1x4xf32>
        %58 = frisk.pack_frag %57 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %59 = frisk.insert_frag %58, %arg13[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %60 = frisk.extract_frag %arg8[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %61 = frisk.extract_frag %59[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %62 = frisk.elementwise_frag "mul" %60, %61 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %63 = frisk.insert_frag %62, %arg14[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %64 = frisk.extract_frag %63[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %65 = frisk.extract_frag %55[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %66 = frisk.elementwise_frag "add" %64, %65 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %67 = frisk.insert_frag %66, %arg15[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %55, %59, %63, %67 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %44 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %44 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_16 : memref<128x32xf16, 3>
      %45 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %52 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_14]
        %53 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_14]
        %54 = frisk.read_matrix_from_shm %alloc_16[%52, %53] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %55 = vector.bitcast %54 : vector<4xi32> to vector<8xf16>
        %56 = vector.extract %55[0] : f16 from vector<8xf16>
        %57 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_14]
        %58 = vector.extract %55[%57] : f16 from vector<8xf16>
        %59 = vector.extract %55[1] : f16 from vector<8xf16>
        %60 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_14]
        %61 = vector.extract %55[%60] : f16 from vector<8xf16>
        %62 = vector.extract %55[2] : f16 from vector<8xf16>
        %63 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_14]
        %64 = vector.extract %55[%63] : f16 from vector<8xf16>
        %65 = vector.extract %55[3] : f16 from vector<8xf16>
        %66 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_14]
        %67 = vector.extract %55[%66] : f16 from vector<8xf16>
        %68 = vector.from_elements %56, %58, %59, %61, %62, %64, %65, %67 : vector<4x2xf16>
        %69 = frisk.pack_frag %68 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %70 = frisk.insert_frag %69, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %70 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %46:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %52 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %53 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %54 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %39) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %68 = frisk.extract_frag %36[%52, %arg16] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %69 = frisk.extract_frag %45[%arg16, %53] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %70 = frisk.gemm_frag %68, %69, acc = %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %70 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %55 = frisk.insert_frag %54, %arg12[%52, %53] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        %56 = vector.extract %33#1[%52, 0] : f32 from vector<2x1xf32>
        %57 = vector.from_elements %56, %56, %56, %56 : vector<1x4xf32>
        %58 = frisk.pack_frag %57 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %59 = frisk.insert_frag %58, %arg13[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %60 = frisk.extract_frag %arg9[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %61 = frisk.extract_frag %59[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %62 = frisk.elementwise_frag "mul" %60, %61 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %63 = frisk.insert_frag %62, %arg14[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %64 = frisk.extract_frag %63[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %65 = frisk.extract_frag %55[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %66 = frisk.elementwise_frag "add" %64, %65 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %67 = frisk.insert_frag %66, %arg15[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %55, %59, %63, %67 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %47 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %47 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_17 : memref<128x32xf16, 3>
      %48 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %52 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_14]
        %53 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_14]
        %54 = frisk.read_matrix_from_shm %alloc_17[%52, %53] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %55 = vector.bitcast %54 : vector<4xi32> to vector<8xf16>
        %56 = vector.extract %55[0] : f16 from vector<8xf16>
        %57 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_14]
        %58 = vector.extract %55[%57] : f16 from vector<8xf16>
        %59 = vector.extract %55[1] : f16 from vector<8xf16>
        %60 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_14]
        %61 = vector.extract %55[%60] : f16 from vector<8xf16>
        %62 = vector.extract %55[2] : f16 from vector<8xf16>
        %63 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_14]
        %64 = vector.extract %55[%63] : f16 from vector<8xf16>
        %65 = vector.extract %55[3] : f16 from vector<8xf16>
        %66 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_14]
        %67 = vector.extract %55[%66] : f16 from vector<8xf16>
        %68 = vector.from_elements %56, %58, %59, %61, %62, %64, %65, %67 : vector<4x2xf16>
        %69 = frisk.pack_frag %68 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %70 = frisk.insert_frag %69, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %70 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %49:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %52 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %53 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %54 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %39) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %70 = frisk.extract_frag %36[%52, %arg17] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %71 = frisk.extract_frag %48[%arg17, %53] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %72 = frisk.gemm_frag %70, %71, acc = %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %72 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %55 = frisk.insert_frag %54, %arg12[%52, %53] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        %56 = vector.extract %33#1[%52, 0] : f32 from vector<2x1xf32>
        %57 = vector.from_elements %56, %56, %56, %56 : vector<1x4xf32>
        %58 = frisk.pack_frag %57 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %59 = frisk.insert_frag %58, %arg13[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %60 = frisk.extract_frag %arg10[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %61 = frisk.extract_frag %59[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %62 = frisk.elementwise_frag "mul" %60, %61 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %63 = frisk.insert_frag %62, %arg14[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %64 = frisk.extract_frag %63[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %65 = frisk.extract_frag %55[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %66 = frisk.elementwise_frag "add" %64, %65 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %67 = frisk.insert_frag %66, %arg15[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %68 = frisk.extract_frag %40#3[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %69 = frisk.insert_frag %68, %arg16[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %55, %59, %63, %67, %69 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %50:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11, %arg13 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %52 = frisk.extract_frag %35#1[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %53 = frisk.insert_frag %52, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        %54 = frisk.extract_frag %31#1[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %55 = frisk.insert_frag %54, %arg13[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %53, %55 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %51:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %52 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %53 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %54 = frisk.extract_frag %43#3[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %55 = frisk.insert_frag %54, %arg12[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %56 = frisk.extract_frag %46#3[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %57 = frisk.insert_frag %56, %arg13[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %58 = frisk.extract_frag %49#3[%52, %53] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %59 = frisk.insert_frag %58, %arg14[%52, %53] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %55, %57, %59 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %49#4, %50#0, %50#1, %51#0, %51#1, %51#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {local_yield_transformed = true}
    %thread_id_x = gpu.thread_id x
    %13 = arith.index_cast %thread_id_x : index to i32
    %14 = arith.andi %13, %c16_i32 : i32
    %15 = arith.cmpi ne, %14, %c0_i32 : i32
    %16 = arith.andi %13, %c32_i32 : i32
    %17 = arith.cmpi ne, %16, %c0_i32 : i32
    %18 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %19:8 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_0, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_6, %arg12 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %28 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %29 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %30 = frisk.extract_frag %12#5[%28, %29] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %31 = frisk.unpack_frag %30 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
      %32 = vector.extract %31[0, 0] : f32 from vector<1x4xf32>
      %33 = vector.extract %31[0, 1] : f32 from vector<1x4xf32>
      %34 = vector.extract %31[0, 2] : f32 from vector<1x4xf32>
      %35 = vector.extract %31[0, 3] : f32 from vector<1x4xf32>
      %36 = arith.select %15, %32, %33 : f32
      %shuffleResult, %valid = gpu.shuffle xor %36, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %37 = arith.select %15, %shuffleResult, %32 : f32
      %38 = arith.select %15, %33, %shuffleResult : f32
      %39 = arith.select %15, %34, %35 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %39, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %40 = arith.select %15, %shuffleResult_14, %34 : f32
      %41 = arith.select %15, %35, %shuffleResult_14 : f32
      %42 = arith.select %17, %37, %40 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %42, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %43 = arith.select %17, %shuffleResult_16, %37 : f32
      %44 = arith.select %17, %40, %shuffleResult_16 : f32
      %45 = arith.select %17, %38, %41 : f32
      %shuffleResult_18, %valid_19 = gpu.shuffle xor %45, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %46 = arith.select %17, %shuffleResult_18, %38 : f32
      %47 = arith.select %17, %41, %shuffleResult_18 : f32
      %48 = vector.from_elements %43, %46, %44, %47 : vector<1x4xf32>
      %49 = frisk.pack_frag %48 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %50 = frisk.insert_frag %49, %arg5[%28, %29] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %51 = frisk.extract_frag %12#4[%28, %29] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %52 = frisk.unpack_frag %51 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
      %53 = vector.extract %52[0, 0] : f32 from vector<1x4xf32>
      %54 = vector.extract %52[0, 1] : f32 from vector<1x4xf32>
      %55 = vector.extract %52[0, 2] : f32 from vector<1x4xf32>
      %56 = vector.extract %52[0, 3] : f32 from vector<1x4xf32>
      %57 = arith.select %15, %53, %54 : f32
      %shuffleResult_20, %valid_21 = gpu.shuffle xor %57, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %58 = arith.select %15, %shuffleResult_20, %53 : f32
      %59 = arith.select %15, %54, %shuffleResult_20 : f32
      %60 = arith.select %15, %55, %56 : f32
      %shuffleResult_22, %valid_23 = gpu.shuffle xor %60, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %61 = arith.select %15, %shuffleResult_22, %55 : f32
      %62 = arith.select %15, %56, %shuffleResult_22 : f32
      %63 = arith.select %17, %58, %61 : f32
      %shuffleResult_24, %valid_25 = gpu.shuffle xor %63, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %64 = arith.select %17, %shuffleResult_24, %58 : f32
      %65 = arith.select %17, %61, %shuffleResult_24 : f32
      %66 = arith.select %17, %59, %62 : f32
      %shuffleResult_26, %valid_27 = gpu.shuffle xor %66, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %67 = arith.select %17, %shuffleResult_26, %59 : f32
      %68 = arith.select %17, %62, %shuffleResult_26 : f32
      %69 = vector.from_elements %64, %67, %65, %68 : vector<1x4xf32>
      %70 = frisk.pack_frag %69 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %71 = frisk.insert_frag %70, %arg6[%28, %29] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %72 = frisk.extract_frag %12#3[%28, %29] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = frisk.unpack_frag %72 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
      %74 = vector.extract %73[0, 0] : f32 from vector<1x4xf32>
      %75 = vector.extract %73[0, 1] : f32 from vector<1x4xf32>
      %76 = vector.extract %73[0, 2] : f32 from vector<1x4xf32>
      %77 = vector.extract %73[0, 3] : f32 from vector<1x4xf32>
      %78 = arith.select %15, %74, %75 : f32
      %shuffleResult_28, %valid_29 = gpu.shuffle xor %78, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %79 = arith.select %15, %shuffleResult_28, %74 : f32
      %80 = arith.select %15, %75, %shuffleResult_28 : f32
      %81 = arith.select %15, %76, %77 : f32
      %shuffleResult_30, %valid_31 = gpu.shuffle xor %81, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %82 = arith.select %15, %shuffleResult_30, %76 : f32
      %83 = arith.select %15, %77, %shuffleResult_30 : f32
      %84 = arith.select %17, %79, %82 : f32
      %shuffleResult_32, %valid_33 = gpu.shuffle xor %84, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %85 = arith.select %17, %shuffleResult_32, %79 : f32
      %86 = arith.select %17, %82, %shuffleResult_32 : f32
      %87 = arith.select %17, %80, %83 : f32
      %shuffleResult_34, %valid_35 = gpu.shuffle xor %87, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %88 = arith.select %17, %shuffleResult_34, %80 : f32
      %89 = arith.select %17, %83, %shuffleResult_34 : f32
      %90 = vector.from_elements %85, %88, %86, %89 : vector<1x4xf32>
      %91 = frisk.pack_frag %90 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %92 = frisk.insert_frag %91, %arg7[%28, %29] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %93 = frisk.extract_frag %12#0[%28, %29] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %94 = frisk.unpack_frag %93 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
      %95 = vector.extract %94[0, 0] : f32 from vector<1x4xf32>
      %96 = vector.extract %94[0, 1] : f32 from vector<1x4xf32>
      %97 = vector.extract %94[0, 2] : f32 from vector<1x4xf32>
      %98 = vector.extract %94[0, 3] : f32 from vector<1x4xf32>
      %99 = arith.select %15, %95, %96 : f32
      %shuffleResult_36, %valid_37 = gpu.shuffle xor %99, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %100 = arith.select %15, %shuffleResult_36, %95 : f32
      %101 = arith.select %15, %96, %shuffleResult_36 : f32
      %102 = arith.select %15, %97, %98 : f32
      %shuffleResult_38, %valid_39 = gpu.shuffle xor %102, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %103 = arith.select %15, %shuffleResult_38, %97 : f32
      %104 = arith.select %15, %98, %shuffleResult_38 : f32
      %105 = arith.select %17, %100, %103 : f32
      %shuffleResult_40, %valid_41 = gpu.shuffle xor %105, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %106 = arith.select %17, %shuffleResult_40, %100 : f32
      %107 = arith.select %17, %103, %shuffleResult_40 : f32
      %108 = arith.select %17, %101, %104 : f32
      %shuffleResult_42, %valid_43 = gpu.shuffle xor %108, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %109 = arith.select %17, %shuffleResult_42, %101 : f32
      %110 = arith.select %17, %104, %shuffleResult_42 : f32
      %111 = vector.from_elements %106, %109, %107, %110 : vector<1x4xf32>
      %112 = frisk.pack_frag %111 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %113 = frisk.insert_frag %112, %arg8[%28, %29] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %114 = vector.extract %12#1[%28, 0] : f32 from vector<2x1xf32>
      %115 = vector.from_elements %114, %114, %114, %114 : vector<1x4xf32>
      %116 = frisk.pack_frag %115 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %117 = frisk.insert_frag %116, %arg9[%28, %29] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %118 = frisk.extract_frag %113[%28, %29] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %119 = frisk.extract_frag %117[%28, %29] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %120 = frisk.elementwise_frag "div" %118, %119 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %121 = frisk.insert_frag %120, %arg10[%28, %29] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %122 = frisk.extract_frag %121[%28, %29] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %123 = frisk.elementwise_frag "cast" %122 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %124 = frisk.insert_frag %123, %arg11[%28, %29] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      %125 = frisk.extract_frag %124[%28, %29] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %126 = frisk.insert_frag %125, %arg12[%28, %29] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %50, %71, %92, %113, %117, %121, %124, %126 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    %20 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %21 = arith.cmpi ult, %20, %c4 : index
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        %28 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %29 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %30 = frisk.extract_frag %19#7[%28, %29] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %30, %18[%28, %29] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %22 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %23:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %28 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %29 = vector.extract %12#1[%28, 0] : f32 from vector<2x1xf32>
      %30 = vector.from_elements %29, %29, %29, %29 : vector<1x4xf32>
      %31 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %32 = frisk.pack_frag %30 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %33 = frisk.insert_frag %32, %arg5[%28, %31] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %34 = frisk.extract_frag %19#2[%28, %31] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %35 = frisk.extract_frag %33[%28, %31] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %36 = frisk.elementwise_frag "div" %34, %35 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %37 = frisk.insert_frag %36, %arg6[%28, %31] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %38 = frisk.extract_frag %37[%28, %31] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %39 = frisk.elementwise_frag "cast" %38 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %40 = frisk.insert_frag %39, %arg7[%28, %31] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      %41 = frisk.extract_frag %40[%28, %31] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %42 = frisk.insert_frag %41, %arg8[%28, %31] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %33, %37, %40, %42 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        %28 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %29 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %30 = frisk.extract_frag %23#3[%28, %29] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %30, %22[%28, %29] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %24 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %25:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %28 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %29 = vector.extract %12#1[%28, 0] : f32 from vector<2x1xf32>
      %30 = vector.from_elements %29, %29, %29, %29 : vector<1x4xf32>
      %31 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %32 = frisk.pack_frag %30 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %33 = frisk.insert_frag %32, %arg5[%28, %31] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %34 = frisk.extract_frag %19#1[%28, %31] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %35 = frisk.extract_frag %33[%28, %31] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %36 = frisk.elementwise_frag "div" %34, %35 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %37 = frisk.insert_frag %36, %arg6[%28, %31] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %38 = frisk.extract_frag %37[%28, %31] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %39 = frisk.elementwise_frag "cast" %38 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %40 = frisk.insert_frag %39, %arg7[%28, %31] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      %41 = frisk.extract_frag %40[%28, %31] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %42 = frisk.insert_frag %41, %arg8[%28, %31] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %33, %37, %40, %42 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        %28 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %29 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %30 = frisk.extract_frag %25#3[%28, %29] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %30, %24[%28, %29] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %26 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %27:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %28 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %29 = vector.extract %12#1[%28, 0] : f32 from vector<2x1xf32>
      %30 = vector.from_elements %29, %29, %29, %29 : vector<1x4xf32>
      %31 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %32 = frisk.pack_frag %30 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %33 = frisk.insert_frag %32, %arg5[%28, %31] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %34 = frisk.extract_frag %19#0[%28, %31] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %35 = frisk.extract_frag %33[%28, %31] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %36 = frisk.elementwise_frag "div" %34, %35 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %37 = frisk.insert_frag %36, %arg6[%28, %31] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %38 = frisk.extract_frag %37[%28, %31] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %39 = frisk.elementwise_frag "cast" %38 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %40 = frisk.insert_frag %39, %arg7[%28, %31] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      %41 = frisk.extract_frag %40[%28, %31] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %42 = frisk.insert_frag %41, %arg8[%28, %31] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %33, %37, %40, %42 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        %28 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %29 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %30 = frisk.extract_frag %27#3[%28, %29] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %30, %26[%28, %29] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
