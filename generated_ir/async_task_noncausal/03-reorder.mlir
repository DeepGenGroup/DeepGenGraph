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
    %c4 = arith.constant 4 : index
    %cst = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_8 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
    %c0 = arith.constant 0 : index
    %cst_9 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_10 = arith.constant 0xFF800000 : f32
    %cst_11 = arith.constant 0.127531052 : f32
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
    %8 = arith.truncf %cst_11 : f32 to f16
    %9:4 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_9, %arg6 = %cst_9, %arg7 = %cst_9, %arg8 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %35 = frisk.pack_frag %cst_8 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
      %36 = frisk.insert_frag %35, %arg5[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      %37 = frisk.broadcast_frag %cst_10 : f32 -> <vector<1x1xf32>, {}>
      %38 = frisk.insert_frag %37, %arg6[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      %39 = frisk.elementwise_frag "add" %35, %37 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %40 = frisk.insert_frag %39, %arg7[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      %41 = frisk.elementwise_frag "cast" %39 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %42 = frisk.insert_frag %41, %arg8[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %36, %38, %40, %42 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %11 = frisk.to_threadTile %cst_6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %12:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#3, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %35 = frisk.to_threadTile %cst_5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %36 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %35) -> (vector<2x32xf32>) {
        %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %69 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        %70 = frisk.async_copy_gm_to_shm %69 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
        %71 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %70 : !frisk.async_task
        %72 = frisk.async_copy_gm_to_shm %71 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
        %73 = frisk.load_frag %alloc_16[%c0, %c0] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> <vector<1x4xf16>, {}>
        %74:4 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %cst_4, %arg15 = %cst_4, %arg16 = %cst_4, %arg17 = %73) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, !frisk.fragment<vector<1x4xf16>, {}>) {
          %80 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg13)
          %81 = affine.apply affine_map<(d0) -> (d0 - ((d0 + 1) floordiv 2) * 2 + 1)>(%arg13)
          %82 = frisk.load_frag %alloc_16[%80, %81] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> <vector<1x4xf16>, {}>
          %83 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg13)
          %84 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg13)
          %85 = frisk.insert_frag %arg17, %arg14[%83, %84] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          %86 = frisk.broadcast_frag %8 : f16 -> <vector<1x4xf16>, {}>
          %87 = frisk.insert_frag %86, %arg15[%83, %84] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          %88 = frisk.extract_frag %85[%83, %84] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %89 = frisk.extract_frag %87[%83, %84] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %90 = frisk.elementwise_frag "mul" %88, %89 : (!frisk.fragment<vector<1x4xf16>, {}>, !frisk.fragment<vector<1x4xf16>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
          %91 = frisk.insert_frag %90, %arg16[%83, %84] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %85, %87, %91, %82 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, !frisk.fragment<vector<1x4xf16>, {}>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
        %75 = frisk.broadcast_frag %8 : f16 -> <vector<1x4xf16>, {}>
        %76 = frisk.elementwise_frag "mul" %74#3, %75 : (!frisk.fragment<vector<1x4xf16>, {}>, !frisk.fragment<vector<1x4xf16>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %77 = frisk.insert_frag %76, %74#2[%c1, %c1] : <vector<1x4xf16>, {}> into vector<2x8xf16>
        frisk.wait_buffer_ready %72 : !frisk.async_task
        %78 = frisk.pack_frag %cst_2 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %79:2 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_3, %arg15 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
          %80 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %81 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %82 = frisk.load_frag %alloc_17[%c0, %81] {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 3, 0>, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tile_type = vector<8x8xf16>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %83:2 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %78, %arg18 = %82) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, !frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %93 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg16)
            %94 = frisk.load_frag %alloc_17[%93, %81] {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 3, 0>, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tile_type = vector<8x8xf16>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %95 = frisk.extract_frag %77[%80, %arg16] : vector<2x8xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %96 = frisk.gemm_frag %95, %arg18, acc = %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            affine.yield %96, %94 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, !frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
          %84 = frisk.extract_frag %77[%80, %c1] : vector<2x8xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %85 = frisk.gemm_frag %84, %83#1, acc = %83#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %86 = frisk.unpack_frag %85 : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> vector<1x4xf32>
          %87 = frisk.pack_frag %86 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
          %88 = frisk.extract_frag %arg12[%80, %81] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %89 = frisk.elementwise_frag "add" %88, %87 {frisk.fragment_accumulate} : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %90 = frisk.insert_frag %89, %arg14[%80, %81] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          %91 = frisk.extract_frag %90[%80, %81] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %92 = frisk.insert_frag %91, %arg15[%80, %81] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %90, %92 : vector<2x32xf32>, vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
        affine.yield %79#1 : vector<2x32xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %37 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %69 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %70 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %71 = frisk.extract_frag %36[%69, %70] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %72 = frisk.elementwise_frag "cast" %71 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %73 = frisk.insert_frag %72, %arg12[%69, %70] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %73 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %38 = frisk.pack_frag %37 : vector<2x32xf32> -> <vector<2x32xf32>, {}>
      %39 = frisk.reduce_frag "max" %38 {dim = 1 : i64, lane_extent = 4 : i64, lane_stride = 16 : i64, output_fragment_shape = array<i64: 1, 1>, reduction_width = 4 : i64, warp_threads = 64 : i64} : <vector<2x32xf32>, {}> -> <vector<2x1xf32>, {}>
      %40 = frisk.unpack_frag %39 : <vector<2x1xf32>, {}> -> vector<2x1xf32>
      %41 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %69 = frisk.extract_frag %40[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %70 = frisk.extract_frag %arg7[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %71 = frisk.elementwise_frag "max" %69, %70 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %72 = frisk.insert_frag %71, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %72 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %42:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %69 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %70 = frisk.extract_frag %41[%69, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %71 = frisk.broadcast_frag %70 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %72 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %73 = frisk.insert_frag %71, %arg12[%69, %72] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        %74 = frisk.extract_frag %36[%69, %72] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %75 = frisk.extract_frag %73[%69, %72] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %76 = frisk.elementwise_frag "sub" %74, %75 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %77 = frisk.insert_frag %76, %arg13[%69, %72] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %73, %77 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %43:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9, %arg14 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %69 = frisk.extract_frag %arg7[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %70 = frisk.extract_frag %41[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %71 = frisk.elementwise_frag "sub" %69, %70 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %72 = frisk.insert_frag %71, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        %73 = frisk.elementwise_frag "exp2" %71 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %74 = frisk.insert_frag %73, %arg13[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        %75 = frisk.extract_frag %arg6[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %76 = frisk.elementwise_frag "mul" %75, %73 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %77 = frisk.insert_frag %76, %arg14[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %72, %74, %77 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %44:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %69 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %70 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %71 = frisk.extract_frag %42#1[%69, %70] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %72 = frisk.elementwise_frag "exp2" %71 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %73 = frisk.insert_frag %72, %arg12[%69, %70] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        %74 = frisk.extract_frag %73[%69, %70] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %75 = frisk.elementwise_frag "cast" %74 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %76 = frisk.insert_frag %75, %arg13[%69, %70] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %73, %76 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %45 = frisk.pack_frag %44#1 : vector<2x32xf32> -> <vector<2x32xf32>, {}>
      %46 = frisk.reduce_frag "add" %45 {dim = 1 : i64, lane_extent = 4 : i64, lane_stride = 16 : i64, output_fragment_shape = array<i64: 1, 1>, reduction_width = 4 : i64, warp_threads = 64 : i64} : <vector<2x32xf32>, {}> -> <vector<2x1xf32>, {}>
      %47 = frisk.unpack_frag %46 : <vector<2x1xf32>, {}> -> vector<2x1xf32>
      %48 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %69 = frisk.extract_frag %47[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %70 = frisk.extract_frag %43#2[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %71 = frisk.elementwise_frag "add" %69, %70 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %72 = frisk.insert_frag %71, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %72 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %49 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_1) -> (vector<2x32xf16>) {
        %69 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %70 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %71 = frisk.extract_frag %44#0[%69, %70] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %72 = frisk.elementwise_frag "cast" %71 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %73 = frisk.insert_frag %72, %arg12[%69, %70] : <vector<1x4xf16>, {}> into vector<2x32xf16>
        affine.yield %73 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %50 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %51 = frisk.async_copy_gm_to_shm %50 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %51 : !frisk.async_task
      %thread_id_x_12 = gpu.thread_id x
      %52 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %69 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_12]
        %70 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_12]
        %71 = frisk.read_matrix_from_shm %alloc[%69, %70] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %72 = vector.bitcast %71 : vector<4xi32> to vector<8xf16>
        %73 = vector.extract %72[0] : f16 from vector<8xf16>
        %74 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_12]
        %75 = vector.extract %72[%74] : f16 from vector<8xf16>
        %76 = vector.extract %72[1] : f16 from vector<8xf16>
        %77 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_12]
        %78 = vector.extract %72[%77] : f16 from vector<8xf16>
        %79 = vector.extract %72[2] : f16 from vector<8xf16>
        %80 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_12]
        %81 = vector.extract %72[%80] : f16 from vector<8xf16>
        %82 = vector.extract %72[3] : f16 from vector<8xf16>
        %83 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_12]
        %84 = vector.extract %72[%83] : f16 from vector<8xf16>
        %85 = vector.from_elements %73, %75, %76, %78, %79, %81, %82, %84 : vector<4x2xf16>
        %86 = frisk.pack_frag %85 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %87 = frisk.insert_frag %86, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %87 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %53 = frisk.pack_frag %cst_2 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %54:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %69 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %70 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %71 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %53) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %84 = frisk.extract_frag %49[%69, %arg16] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %85 = frisk.extract_frag %52[%arg16, %70] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %86 = frisk.gemm_frag %84, %85, acc = %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %86 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %72 = frisk.insert_frag %71, %arg12[%69, %70] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        %73 = frisk.extract_frag %43#1[%69, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %74 = frisk.broadcast_frag %73 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %75 = frisk.insert_frag %74, %arg13[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %76 = frisk.extract_frag %arg5[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %77 = frisk.extract_frag %75[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %78 = frisk.elementwise_frag "mul" %76, %77 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %79 = frisk.insert_frag %78, %arg14[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %80 = frisk.extract_frag %79[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %81 = frisk.extract_frag %72[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %82 = frisk.elementwise_frag "add" %80, %81 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %83 = frisk.insert_frag %82, %arg15[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %72, %75, %79, %83 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_13 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %55 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %56 = frisk.async_copy_gm_to_shm %55 to %alloc_13 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %56 : !frisk.async_task
      %57 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %69 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_12]
        %70 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_12]
        %71 = frisk.read_matrix_from_shm %alloc_13[%69, %70] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %72 = vector.bitcast %71 : vector<4xi32> to vector<8xf16>
        %73 = vector.extract %72[0] : f16 from vector<8xf16>
        %74 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_12]
        %75 = vector.extract %72[%74] : f16 from vector<8xf16>
        %76 = vector.extract %72[1] : f16 from vector<8xf16>
        %77 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_12]
        %78 = vector.extract %72[%77] : f16 from vector<8xf16>
        %79 = vector.extract %72[2] : f16 from vector<8xf16>
        %80 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_12]
        %81 = vector.extract %72[%80] : f16 from vector<8xf16>
        %82 = vector.extract %72[3] : f16 from vector<8xf16>
        %83 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_12]
        %84 = vector.extract %72[%83] : f16 from vector<8xf16>
        %85 = vector.from_elements %73, %75, %76, %78, %79, %81, %82, %84 : vector<4x2xf16>
        %86 = frisk.pack_frag %85 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %87 = frisk.insert_frag %86, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %87 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %58:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %69 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %70 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %71 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %53) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %84 = frisk.extract_frag %49[%69, %arg16] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %85 = frisk.extract_frag %57[%arg16, %70] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %86 = frisk.gemm_frag %84, %85, acc = %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %86 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %72 = frisk.insert_frag %71, %arg12[%69, %70] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        %73 = frisk.extract_frag %43#1[%69, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %74 = frisk.broadcast_frag %73 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %75 = frisk.insert_frag %74, %arg13[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %76 = frisk.extract_frag %arg8[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %77 = frisk.extract_frag %75[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %78 = frisk.elementwise_frag "mul" %76, %77 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %79 = frisk.insert_frag %78, %arg14[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %80 = frisk.extract_frag %79[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %81 = frisk.extract_frag %72[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %82 = frisk.elementwise_frag "add" %80, %81 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %83 = frisk.insert_frag %82, %arg15[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %72, %75, %79, %83 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_14 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %59 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %60 = frisk.async_copy_gm_to_shm %59 to %alloc_14 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %60 : !frisk.async_task
      %61 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %69 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_12]
        %70 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_12]
        %71 = frisk.read_matrix_from_shm %alloc_14[%69, %70] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %72 = vector.bitcast %71 : vector<4xi32> to vector<8xf16>
        %73 = vector.extract %72[0] : f16 from vector<8xf16>
        %74 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_12]
        %75 = vector.extract %72[%74] : f16 from vector<8xf16>
        %76 = vector.extract %72[1] : f16 from vector<8xf16>
        %77 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_12]
        %78 = vector.extract %72[%77] : f16 from vector<8xf16>
        %79 = vector.extract %72[2] : f16 from vector<8xf16>
        %80 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_12]
        %81 = vector.extract %72[%80] : f16 from vector<8xf16>
        %82 = vector.extract %72[3] : f16 from vector<8xf16>
        %83 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_12]
        %84 = vector.extract %72[%83] : f16 from vector<8xf16>
        %85 = vector.from_elements %73, %75, %76, %78, %79, %81, %82, %84 : vector<4x2xf16>
        %86 = frisk.pack_frag %85 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %87 = frisk.insert_frag %86, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %87 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %62:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %69 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %70 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %71 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %53) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %84 = frisk.extract_frag %49[%69, %arg16] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %85 = frisk.extract_frag %61[%arg16, %70] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %86 = frisk.gemm_frag %84, %85, acc = %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %86 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %72 = frisk.insert_frag %71, %arg12[%69, %70] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        %73 = frisk.extract_frag %43#1[%69, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %74 = frisk.broadcast_frag %73 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %75 = frisk.insert_frag %74, %arg13[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %76 = frisk.extract_frag %arg9[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %77 = frisk.extract_frag %75[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %78 = frisk.elementwise_frag "mul" %76, %77 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %79 = frisk.insert_frag %78, %arg14[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %80 = frisk.extract_frag %79[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %81 = frisk.extract_frag %72[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %82 = frisk.elementwise_frag "add" %80, %81 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %83 = frisk.insert_frag %82, %arg15[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %72, %75, %79, %83 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %63 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %64 = frisk.async_copy_gm_to_shm %63 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %64 : !frisk.async_task
      %65 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %69 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_12]
        %70 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_12]
        %71 = frisk.read_matrix_from_shm %alloc_15[%69, %70] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %72 = vector.bitcast %71 : vector<4xi32> to vector<8xf16>
        %73 = vector.extract %72[0] : f16 from vector<8xf16>
        %74 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_12]
        %75 = vector.extract %72[%74] : f16 from vector<8xf16>
        %76 = vector.extract %72[1] : f16 from vector<8xf16>
        %77 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_12]
        %78 = vector.extract %72[%77] : f16 from vector<8xf16>
        %79 = vector.extract %72[2] : f16 from vector<8xf16>
        %80 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_12]
        %81 = vector.extract %72[%80] : f16 from vector<8xf16>
        %82 = vector.extract %72[3] : f16 from vector<8xf16>
        %83 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_12]
        %84 = vector.extract %72[%83] : f16 from vector<8xf16>
        %85 = vector.from_elements %73, %75, %76, %78, %79, %81, %82, %84 : vector<4x2xf16>
        %86 = frisk.pack_frag %85 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %87 = frisk.insert_frag %86, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %87 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %66:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %69 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %70 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %71 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %53) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %86 = frisk.extract_frag %49[%69, %arg17] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %87 = frisk.extract_frag %65[%arg17, %70] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %88 = frisk.gemm_frag %86, %87, acc = %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %88 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %72 = frisk.insert_frag %71, %arg12[%69, %70] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        %73 = frisk.extract_frag %43#1[%69, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %74 = frisk.broadcast_frag %73 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %75 = frisk.insert_frag %74, %arg13[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %76 = frisk.extract_frag %arg10[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %77 = frisk.extract_frag %75[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %78 = frisk.elementwise_frag "mul" %76, %77 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %79 = frisk.insert_frag %78, %arg14[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %80 = frisk.extract_frag %79[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %81 = frisk.extract_frag %72[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %82 = frisk.elementwise_frag "add" %80, %81 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %83 = frisk.insert_frag %82, %arg15[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %84 = frisk.extract_frag %54#3[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %85 = frisk.insert_frag %84, %arg16[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %72, %75, %79, %83, %85 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %67:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %69 = frisk.extract_frag %48[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %70 = frisk.insert_frag %69, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        %71 = frisk.extract_frag %41[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %72 = frisk.insert_frag %71, %arg13[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %70, %72 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %68:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %69 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %70 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %71 = frisk.extract_frag %58#3[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %72 = frisk.insert_frag %71, %arg12[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %73 = frisk.extract_frag %62#3[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %74 = frisk.insert_frag %73, %arg13[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        %75 = frisk.extract_frag %66#3[%69, %70] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %76 = frisk.insert_frag %75, %arg14[%69, %70] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %72, %74, %76 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %66#4, %67#0, %67#1, %68#0, %68#1, %68#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {local_yield_transformed = true}
    %13 = frisk.pack_frag %12#5 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %14 = frisk.convert_layout_frag %13 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %15 = frisk.unpack_frag %14 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %16 = frisk.pack_frag %12#4 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %17 = frisk.convert_layout_frag %16 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %18 = frisk.unpack_frag %17 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %19 = frisk.pack_frag %12#3 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %20 = frisk.convert_layout_frag %19 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %21 = frisk.unpack_frag %20 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %22 = frisk.pack_frag %12#0 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %23 = frisk.convert_layout_frag %22 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %24 = frisk.unpack_frag %23 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %25 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %26:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %35 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %36 = frisk.extract_frag %12#1[%35, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %37 = frisk.broadcast_frag %36 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %38 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %39 = frisk.insert_frag %37, %arg5[%35, %38] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %40 = frisk.extract_frag %24[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %41 = frisk.extract_frag %39[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %42 = frisk.elementwise_frag "div" %40, %41 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %43 = frisk.insert_frag %42, %arg6[%35, %38] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %44 = frisk.extract_frag %43[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %45 = frisk.elementwise_frag "cast" %44 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %46 = frisk.insert_frag %45, %arg7[%35, %38] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      %47 = frisk.extract_frag %46[%35, %38] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %48 = frisk.insert_frag %47, %arg8[%35, %38] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %39, %43, %46, %48 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x = gpu.thread_id x
    %27 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %28 = arith.cmpi ult, %27, %c4 : index
    scf.if %28 {
      affine.for %arg4 = 0 to 4 {
        %35 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %36 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %37 = frisk.extract_frag %26#3[%35, %36] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %37, %25[%35, %36] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %29 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %30:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %35 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %36 = frisk.extract_frag %12#1[%35, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %37 = frisk.broadcast_frag %36 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %38 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %39 = frisk.insert_frag %37, %arg5[%35, %38] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %40 = frisk.extract_frag %21[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %41 = frisk.extract_frag %39[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %42 = frisk.elementwise_frag "div" %40, %41 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %43 = frisk.insert_frag %42, %arg6[%35, %38] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %44 = frisk.extract_frag %43[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %45 = frisk.elementwise_frag "cast" %44 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %46 = frisk.insert_frag %45, %arg7[%35, %38] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      %47 = frisk.extract_frag %46[%35, %38] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %48 = frisk.insert_frag %47, %arg8[%35, %38] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %39, %43, %46, %48 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %28 {
      affine.for %arg4 = 0 to 4 {
        %35 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %36 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %37 = frisk.extract_frag %30#3[%35, %36] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %37, %29[%35, %36] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %31 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %32:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %35 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %36 = frisk.extract_frag %12#1[%35, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %37 = frisk.broadcast_frag %36 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %38 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %39 = frisk.insert_frag %37, %arg5[%35, %38] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %40 = frisk.extract_frag %18[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %41 = frisk.extract_frag %39[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %42 = frisk.elementwise_frag "div" %40, %41 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %43 = frisk.insert_frag %42, %arg6[%35, %38] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %44 = frisk.extract_frag %43[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %45 = frisk.elementwise_frag "cast" %44 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %46 = frisk.insert_frag %45, %arg7[%35, %38] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      %47 = frisk.extract_frag %46[%35, %38] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %48 = frisk.insert_frag %47, %arg8[%35, %38] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %39, %43, %46, %48 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %28 {
      affine.for %arg4 = 0 to 4 {
        %35 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %36 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %37 = frisk.extract_frag %32#3[%35, %36] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %37, %31[%35, %36] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %33 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %34:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %35 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %36 = frisk.extract_frag %12#1[%35, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %37 = frisk.broadcast_frag %36 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %38 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %39 = frisk.insert_frag %37, %arg5[%35, %38] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %40 = frisk.extract_frag %15[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %41 = frisk.extract_frag %39[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %42 = frisk.elementwise_frag "div" %40, %41 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %43 = frisk.insert_frag %42, %arg6[%35, %38] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      %44 = frisk.extract_frag %43[%35, %38] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %45 = frisk.elementwise_frag "cast" %44 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %46 = frisk.insert_frag %45, %arg7[%35, %38] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      %47 = frisk.extract_frag %46[%35, %38] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %48 = frisk.insert_frag %47, %arg8[%35, %38] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %39, %43, %46, %48 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %28 {
      affine.for %arg4 = 0 to 4 {
        %35 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %36 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %37 = frisk.extract_frag %34#3[%35, %36] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %37, %33[%35, %36] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
