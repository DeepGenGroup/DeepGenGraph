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
    %9 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_11) -> (vector<2x1xf32>) {
      %70 = frisk.pack_frag %cst_10 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
      %71 = frisk.insert_frag %70, %arg5[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %71 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_11) -> (vector<2x1xf32>) {
      %70 = frisk.extract_frag %9[%arg4, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %71 = frisk.extract_frag %cst[%arg4, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %72 = frisk.elementwise_frag "add" %70, %71 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %73 = frisk.insert_frag %72, %arg5[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %73 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_11) -> (vector<2x1xf32>) {
      %70 = frisk.extract_frag %10[%arg4, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %71 = frisk.elementwise_frag "cast" %70 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %72 = frisk.insert_frag %71, %arg5[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %72 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %12 = frisk.to_threadTile %cst_9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %13 = frisk.to_threadTile %cst_8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %14 = frisk.to_threadTile %cst_9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %15 = frisk.to_threadTile %cst_9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %16 = frisk.to_threadTile %cst_9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %17:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %12, %arg6 = %13, %arg7 = %11, %arg8 = %14, %arg9 = %15, %arg10 = %16) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %70 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %71 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %70) -> (vector<2x32xf32>) {
        %alloc_28 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_29 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %119 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %119 to %alloc_28 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %120 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_28 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %120 to %alloc_29 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %121 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_6) -> (vector<2x8xf16>) {
          %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %128 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg13)
          %129 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%127)
          %130 = frisk.load_frag %alloc_28[%128, %129] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> <vector<1x4xf16>, {}>
          %131 = frisk.insert_frag %130, %arg14[%128, %129] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %131 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
        %122 = vector.broadcast %8 : f16 to vector<2x8xf16>
        %123 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_6) -> (vector<2x8xf16>) {
          %127 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg13)
          %128 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg13)
          %129 = frisk.extract_frag %121[%127, %128] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %130 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg13)
          %131 = frisk.extract_frag %122[%127, %130] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %132 = frisk.elementwise_frag "mul" %129, %131 : (!frisk.fragment<vector<1x4xf16>, {}>, !frisk.fragment<vector<1x4xf16>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
          %133 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg13)
          %134 = frisk.insert_frag %132, %arg14[%127, %133] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %134 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        frisk.wait_buffer_ready %alloc_29 : memref<32x128xf16, 3>
        %124 = frisk.pack_frag %cst_4 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %125 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5) -> (vector<2x32xf32>) {
          %127 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %128 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %129 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %124) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %139 = frisk.extract_frag %123[%127, %arg15] : vector<2x8xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %140 = frisk.load_frag %alloc_29[%arg15, %128] {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 3, 0>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tile_type = vector<8x8xf16>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %141 = frisk.gemm_frag %139, %140, acc = %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            affine.yield %141 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %130 = frisk.unpack_frag %129 : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> vector<1x4xf32>
          %131 = frisk.pack_frag %130 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
          %132 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %133 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %134 = frisk.extract_frag %arg12[%132, %133] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %135 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %136 = frisk.elementwise_frag "add" %134, %131 {frisk.fragment_accumulate} : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %137 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %138 = frisk.insert_frag %136, %arg14[%132, %137] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %138 : vector<2x32xf32>
        } {frisk.accumulator_fused, frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %126 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5) -> (vector<2x32xf32>) {
          %127 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %128 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %129 = frisk.extract_frag %125[%127, %128] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %130 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %131 = frisk.insert_frag %129, %arg14[%127, %130] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %131 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        affine.yield %126 : vector<2x32xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %72 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %121 = frisk.extract_frag %71[%119, %120] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %122 = frisk.elementwise_frag "cast" %121 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %123 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %124 = frisk.insert_frag %122, %arg12[%119, %123] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %124 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %73 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %119 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_12) -> (f32) {
          %125 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %126 = vector.extract %72[%arg11, %125] : f32 from vector<2x32xf32>
          %127 = arith.maxnumf %arg14, %126 : f32
          %128 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %129 = vector.extract %72[%arg11, %128] : f32 from vector<2x32xf32>
          %130 = arith.maxnumf %127, %129 : f32
          %131 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %132 = vector.extract %72[%arg11, %131] : f32 from vector<2x32xf32>
          %133 = arith.maxnumf %130, %132 : f32
          %134 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %135 = vector.extract %72[%arg11, %134] : f32 from vector<2x32xf32>
          %136 = arith.maxnumf %133, %135 : f32
          affine.yield %136 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %119, %c16_i32, %c64_i32 : f32
        %120 = arith.maxnumf %119, %shuffleResult : f32
        %shuffleResult_28, %valid_29 = gpu.shuffle xor %120, %c32_i32, %c64_i32 : f32
        %121 = arith.maxnumf %120, %shuffleResult_28 : f32
        %122 = vector.broadcast %121 : f32 to vector<1x1xf32>
        %123 = frisk.pack_frag %122 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %124 = frisk.insert_frag %123, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %124 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %74 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %119 = frisk.extract_frag %73[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %120 = frisk.extract_frag %arg7[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %121 = frisk.elementwise_frag "max" %119, %120 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %122 = frisk.insert_frag %121, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %122 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %75 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = vector.extract %74[%119, 0] : f32 from vector<2x1xf32>
        %121 = vector.extract %74[%119, 0] : f32 from vector<2x1xf32>
        %122 = vector.extract %74[%119, 0] : f32 from vector<2x1xf32>
        %123 = vector.extract %74[%119, 0] : f32 from vector<2x1xf32>
        %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %126 = frisk.pack_frag %124 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %127 = frisk.insert_frag %126, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %127 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %76 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %121 = frisk.extract_frag %71[%119, %120] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %123 = frisk.extract_frag %75[%119, %122] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %124 = frisk.elementwise_frag "sub" %121, %123 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %126 = frisk.insert_frag %124, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %126 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %77 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %119 = frisk.extract_frag %arg7[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %120 = frisk.extract_frag %74[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %121 = frisk.elementwise_frag "sub" %119, %120 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %122 = frisk.insert_frag %121, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %122 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %78 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %119 = frisk.extract_frag %77[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %120 = frisk.elementwise_frag "exp2" %119 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %121 = frisk.insert_frag %120, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %121 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %79 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %119 = frisk.extract_frag %arg6[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %120 = frisk.extract_frag %78[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %121 = frisk.elementwise_frag "mul" %119, %120 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %122 = frisk.insert_frag %121, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %122 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %80 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %121 = frisk.extract_frag %76[%119, %120] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %122 = frisk.elementwise_frag "exp2" %121 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %123 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %124 = frisk.insert_frag %122, %arg12[%119, %123] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %124 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %81 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %121 = frisk.extract_frag %80[%119, %120] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %122 = frisk.elementwise_frag "cast" %121 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %123 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %124 = frisk.insert_frag %122, %arg12[%119, %123] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %124 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %82 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %119 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_3) -> (f32) {
          %125 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %126 = vector.extract %81[%arg11, %125] : f32 from vector<2x32xf32>
          %127 = arith.addf %arg14, %126 : f32
          %128 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %129 = vector.extract %81[%arg11, %128] : f32 from vector<2x32xf32>
          %130 = arith.addf %127, %129 : f32
          %131 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %132 = vector.extract %81[%arg11, %131] : f32 from vector<2x32xf32>
          %133 = arith.addf %130, %132 : f32
          %134 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %135 = vector.extract %81[%arg11, %134] : f32 from vector<2x32xf32>
          %136 = arith.addf %133, %135 : f32
          affine.yield %136 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %119, %c16_i32, %c64_i32 : f32
        %120 = arith.addf %119, %shuffleResult : f32
        %shuffleResult_28, %valid_29 = gpu.shuffle xor %120, %c32_i32, %c64_i32 : f32
        %121 = arith.addf %120, %shuffleResult_28 : f32
        %122 = vector.broadcast %121 : f32 to vector<1x1xf32>
        %123 = frisk.pack_frag %122 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %124 = frisk.insert_frag %123, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %124 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %83 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %119 = frisk.extract_frag %82[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %120 = frisk.extract_frag %79[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %121 = frisk.elementwise_frag "add" %119, %120 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %122 = frisk.insert_frag %121, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %122 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %84 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %121 = frisk.extract_frag %80[%119, %120] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %122 = frisk.elementwise_frag "cast" %121 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %123 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %124 = frisk.insert_frag %122, %arg12[%119, %123] : <vector<1x4xf16>, {}> into vector<2x32xf16>
        affine.yield %124 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %85 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %85 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_21 = gpu.thread_id x
      %86 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_21]
        %120 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_21]
        %121 = frisk.read_matrix_from_shm %alloc[%119, %120] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %122 = vector.bitcast %121 : vector<4xi32> to vector<8xf16>
        %123 = vector.extract %122[0] : f16 from vector<8xf16>
        %124 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_21]
        %125 = vector.extract %122[%124] : f16 from vector<8xf16>
        %126 = vector.extract %122[1] : f16 from vector<8xf16>
        %127 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_21]
        %128 = vector.extract %122[%127] : f16 from vector<8xf16>
        %129 = vector.extract %122[2] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_21]
        %131 = vector.extract %122[%130] : f16 from vector<8xf16>
        %132 = vector.extract %122[3] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_21]
        %134 = vector.extract %122[%133] : f16 from vector<8xf16>
        %135 = vector.from_elements %123, %125, %126, %128, %129, %131, %132, %134 : vector<4x2xf16>
        %136 = frisk.pack_frag %135 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %137 = frisk.insert_frag %136, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %137 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %87 = frisk.pack_frag %cst_4 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %88 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %87) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %123 = frisk.extract_frag %84[%119, %arg13] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %124 = frisk.extract_frag %86[%arg13, %120] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %125 = frisk.gemm_frag %123, %124, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %125 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %122 = frisk.insert_frag %121, %arg12[%119, %120] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %122 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %89 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %121 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %122 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %123 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.pack_frag %124 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %127 = frisk.insert_frag %126, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %90 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %arg5[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.extract_frag %89[%119, %122] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %124 = frisk.elementwise_frag "mul" %121, %123 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.insert_frag %124, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %126 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %91 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %90[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.extract_frag %88[%119, %122] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %124 = frisk.elementwise_frag "add" %121, %123 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.insert_frag %124, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %126 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_22 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %92 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %92 to %alloc_22 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_22 : memref<128x32xf16, 3>
      %thread_id_x_23 = gpu.thread_id x
      %93 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_23]
        %120 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_23]
        %121 = frisk.read_matrix_from_shm %alloc_22[%119, %120] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %122 = vector.bitcast %121 : vector<4xi32> to vector<8xf16>
        %123 = vector.extract %122[0] : f16 from vector<8xf16>
        %124 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_23]
        %125 = vector.extract %122[%124] : f16 from vector<8xf16>
        %126 = vector.extract %122[1] : f16 from vector<8xf16>
        %127 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_23]
        %128 = vector.extract %122[%127] : f16 from vector<8xf16>
        %129 = vector.extract %122[2] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_23]
        %131 = vector.extract %122[%130] : f16 from vector<8xf16>
        %132 = vector.extract %122[3] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_23]
        %134 = vector.extract %122[%133] : f16 from vector<8xf16>
        %135 = vector.from_elements %123, %125, %126, %128, %129, %131, %132, %134 : vector<4x2xf16>
        %136 = frisk.pack_frag %135 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %137 = frisk.insert_frag %136, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %137 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %94 = frisk.pack_frag %cst_4 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %95 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %94) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %123 = frisk.extract_frag %84[%119, %arg13] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %124 = frisk.extract_frag %93[%arg13, %120] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %125 = frisk.gemm_frag %123, %124, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %125 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %122 = frisk.insert_frag %121, %arg12[%119, %120] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %122 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %96 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %121 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %122 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %123 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.pack_frag %124 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %127 = frisk.insert_frag %126, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %97 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %arg8[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.extract_frag %96[%119, %122] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %124 = frisk.elementwise_frag "mul" %121, %123 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.insert_frag %124, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %126 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %98 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %97[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.extract_frag %95[%119, %122] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %124 = frisk.elementwise_frag "add" %121, %123 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.insert_frag %124, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %126 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_24 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %99 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %99 to %alloc_24 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_24 : memref<128x32xf16, 3>
      %thread_id_x_25 = gpu.thread_id x
      %100 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_25]
        %120 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_25]
        %121 = frisk.read_matrix_from_shm %alloc_24[%119, %120] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %122 = vector.bitcast %121 : vector<4xi32> to vector<8xf16>
        %123 = vector.extract %122[0] : f16 from vector<8xf16>
        %124 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_25]
        %125 = vector.extract %122[%124] : f16 from vector<8xf16>
        %126 = vector.extract %122[1] : f16 from vector<8xf16>
        %127 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_25]
        %128 = vector.extract %122[%127] : f16 from vector<8xf16>
        %129 = vector.extract %122[2] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_25]
        %131 = vector.extract %122[%130] : f16 from vector<8xf16>
        %132 = vector.extract %122[3] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_25]
        %134 = vector.extract %122[%133] : f16 from vector<8xf16>
        %135 = vector.from_elements %123, %125, %126, %128, %129, %131, %132, %134 : vector<4x2xf16>
        %136 = frisk.pack_frag %135 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %137 = frisk.insert_frag %136, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %137 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %101 = frisk.pack_frag %cst_4 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %102 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %101) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %123 = frisk.extract_frag %84[%119, %arg13] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %124 = frisk.extract_frag %100[%arg13, %120] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %125 = frisk.gemm_frag %123, %124, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %125 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %122 = frisk.insert_frag %121, %arg12[%119, %120] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %122 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %103 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %121 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %122 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %123 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.pack_frag %124 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %127 = frisk.insert_frag %126, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %104 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %arg9[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.extract_frag %103[%119, %122] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %124 = frisk.elementwise_frag "mul" %121, %123 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.insert_frag %124, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %126 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %105 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %104[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.extract_frag %102[%119, %122] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %124 = frisk.elementwise_frag "add" %121, %123 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.insert_frag %124, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %126 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_26 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %106 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %106 to %alloc_26 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_26 : memref<128x32xf16, 3>
      %thread_id_x_27 = gpu.thread_id x
      %107 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_27]
        %120 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_27]
        %121 = frisk.read_matrix_from_shm %alloc_26[%119, %120] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %122 = vector.bitcast %121 : vector<4xi32> to vector<8xf16>
        %123 = vector.extract %122[0] : f16 from vector<8xf16>
        %124 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_27]
        %125 = vector.extract %122[%124] : f16 from vector<8xf16>
        %126 = vector.extract %122[1] : f16 from vector<8xf16>
        %127 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_27]
        %128 = vector.extract %122[%127] : f16 from vector<8xf16>
        %129 = vector.extract %122[2] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_27]
        %131 = vector.extract %122[%130] : f16 from vector<8xf16>
        %132 = vector.extract %122[3] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_27]
        %134 = vector.extract %122[%133] : f16 from vector<8xf16>
        %135 = vector.from_elements %123, %125, %126, %128, %129, %131, %132, %134 : vector<4x2xf16>
        %136 = frisk.pack_frag %135 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %137 = frisk.insert_frag %136, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %137 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %108 = frisk.pack_frag %cst_4 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %109 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %108) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %123 = frisk.extract_frag %84[%119, %arg13] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %124 = frisk.extract_frag %107[%arg13, %120] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %125 = frisk.gemm_frag %123, %124, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %125 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %122 = frisk.insert_frag %121, %arg12[%119, %120] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %122 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %110 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %121 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %122 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %123 = vector.extract %78[%119, 0] : f32 from vector<2x1xf32>
        %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.pack_frag %124 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %127 = frisk.insert_frag %126, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %111 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %arg10[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.extract_frag %110[%119, %122] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %124 = frisk.elementwise_frag "mul" %121, %123 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.insert_frag %124, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %126 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %112 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %111[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.extract_frag %109[%119, %122] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %124 = frisk.elementwise_frag "add" %121, %123 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %125 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %126 = frisk.insert_frag %124, %arg12[%119, %125] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %126 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %113 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %91[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.insert_frag %121, %arg12[%119, %122] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %123 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %114 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %119 = frisk.extract_frag %83[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %120 = frisk.insert_frag %119, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %120 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %115 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %119 = frisk.extract_frag %74[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %120 = frisk.insert_frag %119, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %120 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %116 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %98[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.insert_frag %121, %arg12[%119, %122] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %123 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %117 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %105[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.insert_frag %121, %arg12[%119, %122] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %123 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %118 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %121 = frisk.extract_frag %112[%119, %120] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %123 = frisk.insert_frag %121, %arg12[%119, %122] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %123 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %113, %114, %115, %116, %117, %118 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {local_yield_transformed = true}
    %thread_id_x = gpu.thread_id x
    %18 = arith.index_cast %thread_id_x : index to i32
    %19 = arith.andi %18, %c16_i32 : i32
    %20 = arith.cmpi ne, %19, %c0_i32 : i32
    %21 = arith.andi %18, %c32_i32 : i32
    %22 = arith.cmpi ne, %21, %c0_i32 : i32
    %23 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %17#5[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = frisk.unpack_frag %72 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
      %74 = vector.extract %73[0, 0] : f32 from vector<1x4xf32>
      %75 = vector.extract %73[0, 1] : f32 from vector<1x4xf32>
      %76 = vector.extract %73[0, 2] : f32 from vector<1x4xf32>
      %77 = vector.extract %73[0, 3] : f32 from vector<1x4xf32>
      %78 = arith.select %20, %74, %75 : f32
      %shuffleResult, %valid = gpu.shuffle xor %78, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %79 = arith.select %20, %shuffleResult, %74 : f32
      %80 = arith.select %20, %75, %shuffleResult : f32
      %81 = arith.select %20, %76, %77 : f32
      %shuffleResult_21, %valid_22 = gpu.shuffle xor %81, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %82 = arith.select %20, %shuffleResult_21, %76 : f32
      %83 = arith.select %20, %77, %shuffleResult_21 : f32
      %84 = arith.select %22, %79, %82 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %84, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %85 = arith.select %22, %shuffleResult_23, %79 : f32
      %86 = arith.select %22, %82, %shuffleResult_23 : f32
      %87 = arith.select %22, %80, %83 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %87, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %88 = arith.select %22, %shuffleResult_25, %80 : f32
      %89 = arith.select %22, %83, %shuffleResult_25 : f32
      %90 = vector.from_elements %85, %88, %86, %89 : vector<1x4xf32>
      %91 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %92 = frisk.pack_frag %90 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %93 = frisk.insert_frag %92, %arg5[%70, %91] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %93 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_14 = gpu.thread_id x
    %24 = arith.index_cast %thread_id_x_14 : index to i32
    %25 = arith.andi %24, %c16_i32 : i32
    %26 = arith.cmpi ne, %25, %c0_i32 : i32
    %27 = arith.andi %24, %c32_i32 : i32
    %28 = arith.cmpi ne, %27, %c0_i32 : i32
    %29 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %17#4[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = frisk.unpack_frag %72 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
      %74 = vector.extract %73[0, 0] : f32 from vector<1x4xf32>
      %75 = vector.extract %73[0, 1] : f32 from vector<1x4xf32>
      %76 = vector.extract %73[0, 2] : f32 from vector<1x4xf32>
      %77 = vector.extract %73[0, 3] : f32 from vector<1x4xf32>
      %78 = arith.select %26, %74, %75 : f32
      %shuffleResult, %valid = gpu.shuffle xor %78, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %79 = arith.select %26, %shuffleResult, %74 : f32
      %80 = arith.select %26, %75, %shuffleResult : f32
      %81 = arith.select %26, %76, %77 : f32
      %shuffleResult_21, %valid_22 = gpu.shuffle xor %81, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %82 = arith.select %26, %shuffleResult_21, %76 : f32
      %83 = arith.select %26, %77, %shuffleResult_21 : f32
      %84 = arith.select %28, %79, %82 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %84, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %85 = arith.select %28, %shuffleResult_23, %79 : f32
      %86 = arith.select %28, %82, %shuffleResult_23 : f32
      %87 = arith.select %28, %80, %83 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %87, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %88 = arith.select %28, %shuffleResult_25, %80 : f32
      %89 = arith.select %28, %83, %shuffleResult_25 : f32
      %90 = vector.from_elements %85, %88, %86, %89 : vector<1x4xf32>
      %91 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %92 = frisk.pack_frag %90 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %93 = frisk.insert_frag %92, %arg5[%70, %91] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %93 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_15 = gpu.thread_id x
    %30 = arith.index_cast %thread_id_x_15 : index to i32
    %31 = arith.andi %30, %c16_i32 : i32
    %32 = arith.cmpi ne, %31, %c0_i32 : i32
    %33 = arith.andi %30, %c32_i32 : i32
    %34 = arith.cmpi ne, %33, %c0_i32 : i32
    %35 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %17#3[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = frisk.unpack_frag %72 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
      %74 = vector.extract %73[0, 0] : f32 from vector<1x4xf32>
      %75 = vector.extract %73[0, 1] : f32 from vector<1x4xf32>
      %76 = vector.extract %73[0, 2] : f32 from vector<1x4xf32>
      %77 = vector.extract %73[0, 3] : f32 from vector<1x4xf32>
      %78 = arith.select %32, %74, %75 : f32
      %shuffleResult, %valid = gpu.shuffle xor %78, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %79 = arith.select %32, %shuffleResult, %74 : f32
      %80 = arith.select %32, %75, %shuffleResult : f32
      %81 = arith.select %32, %76, %77 : f32
      %shuffleResult_21, %valid_22 = gpu.shuffle xor %81, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %82 = arith.select %32, %shuffleResult_21, %76 : f32
      %83 = arith.select %32, %77, %shuffleResult_21 : f32
      %84 = arith.select %34, %79, %82 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %84, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %85 = arith.select %34, %shuffleResult_23, %79 : f32
      %86 = arith.select %34, %82, %shuffleResult_23 : f32
      %87 = arith.select %34, %80, %83 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %87, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %88 = arith.select %34, %shuffleResult_25, %80 : f32
      %89 = arith.select %34, %83, %shuffleResult_25 : f32
      %90 = vector.from_elements %85, %88, %86, %89 : vector<1x4xf32>
      %91 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %92 = frisk.pack_frag %90 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %93 = frisk.insert_frag %92, %arg5[%70, %91] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %93 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_16 = gpu.thread_id x
    %36 = arith.index_cast %thread_id_x_16 : index to i32
    %37 = arith.andi %36, %c16_i32 : i32
    %38 = arith.cmpi ne, %37, %c0_i32 : i32
    %39 = arith.andi %36, %c32_i32 : i32
    %40 = arith.cmpi ne, %39, %c0_i32 : i32
    %41 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %17#0[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = frisk.unpack_frag %72 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
      %74 = vector.extract %73[0, 0] : f32 from vector<1x4xf32>
      %75 = vector.extract %73[0, 1] : f32 from vector<1x4xf32>
      %76 = vector.extract %73[0, 2] : f32 from vector<1x4xf32>
      %77 = vector.extract %73[0, 3] : f32 from vector<1x4xf32>
      %78 = arith.select %38, %74, %75 : f32
      %shuffleResult, %valid = gpu.shuffle xor %78, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %79 = arith.select %38, %shuffleResult, %74 : f32
      %80 = arith.select %38, %75, %shuffleResult : f32
      %81 = arith.select %38, %76, %77 : f32
      %shuffleResult_21, %valid_22 = gpu.shuffle xor %81, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %82 = arith.select %38, %shuffleResult_21, %76 : f32
      %83 = arith.select %38, %77, %shuffleResult_21 : f32
      %84 = arith.select %40, %79, %82 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %84, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %85 = arith.select %40, %shuffleResult_23, %79 : f32
      %86 = arith.select %40, %82, %shuffleResult_23 : f32
      %87 = arith.select %40, %80, %83 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %87, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %88 = arith.select %40, %shuffleResult_25, %80 : f32
      %89 = arith.select %40, %83, %shuffleResult_25 : f32
      %90 = vector.from_elements %85, %88, %86, %89 : vector<1x4xf32>
      %91 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %92 = frisk.pack_frag %90 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %93 = frisk.insert_frag %92, %arg5[%70, %91] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %93 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %42 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %72 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %73 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %74 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
      %76 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %77 = frisk.pack_frag %75 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %78 = frisk.insert_frag %77, %arg5[%70, %76] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %78 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %43 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %41[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %74 = frisk.extract_frag %42[%70, %73] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %75 = frisk.elementwise_frag "div" %72, %74 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %76 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %77 = frisk.insert_frag %75, %arg5[%70, %76] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %44 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %43[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = frisk.elementwise_frag "cast" %72 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %74 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %75 = frisk.insert_frag %73, %arg5[%70, %74] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %75 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %45 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %46 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %44[%70, %71] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %73 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %74 = frisk.insert_frag %72, %arg5[%70, %73] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %74 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_17 = gpu.thread_id x
    %47 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x_17]
    %48 = arith.cmpi ult, %47, %c4 : index
    scf.if %48 {
      affine.for %arg4 = 0 to 4 {
        %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %72 = frisk.extract_frag %46[%70, %71] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %72, %45[%70, %71] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %49 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %72 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %73 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %74 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
      %76 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %77 = frisk.pack_frag %75 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %78 = frisk.insert_frag %77, %arg5[%70, %76] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %78 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %50 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %35[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %74 = frisk.extract_frag %49[%70, %73] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %75 = frisk.elementwise_frag "div" %72, %74 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %76 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %77 = frisk.insert_frag %75, %arg5[%70, %76] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %51 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %50[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = frisk.elementwise_frag "cast" %72 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %74 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %75 = frisk.insert_frag %73, %arg5[%70, %74] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %75 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %52 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %53 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %51[%70, %71] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %73 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %74 = frisk.insert_frag %72, %arg5[%70, %73] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %74 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_18 = gpu.thread_id x
    %54 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x_18]
    %55 = arith.cmpi ult, %54, %c4 : index
    scf.if %55 {
      affine.for %arg4 = 0 to 4 {
        %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %72 = frisk.extract_frag %53[%70, %71] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %72, %52[%70, %71] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %56 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %72 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %73 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %74 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
      %76 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %77 = frisk.pack_frag %75 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %78 = frisk.insert_frag %77, %arg5[%70, %76] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %78 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %57 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %29[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %74 = frisk.extract_frag %56[%70, %73] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %75 = frisk.elementwise_frag "div" %72, %74 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %76 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %77 = frisk.insert_frag %75, %arg5[%70, %76] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %58 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %57[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = frisk.elementwise_frag "cast" %72 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %74 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %75 = frisk.insert_frag %73, %arg5[%70, %74] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %75 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %59 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %60 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %58[%70, %71] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %73 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %74 = frisk.insert_frag %72, %arg5[%70, %73] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %74 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_19 = gpu.thread_id x
    %61 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x_19]
    %62 = arith.cmpi ult, %61, %c4 : index
    scf.if %62 {
      affine.for %arg4 = 0 to 4 {
        %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %72 = frisk.extract_frag %60[%70, %71] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %72, %59[%70, %71] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %63 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %72 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %73 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %74 = vector.extract %17#1[%70, 0] : f32 from vector<2x1xf32>
      %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
      %76 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %77 = frisk.pack_frag %75 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %78 = frisk.insert_frag %77, %arg5[%70, %76] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %78 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %64 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %23[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %74 = frisk.extract_frag %63[%70, %73] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %75 = frisk.elementwise_frag "div" %72, %74 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %76 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %77 = frisk.insert_frag %75, %arg5[%70, %76] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %65 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %64[%70, %71] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %73 = frisk.elementwise_frag "cast" %72 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %74 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %75 = frisk.insert_frag %73, %arg5[%70, %74] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %75 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %66 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %67 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %72 = frisk.extract_frag %65[%70, %71] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %73 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %74 = frisk.insert_frag %72, %arg5[%70, %73] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %74 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_20 = gpu.thread_id x
    %68 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x_20]
    %69 = arith.cmpi ult, %68, %c4 : index
    scf.if %69 {
      affine.for %arg4 = 0 to 4 {
        %70 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %71 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %72 = frisk.extract_frag %67[%70, %71] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %72, %66[%70, %71] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
