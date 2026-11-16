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
    %9 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_9) -> (vector<2x1xf32>) {
      %59 = frisk.pack_frag %cst_8 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
      %60 = frisk.insert_frag %59, %arg5[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %60 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_9) -> (vector<2x1xf32>) {
      %59 = frisk.broadcast_frag %cst_10 : f32 -> <vector<1x1xf32>, {}>
      %60 = frisk.insert_frag %59, %arg5[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %60 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_9) -> (vector<2x1xf32>) {
      %59 = frisk.extract_frag %9[%arg4, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %60 = frisk.extract_frag %10[%arg4, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %61 = frisk.elementwise_frag "add" %59, %60 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %62 = frisk.insert_frag %61, %arg5[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %62 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %12 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_9) -> (vector<2x1xf32>) {
      %59 = frisk.extract_frag %11[%arg4, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %60 = frisk.elementwise_frag "cast" %59 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %61 = frisk.insert_frag %60, %arg5[%arg4, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %61 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %13 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %14 = frisk.to_threadTile %cst_6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %15 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %16 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %17 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %18:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %13, %arg6 = %14, %arg7 = %12, %arg8 = %15, %arg9 = %16, %arg10 = %17) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %59 = frisk.to_threadTile %cst_5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %60 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %59) -> (vector<2x32xf32>) {
        %alloc_22 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %112 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %112 to %alloc_22 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %113 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_22 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %113 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %114 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_4) -> (vector<2x8xf16>) {
          %120 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %121 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg13)
          %122 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%120)
          %123 = frisk.load_frag %alloc_22[%121, %122] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> <vector<1x4xf16>, {}>
          %124 = frisk.insert_frag %123, %arg14[%121, %122] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %124 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
        %115 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_4) -> (vector<2x8xf16>) {
          %120 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg13)
          %121 = frisk.broadcast_frag %8 : f16 -> <vector<1x4xf16>, {}>
          %122 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg13)
          %123 = frisk.insert_frag %121, %arg14[%120, %122] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %123 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        %116 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_4) -> (vector<2x8xf16>) {
          %120 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg13)
          %121 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg13)
          %122 = frisk.extract_frag %114[%120, %121] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %123 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg13)
          %124 = frisk.extract_frag %115[%120, %123] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %125 = frisk.elementwise_frag "mul" %122, %124 : (!frisk.fragment<vector<1x4xf16>, {}>, !frisk.fragment<vector<1x4xf16>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
          %126 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg13)
          %127 = frisk.insert_frag %125, %arg14[%120, %126] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %127 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        frisk.wait_buffer_ready %alloc_23 : memref<32x128xf16, 3>
        %117 = frisk.pack_frag %cst_2 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %118 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_3) -> (vector<2x32xf32>) {
          %120 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %121 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %122 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %117) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %132 = frisk.extract_frag %116[%120, %arg15] : vector<2x8xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %133 = frisk.load_frag %alloc_23[%arg15, %121] {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 3, 0>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tile_type = vector<8x8xf16>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %134 = frisk.gemm_frag %132, %133, acc = %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            affine.yield %134 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %123 = frisk.unpack_frag %122 : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> vector<1x4xf32>
          %124 = frisk.pack_frag %123 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
          %125 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %126 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %127 = frisk.extract_frag %arg12[%125, %126] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %128 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %129 = frisk.elementwise_frag "add" %127, %124 {frisk.fragment_accumulate} : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %130 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %131 = frisk.insert_frag %129, %arg14[%125, %130] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %131 : vector<2x32xf32>
        } {frisk.accumulator_fused, frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %119 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_3) -> (vector<2x32xf32>) {
          %120 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %121 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %122 = frisk.extract_frag %118[%120, %121] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %123 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
          %124 = frisk.insert_frag %122, %arg14[%120, %123] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %124 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        affine.yield %119 : vector<2x32xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %61 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %114 = frisk.extract_frag %60[%112, %113] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %115 = frisk.elementwise_frag "cast" %114 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %117 = frisk.insert_frag %115, %arg12[%112, %116] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %117 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %62 = frisk.pack_frag %61 : vector<2x32xf32> -> <vector<2x32xf32>, {}>
      %63 = frisk.reduce_frag "max" %62 {dim = 1 : i64, lane_extent = 4 : i64, lane_stride = 16 : i64, output_fragment_shape = array<i64: 1, 1>, reduction_width = 4 : i64, warp_threads = 64 : i64} : <vector<2x32xf32>, {}> -> <vector<2x1xf32>, {}>
      %64 = frisk.unpack_frag %63 : <vector<2x1xf32>, {}> -> vector<2x1xf32>
      %65 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %112 = frisk.extract_frag %64[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %113 = frisk.extract_frag %arg7[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.elementwise_frag "max" %112, %113 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %115 = frisk.insert_frag %114, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %115 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %66 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %113 = frisk.extract_frag %65[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.broadcast_frag %113 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %116 = frisk.insert_frag %114, %arg12[%112, %115] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %116 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %67 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %114 = frisk.extract_frag %60[%112, %113] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %116 = frisk.extract_frag %66[%112, %115] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %117 = frisk.elementwise_frag "sub" %114, %116 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %118 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %119 = frisk.insert_frag %117, %arg12[%112, %118] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %119 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %68 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %112 = frisk.extract_frag %arg7[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %113 = frisk.extract_frag %65[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.elementwise_frag "sub" %112, %113 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %115 = frisk.insert_frag %114, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %115 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %69 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %112 = frisk.extract_frag %68[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %113 = frisk.elementwise_frag "exp2" %112 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %114 = frisk.insert_frag %113, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %114 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %70 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %112 = frisk.extract_frag %arg6[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %113 = frisk.extract_frag %69[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.elementwise_frag "mul" %112, %113 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %115 = frisk.insert_frag %114, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %115 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %71 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %114 = frisk.extract_frag %67[%112, %113] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %115 = frisk.elementwise_frag "exp2" %114 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %117 = frisk.insert_frag %115, %arg12[%112, %116] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %117 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %72 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %114 = frisk.extract_frag %71[%112, %113] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %115 = frisk.elementwise_frag "cast" %114 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %117 = frisk.insert_frag %115, %arg12[%112, %116] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %117 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %73 = frisk.pack_frag %72 : vector<2x32xf32> -> <vector<2x32xf32>, {}>
      %74 = frisk.reduce_frag "add" %73 {dim = 1 : i64, lane_extent = 4 : i64, lane_stride = 16 : i64, output_fragment_shape = array<i64: 1, 1>, reduction_width = 4 : i64, warp_threads = 64 : i64} : <vector<2x32xf32>, {}> -> <vector<2x1xf32>, {}>
      %75 = frisk.unpack_frag %74 : <vector<2x1xf32>, {}> -> vector<2x1xf32>
      %76 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %112 = frisk.extract_frag %75[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %113 = frisk.extract_frag %70[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.elementwise_frag "add" %112, %113 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %115 = frisk.insert_frag %114, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %115 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %77 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_1) -> (vector<2x32xf16>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %114 = frisk.extract_frag %71[%112, %113] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %115 = frisk.elementwise_frag "cast" %114 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %117 = frisk.insert_frag %115, %arg12[%112, %116] : <vector<1x4xf16>, {}> into vector<2x32xf16>
        affine.yield %117 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %78 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %78 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_15 = gpu.thread_id x
      %79 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_15]
        %113 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_15]
        %114 = frisk.read_matrix_from_shm %alloc[%112, %113] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %115 = vector.bitcast %114 : vector<4xi32> to vector<8xf16>
        %116 = vector.extract %115[0] : f16 from vector<8xf16>
        %117 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_15]
        %118 = vector.extract %115[%117] : f16 from vector<8xf16>
        %119 = vector.extract %115[1] : f16 from vector<8xf16>
        %120 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_15]
        %121 = vector.extract %115[%120] : f16 from vector<8xf16>
        %122 = vector.extract %115[2] : f16 from vector<8xf16>
        %123 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_15]
        %124 = vector.extract %115[%123] : f16 from vector<8xf16>
        %125 = vector.extract %115[3] : f16 from vector<8xf16>
        %126 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_15]
        %127 = vector.extract %115[%126] : f16 from vector<8xf16>
        %128 = vector.from_elements %116, %118, %119, %121, %122, %124, %125, %127 : vector<4x2xf16>
        %129 = frisk.pack_frag %128 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %130 = frisk.insert_frag %129, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %130 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %80 = frisk.pack_frag %cst_2 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %81 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %80) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %116 = frisk.extract_frag %77[%112, %arg13] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %117 = frisk.extract_frag %79[%arg13, %113] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %118 = frisk.gemm_frag %116, %117, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %118 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %115 = frisk.insert_frag %114, %arg12[%112, %113] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %115 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %82 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = frisk.extract_frag %69[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.broadcast_frag %113 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.insert_frag %114, %arg12[%112, %115] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %83 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %arg5[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.extract_frag %82[%112, %115] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %117 = frisk.elementwise_frag "mul" %114, %116 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %118 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %119 = frisk.insert_frag %117, %arg12[%112, %118] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %119 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %84 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %83[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.extract_frag %81[%112, %115] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %117 = frisk.elementwise_frag "add" %114, %116 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %118 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %119 = frisk.insert_frag %117, %arg12[%112, %118] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %119 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %85 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %85 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_16 : memref<128x32xf16, 3>
      %thread_id_x_17 = gpu.thread_id x
      %86 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_17]
        %113 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_17]
        %114 = frisk.read_matrix_from_shm %alloc_16[%112, %113] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %115 = vector.bitcast %114 : vector<4xi32> to vector<8xf16>
        %116 = vector.extract %115[0] : f16 from vector<8xf16>
        %117 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_17]
        %118 = vector.extract %115[%117] : f16 from vector<8xf16>
        %119 = vector.extract %115[1] : f16 from vector<8xf16>
        %120 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_17]
        %121 = vector.extract %115[%120] : f16 from vector<8xf16>
        %122 = vector.extract %115[2] : f16 from vector<8xf16>
        %123 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_17]
        %124 = vector.extract %115[%123] : f16 from vector<8xf16>
        %125 = vector.extract %115[3] : f16 from vector<8xf16>
        %126 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_17]
        %127 = vector.extract %115[%126] : f16 from vector<8xf16>
        %128 = vector.from_elements %116, %118, %119, %121, %122, %124, %125, %127 : vector<4x2xf16>
        %129 = frisk.pack_frag %128 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %130 = frisk.insert_frag %129, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %130 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %87 = frisk.pack_frag %cst_2 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %88 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %87) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %116 = frisk.extract_frag %77[%112, %arg13] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %117 = frisk.extract_frag %86[%arg13, %113] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %118 = frisk.gemm_frag %116, %117, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %118 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %115 = frisk.insert_frag %114, %arg12[%112, %113] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %115 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %89 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = frisk.extract_frag %69[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.broadcast_frag %113 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.insert_frag %114, %arg12[%112, %115] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %90 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %arg8[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.extract_frag %89[%112, %115] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %117 = frisk.elementwise_frag "mul" %114, %116 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %118 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %119 = frisk.insert_frag %117, %arg12[%112, %118] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %119 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %91 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %90[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.extract_frag %88[%112, %115] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %117 = frisk.elementwise_frag "add" %114, %116 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %118 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %119 = frisk.insert_frag %117, %arg12[%112, %118] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %119 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %92 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %92 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_18 : memref<128x32xf16, 3>
      %thread_id_x_19 = gpu.thread_id x
      %93 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_19]
        %113 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_19]
        %114 = frisk.read_matrix_from_shm %alloc_18[%112, %113] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %115 = vector.bitcast %114 : vector<4xi32> to vector<8xf16>
        %116 = vector.extract %115[0] : f16 from vector<8xf16>
        %117 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_19]
        %118 = vector.extract %115[%117] : f16 from vector<8xf16>
        %119 = vector.extract %115[1] : f16 from vector<8xf16>
        %120 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_19]
        %121 = vector.extract %115[%120] : f16 from vector<8xf16>
        %122 = vector.extract %115[2] : f16 from vector<8xf16>
        %123 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_19]
        %124 = vector.extract %115[%123] : f16 from vector<8xf16>
        %125 = vector.extract %115[3] : f16 from vector<8xf16>
        %126 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_19]
        %127 = vector.extract %115[%126] : f16 from vector<8xf16>
        %128 = vector.from_elements %116, %118, %119, %121, %122, %124, %125, %127 : vector<4x2xf16>
        %129 = frisk.pack_frag %128 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %130 = frisk.insert_frag %129, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %130 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %94 = frisk.pack_frag %cst_2 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %95 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %94) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %116 = frisk.extract_frag %77[%112, %arg13] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %117 = frisk.extract_frag %93[%arg13, %113] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %118 = frisk.gemm_frag %116, %117, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %118 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %115 = frisk.insert_frag %114, %arg12[%112, %113] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %115 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %96 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = frisk.extract_frag %69[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.broadcast_frag %113 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.insert_frag %114, %arg12[%112, %115] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %97 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %arg9[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.extract_frag %96[%112, %115] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %117 = frisk.elementwise_frag "mul" %114, %116 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %118 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %119 = frisk.insert_frag %117, %arg12[%112, %118] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %119 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %98 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %97[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.extract_frag %95[%112, %115] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %117 = frisk.elementwise_frag "add" %114, %116 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %118 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %119 = frisk.insert_frag %117, %arg12[%112, %118] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %119 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_20 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %99 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %99 to %alloc_20 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_20 : memref<128x32xf16, 3>
      %thread_id_x_21 = gpu.thread_id x
      %100 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_21]
        %113 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_21]
        %114 = frisk.read_matrix_from_shm %alloc_20[%112, %113] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %115 = vector.bitcast %114 : vector<4xi32> to vector<8xf16>
        %116 = vector.extract %115[0] : f16 from vector<8xf16>
        %117 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_21]
        %118 = vector.extract %115[%117] : f16 from vector<8xf16>
        %119 = vector.extract %115[1] : f16 from vector<8xf16>
        %120 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_21]
        %121 = vector.extract %115[%120] : f16 from vector<8xf16>
        %122 = vector.extract %115[2] : f16 from vector<8xf16>
        %123 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_21]
        %124 = vector.extract %115[%123] : f16 from vector<8xf16>
        %125 = vector.extract %115[3] : f16 from vector<8xf16>
        %126 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_21]
        %127 = vector.extract %115[%126] : f16 from vector<8xf16>
        %128 = vector.from_elements %116, %118, %119, %121, %122, %124, %125, %127 : vector<4x2xf16>
        %129 = frisk.pack_frag %128 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %130 = frisk.insert_frag %129, %arg12[%arg11, %c0] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %130 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %101 = frisk.pack_frag %cst_2 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %102 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %101) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %116 = frisk.extract_frag %77[%112, %arg13] : vector<2x32xf16> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %117 = frisk.extract_frag %100[%arg13, %113] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %118 = frisk.gemm_frag %116, %117, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %118 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %115 = frisk.insert_frag %114, %arg12[%112, %113] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %115 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %103 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = frisk.extract_frag %69[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.broadcast_frag %113 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.insert_frag %114, %arg12[%112, %115] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %104 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %arg10[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.extract_frag %103[%112, %115] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %117 = frisk.elementwise_frag "mul" %114, %116 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %118 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %119 = frisk.insert_frag %117, %arg12[%112, %118] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %119 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %105 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %104[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.extract_frag %102[%112, %115] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %117 = frisk.elementwise_frag "add" %114, %116 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %118 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %119 = frisk.insert_frag %117, %arg12[%112, %118] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %119 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %106 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %84[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.insert_frag %114, %arg12[%112, %115] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %107 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %112 = frisk.extract_frag %76[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %113 = frisk.insert_frag %112, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %113 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %108 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %112 = frisk.extract_frag %65[%arg11, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %113 = frisk.insert_frag %112, %arg12[%arg11, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %113 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %109 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %91[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.insert_frag %114, %arg12[%112, %115] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %110 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %98[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.insert_frag %114, %arg12[%112, %115] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %111 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %114 = frisk.extract_frag %105[%112, %113] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %115 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %116 = frisk.insert_frag %114, %arg12[%112, %115] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %106, %107, %108, %109, %110, %111 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {local_yield_transformed = true}
    %19 = frisk.pack_frag %18#5 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %20 = frisk.convert_layout_frag %19 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %21 = frisk.unpack_frag %20 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %22 = frisk.pack_frag %18#4 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %23 = frisk.convert_layout_frag %22 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %24 = frisk.unpack_frag %23 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %25 = frisk.pack_frag %18#3 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %26 = frisk.convert_layout_frag %25 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %27 = frisk.unpack_frag %26 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %28 = frisk.pack_frag %18#0 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %29 = frisk.convert_layout_frag %28 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %30 = frisk.unpack_frag %29 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %31 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = frisk.extract_frag %18#1[%59, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %61 = frisk.broadcast_frag %60 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.insert_frag %61, %arg5[%59, %62] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %63 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %32 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %30[%59, %60] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.extract_frag %31[%59, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = frisk.elementwise_frag "div" %61, %63 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %65 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %66 = frisk.insert_frag %64, %arg5[%59, %65] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %33 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_4) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %32[%59, %60] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %62 = frisk.elementwise_frag "cast" %61 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %63 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %64 = frisk.insert_frag %62, %arg5[%59, %63] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %64 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %34 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %35 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_4) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %33[%59, %60] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.insert_frag %61, %arg5[%59, %62] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %63 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x = gpu.thread_id x
    %36 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %37 = arith.cmpi ult, %36, %c4 : index
    scf.if %37 {
      affine.for %arg4 = 0 to 4 {
        %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %61 = frisk.extract_frag %35[%59, %60] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %61, %34[%59, %60] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %38 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = frisk.extract_frag %18#1[%59, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %61 = frisk.broadcast_frag %60 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.insert_frag %61, %arg5[%59, %62] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %63 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %39 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %27[%59, %60] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.extract_frag %38[%59, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = frisk.elementwise_frag "div" %61, %63 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %65 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %66 = frisk.insert_frag %64, %arg5[%59, %65] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %40 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_4) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %39[%59, %60] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %62 = frisk.elementwise_frag "cast" %61 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %63 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %64 = frisk.insert_frag %62, %arg5[%59, %63] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %64 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %41 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %42 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_4) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %40[%59, %60] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.insert_frag %61, %arg5[%59, %62] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %63 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_12 = gpu.thread_id x
    %43 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x_12]
    %44 = arith.cmpi ult, %43, %c4 : index
    scf.if %44 {
      affine.for %arg4 = 0 to 4 {
        %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %61 = frisk.extract_frag %42[%59, %60] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %61, %41[%59, %60] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %45 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = frisk.extract_frag %18#1[%59, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %61 = frisk.broadcast_frag %60 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.insert_frag %61, %arg5[%59, %62] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %63 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %46 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %24[%59, %60] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.extract_frag %45[%59, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = frisk.elementwise_frag "div" %61, %63 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %65 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %66 = frisk.insert_frag %64, %arg5[%59, %65] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %47 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_4) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %46[%59, %60] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %62 = frisk.elementwise_frag "cast" %61 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %63 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %64 = frisk.insert_frag %62, %arg5[%59, %63] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %64 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %48 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %49 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_4) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %47[%59, %60] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.insert_frag %61, %arg5[%59, %62] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %63 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_13 = gpu.thread_id x
    %50 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x_13]
    %51 = arith.cmpi ult, %50, %c4 : index
    scf.if %51 {
      affine.for %arg4 = 0 to 4 {
        %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %61 = frisk.extract_frag %49[%59, %60] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %61, %48[%59, %60] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %52 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = frisk.extract_frag %18#1[%59, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %61 = frisk.broadcast_frag %60 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.insert_frag %61, %arg5[%59, %62] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %63 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %53 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %21[%59, %60] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.extract_frag %52[%59, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = frisk.elementwise_frag "div" %61, %63 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %65 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %66 = frisk.insert_frag %64, %arg5[%59, %65] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %54 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_4) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %53[%59, %60] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %62 = frisk.elementwise_frag "cast" %61 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %63 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %64 = frisk.insert_frag %62, %arg5[%59, %63] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %64 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %55 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %56 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_4) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %54[%59, %60] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %62 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %63 = frisk.insert_frag %61, %arg5[%59, %62] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %63 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_14 = gpu.thread_id x
    %57 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x_14]
    %58 = arith.cmpi ult, %57, %c4 : index
    scf.if %58 {
      affine.for %arg4 = 0 to 4 {
        %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
        %61 = frisk.extract_frag %56[%59, %60] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %61, %55[%59, %60] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
