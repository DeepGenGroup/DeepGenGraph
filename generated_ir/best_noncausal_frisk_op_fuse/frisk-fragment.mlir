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
    %8 = arith.truncf %cst : f32 to f16
    %cst_3 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %9 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_3) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %48 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %cst_34 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
      %49 = frisk.pack_frag %cst_34 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
      %50 = frisk.insert_frag %49, %arg5[%48, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %50 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %10 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_4) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %48 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %49 = frisk.broadcast_frag %cst_2 : f32 -> <vector<1x1xf32>, {}>
      %50 = frisk.insert_frag %49, %arg5[%48, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %50 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_5) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %48 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %49 = frisk.extract_frag %9[%48, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %50 = frisk.extract_frag %10[%48, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %51 = frisk.elementwise_frag "add" %49, %50 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %52 = frisk.insert_frag %51, %arg5[%48, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %52 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %12 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_8) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %48 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %49 = frisk.extract_frag %11[%48, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %50 = frisk.elementwise_frag "cast" %49 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %51 = frisk.insert_frag %50, %arg5[%48, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %51 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %13 = frisk.from_threadTile %12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %14:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_6, %arg6 = %cst_7, %arg7 = %13, %arg8 = %cst_9, %arg9 = %cst_10, %arg10 = %cst_11) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %cst_34 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %48 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_34) -> (vector<128x128xf32>) {
        %alloc_71 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_72 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %112 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        %113 = frisk.async_copy_gm_to_shm %112 to %alloc_71 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
        %114 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %113 : !frisk.async_task
        %115 = frisk.async_copy_gm_to_shm %114 to %alloc_72 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
        %116 = frisk.to_threadTile %alloc_71 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
        %cst_73 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
        %117 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_73) -> (vector<2x8xf16>) {
          %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %126 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg13)
          %127 = frisk.broadcast_frag %8 : f16 -> <vector<1x4xf16>, {}>
          %128 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%125)
          %129 = frisk.insert_frag %127, %arg14[%126, %128] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %129 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        %cst_74 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
        %118 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_74) -> (vector<2x8xf16>) {
          %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %126 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg13)
          %127 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%125)
          %128 = frisk.extract_frag %116[%126, %127] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %129 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%125)
          %130 = frisk.extract_frag %117[%126, %129] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %131 = frisk.elementwise_frag "mul" %128, %130 : (!frisk.fragment<vector<1x4xf16>, {}>, !frisk.fragment<vector<1x4xf16>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
          %132 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%125)
          %133 = frisk.insert_frag %131, %arg14[%126, %132] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %133 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        %119 = frisk.from_threadTile %118 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf16> -> memref<128x32xf16, 3>
        %120 = frisk.to_threadTile %arg12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
        %cst_75 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %121 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_75) -> (vector<2x32xf32>) {
          %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %126 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %127 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%125)
          %128 = frisk.extract_frag %120[%126, %127] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %129 = frisk.elementwise_frag "cast" %128 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %130 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%125)
          %131 = frisk.insert_frag %129, %arg14[%126, %130] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %131 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        frisk.wait_buffer_ready %115 : !frisk.async_task
        %122 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %121) -> (vector<2x32xf32>) {
          %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %126 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %127 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%125)
          %128 = frisk.extract_frag %121[%126, %127] : vector<2x32xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %129 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %128) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %131 = frisk.load_frag %119[%126, %arg15] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %132 = frisk.load_frag %alloc_72[%arg15, %127] {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 3, 0>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tile_type = vector<8x8xf16>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %133 = frisk.gemm_frag %131, %132, acc = %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            affine.yield %133 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %130 = frisk.insert_frag %129, %arg14[%126, %127] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x32xf32>
          affine.yield %130 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %cst_76 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %123 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_76) -> (vector<2x32xf32>) {
          %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %126 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %127 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%125)
          %128 = frisk.extract_frag %122[%126, %127] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %129 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%125)
          %130 = frisk.insert_frag %128, %arg14[%126, %129] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %130 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %124 = frisk.from_threadTile %123 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
        affine.yield %124 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %49 = frisk.to_threadTile %48 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_35 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %50 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_35) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %49[%113, %114] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %116 = frisk.elementwise_frag "cast" %115 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %117 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %118 = frisk.insert_frag %116, %arg12[%113, %117] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %118 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %51 = frisk.pack_frag %50 : vector<2x32xf32> -> <vector<2x32xf32>, {}>
      %52 = frisk.reduce_frag "max" %51 {dim = 1 : i64, lane_extent = 4 : i64, lane_stride = 16 : i64, output_fragment_shape = array<i64: 1, 1>, reduction_width = 4 : i64, warp_threads = 64 : i64} : <vector<2x32xf32>, {}> -> <vector<2x1xf32>, {}>
      %53 = frisk.unpack_frag %52 : <vector<2x1xf32>, {}> -> vector<2x1xf32>
      %54 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_36 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %55 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_36) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %113 = frisk.extract_frag %53[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.extract_frag %54[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %115 = frisk.elementwise_frag "max" %113, %114 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %116 = frisk.insert_frag %115, %arg12[%112, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %116 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %56 = frisk.to_threadTile %48 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_37 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %57 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_37) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %114 = frisk.extract_frag %55[%113, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %115 = frisk.broadcast_frag %114 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.insert_frag %115, %arg12[%113, %116] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %117 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_38 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %58 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_38) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %56[%113, %114] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.extract_frag %57[%113, %116] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %118 = frisk.elementwise_frag "sub" %115, %117 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %120 = frisk.insert_frag %118, %arg12[%113, %119] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %120 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %59 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_39 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %60 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_39) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %113 = frisk.extract_frag %59[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.extract_frag %55[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %115 = frisk.elementwise_frag "sub" %113, %114 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %116 = frisk.insert_frag %115, %arg12[%112, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %116 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_40 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %61 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_40) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %113 = frisk.extract_frag %60[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.elementwise_frag "exp2" %113 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %115 = frisk.insert_frag %114, %arg12[%112, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %115 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %62 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_41 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %63 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_41) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %113 = frisk.extract_frag %62[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.extract_frag %61[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %115 = frisk.elementwise_frag "mul" %113, %114 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %116 = frisk.insert_frag %115, %arg12[%112, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %116 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_42 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %64 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_42) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %58[%113, %114] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %116 = frisk.elementwise_frag "exp2" %115 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %117 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %118 = frisk.insert_frag %116, %arg12[%113, %117] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %118 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_43 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %65 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_43) -> (vector<2x32xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %64[%113, %114] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %116 = frisk.elementwise_frag "cast" %115 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %117 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %118 = frisk.insert_frag %116, %arg12[%113, %117] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %118 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %66 = frisk.pack_frag %65 : vector<2x32xf32> -> <vector<2x32xf32>, {}>
      %67 = frisk.reduce_frag "add" %66 {dim = 1 : i64, lane_extent = 4 : i64, lane_stride = 16 : i64, output_fragment_shape = array<i64: 1, 1>, reduction_width = 4 : i64, warp_threads = 64 : i64} : <vector<2x32xf32>, {}> -> <vector<2x1xf32>, {}>
      %68 = frisk.unpack_frag %67 : <vector<2x1xf32>, {}> -> vector<2x1xf32>
      %cst_44 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %69 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_44) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %113 = frisk.extract_frag %68[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.extract_frag %63[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %115 = frisk.elementwise_frag "add" %113, %114 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %116 = frisk.insert_frag %115, %arg12[%112, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %116 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_45 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
      %70 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_45) -> (vector<2x32xf16>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %64[%113, %114] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %116 = frisk.elementwise_frag "cast" %115 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %117 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %118 = frisk.insert_frag %116, %arg12[%113, %117] : <vector<1x4xf16>, {}> into vector<2x32xf16>
        affine.yield %118 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %71 = frisk.from_threadTile %70 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x128xf16, 3>
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %72 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %73 = frisk.async_copy_gm_to_shm %72 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %73 : !frisk.async_task
      %74 = frisk.to_threadTile %arg5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %cst_46 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %75 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_46) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %114 = frisk.extract_frag %61[%113, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %115 = frisk.broadcast_frag %114 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.insert_frag %115, %arg12[%113, %116] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_47 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %76 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_47) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %74[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.extract_frag %75[%113, %116] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %118 = frisk.elementwise_frag "mul" %115, %117 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %120 = frisk.insert_frag %118, %arg12[%113, %119] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %120 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %thread_id_x_48 = gpu.thread_id x
      %cst_49 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %77 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_49) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_48]
        %114 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_48]
        %115 = frisk.read_matrix_from_shm %alloc[%113, %114] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %116 = vector.bitcast %115 : vector<4xi32> to vector<8xf16>
        %cst_71 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
        %117 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_48]
        %118 = vector.extract %116[%117] : f16 from vector<8xf16>
        %119 = vector.insert %118, %cst_71 [0, 0] : f16 into vector<4x2xf16>
        %120 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_48]
        %121 = vector.extract %116[%120] : f16 from vector<8xf16>
        %122 = vector.insert %121, %119 [0, 1] : f16 into vector<4x2xf16>
        %123 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_48]
        %124 = vector.extract %116[%123] : f16 from vector<8xf16>
        %125 = vector.insert %124, %122 [1, 0] : f16 into vector<4x2xf16>
        %126 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_48]
        %127 = vector.extract %116[%126] : f16 from vector<8xf16>
        %128 = vector.insert %127, %125 [1, 1] : f16 into vector<4x2xf16>
        %129 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_48]
        %130 = vector.extract %116[%129] : f16 from vector<8xf16>
        %131 = vector.insert %130, %128 [2, 0] : f16 into vector<4x2xf16>
        %132 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_48]
        %133 = vector.extract %116[%132] : f16 from vector<8xf16>
        %134 = vector.insert %133, %131 [2, 1] : f16 into vector<4x2xf16>
        %135 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_48]
        %136 = vector.extract %116[%135] : f16 from vector<8xf16>
        %137 = vector.insert %136, %134 [3, 0] : f16 into vector<4x2xf16>
        %138 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_48]
        %139 = vector.extract %116[%138] : f16 from vector<8xf16>
        %140 = vector.insert %139, %137 [3, 1] : f16 into vector<4x2xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %142 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %143 = frisk.pack_frag %140 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %142] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %144 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %78 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %76) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %76[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %116 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %115) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %118 = frisk.load_frag %71[%113, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %119 = frisk.extract_frag %77[%arg13, %114] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %120 = frisk.gemm_frag %118, %119, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %120 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %117 = frisk.insert_frag %116, %arg12[%113, %114] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %alloc_50 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %79 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %80 = frisk.async_copy_gm_to_shm %79 to %alloc_50 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %81 = frisk.to_threadTile %arg8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %cst_51 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %82 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_51) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %114 = frisk.extract_frag %61[%113, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %115 = frisk.broadcast_frag %114 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.insert_frag %115, %arg12[%113, %116] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_52 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %83 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_52) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %81[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.extract_frag %82[%113, %116] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %118 = frisk.elementwise_frag "mul" %115, %117 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %120 = frisk.insert_frag %118, %arg12[%113, %119] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %120 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %80 : !frisk.async_task
      %thread_id_x_53 = gpu.thread_id x
      %cst_54 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %84 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_54) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_53]
        %114 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_53]
        %115 = frisk.read_matrix_from_shm %alloc_50[%113, %114] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %116 = vector.bitcast %115 : vector<4xi32> to vector<8xf16>
        %cst_71 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
        %117 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_53]
        %118 = vector.extract %116[%117] : f16 from vector<8xf16>
        %119 = vector.insert %118, %cst_71 [0, 0] : f16 into vector<4x2xf16>
        %120 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_53]
        %121 = vector.extract %116[%120] : f16 from vector<8xf16>
        %122 = vector.insert %121, %119 [0, 1] : f16 into vector<4x2xf16>
        %123 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_53]
        %124 = vector.extract %116[%123] : f16 from vector<8xf16>
        %125 = vector.insert %124, %122 [1, 0] : f16 into vector<4x2xf16>
        %126 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_53]
        %127 = vector.extract %116[%126] : f16 from vector<8xf16>
        %128 = vector.insert %127, %125 [1, 1] : f16 into vector<4x2xf16>
        %129 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_53]
        %130 = vector.extract %116[%129] : f16 from vector<8xf16>
        %131 = vector.insert %130, %128 [2, 0] : f16 into vector<4x2xf16>
        %132 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_53]
        %133 = vector.extract %116[%132] : f16 from vector<8xf16>
        %134 = vector.insert %133, %131 [2, 1] : f16 into vector<4x2xf16>
        %135 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_53]
        %136 = vector.extract %116[%135] : f16 from vector<8xf16>
        %137 = vector.insert %136, %134 [3, 0] : f16 into vector<4x2xf16>
        %138 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_53]
        %139 = vector.extract %116[%138] : f16 from vector<8xf16>
        %140 = vector.insert %139, %137 [3, 1] : f16 into vector<4x2xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %142 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %143 = frisk.pack_frag %140 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %142] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %144 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %85 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %83) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %83[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %116 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %115) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %118 = frisk.load_frag %71[%113, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %119 = frisk.extract_frag %84[%arg13, %114] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %120 = frisk.gemm_frag %118, %119, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %120 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %117 = frisk.insert_frag %116, %arg12[%113, %114] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %alloc_55 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %86 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %87 = frisk.async_copy_gm_to_shm %86 to %alloc_55 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %88 = frisk.to_threadTile %arg9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %cst_56 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %89 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_56) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %114 = frisk.extract_frag %61[%113, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %115 = frisk.broadcast_frag %114 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.insert_frag %115, %arg12[%113, %116] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_57 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %90 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_57) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %88[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.extract_frag %89[%113, %116] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %118 = frisk.elementwise_frag "mul" %115, %117 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %120 = frisk.insert_frag %118, %arg12[%113, %119] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %120 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %87 : !frisk.async_task
      %thread_id_x_58 = gpu.thread_id x
      %cst_59 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %91 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_59) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_58]
        %114 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_58]
        %115 = frisk.read_matrix_from_shm %alloc_55[%113, %114] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %116 = vector.bitcast %115 : vector<4xi32> to vector<8xf16>
        %cst_71 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
        %117 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_58]
        %118 = vector.extract %116[%117] : f16 from vector<8xf16>
        %119 = vector.insert %118, %cst_71 [0, 0] : f16 into vector<4x2xf16>
        %120 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_58]
        %121 = vector.extract %116[%120] : f16 from vector<8xf16>
        %122 = vector.insert %121, %119 [0, 1] : f16 into vector<4x2xf16>
        %123 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_58]
        %124 = vector.extract %116[%123] : f16 from vector<8xf16>
        %125 = vector.insert %124, %122 [1, 0] : f16 into vector<4x2xf16>
        %126 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_58]
        %127 = vector.extract %116[%126] : f16 from vector<8xf16>
        %128 = vector.insert %127, %125 [1, 1] : f16 into vector<4x2xf16>
        %129 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_58]
        %130 = vector.extract %116[%129] : f16 from vector<8xf16>
        %131 = vector.insert %130, %128 [2, 0] : f16 into vector<4x2xf16>
        %132 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_58]
        %133 = vector.extract %116[%132] : f16 from vector<8xf16>
        %134 = vector.insert %133, %131 [2, 1] : f16 into vector<4x2xf16>
        %135 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_58]
        %136 = vector.extract %116[%135] : f16 from vector<8xf16>
        %137 = vector.insert %136, %134 [3, 0] : f16 into vector<4x2xf16>
        %138 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_58]
        %139 = vector.extract %116[%138] : f16 from vector<8xf16>
        %140 = vector.insert %139, %137 [3, 1] : f16 into vector<4x2xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %142 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %143 = frisk.pack_frag %140 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %142] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %144 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %92 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %90) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %90[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %116 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %115) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %118 = frisk.load_frag %71[%113, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %119 = frisk.extract_frag %91[%arg13, %114] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %120 = frisk.gemm_frag %118, %119, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %120 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %117 = frisk.insert_frag %116, %arg12[%113, %114] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %alloc_60 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %93 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %94 = frisk.async_copy_gm_to_shm %93 to %alloc_60 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %95 = frisk.to_threadTile %arg10 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %cst_61 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %96 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_61) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %114 = frisk.extract_frag %61[%113, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %115 = frisk.broadcast_frag %114 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.insert_frag %115, %arg12[%113, %116] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_62 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %97 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_62) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %95[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.extract_frag %96[%113, %116] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %118 = frisk.elementwise_frag "mul" %115, %117 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %120 = frisk.insert_frag %118, %arg12[%113, %119] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %120 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %94 : !frisk.async_task
      %thread_id_x_63 = gpu.thread_id x
      %cst_64 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %98 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_64) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_63]
        %114 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_63]
        %115 = frisk.read_matrix_from_shm %alloc_60[%113, %114] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %116 = vector.bitcast %115 : vector<4xi32> to vector<8xf16>
        %cst_71 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
        %117 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_63]
        %118 = vector.extract %116[%117] : f16 from vector<8xf16>
        %119 = vector.insert %118, %cst_71 [0, 0] : f16 into vector<4x2xf16>
        %120 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_63]
        %121 = vector.extract %116[%120] : f16 from vector<8xf16>
        %122 = vector.insert %121, %119 [0, 1] : f16 into vector<4x2xf16>
        %123 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_63]
        %124 = vector.extract %116[%123] : f16 from vector<8xf16>
        %125 = vector.insert %124, %122 [1, 0] : f16 into vector<4x2xf16>
        %126 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_63]
        %127 = vector.extract %116[%126] : f16 from vector<8xf16>
        %128 = vector.insert %127, %125 [1, 1] : f16 into vector<4x2xf16>
        %129 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_63]
        %130 = vector.extract %116[%129] : f16 from vector<8xf16>
        %131 = vector.insert %130, %128 [2, 0] : f16 into vector<4x2xf16>
        %132 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_63]
        %133 = vector.extract %116[%132] : f16 from vector<8xf16>
        %134 = vector.insert %133, %131 [2, 1] : f16 into vector<4x2xf16>
        %135 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_63]
        %136 = vector.extract %116[%135] : f16 from vector<8xf16>
        %137 = vector.insert %136, %134 [3, 0] : f16 into vector<4x2xf16>
        %138 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_63]
        %139 = vector.extract %116[%138] : f16 from vector<8xf16>
        %140 = vector.insert %139, %137 [3, 1] : f16 into vector<4x2xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %142 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %143 = frisk.pack_frag %140 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %142] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %144 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %99 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %97) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %97[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %116 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %115) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %118 = frisk.load_frag %71[%113, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %119 = frisk.extract_frag %98[%arg13, %114] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %120 = frisk.gemm_frag %118, %119, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %120 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %117 = frisk.insert_frag %116, %arg12[%113, %114] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %cst_65 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %100 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_65) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %78[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.insert_frag %115, %arg12[%113, %116] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %101 = frisk.from_threadTile %100 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_66 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %102 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_66) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %113 = frisk.extract_frag %69[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.insert_frag %113, %arg12[%112, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %114 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %103 = frisk.from_threadTile %102 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_67 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %104 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_67) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %113 = frisk.extract_frag %55[%112, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %114 = frisk.insert_frag %113, %arg12[%112, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %114 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %105 = frisk.from_threadTile %104 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_68 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %106 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_68) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %85[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.insert_frag %115, %arg12[%113, %116] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %107 = frisk.from_threadTile %106 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_69 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %108 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_69) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %92[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.insert_frag %115, %arg12[%113, %116] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %109 = frisk.from_threadTile %108 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_70 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %110 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_70) -> (vector<2x8xf32>) {
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %113 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %114 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %115 = frisk.extract_frag %99[%113, %114] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %116 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%112)
        %117 = frisk.insert_frag %115, %arg12[%113, %116] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %111 = frisk.from_threadTile %110 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      affine.yield %101, %103, %105, %107, %109, %111 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %15 = frisk.to_threadTile %14#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %16 = frisk.to_threadTile %14#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %17 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_12) -> (vector<2x8xf32>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %c0 = arith.constant 0 : index
      %50 = frisk.extract_frag %15[%49, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %51 = frisk.broadcast_frag %50 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.insert_frag %51, %arg5[%49, %52] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %53 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_13 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %18 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_13) -> (vector<2x8xf32>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %16[%49, %50] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.extract_frag %17[%49, %52] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %54 = frisk.elementwise_frag "div" %51, %53 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %55 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %56 = frisk.insert_frag %54, %arg5[%49, %55] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %56 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_14 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %19 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_14) -> (vector<2x8xf16>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %18[%49, %50] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %52 = frisk.elementwise_frag "cast" %51 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %53 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %54 = frisk.insert_frag %52, %arg5[%49, %53] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %54 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %20 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_15 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %21 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_15) -> (vector<2x8xf16>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %19[%49, %50] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.insert_frag %51, %arg5[%49, %52] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %53 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x = gpu.thread_id x
    %22 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x)
    %c4 = arith.constant 4 : index
    %23 = arith.cmpi ult, %22, %c4 : index
    scf.if %23 {
      affine.for %arg4 = 0 to 4 {
        %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
        %51 = frisk.extract_frag %21[%49, %50] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %51, %20[%49, %50] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %24 = frisk.to_threadTile %14#3 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %cst_16 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %25 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_16) -> (vector<2x8xf32>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %c0 = arith.constant 0 : index
      %50 = frisk.extract_frag %15[%49, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %51 = frisk.broadcast_frag %50 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.insert_frag %51, %arg5[%49, %52] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %53 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_17 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %26 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_17) -> (vector<2x8xf32>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %24[%49, %50] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.extract_frag %25[%49, %52] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %54 = frisk.elementwise_frag "div" %51, %53 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %55 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %56 = frisk.insert_frag %54, %arg5[%49, %55] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %56 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_18 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %27 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_18) -> (vector<2x8xf16>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %26[%49, %50] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %52 = frisk.elementwise_frag "cast" %51 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %53 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %54 = frisk.insert_frag %52, %arg5[%49, %53] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %54 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %28 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_19 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %29 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_19) -> (vector<2x8xf16>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %27[%49, %50] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.insert_frag %51, %arg5[%49, %52] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %53 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_20 = gpu.thread_id x
    %30 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_20)
    %c4_21 = arith.constant 4 : index
    %31 = arith.cmpi ult, %30, %c4_21 : index
    scf.if %31 {
      affine.for %arg4 = 0 to 4 {
        %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
        %51 = frisk.extract_frag %29[%49, %50] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %51, %28[%49, %50] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %32 = frisk.to_threadTile %14#4 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %cst_22 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %33 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_22) -> (vector<2x8xf32>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %c0 = arith.constant 0 : index
      %50 = frisk.extract_frag %15[%49, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %51 = frisk.broadcast_frag %50 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.insert_frag %51, %arg5[%49, %52] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %53 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_23 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %34 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_23) -> (vector<2x8xf32>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %32[%49, %50] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.extract_frag %33[%49, %52] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %54 = frisk.elementwise_frag "div" %51, %53 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %55 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %56 = frisk.insert_frag %54, %arg5[%49, %55] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %56 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_24 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %35 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_24) -> (vector<2x8xf16>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %34[%49, %50] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %52 = frisk.elementwise_frag "cast" %51 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %53 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %54 = frisk.insert_frag %52, %arg5[%49, %53] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %54 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %36 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_25 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %37 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_25) -> (vector<2x8xf16>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %35[%49, %50] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.insert_frag %51, %arg5[%49, %52] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %53 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_26 = gpu.thread_id x
    %38 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_26)
    %c4_27 = arith.constant 4 : index
    %39 = arith.cmpi ult, %38, %c4_27 : index
    scf.if %39 {
      affine.for %arg4 = 0 to 4 {
        %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
        %51 = frisk.extract_frag %37[%49, %50] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %51, %36[%49, %50] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %40 = frisk.to_threadTile %14#5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %cst_28 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %41 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_28) -> (vector<2x8xf32>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %c0 = arith.constant 0 : index
      %50 = frisk.extract_frag %15[%49, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %51 = frisk.broadcast_frag %50 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.insert_frag %51, %arg5[%49, %52] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %53 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_29 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %42 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_29) -> (vector<2x8xf32>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %40[%49, %50] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.extract_frag %41[%49, %52] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %54 = frisk.elementwise_frag "div" %51, %53 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %55 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %56 = frisk.insert_frag %54, %arg5[%49, %55] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %56 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_30 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %43 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_30) -> (vector<2x8xf16>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %42[%49, %50] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %52 = frisk.elementwise_frag "cast" %51 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %53 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %54 = frisk.insert_frag %52, %arg5[%49, %53] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %54 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %44 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_31 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %45 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_31) -> (vector<2x8xf16>) {
      %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %51 = frisk.extract_frag %43[%49, %50] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %52 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
      %53 = frisk.insert_frag %51, %arg5[%49, %52] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %53 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_32 = gpu.thread_id x
    %46 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_32)
    %c4_33 = arith.constant 4 : index
    %47 = arith.cmpi ult, %46, %c4_33 : index
    scf.if %47 {
      affine.for %arg4 = 0 to 4 {
        %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %49 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        %50 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%48)
        %51 = frisk.extract_frag %45[%49, %50] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %51, %44[%49, %50] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
