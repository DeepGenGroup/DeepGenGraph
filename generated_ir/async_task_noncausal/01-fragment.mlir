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
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %cst_34 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
      %61 = frisk.pack_frag %cst_34 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
      %62 = frisk.insert_frag %61, %arg5[%60, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %62 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %10 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_4) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.broadcast_frag %cst_2 : f32 -> <vector<1x1xf32>, {}>
      %62 = frisk.insert_frag %61, %arg5[%60, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %62 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_5) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %9[%60, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %62 = frisk.extract_frag %10[%60, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %63 = frisk.elementwise_frag "add" %61, %62 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %64 = frisk.insert_frag %63, %arg5[%60, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %64 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %12 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_8) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %60 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %61 = frisk.extract_frag %11[%60, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %62 = frisk.elementwise_frag "cast" %61 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
      %63 = frisk.insert_frag %62, %arg5[%60, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %63 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %13 = frisk.from_threadTile %12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %14:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_6, %arg6 = %cst_7, %arg7 = %13, %arg8 = %cst_9, %arg9 = %cst_10, %arg10 = %cst_11) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %cst_34 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %60 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_34) -> (vector<128x128xf32>) {
        %alloc_83 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_84 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %137 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        %138 = frisk.async_copy_gm_to_shm %137 to %alloc_83 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
        %139 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %138 : !frisk.async_task
        %140 = frisk.async_copy_gm_to_shm %139 to %alloc_84 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
        %141 = frisk.to_threadTile %alloc_83 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
        %cst_85 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
        %142 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_85) -> (vector<2x8xf16>) {
          %151 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %152 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg13)
          %153 = frisk.broadcast_frag %8 : f16 -> <vector<1x4xf16>, {}>
          %154 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %155 = frisk.insert_frag %153, %arg14[%152, %154] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %155 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        %cst_86 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
        %143 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_86) -> (vector<2x8xf16>) {
          %151 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %152 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg13)
          %153 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %154 = frisk.extract_frag %141[%152, %153] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %155 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %156 = frisk.extract_frag %142[%152, %155] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          %157 = frisk.elementwise_frag "mul" %154, %156 : (!frisk.fragment<vector<1x4xf16>, {}>, !frisk.fragment<vector<1x4xf16>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
          %158 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %159 = frisk.insert_frag %157, %arg14[%152, %158] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %159 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        %144 = frisk.from_threadTile %143 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf16> -> memref<128x32xf16, 3>
        frisk.wait_buffer_ready %140 : !frisk.async_task
        %cst_87 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %cst_88 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %145 = frisk.pack_frag %cst_88 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %146 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_87) -> (vector<2x32xf32>) {
          %151 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %152 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %153 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %154 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %145) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %156 = frisk.load_frag %144[%152, %arg15] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %157 = frisk.load_frag %alloc_84[%arg15, %153] {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 3, 0>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tile_type = vector<8x8xf16>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            %158 = frisk.gemm_frag %156, %157, acc = %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            affine.yield %158 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %155 = frisk.insert_frag %154, %arg14[%152, %153] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x32xf32>
          affine.yield %155 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %147 = frisk.to_threadTile %arg12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
        %cst_89 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %148 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_89) -> (vector<2x32xf32>) {
          %151 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %152 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %153 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %154 = frisk.extract_frag %147[%152, %153] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %155 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %156 = frisk.extract_frag %146[%152, %155] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %157 = frisk.elementwise_frag "add" %154, %156 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %158 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %159 = frisk.insert_frag %157, %arg14[%152, %158] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %159 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %cst_90 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %149 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_90) -> (vector<2x32xf32>) {
          %151 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %152 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %153 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %154 = frisk.extract_frag %148[%152, %153] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %155 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%151)
          %156 = frisk.insert_frag %154, %arg14[%152, %155] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %156 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %150 = frisk.from_threadTile %149 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
        affine.yield %150 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %61 = frisk.to_threadTile %60 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_35 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %62 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_35) -> (vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %61[%138, %139] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %141 = frisk.elementwise_frag "cast" %140 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %142 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %143 = frisk.insert_frag %141, %arg12[%138, %142] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %143 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %63 = frisk.pack_frag %62 : vector<2x32xf32> -> <vector<2x32xf32>, {}>
      %64 = frisk.reduce_frag "max" %63 {dim = 1 : i64, lane_extent = 4 : i64, lane_stride = 16 : i64, output_fragment_shape = array<i64: 1, 1>, reduction_width = 4 : i64, warp_threads = 64 : i64} : <vector<2x32xf32>, {}> -> <vector<2x1xf32>, {}>
      %65 = frisk.unpack_frag %64 : <vector<2x1xf32>, {}> -> vector<2x1xf32>
      %66 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_36 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %67 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_36) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %138 = frisk.extract_frag %65[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %139 = frisk.extract_frag %66[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %140 = frisk.elementwise_frag "max" %138, %139 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %141 = frisk.insert_frag %140, %arg12[%137, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %68 = frisk.to_threadTile %60 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_37 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %69 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_37) -> (vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %139 = frisk.extract_frag %67[%138, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %140 = frisk.broadcast_frag %139 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.insert_frag %140, %arg12[%138, %141] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %142 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_38 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %70 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_38) -> (vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %68[%138, %139] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.extract_frag %69[%138, %141] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %143 = frisk.elementwise_frag "sub" %140, %142 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %145 = frisk.insert_frag %143, %arg12[%138, %144] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %145 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %71 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_39 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %72 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_39) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %138 = frisk.extract_frag %71[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %139 = frisk.extract_frag %67[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %140 = frisk.elementwise_frag "sub" %138, %139 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %141 = frisk.insert_frag %140, %arg12[%137, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_40 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %73 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_40) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %138 = frisk.extract_frag %72[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %139 = frisk.elementwise_frag "exp2" %138 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %140 = frisk.insert_frag %139, %arg12[%137, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %140 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %74 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_41 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %75 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_41) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %138 = frisk.extract_frag %74[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %139 = frisk.extract_frag %73[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %140 = frisk.elementwise_frag "mul" %138, %139 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %141 = frisk.insert_frag %140, %arg12[%137, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_42 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %76 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_42) -> (vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %70[%138, %139] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %141 = frisk.elementwise_frag "exp2" %140 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %142 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %143 = frisk.insert_frag %141, %arg12[%138, %142] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %143 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_43 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %77 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_43) -> (vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %76[%138, %139] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %141 = frisk.elementwise_frag "cast" %140 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %142 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %143 = frisk.insert_frag %141, %arg12[%138, %142] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %143 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %78 = frisk.pack_frag %77 : vector<2x32xf32> -> <vector<2x32xf32>, {}>
      %79 = frisk.reduce_frag "add" %78 {dim = 1 : i64, lane_extent = 4 : i64, lane_stride = 16 : i64, output_fragment_shape = array<i64: 1, 1>, reduction_width = 4 : i64, warp_threads = 64 : i64} : <vector<2x32xf32>, {}> -> <vector<2x1xf32>, {}>
      %80 = frisk.unpack_frag %79 : <vector<2x1xf32>, {}> -> vector<2x1xf32>
      %cst_44 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %81 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_44) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %138 = frisk.extract_frag %80[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %139 = frisk.extract_frag %75[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %140 = frisk.elementwise_frag "add" %138, %139 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %141 = frisk.insert_frag %140, %arg12[%137, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_45 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
      %82 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_45) -> (vector<2x32xf16>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %76[%138, %139] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
        %141 = frisk.elementwise_frag "cast" %140 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %142 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %143 = frisk.insert_frag %141, %arg12[%138, %142] : <vector<1x4xf16>, {}> into vector<2x32xf16>
        affine.yield %143 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %83 = frisk.from_threadTile %82 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x128xf16, 3>
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %84 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %85 = frisk.async_copy_gm_to_shm %84 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %85 : !frisk.async_task
      %thread_id_x_46 = gpu.thread_id x
      %cst_47 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %86 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_47) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_46]
        %139 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_46]
        %140 = frisk.read_matrix_from_shm %alloc[%138, %139] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %141 = vector.bitcast %140 : vector<4xi32> to vector<8xf16>
        %cst_83 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
        %142 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_46]
        %143 = vector.extract %141[%142] : f16 from vector<8xf16>
        %144 = vector.insert %143, %cst_83 [0, 0] : f16 into vector<4x2xf16>
        %145 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_46]
        %146 = vector.extract %141[%145] : f16 from vector<8xf16>
        %147 = vector.insert %146, %144 [0, 1] : f16 into vector<4x2xf16>
        %148 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_46]
        %149 = vector.extract %141[%148] : f16 from vector<8xf16>
        %150 = vector.insert %149, %147 [1, 0] : f16 into vector<4x2xf16>
        %151 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_46]
        %152 = vector.extract %141[%151] : f16 from vector<8xf16>
        %153 = vector.insert %152, %150 [1, 1] : f16 into vector<4x2xf16>
        %154 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_46]
        %155 = vector.extract %141[%154] : f16 from vector<8xf16>
        %156 = vector.insert %155, %153 [2, 0] : f16 into vector<4x2xf16>
        %157 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_46]
        %158 = vector.extract %141[%157] : f16 from vector<8xf16>
        %159 = vector.insert %158, %156 [2, 1] : f16 into vector<4x2xf16>
        %160 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_46]
        %161 = vector.extract %141[%160] : f16 from vector<8xf16>
        %162 = vector.insert %161, %159 [3, 0] : f16 into vector<4x2xf16>
        %163 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_46]
        %164 = vector.extract %141[%163] : f16 from vector<8xf16>
        %165 = vector.insert %164, %162 [3, 1] : f16 into vector<4x2xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %167 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %168 = frisk.pack_frag %165 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %169 = frisk.insert_frag %168, %arg12[%166, %167] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %169 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_48 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_49 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %87 = frisk.pack_frag %cst_49 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %88 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_48) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %87) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %142 = frisk.load_frag %83[%138, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %143 = frisk.extract_frag %86[%arg13, %139] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %144 = frisk.gemm_frag %142, %143, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %144 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %141 = frisk.insert_frag %140, %arg12[%138, %139] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %141 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %89 = frisk.from_threadTile %73 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> memref<128x1xf32>
      %90 = frisk.to_threadTile %arg5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %91 = frisk.to_threadTile %89 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_50 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %92 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_50) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %139 = frisk.extract_frag %91[%138, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %140 = frisk.broadcast_frag %139 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.insert_frag %140, %arg12[%138, %141] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %142 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_51 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %93 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_51) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %90[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.extract_frag %92[%138, %141] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %143 = frisk.elementwise_frag "mul" %140, %142 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %145 = frisk.insert_frag %143, %arg12[%138, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_52 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %94 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_52) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %93[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.extract_frag %88[%138, %141] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %143 = frisk.elementwise_frag "add" %140, %142 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %145 = frisk.insert_frag %143, %arg12[%138, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_53 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %95 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %96 = frisk.async_copy_gm_to_shm %95 to %alloc_53 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %96 : !frisk.async_task
      %thread_id_x_54 = gpu.thread_id x
      %cst_55 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %97 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_55) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_54]
        %139 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_54]
        %140 = frisk.read_matrix_from_shm %alloc_53[%138, %139] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %141 = vector.bitcast %140 : vector<4xi32> to vector<8xf16>
        %cst_83 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
        %142 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_54]
        %143 = vector.extract %141[%142] : f16 from vector<8xf16>
        %144 = vector.insert %143, %cst_83 [0, 0] : f16 into vector<4x2xf16>
        %145 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_54]
        %146 = vector.extract %141[%145] : f16 from vector<8xf16>
        %147 = vector.insert %146, %144 [0, 1] : f16 into vector<4x2xf16>
        %148 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_54]
        %149 = vector.extract %141[%148] : f16 from vector<8xf16>
        %150 = vector.insert %149, %147 [1, 0] : f16 into vector<4x2xf16>
        %151 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_54]
        %152 = vector.extract %141[%151] : f16 from vector<8xf16>
        %153 = vector.insert %152, %150 [1, 1] : f16 into vector<4x2xf16>
        %154 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_54]
        %155 = vector.extract %141[%154] : f16 from vector<8xf16>
        %156 = vector.insert %155, %153 [2, 0] : f16 into vector<4x2xf16>
        %157 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_54]
        %158 = vector.extract %141[%157] : f16 from vector<8xf16>
        %159 = vector.insert %158, %156 [2, 1] : f16 into vector<4x2xf16>
        %160 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_54]
        %161 = vector.extract %141[%160] : f16 from vector<8xf16>
        %162 = vector.insert %161, %159 [3, 0] : f16 into vector<4x2xf16>
        %163 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_54]
        %164 = vector.extract %141[%163] : f16 from vector<8xf16>
        %165 = vector.insert %164, %162 [3, 1] : f16 into vector<4x2xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %167 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %168 = frisk.pack_frag %165 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %169 = frisk.insert_frag %168, %arg12[%166, %167] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %169 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_56 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %98 = frisk.pack_frag %cst_57 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %99 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_56) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %98) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %142 = frisk.load_frag %83[%138, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %143 = frisk.extract_frag %97[%arg13, %139] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %144 = frisk.gemm_frag %142, %143, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %144 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %141 = frisk.insert_frag %140, %arg12[%138, %139] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %141 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %100 = frisk.to_threadTile %arg8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %101 = frisk.to_threadTile %89 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_58 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %102 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_58) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %139 = frisk.extract_frag %101[%138, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %140 = frisk.broadcast_frag %139 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.insert_frag %140, %arg12[%138, %141] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %142 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_59 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %103 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_59) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %100[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.extract_frag %102[%138, %141] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %143 = frisk.elementwise_frag "mul" %140, %142 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %145 = frisk.insert_frag %143, %arg12[%138, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_60 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %104 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_60) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %103[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.extract_frag %99[%138, %141] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %143 = frisk.elementwise_frag "add" %140, %142 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %145 = frisk.insert_frag %143, %arg12[%138, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_61 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %105 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %106 = frisk.async_copy_gm_to_shm %105 to %alloc_61 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %106 : !frisk.async_task
      %thread_id_x_62 = gpu.thread_id x
      %cst_63 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %107 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_63) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_62]
        %139 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_62]
        %140 = frisk.read_matrix_from_shm %alloc_61[%138, %139] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %141 = vector.bitcast %140 : vector<4xi32> to vector<8xf16>
        %cst_83 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
        %142 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_62]
        %143 = vector.extract %141[%142] : f16 from vector<8xf16>
        %144 = vector.insert %143, %cst_83 [0, 0] : f16 into vector<4x2xf16>
        %145 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_62]
        %146 = vector.extract %141[%145] : f16 from vector<8xf16>
        %147 = vector.insert %146, %144 [0, 1] : f16 into vector<4x2xf16>
        %148 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_62]
        %149 = vector.extract %141[%148] : f16 from vector<8xf16>
        %150 = vector.insert %149, %147 [1, 0] : f16 into vector<4x2xf16>
        %151 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_62]
        %152 = vector.extract %141[%151] : f16 from vector<8xf16>
        %153 = vector.insert %152, %150 [1, 1] : f16 into vector<4x2xf16>
        %154 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_62]
        %155 = vector.extract %141[%154] : f16 from vector<8xf16>
        %156 = vector.insert %155, %153 [2, 0] : f16 into vector<4x2xf16>
        %157 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_62]
        %158 = vector.extract %141[%157] : f16 from vector<8xf16>
        %159 = vector.insert %158, %156 [2, 1] : f16 into vector<4x2xf16>
        %160 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_62]
        %161 = vector.extract %141[%160] : f16 from vector<8xf16>
        %162 = vector.insert %161, %159 [3, 0] : f16 into vector<4x2xf16>
        %163 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_62]
        %164 = vector.extract %141[%163] : f16 from vector<8xf16>
        %165 = vector.insert %164, %162 [3, 1] : f16 into vector<4x2xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %167 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %168 = frisk.pack_frag %165 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %169 = frisk.insert_frag %168, %arg12[%166, %167] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %169 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_64 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_65 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %108 = frisk.pack_frag %cst_65 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %109 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_64) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %108) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %142 = frisk.load_frag %83[%138, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %143 = frisk.extract_frag %107[%arg13, %139] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %144 = frisk.gemm_frag %142, %143, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %144 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %141 = frisk.insert_frag %140, %arg12[%138, %139] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %141 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %110 = frisk.to_threadTile %arg9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %111 = frisk.to_threadTile %89 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_66 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %112 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_66) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %139 = frisk.extract_frag %111[%138, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %140 = frisk.broadcast_frag %139 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.insert_frag %140, %arg12[%138, %141] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %142 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_67 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %113 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_67) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %110[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.extract_frag %112[%138, %141] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %143 = frisk.elementwise_frag "mul" %140, %142 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %145 = frisk.insert_frag %143, %arg12[%138, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_68 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %114 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_68) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %113[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.extract_frag %109[%138, %141] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %143 = frisk.elementwise_frag "add" %140, %142 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %145 = frisk.insert_frag %143, %arg12[%138, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_69 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %115 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %116 = frisk.async_copy_gm_to_shm %115 to %alloc_69 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %116 : !frisk.async_task
      %thread_id_x_70 = gpu.thread_id x
      %cst_71 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %117 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_71) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_70]
        %139 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_70]
        %140 = frisk.read_matrix_from_shm %alloc_69[%138, %139] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %141 = vector.bitcast %140 : vector<4xi32> to vector<8xf16>
        %cst_83 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
        %142 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_70]
        %143 = vector.extract %141[%142] : f16 from vector<8xf16>
        %144 = vector.insert %143, %cst_83 [0, 0] : f16 into vector<4x2xf16>
        %145 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_70]
        %146 = vector.extract %141[%145] : f16 from vector<8xf16>
        %147 = vector.insert %146, %144 [0, 1] : f16 into vector<4x2xf16>
        %148 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_70]
        %149 = vector.extract %141[%148] : f16 from vector<8xf16>
        %150 = vector.insert %149, %147 [1, 0] : f16 into vector<4x2xf16>
        %151 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_70]
        %152 = vector.extract %141[%151] : f16 from vector<8xf16>
        %153 = vector.insert %152, %150 [1, 1] : f16 into vector<4x2xf16>
        %154 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_70]
        %155 = vector.extract %141[%154] : f16 from vector<8xf16>
        %156 = vector.insert %155, %153 [2, 0] : f16 into vector<4x2xf16>
        %157 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_70]
        %158 = vector.extract %141[%157] : f16 from vector<8xf16>
        %159 = vector.insert %158, %156 [2, 1] : f16 into vector<4x2xf16>
        %160 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_70]
        %161 = vector.extract %141[%160] : f16 from vector<8xf16>
        %162 = vector.insert %161, %159 [3, 0] : f16 into vector<4x2xf16>
        %163 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_70]
        %164 = vector.extract %141[%163] : f16 from vector<8xf16>
        %165 = vector.insert %164, %162 [3, 1] : f16 into vector<4x2xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %167 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %168 = frisk.pack_frag %165 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %169 = frisk.insert_frag %168, %arg12[%166, %167] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %169 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_72 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_73 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %118 = frisk.pack_frag %cst_73 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %119 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_72) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %118) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %142 = frisk.load_frag %83[%138, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %143 = frisk.extract_frag %117[%arg13, %139] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          %144 = frisk.gemm_frag %142, %143, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          affine.yield %144 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %141 = frisk.insert_frag %140, %arg12[%138, %139] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %141 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %120 = frisk.to_threadTile %arg10 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %121 = frisk.to_threadTile %89 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_74 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %122 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_74) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %c0 = arith.constant 0 : index
        %139 = frisk.extract_frag %121[%138, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %140 = frisk.broadcast_frag %139 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.insert_frag %140, %arg12[%138, %141] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %142 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_75 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %123 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_75) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %120[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.extract_frag %122[%138, %141] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %143 = frisk.elementwise_frag "mul" %140, %142 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %145 = frisk.insert_frag %143, %arg12[%138, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_76 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %124 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_76) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %123[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.extract_frag %119[%138, %141] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %143 = frisk.elementwise_frag "add" %140, %142 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %145 = frisk.insert_frag %143, %arg12[%138, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_77 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %125 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_77) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %94[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.insert_frag %140, %arg12[%138, %141] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %142 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %126 = frisk.from_threadTile %125 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_78 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %127 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_78) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %138 = frisk.extract_frag %81[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %139 = frisk.insert_frag %138, %arg12[%137, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %139 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %128 = frisk.from_threadTile %127 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_79 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %129 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_79) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %137 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %138 = frisk.extract_frag %67[%137, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %139 = frisk.insert_frag %138, %arg12[%137, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %139 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %130 = frisk.from_threadTile %129 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_80 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %131 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_80) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %104[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.insert_frag %140, %arg12[%138, %141] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %142 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %132 = frisk.from_threadTile %131 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_81 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %133 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_81) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %114[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.insert_frag %140, %arg12[%138, %141] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %142 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %134 = frisk.from_threadTile %133 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_82 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %135 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_82) -> (vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %140 = frisk.extract_frag %124[%138, %139] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%137)
        %142 = frisk.insert_frag %140, %arg12[%138, %141] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %142 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %136 = frisk.from_threadTile %135 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      affine.yield %126, %128, %130, %132, %134, %136 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %15 = frisk.to_threadTile %14#5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %16 = frisk.pack_frag %15 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %17 = frisk.convert_layout_frag %16 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %18 = frisk.unpack_frag %17 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %19 = frisk.to_threadTile %14#4 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %20 = frisk.pack_frag %19 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %21 = frisk.convert_layout_frag %20 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %22 = frisk.unpack_frag %21 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %23 = frisk.to_threadTile %14#3 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %24 = frisk.pack_frag %23 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %25 = frisk.convert_layout_frag %24 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %26 = frisk.unpack_frag %25 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %27 = frisk.to_threadTile %14#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %28 = frisk.pack_frag %27 : vector<2x8xf32> -> <vector<2x8xf32>, {}>
    %29 = frisk.convert_layout_frag %28 {block_type = vector<128x32xf32>, shuffle_plan = {algorithm = "xor_transpose", registers = 4 : i64, width = 64 : i64, xor_masks = array<i64: 16, 32>}, source_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}, target_layout = {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}} : <vector<2x8xf32>, {}> -> <vector<2x8xf32>, {}>
    %30 = frisk.unpack_frag %29 : <vector<2x8xf32>, {}> -> vector<2x8xf32>
    %31 = frisk.to_threadTile %14#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %32 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_12) -> (vector<2x8xf32>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %c0 = arith.constant 0 : index
      %62 = frisk.extract_frag %31[%61, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %63 = frisk.broadcast_frag %62 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.insert_frag %63, %arg5[%61, %64] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %65 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_13 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %33 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_13) -> (vector<2x8xf32>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %30[%61, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.extract_frag %32[%61, %64] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %66 = frisk.elementwise_frag "div" %63, %65 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %67 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %68 = frisk.insert_frag %66, %arg5[%61, %67] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %68 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_14 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %34 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_14) -> (vector<2x8xf16>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %33[%61, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = frisk.elementwise_frag "cast" %63 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %65 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %66 = frisk.insert_frag %64, %arg5[%61, %65] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %35 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_15 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %36 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_15) -> (vector<2x8xf16>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %34[%61, %62] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.insert_frag %63, %arg5[%61, %64] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %65 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x = gpu.thread_id x
    %37 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x)
    %c4 = arith.constant 4 : index
    %38 = arith.cmpi ult, %37, %c4 : index
    scf.if %38 {
      affine.for %arg4 = 0 to 4 {
        %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
        %63 = frisk.extract_frag %36[%61, %62] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %63, %35[%61, %62] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_16 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %39 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_16) -> (vector<2x8xf32>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %c0 = arith.constant 0 : index
      %62 = frisk.extract_frag %31[%61, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %63 = frisk.broadcast_frag %62 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.insert_frag %63, %arg5[%61, %64] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %65 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_17 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %40 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_17) -> (vector<2x8xf32>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %26[%61, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.extract_frag %39[%61, %64] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %66 = frisk.elementwise_frag "div" %63, %65 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %67 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %68 = frisk.insert_frag %66, %arg5[%61, %67] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %68 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_18 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %41 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_18) -> (vector<2x8xf16>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %40[%61, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = frisk.elementwise_frag "cast" %63 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %65 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %66 = frisk.insert_frag %64, %arg5[%61, %65] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %42 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_19 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %43 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_19) -> (vector<2x8xf16>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %41[%61, %62] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.insert_frag %63, %arg5[%61, %64] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %65 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_20 = gpu.thread_id x
    %44 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_20)
    %c4_21 = arith.constant 4 : index
    %45 = arith.cmpi ult, %44, %c4_21 : index
    scf.if %45 {
      affine.for %arg4 = 0 to 4 {
        %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
        %63 = frisk.extract_frag %43[%61, %62] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %63, %42[%61, %62] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_22 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %46 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_22) -> (vector<2x8xf32>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %c0 = arith.constant 0 : index
      %62 = frisk.extract_frag %31[%61, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %63 = frisk.broadcast_frag %62 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.insert_frag %63, %arg5[%61, %64] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %65 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_23 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %47 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_23) -> (vector<2x8xf32>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %22[%61, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.extract_frag %46[%61, %64] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %66 = frisk.elementwise_frag "div" %63, %65 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %67 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %68 = frisk.insert_frag %66, %arg5[%61, %67] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %68 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_24 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %48 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_24) -> (vector<2x8xf16>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %47[%61, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = frisk.elementwise_frag "cast" %63 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %65 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %66 = frisk.insert_frag %64, %arg5[%61, %65] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %49 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_25 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %50 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_25) -> (vector<2x8xf16>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %48[%61, %62] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.insert_frag %63, %arg5[%61, %64] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %65 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_26 = gpu.thread_id x
    %51 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_26)
    %c4_27 = arith.constant 4 : index
    %52 = arith.cmpi ult, %51, %c4_27 : index
    scf.if %52 {
      affine.for %arg4 = 0 to 4 {
        %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
        %63 = frisk.extract_frag %50[%61, %62] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %63, %49[%61, %62] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_28 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %53 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_28) -> (vector<2x8xf32>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %c0 = arith.constant 0 : index
      %62 = frisk.extract_frag %31[%61, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
      %63 = frisk.broadcast_frag %62 : !frisk.fragment<vector<1x1xf32>, {}> -> <vector<1x4xf32>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.insert_frag %63, %arg5[%61, %64] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %65 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_29 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %54 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_29) -> (vector<2x8xf32>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %18[%61, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.extract_frag %53[%61, %64] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %66 = frisk.elementwise_frag "div" %63, %65 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
      %67 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %68 = frisk.insert_frag %66, %arg5[%61, %67] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %68 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_30 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %55 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_30) -> (vector<2x8xf16>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %54[%61, %62] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
      %64 = frisk.elementwise_frag "cast" %63 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
      %65 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %66 = frisk.insert_frag %64, %arg5[%61, %65] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %56 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_31 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %57 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_31) -> (vector<2x8xf16>) {
      %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %63 = frisk.extract_frag %55[%61, %62] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
      %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
      %65 = frisk.insert_frag %63, %arg5[%61, %64] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %65 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_32 = gpu.thread_id x
    %58 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_32)
    %c4_33 = arith.constant 4 : index
    %59 = arith.cmpi ult, %58, %c4_33 : index
    scf.if %59 {
      affine.for %arg4 = 0 to 4 {
        %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %61 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%60)
        %63 = frisk.extract_frag %57[%61, %62] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        frisk.store_frag %63, %56[%61, %62] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
