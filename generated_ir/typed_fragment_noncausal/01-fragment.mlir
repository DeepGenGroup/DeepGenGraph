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
      %72 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %73 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_53 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
        frisk.fragment_yield %cst_53 : vector<1x1xf32>
      }
      %74 = frisk.pack_frag %73 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
      %75 = frisk.insert_frag %74, %arg5[%72, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %75 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = vector.broadcast %cst_2 : f32 to vector<2x1xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_4) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %72 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %73 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %76 = frisk.extract_frag %9[%72, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %77 = frisk.extract_frag %10[%72, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %78 = frisk.elementwise_frag "add" %76, %77 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %79 = frisk.unpack_frag %78 {tiled = true} : <vector<1x1xf32>, {}> -> vector<1x1xf32>
        frisk.fragment_yield %79 : vector<1x1xf32>
      }
      %74 = frisk.pack_frag %73 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
      %75 = frisk.insert_frag %74, %arg5[%72, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %75 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %12 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_7) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %72 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %73 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %76 = frisk.extract_frag %11[%72, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
        %77 = frisk.elementwise_frag "cast" %76 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
        %78 = frisk.unpack_frag %77 {tiled = true} : <vector<1x1xf32>, {}> -> vector<1x1xf32>
        frisk.fragment_yield %78 : vector<1x1xf32>
      }
      %74 = frisk.pack_frag %73 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
      %75 = frisk.insert_frag %74, %arg5[%72, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
      affine.yield %75 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %13 = frisk.from_threadTile %12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %14:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_5, %arg6 = %cst_6, %arg7 = %13, %arg8 = %cst_8, %arg9 = %cst_9, %arg10 = %cst_10) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %cst_53 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %72 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_53) -> (vector<128x128xf32>) {
        %alloc_106 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_107 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %141 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %141 to %alloc_106 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %142 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_106 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %142 to %alloc_107 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %143 = frisk.to_threadTile %alloc_106 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
        %144 = vector.broadcast %8 : f16 to vector<2x8xf16>
        %cst_108 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
        %145 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_108) -> (vector<2x8xf16>) {
          %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %154 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg13)
          %155 = frisk.fragment "compute" -> (vector<1x4xf16>) {
            %159 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%153)
            %160 = frisk.extract_frag %143[%154, %159] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
            %161 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%153)
            %162 = frisk.extract_frag %144[%154, %161] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
            %163 = frisk.elementwise_frag "mul" %160, %162 : (!frisk.fragment<vector<1x4xf16>, {}>, !frisk.fragment<vector<1x4xf16>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
            %164 = frisk.unpack_frag %163 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
            frisk.fragment_yield %164 : vector<1x4xf16>
          }
          %156 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%153)
          %157 = frisk.pack_frag %155 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
          %158 = frisk.insert_frag %157, %arg14[%154, %156] : <vector<1x4xf16>, {}> into vector<2x8xf16>
          affine.yield %158 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        %146 = frisk.from_threadTile %145 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf16> -> memref<128x32xf16, 3>
        frisk.wait_buffer_ready %alloc_107 : memref<32x128xf16, 3>
        %cst_109 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %cst_110 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %147 = frisk.pack_frag %cst_110 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        %148 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_109) -> (vector<2x32xf32>) {
          %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %154 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %155 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%153)
          %156 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %147) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %158 = frisk.fragment "load" -> (!frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
              %161 = frisk.load_frag %146[%154, %arg15] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
              frisk.fragment_yield %161 : !frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            }
            %159 = frisk.fragment "load" -> (!frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
              %161 = frisk.load_frag %alloc_107[%arg15, %155] {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 3, 0>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tile_type = vector<8x8xf16>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
              frisk.fragment_yield %161 : !frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            }
            %160 = frisk.fragment "compute" -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
              %161 = frisk.gemm_frag %158, %159, acc = %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
              frisk.fragment_yield %161 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            }
            affine.yield %160 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %157 = frisk.insert_frag %156, %arg14[%154, %155] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x32xf32>
          affine.yield %157 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %149 = frisk.to_threadTile %arg12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
        %cst_111 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %150 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_111) -> (vector<2x32xf32>) {
          %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %154 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %155 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %159 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%153)
            %160 = frisk.extract_frag %149[%154, %159] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
            %161 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%153)
            %162 = frisk.extract_frag %148[%154, %161] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
            %163 = frisk.elementwise_frag "add" %160, %162 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
            %164 = frisk.unpack_frag %163 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
            frisk.fragment_yield %164 : vector<1x4xf32>
          }
          %156 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%153)
          %157 = frisk.pack_frag %155 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
          %158 = frisk.insert_frag %157, %arg14[%154, %156] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %158 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %cst_112 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %151 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_112) -> (vector<2x32xf32>) {
          %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %154 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %155 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %159 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%153)
            %160 = frisk.extract_frag %150[%154, %159] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
            %161 = frisk.unpack_frag %160 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
            frisk.fragment_yield %161 : vector<1x4xf32>
          }
          %156 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%153)
          %157 = frisk.pack_frag %155 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
          %158 = frisk.insert_frag %157, %arg14[%154, %156] : <vector<1x4xf32>, {}> into vector<2x32xf32>
          affine.yield %158 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %152 = frisk.from_threadTile %151 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
        affine.yield %152 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %73 = frisk.to_threadTile %72 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_54 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %74 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_54) -> (vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %73[%142, %147] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %149 = frisk.elementwise_frag "cast" %148 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %150 = frisk.unpack_frag %149 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %150 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %146 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_55 = arith.constant 0xFF800000 : f32
      %cst_56 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %75 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_56) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %142 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %145 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_55) -> (f32) {
            %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
            %150 = frisk.fragment "compute" -> (f32) {
              %151 = vector.extract %74[%141, %149] : f32 from vector<2x32xf32>
              %152 = arith.maxnumf %arg14, %151 : f32
              %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
              %154 = vector.extract %74[%141, %153] : f32 from vector<2x32xf32>
              %155 = arith.maxnumf %152, %154 : f32
              %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
              %157 = vector.extract %74[%141, %156] : f32 from vector<2x32xf32>
              %158 = arith.maxnumf %155, %157 : f32
              %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
              %160 = vector.extract %74[%141, %159] : f32 from vector<2x32xf32>
              %161 = arith.maxnumf %158, %160 : f32
              frisk.fragment_yield %161 : f32
            }
            affine.yield %150 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_107 = arith.constant 64 : i32
          %c16_i32_108 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %145, %c16_i32_108, %c64_i32_107 : f32
          %146 = arith.maxnumf %145, %shuffleResult : f32
          %c64_i32_109 = arith.constant 64 : i32
          %c32_i32_110 = arith.constant 32 : i32
          %shuffleResult_111, %valid_112 = gpu.shuffle xor %146, %c32_i32_110, %c64_i32_109 : f32
          %147 = arith.maxnumf %146, %shuffleResult_111 : f32
          %148 = vector.insert %147, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %148 : vector<1x1xf32>
        }
        %143 = frisk.pack_frag %142 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %76 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_57 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %77 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_57) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %142 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %145 = frisk.extract_frag %75[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %146 = frisk.extract_frag %76[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %147 = frisk.elementwise_frag "max" %145, %146 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
          %148 = frisk.unpack_frag %147 {tiled = true} : <vector<1x1xf32>, {}> -> vector<1x1xf32>
          frisk.fragment_yield %148 : vector<1x1xf32>
        }
        %143 = frisk.pack_frag %142 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %78 = frisk.to_threadTile %72 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_58 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %79 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_58) -> (vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %147 = vector.extract %77[%142, %c0] : f32 from vector<2x1xf32>
          %148 = vector.insert %147, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %149 = vector.extract %77[%142, %c0_107] : f32 from vector<2x1xf32>
          %150 = vector.insert %149, %148 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %151 = vector.extract %77[%142, %c0_108] : f32 from vector<2x1xf32>
          %152 = vector.insert %151, %150 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %153 = vector.extract %77[%142, %c0_109] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %152 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %154 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %146 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %cst_59 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %80 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_59) -> (vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %78[%142, %147] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %149 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %150 = frisk.extract_frag %79[%142, %149] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %151 = frisk.elementwise_frag "sub" %148, %150 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %152 = frisk.unpack_frag %151 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %152 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %146 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %81 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_60 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %82 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_60) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %142 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %145 = frisk.extract_frag %81[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %146 = frisk.extract_frag %77[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %147 = frisk.elementwise_frag "sub" %145, %146 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
          %148 = frisk.unpack_frag %147 {tiled = true} : <vector<1x1xf32>, {}> -> vector<1x1xf32>
          frisk.fragment_yield %148 : vector<1x1xf32>
        }
        %143 = frisk.pack_frag %142 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_61 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %83 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_61) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %142 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %145 = frisk.extract_frag %82[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %146 = frisk.elementwise_frag "exp2" %145 : (!frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
          %147 = frisk.unpack_frag %146 {tiled = true} : <vector<1x1xf32>, {}> -> vector<1x1xf32>
          frisk.fragment_yield %147 : vector<1x1xf32>
        }
        %143 = frisk.pack_frag %142 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %84 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_62 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %85 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_62) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %142 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %145 = frisk.extract_frag %84[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %146 = frisk.extract_frag %83[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %147 = frisk.elementwise_frag "mul" %145, %146 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
          %148 = frisk.unpack_frag %147 {tiled = true} : <vector<1x1xf32>, {}> -> vector<1x1xf32>
          frisk.fragment_yield %148 : vector<1x1xf32>
        }
        %143 = frisk.pack_frag %142 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_63 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %86 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_63) -> (vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %80[%142, %147] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %149 = frisk.elementwise_frag "exp2" %148 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %150 = frisk.unpack_frag %149 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %150 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %146 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_64 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %87 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_64) -> (vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %86[%142, %147] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %149 = frisk.elementwise_frag "cast" %148 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %150 = frisk.unpack_frag %149 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %150 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x32xf32>
        affine.yield %146 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_65 = arith.constant 0.000000e+00 : f32
      %cst_66 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %88 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_66) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %142 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %145 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_65) -> (f32) {
            %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
            %150 = frisk.fragment "compute" -> (f32) {
              %151 = vector.extract %87[%141, %149] : f32 from vector<2x32xf32>
              %152 = arith.addf %arg14, %151 : f32
              %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
              %154 = vector.extract %87[%141, %153] : f32 from vector<2x32xf32>
              %155 = arith.addf %152, %154 : f32
              %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
              %157 = vector.extract %87[%141, %156] : f32 from vector<2x32xf32>
              %158 = arith.addf %155, %157 : f32
              %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
              %160 = vector.extract %87[%141, %159] : f32 from vector<2x32xf32>
              %161 = arith.addf %158, %160 : f32
              frisk.fragment_yield %161 : f32
            }
            affine.yield %150 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_107 = arith.constant 64 : i32
          %c16_i32_108 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %145, %c16_i32_108, %c64_i32_107 : f32
          %146 = arith.addf %145, %shuffleResult : f32
          %c64_i32_109 = arith.constant 64 : i32
          %c32_i32_110 = arith.constant 32 : i32
          %shuffleResult_111, %valid_112 = gpu.shuffle xor %146, %c32_i32_110, %c64_i32_109 : f32
          %147 = arith.addf %146, %shuffleResult_111 : f32
          %148 = vector.insert %147, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %148 : vector<1x1xf32>
        }
        %143 = frisk.pack_frag %142 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %cst_67 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %89 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_67) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %142 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %145 = frisk.extract_frag %88[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %146 = frisk.extract_frag %85[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %147 = frisk.elementwise_frag "add" %145, %146 : (!frisk.fragment<vector<1x1xf32>, {}>, !frisk.fragment<vector<1x1xf32>, {}>) -> !frisk.fragment<vector<1x1xf32>, {}>
          %148 = frisk.unpack_frag %147 {tiled = true} : <vector<1x1xf32>, {}> -> vector<1x1xf32>
          frisk.fragment_yield %148 : vector<1x1xf32>
        }
        %143 = frisk.pack_frag %142 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_68 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
      %90 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_68) -> (vector<2x32xf16>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %86[%142, %147] : vector<2x32xf32> -> <vector<1x4xf32>, {}>
          %149 = frisk.elementwise_frag "cast" %148 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
          %150 = frisk.unpack_frag %149 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
          frisk.fragment_yield %150 : vector<1x4xf16>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf16>, {}> into vector<2x32xf16>
        affine.yield %146 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %91 = frisk.from_threadTile %90 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x128xf16, 3>
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %92 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %92 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_69 = gpu.thread_id x
      %cst_70 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %93 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_70) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %147 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_69]
          %148 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_69]
          %149 = frisk.read_matrix_from_shm %alloc[%147, %148] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %150 = vector.bitcast %149 : vector<4xi32> to vector<8xf16>
          %cst_106 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %151 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_69]
          %152 = vector.extract %150[%151] : f16 from vector<8xf16>
          %153 = vector.insert %152, %cst_106 [0, 0] : f16 into vector<4x2xf16>
          %154 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_69]
          %155 = vector.extract %150[%154] : f16 from vector<8xf16>
          %156 = vector.insert %155, %153 [0, 1] : f16 into vector<4x2xf16>
          %157 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_69]
          %158 = vector.extract %150[%157] : f16 from vector<8xf16>
          %159 = vector.insert %158, %156 [1, 0] : f16 into vector<4x2xf16>
          %160 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_69]
          %161 = vector.extract %150[%160] : f16 from vector<8xf16>
          %162 = vector.insert %161, %159 [1, 1] : f16 into vector<4x2xf16>
          %163 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_69]
          %164 = vector.extract %150[%163] : f16 from vector<8xf16>
          %165 = vector.insert %164, %162 [2, 0] : f16 into vector<4x2xf16>
          %166 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_69]
          %167 = vector.extract %150[%166] : f16 from vector<8xf16>
          %168 = vector.insert %167, %165 [2, 1] : f16 into vector<4x2xf16>
          %169 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_69]
          %170 = vector.extract %150[%169] : f16 from vector<8xf16>
          %171 = vector.insert %170, %168 [3, 0] : f16 into vector<4x2xf16>
          %172 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_69]
          %173 = vector.extract %150[%172] : f16 from vector<8xf16>
          %174 = vector.insert %173, %171 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %174 : vector<4x2xf16>
        }
        %143 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %145 = frisk.pack_frag %142 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %146 = frisk.insert_frag %145, %arg12[%143, %144] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %146 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_71 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_72 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %94 = frisk.pack_frag %cst_72 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %95 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_71) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %144 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %94) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %146 = frisk.fragment "load" -> (!frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.load_frag %91[%142, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          %147 = frisk.fragment "load" -> (!frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.extract_frag %93[%arg13, %143] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          %148 = frisk.fragment "compute" -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.gemm_frag %146, %147, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          affine.yield %148 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %145 = frisk.insert_frag %144, %arg12[%142, %143] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %96 = frisk.from_threadTile %83 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> memref<128x1xf32>
      %97 = frisk.to_threadTile %arg5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %98 = frisk.to_threadTile %96 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_73 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %99 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_73) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %147 = vector.extract %98[%142, %c0] : f32 from vector<2x1xf32>
          %148 = vector.insert %147, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %149 = vector.extract %98[%142, %c0_107] : f32 from vector<2x1xf32>
          %150 = vector.insert %149, %148 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %151 = vector.extract %98[%142, %c0_108] : f32 from vector<2x1xf32>
          %152 = vector.insert %151, %150 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %153 = vector.extract %98[%142, %c0_109] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %152 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %154 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_74 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %100 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_74) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %97[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %150 = frisk.extract_frag %99[%142, %149] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %151 = frisk.elementwise_frag "mul" %148, %150 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %152 = frisk.unpack_frag %151 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %152 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_75 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %101 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_75) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %100[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %150 = frisk.extract_frag %95[%142, %149] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %151 = frisk.elementwise_frag "add" %148, %150 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %152 = frisk.unpack_frag %151 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %152 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_76 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %102 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %102 to %alloc_76 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_76 : memref<128x32xf16, 3>
      %thread_id_x_77 = gpu.thread_id x
      %cst_78 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %103 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_78) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %147 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_77]
          %148 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_77]
          %149 = frisk.read_matrix_from_shm %alloc_76[%147, %148] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %150 = vector.bitcast %149 : vector<4xi32> to vector<8xf16>
          %cst_106 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %151 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_77]
          %152 = vector.extract %150[%151] : f16 from vector<8xf16>
          %153 = vector.insert %152, %cst_106 [0, 0] : f16 into vector<4x2xf16>
          %154 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_77]
          %155 = vector.extract %150[%154] : f16 from vector<8xf16>
          %156 = vector.insert %155, %153 [0, 1] : f16 into vector<4x2xf16>
          %157 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_77]
          %158 = vector.extract %150[%157] : f16 from vector<8xf16>
          %159 = vector.insert %158, %156 [1, 0] : f16 into vector<4x2xf16>
          %160 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_77]
          %161 = vector.extract %150[%160] : f16 from vector<8xf16>
          %162 = vector.insert %161, %159 [1, 1] : f16 into vector<4x2xf16>
          %163 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_77]
          %164 = vector.extract %150[%163] : f16 from vector<8xf16>
          %165 = vector.insert %164, %162 [2, 0] : f16 into vector<4x2xf16>
          %166 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_77]
          %167 = vector.extract %150[%166] : f16 from vector<8xf16>
          %168 = vector.insert %167, %165 [2, 1] : f16 into vector<4x2xf16>
          %169 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_77]
          %170 = vector.extract %150[%169] : f16 from vector<8xf16>
          %171 = vector.insert %170, %168 [3, 0] : f16 into vector<4x2xf16>
          %172 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_77]
          %173 = vector.extract %150[%172] : f16 from vector<8xf16>
          %174 = vector.insert %173, %171 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %174 : vector<4x2xf16>
        }
        %143 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %145 = frisk.pack_frag %142 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %146 = frisk.insert_frag %145, %arg12[%143, %144] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %146 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_79 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_80 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %104 = frisk.pack_frag %cst_80 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %105 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_79) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %144 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %104) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %146 = frisk.fragment "load" -> (!frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.load_frag %91[%142, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          %147 = frisk.fragment "load" -> (!frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.extract_frag %103[%arg13, %143] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          %148 = frisk.fragment "compute" -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.gemm_frag %146, %147, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          affine.yield %148 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %145 = frisk.insert_frag %144, %arg12[%142, %143] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %106 = frisk.to_threadTile %arg8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %107 = frisk.to_threadTile %96 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_81 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %108 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_81) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %147 = vector.extract %107[%142, %c0] : f32 from vector<2x1xf32>
          %148 = vector.insert %147, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %149 = vector.extract %107[%142, %c0_107] : f32 from vector<2x1xf32>
          %150 = vector.insert %149, %148 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %151 = vector.extract %107[%142, %c0_108] : f32 from vector<2x1xf32>
          %152 = vector.insert %151, %150 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %153 = vector.extract %107[%142, %c0_109] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %152 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %154 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_82 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %109 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_82) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %106[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %150 = frisk.extract_frag %108[%142, %149] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %151 = frisk.elementwise_frag "mul" %148, %150 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %152 = frisk.unpack_frag %151 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %152 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_83 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %110 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_83) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %109[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %150 = frisk.extract_frag %105[%142, %149] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %151 = frisk.elementwise_frag "add" %148, %150 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %152 = frisk.unpack_frag %151 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %152 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_84 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %111 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %111 to %alloc_84 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_84 : memref<128x32xf16, 3>
      %thread_id_x_85 = gpu.thread_id x
      %cst_86 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %112 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_86) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %147 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_85]
          %148 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_85]
          %149 = frisk.read_matrix_from_shm %alloc_84[%147, %148] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %150 = vector.bitcast %149 : vector<4xi32> to vector<8xf16>
          %cst_106 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %151 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_85]
          %152 = vector.extract %150[%151] : f16 from vector<8xf16>
          %153 = vector.insert %152, %cst_106 [0, 0] : f16 into vector<4x2xf16>
          %154 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_85]
          %155 = vector.extract %150[%154] : f16 from vector<8xf16>
          %156 = vector.insert %155, %153 [0, 1] : f16 into vector<4x2xf16>
          %157 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_85]
          %158 = vector.extract %150[%157] : f16 from vector<8xf16>
          %159 = vector.insert %158, %156 [1, 0] : f16 into vector<4x2xf16>
          %160 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_85]
          %161 = vector.extract %150[%160] : f16 from vector<8xf16>
          %162 = vector.insert %161, %159 [1, 1] : f16 into vector<4x2xf16>
          %163 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_85]
          %164 = vector.extract %150[%163] : f16 from vector<8xf16>
          %165 = vector.insert %164, %162 [2, 0] : f16 into vector<4x2xf16>
          %166 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_85]
          %167 = vector.extract %150[%166] : f16 from vector<8xf16>
          %168 = vector.insert %167, %165 [2, 1] : f16 into vector<4x2xf16>
          %169 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_85]
          %170 = vector.extract %150[%169] : f16 from vector<8xf16>
          %171 = vector.insert %170, %168 [3, 0] : f16 into vector<4x2xf16>
          %172 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_85]
          %173 = vector.extract %150[%172] : f16 from vector<8xf16>
          %174 = vector.insert %173, %171 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %174 : vector<4x2xf16>
        }
        %143 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %145 = frisk.pack_frag %142 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %146 = frisk.insert_frag %145, %arg12[%143, %144] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %146 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_87 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_88 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %113 = frisk.pack_frag %cst_88 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %114 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_87) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %144 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %113) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %146 = frisk.fragment "load" -> (!frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.load_frag %91[%142, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          %147 = frisk.fragment "load" -> (!frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.extract_frag %112[%arg13, %143] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          %148 = frisk.fragment "compute" -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.gemm_frag %146, %147, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          affine.yield %148 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %145 = frisk.insert_frag %144, %arg12[%142, %143] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %115 = frisk.to_threadTile %arg9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %116 = frisk.to_threadTile %96 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_89 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %117 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_89) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %147 = vector.extract %116[%142, %c0] : f32 from vector<2x1xf32>
          %148 = vector.insert %147, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %149 = vector.extract %116[%142, %c0_107] : f32 from vector<2x1xf32>
          %150 = vector.insert %149, %148 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %151 = vector.extract %116[%142, %c0_108] : f32 from vector<2x1xf32>
          %152 = vector.insert %151, %150 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %153 = vector.extract %116[%142, %c0_109] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %152 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %154 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_90 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %118 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_90) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %115[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %150 = frisk.extract_frag %117[%142, %149] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %151 = frisk.elementwise_frag "mul" %148, %150 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %152 = frisk.unpack_frag %151 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %152 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_91 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %119 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_91) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %118[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %150 = frisk.extract_frag %114[%142, %149] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %151 = frisk.elementwise_frag "add" %148, %150 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %152 = frisk.unpack_frag %151 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %152 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_92 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %120 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %120 to %alloc_92 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_92 : memref<128x32xf16, 3>
      %thread_id_x_93 = gpu.thread_id x
      %cst_94 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %121 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_94) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %142 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %147 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_93]
          %148 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_93]
          %149 = frisk.read_matrix_from_shm %alloc_92[%147, %148] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %150 = vector.bitcast %149 : vector<4xi32> to vector<8xf16>
          %cst_106 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %151 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_93]
          %152 = vector.extract %150[%151] : f16 from vector<8xf16>
          %153 = vector.insert %152, %cst_106 [0, 0] : f16 into vector<4x2xf16>
          %154 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_93]
          %155 = vector.extract %150[%154] : f16 from vector<8xf16>
          %156 = vector.insert %155, %153 [0, 1] : f16 into vector<4x2xf16>
          %157 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_93]
          %158 = vector.extract %150[%157] : f16 from vector<8xf16>
          %159 = vector.insert %158, %156 [1, 0] : f16 into vector<4x2xf16>
          %160 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_93]
          %161 = vector.extract %150[%160] : f16 from vector<8xf16>
          %162 = vector.insert %161, %159 [1, 1] : f16 into vector<4x2xf16>
          %163 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_93]
          %164 = vector.extract %150[%163] : f16 from vector<8xf16>
          %165 = vector.insert %164, %162 [2, 0] : f16 into vector<4x2xf16>
          %166 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_93]
          %167 = vector.extract %150[%166] : f16 from vector<8xf16>
          %168 = vector.insert %167, %165 [2, 1] : f16 into vector<4x2xf16>
          %169 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_93]
          %170 = vector.extract %150[%169] : f16 from vector<8xf16>
          %171 = vector.insert %170, %168 [3, 0] : f16 into vector<4x2xf16>
          %172 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_93]
          %173 = vector.extract %150[%172] : f16 from vector<8xf16>
          %174 = vector.insert %173, %171 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %174 : vector<4x2xf16>
        }
        %143 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c0)
        %145 = frisk.pack_frag %142 : vector<4x2xf16> -> <vector<4x2xf16>, {}>
        %146 = frisk.insert_frag %145, %arg12[%143, %144] : <vector<4x2xf16>, {}> into vector<32x2xf16>
        affine.yield %146 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_95 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_96 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %122 = frisk.pack_frag %cst_96 : vector<1x4xf32> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
      %123 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_95) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %144 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %122) -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
          %146 = frisk.fragment "load" -> (!frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.load_frag %91[%142, %arg13] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, calc_axes = array<i64: 0, 3>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, reuse_axes = array<i64: 1, 1>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tile_type = vector<2x32xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          %147 = frisk.fragment "load" -> (!frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.extract_frag %121[%arg13, %143] : vector<32x2xf16> -> <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          %148 = frisk.fragment "compute" -> (!frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>) {
            %149 = frisk.gemm_frag %146, %147, acc = %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3"} : <vector<1x4xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "A", thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<4x1xf16>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "B", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>, <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> -> <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
            frisk.fragment_yield %149 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
          }
          affine.yield %148 : !frisk.fragment<vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %145 = frisk.insert_frag %144, %arg12[%142, %143] : <vector<1x4xf32>, {instruction = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", role = "C", thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64}> into vector<2x8xf32>
        affine.yield %145 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %124 = frisk.to_threadTile %arg10 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %125 = frisk.to_threadTile %96 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_97 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %126 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_97) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %147 = vector.extract %125[%142, %c0] : f32 from vector<2x1xf32>
          %148 = vector.insert %147, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %149 = vector.extract %125[%142, %c0_107] : f32 from vector<2x1xf32>
          %150 = vector.insert %149, %148 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %151 = vector.extract %125[%142, %c0_108] : f32 from vector<2x1xf32>
          %152 = vector.insert %151, %150 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %153 = vector.extract %125[%142, %c0_109] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %152 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %154 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_98 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %127 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_98) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %124[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %150 = frisk.extract_frag %126[%142, %149] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %151 = frisk.elementwise_frag "mul" %148, %150 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %152 = frisk.unpack_frag %151 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %152 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_99 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %128 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_99) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %127[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %150 = frisk.extract_frag %123[%142, %149] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %151 = frisk.elementwise_frag "add" %148, %150 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
          %152 = frisk.unpack_frag %151 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %152 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_100 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %129 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_100) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %101[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = frisk.unpack_frag %148 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %149 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %130 = frisk.from_threadTile %129 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_101 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %131 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_101) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %142 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %145 = frisk.extract_frag %89[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %146 = frisk.unpack_frag %145 {tiled = true} : <vector<1x1xf32>, {}> -> vector<1x1xf32>
          frisk.fragment_yield %146 : vector<1x1xf32>
        }
        %143 = frisk.pack_frag %142 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %132 = frisk.from_threadTile %131 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_102 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %133 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_102) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %141 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %142 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %145 = frisk.extract_frag %77[%141, %c0] : vector<2x1xf32> -> <vector<1x1xf32>, {}>
          %146 = frisk.unpack_frag %145 {tiled = true} : <vector<1x1xf32>, {}> -> vector<1x1xf32>
          frisk.fragment_yield %146 : vector<1x1xf32>
        }
        %143 = frisk.pack_frag %142 : vector<1x1xf32> -> <vector<1x1xf32>, {}>
        %144 = frisk.insert_frag %143, %arg12[%141, %c0] : <vector<1x1xf32>, {}> into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %134 = frisk.from_threadTile %133 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_103 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %135 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_103) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %110[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = frisk.unpack_frag %148 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %149 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %136 = frisk.from_threadTile %135 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_104 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %137 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_104) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %119[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = frisk.unpack_frag %148 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %149 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %138 = frisk.from_threadTile %137 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_105 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %139 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_105) -> (vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %143 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %147 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
          %148 = frisk.extract_frag %128[%142, %147] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
          %149 = frisk.unpack_frag %148 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
          frisk.fragment_yield %149 : vector<1x4xf32>
        }
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%141)
        %145 = frisk.pack_frag %143 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
        %146 = frisk.insert_frag %145, %arg12[%142, %144] : <vector<1x4xf32>, {}> into vector<2x8xf32>
        affine.yield %146 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %140 = frisk.from_threadTile %139 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      affine.yield %130, %132, %134, %136, %138, %140 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %15 = frisk.to_threadTile %14#5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x = gpu.thread_id x
    %16 = arith.index_cast %thread_id_x : index to i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %17 = arith.andi %16, %c16_i32 : i32
    %18 = arith.cmpi ne, %17, %c0_i32 : i32
    %c32_i32 = arith.constant 32 : i32
    %19 = arith.andi %16, %c32_i32 : i32
    %20 = arith.cmpi ne, %19, %c0_i32 : i32
    %cst_11 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %21 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_11) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %15[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = frisk.unpack_frag %79 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
        %81 = vector.extract %80[0, 0] : f32 from vector<1x4xf32>
        %82 = vector.extract %80[0, 1] : f32 from vector<1x4xf32>
        %83 = vector.extract %80[0, 2] : f32 from vector<1x4xf32>
        %84 = vector.extract %80[0, 3] : f32 from vector<1x4xf32>
        %85 = arith.select %18, %81, %82 : f32
        %shuffleResult, %valid = gpu.shuffle xor %85, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %86 = arith.select %18, %shuffleResult, %81 : f32
        %87 = arith.select %18, %82, %shuffleResult : f32
        %88 = arith.select %18, %83, %84 : f32
        %shuffleResult_53, %valid_54 = gpu.shuffle xor %88, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %89 = arith.select %18, %shuffleResult_53, %83 : f32
        %90 = arith.select %18, %84, %shuffleResult_53 : f32
        %91 = arith.select %20, %86, %89 : f32
        %shuffleResult_55, %valid_56 = gpu.shuffle xor %91, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %92 = arith.select %20, %shuffleResult_55, %86 : f32
        %93 = arith.select %20, %89, %shuffleResult_55 : f32
        %94 = arith.select %20, %87, %90 : f32
        %shuffleResult_57, %valid_58 = gpu.shuffle xor %94, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %95 = arith.select %20, %shuffleResult_57, %87 : f32
        %96 = arith.select %20, %90, %shuffleResult_57 : f32
        %97 = vector.insert %92, %80 [0, 0] : f32 into vector<1x4xf32>
        %98 = vector.insert %95, %97 [0, 1] : f32 into vector<1x4xf32>
        %99 = vector.insert %93, %98 [0, 2] : f32 into vector<1x4xf32>
        %100 = vector.insert %96, %99 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %100 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %22 = frisk.to_threadTile %14#4 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x_12 = gpu.thread_id x
    %23 = arith.index_cast %thread_id_x_12 : index to i32
    %c0_i32_13 = arith.constant 0 : i32
    %c64_i32_14 = arith.constant 64 : i32
    %c16_i32_15 = arith.constant 16 : i32
    %24 = arith.andi %23, %c16_i32_15 : i32
    %25 = arith.cmpi ne, %24, %c0_i32_13 : i32
    %c32_i32_16 = arith.constant 32 : i32
    %26 = arith.andi %23, %c32_i32_16 : i32
    %27 = arith.cmpi ne, %26, %c0_i32_13 : i32
    %cst_17 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %28 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_17) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %22[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = frisk.unpack_frag %79 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
        %81 = vector.extract %80[0, 0] : f32 from vector<1x4xf32>
        %82 = vector.extract %80[0, 1] : f32 from vector<1x4xf32>
        %83 = vector.extract %80[0, 2] : f32 from vector<1x4xf32>
        %84 = vector.extract %80[0, 3] : f32 from vector<1x4xf32>
        %85 = arith.select %25, %81, %82 : f32
        %shuffleResult, %valid = gpu.shuffle xor %85, %c16_i32_15, %c64_i32_14 {frisk.layout_role = "wmmaC"} : f32
        %86 = arith.select %25, %shuffleResult, %81 : f32
        %87 = arith.select %25, %82, %shuffleResult : f32
        %88 = arith.select %25, %83, %84 : f32
        %shuffleResult_53, %valid_54 = gpu.shuffle xor %88, %c16_i32_15, %c64_i32_14 {frisk.layout_role = "wmmaC"} : f32
        %89 = arith.select %25, %shuffleResult_53, %83 : f32
        %90 = arith.select %25, %84, %shuffleResult_53 : f32
        %91 = arith.select %27, %86, %89 : f32
        %shuffleResult_55, %valid_56 = gpu.shuffle xor %91, %c32_i32_16, %c64_i32_14 {frisk.layout_role = "wmmaC"} : f32
        %92 = arith.select %27, %shuffleResult_55, %86 : f32
        %93 = arith.select %27, %89, %shuffleResult_55 : f32
        %94 = arith.select %27, %87, %90 : f32
        %shuffleResult_57, %valid_58 = gpu.shuffle xor %94, %c32_i32_16, %c64_i32_14 {frisk.layout_role = "wmmaC"} : f32
        %95 = arith.select %27, %shuffleResult_57, %87 : f32
        %96 = arith.select %27, %90, %shuffleResult_57 : f32
        %97 = vector.insert %92, %80 [0, 0] : f32 into vector<1x4xf32>
        %98 = vector.insert %95, %97 [0, 1] : f32 into vector<1x4xf32>
        %99 = vector.insert %93, %98 [0, 2] : f32 into vector<1x4xf32>
        %100 = vector.insert %96, %99 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %100 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %29 = frisk.to_threadTile %14#3 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x_18 = gpu.thread_id x
    %30 = arith.index_cast %thread_id_x_18 : index to i32
    %c0_i32_19 = arith.constant 0 : i32
    %c64_i32_20 = arith.constant 64 : i32
    %c16_i32_21 = arith.constant 16 : i32
    %31 = arith.andi %30, %c16_i32_21 : i32
    %32 = arith.cmpi ne, %31, %c0_i32_19 : i32
    %c32_i32_22 = arith.constant 32 : i32
    %33 = arith.andi %30, %c32_i32_22 : i32
    %34 = arith.cmpi ne, %33, %c0_i32_19 : i32
    %cst_23 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %35 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_23) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %29[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = frisk.unpack_frag %79 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
        %81 = vector.extract %80[0, 0] : f32 from vector<1x4xf32>
        %82 = vector.extract %80[0, 1] : f32 from vector<1x4xf32>
        %83 = vector.extract %80[0, 2] : f32 from vector<1x4xf32>
        %84 = vector.extract %80[0, 3] : f32 from vector<1x4xf32>
        %85 = arith.select %32, %81, %82 : f32
        %shuffleResult, %valid = gpu.shuffle xor %85, %c16_i32_21, %c64_i32_20 {frisk.layout_role = "wmmaC"} : f32
        %86 = arith.select %32, %shuffleResult, %81 : f32
        %87 = arith.select %32, %82, %shuffleResult : f32
        %88 = arith.select %32, %83, %84 : f32
        %shuffleResult_53, %valid_54 = gpu.shuffle xor %88, %c16_i32_21, %c64_i32_20 {frisk.layout_role = "wmmaC"} : f32
        %89 = arith.select %32, %shuffleResult_53, %83 : f32
        %90 = arith.select %32, %84, %shuffleResult_53 : f32
        %91 = arith.select %34, %86, %89 : f32
        %shuffleResult_55, %valid_56 = gpu.shuffle xor %91, %c32_i32_22, %c64_i32_20 {frisk.layout_role = "wmmaC"} : f32
        %92 = arith.select %34, %shuffleResult_55, %86 : f32
        %93 = arith.select %34, %89, %shuffleResult_55 : f32
        %94 = arith.select %34, %87, %90 : f32
        %shuffleResult_57, %valid_58 = gpu.shuffle xor %94, %c32_i32_22, %c64_i32_20 {frisk.layout_role = "wmmaC"} : f32
        %95 = arith.select %34, %shuffleResult_57, %87 : f32
        %96 = arith.select %34, %90, %shuffleResult_57 : f32
        %97 = vector.insert %92, %80 [0, 0] : f32 into vector<1x4xf32>
        %98 = vector.insert %95, %97 [0, 1] : f32 into vector<1x4xf32>
        %99 = vector.insert %93, %98 [0, 2] : f32 into vector<1x4xf32>
        %100 = vector.insert %96, %99 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %100 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %36 = frisk.to_threadTile %14#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x_24 = gpu.thread_id x
    %37 = arith.index_cast %thread_id_x_24 : index to i32
    %c0_i32_25 = arith.constant 0 : i32
    %c64_i32_26 = arith.constant 64 : i32
    %c16_i32_27 = arith.constant 16 : i32
    %38 = arith.andi %37, %c16_i32_27 : i32
    %39 = arith.cmpi ne, %38, %c0_i32_25 : i32
    %c32_i32_28 = arith.constant 32 : i32
    %40 = arith.andi %37, %c32_i32_28 : i32
    %41 = arith.cmpi ne, %40, %c0_i32_25 : i32
    %cst_29 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %42 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_29) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %36[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = frisk.unpack_frag %79 : <vector<1x4xf32>, {}> -> vector<1x4xf32>
        %81 = vector.extract %80[0, 0] : f32 from vector<1x4xf32>
        %82 = vector.extract %80[0, 1] : f32 from vector<1x4xf32>
        %83 = vector.extract %80[0, 2] : f32 from vector<1x4xf32>
        %84 = vector.extract %80[0, 3] : f32 from vector<1x4xf32>
        %85 = arith.select %39, %81, %82 : f32
        %shuffleResult, %valid = gpu.shuffle xor %85, %c16_i32_27, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
        %86 = arith.select %39, %shuffleResult, %81 : f32
        %87 = arith.select %39, %82, %shuffleResult : f32
        %88 = arith.select %39, %83, %84 : f32
        %shuffleResult_53, %valid_54 = gpu.shuffle xor %88, %c16_i32_27, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
        %89 = arith.select %39, %shuffleResult_53, %83 : f32
        %90 = arith.select %39, %84, %shuffleResult_53 : f32
        %91 = arith.select %41, %86, %89 : f32
        %shuffleResult_55, %valid_56 = gpu.shuffle xor %91, %c32_i32_28, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
        %92 = arith.select %41, %shuffleResult_55, %86 : f32
        %93 = arith.select %41, %89, %shuffleResult_55 : f32
        %94 = arith.select %41, %87, %90 : f32
        %shuffleResult_57, %valid_58 = gpu.shuffle xor %94, %c32_i32_28, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
        %95 = arith.select %41, %shuffleResult_57, %87 : f32
        %96 = arith.select %41, %90, %shuffleResult_57 : f32
        %97 = vector.insert %92, %80 [0, 0] : f32 into vector<1x4xf32>
        %98 = vector.insert %95, %97 [0, 1] : f32 into vector<1x4xf32>
        %99 = vector.insert %93, %98 [0, 2] : f32 into vector<1x4xf32>
        %100 = vector.insert %96, %99 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %100 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %43 = frisk.to_threadTile %14#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %44 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_30) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %78 = vector.extract %43[%73, %c0] : f32 from vector<2x1xf32>
        %79 = vector.insert %78, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %c0_54 = arith.constant 0 : index
        %80 = vector.extract %43[%73, %c0_54] : f32 from vector<2x1xf32>
        %81 = vector.insert %80, %79 [0, 1] : f32 into vector<1x4xf32>
        %c0_55 = arith.constant 0 : index
        %82 = vector.extract %43[%73, %c0_55] : f32 from vector<2x1xf32>
        %83 = vector.insert %82, %81 [0, 2] : f32 into vector<1x4xf32>
        %c0_56 = arith.constant 0 : index
        %84 = vector.extract %43[%73, %c0_56] : f32 from vector<2x1xf32>
        %85 = vector.insert %84, %83 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %85 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_31 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %45 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_31) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %42[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %81 = frisk.extract_frag %44[%73, %80] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %82 = frisk.elementwise_frag "div" %79, %81 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %83 = frisk.unpack_frag %82 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
        frisk.fragment_yield %83 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_32 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %46 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_32) -> (vector<2x8xf16>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %45[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = frisk.elementwise_frag "cast" %79 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %81 = frisk.unpack_frag %80 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
        frisk.fragment_yield %81 : vector<1x4xf16>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %77 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %47 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_33 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %48 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_33) -> (vector<2x8xf16>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %46[%73, %78] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        %80 = frisk.unpack_frag %79 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
        frisk.fragment_yield %80 : vector<1x4xf16>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %77 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_34 = gpu.thread_id x
    %49 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_34)
    %c4 = arith.constant 4 : index
    %50 = arith.cmpi ult, %49, %c4 : index
    scf.if %50 {
      affine.for %arg4 = 0 to 4 {
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %74 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
          %75 = frisk.extract_frag %48[%73, %74] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          frisk.store_frag %75, %47[%73, %74] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_35 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %51 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_35) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %78 = vector.extract %43[%73, %c0] : f32 from vector<2x1xf32>
        %79 = vector.insert %78, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %c0_54 = arith.constant 0 : index
        %80 = vector.extract %43[%73, %c0_54] : f32 from vector<2x1xf32>
        %81 = vector.insert %80, %79 [0, 1] : f32 into vector<1x4xf32>
        %c0_55 = arith.constant 0 : index
        %82 = vector.extract %43[%73, %c0_55] : f32 from vector<2x1xf32>
        %83 = vector.insert %82, %81 [0, 2] : f32 into vector<1x4xf32>
        %c0_56 = arith.constant 0 : index
        %84 = vector.extract %43[%73, %c0_56] : f32 from vector<2x1xf32>
        %85 = vector.insert %84, %83 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %85 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_36 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %52 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_36) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %35[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %81 = frisk.extract_frag %51[%73, %80] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %82 = frisk.elementwise_frag "div" %79, %81 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %83 = frisk.unpack_frag %82 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
        frisk.fragment_yield %83 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_37 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %53 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_37) -> (vector<2x8xf16>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %52[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = frisk.elementwise_frag "cast" %79 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %81 = frisk.unpack_frag %80 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
        frisk.fragment_yield %81 : vector<1x4xf16>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %77 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %54 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_38 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %55 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_38) -> (vector<2x8xf16>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %53[%73, %78] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        %80 = frisk.unpack_frag %79 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
        frisk.fragment_yield %80 : vector<1x4xf16>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %77 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_39 = gpu.thread_id x
    %56 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_39)
    %c4_40 = arith.constant 4 : index
    %57 = arith.cmpi ult, %56, %c4_40 : index
    scf.if %57 {
      affine.for %arg4 = 0 to 4 {
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %74 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
          %75 = frisk.extract_frag %55[%73, %74] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          frisk.store_frag %75, %54[%73, %74] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_41 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %58 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_41) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %78 = vector.extract %43[%73, %c0] : f32 from vector<2x1xf32>
        %79 = vector.insert %78, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %c0_54 = arith.constant 0 : index
        %80 = vector.extract %43[%73, %c0_54] : f32 from vector<2x1xf32>
        %81 = vector.insert %80, %79 [0, 1] : f32 into vector<1x4xf32>
        %c0_55 = arith.constant 0 : index
        %82 = vector.extract %43[%73, %c0_55] : f32 from vector<2x1xf32>
        %83 = vector.insert %82, %81 [0, 2] : f32 into vector<1x4xf32>
        %c0_56 = arith.constant 0 : index
        %84 = vector.extract %43[%73, %c0_56] : f32 from vector<2x1xf32>
        %85 = vector.insert %84, %83 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %85 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_42 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %59 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_42) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %28[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %81 = frisk.extract_frag %58[%73, %80] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %82 = frisk.elementwise_frag "div" %79, %81 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %83 = frisk.unpack_frag %82 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
        frisk.fragment_yield %83 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_43 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %60 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_43) -> (vector<2x8xf16>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %59[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = frisk.elementwise_frag "cast" %79 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %81 = frisk.unpack_frag %80 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
        frisk.fragment_yield %81 : vector<1x4xf16>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %77 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %61 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_44 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %62 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_44) -> (vector<2x8xf16>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %60[%73, %78] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        %80 = frisk.unpack_frag %79 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
        frisk.fragment_yield %80 : vector<1x4xf16>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %77 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_45 = gpu.thread_id x
    %63 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_45)
    %c4_46 = arith.constant 4 : index
    %64 = arith.cmpi ult, %63, %c4_46 : index
    scf.if %64 {
      affine.for %arg4 = 0 to 4 {
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %74 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
          %75 = frisk.extract_frag %62[%73, %74] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          frisk.store_frag %75, %61[%73, %74] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_47 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %65 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_47) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %78 = vector.extract %43[%73, %c0] : f32 from vector<2x1xf32>
        %79 = vector.insert %78, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %c0_54 = arith.constant 0 : index
        %80 = vector.extract %43[%73, %c0_54] : f32 from vector<2x1xf32>
        %81 = vector.insert %80, %79 [0, 1] : f32 into vector<1x4xf32>
        %c0_55 = arith.constant 0 : index
        %82 = vector.extract %43[%73, %c0_55] : f32 from vector<2x1xf32>
        %83 = vector.insert %82, %81 [0, 2] : f32 into vector<1x4xf32>
        %c0_56 = arith.constant 0 : index
        %84 = vector.extract %43[%73, %c0_56] : f32 from vector<2x1xf32>
        %85 = vector.insert %84, %83 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %85 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_48 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %66 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_48) -> (vector<2x8xf32>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %21[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %81 = frisk.extract_frag %65[%73, %80] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %82 = frisk.elementwise_frag "div" %79, %81 : (!frisk.fragment<vector<1x4xf32>, {}>, !frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf32>, {}>
        %83 = frisk.unpack_frag %82 {tiled = true} : <vector<1x4xf32>, {}> -> vector<1x4xf32>
        frisk.fragment_yield %83 : vector<1x4xf32>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf32> -> <vector<1x4xf32>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf32>, {}> into vector<2x8xf32>
      affine.yield %77 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_49 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %67 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_49) -> (vector<2x8xf16>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %66[%73, %78] : vector<2x8xf32> -> <vector<1x4xf32>, {}>
        %80 = frisk.elementwise_frag "cast" %79 : (!frisk.fragment<vector<1x4xf32>, {}>) -> !frisk.fragment<vector<1x4xf16>, {}>
        %81 = frisk.unpack_frag %80 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
        frisk.fragment_yield %81 : vector<1x4xf16>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %77 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %68 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_50 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %69 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_50) -> (vector<2x8xf16>) {
      %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %74 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %78 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
        %79 = frisk.extract_frag %67[%73, %78] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
        %80 = frisk.unpack_frag %79 {tiled = true} : <vector<1x4xf16>, {}> -> vector<1x4xf16>
        frisk.fragment_yield %80 : vector<1x4xf16>
      }
      %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
      %76 = frisk.pack_frag %74 : vector<1x4xf16> -> <vector<1x4xf16>, {}>
      %77 = frisk.insert_frag %76, %arg5[%73, %75] : <vector<1x4xf16>, {}> into vector<2x8xf16>
      affine.yield %77 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_51 = gpu.thread_id x
    %70 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_51)
    %c4_52 = arith.constant 4 : index
    %71 = arith.cmpi ult, %70, %c4_52 : index
    scf.if %71 {
      affine.for %arg4 = 0 to 4 {
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %74 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%72)
          %75 = frisk.extract_frag %69[%73, %74] : vector<2x8xf16> -> <vector<1x4xf16>, {}>
          frisk.store_frag %75, %68[%73, %74] {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tile_type = vector<2x8xf16>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : <vector<1x4xf16>, {}>, memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
