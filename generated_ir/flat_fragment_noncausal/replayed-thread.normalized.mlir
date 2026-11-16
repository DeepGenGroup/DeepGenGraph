#map = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>
#map3 = affine_map<(d0) -> (d0 floordiv 2)>
#map4 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>
#map5 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>
#map6 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>
#map7 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>
#map8 = affine_map<(d0) -> (d0 floordiv 8)>
#map9 = affine_map<(d0) -> (d0 * 4)>
#map10 = affine_map<(d0) -> (d0 * 4 + 1)>
#map11 = affine_map<(d0) -> (d0 * 4 + 2)>
#map12 = affine_map<(d0) -> (d0 * 4 + 3)>
#map13 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>
#map14 = affine_map<(d0) -> (d0 mod 8)>
#map15 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>
#map16 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>
#map17 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>
#map18 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>
#map19 = affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>
#map20 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
#map21 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>
#map22 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>
#map23 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>
#map24 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>
#map25 = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>
#map26 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>
#map27 = affine_map<(d0) -> (d0 mod 2)>
#map28 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>
#map29 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>
#map30 = affine_map<()[s0] -> (s0 floordiv 64)>
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
    %cst = arith.constant dense<0xFF800000> : vector<1x1xf32>
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
    %cst_10 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_11 = arith.constant 0xFF800000 : f32
    %cst_12 = arith.constant 0.127531052 : f32
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
    %8 = arith.truncf %cst_12 : f32 to f16
    %9 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_10) -> (vector<2x1xf32>) {
      %47 = vector.insert %cst_3, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %47 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_10) -> (vector<2x1xf32>) {
      %47 = vector.extract %9[%arg4, 0] : f32 from vector<2x1xf32>
      %48 = vector.broadcast %47 : f32 to vector<1x1xf32>
      %49 = arith.addf %48, %cst {origin = "frisk.add", tiled = true} : vector<1x1xf32>
      %50 = vector.extract %49[0, 0] : f32 from vector<1x1xf32>
      %51 = vector.insert %50, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %51 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_10) -> (vector<2x1xf32>) {
      %47 = vector.extract %10[%arg4, 0] : f32 from vector<2x1xf32>
      %48 = vector.insert %47, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %48 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %12 = frisk.from_threadTile %11 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
    %13:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_9, %arg6 = %cst_8, %arg7 = %12, %arg8 = %cst_9, %arg9 = %cst_9, %arg10 = %cst_9) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %47 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_7) -> (vector<128x128xf32>) {
        %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %104 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %104 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %105 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_17 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %105 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %106 = frisk.to_threadTile %alloc_17 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map2, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
        %107 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_6) -> (vector<2x8xf16>) {
          %113 = affine.apply #map3(%arg13)
          %114 = affine.apply #map4(%arg13)
          %115 = vector.extract %106[%113, %114] : f16 from vector<2x8xf16>
          %116 = affine.apply #map5(%arg13)
          %117 = vector.extract %106[%113, %116] : f16 from vector<2x8xf16>
          %118 = affine.apply #map6(%arg13)
          %119 = vector.extract %106[%113, %118] : f16 from vector<2x8xf16>
          %120 = affine.apply #map7(%arg13)
          %121 = vector.extract %106[%113, %120] : f16 from vector<2x8xf16>
          %122 = vector.from_elements %115, %117, %119, %121 : vector<1x4xf16>
          %123 = vector.broadcast %8 : f16 to vector<1x4xf16>
          %124 = arith.mulf %122, %123 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
          %125 = vector.extract %124[0, 0] : f16 from vector<1x4xf16>
          %126 = vector.insert %125, %arg14 [%113, %114] : f16 into vector<2x8xf16>
          %127 = vector.extract %124[0, 1] : f16 from vector<1x4xf16>
          %128 = vector.insert %127, %126 [%113, %116] : f16 into vector<2x8xf16>
          %129 = vector.extract %124[0, 2] : f16 from vector<1x4xf16>
          %130 = vector.insert %129, %128 [%113, %118] : f16 into vector<2x8xf16>
          %131 = vector.extract %124[0, 3] : f16 from vector<1x4xf16>
          %132 = vector.insert %131, %130 [%113, %120] : f16 into vector<2x8xf16>
          affine.yield %132 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        frisk.wait_buffer_ready %alloc_18 : memref<32x128xf16, 3>
        %108 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5) -> (vector<2x32xf32>) {
          %113 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %cst_4) -> (vector<1x4xf32>) {
            %127 = affine.apply #map8(%arg13)
            %128 = affine.apply #map9(%arg15)
            %129 = vector.extract %107[%127, %128] : f16 from vector<2x8xf16>
            %130 = affine.apply #map10(%arg15)
            %131 = vector.extract %107[%127, %130] : f16 from vector<2x8xf16>
            %132 = affine.apply #map11(%arg15)
            %133 = vector.extract %107[%127, %132] : f16 from vector<2x8xf16>
            %134 = affine.apply #map12(%arg15)
            %135 = vector.extract %107[%127, %134] : f16 from vector<2x8xf16>
            %136 = vector.from_elements %129, %131, %133, %135 : vector<1x4xf16>
            %137 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map13, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
            %138 = affine.apply #map14(%arg13)
            %139 = vector.extract %137[%128, %138] : f16 from vector<8x8xf16>
            %140 = vector.extract %137[%130, %138] : f16 from vector<8x8xf16>
            %141 = vector.extract %137[%132, %138] : f16 from vector<8x8xf16>
            %142 = vector.extract %137[%134, %138] : f16 from vector<8x8xf16>
            %143 = vector.from_elements %139, %140, %141, %142 : vector<4x1xf16>
            %144 = frisk.warp_mma_rr %136, %143, %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            affine.yield %144 : vector<1x4xf32>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %114 = affine.apply #map8(%arg13)
          %115 = affine.apply #map15(%arg13)
          %116 = vector.extract %113[0, 0] : f32 from vector<1x4xf32>
          %117 = vector.insert %116, %arg14 [%114, %115] : f32 into vector<2x32xf32>
          %118 = affine.apply #map16(%arg13)
          %119 = vector.extract %113[0, 1] : f32 from vector<1x4xf32>
          %120 = vector.insert %119, %117 [%114, %118] : f32 into vector<2x32xf32>
          %121 = affine.apply #map17(%arg13)
          %122 = vector.extract %113[0, 2] : f32 from vector<1x4xf32>
          %123 = vector.insert %122, %120 [%114, %121] : f32 into vector<2x32xf32>
          %124 = affine.apply #map18(%arg13)
          %125 = vector.extract %113[0, 3] : f32 from vector<1x4xf32>
          %126 = vector.insert %125, %123 [%114, %124] : f32 into vector<2x32xf32>
          affine.yield %126 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %109 = frisk.to_threadTile %arg12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
        %110 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5) -> (vector<2x32xf32>) {
          %113 = affine.apply #map8(%arg13)
          %114 = affine.apply #map15(%arg13)
          %115 = vector.extract %109[%113, %114] : f32 from vector<2x32xf32>
          %116 = affine.apply #map16(%arg13)
          %117 = vector.extract %109[%113, %116] : f32 from vector<2x32xf32>
          %118 = affine.apply #map17(%arg13)
          %119 = vector.extract %109[%113, %118] : f32 from vector<2x32xf32>
          %120 = affine.apply #map18(%arg13)
          %121 = vector.extract %109[%113, %120] : f32 from vector<2x32xf32>
          %122 = vector.from_elements %115, %117, %119, %121 : vector<1x4xf32>
          %123 = vector.extract %108[%113, %114] : f32 from vector<2x32xf32>
          %124 = vector.extract %108[%113, %116] : f32 from vector<2x32xf32>
          %125 = vector.extract %108[%113, %118] : f32 from vector<2x32xf32>
          %126 = vector.extract %108[%113, %120] : f32 from vector<2x32xf32>
          %127 = vector.from_elements %123, %124, %125, %126 : vector<1x4xf32>
          %128 = arith.addf %122, %127 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          %129 = vector.extract %128[0, 0] : f32 from vector<1x4xf32>
          %130 = vector.insert %129, %arg14 [%113, %114] : f32 into vector<2x32xf32>
          %131 = vector.extract %128[0, 1] : f32 from vector<1x4xf32>
          %132 = vector.insert %131, %130 [%113, %116] : f32 into vector<2x32xf32>
          %133 = vector.extract %128[0, 2] : f32 from vector<1x4xf32>
          %134 = vector.insert %133, %132 [%113, %118] : f32 into vector<2x32xf32>
          %135 = vector.extract %128[0, 3] : f32 from vector<1x4xf32>
          %136 = vector.insert %135, %134 [%113, %120] : f32 into vector<2x32xf32>
          affine.yield %136 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %111 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5) -> (vector<2x32xf32>) {
          %113 = affine.apply #map8(%arg13)
          %114 = affine.apply #map15(%arg13)
          %115 = vector.extract %110[%113, %114] : f32 from vector<2x32xf32>
          %116 = affine.apply #map16(%arg13)
          %117 = vector.extract %110[%113, %116] : f32 from vector<2x32xf32>
          %118 = affine.apply #map17(%arg13)
          %119 = vector.extract %110[%113, %118] : f32 from vector<2x32xf32>
          %120 = affine.apply #map18(%arg13)
          %121 = vector.extract %110[%113, %120] : f32 from vector<2x32xf32>
          %122 = vector.insert %115, %arg14 [%113, %114] : f32 into vector<2x32xf32>
          %123 = vector.insert %117, %122 [%113, %116] : f32 into vector<2x32xf32>
          %124 = vector.insert %119, %123 [%113, %118] : f32 into vector<2x32xf32>
          %125 = vector.insert %121, %124 [%113, %120] : f32 into vector<2x32xf32>
          affine.yield %125 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %112 = frisk.from_threadTile %111 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
        affine.yield %112 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %48 = frisk.to_threadTile %47 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %49 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = affine.apply #map8(%arg11)
        %105 = affine.apply #map15(%arg11)
        %106 = vector.extract %48[%104, %105] : f32 from vector<2x32xf32>
        %107 = affine.apply #map16(%arg11)
        %108 = vector.extract %48[%104, %107] : f32 from vector<2x32xf32>
        %109 = affine.apply #map17(%arg11)
        %110 = vector.extract %48[%104, %109] : f32 from vector<2x32xf32>
        %111 = affine.apply #map18(%arg11)
        %112 = vector.extract %48[%104, %111] : f32 from vector<2x32xf32>
        %113 = vector.insert %106, %arg12 [%104, %105] : f32 into vector<2x32xf32>
        %114 = vector.insert %108, %113 [%104, %107] : f32 into vector<2x32xf32>
        %115 = vector.insert %110, %114 [%104, %109] : f32 into vector<2x32xf32>
        %116 = vector.insert %112, %115 [%104, %111] : f32 into vector<2x32xf32>
        affine.yield %116 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %50 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_10) -> (vector<2x1xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_11) -> (f32) {
          %108 = affine.apply #map9(%arg13)
          %109 = vector.extract %49[%arg11, %108] : f32 from vector<2x32xf32>
          %110 = arith.maxnumf %arg14, %109 : f32
          %111 = affine.apply #map10(%arg13)
          %112 = vector.extract %49[%arg11, %111] : f32 from vector<2x32xf32>
          %113 = arith.maxnumf %110, %112 : f32
          %114 = affine.apply #map11(%arg13)
          %115 = vector.extract %49[%arg11, %114] : f32 from vector<2x32xf32>
          %116 = arith.maxnumf %113, %115 : f32
          %117 = affine.apply #map12(%arg13)
          %118 = vector.extract %49[%arg11, %117] : f32 from vector<2x32xf32>
          %119 = arith.maxnumf %116, %118 : f32
          affine.yield %119 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %104, %c16_i32, %c64_i32 : f32
        %105 = arith.maxnumf %104, %shuffleResult : f32
        %shuffleResult_17, %valid_18 = gpu.shuffle xor %105, %c32_i32, %c64_i32 : f32
        %106 = arith.maxnumf %105, %shuffleResult_17 : f32
        %107 = vector.insert %106, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %107 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %51 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %52 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_10) -> (vector<2x1xf32>) {
        %104 = vector.extract %50[%arg11, 0] : f32 from vector<2x1xf32>
        %105 = vector.broadcast %104 : f32 to vector<1x1xf32>
        %106 = vector.extract %51[%arg11, 0] : f32 from vector<2x1xf32>
        %107 = vector.broadcast %106 : f32 to vector<1x1xf32>
        %108 = arith.maxnumf %105, %107 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %109 = vector.extract %108[0, 0] : f32 from vector<1x1xf32>
        %110 = vector.insert %109, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %110 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %53 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = affine.apply #map8(%arg11)
        %105 = vector.extract %52[%104, 0] : f32 from vector<2x1xf32>
        %106 = affine.apply #map15(%arg11)
        %107 = vector.insert %105, %arg12 [%104, %106] : f32 into vector<2x32xf32>
        %108 = affine.apply #map16(%arg11)
        %109 = vector.insert %105, %107 [%104, %108] : f32 into vector<2x32xf32>
        %110 = affine.apply #map17(%arg11)
        %111 = vector.insert %105, %109 [%104, %110] : f32 into vector<2x32xf32>
        %112 = affine.apply #map18(%arg11)
        %113 = vector.insert %105, %111 [%104, %112] : f32 into vector<2x32xf32>
        affine.yield %113 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %54 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = affine.apply #map8(%arg11)
        %105 = affine.apply #map15(%arg11)
        %106 = vector.extract %48[%104, %105] : f32 from vector<2x32xf32>
        %107 = affine.apply #map16(%arg11)
        %108 = vector.extract %48[%104, %107] : f32 from vector<2x32xf32>
        %109 = affine.apply #map17(%arg11)
        %110 = vector.extract %48[%104, %109] : f32 from vector<2x32xf32>
        %111 = affine.apply #map18(%arg11)
        %112 = vector.extract %48[%104, %111] : f32 from vector<2x32xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = vector.extract %53[%104, %105] : f32 from vector<2x32xf32>
        %115 = vector.extract %53[%104, %107] : f32 from vector<2x32xf32>
        %116 = vector.extract %53[%104, %109] : f32 from vector<2x32xf32>
        %117 = vector.extract %53[%104, %111] : f32 from vector<2x32xf32>
        %118 = vector.from_elements %114, %115, %116, %117 : vector<1x4xf32>
        %119 = arith.subf %113, %118 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %120 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %121 = vector.insert %120, %arg12 [%104, %105] : f32 into vector<2x32xf32>
        %122 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %121 [%104, %107] : f32 into vector<2x32xf32>
        %124 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %123 [%104, %109] : f32 into vector<2x32xf32>
        %126 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %127 = vector.insert %126, %125 [%104, %111] : f32 into vector<2x32xf32>
        affine.yield %127 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %55 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_10) -> (vector<2x1xf32>) {
        %104 = vector.extract %51[%arg11, 0] : f32 from vector<2x1xf32>
        %105 = vector.broadcast %104 : f32 to vector<1x1xf32>
        %106 = vector.extract %52[%arg11, 0] : f32 from vector<2x1xf32>
        %107 = vector.broadcast %106 : f32 to vector<1x1xf32>
        %108 = arith.subf %105, %107 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %109 = vector.extract %108[0, 0] : f32 from vector<1x1xf32>
        %110 = vector.insert %109, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %110 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %56 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_10) -> (vector<2x1xf32>) {
        %104 = vector.extract %55[%arg11, 0] : f32 from vector<2x1xf32>
        %105 = vector.broadcast %104 : f32 to vector<1x1xf32>
        %106 = math.exp2 %105 {tiled = true} : vector<1x1xf32>
        %107 = vector.extract %106[0, 0] : f32 from vector<1x1xf32>
        %108 = vector.insert %107, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %108 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %57 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %58 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_10) -> (vector<2x1xf32>) {
        %104 = vector.extract %57[%arg11, 0] : f32 from vector<2x1xf32>
        %105 = vector.broadcast %104 : f32 to vector<1x1xf32>
        %106 = vector.extract %56[%arg11, 0] : f32 from vector<2x1xf32>
        %107 = vector.broadcast %106 : f32 to vector<1x1xf32>
        %108 = arith.mulf %105, %107 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %109 = vector.extract %108[0, 0] : f32 from vector<1x1xf32>
        %110 = vector.insert %109, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %110 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %59 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = affine.apply #map8(%arg11)
        %105 = affine.apply #map15(%arg11)
        %106 = vector.extract %54[%104, %105] : f32 from vector<2x32xf32>
        %107 = affine.apply #map16(%arg11)
        %108 = vector.extract %54[%104, %107] : f32 from vector<2x32xf32>
        %109 = affine.apply #map17(%arg11)
        %110 = vector.extract %54[%104, %109] : f32 from vector<2x32xf32>
        %111 = affine.apply #map18(%arg11)
        %112 = vector.extract %54[%104, %111] : f32 from vector<2x32xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = math.exp2 %113 {tiled = true} : vector<1x4xf32>
        %115 = vector.extract %114[0, 0] : f32 from vector<1x4xf32>
        %116 = vector.insert %115, %arg12 [%104, %105] : f32 into vector<2x32xf32>
        %117 = vector.extract %114[0, 1] : f32 from vector<1x4xf32>
        %118 = vector.insert %117, %116 [%104, %107] : f32 into vector<2x32xf32>
        %119 = vector.extract %114[0, 2] : f32 from vector<1x4xf32>
        %120 = vector.insert %119, %118 [%104, %109] : f32 into vector<2x32xf32>
        %121 = vector.extract %114[0, 3] : f32 from vector<1x4xf32>
        %122 = vector.insert %121, %120 [%104, %111] : f32 into vector<2x32xf32>
        affine.yield %122 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %60 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = affine.apply #map8(%arg11)
        %105 = affine.apply #map15(%arg11)
        %106 = vector.extract %59[%104, %105] : f32 from vector<2x32xf32>
        %107 = affine.apply #map16(%arg11)
        %108 = vector.extract %59[%104, %107] : f32 from vector<2x32xf32>
        %109 = affine.apply #map17(%arg11)
        %110 = vector.extract %59[%104, %109] : f32 from vector<2x32xf32>
        %111 = affine.apply #map18(%arg11)
        %112 = vector.extract %59[%104, %111] : f32 from vector<2x32xf32>
        %113 = vector.insert %106, %arg12 [%104, %105] : f32 into vector<2x32xf32>
        %114 = vector.insert %108, %113 [%104, %107] : f32 into vector<2x32xf32>
        %115 = vector.insert %110, %114 [%104, %109] : f32 into vector<2x32xf32>
        %116 = vector.insert %112, %115 [%104, %111] : f32 into vector<2x32xf32>
        affine.yield %116 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %61 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_10) -> (vector<2x1xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_3) -> (f32) {
          %108 = affine.apply #map9(%arg13)
          %109 = vector.extract %60[%arg11, %108] : f32 from vector<2x32xf32>
          %110 = arith.addf %arg14, %109 : f32
          %111 = affine.apply #map10(%arg13)
          %112 = vector.extract %60[%arg11, %111] : f32 from vector<2x32xf32>
          %113 = arith.addf %110, %112 : f32
          %114 = affine.apply #map11(%arg13)
          %115 = vector.extract %60[%arg11, %114] : f32 from vector<2x32xf32>
          %116 = arith.addf %113, %115 : f32
          %117 = affine.apply #map12(%arg13)
          %118 = vector.extract %60[%arg11, %117] : f32 from vector<2x32xf32>
          %119 = arith.addf %116, %118 : f32
          affine.yield %119 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %104, %c16_i32, %c64_i32 : f32
        %105 = arith.addf %104, %shuffleResult : f32
        %shuffleResult_17, %valid_18 = gpu.shuffle xor %105, %c32_i32, %c64_i32 : f32
        %106 = arith.addf %105, %shuffleResult_17 : f32
        %107 = vector.insert %106, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %107 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %62 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_10) -> (vector<2x1xf32>) {
        %104 = vector.extract %61[%arg11, 0] : f32 from vector<2x1xf32>
        %105 = vector.broadcast %104 : f32 to vector<1x1xf32>
        %106 = vector.extract %58[%arg11, 0] : f32 from vector<2x1xf32>
        %107 = vector.broadcast %106 : f32 to vector<1x1xf32>
        %108 = arith.addf %105, %107 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %109 = vector.extract %108[0, 0] : f32 from vector<1x1xf32>
        %110 = vector.insert %109, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %110 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %63 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %104 = affine.apply #map8(%arg11)
        %105 = affine.apply #map15(%arg11)
        %106 = vector.extract %59[%104, %105] : f32 from vector<2x32xf32>
        %107 = affine.apply #map16(%arg11)
        %108 = vector.extract %59[%104, %107] : f32 from vector<2x32xf32>
        %109 = affine.apply #map17(%arg11)
        %110 = vector.extract %59[%104, %109] : f32 from vector<2x32xf32>
        %111 = affine.apply #map18(%arg11)
        %112 = vector.extract %59[%104, %111] : f32 from vector<2x32xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = arith.truncf %113 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %115 = vector.extract %114[0, 0] : f16 from vector<1x4xf16>
        %116 = vector.insert %115, %arg12 [%104, %105] : f16 into vector<2x32xf16>
        %117 = vector.extract %114[0, 1] : f16 from vector<1x4xf16>
        %118 = vector.insert %117, %116 [%104, %107] : f16 into vector<2x32xf16>
        %119 = vector.extract %114[0, 2] : f16 from vector<1x4xf16>
        %120 = vector.insert %119, %118 [%104, %109] : f16 into vector<2x32xf16>
        %121 = vector.extract %114[0, 3] : f16 from vector<1x4xf16>
        %122 = vector.insert %121, %120 [%104, %111] : f16 into vector<2x32xf16>
        affine.yield %122 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %64 = frisk.from_threadTile %63 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x128xf16, 3>
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %65 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %65 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_13 = gpu.thread_id x
      %66 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %104 = affine.apply #map19(%arg11)[%thread_id_x_13]
        %105 = affine.apply #map20()[%thread_id_x_13]
        %106 = frisk.read_matrix_from_shm %alloc[%104, %105] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %107 = vector.bitcast %106 : vector<4xi32> to vector<8xf16>
        %108 = vector.extract %107[0] : f16 from vector<8xf16>
        %109 = affine.apply #map21()[%thread_id_x_13]
        %110 = vector.extract %107[%109] : f16 from vector<8xf16>
        %111 = vector.extract %107[1] : f16 from vector<8xf16>
        %112 = affine.apply #map22()[%thread_id_x_13]
        %113 = vector.extract %107[%112] : f16 from vector<8xf16>
        %114 = vector.extract %107[2] : f16 from vector<8xf16>
        %115 = affine.apply #map23()[%thread_id_x_13]
        %116 = vector.extract %107[%115] : f16 from vector<8xf16>
        %117 = vector.extract %107[3] : f16 from vector<8xf16>
        %118 = affine.apply #map24()[%thread_id_x_13]
        %119 = vector.extract %107[%118] : f16 from vector<8xf16>
        %120 = affine.apply #map9(%arg11)
        %121 = vector.insert %108, %arg12 [%120, 0] : f16 into vector<32x2xf16>
        %122 = vector.insert %110, %121 [%120, 1] : f16 into vector<32x2xf16>
        %123 = affine.apply #map10(%arg11)
        %124 = vector.insert %111, %122 [%123, 0] : f16 into vector<32x2xf16>
        %125 = vector.insert %113, %124 [%123, 1] : f16 into vector<32x2xf16>
        %126 = affine.apply #map11(%arg11)
        %127 = vector.insert %114, %125 [%126, 0] : f16 into vector<32x2xf16>
        %128 = vector.insert %116, %127 [%126, 1] : f16 into vector<32x2xf16>
        %129 = affine.apply #map12(%arg11)
        %130 = vector.insert %117, %128 [%129, 0] : f16 into vector<32x2xf16>
        %131 = vector.insert %119, %130 [%129, 1] : f16 into vector<32x2xf16>
        affine.yield %131 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %67 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_4) -> (vector<1x4xf32>) {
          %118 = frisk.to_threadTile %64 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map25, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map26, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
          %119 = affine.apply #map3(%arg11)
          %120 = affine.apply #map9(%arg13)
          %121 = vector.extract %118[%119, %120] : f16 from vector<2x32xf16>
          %122 = affine.apply #map10(%arg13)
          %123 = vector.extract %118[%119, %122] : f16 from vector<2x32xf16>
          %124 = affine.apply #map11(%arg13)
          %125 = vector.extract %118[%119, %124] : f16 from vector<2x32xf16>
          %126 = affine.apply #map12(%arg13)
          %127 = vector.extract %118[%119, %126] : f16 from vector<2x32xf16>
          %128 = vector.from_elements %121, %123, %125, %127 : vector<1x4xf16>
          %129 = affine.apply #map27(%arg11)
          %130 = vector.extract %66[%120, %129] : f16 from vector<32x2xf16>
          %131 = vector.extract %66[%122, %129] : f16 from vector<32x2xf16>
          %132 = vector.extract %66[%124, %129] : f16 from vector<32x2xf16>
          %133 = vector.extract %66[%126, %129] : f16 from vector<32x2xf16>
          %134 = vector.from_elements %130, %131, %132, %133 : vector<4x1xf16>
          %135 = frisk.warp_mma_rr %128, %134, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %135 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %105 = affine.apply #map3(%arg11)
        %106 = affine.apply #map4(%arg11)
        %107 = vector.extract %104[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%105, %106] : f32 into vector<2x8xf32>
        %109 = affine.apply #map5(%arg11)
        %110 = vector.extract %104[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%105, %109] : f32 into vector<2x8xf32>
        %112 = affine.apply #map6(%arg11)
        %113 = vector.extract %104[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%105, %112] : f32 into vector<2x8xf32>
        %115 = affine.apply #map7(%arg11)
        %116 = vector.extract %104[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%105, %115] : f32 into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %68 = frisk.from_threadTile %56 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = #map28, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> memref<128x1xf32>
      %69 = frisk.to_threadTile %arg5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %70 = frisk.to_threadTile %68 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %71 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = vector.extract %70[%104, 0] : f32 from vector<2x1xf32>
        %106 = affine.apply #map4(%arg11)
        %107 = vector.insert %105, %arg12 [%104, %106] : f32 into vector<2x8xf32>
        %108 = affine.apply #map5(%arg11)
        %109 = vector.insert %105, %107 [%104, %108] : f32 into vector<2x8xf32>
        %110 = affine.apply #map6(%arg11)
        %111 = vector.insert %105, %109 [%104, %110] : f32 into vector<2x8xf32>
        %112 = affine.apply #map7(%arg11)
        %113 = vector.insert %105, %111 [%104, %112] : f32 into vector<2x8xf32>
        affine.yield %113 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %72 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %69[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %69[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %69[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %69[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = vector.extract %71[%104, %105] : f32 from vector<2x8xf32>
        %115 = vector.extract %71[%104, %107] : f32 from vector<2x8xf32>
        %116 = vector.extract %71[%104, %109] : f32 from vector<2x8xf32>
        %117 = vector.extract %71[%104, %111] : f32 from vector<2x8xf32>
        %118 = vector.from_elements %114, %115, %116, %117 : vector<1x4xf32>
        %119 = arith.mulf %113, %118 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %120 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %121 = vector.insert %120, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %122 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %121 [%104, %107] : f32 into vector<2x8xf32>
        %124 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %123 [%104, %109] : f32 into vector<2x8xf32>
        %126 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %127 = vector.insert %126, %125 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %73 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %72[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %72[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %72[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %72[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = vector.extract %67[%104, %105] : f32 from vector<2x8xf32>
        %115 = vector.extract %67[%104, %107] : f32 from vector<2x8xf32>
        %116 = vector.extract %67[%104, %109] : f32 from vector<2x8xf32>
        %117 = vector.extract %67[%104, %111] : f32 from vector<2x8xf32>
        %118 = vector.from_elements %114, %115, %116, %117 : vector<1x4xf32>
        %119 = arith.addf %113, %118 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %120 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %121 = vector.insert %120, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %122 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %121 [%104, %107] : f32 into vector<2x8xf32>
        %124 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %123 [%104, %109] : f32 into vector<2x8xf32>
        %126 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %127 = vector.insert %126, %125 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_14 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %74 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %74 to %alloc_14 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_14 : memref<128x32xf16, 3>
      %75 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %104 = affine.apply #map19(%arg11)[%thread_id_x_13]
        %105 = affine.apply #map20()[%thread_id_x_13]
        %106 = frisk.read_matrix_from_shm %alloc_14[%104, %105] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %107 = vector.bitcast %106 : vector<4xi32> to vector<8xf16>
        %108 = vector.extract %107[0] : f16 from vector<8xf16>
        %109 = affine.apply #map21()[%thread_id_x_13]
        %110 = vector.extract %107[%109] : f16 from vector<8xf16>
        %111 = vector.extract %107[1] : f16 from vector<8xf16>
        %112 = affine.apply #map22()[%thread_id_x_13]
        %113 = vector.extract %107[%112] : f16 from vector<8xf16>
        %114 = vector.extract %107[2] : f16 from vector<8xf16>
        %115 = affine.apply #map23()[%thread_id_x_13]
        %116 = vector.extract %107[%115] : f16 from vector<8xf16>
        %117 = vector.extract %107[3] : f16 from vector<8xf16>
        %118 = affine.apply #map24()[%thread_id_x_13]
        %119 = vector.extract %107[%118] : f16 from vector<8xf16>
        %120 = affine.apply #map9(%arg11)
        %121 = vector.insert %108, %arg12 [%120, 0] : f16 into vector<32x2xf16>
        %122 = vector.insert %110, %121 [%120, 1] : f16 into vector<32x2xf16>
        %123 = affine.apply #map10(%arg11)
        %124 = vector.insert %111, %122 [%123, 0] : f16 into vector<32x2xf16>
        %125 = vector.insert %113, %124 [%123, 1] : f16 into vector<32x2xf16>
        %126 = affine.apply #map11(%arg11)
        %127 = vector.insert %114, %125 [%126, 0] : f16 into vector<32x2xf16>
        %128 = vector.insert %116, %127 [%126, 1] : f16 into vector<32x2xf16>
        %129 = affine.apply #map12(%arg11)
        %130 = vector.insert %117, %128 [%129, 0] : f16 into vector<32x2xf16>
        %131 = vector.insert %119, %130 [%129, 1] : f16 into vector<32x2xf16>
        affine.yield %131 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %76 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_4) -> (vector<1x4xf32>) {
          %118 = frisk.to_threadTile %64 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map25, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map26, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
          %119 = affine.apply #map3(%arg11)
          %120 = affine.apply #map9(%arg13)
          %121 = vector.extract %118[%119, %120] : f16 from vector<2x32xf16>
          %122 = affine.apply #map10(%arg13)
          %123 = vector.extract %118[%119, %122] : f16 from vector<2x32xf16>
          %124 = affine.apply #map11(%arg13)
          %125 = vector.extract %118[%119, %124] : f16 from vector<2x32xf16>
          %126 = affine.apply #map12(%arg13)
          %127 = vector.extract %118[%119, %126] : f16 from vector<2x32xf16>
          %128 = vector.from_elements %121, %123, %125, %127 : vector<1x4xf16>
          %129 = affine.apply #map27(%arg11)
          %130 = vector.extract %75[%120, %129] : f16 from vector<32x2xf16>
          %131 = vector.extract %75[%122, %129] : f16 from vector<32x2xf16>
          %132 = vector.extract %75[%124, %129] : f16 from vector<32x2xf16>
          %133 = vector.extract %75[%126, %129] : f16 from vector<32x2xf16>
          %134 = vector.from_elements %130, %131, %132, %133 : vector<4x1xf16>
          %135 = frisk.warp_mma_rr %128, %134, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %135 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %105 = affine.apply #map3(%arg11)
        %106 = affine.apply #map4(%arg11)
        %107 = vector.extract %104[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%105, %106] : f32 into vector<2x8xf32>
        %109 = affine.apply #map5(%arg11)
        %110 = vector.extract %104[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%105, %109] : f32 into vector<2x8xf32>
        %112 = affine.apply #map6(%arg11)
        %113 = vector.extract %104[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%105, %112] : f32 into vector<2x8xf32>
        %115 = affine.apply #map7(%arg11)
        %116 = vector.extract %104[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%105, %115] : f32 into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %77 = frisk.to_threadTile %arg8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %78 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %77[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %77[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %77[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %77[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = vector.extract %71[%104, %105] : f32 from vector<2x8xf32>
        %115 = vector.extract %71[%104, %107] : f32 from vector<2x8xf32>
        %116 = vector.extract %71[%104, %109] : f32 from vector<2x8xf32>
        %117 = vector.extract %71[%104, %111] : f32 from vector<2x8xf32>
        %118 = vector.from_elements %114, %115, %116, %117 : vector<1x4xf32>
        %119 = arith.mulf %113, %118 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %120 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %121 = vector.insert %120, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %122 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %121 [%104, %107] : f32 into vector<2x8xf32>
        %124 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %123 [%104, %109] : f32 into vector<2x8xf32>
        %126 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %127 = vector.insert %126, %125 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %79 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %78[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %78[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %78[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %78[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = vector.extract %76[%104, %105] : f32 from vector<2x8xf32>
        %115 = vector.extract %76[%104, %107] : f32 from vector<2x8xf32>
        %116 = vector.extract %76[%104, %109] : f32 from vector<2x8xf32>
        %117 = vector.extract %76[%104, %111] : f32 from vector<2x8xf32>
        %118 = vector.from_elements %114, %115, %116, %117 : vector<1x4xf32>
        %119 = arith.addf %113, %118 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %120 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %121 = vector.insert %120, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %122 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %121 [%104, %107] : f32 into vector<2x8xf32>
        %124 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %123 [%104, %109] : f32 into vector<2x8xf32>
        %126 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %127 = vector.insert %126, %125 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %80 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %80 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<128x32xf16, 3>
      %81 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %104 = affine.apply #map19(%arg11)[%thread_id_x_13]
        %105 = affine.apply #map20()[%thread_id_x_13]
        %106 = frisk.read_matrix_from_shm %alloc_15[%104, %105] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %107 = vector.bitcast %106 : vector<4xi32> to vector<8xf16>
        %108 = vector.extract %107[0] : f16 from vector<8xf16>
        %109 = affine.apply #map21()[%thread_id_x_13]
        %110 = vector.extract %107[%109] : f16 from vector<8xf16>
        %111 = vector.extract %107[1] : f16 from vector<8xf16>
        %112 = affine.apply #map22()[%thread_id_x_13]
        %113 = vector.extract %107[%112] : f16 from vector<8xf16>
        %114 = vector.extract %107[2] : f16 from vector<8xf16>
        %115 = affine.apply #map23()[%thread_id_x_13]
        %116 = vector.extract %107[%115] : f16 from vector<8xf16>
        %117 = vector.extract %107[3] : f16 from vector<8xf16>
        %118 = affine.apply #map24()[%thread_id_x_13]
        %119 = vector.extract %107[%118] : f16 from vector<8xf16>
        %120 = affine.apply #map9(%arg11)
        %121 = vector.insert %108, %arg12 [%120, 0] : f16 into vector<32x2xf16>
        %122 = vector.insert %110, %121 [%120, 1] : f16 into vector<32x2xf16>
        %123 = affine.apply #map10(%arg11)
        %124 = vector.insert %111, %122 [%123, 0] : f16 into vector<32x2xf16>
        %125 = vector.insert %113, %124 [%123, 1] : f16 into vector<32x2xf16>
        %126 = affine.apply #map11(%arg11)
        %127 = vector.insert %114, %125 [%126, 0] : f16 into vector<32x2xf16>
        %128 = vector.insert %116, %127 [%126, 1] : f16 into vector<32x2xf16>
        %129 = affine.apply #map12(%arg11)
        %130 = vector.insert %117, %128 [%129, 0] : f16 into vector<32x2xf16>
        %131 = vector.insert %119, %130 [%129, 1] : f16 into vector<32x2xf16>
        affine.yield %131 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %82 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_4) -> (vector<1x4xf32>) {
          %118 = frisk.to_threadTile %64 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map25, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map26, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
          %119 = affine.apply #map3(%arg11)
          %120 = affine.apply #map9(%arg13)
          %121 = vector.extract %118[%119, %120] : f16 from vector<2x32xf16>
          %122 = affine.apply #map10(%arg13)
          %123 = vector.extract %118[%119, %122] : f16 from vector<2x32xf16>
          %124 = affine.apply #map11(%arg13)
          %125 = vector.extract %118[%119, %124] : f16 from vector<2x32xf16>
          %126 = affine.apply #map12(%arg13)
          %127 = vector.extract %118[%119, %126] : f16 from vector<2x32xf16>
          %128 = vector.from_elements %121, %123, %125, %127 : vector<1x4xf16>
          %129 = affine.apply #map27(%arg11)
          %130 = vector.extract %81[%120, %129] : f16 from vector<32x2xf16>
          %131 = vector.extract %81[%122, %129] : f16 from vector<32x2xf16>
          %132 = vector.extract %81[%124, %129] : f16 from vector<32x2xf16>
          %133 = vector.extract %81[%126, %129] : f16 from vector<32x2xf16>
          %134 = vector.from_elements %130, %131, %132, %133 : vector<4x1xf16>
          %135 = frisk.warp_mma_rr %128, %134, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %135 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %105 = affine.apply #map3(%arg11)
        %106 = affine.apply #map4(%arg11)
        %107 = vector.extract %104[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%105, %106] : f32 into vector<2x8xf32>
        %109 = affine.apply #map5(%arg11)
        %110 = vector.extract %104[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%105, %109] : f32 into vector<2x8xf32>
        %112 = affine.apply #map6(%arg11)
        %113 = vector.extract %104[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%105, %112] : f32 into vector<2x8xf32>
        %115 = affine.apply #map7(%arg11)
        %116 = vector.extract %104[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%105, %115] : f32 into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %83 = frisk.to_threadTile %arg9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %84 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %83[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %83[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %83[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %83[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = vector.extract %71[%104, %105] : f32 from vector<2x8xf32>
        %115 = vector.extract %71[%104, %107] : f32 from vector<2x8xf32>
        %116 = vector.extract %71[%104, %109] : f32 from vector<2x8xf32>
        %117 = vector.extract %71[%104, %111] : f32 from vector<2x8xf32>
        %118 = vector.from_elements %114, %115, %116, %117 : vector<1x4xf32>
        %119 = arith.mulf %113, %118 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %120 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %121 = vector.insert %120, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %122 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %121 [%104, %107] : f32 into vector<2x8xf32>
        %124 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %123 [%104, %109] : f32 into vector<2x8xf32>
        %126 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %127 = vector.insert %126, %125 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %85 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %84[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %84[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %84[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %84[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = vector.extract %82[%104, %105] : f32 from vector<2x8xf32>
        %115 = vector.extract %82[%104, %107] : f32 from vector<2x8xf32>
        %116 = vector.extract %82[%104, %109] : f32 from vector<2x8xf32>
        %117 = vector.extract %82[%104, %111] : f32 from vector<2x8xf32>
        %118 = vector.from_elements %114, %115, %116, %117 : vector<1x4xf32>
        %119 = arith.addf %113, %118 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %120 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %121 = vector.insert %120, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %122 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %121 [%104, %107] : f32 into vector<2x8xf32>
        %124 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %123 [%104, %109] : f32 into vector<2x8xf32>
        %126 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %127 = vector.insert %126, %125 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %86 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %86 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_16 : memref<128x32xf16, 3>
      %87 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %104 = affine.apply #map19(%arg11)[%thread_id_x_13]
        %105 = affine.apply #map20()[%thread_id_x_13]
        %106 = frisk.read_matrix_from_shm %alloc_16[%104, %105] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %107 = vector.bitcast %106 : vector<4xi32> to vector<8xf16>
        %108 = vector.extract %107[0] : f16 from vector<8xf16>
        %109 = affine.apply #map21()[%thread_id_x_13]
        %110 = vector.extract %107[%109] : f16 from vector<8xf16>
        %111 = vector.extract %107[1] : f16 from vector<8xf16>
        %112 = affine.apply #map22()[%thread_id_x_13]
        %113 = vector.extract %107[%112] : f16 from vector<8xf16>
        %114 = vector.extract %107[2] : f16 from vector<8xf16>
        %115 = affine.apply #map23()[%thread_id_x_13]
        %116 = vector.extract %107[%115] : f16 from vector<8xf16>
        %117 = vector.extract %107[3] : f16 from vector<8xf16>
        %118 = affine.apply #map24()[%thread_id_x_13]
        %119 = vector.extract %107[%118] : f16 from vector<8xf16>
        %120 = affine.apply #map9(%arg11)
        %121 = vector.insert %108, %arg12 [%120, 0] : f16 into vector<32x2xf16>
        %122 = vector.insert %110, %121 [%120, 1] : f16 into vector<32x2xf16>
        %123 = affine.apply #map10(%arg11)
        %124 = vector.insert %111, %122 [%123, 0] : f16 into vector<32x2xf16>
        %125 = vector.insert %113, %124 [%123, 1] : f16 into vector<32x2xf16>
        %126 = affine.apply #map11(%arg11)
        %127 = vector.insert %114, %125 [%126, 0] : f16 into vector<32x2xf16>
        %128 = vector.insert %116, %127 [%126, 1] : f16 into vector<32x2xf16>
        %129 = affine.apply #map12(%arg11)
        %130 = vector.insert %117, %128 [%129, 0] : f16 into vector<32x2xf16>
        %131 = vector.insert %119, %130 [%129, 1] : f16 into vector<32x2xf16>
        affine.yield %131 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %88 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_4) -> (vector<1x4xf32>) {
          %118 = frisk.to_threadTile %64 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map25, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map26, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
          %119 = affine.apply #map3(%arg11)
          %120 = affine.apply #map9(%arg13)
          %121 = vector.extract %118[%119, %120] : f16 from vector<2x32xf16>
          %122 = affine.apply #map10(%arg13)
          %123 = vector.extract %118[%119, %122] : f16 from vector<2x32xf16>
          %124 = affine.apply #map11(%arg13)
          %125 = vector.extract %118[%119, %124] : f16 from vector<2x32xf16>
          %126 = affine.apply #map12(%arg13)
          %127 = vector.extract %118[%119, %126] : f16 from vector<2x32xf16>
          %128 = vector.from_elements %121, %123, %125, %127 : vector<1x4xf16>
          %129 = affine.apply #map27(%arg11)
          %130 = vector.extract %87[%120, %129] : f16 from vector<32x2xf16>
          %131 = vector.extract %87[%122, %129] : f16 from vector<32x2xf16>
          %132 = vector.extract %87[%124, %129] : f16 from vector<32x2xf16>
          %133 = vector.extract %87[%126, %129] : f16 from vector<32x2xf16>
          %134 = vector.from_elements %130, %131, %132, %133 : vector<4x1xf16>
          %135 = frisk.warp_mma_rr %128, %134, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %135 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %105 = affine.apply #map3(%arg11)
        %106 = affine.apply #map4(%arg11)
        %107 = vector.extract %104[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%105, %106] : f32 into vector<2x8xf32>
        %109 = affine.apply #map5(%arg11)
        %110 = vector.extract %104[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%105, %109] : f32 into vector<2x8xf32>
        %112 = affine.apply #map6(%arg11)
        %113 = vector.extract %104[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%105, %112] : f32 into vector<2x8xf32>
        %115 = affine.apply #map7(%arg11)
        %116 = vector.extract %104[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%105, %115] : f32 into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %89 = frisk.to_threadTile %arg10 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %90 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %89[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %89[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %89[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %89[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = vector.extract %71[%104, %105] : f32 from vector<2x8xf32>
        %115 = vector.extract %71[%104, %107] : f32 from vector<2x8xf32>
        %116 = vector.extract %71[%104, %109] : f32 from vector<2x8xf32>
        %117 = vector.extract %71[%104, %111] : f32 from vector<2x8xf32>
        %118 = vector.from_elements %114, %115, %116, %117 : vector<1x4xf32>
        %119 = arith.mulf %113, %118 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %120 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %121 = vector.insert %120, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %122 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %121 [%104, %107] : f32 into vector<2x8xf32>
        %124 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %123 [%104, %109] : f32 into vector<2x8xf32>
        %126 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %127 = vector.insert %126, %125 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %91 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %90[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %90[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %90[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %90[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.from_elements %106, %108, %110, %112 : vector<1x4xf32>
        %114 = vector.extract %88[%104, %105] : f32 from vector<2x8xf32>
        %115 = vector.extract %88[%104, %107] : f32 from vector<2x8xf32>
        %116 = vector.extract %88[%104, %109] : f32 from vector<2x8xf32>
        %117 = vector.extract %88[%104, %111] : f32 from vector<2x8xf32>
        %118 = vector.from_elements %114, %115, %116, %117 : vector<1x4xf32>
        %119 = arith.addf %113, %118 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %120 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %121 = vector.insert %120, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %122 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %121 [%104, %107] : f32 into vector<2x8xf32>
        %124 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %123 [%104, %109] : f32 into vector<2x8xf32>
        %126 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %127 = vector.insert %126, %125 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %127 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %92 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %73[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %73[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %73[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %73[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.insert %106, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %114 = vector.insert %108, %113 [%104, %107] : f32 into vector<2x8xf32>
        %115 = vector.insert %110, %114 [%104, %109] : f32 into vector<2x8xf32>
        %116 = vector.insert %112, %115 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %93 = frisk.from_threadTile %92 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %94 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_10) -> (vector<2x1xf32>) {
        %104 = vector.extract %62[%arg11, 0] : f32 from vector<2x1xf32>
        %105 = vector.insert %104, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %105 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %95 = frisk.from_threadTile %94 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %96 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_10) -> (vector<2x1xf32>) {
        %104 = vector.extract %52[%arg11, 0] : f32 from vector<2x1xf32>
        %105 = vector.insert %104, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %105 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %97 = frisk.from_threadTile %96 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %98 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %79[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %79[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %79[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %79[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.insert %106, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %114 = vector.insert %108, %113 [%104, %107] : f32 into vector<2x8xf32>
        %115 = vector.insert %110, %114 [%104, %109] : f32 into vector<2x8xf32>
        %116 = vector.insert %112, %115 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %99 = frisk.from_threadTile %98 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %100 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %85[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %85[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %85[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %85[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.insert %106, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %114 = vector.insert %108, %113 [%104, %107] : f32 into vector<2x8xf32>
        %115 = vector.insert %110, %114 [%104, %109] : f32 into vector<2x8xf32>
        %116 = vector.insert %112, %115 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %101 = frisk.from_threadTile %100 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %102 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.apply #map3(%arg11)
        %105 = affine.apply #map4(%arg11)
        %106 = vector.extract %91[%104, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply #map5(%arg11)
        %108 = vector.extract %91[%104, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply #map6(%arg11)
        %110 = vector.extract %91[%104, %109] : f32 from vector<2x8xf32>
        %111 = affine.apply #map7(%arg11)
        %112 = vector.extract %91[%104, %111] : f32 from vector<2x8xf32>
        %113 = vector.insert %106, %arg12 [%104, %105] : f32 into vector<2x8xf32>
        %114 = vector.insert %108, %113 [%104, %107] : f32 into vector<2x8xf32>
        %115 = vector.insert %110, %114 [%104, %109] : f32 into vector<2x8xf32>
        %116 = vector.insert %112, %115 [%104, %111] : f32 into vector<2x8xf32>
        affine.yield %116 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %103 = frisk.from_threadTile %102 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      affine.yield %93, %95, %97, %99, %101, %103 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %14 = frisk.to_threadTile %13#5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x = gpu.thread_id x
    %15 = arith.index_cast %thread_id_x : index to i32
    %16 = arith.andi %15, %c16_i32 : i32
    %17 = arith.cmpi ne, %16, %c0_i32 : i32
    %18 = arith.andi %15, %c32_i32 : i32
    %19 = arith.cmpi ne, %18, %c0_i32 : i32
    %20 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %14[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %14[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %14[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %14[%47, %54] : f32 from vector<2x8xf32>
      %56 = arith.select %17, %49, %51 : f32
      %shuffleResult, %valid = gpu.shuffle xor %56, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %57 = arith.select %17, %shuffleResult, %49 : f32
      %58 = arith.select %17, %51, %shuffleResult : f32
      %59 = arith.select %17, %53, %55 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %59, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %60 = arith.select %17, %shuffleResult_13, %53 : f32
      %61 = arith.select %17, %55, %shuffleResult_13 : f32
      %62 = arith.select %19, %57, %60 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %62, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %63 = arith.select %19, %shuffleResult_15, %57 : f32
      %64 = arith.select %19, %60, %shuffleResult_15 : f32
      %65 = arith.select %19, %58, %61 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %65, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %66 = arith.select %19, %shuffleResult_17, %58 : f32
      %67 = arith.select %19, %61, %shuffleResult_17 : f32
      %68 = vector.insert %63, %arg5 [%47, %48] : f32 into vector<2x8xf32>
      %69 = vector.insert %66, %68 [%47, %50] : f32 into vector<2x8xf32>
      %70 = vector.insert %64, %69 [%47, %52] : f32 into vector<2x8xf32>
      %71 = vector.insert %67, %70 [%47, %54] : f32 into vector<2x8xf32>
      affine.yield %71 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %21 = frisk.to_threadTile %13#4 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %22 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %21[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %21[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %21[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %21[%47, %54] : f32 from vector<2x8xf32>
      %56 = arith.select %17, %49, %51 : f32
      %shuffleResult, %valid = gpu.shuffle xor %56, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %57 = arith.select %17, %shuffleResult, %49 : f32
      %58 = arith.select %17, %51, %shuffleResult : f32
      %59 = arith.select %17, %53, %55 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %59, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %60 = arith.select %17, %shuffleResult_13, %53 : f32
      %61 = arith.select %17, %55, %shuffleResult_13 : f32
      %62 = arith.select %19, %57, %60 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %62, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %63 = arith.select %19, %shuffleResult_15, %57 : f32
      %64 = arith.select %19, %60, %shuffleResult_15 : f32
      %65 = arith.select %19, %58, %61 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %65, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %66 = arith.select %19, %shuffleResult_17, %58 : f32
      %67 = arith.select %19, %61, %shuffleResult_17 : f32
      %68 = vector.insert %63, %arg5 [%47, %48] : f32 into vector<2x8xf32>
      %69 = vector.insert %66, %68 [%47, %50] : f32 into vector<2x8xf32>
      %70 = vector.insert %64, %69 [%47, %52] : f32 into vector<2x8xf32>
      %71 = vector.insert %67, %70 [%47, %54] : f32 into vector<2x8xf32>
      affine.yield %71 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %23 = frisk.to_threadTile %13#3 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %24 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %23[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %23[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %23[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %23[%47, %54] : f32 from vector<2x8xf32>
      %56 = arith.select %17, %49, %51 : f32
      %shuffleResult, %valid = gpu.shuffle xor %56, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %57 = arith.select %17, %shuffleResult, %49 : f32
      %58 = arith.select %17, %51, %shuffleResult : f32
      %59 = arith.select %17, %53, %55 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %59, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %60 = arith.select %17, %shuffleResult_13, %53 : f32
      %61 = arith.select %17, %55, %shuffleResult_13 : f32
      %62 = arith.select %19, %57, %60 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %62, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %63 = arith.select %19, %shuffleResult_15, %57 : f32
      %64 = arith.select %19, %60, %shuffleResult_15 : f32
      %65 = arith.select %19, %58, %61 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %65, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %66 = arith.select %19, %shuffleResult_17, %58 : f32
      %67 = arith.select %19, %61, %shuffleResult_17 : f32
      %68 = vector.insert %63, %arg5 [%47, %48] : f32 into vector<2x8xf32>
      %69 = vector.insert %66, %68 [%47, %50] : f32 into vector<2x8xf32>
      %70 = vector.insert %64, %69 [%47, %52] : f32 into vector<2x8xf32>
      %71 = vector.insert %67, %70 [%47, %54] : f32 into vector<2x8xf32>
      affine.yield %71 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %25 = frisk.to_threadTile %13#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %26 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %25[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %25[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %25[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %25[%47, %54] : f32 from vector<2x8xf32>
      %56 = arith.select %17, %49, %51 : f32
      %shuffleResult, %valid = gpu.shuffle xor %56, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %57 = arith.select %17, %shuffleResult, %49 : f32
      %58 = arith.select %17, %51, %shuffleResult : f32
      %59 = arith.select %17, %53, %55 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %59, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %60 = arith.select %17, %shuffleResult_13, %53 : f32
      %61 = arith.select %17, %55, %shuffleResult_13 : f32
      %62 = arith.select %19, %57, %60 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %62, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %63 = arith.select %19, %shuffleResult_15, %57 : f32
      %64 = arith.select %19, %60, %shuffleResult_15 : f32
      %65 = arith.select %19, %58, %61 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %65, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %66 = arith.select %19, %shuffleResult_17, %58 : f32
      %67 = arith.select %19, %61, %shuffleResult_17 : f32
      %68 = vector.insert %63, %arg5 [%47, %48] : f32 into vector<2x8xf32>
      %69 = vector.insert %66, %68 [%47, %50] : f32 into vector<2x8xf32>
      %70 = vector.insert %64, %69 [%47, %52] : f32 into vector<2x8xf32>
      %71 = vector.insert %67, %70 [%47, %54] : f32 into vector<2x8xf32>
      affine.yield %71 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %27 = frisk.to_threadTile %13#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %28 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = vector.extract %27[%47, 0] : f32 from vector<2x1xf32>
      %49 = affine.apply #map4(%arg4)
      %50 = vector.insert %48, %arg5 [%47, %49] : f32 into vector<2x8xf32>
      %51 = affine.apply #map5(%arg4)
      %52 = vector.insert %48, %50 [%47, %51] : f32 into vector<2x8xf32>
      %53 = affine.apply #map6(%arg4)
      %54 = vector.insert %48, %52 [%47, %53] : f32 into vector<2x8xf32>
      %55 = affine.apply #map7(%arg4)
      %56 = vector.insert %48, %54 [%47, %55] : f32 into vector<2x8xf32>
      affine.yield %56 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %29 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %26[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %26[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %26[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %26[%47, %54] : f32 from vector<2x8xf32>
      %56 = vector.from_elements %49, %51, %53, %55 : vector<1x4xf32>
      %57 = vector.extract %28[%47, %48] : f32 from vector<2x8xf32>
      %58 = vector.extract %28[%47, %50] : f32 from vector<2x8xf32>
      %59 = vector.extract %28[%47, %52] : f32 from vector<2x8xf32>
      %60 = vector.extract %28[%47, %54] : f32 from vector<2x8xf32>
      %61 = vector.from_elements %57, %58, %59, %60 : vector<1x4xf32>
      %62 = arith.divf %56, %61 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
      %64 = vector.insert %63, %arg5 [%47, %48] : f32 into vector<2x8xf32>
      %65 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %64 [%47, %50] : f32 into vector<2x8xf32>
      %67 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
      %68 = vector.insert %67, %66 [%47, %52] : f32 into vector<2x8xf32>
      %69 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
      %70 = vector.insert %69, %68 [%47, %54] : f32 into vector<2x8xf32>
      affine.yield %70 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %30 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %29[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %29[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %29[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %29[%47, %54] : f32 from vector<2x8xf32>
      %56 = vector.from_elements %49, %51, %53, %55 : vector<1x4xf32>
      %57 = arith.truncf %56 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %58 = vector.extract %57[0, 0] : f16 from vector<1x4xf16>
      %59 = vector.insert %58, %arg5 [%47, %48] : f16 into vector<2x8xf16>
      %60 = vector.extract %57[0, 1] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %59 [%47, %50] : f16 into vector<2x8xf16>
      %62 = vector.extract %57[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%47, %52] : f16 into vector<2x8xf16>
      %64 = vector.extract %57[0, 3] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%47, %54] : f16 into vector<2x8xf16>
      affine.yield %65 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %31 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %32 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %30[%47, %48] : f16 from vector<2x8xf16>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %30[%47, %50] : f16 from vector<2x8xf16>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %30[%47, %52] : f16 from vector<2x8xf16>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %30[%47, %54] : f16 from vector<2x8xf16>
      %56 = vector.insert %49, %arg5 [%47, %48] : f16 into vector<2x8xf16>
      %57 = vector.insert %51, %56 [%47, %50] : f16 into vector<2x8xf16>
      %58 = vector.insert %53, %57 [%47, %52] : f16 into vector<2x8xf16>
      %59 = vector.insert %55, %58 [%47, %54] : f16 into vector<2x8xf16>
      affine.yield %59 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %33 = affine.apply #map30()[%thread_id_x]
    %34 = arith.cmpi ult, %33, %c4 : index
    scf.if %34 {
      affine.for %arg4 = 0 to 4 {
        %47 = affine.apply #map3(%arg4)
        %48 = affine.apply #map4(%arg4)
        %49 = vector.extract %32[%47, %48] : f16 from vector<2x8xf16>
        %50 = affine.apply #map5(%arg4)
        %51 = vector.extract %32[%47, %50] : f16 from vector<2x8xf16>
        %52 = affine.apply #map6(%arg4)
        %53 = vector.extract %32[%47, %52] : f16 from vector<2x8xf16>
        %54 = affine.apply #map7(%arg4)
        %55 = vector.extract %32[%47, %54] : f16 from vector<2x8xf16>
        affine.store %49, %31[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %51, %31[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %53, %31[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %55, %31[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %35 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %24[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %24[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %24[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %24[%47, %54] : f32 from vector<2x8xf32>
      %56 = vector.from_elements %49, %51, %53, %55 : vector<1x4xf32>
      %57 = vector.extract %28[%47, %48] : f32 from vector<2x8xf32>
      %58 = vector.extract %28[%47, %50] : f32 from vector<2x8xf32>
      %59 = vector.extract %28[%47, %52] : f32 from vector<2x8xf32>
      %60 = vector.extract %28[%47, %54] : f32 from vector<2x8xf32>
      %61 = vector.from_elements %57, %58, %59, %60 : vector<1x4xf32>
      %62 = arith.divf %56, %61 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
      %64 = vector.insert %63, %arg5 [%47, %48] : f32 into vector<2x8xf32>
      %65 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %64 [%47, %50] : f32 into vector<2x8xf32>
      %67 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
      %68 = vector.insert %67, %66 [%47, %52] : f32 into vector<2x8xf32>
      %69 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
      %70 = vector.insert %69, %68 [%47, %54] : f32 into vector<2x8xf32>
      affine.yield %70 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %36 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %35[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %35[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %35[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %35[%47, %54] : f32 from vector<2x8xf32>
      %56 = vector.from_elements %49, %51, %53, %55 : vector<1x4xf32>
      %57 = arith.truncf %56 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %58 = vector.extract %57[0, 0] : f16 from vector<1x4xf16>
      %59 = vector.insert %58, %arg5 [%47, %48] : f16 into vector<2x8xf16>
      %60 = vector.extract %57[0, 1] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %59 [%47, %50] : f16 into vector<2x8xf16>
      %62 = vector.extract %57[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%47, %52] : f16 into vector<2x8xf16>
      %64 = vector.extract %57[0, 3] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%47, %54] : f16 into vector<2x8xf16>
      affine.yield %65 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %37 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %38 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %36[%47, %48] : f16 from vector<2x8xf16>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %36[%47, %50] : f16 from vector<2x8xf16>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %36[%47, %52] : f16 from vector<2x8xf16>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %36[%47, %54] : f16 from vector<2x8xf16>
      %56 = vector.insert %49, %arg5 [%47, %48] : f16 into vector<2x8xf16>
      %57 = vector.insert %51, %56 [%47, %50] : f16 into vector<2x8xf16>
      %58 = vector.insert %53, %57 [%47, %52] : f16 into vector<2x8xf16>
      %59 = vector.insert %55, %58 [%47, %54] : f16 into vector<2x8xf16>
      affine.yield %59 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %34 {
      affine.for %arg4 = 0 to 4 {
        %47 = affine.apply #map3(%arg4)
        %48 = affine.apply #map4(%arg4)
        %49 = vector.extract %38[%47, %48] : f16 from vector<2x8xf16>
        %50 = affine.apply #map5(%arg4)
        %51 = vector.extract %38[%47, %50] : f16 from vector<2x8xf16>
        %52 = affine.apply #map6(%arg4)
        %53 = vector.extract %38[%47, %52] : f16 from vector<2x8xf16>
        %54 = affine.apply #map7(%arg4)
        %55 = vector.extract %38[%47, %54] : f16 from vector<2x8xf16>
        affine.store %49, %37[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %51, %37[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %53, %37[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %55, %37[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %39 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %22[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %22[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %22[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %22[%47, %54] : f32 from vector<2x8xf32>
      %56 = vector.from_elements %49, %51, %53, %55 : vector<1x4xf32>
      %57 = vector.extract %28[%47, %48] : f32 from vector<2x8xf32>
      %58 = vector.extract %28[%47, %50] : f32 from vector<2x8xf32>
      %59 = vector.extract %28[%47, %52] : f32 from vector<2x8xf32>
      %60 = vector.extract %28[%47, %54] : f32 from vector<2x8xf32>
      %61 = vector.from_elements %57, %58, %59, %60 : vector<1x4xf32>
      %62 = arith.divf %56, %61 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
      %64 = vector.insert %63, %arg5 [%47, %48] : f32 into vector<2x8xf32>
      %65 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %64 [%47, %50] : f32 into vector<2x8xf32>
      %67 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
      %68 = vector.insert %67, %66 [%47, %52] : f32 into vector<2x8xf32>
      %69 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
      %70 = vector.insert %69, %68 [%47, %54] : f32 into vector<2x8xf32>
      affine.yield %70 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %40 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %39[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %39[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %39[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %39[%47, %54] : f32 from vector<2x8xf32>
      %56 = vector.from_elements %49, %51, %53, %55 : vector<1x4xf32>
      %57 = arith.truncf %56 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %58 = vector.extract %57[0, 0] : f16 from vector<1x4xf16>
      %59 = vector.insert %58, %arg5 [%47, %48] : f16 into vector<2x8xf16>
      %60 = vector.extract %57[0, 1] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %59 [%47, %50] : f16 into vector<2x8xf16>
      %62 = vector.extract %57[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%47, %52] : f16 into vector<2x8xf16>
      %64 = vector.extract %57[0, 3] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%47, %54] : f16 into vector<2x8xf16>
      affine.yield %65 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %41 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %42 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %40[%47, %48] : f16 from vector<2x8xf16>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %40[%47, %50] : f16 from vector<2x8xf16>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %40[%47, %52] : f16 from vector<2x8xf16>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %40[%47, %54] : f16 from vector<2x8xf16>
      %56 = vector.insert %49, %arg5 [%47, %48] : f16 into vector<2x8xf16>
      %57 = vector.insert %51, %56 [%47, %50] : f16 into vector<2x8xf16>
      %58 = vector.insert %53, %57 [%47, %52] : f16 into vector<2x8xf16>
      %59 = vector.insert %55, %58 [%47, %54] : f16 into vector<2x8xf16>
      affine.yield %59 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %34 {
      affine.for %arg4 = 0 to 4 {
        %47 = affine.apply #map3(%arg4)
        %48 = affine.apply #map4(%arg4)
        %49 = vector.extract %42[%47, %48] : f16 from vector<2x8xf16>
        %50 = affine.apply #map5(%arg4)
        %51 = vector.extract %42[%47, %50] : f16 from vector<2x8xf16>
        %52 = affine.apply #map6(%arg4)
        %53 = vector.extract %42[%47, %52] : f16 from vector<2x8xf16>
        %54 = affine.apply #map7(%arg4)
        %55 = vector.extract %42[%47, %54] : f16 from vector<2x8xf16>
        affine.store %49, %41[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %51, %41[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %53, %41[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %55, %41[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %43 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %20[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %20[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %20[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %20[%47, %54] : f32 from vector<2x8xf32>
      %56 = vector.from_elements %49, %51, %53, %55 : vector<1x4xf32>
      %57 = vector.extract %28[%47, %48] : f32 from vector<2x8xf32>
      %58 = vector.extract %28[%47, %50] : f32 from vector<2x8xf32>
      %59 = vector.extract %28[%47, %52] : f32 from vector<2x8xf32>
      %60 = vector.extract %28[%47, %54] : f32 from vector<2x8xf32>
      %61 = vector.from_elements %57, %58, %59, %60 : vector<1x4xf32>
      %62 = arith.divf %56, %61 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
      %64 = vector.insert %63, %arg5 [%47, %48] : f32 into vector<2x8xf32>
      %65 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %64 [%47, %50] : f32 into vector<2x8xf32>
      %67 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
      %68 = vector.insert %67, %66 [%47, %52] : f32 into vector<2x8xf32>
      %69 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
      %70 = vector.insert %69, %68 [%47, %54] : f32 into vector<2x8xf32>
      affine.yield %70 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %44 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %43[%47, %48] : f32 from vector<2x8xf32>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %43[%47, %50] : f32 from vector<2x8xf32>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %43[%47, %52] : f32 from vector<2x8xf32>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %43[%47, %54] : f32 from vector<2x8xf32>
      %56 = vector.from_elements %49, %51, %53, %55 : vector<1x4xf32>
      %57 = arith.truncf %56 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %58 = vector.extract %57[0, 0] : f16 from vector<1x4xf16>
      %59 = vector.insert %58, %arg5 [%47, %48] : f16 into vector<2x8xf16>
      %60 = vector.extract %57[0, 1] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %59 [%47, %50] : f16 into vector<2x8xf16>
      %62 = vector.extract %57[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%47, %52] : f16 into vector<2x8xf16>
      %64 = vector.extract %57[0, 3] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%47, %54] : f16 into vector<2x8xf16>
      affine.yield %65 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %45 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %46 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = affine.apply #map3(%arg4)
      %48 = affine.apply #map4(%arg4)
      %49 = vector.extract %44[%47, %48] : f16 from vector<2x8xf16>
      %50 = affine.apply #map5(%arg4)
      %51 = vector.extract %44[%47, %50] : f16 from vector<2x8xf16>
      %52 = affine.apply #map6(%arg4)
      %53 = vector.extract %44[%47, %52] : f16 from vector<2x8xf16>
      %54 = affine.apply #map7(%arg4)
      %55 = vector.extract %44[%47, %54] : f16 from vector<2x8xf16>
      %56 = vector.insert %49, %arg5 [%47, %48] : f16 into vector<2x8xf16>
      %57 = vector.insert %51, %56 [%47, %50] : f16 into vector<2x8xf16>
      %58 = vector.insert %53, %57 [%47, %52] : f16 into vector<2x8xf16>
      %59 = vector.insert %55, %58 [%47, %54] : f16 into vector<2x8xf16>
      affine.yield %59 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %34 {
      affine.for %arg4 = 0 to 4 {
        %47 = affine.apply #map3(%arg4)
        %48 = affine.apply #map4(%arg4)
        %49 = vector.extract %46[%47, %48] : f16 from vector<2x8xf16>
        %50 = affine.apply #map5(%arg4)
        %51 = vector.extract %46[%47, %50] : f16 from vector<2x8xf16>
        %52 = affine.apply #map6(%arg4)
        %53 = vector.extract %46[%47, %52] : f16 from vector<2x8xf16>
        %54 = affine.apply #map7(%arg4)
        %55 = vector.extract %46[%47, %54] : f16 from vector<2x8xf16>
        affine.store %49, %45[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %51, %45[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %53, %45[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %55, %45[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

