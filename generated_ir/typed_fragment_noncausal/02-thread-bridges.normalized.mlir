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
    %cst_10 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
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
      %47 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        frisk.fragment_yield %cst_10 : vector<1x1xf32>
      }
      %48 = vector.extract %47[0, 0] : f32 from vector<1x1xf32>
      %49 = vector.insert %48, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %49 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_11) -> (vector<2x1xf32>) {
      %47 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %50 = vector.extract %9[%arg4, 0] : f32 from vector<2x1xf32>
        %51 = vector.broadcast %50 : f32 to vector<1x1xf32>
        %52 = arith.addf %51, %cst {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        frisk.fragment_yield %52 : vector<1x1xf32>
      }
      %48 = vector.extract %47[0, 0] : f32 from vector<1x1xf32>
      %49 = vector.insert %48, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %49 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_11) -> (vector<2x1xf32>) {
      %47 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %50 = vector.extract %10[%arg4, 0] : f32 from vector<2x1xf32>
        %51 = vector.broadcast %50 : f32 to vector<1x1xf32>
        frisk.fragment_yield %51 : vector<1x1xf32>
      }
      %48 = vector.extract %47[0, 0] : f32 from vector<1x1xf32>
      %49 = vector.insert %48, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %49 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %12 = frisk.from_threadTile %11 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
    %13:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_9, %arg6 = %cst_8, %arg7 = %12, %arg8 = %cst_9, %arg9 = %cst_9, %arg10 = %cst_9) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %47 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_7) -> (vector<128x128xf32>) {
        %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %104 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %104 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %105 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_18 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %105 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %106 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map2, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
        %107 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_6) -> (vector<2x8xf16>) {
          %113 = frisk.fragment "compute" -> (vector<1x4xf16>) {
            %127 = affine.apply #map3(%arg13)
            %128 = affine.apply #map4(%arg13)
            %129 = vector.extract %106[%127, %128] : f16 from vector<2x8xf16>
            %130 = affine.apply #map5(%arg13)
            %131 = vector.extract %106[%127, %130] : f16 from vector<2x8xf16>
            %132 = affine.apply #map6(%arg13)
            %133 = vector.extract %106[%127, %132] : f16 from vector<2x8xf16>
            %134 = affine.apply #map7(%arg13)
            %135 = vector.extract %106[%127, %134] : f16 from vector<2x8xf16>
            %136 = vector.from_elements %129, %131, %133, %135 : vector<1x4xf16>
            %137 = vector.broadcast %8 : f16 to vector<1x4xf16>
            %138 = arith.mulf %136, %137 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
            frisk.fragment_yield %138 : vector<1x4xf16>
          }
          %114 = affine.apply #map3(%arg13)
          %115 = affine.apply #map4(%arg13)
          %116 = vector.extract %113[0, 0] : f16 from vector<1x4xf16>
          %117 = vector.insert %116, %arg14 [%114, %115] : f16 into vector<2x8xf16>
          %118 = affine.apply #map5(%arg13)
          %119 = vector.extract %113[0, 1] : f16 from vector<1x4xf16>
          %120 = vector.insert %119, %117 [%114, %118] : f16 into vector<2x8xf16>
          %121 = affine.apply #map6(%arg13)
          %122 = vector.extract %113[0, 2] : f16 from vector<1x4xf16>
          %123 = vector.insert %122, %120 [%114, %121] : f16 into vector<2x8xf16>
          %124 = affine.apply #map7(%arg13)
          %125 = vector.extract %113[0, 3] : f16 from vector<1x4xf16>
          %126 = vector.insert %125, %123 [%114, %124] : f16 into vector<2x8xf16>
          affine.yield %126 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        frisk.wait_buffer_ready %alloc_19 : memref<32x128xf16, 3>
        %108 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5) -> (vector<2x32xf32>) {
          %113 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %cst_4) -> (vector<1x4xf32>) {
            %127 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %130 = affine.apply #map8(%arg13)
              %131 = affine.apply #map9(%arg15)
              %132 = vector.extract %107[%130, %131] : f16 from vector<2x8xf16>
              %133 = affine.apply #map10(%arg15)
              %134 = vector.extract %107[%130, %133] : f16 from vector<2x8xf16>
              %135 = affine.apply #map11(%arg15)
              %136 = vector.extract %107[%130, %135] : f16 from vector<2x8xf16>
              %137 = affine.apply #map12(%arg15)
              %138 = vector.extract %107[%130, %137] : f16 from vector<2x8xf16>
              %139 = vector.from_elements %132, %134, %136, %138 : vector<1x4xf16>
              frisk.fragment_yield %139 : vector<1x4xf16>
            }
            %128 = frisk.fragment "load" -> (vector<4x1xf16>) {
              %130 = frisk.to_threadTile %alloc_19 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map13, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
              %131 = affine.apply #map9(%arg15)
              %132 = affine.apply #map14(%arg13)
              %133 = vector.extract %130[%131, %132] : f16 from vector<8x8xf16>
              %134 = affine.apply #map10(%arg15)
              %135 = vector.extract %130[%134, %132] : f16 from vector<8x8xf16>
              %136 = affine.apply #map11(%arg15)
              %137 = vector.extract %130[%136, %132] : f16 from vector<8x8xf16>
              %138 = affine.apply #map12(%arg15)
              %139 = vector.extract %130[%138, %132] : f16 from vector<8x8xf16>
              %140 = vector.from_elements %133, %135, %137, %139 : vector<4x1xf16>
              frisk.fragment_yield %140 : vector<4x1xf16>
            }
            %129 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %130 = frisk.warp_mma_rr %127, %128, %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %130 : vector<1x4xf32>
            }
            affine.yield %129 : vector<1x4xf32>
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
          %113 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %127 = affine.apply #map8(%arg13)
            %128 = affine.apply #map15(%arg13)
            %129 = vector.extract %109[%127, %128] : f32 from vector<2x32xf32>
            %130 = affine.apply #map16(%arg13)
            %131 = vector.extract %109[%127, %130] : f32 from vector<2x32xf32>
            %132 = affine.apply #map17(%arg13)
            %133 = vector.extract %109[%127, %132] : f32 from vector<2x32xf32>
            %134 = affine.apply #map18(%arg13)
            %135 = vector.extract %109[%127, %134] : f32 from vector<2x32xf32>
            %136 = vector.from_elements %129, %131, %133, %135 : vector<1x4xf32>
            %137 = vector.extract %108[%127, %128] : f32 from vector<2x32xf32>
            %138 = vector.extract %108[%127, %130] : f32 from vector<2x32xf32>
            %139 = vector.extract %108[%127, %132] : f32 from vector<2x32xf32>
            %140 = vector.extract %108[%127, %134] : f32 from vector<2x32xf32>
            %141 = vector.from_elements %137, %138, %139, %140 : vector<1x4xf32>
            %142 = arith.addf %136, %141 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %142 : vector<1x4xf32>
          }
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
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %111 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5) -> (vector<2x32xf32>) {
          %113 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %127 = affine.apply #map8(%arg13)
            %128 = affine.apply #map15(%arg13)
            %129 = vector.extract %110[%127, %128] : f32 from vector<2x32xf32>
            %130 = affine.apply #map16(%arg13)
            %131 = vector.extract %110[%127, %130] : f32 from vector<2x32xf32>
            %132 = affine.apply #map17(%arg13)
            %133 = vector.extract %110[%127, %132] : f32 from vector<2x32xf32>
            %134 = affine.apply #map18(%arg13)
            %135 = vector.extract %110[%127, %134] : f32 from vector<2x32xf32>
            %136 = vector.from_elements %129, %131, %133, %135 : vector<1x4xf32>
            frisk.fragment_yield %136 : vector<1x4xf32>
          }
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
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %112 = frisk.from_threadTile %111 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
        affine.yield %112 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %48 = frisk.to_threadTile %47 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %49 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map8(%arg11)
          %119 = affine.apply #map15(%arg11)
          %120 = vector.extract %48[%118, %119] : f32 from vector<2x32xf32>
          %121 = affine.apply #map16(%arg11)
          %122 = vector.extract %48[%118, %121] : f32 from vector<2x32xf32>
          %123 = affine.apply #map17(%arg11)
          %124 = vector.extract %48[%118, %123] : f32 from vector<2x32xf32>
          %125 = affine.apply #map18(%arg11)
          %126 = vector.extract %48[%118, %125] : f32 from vector<2x32xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          frisk.fragment_yield %127 : vector<1x4xf32>
        }
        %105 = affine.apply #map8(%arg11)
        %106 = affine.apply #map15(%arg11)
        %107 = vector.extract %104[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%105, %106] : f32 into vector<2x32xf32>
        %109 = affine.apply #map16(%arg11)
        %110 = vector.extract %104[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%105, %109] : f32 into vector<2x32xf32>
        %112 = affine.apply #map17(%arg11)
        %113 = vector.extract %104[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%105, %112] : f32 into vector<2x32xf32>
        %115 = affine.apply #map18(%arg11)
        %116 = vector.extract %104[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%105, %115] : f32 into vector<2x32xf32>
        affine.yield %117 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %50 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %107 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_12) -> (f32) {
            %111 = affine.apply #map9(%arg13)
            %112 = frisk.fragment "compute" -> (f32) {
              %113 = vector.extract %49[%arg11, %111] : f32 from vector<2x32xf32>
              %114 = arith.maxnumf %arg14, %113 : f32
              %115 = affine.apply #map10(%arg13)
              %116 = vector.extract %49[%arg11, %115] : f32 from vector<2x32xf32>
              %117 = arith.maxnumf %114, %116 : f32
              %118 = affine.apply #map11(%arg13)
              %119 = vector.extract %49[%arg11, %118] : f32 from vector<2x32xf32>
              %120 = arith.maxnumf %117, %119 : f32
              %121 = affine.apply #map12(%arg13)
              %122 = vector.extract %49[%arg11, %121] : f32 from vector<2x32xf32>
              %123 = arith.maxnumf %120, %122 : f32
              frisk.fragment_yield %123 : f32
            }
            affine.yield %112 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %shuffleResult, %valid = gpu.shuffle xor %107, %c16_i32, %c64_i32 : f32
          %108 = arith.maxnumf %107, %shuffleResult : f32
          %shuffleResult_18, %valid_19 = gpu.shuffle xor %108, %c32_i32, %c64_i32 : f32
          %109 = arith.maxnumf %108, %shuffleResult_18 : f32
          %110 = vector.broadcast %109 : f32 to vector<1x1xf32>
          frisk.fragment_yield %110 : vector<1x1xf32>
        }
        %105 = vector.extract %104[0, 0] : f32 from vector<1x1xf32>
        %106 = vector.insert %105, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %106 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %51 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %52 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %107 = vector.extract %50[%arg11, 0] : f32 from vector<2x1xf32>
          %108 = vector.broadcast %107 : f32 to vector<1x1xf32>
          %109 = vector.extract %51[%arg11, 0] : f32 from vector<2x1xf32>
          %110 = vector.broadcast %109 : f32 to vector<1x1xf32>
          %111 = arith.maxnumf %108, %110 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %111 : vector<1x1xf32>
        }
        %105 = vector.extract %104[0, 0] : f32 from vector<1x1xf32>
        %106 = vector.insert %105, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %106 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %53 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = affine.apply #map8(%arg11)
        %105 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = vector.extract %52[%104, 0] : f32 from vector<2x1xf32>
          %119 = vector.from_elements %118, %118, %118, %118 : vector<1x4xf32>
          frisk.fragment_yield %119 : vector<1x4xf32>
        }
        %106 = affine.apply #map15(%arg11)
        %107 = vector.extract %105[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%104, %106] : f32 into vector<2x32xf32>
        %109 = affine.apply #map16(%arg11)
        %110 = vector.extract %105[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%104, %109] : f32 into vector<2x32xf32>
        %112 = affine.apply #map17(%arg11)
        %113 = vector.extract %105[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%104, %112] : f32 into vector<2x32xf32>
        %115 = affine.apply #map18(%arg11)
        %116 = vector.extract %105[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%104, %115] : f32 into vector<2x32xf32>
        affine.yield %117 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %54 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map8(%arg11)
          %119 = affine.apply #map15(%arg11)
          %120 = vector.extract %48[%118, %119] : f32 from vector<2x32xf32>
          %121 = affine.apply #map16(%arg11)
          %122 = vector.extract %48[%118, %121] : f32 from vector<2x32xf32>
          %123 = affine.apply #map17(%arg11)
          %124 = vector.extract %48[%118, %123] : f32 from vector<2x32xf32>
          %125 = affine.apply #map18(%arg11)
          %126 = vector.extract %48[%118, %125] : f32 from vector<2x32xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = vector.extract %53[%118, %119] : f32 from vector<2x32xf32>
          %129 = vector.extract %53[%118, %121] : f32 from vector<2x32xf32>
          %130 = vector.extract %53[%118, %123] : f32 from vector<2x32xf32>
          %131 = vector.extract %53[%118, %125] : f32 from vector<2x32xf32>
          %132 = vector.from_elements %128, %129, %130, %131 : vector<1x4xf32>
          %133 = arith.subf %127, %132 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %133 : vector<1x4xf32>
        }
        %105 = affine.apply #map8(%arg11)
        %106 = affine.apply #map15(%arg11)
        %107 = vector.extract %104[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%105, %106] : f32 into vector<2x32xf32>
        %109 = affine.apply #map16(%arg11)
        %110 = vector.extract %104[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%105, %109] : f32 into vector<2x32xf32>
        %112 = affine.apply #map17(%arg11)
        %113 = vector.extract %104[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%105, %112] : f32 into vector<2x32xf32>
        %115 = affine.apply #map18(%arg11)
        %116 = vector.extract %104[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%105, %115] : f32 into vector<2x32xf32>
        affine.yield %117 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %55 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %107 = vector.extract %51[%arg11, 0] : f32 from vector<2x1xf32>
          %108 = vector.broadcast %107 : f32 to vector<1x1xf32>
          %109 = vector.extract %52[%arg11, 0] : f32 from vector<2x1xf32>
          %110 = vector.broadcast %109 : f32 to vector<1x1xf32>
          %111 = arith.subf %108, %110 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %111 : vector<1x1xf32>
        }
        %105 = vector.extract %104[0, 0] : f32 from vector<1x1xf32>
        %106 = vector.insert %105, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %106 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %56 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %107 = vector.extract %55[%arg11, 0] : f32 from vector<2x1xf32>
          %108 = vector.broadcast %107 : f32 to vector<1x1xf32>
          %109 = math.exp2 %108 {tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %109 : vector<1x1xf32>
        }
        %105 = vector.extract %104[0, 0] : f32 from vector<1x1xf32>
        %106 = vector.insert %105, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %106 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %57 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %58 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %107 = vector.extract %57[%arg11, 0] : f32 from vector<2x1xf32>
          %108 = vector.broadcast %107 : f32 to vector<1x1xf32>
          %109 = vector.extract %56[%arg11, 0] : f32 from vector<2x1xf32>
          %110 = vector.broadcast %109 : f32 to vector<1x1xf32>
          %111 = arith.mulf %108, %110 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %111 : vector<1x1xf32>
        }
        %105 = vector.extract %104[0, 0] : f32 from vector<1x1xf32>
        %106 = vector.insert %105, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %106 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %59 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map8(%arg11)
          %119 = affine.apply #map15(%arg11)
          %120 = vector.extract %54[%118, %119] : f32 from vector<2x32xf32>
          %121 = affine.apply #map16(%arg11)
          %122 = vector.extract %54[%118, %121] : f32 from vector<2x32xf32>
          %123 = affine.apply #map17(%arg11)
          %124 = vector.extract %54[%118, %123] : f32 from vector<2x32xf32>
          %125 = affine.apply #map18(%arg11)
          %126 = vector.extract %54[%118, %125] : f32 from vector<2x32xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = math.exp2 %127 {tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %128 : vector<1x4xf32>
        }
        %105 = affine.apply #map8(%arg11)
        %106 = affine.apply #map15(%arg11)
        %107 = vector.extract %104[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%105, %106] : f32 into vector<2x32xf32>
        %109 = affine.apply #map16(%arg11)
        %110 = vector.extract %104[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%105, %109] : f32 into vector<2x32xf32>
        %112 = affine.apply #map17(%arg11)
        %113 = vector.extract %104[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%105, %112] : f32 into vector<2x32xf32>
        %115 = affine.apply #map18(%arg11)
        %116 = vector.extract %104[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%105, %115] : f32 into vector<2x32xf32>
        affine.yield %117 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %60 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map8(%arg11)
          %119 = affine.apply #map15(%arg11)
          %120 = vector.extract %59[%118, %119] : f32 from vector<2x32xf32>
          %121 = affine.apply #map16(%arg11)
          %122 = vector.extract %59[%118, %121] : f32 from vector<2x32xf32>
          %123 = affine.apply #map17(%arg11)
          %124 = vector.extract %59[%118, %123] : f32 from vector<2x32xf32>
          %125 = affine.apply #map18(%arg11)
          %126 = vector.extract %59[%118, %125] : f32 from vector<2x32xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          frisk.fragment_yield %127 : vector<1x4xf32>
        }
        %105 = affine.apply #map8(%arg11)
        %106 = affine.apply #map15(%arg11)
        %107 = vector.extract %104[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%105, %106] : f32 into vector<2x32xf32>
        %109 = affine.apply #map16(%arg11)
        %110 = vector.extract %104[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%105, %109] : f32 into vector<2x32xf32>
        %112 = affine.apply #map17(%arg11)
        %113 = vector.extract %104[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%105, %112] : f32 into vector<2x32xf32>
        %115 = affine.apply #map18(%arg11)
        %116 = vector.extract %104[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%105, %115] : f32 into vector<2x32xf32>
        affine.yield %117 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %61 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %107 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_3) -> (f32) {
            %111 = affine.apply #map9(%arg13)
            %112 = frisk.fragment "compute" -> (f32) {
              %113 = vector.extract %60[%arg11, %111] : f32 from vector<2x32xf32>
              %114 = arith.addf %arg14, %113 : f32
              %115 = affine.apply #map10(%arg13)
              %116 = vector.extract %60[%arg11, %115] : f32 from vector<2x32xf32>
              %117 = arith.addf %114, %116 : f32
              %118 = affine.apply #map11(%arg13)
              %119 = vector.extract %60[%arg11, %118] : f32 from vector<2x32xf32>
              %120 = arith.addf %117, %119 : f32
              %121 = affine.apply #map12(%arg13)
              %122 = vector.extract %60[%arg11, %121] : f32 from vector<2x32xf32>
              %123 = arith.addf %120, %122 : f32
              frisk.fragment_yield %123 : f32
            }
            affine.yield %112 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %shuffleResult, %valid = gpu.shuffle xor %107, %c16_i32, %c64_i32 : f32
          %108 = arith.addf %107, %shuffleResult : f32
          %shuffleResult_18, %valid_19 = gpu.shuffle xor %108, %c32_i32, %c64_i32 : f32
          %109 = arith.addf %108, %shuffleResult_18 : f32
          %110 = vector.broadcast %109 : f32 to vector<1x1xf32>
          frisk.fragment_yield %110 : vector<1x1xf32>
        }
        %105 = vector.extract %104[0, 0] : f32 from vector<1x1xf32>
        %106 = vector.insert %105, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %106 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %62 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %107 = vector.extract %61[%arg11, 0] : f32 from vector<2x1xf32>
          %108 = vector.broadcast %107 : f32 to vector<1x1xf32>
          %109 = vector.extract %58[%arg11, 0] : f32 from vector<2x1xf32>
          %110 = vector.broadcast %109 : f32 to vector<1x1xf32>
          %111 = arith.addf %108, %110 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %111 : vector<1x1xf32>
        }
        %105 = vector.extract %104[0, 0] : f32 from vector<1x1xf32>
        %106 = vector.insert %105, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %106 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %63 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %118 = affine.apply #map8(%arg11)
          %119 = affine.apply #map15(%arg11)
          %120 = vector.extract %59[%118, %119] : f32 from vector<2x32xf32>
          %121 = affine.apply #map16(%arg11)
          %122 = vector.extract %59[%118, %121] : f32 from vector<2x32xf32>
          %123 = affine.apply #map17(%arg11)
          %124 = vector.extract %59[%118, %123] : f32 from vector<2x32xf32>
          %125 = affine.apply #map18(%arg11)
          %126 = vector.extract %59[%118, %125] : f32 from vector<2x32xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = arith.truncf %127 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %128 : vector<1x4xf16>
        }
        %105 = affine.apply #map8(%arg11)
        %106 = affine.apply #map15(%arg11)
        %107 = vector.extract %104[0, 0] : f16 from vector<1x4xf16>
        %108 = vector.insert %107, %arg12 [%105, %106] : f16 into vector<2x32xf16>
        %109 = affine.apply #map16(%arg11)
        %110 = vector.extract %104[0, 1] : f16 from vector<1x4xf16>
        %111 = vector.insert %110, %108 [%105, %109] : f16 into vector<2x32xf16>
        %112 = affine.apply #map17(%arg11)
        %113 = vector.extract %104[0, 2] : f16 from vector<1x4xf16>
        %114 = vector.insert %113, %111 [%105, %112] : f16 into vector<2x32xf16>
        %115 = affine.apply #map18(%arg11)
        %116 = vector.extract %104[0, 3] : f16 from vector<1x4xf16>
        %117 = vector.insert %116, %114 [%105, %115] : f16 into vector<2x32xf16>
        affine.yield %117 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %64 = frisk.from_threadTile %63 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x128xf16, 3>
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %65 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %65 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_14 = gpu.thread_id x
      %66 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %104 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %125 = affine.apply #map19(%arg11)[%thread_id_x_14]
          %126 = affine.apply #map20()[%thread_id_x_14]
          %127 = frisk.read_matrix_from_shm %alloc[%125, %126] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %128 = vector.bitcast %127 : vector<4xi32> to vector<8xf16>
          %129 = vector.extract %128[0] : f16 from vector<8xf16>
          %130 = affine.apply #map21()[%thread_id_x_14]
          %131 = vector.extract %128[%130] : f16 from vector<8xf16>
          %132 = vector.extract %128[1] : f16 from vector<8xf16>
          %133 = affine.apply #map22()[%thread_id_x_14]
          %134 = vector.extract %128[%133] : f16 from vector<8xf16>
          %135 = vector.extract %128[2] : f16 from vector<8xf16>
          %136 = affine.apply #map23()[%thread_id_x_14]
          %137 = vector.extract %128[%136] : f16 from vector<8xf16>
          %138 = vector.extract %128[3] : f16 from vector<8xf16>
          %139 = affine.apply #map24()[%thread_id_x_14]
          %140 = vector.extract %128[%139] : f16 from vector<8xf16>
          %141 = vector.from_elements %129, %131, %132, %134, %135, %137, %138, %140 : vector<4x2xf16>
          frisk.fragment_yield %141 : vector<4x2xf16>
        }
        %105 = affine.apply #map9(%arg11)
        %106 = vector.extract %104[0, 0] : f16 from vector<4x2xf16>
        %107 = vector.insert %106, %arg12 [%105, 0] : f16 into vector<32x2xf16>
        %108 = vector.extract %104[0, 1] : f16 from vector<4x2xf16>
        %109 = vector.insert %108, %107 [%105, 1] : f16 into vector<32x2xf16>
        %110 = affine.apply #map10(%arg11)
        %111 = vector.extract %104[1, 0] : f16 from vector<4x2xf16>
        %112 = vector.insert %111, %109 [%110, 0] : f16 into vector<32x2xf16>
        %113 = vector.extract %104[1, 1] : f16 from vector<4x2xf16>
        %114 = vector.insert %113, %112 [%110, 1] : f16 into vector<32x2xf16>
        %115 = affine.apply #map11(%arg11)
        %116 = vector.extract %104[2, 0] : f16 from vector<4x2xf16>
        %117 = vector.insert %116, %114 [%115, 0] : f16 into vector<32x2xf16>
        %118 = vector.extract %104[2, 1] : f16 from vector<4x2xf16>
        %119 = vector.insert %118, %117 [%115, 1] : f16 into vector<32x2xf16>
        %120 = affine.apply #map12(%arg11)
        %121 = vector.extract %104[3, 0] : f16 from vector<4x2xf16>
        %122 = vector.insert %121, %119 [%120, 0] : f16 into vector<32x2xf16>
        %123 = vector.extract %104[3, 1] : f16 from vector<4x2xf16>
        %124 = vector.insert %123, %122 [%120, 1] : f16 into vector<32x2xf16>
        affine.yield %124 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %67 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_4) -> (vector<1x4xf32>) {
          %118 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %121 = frisk.to_threadTile %64 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map25, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map26, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %122 = affine.apply #map3(%arg11)
            %123 = affine.apply #map9(%arg13)
            %124 = vector.extract %121[%122, %123] : f16 from vector<2x32xf16>
            %125 = affine.apply #map10(%arg13)
            %126 = vector.extract %121[%122, %125] : f16 from vector<2x32xf16>
            %127 = affine.apply #map11(%arg13)
            %128 = vector.extract %121[%122, %127] : f16 from vector<2x32xf16>
            %129 = affine.apply #map12(%arg13)
            %130 = vector.extract %121[%122, %129] : f16 from vector<2x32xf16>
            %131 = vector.from_elements %124, %126, %128, %130 : vector<1x4xf16>
            frisk.fragment_yield %131 : vector<1x4xf16>
          }
          %119 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %121 = affine.apply #map9(%arg13)
            %122 = affine.apply #map27(%arg11)
            %123 = vector.extract %66[%121, %122] : f16 from vector<32x2xf16>
            %124 = affine.apply #map10(%arg13)
            %125 = vector.extract %66[%124, %122] : f16 from vector<32x2xf16>
            %126 = affine.apply #map11(%arg13)
            %127 = vector.extract %66[%126, %122] : f16 from vector<32x2xf16>
            %128 = affine.apply #map12(%arg13)
            %129 = vector.extract %66[%128, %122] : f16 from vector<32x2xf16>
            %130 = vector.from_elements %123, %125, %127, %129 : vector<4x1xf16>
            frisk.fragment_yield %130 : vector<4x1xf16>
          }
          %120 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %121 = frisk.warp_mma_rr %118, %119, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %121 : vector<1x4xf32>
          }
          affine.yield %120 : vector<1x4xf32>
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
        %105 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = vector.extract %70[%104, 0] : f32 from vector<2x1xf32>
          %119 = vector.from_elements %118, %118, %118, %118 : vector<1x4xf32>
          frisk.fragment_yield %119 : vector<1x4xf32>
        }
        %106 = affine.apply #map4(%arg11)
        %107 = vector.extract %105[0, 0] : f32 from vector<1x4xf32>
        %108 = vector.insert %107, %arg12 [%104, %106] : f32 into vector<2x8xf32>
        %109 = affine.apply #map5(%arg11)
        %110 = vector.extract %105[0, 1] : f32 from vector<1x4xf32>
        %111 = vector.insert %110, %108 [%104, %109] : f32 into vector<2x8xf32>
        %112 = affine.apply #map6(%arg11)
        %113 = vector.extract %105[0, 2] : f32 from vector<1x4xf32>
        %114 = vector.insert %113, %111 [%104, %112] : f32 into vector<2x8xf32>
        %115 = affine.apply #map7(%arg11)
        %116 = vector.extract %105[0, 3] : f32 from vector<1x4xf32>
        %117 = vector.insert %116, %114 [%104, %115] : f32 into vector<2x8xf32>
        affine.yield %117 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %72 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %69[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %69[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %69[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %69[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = vector.extract %71[%118, %119] : f32 from vector<2x8xf32>
          %129 = vector.extract %71[%118, %121] : f32 from vector<2x8xf32>
          %130 = vector.extract %71[%118, %123] : f32 from vector<2x8xf32>
          %131 = vector.extract %71[%118, %125] : f32 from vector<2x8xf32>
          %132 = vector.from_elements %128, %129, %130, %131 : vector<1x4xf32>
          %133 = arith.mulf %127, %132 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %133 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %73 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %72[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %72[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %72[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %72[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = vector.extract %67[%118, %119] : f32 from vector<2x8xf32>
          %129 = vector.extract %67[%118, %121] : f32 from vector<2x8xf32>
          %130 = vector.extract %67[%118, %123] : f32 from vector<2x8xf32>
          %131 = vector.extract %67[%118, %125] : f32 from vector<2x8xf32>
          %132 = vector.from_elements %128, %129, %130, %131 : vector<1x4xf32>
          %133 = arith.addf %127, %132 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %133 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %74 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %74 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<128x32xf16, 3>
      %75 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %104 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %125 = affine.apply #map19(%arg11)[%thread_id_x_14]
          %126 = affine.apply #map20()[%thread_id_x_14]
          %127 = frisk.read_matrix_from_shm %alloc_15[%125, %126] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %128 = vector.bitcast %127 : vector<4xi32> to vector<8xf16>
          %129 = vector.extract %128[0] : f16 from vector<8xf16>
          %130 = affine.apply #map21()[%thread_id_x_14]
          %131 = vector.extract %128[%130] : f16 from vector<8xf16>
          %132 = vector.extract %128[1] : f16 from vector<8xf16>
          %133 = affine.apply #map22()[%thread_id_x_14]
          %134 = vector.extract %128[%133] : f16 from vector<8xf16>
          %135 = vector.extract %128[2] : f16 from vector<8xf16>
          %136 = affine.apply #map23()[%thread_id_x_14]
          %137 = vector.extract %128[%136] : f16 from vector<8xf16>
          %138 = vector.extract %128[3] : f16 from vector<8xf16>
          %139 = affine.apply #map24()[%thread_id_x_14]
          %140 = vector.extract %128[%139] : f16 from vector<8xf16>
          %141 = vector.from_elements %129, %131, %132, %134, %135, %137, %138, %140 : vector<4x2xf16>
          frisk.fragment_yield %141 : vector<4x2xf16>
        }
        %105 = affine.apply #map9(%arg11)
        %106 = vector.extract %104[0, 0] : f16 from vector<4x2xf16>
        %107 = vector.insert %106, %arg12 [%105, 0] : f16 into vector<32x2xf16>
        %108 = vector.extract %104[0, 1] : f16 from vector<4x2xf16>
        %109 = vector.insert %108, %107 [%105, 1] : f16 into vector<32x2xf16>
        %110 = affine.apply #map10(%arg11)
        %111 = vector.extract %104[1, 0] : f16 from vector<4x2xf16>
        %112 = vector.insert %111, %109 [%110, 0] : f16 into vector<32x2xf16>
        %113 = vector.extract %104[1, 1] : f16 from vector<4x2xf16>
        %114 = vector.insert %113, %112 [%110, 1] : f16 into vector<32x2xf16>
        %115 = affine.apply #map11(%arg11)
        %116 = vector.extract %104[2, 0] : f16 from vector<4x2xf16>
        %117 = vector.insert %116, %114 [%115, 0] : f16 into vector<32x2xf16>
        %118 = vector.extract %104[2, 1] : f16 from vector<4x2xf16>
        %119 = vector.insert %118, %117 [%115, 1] : f16 into vector<32x2xf16>
        %120 = affine.apply #map12(%arg11)
        %121 = vector.extract %104[3, 0] : f16 from vector<4x2xf16>
        %122 = vector.insert %121, %119 [%120, 0] : f16 into vector<32x2xf16>
        %123 = vector.extract %104[3, 1] : f16 from vector<4x2xf16>
        %124 = vector.insert %123, %122 [%120, 1] : f16 into vector<32x2xf16>
        affine.yield %124 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %76 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_4) -> (vector<1x4xf32>) {
          %118 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %121 = frisk.to_threadTile %64 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map25, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map26, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %122 = affine.apply #map3(%arg11)
            %123 = affine.apply #map9(%arg13)
            %124 = vector.extract %121[%122, %123] : f16 from vector<2x32xf16>
            %125 = affine.apply #map10(%arg13)
            %126 = vector.extract %121[%122, %125] : f16 from vector<2x32xf16>
            %127 = affine.apply #map11(%arg13)
            %128 = vector.extract %121[%122, %127] : f16 from vector<2x32xf16>
            %129 = affine.apply #map12(%arg13)
            %130 = vector.extract %121[%122, %129] : f16 from vector<2x32xf16>
            %131 = vector.from_elements %124, %126, %128, %130 : vector<1x4xf16>
            frisk.fragment_yield %131 : vector<1x4xf16>
          }
          %119 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %121 = affine.apply #map9(%arg13)
            %122 = affine.apply #map27(%arg11)
            %123 = vector.extract %75[%121, %122] : f16 from vector<32x2xf16>
            %124 = affine.apply #map10(%arg13)
            %125 = vector.extract %75[%124, %122] : f16 from vector<32x2xf16>
            %126 = affine.apply #map11(%arg13)
            %127 = vector.extract %75[%126, %122] : f16 from vector<32x2xf16>
            %128 = affine.apply #map12(%arg13)
            %129 = vector.extract %75[%128, %122] : f16 from vector<32x2xf16>
            %130 = vector.from_elements %123, %125, %127, %129 : vector<4x1xf16>
            frisk.fragment_yield %130 : vector<4x1xf16>
          }
          %120 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %121 = frisk.warp_mma_rr %118, %119, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %121 : vector<1x4xf32>
          }
          affine.yield %120 : vector<1x4xf32>
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
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %77[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %77[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %77[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %77[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = vector.extract %71[%118, %119] : f32 from vector<2x8xf32>
          %129 = vector.extract %71[%118, %121] : f32 from vector<2x8xf32>
          %130 = vector.extract %71[%118, %123] : f32 from vector<2x8xf32>
          %131 = vector.extract %71[%118, %125] : f32 from vector<2x8xf32>
          %132 = vector.from_elements %128, %129, %130, %131 : vector<1x4xf32>
          %133 = arith.mulf %127, %132 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %133 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %79 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %78[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %78[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %78[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %78[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = vector.extract %76[%118, %119] : f32 from vector<2x8xf32>
          %129 = vector.extract %76[%118, %121] : f32 from vector<2x8xf32>
          %130 = vector.extract %76[%118, %123] : f32 from vector<2x8xf32>
          %131 = vector.extract %76[%118, %125] : f32 from vector<2x8xf32>
          %132 = vector.from_elements %128, %129, %130, %131 : vector<1x4xf32>
          %133 = arith.addf %127, %132 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %133 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %80 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %80 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_16 : memref<128x32xf16, 3>
      %81 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %104 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %125 = affine.apply #map19(%arg11)[%thread_id_x_14]
          %126 = affine.apply #map20()[%thread_id_x_14]
          %127 = frisk.read_matrix_from_shm %alloc_16[%125, %126] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %128 = vector.bitcast %127 : vector<4xi32> to vector<8xf16>
          %129 = vector.extract %128[0] : f16 from vector<8xf16>
          %130 = affine.apply #map21()[%thread_id_x_14]
          %131 = vector.extract %128[%130] : f16 from vector<8xf16>
          %132 = vector.extract %128[1] : f16 from vector<8xf16>
          %133 = affine.apply #map22()[%thread_id_x_14]
          %134 = vector.extract %128[%133] : f16 from vector<8xf16>
          %135 = vector.extract %128[2] : f16 from vector<8xf16>
          %136 = affine.apply #map23()[%thread_id_x_14]
          %137 = vector.extract %128[%136] : f16 from vector<8xf16>
          %138 = vector.extract %128[3] : f16 from vector<8xf16>
          %139 = affine.apply #map24()[%thread_id_x_14]
          %140 = vector.extract %128[%139] : f16 from vector<8xf16>
          %141 = vector.from_elements %129, %131, %132, %134, %135, %137, %138, %140 : vector<4x2xf16>
          frisk.fragment_yield %141 : vector<4x2xf16>
        }
        %105 = affine.apply #map9(%arg11)
        %106 = vector.extract %104[0, 0] : f16 from vector<4x2xf16>
        %107 = vector.insert %106, %arg12 [%105, 0] : f16 into vector<32x2xf16>
        %108 = vector.extract %104[0, 1] : f16 from vector<4x2xf16>
        %109 = vector.insert %108, %107 [%105, 1] : f16 into vector<32x2xf16>
        %110 = affine.apply #map10(%arg11)
        %111 = vector.extract %104[1, 0] : f16 from vector<4x2xf16>
        %112 = vector.insert %111, %109 [%110, 0] : f16 into vector<32x2xf16>
        %113 = vector.extract %104[1, 1] : f16 from vector<4x2xf16>
        %114 = vector.insert %113, %112 [%110, 1] : f16 into vector<32x2xf16>
        %115 = affine.apply #map11(%arg11)
        %116 = vector.extract %104[2, 0] : f16 from vector<4x2xf16>
        %117 = vector.insert %116, %114 [%115, 0] : f16 into vector<32x2xf16>
        %118 = vector.extract %104[2, 1] : f16 from vector<4x2xf16>
        %119 = vector.insert %118, %117 [%115, 1] : f16 into vector<32x2xf16>
        %120 = affine.apply #map12(%arg11)
        %121 = vector.extract %104[3, 0] : f16 from vector<4x2xf16>
        %122 = vector.insert %121, %119 [%120, 0] : f16 into vector<32x2xf16>
        %123 = vector.extract %104[3, 1] : f16 from vector<4x2xf16>
        %124 = vector.insert %123, %122 [%120, 1] : f16 into vector<32x2xf16>
        affine.yield %124 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %82 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_4) -> (vector<1x4xf32>) {
          %118 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %121 = frisk.to_threadTile %64 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map25, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map26, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %122 = affine.apply #map3(%arg11)
            %123 = affine.apply #map9(%arg13)
            %124 = vector.extract %121[%122, %123] : f16 from vector<2x32xf16>
            %125 = affine.apply #map10(%arg13)
            %126 = vector.extract %121[%122, %125] : f16 from vector<2x32xf16>
            %127 = affine.apply #map11(%arg13)
            %128 = vector.extract %121[%122, %127] : f16 from vector<2x32xf16>
            %129 = affine.apply #map12(%arg13)
            %130 = vector.extract %121[%122, %129] : f16 from vector<2x32xf16>
            %131 = vector.from_elements %124, %126, %128, %130 : vector<1x4xf16>
            frisk.fragment_yield %131 : vector<1x4xf16>
          }
          %119 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %121 = affine.apply #map9(%arg13)
            %122 = affine.apply #map27(%arg11)
            %123 = vector.extract %81[%121, %122] : f16 from vector<32x2xf16>
            %124 = affine.apply #map10(%arg13)
            %125 = vector.extract %81[%124, %122] : f16 from vector<32x2xf16>
            %126 = affine.apply #map11(%arg13)
            %127 = vector.extract %81[%126, %122] : f16 from vector<32x2xf16>
            %128 = affine.apply #map12(%arg13)
            %129 = vector.extract %81[%128, %122] : f16 from vector<32x2xf16>
            %130 = vector.from_elements %123, %125, %127, %129 : vector<4x1xf16>
            frisk.fragment_yield %130 : vector<4x1xf16>
          }
          %120 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %121 = frisk.warp_mma_rr %118, %119, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %121 : vector<1x4xf32>
          }
          affine.yield %120 : vector<1x4xf32>
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
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %83[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %83[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %83[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %83[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = vector.extract %71[%118, %119] : f32 from vector<2x8xf32>
          %129 = vector.extract %71[%118, %121] : f32 from vector<2x8xf32>
          %130 = vector.extract %71[%118, %123] : f32 from vector<2x8xf32>
          %131 = vector.extract %71[%118, %125] : f32 from vector<2x8xf32>
          %132 = vector.from_elements %128, %129, %130, %131 : vector<1x4xf32>
          %133 = arith.mulf %127, %132 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %133 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %85 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %84[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %84[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %84[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %84[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = vector.extract %82[%118, %119] : f32 from vector<2x8xf32>
          %129 = vector.extract %82[%118, %121] : f32 from vector<2x8xf32>
          %130 = vector.extract %82[%118, %123] : f32 from vector<2x8xf32>
          %131 = vector.extract %82[%118, %125] : f32 from vector<2x8xf32>
          %132 = vector.from_elements %128, %129, %130, %131 : vector<1x4xf32>
          %133 = arith.addf %127, %132 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %133 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %86 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %86 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_17 : memref<128x32xf16, 3>
      %87 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %104 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %125 = affine.apply #map19(%arg11)[%thread_id_x_14]
          %126 = affine.apply #map20()[%thread_id_x_14]
          %127 = frisk.read_matrix_from_shm %alloc_17[%125, %126] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %128 = vector.bitcast %127 : vector<4xi32> to vector<8xf16>
          %129 = vector.extract %128[0] : f16 from vector<8xf16>
          %130 = affine.apply #map21()[%thread_id_x_14]
          %131 = vector.extract %128[%130] : f16 from vector<8xf16>
          %132 = vector.extract %128[1] : f16 from vector<8xf16>
          %133 = affine.apply #map22()[%thread_id_x_14]
          %134 = vector.extract %128[%133] : f16 from vector<8xf16>
          %135 = vector.extract %128[2] : f16 from vector<8xf16>
          %136 = affine.apply #map23()[%thread_id_x_14]
          %137 = vector.extract %128[%136] : f16 from vector<8xf16>
          %138 = vector.extract %128[3] : f16 from vector<8xf16>
          %139 = affine.apply #map24()[%thread_id_x_14]
          %140 = vector.extract %128[%139] : f16 from vector<8xf16>
          %141 = vector.from_elements %129, %131, %132, %134, %135, %137, %138, %140 : vector<4x2xf16>
          frisk.fragment_yield %141 : vector<4x2xf16>
        }
        %105 = affine.apply #map9(%arg11)
        %106 = vector.extract %104[0, 0] : f16 from vector<4x2xf16>
        %107 = vector.insert %106, %arg12 [%105, 0] : f16 into vector<32x2xf16>
        %108 = vector.extract %104[0, 1] : f16 from vector<4x2xf16>
        %109 = vector.insert %108, %107 [%105, 1] : f16 into vector<32x2xf16>
        %110 = affine.apply #map10(%arg11)
        %111 = vector.extract %104[1, 0] : f16 from vector<4x2xf16>
        %112 = vector.insert %111, %109 [%110, 0] : f16 into vector<32x2xf16>
        %113 = vector.extract %104[1, 1] : f16 from vector<4x2xf16>
        %114 = vector.insert %113, %112 [%110, 1] : f16 into vector<32x2xf16>
        %115 = affine.apply #map11(%arg11)
        %116 = vector.extract %104[2, 0] : f16 from vector<4x2xf16>
        %117 = vector.insert %116, %114 [%115, 0] : f16 into vector<32x2xf16>
        %118 = vector.extract %104[2, 1] : f16 from vector<4x2xf16>
        %119 = vector.insert %118, %117 [%115, 1] : f16 into vector<32x2xf16>
        %120 = affine.apply #map12(%arg11)
        %121 = vector.extract %104[3, 0] : f16 from vector<4x2xf16>
        %122 = vector.insert %121, %119 [%120, 0] : f16 into vector<32x2xf16>
        %123 = vector.extract %104[3, 1] : f16 from vector<4x2xf16>
        %124 = vector.insert %123, %122 [%120, 1] : f16 into vector<32x2xf16>
        affine.yield %124 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %88 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_4) -> (vector<1x4xf32>) {
          %118 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %121 = frisk.to_threadTile %64 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map25, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map26, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %122 = affine.apply #map3(%arg11)
            %123 = affine.apply #map9(%arg13)
            %124 = vector.extract %121[%122, %123] : f16 from vector<2x32xf16>
            %125 = affine.apply #map10(%arg13)
            %126 = vector.extract %121[%122, %125] : f16 from vector<2x32xf16>
            %127 = affine.apply #map11(%arg13)
            %128 = vector.extract %121[%122, %127] : f16 from vector<2x32xf16>
            %129 = affine.apply #map12(%arg13)
            %130 = vector.extract %121[%122, %129] : f16 from vector<2x32xf16>
            %131 = vector.from_elements %124, %126, %128, %130 : vector<1x4xf16>
            frisk.fragment_yield %131 : vector<1x4xf16>
          }
          %119 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %121 = affine.apply #map9(%arg13)
            %122 = affine.apply #map27(%arg11)
            %123 = vector.extract %87[%121, %122] : f16 from vector<32x2xf16>
            %124 = affine.apply #map10(%arg13)
            %125 = vector.extract %87[%124, %122] : f16 from vector<32x2xf16>
            %126 = affine.apply #map11(%arg13)
            %127 = vector.extract %87[%126, %122] : f16 from vector<32x2xf16>
            %128 = affine.apply #map12(%arg13)
            %129 = vector.extract %87[%128, %122] : f16 from vector<32x2xf16>
            %130 = vector.from_elements %123, %125, %127, %129 : vector<4x1xf16>
            frisk.fragment_yield %130 : vector<4x1xf16>
          }
          %120 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %121 = frisk.warp_mma_rr %118, %119, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %121 : vector<1x4xf32>
          }
          affine.yield %120 : vector<1x4xf32>
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
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %89[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %89[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %89[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %89[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = vector.extract %71[%118, %119] : f32 from vector<2x8xf32>
          %129 = vector.extract %71[%118, %121] : f32 from vector<2x8xf32>
          %130 = vector.extract %71[%118, %123] : f32 from vector<2x8xf32>
          %131 = vector.extract %71[%118, %125] : f32 from vector<2x8xf32>
          %132 = vector.from_elements %128, %129, %130, %131 : vector<1x4xf32>
          %133 = arith.mulf %127, %132 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %133 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %91 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %90[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %90[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %90[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %90[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          %128 = vector.extract %88[%118, %119] : f32 from vector<2x8xf32>
          %129 = vector.extract %88[%118, %121] : f32 from vector<2x8xf32>
          %130 = vector.extract %88[%118, %123] : f32 from vector<2x8xf32>
          %131 = vector.extract %88[%118, %125] : f32 from vector<2x8xf32>
          %132 = vector.from_elements %128, %129, %130, %131 : vector<1x4xf32>
          %133 = arith.addf %127, %132 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %133 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %92 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %73[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %73[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %73[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %73[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          frisk.fragment_yield %127 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %93 = frisk.from_threadTile %92 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %94 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %107 = vector.extract %62[%arg11, 0] : f32 from vector<2x1xf32>
          %108 = vector.broadcast %107 : f32 to vector<1x1xf32>
          frisk.fragment_yield %108 : vector<1x1xf32>
        }
        %105 = vector.extract %104[0, 0] : f32 from vector<1x1xf32>
        %106 = vector.insert %105, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %106 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %95 = frisk.from_threadTile %94 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %96 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11) -> (vector<2x1xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %107 = vector.extract %52[%arg11, 0] : f32 from vector<2x1xf32>
          %108 = vector.broadcast %107 : f32 to vector<1x1xf32>
          frisk.fragment_yield %108 : vector<1x1xf32>
        }
        %105 = vector.extract %104[0, 0] : f32 from vector<1x1xf32>
        %106 = vector.insert %105, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %106 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %97 = frisk.from_threadTile %96 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %98 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %79[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %79[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %79[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %79[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          frisk.fragment_yield %127 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %99 = frisk.from_threadTile %98 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %100 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %85[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %85[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %85[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %85[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          frisk.fragment_yield %127 : vector<1x4xf32>
        }
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
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %101 = frisk.from_threadTile %100 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %102 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0) -> (vector<2x8xf32>) {
        %104 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %118 = affine.apply #map3(%arg11)
          %119 = affine.apply #map4(%arg11)
          %120 = vector.extract %91[%118, %119] : f32 from vector<2x8xf32>
          %121 = affine.apply #map5(%arg11)
          %122 = vector.extract %91[%118, %121] : f32 from vector<2x8xf32>
          %123 = affine.apply #map6(%arg11)
          %124 = vector.extract %91[%118, %123] : f32 from vector<2x8xf32>
          %125 = affine.apply #map7(%arg11)
          %126 = vector.extract %91[%118, %125] : f32 from vector<2x8xf32>
          %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
          frisk.fragment_yield %127 : vector<1x4xf32>
        }
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
      %47 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %14[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %14[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %14[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %14[%61, %68] : f32 from vector<2x8xf32>
        %70 = arith.select %17, %63, %65 : f32
        %shuffleResult, %valid = gpu.shuffle xor %70, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %71 = arith.select %17, %shuffleResult, %63 : f32
        %72 = arith.select %17, %65, %shuffleResult : f32
        %73 = arith.select %17, %67, %69 : f32
        %shuffleResult_14, %valid_15 = gpu.shuffle xor %73, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %74 = arith.select %17, %shuffleResult_14, %67 : f32
        %75 = arith.select %17, %69, %shuffleResult_14 : f32
        %76 = arith.select %19, %71, %74 : f32
        %shuffleResult_16, %valid_17 = gpu.shuffle xor %76, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %77 = arith.select %19, %shuffleResult_16, %71 : f32
        %78 = arith.select %19, %74, %shuffleResult_16 : f32
        %79 = arith.select %19, %72, %75 : f32
        %shuffleResult_18, %valid_19 = gpu.shuffle xor %79, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %80 = arith.select %19, %shuffleResult_18, %72 : f32
        %81 = arith.select %19, %75, %shuffleResult_18 : f32
        %82 = vector.from_elements %77, %80, %78, %81 : vector<1x4xf32>
        frisk.fragment_yield %82 : vector<1x4xf32>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %arg5 [%48, %49] : f32 into vector<2x8xf32>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %51 [%48, %52] : f32 into vector<2x8xf32>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %54 [%48, %55] : f32 into vector<2x8xf32>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%48, %58] : f32 into vector<2x8xf32>
      affine.yield %60 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %21 = frisk.to_threadTile %13#4 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %22 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %21[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %21[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %21[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %21[%61, %68] : f32 from vector<2x8xf32>
        %70 = arith.select %17, %63, %65 : f32
        %shuffleResult, %valid = gpu.shuffle xor %70, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %71 = arith.select %17, %shuffleResult, %63 : f32
        %72 = arith.select %17, %65, %shuffleResult : f32
        %73 = arith.select %17, %67, %69 : f32
        %shuffleResult_14, %valid_15 = gpu.shuffle xor %73, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %74 = arith.select %17, %shuffleResult_14, %67 : f32
        %75 = arith.select %17, %69, %shuffleResult_14 : f32
        %76 = arith.select %19, %71, %74 : f32
        %shuffleResult_16, %valid_17 = gpu.shuffle xor %76, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %77 = arith.select %19, %shuffleResult_16, %71 : f32
        %78 = arith.select %19, %74, %shuffleResult_16 : f32
        %79 = arith.select %19, %72, %75 : f32
        %shuffleResult_18, %valid_19 = gpu.shuffle xor %79, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %80 = arith.select %19, %shuffleResult_18, %72 : f32
        %81 = arith.select %19, %75, %shuffleResult_18 : f32
        %82 = vector.from_elements %77, %80, %78, %81 : vector<1x4xf32>
        frisk.fragment_yield %82 : vector<1x4xf32>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %arg5 [%48, %49] : f32 into vector<2x8xf32>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %51 [%48, %52] : f32 into vector<2x8xf32>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %54 [%48, %55] : f32 into vector<2x8xf32>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%48, %58] : f32 into vector<2x8xf32>
      affine.yield %60 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %23 = frisk.to_threadTile %13#3 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %24 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %23[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %23[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %23[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %23[%61, %68] : f32 from vector<2x8xf32>
        %70 = arith.select %17, %63, %65 : f32
        %shuffleResult, %valid = gpu.shuffle xor %70, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %71 = arith.select %17, %shuffleResult, %63 : f32
        %72 = arith.select %17, %65, %shuffleResult : f32
        %73 = arith.select %17, %67, %69 : f32
        %shuffleResult_14, %valid_15 = gpu.shuffle xor %73, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %74 = arith.select %17, %shuffleResult_14, %67 : f32
        %75 = arith.select %17, %69, %shuffleResult_14 : f32
        %76 = arith.select %19, %71, %74 : f32
        %shuffleResult_16, %valid_17 = gpu.shuffle xor %76, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %77 = arith.select %19, %shuffleResult_16, %71 : f32
        %78 = arith.select %19, %74, %shuffleResult_16 : f32
        %79 = arith.select %19, %72, %75 : f32
        %shuffleResult_18, %valid_19 = gpu.shuffle xor %79, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %80 = arith.select %19, %shuffleResult_18, %72 : f32
        %81 = arith.select %19, %75, %shuffleResult_18 : f32
        %82 = vector.from_elements %77, %80, %78, %81 : vector<1x4xf32>
        frisk.fragment_yield %82 : vector<1x4xf32>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %arg5 [%48, %49] : f32 into vector<2x8xf32>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %51 [%48, %52] : f32 into vector<2x8xf32>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %54 [%48, %55] : f32 into vector<2x8xf32>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%48, %58] : f32 into vector<2x8xf32>
      affine.yield %60 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %25 = frisk.to_threadTile %13#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map29, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %26 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %25[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %25[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %25[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %25[%61, %68] : f32 from vector<2x8xf32>
        %70 = arith.select %17, %63, %65 : f32
        %shuffleResult, %valid = gpu.shuffle xor %70, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %71 = arith.select %17, %shuffleResult, %63 : f32
        %72 = arith.select %17, %65, %shuffleResult : f32
        %73 = arith.select %17, %67, %69 : f32
        %shuffleResult_14, %valid_15 = gpu.shuffle xor %73, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %74 = arith.select %17, %shuffleResult_14, %67 : f32
        %75 = arith.select %17, %69, %shuffleResult_14 : f32
        %76 = arith.select %19, %71, %74 : f32
        %shuffleResult_16, %valid_17 = gpu.shuffle xor %76, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %77 = arith.select %19, %shuffleResult_16, %71 : f32
        %78 = arith.select %19, %74, %shuffleResult_16 : f32
        %79 = arith.select %19, %72, %75 : f32
        %shuffleResult_18, %valid_19 = gpu.shuffle xor %79, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %80 = arith.select %19, %shuffleResult_18, %72 : f32
        %81 = arith.select %19, %75, %shuffleResult_18 : f32
        %82 = vector.from_elements %77, %80, %78, %81 : vector<1x4xf32>
        frisk.fragment_yield %82 : vector<1x4xf32>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %arg5 [%48, %49] : f32 into vector<2x8xf32>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %51 [%48, %52] : f32 into vector<2x8xf32>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %54 [%48, %55] : f32 into vector<2x8xf32>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%48, %58] : f32 into vector<2x8xf32>
      affine.yield %60 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %27 = frisk.to_threadTile %13#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map1, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %28 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = affine.apply #map3(%arg4)
      %48 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %61 = vector.extract %27[%47, 0] : f32 from vector<2x1xf32>
        %62 = vector.from_elements %61, %61, %61, %61 : vector<1x4xf32>
        frisk.fragment_yield %62 : vector<1x4xf32>
      }
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %48[0, 0] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %arg5 [%47, %49] : f32 into vector<2x8xf32>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %48[0, 1] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %51 [%47, %52] : f32 into vector<2x8xf32>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %48[0, 2] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %54 [%47, %55] : f32 into vector<2x8xf32>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %48[0, 3] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%47, %58] : f32 into vector<2x8xf32>
      affine.yield %60 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %29 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %26[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %26[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %26[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %26[%61, %68] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = vector.extract %28[%61, %62] : f32 from vector<2x8xf32>
        %72 = vector.extract %28[%61, %64] : f32 from vector<2x8xf32>
        %73 = vector.extract %28[%61, %66] : f32 from vector<2x8xf32>
        %74 = vector.extract %28[%61, %68] : f32 from vector<2x8xf32>
        %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
        %76 = arith.divf %70, %75 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %76 : vector<1x4xf32>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %arg5 [%48, %49] : f32 into vector<2x8xf32>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %51 [%48, %52] : f32 into vector<2x8xf32>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %54 [%48, %55] : f32 into vector<2x8xf32>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%48, %58] : f32 into vector<2x8xf32>
      affine.yield %60 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %30 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %29[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %29[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %29[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %29[%61, %68] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = arith.truncf %70 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %71 : vector<1x4xf16>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f16 from vector<1x4xf16>
      %51 = vector.insert %50, %arg5 [%48, %49] : f16 into vector<2x8xf16>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %51 [%48, %52] : f16 into vector<2x8xf16>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %54 [%48, %55] : f16 into vector<2x8xf16>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%48, %58] : f16 into vector<2x8xf16>
      affine.yield %60 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %31 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %32 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %30[%61, %62] : f16 from vector<2x8xf16>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %30[%61, %64] : f16 from vector<2x8xf16>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %30[%61, %66] : f16 from vector<2x8xf16>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %30[%61, %68] : f16 from vector<2x8xf16>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf16>
        frisk.fragment_yield %70 : vector<1x4xf16>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f16 from vector<1x4xf16>
      %51 = vector.insert %50, %arg5 [%48, %49] : f16 into vector<2x8xf16>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %51 [%48, %52] : f16 into vector<2x8xf16>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %54 [%48, %55] : f16 into vector<2x8xf16>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%48, %58] : f16 into vector<2x8xf16>
      affine.yield %60 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %33 = affine.apply #map30()[%thread_id_x]
    %34 = arith.cmpi ult, %33, %c4 : index
    scf.if %34 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
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
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %35 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %24[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %24[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %24[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %24[%61, %68] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = vector.extract %28[%61, %62] : f32 from vector<2x8xf32>
        %72 = vector.extract %28[%61, %64] : f32 from vector<2x8xf32>
        %73 = vector.extract %28[%61, %66] : f32 from vector<2x8xf32>
        %74 = vector.extract %28[%61, %68] : f32 from vector<2x8xf32>
        %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
        %76 = arith.divf %70, %75 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %76 : vector<1x4xf32>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %arg5 [%48, %49] : f32 into vector<2x8xf32>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %51 [%48, %52] : f32 into vector<2x8xf32>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %54 [%48, %55] : f32 into vector<2x8xf32>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%48, %58] : f32 into vector<2x8xf32>
      affine.yield %60 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %36 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %35[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %35[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %35[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %35[%61, %68] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = arith.truncf %70 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %71 : vector<1x4xf16>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f16 from vector<1x4xf16>
      %51 = vector.insert %50, %arg5 [%48, %49] : f16 into vector<2x8xf16>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %51 [%48, %52] : f16 into vector<2x8xf16>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %54 [%48, %55] : f16 into vector<2x8xf16>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%48, %58] : f16 into vector<2x8xf16>
      affine.yield %60 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %37 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %38 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %36[%61, %62] : f16 from vector<2x8xf16>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %36[%61, %64] : f16 from vector<2x8xf16>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %36[%61, %66] : f16 from vector<2x8xf16>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %36[%61, %68] : f16 from vector<2x8xf16>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf16>
        frisk.fragment_yield %70 : vector<1x4xf16>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f16 from vector<1x4xf16>
      %51 = vector.insert %50, %arg5 [%48, %49] : f16 into vector<2x8xf16>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %51 [%48, %52] : f16 into vector<2x8xf16>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %54 [%48, %55] : f16 into vector<2x8xf16>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%48, %58] : f16 into vector<2x8xf16>
      affine.yield %60 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %34 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
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
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %39 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %22[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %22[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %22[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %22[%61, %68] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = vector.extract %28[%61, %62] : f32 from vector<2x8xf32>
        %72 = vector.extract %28[%61, %64] : f32 from vector<2x8xf32>
        %73 = vector.extract %28[%61, %66] : f32 from vector<2x8xf32>
        %74 = vector.extract %28[%61, %68] : f32 from vector<2x8xf32>
        %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
        %76 = arith.divf %70, %75 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %76 : vector<1x4xf32>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %arg5 [%48, %49] : f32 into vector<2x8xf32>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %51 [%48, %52] : f32 into vector<2x8xf32>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %54 [%48, %55] : f32 into vector<2x8xf32>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%48, %58] : f32 into vector<2x8xf32>
      affine.yield %60 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %40 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %39[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %39[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %39[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %39[%61, %68] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = arith.truncf %70 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %71 : vector<1x4xf16>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f16 from vector<1x4xf16>
      %51 = vector.insert %50, %arg5 [%48, %49] : f16 into vector<2x8xf16>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %51 [%48, %52] : f16 into vector<2x8xf16>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %54 [%48, %55] : f16 into vector<2x8xf16>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%48, %58] : f16 into vector<2x8xf16>
      affine.yield %60 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %41 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %42 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %40[%61, %62] : f16 from vector<2x8xf16>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %40[%61, %64] : f16 from vector<2x8xf16>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %40[%61, %66] : f16 from vector<2x8xf16>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %40[%61, %68] : f16 from vector<2x8xf16>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf16>
        frisk.fragment_yield %70 : vector<1x4xf16>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f16 from vector<1x4xf16>
      %51 = vector.insert %50, %arg5 [%48, %49] : f16 into vector<2x8xf16>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %51 [%48, %52] : f16 into vector<2x8xf16>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %54 [%48, %55] : f16 into vector<2x8xf16>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%48, %58] : f16 into vector<2x8xf16>
      affine.yield %60 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %34 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
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
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %43 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %20[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %20[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %20[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %20[%61, %68] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = vector.extract %28[%61, %62] : f32 from vector<2x8xf32>
        %72 = vector.extract %28[%61, %64] : f32 from vector<2x8xf32>
        %73 = vector.extract %28[%61, %66] : f32 from vector<2x8xf32>
        %74 = vector.extract %28[%61, %68] : f32 from vector<2x8xf32>
        %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
        %76 = arith.divf %70, %75 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %76 : vector<1x4xf32>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %arg5 [%48, %49] : f32 into vector<2x8xf32>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %51 [%48, %52] : f32 into vector<2x8xf32>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %54 [%48, %55] : f32 into vector<2x8xf32>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%48, %58] : f32 into vector<2x8xf32>
      affine.yield %60 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %44 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %43[%61, %62] : f32 from vector<2x8xf32>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %43[%61, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %43[%61, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %43[%61, %68] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = arith.truncf %70 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %71 : vector<1x4xf16>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f16 from vector<1x4xf16>
      %51 = vector.insert %50, %arg5 [%48, %49] : f16 into vector<2x8xf16>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %51 [%48, %52] : f16 into vector<2x8xf16>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %54 [%48, %55] : f16 into vector<2x8xf16>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%48, %58] : f16 into vector<2x8xf16>
      affine.yield %60 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %45 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %46 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_6) -> (vector<2x8xf16>) {
      %47 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %61 = affine.apply #map3(%arg4)
        %62 = affine.apply #map4(%arg4)
        %63 = vector.extract %44[%61, %62] : f16 from vector<2x8xf16>
        %64 = affine.apply #map5(%arg4)
        %65 = vector.extract %44[%61, %64] : f16 from vector<2x8xf16>
        %66 = affine.apply #map6(%arg4)
        %67 = vector.extract %44[%61, %66] : f16 from vector<2x8xf16>
        %68 = affine.apply #map7(%arg4)
        %69 = vector.extract %44[%61, %68] : f16 from vector<2x8xf16>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf16>
        frisk.fragment_yield %70 : vector<1x4xf16>
      }
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map4(%arg4)
      %50 = vector.extract %47[0, 0] : f16 from vector<1x4xf16>
      %51 = vector.insert %50, %arg5 [%48, %49] : f16 into vector<2x8xf16>
      %52 = affine.apply #map5(%arg4)
      %53 = vector.extract %47[0, 1] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %51 [%48, %52] : f16 into vector<2x8xf16>
      %55 = affine.apply #map6(%arg4)
      %56 = vector.extract %47[0, 2] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %54 [%48, %55] : f16 into vector<2x8xf16>
      %58 = affine.apply #map7(%arg4)
      %59 = vector.extract %47[0, 3] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%48, %58] : f16 into vector<2x8xf16>
      affine.yield %60 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %34 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
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
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

