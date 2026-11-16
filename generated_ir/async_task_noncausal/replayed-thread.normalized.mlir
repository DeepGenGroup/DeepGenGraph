#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>
#map1 = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>
#map4 = affine_map<(d0) -> ((d0 + 1) floordiv 2)>
#map5 = affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>
#map6 = affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>
#map7 = affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>
#map8 = affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>
#map9 = affine_map<(d0) -> (d0 floordiv 2)>
#map10 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>
#map11 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>
#map12 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>
#map13 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>
#map14 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>
#map15 = affine_map<(d0) -> (d0 mod 8)>
#map16 = affine_map<(d0) -> (d0 * 4 + 4)>
#map17 = affine_map<(d0) -> (d0 * 4 + 5)>
#map18 = affine_map<(d0) -> (d0 * 4 + 6)>
#map19 = affine_map<(d0) -> (d0 * 4 + 7)>
#map20 = affine_map<(d0) -> (d0 floordiv 8)>
#map21 = affine_map<(d0) -> (d0 * 4)>
#map22 = affine_map<(d0) -> (d0 * 4 + 1)>
#map23 = affine_map<(d0) -> (d0 * 4 + 2)>
#map24 = affine_map<(d0) -> (d0 * 4 + 3)>
#map25 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>
#map26 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>
#map27 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>
#map28 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>
#map29 = affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>
#map30 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
#map31 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>
#map32 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>
#map33 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>
#map34 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>
#map35 = affine_map<(d0) -> (d0 mod 2)>
#map36 = affine_map<()[s0] -> (s0 floordiv 64)>
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
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c96 = arith.constant 96 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
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
      %105 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %106 = vector.insert %cst_10, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %107 = vector.insert %cst_10, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %108 = vector.insert %cst_10, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %105, %106, %107, %108 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = frisk.to_threadTile %cst_8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %11 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map2, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %12 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c0) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c0 floordiv 128) mod 4096, %c0 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %13 = frisk.async_copy_gm_to_shm %12 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %13 : !frisk.async_task
    %14 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c32) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c32 floordiv 128) mod 4096, %c32 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %alloc_12 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %15 = frisk.async_copy_gm_to_shm %14 to %alloc_12 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %16 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %17 = vector.extract %16[0, 0] : f16 from vector<2x8xf16>
    %18 = vector.extract %16[0, 1] : f16 from vector<2x8xf16>
    %19 = vector.extract %16[0, 2] : f16 from vector<2x8xf16>
    %20 = vector.extract %16[0, 3] : f16 from vector<2x8xf16>
    %21 = vector.from_elements %17, %18, %19, %20 : vector<1x4xf16>
    %22:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %21) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %105 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %106 = affine.apply #map4(%arg4)
      %107 = affine.apply #map5(%arg4)
      %108 = vector.extract %105[%106, %107] : f16 from vector<2x8xf16>
      %109 = affine.apply #map6(%arg4)
      %110 = vector.extract %105[%106, %109] : f16 from vector<2x8xf16>
      %111 = affine.apply #map7(%arg4)
      %112 = vector.extract %105[%106, %111] : f16 from vector<2x8xf16>
      %113 = affine.apply #map8(%arg4)
      %114 = vector.extract %105[%106, %113] : f16 from vector<2x8xf16>
      %115 = vector.from_elements %108, %110, %112, %114 : vector<1x4xf16>
      %116 = affine.apply #map9(%arg4)
      %117 = affine.apply #map10(%arg4)
      %118 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg5 [%116, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply #map11(%arg4)
      %121 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%116, %120] : f16 into vector<2x8xf16>
      %123 = affine.apply #map12(%arg4)
      %124 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %122 [%116, %123] : f16 into vector<2x8xf16>
      %126 = affine.apply #map13(%arg4)
      %127 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %125 [%116, %126] : f16 into vector<2x8xf16>
      %129 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %130 = vector.insert %8, %arg6 [%116, %117] : f16 into vector<2x8xf16>
      %131 = vector.insert %8, %130 [%116, %120] : f16 into vector<2x8xf16>
      %132 = vector.insert %8, %131 [%116, %123] : f16 into vector<2x8xf16>
      %133 = vector.insert %8, %132 [%116, %126] : f16 into vector<2x8xf16>
      %134 = arith.mulf %arg8, %129 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%116, %117] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%116, %120] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%116, %123] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%116, %126] : f16 into vector<2x8xf16>
      affine.yield %128, %133, %142, %115 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %23 = vector.broadcast %8 : f16 to vector<1x4xf16>
    %24 = arith.mulf %22#3, %23 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %25 = vector.extract %24[0, 0] : f16 from vector<1x4xf16>
    %26 = vector.insert %25, %22#2 [1, 4] : f16 into vector<2x8xf16>
    %27 = vector.extract %24[0, 1] : f16 from vector<1x4xf16>
    %28 = vector.insert %27, %26 [1, 5] : f16 into vector<2x8xf16>
    %29 = vector.extract %24[0, 2] : f16 from vector<1x4xf16>
    %30 = vector.insert %29, %28 [1, 6] : f16 into vector<2x8xf16>
    %31 = vector.extract %24[0, 3] : f16 from vector<1x4xf16>
    %32 = vector.insert %31, %30 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %15 : !frisk.async_task
    %33 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c64) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c64 floordiv 128) mod 4096, %c64 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %alloc_13 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %34 = frisk.async_copy_gm_to_shm %33 to %alloc_13 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %35 = frisk.to_threadTile %alloc_12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %36 = vector.extract %35[0, 0] : f16 from vector<2x8xf16>
    %37 = vector.extract %35[0, 1] : f16 from vector<2x8xf16>
    %38 = vector.extract %35[0, 2] : f16 from vector<2x8xf16>
    %39 = vector.extract %35[0, 3] : f16 from vector<2x8xf16>
    %40 = vector.from_elements %36, %37, %38, %39 : vector<1x4xf16>
    %41:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %40) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %105 = frisk.to_threadTile %alloc_12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %106 = affine.apply #map4(%arg4)
      %107 = affine.apply #map5(%arg4)
      %108 = vector.extract %105[%106, %107] : f16 from vector<2x8xf16>
      %109 = affine.apply #map6(%arg4)
      %110 = vector.extract %105[%106, %109] : f16 from vector<2x8xf16>
      %111 = affine.apply #map7(%arg4)
      %112 = vector.extract %105[%106, %111] : f16 from vector<2x8xf16>
      %113 = affine.apply #map8(%arg4)
      %114 = vector.extract %105[%106, %113] : f16 from vector<2x8xf16>
      %115 = vector.from_elements %108, %110, %112, %114 : vector<1x4xf16>
      %116 = affine.apply #map9(%arg4)
      %117 = affine.apply #map10(%arg4)
      %118 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg5 [%116, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply #map11(%arg4)
      %121 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%116, %120] : f16 into vector<2x8xf16>
      %123 = affine.apply #map12(%arg4)
      %124 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %122 [%116, %123] : f16 into vector<2x8xf16>
      %126 = affine.apply #map13(%arg4)
      %127 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %125 [%116, %126] : f16 into vector<2x8xf16>
      %129 = vector.insert %8, %arg6 [%116, %117] : f16 into vector<2x8xf16>
      %130 = vector.insert %8, %129 [%116, %120] : f16 into vector<2x8xf16>
      %131 = vector.insert %8, %130 [%116, %123] : f16 into vector<2x8xf16>
      %132 = vector.insert %8, %131 [%116, %126] : f16 into vector<2x8xf16>
      %133 = arith.mulf %arg8, %23 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%116, %117] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%116, %120] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%116, %123] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%116, %126] : f16 into vector<2x8xf16>
      affine.yield %128, %132, %141, %115 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %42 = arith.mulf %41#3, %23 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %43 = vector.extract %42[0, 0] : f16 from vector<1x4xf16>
    %44 = vector.insert %43, %41#2 [1, 4] : f16 into vector<2x8xf16>
    %45 = vector.extract %42[0, 1] : f16 from vector<1x4xf16>
    %46 = vector.insert %45, %44 [1, 5] : f16 into vector<2x8xf16>
    %47 = vector.extract %42[0, 2] : f16 from vector<1x4xf16>
    %48 = vector.insert %47, %46 [1, 6] : f16 into vector<2x8xf16>
    %49 = vector.extract %42[0, 3] : f16 from vector<1x4xf16>
    %50 = vector.insert %49, %48 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %34 : !frisk.async_task
    %51 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c96) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c96 floordiv 128) mod 4096, %c96 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %alloc_14 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %52 = frisk.async_copy_gm_to_shm %51 to %alloc_14 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %53 = frisk.to_threadTile %alloc_13 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %54 = vector.extract %53[0, 0] : f16 from vector<2x8xf16>
    %55 = vector.extract %53[0, 1] : f16 from vector<2x8xf16>
    %56 = vector.extract %53[0, 2] : f16 from vector<2x8xf16>
    %57 = vector.extract %53[0, 3] : f16 from vector<2x8xf16>
    %58 = vector.from_elements %54, %55, %56, %57 : vector<1x4xf16>
    %59:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %58) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %105 = frisk.to_threadTile %alloc_13 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %106 = affine.apply #map4(%arg4)
      %107 = affine.apply #map5(%arg4)
      %108 = vector.extract %105[%106, %107] : f16 from vector<2x8xf16>
      %109 = affine.apply #map6(%arg4)
      %110 = vector.extract %105[%106, %109] : f16 from vector<2x8xf16>
      %111 = affine.apply #map7(%arg4)
      %112 = vector.extract %105[%106, %111] : f16 from vector<2x8xf16>
      %113 = affine.apply #map8(%arg4)
      %114 = vector.extract %105[%106, %113] : f16 from vector<2x8xf16>
      %115 = vector.from_elements %108, %110, %112, %114 : vector<1x4xf16>
      %116 = affine.apply #map9(%arg4)
      %117 = affine.apply #map10(%arg4)
      %118 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg5 [%116, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply #map11(%arg4)
      %121 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%116, %120] : f16 into vector<2x8xf16>
      %123 = affine.apply #map12(%arg4)
      %124 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %122 [%116, %123] : f16 into vector<2x8xf16>
      %126 = affine.apply #map13(%arg4)
      %127 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %125 [%116, %126] : f16 into vector<2x8xf16>
      %129 = vector.insert %8, %arg6 [%116, %117] : f16 into vector<2x8xf16>
      %130 = vector.insert %8, %129 [%116, %120] : f16 into vector<2x8xf16>
      %131 = vector.insert %8, %130 [%116, %123] : f16 into vector<2x8xf16>
      %132 = vector.insert %8, %131 [%116, %126] : f16 into vector<2x8xf16>
      %133 = arith.mulf %arg8, %23 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%116, %117] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%116, %120] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%116, %123] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%116, %126] : f16 into vector<2x8xf16>
      affine.yield %128, %132, %141, %115 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %60 = arith.mulf %59#3, %23 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %61 = vector.extract %60[0, 0] : f16 from vector<1x4xf16>
    %62 = vector.insert %61, %59#2 [1, 4] : f16 into vector<2x8xf16>
    %63 = vector.extract %60[0, 1] : f16 from vector<1x4xf16>
    %64 = vector.insert %63, %62 [1, 5] : f16 into vector<2x8xf16>
    %65 = vector.extract %60[0, 2] : f16 from vector<1x4xf16>
    %66 = vector.insert %65, %64 [1, 6] : f16 into vector<2x8xf16>
    %67 = vector.extract %60[0, 3] : f16 from vector<1x4xf16>
    %68 = vector.insert %67, %66 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %52 : !frisk.async_task
    %69 = frisk.to_threadTile %alloc_14 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %70 = vector.extract %69[0, 0] : f16 from vector<2x8xf16>
    %71 = vector.extract %69[0, 1] : f16 from vector<2x8xf16>
    %72 = vector.extract %69[0, 2] : f16 from vector<2x8xf16>
    %73 = vector.extract %69[0, 3] : f16 from vector<2x8xf16>
    %74 = vector.from_elements %70, %71, %72, %73 : vector<1x4xf16>
    %75:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %74) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %105 = frisk.to_threadTile %alloc_14 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %106 = affine.apply #map4(%arg4)
      %107 = affine.apply #map5(%arg4)
      %108 = vector.extract %105[%106, %107] : f16 from vector<2x8xf16>
      %109 = affine.apply #map6(%arg4)
      %110 = vector.extract %105[%106, %109] : f16 from vector<2x8xf16>
      %111 = affine.apply #map7(%arg4)
      %112 = vector.extract %105[%106, %111] : f16 from vector<2x8xf16>
      %113 = affine.apply #map8(%arg4)
      %114 = vector.extract %105[%106, %113] : f16 from vector<2x8xf16>
      %115 = vector.from_elements %108, %110, %112, %114 : vector<1x4xf16>
      %116 = affine.apply #map9(%arg4)
      %117 = affine.apply #map10(%arg4)
      %118 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg5 [%116, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply #map11(%arg4)
      %121 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%116, %120] : f16 into vector<2x8xf16>
      %123 = affine.apply #map12(%arg4)
      %124 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %122 [%116, %123] : f16 into vector<2x8xf16>
      %126 = affine.apply #map13(%arg4)
      %127 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %125 [%116, %126] : f16 into vector<2x8xf16>
      %129 = vector.insert %8, %arg6 [%116, %117] : f16 into vector<2x8xf16>
      %130 = vector.insert %8, %129 [%116, %120] : f16 into vector<2x8xf16>
      %131 = vector.insert %8, %130 [%116, %123] : f16 into vector<2x8xf16>
      %132 = vector.insert %8, %131 [%116, %126] : f16 into vector<2x8xf16>
      %133 = arith.mulf %arg8, %23 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%116, %117] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%116, %120] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%116, %123] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%116, %126] : f16 into vector<2x8xf16>
      affine.yield %128, %132, %141, %115 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %76 = arith.mulf %75#3, %23 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
    %78 = vector.insert %77, %75#2 [1, 4] : f16 into vector<2x8xf16>
    %79 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
    %80 = vector.insert %79, %78 [1, 5] : f16 into vector<2x8xf16>
    %81 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
    %82 = vector.insert %81, %80 [1, 6] : f16 into vector<2x8xf16>
    %83 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
    %84 = vector.insert %83, %82 [1, 7] : f16 into vector<2x8xf16>
    %85:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#3, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %105 = frisk.to_threadTile %cst_6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map2, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %106 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c0 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c0 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      %107 = frisk.async_copy_gm_to_shm %106 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %107 : !frisk.async_task
      %108 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c32 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c32 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %109 = frisk.async_copy_gm_to_shm %108 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %110:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %145 = frisk.to_threadTile %alloc_15 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %146 = affine.apply #map15(%arg11)
        %147 = vector.extract %145[0, %146] : f16 from vector<8x8xf16>
        %148 = vector.extract %145[1, %146] : f16 from vector<8x8xf16>
        %149 = vector.extract %145[2, %146] : f16 from vector<8x8xf16>
        %150 = vector.extract %145[3, %146] : f16 from vector<8x8xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<4x1xf16>
        %152:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %151) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %182 = frisk.to_threadTile %alloc_15 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %183 = affine.apply #map16(%arg14)
          %184 = vector.extract %182[%183, %146] : f16 from vector<8x8xf16>
          %185 = affine.apply #map17(%arg14)
          %186 = vector.extract %182[%185, %146] : f16 from vector<8x8xf16>
          %187 = affine.apply #map18(%arg14)
          %188 = vector.extract %182[%187, %146] : f16 from vector<8x8xf16>
          %189 = affine.apply #map19(%arg14)
          %190 = vector.extract %182[%189, %146] : f16 from vector<8x8xf16>
          %191 = vector.from_elements %184, %186, %188, %190 : vector<4x1xf16>
          %192 = affine.apply #map20(%arg11)
          %193 = affine.apply #map21(%arg14)
          %194 = vector.extract %32[%192, %193] : f16 from vector<2x8xf16>
          %195 = affine.apply #map22(%arg14)
          %196 = vector.extract %32[%192, %195] : f16 from vector<2x8xf16>
          %197 = affine.apply #map23(%arg14)
          %198 = vector.extract %32[%192, %197] : f16 from vector<2x8xf16>
          %199 = affine.apply #map24(%arg14)
          %200 = vector.extract %32[%192, %199] : f16 from vector<2x8xf16>
          %201 = vector.from_elements %194, %196, %198, %200 : vector<1x4xf16>
          %202 = frisk.warp_mma_rr %201, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %202, %191 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %153 = affine.apply #map20(%arg11)
        %154 = vector.extract %32[%153, 4] : f16 from vector<2x8xf16>
        %155 = vector.extract %32[%153, 5] : f16 from vector<2x8xf16>
        %156 = vector.extract %32[%153, 6] : f16 from vector<2x8xf16>
        %157 = vector.extract %32[%153, 7] : f16 from vector<2x8xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = frisk.warp_mma_rr %158, %152#1, %152#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %160 = affine.apply #map25(%arg11)
        %161 = vector.extract %105[%153, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map26(%arg11)
        %163 = vector.extract %105[%153, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map27(%arg11)
        %165 = vector.extract %105[%153, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map28(%arg11)
        %167 = vector.extract %105[%153, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = arith.addf %168, %159 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %170 = vector.extract %169[0, 0] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %arg12 [%153, %160] : f32 into vector<2x32xf32>
        %172 = vector.extract %169[0, 1] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%153, %162] : f32 into vector<2x32xf32>
        %174 = vector.extract %169[0, 2] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%153, %164] : f32 into vector<2x32xf32>
        %176 = vector.extract %169[0, 3] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%153, %166] : f32 into vector<2x32xf32>
        %178 = vector.insert %170, %arg13 [%153, %160] : f32 into vector<2x32xf32>
        %179 = vector.insert %172, %178 [%153, %162] : f32 into vector<2x32xf32>
        %180 = vector.insert %174, %179 [%153, %164] : f32 into vector<2x32xf32>
        %181 = vector.insert %176, %180 [%153, %166] : f32 into vector<2x32xf32>
        affine.yield %177, %181 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %109 : !frisk.async_task
      %111 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c64 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c64 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %112 = frisk.async_copy_gm_to_shm %111 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %113:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %145 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %146 = affine.apply #map15(%arg11)
        %147 = vector.extract %145[0, %146] : f16 from vector<8x8xf16>
        %148 = vector.extract %145[1, %146] : f16 from vector<8x8xf16>
        %149 = vector.extract %145[2, %146] : f16 from vector<8x8xf16>
        %150 = vector.extract %145[3, %146] : f16 from vector<8x8xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<4x1xf16>
        %152:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %151) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %182 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %183 = affine.apply #map16(%arg14)
          %184 = vector.extract %182[%183, %146] : f16 from vector<8x8xf16>
          %185 = affine.apply #map17(%arg14)
          %186 = vector.extract %182[%185, %146] : f16 from vector<8x8xf16>
          %187 = affine.apply #map18(%arg14)
          %188 = vector.extract %182[%187, %146] : f16 from vector<8x8xf16>
          %189 = affine.apply #map19(%arg14)
          %190 = vector.extract %182[%189, %146] : f16 from vector<8x8xf16>
          %191 = vector.from_elements %184, %186, %188, %190 : vector<4x1xf16>
          %192 = affine.apply #map20(%arg11)
          %193 = affine.apply #map21(%arg14)
          %194 = vector.extract %50[%192, %193] : f16 from vector<2x8xf16>
          %195 = affine.apply #map22(%arg14)
          %196 = vector.extract %50[%192, %195] : f16 from vector<2x8xf16>
          %197 = affine.apply #map23(%arg14)
          %198 = vector.extract %50[%192, %197] : f16 from vector<2x8xf16>
          %199 = affine.apply #map24(%arg14)
          %200 = vector.extract %50[%192, %199] : f16 from vector<2x8xf16>
          %201 = vector.from_elements %194, %196, %198, %200 : vector<1x4xf16>
          %202 = frisk.warp_mma_rr %201, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %202, %191 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %153 = affine.apply #map20(%arg11)
        %154 = vector.extract %50[%153, 4] : f16 from vector<2x8xf16>
        %155 = vector.extract %50[%153, 5] : f16 from vector<2x8xf16>
        %156 = vector.extract %50[%153, 6] : f16 from vector<2x8xf16>
        %157 = vector.extract %50[%153, 7] : f16 from vector<2x8xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = frisk.warp_mma_rr %158, %152#1, %152#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %160 = affine.apply #map25(%arg11)
        %161 = vector.extract %110#1[%153, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map26(%arg11)
        %163 = vector.extract %110#1[%153, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map27(%arg11)
        %165 = vector.extract %110#1[%153, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map28(%arg11)
        %167 = vector.extract %110#1[%153, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = arith.addf %168, %159 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %170 = vector.extract %169[0, 0] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %arg12 [%153, %160] : f32 into vector<2x32xf32>
        %172 = vector.extract %169[0, 1] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%153, %162] : f32 into vector<2x32xf32>
        %174 = vector.extract %169[0, 2] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%153, %164] : f32 into vector<2x32xf32>
        %176 = vector.extract %169[0, 3] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%153, %166] : f32 into vector<2x32xf32>
        %178 = vector.insert %170, %arg13 [%153, %160] : f32 into vector<2x32xf32>
        %179 = vector.insert %172, %178 [%153, %162] : f32 into vector<2x32xf32>
        %180 = vector.insert %174, %179 [%153, %164] : f32 into vector<2x32xf32>
        %181 = vector.insert %176, %180 [%153, %166] : f32 into vector<2x32xf32>
        affine.yield %177, %181 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %112 : !frisk.async_task
      %114 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c96 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c96 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %115 = frisk.async_copy_gm_to_shm %114 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %116:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %145 = frisk.to_threadTile %alloc_17 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %146 = affine.apply #map15(%arg11)
        %147 = vector.extract %145[0, %146] : f16 from vector<8x8xf16>
        %148 = vector.extract %145[1, %146] : f16 from vector<8x8xf16>
        %149 = vector.extract %145[2, %146] : f16 from vector<8x8xf16>
        %150 = vector.extract %145[3, %146] : f16 from vector<8x8xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<4x1xf16>
        %152:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %151) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %182 = frisk.to_threadTile %alloc_17 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %183 = affine.apply #map16(%arg14)
          %184 = vector.extract %182[%183, %146] : f16 from vector<8x8xf16>
          %185 = affine.apply #map17(%arg14)
          %186 = vector.extract %182[%185, %146] : f16 from vector<8x8xf16>
          %187 = affine.apply #map18(%arg14)
          %188 = vector.extract %182[%187, %146] : f16 from vector<8x8xf16>
          %189 = affine.apply #map19(%arg14)
          %190 = vector.extract %182[%189, %146] : f16 from vector<8x8xf16>
          %191 = vector.from_elements %184, %186, %188, %190 : vector<4x1xf16>
          %192 = affine.apply #map20(%arg11)
          %193 = affine.apply #map21(%arg14)
          %194 = vector.extract %68[%192, %193] : f16 from vector<2x8xf16>
          %195 = affine.apply #map22(%arg14)
          %196 = vector.extract %68[%192, %195] : f16 from vector<2x8xf16>
          %197 = affine.apply #map23(%arg14)
          %198 = vector.extract %68[%192, %197] : f16 from vector<2x8xf16>
          %199 = affine.apply #map24(%arg14)
          %200 = vector.extract %68[%192, %199] : f16 from vector<2x8xf16>
          %201 = vector.from_elements %194, %196, %198, %200 : vector<1x4xf16>
          %202 = frisk.warp_mma_rr %201, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %202, %191 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %153 = affine.apply #map20(%arg11)
        %154 = vector.extract %68[%153, 4] : f16 from vector<2x8xf16>
        %155 = vector.extract %68[%153, 5] : f16 from vector<2x8xf16>
        %156 = vector.extract %68[%153, 6] : f16 from vector<2x8xf16>
        %157 = vector.extract %68[%153, 7] : f16 from vector<2x8xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = frisk.warp_mma_rr %158, %152#1, %152#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %160 = affine.apply #map25(%arg11)
        %161 = vector.extract %113#1[%153, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map26(%arg11)
        %163 = vector.extract %113#1[%153, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map27(%arg11)
        %165 = vector.extract %113#1[%153, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map28(%arg11)
        %167 = vector.extract %113#1[%153, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = arith.addf %168, %159 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %170 = vector.extract %169[0, 0] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %arg12 [%153, %160] : f32 into vector<2x32xf32>
        %172 = vector.extract %169[0, 1] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%153, %162] : f32 into vector<2x32xf32>
        %174 = vector.extract %169[0, 2] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%153, %164] : f32 into vector<2x32xf32>
        %176 = vector.extract %169[0, 3] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%153, %166] : f32 into vector<2x32xf32>
        %178 = vector.insert %170, %arg13 [%153, %160] : f32 into vector<2x32xf32>
        %179 = vector.insert %172, %178 [%153, %162] : f32 into vector<2x32xf32>
        %180 = vector.insert %174, %179 [%153, %164] : f32 into vector<2x32xf32>
        %181 = vector.insert %176, %180 [%153, %166] : f32 into vector<2x32xf32>
        affine.yield %177, %181 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %115 : !frisk.async_task
      %117 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %118 = frisk.async_copy_gm_to_shm %117 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %119:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %145 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %146 = affine.apply #map15(%arg11)
        %147 = vector.extract %145[0, %146] : f16 from vector<8x8xf16>
        %148 = vector.extract %145[1, %146] : f16 from vector<8x8xf16>
        %149 = vector.extract %145[2, %146] : f16 from vector<8x8xf16>
        %150 = vector.extract %145[3, %146] : f16 from vector<8x8xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<4x1xf16>
        %152:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %151) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %182 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %183 = affine.apply #map16(%arg14)
          %184 = vector.extract %182[%183, %146] : f16 from vector<8x8xf16>
          %185 = affine.apply #map17(%arg14)
          %186 = vector.extract %182[%185, %146] : f16 from vector<8x8xf16>
          %187 = affine.apply #map18(%arg14)
          %188 = vector.extract %182[%187, %146] : f16 from vector<8x8xf16>
          %189 = affine.apply #map19(%arg14)
          %190 = vector.extract %182[%189, %146] : f16 from vector<8x8xf16>
          %191 = vector.from_elements %184, %186, %188, %190 : vector<4x1xf16>
          %192 = affine.apply #map20(%arg11)
          %193 = affine.apply #map21(%arg14)
          %194 = vector.extract %84[%192, %193] : f16 from vector<2x8xf16>
          %195 = affine.apply #map22(%arg14)
          %196 = vector.extract %84[%192, %195] : f16 from vector<2x8xf16>
          %197 = affine.apply #map23(%arg14)
          %198 = vector.extract %84[%192, %197] : f16 from vector<2x8xf16>
          %199 = affine.apply #map24(%arg14)
          %200 = vector.extract %84[%192, %199] : f16 from vector<2x8xf16>
          %201 = vector.from_elements %194, %196, %198, %200 : vector<1x4xf16>
          %202 = frisk.warp_mma_rr %201, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %202, %191 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %153 = affine.apply #map20(%arg11)
        %154 = vector.extract %84[%153, 4] : f16 from vector<2x8xf16>
        %155 = vector.extract %84[%153, 5] : f16 from vector<2x8xf16>
        %156 = vector.extract %84[%153, 6] : f16 from vector<2x8xf16>
        %157 = vector.extract %84[%153, 7] : f16 from vector<2x8xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = frisk.warp_mma_rr %158, %152#1, %152#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %160 = affine.apply #map25(%arg11)
        %161 = vector.extract %116#1[%153, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply #map26(%arg11)
        %163 = vector.extract %116#1[%153, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply #map27(%arg11)
        %165 = vector.extract %116#1[%153, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply #map28(%arg11)
        %167 = vector.extract %116#1[%153, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = arith.addf %168, %159 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %170 = vector.extract %169[0, 0] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %arg12 [%153, %160] : f32 into vector<2x32xf32>
        %172 = vector.extract %169[0, 1] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%153, %162] : f32 into vector<2x32xf32>
        %174 = vector.extract %169[0, 2] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%153, %164] : f32 into vector<2x32xf32>
        %176 = vector.extract %169[0, 3] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%153, %166] : f32 into vector<2x32xf32>
        %178 = vector.insert %170, %arg13 [%153, %160] : f32 into vector<2x32xf32>
        %179 = vector.insert %172, %178 [%153, %162] : f32 into vector<2x32xf32>
        %180 = vector.insert %174, %179 [%153, %164] : f32 into vector<2x32xf32>
        %181 = vector.insert %176, %180 [%153, %166] : f32 into vector<2x32xf32>
        affine.yield %177, %181 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %120 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %145 = affine.apply #map20(%arg11)
        %146 = affine.apply #map25(%arg11)
        %147 = vector.extract %119#1[%145, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply #map26(%arg11)
        %149 = vector.extract %119#1[%145, %148] : f32 from vector<2x32xf32>
        %150 = affine.apply #map27(%arg11)
        %151 = vector.extract %119#1[%145, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map28(%arg11)
        %153 = vector.extract %119#1[%145, %152] : f32 from vector<2x32xf32>
        %154 = vector.insert %147, %arg12 [%145, %146] : f32 into vector<2x32xf32>
        %155 = vector.insert %149, %154 [%145, %148] : f32 into vector<2x32xf32>
        %156 = vector.insert %151, %155 [%145, %150] : f32 into vector<2x32xf32>
        %157 = vector.insert %153, %156 [%145, %152] : f32 into vector<2x32xf32>
        affine.yield %157 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %121 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %145 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_10) -> (f32) {
          %149 = affine.apply #map21(%arg13)
          %150 = vector.extract %120[%arg11, %149] : f32 from vector<2x32xf32>
          %151 = arith.maxnumf %arg14, %150 : f32
          %152 = affine.apply #map22(%arg13)
          %153 = vector.extract %120[%arg11, %152] : f32 from vector<2x32xf32>
          %154 = arith.maxnumf %151, %153 : f32
          %155 = affine.apply #map23(%arg13)
          %156 = vector.extract %120[%arg11, %155] : f32 from vector<2x32xf32>
          %157 = arith.maxnumf %154, %156 : f32
          %158 = affine.apply #map24(%arg13)
          %159 = vector.extract %120[%arg11, %158] : f32 from vector<2x32xf32>
          %160 = arith.maxnumf %157, %159 : f32
          affine.yield %160 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %145, %c16_i32, %c64_i32 : f32
        %146 = arith.maxnumf %145, %shuffleResult : f32
        %shuffleResult_24, %valid_25 = gpu.shuffle xor %146, %c32_i32, %c64_i32 : f32
        %147 = arith.maxnumf %146, %shuffleResult_24 : f32
        %148 = vector.insert %147, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %148 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %122 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %145 = vector.extract %121[%arg11, 0] : f32 from vector<2x1xf32>
        %146 = vector.broadcast %145 : f32 to vector<1x1xf32>
        %147 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %148 = vector.broadcast %147 : f32 to vector<1x1xf32>
        %149 = arith.maxnumf %146, %148 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %150 = vector.extract %149[0, 0] : f32 from vector<1x1xf32>
        %151 = vector.insert %150, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %151 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %123:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %145 = affine.apply #map20(%arg11)
        %146 = vector.extract %122[%145, 0] : f32 from vector<2x1xf32>
        %147 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3) -> (vector<1x4xf32>) {
          %174 = vector.broadcast %146 : f32 to vector<1x4xf32>
          affine.yield %174 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %148 = affine.apply #map25(%arg11)
        %149 = vector.extract %147[0, 0] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %arg12 [%145, %148] : f32 into vector<2x32xf32>
        %151 = affine.apply #map26(%arg11)
        %152 = vector.extract %147[0, 1] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %150 [%145, %151] : f32 into vector<2x32xf32>
        %154 = affine.apply #map27(%arg11)
        %155 = vector.extract %147[0, 2] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%145, %154] : f32 into vector<2x32xf32>
        %157 = affine.apply #map28(%arg11)
        %158 = vector.extract %147[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%145, %157] : f32 into vector<2x32xf32>
        %160 = vector.extract %119#1[%145, %148] : f32 from vector<2x32xf32>
        %161 = vector.extract %119#1[%145, %151] : f32 from vector<2x32xf32>
        %162 = vector.extract %119#1[%145, %154] : f32 from vector<2x32xf32>
        %163 = vector.extract %119#1[%145, %157] : f32 from vector<2x32xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.subf %164, %147 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg13 [%145, %148] : f32 into vector<2x32xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%145, %151] : f32 into vector<2x32xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%145, %154] : f32 into vector<2x32xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%145, %157] : f32 into vector<2x32xf32>
        affine.yield %159, %173 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %124:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9, %arg14 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %145 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %146 = vector.broadcast %145 : f32 to vector<1x1xf32>
        %147 = vector.extract %122[%arg11, 0] : f32 from vector<2x1xf32>
        %148 = vector.broadcast %147 : f32 to vector<1x1xf32>
        %149 = arith.subf %146, %148 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %150 = vector.extract %149[0, 0] : f32 from vector<1x1xf32>
        %151 = vector.insert %150, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %152 = math.exp2 %149 {tiled = true} : vector<1x1xf32>
        %153 = vector.extract %152[0, 0] : f32 from vector<1x1xf32>
        %154 = vector.insert %153, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %155 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %156 = vector.broadcast %155 : f32 to vector<1x1xf32>
        %157 = arith.mulf %156, %152 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %158 = vector.extract %157[0, 0] : f32 from vector<1x1xf32>
        %159 = vector.insert %158, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %151, %154, %159 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %125:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %145 = affine.apply #map20(%arg11)
        %146 = affine.apply #map25(%arg11)
        %147 = vector.extract %123#1[%145, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply #map26(%arg11)
        %149 = vector.extract %123#1[%145, %148] : f32 from vector<2x32xf32>
        %150 = affine.apply #map27(%arg11)
        %151 = vector.extract %123#1[%145, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map28(%arg11)
        %153 = vector.extract %123#1[%145, %152] : f32 from vector<2x32xf32>
        %154 = vector.from_elements %147, %149, %151, %153 : vector<1x4xf32>
        %155 = math.exp2 %154 {tiled = true} : vector<1x4xf32>
        %156 = vector.extract %155[0, 0] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %arg12 [%145, %146] : f32 into vector<2x32xf32>
        %158 = vector.extract %155[0, 1] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%145, %148] : f32 into vector<2x32xf32>
        %160 = vector.extract %155[0, 2] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%145, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %155[0, 3] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%145, %152] : f32 into vector<2x32xf32>
        %164 = vector.insert %156, %arg13 [%145, %146] : f32 into vector<2x32xf32>
        %165 = vector.insert %158, %164 [%145, %148] : f32 into vector<2x32xf32>
        %166 = vector.insert %160, %165 [%145, %150] : f32 into vector<2x32xf32>
        %167 = vector.insert %162, %166 [%145, %152] : f32 into vector<2x32xf32>
        affine.yield %163, %167 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %126 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %145 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst) -> (f32) {
          %149 = affine.apply #map21(%arg13)
          %150 = vector.extract %125#1[%arg11, %149] : f32 from vector<2x32xf32>
          %151 = arith.addf %arg14, %150 : f32
          %152 = affine.apply #map22(%arg13)
          %153 = vector.extract %125#1[%arg11, %152] : f32 from vector<2x32xf32>
          %154 = arith.addf %151, %153 : f32
          %155 = affine.apply #map23(%arg13)
          %156 = vector.extract %125#1[%arg11, %155] : f32 from vector<2x32xf32>
          %157 = arith.addf %154, %156 : f32
          %158 = affine.apply #map24(%arg13)
          %159 = vector.extract %125#1[%arg11, %158] : f32 from vector<2x32xf32>
          %160 = arith.addf %157, %159 : f32
          affine.yield %160 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %145, %c16_i32, %c64_i32 : f32
        %146 = arith.addf %145, %shuffleResult : f32
        %shuffleResult_24, %valid_25 = gpu.shuffle xor %146, %c32_i32, %c64_i32 : f32
        %147 = arith.addf %146, %shuffleResult_24 : f32
        %148 = vector.insert %147, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %148 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %127 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %145 = vector.extract %126[%arg11, 0] : f32 from vector<2x1xf32>
        %146 = vector.broadcast %145 : f32 to vector<1x1xf32>
        %147 = vector.extract %124#2[%arg11, 0] : f32 from vector<2x1xf32>
        %148 = vector.broadcast %147 : f32 to vector<1x1xf32>
        %149 = arith.addf %146, %148 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %150 = vector.extract %149[0, 0] : f32 from vector<1x1xf32>
        %151 = vector.insert %150, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %151 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %128 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %145 = affine.apply #map20(%arg11)
        %146 = affine.apply #map25(%arg11)
        %147 = vector.extract %125#0[%145, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply #map26(%arg11)
        %149 = vector.extract %125#0[%145, %148] : f32 from vector<2x32xf32>
        %150 = affine.apply #map27(%arg11)
        %151 = vector.extract %125#0[%145, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map28(%arg11)
        %153 = vector.extract %125#0[%145, %152] : f32 from vector<2x32xf32>
        %154 = vector.from_elements %147, %149, %151, %153 : vector<1x4xf32>
        %155 = arith.truncf %154 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %156 = vector.extract %155[0, 0] : f16 from vector<1x4xf16>
        %157 = vector.insert %156, %arg12 [%145, %146] : f16 into vector<2x32xf16>
        %158 = vector.extract %155[0, 1] : f16 from vector<1x4xf16>
        %159 = vector.insert %158, %157 [%145, %148] : f16 into vector<2x32xf16>
        %160 = vector.extract %155[0, 2] : f16 from vector<1x4xf16>
        %161 = vector.insert %160, %159 [%145, %150] : f16 into vector<2x32xf16>
        %162 = vector.extract %155[0, 3] : f16 from vector<1x4xf16>
        %163 = vector.insert %162, %161 [%145, %152] : f16 into vector<2x32xf16>
        affine.yield %163 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %thread_id_x_20 = gpu.thread_id x
      frisk.wait_buffer_ready %118 : !frisk.async_task
      %129 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %145 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %146 = affine.apply #map30()[%thread_id_x_20]
        %147 = frisk.read_matrix_from_shm %alloc_19[%145, %146] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %148 = vector.bitcast %147 : vector<4xi32> to vector<8xf16>
        %149 = vector.extract %148[0] : f16 from vector<8xf16>
        %150 = affine.apply #map31()[%thread_id_x_20]
        %151 = vector.extract %148[%150] : f16 from vector<8xf16>
        %152 = vector.extract %148[1] : f16 from vector<8xf16>
        %153 = affine.apply #map32()[%thread_id_x_20]
        %154 = vector.extract %148[%153] : f16 from vector<8xf16>
        %155 = vector.extract %148[2] : f16 from vector<8xf16>
        %156 = affine.apply #map33()[%thread_id_x_20]
        %157 = vector.extract %148[%156] : f16 from vector<8xf16>
        %158 = vector.extract %148[3] : f16 from vector<8xf16>
        %159 = affine.apply #map34()[%thread_id_x_20]
        %160 = vector.extract %148[%159] : f16 from vector<8xf16>
        %161 = affine.apply #map21(%arg11)
        %162 = vector.insert %149, %arg12 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        %164 = affine.apply #map22(%arg11)
        %165 = vector.insert %152, %163 [%164, 0] : f16 into vector<32x2xf16>
        %166 = vector.insert %154, %165 [%164, 1] : f16 into vector<32x2xf16>
        %167 = affine.apply #map23(%arg11)
        %168 = vector.insert %155, %166 [%167, 0] : f16 into vector<32x2xf16>
        %169 = vector.insert %157, %168 [%167, 1] : f16 into vector<32x2xf16>
        %170 = affine.apply #map24(%arg11)
        %171 = vector.insert %158, %169 [%170, 0] : f16 into vector<32x2xf16>
        %172 = vector.insert %160, %171 [%170, 1] : f16 into vector<32x2xf16>
        affine.yield %172 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %130 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %131 = frisk.async_copy_gm_to_shm %130 to %alloc_21 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %132:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %145 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %192 = affine.apply #map9(%arg11)
          %193 = affine.apply #map21(%arg16)
          %194 = vector.extract %128[%192, %193] : f16 from vector<2x32xf16>
          %195 = affine.apply #map22(%arg16)
          %196 = vector.extract %128[%192, %195] : f16 from vector<2x32xf16>
          %197 = affine.apply #map23(%arg16)
          %198 = vector.extract %128[%192, %197] : f16 from vector<2x32xf16>
          %199 = affine.apply #map24(%arg16)
          %200 = vector.extract %128[%192, %199] : f16 from vector<2x32xf16>
          %201 = vector.from_elements %194, %196, %198, %200 : vector<1x4xf16>
          %202 = affine.apply #map35(%arg11)
          %203 = vector.extract %129[%193, %202] : f16 from vector<32x2xf16>
          %204 = vector.extract %129[%195, %202] : f16 from vector<32x2xf16>
          %205 = vector.extract %129[%197, %202] : f16 from vector<32x2xf16>
          %206 = vector.extract %129[%199, %202] : f16 from vector<32x2xf16>
          %207 = vector.from_elements %203, %204, %205, %206 : vector<4x1xf16>
          %208 = frisk.warp_mma_rr %201, %207, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %208 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %146 = affine.apply #map9(%arg11)
        %147 = affine.apply #map10(%arg11)
        %148 = vector.extract %145[0, 0] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %arg12 [%146, %147] : f32 into vector<2x8xf32>
        %150 = affine.apply #map11(%arg11)
        %151 = vector.extract %145[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %149 [%146, %150] : f32 into vector<2x8xf32>
        %153 = affine.apply #map12(%arg11)
        %154 = vector.extract %145[0, 2] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %152 [%146, %153] : f32 into vector<2x8xf32>
        %156 = affine.apply #map13(%arg11)
        %157 = vector.extract %145[0, 3] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %155 [%146, %156] : f32 into vector<2x8xf32>
        %159 = vector.extract %124#1[%146, 0] : f32 from vector<2x1xf32>
        %160 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %192 = vector.broadcast %159 : f32 to vector<1x4xf32>
          affine.yield %192 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %161 = vector.extract %160[0, 0] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %arg13 [%146, %147] : f32 into vector<2x8xf32>
        %163 = vector.extract %160[0, 1] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %162 [%146, %150] : f32 into vector<2x8xf32>
        %165 = vector.extract %160[0, 2] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%146, %153] : f32 into vector<2x8xf32>
        %167 = vector.extract %160[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%146, %156] : f32 into vector<2x8xf32>
        %169 = vector.extract %arg5[%146, %147] : f32 from vector<2x8xf32>
        %170 = vector.extract %arg5[%146, %150] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg5[%146, %153] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg5[%146, %156] : f32 from vector<2x8xf32>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf32>
        %174 = arith.mulf %173, %160 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg14 [%146, %147] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%146, %150] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%146, %153] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%146, %156] : f32 into vector<2x8xf32>
        %183 = arith.addf %174, %145 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %184 = vector.extract %183[0, 0] : f32 from vector<1x4xf32>
        %185 = vector.insert %184, %arg15 [%146, %147] : f32 into vector<2x8xf32>
        %186 = vector.extract %183[0, 1] : f32 from vector<1x4xf32>
        %187 = vector.insert %186, %185 [%146, %150] : f32 into vector<2x8xf32>
        %188 = vector.extract %183[0, 2] : f32 from vector<1x4xf32>
        %189 = vector.insert %188, %187 [%146, %153] : f32 into vector<2x8xf32>
        %190 = vector.extract %183[0, 3] : f32 from vector<1x4xf32>
        %191 = vector.insert %190, %189 [%146, %156] : f32 into vector<2x8xf32>
        affine.yield %158, %168, %182, %191 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %131 : !frisk.async_task
      %133 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %145 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %146 = affine.apply #map30()[%thread_id_x_20]
        %147 = frisk.read_matrix_from_shm %alloc_21[%145, %146] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %148 = vector.bitcast %147 : vector<4xi32> to vector<8xf16>
        %149 = vector.extract %148[0] : f16 from vector<8xf16>
        %150 = affine.apply #map31()[%thread_id_x_20]
        %151 = vector.extract %148[%150] : f16 from vector<8xf16>
        %152 = vector.extract %148[1] : f16 from vector<8xf16>
        %153 = affine.apply #map32()[%thread_id_x_20]
        %154 = vector.extract %148[%153] : f16 from vector<8xf16>
        %155 = vector.extract %148[2] : f16 from vector<8xf16>
        %156 = affine.apply #map33()[%thread_id_x_20]
        %157 = vector.extract %148[%156] : f16 from vector<8xf16>
        %158 = vector.extract %148[3] : f16 from vector<8xf16>
        %159 = affine.apply #map34()[%thread_id_x_20]
        %160 = vector.extract %148[%159] : f16 from vector<8xf16>
        %161 = affine.apply #map21(%arg11)
        %162 = vector.insert %149, %arg12 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        %164 = affine.apply #map22(%arg11)
        %165 = vector.insert %152, %163 [%164, 0] : f16 into vector<32x2xf16>
        %166 = vector.insert %154, %165 [%164, 1] : f16 into vector<32x2xf16>
        %167 = affine.apply #map23(%arg11)
        %168 = vector.insert %155, %166 [%167, 0] : f16 into vector<32x2xf16>
        %169 = vector.insert %157, %168 [%167, 1] : f16 into vector<32x2xf16>
        %170 = affine.apply #map24(%arg11)
        %171 = vector.insert %158, %169 [%170, 0] : f16 into vector<32x2xf16>
        %172 = vector.insert %160, %171 [%170, 1] : f16 into vector<32x2xf16>
        affine.yield %172 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %134 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %alloc_22 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %135 = frisk.async_copy_gm_to_shm %134 to %alloc_22 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %136:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %145 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %192 = affine.apply #map9(%arg11)
          %193 = affine.apply #map21(%arg16)
          %194 = vector.extract %128[%192, %193] : f16 from vector<2x32xf16>
          %195 = affine.apply #map22(%arg16)
          %196 = vector.extract %128[%192, %195] : f16 from vector<2x32xf16>
          %197 = affine.apply #map23(%arg16)
          %198 = vector.extract %128[%192, %197] : f16 from vector<2x32xf16>
          %199 = affine.apply #map24(%arg16)
          %200 = vector.extract %128[%192, %199] : f16 from vector<2x32xf16>
          %201 = vector.from_elements %194, %196, %198, %200 : vector<1x4xf16>
          %202 = affine.apply #map35(%arg11)
          %203 = vector.extract %133[%193, %202] : f16 from vector<32x2xf16>
          %204 = vector.extract %133[%195, %202] : f16 from vector<32x2xf16>
          %205 = vector.extract %133[%197, %202] : f16 from vector<32x2xf16>
          %206 = vector.extract %133[%199, %202] : f16 from vector<32x2xf16>
          %207 = vector.from_elements %203, %204, %205, %206 : vector<4x1xf16>
          %208 = frisk.warp_mma_rr %201, %207, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %208 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %146 = affine.apply #map9(%arg11)
        %147 = affine.apply #map10(%arg11)
        %148 = vector.extract %145[0, 0] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %arg12 [%146, %147] : f32 into vector<2x8xf32>
        %150 = affine.apply #map11(%arg11)
        %151 = vector.extract %145[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %149 [%146, %150] : f32 into vector<2x8xf32>
        %153 = affine.apply #map12(%arg11)
        %154 = vector.extract %145[0, 2] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %152 [%146, %153] : f32 into vector<2x8xf32>
        %156 = affine.apply #map13(%arg11)
        %157 = vector.extract %145[0, 3] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %155 [%146, %156] : f32 into vector<2x8xf32>
        %159 = vector.extract %124#1[%146, 0] : f32 from vector<2x1xf32>
        %160 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %192 = vector.broadcast %159 : f32 to vector<1x4xf32>
          affine.yield %192 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %161 = vector.extract %160[0, 0] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %arg13 [%146, %147] : f32 into vector<2x8xf32>
        %163 = vector.extract %160[0, 1] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %162 [%146, %150] : f32 into vector<2x8xf32>
        %165 = vector.extract %160[0, 2] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%146, %153] : f32 into vector<2x8xf32>
        %167 = vector.extract %160[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%146, %156] : f32 into vector<2x8xf32>
        %169 = vector.extract %arg8[%146, %147] : f32 from vector<2x8xf32>
        %170 = vector.extract %arg8[%146, %150] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg8[%146, %153] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg8[%146, %156] : f32 from vector<2x8xf32>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf32>
        %174 = arith.mulf %173, %160 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg14 [%146, %147] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%146, %150] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%146, %153] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%146, %156] : f32 into vector<2x8xf32>
        %183 = arith.addf %174, %145 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %184 = vector.extract %183[0, 0] : f32 from vector<1x4xf32>
        %185 = vector.insert %184, %arg15 [%146, %147] : f32 into vector<2x8xf32>
        %186 = vector.extract %183[0, 1] : f32 from vector<1x4xf32>
        %187 = vector.insert %186, %185 [%146, %150] : f32 into vector<2x8xf32>
        %188 = vector.extract %183[0, 2] : f32 from vector<1x4xf32>
        %189 = vector.insert %188, %187 [%146, %153] : f32 into vector<2x8xf32>
        %190 = vector.extract %183[0, 3] : f32 from vector<1x4xf32>
        %191 = vector.insert %190, %189 [%146, %156] : f32 into vector<2x8xf32>
        affine.yield %158, %168, %182, %191 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %135 : !frisk.async_task
      %137 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %145 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %146 = affine.apply #map30()[%thread_id_x_20]
        %147 = frisk.read_matrix_from_shm %alloc_22[%145, %146] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %148 = vector.bitcast %147 : vector<4xi32> to vector<8xf16>
        %149 = vector.extract %148[0] : f16 from vector<8xf16>
        %150 = affine.apply #map31()[%thread_id_x_20]
        %151 = vector.extract %148[%150] : f16 from vector<8xf16>
        %152 = vector.extract %148[1] : f16 from vector<8xf16>
        %153 = affine.apply #map32()[%thread_id_x_20]
        %154 = vector.extract %148[%153] : f16 from vector<8xf16>
        %155 = vector.extract %148[2] : f16 from vector<8xf16>
        %156 = affine.apply #map33()[%thread_id_x_20]
        %157 = vector.extract %148[%156] : f16 from vector<8xf16>
        %158 = vector.extract %148[3] : f16 from vector<8xf16>
        %159 = affine.apply #map34()[%thread_id_x_20]
        %160 = vector.extract %148[%159] : f16 from vector<8xf16>
        %161 = affine.apply #map21(%arg11)
        %162 = vector.insert %149, %arg12 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        %164 = affine.apply #map22(%arg11)
        %165 = vector.insert %152, %163 [%164, 0] : f16 into vector<32x2xf16>
        %166 = vector.insert %154, %165 [%164, 1] : f16 into vector<32x2xf16>
        %167 = affine.apply #map23(%arg11)
        %168 = vector.insert %155, %166 [%167, 0] : f16 into vector<32x2xf16>
        %169 = vector.insert %157, %168 [%167, 1] : f16 into vector<32x2xf16>
        %170 = affine.apply #map24(%arg11)
        %171 = vector.insert %158, %169 [%170, 0] : f16 into vector<32x2xf16>
        %172 = vector.insert %160, %171 [%170, 1] : f16 into vector<32x2xf16>
        affine.yield %172 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %138 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %139 = frisk.async_copy_gm_to_shm %138 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %140:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %145 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %192 = affine.apply #map9(%arg11)
          %193 = affine.apply #map21(%arg16)
          %194 = vector.extract %128[%192, %193] : f16 from vector<2x32xf16>
          %195 = affine.apply #map22(%arg16)
          %196 = vector.extract %128[%192, %195] : f16 from vector<2x32xf16>
          %197 = affine.apply #map23(%arg16)
          %198 = vector.extract %128[%192, %197] : f16 from vector<2x32xf16>
          %199 = affine.apply #map24(%arg16)
          %200 = vector.extract %128[%192, %199] : f16 from vector<2x32xf16>
          %201 = vector.from_elements %194, %196, %198, %200 : vector<1x4xf16>
          %202 = affine.apply #map35(%arg11)
          %203 = vector.extract %137[%193, %202] : f16 from vector<32x2xf16>
          %204 = vector.extract %137[%195, %202] : f16 from vector<32x2xf16>
          %205 = vector.extract %137[%197, %202] : f16 from vector<32x2xf16>
          %206 = vector.extract %137[%199, %202] : f16 from vector<32x2xf16>
          %207 = vector.from_elements %203, %204, %205, %206 : vector<4x1xf16>
          %208 = frisk.warp_mma_rr %201, %207, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %208 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %146 = affine.apply #map9(%arg11)
        %147 = affine.apply #map10(%arg11)
        %148 = vector.extract %145[0, 0] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %arg12 [%146, %147] : f32 into vector<2x8xf32>
        %150 = affine.apply #map11(%arg11)
        %151 = vector.extract %145[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %149 [%146, %150] : f32 into vector<2x8xf32>
        %153 = affine.apply #map12(%arg11)
        %154 = vector.extract %145[0, 2] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %152 [%146, %153] : f32 into vector<2x8xf32>
        %156 = affine.apply #map13(%arg11)
        %157 = vector.extract %145[0, 3] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %155 [%146, %156] : f32 into vector<2x8xf32>
        %159 = vector.extract %124#1[%146, 0] : f32 from vector<2x1xf32>
        %160 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %192 = vector.broadcast %159 : f32 to vector<1x4xf32>
          affine.yield %192 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %161 = vector.extract %160[0, 0] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %arg13 [%146, %147] : f32 into vector<2x8xf32>
        %163 = vector.extract %160[0, 1] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %162 [%146, %150] : f32 into vector<2x8xf32>
        %165 = vector.extract %160[0, 2] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%146, %153] : f32 into vector<2x8xf32>
        %167 = vector.extract %160[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%146, %156] : f32 into vector<2x8xf32>
        %169 = vector.extract %arg9[%146, %147] : f32 from vector<2x8xf32>
        %170 = vector.extract %arg9[%146, %150] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg9[%146, %153] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg9[%146, %156] : f32 from vector<2x8xf32>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf32>
        %174 = arith.mulf %173, %160 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg14 [%146, %147] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%146, %150] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%146, %153] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%146, %156] : f32 into vector<2x8xf32>
        %183 = arith.addf %174, %145 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %184 = vector.extract %183[0, 0] : f32 from vector<1x4xf32>
        %185 = vector.insert %184, %arg15 [%146, %147] : f32 into vector<2x8xf32>
        %186 = vector.extract %183[0, 1] : f32 from vector<1x4xf32>
        %187 = vector.insert %186, %185 [%146, %150] : f32 into vector<2x8xf32>
        %188 = vector.extract %183[0, 2] : f32 from vector<1x4xf32>
        %189 = vector.insert %188, %187 [%146, %153] : f32 into vector<2x8xf32>
        %190 = vector.extract %183[0, 3] : f32 from vector<1x4xf32>
        %191 = vector.insert %190, %189 [%146, %156] : f32 into vector<2x8xf32>
        affine.yield %158, %168, %182, %191 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %139 : !frisk.async_task
      %141 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %145 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %146 = affine.apply #map30()[%thread_id_x_20]
        %147 = frisk.read_matrix_from_shm %alloc_23[%145, %146] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %148 = vector.bitcast %147 : vector<4xi32> to vector<8xf16>
        %149 = vector.extract %148[0] : f16 from vector<8xf16>
        %150 = affine.apply #map31()[%thread_id_x_20]
        %151 = vector.extract %148[%150] : f16 from vector<8xf16>
        %152 = vector.extract %148[1] : f16 from vector<8xf16>
        %153 = affine.apply #map32()[%thread_id_x_20]
        %154 = vector.extract %148[%153] : f16 from vector<8xf16>
        %155 = vector.extract %148[2] : f16 from vector<8xf16>
        %156 = affine.apply #map33()[%thread_id_x_20]
        %157 = vector.extract %148[%156] : f16 from vector<8xf16>
        %158 = vector.extract %148[3] : f16 from vector<8xf16>
        %159 = affine.apply #map34()[%thread_id_x_20]
        %160 = vector.extract %148[%159] : f16 from vector<8xf16>
        %161 = affine.apply #map21(%arg11)
        %162 = vector.insert %149, %arg12 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        %164 = affine.apply #map22(%arg11)
        %165 = vector.insert %152, %163 [%164, 0] : f16 into vector<32x2xf16>
        %166 = vector.insert %154, %165 [%164, 1] : f16 into vector<32x2xf16>
        %167 = affine.apply #map23(%arg11)
        %168 = vector.insert %155, %166 [%167, 0] : f16 into vector<32x2xf16>
        %169 = vector.insert %157, %168 [%167, 1] : f16 into vector<32x2xf16>
        %170 = affine.apply #map24(%arg11)
        %171 = vector.insert %158, %169 [%170, 0] : f16 into vector<32x2xf16>
        %172 = vector.insert %160, %171 [%170, 1] : f16 into vector<32x2xf16>
        affine.yield %172 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %142:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %145 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %200 = affine.apply #map9(%arg11)
          %201 = affine.apply #map21(%arg17)
          %202 = vector.extract %128[%200, %201] : f16 from vector<2x32xf16>
          %203 = affine.apply #map22(%arg17)
          %204 = vector.extract %128[%200, %203] : f16 from vector<2x32xf16>
          %205 = affine.apply #map23(%arg17)
          %206 = vector.extract %128[%200, %205] : f16 from vector<2x32xf16>
          %207 = affine.apply #map24(%arg17)
          %208 = vector.extract %128[%200, %207] : f16 from vector<2x32xf16>
          %209 = vector.from_elements %202, %204, %206, %208 : vector<1x4xf16>
          %210 = affine.apply #map35(%arg11)
          %211 = vector.extract %141[%201, %210] : f16 from vector<32x2xf16>
          %212 = vector.extract %141[%203, %210] : f16 from vector<32x2xf16>
          %213 = vector.extract %141[%205, %210] : f16 from vector<32x2xf16>
          %214 = vector.extract %141[%207, %210] : f16 from vector<32x2xf16>
          %215 = vector.from_elements %211, %212, %213, %214 : vector<4x1xf16>
          %216 = frisk.warp_mma_rr %209, %215, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %216 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %146 = affine.apply #map9(%arg11)
        %147 = affine.apply #map10(%arg11)
        %148 = vector.extract %145[0, 0] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %arg12 [%146, %147] : f32 into vector<2x8xf32>
        %150 = affine.apply #map11(%arg11)
        %151 = vector.extract %145[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %149 [%146, %150] : f32 into vector<2x8xf32>
        %153 = affine.apply #map12(%arg11)
        %154 = vector.extract %145[0, 2] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %152 [%146, %153] : f32 into vector<2x8xf32>
        %156 = affine.apply #map13(%arg11)
        %157 = vector.extract %145[0, 3] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %155 [%146, %156] : f32 into vector<2x8xf32>
        %159 = vector.extract %124#1[%146, 0] : f32 from vector<2x1xf32>
        %160 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %200 = vector.broadcast %159 : f32 to vector<1x4xf32>
          affine.yield %200 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %161 = vector.extract %160[0, 0] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %arg13 [%146, %147] : f32 into vector<2x8xf32>
        %163 = vector.extract %160[0, 1] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %162 [%146, %150] : f32 into vector<2x8xf32>
        %165 = vector.extract %160[0, 2] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%146, %153] : f32 into vector<2x8xf32>
        %167 = vector.extract %160[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%146, %156] : f32 into vector<2x8xf32>
        %169 = vector.extract %arg10[%146, %147] : f32 from vector<2x8xf32>
        %170 = vector.extract %arg10[%146, %150] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg10[%146, %153] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg10[%146, %156] : f32 from vector<2x8xf32>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf32>
        %174 = arith.mulf %173, %160 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg14 [%146, %147] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%146, %150] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%146, %153] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%146, %156] : f32 into vector<2x8xf32>
        %183 = arith.addf %174, %145 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %184 = vector.extract %183[0, 0] : f32 from vector<1x4xf32>
        %185 = vector.insert %184, %arg15 [%146, %147] : f32 into vector<2x8xf32>
        %186 = vector.extract %183[0, 1] : f32 from vector<1x4xf32>
        %187 = vector.insert %186, %185 [%146, %150] : f32 into vector<2x8xf32>
        %188 = vector.extract %183[0, 2] : f32 from vector<1x4xf32>
        %189 = vector.insert %188, %187 [%146, %153] : f32 into vector<2x8xf32>
        %190 = vector.extract %183[0, 3] : f32 from vector<1x4xf32>
        %191 = vector.insert %190, %189 [%146, %156] : f32 into vector<2x8xf32>
        %192 = vector.extract %132#3[%146, %147] : f32 from vector<2x8xf32>
        %193 = vector.extract %132#3[%146, %150] : f32 from vector<2x8xf32>
        %194 = vector.extract %132#3[%146, %153] : f32 from vector<2x8xf32>
        %195 = vector.extract %132#3[%146, %156] : f32 from vector<2x8xf32>
        %196 = vector.insert %192, %arg16 [%146, %147] : f32 into vector<2x8xf32>
        %197 = vector.insert %193, %196 [%146, %150] : f32 into vector<2x8xf32>
        %198 = vector.insert %194, %197 [%146, %153] : f32 into vector<2x8xf32>
        %199 = vector.insert %195, %198 [%146, %156] : f32 into vector<2x8xf32>
        affine.yield %158, %168, %182, %191, %199 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %143:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %145 = vector.extract %127[%arg11, 0] : f32 from vector<2x1xf32>
        %146 = vector.insert %145, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %147 = vector.extract %122[%arg11, 0] : f32 from vector<2x1xf32>
        %148 = vector.insert %147, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %146, %148 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %144:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %145 = affine.apply #map9(%arg11)
        %146 = affine.apply #map10(%arg11)
        %147 = vector.extract %136#3[%145, %146] : f32 from vector<2x8xf32>
        %148 = affine.apply #map11(%arg11)
        %149 = vector.extract %136#3[%145, %148] : f32 from vector<2x8xf32>
        %150 = affine.apply #map12(%arg11)
        %151 = vector.extract %136#3[%145, %150] : f32 from vector<2x8xf32>
        %152 = affine.apply #map13(%arg11)
        %153 = vector.extract %136#3[%145, %152] : f32 from vector<2x8xf32>
        %154 = vector.insert %147, %arg12 [%145, %146] : f32 into vector<2x8xf32>
        %155 = vector.insert %149, %154 [%145, %148] : f32 into vector<2x8xf32>
        %156 = vector.insert %151, %155 [%145, %150] : f32 into vector<2x8xf32>
        %157 = vector.insert %153, %156 [%145, %152] : f32 into vector<2x8xf32>
        %158 = vector.extract %140#3[%145, %146] : f32 from vector<2x8xf32>
        %159 = vector.extract %140#3[%145, %148] : f32 from vector<2x8xf32>
        %160 = vector.extract %140#3[%145, %150] : f32 from vector<2x8xf32>
        %161 = vector.extract %140#3[%145, %152] : f32 from vector<2x8xf32>
        %162 = vector.insert %158, %arg13 [%145, %146] : f32 into vector<2x8xf32>
        %163 = vector.insert %159, %162 [%145, %148] : f32 into vector<2x8xf32>
        %164 = vector.insert %160, %163 [%145, %150] : f32 into vector<2x8xf32>
        %165 = vector.insert %161, %164 [%145, %152] : f32 into vector<2x8xf32>
        %166 = vector.extract %142#3[%145, %146] : f32 from vector<2x8xf32>
        %167 = vector.extract %142#3[%145, %148] : f32 from vector<2x8xf32>
        %168 = vector.extract %142#3[%145, %150] : f32 from vector<2x8xf32>
        %169 = vector.extract %142#3[%145, %152] : f32 from vector<2x8xf32>
        %170 = vector.insert %166, %arg14 [%145, %146] : f32 into vector<2x8xf32>
        %171 = vector.insert %167, %170 [%145, %148] : f32 into vector<2x8xf32>
        %172 = vector.insert %168, %171 [%145, %150] : f32 into vector<2x8xf32>
        %173 = vector.insert %169, %172 [%145, %152] : f32 into vector<2x8xf32>
        affine.yield %157, %165, %173 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %142#4, %143#0, %143#1, %144#0, %144#1, %144#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %thread_id_x = gpu.thread_id x
    %86 = arith.index_cast %thread_id_x : index to i32
    %87 = arith.andi %86, %c16_i32 : i32
    %88 = arith.cmpi ne, %87, %c0_i32 : i32
    %89 = arith.andi %86, %c32_i32 : i32
    %90 = arith.cmpi ne, %89, %c0_i32 : i32
    %91 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %105 = affine.apply #map10(%arg4)
      %106 = affine.apply #map9(%arg4)
      %107 = vector.extract %85#5[%106, %105] : f32 from vector<2x8xf32>
      %108 = affine.apply #map11(%arg4)
      %109 = vector.extract %85#5[%106, %108] : f32 from vector<2x8xf32>
      %110 = affine.apply #map12(%arg4)
      %111 = vector.extract %85#5[%106, %110] : f32 from vector<2x8xf32>
      %112 = affine.apply #map13(%arg4)
      %113 = vector.extract %85#5[%106, %112] : f32 from vector<2x8xf32>
      %114 = arith.select %88, %107, %109 : f32
      %shuffleResult, %valid = gpu.shuffle xor %114, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %88, %shuffleResult, %107 : f32
      %116 = arith.select %88, %109, %shuffleResult : f32
      %117 = arith.select %88, %111, %113 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %88, %shuffleResult_15, %111 : f32
      %119 = arith.select %88, %113, %shuffleResult_15 : f32
      %120 = arith.select %90, %115, %118 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %120, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %90, %shuffleResult_17, %115 : f32
      %122 = arith.select %90, %118, %shuffleResult_17 : f32
      %123 = arith.select %90, %116, %119 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %90, %shuffleResult_19, %116 : f32
      %125 = arith.select %90, %119, %shuffleResult_19 : f32
      %126 = vector.insert %121, %arg5 [%106, %105] : f32 into vector<2x8xf32>
      %127 = vector.insert %124, %126 [%106, %108] : f32 into vector<2x8xf32>
      %128 = vector.insert %122, %127 [%106, %110] : f32 into vector<2x8xf32>
      %129 = vector.insert %125, %128 [%106, %112] : f32 into vector<2x8xf32>
      affine.yield %129 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %92 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %105 = affine.apply #map10(%arg4)
      %106 = affine.apply #map9(%arg4)
      %107 = vector.extract %85#4[%106, %105] : f32 from vector<2x8xf32>
      %108 = affine.apply #map11(%arg4)
      %109 = vector.extract %85#4[%106, %108] : f32 from vector<2x8xf32>
      %110 = affine.apply #map12(%arg4)
      %111 = vector.extract %85#4[%106, %110] : f32 from vector<2x8xf32>
      %112 = affine.apply #map13(%arg4)
      %113 = vector.extract %85#4[%106, %112] : f32 from vector<2x8xf32>
      %114 = arith.select %88, %107, %109 : f32
      %shuffleResult, %valid = gpu.shuffle xor %114, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %88, %shuffleResult, %107 : f32
      %116 = arith.select %88, %109, %shuffleResult : f32
      %117 = arith.select %88, %111, %113 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %88, %shuffleResult_15, %111 : f32
      %119 = arith.select %88, %113, %shuffleResult_15 : f32
      %120 = arith.select %90, %115, %118 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %120, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %90, %shuffleResult_17, %115 : f32
      %122 = arith.select %90, %118, %shuffleResult_17 : f32
      %123 = arith.select %90, %116, %119 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %90, %shuffleResult_19, %116 : f32
      %125 = arith.select %90, %119, %shuffleResult_19 : f32
      %126 = vector.insert %121, %arg5 [%106, %105] : f32 into vector<2x8xf32>
      %127 = vector.insert %124, %126 [%106, %108] : f32 into vector<2x8xf32>
      %128 = vector.insert %122, %127 [%106, %110] : f32 into vector<2x8xf32>
      %129 = vector.insert %125, %128 [%106, %112] : f32 into vector<2x8xf32>
      affine.yield %129 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %93 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %105 = affine.apply #map10(%arg4)
      %106 = affine.apply #map9(%arg4)
      %107 = vector.extract %85#3[%106, %105] : f32 from vector<2x8xf32>
      %108 = affine.apply #map11(%arg4)
      %109 = vector.extract %85#3[%106, %108] : f32 from vector<2x8xf32>
      %110 = affine.apply #map12(%arg4)
      %111 = vector.extract %85#3[%106, %110] : f32 from vector<2x8xf32>
      %112 = affine.apply #map13(%arg4)
      %113 = vector.extract %85#3[%106, %112] : f32 from vector<2x8xf32>
      %114 = arith.select %88, %107, %109 : f32
      %shuffleResult, %valid = gpu.shuffle xor %114, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %88, %shuffleResult, %107 : f32
      %116 = arith.select %88, %109, %shuffleResult : f32
      %117 = arith.select %88, %111, %113 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %88, %shuffleResult_15, %111 : f32
      %119 = arith.select %88, %113, %shuffleResult_15 : f32
      %120 = arith.select %90, %115, %118 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %120, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %90, %shuffleResult_17, %115 : f32
      %122 = arith.select %90, %118, %shuffleResult_17 : f32
      %123 = arith.select %90, %116, %119 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %90, %shuffleResult_19, %116 : f32
      %125 = arith.select %90, %119, %shuffleResult_19 : f32
      %126 = vector.insert %121, %arg5 [%106, %105] : f32 into vector<2x8xf32>
      %127 = vector.insert %124, %126 [%106, %108] : f32 into vector<2x8xf32>
      %128 = vector.insert %122, %127 [%106, %110] : f32 into vector<2x8xf32>
      %129 = vector.insert %125, %128 [%106, %112] : f32 into vector<2x8xf32>
      affine.yield %129 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %94 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %105 = affine.apply #map10(%arg4)
      %106 = affine.apply #map9(%arg4)
      %107 = vector.extract %85#0[%106, %105] : f32 from vector<2x8xf32>
      %108 = affine.apply #map11(%arg4)
      %109 = vector.extract %85#0[%106, %108] : f32 from vector<2x8xf32>
      %110 = affine.apply #map12(%arg4)
      %111 = vector.extract %85#0[%106, %110] : f32 from vector<2x8xf32>
      %112 = affine.apply #map13(%arg4)
      %113 = vector.extract %85#0[%106, %112] : f32 from vector<2x8xf32>
      %114 = arith.select %88, %107, %109 : f32
      %shuffleResult, %valid = gpu.shuffle xor %114, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %88, %shuffleResult, %107 : f32
      %116 = arith.select %88, %109, %shuffleResult : f32
      %117 = arith.select %88, %111, %113 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %88, %shuffleResult_15, %111 : f32
      %119 = arith.select %88, %113, %shuffleResult_15 : f32
      %120 = arith.select %90, %115, %118 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %120, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %90, %shuffleResult_17, %115 : f32
      %122 = arith.select %90, %118, %shuffleResult_17 : f32
      %123 = arith.select %90, %116, %119 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %90, %shuffleResult_19, %116 : f32
      %125 = arith.select %90, %119, %shuffleResult_19 : f32
      %126 = vector.insert %121, %arg5 [%106, %105] : f32 into vector<2x8xf32>
      %127 = vector.insert %124, %126 [%106, %108] : f32 into vector<2x8xf32>
      %128 = vector.insert %122, %127 [%106, %110] : f32 into vector<2x8xf32>
      %129 = vector.insert %125, %128 [%106, %112] : f32 into vector<2x8xf32>
      affine.yield %129 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %95 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %96:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %105 = affine.apply #map9(%arg4)
      %106 = vector.extract %85#1[%105, 0] : f32 from vector<2x1xf32>
      %107 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %147 = vector.broadcast %106 : f32 to vector<1x4xf32>
        affine.yield %147 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %108 = affine.apply #map10(%arg4)
      %109 = vector.extract %107[0, 0] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %arg5 [%105, %108] : f32 into vector<2x8xf32>
      %111 = affine.apply #map11(%arg4)
      %112 = vector.extract %107[0, 1] : f32 from vector<1x4xf32>
      %113 = vector.insert %112, %110 [%105, %111] : f32 into vector<2x8xf32>
      %114 = affine.apply #map12(%arg4)
      %115 = vector.extract %107[0, 2] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %113 [%105, %114] : f32 into vector<2x8xf32>
      %117 = affine.apply #map13(%arg4)
      %118 = vector.extract %107[0, 3] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %116 [%105, %117] : f32 into vector<2x8xf32>
      %120 = vector.extract %94[%105, %108] : f32 from vector<2x8xf32>
      %121 = vector.extract %94[%105, %111] : f32 from vector<2x8xf32>
      %122 = vector.extract %94[%105, %114] : f32 from vector<2x8xf32>
      %123 = vector.extract %94[%105, %117] : f32 from vector<2x8xf32>
      %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
      %125 = arith.divf %124, %107 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %126 = vector.extract %125[0, 0] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %arg6 [%105, %108] : f32 into vector<2x8xf32>
      %128 = vector.extract %125[0, 1] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%105, %111] : f32 into vector<2x8xf32>
      %130 = vector.extract %125[0, 2] : f32 from vector<1x4xf32>
      %131 = vector.insert %130, %129 [%105, %114] : f32 into vector<2x8xf32>
      %132 = vector.extract %125[0, 3] : f32 from vector<1x4xf32>
      %133 = vector.insert %132, %131 [%105, %117] : f32 into vector<2x8xf32>
      %134 = arith.truncf %125 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%105, %108] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%105, %111] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%105, %114] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%105, %117] : f16 into vector<2x8xf16>
      %143 = vector.insert %135, %arg8 [%105, %108] : f16 into vector<2x8xf16>
      %144 = vector.insert %137, %143 [%105, %111] : f16 into vector<2x8xf16>
      %145 = vector.insert %139, %144 [%105, %114] : f16 into vector<2x8xf16>
      %146 = vector.insert %141, %145 [%105, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %133, %142, %146 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %97 = affine.apply #map36()[%thread_id_x]
    %98 = arith.cmpi ult, %97, %c4 : index
    scf.if %98 {
      affine.for %arg4 = 0 to 4 {
        %105 = affine.apply #map9(%arg4)
        %106 = affine.apply #map10(%arg4)
        %107 = vector.extract %96#3[%105, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply #map11(%arg4)
        %109 = vector.extract %96#3[%105, %108] : f16 from vector<2x8xf16>
        %110 = affine.apply #map12(%arg4)
        %111 = vector.extract %96#3[%105, %110] : f16 from vector<2x8xf16>
        %112 = affine.apply #map13(%arg4)
        %113 = vector.extract %96#3[%105, %112] : f16 from vector<2x8xf16>
        affine.store %107, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %109, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %111, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %113, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %99 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %100:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %105 = affine.apply #map9(%arg4)
      %106 = vector.extract %85#1[%105, 0] : f32 from vector<2x1xf32>
      %107 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %147 = vector.broadcast %106 : f32 to vector<1x4xf32>
        affine.yield %147 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %108 = affine.apply #map10(%arg4)
      %109 = vector.extract %107[0, 0] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %arg5 [%105, %108] : f32 into vector<2x8xf32>
      %111 = affine.apply #map11(%arg4)
      %112 = vector.extract %107[0, 1] : f32 from vector<1x4xf32>
      %113 = vector.insert %112, %110 [%105, %111] : f32 into vector<2x8xf32>
      %114 = affine.apply #map12(%arg4)
      %115 = vector.extract %107[0, 2] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %113 [%105, %114] : f32 into vector<2x8xf32>
      %117 = affine.apply #map13(%arg4)
      %118 = vector.extract %107[0, 3] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %116 [%105, %117] : f32 into vector<2x8xf32>
      %120 = vector.extract %93[%105, %108] : f32 from vector<2x8xf32>
      %121 = vector.extract %93[%105, %111] : f32 from vector<2x8xf32>
      %122 = vector.extract %93[%105, %114] : f32 from vector<2x8xf32>
      %123 = vector.extract %93[%105, %117] : f32 from vector<2x8xf32>
      %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
      %125 = arith.divf %124, %107 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %126 = vector.extract %125[0, 0] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %arg6 [%105, %108] : f32 into vector<2x8xf32>
      %128 = vector.extract %125[0, 1] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%105, %111] : f32 into vector<2x8xf32>
      %130 = vector.extract %125[0, 2] : f32 from vector<1x4xf32>
      %131 = vector.insert %130, %129 [%105, %114] : f32 into vector<2x8xf32>
      %132 = vector.extract %125[0, 3] : f32 from vector<1x4xf32>
      %133 = vector.insert %132, %131 [%105, %117] : f32 into vector<2x8xf32>
      %134 = arith.truncf %125 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%105, %108] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%105, %111] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%105, %114] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%105, %117] : f16 into vector<2x8xf16>
      %143 = vector.insert %135, %arg8 [%105, %108] : f16 into vector<2x8xf16>
      %144 = vector.insert %137, %143 [%105, %111] : f16 into vector<2x8xf16>
      %145 = vector.insert %139, %144 [%105, %114] : f16 into vector<2x8xf16>
      %146 = vector.insert %141, %145 [%105, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %133, %142, %146 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %98 {
      affine.for %arg4 = 0 to 4 {
        %105 = affine.apply #map9(%arg4)
        %106 = affine.apply #map10(%arg4)
        %107 = vector.extract %100#3[%105, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply #map11(%arg4)
        %109 = vector.extract %100#3[%105, %108] : f16 from vector<2x8xf16>
        %110 = affine.apply #map12(%arg4)
        %111 = vector.extract %100#3[%105, %110] : f16 from vector<2x8xf16>
        %112 = affine.apply #map13(%arg4)
        %113 = vector.extract %100#3[%105, %112] : f16 from vector<2x8xf16>
        affine.store %107, %99[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %109, %99[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %111, %99[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %113, %99[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %101 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %102:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %105 = affine.apply #map9(%arg4)
      %106 = vector.extract %85#1[%105, 0] : f32 from vector<2x1xf32>
      %107 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %147 = vector.broadcast %106 : f32 to vector<1x4xf32>
        affine.yield %147 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %108 = affine.apply #map10(%arg4)
      %109 = vector.extract %107[0, 0] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %arg5 [%105, %108] : f32 into vector<2x8xf32>
      %111 = affine.apply #map11(%arg4)
      %112 = vector.extract %107[0, 1] : f32 from vector<1x4xf32>
      %113 = vector.insert %112, %110 [%105, %111] : f32 into vector<2x8xf32>
      %114 = affine.apply #map12(%arg4)
      %115 = vector.extract %107[0, 2] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %113 [%105, %114] : f32 into vector<2x8xf32>
      %117 = affine.apply #map13(%arg4)
      %118 = vector.extract %107[0, 3] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %116 [%105, %117] : f32 into vector<2x8xf32>
      %120 = vector.extract %92[%105, %108] : f32 from vector<2x8xf32>
      %121 = vector.extract %92[%105, %111] : f32 from vector<2x8xf32>
      %122 = vector.extract %92[%105, %114] : f32 from vector<2x8xf32>
      %123 = vector.extract %92[%105, %117] : f32 from vector<2x8xf32>
      %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
      %125 = arith.divf %124, %107 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %126 = vector.extract %125[0, 0] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %arg6 [%105, %108] : f32 into vector<2x8xf32>
      %128 = vector.extract %125[0, 1] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%105, %111] : f32 into vector<2x8xf32>
      %130 = vector.extract %125[0, 2] : f32 from vector<1x4xf32>
      %131 = vector.insert %130, %129 [%105, %114] : f32 into vector<2x8xf32>
      %132 = vector.extract %125[0, 3] : f32 from vector<1x4xf32>
      %133 = vector.insert %132, %131 [%105, %117] : f32 into vector<2x8xf32>
      %134 = arith.truncf %125 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%105, %108] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%105, %111] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%105, %114] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%105, %117] : f16 into vector<2x8xf16>
      %143 = vector.insert %135, %arg8 [%105, %108] : f16 into vector<2x8xf16>
      %144 = vector.insert %137, %143 [%105, %111] : f16 into vector<2x8xf16>
      %145 = vector.insert %139, %144 [%105, %114] : f16 into vector<2x8xf16>
      %146 = vector.insert %141, %145 [%105, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %133, %142, %146 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %98 {
      affine.for %arg4 = 0 to 4 {
        %105 = affine.apply #map9(%arg4)
        %106 = affine.apply #map10(%arg4)
        %107 = vector.extract %102#3[%105, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply #map11(%arg4)
        %109 = vector.extract %102#3[%105, %108] : f16 from vector<2x8xf16>
        %110 = affine.apply #map12(%arg4)
        %111 = vector.extract %102#3[%105, %110] : f16 from vector<2x8xf16>
        %112 = affine.apply #map13(%arg4)
        %113 = vector.extract %102#3[%105, %112] : f16 from vector<2x8xf16>
        affine.store %107, %101[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %109, %101[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %111, %101[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %113, %101[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %103 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %104:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %105 = affine.apply #map9(%arg4)
      %106 = vector.extract %85#1[%105, 0] : f32 from vector<2x1xf32>
      %107 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %147 = vector.broadcast %106 : f32 to vector<1x4xf32>
        affine.yield %147 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %108 = affine.apply #map10(%arg4)
      %109 = vector.extract %107[0, 0] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %arg5 [%105, %108] : f32 into vector<2x8xf32>
      %111 = affine.apply #map11(%arg4)
      %112 = vector.extract %107[0, 1] : f32 from vector<1x4xf32>
      %113 = vector.insert %112, %110 [%105, %111] : f32 into vector<2x8xf32>
      %114 = affine.apply #map12(%arg4)
      %115 = vector.extract %107[0, 2] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %113 [%105, %114] : f32 into vector<2x8xf32>
      %117 = affine.apply #map13(%arg4)
      %118 = vector.extract %107[0, 3] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %116 [%105, %117] : f32 into vector<2x8xf32>
      %120 = vector.extract %91[%105, %108] : f32 from vector<2x8xf32>
      %121 = vector.extract %91[%105, %111] : f32 from vector<2x8xf32>
      %122 = vector.extract %91[%105, %114] : f32 from vector<2x8xf32>
      %123 = vector.extract %91[%105, %117] : f32 from vector<2x8xf32>
      %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
      %125 = arith.divf %124, %107 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %126 = vector.extract %125[0, 0] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %arg6 [%105, %108] : f32 into vector<2x8xf32>
      %128 = vector.extract %125[0, 1] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%105, %111] : f32 into vector<2x8xf32>
      %130 = vector.extract %125[0, 2] : f32 from vector<1x4xf32>
      %131 = vector.insert %130, %129 [%105, %114] : f32 into vector<2x8xf32>
      %132 = vector.extract %125[0, 3] : f32 from vector<1x4xf32>
      %133 = vector.insert %132, %131 [%105, %117] : f32 into vector<2x8xf32>
      %134 = arith.truncf %125 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%105, %108] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%105, %111] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%105, %114] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%105, %117] : f16 into vector<2x8xf16>
      %143 = vector.insert %135, %arg8 [%105, %108] : f16 into vector<2x8xf16>
      %144 = vector.insert %137, %143 [%105, %111] : f16 into vector<2x8xf16>
      %145 = vector.insert %139, %144 [%105, %114] : f16 into vector<2x8xf16>
      %146 = vector.insert %141, %145 [%105, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %133, %142, %146 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %98 {
      affine.for %arg4 = 0 to 4 {
        %105 = affine.apply #map9(%arg4)
        %106 = affine.apply #map10(%arg4)
        %107 = vector.extract %104#3[%105, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply #map11(%arg4)
        %109 = vector.extract %104#3[%105, %108] : f16 from vector<2x8xf16>
        %110 = affine.apply #map12(%arg4)
        %111 = vector.extract %104#3[%105, %110] : f16 from vector<2x8xf16>
        %112 = affine.apply #map13(%arg4)
        %113 = vector.extract %104#3[%105, %112] : f16 from vector<2x8xf16>
        affine.store %107, %103[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %109, %103[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %111, %103[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %113, %103[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

