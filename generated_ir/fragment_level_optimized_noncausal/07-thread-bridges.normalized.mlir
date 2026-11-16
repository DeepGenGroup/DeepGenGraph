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
    %c96 = arith.constant 96 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
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
    %9:3 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_9, %arg6 = %cst_9, %arg7 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %97 = vector.insert %cst_2, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %98 = vector.insert %cst_10, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %99 = vector.insert %cst_10, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %97, %98, %99 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = frisk.to_threadTile %cst_8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %11 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map2, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %12 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c0) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c0 floordiv 128) mod 4096, %c0 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %12 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
    %13 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %14 = vector.extract %13[0, 0] : f16 from vector<2x8xf16>
    %15 = vector.extract %13[0, 1] : f16 from vector<2x8xf16>
    %16 = vector.extract %13[0, 2] : f16 from vector<2x8xf16>
    %17 = vector.extract %13[0, 3] : f16 from vector<2x8xf16>
    %18 = vector.from_elements %14, %15, %16, %17 : vector<1x4xf16>
    %19:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %18) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %97 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %98 = affine.apply #map4(%arg4)
      %99 = affine.apply #map5(%arg4)
      %100 = vector.extract %97[%98, %99] : f16 from vector<2x8xf16>
      %101 = affine.apply #map6(%arg4)
      %102 = vector.extract %97[%98, %101] : f16 from vector<2x8xf16>
      %103 = affine.apply #map7(%arg4)
      %104 = vector.extract %97[%98, %103] : f16 from vector<2x8xf16>
      %105 = affine.apply #map8(%arg4)
      %106 = vector.extract %97[%98, %105] : f16 from vector<2x8xf16>
      %107 = vector.from_elements %100, %102, %104, %106 : vector<1x4xf16>
      %108 = affine.apply #map9(%arg4)
      %109 = affine.apply #map10(%arg4)
      %110 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %111 = vector.insert %110, %arg5 [%108, %109] : f16 into vector<2x8xf16>
      %112 = affine.apply #map11(%arg4)
      %113 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %111 [%108, %112] : f16 into vector<2x8xf16>
      %115 = affine.apply #map12(%arg4)
      %116 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %114 [%108, %115] : f16 into vector<2x8xf16>
      %118 = affine.apply #map13(%arg4)
      %119 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %117 [%108, %118] : f16 into vector<2x8xf16>
      %121 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %122 = arith.mulf %arg7, %121 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %123 = vector.extract %122[0, 0] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %arg6 [%108, %109] : f16 into vector<2x8xf16>
      %125 = vector.extract %122[0, 1] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%108, %112] : f16 into vector<2x8xf16>
      %127 = vector.extract %122[0, 2] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%108, %115] : f16 into vector<2x8xf16>
      %129 = vector.extract %122[0, 3] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%108, %118] : f16 into vector<2x8xf16>
      affine.yield %120, %130, %107 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %20 = vector.broadcast %8 : f16 to vector<1x4xf16>
    %21 = arith.mulf %19#2, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %22 = vector.extract %21[0, 0] : f16 from vector<1x4xf16>
    %23 = vector.insert %22, %19#1 [1, 4] : f16 into vector<2x8xf16>
    %24 = vector.extract %21[0, 1] : f16 from vector<1x4xf16>
    %25 = vector.insert %24, %23 [1, 5] : f16 into vector<2x8xf16>
    %26 = vector.extract %21[0, 2] : f16 from vector<1x4xf16>
    %27 = vector.insert %26, %25 [1, 6] : f16 into vector<2x8xf16>
    %28 = vector.extract %21[0, 3] : f16 from vector<1x4xf16>
    %29 = vector.insert %28, %27 [1, 7] : f16 into vector<2x8xf16>
    %alloc_12 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %30 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c32) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c32 floordiv 128) mod 4096, %c32 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %30 to %alloc_12 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_12 : memref<128x32xf16, 3>
    %31 = frisk.to_threadTile %alloc_12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %32 = vector.extract %31[0, 0] : f16 from vector<2x8xf16>
    %33 = vector.extract %31[0, 1] : f16 from vector<2x8xf16>
    %34 = vector.extract %31[0, 2] : f16 from vector<2x8xf16>
    %35 = vector.extract %31[0, 3] : f16 from vector<2x8xf16>
    %36 = vector.from_elements %32, %33, %34, %35 : vector<1x4xf16>
    %37:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %36) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %97 = frisk.to_threadTile %alloc_12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %98 = affine.apply #map4(%arg4)
      %99 = affine.apply #map5(%arg4)
      %100 = vector.extract %97[%98, %99] : f16 from vector<2x8xf16>
      %101 = affine.apply #map6(%arg4)
      %102 = vector.extract %97[%98, %101] : f16 from vector<2x8xf16>
      %103 = affine.apply #map7(%arg4)
      %104 = vector.extract %97[%98, %103] : f16 from vector<2x8xf16>
      %105 = affine.apply #map8(%arg4)
      %106 = vector.extract %97[%98, %105] : f16 from vector<2x8xf16>
      %107 = vector.from_elements %100, %102, %104, %106 : vector<1x4xf16>
      %108 = affine.apply #map9(%arg4)
      %109 = affine.apply #map10(%arg4)
      %110 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %111 = vector.insert %110, %arg5 [%108, %109] : f16 into vector<2x8xf16>
      %112 = affine.apply #map11(%arg4)
      %113 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %111 [%108, %112] : f16 into vector<2x8xf16>
      %115 = affine.apply #map12(%arg4)
      %116 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %114 [%108, %115] : f16 into vector<2x8xf16>
      %118 = affine.apply #map13(%arg4)
      %119 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %117 [%108, %118] : f16 into vector<2x8xf16>
      %121 = arith.mulf %arg7, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %122 = vector.extract %121[0, 0] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %arg6 [%108, %109] : f16 into vector<2x8xf16>
      %124 = vector.extract %121[0, 1] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %123 [%108, %112] : f16 into vector<2x8xf16>
      %126 = vector.extract %121[0, 2] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %125 [%108, %115] : f16 into vector<2x8xf16>
      %128 = vector.extract %121[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%108, %118] : f16 into vector<2x8xf16>
      affine.yield %120, %129, %107 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %38 = arith.mulf %37#2, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %39 = vector.extract %38[0, 0] : f16 from vector<1x4xf16>
    %40 = vector.insert %39, %37#1 [1, 4] : f16 into vector<2x8xf16>
    %41 = vector.extract %38[0, 1] : f16 from vector<1x4xf16>
    %42 = vector.insert %41, %40 [1, 5] : f16 into vector<2x8xf16>
    %43 = vector.extract %38[0, 2] : f16 from vector<1x4xf16>
    %44 = vector.insert %43, %42 [1, 6] : f16 into vector<2x8xf16>
    %45 = vector.extract %38[0, 3] : f16 from vector<1x4xf16>
    %46 = vector.insert %45, %44 [1, 7] : f16 into vector<2x8xf16>
    %alloc_13 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %47 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c64) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c64 floordiv 128) mod 4096, %c64 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %47 to %alloc_13 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_13 : memref<128x32xf16, 3>
    %48 = frisk.to_threadTile %alloc_13 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %49 = vector.extract %48[0, 0] : f16 from vector<2x8xf16>
    %50 = vector.extract %48[0, 1] : f16 from vector<2x8xf16>
    %51 = vector.extract %48[0, 2] : f16 from vector<2x8xf16>
    %52 = vector.extract %48[0, 3] : f16 from vector<2x8xf16>
    %53 = vector.from_elements %49, %50, %51, %52 : vector<1x4xf16>
    %54:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %53) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %97 = frisk.to_threadTile %alloc_13 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %98 = affine.apply #map4(%arg4)
      %99 = affine.apply #map5(%arg4)
      %100 = vector.extract %97[%98, %99] : f16 from vector<2x8xf16>
      %101 = affine.apply #map6(%arg4)
      %102 = vector.extract %97[%98, %101] : f16 from vector<2x8xf16>
      %103 = affine.apply #map7(%arg4)
      %104 = vector.extract %97[%98, %103] : f16 from vector<2x8xf16>
      %105 = affine.apply #map8(%arg4)
      %106 = vector.extract %97[%98, %105] : f16 from vector<2x8xf16>
      %107 = vector.from_elements %100, %102, %104, %106 : vector<1x4xf16>
      %108 = affine.apply #map9(%arg4)
      %109 = affine.apply #map10(%arg4)
      %110 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %111 = vector.insert %110, %arg5 [%108, %109] : f16 into vector<2x8xf16>
      %112 = affine.apply #map11(%arg4)
      %113 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %111 [%108, %112] : f16 into vector<2x8xf16>
      %115 = affine.apply #map12(%arg4)
      %116 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %114 [%108, %115] : f16 into vector<2x8xf16>
      %118 = affine.apply #map13(%arg4)
      %119 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %117 [%108, %118] : f16 into vector<2x8xf16>
      %121 = arith.mulf %arg7, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %122 = vector.extract %121[0, 0] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %arg6 [%108, %109] : f16 into vector<2x8xf16>
      %124 = vector.extract %121[0, 1] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %123 [%108, %112] : f16 into vector<2x8xf16>
      %126 = vector.extract %121[0, 2] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %125 [%108, %115] : f16 into vector<2x8xf16>
      %128 = vector.extract %121[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%108, %118] : f16 into vector<2x8xf16>
      affine.yield %120, %129, %107 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %55 = arith.mulf %54#2, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
    %57 = vector.insert %56, %54#1 [1, 4] : f16 into vector<2x8xf16>
    %58 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
    %59 = vector.insert %58, %57 [1, 5] : f16 into vector<2x8xf16>
    %60 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
    %61 = vector.insert %60, %59 [1, 6] : f16 into vector<2x8xf16>
    %62 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
    %63 = vector.insert %62, %61 [1, 7] : f16 into vector<2x8xf16>
    %alloc_14 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %64 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c96) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c96 floordiv 128) mod 4096, %c96 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %64 to %alloc_14 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_14 : memref<128x32xf16, 3>
    %65 = frisk.to_threadTile %alloc_14 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %66 = vector.extract %65[0, 0] : f16 from vector<2x8xf16>
    %67 = vector.extract %65[0, 1] : f16 from vector<2x8xf16>
    %68 = vector.extract %65[0, 2] : f16 from vector<2x8xf16>
    %69 = vector.extract %65[0, 3] : f16 from vector<2x8xf16>
    %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf16>
    %71:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %70) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %97 = frisk.to_threadTile %alloc_14 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %98 = affine.apply #map4(%arg4)
      %99 = affine.apply #map5(%arg4)
      %100 = vector.extract %97[%98, %99] : f16 from vector<2x8xf16>
      %101 = affine.apply #map6(%arg4)
      %102 = vector.extract %97[%98, %101] : f16 from vector<2x8xf16>
      %103 = affine.apply #map7(%arg4)
      %104 = vector.extract %97[%98, %103] : f16 from vector<2x8xf16>
      %105 = affine.apply #map8(%arg4)
      %106 = vector.extract %97[%98, %105] : f16 from vector<2x8xf16>
      %107 = vector.from_elements %100, %102, %104, %106 : vector<1x4xf16>
      %108 = affine.apply #map9(%arg4)
      %109 = affine.apply #map10(%arg4)
      %110 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %111 = vector.insert %110, %arg5 [%108, %109] : f16 into vector<2x8xf16>
      %112 = affine.apply #map11(%arg4)
      %113 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %111 [%108, %112] : f16 into vector<2x8xf16>
      %115 = affine.apply #map12(%arg4)
      %116 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %114 [%108, %115] : f16 into vector<2x8xf16>
      %118 = affine.apply #map13(%arg4)
      %119 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %117 [%108, %118] : f16 into vector<2x8xf16>
      %121 = arith.mulf %arg7, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %122 = vector.extract %121[0, 0] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %arg6 [%108, %109] : f16 into vector<2x8xf16>
      %124 = vector.extract %121[0, 1] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %123 [%108, %112] : f16 into vector<2x8xf16>
      %126 = vector.extract %121[0, 2] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %125 [%108, %115] : f16 into vector<2x8xf16>
      %128 = vector.extract %121[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%108, %118] : f16 into vector<2x8xf16>
      affine.yield %120, %129, %107 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %72 = arith.mulf %71#2, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %73 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
    %74 = vector.insert %73, %71#1 [1, 4] : f16 into vector<2x8xf16>
    %75 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
    %76 = vector.insert %75, %74 [1, 5] : f16 into vector<2x8xf16>
    %77 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
    %78 = vector.insert %77, %76 [1, 6] : f16 into vector<2x8xf16>
    %79 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
    %80 = vector.insert %79, %78 [1, 7] : f16 into vector<2x8xf16>
    %81:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#2, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %97 = frisk.to_threadTile %cst_6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map2, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %98 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c0 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c0 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %98 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<32x128xf16, 3>
      %99:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %127 = frisk.to_threadTile %alloc_15 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %128 = affine.apply #map15(%arg11)
        %129 = vector.extract %127[0, %128] : f16 from vector<8x8xf16>
        %130 = vector.extract %127[1, %128] : f16 from vector<8x8xf16>
        %131 = vector.extract %127[2, %128] : f16 from vector<8x8xf16>
        %132 = vector.extract %127[3, %128] : f16 from vector<8x8xf16>
        %133 = vector.from_elements %129, %130, %131, %132 : vector<4x1xf16>
        %134:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %133) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %164 = frisk.to_threadTile %alloc_15 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %165 = affine.apply #map16(%arg14)
          %166 = vector.extract %164[%165, %128] : f16 from vector<8x8xf16>
          %167 = affine.apply #map17(%arg14)
          %168 = vector.extract %164[%167, %128] : f16 from vector<8x8xf16>
          %169 = affine.apply #map18(%arg14)
          %170 = vector.extract %164[%169, %128] : f16 from vector<8x8xf16>
          %171 = affine.apply #map19(%arg14)
          %172 = vector.extract %164[%171, %128] : f16 from vector<8x8xf16>
          %173 = vector.from_elements %166, %168, %170, %172 : vector<4x1xf16>
          %174 = affine.apply #map20(%arg11)
          %175 = affine.apply #map21(%arg14)
          %176 = vector.extract %29[%174, %175] : f16 from vector<2x8xf16>
          %177 = affine.apply #map22(%arg14)
          %178 = vector.extract %29[%174, %177] : f16 from vector<2x8xf16>
          %179 = affine.apply #map23(%arg14)
          %180 = vector.extract %29[%174, %179] : f16 from vector<2x8xf16>
          %181 = affine.apply #map24(%arg14)
          %182 = vector.extract %29[%174, %181] : f16 from vector<2x8xf16>
          %183 = vector.from_elements %176, %178, %180, %182 : vector<1x4xf16>
          %184 = frisk.warp_mma_rr %183, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %184, %173 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %135 = affine.apply #map20(%arg11)
        %136 = vector.extract %29[%135, 4] : f16 from vector<2x8xf16>
        %137 = vector.extract %29[%135, 5] : f16 from vector<2x8xf16>
        %138 = vector.extract %29[%135, 6] : f16 from vector<2x8xf16>
        %139 = vector.extract %29[%135, 7] : f16 from vector<2x8xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<1x4xf16>
        %141 = frisk.warp_mma_rr %140, %134#1, %134#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %142 = affine.apply #map25(%arg11)
        %143 = vector.extract %97[%135, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply #map26(%arg11)
        %145 = vector.extract %97[%135, %144] : f32 from vector<2x32xf32>
        %146 = affine.apply #map27(%arg11)
        %147 = vector.extract %97[%135, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply #map28(%arg11)
        %149 = vector.extract %97[%135, %148] : f32 from vector<2x32xf32>
        %150 = vector.from_elements %143, %145, %147, %149 : vector<1x4xf32>
        %151 = arith.addf %150, %141 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%135, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%135, %144] : f32 into vector<2x32xf32>
        %156 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%135, %146] : f32 into vector<2x32xf32>
        %158 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%135, %148] : f32 into vector<2x32xf32>
        %160 = vector.insert %152, %arg13 [%135, %142] : f32 into vector<2x32xf32>
        %161 = vector.insert %154, %160 [%135, %144] : f32 into vector<2x32xf32>
        %162 = vector.insert %156, %161 [%135, %146] : f32 into vector<2x32xf32>
        %163 = vector.insert %158, %162 [%135, %148] : f32 into vector<2x32xf32>
        affine.yield %159, %163 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %100 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c32 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c32 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %100 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_16 : memref<32x128xf16, 3>
      %101:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %127 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %128 = affine.apply #map15(%arg11)
        %129 = vector.extract %127[0, %128] : f16 from vector<8x8xf16>
        %130 = vector.extract %127[1, %128] : f16 from vector<8x8xf16>
        %131 = vector.extract %127[2, %128] : f16 from vector<8x8xf16>
        %132 = vector.extract %127[3, %128] : f16 from vector<8x8xf16>
        %133 = vector.from_elements %129, %130, %131, %132 : vector<4x1xf16>
        %134:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %133) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %164 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %165 = affine.apply #map16(%arg14)
          %166 = vector.extract %164[%165, %128] : f16 from vector<8x8xf16>
          %167 = affine.apply #map17(%arg14)
          %168 = vector.extract %164[%167, %128] : f16 from vector<8x8xf16>
          %169 = affine.apply #map18(%arg14)
          %170 = vector.extract %164[%169, %128] : f16 from vector<8x8xf16>
          %171 = affine.apply #map19(%arg14)
          %172 = vector.extract %164[%171, %128] : f16 from vector<8x8xf16>
          %173 = vector.from_elements %166, %168, %170, %172 : vector<4x1xf16>
          %174 = affine.apply #map20(%arg11)
          %175 = affine.apply #map21(%arg14)
          %176 = vector.extract %46[%174, %175] : f16 from vector<2x8xf16>
          %177 = affine.apply #map22(%arg14)
          %178 = vector.extract %46[%174, %177] : f16 from vector<2x8xf16>
          %179 = affine.apply #map23(%arg14)
          %180 = vector.extract %46[%174, %179] : f16 from vector<2x8xf16>
          %181 = affine.apply #map24(%arg14)
          %182 = vector.extract %46[%174, %181] : f16 from vector<2x8xf16>
          %183 = vector.from_elements %176, %178, %180, %182 : vector<1x4xf16>
          %184 = frisk.warp_mma_rr %183, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %184, %173 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %135 = affine.apply #map20(%arg11)
        %136 = vector.extract %46[%135, 4] : f16 from vector<2x8xf16>
        %137 = vector.extract %46[%135, 5] : f16 from vector<2x8xf16>
        %138 = vector.extract %46[%135, 6] : f16 from vector<2x8xf16>
        %139 = vector.extract %46[%135, 7] : f16 from vector<2x8xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<1x4xf16>
        %141 = frisk.warp_mma_rr %140, %134#1, %134#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %142 = affine.apply #map25(%arg11)
        %143 = vector.extract %99#1[%135, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply #map26(%arg11)
        %145 = vector.extract %99#1[%135, %144] : f32 from vector<2x32xf32>
        %146 = affine.apply #map27(%arg11)
        %147 = vector.extract %99#1[%135, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply #map28(%arg11)
        %149 = vector.extract %99#1[%135, %148] : f32 from vector<2x32xf32>
        %150 = vector.from_elements %143, %145, %147, %149 : vector<1x4xf32>
        %151 = arith.addf %150, %141 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%135, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%135, %144] : f32 into vector<2x32xf32>
        %156 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%135, %146] : f32 into vector<2x32xf32>
        %158 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%135, %148] : f32 into vector<2x32xf32>
        %160 = vector.insert %152, %arg13 [%135, %142] : f32 into vector<2x32xf32>
        %161 = vector.insert %154, %160 [%135, %144] : f32 into vector<2x32xf32>
        %162 = vector.insert %156, %161 [%135, %146] : f32 into vector<2x32xf32>
        %163 = vector.insert %158, %162 [%135, %148] : f32 into vector<2x32xf32>
        affine.yield %159, %163 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %102 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c64 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c64 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %102 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_17 : memref<32x128xf16, 3>
      %103:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %127 = frisk.to_threadTile %alloc_17 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %128 = affine.apply #map15(%arg11)
        %129 = vector.extract %127[0, %128] : f16 from vector<8x8xf16>
        %130 = vector.extract %127[1, %128] : f16 from vector<8x8xf16>
        %131 = vector.extract %127[2, %128] : f16 from vector<8x8xf16>
        %132 = vector.extract %127[3, %128] : f16 from vector<8x8xf16>
        %133 = vector.from_elements %129, %130, %131, %132 : vector<4x1xf16>
        %134:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %133) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %164 = frisk.to_threadTile %alloc_17 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %165 = affine.apply #map16(%arg14)
          %166 = vector.extract %164[%165, %128] : f16 from vector<8x8xf16>
          %167 = affine.apply #map17(%arg14)
          %168 = vector.extract %164[%167, %128] : f16 from vector<8x8xf16>
          %169 = affine.apply #map18(%arg14)
          %170 = vector.extract %164[%169, %128] : f16 from vector<8x8xf16>
          %171 = affine.apply #map19(%arg14)
          %172 = vector.extract %164[%171, %128] : f16 from vector<8x8xf16>
          %173 = vector.from_elements %166, %168, %170, %172 : vector<4x1xf16>
          %174 = affine.apply #map20(%arg11)
          %175 = affine.apply #map21(%arg14)
          %176 = vector.extract %63[%174, %175] : f16 from vector<2x8xf16>
          %177 = affine.apply #map22(%arg14)
          %178 = vector.extract %63[%174, %177] : f16 from vector<2x8xf16>
          %179 = affine.apply #map23(%arg14)
          %180 = vector.extract %63[%174, %179] : f16 from vector<2x8xf16>
          %181 = affine.apply #map24(%arg14)
          %182 = vector.extract %63[%174, %181] : f16 from vector<2x8xf16>
          %183 = vector.from_elements %176, %178, %180, %182 : vector<1x4xf16>
          %184 = frisk.warp_mma_rr %183, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %184, %173 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %135 = affine.apply #map20(%arg11)
        %136 = vector.extract %63[%135, 4] : f16 from vector<2x8xf16>
        %137 = vector.extract %63[%135, 5] : f16 from vector<2x8xf16>
        %138 = vector.extract %63[%135, 6] : f16 from vector<2x8xf16>
        %139 = vector.extract %63[%135, 7] : f16 from vector<2x8xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<1x4xf16>
        %141 = frisk.warp_mma_rr %140, %134#1, %134#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %142 = affine.apply #map25(%arg11)
        %143 = vector.extract %101#1[%135, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply #map26(%arg11)
        %145 = vector.extract %101#1[%135, %144] : f32 from vector<2x32xf32>
        %146 = affine.apply #map27(%arg11)
        %147 = vector.extract %101#1[%135, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply #map28(%arg11)
        %149 = vector.extract %101#1[%135, %148] : f32 from vector<2x32xf32>
        %150 = vector.from_elements %143, %145, %147, %149 : vector<1x4xf32>
        %151 = arith.addf %150, %141 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%135, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%135, %144] : f32 into vector<2x32xf32>
        %156 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%135, %146] : f32 into vector<2x32xf32>
        %158 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%135, %148] : f32 into vector<2x32xf32>
        %160 = vector.insert %152, %arg13 [%135, %142] : f32 into vector<2x32xf32>
        %161 = vector.insert %154, %160 [%135, %144] : f32 into vector<2x32xf32>
        %162 = vector.insert %156, %161 [%135, %146] : f32 into vector<2x32xf32>
        %163 = vector.insert %158, %162 [%135, %148] : f32 into vector<2x32xf32>
        affine.yield %159, %163 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %104 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c96 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c96 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %104 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_18 : memref<32x128xf16, 3>
      %105:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %127 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %128 = affine.apply #map15(%arg11)
        %129 = vector.extract %127[0, %128] : f16 from vector<8x8xf16>
        %130 = vector.extract %127[1, %128] : f16 from vector<8x8xf16>
        %131 = vector.extract %127[2, %128] : f16 from vector<8x8xf16>
        %132 = vector.extract %127[3, %128] : f16 from vector<8x8xf16>
        %133 = vector.from_elements %129, %130, %131, %132 : vector<4x1xf16>
        %134:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %133) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %164 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %165 = affine.apply #map16(%arg14)
          %166 = vector.extract %164[%165, %128] : f16 from vector<8x8xf16>
          %167 = affine.apply #map17(%arg14)
          %168 = vector.extract %164[%167, %128] : f16 from vector<8x8xf16>
          %169 = affine.apply #map18(%arg14)
          %170 = vector.extract %164[%169, %128] : f16 from vector<8x8xf16>
          %171 = affine.apply #map19(%arg14)
          %172 = vector.extract %164[%171, %128] : f16 from vector<8x8xf16>
          %173 = vector.from_elements %166, %168, %170, %172 : vector<4x1xf16>
          %174 = affine.apply #map20(%arg11)
          %175 = affine.apply #map21(%arg14)
          %176 = vector.extract %80[%174, %175] : f16 from vector<2x8xf16>
          %177 = affine.apply #map22(%arg14)
          %178 = vector.extract %80[%174, %177] : f16 from vector<2x8xf16>
          %179 = affine.apply #map23(%arg14)
          %180 = vector.extract %80[%174, %179] : f16 from vector<2x8xf16>
          %181 = affine.apply #map24(%arg14)
          %182 = vector.extract %80[%174, %181] : f16 from vector<2x8xf16>
          %183 = vector.from_elements %176, %178, %180, %182 : vector<1x4xf16>
          %184 = frisk.warp_mma_rr %183, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %184, %173 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %135 = affine.apply #map20(%arg11)
        %136 = vector.extract %80[%135, 4] : f16 from vector<2x8xf16>
        %137 = vector.extract %80[%135, 5] : f16 from vector<2x8xf16>
        %138 = vector.extract %80[%135, 6] : f16 from vector<2x8xf16>
        %139 = vector.extract %80[%135, 7] : f16 from vector<2x8xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<1x4xf16>
        %141 = frisk.warp_mma_rr %140, %134#1, %134#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %142 = affine.apply #map25(%arg11)
        %143 = vector.extract %103#1[%135, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply #map26(%arg11)
        %145 = vector.extract %103#1[%135, %144] : f32 from vector<2x32xf32>
        %146 = affine.apply #map27(%arg11)
        %147 = vector.extract %103#1[%135, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply #map28(%arg11)
        %149 = vector.extract %103#1[%135, %148] : f32 from vector<2x32xf32>
        %150 = vector.from_elements %143, %145, %147, %149 : vector<1x4xf32>
        %151 = arith.addf %150, %141 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%135, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%135, %144] : f32 into vector<2x32xf32>
        %156 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%135, %146] : f32 into vector<2x32xf32>
        %158 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%135, %148] : f32 into vector<2x32xf32>
        %160 = vector.insert %152, %arg13 [%135, %142] : f32 into vector<2x32xf32>
        %161 = vector.insert %154, %160 [%135, %144] : f32 into vector<2x32xf32>
        %162 = vector.insert %156, %161 [%135, %146] : f32 into vector<2x32xf32>
        %163 = vector.insert %158, %162 [%135, %148] : f32 into vector<2x32xf32>
        affine.yield %159, %163 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %106 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %127 = affine.apply #map20(%arg11)
        %128 = affine.apply #map25(%arg11)
        %129 = vector.extract %105#1[%127, %128] : f32 from vector<2x32xf32>
        %130 = affine.apply #map26(%arg11)
        %131 = vector.extract %105#1[%127, %130] : f32 from vector<2x32xf32>
        %132 = affine.apply #map27(%arg11)
        %133 = vector.extract %105#1[%127, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply #map28(%arg11)
        %135 = vector.extract %105#1[%127, %134] : f32 from vector<2x32xf32>
        %136 = vector.insert %129, %arg12 [%127, %128] : f32 into vector<2x32xf32>
        %137 = vector.insert %131, %136 [%127, %130] : f32 into vector<2x32xf32>
        %138 = vector.insert %133, %137 [%127, %132] : f32 into vector<2x32xf32>
        %139 = vector.insert %135, %138 [%127, %134] : f32 into vector<2x32xf32>
        affine.yield %139 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %107:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %127 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_10) -> (f32) {
          %137 = affine.apply #map21(%arg14)
          %138 = vector.extract %106[%arg11, %137] : f32 from vector<2x32xf32>
          %139 = arith.maxnumf %arg15, %138 : f32
          %140 = affine.apply #map22(%arg14)
          %141 = vector.extract %106[%arg11, %140] : f32 from vector<2x32xf32>
          %142 = arith.maxnumf %139, %141 : f32
          %143 = affine.apply #map23(%arg14)
          %144 = vector.extract %106[%arg11, %143] : f32 from vector<2x32xf32>
          %145 = arith.maxnumf %142, %144 : f32
          %146 = affine.apply #map24(%arg14)
          %147 = vector.extract %106[%arg11, %146] : f32 from vector<2x32xf32>
          %148 = arith.maxnumf %145, %147 : f32
          affine.yield %148 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %127, %c16_i32, %c64_i32 : f32
        %128 = arith.maxnumf %127, %shuffleResult : f32
        %shuffleResult_24, %valid_25 = gpu.shuffle xor %128, %c32_i32, %c64_i32 : f32
        %129 = arith.maxnumf %128, %shuffleResult_24 : f32
        %130 = vector.broadcast %129 : f32 to vector<1x1xf32>
        %131 = vector.insert %129, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %132 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %133 = vector.broadcast %132 : f32 to vector<1x1xf32>
        %134 = arith.maxnumf %130, %133 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %135 = vector.extract %134[0, 0] : f32 from vector<1x1xf32>
        %136 = vector.insert %135, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %131, %136 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %108:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %127 = affine.apply #map20(%arg11)
        %128 = vector.extract %107#1[%127, 0] : f32 from vector<2x1xf32>
        %129 = vector.broadcast %128 : f32 to vector<1x4xf32>
        %130 = affine.apply #map25(%arg11)
        %131 = vector.insert %128, %arg12 [%127, %130] : f32 into vector<2x32xf32>
        %132 = affine.apply #map26(%arg11)
        %133 = vector.insert %128, %131 [%127, %132] : f32 into vector<2x32xf32>
        %134 = affine.apply #map27(%arg11)
        %135 = vector.insert %128, %133 [%127, %134] : f32 into vector<2x32xf32>
        %136 = affine.apply #map28(%arg11)
        %137 = vector.insert %128, %135 [%127, %136] : f32 into vector<2x32xf32>
        %138 = vector.extract %105#1[%127, %130] : f32 from vector<2x32xf32>
        %139 = vector.extract %105#1[%127, %132] : f32 from vector<2x32xf32>
        %140 = vector.extract %105#1[%127, %134] : f32 from vector<2x32xf32>
        %141 = vector.extract %105#1[%127, %136] : f32 from vector<2x32xf32>
        %142 = vector.from_elements %138, %139, %140, %141 : vector<1x4xf32>
        %143 = arith.subf %142, %129 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %144 = vector.extract %143[0, 0] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %arg13 [%127, %130] : f32 into vector<2x32xf32>
        %146 = vector.extract %143[0, 1] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%127, %132] : f32 into vector<2x32xf32>
        %148 = vector.extract %143[0, 2] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%127, %134] : f32 into vector<2x32xf32>
        %150 = vector.extract %143[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%127, %136] : f32 into vector<2x32xf32>
        affine.yield %137, %151 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>}
      %109:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9, %arg14 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %127 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %128 = vector.broadcast %127 : f32 to vector<1x1xf32>
        %129 = vector.extract %107#1[%arg11, 0] : f32 from vector<2x1xf32>
        %130 = vector.broadcast %129 : f32 to vector<1x1xf32>
        %131 = arith.subf %128, %130 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %132 = vector.extract %131[0, 0] : f32 from vector<1x1xf32>
        %133 = vector.insert %132, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %134 = math.exp2 %131 {tiled = true} : vector<1x1xf32>
        %135 = vector.extract %134[0, 0] : f32 from vector<1x1xf32>
        %136 = vector.insert %135, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %137 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %138 = vector.broadcast %137 : f32 to vector<1x1xf32>
        %139 = arith.mulf %138, %134 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %140 = vector.extract %139[0, 0] : f32 from vector<1x1xf32>
        %141 = vector.insert %140, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %133, %136, %141 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %110:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %127 = affine.apply #map20(%arg11)
        %128 = affine.apply #map25(%arg11)
        %129 = vector.extract %108#1[%127, %128] : f32 from vector<2x32xf32>
        %130 = affine.apply #map26(%arg11)
        %131 = vector.extract %108#1[%127, %130] : f32 from vector<2x32xf32>
        %132 = affine.apply #map27(%arg11)
        %133 = vector.extract %108#1[%127, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply #map28(%arg11)
        %135 = vector.extract %108#1[%127, %134] : f32 from vector<2x32xf32>
        %136 = vector.from_elements %129, %131, %133, %135 : vector<1x4xf32>
        %137 = math.exp2 %136 {tiled = true} : vector<1x4xf32>
        %138 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %139 = vector.insert %138, %arg12 [%127, %128] : f32 into vector<2x32xf32>
        %140 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %139 [%127, %130] : f32 into vector<2x32xf32>
        %142 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %141 [%127, %132] : f32 into vector<2x32xf32>
        %144 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%127, %134] : f32 into vector<2x32xf32>
        %146 = vector.insert %138, %arg13 [%127, %128] : f32 into vector<2x32xf32>
        %147 = vector.insert %140, %146 [%127, %130] : f32 into vector<2x32xf32>
        %148 = vector.insert %142, %147 [%127, %132] : f32 into vector<2x32xf32>
        %149 = vector.insert %144, %148 [%127, %134] : f32 into vector<2x32xf32>
        affine.yield %145, %149 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %111:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %127 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_2) -> (f32) {
          %137 = affine.apply #map21(%arg14)
          %138 = vector.extract %110#1[%arg11, %137] : f32 from vector<2x32xf32>
          %139 = arith.addf %arg15, %138 : f32
          %140 = affine.apply #map22(%arg14)
          %141 = vector.extract %110#1[%arg11, %140] : f32 from vector<2x32xf32>
          %142 = arith.addf %139, %141 : f32
          %143 = affine.apply #map23(%arg14)
          %144 = vector.extract %110#1[%arg11, %143] : f32 from vector<2x32xf32>
          %145 = arith.addf %142, %144 : f32
          %146 = affine.apply #map24(%arg14)
          %147 = vector.extract %110#1[%arg11, %146] : f32 from vector<2x32xf32>
          %148 = arith.addf %145, %147 : f32
          affine.yield %148 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %127, %c16_i32, %c64_i32 : f32
        %128 = arith.addf %127, %shuffleResult : f32
        %shuffleResult_24, %valid_25 = gpu.shuffle xor %128, %c32_i32, %c64_i32 : f32
        %129 = arith.addf %128, %shuffleResult_24 : f32
        %130 = vector.broadcast %129 : f32 to vector<1x1xf32>
        %131 = vector.insert %129, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %132 = vector.extract %109#2[%arg11, 0] : f32 from vector<2x1xf32>
        %133 = vector.broadcast %132 : f32 to vector<1x1xf32>
        %134 = arith.addf %130, %133 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %135 = vector.extract %134[0, 0] : f32 from vector<1x1xf32>
        %136 = vector.insert %135, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %131, %136 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %112 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_1) -> (vector<2x32xf16>) {
        %127 = affine.apply #map20(%arg11)
        %128 = affine.apply #map25(%arg11)
        %129 = vector.extract %110#0[%127, %128] : f32 from vector<2x32xf32>
        %130 = affine.apply #map26(%arg11)
        %131 = vector.extract %110#0[%127, %130] : f32 from vector<2x32xf32>
        %132 = affine.apply #map27(%arg11)
        %133 = vector.extract %110#0[%127, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply #map28(%arg11)
        %135 = vector.extract %110#0[%127, %134] : f32 from vector<2x32xf32>
        %136 = vector.from_elements %129, %131, %133, %135 : vector<1x4xf32>
        %137 = arith.truncf %136 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %138 = vector.extract %137[0, 0] : f16 from vector<1x4xf16>
        %139 = vector.insert %138, %arg12 [%127, %128] : f16 into vector<2x32xf16>
        %140 = vector.extract %137[0, 1] : f16 from vector<1x4xf16>
        %141 = vector.insert %140, %139 [%127, %130] : f16 into vector<2x32xf16>
        %142 = vector.extract %137[0, 2] : f16 from vector<1x4xf16>
        %143 = vector.insert %142, %141 [%127, %132] : f16 into vector<2x32xf16>
        %144 = vector.extract %137[0, 3] : f16 from vector<1x4xf16>
        %145 = vector.insert %144, %143 [%127, %134] : f16 into vector<2x32xf16>
        affine.yield %145 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %113 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %113 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_19 : memref<128x32xf16, 3>
      %thread_id_x_20 = gpu.thread_id x
      %114 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %127 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %128 = affine.apply #map30()[%thread_id_x_20]
        %129 = frisk.read_matrix_from_shm %alloc_19[%127, %128] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %130 = vector.bitcast %129 : vector<4xi32> to vector<8xf16>
        %131 = vector.extract %130[0] : f16 from vector<8xf16>
        %132 = affine.apply #map31()[%thread_id_x_20]
        %133 = vector.extract %130[%132] : f16 from vector<8xf16>
        %134 = vector.extract %130[1] : f16 from vector<8xf16>
        %135 = affine.apply #map32()[%thread_id_x_20]
        %136 = vector.extract %130[%135] : f16 from vector<8xf16>
        %137 = vector.extract %130[2] : f16 from vector<8xf16>
        %138 = affine.apply #map33()[%thread_id_x_20]
        %139 = vector.extract %130[%138] : f16 from vector<8xf16>
        %140 = vector.extract %130[3] : f16 from vector<8xf16>
        %141 = affine.apply #map34()[%thread_id_x_20]
        %142 = vector.extract %130[%141] : f16 from vector<8xf16>
        %143 = affine.apply #map21(%arg11)
        %144 = vector.insert %131, %arg12 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply #map22(%arg11)
        %147 = vector.insert %134, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %136, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply #map23(%arg11)
        %150 = vector.insert %137, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        %152 = affine.apply #map24(%arg11)
        %153 = vector.insert %140, %151 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        affine.yield %154 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %115:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %127 = affine.apply #map9(%arg11)
        %128 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %170 = affine.apply #map21(%arg16)
          %171 = vector.extract %112[%127, %170] : f16 from vector<2x32xf16>
          %172 = affine.apply #map22(%arg16)
          %173 = vector.extract %112[%127, %172] : f16 from vector<2x32xf16>
          %174 = affine.apply #map23(%arg16)
          %175 = vector.extract %112[%127, %174] : f16 from vector<2x32xf16>
          %176 = affine.apply #map24(%arg16)
          %177 = vector.extract %112[%127, %176] : f16 from vector<2x32xf16>
          %178 = vector.from_elements %171, %173, %175, %177 : vector<1x4xf16>
          %179 = affine.apply #map35(%arg11)
          %180 = vector.extract %114[%170, %179] : f16 from vector<32x2xf16>
          %181 = vector.extract %114[%172, %179] : f16 from vector<32x2xf16>
          %182 = vector.extract %114[%174, %179] : f16 from vector<32x2xf16>
          %183 = vector.extract %114[%176, %179] : f16 from vector<32x2xf16>
          %184 = vector.from_elements %180, %181, %182, %183 : vector<4x1xf16>
          %185 = frisk.warp_mma_rr %178, %184, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %185 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %129 = affine.apply #map10(%arg11)
        %130 = vector.extract %128[0, 0] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %arg12 [%127, %129] : f32 into vector<2x8xf32>
        %132 = affine.apply #map11(%arg11)
        %133 = vector.extract %128[0, 1] : f32 from vector<1x4xf32>
        %134 = vector.insert %133, %131 [%127, %132] : f32 into vector<2x8xf32>
        %135 = affine.apply #map12(%arg11)
        %136 = vector.extract %128[0, 2] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %134 [%127, %135] : f32 into vector<2x8xf32>
        %138 = affine.apply #map13(%arg11)
        %139 = vector.extract %128[0, 3] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %137 [%127, %138] : f32 into vector<2x8xf32>
        %141 = vector.extract %109#1[%127, 0] : f32 from vector<2x1xf32>
        %142 = vector.broadcast %141 : f32 to vector<1x4xf32>
        %143 = vector.insert %141, %arg13 [%127, %129] : f32 into vector<2x8xf32>
        %144 = vector.insert %141, %143 [%127, %132] : f32 into vector<2x8xf32>
        %145 = vector.insert %141, %144 [%127, %135] : f32 into vector<2x8xf32>
        %146 = vector.insert %141, %145 [%127, %138] : f32 into vector<2x8xf32>
        %147 = vector.extract %arg5[%127, %129] : f32 from vector<2x8xf32>
        %148 = vector.extract %arg5[%127, %132] : f32 from vector<2x8xf32>
        %149 = vector.extract %arg5[%127, %135] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg5[%127, %138] : f32 from vector<2x8xf32>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<1x4xf32>
        %152 = arith.mulf %151, %142 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg14 [%127, %129] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%127, %132] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%127, %135] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%127, %138] : f32 into vector<2x8xf32>
        %161 = arith.addf %152, %128 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %162 = vector.extract %161[0, 0] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %arg15 [%127, %129] : f32 into vector<2x8xf32>
        %164 = vector.extract %161[0, 1] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%127, %132] : f32 into vector<2x8xf32>
        %166 = vector.extract %161[0, 2] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%127, %135] : f32 into vector<2x8xf32>
        %168 = vector.extract %161[0, 3] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%127, %138] : f32 into vector<2x8xf32>
        affine.yield %140, %146, %160, %169 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %116 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %116 to %alloc_21 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_21 : memref<128x32xf16, 3>
      %117 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %127 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %128 = affine.apply #map30()[%thread_id_x_20]
        %129 = frisk.read_matrix_from_shm %alloc_21[%127, %128] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %130 = vector.bitcast %129 : vector<4xi32> to vector<8xf16>
        %131 = vector.extract %130[0] : f16 from vector<8xf16>
        %132 = affine.apply #map31()[%thread_id_x_20]
        %133 = vector.extract %130[%132] : f16 from vector<8xf16>
        %134 = vector.extract %130[1] : f16 from vector<8xf16>
        %135 = affine.apply #map32()[%thread_id_x_20]
        %136 = vector.extract %130[%135] : f16 from vector<8xf16>
        %137 = vector.extract %130[2] : f16 from vector<8xf16>
        %138 = affine.apply #map33()[%thread_id_x_20]
        %139 = vector.extract %130[%138] : f16 from vector<8xf16>
        %140 = vector.extract %130[3] : f16 from vector<8xf16>
        %141 = affine.apply #map34()[%thread_id_x_20]
        %142 = vector.extract %130[%141] : f16 from vector<8xf16>
        %143 = affine.apply #map21(%arg11)
        %144 = vector.insert %131, %arg12 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply #map22(%arg11)
        %147 = vector.insert %134, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %136, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply #map23(%arg11)
        %150 = vector.insert %137, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        %152 = affine.apply #map24(%arg11)
        %153 = vector.insert %140, %151 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        affine.yield %154 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %118:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %127 = affine.apply #map9(%arg11)
        %128 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %170 = affine.apply #map21(%arg16)
          %171 = vector.extract %112[%127, %170] : f16 from vector<2x32xf16>
          %172 = affine.apply #map22(%arg16)
          %173 = vector.extract %112[%127, %172] : f16 from vector<2x32xf16>
          %174 = affine.apply #map23(%arg16)
          %175 = vector.extract %112[%127, %174] : f16 from vector<2x32xf16>
          %176 = affine.apply #map24(%arg16)
          %177 = vector.extract %112[%127, %176] : f16 from vector<2x32xf16>
          %178 = vector.from_elements %171, %173, %175, %177 : vector<1x4xf16>
          %179 = affine.apply #map35(%arg11)
          %180 = vector.extract %117[%170, %179] : f16 from vector<32x2xf16>
          %181 = vector.extract %117[%172, %179] : f16 from vector<32x2xf16>
          %182 = vector.extract %117[%174, %179] : f16 from vector<32x2xf16>
          %183 = vector.extract %117[%176, %179] : f16 from vector<32x2xf16>
          %184 = vector.from_elements %180, %181, %182, %183 : vector<4x1xf16>
          %185 = frisk.warp_mma_rr %178, %184, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %185 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %129 = affine.apply #map10(%arg11)
        %130 = vector.extract %128[0, 0] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %arg12 [%127, %129] : f32 into vector<2x8xf32>
        %132 = affine.apply #map11(%arg11)
        %133 = vector.extract %128[0, 1] : f32 from vector<1x4xf32>
        %134 = vector.insert %133, %131 [%127, %132] : f32 into vector<2x8xf32>
        %135 = affine.apply #map12(%arg11)
        %136 = vector.extract %128[0, 2] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %134 [%127, %135] : f32 into vector<2x8xf32>
        %138 = affine.apply #map13(%arg11)
        %139 = vector.extract %128[0, 3] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %137 [%127, %138] : f32 into vector<2x8xf32>
        %141 = vector.extract %109#1[%127, 0] : f32 from vector<2x1xf32>
        %142 = vector.broadcast %141 : f32 to vector<1x4xf32>
        %143 = vector.insert %141, %arg13 [%127, %129] : f32 into vector<2x8xf32>
        %144 = vector.insert %141, %143 [%127, %132] : f32 into vector<2x8xf32>
        %145 = vector.insert %141, %144 [%127, %135] : f32 into vector<2x8xf32>
        %146 = vector.insert %141, %145 [%127, %138] : f32 into vector<2x8xf32>
        %147 = vector.extract %arg8[%127, %129] : f32 from vector<2x8xf32>
        %148 = vector.extract %arg8[%127, %132] : f32 from vector<2x8xf32>
        %149 = vector.extract %arg8[%127, %135] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg8[%127, %138] : f32 from vector<2x8xf32>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<1x4xf32>
        %152 = arith.mulf %151, %142 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg14 [%127, %129] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%127, %132] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%127, %135] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%127, %138] : f32 into vector<2x8xf32>
        %161 = arith.addf %152, %128 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %162 = vector.extract %161[0, 0] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %arg15 [%127, %129] : f32 into vector<2x8xf32>
        %164 = vector.extract %161[0, 1] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%127, %132] : f32 into vector<2x8xf32>
        %166 = vector.extract %161[0, 2] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%127, %135] : f32 into vector<2x8xf32>
        %168 = vector.extract %161[0, 3] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%127, %138] : f32 into vector<2x8xf32>
        affine.yield %140, %146, %160, %169 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_22 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %119 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %119 to %alloc_22 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_22 : memref<128x32xf16, 3>
      %120 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %127 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %128 = affine.apply #map30()[%thread_id_x_20]
        %129 = frisk.read_matrix_from_shm %alloc_22[%127, %128] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %130 = vector.bitcast %129 : vector<4xi32> to vector<8xf16>
        %131 = vector.extract %130[0] : f16 from vector<8xf16>
        %132 = affine.apply #map31()[%thread_id_x_20]
        %133 = vector.extract %130[%132] : f16 from vector<8xf16>
        %134 = vector.extract %130[1] : f16 from vector<8xf16>
        %135 = affine.apply #map32()[%thread_id_x_20]
        %136 = vector.extract %130[%135] : f16 from vector<8xf16>
        %137 = vector.extract %130[2] : f16 from vector<8xf16>
        %138 = affine.apply #map33()[%thread_id_x_20]
        %139 = vector.extract %130[%138] : f16 from vector<8xf16>
        %140 = vector.extract %130[3] : f16 from vector<8xf16>
        %141 = affine.apply #map34()[%thread_id_x_20]
        %142 = vector.extract %130[%141] : f16 from vector<8xf16>
        %143 = affine.apply #map21(%arg11)
        %144 = vector.insert %131, %arg12 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply #map22(%arg11)
        %147 = vector.insert %134, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %136, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply #map23(%arg11)
        %150 = vector.insert %137, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        %152 = affine.apply #map24(%arg11)
        %153 = vector.insert %140, %151 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        affine.yield %154 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %121:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %127 = affine.apply #map9(%arg11)
        %128 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %170 = affine.apply #map21(%arg16)
          %171 = vector.extract %112[%127, %170] : f16 from vector<2x32xf16>
          %172 = affine.apply #map22(%arg16)
          %173 = vector.extract %112[%127, %172] : f16 from vector<2x32xf16>
          %174 = affine.apply #map23(%arg16)
          %175 = vector.extract %112[%127, %174] : f16 from vector<2x32xf16>
          %176 = affine.apply #map24(%arg16)
          %177 = vector.extract %112[%127, %176] : f16 from vector<2x32xf16>
          %178 = vector.from_elements %171, %173, %175, %177 : vector<1x4xf16>
          %179 = affine.apply #map35(%arg11)
          %180 = vector.extract %120[%170, %179] : f16 from vector<32x2xf16>
          %181 = vector.extract %120[%172, %179] : f16 from vector<32x2xf16>
          %182 = vector.extract %120[%174, %179] : f16 from vector<32x2xf16>
          %183 = vector.extract %120[%176, %179] : f16 from vector<32x2xf16>
          %184 = vector.from_elements %180, %181, %182, %183 : vector<4x1xf16>
          %185 = frisk.warp_mma_rr %178, %184, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %185 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %129 = affine.apply #map10(%arg11)
        %130 = vector.extract %128[0, 0] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %arg12 [%127, %129] : f32 into vector<2x8xf32>
        %132 = affine.apply #map11(%arg11)
        %133 = vector.extract %128[0, 1] : f32 from vector<1x4xf32>
        %134 = vector.insert %133, %131 [%127, %132] : f32 into vector<2x8xf32>
        %135 = affine.apply #map12(%arg11)
        %136 = vector.extract %128[0, 2] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %134 [%127, %135] : f32 into vector<2x8xf32>
        %138 = affine.apply #map13(%arg11)
        %139 = vector.extract %128[0, 3] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %137 [%127, %138] : f32 into vector<2x8xf32>
        %141 = vector.extract %109#1[%127, 0] : f32 from vector<2x1xf32>
        %142 = vector.broadcast %141 : f32 to vector<1x4xf32>
        %143 = vector.insert %141, %arg13 [%127, %129] : f32 into vector<2x8xf32>
        %144 = vector.insert %141, %143 [%127, %132] : f32 into vector<2x8xf32>
        %145 = vector.insert %141, %144 [%127, %135] : f32 into vector<2x8xf32>
        %146 = vector.insert %141, %145 [%127, %138] : f32 into vector<2x8xf32>
        %147 = vector.extract %arg9[%127, %129] : f32 from vector<2x8xf32>
        %148 = vector.extract %arg9[%127, %132] : f32 from vector<2x8xf32>
        %149 = vector.extract %arg9[%127, %135] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg9[%127, %138] : f32 from vector<2x8xf32>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<1x4xf32>
        %152 = arith.mulf %151, %142 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg14 [%127, %129] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%127, %132] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%127, %135] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%127, %138] : f32 into vector<2x8xf32>
        %161 = arith.addf %152, %128 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %162 = vector.extract %161[0, 0] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %arg15 [%127, %129] : f32 into vector<2x8xf32>
        %164 = vector.extract %161[0, 1] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%127, %132] : f32 into vector<2x8xf32>
        %166 = vector.extract %161[0, 2] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%127, %135] : f32 into vector<2x8xf32>
        %168 = vector.extract %161[0, 3] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%127, %138] : f32 into vector<2x8xf32>
        affine.yield %140, %146, %160, %169 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %122 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %122 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_23 : memref<128x32xf16, 3>
      %123 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %127 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %128 = affine.apply #map30()[%thread_id_x_20]
        %129 = frisk.read_matrix_from_shm %alloc_23[%127, %128] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %130 = vector.bitcast %129 : vector<4xi32> to vector<8xf16>
        %131 = vector.extract %130[0] : f16 from vector<8xf16>
        %132 = affine.apply #map31()[%thread_id_x_20]
        %133 = vector.extract %130[%132] : f16 from vector<8xf16>
        %134 = vector.extract %130[1] : f16 from vector<8xf16>
        %135 = affine.apply #map32()[%thread_id_x_20]
        %136 = vector.extract %130[%135] : f16 from vector<8xf16>
        %137 = vector.extract %130[2] : f16 from vector<8xf16>
        %138 = affine.apply #map33()[%thread_id_x_20]
        %139 = vector.extract %130[%138] : f16 from vector<8xf16>
        %140 = vector.extract %130[3] : f16 from vector<8xf16>
        %141 = affine.apply #map34()[%thread_id_x_20]
        %142 = vector.extract %130[%141] : f16 from vector<8xf16>
        %143 = affine.apply #map21(%arg11)
        %144 = vector.insert %131, %arg12 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply #map22(%arg11)
        %147 = vector.insert %134, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %136, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply #map23(%arg11)
        %150 = vector.insert %137, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        %152 = affine.apply #map24(%arg11)
        %153 = vector.insert %140, %151 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        affine.yield %154 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %124:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %127 = affine.apply #map9(%arg11)
        %128 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %178 = affine.apply #map21(%arg17)
          %179 = vector.extract %112[%127, %178] : f16 from vector<2x32xf16>
          %180 = affine.apply #map22(%arg17)
          %181 = vector.extract %112[%127, %180] : f16 from vector<2x32xf16>
          %182 = affine.apply #map23(%arg17)
          %183 = vector.extract %112[%127, %182] : f16 from vector<2x32xf16>
          %184 = affine.apply #map24(%arg17)
          %185 = vector.extract %112[%127, %184] : f16 from vector<2x32xf16>
          %186 = vector.from_elements %179, %181, %183, %185 : vector<1x4xf16>
          %187 = affine.apply #map35(%arg11)
          %188 = vector.extract %123[%178, %187] : f16 from vector<32x2xf16>
          %189 = vector.extract %123[%180, %187] : f16 from vector<32x2xf16>
          %190 = vector.extract %123[%182, %187] : f16 from vector<32x2xf16>
          %191 = vector.extract %123[%184, %187] : f16 from vector<32x2xf16>
          %192 = vector.from_elements %188, %189, %190, %191 : vector<4x1xf16>
          %193 = frisk.warp_mma_rr %186, %192, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %193 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %129 = affine.apply #map10(%arg11)
        %130 = vector.extract %128[0, 0] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %arg12 [%127, %129] : f32 into vector<2x8xf32>
        %132 = affine.apply #map11(%arg11)
        %133 = vector.extract %128[0, 1] : f32 from vector<1x4xf32>
        %134 = vector.insert %133, %131 [%127, %132] : f32 into vector<2x8xf32>
        %135 = affine.apply #map12(%arg11)
        %136 = vector.extract %128[0, 2] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %134 [%127, %135] : f32 into vector<2x8xf32>
        %138 = affine.apply #map13(%arg11)
        %139 = vector.extract %128[0, 3] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %137 [%127, %138] : f32 into vector<2x8xf32>
        %141 = vector.extract %109#1[%127, 0] : f32 from vector<2x1xf32>
        %142 = vector.broadcast %141 : f32 to vector<1x4xf32>
        %143 = vector.insert %141, %arg13 [%127, %129] : f32 into vector<2x8xf32>
        %144 = vector.insert %141, %143 [%127, %132] : f32 into vector<2x8xf32>
        %145 = vector.insert %141, %144 [%127, %135] : f32 into vector<2x8xf32>
        %146 = vector.insert %141, %145 [%127, %138] : f32 into vector<2x8xf32>
        %147 = vector.extract %arg10[%127, %129] : f32 from vector<2x8xf32>
        %148 = vector.extract %arg10[%127, %132] : f32 from vector<2x8xf32>
        %149 = vector.extract %arg10[%127, %135] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg10[%127, %138] : f32 from vector<2x8xf32>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<1x4xf32>
        %152 = arith.mulf %151, %142 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg14 [%127, %129] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%127, %132] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%127, %135] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%127, %138] : f32 into vector<2x8xf32>
        %161 = arith.addf %152, %128 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %162 = vector.extract %161[0, 0] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %arg15 [%127, %129] : f32 into vector<2x8xf32>
        %164 = vector.extract %161[0, 1] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%127, %132] : f32 into vector<2x8xf32>
        %166 = vector.extract %161[0, 2] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%127, %135] : f32 into vector<2x8xf32>
        %168 = vector.extract %161[0, 3] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%127, %138] : f32 into vector<2x8xf32>
        %170 = vector.extract %115#3[%127, %129] : f32 from vector<2x8xf32>
        %171 = vector.extract %115#3[%127, %132] : f32 from vector<2x8xf32>
        %172 = vector.extract %115#3[%127, %135] : f32 from vector<2x8xf32>
        %173 = vector.extract %115#3[%127, %138] : f32 from vector<2x8xf32>
        %174 = vector.insert %170, %arg16 [%127, %129] : f32 into vector<2x8xf32>
        %175 = vector.insert %171, %174 [%127, %132] : f32 into vector<2x8xf32>
        %176 = vector.insert %172, %175 [%127, %135] : f32 into vector<2x8xf32>
        %177 = vector.insert %173, %176 [%127, %138] : f32 into vector<2x8xf32>
        affine.yield %140, %146, %160, %169, %177 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %125:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %127 = vector.extract %111#1[%arg11, 0] : f32 from vector<2x1xf32>
        %128 = vector.insert %127, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %129 = vector.extract %107#1[%arg11, 0] : f32 from vector<2x1xf32>
        %130 = vector.insert %129, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %128, %130 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %126:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %127 = affine.apply #map9(%arg11)
        %128 = affine.apply #map10(%arg11)
        %129 = vector.extract %118#3[%127, %128] : f32 from vector<2x8xf32>
        %130 = affine.apply #map11(%arg11)
        %131 = vector.extract %118#3[%127, %130] : f32 from vector<2x8xf32>
        %132 = affine.apply #map12(%arg11)
        %133 = vector.extract %118#3[%127, %132] : f32 from vector<2x8xf32>
        %134 = affine.apply #map13(%arg11)
        %135 = vector.extract %118#3[%127, %134] : f32 from vector<2x8xf32>
        %136 = vector.insert %129, %arg12 [%127, %128] : f32 into vector<2x8xf32>
        %137 = vector.insert %131, %136 [%127, %130] : f32 into vector<2x8xf32>
        %138 = vector.insert %133, %137 [%127, %132] : f32 into vector<2x8xf32>
        %139 = vector.insert %135, %138 [%127, %134] : f32 into vector<2x8xf32>
        %140 = vector.extract %121#3[%127, %128] : f32 from vector<2x8xf32>
        %141 = vector.extract %121#3[%127, %130] : f32 from vector<2x8xf32>
        %142 = vector.extract %121#3[%127, %132] : f32 from vector<2x8xf32>
        %143 = vector.extract %121#3[%127, %134] : f32 from vector<2x8xf32>
        %144 = vector.insert %140, %arg13 [%127, %128] : f32 into vector<2x8xf32>
        %145 = vector.insert %141, %144 [%127, %130] : f32 into vector<2x8xf32>
        %146 = vector.insert %142, %145 [%127, %132] : f32 into vector<2x8xf32>
        %147 = vector.insert %143, %146 [%127, %134] : f32 into vector<2x8xf32>
        %148 = vector.extract %124#3[%127, %128] : f32 from vector<2x8xf32>
        %149 = vector.extract %124#3[%127, %130] : f32 from vector<2x8xf32>
        %150 = vector.extract %124#3[%127, %132] : f32 from vector<2x8xf32>
        %151 = vector.extract %124#3[%127, %134] : f32 from vector<2x8xf32>
        %152 = vector.insert %148, %arg14 [%127, %128] : f32 into vector<2x8xf32>
        %153 = vector.insert %149, %152 [%127, %130] : f32 into vector<2x8xf32>
        %154 = vector.insert %150, %153 [%127, %132] : f32 into vector<2x8xf32>
        %155 = vector.insert %151, %154 [%127, %134] : f32 into vector<2x8xf32>
        affine.yield %139, %147, %155 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %124#4, %125#0, %125#1, %126#0, %126#1, %126#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %thread_id_x = gpu.thread_id x
    %82 = arith.index_cast %thread_id_x : index to i32
    %83 = arith.andi %82, %c16_i32 : i32
    %84 = arith.cmpi ne, %83, %c0_i32 : i32
    %85 = arith.andi %82, %c32_i32 : i32
    %86 = arith.cmpi ne, %85, %c0_i32 : i32
    %87 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %88:8 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst_5, %arg12 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %97 = affine.apply #map9(%arg4)
      %98 = affine.apply #map10(%arg4)
      %99 = vector.extract %81#5[%97, %98] : f32 from vector<2x8xf32>
      %100 = affine.apply #map11(%arg4)
      %101 = vector.extract %81#5[%97, %100] : f32 from vector<2x8xf32>
      %102 = affine.apply #map12(%arg4)
      %103 = vector.extract %81#5[%97, %102] : f32 from vector<2x8xf32>
      %104 = affine.apply #map13(%arg4)
      %105 = vector.extract %81#5[%97, %104] : f32 from vector<2x8xf32>
      %106 = arith.select %84, %99, %101 : f32
      %shuffleResult, %valid = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %107 = arith.select %84, %shuffleResult, %99 : f32
      %108 = arith.select %84, %101, %shuffleResult : f32
      %109 = arith.select %84, %103, %105 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %110 = arith.select %84, %shuffleResult_15, %103 : f32
      %111 = arith.select %84, %105, %shuffleResult_15 : f32
      %112 = arith.select %86, %107, %110 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %113 = arith.select %86, %shuffleResult_17, %107 : f32
      %114 = arith.select %86, %110, %shuffleResult_17 : f32
      %115 = arith.select %86, %108, %111 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %86, %shuffleResult_19, %108 : f32
      %117 = arith.select %86, %111, %shuffleResult_19 : f32
      %118 = vector.insert %113, %arg5 [%97, %98] : f32 into vector<2x8xf32>
      %119 = vector.insert %116, %118 [%97, %100] : f32 into vector<2x8xf32>
      %120 = vector.insert %114, %119 [%97, %102] : f32 into vector<2x8xf32>
      %121 = vector.insert %117, %120 [%97, %104] : f32 into vector<2x8xf32>
      %122 = vector.extract %81#4[%97, %98] : f32 from vector<2x8xf32>
      %123 = vector.extract %81#4[%97, %100] : f32 from vector<2x8xf32>
      %124 = vector.extract %81#4[%97, %102] : f32 from vector<2x8xf32>
      %125 = vector.extract %81#4[%97, %104] : f32 from vector<2x8xf32>
      %126 = arith.select %84, %122, %123 : f32
      %shuffleResult_21, %valid_22 = gpu.shuffle xor %126, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %127 = arith.select %84, %shuffleResult_21, %122 : f32
      %128 = arith.select %84, %123, %shuffleResult_21 : f32
      %129 = arith.select %84, %124, %125 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %129, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %130 = arith.select %84, %shuffleResult_23, %124 : f32
      %131 = arith.select %84, %125, %shuffleResult_23 : f32
      %132 = arith.select %86, %127, %130 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %132, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %133 = arith.select %86, %shuffleResult_25, %127 : f32
      %134 = arith.select %86, %130, %shuffleResult_25 : f32
      %135 = arith.select %86, %128, %131 : f32
      %shuffleResult_27, %valid_28 = gpu.shuffle xor %135, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %136 = arith.select %86, %shuffleResult_27, %128 : f32
      %137 = arith.select %86, %131, %shuffleResult_27 : f32
      %138 = vector.insert %133, %arg6 [%97, %98] : f32 into vector<2x8xf32>
      %139 = vector.insert %136, %138 [%97, %100] : f32 into vector<2x8xf32>
      %140 = vector.insert %134, %139 [%97, %102] : f32 into vector<2x8xf32>
      %141 = vector.insert %137, %140 [%97, %104] : f32 into vector<2x8xf32>
      %142 = vector.extract %81#3[%97, %98] : f32 from vector<2x8xf32>
      %143 = vector.extract %81#3[%97, %100] : f32 from vector<2x8xf32>
      %144 = vector.extract %81#3[%97, %102] : f32 from vector<2x8xf32>
      %145 = vector.extract %81#3[%97, %104] : f32 from vector<2x8xf32>
      %146 = arith.select %84, %142, %143 : f32
      %shuffleResult_29, %valid_30 = gpu.shuffle xor %146, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %147 = arith.select %84, %shuffleResult_29, %142 : f32
      %148 = arith.select %84, %143, %shuffleResult_29 : f32
      %149 = arith.select %84, %144, %145 : f32
      %shuffleResult_31, %valid_32 = gpu.shuffle xor %149, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %150 = arith.select %84, %shuffleResult_31, %144 : f32
      %151 = arith.select %84, %145, %shuffleResult_31 : f32
      %152 = arith.select %86, %147, %150 : f32
      %shuffleResult_33, %valid_34 = gpu.shuffle xor %152, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %153 = arith.select %86, %shuffleResult_33, %147 : f32
      %154 = arith.select %86, %150, %shuffleResult_33 : f32
      %155 = arith.select %86, %148, %151 : f32
      %shuffleResult_35, %valid_36 = gpu.shuffle xor %155, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %156 = arith.select %86, %shuffleResult_35, %148 : f32
      %157 = arith.select %86, %151, %shuffleResult_35 : f32
      %158 = vector.insert %153, %arg7 [%97, %98] : f32 into vector<2x8xf32>
      %159 = vector.insert %156, %158 [%97, %100] : f32 into vector<2x8xf32>
      %160 = vector.insert %154, %159 [%97, %102] : f32 into vector<2x8xf32>
      %161 = vector.insert %157, %160 [%97, %104] : f32 into vector<2x8xf32>
      %162 = vector.extract %81#0[%97, %98] : f32 from vector<2x8xf32>
      %163 = vector.extract %81#0[%97, %100] : f32 from vector<2x8xf32>
      %164 = vector.extract %81#0[%97, %102] : f32 from vector<2x8xf32>
      %165 = vector.extract %81#0[%97, %104] : f32 from vector<2x8xf32>
      %166 = arith.select %84, %162, %163 : f32
      %shuffleResult_37, %valid_38 = gpu.shuffle xor %166, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %167 = arith.select %84, %shuffleResult_37, %162 : f32
      %168 = arith.select %84, %163, %shuffleResult_37 : f32
      %169 = arith.select %84, %164, %165 : f32
      %shuffleResult_39, %valid_40 = gpu.shuffle xor %169, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %170 = arith.select %84, %shuffleResult_39, %164 : f32
      %171 = arith.select %84, %165, %shuffleResult_39 : f32
      %172 = arith.select %86, %167, %170 : f32
      %shuffleResult_41, %valid_42 = gpu.shuffle xor %172, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %173 = arith.select %86, %shuffleResult_41, %167 : f32
      %174 = arith.select %86, %170, %shuffleResult_41 : f32
      %175 = arith.select %86, %168, %171 : f32
      %shuffleResult_43, %valid_44 = gpu.shuffle xor %175, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %176 = arith.select %86, %shuffleResult_43, %168 : f32
      %177 = arith.select %86, %171, %shuffleResult_43 : f32
      %178 = vector.from_elements %173, %176, %174, %177 : vector<1x4xf32>
      %179 = vector.insert %173, %arg8 [%97, %98] : f32 into vector<2x8xf32>
      %180 = vector.insert %176, %179 [%97, %100] : f32 into vector<2x8xf32>
      %181 = vector.insert %174, %180 [%97, %102] : f32 into vector<2x8xf32>
      %182 = vector.insert %177, %181 [%97, %104] : f32 into vector<2x8xf32>
      %183 = vector.extract %81#1[%97, 0] : f32 from vector<2x1xf32>
      %184 = vector.broadcast %183 : f32 to vector<1x4xf32>
      %185 = vector.insert %183, %arg9 [%97, %98] : f32 into vector<2x8xf32>
      %186 = vector.insert %183, %185 [%97, %100] : f32 into vector<2x8xf32>
      %187 = vector.insert %183, %186 [%97, %102] : f32 into vector<2x8xf32>
      %188 = vector.insert %183, %187 [%97, %104] : f32 into vector<2x8xf32>
      %189 = arith.divf %178, %184 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %190 = vector.extract %189[0, 0] : f32 from vector<1x4xf32>
      %191 = vector.insert %190, %arg10 [%97, %98] : f32 into vector<2x8xf32>
      %192 = vector.extract %189[0, 1] : f32 from vector<1x4xf32>
      %193 = vector.insert %192, %191 [%97, %100] : f32 into vector<2x8xf32>
      %194 = vector.extract %189[0, 2] : f32 from vector<1x4xf32>
      %195 = vector.insert %194, %193 [%97, %102] : f32 into vector<2x8xf32>
      %196 = vector.extract %189[0, 3] : f32 from vector<1x4xf32>
      %197 = vector.insert %196, %195 [%97, %104] : f32 into vector<2x8xf32>
      %198 = arith.truncf %189 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %199 = vector.extract %198[0, 0] : f16 from vector<1x4xf16>
      %200 = vector.insert %199, %arg11 [%97, %98] : f16 into vector<2x8xf16>
      %201 = vector.extract %198[0, 1] : f16 from vector<1x4xf16>
      %202 = vector.insert %201, %200 [%97, %100] : f16 into vector<2x8xf16>
      %203 = vector.extract %198[0, 2] : f16 from vector<1x4xf16>
      %204 = vector.insert %203, %202 [%97, %102] : f16 into vector<2x8xf16>
      %205 = vector.extract %198[0, 3] : f16 from vector<1x4xf16>
      %206 = vector.insert %205, %204 [%97, %104] : f16 into vector<2x8xf16>
      %207 = vector.insert %199, %arg12 [%97, %98] : f16 into vector<2x8xf16>
      %208 = vector.insert %201, %207 [%97, %100] : f16 into vector<2x8xf16>
      %209 = vector.insert %203, %208 [%97, %102] : f16 into vector<2x8xf16>
      %210 = vector.insert %205, %209 [%97, %104] : f16 into vector<2x8xf16>
      affine.yield %121, %141, %161, %182, %188, %197, %206, %210 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    %89 = affine.apply #map36()[%thread_id_x]
    %90 = arith.cmpi ult, %89, %c4 : index
    scf.if %90 {
      affine.for %arg4 = 0 to 4 {
        %97 = affine.apply #map9(%arg4)
        %98 = affine.apply #map10(%arg4)
        %99 = vector.extract %88#7[%97, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply #map11(%arg4)
        %101 = vector.extract %88#7[%97, %100] : f16 from vector<2x8xf16>
        %102 = affine.apply #map12(%arg4)
        %103 = vector.extract %88#7[%97, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply #map13(%arg4)
        %105 = vector.extract %88#7[%97, %104] : f16 from vector<2x8xf16>
        affine.store %99, %87[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %101, %87[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %103, %87[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %105, %87[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %91 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %92:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %97 = affine.apply #map9(%arg4)
      %98 = vector.extract %81#1[%97, 0] : f32 from vector<2x1xf32>
      %99 = vector.broadcast %98 : f32 to vector<1x4xf32>
      %100 = affine.apply #map10(%arg4)
      %101 = vector.insert %98, %arg5 [%97, %100] : f32 into vector<2x8xf32>
      %102 = affine.apply #map11(%arg4)
      %103 = vector.insert %98, %101 [%97, %102] : f32 into vector<2x8xf32>
      %104 = affine.apply #map12(%arg4)
      %105 = vector.insert %98, %103 [%97, %104] : f32 into vector<2x8xf32>
      %106 = affine.apply #map13(%arg4)
      %107 = vector.insert %98, %105 [%97, %106] : f32 into vector<2x8xf32>
      %108 = vector.extract %88#2[%97, %100] : f32 from vector<2x8xf32>
      %109 = vector.extract %88#2[%97, %102] : f32 from vector<2x8xf32>
      %110 = vector.extract %88#2[%97, %104] : f32 from vector<2x8xf32>
      %111 = vector.extract %88#2[%97, %106] : f32 from vector<2x8xf32>
      %112 = vector.from_elements %108, %109, %110, %111 : vector<1x4xf32>
      %113 = arith.divf %112, %99 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %114 = vector.extract %113[0, 0] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %arg6 [%97, %100] : f32 into vector<2x8xf32>
      %116 = vector.extract %113[0, 1] : f32 from vector<1x4xf32>
      %117 = vector.insert %116, %115 [%97, %102] : f32 into vector<2x8xf32>
      %118 = vector.extract %113[0, 2] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %117 [%97, %104] : f32 into vector<2x8xf32>
      %120 = vector.extract %113[0, 3] : f32 from vector<1x4xf32>
      %121 = vector.insert %120, %119 [%97, %106] : f32 into vector<2x8xf32>
      %122 = arith.truncf %113 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %123 = vector.extract %122[0, 0] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %arg7 [%97, %100] : f16 into vector<2x8xf16>
      %125 = vector.extract %122[0, 1] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%97, %102] : f16 into vector<2x8xf16>
      %127 = vector.extract %122[0, 2] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%97, %104] : f16 into vector<2x8xf16>
      %129 = vector.extract %122[0, 3] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%97, %106] : f16 into vector<2x8xf16>
      %131 = vector.insert %123, %arg8 [%97, %100] : f16 into vector<2x8xf16>
      %132 = vector.insert %125, %131 [%97, %102] : f16 into vector<2x8xf16>
      %133 = vector.insert %127, %132 [%97, %104] : f16 into vector<2x8xf16>
      %134 = vector.insert %129, %133 [%97, %106] : f16 into vector<2x8xf16>
      affine.yield %107, %121, %130, %134 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %90 {
      affine.for %arg4 = 0 to 4 {
        %97 = affine.apply #map9(%arg4)
        %98 = affine.apply #map10(%arg4)
        %99 = vector.extract %92#3[%97, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply #map11(%arg4)
        %101 = vector.extract %92#3[%97, %100] : f16 from vector<2x8xf16>
        %102 = affine.apply #map12(%arg4)
        %103 = vector.extract %92#3[%97, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply #map13(%arg4)
        %105 = vector.extract %92#3[%97, %104] : f16 from vector<2x8xf16>
        affine.store %99, %91[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %101, %91[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %103, %91[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %105, %91[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %93 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %94:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %97 = affine.apply #map9(%arg4)
      %98 = vector.extract %81#1[%97, 0] : f32 from vector<2x1xf32>
      %99 = vector.broadcast %98 : f32 to vector<1x4xf32>
      %100 = affine.apply #map10(%arg4)
      %101 = vector.insert %98, %arg5 [%97, %100] : f32 into vector<2x8xf32>
      %102 = affine.apply #map11(%arg4)
      %103 = vector.insert %98, %101 [%97, %102] : f32 into vector<2x8xf32>
      %104 = affine.apply #map12(%arg4)
      %105 = vector.insert %98, %103 [%97, %104] : f32 into vector<2x8xf32>
      %106 = affine.apply #map13(%arg4)
      %107 = vector.insert %98, %105 [%97, %106] : f32 into vector<2x8xf32>
      %108 = vector.extract %88#1[%97, %100] : f32 from vector<2x8xf32>
      %109 = vector.extract %88#1[%97, %102] : f32 from vector<2x8xf32>
      %110 = vector.extract %88#1[%97, %104] : f32 from vector<2x8xf32>
      %111 = vector.extract %88#1[%97, %106] : f32 from vector<2x8xf32>
      %112 = vector.from_elements %108, %109, %110, %111 : vector<1x4xf32>
      %113 = arith.divf %112, %99 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %114 = vector.extract %113[0, 0] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %arg6 [%97, %100] : f32 into vector<2x8xf32>
      %116 = vector.extract %113[0, 1] : f32 from vector<1x4xf32>
      %117 = vector.insert %116, %115 [%97, %102] : f32 into vector<2x8xf32>
      %118 = vector.extract %113[0, 2] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %117 [%97, %104] : f32 into vector<2x8xf32>
      %120 = vector.extract %113[0, 3] : f32 from vector<1x4xf32>
      %121 = vector.insert %120, %119 [%97, %106] : f32 into vector<2x8xf32>
      %122 = arith.truncf %113 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %123 = vector.extract %122[0, 0] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %arg7 [%97, %100] : f16 into vector<2x8xf16>
      %125 = vector.extract %122[0, 1] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%97, %102] : f16 into vector<2x8xf16>
      %127 = vector.extract %122[0, 2] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%97, %104] : f16 into vector<2x8xf16>
      %129 = vector.extract %122[0, 3] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%97, %106] : f16 into vector<2x8xf16>
      %131 = vector.insert %123, %arg8 [%97, %100] : f16 into vector<2x8xf16>
      %132 = vector.insert %125, %131 [%97, %102] : f16 into vector<2x8xf16>
      %133 = vector.insert %127, %132 [%97, %104] : f16 into vector<2x8xf16>
      %134 = vector.insert %129, %133 [%97, %106] : f16 into vector<2x8xf16>
      affine.yield %107, %121, %130, %134 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %90 {
      affine.for %arg4 = 0 to 4 {
        %97 = affine.apply #map9(%arg4)
        %98 = affine.apply #map10(%arg4)
        %99 = vector.extract %94#3[%97, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply #map11(%arg4)
        %101 = vector.extract %94#3[%97, %100] : f16 from vector<2x8xf16>
        %102 = affine.apply #map12(%arg4)
        %103 = vector.extract %94#3[%97, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply #map13(%arg4)
        %105 = vector.extract %94#3[%97, %104] : f16 from vector<2x8xf16>
        affine.store %99, %93[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %101, %93[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %103, %93[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %105, %93[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %95 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %96:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %97 = affine.apply #map9(%arg4)
      %98 = vector.extract %81#1[%97, 0] : f32 from vector<2x1xf32>
      %99 = vector.broadcast %98 : f32 to vector<1x4xf32>
      %100 = affine.apply #map10(%arg4)
      %101 = vector.insert %98, %arg5 [%97, %100] : f32 into vector<2x8xf32>
      %102 = affine.apply #map11(%arg4)
      %103 = vector.insert %98, %101 [%97, %102] : f32 into vector<2x8xf32>
      %104 = affine.apply #map12(%arg4)
      %105 = vector.insert %98, %103 [%97, %104] : f32 into vector<2x8xf32>
      %106 = affine.apply #map13(%arg4)
      %107 = vector.insert %98, %105 [%97, %106] : f32 into vector<2x8xf32>
      %108 = vector.extract %88#0[%97, %100] : f32 from vector<2x8xf32>
      %109 = vector.extract %88#0[%97, %102] : f32 from vector<2x8xf32>
      %110 = vector.extract %88#0[%97, %104] : f32 from vector<2x8xf32>
      %111 = vector.extract %88#0[%97, %106] : f32 from vector<2x8xf32>
      %112 = vector.from_elements %108, %109, %110, %111 : vector<1x4xf32>
      %113 = arith.divf %112, %99 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %114 = vector.extract %113[0, 0] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %arg6 [%97, %100] : f32 into vector<2x8xf32>
      %116 = vector.extract %113[0, 1] : f32 from vector<1x4xf32>
      %117 = vector.insert %116, %115 [%97, %102] : f32 into vector<2x8xf32>
      %118 = vector.extract %113[0, 2] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %117 [%97, %104] : f32 into vector<2x8xf32>
      %120 = vector.extract %113[0, 3] : f32 from vector<1x4xf32>
      %121 = vector.insert %120, %119 [%97, %106] : f32 into vector<2x8xf32>
      %122 = arith.truncf %113 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %123 = vector.extract %122[0, 0] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %arg7 [%97, %100] : f16 into vector<2x8xf16>
      %125 = vector.extract %122[0, 1] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%97, %102] : f16 into vector<2x8xf16>
      %127 = vector.extract %122[0, 2] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%97, %104] : f16 into vector<2x8xf16>
      %129 = vector.extract %122[0, 3] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%97, %106] : f16 into vector<2x8xf16>
      %131 = vector.insert %123, %arg8 [%97, %100] : f16 into vector<2x8xf16>
      %132 = vector.insert %125, %131 [%97, %102] : f16 into vector<2x8xf16>
      %133 = vector.insert %127, %132 [%97, %104] : f16 into vector<2x8xf16>
      %134 = vector.insert %129, %133 [%97, %106] : f16 into vector<2x8xf16>
      affine.yield %107, %121, %130, %134 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %90 {
      affine.for %arg4 = 0 to 4 {
        %97 = affine.apply #map9(%arg4)
        %98 = affine.apply #map10(%arg4)
        %99 = vector.extract %96#3[%97, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply #map11(%arg4)
        %101 = vector.extract %96#3[%97, %100] : f16 from vector<2x8xf16>
        %102 = affine.apply #map12(%arg4)
        %103 = vector.extract %96#3[%97, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply #map13(%arg4)
        %105 = vector.extract %96#3[%97, %104] : f16 from vector<2x8xf16>
        affine.store %99, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %101, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %103, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %105, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

