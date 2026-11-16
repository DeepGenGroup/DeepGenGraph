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
      %101 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %102 = vector.insert %cst_10, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %103 = vector.insert %cst_10, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %104 = vector.insert %cst_10, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %101, %102, %103, %104 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
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
    %19:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %18) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %101 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %102 = affine.apply #map4(%arg4)
      %103 = affine.apply #map5(%arg4)
      %104 = vector.extract %101[%102, %103] : f16 from vector<2x8xf16>
      %105 = affine.apply #map6(%arg4)
      %106 = vector.extract %101[%102, %105] : f16 from vector<2x8xf16>
      %107 = affine.apply #map7(%arg4)
      %108 = vector.extract %101[%102, %107] : f16 from vector<2x8xf16>
      %109 = affine.apply #map8(%arg4)
      %110 = vector.extract %101[%102, %109] : f16 from vector<2x8xf16>
      %111 = vector.from_elements %104, %106, %108, %110 : vector<1x4xf16>
      %112 = affine.apply #map9(%arg4)
      %113 = affine.apply #map10(%arg4)
      %114 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %arg5 [%112, %113] : f16 into vector<2x8xf16>
      %116 = affine.apply #map11(%arg4)
      %117 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %115 [%112, %116] : f16 into vector<2x8xf16>
      %119 = affine.apply #map12(%arg4)
      %120 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %118 [%112, %119] : f16 into vector<2x8xf16>
      %122 = affine.apply #map13(%arg4)
      %123 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %121 [%112, %122] : f16 into vector<2x8xf16>
      %125 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %126 = vector.insert %8, %arg6 [%112, %113] : f16 into vector<2x8xf16>
      %127 = vector.insert %8, %126 [%112, %116] : f16 into vector<2x8xf16>
      %128 = vector.insert %8, %127 [%112, %119] : f16 into vector<2x8xf16>
      %129 = vector.insert %8, %128 [%112, %122] : f16 into vector<2x8xf16>
      %130 = arith.mulf %arg8, %125 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %131 = vector.extract %130[0, 0] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %arg7 [%112, %113] : f16 into vector<2x8xf16>
      %133 = vector.extract %130[0, 1] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%112, %116] : f16 into vector<2x8xf16>
      %135 = vector.extract %130[0, 2] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%112, %119] : f16 into vector<2x8xf16>
      %137 = vector.extract %130[0, 3] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%112, %122] : f16 into vector<2x8xf16>
      affine.yield %124, %129, %138, %111 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %20 = vector.broadcast %8 : f16 to vector<1x4xf16>
    %21 = arith.mulf %19#3, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %22 = vector.extract %21[0, 0] : f16 from vector<1x4xf16>
    %23 = vector.insert %22, %19#2 [1, 4] : f16 into vector<2x8xf16>
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
    %37:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %36) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %101 = frisk.to_threadTile %alloc_12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %102 = affine.apply #map4(%arg4)
      %103 = affine.apply #map5(%arg4)
      %104 = vector.extract %101[%102, %103] : f16 from vector<2x8xf16>
      %105 = affine.apply #map6(%arg4)
      %106 = vector.extract %101[%102, %105] : f16 from vector<2x8xf16>
      %107 = affine.apply #map7(%arg4)
      %108 = vector.extract %101[%102, %107] : f16 from vector<2x8xf16>
      %109 = affine.apply #map8(%arg4)
      %110 = vector.extract %101[%102, %109] : f16 from vector<2x8xf16>
      %111 = vector.from_elements %104, %106, %108, %110 : vector<1x4xf16>
      %112 = affine.apply #map9(%arg4)
      %113 = affine.apply #map10(%arg4)
      %114 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %arg5 [%112, %113] : f16 into vector<2x8xf16>
      %116 = affine.apply #map11(%arg4)
      %117 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %115 [%112, %116] : f16 into vector<2x8xf16>
      %119 = affine.apply #map12(%arg4)
      %120 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %118 [%112, %119] : f16 into vector<2x8xf16>
      %122 = affine.apply #map13(%arg4)
      %123 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %121 [%112, %122] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %arg6 [%112, %113] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%112, %116] : f16 into vector<2x8xf16>
      %127 = vector.insert %8, %126 [%112, %119] : f16 into vector<2x8xf16>
      %128 = vector.insert %8, %127 [%112, %122] : f16 into vector<2x8xf16>
      %129 = arith.mulf %arg8, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %130 = vector.extract %129[0, 0] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %arg7 [%112, %113] : f16 into vector<2x8xf16>
      %132 = vector.extract %129[0, 1] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%112, %116] : f16 into vector<2x8xf16>
      %134 = vector.extract %129[0, 2] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%112, %119] : f16 into vector<2x8xf16>
      %136 = vector.extract %129[0, 3] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%112, %122] : f16 into vector<2x8xf16>
      affine.yield %124, %128, %137, %111 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %38 = arith.mulf %37#3, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %39 = vector.extract %38[0, 0] : f16 from vector<1x4xf16>
    %40 = vector.insert %39, %37#2 [1, 4] : f16 into vector<2x8xf16>
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
    %54:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %53) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %101 = frisk.to_threadTile %alloc_13 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %102 = affine.apply #map4(%arg4)
      %103 = affine.apply #map5(%arg4)
      %104 = vector.extract %101[%102, %103] : f16 from vector<2x8xf16>
      %105 = affine.apply #map6(%arg4)
      %106 = vector.extract %101[%102, %105] : f16 from vector<2x8xf16>
      %107 = affine.apply #map7(%arg4)
      %108 = vector.extract %101[%102, %107] : f16 from vector<2x8xf16>
      %109 = affine.apply #map8(%arg4)
      %110 = vector.extract %101[%102, %109] : f16 from vector<2x8xf16>
      %111 = vector.from_elements %104, %106, %108, %110 : vector<1x4xf16>
      %112 = affine.apply #map9(%arg4)
      %113 = affine.apply #map10(%arg4)
      %114 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %arg5 [%112, %113] : f16 into vector<2x8xf16>
      %116 = affine.apply #map11(%arg4)
      %117 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %115 [%112, %116] : f16 into vector<2x8xf16>
      %119 = affine.apply #map12(%arg4)
      %120 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %118 [%112, %119] : f16 into vector<2x8xf16>
      %122 = affine.apply #map13(%arg4)
      %123 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %121 [%112, %122] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %arg6 [%112, %113] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%112, %116] : f16 into vector<2x8xf16>
      %127 = vector.insert %8, %126 [%112, %119] : f16 into vector<2x8xf16>
      %128 = vector.insert %8, %127 [%112, %122] : f16 into vector<2x8xf16>
      %129 = arith.mulf %arg8, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %130 = vector.extract %129[0, 0] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %arg7 [%112, %113] : f16 into vector<2x8xf16>
      %132 = vector.extract %129[0, 1] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%112, %116] : f16 into vector<2x8xf16>
      %134 = vector.extract %129[0, 2] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%112, %119] : f16 into vector<2x8xf16>
      %136 = vector.extract %129[0, 3] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%112, %122] : f16 into vector<2x8xf16>
      affine.yield %124, %128, %137, %111 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %55 = arith.mulf %54#3, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
    %57 = vector.insert %56, %54#2 [1, 4] : f16 into vector<2x8xf16>
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
    %71:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %70) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %101 = frisk.to_threadTile %alloc_14 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %102 = affine.apply #map4(%arg4)
      %103 = affine.apply #map5(%arg4)
      %104 = vector.extract %101[%102, %103] : f16 from vector<2x8xf16>
      %105 = affine.apply #map6(%arg4)
      %106 = vector.extract %101[%102, %105] : f16 from vector<2x8xf16>
      %107 = affine.apply #map7(%arg4)
      %108 = vector.extract %101[%102, %107] : f16 from vector<2x8xf16>
      %109 = affine.apply #map8(%arg4)
      %110 = vector.extract %101[%102, %109] : f16 from vector<2x8xf16>
      %111 = vector.from_elements %104, %106, %108, %110 : vector<1x4xf16>
      %112 = affine.apply #map9(%arg4)
      %113 = affine.apply #map10(%arg4)
      %114 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %arg5 [%112, %113] : f16 into vector<2x8xf16>
      %116 = affine.apply #map11(%arg4)
      %117 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %115 [%112, %116] : f16 into vector<2x8xf16>
      %119 = affine.apply #map12(%arg4)
      %120 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %118 [%112, %119] : f16 into vector<2x8xf16>
      %122 = affine.apply #map13(%arg4)
      %123 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %121 [%112, %122] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %arg6 [%112, %113] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%112, %116] : f16 into vector<2x8xf16>
      %127 = vector.insert %8, %126 [%112, %119] : f16 into vector<2x8xf16>
      %128 = vector.insert %8, %127 [%112, %122] : f16 into vector<2x8xf16>
      %129 = arith.mulf %arg8, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %130 = vector.extract %129[0, 0] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %arg7 [%112, %113] : f16 into vector<2x8xf16>
      %132 = vector.extract %129[0, 1] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%112, %116] : f16 into vector<2x8xf16>
      %134 = vector.extract %129[0, 2] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%112, %119] : f16 into vector<2x8xf16>
      %136 = vector.extract %129[0, 3] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%112, %122] : f16 into vector<2x8xf16>
      affine.yield %124, %128, %137, %111 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %72 = arith.mulf %71#3, %20 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %73 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
    %74 = vector.insert %73, %71#2 [1, 4] : f16 into vector<2x8xf16>
    %75 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
    %76 = vector.insert %75, %74 [1, 5] : f16 into vector<2x8xf16>
    %77 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
    %78 = vector.insert %77, %76 [1, 6] : f16 into vector<2x8xf16>
    %79 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
    %80 = vector.insert %79, %78 [1, 7] : f16 into vector<2x8xf16>
    %81:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#3, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %101 = frisk.to_threadTile %cst_6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map2, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %102 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c0 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c0 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %102 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<32x128xf16, 3>
      %103:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %133 = frisk.to_threadTile %alloc_15 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %134 = affine.apply #map15(%arg11)
        %135 = vector.extract %133[0, %134] : f16 from vector<8x8xf16>
        %136 = vector.extract %133[1, %134] : f16 from vector<8x8xf16>
        %137 = vector.extract %133[2, %134] : f16 from vector<8x8xf16>
        %138 = vector.extract %133[3, %134] : f16 from vector<8x8xf16>
        %139 = vector.from_elements %135, %136, %137, %138 : vector<4x1xf16>
        %140:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %139) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %170 = frisk.to_threadTile %alloc_15 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %171 = affine.apply #map16(%arg14)
          %172 = vector.extract %170[%171, %134] : f16 from vector<8x8xf16>
          %173 = affine.apply #map17(%arg14)
          %174 = vector.extract %170[%173, %134] : f16 from vector<8x8xf16>
          %175 = affine.apply #map18(%arg14)
          %176 = vector.extract %170[%175, %134] : f16 from vector<8x8xf16>
          %177 = affine.apply #map19(%arg14)
          %178 = vector.extract %170[%177, %134] : f16 from vector<8x8xf16>
          %179 = vector.from_elements %172, %174, %176, %178 : vector<4x1xf16>
          %180 = affine.apply #map20(%arg11)
          %181 = affine.apply #map21(%arg14)
          %182 = vector.extract %29[%180, %181] : f16 from vector<2x8xf16>
          %183 = affine.apply #map22(%arg14)
          %184 = vector.extract %29[%180, %183] : f16 from vector<2x8xf16>
          %185 = affine.apply #map23(%arg14)
          %186 = vector.extract %29[%180, %185] : f16 from vector<2x8xf16>
          %187 = affine.apply #map24(%arg14)
          %188 = vector.extract %29[%180, %187] : f16 from vector<2x8xf16>
          %189 = vector.from_elements %182, %184, %186, %188 : vector<1x4xf16>
          %190 = frisk.warp_mma_rr %189, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %190, %179 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %141 = affine.apply #map20(%arg11)
        %142 = vector.extract %29[%141, 4] : f16 from vector<2x8xf16>
        %143 = vector.extract %29[%141, 5] : f16 from vector<2x8xf16>
        %144 = vector.extract %29[%141, 6] : f16 from vector<2x8xf16>
        %145 = vector.extract %29[%141, 7] : f16 from vector<2x8xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = frisk.warp_mma_rr %146, %140#1, %140#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %148 = affine.apply #map25(%arg11)
        %149 = vector.extract %101[%141, %148] : f32 from vector<2x32xf32>
        %150 = affine.apply #map26(%arg11)
        %151 = vector.extract %101[%141, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map27(%arg11)
        %153 = vector.extract %101[%141, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply #map28(%arg11)
        %155 = vector.extract %101[%141, %154] : f32 from vector<2x32xf32>
        %156 = vector.from_elements %149, %151, %153, %155 : vector<1x4xf32>
        %157 = arith.addf %156, %147 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%141, %148] : f32 into vector<2x32xf32>
        %160 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%141, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%141, %152] : f32 into vector<2x32xf32>
        %164 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%141, %154] : f32 into vector<2x32xf32>
        %166 = vector.insert %158, %arg13 [%141, %148] : f32 into vector<2x32xf32>
        %167 = vector.insert %160, %166 [%141, %150] : f32 into vector<2x32xf32>
        %168 = vector.insert %162, %167 [%141, %152] : f32 into vector<2x32xf32>
        %169 = vector.insert %164, %168 [%141, %154] : f32 into vector<2x32xf32>
        affine.yield %165, %169 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %104 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c32 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c32 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %104 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_16 : memref<32x128xf16, 3>
      %105:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %133 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %134 = affine.apply #map15(%arg11)
        %135 = vector.extract %133[0, %134] : f16 from vector<8x8xf16>
        %136 = vector.extract %133[1, %134] : f16 from vector<8x8xf16>
        %137 = vector.extract %133[2, %134] : f16 from vector<8x8xf16>
        %138 = vector.extract %133[3, %134] : f16 from vector<8x8xf16>
        %139 = vector.from_elements %135, %136, %137, %138 : vector<4x1xf16>
        %140:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %139) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %170 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %171 = affine.apply #map16(%arg14)
          %172 = vector.extract %170[%171, %134] : f16 from vector<8x8xf16>
          %173 = affine.apply #map17(%arg14)
          %174 = vector.extract %170[%173, %134] : f16 from vector<8x8xf16>
          %175 = affine.apply #map18(%arg14)
          %176 = vector.extract %170[%175, %134] : f16 from vector<8x8xf16>
          %177 = affine.apply #map19(%arg14)
          %178 = vector.extract %170[%177, %134] : f16 from vector<8x8xf16>
          %179 = vector.from_elements %172, %174, %176, %178 : vector<4x1xf16>
          %180 = affine.apply #map20(%arg11)
          %181 = affine.apply #map21(%arg14)
          %182 = vector.extract %46[%180, %181] : f16 from vector<2x8xf16>
          %183 = affine.apply #map22(%arg14)
          %184 = vector.extract %46[%180, %183] : f16 from vector<2x8xf16>
          %185 = affine.apply #map23(%arg14)
          %186 = vector.extract %46[%180, %185] : f16 from vector<2x8xf16>
          %187 = affine.apply #map24(%arg14)
          %188 = vector.extract %46[%180, %187] : f16 from vector<2x8xf16>
          %189 = vector.from_elements %182, %184, %186, %188 : vector<1x4xf16>
          %190 = frisk.warp_mma_rr %189, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %190, %179 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %141 = affine.apply #map20(%arg11)
        %142 = vector.extract %46[%141, 4] : f16 from vector<2x8xf16>
        %143 = vector.extract %46[%141, 5] : f16 from vector<2x8xf16>
        %144 = vector.extract %46[%141, 6] : f16 from vector<2x8xf16>
        %145 = vector.extract %46[%141, 7] : f16 from vector<2x8xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = frisk.warp_mma_rr %146, %140#1, %140#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %148 = affine.apply #map25(%arg11)
        %149 = vector.extract %103#1[%141, %148] : f32 from vector<2x32xf32>
        %150 = affine.apply #map26(%arg11)
        %151 = vector.extract %103#1[%141, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map27(%arg11)
        %153 = vector.extract %103#1[%141, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply #map28(%arg11)
        %155 = vector.extract %103#1[%141, %154] : f32 from vector<2x32xf32>
        %156 = vector.from_elements %149, %151, %153, %155 : vector<1x4xf32>
        %157 = arith.addf %156, %147 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%141, %148] : f32 into vector<2x32xf32>
        %160 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%141, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%141, %152] : f32 into vector<2x32xf32>
        %164 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%141, %154] : f32 into vector<2x32xf32>
        %166 = vector.insert %158, %arg13 [%141, %148] : f32 into vector<2x32xf32>
        %167 = vector.insert %160, %166 [%141, %150] : f32 into vector<2x32xf32>
        %168 = vector.insert %162, %167 [%141, %152] : f32 into vector<2x32xf32>
        %169 = vector.insert %164, %168 [%141, %154] : f32 into vector<2x32xf32>
        affine.yield %165, %169 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %106 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c64 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c64 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %106 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_17 : memref<32x128xf16, 3>
      %107:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %133 = frisk.to_threadTile %alloc_17 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %134 = affine.apply #map15(%arg11)
        %135 = vector.extract %133[0, %134] : f16 from vector<8x8xf16>
        %136 = vector.extract %133[1, %134] : f16 from vector<8x8xf16>
        %137 = vector.extract %133[2, %134] : f16 from vector<8x8xf16>
        %138 = vector.extract %133[3, %134] : f16 from vector<8x8xf16>
        %139 = vector.from_elements %135, %136, %137, %138 : vector<4x1xf16>
        %140:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %139) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %170 = frisk.to_threadTile %alloc_17 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %171 = affine.apply #map16(%arg14)
          %172 = vector.extract %170[%171, %134] : f16 from vector<8x8xf16>
          %173 = affine.apply #map17(%arg14)
          %174 = vector.extract %170[%173, %134] : f16 from vector<8x8xf16>
          %175 = affine.apply #map18(%arg14)
          %176 = vector.extract %170[%175, %134] : f16 from vector<8x8xf16>
          %177 = affine.apply #map19(%arg14)
          %178 = vector.extract %170[%177, %134] : f16 from vector<8x8xf16>
          %179 = vector.from_elements %172, %174, %176, %178 : vector<4x1xf16>
          %180 = affine.apply #map20(%arg11)
          %181 = affine.apply #map21(%arg14)
          %182 = vector.extract %63[%180, %181] : f16 from vector<2x8xf16>
          %183 = affine.apply #map22(%arg14)
          %184 = vector.extract %63[%180, %183] : f16 from vector<2x8xf16>
          %185 = affine.apply #map23(%arg14)
          %186 = vector.extract %63[%180, %185] : f16 from vector<2x8xf16>
          %187 = affine.apply #map24(%arg14)
          %188 = vector.extract %63[%180, %187] : f16 from vector<2x8xf16>
          %189 = vector.from_elements %182, %184, %186, %188 : vector<1x4xf16>
          %190 = frisk.warp_mma_rr %189, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %190, %179 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %141 = affine.apply #map20(%arg11)
        %142 = vector.extract %63[%141, 4] : f16 from vector<2x8xf16>
        %143 = vector.extract %63[%141, 5] : f16 from vector<2x8xf16>
        %144 = vector.extract %63[%141, 6] : f16 from vector<2x8xf16>
        %145 = vector.extract %63[%141, 7] : f16 from vector<2x8xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = frisk.warp_mma_rr %146, %140#1, %140#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %148 = affine.apply #map25(%arg11)
        %149 = vector.extract %105#1[%141, %148] : f32 from vector<2x32xf32>
        %150 = affine.apply #map26(%arg11)
        %151 = vector.extract %105#1[%141, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map27(%arg11)
        %153 = vector.extract %105#1[%141, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply #map28(%arg11)
        %155 = vector.extract %105#1[%141, %154] : f32 from vector<2x32xf32>
        %156 = vector.from_elements %149, %151, %153, %155 : vector<1x4xf32>
        %157 = arith.addf %156, %147 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%141, %148] : f32 into vector<2x32xf32>
        %160 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%141, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%141, %152] : f32 into vector<2x32xf32>
        %164 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%141, %154] : f32 into vector<2x32xf32>
        %166 = vector.insert %158, %arg13 [%141, %148] : f32 into vector<2x32xf32>
        %167 = vector.insert %160, %166 [%141, %150] : f32 into vector<2x32xf32>
        %168 = vector.insert %162, %167 [%141, %152] : f32 into vector<2x32xf32>
        %169 = vector.insert %164, %168 [%141, %154] : f32 into vector<2x32xf32>
        affine.yield %165, %169 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %108 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c96 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c96 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %108 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_18 : memref<32x128xf16, 3>
      %109:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %133 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %134 = affine.apply #map15(%arg11)
        %135 = vector.extract %133[0, %134] : f16 from vector<8x8xf16>
        %136 = vector.extract %133[1, %134] : f16 from vector<8x8xf16>
        %137 = vector.extract %133[2, %134] : f16 from vector<8x8xf16>
        %138 = vector.extract %133[3, %134] : f16 from vector<8x8xf16>
        %139 = vector.from_elements %135, %136, %137, %138 : vector<4x1xf16>
        %140:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %139) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %170 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = #map1, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map14, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %171 = affine.apply #map16(%arg14)
          %172 = vector.extract %170[%171, %134] : f16 from vector<8x8xf16>
          %173 = affine.apply #map17(%arg14)
          %174 = vector.extract %170[%173, %134] : f16 from vector<8x8xf16>
          %175 = affine.apply #map18(%arg14)
          %176 = vector.extract %170[%175, %134] : f16 from vector<8x8xf16>
          %177 = affine.apply #map19(%arg14)
          %178 = vector.extract %170[%177, %134] : f16 from vector<8x8xf16>
          %179 = vector.from_elements %172, %174, %176, %178 : vector<4x1xf16>
          %180 = affine.apply #map20(%arg11)
          %181 = affine.apply #map21(%arg14)
          %182 = vector.extract %80[%180, %181] : f16 from vector<2x8xf16>
          %183 = affine.apply #map22(%arg14)
          %184 = vector.extract %80[%180, %183] : f16 from vector<2x8xf16>
          %185 = affine.apply #map23(%arg14)
          %186 = vector.extract %80[%180, %185] : f16 from vector<2x8xf16>
          %187 = affine.apply #map24(%arg14)
          %188 = vector.extract %80[%180, %187] : f16 from vector<2x8xf16>
          %189 = vector.from_elements %182, %184, %186, %188 : vector<1x4xf16>
          %190 = frisk.warp_mma_rr %189, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %190, %179 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %141 = affine.apply #map20(%arg11)
        %142 = vector.extract %80[%141, 4] : f16 from vector<2x8xf16>
        %143 = vector.extract %80[%141, 5] : f16 from vector<2x8xf16>
        %144 = vector.extract %80[%141, 6] : f16 from vector<2x8xf16>
        %145 = vector.extract %80[%141, 7] : f16 from vector<2x8xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = frisk.warp_mma_rr %146, %140#1, %140#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %148 = affine.apply #map25(%arg11)
        %149 = vector.extract %107#1[%141, %148] : f32 from vector<2x32xf32>
        %150 = affine.apply #map26(%arg11)
        %151 = vector.extract %107#1[%141, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map27(%arg11)
        %153 = vector.extract %107#1[%141, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply #map28(%arg11)
        %155 = vector.extract %107#1[%141, %154] : f32 from vector<2x32xf32>
        %156 = vector.from_elements %149, %151, %153, %155 : vector<1x4xf32>
        %157 = arith.addf %156, %147 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%141, %148] : f32 into vector<2x32xf32>
        %160 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%141, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%141, %152] : f32 into vector<2x32xf32>
        %164 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%141, %154] : f32 into vector<2x32xf32>
        %166 = vector.insert %158, %arg13 [%141, %148] : f32 into vector<2x32xf32>
        %167 = vector.insert %160, %166 [%141, %150] : f32 into vector<2x32xf32>
        %168 = vector.insert %162, %167 [%141, %152] : f32 into vector<2x32xf32>
        %169 = vector.insert %164, %168 [%141, %154] : f32 into vector<2x32xf32>
        affine.yield %165, %169 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %110 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %133 = affine.apply #map20(%arg11)
        %134 = affine.apply #map25(%arg11)
        %135 = vector.extract %109#1[%133, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply #map26(%arg11)
        %137 = vector.extract %109#1[%133, %136] : f32 from vector<2x32xf32>
        %138 = affine.apply #map27(%arg11)
        %139 = vector.extract %109#1[%133, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply #map28(%arg11)
        %141 = vector.extract %109#1[%133, %140] : f32 from vector<2x32xf32>
        %142 = vector.insert %135, %arg12 [%133, %134] : f32 into vector<2x32xf32>
        %143 = vector.insert %137, %142 [%133, %136] : f32 into vector<2x32xf32>
        %144 = vector.insert %139, %143 [%133, %138] : f32 into vector<2x32xf32>
        %145 = vector.insert %141, %144 [%133, %140] : f32 into vector<2x32xf32>
        affine.yield %145 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %111 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %133 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_10) -> (f32) {
          %137 = affine.apply #map21(%arg13)
          %138 = vector.extract %110[%arg11, %137] : f32 from vector<2x32xf32>
          %139 = arith.maxnumf %arg14, %138 : f32
          %140 = affine.apply #map22(%arg13)
          %141 = vector.extract %110[%arg11, %140] : f32 from vector<2x32xf32>
          %142 = arith.maxnumf %139, %141 : f32
          %143 = affine.apply #map23(%arg13)
          %144 = vector.extract %110[%arg11, %143] : f32 from vector<2x32xf32>
          %145 = arith.maxnumf %142, %144 : f32
          %146 = affine.apply #map24(%arg13)
          %147 = vector.extract %110[%arg11, %146] : f32 from vector<2x32xf32>
          %148 = arith.maxnumf %145, %147 : f32
          affine.yield %148 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %133, %c16_i32, %c64_i32 : f32
        %134 = arith.maxnumf %133, %shuffleResult : f32
        %shuffleResult_24, %valid_25 = gpu.shuffle xor %134, %c32_i32, %c64_i32 : f32
        %135 = arith.maxnumf %134, %shuffleResult_24 : f32
        %136 = vector.insert %135, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %136 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %112 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %133 = vector.extract %111[%arg11, 0] : f32 from vector<2x1xf32>
        %134 = vector.broadcast %133 : f32 to vector<1x1xf32>
        %135 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %136 = vector.broadcast %135 : f32 to vector<1x1xf32>
        %137 = arith.maxnumf %134, %136 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %138 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %139 = vector.insert %138, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %139 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %113:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %133 = affine.apply #map20(%arg11)
        %134 = vector.extract %112[%133, 0] : f32 from vector<2x1xf32>
        %135 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3) -> (vector<1x4xf32>) {
          %162 = vector.broadcast %134 : f32 to vector<1x4xf32>
          affine.yield %162 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %136 = affine.apply #map25(%arg11)
        %137 = vector.extract %135[0, 0] : f32 from vector<1x4xf32>
        %138 = vector.insert %137, %arg12 [%133, %136] : f32 into vector<2x32xf32>
        %139 = affine.apply #map26(%arg11)
        %140 = vector.extract %135[0, 1] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %138 [%133, %139] : f32 into vector<2x32xf32>
        %142 = affine.apply #map27(%arg11)
        %143 = vector.extract %135[0, 2] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%133, %142] : f32 into vector<2x32xf32>
        %145 = affine.apply #map28(%arg11)
        %146 = vector.extract %135[0, 3] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%133, %145] : f32 into vector<2x32xf32>
        %148 = vector.extract %109#1[%133, %136] : f32 from vector<2x32xf32>
        %149 = vector.extract %109#1[%133, %139] : f32 from vector<2x32xf32>
        %150 = vector.extract %109#1[%133, %142] : f32 from vector<2x32xf32>
        %151 = vector.extract %109#1[%133, %145] : f32 from vector<2x32xf32>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf32>
        %153 = arith.subf %152, %135 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %arg13 [%133, %136] : f32 into vector<2x32xf32>
        %156 = vector.extract %153[0, 1] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%133, %139] : f32 into vector<2x32xf32>
        %158 = vector.extract %153[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%133, %142] : f32 into vector<2x32xf32>
        %160 = vector.extract %153[0, 3] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%133, %145] : f32 into vector<2x32xf32>
        affine.yield %147, %161 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %114:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9, %arg14 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %133 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %134 = vector.broadcast %133 : f32 to vector<1x1xf32>
        %135 = vector.extract %112[%arg11, 0] : f32 from vector<2x1xf32>
        %136 = vector.broadcast %135 : f32 to vector<1x1xf32>
        %137 = arith.subf %134, %136 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %138 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %139 = vector.insert %138, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %140 = math.exp2 %137 {tiled = true} : vector<1x1xf32>
        %141 = vector.extract %140[0, 0] : f32 from vector<1x1xf32>
        %142 = vector.insert %141, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %143 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %144 = vector.broadcast %143 : f32 to vector<1x1xf32>
        %145 = arith.mulf %144, %140 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %146 = vector.extract %145[0, 0] : f32 from vector<1x1xf32>
        %147 = vector.insert %146, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %139, %142, %147 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %115:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %133 = affine.apply #map20(%arg11)
        %134 = affine.apply #map25(%arg11)
        %135 = vector.extract %113#1[%133, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply #map26(%arg11)
        %137 = vector.extract %113#1[%133, %136] : f32 from vector<2x32xf32>
        %138 = affine.apply #map27(%arg11)
        %139 = vector.extract %113#1[%133, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply #map28(%arg11)
        %141 = vector.extract %113#1[%133, %140] : f32 from vector<2x32xf32>
        %142 = vector.from_elements %135, %137, %139, %141 : vector<1x4xf32>
        %143 = math.exp2 %142 {tiled = true} : vector<1x4xf32>
        %144 = vector.extract %143[0, 0] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %arg12 [%133, %134] : f32 into vector<2x32xf32>
        %146 = vector.extract %143[0, 1] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%133, %136] : f32 into vector<2x32xf32>
        %148 = vector.extract %143[0, 2] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%133, %138] : f32 into vector<2x32xf32>
        %150 = vector.extract %143[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%133, %140] : f32 into vector<2x32xf32>
        %152 = vector.insert %144, %arg13 [%133, %134] : f32 into vector<2x32xf32>
        %153 = vector.insert %146, %152 [%133, %136] : f32 into vector<2x32xf32>
        %154 = vector.insert %148, %153 [%133, %138] : f32 into vector<2x32xf32>
        %155 = vector.insert %150, %154 [%133, %140] : f32 into vector<2x32xf32>
        affine.yield %151, %155 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %116 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %133 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst) -> (f32) {
          %137 = affine.apply #map21(%arg13)
          %138 = vector.extract %115#1[%arg11, %137] : f32 from vector<2x32xf32>
          %139 = arith.addf %arg14, %138 : f32
          %140 = affine.apply #map22(%arg13)
          %141 = vector.extract %115#1[%arg11, %140] : f32 from vector<2x32xf32>
          %142 = arith.addf %139, %141 : f32
          %143 = affine.apply #map23(%arg13)
          %144 = vector.extract %115#1[%arg11, %143] : f32 from vector<2x32xf32>
          %145 = arith.addf %142, %144 : f32
          %146 = affine.apply #map24(%arg13)
          %147 = vector.extract %115#1[%arg11, %146] : f32 from vector<2x32xf32>
          %148 = arith.addf %145, %147 : f32
          affine.yield %148 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %133, %c16_i32, %c64_i32 : f32
        %134 = arith.addf %133, %shuffleResult : f32
        %shuffleResult_24, %valid_25 = gpu.shuffle xor %134, %c32_i32, %c64_i32 : f32
        %135 = arith.addf %134, %shuffleResult_24 : f32
        %136 = vector.insert %135, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %136 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %117 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %133 = vector.extract %116[%arg11, 0] : f32 from vector<2x1xf32>
        %134 = vector.broadcast %133 : f32 to vector<1x1xf32>
        %135 = vector.extract %114#2[%arg11, 0] : f32 from vector<2x1xf32>
        %136 = vector.broadcast %135 : f32 to vector<1x1xf32>
        %137 = arith.addf %134, %136 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %138 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %139 = vector.insert %138, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %139 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %118 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %133 = affine.apply #map20(%arg11)
        %134 = affine.apply #map25(%arg11)
        %135 = vector.extract %115#0[%133, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply #map26(%arg11)
        %137 = vector.extract %115#0[%133, %136] : f32 from vector<2x32xf32>
        %138 = affine.apply #map27(%arg11)
        %139 = vector.extract %115#0[%133, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply #map28(%arg11)
        %141 = vector.extract %115#0[%133, %140] : f32 from vector<2x32xf32>
        %142 = vector.from_elements %135, %137, %139, %141 : vector<1x4xf32>
        %143 = arith.truncf %142 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %144 = vector.extract %143[0, 0] : f16 from vector<1x4xf16>
        %145 = vector.insert %144, %arg12 [%133, %134] : f16 into vector<2x32xf16>
        %146 = vector.extract %143[0, 1] : f16 from vector<1x4xf16>
        %147 = vector.insert %146, %145 [%133, %136] : f16 into vector<2x32xf16>
        %148 = vector.extract %143[0, 2] : f16 from vector<1x4xf16>
        %149 = vector.insert %148, %147 [%133, %138] : f16 into vector<2x32xf16>
        %150 = vector.extract %143[0, 3] : f16 from vector<1x4xf16>
        %151 = vector.insert %150, %149 [%133, %140] : f16 into vector<2x32xf16>
        affine.yield %151 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %119 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %119 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_19 : memref<128x32xf16, 3>
      %thread_id_x_20 = gpu.thread_id x
      %120 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %133 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %134 = affine.apply #map30()[%thread_id_x_20]
        %135 = frisk.read_matrix_from_shm %alloc_19[%133, %134] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %136 = vector.bitcast %135 : vector<4xi32> to vector<8xf16>
        %137 = vector.extract %136[0] : f16 from vector<8xf16>
        %138 = affine.apply #map31()[%thread_id_x_20]
        %139 = vector.extract %136[%138] : f16 from vector<8xf16>
        %140 = vector.extract %136[1] : f16 from vector<8xf16>
        %141 = affine.apply #map32()[%thread_id_x_20]
        %142 = vector.extract %136[%141] : f16 from vector<8xf16>
        %143 = vector.extract %136[2] : f16 from vector<8xf16>
        %144 = affine.apply #map33()[%thread_id_x_20]
        %145 = vector.extract %136[%144] : f16 from vector<8xf16>
        %146 = vector.extract %136[3] : f16 from vector<8xf16>
        %147 = affine.apply #map34()[%thread_id_x_20]
        %148 = vector.extract %136[%147] : f16 from vector<8xf16>
        %149 = affine.apply #map21(%arg11)
        %150 = vector.insert %137, %arg12 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        %152 = affine.apply #map22(%arg11)
        %153 = vector.insert %140, %151 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply #map23(%arg11)
        %156 = vector.insert %143, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %145, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply #map24(%arg11)
        %159 = vector.insert %146, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %148, %159 [%158, 1] : f16 into vector<32x2xf16>
        affine.yield %160 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %121:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %133 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %180 = affine.apply #map9(%arg11)
          %181 = affine.apply #map21(%arg16)
          %182 = vector.extract %118[%180, %181] : f16 from vector<2x32xf16>
          %183 = affine.apply #map22(%arg16)
          %184 = vector.extract %118[%180, %183] : f16 from vector<2x32xf16>
          %185 = affine.apply #map23(%arg16)
          %186 = vector.extract %118[%180, %185] : f16 from vector<2x32xf16>
          %187 = affine.apply #map24(%arg16)
          %188 = vector.extract %118[%180, %187] : f16 from vector<2x32xf16>
          %189 = vector.from_elements %182, %184, %186, %188 : vector<1x4xf16>
          %190 = affine.apply #map35(%arg11)
          %191 = vector.extract %120[%181, %190] : f16 from vector<32x2xf16>
          %192 = vector.extract %120[%183, %190] : f16 from vector<32x2xf16>
          %193 = vector.extract %120[%185, %190] : f16 from vector<32x2xf16>
          %194 = vector.extract %120[%187, %190] : f16 from vector<32x2xf16>
          %195 = vector.from_elements %191, %192, %193, %194 : vector<4x1xf16>
          %196 = frisk.warp_mma_rr %189, %195, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %196 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %134 = affine.apply #map9(%arg11)
        %135 = affine.apply #map10(%arg11)
        %136 = vector.extract %133[0, 0] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %arg12 [%134, %135] : f32 into vector<2x8xf32>
        %138 = affine.apply #map11(%arg11)
        %139 = vector.extract %133[0, 1] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %137 [%134, %138] : f32 into vector<2x8xf32>
        %141 = affine.apply #map12(%arg11)
        %142 = vector.extract %133[0, 2] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %140 [%134, %141] : f32 into vector<2x8xf32>
        %144 = affine.apply #map13(%arg11)
        %145 = vector.extract %133[0, 3] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %143 [%134, %144] : f32 into vector<2x8xf32>
        %147 = vector.extract %114#1[%134, 0] : f32 from vector<2x1xf32>
        %148 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %180 = vector.broadcast %147 : f32 to vector<1x4xf32>
          affine.yield %180 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %149 = vector.extract %148[0, 0] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %arg13 [%134, %135] : f32 into vector<2x8xf32>
        %151 = vector.extract %148[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%134, %138] : f32 into vector<2x8xf32>
        %153 = vector.extract %148[0, 2] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %152 [%134, %141] : f32 into vector<2x8xf32>
        %155 = vector.extract %148[0, 3] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%134, %144] : f32 into vector<2x8xf32>
        %157 = vector.extract %arg5[%134, %135] : f32 from vector<2x8xf32>
        %158 = vector.extract %arg5[%134, %138] : f32 from vector<2x8xf32>
        %159 = vector.extract %arg5[%134, %141] : f32 from vector<2x8xf32>
        %160 = vector.extract %arg5[%134, %144] : f32 from vector<2x8xf32>
        %161 = vector.from_elements %157, %158, %159, %160 : vector<1x4xf32>
        %162 = arith.mulf %161, %148 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %arg14 [%134, %135] : f32 into vector<2x8xf32>
        %165 = vector.extract %162[0, 1] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%134, %138] : f32 into vector<2x8xf32>
        %167 = vector.extract %162[0, 2] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%134, %141] : f32 into vector<2x8xf32>
        %169 = vector.extract %162[0, 3] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%134, %144] : f32 into vector<2x8xf32>
        %171 = arith.addf %162, %133 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %172 = vector.extract %171[0, 0] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %arg15 [%134, %135] : f32 into vector<2x8xf32>
        %174 = vector.extract %171[0, 1] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%134, %138] : f32 into vector<2x8xf32>
        %176 = vector.extract %171[0, 2] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%134, %141] : f32 into vector<2x8xf32>
        %178 = vector.extract %171[0, 3] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%134, %144] : f32 into vector<2x8xf32>
        affine.yield %146, %156, %170, %179 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %122 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %122 to %alloc_21 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_21 : memref<128x32xf16, 3>
      %123 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %133 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %134 = affine.apply #map30()[%thread_id_x_20]
        %135 = frisk.read_matrix_from_shm %alloc_21[%133, %134] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %136 = vector.bitcast %135 : vector<4xi32> to vector<8xf16>
        %137 = vector.extract %136[0] : f16 from vector<8xf16>
        %138 = affine.apply #map31()[%thread_id_x_20]
        %139 = vector.extract %136[%138] : f16 from vector<8xf16>
        %140 = vector.extract %136[1] : f16 from vector<8xf16>
        %141 = affine.apply #map32()[%thread_id_x_20]
        %142 = vector.extract %136[%141] : f16 from vector<8xf16>
        %143 = vector.extract %136[2] : f16 from vector<8xf16>
        %144 = affine.apply #map33()[%thread_id_x_20]
        %145 = vector.extract %136[%144] : f16 from vector<8xf16>
        %146 = vector.extract %136[3] : f16 from vector<8xf16>
        %147 = affine.apply #map34()[%thread_id_x_20]
        %148 = vector.extract %136[%147] : f16 from vector<8xf16>
        %149 = affine.apply #map21(%arg11)
        %150 = vector.insert %137, %arg12 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        %152 = affine.apply #map22(%arg11)
        %153 = vector.insert %140, %151 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply #map23(%arg11)
        %156 = vector.insert %143, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %145, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply #map24(%arg11)
        %159 = vector.insert %146, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %148, %159 [%158, 1] : f16 into vector<32x2xf16>
        affine.yield %160 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %124:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %133 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %180 = affine.apply #map9(%arg11)
          %181 = affine.apply #map21(%arg16)
          %182 = vector.extract %118[%180, %181] : f16 from vector<2x32xf16>
          %183 = affine.apply #map22(%arg16)
          %184 = vector.extract %118[%180, %183] : f16 from vector<2x32xf16>
          %185 = affine.apply #map23(%arg16)
          %186 = vector.extract %118[%180, %185] : f16 from vector<2x32xf16>
          %187 = affine.apply #map24(%arg16)
          %188 = vector.extract %118[%180, %187] : f16 from vector<2x32xf16>
          %189 = vector.from_elements %182, %184, %186, %188 : vector<1x4xf16>
          %190 = affine.apply #map35(%arg11)
          %191 = vector.extract %123[%181, %190] : f16 from vector<32x2xf16>
          %192 = vector.extract %123[%183, %190] : f16 from vector<32x2xf16>
          %193 = vector.extract %123[%185, %190] : f16 from vector<32x2xf16>
          %194 = vector.extract %123[%187, %190] : f16 from vector<32x2xf16>
          %195 = vector.from_elements %191, %192, %193, %194 : vector<4x1xf16>
          %196 = frisk.warp_mma_rr %189, %195, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %196 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %134 = affine.apply #map9(%arg11)
        %135 = affine.apply #map10(%arg11)
        %136 = vector.extract %133[0, 0] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %arg12 [%134, %135] : f32 into vector<2x8xf32>
        %138 = affine.apply #map11(%arg11)
        %139 = vector.extract %133[0, 1] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %137 [%134, %138] : f32 into vector<2x8xf32>
        %141 = affine.apply #map12(%arg11)
        %142 = vector.extract %133[0, 2] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %140 [%134, %141] : f32 into vector<2x8xf32>
        %144 = affine.apply #map13(%arg11)
        %145 = vector.extract %133[0, 3] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %143 [%134, %144] : f32 into vector<2x8xf32>
        %147 = vector.extract %114#1[%134, 0] : f32 from vector<2x1xf32>
        %148 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %180 = vector.broadcast %147 : f32 to vector<1x4xf32>
          affine.yield %180 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %149 = vector.extract %148[0, 0] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %arg13 [%134, %135] : f32 into vector<2x8xf32>
        %151 = vector.extract %148[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%134, %138] : f32 into vector<2x8xf32>
        %153 = vector.extract %148[0, 2] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %152 [%134, %141] : f32 into vector<2x8xf32>
        %155 = vector.extract %148[0, 3] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%134, %144] : f32 into vector<2x8xf32>
        %157 = vector.extract %arg8[%134, %135] : f32 from vector<2x8xf32>
        %158 = vector.extract %arg8[%134, %138] : f32 from vector<2x8xf32>
        %159 = vector.extract %arg8[%134, %141] : f32 from vector<2x8xf32>
        %160 = vector.extract %arg8[%134, %144] : f32 from vector<2x8xf32>
        %161 = vector.from_elements %157, %158, %159, %160 : vector<1x4xf32>
        %162 = arith.mulf %161, %148 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %arg14 [%134, %135] : f32 into vector<2x8xf32>
        %165 = vector.extract %162[0, 1] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%134, %138] : f32 into vector<2x8xf32>
        %167 = vector.extract %162[0, 2] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%134, %141] : f32 into vector<2x8xf32>
        %169 = vector.extract %162[0, 3] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%134, %144] : f32 into vector<2x8xf32>
        %171 = arith.addf %162, %133 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %172 = vector.extract %171[0, 0] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %arg15 [%134, %135] : f32 into vector<2x8xf32>
        %174 = vector.extract %171[0, 1] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%134, %138] : f32 into vector<2x8xf32>
        %176 = vector.extract %171[0, 2] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%134, %141] : f32 into vector<2x8xf32>
        %178 = vector.extract %171[0, 3] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%134, %144] : f32 into vector<2x8xf32>
        affine.yield %146, %156, %170, %179 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_22 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %125 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %125 to %alloc_22 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_22 : memref<128x32xf16, 3>
      %126 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %133 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %134 = affine.apply #map30()[%thread_id_x_20]
        %135 = frisk.read_matrix_from_shm %alloc_22[%133, %134] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %136 = vector.bitcast %135 : vector<4xi32> to vector<8xf16>
        %137 = vector.extract %136[0] : f16 from vector<8xf16>
        %138 = affine.apply #map31()[%thread_id_x_20]
        %139 = vector.extract %136[%138] : f16 from vector<8xf16>
        %140 = vector.extract %136[1] : f16 from vector<8xf16>
        %141 = affine.apply #map32()[%thread_id_x_20]
        %142 = vector.extract %136[%141] : f16 from vector<8xf16>
        %143 = vector.extract %136[2] : f16 from vector<8xf16>
        %144 = affine.apply #map33()[%thread_id_x_20]
        %145 = vector.extract %136[%144] : f16 from vector<8xf16>
        %146 = vector.extract %136[3] : f16 from vector<8xf16>
        %147 = affine.apply #map34()[%thread_id_x_20]
        %148 = vector.extract %136[%147] : f16 from vector<8xf16>
        %149 = affine.apply #map21(%arg11)
        %150 = vector.insert %137, %arg12 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        %152 = affine.apply #map22(%arg11)
        %153 = vector.insert %140, %151 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply #map23(%arg11)
        %156 = vector.insert %143, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %145, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply #map24(%arg11)
        %159 = vector.insert %146, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %148, %159 [%158, 1] : f16 into vector<32x2xf16>
        affine.yield %160 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %127:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %133 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %180 = affine.apply #map9(%arg11)
          %181 = affine.apply #map21(%arg16)
          %182 = vector.extract %118[%180, %181] : f16 from vector<2x32xf16>
          %183 = affine.apply #map22(%arg16)
          %184 = vector.extract %118[%180, %183] : f16 from vector<2x32xf16>
          %185 = affine.apply #map23(%arg16)
          %186 = vector.extract %118[%180, %185] : f16 from vector<2x32xf16>
          %187 = affine.apply #map24(%arg16)
          %188 = vector.extract %118[%180, %187] : f16 from vector<2x32xf16>
          %189 = vector.from_elements %182, %184, %186, %188 : vector<1x4xf16>
          %190 = affine.apply #map35(%arg11)
          %191 = vector.extract %126[%181, %190] : f16 from vector<32x2xf16>
          %192 = vector.extract %126[%183, %190] : f16 from vector<32x2xf16>
          %193 = vector.extract %126[%185, %190] : f16 from vector<32x2xf16>
          %194 = vector.extract %126[%187, %190] : f16 from vector<32x2xf16>
          %195 = vector.from_elements %191, %192, %193, %194 : vector<4x1xf16>
          %196 = frisk.warp_mma_rr %189, %195, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %196 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %134 = affine.apply #map9(%arg11)
        %135 = affine.apply #map10(%arg11)
        %136 = vector.extract %133[0, 0] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %arg12 [%134, %135] : f32 into vector<2x8xf32>
        %138 = affine.apply #map11(%arg11)
        %139 = vector.extract %133[0, 1] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %137 [%134, %138] : f32 into vector<2x8xf32>
        %141 = affine.apply #map12(%arg11)
        %142 = vector.extract %133[0, 2] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %140 [%134, %141] : f32 into vector<2x8xf32>
        %144 = affine.apply #map13(%arg11)
        %145 = vector.extract %133[0, 3] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %143 [%134, %144] : f32 into vector<2x8xf32>
        %147 = vector.extract %114#1[%134, 0] : f32 from vector<2x1xf32>
        %148 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %180 = vector.broadcast %147 : f32 to vector<1x4xf32>
          affine.yield %180 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %149 = vector.extract %148[0, 0] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %arg13 [%134, %135] : f32 into vector<2x8xf32>
        %151 = vector.extract %148[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%134, %138] : f32 into vector<2x8xf32>
        %153 = vector.extract %148[0, 2] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %152 [%134, %141] : f32 into vector<2x8xf32>
        %155 = vector.extract %148[0, 3] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%134, %144] : f32 into vector<2x8xf32>
        %157 = vector.extract %arg9[%134, %135] : f32 from vector<2x8xf32>
        %158 = vector.extract %arg9[%134, %138] : f32 from vector<2x8xf32>
        %159 = vector.extract %arg9[%134, %141] : f32 from vector<2x8xf32>
        %160 = vector.extract %arg9[%134, %144] : f32 from vector<2x8xf32>
        %161 = vector.from_elements %157, %158, %159, %160 : vector<1x4xf32>
        %162 = arith.mulf %161, %148 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %arg14 [%134, %135] : f32 into vector<2x8xf32>
        %165 = vector.extract %162[0, 1] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%134, %138] : f32 into vector<2x8xf32>
        %167 = vector.extract %162[0, 2] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%134, %141] : f32 into vector<2x8xf32>
        %169 = vector.extract %162[0, 3] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%134, %144] : f32 into vector<2x8xf32>
        %171 = arith.addf %162, %133 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %172 = vector.extract %171[0, 0] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %arg15 [%134, %135] : f32 into vector<2x8xf32>
        %174 = vector.extract %171[0, 1] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%134, %138] : f32 into vector<2x8xf32>
        %176 = vector.extract %171[0, 2] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%134, %141] : f32 into vector<2x8xf32>
        %178 = vector.extract %171[0, 3] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%134, %144] : f32 into vector<2x8xf32>
        affine.yield %146, %156, %170, %179 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %128 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %128 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_23 : memref<128x32xf16, 3>
      %129 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %133 = affine.apply #map29(%arg11)[%thread_id_x_20]
        %134 = affine.apply #map30()[%thread_id_x_20]
        %135 = frisk.read_matrix_from_shm %alloc_23[%133, %134] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %136 = vector.bitcast %135 : vector<4xi32> to vector<8xf16>
        %137 = vector.extract %136[0] : f16 from vector<8xf16>
        %138 = affine.apply #map31()[%thread_id_x_20]
        %139 = vector.extract %136[%138] : f16 from vector<8xf16>
        %140 = vector.extract %136[1] : f16 from vector<8xf16>
        %141 = affine.apply #map32()[%thread_id_x_20]
        %142 = vector.extract %136[%141] : f16 from vector<8xf16>
        %143 = vector.extract %136[2] : f16 from vector<8xf16>
        %144 = affine.apply #map33()[%thread_id_x_20]
        %145 = vector.extract %136[%144] : f16 from vector<8xf16>
        %146 = vector.extract %136[3] : f16 from vector<8xf16>
        %147 = affine.apply #map34()[%thread_id_x_20]
        %148 = vector.extract %136[%147] : f16 from vector<8xf16>
        %149 = affine.apply #map21(%arg11)
        %150 = vector.insert %137, %arg12 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        %152 = affine.apply #map22(%arg11)
        %153 = vector.insert %140, %151 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply #map23(%arg11)
        %156 = vector.insert %143, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %145, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply #map24(%arg11)
        %159 = vector.insert %146, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %148, %159 [%158, 1] : f16 into vector<32x2xf16>
        affine.yield %160 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %130:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %133 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %188 = affine.apply #map9(%arg11)
          %189 = affine.apply #map21(%arg17)
          %190 = vector.extract %118[%188, %189] : f16 from vector<2x32xf16>
          %191 = affine.apply #map22(%arg17)
          %192 = vector.extract %118[%188, %191] : f16 from vector<2x32xf16>
          %193 = affine.apply #map23(%arg17)
          %194 = vector.extract %118[%188, %193] : f16 from vector<2x32xf16>
          %195 = affine.apply #map24(%arg17)
          %196 = vector.extract %118[%188, %195] : f16 from vector<2x32xf16>
          %197 = vector.from_elements %190, %192, %194, %196 : vector<1x4xf16>
          %198 = affine.apply #map35(%arg11)
          %199 = vector.extract %129[%189, %198] : f16 from vector<32x2xf16>
          %200 = vector.extract %129[%191, %198] : f16 from vector<32x2xf16>
          %201 = vector.extract %129[%193, %198] : f16 from vector<32x2xf16>
          %202 = vector.extract %129[%195, %198] : f16 from vector<32x2xf16>
          %203 = vector.from_elements %199, %200, %201, %202 : vector<4x1xf16>
          %204 = frisk.warp_mma_rr %197, %203, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %204 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %134 = affine.apply #map9(%arg11)
        %135 = affine.apply #map10(%arg11)
        %136 = vector.extract %133[0, 0] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %arg12 [%134, %135] : f32 into vector<2x8xf32>
        %138 = affine.apply #map11(%arg11)
        %139 = vector.extract %133[0, 1] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %137 [%134, %138] : f32 into vector<2x8xf32>
        %141 = affine.apply #map12(%arg11)
        %142 = vector.extract %133[0, 2] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %140 [%134, %141] : f32 into vector<2x8xf32>
        %144 = affine.apply #map13(%arg11)
        %145 = vector.extract %133[0, 3] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %143 [%134, %144] : f32 into vector<2x8xf32>
        %147 = vector.extract %114#1[%134, 0] : f32 from vector<2x1xf32>
        %148 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %188 = vector.broadcast %147 : f32 to vector<1x4xf32>
          affine.yield %188 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %149 = vector.extract %148[0, 0] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %arg13 [%134, %135] : f32 into vector<2x8xf32>
        %151 = vector.extract %148[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%134, %138] : f32 into vector<2x8xf32>
        %153 = vector.extract %148[0, 2] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %152 [%134, %141] : f32 into vector<2x8xf32>
        %155 = vector.extract %148[0, 3] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%134, %144] : f32 into vector<2x8xf32>
        %157 = vector.extract %arg10[%134, %135] : f32 from vector<2x8xf32>
        %158 = vector.extract %arg10[%134, %138] : f32 from vector<2x8xf32>
        %159 = vector.extract %arg10[%134, %141] : f32 from vector<2x8xf32>
        %160 = vector.extract %arg10[%134, %144] : f32 from vector<2x8xf32>
        %161 = vector.from_elements %157, %158, %159, %160 : vector<1x4xf32>
        %162 = arith.mulf %161, %148 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %arg14 [%134, %135] : f32 into vector<2x8xf32>
        %165 = vector.extract %162[0, 1] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%134, %138] : f32 into vector<2x8xf32>
        %167 = vector.extract %162[0, 2] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%134, %141] : f32 into vector<2x8xf32>
        %169 = vector.extract %162[0, 3] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%134, %144] : f32 into vector<2x8xf32>
        %171 = arith.addf %162, %133 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %172 = vector.extract %171[0, 0] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %arg15 [%134, %135] : f32 into vector<2x8xf32>
        %174 = vector.extract %171[0, 1] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%134, %138] : f32 into vector<2x8xf32>
        %176 = vector.extract %171[0, 2] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%134, %141] : f32 into vector<2x8xf32>
        %178 = vector.extract %171[0, 3] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%134, %144] : f32 into vector<2x8xf32>
        %180 = vector.extract %121#3[%134, %135] : f32 from vector<2x8xf32>
        %181 = vector.extract %121#3[%134, %138] : f32 from vector<2x8xf32>
        %182 = vector.extract %121#3[%134, %141] : f32 from vector<2x8xf32>
        %183 = vector.extract %121#3[%134, %144] : f32 from vector<2x8xf32>
        %184 = vector.insert %180, %arg16 [%134, %135] : f32 into vector<2x8xf32>
        %185 = vector.insert %181, %184 [%134, %138] : f32 into vector<2x8xf32>
        %186 = vector.insert %182, %185 [%134, %141] : f32 into vector<2x8xf32>
        %187 = vector.insert %183, %186 [%134, %144] : f32 into vector<2x8xf32>
        affine.yield %146, %156, %170, %179, %187 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %131:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %133 = vector.extract %117[%arg11, 0] : f32 from vector<2x1xf32>
        %134 = vector.insert %133, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %135 = vector.extract %112[%arg11, 0] : f32 from vector<2x1xf32>
        %136 = vector.insert %135, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %134, %136 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %132:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %133 = affine.apply #map9(%arg11)
        %134 = affine.apply #map10(%arg11)
        %135 = vector.extract %124#3[%133, %134] : f32 from vector<2x8xf32>
        %136 = affine.apply #map11(%arg11)
        %137 = vector.extract %124#3[%133, %136] : f32 from vector<2x8xf32>
        %138 = affine.apply #map12(%arg11)
        %139 = vector.extract %124#3[%133, %138] : f32 from vector<2x8xf32>
        %140 = affine.apply #map13(%arg11)
        %141 = vector.extract %124#3[%133, %140] : f32 from vector<2x8xf32>
        %142 = vector.insert %135, %arg12 [%133, %134] : f32 into vector<2x8xf32>
        %143 = vector.insert %137, %142 [%133, %136] : f32 into vector<2x8xf32>
        %144 = vector.insert %139, %143 [%133, %138] : f32 into vector<2x8xf32>
        %145 = vector.insert %141, %144 [%133, %140] : f32 into vector<2x8xf32>
        %146 = vector.extract %127#3[%133, %134] : f32 from vector<2x8xf32>
        %147 = vector.extract %127#3[%133, %136] : f32 from vector<2x8xf32>
        %148 = vector.extract %127#3[%133, %138] : f32 from vector<2x8xf32>
        %149 = vector.extract %127#3[%133, %140] : f32 from vector<2x8xf32>
        %150 = vector.insert %146, %arg13 [%133, %134] : f32 into vector<2x8xf32>
        %151 = vector.insert %147, %150 [%133, %136] : f32 into vector<2x8xf32>
        %152 = vector.insert %148, %151 [%133, %138] : f32 into vector<2x8xf32>
        %153 = vector.insert %149, %152 [%133, %140] : f32 into vector<2x8xf32>
        %154 = vector.extract %130#3[%133, %134] : f32 from vector<2x8xf32>
        %155 = vector.extract %130#3[%133, %136] : f32 from vector<2x8xf32>
        %156 = vector.extract %130#3[%133, %138] : f32 from vector<2x8xf32>
        %157 = vector.extract %130#3[%133, %140] : f32 from vector<2x8xf32>
        %158 = vector.insert %154, %arg14 [%133, %134] : f32 into vector<2x8xf32>
        %159 = vector.insert %155, %158 [%133, %136] : f32 into vector<2x8xf32>
        %160 = vector.insert %156, %159 [%133, %138] : f32 into vector<2x8xf32>
        %161 = vector.insert %157, %160 [%133, %140] : f32 into vector<2x8xf32>
        affine.yield %145, %153, %161 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %130#4, %131#0, %131#1, %132#0, %132#1, %132#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %thread_id_x = gpu.thread_id x
    %82 = arith.index_cast %thread_id_x : index to i32
    %83 = arith.andi %82, %c16_i32 : i32
    %84 = arith.cmpi ne, %83, %c0_i32 : i32
    %85 = arith.andi %82, %c32_i32 : i32
    %86 = arith.cmpi ne, %85, %c0_i32 : i32
    %87 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %101 = affine.apply #map10(%arg4)
      %102 = affine.apply #map9(%arg4)
      %103 = vector.extract %81#5[%102, %101] : f32 from vector<2x8xf32>
      %104 = affine.apply #map11(%arg4)
      %105 = vector.extract %81#5[%102, %104] : f32 from vector<2x8xf32>
      %106 = affine.apply #map12(%arg4)
      %107 = vector.extract %81#5[%102, %106] : f32 from vector<2x8xf32>
      %108 = affine.apply #map13(%arg4)
      %109 = vector.extract %81#5[%102, %108] : f32 from vector<2x8xf32>
      %110 = arith.select %84, %103, %105 : f32
      %shuffleResult, %valid = gpu.shuffle xor %110, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %84, %shuffleResult, %103 : f32
      %112 = arith.select %84, %105, %shuffleResult : f32
      %113 = arith.select %84, %107, %109 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %113, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %84, %shuffleResult_15, %107 : f32
      %115 = arith.select %84, %109, %shuffleResult_15 : f32
      %116 = arith.select %86, %111, %114 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %116, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %86, %shuffleResult_17, %111 : f32
      %118 = arith.select %86, %114, %shuffleResult_17 : f32
      %119 = arith.select %86, %112, %115 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %119, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %86, %shuffleResult_19, %112 : f32
      %121 = arith.select %86, %115, %shuffleResult_19 : f32
      %122 = vector.insert %117, %arg5 [%102, %101] : f32 into vector<2x8xf32>
      %123 = vector.insert %120, %122 [%102, %104] : f32 into vector<2x8xf32>
      %124 = vector.insert %118, %123 [%102, %106] : f32 into vector<2x8xf32>
      %125 = vector.insert %121, %124 [%102, %108] : f32 into vector<2x8xf32>
      affine.yield %125 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %88 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %101 = affine.apply #map10(%arg4)
      %102 = affine.apply #map9(%arg4)
      %103 = vector.extract %81#4[%102, %101] : f32 from vector<2x8xf32>
      %104 = affine.apply #map11(%arg4)
      %105 = vector.extract %81#4[%102, %104] : f32 from vector<2x8xf32>
      %106 = affine.apply #map12(%arg4)
      %107 = vector.extract %81#4[%102, %106] : f32 from vector<2x8xf32>
      %108 = affine.apply #map13(%arg4)
      %109 = vector.extract %81#4[%102, %108] : f32 from vector<2x8xf32>
      %110 = arith.select %84, %103, %105 : f32
      %shuffleResult, %valid = gpu.shuffle xor %110, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %84, %shuffleResult, %103 : f32
      %112 = arith.select %84, %105, %shuffleResult : f32
      %113 = arith.select %84, %107, %109 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %113, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %84, %shuffleResult_15, %107 : f32
      %115 = arith.select %84, %109, %shuffleResult_15 : f32
      %116 = arith.select %86, %111, %114 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %116, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %86, %shuffleResult_17, %111 : f32
      %118 = arith.select %86, %114, %shuffleResult_17 : f32
      %119 = arith.select %86, %112, %115 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %119, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %86, %shuffleResult_19, %112 : f32
      %121 = arith.select %86, %115, %shuffleResult_19 : f32
      %122 = vector.insert %117, %arg5 [%102, %101] : f32 into vector<2x8xf32>
      %123 = vector.insert %120, %122 [%102, %104] : f32 into vector<2x8xf32>
      %124 = vector.insert %118, %123 [%102, %106] : f32 into vector<2x8xf32>
      %125 = vector.insert %121, %124 [%102, %108] : f32 into vector<2x8xf32>
      affine.yield %125 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %89 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %101 = affine.apply #map10(%arg4)
      %102 = affine.apply #map9(%arg4)
      %103 = vector.extract %81#3[%102, %101] : f32 from vector<2x8xf32>
      %104 = affine.apply #map11(%arg4)
      %105 = vector.extract %81#3[%102, %104] : f32 from vector<2x8xf32>
      %106 = affine.apply #map12(%arg4)
      %107 = vector.extract %81#3[%102, %106] : f32 from vector<2x8xf32>
      %108 = affine.apply #map13(%arg4)
      %109 = vector.extract %81#3[%102, %108] : f32 from vector<2x8xf32>
      %110 = arith.select %84, %103, %105 : f32
      %shuffleResult, %valid = gpu.shuffle xor %110, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %84, %shuffleResult, %103 : f32
      %112 = arith.select %84, %105, %shuffleResult : f32
      %113 = arith.select %84, %107, %109 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %113, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %84, %shuffleResult_15, %107 : f32
      %115 = arith.select %84, %109, %shuffleResult_15 : f32
      %116 = arith.select %86, %111, %114 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %116, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %86, %shuffleResult_17, %111 : f32
      %118 = arith.select %86, %114, %shuffleResult_17 : f32
      %119 = arith.select %86, %112, %115 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %119, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %86, %shuffleResult_19, %112 : f32
      %121 = arith.select %86, %115, %shuffleResult_19 : f32
      %122 = vector.insert %117, %arg5 [%102, %101] : f32 into vector<2x8xf32>
      %123 = vector.insert %120, %122 [%102, %104] : f32 into vector<2x8xf32>
      %124 = vector.insert %118, %123 [%102, %106] : f32 into vector<2x8xf32>
      %125 = vector.insert %121, %124 [%102, %108] : f32 into vector<2x8xf32>
      affine.yield %125 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %90 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %101 = affine.apply #map10(%arg4)
      %102 = affine.apply #map9(%arg4)
      %103 = vector.extract %81#0[%102, %101] : f32 from vector<2x8xf32>
      %104 = affine.apply #map11(%arg4)
      %105 = vector.extract %81#0[%102, %104] : f32 from vector<2x8xf32>
      %106 = affine.apply #map12(%arg4)
      %107 = vector.extract %81#0[%102, %106] : f32 from vector<2x8xf32>
      %108 = affine.apply #map13(%arg4)
      %109 = vector.extract %81#0[%102, %108] : f32 from vector<2x8xf32>
      %110 = arith.select %84, %103, %105 : f32
      %shuffleResult, %valid = gpu.shuffle xor %110, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %84, %shuffleResult, %103 : f32
      %112 = arith.select %84, %105, %shuffleResult : f32
      %113 = arith.select %84, %107, %109 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %113, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %84, %shuffleResult_15, %107 : f32
      %115 = arith.select %84, %109, %shuffleResult_15 : f32
      %116 = arith.select %86, %111, %114 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %116, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %86, %shuffleResult_17, %111 : f32
      %118 = arith.select %86, %114, %shuffleResult_17 : f32
      %119 = arith.select %86, %112, %115 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %119, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %86, %shuffleResult_19, %112 : f32
      %121 = arith.select %86, %115, %shuffleResult_19 : f32
      %122 = vector.insert %117, %arg5 [%102, %101] : f32 into vector<2x8xf32>
      %123 = vector.insert %120, %122 [%102, %104] : f32 into vector<2x8xf32>
      %124 = vector.insert %118, %123 [%102, %106] : f32 into vector<2x8xf32>
      %125 = vector.insert %121, %124 [%102, %108] : f32 into vector<2x8xf32>
      affine.yield %125 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %91 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %92:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %101 = affine.apply #map9(%arg4)
      %102 = vector.extract %81#1[%101, 0] : f32 from vector<2x1xf32>
      %103 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %143 = vector.broadcast %102 : f32 to vector<1x4xf32>
        affine.yield %143 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %104 = affine.apply #map10(%arg4)
      %105 = vector.extract %103[0, 0] : f32 from vector<1x4xf32>
      %106 = vector.insert %105, %arg5 [%101, %104] : f32 into vector<2x8xf32>
      %107 = affine.apply #map11(%arg4)
      %108 = vector.extract %103[0, 1] : f32 from vector<1x4xf32>
      %109 = vector.insert %108, %106 [%101, %107] : f32 into vector<2x8xf32>
      %110 = affine.apply #map12(%arg4)
      %111 = vector.extract %103[0, 2] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %109 [%101, %110] : f32 into vector<2x8xf32>
      %113 = affine.apply #map13(%arg4)
      %114 = vector.extract %103[0, 3] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %112 [%101, %113] : f32 into vector<2x8xf32>
      %116 = vector.extract %90[%101, %104] : f32 from vector<2x8xf32>
      %117 = vector.extract %90[%101, %107] : f32 from vector<2x8xf32>
      %118 = vector.extract %90[%101, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %90[%101, %113] : f32 from vector<2x8xf32>
      %120 = vector.from_elements %116, %117, %118, %119 : vector<1x4xf32>
      %121 = arith.divf %120, %103 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %122 = vector.extract %121[0, 0] : f32 from vector<1x4xf32>
      %123 = vector.insert %122, %arg6 [%101, %104] : f32 into vector<2x8xf32>
      %124 = vector.extract %121[0, 1] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %123 [%101, %107] : f32 into vector<2x8xf32>
      %126 = vector.extract %121[0, 2] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %125 [%101, %110] : f32 into vector<2x8xf32>
      %128 = vector.extract %121[0, 3] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%101, %113] : f32 into vector<2x8xf32>
      %130 = arith.truncf %121 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %131 = vector.extract %130[0, 0] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %arg7 [%101, %104] : f16 into vector<2x8xf16>
      %133 = vector.extract %130[0, 1] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%101, %107] : f16 into vector<2x8xf16>
      %135 = vector.extract %130[0, 2] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%101, %110] : f16 into vector<2x8xf16>
      %137 = vector.extract %130[0, 3] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%101, %113] : f16 into vector<2x8xf16>
      %139 = vector.insert %131, %arg8 [%101, %104] : f16 into vector<2x8xf16>
      %140 = vector.insert %133, %139 [%101, %107] : f16 into vector<2x8xf16>
      %141 = vector.insert %135, %140 [%101, %110] : f16 into vector<2x8xf16>
      %142 = vector.insert %137, %141 [%101, %113] : f16 into vector<2x8xf16>
      affine.yield %115, %129, %138, %142 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %93 = affine.apply #map36()[%thread_id_x]
    %94 = arith.cmpi ult, %93, %c4 : index
    scf.if %94 {
      affine.for %arg4 = 0 to 4 {
        %101 = affine.apply #map9(%arg4)
        %102 = affine.apply #map10(%arg4)
        %103 = vector.extract %92#3[%101, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply #map11(%arg4)
        %105 = vector.extract %92#3[%101, %104] : f16 from vector<2x8xf16>
        %106 = affine.apply #map12(%arg4)
        %107 = vector.extract %92#3[%101, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply #map13(%arg4)
        %109 = vector.extract %92#3[%101, %108] : f16 from vector<2x8xf16>
        affine.store %103, %91[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %105, %91[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %107, %91[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %109, %91[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %95 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %96:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %101 = affine.apply #map9(%arg4)
      %102 = vector.extract %81#1[%101, 0] : f32 from vector<2x1xf32>
      %103 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %143 = vector.broadcast %102 : f32 to vector<1x4xf32>
        affine.yield %143 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %104 = affine.apply #map10(%arg4)
      %105 = vector.extract %103[0, 0] : f32 from vector<1x4xf32>
      %106 = vector.insert %105, %arg5 [%101, %104] : f32 into vector<2x8xf32>
      %107 = affine.apply #map11(%arg4)
      %108 = vector.extract %103[0, 1] : f32 from vector<1x4xf32>
      %109 = vector.insert %108, %106 [%101, %107] : f32 into vector<2x8xf32>
      %110 = affine.apply #map12(%arg4)
      %111 = vector.extract %103[0, 2] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %109 [%101, %110] : f32 into vector<2x8xf32>
      %113 = affine.apply #map13(%arg4)
      %114 = vector.extract %103[0, 3] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %112 [%101, %113] : f32 into vector<2x8xf32>
      %116 = vector.extract %89[%101, %104] : f32 from vector<2x8xf32>
      %117 = vector.extract %89[%101, %107] : f32 from vector<2x8xf32>
      %118 = vector.extract %89[%101, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %89[%101, %113] : f32 from vector<2x8xf32>
      %120 = vector.from_elements %116, %117, %118, %119 : vector<1x4xf32>
      %121 = arith.divf %120, %103 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %122 = vector.extract %121[0, 0] : f32 from vector<1x4xf32>
      %123 = vector.insert %122, %arg6 [%101, %104] : f32 into vector<2x8xf32>
      %124 = vector.extract %121[0, 1] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %123 [%101, %107] : f32 into vector<2x8xf32>
      %126 = vector.extract %121[0, 2] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %125 [%101, %110] : f32 into vector<2x8xf32>
      %128 = vector.extract %121[0, 3] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%101, %113] : f32 into vector<2x8xf32>
      %130 = arith.truncf %121 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %131 = vector.extract %130[0, 0] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %arg7 [%101, %104] : f16 into vector<2x8xf16>
      %133 = vector.extract %130[0, 1] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%101, %107] : f16 into vector<2x8xf16>
      %135 = vector.extract %130[0, 2] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%101, %110] : f16 into vector<2x8xf16>
      %137 = vector.extract %130[0, 3] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%101, %113] : f16 into vector<2x8xf16>
      %139 = vector.insert %131, %arg8 [%101, %104] : f16 into vector<2x8xf16>
      %140 = vector.insert %133, %139 [%101, %107] : f16 into vector<2x8xf16>
      %141 = vector.insert %135, %140 [%101, %110] : f16 into vector<2x8xf16>
      %142 = vector.insert %137, %141 [%101, %113] : f16 into vector<2x8xf16>
      affine.yield %115, %129, %138, %142 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %94 {
      affine.for %arg4 = 0 to 4 {
        %101 = affine.apply #map9(%arg4)
        %102 = affine.apply #map10(%arg4)
        %103 = vector.extract %96#3[%101, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply #map11(%arg4)
        %105 = vector.extract %96#3[%101, %104] : f16 from vector<2x8xf16>
        %106 = affine.apply #map12(%arg4)
        %107 = vector.extract %96#3[%101, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply #map13(%arg4)
        %109 = vector.extract %96#3[%101, %108] : f16 from vector<2x8xf16>
        affine.store %103, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %105, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %107, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %109, %95[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %97 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %98:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %101 = affine.apply #map9(%arg4)
      %102 = vector.extract %81#1[%101, 0] : f32 from vector<2x1xf32>
      %103 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %143 = vector.broadcast %102 : f32 to vector<1x4xf32>
        affine.yield %143 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %104 = affine.apply #map10(%arg4)
      %105 = vector.extract %103[0, 0] : f32 from vector<1x4xf32>
      %106 = vector.insert %105, %arg5 [%101, %104] : f32 into vector<2x8xf32>
      %107 = affine.apply #map11(%arg4)
      %108 = vector.extract %103[0, 1] : f32 from vector<1x4xf32>
      %109 = vector.insert %108, %106 [%101, %107] : f32 into vector<2x8xf32>
      %110 = affine.apply #map12(%arg4)
      %111 = vector.extract %103[0, 2] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %109 [%101, %110] : f32 into vector<2x8xf32>
      %113 = affine.apply #map13(%arg4)
      %114 = vector.extract %103[0, 3] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %112 [%101, %113] : f32 into vector<2x8xf32>
      %116 = vector.extract %88[%101, %104] : f32 from vector<2x8xf32>
      %117 = vector.extract %88[%101, %107] : f32 from vector<2x8xf32>
      %118 = vector.extract %88[%101, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %88[%101, %113] : f32 from vector<2x8xf32>
      %120 = vector.from_elements %116, %117, %118, %119 : vector<1x4xf32>
      %121 = arith.divf %120, %103 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %122 = vector.extract %121[0, 0] : f32 from vector<1x4xf32>
      %123 = vector.insert %122, %arg6 [%101, %104] : f32 into vector<2x8xf32>
      %124 = vector.extract %121[0, 1] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %123 [%101, %107] : f32 into vector<2x8xf32>
      %126 = vector.extract %121[0, 2] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %125 [%101, %110] : f32 into vector<2x8xf32>
      %128 = vector.extract %121[0, 3] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%101, %113] : f32 into vector<2x8xf32>
      %130 = arith.truncf %121 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %131 = vector.extract %130[0, 0] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %arg7 [%101, %104] : f16 into vector<2x8xf16>
      %133 = vector.extract %130[0, 1] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%101, %107] : f16 into vector<2x8xf16>
      %135 = vector.extract %130[0, 2] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%101, %110] : f16 into vector<2x8xf16>
      %137 = vector.extract %130[0, 3] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%101, %113] : f16 into vector<2x8xf16>
      %139 = vector.insert %131, %arg8 [%101, %104] : f16 into vector<2x8xf16>
      %140 = vector.insert %133, %139 [%101, %107] : f16 into vector<2x8xf16>
      %141 = vector.insert %135, %140 [%101, %110] : f16 into vector<2x8xf16>
      %142 = vector.insert %137, %141 [%101, %113] : f16 into vector<2x8xf16>
      affine.yield %115, %129, %138, %142 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %94 {
      affine.for %arg4 = 0 to 4 {
        %101 = affine.apply #map9(%arg4)
        %102 = affine.apply #map10(%arg4)
        %103 = vector.extract %98#3[%101, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply #map11(%arg4)
        %105 = vector.extract %98#3[%101, %104] : f16 from vector<2x8xf16>
        %106 = affine.apply #map12(%arg4)
        %107 = vector.extract %98#3[%101, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply #map13(%arg4)
        %109 = vector.extract %98#3[%101, %108] : f16 from vector<2x8xf16>
        affine.store %103, %97[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %105, %97[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %107, %97[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %109, %97[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %99 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %100:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %101 = affine.apply #map9(%arg4)
      %102 = vector.extract %81#1[%101, 0] : f32 from vector<2x1xf32>
      %103 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %143 = vector.broadcast %102 : f32 to vector<1x4xf32>
        affine.yield %143 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %104 = affine.apply #map10(%arg4)
      %105 = vector.extract %103[0, 0] : f32 from vector<1x4xf32>
      %106 = vector.insert %105, %arg5 [%101, %104] : f32 into vector<2x8xf32>
      %107 = affine.apply #map11(%arg4)
      %108 = vector.extract %103[0, 1] : f32 from vector<1x4xf32>
      %109 = vector.insert %108, %106 [%101, %107] : f32 into vector<2x8xf32>
      %110 = affine.apply #map12(%arg4)
      %111 = vector.extract %103[0, 2] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %109 [%101, %110] : f32 into vector<2x8xf32>
      %113 = affine.apply #map13(%arg4)
      %114 = vector.extract %103[0, 3] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %112 [%101, %113] : f32 into vector<2x8xf32>
      %116 = vector.extract %87[%101, %104] : f32 from vector<2x8xf32>
      %117 = vector.extract %87[%101, %107] : f32 from vector<2x8xf32>
      %118 = vector.extract %87[%101, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %87[%101, %113] : f32 from vector<2x8xf32>
      %120 = vector.from_elements %116, %117, %118, %119 : vector<1x4xf32>
      %121 = arith.divf %120, %103 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %122 = vector.extract %121[0, 0] : f32 from vector<1x4xf32>
      %123 = vector.insert %122, %arg6 [%101, %104] : f32 into vector<2x8xf32>
      %124 = vector.extract %121[0, 1] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %123 [%101, %107] : f32 into vector<2x8xf32>
      %126 = vector.extract %121[0, 2] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %125 [%101, %110] : f32 into vector<2x8xf32>
      %128 = vector.extract %121[0, 3] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%101, %113] : f32 into vector<2x8xf32>
      %130 = arith.truncf %121 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %131 = vector.extract %130[0, 0] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %arg7 [%101, %104] : f16 into vector<2x8xf16>
      %133 = vector.extract %130[0, 1] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%101, %107] : f16 into vector<2x8xf16>
      %135 = vector.extract %130[0, 2] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%101, %110] : f16 into vector<2x8xf16>
      %137 = vector.extract %130[0, 3] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%101, %113] : f16 into vector<2x8xf16>
      %139 = vector.insert %131, %arg8 [%101, %104] : f16 into vector<2x8xf16>
      %140 = vector.insert %133, %139 [%101, %107] : f16 into vector<2x8xf16>
      %141 = vector.insert %135, %140 [%101, %110] : f16 into vector<2x8xf16>
      %142 = vector.insert %137, %141 [%101, %113] : f16 into vector<2x8xf16>
      affine.yield %115, %129, %138, %142 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %94 {
      affine.for %arg4 = 0 to 4 {
        %101 = affine.apply #map9(%arg4)
        %102 = affine.apply #map10(%arg4)
        %103 = vector.extract %100#3[%101, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply #map11(%arg4)
        %105 = vector.extract %100#3[%101, %104] : f16 from vector<2x8xf16>
        %106 = affine.apply #map12(%arg4)
        %107 = vector.extract %100#3[%101, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply #map13(%arg4)
        %109 = vector.extract %100#3[%101, %108] : f16 from vector<2x8xf16>
        affine.store %103, %99[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<128x32xf16, 1>
        affine.store %105, %99[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<128x32xf16, 1>
        affine.store %107, %99[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<128x32xf16, 1>
        affine.store %109, %99[(%arg4 floordiv 2) * 64 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

