#map = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>
#map1 = affine_map<(d0) -> (d0 floordiv 2)>
#map2 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>
#map3 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>
#map4 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>
#map5 = affine_map<()[s0, s1, s2] -> ((s0 * 32 + s1 + s2 floordiv 32) mod 32)>
#map6 = affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>
#map7 = affine_map<()[s0] -> (s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>
#map8 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
#map9 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 1)>
#map10 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 2)>
#map11 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 3)>
#map12 = affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>
#map13 = affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map14 = affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map15 = affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map16 = affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map17 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>
#map18 = affine_map<(d0) -> (d0 floordiv 8)>
#map19 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>
#map20 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>
#map21 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>
#map22 = affine_map<()[s0] -> (s0 mod 32)>
#map23 = affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>
#map24 = affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map25 = affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map26 = affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map27 = affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map28 = affine_map<(d0) -> (d0 * 4)>
#map29 = affine_map<(d0) -> (d0 * 4 + 1)>
#map30 = affine_map<(d0) -> (d0 * 4 + 2)>
#map31 = affine_map<(d0) -> (d0 * 4 + 3)>
#map32 = affine_map<(d0)[s0, s1] -> (((d0 + s0 * 16777216 + s1 * 524288 + 131072) floordiv 524288) mod 32)>
#map33 = affine_map<(d0)[s0, s1] -> (((d0 + s0 * 16777216 + s1 * 524288 + 262144) floordiv 524288) mod 32)>
#map34 = affine_map<(d0)[s0, s1] -> (((d0 + s0 * 16777216 + s1 * 524288 + 393216) floordiv 524288) mod 32)>
#map35 = affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>
#map36 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
#map37 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>
#map38 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>
#map39 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>
#map40 = affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>
#map41 = affine_map<(d0) -> (d0 mod 2)>
#map42 = affine_map<()[s0] -> (s0 floordiv 64)>
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
    %cst_6 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_7 = arith.constant 0xFF800000 : f32
    %cst_8 = arith.constant 0.127531052 : f32
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
    %8 = arith.truncf %cst_8 : f32 to f16
    %9:3 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6, %arg6 = %cst_6, %arg7 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %92 = vector.insert %cst_2, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %93 = vector.insert %cst_7, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %94 = vector.insert %cst_7, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %92, %93, %94 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %92 = affine.apply #map(%arg4)
      %93 = affine.apply #map1(%arg4)
      %94 = vector.insert %cst_2, %arg5 [%93, %92] : f32 into vector<2x8xf32>
      %95 = affine.apply #map2(%arg4)
      %96 = vector.insert %cst_2, %94 [%93, %95] : f32 into vector<2x8xf32>
      %97 = affine.apply #map3(%arg4)
      %98 = vector.insert %cst_2, %96 [%93, %97] : f32 into vector<2x8xf32>
      %99 = affine.apply #map4(%arg4)
      %100 = vector.insert %cst_2, %98 [%93, %99] : f32 into vector<2x8xf32>
      affine.yield %100 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6) -> (vector<2x1xf32>) {
      %92 = vector.insert %cst_2, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %92 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %12 = affine.apply #map5()[%block_id_z, %7, %6]
    %13 = affine.apply #map6()[%6]
    %subview = memref.subview %arg0[0, %12, %13, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
    %thread_id_x = gpu.thread_id x
    %14 = affine.apply #map7()[%thread_id_x]
    %15 = affine.apply #map8()[%thread_id_x]
    %16 = memref.load %alloc[%14, %15] : memref<128x32xf16, 3>
    %17 = affine.apply #map9()[%thread_id_x]
    %18 = memref.load %alloc[%14, %17] : memref<128x32xf16, 3>
    %19 = affine.apply #map10()[%thread_id_x]
    %20 = memref.load %alloc[%14, %19] : memref<128x32xf16, 3>
    %21 = affine.apply #map11()[%thread_id_x]
    %22 = memref.load %alloc[%14, %21] : memref<128x32xf16, 3>
    %23 = vector.from_elements %16, %18, %20, %22 : vector<1x4xf16>
    %24:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %23) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %92 = affine.apply #map12(%arg4)[%thread_id_x]
      %93 = affine.apply #map13(%arg4)[%thread_id_x]
      %94 = memref.load %alloc[%92, %93] : memref<128x32xf16, 3>
      %95 = affine.apply #map14(%arg4)[%thread_id_x]
      %96 = memref.load %alloc[%92, %95] : memref<128x32xf16, 3>
      %97 = affine.apply #map15(%arg4)[%thread_id_x]
      %98 = memref.load %alloc[%92, %97] : memref<128x32xf16, 3>
      %99 = affine.apply #map16(%arg4)[%thread_id_x]
      %100 = memref.load %alloc[%92, %99] : memref<128x32xf16, 3>
      %101 = vector.from_elements %94, %96, %98, %100 : vector<1x4xf16>
      %102 = affine.apply #map1(%arg4)
      %103 = affine.apply #map(%arg4)
      %104 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %105 = vector.insert %104, %arg5 [%102, %103] : f16 into vector<2x8xf16>
      %106 = affine.apply #map2(%arg4)
      %107 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %108 = vector.insert %107, %105 [%102, %106] : f16 into vector<2x8xf16>
      %109 = affine.apply #map3(%arg4)
      %110 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %111 = vector.insert %110, %108 [%102, %109] : f16 into vector<2x8xf16>
      %112 = affine.apply #map4(%arg4)
      %113 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %111 [%102, %112] : f16 into vector<2x8xf16>
      %115 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %116 = arith.mulf %arg7, %115 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %117 = vector.extract %116[0, 0] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %arg6 [%102, %103] : f16 into vector<2x8xf16>
      %119 = vector.extract %116[0, 1] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %118 [%102, %106] : f16 into vector<2x8xf16>
      %121 = vector.extract %116[0, 2] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %120 [%102, %109] : f16 into vector<2x8xf16>
      %123 = vector.extract %116[0, 3] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%102, %112] : f16 into vector<2x8xf16>
      affine.yield %114, %124, %101 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %25 = vector.broadcast %8 : f16 to vector<1x4xf16>
    %26 = arith.mulf %24#2, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %27 = vector.extract %26[0, 0] : f16 from vector<1x4xf16>
    %28 = vector.insert %27, %24#1 [1, 4] : f16 into vector<2x8xf16>
    %29 = vector.extract %26[0, 1] : f16 from vector<1x4xf16>
    %30 = vector.insert %29, %28 [1, 5] : f16 into vector<2x8xf16>
    %31 = vector.extract %26[0, 2] : f16 from vector<1x4xf16>
    %32 = vector.insert %31, %30 [1, 6] : f16 into vector<2x8xf16>
    %33 = vector.extract %26[0, 3] : f16 from vector<1x4xf16>
    %34 = vector.insert %33, %32 [1, 7] : f16 into vector<2x8xf16>
    %alloc_9 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_10 = memref.subview %arg0[0, %12, %13, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview_10 to %alloc_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_9 : memref<128x32xf16, 3>
    %35 = memref.load %alloc_9[%14, %15] : memref<128x32xf16, 3>
    %36 = memref.load %alloc_9[%14, %17] : memref<128x32xf16, 3>
    %37 = memref.load %alloc_9[%14, %19] : memref<128x32xf16, 3>
    %38 = memref.load %alloc_9[%14, %21] : memref<128x32xf16, 3>
    %39 = vector.from_elements %35, %36, %37, %38 : vector<1x4xf16>
    %40:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %39) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %92 = affine.apply #map12(%arg4)[%thread_id_x]
      %93 = affine.apply #map13(%arg4)[%thread_id_x]
      %94 = memref.load %alloc_9[%92, %93] : memref<128x32xf16, 3>
      %95 = affine.apply #map14(%arg4)[%thread_id_x]
      %96 = memref.load %alloc_9[%92, %95] : memref<128x32xf16, 3>
      %97 = affine.apply #map15(%arg4)[%thread_id_x]
      %98 = memref.load %alloc_9[%92, %97] : memref<128x32xf16, 3>
      %99 = affine.apply #map16(%arg4)[%thread_id_x]
      %100 = memref.load %alloc_9[%92, %99] : memref<128x32xf16, 3>
      %101 = vector.from_elements %94, %96, %98, %100 : vector<1x4xf16>
      %102 = affine.apply #map1(%arg4)
      %103 = affine.apply #map(%arg4)
      %104 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %105 = vector.insert %104, %arg5 [%102, %103] : f16 into vector<2x8xf16>
      %106 = affine.apply #map2(%arg4)
      %107 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %108 = vector.insert %107, %105 [%102, %106] : f16 into vector<2x8xf16>
      %109 = affine.apply #map3(%arg4)
      %110 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %111 = vector.insert %110, %108 [%102, %109] : f16 into vector<2x8xf16>
      %112 = affine.apply #map4(%arg4)
      %113 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %111 [%102, %112] : f16 into vector<2x8xf16>
      %115 = arith.mulf %arg7, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %116 = vector.extract %115[0, 0] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %arg6 [%102, %103] : f16 into vector<2x8xf16>
      %118 = vector.extract %115[0, 1] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %117 [%102, %106] : f16 into vector<2x8xf16>
      %120 = vector.extract %115[0, 2] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %119 [%102, %109] : f16 into vector<2x8xf16>
      %122 = vector.extract %115[0, 3] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %121 [%102, %112] : f16 into vector<2x8xf16>
      affine.yield %114, %123, %101 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %41 = arith.mulf %40#2, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %42 = vector.extract %41[0, 0] : f16 from vector<1x4xf16>
    %43 = vector.insert %42, %40#1 [1, 4] : f16 into vector<2x8xf16>
    %44 = vector.extract %41[0, 1] : f16 from vector<1x4xf16>
    %45 = vector.insert %44, %43 [1, 5] : f16 into vector<2x8xf16>
    %46 = vector.extract %41[0, 2] : f16 from vector<1x4xf16>
    %47 = vector.insert %46, %45 [1, 6] : f16 into vector<2x8xf16>
    %48 = vector.extract %41[0, 3] : f16 from vector<1x4xf16>
    %49 = vector.insert %48, %47 [1, 7] : f16 into vector<2x8xf16>
    %alloc_11 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_12 = memref.subview %arg0[0, %12, %13, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview_12 to %alloc_11 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_11 : memref<128x32xf16, 3>
    %50 = memref.load %alloc_11[%14, %15] : memref<128x32xf16, 3>
    %51 = memref.load %alloc_11[%14, %17] : memref<128x32xf16, 3>
    %52 = memref.load %alloc_11[%14, %19] : memref<128x32xf16, 3>
    %53 = memref.load %alloc_11[%14, %21] : memref<128x32xf16, 3>
    %54 = vector.from_elements %50, %51, %52, %53 : vector<1x4xf16>
    %55:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %54) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %92 = affine.apply #map12(%arg4)[%thread_id_x]
      %93 = affine.apply #map13(%arg4)[%thread_id_x]
      %94 = memref.load %alloc_11[%92, %93] : memref<128x32xf16, 3>
      %95 = affine.apply #map14(%arg4)[%thread_id_x]
      %96 = memref.load %alloc_11[%92, %95] : memref<128x32xf16, 3>
      %97 = affine.apply #map15(%arg4)[%thread_id_x]
      %98 = memref.load %alloc_11[%92, %97] : memref<128x32xf16, 3>
      %99 = affine.apply #map16(%arg4)[%thread_id_x]
      %100 = memref.load %alloc_11[%92, %99] : memref<128x32xf16, 3>
      %101 = vector.from_elements %94, %96, %98, %100 : vector<1x4xf16>
      %102 = affine.apply #map1(%arg4)
      %103 = affine.apply #map(%arg4)
      %104 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %105 = vector.insert %104, %arg5 [%102, %103] : f16 into vector<2x8xf16>
      %106 = affine.apply #map2(%arg4)
      %107 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %108 = vector.insert %107, %105 [%102, %106] : f16 into vector<2x8xf16>
      %109 = affine.apply #map3(%arg4)
      %110 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %111 = vector.insert %110, %108 [%102, %109] : f16 into vector<2x8xf16>
      %112 = affine.apply #map4(%arg4)
      %113 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %111 [%102, %112] : f16 into vector<2x8xf16>
      %115 = arith.mulf %arg7, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %116 = vector.extract %115[0, 0] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %arg6 [%102, %103] : f16 into vector<2x8xf16>
      %118 = vector.extract %115[0, 1] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %117 [%102, %106] : f16 into vector<2x8xf16>
      %120 = vector.extract %115[0, 2] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %119 [%102, %109] : f16 into vector<2x8xf16>
      %122 = vector.extract %115[0, 3] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %121 [%102, %112] : f16 into vector<2x8xf16>
      affine.yield %114, %123, %101 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %56 = arith.mulf %55#2, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %57 = vector.extract %56[0, 0] : f16 from vector<1x4xf16>
    %58 = vector.insert %57, %55#1 [1, 4] : f16 into vector<2x8xf16>
    %59 = vector.extract %56[0, 1] : f16 from vector<1x4xf16>
    %60 = vector.insert %59, %58 [1, 5] : f16 into vector<2x8xf16>
    %61 = vector.extract %56[0, 2] : f16 from vector<1x4xf16>
    %62 = vector.insert %61, %60 [1, 6] : f16 into vector<2x8xf16>
    %63 = vector.extract %56[0, 3] : f16 from vector<1x4xf16>
    %64 = vector.insert %63, %62 [1, 7] : f16 into vector<2x8xf16>
    %alloc_13 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_14 = memref.subview %arg0[0, %12, %13, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview_14 to %alloc_13 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_13 : memref<128x32xf16, 3>
    %65 = memref.load %alloc_13[%14, %15] : memref<128x32xf16, 3>
    %66 = memref.load %alloc_13[%14, %17] : memref<128x32xf16, 3>
    %67 = memref.load %alloc_13[%14, %19] : memref<128x32xf16, 3>
    %68 = memref.load %alloc_13[%14, %21] : memref<128x32xf16, 3>
    %69 = vector.from_elements %65, %66, %67, %68 : vector<1x4xf16>
    %70:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %69) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %92 = affine.apply #map12(%arg4)[%thread_id_x]
      %93 = affine.apply #map13(%arg4)[%thread_id_x]
      %94 = memref.load %alloc_13[%92, %93] : memref<128x32xf16, 3>
      %95 = affine.apply #map14(%arg4)[%thread_id_x]
      %96 = memref.load %alloc_13[%92, %95] : memref<128x32xf16, 3>
      %97 = affine.apply #map15(%arg4)[%thread_id_x]
      %98 = memref.load %alloc_13[%92, %97] : memref<128x32xf16, 3>
      %99 = affine.apply #map16(%arg4)[%thread_id_x]
      %100 = memref.load %alloc_13[%92, %99] : memref<128x32xf16, 3>
      %101 = vector.from_elements %94, %96, %98, %100 : vector<1x4xf16>
      %102 = affine.apply #map1(%arg4)
      %103 = affine.apply #map(%arg4)
      %104 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %105 = vector.insert %104, %arg5 [%102, %103] : f16 into vector<2x8xf16>
      %106 = affine.apply #map2(%arg4)
      %107 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %108 = vector.insert %107, %105 [%102, %106] : f16 into vector<2x8xf16>
      %109 = affine.apply #map3(%arg4)
      %110 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %111 = vector.insert %110, %108 [%102, %109] : f16 into vector<2x8xf16>
      %112 = affine.apply #map4(%arg4)
      %113 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %111 [%102, %112] : f16 into vector<2x8xf16>
      %115 = arith.mulf %arg7, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %116 = vector.extract %115[0, 0] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %arg6 [%102, %103] : f16 into vector<2x8xf16>
      %118 = vector.extract %115[0, 1] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %117 [%102, %106] : f16 into vector<2x8xf16>
      %120 = vector.extract %115[0, 2] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %119 [%102, %109] : f16 into vector<2x8xf16>
      %122 = vector.extract %115[0, 3] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %121 [%102, %112] : f16 into vector<2x8xf16>
      affine.yield %114, %123, %101 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %71 = arith.mulf %70#2, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %72 = vector.extract %71[0, 0] : f16 from vector<1x4xf16>
    %73 = vector.insert %72, %70#1 [1, 4] : f16 into vector<2x8xf16>
    %74 = vector.extract %71[0, 1] : f16 from vector<1x4xf16>
    %75 = vector.insert %74, %73 [1, 5] : f16 into vector<2x8xf16>
    %76 = vector.extract %71[0, 2] : f16 from vector<1x4xf16>
    %77 = vector.insert %76, %75 [1, 6] : f16 into vector<2x8xf16>
    %78 = vector.extract %71[0, 3] : f16 from vector<1x4xf16>
    %79 = vector.insert %78, %77 [1, 7] : f16 into vector<2x8xf16>
    %80:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#2, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %92 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %118 = affine.apply #map17(%arg11)
        %119 = affine.apply #map18(%arg11)
        %120 = vector.insert %cst_2, %arg12 [%119, %118] : f32 into vector<2x32xf32>
        %121 = affine.apply #map19(%arg11)
        %122 = vector.insert %cst_2, %120 [%119, %121] : f32 into vector<2x32xf32>
        %123 = affine.apply #map20(%arg11)
        %124 = vector.insert %cst_2, %122 [%119, %123] : f32 into vector<2x32xf32>
        %125 = affine.apply #map21(%arg11)
        %126 = vector.insert %cst_2, %124 [%119, %125] : f32 into vector<2x32xf32>
        affine.yield %126 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %93 = affine.apply #map22()[%7]
      %subview_16 = memref.subview %transpose[0, %93, 0, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_16 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<32x128xf16, 3>
      %94:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %118 = affine.apply #map23(%arg11)[%thread_id_x]
        %119 = memref.load %alloc_15[%15, %118] : memref<32x128xf16, 3>
        %120 = memref.load %alloc_15[%17, %118] : memref<32x128xf16, 3>
        %121 = memref.load %alloc_15[%19, %118] : memref<32x128xf16, 3>
        %122 = memref.load %alloc_15[%21, %118] : memref<32x128xf16, 3>
        %123 = vector.from_elements %119, %120, %121, %122 : vector<4x1xf16>
        %124:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %123) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %154 = affine.apply #map24(%arg14)[%thread_id_x]
          %155 = memref.load %alloc_15[%154, %118] : memref<32x128xf16, 3>
          %156 = affine.apply #map25(%arg14)[%thread_id_x]
          %157 = memref.load %alloc_15[%156, %118] : memref<32x128xf16, 3>
          %158 = affine.apply #map26(%arg14)[%thread_id_x]
          %159 = memref.load %alloc_15[%158, %118] : memref<32x128xf16, 3>
          %160 = affine.apply #map27(%arg14)[%thread_id_x]
          %161 = memref.load %alloc_15[%160, %118] : memref<32x128xf16, 3>
          %162 = vector.from_elements %155, %157, %159, %161 : vector<4x1xf16>
          %163 = affine.apply #map18(%arg11)
          %164 = affine.apply #map28(%arg14)
          %165 = vector.extract %34[%163, %164] : f16 from vector<2x8xf16>
          %166 = affine.apply #map29(%arg14)
          %167 = vector.extract %34[%163, %166] : f16 from vector<2x8xf16>
          %168 = affine.apply #map30(%arg14)
          %169 = vector.extract %34[%163, %168] : f16 from vector<2x8xf16>
          %170 = affine.apply #map31(%arg14)
          %171 = vector.extract %34[%163, %170] : f16 from vector<2x8xf16>
          %172 = vector.from_elements %165, %167, %169, %171 : vector<1x4xf16>
          %173 = frisk.warp_mma_rr %172, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %173, %162 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %125 = affine.apply #map18(%arg11)
        %126 = vector.extract %34[%125, 4] : f16 from vector<2x8xf16>
        %127 = vector.extract %34[%125, 5] : f16 from vector<2x8xf16>
        %128 = vector.extract %34[%125, 6] : f16 from vector<2x8xf16>
        %129 = vector.extract %34[%125, 7] : f16 from vector<2x8xf16>
        %130 = vector.from_elements %126, %127, %128, %129 : vector<1x4xf16>
        %131 = frisk.warp_mma_rr %130, %124#1, %124#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %132 = affine.apply #map17(%arg11)
        %133 = vector.extract %92[%125, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply #map19(%arg11)
        %135 = vector.extract %92[%125, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply #map20(%arg11)
        %137 = vector.extract %92[%125, %136] : f32 from vector<2x32xf32>
        %138 = affine.apply #map21(%arg11)
        %139 = vector.extract %92[%125, %138] : f32 from vector<2x32xf32>
        %140 = vector.from_elements %133, %135, %137, %139 : vector<1x4xf32>
        %141 = arith.addf %140, %131 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %142 = vector.extract %141[0, 0] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %arg12 [%125, %132] : f32 into vector<2x32xf32>
        %144 = vector.extract %141[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%125, %134] : f32 into vector<2x32xf32>
        %146 = vector.extract %141[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%125, %136] : f32 into vector<2x32xf32>
        %148 = vector.extract %141[0, 3] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%125, %138] : f32 into vector<2x32xf32>
        %150 = vector.insert %142, %arg13 [%125, %132] : f32 into vector<2x32xf32>
        %151 = vector.insert %144, %150 [%125, %134] : f32 into vector<2x32xf32>
        %152 = vector.insert %146, %151 [%125, %136] : f32 into vector<2x32xf32>
        %153 = vector.insert %148, %152 [%125, %138] : f32 into vector<2x32xf32>
        affine.yield %149, %153 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %95 = affine.apply #map32(%arg4)[%block_id_z, %7]
      %subview_18 = memref.subview %transpose[0, %95, 32, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_18 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_17 : memref<32x128xf16, 3>
      %96:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %118 = affine.apply #map23(%arg11)[%thread_id_x]
        %119 = memref.load %alloc_17[%15, %118] : memref<32x128xf16, 3>
        %120 = memref.load %alloc_17[%17, %118] : memref<32x128xf16, 3>
        %121 = memref.load %alloc_17[%19, %118] : memref<32x128xf16, 3>
        %122 = memref.load %alloc_17[%21, %118] : memref<32x128xf16, 3>
        %123 = vector.from_elements %119, %120, %121, %122 : vector<4x1xf16>
        %124:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %123) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %154 = affine.apply #map24(%arg14)[%thread_id_x]
          %155 = memref.load %alloc_17[%154, %118] : memref<32x128xf16, 3>
          %156 = affine.apply #map25(%arg14)[%thread_id_x]
          %157 = memref.load %alloc_17[%156, %118] : memref<32x128xf16, 3>
          %158 = affine.apply #map26(%arg14)[%thread_id_x]
          %159 = memref.load %alloc_17[%158, %118] : memref<32x128xf16, 3>
          %160 = affine.apply #map27(%arg14)[%thread_id_x]
          %161 = memref.load %alloc_17[%160, %118] : memref<32x128xf16, 3>
          %162 = vector.from_elements %155, %157, %159, %161 : vector<4x1xf16>
          %163 = affine.apply #map18(%arg11)
          %164 = affine.apply #map28(%arg14)
          %165 = vector.extract %49[%163, %164] : f16 from vector<2x8xf16>
          %166 = affine.apply #map29(%arg14)
          %167 = vector.extract %49[%163, %166] : f16 from vector<2x8xf16>
          %168 = affine.apply #map30(%arg14)
          %169 = vector.extract %49[%163, %168] : f16 from vector<2x8xf16>
          %170 = affine.apply #map31(%arg14)
          %171 = vector.extract %49[%163, %170] : f16 from vector<2x8xf16>
          %172 = vector.from_elements %165, %167, %169, %171 : vector<1x4xf16>
          %173 = frisk.warp_mma_rr %172, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %173, %162 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %125 = affine.apply #map18(%arg11)
        %126 = vector.extract %49[%125, 4] : f16 from vector<2x8xf16>
        %127 = vector.extract %49[%125, 5] : f16 from vector<2x8xf16>
        %128 = vector.extract %49[%125, 6] : f16 from vector<2x8xf16>
        %129 = vector.extract %49[%125, 7] : f16 from vector<2x8xf16>
        %130 = vector.from_elements %126, %127, %128, %129 : vector<1x4xf16>
        %131 = frisk.warp_mma_rr %130, %124#1, %124#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %132 = affine.apply #map17(%arg11)
        %133 = vector.extract %94#1[%125, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply #map19(%arg11)
        %135 = vector.extract %94#1[%125, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply #map20(%arg11)
        %137 = vector.extract %94#1[%125, %136] : f32 from vector<2x32xf32>
        %138 = affine.apply #map21(%arg11)
        %139 = vector.extract %94#1[%125, %138] : f32 from vector<2x32xf32>
        %140 = vector.from_elements %133, %135, %137, %139 : vector<1x4xf32>
        %141 = arith.addf %140, %131 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %142 = vector.extract %141[0, 0] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %arg12 [%125, %132] : f32 into vector<2x32xf32>
        %144 = vector.extract %141[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%125, %134] : f32 into vector<2x32xf32>
        %146 = vector.extract %141[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%125, %136] : f32 into vector<2x32xf32>
        %148 = vector.extract %141[0, 3] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%125, %138] : f32 into vector<2x32xf32>
        %150 = vector.insert %142, %arg13 [%125, %132] : f32 into vector<2x32xf32>
        %151 = vector.insert %144, %150 [%125, %134] : f32 into vector<2x32xf32>
        %152 = vector.insert %146, %151 [%125, %136] : f32 into vector<2x32xf32>
        %153 = vector.insert %148, %152 [%125, %138] : f32 into vector<2x32xf32>
        affine.yield %149, %153 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %97 = affine.apply #map33(%arg4)[%block_id_z, %7]
      %subview_20 = memref.subview %transpose[0, %97, 64, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_20 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_19 : memref<32x128xf16, 3>
      %98:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %118 = affine.apply #map23(%arg11)[%thread_id_x]
        %119 = memref.load %alloc_19[%15, %118] : memref<32x128xf16, 3>
        %120 = memref.load %alloc_19[%17, %118] : memref<32x128xf16, 3>
        %121 = memref.load %alloc_19[%19, %118] : memref<32x128xf16, 3>
        %122 = memref.load %alloc_19[%21, %118] : memref<32x128xf16, 3>
        %123 = vector.from_elements %119, %120, %121, %122 : vector<4x1xf16>
        %124:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %123) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %154 = affine.apply #map24(%arg14)[%thread_id_x]
          %155 = memref.load %alloc_19[%154, %118] : memref<32x128xf16, 3>
          %156 = affine.apply #map25(%arg14)[%thread_id_x]
          %157 = memref.load %alloc_19[%156, %118] : memref<32x128xf16, 3>
          %158 = affine.apply #map26(%arg14)[%thread_id_x]
          %159 = memref.load %alloc_19[%158, %118] : memref<32x128xf16, 3>
          %160 = affine.apply #map27(%arg14)[%thread_id_x]
          %161 = memref.load %alloc_19[%160, %118] : memref<32x128xf16, 3>
          %162 = vector.from_elements %155, %157, %159, %161 : vector<4x1xf16>
          %163 = affine.apply #map18(%arg11)
          %164 = affine.apply #map28(%arg14)
          %165 = vector.extract %64[%163, %164] : f16 from vector<2x8xf16>
          %166 = affine.apply #map29(%arg14)
          %167 = vector.extract %64[%163, %166] : f16 from vector<2x8xf16>
          %168 = affine.apply #map30(%arg14)
          %169 = vector.extract %64[%163, %168] : f16 from vector<2x8xf16>
          %170 = affine.apply #map31(%arg14)
          %171 = vector.extract %64[%163, %170] : f16 from vector<2x8xf16>
          %172 = vector.from_elements %165, %167, %169, %171 : vector<1x4xf16>
          %173 = frisk.warp_mma_rr %172, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %173, %162 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %125 = affine.apply #map18(%arg11)
        %126 = vector.extract %64[%125, 4] : f16 from vector<2x8xf16>
        %127 = vector.extract %64[%125, 5] : f16 from vector<2x8xf16>
        %128 = vector.extract %64[%125, 6] : f16 from vector<2x8xf16>
        %129 = vector.extract %64[%125, 7] : f16 from vector<2x8xf16>
        %130 = vector.from_elements %126, %127, %128, %129 : vector<1x4xf16>
        %131 = frisk.warp_mma_rr %130, %124#1, %124#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %132 = affine.apply #map17(%arg11)
        %133 = vector.extract %96#1[%125, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply #map19(%arg11)
        %135 = vector.extract %96#1[%125, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply #map20(%arg11)
        %137 = vector.extract %96#1[%125, %136] : f32 from vector<2x32xf32>
        %138 = affine.apply #map21(%arg11)
        %139 = vector.extract %96#1[%125, %138] : f32 from vector<2x32xf32>
        %140 = vector.from_elements %133, %135, %137, %139 : vector<1x4xf32>
        %141 = arith.addf %140, %131 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %142 = vector.extract %141[0, 0] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %arg12 [%125, %132] : f32 into vector<2x32xf32>
        %144 = vector.extract %141[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%125, %134] : f32 into vector<2x32xf32>
        %146 = vector.extract %141[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%125, %136] : f32 into vector<2x32xf32>
        %148 = vector.extract %141[0, 3] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%125, %138] : f32 into vector<2x32xf32>
        %150 = vector.insert %142, %arg13 [%125, %132] : f32 into vector<2x32xf32>
        %151 = vector.insert %144, %150 [%125, %134] : f32 into vector<2x32xf32>
        %152 = vector.insert %146, %151 [%125, %136] : f32 into vector<2x32xf32>
        %153 = vector.insert %148, %152 [%125, %138] : f32 into vector<2x32xf32>
        affine.yield %149, %153 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %99 = affine.apply #map34(%arg4)[%block_id_z, %7]
      %subview_22 = memref.subview %transpose[0, %99, 96, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_22 to %alloc_21 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_21 : memref<32x128xf16, 3>
      %100:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %118 = affine.apply #map23(%arg11)[%thread_id_x]
        %119 = memref.load %alloc_21[%15, %118] : memref<32x128xf16, 3>
        %120 = memref.load %alloc_21[%17, %118] : memref<32x128xf16, 3>
        %121 = memref.load %alloc_21[%19, %118] : memref<32x128xf16, 3>
        %122 = memref.load %alloc_21[%21, %118] : memref<32x128xf16, 3>
        %123 = vector.from_elements %119, %120, %121, %122 : vector<4x1xf16>
        %124:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %123) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %154 = affine.apply #map24(%arg14)[%thread_id_x]
          %155 = memref.load %alloc_21[%154, %118] : memref<32x128xf16, 3>
          %156 = affine.apply #map25(%arg14)[%thread_id_x]
          %157 = memref.load %alloc_21[%156, %118] : memref<32x128xf16, 3>
          %158 = affine.apply #map26(%arg14)[%thread_id_x]
          %159 = memref.load %alloc_21[%158, %118] : memref<32x128xf16, 3>
          %160 = affine.apply #map27(%arg14)[%thread_id_x]
          %161 = memref.load %alloc_21[%160, %118] : memref<32x128xf16, 3>
          %162 = vector.from_elements %155, %157, %159, %161 : vector<4x1xf16>
          %163 = affine.apply #map18(%arg11)
          %164 = affine.apply #map28(%arg14)
          %165 = vector.extract %79[%163, %164] : f16 from vector<2x8xf16>
          %166 = affine.apply #map29(%arg14)
          %167 = vector.extract %79[%163, %166] : f16 from vector<2x8xf16>
          %168 = affine.apply #map30(%arg14)
          %169 = vector.extract %79[%163, %168] : f16 from vector<2x8xf16>
          %170 = affine.apply #map31(%arg14)
          %171 = vector.extract %79[%163, %170] : f16 from vector<2x8xf16>
          %172 = vector.from_elements %165, %167, %169, %171 : vector<1x4xf16>
          %173 = frisk.warp_mma_rr %172, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %173, %162 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %125 = affine.apply #map18(%arg11)
        %126 = vector.extract %79[%125, 4] : f16 from vector<2x8xf16>
        %127 = vector.extract %79[%125, 5] : f16 from vector<2x8xf16>
        %128 = vector.extract %79[%125, 6] : f16 from vector<2x8xf16>
        %129 = vector.extract %79[%125, 7] : f16 from vector<2x8xf16>
        %130 = vector.from_elements %126, %127, %128, %129 : vector<1x4xf16>
        %131 = frisk.warp_mma_rr %130, %124#1, %124#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %132 = affine.apply #map17(%arg11)
        %133 = vector.extract %98#1[%125, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply #map19(%arg11)
        %135 = vector.extract %98#1[%125, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply #map20(%arg11)
        %137 = vector.extract %98#1[%125, %136] : f32 from vector<2x32xf32>
        %138 = affine.apply #map21(%arg11)
        %139 = vector.extract %98#1[%125, %138] : f32 from vector<2x32xf32>
        %140 = vector.from_elements %133, %135, %137, %139 : vector<1x4xf32>
        %141 = arith.addf %140, %131 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %142 = vector.extract %141[0, 0] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %arg12 [%125, %132] : f32 into vector<2x32xf32>
        %144 = vector.extract %141[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%125, %134] : f32 into vector<2x32xf32>
        %146 = vector.extract %141[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%125, %136] : f32 into vector<2x32xf32>
        %148 = vector.extract %141[0, 3] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%125, %138] : f32 into vector<2x32xf32>
        %150 = vector.insert %142, %arg13 [%125, %132] : f32 into vector<2x32xf32>
        %151 = vector.insert %144, %150 [%125, %134] : f32 into vector<2x32xf32>
        %152 = vector.insert %146, %151 [%125, %136] : f32 into vector<2x32xf32>
        %153 = vector.insert %148, %152 [%125, %138] : f32 into vector<2x32xf32>
        affine.yield %149, %153 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %101 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %118 = affine.apply #map18(%arg11)
        %119 = affine.apply #map17(%arg11)
        %120 = vector.extract %100#1[%118, %119] : f32 from vector<2x32xf32>
        %121 = affine.apply #map19(%arg11)
        %122 = vector.extract %100#1[%118, %121] : f32 from vector<2x32xf32>
        %123 = affine.apply #map20(%arg11)
        %124 = vector.extract %100#1[%118, %123] : f32 from vector<2x32xf32>
        %125 = affine.apply #map21(%arg11)
        %126 = vector.extract %100#1[%118, %125] : f32 from vector<2x32xf32>
        %127 = vector.insert %120, %arg12 [%118, %119] : f32 into vector<2x32xf32>
        %128 = vector.insert %122, %127 [%118, %121] : f32 into vector<2x32xf32>
        %129 = vector.insert %124, %128 [%118, %123] : f32 into vector<2x32xf32>
        %130 = vector.insert %126, %129 [%118, %125] : f32 into vector<2x32xf32>
        affine.yield %130 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %102:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %118 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_7) -> (f32) {
          %128 = affine.apply #map28(%arg14)
          %129 = vector.extract %101[%arg11, %128] : f32 from vector<2x32xf32>
          %130 = arith.maxnumf %arg15, %129 : f32
          %131 = affine.apply #map29(%arg14)
          %132 = vector.extract %101[%arg11, %131] : f32 from vector<2x32xf32>
          %133 = arith.maxnumf %130, %132 : f32
          %134 = affine.apply #map30(%arg14)
          %135 = vector.extract %101[%arg11, %134] : f32 from vector<2x32xf32>
          %136 = arith.maxnumf %133, %135 : f32
          %137 = affine.apply #map31(%arg14)
          %138 = vector.extract %101[%arg11, %137] : f32 from vector<2x32xf32>
          %139 = arith.maxnumf %136, %138 : f32
          affine.yield %139 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %118, %c16_i32, %c64_i32 : f32
        %119 = arith.maxnumf %118, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %119, %c32_i32, %c64_i32 : f32
        %120 = arith.maxnumf %119, %shuffleResult_31 : f32
        %121 = vector.broadcast %120 : f32 to vector<1x1xf32>
        %122 = vector.insert %120, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %123 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %124 = vector.broadcast %123 : f32 to vector<1x1xf32>
        %125 = arith.maxnumf %121, %124 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %126 = vector.extract %125[0, 0] : f32 from vector<1x1xf32>
        %127 = vector.insert %126, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %122, %127 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %103:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %118 = affine.apply #map18(%arg11)
        %119 = vector.extract %102#1[%118, 0] : f32 from vector<2x1xf32>
        %120 = vector.broadcast %119 : f32 to vector<1x4xf32>
        %121 = affine.apply #map17(%arg11)
        %122 = vector.insert %119, %arg12 [%118, %121] : f32 into vector<2x32xf32>
        %123 = affine.apply #map19(%arg11)
        %124 = vector.insert %119, %122 [%118, %123] : f32 into vector<2x32xf32>
        %125 = affine.apply #map20(%arg11)
        %126 = vector.insert %119, %124 [%118, %125] : f32 into vector<2x32xf32>
        %127 = affine.apply #map21(%arg11)
        %128 = vector.insert %119, %126 [%118, %127] : f32 into vector<2x32xf32>
        %129 = vector.extract %100#1[%118, %121] : f32 from vector<2x32xf32>
        %130 = vector.extract %100#1[%118, %123] : f32 from vector<2x32xf32>
        %131 = vector.extract %100#1[%118, %125] : f32 from vector<2x32xf32>
        %132 = vector.extract %100#1[%118, %127] : f32 from vector<2x32xf32>
        %133 = vector.from_elements %129, %130, %131, %132 : vector<1x4xf32>
        %134 = arith.subf %133, %120 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %135 = vector.extract %134[0, 0] : f32 from vector<1x4xf32>
        %136 = vector.insert %135, %arg13 [%118, %121] : f32 into vector<2x32xf32>
        %137 = vector.extract %134[0, 1] : f32 from vector<1x4xf32>
        %138 = vector.insert %137, %136 [%118, %123] : f32 into vector<2x32xf32>
        %139 = vector.extract %134[0, 2] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %138 [%118, %125] : f32 into vector<2x32xf32>
        %141 = vector.extract %134[0, 3] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %140 [%118, %127] : f32 into vector<2x32xf32>
        affine.yield %128, %142 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>}
      %104:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6, %arg14 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %118 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %119 = vector.broadcast %118 : f32 to vector<1x1xf32>
        %120 = vector.extract %102#1[%arg11, 0] : f32 from vector<2x1xf32>
        %121 = vector.broadcast %120 : f32 to vector<1x1xf32>
        %122 = arith.subf %119, %121 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %123 = vector.extract %122[0, 0] : f32 from vector<1x1xf32>
        %124 = vector.insert %123, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %125 = math.exp2 %122 {tiled = true} : vector<1x1xf32>
        %126 = vector.extract %125[0, 0] : f32 from vector<1x1xf32>
        %127 = vector.insert %126, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %128 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %129 = vector.broadcast %128 : f32 to vector<1x1xf32>
        %130 = arith.mulf %129, %125 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %131 = vector.extract %130[0, 0] : f32 from vector<1x1xf32>
        %132 = vector.insert %131, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %124, %127, %132 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %105:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %118 = affine.apply #map18(%arg11)
        %119 = affine.apply #map17(%arg11)
        %120 = vector.extract %103#1[%118, %119] : f32 from vector<2x32xf32>
        %121 = affine.apply #map19(%arg11)
        %122 = vector.extract %103#1[%118, %121] : f32 from vector<2x32xf32>
        %123 = affine.apply #map20(%arg11)
        %124 = vector.extract %103#1[%118, %123] : f32 from vector<2x32xf32>
        %125 = affine.apply #map21(%arg11)
        %126 = vector.extract %103#1[%118, %125] : f32 from vector<2x32xf32>
        %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
        %128 = math.exp2 %127 {tiled = true} : vector<1x4xf32>
        %129 = vector.extract %128[0, 0] : f32 from vector<1x4xf32>
        %130 = vector.insert %129, %arg12 [%118, %119] : f32 into vector<2x32xf32>
        %131 = vector.extract %128[0, 1] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %130 [%118, %121] : f32 into vector<2x32xf32>
        %133 = vector.extract %128[0, 2] : f32 from vector<1x4xf32>
        %134 = vector.insert %133, %132 [%118, %123] : f32 into vector<2x32xf32>
        %135 = vector.extract %128[0, 3] : f32 from vector<1x4xf32>
        %136 = vector.insert %135, %134 [%118, %125] : f32 into vector<2x32xf32>
        %137 = vector.insert %129, %arg13 [%118, %119] : f32 into vector<2x32xf32>
        %138 = vector.insert %131, %137 [%118, %121] : f32 into vector<2x32xf32>
        %139 = vector.insert %133, %138 [%118, %123] : f32 into vector<2x32xf32>
        %140 = vector.insert %135, %139 [%118, %125] : f32 into vector<2x32xf32>
        affine.yield %136, %140 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %106:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %118 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_2) -> (f32) {
          %128 = affine.apply #map28(%arg14)
          %129 = vector.extract %105#1[%arg11, %128] : f32 from vector<2x32xf32>
          %130 = arith.addf %arg15, %129 : f32
          %131 = affine.apply #map29(%arg14)
          %132 = vector.extract %105#1[%arg11, %131] : f32 from vector<2x32xf32>
          %133 = arith.addf %130, %132 : f32
          %134 = affine.apply #map30(%arg14)
          %135 = vector.extract %105#1[%arg11, %134] : f32 from vector<2x32xf32>
          %136 = arith.addf %133, %135 : f32
          %137 = affine.apply #map31(%arg14)
          %138 = vector.extract %105#1[%arg11, %137] : f32 from vector<2x32xf32>
          %139 = arith.addf %136, %138 : f32
          affine.yield %139 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %118, %c16_i32, %c64_i32 : f32
        %119 = arith.addf %118, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %119, %c32_i32, %c64_i32 : f32
        %120 = arith.addf %119, %shuffleResult_31 : f32
        %121 = vector.broadcast %120 : f32 to vector<1x1xf32>
        %122 = vector.insert %120, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %123 = vector.extract %104#2[%arg11, 0] : f32 from vector<2x1xf32>
        %124 = vector.broadcast %123 : f32 to vector<1x1xf32>
        %125 = arith.addf %121, %124 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %126 = vector.extract %125[0, 0] : f32 from vector<1x1xf32>
        %127 = vector.insert %126, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %122, %127 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %107 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_1) -> (vector<2x32xf16>) {
        %118 = affine.apply #map18(%arg11)
        %119 = affine.apply #map17(%arg11)
        %120 = vector.extract %105#0[%118, %119] : f32 from vector<2x32xf32>
        %121 = affine.apply #map19(%arg11)
        %122 = vector.extract %105#0[%118, %121] : f32 from vector<2x32xf32>
        %123 = affine.apply #map20(%arg11)
        %124 = vector.extract %105#0[%118, %123] : f32 from vector<2x32xf32>
        %125 = affine.apply #map21(%arg11)
        %126 = vector.extract %105#0[%118, %125] : f32 from vector<2x32xf32>
        %127 = vector.from_elements %120, %122, %124, %126 : vector<1x4xf32>
        %128 = arith.truncf %127 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %129 = vector.extract %128[0, 0] : f16 from vector<1x4xf16>
        %130 = vector.insert %129, %arg12 [%118, %119] : f16 into vector<2x32xf16>
        %131 = vector.extract %128[0, 1] : f16 from vector<1x4xf16>
        %132 = vector.insert %131, %130 [%118, %121] : f16 into vector<2x32xf16>
        %133 = vector.extract %128[0, 2] : f16 from vector<1x4xf16>
        %134 = vector.insert %133, %132 [%118, %123] : f16 into vector<2x32xf16>
        %135 = vector.extract %128[0, 3] : f16 from vector<1x4xf16>
        %136 = vector.insert %135, %134 [%118, %125] : f16 into vector<2x32xf16>
        affine.yield %136 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_24 = memref.subview %arg2[0, %93, %arg4, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_24 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_23 : memref<128x32xf16, 3>
      %108 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %118 = affine.apply #map35(%arg11)[%thread_id_x]
        %119 = affine.apply #map36()[%thread_id_x]
        %120 = frisk.read_matrix_from_shm %alloc_23[%118, %119] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %121 = vector.bitcast %120 : vector<4xi32> to vector<8xf16>
        %122 = vector.extract %121[0] : f16 from vector<8xf16>
        %123 = affine.apply #map37()[%thread_id_x]
        %124 = vector.extract %121[%123] : f16 from vector<8xf16>
        %125 = vector.extract %121[1] : f16 from vector<8xf16>
        %126 = affine.apply #map38()[%thread_id_x]
        %127 = vector.extract %121[%126] : f16 from vector<8xf16>
        %128 = vector.extract %121[2] : f16 from vector<8xf16>
        %129 = affine.apply #map39()[%thread_id_x]
        %130 = vector.extract %121[%129] : f16 from vector<8xf16>
        %131 = vector.extract %121[3] : f16 from vector<8xf16>
        %132 = affine.apply #map40()[%thread_id_x]
        %133 = vector.extract %121[%132] : f16 from vector<8xf16>
        %134 = affine.apply #map28(%arg11)
        %135 = vector.insert %122, %arg12 [%134, 0] : f16 into vector<32x2xf16>
        %136 = vector.insert %124, %135 [%134, 1] : f16 into vector<32x2xf16>
        %137 = affine.apply #map29(%arg11)
        %138 = vector.insert %125, %136 [%137, 0] : f16 into vector<32x2xf16>
        %139 = vector.insert %127, %138 [%137, 1] : f16 into vector<32x2xf16>
        %140 = affine.apply #map30(%arg11)
        %141 = vector.insert %128, %139 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %130, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply #map31(%arg11)
        %144 = vector.insert %131, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        affine.yield %145 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %109:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %118 = affine.apply #map1(%arg11)
        %119 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %161 = affine.apply #map28(%arg16)
          %162 = vector.extract %107[%118, %161] : f16 from vector<2x32xf16>
          %163 = affine.apply #map29(%arg16)
          %164 = vector.extract %107[%118, %163] : f16 from vector<2x32xf16>
          %165 = affine.apply #map30(%arg16)
          %166 = vector.extract %107[%118, %165] : f16 from vector<2x32xf16>
          %167 = affine.apply #map31(%arg16)
          %168 = vector.extract %107[%118, %167] : f16 from vector<2x32xf16>
          %169 = vector.from_elements %162, %164, %166, %168 : vector<1x4xf16>
          %170 = affine.apply #map41(%arg11)
          %171 = vector.extract %108[%161, %170] : f16 from vector<32x2xf16>
          %172 = vector.extract %108[%163, %170] : f16 from vector<32x2xf16>
          %173 = vector.extract %108[%165, %170] : f16 from vector<32x2xf16>
          %174 = vector.extract %108[%167, %170] : f16 from vector<32x2xf16>
          %175 = vector.from_elements %171, %172, %173, %174 : vector<4x1xf16>
          %176 = frisk.warp_mma_rr %169, %175, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %176 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %120 = affine.apply #map(%arg11)
        %121 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %122 = vector.insert %121, %arg12 [%118, %120] : f32 into vector<2x8xf32>
        %123 = affine.apply #map2(%arg11)
        %124 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %122 [%118, %123] : f32 into vector<2x8xf32>
        %126 = affine.apply #map3(%arg11)
        %127 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %128 = vector.insert %127, %125 [%118, %126] : f32 into vector<2x8xf32>
        %129 = affine.apply #map4(%arg11)
        %130 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %128 [%118, %129] : f32 into vector<2x8xf32>
        %132 = vector.extract %104#1[%118, 0] : f32 from vector<2x1xf32>
        %133 = vector.broadcast %132 : f32 to vector<1x4xf32>
        %134 = vector.insert %132, %arg13 [%118, %120] : f32 into vector<2x8xf32>
        %135 = vector.insert %132, %134 [%118, %123] : f32 into vector<2x8xf32>
        %136 = vector.insert %132, %135 [%118, %126] : f32 into vector<2x8xf32>
        %137 = vector.insert %132, %136 [%118, %129] : f32 into vector<2x8xf32>
        %138 = vector.extract %arg5[%118, %120] : f32 from vector<2x8xf32>
        %139 = vector.extract %arg5[%118, %123] : f32 from vector<2x8xf32>
        %140 = vector.extract %arg5[%118, %126] : f32 from vector<2x8xf32>
        %141 = vector.extract %arg5[%118, %129] : f32 from vector<2x8xf32>
        %142 = vector.from_elements %138, %139, %140, %141 : vector<1x4xf32>
        %143 = arith.mulf %142, %133 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %144 = vector.extract %143[0, 0] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %arg14 [%118, %120] : f32 into vector<2x8xf32>
        %146 = vector.extract %143[0, 1] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%118, %123] : f32 into vector<2x8xf32>
        %148 = vector.extract %143[0, 2] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%118, %126] : f32 into vector<2x8xf32>
        %150 = vector.extract %143[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%118, %129] : f32 into vector<2x8xf32>
        %152 = arith.addf %143, %119 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg15 [%118, %120] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%118, %123] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%118, %126] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%118, %129] : f32 into vector<2x8xf32>
        affine.yield %131, %137, %151, %160 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_25 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_26 = memref.subview %arg2[0, %93, %arg4, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_26 to %alloc_25 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_25 : memref<128x32xf16, 3>
      %110 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %118 = affine.apply #map35(%arg11)[%thread_id_x]
        %119 = affine.apply #map36()[%thread_id_x]
        %120 = frisk.read_matrix_from_shm %alloc_25[%118, %119] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %121 = vector.bitcast %120 : vector<4xi32> to vector<8xf16>
        %122 = vector.extract %121[0] : f16 from vector<8xf16>
        %123 = affine.apply #map37()[%thread_id_x]
        %124 = vector.extract %121[%123] : f16 from vector<8xf16>
        %125 = vector.extract %121[1] : f16 from vector<8xf16>
        %126 = affine.apply #map38()[%thread_id_x]
        %127 = vector.extract %121[%126] : f16 from vector<8xf16>
        %128 = vector.extract %121[2] : f16 from vector<8xf16>
        %129 = affine.apply #map39()[%thread_id_x]
        %130 = vector.extract %121[%129] : f16 from vector<8xf16>
        %131 = vector.extract %121[3] : f16 from vector<8xf16>
        %132 = affine.apply #map40()[%thread_id_x]
        %133 = vector.extract %121[%132] : f16 from vector<8xf16>
        %134 = affine.apply #map28(%arg11)
        %135 = vector.insert %122, %arg12 [%134, 0] : f16 into vector<32x2xf16>
        %136 = vector.insert %124, %135 [%134, 1] : f16 into vector<32x2xf16>
        %137 = affine.apply #map29(%arg11)
        %138 = vector.insert %125, %136 [%137, 0] : f16 into vector<32x2xf16>
        %139 = vector.insert %127, %138 [%137, 1] : f16 into vector<32x2xf16>
        %140 = affine.apply #map30(%arg11)
        %141 = vector.insert %128, %139 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %130, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply #map31(%arg11)
        %144 = vector.insert %131, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        affine.yield %145 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %111:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %118 = affine.apply #map1(%arg11)
        %119 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %161 = affine.apply #map28(%arg16)
          %162 = vector.extract %107[%118, %161] : f16 from vector<2x32xf16>
          %163 = affine.apply #map29(%arg16)
          %164 = vector.extract %107[%118, %163] : f16 from vector<2x32xf16>
          %165 = affine.apply #map30(%arg16)
          %166 = vector.extract %107[%118, %165] : f16 from vector<2x32xf16>
          %167 = affine.apply #map31(%arg16)
          %168 = vector.extract %107[%118, %167] : f16 from vector<2x32xf16>
          %169 = vector.from_elements %162, %164, %166, %168 : vector<1x4xf16>
          %170 = affine.apply #map41(%arg11)
          %171 = vector.extract %110[%161, %170] : f16 from vector<32x2xf16>
          %172 = vector.extract %110[%163, %170] : f16 from vector<32x2xf16>
          %173 = vector.extract %110[%165, %170] : f16 from vector<32x2xf16>
          %174 = vector.extract %110[%167, %170] : f16 from vector<32x2xf16>
          %175 = vector.from_elements %171, %172, %173, %174 : vector<4x1xf16>
          %176 = frisk.warp_mma_rr %169, %175, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %176 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %120 = affine.apply #map(%arg11)
        %121 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %122 = vector.insert %121, %arg12 [%118, %120] : f32 into vector<2x8xf32>
        %123 = affine.apply #map2(%arg11)
        %124 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %122 [%118, %123] : f32 into vector<2x8xf32>
        %126 = affine.apply #map3(%arg11)
        %127 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %128 = vector.insert %127, %125 [%118, %126] : f32 into vector<2x8xf32>
        %129 = affine.apply #map4(%arg11)
        %130 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %128 [%118, %129] : f32 into vector<2x8xf32>
        %132 = vector.extract %104#1[%118, 0] : f32 from vector<2x1xf32>
        %133 = vector.broadcast %132 : f32 to vector<1x4xf32>
        %134 = vector.insert %132, %arg13 [%118, %120] : f32 into vector<2x8xf32>
        %135 = vector.insert %132, %134 [%118, %123] : f32 into vector<2x8xf32>
        %136 = vector.insert %132, %135 [%118, %126] : f32 into vector<2x8xf32>
        %137 = vector.insert %132, %136 [%118, %129] : f32 into vector<2x8xf32>
        %138 = vector.extract %arg8[%118, %120] : f32 from vector<2x8xf32>
        %139 = vector.extract %arg8[%118, %123] : f32 from vector<2x8xf32>
        %140 = vector.extract %arg8[%118, %126] : f32 from vector<2x8xf32>
        %141 = vector.extract %arg8[%118, %129] : f32 from vector<2x8xf32>
        %142 = vector.from_elements %138, %139, %140, %141 : vector<1x4xf32>
        %143 = arith.mulf %142, %133 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %144 = vector.extract %143[0, 0] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %arg14 [%118, %120] : f32 into vector<2x8xf32>
        %146 = vector.extract %143[0, 1] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%118, %123] : f32 into vector<2x8xf32>
        %148 = vector.extract %143[0, 2] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%118, %126] : f32 into vector<2x8xf32>
        %150 = vector.extract %143[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%118, %129] : f32 into vector<2x8xf32>
        %152 = arith.addf %143, %119 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg15 [%118, %120] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%118, %123] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%118, %126] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%118, %129] : f32 into vector<2x8xf32>
        affine.yield %131, %137, %151, %160 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_27 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_28 = memref.subview %arg2[0, %93, %arg4, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_28 to %alloc_27 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_27 : memref<128x32xf16, 3>
      %112 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %118 = affine.apply #map35(%arg11)[%thread_id_x]
        %119 = affine.apply #map36()[%thread_id_x]
        %120 = frisk.read_matrix_from_shm %alloc_27[%118, %119] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %121 = vector.bitcast %120 : vector<4xi32> to vector<8xf16>
        %122 = vector.extract %121[0] : f16 from vector<8xf16>
        %123 = affine.apply #map37()[%thread_id_x]
        %124 = vector.extract %121[%123] : f16 from vector<8xf16>
        %125 = vector.extract %121[1] : f16 from vector<8xf16>
        %126 = affine.apply #map38()[%thread_id_x]
        %127 = vector.extract %121[%126] : f16 from vector<8xf16>
        %128 = vector.extract %121[2] : f16 from vector<8xf16>
        %129 = affine.apply #map39()[%thread_id_x]
        %130 = vector.extract %121[%129] : f16 from vector<8xf16>
        %131 = vector.extract %121[3] : f16 from vector<8xf16>
        %132 = affine.apply #map40()[%thread_id_x]
        %133 = vector.extract %121[%132] : f16 from vector<8xf16>
        %134 = affine.apply #map28(%arg11)
        %135 = vector.insert %122, %arg12 [%134, 0] : f16 into vector<32x2xf16>
        %136 = vector.insert %124, %135 [%134, 1] : f16 into vector<32x2xf16>
        %137 = affine.apply #map29(%arg11)
        %138 = vector.insert %125, %136 [%137, 0] : f16 into vector<32x2xf16>
        %139 = vector.insert %127, %138 [%137, 1] : f16 into vector<32x2xf16>
        %140 = affine.apply #map30(%arg11)
        %141 = vector.insert %128, %139 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %130, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply #map31(%arg11)
        %144 = vector.insert %131, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        affine.yield %145 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %113:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %118 = affine.apply #map1(%arg11)
        %119 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %161 = affine.apply #map28(%arg16)
          %162 = vector.extract %107[%118, %161] : f16 from vector<2x32xf16>
          %163 = affine.apply #map29(%arg16)
          %164 = vector.extract %107[%118, %163] : f16 from vector<2x32xf16>
          %165 = affine.apply #map30(%arg16)
          %166 = vector.extract %107[%118, %165] : f16 from vector<2x32xf16>
          %167 = affine.apply #map31(%arg16)
          %168 = vector.extract %107[%118, %167] : f16 from vector<2x32xf16>
          %169 = vector.from_elements %162, %164, %166, %168 : vector<1x4xf16>
          %170 = affine.apply #map41(%arg11)
          %171 = vector.extract %112[%161, %170] : f16 from vector<32x2xf16>
          %172 = vector.extract %112[%163, %170] : f16 from vector<32x2xf16>
          %173 = vector.extract %112[%165, %170] : f16 from vector<32x2xf16>
          %174 = vector.extract %112[%167, %170] : f16 from vector<32x2xf16>
          %175 = vector.from_elements %171, %172, %173, %174 : vector<4x1xf16>
          %176 = frisk.warp_mma_rr %169, %175, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %176 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %120 = affine.apply #map(%arg11)
        %121 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %122 = vector.insert %121, %arg12 [%118, %120] : f32 into vector<2x8xf32>
        %123 = affine.apply #map2(%arg11)
        %124 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %122 [%118, %123] : f32 into vector<2x8xf32>
        %126 = affine.apply #map3(%arg11)
        %127 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %128 = vector.insert %127, %125 [%118, %126] : f32 into vector<2x8xf32>
        %129 = affine.apply #map4(%arg11)
        %130 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %128 [%118, %129] : f32 into vector<2x8xf32>
        %132 = vector.extract %104#1[%118, 0] : f32 from vector<2x1xf32>
        %133 = vector.broadcast %132 : f32 to vector<1x4xf32>
        %134 = vector.insert %132, %arg13 [%118, %120] : f32 into vector<2x8xf32>
        %135 = vector.insert %132, %134 [%118, %123] : f32 into vector<2x8xf32>
        %136 = vector.insert %132, %135 [%118, %126] : f32 into vector<2x8xf32>
        %137 = vector.insert %132, %136 [%118, %129] : f32 into vector<2x8xf32>
        %138 = vector.extract %arg9[%118, %120] : f32 from vector<2x8xf32>
        %139 = vector.extract %arg9[%118, %123] : f32 from vector<2x8xf32>
        %140 = vector.extract %arg9[%118, %126] : f32 from vector<2x8xf32>
        %141 = vector.extract %arg9[%118, %129] : f32 from vector<2x8xf32>
        %142 = vector.from_elements %138, %139, %140, %141 : vector<1x4xf32>
        %143 = arith.mulf %142, %133 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %144 = vector.extract %143[0, 0] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %arg14 [%118, %120] : f32 into vector<2x8xf32>
        %146 = vector.extract %143[0, 1] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%118, %123] : f32 into vector<2x8xf32>
        %148 = vector.extract %143[0, 2] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%118, %126] : f32 into vector<2x8xf32>
        %150 = vector.extract %143[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%118, %129] : f32 into vector<2x8xf32>
        %152 = arith.addf %143, %119 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg15 [%118, %120] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%118, %123] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%118, %126] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%118, %129] : f32 into vector<2x8xf32>
        affine.yield %131, %137, %151, %160 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_29 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_30 = memref.subview %arg2[0, %93, %arg4, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_30 to %alloc_29 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_29 : memref<128x32xf16, 3>
      %114 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %118 = affine.apply #map35(%arg11)[%thread_id_x]
        %119 = affine.apply #map36()[%thread_id_x]
        %120 = frisk.read_matrix_from_shm %alloc_29[%118, %119] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %121 = vector.bitcast %120 : vector<4xi32> to vector<8xf16>
        %122 = vector.extract %121[0] : f16 from vector<8xf16>
        %123 = affine.apply #map37()[%thread_id_x]
        %124 = vector.extract %121[%123] : f16 from vector<8xf16>
        %125 = vector.extract %121[1] : f16 from vector<8xf16>
        %126 = affine.apply #map38()[%thread_id_x]
        %127 = vector.extract %121[%126] : f16 from vector<8xf16>
        %128 = vector.extract %121[2] : f16 from vector<8xf16>
        %129 = affine.apply #map39()[%thread_id_x]
        %130 = vector.extract %121[%129] : f16 from vector<8xf16>
        %131 = vector.extract %121[3] : f16 from vector<8xf16>
        %132 = affine.apply #map40()[%thread_id_x]
        %133 = vector.extract %121[%132] : f16 from vector<8xf16>
        %134 = affine.apply #map28(%arg11)
        %135 = vector.insert %122, %arg12 [%134, 0] : f16 into vector<32x2xf16>
        %136 = vector.insert %124, %135 [%134, 1] : f16 into vector<32x2xf16>
        %137 = affine.apply #map29(%arg11)
        %138 = vector.insert %125, %136 [%137, 0] : f16 into vector<32x2xf16>
        %139 = vector.insert %127, %138 [%137, 1] : f16 into vector<32x2xf16>
        %140 = affine.apply #map30(%arg11)
        %141 = vector.insert %128, %139 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %130, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply #map31(%arg11)
        %144 = vector.insert %131, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        affine.yield %145 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %115:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %118 = affine.apply #map1(%arg11)
        %119 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %169 = affine.apply #map28(%arg17)
          %170 = vector.extract %107[%118, %169] : f16 from vector<2x32xf16>
          %171 = affine.apply #map29(%arg17)
          %172 = vector.extract %107[%118, %171] : f16 from vector<2x32xf16>
          %173 = affine.apply #map30(%arg17)
          %174 = vector.extract %107[%118, %173] : f16 from vector<2x32xf16>
          %175 = affine.apply #map31(%arg17)
          %176 = vector.extract %107[%118, %175] : f16 from vector<2x32xf16>
          %177 = vector.from_elements %170, %172, %174, %176 : vector<1x4xf16>
          %178 = affine.apply #map41(%arg11)
          %179 = vector.extract %114[%169, %178] : f16 from vector<32x2xf16>
          %180 = vector.extract %114[%171, %178] : f16 from vector<32x2xf16>
          %181 = vector.extract %114[%173, %178] : f16 from vector<32x2xf16>
          %182 = vector.extract %114[%175, %178] : f16 from vector<32x2xf16>
          %183 = vector.from_elements %179, %180, %181, %182 : vector<4x1xf16>
          %184 = frisk.warp_mma_rr %177, %183, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %184 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %120 = affine.apply #map(%arg11)
        %121 = vector.extract %119[0, 0] : f32 from vector<1x4xf32>
        %122 = vector.insert %121, %arg12 [%118, %120] : f32 into vector<2x8xf32>
        %123 = affine.apply #map2(%arg11)
        %124 = vector.extract %119[0, 1] : f32 from vector<1x4xf32>
        %125 = vector.insert %124, %122 [%118, %123] : f32 into vector<2x8xf32>
        %126 = affine.apply #map3(%arg11)
        %127 = vector.extract %119[0, 2] : f32 from vector<1x4xf32>
        %128 = vector.insert %127, %125 [%118, %126] : f32 into vector<2x8xf32>
        %129 = affine.apply #map4(%arg11)
        %130 = vector.extract %119[0, 3] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %128 [%118, %129] : f32 into vector<2x8xf32>
        %132 = vector.extract %104#1[%118, 0] : f32 from vector<2x1xf32>
        %133 = vector.broadcast %132 : f32 to vector<1x4xf32>
        %134 = vector.insert %132, %arg13 [%118, %120] : f32 into vector<2x8xf32>
        %135 = vector.insert %132, %134 [%118, %123] : f32 into vector<2x8xf32>
        %136 = vector.insert %132, %135 [%118, %126] : f32 into vector<2x8xf32>
        %137 = vector.insert %132, %136 [%118, %129] : f32 into vector<2x8xf32>
        %138 = vector.extract %arg10[%118, %120] : f32 from vector<2x8xf32>
        %139 = vector.extract %arg10[%118, %123] : f32 from vector<2x8xf32>
        %140 = vector.extract %arg10[%118, %126] : f32 from vector<2x8xf32>
        %141 = vector.extract %arg10[%118, %129] : f32 from vector<2x8xf32>
        %142 = vector.from_elements %138, %139, %140, %141 : vector<1x4xf32>
        %143 = arith.mulf %142, %133 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %144 = vector.extract %143[0, 0] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %arg14 [%118, %120] : f32 into vector<2x8xf32>
        %146 = vector.extract %143[0, 1] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%118, %123] : f32 into vector<2x8xf32>
        %148 = vector.extract %143[0, 2] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%118, %126] : f32 into vector<2x8xf32>
        %150 = vector.extract %143[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%118, %129] : f32 into vector<2x8xf32>
        %152 = arith.addf %143, %119 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg15 [%118, %120] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%118, %123] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%118, %126] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%118, %129] : f32 into vector<2x8xf32>
        %161 = vector.extract %109#3[%118, %120] : f32 from vector<2x8xf32>
        %162 = vector.extract %109#3[%118, %123] : f32 from vector<2x8xf32>
        %163 = vector.extract %109#3[%118, %126] : f32 from vector<2x8xf32>
        %164 = vector.extract %109#3[%118, %129] : f32 from vector<2x8xf32>
        %165 = vector.insert %161, %arg16 [%118, %120] : f32 into vector<2x8xf32>
        %166 = vector.insert %162, %165 [%118, %123] : f32 into vector<2x8xf32>
        %167 = vector.insert %163, %166 [%118, %126] : f32 into vector<2x8xf32>
        %168 = vector.insert %164, %167 [%118, %129] : f32 into vector<2x8xf32>
        affine.yield %131, %137, %151, %160, %168 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %116:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %118 = vector.extract %106#1[%arg11, 0] : f32 from vector<2x1xf32>
        %119 = vector.insert %118, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %120 = vector.extract %102#1[%arg11, 0] : f32 from vector<2x1xf32>
        %121 = vector.insert %120, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %119, %121 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %117:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %118 = affine.apply #map1(%arg11)
        %119 = affine.apply #map(%arg11)
        %120 = vector.extract %111#3[%118, %119] : f32 from vector<2x8xf32>
        %121 = affine.apply #map2(%arg11)
        %122 = vector.extract %111#3[%118, %121] : f32 from vector<2x8xf32>
        %123 = affine.apply #map3(%arg11)
        %124 = vector.extract %111#3[%118, %123] : f32 from vector<2x8xf32>
        %125 = affine.apply #map4(%arg11)
        %126 = vector.extract %111#3[%118, %125] : f32 from vector<2x8xf32>
        %127 = vector.insert %120, %arg12 [%118, %119] : f32 into vector<2x8xf32>
        %128 = vector.insert %122, %127 [%118, %121] : f32 into vector<2x8xf32>
        %129 = vector.insert %124, %128 [%118, %123] : f32 into vector<2x8xf32>
        %130 = vector.insert %126, %129 [%118, %125] : f32 into vector<2x8xf32>
        %131 = vector.extract %113#3[%118, %119] : f32 from vector<2x8xf32>
        %132 = vector.extract %113#3[%118, %121] : f32 from vector<2x8xf32>
        %133 = vector.extract %113#3[%118, %123] : f32 from vector<2x8xf32>
        %134 = vector.extract %113#3[%118, %125] : f32 from vector<2x8xf32>
        %135 = vector.insert %131, %arg13 [%118, %119] : f32 into vector<2x8xf32>
        %136 = vector.insert %132, %135 [%118, %121] : f32 into vector<2x8xf32>
        %137 = vector.insert %133, %136 [%118, %123] : f32 into vector<2x8xf32>
        %138 = vector.insert %134, %137 [%118, %125] : f32 into vector<2x8xf32>
        %139 = vector.extract %115#3[%118, %119] : f32 from vector<2x8xf32>
        %140 = vector.extract %115#3[%118, %121] : f32 from vector<2x8xf32>
        %141 = vector.extract %115#3[%118, %123] : f32 from vector<2x8xf32>
        %142 = vector.extract %115#3[%118, %125] : f32 from vector<2x8xf32>
        %143 = vector.insert %139, %arg14 [%118, %119] : f32 into vector<2x8xf32>
        %144 = vector.insert %140, %143 [%118, %121] : f32 into vector<2x8xf32>
        %145 = vector.insert %141, %144 [%118, %123] : f32 into vector<2x8xf32>
        %146 = vector.insert %142, %145 [%118, %125] : f32 into vector<2x8xf32>
        affine.yield %130, %138, %146 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %115#4, %116#0, %116#1, %117#0, %117#1, %117#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %81 = arith.index_cast %thread_id_x : index to i32
    %82 = arith.andi %81, %c16_i32 : i32
    %83 = arith.cmpi ne, %82, %c0_i32 : i32
    %84 = arith.andi %81, %c32_i32 : i32
    %85 = arith.cmpi ne, %84, %c0_i32 : i32
    %86:8 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst_5, %arg12 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %92 = affine.apply #map1(%arg4)
      %93 = affine.apply #map(%arg4)
      %94 = vector.extract %80#5[%92, %93] : f32 from vector<2x8xf32>
      %95 = affine.apply #map2(%arg4)
      %96 = vector.extract %80#5[%92, %95] : f32 from vector<2x8xf32>
      %97 = affine.apply #map3(%arg4)
      %98 = vector.extract %80#5[%92, %97] : f32 from vector<2x8xf32>
      %99 = affine.apply #map4(%arg4)
      %100 = vector.extract %80#5[%92, %99] : f32 from vector<2x8xf32>
      %101 = arith.select %83, %94, %96 : f32
      %shuffleResult, %valid = gpu.shuffle xor %101, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %102 = arith.select %83, %shuffleResult, %94 : f32
      %103 = arith.select %83, %96, %shuffleResult : f32
      %104 = arith.select %83, %98, %100 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %104, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %105 = arith.select %83, %shuffleResult_15, %98 : f32
      %106 = arith.select %83, %100, %shuffleResult_15 : f32
      %107 = arith.select %85, %102, %105 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %107, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %108 = arith.select %85, %shuffleResult_17, %102 : f32
      %109 = arith.select %85, %105, %shuffleResult_17 : f32
      %110 = arith.select %85, %103, %106 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %110, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %85, %shuffleResult_19, %103 : f32
      %112 = arith.select %85, %106, %shuffleResult_19 : f32
      %113 = vector.insert %108, %arg5 [%92, %93] : f32 into vector<2x8xf32>
      %114 = vector.insert %111, %113 [%92, %95] : f32 into vector<2x8xf32>
      %115 = vector.insert %109, %114 [%92, %97] : f32 into vector<2x8xf32>
      %116 = vector.insert %112, %115 [%92, %99] : f32 into vector<2x8xf32>
      %117 = vector.extract %80#4[%92, %93] : f32 from vector<2x8xf32>
      %118 = vector.extract %80#4[%92, %95] : f32 from vector<2x8xf32>
      %119 = vector.extract %80#4[%92, %97] : f32 from vector<2x8xf32>
      %120 = vector.extract %80#4[%92, %99] : f32 from vector<2x8xf32>
      %121 = arith.select %83, %117, %118 : f32
      %shuffleResult_21, %valid_22 = gpu.shuffle xor %121, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %122 = arith.select %83, %shuffleResult_21, %117 : f32
      %123 = arith.select %83, %118, %shuffleResult_21 : f32
      %124 = arith.select %83, %119, %120 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %124, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %125 = arith.select %83, %shuffleResult_23, %119 : f32
      %126 = arith.select %83, %120, %shuffleResult_23 : f32
      %127 = arith.select %85, %122, %125 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %127, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %128 = arith.select %85, %shuffleResult_25, %122 : f32
      %129 = arith.select %85, %125, %shuffleResult_25 : f32
      %130 = arith.select %85, %123, %126 : f32
      %shuffleResult_27, %valid_28 = gpu.shuffle xor %130, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %131 = arith.select %85, %shuffleResult_27, %123 : f32
      %132 = arith.select %85, %126, %shuffleResult_27 : f32
      %133 = vector.insert %128, %arg6 [%92, %93] : f32 into vector<2x8xf32>
      %134 = vector.insert %131, %133 [%92, %95] : f32 into vector<2x8xf32>
      %135 = vector.insert %129, %134 [%92, %97] : f32 into vector<2x8xf32>
      %136 = vector.insert %132, %135 [%92, %99] : f32 into vector<2x8xf32>
      %137 = vector.extract %80#3[%92, %93] : f32 from vector<2x8xf32>
      %138 = vector.extract %80#3[%92, %95] : f32 from vector<2x8xf32>
      %139 = vector.extract %80#3[%92, %97] : f32 from vector<2x8xf32>
      %140 = vector.extract %80#3[%92, %99] : f32 from vector<2x8xf32>
      %141 = arith.select %83, %137, %138 : f32
      %shuffleResult_29, %valid_30 = gpu.shuffle xor %141, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %142 = arith.select %83, %shuffleResult_29, %137 : f32
      %143 = arith.select %83, %138, %shuffleResult_29 : f32
      %144 = arith.select %83, %139, %140 : f32
      %shuffleResult_31, %valid_32 = gpu.shuffle xor %144, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %145 = arith.select %83, %shuffleResult_31, %139 : f32
      %146 = arith.select %83, %140, %shuffleResult_31 : f32
      %147 = arith.select %85, %142, %145 : f32
      %shuffleResult_33, %valid_34 = gpu.shuffle xor %147, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %148 = arith.select %85, %shuffleResult_33, %142 : f32
      %149 = arith.select %85, %145, %shuffleResult_33 : f32
      %150 = arith.select %85, %143, %146 : f32
      %shuffleResult_35, %valid_36 = gpu.shuffle xor %150, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %151 = arith.select %85, %shuffleResult_35, %143 : f32
      %152 = arith.select %85, %146, %shuffleResult_35 : f32
      %153 = vector.insert %148, %arg7 [%92, %93] : f32 into vector<2x8xf32>
      %154 = vector.insert %151, %153 [%92, %95] : f32 into vector<2x8xf32>
      %155 = vector.insert %149, %154 [%92, %97] : f32 into vector<2x8xf32>
      %156 = vector.insert %152, %155 [%92, %99] : f32 into vector<2x8xf32>
      %157 = vector.extract %80#0[%92, %93] : f32 from vector<2x8xf32>
      %158 = vector.extract %80#0[%92, %95] : f32 from vector<2x8xf32>
      %159 = vector.extract %80#0[%92, %97] : f32 from vector<2x8xf32>
      %160 = vector.extract %80#0[%92, %99] : f32 from vector<2x8xf32>
      %161 = arith.select %83, %157, %158 : f32
      %shuffleResult_37, %valid_38 = gpu.shuffle xor %161, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %162 = arith.select %83, %shuffleResult_37, %157 : f32
      %163 = arith.select %83, %158, %shuffleResult_37 : f32
      %164 = arith.select %83, %159, %160 : f32
      %shuffleResult_39, %valid_40 = gpu.shuffle xor %164, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %165 = arith.select %83, %shuffleResult_39, %159 : f32
      %166 = arith.select %83, %160, %shuffleResult_39 : f32
      %167 = arith.select %85, %162, %165 : f32
      %shuffleResult_41, %valid_42 = gpu.shuffle xor %167, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %168 = arith.select %85, %shuffleResult_41, %162 : f32
      %169 = arith.select %85, %165, %shuffleResult_41 : f32
      %170 = arith.select %85, %163, %166 : f32
      %shuffleResult_43, %valid_44 = gpu.shuffle xor %170, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %171 = arith.select %85, %shuffleResult_43, %163 : f32
      %172 = arith.select %85, %166, %shuffleResult_43 : f32
      %173 = vector.from_elements %168, %171, %169, %172 : vector<1x4xf32>
      %174 = vector.insert %168, %arg8 [%92, %93] : f32 into vector<2x8xf32>
      %175 = vector.insert %171, %174 [%92, %95] : f32 into vector<2x8xf32>
      %176 = vector.insert %169, %175 [%92, %97] : f32 into vector<2x8xf32>
      %177 = vector.insert %172, %176 [%92, %99] : f32 into vector<2x8xf32>
      %178 = vector.extract %80#1[%92, 0] : f32 from vector<2x1xf32>
      %179 = vector.broadcast %178 : f32 to vector<1x4xf32>
      %180 = vector.insert %178, %arg9 [%92, %93] : f32 into vector<2x8xf32>
      %181 = vector.insert %178, %180 [%92, %95] : f32 into vector<2x8xf32>
      %182 = vector.insert %178, %181 [%92, %97] : f32 into vector<2x8xf32>
      %183 = vector.insert %178, %182 [%92, %99] : f32 into vector<2x8xf32>
      %184 = arith.divf %173, %179 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %185 = vector.extract %184[0, 0] : f32 from vector<1x4xf32>
      %186 = vector.insert %185, %arg10 [%92, %93] : f32 into vector<2x8xf32>
      %187 = vector.extract %184[0, 1] : f32 from vector<1x4xf32>
      %188 = vector.insert %187, %186 [%92, %95] : f32 into vector<2x8xf32>
      %189 = vector.extract %184[0, 2] : f32 from vector<1x4xf32>
      %190 = vector.insert %189, %188 [%92, %97] : f32 into vector<2x8xf32>
      %191 = vector.extract %184[0, 3] : f32 from vector<1x4xf32>
      %192 = vector.insert %191, %190 [%92, %99] : f32 into vector<2x8xf32>
      %193 = arith.truncf %184 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %194 = vector.extract %193[0, 0] : f16 from vector<1x4xf16>
      %195 = vector.insert %194, %arg11 [%92, %93] : f16 into vector<2x8xf16>
      %196 = vector.extract %193[0, 1] : f16 from vector<1x4xf16>
      %197 = vector.insert %196, %195 [%92, %95] : f16 into vector<2x8xf16>
      %198 = vector.extract %193[0, 2] : f16 from vector<1x4xf16>
      %199 = vector.insert %198, %197 [%92, %97] : f16 into vector<2x8xf16>
      %200 = vector.extract %193[0, 3] : f16 from vector<1x4xf16>
      %201 = vector.insert %200, %199 [%92, %99] : f16 into vector<2x8xf16>
      %202 = vector.insert %194, %arg12 [%92, %93] : f16 into vector<2x8xf16>
      %203 = vector.insert %196, %202 [%92, %95] : f16 into vector<2x8xf16>
      %204 = vector.insert %198, %203 [%92, %97] : f16 into vector<2x8xf16>
      %205 = vector.insert %200, %204 [%92, %99] : f16 into vector<2x8xf16>
      affine.yield %116, %136, %156, %177, %183, %192, %201, %205 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    %87 = affine.apply #map42()[%thread_id_x]
    %88 = arith.cmpi ult, %87, %c4 : index
    scf.if %88 {
      affine.for %arg4 = 0 to 4 {
        %92 = affine.apply #map1(%arg4)
        %93 = affine.apply #map(%arg4)
        %94 = vector.extract %86#7[%92, %93] : f16 from vector<2x8xf16>
        %95 = affine.apply #map2(%arg4)
        %96 = vector.extract %86#7[%92, %95] : f16 from vector<2x8xf16>
        %97 = affine.apply #map3(%arg4)
        %98 = vector.extract %86#7[%92, %97] : f16 from vector<2x8xf16>
        %99 = affine.apply #map4(%arg4)
        %100 = vector.extract %86#7[%92, %99] : f16 from vector<2x8xf16>
        affine.store %94, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %96, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %98, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %100, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %89:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %92 = affine.apply #map1(%arg4)
      %93 = vector.extract %80#1[%92, 0] : f32 from vector<2x1xf32>
      %94 = vector.broadcast %93 : f32 to vector<1x4xf32>
      %95 = affine.apply #map(%arg4)
      %96 = vector.insert %93, %arg5 [%92, %95] : f32 into vector<2x8xf32>
      %97 = affine.apply #map2(%arg4)
      %98 = vector.insert %93, %96 [%92, %97] : f32 into vector<2x8xf32>
      %99 = affine.apply #map3(%arg4)
      %100 = vector.insert %93, %98 [%92, %99] : f32 into vector<2x8xf32>
      %101 = affine.apply #map4(%arg4)
      %102 = vector.insert %93, %100 [%92, %101] : f32 into vector<2x8xf32>
      %103 = vector.extract %86#2[%92, %95] : f32 from vector<2x8xf32>
      %104 = vector.extract %86#2[%92, %97] : f32 from vector<2x8xf32>
      %105 = vector.extract %86#2[%92, %99] : f32 from vector<2x8xf32>
      %106 = vector.extract %86#2[%92, %101] : f32 from vector<2x8xf32>
      %107 = vector.from_elements %103, %104, %105, %106 : vector<1x4xf32>
      %108 = arith.divf %107, %94 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %109 = vector.extract %108[0, 0] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %arg6 [%92, %95] : f32 into vector<2x8xf32>
      %111 = vector.extract %108[0, 1] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %110 [%92, %97] : f32 into vector<2x8xf32>
      %113 = vector.extract %108[0, 2] : f32 from vector<1x4xf32>
      %114 = vector.insert %113, %112 [%92, %99] : f32 into vector<2x8xf32>
      %115 = vector.extract %108[0, 3] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %114 [%92, %101] : f32 into vector<2x8xf32>
      %117 = arith.truncf %108 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %118 = vector.extract %117[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg7 [%92, %95] : f16 into vector<2x8xf16>
      %120 = vector.extract %117[0, 1] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %119 [%92, %97] : f16 into vector<2x8xf16>
      %122 = vector.extract %117[0, 2] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %121 [%92, %99] : f16 into vector<2x8xf16>
      %124 = vector.extract %117[0, 3] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %123 [%92, %101] : f16 into vector<2x8xf16>
      %126 = vector.insert %118, %arg8 [%92, %95] : f16 into vector<2x8xf16>
      %127 = vector.insert %120, %126 [%92, %97] : f16 into vector<2x8xf16>
      %128 = vector.insert %122, %127 [%92, %99] : f16 into vector<2x8xf16>
      %129 = vector.insert %124, %128 [%92, %101] : f16 into vector<2x8xf16>
      affine.yield %102, %116, %125, %129 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %88 {
      affine.for %arg4 = 0 to 4 {
        %92 = affine.apply #map1(%arg4)
        %93 = affine.apply #map(%arg4)
        %94 = vector.extract %89#3[%92, %93] : f16 from vector<2x8xf16>
        %95 = affine.apply #map2(%arg4)
        %96 = vector.extract %89#3[%92, %95] : f16 from vector<2x8xf16>
        %97 = affine.apply #map3(%arg4)
        %98 = vector.extract %89#3[%92, %97] : f16 from vector<2x8xf16>
        %99 = affine.apply #map4(%arg4)
        %100 = vector.extract %89#3[%92, %99] : f16 from vector<2x8xf16>
        affine.store %94, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %96, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %98, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %100, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %90:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %92 = affine.apply #map1(%arg4)
      %93 = vector.extract %80#1[%92, 0] : f32 from vector<2x1xf32>
      %94 = vector.broadcast %93 : f32 to vector<1x4xf32>
      %95 = affine.apply #map(%arg4)
      %96 = vector.insert %93, %arg5 [%92, %95] : f32 into vector<2x8xf32>
      %97 = affine.apply #map2(%arg4)
      %98 = vector.insert %93, %96 [%92, %97] : f32 into vector<2x8xf32>
      %99 = affine.apply #map3(%arg4)
      %100 = vector.insert %93, %98 [%92, %99] : f32 into vector<2x8xf32>
      %101 = affine.apply #map4(%arg4)
      %102 = vector.insert %93, %100 [%92, %101] : f32 into vector<2x8xf32>
      %103 = vector.extract %86#1[%92, %95] : f32 from vector<2x8xf32>
      %104 = vector.extract %86#1[%92, %97] : f32 from vector<2x8xf32>
      %105 = vector.extract %86#1[%92, %99] : f32 from vector<2x8xf32>
      %106 = vector.extract %86#1[%92, %101] : f32 from vector<2x8xf32>
      %107 = vector.from_elements %103, %104, %105, %106 : vector<1x4xf32>
      %108 = arith.divf %107, %94 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %109 = vector.extract %108[0, 0] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %arg6 [%92, %95] : f32 into vector<2x8xf32>
      %111 = vector.extract %108[0, 1] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %110 [%92, %97] : f32 into vector<2x8xf32>
      %113 = vector.extract %108[0, 2] : f32 from vector<1x4xf32>
      %114 = vector.insert %113, %112 [%92, %99] : f32 into vector<2x8xf32>
      %115 = vector.extract %108[0, 3] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %114 [%92, %101] : f32 into vector<2x8xf32>
      %117 = arith.truncf %108 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %118 = vector.extract %117[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg7 [%92, %95] : f16 into vector<2x8xf16>
      %120 = vector.extract %117[0, 1] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %119 [%92, %97] : f16 into vector<2x8xf16>
      %122 = vector.extract %117[0, 2] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %121 [%92, %99] : f16 into vector<2x8xf16>
      %124 = vector.extract %117[0, 3] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %123 [%92, %101] : f16 into vector<2x8xf16>
      %126 = vector.insert %118, %arg8 [%92, %95] : f16 into vector<2x8xf16>
      %127 = vector.insert %120, %126 [%92, %97] : f16 into vector<2x8xf16>
      %128 = vector.insert %122, %127 [%92, %99] : f16 into vector<2x8xf16>
      %129 = vector.insert %124, %128 [%92, %101] : f16 into vector<2x8xf16>
      affine.yield %102, %116, %125, %129 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %88 {
      affine.for %arg4 = 0 to 4 {
        %92 = affine.apply #map1(%arg4)
        %93 = affine.apply #map(%arg4)
        %94 = vector.extract %90#3[%92, %93] : f16 from vector<2x8xf16>
        %95 = affine.apply #map2(%arg4)
        %96 = vector.extract %90#3[%92, %95] : f16 from vector<2x8xf16>
        %97 = affine.apply #map3(%arg4)
        %98 = vector.extract %90#3[%92, %97] : f16 from vector<2x8xf16>
        %99 = affine.apply #map4(%arg4)
        %100 = vector.extract %90#3[%92, %99] : f16 from vector<2x8xf16>
        affine.store %94, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %96, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %98, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %100, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %91:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %92 = affine.apply #map1(%arg4)
      %93 = vector.extract %80#1[%92, 0] : f32 from vector<2x1xf32>
      %94 = vector.broadcast %93 : f32 to vector<1x4xf32>
      %95 = affine.apply #map(%arg4)
      %96 = vector.insert %93, %arg5 [%92, %95] : f32 into vector<2x8xf32>
      %97 = affine.apply #map2(%arg4)
      %98 = vector.insert %93, %96 [%92, %97] : f32 into vector<2x8xf32>
      %99 = affine.apply #map3(%arg4)
      %100 = vector.insert %93, %98 [%92, %99] : f32 into vector<2x8xf32>
      %101 = affine.apply #map4(%arg4)
      %102 = vector.insert %93, %100 [%92, %101] : f32 into vector<2x8xf32>
      %103 = vector.extract %86#0[%92, %95] : f32 from vector<2x8xf32>
      %104 = vector.extract %86#0[%92, %97] : f32 from vector<2x8xf32>
      %105 = vector.extract %86#0[%92, %99] : f32 from vector<2x8xf32>
      %106 = vector.extract %86#0[%92, %101] : f32 from vector<2x8xf32>
      %107 = vector.from_elements %103, %104, %105, %106 : vector<1x4xf32>
      %108 = arith.divf %107, %94 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %109 = vector.extract %108[0, 0] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %arg6 [%92, %95] : f32 into vector<2x8xf32>
      %111 = vector.extract %108[0, 1] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %110 [%92, %97] : f32 into vector<2x8xf32>
      %113 = vector.extract %108[0, 2] : f32 from vector<1x4xf32>
      %114 = vector.insert %113, %112 [%92, %99] : f32 into vector<2x8xf32>
      %115 = vector.extract %108[0, 3] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %114 [%92, %101] : f32 into vector<2x8xf32>
      %117 = arith.truncf %108 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %118 = vector.extract %117[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg7 [%92, %95] : f16 into vector<2x8xf16>
      %120 = vector.extract %117[0, 1] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %119 [%92, %97] : f16 into vector<2x8xf16>
      %122 = vector.extract %117[0, 2] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %121 [%92, %99] : f16 into vector<2x8xf16>
      %124 = vector.extract %117[0, 3] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %123 [%92, %101] : f16 into vector<2x8xf16>
      %126 = vector.insert %118, %arg8 [%92, %95] : f16 into vector<2x8xf16>
      %127 = vector.insert %120, %126 [%92, %97] : f16 into vector<2x8xf16>
      %128 = vector.insert %122, %127 [%92, %99] : f16 into vector<2x8xf16>
      %129 = vector.insert %124, %128 [%92, %101] : f16 into vector<2x8xf16>
      affine.yield %102, %116, %125, %129 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %88 {
      affine.for %arg4 = 0 to 4 {
        %92 = affine.apply #map1(%arg4)
        %93 = affine.apply #map(%arg4)
        %94 = vector.extract %91#3[%92, %93] : f16 from vector<2x8xf16>
        %95 = affine.apply #map2(%arg4)
        %96 = vector.extract %91#3[%92, %95] : f16 from vector<2x8xf16>
        %97 = affine.apply #map3(%arg4)
        %98 = vector.extract %91#3[%92, %97] : f16 from vector<2x8xf16>
        %99 = affine.apply #map4(%arg4)
        %100 = vector.extract %91#3[%92, %99] : f16 from vector<2x8xf16>
        affine.store %94, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %96, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %98, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %100, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

