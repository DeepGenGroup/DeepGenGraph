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
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
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
    %9:4 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6, %arg6 = %cst_6, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %96 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %97 = vector.insert %cst_7, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %98 = vector.insert %cst_7, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %99 = vector.insert %cst_7, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %96, %97, %98, %99 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %96 = affine.apply #map(%arg4)
      %97 = affine.apply #map1(%arg4)
      %98 = vector.insert %cst, %arg5 [%97, %96] : f32 into vector<2x8xf32>
      %99 = affine.apply #map2(%arg4)
      %100 = vector.insert %cst, %98 [%97, %99] : f32 into vector<2x8xf32>
      %101 = affine.apply #map3(%arg4)
      %102 = vector.insert %cst, %100 [%97, %101] : f32 into vector<2x8xf32>
      %103 = affine.apply #map4(%arg4)
      %104 = vector.insert %cst, %102 [%97, %103] : f32 into vector<2x8xf32>
      affine.yield %104 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6) -> (vector<2x1xf32>) {
      %96 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %96 : vector<2x1xf32>
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
    %24:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %23) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %96 = affine.apply #map12(%arg4)[%thread_id_x]
      %97 = affine.apply #map13(%arg4)[%thread_id_x]
      %98 = memref.load %alloc[%96, %97] : memref<128x32xf16, 3>
      %99 = affine.apply #map14(%arg4)[%thread_id_x]
      %100 = memref.load %alloc[%96, %99] : memref<128x32xf16, 3>
      %101 = affine.apply #map15(%arg4)[%thread_id_x]
      %102 = memref.load %alloc[%96, %101] : memref<128x32xf16, 3>
      %103 = affine.apply #map16(%arg4)[%thread_id_x]
      %104 = memref.load %alloc[%96, %103] : memref<128x32xf16, 3>
      %105 = vector.from_elements %98, %100, %102, %104 : vector<1x4xf16>
      %106 = affine.apply #map1(%arg4)
      %107 = affine.apply #map(%arg4)
      %108 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %109 = vector.insert %108, %arg5 [%106, %107] : f16 into vector<2x8xf16>
      %110 = affine.apply #map2(%arg4)
      %111 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %112 = vector.insert %111, %109 [%106, %110] : f16 into vector<2x8xf16>
      %113 = affine.apply #map3(%arg4)
      %114 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %112 [%106, %113] : f16 into vector<2x8xf16>
      %116 = affine.apply #map4(%arg4)
      %117 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %115 [%106, %116] : f16 into vector<2x8xf16>
      %119 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %120 = vector.insert %8, %arg6 [%106, %107] : f16 into vector<2x8xf16>
      %121 = vector.insert %8, %120 [%106, %110] : f16 into vector<2x8xf16>
      %122 = vector.insert %8, %121 [%106, %113] : f16 into vector<2x8xf16>
      %123 = vector.insert %8, %122 [%106, %116] : f16 into vector<2x8xf16>
      %124 = arith.mulf %arg8, %119 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %125 = vector.extract %124[0, 0] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %arg7 [%106, %107] : f16 into vector<2x8xf16>
      %127 = vector.extract %124[0, 1] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%106, %110] : f16 into vector<2x8xf16>
      %129 = vector.extract %124[0, 2] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%106, %113] : f16 into vector<2x8xf16>
      %131 = vector.extract %124[0, 3] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%106, %116] : f16 into vector<2x8xf16>
      affine.yield %118, %123, %132, %105 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %25 = vector.broadcast %8 : f16 to vector<1x4xf16>
    %26 = arith.mulf %24#3, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %27 = vector.extract %26[0, 0] : f16 from vector<1x4xf16>
    %28 = vector.insert %27, %24#2 [1, 4] : f16 into vector<2x8xf16>
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
    %40:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %39) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %96 = affine.apply #map12(%arg4)[%thread_id_x]
      %97 = affine.apply #map13(%arg4)[%thread_id_x]
      %98 = memref.load %alloc_9[%96, %97] : memref<128x32xf16, 3>
      %99 = affine.apply #map14(%arg4)[%thread_id_x]
      %100 = memref.load %alloc_9[%96, %99] : memref<128x32xf16, 3>
      %101 = affine.apply #map15(%arg4)[%thread_id_x]
      %102 = memref.load %alloc_9[%96, %101] : memref<128x32xf16, 3>
      %103 = affine.apply #map16(%arg4)[%thread_id_x]
      %104 = memref.load %alloc_9[%96, %103] : memref<128x32xf16, 3>
      %105 = vector.from_elements %98, %100, %102, %104 : vector<1x4xf16>
      %106 = affine.apply #map1(%arg4)
      %107 = affine.apply #map(%arg4)
      %108 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %109 = vector.insert %108, %arg5 [%106, %107] : f16 into vector<2x8xf16>
      %110 = affine.apply #map2(%arg4)
      %111 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %112 = vector.insert %111, %109 [%106, %110] : f16 into vector<2x8xf16>
      %113 = affine.apply #map3(%arg4)
      %114 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %112 [%106, %113] : f16 into vector<2x8xf16>
      %116 = affine.apply #map4(%arg4)
      %117 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %115 [%106, %116] : f16 into vector<2x8xf16>
      %119 = vector.insert %8, %arg6 [%106, %107] : f16 into vector<2x8xf16>
      %120 = vector.insert %8, %119 [%106, %110] : f16 into vector<2x8xf16>
      %121 = vector.insert %8, %120 [%106, %113] : f16 into vector<2x8xf16>
      %122 = vector.insert %8, %121 [%106, %116] : f16 into vector<2x8xf16>
      %123 = arith.mulf %arg8, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %124 = vector.extract %123[0, 0] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %arg7 [%106, %107] : f16 into vector<2x8xf16>
      %126 = vector.extract %123[0, 1] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %125 [%106, %110] : f16 into vector<2x8xf16>
      %128 = vector.extract %123[0, 2] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%106, %113] : f16 into vector<2x8xf16>
      %130 = vector.extract %123[0, 3] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%106, %116] : f16 into vector<2x8xf16>
      affine.yield %118, %122, %131, %105 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %41 = arith.mulf %40#3, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %42 = vector.extract %41[0, 0] : f16 from vector<1x4xf16>
    %43 = vector.insert %42, %40#2 [1, 4] : f16 into vector<2x8xf16>
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
    %55:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %54) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %96 = affine.apply #map12(%arg4)[%thread_id_x]
      %97 = affine.apply #map13(%arg4)[%thread_id_x]
      %98 = memref.load %alloc_11[%96, %97] : memref<128x32xf16, 3>
      %99 = affine.apply #map14(%arg4)[%thread_id_x]
      %100 = memref.load %alloc_11[%96, %99] : memref<128x32xf16, 3>
      %101 = affine.apply #map15(%arg4)[%thread_id_x]
      %102 = memref.load %alloc_11[%96, %101] : memref<128x32xf16, 3>
      %103 = affine.apply #map16(%arg4)[%thread_id_x]
      %104 = memref.load %alloc_11[%96, %103] : memref<128x32xf16, 3>
      %105 = vector.from_elements %98, %100, %102, %104 : vector<1x4xf16>
      %106 = affine.apply #map1(%arg4)
      %107 = affine.apply #map(%arg4)
      %108 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %109 = vector.insert %108, %arg5 [%106, %107] : f16 into vector<2x8xf16>
      %110 = affine.apply #map2(%arg4)
      %111 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %112 = vector.insert %111, %109 [%106, %110] : f16 into vector<2x8xf16>
      %113 = affine.apply #map3(%arg4)
      %114 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %112 [%106, %113] : f16 into vector<2x8xf16>
      %116 = affine.apply #map4(%arg4)
      %117 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %115 [%106, %116] : f16 into vector<2x8xf16>
      %119 = vector.insert %8, %arg6 [%106, %107] : f16 into vector<2x8xf16>
      %120 = vector.insert %8, %119 [%106, %110] : f16 into vector<2x8xf16>
      %121 = vector.insert %8, %120 [%106, %113] : f16 into vector<2x8xf16>
      %122 = vector.insert %8, %121 [%106, %116] : f16 into vector<2x8xf16>
      %123 = arith.mulf %arg8, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %124 = vector.extract %123[0, 0] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %arg7 [%106, %107] : f16 into vector<2x8xf16>
      %126 = vector.extract %123[0, 1] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %125 [%106, %110] : f16 into vector<2x8xf16>
      %128 = vector.extract %123[0, 2] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%106, %113] : f16 into vector<2x8xf16>
      %130 = vector.extract %123[0, 3] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%106, %116] : f16 into vector<2x8xf16>
      affine.yield %118, %122, %131, %105 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %56 = arith.mulf %55#3, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %57 = vector.extract %56[0, 0] : f16 from vector<1x4xf16>
    %58 = vector.insert %57, %55#2 [1, 4] : f16 into vector<2x8xf16>
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
    %70:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %69) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %96 = affine.apply #map12(%arg4)[%thread_id_x]
      %97 = affine.apply #map13(%arg4)[%thread_id_x]
      %98 = memref.load %alloc_13[%96, %97] : memref<128x32xf16, 3>
      %99 = affine.apply #map14(%arg4)[%thread_id_x]
      %100 = memref.load %alloc_13[%96, %99] : memref<128x32xf16, 3>
      %101 = affine.apply #map15(%arg4)[%thread_id_x]
      %102 = memref.load %alloc_13[%96, %101] : memref<128x32xf16, 3>
      %103 = affine.apply #map16(%arg4)[%thread_id_x]
      %104 = memref.load %alloc_13[%96, %103] : memref<128x32xf16, 3>
      %105 = vector.from_elements %98, %100, %102, %104 : vector<1x4xf16>
      %106 = affine.apply #map1(%arg4)
      %107 = affine.apply #map(%arg4)
      %108 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %109 = vector.insert %108, %arg5 [%106, %107] : f16 into vector<2x8xf16>
      %110 = affine.apply #map2(%arg4)
      %111 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %112 = vector.insert %111, %109 [%106, %110] : f16 into vector<2x8xf16>
      %113 = affine.apply #map3(%arg4)
      %114 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %112 [%106, %113] : f16 into vector<2x8xf16>
      %116 = affine.apply #map4(%arg4)
      %117 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %115 [%106, %116] : f16 into vector<2x8xf16>
      %119 = vector.insert %8, %arg6 [%106, %107] : f16 into vector<2x8xf16>
      %120 = vector.insert %8, %119 [%106, %110] : f16 into vector<2x8xf16>
      %121 = vector.insert %8, %120 [%106, %113] : f16 into vector<2x8xf16>
      %122 = vector.insert %8, %121 [%106, %116] : f16 into vector<2x8xf16>
      %123 = arith.mulf %arg8, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %124 = vector.extract %123[0, 0] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %arg7 [%106, %107] : f16 into vector<2x8xf16>
      %126 = vector.extract %123[0, 1] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %125 [%106, %110] : f16 into vector<2x8xf16>
      %128 = vector.extract %123[0, 2] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%106, %113] : f16 into vector<2x8xf16>
      %130 = vector.extract %123[0, 3] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%106, %116] : f16 into vector<2x8xf16>
      affine.yield %118, %122, %131, %105 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %71 = arith.mulf %70#3, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %72 = vector.extract %71[0, 0] : f16 from vector<1x4xf16>
    %73 = vector.insert %72, %70#2 [1, 4] : f16 into vector<2x8xf16>
    %74 = vector.extract %71[0, 1] : f16 from vector<1x4xf16>
    %75 = vector.insert %74, %73 [1, 5] : f16 into vector<2x8xf16>
    %76 = vector.extract %71[0, 2] : f16 from vector<1x4xf16>
    %77 = vector.insert %76, %75 [1, 6] : f16 into vector<2x8xf16>
    %78 = vector.extract %71[0, 3] : f16 from vector<1x4xf16>
    %79 = vector.insert %78, %77 [1, 7] : f16 into vector<2x8xf16>
    %80:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#3, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %96 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %124 = affine.apply #map17(%arg11)
        %125 = affine.apply #map18(%arg11)
        %126 = vector.insert %cst, %arg12 [%125, %124] : f32 into vector<2x32xf32>
        %127 = affine.apply #map19(%arg11)
        %128 = vector.insert %cst, %126 [%125, %127] : f32 into vector<2x32xf32>
        %129 = affine.apply #map20(%arg11)
        %130 = vector.insert %cst, %128 [%125, %129] : f32 into vector<2x32xf32>
        %131 = affine.apply #map21(%arg11)
        %132 = vector.insert %cst, %130 [%125, %131] : f32 into vector<2x32xf32>
        affine.yield %132 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %97 = affine.apply #map22()[%7]
      %subview_16 = memref.subview %transpose[0, %97, 0, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_16 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<32x128xf16, 3>
      %98:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %124 = affine.apply #map23(%arg11)[%thread_id_x]
        %125 = memref.load %alloc_15[%15, %124] : memref<32x128xf16, 3>
        %126 = memref.load %alloc_15[%17, %124] : memref<32x128xf16, 3>
        %127 = memref.load %alloc_15[%19, %124] : memref<32x128xf16, 3>
        %128 = memref.load %alloc_15[%21, %124] : memref<32x128xf16, 3>
        %129 = vector.from_elements %125, %126, %127, %128 : vector<4x1xf16>
        %130:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %129) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %160 = affine.apply #map24(%arg14)[%thread_id_x]
          %161 = memref.load %alloc_15[%160, %124] : memref<32x128xf16, 3>
          %162 = affine.apply #map25(%arg14)[%thread_id_x]
          %163 = memref.load %alloc_15[%162, %124] : memref<32x128xf16, 3>
          %164 = affine.apply #map26(%arg14)[%thread_id_x]
          %165 = memref.load %alloc_15[%164, %124] : memref<32x128xf16, 3>
          %166 = affine.apply #map27(%arg14)[%thread_id_x]
          %167 = memref.load %alloc_15[%166, %124] : memref<32x128xf16, 3>
          %168 = vector.from_elements %161, %163, %165, %167 : vector<4x1xf16>
          %169 = affine.apply #map18(%arg11)
          %170 = affine.apply #map28(%arg14)
          %171 = vector.extract %34[%169, %170] : f16 from vector<2x8xf16>
          %172 = affine.apply #map29(%arg14)
          %173 = vector.extract %34[%169, %172] : f16 from vector<2x8xf16>
          %174 = affine.apply #map30(%arg14)
          %175 = vector.extract %34[%169, %174] : f16 from vector<2x8xf16>
          %176 = affine.apply #map31(%arg14)
          %177 = vector.extract %34[%169, %176] : f16 from vector<2x8xf16>
          %178 = vector.from_elements %171, %173, %175, %177 : vector<1x4xf16>
          %179 = frisk.warp_mma_rr %178, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %179, %168 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %131 = affine.apply #map18(%arg11)
        %132 = vector.extract %34[%131, 4] : f16 from vector<2x8xf16>
        %133 = vector.extract %34[%131, 5] : f16 from vector<2x8xf16>
        %134 = vector.extract %34[%131, 6] : f16 from vector<2x8xf16>
        %135 = vector.extract %34[%131, 7] : f16 from vector<2x8xf16>
        %136 = vector.from_elements %132, %133, %134, %135 : vector<1x4xf16>
        %137 = frisk.warp_mma_rr %136, %130#1, %130#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %138 = affine.apply #map17(%arg11)
        %139 = vector.extract %96[%131, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply #map19(%arg11)
        %141 = vector.extract %96[%131, %140] : f32 from vector<2x32xf32>
        %142 = affine.apply #map20(%arg11)
        %143 = vector.extract %96[%131, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply #map21(%arg11)
        %145 = vector.extract %96[%131, %144] : f32 from vector<2x32xf32>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<1x4xf32>
        %147 = arith.addf %146, %137 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %148 = vector.extract %147[0, 0] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %arg12 [%131, %138] : f32 into vector<2x32xf32>
        %150 = vector.extract %147[0, 1] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%131, %140] : f32 into vector<2x32xf32>
        %152 = vector.extract %147[0, 2] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %151 [%131, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %147[0, 3] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%131, %144] : f32 into vector<2x32xf32>
        %156 = vector.insert %148, %arg13 [%131, %138] : f32 into vector<2x32xf32>
        %157 = vector.insert %150, %156 [%131, %140] : f32 into vector<2x32xf32>
        %158 = vector.insert %152, %157 [%131, %142] : f32 into vector<2x32xf32>
        %159 = vector.insert %154, %158 [%131, %144] : f32 into vector<2x32xf32>
        affine.yield %155, %159 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %99 = affine.apply #map32(%arg4)[%block_id_z, %7]
      %subview_18 = memref.subview %transpose[0, %99, 32, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_18 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_17 : memref<32x128xf16, 3>
      %100:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %124 = affine.apply #map23(%arg11)[%thread_id_x]
        %125 = memref.load %alloc_17[%15, %124] : memref<32x128xf16, 3>
        %126 = memref.load %alloc_17[%17, %124] : memref<32x128xf16, 3>
        %127 = memref.load %alloc_17[%19, %124] : memref<32x128xf16, 3>
        %128 = memref.load %alloc_17[%21, %124] : memref<32x128xf16, 3>
        %129 = vector.from_elements %125, %126, %127, %128 : vector<4x1xf16>
        %130:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %129) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %160 = affine.apply #map24(%arg14)[%thread_id_x]
          %161 = memref.load %alloc_17[%160, %124] : memref<32x128xf16, 3>
          %162 = affine.apply #map25(%arg14)[%thread_id_x]
          %163 = memref.load %alloc_17[%162, %124] : memref<32x128xf16, 3>
          %164 = affine.apply #map26(%arg14)[%thread_id_x]
          %165 = memref.load %alloc_17[%164, %124] : memref<32x128xf16, 3>
          %166 = affine.apply #map27(%arg14)[%thread_id_x]
          %167 = memref.load %alloc_17[%166, %124] : memref<32x128xf16, 3>
          %168 = vector.from_elements %161, %163, %165, %167 : vector<4x1xf16>
          %169 = affine.apply #map18(%arg11)
          %170 = affine.apply #map28(%arg14)
          %171 = vector.extract %49[%169, %170] : f16 from vector<2x8xf16>
          %172 = affine.apply #map29(%arg14)
          %173 = vector.extract %49[%169, %172] : f16 from vector<2x8xf16>
          %174 = affine.apply #map30(%arg14)
          %175 = vector.extract %49[%169, %174] : f16 from vector<2x8xf16>
          %176 = affine.apply #map31(%arg14)
          %177 = vector.extract %49[%169, %176] : f16 from vector<2x8xf16>
          %178 = vector.from_elements %171, %173, %175, %177 : vector<1x4xf16>
          %179 = frisk.warp_mma_rr %178, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %179, %168 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %131 = affine.apply #map18(%arg11)
        %132 = vector.extract %49[%131, 4] : f16 from vector<2x8xf16>
        %133 = vector.extract %49[%131, 5] : f16 from vector<2x8xf16>
        %134 = vector.extract %49[%131, 6] : f16 from vector<2x8xf16>
        %135 = vector.extract %49[%131, 7] : f16 from vector<2x8xf16>
        %136 = vector.from_elements %132, %133, %134, %135 : vector<1x4xf16>
        %137 = frisk.warp_mma_rr %136, %130#1, %130#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %138 = affine.apply #map17(%arg11)
        %139 = vector.extract %98#1[%131, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply #map19(%arg11)
        %141 = vector.extract %98#1[%131, %140] : f32 from vector<2x32xf32>
        %142 = affine.apply #map20(%arg11)
        %143 = vector.extract %98#1[%131, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply #map21(%arg11)
        %145 = vector.extract %98#1[%131, %144] : f32 from vector<2x32xf32>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<1x4xf32>
        %147 = arith.addf %146, %137 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %148 = vector.extract %147[0, 0] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %arg12 [%131, %138] : f32 into vector<2x32xf32>
        %150 = vector.extract %147[0, 1] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%131, %140] : f32 into vector<2x32xf32>
        %152 = vector.extract %147[0, 2] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %151 [%131, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %147[0, 3] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%131, %144] : f32 into vector<2x32xf32>
        %156 = vector.insert %148, %arg13 [%131, %138] : f32 into vector<2x32xf32>
        %157 = vector.insert %150, %156 [%131, %140] : f32 into vector<2x32xf32>
        %158 = vector.insert %152, %157 [%131, %142] : f32 into vector<2x32xf32>
        %159 = vector.insert %154, %158 [%131, %144] : f32 into vector<2x32xf32>
        affine.yield %155, %159 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %101 = affine.apply #map33(%arg4)[%block_id_z, %7]
      %subview_20 = memref.subview %transpose[0, %101, 64, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_20 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_19 : memref<32x128xf16, 3>
      %102:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %124 = affine.apply #map23(%arg11)[%thread_id_x]
        %125 = memref.load %alloc_19[%15, %124] : memref<32x128xf16, 3>
        %126 = memref.load %alloc_19[%17, %124] : memref<32x128xf16, 3>
        %127 = memref.load %alloc_19[%19, %124] : memref<32x128xf16, 3>
        %128 = memref.load %alloc_19[%21, %124] : memref<32x128xf16, 3>
        %129 = vector.from_elements %125, %126, %127, %128 : vector<4x1xf16>
        %130:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %129) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %160 = affine.apply #map24(%arg14)[%thread_id_x]
          %161 = memref.load %alloc_19[%160, %124] : memref<32x128xf16, 3>
          %162 = affine.apply #map25(%arg14)[%thread_id_x]
          %163 = memref.load %alloc_19[%162, %124] : memref<32x128xf16, 3>
          %164 = affine.apply #map26(%arg14)[%thread_id_x]
          %165 = memref.load %alloc_19[%164, %124] : memref<32x128xf16, 3>
          %166 = affine.apply #map27(%arg14)[%thread_id_x]
          %167 = memref.load %alloc_19[%166, %124] : memref<32x128xf16, 3>
          %168 = vector.from_elements %161, %163, %165, %167 : vector<4x1xf16>
          %169 = affine.apply #map18(%arg11)
          %170 = affine.apply #map28(%arg14)
          %171 = vector.extract %64[%169, %170] : f16 from vector<2x8xf16>
          %172 = affine.apply #map29(%arg14)
          %173 = vector.extract %64[%169, %172] : f16 from vector<2x8xf16>
          %174 = affine.apply #map30(%arg14)
          %175 = vector.extract %64[%169, %174] : f16 from vector<2x8xf16>
          %176 = affine.apply #map31(%arg14)
          %177 = vector.extract %64[%169, %176] : f16 from vector<2x8xf16>
          %178 = vector.from_elements %171, %173, %175, %177 : vector<1x4xf16>
          %179 = frisk.warp_mma_rr %178, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %179, %168 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %131 = affine.apply #map18(%arg11)
        %132 = vector.extract %64[%131, 4] : f16 from vector<2x8xf16>
        %133 = vector.extract %64[%131, 5] : f16 from vector<2x8xf16>
        %134 = vector.extract %64[%131, 6] : f16 from vector<2x8xf16>
        %135 = vector.extract %64[%131, 7] : f16 from vector<2x8xf16>
        %136 = vector.from_elements %132, %133, %134, %135 : vector<1x4xf16>
        %137 = frisk.warp_mma_rr %136, %130#1, %130#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %138 = affine.apply #map17(%arg11)
        %139 = vector.extract %100#1[%131, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply #map19(%arg11)
        %141 = vector.extract %100#1[%131, %140] : f32 from vector<2x32xf32>
        %142 = affine.apply #map20(%arg11)
        %143 = vector.extract %100#1[%131, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply #map21(%arg11)
        %145 = vector.extract %100#1[%131, %144] : f32 from vector<2x32xf32>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<1x4xf32>
        %147 = arith.addf %146, %137 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %148 = vector.extract %147[0, 0] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %arg12 [%131, %138] : f32 into vector<2x32xf32>
        %150 = vector.extract %147[0, 1] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%131, %140] : f32 into vector<2x32xf32>
        %152 = vector.extract %147[0, 2] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %151 [%131, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %147[0, 3] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%131, %144] : f32 into vector<2x32xf32>
        %156 = vector.insert %148, %arg13 [%131, %138] : f32 into vector<2x32xf32>
        %157 = vector.insert %150, %156 [%131, %140] : f32 into vector<2x32xf32>
        %158 = vector.insert %152, %157 [%131, %142] : f32 into vector<2x32xf32>
        %159 = vector.insert %154, %158 [%131, %144] : f32 into vector<2x32xf32>
        affine.yield %155, %159 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %103 = affine.apply #map34(%arg4)[%block_id_z, %7]
      %subview_22 = memref.subview %transpose[0, %103, 96, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_22 to %alloc_21 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_21 : memref<32x128xf16, 3>
      %104:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %124 = affine.apply #map23(%arg11)[%thread_id_x]
        %125 = memref.load %alloc_21[%15, %124] : memref<32x128xf16, 3>
        %126 = memref.load %alloc_21[%17, %124] : memref<32x128xf16, 3>
        %127 = memref.load %alloc_21[%19, %124] : memref<32x128xf16, 3>
        %128 = memref.load %alloc_21[%21, %124] : memref<32x128xf16, 3>
        %129 = vector.from_elements %125, %126, %127, %128 : vector<4x1xf16>
        %130:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %129) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %160 = affine.apply #map24(%arg14)[%thread_id_x]
          %161 = memref.load %alloc_21[%160, %124] : memref<32x128xf16, 3>
          %162 = affine.apply #map25(%arg14)[%thread_id_x]
          %163 = memref.load %alloc_21[%162, %124] : memref<32x128xf16, 3>
          %164 = affine.apply #map26(%arg14)[%thread_id_x]
          %165 = memref.load %alloc_21[%164, %124] : memref<32x128xf16, 3>
          %166 = affine.apply #map27(%arg14)[%thread_id_x]
          %167 = memref.load %alloc_21[%166, %124] : memref<32x128xf16, 3>
          %168 = vector.from_elements %161, %163, %165, %167 : vector<4x1xf16>
          %169 = affine.apply #map18(%arg11)
          %170 = affine.apply #map28(%arg14)
          %171 = vector.extract %79[%169, %170] : f16 from vector<2x8xf16>
          %172 = affine.apply #map29(%arg14)
          %173 = vector.extract %79[%169, %172] : f16 from vector<2x8xf16>
          %174 = affine.apply #map30(%arg14)
          %175 = vector.extract %79[%169, %174] : f16 from vector<2x8xf16>
          %176 = affine.apply #map31(%arg14)
          %177 = vector.extract %79[%169, %176] : f16 from vector<2x8xf16>
          %178 = vector.from_elements %171, %173, %175, %177 : vector<1x4xf16>
          %179 = frisk.warp_mma_rr %178, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %179, %168 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %131 = affine.apply #map18(%arg11)
        %132 = vector.extract %79[%131, 4] : f16 from vector<2x8xf16>
        %133 = vector.extract %79[%131, 5] : f16 from vector<2x8xf16>
        %134 = vector.extract %79[%131, 6] : f16 from vector<2x8xf16>
        %135 = vector.extract %79[%131, 7] : f16 from vector<2x8xf16>
        %136 = vector.from_elements %132, %133, %134, %135 : vector<1x4xf16>
        %137 = frisk.warp_mma_rr %136, %130#1, %130#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %138 = affine.apply #map17(%arg11)
        %139 = vector.extract %102#1[%131, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply #map19(%arg11)
        %141 = vector.extract %102#1[%131, %140] : f32 from vector<2x32xf32>
        %142 = affine.apply #map20(%arg11)
        %143 = vector.extract %102#1[%131, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply #map21(%arg11)
        %145 = vector.extract %102#1[%131, %144] : f32 from vector<2x32xf32>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<1x4xf32>
        %147 = arith.addf %146, %137 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %148 = vector.extract %147[0, 0] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %arg12 [%131, %138] : f32 into vector<2x32xf32>
        %150 = vector.extract %147[0, 1] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%131, %140] : f32 into vector<2x32xf32>
        %152 = vector.extract %147[0, 2] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %151 [%131, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %147[0, 3] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%131, %144] : f32 into vector<2x32xf32>
        %156 = vector.insert %148, %arg13 [%131, %138] : f32 into vector<2x32xf32>
        %157 = vector.insert %150, %156 [%131, %140] : f32 into vector<2x32xf32>
        %158 = vector.insert %152, %157 [%131, %142] : f32 into vector<2x32xf32>
        %159 = vector.insert %154, %158 [%131, %144] : f32 into vector<2x32xf32>
        affine.yield %155, %159 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %105 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %124 = affine.apply #map18(%arg11)
        %125 = affine.apply #map17(%arg11)
        %126 = vector.extract %104#1[%124, %125] : f32 from vector<2x32xf32>
        %127 = affine.apply #map19(%arg11)
        %128 = vector.extract %104#1[%124, %127] : f32 from vector<2x32xf32>
        %129 = affine.apply #map20(%arg11)
        %130 = vector.extract %104#1[%124, %129] : f32 from vector<2x32xf32>
        %131 = affine.apply #map21(%arg11)
        %132 = vector.extract %104#1[%124, %131] : f32 from vector<2x32xf32>
        %133 = vector.insert %126, %arg12 [%124, %125] : f32 into vector<2x32xf32>
        %134 = vector.insert %128, %133 [%124, %127] : f32 into vector<2x32xf32>
        %135 = vector.insert %130, %134 [%124, %129] : f32 into vector<2x32xf32>
        %136 = vector.insert %132, %135 [%124, %131] : f32 into vector<2x32xf32>
        affine.yield %136 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %106 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %124 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_7) -> (f32) {
          %128 = affine.apply #map28(%arg13)
          %129 = vector.extract %105[%arg11, %128] : f32 from vector<2x32xf32>
          %130 = arith.maxnumf %arg14, %129 : f32
          %131 = affine.apply #map29(%arg13)
          %132 = vector.extract %105[%arg11, %131] : f32 from vector<2x32xf32>
          %133 = arith.maxnumf %130, %132 : f32
          %134 = affine.apply #map30(%arg13)
          %135 = vector.extract %105[%arg11, %134] : f32 from vector<2x32xf32>
          %136 = arith.maxnumf %133, %135 : f32
          %137 = affine.apply #map31(%arg13)
          %138 = vector.extract %105[%arg11, %137] : f32 from vector<2x32xf32>
          %139 = arith.maxnumf %136, %138 : f32
          affine.yield %139 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %124, %c16_i32, %c64_i32 : f32
        %125 = arith.maxnumf %124, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %125, %c32_i32, %c64_i32 : f32
        %126 = arith.maxnumf %125, %shuffleResult_31 : f32
        %127 = vector.insert %126, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %127 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %107 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %124 = vector.extract %106[%arg11, 0] : f32 from vector<2x1xf32>
        %125 = vector.broadcast %124 : f32 to vector<1x1xf32>
        %126 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %127 = vector.broadcast %126 : f32 to vector<1x1xf32>
        %128 = arith.maxnumf %125, %127 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %129 = vector.extract %128[0, 0] : f32 from vector<1x1xf32>
        %130 = vector.insert %129, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %130 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %108:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %124 = affine.apply #map18(%arg11)
        %125 = vector.extract %107[%124, 0] : f32 from vector<2x1xf32>
        %126 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3) -> (vector<1x4xf32>) {
          %153 = vector.broadcast %125 : f32 to vector<1x4xf32>
          affine.yield %153 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %127 = affine.apply #map17(%arg11)
        %128 = vector.extract %126[0, 0] : f32 from vector<1x4xf32>
        %129 = vector.insert %128, %arg12 [%124, %127] : f32 into vector<2x32xf32>
        %130 = affine.apply #map19(%arg11)
        %131 = vector.extract %126[0, 1] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %129 [%124, %130] : f32 into vector<2x32xf32>
        %133 = affine.apply #map20(%arg11)
        %134 = vector.extract %126[0, 2] : f32 from vector<1x4xf32>
        %135 = vector.insert %134, %132 [%124, %133] : f32 into vector<2x32xf32>
        %136 = affine.apply #map21(%arg11)
        %137 = vector.extract %126[0, 3] : f32 from vector<1x4xf32>
        %138 = vector.insert %137, %135 [%124, %136] : f32 into vector<2x32xf32>
        %139 = vector.extract %104#1[%124, %127] : f32 from vector<2x32xf32>
        %140 = vector.extract %104#1[%124, %130] : f32 from vector<2x32xf32>
        %141 = vector.extract %104#1[%124, %133] : f32 from vector<2x32xf32>
        %142 = vector.extract %104#1[%124, %136] : f32 from vector<2x32xf32>
        %143 = vector.from_elements %139, %140, %141, %142 : vector<1x4xf32>
        %144 = arith.subf %143, %126 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %145 = vector.extract %144[0, 0] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %arg13 [%124, %127] : f32 into vector<2x32xf32>
        %147 = vector.extract %144[0, 1] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%124, %130] : f32 into vector<2x32xf32>
        %149 = vector.extract %144[0, 2] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%124, %133] : f32 into vector<2x32xf32>
        %151 = vector.extract %144[0, 3] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%124, %136] : f32 into vector<2x32xf32>
        affine.yield %138, %152 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %109:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6, %arg14 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %124 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %125 = vector.broadcast %124 : f32 to vector<1x1xf32>
        %126 = vector.extract %107[%arg11, 0] : f32 from vector<2x1xf32>
        %127 = vector.broadcast %126 : f32 to vector<1x1xf32>
        %128 = arith.subf %125, %127 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %129 = vector.extract %128[0, 0] : f32 from vector<1x1xf32>
        %130 = vector.insert %129, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %131 = math.exp2 %128 {tiled = true} : vector<1x1xf32>
        %132 = vector.extract %131[0, 0] : f32 from vector<1x1xf32>
        %133 = vector.insert %132, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %134 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %135 = vector.broadcast %134 : f32 to vector<1x1xf32>
        %136 = arith.mulf %135, %131 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %137 = vector.extract %136[0, 0] : f32 from vector<1x1xf32>
        %138 = vector.insert %137, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %130, %133, %138 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %110:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %124 = affine.apply #map18(%arg11)
        %125 = affine.apply #map17(%arg11)
        %126 = vector.extract %108#1[%124, %125] : f32 from vector<2x32xf32>
        %127 = affine.apply #map19(%arg11)
        %128 = vector.extract %108#1[%124, %127] : f32 from vector<2x32xf32>
        %129 = affine.apply #map20(%arg11)
        %130 = vector.extract %108#1[%124, %129] : f32 from vector<2x32xf32>
        %131 = affine.apply #map21(%arg11)
        %132 = vector.extract %108#1[%124, %131] : f32 from vector<2x32xf32>
        %133 = vector.from_elements %126, %128, %130, %132 : vector<1x4xf32>
        %134 = math.exp2 %133 {tiled = true} : vector<1x4xf32>
        %135 = vector.extract %134[0, 0] : f32 from vector<1x4xf32>
        %136 = vector.insert %135, %arg12 [%124, %125] : f32 into vector<2x32xf32>
        %137 = vector.extract %134[0, 1] : f32 from vector<1x4xf32>
        %138 = vector.insert %137, %136 [%124, %127] : f32 into vector<2x32xf32>
        %139 = vector.extract %134[0, 2] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %138 [%124, %129] : f32 into vector<2x32xf32>
        %141 = vector.extract %134[0, 3] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %140 [%124, %131] : f32 into vector<2x32xf32>
        %143 = vector.insert %135, %arg13 [%124, %125] : f32 into vector<2x32xf32>
        %144 = vector.insert %137, %143 [%124, %127] : f32 into vector<2x32xf32>
        %145 = vector.insert %139, %144 [%124, %129] : f32 into vector<2x32xf32>
        %146 = vector.insert %141, %145 [%124, %131] : f32 into vector<2x32xf32>
        affine.yield %142, %146 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %111 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %124 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst) -> (f32) {
          %128 = affine.apply #map28(%arg13)
          %129 = vector.extract %110#1[%arg11, %128] : f32 from vector<2x32xf32>
          %130 = arith.addf %arg14, %129 : f32
          %131 = affine.apply #map29(%arg13)
          %132 = vector.extract %110#1[%arg11, %131] : f32 from vector<2x32xf32>
          %133 = arith.addf %130, %132 : f32
          %134 = affine.apply #map30(%arg13)
          %135 = vector.extract %110#1[%arg11, %134] : f32 from vector<2x32xf32>
          %136 = arith.addf %133, %135 : f32
          %137 = affine.apply #map31(%arg13)
          %138 = vector.extract %110#1[%arg11, %137] : f32 from vector<2x32xf32>
          %139 = arith.addf %136, %138 : f32
          affine.yield %139 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %124, %c16_i32, %c64_i32 : f32
        %125 = arith.addf %124, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %125, %c32_i32, %c64_i32 : f32
        %126 = arith.addf %125, %shuffleResult_31 : f32
        %127 = vector.insert %126, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %127 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %112 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %124 = vector.extract %111[%arg11, 0] : f32 from vector<2x1xf32>
        %125 = vector.broadcast %124 : f32 to vector<1x1xf32>
        %126 = vector.extract %109#2[%arg11, 0] : f32 from vector<2x1xf32>
        %127 = vector.broadcast %126 : f32 to vector<1x1xf32>
        %128 = arith.addf %125, %127 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %129 = vector.extract %128[0, 0] : f32 from vector<1x1xf32>
        %130 = vector.insert %129, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %130 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %113 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %124 = affine.apply #map18(%arg11)
        %125 = affine.apply #map17(%arg11)
        %126 = vector.extract %110#0[%124, %125] : f32 from vector<2x32xf32>
        %127 = affine.apply #map19(%arg11)
        %128 = vector.extract %110#0[%124, %127] : f32 from vector<2x32xf32>
        %129 = affine.apply #map20(%arg11)
        %130 = vector.extract %110#0[%124, %129] : f32 from vector<2x32xf32>
        %131 = affine.apply #map21(%arg11)
        %132 = vector.extract %110#0[%124, %131] : f32 from vector<2x32xf32>
        %133 = vector.from_elements %126, %128, %130, %132 : vector<1x4xf32>
        %134 = arith.truncf %133 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
        %136 = vector.insert %135, %arg12 [%124, %125] : f16 into vector<2x32xf16>
        %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
        %138 = vector.insert %137, %136 [%124, %127] : f16 into vector<2x32xf16>
        %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
        %140 = vector.insert %139, %138 [%124, %129] : f16 into vector<2x32xf16>
        %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
        %142 = vector.insert %141, %140 [%124, %131] : f16 into vector<2x32xf16>
        affine.yield %142 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_24 = memref.subview %arg2[0, %97, %arg4, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_24 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_23 : memref<128x32xf16, 3>
      %114 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %124 = affine.apply #map35(%arg11)[%thread_id_x]
        %125 = affine.apply #map36()[%thread_id_x]
        %126 = frisk.read_matrix_from_shm %alloc_23[%124, %125] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %127 = vector.bitcast %126 : vector<4xi32> to vector<8xf16>
        %128 = vector.extract %127[0] : f16 from vector<8xf16>
        %129 = affine.apply #map37()[%thread_id_x]
        %130 = vector.extract %127[%129] : f16 from vector<8xf16>
        %131 = vector.extract %127[1] : f16 from vector<8xf16>
        %132 = affine.apply #map38()[%thread_id_x]
        %133 = vector.extract %127[%132] : f16 from vector<8xf16>
        %134 = vector.extract %127[2] : f16 from vector<8xf16>
        %135 = affine.apply #map39()[%thread_id_x]
        %136 = vector.extract %127[%135] : f16 from vector<8xf16>
        %137 = vector.extract %127[3] : f16 from vector<8xf16>
        %138 = affine.apply #map40()[%thread_id_x]
        %139 = vector.extract %127[%138] : f16 from vector<8xf16>
        %140 = affine.apply #map28(%arg11)
        %141 = vector.insert %128, %arg12 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %130, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply #map29(%arg11)
        %144 = vector.insert %131, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply #map30(%arg11)
        %147 = vector.insert %134, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %136, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply #map31(%arg11)
        %150 = vector.insert %137, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        affine.yield %151 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %115:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %124 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %171 = affine.apply #map1(%arg11)
          %172 = affine.apply #map28(%arg16)
          %173 = vector.extract %113[%171, %172] : f16 from vector<2x32xf16>
          %174 = affine.apply #map29(%arg16)
          %175 = vector.extract %113[%171, %174] : f16 from vector<2x32xf16>
          %176 = affine.apply #map30(%arg16)
          %177 = vector.extract %113[%171, %176] : f16 from vector<2x32xf16>
          %178 = affine.apply #map31(%arg16)
          %179 = vector.extract %113[%171, %178] : f16 from vector<2x32xf16>
          %180 = vector.from_elements %173, %175, %177, %179 : vector<1x4xf16>
          %181 = affine.apply #map41(%arg11)
          %182 = vector.extract %114[%172, %181] : f16 from vector<32x2xf16>
          %183 = vector.extract %114[%174, %181] : f16 from vector<32x2xf16>
          %184 = vector.extract %114[%176, %181] : f16 from vector<32x2xf16>
          %185 = vector.extract %114[%178, %181] : f16 from vector<32x2xf16>
          %186 = vector.from_elements %182, %183, %184, %185 : vector<4x1xf16>
          %187 = frisk.warp_mma_rr %180, %186, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %187 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %125 = affine.apply #map1(%arg11)
        %126 = affine.apply #map(%arg11)
        %127 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
        %128 = vector.insert %127, %arg12 [%125, %126] : f32 into vector<2x8xf32>
        %129 = affine.apply #map2(%arg11)
        %130 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %128 [%125, %129] : f32 into vector<2x8xf32>
        %132 = affine.apply #map3(%arg11)
        %133 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
        %134 = vector.insert %133, %131 [%125, %132] : f32 into vector<2x8xf32>
        %135 = affine.apply #map4(%arg11)
        %136 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %134 [%125, %135] : f32 into vector<2x8xf32>
        %138 = vector.extract %109#1[%125, 0] : f32 from vector<2x1xf32>
        %139 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %171 = vector.broadcast %138 : f32 to vector<1x4xf32>
          affine.yield %171 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %140 = vector.extract %139[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg13 [%125, %126] : f32 into vector<2x8xf32>
        %142 = vector.extract %139[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %141 [%125, %129] : f32 into vector<2x8xf32>
        %144 = vector.extract %139[0, 2] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%125, %132] : f32 into vector<2x8xf32>
        %146 = vector.extract %139[0, 3] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%125, %135] : f32 into vector<2x8xf32>
        %148 = vector.extract %arg5[%125, %126] : f32 from vector<2x8xf32>
        %149 = vector.extract %arg5[%125, %129] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg5[%125, %132] : f32 from vector<2x8xf32>
        %151 = vector.extract %arg5[%125, %135] : f32 from vector<2x8xf32>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf32>
        %153 = arith.mulf %152, %139 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %arg14 [%125, %126] : f32 into vector<2x8xf32>
        %156 = vector.extract %153[0, 1] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%125, %129] : f32 into vector<2x8xf32>
        %158 = vector.extract %153[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%125, %132] : f32 into vector<2x8xf32>
        %160 = vector.extract %153[0, 3] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%125, %135] : f32 into vector<2x8xf32>
        %162 = arith.addf %153, %124 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %arg15 [%125, %126] : f32 into vector<2x8xf32>
        %165 = vector.extract %162[0, 1] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%125, %129] : f32 into vector<2x8xf32>
        %167 = vector.extract %162[0, 2] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%125, %132] : f32 into vector<2x8xf32>
        %169 = vector.extract %162[0, 3] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%125, %135] : f32 into vector<2x8xf32>
        affine.yield %137, %147, %161, %170 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_25 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_26 = memref.subview %arg2[0, %97, %arg4, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_26 to %alloc_25 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_25 : memref<128x32xf16, 3>
      %116 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %124 = affine.apply #map35(%arg11)[%thread_id_x]
        %125 = affine.apply #map36()[%thread_id_x]
        %126 = frisk.read_matrix_from_shm %alloc_25[%124, %125] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %127 = vector.bitcast %126 : vector<4xi32> to vector<8xf16>
        %128 = vector.extract %127[0] : f16 from vector<8xf16>
        %129 = affine.apply #map37()[%thread_id_x]
        %130 = vector.extract %127[%129] : f16 from vector<8xf16>
        %131 = vector.extract %127[1] : f16 from vector<8xf16>
        %132 = affine.apply #map38()[%thread_id_x]
        %133 = vector.extract %127[%132] : f16 from vector<8xf16>
        %134 = vector.extract %127[2] : f16 from vector<8xf16>
        %135 = affine.apply #map39()[%thread_id_x]
        %136 = vector.extract %127[%135] : f16 from vector<8xf16>
        %137 = vector.extract %127[3] : f16 from vector<8xf16>
        %138 = affine.apply #map40()[%thread_id_x]
        %139 = vector.extract %127[%138] : f16 from vector<8xf16>
        %140 = affine.apply #map28(%arg11)
        %141 = vector.insert %128, %arg12 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %130, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply #map29(%arg11)
        %144 = vector.insert %131, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply #map30(%arg11)
        %147 = vector.insert %134, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %136, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply #map31(%arg11)
        %150 = vector.insert %137, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        affine.yield %151 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %117:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %124 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %171 = affine.apply #map1(%arg11)
          %172 = affine.apply #map28(%arg16)
          %173 = vector.extract %113[%171, %172] : f16 from vector<2x32xf16>
          %174 = affine.apply #map29(%arg16)
          %175 = vector.extract %113[%171, %174] : f16 from vector<2x32xf16>
          %176 = affine.apply #map30(%arg16)
          %177 = vector.extract %113[%171, %176] : f16 from vector<2x32xf16>
          %178 = affine.apply #map31(%arg16)
          %179 = vector.extract %113[%171, %178] : f16 from vector<2x32xf16>
          %180 = vector.from_elements %173, %175, %177, %179 : vector<1x4xf16>
          %181 = affine.apply #map41(%arg11)
          %182 = vector.extract %116[%172, %181] : f16 from vector<32x2xf16>
          %183 = vector.extract %116[%174, %181] : f16 from vector<32x2xf16>
          %184 = vector.extract %116[%176, %181] : f16 from vector<32x2xf16>
          %185 = vector.extract %116[%178, %181] : f16 from vector<32x2xf16>
          %186 = vector.from_elements %182, %183, %184, %185 : vector<4x1xf16>
          %187 = frisk.warp_mma_rr %180, %186, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %187 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %125 = affine.apply #map1(%arg11)
        %126 = affine.apply #map(%arg11)
        %127 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
        %128 = vector.insert %127, %arg12 [%125, %126] : f32 into vector<2x8xf32>
        %129 = affine.apply #map2(%arg11)
        %130 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %128 [%125, %129] : f32 into vector<2x8xf32>
        %132 = affine.apply #map3(%arg11)
        %133 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
        %134 = vector.insert %133, %131 [%125, %132] : f32 into vector<2x8xf32>
        %135 = affine.apply #map4(%arg11)
        %136 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %134 [%125, %135] : f32 into vector<2x8xf32>
        %138 = vector.extract %109#1[%125, 0] : f32 from vector<2x1xf32>
        %139 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %171 = vector.broadcast %138 : f32 to vector<1x4xf32>
          affine.yield %171 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %140 = vector.extract %139[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg13 [%125, %126] : f32 into vector<2x8xf32>
        %142 = vector.extract %139[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %141 [%125, %129] : f32 into vector<2x8xf32>
        %144 = vector.extract %139[0, 2] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%125, %132] : f32 into vector<2x8xf32>
        %146 = vector.extract %139[0, 3] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%125, %135] : f32 into vector<2x8xf32>
        %148 = vector.extract %arg8[%125, %126] : f32 from vector<2x8xf32>
        %149 = vector.extract %arg8[%125, %129] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg8[%125, %132] : f32 from vector<2x8xf32>
        %151 = vector.extract %arg8[%125, %135] : f32 from vector<2x8xf32>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf32>
        %153 = arith.mulf %152, %139 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %arg14 [%125, %126] : f32 into vector<2x8xf32>
        %156 = vector.extract %153[0, 1] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%125, %129] : f32 into vector<2x8xf32>
        %158 = vector.extract %153[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%125, %132] : f32 into vector<2x8xf32>
        %160 = vector.extract %153[0, 3] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%125, %135] : f32 into vector<2x8xf32>
        %162 = arith.addf %153, %124 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %arg15 [%125, %126] : f32 into vector<2x8xf32>
        %165 = vector.extract %162[0, 1] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%125, %129] : f32 into vector<2x8xf32>
        %167 = vector.extract %162[0, 2] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%125, %132] : f32 into vector<2x8xf32>
        %169 = vector.extract %162[0, 3] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%125, %135] : f32 into vector<2x8xf32>
        affine.yield %137, %147, %161, %170 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_27 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_28 = memref.subview %arg2[0, %97, %arg4, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_28 to %alloc_27 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_27 : memref<128x32xf16, 3>
      %118 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %124 = affine.apply #map35(%arg11)[%thread_id_x]
        %125 = affine.apply #map36()[%thread_id_x]
        %126 = frisk.read_matrix_from_shm %alloc_27[%124, %125] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %127 = vector.bitcast %126 : vector<4xi32> to vector<8xf16>
        %128 = vector.extract %127[0] : f16 from vector<8xf16>
        %129 = affine.apply #map37()[%thread_id_x]
        %130 = vector.extract %127[%129] : f16 from vector<8xf16>
        %131 = vector.extract %127[1] : f16 from vector<8xf16>
        %132 = affine.apply #map38()[%thread_id_x]
        %133 = vector.extract %127[%132] : f16 from vector<8xf16>
        %134 = vector.extract %127[2] : f16 from vector<8xf16>
        %135 = affine.apply #map39()[%thread_id_x]
        %136 = vector.extract %127[%135] : f16 from vector<8xf16>
        %137 = vector.extract %127[3] : f16 from vector<8xf16>
        %138 = affine.apply #map40()[%thread_id_x]
        %139 = vector.extract %127[%138] : f16 from vector<8xf16>
        %140 = affine.apply #map28(%arg11)
        %141 = vector.insert %128, %arg12 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %130, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply #map29(%arg11)
        %144 = vector.insert %131, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply #map30(%arg11)
        %147 = vector.insert %134, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %136, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply #map31(%arg11)
        %150 = vector.insert %137, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        affine.yield %151 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %119:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %124 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %171 = affine.apply #map1(%arg11)
          %172 = affine.apply #map28(%arg16)
          %173 = vector.extract %113[%171, %172] : f16 from vector<2x32xf16>
          %174 = affine.apply #map29(%arg16)
          %175 = vector.extract %113[%171, %174] : f16 from vector<2x32xf16>
          %176 = affine.apply #map30(%arg16)
          %177 = vector.extract %113[%171, %176] : f16 from vector<2x32xf16>
          %178 = affine.apply #map31(%arg16)
          %179 = vector.extract %113[%171, %178] : f16 from vector<2x32xf16>
          %180 = vector.from_elements %173, %175, %177, %179 : vector<1x4xf16>
          %181 = affine.apply #map41(%arg11)
          %182 = vector.extract %118[%172, %181] : f16 from vector<32x2xf16>
          %183 = vector.extract %118[%174, %181] : f16 from vector<32x2xf16>
          %184 = vector.extract %118[%176, %181] : f16 from vector<32x2xf16>
          %185 = vector.extract %118[%178, %181] : f16 from vector<32x2xf16>
          %186 = vector.from_elements %182, %183, %184, %185 : vector<4x1xf16>
          %187 = frisk.warp_mma_rr %180, %186, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %187 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %125 = affine.apply #map1(%arg11)
        %126 = affine.apply #map(%arg11)
        %127 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
        %128 = vector.insert %127, %arg12 [%125, %126] : f32 into vector<2x8xf32>
        %129 = affine.apply #map2(%arg11)
        %130 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %128 [%125, %129] : f32 into vector<2x8xf32>
        %132 = affine.apply #map3(%arg11)
        %133 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
        %134 = vector.insert %133, %131 [%125, %132] : f32 into vector<2x8xf32>
        %135 = affine.apply #map4(%arg11)
        %136 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %134 [%125, %135] : f32 into vector<2x8xf32>
        %138 = vector.extract %109#1[%125, 0] : f32 from vector<2x1xf32>
        %139 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %171 = vector.broadcast %138 : f32 to vector<1x4xf32>
          affine.yield %171 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %140 = vector.extract %139[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg13 [%125, %126] : f32 into vector<2x8xf32>
        %142 = vector.extract %139[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %141 [%125, %129] : f32 into vector<2x8xf32>
        %144 = vector.extract %139[0, 2] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%125, %132] : f32 into vector<2x8xf32>
        %146 = vector.extract %139[0, 3] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%125, %135] : f32 into vector<2x8xf32>
        %148 = vector.extract %arg9[%125, %126] : f32 from vector<2x8xf32>
        %149 = vector.extract %arg9[%125, %129] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg9[%125, %132] : f32 from vector<2x8xf32>
        %151 = vector.extract %arg9[%125, %135] : f32 from vector<2x8xf32>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf32>
        %153 = arith.mulf %152, %139 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %arg14 [%125, %126] : f32 into vector<2x8xf32>
        %156 = vector.extract %153[0, 1] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%125, %129] : f32 into vector<2x8xf32>
        %158 = vector.extract %153[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%125, %132] : f32 into vector<2x8xf32>
        %160 = vector.extract %153[0, 3] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%125, %135] : f32 into vector<2x8xf32>
        %162 = arith.addf %153, %124 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %arg15 [%125, %126] : f32 into vector<2x8xf32>
        %165 = vector.extract %162[0, 1] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%125, %129] : f32 into vector<2x8xf32>
        %167 = vector.extract %162[0, 2] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%125, %132] : f32 into vector<2x8xf32>
        %169 = vector.extract %162[0, 3] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%125, %135] : f32 into vector<2x8xf32>
        affine.yield %137, %147, %161, %170 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_29 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_30 = memref.subview %arg2[0, %97, %arg4, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_30 to %alloc_29 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_29 : memref<128x32xf16, 3>
      %120 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %124 = affine.apply #map35(%arg11)[%thread_id_x]
        %125 = affine.apply #map36()[%thread_id_x]
        %126 = frisk.read_matrix_from_shm %alloc_29[%124, %125] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %127 = vector.bitcast %126 : vector<4xi32> to vector<8xf16>
        %128 = vector.extract %127[0] : f16 from vector<8xf16>
        %129 = affine.apply #map37()[%thread_id_x]
        %130 = vector.extract %127[%129] : f16 from vector<8xf16>
        %131 = vector.extract %127[1] : f16 from vector<8xf16>
        %132 = affine.apply #map38()[%thread_id_x]
        %133 = vector.extract %127[%132] : f16 from vector<8xf16>
        %134 = vector.extract %127[2] : f16 from vector<8xf16>
        %135 = affine.apply #map39()[%thread_id_x]
        %136 = vector.extract %127[%135] : f16 from vector<8xf16>
        %137 = vector.extract %127[3] : f16 from vector<8xf16>
        %138 = affine.apply #map40()[%thread_id_x]
        %139 = vector.extract %127[%138] : f16 from vector<8xf16>
        %140 = affine.apply #map28(%arg11)
        %141 = vector.insert %128, %arg12 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %130, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply #map29(%arg11)
        %144 = vector.insert %131, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply #map30(%arg11)
        %147 = vector.insert %134, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %136, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply #map31(%arg11)
        %150 = vector.insert %137, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        affine.yield %151 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %121:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %124 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %179 = affine.apply #map1(%arg11)
          %180 = affine.apply #map28(%arg17)
          %181 = vector.extract %113[%179, %180] : f16 from vector<2x32xf16>
          %182 = affine.apply #map29(%arg17)
          %183 = vector.extract %113[%179, %182] : f16 from vector<2x32xf16>
          %184 = affine.apply #map30(%arg17)
          %185 = vector.extract %113[%179, %184] : f16 from vector<2x32xf16>
          %186 = affine.apply #map31(%arg17)
          %187 = vector.extract %113[%179, %186] : f16 from vector<2x32xf16>
          %188 = vector.from_elements %181, %183, %185, %187 : vector<1x4xf16>
          %189 = affine.apply #map41(%arg11)
          %190 = vector.extract %120[%180, %189] : f16 from vector<32x2xf16>
          %191 = vector.extract %120[%182, %189] : f16 from vector<32x2xf16>
          %192 = vector.extract %120[%184, %189] : f16 from vector<32x2xf16>
          %193 = vector.extract %120[%186, %189] : f16 from vector<32x2xf16>
          %194 = vector.from_elements %190, %191, %192, %193 : vector<4x1xf16>
          %195 = frisk.warp_mma_rr %188, %194, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %195 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %125 = affine.apply #map1(%arg11)
        %126 = affine.apply #map(%arg11)
        %127 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
        %128 = vector.insert %127, %arg12 [%125, %126] : f32 into vector<2x8xf32>
        %129 = affine.apply #map2(%arg11)
        %130 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %128 [%125, %129] : f32 into vector<2x8xf32>
        %132 = affine.apply #map3(%arg11)
        %133 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
        %134 = vector.insert %133, %131 [%125, %132] : f32 into vector<2x8xf32>
        %135 = affine.apply #map4(%arg11)
        %136 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %134 [%125, %135] : f32 into vector<2x8xf32>
        %138 = vector.extract %109#1[%125, 0] : f32 from vector<2x1xf32>
        %139 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %179 = vector.broadcast %138 : f32 to vector<1x4xf32>
          affine.yield %179 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %140 = vector.extract %139[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg13 [%125, %126] : f32 into vector<2x8xf32>
        %142 = vector.extract %139[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %141 [%125, %129] : f32 into vector<2x8xf32>
        %144 = vector.extract %139[0, 2] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%125, %132] : f32 into vector<2x8xf32>
        %146 = vector.extract %139[0, 3] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%125, %135] : f32 into vector<2x8xf32>
        %148 = vector.extract %arg10[%125, %126] : f32 from vector<2x8xf32>
        %149 = vector.extract %arg10[%125, %129] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg10[%125, %132] : f32 from vector<2x8xf32>
        %151 = vector.extract %arg10[%125, %135] : f32 from vector<2x8xf32>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf32>
        %153 = arith.mulf %152, %139 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %arg14 [%125, %126] : f32 into vector<2x8xf32>
        %156 = vector.extract %153[0, 1] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%125, %129] : f32 into vector<2x8xf32>
        %158 = vector.extract %153[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%125, %132] : f32 into vector<2x8xf32>
        %160 = vector.extract %153[0, 3] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%125, %135] : f32 into vector<2x8xf32>
        %162 = arith.addf %153, %124 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %arg15 [%125, %126] : f32 into vector<2x8xf32>
        %165 = vector.extract %162[0, 1] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%125, %129] : f32 into vector<2x8xf32>
        %167 = vector.extract %162[0, 2] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%125, %132] : f32 into vector<2x8xf32>
        %169 = vector.extract %162[0, 3] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%125, %135] : f32 into vector<2x8xf32>
        %171 = vector.extract %115#3[%125, %126] : f32 from vector<2x8xf32>
        %172 = vector.extract %115#3[%125, %129] : f32 from vector<2x8xf32>
        %173 = vector.extract %115#3[%125, %132] : f32 from vector<2x8xf32>
        %174 = vector.extract %115#3[%125, %135] : f32 from vector<2x8xf32>
        %175 = vector.insert %171, %arg16 [%125, %126] : f32 into vector<2x8xf32>
        %176 = vector.insert %172, %175 [%125, %129] : f32 into vector<2x8xf32>
        %177 = vector.insert %173, %176 [%125, %132] : f32 into vector<2x8xf32>
        %178 = vector.insert %174, %177 [%125, %135] : f32 into vector<2x8xf32>
        affine.yield %137, %147, %161, %170, %178 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %122:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %124 = vector.extract %112[%arg11, 0] : f32 from vector<2x1xf32>
        %125 = vector.insert %124, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %126 = vector.extract %107[%arg11, 0] : f32 from vector<2x1xf32>
        %127 = vector.insert %126, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %125, %127 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %123:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %124 = affine.apply #map1(%arg11)
        %125 = affine.apply #map(%arg11)
        %126 = vector.extract %117#3[%124, %125] : f32 from vector<2x8xf32>
        %127 = affine.apply #map2(%arg11)
        %128 = vector.extract %117#3[%124, %127] : f32 from vector<2x8xf32>
        %129 = affine.apply #map3(%arg11)
        %130 = vector.extract %117#3[%124, %129] : f32 from vector<2x8xf32>
        %131 = affine.apply #map4(%arg11)
        %132 = vector.extract %117#3[%124, %131] : f32 from vector<2x8xf32>
        %133 = vector.insert %126, %arg12 [%124, %125] : f32 into vector<2x8xf32>
        %134 = vector.insert %128, %133 [%124, %127] : f32 into vector<2x8xf32>
        %135 = vector.insert %130, %134 [%124, %129] : f32 into vector<2x8xf32>
        %136 = vector.insert %132, %135 [%124, %131] : f32 into vector<2x8xf32>
        %137 = vector.extract %119#3[%124, %125] : f32 from vector<2x8xf32>
        %138 = vector.extract %119#3[%124, %127] : f32 from vector<2x8xf32>
        %139 = vector.extract %119#3[%124, %129] : f32 from vector<2x8xf32>
        %140 = vector.extract %119#3[%124, %131] : f32 from vector<2x8xf32>
        %141 = vector.insert %137, %arg13 [%124, %125] : f32 into vector<2x8xf32>
        %142 = vector.insert %138, %141 [%124, %127] : f32 into vector<2x8xf32>
        %143 = vector.insert %139, %142 [%124, %129] : f32 into vector<2x8xf32>
        %144 = vector.insert %140, %143 [%124, %131] : f32 into vector<2x8xf32>
        %145 = vector.extract %121#3[%124, %125] : f32 from vector<2x8xf32>
        %146 = vector.extract %121#3[%124, %127] : f32 from vector<2x8xf32>
        %147 = vector.extract %121#3[%124, %129] : f32 from vector<2x8xf32>
        %148 = vector.extract %121#3[%124, %131] : f32 from vector<2x8xf32>
        %149 = vector.insert %145, %arg14 [%124, %125] : f32 into vector<2x8xf32>
        %150 = vector.insert %146, %149 [%124, %127] : f32 into vector<2x8xf32>
        %151 = vector.insert %147, %150 [%124, %129] : f32 into vector<2x8xf32>
        %152 = vector.insert %148, %151 [%124, %131] : f32 into vector<2x8xf32>
        affine.yield %136, %144, %152 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %121#4, %122#0, %122#1, %123#0, %123#1, %123#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %81 = arith.index_cast %thread_id_x : index to i32
    %82 = arith.andi %81, %c16_i32 : i32
    %83 = arith.cmpi ne, %82, %c0_i32 : i32
    %84 = arith.andi %81, %c32_i32 : i32
    %85 = arith.cmpi ne, %84, %c0_i32 : i32
    %86 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %96 = affine.apply #map(%arg4)
      %97 = affine.apply #map1(%arg4)
      %98 = vector.extract %80#5[%97, %96] : f32 from vector<2x8xf32>
      %99 = affine.apply #map2(%arg4)
      %100 = vector.extract %80#5[%97, %99] : f32 from vector<2x8xf32>
      %101 = affine.apply #map3(%arg4)
      %102 = vector.extract %80#5[%97, %101] : f32 from vector<2x8xf32>
      %103 = affine.apply #map4(%arg4)
      %104 = vector.extract %80#5[%97, %103] : f32 from vector<2x8xf32>
      %105 = arith.select %83, %98, %100 : f32
      %shuffleResult, %valid = gpu.shuffle xor %105, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %106 = arith.select %83, %shuffleResult, %98 : f32
      %107 = arith.select %83, %100, %shuffleResult : f32
      %108 = arith.select %83, %102, %104 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %108, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %109 = arith.select %83, %shuffleResult_15, %102 : f32
      %110 = arith.select %83, %104, %shuffleResult_15 : f32
      %111 = arith.select %85, %106, %109 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %111, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %112 = arith.select %85, %shuffleResult_17, %106 : f32
      %113 = arith.select %85, %109, %shuffleResult_17 : f32
      %114 = arith.select %85, %107, %110 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %114, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %85, %shuffleResult_19, %107 : f32
      %116 = arith.select %85, %110, %shuffleResult_19 : f32
      %117 = vector.insert %112, %arg5 [%97, %96] : f32 into vector<2x8xf32>
      %118 = vector.insert %115, %117 [%97, %99] : f32 into vector<2x8xf32>
      %119 = vector.insert %113, %118 [%97, %101] : f32 into vector<2x8xf32>
      %120 = vector.insert %116, %119 [%97, %103] : f32 into vector<2x8xf32>
      affine.yield %120 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %87 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %96 = affine.apply #map(%arg4)
      %97 = affine.apply #map1(%arg4)
      %98 = vector.extract %80#4[%97, %96] : f32 from vector<2x8xf32>
      %99 = affine.apply #map2(%arg4)
      %100 = vector.extract %80#4[%97, %99] : f32 from vector<2x8xf32>
      %101 = affine.apply #map3(%arg4)
      %102 = vector.extract %80#4[%97, %101] : f32 from vector<2x8xf32>
      %103 = affine.apply #map4(%arg4)
      %104 = vector.extract %80#4[%97, %103] : f32 from vector<2x8xf32>
      %105 = arith.select %83, %98, %100 : f32
      %shuffleResult, %valid = gpu.shuffle xor %105, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %106 = arith.select %83, %shuffleResult, %98 : f32
      %107 = arith.select %83, %100, %shuffleResult : f32
      %108 = arith.select %83, %102, %104 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %108, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %109 = arith.select %83, %shuffleResult_15, %102 : f32
      %110 = arith.select %83, %104, %shuffleResult_15 : f32
      %111 = arith.select %85, %106, %109 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %111, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %112 = arith.select %85, %shuffleResult_17, %106 : f32
      %113 = arith.select %85, %109, %shuffleResult_17 : f32
      %114 = arith.select %85, %107, %110 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %114, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %85, %shuffleResult_19, %107 : f32
      %116 = arith.select %85, %110, %shuffleResult_19 : f32
      %117 = vector.insert %112, %arg5 [%97, %96] : f32 into vector<2x8xf32>
      %118 = vector.insert %115, %117 [%97, %99] : f32 into vector<2x8xf32>
      %119 = vector.insert %113, %118 [%97, %101] : f32 into vector<2x8xf32>
      %120 = vector.insert %116, %119 [%97, %103] : f32 into vector<2x8xf32>
      affine.yield %120 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %88 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %96 = affine.apply #map(%arg4)
      %97 = affine.apply #map1(%arg4)
      %98 = vector.extract %80#3[%97, %96] : f32 from vector<2x8xf32>
      %99 = affine.apply #map2(%arg4)
      %100 = vector.extract %80#3[%97, %99] : f32 from vector<2x8xf32>
      %101 = affine.apply #map3(%arg4)
      %102 = vector.extract %80#3[%97, %101] : f32 from vector<2x8xf32>
      %103 = affine.apply #map4(%arg4)
      %104 = vector.extract %80#3[%97, %103] : f32 from vector<2x8xf32>
      %105 = arith.select %83, %98, %100 : f32
      %shuffleResult, %valid = gpu.shuffle xor %105, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %106 = arith.select %83, %shuffleResult, %98 : f32
      %107 = arith.select %83, %100, %shuffleResult : f32
      %108 = arith.select %83, %102, %104 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %108, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %109 = arith.select %83, %shuffleResult_15, %102 : f32
      %110 = arith.select %83, %104, %shuffleResult_15 : f32
      %111 = arith.select %85, %106, %109 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %111, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %112 = arith.select %85, %shuffleResult_17, %106 : f32
      %113 = arith.select %85, %109, %shuffleResult_17 : f32
      %114 = arith.select %85, %107, %110 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %114, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %85, %shuffleResult_19, %107 : f32
      %116 = arith.select %85, %110, %shuffleResult_19 : f32
      %117 = vector.insert %112, %arg5 [%97, %96] : f32 into vector<2x8xf32>
      %118 = vector.insert %115, %117 [%97, %99] : f32 into vector<2x8xf32>
      %119 = vector.insert %113, %118 [%97, %101] : f32 into vector<2x8xf32>
      %120 = vector.insert %116, %119 [%97, %103] : f32 into vector<2x8xf32>
      affine.yield %120 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %89 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %96 = affine.apply #map(%arg4)
      %97 = affine.apply #map1(%arg4)
      %98 = vector.extract %80#0[%97, %96] : f32 from vector<2x8xf32>
      %99 = affine.apply #map2(%arg4)
      %100 = vector.extract %80#0[%97, %99] : f32 from vector<2x8xf32>
      %101 = affine.apply #map3(%arg4)
      %102 = vector.extract %80#0[%97, %101] : f32 from vector<2x8xf32>
      %103 = affine.apply #map4(%arg4)
      %104 = vector.extract %80#0[%97, %103] : f32 from vector<2x8xf32>
      %105 = arith.select %83, %98, %100 : f32
      %shuffleResult, %valid = gpu.shuffle xor %105, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %106 = arith.select %83, %shuffleResult, %98 : f32
      %107 = arith.select %83, %100, %shuffleResult : f32
      %108 = arith.select %83, %102, %104 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %108, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %109 = arith.select %83, %shuffleResult_15, %102 : f32
      %110 = arith.select %83, %104, %shuffleResult_15 : f32
      %111 = arith.select %85, %106, %109 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %111, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %112 = arith.select %85, %shuffleResult_17, %106 : f32
      %113 = arith.select %85, %109, %shuffleResult_17 : f32
      %114 = arith.select %85, %107, %110 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %114, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %85, %shuffleResult_19, %107 : f32
      %116 = arith.select %85, %110, %shuffleResult_19 : f32
      %117 = vector.insert %112, %arg5 [%97, %96] : f32 into vector<2x8xf32>
      %118 = vector.insert %115, %117 [%97, %99] : f32 into vector<2x8xf32>
      %119 = vector.insert %113, %118 [%97, %101] : f32 into vector<2x8xf32>
      %120 = vector.insert %116, %119 [%97, %103] : f32 into vector<2x8xf32>
      affine.yield %120 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %90:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %96 = affine.apply #map1(%arg4)
      %97 = vector.extract %80#1[%96, 0] : f32 from vector<2x1xf32>
      %98 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %138 = vector.broadcast %97 : f32 to vector<1x4xf32>
        affine.yield %138 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %99 = affine.apply #map(%arg4)
      %100 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
      %101 = vector.insert %100, %arg5 [%96, %99] : f32 into vector<2x8xf32>
      %102 = affine.apply #map2(%arg4)
      %103 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
      %104 = vector.insert %103, %101 [%96, %102] : f32 into vector<2x8xf32>
      %105 = affine.apply #map3(%arg4)
      %106 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
      %107 = vector.insert %106, %104 [%96, %105] : f32 into vector<2x8xf32>
      %108 = affine.apply #map4(%arg4)
      %109 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %107 [%96, %108] : f32 into vector<2x8xf32>
      %111 = vector.extract %89[%96, %99] : f32 from vector<2x8xf32>
      %112 = vector.extract %89[%96, %102] : f32 from vector<2x8xf32>
      %113 = vector.extract %89[%96, %105] : f32 from vector<2x8xf32>
      %114 = vector.extract %89[%96, %108] : f32 from vector<2x8xf32>
      %115 = vector.from_elements %111, %112, %113, %114 : vector<1x4xf32>
      %116 = arith.divf %115, %98 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %117 = vector.extract %116[0, 0] : f32 from vector<1x4xf32>
      %118 = vector.insert %117, %arg6 [%96, %99] : f32 into vector<2x8xf32>
      %119 = vector.extract %116[0, 1] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %118 [%96, %102] : f32 into vector<2x8xf32>
      %121 = vector.extract %116[0, 2] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%96, %105] : f32 into vector<2x8xf32>
      %123 = vector.extract %116[0, 3] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%96, %108] : f32 into vector<2x8xf32>
      %125 = arith.truncf %116 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %126 = vector.extract %125[0, 0] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %arg7 [%96, %99] : f16 into vector<2x8xf16>
      %128 = vector.extract %125[0, 1] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%96, %102] : f16 into vector<2x8xf16>
      %130 = vector.extract %125[0, 2] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%96, %105] : f16 into vector<2x8xf16>
      %132 = vector.extract %125[0, 3] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%96, %108] : f16 into vector<2x8xf16>
      %134 = vector.insert %126, %arg8 [%96, %99] : f16 into vector<2x8xf16>
      %135 = vector.insert %128, %134 [%96, %102] : f16 into vector<2x8xf16>
      %136 = vector.insert %130, %135 [%96, %105] : f16 into vector<2x8xf16>
      %137 = vector.insert %132, %136 [%96, %108] : f16 into vector<2x8xf16>
      affine.yield %110, %124, %133, %137 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %91 = affine.apply #map42()[%thread_id_x]
    %92 = arith.cmpi ult, %91, %c4 : index
    scf.if %92 {
      affine.for %arg4 = 0 to 4 {
        %96 = affine.apply #map1(%arg4)
        %97 = affine.apply #map(%arg4)
        %98 = vector.extract %90#3[%96, %97] : f16 from vector<2x8xf16>
        %99 = affine.apply #map2(%arg4)
        %100 = vector.extract %90#3[%96, %99] : f16 from vector<2x8xf16>
        %101 = affine.apply #map3(%arg4)
        %102 = vector.extract %90#3[%96, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply #map4(%arg4)
        %104 = vector.extract %90#3[%96, %103] : f16 from vector<2x8xf16>
        affine.store %98, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %100, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %102, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %93:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %96 = affine.apply #map1(%arg4)
      %97 = vector.extract %80#1[%96, 0] : f32 from vector<2x1xf32>
      %98 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %138 = vector.broadcast %97 : f32 to vector<1x4xf32>
        affine.yield %138 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %99 = affine.apply #map(%arg4)
      %100 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
      %101 = vector.insert %100, %arg5 [%96, %99] : f32 into vector<2x8xf32>
      %102 = affine.apply #map2(%arg4)
      %103 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
      %104 = vector.insert %103, %101 [%96, %102] : f32 into vector<2x8xf32>
      %105 = affine.apply #map3(%arg4)
      %106 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
      %107 = vector.insert %106, %104 [%96, %105] : f32 into vector<2x8xf32>
      %108 = affine.apply #map4(%arg4)
      %109 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %107 [%96, %108] : f32 into vector<2x8xf32>
      %111 = vector.extract %88[%96, %99] : f32 from vector<2x8xf32>
      %112 = vector.extract %88[%96, %102] : f32 from vector<2x8xf32>
      %113 = vector.extract %88[%96, %105] : f32 from vector<2x8xf32>
      %114 = vector.extract %88[%96, %108] : f32 from vector<2x8xf32>
      %115 = vector.from_elements %111, %112, %113, %114 : vector<1x4xf32>
      %116 = arith.divf %115, %98 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %117 = vector.extract %116[0, 0] : f32 from vector<1x4xf32>
      %118 = vector.insert %117, %arg6 [%96, %99] : f32 into vector<2x8xf32>
      %119 = vector.extract %116[0, 1] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %118 [%96, %102] : f32 into vector<2x8xf32>
      %121 = vector.extract %116[0, 2] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%96, %105] : f32 into vector<2x8xf32>
      %123 = vector.extract %116[0, 3] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%96, %108] : f32 into vector<2x8xf32>
      %125 = arith.truncf %116 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %126 = vector.extract %125[0, 0] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %arg7 [%96, %99] : f16 into vector<2x8xf16>
      %128 = vector.extract %125[0, 1] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%96, %102] : f16 into vector<2x8xf16>
      %130 = vector.extract %125[0, 2] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%96, %105] : f16 into vector<2x8xf16>
      %132 = vector.extract %125[0, 3] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%96, %108] : f16 into vector<2x8xf16>
      %134 = vector.insert %126, %arg8 [%96, %99] : f16 into vector<2x8xf16>
      %135 = vector.insert %128, %134 [%96, %102] : f16 into vector<2x8xf16>
      %136 = vector.insert %130, %135 [%96, %105] : f16 into vector<2x8xf16>
      %137 = vector.insert %132, %136 [%96, %108] : f16 into vector<2x8xf16>
      affine.yield %110, %124, %133, %137 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %92 {
      affine.for %arg4 = 0 to 4 {
        %96 = affine.apply #map1(%arg4)
        %97 = affine.apply #map(%arg4)
        %98 = vector.extract %93#3[%96, %97] : f16 from vector<2x8xf16>
        %99 = affine.apply #map2(%arg4)
        %100 = vector.extract %93#3[%96, %99] : f16 from vector<2x8xf16>
        %101 = affine.apply #map3(%arg4)
        %102 = vector.extract %93#3[%96, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply #map4(%arg4)
        %104 = vector.extract %93#3[%96, %103] : f16 from vector<2x8xf16>
        affine.store %98, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %100, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %102, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %94:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %96 = affine.apply #map1(%arg4)
      %97 = vector.extract %80#1[%96, 0] : f32 from vector<2x1xf32>
      %98 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %138 = vector.broadcast %97 : f32 to vector<1x4xf32>
        affine.yield %138 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %99 = affine.apply #map(%arg4)
      %100 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
      %101 = vector.insert %100, %arg5 [%96, %99] : f32 into vector<2x8xf32>
      %102 = affine.apply #map2(%arg4)
      %103 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
      %104 = vector.insert %103, %101 [%96, %102] : f32 into vector<2x8xf32>
      %105 = affine.apply #map3(%arg4)
      %106 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
      %107 = vector.insert %106, %104 [%96, %105] : f32 into vector<2x8xf32>
      %108 = affine.apply #map4(%arg4)
      %109 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %107 [%96, %108] : f32 into vector<2x8xf32>
      %111 = vector.extract %87[%96, %99] : f32 from vector<2x8xf32>
      %112 = vector.extract %87[%96, %102] : f32 from vector<2x8xf32>
      %113 = vector.extract %87[%96, %105] : f32 from vector<2x8xf32>
      %114 = vector.extract %87[%96, %108] : f32 from vector<2x8xf32>
      %115 = vector.from_elements %111, %112, %113, %114 : vector<1x4xf32>
      %116 = arith.divf %115, %98 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %117 = vector.extract %116[0, 0] : f32 from vector<1x4xf32>
      %118 = vector.insert %117, %arg6 [%96, %99] : f32 into vector<2x8xf32>
      %119 = vector.extract %116[0, 1] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %118 [%96, %102] : f32 into vector<2x8xf32>
      %121 = vector.extract %116[0, 2] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%96, %105] : f32 into vector<2x8xf32>
      %123 = vector.extract %116[0, 3] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%96, %108] : f32 into vector<2x8xf32>
      %125 = arith.truncf %116 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %126 = vector.extract %125[0, 0] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %arg7 [%96, %99] : f16 into vector<2x8xf16>
      %128 = vector.extract %125[0, 1] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%96, %102] : f16 into vector<2x8xf16>
      %130 = vector.extract %125[0, 2] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%96, %105] : f16 into vector<2x8xf16>
      %132 = vector.extract %125[0, 3] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%96, %108] : f16 into vector<2x8xf16>
      %134 = vector.insert %126, %arg8 [%96, %99] : f16 into vector<2x8xf16>
      %135 = vector.insert %128, %134 [%96, %102] : f16 into vector<2x8xf16>
      %136 = vector.insert %130, %135 [%96, %105] : f16 into vector<2x8xf16>
      %137 = vector.insert %132, %136 [%96, %108] : f16 into vector<2x8xf16>
      affine.yield %110, %124, %133, %137 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %92 {
      affine.for %arg4 = 0 to 4 {
        %96 = affine.apply #map1(%arg4)
        %97 = affine.apply #map(%arg4)
        %98 = vector.extract %94#3[%96, %97] : f16 from vector<2x8xf16>
        %99 = affine.apply #map2(%arg4)
        %100 = vector.extract %94#3[%96, %99] : f16 from vector<2x8xf16>
        %101 = affine.apply #map3(%arg4)
        %102 = vector.extract %94#3[%96, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply #map4(%arg4)
        %104 = vector.extract %94#3[%96, %103] : f16 from vector<2x8xf16>
        affine.store %98, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %100, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %102, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %95:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %96 = affine.apply #map1(%arg4)
      %97 = vector.extract %80#1[%96, 0] : f32 from vector<2x1xf32>
      %98 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %138 = vector.broadcast %97 : f32 to vector<1x4xf32>
        affine.yield %138 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %99 = affine.apply #map(%arg4)
      %100 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
      %101 = vector.insert %100, %arg5 [%96, %99] : f32 into vector<2x8xf32>
      %102 = affine.apply #map2(%arg4)
      %103 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
      %104 = vector.insert %103, %101 [%96, %102] : f32 into vector<2x8xf32>
      %105 = affine.apply #map3(%arg4)
      %106 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
      %107 = vector.insert %106, %104 [%96, %105] : f32 into vector<2x8xf32>
      %108 = affine.apply #map4(%arg4)
      %109 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
      %110 = vector.insert %109, %107 [%96, %108] : f32 into vector<2x8xf32>
      %111 = vector.extract %86[%96, %99] : f32 from vector<2x8xf32>
      %112 = vector.extract %86[%96, %102] : f32 from vector<2x8xf32>
      %113 = vector.extract %86[%96, %105] : f32 from vector<2x8xf32>
      %114 = vector.extract %86[%96, %108] : f32 from vector<2x8xf32>
      %115 = vector.from_elements %111, %112, %113, %114 : vector<1x4xf32>
      %116 = arith.divf %115, %98 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %117 = vector.extract %116[0, 0] : f32 from vector<1x4xf32>
      %118 = vector.insert %117, %arg6 [%96, %99] : f32 into vector<2x8xf32>
      %119 = vector.extract %116[0, 1] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %118 [%96, %102] : f32 into vector<2x8xf32>
      %121 = vector.extract %116[0, 2] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%96, %105] : f32 into vector<2x8xf32>
      %123 = vector.extract %116[0, 3] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%96, %108] : f32 into vector<2x8xf32>
      %125 = arith.truncf %116 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %126 = vector.extract %125[0, 0] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %arg7 [%96, %99] : f16 into vector<2x8xf16>
      %128 = vector.extract %125[0, 1] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%96, %102] : f16 into vector<2x8xf16>
      %130 = vector.extract %125[0, 2] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%96, %105] : f16 into vector<2x8xf16>
      %132 = vector.extract %125[0, 3] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%96, %108] : f16 into vector<2x8xf16>
      %134 = vector.insert %126, %arg8 [%96, %99] : f16 into vector<2x8xf16>
      %135 = vector.insert %128, %134 [%96, %102] : f16 into vector<2x8xf16>
      %136 = vector.insert %130, %135 [%96, %105] : f16 into vector<2x8xf16>
      %137 = vector.insert %132, %136 [%96, %108] : f16 into vector<2x8xf16>
      affine.yield %110, %124, %133, %137 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %92 {
      affine.for %arg4 = 0 to 4 {
        %96 = affine.apply #map1(%arg4)
        %97 = affine.apply #map(%arg4)
        %98 = vector.extract %95#3[%96, %97] : f16 from vector<2x8xf16>
        %99 = affine.apply #map2(%arg4)
        %100 = vector.extract %95#3[%96, %99] : f16 from vector<2x8xf16>
        %101 = affine.apply #map3(%arg4)
        %102 = vector.extract %95#3[%96, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply #map4(%arg4)
        %104 = vector.extract %95#3[%96, %103] : f16 from vector<2x8xf16>
        affine.store %98, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %100, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %102, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

