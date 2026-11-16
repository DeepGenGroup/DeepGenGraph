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
#map23 = affine_map<(d0)[s0, s1] -> (((d0 + s0 * 16777216 + s1 * 524288 + 131072) floordiv 524288) mod 32)>
#map24 = affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>
#map25 = affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map26 = affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map27 = affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map28 = affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>
#map29 = affine_map<(d0) -> (d0 * 4)>
#map30 = affine_map<(d0) -> (d0 * 4 + 1)>
#map31 = affine_map<(d0) -> (d0 * 4 + 2)>
#map32 = affine_map<(d0) -> (d0 * 4 + 3)>
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
      %100 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %101 = vector.insert %cst_7, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %102 = vector.insert %cst_7, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %103 = vector.insert %cst_7, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %100, %101, %102, %103 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %100 = affine.apply #map(%arg4)
      %101 = affine.apply #map1(%arg4)
      %102 = vector.insert %cst, %arg5 [%101, %100] : f32 into vector<2x8xf32>
      %103 = affine.apply #map2(%arg4)
      %104 = vector.insert %cst, %102 [%101, %103] : f32 into vector<2x8xf32>
      %105 = affine.apply #map3(%arg4)
      %106 = vector.insert %cst, %104 [%101, %105] : f32 into vector<2x8xf32>
      %107 = affine.apply #map4(%arg4)
      %108 = vector.insert %cst, %106 [%101, %107] : f32 into vector<2x8xf32>
      affine.yield %108 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6) -> (vector<2x1xf32>) {
      %100 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %100 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %12 = affine.apply #map5()[%block_id_z, %7, %6]
    %13 = affine.apply #map6()[%6]
    %subview = memref.subview %arg0[0, %12, %13, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %14 = frisk.async_copy_gm_to_shm %subview to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %14 : !frisk.async_task
    %alloc_9 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_10 = memref.subview %arg0[0, %12, %13, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %15 = frisk.async_copy_gm_to_shm %subview_10 to %alloc_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %thread_id_x = gpu.thread_id x
    %16 = affine.apply #map7()[%thread_id_x]
    %17 = affine.apply #map8()[%thread_id_x]
    %18 = memref.load %alloc[%16, %17] : memref<128x32xf16, 3>
    %19 = affine.apply #map9()[%thread_id_x]
    %20 = memref.load %alloc[%16, %19] : memref<128x32xf16, 3>
    %21 = affine.apply #map10()[%thread_id_x]
    %22 = memref.load %alloc[%16, %21] : memref<128x32xf16, 3>
    %23 = affine.apply #map11()[%thread_id_x]
    %24 = memref.load %alloc[%16, %23] : memref<128x32xf16, 3>
    %25 = vector.from_elements %18, %20, %22, %24 : vector<1x4xf16>
    %26:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %25) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %100 = affine.apply #map12(%arg4)[%thread_id_x]
      %101 = affine.apply #map13(%arg4)[%thread_id_x]
      %102 = memref.load %alloc[%100, %101] : memref<128x32xf16, 3>
      %103 = affine.apply #map14(%arg4)[%thread_id_x]
      %104 = memref.load %alloc[%100, %103] : memref<128x32xf16, 3>
      %105 = affine.apply #map15(%arg4)[%thread_id_x]
      %106 = memref.load %alloc[%100, %105] : memref<128x32xf16, 3>
      %107 = affine.apply #map16(%arg4)[%thread_id_x]
      %108 = memref.load %alloc[%100, %107] : memref<128x32xf16, 3>
      %109 = vector.from_elements %102, %104, %106, %108 : vector<1x4xf16>
      %110 = affine.apply #map1(%arg4)
      %111 = affine.apply #map(%arg4)
      %112 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %arg5 [%110, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply #map2(%arg4)
      %115 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%110, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply #map3(%arg4)
      %118 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%110, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply #map4(%arg4)
      %121 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%110, %120] : f16 into vector<2x8xf16>
      %123 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %124 = vector.insert %8, %arg6 [%110, %111] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %124 [%110, %114] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%110, %117] : f16 into vector<2x8xf16>
      %127 = vector.insert %8, %126 [%110, %120] : f16 into vector<2x8xf16>
      %128 = arith.mulf %arg8, %123 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %129 = vector.extract %128[0, 0] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %arg7 [%110, %111] : f16 into vector<2x8xf16>
      %131 = vector.extract %128[0, 1] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%110, %114] : f16 into vector<2x8xf16>
      %133 = vector.extract %128[0, 2] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%110, %117] : f16 into vector<2x8xf16>
      %135 = vector.extract %128[0, 3] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%110, %120] : f16 into vector<2x8xf16>
      affine.yield %122, %127, %136, %109 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %27 = vector.broadcast %8 : f16 to vector<1x4xf16>
    %28 = arith.mulf %26#3, %27 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %29 = vector.extract %28[0, 0] : f16 from vector<1x4xf16>
    %30 = vector.insert %29, %26#2 [1, 4] : f16 into vector<2x8xf16>
    %31 = vector.extract %28[0, 1] : f16 from vector<1x4xf16>
    %32 = vector.insert %31, %30 [1, 5] : f16 into vector<2x8xf16>
    %33 = vector.extract %28[0, 2] : f16 from vector<1x4xf16>
    %34 = vector.insert %33, %32 [1, 6] : f16 into vector<2x8xf16>
    %35 = vector.extract %28[0, 3] : f16 from vector<1x4xf16>
    %36 = vector.insert %35, %34 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %15 : !frisk.async_task
    %alloc_11 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_12 = memref.subview %arg0[0, %12, %13, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %37 = frisk.async_copy_gm_to_shm %subview_12 to %alloc_11 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %38 = memref.load %alloc_9[%16, %17] : memref<128x32xf16, 3>
    %39 = memref.load %alloc_9[%16, %19] : memref<128x32xf16, 3>
    %40 = memref.load %alloc_9[%16, %21] : memref<128x32xf16, 3>
    %41 = memref.load %alloc_9[%16, %23] : memref<128x32xf16, 3>
    %42 = vector.from_elements %38, %39, %40, %41 : vector<1x4xf16>
    %43:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %42) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %100 = affine.apply #map12(%arg4)[%thread_id_x]
      %101 = affine.apply #map13(%arg4)[%thread_id_x]
      %102 = memref.load %alloc_9[%100, %101] : memref<128x32xf16, 3>
      %103 = affine.apply #map14(%arg4)[%thread_id_x]
      %104 = memref.load %alloc_9[%100, %103] : memref<128x32xf16, 3>
      %105 = affine.apply #map15(%arg4)[%thread_id_x]
      %106 = memref.load %alloc_9[%100, %105] : memref<128x32xf16, 3>
      %107 = affine.apply #map16(%arg4)[%thread_id_x]
      %108 = memref.load %alloc_9[%100, %107] : memref<128x32xf16, 3>
      %109 = vector.from_elements %102, %104, %106, %108 : vector<1x4xf16>
      %110 = affine.apply #map1(%arg4)
      %111 = affine.apply #map(%arg4)
      %112 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %arg5 [%110, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply #map2(%arg4)
      %115 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%110, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply #map3(%arg4)
      %118 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%110, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply #map4(%arg4)
      %121 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%110, %120] : f16 into vector<2x8xf16>
      %123 = vector.insert %8, %arg6 [%110, %111] : f16 into vector<2x8xf16>
      %124 = vector.insert %8, %123 [%110, %114] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %124 [%110, %117] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%110, %120] : f16 into vector<2x8xf16>
      %127 = arith.mulf %arg8, %27 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %128 = vector.extract %127[0, 0] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %arg7 [%110, %111] : f16 into vector<2x8xf16>
      %130 = vector.extract %127[0, 1] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%110, %114] : f16 into vector<2x8xf16>
      %132 = vector.extract %127[0, 2] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%110, %117] : f16 into vector<2x8xf16>
      %134 = vector.extract %127[0, 3] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%110, %120] : f16 into vector<2x8xf16>
      affine.yield %122, %126, %135, %109 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %44 = arith.mulf %43#3, %27 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %45 = vector.extract %44[0, 0] : f16 from vector<1x4xf16>
    %46 = vector.insert %45, %43#2 [1, 4] : f16 into vector<2x8xf16>
    %47 = vector.extract %44[0, 1] : f16 from vector<1x4xf16>
    %48 = vector.insert %47, %46 [1, 5] : f16 into vector<2x8xf16>
    %49 = vector.extract %44[0, 2] : f16 from vector<1x4xf16>
    %50 = vector.insert %49, %48 [1, 6] : f16 into vector<2x8xf16>
    %51 = vector.extract %44[0, 3] : f16 from vector<1x4xf16>
    %52 = vector.insert %51, %50 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %37 : !frisk.async_task
    %alloc_13 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_14 = memref.subview %arg0[0, %12, %13, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %53 = frisk.async_copy_gm_to_shm %subview_14 to %alloc_13 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %54 = memref.load %alloc_11[%16, %17] : memref<128x32xf16, 3>
    %55 = memref.load %alloc_11[%16, %19] : memref<128x32xf16, 3>
    %56 = memref.load %alloc_11[%16, %21] : memref<128x32xf16, 3>
    %57 = memref.load %alloc_11[%16, %23] : memref<128x32xf16, 3>
    %58 = vector.from_elements %54, %55, %56, %57 : vector<1x4xf16>
    %59:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %58) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %100 = affine.apply #map12(%arg4)[%thread_id_x]
      %101 = affine.apply #map13(%arg4)[%thread_id_x]
      %102 = memref.load %alloc_11[%100, %101] : memref<128x32xf16, 3>
      %103 = affine.apply #map14(%arg4)[%thread_id_x]
      %104 = memref.load %alloc_11[%100, %103] : memref<128x32xf16, 3>
      %105 = affine.apply #map15(%arg4)[%thread_id_x]
      %106 = memref.load %alloc_11[%100, %105] : memref<128x32xf16, 3>
      %107 = affine.apply #map16(%arg4)[%thread_id_x]
      %108 = memref.load %alloc_11[%100, %107] : memref<128x32xf16, 3>
      %109 = vector.from_elements %102, %104, %106, %108 : vector<1x4xf16>
      %110 = affine.apply #map1(%arg4)
      %111 = affine.apply #map(%arg4)
      %112 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %arg5 [%110, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply #map2(%arg4)
      %115 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%110, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply #map3(%arg4)
      %118 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%110, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply #map4(%arg4)
      %121 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%110, %120] : f16 into vector<2x8xf16>
      %123 = vector.insert %8, %arg6 [%110, %111] : f16 into vector<2x8xf16>
      %124 = vector.insert %8, %123 [%110, %114] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %124 [%110, %117] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%110, %120] : f16 into vector<2x8xf16>
      %127 = arith.mulf %arg8, %27 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %128 = vector.extract %127[0, 0] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %arg7 [%110, %111] : f16 into vector<2x8xf16>
      %130 = vector.extract %127[0, 1] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%110, %114] : f16 into vector<2x8xf16>
      %132 = vector.extract %127[0, 2] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%110, %117] : f16 into vector<2x8xf16>
      %134 = vector.extract %127[0, 3] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%110, %120] : f16 into vector<2x8xf16>
      affine.yield %122, %126, %135, %109 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %60 = arith.mulf %59#3, %27 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %61 = vector.extract %60[0, 0] : f16 from vector<1x4xf16>
    %62 = vector.insert %61, %59#2 [1, 4] : f16 into vector<2x8xf16>
    %63 = vector.extract %60[0, 1] : f16 from vector<1x4xf16>
    %64 = vector.insert %63, %62 [1, 5] : f16 into vector<2x8xf16>
    %65 = vector.extract %60[0, 2] : f16 from vector<1x4xf16>
    %66 = vector.insert %65, %64 [1, 6] : f16 into vector<2x8xf16>
    %67 = vector.extract %60[0, 3] : f16 from vector<1x4xf16>
    %68 = vector.insert %67, %66 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %53 : !frisk.async_task
    %69 = memref.load %alloc_13[%16, %17] : memref<128x32xf16, 3>
    %70 = memref.load %alloc_13[%16, %19] : memref<128x32xf16, 3>
    %71 = memref.load %alloc_13[%16, %21] : memref<128x32xf16, 3>
    %72 = memref.load %alloc_13[%16, %23] : memref<128x32xf16, 3>
    %73 = vector.from_elements %69, %70, %71, %72 : vector<1x4xf16>
    %74:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %73) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %100 = affine.apply #map12(%arg4)[%thread_id_x]
      %101 = affine.apply #map13(%arg4)[%thread_id_x]
      %102 = memref.load %alloc_13[%100, %101] : memref<128x32xf16, 3>
      %103 = affine.apply #map14(%arg4)[%thread_id_x]
      %104 = memref.load %alloc_13[%100, %103] : memref<128x32xf16, 3>
      %105 = affine.apply #map15(%arg4)[%thread_id_x]
      %106 = memref.load %alloc_13[%100, %105] : memref<128x32xf16, 3>
      %107 = affine.apply #map16(%arg4)[%thread_id_x]
      %108 = memref.load %alloc_13[%100, %107] : memref<128x32xf16, 3>
      %109 = vector.from_elements %102, %104, %106, %108 : vector<1x4xf16>
      %110 = affine.apply #map1(%arg4)
      %111 = affine.apply #map(%arg4)
      %112 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %arg5 [%110, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply #map2(%arg4)
      %115 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%110, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply #map3(%arg4)
      %118 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%110, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply #map4(%arg4)
      %121 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%110, %120] : f16 into vector<2x8xf16>
      %123 = vector.insert %8, %arg6 [%110, %111] : f16 into vector<2x8xf16>
      %124 = vector.insert %8, %123 [%110, %114] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %124 [%110, %117] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%110, %120] : f16 into vector<2x8xf16>
      %127 = arith.mulf %arg8, %27 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %128 = vector.extract %127[0, 0] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %arg7 [%110, %111] : f16 into vector<2x8xf16>
      %130 = vector.extract %127[0, 1] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%110, %114] : f16 into vector<2x8xf16>
      %132 = vector.extract %127[0, 2] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%110, %117] : f16 into vector<2x8xf16>
      %134 = vector.extract %127[0, 3] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%110, %120] : f16 into vector<2x8xf16>
      affine.yield %122, %126, %135, %109 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %75 = arith.mulf %74#3, %27 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %76 = vector.extract %75[0, 0] : f16 from vector<1x4xf16>
    %77 = vector.insert %76, %74#2 [1, 4] : f16 into vector<2x8xf16>
    %78 = vector.extract %75[0, 1] : f16 from vector<1x4xf16>
    %79 = vector.insert %78, %77 [1, 5] : f16 into vector<2x8xf16>
    %80 = vector.extract %75[0, 2] : f16 from vector<1x4xf16>
    %81 = vector.insert %80, %79 [1, 6] : f16 into vector<2x8xf16>
    %82 = vector.extract %75[0, 3] : f16 from vector<1x4xf16>
    %83 = vector.insert %82, %81 [1, 7] : f16 into vector<2x8xf16>
    %84:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#3, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %100 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %136 = affine.apply #map17(%arg11)
        %137 = affine.apply #map18(%arg11)
        %138 = vector.insert %cst, %arg12 [%137, %136] : f32 into vector<2x32xf32>
        %139 = affine.apply #map19(%arg11)
        %140 = vector.insert %cst, %138 [%137, %139] : f32 into vector<2x32xf32>
        %141 = affine.apply #map20(%arg11)
        %142 = vector.insert %cst, %140 [%137, %141] : f32 into vector<2x32xf32>
        %143 = affine.apply #map21(%arg11)
        %144 = vector.insert %cst, %142 [%137, %143] : f32 into vector<2x32xf32>
        affine.yield %144 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %101 = affine.apply #map22()[%7]
      %subview_16 = memref.subview %transpose[0, %101, 0, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %102 = frisk.async_copy_gm_to_shm %subview_16 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %102 : !frisk.async_task
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %103 = affine.apply #map23(%arg4)[%block_id_z, %7]
      %subview_18 = memref.subview %transpose[0, %103, 32, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %104 = frisk.async_copy_gm_to_shm %subview_18 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %105:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %136 = affine.apply #map24(%arg11)[%thread_id_x]
        %137 = memref.load %alloc_15[%17, %136] : memref<32x128xf16, 3>
        %138 = memref.load %alloc_15[%19, %136] : memref<32x128xf16, 3>
        %139 = memref.load %alloc_15[%21, %136] : memref<32x128xf16, 3>
        %140 = memref.load %alloc_15[%23, %136] : memref<32x128xf16, 3>
        %141 = vector.from_elements %137, %138, %139, %140 : vector<4x1xf16>
        %142:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %141) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %172 = affine.apply #map25(%arg14)[%thread_id_x]
          %173 = memref.load %alloc_15[%172, %136] : memref<32x128xf16, 3>
          %174 = affine.apply #map26(%arg14)[%thread_id_x]
          %175 = memref.load %alloc_15[%174, %136] : memref<32x128xf16, 3>
          %176 = affine.apply #map27(%arg14)[%thread_id_x]
          %177 = memref.load %alloc_15[%176, %136] : memref<32x128xf16, 3>
          %178 = affine.apply #map28(%arg14)[%thread_id_x]
          %179 = memref.load %alloc_15[%178, %136] : memref<32x128xf16, 3>
          %180 = vector.from_elements %173, %175, %177, %179 : vector<4x1xf16>
          %181 = affine.apply #map18(%arg11)
          %182 = affine.apply #map29(%arg14)
          %183 = vector.extract %36[%181, %182] : f16 from vector<2x8xf16>
          %184 = affine.apply #map30(%arg14)
          %185 = vector.extract %36[%181, %184] : f16 from vector<2x8xf16>
          %186 = affine.apply #map31(%arg14)
          %187 = vector.extract %36[%181, %186] : f16 from vector<2x8xf16>
          %188 = affine.apply #map32(%arg14)
          %189 = vector.extract %36[%181, %188] : f16 from vector<2x8xf16>
          %190 = vector.from_elements %183, %185, %187, %189 : vector<1x4xf16>
          %191 = frisk.warp_mma_rr %190, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %191, %180 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %143 = affine.apply #map18(%arg11)
        %144 = vector.extract %36[%143, 4] : f16 from vector<2x8xf16>
        %145 = vector.extract %36[%143, 5] : f16 from vector<2x8xf16>
        %146 = vector.extract %36[%143, 6] : f16 from vector<2x8xf16>
        %147 = vector.extract %36[%143, 7] : f16 from vector<2x8xf16>
        %148 = vector.from_elements %144, %145, %146, %147 : vector<1x4xf16>
        %149 = frisk.warp_mma_rr %148, %142#1, %142#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %150 = affine.apply #map17(%arg11)
        %151 = vector.extract %100[%143, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map19(%arg11)
        %153 = vector.extract %100[%143, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply #map20(%arg11)
        %155 = vector.extract %100[%143, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply #map21(%arg11)
        %157 = vector.extract %100[%143, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = arith.addf %158, %149 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %160 = vector.extract %159[0, 0] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %arg12 [%143, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %159[0, 1] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%143, %152] : f32 into vector<2x32xf32>
        %164 = vector.extract %159[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%143, %154] : f32 into vector<2x32xf32>
        %166 = vector.extract %159[0, 3] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%143, %156] : f32 into vector<2x32xf32>
        %168 = vector.insert %160, %arg13 [%143, %150] : f32 into vector<2x32xf32>
        %169 = vector.insert %162, %168 [%143, %152] : f32 into vector<2x32xf32>
        %170 = vector.insert %164, %169 [%143, %154] : f32 into vector<2x32xf32>
        %171 = vector.insert %166, %170 [%143, %156] : f32 into vector<2x32xf32>
        affine.yield %167, %171 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %104 : !frisk.async_task
      %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %106 = affine.apply #map33(%arg4)[%block_id_z, %7]
      %subview_20 = memref.subview %transpose[0, %106, 64, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %107 = frisk.async_copy_gm_to_shm %subview_20 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %108:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %136 = affine.apply #map24(%arg11)[%thread_id_x]
        %137 = memref.load %alloc_17[%17, %136] : memref<32x128xf16, 3>
        %138 = memref.load %alloc_17[%19, %136] : memref<32x128xf16, 3>
        %139 = memref.load %alloc_17[%21, %136] : memref<32x128xf16, 3>
        %140 = memref.load %alloc_17[%23, %136] : memref<32x128xf16, 3>
        %141 = vector.from_elements %137, %138, %139, %140 : vector<4x1xf16>
        %142:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %141) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %172 = affine.apply #map25(%arg14)[%thread_id_x]
          %173 = memref.load %alloc_17[%172, %136] : memref<32x128xf16, 3>
          %174 = affine.apply #map26(%arg14)[%thread_id_x]
          %175 = memref.load %alloc_17[%174, %136] : memref<32x128xf16, 3>
          %176 = affine.apply #map27(%arg14)[%thread_id_x]
          %177 = memref.load %alloc_17[%176, %136] : memref<32x128xf16, 3>
          %178 = affine.apply #map28(%arg14)[%thread_id_x]
          %179 = memref.load %alloc_17[%178, %136] : memref<32x128xf16, 3>
          %180 = vector.from_elements %173, %175, %177, %179 : vector<4x1xf16>
          %181 = affine.apply #map18(%arg11)
          %182 = affine.apply #map29(%arg14)
          %183 = vector.extract %52[%181, %182] : f16 from vector<2x8xf16>
          %184 = affine.apply #map30(%arg14)
          %185 = vector.extract %52[%181, %184] : f16 from vector<2x8xf16>
          %186 = affine.apply #map31(%arg14)
          %187 = vector.extract %52[%181, %186] : f16 from vector<2x8xf16>
          %188 = affine.apply #map32(%arg14)
          %189 = vector.extract %52[%181, %188] : f16 from vector<2x8xf16>
          %190 = vector.from_elements %183, %185, %187, %189 : vector<1x4xf16>
          %191 = frisk.warp_mma_rr %190, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %191, %180 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %143 = affine.apply #map18(%arg11)
        %144 = vector.extract %52[%143, 4] : f16 from vector<2x8xf16>
        %145 = vector.extract %52[%143, 5] : f16 from vector<2x8xf16>
        %146 = vector.extract %52[%143, 6] : f16 from vector<2x8xf16>
        %147 = vector.extract %52[%143, 7] : f16 from vector<2x8xf16>
        %148 = vector.from_elements %144, %145, %146, %147 : vector<1x4xf16>
        %149 = frisk.warp_mma_rr %148, %142#1, %142#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %150 = affine.apply #map17(%arg11)
        %151 = vector.extract %105#1[%143, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map19(%arg11)
        %153 = vector.extract %105#1[%143, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply #map20(%arg11)
        %155 = vector.extract %105#1[%143, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply #map21(%arg11)
        %157 = vector.extract %105#1[%143, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = arith.addf %158, %149 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %160 = vector.extract %159[0, 0] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %arg12 [%143, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %159[0, 1] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%143, %152] : f32 into vector<2x32xf32>
        %164 = vector.extract %159[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%143, %154] : f32 into vector<2x32xf32>
        %166 = vector.extract %159[0, 3] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%143, %156] : f32 into vector<2x32xf32>
        %168 = vector.insert %160, %arg13 [%143, %150] : f32 into vector<2x32xf32>
        %169 = vector.insert %162, %168 [%143, %152] : f32 into vector<2x32xf32>
        %170 = vector.insert %164, %169 [%143, %154] : f32 into vector<2x32xf32>
        %171 = vector.insert %166, %170 [%143, %156] : f32 into vector<2x32xf32>
        affine.yield %167, %171 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %107 : !frisk.async_task
      %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %109 = affine.apply #map34(%arg4)[%block_id_z, %7]
      %subview_22 = memref.subview %transpose[0, %109, 96, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %110 = frisk.async_copy_gm_to_shm %subview_22 to %alloc_21 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %111:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %136 = affine.apply #map24(%arg11)[%thread_id_x]
        %137 = memref.load %alloc_19[%17, %136] : memref<32x128xf16, 3>
        %138 = memref.load %alloc_19[%19, %136] : memref<32x128xf16, 3>
        %139 = memref.load %alloc_19[%21, %136] : memref<32x128xf16, 3>
        %140 = memref.load %alloc_19[%23, %136] : memref<32x128xf16, 3>
        %141 = vector.from_elements %137, %138, %139, %140 : vector<4x1xf16>
        %142:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %141) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %172 = affine.apply #map25(%arg14)[%thread_id_x]
          %173 = memref.load %alloc_19[%172, %136] : memref<32x128xf16, 3>
          %174 = affine.apply #map26(%arg14)[%thread_id_x]
          %175 = memref.load %alloc_19[%174, %136] : memref<32x128xf16, 3>
          %176 = affine.apply #map27(%arg14)[%thread_id_x]
          %177 = memref.load %alloc_19[%176, %136] : memref<32x128xf16, 3>
          %178 = affine.apply #map28(%arg14)[%thread_id_x]
          %179 = memref.load %alloc_19[%178, %136] : memref<32x128xf16, 3>
          %180 = vector.from_elements %173, %175, %177, %179 : vector<4x1xf16>
          %181 = affine.apply #map18(%arg11)
          %182 = affine.apply #map29(%arg14)
          %183 = vector.extract %68[%181, %182] : f16 from vector<2x8xf16>
          %184 = affine.apply #map30(%arg14)
          %185 = vector.extract %68[%181, %184] : f16 from vector<2x8xf16>
          %186 = affine.apply #map31(%arg14)
          %187 = vector.extract %68[%181, %186] : f16 from vector<2x8xf16>
          %188 = affine.apply #map32(%arg14)
          %189 = vector.extract %68[%181, %188] : f16 from vector<2x8xf16>
          %190 = vector.from_elements %183, %185, %187, %189 : vector<1x4xf16>
          %191 = frisk.warp_mma_rr %190, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %191, %180 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %143 = affine.apply #map18(%arg11)
        %144 = vector.extract %68[%143, 4] : f16 from vector<2x8xf16>
        %145 = vector.extract %68[%143, 5] : f16 from vector<2x8xf16>
        %146 = vector.extract %68[%143, 6] : f16 from vector<2x8xf16>
        %147 = vector.extract %68[%143, 7] : f16 from vector<2x8xf16>
        %148 = vector.from_elements %144, %145, %146, %147 : vector<1x4xf16>
        %149 = frisk.warp_mma_rr %148, %142#1, %142#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %150 = affine.apply #map17(%arg11)
        %151 = vector.extract %108#1[%143, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map19(%arg11)
        %153 = vector.extract %108#1[%143, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply #map20(%arg11)
        %155 = vector.extract %108#1[%143, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply #map21(%arg11)
        %157 = vector.extract %108#1[%143, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = arith.addf %158, %149 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %160 = vector.extract %159[0, 0] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %arg12 [%143, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %159[0, 1] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%143, %152] : f32 into vector<2x32xf32>
        %164 = vector.extract %159[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%143, %154] : f32 into vector<2x32xf32>
        %166 = vector.extract %159[0, 3] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%143, %156] : f32 into vector<2x32xf32>
        %168 = vector.insert %160, %arg13 [%143, %150] : f32 into vector<2x32xf32>
        %169 = vector.insert %162, %168 [%143, %152] : f32 into vector<2x32xf32>
        %170 = vector.insert %164, %169 [%143, %154] : f32 into vector<2x32xf32>
        %171 = vector.insert %166, %170 [%143, %156] : f32 into vector<2x32xf32>
        affine.yield %167, %171 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %110 : !frisk.async_task
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_24 = memref.subview %arg2[0, %101, %arg4, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %112 = frisk.async_copy_gm_to_shm %subview_24 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %113:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %136 = affine.apply #map24(%arg11)[%thread_id_x]
        %137 = memref.load %alloc_21[%17, %136] : memref<32x128xf16, 3>
        %138 = memref.load %alloc_21[%19, %136] : memref<32x128xf16, 3>
        %139 = memref.load %alloc_21[%21, %136] : memref<32x128xf16, 3>
        %140 = memref.load %alloc_21[%23, %136] : memref<32x128xf16, 3>
        %141 = vector.from_elements %137, %138, %139, %140 : vector<4x1xf16>
        %142:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %141) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %172 = affine.apply #map25(%arg14)[%thread_id_x]
          %173 = memref.load %alloc_21[%172, %136] : memref<32x128xf16, 3>
          %174 = affine.apply #map26(%arg14)[%thread_id_x]
          %175 = memref.load %alloc_21[%174, %136] : memref<32x128xf16, 3>
          %176 = affine.apply #map27(%arg14)[%thread_id_x]
          %177 = memref.load %alloc_21[%176, %136] : memref<32x128xf16, 3>
          %178 = affine.apply #map28(%arg14)[%thread_id_x]
          %179 = memref.load %alloc_21[%178, %136] : memref<32x128xf16, 3>
          %180 = vector.from_elements %173, %175, %177, %179 : vector<4x1xf16>
          %181 = affine.apply #map18(%arg11)
          %182 = affine.apply #map29(%arg14)
          %183 = vector.extract %83[%181, %182] : f16 from vector<2x8xf16>
          %184 = affine.apply #map30(%arg14)
          %185 = vector.extract %83[%181, %184] : f16 from vector<2x8xf16>
          %186 = affine.apply #map31(%arg14)
          %187 = vector.extract %83[%181, %186] : f16 from vector<2x8xf16>
          %188 = affine.apply #map32(%arg14)
          %189 = vector.extract %83[%181, %188] : f16 from vector<2x8xf16>
          %190 = vector.from_elements %183, %185, %187, %189 : vector<1x4xf16>
          %191 = frisk.warp_mma_rr %190, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %191, %180 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %143 = affine.apply #map18(%arg11)
        %144 = vector.extract %83[%143, 4] : f16 from vector<2x8xf16>
        %145 = vector.extract %83[%143, 5] : f16 from vector<2x8xf16>
        %146 = vector.extract %83[%143, 6] : f16 from vector<2x8xf16>
        %147 = vector.extract %83[%143, 7] : f16 from vector<2x8xf16>
        %148 = vector.from_elements %144, %145, %146, %147 : vector<1x4xf16>
        %149 = frisk.warp_mma_rr %148, %142#1, %142#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %150 = affine.apply #map17(%arg11)
        %151 = vector.extract %111#1[%143, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply #map19(%arg11)
        %153 = vector.extract %111#1[%143, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply #map20(%arg11)
        %155 = vector.extract %111#1[%143, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply #map21(%arg11)
        %157 = vector.extract %111#1[%143, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = arith.addf %158, %149 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %160 = vector.extract %159[0, 0] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %arg12 [%143, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %159[0, 1] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%143, %152] : f32 into vector<2x32xf32>
        %164 = vector.extract %159[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%143, %154] : f32 into vector<2x32xf32>
        %166 = vector.extract %159[0, 3] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%143, %156] : f32 into vector<2x32xf32>
        %168 = vector.insert %160, %arg13 [%143, %150] : f32 into vector<2x32xf32>
        %169 = vector.insert %162, %168 [%143, %152] : f32 into vector<2x32xf32>
        %170 = vector.insert %164, %169 [%143, %154] : f32 into vector<2x32xf32>
        %171 = vector.insert %166, %170 [%143, %156] : f32 into vector<2x32xf32>
        affine.yield %167, %171 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %114 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %136 = affine.apply #map18(%arg11)
        %137 = affine.apply #map17(%arg11)
        %138 = vector.extract %113#1[%136, %137] : f32 from vector<2x32xf32>
        %139 = affine.apply #map19(%arg11)
        %140 = vector.extract %113#1[%136, %139] : f32 from vector<2x32xf32>
        %141 = affine.apply #map20(%arg11)
        %142 = vector.extract %113#1[%136, %141] : f32 from vector<2x32xf32>
        %143 = affine.apply #map21(%arg11)
        %144 = vector.extract %113#1[%136, %143] : f32 from vector<2x32xf32>
        %145 = vector.insert %138, %arg12 [%136, %137] : f32 into vector<2x32xf32>
        %146 = vector.insert %140, %145 [%136, %139] : f32 into vector<2x32xf32>
        %147 = vector.insert %142, %146 [%136, %141] : f32 into vector<2x32xf32>
        %148 = vector.insert %144, %147 [%136, %143] : f32 into vector<2x32xf32>
        affine.yield %148 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %115 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %136 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_7) -> (f32) {
          %140 = affine.apply #map29(%arg13)
          %141 = vector.extract %114[%arg11, %140] : f32 from vector<2x32xf32>
          %142 = arith.maxnumf %arg14, %141 : f32
          %143 = affine.apply #map30(%arg13)
          %144 = vector.extract %114[%arg11, %143] : f32 from vector<2x32xf32>
          %145 = arith.maxnumf %142, %144 : f32
          %146 = affine.apply #map31(%arg13)
          %147 = vector.extract %114[%arg11, %146] : f32 from vector<2x32xf32>
          %148 = arith.maxnumf %145, %147 : f32
          %149 = affine.apply #map32(%arg13)
          %150 = vector.extract %114[%arg11, %149] : f32 from vector<2x32xf32>
          %151 = arith.maxnumf %148, %150 : f32
          affine.yield %151 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %136, %c16_i32, %c64_i32 : f32
        %137 = arith.maxnumf %136, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %137, %c32_i32, %c64_i32 : f32
        %138 = arith.maxnumf %137, %shuffleResult_31 : f32
        %139 = vector.insert %138, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %139 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %116 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %136 = vector.extract %115[%arg11, 0] : f32 from vector<2x1xf32>
        %137 = vector.broadcast %136 : f32 to vector<1x1xf32>
        %138 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %139 = vector.broadcast %138 : f32 to vector<1x1xf32>
        %140 = arith.maxnumf %137, %139 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %141 = vector.extract %140[0, 0] : f32 from vector<1x1xf32>
        %142 = vector.insert %141, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %142 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %117:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %136 = affine.apply #map18(%arg11)
        %137 = vector.extract %116[%136, 0] : f32 from vector<2x1xf32>
        %138 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3) -> (vector<1x4xf32>) {
          %165 = vector.broadcast %137 : f32 to vector<1x4xf32>
          affine.yield %165 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %139 = affine.apply #map17(%arg11)
        %140 = vector.extract %138[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%136, %139] : f32 into vector<2x32xf32>
        %142 = affine.apply #map19(%arg11)
        %143 = vector.extract %138[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%136, %142] : f32 into vector<2x32xf32>
        %145 = affine.apply #map20(%arg11)
        %146 = vector.extract %138[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%136, %145] : f32 into vector<2x32xf32>
        %148 = affine.apply #map21(%arg11)
        %149 = vector.extract %138[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%136, %148] : f32 into vector<2x32xf32>
        %151 = vector.extract %113#1[%136, %139] : f32 from vector<2x32xf32>
        %152 = vector.extract %113#1[%136, %142] : f32 from vector<2x32xf32>
        %153 = vector.extract %113#1[%136, %145] : f32 from vector<2x32xf32>
        %154 = vector.extract %113#1[%136, %148] : f32 from vector<2x32xf32>
        %155 = vector.from_elements %151, %152, %153, %154 : vector<1x4xf32>
        %156 = arith.subf %155, %138 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %157 = vector.extract %156[0, 0] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %arg13 [%136, %139] : f32 into vector<2x32xf32>
        %159 = vector.extract %156[0, 1] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%136, %142] : f32 into vector<2x32xf32>
        %161 = vector.extract %156[0, 2] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %160 [%136, %145] : f32 into vector<2x32xf32>
        %163 = vector.extract %156[0, 3] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %162 [%136, %148] : f32 into vector<2x32xf32>
        affine.yield %150, %164 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %118:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6, %arg14 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %136 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %137 = vector.broadcast %136 : f32 to vector<1x1xf32>
        %138 = vector.extract %116[%arg11, 0] : f32 from vector<2x1xf32>
        %139 = vector.broadcast %138 : f32 to vector<1x1xf32>
        %140 = arith.subf %137, %139 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %141 = vector.extract %140[0, 0] : f32 from vector<1x1xf32>
        %142 = vector.insert %141, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %143 = math.exp2 %140 {tiled = true} : vector<1x1xf32>
        %144 = vector.extract %143[0, 0] : f32 from vector<1x1xf32>
        %145 = vector.insert %144, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %146 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %147 = vector.broadcast %146 : f32 to vector<1x1xf32>
        %148 = arith.mulf %147, %143 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %149 = vector.extract %148[0, 0] : f32 from vector<1x1xf32>
        %150 = vector.insert %149, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %142, %145, %150 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %119:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %136 = affine.apply #map18(%arg11)
        %137 = affine.apply #map17(%arg11)
        %138 = vector.extract %117#1[%136, %137] : f32 from vector<2x32xf32>
        %139 = affine.apply #map19(%arg11)
        %140 = vector.extract %117#1[%136, %139] : f32 from vector<2x32xf32>
        %141 = affine.apply #map20(%arg11)
        %142 = vector.extract %117#1[%136, %141] : f32 from vector<2x32xf32>
        %143 = affine.apply #map21(%arg11)
        %144 = vector.extract %117#1[%136, %143] : f32 from vector<2x32xf32>
        %145 = vector.from_elements %138, %140, %142, %144 : vector<1x4xf32>
        %146 = math.exp2 %145 {tiled = true} : vector<1x4xf32>
        %147 = vector.extract %146[0, 0] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %arg12 [%136, %137] : f32 into vector<2x32xf32>
        %149 = vector.extract %146[0, 1] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%136, %139] : f32 into vector<2x32xf32>
        %151 = vector.extract %146[0, 2] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%136, %141] : f32 into vector<2x32xf32>
        %153 = vector.extract %146[0, 3] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %152 [%136, %143] : f32 into vector<2x32xf32>
        %155 = vector.insert %147, %arg13 [%136, %137] : f32 into vector<2x32xf32>
        %156 = vector.insert %149, %155 [%136, %139] : f32 into vector<2x32xf32>
        %157 = vector.insert %151, %156 [%136, %141] : f32 into vector<2x32xf32>
        %158 = vector.insert %153, %157 [%136, %143] : f32 into vector<2x32xf32>
        affine.yield %154, %158 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %120 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %136 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst) -> (f32) {
          %140 = affine.apply #map29(%arg13)
          %141 = vector.extract %119#1[%arg11, %140] : f32 from vector<2x32xf32>
          %142 = arith.addf %arg14, %141 : f32
          %143 = affine.apply #map30(%arg13)
          %144 = vector.extract %119#1[%arg11, %143] : f32 from vector<2x32xf32>
          %145 = arith.addf %142, %144 : f32
          %146 = affine.apply #map31(%arg13)
          %147 = vector.extract %119#1[%arg11, %146] : f32 from vector<2x32xf32>
          %148 = arith.addf %145, %147 : f32
          %149 = affine.apply #map32(%arg13)
          %150 = vector.extract %119#1[%arg11, %149] : f32 from vector<2x32xf32>
          %151 = arith.addf %148, %150 : f32
          affine.yield %151 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %136, %c16_i32, %c64_i32 : f32
        %137 = arith.addf %136, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %137, %c32_i32, %c64_i32 : f32
        %138 = arith.addf %137, %shuffleResult_31 : f32
        %139 = vector.insert %138, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %139 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %121 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %136 = vector.extract %120[%arg11, 0] : f32 from vector<2x1xf32>
        %137 = vector.broadcast %136 : f32 to vector<1x1xf32>
        %138 = vector.extract %118#2[%arg11, 0] : f32 from vector<2x1xf32>
        %139 = vector.broadcast %138 : f32 to vector<1x1xf32>
        %140 = arith.addf %137, %139 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %141 = vector.extract %140[0, 0] : f32 from vector<1x1xf32>
        %142 = vector.insert %141, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %142 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %122 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %136 = affine.apply #map18(%arg11)
        %137 = affine.apply #map17(%arg11)
        %138 = vector.extract %119#0[%136, %137] : f32 from vector<2x32xf32>
        %139 = affine.apply #map19(%arg11)
        %140 = vector.extract %119#0[%136, %139] : f32 from vector<2x32xf32>
        %141 = affine.apply #map20(%arg11)
        %142 = vector.extract %119#0[%136, %141] : f32 from vector<2x32xf32>
        %143 = affine.apply #map21(%arg11)
        %144 = vector.extract %119#0[%136, %143] : f32 from vector<2x32xf32>
        %145 = vector.from_elements %138, %140, %142, %144 : vector<1x4xf32>
        %146 = arith.truncf %145 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %147 = vector.extract %146[0, 0] : f16 from vector<1x4xf16>
        %148 = vector.insert %147, %arg12 [%136, %137] : f16 into vector<2x32xf16>
        %149 = vector.extract %146[0, 1] : f16 from vector<1x4xf16>
        %150 = vector.insert %149, %148 [%136, %139] : f16 into vector<2x32xf16>
        %151 = vector.extract %146[0, 2] : f16 from vector<1x4xf16>
        %152 = vector.insert %151, %150 [%136, %141] : f16 into vector<2x32xf16>
        %153 = vector.extract %146[0, 3] : f16 from vector<1x4xf16>
        %154 = vector.insert %153, %152 [%136, %143] : f16 into vector<2x32xf16>
        affine.yield %154 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %112 : !frisk.async_task
      %123 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %136 = affine.apply #map35(%arg11)[%thread_id_x]
        %137 = affine.apply #map36()[%thread_id_x]
        %138 = frisk.read_matrix_from_shm %alloc_23[%136, %137] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %139 = vector.bitcast %138 : vector<4xi32> to vector<8xf16>
        %140 = vector.extract %139[0] : f16 from vector<8xf16>
        %141 = affine.apply #map37()[%thread_id_x]
        %142 = vector.extract %139[%141] : f16 from vector<8xf16>
        %143 = vector.extract %139[1] : f16 from vector<8xf16>
        %144 = affine.apply #map38()[%thread_id_x]
        %145 = vector.extract %139[%144] : f16 from vector<8xf16>
        %146 = vector.extract %139[2] : f16 from vector<8xf16>
        %147 = affine.apply #map39()[%thread_id_x]
        %148 = vector.extract %139[%147] : f16 from vector<8xf16>
        %149 = vector.extract %139[3] : f16 from vector<8xf16>
        %150 = affine.apply #map40()[%thread_id_x]
        %151 = vector.extract %139[%150] : f16 from vector<8xf16>
        %152 = affine.apply #map29(%arg11)
        %153 = vector.insert %140, %arg12 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply #map30(%arg11)
        %156 = vector.insert %143, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %145, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply #map31(%arg11)
        %159 = vector.insert %146, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %148, %159 [%158, 1] : f16 into vector<32x2xf16>
        %161 = affine.apply #map32(%arg11)
        %162 = vector.insert %149, %160 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        affine.yield %163 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %alloc_25 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_26 = memref.subview %arg2[0, %101, %arg4, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %124 = frisk.async_copy_gm_to_shm %subview_26 to %alloc_25 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %125:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %136 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %183 = affine.apply #map1(%arg11)
          %184 = affine.apply #map29(%arg16)
          %185 = vector.extract %122[%183, %184] : f16 from vector<2x32xf16>
          %186 = affine.apply #map30(%arg16)
          %187 = vector.extract %122[%183, %186] : f16 from vector<2x32xf16>
          %188 = affine.apply #map31(%arg16)
          %189 = vector.extract %122[%183, %188] : f16 from vector<2x32xf16>
          %190 = affine.apply #map32(%arg16)
          %191 = vector.extract %122[%183, %190] : f16 from vector<2x32xf16>
          %192 = vector.from_elements %185, %187, %189, %191 : vector<1x4xf16>
          %193 = affine.apply #map41(%arg11)
          %194 = vector.extract %123[%184, %193] : f16 from vector<32x2xf16>
          %195 = vector.extract %123[%186, %193] : f16 from vector<32x2xf16>
          %196 = vector.extract %123[%188, %193] : f16 from vector<32x2xf16>
          %197 = vector.extract %123[%190, %193] : f16 from vector<32x2xf16>
          %198 = vector.from_elements %194, %195, %196, %197 : vector<4x1xf16>
          %199 = frisk.warp_mma_rr %192, %198, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %199 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %137 = affine.apply #map1(%arg11)
        %138 = affine.apply #map(%arg11)
        %139 = vector.extract %136[0, 0] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %arg12 [%137, %138] : f32 into vector<2x8xf32>
        %141 = affine.apply #map2(%arg11)
        %142 = vector.extract %136[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %140 [%137, %141] : f32 into vector<2x8xf32>
        %144 = affine.apply #map3(%arg11)
        %145 = vector.extract %136[0, 2] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %143 [%137, %144] : f32 into vector<2x8xf32>
        %147 = affine.apply #map4(%arg11)
        %148 = vector.extract %136[0, 3] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %146 [%137, %147] : f32 into vector<2x8xf32>
        %150 = vector.extract %118#1[%137, 0] : f32 from vector<2x1xf32>
        %151 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %183 = vector.broadcast %150 : f32 to vector<1x4xf32>
          affine.yield %183 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg13 [%137, %138] : f32 into vector<2x8xf32>
        %154 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%137, %141] : f32 into vector<2x8xf32>
        %156 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%137, %144] : f32 into vector<2x8xf32>
        %158 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%137, %147] : f32 into vector<2x8xf32>
        %160 = vector.extract %arg5[%137, %138] : f32 from vector<2x8xf32>
        %161 = vector.extract %arg5[%137, %141] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg5[%137, %144] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg5[%137, %147] : f32 from vector<2x8xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.mulf %164, %151 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg14 [%137, %138] : f32 into vector<2x8xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%137, %141] : f32 into vector<2x8xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%137, %144] : f32 into vector<2x8xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%137, %147] : f32 into vector<2x8xf32>
        %174 = arith.addf %165, %136 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg15 [%137, %138] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%137, %141] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%137, %144] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%137, %147] : f32 into vector<2x8xf32>
        affine.yield %149, %159, %173, %182 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %124 : !frisk.async_task
      %126 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %136 = affine.apply #map35(%arg11)[%thread_id_x]
        %137 = affine.apply #map36()[%thread_id_x]
        %138 = frisk.read_matrix_from_shm %alloc_25[%136, %137] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %139 = vector.bitcast %138 : vector<4xi32> to vector<8xf16>
        %140 = vector.extract %139[0] : f16 from vector<8xf16>
        %141 = affine.apply #map37()[%thread_id_x]
        %142 = vector.extract %139[%141] : f16 from vector<8xf16>
        %143 = vector.extract %139[1] : f16 from vector<8xf16>
        %144 = affine.apply #map38()[%thread_id_x]
        %145 = vector.extract %139[%144] : f16 from vector<8xf16>
        %146 = vector.extract %139[2] : f16 from vector<8xf16>
        %147 = affine.apply #map39()[%thread_id_x]
        %148 = vector.extract %139[%147] : f16 from vector<8xf16>
        %149 = vector.extract %139[3] : f16 from vector<8xf16>
        %150 = affine.apply #map40()[%thread_id_x]
        %151 = vector.extract %139[%150] : f16 from vector<8xf16>
        %152 = affine.apply #map29(%arg11)
        %153 = vector.insert %140, %arg12 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply #map30(%arg11)
        %156 = vector.insert %143, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %145, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply #map31(%arg11)
        %159 = vector.insert %146, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %148, %159 [%158, 1] : f16 into vector<32x2xf16>
        %161 = affine.apply #map32(%arg11)
        %162 = vector.insert %149, %160 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        affine.yield %163 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %alloc_27 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_28 = memref.subview %arg2[0, %101, %arg4, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %127 = frisk.async_copy_gm_to_shm %subview_28 to %alloc_27 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %128:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %136 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %183 = affine.apply #map1(%arg11)
          %184 = affine.apply #map29(%arg16)
          %185 = vector.extract %122[%183, %184] : f16 from vector<2x32xf16>
          %186 = affine.apply #map30(%arg16)
          %187 = vector.extract %122[%183, %186] : f16 from vector<2x32xf16>
          %188 = affine.apply #map31(%arg16)
          %189 = vector.extract %122[%183, %188] : f16 from vector<2x32xf16>
          %190 = affine.apply #map32(%arg16)
          %191 = vector.extract %122[%183, %190] : f16 from vector<2x32xf16>
          %192 = vector.from_elements %185, %187, %189, %191 : vector<1x4xf16>
          %193 = affine.apply #map41(%arg11)
          %194 = vector.extract %126[%184, %193] : f16 from vector<32x2xf16>
          %195 = vector.extract %126[%186, %193] : f16 from vector<32x2xf16>
          %196 = vector.extract %126[%188, %193] : f16 from vector<32x2xf16>
          %197 = vector.extract %126[%190, %193] : f16 from vector<32x2xf16>
          %198 = vector.from_elements %194, %195, %196, %197 : vector<4x1xf16>
          %199 = frisk.warp_mma_rr %192, %198, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %199 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %137 = affine.apply #map1(%arg11)
        %138 = affine.apply #map(%arg11)
        %139 = vector.extract %136[0, 0] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %arg12 [%137, %138] : f32 into vector<2x8xf32>
        %141 = affine.apply #map2(%arg11)
        %142 = vector.extract %136[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %140 [%137, %141] : f32 into vector<2x8xf32>
        %144 = affine.apply #map3(%arg11)
        %145 = vector.extract %136[0, 2] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %143 [%137, %144] : f32 into vector<2x8xf32>
        %147 = affine.apply #map4(%arg11)
        %148 = vector.extract %136[0, 3] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %146 [%137, %147] : f32 into vector<2x8xf32>
        %150 = vector.extract %118#1[%137, 0] : f32 from vector<2x1xf32>
        %151 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %183 = vector.broadcast %150 : f32 to vector<1x4xf32>
          affine.yield %183 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg13 [%137, %138] : f32 into vector<2x8xf32>
        %154 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%137, %141] : f32 into vector<2x8xf32>
        %156 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%137, %144] : f32 into vector<2x8xf32>
        %158 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%137, %147] : f32 into vector<2x8xf32>
        %160 = vector.extract %arg8[%137, %138] : f32 from vector<2x8xf32>
        %161 = vector.extract %arg8[%137, %141] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg8[%137, %144] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg8[%137, %147] : f32 from vector<2x8xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.mulf %164, %151 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg14 [%137, %138] : f32 into vector<2x8xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%137, %141] : f32 into vector<2x8xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%137, %144] : f32 into vector<2x8xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%137, %147] : f32 into vector<2x8xf32>
        %174 = arith.addf %165, %136 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg15 [%137, %138] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%137, %141] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%137, %144] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%137, %147] : f32 into vector<2x8xf32>
        affine.yield %149, %159, %173, %182 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %127 : !frisk.async_task
      %129 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %136 = affine.apply #map35(%arg11)[%thread_id_x]
        %137 = affine.apply #map36()[%thread_id_x]
        %138 = frisk.read_matrix_from_shm %alloc_27[%136, %137] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %139 = vector.bitcast %138 : vector<4xi32> to vector<8xf16>
        %140 = vector.extract %139[0] : f16 from vector<8xf16>
        %141 = affine.apply #map37()[%thread_id_x]
        %142 = vector.extract %139[%141] : f16 from vector<8xf16>
        %143 = vector.extract %139[1] : f16 from vector<8xf16>
        %144 = affine.apply #map38()[%thread_id_x]
        %145 = vector.extract %139[%144] : f16 from vector<8xf16>
        %146 = vector.extract %139[2] : f16 from vector<8xf16>
        %147 = affine.apply #map39()[%thread_id_x]
        %148 = vector.extract %139[%147] : f16 from vector<8xf16>
        %149 = vector.extract %139[3] : f16 from vector<8xf16>
        %150 = affine.apply #map40()[%thread_id_x]
        %151 = vector.extract %139[%150] : f16 from vector<8xf16>
        %152 = affine.apply #map29(%arg11)
        %153 = vector.insert %140, %arg12 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply #map30(%arg11)
        %156 = vector.insert %143, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %145, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply #map31(%arg11)
        %159 = vector.insert %146, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %148, %159 [%158, 1] : f16 into vector<32x2xf16>
        %161 = affine.apply #map32(%arg11)
        %162 = vector.insert %149, %160 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        affine.yield %163 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %alloc_29 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_30 = memref.subview %arg2[0, %101, %arg4, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %130 = frisk.async_copy_gm_to_shm %subview_30 to %alloc_29 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %131:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %136 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %183 = affine.apply #map1(%arg11)
          %184 = affine.apply #map29(%arg16)
          %185 = vector.extract %122[%183, %184] : f16 from vector<2x32xf16>
          %186 = affine.apply #map30(%arg16)
          %187 = vector.extract %122[%183, %186] : f16 from vector<2x32xf16>
          %188 = affine.apply #map31(%arg16)
          %189 = vector.extract %122[%183, %188] : f16 from vector<2x32xf16>
          %190 = affine.apply #map32(%arg16)
          %191 = vector.extract %122[%183, %190] : f16 from vector<2x32xf16>
          %192 = vector.from_elements %185, %187, %189, %191 : vector<1x4xf16>
          %193 = affine.apply #map41(%arg11)
          %194 = vector.extract %129[%184, %193] : f16 from vector<32x2xf16>
          %195 = vector.extract %129[%186, %193] : f16 from vector<32x2xf16>
          %196 = vector.extract %129[%188, %193] : f16 from vector<32x2xf16>
          %197 = vector.extract %129[%190, %193] : f16 from vector<32x2xf16>
          %198 = vector.from_elements %194, %195, %196, %197 : vector<4x1xf16>
          %199 = frisk.warp_mma_rr %192, %198, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %199 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %137 = affine.apply #map1(%arg11)
        %138 = affine.apply #map(%arg11)
        %139 = vector.extract %136[0, 0] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %arg12 [%137, %138] : f32 into vector<2x8xf32>
        %141 = affine.apply #map2(%arg11)
        %142 = vector.extract %136[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %140 [%137, %141] : f32 into vector<2x8xf32>
        %144 = affine.apply #map3(%arg11)
        %145 = vector.extract %136[0, 2] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %143 [%137, %144] : f32 into vector<2x8xf32>
        %147 = affine.apply #map4(%arg11)
        %148 = vector.extract %136[0, 3] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %146 [%137, %147] : f32 into vector<2x8xf32>
        %150 = vector.extract %118#1[%137, 0] : f32 from vector<2x1xf32>
        %151 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %183 = vector.broadcast %150 : f32 to vector<1x4xf32>
          affine.yield %183 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg13 [%137, %138] : f32 into vector<2x8xf32>
        %154 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%137, %141] : f32 into vector<2x8xf32>
        %156 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%137, %144] : f32 into vector<2x8xf32>
        %158 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%137, %147] : f32 into vector<2x8xf32>
        %160 = vector.extract %arg9[%137, %138] : f32 from vector<2x8xf32>
        %161 = vector.extract %arg9[%137, %141] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg9[%137, %144] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg9[%137, %147] : f32 from vector<2x8xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.mulf %164, %151 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg14 [%137, %138] : f32 into vector<2x8xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%137, %141] : f32 into vector<2x8xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%137, %144] : f32 into vector<2x8xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%137, %147] : f32 into vector<2x8xf32>
        %174 = arith.addf %165, %136 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg15 [%137, %138] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%137, %141] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%137, %144] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%137, %147] : f32 into vector<2x8xf32>
        affine.yield %149, %159, %173, %182 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %130 : !frisk.async_task
      %132 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %136 = affine.apply #map35(%arg11)[%thread_id_x]
        %137 = affine.apply #map36()[%thread_id_x]
        %138 = frisk.read_matrix_from_shm %alloc_29[%136, %137] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %139 = vector.bitcast %138 : vector<4xi32> to vector<8xf16>
        %140 = vector.extract %139[0] : f16 from vector<8xf16>
        %141 = affine.apply #map37()[%thread_id_x]
        %142 = vector.extract %139[%141] : f16 from vector<8xf16>
        %143 = vector.extract %139[1] : f16 from vector<8xf16>
        %144 = affine.apply #map38()[%thread_id_x]
        %145 = vector.extract %139[%144] : f16 from vector<8xf16>
        %146 = vector.extract %139[2] : f16 from vector<8xf16>
        %147 = affine.apply #map39()[%thread_id_x]
        %148 = vector.extract %139[%147] : f16 from vector<8xf16>
        %149 = vector.extract %139[3] : f16 from vector<8xf16>
        %150 = affine.apply #map40()[%thread_id_x]
        %151 = vector.extract %139[%150] : f16 from vector<8xf16>
        %152 = affine.apply #map29(%arg11)
        %153 = vector.insert %140, %arg12 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %142, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply #map30(%arg11)
        %156 = vector.insert %143, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %145, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply #map31(%arg11)
        %159 = vector.insert %146, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %148, %159 [%158, 1] : f16 into vector<32x2xf16>
        %161 = affine.apply #map32(%arg11)
        %162 = vector.insert %149, %160 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        affine.yield %163 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %133:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %136 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %191 = affine.apply #map1(%arg11)
          %192 = affine.apply #map29(%arg17)
          %193 = vector.extract %122[%191, %192] : f16 from vector<2x32xf16>
          %194 = affine.apply #map30(%arg17)
          %195 = vector.extract %122[%191, %194] : f16 from vector<2x32xf16>
          %196 = affine.apply #map31(%arg17)
          %197 = vector.extract %122[%191, %196] : f16 from vector<2x32xf16>
          %198 = affine.apply #map32(%arg17)
          %199 = vector.extract %122[%191, %198] : f16 from vector<2x32xf16>
          %200 = vector.from_elements %193, %195, %197, %199 : vector<1x4xf16>
          %201 = affine.apply #map41(%arg11)
          %202 = vector.extract %132[%192, %201] : f16 from vector<32x2xf16>
          %203 = vector.extract %132[%194, %201] : f16 from vector<32x2xf16>
          %204 = vector.extract %132[%196, %201] : f16 from vector<32x2xf16>
          %205 = vector.extract %132[%198, %201] : f16 from vector<32x2xf16>
          %206 = vector.from_elements %202, %203, %204, %205 : vector<4x1xf16>
          %207 = frisk.warp_mma_rr %200, %206, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %207 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %137 = affine.apply #map1(%arg11)
        %138 = affine.apply #map(%arg11)
        %139 = vector.extract %136[0, 0] : f32 from vector<1x4xf32>
        %140 = vector.insert %139, %arg12 [%137, %138] : f32 into vector<2x8xf32>
        %141 = affine.apply #map2(%arg11)
        %142 = vector.extract %136[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %140 [%137, %141] : f32 into vector<2x8xf32>
        %144 = affine.apply #map3(%arg11)
        %145 = vector.extract %136[0, 2] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %143 [%137, %144] : f32 into vector<2x8xf32>
        %147 = affine.apply #map4(%arg11)
        %148 = vector.extract %136[0, 3] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %146 [%137, %147] : f32 into vector<2x8xf32>
        %150 = vector.extract %118#1[%137, 0] : f32 from vector<2x1xf32>
        %151 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %191 = vector.broadcast %150 : f32 to vector<1x4xf32>
          affine.yield %191 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg13 [%137, %138] : f32 into vector<2x8xf32>
        %154 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%137, %141] : f32 into vector<2x8xf32>
        %156 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%137, %144] : f32 into vector<2x8xf32>
        %158 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%137, %147] : f32 into vector<2x8xf32>
        %160 = vector.extract %arg10[%137, %138] : f32 from vector<2x8xf32>
        %161 = vector.extract %arg10[%137, %141] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg10[%137, %144] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg10[%137, %147] : f32 from vector<2x8xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.mulf %164, %151 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg14 [%137, %138] : f32 into vector<2x8xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%137, %141] : f32 into vector<2x8xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%137, %144] : f32 into vector<2x8xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%137, %147] : f32 into vector<2x8xf32>
        %174 = arith.addf %165, %136 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg15 [%137, %138] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%137, %141] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%137, %144] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%137, %147] : f32 into vector<2x8xf32>
        %183 = vector.extract %125#3[%137, %138] : f32 from vector<2x8xf32>
        %184 = vector.extract %125#3[%137, %141] : f32 from vector<2x8xf32>
        %185 = vector.extract %125#3[%137, %144] : f32 from vector<2x8xf32>
        %186 = vector.extract %125#3[%137, %147] : f32 from vector<2x8xf32>
        %187 = vector.insert %183, %arg16 [%137, %138] : f32 into vector<2x8xf32>
        %188 = vector.insert %184, %187 [%137, %141] : f32 into vector<2x8xf32>
        %189 = vector.insert %185, %188 [%137, %144] : f32 into vector<2x8xf32>
        %190 = vector.insert %186, %189 [%137, %147] : f32 into vector<2x8xf32>
        affine.yield %149, %159, %173, %182, %190 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %134:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %136 = vector.extract %121[%arg11, 0] : f32 from vector<2x1xf32>
        %137 = vector.insert %136, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %138 = vector.extract %116[%arg11, 0] : f32 from vector<2x1xf32>
        %139 = vector.insert %138, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %137, %139 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %135:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %136 = affine.apply #map1(%arg11)
        %137 = affine.apply #map(%arg11)
        %138 = vector.extract %128#3[%136, %137] : f32 from vector<2x8xf32>
        %139 = affine.apply #map2(%arg11)
        %140 = vector.extract %128#3[%136, %139] : f32 from vector<2x8xf32>
        %141 = affine.apply #map3(%arg11)
        %142 = vector.extract %128#3[%136, %141] : f32 from vector<2x8xf32>
        %143 = affine.apply #map4(%arg11)
        %144 = vector.extract %128#3[%136, %143] : f32 from vector<2x8xf32>
        %145 = vector.insert %138, %arg12 [%136, %137] : f32 into vector<2x8xf32>
        %146 = vector.insert %140, %145 [%136, %139] : f32 into vector<2x8xf32>
        %147 = vector.insert %142, %146 [%136, %141] : f32 into vector<2x8xf32>
        %148 = vector.insert %144, %147 [%136, %143] : f32 into vector<2x8xf32>
        %149 = vector.extract %131#3[%136, %137] : f32 from vector<2x8xf32>
        %150 = vector.extract %131#3[%136, %139] : f32 from vector<2x8xf32>
        %151 = vector.extract %131#3[%136, %141] : f32 from vector<2x8xf32>
        %152 = vector.extract %131#3[%136, %143] : f32 from vector<2x8xf32>
        %153 = vector.insert %149, %arg13 [%136, %137] : f32 into vector<2x8xf32>
        %154 = vector.insert %150, %153 [%136, %139] : f32 into vector<2x8xf32>
        %155 = vector.insert %151, %154 [%136, %141] : f32 into vector<2x8xf32>
        %156 = vector.insert %152, %155 [%136, %143] : f32 into vector<2x8xf32>
        %157 = vector.extract %133#3[%136, %137] : f32 from vector<2x8xf32>
        %158 = vector.extract %133#3[%136, %139] : f32 from vector<2x8xf32>
        %159 = vector.extract %133#3[%136, %141] : f32 from vector<2x8xf32>
        %160 = vector.extract %133#3[%136, %143] : f32 from vector<2x8xf32>
        %161 = vector.insert %157, %arg14 [%136, %137] : f32 into vector<2x8xf32>
        %162 = vector.insert %158, %161 [%136, %139] : f32 into vector<2x8xf32>
        %163 = vector.insert %159, %162 [%136, %141] : f32 into vector<2x8xf32>
        %164 = vector.insert %160, %163 [%136, %143] : f32 into vector<2x8xf32>
        affine.yield %148, %156, %164 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %133#4, %134#0, %134#1, %135#0, %135#1, %135#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %85 = arith.index_cast %thread_id_x : index to i32
    %86 = arith.andi %85, %c16_i32 : i32
    %87 = arith.cmpi ne, %86, %c0_i32 : i32
    %88 = arith.andi %85, %c32_i32 : i32
    %89 = arith.cmpi ne, %88, %c0_i32 : i32
    %90 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %100 = affine.apply #map(%arg4)
      %101 = affine.apply #map1(%arg4)
      %102 = vector.extract %84#5[%101, %100] : f32 from vector<2x8xf32>
      %103 = affine.apply #map2(%arg4)
      %104 = vector.extract %84#5[%101, %103] : f32 from vector<2x8xf32>
      %105 = affine.apply #map3(%arg4)
      %106 = vector.extract %84#5[%101, %105] : f32 from vector<2x8xf32>
      %107 = affine.apply #map4(%arg4)
      %108 = vector.extract %84#5[%101, %107] : f32 from vector<2x8xf32>
      %109 = arith.select %87, %102, %104 : f32
      %shuffleResult, %valid = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %110 = arith.select %87, %shuffleResult, %102 : f32
      %111 = arith.select %87, %104, %shuffleResult : f32
      %112 = arith.select %87, %106, %108 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %112, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %113 = arith.select %87, %shuffleResult_15, %106 : f32
      %114 = arith.select %87, %108, %shuffleResult_15 : f32
      %115 = arith.select %89, %110, %113 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %89, %shuffleResult_17, %110 : f32
      %117 = arith.select %89, %113, %shuffleResult_17 : f32
      %118 = arith.select %89, %111, %114 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %118, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %119 = arith.select %89, %shuffleResult_19, %111 : f32
      %120 = arith.select %89, %114, %shuffleResult_19 : f32
      %121 = vector.insert %116, %arg5 [%101, %100] : f32 into vector<2x8xf32>
      %122 = vector.insert %119, %121 [%101, %103] : f32 into vector<2x8xf32>
      %123 = vector.insert %117, %122 [%101, %105] : f32 into vector<2x8xf32>
      %124 = vector.insert %120, %123 [%101, %107] : f32 into vector<2x8xf32>
      affine.yield %124 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %91 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %100 = affine.apply #map(%arg4)
      %101 = affine.apply #map1(%arg4)
      %102 = vector.extract %84#4[%101, %100] : f32 from vector<2x8xf32>
      %103 = affine.apply #map2(%arg4)
      %104 = vector.extract %84#4[%101, %103] : f32 from vector<2x8xf32>
      %105 = affine.apply #map3(%arg4)
      %106 = vector.extract %84#4[%101, %105] : f32 from vector<2x8xf32>
      %107 = affine.apply #map4(%arg4)
      %108 = vector.extract %84#4[%101, %107] : f32 from vector<2x8xf32>
      %109 = arith.select %87, %102, %104 : f32
      %shuffleResult, %valid = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %110 = arith.select %87, %shuffleResult, %102 : f32
      %111 = arith.select %87, %104, %shuffleResult : f32
      %112 = arith.select %87, %106, %108 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %112, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %113 = arith.select %87, %shuffleResult_15, %106 : f32
      %114 = arith.select %87, %108, %shuffleResult_15 : f32
      %115 = arith.select %89, %110, %113 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %89, %shuffleResult_17, %110 : f32
      %117 = arith.select %89, %113, %shuffleResult_17 : f32
      %118 = arith.select %89, %111, %114 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %118, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %119 = arith.select %89, %shuffleResult_19, %111 : f32
      %120 = arith.select %89, %114, %shuffleResult_19 : f32
      %121 = vector.insert %116, %arg5 [%101, %100] : f32 into vector<2x8xf32>
      %122 = vector.insert %119, %121 [%101, %103] : f32 into vector<2x8xf32>
      %123 = vector.insert %117, %122 [%101, %105] : f32 into vector<2x8xf32>
      %124 = vector.insert %120, %123 [%101, %107] : f32 into vector<2x8xf32>
      affine.yield %124 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %92 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %100 = affine.apply #map(%arg4)
      %101 = affine.apply #map1(%arg4)
      %102 = vector.extract %84#3[%101, %100] : f32 from vector<2x8xf32>
      %103 = affine.apply #map2(%arg4)
      %104 = vector.extract %84#3[%101, %103] : f32 from vector<2x8xf32>
      %105 = affine.apply #map3(%arg4)
      %106 = vector.extract %84#3[%101, %105] : f32 from vector<2x8xf32>
      %107 = affine.apply #map4(%arg4)
      %108 = vector.extract %84#3[%101, %107] : f32 from vector<2x8xf32>
      %109 = arith.select %87, %102, %104 : f32
      %shuffleResult, %valid = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %110 = arith.select %87, %shuffleResult, %102 : f32
      %111 = arith.select %87, %104, %shuffleResult : f32
      %112 = arith.select %87, %106, %108 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %112, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %113 = arith.select %87, %shuffleResult_15, %106 : f32
      %114 = arith.select %87, %108, %shuffleResult_15 : f32
      %115 = arith.select %89, %110, %113 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %89, %shuffleResult_17, %110 : f32
      %117 = arith.select %89, %113, %shuffleResult_17 : f32
      %118 = arith.select %89, %111, %114 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %118, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %119 = arith.select %89, %shuffleResult_19, %111 : f32
      %120 = arith.select %89, %114, %shuffleResult_19 : f32
      %121 = vector.insert %116, %arg5 [%101, %100] : f32 into vector<2x8xf32>
      %122 = vector.insert %119, %121 [%101, %103] : f32 into vector<2x8xf32>
      %123 = vector.insert %117, %122 [%101, %105] : f32 into vector<2x8xf32>
      %124 = vector.insert %120, %123 [%101, %107] : f32 into vector<2x8xf32>
      affine.yield %124 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %93 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %100 = affine.apply #map(%arg4)
      %101 = affine.apply #map1(%arg4)
      %102 = vector.extract %84#0[%101, %100] : f32 from vector<2x8xf32>
      %103 = affine.apply #map2(%arg4)
      %104 = vector.extract %84#0[%101, %103] : f32 from vector<2x8xf32>
      %105 = affine.apply #map3(%arg4)
      %106 = vector.extract %84#0[%101, %105] : f32 from vector<2x8xf32>
      %107 = affine.apply #map4(%arg4)
      %108 = vector.extract %84#0[%101, %107] : f32 from vector<2x8xf32>
      %109 = arith.select %87, %102, %104 : f32
      %shuffleResult, %valid = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %110 = arith.select %87, %shuffleResult, %102 : f32
      %111 = arith.select %87, %104, %shuffleResult : f32
      %112 = arith.select %87, %106, %108 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %112, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %113 = arith.select %87, %shuffleResult_15, %106 : f32
      %114 = arith.select %87, %108, %shuffleResult_15 : f32
      %115 = arith.select %89, %110, %113 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %89, %shuffleResult_17, %110 : f32
      %117 = arith.select %89, %113, %shuffleResult_17 : f32
      %118 = arith.select %89, %111, %114 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %118, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %119 = arith.select %89, %shuffleResult_19, %111 : f32
      %120 = arith.select %89, %114, %shuffleResult_19 : f32
      %121 = vector.insert %116, %arg5 [%101, %100] : f32 into vector<2x8xf32>
      %122 = vector.insert %119, %121 [%101, %103] : f32 into vector<2x8xf32>
      %123 = vector.insert %117, %122 [%101, %105] : f32 into vector<2x8xf32>
      %124 = vector.insert %120, %123 [%101, %107] : f32 into vector<2x8xf32>
      affine.yield %124 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %94:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %100 = affine.apply #map1(%arg4)
      %101 = vector.extract %84#1[%100, 0] : f32 from vector<2x1xf32>
      %102 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %142 = vector.broadcast %101 : f32 to vector<1x4xf32>
        affine.yield %142 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %103 = affine.apply #map(%arg4)
      %104 = vector.extract %102[0, 0] : f32 from vector<1x4xf32>
      %105 = vector.insert %104, %arg5 [%100, %103] : f32 into vector<2x8xf32>
      %106 = affine.apply #map2(%arg4)
      %107 = vector.extract %102[0, 1] : f32 from vector<1x4xf32>
      %108 = vector.insert %107, %105 [%100, %106] : f32 into vector<2x8xf32>
      %109 = affine.apply #map3(%arg4)
      %110 = vector.extract %102[0, 2] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %108 [%100, %109] : f32 into vector<2x8xf32>
      %112 = affine.apply #map4(%arg4)
      %113 = vector.extract %102[0, 3] : f32 from vector<1x4xf32>
      %114 = vector.insert %113, %111 [%100, %112] : f32 into vector<2x8xf32>
      %115 = vector.extract %93[%100, %103] : f32 from vector<2x8xf32>
      %116 = vector.extract %93[%100, %106] : f32 from vector<2x8xf32>
      %117 = vector.extract %93[%100, %109] : f32 from vector<2x8xf32>
      %118 = vector.extract %93[%100, %112] : f32 from vector<2x8xf32>
      %119 = vector.from_elements %115, %116, %117, %118 : vector<1x4xf32>
      %120 = arith.divf %119, %102 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %121 = vector.extract %120[0, 0] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %arg6 [%100, %103] : f32 into vector<2x8xf32>
      %123 = vector.extract %120[0, 1] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%100, %106] : f32 into vector<2x8xf32>
      %125 = vector.extract %120[0, 2] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%100, %109] : f32 into vector<2x8xf32>
      %127 = vector.extract %120[0, 3] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%100, %112] : f32 into vector<2x8xf32>
      %129 = arith.truncf %120 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %130 = vector.extract %129[0, 0] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %arg7 [%100, %103] : f16 into vector<2x8xf16>
      %132 = vector.extract %129[0, 1] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%100, %106] : f16 into vector<2x8xf16>
      %134 = vector.extract %129[0, 2] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%100, %109] : f16 into vector<2x8xf16>
      %136 = vector.extract %129[0, 3] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%100, %112] : f16 into vector<2x8xf16>
      %138 = vector.insert %130, %arg8 [%100, %103] : f16 into vector<2x8xf16>
      %139 = vector.insert %132, %138 [%100, %106] : f16 into vector<2x8xf16>
      %140 = vector.insert %134, %139 [%100, %109] : f16 into vector<2x8xf16>
      %141 = vector.insert %136, %140 [%100, %112] : f16 into vector<2x8xf16>
      affine.yield %114, %128, %137, %141 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %95 = affine.apply #map42()[%thread_id_x]
    %96 = arith.cmpi ult, %95, %c4 : index
    scf.if %96 {
      affine.for %arg4 = 0 to 4 {
        %100 = affine.apply #map1(%arg4)
        %101 = affine.apply #map(%arg4)
        %102 = vector.extract %94#3[%100, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply #map2(%arg4)
        %104 = vector.extract %94#3[%100, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply #map3(%arg4)
        %106 = vector.extract %94#3[%100, %105] : f16 from vector<2x8xf16>
        %107 = affine.apply #map4(%arg4)
        %108 = vector.extract %94#3[%100, %107] : f16 from vector<2x8xf16>
        affine.store %102, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %108, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %97:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %100 = affine.apply #map1(%arg4)
      %101 = vector.extract %84#1[%100, 0] : f32 from vector<2x1xf32>
      %102 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %142 = vector.broadcast %101 : f32 to vector<1x4xf32>
        affine.yield %142 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %103 = affine.apply #map(%arg4)
      %104 = vector.extract %102[0, 0] : f32 from vector<1x4xf32>
      %105 = vector.insert %104, %arg5 [%100, %103] : f32 into vector<2x8xf32>
      %106 = affine.apply #map2(%arg4)
      %107 = vector.extract %102[0, 1] : f32 from vector<1x4xf32>
      %108 = vector.insert %107, %105 [%100, %106] : f32 into vector<2x8xf32>
      %109 = affine.apply #map3(%arg4)
      %110 = vector.extract %102[0, 2] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %108 [%100, %109] : f32 into vector<2x8xf32>
      %112 = affine.apply #map4(%arg4)
      %113 = vector.extract %102[0, 3] : f32 from vector<1x4xf32>
      %114 = vector.insert %113, %111 [%100, %112] : f32 into vector<2x8xf32>
      %115 = vector.extract %92[%100, %103] : f32 from vector<2x8xf32>
      %116 = vector.extract %92[%100, %106] : f32 from vector<2x8xf32>
      %117 = vector.extract %92[%100, %109] : f32 from vector<2x8xf32>
      %118 = vector.extract %92[%100, %112] : f32 from vector<2x8xf32>
      %119 = vector.from_elements %115, %116, %117, %118 : vector<1x4xf32>
      %120 = arith.divf %119, %102 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %121 = vector.extract %120[0, 0] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %arg6 [%100, %103] : f32 into vector<2x8xf32>
      %123 = vector.extract %120[0, 1] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%100, %106] : f32 into vector<2x8xf32>
      %125 = vector.extract %120[0, 2] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%100, %109] : f32 into vector<2x8xf32>
      %127 = vector.extract %120[0, 3] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%100, %112] : f32 into vector<2x8xf32>
      %129 = arith.truncf %120 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %130 = vector.extract %129[0, 0] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %arg7 [%100, %103] : f16 into vector<2x8xf16>
      %132 = vector.extract %129[0, 1] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%100, %106] : f16 into vector<2x8xf16>
      %134 = vector.extract %129[0, 2] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%100, %109] : f16 into vector<2x8xf16>
      %136 = vector.extract %129[0, 3] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%100, %112] : f16 into vector<2x8xf16>
      %138 = vector.insert %130, %arg8 [%100, %103] : f16 into vector<2x8xf16>
      %139 = vector.insert %132, %138 [%100, %106] : f16 into vector<2x8xf16>
      %140 = vector.insert %134, %139 [%100, %109] : f16 into vector<2x8xf16>
      %141 = vector.insert %136, %140 [%100, %112] : f16 into vector<2x8xf16>
      affine.yield %114, %128, %137, %141 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %96 {
      affine.for %arg4 = 0 to 4 {
        %100 = affine.apply #map1(%arg4)
        %101 = affine.apply #map(%arg4)
        %102 = vector.extract %97#3[%100, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply #map2(%arg4)
        %104 = vector.extract %97#3[%100, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply #map3(%arg4)
        %106 = vector.extract %97#3[%100, %105] : f16 from vector<2x8xf16>
        %107 = affine.apply #map4(%arg4)
        %108 = vector.extract %97#3[%100, %107] : f16 from vector<2x8xf16>
        affine.store %102, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %108, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %98:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %100 = affine.apply #map1(%arg4)
      %101 = vector.extract %84#1[%100, 0] : f32 from vector<2x1xf32>
      %102 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %142 = vector.broadcast %101 : f32 to vector<1x4xf32>
        affine.yield %142 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %103 = affine.apply #map(%arg4)
      %104 = vector.extract %102[0, 0] : f32 from vector<1x4xf32>
      %105 = vector.insert %104, %arg5 [%100, %103] : f32 into vector<2x8xf32>
      %106 = affine.apply #map2(%arg4)
      %107 = vector.extract %102[0, 1] : f32 from vector<1x4xf32>
      %108 = vector.insert %107, %105 [%100, %106] : f32 into vector<2x8xf32>
      %109 = affine.apply #map3(%arg4)
      %110 = vector.extract %102[0, 2] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %108 [%100, %109] : f32 into vector<2x8xf32>
      %112 = affine.apply #map4(%arg4)
      %113 = vector.extract %102[0, 3] : f32 from vector<1x4xf32>
      %114 = vector.insert %113, %111 [%100, %112] : f32 into vector<2x8xf32>
      %115 = vector.extract %91[%100, %103] : f32 from vector<2x8xf32>
      %116 = vector.extract %91[%100, %106] : f32 from vector<2x8xf32>
      %117 = vector.extract %91[%100, %109] : f32 from vector<2x8xf32>
      %118 = vector.extract %91[%100, %112] : f32 from vector<2x8xf32>
      %119 = vector.from_elements %115, %116, %117, %118 : vector<1x4xf32>
      %120 = arith.divf %119, %102 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %121 = vector.extract %120[0, 0] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %arg6 [%100, %103] : f32 into vector<2x8xf32>
      %123 = vector.extract %120[0, 1] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%100, %106] : f32 into vector<2x8xf32>
      %125 = vector.extract %120[0, 2] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%100, %109] : f32 into vector<2x8xf32>
      %127 = vector.extract %120[0, 3] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%100, %112] : f32 into vector<2x8xf32>
      %129 = arith.truncf %120 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %130 = vector.extract %129[0, 0] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %arg7 [%100, %103] : f16 into vector<2x8xf16>
      %132 = vector.extract %129[0, 1] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%100, %106] : f16 into vector<2x8xf16>
      %134 = vector.extract %129[0, 2] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%100, %109] : f16 into vector<2x8xf16>
      %136 = vector.extract %129[0, 3] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%100, %112] : f16 into vector<2x8xf16>
      %138 = vector.insert %130, %arg8 [%100, %103] : f16 into vector<2x8xf16>
      %139 = vector.insert %132, %138 [%100, %106] : f16 into vector<2x8xf16>
      %140 = vector.insert %134, %139 [%100, %109] : f16 into vector<2x8xf16>
      %141 = vector.insert %136, %140 [%100, %112] : f16 into vector<2x8xf16>
      affine.yield %114, %128, %137, %141 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %96 {
      affine.for %arg4 = 0 to 4 {
        %100 = affine.apply #map1(%arg4)
        %101 = affine.apply #map(%arg4)
        %102 = vector.extract %98#3[%100, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply #map2(%arg4)
        %104 = vector.extract %98#3[%100, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply #map3(%arg4)
        %106 = vector.extract %98#3[%100, %105] : f16 from vector<2x8xf16>
        %107 = affine.apply #map4(%arg4)
        %108 = vector.extract %98#3[%100, %107] : f16 from vector<2x8xf16>
        affine.store %102, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %108, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %99:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %100 = affine.apply #map1(%arg4)
      %101 = vector.extract %84#1[%100, 0] : f32 from vector<2x1xf32>
      %102 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %142 = vector.broadcast %101 : f32 to vector<1x4xf32>
        affine.yield %142 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %103 = affine.apply #map(%arg4)
      %104 = vector.extract %102[0, 0] : f32 from vector<1x4xf32>
      %105 = vector.insert %104, %arg5 [%100, %103] : f32 into vector<2x8xf32>
      %106 = affine.apply #map2(%arg4)
      %107 = vector.extract %102[0, 1] : f32 from vector<1x4xf32>
      %108 = vector.insert %107, %105 [%100, %106] : f32 into vector<2x8xf32>
      %109 = affine.apply #map3(%arg4)
      %110 = vector.extract %102[0, 2] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %108 [%100, %109] : f32 into vector<2x8xf32>
      %112 = affine.apply #map4(%arg4)
      %113 = vector.extract %102[0, 3] : f32 from vector<1x4xf32>
      %114 = vector.insert %113, %111 [%100, %112] : f32 into vector<2x8xf32>
      %115 = vector.extract %90[%100, %103] : f32 from vector<2x8xf32>
      %116 = vector.extract %90[%100, %106] : f32 from vector<2x8xf32>
      %117 = vector.extract %90[%100, %109] : f32 from vector<2x8xf32>
      %118 = vector.extract %90[%100, %112] : f32 from vector<2x8xf32>
      %119 = vector.from_elements %115, %116, %117, %118 : vector<1x4xf32>
      %120 = arith.divf %119, %102 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %121 = vector.extract %120[0, 0] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %arg6 [%100, %103] : f32 into vector<2x8xf32>
      %123 = vector.extract %120[0, 1] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%100, %106] : f32 into vector<2x8xf32>
      %125 = vector.extract %120[0, 2] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%100, %109] : f32 into vector<2x8xf32>
      %127 = vector.extract %120[0, 3] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%100, %112] : f32 into vector<2x8xf32>
      %129 = arith.truncf %120 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %130 = vector.extract %129[0, 0] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %arg7 [%100, %103] : f16 into vector<2x8xf16>
      %132 = vector.extract %129[0, 1] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%100, %106] : f16 into vector<2x8xf16>
      %134 = vector.extract %129[0, 2] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%100, %109] : f16 into vector<2x8xf16>
      %136 = vector.extract %129[0, 3] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%100, %112] : f16 into vector<2x8xf16>
      %138 = vector.insert %130, %arg8 [%100, %103] : f16 into vector<2x8xf16>
      %139 = vector.insert %132, %138 [%100, %106] : f16 into vector<2x8xf16>
      %140 = vector.insert %134, %139 [%100, %109] : f16 into vector<2x8xf16>
      %141 = vector.insert %136, %140 [%100, %112] : f16 into vector<2x8xf16>
      affine.yield %114, %128, %137, %141 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %96 {
      affine.for %arg4 = 0 to 4 {
        %100 = affine.apply #map1(%arg4)
        %101 = affine.apply #map(%arg4)
        %102 = vector.extract %99#3[%100, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply #map2(%arg4)
        %104 = vector.extract %99#3[%100, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply #map3(%arg4)
        %106 = vector.extract %99#3[%100, %105] : f16 from vector<2x8xf16>
        %107 = affine.apply #map4(%arg4)
        %108 = vector.extract %99#3[%100, %107] : f16 from vector<2x8xf16>
        affine.store %102, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %108, %arg3[0, (symbol(%block_id_z) * 32 + symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

