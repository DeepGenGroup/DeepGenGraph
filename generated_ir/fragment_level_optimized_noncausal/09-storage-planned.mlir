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
    %c8192 = arith.constant 8192 : index
    %cst = arith.constant 0.127531052 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %cst_6 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() alignment = 16 {frisk.shm_pool, tiled = true} : memref<16384xi8, 3>
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
    %8 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 + s2 * 32 + s0 floordiv 32) mod 32)>()[%6, %7, %block_id_z]
    %9 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%6]
    %subview = memref.subview %arg0[0, %8, %9, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %view = memref.view %alloc[%c0][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    frisk.async_copy_gm_to_shm %subview to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %10 = arith.truncf %cst : f32 to f16
    %11:3 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1, %arg6 = %cst_1, %arg7 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %98 = vector.insert %cst_5, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %99 = vector.insert %cst_0, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %100 = vector.insert %cst_0, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %98, %99, %100 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %12 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8) -> (vector<2x8xf32>) {
      %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %99 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %100 = vector.insert %cst_5, %arg5 [%99, %98] : f32 into vector<2x8xf32>
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %102 = vector.insert %cst_5, %100 [%99, %101] : f32 into vector<2x8xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %104 = vector.insert %cst_5, %102 [%99, %103] : f32 into vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %106 = vector.insert %cst_5, %104 [%99, %105] : f32 into vector<2x8xf32>
      affine.yield %106 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %13 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1) -> (vector<2x1xf32>) {
      %98 = vector.insert %cst_5, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %98 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    %thread_id_x = gpu.thread_id x
    %14 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128)>()[%thread_id_x]
    frisk.wait_buffer_ready %view : memref<4096xf16, 3>
    %15 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%6, %7]
    %subview_9 = memref.subview %arg0[0, %15, %9, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %view_10 = memref.view %alloc[%c8192][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    frisk.async_copy_gm_to_shm %subview_9 to %view_10 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
    %16 = memref.load %view[%14] : memref<4096xf16, 3>
    %17 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 1)>()[%thread_id_x]
    %18 = memref.load %view[%17] : memref<4096xf16, 3>
    %19 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2)>()[%thread_id_x]
    %20 = memref.load %view[%19] : memref<4096xf16, 3>
    %21 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 3)>()[%thread_id_x]
    %22 = memref.load %view[%21] : memref<4096xf16, 3>
    %23 = vector.from_elements %16, %18, %20, %22 : vector<1x4xf16>
    %24 = vector.broadcast %10 : f16 to vector<1x4xf16>
    %25:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %23) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %99 = memref.load %view[%98] : memref<4096xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %101 = memref.load %view[%100] : memref<4096xf16, 3>
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %103 = memref.load %view[%102] : memref<4096xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %105 = memref.load %view[%104] : memref<4096xf16, 3>
      %106 = vector.from_elements %99, %101, %103, %105 : vector<1x4xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %110 = vector.insert %109, %arg5 [%107, %108] : f16 into vector<2x8xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %110 [%107, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%107, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%107, %117] : f16 into vector<2x8xf16>
      %120 = arith.mulf %arg7, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %121 = vector.extract %120[0, 0] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %arg6 [%107, %108] : f16 into vector<2x8xf16>
      %123 = vector.extract %120[0, 1] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%107, %111] : f16 into vector<2x8xf16>
      %125 = vector.extract %120[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%107, %114] : f16 into vector<2x8xf16>
      %127 = vector.extract %120[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%107, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %128, %106 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %26 = arith.mulf %25#2, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %27 = vector.extract %26[0, 0] : f16 from vector<1x4xf16>
    %28 = vector.insert %27, %25#1 [1, 4] : f16 into vector<2x8xf16>
    %29 = vector.extract %26[0, 1] : f16 from vector<1x4xf16>
    %30 = vector.insert %29, %28 [1, 5] : f16 into vector<2x8xf16>
    %31 = vector.extract %26[0, 2] : f16 from vector<1x4xf16>
    %32 = vector.insert %31, %30 [1, 6] : f16 into vector<2x8xf16>
    %33 = vector.extract %26[0, 3] : f16 from vector<1x4xf16>
    %34 = vector.insert %33, %32 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %view_10 : memref<4096xf16, 3>
    %subview_11 = memref.subview %arg0[0, %15, %9, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview_11 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
    %35 = memref.load %view_10[%14] : memref<4096xf16, 3>
    %36 = memref.load %view_10[%17] : memref<4096xf16, 3>
    %37 = memref.load %view_10[%19] : memref<4096xf16, 3>
    %38 = memref.load %view_10[%21] : memref<4096xf16, 3>
    %39 = vector.from_elements %35, %36, %37, %38 : vector<1x4xf16>
    %40:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %39) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %99 = memref.load %view_10[%98] : memref<4096xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %101 = memref.load %view_10[%100] : memref<4096xf16, 3>
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %103 = memref.load %view_10[%102] : memref<4096xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %105 = memref.load %view_10[%104] : memref<4096xf16, 3>
      %106 = vector.from_elements %99, %101, %103, %105 : vector<1x4xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %110 = vector.insert %109, %arg5 [%107, %108] : f16 into vector<2x8xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %110 [%107, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%107, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%107, %117] : f16 into vector<2x8xf16>
      %120 = arith.mulf %arg7, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %121 = vector.extract %120[0, 0] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %arg6 [%107, %108] : f16 into vector<2x8xf16>
      %123 = vector.extract %120[0, 1] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%107, %111] : f16 into vector<2x8xf16>
      %125 = vector.extract %120[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%107, %114] : f16 into vector<2x8xf16>
      %127 = vector.extract %120[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%107, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %128, %106 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %41 = arith.mulf %40#2, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %42 = vector.extract %41[0, 0] : f16 from vector<1x4xf16>
    %43 = vector.insert %42, %40#1 [1, 4] : f16 into vector<2x8xf16>
    %44 = vector.extract %41[0, 1] : f16 from vector<1x4xf16>
    %45 = vector.insert %44, %43 [1, 5] : f16 into vector<2x8xf16>
    %46 = vector.extract %41[0, 2] : f16 from vector<1x4xf16>
    %47 = vector.insert %46, %45 [1, 6] : f16 into vector<2x8xf16>
    %48 = vector.extract %41[0, 3] : f16 from vector<1x4xf16>
    %49 = vector.insert %48, %47 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %view : memref<4096xf16, 3>
    %subview_12 = memref.subview %arg0[0, %15, %9, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview_12 to %view_10 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
    %50 = memref.load %view[%14] : memref<4096xf16, 3>
    %51 = memref.load %view[%17] : memref<4096xf16, 3>
    %52 = memref.load %view[%19] : memref<4096xf16, 3>
    %53 = memref.load %view[%21] : memref<4096xf16, 3>
    %54 = vector.from_elements %50, %51, %52, %53 : vector<1x4xf16>
    %55:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %54) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %99 = memref.load %view[%98] : memref<4096xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %101 = memref.load %view[%100] : memref<4096xf16, 3>
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %103 = memref.load %view[%102] : memref<4096xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %105 = memref.load %view[%104] : memref<4096xf16, 3>
      %106 = vector.from_elements %99, %101, %103, %105 : vector<1x4xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %110 = vector.insert %109, %arg5 [%107, %108] : f16 into vector<2x8xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %110 [%107, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%107, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%107, %117] : f16 into vector<2x8xf16>
      %120 = arith.mulf %arg7, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %121 = vector.extract %120[0, 0] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %arg6 [%107, %108] : f16 into vector<2x8xf16>
      %123 = vector.extract %120[0, 1] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%107, %111] : f16 into vector<2x8xf16>
      %125 = vector.extract %120[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%107, %114] : f16 into vector<2x8xf16>
      %127 = vector.extract %120[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%107, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %128, %106 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %56 = arith.mulf %55#2, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %57 = vector.extract %56[0, 0] : f16 from vector<1x4xf16>
    %58 = vector.insert %57, %55#1 [1, 4] : f16 into vector<2x8xf16>
    %59 = vector.extract %56[0, 1] : f16 from vector<1x4xf16>
    %60 = vector.insert %59, %58 [1, 5] : f16 into vector<2x8xf16>
    %61 = vector.extract %56[0, 2] : f16 from vector<1x4xf16>
    %62 = vector.insert %61, %60 [1, 6] : f16 into vector<2x8xf16>
    %63 = vector.extract %56[0, 3] : f16 from vector<1x4xf16>
    %64 = vector.insert %63, %62 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %view_10 : memref<4096xf16, 3>
    %65 = memref.load %view_10[%14] : memref<4096xf16, 3>
    %66 = memref.load %view_10[%17] : memref<4096xf16, 3>
    %67 = memref.load %view_10[%19] : memref<4096xf16, 3>
    %68 = memref.load %view_10[%21] : memref<4096xf16, 3>
    %69 = vector.from_elements %65, %66, %67, %68 : vector<1x4xf16>
    %70:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %69) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %99 = memref.load %view_10[%98] : memref<4096xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %101 = memref.load %view_10[%100] : memref<4096xf16, 3>
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %103 = memref.load %view_10[%102] : memref<4096xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %105 = memref.load %view_10[%104] : memref<4096xf16, 3>
      %106 = vector.from_elements %99, %101, %103, %105 : vector<1x4xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %110 = vector.insert %109, %arg5 [%107, %108] : f16 into vector<2x8xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %110 [%107, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%107, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%107, %117] : f16 into vector<2x8xf16>
      %120 = arith.mulf %arg7, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %121 = vector.extract %120[0, 0] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %arg6 [%107, %108] : f16 into vector<2x8xf16>
      %123 = vector.extract %120[0, 1] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%107, %111] : f16 into vector<2x8xf16>
      %125 = vector.extract %120[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%107, %114] : f16 into vector<2x8xf16>
      %127 = vector.extract %120[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%107, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %128, %106 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %71 = arith.mulf %70#2, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %72 = vector.extract %71[0, 0] : f16 from vector<1x4xf16>
    %73 = vector.insert %72, %70#1 [1, 4] : f16 into vector<2x8xf16>
    %74 = vector.extract %71[0, 1] : f16 from vector<1x4xf16>
    %75 = vector.insert %74, %73 [1, 5] : f16 into vector<2x8xf16>
    %76 = vector.extract %71[0, 2] : f16 from vector<1x4xf16>
    %77 = vector.insert %76, %75 [1, 6] : f16 into vector<2x8xf16>
    %78 = vector.extract %71[0, 3] : f16 from vector<1x4xf16>
    %79 = vector.insert %78, %77 [1, 7] : f16 into vector<2x8xf16>
    gpu.barrier
    %80 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
    %81 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_3) -> (vector<2x32xf32>) {
      %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %99 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg4)
      %100 = vector.insert %cst_5, %arg5 [%99, %98] : f32 into vector<2x32xf32>
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %102 = vector.insert %cst_5, %100 [%99, %101] : f32 into vector<2x32xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %104 = vector.insert %cst_5, %102 [%99, %103] : f32 into vector<2x32xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %106 = vector.insert %cst_5, %104 [%99, %105] : f32 into vector<2x32xf32>
      affine.yield %106 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
    %82 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
    %83 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
    %84 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
    %85 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
    %86:6 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %12, %arg6 = %13, %arg7 = %11#2, %arg8 = %12, %arg9 = %12, %arg10 = %12) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %98 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %subview_13 = memref.subview %transpose[0, %80, 0, %98] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_13 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %99 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_14 = memref.subview %transpose[0, %99, 32, %98] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_14 to %view_10 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
      %100:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %123 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %124 = memref.load %view[%123] : memref<4096xf16, 3>
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %126 = memref.load %view[%125] : memref<4096xf16, 3>
        %127 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %128 = memref.load %view[%127] : memref<4096xf16, 3>
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %130 = memref.load %view[%129] : memref<4096xf16, 3>
        %131 = vector.from_elements %124, %126, %128, %130 : vector<4x1xf16>
        %132 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %133 = memref.load %view[%132] : memref<4096xf16, 3>
        %134 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %135 = memref.load %view[%134] : memref<4096xf16, 3>
        %136 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %137 = memref.load %view[%136] : memref<4096xf16, 3>
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %139 = memref.load %view[%138] : memref<4096xf16, 3>
        %140 = vector.from_elements %133, %135, %137, %139 : vector<4x1xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %142 = vector.extract %34[%141, 0] : f16 from vector<2x8xf16>
        %143 = vector.extract %34[%141, 1] : f16 from vector<2x8xf16>
        %144 = vector.extract %34[%141, 2] : f16 from vector<2x8xf16>
        %145 = vector.extract %34[%141, 3] : f16 from vector<2x8xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = frisk.warp_mma_rr %146, %131, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %148 = vector.extract %34[%141, 4] : f16 from vector<2x8xf16>
        %149 = vector.extract %34[%141, 5] : f16 from vector<2x8xf16>
        %150 = vector.extract %34[%141, 6] : f16 from vector<2x8xf16>
        %151 = vector.extract %34[%141, 7] : f16 from vector<2x8xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf16>
        %153 = frisk.warp_mma_rr %152, %140, %147 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %155 = vector.extract %81[%141, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %157 = vector.extract %81[%141, %156] : f32 from vector<2x32xf32>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %159 = vector.extract %81[%141, %158] : f32 from vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %161 = vector.extract %81[%141, %160] : f32 from vector<2x32xf32>
        %162 = vector.from_elements %155, %157, %159, %161 : vector<1x4xf32>
        %163 = arith.addf %162, %153 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg12 [%141, %154] : f32 into vector<2x32xf32>
        %166 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%141, %156] : f32 into vector<2x32xf32>
        %168 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%141, %158] : f32 into vector<2x32xf32>
        %170 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%141, %160] : f32 into vector<2x32xf32>
        %172 = vector.insert %164, %arg13 [%141, %154] : f32 into vector<2x32xf32>
        %173 = vector.insert %166, %172 [%141, %156] : f32 into vector<2x32xf32>
        %174 = vector.insert %168, %173 [%141, %158] : f32 into vector<2x32xf32>
        %175 = vector.insert %170, %174 [%141, %160] : f32 into vector<2x32xf32>
        affine.yield %171, %175 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %view_10 : memref<4096xf16, 3>
      %101 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 2048) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_15 = memref.subview %transpose[0, %101, 64, %98] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_15 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
      %102:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %123 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %124 = memref.load %view_10[%123] : memref<4096xf16, 3>
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %126 = memref.load %view_10[%125] : memref<4096xf16, 3>
        %127 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %128 = memref.load %view_10[%127] : memref<4096xf16, 3>
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %130 = memref.load %view_10[%129] : memref<4096xf16, 3>
        %131 = vector.from_elements %124, %126, %128, %130 : vector<4x1xf16>
        %132 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %133 = memref.load %view_10[%132] : memref<4096xf16, 3>
        %134 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %135 = memref.load %view_10[%134] : memref<4096xf16, 3>
        %136 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %137 = memref.load %view_10[%136] : memref<4096xf16, 3>
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %139 = memref.load %view_10[%138] : memref<4096xf16, 3>
        %140 = vector.from_elements %133, %135, %137, %139 : vector<4x1xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %142 = vector.extract %49[%141, 0] : f16 from vector<2x8xf16>
        %143 = vector.extract %49[%141, 1] : f16 from vector<2x8xf16>
        %144 = vector.extract %49[%141, 2] : f16 from vector<2x8xf16>
        %145 = vector.extract %49[%141, 3] : f16 from vector<2x8xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = frisk.warp_mma_rr %146, %131, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %148 = vector.extract %49[%141, 4] : f16 from vector<2x8xf16>
        %149 = vector.extract %49[%141, 5] : f16 from vector<2x8xf16>
        %150 = vector.extract %49[%141, 6] : f16 from vector<2x8xf16>
        %151 = vector.extract %49[%141, 7] : f16 from vector<2x8xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf16>
        %153 = frisk.warp_mma_rr %152, %140, %147 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %155 = vector.extract %100#1[%141, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %157 = vector.extract %100#1[%141, %156] : f32 from vector<2x32xf32>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %159 = vector.extract %100#1[%141, %158] : f32 from vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %161 = vector.extract %100#1[%141, %160] : f32 from vector<2x32xf32>
        %162 = vector.from_elements %155, %157, %159, %161 : vector<1x4xf32>
        %163 = arith.addf %162, %153 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg12 [%141, %154] : f32 into vector<2x32xf32>
        %166 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%141, %156] : f32 into vector<2x32xf32>
        %168 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%141, %158] : f32 into vector<2x32xf32>
        %170 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%141, %160] : f32 into vector<2x32xf32>
        %172 = vector.insert %164, %arg13 [%141, %154] : f32 into vector<2x32xf32>
        %173 = vector.insert %166, %172 [%141, %156] : f32 into vector<2x32xf32>
        %174 = vector.insert %168, %173 [%141, %158] : f32 into vector<2x32xf32>
        %175 = vector.insert %170, %174 [%141, %160] : f32 into vector<2x32xf32>
        affine.yield %171, %175 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %103 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 3072) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_16 = memref.subview %transpose[0, %103, 96, %98] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_16 to %view_10 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
      %104:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %123 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %124 = memref.load %view[%123] : memref<4096xf16, 3>
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %126 = memref.load %view[%125] : memref<4096xf16, 3>
        %127 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %128 = memref.load %view[%127] : memref<4096xf16, 3>
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %130 = memref.load %view[%129] : memref<4096xf16, 3>
        %131 = vector.from_elements %124, %126, %128, %130 : vector<4x1xf16>
        %132 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %133 = memref.load %view[%132] : memref<4096xf16, 3>
        %134 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %135 = memref.load %view[%134] : memref<4096xf16, 3>
        %136 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %137 = memref.load %view[%136] : memref<4096xf16, 3>
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %139 = memref.load %view[%138] : memref<4096xf16, 3>
        %140 = vector.from_elements %133, %135, %137, %139 : vector<4x1xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %142 = vector.extract %64[%141, 0] : f16 from vector<2x8xf16>
        %143 = vector.extract %64[%141, 1] : f16 from vector<2x8xf16>
        %144 = vector.extract %64[%141, 2] : f16 from vector<2x8xf16>
        %145 = vector.extract %64[%141, 3] : f16 from vector<2x8xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = frisk.warp_mma_rr %146, %131, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %148 = vector.extract %64[%141, 4] : f16 from vector<2x8xf16>
        %149 = vector.extract %64[%141, 5] : f16 from vector<2x8xf16>
        %150 = vector.extract %64[%141, 6] : f16 from vector<2x8xf16>
        %151 = vector.extract %64[%141, 7] : f16 from vector<2x8xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf16>
        %153 = frisk.warp_mma_rr %152, %140, %147 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %155 = vector.extract %102#1[%141, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %157 = vector.extract %102#1[%141, %156] : f32 from vector<2x32xf32>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %159 = vector.extract %102#1[%141, %158] : f32 from vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %161 = vector.extract %102#1[%141, %160] : f32 from vector<2x32xf32>
        %162 = vector.from_elements %155, %157, %159, %161 : vector<1x4xf32>
        %163 = arith.addf %162, %153 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg12 [%141, %154] : f32 into vector<2x32xf32>
        %166 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%141, %156] : f32 into vector<2x32xf32>
        %168 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%141, %158] : f32 into vector<2x32xf32>
        %170 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%141, %160] : f32 into vector<2x32xf32>
        %172 = vector.insert %164, %arg13 [%141, %154] : f32 into vector<2x32xf32>
        %173 = vector.insert %166, %172 [%141, %156] : f32 into vector<2x32xf32>
        %174 = vector.insert %168, %173 [%141, %158] : f32 into vector<2x32xf32>
        %175 = vector.insert %170, %174 [%141, %160] : f32 into vector<2x32xf32>
        affine.yield %171, %175 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %view_10 : memref<4096xf16, 3>
      %subview_17 = memref.subview %arg2[0, %80, %98, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_17 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %105:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %123 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %124 = memref.load %view_10[%123] : memref<4096xf16, 3>
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %126 = memref.load %view_10[%125] : memref<4096xf16, 3>
        %127 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %128 = memref.load %view_10[%127] : memref<4096xf16, 3>
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %130 = memref.load %view_10[%129] : memref<4096xf16, 3>
        %131 = vector.from_elements %124, %126, %128, %130 : vector<4x1xf16>
        %132 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %133 = memref.load %view_10[%132] : memref<4096xf16, 3>
        %134 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %135 = memref.load %view_10[%134] : memref<4096xf16, 3>
        %136 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %137 = memref.load %view_10[%136] : memref<4096xf16, 3>
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %139 = memref.load %view_10[%138] : memref<4096xf16, 3>
        %140 = vector.from_elements %133, %135, %137, %139 : vector<4x1xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %142 = vector.extract %79[%141, 0] : f16 from vector<2x8xf16>
        %143 = vector.extract %79[%141, 1] : f16 from vector<2x8xf16>
        %144 = vector.extract %79[%141, 2] : f16 from vector<2x8xf16>
        %145 = vector.extract %79[%141, 3] : f16 from vector<2x8xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = frisk.warp_mma_rr %146, %131, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %148 = vector.extract %79[%141, 4] : f16 from vector<2x8xf16>
        %149 = vector.extract %79[%141, 5] : f16 from vector<2x8xf16>
        %150 = vector.extract %79[%141, 6] : f16 from vector<2x8xf16>
        %151 = vector.extract %79[%141, 7] : f16 from vector<2x8xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf16>
        %153 = frisk.warp_mma_rr %152, %140, %147 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %155 = vector.extract %104#1[%141, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %157 = vector.extract %104#1[%141, %156] : f32 from vector<2x32xf32>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %159 = vector.extract %104#1[%141, %158] : f32 from vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %161 = vector.extract %104#1[%141, %160] : f32 from vector<2x32xf32>
        %162 = vector.from_elements %155, %157, %159, %161 : vector<1x4xf32>
        %163 = arith.addf %162, %153 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg12 [%141, %154] : f32 into vector<2x32xf32>
        %166 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%141, %156] : f32 into vector<2x32xf32>
        %168 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%141, %158] : f32 into vector<2x32xf32>
        %170 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%141, %160] : f32 into vector<2x32xf32>
        %172 = vector.insert %164, %arg13 [%141, %154] : f32 into vector<2x32xf32>
        %173 = vector.insert %166, %172 [%141, %156] : f32 into vector<2x32xf32>
        %174 = vector.insert %168, %173 [%141, %158] : f32 into vector<2x32xf32>
        %175 = vector.insert %170, %174 [%141, %160] : f32 into vector<2x32xf32>
        affine.yield %171, %175 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %106 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %123 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %125 = vector.extract %105#1[%123, %124] : f32 from vector<2x32xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %127 = vector.extract %105#1[%123, %126] : f32 from vector<2x32xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %129 = vector.extract %105#1[%123, %128] : f32 from vector<2x32xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %131 = vector.extract %105#1[%123, %130] : f32 from vector<2x32xf32>
        %132 = vector.insert %125, %arg12 [%123, %124] : f32 into vector<2x32xf32>
        %133 = vector.insert %127, %132 [%123, %126] : f32 into vector<2x32xf32>
        %134 = vector.insert %129, %133 [%123, %128] : f32 into vector<2x32xf32>
        %135 = vector.insert %131, %134 [%123, %130] : f32 into vector<2x32xf32>
        affine.yield %135 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %107:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %123 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_0) -> (f32) {
          %133 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %134 = vector.extract %106[%arg11, %133] : f32 from vector<2x32xf32>
          %135 = arith.maxnumf %arg15, %134 : f32
          %136 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %137 = vector.extract %106[%arg11, %136] : f32 from vector<2x32xf32>
          %138 = arith.maxnumf %135, %137 : f32
          %139 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %140 = vector.extract %106[%arg11, %139] : f32 from vector<2x32xf32>
          %141 = arith.maxnumf %138, %140 : f32
          %142 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %143 = vector.extract %106[%arg11, %142] : f32 from vector<2x32xf32>
          %144 = arith.maxnumf %141, %143 : f32
          affine.yield %144 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %123, %c16_i32, %c64_i32 : f32
        %124 = arith.maxnumf %123, %shuffleResult : f32
        %shuffleResult_21, %valid_22 = gpu.shuffle xor %124, %c32_i32, %c64_i32 : f32
        %125 = arith.maxnumf %124, %shuffleResult_21 : f32
        %126 = vector.broadcast %125 : f32 to vector<1x1xf32>
        %127 = vector.insert %125, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %128 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %129 = vector.broadcast %128 : f32 to vector<1x1xf32>
        %130 = arith.maxnumf %126, %129 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %131 = vector.extract %130[0, 0] : f32 from vector<1x1xf32>
        %132 = vector.insert %131, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %127, %132 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %108:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %123 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %124 = vector.extract %107#1[%123, 0] : f32 from vector<2x1xf32>
        %125 = vector.broadcast %124 : f32 to vector<1x4xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %127 = vector.insert %124, %arg12 [%123, %126] : f32 into vector<2x32xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %129 = vector.insert %124, %127 [%123, %128] : f32 into vector<2x32xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %131 = vector.insert %124, %129 [%123, %130] : f32 into vector<2x32xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %133 = vector.insert %124, %131 [%123, %132] : f32 into vector<2x32xf32>
        %134 = vector.extract %105#1[%123, %126] : f32 from vector<2x32xf32>
        %135 = vector.extract %105#1[%123, %128] : f32 from vector<2x32xf32>
        %136 = vector.extract %105#1[%123, %130] : f32 from vector<2x32xf32>
        %137 = vector.extract %105#1[%123, %132] : f32 from vector<2x32xf32>
        %138 = vector.from_elements %134, %135, %136, %137 : vector<1x4xf32>
        %139 = arith.subf %138, %125 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %140 = vector.extract %139[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg13 [%123, %126] : f32 into vector<2x32xf32>
        %142 = vector.extract %139[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %141 [%123, %128] : f32 into vector<2x32xf32>
        %144 = vector.extract %139[0, 2] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%123, %130] : f32 into vector<2x32xf32>
        %146 = vector.extract %139[0, 3] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%123, %132] : f32 into vector<2x32xf32>
        affine.yield %133, %147 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>}
      %109:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %123 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %124 = vector.broadcast %123 : f32 to vector<1x1xf32>
        %125 = vector.extract %107#1[%arg11, 0] : f32 from vector<2x1xf32>
        %126 = vector.broadcast %125 : f32 to vector<1x1xf32>
        %127 = arith.subf %124, %126 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %128 = vector.extract %127[0, 0] : f32 from vector<1x1xf32>
        %129 = vector.insert %128, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %130 = math.exp2 %127 {tiled = true} : vector<1x1xf32>
        %131 = vector.extract %130[0, 0] : f32 from vector<1x1xf32>
        %132 = vector.insert %131, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %133 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %134 = vector.broadcast %133 : f32 to vector<1x1xf32>
        %135 = arith.mulf %134, %130 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %136 = vector.extract %135[0, 0] : f32 from vector<1x1xf32>
        %137 = vector.insert %136, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %129, %132, %137 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %110:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %123 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %125 = vector.extract %108#1[%123, %124] : f32 from vector<2x32xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %127 = vector.extract %108#1[%123, %126] : f32 from vector<2x32xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %129 = vector.extract %108#1[%123, %128] : f32 from vector<2x32xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %131 = vector.extract %108#1[%123, %130] : f32 from vector<2x32xf32>
        %132 = vector.from_elements %125, %127, %129, %131 : vector<1x4xf32>
        %133 = math.exp2 %132 {tiled = true} : vector<1x4xf32>
        %134 = vector.extract %133[0, 0] : f32 from vector<1x4xf32>
        %135 = vector.insert %134, %arg12 [%123, %124] : f32 into vector<2x32xf32>
        %136 = vector.extract %133[0, 1] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %135 [%123, %126] : f32 into vector<2x32xf32>
        %138 = vector.extract %133[0, 2] : f32 from vector<1x4xf32>
        %139 = vector.insert %138, %137 [%123, %128] : f32 into vector<2x32xf32>
        %140 = vector.extract %133[0, 3] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %139 [%123, %130] : f32 into vector<2x32xf32>
        %142 = vector.insert %134, %arg13 [%123, %124] : f32 into vector<2x32xf32>
        %143 = vector.insert %136, %142 [%123, %126] : f32 into vector<2x32xf32>
        %144 = vector.insert %138, %143 [%123, %128] : f32 into vector<2x32xf32>
        %145 = vector.insert %140, %144 [%123, %130] : f32 into vector<2x32xf32>
        affine.yield %141, %145 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %111:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %123 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_5) -> (f32) {
          %133 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %134 = vector.extract %110#1[%arg11, %133] : f32 from vector<2x32xf32>
          %135 = arith.addf %arg15, %134 : f32
          %136 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %137 = vector.extract %110#1[%arg11, %136] : f32 from vector<2x32xf32>
          %138 = arith.addf %135, %137 : f32
          %139 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %140 = vector.extract %110#1[%arg11, %139] : f32 from vector<2x32xf32>
          %141 = arith.addf %138, %140 : f32
          %142 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %143 = vector.extract %110#1[%arg11, %142] : f32 from vector<2x32xf32>
          %144 = arith.addf %141, %143 : f32
          affine.yield %144 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %123, %c16_i32, %c64_i32 : f32
        %124 = arith.addf %123, %shuffleResult : f32
        %shuffleResult_21, %valid_22 = gpu.shuffle xor %124, %c32_i32, %c64_i32 : f32
        %125 = arith.addf %124, %shuffleResult_21 : f32
        %126 = vector.broadcast %125 : f32 to vector<1x1xf32>
        %127 = vector.insert %125, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %128 = vector.extract %109#2[%arg11, 0] : f32 from vector<2x1xf32>
        %129 = vector.broadcast %128 : f32 to vector<1x1xf32>
        %130 = arith.addf %126, %129 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %131 = vector.extract %130[0, 0] : f32 from vector<1x1xf32>
        %132 = vector.insert %131, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %127, %132 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %112 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_6) -> (vector<2x32xf16>) {
        %123 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %125 = vector.extract %110#0[%123, %124] : f32 from vector<2x32xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %127 = vector.extract %110#0[%123, %126] : f32 from vector<2x32xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %129 = vector.extract %110#0[%123, %128] : f32 from vector<2x32xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %131 = vector.extract %110#0[%123, %130] : f32 from vector<2x32xf32>
        %132 = vector.from_elements %125, %127, %129, %131 : vector<1x4xf32>
        %133 = arith.truncf %132 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
        %135 = vector.insert %134, %arg12 [%123, %124] : f16 into vector<2x32xf16>
        %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
        %137 = vector.insert %136, %135 [%123, %126] : f16 into vector<2x32xf16>
        %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
        %139 = vector.insert %138, %137 [%123, %128] : f16 into vector<2x32xf16>
        %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
        %141 = vector.insert %140, %139 [%123, %130] : f16 into vector<2x32xf16>
        affine.yield %141 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %113 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_7) -> (vector<32x2xf16>) {
        %123 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %124 = frisk.read_matrix_from_shm %view[%123] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %125 = vector.bitcast %124 : vector<4xi32> to vector<8xf16>
        %126 = vector.extract %125[0] : f16 from vector<8xf16>
        %127 = vector.extract %125[%82] : f16 from vector<8xf16>
        %128 = vector.extract %125[1] : f16 from vector<8xf16>
        %129 = vector.extract %125[%83] : f16 from vector<8xf16>
        %130 = vector.extract %125[2] : f16 from vector<8xf16>
        %131 = vector.extract %125[%84] : f16 from vector<8xf16>
        %132 = vector.extract %125[3] : f16 from vector<8xf16>
        %133 = vector.extract %125[%85] : f16 from vector<8xf16>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %135 = vector.insert %126, %arg12 [%134, 0] : f16 into vector<32x2xf16>
        %136 = vector.insert %127, %135 [%134, 1] : f16 into vector<32x2xf16>
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %138 = vector.insert %128, %136 [%137, 0] : f16 into vector<32x2xf16>
        %139 = vector.insert %129, %138 [%137, 1] : f16 into vector<32x2xf16>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %141 = vector.insert %130, %139 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %131, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %144 = vector.insert %132, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        affine.yield %145 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_18 = memref.subview %arg2[0, %80, %98, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      frisk.async_copy_gm_to_shm %subview_18 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %114:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_8, %arg13 = %cst_8, %arg14 = %cst_8, %arg15 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %123 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %124 = vector.extract %112[%123, 0] : f16 from vector<2x32xf16>
        %125 = vector.extract %112[%123, 1] : f16 from vector<2x32xf16>
        %126 = vector.extract %112[%123, 2] : f16 from vector<2x32xf16>
        %127 = vector.extract %112[%123, 3] : f16 from vector<2x32xf16>
        %128 = vector.from_elements %124, %125, %126, %127 : vector<1x4xf16>
        %129 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %130 = vector.extract %113[0, %129] : f16 from vector<32x2xf16>
        %131 = vector.extract %113[1, %129] : f16 from vector<32x2xf16>
        %132 = vector.extract %113[2, %129] : f16 from vector<32x2xf16>
        %133 = vector.extract %113[3, %129] : f16 from vector<32x2xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<4x1xf16>
        %135 = frisk.warp_mma_rr %128, %134, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %136 = vector.extract %112[%123, 4] : f16 from vector<2x32xf16>
        %137 = vector.extract %112[%123, 5] : f16 from vector<2x32xf16>
        %138 = vector.extract %112[%123, 6] : f16 from vector<2x32xf16>
        %139 = vector.extract %112[%123, 7] : f16 from vector<2x32xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<1x4xf16>
        %141 = vector.extract %113[4, %129] : f16 from vector<32x2xf16>
        %142 = vector.extract %113[5, %129] : f16 from vector<32x2xf16>
        %143 = vector.extract %113[6, %129] : f16 from vector<32x2xf16>
        %144 = vector.extract %113[7, %129] : f16 from vector<32x2xf16>
        %145 = vector.from_elements %141, %142, %143, %144 : vector<4x1xf16>
        %146 = frisk.warp_mma_rr %140, %145, %135 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %147 = vector.extract %112[%123, 8] : f16 from vector<2x32xf16>
        %148 = vector.extract %112[%123, 9] : f16 from vector<2x32xf16>
        %149 = vector.extract %112[%123, 10] : f16 from vector<2x32xf16>
        %150 = vector.extract %112[%123, 11] : f16 from vector<2x32xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<1x4xf16>
        %152 = vector.extract %113[8, %129] : f16 from vector<32x2xf16>
        %153 = vector.extract %113[9, %129] : f16 from vector<32x2xf16>
        %154 = vector.extract %113[10, %129] : f16 from vector<32x2xf16>
        %155 = vector.extract %113[11, %129] : f16 from vector<32x2xf16>
        %156 = vector.from_elements %152, %153, %154, %155 : vector<4x1xf16>
        %157 = frisk.warp_mma_rr %151, %156, %146 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %158 = vector.extract %112[%123, 12] : f16 from vector<2x32xf16>
        %159 = vector.extract %112[%123, 13] : f16 from vector<2x32xf16>
        %160 = vector.extract %112[%123, 14] : f16 from vector<2x32xf16>
        %161 = vector.extract %112[%123, 15] : f16 from vector<2x32xf16>
        %162 = vector.from_elements %158, %159, %160, %161 : vector<1x4xf16>
        %163 = vector.extract %113[12, %129] : f16 from vector<32x2xf16>
        %164 = vector.extract %113[13, %129] : f16 from vector<32x2xf16>
        %165 = vector.extract %113[14, %129] : f16 from vector<32x2xf16>
        %166 = vector.extract %113[15, %129] : f16 from vector<32x2xf16>
        %167 = vector.from_elements %163, %164, %165, %166 : vector<4x1xf16>
        %168 = frisk.warp_mma_rr %162, %167, %157 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %169 = vector.extract %112[%123, 16] : f16 from vector<2x32xf16>
        %170 = vector.extract %112[%123, 17] : f16 from vector<2x32xf16>
        %171 = vector.extract %112[%123, 18] : f16 from vector<2x32xf16>
        %172 = vector.extract %112[%123, 19] : f16 from vector<2x32xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf16>
        %174 = vector.extract %113[16, %129] : f16 from vector<32x2xf16>
        %175 = vector.extract %113[17, %129] : f16 from vector<32x2xf16>
        %176 = vector.extract %113[18, %129] : f16 from vector<32x2xf16>
        %177 = vector.extract %113[19, %129] : f16 from vector<32x2xf16>
        %178 = vector.from_elements %174, %175, %176, %177 : vector<4x1xf16>
        %179 = frisk.warp_mma_rr %173, %178, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %180 = vector.extract %112[%123, 20] : f16 from vector<2x32xf16>
        %181 = vector.extract %112[%123, 21] : f16 from vector<2x32xf16>
        %182 = vector.extract %112[%123, 22] : f16 from vector<2x32xf16>
        %183 = vector.extract %112[%123, 23] : f16 from vector<2x32xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<1x4xf16>
        %185 = vector.extract %113[20, %129] : f16 from vector<32x2xf16>
        %186 = vector.extract %113[21, %129] : f16 from vector<32x2xf16>
        %187 = vector.extract %113[22, %129] : f16 from vector<32x2xf16>
        %188 = vector.extract %113[23, %129] : f16 from vector<32x2xf16>
        %189 = vector.from_elements %185, %186, %187, %188 : vector<4x1xf16>
        %190 = frisk.warp_mma_rr %184, %189, %179 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %191 = vector.extract %112[%123, 24] : f16 from vector<2x32xf16>
        %192 = vector.extract %112[%123, 25] : f16 from vector<2x32xf16>
        %193 = vector.extract %112[%123, 26] : f16 from vector<2x32xf16>
        %194 = vector.extract %112[%123, 27] : f16 from vector<2x32xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<1x4xf16>
        %196 = vector.extract %113[24, %129] : f16 from vector<32x2xf16>
        %197 = vector.extract %113[25, %129] : f16 from vector<32x2xf16>
        %198 = vector.extract %113[26, %129] : f16 from vector<32x2xf16>
        %199 = vector.extract %113[27, %129] : f16 from vector<32x2xf16>
        %200 = vector.from_elements %196, %197, %198, %199 : vector<4x1xf16>
        %201 = frisk.warp_mma_rr %195, %200, %190 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %202 = vector.extract %112[%123, 28] : f16 from vector<2x32xf16>
        %203 = vector.extract %112[%123, 29] : f16 from vector<2x32xf16>
        %204 = vector.extract %112[%123, 30] : f16 from vector<2x32xf16>
        %205 = vector.extract %112[%123, 31] : f16 from vector<2x32xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<1x4xf16>
        %207 = vector.extract %113[28, %129] : f16 from vector<32x2xf16>
        %208 = vector.extract %113[29, %129] : f16 from vector<32x2xf16>
        %209 = vector.extract %113[30, %129] : f16 from vector<32x2xf16>
        %210 = vector.extract %113[31, %129] : f16 from vector<32x2xf16>
        %211 = vector.from_elements %207, %208, %209, %210 : vector<4x1xf16>
        %212 = frisk.warp_mma_rr %206, %211, %201 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %214 = vector.extract %212[0, 0] : f32 from vector<1x4xf32>
        %215 = vector.insert %214, %arg12 [%123, %213] : f32 into vector<2x8xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %217 = vector.extract %212[0, 1] : f32 from vector<1x4xf32>
        %218 = vector.insert %217, %215 [%123, %216] : f32 into vector<2x8xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %220 = vector.extract %212[0, 2] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %218 [%123, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %223 = vector.extract %212[0, 3] : f32 from vector<1x4xf32>
        %224 = vector.insert %223, %221 [%123, %222] : f32 into vector<2x8xf32>
        %225 = vector.extract %109#1[%123, 0] : f32 from vector<2x1xf32>
        %226 = vector.broadcast %225 : f32 to vector<1x4xf32>
        %227 = vector.insert %225, %arg13 [%123, %213] : f32 into vector<2x8xf32>
        %228 = vector.insert %225, %227 [%123, %216] : f32 into vector<2x8xf32>
        %229 = vector.insert %225, %228 [%123, %219] : f32 into vector<2x8xf32>
        %230 = vector.insert %225, %229 [%123, %222] : f32 into vector<2x8xf32>
        %231 = vector.extract %arg5[%123, %213] : f32 from vector<2x8xf32>
        %232 = vector.extract %arg5[%123, %216] : f32 from vector<2x8xf32>
        %233 = vector.extract %arg5[%123, %219] : f32 from vector<2x8xf32>
        %234 = vector.extract %arg5[%123, %222] : f32 from vector<2x8xf32>
        %235 = vector.from_elements %231, %232, %233, %234 : vector<1x4xf32>
        %236 = arith.mulf %235, %226 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %237 = vector.extract %236[0, 0] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %arg14 [%123, %213] : f32 into vector<2x8xf32>
        %239 = vector.extract %236[0, 1] : f32 from vector<1x4xf32>
        %240 = vector.insert %239, %238 [%123, %216] : f32 into vector<2x8xf32>
        %241 = vector.extract %236[0, 2] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %240 [%123, %219] : f32 into vector<2x8xf32>
        %243 = vector.extract %236[0, 3] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %242 [%123, %222] : f32 into vector<2x8xf32>
        %245 = arith.addf %236, %212 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %246 = vector.extract %245[0, 0] : f32 from vector<1x4xf32>
        %247 = vector.insert %246, %arg15 [%123, %213] : f32 into vector<2x8xf32>
        %248 = vector.extract %245[0, 1] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %247 [%123, %216] : f32 into vector<2x8xf32>
        %250 = vector.extract %245[0, 2] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %249 [%123, %219] : f32 into vector<2x8xf32>
        %252 = vector.extract %245[0, 3] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %251 [%123, %222] : f32 into vector<2x8xf32>
        affine.yield %224, %230, %244, %253 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %115 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_7) -> (vector<32x2xf16>) {
        %123 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %124 = frisk.read_matrix_from_shm %view[%123] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %125 = vector.bitcast %124 : vector<4xi32> to vector<8xf16>
        %126 = vector.extract %125[0] : f16 from vector<8xf16>
        %127 = vector.extract %125[%82] : f16 from vector<8xf16>
        %128 = vector.extract %125[1] : f16 from vector<8xf16>
        %129 = vector.extract %125[%83] : f16 from vector<8xf16>
        %130 = vector.extract %125[2] : f16 from vector<8xf16>
        %131 = vector.extract %125[%84] : f16 from vector<8xf16>
        %132 = vector.extract %125[3] : f16 from vector<8xf16>
        %133 = vector.extract %125[%85] : f16 from vector<8xf16>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %135 = vector.insert %126, %arg12 [%134, 0] : f16 into vector<32x2xf16>
        %136 = vector.insert %127, %135 [%134, 1] : f16 into vector<32x2xf16>
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %138 = vector.insert %128, %136 [%137, 0] : f16 into vector<32x2xf16>
        %139 = vector.insert %129, %138 [%137, 1] : f16 into vector<32x2xf16>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %141 = vector.insert %130, %139 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %131, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %144 = vector.insert %132, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        affine.yield %145 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_19 = memref.subview %arg2[0, %80, %98, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      frisk.async_copy_gm_to_shm %subview_19 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %116:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_8, %arg13 = %cst_8, %arg14 = %cst_8, %arg15 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %123 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %124 = vector.extract %112[%123, 0] : f16 from vector<2x32xf16>
        %125 = vector.extract %112[%123, 1] : f16 from vector<2x32xf16>
        %126 = vector.extract %112[%123, 2] : f16 from vector<2x32xf16>
        %127 = vector.extract %112[%123, 3] : f16 from vector<2x32xf16>
        %128 = vector.from_elements %124, %125, %126, %127 : vector<1x4xf16>
        %129 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %130 = vector.extract %115[0, %129] : f16 from vector<32x2xf16>
        %131 = vector.extract %115[1, %129] : f16 from vector<32x2xf16>
        %132 = vector.extract %115[2, %129] : f16 from vector<32x2xf16>
        %133 = vector.extract %115[3, %129] : f16 from vector<32x2xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<4x1xf16>
        %135 = frisk.warp_mma_rr %128, %134, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %136 = vector.extract %112[%123, 4] : f16 from vector<2x32xf16>
        %137 = vector.extract %112[%123, 5] : f16 from vector<2x32xf16>
        %138 = vector.extract %112[%123, 6] : f16 from vector<2x32xf16>
        %139 = vector.extract %112[%123, 7] : f16 from vector<2x32xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<1x4xf16>
        %141 = vector.extract %115[4, %129] : f16 from vector<32x2xf16>
        %142 = vector.extract %115[5, %129] : f16 from vector<32x2xf16>
        %143 = vector.extract %115[6, %129] : f16 from vector<32x2xf16>
        %144 = vector.extract %115[7, %129] : f16 from vector<32x2xf16>
        %145 = vector.from_elements %141, %142, %143, %144 : vector<4x1xf16>
        %146 = frisk.warp_mma_rr %140, %145, %135 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %147 = vector.extract %112[%123, 8] : f16 from vector<2x32xf16>
        %148 = vector.extract %112[%123, 9] : f16 from vector<2x32xf16>
        %149 = vector.extract %112[%123, 10] : f16 from vector<2x32xf16>
        %150 = vector.extract %112[%123, 11] : f16 from vector<2x32xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<1x4xf16>
        %152 = vector.extract %115[8, %129] : f16 from vector<32x2xf16>
        %153 = vector.extract %115[9, %129] : f16 from vector<32x2xf16>
        %154 = vector.extract %115[10, %129] : f16 from vector<32x2xf16>
        %155 = vector.extract %115[11, %129] : f16 from vector<32x2xf16>
        %156 = vector.from_elements %152, %153, %154, %155 : vector<4x1xf16>
        %157 = frisk.warp_mma_rr %151, %156, %146 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %158 = vector.extract %112[%123, 12] : f16 from vector<2x32xf16>
        %159 = vector.extract %112[%123, 13] : f16 from vector<2x32xf16>
        %160 = vector.extract %112[%123, 14] : f16 from vector<2x32xf16>
        %161 = vector.extract %112[%123, 15] : f16 from vector<2x32xf16>
        %162 = vector.from_elements %158, %159, %160, %161 : vector<1x4xf16>
        %163 = vector.extract %115[12, %129] : f16 from vector<32x2xf16>
        %164 = vector.extract %115[13, %129] : f16 from vector<32x2xf16>
        %165 = vector.extract %115[14, %129] : f16 from vector<32x2xf16>
        %166 = vector.extract %115[15, %129] : f16 from vector<32x2xf16>
        %167 = vector.from_elements %163, %164, %165, %166 : vector<4x1xf16>
        %168 = frisk.warp_mma_rr %162, %167, %157 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %169 = vector.extract %112[%123, 16] : f16 from vector<2x32xf16>
        %170 = vector.extract %112[%123, 17] : f16 from vector<2x32xf16>
        %171 = vector.extract %112[%123, 18] : f16 from vector<2x32xf16>
        %172 = vector.extract %112[%123, 19] : f16 from vector<2x32xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf16>
        %174 = vector.extract %115[16, %129] : f16 from vector<32x2xf16>
        %175 = vector.extract %115[17, %129] : f16 from vector<32x2xf16>
        %176 = vector.extract %115[18, %129] : f16 from vector<32x2xf16>
        %177 = vector.extract %115[19, %129] : f16 from vector<32x2xf16>
        %178 = vector.from_elements %174, %175, %176, %177 : vector<4x1xf16>
        %179 = frisk.warp_mma_rr %173, %178, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %180 = vector.extract %112[%123, 20] : f16 from vector<2x32xf16>
        %181 = vector.extract %112[%123, 21] : f16 from vector<2x32xf16>
        %182 = vector.extract %112[%123, 22] : f16 from vector<2x32xf16>
        %183 = vector.extract %112[%123, 23] : f16 from vector<2x32xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<1x4xf16>
        %185 = vector.extract %115[20, %129] : f16 from vector<32x2xf16>
        %186 = vector.extract %115[21, %129] : f16 from vector<32x2xf16>
        %187 = vector.extract %115[22, %129] : f16 from vector<32x2xf16>
        %188 = vector.extract %115[23, %129] : f16 from vector<32x2xf16>
        %189 = vector.from_elements %185, %186, %187, %188 : vector<4x1xf16>
        %190 = frisk.warp_mma_rr %184, %189, %179 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %191 = vector.extract %112[%123, 24] : f16 from vector<2x32xf16>
        %192 = vector.extract %112[%123, 25] : f16 from vector<2x32xf16>
        %193 = vector.extract %112[%123, 26] : f16 from vector<2x32xf16>
        %194 = vector.extract %112[%123, 27] : f16 from vector<2x32xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<1x4xf16>
        %196 = vector.extract %115[24, %129] : f16 from vector<32x2xf16>
        %197 = vector.extract %115[25, %129] : f16 from vector<32x2xf16>
        %198 = vector.extract %115[26, %129] : f16 from vector<32x2xf16>
        %199 = vector.extract %115[27, %129] : f16 from vector<32x2xf16>
        %200 = vector.from_elements %196, %197, %198, %199 : vector<4x1xf16>
        %201 = frisk.warp_mma_rr %195, %200, %190 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %202 = vector.extract %112[%123, 28] : f16 from vector<2x32xf16>
        %203 = vector.extract %112[%123, 29] : f16 from vector<2x32xf16>
        %204 = vector.extract %112[%123, 30] : f16 from vector<2x32xf16>
        %205 = vector.extract %112[%123, 31] : f16 from vector<2x32xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<1x4xf16>
        %207 = vector.extract %115[28, %129] : f16 from vector<32x2xf16>
        %208 = vector.extract %115[29, %129] : f16 from vector<32x2xf16>
        %209 = vector.extract %115[30, %129] : f16 from vector<32x2xf16>
        %210 = vector.extract %115[31, %129] : f16 from vector<32x2xf16>
        %211 = vector.from_elements %207, %208, %209, %210 : vector<4x1xf16>
        %212 = frisk.warp_mma_rr %206, %211, %201 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %214 = vector.extract %212[0, 0] : f32 from vector<1x4xf32>
        %215 = vector.insert %214, %arg12 [%123, %213] : f32 into vector<2x8xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %217 = vector.extract %212[0, 1] : f32 from vector<1x4xf32>
        %218 = vector.insert %217, %215 [%123, %216] : f32 into vector<2x8xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %220 = vector.extract %212[0, 2] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %218 [%123, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %223 = vector.extract %212[0, 3] : f32 from vector<1x4xf32>
        %224 = vector.insert %223, %221 [%123, %222] : f32 into vector<2x8xf32>
        %225 = vector.extract %109#1[%123, 0] : f32 from vector<2x1xf32>
        %226 = vector.broadcast %225 : f32 to vector<1x4xf32>
        %227 = vector.insert %225, %arg13 [%123, %213] : f32 into vector<2x8xf32>
        %228 = vector.insert %225, %227 [%123, %216] : f32 into vector<2x8xf32>
        %229 = vector.insert %225, %228 [%123, %219] : f32 into vector<2x8xf32>
        %230 = vector.insert %225, %229 [%123, %222] : f32 into vector<2x8xf32>
        %231 = vector.extract %arg8[%123, %213] : f32 from vector<2x8xf32>
        %232 = vector.extract %arg8[%123, %216] : f32 from vector<2x8xf32>
        %233 = vector.extract %arg8[%123, %219] : f32 from vector<2x8xf32>
        %234 = vector.extract %arg8[%123, %222] : f32 from vector<2x8xf32>
        %235 = vector.from_elements %231, %232, %233, %234 : vector<1x4xf32>
        %236 = arith.mulf %235, %226 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %237 = vector.extract %236[0, 0] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %arg14 [%123, %213] : f32 into vector<2x8xf32>
        %239 = vector.extract %236[0, 1] : f32 from vector<1x4xf32>
        %240 = vector.insert %239, %238 [%123, %216] : f32 into vector<2x8xf32>
        %241 = vector.extract %236[0, 2] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %240 [%123, %219] : f32 into vector<2x8xf32>
        %243 = vector.extract %236[0, 3] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %242 [%123, %222] : f32 into vector<2x8xf32>
        %245 = arith.addf %236, %212 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %246 = vector.extract %245[0, 0] : f32 from vector<1x4xf32>
        %247 = vector.insert %246, %arg15 [%123, %213] : f32 into vector<2x8xf32>
        %248 = vector.extract %245[0, 1] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %247 [%123, %216] : f32 into vector<2x8xf32>
        %250 = vector.extract %245[0, 2] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %249 [%123, %219] : f32 into vector<2x8xf32>
        %252 = vector.extract %245[0, 3] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %251 [%123, %222] : f32 into vector<2x8xf32>
        affine.yield %224, %230, %244, %253 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %117 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_7) -> (vector<32x2xf16>) {
        %123 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %124 = frisk.read_matrix_from_shm %view[%123] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %125 = vector.bitcast %124 : vector<4xi32> to vector<8xf16>
        %126 = vector.extract %125[0] : f16 from vector<8xf16>
        %127 = vector.extract %125[%82] : f16 from vector<8xf16>
        %128 = vector.extract %125[1] : f16 from vector<8xf16>
        %129 = vector.extract %125[%83] : f16 from vector<8xf16>
        %130 = vector.extract %125[2] : f16 from vector<8xf16>
        %131 = vector.extract %125[%84] : f16 from vector<8xf16>
        %132 = vector.extract %125[3] : f16 from vector<8xf16>
        %133 = vector.extract %125[%85] : f16 from vector<8xf16>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %135 = vector.insert %126, %arg12 [%134, 0] : f16 into vector<32x2xf16>
        %136 = vector.insert %127, %135 [%134, 1] : f16 into vector<32x2xf16>
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %138 = vector.insert %128, %136 [%137, 0] : f16 into vector<32x2xf16>
        %139 = vector.insert %129, %138 [%137, 1] : f16 into vector<32x2xf16>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %141 = vector.insert %130, %139 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %131, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %144 = vector.insert %132, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        affine.yield %145 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_20 = memref.subview %arg2[0, %80, %98, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      frisk.async_copy_gm_to_shm %subview_20 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %118:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_8, %arg13 = %cst_8, %arg14 = %cst_8, %arg15 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %123 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %124 = vector.extract %112[%123, 0] : f16 from vector<2x32xf16>
        %125 = vector.extract %112[%123, 1] : f16 from vector<2x32xf16>
        %126 = vector.extract %112[%123, 2] : f16 from vector<2x32xf16>
        %127 = vector.extract %112[%123, 3] : f16 from vector<2x32xf16>
        %128 = vector.from_elements %124, %125, %126, %127 : vector<1x4xf16>
        %129 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %130 = vector.extract %117[0, %129] : f16 from vector<32x2xf16>
        %131 = vector.extract %117[1, %129] : f16 from vector<32x2xf16>
        %132 = vector.extract %117[2, %129] : f16 from vector<32x2xf16>
        %133 = vector.extract %117[3, %129] : f16 from vector<32x2xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<4x1xf16>
        %135 = frisk.warp_mma_rr %128, %134, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %136 = vector.extract %112[%123, 4] : f16 from vector<2x32xf16>
        %137 = vector.extract %112[%123, 5] : f16 from vector<2x32xf16>
        %138 = vector.extract %112[%123, 6] : f16 from vector<2x32xf16>
        %139 = vector.extract %112[%123, 7] : f16 from vector<2x32xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<1x4xf16>
        %141 = vector.extract %117[4, %129] : f16 from vector<32x2xf16>
        %142 = vector.extract %117[5, %129] : f16 from vector<32x2xf16>
        %143 = vector.extract %117[6, %129] : f16 from vector<32x2xf16>
        %144 = vector.extract %117[7, %129] : f16 from vector<32x2xf16>
        %145 = vector.from_elements %141, %142, %143, %144 : vector<4x1xf16>
        %146 = frisk.warp_mma_rr %140, %145, %135 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %147 = vector.extract %112[%123, 8] : f16 from vector<2x32xf16>
        %148 = vector.extract %112[%123, 9] : f16 from vector<2x32xf16>
        %149 = vector.extract %112[%123, 10] : f16 from vector<2x32xf16>
        %150 = vector.extract %112[%123, 11] : f16 from vector<2x32xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<1x4xf16>
        %152 = vector.extract %117[8, %129] : f16 from vector<32x2xf16>
        %153 = vector.extract %117[9, %129] : f16 from vector<32x2xf16>
        %154 = vector.extract %117[10, %129] : f16 from vector<32x2xf16>
        %155 = vector.extract %117[11, %129] : f16 from vector<32x2xf16>
        %156 = vector.from_elements %152, %153, %154, %155 : vector<4x1xf16>
        %157 = frisk.warp_mma_rr %151, %156, %146 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %158 = vector.extract %112[%123, 12] : f16 from vector<2x32xf16>
        %159 = vector.extract %112[%123, 13] : f16 from vector<2x32xf16>
        %160 = vector.extract %112[%123, 14] : f16 from vector<2x32xf16>
        %161 = vector.extract %112[%123, 15] : f16 from vector<2x32xf16>
        %162 = vector.from_elements %158, %159, %160, %161 : vector<1x4xf16>
        %163 = vector.extract %117[12, %129] : f16 from vector<32x2xf16>
        %164 = vector.extract %117[13, %129] : f16 from vector<32x2xf16>
        %165 = vector.extract %117[14, %129] : f16 from vector<32x2xf16>
        %166 = vector.extract %117[15, %129] : f16 from vector<32x2xf16>
        %167 = vector.from_elements %163, %164, %165, %166 : vector<4x1xf16>
        %168 = frisk.warp_mma_rr %162, %167, %157 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %169 = vector.extract %112[%123, 16] : f16 from vector<2x32xf16>
        %170 = vector.extract %112[%123, 17] : f16 from vector<2x32xf16>
        %171 = vector.extract %112[%123, 18] : f16 from vector<2x32xf16>
        %172 = vector.extract %112[%123, 19] : f16 from vector<2x32xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf16>
        %174 = vector.extract %117[16, %129] : f16 from vector<32x2xf16>
        %175 = vector.extract %117[17, %129] : f16 from vector<32x2xf16>
        %176 = vector.extract %117[18, %129] : f16 from vector<32x2xf16>
        %177 = vector.extract %117[19, %129] : f16 from vector<32x2xf16>
        %178 = vector.from_elements %174, %175, %176, %177 : vector<4x1xf16>
        %179 = frisk.warp_mma_rr %173, %178, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %180 = vector.extract %112[%123, 20] : f16 from vector<2x32xf16>
        %181 = vector.extract %112[%123, 21] : f16 from vector<2x32xf16>
        %182 = vector.extract %112[%123, 22] : f16 from vector<2x32xf16>
        %183 = vector.extract %112[%123, 23] : f16 from vector<2x32xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<1x4xf16>
        %185 = vector.extract %117[20, %129] : f16 from vector<32x2xf16>
        %186 = vector.extract %117[21, %129] : f16 from vector<32x2xf16>
        %187 = vector.extract %117[22, %129] : f16 from vector<32x2xf16>
        %188 = vector.extract %117[23, %129] : f16 from vector<32x2xf16>
        %189 = vector.from_elements %185, %186, %187, %188 : vector<4x1xf16>
        %190 = frisk.warp_mma_rr %184, %189, %179 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %191 = vector.extract %112[%123, 24] : f16 from vector<2x32xf16>
        %192 = vector.extract %112[%123, 25] : f16 from vector<2x32xf16>
        %193 = vector.extract %112[%123, 26] : f16 from vector<2x32xf16>
        %194 = vector.extract %112[%123, 27] : f16 from vector<2x32xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<1x4xf16>
        %196 = vector.extract %117[24, %129] : f16 from vector<32x2xf16>
        %197 = vector.extract %117[25, %129] : f16 from vector<32x2xf16>
        %198 = vector.extract %117[26, %129] : f16 from vector<32x2xf16>
        %199 = vector.extract %117[27, %129] : f16 from vector<32x2xf16>
        %200 = vector.from_elements %196, %197, %198, %199 : vector<4x1xf16>
        %201 = frisk.warp_mma_rr %195, %200, %190 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %202 = vector.extract %112[%123, 28] : f16 from vector<2x32xf16>
        %203 = vector.extract %112[%123, 29] : f16 from vector<2x32xf16>
        %204 = vector.extract %112[%123, 30] : f16 from vector<2x32xf16>
        %205 = vector.extract %112[%123, 31] : f16 from vector<2x32xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<1x4xf16>
        %207 = vector.extract %117[28, %129] : f16 from vector<32x2xf16>
        %208 = vector.extract %117[29, %129] : f16 from vector<32x2xf16>
        %209 = vector.extract %117[30, %129] : f16 from vector<32x2xf16>
        %210 = vector.extract %117[31, %129] : f16 from vector<32x2xf16>
        %211 = vector.from_elements %207, %208, %209, %210 : vector<4x1xf16>
        %212 = frisk.warp_mma_rr %206, %211, %201 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %214 = vector.extract %212[0, 0] : f32 from vector<1x4xf32>
        %215 = vector.insert %214, %arg12 [%123, %213] : f32 into vector<2x8xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %217 = vector.extract %212[0, 1] : f32 from vector<1x4xf32>
        %218 = vector.insert %217, %215 [%123, %216] : f32 into vector<2x8xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %220 = vector.extract %212[0, 2] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %218 [%123, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %223 = vector.extract %212[0, 3] : f32 from vector<1x4xf32>
        %224 = vector.insert %223, %221 [%123, %222] : f32 into vector<2x8xf32>
        %225 = vector.extract %109#1[%123, 0] : f32 from vector<2x1xf32>
        %226 = vector.broadcast %225 : f32 to vector<1x4xf32>
        %227 = vector.insert %225, %arg13 [%123, %213] : f32 into vector<2x8xf32>
        %228 = vector.insert %225, %227 [%123, %216] : f32 into vector<2x8xf32>
        %229 = vector.insert %225, %228 [%123, %219] : f32 into vector<2x8xf32>
        %230 = vector.insert %225, %229 [%123, %222] : f32 into vector<2x8xf32>
        %231 = vector.extract %arg9[%123, %213] : f32 from vector<2x8xf32>
        %232 = vector.extract %arg9[%123, %216] : f32 from vector<2x8xf32>
        %233 = vector.extract %arg9[%123, %219] : f32 from vector<2x8xf32>
        %234 = vector.extract %arg9[%123, %222] : f32 from vector<2x8xf32>
        %235 = vector.from_elements %231, %232, %233, %234 : vector<1x4xf32>
        %236 = arith.mulf %235, %226 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %237 = vector.extract %236[0, 0] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %arg14 [%123, %213] : f32 into vector<2x8xf32>
        %239 = vector.extract %236[0, 1] : f32 from vector<1x4xf32>
        %240 = vector.insert %239, %238 [%123, %216] : f32 into vector<2x8xf32>
        %241 = vector.extract %236[0, 2] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %240 [%123, %219] : f32 into vector<2x8xf32>
        %243 = vector.extract %236[0, 3] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %242 [%123, %222] : f32 into vector<2x8xf32>
        %245 = arith.addf %236, %212 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %246 = vector.extract %245[0, 0] : f32 from vector<1x4xf32>
        %247 = vector.insert %246, %arg15 [%123, %213] : f32 into vector<2x8xf32>
        %248 = vector.extract %245[0, 1] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %247 [%123, %216] : f32 into vector<2x8xf32>
        %250 = vector.extract %245[0, 2] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %249 [%123, %219] : f32 into vector<2x8xf32>
        %252 = vector.extract %245[0, 3] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %251 [%123, %222] : f32 into vector<2x8xf32>
        affine.yield %224, %230, %244, %253 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %119 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_7) -> (vector<32x2xf16>) {
        %123 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %124 = frisk.read_matrix_from_shm %view[%123] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %125 = vector.bitcast %124 : vector<4xi32> to vector<8xf16>
        %126 = vector.extract %125[0] : f16 from vector<8xf16>
        %127 = vector.extract %125[%82] : f16 from vector<8xf16>
        %128 = vector.extract %125[1] : f16 from vector<8xf16>
        %129 = vector.extract %125[%83] : f16 from vector<8xf16>
        %130 = vector.extract %125[2] : f16 from vector<8xf16>
        %131 = vector.extract %125[%84] : f16 from vector<8xf16>
        %132 = vector.extract %125[3] : f16 from vector<8xf16>
        %133 = vector.extract %125[%85] : f16 from vector<8xf16>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %135 = vector.insert %126, %arg12 [%134, 0] : f16 into vector<32x2xf16>
        %136 = vector.insert %127, %135 [%134, 1] : f16 into vector<32x2xf16>
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %138 = vector.insert %128, %136 [%137, 0] : f16 into vector<32x2xf16>
        %139 = vector.insert %129, %138 [%137, 1] : f16 into vector<32x2xf16>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %141 = vector.insert %130, %139 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %131, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %144 = vector.insert %132, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %133, %144 [%143, 1] : f16 into vector<32x2xf16>
        affine.yield %145 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %120:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_8, %arg13 = %cst_8, %arg14 = %cst_8, %arg15 = %cst_8, %arg16 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %123 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %124 = vector.extract %112[%123, 0] : f16 from vector<2x32xf16>
        %125 = vector.extract %112[%123, 1] : f16 from vector<2x32xf16>
        %126 = vector.extract %112[%123, 2] : f16 from vector<2x32xf16>
        %127 = vector.extract %112[%123, 3] : f16 from vector<2x32xf16>
        %128 = vector.from_elements %124, %125, %126, %127 : vector<1x4xf16>
        %129 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %130 = vector.extract %119[0, %129] : f16 from vector<32x2xf16>
        %131 = vector.extract %119[1, %129] : f16 from vector<32x2xf16>
        %132 = vector.extract %119[2, %129] : f16 from vector<32x2xf16>
        %133 = vector.extract %119[3, %129] : f16 from vector<32x2xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<4x1xf16>
        %135 = frisk.warp_mma_rr %128, %134, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %136 = vector.extract %112[%123, 4] : f16 from vector<2x32xf16>
        %137 = vector.extract %112[%123, 5] : f16 from vector<2x32xf16>
        %138 = vector.extract %112[%123, 6] : f16 from vector<2x32xf16>
        %139 = vector.extract %112[%123, 7] : f16 from vector<2x32xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<1x4xf16>
        %141 = vector.extract %119[4, %129] : f16 from vector<32x2xf16>
        %142 = vector.extract %119[5, %129] : f16 from vector<32x2xf16>
        %143 = vector.extract %119[6, %129] : f16 from vector<32x2xf16>
        %144 = vector.extract %119[7, %129] : f16 from vector<32x2xf16>
        %145 = vector.from_elements %141, %142, %143, %144 : vector<4x1xf16>
        %146 = frisk.warp_mma_rr %140, %145, %135 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %147 = vector.extract %112[%123, 8] : f16 from vector<2x32xf16>
        %148 = vector.extract %112[%123, 9] : f16 from vector<2x32xf16>
        %149 = vector.extract %112[%123, 10] : f16 from vector<2x32xf16>
        %150 = vector.extract %112[%123, 11] : f16 from vector<2x32xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<1x4xf16>
        %152 = vector.extract %119[8, %129] : f16 from vector<32x2xf16>
        %153 = vector.extract %119[9, %129] : f16 from vector<32x2xf16>
        %154 = vector.extract %119[10, %129] : f16 from vector<32x2xf16>
        %155 = vector.extract %119[11, %129] : f16 from vector<32x2xf16>
        %156 = vector.from_elements %152, %153, %154, %155 : vector<4x1xf16>
        %157 = frisk.warp_mma_rr %151, %156, %146 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %158 = vector.extract %112[%123, 12] : f16 from vector<2x32xf16>
        %159 = vector.extract %112[%123, 13] : f16 from vector<2x32xf16>
        %160 = vector.extract %112[%123, 14] : f16 from vector<2x32xf16>
        %161 = vector.extract %112[%123, 15] : f16 from vector<2x32xf16>
        %162 = vector.from_elements %158, %159, %160, %161 : vector<1x4xf16>
        %163 = vector.extract %119[12, %129] : f16 from vector<32x2xf16>
        %164 = vector.extract %119[13, %129] : f16 from vector<32x2xf16>
        %165 = vector.extract %119[14, %129] : f16 from vector<32x2xf16>
        %166 = vector.extract %119[15, %129] : f16 from vector<32x2xf16>
        %167 = vector.from_elements %163, %164, %165, %166 : vector<4x1xf16>
        %168 = frisk.warp_mma_rr %162, %167, %157 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %169 = vector.extract %112[%123, 16] : f16 from vector<2x32xf16>
        %170 = vector.extract %112[%123, 17] : f16 from vector<2x32xf16>
        %171 = vector.extract %112[%123, 18] : f16 from vector<2x32xf16>
        %172 = vector.extract %112[%123, 19] : f16 from vector<2x32xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf16>
        %174 = vector.extract %119[16, %129] : f16 from vector<32x2xf16>
        %175 = vector.extract %119[17, %129] : f16 from vector<32x2xf16>
        %176 = vector.extract %119[18, %129] : f16 from vector<32x2xf16>
        %177 = vector.extract %119[19, %129] : f16 from vector<32x2xf16>
        %178 = vector.from_elements %174, %175, %176, %177 : vector<4x1xf16>
        %179 = frisk.warp_mma_rr %173, %178, %168 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %180 = vector.extract %112[%123, 20] : f16 from vector<2x32xf16>
        %181 = vector.extract %112[%123, 21] : f16 from vector<2x32xf16>
        %182 = vector.extract %112[%123, 22] : f16 from vector<2x32xf16>
        %183 = vector.extract %112[%123, 23] : f16 from vector<2x32xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<1x4xf16>
        %185 = vector.extract %119[20, %129] : f16 from vector<32x2xf16>
        %186 = vector.extract %119[21, %129] : f16 from vector<32x2xf16>
        %187 = vector.extract %119[22, %129] : f16 from vector<32x2xf16>
        %188 = vector.extract %119[23, %129] : f16 from vector<32x2xf16>
        %189 = vector.from_elements %185, %186, %187, %188 : vector<4x1xf16>
        %190 = frisk.warp_mma_rr %184, %189, %179 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %191 = vector.extract %112[%123, 24] : f16 from vector<2x32xf16>
        %192 = vector.extract %112[%123, 25] : f16 from vector<2x32xf16>
        %193 = vector.extract %112[%123, 26] : f16 from vector<2x32xf16>
        %194 = vector.extract %112[%123, 27] : f16 from vector<2x32xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<1x4xf16>
        %196 = vector.extract %119[24, %129] : f16 from vector<32x2xf16>
        %197 = vector.extract %119[25, %129] : f16 from vector<32x2xf16>
        %198 = vector.extract %119[26, %129] : f16 from vector<32x2xf16>
        %199 = vector.extract %119[27, %129] : f16 from vector<32x2xf16>
        %200 = vector.from_elements %196, %197, %198, %199 : vector<4x1xf16>
        %201 = frisk.warp_mma_rr %195, %200, %190 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %202 = vector.extract %112[%123, 28] : f16 from vector<2x32xf16>
        %203 = vector.extract %112[%123, 29] : f16 from vector<2x32xf16>
        %204 = vector.extract %112[%123, 30] : f16 from vector<2x32xf16>
        %205 = vector.extract %112[%123, 31] : f16 from vector<2x32xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<1x4xf16>
        %207 = vector.extract %119[28, %129] : f16 from vector<32x2xf16>
        %208 = vector.extract %119[29, %129] : f16 from vector<32x2xf16>
        %209 = vector.extract %119[30, %129] : f16 from vector<32x2xf16>
        %210 = vector.extract %119[31, %129] : f16 from vector<32x2xf16>
        %211 = vector.from_elements %207, %208, %209, %210 : vector<4x1xf16>
        %212 = frisk.warp_mma_rr %206, %211, %201 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %214 = vector.extract %212[0, 0] : f32 from vector<1x4xf32>
        %215 = vector.insert %214, %arg12 [%123, %213] : f32 into vector<2x8xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %217 = vector.extract %212[0, 1] : f32 from vector<1x4xf32>
        %218 = vector.insert %217, %215 [%123, %216] : f32 into vector<2x8xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %220 = vector.extract %212[0, 2] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %218 [%123, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %223 = vector.extract %212[0, 3] : f32 from vector<1x4xf32>
        %224 = vector.insert %223, %221 [%123, %222] : f32 into vector<2x8xf32>
        %225 = vector.extract %109#1[%123, 0] : f32 from vector<2x1xf32>
        %226 = vector.broadcast %225 : f32 to vector<1x4xf32>
        %227 = vector.insert %225, %arg13 [%123, %213] : f32 into vector<2x8xf32>
        %228 = vector.insert %225, %227 [%123, %216] : f32 into vector<2x8xf32>
        %229 = vector.insert %225, %228 [%123, %219] : f32 into vector<2x8xf32>
        %230 = vector.insert %225, %229 [%123, %222] : f32 into vector<2x8xf32>
        %231 = vector.extract %arg10[%123, %213] : f32 from vector<2x8xf32>
        %232 = vector.extract %arg10[%123, %216] : f32 from vector<2x8xf32>
        %233 = vector.extract %arg10[%123, %219] : f32 from vector<2x8xf32>
        %234 = vector.extract %arg10[%123, %222] : f32 from vector<2x8xf32>
        %235 = vector.from_elements %231, %232, %233, %234 : vector<1x4xf32>
        %236 = arith.mulf %235, %226 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %237 = vector.extract %236[0, 0] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %arg14 [%123, %213] : f32 into vector<2x8xf32>
        %239 = vector.extract %236[0, 1] : f32 from vector<1x4xf32>
        %240 = vector.insert %239, %238 [%123, %216] : f32 into vector<2x8xf32>
        %241 = vector.extract %236[0, 2] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %240 [%123, %219] : f32 into vector<2x8xf32>
        %243 = vector.extract %236[0, 3] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %242 [%123, %222] : f32 into vector<2x8xf32>
        %245 = arith.addf %236, %212 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %246 = vector.extract %245[0, 0] : f32 from vector<1x4xf32>
        %247 = vector.insert %246, %arg15 [%123, %213] : f32 into vector<2x8xf32>
        %248 = vector.extract %245[0, 1] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %247 [%123, %216] : f32 into vector<2x8xf32>
        %250 = vector.extract %245[0, 2] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %249 [%123, %219] : f32 into vector<2x8xf32>
        %252 = vector.extract %245[0, 3] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %251 [%123, %222] : f32 into vector<2x8xf32>
        %254 = vector.extract %114#3[%123, %213] : f32 from vector<2x8xf32>
        %255 = vector.extract %114#3[%123, %216] : f32 from vector<2x8xf32>
        %256 = vector.extract %114#3[%123, %219] : f32 from vector<2x8xf32>
        %257 = vector.extract %114#3[%123, %222] : f32 from vector<2x8xf32>
        %258 = vector.insert %254, %arg16 [%123, %213] : f32 into vector<2x8xf32>
        %259 = vector.insert %255, %258 [%123, %216] : f32 into vector<2x8xf32>
        %260 = vector.insert %256, %259 [%123, %219] : f32 into vector<2x8xf32>
        %261 = vector.insert %257, %260 [%123, %222] : f32 into vector<2x8xf32>
        affine.yield %224, %230, %244, %253, %261 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %121:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %123 = vector.extract %111#1[%arg11, 0] : f32 from vector<2x1xf32>
        %124 = vector.insert %123, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %125 = vector.extract %107#1[%arg11, 0] : f32 from vector<2x1xf32>
        %126 = vector.insert %125, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %124, %126 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %122:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_8, %arg13 = %cst_8, %arg14 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %123 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %125 = vector.extract %116#3[%123, %124] : f32 from vector<2x8xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %127 = vector.extract %116#3[%123, %126] : f32 from vector<2x8xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %129 = vector.extract %116#3[%123, %128] : f32 from vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %131 = vector.extract %116#3[%123, %130] : f32 from vector<2x8xf32>
        %132 = vector.insert %125, %arg12 [%123, %124] : f32 into vector<2x8xf32>
        %133 = vector.insert %127, %132 [%123, %126] : f32 into vector<2x8xf32>
        %134 = vector.insert %129, %133 [%123, %128] : f32 into vector<2x8xf32>
        %135 = vector.insert %131, %134 [%123, %130] : f32 into vector<2x8xf32>
        %136 = vector.extract %118#3[%123, %124] : f32 from vector<2x8xf32>
        %137 = vector.extract %118#3[%123, %126] : f32 from vector<2x8xf32>
        %138 = vector.extract %118#3[%123, %128] : f32 from vector<2x8xf32>
        %139 = vector.extract %118#3[%123, %130] : f32 from vector<2x8xf32>
        %140 = vector.insert %136, %arg13 [%123, %124] : f32 into vector<2x8xf32>
        %141 = vector.insert %137, %140 [%123, %126] : f32 into vector<2x8xf32>
        %142 = vector.insert %138, %141 [%123, %128] : f32 into vector<2x8xf32>
        %143 = vector.insert %139, %142 [%123, %130] : f32 into vector<2x8xf32>
        %144 = vector.extract %120#3[%123, %124] : f32 from vector<2x8xf32>
        %145 = vector.extract %120#3[%123, %126] : f32 from vector<2x8xf32>
        %146 = vector.extract %120#3[%123, %128] : f32 from vector<2x8xf32>
        %147 = vector.extract %120#3[%123, %130] : f32 from vector<2x8xf32>
        %148 = vector.insert %144, %arg14 [%123, %124] : f32 into vector<2x8xf32>
        %149 = vector.insert %145, %148 [%123, %126] : f32 into vector<2x8xf32>
        %150 = vector.insert %146, %149 [%123, %128] : f32 into vector<2x8xf32>
        %151 = vector.insert %147, %150 [%123, %130] : f32 into vector<2x8xf32>
        affine.yield %135, %143, %151 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      gpu.barrier
      affine.yield %120#4, %121#0, %121#1, %122#0, %122#1, %122#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %87 = arith.index_cast %thread_id_x : index to i32
    %88 = arith.andi %87, %c16_i32 : i32
    %89 = arith.cmpi ne, %88, %c0_i32 : i32
    %90 = arith.andi %87, %c32_i32 : i32
    %91 = arith.cmpi ne, %90, %c0_i32 : i32
    %92:8 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_8, %arg8 = %cst_8, %arg9 = %cst_8, %arg10 = %cst_8, %arg11 = %cst_2, %arg12 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %100 = vector.extract %86#5[%98, %99] : f32 from vector<2x8xf32>
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %102 = vector.extract %86#5[%98, %101] : f32 from vector<2x8xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %104 = vector.extract %86#5[%98, %103] : f32 from vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %106 = vector.extract %86#5[%98, %105] : f32 from vector<2x8xf32>
      %107 = arith.select %89, %100, %102 : f32
      %shuffleResult, %valid = gpu.shuffle xor %107, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %108 = arith.select %89, %shuffleResult, %100 : f32
      %109 = arith.select %89, %102, %shuffleResult : f32
      %110 = arith.select %89, %104, %106 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %110, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %89, %shuffleResult_13, %104 : f32
      %112 = arith.select %89, %106, %shuffleResult_13 : f32
      %113 = arith.select %91, %108, %111 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %113, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %91, %shuffleResult_15, %108 : f32
      %115 = arith.select %91, %111, %shuffleResult_15 : f32
      %116 = arith.select %91, %109, %112 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %116, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %91, %shuffleResult_17, %109 : f32
      %118 = arith.select %91, %112, %shuffleResult_17 : f32
      %119 = vector.insert %114, %arg5 [%98, %99] : f32 into vector<2x8xf32>
      %120 = vector.insert %117, %119 [%98, %101] : f32 into vector<2x8xf32>
      %121 = vector.insert %115, %120 [%98, %103] : f32 into vector<2x8xf32>
      %122 = vector.insert %118, %121 [%98, %105] : f32 into vector<2x8xf32>
      %123 = vector.extract %86#4[%98, %99] : f32 from vector<2x8xf32>
      %124 = vector.extract %86#4[%98, %101] : f32 from vector<2x8xf32>
      %125 = vector.extract %86#4[%98, %103] : f32 from vector<2x8xf32>
      %126 = vector.extract %86#4[%98, %105] : f32 from vector<2x8xf32>
      %127 = arith.select %89, %123, %124 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %127, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %128 = arith.select %89, %shuffleResult_19, %123 : f32
      %129 = arith.select %89, %124, %shuffleResult_19 : f32
      %130 = arith.select %89, %125, %126 : f32
      %shuffleResult_21, %valid_22 = gpu.shuffle xor %130, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %131 = arith.select %89, %shuffleResult_21, %125 : f32
      %132 = arith.select %89, %126, %shuffleResult_21 : f32
      %133 = arith.select %91, %128, %131 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %133, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %134 = arith.select %91, %shuffleResult_23, %128 : f32
      %135 = arith.select %91, %131, %shuffleResult_23 : f32
      %136 = arith.select %91, %129, %132 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %136, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %137 = arith.select %91, %shuffleResult_25, %129 : f32
      %138 = arith.select %91, %132, %shuffleResult_25 : f32
      %139 = vector.insert %134, %arg6 [%98, %99] : f32 into vector<2x8xf32>
      %140 = vector.insert %137, %139 [%98, %101] : f32 into vector<2x8xf32>
      %141 = vector.insert %135, %140 [%98, %103] : f32 into vector<2x8xf32>
      %142 = vector.insert %138, %141 [%98, %105] : f32 into vector<2x8xf32>
      %143 = vector.extract %86#3[%98, %99] : f32 from vector<2x8xf32>
      %144 = vector.extract %86#3[%98, %101] : f32 from vector<2x8xf32>
      %145 = vector.extract %86#3[%98, %103] : f32 from vector<2x8xf32>
      %146 = vector.extract %86#3[%98, %105] : f32 from vector<2x8xf32>
      %147 = arith.select %89, %143, %144 : f32
      %shuffleResult_27, %valid_28 = gpu.shuffle xor %147, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %148 = arith.select %89, %shuffleResult_27, %143 : f32
      %149 = arith.select %89, %144, %shuffleResult_27 : f32
      %150 = arith.select %89, %145, %146 : f32
      %shuffleResult_29, %valid_30 = gpu.shuffle xor %150, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %151 = arith.select %89, %shuffleResult_29, %145 : f32
      %152 = arith.select %89, %146, %shuffleResult_29 : f32
      %153 = arith.select %91, %148, %151 : f32
      %shuffleResult_31, %valid_32 = gpu.shuffle xor %153, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %154 = arith.select %91, %shuffleResult_31, %148 : f32
      %155 = arith.select %91, %151, %shuffleResult_31 : f32
      %156 = arith.select %91, %149, %152 : f32
      %shuffleResult_33, %valid_34 = gpu.shuffle xor %156, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %157 = arith.select %91, %shuffleResult_33, %149 : f32
      %158 = arith.select %91, %152, %shuffleResult_33 : f32
      %159 = vector.insert %154, %arg7 [%98, %99] : f32 into vector<2x8xf32>
      %160 = vector.insert %157, %159 [%98, %101] : f32 into vector<2x8xf32>
      %161 = vector.insert %155, %160 [%98, %103] : f32 into vector<2x8xf32>
      %162 = vector.insert %158, %161 [%98, %105] : f32 into vector<2x8xf32>
      %163 = vector.extract %86#0[%98, %99] : f32 from vector<2x8xf32>
      %164 = vector.extract %86#0[%98, %101] : f32 from vector<2x8xf32>
      %165 = vector.extract %86#0[%98, %103] : f32 from vector<2x8xf32>
      %166 = vector.extract %86#0[%98, %105] : f32 from vector<2x8xf32>
      %167 = arith.select %89, %163, %164 : f32
      %shuffleResult_35, %valid_36 = gpu.shuffle xor %167, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %168 = arith.select %89, %shuffleResult_35, %163 : f32
      %169 = arith.select %89, %164, %shuffleResult_35 : f32
      %170 = arith.select %89, %165, %166 : f32
      %shuffleResult_37, %valid_38 = gpu.shuffle xor %170, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %171 = arith.select %89, %shuffleResult_37, %165 : f32
      %172 = arith.select %89, %166, %shuffleResult_37 : f32
      %173 = arith.select %91, %168, %171 : f32
      %shuffleResult_39, %valid_40 = gpu.shuffle xor %173, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %174 = arith.select %91, %shuffleResult_39, %168 : f32
      %175 = arith.select %91, %171, %shuffleResult_39 : f32
      %176 = arith.select %91, %169, %172 : f32
      %shuffleResult_41, %valid_42 = gpu.shuffle xor %176, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %177 = arith.select %91, %shuffleResult_41, %169 : f32
      %178 = arith.select %91, %172, %shuffleResult_41 : f32
      %179 = vector.from_elements %174, %177, %175, %178 : vector<1x4xf32>
      %180 = vector.insert %174, %arg8 [%98, %99] : f32 into vector<2x8xf32>
      %181 = vector.insert %177, %180 [%98, %101] : f32 into vector<2x8xf32>
      %182 = vector.insert %175, %181 [%98, %103] : f32 into vector<2x8xf32>
      %183 = vector.insert %178, %182 [%98, %105] : f32 into vector<2x8xf32>
      %184 = vector.extract %86#1[%98, 0] : f32 from vector<2x1xf32>
      %185 = vector.broadcast %184 : f32 to vector<1x4xf32>
      %186 = vector.insert %184, %arg9 [%98, %99] : f32 into vector<2x8xf32>
      %187 = vector.insert %184, %186 [%98, %101] : f32 into vector<2x8xf32>
      %188 = vector.insert %184, %187 [%98, %103] : f32 into vector<2x8xf32>
      %189 = vector.insert %184, %188 [%98, %105] : f32 into vector<2x8xf32>
      %190 = arith.divf %179, %185 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %191 = vector.extract %190[0, 0] : f32 from vector<1x4xf32>
      %192 = vector.insert %191, %arg10 [%98, %99] : f32 into vector<2x8xf32>
      %193 = vector.extract %190[0, 1] : f32 from vector<1x4xf32>
      %194 = vector.insert %193, %192 [%98, %101] : f32 into vector<2x8xf32>
      %195 = vector.extract %190[0, 2] : f32 from vector<1x4xf32>
      %196 = vector.insert %195, %194 [%98, %103] : f32 into vector<2x8xf32>
      %197 = vector.extract %190[0, 3] : f32 from vector<1x4xf32>
      %198 = vector.insert %197, %196 [%98, %105] : f32 into vector<2x8xf32>
      %199 = arith.truncf %190 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %200 = vector.extract %199[0, 0] : f16 from vector<1x4xf16>
      %201 = vector.insert %200, %arg11 [%98, %99] : f16 into vector<2x8xf16>
      %202 = vector.extract %199[0, 1] : f16 from vector<1x4xf16>
      %203 = vector.insert %202, %201 [%98, %101] : f16 into vector<2x8xf16>
      %204 = vector.extract %199[0, 2] : f16 from vector<1x4xf16>
      %205 = vector.insert %204, %203 [%98, %103] : f16 into vector<2x8xf16>
      %206 = vector.extract %199[0, 3] : f16 from vector<1x4xf16>
      %207 = vector.insert %206, %205 [%98, %105] : f16 into vector<2x8xf16>
      %208 = vector.insert %200, %arg12 [%98, %99] : f16 into vector<2x8xf16>
      %209 = vector.insert %202, %208 [%98, %101] : f16 into vector<2x8xf16>
      %210 = vector.insert %204, %209 [%98, %103] : f16 into vector<2x8xf16>
      %211 = vector.insert %206, %210 [%98, %105] : f16 into vector<2x8xf16>
      affine.yield %122, %142, %162, %183, %189, %198, %207, %211 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    %93 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %94 = arith.cmpi ult, %93, %c4 : index
    scf.if %94 {
      affine.for %arg4 = 0 to 4 {
        %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %100 = vector.extract %92#7[%98, %99] : f16 from vector<2x8xf16>
        %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %102 = vector.extract %92#7[%98, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %104 = vector.extract %92#7[%98, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %106 = vector.extract %92#7[%98, %105] : f16 from vector<2x8xf16>
        affine.store %100, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %102, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %95:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %99 = vector.extract %86#1[%98, 0] : f32 from vector<2x1xf32>
      %100 = vector.broadcast %99 : f32 to vector<1x4xf32>
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %102 = vector.insert %99, %arg5 [%98, %101] : f32 into vector<2x8xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %104 = vector.insert %99, %102 [%98, %103] : f32 into vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %106 = vector.insert %99, %104 [%98, %105] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %108 = vector.insert %99, %106 [%98, %107] : f32 into vector<2x8xf32>
      %109 = vector.extract %92#2[%98, %101] : f32 from vector<2x8xf32>
      %110 = vector.extract %92#2[%98, %103] : f32 from vector<2x8xf32>
      %111 = vector.extract %92#2[%98, %105] : f32 from vector<2x8xf32>
      %112 = vector.extract %92#2[%98, %107] : f32 from vector<2x8xf32>
      %113 = vector.from_elements %109, %110, %111, %112 : vector<1x4xf32>
      %114 = arith.divf %113, %100 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %115 = vector.extract %114[0, 0] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %arg6 [%98, %101] : f32 into vector<2x8xf32>
      %117 = vector.extract %114[0, 1] : f32 from vector<1x4xf32>
      %118 = vector.insert %117, %116 [%98, %103] : f32 into vector<2x8xf32>
      %119 = vector.extract %114[0, 2] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %118 [%98, %105] : f32 into vector<2x8xf32>
      %121 = vector.extract %114[0, 3] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%98, %107] : f32 into vector<2x8xf32>
      %123 = arith.truncf %114 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %124 = vector.extract %123[0, 0] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %arg7 [%98, %101] : f16 into vector<2x8xf16>
      %126 = vector.extract %123[0, 1] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %125 [%98, %103] : f16 into vector<2x8xf16>
      %128 = vector.extract %123[0, 2] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%98, %105] : f16 into vector<2x8xf16>
      %130 = vector.extract %123[0, 3] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%98, %107] : f16 into vector<2x8xf16>
      %132 = vector.insert %124, %arg8 [%98, %101] : f16 into vector<2x8xf16>
      %133 = vector.insert %126, %132 [%98, %103] : f16 into vector<2x8xf16>
      %134 = vector.insert %128, %133 [%98, %105] : f16 into vector<2x8xf16>
      %135 = vector.insert %130, %134 [%98, %107] : f16 into vector<2x8xf16>
      affine.yield %108, %122, %131, %135 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %94 {
      affine.for %arg4 = 0 to 4 {
        %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %100 = vector.extract %95#3[%98, %99] : f16 from vector<2x8xf16>
        %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %102 = vector.extract %95#3[%98, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %104 = vector.extract %95#3[%98, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %106 = vector.extract %95#3[%98, %105] : f16 from vector<2x8xf16>
        affine.store %100, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %102, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %96:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %99 = vector.extract %86#1[%98, 0] : f32 from vector<2x1xf32>
      %100 = vector.broadcast %99 : f32 to vector<1x4xf32>
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %102 = vector.insert %99, %arg5 [%98, %101] : f32 into vector<2x8xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %104 = vector.insert %99, %102 [%98, %103] : f32 into vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %106 = vector.insert %99, %104 [%98, %105] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %108 = vector.insert %99, %106 [%98, %107] : f32 into vector<2x8xf32>
      %109 = vector.extract %92#1[%98, %101] : f32 from vector<2x8xf32>
      %110 = vector.extract %92#1[%98, %103] : f32 from vector<2x8xf32>
      %111 = vector.extract %92#1[%98, %105] : f32 from vector<2x8xf32>
      %112 = vector.extract %92#1[%98, %107] : f32 from vector<2x8xf32>
      %113 = vector.from_elements %109, %110, %111, %112 : vector<1x4xf32>
      %114 = arith.divf %113, %100 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %115 = vector.extract %114[0, 0] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %arg6 [%98, %101] : f32 into vector<2x8xf32>
      %117 = vector.extract %114[0, 1] : f32 from vector<1x4xf32>
      %118 = vector.insert %117, %116 [%98, %103] : f32 into vector<2x8xf32>
      %119 = vector.extract %114[0, 2] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %118 [%98, %105] : f32 into vector<2x8xf32>
      %121 = vector.extract %114[0, 3] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%98, %107] : f32 into vector<2x8xf32>
      %123 = arith.truncf %114 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %124 = vector.extract %123[0, 0] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %arg7 [%98, %101] : f16 into vector<2x8xf16>
      %126 = vector.extract %123[0, 1] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %125 [%98, %103] : f16 into vector<2x8xf16>
      %128 = vector.extract %123[0, 2] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%98, %105] : f16 into vector<2x8xf16>
      %130 = vector.extract %123[0, 3] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%98, %107] : f16 into vector<2x8xf16>
      %132 = vector.insert %124, %arg8 [%98, %101] : f16 into vector<2x8xf16>
      %133 = vector.insert %126, %132 [%98, %103] : f16 into vector<2x8xf16>
      %134 = vector.insert %128, %133 [%98, %105] : f16 into vector<2x8xf16>
      %135 = vector.insert %130, %134 [%98, %107] : f16 into vector<2x8xf16>
      affine.yield %108, %122, %131, %135 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %94 {
      affine.for %arg4 = 0 to 4 {
        %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %100 = vector.extract %96#3[%98, %99] : f16 from vector<2x8xf16>
        %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %102 = vector.extract %96#3[%98, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %104 = vector.extract %96#3[%98, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %106 = vector.extract %96#3[%98, %105] : f16 from vector<2x8xf16>
        affine.store %100, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %102, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %97:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %99 = vector.extract %86#1[%98, 0] : f32 from vector<2x1xf32>
      %100 = vector.broadcast %99 : f32 to vector<1x4xf32>
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %102 = vector.insert %99, %arg5 [%98, %101] : f32 into vector<2x8xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %104 = vector.insert %99, %102 [%98, %103] : f32 into vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %106 = vector.insert %99, %104 [%98, %105] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %108 = vector.insert %99, %106 [%98, %107] : f32 into vector<2x8xf32>
      %109 = vector.extract %92#0[%98, %101] : f32 from vector<2x8xf32>
      %110 = vector.extract %92#0[%98, %103] : f32 from vector<2x8xf32>
      %111 = vector.extract %92#0[%98, %105] : f32 from vector<2x8xf32>
      %112 = vector.extract %92#0[%98, %107] : f32 from vector<2x8xf32>
      %113 = vector.from_elements %109, %110, %111, %112 : vector<1x4xf32>
      %114 = arith.divf %113, %100 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %115 = vector.extract %114[0, 0] : f32 from vector<1x4xf32>
      %116 = vector.insert %115, %arg6 [%98, %101] : f32 into vector<2x8xf32>
      %117 = vector.extract %114[0, 1] : f32 from vector<1x4xf32>
      %118 = vector.insert %117, %116 [%98, %103] : f32 into vector<2x8xf32>
      %119 = vector.extract %114[0, 2] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %118 [%98, %105] : f32 into vector<2x8xf32>
      %121 = vector.extract %114[0, 3] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%98, %107] : f32 into vector<2x8xf32>
      %123 = arith.truncf %114 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %124 = vector.extract %123[0, 0] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %arg7 [%98, %101] : f16 into vector<2x8xf16>
      %126 = vector.extract %123[0, 1] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %125 [%98, %103] : f16 into vector<2x8xf16>
      %128 = vector.extract %123[0, 2] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%98, %105] : f16 into vector<2x8xf16>
      %130 = vector.extract %123[0, 3] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%98, %107] : f16 into vector<2x8xf16>
      %132 = vector.insert %124, %arg8 [%98, %101] : f16 into vector<2x8xf16>
      %133 = vector.insert %126, %132 [%98, %103] : f16 into vector<2x8xf16>
      %134 = vector.insert %128, %133 [%98, %105] : f16 into vector<2x8xf16>
      %135 = vector.insert %130, %134 [%98, %107] : f16 into vector<2x8xf16>
      affine.yield %108, %122, %131, %135 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %94 {
      affine.for %arg4 = 0 to 4 {
        %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %100 = vector.extract %97#3[%98, %99] : f16 from vector<2x8xf16>
        %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %102 = vector.extract %97#3[%98, %101] : f16 from vector<2x8xf16>
        %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %104 = vector.extract %97#3[%98, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %106 = vector.extract %97#3[%98, %105] : f16 from vector<2x8xf16>
        affine.store %100, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %102, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %104, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
