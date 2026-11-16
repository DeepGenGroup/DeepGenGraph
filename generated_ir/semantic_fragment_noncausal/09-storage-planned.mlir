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
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %c4 = arith.constant 4 : index
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_8 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
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
    %11:4 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1, %arg6 = %cst_1, %arg7 = %cst_1, %arg8 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %102 = vector.insert %cst_8, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %103 = vector.insert %cst_0, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %104 = vector.insert %cst_0, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %105 = vector.insert %cst_0, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %102, %103, %104, %105 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %12 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %104 = vector.insert %cst_8, %arg5 [%103, %102] : f32 into vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %106 = vector.insert %cst_8, %104 [%103, %105] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %108 = vector.insert %cst_8, %106 [%103, %107] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %110 = vector.insert %cst_8, %108 [%103, %109] : f32 into vector<2x8xf32>
      affine.yield %110 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %13 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1) -> (vector<2x1xf32>) {
      %102 = vector.insert %cst_8, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %102 : vector<2x1xf32>
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
    %25:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %23) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %103 = memref.load %view[%102] : memref<4096xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %105 = memref.load %view[%104] : memref<4096xf16, 3>
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %107 = memref.load %view[%106] : memref<4096xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %109 = memref.load %view[%108] : memref<4096xf16, 3>
      %110 = vector.from_elements %103, %105, %107, %109 : vector<1x4xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %113 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %arg5 [%111, %112] : f16 into vector<2x8xf16>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %116 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %114 [%111, %115] : f16 into vector<2x8xf16>
      %118 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %119 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %117 [%111, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %122 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%111, %121] : f16 into vector<2x8xf16>
      %124 = vector.insert %10, %arg6 [%111, %112] : f16 into vector<2x8xf16>
      %125 = vector.insert %10, %124 [%111, %115] : f16 into vector<2x8xf16>
      %126 = vector.insert %10, %125 [%111, %118] : f16 into vector<2x8xf16>
      %127 = vector.insert %10, %126 [%111, %121] : f16 into vector<2x8xf16>
      %128 = arith.mulf %arg8, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %129 = vector.extract %128[0, 0] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %arg7 [%111, %112] : f16 into vector<2x8xf16>
      %131 = vector.extract %128[0, 1] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%111, %115] : f16 into vector<2x8xf16>
      %133 = vector.extract %128[0, 2] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%111, %118] : f16 into vector<2x8xf16>
      %135 = vector.extract %128[0, 3] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%111, %121] : f16 into vector<2x8xf16>
      affine.yield %123, %127, %136, %110 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %26 = arith.mulf %25#3, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %27 = vector.extract %26[0, 0] : f16 from vector<1x4xf16>
    %28 = vector.insert %27, %25#2 [1, 4] : f16 into vector<2x8xf16>
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
    %40:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %39) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %103 = memref.load %view_10[%102] : memref<4096xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %105 = memref.load %view_10[%104] : memref<4096xf16, 3>
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %107 = memref.load %view_10[%106] : memref<4096xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %109 = memref.load %view_10[%108] : memref<4096xf16, 3>
      %110 = vector.from_elements %103, %105, %107, %109 : vector<1x4xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %113 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %arg5 [%111, %112] : f16 into vector<2x8xf16>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %116 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %114 [%111, %115] : f16 into vector<2x8xf16>
      %118 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %119 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %117 [%111, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %122 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%111, %121] : f16 into vector<2x8xf16>
      %124 = vector.insert %10, %arg6 [%111, %112] : f16 into vector<2x8xf16>
      %125 = vector.insert %10, %124 [%111, %115] : f16 into vector<2x8xf16>
      %126 = vector.insert %10, %125 [%111, %118] : f16 into vector<2x8xf16>
      %127 = vector.insert %10, %126 [%111, %121] : f16 into vector<2x8xf16>
      %128 = arith.mulf %arg8, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %129 = vector.extract %128[0, 0] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %arg7 [%111, %112] : f16 into vector<2x8xf16>
      %131 = vector.extract %128[0, 1] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%111, %115] : f16 into vector<2x8xf16>
      %133 = vector.extract %128[0, 2] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%111, %118] : f16 into vector<2x8xf16>
      %135 = vector.extract %128[0, 3] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%111, %121] : f16 into vector<2x8xf16>
      affine.yield %123, %127, %136, %110 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %41 = arith.mulf %40#3, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %42 = vector.extract %41[0, 0] : f16 from vector<1x4xf16>
    %43 = vector.insert %42, %40#2 [1, 4] : f16 into vector<2x8xf16>
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
    %55:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %54) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %103 = memref.load %view[%102] : memref<4096xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %105 = memref.load %view[%104] : memref<4096xf16, 3>
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %107 = memref.load %view[%106] : memref<4096xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %109 = memref.load %view[%108] : memref<4096xf16, 3>
      %110 = vector.from_elements %103, %105, %107, %109 : vector<1x4xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %113 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %arg5 [%111, %112] : f16 into vector<2x8xf16>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %116 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %114 [%111, %115] : f16 into vector<2x8xf16>
      %118 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %119 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %117 [%111, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %122 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%111, %121] : f16 into vector<2x8xf16>
      %124 = vector.insert %10, %arg6 [%111, %112] : f16 into vector<2x8xf16>
      %125 = vector.insert %10, %124 [%111, %115] : f16 into vector<2x8xf16>
      %126 = vector.insert %10, %125 [%111, %118] : f16 into vector<2x8xf16>
      %127 = vector.insert %10, %126 [%111, %121] : f16 into vector<2x8xf16>
      %128 = arith.mulf %arg8, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %129 = vector.extract %128[0, 0] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %arg7 [%111, %112] : f16 into vector<2x8xf16>
      %131 = vector.extract %128[0, 1] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%111, %115] : f16 into vector<2x8xf16>
      %133 = vector.extract %128[0, 2] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%111, %118] : f16 into vector<2x8xf16>
      %135 = vector.extract %128[0, 3] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%111, %121] : f16 into vector<2x8xf16>
      affine.yield %123, %127, %136, %110 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %56 = arith.mulf %55#3, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %57 = vector.extract %56[0, 0] : f16 from vector<1x4xf16>
    %58 = vector.insert %57, %55#2 [1, 4] : f16 into vector<2x8xf16>
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
    %70:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %69) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %103 = memref.load %view_10[%102] : memref<4096xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %105 = memref.load %view_10[%104] : memref<4096xf16, 3>
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %107 = memref.load %view_10[%106] : memref<4096xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %109 = memref.load %view_10[%108] : memref<4096xf16, 3>
      %110 = vector.from_elements %103, %105, %107, %109 : vector<1x4xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %113 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %114 = vector.insert %113, %arg5 [%111, %112] : f16 into vector<2x8xf16>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %116 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %117 = vector.insert %116, %114 [%111, %115] : f16 into vector<2x8xf16>
      %118 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %119 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %117 [%111, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %122 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%111, %121] : f16 into vector<2x8xf16>
      %124 = vector.insert %10, %arg6 [%111, %112] : f16 into vector<2x8xf16>
      %125 = vector.insert %10, %124 [%111, %115] : f16 into vector<2x8xf16>
      %126 = vector.insert %10, %125 [%111, %118] : f16 into vector<2x8xf16>
      %127 = vector.insert %10, %126 [%111, %121] : f16 into vector<2x8xf16>
      %128 = arith.mulf %arg8, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %129 = vector.extract %128[0, 0] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %arg7 [%111, %112] : f16 into vector<2x8xf16>
      %131 = vector.extract %128[0, 1] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%111, %115] : f16 into vector<2x8xf16>
      %133 = vector.extract %128[0, 2] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%111, %118] : f16 into vector<2x8xf16>
      %135 = vector.extract %128[0, 3] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%111, %121] : f16 into vector<2x8xf16>
      affine.yield %123, %127, %136, %110 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %71 = arith.mulf %70#3, %24 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %72 = vector.extract %71[0, 0] : f16 from vector<1x4xf16>
    %73 = vector.insert %72, %70#2 [1, 4] : f16 into vector<2x8xf16>
    %74 = vector.extract %71[0, 1] : f16 from vector<1x4xf16>
    %75 = vector.insert %74, %73 [1, 5] : f16 into vector<2x8xf16>
    %76 = vector.extract %71[0, 2] : f16 from vector<1x4xf16>
    %77 = vector.insert %76, %75 [1, 6] : f16 into vector<2x8xf16>
    %78 = vector.extract %71[0, 3] : f16 from vector<1x4xf16>
    %79 = vector.insert %78, %77 [1, 7] : f16 into vector<2x8xf16>
    gpu.barrier
    %80 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
    %81 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_3) -> (vector<2x32xf32>) {
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg4)
      %104 = vector.insert %cst_8, %arg5 [%103, %102] : f32 into vector<2x32xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %106 = vector.insert %cst_8, %104 [%103, %105] : f32 into vector<2x32xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %108 = vector.insert %cst_8, %106 [%103, %107] : f32 into vector<2x32xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %110 = vector.insert %cst_8, %108 [%103, %109] : f32 into vector<2x32xf32>
      affine.yield %110 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
    %82 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
    %83 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
    %84 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
    %85 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
    %86:6 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %12, %arg6 = %13, %arg7 = %11#3, %arg8 = %12, %arg9 = %12, %arg10 = %12) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %102 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %subview_13 = memref.subview %transpose[0, %80, 0, %102] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_13 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %103 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_14 = memref.subview %transpose[0, %103, 32, %102] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_14 to %view_10 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
      %104:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %130 = memref.load %view[%129] : memref<4096xf16, 3>
        %131 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %132 = memref.load %view[%131] : memref<4096xf16, 3>
        %133 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %134 = memref.load %view[%133] : memref<4096xf16, 3>
        %135 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %136 = memref.load %view[%135] : memref<4096xf16, 3>
        %137 = vector.from_elements %130, %132, %134, %136 : vector<4x1xf16>
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %139 = memref.load %view[%138] : memref<4096xf16, 3>
        %140 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %141 = memref.load %view[%140] : memref<4096xf16, 3>
        %142 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %143 = memref.load %view[%142] : memref<4096xf16, 3>
        %144 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %145 = memref.load %view[%144] : memref<4096xf16, 3>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<4x1xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %148 = vector.extract %34[%147, 0] : f16 from vector<2x8xf16>
        %149 = vector.extract %34[%147, 1] : f16 from vector<2x8xf16>
        %150 = vector.extract %34[%147, 2] : f16 from vector<2x8xf16>
        %151 = vector.extract %34[%147, 3] : f16 from vector<2x8xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf16>
        %153 = frisk.warp_mma_rr %152, %137, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = vector.extract %34[%147, 4] : f16 from vector<2x8xf16>
        %155 = vector.extract %34[%147, 5] : f16 from vector<2x8xf16>
        %156 = vector.extract %34[%147, 6] : f16 from vector<2x8xf16>
        %157 = vector.extract %34[%147, 7] : f16 from vector<2x8xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = frisk.warp_mma_rr %158, %146, %153 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %161 = vector.extract %81[%147, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %163 = vector.extract %81[%147, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %165 = vector.extract %81[%147, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %81[%147, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = arith.addf %168, %159 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %170 = vector.extract %169[0, 0] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %arg12 [%147, %160] : f32 into vector<2x32xf32>
        %172 = vector.extract %169[0, 1] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%147, %162] : f32 into vector<2x32xf32>
        %174 = vector.extract %169[0, 2] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%147, %164] : f32 into vector<2x32xf32>
        %176 = vector.extract %169[0, 3] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%147, %166] : f32 into vector<2x32xf32>
        %178 = vector.insert %170, %arg13 [%147, %160] : f32 into vector<2x32xf32>
        %179 = vector.insert %172, %178 [%147, %162] : f32 into vector<2x32xf32>
        %180 = vector.insert %174, %179 [%147, %164] : f32 into vector<2x32xf32>
        %181 = vector.insert %176, %180 [%147, %166] : f32 into vector<2x32xf32>
        affine.yield %177, %181 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %view_10 : memref<4096xf16, 3>
      %105 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 2048) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_15 = memref.subview %transpose[0, %105, 64, %102] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_15 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
      %106:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %130 = memref.load %view_10[%129] : memref<4096xf16, 3>
        %131 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %132 = memref.load %view_10[%131] : memref<4096xf16, 3>
        %133 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %134 = memref.load %view_10[%133] : memref<4096xf16, 3>
        %135 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %136 = memref.load %view_10[%135] : memref<4096xf16, 3>
        %137 = vector.from_elements %130, %132, %134, %136 : vector<4x1xf16>
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %139 = memref.load %view_10[%138] : memref<4096xf16, 3>
        %140 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %141 = memref.load %view_10[%140] : memref<4096xf16, 3>
        %142 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %143 = memref.load %view_10[%142] : memref<4096xf16, 3>
        %144 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %145 = memref.load %view_10[%144] : memref<4096xf16, 3>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<4x1xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %148 = vector.extract %49[%147, 0] : f16 from vector<2x8xf16>
        %149 = vector.extract %49[%147, 1] : f16 from vector<2x8xf16>
        %150 = vector.extract %49[%147, 2] : f16 from vector<2x8xf16>
        %151 = vector.extract %49[%147, 3] : f16 from vector<2x8xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf16>
        %153 = frisk.warp_mma_rr %152, %137, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = vector.extract %49[%147, 4] : f16 from vector<2x8xf16>
        %155 = vector.extract %49[%147, 5] : f16 from vector<2x8xf16>
        %156 = vector.extract %49[%147, 6] : f16 from vector<2x8xf16>
        %157 = vector.extract %49[%147, 7] : f16 from vector<2x8xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = frisk.warp_mma_rr %158, %146, %153 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %161 = vector.extract %104#1[%147, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %163 = vector.extract %104#1[%147, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %165 = vector.extract %104#1[%147, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %104#1[%147, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = arith.addf %168, %159 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %170 = vector.extract %169[0, 0] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %arg12 [%147, %160] : f32 into vector<2x32xf32>
        %172 = vector.extract %169[0, 1] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%147, %162] : f32 into vector<2x32xf32>
        %174 = vector.extract %169[0, 2] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%147, %164] : f32 into vector<2x32xf32>
        %176 = vector.extract %169[0, 3] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%147, %166] : f32 into vector<2x32xf32>
        %178 = vector.insert %170, %arg13 [%147, %160] : f32 into vector<2x32xf32>
        %179 = vector.insert %172, %178 [%147, %162] : f32 into vector<2x32xf32>
        %180 = vector.insert %174, %179 [%147, %164] : f32 into vector<2x32xf32>
        %181 = vector.insert %176, %180 [%147, %166] : f32 into vector<2x32xf32>
        affine.yield %177, %181 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %107 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 3072) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_16 = memref.subview %transpose[0, %107, 96, %102] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_16 to %view_10 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
      %108:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %130 = memref.load %view[%129] : memref<4096xf16, 3>
        %131 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %132 = memref.load %view[%131] : memref<4096xf16, 3>
        %133 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %134 = memref.load %view[%133] : memref<4096xf16, 3>
        %135 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %136 = memref.load %view[%135] : memref<4096xf16, 3>
        %137 = vector.from_elements %130, %132, %134, %136 : vector<4x1xf16>
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %139 = memref.load %view[%138] : memref<4096xf16, 3>
        %140 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %141 = memref.load %view[%140] : memref<4096xf16, 3>
        %142 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %143 = memref.load %view[%142] : memref<4096xf16, 3>
        %144 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %145 = memref.load %view[%144] : memref<4096xf16, 3>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<4x1xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %148 = vector.extract %64[%147, 0] : f16 from vector<2x8xf16>
        %149 = vector.extract %64[%147, 1] : f16 from vector<2x8xf16>
        %150 = vector.extract %64[%147, 2] : f16 from vector<2x8xf16>
        %151 = vector.extract %64[%147, 3] : f16 from vector<2x8xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf16>
        %153 = frisk.warp_mma_rr %152, %137, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = vector.extract %64[%147, 4] : f16 from vector<2x8xf16>
        %155 = vector.extract %64[%147, 5] : f16 from vector<2x8xf16>
        %156 = vector.extract %64[%147, 6] : f16 from vector<2x8xf16>
        %157 = vector.extract %64[%147, 7] : f16 from vector<2x8xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = frisk.warp_mma_rr %158, %146, %153 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %161 = vector.extract %106#1[%147, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %163 = vector.extract %106#1[%147, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %165 = vector.extract %106#1[%147, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %106#1[%147, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = arith.addf %168, %159 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %170 = vector.extract %169[0, 0] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %arg12 [%147, %160] : f32 into vector<2x32xf32>
        %172 = vector.extract %169[0, 1] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%147, %162] : f32 into vector<2x32xf32>
        %174 = vector.extract %169[0, 2] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%147, %164] : f32 into vector<2x32xf32>
        %176 = vector.extract %169[0, 3] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%147, %166] : f32 into vector<2x32xf32>
        %178 = vector.insert %170, %arg13 [%147, %160] : f32 into vector<2x32xf32>
        %179 = vector.insert %172, %178 [%147, %162] : f32 into vector<2x32xf32>
        %180 = vector.insert %174, %179 [%147, %164] : f32 into vector<2x32xf32>
        %181 = vector.insert %176, %180 [%147, %166] : f32 into vector<2x32xf32>
        affine.yield %177, %181 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %view_10 : memref<4096xf16, 3>
      %subview_17 = memref.subview %arg2[0, %80, %102, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_17 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %109:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %130 = memref.load %view_10[%129] : memref<4096xf16, 3>
        %131 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %132 = memref.load %view_10[%131] : memref<4096xf16, 3>
        %133 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %134 = memref.load %view_10[%133] : memref<4096xf16, 3>
        %135 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %136 = memref.load %view_10[%135] : memref<4096xf16, 3>
        %137 = vector.from_elements %130, %132, %134, %136 : vector<4x1xf16>
        %138 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %139 = memref.load %view_10[%138] : memref<4096xf16, 3>
        %140 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %141 = memref.load %view_10[%140] : memref<4096xf16, 3>
        %142 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %143 = memref.load %view_10[%142] : memref<4096xf16, 3>
        %144 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %145 = memref.load %view_10[%144] : memref<4096xf16, 3>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<4x1xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %148 = vector.extract %79[%147, 0] : f16 from vector<2x8xf16>
        %149 = vector.extract %79[%147, 1] : f16 from vector<2x8xf16>
        %150 = vector.extract %79[%147, 2] : f16 from vector<2x8xf16>
        %151 = vector.extract %79[%147, 3] : f16 from vector<2x8xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<1x4xf16>
        %153 = frisk.warp_mma_rr %152, %137, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = vector.extract %79[%147, 4] : f16 from vector<2x8xf16>
        %155 = vector.extract %79[%147, 5] : f16 from vector<2x8xf16>
        %156 = vector.extract %79[%147, 6] : f16 from vector<2x8xf16>
        %157 = vector.extract %79[%147, 7] : f16 from vector<2x8xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = frisk.warp_mma_rr %158, %146, %153 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %161 = vector.extract %108#1[%147, %160] : f32 from vector<2x32xf32>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %163 = vector.extract %108#1[%147, %162] : f32 from vector<2x32xf32>
        %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %165 = vector.extract %108#1[%147, %164] : f32 from vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %108#1[%147, %166] : f32 from vector<2x32xf32>
        %168 = vector.from_elements %161, %163, %165, %167 : vector<1x4xf32>
        %169 = arith.addf %168, %159 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %170 = vector.extract %169[0, 0] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %arg12 [%147, %160] : f32 into vector<2x32xf32>
        %172 = vector.extract %169[0, 1] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%147, %162] : f32 into vector<2x32xf32>
        %174 = vector.extract %169[0, 2] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %173 [%147, %164] : f32 into vector<2x32xf32>
        %176 = vector.extract %169[0, 3] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%147, %166] : f32 into vector<2x32xf32>
        %178 = vector.insert %170, %arg13 [%147, %160] : f32 into vector<2x32xf32>
        %179 = vector.insert %172, %178 [%147, %162] : f32 into vector<2x32xf32>
        %180 = vector.insert %174, %179 [%147, %164] : f32 into vector<2x32xf32>
        %181 = vector.insert %176, %180 [%147, %166] : f32 into vector<2x32xf32>
        affine.yield %177, %181 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %110 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %129 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %131 = vector.extract %109#1[%129, %130] : f32 from vector<2x32xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %133 = vector.extract %109#1[%129, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %135 = vector.extract %109#1[%129, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %137 = vector.extract %109#1[%129, %136] : f32 from vector<2x32xf32>
        %138 = vector.insert %131, %arg12 [%129, %130] : f32 into vector<2x32xf32>
        %139 = vector.insert %133, %138 [%129, %132] : f32 into vector<2x32xf32>
        %140 = vector.insert %135, %139 [%129, %134] : f32 into vector<2x32xf32>
        %141 = vector.insert %137, %140 [%129, %136] : f32 into vector<2x32xf32>
        affine.yield %141 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %111 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %129 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_0) -> (f32) {
          %133 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %134 = vector.extract %110[%arg11, %133] : f32 from vector<2x32xf32>
          %135 = arith.maxnumf %arg14, %134 : f32
          %136 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %137 = vector.extract %110[%arg11, %136] : f32 from vector<2x32xf32>
          %138 = arith.maxnumf %135, %137 : f32
          %139 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %140 = vector.extract %110[%arg11, %139] : f32 from vector<2x32xf32>
          %141 = arith.maxnumf %138, %140 : f32
          %142 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %143 = vector.extract %110[%arg11, %142] : f32 from vector<2x32xf32>
          %144 = arith.maxnumf %141, %143 : f32
          affine.yield %144 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %129, %c16_i32, %c64_i32 : f32
        %130 = arith.maxnumf %129, %shuffleResult : f32
        %shuffleResult_21, %valid_22 = gpu.shuffle xor %130, %c32_i32, %c64_i32 : f32
        %131 = arith.maxnumf %130, %shuffleResult_21 : f32
        %132 = vector.insert %131, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %132 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %112 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %129 = vector.extract %111[%arg11, 0] : f32 from vector<2x1xf32>
        %130 = vector.broadcast %129 : f32 to vector<1x1xf32>
        %131 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %132 = vector.broadcast %131 : f32 to vector<1x1xf32>
        %133 = arith.maxnumf %130, %132 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %134 = vector.extract %133[0, 0] : f32 from vector<1x1xf32>
        %135 = vector.insert %134, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %135 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %113:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %129 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %130 = vector.extract %112[%129, 0] : f32 from vector<2x1xf32>
        %131 = vector.broadcast %130 : f32 to vector<1x4xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %133 = vector.insert %130, %arg12 [%129, %132] : f32 into vector<2x32xf32>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %135 = vector.insert %130, %133 [%129, %134] : f32 into vector<2x32xf32>
        %136 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %137 = vector.insert %130, %135 [%129, %136] : f32 into vector<2x32xf32>
        %138 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %139 = vector.insert %130, %137 [%129, %138] : f32 into vector<2x32xf32>
        %140 = vector.extract %109#1[%129, %132] : f32 from vector<2x32xf32>
        %141 = vector.extract %109#1[%129, %134] : f32 from vector<2x32xf32>
        %142 = vector.extract %109#1[%129, %136] : f32 from vector<2x32xf32>
        %143 = vector.extract %109#1[%129, %138] : f32 from vector<2x32xf32>
        %144 = vector.from_elements %140, %141, %142, %143 : vector<1x4xf32>
        %145 = arith.subf %144, %131 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %146 = vector.extract %145[0, 0] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %arg13 [%129, %132] : f32 into vector<2x32xf32>
        %148 = vector.extract %145[0, 1] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%129, %134] : f32 into vector<2x32xf32>
        %150 = vector.extract %145[0, 2] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%129, %136] : f32 into vector<2x32xf32>
        %152 = vector.extract %145[0, 3] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %151 [%129, %138] : f32 into vector<2x32xf32>
        affine.yield %139, %153 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %114:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %129 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %130 = vector.broadcast %129 : f32 to vector<1x1xf32>
        %131 = vector.extract %112[%arg11, 0] : f32 from vector<2x1xf32>
        %132 = vector.broadcast %131 : f32 to vector<1x1xf32>
        %133 = arith.subf %130, %132 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %134 = vector.extract %133[0, 0] : f32 from vector<1x1xf32>
        %135 = vector.insert %134, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %136 = math.exp2 %133 {tiled = true} : vector<1x1xf32>
        %137 = vector.extract %136[0, 0] : f32 from vector<1x1xf32>
        %138 = vector.insert %137, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %139 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %140 = vector.broadcast %139 : f32 to vector<1x1xf32>
        %141 = arith.mulf %140, %136 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %142 = vector.extract %141[0, 0] : f32 from vector<1x1xf32>
        %143 = vector.insert %142, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %135, %138, %143 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %115:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %129 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %131 = vector.extract %113#1[%129, %130] : f32 from vector<2x32xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %133 = vector.extract %113#1[%129, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %135 = vector.extract %113#1[%129, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %137 = vector.extract %113#1[%129, %136] : f32 from vector<2x32xf32>
        %138 = vector.from_elements %131, %133, %135, %137 : vector<1x4xf32>
        %139 = math.exp2 %138 {tiled = true} : vector<1x4xf32>
        %140 = vector.extract %139[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%129, %130] : f32 into vector<2x32xf32>
        %142 = vector.extract %139[0, 1] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %141 [%129, %132] : f32 into vector<2x32xf32>
        %144 = vector.extract %139[0, 2] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %143 [%129, %134] : f32 into vector<2x32xf32>
        %146 = vector.extract %139[0, 3] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %145 [%129, %136] : f32 into vector<2x32xf32>
        %148 = vector.insert %140, %arg13 [%129, %130] : f32 into vector<2x32xf32>
        %149 = vector.insert %142, %148 [%129, %132] : f32 into vector<2x32xf32>
        %150 = vector.insert %144, %149 [%129, %134] : f32 into vector<2x32xf32>
        %151 = vector.insert %146, %150 [%129, %136] : f32 into vector<2x32xf32>
        affine.yield %147, %151 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %116 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %129 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_8) -> (f32) {
          %133 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %134 = vector.extract %115#1[%arg11, %133] : f32 from vector<2x32xf32>
          %135 = arith.addf %arg14, %134 : f32
          %136 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %137 = vector.extract %115#1[%arg11, %136] : f32 from vector<2x32xf32>
          %138 = arith.addf %135, %137 : f32
          %139 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %140 = vector.extract %115#1[%arg11, %139] : f32 from vector<2x32xf32>
          %141 = arith.addf %138, %140 : f32
          %142 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %143 = vector.extract %115#1[%arg11, %142] : f32 from vector<2x32xf32>
          %144 = arith.addf %141, %143 : f32
          affine.yield %144 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %129, %c16_i32, %c64_i32 : f32
        %130 = arith.addf %129, %shuffleResult : f32
        %shuffleResult_21, %valid_22 = gpu.shuffle xor %130, %c32_i32, %c64_i32 : f32
        %131 = arith.addf %130, %shuffleResult_21 : f32
        %132 = vector.insert %131, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %132 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %117 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %129 = vector.extract %116[%arg11, 0] : f32 from vector<2x1xf32>
        %130 = vector.broadcast %129 : f32 to vector<1x1xf32>
        %131 = vector.extract %114#2[%arg11, 0] : f32 from vector<2x1xf32>
        %132 = vector.broadcast %131 : f32 to vector<1x1xf32>
        %133 = arith.addf %130, %132 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %134 = vector.extract %133[0, 0] : f32 from vector<1x1xf32>
        %135 = vector.insert %134, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %135 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %118 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf16>) {
        %129 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %131 = vector.extract %115#0[%129, %130] : f32 from vector<2x32xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %133 = vector.extract %115#0[%129, %132] : f32 from vector<2x32xf32>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %135 = vector.extract %115#0[%129, %134] : f32 from vector<2x32xf32>
        %136 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %137 = vector.extract %115#0[%129, %136] : f32 from vector<2x32xf32>
        %138 = vector.from_elements %131, %133, %135, %137 : vector<1x4xf32>
        %139 = arith.truncf %138 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %140 = vector.extract %139[0, 0] : f16 from vector<1x4xf16>
        %141 = vector.insert %140, %arg12 [%129, %130] : f16 into vector<2x32xf16>
        %142 = vector.extract %139[0, 1] : f16 from vector<1x4xf16>
        %143 = vector.insert %142, %141 [%129, %132] : f16 into vector<2x32xf16>
        %144 = vector.extract %139[0, 2] : f16 from vector<1x4xf16>
        %145 = vector.insert %144, %143 [%129, %134] : f16 into vector<2x32xf16>
        %146 = vector.extract %139[0, 3] : f16 from vector<1x4xf16>
        %147 = vector.insert %146, %145 [%129, %136] : f16 into vector<2x32xf16>
        affine.yield %147 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %119 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %130 = frisk.read_matrix_from_shm %view[%129] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %131 = vector.bitcast %130 : vector<4xi32> to vector<8xf16>
        %132 = vector.extract %131[0] : f16 from vector<8xf16>
        %133 = vector.extract %131[%82] : f16 from vector<8xf16>
        %134 = vector.extract %131[1] : f16 from vector<8xf16>
        %135 = vector.extract %131[%83] : f16 from vector<8xf16>
        %136 = vector.extract %131[2] : f16 from vector<8xf16>
        %137 = vector.extract %131[%84] : f16 from vector<8xf16>
        %138 = vector.extract %131[3] : f16 from vector<8xf16>
        %139 = vector.extract %131[%85] : f16 from vector<8xf16>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %141 = vector.insert %132, %arg12 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %133, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %144 = vector.insert %134, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %135, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %147 = vector.insert %136, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %137, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %150 = vector.insert %138, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        affine.yield %151 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_18 = memref.subview %arg2[0, %80, %102, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      frisk.async_copy_gm_to_shm %subview_18 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %120:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7, %arg15 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %129 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %130 = vector.extract %118[%129, 0] : f16 from vector<2x32xf16>
        %131 = vector.extract %118[%129, 1] : f16 from vector<2x32xf16>
        %132 = vector.extract %118[%129, 2] : f16 from vector<2x32xf16>
        %133 = vector.extract %118[%129, 3] : f16 from vector<2x32xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf16>
        %135 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %136 = vector.extract %119[0, %135] : f16 from vector<32x2xf16>
        %137 = vector.extract %119[1, %135] : f16 from vector<32x2xf16>
        %138 = vector.extract %119[2, %135] : f16 from vector<32x2xf16>
        %139 = vector.extract %119[3, %135] : f16 from vector<32x2xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<4x1xf16>
        %141 = frisk.warp_mma_rr %134, %140, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %142 = vector.extract %118[%129, 4] : f16 from vector<2x32xf16>
        %143 = vector.extract %118[%129, 5] : f16 from vector<2x32xf16>
        %144 = vector.extract %118[%129, 6] : f16 from vector<2x32xf16>
        %145 = vector.extract %118[%129, 7] : f16 from vector<2x32xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = vector.extract %119[4, %135] : f16 from vector<32x2xf16>
        %148 = vector.extract %119[5, %135] : f16 from vector<32x2xf16>
        %149 = vector.extract %119[6, %135] : f16 from vector<32x2xf16>
        %150 = vector.extract %119[7, %135] : f16 from vector<32x2xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<4x1xf16>
        %152 = frisk.warp_mma_rr %146, %151, %141 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %153 = vector.extract %118[%129, 8] : f16 from vector<2x32xf16>
        %154 = vector.extract %118[%129, 9] : f16 from vector<2x32xf16>
        %155 = vector.extract %118[%129, 10] : f16 from vector<2x32xf16>
        %156 = vector.extract %118[%129, 11] : f16 from vector<2x32xf16>
        %157 = vector.from_elements %153, %154, %155, %156 : vector<1x4xf16>
        %158 = vector.extract %119[8, %135] : f16 from vector<32x2xf16>
        %159 = vector.extract %119[9, %135] : f16 from vector<32x2xf16>
        %160 = vector.extract %119[10, %135] : f16 from vector<32x2xf16>
        %161 = vector.extract %119[11, %135] : f16 from vector<32x2xf16>
        %162 = vector.from_elements %158, %159, %160, %161 : vector<4x1xf16>
        %163 = frisk.warp_mma_rr %157, %162, %152 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %164 = vector.extract %118[%129, 12] : f16 from vector<2x32xf16>
        %165 = vector.extract %118[%129, 13] : f16 from vector<2x32xf16>
        %166 = vector.extract %118[%129, 14] : f16 from vector<2x32xf16>
        %167 = vector.extract %118[%129, 15] : f16 from vector<2x32xf16>
        %168 = vector.from_elements %164, %165, %166, %167 : vector<1x4xf16>
        %169 = vector.extract %119[12, %135] : f16 from vector<32x2xf16>
        %170 = vector.extract %119[13, %135] : f16 from vector<32x2xf16>
        %171 = vector.extract %119[14, %135] : f16 from vector<32x2xf16>
        %172 = vector.extract %119[15, %135] : f16 from vector<32x2xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<4x1xf16>
        %174 = frisk.warp_mma_rr %168, %173, %163 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %175 = vector.extract %118[%129, 16] : f16 from vector<2x32xf16>
        %176 = vector.extract %118[%129, 17] : f16 from vector<2x32xf16>
        %177 = vector.extract %118[%129, 18] : f16 from vector<2x32xf16>
        %178 = vector.extract %118[%129, 19] : f16 from vector<2x32xf16>
        %179 = vector.from_elements %175, %176, %177, %178 : vector<1x4xf16>
        %180 = vector.extract %119[16, %135] : f16 from vector<32x2xf16>
        %181 = vector.extract %119[17, %135] : f16 from vector<32x2xf16>
        %182 = vector.extract %119[18, %135] : f16 from vector<32x2xf16>
        %183 = vector.extract %119[19, %135] : f16 from vector<32x2xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<4x1xf16>
        %185 = frisk.warp_mma_rr %179, %184, %174 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %186 = vector.extract %118[%129, 20] : f16 from vector<2x32xf16>
        %187 = vector.extract %118[%129, 21] : f16 from vector<2x32xf16>
        %188 = vector.extract %118[%129, 22] : f16 from vector<2x32xf16>
        %189 = vector.extract %118[%129, 23] : f16 from vector<2x32xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<1x4xf16>
        %191 = vector.extract %119[20, %135] : f16 from vector<32x2xf16>
        %192 = vector.extract %119[21, %135] : f16 from vector<32x2xf16>
        %193 = vector.extract %119[22, %135] : f16 from vector<32x2xf16>
        %194 = vector.extract %119[23, %135] : f16 from vector<32x2xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<4x1xf16>
        %196 = frisk.warp_mma_rr %190, %195, %185 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %197 = vector.extract %118[%129, 24] : f16 from vector<2x32xf16>
        %198 = vector.extract %118[%129, 25] : f16 from vector<2x32xf16>
        %199 = vector.extract %118[%129, 26] : f16 from vector<2x32xf16>
        %200 = vector.extract %118[%129, 27] : f16 from vector<2x32xf16>
        %201 = vector.from_elements %197, %198, %199, %200 : vector<1x4xf16>
        %202 = vector.extract %119[24, %135] : f16 from vector<32x2xf16>
        %203 = vector.extract %119[25, %135] : f16 from vector<32x2xf16>
        %204 = vector.extract %119[26, %135] : f16 from vector<32x2xf16>
        %205 = vector.extract %119[27, %135] : f16 from vector<32x2xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<4x1xf16>
        %207 = frisk.warp_mma_rr %201, %206, %196 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %208 = vector.extract %118[%129, 28] : f16 from vector<2x32xf16>
        %209 = vector.extract %118[%129, 29] : f16 from vector<2x32xf16>
        %210 = vector.extract %118[%129, 30] : f16 from vector<2x32xf16>
        %211 = vector.extract %118[%129, 31] : f16 from vector<2x32xf16>
        %212 = vector.from_elements %208, %209, %210, %211 : vector<1x4xf16>
        %213 = vector.extract %119[28, %135] : f16 from vector<32x2xf16>
        %214 = vector.extract %119[29, %135] : f16 from vector<32x2xf16>
        %215 = vector.extract %119[30, %135] : f16 from vector<32x2xf16>
        %216 = vector.extract %119[31, %135] : f16 from vector<32x2xf16>
        %217 = vector.from_elements %213, %214, %215, %216 : vector<4x1xf16>
        %218 = frisk.warp_mma_rr %212, %217, %207 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %220 = vector.extract %218[0, 0] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %arg12 [%129, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %223 = vector.extract %218[0, 1] : f32 from vector<1x4xf32>
        %224 = vector.insert %223, %221 [%129, %222] : f32 into vector<2x8xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %226 = vector.extract %218[0, 2] : f32 from vector<1x4xf32>
        %227 = vector.insert %226, %224 [%129, %225] : f32 into vector<2x8xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %229 = vector.extract %218[0, 3] : f32 from vector<1x4xf32>
        %230 = vector.insert %229, %227 [%129, %228] : f32 into vector<2x8xf32>
        %231 = vector.extract %114#1[%129, 0] : f32 from vector<2x1xf32>
        %232 = vector.broadcast %231 : f32 to vector<1x4xf32>
        %233 = vector.insert %231, %arg13 [%129, %219] : f32 into vector<2x8xf32>
        %234 = vector.insert %231, %233 [%129, %222] : f32 into vector<2x8xf32>
        %235 = vector.insert %231, %234 [%129, %225] : f32 into vector<2x8xf32>
        %236 = vector.insert %231, %235 [%129, %228] : f32 into vector<2x8xf32>
        %237 = vector.extract %arg5[%129, %219] : f32 from vector<2x8xf32>
        %238 = vector.extract %arg5[%129, %222] : f32 from vector<2x8xf32>
        %239 = vector.extract %arg5[%129, %225] : f32 from vector<2x8xf32>
        %240 = vector.extract %arg5[%129, %228] : f32 from vector<2x8xf32>
        %241 = vector.from_elements %237, %238, %239, %240 : vector<1x4xf32>
        %242 = arith.mulf %241, %232 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %243 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %arg14 [%129, %219] : f32 into vector<2x8xf32>
        %245 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %246 = vector.insert %245, %244 [%129, %222] : f32 into vector<2x8xf32>
        %247 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %246 [%129, %225] : f32 into vector<2x8xf32>
        %249 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %250 = vector.insert %249, %248 [%129, %228] : f32 into vector<2x8xf32>
        %251 = arith.addf %242, %218 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %252 = vector.extract %251[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg15 [%129, %219] : f32 into vector<2x8xf32>
        %254 = vector.extract %251[0, 1] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %253 [%129, %222] : f32 into vector<2x8xf32>
        %256 = vector.extract %251[0, 2] : f32 from vector<1x4xf32>
        %257 = vector.insert %256, %255 [%129, %225] : f32 into vector<2x8xf32>
        %258 = vector.extract %251[0, 3] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %257 [%129, %228] : f32 into vector<2x8xf32>
        affine.yield %230, %236, %250, %259 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %121 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %130 = frisk.read_matrix_from_shm %view[%129] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %131 = vector.bitcast %130 : vector<4xi32> to vector<8xf16>
        %132 = vector.extract %131[0] : f16 from vector<8xf16>
        %133 = vector.extract %131[%82] : f16 from vector<8xf16>
        %134 = vector.extract %131[1] : f16 from vector<8xf16>
        %135 = vector.extract %131[%83] : f16 from vector<8xf16>
        %136 = vector.extract %131[2] : f16 from vector<8xf16>
        %137 = vector.extract %131[%84] : f16 from vector<8xf16>
        %138 = vector.extract %131[3] : f16 from vector<8xf16>
        %139 = vector.extract %131[%85] : f16 from vector<8xf16>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %141 = vector.insert %132, %arg12 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %133, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %144 = vector.insert %134, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %135, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %147 = vector.insert %136, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %137, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %150 = vector.insert %138, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        affine.yield %151 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_19 = memref.subview %arg2[0, %80, %102, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      frisk.async_copy_gm_to_shm %subview_19 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %122:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7, %arg15 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %129 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %130 = vector.extract %118[%129, 0] : f16 from vector<2x32xf16>
        %131 = vector.extract %118[%129, 1] : f16 from vector<2x32xf16>
        %132 = vector.extract %118[%129, 2] : f16 from vector<2x32xf16>
        %133 = vector.extract %118[%129, 3] : f16 from vector<2x32xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf16>
        %135 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %136 = vector.extract %121[0, %135] : f16 from vector<32x2xf16>
        %137 = vector.extract %121[1, %135] : f16 from vector<32x2xf16>
        %138 = vector.extract %121[2, %135] : f16 from vector<32x2xf16>
        %139 = vector.extract %121[3, %135] : f16 from vector<32x2xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<4x1xf16>
        %141 = frisk.warp_mma_rr %134, %140, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %142 = vector.extract %118[%129, 4] : f16 from vector<2x32xf16>
        %143 = vector.extract %118[%129, 5] : f16 from vector<2x32xf16>
        %144 = vector.extract %118[%129, 6] : f16 from vector<2x32xf16>
        %145 = vector.extract %118[%129, 7] : f16 from vector<2x32xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = vector.extract %121[4, %135] : f16 from vector<32x2xf16>
        %148 = vector.extract %121[5, %135] : f16 from vector<32x2xf16>
        %149 = vector.extract %121[6, %135] : f16 from vector<32x2xf16>
        %150 = vector.extract %121[7, %135] : f16 from vector<32x2xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<4x1xf16>
        %152 = frisk.warp_mma_rr %146, %151, %141 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %153 = vector.extract %118[%129, 8] : f16 from vector<2x32xf16>
        %154 = vector.extract %118[%129, 9] : f16 from vector<2x32xf16>
        %155 = vector.extract %118[%129, 10] : f16 from vector<2x32xf16>
        %156 = vector.extract %118[%129, 11] : f16 from vector<2x32xf16>
        %157 = vector.from_elements %153, %154, %155, %156 : vector<1x4xf16>
        %158 = vector.extract %121[8, %135] : f16 from vector<32x2xf16>
        %159 = vector.extract %121[9, %135] : f16 from vector<32x2xf16>
        %160 = vector.extract %121[10, %135] : f16 from vector<32x2xf16>
        %161 = vector.extract %121[11, %135] : f16 from vector<32x2xf16>
        %162 = vector.from_elements %158, %159, %160, %161 : vector<4x1xf16>
        %163 = frisk.warp_mma_rr %157, %162, %152 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %164 = vector.extract %118[%129, 12] : f16 from vector<2x32xf16>
        %165 = vector.extract %118[%129, 13] : f16 from vector<2x32xf16>
        %166 = vector.extract %118[%129, 14] : f16 from vector<2x32xf16>
        %167 = vector.extract %118[%129, 15] : f16 from vector<2x32xf16>
        %168 = vector.from_elements %164, %165, %166, %167 : vector<1x4xf16>
        %169 = vector.extract %121[12, %135] : f16 from vector<32x2xf16>
        %170 = vector.extract %121[13, %135] : f16 from vector<32x2xf16>
        %171 = vector.extract %121[14, %135] : f16 from vector<32x2xf16>
        %172 = vector.extract %121[15, %135] : f16 from vector<32x2xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<4x1xf16>
        %174 = frisk.warp_mma_rr %168, %173, %163 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %175 = vector.extract %118[%129, 16] : f16 from vector<2x32xf16>
        %176 = vector.extract %118[%129, 17] : f16 from vector<2x32xf16>
        %177 = vector.extract %118[%129, 18] : f16 from vector<2x32xf16>
        %178 = vector.extract %118[%129, 19] : f16 from vector<2x32xf16>
        %179 = vector.from_elements %175, %176, %177, %178 : vector<1x4xf16>
        %180 = vector.extract %121[16, %135] : f16 from vector<32x2xf16>
        %181 = vector.extract %121[17, %135] : f16 from vector<32x2xf16>
        %182 = vector.extract %121[18, %135] : f16 from vector<32x2xf16>
        %183 = vector.extract %121[19, %135] : f16 from vector<32x2xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<4x1xf16>
        %185 = frisk.warp_mma_rr %179, %184, %174 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %186 = vector.extract %118[%129, 20] : f16 from vector<2x32xf16>
        %187 = vector.extract %118[%129, 21] : f16 from vector<2x32xf16>
        %188 = vector.extract %118[%129, 22] : f16 from vector<2x32xf16>
        %189 = vector.extract %118[%129, 23] : f16 from vector<2x32xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<1x4xf16>
        %191 = vector.extract %121[20, %135] : f16 from vector<32x2xf16>
        %192 = vector.extract %121[21, %135] : f16 from vector<32x2xf16>
        %193 = vector.extract %121[22, %135] : f16 from vector<32x2xf16>
        %194 = vector.extract %121[23, %135] : f16 from vector<32x2xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<4x1xf16>
        %196 = frisk.warp_mma_rr %190, %195, %185 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %197 = vector.extract %118[%129, 24] : f16 from vector<2x32xf16>
        %198 = vector.extract %118[%129, 25] : f16 from vector<2x32xf16>
        %199 = vector.extract %118[%129, 26] : f16 from vector<2x32xf16>
        %200 = vector.extract %118[%129, 27] : f16 from vector<2x32xf16>
        %201 = vector.from_elements %197, %198, %199, %200 : vector<1x4xf16>
        %202 = vector.extract %121[24, %135] : f16 from vector<32x2xf16>
        %203 = vector.extract %121[25, %135] : f16 from vector<32x2xf16>
        %204 = vector.extract %121[26, %135] : f16 from vector<32x2xf16>
        %205 = vector.extract %121[27, %135] : f16 from vector<32x2xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<4x1xf16>
        %207 = frisk.warp_mma_rr %201, %206, %196 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %208 = vector.extract %118[%129, 28] : f16 from vector<2x32xf16>
        %209 = vector.extract %118[%129, 29] : f16 from vector<2x32xf16>
        %210 = vector.extract %118[%129, 30] : f16 from vector<2x32xf16>
        %211 = vector.extract %118[%129, 31] : f16 from vector<2x32xf16>
        %212 = vector.from_elements %208, %209, %210, %211 : vector<1x4xf16>
        %213 = vector.extract %121[28, %135] : f16 from vector<32x2xf16>
        %214 = vector.extract %121[29, %135] : f16 from vector<32x2xf16>
        %215 = vector.extract %121[30, %135] : f16 from vector<32x2xf16>
        %216 = vector.extract %121[31, %135] : f16 from vector<32x2xf16>
        %217 = vector.from_elements %213, %214, %215, %216 : vector<4x1xf16>
        %218 = frisk.warp_mma_rr %212, %217, %207 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %220 = vector.extract %218[0, 0] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %arg12 [%129, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %223 = vector.extract %218[0, 1] : f32 from vector<1x4xf32>
        %224 = vector.insert %223, %221 [%129, %222] : f32 into vector<2x8xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %226 = vector.extract %218[0, 2] : f32 from vector<1x4xf32>
        %227 = vector.insert %226, %224 [%129, %225] : f32 into vector<2x8xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %229 = vector.extract %218[0, 3] : f32 from vector<1x4xf32>
        %230 = vector.insert %229, %227 [%129, %228] : f32 into vector<2x8xf32>
        %231 = vector.extract %114#1[%129, 0] : f32 from vector<2x1xf32>
        %232 = vector.broadcast %231 : f32 to vector<1x4xf32>
        %233 = vector.insert %231, %arg13 [%129, %219] : f32 into vector<2x8xf32>
        %234 = vector.insert %231, %233 [%129, %222] : f32 into vector<2x8xf32>
        %235 = vector.insert %231, %234 [%129, %225] : f32 into vector<2x8xf32>
        %236 = vector.insert %231, %235 [%129, %228] : f32 into vector<2x8xf32>
        %237 = vector.extract %arg8[%129, %219] : f32 from vector<2x8xf32>
        %238 = vector.extract %arg8[%129, %222] : f32 from vector<2x8xf32>
        %239 = vector.extract %arg8[%129, %225] : f32 from vector<2x8xf32>
        %240 = vector.extract %arg8[%129, %228] : f32 from vector<2x8xf32>
        %241 = vector.from_elements %237, %238, %239, %240 : vector<1x4xf32>
        %242 = arith.mulf %241, %232 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %243 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %arg14 [%129, %219] : f32 into vector<2x8xf32>
        %245 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %246 = vector.insert %245, %244 [%129, %222] : f32 into vector<2x8xf32>
        %247 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %246 [%129, %225] : f32 into vector<2x8xf32>
        %249 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %250 = vector.insert %249, %248 [%129, %228] : f32 into vector<2x8xf32>
        %251 = arith.addf %242, %218 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %252 = vector.extract %251[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg15 [%129, %219] : f32 into vector<2x8xf32>
        %254 = vector.extract %251[0, 1] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %253 [%129, %222] : f32 into vector<2x8xf32>
        %256 = vector.extract %251[0, 2] : f32 from vector<1x4xf32>
        %257 = vector.insert %256, %255 [%129, %225] : f32 into vector<2x8xf32>
        %258 = vector.extract %251[0, 3] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %257 [%129, %228] : f32 into vector<2x8xf32>
        affine.yield %230, %236, %250, %259 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %123 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %130 = frisk.read_matrix_from_shm %view[%129] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %131 = vector.bitcast %130 : vector<4xi32> to vector<8xf16>
        %132 = vector.extract %131[0] : f16 from vector<8xf16>
        %133 = vector.extract %131[%82] : f16 from vector<8xf16>
        %134 = vector.extract %131[1] : f16 from vector<8xf16>
        %135 = vector.extract %131[%83] : f16 from vector<8xf16>
        %136 = vector.extract %131[2] : f16 from vector<8xf16>
        %137 = vector.extract %131[%84] : f16 from vector<8xf16>
        %138 = vector.extract %131[3] : f16 from vector<8xf16>
        %139 = vector.extract %131[%85] : f16 from vector<8xf16>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %141 = vector.insert %132, %arg12 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %133, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %144 = vector.insert %134, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %135, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %147 = vector.insert %136, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %137, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %150 = vector.insert %138, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        affine.yield %151 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_20 = memref.subview %arg2[0, %80, %102, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      frisk.async_copy_gm_to_shm %subview_20 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %124:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7, %arg15 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %129 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %130 = vector.extract %118[%129, 0] : f16 from vector<2x32xf16>
        %131 = vector.extract %118[%129, 1] : f16 from vector<2x32xf16>
        %132 = vector.extract %118[%129, 2] : f16 from vector<2x32xf16>
        %133 = vector.extract %118[%129, 3] : f16 from vector<2x32xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf16>
        %135 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %136 = vector.extract %123[0, %135] : f16 from vector<32x2xf16>
        %137 = vector.extract %123[1, %135] : f16 from vector<32x2xf16>
        %138 = vector.extract %123[2, %135] : f16 from vector<32x2xf16>
        %139 = vector.extract %123[3, %135] : f16 from vector<32x2xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<4x1xf16>
        %141 = frisk.warp_mma_rr %134, %140, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %142 = vector.extract %118[%129, 4] : f16 from vector<2x32xf16>
        %143 = vector.extract %118[%129, 5] : f16 from vector<2x32xf16>
        %144 = vector.extract %118[%129, 6] : f16 from vector<2x32xf16>
        %145 = vector.extract %118[%129, 7] : f16 from vector<2x32xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = vector.extract %123[4, %135] : f16 from vector<32x2xf16>
        %148 = vector.extract %123[5, %135] : f16 from vector<32x2xf16>
        %149 = vector.extract %123[6, %135] : f16 from vector<32x2xf16>
        %150 = vector.extract %123[7, %135] : f16 from vector<32x2xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<4x1xf16>
        %152 = frisk.warp_mma_rr %146, %151, %141 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %153 = vector.extract %118[%129, 8] : f16 from vector<2x32xf16>
        %154 = vector.extract %118[%129, 9] : f16 from vector<2x32xf16>
        %155 = vector.extract %118[%129, 10] : f16 from vector<2x32xf16>
        %156 = vector.extract %118[%129, 11] : f16 from vector<2x32xf16>
        %157 = vector.from_elements %153, %154, %155, %156 : vector<1x4xf16>
        %158 = vector.extract %123[8, %135] : f16 from vector<32x2xf16>
        %159 = vector.extract %123[9, %135] : f16 from vector<32x2xf16>
        %160 = vector.extract %123[10, %135] : f16 from vector<32x2xf16>
        %161 = vector.extract %123[11, %135] : f16 from vector<32x2xf16>
        %162 = vector.from_elements %158, %159, %160, %161 : vector<4x1xf16>
        %163 = frisk.warp_mma_rr %157, %162, %152 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %164 = vector.extract %118[%129, 12] : f16 from vector<2x32xf16>
        %165 = vector.extract %118[%129, 13] : f16 from vector<2x32xf16>
        %166 = vector.extract %118[%129, 14] : f16 from vector<2x32xf16>
        %167 = vector.extract %118[%129, 15] : f16 from vector<2x32xf16>
        %168 = vector.from_elements %164, %165, %166, %167 : vector<1x4xf16>
        %169 = vector.extract %123[12, %135] : f16 from vector<32x2xf16>
        %170 = vector.extract %123[13, %135] : f16 from vector<32x2xf16>
        %171 = vector.extract %123[14, %135] : f16 from vector<32x2xf16>
        %172 = vector.extract %123[15, %135] : f16 from vector<32x2xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<4x1xf16>
        %174 = frisk.warp_mma_rr %168, %173, %163 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %175 = vector.extract %118[%129, 16] : f16 from vector<2x32xf16>
        %176 = vector.extract %118[%129, 17] : f16 from vector<2x32xf16>
        %177 = vector.extract %118[%129, 18] : f16 from vector<2x32xf16>
        %178 = vector.extract %118[%129, 19] : f16 from vector<2x32xf16>
        %179 = vector.from_elements %175, %176, %177, %178 : vector<1x4xf16>
        %180 = vector.extract %123[16, %135] : f16 from vector<32x2xf16>
        %181 = vector.extract %123[17, %135] : f16 from vector<32x2xf16>
        %182 = vector.extract %123[18, %135] : f16 from vector<32x2xf16>
        %183 = vector.extract %123[19, %135] : f16 from vector<32x2xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<4x1xf16>
        %185 = frisk.warp_mma_rr %179, %184, %174 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %186 = vector.extract %118[%129, 20] : f16 from vector<2x32xf16>
        %187 = vector.extract %118[%129, 21] : f16 from vector<2x32xf16>
        %188 = vector.extract %118[%129, 22] : f16 from vector<2x32xf16>
        %189 = vector.extract %118[%129, 23] : f16 from vector<2x32xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<1x4xf16>
        %191 = vector.extract %123[20, %135] : f16 from vector<32x2xf16>
        %192 = vector.extract %123[21, %135] : f16 from vector<32x2xf16>
        %193 = vector.extract %123[22, %135] : f16 from vector<32x2xf16>
        %194 = vector.extract %123[23, %135] : f16 from vector<32x2xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<4x1xf16>
        %196 = frisk.warp_mma_rr %190, %195, %185 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %197 = vector.extract %118[%129, 24] : f16 from vector<2x32xf16>
        %198 = vector.extract %118[%129, 25] : f16 from vector<2x32xf16>
        %199 = vector.extract %118[%129, 26] : f16 from vector<2x32xf16>
        %200 = vector.extract %118[%129, 27] : f16 from vector<2x32xf16>
        %201 = vector.from_elements %197, %198, %199, %200 : vector<1x4xf16>
        %202 = vector.extract %123[24, %135] : f16 from vector<32x2xf16>
        %203 = vector.extract %123[25, %135] : f16 from vector<32x2xf16>
        %204 = vector.extract %123[26, %135] : f16 from vector<32x2xf16>
        %205 = vector.extract %123[27, %135] : f16 from vector<32x2xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<4x1xf16>
        %207 = frisk.warp_mma_rr %201, %206, %196 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %208 = vector.extract %118[%129, 28] : f16 from vector<2x32xf16>
        %209 = vector.extract %118[%129, 29] : f16 from vector<2x32xf16>
        %210 = vector.extract %118[%129, 30] : f16 from vector<2x32xf16>
        %211 = vector.extract %118[%129, 31] : f16 from vector<2x32xf16>
        %212 = vector.from_elements %208, %209, %210, %211 : vector<1x4xf16>
        %213 = vector.extract %123[28, %135] : f16 from vector<32x2xf16>
        %214 = vector.extract %123[29, %135] : f16 from vector<32x2xf16>
        %215 = vector.extract %123[30, %135] : f16 from vector<32x2xf16>
        %216 = vector.extract %123[31, %135] : f16 from vector<32x2xf16>
        %217 = vector.from_elements %213, %214, %215, %216 : vector<4x1xf16>
        %218 = frisk.warp_mma_rr %212, %217, %207 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %220 = vector.extract %218[0, 0] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %arg12 [%129, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %223 = vector.extract %218[0, 1] : f32 from vector<1x4xf32>
        %224 = vector.insert %223, %221 [%129, %222] : f32 into vector<2x8xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %226 = vector.extract %218[0, 2] : f32 from vector<1x4xf32>
        %227 = vector.insert %226, %224 [%129, %225] : f32 into vector<2x8xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %229 = vector.extract %218[0, 3] : f32 from vector<1x4xf32>
        %230 = vector.insert %229, %227 [%129, %228] : f32 into vector<2x8xf32>
        %231 = vector.extract %114#1[%129, 0] : f32 from vector<2x1xf32>
        %232 = vector.broadcast %231 : f32 to vector<1x4xf32>
        %233 = vector.insert %231, %arg13 [%129, %219] : f32 into vector<2x8xf32>
        %234 = vector.insert %231, %233 [%129, %222] : f32 into vector<2x8xf32>
        %235 = vector.insert %231, %234 [%129, %225] : f32 into vector<2x8xf32>
        %236 = vector.insert %231, %235 [%129, %228] : f32 into vector<2x8xf32>
        %237 = vector.extract %arg9[%129, %219] : f32 from vector<2x8xf32>
        %238 = vector.extract %arg9[%129, %222] : f32 from vector<2x8xf32>
        %239 = vector.extract %arg9[%129, %225] : f32 from vector<2x8xf32>
        %240 = vector.extract %arg9[%129, %228] : f32 from vector<2x8xf32>
        %241 = vector.from_elements %237, %238, %239, %240 : vector<1x4xf32>
        %242 = arith.mulf %241, %232 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %243 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %arg14 [%129, %219] : f32 into vector<2x8xf32>
        %245 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %246 = vector.insert %245, %244 [%129, %222] : f32 into vector<2x8xf32>
        %247 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %246 [%129, %225] : f32 into vector<2x8xf32>
        %249 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %250 = vector.insert %249, %248 [%129, %228] : f32 into vector<2x8xf32>
        %251 = arith.addf %242, %218 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %252 = vector.extract %251[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg15 [%129, %219] : f32 into vector<2x8xf32>
        %254 = vector.extract %251[0, 1] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %253 [%129, %222] : f32 into vector<2x8xf32>
        %256 = vector.extract %251[0, 2] : f32 from vector<1x4xf32>
        %257 = vector.insert %256, %255 [%129, %225] : f32 into vector<2x8xf32>
        %258 = vector.extract %251[0, 3] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %257 [%129, %228] : f32 into vector<2x8xf32>
        affine.yield %230, %236, %250, %259 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %125 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %129 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %130 = frisk.read_matrix_from_shm %view[%129] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %131 = vector.bitcast %130 : vector<4xi32> to vector<8xf16>
        %132 = vector.extract %131[0] : f16 from vector<8xf16>
        %133 = vector.extract %131[%82] : f16 from vector<8xf16>
        %134 = vector.extract %131[1] : f16 from vector<8xf16>
        %135 = vector.extract %131[%83] : f16 from vector<8xf16>
        %136 = vector.extract %131[2] : f16 from vector<8xf16>
        %137 = vector.extract %131[%84] : f16 from vector<8xf16>
        %138 = vector.extract %131[3] : f16 from vector<8xf16>
        %139 = vector.extract %131[%85] : f16 from vector<8xf16>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %141 = vector.insert %132, %arg12 [%140, 0] : f16 into vector<32x2xf16>
        %142 = vector.insert %133, %141 [%140, 1] : f16 into vector<32x2xf16>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %144 = vector.insert %134, %142 [%143, 0] : f16 into vector<32x2xf16>
        %145 = vector.insert %135, %144 [%143, 1] : f16 into vector<32x2xf16>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %147 = vector.insert %136, %145 [%146, 0] : f16 into vector<32x2xf16>
        %148 = vector.insert %137, %147 [%146, 1] : f16 into vector<32x2xf16>
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %150 = vector.insert %138, %148 [%149, 0] : f16 into vector<32x2xf16>
        %151 = vector.insert %139, %150 [%149, 1] : f16 into vector<32x2xf16>
        affine.yield %151 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %126:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7, %arg15 = %cst_7, %arg16 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %129 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %130 = vector.extract %118[%129, 0] : f16 from vector<2x32xf16>
        %131 = vector.extract %118[%129, 1] : f16 from vector<2x32xf16>
        %132 = vector.extract %118[%129, 2] : f16 from vector<2x32xf16>
        %133 = vector.extract %118[%129, 3] : f16 from vector<2x32xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf16>
        %135 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %136 = vector.extract %125[0, %135] : f16 from vector<32x2xf16>
        %137 = vector.extract %125[1, %135] : f16 from vector<32x2xf16>
        %138 = vector.extract %125[2, %135] : f16 from vector<32x2xf16>
        %139 = vector.extract %125[3, %135] : f16 from vector<32x2xf16>
        %140 = vector.from_elements %136, %137, %138, %139 : vector<4x1xf16>
        %141 = frisk.warp_mma_rr %134, %140, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %142 = vector.extract %118[%129, 4] : f16 from vector<2x32xf16>
        %143 = vector.extract %118[%129, 5] : f16 from vector<2x32xf16>
        %144 = vector.extract %118[%129, 6] : f16 from vector<2x32xf16>
        %145 = vector.extract %118[%129, 7] : f16 from vector<2x32xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = vector.extract %125[4, %135] : f16 from vector<32x2xf16>
        %148 = vector.extract %125[5, %135] : f16 from vector<32x2xf16>
        %149 = vector.extract %125[6, %135] : f16 from vector<32x2xf16>
        %150 = vector.extract %125[7, %135] : f16 from vector<32x2xf16>
        %151 = vector.from_elements %147, %148, %149, %150 : vector<4x1xf16>
        %152 = frisk.warp_mma_rr %146, %151, %141 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %153 = vector.extract %118[%129, 8] : f16 from vector<2x32xf16>
        %154 = vector.extract %118[%129, 9] : f16 from vector<2x32xf16>
        %155 = vector.extract %118[%129, 10] : f16 from vector<2x32xf16>
        %156 = vector.extract %118[%129, 11] : f16 from vector<2x32xf16>
        %157 = vector.from_elements %153, %154, %155, %156 : vector<1x4xf16>
        %158 = vector.extract %125[8, %135] : f16 from vector<32x2xf16>
        %159 = vector.extract %125[9, %135] : f16 from vector<32x2xf16>
        %160 = vector.extract %125[10, %135] : f16 from vector<32x2xf16>
        %161 = vector.extract %125[11, %135] : f16 from vector<32x2xf16>
        %162 = vector.from_elements %158, %159, %160, %161 : vector<4x1xf16>
        %163 = frisk.warp_mma_rr %157, %162, %152 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %164 = vector.extract %118[%129, 12] : f16 from vector<2x32xf16>
        %165 = vector.extract %118[%129, 13] : f16 from vector<2x32xf16>
        %166 = vector.extract %118[%129, 14] : f16 from vector<2x32xf16>
        %167 = vector.extract %118[%129, 15] : f16 from vector<2x32xf16>
        %168 = vector.from_elements %164, %165, %166, %167 : vector<1x4xf16>
        %169 = vector.extract %125[12, %135] : f16 from vector<32x2xf16>
        %170 = vector.extract %125[13, %135] : f16 from vector<32x2xf16>
        %171 = vector.extract %125[14, %135] : f16 from vector<32x2xf16>
        %172 = vector.extract %125[15, %135] : f16 from vector<32x2xf16>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<4x1xf16>
        %174 = frisk.warp_mma_rr %168, %173, %163 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %175 = vector.extract %118[%129, 16] : f16 from vector<2x32xf16>
        %176 = vector.extract %118[%129, 17] : f16 from vector<2x32xf16>
        %177 = vector.extract %118[%129, 18] : f16 from vector<2x32xf16>
        %178 = vector.extract %118[%129, 19] : f16 from vector<2x32xf16>
        %179 = vector.from_elements %175, %176, %177, %178 : vector<1x4xf16>
        %180 = vector.extract %125[16, %135] : f16 from vector<32x2xf16>
        %181 = vector.extract %125[17, %135] : f16 from vector<32x2xf16>
        %182 = vector.extract %125[18, %135] : f16 from vector<32x2xf16>
        %183 = vector.extract %125[19, %135] : f16 from vector<32x2xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<4x1xf16>
        %185 = frisk.warp_mma_rr %179, %184, %174 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %186 = vector.extract %118[%129, 20] : f16 from vector<2x32xf16>
        %187 = vector.extract %118[%129, 21] : f16 from vector<2x32xf16>
        %188 = vector.extract %118[%129, 22] : f16 from vector<2x32xf16>
        %189 = vector.extract %118[%129, 23] : f16 from vector<2x32xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<1x4xf16>
        %191 = vector.extract %125[20, %135] : f16 from vector<32x2xf16>
        %192 = vector.extract %125[21, %135] : f16 from vector<32x2xf16>
        %193 = vector.extract %125[22, %135] : f16 from vector<32x2xf16>
        %194 = vector.extract %125[23, %135] : f16 from vector<32x2xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<4x1xf16>
        %196 = frisk.warp_mma_rr %190, %195, %185 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %197 = vector.extract %118[%129, 24] : f16 from vector<2x32xf16>
        %198 = vector.extract %118[%129, 25] : f16 from vector<2x32xf16>
        %199 = vector.extract %118[%129, 26] : f16 from vector<2x32xf16>
        %200 = vector.extract %118[%129, 27] : f16 from vector<2x32xf16>
        %201 = vector.from_elements %197, %198, %199, %200 : vector<1x4xf16>
        %202 = vector.extract %125[24, %135] : f16 from vector<32x2xf16>
        %203 = vector.extract %125[25, %135] : f16 from vector<32x2xf16>
        %204 = vector.extract %125[26, %135] : f16 from vector<32x2xf16>
        %205 = vector.extract %125[27, %135] : f16 from vector<32x2xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<4x1xf16>
        %207 = frisk.warp_mma_rr %201, %206, %196 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %208 = vector.extract %118[%129, 28] : f16 from vector<2x32xf16>
        %209 = vector.extract %118[%129, 29] : f16 from vector<2x32xf16>
        %210 = vector.extract %118[%129, 30] : f16 from vector<2x32xf16>
        %211 = vector.extract %118[%129, 31] : f16 from vector<2x32xf16>
        %212 = vector.from_elements %208, %209, %210, %211 : vector<1x4xf16>
        %213 = vector.extract %125[28, %135] : f16 from vector<32x2xf16>
        %214 = vector.extract %125[29, %135] : f16 from vector<32x2xf16>
        %215 = vector.extract %125[30, %135] : f16 from vector<32x2xf16>
        %216 = vector.extract %125[31, %135] : f16 from vector<32x2xf16>
        %217 = vector.from_elements %213, %214, %215, %216 : vector<4x1xf16>
        %218 = frisk.warp_mma_rr %212, %217, %207 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %220 = vector.extract %218[0, 0] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %arg12 [%129, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %223 = vector.extract %218[0, 1] : f32 from vector<1x4xf32>
        %224 = vector.insert %223, %221 [%129, %222] : f32 into vector<2x8xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %226 = vector.extract %218[0, 2] : f32 from vector<1x4xf32>
        %227 = vector.insert %226, %224 [%129, %225] : f32 into vector<2x8xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %229 = vector.extract %218[0, 3] : f32 from vector<1x4xf32>
        %230 = vector.insert %229, %227 [%129, %228] : f32 into vector<2x8xf32>
        %231 = vector.extract %114#1[%129, 0] : f32 from vector<2x1xf32>
        %232 = vector.broadcast %231 : f32 to vector<1x4xf32>
        %233 = vector.insert %231, %arg13 [%129, %219] : f32 into vector<2x8xf32>
        %234 = vector.insert %231, %233 [%129, %222] : f32 into vector<2x8xf32>
        %235 = vector.insert %231, %234 [%129, %225] : f32 into vector<2x8xf32>
        %236 = vector.insert %231, %235 [%129, %228] : f32 into vector<2x8xf32>
        %237 = vector.extract %arg10[%129, %219] : f32 from vector<2x8xf32>
        %238 = vector.extract %arg10[%129, %222] : f32 from vector<2x8xf32>
        %239 = vector.extract %arg10[%129, %225] : f32 from vector<2x8xf32>
        %240 = vector.extract %arg10[%129, %228] : f32 from vector<2x8xf32>
        %241 = vector.from_elements %237, %238, %239, %240 : vector<1x4xf32>
        %242 = arith.mulf %241, %232 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %243 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %arg14 [%129, %219] : f32 into vector<2x8xf32>
        %245 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %246 = vector.insert %245, %244 [%129, %222] : f32 into vector<2x8xf32>
        %247 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %246 [%129, %225] : f32 into vector<2x8xf32>
        %249 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %250 = vector.insert %249, %248 [%129, %228] : f32 into vector<2x8xf32>
        %251 = arith.addf %242, %218 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %252 = vector.extract %251[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg15 [%129, %219] : f32 into vector<2x8xf32>
        %254 = vector.extract %251[0, 1] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %253 [%129, %222] : f32 into vector<2x8xf32>
        %256 = vector.extract %251[0, 2] : f32 from vector<1x4xf32>
        %257 = vector.insert %256, %255 [%129, %225] : f32 into vector<2x8xf32>
        %258 = vector.extract %251[0, 3] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %257 [%129, %228] : f32 into vector<2x8xf32>
        %260 = vector.extract %120#3[%129, %219] : f32 from vector<2x8xf32>
        %261 = vector.extract %120#3[%129, %222] : f32 from vector<2x8xf32>
        %262 = vector.extract %120#3[%129, %225] : f32 from vector<2x8xf32>
        %263 = vector.extract %120#3[%129, %228] : f32 from vector<2x8xf32>
        %264 = vector.insert %260, %arg16 [%129, %219] : f32 into vector<2x8xf32>
        %265 = vector.insert %261, %264 [%129, %222] : f32 into vector<2x8xf32>
        %266 = vector.insert %262, %265 [%129, %225] : f32 into vector<2x8xf32>
        %267 = vector.insert %263, %266 [%129, %228] : f32 into vector<2x8xf32>
        affine.yield %230, %236, %250, %259, %267 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %127:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %129 = vector.extract %117[%arg11, 0] : f32 from vector<2x1xf32>
        %130 = vector.insert %129, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %131 = vector.extract %112[%arg11, 0] : f32 from vector<2x1xf32>
        %132 = vector.insert %131, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %130, %132 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %128:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %129 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %131 = vector.extract %122#3[%129, %130] : f32 from vector<2x8xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %133 = vector.extract %122#3[%129, %132] : f32 from vector<2x8xf32>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %135 = vector.extract %122#3[%129, %134] : f32 from vector<2x8xf32>
        %136 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %137 = vector.extract %122#3[%129, %136] : f32 from vector<2x8xf32>
        %138 = vector.insert %131, %arg12 [%129, %130] : f32 into vector<2x8xf32>
        %139 = vector.insert %133, %138 [%129, %132] : f32 into vector<2x8xf32>
        %140 = vector.insert %135, %139 [%129, %134] : f32 into vector<2x8xf32>
        %141 = vector.insert %137, %140 [%129, %136] : f32 into vector<2x8xf32>
        %142 = vector.extract %124#3[%129, %130] : f32 from vector<2x8xf32>
        %143 = vector.extract %124#3[%129, %132] : f32 from vector<2x8xf32>
        %144 = vector.extract %124#3[%129, %134] : f32 from vector<2x8xf32>
        %145 = vector.extract %124#3[%129, %136] : f32 from vector<2x8xf32>
        %146 = vector.insert %142, %arg13 [%129, %130] : f32 into vector<2x8xf32>
        %147 = vector.insert %143, %146 [%129, %132] : f32 into vector<2x8xf32>
        %148 = vector.insert %144, %147 [%129, %134] : f32 into vector<2x8xf32>
        %149 = vector.insert %145, %148 [%129, %136] : f32 into vector<2x8xf32>
        %150 = vector.extract %126#3[%129, %130] : f32 from vector<2x8xf32>
        %151 = vector.extract %126#3[%129, %132] : f32 from vector<2x8xf32>
        %152 = vector.extract %126#3[%129, %134] : f32 from vector<2x8xf32>
        %153 = vector.extract %126#3[%129, %136] : f32 from vector<2x8xf32>
        %154 = vector.insert %150, %arg14 [%129, %130] : f32 into vector<2x8xf32>
        %155 = vector.insert %151, %154 [%129, %132] : f32 into vector<2x8xf32>
        %156 = vector.insert %152, %155 [%129, %134] : f32 into vector<2x8xf32>
        %157 = vector.insert %153, %156 [%129, %136] : f32 into vector<2x8xf32>
        affine.yield %141, %149, %157 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      gpu.barrier
      affine.yield %126#4, %127#0, %127#1, %128#0, %128#1, %128#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %87 = arith.index_cast %thread_id_x : index to i32
    %88 = arith.andi %87, %c16_i32 : i32
    %89 = arith.cmpi ne, %88, %c0_i32 : i32
    %90 = arith.andi %87, %c32_i32 : i32
    %91 = arith.cmpi ne, %90, %c0_i32 : i32
    %92 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %104 = vector.extract %86#5[%103, %102] : f32 from vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %106 = vector.extract %86#5[%103, %105] : f32 from vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %108 = vector.extract %86#5[%103, %107] : f32 from vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %110 = vector.extract %86#5[%103, %109] : f32 from vector<2x8xf32>
      %111 = arith.select %89, %104, %106 : f32
      %shuffleResult, %valid = gpu.shuffle xor %111, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %112 = arith.select %89, %shuffleResult, %104 : f32
      %113 = arith.select %89, %106, %shuffleResult : f32
      %114 = arith.select %89, %108, %110 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %114, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %89, %shuffleResult_13, %108 : f32
      %116 = arith.select %89, %110, %shuffleResult_13 : f32
      %117 = arith.select %91, %112, %115 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %117, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %91, %shuffleResult_15, %112 : f32
      %119 = arith.select %91, %115, %shuffleResult_15 : f32
      %120 = arith.select %91, %113, %116 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %120, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %91, %shuffleResult_17, %113 : f32
      %122 = arith.select %91, %116, %shuffleResult_17 : f32
      %123 = vector.insert %118, %arg5 [%103, %102] : f32 into vector<2x8xf32>
      %124 = vector.insert %121, %123 [%103, %105] : f32 into vector<2x8xf32>
      %125 = vector.insert %119, %124 [%103, %107] : f32 into vector<2x8xf32>
      %126 = vector.insert %122, %125 [%103, %109] : f32 into vector<2x8xf32>
      affine.yield %126 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %93 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %104 = vector.extract %86#4[%103, %102] : f32 from vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %106 = vector.extract %86#4[%103, %105] : f32 from vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %108 = vector.extract %86#4[%103, %107] : f32 from vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %110 = vector.extract %86#4[%103, %109] : f32 from vector<2x8xf32>
      %111 = arith.select %89, %104, %106 : f32
      %shuffleResult, %valid = gpu.shuffle xor %111, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %112 = arith.select %89, %shuffleResult, %104 : f32
      %113 = arith.select %89, %106, %shuffleResult : f32
      %114 = arith.select %89, %108, %110 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %114, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %89, %shuffleResult_13, %108 : f32
      %116 = arith.select %89, %110, %shuffleResult_13 : f32
      %117 = arith.select %91, %112, %115 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %117, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %91, %shuffleResult_15, %112 : f32
      %119 = arith.select %91, %115, %shuffleResult_15 : f32
      %120 = arith.select %91, %113, %116 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %120, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %91, %shuffleResult_17, %113 : f32
      %122 = arith.select %91, %116, %shuffleResult_17 : f32
      %123 = vector.insert %118, %arg5 [%103, %102] : f32 into vector<2x8xf32>
      %124 = vector.insert %121, %123 [%103, %105] : f32 into vector<2x8xf32>
      %125 = vector.insert %119, %124 [%103, %107] : f32 into vector<2x8xf32>
      %126 = vector.insert %122, %125 [%103, %109] : f32 into vector<2x8xf32>
      affine.yield %126 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %94 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %104 = vector.extract %86#3[%103, %102] : f32 from vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %106 = vector.extract %86#3[%103, %105] : f32 from vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %108 = vector.extract %86#3[%103, %107] : f32 from vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %110 = vector.extract %86#3[%103, %109] : f32 from vector<2x8xf32>
      %111 = arith.select %89, %104, %106 : f32
      %shuffleResult, %valid = gpu.shuffle xor %111, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %112 = arith.select %89, %shuffleResult, %104 : f32
      %113 = arith.select %89, %106, %shuffleResult : f32
      %114 = arith.select %89, %108, %110 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %114, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %89, %shuffleResult_13, %108 : f32
      %116 = arith.select %89, %110, %shuffleResult_13 : f32
      %117 = arith.select %91, %112, %115 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %117, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %91, %shuffleResult_15, %112 : f32
      %119 = arith.select %91, %115, %shuffleResult_15 : f32
      %120 = arith.select %91, %113, %116 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %120, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %91, %shuffleResult_17, %113 : f32
      %122 = arith.select %91, %116, %shuffleResult_17 : f32
      %123 = vector.insert %118, %arg5 [%103, %102] : f32 into vector<2x8xf32>
      %124 = vector.insert %121, %123 [%103, %105] : f32 into vector<2x8xf32>
      %125 = vector.insert %119, %124 [%103, %107] : f32 into vector<2x8xf32>
      %126 = vector.insert %122, %125 [%103, %109] : f32 into vector<2x8xf32>
      affine.yield %126 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %95 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %104 = vector.extract %86#0[%103, %102] : f32 from vector<2x8xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %106 = vector.extract %86#0[%103, %105] : f32 from vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %108 = vector.extract %86#0[%103, %107] : f32 from vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %110 = vector.extract %86#0[%103, %109] : f32 from vector<2x8xf32>
      %111 = arith.select %89, %104, %106 : f32
      %shuffleResult, %valid = gpu.shuffle xor %111, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %112 = arith.select %89, %shuffleResult, %104 : f32
      %113 = arith.select %89, %106, %shuffleResult : f32
      %114 = arith.select %89, %108, %110 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %114, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %115 = arith.select %89, %shuffleResult_13, %108 : f32
      %116 = arith.select %89, %110, %shuffleResult_13 : f32
      %117 = arith.select %91, %112, %115 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %117, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %91, %shuffleResult_15, %112 : f32
      %119 = arith.select %91, %115, %shuffleResult_15 : f32
      %120 = arith.select %91, %113, %116 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %120, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %91, %shuffleResult_17, %113 : f32
      %122 = arith.select %91, %116, %shuffleResult_17 : f32
      %123 = vector.insert %118, %arg5 [%103, %102] : f32 into vector<2x8xf32>
      %124 = vector.insert %121, %123 [%103, %105] : f32 into vector<2x8xf32>
      %125 = vector.insert %119, %124 [%103, %107] : f32 into vector<2x8xf32>
      %126 = vector.insert %122, %125 [%103, %109] : f32 into vector<2x8xf32>
      affine.yield %126 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %96:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7, %arg6 = %cst_7, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %103 = vector.extract %86#1[%102, 0] : f32 from vector<2x1xf32>
      %104 = vector.broadcast %103 : f32 to vector<1x4xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %106 = vector.insert %103, %arg5 [%102, %105] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.insert %103, %106 [%102, %107] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %110 = vector.insert %103, %108 [%102, %109] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %112 = vector.insert %103, %110 [%102, %111] : f32 into vector<2x8xf32>
      %113 = vector.extract %95[%102, %105] : f32 from vector<2x8xf32>
      %114 = vector.extract %95[%102, %107] : f32 from vector<2x8xf32>
      %115 = vector.extract %95[%102, %109] : f32 from vector<2x8xf32>
      %116 = vector.extract %95[%102, %111] : f32 from vector<2x8xf32>
      %117 = vector.from_elements %113, %114, %115, %116 : vector<1x4xf32>
      %118 = arith.divf %117, %104 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %119 = vector.extract %118[0, 0] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %arg6 [%102, %105] : f32 into vector<2x8xf32>
      %121 = vector.extract %118[0, 1] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%102, %107] : f32 into vector<2x8xf32>
      %123 = vector.extract %118[0, 2] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%102, %109] : f32 into vector<2x8xf32>
      %125 = vector.extract %118[0, 3] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%102, %111] : f32 into vector<2x8xf32>
      %127 = arith.truncf %118 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %128 = vector.extract %127[0, 0] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %arg7 [%102, %105] : f16 into vector<2x8xf16>
      %130 = vector.extract %127[0, 1] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%102, %107] : f16 into vector<2x8xf16>
      %132 = vector.extract %127[0, 2] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%102, %109] : f16 into vector<2x8xf16>
      %134 = vector.extract %127[0, 3] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%102, %111] : f16 into vector<2x8xf16>
      %136 = vector.insert %128, %arg8 [%102, %105] : f16 into vector<2x8xf16>
      %137 = vector.insert %130, %136 [%102, %107] : f16 into vector<2x8xf16>
      %138 = vector.insert %132, %137 [%102, %109] : f16 into vector<2x8xf16>
      %139 = vector.insert %134, %138 [%102, %111] : f16 into vector<2x8xf16>
      affine.yield %112, %126, %135, %139 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %97 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %98 = arith.cmpi ult, %97, %c4 : index
    scf.if %98 {
      affine.for %arg4 = 0 to 4 {
        %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %104 = vector.extract %96#3[%102, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %106 = vector.extract %96#3[%102, %105] : f16 from vector<2x8xf16>
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %108 = vector.extract %96#3[%102, %107] : f16 from vector<2x8xf16>
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %110 = vector.extract %96#3[%102, %109] : f16 from vector<2x8xf16>
        affine.store %104, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %108, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %110, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %99:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7, %arg6 = %cst_7, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %103 = vector.extract %86#1[%102, 0] : f32 from vector<2x1xf32>
      %104 = vector.broadcast %103 : f32 to vector<1x4xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %106 = vector.insert %103, %arg5 [%102, %105] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.insert %103, %106 [%102, %107] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %110 = vector.insert %103, %108 [%102, %109] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %112 = vector.insert %103, %110 [%102, %111] : f32 into vector<2x8xf32>
      %113 = vector.extract %94[%102, %105] : f32 from vector<2x8xf32>
      %114 = vector.extract %94[%102, %107] : f32 from vector<2x8xf32>
      %115 = vector.extract %94[%102, %109] : f32 from vector<2x8xf32>
      %116 = vector.extract %94[%102, %111] : f32 from vector<2x8xf32>
      %117 = vector.from_elements %113, %114, %115, %116 : vector<1x4xf32>
      %118 = arith.divf %117, %104 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %119 = vector.extract %118[0, 0] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %arg6 [%102, %105] : f32 into vector<2x8xf32>
      %121 = vector.extract %118[0, 1] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%102, %107] : f32 into vector<2x8xf32>
      %123 = vector.extract %118[0, 2] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%102, %109] : f32 into vector<2x8xf32>
      %125 = vector.extract %118[0, 3] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%102, %111] : f32 into vector<2x8xf32>
      %127 = arith.truncf %118 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %128 = vector.extract %127[0, 0] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %arg7 [%102, %105] : f16 into vector<2x8xf16>
      %130 = vector.extract %127[0, 1] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%102, %107] : f16 into vector<2x8xf16>
      %132 = vector.extract %127[0, 2] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%102, %109] : f16 into vector<2x8xf16>
      %134 = vector.extract %127[0, 3] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%102, %111] : f16 into vector<2x8xf16>
      %136 = vector.insert %128, %arg8 [%102, %105] : f16 into vector<2x8xf16>
      %137 = vector.insert %130, %136 [%102, %107] : f16 into vector<2x8xf16>
      %138 = vector.insert %132, %137 [%102, %109] : f16 into vector<2x8xf16>
      %139 = vector.insert %134, %138 [%102, %111] : f16 into vector<2x8xf16>
      affine.yield %112, %126, %135, %139 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %98 {
      affine.for %arg4 = 0 to 4 {
        %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %104 = vector.extract %99#3[%102, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %106 = vector.extract %99#3[%102, %105] : f16 from vector<2x8xf16>
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %108 = vector.extract %99#3[%102, %107] : f16 from vector<2x8xf16>
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %110 = vector.extract %99#3[%102, %109] : f16 from vector<2x8xf16>
        affine.store %104, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %108, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %110, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %100:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7, %arg6 = %cst_7, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %103 = vector.extract %86#1[%102, 0] : f32 from vector<2x1xf32>
      %104 = vector.broadcast %103 : f32 to vector<1x4xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %106 = vector.insert %103, %arg5 [%102, %105] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.insert %103, %106 [%102, %107] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %110 = vector.insert %103, %108 [%102, %109] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %112 = vector.insert %103, %110 [%102, %111] : f32 into vector<2x8xf32>
      %113 = vector.extract %93[%102, %105] : f32 from vector<2x8xf32>
      %114 = vector.extract %93[%102, %107] : f32 from vector<2x8xf32>
      %115 = vector.extract %93[%102, %109] : f32 from vector<2x8xf32>
      %116 = vector.extract %93[%102, %111] : f32 from vector<2x8xf32>
      %117 = vector.from_elements %113, %114, %115, %116 : vector<1x4xf32>
      %118 = arith.divf %117, %104 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %119 = vector.extract %118[0, 0] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %arg6 [%102, %105] : f32 into vector<2x8xf32>
      %121 = vector.extract %118[0, 1] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%102, %107] : f32 into vector<2x8xf32>
      %123 = vector.extract %118[0, 2] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%102, %109] : f32 into vector<2x8xf32>
      %125 = vector.extract %118[0, 3] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%102, %111] : f32 into vector<2x8xf32>
      %127 = arith.truncf %118 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %128 = vector.extract %127[0, 0] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %arg7 [%102, %105] : f16 into vector<2x8xf16>
      %130 = vector.extract %127[0, 1] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%102, %107] : f16 into vector<2x8xf16>
      %132 = vector.extract %127[0, 2] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%102, %109] : f16 into vector<2x8xf16>
      %134 = vector.extract %127[0, 3] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%102, %111] : f16 into vector<2x8xf16>
      %136 = vector.insert %128, %arg8 [%102, %105] : f16 into vector<2x8xf16>
      %137 = vector.insert %130, %136 [%102, %107] : f16 into vector<2x8xf16>
      %138 = vector.insert %132, %137 [%102, %109] : f16 into vector<2x8xf16>
      %139 = vector.insert %134, %138 [%102, %111] : f16 into vector<2x8xf16>
      affine.yield %112, %126, %135, %139 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %98 {
      affine.for %arg4 = 0 to 4 {
        %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %104 = vector.extract %100#3[%102, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %106 = vector.extract %100#3[%102, %105] : f16 from vector<2x8xf16>
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %108 = vector.extract %100#3[%102, %107] : f16 from vector<2x8xf16>
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %110 = vector.extract %100#3[%102, %109] : f16 from vector<2x8xf16>
        affine.store %104, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %108, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %110, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %101:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7, %arg6 = %cst_7, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %103 = vector.extract %86#1[%102, 0] : f32 from vector<2x1xf32>
      %104 = vector.broadcast %103 : f32 to vector<1x4xf32>
      %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %106 = vector.insert %103, %arg5 [%102, %105] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.insert %103, %106 [%102, %107] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %110 = vector.insert %103, %108 [%102, %109] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %112 = vector.insert %103, %110 [%102, %111] : f32 into vector<2x8xf32>
      %113 = vector.extract %92[%102, %105] : f32 from vector<2x8xf32>
      %114 = vector.extract %92[%102, %107] : f32 from vector<2x8xf32>
      %115 = vector.extract %92[%102, %109] : f32 from vector<2x8xf32>
      %116 = vector.extract %92[%102, %111] : f32 from vector<2x8xf32>
      %117 = vector.from_elements %113, %114, %115, %116 : vector<1x4xf32>
      %118 = arith.divf %117, %104 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %119 = vector.extract %118[0, 0] : f32 from vector<1x4xf32>
      %120 = vector.insert %119, %arg6 [%102, %105] : f32 into vector<2x8xf32>
      %121 = vector.extract %118[0, 1] : f32 from vector<1x4xf32>
      %122 = vector.insert %121, %120 [%102, %107] : f32 into vector<2x8xf32>
      %123 = vector.extract %118[0, 2] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %122 [%102, %109] : f32 into vector<2x8xf32>
      %125 = vector.extract %118[0, 3] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%102, %111] : f32 into vector<2x8xf32>
      %127 = arith.truncf %118 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %128 = vector.extract %127[0, 0] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %arg7 [%102, %105] : f16 into vector<2x8xf16>
      %130 = vector.extract %127[0, 1] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%102, %107] : f16 into vector<2x8xf16>
      %132 = vector.extract %127[0, 2] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%102, %109] : f16 into vector<2x8xf16>
      %134 = vector.extract %127[0, 3] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%102, %111] : f16 into vector<2x8xf16>
      %136 = vector.insert %128, %arg8 [%102, %105] : f16 into vector<2x8xf16>
      %137 = vector.insert %130, %136 [%102, %107] : f16 into vector<2x8xf16>
      %138 = vector.insert %132, %137 [%102, %109] : f16 into vector<2x8xf16>
      %139 = vector.insert %134, %138 [%102, %111] : f16 into vector<2x8xf16>
      affine.yield %112, %126, %135, %139 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %98 {
      affine.for %arg4 = 0 to 4 {
        %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %104 = vector.extract %101#3[%102, %103] : f16 from vector<2x8xf16>
        %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %106 = vector.extract %101#3[%102, %105] : f16 from vector<2x8xf16>
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %108 = vector.extract %101#3[%102, %107] : f16 from vector<2x8xf16>
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %110 = vector.extract %101#3[%102, %109] : f16 from vector<2x8xf16>
        affine.store %104, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %106, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %108, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %110, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
