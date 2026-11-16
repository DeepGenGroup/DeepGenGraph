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
    %10 = frisk.async_copy_gm_to_shm %subview to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %11 = arith.truncf %cst : f32 to f16
    %12:4 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1, %arg6 = %cst_1, %arg7 = %cst_1, %arg8 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %106 = vector.insert %cst_8, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %107 = vector.insert %cst_0, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %108 = vector.insert %cst_0, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %109 = vector.insert %cst_0, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %106, %107, %108, %109 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %13 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = vector.insert %cst_8, %arg5 [%107, %106] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %110 = vector.insert %cst_8, %108 [%107, %109] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %112 = vector.insert %cst_8, %110 [%107, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.insert %cst_8, %112 [%107, %113] : f32 into vector<2x8xf32>
      affine.yield %114 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %14 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1) -> (vector<2x1xf32>) {
      %106 = vector.insert %cst_8, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %106 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    frisk.wait_buffer_ready %10 : !frisk.async_task
    %view_9 = memref.view %alloc[%c8192][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %15 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%6, %7]
    %subview_10 = memref.subview %arg0[0, %15, %9, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %16 = frisk.async_copy_gm_to_shm %subview_10 to %view_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %thread_id_x = gpu.thread_id x
    %17 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128)>()[%thread_id_x]
    %18 = memref.load %view[%17] : memref<4096xf16, 3>
    %19 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 1)>()[%thread_id_x]
    %20 = memref.load %view[%19] : memref<4096xf16, 3>
    %21 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2)>()[%thread_id_x]
    %22 = memref.load %view[%21] : memref<4096xf16, 3>
    %23 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 3)>()[%thread_id_x]
    %24 = memref.load %view[%23] : memref<4096xf16, 3>
    %25 = vector.from_elements %18, %20, %22, %24 : vector<1x4xf16>
    %26 = vector.broadcast %11 : f16 to vector<1x4xf16>
    %27:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %25) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %107 = memref.load %view[%106] : memref<4096xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %109 = memref.load %view[%108] : memref<4096xf16, 3>
      %110 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %111 = memref.load %view[%110] : memref<4096xf16, 3>
      %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %113 = memref.load %view[%112] : memref<4096xf16, 3>
      %114 = vector.from_elements %107, %109, %111, %113 : vector<1x4xf16>
      %115 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %116 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %117 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %arg5 [%115, %116] : f16 into vector<2x8xf16>
      %119 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %120 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %118 [%115, %119] : f16 into vector<2x8xf16>
      %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %123 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %121 [%115, %122] : f16 into vector<2x8xf16>
      %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %126 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %124 [%115, %125] : f16 into vector<2x8xf16>
      %128 = vector.insert %11, %arg6 [%115, %116] : f16 into vector<2x8xf16>
      %129 = vector.insert %11, %128 [%115, %119] : f16 into vector<2x8xf16>
      %130 = vector.insert %11, %129 [%115, %122] : f16 into vector<2x8xf16>
      %131 = vector.insert %11, %130 [%115, %125] : f16 into vector<2x8xf16>
      %132 = arith.mulf %arg8, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %133 = vector.extract %132[0, 0] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %arg7 [%115, %116] : f16 into vector<2x8xf16>
      %135 = vector.extract %132[0, 1] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%115, %119] : f16 into vector<2x8xf16>
      %137 = vector.extract %132[0, 2] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%115, %122] : f16 into vector<2x8xf16>
      %139 = vector.extract %132[0, 3] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%115, %125] : f16 into vector<2x8xf16>
      affine.yield %127, %131, %140, %114 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %28 = arith.mulf %27#3, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %29 = vector.extract %28[0, 0] : f16 from vector<1x4xf16>
    %30 = vector.insert %29, %27#2 [1, 4] : f16 into vector<2x8xf16>
    %31 = vector.extract %28[0, 1] : f16 from vector<1x4xf16>
    %32 = vector.insert %31, %30 [1, 5] : f16 into vector<2x8xf16>
    %33 = vector.extract %28[0, 2] : f16 from vector<1x4xf16>
    %34 = vector.insert %33, %32 [1, 6] : f16 into vector<2x8xf16>
    %35 = vector.extract %28[0, 3] : f16 from vector<1x4xf16>
    %36 = vector.insert %35, %34 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %16 : !frisk.async_task
    %subview_11 = memref.subview %arg0[0, %15, %9, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %37 = frisk.async_copy_gm_to_shm %subview_11 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %38 = memref.load %view_9[%17] : memref<4096xf16, 3>
    %39 = memref.load %view_9[%19] : memref<4096xf16, 3>
    %40 = memref.load %view_9[%21] : memref<4096xf16, 3>
    %41 = memref.load %view_9[%23] : memref<4096xf16, 3>
    %42 = vector.from_elements %38, %39, %40, %41 : vector<1x4xf16>
    %43:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %42) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %107 = memref.load %view_9[%106] : memref<4096xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %109 = memref.load %view_9[%108] : memref<4096xf16, 3>
      %110 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %111 = memref.load %view_9[%110] : memref<4096xf16, 3>
      %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %113 = memref.load %view_9[%112] : memref<4096xf16, 3>
      %114 = vector.from_elements %107, %109, %111, %113 : vector<1x4xf16>
      %115 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %116 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %117 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %arg5 [%115, %116] : f16 into vector<2x8xf16>
      %119 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %120 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %118 [%115, %119] : f16 into vector<2x8xf16>
      %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %123 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %121 [%115, %122] : f16 into vector<2x8xf16>
      %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %126 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %124 [%115, %125] : f16 into vector<2x8xf16>
      %128 = vector.insert %11, %arg6 [%115, %116] : f16 into vector<2x8xf16>
      %129 = vector.insert %11, %128 [%115, %119] : f16 into vector<2x8xf16>
      %130 = vector.insert %11, %129 [%115, %122] : f16 into vector<2x8xf16>
      %131 = vector.insert %11, %130 [%115, %125] : f16 into vector<2x8xf16>
      %132 = arith.mulf %arg8, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %133 = vector.extract %132[0, 0] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %arg7 [%115, %116] : f16 into vector<2x8xf16>
      %135 = vector.extract %132[0, 1] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%115, %119] : f16 into vector<2x8xf16>
      %137 = vector.extract %132[0, 2] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%115, %122] : f16 into vector<2x8xf16>
      %139 = vector.extract %132[0, 3] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%115, %125] : f16 into vector<2x8xf16>
      affine.yield %127, %131, %140, %114 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %44 = arith.mulf %43#3, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %45 = vector.extract %44[0, 0] : f16 from vector<1x4xf16>
    %46 = vector.insert %45, %43#2 [1, 4] : f16 into vector<2x8xf16>
    %47 = vector.extract %44[0, 1] : f16 from vector<1x4xf16>
    %48 = vector.insert %47, %46 [1, 5] : f16 into vector<2x8xf16>
    %49 = vector.extract %44[0, 2] : f16 from vector<1x4xf16>
    %50 = vector.insert %49, %48 [1, 6] : f16 into vector<2x8xf16>
    %51 = vector.extract %44[0, 3] : f16 from vector<1x4xf16>
    %52 = vector.insert %51, %50 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %37 : !frisk.async_task
    %subview_12 = memref.subview %arg0[0, %15, %9, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %53 = frisk.async_copy_gm_to_shm %subview_12 to %view_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %54 = memref.load %view[%17] : memref<4096xf16, 3>
    %55 = memref.load %view[%19] : memref<4096xf16, 3>
    %56 = memref.load %view[%21] : memref<4096xf16, 3>
    %57 = memref.load %view[%23] : memref<4096xf16, 3>
    %58 = vector.from_elements %54, %55, %56, %57 : vector<1x4xf16>
    %59:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %58) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %107 = memref.load %view[%106] : memref<4096xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %109 = memref.load %view[%108] : memref<4096xf16, 3>
      %110 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %111 = memref.load %view[%110] : memref<4096xf16, 3>
      %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %113 = memref.load %view[%112] : memref<4096xf16, 3>
      %114 = vector.from_elements %107, %109, %111, %113 : vector<1x4xf16>
      %115 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %116 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %117 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %arg5 [%115, %116] : f16 into vector<2x8xf16>
      %119 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %120 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %118 [%115, %119] : f16 into vector<2x8xf16>
      %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %123 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %121 [%115, %122] : f16 into vector<2x8xf16>
      %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %126 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %124 [%115, %125] : f16 into vector<2x8xf16>
      %128 = vector.insert %11, %arg6 [%115, %116] : f16 into vector<2x8xf16>
      %129 = vector.insert %11, %128 [%115, %119] : f16 into vector<2x8xf16>
      %130 = vector.insert %11, %129 [%115, %122] : f16 into vector<2x8xf16>
      %131 = vector.insert %11, %130 [%115, %125] : f16 into vector<2x8xf16>
      %132 = arith.mulf %arg8, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %133 = vector.extract %132[0, 0] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %arg7 [%115, %116] : f16 into vector<2x8xf16>
      %135 = vector.extract %132[0, 1] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%115, %119] : f16 into vector<2x8xf16>
      %137 = vector.extract %132[0, 2] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%115, %122] : f16 into vector<2x8xf16>
      %139 = vector.extract %132[0, 3] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%115, %125] : f16 into vector<2x8xf16>
      affine.yield %127, %131, %140, %114 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %60 = arith.mulf %59#3, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %61 = vector.extract %60[0, 0] : f16 from vector<1x4xf16>
    %62 = vector.insert %61, %59#2 [1, 4] : f16 into vector<2x8xf16>
    %63 = vector.extract %60[0, 1] : f16 from vector<1x4xf16>
    %64 = vector.insert %63, %62 [1, 5] : f16 into vector<2x8xf16>
    %65 = vector.extract %60[0, 2] : f16 from vector<1x4xf16>
    %66 = vector.insert %65, %64 [1, 6] : f16 into vector<2x8xf16>
    %67 = vector.extract %60[0, 3] : f16 from vector<1x4xf16>
    %68 = vector.insert %67, %66 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %53 : !frisk.async_task
    %69 = memref.load %view_9[%17] : memref<4096xf16, 3>
    %70 = memref.load %view_9[%19] : memref<4096xf16, 3>
    %71 = memref.load %view_9[%21] : memref<4096xf16, 3>
    %72 = memref.load %view_9[%23] : memref<4096xf16, 3>
    %73 = vector.from_elements %69, %70, %71, %72 : vector<1x4xf16>
    %74:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %73) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %107 = memref.load %view_9[%106] : memref<4096xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %109 = memref.load %view_9[%108] : memref<4096xf16, 3>
      %110 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %111 = memref.load %view_9[%110] : memref<4096xf16, 3>
      %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %113 = memref.load %view_9[%112] : memref<4096xf16, 3>
      %114 = vector.from_elements %107, %109, %111, %113 : vector<1x4xf16>
      %115 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %116 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %117 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %arg5 [%115, %116] : f16 into vector<2x8xf16>
      %119 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %120 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %118 [%115, %119] : f16 into vector<2x8xf16>
      %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %123 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %121 [%115, %122] : f16 into vector<2x8xf16>
      %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %126 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %124 [%115, %125] : f16 into vector<2x8xf16>
      %128 = vector.insert %11, %arg6 [%115, %116] : f16 into vector<2x8xf16>
      %129 = vector.insert %11, %128 [%115, %119] : f16 into vector<2x8xf16>
      %130 = vector.insert %11, %129 [%115, %122] : f16 into vector<2x8xf16>
      %131 = vector.insert %11, %130 [%115, %125] : f16 into vector<2x8xf16>
      %132 = arith.mulf %arg8, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %133 = vector.extract %132[0, 0] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %arg7 [%115, %116] : f16 into vector<2x8xf16>
      %135 = vector.extract %132[0, 1] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%115, %119] : f16 into vector<2x8xf16>
      %137 = vector.extract %132[0, 2] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%115, %122] : f16 into vector<2x8xf16>
      %139 = vector.extract %132[0, 3] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%115, %125] : f16 into vector<2x8xf16>
      affine.yield %127, %131, %140, %114 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %75 = arith.mulf %74#3, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %76 = vector.extract %75[0, 0] : f16 from vector<1x4xf16>
    %77 = vector.insert %76, %74#2 [1, 4] : f16 into vector<2x8xf16>
    %78 = vector.extract %75[0, 1] : f16 from vector<1x4xf16>
    %79 = vector.insert %78, %77 [1, 5] : f16 into vector<2x8xf16>
    %80 = vector.extract %75[0, 2] : f16 from vector<1x4xf16>
    %81 = vector.insert %80, %79 [1, 6] : f16 into vector<2x8xf16>
    %82 = vector.extract %75[0, 3] : f16 from vector<1x4xf16>
    %83 = vector.insert %82, %81 [1, 7] : f16 into vector<2x8xf16>
    gpu.barrier
    %84 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
    %85 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_3) -> (vector<2x32xf32>) {
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg4)
      %108 = vector.insert %cst_8, %arg5 [%107, %106] : f32 into vector<2x32xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %110 = vector.insert %cst_8, %108 [%107, %109] : f32 into vector<2x32xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %112 = vector.insert %cst_8, %110 [%107, %111] : f32 into vector<2x32xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %114 = vector.insert %cst_8, %112 [%107, %113] : f32 into vector<2x32xf32>
      affine.yield %114 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
    %86 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
    %87 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
    %88 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
    %89 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
    %90:6 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %13, %arg6 = %14, %arg7 = %12#3, %arg8 = %13, %arg9 = %13, %arg10 = %13) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %106 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %subview_13 = memref.subview %transpose[0, %84, 0, %106] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %107 = frisk.async_copy_gm_to_shm %subview_13 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %107 : !frisk.async_task
      %108 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_14 = memref.subview %transpose[0, %108, 32, %106] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %109 = frisk.async_copy_gm_to_shm %subview_14 to %view_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %110:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %142 = memref.load %view[%141] : memref<4096xf16, 3>
        %143 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %144 = memref.load %view[%143] : memref<4096xf16, 3>
        %145 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %146 = memref.load %view[%145] : memref<4096xf16, 3>
        %147 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %148 = memref.load %view[%147] : memref<4096xf16, 3>
        %149 = vector.from_elements %142, %144, %146, %148 : vector<4x1xf16>
        %150 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %151 = memref.load %view[%150] : memref<4096xf16, 3>
        %152 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %153 = memref.load %view[%152] : memref<4096xf16, 3>
        %154 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %155 = memref.load %view[%154] : memref<4096xf16, 3>
        %156 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %157 = memref.load %view[%156] : memref<4096xf16, 3>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<4x1xf16>
        %159 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %160 = vector.extract %36[%159, 0] : f16 from vector<2x8xf16>
        %161 = vector.extract %36[%159, 1] : f16 from vector<2x8xf16>
        %162 = vector.extract %36[%159, 2] : f16 from vector<2x8xf16>
        %163 = vector.extract %36[%159, 3] : f16 from vector<2x8xf16>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf16>
        %165 = frisk.warp_mma_rr %164, %149, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %166 = vector.extract %36[%159, 4] : f16 from vector<2x8xf16>
        %167 = vector.extract %36[%159, 5] : f16 from vector<2x8xf16>
        %168 = vector.extract %36[%159, 6] : f16 from vector<2x8xf16>
        %169 = vector.extract %36[%159, 7] : f16 from vector<2x8xf16>
        %170 = vector.from_elements %166, %167, %168, %169 : vector<1x4xf16>
        %171 = frisk.warp_mma_rr %170, %158, %165 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %173 = vector.extract %85[%159, %172] : f32 from vector<2x32xf32>
        %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %175 = vector.extract %85[%159, %174] : f32 from vector<2x32xf32>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %177 = vector.extract %85[%159, %176] : f32 from vector<2x32xf32>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %179 = vector.extract %85[%159, %178] : f32 from vector<2x32xf32>
        %180 = vector.from_elements %173, %175, %177, %179 : vector<1x4xf32>
        %181 = arith.addf %180, %171 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %182 = vector.extract %181[0, 0] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %arg12 [%159, %172] : f32 into vector<2x32xf32>
        %184 = vector.extract %181[0, 1] : f32 from vector<1x4xf32>
        %185 = vector.insert %184, %183 [%159, %174] : f32 into vector<2x32xf32>
        %186 = vector.extract %181[0, 2] : f32 from vector<1x4xf32>
        %187 = vector.insert %186, %185 [%159, %176] : f32 into vector<2x32xf32>
        %188 = vector.extract %181[0, 3] : f32 from vector<1x4xf32>
        %189 = vector.insert %188, %187 [%159, %178] : f32 into vector<2x32xf32>
        %190 = vector.insert %182, %arg13 [%159, %172] : f32 into vector<2x32xf32>
        %191 = vector.insert %184, %190 [%159, %174] : f32 into vector<2x32xf32>
        %192 = vector.insert %186, %191 [%159, %176] : f32 into vector<2x32xf32>
        %193 = vector.insert %188, %192 [%159, %178] : f32 into vector<2x32xf32>
        affine.yield %189, %193 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %109 : !frisk.async_task
      %111 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 2048) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_15 = memref.subview %transpose[0, %111, 64, %106] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %112 = frisk.async_copy_gm_to_shm %subview_15 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %113:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %142 = memref.load %view_9[%141] : memref<4096xf16, 3>
        %143 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %144 = memref.load %view_9[%143] : memref<4096xf16, 3>
        %145 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %146 = memref.load %view_9[%145] : memref<4096xf16, 3>
        %147 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %148 = memref.load %view_9[%147] : memref<4096xf16, 3>
        %149 = vector.from_elements %142, %144, %146, %148 : vector<4x1xf16>
        %150 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %151 = memref.load %view_9[%150] : memref<4096xf16, 3>
        %152 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %153 = memref.load %view_9[%152] : memref<4096xf16, 3>
        %154 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %155 = memref.load %view_9[%154] : memref<4096xf16, 3>
        %156 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %157 = memref.load %view_9[%156] : memref<4096xf16, 3>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<4x1xf16>
        %159 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %160 = vector.extract %52[%159, 0] : f16 from vector<2x8xf16>
        %161 = vector.extract %52[%159, 1] : f16 from vector<2x8xf16>
        %162 = vector.extract %52[%159, 2] : f16 from vector<2x8xf16>
        %163 = vector.extract %52[%159, 3] : f16 from vector<2x8xf16>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf16>
        %165 = frisk.warp_mma_rr %164, %149, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %166 = vector.extract %52[%159, 4] : f16 from vector<2x8xf16>
        %167 = vector.extract %52[%159, 5] : f16 from vector<2x8xf16>
        %168 = vector.extract %52[%159, 6] : f16 from vector<2x8xf16>
        %169 = vector.extract %52[%159, 7] : f16 from vector<2x8xf16>
        %170 = vector.from_elements %166, %167, %168, %169 : vector<1x4xf16>
        %171 = frisk.warp_mma_rr %170, %158, %165 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %173 = vector.extract %110#1[%159, %172] : f32 from vector<2x32xf32>
        %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %175 = vector.extract %110#1[%159, %174] : f32 from vector<2x32xf32>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %177 = vector.extract %110#1[%159, %176] : f32 from vector<2x32xf32>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %179 = vector.extract %110#1[%159, %178] : f32 from vector<2x32xf32>
        %180 = vector.from_elements %173, %175, %177, %179 : vector<1x4xf32>
        %181 = arith.addf %180, %171 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %182 = vector.extract %181[0, 0] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %arg12 [%159, %172] : f32 into vector<2x32xf32>
        %184 = vector.extract %181[0, 1] : f32 from vector<1x4xf32>
        %185 = vector.insert %184, %183 [%159, %174] : f32 into vector<2x32xf32>
        %186 = vector.extract %181[0, 2] : f32 from vector<1x4xf32>
        %187 = vector.insert %186, %185 [%159, %176] : f32 into vector<2x32xf32>
        %188 = vector.extract %181[0, 3] : f32 from vector<1x4xf32>
        %189 = vector.insert %188, %187 [%159, %178] : f32 into vector<2x32xf32>
        %190 = vector.insert %182, %arg13 [%159, %172] : f32 into vector<2x32xf32>
        %191 = vector.insert %184, %190 [%159, %174] : f32 into vector<2x32xf32>
        %192 = vector.insert %186, %191 [%159, %176] : f32 into vector<2x32xf32>
        %193 = vector.insert %188, %192 [%159, %178] : f32 into vector<2x32xf32>
        affine.yield %189, %193 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %112 : !frisk.async_task
      %114 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 3072) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_16 = memref.subview %transpose[0, %114, 96, %106] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %115 = frisk.async_copy_gm_to_shm %subview_16 to %view_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %116:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %142 = memref.load %view[%141] : memref<4096xf16, 3>
        %143 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %144 = memref.load %view[%143] : memref<4096xf16, 3>
        %145 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %146 = memref.load %view[%145] : memref<4096xf16, 3>
        %147 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %148 = memref.load %view[%147] : memref<4096xf16, 3>
        %149 = vector.from_elements %142, %144, %146, %148 : vector<4x1xf16>
        %150 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %151 = memref.load %view[%150] : memref<4096xf16, 3>
        %152 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %153 = memref.load %view[%152] : memref<4096xf16, 3>
        %154 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %155 = memref.load %view[%154] : memref<4096xf16, 3>
        %156 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %157 = memref.load %view[%156] : memref<4096xf16, 3>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<4x1xf16>
        %159 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %160 = vector.extract %68[%159, 0] : f16 from vector<2x8xf16>
        %161 = vector.extract %68[%159, 1] : f16 from vector<2x8xf16>
        %162 = vector.extract %68[%159, 2] : f16 from vector<2x8xf16>
        %163 = vector.extract %68[%159, 3] : f16 from vector<2x8xf16>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf16>
        %165 = frisk.warp_mma_rr %164, %149, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %166 = vector.extract %68[%159, 4] : f16 from vector<2x8xf16>
        %167 = vector.extract %68[%159, 5] : f16 from vector<2x8xf16>
        %168 = vector.extract %68[%159, 6] : f16 from vector<2x8xf16>
        %169 = vector.extract %68[%159, 7] : f16 from vector<2x8xf16>
        %170 = vector.from_elements %166, %167, %168, %169 : vector<1x4xf16>
        %171 = frisk.warp_mma_rr %170, %158, %165 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %173 = vector.extract %113#1[%159, %172] : f32 from vector<2x32xf32>
        %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %175 = vector.extract %113#1[%159, %174] : f32 from vector<2x32xf32>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %177 = vector.extract %113#1[%159, %176] : f32 from vector<2x32xf32>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %179 = vector.extract %113#1[%159, %178] : f32 from vector<2x32xf32>
        %180 = vector.from_elements %173, %175, %177, %179 : vector<1x4xf32>
        %181 = arith.addf %180, %171 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %182 = vector.extract %181[0, 0] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %arg12 [%159, %172] : f32 into vector<2x32xf32>
        %184 = vector.extract %181[0, 1] : f32 from vector<1x4xf32>
        %185 = vector.insert %184, %183 [%159, %174] : f32 into vector<2x32xf32>
        %186 = vector.extract %181[0, 2] : f32 from vector<1x4xf32>
        %187 = vector.insert %186, %185 [%159, %176] : f32 into vector<2x32xf32>
        %188 = vector.extract %181[0, 3] : f32 from vector<1x4xf32>
        %189 = vector.insert %188, %187 [%159, %178] : f32 into vector<2x32xf32>
        %190 = vector.insert %182, %arg13 [%159, %172] : f32 into vector<2x32xf32>
        %191 = vector.insert %184, %190 [%159, %174] : f32 into vector<2x32xf32>
        %192 = vector.insert %186, %191 [%159, %176] : f32 into vector<2x32xf32>
        %193 = vector.insert %188, %192 [%159, %178] : f32 into vector<2x32xf32>
        affine.yield %189, %193 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %115 : !frisk.async_task
      %subview_17 = memref.subview %arg2[0, %84, %106, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %117 = frisk.async_copy_gm_to_shm %subview_17 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %118:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %142 = memref.load %view_9[%141] : memref<4096xf16, 3>
        %143 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %144 = memref.load %view_9[%143] : memref<4096xf16, 3>
        %145 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %146 = memref.load %view_9[%145] : memref<4096xf16, 3>
        %147 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %148 = memref.load %view_9[%147] : memref<4096xf16, 3>
        %149 = vector.from_elements %142, %144, %146, %148 : vector<4x1xf16>
        %150 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %151 = memref.load %view_9[%150] : memref<4096xf16, 3>
        %152 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %153 = memref.load %view_9[%152] : memref<4096xf16, 3>
        %154 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %155 = memref.load %view_9[%154] : memref<4096xf16, 3>
        %156 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %157 = memref.load %view_9[%156] : memref<4096xf16, 3>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<4x1xf16>
        %159 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %160 = vector.extract %83[%159, 0] : f16 from vector<2x8xf16>
        %161 = vector.extract %83[%159, 1] : f16 from vector<2x8xf16>
        %162 = vector.extract %83[%159, 2] : f16 from vector<2x8xf16>
        %163 = vector.extract %83[%159, 3] : f16 from vector<2x8xf16>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf16>
        %165 = frisk.warp_mma_rr %164, %149, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %166 = vector.extract %83[%159, 4] : f16 from vector<2x8xf16>
        %167 = vector.extract %83[%159, 5] : f16 from vector<2x8xf16>
        %168 = vector.extract %83[%159, 6] : f16 from vector<2x8xf16>
        %169 = vector.extract %83[%159, 7] : f16 from vector<2x8xf16>
        %170 = vector.from_elements %166, %167, %168, %169 : vector<1x4xf16>
        %171 = frisk.warp_mma_rr %170, %158, %165 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %173 = vector.extract %116#1[%159, %172] : f32 from vector<2x32xf32>
        %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %175 = vector.extract %116#1[%159, %174] : f32 from vector<2x32xf32>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %177 = vector.extract %116#1[%159, %176] : f32 from vector<2x32xf32>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %179 = vector.extract %116#1[%159, %178] : f32 from vector<2x32xf32>
        %180 = vector.from_elements %173, %175, %177, %179 : vector<1x4xf32>
        %181 = arith.addf %180, %171 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %182 = vector.extract %181[0, 0] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %arg12 [%159, %172] : f32 into vector<2x32xf32>
        %184 = vector.extract %181[0, 1] : f32 from vector<1x4xf32>
        %185 = vector.insert %184, %183 [%159, %174] : f32 into vector<2x32xf32>
        %186 = vector.extract %181[0, 2] : f32 from vector<1x4xf32>
        %187 = vector.insert %186, %185 [%159, %176] : f32 into vector<2x32xf32>
        %188 = vector.extract %181[0, 3] : f32 from vector<1x4xf32>
        %189 = vector.insert %188, %187 [%159, %178] : f32 into vector<2x32xf32>
        %190 = vector.insert %182, %arg13 [%159, %172] : f32 into vector<2x32xf32>
        %191 = vector.insert %184, %190 [%159, %174] : f32 into vector<2x32xf32>
        %192 = vector.insert %186, %191 [%159, %176] : f32 into vector<2x32xf32>
        %193 = vector.insert %188, %192 [%159, %178] : f32 into vector<2x32xf32>
        affine.yield %189, %193 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %119 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %143 = vector.extract %118#1[%141, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %145 = vector.extract %118#1[%141, %144] : f32 from vector<2x32xf32>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %147 = vector.extract %118#1[%141, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %149 = vector.extract %118#1[%141, %148] : f32 from vector<2x32xf32>
        %150 = vector.insert %143, %arg12 [%141, %142] : f32 into vector<2x32xf32>
        %151 = vector.insert %145, %150 [%141, %144] : f32 into vector<2x32xf32>
        %152 = vector.insert %147, %151 [%141, %146] : f32 into vector<2x32xf32>
        %153 = vector.insert %149, %152 [%141, %148] : f32 into vector<2x32xf32>
        affine.yield %153 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %120 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %141 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_0) -> (f32) {
          %145 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %146 = vector.extract %119[%arg11, %145] : f32 from vector<2x32xf32>
          %147 = arith.maxnumf %arg14, %146 : f32
          %148 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %149 = vector.extract %119[%arg11, %148] : f32 from vector<2x32xf32>
          %150 = arith.maxnumf %147, %149 : f32
          %151 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %152 = vector.extract %119[%arg11, %151] : f32 from vector<2x32xf32>
          %153 = arith.maxnumf %150, %152 : f32
          %154 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %155 = vector.extract %119[%arg11, %154] : f32 from vector<2x32xf32>
          %156 = arith.maxnumf %153, %155 : f32
          affine.yield %156 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %141, %c16_i32, %c64_i32 : f32
        %142 = arith.maxnumf %141, %shuffleResult : f32
        %shuffleResult_21, %valid_22 = gpu.shuffle xor %142, %c32_i32, %c64_i32 : f32
        %143 = arith.maxnumf %142, %shuffleResult_21 : f32
        %144 = vector.insert %143, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %121 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %141 = vector.extract %120[%arg11, 0] : f32 from vector<2x1xf32>
        %142 = vector.broadcast %141 : f32 to vector<1x1xf32>
        %143 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %144 = vector.broadcast %143 : f32 to vector<1x1xf32>
        %145 = arith.maxnumf %142, %144 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %146 = vector.extract %145[0, 0] : f32 from vector<1x1xf32>
        %147 = vector.insert %146, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %147 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %122:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %142 = vector.extract %121[%141, 0] : f32 from vector<2x1xf32>
        %143 = vector.broadcast %142 : f32 to vector<1x4xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %145 = vector.insert %142, %arg12 [%141, %144] : f32 into vector<2x32xf32>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %147 = vector.insert %142, %145 [%141, %146] : f32 into vector<2x32xf32>
        %148 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %149 = vector.insert %142, %147 [%141, %148] : f32 into vector<2x32xf32>
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %151 = vector.insert %142, %149 [%141, %150] : f32 into vector<2x32xf32>
        %152 = vector.extract %118#1[%141, %144] : f32 from vector<2x32xf32>
        %153 = vector.extract %118#1[%141, %146] : f32 from vector<2x32xf32>
        %154 = vector.extract %118#1[%141, %148] : f32 from vector<2x32xf32>
        %155 = vector.extract %118#1[%141, %150] : f32 from vector<2x32xf32>
        %156 = vector.from_elements %152, %153, %154, %155 : vector<1x4xf32>
        %157 = arith.subf %156, %143 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg13 [%141, %144] : f32 into vector<2x32xf32>
        %160 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%141, %146] : f32 into vector<2x32xf32>
        %162 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%141, %148] : f32 into vector<2x32xf32>
        %164 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%141, %150] : f32 into vector<2x32xf32>
        affine.yield %151, %165 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %123:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %141 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %142 = vector.broadcast %141 : f32 to vector<1x1xf32>
        %143 = vector.extract %121[%arg11, 0] : f32 from vector<2x1xf32>
        %144 = vector.broadcast %143 : f32 to vector<1x1xf32>
        %145 = arith.subf %142, %144 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %146 = vector.extract %145[0, 0] : f32 from vector<1x1xf32>
        %147 = vector.insert %146, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %148 = math.exp2 %145 {tiled = true} : vector<1x1xf32>
        %149 = vector.extract %148[0, 0] : f32 from vector<1x1xf32>
        %150 = vector.insert %149, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %151 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %152 = vector.broadcast %151 : f32 to vector<1x1xf32>
        %153 = arith.mulf %152, %148 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x1xf32>
        %155 = vector.insert %154, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %147, %150, %155 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %124:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %143 = vector.extract %122#1[%141, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %145 = vector.extract %122#1[%141, %144] : f32 from vector<2x32xf32>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %147 = vector.extract %122#1[%141, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %149 = vector.extract %122#1[%141, %148] : f32 from vector<2x32xf32>
        %150 = vector.from_elements %143, %145, %147, %149 : vector<1x4xf32>
        %151 = math.exp2 %150 {tiled = true} : vector<1x4xf32>
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%141, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%141, %144] : f32 into vector<2x32xf32>
        %156 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%141, %146] : f32 into vector<2x32xf32>
        %158 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%141, %148] : f32 into vector<2x32xf32>
        %160 = vector.insert %152, %arg13 [%141, %142] : f32 into vector<2x32xf32>
        %161 = vector.insert %154, %160 [%141, %144] : f32 into vector<2x32xf32>
        %162 = vector.insert %156, %161 [%141, %146] : f32 into vector<2x32xf32>
        %163 = vector.insert %158, %162 [%141, %148] : f32 into vector<2x32xf32>
        affine.yield %159, %163 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %125 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %141 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_8) -> (f32) {
          %145 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %146 = vector.extract %124#1[%arg11, %145] : f32 from vector<2x32xf32>
          %147 = arith.addf %arg14, %146 : f32
          %148 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %149 = vector.extract %124#1[%arg11, %148] : f32 from vector<2x32xf32>
          %150 = arith.addf %147, %149 : f32
          %151 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %152 = vector.extract %124#1[%arg11, %151] : f32 from vector<2x32xf32>
          %153 = arith.addf %150, %152 : f32
          %154 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %155 = vector.extract %124#1[%arg11, %154] : f32 from vector<2x32xf32>
          %156 = arith.addf %153, %155 : f32
          affine.yield %156 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %141, %c16_i32, %c64_i32 : f32
        %142 = arith.addf %141, %shuffleResult : f32
        %shuffleResult_21, %valid_22 = gpu.shuffle xor %142, %c32_i32, %c64_i32 : f32
        %143 = arith.addf %142, %shuffleResult_21 : f32
        %144 = vector.insert %143, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %144 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %126 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %141 = vector.extract %125[%arg11, 0] : f32 from vector<2x1xf32>
        %142 = vector.broadcast %141 : f32 to vector<1x1xf32>
        %143 = vector.extract %123#2[%arg11, 0] : f32 from vector<2x1xf32>
        %144 = vector.broadcast %143 : f32 to vector<1x1xf32>
        %145 = arith.addf %142, %144 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %146 = vector.extract %145[0, 0] : f32 from vector<1x1xf32>
        %147 = vector.insert %146, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %147 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %127 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf16>) {
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %143 = vector.extract %124#0[%141, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %145 = vector.extract %124#0[%141, %144] : f32 from vector<2x32xf32>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %147 = vector.extract %124#0[%141, %146] : f32 from vector<2x32xf32>
        %148 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %149 = vector.extract %124#0[%141, %148] : f32 from vector<2x32xf32>
        %150 = vector.from_elements %143, %145, %147, %149 : vector<1x4xf32>
        %151 = arith.truncf %150 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %152 = vector.extract %151[0, 0] : f16 from vector<1x4xf16>
        %153 = vector.insert %152, %arg12 [%141, %142] : f16 into vector<2x32xf16>
        %154 = vector.extract %151[0, 1] : f16 from vector<1x4xf16>
        %155 = vector.insert %154, %153 [%141, %144] : f16 into vector<2x32xf16>
        %156 = vector.extract %151[0, 2] : f16 from vector<1x4xf16>
        %157 = vector.insert %156, %155 [%141, %146] : f16 into vector<2x32xf16>
        %158 = vector.extract %151[0, 3] : f16 from vector<1x4xf16>
        %159 = vector.insert %158, %157 [%141, %148] : f16 into vector<2x32xf16>
        affine.yield %159 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %117 : !frisk.async_task
      %128 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %141 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %142 = frisk.read_matrix_from_shm %view[%141] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %143 = vector.bitcast %142 : vector<4xi32> to vector<8xf16>
        %144 = vector.extract %143[0] : f16 from vector<8xf16>
        %145 = vector.extract %143[%86] : f16 from vector<8xf16>
        %146 = vector.extract %143[1] : f16 from vector<8xf16>
        %147 = vector.extract %143[%87] : f16 from vector<8xf16>
        %148 = vector.extract %143[2] : f16 from vector<8xf16>
        %149 = vector.extract %143[%88] : f16 from vector<8xf16>
        %150 = vector.extract %143[3] : f16 from vector<8xf16>
        %151 = vector.extract %143[%89] : f16 from vector<8xf16>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %153 = vector.insert %144, %arg12 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %145, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %156 = vector.insert %146, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %147, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %159 = vector.insert %148, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %149, %159 [%158, 1] : f16 into vector<32x2xf16>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %162 = vector.insert %150, %160 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        affine.yield %163 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_18 = memref.subview %arg2[0, %84, %106, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      %129 = frisk.async_copy_gm_to_shm %subview_18 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %130:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7, %arg15 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %142 = vector.extract %127[%141, 0] : f16 from vector<2x32xf16>
        %143 = vector.extract %127[%141, 1] : f16 from vector<2x32xf16>
        %144 = vector.extract %127[%141, 2] : f16 from vector<2x32xf16>
        %145 = vector.extract %127[%141, 3] : f16 from vector<2x32xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %148 = vector.extract %128[0, %147] : f16 from vector<32x2xf16>
        %149 = vector.extract %128[1, %147] : f16 from vector<32x2xf16>
        %150 = vector.extract %128[2, %147] : f16 from vector<32x2xf16>
        %151 = vector.extract %128[3, %147] : f16 from vector<32x2xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<4x1xf16>
        %153 = frisk.warp_mma_rr %146, %152, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = vector.extract %127[%141, 4] : f16 from vector<2x32xf16>
        %155 = vector.extract %127[%141, 5] : f16 from vector<2x32xf16>
        %156 = vector.extract %127[%141, 6] : f16 from vector<2x32xf16>
        %157 = vector.extract %127[%141, 7] : f16 from vector<2x32xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = vector.extract %128[4, %147] : f16 from vector<32x2xf16>
        %160 = vector.extract %128[5, %147] : f16 from vector<32x2xf16>
        %161 = vector.extract %128[6, %147] : f16 from vector<32x2xf16>
        %162 = vector.extract %128[7, %147] : f16 from vector<32x2xf16>
        %163 = vector.from_elements %159, %160, %161, %162 : vector<4x1xf16>
        %164 = frisk.warp_mma_rr %158, %163, %153 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %165 = vector.extract %127[%141, 8] : f16 from vector<2x32xf16>
        %166 = vector.extract %127[%141, 9] : f16 from vector<2x32xf16>
        %167 = vector.extract %127[%141, 10] : f16 from vector<2x32xf16>
        %168 = vector.extract %127[%141, 11] : f16 from vector<2x32xf16>
        %169 = vector.from_elements %165, %166, %167, %168 : vector<1x4xf16>
        %170 = vector.extract %128[8, %147] : f16 from vector<32x2xf16>
        %171 = vector.extract %128[9, %147] : f16 from vector<32x2xf16>
        %172 = vector.extract %128[10, %147] : f16 from vector<32x2xf16>
        %173 = vector.extract %128[11, %147] : f16 from vector<32x2xf16>
        %174 = vector.from_elements %170, %171, %172, %173 : vector<4x1xf16>
        %175 = frisk.warp_mma_rr %169, %174, %164 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %176 = vector.extract %127[%141, 12] : f16 from vector<2x32xf16>
        %177 = vector.extract %127[%141, 13] : f16 from vector<2x32xf16>
        %178 = vector.extract %127[%141, 14] : f16 from vector<2x32xf16>
        %179 = vector.extract %127[%141, 15] : f16 from vector<2x32xf16>
        %180 = vector.from_elements %176, %177, %178, %179 : vector<1x4xf16>
        %181 = vector.extract %128[12, %147] : f16 from vector<32x2xf16>
        %182 = vector.extract %128[13, %147] : f16 from vector<32x2xf16>
        %183 = vector.extract %128[14, %147] : f16 from vector<32x2xf16>
        %184 = vector.extract %128[15, %147] : f16 from vector<32x2xf16>
        %185 = vector.from_elements %181, %182, %183, %184 : vector<4x1xf16>
        %186 = frisk.warp_mma_rr %180, %185, %175 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %187 = vector.extract %127[%141, 16] : f16 from vector<2x32xf16>
        %188 = vector.extract %127[%141, 17] : f16 from vector<2x32xf16>
        %189 = vector.extract %127[%141, 18] : f16 from vector<2x32xf16>
        %190 = vector.extract %127[%141, 19] : f16 from vector<2x32xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<1x4xf16>
        %192 = vector.extract %128[16, %147] : f16 from vector<32x2xf16>
        %193 = vector.extract %128[17, %147] : f16 from vector<32x2xf16>
        %194 = vector.extract %128[18, %147] : f16 from vector<32x2xf16>
        %195 = vector.extract %128[19, %147] : f16 from vector<32x2xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<4x1xf16>
        %197 = frisk.warp_mma_rr %191, %196, %186 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %198 = vector.extract %127[%141, 20] : f16 from vector<2x32xf16>
        %199 = vector.extract %127[%141, 21] : f16 from vector<2x32xf16>
        %200 = vector.extract %127[%141, 22] : f16 from vector<2x32xf16>
        %201 = vector.extract %127[%141, 23] : f16 from vector<2x32xf16>
        %202 = vector.from_elements %198, %199, %200, %201 : vector<1x4xf16>
        %203 = vector.extract %128[20, %147] : f16 from vector<32x2xf16>
        %204 = vector.extract %128[21, %147] : f16 from vector<32x2xf16>
        %205 = vector.extract %128[22, %147] : f16 from vector<32x2xf16>
        %206 = vector.extract %128[23, %147] : f16 from vector<32x2xf16>
        %207 = vector.from_elements %203, %204, %205, %206 : vector<4x1xf16>
        %208 = frisk.warp_mma_rr %202, %207, %197 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %209 = vector.extract %127[%141, 24] : f16 from vector<2x32xf16>
        %210 = vector.extract %127[%141, 25] : f16 from vector<2x32xf16>
        %211 = vector.extract %127[%141, 26] : f16 from vector<2x32xf16>
        %212 = vector.extract %127[%141, 27] : f16 from vector<2x32xf16>
        %213 = vector.from_elements %209, %210, %211, %212 : vector<1x4xf16>
        %214 = vector.extract %128[24, %147] : f16 from vector<32x2xf16>
        %215 = vector.extract %128[25, %147] : f16 from vector<32x2xf16>
        %216 = vector.extract %128[26, %147] : f16 from vector<32x2xf16>
        %217 = vector.extract %128[27, %147] : f16 from vector<32x2xf16>
        %218 = vector.from_elements %214, %215, %216, %217 : vector<4x1xf16>
        %219 = frisk.warp_mma_rr %213, %218, %208 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %220 = vector.extract %127[%141, 28] : f16 from vector<2x32xf16>
        %221 = vector.extract %127[%141, 29] : f16 from vector<2x32xf16>
        %222 = vector.extract %127[%141, 30] : f16 from vector<2x32xf16>
        %223 = vector.extract %127[%141, 31] : f16 from vector<2x32xf16>
        %224 = vector.from_elements %220, %221, %222, %223 : vector<1x4xf16>
        %225 = vector.extract %128[28, %147] : f16 from vector<32x2xf16>
        %226 = vector.extract %128[29, %147] : f16 from vector<32x2xf16>
        %227 = vector.extract %128[30, %147] : f16 from vector<32x2xf16>
        %228 = vector.extract %128[31, %147] : f16 from vector<32x2xf16>
        %229 = vector.from_elements %225, %226, %227, %228 : vector<4x1xf16>
        %230 = frisk.warp_mma_rr %224, %229, %219 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %232 = vector.extract %230[0, 0] : f32 from vector<1x4xf32>
        %233 = vector.insert %232, %arg12 [%141, %231] : f32 into vector<2x8xf32>
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %235 = vector.extract %230[0, 1] : f32 from vector<1x4xf32>
        %236 = vector.insert %235, %233 [%141, %234] : f32 into vector<2x8xf32>
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %238 = vector.extract %230[0, 2] : f32 from vector<1x4xf32>
        %239 = vector.insert %238, %236 [%141, %237] : f32 into vector<2x8xf32>
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %241 = vector.extract %230[0, 3] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %239 [%141, %240] : f32 into vector<2x8xf32>
        %243 = vector.extract %123#1[%141, 0] : f32 from vector<2x1xf32>
        %244 = vector.broadcast %243 : f32 to vector<1x4xf32>
        %245 = vector.insert %243, %arg13 [%141, %231] : f32 into vector<2x8xf32>
        %246 = vector.insert %243, %245 [%141, %234] : f32 into vector<2x8xf32>
        %247 = vector.insert %243, %246 [%141, %237] : f32 into vector<2x8xf32>
        %248 = vector.insert %243, %247 [%141, %240] : f32 into vector<2x8xf32>
        %249 = vector.extract %arg5[%141, %231] : f32 from vector<2x8xf32>
        %250 = vector.extract %arg5[%141, %234] : f32 from vector<2x8xf32>
        %251 = vector.extract %arg5[%141, %237] : f32 from vector<2x8xf32>
        %252 = vector.extract %arg5[%141, %240] : f32 from vector<2x8xf32>
        %253 = vector.from_elements %249, %250, %251, %252 : vector<1x4xf32>
        %254 = arith.mulf %253, %244 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %255 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %arg14 [%141, %231] : f32 into vector<2x8xf32>
        %257 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %256 [%141, %234] : f32 into vector<2x8xf32>
        %259 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %258 [%141, %237] : f32 into vector<2x8xf32>
        %261 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %260 [%141, %240] : f32 into vector<2x8xf32>
        %263 = arith.addf %254, %230 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %264 = vector.extract %263[0, 0] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %arg15 [%141, %231] : f32 into vector<2x8xf32>
        %266 = vector.extract %263[0, 1] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %265 [%141, %234] : f32 into vector<2x8xf32>
        %268 = vector.extract %263[0, 2] : f32 from vector<1x4xf32>
        %269 = vector.insert %268, %267 [%141, %237] : f32 into vector<2x8xf32>
        %270 = vector.extract %263[0, 3] : f32 from vector<1x4xf32>
        %271 = vector.insert %270, %269 [%141, %240] : f32 into vector<2x8xf32>
        affine.yield %242, %248, %262, %271 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %129 : !frisk.async_task
      %131 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %141 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %142 = frisk.read_matrix_from_shm %view[%141] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %143 = vector.bitcast %142 : vector<4xi32> to vector<8xf16>
        %144 = vector.extract %143[0] : f16 from vector<8xf16>
        %145 = vector.extract %143[%86] : f16 from vector<8xf16>
        %146 = vector.extract %143[1] : f16 from vector<8xf16>
        %147 = vector.extract %143[%87] : f16 from vector<8xf16>
        %148 = vector.extract %143[2] : f16 from vector<8xf16>
        %149 = vector.extract %143[%88] : f16 from vector<8xf16>
        %150 = vector.extract %143[3] : f16 from vector<8xf16>
        %151 = vector.extract %143[%89] : f16 from vector<8xf16>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %153 = vector.insert %144, %arg12 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %145, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %156 = vector.insert %146, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %147, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %159 = vector.insert %148, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %149, %159 [%158, 1] : f16 into vector<32x2xf16>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %162 = vector.insert %150, %160 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        affine.yield %163 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_19 = memref.subview %arg2[0, %84, %106, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      %132 = frisk.async_copy_gm_to_shm %subview_19 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %133:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7, %arg15 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %142 = vector.extract %127[%141, 0] : f16 from vector<2x32xf16>
        %143 = vector.extract %127[%141, 1] : f16 from vector<2x32xf16>
        %144 = vector.extract %127[%141, 2] : f16 from vector<2x32xf16>
        %145 = vector.extract %127[%141, 3] : f16 from vector<2x32xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %148 = vector.extract %131[0, %147] : f16 from vector<32x2xf16>
        %149 = vector.extract %131[1, %147] : f16 from vector<32x2xf16>
        %150 = vector.extract %131[2, %147] : f16 from vector<32x2xf16>
        %151 = vector.extract %131[3, %147] : f16 from vector<32x2xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<4x1xf16>
        %153 = frisk.warp_mma_rr %146, %152, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = vector.extract %127[%141, 4] : f16 from vector<2x32xf16>
        %155 = vector.extract %127[%141, 5] : f16 from vector<2x32xf16>
        %156 = vector.extract %127[%141, 6] : f16 from vector<2x32xf16>
        %157 = vector.extract %127[%141, 7] : f16 from vector<2x32xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = vector.extract %131[4, %147] : f16 from vector<32x2xf16>
        %160 = vector.extract %131[5, %147] : f16 from vector<32x2xf16>
        %161 = vector.extract %131[6, %147] : f16 from vector<32x2xf16>
        %162 = vector.extract %131[7, %147] : f16 from vector<32x2xf16>
        %163 = vector.from_elements %159, %160, %161, %162 : vector<4x1xf16>
        %164 = frisk.warp_mma_rr %158, %163, %153 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %165 = vector.extract %127[%141, 8] : f16 from vector<2x32xf16>
        %166 = vector.extract %127[%141, 9] : f16 from vector<2x32xf16>
        %167 = vector.extract %127[%141, 10] : f16 from vector<2x32xf16>
        %168 = vector.extract %127[%141, 11] : f16 from vector<2x32xf16>
        %169 = vector.from_elements %165, %166, %167, %168 : vector<1x4xf16>
        %170 = vector.extract %131[8, %147] : f16 from vector<32x2xf16>
        %171 = vector.extract %131[9, %147] : f16 from vector<32x2xf16>
        %172 = vector.extract %131[10, %147] : f16 from vector<32x2xf16>
        %173 = vector.extract %131[11, %147] : f16 from vector<32x2xf16>
        %174 = vector.from_elements %170, %171, %172, %173 : vector<4x1xf16>
        %175 = frisk.warp_mma_rr %169, %174, %164 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %176 = vector.extract %127[%141, 12] : f16 from vector<2x32xf16>
        %177 = vector.extract %127[%141, 13] : f16 from vector<2x32xf16>
        %178 = vector.extract %127[%141, 14] : f16 from vector<2x32xf16>
        %179 = vector.extract %127[%141, 15] : f16 from vector<2x32xf16>
        %180 = vector.from_elements %176, %177, %178, %179 : vector<1x4xf16>
        %181 = vector.extract %131[12, %147] : f16 from vector<32x2xf16>
        %182 = vector.extract %131[13, %147] : f16 from vector<32x2xf16>
        %183 = vector.extract %131[14, %147] : f16 from vector<32x2xf16>
        %184 = vector.extract %131[15, %147] : f16 from vector<32x2xf16>
        %185 = vector.from_elements %181, %182, %183, %184 : vector<4x1xf16>
        %186 = frisk.warp_mma_rr %180, %185, %175 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %187 = vector.extract %127[%141, 16] : f16 from vector<2x32xf16>
        %188 = vector.extract %127[%141, 17] : f16 from vector<2x32xf16>
        %189 = vector.extract %127[%141, 18] : f16 from vector<2x32xf16>
        %190 = vector.extract %127[%141, 19] : f16 from vector<2x32xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<1x4xf16>
        %192 = vector.extract %131[16, %147] : f16 from vector<32x2xf16>
        %193 = vector.extract %131[17, %147] : f16 from vector<32x2xf16>
        %194 = vector.extract %131[18, %147] : f16 from vector<32x2xf16>
        %195 = vector.extract %131[19, %147] : f16 from vector<32x2xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<4x1xf16>
        %197 = frisk.warp_mma_rr %191, %196, %186 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %198 = vector.extract %127[%141, 20] : f16 from vector<2x32xf16>
        %199 = vector.extract %127[%141, 21] : f16 from vector<2x32xf16>
        %200 = vector.extract %127[%141, 22] : f16 from vector<2x32xf16>
        %201 = vector.extract %127[%141, 23] : f16 from vector<2x32xf16>
        %202 = vector.from_elements %198, %199, %200, %201 : vector<1x4xf16>
        %203 = vector.extract %131[20, %147] : f16 from vector<32x2xf16>
        %204 = vector.extract %131[21, %147] : f16 from vector<32x2xf16>
        %205 = vector.extract %131[22, %147] : f16 from vector<32x2xf16>
        %206 = vector.extract %131[23, %147] : f16 from vector<32x2xf16>
        %207 = vector.from_elements %203, %204, %205, %206 : vector<4x1xf16>
        %208 = frisk.warp_mma_rr %202, %207, %197 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %209 = vector.extract %127[%141, 24] : f16 from vector<2x32xf16>
        %210 = vector.extract %127[%141, 25] : f16 from vector<2x32xf16>
        %211 = vector.extract %127[%141, 26] : f16 from vector<2x32xf16>
        %212 = vector.extract %127[%141, 27] : f16 from vector<2x32xf16>
        %213 = vector.from_elements %209, %210, %211, %212 : vector<1x4xf16>
        %214 = vector.extract %131[24, %147] : f16 from vector<32x2xf16>
        %215 = vector.extract %131[25, %147] : f16 from vector<32x2xf16>
        %216 = vector.extract %131[26, %147] : f16 from vector<32x2xf16>
        %217 = vector.extract %131[27, %147] : f16 from vector<32x2xf16>
        %218 = vector.from_elements %214, %215, %216, %217 : vector<4x1xf16>
        %219 = frisk.warp_mma_rr %213, %218, %208 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %220 = vector.extract %127[%141, 28] : f16 from vector<2x32xf16>
        %221 = vector.extract %127[%141, 29] : f16 from vector<2x32xf16>
        %222 = vector.extract %127[%141, 30] : f16 from vector<2x32xf16>
        %223 = vector.extract %127[%141, 31] : f16 from vector<2x32xf16>
        %224 = vector.from_elements %220, %221, %222, %223 : vector<1x4xf16>
        %225 = vector.extract %131[28, %147] : f16 from vector<32x2xf16>
        %226 = vector.extract %131[29, %147] : f16 from vector<32x2xf16>
        %227 = vector.extract %131[30, %147] : f16 from vector<32x2xf16>
        %228 = vector.extract %131[31, %147] : f16 from vector<32x2xf16>
        %229 = vector.from_elements %225, %226, %227, %228 : vector<4x1xf16>
        %230 = frisk.warp_mma_rr %224, %229, %219 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %232 = vector.extract %230[0, 0] : f32 from vector<1x4xf32>
        %233 = vector.insert %232, %arg12 [%141, %231] : f32 into vector<2x8xf32>
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %235 = vector.extract %230[0, 1] : f32 from vector<1x4xf32>
        %236 = vector.insert %235, %233 [%141, %234] : f32 into vector<2x8xf32>
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %238 = vector.extract %230[0, 2] : f32 from vector<1x4xf32>
        %239 = vector.insert %238, %236 [%141, %237] : f32 into vector<2x8xf32>
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %241 = vector.extract %230[0, 3] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %239 [%141, %240] : f32 into vector<2x8xf32>
        %243 = vector.extract %123#1[%141, 0] : f32 from vector<2x1xf32>
        %244 = vector.broadcast %243 : f32 to vector<1x4xf32>
        %245 = vector.insert %243, %arg13 [%141, %231] : f32 into vector<2x8xf32>
        %246 = vector.insert %243, %245 [%141, %234] : f32 into vector<2x8xf32>
        %247 = vector.insert %243, %246 [%141, %237] : f32 into vector<2x8xf32>
        %248 = vector.insert %243, %247 [%141, %240] : f32 into vector<2x8xf32>
        %249 = vector.extract %arg8[%141, %231] : f32 from vector<2x8xf32>
        %250 = vector.extract %arg8[%141, %234] : f32 from vector<2x8xf32>
        %251 = vector.extract %arg8[%141, %237] : f32 from vector<2x8xf32>
        %252 = vector.extract %arg8[%141, %240] : f32 from vector<2x8xf32>
        %253 = vector.from_elements %249, %250, %251, %252 : vector<1x4xf32>
        %254 = arith.mulf %253, %244 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %255 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %arg14 [%141, %231] : f32 into vector<2x8xf32>
        %257 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %256 [%141, %234] : f32 into vector<2x8xf32>
        %259 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %258 [%141, %237] : f32 into vector<2x8xf32>
        %261 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %260 [%141, %240] : f32 into vector<2x8xf32>
        %263 = arith.addf %254, %230 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %264 = vector.extract %263[0, 0] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %arg15 [%141, %231] : f32 into vector<2x8xf32>
        %266 = vector.extract %263[0, 1] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %265 [%141, %234] : f32 into vector<2x8xf32>
        %268 = vector.extract %263[0, 2] : f32 from vector<1x4xf32>
        %269 = vector.insert %268, %267 [%141, %237] : f32 into vector<2x8xf32>
        %270 = vector.extract %263[0, 3] : f32 from vector<1x4xf32>
        %271 = vector.insert %270, %269 [%141, %240] : f32 into vector<2x8xf32>
        affine.yield %242, %248, %262, %271 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %132 : !frisk.async_task
      %134 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %141 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %142 = frisk.read_matrix_from_shm %view[%141] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %143 = vector.bitcast %142 : vector<4xi32> to vector<8xf16>
        %144 = vector.extract %143[0] : f16 from vector<8xf16>
        %145 = vector.extract %143[%86] : f16 from vector<8xf16>
        %146 = vector.extract %143[1] : f16 from vector<8xf16>
        %147 = vector.extract %143[%87] : f16 from vector<8xf16>
        %148 = vector.extract %143[2] : f16 from vector<8xf16>
        %149 = vector.extract %143[%88] : f16 from vector<8xf16>
        %150 = vector.extract %143[3] : f16 from vector<8xf16>
        %151 = vector.extract %143[%89] : f16 from vector<8xf16>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %153 = vector.insert %144, %arg12 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %145, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %156 = vector.insert %146, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %147, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %159 = vector.insert %148, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %149, %159 [%158, 1] : f16 into vector<32x2xf16>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %162 = vector.insert %150, %160 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        affine.yield %163 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_20 = memref.subview %arg2[0, %84, %106, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      %135 = frisk.async_copy_gm_to_shm %subview_20 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %136:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7, %arg15 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %142 = vector.extract %127[%141, 0] : f16 from vector<2x32xf16>
        %143 = vector.extract %127[%141, 1] : f16 from vector<2x32xf16>
        %144 = vector.extract %127[%141, 2] : f16 from vector<2x32xf16>
        %145 = vector.extract %127[%141, 3] : f16 from vector<2x32xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %148 = vector.extract %134[0, %147] : f16 from vector<32x2xf16>
        %149 = vector.extract %134[1, %147] : f16 from vector<32x2xf16>
        %150 = vector.extract %134[2, %147] : f16 from vector<32x2xf16>
        %151 = vector.extract %134[3, %147] : f16 from vector<32x2xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<4x1xf16>
        %153 = frisk.warp_mma_rr %146, %152, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = vector.extract %127[%141, 4] : f16 from vector<2x32xf16>
        %155 = vector.extract %127[%141, 5] : f16 from vector<2x32xf16>
        %156 = vector.extract %127[%141, 6] : f16 from vector<2x32xf16>
        %157 = vector.extract %127[%141, 7] : f16 from vector<2x32xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = vector.extract %134[4, %147] : f16 from vector<32x2xf16>
        %160 = vector.extract %134[5, %147] : f16 from vector<32x2xf16>
        %161 = vector.extract %134[6, %147] : f16 from vector<32x2xf16>
        %162 = vector.extract %134[7, %147] : f16 from vector<32x2xf16>
        %163 = vector.from_elements %159, %160, %161, %162 : vector<4x1xf16>
        %164 = frisk.warp_mma_rr %158, %163, %153 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %165 = vector.extract %127[%141, 8] : f16 from vector<2x32xf16>
        %166 = vector.extract %127[%141, 9] : f16 from vector<2x32xf16>
        %167 = vector.extract %127[%141, 10] : f16 from vector<2x32xf16>
        %168 = vector.extract %127[%141, 11] : f16 from vector<2x32xf16>
        %169 = vector.from_elements %165, %166, %167, %168 : vector<1x4xf16>
        %170 = vector.extract %134[8, %147] : f16 from vector<32x2xf16>
        %171 = vector.extract %134[9, %147] : f16 from vector<32x2xf16>
        %172 = vector.extract %134[10, %147] : f16 from vector<32x2xf16>
        %173 = vector.extract %134[11, %147] : f16 from vector<32x2xf16>
        %174 = vector.from_elements %170, %171, %172, %173 : vector<4x1xf16>
        %175 = frisk.warp_mma_rr %169, %174, %164 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %176 = vector.extract %127[%141, 12] : f16 from vector<2x32xf16>
        %177 = vector.extract %127[%141, 13] : f16 from vector<2x32xf16>
        %178 = vector.extract %127[%141, 14] : f16 from vector<2x32xf16>
        %179 = vector.extract %127[%141, 15] : f16 from vector<2x32xf16>
        %180 = vector.from_elements %176, %177, %178, %179 : vector<1x4xf16>
        %181 = vector.extract %134[12, %147] : f16 from vector<32x2xf16>
        %182 = vector.extract %134[13, %147] : f16 from vector<32x2xf16>
        %183 = vector.extract %134[14, %147] : f16 from vector<32x2xf16>
        %184 = vector.extract %134[15, %147] : f16 from vector<32x2xf16>
        %185 = vector.from_elements %181, %182, %183, %184 : vector<4x1xf16>
        %186 = frisk.warp_mma_rr %180, %185, %175 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %187 = vector.extract %127[%141, 16] : f16 from vector<2x32xf16>
        %188 = vector.extract %127[%141, 17] : f16 from vector<2x32xf16>
        %189 = vector.extract %127[%141, 18] : f16 from vector<2x32xf16>
        %190 = vector.extract %127[%141, 19] : f16 from vector<2x32xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<1x4xf16>
        %192 = vector.extract %134[16, %147] : f16 from vector<32x2xf16>
        %193 = vector.extract %134[17, %147] : f16 from vector<32x2xf16>
        %194 = vector.extract %134[18, %147] : f16 from vector<32x2xf16>
        %195 = vector.extract %134[19, %147] : f16 from vector<32x2xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<4x1xf16>
        %197 = frisk.warp_mma_rr %191, %196, %186 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %198 = vector.extract %127[%141, 20] : f16 from vector<2x32xf16>
        %199 = vector.extract %127[%141, 21] : f16 from vector<2x32xf16>
        %200 = vector.extract %127[%141, 22] : f16 from vector<2x32xf16>
        %201 = vector.extract %127[%141, 23] : f16 from vector<2x32xf16>
        %202 = vector.from_elements %198, %199, %200, %201 : vector<1x4xf16>
        %203 = vector.extract %134[20, %147] : f16 from vector<32x2xf16>
        %204 = vector.extract %134[21, %147] : f16 from vector<32x2xf16>
        %205 = vector.extract %134[22, %147] : f16 from vector<32x2xf16>
        %206 = vector.extract %134[23, %147] : f16 from vector<32x2xf16>
        %207 = vector.from_elements %203, %204, %205, %206 : vector<4x1xf16>
        %208 = frisk.warp_mma_rr %202, %207, %197 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %209 = vector.extract %127[%141, 24] : f16 from vector<2x32xf16>
        %210 = vector.extract %127[%141, 25] : f16 from vector<2x32xf16>
        %211 = vector.extract %127[%141, 26] : f16 from vector<2x32xf16>
        %212 = vector.extract %127[%141, 27] : f16 from vector<2x32xf16>
        %213 = vector.from_elements %209, %210, %211, %212 : vector<1x4xf16>
        %214 = vector.extract %134[24, %147] : f16 from vector<32x2xf16>
        %215 = vector.extract %134[25, %147] : f16 from vector<32x2xf16>
        %216 = vector.extract %134[26, %147] : f16 from vector<32x2xf16>
        %217 = vector.extract %134[27, %147] : f16 from vector<32x2xf16>
        %218 = vector.from_elements %214, %215, %216, %217 : vector<4x1xf16>
        %219 = frisk.warp_mma_rr %213, %218, %208 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %220 = vector.extract %127[%141, 28] : f16 from vector<2x32xf16>
        %221 = vector.extract %127[%141, 29] : f16 from vector<2x32xf16>
        %222 = vector.extract %127[%141, 30] : f16 from vector<2x32xf16>
        %223 = vector.extract %127[%141, 31] : f16 from vector<2x32xf16>
        %224 = vector.from_elements %220, %221, %222, %223 : vector<1x4xf16>
        %225 = vector.extract %134[28, %147] : f16 from vector<32x2xf16>
        %226 = vector.extract %134[29, %147] : f16 from vector<32x2xf16>
        %227 = vector.extract %134[30, %147] : f16 from vector<32x2xf16>
        %228 = vector.extract %134[31, %147] : f16 from vector<32x2xf16>
        %229 = vector.from_elements %225, %226, %227, %228 : vector<4x1xf16>
        %230 = frisk.warp_mma_rr %224, %229, %219 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %232 = vector.extract %230[0, 0] : f32 from vector<1x4xf32>
        %233 = vector.insert %232, %arg12 [%141, %231] : f32 into vector<2x8xf32>
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %235 = vector.extract %230[0, 1] : f32 from vector<1x4xf32>
        %236 = vector.insert %235, %233 [%141, %234] : f32 into vector<2x8xf32>
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %238 = vector.extract %230[0, 2] : f32 from vector<1x4xf32>
        %239 = vector.insert %238, %236 [%141, %237] : f32 into vector<2x8xf32>
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %241 = vector.extract %230[0, 3] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %239 [%141, %240] : f32 into vector<2x8xf32>
        %243 = vector.extract %123#1[%141, 0] : f32 from vector<2x1xf32>
        %244 = vector.broadcast %243 : f32 to vector<1x4xf32>
        %245 = vector.insert %243, %arg13 [%141, %231] : f32 into vector<2x8xf32>
        %246 = vector.insert %243, %245 [%141, %234] : f32 into vector<2x8xf32>
        %247 = vector.insert %243, %246 [%141, %237] : f32 into vector<2x8xf32>
        %248 = vector.insert %243, %247 [%141, %240] : f32 into vector<2x8xf32>
        %249 = vector.extract %arg9[%141, %231] : f32 from vector<2x8xf32>
        %250 = vector.extract %arg9[%141, %234] : f32 from vector<2x8xf32>
        %251 = vector.extract %arg9[%141, %237] : f32 from vector<2x8xf32>
        %252 = vector.extract %arg9[%141, %240] : f32 from vector<2x8xf32>
        %253 = vector.from_elements %249, %250, %251, %252 : vector<1x4xf32>
        %254 = arith.mulf %253, %244 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %255 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %arg14 [%141, %231] : f32 into vector<2x8xf32>
        %257 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %256 [%141, %234] : f32 into vector<2x8xf32>
        %259 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %258 [%141, %237] : f32 into vector<2x8xf32>
        %261 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %260 [%141, %240] : f32 into vector<2x8xf32>
        %263 = arith.addf %254, %230 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %264 = vector.extract %263[0, 0] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %arg15 [%141, %231] : f32 into vector<2x8xf32>
        %266 = vector.extract %263[0, 1] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %265 [%141, %234] : f32 into vector<2x8xf32>
        %268 = vector.extract %263[0, 2] : f32 from vector<1x4xf32>
        %269 = vector.insert %268, %267 [%141, %237] : f32 into vector<2x8xf32>
        %270 = vector.extract %263[0, 3] : f32 from vector<1x4xf32>
        %271 = vector.insert %270, %269 [%141, %240] : f32 into vector<2x8xf32>
        affine.yield %242, %248, %262, %271 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %135 : !frisk.async_task
      %137 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %141 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %142 = frisk.read_matrix_from_shm %view[%141] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %143 = vector.bitcast %142 : vector<4xi32> to vector<8xf16>
        %144 = vector.extract %143[0] : f16 from vector<8xf16>
        %145 = vector.extract %143[%86] : f16 from vector<8xf16>
        %146 = vector.extract %143[1] : f16 from vector<8xf16>
        %147 = vector.extract %143[%87] : f16 from vector<8xf16>
        %148 = vector.extract %143[2] : f16 from vector<8xf16>
        %149 = vector.extract %143[%88] : f16 from vector<8xf16>
        %150 = vector.extract %143[3] : f16 from vector<8xf16>
        %151 = vector.extract %143[%89] : f16 from vector<8xf16>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %153 = vector.insert %144, %arg12 [%152, 0] : f16 into vector<32x2xf16>
        %154 = vector.insert %145, %153 [%152, 1] : f16 into vector<32x2xf16>
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %156 = vector.insert %146, %154 [%155, 0] : f16 into vector<32x2xf16>
        %157 = vector.insert %147, %156 [%155, 1] : f16 into vector<32x2xf16>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %159 = vector.insert %148, %157 [%158, 0] : f16 into vector<32x2xf16>
        %160 = vector.insert %149, %159 [%158, 1] : f16 into vector<32x2xf16>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %162 = vector.insert %150, %160 [%161, 0] : f16 into vector<32x2xf16>
        %163 = vector.insert %151, %162 [%161, 1] : f16 into vector<32x2xf16>
        affine.yield %163 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %138:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7, %arg15 = %cst_7, %arg16 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %142 = vector.extract %127[%141, 0] : f16 from vector<2x32xf16>
        %143 = vector.extract %127[%141, 1] : f16 from vector<2x32xf16>
        %144 = vector.extract %127[%141, 2] : f16 from vector<2x32xf16>
        %145 = vector.extract %127[%141, 3] : f16 from vector<2x32xf16>
        %146 = vector.from_elements %142, %143, %144, %145 : vector<1x4xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %148 = vector.extract %137[0, %147] : f16 from vector<32x2xf16>
        %149 = vector.extract %137[1, %147] : f16 from vector<32x2xf16>
        %150 = vector.extract %137[2, %147] : f16 from vector<32x2xf16>
        %151 = vector.extract %137[3, %147] : f16 from vector<32x2xf16>
        %152 = vector.from_elements %148, %149, %150, %151 : vector<4x1xf16>
        %153 = frisk.warp_mma_rr %146, %152, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %154 = vector.extract %127[%141, 4] : f16 from vector<2x32xf16>
        %155 = vector.extract %127[%141, 5] : f16 from vector<2x32xf16>
        %156 = vector.extract %127[%141, 6] : f16 from vector<2x32xf16>
        %157 = vector.extract %127[%141, 7] : f16 from vector<2x32xf16>
        %158 = vector.from_elements %154, %155, %156, %157 : vector<1x4xf16>
        %159 = vector.extract %137[4, %147] : f16 from vector<32x2xf16>
        %160 = vector.extract %137[5, %147] : f16 from vector<32x2xf16>
        %161 = vector.extract %137[6, %147] : f16 from vector<32x2xf16>
        %162 = vector.extract %137[7, %147] : f16 from vector<32x2xf16>
        %163 = vector.from_elements %159, %160, %161, %162 : vector<4x1xf16>
        %164 = frisk.warp_mma_rr %158, %163, %153 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %165 = vector.extract %127[%141, 8] : f16 from vector<2x32xf16>
        %166 = vector.extract %127[%141, 9] : f16 from vector<2x32xf16>
        %167 = vector.extract %127[%141, 10] : f16 from vector<2x32xf16>
        %168 = vector.extract %127[%141, 11] : f16 from vector<2x32xf16>
        %169 = vector.from_elements %165, %166, %167, %168 : vector<1x4xf16>
        %170 = vector.extract %137[8, %147] : f16 from vector<32x2xf16>
        %171 = vector.extract %137[9, %147] : f16 from vector<32x2xf16>
        %172 = vector.extract %137[10, %147] : f16 from vector<32x2xf16>
        %173 = vector.extract %137[11, %147] : f16 from vector<32x2xf16>
        %174 = vector.from_elements %170, %171, %172, %173 : vector<4x1xf16>
        %175 = frisk.warp_mma_rr %169, %174, %164 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %176 = vector.extract %127[%141, 12] : f16 from vector<2x32xf16>
        %177 = vector.extract %127[%141, 13] : f16 from vector<2x32xf16>
        %178 = vector.extract %127[%141, 14] : f16 from vector<2x32xf16>
        %179 = vector.extract %127[%141, 15] : f16 from vector<2x32xf16>
        %180 = vector.from_elements %176, %177, %178, %179 : vector<1x4xf16>
        %181 = vector.extract %137[12, %147] : f16 from vector<32x2xf16>
        %182 = vector.extract %137[13, %147] : f16 from vector<32x2xf16>
        %183 = vector.extract %137[14, %147] : f16 from vector<32x2xf16>
        %184 = vector.extract %137[15, %147] : f16 from vector<32x2xf16>
        %185 = vector.from_elements %181, %182, %183, %184 : vector<4x1xf16>
        %186 = frisk.warp_mma_rr %180, %185, %175 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %187 = vector.extract %127[%141, 16] : f16 from vector<2x32xf16>
        %188 = vector.extract %127[%141, 17] : f16 from vector<2x32xf16>
        %189 = vector.extract %127[%141, 18] : f16 from vector<2x32xf16>
        %190 = vector.extract %127[%141, 19] : f16 from vector<2x32xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<1x4xf16>
        %192 = vector.extract %137[16, %147] : f16 from vector<32x2xf16>
        %193 = vector.extract %137[17, %147] : f16 from vector<32x2xf16>
        %194 = vector.extract %137[18, %147] : f16 from vector<32x2xf16>
        %195 = vector.extract %137[19, %147] : f16 from vector<32x2xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<4x1xf16>
        %197 = frisk.warp_mma_rr %191, %196, %186 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %198 = vector.extract %127[%141, 20] : f16 from vector<2x32xf16>
        %199 = vector.extract %127[%141, 21] : f16 from vector<2x32xf16>
        %200 = vector.extract %127[%141, 22] : f16 from vector<2x32xf16>
        %201 = vector.extract %127[%141, 23] : f16 from vector<2x32xf16>
        %202 = vector.from_elements %198, %199, %200, %201 : vector<1x4xf16>
        %203 = vector.extract %137[20, %147] : f16 from vector<32x2xf16>
        %204 = vector.extract %137[21, %147] : f16 from vector<32x2xf16>
        %205 = vector.extract %137[22, %147] : f16 from vector<32x2xf16>
        %206 = vector.extract %137[23, %147] : f16 from vector<32x2xf16>
        %207 = vector.from_elements %203, %204, %205, %206 : vector<4x1xf16>
        %208 = frisk.warp_mma_rr %202, %207, %197 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %209 = vector.extract %127[%141, 24] : f16 from vector<2x32xf16>
        %210 = vector.extract %127[%141, 25] : f16 from vector<2x32xf16>
        %211 = vector.extract %127[%141, 26] : f16 from vector<2x32xf16>
        %212 = vector.extract %127[%141, 27] : f16 from vector<2x32xf16>
        %213 = vector.from_elements %209, %210, %211, %212 : vector<1x4xf16>
        %214 = vector.extract %137[24, %147] : f16 from vector<32x2xf16>
        %215 = vector.extract %137[25, %147] : f16 from vector<32x2xf16>
        %216 = vector.extract %137[26, %147] : f16 from vector<32x2xf16>
        %217 = vector.extract %137[27, %147] : f16 from vector<32x2xf16>
        %218 = vector.from_elements %214, %215, %216, %217 : vector<4x1xf16>
        %219 = frisk.warp_mma_rr %213, %218, %208 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %220 = vector.extract %127[%141, 28] : f16 from vector<2x32xf16>
        %221 = vector.extract %127[%141, 29] : f16 from vector<2x32xf16>
        %222 = vector.extract %127[%141, 30] : f16 from vector<2x32xf16>
        %223 = vector.extract %127[%141, 31] : f16 from vector<2x32xf16>
        %224 = vector.from_elements %220, %221, %222, %223 : vector<1x4xf16>
        %225 = vector.extract %137[28, %147] : f16 from vector<32x2xf16>
        %226 = vector.extract %137[29, %147] : f16 from vector<32x2xf16>
        %227 = vector.extract %137[30, %147] : f16 from vector<32x2xf16>
        %228 = vector.extract %137[31, %147] : f16 from vector<32x2xf16>
        %229 = vector.from_elements %225, %226, %227, %228 : vector<4x1xf16>
        %230 = frisk.warp_mma_rr %224, %229, %219 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %232 = vector.extract %230[0, 0] : f32 from vector<1x4xf32>
        %233 = vector.insert %232, %arg12 [%141, %231] : f32 into vector<2x8xf32>
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %235 = vector.extract %230[0, 1] : f32 from vector<1x4xf32>
        %236 = vector.insert %235, %233 [%141, %234] : f32 into vector<2x8xf32>
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %238 = vector.extract %230[0, 2] : f32 from vector<1x4xf32>
        %239 = vector.insert %238, %236 [%141, %237] : f32 into vector<2x8xf32>
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %241 = vector.extract %230[0, 3] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %239 [%141, %240] : f32 into vector<2x8xf32>
        %243 = vector.extract %123#1[%141, 0] : f32 from vector<2x1xf32>
        %244 = vector.broadcast %243 : f32 to vector<1x4xf32>
        %245 = vector.insert %243, %arg13 [%141, %231] : f32 into vector<2x8xf32>
        %246 = vector.insert %243, %245 [%141, %234] : f32 into vector<2x8xf32>
        %247 = vector.insert %243, %246 [%141, %237] : f32 into vector<2x8xf32>
        %248 = vector.insert %243, %247 [%141, %240] : f32 into vector<2x8xf32>
        %249 = vector.extract %arg10[%141, %231] : f32 from vector<2x8xf32>
        %250 = vector.extract %arg10[%141, %234] : f32 from vector<2x8xf32>
        %251 = vector.extract %arg10[%141, %237] : f32 from vector<2x8xf32>
        %252 = vector.extract %arg10[%141, %240] : f32 from vector<2x8xf32>
        %253 = vector.from_elements %249, %250, %251, %252 : vector<1x4xf32>
        %254 = arith.mulf %253, %244 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %255 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %arg14 [%141, %231] : f32 into vector<2x8xf32>
        %257 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %256 [%141, %234] : f32 into vector<2x8xf32>
        %259 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %258 [%141, %237] : f32 into vector<2x8xf32>
        %261 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %260 [%141, %240] : f32 into vector<2x8xf32>
        %263 = arith.addf %254, %230 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %264 = vector.extract %263[0, 0] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %arg15 [%141, %231] : f32 into vector<2x8xf32>
        %266 = vector.extract %263[0, 1] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %265 [%141, %234] : f32 into vector<2x8xf32>
        %268 = vector.extract %263[0, 2] : f32 from vector<1x4xf32>
        %269 = vector.insert %268, %267 [%141, %237] : f32 into vector<2x8xf32>
        %270 = vector.extract %263[0, 3] : f32 from vector<1x4xf32>
        %271 = vector.insert %270, %269 [%141, %240] : f32 into vector<2x8xf32>
        %272 = vector.extract %130#3[%141, %231] : f32 from vector<2x8xf32>
        %273 = vector.extract %130#3[%141, %234] : f32 from vector<2x8xf32>
        %274 = vector.extract %130#3[%141, %237] : f32 from vector<2x8xf32>
        %275 = vector.extract %130#3[%141, %240] : f32 from vector<2x8xf32>
        %276 = vector.insert %272, %arg16 [%141, %231] : f32 into vector<2x8xf32>
        %277 = vector.insert %273, %276 [%141, %234] : f32 into vector<2x8xf32>
        %278 = vector.insert %274, %277 [%141, %237] : f32 into vector<2x8xf32>
        %279 = vector.insert %275, %278 [%141, %240] : f32 into vector<2x8xf32>
        affine.yield %242, %248, %262, %271, %279 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %139:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %141 = vector.extract %126[%arg11, 0] : f32 from vector<2x1xf32>
        %142 = vector.insert %141, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %143 = vector.extract %121[%arg11, 0] : f32 from vector<2x1xf32>
        %144 = vector.insert %143, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %142, %144 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %140:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %141 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %143 = vector.extract %133#3[%141, %142] : f32 from vector<2x8xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %145 = vector.extract %133#3[%141, %144] : f32 from vector<2x8xf32>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %147 = vector.extract %133#3[%141, %146] : f32 from vector<2x8xf32>
        %148 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %149 = vector.extract %133#3[%141, %148] : f32 from vector<2x8xf32>
        %150 = vector.insert %143, %arg12 [%141, %142] : f32 into vector<2x8xf32>
        %151 = vector.insert %145, %150 [%141, %144] : f32 into vector<2x8xf32>
        %152 = vector.insert %147, %151 [%141, %146] : f32 into vector<2x8xf32>
        %153 = vector.insert %149, %152 [%141, %148] : f32 into vector<2x8xf32>
        %154 = vector.extract %136#3[%141, %142] : f32 from vector<2x8xf32>
        %155 = vector.extract %136#3[%141, %144] : f32 from vector<2x8xf32>
        %156 = vector.extract %136#3[%141, %146] : f32 from vector<2x8xf32>
        %157 = vector.extract %136#3[%141, %148] : f32 from vector<2x8xf32>
        %158 = vector.insert %154, %arg13 [%141, %142] : f32 into vector<2x8xf32>
        %159 = vector.insert %155, %158 [%141, %144] : f32 into vector<2x8xf32>
        %160 = vector.insert %156, %159 [%141, %146] : f32 into vector<2x8xf32>
        %161 = vector.insert %157, %160 [%141, %148] : f32 into vector<2x8xf32>
        %162 = vector.extract %138#3[%141, %142] : f32 from vector<2x8xf32>
        %163 = vector.extract %138#3[%141, %144] : f32 from vector<2x8xf32>
        %164 = vector.extract %138#3[%141, %146] : f32 from vector<2x8xf32>
        %165 = vector.extract %138#3[%141, %148] : f32 from vector<2x8xf32>
        %166 = vector.insert %162, %arg14 [%141, %142] : f32 into vector<2x8xf32>
        %167 = vector.insert %163, %166 [%141, %144] : f32 into vector<2x8xf32>
        %168 = vector.insert %164, %167 [%141, %146] : f32 into vector<2x8xf32>
        %169 = vector.insert %165, %168 [%141, %148] : f32 into vector<2x8xf32>
        affine.yield %153, %161, %169 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      gpu.barrier
      affine.yield %138#4, %139#0, %139#1, %140#0, %140#1, %140#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %91 = arith.index_cast %thread_id_x : index to i32
    %92 = arith.andi %91, %c16_i32 : i32
    %93 = arith.cmpi ne, %92, %c0_i32 : i32
    %94 = arith.andi %91, %c32_i32 : i32
    %95 = arith.cmpi ne, %94, %c0_i32 : i32
    %96 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = vector.extract %90#5[%107, %106] : f32 from vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %110 = vector.extract %90#5[%107, %109] : f32 from vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %112 = vector.extract %90#5[%107, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %90#5[%107, %113] : f32 from vector<2x8xf32>
      %115 = arith.select %93, %108, %110 : f32
      %shuffleResult, %valid = gpu.shuffle xor %115, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %93, %shuffleResult, %108 : f32
      %117 = arith.select %93, %110, %shuffleResult : f32
      %118 = arith.select %93, %112, %114 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %118, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %119 = arith.select %93, %shuffleResult_13, %112 : f32
      %120 = arith.select %93, %114, %shuffleResult_13 : f32
      %121 = arith.select %95, %116, %119 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %121, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %122 = arith.select %95, %shuffleResult_15, %116 : f32
      %123 = arith.select %95, %119, %shuffleResult_15 : f32
      %124 = arith.select %95, %117, %120 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %124, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %125 = arith.select %95, %shuffleResult_17, %117 : f32
      %126 = arith.select %95, %120, %shuffleResult_17 : f32
      %127 = vector.insert %122, %arg5 [%107, %106] : f32 into vector<2x8xf32>
      %128 = vector.insert %125, %127 [%107, %109] : f32 into vector<2x8xf32>
      %129 = vector.insert %123, %128 [%107, %111] : f32 into vector<2x8xf32>
      %130 = vector.insert %126, %129 [%107, %113] : f32 into vector<2x8xf32>
      affine.yield %130 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %97 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = vector.extract %90#4[%107, %106] : f32 from vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %110 = vector.extract %90#4[%107, %109] : f32 from vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %112 = vector.extract %90#4[%107, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %90#4[%107, %113] : f32 from vector<2x8xf32>
      %115 = arith.select %93, %108, %110 : f32
      %shuffleResult, %valid = gpu.shuffle xor %115, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %93, %shuffleResult, %108 : f32
      %117 = arith.select %93, %110, %shuffleResult : f32
      %118 = arith.select %93, %112, %114 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %118, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %119 = arith.select %93, %shuffleResult_13, %112 : f32
      %120 = arith.select %93, %114, %shuffleResult_13 : f32
      %121 = arith.select %95, %116, %119 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %121, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %122 = arith.select %95, %shuffleResult_15, %116 : f32
      %123 = arith.select %95, %119, %shuffleResult_15 : f32
      %124 = arith.select %95, %117, %120 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %124, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %125 = arith.select %95, %shuffleResult_17, %117 : f32
      %126 = arith.select %95, %120, %shuffleResult_17 : f32
      %127 = vector.insert %122, %arg5 [%107, %106] : f32 into vector<2x8xf32>
      %128 = vector.insert %125, %127 [%107, %109] : f32 into vector<2x8xf32>
      %129 = vector.insert %123, %128 [%107, %111] : f32 into vector<2x8xf32>
      %130 = vector.insert %126, %129 [%107, %113] : f32 into vector<2x8xf32>
      affine.yield %130 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %98 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = vector.extract %90#3[%107, %106] : f32 from vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %110 = vector.extract %90#3[%107, %109] : f32 from vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %112 = vector.extract %90#3[%107, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %90#3[%107, %113] : f32 from vector<2x8xf32>
      %115 = arith.select %93, %108, %110 : f32
      %shuffleResult, %valid = gpu.shuffle xor %115, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %93, %shuffleResult, %108 : f32
      %117 = arith.select %93, %110, %shuffleResult : f32
      %118 = arith.select %93, %112, %114 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %118, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %119 = arith.select %93, %shuffleResult_13, %112 : f32
      %120 = arith.select %93, %114, %shuffleResult_13 : f32
      %121 = arith.select %95, %116, %119 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %121, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %122 = arith.select %95, %shuffleResult_15, %116 : f32
      %123 = arith.select %95, %119, %shuffleResult_15 : f32
      %124 = arith.select %95, %117, %120 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %124, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %125 = arith.select %95, %shuffleResult_17, %117 : f32
      %126 = arith.select %95, %120, %shuffleResult_17 : f32
      %127 = vector.insert %122, %arg5 [%107, %106] : f32 into vector<2x8xf32>
      %128 = vector.insert %125, %127 [%107, %109] : f32 into vector<2x8xf32>
      %129 = vector.insert %123, %128 [%107, %111] : f32 into vector<2x8xf32>
      %130 = vector.insert %126, %129 [%107, %113] : f32 into vector<2x8xf32>
      affine.yield %130 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %99 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7) -> (vector<2x8xf32>) {
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = vector.extract %90#0[%107, %106] : f32 from vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %110 = vector.extract %90#0[%107, %109] : f32 from vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %112 = vector.extract %90#0[%107, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %90#0[%107, %113] : f32 from vector<2x8xf32>
      %115 = arith.select %93, %108, %110 : f32
      %shuffleResult, %valid = gpu.shuffle xor %115, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %93, %shuffleResult, %108 : f32
      %117 = arith.select %93, %110, %shuffleResult : f32
      %118 = arith.select %93, %112, %114 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %118, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %119 = arith.select %93, %shuffleResult_13, %112 : f32
      %120 = arith.select %93, %114, %shuffleResult_13 : f32
      %121 = arith.select %95, %116, %119 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %121, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %122 = arith.select %95, %shuffleResult_15, %116 : f32
      %123 = arith.select %95, %119, %shuffleResult_15 : f32
      %124 = arith.select %95, %117, %120 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %124, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %125 = arith.select %95, %shuffleResult_17, %117 : f32
      %126 = arith.select %95, %120, %shuffleResult_17 : f32
      %127 = vector.insert %122, %arg5 [%107, %106] : f32 into vector<2x8xf32>
      %128 = vector.insert %125, %127 [%107, %109] : f32 into vector<2x8xf32>
      %129 = vector.insert %123, %128 [%107, %111] : f32 into vector<2x8xf32>
      %130 = vector.insert %126, %129 [%107, %113] : f32 into vector<2x8xf32>
      affine.yield %130 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %100:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7, %arg6 = %cst_7, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %106 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %107 = vector.extract %90#1[%106, 0] : f32 from vector<2x1xf32>
      %108 = vector.broadcast %107 : f32 to vector<1x4xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %110 = vector.insert %107, %arg5 [%106, %109] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.insert %107, %110 [%106, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %114 = vector.insert %107, %112 [%106, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %116 = vector.insert %107, %114 [%106, %115] : f32 into vector<2x8xf32>
      %117 = vector.extract %99[%106, %109] : f32 from vector<2x8xf32>
      %118 = vector.extract %99[%106, %111] : f32 from vector<2x8xf32>
      %119 = vector.extract %99[%106, %113] : f32 from vector<2x8xf32>
      %120 = vector.extract %99[%106, %115] : f32 from vector<2x8xf32>
      %121 = vector.from_elements %117, %118, %119, %120 : vector<1x4xf32>
      %122 = arith.divf %121, %108 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %123 = vector.extract %122[0, 0] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %arg6 [%106, %109] : f32 into vector<2x8xf32>
      %125 = vector.extract %122[0, 1] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%106, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %122[0, 2] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%106, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %122[0, 3] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%106, %115] : f32 into vector<2x8xf32>
      %131 = arith.truncf %122 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %132 = vector.extract %131[0, 0] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %arg7 [%106, %109] : f16 into vector<2x8xf16>
      %134 = vector.extract %131[0, 1] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%106, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %131[0, 2] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%106, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %131[0, 3] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%106, %115] : f16 into vector<2x8xf16>
      %140 = vector.insert %132, %arg8 [%106, %109] : f16 into vector<2x8xf16>
      %141 = vector.insert %134, %140 [%106, %111] : f16 into vector<2x8xf16>
      %142 = vector.insert %136, %141 [%106, %113] : f16 into vector<2x8xf16>
      %143 = vector.insert %138, %142 [%106, %115] : f16 into vector<2x8xf16>
      affine.yield %116, %130, %139, %143 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %101 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %102 = arith.cmpi ult, %101, %c4 : index
    scf.if %102 {
      affine.for %arg4 = 0 to 4 {
        %106 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %108 = vector.extract %100#3[%106, %107] : f16 from vector<2x8xf16>
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %110 = vector.extract %100#3[%106, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %112 = vector.extract %100#3[%106, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %114 = vector.extract %100#3[%106, %113] : f16 from vector<2x8xf16>
        affine.store %108, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %110, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %103:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7, %arg6 = %cst_7, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %106 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %107 = vector.extract %90#1[%106, 0] : f32 from vector<2x1xf32>
      %108 = vector.broadcast %107 : f32 to vector<1x4xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %110 = vector.insert %107, %arg5 [%106, %109] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.insert %107, %110 [%106, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %114 = vector.insert %107, %112 [%106, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %116 = vector.insert %107, %114 [%106, %115] : f32 into vector<2x8xf32>
      %117 = vector.extract %98[%106, %109] : f32 from vector<2x8xf32>
      %118 = vector.extract %98[%106, %111] : f32 from vector<2x8xf32>
      %119 = vector.extract %98[%106, %113] : f32 from vector<2x8xf32>
      %120 = vector.extract %98[%106, %115] : f32 from vector<2x8xf32>
      %121 = vector.from_elements %117, %118, %119, %120 : vector<1x4xf32>
      %122 = arith.divf %121, %108 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %123 = vector.extract %122[0, 0] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %arg6 [%106, %109] : f32 into vector<2x8xf32>
      %125 = vector.extract %122[0, 1] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%106, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %122[0, 2] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%106, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %122[0, 3] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%106, %115] : f32 into vector<2x8xf32>
      %131 = arith.truncf %122 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %132 = vector.extract %131[0, 0] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %arg7 [%106, %109] : f16 into vector<2x8xf16>
      %134 = vector.extract %131[0, 1] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%106, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %131[0, 2] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%106, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %131[0, 3] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%106, %115] : f16 into vector<2x8xf16>
      %140 = vector.insert %132, %arg8 [%106, %109] : f16 into vector<2x8xf16>
      %141 = vector.insert %134, %140 [%106, %111] : f16 into vector<2x8xf16>
      %142 = vector.insert %136, %141 [%106, %113] : f16 into vector<2x8xf16>
      %143 = vector.insert %138, %142 [%106, %115] : f16 into vector<2x8xf16>
      affine.yield %116, %130, %139, %143 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %102 {
      affine.for %arg4 = 0 to 4 {
        %106 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %108 = vector.extract %103#3[%106, %107] : f16 from vector<2x8xf16>
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %110 = vector.extract %103#3[%106, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %112 = vector.extract %103#3[%106, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %114 = vector.extract %103#3[%106, %113] : f16 from vector<2x8xf16>
        affine.store %108, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %110, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %104:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7, %arg6 = %cst_7, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %106 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %107 = vector.extract %90#1[%106, 0] : f32 from vector<2x1xf32>
      %108 = vector.broadcast %107 : f32 to vector<1x4xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %110 = vector.insert %107, %arg5 [%106, %109] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.insert %107, %110 [%106, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %114 = vector.insert %107, %112 [%106, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %116 = vector.insert %107, %114 [%106, %115] : f32 into vector<2x8xf32>
      %117 = vector.extract %97[%106, %109] : f32 from vector<2x8xf32>
      %118 = vector.extract %97[%106, %111] : f32 from vector<2x8xf32>
      %119 = vector.extract %97[%106, %113] : f32 from vector<2x8xf32>
      %120 = vector.extract %97[%106, %115] : f32 from vector<2x8xf32>
      %121 = vector.from_elements %117, %118, %119, %120 : vector<1x4xf32>
      %122 = arith.divf %121, %108 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %123 = vector.extract %122[0, 0] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %arg6 [%106, %109] : f32 into vector<2x8xf32>
      %125 = vector.extract %122[0, 1] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%106, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %122[0, 2] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%106, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %122[0, 3] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%106, %115] : f32 into vector<2x8xf32>
      %131 = arith.truncf %122 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %132 = vector.extract %131[0, 0] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %arg7 [%106, %109] : f16 into vector<2x8xf16>
      %134 = vector.extract %131[0, 1] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%106, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %131[0, 2] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%106, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %131[0, 3] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%106, %115] : f16 into vector<2x8xf16>
      %140 = vector.insert %132, %arg8 [%106, %109] : f16 into vector<2x8xf16>
      %141 = vector.insert %134, %140 [%106, %111] : f16 into vector<2x8xf16>
      %142 = vector.insert %136, %141 [%106, %113] : f16 into vector<2x8xf16>
      %143 = vector.insert %138, %142 [%106, %115] : f16 into vector<2x8xf16>
      affine.yield %116, %130, %139, %143 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %102 {
      affine.for %arg4 = 0 to 4 {
        %106 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %108 = vector.extract %104#3[%106, %107] : f16 from vector<2x8xf16>
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %110 = vector.extract %104#3[%106, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %112 = vector.extract %104#3[%106, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %114 = vector.extract %104#3[%106, %113] : f16 from vector<2x8xf16>
        affine.store %108, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %110, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %105:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_7, %arg6 = %cst_7, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %106 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %107 = vector.extract %90#1[%106, 0] : f32 from vector<2x1xf32>
      %108 = vector.broadcast %107 : f32 to vector<1x4xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %110 = vector.insert %107, %arg5 [%106, %109] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.insert %107, %110 [%106, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %114 = vector.insert %107, %112 [%106, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %116 = vector.insert %107, %114 [%106, %115] : f32 into vector<2x8xf32>
      %117 = vector.extract %96[%106, %109] : f32 from vector<2x8xf32>
      %118 = vector.extract %96[%106, %111] : f32 from vector<2x8xf32>
      %119 = vector.extract %96[%106, %113] : f32 from vector<2x8xf32>
      %120 = vector.extract %96[%106, %115] : f32 from vector<2x8xf32>
      %121 = vector.from_elements %117, %118, %119, %120 : vector<1x4xf32>
      %122 = arith.divf %121, %108 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %123 = vector.extract %122[0, 0] : f32 from vector<1x4xf32>
      %124 = vector.insert %123, %arg6 [%106, %109] : f32 into vector<2x8xf32>
      %125 = vector.extract %122[0, 1] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %124 [%106, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %122[0, 2] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%106, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %122[0, 3] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%106, %115] : f32 into vector<2x8xf32>
      %131 = arith.truncf %122 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %132 = vector.extract %131[0, 0] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %arg7 [%106, %109] : f16 into vector<2x8xf16>
      %134 = vector.extract %131[0, 1] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%106, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %131[0, 2] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%106, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %131[0, 3] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%106, %115] : f16 into vector<2x8xf16>
      %140 = vector.insert %132, %arg8 [%106, %109] : f16 into vector<2x8xf16>
      %141 = vector.insert %134, %140 [%106, %111] : f16 into vector<2x8xf16>
      %142 = vector.insert %136, %141 [%106, %113] : f16 into vector<2x8xf16>
      %143 = vector.insert %138, %142 [%106, %115] : f16 into vector<2x8xf16>
      affine.yield %116, %130, %139, %143 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %102 {
      affine.for %arg4 = 0 to 4 {
        %106 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %108 = vector.extract %105#3[%106, %107] : f16 from vector<2x8xf16>
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %110 = vector.extract %105#3[%106, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %112 = vector.extract %105#3[%106, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %114 = vector.extract %105#3[%106, %113] : f16 from vector<2x8xf16>
        affine.store %108, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %110, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
