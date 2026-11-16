module {
  func.func @Attn(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> {
    %cst = arith.constant dense<1.131250e+01> : tensor<1xf32>
    %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFF800000 : f32
    %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16>
    %4 = deepgengraph.precise_dot_op %1, %3, acc = f32 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf32>
    %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf32>, tensor<1xf32>) -> tensor<1x32x4096x4096xf32>
    %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %7 = deepgengraph.reduce(%6), dim = -1, op = MAX, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %8 = deepgengraph.sub %6, %7 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %9 = deepgengraph.exp %8 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %10 = deepgengraph.reduce(%9), dim = -1, op = ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %11 = deepgengraph.div %9, %10 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %12 = deepgengraph.convert %11, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
    %13 = deepgengraph.dot %12, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16>
    %14 = deepgengraph.permute %13, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16>
    return %14 : tensor<1x4096x32x128xf16>
  }
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 4, 0, 1, 1>, thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
    %c8192 = arith.constant 8192 : index
    %cst = arith.constant 0.127531052 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c31 = arith.constant 31 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() alignment = 16 {frisk.shm_pool, tiled = true} : memref<16384xi8, 3>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %0 = arith.muli %block_id_y, %c32 : index
    %1 = arith.addi %block_id_x, %0 : index
    %2 = arith.divui %1, %c128 : index
    %3 = arith.muli %2, %c4 : index
    %4 = arith.remui %1, %c128 : index
    %5 = arith.remui %4, %c4 : index
    %6 = arith.addi %3, %5 : index
    %7 = arith.subi %c31, %6 : index
    %8 = arith.divui %4, %c4 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %9 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 + s2 * 32 + s0 floordiv 32) mod 32)>()[%7, %8, %block_id_z]
    %10 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%7]
    %subview = memref.subview %arg0[0, %9, %10, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %view = memref.view %alloc[%c0][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %11 = frisk.async_copy_gm_to_shm %subview to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %12 = arith.truncf %cst : f32 to f16
    %13 = arith.muli %7, %c128 : index
    %14:4 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1, %arg6 = %cst_1, %arg7 = %cst_1, %arg8 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %108 = vector.insert %cst_7, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %109 = vector.insert %cst_0, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %110 = vector.insert %cst_0, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %111 = vector.insert %cst_0, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %108, %109, %110, %111 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %15 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %110 = vector.insert %cst_7, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.insert %cst_7, %110 [%109, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %114 = vector.insert %cst_7, %112 [%109, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %116 = vector.insert %cst_7, %114 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %116 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %16 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1) -> (vector<2x1xf32>) {
      %108 = vector.insert %cst_7, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %108 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    frisk.wait_buffer_ready %11 : !frisk.async_task
    %view_8 = memref.view %alloc[%c8192][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %17 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%7, %8]
    %subview_9 = memref.subview %arg0[0, %17, %10, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %18 = frisk.async_copy_gm_to_shm %subview_9 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %thread_id_x = gpu.thread_id x
    %19 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128)>()[%thread_id_x]
    %20 = memref.load %view[%19] : memref<4096xf16, 3>
    %21 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 1)>()[%thread_id_x]
    %22 = memref.load %view[%21] : memref<4096xf16, 3>
    %23 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2)>()[%thread_id_x]
    %24 = memref.load %view[%23] : memref<4096xf16, 3>
    %25 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 3)>()[%thread_id_x]
    %26 = memref.load %view[%25] : memref<4096xf16, 3>
    %27 = vector.from_elements %20, %22, %24, %26 : vector<1x4xf16>
    %28 = vector.broadcast %12 : f16 to vector<1x4xf16>
    %29:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %27) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %109 = memref.load %view[%108] : memref<4096xf16, 3>
      %110 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %111 = memref.load %view[%110] : memref<4096xf16, 3>
      %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %113 = memref.load %view[%112] : memref<4096xf16, 3>
      %114 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %115 = memref.load %view[%114] : memref<4096xf16, 3>
      %116 = vector.from_elements %109, %111, %113, %115 : vector<1x4xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %118 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %119 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg5 [%117, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %122 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%117, %121] : f16 into vector<2x8xf16>
      %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %125 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %123 [%117, %124] : f16 into vector<2x8xf16>
      %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %128 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %126 [%117, %127] : f16 into vector<2x8xf16>
      %130 = vector.insert %12, %arg6 [%117, %118] : f16 into vector<2x8xf16>
      %131 = vector.insert %12, %130 [%117, %121] : f16 into vector<2x8xf16>
      %132 = vector.insert %12, %131 [%117, %124] : f16 into vector<2x8xf16>
      %133 = vector.insert %12, %132 [%117, %127] : f16 into vector<2x8xf16>
      %134 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%117, %118] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%117, %121] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%117, %124] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%117, %127] : f16 into vector<2x8xf16>
      affine.yield %129, %133, %142, %116 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %30 = arith.mulf %29#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %31 = vector.extract %30[0, 0] : f16 from vector<1x4xf16>
    %32 = vector.insert %31, %29#2 [1, 4] : f16 into vector<2x8xf16>
    %33 = vector.extract %30[0, 1] : f16 from vector<1x4xf16>
    %34 = vector.insert %33, %32 [1, 5] : f16 into vector<2x8xf16>
    %35 = vector.extract %30[0, 2] : f16 from vector<1x4xf16>
    %36 = vector.insert %35, %34 [1, 6] : f16 into vector<2x8xf16>
    %37 = vector.extract %30[0, 3] : f16 from vector<1x4xf16>
    %38 = vector.insert %37, %36 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %18 : !frisk.async_task
    %subview_10 = memref.subview %arg0[0, %17, %10, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %39 = frisk.async_copy_gm_to_shm %subview_10 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %40 = memref.load %view_8[%19] : memref<4096xf16, 3>
    %41 = memref.load %view_8[%21] : memref<4096xf16, 3>
    %42 = memref.load %view_8[%23] : memref<4096xf16, 3>
    %43 = memref.load %view_8[%25] : memref<4096xf16, 3>
    %44 = vector.from_elements %40, %41, %42, %43 : vector<1x4xf16>
    %45:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %44) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %109 = memref.load %view_8[%108] : memref<4096xf16, 3>
      %110 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %111 = memref.load %view_8[%110] : memref<4096xf16, 3>
      %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %113 = memref.load %view_8[%112] : memref<4096xf16, 3>
      %114 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %115 = memref.load %view_8[%114] : memref<4096xf16, 3>
      %116 = vector.from_elements %109, %111, %113, %115 : vector<1x4xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %118 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %119 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg5 [%117, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %122 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%117, %121] : f16 into vector<2x8xf16>
      %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %125 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %123 [%117, %124] : f16 into vector<2x8xf16>
      %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %128 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %126 [%117, %127] : f16 into vector<2x8xf16>
      %130 = vector.insert %12, %arg6 [%117, %118] : f16 into vector<2x8xf16>
      %131 = vector.insert %12, %130 [%117, %121] : f16 into vector<2x8xf16>
      %132 = vector.insert %12, %131 [%117, %124] : f16 into vector<2x8xf16>
      %133 = vector.insert %12, %132 [%117, %127] : f16 into vector<2x8xf16>
      %134 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%117, %118] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%117, %121] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%117, %124] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%117, %127] : f16 into vector<2x8xf16>
      affine.yield %129, %133, %142, %116 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %46 = arith.mulf %45#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %47 = vector.extract %46[0, 0] : f16 from vector<1x4xf16>
    %48 = vector.insert %47, %45#2 [1, 4] : f16 into vector<2x8xf16>
    %49 = vector.extract %46[0, 1] : f16 from vector<1x4xf16>
    %50 = vector.insert %49, %48 [1, 5] : f16 into vector<2x8xf16>
    %51 = vector.extract %46[0, 2] : f16 from vector<1x4xf16>
    %52 = vector.insert %51, %50 [1, 6] : f16 into vector<2x8xf16>
    %53 = vector.extract %46[0, 3] : f16 from vector<1x4xf16>
    %54 = vector.insert %53, %52 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %39 : !frisk.async_task
    %subview_11 = memref.subview %arg0[0, %17, %10, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %55 = frisk.async_copy_gm_to_shm %subview_11 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %56 = memref.load %view[%19] : memref<4096xf16, 3>
    %57 = memref.load %view[%21] : memref<4096xf16, 3>
    %58 = memref.load %view[%23] : memref<4096xf16, 3>
    %59 = memref.load %view[%25] : memref<4096xf16, 3>
    %60 = vector.from_elements %56, %57, %58, %59 : vector<1x4xf16>
    %61:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %60) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %109 = memref.load %view[%108] : memref<4096xf16, 3>
      %110 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %111 = memref.load %view[%110] : memref<4096xf16, 3>
      %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %113 = memref.load %view[%112] : memref<4096xf16, 3>
      %114 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %115 = memref.load %view[%114] : memref<4096xf16, 3>
      %116 = vector.from_elements %109, %111, %113, %115 : vector<1x4xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %118 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %119 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg5 [%117, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %122 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%117, %121] : f16 into vector<2x8xf16>
      %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %125 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %123 [%117, %124] : f16 into vector<2x8xf16>
      %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %128 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %126 [%117, %127] : f16 into vector<2x8xf16>
      %130 = vector.insert %12, %arg6 [%117, %118] : f16 into vector<2x8xf16>
      %131 = vector.insert %12, %130 [%117, %121] : f16 into vector<2x8xf16>
      %132 = vector.insert %12, %131 [%117, %124] : f16 into vector<2x8xf16>
      %133 = vector.insert %12, %132 [%117, %127] : f16 into vector<2x8xf16>
      %134 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%117, %118] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%117, %121] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%117, %124] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%117, %127] : f16 into vector<2x8xf16>
      affine.yield %129, %133, %142, %116 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %62 = arith.mulf %61#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %63 = vector.extract %62[0, 0] : f16 from vector<1x4xf16>
    %64 = vector.insert %63, %61#2 [1, 4] : f16 into vector<2x8xf16>
    %65 = vector.extract %62[0, 1] : f16 from vector<1x4xf16>
    %66 = vector.insert %65, %64 [1, 5] : f16 into vector<2x8xf16>
    %67 = vector.extract %62[0, 2] : f16 from vector<1x4xf16>
    %68 = vector.insert %67, %66 [1, 6] : f16 into vector<2x8xf16>
    %69 = vector.extract %62[0, 3] : f16 from vector<1x4xf16>
    %70 = vector.insert %69, %68 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %55 : !frisk.async_task
    %71 = memref.load %view_8[%19] : memref<4096xf16, 3>
    %72 = memref.load %view_8[%21] : memref<4096xf16, 3>
    %73 = memref.load %view_8[%23] : memref<4096xf16, 3>
    %74 = memref.load %view_8[%25] : memref<4096xf16, 3>
    %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf16>
    %76:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %75) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %109 = memref.load %view_8[%108] : memref<4096xf16, 3>
      %110 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %111 = memref.load %view_8[%110] : memref<4096xf16, 3>
      %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %113 = memref.load %view_8[%112] : memref<4096xf16, 3>
      %114 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %115 = memref.load %view_8[%114] : memref<4096xf16, 3>
      %116 = vector.from_elements %109, %111, %113, %115 : vector<1x4xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %118 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %119 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg5 [%117, %118] : f16 into vector<2x8xf16>
      %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %122 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %120 [%117, %121] : f16 into vector<2x8xf16>
      %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %125 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %123 [%117, %124] : f16 into vector<2x8xf16>
      %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %128 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %126 [%117, %127] : f16 into vector<2x8xf16>
      %130 = vector.insert %12, %arg6 [%117, %118] : f16 into vector<2x8xf16>
      %131 = vector.insert %12, %130 [%117, %121] : f16 into vector<2x8xf16>
      %132 = vector.insert %12, %131 [%117, %124] : f16 into vector<2x8xf16>
      %133 = vector.insert %12, %132 [%117, %127] : f16 into vector<2x8xf16>
      %134 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %135 = vector.extract %134[0, 0] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %arg7 [%117, %118] : f16 into vector<2x8xf16>
      %137 = vector.extract %134[0, 1] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%117, %121] : f16 into vector<2x8xf16>
      %139 = vector.extract %134[0, 2] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%117, %124] : f16 into vector<2x8xf16>
      %141 = vector.extract %134[0, 3] : f16 from vector<1x4xf16>
      %142 = vector.insert %141, %140 [%117, %127] : f16 into vector<2x8xf16>
      affine.yield %129, %133, %142, %116 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %77 = arith.mulf %76#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %78 = vector.extract %77[0, 0] : f16 from vector<1x4xf16>
    %79 = vector.insert %78, %76#2 [1, 4] : f16 into vector<2x8xf16>
    %80 = vector.extract %77[0, 1] : f16 from vector<1x4xf16>
    %81 = vector.insert %80, %79 [1, 5] : f16 into vector<2x8xf16>
    %82 = vector.extract %77[0, 2] : f16 from vector<1x4xf16>
    %83 = vector.insert %82, %81 [1, 6] : f16 into vector<2x8xf16>
    %84 = vector.extract %77[0, 3] : f16 from vector<1x4xf16>
    %85 = vector.insert %84, %83 [1, 7] : f16 into vector<2x8xf16>
    gpu.barrier
    %86 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_3) -> (vector<2x32xf32>) {
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %109 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg4)
      %110 = vector.insert %cst_7, %arg5 [%109, %108] : f32 into vector<2x32xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %112 = vector.insert %cst_7, %110 [%109, %111] : f32 into vector<2x32xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %114 = vector.insert %cst_7, %112 [%109, %113] : f32 into vector<2x32xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %116 = vector.insert %cst_7, %114 [%109, %115] : f32 into vector<2x32xf32>
      affine.yield %116 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
    %87 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_3) -> (vector<2x32xf32>) {
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg4)
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %110 = vector.extract %86[%108, %109] : f32 from vector<2x32xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %112 = vector.extract %86[%108, %111] : f32 from vector<2x32xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %114 = vector.extract %86[%108, %113] : f32 from vector<2x32xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %116 = vector.extract %86[%108, %115] : f32 from vector<2x32xf32>
      %117 = vector.insert %110, %arg5 [%108, %109] : f32 into vector<2x32xf32>
      %118 = vector.insert %112, %117 [%108, %111] : f32 into vector<2x32xf32>
      %119 = vector.insert %114, %118 [%108, %113] : f32 into vector<2x32xf32>
      %120 = vector.insert %116, %119 [%108, %115] : f32 into vector<2x32xf32>
      affine.yield %120 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %88 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
    %89 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
    %90 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
    %91 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
    %92:6 = affine.for %arg4 = 0 to affine_map<()[s0] -> (s0 + 1)>()[%7] iter_args(%arg5 = %15, %arg6 = %16, %arg7 = %14#3, %arg8 = %15, %arg9 = %15, %arg10 = %15) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %108 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 floordiv 4096 + s0 + s1 * 32) mod 32)>(%arg4)[%8, %block_id_z]
      %109 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 128)>(%arg4)
      %110 = affine.apply affine_map<(d0) -> (d0 * 128 - (d0 floordiv 32) * 4096)>(%arg4)
      %subview_12 = memref.subview %transpose[0, %108, %109, %110] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %111 = frisk.async_copy_gm_to_shm %subview_12 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %111 : !frisk.async_task
      %112 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32)>(%arg4)[%8, %block_id_z]
      %113 = affine.apply affine_map<(d0)[s0, s1] -> (d0 floordiv 32 + ((d0 floordiv 32 + s0 * 128 + s1 * 4096 + 32) floordiv 128) * -128 + s0 * 128 + s1 * 4096 + 32)>(%arg4)[%8, %block_id_z]
      %subview_13 = memref.subview %transpose[0, %112, %113, %110] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %114 = frisk.async_copy_gm_to_shm %subview_13 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %115:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %87, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %87[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %87[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %87[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %87[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %169 = memref.load %view[%168] : memref<4096xf16, 3>
        %170 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %171 = memref.load %view[%170] : memref<4096xf16, 3>
        %172 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %173 = memref.load %view[%172] : memref<4096xf16, 3>
        %174 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %175 = memref.load %view[%174] : memref<4096xf16, 3>
        %176 = vector.from_elements %169, %171, %173, %175 : vector<4x1xf16>
        %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %178 = memref.load %view[%177] : memref<4096xf16, 3>
        %179 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %180 = memref.load %view[%179] : memref<4096xf16, 3>
        %181 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %182 = memref.load %view[%181] : memref<4096xf16, 3>
        %183 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %184 = memref.load %view[%183] : memref<4096xf16, 3>
        %185 = vector.from_elements %178, %180, %182, %184 : vector<4x1xf16>
        %186 = vector.extract %38[%158, 0] : f16 from vector<2x8xf16>
        %187 = vector.extract %38[%158, 1] : f16 from vector<2x8xf16>
        %188 = vector.extract %38[%158, 2] : f16 from vector<2x8xf16>
        %189 = vector.extract %38[%158, 3] : f16 from vector<2x8xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<1x4xf16>
        %191 = frisk.warp_mma_rr %190, %176, %167 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %192 = vector.extract %38[%158, 4] : f16 from vector<2x8xf16>
        %193 = vector.extract %38[%158, 5] : f16 from vector<2x8xf16>
        %194 = vector.extract %38[%158, 6] : f16 from vector<2x8xf16>
        %195 = vector.extract %38[%158, 7] : f16 from vector<2x8xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<1x4xf16>
        %197 = frisk.warp_mma_rr %196, %185, %191 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %198 = vector.extract %197[0, 0] : f32 from vector<1x4xf32>
        %199 = vector.insert %198, %arg12 [%158, %159] : f32 into vector<2x32xf32>
        %200 = vector.extract %197[0, 1] : f32 from vector<1x4xf32>
        %201 = vector.insert %200, %199 [%158, %161] : f32 into vector<2x32xf32>
        %202 = vector.extract %197[0, 2] : f32 from vector<1x4xf32>
        %203 = vector.insert %202, %201 [%158, %163] : f32 into vector<2x32xf32>
        %204 = vector.extract %197[0, 3] : f32 from vector<1x4xf32>
        %205 = vector.insert %204, %203 [%158, %165] : f32 into vector<2x32xf32>
        %206 = vector.insert %198, %arg13 [%158, %159] : f32 into vector<2x32xf32>
        %207 = vector.insert %200, %206 [%158, %161] : f32 into vector<2x32xf32>
        %208 = vector.insert %202, %207 [%158, %163] : f32 into vector<2x32xf32>
        %209 = vector.insert %204, %208 [%158, %165] : f32 into vector<2x32xf32>
        affine.yield %205, %209 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %116 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %115#1[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %115#1[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %115#1[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %115#1[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.insert %160, %arg12 [%158, %159] : f32 into vector<2x32xf32>
        %168 = vector.insert %162, %167 [%158, %161] : f32 into vector<2x32xf32>
        %169 = vector.insert %164, %168 [%158, %163] : f32 into vector<2x32xf32>
        %170 = vector.insert %166, %169 [%158, %165] : f32 into vector<2x32xf32>
        affine.yield %170 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %114 : !frisk.async_task
      %117 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 2048) floordiv 4096) mod 32)>(%arg4)[%8, %block_id_z]
      %118 = affine.apply affine_map<(d0)[s0, s1] -> (d0 floordiv 32 + ((d0 floordiv 32 + s0 * 128 + s1 * 4096 + 64) floordiv 128) * -128 + s0 * 128 + s1 * 4096 + 64)>(%arg4)[%8, %block_id_z]
      %subview_14 = memref.subview %transpose[0, %117, %118, %110] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %119 = frisk.async_copy_gm_to_shm %subview_14 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %120:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %116, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %116[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %116[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %116[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %116[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %169 = memref.load %view_8[%168] : memref<4096xf16, 3>
        %170 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %171 = memref.load %view_8[%170] : memref<4096xf16, 3>
        %172 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %173 = memref.load %view_8[%172] : memref<4096xf16, 3>
        %174 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %175 = memref.load %view_8[%174] : memref<4096xf16, 3>
        %176 = vector.from_elements %169, %171, %173, %175 : vector<4x1xf16>
        %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %178 = memref.load %view_8[%177] : memref<4096xf16, 3>
        %179 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %180 = memref.load %view_8[%179] : memref<4096xf16, 3>
        %181 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %182 = memref.load %view_8[%181] : memref<4096xf16, 3>
        %183 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %184 = memref.load %view_8[%183] : memref<4096xf16, 3>
        %185 = vector.from_elements %178, %180, %182, %184 : vector<4x1xf16>
        %186 = vector.extract %54[%158, 0] : f16 from vector<2x8xf16>
        %187 = vector.extract %54[%158, 1] : f16 from vector<2x8xf16>
        %188 = vector.extract %54[%158, 2] : f16 from vector<2x8xf16>
        %189 = vector.extract %54[%158, 3] : f16 from vector<2x8xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<1x4xf16>
        %191 = frisk.warp_mma_rr %190, %176, %167 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %192 = vector.extract %54[%158, 4] : f16 from vector<2x8xf16>
        %193 = vector.extract %54[%158, 5] : f16 from vector<2x8xf16>
        %194 = vector.extract %54[%158, 6] : f16 from vector<2x8xf16>
        %195 = vector.extract %54[%158, 7] : f16 from vector<2x8xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<1x4xf16>
        %197 = frisk.warp_mma_rr %196, %185, %191 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %198 = vector.extract %197[0, 0] : f32 from vector<1x4xf32>
        %199 = vector.insert %198, %arg12 [%158, %159] : f32 into vector<2x32xf32>
        %200 = vector.extract %197[0, 1] : f32 from vector<1x4xf32>
        %201 = vector.insert %200, %199 [%158, %161] : f32 into vector<2x32xf32>
        %202 = vector.extract %197[0, 2] : f32 from vector<1x4xf32>
        %203 = vector.insert %202, %201 [%158, %163] : f32 into vector<2x32xf32>
        %204 = vector.extract %197[0, 3] : f32 from vector<1x4xf32>
        %205 = vector.insert %204, %203 [%158, %165] : f32 into vector<2x32xf32>
        %206 = vector.insert %198, %arg13 [%158, %159] : f32 into vector<2x32xf32>
        %207 = vector.insert %200, %206 [%158, %161] : f32 into vector<2x32xf32>
        %208 = vector.insert %202, %207 [%158, %163] : f32 into vector<2x32xf32>
        %209 = vector.insert %204, %208 [%158, %165] : f32 into vector<2x32xf32>
        affine.yield %205, %209 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %121 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %120#1[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %120#1[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %120#1[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %120#1[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.insert %160, %arg12 [%158, %159] : f32 into vector<2x32xf32>
        %168 = vector.insert %162, %167 [%158, %161] : f32 into vector<2x32xf32>
        %169 = vector.insert %164, %168 [%158, %163] : f32 into vector<2x32xf32>
        %170 = vector.insert %166, %169 [%158, %165] : f32 into vector<2x32xf32>
        affine.yield %170 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %119 : !frisk.async_task
      %122 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 3072) floordiv 4096) mod 32)>(%arg4)[%8, %block_id_z]
      %123 = affine.apply affine_map<(d0)[s0, s1] -> (d0 floordiv 32 + ((d0 floordiv 32 + s0 * 128 + s1 * 4096 + 96) floordiv 128) * -128 + s0 * 128 + s1 * 4096 + 96)>(%arg4)[%8, %block_id_z]
      %subview_15 = memref.subview %transpose[0, %122, %123, %110] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %124 = frisk.async_copy_gm_to_shm %subview_15 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %125:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %121, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %121[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %121[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %121[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %121[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %169 = memref.load %view[%168] : memref<4096xf16, 3>
        %170 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %171 = memref.load %view[%170] : memref<4096xf16, 3>
        %172 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %173 = memref.load %view[%172] : memref<4096xf16, 3>
        %174 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %175 = memref.load %view[%174] : memref<4096xf16, 3>
        %176 = vector.from_elements %169, %171, %173, %175 : vector<4x1xf16>
        %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %178 = memref.load %view[%177] : memref<4096xf16, 3>
        %179 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %180 = memref.load %view[%179] : memref<4096xf16, 3>
        %181 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %182 = memref.load %view[%181] : memref<4096xf16, 3>
        %183 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %184 = memref.load %view[%183] : memref<4096xf16, 3>
        %185 = vector.from_elements %178, %180, %182, %184 : vector<4x1xf16>
        %186 = vector.extract %70[%158, 0] : f16 from vector<2x8xf16>
        %187 = vector.extract %70[%158, 1] : f16 from vector<2x8xf16>
        %188 = vector.extract %70[%158, 2] : f16 from vector<2x8xf16>
        %189 = vector.extract %70[%158, 3] : f16 from vector<2x8xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<1x4xf16>
        %191 = frisk.warp_mma_rr %190, %176, %167 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %192 = vector.extract %70[%158, 4] : f16 from vector<2x8xf16>
        %193 = vector.extract %70[%158, 5] : f16 from vector<2x8xf16>
        %194 = vector.extract %70[%158, 6] : f16 from vector<2x8xf16>
        %195 = vector.extract %70[%158, 7] : f16 from vector<2x8xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<1x4xf16>
        %197 = frisk.warp_mma_rr %196, %185, %191 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %198 = vector.extract %197[0, 0] : f32 from vector<1x4xf32>
        %199 = vector.insert %198, %arg12 [%158, %159] : f32 into vector<2x32xf32>
        %200 = vector.extract %197[0, 1] : f32 from vector<1x4xf32>
        %201 = vector.insert %200, %199 [%158, %161] : f32 into vector<2x32xf32>
        %202 = vector.extract %197[0, 2] : f32 from vector<1x4xf32>
        %203 = vector.insert %202, %201 [%158, %163] : f32 into vector<2x32xf32>
        %204 = vector.extract %197[0, 3] : f32 from vector<1x4xf32>
        %205 = vector.insert %204, %203 [%158, %165] : f32 into vector<2x32xf32>
        %206 = vector.insert %198, %arg13 [%158, %159] : f32 into vector<2x32xf32>
        %207 = vector.insert %200, %206 [%158, %161] : f32 into vector<2x32xf32>
        %208 = vector.insert %202, %207 [%158, %163] : f32 into vector<2x32xf32>
        %209 = vector.insert %204, %208 [%158, %165] : f32 into vector<2x32xf32>
        affine.yield %205, %209 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %126 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %125#1[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %125#1[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %125#1[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %125#1[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.insert %160, %arg12 [%158, %159] : f32 into vector<2x32xf32>
        %168 = vector.insert %162, %167 [%158, %161] : f32 into vector<2x32xf32>
        %169 = vector.insert %164, %168 [%158, %163] : f32 into vector<2x32xf32>
        %170 = vector.insert %166, %169 [%158, %165] : f32 into vector<2x32xf32>
        affine.yield %170 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %124 : !frisk.async_task
      %127 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 floordiv 32 + s0 + s1 * 32) mod 32)>(%arg4)[%8, %block_id_z]
      %subview_16 = memref.subview %arg2[0, %127, %110, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %128 = frisk.async_copy_gm_to_shm %subview_16 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %129:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %126, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %126[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %126[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %126[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %126[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %169 = memref.load %view_8[%168] : memref<4096xf16, 3>
        %170 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %171 = memref.load %view_8[%170] : memref<4096xf16, 3>
        %172 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %173 = memref.load %view_8[%172] : memref<4096xf16, 3>
        %174 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %175 = memref.load %view_8[%174] : memref<4096xf16, 3>
        %176 = vector.from_elements %169, %171, %173, %175 : vector<4x1xf16>
        %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %178 = memref.load %view_8[%177] : memref<4096xf16, 3>
        %179 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %180 = memref.load %view_8[%179] : memref<4096xf16, 3>
        %181 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %182 = memref.load %view_8[%181] : memref<4096xf16, 3>
        %183 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %184 = memref.load %view_8[%183] : memref<4096xf16, 3>
        %185 = vector.from_elements %178, %180, %182, %184 : vector<4x1xf16>
        %186 = vector.extract %85[%158, 0] : f16 from vector<2x8xf16>
        %187 = vector.extract %85[%158, 1] : f16 from vector<2x8xf16>
        %188 = vector.extract %85[%158, 2] : f16 from vector<2x8xf16>
        %189 = vector.extract %85[%158, 3] : f16 from vector<2x8xf16>
        %190 = vector.from_elements %186, %187, %188, %189 : vector<1x4xf16>
        %191 = frisk.warp_mma_rr %190, %176, %167 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %192 = vector.extract %85[%158, 4] : f16 from vector<2x8xf16>
        %193 = vector.extract %85[%158, 5] : f16 from vector<2x8xf16>
        %194 = vector.extract %85[%158, 6] : f16 from vector<2x8xf16>
        %195 = vector.extract %85[%158, 7] : f16 from vector<2x8xf16>
        %196 = vector.from_elements %192, %193, %194, %195 : vector<1x4xf16>
        %197 = frisk.warp_mma_rr %196, %185, %191 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %198 = vector.extract %197[0, 0] : f32 from vector<1x4xf32>
        %199 = vector.insert %198, %arg12 [%158, %159] : f32 into vector<2x32xf32>
        %200 = vector.extract %197[0, 1] : f32 from vector<1x4xf32>
        %201 = vector.insert %200, %199 [%158, %161] : f32 into vector<2x32xf32>
        %202 = vector.extract %197[0, 2] : f32 from vector<1x4xf32>
        %203 = vector.insert %202, %201 [%158, %163] : f32 into vector<2x32xf32>
        %204 = vector.extract %197[0, 3] : f32 from vector<1x4xf32>
        %205 = vector.insert %204, %203 [%158, %165] : f32 into vector<2x32xf32>
        %206 = vector.insert %198, %arg13 [%158, %159] : f32 into vector<2x32xf32>
        %207 = vector.insert %200, %206 [%158, %161] : f32 into vector<2x32xf32>
        %208 = vector.insert %202, %207 [%158, %163] : f32 into vector<2x32xf32>
        %209 = vector.insert %204, %208 [%158, %165] : f32 into vector<2x32xf32>
        affine.yield %205, %209 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %130 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %160 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 floordiv 8) * 64 + s0 + s1 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg11)[%thread_id_x, %13]
        %161 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 * 128 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + ((s0 mod 64) floordiv 16) * 16)>(%arg11, %arg4)[%thread_id_x]
        %162 = arith.index_cast %160 : index to i32
        %163 = arith.index_cast %161 : index to i32
        %164 = arith.addi %162, %c1_i32 : i32
        %165 = arith.cmpi ule, %164, %163 {frisk.causal_mask_i32} : i32
        %166 = vector.extract %129#1[%159, %158] : f32 from vector<2x32xf32>
        %167 = arith.select %165, %cst_0, %166 {frisk.causal_mask_select} : f32
        %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %169 = affine.apply affine_map<(d0, d1)[s0] -> (d1 * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 4) floordiv 64) * 64 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 4) floordiv 16) * 4 + (((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 4) floordiv 16) floordiv 4) * -16 + ((s0 mod 64) floordiv 16) * 16 + 1)>(%arg11, %arg4)[%thread_id_x]
        %170 = arith.index_cast %169 : index to i32
        %171 = arith.cmpi ule, %164, %170 {frisk.causal_mask_i32} : i32
        %172 = vector.extract %129#1[%159, %168] : f32 from vector<2x32xf32>
        %173 = arith.select %171, %cst_0, %172 {frisk.causal_mask_select} : f32
        %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %175 = affine.apply affine_map<(d0, d1)[s0] -> (d1 * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 8) floordiv 64) * 64 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 8) floordiv 16) * 4 + (((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 8) floordiv 16) floordiv 4) * -16 + ((s0 mod 64) floordiv 16) * 16 + 2)>(%arg11, %arg4)[%thread_id_x]
        %176 = arith.index_cast %175 : index to i32
        %177 = arith.cmpi ule, %164, %176 {frisk.causal_mask_i32} : i32
        %178 = vector.extract %129#1[%159, %174] : f32 from vector<2x32xf32>
        %179 = arith.select %177, %cst_0, %178 {frisk.causal_mask_select} : f32
        %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %181 = affine.apply affine_map<(d0, d1)[s0] -> (d1 * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 12) floordiv 64) * 64 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 12) floordiv 16) * 4 + (((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 12) floordiv 16) floordiv 4) * -16 + ((s0 mod 64) floordiv 16) * 16 + 3)>(%arg11, %arg4)[%thread_id_x]
        %182 = arith.index_cast %181 : index to i32
        %183 = arith.cmpi ule, %164, %182 {frisk.causal_mask_i32} : i32
        %184 = vector.extract %129#1[%159, %180] : f32 from vector<2x32xf32>
        %185 = arith.select %183, %cst_0, %184 {frisk.causal_mask_select} : f32
        %186 = vector.insert %167, %arg12 [%159, %158] : f32 into vector<2x32xf32>
        %187 = vector.insert %173, %186 [%159, %168] : f32 into vector<2x32xf32>
        %188 = vector.insert %179, %187 [%159, %174] : f32 into vector<2x32xf32>
        %189 = vector.insert %185, %188 [%159, %180] : f32 into vector<2x32xf32>
        affine.yield %189 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "mask_fragment", tiled = true}
      %131 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %130[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %130[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %130[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %130[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.insert %160, %arg12 [%158, %159] : f32 into vector<2x32xf32>
        %168 = vector.insert %162, %167 [%158, %161] : f32 into vector<2x32xf32>
        %169 = vector.insert %164, %168 [%158, %163] : f32 into vector<2x32xf32>
        %170 = vector.insert %166, %169 [%158, %165] : f32 into vector<2x32xf32>
        affine.yield %170 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %132 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %158 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_0) -> (f32) {
          %162 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %163 = vector.extract %131[%arg11, %162] : f32 from vector<2x32xf32>
          %164 = arith.maxnumf %arg14, %163 {frisk.try_multichain} : f32
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %166 = vector.extract %131[%arg11, %165] : f32 from vector<2x32xf32>
          %167 = arith.maxnumf %164, %166 {frisk.try_multichain} : f32
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %169 = vector.extract %131[%arg11, %168] : f32 from vector<2x32xf32>
          %170 = arith.maxnumf %167, %169 {frisk.try_multichain} : f32
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %172 = vector.extract %131[%arg11, %171] : f32 from vector<2x32xf32>
          %173 = arith.maxnumf %170, %172 {frisk.try_multichain} : f32
          affine.yield %173 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %158, %c16_i32, %c64_i32 : f32
        %159 = arith.maxnumf %158, %shuffleResult : f32
        %shuffleResult_20, %valid_21 = gpu.shuffle xor %159, %c32_i32, %c64_i32 : f32
        %160 = arith.maxnumf %159, %shuffleResult_20 : f32
        %161 = vector.insert %160, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %161 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %133 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %158 = vector.extract %132[%arg11, 0] : f32 from vector<2x1xf32>
        %159 = vector.broadcast %158 : f32 to vector<1x1xf32>
        %160 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %161 = vector.broadcast %160 : f32 to vector<1x1xf32>
        %162 = arith.maxnumf %159, %161 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %134:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = vector.extract %133[%158, 0] : f32 from vector<2x1xf32>
        %160 = vector.broadcast %159 : f32 to vector<1x4xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %162 = vector.insert %159, %arg12 [%158, %161] : f32 into vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %164 = vector.insert %159, %162 [%158, %163] : f32 into vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %166 = vector.insert %159, %164 [%158, %165] : f32 into vector<2x32xf32>
        %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %168 = vector.insert %159, %166 [%158, %167] : f32 into vector<2x32xf32>
        %169 = vector.extract %130[%158, %161] : f32 from vector<2x32xf32>
        %170 = vector.extract %130[%158, %163] : f32 from vector<2x32xf32>
        %171 = vector.extract %130[%158, %165] : f32 from vector<2x32xf32>
        %172 = vector.extract %130[%158, %167] : f32 from vector<2x32xf32>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf32>
        %174 = arith.subf %173, %160 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg13 [%158, %161] : f32 into vector<2x32xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%158, %163] : f32 into vector<2x32xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%158, %165] : f32 into vector<2x32xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%158, %167] : f32 into vector<2x32xf32>
        affine.yield %168, %182 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %135:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %158 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %159 = vector.broadcast %158 : f32 to vector<1x1xf32>
        %160 = vector.extract %133[%arg11, 0] : f32 from vector<2x1xf32>
        %161 = vector.broadcast %160 : f32 to vector<1x1xf32>
        %162 = arith.subf %159, %161 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %165 = math.exp2 %162 {tiled = true} : vector<1x1xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x1xf32>
        %167 = vector.insert %166, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %168 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %169 = vector.broadcast %168 : f32 to vector<1x1xf32>
        %170 = arith.mulf %169, %165 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %171 = vector.extract %170[0, 0] : f32 from vector<1x1xf32>
        %172 = vector.insert %171, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %164, %167, %172 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %136:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %134#1[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %134#1[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %134#1[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %134#1[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = math.exp2 %167 {tiled = true} : vector<1x4xf32>
        %169 = vector.extract %168[0, 0] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %arg12 [%158, %159] : f32 into vector<2x32xf32>
        %171 = vector.extract %168[0, 1] : f32 from vector<1x4xf32>
        %172 = vector.insert %171, %170 [%158, %161] : f32 into vector<2x32xf32>
        %173 = vector.extract %168[0, 2] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %172 [%158, %163] : f32 into vector<2x32xf32>
        %175 = vector.extract %168[0, 3] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %174 [%158, %165] : f32 into vector<2x32xf32>
        %177 = vector.insert %169, %arg13 [%158, %159] : f32 into vector<2x32xf32>
        %178 = vector.insert %171, %177 [%158, %161] : f32 into vector<2x32xf32>
        %179 = vector.insert %173, %178 [%158, %163] : f32 into vector<2x32xf32>
        %180 = vector.insert %175, %179 [%158, %165] : f32 into vector<2x32xf32>
        affine.yield %176, %180 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %137 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %158 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_7) -> (f32) {
          %162 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %163 = vector.extract %136#1[%arg11, %162] : f32 from vector<2x32xf32>
          %164 = arith.addf %arg14, %163 {frisk.try_multichain} : f32
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %166 = vector.extract %136#1[%arg11, %165] : f32 from vector<2x32xf32>
          %167 = arith.addf %164, %166 {frisk.try_multichain} : f32
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %169 = vector.extract %136#1[%arg11, %168] : f32 from vector<2x32xf32>
          %170 = arith.addf %167, %169 {frisk.try_multichain} : f32
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %172 = vector.extract %136#1[%arg11, %171] : f32 from vector<2x32xf32>
          %173 = arith.addf %170, %172 {frisk.try_multichain} : f32
          affine.yield %173 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %158, %c16_i32, %c64_i32 : f32
        %159 = arith.addf %158, %shuffleResult : f32
        %shuffleResult_20, %valid_21 = gpu.shuffle xor %159, %c32_i32, %c64_i32 : f32
        %160 = arith.addf %159, %shuffleResult_20 : f32
        %161 = vector.insert %160, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %161 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %138 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %158 = vector.extract %137[%arg11, 0] : f32 from vector<2x1xf32>
        %159 = vector.broadcast %158 : f32 to vector<1x1xf32>
        %160 = vector.extract %135#2[%arg11, 0] : f32 from vector<2x1xf32>
        %161 = vector.broadcast %160 : f32 to vector<1x1xf32>
        %162 = arith.addf %159, %161 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %139 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf16>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %160 = vector.extract %136#0[%158, %159] : f32 from vector<2x32xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %162 = vector.extract %136#0[%158, %161] : f32 from vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %136#0[%158, %163] : f32 from vector<2x32xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %166 = vector.extract %136#0[%158, %165] : f32 from vector<2x32xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = arith.truncf %167 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %169 = vector.extract %168[0, 0] : f16 from vector<1x4xf16>
        %170 = vector.insert %169, %arg12 [%158, %159] : f16 into vector<2x32xf16>
        %171 = vector.extract %168[0, 1] : f16 from vector<1x4xf16>
        %172 = vector.insert %171, %170 [%158, %161] : f16 into vector<2x32xf16>
        %173 = vector.extract %168[0, 2] : f16 from vector<1x4xf16>
        %174 = vector.insert %173, %172 [%158, %163] : f16 into vector<2x32xf16>
        %175 = vector.extract %168[0, 3] : f16 from vector<1x4xf16>
        %176 = vector.insert %175, %174 [%158, %165] : f16 into vector<2x32xf16>
        affine.yield %176 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %140:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %159 = vector.extract %135#1[%158, 0] : f32 from vector<2x1xf32>
        %160 = vector.broadcast %159 : f32 to vector<1x4xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %162 = vector.insert %159, %arg12 [%158, %161] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %164 = vector.insert %159, %162 [%158, %163] : f32 into vector<2x8xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %166 = vector.insert %159, %164 [%158, %165] : f32 into vector<2x8xf32>
        %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %168 = vector.insert %159, %166 [%158, %167] : f32 into vector<2x8xf32>
        %169 = vector.extract %arg5[%158, %161] : f32 from vector<2x8xf32>
        %170 = vector.extract %arg5[%158, %163] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg5[%158, %165] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg5[%158, %167] : f32 from vector<2x8xf32>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf32>
        %174 = arith.mulf %173, %160 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg13 [%158, %161] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%158, %163] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%158, %165] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%158, %167] : f32 into vector<2x8xf32>
        affine.yield %168, %182 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %128 : !frisk.async_task
      %141 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 32 + s0) mod 32)>(%arg4)[%8]
      %subview_17 = memref.subview %arg2[0, %141, %110, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %142 = frisk.async_copy_gm_to_shm %subview_17 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %143 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %158 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %159 = frisk.read_matrix_from_shm %view[%158] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %160 = vector.bitcast %159 : vector<4xi32> to vector<8xf16>
        %161 = vector.extract %160[0] : f16 from vector<8xf16>
        %162 = vector.extract %160[%88] : f16 from vector<8xf16>
        %163 = vector.extract %160[1] : f16 from vector<8xf16>
        %164 = vector.extract %160[%89] : f16 from vector<8xf16>
        %165 = vector.extract %160[2] : f16 from vector<8xf16>
        %166 = vector.extract %160[%90] : f16 from vector<8xf16>
        %167 = vector.extract %160[3] : f16 from vector<8xf16>
        %168 = vector.extract %160[%91] : f16 from vector<8xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %170 = vector.insert %161, %arg12 [%169, 0] : f16 into vector<32x2xf16>
        %171 = vector.insert %162, %170 [%169, 1] : f16 into vector<32x2xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %173 = vector.insert %163, %171 [%172, 0] : f16 into vector<32x2xf16>
        %174 = vector.insert %164, %173 [%172, 1] : f16 into vector<32x2xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %176 = vector.insert %165, %174 [%175, 0] : f16 into vector<32x2xf16>
        %177 = vector.insert %166, %176 [%175, 1] : f16 into vector<32x2xf16>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %179 = vector.insert %167, %177 [%178, 0] : f16 into vector<32x2xf16>
        %180 = vector.insert %168, %179 [%178, 1] : f16 into vector<32x2xf16>
        affine.yield %180 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %144 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %140#1) -> (vector<2x8xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %160 = vector.extract %140#1[%158, %159] : f32 from vector<2x8xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %162 = vector.extract %140#1[%158, %161] : f32 from vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %140#1[%158, %163] : f32 from vector<2x8xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %166 = vector.extract %140#1[%158, %165] : f32 from vector<2x8xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = vector.extract %139[%158, 0] : f16 from vector<2x32xf16>
        %169 = vector.extract %139[%158, 1] : f16 from vector<2x32xf16>
        %170 = vector.extract %139[%158, 2] : f16 from vector<2x32xf16>
        %171 = vector.extract %139[%158, 3] : f16 from vector<2x32xf16>
        %172 = vector.from_elements %168, %169, %170, %171 : vector<1x4xf16>
        %173 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %174 = vector.extract %143[0, %173] : f16 from vector<32x2xf16>
        %175 = vector.extract %143[1, %173] : f16 from vector<32x2xf16>
        %176 = vector.extract %143[2, %173] : f16 from vector<32x2xf16>
        %177 = vector.extract %143[3, %173] : f16 from vector<32x2xf16>
        %178 = vector.from_elements %174, %175, %176, %177 : vector<4x1xf16>
        %179 = frisk.warp_mma_rr %172, %178, %167 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %180 = vector.extract %139[%158, 4] : f16 from vector<2x32xf16>
        %181 = vector.extract %139[%158, 5] : f16 from vector<2x32xf16>
        %182 = vector.extract %139[%158, 6] : f16 from vector<2x32xf16>
        %183 = vector.extract %139[%158, 7] : f16 from vector<2x32xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<1x4xf16>
        %185 = vector.extract %143[4, %173] : f16 from vector<32x2xf16>
        %186 = vector.extract %143[5, %173] : f16 from vector<32x2xf16>
        %187 = vector.extract %143[6, %173] : f16 from vector<32x2xf16>
        %188 = vector.extract %143[7, %173] : f16 from vector<32x2xf16>
        %189 = vector.from_elements %185, %186, %187, %188 : vector<4x1xf16>
        %190 = frisk.warp_mma_rr %184, %189, %179 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %191 = vector.extract %139[%158, 8] : f16 from vector<2x32xf16>
        %192 = vector.extract %139[%158, 9] : f16 from vector<2x32xf16>
        %193 = vector.extract %139[%158, 10] : f16 from vector<2x32xf16>
        %194 = vector.extract %139[%158, 11] : f16 from vector<2x32xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<1x4xf16>
        %196 = vector.extract %143[8, %173] : f16 from vector<32x2xf16>
        %197 = vector.extract %143[9, %173] : f16 from vector<32x2xf16>
        %198 = vector.extract %143[10, %173] : f16 from vector<32x2xf16>
        %199 = vector.extract %143[11, %173] : f16 from vector<32x2xf16>
        %200 = vector.from_elements %196, %197, %198, %199 : vector<4x1xf16>
        %201 = frisk.warp_mma_rr %195, %200, %190 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %202 = vector.extract %139[%158, 12] : f16 from vector<2x32xf16>
        %203 = vector.extract %139[%158, 13] : f16 from vector<2x32xf16>
        %204 = vector.extract %139[%158, 14] : f16 from vector<2x32xf16>
        %205 = vector.extract %139[%158, 15] : f16 from vector<2x32xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<1x4xf16>
        %207 = vector.extract %143[12, %173] : f16 from vector<32x2xf16>
        %208 = vector.extract %143[13, %173] : f16 from vector<32x2xf16>
        %209 = vector.extract %143[14, %173] : f16 from vector<32x2xf16>
        %210 = vector.extract %143[15, %173] : f16 from vector<32x2xf16>
        %211 = vector.from_elements %207, %208, %209, %210 : vector<4x1xf16>
        %212 = frisk.warp_mma_rr %206, %211, %201 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %213 = vector.extract %139[%158, 16] : f16 from vector<2x32xf16>
        %214 = vector.extract %139[%158, 17] : f16 from vector<2x32xf16>
        %215 = vector.extract %139[%158, 18] : f16 from vector<2x32xf16>
        %216 = vector.extract %139[%158, 19] : f16 from vector<2x32xf16>
        %217 = vector.from_elements %213, %214, %215, %216 : vector<1x4xf16>
        %218 = vector.extract %143[16, %173] : f16 from vector<32x2xf16>
        %219 = vector.extract %143[17, %173] : f16 from vector<32x2xf16>
        %220 = vector.extract %143[18, %173] : f16 from vector<32x2xf16>
        %221 = vector.extract %143[19, %173] : f16 from vector<32x2xf16>
        %222 = vector.from_elements %218, %219, %220, %221 : vector<4x1xf16>
        %223 = frisk.warp_mma_rr %217, %222, %212 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %224 = vector.extract %139[%158, 20] : f16 from vector<2x32xf16>
        %225 = vector.extract %139[%158, 21] : f16 from vector<2x32xf16>
        %226 = vector.extract %139[%158, 22] : f16 from vector<2x32xf16>
        %227 = vector.extract %139[%158, 23] : f16 from vector<2x32xf16>
        %228 = vector.from_elements %224, %225, %226, %227 : vector<1x4xf16>
        %229 = vector.extract %143[20, %173] : f16 from vector<32x2xf16>
        %230 = vector.extract %143[21, %173] : f16 from vector<32x2xf16>
        %231 = vector.extract %143[22, %173] : f16 from vector<32x2xf16>
        %232 = vector.extract %143[23, %173] : f16 from vector<32x2xf16>
        %233 = vector.from_elements %229, %230, %231, %232 : vector<4x1xf16>
        %234 = frisk.warp_mma_rr %228, %233, %223 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %235 = vector.extract %139[%158, 24] : f16 from vector<2x32xf16>
        %236 = vector.extract %139[%158, 25] : f16 from vector<2x32xf16>
        %237 = vector.extract %139[%158, 26] : f16 from vector<2x32xf16>
        %238 = vector.extract %139[%158, 27] : f16 from vector<2x32xf16>
        %239 = vector.from_elements %235, %236, %237, %238 : vector<1x4xf16>
        %240 = vector.extract %143[24, %173] : f16 from vector<32x2xf16>
        %241 = vector.extract %143[25, %173] : f16 from vector<32x2xf16>
        %242 = vector.extract %143[26, %173] : f16 from vector<32x2xf16>
        %243 = vector.extract %143[27, %173] : f16 from vector<32x2xf16>
        %244 = vector.from_elements %240, %241, %242, %243 : vector<4x1xf16>
        %245 = frisk.warp_mma_rr %239, %244, %234 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %246 = vector.extract %139[%158, 28] : f16 from vector<2x32xf16>
        %247 = vector.extract %139[%158, 29] : f16 from vector<2x32xf16>
        %248 = vector.extract %139[%158, 30] : f16 from vector<2x32xf16>
        %249 = vector.extract %139[%158, 31] : f16 from vector<2x32xf16>
        %250 = vector.from_elements %246, %247, %248, %249 : vector<1x4xf16>
        %251 = vector.extract %143[28, %173] : f16 from vector<32x2xf16>
        %252 = vector.extract %143[29, %173] : f16 from vector<32x2xf16>
        %253 = vector.extract %143[30, %173] : f16 from vector<32x2xf16>
        %254 = vector.extract %143[31, %173] : f16 from vector<32x2xf16>
        %255 = vector.from_elements %251, %252, %253, %254 : vector<4x1xf16>
        %256 = frisk.warp_mma_rr %250, %255, %245 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %257 = vector.extract %256[0, 0] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %arg12 [%158, %159] : f32 into vector<2x8xf32>
        %259 = vector.extract %256[0, 1] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %258 [%158, %161] : f32 into vector<2x8xf32>
        %261 = vector.extract %256[0, 2] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %260 [%158, %163] : f32 into vector<2x8xf32>
        %263 = vector.extract %256[0, 3] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %262 [%158, %165] : f32 into vector<2x8xf32>
        affine.yield %264 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %145:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %159 = vector.extract %135#1[%158, 0] : f32 from vector<2x1xf32>
        %160 = vector.broadcast %159 : f32 to vector<1x4xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %162 = vector.insert %159, %arg12 [%158, %161] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %164 = vector.insert %159, %162 [%158, %163] : f32 into vector<2x8xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %166 = vector.insert %159, %164 [%158, %165] : f32 into vector<2x8xf32>
        %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %168 = vector.insert %159, %166 [%158, %167] : f32 into vector<2x8xf32>
        %169 = vector.extract %arg8[%158, %161] : f32 from vector<2x8xf32>
        %170 = vector.extract %arg8[%158, %163] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg8[%158, %165] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg8[%158, %167] : f32 from vector<2x8xf32>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf32>
        %174 = arith.mulf %173, %160 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg13 [%158, %161] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%158, %163] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%158, %165] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%158, %167] : f32 into vector<2x8xf32>
        affine.yield %168, %182 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %142 : !frisk.async_task
      %subview_18 = memref.subview %arg2[0, %141, %110, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %146 = frisk.async_copy_gm_to_shm %subview_18 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %147 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %158 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %159 = frisk.read_matrix_from_shm %view_8[%158] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %160 = vector.bitcast %159 : vector<4xi32> to vector<8xf16>
        %161 = vector.extract %160[0] : f16 from vector<8xf16>
        %162 = vector.extract %160[%88] : f16 from vector<8xf16>
        %163 = vector.extract %160[1] : f16 from vector<8xf16>
        %164 = vector.extract %160[%89] : f16 from vector<8xf16>
        %165 = vector.extract %160[2] : f16 from vector<8xf16>
        %166 = vector.extract %160[%90] : f16 from vector<8xf16>
        %167 = vector.extract %160[3] : f16 from vector<8xf16>
        %168 = vector.extract %160[%91] : f16 from vector<8xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %170 = vector.insert %161, %arg12 [%169, 0] : f16 into vector<32x2xf16>
        %171 = vector.insert %162, %170 [%169, 1] : f16 into vector<32x2xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %173 = vector.insert %163, %171 [%172, 0] : f16 into vector<32x2xf16>
        %174 = vector.insert %164, %173 [%172, 1] : f16 into vector<32x2xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %176 = vector.insert %165, %174 [%175, 0] : f16 into vector<32x2xf16>
        %177 = vector.insert %166, %176 [%175, 1] : f16 into vector<32x2xf16>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %179 = vector.insert %167, %177 [%178, 0] : f16 into vector<32x2xf16>
        %180 = vector.insert %168, %179 [%178, 1] : f16 into vector<32x2xf16>
        affine.yield %180 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %148 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %145#1) -> (vector<2x8xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %160 = vector.extract %145#1[%158, %159] : f32 from vector<2x8xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %162 = vector.extract %145#1[%158, %161] : f32 from vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %145#1[%158, %163] : f32 from vector<2x8xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %166 = vector.extract %145#1[%158, %165] : f32 from vector<2x8xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = vector.extract %139[%158, 0] : f16 from vector<2x32xf16>
        %169 = vector.extract %139[%158, 1] : f16 from vector<2x32xf16>
        %170 = vector.extract %139[%158, 2] : f16 from vector<2x32xf16>
        %171 = vector.extract %139[%158, 3] : f16 from vector<2x32xf16>
        %172 = vector.from_elements %168, %169, %170, %171 : vector<1x4xf16>
        %173 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %174 = vector.extract %147[0, %173] : f16 from vector<32x2xf16>
        %175 = vector.extract %147[1, %173] : f16 from vector<32x2xf16>
        %176 = vector.extract %147[2, %173] : f16 from vector<32x2xf16>
        %177 = vector.extract %147[3, %173] : f16 from vector<32x2xf16>
        %178 = vector.from_elements %174, %175, %176, %177 : vector<4x1xf16>
        %179 = frisk.warp_mma_rr %172, %178, %167 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %180 = vector.extract %139[%158, 4] : f16 from vector<2x32xf16>
        %181 = vector.extract %139[%158, 5] : f16 from vector<2x32xf16>
        %182 = vector.extract %139[%158, 6] : f16 from vector<2x32xf16>
        %183 = vector.extract %139[%158, 7] : f16 from vector<2x32xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<1x4xf16>
        %185 = vector.extract %147[4, %173] : f16 from vector<32x2xf16>
        %186 = vector.extract %147[5, %173] : f16 from vector<32x2xf16>
        %187 = vector.extract %147[6, %173] : f16 from vector<32x2xf16>
        %188 = vector.extract %147[7, %173] : f16 from vector<32x2xf16>
        %189 = vector.from_elements %185, %186, %187, %188 : vector<4x1xf16>
        %190 = frisk.warp_mma_rr %184, %189, %179 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %191 = vector.extract %139[%158, 8] : f16 from vector<2x32xf16>
        %192 = vector.extract %139[%158, 9] : f16 from vector<2x32xf16>
        %193 = vector.extract %139[%158, 10] : f16 from vector<2x32xf16>
        %194 = vector.extract %139[%158, 11] : f16 from vector<2x32xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<1x4xf16>
        %196 = vector.extract %147[8, %173] : f16 from vector<32x2xf16>
        %197 = vector.extract %147[9, %173] : f16 from vector<32x2xf16>
        %198 = vector.extract %147[10, %173] : f16 from vector<32x2xf16>
        %199 = vector.extract %147[11, %173] : f16 from vector<32x2xf16>
        %200 = vector.from_elements %196, %197, %198, %199 : vector<4x1xf16>
        %201 = frisk.warp_mma_rr %195, %200, %190 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %202 = vector.extract %139[%158, 12] : f16 from vector<2x32xf16>
        %203 = vector.extract %139[%158, 13] : f16 from vector<2x32xf16>
        %204 = vector.extract %139[%158, 14] : f16 from vector<2x32xf16>
        %205 = vector.extract %139[%158, 15] : f16 from vector<2x32xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<1x4xf16>
        %207 = vector.extract %147[12, %173] : f16 from vector<32x2xf16>
        %208 = vector.extract %147[13, %173] : f16 from vector<32x2xf16>
        %209 = vector.extract %147[14, %173] : f16 from vector<32x2xf16>
        %210 = vector.extract %147[15, %173] : f16 from vector<32x2xf16>
        %211 = vector.from_elements %207, %208, %209, %210 : vector<4x1xf16>
        %212 = frisk.warp_mma_rr %206, %211, %201 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %213 = vector.extract %139[%158, 16] : f16 from vector<2x32xf16>
        %214 = vector.extract %139[%158, 17] : f16 from vector<2x32xf16>
        %215 = vector.extract %139[%158, 18] : f16 from vector<2x32xf16>
        %216 = vector.extract %139[%158, 19] : f16 from vector<2x32xf16>
        %217 = vector.from_elements %213, %214, %215, %216 : vector<1x4xf16>
        %218 = vector.extract %147[16, %173] : f16 from vector<32x2xf16>
        %219 = vector.extract %147[17, %173] : f16 from vector<32x2xf16>
        %220 = vector.extract %147[18, %173] : f16 from vector<32x2xf16>
        %221 = vector.extract %147[19, %173] : f16 from vector<32x2xf16>
        %222 = vector.from_elements %218, %219, %220, %221 : vector<4x1xf16>
        %223 = frisk.warp_mma_rr %217, %222, %212 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %224 = vector.extract %139[%158, 20] : f16 from vector<2x32xf16>
        %225 = vector.extract %139[%158, 21] : f16 from vector<2x32xf16>
        %226 = vector.extract %139[%158, 22] : f16 from vector<2x32xf16>
        %227 = vector.extract %139[%158, 23] : f16 from vector<2x32xf16>
        %228 = vector.from_elements %224, %225, %226, %227 : vector<1x4xf16>
        %229 = vector.extract %147[20, %173] : f16 from vector<32x2xf16>
        %230 = vector.extract %147[21, %173] : f16 from vector<32x2xf16>
        %231 = vector.extract %147[22, %173] : f16 from vector<32x2xf16>
        %232 = vector.extract %147[23, %173] : f16 from vector<32x2xf16>
        %233 = vector.from_elements %229, %230, %231, %232 : vector<4x1xf16>
        %234 = frisk.warp_mma_rr %228, %233, %223 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %235 = vector.extract %139[%158, 24] : f16 from vector<2x32xf16>
        %236 = vector.extract %139[%158, 25] : f16 from vector<2x32xf16>
        %237 = vector.extract %139[%158, 26] : f16 from vector<2x32xf16>
        %238 = vector.extract %139[%158, 27] : f16 from vector<2x32xf16>
        %239 = vector.from_elements %235, %236, %237, %238 : vector<1x4xf16>
        %240 = vector.extract %147[24, %173] : f16 from vector<32x2xf16>
        %241 = vector.extract %147[25, %173] : f16 from vector<32x2xf16>
        %242 = vector.extract %147[26, %173] : f16 from vector<32x2xf16>
        %243 = vector.extract %147[27, %173] : f16 from vector<32x2xf16>
        %244 = vector.from_elements %240, %241, %242, %243 : vector<4x1xf16>
        %245 = frisk.warp_mma_rr %239, %244, %234 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %246 = vector.extract %139[%158, 28] : f16 from vector<2x32xf16>
        %247 = vector.extract %139[%158, 29] : f16 from vector<2x32xf16>
        %248 = vector.extract %139[%158, 30] : f16 from vector<2x32xf16>
        %249 = vector.extract %139[%158, 31] : f16 from vector<2x32xf16>
        %250 = vector.from_elements %246, %247, %248, %249 : vector<1x4xf16>
        %251 = vector.extract %147[28, %173] : f16 from vector<32x2xf16>
        %252 = vector.extract %147[29, %173] : f16 from vector<32x2xf16>
        %253 = vector.extract %147[30, %173] : f16 from vector<32x2xf16>
        %254 = vector.extract %147[31, %173] : f16 from vector<32x2xf16>
        %255 = vector.from_elements %251, %252, %253, %254 : vector<4x1xf16>
        %256 = frisk.warp_mma_rr %250, %255, %245 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %257 = vector.extract %256[0, 0] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %arg12 [%158, %159] : f32 into vector<2x8xf32>
        %259 = vector.extract %256[0, 1] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %258 [%158, %161] : f32 into vector<2x8xf32>
        %261 = vector.extract %256[0, 2] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %260 [%158, %163] : f32 into vector<2x8xf32>
        %263 = vector.extract %256[0, 3] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %262 [%158, %165] : f32 into vector<2x8xf32>
        affine.yield %264 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %149:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %159 = vector.extract %135#1[%158, 0] : f32 from vector<2x1xf32>
        %160 = vector.broadcast %159 : f32 to vector<1x4xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %162 = vector.insert %159, %arg12 [%158, %161] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %164 = vector.insert %159, %162 [%158, %163] : f32 into vector<2x8xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %166 = vector.insert %159, %164 [%158, %165] : f32 into vector<2x8xf32>
        %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %168 = vector.insert %159, %166 [%158, %167] : f32 into vector<2x8xf32>
        %169 = vector.extract %arg9[%158, %161] : f32 from vector<2x8xf32>
        %170 = vector.extract %arg9[%158, %163] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg9[%158, %165] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg9[%158, %167] : f32 from vector<2x8xf32>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf32>
        %174 = arith.mulf %173, %160 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg13 [%158, %161] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%158, %163] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%158, %165] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%158, %167] : f32 into vector<2x8xf32>
        affine.yield %168, %182 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %146 : !frisk.async_task
      %subview_19 = memref.subview %arg2[0, %141, %110, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %150 = frisk.async_copy_gm_to_shm %subview_19 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %151 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %158 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %159 = frisk.read_matrix_from_shm %view[%158] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %160 = vector.bitcast %159 : vector<4xi32> to vector<8xf16>
        %161 = vector.extract %160[0] : f16 from vector<8xf16>
        %162 = vector.extract %160[%88] : f16 from vector<8xf16>
        %163 = vector.extract %160[1] : f16 from vector<8xf16>
        %164 = vector.extract %160[%89] : f16 from vector<8xf16>
        %165 = vector.extract %160[2] : f16 from vector<8xf16>
        %166 = vector.extract %160[%90] : f16 from vector<8xf16>
        %167 = vector.extract %160[3] : f16 from vector<8xf16>
        %168 = vector.extract %160[%91] : f16 from vector<8xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %170 = vector.insert %161, %arg12 [%169, 0] : f16 into vector<32x2xf16>
        %171 = vector.insert %162, %170 [%169, 1] : f16 into vector<32x2xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %173 = vector.insert %163, %171 [%172, 0] : f16 into vector<32x2xf16>
        %174 = vector.insert %164, %173 [%172, 1] : f16 into vector<32x2xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %176 = vector.insert %165, %174 [%175, 0] : f16 into vector<32x2xf16>
        %177 = vector.insert %166, %176 [%175, 1] : f16 into vector<32x2xf16>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %179 = vector.insert %167, %177 [%178, 0] : f16 into vector<32x2xf16>
        %180 = vector.insert %168, %179 [%178, 1] : f16 into vector<32x2xf16>
        affine.yield %180 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %152 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %149#1) -> (vector<2x8xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %160 = vector.extract %149#1[%158, %159] : f32 from vector<2x8xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %162 = vector.extract %149#1[%158, %161] : f32 from vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %149#1[%158, %163] : f32 from vector<2x8xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %166 = vector.extract %149#1[%158, %165] : f32 from vector<2x8xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = vector.extract %139[%158, 0] : f16 from vector<2x32xf16>
        %169 = vector.extract %139[%158, 1] : f16 from vector<2x32xf16>
        %170 = vector.extract %139[%158, 2] : f16 from vector<2x32xf16>
        %171 = vector.extract %139[%158, 3] : f16 from vector<2x32xf16>
        %172 = vector.from_elements %168, %169, %170, %171 : vector<1x4xf16>
        %173 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %174 = vector.extract %151[0, %173] : f16 from vector<32x2xf16>
        %175 = vector.extract %151[1, %173] : f16 from vector<32x2xf16>
        %176 = vector.extract %151[2, %173] : f16 from vector<32x2xf16>
        %177 = vector.extract %151[3, %173] : f16 from vector<32x2xf16>
        %178 = vector.from_elements %174, %175, %176, %177 : vector<4x1xf16>
        %179 = frisk.warp_mma_rr %172, %178, %167 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %180 = vector.extract %139[%158, 4] : f16 from vector<2x32xf16>
        %181 = vector.extract %139[%158, 5] : f16 from vector<2x32xf16>
        %182 = vector.extract %139[%158, 6] : f16 from vector<2x32xf16>
        %183 = vector.extract %139[%158, 7] : f16 from vector<2x32xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<1x4xf16>
        %185 = vector.extract %151[4, %173] : f16 from vector<32x2xf16>
        %186 = vector.extract %151[5, %173] : f16 from vector<32x2xf16>
        %187 = vector.extract %151[6, %173] : f16 from vector<32x2xf16>
        %188 = vector.extract %151[7, %173] : f16 from vector<32x2xf16>
        %189 = vector.from_elements %185, %186, %187, %188 : vector<4x1xf16>
        %190 = frisk.warp_mma_rr %184, %189, %179 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %191 = vector.extract %139[%158, 8] : f16 from vector<2x32xf16>
        %192 = vector.extract %139[%158, 9] : f16 from vector<2x32xf16>
        %193 = vector.extract %139[%158, 10] : f16 from vector<2x32xf16>
        %194 = vector.extract %139[%158, 11] : f16 from vector<2x32xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<1x4xf16>
        %196 = vector.extract %151[8, %173] : f16 from vector<32x2xf16>
        %197 = vector.extract %151[9, %173] : f16 from vector<32x2xf16>
        %198 = vector.extract %151[10, %173] : f16 from vector<32x2xf16>
        %199 = vector.extract %151[11, %173] : f16 from vector<32x2xf16>
        %200 = vector.from_elements %196, %197, %198, %199 : vector<4x1xf16>
        %201 = frisk.warp_mma_rr %195, %200, %190 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %202 = vector.extract %139[%158, 12] : f16 from vector<2x32xf16>
        %203 = vector.extract %139[%158, 13] : f16 from vector<2x32xf16>
        %204 = vector.extract %139[%158, 14] : f16 from vector<2x32xf16>
        %205 = vector.extract %139[%158, 15] : f16 from vector<2x32xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<1x4xf16>
        %207 = vector.extract %151[12, %173] : f16 from vector<32x2xf16>
        %208 = vector.extract %151[13, %173] : f16 from vector<32x2xf16>
        %209 = vector.extract %151[14, %173] : f16 from vector<32x2xf16>
        %210 = vector.extract %151[15, %173] : f16 from vector<32x2xf16>
        %211 = vector.from_elements %207, %208, %209, %210 : vector<4x1xf16>
        %212 = frisk.warp_mma_rr %206, %211, %201 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %213 = vector.extract %139[%158, 16] : f16 from vector<2x32xf16>
        %214 = vector.extract %139[%158, 17] : f16 from vector<2x32xf16>
        %215 = vector.extract %139[%158, 18] : f16 from vector<2x32xf16>
        %216 = vector.extract %139[%158, 19] : f16 from vector<2x32xf16>
        %217 = vector.from_elements %213, %214, %215, %216 : vector<1x4xf16>
        %218 = vector.extract %151[16, %173] : f16 from vector<32x2xf16>
        %219 = vector.extract %151[17, %173] : f16 from vector<32x2xf16>
        %220 = vector.extract %151[18, %173] : f16 from vector<32x2xf16>
        %221 = vector.extract %151[19, %173] : f16 from vector<32x2xf16>
        %222 = vector.from_elements %218, %219, %220, %221 : vector<4x1xf16>
        %223 = frisk.warp_mma_rr %217, %222, %212 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %224 = vector.extract %139[%158, 20] : f16 from vector<2x32xf16>
        %225 = vector.extract %139[%158, 21] : f16 from vector<2x32xf16>
        %226 = vector.extract %139[%158, 22] : f16 from vector<2x32xf16>
        %227 = vector.extract %139[%158, 23] : f16 from vector<2x32xf16>
        %228 = vector.from_elements %224, %225, %226, %227 : vector<1x4xf16>
        %229 = vector.extract %151[20, %173] : f16 from vector<32x2xf16>
        %230 = vector.extract %151[21, %173] : f16 from vector<32x2xf16>
        %231 = vector.extract %151[22, %173] : f16 from vector<32x2xf16>
        %232 = vector.extract %151[23, %173] : f16 from vector<32x2xf16>
        %233 = vector.from_elements %229, %230, %231, %232 : vector<4x1xf16>
        %234 = frisk.warp_mma_rr %228, %233, %223 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %235 = vector.extract %139[%158, 24] : f16 from vector<2x32xf16>
        %236 = vector.extract %139[%158, 25] : f16 from vector<2x32xf16>
        %237 = vector.extract %139[%158, 26] : f16 from vector<2x32xf16>
        %238 = vector.extract %139[%158, 27] : f16 from vector<2x32xf16>
        %239 = vector.from_elements %235, %236, %237, %238 : vector<1x4xf16>
        %240 = vector.extract %151[24, %173] : f16 from vector<32x2xf16>
        %241 = vector.extract %151[25, %173] : f16 from vector<32x2xf16>
        %242 = vector.extract %151[26, %173] : f16 from vector<32x2xf16>
        %243 = vector.extract %151[27, %173] : f16 from vector<32x2xf16>
        %244 = vector.from_elements %240, %241, %242, %243 : vector<4x1xf16>
        %245 = frisk.warp_mma_rr %239, %244, %234 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %246 = vector.extract %139[%158, 28] : f16 from vector<2x32xf16>
        %247 = vector.extract %139[%158, 29] : f16 from vector<2x32xf16>
        %248 = vector.extract %139[%158, 30] : f16 from vector<2x32xf16>
        %249 = vector.extract %139[%158, 31] : f16 from vector<2x32xf16>
        %250 = vector.from_elements %246, %247, %248, %249 : vector<1x4xf16>
        %251 = vector.extract %151[28, %173] : f16 from vector<32x2xf16>
        %252 = vector.extract %151[29, %173] : f16 from vector<32x2xf16>
        %253 = vector.extract %151[30, %173] : f16 from vector<32x2xf16>
        %254 = vector.extract %151[31, %173] : f16 from vector<32x2xf16>
        %255 = vector.from_elements %251, %252, %253, %254 : vector<4x1xf16>
        %256 = frisk.warp_mma_rr %250, %255, %245 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %257 = vector.extract %256[0, 0] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %arg12 [%158, %159] : f32 into vector<2x8xf32>
        %259 = vector.extract %256[0, 1] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %258 [%158, %161] : f32 into vector<2x8xf32>
        %261 = vector.extract %256[0, 2] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %260 [%158, %163] : f32 into vector<2x8xf32>
        %263 = vector.extract %256[0, 3] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %262 [%158, %165] : f32 into vector<2x8xf32>
        affine.yield %264 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %153:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %159 = vector.extract %135#1[%158, 0] : f32 from vector<2x1xf32>
        %160 = vector.broadcast %159 : f32 to vector<1x4xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %162 = vector.insert %159, %arg12 [%158, %161] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %164 = vector.insert %159, %162 [%158, %163] : f32 into vector<2x8xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %166 = vector.insert %159, %164 [%158, %165] : f32 into vector<2x8xf32>
        %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %168 = vector.insert %159, %166 [%158, %167] : f32 into vector<2x8xf32>
        %169 = vector.extract %arg10[%158, %161] : f32 from vector<2x8xf32>
        %170 = vector.extract %arg10[%158, %163] : f32 from vector<2x8xf32>
        %171 = vector.extract %arg10[%158, %165] : f32 from vector<2x8xf32>
        %172 = vector.extract %arg10[%158, %167] : f32 from vector<2x8xf32>
        %173 = vector.from_elements %169, %170, %171, %172 : vector<1x4xf32>
        %174 = arith.mulf %173, %160 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %175 = vector.extract %174[0, 0] : f32 from vector<1x4xf32>
        %176 = vector.insert %175, %arg13 [%158, %161] : f32 into vector<2x8xf32>
        %177 = vector.extract %174[0, 1] : f32 from vector<1x4xf32>
        %178 = vector.insert %177, %176 [%158, %163] : f32 into vector<2x8xf32>
        %179 = vector.extract %174[0, 2] : f32 from vector<1x4xf32>
        %180 = vector.insert %179, %178 [%158, %165] : f32 into vector<2x8xf32>
        %181 = vector.extract %174[0, 3] : f32 from vector<1x4xf32>
        %182 = vector.insert %181, %180 [%158, %167] : f32 into vector<2x8xf32>
        affine.yield %168, %182 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %150 : !frisk.async_task
      %154 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %158 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %159 = frisk.read_matrix_from_shm %view_8[%158] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %160 = vector.bitcast %159 : vector<4xi32> to vector<8xf16>
        %161 = vector.extract %160[0] : f16 from vector<8xf16>
        %162 = vector.extract %160[%88] : f16 from vector<8xf16>
        %163 = vector.extract %160[1] : f16 from vector<8xf16>
        %164 = vector.extract %160[%89] : f16 from vector<8xf16>
        %165 = vector.extract %160[2] : f16 from vector<8xf16>
        %166 = vector.extract %160[%90] : f16 from vector<8xf16>
        %167 = vector.extract %160[3] : f16 from vector<8xf16>
        %168 = vector.extract %160[%91] : f16 from vector<8xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %170 = vector.insert %161, %arg12 [%169, 0] : f16 into vector<32x2xf16>
        %171 = vector.insert %162, %170 [%169, 1] : f16 into vector<32x2xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %173 = vector.insert %163, %171 [%172, 0] : f16 into vector<32x2xf16>
        %174 = vector.insert %164, %173 [%172, 1] : f16 into vector<32x2xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %176 = vector.insert %165, %174 [%175, 0] : f16 into vector<32x2xf16>
        %177 = vector.insert %166, %176 [%175, 1] : f16 into vector<32x2xf16>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %179 = vector.insert %167, %177 [%178, 0] : f16 into vector<32x2xf16>
        %180 = vector.insert %168, %179 [%178, 1] : f16 into vector<32x2xf16>
        affine.yield %180 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %155:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %153#1, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %160 = vector.extract %153#1[%158, %159] : f32 from vector<2x8xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %162 = vector.extract %153#1[%158, %161] : f32 from vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %153#1[%158, %163] : f32 from vector<2x8xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %166 = vector.extract %153#1[%158, %165] : f32 from vector<2x8xf32>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<1x4xf32>
        %168 = vector.extract %139[%158, 0] : f16 from vector<2x32xf16>
        %169 = vector.extract %139[%158, 1] : f16 from vector<2x32xf16>
        %170 = vector.extract %139[%158, 2] : f16 from vector<2x32xf16>
        %171 = vector.extract %139[%158, 3] : f16 from vector<2x32xf16>
        %172 = vector.from_elements %168, %169, %170, %171 : vector<1x4xf16>
        %173 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %174 = vector.extract %154[0, %173] : f16 from vector<32x2xf16>
        %175 = vector.extract %154[1, %173] : f16 from vector<32x2xf16>
        %176 = vector.extract %154[2, %173] : f16 from vector<32x2xf16>
        %177 = vector.extract %154[3, %173] : f16 from vector<32x2xf16>
        %178 = vector.from_elements %174, %175, %176, %177 : vector<4x1xf16>
        %179 = frisk.warp_mma_rr %172, %178, %167 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %180 = vector.extract %139[%158, 4] : f16 from vector<2x32xf16>
        %181 = vector.extract %139[%158, 5] : f16 from vector<2x32xf16>
        %182 = vector.extract %139[%158, 6] : f16 from vector<2x32xf16>
        %183 = vector.extract %139[%158, 7] : f16 from vector<2x32xf16>
        %184 = vector.from_elements %180, %181, %182, %183 : vector<1x4xf16>
        %185 = vector.extract %154[4, %173] : f16 from vector<32x2xf16>
        %186 = vector.extract %154[5, %173] : f16 from vector<32x2xf16>
        %187 = vector.extract %154[6, %173] : f16 from vector<32x2xf16>
        %188 = vector.extract %154[7, %173] : f16 from vector<32x2xf16>
        %189 = vector.from_elements %185, %186, %187, %188 : vector<4x1xf16>
        %190 = frisk.warp_mma_rr %184, %189, %179 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %191 = vector.extract %139[%158, 8] : f16 from vector<2x32xf16>
        %192 = vector.extract %139[%158, 9] : f16 from vector<2x32xf16>
        %193 = vector.extract %139[%158, 10] : f16 from vector<2x32xf16>
        %194 = vector.extract %139[%158, 11] : f16 from vector<2x32xf16>
        %195 = vector.from_elements %191, %192, %193, %194 : vector<1x4xf16>
        %196 = vector.extract %154[8, %173] : f16 from vector<32x2xf16>
        %197 = vector.extract %154[9, %173] : f16 from vector<32x2xf16>
        %198 = vector.extract %154[10, %173] : f16 from vector<32x2xf16>
        %199 = vector.extract %154[11, %173] : f16 from vector<32x2xf16>
        %200 = vector.from_elements %196, %197, %198, %199 : vector<4x1xf16>
        %201 = frisk.warp_mma_rr %195, %200, %190 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %202 = vector.extract %139[%158, 12] : f16 from vector<2x32xf16>
        %203 = vector.extract %139[%158, 13] : f16 from vector<2x32xf16>
        %204 = vector.extract %139[%158, 14] : f16 from vector<2x32xf16>
        %205 = vector.extract %139[%158, 15] : f16 from vector<2x32xf16>
        %206 = vector.from_elements %202, %203, %204, %205 : vector<1x4xf16>
        %207 = vector.extract %154[12, %173] : f16 from vector<32x2xf16>
        %208 = vector.extract %154[13, %173] : f16 from vector<32x2xf16>
        %209 = vector.extract %154[14, %173] : f16 from vector<32x2xf16>
        %210 = vector.extract %154[15, %173] : f16 from vector<32x2xf16>
        %211 = vector.from_elements %207, %208, %209, %210 : vector<4x1xf16>
        %212 = frisk.warp_mma_rr %206, %211, %201 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %213 = vector.extract %139[%158, 16] : f16 from vector<2x32xf16>
        %214 = vector.extract %139[%158, 17] : f16 from vector<2x32xf16>
        %215 = vector.extract %139[%158, 18] : f16 from vector<2x32xf16>
        %216 = vector.extract %139[%158, 19] : f16 from vector<2x32xf16>
        %217 = vector.from_elements %213, %214, %215, %216 : vector<1x4xf16>
        %218 = vector.extract %154[16, %173] : f16 from vector<32x2xf16>
        %219 = vector.extract %154[17, %173] : f16 from vector<32x2xf16>
        %220 = vector.extract %154[18, %173] : f16 from vector<32x2xf16>
        %221 = vector.extract %154[19, %173] : f16 from vector<32x2xf16>
        %222 = vector.from_elements %218, %219, %220, %221 : vector<4x1xf16>
        %223 = frisk.warp_mma_rr %217, %222, %212 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %224 = vector.extract %139[%158, 20] : f16 from vector<2x32xf16>
        %225 = vector.extract %139[%158, 21] : f16 from vector<2x32xf16>
        %226 = vector.extract %139[%158, 22] : f16 from vector<2x32xf16>
        %227 = vector.extract %139[%158, 23] : f16 from vector<2x32xf16>
        %228 = vector.from_elements %224, %225, %226, %227 : vector<1x4xf16>
        %229 = vector.extract %154[20, %173] : f16 from vector<32x2xf16>
        %230 = vector.extract %154[21, %173] : f16 from vector<32x2xf16>
        %231 = vector.extract %154[22, %173] : f16 from vector<32x2xf16>
        %232 = vector.extract %154[23, %173] : f16 from vector<32x2xf16>
        %233 = vector.from_elements %229, %230, %231, %232 : vector<4x1xf16>
        %234 = frisk.warp_mma_rr %228, %233, %223 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %235 = vector.extract %139[%158, 24] : f16 from vector<2x32xf16>
        %236 = vector.extract %139[%158, 25] : f16 from vector<2x32xf16>
        %237 = vector.extract %139[%158, 26] : f16 from vector<2x32xf16>
        %238 = vector.extract %139[%158, 27] : f16 from vector<2x32xf16>
        %239 = vector.from_elements %235, %236, %237, %238 : vector<1x4xf16>
        %240 = vector.extract %154[24, %173] : f16 from vector<32x2xf16>
        %241 = vector.extract %154[25, %173] : f16 from vector<32x2xf16>
        %242 = vector.extract %154[26, %173] : f16 from vector<32x2xf16>
        %243 = vector.extract %154[27, %173] : f16 from vector<32x2xf16>
        %244 = vector.from_elements %240, %241, %242, %243 : vector<4x1xf16>
        %245 = frisk.warp_mma_rr %239, %244, %234 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %246 = vector.extract %139[%158, 28] : f16 from vector<2x32xf16>
        %247 = vector.extract %139[%158, 29] : f16 from vector<2x32xf16>
        %248 = vector.extract %139[%158, 30] : f16 from vector<2x32xf16>
        %249 = vector.extract %139[%158, 31] : f16 from vector<2x32xf16>
        %250 = vector.from_elements %246, %247, %248, %249 : vector<1x4xf16>
        %251 = vector.extract %154[28, %173] : f16 from vector<32x2xf16>
        %252 = vector.extract %154[29, %173] : f16 from vector<32x2xf16>
        %253 = vector.extract %154[30, %173] : f16 from vector<32x2xf16>
        %254 = vector.extract %154[31, %173] : f16 from vector<32x2xf16>
        %255 = vector.from_elements %251, %252, %253, %254 : vector<4x1xf16>
        %256 = frisk.warp_mma_rr %250, %255, %245 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %257 = vector.extract %256[0, 0] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %arg12 [%158, %159] : f32 into vector<2x8xf32>
        %259 = vector.extract %256[0, 1] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %258 [%158, %161] : f32 into vector<2x8xf32>
        %261 = vector.extract %256[0, 2] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %260 [%158, %163] : f32 into vector<2x8xf32>
        %263 = vector.extract %256[0, 3] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %262 [%158, %165] : f32 into vector<2x8xf32>
        %265 = vector.extract %144[%158, %159] : f32 from vector<2x8xf32>
        %266 = vector.extract %144[%158, %161] : f32 from vector<2x8xf32>
        %267 = vector.extract %144[%158, %163] : f32 from vector<2x8xf32>
        %268 = vector.extract %144[%158, %165] : f32 from vector<2x8xf32>
        %269 = vector.insert %265, %arg13 [%158, %159] : f32 into vector<2x8xf32>
        %270 = vector.insert %266, %269 [%158, %161] : f32 into vector<2x8xf32>
        %271 = vector.insert %267, %270 [%158, %163] : f32 into vector<2x8xf32>
        %272 = vector.insert %268, %271 [%158, %165] : f32 into vector<2x8xf32>
        affine.yield %264, %272 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %156:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %158 = vector.extract %138[%arg11, 0] : f32 from vector<2x1xf32>
        %159 = vector.insert %158, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %160 = vector.extract %133[%arg11, 0] : f32 from vector<2x1xf32>
        %161 = vector.insert %160, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %159, %161 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %157:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5, %arg14 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %158 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %160 = vector.extract %148[%158, %159] : f32 from vector<2x8xf32>
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %162 = vector.extract %148[%158, %161] : f32 from vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %148[%158, %163] : f32 from vector<2x8xf32>
        %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %166 = vector.extract %148[%158, %165] : f32 from vector<2x8xf32>
        %167 = vector.insert %160, %arg12 [%158, %159] : f32 into vector<2x8xf32>
        %168 = vector.insert %162, %167 [%158, %161] : f32 into vector<2x8xf32>
        %169 = vector.insert %164, %168 [%158, %163] : f32 into vector<2x8xf32>
        %170 = vector.insert %166, %169 [%158, %165] : f32 into vector<2x8xf32>
        %171 = vector.extract %152[%158, %159] : f32 from vector<2x8xf32>
        %172 = vector.extract %152[%158, %161] : f32 from vector<2x8xf32>
        %173 = vector.extract %152[%158, %163] : f32 from vector<2x8xf32>
        %174 = vector.extract %152[%158, %165] : f32 from vector<2x8xf32>
        %175 = vector.insert %171, %arg13 [%158, %159] : f32 into vector<2x8xf32>
        %176 = vector.insert %172, %175 [%158, %161] : f32 into vector<2x8xf32>
        %177 = vector.insert %173, %176 [%158, %163] : f32 into vector<2x8xf32>
        %178 = vector.insert %174, %177 [%158, %165] : f32 into vector<2x8xf32>
        %179 = vector.extract %155#0[%158, %159] : f32 from vector<2x8xf32>
        %180 = vector.extract %155#0[%158, %161] : f32 from vector<2x8xf32>
        %181 = vector.extract %155#0[%158, %163] : f32 from vector<2x8xf32>
        %182 = vector.extract %155#0[%158, %165] : f32 from vector<2x8xf32>
        %183 = vector.insert %179, %arg14 [%158, %159] : f32 into vector<2x8xf32>
        %184 = vector.insert %180, %183 [%158, %161] : f32 into vector<2x8xf32>
        %185 = vector.insert %181, %184 [%158, %163] : f32 into vector<2x8xf32>
        %186 = vector.insert %182, %185 [%158, %165] : f32 into vector<2x8xf32>
        affine.yield %170, %178, %186 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      gpu.barrier
      affine.yield %155#1, %156#0, %156#1, %157#0, %157#1, %157#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %93 = arith.index_cast %thread_id_x : index to i32
    %94 = arith.andi %93, %c16_i32 : i32
    %95 = arith.cmpi ne, %94, %c0_i32 : i32
    %96 = arith.andi %93, %c32_i32 : i32
    %97 = arith.cmpi ne, %96, %c0_i32 : i32
    %98 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %110 = vector.extract %92#5[%109, %108] : f32 from vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %92#5[%109, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %114 = vector.extract %92#5[%109, %113] : f32 from vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %116 = vector.extract %92#5[%109, %115] : f32 from vector<2x8xf32>
      %117 = arith.select %95, %110, %112 : f32
      %shuffleResult, %valid = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %95, %shuffleResult, %110 : f32
      %119 = arith.select %95, %112, %shuffleResult : f32
      %120 = arith.select %95, %114, %116 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %120, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %95, %shuffleResult_12, %114 : f32
      %122 = arith.select %95, %116, %shuffleResult_12 : f32
      %123 = arith.select %97, %118, %121 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %97, %shuffleResult_14, %118 : f32
      %125 = arith.select %97, %121, %shuffleResult_14 : f32
      %126 = arith.select %97, %119, %122 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %126, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %127 = arith.select %97, %shuffleResult_16, %119 : f32
      %128 = arith.select %97, %122, %shuffleResult_16 : f32
      %129 = vector.insert %124, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %130 = vector.insert %127, %129 [%109, %111] : f32 into vector<2x8xf32>
      %131 = vector.insert %125, %130 [%109, %113] : f32 into vector<2x8xf32>
      %132 = vector.insert %128, %131 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %132 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %99 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %110 = vector.extract %92#4[%109, %108] : f32 from vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %92#4[%109, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %114 = vector.extract %92#4[%109, %113] : f32 from vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %116 = vector.extract %92#4[%109, %115] : f32 from vector<2x8xf32>
      %117 = arith.select %95, %110, %112 : f32
      %shuffleResult, %valid = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %95, %shuffleResult, %110 : f32
      %119 = arith.select %95, %112, %shuffleResult : f32
      %120 = arith.select %95, %114, %116 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %120, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %95, %shuffleResult_12, %114 : f32
      %122 = arith.select %95, %116, %shuffleResult_12 : f32
      %123 = arith.select %97, %118, %121 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %97, %shuffleResult_14, %118 : f32
      %125 = arith.select %97, %121, %shuffleResult_14 : f32
      %126 = arith.select %97, %119, %122 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %126, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %127 = arith.select %97, %shuffleResult_16, %119 : f32
      %128 = arith.select %97, %122, %shuffleResult_16 : f32
      %129 = vector.insert %124, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %130 = vector.insert %127, %129 [%109, %111] : f32 into vector<2x8xf32>
      %131 = vector.insert %125, %130 [%109, %113] : f32 into vector<2x8xf32>
      %132 = vector.insert %128, %131 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %132 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %100 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %110 = vector.extract %92#3[%109, %108] : f32 from vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %92#3[%109, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %114 = vector.extract %92#3[%109, %113] : f32 from vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %116 = vector.extract %92#3[%109, %115] : f32 from vector<2x8xf32>
      %117 = arith.select %95, %110, %112 : f32
      %shuffleResult, %valid = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %95, %shuffleResult, %110 : f32
      %119 = arith.select %95, %112, %shuffleResult : f32
      %120 = arith.select %95, %114, %116 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %120, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %95, %shuffleResult_12, %114 : f32
      %122 = arith.select %95, %116, %shuffleResult_12 : f32
      %123 = arith.select %97, %118, %121 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %97, %shuffleResult_14, %118 : f32
      %125 = arith.select %97, %121, %shuffleResult_14 : f32
      %126 = arith.select %97, %119, %122 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %126, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %127 = arith.select %97, %shuffleResult_16, %119 : f32
      %128 = arith.select %97, %122, %shuffleResult_16 : f32
      %129 = vector.insert %124, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %130 = vector.insert %127, %129 [%109, %111] : f32 into vector<2x8xf32>
      %131 = vector.insert %125, %130 [%109, %113] : f32 into vector<2x8xf32>
      %132 = vector.insert %128, %131 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %132 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %101 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %110 = vector.extract %92#0[%109, %108] : f32 from vector<2x8xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %92#0[%109, %111] : f32 from vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %114 = vector.extract %92#0[%109, %113] : f32 from vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %116 = vector.extract %92#0[%109, %115] : f32 from vector<2x8xf32>
      %117 = arith.select %95, %110, %112 : f32
      %shuffleResult, %valid = gpu.shuffle xor %117, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %118 = arith.select %95, %shuffleResult, %110 : f32
      %119 = arith.select %95, %112, %shuffleResult : f32
      %120 = arith.select %95, %114, %116 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %120, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %121 = arith.select %95, %shuffleResult_12, %114 : f32
      %122 = arith.select %95, %116, %shuffleResult_12 : f32
      %123 = arith.select %97, %118, %121 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %123, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %124 = arith.select %97, %shuffleResult_14, %118 : f32
      %125 = arith.select %97, %121, %shuffleResult_14 : f32
      %126 = arith.select %97, %119, %122 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %126, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %127 = arith.select %97, %shuffleResult_16, %119 : f32
      %128 = arith.select %97, %122, %shuffleResult_16 : f32
      %129 = vector.insert %124, %arg5 [%109, %108] : f32 into vector<2x8xf32>
      %130 = vector.insert %127, %129 [%109, %111] : f32 into vector<2x8xf32>
      %131 = vector.insert %125, %130 [%109, %113] : f32 into vector<2x8xf32>
      %132 = vector.insert %128, %131 [%109, %115] : f32 into vector<2x8xf32>
      affine.yield %132 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %102:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %109 = vector.extract %92#1[%108, 0] : f32 from vector<2x1xf32>
      %110 = vector.broadcast %109 : f32 to vector<1x4xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %112 = vector.insert %109, %arg5 [%108, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %114 = vector.insert %109, %112 [%108, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %116 = vector.insert %109, %114 [%108, %115] : f32 into vector<2x8xf32>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.insert %109, %116 [%108, %117] : f32 into vector<2x8xf32>
      %119 = vector.extract %101[%108, %111] : f32 from vector<2x8xf32>
      %120 = vector.extract %101[%108, %113] : f32 from vector<2x8xf32>
      %121 = vector.extract %101[%108, %115] : f32 from vector<2x8xf32>
      %122 = vector.extract %101[%108, %117] : f32 from vector<2x8xf32>
      %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf32>
      %124 = arith.divf %123, %110 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %125 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %arg6 [%108, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%108, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%108, %115] : f32 into vector<2x8xf32>
      %131 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
      %132 = vector.insert %131, %130 [%108, %117] : f32 into vector<2x8xf32>
      %133 = arith.truncf %124 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%108, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%108, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%108, %115] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%108, %117] : f16 into vector<2x8xf16>
      %142 = vector.insert %134, %arg8 [%108, %111] : f16 into vector<2x8xf16>
      %143 = vector.insert %136, %142 [%108, %113] : f16 into vector<2x8xf16>
      %144 = vector.insert %138, %143 [%108, %115] : f16 into vector<2x8xf16>
      %145 = vector.insert %140, %144 [%108, %117] : f16 into vector<2x8xf16>
      affine.yield %118, %132, %141, %145 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %103 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %104 = arith.cmpi ult, %103, %c4 : index
    scf.if %104 {
      affine.for %arg4 = 0 to 4 {
        %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %110 = vector.extract %102#3[%108, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %112 = vector.extract %102#3[%108, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %114 = vector.extract %102#3[%108, %113] : f16 from vector<2x8xf16>
        %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %116 = vector.extract %102#3[%108, %115] : f16 from vector<2x8xf16>
        affine.store %110, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %116, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %105:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %109 = vector.extract %92#1[%108, 0] : f32 from vector<2x1xf32>
      %110 = vector.broadcast %109 : f32 to vector<1x4xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %112 = vector.insert %109, %arg5 [%108, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %114 = vector.insert %109, %112 [%108, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %116 = vector.insert %109, %114 [%108, %115] : f32 into vector<2x8xf32>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.insert %109, %116 [%108, %117] : f32 into vector<2x8xf32>
      %119 = vector.extract %100[%108, %111] : f32 from vector<2x8xf32>
      %120 = vector.extract %100[%108, %113] : f32 from vector<2x8xf32>
      %121 = vector.extract %100[%108, %115] : f32 from vector<2x8xf32>
      %122 = vector.extract %100[%108, %117] : f32 from vector<2x8xf32>
      %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf32>
      %124 = arith.divf %123, %110 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %125 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %arg6 [%108, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%108, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%108, %115] : f32 into vector<2x8xf32>
      %131 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
      %132 = vector.insert %131, %130 [%108, %117] : f32 into vector<2x8xf32>
      %133 = arith.truncf %124 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%108, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%108, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%108, %115] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%108, %117] : f16 into vector<2x8xf16>
      %142 = vector.insert %134, %arg8 [%108, %111] : f16 into vector<2x8xf16>
      %143 = vector.insert %136, %142 [%108, %113] : f16 into vector<2x8xf16>
      %144 = vector.insert %138, %143 [%108, %115] : f16 into vector<2x8xf16>
      %145 = vector.insert %140, %144 [%108, %117] : f16 into vector<2x8xf16>
      affine.yield %118, %132, %141, %145 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %104 {
      affine.for %arg4 = 0 to 4 {
        %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %110 = vector.extract %105#3[%108, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %112 = vector.extract %105#3[%108, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %114 = vector.extract %105#3[%108, %113] : f16 from vector<2x8xf16>
        %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %116 = vector.extract %105#3[%108, %115] : f16 from vector<2x8xf16>
        affine.store %110, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %116, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %106:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %109 = vector.extract %92#1[%108, 0] : f32 from vector<2x1xf32>
      %110 = vector.broadcast %109 : f32 to vector<1x4xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %112 = vector.insert %109, %arg5 [%108, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %114 = vector.insert %109, %112 [%108, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %116 = vector.insert %109, %114 [%108, %115] : f32 into vector<2x8xf32>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.insert %109, %116 [%108, %117] : f32 into vector<2x8xf32>
      %119 = vector.extract %99[%108, %111] : f32 from vector<2x8xf32>
      %120 = vector.extract %99[%108, %113] : f32 from vector<2x8xf32>
      %121 = vector.extract %99[%108, %115] : f32 from vector<2x8xf32>
      %122 = vector.extract %99[%108, %117] : f32 from vector<2x8xf32>
      %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf32>
      %124 = arith.divf %123, %110 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %125 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %arg6 [%108, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%108, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%108, %115] : f32 into vector<2x8xf32>
      %131 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
      %132 = vector.insert %131, %130 [%108, %117] : f32 into vector<2x8xf32>
      %133 = arith.truncf %124 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%108, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%108, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%108, %115] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%108, %117] : f16 into vector<2x8xf16>
      %142 = vector.insert %134, %arg8 [%108, %111] : f16 into vector<2x8xf16>
      %143 = vector.insert %136, %142 [%108, %113] : f16 into vector<2x8xf16>
      %144 = vector.insert %138, %143 [%108, %115] : f16 into vector<2x8xf16>
      %145 = vector.insert %140, %144 [%108, %117] : f16 into vector<2x8xf16>
      affine.yield %118, %132, %141, %145 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %104 {
      affine.for %arg4 = 0 to 4 {
        %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %110 = vector.extract %106#3[%108, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %112 = vector.extract %106#3[%108, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %114 = vector.extract %106#3[%108, %113] : f16 from vector<2x8xf16>
        %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %116 = vector.extract %106#3[%108, %115] : f16 from vector<2x8xf16>
        affine.store %110, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %116, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %107:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %109 = vector.extract %92#1[%108, 0] : f32 from vector<2x1xf32>
      %110 = vector.broadcast %109 : f32 to vector<1x4xf32>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %112 = vector.insert %109, %arg5 [%108, %111] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %114 = vector.insert %109, %112 [%108, %113] : f32 into vector<2x8xf32>
      %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %116 = vector.insert %109, %114 [%108, %115] : f32 into vector<2x8xf32>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.insert %109, %116 [%108, %117] : f32 into vector<2x8xf32>
      %119 = vector.extract %98[%108, %111] : f32 from vector<2x8xf32>
      %120 = vector.extract %98[%108, %113] : f32 from vector<2x8xf32>
      %121 = vector.extract %98[%108, %115] : f32 from vector<2x8xf32>
      %122 = vector.extract %98[%108, %117] : f32 from vector<2x8xf32>
      %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf32>
      %124 = arith.divf %123, %110 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %125 = vector.extract %124[0, 0] : f32 from vector<1x4xf32>
      %126 = vector.insert %125, %arg6 [%108, %111] : f32 into vector<2x8xf32>
      %127 = vector.extract %124[0, 1] : f32 from vector<1x4xf32>
      %128 = vector.insert %127, %126 [%108, %113] : f32 into vector<2x8xf32>
      %129 = vector.extract %124[0, 2] : f32 from vector<1x4xf32>
      %130 = vector.insert %129, %128 [%108, %115] : f32 into vector<2x8xf32>
      %131 = vector.extract %124[0, 3] : f32 from vector<1x4xf32>
      %132 = vector.insert %131, %130 [%108, %117] : f32 into vector<2x8xf32>
      %133 = arith.truncf %124 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %134 = vector.extract %133[0, 0] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %arg7 [%108, %111] : f16 into vector<2x8xf16>
      %136 = vector.extract %133[0, 1] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%108, %113] : f16 into vector<2x8xf16>
      %138 = vector.extract %133[0, 2] : f16 from vector<1x4xf16>
      %139 = vector.insert %138, %137 [%108, %115] : f16 into vector<2x8xf16>
      %140 = vector.extract %133[0, 3] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %139 [%108, %117] : f16 into vector<2x8xf16>
      %142 = vector.insert %134, %arg8 [%108, %111] : f16 into vector<2x8xf16>
      %143 = vector.insert %136, %142 [%108, %113] : f16 into vector<2x8xf16>
      %144 = vector.insert %138, %143 [%108, %115] : f16 into vector<2x8xf16>
      %145 = vector.insert %140, %144 [%108, %117] : f16 into vector<2x8xf16>
      affine.yield %118, %132, %141, %145 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %104 {
      affine.for %arg4 = 0 to 4 {
        %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %110 = vector.extract %107#3[%108, %109] : f16 from vector<2x8xf16>
        %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %112 = vector.extract %107#3[%108, %111] : f16 from vector<2x8xf16>
        %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %114 = vector.extract %107#3[%108, %113] : f16 from vector<2x8xf16>
        %115 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %116 = vector.extract %107#3[%108, %115] : f16 from vector<2x8xf16>
        affine.store %110, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %112, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %114, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %116, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
