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
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %c4 = arith.constant 4 : index
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_7 = arith.constant 0.000000e+00 : f32
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
      %107 = vector.insert %cst_7, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %108 = vector.insert %cst_0, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %109 = vector.insert %cst_0, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %110 = vector.insert %cst_0, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %107, %108, %109, %110 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %13 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %109 = vector.insert %cst_7, %arg5 [%108, %107] : f32 into vector<2x8xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %111 = vector.insert %cst_7, %109 [%108, %110] : f32 into vector<2x8xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %113 = vector.insert %cst_7, %111 [%108, %112] : f32 into vector<2x8xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %115 = vector.insert %cst_7, %113 [%108, %114] : f32 into vector<2x8xf32>
      affine.yield %115 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %14 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_1) -> (vector<2x1xf32>) {
      %107 = vector.insert %cst_7, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %107 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    frisk.wait_buffer_ready %10 : !frisk.async_task
    %view_8 = memref.view %alloc[%c8192][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %15 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%6, %7]
    %subview_9 = memref.subview %arg0[0, %15, %9, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %16 = frisk.async_copy_gm_to_shm %subview_9 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
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
      %107 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %108 = memref.load %view[%107] : memref<4096xf16, 3>
      %109 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %110 = memref.load %view[%109] : memref<4096xf16, 3>
      %111 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %112 = memref.load %view[%111] : memref<4096xf16, 3>
      %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %114 = memref.load %view[%113] : memref<4096xf16, 3>
      %115 = vector.from_elements %108, %110, %112, %114 : vector<1x4xf16>
      %116 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %118 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg5 [%116, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %121 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%116, %120] : f16 into vector<2x8xf16>
      %123 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %124 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %122 [%116, %123] : f16 into vector<2x8xf16>
      %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %127 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %125 [%116, %126] : f16 into vector<2x8xf16>
      %129 = vector.insert %11, %arg6 [%116, %117] : f16 into vector<2x8xf16>
      %130 = vector.insert %11, %129 [%116, %120] : f16 into vector<2x8xf16>
      %131 = vector.insert %11, %130 [%116, %123] : f16 into vector<2x8xf16>
      %132 = vector.insert %11, %131 [%116, %126] : f16 into vector<2x8xf16>
      %133 = arith.mulf %arg8, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
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
    %subview_10 = memref.subview %arg0[0, %15, %9, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %37 = frisk.async_copy_gm_to_shm %subview_10 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %38 = memref.load %view_8[%17] : memref<4096xf16, 3>
    %39 = memref.load %view_8[%19] : memref<4096xf16, 3>
    %40 = memref.load %view_8[%21] : memref<4096xf16, 3>
    %41 = memref.load %view_8[%23] : memref<4096xf16, 3>
    %42 = vector.from_elements %38, %39, %40, %41 : vector<1x4xf16>
    %43:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %42) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %107 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %108 = memref.load %view_8[%107] : memref<4096xf16, 3>
      %109 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %110 = memref.load %view_8[%109] : memref<4096xf16, 3>
      %111 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %112 = memref.load %view_8[%111] : memref<4096xf16, 3>
      %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %114 = memref.load %view_8[%113] : memref<4096xf16, 3>
      %115 = vector.from_elements %108, %110, %112, %114 : vector<1x4xf16>
      %116 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %118 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg5 [%116, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %121 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%116, %120] : f16 into vector<2x8xf16>
      %123 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %124 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %122 [%116, %123] : f16 into vector<2x8xf16>
      %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %127 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %125 [%116, %126] : f16 into vector<2x8xf16>
      %129 = vector.insert %11, %arg6 [%116, %117] : f16 into vector<2x8xf16>
      %130 = vector.insert %11, %129 [%116, %120] : f16 into vector<2x8xf16>
      %131 = vector.insert %11, %130 [%116, %123] : f16 into vector<2x8xf16>
      %132 = vector.insert %11, %131 [%116, %126] : f16 into vector<2x8xf16>
      %133 = arith.mulf %arg8, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
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
    %subview_11 = memref.subview %arg0[0, %15, %9, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %53 = frisk.async_copy_gm_to_shm %subview_11 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %54 = memref.load %view[%17] : memref<4096xf16, 3>
    %55 = memref.load %view[%19] : memref<4096xf16, 3>
    %56 = memref.load %view[%21] : memref<4096xf16, 3>
    %57 = memref.load %view[%23] : memref<4096xf16, 3>
    %58 = vector.from_elements %54, %55, %56, %57 : vector<1x4xf16>
    %59:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %58) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %107 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %108 = memref.load %view[%107] : memref<4096xf16, 3>
      %109 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %110 = memref.load %view[%109] : memref<4096xf16, 3>
      %111 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %112 = memref.load %view[%111] : memref<4096xf16, 3>
      %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %114 = memref.load %view[%113] : memref<4096xf16, 3>
      %115 = vector.from_elements %108, %110, %112, %114 : vector<1x4xf16>
      %116 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %118 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg5 [%116, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %121 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%116, %120] : f16 into vector<2x8xf16>
      %123 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %124 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %122 [%116, %123] : f16 into vector<2x8xf16>
      %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %127 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %125 [%116, %126] : f16 into vector<2x8xf16>
      %129 = vector.insert %11, %arg6 [%116, %117] : f16 into vector<2x8xf16>
      %130 = vector.insert %11, %129 [%116, %120] : f16 into vector<2x8xf16>
      %131 = vector.insert %11, %130 [%116, %123] : f16 into vector<2x8xf16>
      %132 = vector.insert %11, %131 [%116, %126] : f16 into vector<2x8xf16>
      %133 = arith.mulf %arg8, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
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
    %69 = memref.load %view_8[%17] : memref<4096xf16, 3>
    %70 = memref.load %view_8[%19] : memref<4096xf16, 3>
    %71 = memref.load %view_8[%21] : memref<4096xf16, 3>
    %72 = memref.load %view_8[%23] : memref<4096xf16, 3>
    %73 = vector.from_elements %69, %70, %71, %72 : vector<1x4xf16>
    %74:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %73) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %107 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg4)[%thread_id_x]
      %108 = memref.load %view_8[%107] : memref<4096xf16, 3>
      %109 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg4)[%thread_id_x]
      %110 = memref.load %view_8[%109] : memref<4096xf16, 3>
      %111 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg4)[%thread_id_x]
      %112 = memref.load %view_8[%111] : memref<4096xf16, 3>
      %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg4)[%thread_id_x]
      %114 = memref.load %view_8[%113] : memref<4096xf16, 3>
      %115 = vector.from_elements %108, %110, %112, %114 : vector<1x4xf16>
      %116 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %118 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg5 [%116, %117] : f16 into vector<2x8xf16>
      %120 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %121 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %119 [%116, %120] : f16 into vector<2x8xf16>
      %123 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %124 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %122 [%116, %123] : f16 into vector<2x8xf16>
      %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %127 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %125 [%116, %126] : f16 into vector<2x8xf16>
      %129 = vector.insert %11, %arg6 [%116, %117] : f16 into vector<2x8xf16>
      %130 = vector.insert %11, %129 [%116, %120] : f16 into vector<2x8xf16>
      %131 = vector.insert %11, %130 [%116, %123] : f16 into vector<2x8xf16>
      %132 = vector.insert %11, %131 [%116, %126] : f16 into vector<2x8xf16>
      %133 = arith.mulf %arg8, %26 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
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
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg4)
      %109 = vector.insert %cst_7, %arg5 [%108, %107] : f32 into vector<2x32xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %111 = vector.insert %cst_7, %109 [%108, %110] : f32 into vector<2x32xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %113 = vector.insert %cst_7, %111 [%108, %112] : f32 into vector<2x32xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %115 = vector.insert %cst_7, %113 [%108, %114] : f32 into vector<2x32xf32>
      affine.yield %115 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
    %86 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_3) -> (vector<2x32xf32>) {
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %109 = vector.extract %85[%107, %108] : f32 from vector<2x32xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %111 = vector.extract %85[%107, %110] : f32 from vector<2x32xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %113 = vector.extract %85[%107, %112] : f32 from vector<2x32xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %115 = vector.extract %85[%107, %114] : f32 from vector<2x32xf32>
      %116 = vector.insert %109, %arg5 [%107, %108] : f32 into vector<2x32xf32>
      %117 = vector.insert %111, %116 [%107, %110] : f32 into vector<2x32xf32>
      %118 = vector.insert %113, %117 [%107, %112] : f32 into vector<2x32xf32>
      %119 = vector.insert %115, %118 [%107, %114] : f32 into vector<2x32xf32>
      affine.yield %119 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %87 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
    %88 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
    %89 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
    %90 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
    %91:6 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %13, %arg6 = %14, %arg7 = %12#3, %arg8 = %13, %arg9 = %13, %arg10 = %13) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %107 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %subview_12 = memref.subview %transpose[0, %84, 0, %107] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %108 = frisk.async_copy_gm_to_shm %subview_12 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %108 : !frisk.async_task
      %109 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_13 = memref.subview %transpose[0, %109, 32, %107] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %110 = frisk.async_copy_gm_to_shm %subview_13 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %111:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %86, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %86[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %86[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %86[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %86[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %160 = memref.load %view[%159] : memref<4096xf16, 3>
        %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %162 = memref.load %view[%161] : memref<4096xf16, 3>
        %163 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %164 = memref.load %view[%163] : memref<4096xf16, 3>
        %165 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %166 = memref.load %view[%165] : memref<4096xf16, 3>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<4x1xf16>
        %168 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %169 = memref.load %view[%168] : memref<4096xf16, 3>
        %170 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %171 = memref.load %view[%170] : memref<4096xf16, 3>
        %172 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %173 = memref.load %view[%172] : memref<4096xf16, 3>
        %174 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %175 = memref.load %view[%174] : memref<4096xf16, 3>
        %176 = vector.from_elements %169, %171, %173, %175 : vector<4x1xf16>
        %177 = vector.extract %36[%149, 0] : f16 from vector<2x8xf16>
        %178 = vector.extract %36[%149, 1] : f16 from vector<2x8xf16>
        %179 = vector.extract %36[%149, 2] : f16 from vector<2x8xf16>
        %180 = vector.extract %36[%149, 3] : f16 from vector<2x8xf16>
        %181 = vector.from_elements %177, %178, %179, %180 : vector<1x4xf16>
        %182 = frisk.warp_mma_rr %181, %167, %158 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %183 = vector.extract %36[%149, 4] : f16 from vector<2x8xf16>
        %184 = vector.extract %36[%149, 5] : f16 from vector<2x8xf16>
        %185 = vector.extract %36[%149, 6] : f16 from vector<2x8xf16>
        %186 = vector.extract %36[%149, 7] : f16 from vector<2x8xf16>
        %187 = vector.from_elements %183, %184, %185, %186 : vector<1x4xf16>
        %188 = frisk.warp_mma_rr %187, %176, %182 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %189 = vector.extract %188[0, 0] : f32 from vector<1x4xf32>
        %190 = vector.insert %189, %arg12 [%149, %150] : f32 into vector<2x32xf32>
        %191 = vector.extract %188[0, 1] : f32 from vector<1x4xf32>
        %192 = vector.insert %191, %190 [%149, %152] : f32 into vector<2x32xf32>
        %193 = vector.extract %188[0, 2] : f32 from vector<1x4xf32>
        %194 = vector.insert %193, %192 [%149, %154] : f32 into vector<2x32xf32>
        %195 = vector.extract %188[0, 3] : f32 from vector<1x4xf32>
        %196 = vector.insert %195, %194 [%149, %156] : f32 into vector<2x32xf32>
        %197 = vector.insert %189, %arg13 [%149, %150] : f32 into vector<2x32xf32>
        %198 = vector.insert %191, %197 [%149, %152] : f32 into vector<2x32xf32>
        %199 = vector.insert %193, %198 [%149, %154] : f32 into vector<2x32xf32>
        %200 = vector.insert %195, %199 [%149, %156] : f32 into vector<2x32xf32>
        affine.yield %196, %200 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %112 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %111#1[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %111#1[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %111#1[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %111#1[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.insert %151, %arg12 [%149, %150] : f32 into vector<2x32xf32>
        %159 = vector.insert %153, %158 [%149, %152] : f32 into vector<2x32xf32>
        %160 = vector.insert %155, %159 [%149, %154] : f32 into vector<2x32xf32>
        %161 = vector.insert %157, %160 [%149, %156] : f32 into vector<2x32xf32>
        affine.yield %161 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %110 : !frisk.async_task
      %113 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 2048) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_14 = memref.subview %transpose[0, %113, 64, %107] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %114 = frisk.async_copy_gm_to_shm %subview_14 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %115:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %112, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %112[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %112[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %112[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %112[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %160 = memref.load %view_8[%159] : memref<4096xf16, 3>
        %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %162 = memref.load %view_8[%161] : memref<4096xf16, 3>
        %163 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %164 = memref.load %view_8[%163] : memref<4096xf16, 3>
        %165 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %166 = memref.load %view_8[%165] : memref<4096xf16, 3>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<4x1xf16>
        %168 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %169 = memref.load %view_8[%168] : memref<4096xf16, 3>
        %170 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %171 = memref.load %view_8[%170] : memref<4096xf16, 3>
        %172 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %173 = memref.load %view_8[%172] : memref<4096xf16, 3>
        %174 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %175 = memref.load %view_8[%174] : memref<4096xf16, 3>
        %176 = vector.from_elements %169, %171, %173, %175 : vector<4x1xf16>
        %177 = vector.extract %52[%149, 0] : f16 from vector<2x8xf16>
        %178 = vector.extract %52[%149, 1] : f16 from vector<2x8xf16>
        %179 = vector.extract %52[%149, 2] : f16 from vector<2x8xf16>
        %180 = vector.extract %52[%149, 3] : f16 from vector<2x8xf16>
        %181 = vector.from_elements %177, %178, %179, %180 : vector<1x4xf16>
        %182 = frisk.warp_mma_rr %181, %167, %158 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %183 = vector.extract %52[%149, 4] : f16 from vector<2x8xf16>
        %184 = vector.extract %52[%149, 5] : f16 from vector<2x8xf16>
        %185 = vector.extract %52[%149, 6] : f16 from vector<2x8xf16>
        %186 = vector.extract %52[%149, 7] : f16 from vector<2x8xf16>
        %187 = vector.from_elements %183, %184, %185, %186 : vector<1x4xf16>
        %188 = frisk.warp_mma_rr %187, %176, %182 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %189 = vector.extract %188[0, 0] : f32 from vector<1x4xf32>
        %190 = vector.insert %189, %arg12 [%149, %150] : f32 into vector<2x32xf32>
        %191 = vector.extract %188[0, 1] : f32 from vector<1x4xf32>
        %192 = vector.insert %191, %190 [%149, %152] : f32 into vector<2x32xf32>
        %193 = vector.extract %188[0, 2] : f32 from vector<1x4xf32>
        %194 = vector.insert %193, %192 [%149, %154] : f32 into vector<2x32xf32>
        %195 = vector.extract %188[0, 3] : f32 from vector<1x4xf32>
        %196 = vector.insert %195, %194 [%149, %156] : f32 into vector<2x32xf32>
        %197 = vector.insert %189, %arg13 [%149, %150] : f32 into vector<2x32xf32>
        %198 = vector.insert %191, %197 [%149, %152] : f32 into vector<2x32xf32>
        %199 = vector.insert %193, %198 [%149, %154] : f32 into vector<2x32xf32>
        %200 = vector.insert %195, %199 [%149, %156] : f32 into vector<2x32xf32>
        affine.yield %196, %200 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %116 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %115#1[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %115#1[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %115#1[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %115#1[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.insert %151, %arg12 [%149, %150] : f32 into vector<2x32xf32>
        %159 = vector.insert %153, %158 [%149, %152] : f32 into vector<2x32xf32>
        %160 = vector.insert %155, %159 [%149, %154] : f32 into vector<2x32xf32>
        %161 = vector.insert %157, %160 [%149, %156] : f32 into vector<2x32xf32>
        affine.yield %161 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %114 : !frisk.async_task
      %117 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 3072) floordiv 4096) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_15 = memref.subview %transpose[0, %117, 96, %107] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %118 = frisk.async_copy_gm_to_shm %subview_15 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %119:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %116, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %116[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %116[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %116[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %116[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %160 = memref.load %view[%159] : memref<4096xf16, 3>
        %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %162 = memref.load %view[%161] : memref<4096xf16, 3>
        %163 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %164 = memref.load %view[%163] : memref<4096xf16, 3>
        %165 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %166 = memref.load %view[%165] : memref<4096xf16, 3>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<4x1xf16>
        %168 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %169 = memref.load %view[%168] : memref<4096xf16, 3>
        %170 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %171 = memref.load %view[%170] : memref<4096xf16, 3>
        %172 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %173 = memref.load %view[%172] : memref<4096xf16, 3>
        %174 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %175 = memref.load %view[%174] : memref<4096xf16, 3>
        %176 = vector.from_elements %169, %171, %173, %175 : vector<4x1xf16>
        %177 = vector.extract %68[%149, 0] : f16 from vector<2x8xf16>
        %178 = vector.extract %68[%149, 1] : f16 from vector<2x8xf16>
        %179 = vector.extract %68[%149, 2] : f16 from vector<2x8xf16>
        %180 = vector.extract %68[%149, 3] : f16 from vector<2x8xf16>
        %181 = vector.from_elements %177, %178, %179, %180 : vector<1x4xf16>
        %182 = frisk.warp_mma_rr %181, %167, %158 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %183 = vector.extract %68[%149, 4] : f16 from vector<2x8xf16>
        %184 = vector.extract %68[%149, 5] : f16 from vector<2x8xf16>
        %185 = vector.extract %68[%149, 6] : f16 from vector<2x8xf16>
        %186 = vector.extract %68[%149, 7] : f16 from vector<2x8xf16>
        %187 = vector.from_elements %183, %184, %185, %186 : vector<1x4xf16>
        %188 = frisk.warp_mma_rr %187, %176, %182 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %189 = vector.extract %188[0, 0] : f32 from vector<1x4xf32>
        %190 = vector.insert %189, %arg12 [%149, %150] : f32 into vector<2x32xf32>
        %191 = vector.extract %188[0, 1] : f32 from vector<1x4xf32>
        %192 = vector.insert %191, %190 [%149, %152] : f32 into vector<2x32xf32>
        %193 = vector.extract %188[0, 2] : f32 from vector<1x4xf32>
        %194 = vector.insert %193, %192 [%149, %154] : f32 into vector<2x32xf32>
        %195 = vector.extract %188[0, 3] : f32 from vector<1x4xf32>
        %196 = vector.insert %195, %194 [%149, %156] : f32 into vector<2x32xf32>
        %197 = vector.insert %189, %arg13 [%149, %150] : f32 into vector<2x32xf32>
        %198 = vector.insert %191, %197 [%149, %152] : f32 into vector<2x32xf32>
        %199 = vector.insert %193, %198 [%149, %154] : f32 into vector<2x32xf32>
        %200 = vector.insert %195, %199 [%149, %156] : f32 into vector<2x32xf32>
        affine.yield %196, %200 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %120 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %119#1[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %119#1[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %119#1[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %119#1[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.insert %151, %arg12 [%149, %150] : f32 into vector<2x32xf32>
        %159 = vector.insert %153, %158 [%149, %152] : f32 into vector<2x32xf32>
        %160 = vector.insert %155, %159 [%149, %154] : f32 into vector<2x32xf32>
        %161 = vector.insert %157, %160 [%149, %156] : f32 into vector<2x32xf32>
        affine.yield %161 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %118 : !frisk.async_task
      %subview_16 = memref.subview %arg2[0, %84, %107, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %121 = frisk.async_copy_gm_to_shm %subview_16 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %122:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %120, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %120[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %120[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %120[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %120[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg11)[%thread_id_x]
        %160 = memref.load %view_8[%159] : memref<4096xf16, 3>
        %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg11)[%thread_id_x]
        %162 = memref.load %view_8[%161] : memref<4096xf16, 3>
        %163 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg11)[%thread_id_x]
        %164 = memref.load %view_8[%163] : memref<4096xf16, 3>
        %165 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg11)[%thread_id_x]
        %166 = memref.load %view_8[%165] : memref<4096xf16, 3>
        %167 = vector.from_elements %160, %162, %164, %166 : vector<4x1xf16>
        %168 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -512 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 1024 + ((d0 mod 8) floordiv 4) * -256 + s0 * 256 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) floordiv 2) * -64 + ((d0 + (d0 floordiv 8) * -8 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 16 + ((d0 mod 8) floordiv 4) * -4 + s0 * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4) floordiv 16) mod 2) floordiv 2) * -64 + (d0 floordiv 2) * 64 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg11)[%thread_id_x]
        %169 = memref.load %view_8[%168] : memref<4096xf16, 3>
        %170 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 8 + 17) floordiv 8) * 8 + 17)>(%arg11)[%thread_id_x]
        %171 = memref.load %view_8[%170] : memref<4096xf16, 3>
        %172 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 4 + 9) floordiv 4) * 8 + 18)>(%arg11)[%thread_id_x]
        %173 = memref.load %view_8[%172] : memref<4096xf16, 3>
        %174 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) * 32 + (((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) floordiv 2) * -64 + ((((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 16) mod 2 + ((d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + (((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) floordiv 4) mod 2) floordiv 2) * -64 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 8 + 19) floordiv 8) * 8 + 19)>(%arg11)[%thread_id_x]
        %175 = memref.load %view_8[%174] : memref<4096xf16, 3>
        %176 = vector.from_elements %169, %171, %173, %175 : vector<4x1xf16>
        %177 = vector.extract %83[%149, 0] : f16 from vector<2x8xf16>
        %178 = vector.extract %83[%149, 1] : f16 from vector<2x8xf16>
        %179 = vector.extract %83[%149, 2] : f16 from vector<2x8xf16>
        %180 = vector.extract %83[%149, 3] : f16 from vector<2x8xf16>
        %181 = vector.from_elements %177, %178, %179, %180 : vector<1x4xf16>
        %182 = frisk.warp_mma_rr %181, %167, %158 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %183 = vector.extract %83[%149, 4] : f16 from vector<2x8xf16>
        %184 = vector.extract %83[%149, 5] : f16 from vector<2x8xf16>
        %185 = vector.extract %83[%149, 6] : f16 from vector<2x8xf16>
        %186 = vector.extract %83[%149, 7] : f16 from vector<2x8xf16>
        %187 = vector.from_elements %183, %184, %185, %186 : vector<1x4xf16>
        %188 = frisk.warp_mma_rr %187, %176, %182 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %189 = vector.extract %188[0, 0] : f32 from vector<1x4xf32>
        %190 = vector.insert %189, %arg12 [%149, %150] : f32 into vector<2x32xf32>
        %191 = vector.extract %188[0, 1] : f32 from vector<1x4xf32>
        %192 = vector.insert %191, %190 [%149, %152] : f32 into vector<2x32xf32>
        %193 = vector.extract %188[0, 2] : f32 from vector<1x4xf32>
        %194 = vector.insert %193, %192 [%149, %154] : f32 into vector<2x32xf32>
        %195 = vector.extract %188[0, 3] : f32 from vector<1x4xf32>
        %196 = vector.insert %195, %194 [%149, %156] : f32 into vector<2x32xf32>
        %197 = vector.insert %189, %arg13 [%149, %150] : f32 into vector<2x32xf32>
        %198 = vector.insert %191, %197 [%149, %152] : f32 into vector<2x32xf32>
        %199 = vector.insert %193, %198 [%149, %154] : f32 into vector<2x32xf32>
        %200 = vector.insert %195, %199 [%149, %156] : f32 into vector<2x32xf32>
        affine.yield %196, %200 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %123 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %122#1[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %122#1[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %122#1[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %122#1[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.insert %151, %arg12 [%149, %150] : f32 into vector<2x32xf32>
        %159 = vector.insert %153, %158 [%149, %152] : f32 into vector<2x32xf32>
        %160 = vector.insert %155, %159 [%149, %154] : f32 into vector<2x32xf32>
        %161 = vector.insert %157, %160 [%149, %156] : f32 into vector<2x32xf32>
        affine.yield %161 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %124 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %149 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_0) -> (f32) {
          %153 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %154 = vector.extract %123[%arg11, %153] : f32 from vector<2x32xf32>
          %155 = arith.maxnumf %arg14, %154 {frisk.try_multichain} : f32
          %156 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %157 = vector.extract %123[%arg11, %156] : f32 from vector<2x32xf32>
          %158 = arith.maxnumf %155, %157 {frisk.try_multichain} : f32
          %159 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %160 = vector.extract %123[%arg11, %159] : f32 from vector<2x32xf32>
          %161 = arith.maxnumf %158, %160 {frisk.try_multichain} : f32
          %162 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %163 = vector.extract %123[%arg11, %162] : f32 from vector<2x32xf32>
          %164 = arith.maxnumf %161, %163 {frisk.try_multichain} : f32
          affine.yield %164 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %149, %c16_i32, %c64_i32 : f32
        %150 = arith.maxnumf %149, %shuffleResult : f32
        %shuffleResult_20, %valid_21 = gpu.shuffle xor %150, %c32_i32, %c64_i32 : f32
        %151 = arith.maxnumf %150, %shuffleResult_20 : f32
        %152 = vector.insert %151, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %125 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %149 = vector.extract %124[%arg11, 0] : f32 from vector<2x1xf32>
        %150 = vector.broadcast %149 : f32 to vector<1x1xf32>
        %151 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %152 = vector.broadcast %151 : f32 to vector<1x1xf32>
        %153 = arith.maxnumf %150, %152 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x1xf32>
        %155 = vector.insert %154, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %155 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %126:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = vector.extract %125[%149, 0] : f32 from vector<2x1xf32>
        %151 = vector.broadcast %150 : f32 to vector<1x4xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %153 = vector.insert %150, %arg12 [%149, %152] : f32 into vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %155 = vector.insert %150, %153 [%149, %154] : f32 into vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %157 = vector.insert %150, %155 [%149, %156] : f32 into vector<2x32xf32>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %159 = vector.insert %150, %157 [%149, %158] : f32 into vector<2x32xf32>
        %160 = vector.extract %122#1[%149, %152] : f32 from vector<2x32xf32>
        %161 = vector.extract %122#1[%149, %154] : f32 from vector<2x32xf32>
        %162 = vector.extract %122#1[%149, %156] : f32 from vector<2x32xf32>
        %163 = vector.extract %122#1[%149, %158] : f32 from vector<2x32xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.subf %164, %151 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg13 [%149, %152] : f32 into vector<2x32xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%149, %154] : f32 into vector<2x32xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%149, %156] : f32 into vector<2x32xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%149, %158] : f32 into vector<2x32xf32>
        affine.yield %159, %173 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %127:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %149 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %150 = vector.broadcast %149 : f32 to vector<1x1xf32>
        %151 = vector.extract %125[%arg11, 0] : f32 from vector<2x1xf32>
        %152 = vector.broadcast %151 : f32 to vector<1x1xf32>
        %153 = arith.subf %150, %152 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x1xf32>
        %155 = vector.insert %154, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %156 = math.exp2 %153 {tiled = true} : vector<1x1xf32>
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %159 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %160 = vector.broadcast %159 : f32 to vector<1x1xf32>
        %161 = arith.mulf %160, %156 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %162 = vector.extract %161[0, 0] : f32 from vector<1x1xf32>
        %163 = vector.insert %162, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %155, %158, %163 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %128:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %126#1[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %126#1[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %126#1[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %126#1[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = math.exp2 %158 {tiled = true} : vector<1x4xf32>
        %160 = vector.extract %159[0, 0] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %arg12 [%149, %150] : f32 into vector<2x32xf32>
        %162 = vector.extract %159[0, 1] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%149, %152] : f32 into vector<2x32xf32>
        %164 = vector.extract %159[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%149, %154] : f32 into vector<2x32xf32>
        %166 = vector.extract %159[0, 3] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%149, %156] : f32 into vector<2x32xf32>
        %168 = vector.insert %160, %arg13 [%149, %150] : f32 into vector<2x32xf32>
        %169 = vector.insert %162, %168 [%149, %152] : f32 into vector<2x32xf32>
        %170 = vector.insert %164, %169 [%149, %154] : f32 into vector<2x32xf32>
        %171 = vector.insert %166, %170 [%149, %156] : f32 into vector<2x32xf32>
        affine.yield %167, %171 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %129 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %149 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_7) -> (f32) {
          %153 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %154 = vector.extract %128#1[%arg11, %153] : f32 from vector<2x32xf32>
          %155 = arith.addf %arg14, %154 {frisk.try_multichain} : f32
          %156 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %157 = vector.extract %128#1[%arg11, %156] : f32 from vector<2x32xf32>
          %158 = arith.addf %155, %157 {frisk.try_multichain} : f32
          %159 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %160 = vector.extract %128#1[%arg11, %159] : f32 from vector<2x32xf32>
          %161 = arith.addf %158, %160 {frisk.try_multichain} : f32
          %162 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %163 = vector.extract %128#1[%arg11, %162] : f32 from vector<2x32xf32>
          %164 = arith.addf %161, %163 {frisk.try_multichain} : f32
          affine.yield %164 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %149, %c16_i32, %c64_i32 : f32
        %150 = arith.addf %149, %shuffleResult : f32
        %shuffleResult_20, %valid_21 = gpu.shuffle xor %150, %c32_i32, %c64_i32 : f32
        %151 = arith.addf %150, %shuffleResult_20 : f32
        %152 = vector.insert %151, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %130 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1) -> (vector<2x1xf32>) {
        %149 = vector.extract %129[%arg11, 0] : f32 from vector<2x1xf32>
        %150 = vector.broadcast %149 : f32 to vector<1x1xf32>
        %151 = vector.extract %127#2[%arg11, 0] : f32 from vector<2x1xf32>
        %152 = vector.broadcast %151 : f32 to vector<1x1xf32>
        %153 = arith.addf %150, %152 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x1xf32>
        %155 = vector.insert %154, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %155 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %131 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf16>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %151 = vector.extract %128#0[%149, %150] : f32 from vector<2x32xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %153 = vector.extract %128#0[%149, %152] : f32 from vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %155 = vector.extract %128#0[%149, %154] : f32 from vector<2x32xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %157 = vector.extract %128#0[%149, %156] : f32 from vector<2x32xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = arith.truncf %158 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %160 = vector.extract %159[0, 0] : f16 from vector<1x4xf16>
        %161 = vector.insert %160, %arg12 [%149, %150] : f16 into vector<2x32xf16>
        %162 = vector.extract %159[0, 1] : f16 from vector<1x4xf16>
        %163 = vector.insert %162, %161 [%149, %152] : f16 into vector<2x32xf16>
        %164 = vector.extract %159[0, 2] : f16 from vector<1x4xf16>
        %165 = vector.insert %164, %163 [%149, %154] : f16 into vector<2x32xf16>
        %166 = vector.extract %159[0, 3] : f16 from vector<1x4xf16>
        %167 = vector.insert %166, %165 [%149, %156] : f16 into vector<2x32xf16>
        affine.yield %167 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %132:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %150 = vector.extract %127#1[%149, 0] : f32 from vector<2x1xf32>
        %151 = vector.broadcast %150 : f32 to vector<1x4xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %153 = vector.insert %150, %arg12 [%149, %152] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.insert %150, %153 [%149, %154] : f32 into vector<2x8xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %157 = vector.insert %150, %155 [%149, %156] : f32 into vector<2x8xf32>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %159 = vector.insert %150, %157 [%149, %158] : f32 into vector<2x8xf32>
        %160 = vector.extract %arg5[%149, %152] : f32 from vector<2x8xf32>
        %161 = vector.extract %arg5[%149, %154] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg5[%149, %156] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg5[%149, %158] : f32 from vector<2x8xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.mulf %164, %151 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg13 [%149, %152] : f32 into vector<2x8xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%149, %154] : f32 into vector<2x8xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%149, %156] : f32 into vector<2x8xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%149, %158] : f32 into vector<2x8xf32>
        affine.yield %159, %173 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %121 : !frisk.async_task
      %subview_17 = memref.subview %arg2[0, %84, %107, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %133 = frisk.async_copy_gm_to_shm %subview_17 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %134 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %149 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %150 = frisk.read_matrix_from_shm %view[%149] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %151 = vector.bitcast %150 : vector<4xi32> to vector<8xf16>
        %152 = vector.extract %151[0] : f16 from vector<8xf16>
        %153 = vector.extract %151[%87] : f16 from vector<8xf16>
        %154 = vector.extract %151[1] : f16 from vector<8xf16>
        %155 = vector.extract %151[%88] : f16 from vector<8xf16>
        %156 = vector.extract %151[2] : f16 from vector<8xf16>
        %157 = vector.extract %151[%89] : f16 from vector<8xf16>
        %158 = vector.extract %151[3] : f16 from vector<8xf16>
        %159 = vector.extract %151[%90] : f16 from vector<8xf16>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %161 = vector.insert %152, %arg12 [%160, 0] : f16 into vector<32x2xf16>
        %162 = vector.insert %153, %161 [%160, 1] : f16 into vector<32x2xf16>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %164 = vector.insert %154, %162 [%163, 0] : f16 into vector<32x2xf16>
        %165 = vector.insert %155, %164 [%163, 1] : f16 into vector<32x2xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %167 = vector.insert %156, %165 [%166, 0] : f16 into vector<32x2xf16>
        %168 = vector.insert %157, %167 [%166, 1] : f16 into vector<32x2xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %170 = vector.insert %158, %168 [%169, 0] : f16 into vector<32x2xf16>
        %171 = vector.insert %159, %170 [%169, 1] : f16 into vector<32x2xf16>
        affine.yield %171 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %135 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %132#1) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %151 = vector.extract %132#1[%149, %150] : f32 from vector<2x8xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %153 = vector.extract %132#1[%149, %152] : f32 from vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %155 = vector.extract %132#1[%149, %154] : f32 from vector<2x8xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %157 = vector.extract %132#1[%149, %156] : f32 from vector<2x8xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = vector.extract %131[%149, 0] : f16 from vector<2x32xf16>
        %160 = vector.extract %131[%149, 1] : f16 from vector<2x32xf16>
        %161 = vector.extract %131[%149, 2] : f16 from vector<2x32xf16>
        %162 = vector.extract %131[%149, 3] : f16 from vector<2x32xf16>
        %163 = vector.from_elements %159, %160, %161, %162 : vector<1x4xf16>
        %164 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %165 = vector.extract %134[0, %164] : f16 from vector<32x2xf16>
        %166 = vector.extract %134[1, %164] : f16 from vector<32x2xf16>
        %167 = vector.extract %134[2, %164] : f16 from vector<32x2xf16>
        %168 = vector.extract %134[3, %164] : f16 from vector<32x2xf16>
        %169 = vector.from_elements %165, %166, %167, %168 : vector<4x1xf16>
        %170 = frisk.warp_mma_rr %163, %169, %158 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %171 = vector.extract %131[%149, 4] : f16 from vector<2x32xf16>
        %172 = vector.extract %131[%149, 5] : f16 from vector<2x32xf16>
        %173 = vector.extract %131[%149, 6] : f16 from vector<2x32xf16>
        %174 = vector.extract %131[%149, 7] : f16 from vector<2x32xf16>
        %175 = vector.from_elements %171, %172, %173, %174 : vector<1x4xf16>
        %176 = vector.extract %134[4, %164] : f16 from vector<32x2xf16>
        %177 = vector.extract %134[5, %164] : f16 from vector<32x2xf16>
        %178 = vector.extract %134[6, %164] : f16 from vector<32x2xf16>
        %179 = vector.extract %134[7, %164] : f16 from vector<32x2xf16>
        %180 = vector.from_elements %176, %177, %178, %179 : vector<4x1xf16>
        %181 = frisk.warp_mma_rr %175, %180, %170 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %182 = vector.extract %131[%149, 8] : f16 from vector<2x32xf16>
        %183 = vector.extract %131[%149, 9] : f16 from vector<2x32xf16>
        %184 = vector.extract %131[%149, 10] : f16 from vector<2x32xf16>
        %185 = vector.extract %131[%149, 11] : f16 from vector<2x32xf16>
        %186 = vector.from_elements %182, %183, %184, %185 : vector<1x4xf16>
        %187 = vector.extract %134[8, %164] : f16 from vector<32x2xf16>
        %188 = vector.extract %134[9, %164] : f16 from vector<32x2xf16>
        %189 = vector.extract %134[10, %164] : f16 from vector<32x2xf16>
        %190 = vector.extract %134[11, %164] : f16 from vector<32x2xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<4x1xf16>
        %192 = frisk.warp_mma_rr %186, %191, %181 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %193 = vector.extract %131[%149, 12] : f16 from vector<2x32xf16>
        %194 = vector.extract %131[%149, 13] : f16 from vector<2x32xf16>
        %195 = vector.extract %131[%149, 14] : f16 from vector<2x32xf16>
        %196 = vector.extract %131[%149, 15] : f16 from vector<2x32xf16>
        %197 = vector.from_elements %193, %194, %195, %196 : vector<1x4xf16>
        %198 = vector.extract %134[12, %164] : f16 from vector<32x2xf16>
        %199 = vector.extract %134[13, %164] : f16 from vector<32x2xf16>
        %200 = vector.extract %134[14, %164] : f16 from vector<32x2xf16>
        %201 = vector.extract %134[15, %164] : f16 from vector<32x2xf16>
        %202 = vector.from_elements %198, %199, %200, %201 : vector<4x1xf16>
        %203 = frisk.warp_mma_rr %197, %202, %192 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %204 = vector.extract %131[%149, 16] : f16 from vector<2x32xf16>
        %205 = vector.extract %131[%149, 17] : f16 from vector<2x32xf16>
        %206 = vector.extract %131[%149, 18] : f16 from vector<2x32xf16>
        %207 = vector.extract %131[%149, 19] : f16 from vector<2x32xf16>
        %208 = vector.from_elements %204, %205, %206, %207 : vector<1x4xf16>
        %209 = vector.extract %134[16, %164] : f16 from vector<32x2xf16>
        %210 = vector.extract %134[17, %164] : f16 from vector<32x2xf16>
        %211 = vector.extract %134[18, %164] : f16 from vector<32x2xf16>
        %212 = vector.extract %134[19, %164] : f16 from vector<32x2xf16>
        %213 = vector.from_elements %209, %210, %211, %212 : vector<4x1xf16>
        %214 = frisk.warp_mma_rr %208, %213, %203 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %215 = vector.extract %131[%149, 20] : f16 from vector<2x32xf16>
        %216 = vector.extract %131[%149, 21] : f16 from vector<2x32xf16>
        %217 = vector.extract %131[%149, 22] : f16 from vector<2x32xf16>
        %218 = vector.extract %131[%149, 23] : f16 from vector<2x32xf16>
        %219 = vector.from_elements %215, %216, %217, %218 : vector<1x4xf16>
        %220 = vector.extract %134[20, %164] : f16 from vector<32x2xf16>
        %221 = vector.extract %134[21, %164] : f16 from vector<32x2xf16>
        %222 = vector.extract %134[22, %164] : f16 from vector<32x2xf16>
        %223 = vector.extract %134[23, %164] : f16 from vector<32x2xf16>
        %224 = vector.from_elements %220, %221, %222, %223 : vector<4x1xf16>
        %225 = frisk.warp_mma_rr %219, %224, %214 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %226 = vector.extract %131[%149, 24] : f16 from vector<2x32xf16>
        %227 = vector.extract %131[%149, 25] : f16 from vector<2x32xf16>
        %228 = vector.extract %131[%149, 26] : f16 from vector<2x32xf16>
        %229 = vector.extract %131[%149, 27] : f16 from vector<2x32xf16>
        %230 = vector.from_elements %226, %227, %228, %229 : vector<1x4xf16>
        %231 = vector.extract %134[24, %164] : f16 from vector<32x2xf16>
        %232 = vector.extract %134[25, %164] : f16 from vector<32x2xf16>
        %233 = vector.extract %134[26, %164] : f16 from vector<32x2xf16>
        %234 = vector.extract %134[27, %164] : f16 from vector<32x2xf16>
        %235 = vector.from_elements %231, %232, %233, %234 : vector<4x1xf16>
        %236 = frisk.warp_mma_rr %230, %235, %225 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %237 = vector.extract %131[%149, 28] : f16 from vector<2x32xf16>
        %238 = vector.extract %131[%149, 29] : f16 from vector<2x32xf16>
        %239 = vector.extract %131[%149, 30] : f16 from vector<2x32xf16>
        %240 = vector.extract %131[%149, 31] : f16 from vector<2x32xf16>
        %241 = vector.from_elements %237, %238, %239, %240 : vector<1x4xf16>
        %242 = vector.extract %134[28, %164] : f16 from vector<32x2xf16>
        %243 = vector.extract %134[29, %164] : f16 from vector<32x2xf16>
        %244 = vector.extract %134[30, %164] : f16 from vector<32x2xf16>
        %245 = vector.extract %134[31, %164] : f16 from vector<32x2xf16>
        %246 = vector.from_elements %242, %243, %244, %245 : vector<4x1xf16>
        %247 = frisk.warp_mma_rr %241, %246, %236 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %248 = vector.extract %247[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg12 [%149, %150] : f32 into vector<2x8xf32>
        %250 = vector.extract %247[0, 1] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %249 [%149, %152] : f32 into vector<2x8xf32>
        %252 = vector.extract %247[0, 2] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %251 [%149, %154] : f32 into vector<2x8xf32>
        %254 = vector.extract %247[0, 3] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %253 [%149, %156] : f32 into vector<2x8xf32>
        affine.yield %255 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %136:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %150 = vector.extract %127#1[%149, 0] : f32 from vector<2x1xf32>
        %151 = vector.broadcast %150 : f32 to vector<1x4xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %153 = vector.insert %150, %arg12 [%149, %152] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.insert %150, %153 [%149, %154] : f32 into vector<2x8xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %157 = vector.insert %150, %155 [%149, %156] : f32 into vector<2x8xf32>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %159 = vector.insert %150, %157 [%149, %158] : f32 into vector<2x8xf32>
        %160 = vector.extract %arg8[%149, %152] : f32 from vector<2x8xf32>
        %161 = vector.extract %arg8[%149, %154] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg8[%149, %156] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg8[%149, %158] : f32 from vector<2x8xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.mulf %164, %151 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg13 [%149, %152] : f32 into vector<2x8xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%149, %154] : f32 into vector<2x8xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%149, %156] : f32 into vector<2x8xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%149, %158] : f32 into vector<2x8xf32>
        affine.yield %159, %173 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %133 : !frisk.async_task
      %subview_18 = memref.subview %arg2[0, %84, %107, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %137 = frisk.async_copy_gm_to_shm %subview_18 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %138 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %149 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %150 = frisk.read_matrix_from_shm %view_8[%149] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %151 = vector.bitcast %150 : vector<4xi32> to vector<8xf16>
        %152 = vector.extract %151[0] : f16 from vector<8xf16>
        %153 = vector.extract %151[%87] : f16 from vector<8xf16>
        %154 = vector.extract %151[1] : f16 from vector<8xf16>
        %155 = vector.extract %151[%88] : f16 from vector<8xf16>
        %156 = vector.extract %151[2] : f16 from vector<8xf16>
        %157 = vector.extract %151[%89] : f16 from vector<8xf16>
        %158 = vector.extract %151[3] : f16 from vector<8xf16>
        %159 = vector.extract %151[%90] : f16 from vector<8xf16>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %161 = vector.insert %152, %arg12 [%160, 0] : f16 into vector<32x2xf16>
        %162 = vector.insert %153, %161 [%160, 1] : f16 into vector<32x2xf16>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %164 = vector.insert %154, %162 [%163, 0] : f16 into vector<32x2xf16>
        %165 = vector.insert %155, %164 [%163, 1] : f16 into vector<32x2xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %167 = vector.insert %156, %165 [%166, 0] : f16 into vector<32x2xf16>
        %168 = vector.insert %157, %167 [%166, 1] : f16 into vector<32x2xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %170 = vector.insert %158, %168 [%169, 0] : f16 into vector<32x2xf16>
        %171 = vector.insert %159, %170 [%169, 1] : f16 into vector<32x2xf16>
        affine.yield %171 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %139 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %136#1) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %151 = vector.extract %136#1[%149, %150] : f32 from vector<2x8xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %153 = vector.extract %136#1[%149, %152] : f32 from vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %155 = vector.extract %136#1[%149, %154] : f32 from vector<2x8xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %157 = vector.extract %136#1[%149, %156] : f32 from vector<2x8xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = vector.extract %131[%149, 0] : f16 from vector<2x32xf16>
        %160 = vector.extract %131[%149, 1] : f16 from vector<2x32xf16>
        %161 = vector.extract %131[%149, 2] : f16 from vector<2x32xf16>
        %162 = vector.extract %131[%149, 3] : f16 from vector<2x32xf16>
        %163 = vector.from_elements %159, %160, %161, %162 : vector<1x4xf16>
        %164 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %165 = vector.extract %138[0, %164] : f16 from vector<32x2xf16>
        %166 = vector.extract %138[1, %164] : f16 from vector<32x2xf16>
        %167 = vector.extract %138[2, %164] : f16 from vector<32x2xf16>
        %168 = vector.extract %138[3, %164] : f16 from vector<32x2xf16>
        %169 = vector.from_elements %165, %166, %167, %168 : vector<4x1xf16>
        %170 = frisk.warp_mma_rr %163, %169, %158 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %171 = vector.extract %131[%149, 4] : f16 from vector<2x32xf16>
        %172 = vector.extract %131[%149, 5] : f16 from vector<2x32xf16>
        %173 = vector.extract %131[%149, 6] : f16 from vector<2x32xf16>
        %174 = vector.extract %131[%149, 7] : f16 from vector<2x32xf16>
        %175 = vector.from_elements %171, %172, %173, %174 : vector<1x4xf16>
        %176 = vector.extract %138[4, %164] : f16 from vector<32x2xf16>
        %177 = vector.extract %138[5, %164] : f16 from vector<32x2xf16>
        %178 = vector.extract %138[6, %164] : f16 from vector<32x2xf16>
        %179 = vector.extract %138[7, %164] : f16 from vector<32x2xf16>
        %180 = vector.from_elements %176, %177, %178, %179 : vector<4x1xf16>
        %181 = frisk.warp_mma_rr %175, %180, %170 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %182 = vector.extract %131[%149, 8] : f16 from vector<2x32xf16>
        %183 = vector.extract %131[%149, 9] : f16 from vector<2x32xf16>
        %184 = vector.extract %131[%149, 10] : f16 from vector<2x32xf16>
        %185 = vector.extract %131[%149, 11] : f16 from vector<2x32xf16>
        %186 = vector.from_elements %182, %183, %184, %185 : vector<1x4xf16>
        %187 = vector.extract %138[8, %164] : f16 from vector<32x2xf16>
        %188 = vector.extract %138[9, %164] : f16 from vector<32x2xf16>
        %189 = vector.extract %138[10, %164] : f16 from vector<32x2xf16>
        %190 = vector.extract %138[11, %164] : f16 from vector<32x2xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<4x1xf16>
        %192 = frisk.warp_mma_rr %186, %191, %181 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %193 = vector.extract %131[%149, 12] : f16 from vector<2x32xf16>
        %194 = vector.extract %131[%149, 13] : f16 from vector<2x32xf16>
        %195 = vector.extract %131[%149, 14] : f16 from vector<2x32xf16>
        %196 = vector.extract %131[%149, 15] : f16 from vector<2x32xf16>
        %197 = vector.from_elements %193, %194, %195, %196 : vector<1x4xf16>
        %198 = vector.extract %138[12, %164] : f16 from vector<32x2xf16>
        %199 = vector.extract %138[13, %164] : f16 from vector<32x2xf16>
        %200 = vector.extract %138[14, %164] : f16 from vector<32x2xf16>
        %201 = vector.extract %138[15, %164] : f16 from vector<32x2xf16>
        %202 = vector.from_elements %198, %199, %200, %201 : vector<4x1xf16>
        %203 = frisk.warp_mma_rr %197, %202, %192 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %204 = vector.extract %131[%149, 16] : f16 from vector<2x32xf16>
        %205 = vector.extract %131[%149, 17] : f16 from vector<2x32xf16>
        %206 = vector.extract %131[%149, 18] : f16 from vector<2x32xf16>
        %207 = vector.extract %131[%149, 19] : f16 from vector<2x32xf16>
        %208 = vector.from_elements %204, %205, %206, %207 : vector<1x4xf16>
        %209 = vector.extract %138[16, %164] : f16 from vector<32x2xf16>
        %210 = vector.extract %138[17, %164] : f16 from vector<32x2xf16>
        %211 = vector.extract %138[18, %164] : f16 from vector<32x2xf16>
        %212 = vector.extract %138[19, %164] : f16 from vector<32x2xf16>
        %213 = vector.from_elements %209, %210, %211, %212 : vector<4x1xf16>
        %214 = frisk.warp_mma_rr %208, %213, %203 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %215 = vector.extract %131[%149, 20] : f16 from vector<2x32xf16>
        %216 = vector.extract %131[%149, 21] : f16 from vector<2x32xf16>
        %217 = vector.extract %131[%149, 22] : f16 from vector<2x32xf16>
        %218 = vector.extract %131[%149, 23] : f16 from vector<2x32xf16>
        %219 = vector.from_elements %215, %216, %217, %218 : vector<1x4xf16>
        %220 = vector.extract %138[20, %164] : f16 from vector<32x2xf16>
        %221 = vector.extract %138[21, %164] : f16 from vector<32x2xf16>
        %222 = vector.extract %138[22, %164] : f16 from vector<32x2xf16>
        %223 = vector.extract %138[23, %164] : f16 from vector<32x2xf16>
        %224 = vector.from_elements %220, %221, %222, %223 : vector<4x1xf16>
        %225 = frisk.warp_mma_rr %219, %224, %214 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %226 = vector.extract %131[%149, 24] : f16 from vector<2x32xf16>
        %227 = vector.extract %131[%149, 25] : f16 from vector<2x32xf16>
        %228 = vector.extract %131[%149, 26] : f16 from vector<2x32xf16>
        %229 = vector.extract %131[%149, 27] : f16 from vector<2x32xf16>
        %230 = vector.from_elements %226, %227, %228, %229 : vector<1x4xf16>
        %231 = vector.extract %138[24, %164] : f16 from vector<32x2xf16>
        %232 = vector.extract %138[25, %164] : f16 from vector<32x2xf16>
        %233 = vector.extract %138[26, %164] : f16 from vector<32x2xf16>
        %234 = vector.extract %138[27, %164] : f16 from vector<32x2xf16>
        %235 = vector.from_elements %231, %232, %233, %234 : vector<4x1xf16>
        %236 = frisk.warp_mma_rr %230, %235, %225 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %237 = vector.extract %131[%149, 28] : f16 from vector<2x32xf16>
        %238 = vector.extract %131[%149, 29] : f16 from vector<2x32xf16>
        %239 = vector.extract %131[%149, 30] : f16 from vector<2x32xf16>
        %240 = vector.extract %131[%149, 31] : f16 from vector<2x32xf16>
        %241 = vector.from_elements %237, %238, %239, %240 : vector<1x4xf16>
        %242 = vector.extract %138[28, %164] : f16 from vector<32x2xf16>
        %243 = vector.extract %138[29, %164] : f16 from vector<32x2xf16>
        %244 = vector.extract %138[30, %164] : f16 from vector<32x2xf16>
        %245 = vector.extract %138[31, %164] : f16 from vector<32x2xf16>
        %246 = vector.from_elements %242, %243, %244, %245 : vector<4x1xf16>
        %247 = frisk.warp_mma_rr %241, %246, %236 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %248 = vector.extract %247[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg12 [%149, %150] : f32 into vector<2x8xf32>
        %250 = vector.extract %247[0, 1] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %249 [%149, %152] : f32 into vector<2x8xf32>
        %252 = vector.extract %247[0, 2] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %251 [%149, %154] : f32 into vector<2x8xf32>
        %254 = vector.extract %247[0, 3] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %253 [%149, %156] : f32 into vector<2x8xf32>
        affine.yield %255 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %140:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %150 = vector.extract %127#1[%149, 0] : f32 from vector<2x1xf32>
        %151 = vector.broadcast %150 : f32 to vector<1x4xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %153 = vector.insert %150, %arg12 [%149, %152] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.insert %150, %153 [%149, %154] : f32 into vector<2x8xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %157 = vector.insert %150, %155 [%149, %156] : f32 into vector<2x8xf32>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %159 = vector.insert %150, %157 [%149, %158] : f32 into vector<2x8xf32>
        %160 = vector.extract %arg9[%149, %152] : f32 from vector<2x8xf32>
        %161 = vector.extract %arg9[%149, %154] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg9[%149, %156] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg9[%149, %158] : f32 from vector<2x8xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.mulf %164, %151 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg13 [%149, %152] : f32 into vector<2x8xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%149, %154] : f32 into vector<2x8xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%149, %156] : f32 into vector<2x8xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%149, %158] : f32 into vector<2x8xf32>
        affine.yield %159, %173 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %137 : !frisk.async_task
      %subview_19 = memref.subview %arg2[0, %84, %107, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %141 = frisk.async_copy_gm_to_shm %subview_19 to %view_8 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %142 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %149 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %150 = frisk.read_matrix_from_shm %view[%149] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %151 = vector.bitcast %150 : vector<4xi32> to vector<8xf16>
        %152 = vector.extract %151[0] : f16 from vector<8xf16>
        %153 = vector.extract %151[%87] : f16 from vector<8xf16>
        %154 = vector.extract %151[1] : f16 from vector<8xf16>
        %155 = vector.extract %151[%88] : f16 from vector<8xf16>
        %156 = vector.extract %151[2] : f16 from vector<8xf16>
        %157 = vector.extract %151[%89] : f16 from vector<8xf16>
        %158 = vector.extract %151[3] : f16 from vector<8xf16>
        %159 = vector.extract %151[%90] : f16 from vector<8xf16>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %161 = vector.insert %152, %arg12 [%160, 0] : f16 into vector<32x2xf16>
        %162 = vector.insert %153, %161 [%160, 1] : f16 into vector<32x2xf16>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %164 = vector.insert %154, %162 [%163, 0] : f16 into vector<32x2xf16>
        %165 = vector.insert %155, %164 [%163, 1] : f16 into vector<32x2xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %167 = vector.insert %156, %165 [%166, 0] : f16 into vector<32x2xf16>
        %168 = vector.insert %157, %167 [%166, 1] : f16 into vector<32x2xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %170 = vector.insert %158, %168 [%169, 0] : f16 into vector<32x2xf16>
        %171 = vector.insert %159, %170 [%169, 1] : f16 into vector<32x2xf16>
        affine.yield %171 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %143 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %140#1) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %151 = vector.extract %140#1[%149, %150] : f32 from vector<2x8xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %153 = vector.extract %140#1[%149, %152] : f32 from vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %155 = vector.extract %140#1[%149, %154] : f32 from vector<2x8xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %157 = vector.extract %140#1[%149, %156] : f32 from vector<2x8xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = vector.extract %131[%149, 0] : f16 from vector<2x32xf16>
        %160 = vector.extract %131[%149, 1] : f16 from vector<2x32xf16>
        %161 = vector.extract %131[%149, 2] : f16 from vector<2x32xf16>
        %162 = vector.extract %131[%149, 3] : f16 from vector<2x32xf16>
        %163 = vector.from_elements %159, %160, %161, %162 : vector<1x4xf16>
        %164 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %165 = vector.extract %142[0, %164] : f16 from vector<32x2xf16>
        %166 = vector.extract %142[1, %164] : f16 from vector<32x2xf16>
        %167 = vector.extract %142[2, %164] : f16 from vector<32x2xf16>
        %168 = vector.extract %142[3, %164] : f16 from vector<32x2xf16>
        %169 = vector.from_elements %165, %166, %167, %168 : vector<4x1xf16>
        %170 = frisk.warp_mma_rr %163, %169, %158 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %171 = vector.extract %131[%149, 4] : f16 from vector<2x32xf16>
        %172 = vector.extract %131[%149, 5] : f16 from vector<2x32xf16>
        %173 = vector.extract %131[%149, 6] : f16 from vector<2x32xf16>
        %174 = vector.extract %131[%149, 7] : f16 from vector<2x32xf16>
        %175 = vector.from_elements %171, %172, %173, %174 : vector<1x4xf16>
        %176 = vector.extract %142[4, %164] : f16 from vector<32x2xf16>
        %177 = vector.extract %142[5, %164] : f16 from vector<32x2xf16>
        %178 = vector.extract %142[6, %164] : f16 from vector<32x2xf16>
        %179 = vector.extract %142[7, %164] : f16 from vector<32x2xf16>
        %180 = vector.from_elements %176, %177, %178, %179 : vector<4x1xf16>
        %181 = frisk.warp_mma_rr %175, %180, %170 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %182 = vector.extract %131[%149, 8] : f16 from vector<2x32xf16>
        %183 = vector.extract %131[%149, 9] : f16 from vector<2x32xf16>
        %184 = vector.extract %131[%149, 10] : f16 from vector<2x32xf16>
        %185 = vector.extract %131[%149, 11] : f16 from vector<2x32xf16>
        %186 = vector.from_elements %182, %183, %184, %185 : vector<1x4xf16>
        %187 = vector.extract %142[8, %164] : f16 from vector<32x2xf16>
        %188 = vector.extract %142[9, %164] : f16 from vector<32x2xf16>
        %189 = vector.extract %142[10, %164] : f16 from vector<32x2xf16>
        %190 = vector.extract %142[11, %164] : f16 from vector<32x2xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<4x1xf16>
        %192 = frisk.warp_mma_rr %186, %191, %181 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %193 = vector.extract %131[%149, 12] : f16 from vector<2x32xf16>
        %194 = vector.extract %131[%149, 13] : f16 from vector<2x32xf16>
        %195 = vector.extract %131[%149, 14] : f16 from vector<2x32xf16>
        %196 = vector.extract %131[%149, 15] : f16 from vector<2x32xf16>
        %197 = vector.from_elements %193, %194, %195, %196 : vector<1x4xf16>
        %198 = vector.extract %142[12, %164] : f16 from vector<32x2xf16>
        %199 = vector.extract %142[13, %164] : f16 from vector<32x2xf16>
        %200 = vector.extract %142[14, %164] : f16 from vector<32x2xf16>
        %201 = vector.extract %142[15, %164] : f16 from vector<32x2xf16>
        %202 = vector.from_elements %198, %199, %200, %201 : vector<4x1xf16>
        %203 = frisk.warp_mma_rr %197, %202, %192 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %204 = vector.extract %131[%149, 16] : f16 from vector<2x32xf16>
        %205 = vector.extract %131[%149, 17] : f16 from vector<2x32xf16>
        %206 = vector.extract %131[%149, 18] : f16 from vector<2x32xf16>
        %207 = vector.extract %131[%149, 19] : f16 from vector<2x32xf16>
        %208 = vector.from_elements %204, %205, %206, %207 : vector<1x4xf16>
        %209 = vector.extract %142[16, %164] : f16 from vector<32x2xf16>
        %210 = vector.extract %142[17, %164] : f16 from vector<32x2xf16>
        %211 = vector.extract %142[18, %164] : f16 from vector<32x2xf16>
        %212 = vector.extract %142[19, %164] : f16 from vector<32x2xf16>
        %213 = vector.from_elements %209, %210, %211, %212 : vector<4x1xf16>
        %214 = frisk.warp_mma_rr %208, %213, %203 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %215 = vector.extract %131[%149, 20] : f16 from vector<2x32xf16>
        %216 = vector.extract %131[%149, 21] : f16 from vector<2x32xf16>
        %217 = vector.extract %131[%149, 22] : f16 from vector<2x32xf16>
        %218 = vector.extract %131[%149, 23] : f16 from vector<2x32xf16>
        %219 = vector.from_elements %215, %216, %217, %218 : vector<1x4xf16>
        %220 = vector.extract %142[20, %164] : f16 from vector<32x2xf16>
        %221 = vector.extract %142[21, %164] : f16 from vector<32x2xf16>
        %222 = vector.extract %142[22, %164] : f16 from vector<32x2xf16>
        %223 = vector.extract %142[23, %164] : f16 from vector<32x2xf16>
        %224 = vector.from_elements %220, %221, %222, %223 : vector<4x1xf16>
        %225 = frisk.warp_mma_rr %219, %224, %214 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %226 = vector.extract %131[%149, 24] : f16 from vector<2x32xf16>
        %227 = vector.extract %131[%149, 25] : f16 from vector<2x32xf16>
        %228 = vector.extract %131[%149, 26] : f16 from vector<2x32xf16>
        %229 = vector.extract %131[%149, 27] : f16 from vector<2x32xf16>
        %230 = vector.from_elements %226, %227, %228, %229 : vector<1x4xf16>
        %231 = vector.extract %142[24, %164] : f16 from vector<32x2xf16>
        %232 = vector.extract %142[25, %164] : f16 from vector<32x2xf16>
        %233 = vector.extract %142[26, %164] : f16 from vector<32x2xf16>
        %234 = vector.extract %142[27, %164] : f16 from vector<32x2xf16>
        %235 = vector.from_elements %231, %232, %233, %234 : vector<4x1xf16>
        %236 = frisk.warp_mma_rr %230, %235, %225 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %237 = vector.extract %131[%149, 28] : f16 from vector<2x32xf16>
        %238 = vector.extract %131[%149, 29] : f16 from vector<2x32xf16>
        %239 = vector.extract %131[%149, 30] : f16 from vector<2x32xf16>
        %240 = vector.extract %131[%149, 31] : f16 from vector<2x32xf16>
        %241 = vector.from_elements %237, %238, %239, %240 : vector<1x4xf16>
        %242 = vector.extract %142[28, %164] : f16 from vector<32x2xf16>
        %243 = vector.extract %142[29, %164] : f16 from vector<32x2xf16>
        %244 = vector.extract %142[30, %164] : f16 from vector<32x2xf16>
        %245 = vector.extract %142[31, %164] : f16 from vector<32x2xf16>
        %246 = vector.from_elements %242, %243, %244, %245 : vector<4x1xf16>
        %247 = frisk.warp_mma_rr %241, %246, %236 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %248 = vector.extract %247[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg12 [%149, %150] : f32 into vector<2x8xf32>
        %250 = vector.extract %247[0, 1] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %249 [%149, %152] : f32 into vector<2x8xf32>
        %252 = vector.extract %247[0, 2] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %251 [%149, %154] : f32 into vector<2x8xf32>
        %254 = vector.extract %247[0, 3] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %253 [%149, %156] : f32 into vector<2x8xf32>
        affine.yield %255 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %144:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %150 = vector.extract %127#1[%149, 0] : f32 from vector<2x1xf32>
        %151 = vector.broadcast %150 : f32 to vector<1x4xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %153 = vector.insert %150, %arg12 [%149, %152] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.insert %150, %153 [%149, %154] : f32 into vector<2x8xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %157 = vector.insert %150, %155 [%149, %156] : f32 into vector<2x8xf32>
        %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %159 = vector.insert %150, %157 [%149, %158] : f32 into vector<2x8xf32>
        %160 = vector.extract %arg10[%149, %152] : f32 from vector<2x8xf32>
        %161 = vector.extract %arg10[%149, %154] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg10[%149, %156] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg10[%149, %158] : f32 from vector<2x8xf32>
        %164 = vector.from_elements %160, %161, %162, %163 : vector<1x4xf32>
        %165 = arith.mulf %164, %151 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %166 = vector.extract %165[0, 0] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %arg13 [%149, %152] : f32 into vector<2x8xf32>
        %168 = vector.extract %165[0, 1] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%149, %154] : f32 into vector<2x8xf32>
        %170 = vector.extract %165[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%149, %156] : f32 into vector<2x8xf32>
        %172 = vector.extract %165[0, 3] : f32 from vector<1x4xf32>
        %173 = vector.insert %172, %171 [%149, %158] : f32 into vector<2x8xf32>
        affine.yield %159, %173 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %141 : !frisk.async_task
      %145 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_6) -> (vector<32x2xf16>) {
        %149 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x]
        %150 = frisk.read_matrix_from_shm %view_8[%149] {element_bit_width = 16 : i64, frisk.v_double_buffer = 8192 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %151 = vector.bitcast %150 : vector<4xi32> to vector<8xf16>
        %152 = vector.extract %151[0] : f16 from vector<8xf16>
        %153 = vector.extract %151[%87] : f16 from vector<8xf16>
        %154 = vector.extract %151[1] : f16 from vector<8xf16>
        %155 = vector.extract %151[%88] : f16 from vector<8xf16>
        %156 = vector.extract %151[2] : f16 from vector<8xf16>
        %157 = vector.extract %151[%89] : f16 from vector<8xf16>
        %158 = vector.extract %151[3] : f16 from vector<8xf16>
        %159 = vector.extract %151[%90] : f16 from vector<8xf16>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %161 = vector.insert %152, %arg12 [%160, 0] : f16 into vector<32x2xf16>
        %162 = vector.insert %153, %161 [%160, 1] : f16 into vector<32x2xf16>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %164 = vector.insert %154, %162 [%163, 0] : f16 into vector<32x2xf16>
        %165 = vector.insert %155, %164 [%163, 1] : f16 into vector<32x2xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %167 = vector.insert %156, %165 [%166, 0] : f16 into vector<32x2xf16>
        %168 = vector.insert %157, %167 [%166, 1] : f16 into vector<32x2xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %170 = vector.insert %158, %168 [%169, 0] : f16 into vector<32x2xf16>
        %171 = vector.insert %159, %170 [%169, 1] : f16 into vector<32x2xf16>
        affine.yield %171 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %146:2 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %144#1, %arg13 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %151 = vector.extract %144#1[%149, %150] : f32 from vector<2x8xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %153 = vector.extract %144#1[%149, %152] : f32 from vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %155 = vector.extract %144#1[%149, %154] : f32 from vector<2x8xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %157 = vector.extract %144#1[%149, %156] : f32 from vector<2x8xf32>
        %158 = vector.from_elements %151, %153, %155, %157 : vector<1x4xf32>
        %159 = vector.extract %131[%149, 0] : f16 from vector<2x32xf16>
        %160 = vector.extract %131[%149, 1] : f16 from vector<2x32xf16>
        %161 = vector.extract %131[%149, 2] : f16 from vector<2x32xf16>
        %162 = vector.extract %131[%149, 3] : f16 from vector<2x32xf16>
        %163 = vector.from_elements %159, %160, %161, %162 : vector<1x4xf16>
        %164 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %165 = vector.extract %145[0, %164] : f16 from vector<32x2xf16>
        %166 = vector.extract %145[1, %164] : f16 from vector<32x2xf16>
        %167 = vector.extract %145[2, %164] : f16 from vector<32x2xf16>
        %168 = vector.extract %145[3, %164] : f16 from vector<32x2xf16>
        %169 = vector.from_elements %165, %166, %167, %168 : vector<4x1xf16>
        %170 = frisk.warp_mma_rr %163, %169, %158 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %171 = vector.extract %131[%149, 4] : f16 from vector<2x32xf16>
        %172 = vector.extract %131[%149, 5] : f16 from vector<2x32xf16>
        %173 = vector.extract %131[%149, 6] : f16 from vector<2x32xf16>
        %174 = vector.extract %131[%149, 7] : f16 from vector<2x32xf16>
        %175 = vector.from_elements %171, %172, %173, %174 : vector<1x4xf16>
        %176 = vector.extract %145[4, %164] : f16 from vector<32x2xf16>
        %177 = vector.extract %145[5, %164] : f16 from vector<32x2xf16>
        %178 = vector.extract %145[6, %164] : f16 from vector<32x2xf16>
        %179 = vector.extract %145[7, %164] : f16 from vector<32x2xf16>
        %180 = vector.from_elements %176, %177, %178, %179 : vector<4x1xf16>
        %181 = frisk.warp_mma_rr %175, %180, %170 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %182 = vector.extract %131[%149, 8] : f16 from vector<2x32xf16>
        %183 = vector.extract %131[%149, 9] : f16 from vector<2x32xf16>
        %184 = vector.extract %131[%149, 10] : f16 from vector<2x32xf16>
        %185 = vector.extract %131[%149, 11] : f16 from vector<2x32xf16>
        %186 = vector.from_elements %182, %183, %184, %185 : vector<1x4xf16>
        %187 = vector.extract %145[8, %164] : f16 from vector<32x2xf16>
        %188 = vector.extract %145[9, %164] : f16 from vector<32x2xf16>
        %189 = vector.extract %145[10, %164] : f16 from vector<32x2xf16>
        %190 = vector.extract %145[11, %164] : f16 from vector<32x2xf16>
        %191 = vector.from_elements %187, %188, %189, %190 : vector<4x1xf16>
        %192 = frisk.warp_mma_rr %186, %191, %181 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %193 = vector.extract %131[%149, 12] : f16 from vector<2x32xf16>
        %194 = vector.extract %131[%149, 13] : f16 from vector<2x32xf16>
        %195 = vector.extract %131[%149, 14] : f16 from vector<2x32xf16>
        %196 = vector.extract %131[%149, 15] : f16 from vector<2x32xf16>
        %197 = vector.from_elements %193, %194, %195, %196 : vector<1x4xf16>
        %198 = vector.extract %145[12, %164] : f16 from vector<32x2xf16>
        %199 = vector.extract %145[13, %164] : f16 from vector<32x2xf16>
        %200 = vector.extract %145[14, %164] : f16 from vector<32x2xf16>
        %201 = vector.extract %145[15, %164] : f16 from vector<32x2xf16>
        %202 = vector.from_elements %198, %199, %200, %201 : vector<4x1xf16>
        %203 = frisk.warp_mma_rr %197, %202, %192 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %204 = vector.extract %131[%149, 16] : f16 from vector<2x32xf16>
        %205 = vector.extract %131[%149, 17] : f16 from vector<2x32xf16>
        %206 = vector.extract %131[%149, 18] : f16 from vector<2x32xf16>
        %207 = vector.extract %131[%149, 19] : f16 from vector<2x32xf16>
        %208 = vector.from_elements %204, %205, %206, %207 : vector<1x4xf16>
        %209 = vector.extract %145[16, %164] : f16 from vector<32x2xf16>
        %210 = vector.extract %145[17, %164] : f16 from vector<32x2xf16>
        %211 = vector.extract %145[18, %164] : f16 from vector<32x2xf16>
        %212 = vector.extract %145[19, %164] : f16 from vector<32x2xf16>
        %213 = vector.from_elements %209, %210, %211, %212 : vector<4x1xf16>
        %214 = frisk.warp_mma_rr %208, %213, %203 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %215 = vector.extract %131[%149, 20] : f16 from vector<2x32xf16>
        %216 = vector.extract %131[%149, 21] : f16 from vector<2x32xf16>
        %217 = vector.extract %131[%149, 22] : f16 from vector<2x32xf16>
        %218 = vector.extract %131[%149, 23] : f16 from vector<2x32xf16>
        %219 = vector.from_elements %215, %216, %217, %218 : vector<1x4xf16>
        %220 = vector.extract %145[20, %164] : f16 from vector<32x2xf16>
        %221 = vector.extract %145[21, %164] : f16 from vector<32x2xf16>
        %222 = vector.extract %145[22, %164] : f16 from vector<32x2xf16>
        %223 = vector.extract %145[23, %164] : f16 from vector<32x2xf16>
        %224 = vector.from_elements %220, %221, %222, %223 : vector<4x1xf16>
        %225 = frisk.warp_mma_rr %219, %224, %214 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %226 = vector.extract %131[%149, 24] : f16 from vector<2x32xf16>
        %227 = vector.extract %131[%149, 25] : f16 from vector<2x32xf16>
        %228 = vector.extract %131[%149, 26] : f16 from vector<2x32xf16>
        %229 = vector.extract %131[%149, 27] : f16 from vector<2x32xf16>
        %230 = vector.from_elements %226, %227, %228, %229 : vector<1x4xf16>
        %231 = vector.extract %145[24, %164] : f16 from vector<32x2xf16>
        %232 = vector.extract %145[25, %164] : f16 from vector<32x2xf16>
        %233 = vector.extract %145[26, %164] : f16 from vector<32x2xf16>
        %234 = vector.extract %145[27, %164] : f16 from vector<32x2xf16>
        %235 = vector.from_elements %231, %232, %233, %234 : vector<4x1xf16>
        %236 = frisk.warp_mma_rr %230, %235, %225 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %237 = vector.extract %131[%149, 28] : f16 from vector<2x32xf16>
        %238 = vector.extract %131[%149, 29] : f16 from vector<2x32xf16>
        %239 = vector.extract %131[%149, 30] : f16 from vector<2x32xf16>
        %240 = vector.extract %131[%149, 31] : f16 from vector<2x32xf16>
        %241 = vector.from_elements %237, %238, %239, %240 : vector<1x4xf16>
        %242 = vector.extract %145[28, %164] : f16 from vector<32x2xf16>
        %243 = vector.extract %145[29, %164] : f16 from vector<32x2xf16>
        %244 = vector.extract %145[30, %164] : f16 from vector<32x2xf16>
        %245 = vector.extract %145[31, %164] : f16 from vector<32x2xf16>
        %246 = vector.from_elements %242, %243, %244, %245 : vector<4x1xf16>
        %247 = frisk.warp_mma_rr %241, %246, %236 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %248 = vector.extract %247[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg12 [%149, %150] : f32 into vector<2x8xf32>
        %250 = vector.extract %247[0, 1] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %249 [%149, %152] : f32 into vector<2x8xf32>
        %252 = vector.extract %247[0, 2] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %251 [%149, %154] : f32 into vector<2x8xf32>
        %254 = vector.extract %247[0, 3] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %253 [%149, %156] : f32 into vector<2x8xf32>
        %256 = vector.extract %135[%149, %150] : f32 from vector<2x8xf32>
        %257 = vector.extract %135[%149, %152] : f32 from vector<2x8xf32>
        %258 = vector.extract %135[%149, %154] : f32 from vector<2x8xf32>
        %259 = vector.extract %135[%149, %156] : f32 from vector<2x8xf32>
        %260 = vector.insert %256, %arg13 [%149, %150] : f32 into vector<2x8xf32>
        %261 = vector.insert %257, %260 [%149, %152] : f32 into vector<2x8xf32>
        %262 = vector.insert %258, %261 [%149, %154] : f32 into vector<2x8xf32>
        %263 = vector.insert %259, %262 [%149, %156] : f32 into vector<2x8xf32>
        affine.yield %255, %263 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %147:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_1, %arg13 = %cst_1) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %149 = vector.extract %130[%arg11, 0] : f32 from vector<2x1xf32>
        %150 = vector.insert %149, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %151 = vector.extract %125[%arg11, 0] : f32 from vector<2x1xf32>
        %152 = vector.insert %151, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %150, %152 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %148:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_5, %arg13 = %cst_5, %arg14 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %151 = vector.extract %139[%149, %150] : f32 from vector<2x8xf32>
        %152 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %153 = vector.extract %139[%149, %152] : f32 from vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %155 = vector.extract %139[%149, %154] : f32 from vector<2x8xf32>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %157 = vector.extract %139[%149, %156] : f32 from vector<2x8xf32>
        %158 = vector.insert %151, %arg12 [%149, %150] : f32 into vector<2x8xf32>
        %159 = vector.insert %153, %158 [%149, %152] : f32 into vector<2x8xf32>
        %160 = vector.insert %155, %159 [%149, %154] : f32 into vector<2x8xf32>
        %161 = vector.insert %157, %160 [%149, %156] : f32 into vector<2x8xf32>
        %162 = vector.extract %143[%149, %150] : f32 from vector<2x8xf32>
        %163 = vector.extract %143[%149, %152] : f32 from vector<2x8xf32>
        %164 = vector.extract %143[%149, %154] : f32 from vector<2x8xf32>
        %165 = vector.extract %143[%149, %156] : f32 from vector<2x8xf32>
        %166 = vector.insert %162, %arg13 [%149, %150] : f32 into vector<2x8xf32>
        %167 = vector.insert %163, %166 [%149, %152] : f32 into vector<2x8xf32>
        %168 = vector.insert %164, %167 [%149, %154] : f32 into vector<2x8xf32>
        %169 = vector.insert %165, %168 [%149, %156] : f32 into vector<2x8xf32>
        %170 = vector.extract %146#0[%149, %150] : f32 from vector<2x8xf32>
        %171 = vector.extract %146#0[%149, %152] : f32 from vector<2x8xf32>
        %172 = vector.extract %146#0[%149, %154] : f32 from vector<2x8xf32>
        %173 = vector.extract %146#0[%149, %156] : f32 from vector<2x8xf32>
        %174 = vector.insert %170, %arg14 [%149, %150] : f32 into vector<2x8xf32>
        %175 = vector.insert %171, %174 [%149, %152] : f32 into vector<2x8xf32>
        %176 = vector.insert %172, %175 [%149, %154] : f32 into vector<2x8xf32>
        %177 = vector.insert %173, %176 [%149, %156] : f32 into vector<2x8xf32>
        affine.yield %161, %169, %177 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      gpu.barrier
      affine.yield %146#1, %147#0, %147#1, %148#0, %148#1, %148#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %92 = arith.index_cast %thread_id_x : index to i32
    %93 = arith.andi %92, %c16_i32 : i32
    %94 = arith.cmpi ne, %93, %c0_i32 : i32
    %95 = arith.andi %92, %c32_i32 : i32
    %96 = arith.cmpi ne, %95, %c0_i32 : i32
    %97 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %109 = vector.extract %91#5[%108, %107] : f32 from vector<2x8xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %111 = vector.extract %91#5[%108, %110] : f32 from vector<2x8xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %113 = vector.extract %91#5[%108, %112] : f32 from vector<2x8xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %115 = vector.extract %91#5[%108, %114] : f32 from vector<2x8xf32>
      %116 = arith.select %94, %109, %111 : f32
      %shuffleResult, %valid = gpu.shuffle xor %116, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %94, %shuffleResult, %109 : f32
      %118 = arith.select %94, %111, %shuffleResult : f32
      %119 = arith.select %94, %113, %115 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %119, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %94, %shuffleResult_12, %113 : f32
      %121 = arith.select %94, %115, %shuffleResult_12 : f32
      %122 = arith.select %96, %117, %120 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %122, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %123 = arith.select %96, %shuffleResult_14, %117 : f32
      %124 = arith.select %96, %120, %shuffleResult_14 : f32
      %125 = arith.select %96, %118, %121 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %125, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %126 = arith.select %96, %shuffleResult_16, %118 : f32
      %127 = arith.select %96, %121, %shuffleResult_16 : f32
      %128 = vector.insert %123, %arg5 [%108, %107] : f32 into vector<2x8xf32>
      %129 = vector.insert %126, %128 [%108, %110] : f32 into vector<2x8xf32>
      %130 = vector.insert %124, %129 [%108, %112] : f32 into vector<2x8xf32>
      %131 = vector.insert %127, %130 [%108, %114] : f32 into vector<2x8xf32>
      affine.yield %131 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %98 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %109 = vector.extract %91#4[%108, %107] : f32 from vector<2x8xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %111 = vector.extract %91#4[%108, %110] : f32 from vector<2x8xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %113 = vector.extract %91#4[%108, %112] : f32 from vector<2x8xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %115 = vector.extract %91#4[%108, %114] : f32 from vector<2x8xf32>
      %116 = arith.select %94, %109, %111 : f32
      %shuffleResult, %valid = gpu.shuffle xor %116, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %94, %shuffleResult, %109 : f32
      %118 = arith.select %94, %111, %shuffleResult : f32
      %119 = arith.select %94, %113, %115 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %119, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %94, %shuffleResult_12, %113 : f32
      %121 = arith.select %94, %115, %shuffleResult_12 : f32
      %122 = arith.select %96, %117, %120 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %122, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %123 = arith.select %96, %shuffleResult_14, %117 : f32
      %124 = arith.select %96, %120, %shuffleResult_14 : f32
      %125 = arith.select %96, %118, %121 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %125, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %126 = arith.select %96, %shuffleResult_16, %118 : f32
      %127 = arith.select %96, %121, %shuffleResult_16 : f32
      %128 = vector.insert %123, %arg5 [%108, %107] : f32 into vector<2x8xf32>
      %129 = vector.insert %126, %128 [%108, %110] : f32 into vector<2x8xf32>
      %130 = vector.insert %124, %129 [%108, %112] : f32 into vector<2x8xf32>
      %131 = vector.insert %127, %130 [%108, %114] : f32 into vector<2x8xf32>
      affine.yield %131 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %99 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %109 = vector.extract %91#3[%108, %107] : f32 from vector<2x8xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %111 = vector.extract %91#3[%108, %110] : f32 from vector<2x8xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %113 = vector.extract %91#3[%108, %112] : f32 from vector<2x8xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %115 = vector.extract %91#3[%108, %114] : f32 from vector<2x8xf32>
      %116 = arith.select %94, %109, %111 : f32
      %shuffleResult, %valid = gpu.shuffle xor %116, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %94, %shuffleResult, %109 : f32
      %118 = arith.select %94, %111, %shuffleResult : f32
      %119 = arith.select %94, %113, %115 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %119, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %94, %shuffleResult_12, %113 : f32
      %121 = arith.select %94, %115, %shuffleResult_12 : f32
      %122 = arith.select %96, %117, %120 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %122, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %123 = arith.select %96, %shuffleResult_14, %117 : f32
      %124 = arith.select %96, %120, %shuffleResult_14 : f32
      %125 = arith.select %96, %118, %121 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %125, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %126 = arith.select %96, %shuffleResult_16, %118 : f32
      %127 = arith.select %96, %121, %shuffleResult_16 : f32
      %128 = vector.insert %123, %arg5 [%108, %107] : f32 into vector<2x8xf32>
      %129 = vector.insert %126, %128 [%108, %110] : f32 into vector<2x8xf32>
      %130 = vector.insert %124, %129 [%108, %112] : f32 into vector<2x8xf32>
      %131 = vector.insert %127, %130 [%108, %114] : f32 into vector<2x8xf32>
      affine.yield %131 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %100 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5) -> (vector<2x8xf32>) {
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %109 = vector.extract %91#0[%108, %107] : f32 from vector<2x8xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %111 = vector.extract %91#0[%108, %110] : f32 from vector<2x8xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %113 = vector.extract %91#0[%108, %112] : f32 from vector<2x8xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %115 = vector.extract %91#0[%108, %114] : f32 from vector<2x8xf32>
      %116 = arith.select %94, %109, %111 : f32
      %shuffleResult, %valid = gpu.shuffle xor %116, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %94, %shuffleResult, %109 : f32
      %118 = arith.select %94, %111, %shuffleResult : f32
      %119 = arith.select %94, %113, %115 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %119, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %94, %shuffleResult_12, %113 : f32
      %121 = arith.select %94, %115, %shuffleResult_12 : f32
      %122 = arith.select %96, %117, %120 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %122, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %123 = arith.select %96, %shuffleResult_14, %117 : f32
      %124 = arith.select %96, %120, %shuffleResult_14 : f32
      %125 = arith.select %96, %118, %121 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %125, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %126 = arith.select %96, %shuffleResult_16, %118 : f32
      %127 = arith.select %96, %121, %shuffleResult_16 : f32
      %128 = vector.insert %123, %arg5 [%108, %107] : f32 into vector<2x8xf32>
      %129 = vector.insert %126, %128 [%108, %110] : f32 into vector<2x8xf32>
      %130 = vector.insert %124, %129 [%108, %112] : f32 into vector<2x8xf32>
      %131 = vector.insert %127, %130 [%108, %114] : f32 into vector<2x8xf32>
      affine.yield %131 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %101:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = vector.extract %91#1[%107, 0] : f32 from vector<2x1xf32>
      %109 = vector.broadcast %108 : f32 to vector<1x4xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %111 = vector.insert %108, %arg5 [%107, %110] : f32 into vector<2x8xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %113 = vector.insert %108, %111 [%107, %112] : f32 into vector<2x8xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.insert %108, %113 [%107, %114] : f32 into vector<2x8xf32>
      %116 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %117 = vector.insert %108, %115 [%107, %116] : f32 into vector<2x8xf32>
      %118 = vector.extract %100[%107, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %100[%107, %112] : f32 from vector<2x8xf32>
      %120 = vector.extract %100[%107, %114] : f32 from vector<2x8xf32>
      %121 = vector.extract %100[%107, %116] : f32 from vector<2x8xf32>
      %122 = vector.from_elements %118, %119, %120, %121 : vector<1x4xf32>
      %123 = arith.divf %122, %109 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %124 = vector.extract %123[0, 0] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %arg6 [%107, %110] : f32 into vector<2x8xf32>
      %126 = vector.extract %123[0, 1] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %125 [%107, %112] : f32 into vector<2x8xf32>
      %128 = vector.extract %123[0, 2] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%107, %114] : f32 into vector<2x8xf32>
      %130 = vector.extract %123[0, 3] : f32 from vector<1x4xf32>
      %131 = vector.insert %130, %129 [%107, %116] : f32 into vector<2x8xf32>
      %132 = arith.truncf %123 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %133 = vector.extract %132[0, 0] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %arg7 [%107, %110] : f16 into vector<2x8xf16>
      %135 = vector.extract %132[0, 1] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%107, %112] : f16 into vector<2x8xf16>
      %137 = vector.extract %132[0, 2] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%107, %114] : f16 into vector<2x8xf16>
      %139 = vector.extract %132[0, 3] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%107, %116] : f16 into vector<2x8xf16>
      %141 = vector.insert %133, %arg8 [%107, %110] : f16 into vector<2x8xf16>
      %142 = vector.insert %135, %141 [%107, %112] : f16 into vector<2x8xf16>
      %143 = vector.insert %137, %142 [%107, %114] : f16 into vector<2x8xf16>
      %144 = vector.insert %139, %143 [%107, %116] : f16 into vector<2x8xf16>
      affine.yield %117, %131, %140, %144 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %102 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %103 = arith.cmpi ult, %102, %c4 : index
    scf.if %103 {
      affine.for %arg4 = 0 to 4 {
        %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %109 = vector.extract %101#3[%107, %108] : f16 from vector<2x8xf16>
        %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %111 = vector.extract %101#3[%107, %110] : f16 from vector<2x8xf16>
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %113 = vector.extract %101#3[%107, %112] : f16 from vector<2x8xf16>
        %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %115 = vector.extract %101#3[%107, %114] : f16 from vector<2x8xf16>
        affine.store %109, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %111, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %113, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %115, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %104:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = vector.extract %91#1[%107, 0] : f32 from vector<2x1xf32>
      %109 = vector.broadcast %108 : f32 to vector<1x4xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %111 = vector.insert %108, %arg5 [%107, %110] : f32 into vector<2x8xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %113 = vector.insert %108, %111 [%107, %112] : f32 into vector<2x8xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.insert %108, %113 [%107, %114] : f32 into vector<2x8xf32>
      %116 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %117 = vector.insert %108, %115 [%107, %116] : f32 into vector<2x8xf32>
      %118 = vector.extract %99[%107, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %99[%107, %112] : f32 from vector<2x8xf32>
      %120 = vector.extract %99[%107, %114] : f32 from vector<2x8xf32>
      %121 = vector.extract %99[%107, %116] : f32 from vector<2x8xf32>
      %122 = vector.from_elements %118, %119, %120, %121 : vector<1x4xf32>
      %123 = arith.divf %122, %109 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %124 = vector.extract %123[0, 0] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %arg6 [%107, %110] : f32 into vector<2x8xf32>
      %126 = vector.extract %123[0, 1] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %125 [%107, %112] : f32 into vector<2x8xf32>
      %128 = vector.extract %123[0, 2] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%107, %114] : f32 into vector<2x8xf32>
      %130 = vector.extract %123[0, 3] : f32 from vector<1x4xf32>
      %131 = vector.insert %130, %129 [%107, %116] : f32 into vector<2x8xf32>
      %132 = arith.truncf %123 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %133 = vector.extract %132[0, 0] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %arg7 [%107, %110] : f16 into vector<2x8xf16>
      %135 = vector.extract %132[0, 1] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%107, %112] : f16 into vector<2x8xf16>
      %137 = vector.extract %132[0, 2] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%107, %114] : f16 into vector<2x8xf16>
      %139 = vector.extract %132[0, 3] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%107, %116] : f16 into vector<2x8xf16>
      %141 = vector.insert %133, %arg8 [%107, %110] : f16 into vector<2x8xf16>
      %142 = vector.insert %135, %141 [%107, %112] : f16 into vector<2x8xf16>
      %143 = vector.insert %137, %142 [%107, %114] : f16 into vector<2x8xf16>
      %144 = vector.insert %139, %143 [%107, %116] : f16 into vector<2x8xf16>
      affine.yield %117, %131, %140, %144 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %103 {
      affine.for %arg4 = 0 to 4 {
        %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %109 = vector.extract %104#3[%107, %108] : f16 from vector<2x8xf16>
        %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %111 = vector.extract %104#3[%107, %110] : f16 from vector<2x8xf16>
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %113 = vector.extract %104#3[%107, %112] : f16 from vector<2x8xf16>
        %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %115 = vector.extract %104#3[%107, %114] : f16 from vector<2x8xf16>
        affine.store %109, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %111, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %113, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %115, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %105:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = vector.extract %91#1[%107, 0] : f32 from vector<2x1xf32>
      %109 = vector.broadcast %108 : f32 to vector<1x4xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %111 = vector.insert %108, %arg5 [%107, %110] : f32 into vector<2x8xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %113 = vector.insert %108, %111 [%107, %112] : f32 into vector<2x8xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.insert %108, %113 [%107, %114] : f32 into vector<2x8xf32>
      %116 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %117 = vector.insert %108, %115 [%107, %116] : f32 into vector<2x8xf32>
      %118 = vector.extract %98[%107, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %98[%107, %112] : f32 from vector<2x8xf32>
      %120 = vector.extract %98[%107, %114] : f32 from vector<2x8xf32>
      %121 = vector.extract %98[%107, %116] : f32 from vector<2x8xf32>
      %122 = vector.from_elements %118, %119, %120, %121 : vector<1x4xf32>
      %123 = arith.divf %122, %109 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %124 = vector.extract %123[0, 0] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %arg6 [%107, %110] : f32 into vector<2x8xf32>
      %126 = vector.extract %123[0, 1] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %125 [%107, %112] : f32 into vector<2x8xf32>
      %128 = vector.extract %123[0, 2] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%107, %114] : f32 into vector<2x8xf32>
      %130 = vector.extract %123[0, 3] : f32 from vector<1x4xf32>
      %131 = vector.insert %130, %129 [%107, %116] : f32 into vector<2x8xf32>
      %132 = arith.truncf %123 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %133 = vector.extract %132[0, 0] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %arg7 [%107, %110] : f16 into vector<2x8xf16>
      %135 = vector.extract %132[0, 1] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%107, %112] : f16 into vector<2x8xf16>
      %137 = vector.extract %132[0, 2] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%107, %114] : f16 into vector<2x8xf16>
      %139 = vector.extract %132[0, 3] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%107, %116] : f16 into vector<2x8xf16>
      %141 = vector.insert %133, %arg8 [%107, %110] : f16 into vector<2x8xf16>
      %142 = vector.insert %135, %141 [%107, %112] : f16 into vector<2x8xf16>
      %143 = vector.insert %137, %142 [%107, %114] : f16 into vector<2x8xf16>
      %144 = vector.insert %139, %143 [%107, %116] : f16 into vector<2x8xf16>
      affine.yield %117, %131, %140, %144 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %103 {
      affine.for %arg4 = 0 to 4 {
        %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %109 = vector.extract %105#3[%107, %108] : f16 from vector<2x8xf16>
        %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %111 = vector.extract %105#3[%107, %110] : f16 from vector<2x8xf16>
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %113 = vector.extract %105#3[%107, %112] : f16 from vector<2x8xf16>
        %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %115 = vector.extract %105#3[%107, %114] : f16 from vector<2x8xf16>
        affine.store %109, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %111, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %113, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %115, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %106:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = vector.extract %91#1[%107, 0] : f32 from vector<2x1xf32>
      %109 = vector.broadcast %108 : f32 to vector<1x4xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %111 = vector.insert %108, %arg5 [%107, %110] : f32 into vector<2x8xf32>
      %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %113 = vector.insert %108, %111 [%107, %112] : f32 into vector<2x8xf32>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.insert %108, %113 [%107, %114] : f32 into vector<2x8xf32>
      %116 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %117 = vector.insert %108, %115 [%107, %116] : f32 into vector<2x8xf32>
      %118 = vector.extract %97[%107, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %97[%107, %112] : f32 from vector<2x8xf32>
      %120 = vector.extract %97[%107, %114] : f32 from vector<2x8xf32>
      %121 = vector.extract %97[%107, %116] : f32 from vector<2x8xf32>
      %122 = vector.from_elements %118, %119, %120, %121 : vector<1x4xf32>
      %123 = arith.divf %122, %109 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %124 = vector.extract %123[0, 0] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %arg6 [%107, %110] : f32 into vector<2x8xf32>
      %126 = vector.extract %123[0, 1] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %125 [%107, %112] : f32 into vector<2x8xf32>
      %128 = vector.extract %123[0, 2] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%107, %114] : f32 into vector<2x8xf32>
      %130 = vector.extract %123[0, 3] : f32 from vector<1x4xf32>
      %131 = vector.insert %130, %129 [%107, %116] : f32 into vector<2x8xf32>
      %132 = arith.truncf %123 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %133 = vector.extract %132[0, 0] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %arg7 [%107, %110] : f16 into vector<2x8xf16>
      %135 = vector.extract %132[0, 1] : f16 from vector<1x4xf16>
      %136 = vector.insert %135, %134 [%107, %112] : f16 into vector<2x8xf16>
      %137 = vector.extract %132[0, 2] : f16 from vector<1x4xf16>
      %138 = vector.insert %137, %136 [%107, %114] : f16 into vector<2x8xf16>
      %139 = vector.extract %132[0, 3] : f16 from vector<1x4xf16>
      %140 = vector.insert %139, %138 [%107, %116] : f16 into vector<2x8xf16>
      %141 = vector.insert %133, %arg8 [%107, %110] : f16 into vector<2x8xf16>
      %142 = vector.insert %135, %141 [%107, %112] : f16 into vector<2x8xf16>
      %143 = vector.insert %137, %142 [%107, %114] : f16 into vector<2x8xf16>
      %144 = vector.insert %139, %143 [%107, %116] : f16 into vector<2x8xf16>
      affine.yield %117, %131, %140, %144 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %103 {
      affine.for %arg4 = 0 to 4 {
        %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %109 = vector.extract %106#3[%107, %108] : f16 from vector<2x8xf16>
        %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %111 = vector.extract %106#3[%107, %110] : f16 from vector<2x8xf16>
        %112 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %113 = vector.extract %106#3[%107, %112] : f16 from vector<2x8xf16>
        %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %115 = vector.extract %106#3[%107, %114] : f16 from vector<2x8xf16>
        affine.store %109, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %111, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %113, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %115, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
