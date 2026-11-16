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
      %101 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %102 = vector.insert %cst_7, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %103 = vector.insert %cst_7, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %104 = vector.insert %cst_7, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %101, %102, %103, %104 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %103 = vector.insert %cst, %arg5 [%102, %101] : f32 into vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %105 = vector.insert %cst, %103 [%102, %104] : f32 into vector<2x8xf32>
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %107 = vector.insert %cst, %105 [%102, %106] : f32 into vector<2x8xf32>
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %109 = vector.insert %cst, %107 [%102, %108] : f32 into vector<2x8xf32>
      affine.yield %109 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6) -> (vector<2x1xf32>) {
      %101 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %101 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %12 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 + s2 * 32 + s0 floordiv 32) mod 32)>()[%6, %7, %block_id_z]
    %13 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%6]
    %subview = memref.subview %arg0[0, %12, %13, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %14 = frisk.async_copy_gm_to_shm %subview to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %14 : !frisk.async_task
    %alloc_9 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %15 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%6, %7]
    %subview_10 = memref.subview %arg0[0, %15, %13, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %16 = frisk.async_copy_gm_to_shm %subview_10 to %alloc_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %thread_id_x = gpu.thread_id x
    %17 = affine.apply affine_map<()[s0] -> (s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>()[%thread_id_x]
    %18 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>()[%thread_id_x]
    %19 = memref.load %alloc[%17, %18] : memref<128x32xf16, 3>
    %20 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 1)>()[%thread_id_x]
    %21 = memref.load %alloc[%17, %20] : memref<128x32xf16, 3>
    %22 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 2)>()[%thread_id_x]
    %23 = memref.load %alloc[%17, %22] : memref<128x32xf16, 3>
    %24 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 3)>()[%thread_id_x]
    %25 = memref.load %alloc[%17, %24] : memref<128x32xf16, 3>
    %26 = vector.from_elements %19, %21, %23, %25 : vector<1x4xf16>
    %27:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %26) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %101 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %103 = memref.load %alloc[%101, %102] : memref<128x32xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %105 = memref.load %alloc[%101, %104] : memref<128x32xf16, 3>
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %107 = memref.load %alloc[%101, %106] : memref<128x32xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %109 = memref.load %alloc[%101, %108] : memref<128x32xf16, 3>
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
      %124 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %125 = vector.insert %8, %arg6 [%111, %112] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%111, %115] : f16 into vector<2x8xf16>
      %127 = vector.insert %8, %126 [%111, %118] : f16 into vector<2x8xf16>
      %128 = vector.insert %8, %127 [%111, %121] : f16 into vector<2x8xf16>
      %129 = arith.mulf %arg8, %124 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %130 = vector.extract %129[0, 0] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %arg7 [%111, %112] : f16 into vector<2x8xf16>
      %132 = vector.extract %129[0, 1] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%111, %115] : f16 into vector<2x8xf16>
      %134 = vector.extract %129[0, 2] : f16 from vector<1x4xf16>
      %135 = vector.insert %134, %133 [%111, %118] : f16 into vector<2x8xf16>
      %136 = vector.extract %129[0, 3] : f16 from vector<1x4xf16>
      %137 = vector.insert %136, %135 [%111, %121] : f16 into vector<2x8xf16>
      affine.yield %123, %128, %137, %110 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %28 = vector.broadcast %8 : f16 to vector<1x4xf16>
    %29 = arith.mulf %27#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %30 = vector.extract %29[0, 0] : f16 from vector<1x4xf16>
    %31 = vector.insert %30, %27#2 [1, 4] : f16 into vector<2x8xf16>
    %32 = vector.extract %29[0, 1] : f16 from vector<1x4xf16>
    %33 = vector.insert %32, %31 [1, 5] : f16 into vector<2x8xf16>
    %34 = vector.extract %29[0, 2] : f16 from vector<1x4xf16>
    %35 = vector.insert %34, %33 [1, 6] : f16 into vector<2x8xf16>
    %36 = vector.extract %29[0, 3] : f16 from vector<1x4xf16>
    %37 = vector.insert %36, %35 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %16 : !frisk.async_task
    %alloc_11 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_12 = memref.subview %arg0[0, %15, %13, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %38 = frisk.async_copy_gm_to_shm %subview_12 to %alloc_11 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %39 = memref.load %alloc_9[%17, %18] : memref<128x32xf16, 3>
    %40 = memref.load %alloc_9[%17, %20] : memref<128x32xf16, 3>
    %41 = memref.load %alloc_9[%17, %22] : memref<128x32xf16, 3>
    %42 = memref.load %alloc_9[%17, %24] : memref<128x32xf16, 3>
    %43 = vector.from_elements %39, %40, %41, %42 : vector<1x4xf16>
    %44:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %43) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %101 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %103 = memref.load %alloc_9[%101, %102] : memref<128x32xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %105 = memref.load %alloc_9[%101, %104] : memref<128x32xf16, 3>
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %107 = memref.load %alloc_9[%101, %106] : memref<128x32xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %109 = memref.load %alloc_9[%101, %108] : memref<128x32xf16, 3>
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
      %124 = vector.insert %8, %arg6 [%111, %112] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %124 [%111, %115] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%111, %118] : f16 into vector<2x8xf16>
      %127 = vector.insert %8, %126 [%111, %121] : f16 into vector<2x8xf16>
      %128 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
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
    %45 = arith.mulf %44#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %46 = vector.extract %45[0, 0] : f16 from vector<1x4xf16>
    %47 = vector.insert %46, %44#2 [1, 4] : f16 into vector<2x8xf16>
    %48 = vector.extract %45[0, 1] : f16 from vector<1x4xf16>
    %49 = vector.insert %48, %47 [1, 5] : f16 into vector<2x8xf16>
    %50 = vector.extract %45[0, 2] : f16 from vector<1x4xf16>
    %51 = vector.insert %50, %49 [1, 6] : f16 into vector<2x8xf16>
    %52 = vector.extract %45[0, 3] : f16 from vector<1x4xf16>
    %53 = vector.insert %52, %51 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %38 : !frisk.async_task
    %alloc_13 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_14 = memref.subview %arg0[0, %15, %13, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %54 = frisk.async_copy_gm_to_shm %subview_14 to %alloc_13 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %55 = memref.load %alloc_11[%17, %18] : memref<128x32xf16, 3>
    %56 = memref.load %alloc_11[%17, %20] : memref<128x32xf16, 3>
    %57 = memref.load %alloc_11[%17, %22] : memref<128x32xf16, 3>
    %58 = memref.load %alloc_11[%17, %24] : memref<128x32xf16, 3>
    %59 = vector.from_elements %55, %56, %57, %58 : vector<1x4xf16>
    %60:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %59) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %101 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %103 = memref.load %alloc_11[%101, %102] : memref<128x32xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %105 = memref.load %alloc_11[%101, %104] : memref<128x32xf16, 3>
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %107 = memref.load %alloc_11[%101, %106] : memref<128x32xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %109 = memref.load %alloc_11[%101, %108] : memref<128x32xf16, 3>
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
      %124 = vector.insert %8, %arg6 [%111, %112] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %124 [%111, %115] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%111, %118] : f16 into vector<2x8xf16>
      %127 = vector.insert %8, %126 [%111, %121] : f16 into vector<2x8xf16>
      %128 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
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
    %61 = arith.mulf %60#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %62 = vector.extract %61[0, 0] : f16 from vector<1x4xf16>
    %63 = vector.insert %62, %60#2 [1, 4] : f16 into vector<2x8xf16>
    %64 = vector.extract %61[0, 1] : f16 from vector<1x4xf16>
    %65 = vector.insert %64, %63 [1, 5] : f16 into vector<2x8xf16>
    %66 = vector.extract %61[0, 2] : f16 from vector<1x4xf16>
    %67 = vector.insert %66, %65 [1, 6] : f16 into vector<2x8xf16>
    %68 = vector.extract %61[0, 3] : f16 from vector<1x4xf16>
    %69 = vector.insert %68, %67 [1, 7] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %54 : !frisk.async_task
    %70 = memref.load %alloc_13[%17, %18] : memref<128x32xf16, 3>
    %71 = memref.load %alloc_13[%17, %20] : memref<128x32xf16, 3>
    %72 = memref.load %alloc_13[%17, %22] : memref<128x32xf16, 3>
    %73 = memref.load %alloc_13[%17, %24] : memref<128x32xf16, 3>
    %74 = vector.from_elements %70, %71, %72, %73 : vector<1x4xf16>
    %75:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %74) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %101 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %103 = memref.load %alloc_13[%101, %102] : memref<128x32xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %105 = memref.load %alloc_13[%101, %104] : memref<128x32xf16, 3>
      %106 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %107 = memref.load %alloc_13[%101, %106] : memref<128x32xf16, 3>
      %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %109 = memref.load %alloc_13[%101, %108] : memref<128x32xf16, 3>
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
      %124 = vector.insert %8, %arg6 [%111, %112] : f16 into vector<2x8xf16>
      %125 = vector.insert %8, %124 [%111, %115] : f16 into vector<2x8xf16>
      %126 = vector.insert %8, %125 [%111, %118] : f16 into vector<2x8xf16>
      %127 = vector.insert %8, %126 [%111, %121] : f16 into vector<2x8xf16>
      %128 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
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
    %76 = arith.mulf %75#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
    %78 = vector.insert %77, %75#2 [1, 4] : f16 into vector<2x8xf16>
    %79 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
    %80 = vector.insert %79, %78 [1, 5] : f16 into vector<2x8xf16>
    %81 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
    %82 = vector.insert %81, %80 [1, 6] : f16 into vector<2x8xf16>
    %83 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
    %84 = vector.insert %83, %82 [1, 7] : f16 into vector<2x8xf16>
    %85:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#3, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %101 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %139 = vector.insert %cst, %arg12 [%138, %137] : f32 into vector<2x32xf32>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %141 = vector.insert %cst, %139 [%138, %140] : f32 into vector<2x32xf32>
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %143 = vector.insert %cst, %141 [%138, %142] : f32 into vector<2x32xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %145 = vector.insert %cst, %143 [%138, %144] : f32 into vector<2x32xf32>
        affine.yield %145 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %102 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
      %subview_16 = memref.subview %transpose[0, %102, 0, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %103 = frisk.async_copy_gm_to_shm %subview_16 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %103 : !frisk.async_task
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 524288 + s1 * 16777216 + 131072) floordiv 524288) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_18 = memref.subview %transpose[0, %104, 32, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %105 = frisk.async_copy_gm_to_shm %subview_18 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %106:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %138 = memref.load %alloc_15[%18, %137] : memref<32x128xf16, 3>
        %139 = memref.load %alloc_15[%20, %137] : memref<32x128xf16, 3>
        %140 = memref.load %alloc_15[%22, %137] : memref<32x128xf16, 3>
        %141 = memref.load %alloc_15[%24, %137] : memref<32x128xf16, 3>
        %142 = vector.from_elements %138, %139, %140, %141 : vector<4x1xf16>
        %143:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %142) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %173 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %174 = memref.load %alloc_15[%173, %137] : memref<32x128xf16, 3>
          %175 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %176 = memref.load %alloc_15[%175, %137] : memref<32x128xf16, 3>
          %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %178 = memref.load %alloc_15[%177, %137] : memref<32x128xf16, 3>
          %179 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %180 = memref.load %alloc_15[%179, %137] : memref<32x128xf16, 3>
          %181 = vector.from_elements %174, %176, %178, %180 : vector<4x1xf16>
          %182 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %183 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %184 = vector.extract %37[%182, %183] : f16 from vector<2x8xf16>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %186 = vector.extract %37[%182, %185] : f16 from vector<2x8xf16>
          %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %188 = vector.extract %37[%182, %187] : f16 from vector<2x8xf16>
          %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %190 = vector.extract %37[%182, %189] : f16 from vector<2x8xf16>
          %191 = vector.from_elements %184, %186, %188, %190 : vector<1x4xf16>
          %192 = frisk.warp_mma_rr %191, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %192, %181 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %145 = vector.extract %37[%144, 4] : f16 from vector<2x8xf16>
        %146 = vector.extract %37[%144, 5] : f16 from vector<2x8xf16>
        %147 = vector.extract %37[%144, 6] : f16 from vector<2x8xf16>
        %148 = vector.extract %37[%144, 7] : f16 from vector<2x8xf16>
        %149 = vector.from_elements %145, %146, %147, %148 : vector<1x4xf16>
        %150 = frisk.warp_mma_rr %149, %143#1, %143#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %151 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %152 = vector.extract %101[%144, %151] : f32 from vector<2x32xf32>
        %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %154 = vector.extract %101[%144, %153] : f32 from vector<2x32xf32>
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %156 = vector.extract %101[%144, %155] : f32 from vector<2x32xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %158 = vector.extract %101[%144, %157] : f32 from vector<2x32xf32>
        %159 = vector.from_elements %152, %154, %156, %158 : vector<1x4xf32>
        %160 = arith.addf %159, %150 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %161 = vector.extract %160[0, 0] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %arg12 [%144, %151] : f32 into vector<2x32xf32>
        %163 = vector.extract %160[0, 1] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %162 [%144, %153] : f32 into vector<2x32xf32>
        %165 = vector.extract %160[0, 2] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%144, %155] : f32 into vector<2x32xf32>
        %167 = vector.extract %160[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%144, %157] : f32 into vector<2x32xf32>
        %169 = vector.insert %161, %arg13 [%144, %151] : f32 into vector<2x32xf32>
        %170 = vector.insert %163, %169 [%144, %153] : f32 into vector<2x32xf32>
        %171 = vector.insert %165, %170 [%144, %155] : f32 into vector<2x32xf32>
        %172 = vector.insert %167, %171 [%144, %157] : f32 into vector<2x32xf32>
        affine.yield %168, %172 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %105 : !frisk.async_task
      %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %107 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 524288 + s1 * 16777216 + 262144) floordiv 524288) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_20 = memref.subview %transpose[0, %107, 64, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %108 = frisk.async_copy_gm_to_shm %subview_20 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %109:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %138 = memref.load %alloc_17[%18, %137] : memref<32x128xf16, 3>
        %139 = memref.load %alloc_17[%20, %137] : memref<32x128xf16, 3>
        %140 = memref.load %alloc_17[%22, %137] : memref<32x128xf16, 3>
        %141 = memref.load %alloc_17[%24, %137] : memref<32x128xf16, 3>
        %142 = vector.from_elements %138, %139, %140, %141 : vector<4x1xf16>
        %143:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %142) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %173 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %174 = memref.load %alloc_17[%173, %137] : memref<32x128xf16, 3>
          %175 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %176 = memref.load %alloc_17[%175, %137] : memref<32x128xf16, 3>
          %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %178 = memref.load %alloc_17[%177, %137] : memref<32x128xf16, 3>
          %179 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %180 = memref.load %alloc_17[%179, %137] : memref<32x128xf16, 3>
          %181 = vector.from_elements %174, %176, %178, %180 : vector<4x1xf16>
          %182 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %183 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %184 = vector.extract %53[%182, %183] : f16 from vector<2x8xf16>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %186 = vector.extract %53[%182, %185] : f16 from vector<2x8xf16>
          %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %188 = vector.extract %53[%182, %187] : f16 from vector<2x8xf16>
          %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %190 = vector.extract %53[%182, %189] : f16 from vector<2x8xf16>
          %191 = vector.from_elements %184, %186, %188, %190 : vector<1x4xf16>
          %192 = frisk.warp_mma_rr %191, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %192, %181 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %145 = vector.extract %53[%144, 4] : f16 from vector<2x8xf16>
        %146 = vector.extract %53[%144, 5] : f16 from vector<2x8xf16>
        %147 = vector.extract %53[%144, 6] : f16 from vector<2x8xf16>
        %148 = vector.extract %53[%144, 7] : f16 from vector<2x8xf16>
        %149 = vector.from_elements %145, %146, %147, %148 : vector<1x4xf16>
        %150 = frisk.warp_mma_rr %149, %143#1, %143#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %151 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %152 = vector.extract %106#1[%144, %151] : f32 from vector<2x32xf32>
        %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %154 = vector.extract %106#1[%144, %153] : f32 from vector<2x32xf32>
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %156 = vector.extract %106#1[%144, %155] : f32 from vector<2x32xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %158 = vector.extract %106#1[%144, %157] : f32 from vector<2x32xf32>
        %159 = vector.from_elements %152, %154, %156, %158 : vector<1x4xf32>
        %160 = arith.addf %159, %150 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %161 = vector.extract %160[0, 0] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %arg12 [%144, %151] : f32 into vector<2x32xf32>
        %163 = vector.extract %160[0, 1] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %162 [%144, %153] : f32 into vector<2x32xf32>
        %165 = vector.extract %160[0, 2] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%144, %155] : f32 into vector<2x32xf32>
        %167 = vector.extract %160[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%144, %157] : f32 into vector<2x32xf32>
        %169 = vector.insert %161, %arg13 [%144, %151] : f32 into vector<2x32xf32>
        %170 = vector.insert %163, %169 [%144, %153] : f32 into vector<2x32xf32>
        %171 = vector.insert %165, %170 [%144, %155] : f32 into vector<2x32xf32>
        %172 = vector.insert %167, %171 [%144, %157] : f32 into vector<2x32xf32>
        affine.yield %168, %172 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %108 : !frisk.async_task
      %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %110 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 524288 + s1 * 16777216 + 393216) floordiv 524288) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_22 = memref.subview %transpose[0, %110, 96, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %111 = frisk.async_copy_gm_to_shm %subview_22 to %alloc_21 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %112:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %138 = memref.load %alloc_19[%18, %137] : memref<32x128xf16, 3>
        %139 = memref.load %alloc_19[%20, %137] : memref<32x128xf16, 3>
        %140 = memref.load %alloc_19[%22, %137] : memref<32x128xf16, 3>
        %141 = memref.load %alloc_19[%24, %137] : memref<32x128xf16, 3>
        %142 = vector.from_elements %138, %139, %140, %141 : vector<4x1xf16>
        %143:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %142) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %173 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %174 = memref.load %alloc_19[%173, %137] : memref<32x128xf16, 3>
          %175 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %176 = memref.load %alloc_19[%175, %137] : memref<32x128xf16, 3>
          %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %178 = memref.load %alloc_19[%177, %137] : memref<32x128xf16, 3>
          %179 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %180 = memref.load %alloc_19[%179, %137] : memref<32x128xf16, 3>
          %181 = vector.from_elements %174, %176, %178, %180 : vector<4x1xf16>
          %182 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %183 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %184 = vector.extract %69[%182, %183] : f16 from vector<2x8xf16>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %186 = vector.extract %69[%182, %185] : f16 from vector<2x8xf16>
          %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %188 = vector.extract %69[%182, %187] : f16 from vector<2x8xf16>
          %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %190 = vector.extract %69[%182, %189] : f16 from vector<2x8xf16>
          %191 = vector.from_elements %184, %186, %188, %190 : vector<1x4xf16>
          %192 = frisk.warp_mma_rr %191, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %192, %181 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %145 = vector.extract %69[%144, 4] : f16 from vector<2x8xf16>
        %146 = vector.extract %69[%144, 5] : f16 from vector<2x8xf16>
        %147 = vector.extract %69[%144, 6] : f16 from vector<2x8xf16>
        %148 = vector.extract %69[%144, 7] : f16 from vector<2x8xf16>
        %149 = vector.from_elements %145, %146, %147, %148 : vector<1x4xf16>
        %150 = frisk.warp_mma_rr %149, %143#1, %143#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %151 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %152 = vector.extract %109#1[%144, %151] : f32 from vector<2x32xf32>
        %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %154 = vector.extract %109#1[%144, %153] : f32 from vector<2x32xf32>
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %156 = vector.extract %109#1[%144, %155] : f32 from vector<2x32xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %158 = vector.extract %109#1[%144, %157] : f32 from vector<2x32xf32>
        %159 = vector.from_elements %152, %154, %156, %158 : vector<1x4xf32>
        %160 = arith.addf %159, %150 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %161 = vector.extract %160[0, 0] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %arg12 [%144, %151] : f32 into vector<2x32xf32>
        %163 = vector.extract %160[0, 1] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %162 [%144, %153] : f32 into vector<2x32xf32>
        %165 = vector.extract %160[0, 2] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%144, %155] : f32 into vector<2x32xf32>
        %167 = vector.extract %160[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%144, %157] : f32 into vector<2x32xf32>
        %169 = vector.insert %161, %arg13 [%144, %151] : f32 into vector<2x32xf32>
        %170 = vector.insert %163, %169 [%144, %153] : f32 into vector<2x32xf32>
        %171 = vector.insert %165, %170 [%144, %155] : f32 into vector<2x32xf32>
        %172 = vector.insert %167, %171 [%144, %157] : f32 into vector<2x32xf32>
        affine.yield %168, %172 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %111 : !frisk.async_task
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_24 = memref.subview %arg2[0, %102, %arg4, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %113 = frisk.async_copy_gm_to_shm %subview_24 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %114:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %138 = memref.load %alloc_21[%18, %137] : memref<32x128xf16, 3>
        %139 = memref.load %alloc_21[%20, %137] : memref<32x128xf16, 3>
        %140 = memref.load %alloc_21[%22, %137] : memref<32x128xf16, 3>
        %141 = memref.load %alloc_21[%24, %137] : memref<32x128xf16, 3>
        %142 = vector.from_elements %138, %139, %140, %141 : vector<4x1xf16>
        %143:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %142) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %173 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %174 = memref.load %alloc_21[%173, %137] : memref<32x128xf16, 3>
          %175 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %176 = memref.load %alloc_21[%175, %137] : memref<32x128xf16, 3>
          %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %178 = memref.load %alloc_21[%177, %137] : memref<32x128xf16, 3>
          %179 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %180 = memref.load %alloc_21[%179, %137] : memref<32x128xf16, 3>
          %181 = vector.from_elements %174, %176, %178, %180 : vector<4x1xf16>
          %182 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %183 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %184 = vector.extract %84[%182, %183] : f16 from vector<2x8xf16>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %186 = vector.extract %84[%182, %185] : f16 from vector<2x8xf16>
          %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %188 = vector.extract %84[%182, %187] : f16 from vector<2x8xf16>
          %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %190 = vector.extract %84[%182, %189] : f16 from vector<2x8xf16>
          %191 = vector.from_elements %184, %186, %188, %190 : vector<1x4xf16>
          %192 = frisk.warp_mma_rr %191, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %192, %181 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %144 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %145 = vector.extract %84[%144, 4] : f16 from vector<2x8xf16>
        %146 = vector.extract %84[%144, 5] : f16 from vector<2x8xf16>
        %147 = vector.extract %84[%144, 6] : f16 from vector<2x8xf16>
        %148 = vector.extract %84[%144, 7] : f16 from vector<2x8xf16>
        %149 = vector.from_elements %145, %146, %147, %148 : vector<1x4xf16>
        %150 = frisk.warp_mma_rr %149, %143#1, %143#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %151 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %152 = vector.extract %112#1[%144, %151] : f32 from vector<2x32xf32>
        %153 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %154 = vector.extract %112#1[%144, %153] : f32 from vector<2x32xf32>
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %156 = vector.extract %112#1[%144, %155] : f32 from vector<2x32xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %158 = vector.extract %112#1[%144, %157] : f32 from vector<2x32xf32>
        %159 = vector.from_elements %152, %154, %156, %158 : vector<1x4xf32>
        %160 = arith.addf %159, %150 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %161 = vector.extract %160[0, 0] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %arg12 [%144, %151] : f32 into vector<2x32xf32>
        %163 = vector.extract %160[0, 1] : f32 from vector<1x4xf32>
        %164 = vector.insert %163, %162 [%144, %153] : f32 into vector<2x32xf32>
        %165 = vector.extract %160[0, 2] : f32 from vector<1x4xf32>
        %166 = vector.insert %165, %164 [%144, %155] : f32 into vector<2x32xf32>
        %167 = vector.extract %160[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %166 [%144, %157] : f32 into vector<2x32xf32>
        %169 = vector.insert %161, %arg13 [%144, %151] : f32 into vector<2x32xf32>
        %170 = vector.insert %163, %169 [%144, %153] : f32 into vector<2x32xf32>
        %171 = vector.insert %165, %170 [%144, %155] : f32 into vector<2x32xf32>
        %172 = vector.insert %167, %171 [%144, %157] : f32 into vector<2x32xf32>
        affine.yield %168, %172 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %115 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %139 = vector.extract %114#1[%137, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %141 = vector.extract %114#1[%137, %140] : f32 from vector<2x32xf32>
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %143 = vector.extract %114#1[%137, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %145 = vector.extract %114#1[%137, %144] : f32 from vector<2x32xf32>
        %146 = vector.insert %139, %arg12 [%137, %138] : f32 into vector<2x32xf32>
        %147 = vector.insert %141, %146 [%137, %140] : f32 into vector<2x32xf32>
        %148 = vector.insert %143, %147 [%137, %142] : f32 into vector<2x32xf32>
        %149 = vector.insert %145, %148 [%137, %144] : f32 into vector<2x32xf32>
        affine.yield %149 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %116 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %137 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_7) -> (f32) {
          %141 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %142 = vector.extract %115[%arg11, %141] : f32 from vector<2x32xf32>
          %143 = arith.maxnumf %arg14, %142 : f32
          %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %145 = vector.extract %115[%arg11, %144] : f32 from vector<2x32xf32>
          %146 = arith.maxnumf %143, %145 : f32
          %147 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %148 = vector.extract %115[%arg11, %147] : f32 from vector<2x32xf32>
          %149 = arith.maxnumf %146, %148 : f32
          %150 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %151 = vector.extract %115[%arg11, %150] : f32 from vector<2x32xf32>
          %152 = arith.maxnumf %149, %151 : f32
          affine.yield %152 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %137, %c16_i32, %c64_i32 : f32
        %138 = arith.maxnumf %137, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %138, %c32_i32, %c64_i32 : f32
        %139 = arith.maxnumf %138, %shuffleResult_31 : f32
        %140 = vector.insert %139, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %140 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %117 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %137 = vector.extract %116[%arg11, 0] : f32 from vector<2x1xf32>
        %138 = vector.broadcast %137 : f32 to vector<1x1xf32>
        %139 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %140 = vector.broadcast %139 : f32 to vector<1x1xf32>
        %141 = arith.maxnumf %138, %140 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %142 = vector.extract %141[0, 0] : f32 from vector<1x1xf32>
        %143 = vector.insert %142, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %143 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %118:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %138 = vector.extract %117[%137, 0] : f32 from vector<2x1xf32>
        %139 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3) -> (vector<1x4xf32>) {
          %166 = vector.broadcast %138 : f32 to vector<1x4xf32>
          affine.yield %166 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %141 = vector.extract %139[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg12 [%137, %140] : f32 into vector<2x32xf32>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %144 = vector.extract %139[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %142 [%137, %143] : f32 into vector<2x32xf32>
        %146 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %147 = vector.extract %139[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %145 [%137, %146] : f32 into vector<2x32xf32>
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %150 = vector.extract %139[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %148 [%137, %149] : f32 into vector<2x32xf32>
        %152 = vector.extract %114#1[%137, %140] : f32 from vector<2x32xf32>
        %153 = vector.extract %114#1[%137, %143] : f32 from vector<2x32xf32>
        %154 = vector.extract %114#1[%137, %146] : f32 from vector<2x32xf32>
        %155 = vector.extract %114#1[%137, %149] : f32 from vector<2x32xf32>
        %156 = vector.from_elements %152, %153, %154, %155 : vector<1x4xf32>
        %157 = arith.subf %156, %139 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg13 [%137, %140] : f32 into vector<2x32xf32>
        %160 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%137, %143] : f32 into vector<2x32xf32>
        %162 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %163 = vector.insert %162, %161 [%137, %146] : f32 into vector<2x32xf32>
        %164 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %163 [%137, %149] : f32 into vector<2x32xf32>
        affine.yield %151, %165 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %119:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6, %arg14 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %137 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %138 = vector.broadcast %137 : f32 to vector<1x1xf32>
        %139 = vector.extract %117[%arg11, 0] : f32 from vector<2x1xf32>
        %140 = vector.broadcast %139 : f32 to vector<1x1xf32>
        %141 = arith.subf %138, %140 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %142 = vector.extract %141[0, 0] : f32 from vector<1x1xf32>
        %143 = vector.insert %142, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %144 = math.exp2 %141 {tiled = true} : vector<1x1xf32>
        %145 = vector.extract %144[0, 0] : f32 from vector<1x1xf32>
        %146 = vector.insert %145, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %147 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %148 = vector.broadcast %147 : f32 to vector<1x1xf32>
        %149 = arith.mulf %148, %144 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %150 = vector.extract %149[0, 0] : f32 from vector<1x1xf32>
        %151 = vector.insert %150, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %143, %146, %151 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %120:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %139 = vector.extract %118#1[%137, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %141 = vector.extract %118#1[%137, %140] : f32 from vector<2x32xf32>
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %143 = vector.extract %118#1[%137, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %145 = vector.extract %118#1[%137, %144] : f32 from vector<2x32xf32>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<1x4xf32>
        %147 = math.exp2 %146 {tiled = true} : vector<1x4xf32>
        %148 = vector.extract %147[0, 0] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %arg12 [%137, %138] : f32 into vector<2x32xf32>
        %150 = vector.extract %147[0, 1] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%137, %140] : f32 into vector<2x32xf32>
        %152 = vector.extract %147[0, 2] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %151 [%137, %142] : f32 into vector<2x32xf32>
        %154 = vector.extract %147[0, 3] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %153 [%137, %144] : f32 into vector<2x32xf32>
        %156 = vector.insert %148, %arg13 [%137, %138] : f32 into vector<2x32xf32>
        %157 = vector.insert %150, %156 [%137, %140] : f32 into vector<2x32xf32>
        %158 = vector.insert %152, %157 [%137, %142] : f32 into vector<2x32xf32>
        %159 = vector.insert %154, %158 [%137, %144] : f32 into vector<2x32xf32>
        affine.yield %155, %159 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %121 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %137 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst) -> (f32) {
          %141 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %142 = vector.extract %120#1[%arg11, %141] : f32 from vector<2x32xf32>
          %143 = arith.addf %arg14, %142 : f32
          %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %145 = vector.extract %120#1[%arg11, %144] : f32 from vector<2x32xf32>
          %146 = arith.addf %143, %145 : f32
          %147 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %148 = vector.extract %120#1[%arg11, %147] : f32 from vector<2x32xf32>
          %149 = arith.addf %146, %148 : f32
          %150 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %151 = vector.extract %120#1[%arg11, %150] : f32 from vector<2x32xf32>
          %152 = arith.addf %149, %151 : f32
          affine.yield %152 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %137, %c16_i32, %c64_i32 : f32
        %138 = arith.addf %137, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %138, %c32_i32, %c64_i32 : f32
        %139 = arith.addf %138, %shuffleResult_31 : f32
        %140 = vector.insert %139, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %140 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %122 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %137 = vector.extract %121[%arg11, 0] : f32 from vector<2x1xf32>
        %138 = vector.broadcast %137 : f32 to vector<1x1xf32>
        %139 = vector.extract %119#2[%arg11, 0] : f32 from vector<2x1xf32>
        %140 = vector.broadcast %139 : f32 to vector<1x1xf32>
        %141 = arith.addf %138, %140 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %142 = vector.extract %141[0, 0] : f32 from vector<1x1xf32>
        %143 = vector.insert %142, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %143 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %123 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %137 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %139 = vector.extract %120#0[%137, %138] : f32 from vector<2x32xf32>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %141 = vector.extract %120#0[%137, %140] : f32 from vector<2x32xf32>
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %143 = vector.extract %120#0[%137, %142] : f32 from vector<2x32xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %145 = vector.extract %120#0[%137, %144] : f32 from vector<2x32xf32>
        %146 = vector.from_elements %139, %141, %143, %145 : vector<1x4xf32>
        %147 = arith.truncf %146 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %148 = vector.extract %147[0, 0] : f16 from vector<1x4xf16>
        %149 = vector.insert %148, %arg12 [%137, %138] : f16 into vector<2x32xf16>
        %150 = vector.extract %147[0, 1] : f16 from vector<1x4xf16>
        %151 = vector.insert %150, %149 [%137, %140] : f16 into vector<2x32xf16>
        %152 = vector.extract %147[0, 2] : f16 from vector<1x4xf16>
        %153 = vector.insert %152, %151 [%137, %142] : f16 into vector<2x32xf16>
        %154 = vector.extract %147[0, 3] : f16 from vector<1x4xf16>
        %155 = vector.insert %154, %153 [%137, %144] : f16 into vector<2x32xf16>
        affine.yield %155 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %113 : !frisk.async_task
      %124 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %137 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %138 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %139 = frisk.read_matrix_from_shm %alloc_23[%137, %138] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %140 = vector.bitcast %139 : vector<4xi32> to vector<8xf16>
        %141 = vector.extract %140[0] : f16 from vector<8xf16>
        %142 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %143 = vector.extract %140[%142] : f16 from vector<8xf16>
        %144 = vector.extract %140[1] : f16 from vector<8xf16>
        %145 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %146 = vector.extract %140[%145] : f16 from vector<8xf16>
        %147 = vector.extract %140[2] : f16 from vector<8xf16>
        %148 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %149 = vector.extract %140[%148] : f16 from vector<8xf16>
        %150 = vector.extract %140[3] : f16 from vector<8xf16>
        %151 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %152 = vector.extract %140[%151] : f16 from vector<8xf16>
        %153 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %154 = vector.insert %141, %arg12 [%153, 0] : f16 into vector<32x2xf16>
        %155 = vector.insert %143, %154 [%153, 1] : f16 into vector<32x2xf16>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %157 = vector.insert %144, %155 [%156, 0] : f16 into vector<32x2xf16>
        %158 = vector.insert %146, %157 [%156, 1] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %160 = vector.insert %147, %158 [%159, 0] : f16 into vector<32x2xf16>
        %161 = vector.insert %149, %160 [%159, 1] : f16 into vector<32x2xf16>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %163 = vector.insert %150, %161 [%162, 0] : f16 into vector<32x2xf16>
        %164 = vector.insert %152, %163 [%162, 1] : f16 into vector<32x2xf16>
        affine.yield %164 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %alloc_25 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_26 = memref.subview %arg2[0, %102, %arg4, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %125 = frisk.async_copy_gm_to_shm %subview_26 to %alloc_25 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %126:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %137 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %184 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %185 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %186 = vector.extract %123[%184, %185] : f16 from vector<2x32xf16>
          %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %188 = vector.extract %123[%184, %187] : f16 from vector<2x32xf16>
          %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %190 = vector.extract %123[%184, %189] : f16 from vector<2x32xf16>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %192 = vector.extract %123[%184, %191] : f16 from vector<2x32xf16>
          %193 = vector.from_elements %186, %188, %190, %192 : vector<1x4xf16>
          %194 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %195 = vector.extract %124[%185, %194] : f16 from vector<32x2xf16>
          %196 = vector.extract %124[%187, %194] : f16 from vector<32x2xf16>
          %197 = vector.extract %124[%189, %194] : f16 from vector<32x2xf16>
          %198 = vector.extract %124[%191, %194] : f16 from vector<32x2xf16>
          %199 = vector.from_elements %195, %196, %197, %198 : vector<4x1xf16>
          %200 = frisk.warp_mma_rr %193, %199, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %200 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %138 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        %151 = vector.extract %119#1[%138, 0] : f32 from vector<2x1xf32>
        %152 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %184 = vector.broadcast %151 : f32 to vector<1x4xf32>
          affine.yield %184 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg13 [%138, %139] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%138, %142] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%138, %145] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%138, %148] : f32 into vector<2x8xf32>
        %161 = vector.extract %arg5[%138, %139] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg5[%138, %142] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg5[%138, %145] : f32 from vector<2x8xf32>
        %164 = vector.extract %arg5[%138, %148] : f32 from vector<2x8xf32>
        %165 = vector.from_elements %161, %162, %163, %164 : vector<1x4xf32>
        %166 = arith.mulf %165, %152 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %167 = vector.extract %166[0, 0] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %arg14 [%138, %139] : f32 into vector<2x8xf32>
        %169 = vector.extract %166[0, 1] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%138, %142] : f32 into vector<2x8xf32>
        %171 = vector.extract %166[0, 2] : f32 from vector<1x4xf32>
        %172 = vector.insert %171, %170 [%138, %145] : f32 into vector<2x8xf32>
        %173 = vector.extract %166[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %172 [%138, %148] : f32 into vector<2x8xf32>
        %175 = arith.addf %166, %137 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %arg15 [%138, %139] : f32 into vector<2x8xf32>
        %178 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%138, %142] : f32 into vector<2x8xf32>
        %180 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%138, %145] : f32 into vector<2x8xf32>
        %182 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %181 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150, %160, %174, %183 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %125 : !frisk.async_task
      %127 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %137 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %138 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %139 = frisk.read_matrix_from_shm %alloc_25[%137, %138] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %140 = vector.bitcast %139 : vector<4xi32> to vector<8xf16>
        %141 = vector.extract %140[0] : f16 from vector<8xf16>
        %142 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %143 = vector.extract %140[%142] : f16 from vector<8xf16>
        %144 = vector.extract %140[1] : f16 from vector<8xf16>
        %145 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %146 = vector.extract %140[%145] : f16 from vector<8xf16>
        %147 = vector.extract %140[2] : f16 from vector<8xf16>
        %148 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %149 = vector.extract %140[%148] : f16 from vector<8xf16>
        %150 = vector.extract %140[3] : f16 from vector<8xf16>
        %151 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %152 = vector.extract %140[%151] : f16 from vector<8xf16>
        %153 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %154 = vector.insert %141, %arg12 [%153, 0] : f16 into vector<32x2xf16>
        %155 = vector.insert %143, %154 [%153, 1] : f16 into vector<32x2xf16>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %157 = vector.insert %144, %155 [%156, 0] : f16 into vector<32x2xf16>
        %158 = vector.insert %146, %157 [%156, 1] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %160 = vector.insert %147, %158 [%159, 0] : f16 into vector<32x2xf16>
        %161 = vector.insert %149, %160 [%159, 1] : f16 into vector<32x2xf16>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %163 = vector.insert %150, %161 [%162, 0] : f16 into vector<32x2xf16>
        %164 = vector.insert %152, %163 [%162, 1] : f16 into vector<32x2xf16>
        affine.yield %164 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %alloc_27 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_28 = memref.subview %arg2[0, %102, %arg4, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %128 = frisk.async_copy_gm_to_shm %subview_28 to %alloc_27 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %129:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %137 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %184 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %185 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %186 = vector.extract %123[%184, %185] : f16 from vector<2x32xf16>
          %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %188 = vector.extract %123[%184, %187] : f16 from vector<2x32xf16>
          %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %190 = vector.extract %123[%184, %189] : f16 from vector<2x32xf16>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %192 = vector.extract %123[%184, %191] : f16 from vector<2x32xf16>
          %193 = vector.from_elements %186, %188, %190, %192 : vector<1x4xf16>
          %194 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %195 = vector.extract %127[%185, %194] : f16 from vector<32x2xf16>
          %196 = vector.extract %127[%187, %194] : f16 from vector<32x2xf16>
          %197 = vector.extract %127[%189, %194] : f16 from vector<32x2xf16>
          %198 = vector.extract %127[%191, %194] : f16 from vector<32x2xf16>
          %199 = vector.from_elements %195, %196, %197, %198 : vector<4x1xf16>
          %200 = frisk.warp_mma_rr %193, %199, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %200 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %138 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        %151 = vector.extract %119#1[%138, 0] : f32 from vector<2x1xf32>
        %152 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %184 = vector.broadcast %151 : f32 to vector<1x4xf32>
          affine.yield %184 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg13 [%138, %139] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%138, %142] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%138, %145] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%138, %148] : f32 into vector<2x8xf32>
        %161 = vector.extract %arg8[%138, %139] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg8[%138, %142] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg8[%138, %145] : f32 from vector<2x8xf32>
        %164 = vector.extract %arg8[%138, %148] : f32 from vector<2x8xf32>
        %165 = vector.from_elements %161, %162, %163, %164 : vector<1x4xf32>
        %166 = arith.mulf %165, %152 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %167 = vector.extract %166[0, 0] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %arg14 [%138, %139] : f32 into vector<2x8xf32>
        %169 = vector.extract %166[0, 1] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%138, %142] : f32 into vector<2x8xf32>
        %171 = vector.extract %166[0, 2] : f32 from vector<1x4xf32>
        %172 = vector.insert %171, %170 [%138, %145] : f32 into vector<2x8xf32>
        %173 = vector.extract %166[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %172 [%138, %148] : f32 into vector<2x8xf32>
        %175 = arith.addf %166, %137 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %arg15 [%138, %139] : f32 into vector<2x8xf32>
        %178 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%138, %142] : f32 into vector<2x8xf32>
        %180 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%138, %145] : f32 into vector<2x8xf32>
        %182 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %181 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150, %160, %174, %183 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %128 : !frisk.async_task
      %130 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %137 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %138 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %139 = frisk.read_matrix_from_shm %alloc_27[%137, %138] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %140 = vector.bitcast %139 : vector<4xi32> to vector<8xf16>
        %141 = vector.extract %140[0] : f16 from vector<8xf16>
        %142 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %143 = vector.extract %140[%142] : f16 from vector<8xf16>
        %144 = vector.extract %140[1] : f16 from vector<8xf16>
        %145 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %146 = vector.extract %140[%145] : f16 from vector<8xf16>
        %147 = vector.extract %140[2] : f16 from vector<8xf16>
        %148 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %149 = vector.extract %140[%148] : f16 from vector<8xf16>
        %150 = vector.extract %140[3] : f16 from vector<8xf16>
        %151 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %152 = vector.extract %140[%151] : f16 from vector<8xf16>
        %153 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %154 = vector.insert %141, %arg12 [%153, 0] : f16 into vector<32x2xf16>
        %155 = vector.insert %143, %154 [%153, 1] : f16 into vector<32x2xf16>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %157 = vector.insert %144, %155 [%156, 0] : f16 into vector<32x2xf16>
        %158 = vector.insert %146, %157 [%156, 1] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %160 = vector.insert %147, %158 [%159, 0] : f16 into vector<32x2xf16>
        %161 = vector.insert %149, %160 [%159, 1] : f16 into vector<32x2xf16>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %163 = vector.insert %150, %161 [%162, 0] : f16 into vector<32x2xf16>
        %164 = vector.insert %152, %163 [%162, 1] : f16 into vector<32x2xf16>
        affine.yield %164 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %alloc_29 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_30 = memref.subview %arg2[0, %102, %arg4, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %131 = frisk.async_copy_gm_to_shm %subview_30 to %alloc_29 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %132:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %137 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %184 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %185 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %186 = vector.extract %123[%184, %185] : f16 from vector<2x32xf16>
          %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %188 = vector.extract %123[%184, %187] : f16 from vector<2x32xf16>
          %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %190 = vector.extract %123[%184, %189] : f16 from vector<2x32xf16>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %192 = vector.extract %123[%184, %191] : f16 from vector<2x32xf16>
          %193 = vector.from_elements %186, %188, %190, %192 : vector<1x4xf16>
          %194 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %195 = vector.extract %130[%185, %194] : f16 from vector<32x2xf16>
          %196 = vector.extract %130[%187, %194] : f16 from vector<32x2xf16>
          %197 = vector.extract %130[%189, %194] : f16 from vector<32x2xf16>
          %198 = vector.extract %130[%191, %194] : f16 from vector<32x2xf16>
          %199 = vector.from_elements %195, %196, %197, %198 : vector<4x1xf16>
          %200 = frisk.warp_mma_rr %193, %199, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %200 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %138 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        %151 = vector.extract %119#1[%138, 0] : f32 from vector<2x1xf32>
        %152 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %184 = vector.broadcast %151 : f32 to vector<1x4xf32>
          affine.yield %184 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg13 [%138, %139] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%138, %142] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%138, %145] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%138, %148] : f32 into vector<2x8xf32>
        %161 = vector.extract %arg9[%138, %139] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg9[%138, %142] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg9[%138, %145] : f32 from vector<2x8xf32>
        %164 = vector.extract %arg9[%138, %148] : f32 from vector<2x8xf32>
        %165 = vector.from_elements %161, %162, %163, %164 : vector<1x4xf32>
        %166 = arith.mulf %165, %152 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %167 = vector.extract %166[0, 0] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %arg14 [%138, %139] : f32 into vector<2x8xf32>
        %169 = vector.extract %166[0, 1] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%138, %142] : f32 into vector<2x8xf32>
        %171 = vector.extract %166[0, 2] : f32 from vector<1x4xf32>
        %172 = vector.insert %171, %170 [%138, %145] : f32 into vector<2x8xf32>
        %173 = vector.extract %166[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %172 [%138, %148] : f32 into vector<2x8xf32>
        %175 = arith.addf %166, %137 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %arg15 [%138, %139] : f32 into vector<2x8xf32>
        %178 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%138, %142] : f32 into vector<2x8xf32>
        %180 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%138, %145] : f32 into vector<2x8xf32>
        %182 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %181 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150, %160, %174, %183 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %131 : !frisk.async_task
      %133 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %137 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %138 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %139 = frisk.read_matrix_from_shm %alloc_29[%137, %138] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %140 = vector.bitcast %139 : vector<4xi32> to vector<8xf16>
        %141 = vector.extract %140[0] : f16 from vector<8xf16>
        %142 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %143 = vector.extract %140[%142] : f16 from vector<8xf16>
        %144 = vector.extract %140[1] : f16 from vector<8xf16>
        %145 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %146 = vector.extract %140[%145] : f16 from vector<8xf16>
        %147 = vector.extract %140[2] : f16 from vector<8xf16>
        %148 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %149 = vector.extract %140[%148] : f16 from vector<8xf16>
        %150 = vector.extract %140[3] : f16 from vector<8xf16>
        %151 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %152 = vector.extract %140[%151] : f16 from vector<8xf16>
        %153 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %154 = vector.insert %141, %arg12 [%153, 0] : f16 into vector<32x2xf16>
        %155 = vector.insert %143, %154 [%153, 1] : f16 into vector<32x2xf16>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %157 = vector.insert %144, %155 [%156, 0] : f16 into vector<32x2xf16>
        %158 = vector.insert %146, %157 [%156, 1] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %160 = vector.insert %147, %158 [%159, 0] : f16 into vector<32x2xf16>
        %161 = vector.insert %149, %160 [%159, 1] : f16 into vector<32x2xf16>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %163 = vector.insert %150, %161 [%162, 0] : f16 into vector<32x2xf16>
        %164 = vector.insert %152, %163 [%162, 1] : f16 into vector<32x2xf16>
        affine.yield %164 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %134:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %137 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %192 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %193 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %194 = vector.extract %123[%192, %193] : f16 from vector<2x32xf16>
          %195 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %196 = vector.extract %123[%192, %195] : f16 from vector<2x32xf16>
          %197 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %198 = vector.extract %123[%192, %197] : f16 from vector<2x32xf16>
          %199 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %200 = vector.extract %123[%192, %199] : f16 from vector<2x32xf16>
          %201 = vector.from_elements %194, %196, %198, %200 : vector<1x4xf16>
          %202 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %203 = vector.extract %133[%193, %202] : f16 from vector<32x2xf16>
          %204 = vector.extract %133[%195, %202] : f16 from vector<32x2xf16>
          %205 = vector.extract %133[%197, %202] : f16 from vector<32x2xf16>
          %206 = vector.extract %133[%199, %202] : f16 from vector<32x2xf16>
          %207 = vector.from_elements %203, %204, %205, %206 : vector<4x1xf16>
          %208 = frisk.warp_mma_rr %201, %207, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %208 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %138 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        %151 = vector.extract %119#1[%138, 0] : f32 from vector<2x1xf32>
        %152 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %192 = vector.broadcast %151 : f32 to vector<1x4xf32>
          affine.yield %192 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %arg13 [%138, %139] : f32 into vector<2x8xf32>
        %155 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%138, %142] : f32 into vector<2x8xf32>
        %157 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%138, %145] : f32 into vector<2x8xf32>
        %159 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%138, %148] : f32 into vector<2x8xf32>
        %161 = vector.extract %arg10[%138, %139] : f32 from vector<2x8xf32>
        %162 = vector.extract %arg10[%138, %142] : f32 from vector<2x8xf32>
        %163 = vector.extract %arg10[%138, %145] : f32 from vector<2x8xf32>
        %164 = vector.extract %arg10[%138, %148] : f32 from vector<2x8xf32>
        %165 = vector.from_elements %161, %162, %163, %164 : vector<1x4xf32>
        %166 = arith.mulf %165, %152 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %167 = vector.extract %166[0, 0] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %arg14 [%138, %139] : f32 into vector<2x8xf32>
        %169 = vector.extract %166[0, 1] : f32 from vector<1x4xf32>
        %170 = vector.insert %169, %168 [%138, %142] : f32 into vector<2x8xf32>
        %171 = vector.extract %166[0, 2] : f32 from vector<1x4xf32>
        %172 = vector.insert %171, %170 [%138, %145] : f32 into vector<2x8xf32>
        %173 = vector.extract %166[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %172 [%138, %148] : f32 into vector<2x8xf32>
        %175 = arith.addf %166, %137 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %arg15 [%138, %139] : f32 into vector<2x8xf32>
        %178 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%138, %142] : f32 into vector<2x8xf32>
        %180 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%138, %145] : f32 into vector<2x8xf32>
        %182 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
        %183 = vector.insert %182, %181 [%138, %148] : f32 into vector<2x8xf32>
        %184 = vector.extract %126#3[%138, %139] : f32 from vector<2x8xf32>
        %185 = vector.extract %126#3[%138, %142] : f32 from vector<2x8xf32>
        %186 = vector.extract %126#3[%138, %145] : f32 from vector<2x8xf32>
        %187 = vector.extract %126#3[%138, %148] : f32 from vector<2x8xf32>
        %188 = vector.insert %184, %arg16 [%138, %139] : f32 into vector<2x8xf32>
        %189 = vector.insert %185, %188 [%138, %142] : f32 into vector<2x8xf32>
        %190 = vector.insert %186, %189 [%138, %145] : f32 into vector<2x8xf32>
        %191 = vector.insert %187, %190 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150, %160, %174, %183, %191 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %135:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %137 = vector.extract %122[%arg11, 0] : f32 from vector<2x1xf32>
        %138 = vector.insert %137, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %139 = vector.extract %117[%arg11, 0] : f32 from vector<2x1xf32>
        %140 = vector.insert %139, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %138, %140 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %136:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %137 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %138 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %139 = vector.extract %129#3[%137, %138] : f32 from vector<2x8xf32>
        %140 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %141 = vector.extract %129#3[%137, %140] : f32 from vector<2x8xf32>
        %142 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %143 = vector.extract %129#3[%137, %142] : f32 from vector<2x8xf32>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %145 = vector.extract %129#3[%137, %144] : f32 from vector<2x8xf32>
        %146 = vector.insert %139, %arg12 [%137, %138] : f32 into vector<2x8xf32>
        %147 = vector.insert %141, %146 [%137, %140] : f32 into vector<2x8xf32>
        %148 = vector.insert %143, %147 [%137, %142] : f32 into vector<2x8xf32>
        %149 = vector.insert %145, %148 [%137, %144] : f32 into vector<2x8xf32>
        %150 = vector.extract %132#3[%137, %138] : f32 from vector<2x8xf32>
        %151 = vector.extract %132#3[%137, %140] : f32 from vector<2x8xf32>
        %152 = vector.extract %132#3[%137, %142] : f32 from vector<2x8xf32>
        %153 = vector.extract %132#3[%137, %144] : f32 from vector<2x8xf32>
        %154 = vector.insert %150, %arg13 [%137, %138] : f32 into vector<2x8xf32>
        %155 = vector.insert %151, %154 [%137, %140] : f32 into vector<2x8xf32>
        %156 = vector.insert %152, %155 [%137, %142] : f32 into vector<2x8xf32>
        %157 = vector.insert %153, %156 [%137, %144] : f32 into vector<2x8xf32>
        %158 = vector.extract %134#3[%137, %138] : f32 from vector<2x8xf32>
        %159 = vector.extract %134#3[%137, %140] : f32 from vector<2x8xf32>
        %160 = vector.extract %134#3[%137, %142] : f32 from vector<2x8xf32>
        %161 = vector.extract %134#3[%137, %144] : f32 from vector<2x8xf32>
        %162 = vector.insert %158, %arg14 [%137, %138] : f32 into vector<2x8xf32>
        %163 = vector.insert %159, %162 [%137, %140] : f32 into vector<2x8xf32>
        %164 = vector.insert %160, %163 [%137, %142] : f32 into vector<2x8xf32>
        %165 = vector.insert %161, %164 [%137, %144] : f32 into vector<2x8xf32>
        affine.yield %149, %157, %165 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %134#4, %135#0, %135#1, %136#0, %136#1, %136#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %86 = arith.index_cast %thread_id_x : index to i32
    %87 = arith.andi %86, %c16_i32 : i32
    %88 = arith.cmpi ne, %87, %c0_i32 : i32
    %89 = arith.andi %86, %c32_i32 : i32
    %90 = arith.cmpi ne, %89, %c0_i32 : i32
    %91 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %103 = vector.extract %85#5[%102, %101] : f32 from vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %105 = vector.extract %85#5[%102, %104] : f32 from vector<2x8xf32>
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %107 = vector.extract %85#5[%102, %106] : f32 from vector<2x8xf32>
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %109 = vector.extract %85#5[%102, %108] : f32 from vector<2x8xf32>
      %110 = arith.select %88, %103, %105 : f32
      %shuffleResult, %valid = gpu.shuffle xor %110, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %88, %shuffleResult, %103 : f32
      %112 = arith.select %88, %105, %shuffleResult : f32
      %113 = arith.select %88, %107, %109 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %113, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %88, %shuffleResult_15, %107 : f32
      %115 = arith.select %88, %109, %shuffleResult_15 : f32
      %116 = arith.select %90, %111, %114 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %116, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %90, %shuffleResult_17, %111 : f32
      %118 = arith.select %90, %114, %shuffleResult_17 : f32
      %119 = arith.select %90, %112, %115 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %119, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %90, %shuffleResult_19, %112 : f32
      %121 = arith.select %90, %115, %shuffleResult_19 : f32
      %122 = vector.insert %117, %arg5 [%102, %101] : f32 into vector<2x8xf32>
      %123 = vector.insert %120, %122 [%102, %104] : f32 into vector<2x8xf32>
      %124 = vector.insert %118, %123 [%102, %106] : f32 into vector<2x8xf32>
      %125 = vector.insert %121, %124 [%102, %108] : f32 into vector<2x8xf32>
      affine.yield %125 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %92 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %103 = vector.extract %85#4[%102, %101] : f32 from vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %105 = vector.extract %85#4[%102, %104] : f32 from vector<2x8xf32>
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %107 = vector.extract %85#4[%102, %106] : f32 from vector<2x8xf32>
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %109 = vector.extract %85#4[%102, %108] : f32 from vector<2x8xf32>
      %110 = arith.select %88, %103, %105 : f32
      %shuffleResult, %valid = gpu.shuffle xor %110, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %88, %shuffleResult, %103 : f32
      %112 = arith.select %88, %105, %shuffleResult : f32
      %113 = arith.select %88, %107, %109 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %113, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %88, %shuffleResult_15, %107 : f32
      %115 = arith.select %88, %109, %shuffleResult_15 : f32
      %116 = arith.select %90, %111, %114 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %116, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %90, %shuffleResult_17, %111 : f32
      %118 = arith.select %90, %114, %shuffleResult_17 : f32
      %119 = arith.select %90, %112, %115 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %119, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %90, %shuffleResult_19, %112 : f32
      %121 = arith.select %90, %115, %shuffleResult_19 : f32
      %122 = vector.insert %117, %arg5 [%102, %101] : f32 into vector<2x8xf32>
      %123 = vector.insert %120, %122 [%102, %104] : f32 into vector<2x8xf32>
      %124 = vector.insert %118, %123 [%102, %106] : f32 into vector<2x8xf32>
      %125 = vector.insert %121, %124 [%102, %108] : f32 into vector<2x8xf32>
      affine.yield %125 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %93 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %103 = vector.extract %85#3[%102, %101] : f32 from vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %105 = vector.extract %85#3[%102, %104] : f32 from vector<2x8xf32>
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %107 = vector.extract %85#3[%102, %106] : f32 from vector<2x8xf32>
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %109 = vector.extract %85#3[%102, %108] : f32 from vector<2x8xf32>
      %110 = arith.select %88, %103, %105 : f32
      %shuffleResult, %valid = gpu.shuffle xor %110, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %88, %shuffleResult, %103 : f32
      %112 = arith.select %88, %105, %shuffleResult : f32
      %113 = arith.select %88, %107, %109 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %113, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %88, %shuffleResult_15, %107 : f32
      %115 = arith.select %88, %109, %shuffleResult_15 : f32
      %116 = arith.select %90, %111, %114 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %116, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %90, %shuffleResult_17, %111 : f32
      %118 = arith.select %90, %114, %shuffleResult_17 : f32
      %119 = arith.select %90, %112, %115 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %119, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %90, %shuffleResult_19, %112 : f32
      %121 = arith.select %90, %115, %shuffleResult_19 : f32
      %122 = vector.insert %117, %arg5 [%102, %101] : f32 into vector<2x8xf32>
      %123 = vector.insert %120, %122 [%102, %104] : f32 into vector<2x8xf32>
      %124 = vector.insert %118, %123 [%102, %106] : f32 into vector<2x8xf32>
      %125 = vector.insert %121, %124 [%102, %108] : f32 into vector<2x8xf32>
      affine.yield %125 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %94 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %103 = vector.extract %85#0[%102, %101] : f32 from vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %105 = vector.extract %85#0[%102, %104] : f32 from vector<2x8xf32>
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %107 = vector.extract %85#0[%102, %106] : f32 from vector<2x8xf32>
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %109 = vector.extract %85#0[%102, %108] : f32 from vector<2x8xf32>
      %110 = arith.select %88, %103, %105 : f32
      %shuffleResult, %valid = gpu.shuffle xor %110, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %88, %shuffleResult, %103 : f32
      %112 = arith.select %88, %105, %shuffleResult : f32
      %113 = arith.select %88, %107, %109 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %113, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %88, %shuffleResult_15, %107 : f32
      %115 = arith.select %88, %109, %shuffleResult_15 : f32
      %116 = arith.select %90, %111, %114 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %116, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %117 = arith.select %90, %shuffleResult_17, %111 : f32
      %118 = arith.select %90, %114, %shuffleResult_17 : f32
      %119 = arith.select %90, %112, %115 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %119, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %120 = arith.select %90, %shuffleResult_19, %112 : f32
      %121 = arith.select %90, %115, %shuffleResult_19 : f32
      %122 = vector.insert %117, %arg5 [%102, %101] : f32 into vector<2x8xf32>
      %123 = vector.insert %120, %122 [%102, %104] : f32 into vector<2x8xf32>
      %124 = vector.insert %118, %123 [%102, %106] : f32 into vector<2x8xf32>
      %125 = vector.insert %121, %124 [%102, %108] : f32 into vector<2x8xf32>
      affine.yield %125 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %95:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %101 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %102 = vector.extract %85#1[%101, 0] : f32 from vector<2x1xf32>
      %103 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %143 = vector.broadcast %102 : f32 to vector<1x4xf32>
        affine.yield %143 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %105 = vector.extract %103[0, 0] : f32 from vector<1x4xf32>
      %106 = vector.insert %105, %arg5 [%101, %104] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.extract %103[0, 1] : f32 from vector<1x4xf32>
      %109 = vector.insert %108, %106 [%101, %107] : f32 into vector<2x8xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %111 = vector.extract %103[0, 2] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %109 [%101, %110] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %103[0, 3] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %112 [%101, %113] : f32 into vector<2x8xf32>
      %116 = vector.extract %94[%101, %104] : f32 from vector<2x8xf32>
      %117 = vector.extract %94[%101, %107] : f32 from vector<2x8xf32>
      %118 = vector.extract %94[%101, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %94[%101, %113] : f32 from vector<2x8xf32>
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
    %96 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %97 = arith.cmpi ult, %96, %c4 : index
    scf.if %97 {
      affine.for %arg4 = 0 to 4 {
        %101 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %103 = vector.extract %95#3[%101, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %105 = vector.extract %95#3[%101, %104] : f16 from vector<2x8xf16>
        %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %107 = vector.extract %95#3[%101, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %109 = vector.extract %95#3[%101, %108] : f16 from vector<2x8xf16>
        affine.store %103, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %105, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %107, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %109, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %98:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %101 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %102 = vector.extract %85#1[%101, 0] : f32 from vector<2x1xf32>
      %103 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %143 = vector.broadcast %102 : f32 to vector<1x4xf32>
        affine.yield %143 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %105 = vector.extract %103[0, 0] : f32 from vector<1x4xf32>
      %106 = vector.insert %105, %arg5 [%101, %104] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.extract %103[0, 1] : f32 from vector<1x4xf32>
      %109 = vector.insert %108, %106 [%101, %107] : f32 into vector<2x8xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %111 = vector.extract %103[0, 2] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %109 [%101, %110] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %103[0, 3] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %112 [%101, %113] : f32 into vector<2x8xf32>
      %116 = vector.extract %93[%101, %104] : f32 from vector<2x8xf32>
      %117 = vector.extract %93[%101, %107] : f32 from vector<2x8xf32>
      %118 = vector.extract %93[%101, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %93[%101, %113] : f32 from vector<2x8xf32>
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
    scf.if %97 {
      affine.for %arg4 = 0 to 4 {
        %101 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %103 = vector.extract %98#3[%101, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %105 = vector.extract %98#3[%101, %104] : f16 from vector<2x8xf16>
        %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %107 = vector.extract %98#3[%101, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %109 = vector.extract %98#3[%101, %108] : f16 from vector<2x8xf16>
        affine.store %103, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %105, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %107, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %109, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %99:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %101 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %102 = vector.extract %85#1[%101, 0] : f32 from vector<2x1xf32>
      %103 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %143 = vector.broadcast %102 : f32 to vector<1x4xf32>
        affine.yield %143 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %105 = vector.extract %103[0, 0] : f32 from vector<1x4xf32>
      %106 = vector.insert %105, %arg5 [%101, %104] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.extract %103[0, 1] : f32 from vector<1x4xf32>
      %109 = vector.insert %108, %106 [%101, %107] : f32 into vector<2x8xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %111 = vector.extract %103[0, 2] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %109 [%101, %110] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %103[0, 3] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %112 [%101, %113] : f32 into vector<2x8xf32>
      %116 = vector.extract %92[%101, %104] : f32 from vector<2x8xf32>
      %117 = vector.extract %92[%101, %107] : f32 from vector<2x8xf32>
      %118 = vector.extract %92[%101, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %92[%101, %113] : f32 from vector<2x8xf32>
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
    scf.if %97 {
      affine.for %arg4 = 0 to 4 {
        %101 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %103 = vector.extract %99#3[%101, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %105 = vector.extract %99#3[%101, %104] : f16 from vector<2x8xf16>
        %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %107 = vector.extract %99#3[%101, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %109 = vector.extract %99#3[%101, %108] : f16 from vector<2x8xf16>
        affine.store %103, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %105, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %107, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %109, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %100:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %101 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %102 = vector.extract %85#1[%101, 0] : f32 from vector<2x1xf32>
      %103 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %143 = vector.broadcast %102 : f32 to vector<1x4xf32>
        affine.yield %143 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %105 = vector.extract %103[0, 0] : f32 from vector<1x4xf32>
      %106 = vector.insert %105, %arg5 [%101, %104] : f32 into vector<2x8xf32>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.extract %103[0, 1] : f32 from vector<1x4xf32>
      %109 = vector.insert %108, %106 [%101, %107] : f32 into vector<2x8xf32>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %111 = vector.extract %103[0, 2] : f32 from vector<1x4xf32>
      %112 = vector.insert %111, %109 [%101, %110] : f32 into vector<2x8xf32>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %103[0, 3] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %112 [%101, %113] : f32 into vector<2x8xf32>
      %116 = vector.extract %91[%101, %104] : f32 from vector<2x8xf32>
      %117 = vector.extract %91[%101, %107] : f32 from vector<2x8xf32>
      %118 = vector.extract %91[%101, %110] : f32 from vector<2x8xf32>
      %119 = vector.extract %91[%101, %113] : f32 from vector<2x8xf32>
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
    scf.if %97 {
      affine.for %arg4 = 0 to 4 {
        %101 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %103 = vector.extract %100#3[%101, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %105 = vector.extract %100#3[%101, %104] : f16 from vector<2x8xf16>
        %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %107 = vector.extract %100#3[%101, %106] : f16 from vector<2x8xf16>
        %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %109 = vector.extract %100#3[%101, %108] : f16 from vector<2x8xf16>
        affine.store %103, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %105, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %107, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %109, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
