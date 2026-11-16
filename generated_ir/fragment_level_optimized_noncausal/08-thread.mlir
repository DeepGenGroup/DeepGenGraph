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
      %93 = vector.insert %cst_2, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %94 = vector.insert %cst_7, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %95 = vector.insert %cst_7, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %93, %94, %95 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (vector<2x8xf32>) {
      %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %94 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %95 = vector.insert %cst_2, %arg5 [%94, %93] : f32 into vector<2x8xf32>
      %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %97 = vector.insert %cst_2, %95 [%94, %96] : f32 into vector<2x8xf32>
      %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %99 = vector.insert %cst_2, %97 [%94, %98] : f32 into vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %101 = vector.insert %cst_2, %99 [%94, %100] : f32 into vector<2x8xf32>
      affine.yield %101 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6) -> (vector<2x1xf32>) {
      %93 = vector.insert %cst_2, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %93 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %12 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 + s2 * 32 + s0 floordiv 32) mod 32)>()[%6, %7, %block_id_z]
    %13 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%6]
    %subview = memref.subview %arg0[0, %12, %13, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
    %thread_id_x = gpu.thread_id x
    %14 = affine.apply affine_map<()[s0] -> (s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>()[%thread_id_x]
    %15 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>()[%thread_id_x]
    %16 = memref.load %alloc[%14, %15] : memref<128x32xf16, 3>
    %17 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 1)>()[%thread_id_x]
    %18 = memref.load %alloc[%14, %17] : memref<128x32xf16, 3>
    %19 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 2)>()[%thread_id_x]
    %20 = memref.load %alloc[%14, %19] : memref<128x32xf16, 3>
    %21 = affine.apply affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 3)>()[%thread_id_x]
    %22 = memref.load %alloc[%14, %21] : memref<128x32xf16, 3>
    %23 = vector.from_elements %16, %18, %20, %22 : vector<1x4xf16>
    %24:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %23) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %93 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %94 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %95 = memref.load %alloc[%93, %94] : memref<128x32xf16, 3>
      %96 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %97 = memref.load %alloc[%93, %96] : memref<128x32xf16, 3>
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %99 = memref.load %alloc[%93, %98] : memref<128x32xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %101 = memref.load %alloc[%93, %100] : memref<128x32xf16, 3>
      %102 = vector.from_elements %95, %97, %99, %101 : vector<1x4xf16>
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %105 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %106 = vector.insert %105, %arg5 [%103, %104] : f16 into vector<2x8xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %109 = vector.insert %108, %106 [%103, %107] : f16 into vector<2x8xf16>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %111 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %112 = vector.insert %111, %109 [%103, %110] : f16 into vector<2x8xf16>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %112 [%103, %113] : f16 into vector<2x8xf16>
      %116 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %117 = arith.mulf %arg7, %116 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %118 = vector.extract %117[0, 0] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %arg6 [%103, %104] : f16 into vector<2x8xf16>
      %120 = vector.extract %117[0, 1] : f16 from vector<1x4xf16>
      %121 = vector.insert %120, %119 [%103, %107] : f16 into vector<2x8xf16>
      %122 = vector.extract %117[0, 2] : f16 from vector<1x4xf16>
      %123 = vector.insert %122, %121 [%103, %110] : f16 into vector<2x8xf16>
      %124 = vector.extract %117[0, 3] : f16 from vector<1x4xf16>
      %125 = vector.insert %124, %123 [%103, %113] : f16 into vector<2x8xf16>
      affine.yield %115, %125, %102 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
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
    %35 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%6, %7]
    %subview_10 = memref.subview %arg0[0, %35, %13, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview_10 to %alloc_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_9 : memref<128x32xf16, 3>
    %36 = memref.load %alloc_9[%14, %15] : memref<128x32xf16, 3>
    %37 = memref.load %alloc_9[%14, %17] : memref<128x32xf16, 3>
    %38 = memref.load %alloc_9[%14, %19] : memref<128x32xf16, 3>
    %39 = memref.load %alloc_9[%14, %21] : memref<128x32xf16, 3>
    %40 = vector.from_elements %36, %37, %38, %39 : vector<1x4xf16>
    %41:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %40) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %93 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %94 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %95 = memref.load %alloc_9[%93, %94] : memref<128x32xf16, 3>
      %96 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %97 = memref.load %alloc_9[%93, %96] : memref<128x32xf16, 3>
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %99 = memref.load %alloc_9[%93, %98] : memref<128x32xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %101 = memref.load %alloc_9[%93, %100] : memref<128x32xf16, 3>
      %102 = vector.from_elements %95, %97, %99, %101 : vector<1x4xf16>
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %105 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %106 = vector.insert %105, %arg5 [%103, %104] : f16 into vector<2x8xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %109 = vector.insert %108, %106 [%103, %107] : f16 into vector<2x8xf16>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %111 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %112 = vector.insert %111, %109 [%103, %110] : f16 into vector<2x8xf16>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %112 [%103, %113] : f16 into vector<2x8xf16>
      %116 = arith.mulf %arg7, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %117 = vector.extract %116[0, 0] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %arg6 [%103, %104] : f16 into vector<2x8xf16>
      %119 = vector.extract %116[0, 1] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %118 [%103, %107] : f16 into vector<2x8xf16>
      %121 = vector.extract %116[0, 2] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %120 [%103, %110] : f16 into vector<2x8xf16>
      %123 = vector.extract %116[0, 3] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%103, %113] : f16 into vector<2x8xf16>
      affine.yield %115, %124, %102 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %42 = arith.mulf %41#2, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %43 = vector.extract %42[0, 0] : f16 from vector<1x4xf16>
    %44 = vector.insert %43, %41#1 [1, 4] : f16 into vector<2x8xf16>
    %45 = vector.extract %42[0, 1] : f16 from vector<1x4xf16>
    %46 = vector.insert %45, %44 [1, 5] : f16 into vector<2x8xf16>
    %47 = vector.extract %42[0, 2] : f16 from vector<1x4xf16>
    %48 = vector.insert %47, %46 [1, 6] : f16 into vector<2x8xf16>
    %49 = vector.extract %42[0, 3] : f16 from vector<1x4xf16>
    %50 = vector.insert %49, %48 [1, 7] : f16 into vector<2x8xf16>
    %alloc_11 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_12 = memref.subview %arg0[0, %35, %13, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview_12 to %alloc_11 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_11 : memref<128x32xf16, 3>
    %51 = memref.load %alloc_11[%14, %15] : memref<128x32xf16, 3>
    %52 = memref.load %alloc_11[%14, %17] : memref<128x32xf16, 3>
    %53 = memref.load %alloc_11[%14, %19] : memref<128x32xf16, 3>
    %54 = memref.load %alloc_11[%14, %21] : memref<128x32xf16, 3>
    %55 = vector.from_elements %51, %52, %53, %54 : vector<1x4xf16>
    %56:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %55) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %93 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %94 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %95 = memref.load %alloc_11[%93, %94] : memref<128x32xf16, 3>
      %96 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %97 = memref.load %alloc_11[%93, %96] : memref<128x32xf16, 3>
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %99 = memref.load %alloc_11[%93, %98] : memref<128x32xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %101 = memref.load %alloc_11[%93, %100] : memref<128x32xf16, 3>
      %102 = vector.from_elements %95, %97, %99, %101 : vector<1x4xf16>
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %105 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %106 = vector.insert %105, %arg5 [%103, %104] : f16 into vector<2x8xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %109 = vector.insert %108, %106 [%103, %107] : f16 into vector<2x8xf16>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %111 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %112 = vector.insert %111, %109 [%103, %110] : f16 into vector<2x8xf16>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %112 [%103, %113] : f16 into vector<2x8xf16>
      %116 = arith.mulf %arg7, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %117 = vector.extract %116[0, 0] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %arg6 [%103, %104] : f16 into vector<2x8xf16>
      %119 = vector.extract %116[0, 1] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %118 [%103, %107] : f16 into vector<2x8xf16>
      %121 = vector.extract %116[0, 2] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %120 [%103, %110] : f16 into vector<2x8xf16>
      %123 = vector.extract %116[0, 3] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%103, %113] : f16 into vector<2x8xf16>
      affine.yield %115, %124, %102 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %57 = arith.mulf %56#2, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %58 = vector.extract %57[0, 0] : f16 from vector<1x4xf16>
    %59 = vector.insert %58, %56#1 [1, 4] : f16 into vector<2x8xf16>
    %60 = vector.extract %57[0, 1] : f16 from vector<1x4xf16>
    %61 = vector.insert %60, %59 [1, 5] : f16 into vector<2x8xf16>
    %62 = vector.extract %57[0, 2] : f16 from vector<1x4xf16>
    %63 = vector.insert %62, %61 [1, 6] : f16 into vector<2x8xf16>
    %64 = vector.extract %57[0, 3] : f16 from vector<1x4xf16>
    %65 = vector.insert %64, %63 [1, 7] : f16 into vector<2x8xf16>
    %alloc_13 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %subview_14 = memref.subview %arg0[0, %35, %13, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview_14 to %alloc_13 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_13 : memref<128x32xf16, 3>
    %66 = memref.load %alloc_13[%14, %15] : memref<128x32xf16, 3>
    %67 = memref.load %alloc_13[%14, %17] : memref<128x32xf16, 3>
    %68 = memref.load %alloc_13[%14, %19] : memref<128x32xf16, 3>
    %69 = memref.load %alloc_13[%14, %21] : memref<128x32xf16, 3>
    %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf16>
    %71:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %70) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %93 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %94 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %95 = memref.load %alloc_13[%93, %94] : memref<128x32xf16, 3>
      %96 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %97 = memref.load %alloc_13[%93, %96] : memref<128x32xf16, 3>
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %99 = memref.load %alloc_13[%93, %98] : memref<128x32xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %101 = memref.load %alloc_13[%93, %100] : memref<128x32xf16, 3>
      %102 = vector.from_elements %95, %97, %99, %101 : vector<1x4xf16>
      %103 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %105 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %106 = vector.insert %105, %arg5 [%103, %104] : f16 into vector<2x8xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %108 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %109 = vector.insert %108, %106 [%103, %107] : f16 into vector<2x8xf16>
      %110 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %111 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %112 = vector.insert %111, %109 [%103, %110] : f16 into vector<2x8xf16>
      %113 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %114 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %115 = vector.insert %114, %112 [%103, %113] : f16 into vector<2x8xf16>
      %116 = arith.mulf %arg7, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %117 = vector.extract %116[0, 0] : f16 from vector<1x4xf16>
      %118 = vector.insert %117, %arg6 [%103, %104] : f16 into vector<2x8xf16>
      %119 = vector.extract %116[0, 1] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %118 [%103, %107] : f16 into vector<2x8xf16>
      %121 = vector.extract %116[0, 2] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %120 [%103, %110] : f16 into vector<2x8xf16>
      %123 = vector.extract %116[0, 3] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%103, %113] : f16 into vector<2x8xf16>
      affine.yield %115, %124, %102 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %72 = arith.mulf %71#2, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %73 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
    %74 = vector.insert %73, %71#1 [1, 4] : f16 into vector<2x8xf16>
    %75 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
    %76 = vector.insert %75, %74 [1, 5] : f16 into vector<2x8xf16>
    %77 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
    %78 = vector.insert %77, %76 [1, 6] : f16 into vector<2x8xf16>
    %79 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
    %80 = vector.insert %79, %78 [1, 7] : f16 into vector<2x8xf16>
    %81:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#2, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %93 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %121 = vector.insert %cst_2, %arg12 [%120, %119] : f32 into vector<2x32xf32>
        %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %123 = vector.insert %cst_2, %121 [%120, %122] : f32 into vector<2x32xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %125 = vector.insert %cst_2, %123 [%120, %124] : f32 into vector<2x32xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %127 = vector.insert %cst_2, %125 [%120, %126] : f32 into vector<2x32xf32>
        affine.yield %127 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %94 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
      %subview_16 = memref.subview %transpose[0, %94, 0, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_16 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<32x128xf16, 3>
      %95:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %120 = memref.load %alloc_15[%15, %119] : memref<32x128xf16, 3>
        %121 = memref.load %alloc_15[%17, %119] : memref<32x128xf16, 3>
        %122 = memref.load %alloc_15[%19, %119] : memref<32x128xf16, 3>
        %123 = memref.load %alloc_15[%21, %119] : memref<32x128xf16, 3>
        %124 = vector.from_elements %120, %121, %122, %123 : vector<4x1xf16>
        %125:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %124) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %155 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %156 = memref.load %alloc_15[%155, %119] : memref<32x128xf16, 3>
          %157 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %158 = memref.load %alloc_15[%157, %119] : memref<32x128xf16, 3>
          %159 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %160 = memref.load %alloc_15[%159, %119] : memref<32x128xf16, 3>
          %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %162 = memref.load %alloc_15[%161, %119] : memref<32x128xf16, 3>
          %163 = vector.from_elements %156, %158, %160, %162 : vector<4x1xf16>
          %164 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %165 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %166 = vector.extract %34[%164, %165] : f16 from vector<2x8xf16>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %168 = vector.extract %34[%164, %167] : f16 from vector<2x8xf16>
          %169 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %170 = vector.extract %34[%164, %169] : f16 from vector<2x8xf16>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %172 = vector.extract %34[%164, %171] : f16 from vector<2x8xf16>
          %173 = vector.from_elements %166, %168, %170, %172 : vector<1x4xf16>
          %174 = frisk.warp_mma_rr %173, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %174, %163 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %126 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %127 = vector.extract %34[%126, 4] : f16 from vector<2x8xf16>
        %128 = vector.extract %34[%126, 5] : f16 from vector<2x8xf16>
        %129 = vector.extract %34[%126, 6] : f16 from vector<2x8xf16>
        %130 = vector.extract %34[%126, 7] : f16 from vector<2x8xf16>
        %131 = vector.from_elements %127, %128, %129, %130 : vector<1x4xf16>
        %132 = frisk.warp_mma_rr %131, %125#1, %125#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %133 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %134 = vector.extract %93[%126, %133] : f32 from vector<2x32xf32>
        %135 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %136 = vector.extract %93[%126, %135] : f32 from vector<2x32xf32>
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %138 = vector.extract %93[%126, %137] : f32 from vector<2x32xf32>
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %140 = vector.extract %93[%126, %139] : f32 from vector<2x32xf32>
        %141 = vector.from_elements %134, %136, %138, %140 : vector<1x4xf32>
        %142 = arith.addf %141, %132 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %143 = vector.extract %142[0, 0] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %arg12 [%126, %133] : f32 into vector<2x32xf32>
        %145 = vector.extract %142[0, 1] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %144 [%126, %135] : f32 into vector<2x32xf32>
        %147 = vector.extract %142[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%126, %137] : f32 into vector<2x32xf32>
        %149 = vector.extract %142[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%126, %139] : f32 into vector<2x32xf32>
        %151 = vector.insert %143, %arg13 [%126, %133] : f32 into vector<2x32xf32>
        %152 = vector.insert %145, %151 [%126, %135] : f32 into vector<2x32xf32>
        %153 = vector.insert %147, %152 [%126, %137] : f32 into vector<2x32xf32>
        %154 = vector.insert %149, %153 [%126, %139] : f32 into vector<2x32xf32>
        affine.yield %150, %154 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %96 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 524288 + s1 * 16777216 + 131072) floordiv 524288) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_18 = memref.subview %transpose[0, %96, 32, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_18 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_17 : memref<32x128xf16, 3>
      %97:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %120 = memref.load %alloc_17[%15, %119] : memref<32x128xf16, 3>
        %121 = memref.load %alloc_17[%17, %119] : memref<32x128xf16, 3>
        %122 = memref.load %alloc_17[%19, %119] : memref<32x128xf16, 3>
        %123 = memref.load %alloc_17[%21, %119] : memref<32x128xf16, 3>
        %124 = vector.from_elements %120, %121, %122, %123 : vector<4x1xf16>
        %125:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %124) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %155 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %156 = memref.load %alloc_17[%155, %119] : memref<32x128xf16, 3>
          %157 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %158 = memref.load %alloc_17[%157, %119] : memref<32x128xf16, 3>
          %159 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %160 = memref.load %alloc_17[%159, %119] : memref<32x128xf16, 3>
          %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %162 = memref.load %alloc_17[%161, %119] : memref<32x128xf16, 3>
          %163 = vector.from_elements %156, %158, %160, %162 : vector<4x1xf16>
          %164 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %165 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %166 = vector.extract %50[%164, %165] : f16 from vector<2x8xf16>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %168 = vector.extract %50[%164, %167] : f16 from vector<2x8xf16>
          %169 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %170 = vector.extract %50[%164, %169] : f16 from vector<2x8xf16>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %172 = vector.extract %50[%164, %171] : f16 from vector<2x8xf16>
          %173 = vector.from_elements %166, %168, %170, %172 : vector<1x4xf16>
          %174 = frisk.warp_mma_rr %173, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %174, %163 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %126 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %127 = vector.extract %50[%126, 4] : f16 from vector<2x8xf16>
        %128 = vector.extract %50[%126, 5] : f16 from vector<2x8xf16>
        %129 = vector.extract %50[%126, 6] : f16 from vector<2x8xf16>
        %130 = vector.extract %50[%126, 7] : f16 from vector<2x8xf16>
        %131 = vector.from_elements %127, %128, %129, %130 : vector<1x4xf16>
        %132 = frisk.warp_mma_rr %131, %125#1, %125#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %133 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %134 = vector.extract %95#1[%126, %133] : f32 from vector<2x32xf32>
        %135 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %136 = vector.extract %95#1[%126, %135] : f32 from vector<2x32xf32>
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %138 = vector.extract %95#1[%126, %137] : f32 from vector<2x32xf32>
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %140 = vector.extract %95#1[%126, %139] : f32 from vector<2x32xf32>
        %141 = vector.from_elements %134, %136, %138, %140 : vector<1x4xf32>
        %142 = arith.addf %141, %132 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %143 = vector.extract %142[0, 0] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %arg12 [%126, %133] : f32 into vector<2x32xf32>
        %145 = vector.extract %142[0, 1] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %144 [%126, %135] : f32 into vector<2x32xf32>
        %147 = vector.extract %142[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%126, %137] : f32 into vector<2x32xf32>
        %149 = vector.extract %142[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%126, %139] : f32 into vector<2x32xf32>
        %151 = vector.insert %143, %arg13 [%126, %133] : f32 into vector<2x32xf32>
        %152 = vector.insert %145, %151 [%126, %135] : f32 into vector<2x32xf32>
        %153 = vector.insert %147, %152 [%126, %137] : f32 into vector<2x32xf32>
        %154 = vector.insert %149, %153 [%126, %139] : f32 into vector<2x32xf32>
        affine.yield %150, %154 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %98 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 524288 + s1 * 16777216 + 262144) floordiv 524288) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_20 = memref.subview %transpose[0, %98, 64, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_20 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_19 : memref<32x128xf16, 3>
      %99:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %120 = memref.load %alloc_19[%15, %119] : memref<32x128xf16, 3>
        %121 = memref.load %alloc_19[%17, %119] : memref<32x128xf16, 3>
        %122 = memref.load %alloc_19[%19, %119] : memref<32x128xf16, 3>
        %123 = memref.load %alloc_19[%21, %119] : memref<32x128xf16, 3>
        %124 = vector.from_elements %120, %121, %122, %123 : vector<4x1xf16>
        %125:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %124) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %155 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %156 = memref.load %alloc_19[%155, %119] : memref<32x128xf16, 3>
          %157 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %158 = memref.load %alloc_19[%157, %119] : memref<32x128xf16, 3>
          %159 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %160 = memref.load %alloc_19[%159, %119] : memref<32x128xf16, 3>
          %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %162 = memref.load %alloc_19[%161, %119] : memref<32x128xf16, 3>
          %163 = vector.from_elements %156, %158, %160, %162 : vector<4x1xf16>
          %164 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %165 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %166 = vector.extract %65[%164, %165] : f16 from vector<2x8xf16>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %168 = vector.extract %65[%164, %167] : f16 from vector<2x8xf16>
          %169 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %170 = vector.extract %65[%164, %169] : f16 from vector<2x8xf16>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %172 = vector.extract %65[%164, %171] : f16 from vector<2x8xf16>
          %173 = vector.from_elements %166, %168, %170, %172 : vector<1x4xf16>
          %174 = frisk.warp_mma_rr %173, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %174, %163 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %126 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %127 = vector.extract %65[%126, 4] : f16 from vector<2x8xf16>
        %128 = vector.extract %65[%126, 5] : f16 from vector<2x8xf16>
        %129 = vector.extract %65[%126, 6] : f16 from vector<2x8xf16>
        %130 = vector.extract %65[%126, 7] : f16 from vector<2x8xf16>
        %131 = vector.from_elements %127, %128, %129, %130 : vector<1x4xf16>
        %132 = frisk.warp_mma_rr %131, %125#1, %125#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %133 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %134 = vector.extract %97#1[%126, %133] : f32 from vector<2x32xf32>
        %135 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %136 = vector.extract %97#1[%126, %135] : f32 from vector<2x32xf32>
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %138 = vector.extract %97#1[%126, %137] : f32 from vector<2x32xf32>
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %140 = vector.extract %97#1[%126, %139] : f32 from vector<2x32xf32>
        %141 = vector.from_elements %134, %136, %138, %140 : vector<1x4xf32>
        %142 = arith.addf %141, %132 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %143 = vector.extract %142[0, 0] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %arg12 [%126, %133] : f32 into vector<2x32xf32>
        %145 = vector.extract %142[0, 1] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %144 [%126, %135] : f32 into vector<2x32xf32>
        %147 = vector.extract %142[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%126, %137] : f32 into vector<2x32xf32>
        %149 = vector.extract %142[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%126, %139] : f32 into vector<2x32xf32>
        %151 = vector.insert %143, %arg13 [%126, %133] : f32 into vector<2x32xf32>
        %152 = vector.insert %145, %151 [%126, %135] : f32 into vector<2x32xf32>
        %153 = vector.insert %147, %152 [%126, %137] : f32 into vector<2x32xf32>
        %154 = vector.insert %149, %153 [%126, %139] : f32 into vector<2x32xf32>
        affine.yield %150, %154 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 524288 + s1 * 16777216 + 393216) floordiv 524288) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_22 = memref.subview %transpose[0, %100, 96, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_22 to %alloc_21 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_21 : memref<32x128xf16, 3>
      %101:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %120 = memref.load %alloc_21[%15, %119] : memref<32x128xf16, 3>
        %121 = memref.load %alloc_21[%17, %119] : memref<32x128xf16, 3>
        %122 = memref.load %alloc_21[%19, %119] : memref<32x128xf16, 3>
        %123 = memref.load %alloc_21[%21, %119] : memref<32x128xf16, 3>
        %124 = vector.from_elements %120, %121, %122, %123 : vector<4x1xf16>
        %125:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %124) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %155 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %156 = memref.load %alloc_21[%155, %119] : memref<32x128xf16, 3>
          %157 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %158 = memref.load %alloc_21[%157, %119] : memref<32x128xf16, 3>
          %159 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %160 = memref.load %alloc_21[%159, %119] : memref<32x128xf16, 3>
          %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %162 = memref.load %alloc_21[%161, %119] : memref<32x128xf16, 3>
          %163 = vector.from_elements %156, %158, %160, %162 : vector<4x1xf16>
          %164 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %165 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %166 = vector.extract %80[%164, %165] : f16 from vector<2x8xf16>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %168 = vector.extract %80[%164, %167] : f16 from vector<2x8xf16>
          %169 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %170 = vector.extract %80[%164, %169] : f16 from vector<2x8xf16>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %172 = vector.extract %80[%164, %171] : f16 from vector<2x8xf16>
          %173 = vector.from_elements %166, %168, %170, %172 : vector<1x4xf16>
          %174 = frisk.warp_mma_rr %173, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %174, %163 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %126 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %127 = vector.extract %80[%126, 4] : f16 from vector<2x8xf16>
        %128 = vector.extract %80[%126, 5] : f16 from vector<2x8xf16>
        %129 = vector.extract %80[%126, 6] : f16 from vector<2x8xf16>
        %130 = vector.extract %80[%126, 7] : f16 from vector<2x8xf16>
        %131 = vector.from_elements %127, %128, %129, %130 : vector<1x4xf16>
        %132 = frisk.warp_mma_rr %131, %125#1, %125#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %133 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %134 = vector.extract %99#1[%126, %133] : f32 from vector<2x32xf32>
        %135 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %136 = vector.extract %99#1[%126, %135] : f32 from vector<2x32xf32>
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %138 = vector.extract %99#1[%126, %137] : f32 from vector<2x32xf32>
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %140 = vector.extract %99#1[%126, %139] : f32 from vector<2x32xf32>
        %141 = vector.from_elements %134, %136, %138, %140 : vector<1x4xf32>
        %142 = arith.addf %141, %132 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %143 = vector.extract %142[0, 0] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %arg12 [%126, %133] : f32 into vector<2x32xf32>
        %145 = vector.extract %142[0, 1] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %144 [%126, %135] : f32 into vector<2x32xf32>
        %147 = vector.extract %142[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%126, %137] : f32 into vector<2x32xf32>
        %149 = vector.extract %142[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%126, %139] : f32 into vector<2x32xf32>
        %151 = vector.insert %143, %arg13 [%126, %133] : f32 into vector<2x32xf32>
        %152 = vector.insert %145, %151 [%126, %135] : f32 into vector<2x32xf32>
        %153 = vector.insert %147, %152 [%126, %137] : f32 into vector<2x32xf32>
        %154 = vector.insert %149, %153 [%126, %139] : f32 into vector<2x32xf32>
        affine.yield %150, %154 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %102 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %121 = vector.extract %101#1[%119, %120] : f32 from vector<2x32xf32>
        %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %123 = vector.extract %101#1[%119, %122] : f32 from vector<2x32xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %125 = vector.extract %101#1[%119, %124] : f32 from vector<2x32xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %127 = vector.extract %101#1[%119, %126] : f32 from vector<2x32xf32>
        %128 = vector.insert %121, %arg12 [%119, %120] : f32 into vector<2x32xf32>
        %129 = vector.insert %123, %128 [%119, %122] : f32 into vector<2x32xf32>
        %130 = vector.insert %125, %129 [%119, %124] : f32 into vector<2x32xf32>
        %131 = vector.insert %127, %130 [%119, %126] : f32 into vector<2x32xf32>
        affine.yield %131 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %103:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %119 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_7) -> (f32) {
          %129 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %130 = vector.extract %102[%arg11, %129] : f32 from vector<2x32xf32>
          %131 = arith.maxnumf %arg15, %130 : f32
          %132 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %133 = vector.extract %102[%arg11, %132] : f32 from vector<2x32xf32>
          %134 = arith.maxnumf %131, %133 : f32
          %135 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %136 = vector.extract %102[%arg11, %135] : f32 from vector<2x32xf32>
          %137 = arith.maxnumf %134, %136 : f32
          %138 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %139 = vector.extract %102[%arg11, %138] : f32 from vector<2x32xf32>
          %140 = arith.maxnumf %137, %139 : f32
          affine.yield %140 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %119, %c16_i32, %c64_i32 : f32
        %120 = arith.maxnumf %119, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %120, %c32_i32, %c64_i32 : f32
        %121 = arith.maxnumf %120, %shuffleResult_31 : f32
        %122 = vector.broadcast %121 : f32 to vector<1x1xf32>
        %123 = vector.insert %121, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %124 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %125 = vector.broadcast %124 : f32 to vector<1x1xf32>
        %126 = arith.maxnumf %122, %125 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %127 = vector.extract %126[0, 0] : f32 from vector<1x1xf32>
        %128 = vector.insert %127, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %123, %128 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %104:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = vector.extract %103#1[%119, 0] : f32 from vector<2x1xf32>
        %121 = vector.broadcast %120 : f32 to vector<1x4xf32>
        %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %123 = vector.insert %120, %arg12 [%119, %122] : f32 into vector<2x32xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %125 = vector.insert %120, %123 [%119, %124] : f32 into vector<2x32xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %127 = vector.insert %120, %125 [%119, %126] : f32 into vector<2x32xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %129 = vector.insert %120, %127 [%119, %128] : f32 into vector<2x32xf32>
        %130 = vector.extract %101#1[%119, %122] : f32 from vector<2x32xf32>
        %131 = vector.extract %101#1[%119, %124] : f32 from vector<2x32xf32>
        %132 = vector.extract %101#1[%119, %126] : f32 from vector<2x32xf32>
        %133 = vector.extract %101#1[%119, %128] : f32 from vector<2x32xf32>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf32>
        %135 = arith.subf %134, %121 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %136 = vector.extract %135[0, 0] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %arg13 [%119, %122] : f32 into vector<2x32xf32>
        %138 = vector.extract %135[0, 1] : f32 from vector<1x4xf32>
        %139 = vector.insert %138, %137 [%119, %124] : f32 into vector<2x32xf32>
        %140 = vector.extract %135[0, 2] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %139 [%119, %126] : f32 into vector<2x32xf32>
        %142 = vector.extract %135[0, 3] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %141 [%119, %128] : f32 into vector<2x32xf32>
        affine.yield %129, %143 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>}
      %105:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6, %arg14 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %119 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %120 = vector.broadcast %119 : f32 to vector<1x1xf32>
        %121 = vector.extract %103#1[%arg11, 0] : f32 from vector<2x1xf32>
        %122 = vector.broadcast %121 : f32 to vector<1x1xf32>
        %123 = arith.subf %120, %122 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %124 = vector.extract %123[0, 0] : f32 from vector<1x1xf32>
        %125 = vector.insert %124, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %126 = math.exp2 %123 {tiled = true} : vector<1x1xf32>
        %127 = vector.extract %126[0, 0] : f32 from vector<1x1xf32>
        %128 = vector.insert %127, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %129 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %130 = vector.broadcast %129 : f32 to vector<1x1xf32>
        %131 = arith.mulf %130, %126 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %132 = vector.extract %131[0, 0] : f32 from vector<1x1xf32>
        %133 = vector.insert %132, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %125, %128, %133 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %106:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %121 = vector.extract %104#1[%119, %120] : f32 from vector<2x32xf32>
        %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %123 = vector.extract %104#1[%119, %122] : f32 from vector<2x32xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %125 = vector.extract %104#1[%119, %124] : f32 from vector<2x32xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %127 = vector.extract %104#1[%119, %126] : f32 from vector<2x32xf32>
        %128 = vector.from_elements %121, %123, %125, %127 : vector<1x4xf32>
        %129 = math.exp2 %128 {tiled = true} : vector<1x4xf32>
        %130 = vector.extract %129[0, 0] : f32 from vector<1x4xf32>
        %131 = vector.insert %130, %arg12 [%119, %120] : f32 into vector<2x32xf32>
        %132 = vector.extract %129[0, 1] : f32 from vector<1x4xf32>
        %133 = vector.insert %132, %131 [%119, %122] : f32 into vector<2x32xf32>
        %134 = vector.extract %129[0, 2] : f32 from vector<1x4xf32>
        %135 = vector.insert %134, %133 [%119, %124] : f32 into vector<2x32xf32>
        %136 = vector.extract %129[0, 3] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %135 [%119, %126] : f32 into vector<2x32xf32>
        %138 = vector.insert %130, %arg13 [%119, %120] : f32 into vector<2x32xf32>
        %139 = vector.insert %132, %138 [%119, %122] : f32 into vector<2x32xf32>
        %140 = vector.insert %134, %139 [%119, %124] : f32 into vector<2x32xf32>
        %141 = vector.insert %136, %140 [%119, %126] : f32 into vector<2x32xf32>
        affine.yield %137, %141 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %107:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %119 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_2) -> (f32) {
          %129 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %130 = vector.extract %106#1[%arg11, %129] : f32 from vector<2x32xf32>
          %131 = arith.addf %arg15, %130 : f32
          %132 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %133 = vector.extract %106#1[%arg11, %132] : f32 from vector<2x32xf32>
          %134 = arith.addf %131, %133 : f32
          %135 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %136 = vector.extract %106#1[%arg11, %135] : f32 from vector<2x32xf32>
          %137 = arith.addf %134, %136 : f32
          %138 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %139 = vector.extract %106#1[%arg11, %138] : f32 from vector<2x32xf32>
          %140 = arith.addf %137, %139 : f32
          affine.yield %140 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %119, %c16_i32, %c64_i32 : f32
        %120 = arith.addf %119, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %120, %c32_i32, %c64_i32 : f32
        %121 = arith.addf %120, %shuffleResult_31 : f32
        %122 = vector.broadcast %121 : f32 to vector<1x1xf32>
        %123 = vector.insert %121, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %124 = vector.extract %105#2[%arg11, 0] : f32 from vector<2x1xf32>
        %125 = vector.broadcast %124 : f32 to vector<1x1xf32>
        %126 = arith.addf %122, %125 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %127 = vector.extract %126[0, 0] : f32 from vector<1x1xf32>
        %128 = vector.insert %127, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %123, %128 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %108 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_1) -> (vector<2x32xf16>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %121 = vector.extract %106#0[%119, %120] : f32 from vector<2x32xf32>
        %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %123 = vector.extract %106#0[%119, %122] : f32 from vector<2x32xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %125 = vector.extract %106#0[%119, %124] : f32 from vector<2x32xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %127 = vector.extract %106#0[%119, %126] : f32 from vector<2x32xf32>
        %128 = vector.from_elements %121, %123, %125, %127 : vector<1x4xf32>
        %129 = arith.truncf %128 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %130 = vector.extract %129[0, 0] : f16 from vector<1x4xf16>
        %131 = vector.insert %130, %arg12 [%119, %120] : f16 into vector<2x32xf16>
        %132 = vector.extract %129[0, 1] : f16 from vector<1x4xf16>
        %133 = vector.insert %132, %131 [%119, %122] : f16 into vector<2x32xf16>
        %134 = vector.extract %129[0, 2] : f16 from vector<1x4xf16>
        %135 = vector.insert %134, %133 [%119, %124] : f16 into vector<2x32xf16>
        %136 = vector.extract %129[0, 3] : f16 from vector<1x4xf16>
        %137 = vector.insert %136, %135 [%119, %126] : f16 into vector<2x32xf16>
        affine.yield %137 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_24 = memref.subview %arg2[0, %94, %arg4, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_24 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_23 : memref<128x32xf16, 3>
      %109 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %120 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %121 = frisk.read_matrix_from_shm %alloc_23[%119, %120] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %122 = vector.bitcast %121 : vector<4xi32> to vector<8xf16>
        %123 = vector.extract %122[0] : f16 from vector<8xf16>
        %124 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %125 = vector.extract %122[%124] : f16 from vector<8xf16>
        %126 = vector.extract %122[1] : f16 from vector<8xf16>
        %127 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %128 = vector.extract %122[%127] : f16 from vector<8xf16>
        %129 = vector.extract %122[2] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %131 = vector.extract %122[%130] : f16 from vector<8xf16>
        %132 = vector.extract %122[3] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %134 = vector.extract %122[%133] : f16 from vector<8xf16>
        %135 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %136 = vector.insert %123, %arg12 [%135, 0] : f16 into vector<32x2xf16>
        %137 = vector.insert %125, %136 [%135, 1] : f16 into vector<32x2xf16>
        %138 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %139 = vector.insert %126, %137 [%138, 0] : f16 into vector<32x2xf16>
        %140 = vector.insert %128, %139 [%138, 1] : f16 into vector<32x2xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %142 = vector.insert %129, %140 [%141, 0] : f16 into vector<32x2xf16>
        %143 = vector.insert %131, %142 [%141, 1] : f16 into vector<32x2xf16>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %145 = vector.insert %132, %143 [%144, 0] : f16 into vector<32x2xf16>
        %146 = vector.insert %134, %145 [%144, 1] : f16 into vector<32x2xf16>
        affine.yield %146 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %110:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %162 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %163 = vector.extract %108[%119, %162] : f16 from vector<2x32xf16>
          %164 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %165 = vector.extract %108[%119, %164] : f16 from vector<2x32xf16>
          %166 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %167 = vector.extract %108[%119, %166] : f16 from vector<2x32xf16>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %169 = vector.extract %108[%119, %168] : f16 from vector<2x32xf16>
          %170 = vector.from_elements %163, %165, %167, %169 : vector<1x4xf16>
          %171 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %172 = vector.extract %109[%162, %171] : f16 from vector<32x2xf16>
          %173 = vector.extract %109[%164, %171] : f16 from vector<32x2xf16>
          %174 = vector.extract %109[%166, %171] : f16 from vector<32x2xf16>
          %175 = vector.extract %109[%168, %171] : f16 from vector<32x2xf16>
          %176 = vector.from_elements %172, %173, %174, %175 : vector<4x1xf16>
          %177 = frisk.warp_mma_rr %170, %176, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %177 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %122 = vector.extract %120[0, 0] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %arg12 [%119, %121] : f32 into vector<2x8xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %125 = vector.extract %120[0, 1] : f32 from vector<1x4xf32>
        %126 = vector.insert %125, %123 [%119, %124] : f32 into vector<2x8xf32>
        %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %128 = vector.extract %120[0, 2] : f32 from vector<1x4xf32>
        %129 = vector.insert %128, %126 [%119, %127] : f32 into vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %131 = vector.extract %120[0, 3] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %129 [%119, %130] : f32 into vector<2x8xf32>
        %133 = vector.extract %105#1[%119, 0] : f32 from vector<2x1xf32>
        %134 = vector.broadcast %133 : f32 to vector<1x4xf32>
        %135 = vector.insert %133, %arg13 [%119, %121] : f32 into vector<2x8xf32>
        %136 = vector.insert %133, %135 [%119, %124] : f32 into vector<2x8xf32>
        %137 = vector.insert %133, %136 [%119, %127] : f32 into vector<2x8xf32>
        %138 = vector.insert %133, %137 [%119, %130] : f32 into vector<2x8xf32>
        %139 = vector.extract %arg5[%119, %121] : f32 from vector<2x8xf32>
        %140 = vector.extract %arg5[%119, %124] : f32 from vector<2x8xf32>
        %141 = vector.extract %arg5[%119, %127] : f32 from vector<2x8xf32>
        %142 = vector.extract %arg5[%119, %130] : f32 from vector<2x8xf32>
        %143 = vector.from_elements %139, %140, %141, %142 : vector<1x4xf32>
        %144 = arith.mulf %143, %134 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %145 = vector.extract %144[0, 0] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %arg14 [%119, %121] : f32 into vector<2x8xf32>
        %147 = vector.extract %144[0, 1] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%119, %124] : f32 into vector<2x8xf32>
        %149 = vector.extract %144[0, 2] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%119, %127] : f32 into vector<2x8xf32>
        %151 = vector.extract %144[0, 3] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%119, %130] : f32 into vector<2x8xf32>
        %153 = arith.addf %144, %120 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %arg15 [%119, %121] : f32 into vector<2x8xf32>
        %156 = vector.extract %153[0, 1] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%119, %124] : f32 into vector<2x8xf32>
        %158 = vector.extract %153[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%119, %127] : f32 into vector<2x8xf32>
        %160 = vector.extract %153[0, 3] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%119, %130] : f32 into vector<2x8xf32>
        affine.yield %132, %138, %152, %161 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_25 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_26 = memref.subview %arg2[0, %94, %arg4, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_26 to %alloc_25 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_25 : memref<128x32xf16, 3>
      %111 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %120 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %121 = frisk.read_matrix_from_shm %alloc_25[%119, %120] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %122 = vector.bitcast %121 : vector<4xi32> to vector<8xf16>
        %123 = vector.extract %122[0] : f16 from vector<8xf16>
        %124 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %125 = vector.extract %122[%124] : f16 from vector<8xf16>
        %126 = vector.extract %122[1] : f16 from vector<8xf16>
        %127 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %128 = vector.extract %122[%127] : f16 from vector<8xf16>
        %129 = vector.extract %122[2] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %131 = vector.extract %122[%130] : f16 from vector<8xf16>
        %132 = vector.extract %122[3] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %134 = vector.extract %122[%133] : f16 from vector<8xf16>
        %135 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %136 = vector.insert %123, %arg12 [%135, 0] : f16 into vector<32x2xf16>
        %137 = vector.insert %125, %136 [%135, 1] : f16 into vector<32x2xf16>
        %138 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %139 = vector.insert %126, %137 [%138, 0] : f16 into vector<32x2xf16>
        %140 = vector.insert %128, %139 [%138, 1] : f16 into vector<32x2xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %142 = vector.insert %129, %140 [%141, 0] : f16 into vector<32x2xf16>
        %143 = vector.insert %131, %142 [%141, 1] : f16 into vector<32x2xf16>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %145 = vector.insert %132, %143 [%144, 0] : f16 into vector<32x2xf16>
        %146 = vector.insert %134, %145 [%144, 1] : f16 into vector<32x2xf16>
        affine.yield %146 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %112:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %162 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %163 = vector.extract %108[%119, %162] : f16 from vector<2x32xf16>
          %164 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %165 = vector.extract %108[%119, %164] : f16 from vector<2x32xf16>
          %166 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %167 = vector.extract %108[%119, %166] : f16 from vector<2x32xf16>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %169 = vector.extract %108[%119, %168] : f16 from vector<2x32xf16>
          %170 = vector.from_elements %163, %165, %167, %169 : vector<1x4xf16>
          %171 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %172 = vector.extract %111[%162, %171] : f16 from vector<32x2xf16>
          %173 = vector.extract %111[%164, %171] : f16 from vector<32x2xf16>
          %174 = vector.extract %111[%166, %171] : f16 from vector<32x2xf16>
          %175 = vector.extract %111[%168, %171] : f16 from vector<32x2xf16>
          %176 = vector.from_elements %172, %173, %174, %175 : vector<4x1xf16>
          %177 = frisk.warp_mma_rr %170, %176, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %177 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %122 = vector.extract %120[0, 0] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %arg12 [%119, %121] : f32 into vector<2x8xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %125 = vector.extract %120[0, 1] : f32 from vector<1x4xf32>
        %126 = vector.insert %125, %123 [%119, %124] : f32 into vector<2x8xf32>
        %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %128 = vector.extract %120[0, 2] : f32 from vector<1x4xf32>
        %129 = vector.insert %128, %126 [%119, %127] : f32 into vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %131 = vector.extract %120[0, 3] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %129 [%119, %130] : f32 into vector<2x8xf32>
        %133 = vector.extract %105#1[%119, 0] : f32 from vector<2x1xf32>
        %134 = vector.broadcast %133 : f32 to vector<1x4xf32>
        %135 = vector.insert %133, %arg13 [%119, %121] : f32 into vector<2x8xf32>
        %136 = vector.insert %133, %135 [%119, %124] : f32 into vector<2x8xf32>
        %137 = vector.insert %133, %136 [%119, %127] : f32 into vector<2x8xf32>
        %138 = vector.insert %133, %137 [%119, %130] : f32 into vector<2x8xf32>
        %139 = vector.extract %arg8[%119, %121] : f32 from vector<2x8xf32>
        %140 = vector.extract %arg8[%119, %124] : f32 from vector<2x8xf32>
        %141 = vector.extract %arg8[%119, %127] : f32 from vector<2x8xf32>
        %142 = vector.extract %arg8[%119, %130] : f32 from vector<2x8xf32>
        %143 = vector.from_elements %139, %140, %141, %142 : vector<1x4xf32>
        %144 = arith.mulf %143, %134 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %145 = vector.extract %144[0, 0] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %arg14 [%119, %121] : f32 into vector<2x8xf32>
        %147 = vector.extract %144[0, 1] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%119, %124] : f32 into vector<2x8xf32>
        %149 = vector.extract %144[0, 2] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%119, %127] : f32 into vector<2x8xf32>
        %151 = vector.extract %144[0, 3] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%119, %130] : f32 into vector<2x8xf32>
        %153 = arith.addf %144, %120 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %arg15 [%119, %121] : f32 into vector<2x8xf32>
        %156 = vector.extract %153[0, 1] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%119, %124] : f32 into vector<2x8xf32>
        %158 = vector.extract %153[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%119, %127] : f32 into vector<2x8xf32>
        %160 = vector.extract %153[0, 3] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%119, %130] : f32 into vector<2x8xf32>
        affine.yield %132, %138, %152, %161 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_27 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_28 = memref.subview %arg2[0, %94, %arg4, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_28 to %alloc_27 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_27 : memref<128x32xf16, 3>
      %113 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %120 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %121 = frisk.read_matrix_from_shm %alloc_27[%119, %120] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %122 = vector.bitcast %121 : vector<4xi32> to vector<8xf16>
        %123 = vector.extract %122[0] : f16 from vector<8xf16>
        %124 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %125 = vector.extract %122[%124] : f16 from vector<8xf16>
        %126 = vector.extract %122[1] : f16 from vector<8xf16>
        %127 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %128 = vector.extract %122[%127] : f16 from vector<8xf16>
        %129 = vector.extract %122[2] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %131 = vector.extract %122[%130] : f16 from vector<8xf16>
        %132 = vector.extract %122[3] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %134 = vector.extract %122[%133] : f16 from vector<8xf16>
        %135 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %136 = vector.insert %123, %arg12 [%135, 0] : f16 into vector<32x2xf16>
        %137 = vector.insert %125, %136 [%135, 1] : f16 into vector<32x2xf16>
        %138 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %139 = vector.insert %126, %137 [%138, 0] : f16 into vector<32x2xf16>
        %140 = vector.insert %128, %139 [%138, 1] : f16 into vector<32x2xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %142 = vector.insert %129, %140 [%141, 0] : f16 into vector<32x2xf16>
        %143 = vector.insert %131, %142 [%141, 1] : f16 into vector<32x2xf16>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %145 = vector.insert %132, %143 [%144, 0] : f16 into vector<32x2xf16>
        %146 = vector.insert %134, %145 [%144, 1] : f16 into vector<32x2xf16>
        affine.yield %146 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %114:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %162 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %163 = vector.extract %108[%119, %162] : f16 from vector<2x32xf16>
          %164 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %165 = vector.extract %108[%119, %164] : f16 from vector<2x32xf16>
          %166 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %167 = vector.extract %108[%119, %166] : f16 from vector<2x32xf16>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %169 = vector.extract %108[%119, %168] : f16 from vector<2x32xf16>
          %170 = vector.from_elements %163, %165, %167, %169 : vector<1x4xf16>
          %171 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %172 = vector.extract %113[%162, %171] : f16 from vector<32x2xf16>
          %173 = vector.extract %113[%164, %171] : f16 from vector<32x2xf16>
          %174 = vector.extract %113[%166, %171] : f16 from vector<32x2xf16>
          %175 = vector.extract %113[%168, %171] : f16 from vector<32x2xf16>
          %176 = vector.from_elements %172, %173, %174, %175 : vector<4x1xf16>
          %177 = frisk.warp_mma_rr %170, %176, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %177 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %122 = vector.extract %120[0, 0] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %arg12 [%119, %121] : f32 into vector<2x8xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %125 = vector.extract %120[0, 1] : f32 from vector<1x4xf32>
        %126 = vector.insert %125, %123 [%119, %124] : f32 into vector<2x8xf32>
        %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %128 = vector.extract %120[0, 2] : f32 from vector<1x4xf32>
        %129 = vector.insert %128, %126 [%119, %127] : f32 into vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %131 = vector.extract %120[0, 3] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %129 [%119, %130] : f32 into vector<2x8xf32>
        %133 = vector.extract %105#1[%119, 0] : f32 from vector<2x1xf32>
        %134 = vector.broadcast %133 : f32 to vector<1x4xf32>
        %135 = vector.insert %133, %arg13 [%119, %121] : f32 into vector<2x8xf32>
        %136 = vector.insert %133, %135 [%119, %124] : f32 into vector<2x8xf32>
        %137 = vector.insert %133, %136 [%119, %127] : f32 into vector<2x8xf32>
        %138 = vector.insert %133, %137 [%119, %130] : f32 into vector<2x8xf32>
        %139 = vector.extract %arg9[%119, %121] : f32 from vector<2x8xf32>
        %140 = vector.extract %arg9[%119, %124] : f32 from vector<2x8xf32>
        %141 = vector.extract %arg9[%119, %127] : f32 from vector<2x8xf32>
        %142 = vector.extract %arg9[%119, %130] : f32 from vector<2x8xf32>
        %143 = vector.from_elements %139, %140, %141, %142 : vector<1x4xf32>
        %144 = arith.mulf %143, %134 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %145 = vector.extract %144[0, 0] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %arg14 [%119, %121] : f32 into vector<2x8xf32>
        %147 = vector.extract %144[0, 1] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%119, %124] : f32 into vector<2x8xf32>
        %149 = vector.extract %144[0, 2] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%119, %127] : f32 into vector<2x8xf32>
        %151 = vector.extract %144[0, 3] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%119, %130] : f32 into vector<2x8xf32>
        %153 = arith.addf %144, %120 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %arg15 [%119, %121] : f32 into vector<2x8xf32>
        %156 = vector.extract %153[0, 1] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%119, %124] : f32 into vector<2x8xf32>
        %158 = vector.extract %153[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%119, %127] : f32 into vector<2x8xf32>
        %160 = vector.extract %153[0, 3] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%119, %130] : f32 into vector<2x8xf32>
        affine.yield %132, %138, %152, %161 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_29 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_30 = memref.subview %arg2[0, %94, %arg4, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_30 to %alloc_29 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_29 : memref<128x32xf16, 3>
      %115 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %120 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %121 = frisk.read_matrix_from_shm %alloc_29[%119, %120] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %122 = vector.bitcast %121 : vector<4xi32> to vector<8xf16>
        %123 = vector.extract %122[0] : f16 from vector<8xf16>
        %124 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %125 = vector.extract %122[%124] : f16 from vector<8xf16>
        %126 = vector.extract %122[1] : f16 from vector<8xf16>
        %127 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %128 = vector.extract %122[%127] : f16 from vector<8xf16>
        %129 = vector.extract %122[2] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %131 = vector.extract %122[%130] : f16 from vector<8xf16>
        %132 = vector.extract %122[3] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %134 = vector.extract %122[%133] : f16 from vector<8xf16>
        %135 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %136 = vector.insert %123, %arg12 [%135, 0] : f16 into vector<32x2xf16>
        %137 = vector.insert %125, %136 [%135, 1] : f16 into vector<32x2xf16>
        %138 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %139 = vector.insert %126, %137 [%138, 0] : f16 into vector<32x2xf16>
        %140 = vector.insert %128, %139 [%138, 1] : f16 into vector<32x2xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %142 = vector.insert %129, %140 [%141, 0] : f16 into vector<32x2xf16>
        %143 = vector.insert %131, %142 [%141, 1] : f16 into vector<32x2xf16>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %145 = vector.insert %132, %143 [%144, 0] : f16 into vector<32x2xf16>
        %146 = vector.insert %134, %145 [%144, 1] : f16 into vector<32x2xf16>
        affine.yield %146 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %116:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %170 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %171 = vector.extract %108[%119, %170] : f16 from vector<2x32xf16>
          %172 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %173 = vector.extract %108[%119, %172] : f16 from vector<2x32xf16>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %175 = vector.extract %108[%119, %174] : f16 from vector<2x32xf16>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %177 = vector.extract %108[%119, %176] : f16 from vector<2x32xf16>
          %178 = vector.from_elements %171, %173, %175, %177 : vector<1x4xf16>
          %179 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %180 = vector.extract %115[%170, %179] : f16 from vector<32x2xf16>
          %181 = vector.extract %115[%172, %179] : f16 from vector<32x2xf16>
          %182 = vector.extract %115[%174, %179] : f16 from vector<32x2xf16>
          %183 = vector.extract %115[%176, %179] : f16 from vector<32x2xf16>
          %184 = vector.from_elements %180, %181, %182, %183 : vector<4x1xf16>
          %185 = frisk.warp_mma_rr %178, %184, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %185 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %121 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %122 = vector.extract %120[0, 0] : f32 from vector<1x4xf32>
        %123 = vector.insert %122, %arg12 [%119, %121] : f32 into vector<2x8xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %125 = vector.extract %120[0, 1] : f32 from vector<1x4xf32>
        %126 = vector.insert %125, %123 [%119, %124] : f32 into vector<2x8xf32>
        %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %128 = vector.extract %120[0, 2] : f32 from vector<1x4xf32>
        %129 = vector.insert %128, %126 [%119, %127] : f32 into vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %131 = vector.extract %120[0, 3] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %129 [%119, %130] : f32 into vector<2x8xf32>
        %133 = vector.extract %105#1[%119, 0] : f32 from vector<2x1xf32>
        %134 = vector.broadcast %133 : f32 to vector<1x4xf32>
        %135 = vector.insert %133, %arg13 [%119, %121] : f32 into vector<2x8xf32>
        %136 = vector.insert %133, %135 [%119, %124] : f32 into vector<2x8xf32>
        %137 = vector.insert %133, %136 [%119, %127] : f32 into vector<2x8xf32>
        %138 = vector.insert %133, %137 [%119, %130] : f32 into vector<2x8xf32>
        %139 = vector.extract %arg10[%119, %121] : f32 from vector<2x8xf32>
        %140 = vector.extract %arg10[%119, %124] : f32 from vector<2x8xf32>
        %141 = vector.extract %arg10[%119, %127] : f32 from vector<2x8xf32>
        %142 = vector.extract %arg10[%119, %130] : f32 from vector<2x8xf32>
        %143 = vector.from_elements %139, %140, %141, %142 : vector<1x4xf32>
        %144 = arith.mulf %143, %134 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %145 = vector.extract %144[0, 0] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %arg14 [%119, %121] : f32 into vector<2x8xf32>
        %147 = vector.extract %144[0, 1] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%119, %124] : f32 into vector<2x8xf32>
        %149 = vector.extract %144[0, 2] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %148 [%119, %127] : f32 into vector<2x8xf32>
        %151 = vector.extract %144[0, 3] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%119, %130] : f32 into vector<2x8xf32>
        %153 = arith.addf %144, %120 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %154 = vector.extract %153[0, 0] : f32 from vector<1x4xf32>
        %155 = vector.insert %154, %arg15 [%119, %121] : f32 into vector<2x8xf32>
        %156 = vector.extract %153[0, 1] : f32 from vector<1x4xf32>
        %157 = vector.insert %156, %155 [%119, %124] : f32 into vector<2x8xf32>
        %158 = vector.extract %153[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %157 [%119, %127] : f32 into vector<2x8xf32>
        %160 = vector.extract %153[0, 3] : f32 from vector<1x4xf32>
        %161 = vector.insert %160, %159 [%119, %130] : f32 into vector<2x8xf32>
        %162 = vector.extract %110#3[%119, %121] : f32 from vector<2x8xf32>
        %163 = vector.extract %110#3[%119, %124] : f32 from vector<2x8xf32>
        %164 = vector.extract %110#3[%119, %127] : f32 from vector<2x8xf32>
        %165 = vector.extract %110#3[%119, %130] : f32 from vector<2x8xf32>
        %166 = vector.insert %162, %arg16 [%119, %121] : f32 into vector<2x8xf32>
        %167 = vector.insert %163, %166 [%119, %124] : f32 into vector<2x8xf32>
        %168 = vector.insert %164, %167 [%119, %127] : f32 into vector<2x8xf32>
        %169 = vector.insert %165, %168 [%119, %130] : f32 into vector<2x8xf32>
        affine.yield %132, %138, %152, %161, %169 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %117:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %119 = vector.extract %107#1[%arg11, 0] : f32 from vector<2x1xf32>
        %120 = vector.insert %119, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %121 = vector.extract %103#1[%arg11, 0] : f32 from vector<2x1xf32>
        %122 = vector.insert %121, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %120, %122 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %118:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %119 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %120 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %121 = vector.extract %112#3[%119, %120] : f32 from vector<2x8xf32>
        %122 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %123 = vector.extract %112#3[%119, %122] : f32 from vector<2x8xf32>
        %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %125 = vector.extract %112#3[%119, %124] : f32 from vector<2x8xf32>
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %127 = vector.extract %112#3[%119, %126] : f32 from vector<2x8xf32>
        %128 = vector.insert %121, %arg12 [%119, %120] : f32 into vector<2x8xf32>
        %129 = vector.insert %123, %128 [%119, %122] : f32 into vector<2x8xf32>
        %130 = vector.insert %125, %129 [%119, %124] : f32 into vector<2x8xf32>
        %131 = vector.insert %127, %130 [%119, %126] : f32 into vector<2x8xf32>
        %132 = vector.extract %114#3[%119, %120] : f32 from vector<2x8xf32>
        %133 = vector.extract %114#3[%119, %122] : f32 from vector<2x8xf32>
        %134 = vector.extract %114#3[%119, %124] : f32 from vector<2x8xf32>
        %135 = vector.extract %114#3[%119, %126] : f32 from vector<2x8xf32>
        %136 = vector.insert %132, %arg13 [%119, %120] : f32 into vector<2x8xf32>
        %137 = vector.insert %133, %136 [%119, %122] : f32 into vector<2x8xf32>
        %138 = vector.insert %134, %137 [%119, %124] : f32 into vector<2x8xf32>
        %139 = vector.insert %135, %138 [%119, %126] : f32 into vector<2x8xf32>
        %140 = vector.extract %116#3[%119, %120] : f32 from vector<2x8xf32>
        %141 = vector.extract %116#3[%119, %122] : f32 from vector<2x8xf32>
        %142 = vector.extract %116#3[%119, %124] : f32 from vector<2x8xf32>
        %143 = vector.extract %116#3[%119, %126] : f32 from vector<2x8xf32>
        %144 = vector.insert %140, %arg14 [%119, %120] : f32 into vector<2x8xf32>
        %145 = vector.insert %141, %144 [%119, %122] : f32 into vector<2x8xf32>
        %146 = vector.insert %142, %145 [%119, %124] : f32 into vector<2x8xf32>
        %147 = vector.insert %143, %146 [%119, %126] : f32 into vector<2x8xf32>
        affine.yield %131, %139, %147 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %116#4, %117#0, %117#1, %118#0, %118#1, %118#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %82 = arith.index_cast %thread_id_x : index to i32
    %83 = arith.andi %82, %c16_i32 : i32
    %84 = arith.cmpi ne, %83, %c0_i32 : i32
    %85 = arith.andi %82, %c32_i32 : i32
    %86 = arith.cmpi ne, %85, %c0_i32 : i32
    %87:8 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst_5, %arg12 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %93 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %94 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %95 = vector.extract %81#5[%93, %94] : f32 from vector<2x8xf32>
      %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %97 = vector.extract %81#5[%93, %96] : f32 from vector<2x8xf32>
      %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %99 = vector.extract %81#5[%93, %98] : f32 from vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %101 = vector.extract %81#5[%93, %100] : f32 from vector<2x8xf32>
      %102 = arith.select %84, %95, %97 : f32
      %shuffleResult, %valid = gpu.shuffle xor %102, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %103 = arith.select %84, %shuffleResult, %95 : f32
      %104 = arith.select %84, %97, %shuffleResult : f32
      %105 = arith.select %84, %99, %101 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %105, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %106 = arith.select %84, %shuffleResult_15, %99 : f32
      %107 = arith.select %84, %101, %shuffleResult_15 : f32
      %108 = arith.select %86, %103, %106 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %108, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %109 = arith.select %86, %shuffleResult_17, %103 : f32
      %110 = arith.select %86, %106, %shuffleResult_17 : f32
      %111 = arith.select %86, %104, %107 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %111, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %112 = arith.select %86, %shuffleResult_19, %104 : f32
      %113 = arith.select %86, %107, %shuffleResult_19 : f32
      %114 = vector.insert %109, %arg5 [%93, %94] : f32 into vector<2x8xf32>
      %115 = vector.insert %112, %114 [%93, %96] : f32 into vector<2x8xf32>
      %116 = vector.insert %110, %115 [%93, %98] : f32 into vector<2x8xf32>
      %117 = vector.insert %113, %116 [%93, %100] : f32 into vector<2x8xf32>
      %118 = vector.extract %81#4[%93, %94] : f32 from vector<2x8xf32>
      %119 = vector.extract %81#4[%93, %96] : f32 from vector<2x8xf32>
      %120 = vector.extract %81#4[%93, %98] : f32 from vector<2x8xf32>
      %121 = vector.extract %81#4[%93, %100] : f32 from vector<2x8xf32>
      %122 = arith.select %84, %118, %119 : f32
      %shuffleResult_21, %valid_22 = gpu.shuffle xor %122, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %123 = arith.select %84, %shuffleResult_21, %118 : f32
      %124 = arith.select %84, %119, %shuffleResult_21 : f32
      %125 = arith.select %84, %120, %121 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %125, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %126 = arith.select %84, %shuffleResult_23, %120 : f32
      %127 = arith.select %84, %121, %shuffleResult_23 : f32
      %128 = arith.select %86, %123, %126 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %128, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %129 = arith.select %86, %shuffleResult_25, %123 : f32
      %130 = arith.select %86, %126, %shuffleResult_25 : f32
      %131 = arith.select %86, %124, %127 : f32
      %shuffleResult_27, %valid_28 = gpu.shuffle xor %131, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %132 = arith.select %86, %shuffleResult_27, %124 : f32
      %133 = arith.select %86, %127, %shuffleResult_27 : f32
      %134 = vector.insert %129, %arg6 [%93, %94] : f32 into vector<2x8xf32>
      %135 = vector.insert %132, %134 [%93, %96] : f32 into vector<2x8xf32>
      %136 = vector.insert %130, %135 [%93, %98] : f32 into vector<2x8xf32>
      %137 = vector.insert %133, %136 [%93, %100] : f32 into vector<2x8xf32>
      %138 = vector.extract %81#3[%93, %94] : f32 from vector<2x8xf32>
      %139 = vector.extract %81#3[%93, %96] : f32 from vector<2x8xf32>
      %140 = vector.extract %81#3[%93, %98] : f32 from vector<2x8xf32>
      %141 = vector.extract %81#3[%93, %100] : f32 from vector<2x8xf32>
      %142 = arith.select %84, %138, %139 : f32
      %shuffleResult_29, %valid_30 = gpu.shuffle xor %142, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %143 = arith.select %84, %shuffleResult_29, %138 : f32
      %144 = arith.select %84, %139, %shuffleResult_29 : f32
      %145 = arith.select %84, %140, %141 : f32
      %shuffleResult_31, %valid_32 = gpu.shuffle xor %145, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %146 = arith.select %84, %shuffleResult_31, %140 : f32
      %147 = arith.select %84, %141, %shuffleResult_31 : f32
      %148 = arith.select %86, %143, %146 : f32
      %shuffleResult_33, %valid_34 = gpu.shuffle xor %148, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %149 = arith.select %86, %shuffleResult_33, %143 : f32
      %150 = arith.select %86, %146, %shuffleResult_33 : f32
      %151 = arith.select %86, %144, %147 : f32
      %shuffleResult_35, %valid_36 = gpu.shuffle xor %151, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %152 = arith.select %86, %shuffleResult_35, %144 : f32
      %153 = arith.select %86, %147, %shuffleResult_35 : f32
      %154 = vector.insert %149, %arg7 [%93, %94] : f32 into vector<2x8xf32>
      %155 = vector.insert %152, %154 [%93, %96] : f32 into vector<2x8xf32>
      %156 = vector.insert %150, %155 [%93, %98] : f32 into vector<2x8xf32>
      %157 = vector.insert %153, %156 [%93, %100] : f32 into vector<2x8xf32>
      %158 = vector.extract %81#0[%93, %94] : f32 from vector<2x8xf32>
      %159 = vector.extract %81#0[%93, %96] : f32 from vector<2x8xf32>
      %160 = vector.extract %81#0[%93, %98] : f32 from vector<2x8xf32>
      %161 = vector.extract %81#0[%93, %100] : f32 from vector<2x8xf32>
      %162 = arith.select %84, %158, %159 : f32
      %shuffleResult_37, %valid_38 = gpu.shuffle xor %162, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %163 = arith.select %84, %shuffleResult_37, %158 : f32
      %164 = arith.select %84, %159, %shuffleResult_37 : f32
      %165 = arith.select %84, %160, %161 : f32
      %shuffleResult_39, %valid_40 = gpu.shuffle xor %165, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %166 = arith.select %84, %shuffleResult_39, %160 : f32
      %167 = arith.select %84, %161, %shuffleResult_39 : f32
      %168 = arith.select %86, %163, %166 : f32
      %shuffleResult_41, %valid_42 = gpu.shuffle xor %168, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %169 = arith.select %86, %shuffleResult_41, %163 : f32
      %170 = arith.select %86, %166, %shuffleResult_41 : f32
      %171 = arith.select %86, %164, %167 : f32
      %shuffleResult_43, %valid_44 = gpu.shuffle xor %171, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %172 = arith.select %86, %shuffleResult_43, %164 : f32
      %173 = arith.select %86, %167, %shuffleResult_43 : f32
      %174 = vector.from_elements %169, %172, %170, %173 : vector<1x4xf32>
      %175 = vector.insert %169, %arg8 [%93, %94] : f32 into vector<2x8xf32>
      %176 = vector.insert %172, %175 [%93, %96] : f32 into vector<2x8xf32>
      %177 = vector.insert %170, %176 [%93, %98] : f32 into vector<2x8xf32>
      %178 = vector.insert %173, %177 [%93, %100] : f32 into vector<2x8xf32>
      %179 = vector.extract %81#1[%93, 0] : f32 from vector<2x1xf32>
      %180 = vector.broadcast %179 : f32 to vector<1x4xf32>
      %181 = vector.insert %179, %arg9 [%93, %94] : f32 into vector<2x8xf32>
      %182 = vector.insert %179, %181 [%93, %96] : f32 into vector<2x8xf32>
      %183 = vector.insert %179, %182 [%93, %98] : f32 into vector<2x8xf32>
      %184 = vector.insert %179, %183 [%93, %100] : f32 into vector<2x8xf32>
      %185 = arith.divf %174, %180 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %186 = vector.extract %185[0, 0] : f32 from vector<1x4xf32>
      %187 = vector.insert %186, %arg10 [%93, %94] : f32 into vector<2x8xf32>
      %188 = vector.extract %185[0, 1] : f32 from vector<1x4xf32>
      %189 = vector.insert %188, %187 [%93, %96] : f32 into vector<2x8xf32>
      %190 = vector.extract %185[0, 2] : f32 from vector<1x4xf32>
      %191 = vector.insert %190, %189 [%93, %98] : f32 into vector<2x8xf32>
      %192 = vector.extract %185[0, 3] : f32 from vector<1x4xf32>
      %193 = vector.insert %192, %191 [%93, %100] : f32 into vector<2x8xf32>
      %194 = arith.truncf %185 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %195 = vector.extract %194[0, 0] : f16 from vector<1x4xf16>
      %196 = vector.insert %195, %arg11 [%93, %94] : f16 into vector<2x8xf16>
      %197 = vector.extract %194[0, 1] : f16 from vector<1x4xf16>
      %198 = vector.insert %197, %196 [%93, %96] : f16 into vector<2x8xf16>
      %199 = vector.extract %194[0, 2] : f16 from vector<1x4xf16>
      %200 = vector.insert %199, %198 [%93, %98] : f16 into vector<2x8xf16>
      %201 = vector.extract %194[0, 3] : f16 from vector<1x4xf16>
      %202 = vector.insert %201, %200 [%93, %100] : f16 into vector<2x8xf16>
      %203 = vector.insert %195, %arg12 [%93, %94] : f16 into vector<2x8xf16>
      %204 = vector.insert %197, %203 [%93, %96] : f16 into vector<2x8xf16>
      %205 = vector.insert %199, %204 [%93, %98] : f16 into vector<2x8xf16>
      %206 = vector.insert %201, %205 [%93, %100] : f16 into vector<2x8xf16>
      affine.yield %117, %137, %157, %178, %184, %193, %202, %206 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    %88 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %89 = arith.cmpi ult, %88, %c4 : index
    scf.if %89 {
      affine.for %arg4 = 0 to 4 {
        %93 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %94 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %95 = vector.extract %87#7[%93, %94] : f16 from vector<2x8xf16>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %97 = vector.extract %87#7[%93, %96] : f16 from vector<2x8xf16>
        %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %99 = vector.extract %87#7[%93, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %101 = vector.extract %87#7[%93, %100] : f16 from vector<2x8xf16>
        affine.store %95, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %97, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %99, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %101, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %90:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %93 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %94 = vector.extract %81#1[%93, 0] : f32 from vector<2x1xf32>
      %95 = vector.broadcast %94 : f32 to vector<1x4xf32>
      %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %97 = vector.insert %94, %arg5 [%93, %96] : f32 into vector<2x8xf32>
      %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %99 = vector.insert %94, %97 [%93, %98] : f32 into vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %101 = vector.insert %94, %99 [%93, %100] : f32 into vector<2x8xf32>
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %103 = vector.insert %94, %101 [%93, %102] : f32 into vector<2x8xf32>
      %104 = vector.extract %87#2[%93, %96] : f32 from vector<2x8xf32>
      %105 = vector.extract %87#2[%93, %98] : f32 from vector<2x8xf32>
      %106 = vector.extract %87#2[%93, %100] : f32 from vector<2x8xf32>
      %107 = vector.extract %87#2[%93, %102] : f32 from vector<2x8xf32>
      %108 = vector.from_elements %104, %105, %106, %107 : vector<1x4xf32>
      %109 = arith.divf %108, %95 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %110 = vector.extract %109[0, 0] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %arg6 [%93, %96] : f32 into vector<2x8xf32>
      %112 = vector.extract %109[0, 1] : f32 from vector<1x4xf32>
      %113 = vector.insert %112, %111 [%93, %98] : f32 into vector<2x8xf32>
      %114 = vector.extract %109[0, 2] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %113 [%93, %100] : f32 into vector<2x8xf32>
      %116 = vector.extract %109[0, 3] : f32 from vector<1x4xf32>
      %117 = vector.insert %116, %115 [%93, %102] : f32 into vector<2x8xf32>
      %118 = arith.truncf %109 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %119 = vector.extract %118[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg7 [%93, %96] : f16 into vector<2x8xf16>
      %121 = vector.extract %118[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %120 [%93, %98] : f16 into vector<2x8xf16>
      %123 = vector.extract %118[0, 2] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%93, %100] : f16 into vector<2x8xf16>
      %125 = vector.extract %118[0, 3] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%93, %102] : f16 into vector<2x8xf16>
      %127 = vector.insert %119, %arg8 [%93, %96] : f16 into vector<2x8xf16>
      %128 = vector.insert %121, %127 [%93, %98] : f16 into vector<2x8xf16>
      %129 = vector.insert %123, %128 [%93, %100] : f16 into vector<2x8xf16>
      %130 = vector.insert %125, %129 [%93, %102] : f16 into vector<2x8xf16>
      affine.yield %103, %117, %126, %130 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %89 {
      affine.for %arg4 = 0 to 4 {
        %93 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %94 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %95 = vector.extract %90#3[%93, %94] : f16 from vector<2x8xf16>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %97 = vector.extract %90#3[%93, %96] : f16 from vector<2x8xf16>
        %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %99 = vector.extract %90#3[%93, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %101 = vector.extract %90#3[%93, %100] : f16 from vector<2x8xf16>
        affine.store %95, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %97, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %99, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %101, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %91:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %93 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %94 = vector.extract %81#1[%93, 0] : f32 from vector<2x1xf32>
      %95 = vector.broadcast %94 : f32 to vector<1x4xf32>
      %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %97 = vector.insert %94, %arg5 [%93, %96] : f32 into vector<2x8xf32>
      %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %99 = vector.insert %94, %97 [%93, %98] : f32 into vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %101 = vector.insert %94, %99 [%93, %100] : f32 into vector<2x8xf32>
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %103 = vector.insert %94, %101 [%93, %102] : f32 into vector<2x8xf32>
      %104 = vector.extract %87#1[%93, %96] : f32 from vector<2x8xf32>
      %105 = vector.extract %87#1[%93, %98] : f32 from vector<2x8xf32>
      %106 = vector.extract %87#1[%93, %100] : f32 from vector<2x8xf32>
      %107 = vector.extract %87#1[%93, %102] : f32 from vector<2x8xf32>
      %108 = vector.from_elements %104, %105, %106, %107 : vector<1x4xf32>
      %109 = arith.divf %108, %95 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %110 = vector.extract %109[0, 0] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %arg6 [%93, %96] : f32 into vector<2x8xf32>
      %112 = vector.extract %109[0, 1] : f32 from vector<1x4xf32>
      %113 = vector.insert %112, %111 [%93, %98] : f32 into vector<2x8xf32>
      %114 = vector.extract %109[0, 2] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %113 [%93, %100] : f32 into vector<2x8xf32>
      %116 = vector.extract %109[0, 3] : f32 from vector<1x4xf32>
      %117 = vector.insert %116, %115 [%93, %102] : f32 into vector<2x8xf32>
      %118 = arith.truncf %109 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %119 = vector.extract %118[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg7 [%93, %96] : f16 into vector<2x8xf16>
      %121 = vector.extract %118[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %120 [%93, %98] : f16 into vector<2x8xf16>
      %123 = vector.extract %118[0, 2] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%93, %100] : f16 into vector<2x8xf16>
      %125 = vector.extract %118[0, 3] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%93, %102] : f16 into vector<2x8xf16>
      %127 = vector.insert %119, %arg8 [%93, %96] : f16 into vector<2x8xf16>
      %128 = vector.insert %121, %127 [%93, %98] : f16 into vector<2x8xf16>
      %129 = vector.insert %123, %128 [%93, %100] : f16 into vector<2x8xf16>
      %130 = vector.insert %125, %129 [%93, %102] : f16 into vector<2x8xf16>
      affine.yield %103, %117, %126, %130 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %89 {
      affine.for %arg4 = 0 to 4 {
        %93 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %94 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %95 = vector.extract %91#3[%93, %94] : f16 from vector<2x8xf16>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %97 = vector.extract %91#3[%93, %96] : f16 from vector<2x8xf16>
        %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %99 = vector.extract %91#3[%93, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %101 = vector.extract %91#3[%93, %100] : f16 from vector<2x8xf16>
        affine.store %95, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %97, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %99, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %101, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %92:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %93 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %94 = vector.extract %81#1[%93, 0] : f32 from vector<2x1xf32>
      %95 = vector.broadcast %94 : f32 to vector<1x4xf32>
      %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %97 = vector.insert %94, %arg5 [%93, %96] : f32 into vector<2x8xf32>
      %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %99 = vector.insert %94, %97 [%93, %98] : f32 into vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %101 = vector.insert %94, %99 [%93, %100] : f32 into vector<2x8xf32>
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %103 = vector.insert %94, %101 [%93, %102] : f32 into vector<2x8xf32>
      %104 = vector.extract %87#0[%93, %96] : f32 from vector<2x8xf32>
      %105 = vector.extract %87#0[%93, %98] : f32 from vector<2x8xf32>
      %106 = vector.extract %87#0[%93, %100] : f32 from vector<2x8xf32>
      %107 = vector.extract %87#0[%93, %102] : f32 from vector<2x8xf32>
      %108 = vector.from_elements %104, %105, %106, %107 : vector<1x4xf32>
      %109 = arith.divf %108, %95 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %110 = vector.extract %109[0, 0] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %arg6 [%93, %96] : f32 into vector<2x8xf32>
      %112 = vector.extract %109[0, 1] : f32 from vector<1x4xf32>
      %113 = vector.insert %112, %111 [%93, %98] : f32 into vector<2x8xf32>
      %114 = vector.extract %109[0, 2] : f32 from vector<1x4xf32>
      %115 = vector.insert %114, %113 [%93, %100] : f32 into vector<2x8xf32>
      %116 = vector.extract %109[0, 3] : f32 from vector<1x4xf32>
      %117 = vector.insert %116, %115 [%93, %102] : f32 into vector<2x8xf32>
      %118 = arith.truncf %109 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %119 = vector.extract %118[0, 0] : f16 from vector<1x4xf16>
      %120 = vector.insert %119, %arg7 [%93, %96] : f16 into vector<2x8xf16>
      %121 = vector.extract %118[0, 1] : f16 from vector<1x4xf16>
      %122 = vector.insert %121, %120 [%93, %98] : f16 into vector<2x8xf16>
      %123 = vector.extract %118[0, 2] : f16 from vector<1x4xf16>
      %124 = vector.insert %123, %122 [%93, %100] : f16 into vector<2x8xf16>
      %125 = vector.extract %118[0, 3] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %124 [%93, %102] : f16 into vector<2x8xf16>
      %127 = vector.insert %119, %arg8 [%93, %96] : f16 into vector<2x8xf16>
      %128 = vector.insert %121, %127 [%93, %98] : f16 into vector<2x8xf16>
      %129 = vector.insert %123, %128 [%93, %100] : f16 into vector<2x8xf16>
      %130 = vector.insert %125, %129 [%93, %102] : f16 into vector<2x8xf16>
      affine.yield %103, %117, %126, %130 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %89 {
      affine.for %arg4 = 0 to 4 {
        %93 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %94 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %95 = vector.extract %92#3[%93, %94] : f16 from vector<2x8xf16>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %97 = vector.extract %92#3[%93, %96] : f16 from vector<2x8xf16>
        %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %99 = vector.extract %92#3[%93, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %101 = vector.extract %92#3[%93, %100] : f16 from vector<2x8xf16>
        affine.store %95, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %97, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %99, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %101, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
