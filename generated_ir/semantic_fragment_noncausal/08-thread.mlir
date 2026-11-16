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
      %97 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %98 = vector.insert %cst_7, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %99 = vector.insert %cst_7, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      %100 = vector.insert %cst_7, %arg8 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %97, %98, %99, %100 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %97 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %99 = vector.insert %cst, %arg5 [%98, %97] : f32 into vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %101 = vector.insert %cst, %99 [%98, %100] : f32 into vector<2x8xf32>
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %103 = vector.insert %cst, %101 [%98, %102] : f32 into vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %105 = vector.insert %cst, %103 [%98, %104] : f32 into vector<2x8xf32>
      affine.yield %105 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6) -> (vector<2x1xf32>) {
      %97 = vector.insert %cst, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %97 : vector<2x1xf32>
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
    %24:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %23) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %97 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %99 = memref.load %alloc[%97, %98] : memref<128x32xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %101 = memref.load %alloc[%97, %100] : memref<128x32xf16, 3>
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %103 = memref.load %alloc[%97, %102] : memref<128x32xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %105 = memref.load %alloc[%97, %104] : memref<128x32xf16, 3>
      %106 = vector.from_elements %99, %101, %103, %105 : vector<1x4xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %110 = vector.insert %109, %arg5 [%107, %108] : f16 into vector<2x8xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %110 [%107, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%107, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%107, %117] : f16 into vector<2x8xf16>
      %120 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %121 = vector.insert %8, %arg6 [%107, %108] : f16 into vector<2x8xf16>
      %122 = vector.insert %8, %121 [%107, %111] : f16 into vector<2x8xf16>
      %123 = vector.insert %8, %122 [%107, %114] : f16 into vector<2x8xf16>
      %124 = vector.insert %8, %123 [%107, %117] : f16 into vector<2x8xf16>
      %125 = arith.mulf %arg8, %120 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %126 = vector.extract %125[0, 0] : f16 from vector<1x4xf16>
      %127 = vector.insert %126, %arg7 [%107, %108] : f16 into vector<2x8xf16>
      %128 = vector.extract %125[0, 1] : f16 from vector<1x4xf16>
      %129 = vector.insert %128, %127 [%107, %111] : f16 into vector<2x8xf16>
      %130 = vector.extract %125[0, 2] : f16 from vector<1x4xf16>
      %131 = vector.insert %130, %129 [%107, %114] : f16 into vector<2x8xf16>
      %132 = vector.extract %125[0, 3] : f16 from vector<1x4xf16>
      %133 = vector.insert %132, %131 [%107, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %124, %133, %106 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
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
    %35 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%6, %7]
    %subview_10 = memref.subview %arg0[0, %35, %13, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    frisk.async_copy_gm_to_shm %subview_10 to %alloc_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_9 : memref<128x32xf16, 3>
    %36 = memref.load %alloc_9[%14, %15] : memref<128x32xf16, 3>
    %37 = memref.load %alloc_9[%14, %17] : memref<128x32xf16, 3>
    %38 = memref.load %alloc_9[%14, %19] : memref<128x32xf16, 3>
    %39 = memref.load %alloc_9[%14, %21] : memref<128x32xf16, 3>
    %40 = vector.from_elements %36, %37, %38, %39 : vector<1x4xf16>
    %41:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %40) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %97 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %99 = memref.load %alloc_9[%97, %98] : memref<128x32xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %101 = memref.load %alloc_9[%97, %100] : memref<128x32xf16, 3>
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %103 = memref.load %alloc_9[%97, %102] : memref<128x32xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %105 = memref.load %alloc_9[%97, %104] : memref<128x32xf16, 3>
      %106 = vector.from_elements %99, %101, %103, %105 : vector<1x4xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %110 = vector.insert %109, %arg5 [%107, %108] : f16 into vector<2x8xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %110 [%107, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%107, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%107, %117] : f16 into vector<2x8xf16>
      %120 = vector.insert %8, %arg6 [%107, %108] : f16 into vector<2x8xf16>
      %121 = vector.insert %8, %120 [%107, %111] : f16 into vector<2x8xf16>
      %122 = vector.insert %8, %121 [%107, %114] : f16 into vector<2x8xf16>
      %123 = vector.insert %8, %122 [%107, %117] : f16 into vector<2x8xf16>
      %124 = arith.mulf %arg8, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %125 = vector.extract %124[0, 0] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %arg7 [%107, %108] : f16 into vector<2x8xf16>
      %127 = vector.extract %124[0, 1] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%107, %111] : f16 into vector<2x8xf16>
      %129 = vector.extract %124[0, 2] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%107, %114] : f16 into vector<2x8xf16>
      %131 = vector.extract %124[0, 3] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%107, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %123, %132, %106 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %42 = arith.mulf %41#3, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %43 = vector.extract %42[0, 0] : f16 from vector<1x4xf16>
    %44 = vector.insert %43, %41#2 [1, 4] : f16 into vector<2x8xf16>
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
    %56:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %55) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %97 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %99 = memref.load %alloc_11[%97, %98] : memref<128x32xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %101 = memref.load %alloc_11[%97, %100] : memref<128x32xf16, 3>
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %103 = memref.load %alloc_11[%97, %102] : memref<128x32xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %105 = memref.load %alloc_11[%97, %104] : memref<128x32xf16, 3>
      %106 = vector.from_elements %99, %101, %103, %105 : vector<1x4xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %110 = vector.insert %109, %arg5 [%107, %108] : f16 into vector<2x8xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %110 [%107, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%107, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%107, %117] : f16 into vector<2x8xf16>
      %120 = vector.insert %8, %arg6 [%107, %108] : f16 into vector<2x8xf16>
      %121 = vector.insert %8, %120 [%107, %111] : f16 into vector<2x8xf16>
      %122 = vector.insert %8, %121 [%107, %114] : f16 into vector<2x8xf16>
      %123 = vector.insert %8, %122 [%107, %117] : f16 into vector<2x8xf16>
      %124 = arith.mulf %arg8, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %125 = vector.extract %124[0, 0] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %arg7 [%107, %108] : f16 into vector<2x8xf16>
      %127 = vector.extract %124[0, 1] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%107, %111] : f16 into vector<2x8xf16>
      %129 = vector.extract %124[0, 2] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%107, %114] : f16 into vector<2x8xf16>
      %131 = vector.extract %124[0, 3] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%107, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %123, %132, %106 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %57 = arith.mulf %56#3, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %58 = vector.extract %57[0, 0] : f16 from vector<1x4xf16>
    %59 = vector.insert %58, %56#2 [1, 4] : f16 into vector<2x8xf16>
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
    %71:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_5, %arg8 = %70) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %97 = affine.apply affine_map<(d0)[s0] -> (((d0 + 1) floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg4)[%thread_id_x]
      %99 = memref.load %alloc_13[%97, %98] : memref<128x32xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg4)[%thread_id_x]
      %101 = memref.load %alloc_13[%97, %100] : memref<128x32xf16, 3>
      %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg4)[%thread_id_x]
      %103 = memref.load %alloc_13[%97, %102] : memref<128x32xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg4)[%thread_id_x]
      %105 = memref.load %alloc_13[%97, %104] : memref<128x32xf16, 3>
      %106 = vector.from_elements %99, %101, %103, %105 : vector<1x4xf16>
      %107 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %109 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %110 = vector.insert %109, %arg5 [%107, %108] : f16 into vector<2x8xf16>
      %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %112 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %113 = vector.insert %112, %110 [%107, %111] : f16 into vector<2x8xf16>
      %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %115 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %116 = vector.insert %115, %113 [%107, %114] : f16 into vector<2x8xf16>
      %117 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %118 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %119 = vector.insert %118, %116 [%107, %117] : f16 into vector<2x8xf16>
      %120 = vector.insert %8, %arg6 [%107, %108] : f16 into vector<2x8xf16>
      %121 = vector.insert %8, %120 [%107, %111] : f16 into vector<2x8xf16>
      %122 = vector.insert %8, %121 [%107, %114] : f16 into vector<2x8xf16>
      %123 = vector.insert %8, %122 [%107, %117] : f16 into vector<2x8xf16>
      %124 = arith.mulf %arg8, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %125 = vector.extract %124[0, 0] : f16 from vector<1x4xf16>
      %126 = vector.insert %125, %arg7 [%107, %108] : f16 into vector<2x8xf16>
      %127 = vector.extract %124[0, 1] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %126 [%107, %111] : f16 into vector<2x8xf16>
      %129 = vector.extract %124[0, 2] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%107, %114] : f16 into vector<2x8xf16>
      %131 = vector.extract %124[0, 3] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%107, %117] : f16 into vector<2x8xf16>
      affine.yield %119, %123, %132, %106 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %72 = arith.mulf %71#3, %25 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %73 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
    %74 = vector.insert %73, %71#2 [1, 4] : f16 into vector<2x8xf16>
    %75 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
    %76 = vector.insert %75, %74 [1, 5] : f16 into vector<2x8xf16>
    %77 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
    %78 = vector.insert %77, %76 [1, 6] : f16 into vector<2x8xf16>
    %79 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
    %80 = vector.insert %79, %78 [1, 7] : f16 into vector<2x8xf16>
    %81:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#3, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %97 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %125 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %126 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %127 = vector.insert %cst, %arg12 [%126, %125] : f32 into vector<2x32xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %129 = vector.insert %cst, %127 [%126, %128] : f32 into vector<2x32xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %131 = vector.insert %cst, %129 [%126, %130] : f32 into vector<2x32xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %133 = vector.insert %cst, %131 [%126, %132] : f32 into vector<2x32xf32>
        affine.yield %133 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %98 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
      %subview_16 = memref.subview %transpose[0, %98, 0, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_16 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<32x128xf16, 3>
      %99:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %126 = memref.load %alloc_15[%15, %125] : memref<32x128xf16, 3>
        %127 = memref.load %alloc_15[%17, %125] : memref<32x128xf16, 3>
        %128 = memref.load %alloc_15[%19, %125] : memref<32x128xf16, 3>
        %129 = memref.load %alloc_15[%21, %125] : memref<32x128xf16, 3>
        %130 = vector.from_elements %126, %127, %128, %129 : vector<4x1xf16>
        %131:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %130) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %162 = memref.load %alloc_15[%161, %125] : memref<32x128xf16, 3>
          %163 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %164 = memref.load %alloc_15[%163, %125] : memref<32x128xf16, 3>
          %165 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %166 = memref.load %alloc_15[%165, %125] : memref<32x128xf16, 3>
          %167 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %168 = memref.load %alloc_15[%167, %125] : memref<32x128xf16, 3>
          %169 = vector.from_elements %162, %164, %166, %168 : vector<4x1xf16>
          %170 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %171 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %172 = vector.extract %34[%170, %171] : f16 from vector<2x8xf16>
          %173 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %174 = vector.extract %34[%170, %173] : f16 from vector<2x8xf16>
          %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %176 = vector.extract %34[%170, %175] : f16 from vector<2x8xf16>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %178 = vector.extract %34[%170, %177] : f16 from vector<2x8xf16>
          %179 = vector.from_elements %172, %174, %176, %178 : vector<1x4xf16>
          %180 = frisk.warp_mma_rr %179, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %180, %169 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %132 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %133 = vector.extract %34[%132, 4] : f16 from vector<2x8xf16>
        %134 = vector.extract %34[%132, 5] : f16 from vector<2x8xf16>
        %135 = vector.extract %34[%132, 6] : f16 from vector<2x8xf16>
        %136 = vector.extract %34[%132, 7] : f16 from vector<2x8xf16>
        %137 = vector.from_elements %133, %134, %135, %136 : vector<1x4xf16>
        %138 = frisk.warp_mma_rr %137, %131#1, %131#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %140 = vector.extract %97[%132, %139] : f32 from vector<2x32xf32>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %142 = vector.extract %97[%132, %141] : f32 from vector<2x32xf32>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %144 = vector.extract %97[%132, %143] : f32 from vector<2x32xf32>
        %145 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %146 = vector.extract %97[%132, %145] : f32 from vector<2x32xf32>
        %147 = vector.from_elements %140, %142, %144, %146 : vector<1x4xf32>
        %148 = arith.addf %147, %138 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %149 = vector.extract %148[0, 0] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %arg12 [%132, %139] : f32 into vector<2x32xf32>
        %151 = vector.extract %148[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%132, %141] : f32 into vector<2x32xf32>
        %153 = vector.extract %148[0, 2] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %152 [%132, %143] : f32 into vector<2x32xf32>
        %155 = vector.extract %148[0, 3] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%132, %145] : f32 into vector<2x32xf32>
        %157 = vector.insert %149, %arg13 [%132, %139] : f32 into vector<2x32xf32>
        %158 = vector.insert %151, %157 [%132, %141] : f32 into vector<2x32xf32>
        %159 = vector.insert %153, %158 [%132, %143] : f32 into vector<2x32xf32>
        %160 = vector.insert %155, %159 [%132, %145] : f32 into vector<2x32xf32>
        affine.yield %156, %160 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %100 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 524288 + s1 * 16777216 + 131072) floordiv 524288) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_18 = memref.subview %transpose[0, %100, 32, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_18 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_17 : memref<32x128xf16, 3>
      %101:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %126 = memref.load %alloc_17[%15, %125] : memref<32x128xf16, 3>
        %127 = memref.load %alloc_17[%17, %125] : memref<32x128xf16, 3>
        %128 = memref.load %alloc_17[%19, %125] : memref<32x128xf16, 3>
        %129 = memref.load %alloc_17[%21, %125] : memref<32x128xf16, 3>
        %130 = vector.from_elements %126, %127, %128, %129 : vector<4x1xf16>
        %131:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %130) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %162 = memref.load %alloc_17[%161, %125] : memref<32x128xf16, 3>
          %163 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %164 = memref.load %alloc_17[%163, %125] : memref<32x128xf16, 3>
          %165 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %166 = memref.load %alloc_17[%165, %125] : memref<32x128xf16, 3>
          %167 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %168 = memref.load %alloc_17[%167, %125] : memref<32x128xf16, 3>
          %169 = vector.from_elements %162, %164, %166, %168 : vector<4x1xf16>
          %170 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %171 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %172 = vector.extract %50[%170, %171] : f16 from vector<2x8xf16>
          %173 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %174 = vector.extract %50[%170, %173] : f16 from vector<2x8xf16>
          %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %176 = vector.extract %50[%170, %175] : f16 from vector<2x8xf16>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %178 = vector.extract %50[%170, %177] : f16 from vector<2x8xf16>
          %179 = vector.from_elements %172, %174, %176, %178 : vector<1x4xf16>
          %180 = frisk.warp_mma_rr %179, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %180, %169 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %132 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %133 = vector.extract %50[%132, 4] : f16 from vector<2x8xf16>
        %134 = vector.extract %50[%132, 5] : f16 from vector<2x8xf16>
        %135 = vector.extract %50[%132, 6] : f16 from vector<2x8xf16>
        %136 = vector.extract %50[%132, 7] : f16 from vector<2x8xf16>
        %137 = vector.from_elements %133, %134, %135, %136 : vector<1x4xf16>
        %138 = frisk.warp_mma_rr %137, %131#1, %131#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %140 = vector.extract %99#1[%132, %139] : f32 from vector<2x32xf32>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %142 = vector.extract %99#1[%132, %141] : f32 from vector<2x32xf32>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %144 = vector.extract %99#1[%132, %143] : f32 from vector<2x32xf32>
        %145 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %146 = vector.extract %99#1[%132, %145] : f32 from vector<2x32xf32>
        %147 = vector.from_elements %140, %142, %144, %146 : vector<1x4xf32>
        %148 = arith.addf %147, %138 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %149 = vector.extract %148[0, 0] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %arg12 [%132, %139] : f32 into vector<2x32xf32>
        %151 = vector.extract %148[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%132, %141] : f32 into vector<2x32xf32>
        %153 = vector.extract %148[0, 2] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %152 [%132, %143] : f32 into vector<2x32xf32>
        %155 = vector.extract %148[0, 3] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%132, %145] : f32 into vector<2x32xf32>
        %157 = vector.insert %149, %arg13 [%132, %139] : f32 into vector<2x32xf32>
        %158 = vector.insert %151, %157 [%132, %141] : f32 into vector<2x32xf32>
        %159 = vector.insert %153, %158 [%132, %143] : f32 into vector<2x32xf32>
        %160 = vector.insert %155, %159 [%132, %145] : f32 into vector<2x32xf32>
        affine.yield %156, %160 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %102 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 524288 + s1 * 16777216 + 262144) floordiv 524288) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_20 = memref.subview %transpose[0, %102, 64, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_20 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_19 : memref<32x128xf16, 3>
      %103:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %126 = memref.load %alloc_19[%15, %125] : memref<32x128xf16, 3>
        %127 = memref.load %alloc_19[%17, %125] : memref<32x128xf16, 3>
        %128 = memref.load %alloc_19[%19, %125] : memref<32x128xf16, 3>
        %129 = memref.load %alloc_19[%21, %125] : memref<32x128xf16, 3>
        %130 = vector.from_elements %126, %127, %128, %129 : vector<4x1xf16>
        %131:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %130) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %162 = memref.load %alloc_19[%161, %125] : memref<32x128xf16, 3>
          %163 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %164 = memref.load %alloc_19[%163, %125] : memref<32x128xf16, 3>
          %165 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %166 = memref.load %alloc_19[%165, %125] : memref<32x128xf16, 3>
          %167 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %168 = memref.load %alloc_19[%167, %125] : memref<32x128xf16, 3>
          %169 = vector.from_elements %162, %164, %166, %168 : vector<4x1xf16>
          %170 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %171 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %172 = vector.extract %65[%170, %171] : f16 from vector<2x8xf16>
          %173 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %174 = vector.extract %65[%170, %173] : f16 from vector<2x8xf16>
          %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %176 = vector.extract %65[%170, %175] : f16 from vector<2x8xf16>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %178 = vector.extract %65[%170, %177] : f16 from vector<2x8xf16>
          %179 = vector.from_elements %172, %174, %176, %178 : vector<1x4xf16>
          %180 = frisk.warp_mma_rr %179, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %180, %169 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %132 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %133 = vector.extract %65[%132, 4] : f16 from vector<2x8xf16>
        %134 = vector.extract %65[%132, 5] : f16 from vector<2x8xf16>
        %135 = vector.extract %65[%132, 6] : f16 from vector<2x8xf16>
        %136 = vector.extract %65[%132, 7] : f16 from vector<2x8xf16>
        %137 = vector.from_elements %133, %134, %135, %136 : vector<1x4xf16>
        %138 = frisk.warp_mma_rr %137, %131#1, %131#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %140 = vector.extract %101#1[%132, %139] : f32 from vector<2x32xf32>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %142 = vector.extract %101#1[%132, %141] : f32 from vector<2x32xf32>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %144 = vector.extract %101#1[%132, %143] : f32 from vector<2x32xf32>
        %145 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %146 = vector.extract %101#1[%132, %145] : f32 from vector<2x32xf32>
        %147 = vector.from_elements %140, %142, %144, %146 : vector<1x4xf32>
        %148 = arith.addf %147, %138 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %149 = vector.extract %148[0, 0] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %arg12 [%132, %139] : f32 into vector<2x32xf32>
        %151 = vector.extract %148[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%132, %141] : f32 into vector<2x32xf32>
        %153 = vector.extract %148[0, 2] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %152 [%132, %143] : f32 into vector<2x32xf32>
        %155 = vector.extract %148[0, 3] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%132, %145] : f32 into vector<2x32xf32>
        %157 = vector.insert %149, %arg13 [%132, %139] : f32 into vector<2x32xf32>
        %158 = vector.insert %151, %157 [%132, %141] : f32 into vector<2x32xf32>
        %159 = vector.insert %153, %158 [%132, %143] : f32 into vector<2x32xf32>
        %160 = vector.insert %155, %159 [%132, %145] : f32 into vector<2x32xf32>
        affine.yield %156, %160 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %104 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 524288 + s1 * 16777216 + 393216) floordiv 524288) mod 32)>(%arg4)[%7, %block_id_z]
      %subview_22 = memref.subview %transpose[0, %104, 96, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_22 to %alloc_21 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_21 : memref<32x128xf16, 3>
      %105:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg11)[%thread_id_x]
        %126 = memref.load %alloc_21[%15, %125] : memref<32x128xf16, 3>
        %127 = memref.load %alloc_21[%17, %125] : memref<32x128xf16, 3>
        %128 = memref.load %alloc_21[%19, %125] : memref<32x128xf16, 3>
        %129 = memref.load %alloc_21[%21, %125] : memref<32x128xf16, 3>
        %130 = vector.from_elements %126, %127, %128, %129 : vector<4x1xf16>
        %131:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3, %arg16 = %130) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %161 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>(%arg14)[%thread_id_x]
          %162 = memref.load %alloc_21[%161, %125] : memref<32x128xf16, 3>
          %163 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>(%arg14)[%thread_id_x]
          %164 = memref.load %alloc_21[%163, %125] : memref<32x128xf16, 3>
          %165 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>(%arg14)[%thread_id_x]
          %166 = memref.load %alloc_21[%165, %125] : memref<32x128xf16, 3>
          %167 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>(%arg14)[%thread_id_x]
          %168 = memref.load %alloc_21[%167, %125] : memref<32x128xf16, 3>
          %169 = vector.from_elements %162, %164, %166, %168 : vector<4x1xf16>
          %170 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %171 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %172 = vector.extract %80[%170, %171] : f16 from vector<2x8xf16>
          %173 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %174 = vector.extract %80[%170, %173] : f16 from vector<2x8xf16>
          %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %176 = vector.extract %80[%170, %175] : f16 from vector<2x8xf16>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %178 = vector.extract %80[%170, %177] : f16 from vector<2x8xf16>
          %179 = vector.from_elements %172, %174, %176, %178 : vector<1x4xf16>
          %180 = frisk.warp_mma_rr %179, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %180, %169 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %132 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %133 = vector.extract %80[%132, 4] : f16 from vector<2x8xf16>
        %134 = vector.extract %80[%132, 5] : f16 from vector<2x8xf16>
        %135 = vector.extract %80[%132, 6] : f16 from vector<2x8xf16>
        %136 = vector.extract %80[%132, 7] : f16 from vector<2x8xf16>
        %137 = vector.from_elements %133, %134, %135, %136 : vector<1x4xf16>
        %138 = frisk.warp_mma_rr %137, %131#1, %131#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %139 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %140 = vector.extract %103#1[%132, %139] : f32 from vector<2x32xf32>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %142 = vector.extract %103#1[%132, %141] : f32 from vector<2x32xf32>
        %143 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %144 = vector.extract %103#1[%132, %143] : f32 from vector<2x32xf32>
        %145 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %146 = vector.extract %103#1[%132, %145] : f32 from vector<2x32xf32>
        %147 = vector.from_elements %140, %142, %144, %146 : vector<1x4xf32>
        %148 = arith.addf %147, %138 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %149 = vector.extract %148[0, 0] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %arg12 [%132, %139] : f32 into vector<2x32xf32>
        %151 = vector.extract %148[0, 1] : f32 from vector<1x4xf32>
        %152 = vector.insert %151, %150 [%132, %141] : f32 into vector<2x32xf32>
        %153 = vector.extract %148[0, 2] : f32 from vector<1x4xf32>
        %154 = vector.insert %153, %152 [%132, %143] : f32 into vector<2x32xf32>
        %155 = vector.extract %148[0, 3] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %154 [%132, %145] : f32 into vector<2x32xf32>
        %157 = vector.insert %149, %arg13 [%132, %139] : f32 into vector<2x32xf32>
        %158 = vector.insert %151, %157 [%132, %141] : f32 into vector<2x32xf32>
        %159 = vector.insert %153, %158 [%132, %143] : f32 into vector<2x32xf32>
        %160 = vector.insert %155, %159 [%132, %145] : f32 into vector<2x32xf32>
        affine.yield %156, %160 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %106 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4) -> (vector<2x32xf32>) {
        %125 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %127 = vector.extract %105#1[%125, %126] : f32 from vector<2x32xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %129 = vector.extract %105#1[%125, %128] : f32 from vector<2x32xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %131 = vector.extract %105#1[%125, %130] : f32 from vector<2x32xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %133 = vector.extract %105#1[%125, %132] : f32 from vector<2x32xf32>
        %134 = vector.insert %127, %arg12 [%125, %126] : f32 into vector<2x32xf32>
        %135 = vector.insert %129, %134 [%125, %128] : f32 into vector<2x32xf32>
        %136 = vector.insert %131, %135 [%125, %130] : f32 into vector<2x32xf32>
        %137 = vector.insert %133, %136 [%125, %132] : f32 into vector<2x32xf32>
        affine.yield %137 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %107 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %125 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_7) -> (f32) {
          %129 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %130 = vector.extract %106[%arg11, %129] : f32 from vector<2x32xf32>
          %131 = arith.maxnumf %arg14, %130 : f32
          %132 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %133 = vector.extract %106[%arg11, %132] : f32 from vector<2x32xf32>
          %134 = arith.maxnumf %131, %133 : f32
          %135 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %136 = vector.extract %106[%arg11, %135] : f32 from vector<2x32xf32>
          %137 = arith.maxnumf %134, %136 : f32
          %138 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %139 = vector.extract %106[%arg11, %138] : f32 from vector<2x32xf32>
          %140 = arith.maxnumf %137, %139 : f32
          affine.yield %140 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %125, %c16_i32, %c64_i32 : f32
        %126 = arith.maxnumf %125, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %126, %c32_i32, %c64_i32 : f32
        %127 = arith.maxnumf %126, %shuffleResult_31 : f32
        %128 = vector.insert %127, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %128 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %108 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %125 = vector.extract %107[%arg11, 0] : f32 from vector<2x1xf32>
        %126 = vector.broadcast %125 : f32 to vector<1x1xf32>
        %127 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %128 = vector.broadcast %127 : f32 to vector<1x1xf32>
        %129 = arith.maxnumf %126, %128 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %130 = vector.extract %129[0, 0] : f32 from vector<1x1xf32>
        %131 = vector.insert %130, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %131 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %109:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %125 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %126 = vector.extract %108[%125, 0] : f32 from vector<2x1xf32>
        %127 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_3) -> (vector<1x4xf32>) {
          %154 = vector.broadcast %126 : f32 to vector<1x4xf32>
          affine.yield %154 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %129 = vector.extract %127[0, 0] : f32 from vector<1x4xf32>
        %130 = vector.insert %129, %arg12 [%125, %128] : f32 into vector<2x32xf32>
        %131 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %132 = vector.extract %127[0, 1] : f32 from vector<1x4xf32>
        %133 = vector.insert %132, %130 [%125, %131] : f32 into vector<2x32xf32>
        %134 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %135 = vector.extract %127[0, 2] : f32 from vector<1x4xf32>
        %136 = vector.insert %135, %133 [%125, %134] : f32 into vector<2x32xf32>
        %137 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %138 = vector.extract %127[0, 3] : f32 from vector<1x4xf32>
        %139 = vector.insert %138, %136 [%125, %137] : f32 into vector<2x32xf32>
        %140 = vector.extract %105#1[%125, %128] : f32 from vector<2x32xf32>
        %141 = vector.extract %105#1[%125, %131] : f32 from vector<2x32xf32>
        %142 = vector.extract %105#1[%125, %134] : f32 from vector<2x32xf32>
        %143 = vector.extract %105#1[%125, %137] : f32 from vector<2x32xf32>
        %144 = vector.from_elements %140, %141, %142, %143 : vector<1x4xf32>
        %145 = arith.subf %144, %127 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %146 = vector.extract %145[0, 0] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %arg13 [%125, %128] : f32 into vector<2x32xf32>
        %148 = vector.extract %145[0, 1] : f32 from vector<1x4xf32>
        %149 = vector.insert %148, %147 [%125, %131] : f32 into vector<2x32xf32>
        %150 = vector.extract %145[0, 2] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %149 [%125, %134] : f32 into vector<2x32xf32>
        %152 = vector.extract %145[0, 3] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %151 [%125, %137] : f32 into vector<2x32xf32>
        affine.yield %139, %153 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %110:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6, %arg14 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %125 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %126 = vector.broadcast %125 : f32 to vector<1x1xf32>
        %127 = vector.extract %108[%arg11, 0] : f32 from vector<2x1xf32>
        %128 = vector.broadcast %127 : f32 to vector<1x1xf32>
        %129 = arith.subf %126, %128 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %130 = vector.extract %129[0, 0] : f32 from vector<1x1xf32>
        %131 = vector.insert %130, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %132 = math.exp2 %129 {tiled = true} : vector<1x1xf32>
        %133 = vector.extract %132[0, 0] : f32 from vector<1x1xf32>
        %134 = vector.insert %133, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %135 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %136 = vector.broadcast %135 : f32 to vector<1x1xf32>
        %137 = arith.mulf %136, %132 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %138 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %139 = vector.insert %138, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %131, %134, %139 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %111:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_4, %arg13 = %cst_4) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %125 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %127 = vector.extract %109#1[%125, %126] : f32 from vector<2x32xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %129 = vector.extract %109#1[%125, %128] : f32 from vector<2x32xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %131 = vector.extract %109#1[%125, %130] : f32 from vector<2x32xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %133 = vector.extract %109#1[%125, %132] : f32 from vector<2x32xf32>
        %134 = vector.from_elements %127, %129, %131, %133 : vector<1x4xf32>
        %135 = math.exp2 %134 {tiled = true} : vector<1x4xf32>
        %136 = vector.extract %135[0, 0] : f32 from vector<1x4xf32>
        %137 = vector.insert %136, %arg12 [%125, %126] : f32 into vector<2x32xf32>
        %138 = vector.extract %135[0, 1] : f32 from vector<1x4xf32>
        %139 = vector.insert %138, %137 [%125, %128] : f32 into vector<2x32xf32>
        %140 = vector.extract %135[0, 2] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %139 [%125, %130] : f32 into vector<2x32xf32>
        %142 = vector.extract %135[0, 3] : f32 from vector<1x4xf32>
        %143 = vector.insert %142, %141 [%125, %132] : f32 into vector<2x32xf32>
        %144 = vector.insert %136, %arg13 [%125, %126] : f32 into vector<2x32xf32>
        %145 = vector.insert %138, %144 [%125, %128] : f32 into vector<2x32xf32>
        %146 = vector.insert %140, %145 [%125, %130] : f32 into vector<2x32xf32>
        %147 = vector.insert %142, %146 [%125, %132] : f32 into vector<2x32xf32>
        affine.yield %143, %147 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %112 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %125 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst) -> (f32) {
          %129 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %130 = vector.extract %111#1[%arg11, %129] : f32 from vector<2x32xf32>
          %131 = arith.addf %arg14, %130 : f32
          %132 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %133 = vector.extract %111#1[%arg11, %132] : f32 from vector<2x32xf32>
          %134 = arith.addf %131, %133 : f32
          %135 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %136 = vector.extract %111#1[%arg11, %135] : f32 from vector<2x32xf32>
          %137 = arith.addf %134, %136 : f32
          %138 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %139 = vector.extract %111#1[%arg11, %138] : f32 from vector<2x32xf32>
          %140 = arith.addf %137, %139 : f32
          affine.yield %140 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %125, %c16_i32, %c64_i32 : f32
        %126 = arith.addf %125, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %126, %c32_i32, %c64_i32 : f32
        %127 = arith.addf %126, %shuffleResult_31 : f32
        %128 = vector.insert %127, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %128 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %113 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6) -> (vector<2x1xf32>) {
        %125 = vector.extract %112[%arg11, 0] : f32 from vector<2x1xf32>
        %126 = vector.broadcast %125 : f32 to vector<1x1xf32>
        %127 = vector.extract %110#2[%arg11, 0] : f32 from vector<2x1xf32>
        %128 = vector.broadcast %127 : f32 to vector<1x1xf32>
        %129 = arith.addf %126, %128 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %130 = vector.extract %129[0, 0] : f32 from vector<1x1xf32>
        %131 = vector.insert %130, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %131 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %114 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %125 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %127 = vector.extract %111#0[%125, %126] : f32 from vector<2x32xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %129 = vector.extract %111#0[%125, %128] : f32 from vector<2x32xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %131 = vector.extract %111#0[%125, %130] : f32 from vector<2x32xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %133 = vector.extract %111#0[%125, %132] : f32 from vector<2x32xf32>
        %134 = vector.from_elements %127, %129, %131, %133 : vector<1x4xf32>
        %135 = arith.truncf %134 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %136 = vector.extract %135[0, 0] : f16 from vector<1x4xf16>
        %137 = vector.insert %136, %arg12 [%125, %126] : f16 into vector<2x32xf16>
        %138 = vector.extract %135[0, 1] : f16 from vector<1x4xf16>
        %139 = vector.insert %138, %137 [%125, %128] : f16 into vector<2x32xf16>
        %140 = vector.extract %135[0, 2] : f16 from vector<1x4xf16>
        %141 = vector.insert %140, %139 [%125, %130] : f16 into vector<2x32xf16>
        %142 = vector.extract %135[0, 3] : f16 from vector<1x4xf16>
        %143 = vector.insert %142, %141 [%125, %132] : f16 into vector<2x32xf16>
        affine.yield %143 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_24 = memref.subview %arg2[0, %98, %arg4, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_24 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_23 : memref<128x32xf16, 3>
      %115 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %126 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %127 = frisk.read_matrix_from_shm %alloc_23[%125, %126] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %128 = vector.bitcast %127 : vector<4xi32> to vector<8xf16>
        %129 = vector.extract %128[0] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %131 = vector.extract %128[%130] : f16 from vector<8xf16>
        %132 = vector.extract %128[1] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %134 = vector.extract %128[%133] : f16 from vector<8xf16>
        %135 = vector.extract %128[2] : f16 from vector<8xf16>
        %136 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %137 = vector.extract %128[%136] : f16 from vector<8xf16>
        %138 = vector.extract %128[3] : f16 from vector<8xf16>
        %139 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %140 = vector.extract %128[%139] : f16 from vector<8xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %142 = vector.insert %129, %arg12 [%141, 0] : f16 into vector<32x2xf16>
        %143 = vector.insert %131, %142 [%141, 1] : f16 into vector<32x2xf16>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %145 = vector.insert %132, %143 [%144, 0] : f16 into vector<32x2xf16>
        %146 = vector.insert %134, %145 [%144, 1] : f16 into vector<32x2xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %148 = vector.insert %135, %146 [%147, 0] : f16 into vector<32x2xf16>
        %149 = vector.insert %137, %148 [%147, 1] : f16 into vector<32x2xf16>
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %151 = vector.insert %138, %149 [%150, 0] : f16 into vector<32x2xf16>
        %152 = vector.insert %140, %151 [%150, 1] : f16 into vector<32x2xf16>
        affine.yield %152 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %116:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %125 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %173 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %174 = vector.extract %114[%172, %173] : f16 from vector<2x32xf16>
          %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %176 = vector.extract %114[%172, %175] : f16 from vector<2x32xf16>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %178 = vector.extract %114[%172, %177] : f16 from vector<2x32xf16>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %180 = vector.extract %114[%172, %179] : f16 from vector<2x32xf16>
          %181 = vector.from_elements %174, %176, %178, %180 : vector<1x4xf16>
          %182 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %183 = vector.extract %115[%173, %182] : f16 from vector<32x2xf16>
          %184 = vector.extract %115[%175, %182] : f16 from vector<32x2xf16>
          %185 = vector.extract %115[%177, %182] : f16 from vector<32x2xf16>
          %186 = vector.extract %115[%179, %182] : f16 from vector<32x2xf16>
          %187 = vector.from_elements %183, %184, %185, %186 : vector<4x1xf16>
          %188 = frisk.warp_mma_rr %181, %187, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %188 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %126 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %128 = vector.extract %125[0, 0] : f32 from vector<1x4xf32>
        %129 = vector.insert %128, %arg12 [%126, %127] : f32 into vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %131 = vector.extract %125[0, 1] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %129 [%126, %130] : f32 into vector<2x8xf32>
        %133 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %134 = vector.extract %125[0, 2] : f32 from vector<1x4xf32>
        %135 = vector.insert %134, %132 [%126, %133] : f32 into vector<2x8xf32>
        %136 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %137 = vector.extract %125[0, 3] : f32 from vector<1x4xf32>
        %138 = vector.insert %137, %135 [%126, %136] : f32 into vector<2x8xf32>
        %139 = vector.extract %110#1[%126, 0] : f32 from vector<2x1xf32>
        %140 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %172 = vector.broadcast %139 : f32 to vector<1x4xf32>
          affine.yield %172 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %141 = vector.extract %140[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg13 [%126, %127] : f32 into vector<2x8xf32>
        %143 = vector.extract %140[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %142 [%126, %130] : f32 into vector<2x8xf32>
        %145 = vector.extract %140[0, 2] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %144 [%126, %133] : f32 into vector<2x8xf32>
        %147 = vector.extract %140[0, 3] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%126, %136] : f32 into vector<2x8xf32>
        %149 = vector.extract %arg5[%126, %127] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg5[%126, %130] : f32 from vector<2x8xf32>
        %151 = vector.extract %arg5[%126, %133] : f32 from vector<2x8xf32>
        %152 = vector.extract %arg5[%126, %136] : f32 from vector<2x8xf32>
        %153 = vector.from_elements %149, %150, %151, %152 : vector<1x4xf32>
        %154 = arith.mulf %153, %140 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %155 = vector.extract %154[0, 0] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %arg14 [%126, %127] : f32 into vector<2x8xf32>
        %157 = vector.extract %154[0, 1] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%126, %130] : f32 into vector<2x8xf32>
        %159 = vector.extract %154[0, 2] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%126, %133] : f32 into vector<2x8xf32>
        %161 = vector.extract %154[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %160 [%126, %136] : f32 into vector<2x8xf32>
        %163 = arith.addf %154, %125 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg15 [%126, %127] : f32 into vector<2x8xf32>
        %166 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%126, %130] : f32 into vector<2x8xf32>
        %168 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%126, %133] : f32 into vector<2x8xf32>
        %170 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%126, %136] : f32 into vector<2x8xf32>
        affine.yield %138, %148, %162, %171 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_25 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_26 = memref.subview %arg2[0, %98, %arg4, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_26 to %alloc_25 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_25 : memref<128x32xf16, 3>
      %117 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %126 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %127 = frisk.read_matrix_from_shm %alloc_25[%125, %126] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %128 = vector.bitcast %127 : vector<4xi32> to vector<8xf16>
        %129 = vector.extract %128[0] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %131 = vector.extract %128[%130] : f16 from vector<8xf16>
        %132 = vector.extract %128[1] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %134 = vector.extract %128[%133] : f16 from vector<8xf16>
        %135 = vector.extract %128[2] : f16 from vector<8xf16>
        %136 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %137 = vector.extract %128[%136] : f16 from vector<8xf16>
        %138 = vector.extract %128[3] : f16 from vector<8xf16>
        %139 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %140 = vector.extract %128[%139] : f16 from vector<8xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %142 = vector.insert %129, %arg12 [%141, 0] : f16 into vector<32x2xf16>
        %143 = vector.insert %131, %142 [%141, 1] : f16 into vector<32x2xf16>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %145 = vector.insert %132, %143 [%144, 0] : f16 into vector<32x2xf16>
        %146 = vector.insert %134, %145 [%144, 1] : f16 into vector<32x2xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %148 = vector.insert %135, %146 [%147, 0] : f16 into vector<32x2xf16>
        %149 = vector.insert %137, %148 [%147, 1] : f16 into vector<32x2xf16>
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %151 = vector.insert %138, %149 [%150, 0] : f16 into vector<32x2xf16>
        %152 = vector.insert %140, %151 [%150, 1] : f16 into vector<32x2xf16>
        affine.yield %152 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %118:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %125 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %173 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %174 = vector.extract %114[%172, %173] : f16 from vector<2x32xf16>
          %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %176 = vector.extract %114[%172, %175] : f16 from vector<2x32xf16>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %178 = vector.extract %114[%172, %177] : f16 from vector<2x32xf16>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %180 = vector.extract %114[%172, %179] : f16 from vector<2x32xf16>
          %181 = vector.from_elements %174, %176, %178, %180 : vector<1x4xf16>
          %182 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %183 = vector.extract %117[%173, %182] : f16 from vector<32x2xf16>
          %184 = vector.extract %117[%175, %182] : f16 from vector<32x2xf16>
          %185 = vector.extract %117[%177, %182] : f16 from vector<32x2xf16>
          %186 = vector.extract %117[%179, %182] : f16 from vector<32x2xf16>
          %187 = vector.from_elements %183, %184, %185, %186 : vector<4x1xf16>
          %188 = frisk.warp_mma_rr %181, %187, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %188 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %126 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %128 = vector.extract %125[0, 0] : f32 from vector<1x4xf32>
        %129 = vector.insert %128, %arg12 [%126, %127] : f32 into vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %131 = vector.extract %125[0, 1] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %129 [%126, %130] : f32 into vector<2x8xf32>
        %133 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %134 = vector.extract %125[0, 2] : f32 from vector<1x4xf32>
        %135 = vector.insert %134, %132 [%126, %133] : f32 into vector<2x8xf32>
        %136 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %137 = vector.extract %125[0, 3] : f32 from vector<1x4xf32>
        %138 = vector.insert %137, %135 [%126, %136] : f32 into vector<2x8xf32>
        %139 = vector.extract %110#1[%126, 0] : f32 from vector<2x1xf32>
        %140 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %172 = vector.broadcast %139 : f32 to vector<1x4xf32>
          affine.yield %172 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %141 = vector.extract %140[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg13 [%126, %127] : f32 into vector<2x8xf32>
        %143 = vector.extract %140[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %142 [%126, %130] : f32 into vector<2x8xf32>
        %145 = vector.extract %140[0, 2] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %144 [%126, %133] : f32 into vector<2x8xf32>
        %147 = vector.extract %140[0, 3] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%126, %136] : f32 into vector<2x8xf32>
        %149 = vector.extract %arg8[%126, %127] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg8[%126, %130] : f32 from vector<2x8xf32>
        %151 = vector.extract %arg8[%126, %133] : f32 from vector<2x8xf32>
        %152 = vector.extract %arg8[%126, %136] : f32 from vector<2x8xf32>
        %153 = vector.from_elements %149, %150, %151, %152 : vector<1x4xf32>
        %154 = arith.mulf %153, %140 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %155 = vector.extract %154[0, 0] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %arg14 [%126, %127] : f32 into vector<2x8xf32>
        %157 = vector.extract %154[0, 1] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%126, %130] : f32 into vector<2x8xf32>
        %159 = vector.extract %154[0, 2] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%126, %133] : f32 into vector<2x8xf32>
        %161 = vector.extract %154[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %160 [%126, %136] : f32 into vector<2x8xf32>
        %163 = arith.addf %154, %125 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg15 [%126, %127] : f32 into vector<2x8xf32>
        %166 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%126, %130] : f32 into vector<2x8xf32>
        %168 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%126, %133] : f32 into vector<2x8xf32>
        %170 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%126, %136] : f32 into vector<2x8xf32>
        affine.yield %138, %148, %162, %171 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_27 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_28 = memref.subview %arg2[0, %98, %arg4, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_28 to %alloc_27 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_27 : memref<128x32xf16, 3>
      %119 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %126 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %127 = frisk.read_matrix_from_shm %alloc_27[%125, %126] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %128 = vector.bitcast %127 : vector<4xi32> to vector<8xf16>
        %129 = vector.extract %128[0] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %131 = vector.extract %128[%130] : f16 from vector<8xf16>
        %132 = vector.extract %128[1] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %134 = vector.extract %128[%133] : f16 from vector<8xf16>
        %135 = vector.extract %128[2] : f16 from vector<8xf16>
        %136 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %137 = vector.extract %128[%136] : f16 from vector<8xf16>
        %138 = vector.extract %128[3] : f16 from vector<8xf16>
        %139 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %140 = vector.extract %128[%139] : f16 from vector<8xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %142 = vector.insert %129, %arg12 [%141, 0] : f16 into vector<32x2xf16>
        %143 = vector.insert %131, %142 [%141, 1] : f16 into vector<32x2xf16>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %145 = vector.insert %132, %143 [%144, 0] : f16 into vector<32x2xf16>
        %146 = vector.insert %134, %145 [%144, 1] : f16 into vector<32x2xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %148 = vector.insert %135, %146 [%147, 0] : f16 into vector<32x2xf16>
        %149 = vector.insert %137, %148 [%147, 1] : f16 into vector<32x2xf16>
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %151 = vector.insert %138, %149 [%150, 0] : f16 into vector<32x2xf16>
        %152 = vector.insert %140, %151 [%150, 1] : f16 into vector<32x2xf16>
        affine.yield %152 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %120:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %125 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %173 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %174 = vector.extract %114[%172, %173] : f16 from vector<2x32xf16>
          %175 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %176 = vector.extract %114[%172, %175] : f16 from vector<2x32xf16>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %178 = vector.extract %114[%172, %177] : f16 from vector<2x32xf16>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %180 = vector.extract %114[%172, %179] : f16 from vector<2x32xf16>
          %181 = vector.from_elements %174, %176, %178, %180 : vector<1x4xf16>
          %182 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %183 = vector.extract %119[%173, %182] : f16 from vector<32x2xf16>
          %184 = vector.extract %119[%175, %182] : f16 from vector<32x2xf16>
          %185 = vector.extract %119[%177, %182] : f16 from vector<32x2xf16>
          %186 = vector.extract %119[%179, %182] : f16 from vector<32x2xf16>
          %187 = vector.from_elements %183, %184, %185, %186 : vector<4x1xf16>
          %188 = frisk.warp_mma_rr %181, %187, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %188 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %126 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %128 = vector.extract %125[0, 0] : f32 from vector<1x4xf32>
        %129 = vector.insert %128, %arg12 [%126, %127] : f32 into vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %131 = vector.extract %125[0, 1] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %129 [%126, %130] : f32 into vector<2x8xf32>
        %133 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %134 = vector.extract %125[0, 2] : f32 from vector<1x4xf32>
        %135 = vector.insert %134, %132 [%126, %133] : f32 into vector<2x8xf32>
        %136 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %137 = vector.extract %125[0, 3] : f32 from vector<1x4xf32>
        %138 = vector.insert %137, %135 [%126, %136] : f32 into vector<2x8xf32>
        %139 = vector.extract %110#1[%126, 0] : f32 from vector<2x1xf32>
        %140 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_3) -> (vector<1x4xf32>) {
          %172 = vector.broadcast %139 : f32 to vector<1x4xf32>
          affine.yield %172 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %141 = vector.extract %140[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg13 [%126, %127] : f32 into vector<2x8xf32>
        %143 = vector.extract %140[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %142 [%126, %130] : f32 into vector<2x8xf32>
        %145 = vector.extract %140[0, 2] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %144 [%126, %133] : f32 into vector<2x8xf32>
        %147 = vector.extract %140[0, 3] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%126, %136] : f32 into vector<2x8xf32>
        %149 = vector.extract %arg9[%126, %127] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg9[%126, %130] : f32 from vector<2x8xf32>
        %151 = vector.extract %arg9[%126, %133] : f32 from vector<2x8xf32>
        %152 = vector.extract %arg9[%126, %136] : f32 from vector<2x8xf32>
        %153 = vector.from_elements %149, %150, %151, %152 : vector<1x4xf32>
        %154 = arith.mulf %153, %140 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %155 = vector.extract %154[0, 0] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %arg14 [%126, %127] : f32 into vector<2x8xf32>
        %157 = vector.extract %154[0, 1] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%126, %130] : f32 into vector<2x8xf32>
        %159 = vector.extract %154[0, 2] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%126, %133] : f32 into vector<2x8xf32>
        %161 = vector.extract %154[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %160 [%126, %136] : f32 into vector<2x8xf32>
        %163 = arith.addf %154, %125 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg15 [%126, %127] : f32 into vector<2x8xf32>
        %166 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%126, %130] : f32 into vector<2x8xf32>
        %168 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%126, %133] : f32 into vector<2x8xf32>
        %170 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%126, %136] : f32 into vector<2x8xf32>
        affine.yield %138, %148, %162, %171 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_29 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_30 = memref.subview %arg2[0, %98, %arg4, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_30 to %alloc_29 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_29 : memref<128x32xf16, 3>
      %121 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %125 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x]
        %126 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x]
        %127 = frisk.read_matrix_from_shm %alloc_29[%125, %126] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %128 = vector.bitcast %127 : vector<4xi32> to vector<8xf16>
        %129 = vector.extract %128[0] : f16 from vector<8xf16>
        %130 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
        %131 = vector.extract %128[%130] : f16 from vector<8xf16>
        %132 = vector.extract %128[1] : f16 from vector<8xf16>
        %133 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
        %134 = vector.extract %128[%133] : f16 from vector<8xf16>
        %135 = vector.extract %128[2] : f16 from vector<8xf16>
        %136 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
        %137 = vector.extract %128[%136] : f16 from vector<8xf16>
        %138 = vector.extract %128[3] : f16 from vector<8xf16>
        %139 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
        %140 = vector.extract %128[%139] : f16 from vector<8xf16>
        %141 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %142 = vector.insert %129, %arg12 [%141, 0] : f16 into vector<32x2xf16>
        %143 = vector.insert %131, %142 [%141, 1] : f16 into vector<32x2xf16>
        %144 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %145 = vector.insert %132, %143 [%144, 0] : f16 into vector<32x2xf16>
        %146 = vector.insert %134, %145 [%144, 1] : f16 into vector<32x2xf16>
        %147 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %148 = vector.insert %135, %146 [%147, 0] : f16 into vector<32x2xf16>
        %149 = vector.insert %137, %148 [%147, 1] : f16 into vector<32x2xf16>
        %150 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %151 = vector.insert %138, %149 [%150, 0] : f16 into vector<32x2xf16>
        %152 = vector.insert %140, %151 [%150, 1] : f16 into vector<32x2xf16>
        affine.yield %152 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %122:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %125 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %180 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %181 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %182 = vector.extract %114[%180, %181] : f16 from vector<2x32xf16>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %184 = vector.extract %114[%180, %183] : f16 from vector<2x32xf16>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %186 = vector.extract %114[%180, %185] : f16 from vector<2x32xf16>
          %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %188 = vector.extract %114[%180, %187] : f16 from vector<2x32xf16>
          %189 = vector.from_elements %182, %184, %186, %188 : vector<1x4xf16>
          %190 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %191 = vector.extract %121[%181, %190] : f16 from vector<32x2xf16>
          %192 = vector.extract %121[%183, %190] : f16 from vector<32x2xf16>
          %193 = vector.extract %121[%185, %190] : f16 from vector<32x2xf16>
          %194 = vector.extract %121[%187, %190] : f16 from vector<32x2xf16>
          %195 = vector.from_elements %191, %192, %193, %194 : vector<4x1xf16>
          %196 = frisk.warp_mma_rr %189, %195, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %196 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %126 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %128 = vector.extract %125[0, 0] : f32 from vector<1x4xf32>
        %129 = vector.insert %128, %arg12 [%126, %127] : f32 into vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %131 = vector.extract %125[0, 1] : f32 from vector<1x4xf32>
        %132 = vector.insert %131, %129 [%126, %130] : f32 into vector<2x8xf32>
        %133 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %134 = vector.extract %125[0, 2] : f32 from vector<1x4xf32>
        %135 = vector.insert %134, %132 [%126, %133] : f32 into vector<2x8xf32>
        %136 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %137 = vector.extract %125[0, 3] : f32 from vector<1x4xf32>
        %138 = vector.insert %137, %135 [%126, %136] : f32 into vector<2x8xf32>
        %139 = vector.extract %110#1[%126, 0] : f32 from vector<2x1xf32>
        %140 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_3) -> (vector<1x4xf32>) {
          %180 = vector.broadcast %139 : f32 to vector<1x4xf32>
          affine.yield %180 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %141 = vector.extract %140[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg13 [%126, %127] : f32 into vector<2x8xf32>
        %143 = vector.extract %140[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %142 [%126, %130] : f32 into vector<2x8xf32>
        %145 = vector.extract %140[0, 2] : f32 from vector<1x4xf32>
        %146 = vector.insert %145, %144 [%126, %133] : f32 into vector<2x8xf32>
        %147 = vector.extract %140[0, 3] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %146 [%126, %136] : f32 into vector<2x8xf32>
        %149 = vector.extract %arg10[%126, %127] : f32 from vector<2x8xf32>
        %150 = vector.extract %arg10[%126, %130] : f32 from vector<2x8xf32>
        %151 = vector.extract %arg10[%126, %133] : f32 from vector<2x8xf32>
        %152 = vector.extract %arg10[%126, %136] : f32 from vector<2x8xf32>
        %153 = vector.from_elements %149, %150, %151, %152 : vector<1x4xf32>
        %154 = arith.mulf %153, %140 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %155 = vector.extract %154[0, 0] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %arg14 [%126, %127] : f32 into vector<2x8xf32>
        %157 = vector.extract %154[0, 1] : f32 from vector<1x4xf32>
        %158 = vector.insert %157, %156 [%126, %130] : f32 into vector<2x8xf32>
        %159 = vector.extract %154[0, 2] : f32 from vector<1x4xf32>
        %160 = vector.insert %159, %158 [%126, %133] : f32 into vector<2x8xf32>
        %161 = vector.extract %154[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %160 [%126, %136] : f32 into vector<2x8xf32>
        %163 = arith.addf %154, %125 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg15 [%126, %127] : f32 into vector<2x8xf32>
        %166 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %167 = vector.insert %166, %165 [%126, %130] : f32 into vector<2x8xf32>
        %168 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %169 = vector.insert %168, %167 [%126, %133] : f32 into vector<2x8xf32>
        %170 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %169 [%126, %136] : f32 into vector<2x8xf32>
        %172 = vector.extract %116#3[%126, %127] : f32 from vector<2x8xf32>
        %173 = vector.extract %116#3[%126, %130] : f32 from vector<2x8xf32>
        %174 = vector.extract %116#3[%126, %133] : f32 from vector<2x8xf32>
        %175 = vector.extract %116#3[%126, %136] : f32 from vector<2x8xf32>
        %176 = vector.insert %172, %arg16 [%126, %127] : f32 into vector<2x8xf32>
        %177 = vector.insert %173, %176 [%126, %130] : f32 into vector<2x8xf32>
        %178 = vector.insert %174, %177 [%126, %133] : f32 into vector<2x8xf32>
        %179 = vector.insert %175, %178 [%126, %136] : f32 into vector<2x8xf32>
        affine.yield %138, %148, %162, %171, %179 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %123:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_6, %arg13 = %cst_6) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %125 = vector.extract %113[%arg11, 0] : f32 from vector<2x1xf32>
        %126 = vector.insert %125, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %127 = vector.extract %108[%arg11, 0] : f32 from vector<2x1xf32>
        %128 = vector.insert %127, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %126, %128 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %124:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %125 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %126 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %127 = vector.extract %118#3[%125, %126] : f32 from vector<2x8xf32>
        %128 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %129 = vector.extract %118#3[%125, %128] : f32 from vector<2x8xf32>
        %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %131 = vector.extract %118#3[%125, %130] : f32 from vector<2x8xf32>
        %132 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %133 = vector.extract %118#3[%125, %132] : f32 from vector<2x8xf32>
        %134 = vector.insert %127, %arg12 [%125, %126] : f32 into vector<2x8xf32>
        %135 = vector.insert %129, %134 [%125, %128] : f32 into vector<2x8xf32>
        %136 = vector.insert %131, %135 [%125, %130] : f32 into vector<2x8xf32>
        %137 = vector.insert %133, %136 [%125, %132] : f32 into vector<2x8xf32>
        %138 = vector.extract %120#3[%125, %126] : f32 from vector<2x8xf32>
        %139 = vector.extract %120#3[%125, %128] : f32 from vector<2x8xf32>
        %140 = vector.extract %120#3[%125, %130] : f32 from vector<2x8xf32>
        %141 = vector.extract %120#3[%125, %132] : f32 from vector<2x8xf32>
        %142 = vector.insert %138, %arg13 [%125, %126] : f32 into vector<2x8xf32>
        %143 = vector.insert %139, %142 [%125, %128] : f32 into vector<2x8xf32>
        %144 = vector.insert %140, %143 [%125, %130] : f32 into vector<2x8xf32>
        %145 = vector.insert %141, %144 [%125, %132] : f32 into vector<2x8xf32>
        %146 = vector.extract %122#3[%125, %126] : f32 from vector<2x8xf32>
        %147 = vector.extract %122#3[%125, %128] : f32 from vector<2x8xf32>
        %148 = vector.extract %122#3[%125, %130] : f32 from vector<2x8xf32>
        %149 = vector.extract %122#3[%125, %132] : f32 from vector<2x8xf32>
        %150 = vector.insert %146, %arg14 [%125, %126] : f32 into vector<2x8xf32>
        %151 = vector.insert %147, %150 [%125, %128] : f32 into vector<2x8xf32>
        %152 = vector.insert %148, %151 [%125, %130] : f32 into vector<2x8xf32>
        %153 = vector.insert %149, %152 [%125, %132] : f32 into vector<2x8xf32>
        affine.yield %137, %145, %153 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %122#4, %123#0, %123#1, %124#0, %124#1, %124#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %82 = arith.index_cast %thread_id_x : index to i32
    %83 = arith.andi %82, %c16_i32 : i32
    %84 = arith.cmpi ne, %83, %c0_i32 : i32
    %85 = arith.andi %82, %c32_i32 : i32
    %86 = arith.cmpi ne, %85, %c0_i32 : i32
    %87 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %97 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %99 = vector.extract %81#5[%98, %97] : f32 from vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %101 = vector.extract %81#5[%98, %100] : f32 from vector<2x8xf32>
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %103 = vector.extract %81#5[%98, %102] : f32 from vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %105 = vector.extract %81#5[%98, %104] : f32 from vector<2x8xf32>
      %106 = arith.select %84, %99, %101 : f32
      %shuffleResult, %valid = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %107 = arith.select %84, %shuffleResult, %99 : f32
      %108 = arith.select %84, %101, %shuffleResult : f32
      %109 = arith.select %84, %103, %105 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %110 = arith.select %84, %shuffleResult_15, %103 : f32
      %111 = arith.select %84, %105, %shuffleResult_15 : f32
      %112 = arith.select %86, %107, %110 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %113 = arith.select %86, %shuffleResult_17, %107 : f32
      %114 = arith.select %86, %110, %shuffleResult_17 : f32
      %115 = arith.select %86, %108, %111 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %86, %shuffleResult_19, %108 : f32
      %117 = arith.select %86, %111, %shuffleResult_19 : f32
      %118 = vector.insert %113, %arg5 [%98, %97] : f32 into vector<2x8xf32>
      %119 = vector.insert %116, %118 [%98, %100] : f32 into vector<2x8xf32>
      %120 = vector.insert %114, %119 [%98, %102] : f32 into vector<2x8xf32>
      %121 = vector.insert %117, %120 [%98, %104] : f32 into vector<2x8xf32>
      affine.yield %121 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %88 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %97 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %99 = vector.extract %81#4[%98, %97] : f32 from vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %101 = vector.extract %81#4[%98, %100] : f32 from vector<2x8xf32>
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %103 = vector.extract %81#4[%98, %102] : f32 from vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %105 = vector.extract %81#4[%98, %104] : f32 from vector<2x8xf32>
      %106 = arith.select %84, %99, %101 : f32
      %shuffleResult, %valid = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %107 = arith.select %84, %shuffleResult, %99 : f32
      %108 = arith.select %84, %101, %shuffleResult : f32
      %109 = arith.select %84, %103, %105 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %110 = arith.select %84, %shuffleResult_15, %103 : f32
      %111 = arith.select %84, %105, %shuffleResult_15 : f32
      %112 = arith.select %86, %107, %110 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %113 = arith.select %86, %shuffleResult_17, %107 : f32
      %114 = arith.select %86, %110, %shuffleResult_17 : f32
      %115 = arith.select %86, %108, %111 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %86, %shuffleResult_19, %108 : f32
      %117 = arith.select %86, %111, %shuffleResult_19 : f32
      %118 = vector.insert %113, %arg5 [%98, %97] : f32 into vector<2x8xf32>
      %119 = vector.insert %116, %118 [%98, %100] : f32 into vector<2x8xf32>
      %120 = vector.insert %114, %119 [%98, %102] : f32 into vector<2x8xf32>
      %121 = vector.insert %117, %120 [%98, %104] : f32 into vector<2x8xf32>
      affine.yield %121 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %89 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %97 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %99 = vector.extract %81#3[%98, %97] : f32 from vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %101 = vector.extract %81#3[%98, %100] : f32 from vector<2x8xf32>
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %103 = vector.extract %81#3[%98, %102] : f32 from vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %105 = vector.extract %81#3[%98, %104] : f32 from vector<2x8xf32>
      %106 = arith.select %84, %99, %101 : f32
      %shuffleResult, %valid = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %107 = arith.select %84, %shuffleResult, %99 : f32
      %108 = arith.select %84, %101, %shuffleResult : f32
      %109 = arith.select %84, %103, %105 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %110 = arith.select %84, %shuffleResult_15, %103 : f32
      %111 = arith.select %84, %105, %shuffleResult_15 : f32
      %112 = arith.select %86, %107, %110 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %113 = arith.select %86, %shuffleResult_17, %107 : f32
      %114 = arith.select %86, %110, %shuffleResult_17 : f32
      %115 = arith.select %86, %108, %111 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %86, %shuffleResult_19, %108 : f32
      %117 = arith.select %86, %111, %shuffleResult_19 : f32
      %118 = vector.insert %113, %arg5 [%98, %97] : f32 into vector<2x8xf32>
      %119 = vector.insert %116, %118 [%98, %100] : f32 into vector<2x8xf32>
      %120 = vector.insert %114, %119 [%98, %102] : f32 into vector<2x8xf32>
      %121 = vector.insert %117, %120 [%98, %104] : f32 into vector<2x8xf32>
      affine.yield %121 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %90 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %97 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %99 = vector.extract %81#0[%98, %97] : f32 from vector<2x8xf32>
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %101 = vector.extract %81#0[%98, %100] : f32 from vector<2x8xf32>
      %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %103 = vector.extract %81#0[%98, %102] : f32 from vector<2x8xf32>
      %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %105 = vector.extract %81#0[%98, %104] : f32 from vector<2x8xf32>
      %106 = arith.select %84, %99, %101 : f32
      %shuffleResult, %valid = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %107 = arith.select %84, %shuffleResult, %99 : f32
      %108 = arith.select %84, %101, %shuffleResult : f32
      %109 = arith.select %84, %103, %105 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %110 = arith.select %84, %shuffleResult_15, %103 : f32
      %111 = arith.select %84, %105, %shuffleResult_15 : f32
      %112 = arith.select %86, %107, %110 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %113 = arith.select %86, %shuffleResult_17, %107 : f32
      %114 = arith.select %86, %110, %shuffleResult_17 : f32
      %115 = arith.select %86, %108, %111 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %116 = arith.select %86, %shuffleResult_19, %108 : f32
      %117 = arith.select %86, %111, %shuffleResult_19 : f32
      %118 = vector.insert %113, %arg5 [%98, %97] : f32 into vector<2x8xf32>
      %119 = vector.insert %116, %118 [%98, %100] : f32 into vector<2x8xf32>
      %120 = vector.insert %114, %119 [%98, %102] : f32 into vector<2x8xf32>
      %121 = vector.insert %117, %120 [%98, %104] : f32 into vector<2x8xf32>
      affine.yield %121 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %91:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %97 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %98 = vector.extract %81#1[%97, 0] : f32 from vector<2x1xf32>
      %99 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %139 = vector.broadcast %98 : f32 to vector<1x4xf32>
        affine.yield %139 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %101 = vector.extract %99[0, 0] : f32 from vector<1x4xf32>
      %102 = vector.insert %101, %arg5 [%97, %100] : f32 into vector<2x8xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %104 = vector.extract %99[0, 1] : f32 from vector<1x4xf32>
      %105 = vector.insert %104, %102 [%97, %103] : f32 into vector<2x8xf32>
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %107 = vector.extract %99[0, 2] : f32 from vector<1x4xf32>
      %108 = vector.insert %107, %105 [%97, %106] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %110 = vector.extract %99[0, 3] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %108 [%97, %109] : f32 into vector<2x8xf32>
      %112 = vector.extract %90[%97, %100] : f32 from vector<2x8xf32>
      %113 = vector.extract %90[%97, %103] : f32 from vector<2x8xf32>
      %114 = vector.extract %90[%97, %106] : f32 from vector<2x8xf32>
      %115 = vector.extract %90[%97, %109] : f32 from vector<2x8xf32>
      %116 = vector.from_elements %112, %113, %114, %115 : vector<1x4xf32>
      %117 = arith.divf %116, %99 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %118 = vector.extract %117[0, 0] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %arg6 [%97, %100] : f32 into vector<2x8xf32>
      %120 = vector.extract %117[0, 1] : f32 from vector<1x4xf32>
      %121 = vector.insert %120, %119 [%97, %103] : f32 into vector<2x8xf32>
      %122 = vector.extract %117[0, 2] : f32 from vector<1x4xf32>
      %123 = vector.insert %122, %121 [%97, %106] : f32 into vector<2x8xf32>
      %124 = vector.extract %117[0, 3] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %123 [%97, %109] : f32 into vector<2x8xf32>
      %126 = arith.truncf %117 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %127 = vector.extract %126[0, 0] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %arg7 [%97, %100] : f16 into vector<2x8xf16>
      %129 = vector.extract %126[0, 1] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%97, %103] : f16 into vector<2x8xf16>
      %131 = vector.extract %126[0, 2] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%97, %106] : f16 into vector<2x8xf16>
      %133 = vector.extract %126[0, 3] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%97, %109] : f16 into vector<2x8xf16>
      %135 = vector.insert %127, %arg8 [%97, %100] : f16 into vector<2x8xf16>
      %136 = vector.insert %129, %135 [%97, %103] : f16 into vector<2x8xf16>
      %137 = vector.insert %131, %136 [%97, %106] : f16 into vector<2x8xf16>
      %138 = vector.insert %133, %137 [%97, %109] : f16 into vector<2x8xf16>
      affine.yield %111, %125, %134, %138 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %92 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %93 = arith.cmpi ult, %92, %c4 : index
    scf.if %93 {
      affine.for %arg4 = 0 to 4 {
        %97 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %99 = vector.extract %91#3[%97, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %101 = vector.extract %91#3[%97, %100] : f16 from vector<2x8xf16>
        %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %103 = vector.extract %91#3[%97, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %105 = vector.extract %91#3[%97, %104] : f16 from vector<2x8xf16>
        affine.store %99, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %101, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %103, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %105, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %94:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %97 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %98 = vector.extract %81#1[%97, 0] : f32 from vector<2x1xf32>
      %99 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %139 = vector.broadcast %98 : f32 to vector<1x4xf32>
        affine.yield %139 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %101 = vector.extract %99[0, 0] : f32 from vector<1x4xf32>
      %102 = vector.insert %101, %arg5 [%97, %100] : f32 into vector<2x8xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %104 = vector.extract %99[0, 1] : f32 from vector<1x4xf32>
      %105 = vector.insert %104, %102 [%97, %103] : f32 into vector<2x8xf32>
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %107 = vector.extract %99[0, 2] : f32 from vector<1x4xf32>
      %108 = vector.insert %107, %105 [%97, %106] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %110 = vector.extract %99[0, 3] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %108 [%97, %109] : f32 into vector<2x8xf32>
      %112 = vector.extract %89[%97, %100] : f32 from vector<2x8xf32>
      %113 = vector.extract %89[%97, %103] : f32 from vector<2x8xf32>
      %114 = vector.extract %89[%97, %106] : f32 from vector<2x8xf32>
      %115 = vector.extract %89[%97, %109] : f32 from vector<2x8xf32>
      %116 = vector.from_elements %112, %113, %114, %115 : vector<1x4xf32>
      %117 = arith.divf %116, %99 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %118 = vector.extract %117[0, 0] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %arg6 [%97, %100] : f32 into vector<2x8xf32>
      %120 = vector.extract %117[0, 1] : f32 from vector<1x4xf32>
      %121 = vector.insert %120, %119 [%97, %103] : f32 into vector<2x8xf32>
      %122 = vector.extract %117[0, 2] : f32 from vector<1x4xf32>
      %123 = vector.insert %122, %121 [%97, %106] : f32 into vector<2x8xf32>
      %124 = vector.extract %117[0, 3] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %123 [%97, %109] : f32 into vector<2x8xf32>
      %126 = arith.truncf %117 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %127 = vector.extract %126[0, 0] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %arg7 [%97, %100] : f16 into vector<2x8xf16>
      %129 = vector.extract %126[0, 1] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%97, %103] : f16 into vector<2x8xf16>
      %131 = vector.extract %126[0, 2] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%97, %106] : f16 into vector<2x8xf16>
      %133 = vector.extract %126[0, 3] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%97, %109] : f16 into vector<2x8xf16>
      %135 = vector.insert %127, %arg8 [%97, %100] : f16 into vector<2x8xf16>
      %136 = vector.insert %129, %135 [%97, %103] : f16 into vector<2x8xf16>
      %137 = vector.insert %131, %136 [%97, %106] : f16 into vector<2x8xf16>
      %138 = vector.insert %133, %137 [%97, %109] : f16 into vector<2x8xf16>
      affine.yield %111, %125, %134, %138 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %93 {
      affine.for %arg4 = 0 to 4 {
        %97 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %99 = vector.extract %94#3[%97, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %101 = vector.extract %94#3[%97, %100] : f16 from vector<2x8xf16>
        %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %103 = vector.extract %94#3[%97, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %105 = vector.extract %94#3[%97, %104] : f16 from vector<2x8xf16>
        affine.store %99, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %101, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %103, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %105, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %95:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %97 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %98 = vector.extract %81#1[%97, 0] : f32 from vector<2x1xf32>
      %99 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %139 = vector.broadcast %98 : f32 to vector<1x4xf32>
        affine.yield %139 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %101 = vector.extract %99[0, 0] : f32 from vector<1x4xf32>
      %102 = vector.insert %101, %arg5 [%97, %100] : f32 into vector<2x8xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %104 = vector.extract %99[0, 1] : f32 from vector<1x4xf32>
      %105 = vector.insert %104, %102 [%97, %103] : f32 into vector<2x8xf32>
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %107 = vector.extract %99[0, 2] : f32 from vector<1x4xf32>
      %108 = vector.insert %107, %105 [%97, %106] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %110 = vector.extract %99[0, 3] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %108 [%97, %109] : f32 into vector<2x8xf32>
      %112 = vector.extract %88[%97, %100] : f32 from vector<2x8xf32>
      %113 = vector.extract %88[%97, %103] : f32 from vector<2x8xf32>
      %114 = vector.extract %88[%97, %106] : f32 from vector<2x8xf32>
      %115 = vector.extract %88[%97, %109] : f32 from vector<2x8xf32>
      %116 = vector.from_elements %112, %113, %114, %115 : vector<1x4xf32>
      %117 = arith.divf %116, %99 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %118 = vector.extract %117[0, 0] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %arg6 [%97, %100] : f32 into vector<2x8xf32>
      %120 = vector.extract %117[0, 1] : f32 from vector<1x4xf32>
      %121 = vector.insert %120, %119 [%97, %103] : f32 into vector<2x8xf32>
      %122 = vector.extract %117[0, 2] : f32 from vector<1x4xf32>
      %123 = vector.insert %122, %121 [%97, %106] : f32 into vector<2x8xf32>
      %124 = vector.extract %117[0, 3] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %123 [%97, %109] : f32 into vector<2x8xf32>
      %126 = arith.truncf %117 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %127 = vector.extract %126[0, 0] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %arg7 [%97, %100] : f16 into vector<2x8xf16>
      %129 = vector.extract %126[0, 1] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%97, %103] : f16 into vector<2x8xf16>
      %131 = vector.extract %126[0, 2] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%97, %106] : f16 into vector<2x8xf16>
      %133 = vector.extract %126[0, 3] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%97, %109] : f16 into vector<2x8xf16>
      %135 = vector.insert %127, %arg8 [%97, %100] : f16 into vector<2x8xf16>
      %136 = vector.insert %129, %135 [%97, %103] : f16 into vector<2x8xf16>
      %137 = vector.insert %131, %136 [%97, %106] : f16 into vector<2x8xf16>
      %138 = vector.insert %133, %137 [%97, %109] : f16 into vector<2x8xf16>
      affine.yield %111, %125, %134, %138 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %93 {
      affine.for %arg4 = 0 to 4 {
        %97 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %99 = vector.extract %95#3[%97, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %101 = vector.extract %95#3[%97, %100] : f16 from vector<2x8xf16>
        %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %103 = vector.extract %95#3[%97, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %105 = vector.extract %95#3[%97, %104] : f16 from vector<2x8xf16>
        affine.store %99, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %101, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %103, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %105, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %96:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_5, %arg8 = %cst_5) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %97 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %98 = vector.extract %81#1[%97, 0] : f32 from vector<2x1xf32>
      %99 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_3) -> (vector<1x4xf32>) {
        %139 = vector.broadcast %98 : f32 to vector<1x4xf32>
        affine.yield %139 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %101 = vector.extract %99[0, 0] : f32 from vector<1x4xf32>
      %102 = vector.insert %101, %arg5 [%97, %100] : f32 into vector<2x8xf32>
      %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %104 = vector.extract %99[0, 1] : f32 from vector<1x4xf32>
      %105 = vector.insert %104, %102 [%97, %103] : f32 into vector<2x8xf32>
      %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %107 = vector.extract %99[0, 2] : f32 from vector<1x4xf32>
      %108 = vector.insert %107, %105 [%97, %106] : f32 into vector<2x8xf32>
      %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %110 = vector.extract %99[0, 3] : f32 from vector<1x4xf32>
      %111 = vector.insert %110, %108 [%97, %109] : f32 into vector<2x8xf32>
      %112 = vector.extract %87[%97, %100] : f32 from vector<2x8xf32>
      %113 = vector.extract %87[%97, %103] : f32 from vector<2x8xf32>
      %114 = vector.extract %87[%97, %106] : f32 from vector<2x8xf32>
      %115 = vector.extract %87[%97, %109] : f32 from vector<2x8xf32>
      %116 = vector.from_elements %112, %113, %114, %115 : vector<1x4xf32>
      %117 = arith.divf %116, %99 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %118 = vector.extract %117[0, 0] : f32 from vector<1x4xf32>
      %119 = vector.insert %118, %arg6 [%97, %100] : f32 into vector<2x8xf32>
      %120 = vector.extract %117[0, 1] : f32 from vector<1x4xf32>
      %121 = vector.insert %120, %119 [%97, %103] : f32 into vector<2x8xf32>
      %122 = vector.extract %117[0, 2] : f32 from vector<1x4xf32>
      %123 = vector.insert %122, %121 [%97, %106] : f32 into vector<2x8xf32>
      %124 = vector.extract %117[0, 3] : f32 from vector<1x4xf32>
      %125 = vector.insert %124, %123 [%97, %109] : f32 into vector<2x8xf32>
      %126 = arith.truncf %117 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %127 = vector.extract %126[0, 0] : f16 from vector<1x4xf16>
      %128 = vector.insert %127, %arg7 [%97, %100] : f16 into vector<2x8xf16>
      %129 = vector.extract %126[0, 1] : f16 from vector<1x4xf16>
      %130 = vector.insert %129, %128 [%97, %103] : f16 into vector<2x8xf16>
      %131 = vector.extract %126[0, 2] : f16 from vector<1x4xf16>
      %132 = vector.insert %131, %130 [%97, %106] : f16 into vector<2x8xf16>
      %133 = vector.extract %126[0, 3] : f16 from vector<1x4xf16>
      %134 = vector.insert %133, %132 [%97, %109] : f16 into vector<2x8xf16>
      %135 = vector.insert %127, %arg8 [%97, %100] : f16 into vector<2x8xf16>
      %136 = vector.insert %129, %135 [%97, %103] : f16 into vector<2x8xf16>
      %137 = vector.insert %131, %136 [%97, %106] : f16 into vector<2x8xf16>
      %138 = vector.insert %133, %137 [%97, %109] : f16 into vector<2x8xf16>
      affine.yield %111, %125, %134, %138 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %93 {
      affine.for %arg4 = 0 to 4 {
        %97 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %99 = vector.extract %96#3[%97, %98] : f16 from vector<2x8xf16>
        %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %101 = vector.extract %96#3[%97, %100] : f16 from vector<2x8xf16>
        %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %103 = vector.extract %96#3[%97, %102] : f16 from vector<2x8xf16>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %105 = vector.extract %96#3[%97, %104] : f16 from vector<2x8xf16>
        affine.store %99, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %101, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %103, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %105, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
