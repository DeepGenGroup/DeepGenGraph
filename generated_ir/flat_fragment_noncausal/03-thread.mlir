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
    %cst = arith.constant dense<0xFF800000> : vector<1x1xf32>
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_8 = arith.constant 0xFF800000 : f32
    %cst_9 = arith.constant 0.127531052 : f32
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
    %8 = arith.truncf %cst_9 : f32 to f16
    %9:3 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_7, %arg6 = %cst_7, %arg7 = %cst_7) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %25 = vector.insert %cst_3, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %26 = vector.extract %25[%arg4, 0] : f32 from vector<2x1xf32>
      %27 = vector.broadcast %26 : f32 to vector<1x1xf32>
      %28 = arith.addf %27, %cst {origin = "frisk.add", tiled = true} : vector<1x1xf32>
      %29 = vector.extract %28[0, 0] : f32 from vector<1x1xf32>
      %30 = vector.insert %29, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %31 = vector.extract %30[%arg4, 0] : f32 from vector<2x1xf32>
      %32 = vector.insert %31, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %25, %30, %32 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %25 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %26 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %27 = vector.insert %cst_3, %arg5 [%26, %25] : f32 into vector<2x8xf32>
      %28 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %29 = vector.insert %cst_3, %27 [%26, %28] : f32 into vector<2x8xf32>
      %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %31 = vector.insert %cst_3, %29 [%26, %30] : f32 into vector<2x8xf32>
      %32 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %33 = vector.insert %cst_3, %31 [%26, %32] : f32 into vector<2x8xf32>
      affine.yield %33 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_7) -> (vector<2x1xf32>) {
      %25 = vector.insert %cst_3, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %25 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    %12:3 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %25 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %26 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %27 = vector.insert %cst_3, %arg5 [%26, %25] : f32 into vector<2x8xf32>
      %28 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %29 = vector.insert %cst_3, %27 [%26, %28] : f32 into vector<2x8xf32>
      %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %31 = vector.insert %cst_3, %29 [%26, %30] : f32 into vector<2x8xf32>
      %32 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %33 = vector.insert %cst_3, %31 [%26, %32] : f32 into vector<2x8xf32>
      %34 = vector.insert %cst_3, %arg6 [%26, %25] : f32 into vector<2x8xf32>
      %35 = vector.insert %cst_3, %34 [%26, %28] : f32 into vector<2x8xf32>
      %36 = vector.insert %cst_3, %35 [%26, %30] : f32 into vector<2x8xf32>
      %37 = vector.insert %cst_3, %36 [%26, %32] : f32 into vector<2x8xf32>
      %38 = vector.insert %cst_3, %arg7 [%26, %25] : f32 into vector<2x8xf32>
      %39 = vector.insert %cst_3, %38 [%26, %28] : f32 into vector<2x8xf32>
      %40 = vector.insert %cst_3, %39 [%26, %30] : f32 into vector<2x8xf32>
      %41 = vector.insert %cst_3, %40 [%26, %32] : f32 into vector<2x8xf32>
      affine.yield %33, %37, %41 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %13:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#2, %arg8 = %12#0, %arg9 = %12#1, %arg10 = %12#2) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %25 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %45 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %46 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %47 = vector.insert %cst_3, %arg12 [%46, %45] : f32 into vector<2x32xf32>
        %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %49 = vector.insert %cst_3, %47 [%46, %48] : f32 into vector<2x32xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %51 = vector.insert %cst_3, %49 [%46, %50] : f32 into vector<2x32xf32>
        %52 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %53 = vector.insert %cst_3, %51 [%46, %52] : f32 into vector<2x32xf32>
        affine.yield %53 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
      %26 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %25) -> (vector<2x32xf32>) {
        %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %45 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%6, %7]
        %46 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%6]
        %subview_19 = memref.subview %arg0[0, %45, %46, %arg11] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
        frisk.async_copy_gm_to_shm %subview_19 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
        frisk.wait_buffer_ready %alloc_17 : memref<128x32xf16, 3>
        %47 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
        %subview_20 = memref.subview %transpose[0, %47, %arg11, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
        frisk.async_copy_gm_to_shm %subview_20 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
        %thread_id_x_21 = gpu.thread_id x
        %c0 = arith.constant 0 : index
        %48 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%c0)[%thread_id_x_21]
        %49 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4)>(%c0)[%thread_id_x_21]
        %50 = memref.load %alloc_17[%48, %49] {frisk.pipeline_stage = "prologue"} : memref<128x32xf16, 3>
        %51 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 1)>(%c0)[%thread_id_x_21]
        %52 = memref.load %alloc_17[%48, %51] {frisk.pipeline_stage = "prologue"} : memref<128x32xf16, 3>
        %53 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 2)>(%c0)[%thread_id_x_21]
        %54 = memref.load %alloc_17[%48, %53] {frisk.pipeline_stage = "prologue"} : memref<128x32xf16, 3>
        %55 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 3)>(%c0)[%thread_id_x_21]
        %56 = memref.load %alloc_17[%48, %55] {frisk.pipeline_stage = "prologue"} : memref<128x32xf16, 3>
        %57:6 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %cst_6, %arg15 = %cst_6, %arg16 = %50, %arg17 = %52, %arg18 = %54, %arg19 = %56) -> (vector<2x8xf16>, vector<2x8xf16>, f16, f16, f16, f16) {
          %83 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg13)
          %84 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%83)[%thread_id_x_21]
          %85 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4)>(%83)[%thread_id_x_21]
          %86 = memref.load %alloc_17[%84, %85] {frisk.pipeline_stage = "prefetch"} : memref<128x32xf16, 3>
          %87 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 1)>(%83)[%thread_id_x_21]
          %88 = memref.load %alloc_17[%84, %87] {frisk.pipeline_stage = "prefetch"} : memref<128x32xf16, 3>
          %89 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 2)>(%83)[%thread_id_x_21]
          %90 = memref.load %alloc_17[%84, %89] {frisk.pipeline_stage = "prefetch"} : memref<128x32xf16, 3>
          %91 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 3)>(%83)[%thread_id_x_21]
          %92 = memref.load %alloc_17[%84, %91] {frisk.pipeline_stage = "prefetch"} : memref<128x32xf16, 3>
          %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %94 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg13)
          %95 = vector.insert %arg16, %arg14 [%94, %93] : f16 into vector<2x8xf16>
          %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg13)
          %97 = vector.insert %arg17, %95 [%94, %96] : f16 into vector<2x8xf16>
          %98 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg13)
          %99 = vector.insert %arg18, %97 [%94, %98] : f16 into vector<2x8xf16>
          %100 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg13)
          %101 = vector.insert %arg19, %99 [%94, %100] : f16 into vector<2x8xf16>
          %102 = vector.extract %101[%94, %93] : f16 from vector<2x8xf16>
          %103 = vector.extract %101[%94, %96] : f16 from vector<2x8xf16>
          %104 = vector.extract %101[%94, %98] : f16 from vector<2x8xf16>
          %105 = vector.extract %101[%94, %100] : f16 from vector<2x8xf16>
          %106 = vector.from_elements %102, %103, %104, %105 : vector<1x4xf16>
          %107 = vector.broadcast %8 : f16 to vector<1x4xf16>
          %108 = arith.mulf %106, %107 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
          %109 = vector.extract %108[0, 0] : f16 from vector<1x4xf16>
          %110 = vector.insert %109, %arg15 [%94, %93] : f16 into vector<2x8xf16>
          %111 = vector.extract %108[0, 1] : f16 from vector<1x4xf16>
          %112 = vector.insert %111, %110 [%94, %96] : f16 into vector<2x8xf16>
          %113 = vector.extract %108[0, 2] : f16 from vector<1x4xf16>
          %114 = vector.insert %113, %112 [%94, %98] : f16 into vector<2x8xf16>
          %115 = vector.extract %108[0, 3] : f16 from vector<1x4xf16>
          %116 = vector.insert %115, %114 [%94, %100] : f16 into vector<2x8xf16>
          affine.yield %101, %116, %86, %88, %90, %92 : vector<2x8xf16>, vector<2x8xf16>, f16, f16, f16, f16
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
        %c3 = arith.constant 3 : index
        %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%c3)
        %59 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c3)
        %60 = vector.insert %57#2, %57#0 [%59, %58] : f16 into vector<2x8xf16>
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%c3)
        %62 = vector.insert %57#3, %60 [%59, %61] : f16 into vector<2x8xf16>
        %63 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%c3)
        %64 = vector.insert %57#4, %62 [%59, %63] : f16 into vector<2x8xf16>
        %65 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%c3)
        %66 = vector.insert %57#5, %64 [%59, %65] : f16 into vector<2x8xf16>
        %67 = vector.extract %66[%59, %58] : f16 from vector<2x8xf16>
        %68 = vector.extract %66[%59, %61] : f16 from vector<2x8xf16>
        %69 = vector.extract %66[%59, %63] : f16 from vector<2x8xf16>
        %70 = vector.extract %66[%59, %65] : f16 from vector<2x8xf16>
        %71 = vector.from_elements %67, %68, %69, %70 : vector<1x4xf16>
        %72 = vector.broadcast %8 : f16 to vector<1x4xf16>
        %73 = arith.mulf %71, %72 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
        %74 = vector.extract %73[0, 0] : f16 from vector<1x4xf16>
        %75 = vector.insert %74, %57#1 [%59, %58] : f16 into vector<2x8xf16>
        %76 = vector.extract %73[0, 1] : f16 from vector<1x4xf16>
        %77 = vector.insert %76, %75 [%59, %61] : f16 into vector<2x8xf16>
        %78 = vector.extract %73[0, 2] : f16 from vector<1x4xf16>
        %79 = vector.insert %78, %77 [%59, %63] : f16 into vector<2x8xf16>
        %80 = vector.extract %73[0, 3] : f16 from vector<1x4xf16>
        %81 = vector.insert %80, %79 [%59, %65] : f16 into vector<2x8xf16>
        frisk.wait_buffer_ready %alloc_18 : memref<32x128xf16, 3>
        %82:3 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5, %arg15 = %cst_5, %arg16 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>, vector<2x32xf32>) {
          %83 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4)>(%c0)[%thread_id_x_21]
          %84 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg13)[%thread_id_x_21]
          %85 = memref.load %alloc_18[%83, %84] {frisk.pipeline_stage = "prologue"} : memref<32x128xf16, 3>
          %86 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>(%c0)[%thread_id_x_21]
          %87 = memref.load %alloc_18[%86, %84] {frisk.pipeline_stage = "prologue"} : memref<32x128xf16, 3>
          %88 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>(%c0)[%thread_id_x_21]
          %89 = memref.load %alloc_18[%88, %84] {frisk.pipeline_stage = "prologue"} : memref<32x128xf16, 3>
          %90 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>(%c0)[%thread_id_x_21]
          %91 = memref.load %alloc_18[%90, %84] {frisk.pipeline_stage = "prologue"} : memref<32x128xf16, 3>
          %92:5 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_4, %arg19 = %85, %arg20 = %87, %arg21 = %89, %arg22 = %91) -> (vector<1x4xf32>, f16, f16, f16, f16) {
            %144 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg17)
            %145 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4)>(%144)[%thread_id_x_21]
            %146 = memref.load %alloc_18[%145, %84] {frisk.pipeline_stage = "prefetch"} : memref<32x128xf16, 3>
            %147 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>(%144)[%thread_id_x_21]
            %148 = memref.load %alloc_18[%147, %84] {frisk.pipeline_stage = "prefetch"} : memref<32x128xf16, 3>
            %149 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>(%144)[%thread_id_x_21]
            %150 = memref.load %alloc_18[%149, %84] {frisk.pipeline_stage = "prefetch"} : memref<32x128xf16, 3>
            %151 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>(%144)[%thread_id_x_21]
            %152 = memref.load %alloc_18[%151, %84] {frisk.pipeline_stage = "prefetch"} : memref<32x128xf16, 3>
            %153 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
            %154 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
            %155 = vector.extract %81[%153, %154] : f16 from vector<2x8xf16>
            %156 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
            %157 = vector.extract %81[%153, %156] : f16 from vector<2x8xf16>
            %158 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
            %159 = vector.extract %81[%153, %158] : f16 from vector<2x8xf16>
            %160 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
            %161 = vector.extract %81[%153, %160] : f16 from vector<2x8xf16>
            %162 = vector.from_elements %155, %157, %159, %161 : vector<1x4xf16>
            %163 = vector.from_elements %arg19, %arg20, %arg21, %arg22 : vector<4x1xf16>
            %164 = frisk.warp_mma_rr %162, %163, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            affine.yield %164, %146, %148, %150, %152 : vector<1x4xf32>, f16, f16, f16, f16
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
          %c1 = arith.constant 1 : index
          %93 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %94 = affine.apply affine_map<(d0) -> (d0 * 4)>(%c1)
          %95 = vector.extract %81[%93, %94] : f16 from vector<2x8xf16>
          %96 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%c1)
          %97 = vector.extract %81[%93, %96] : f16 from vector<2x8xf16>
          %98 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%c1)
          %99 = vector.extract %81[%93, %98] : f16 from vector<2x8xf16>
          %100 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%c1)
          %101 = vector.extract %81[%93, %100] : f16 from vector<2x8xf16>
          %102 = vector.from_elements %95, %97, %99, %101 : vector<1x4xf16>
          %103 = vector.from_elements %92#1, %92#2, %92#3, %92#4 : vector<4x1xf16>
          %104 = frisk.warp_mma_rr %102, %103, %92#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %106 = vector.extract %104[0, 0] : f32 from vector<1x4xf32>
          %107 = vector.insert %106, %arg14 [%93, %105] : f32 into vector<2x32xf32>
          %108 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %109 = vector.extract %104[0, 1] : f32 from vector<1x4xf32>
          %110 = vector.insert %109, %107 [%93, %108] : f32 into vector<2x32xf32>
          %111 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %112 = vector.extract %104[0, 2] : f32 from vector<1x4xf32>
          %113 = vector.insert %112, %110 [%93, %111] : f32 into vector<2x32xf32>
          %114 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %115 = vector.extract %104[0, 3] : f32 from vector<1x4xf32>
          %116 = vector.insert %115, %113 [%93, %114] : f32 into vector<2x32xf32>
          %117 = vector.extract %arg12[%93, %105] : f32 from vector<2x32xf32>
          %118 = vector.extract %arg12[%93, %108] : f32 from vector<2x32xf32>
          %119 = vector.extract %arg12[%93, %111] : f32 from vector<2x32xf32>
          %120 = vector.extract %arg12[%93, %114] : f32 from vector<2x32xf32>
          %121 = vector.from_elements %117, %118, %119, %120 : vector<1x4xf32>
          %122 = vector.extract %116[%93, %105] : f32 from vector<2x32xf32>
          %123 = vector.extract %116[%93, %108] : f32 from vector<2x32xf32>
          %124 = vector.extract %116[%93, %111] : f32 from vector<2x32xf32>
          %125 = vector.extract %116[%93, %114] : f32 from vector<2x32xf32>
          %126 = vector.from_elements %122, %123, %124, %125 : vector<1x4xf32>
          %127 = arith.addf %121, %126 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          %128 = vector.extract %127[0, 0] : f32 from vector<1x4xf32>
          %129 = vector.insert %128, %arg15 [%93, %105] : f32 into vector<2x32xf32>
          %130 = vector.extract %127[0, 1] : f32 from vector<1x4xf32>
          %131 = vector.insert %130, %129 [%93, %108] : f32 into vector<2x32xf32>
          %132 = vector.extract %127[0, 2] : f32 from vector<1x4xf32>
          %133 = vector.insert %132, %131 [%93, %111] : f32 into vector<2x32xf32>
          %134 = vector.extract %127[0, 3] : f32 from vector<1x4xf32>
          %135 = vector.insert %134, %133 [%93, %114] : f32 into vector<2x32xf32>
          %136 = vector.extract %135[%93, %105] : f32 from vector<2x32xf32>
          %137 = vector.extract %135[%93, %108] : f32 from vector<2x32xf32>
          %138 = vector.extract %135[%93, %111] : f32 from vector<2x32xf32>
          %139 = vector.extract %135[%93, %114] : f32 from vector<2x32xf32>
          %140 = vector.insert %136, %arg16 [%93, %105] : f32 into vector<2x32xf32>
          %141 = vector.insert %137, %140 [%93, %108] : f32 into vector<2x32xf32>
          %142 = vector.insert %138, %141 [%93, %111] : f32 into vector<2x32xf32>
          %143 = vector.insert %139, %142 [%93, %114] : f32 into vector<2x32xf32>
          affine.yield %116, %135, %143 : vector<2x32xf32>, vector<2x32xf32>, vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
        affine.yield %82#2 : vector<2x32xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %27 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %45 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %46 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %47 = vector.extract %26[%45, %46] : f32 from vector<2x32xf32>
        %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %49 = vector.extract %26[%45, %48] : f32 from vector<2x32xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %51 = vector.extract %26[%45, %50] : f32 from vector<2x32xf32>
        %52 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %53 = vector.extract %26[%45, %52] : f32 from vector<2x32xf32>
        %54 = vector.insert %47, %arg12 [%45, %46] : f32 into vector<2x32xf32>
        %55 = vector.insert %49, %54 [%45, %48] : f32 into vector<2x32xf32>
        %56 = vector.insert %51, %55 [%45, %50] : f32 into vector<2x32xf32>
        %57 = vector.insert %53, %56 [%45, %52] : f32 into vector<2x32xf32>
        affine.yield %57 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %28:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_7, %arg13 = %cst_7) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %45 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_8) -> (f32) {
          %56 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %57 = vector.extract %27[%arg11, %56] : f32 from vector<2x32xf32>
          %58 = arith.maxnumf %arg15, %57 : f32
          %59 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %60 = vector.extract %27[%arg11, %59] : f32 from vector<2x32xf32>
          %61 = arith.maxnumf %58, %60 : f32
          %62 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %63 = vector.extract %27[%arg11, %62] : f32 from vector<2x32xf32>
          %64 = arith.maxnumf %61, %63 : f32
          %65 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %66 = vector.extract %27[%arg11, %65] : f32 from vector<2x32xf32>
          %67 = arith.maxnumf %64, %66 : f32
          affine.yield %67 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %45, %c16_i32, %c64_i32 : f32
        %46 = arith.maxnumf %45, %shuffleResult : f32
        %shuffleResult_17, %valid_18 = gpu.shuffle xor %46, %c32_i32, %c64_i32 : f32
        %47 = arith.maxnumf %46, %shuffleResult_17 : f32
        %48 = vector.insert %47, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %49 = vector.extract %48[%arg11, 0] : f32 from vector<2x1xf32>
        %50 = vector.broadcast %49 : f32 to vector<1x1xf32>
        %51 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %52 = vector.broadcast %51 : f32 to vector<1x1xf32>
        %53 = arith.maxnumf %50, %52 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %54 = vector.extract %53[0, 0] : f32 from vector<1x1xf32>
        %55 = vector.insert %54, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %48, %55 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %29:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %45 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %46 = vector.extract %28#1[%45, 0] : f32 from vector<2x1xf32>
        %47 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %48 = vector.insert %46, %arg12 [%45, %47] : f32 into vector<2x32xf32>
        %49 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %50 = vector.insert %46, %48 [%45, %49] : f32 into vector<2x32xf32>
        %51 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %52 = vector.insert %46, %50 [%45, %51] : f32 into vector<2x32xf32>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %54 = vector.insert %46, %52 [%45, %53] : f32 into vector<2x32xf32>
        %55 = vector.extract %26[%45, %47] : f32 from vector<2x32xf32>
        %56 = vector.extract %26[%45, %49] : f32 from vector<2x32xf32>
        %57 = vector.extract %26[%45, %51] : f32 from vector<2x32xf32>
        %58 = vector.extract %26[%45, %53] : f32 from vector<2x32xf32>
        %59 = vector.from_elements %55, %56, %57, %58 : vector<1x4xf32>
        %60 = vector.extract %54[%45, %47] : f32 from vector<2x32xf32>
        %61 = vector.extract %54[%45, %49] : f32 from vector<2x32xf32>
        %62 = vector.extract %54[%45, %51] : f32 from vector<2x32xf32>
        %63 = vector.extract %54[%45, %53] : f32 from vector<2x32xf32>
        %64 = vector.from_elements %60, %61, %62, %63 : vector<1x4xf32>
        %65 = arith.subf %59, %64 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %66 = vector.extract %65[0, 0] : f32 from vector<1x4xf32>
        %67 = vector.insert %66, %arg13 [%45, %47] : f32 into vector<2x32xf32>
        %68 = vector.extract %65[0, 1] : f32 from vector<1x4xf32>
        %69 = vector.insert %68, %67 [%45, %49] : f32 into vector<2x32xf32>
        %70 = vector.extract %65[0, 2] : f32 from vector<1x4xf32>
        %71 = vector.insert %70, %69 [%45, %51] : f32 into vector<2x32xf32>
        %72 = vector.extract %65[0, 3] : f32 from vector<1x4xf32>
        %73 = vector.insert %72, %71 [%45, %53] : f32 into vector<2x32xf32>
        affine.yield %54, %73 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>}
      %30:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_7, %arg13 = %cst_7, %arg14 = %cst_7) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %45 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
        %46 = vector.broadcast %45 : f32 to vector<1x1xf32>
        %47 = vector.extract %28#1[%arg11, 0] : f32 from vector<2x1xf32>
        %48 = vector.broadcast %47 : f32 to vector<1x1xf32>
        %49 = arith.subf %46, %48 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %50 = vector.extract %49[0, 0] : f32 from vector<1x1xf32>
        %51 = vector.insert %50, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %52 = vector.extract %51[%arg11, 0] : f32 from vector<2x1xf32>
        %53 = vector.broadcast %52 : f32 to vector<1x1xf32>
        %54 = math.exp2 %53 {tiled = true} : vector<1x1xf32>
        %55 = vector.extract %54[0, 0] : f32 from vector<1x1xf32>
        %56 = vector.insert %55, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %57 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
        %58 = vector.broadcast %57 : f32 to vector<1x1xf32>
        %59 = vector.extract %56[%arg11, 0] : f32 from vector<2x1xf32>
        %60 = vector.broadcast %59 : f32 to vector<1x1xf32>
        %61 = arith.mulf %58, %60 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %62 = vector.extract %61[0, 0] : f32 from vector<1x1xf32>
        %63 = vector.insert %62, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %51, %56, %63 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %31:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %45 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %46 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %47 = vector.extract %29#1[%45, %46] : f32 from vector<2x32xf32>
        %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %49 = vector.extract %29#1[%45, %48] : f32 from vector<2x32xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %51 = vector.extract %29#1[%45, %50] : f32 from vector<2x32xf32>
        %52 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %53 = vector.extract %29#1[%45, %52] : f32 from vector<2x32xf32>
        %54 = vector.from_elements %47, %49, %51, %53 : vector<1x4xf32>
        %55 = math.exp2 %54 {tiled = true} : vector<1x4xf32>
        %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
        %57 = vector.insert %56, %arg12 [%45, %46] : f32 into vector<2x32xf32>
        %58 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
        %59 = vector.insert %58, %57 [%45, %48] : f32 into vector<2x32xf32>
        %60 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
        %61 = vector.insert %60, %59 [%45, %50] : f32 into vector<2x32xf32>
        %62 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
        %63 = vector.insert %62, %61 [%45, %52] : f32 into vector<2x32xf32>
        %64 = vector.extract %63[%45, %46] : f32 from vector<2x32xf32>
        %65 = vector.extract %63[%45, %48] : f32 from vector<2x32xf32>
        %66 = vector.extract %63[%45, %50] : f32 from vector<2x32xf32>
        %67 = vector.extract %63[%45, %52] : f32 from vector<2x32xf32>
        %68 = vector.insert %64, %arg13 [%45, %46] : f32 into vector<2x32xf32>
        %69 = vector.insert %65, %68 [%45, %48] : f32 into vector<2x32xf32>
        %70 = vector.insert %66, %69 [%45, %50] : f32 into vector<2x32xf32>
        %71 = vector.insert %67, %70 [%45, %52] : f32 into vector<2x32xf32>
        affine.yield %63, %71 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %32:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_7, %arg13 = %cst_7) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %45 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_3) -> (f32) {
          %56 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %57 = vector.extract %31#1[%arg11, %56] : f32 from vector<2x32xf32>
          %58 = arith.addf %arg15, %57 : f32
          %59 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %60 = vector.extract %31#1[%arg11, %59] : f32 from vector<2x32xf32>
          %61 = arith.addf %58, %60 : f32
          %62 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %63 = vector.extract %31#1[%arg11, %62] : f32 from vector<2x32xf32>
          %64 = arith.addf %61, %63 : f32
          %65 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %66 = vector.extract %31#1[%arg11, %65] : f32 from vector<2x32xf32>
          %67 = arith.addf %64, %66 : f32
          affine.yield %67 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %45, %c16_i32, %c64_i32 : f32
        %46 = arith.addf %45, %shuffleResult : f32
        %shuffleResult_17, %valid_18 = gpu.shuffle xor %46, %c32_i32, %c64_i32 : f32
        %47 = arith.addf %46, %shuffleResult_17 : f32
        %48 = vector.insert %47, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %49 = vector.extract %48[%arg11, 0] : f32 from vector<2x1xf32>
        %50 = vector.broadcast %49 : f32 to vector<1x1xf32>
        %51 = vector.extract %30#2[%arg11, 0] : f32 from vector<2x1xf32>
        %52 = vector.broadcast %51 : f32 to vector<1x1xf32>
        %53 = arith.addf %50, %52 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %54 = vector.extract %53[0, 0] : f32 from vector<1x1xf32>
        %55 = vector.insert %54, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %48, %55 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %33 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %45 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %46 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %47 = vector.extract %31#0[%45, %46] : f32 from vector<2x32xf32>
        %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %49 = vector.extract %31#0[%45, %48] : f32 from vector<2x32xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %51 = vector.extract %31#0[%45, %50] : f32 from vector<2x32xf32>
        %52 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %53 = vector.extract %31#0[%45, %52] : f32 from vector<2x32xf32>
        %54 = vector.from_elements %47, %49, %51, %53 : vector<1x4xf32>
        %55 = arith.truncf %54 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
        %57 = vector.insert %56, %arg12 [%45, %46] : f16 into vector<2x32xf16>
        %58 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
        %59 = vector.insert %58, %57 [%45, %48] : f16 into vector<2x32xf16>
        %60 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
        %61 = vector.insert %60, %59 [%45, %50] : f16 into vector<2x32xf16>
        %62 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
        %63 = vector.insert %62, %61 [%45, %52] : f16 into vector<2x32xf16>
        affine.yield %63 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %34 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
      %subview = memref.subview %arg2[0, %34, %arg4, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_10 = gpu.thread_id x
      %35 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %45 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_10]
        %46 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_10]
        %47 = frisk.read_matrix_from_shm %alloc[%45, %46] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %48 = vector.bitcast %47 : vector<4xi32> to vector<8xf16>
        %49 = vector.extract %48[0] : f16 from vector<8xf16>
        %50 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_10]
        %51 = vector.extract %48[%50] : f16 from vector<8xf16>
        %52 = vector.extract %48[1] : f16 from vector<8xf16>
        %53 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_10]
        %54 = vector.extract %48[%53] : f16 from vector<8xf16>
        %55 = vector.extract %48[2] : f16 from vector<8xf16>
        %56 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_10]
        %57 = vector.extract %48[%56] : f16 from vector<8xf16>
        %58 = vector.extract %48[3] : f16 from vector<8xf16>
        %59 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_10]
        %60 = vector.extract %48[%59] : f16 from vector<8xf16>
        %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %62 = vector.insert %49, %arg12 [%61, 0] : f16 into vector<32x2xf16>
        %63 = vector.insert %51, %62 [%61, 1] : f16 into vector<32x2xf16>
        %64 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %65 = vector.insert %52, %63 [%64, 0] : f16 into vector<32x2xf16>
        %66 = vector.insert %54, %65 [%64, 1] : f16 into vector<32x2xf16>
        %67 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %68 = vector.insert %55, %66 [%67, 0] : f16 into vector<32x2xf16>
        %69 = vector.insert %57, %68 [%67, 1] : f16 into vector<32x2xf16>
        %70 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %71 = vector.insert %58, %69 [%70, 0] : f16 into vector<32x2xf16>
        %72 = vector.insert %60, %71 [%70, 1] : f16 into vector<32x2xf16>
        affine.yield %72 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %36:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_4) -> (vector<1x4xf32>) {
          %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %103 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %104 = vector.extract %33[%102, %103] : f16 from vector<2x32xf16>
          %105 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %106 = vector.extract %33[%102, %105] : f16 from vector<2x32xf16>
          %107 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %108 = vector.extract %33[%102, %107] : f16 from vector<2x32xf16>
          %109 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %110 = vector.extract %33[%102, %109] : f16 from vector<2x32xf16>
          %111 = vector.from_elements %104, %106, %108, %110 : vector<1x4xf16>
          %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %113 = vector.extract %35[%103, %112] : f16 from vector<32x2xf16>
          %114 = vector.extract %35[%105, %112] : f16 from vector<32x2xf16>
          %115 = vector.extract %35[%107, %112] : f16 from vector<32x2xf16>
          %116 = vector.extract %35[%109, %112] : f16 from vector<32x2xf16>
          %117 = vector.from_elements %113, %114, %115, %116 : vector<4x1xf16>
          %118 = frisk.warp_mma_rr %111, %117, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %118 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %46 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %47 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %48 = vector.extract %45[0, 0] : f32 from vector<1x4xf32>
        %49 = vector.insert %48, %arg12 [%46, %47] : f32 into vector<2x8xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %51 = vector.extract %45[0, 1] : f32 from vector<1x4xf32>
        %52 = vector.insert %51, %49 [%46, %50] : f32 into vector<2x8xf32>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %54 = vector.extract %45[0, 2] : f32 from vector<1x4xf32>
        %55 = vector.insert %54, %52 [%46, %53] : f32 into vector<2x8xf32>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %57 = vector.extract %45[0, 3] : f32 from vector<1x4xf32>
        %58 = vector.insert %57, %55 [%46, %56] : f32 into vector<2x8xf32>
        %59 = vector.extract %30#1[%46, 0] : f32 from vector<2x1xf32>
        %60 = vector.insert %59, %arg13 [%46, %47] : f32 into vector<2x8xf32>
        %61 = vector.insert %59, %60 [%46, %50] : f32 into vector<2x8xf32>
        %62 = vector.insert %59, %61 [%46, %53] : f32 into vector<2x8xf32>
        %63 = vector.insert %59, %62 [%46, %56] : f32 into vector<2x8xf32>
        %64 = vector.extract %arg5[%46, %47] : f32 from vector<2x8xf32>
        %65 = vector.extract %arg5[%46, %50] : f32 from vector<2x8xf32>
        %66 = vector.extract %arg5[%46, %53] : f32 from vector<2x8xf32>
        %67 = vector.extract %arg5[%46, %56] : f32 from vector<2x8xf32>
        %68 = vector.from_elements %64, %65, %66, %67 : vector<1x4xf32>
        %69 = vector.extract %63[%46, %47] : f32 from vector<2x8xf32>
        %70 = vector.extract %63[%46, %50] : f32 from vector<2x8xf32>
        %71 = vector.extract %63[%46, %53] : f32 from vector<2x8xf32>
        %72 = vector.extract %63[%46, %56] : f32 from vector<2x8xf32>
        %73 = vector.from_elements %69, %70, %71, %72 : vector<1x4xf32>
        %74 = arith.mulf %68, %73 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %75 = vector.extract %74[0, 0] : f32 from vector<1x4xf32>
        %76 = vector.insert %75, %arg14 [%46, %47] : f32 into vector<2x8xf32>
        %77 = vector.extract %74[0, 1] : f32 from vector<1x4xf32>
        %78 = vector.insert %77, %76 [%46, %50] : f32 into vector<2x8xf32>
        %79 = vector.extract %74[0, 2] : f32 from vector<1x4xf32>
        %80 = vector.insert %79, %78 [%46, %53] : f32 into vector<2x8xf32>
        %81 = vector.extract %74[0, 3] : f32 from vector<1x4xf32>
        %82 = vector.insert %81, %80 [%46, %56] : f32 into vector<2x8xf32>
        %83 = vector.extract %82[%46, %47] : f32 from vector<2x8xf32>
        %84 = vector.extract %82[%46, %50] : f32 from vector<2x8xf32>
        %85 = vector.extract %82[%46, %53] : f32 from vector<2x8xf32>
        %86 = vector.extract %82[%46, %56] : f32 from vector<2x8xf32>
        %87 = vector.from_elements %83, %84, %85, %86 : vector<1x4xf32>
        %88 = vector.extract %58[%46, %47] : f32 from vector<2x8xf32>
        %89 = vector.extract %58[%46, %50] : f32 from vector<2x8xf32>
        %90 = vector.extract %58[%46, %53] : f32 from vector<2x8xf32>
        %91 = vector.extract %58[%46, %56] : f32 from vector<2x8xf32>
        %92 = vector.from_elements %88, %89, %90, %91 : vector<1x4xf32>
        %93 = arith.addf %87, %92 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %94 = vector.extract %93[0, 0] : f32 from vector<1x4xf32>
        %95 = vector.insert %94, %arg15 [%46, %47] : f32 into vector<2x8xf32>
        %96 = vector.extract %93[0, 1] : f32 from vector<1x4xf32>
        %97 = vector.insert %96, %95 [%46, %50] : f32 into vector<2x8xf32>
        %98 = vector.extract %93[0, 2] : f32 from vector<1x4xf32>
        %99 = vector.insert %98, %97 [%46, %53] : f32 into vector<2x8xf32>
        %100 = vector.extract %93[0, 3] : f32 from vector<1x4xf32>
        %101 = vector.insert %100, %99 [%46, %56] : f32 into vector<2x8xf32>
        affine.yield %58, %63, %82, %101 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_11 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_12 = memref.subview %arg2[0, %34, %arg4, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_12 to %alloc_11 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_11 : memref<128x32xf16, 3>
      %37 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %45 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_10]
        %46 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_10]
        %47 = frisk.read_matrix_from_shm %alloc_11[%45, %46] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %48 = vector.bitcast %47 : vector<4xi32> to vector<8xf16>
        %49 = vector.extract %48[0] : f16 from vector<8xf16>
        %50 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_10]
        %51 = vector.extract %48[%50] : f16 from vector<8xf16>
        %52 = vector.extract %48[1] : f16 from vector<8xf16>
        %53 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_10]
        %54 = vector.extract %48[%53] : f16 from vector<8xf16>
        %55 = vector.extract %48[2] : f16 from vector<8xf16>
        %56 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_10]
        %57 = vector.extract %48[%56] : f16 from vector<8xf16>
        %58 = vector.extract %48[3] : f16 from vector<8xf16>
        %59 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_10]
        %60 = vector.extract %48[%59] : f16 from vector<8xf16>
        %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %62 = vector.insert %49, %arg12 [%61, 0] : f16 into vector<32x2xf16>
        %63 = vector.insert %51, %62 [%61, 1] : f16 into vector<32x2xf16>
        %64 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %65 = vector.insert %52, %63 [%64, 0] : f16 into vector<32x2xf16>
        %66 = vector.insert %54, %65 [%64, 1] : f16 into vector<32x2xf16>
        %67 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %68 = vector.insert %55, %66 [%67, 0] : f16 into vector<32x2xf16>
        %69 = vector.insert %57, %68 [%67, 1] : f16 into vector<32x2xf16>
        %70 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %71 = vector.insert %58, %69 [%70, 0] : f16 into vector<32x2xf16>
        %72 = vector.insert %60, %71 [%70, 1] : f16 into vector<32x2xf16>
        affine.yield %72 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %38:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_4) -> (vector<1x4xf32>) {
          %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %103 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %104 = vector.extract %33[%102, %103] : f16 from vector<2x32xf16>
          %105 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %106 = vector.extract %33[%102, %105] : f16 from vector<2x32xf16>
          %107 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %108 = vector.extract %33[%102, %107] : f16 from vector<2x32xf16>
          %109 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %110 = vector.extract %33[%102, %109] : f16 from vector<2x32xf16>
          %111 = vector.from_elements %104, %106, %108, %110 : vector<1x4xf16>
          %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %113 = vector.extract %37[%103, %112] : f16 from vector<32x2xf16>
          %114 = vector.extract %37[%105, %112] : f16 from vector<32x2xf16>
          %115 = vector.extract %37[%107, %112] : f16 from vector<32x2xf16>
          %116 = vector.extract %37[%109, %112] : f16 from vector<32x2xf16>
          %117 = vector.from_elements %113, %114, %115, %116 : vector<4x1xf16>
          %118 = frisk.warp_mma_rr %111, %117, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %118 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %46 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %47 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %48 = vector.extract %45[0, 0] : f32 from vector<1x4xf32>
        %49 = vector.insert %48, %arg12 [%46, %47] : f32 into vector<2x8xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %51 = vector.extract %45[0, 1] : f32 from vector<1x4xf32>
        %52 = vector.insert %51, %49 [%46, %50] : f32 into vector<2x8xf32>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %54 = vector.extract %45[0, 2] : f32 from vector<1x4xf32>
        %55 = vector.insert %54, %52 [%46, %53] : f32 into vector<2x8xf32>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %57 = vector.extract %45[0, 3] : f32 from vector<1x4xf32>
        %58 = vector.insert %57, %55 [%46, %56] : f32 into vector<2x8xf32>
        %59 = vector.extract %30#1[%46, 0] : f32 from vector<2x1xf32>
        %60 = vector.insert %59, %arg13 [%46, %47] : f32 into vector<2x8xf32>
        %61 = vector.insert %59, %60 [%46, %50] : f32 into vector<2x8xf32>
        %62 = vector.insert %59, %61 [%46, %53] : f32 into vector<2x8xf32>
        %63 = vector.insert %59, %62 [%46, %56] : f32 into vector<2x8xf32>
        %64 = vector.extract %arg8[%46, %47] : f32 from vector<2x8xf32>
        %65 = vector.extract %arg8[%46, %50] : f32 from vector<2x8xf32>
        %66 = vector.extract %arg8[%46, %53] : f32 from vector<2x8xf32>
        %67 = vector.extract %arg8[%46, %56] : f32 from vector<2x8xf32>
        %68 = vector.from_elements %64, %65, %66, %67 : vector<1x4xf32>
        %69 = vector.extract %63[%46, %47] : f32 from vector<2x8xf32>
        %70 = vector.extract %63[%46, %50] : f32 from vector<2x8xf32>
        %71 = vector.extract %63[%46, %53] : f32 from vector<2x8xf32>
        %72 = vector.extract %63[%46, %56] : f32 from vector<2x8xf32>
        %73 = vector.from_elements %69, %70, %71, %72 : vector<1x4xf32>
        %74 = arith.mulf %68, %73 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %75 = vector.extract %74[0, 0] : f32 from vector<1x4xf32>
        %76 = vector.insert %75, %arg14 [%46, %47] : f32 into vector<2x8xf32>
        %77 = vector.extract %74[0, 1] : f32 from vector<1x4xf32>
        %78 = vector.insert %77, %76 [%46, %50] : f32 into vector<2x8xf32>
        %79 = vector.extract %74[0, 2] : f32 from vector<1x4xf32>
        %80 = vector.insert %79, %78 [%46, %53] : f32 into vector<2x8xf32>
        %81 = vector.extract %74[0, 3] : f32 from vector<1x4xf32>
        %82 = vector.insert %81, %80 [%46, %56] : f32 into vector<2x8xf32>
        %83 = vector.extract %82[%46, %47] : f32 from vector<2x8xf32>
        %84 = vector.extract %82[%46, %50] : f32 from vector<2x8xf32>
        %85 = vector.extract %82[%46, %53] : f32 from vector<2x8xf32>
        %86 = vector.extract %82[%46, %56] : f32 from vector<2x8xf32>
        %87 = vector.from_elements %83, %84, %85, %86 : vector<1x4xf32>
        %88 = vector.extract %58[%46, %47] : f32 from vector<2x8xf32>
        %89 = vector.extract %58[%46, %50] : f32 from vector<2x8xf32>
        %90 = vector.extract %58[%46, %53] : f32 from vector<2x8xf32>
        %91 = vector.extract %58[%46, %56] : f32 from vector<2x8xf32>
        %92 = vector.from_elements %88, %89, %90, %91 : vector<1x4xf32>
        %93 = arith.addf %87, %92 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %94 = vector.extract %93[0, 0] : f32 from vector<1x4xf32>
        %95 = vector.insert %94, %arg15 [%46, %47] : f32 into vector<2x8xf32>
        %96 = vector.extract %93[0, 1] : f32 from vector<1x4xf32>
        %97 = vector.insert %96, %95 [%46, %50] : f32 into vector<2x8xf32>
        %98 = vector.extract %93[0, 2] : f32 from vector<1x4xf32>
        %99 = vector.insert %98, %97 [%46, %53] : f32 into vector<2x8xf32>
        %100 = vector.extract %93[0, 3] : f32 from vector<1x4xf32>
        %101 = vector.insert %100, %99 [%46, %56] : f32 into vector<2x8xf32>
        affine.yield %58, %63, %82, %101 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_13 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_14 = memref.subview %arg2[0, %34, %arg4, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_14 to %alloc_13 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_13 : memref<128x32xf16, 3>
      %39 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %45 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_10]
        %46 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_10]
        %47 = frisk.read_matrix_from_shm %alloc_13[%45, %46] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %48 = vector.bitcast %47 : vector<4xi32> to vector<8xf16>
        %49 = vector.extract %48[0] : f16 from vector<8xf16>
        %50 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_10]
        %51 = vector.extract %48[%50] : f16 from vector<8xf16>
        %52 = vector.extract %48[1] : f16 from vector<8xf16>
        %53 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_10]
        %54 = vector.extract %48[%53] : f16 from vector<8xf16>
        %55 = vector.extract %48[2] : f16 from vector<8xf16>
        %56 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_10]
        %57 = vector.extract %48[%56] : f16 from vector<8xf16>
        %58 = vector.extract %48[3] : f16 from vector<8xf16>
        %59 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_10]
        %60 = vector.extract %48[%59] : f16 from vector<8xf16>
        %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %62 = vector.insert %49, %arg12 [%61, 0] : f16 into vector<32x2xf16>
        %63 = vector.insert %51, %62 [%61, 1] : f16 into vector<32x2xf16>
        %64 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %65 = vector.insert %52, %63 [%64, 0] : f16 into vector<32x2xf16>
        %66 = vector.insert %54, %65 [%64, 1] : f16 into vector<32x2xf16>
        %67 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %68 = vector.insert %55, %66 [%67, 0] : f16 into vector<32x2xf16>
        %69 = vector.insert %57, %68 [%67, 1] : f16 into vector<32x2xf16>
        %70 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %71 = vector.insert %58, %69 [%70, 0] : f16 into vector<32x2xf16>
        %72 = vector.insert %60, %71 [%70, 1] : f16 into vector<32x2xf16>
        affine.yield %72 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %40:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_4) -> (vector<1x4xf32>) {
          %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %103 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %104 = vector.extract %33[%102, %103] : f16 from vector<2x32xf16>
          %105 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %106 = vector.extract %33[%102, %105] : f16 from vector<2x32xf16>
          %107 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %108 = vector.extract %33[%102, %107] : f16 from vector<2x32xf16>
          %109 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %110 = vector.extract %33[%102, %109] : f16 from vector<2x32xf16>
          %111 = vector.from_elements %104, %106, %108, %110 : vector<1x4xf16>
          %112 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %113 = vector.extract %39[%103, %112] : f16 from vector<32x2xf16>
          %114 = vector.extract %39[%105, %112] : f16 from vector<32x2xf16>
          %115 = vector.extract %39[%107, %112] : f16 from vector<32x2xf16>
          %116 = vector.extract %39[%109, %112] : f16 from vector<32x2xf16>
          %117 = vector.from_elements %113, %114, %115, %116 : vector<4x1xf16>
          %118 = frisk.warp_mma_rr %111, %117, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %118 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %46 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %47 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %48 = vector.extract %45[0, 0] : f32 from vector<1x4xf32>
        %49 = vector.insert %48, %arg12 [%46, %47] : f32 into vector<2x8xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %51 = vector.extract %45[0, 1] : f32 from vector<1x4xf32>
        %52 = vector.insert %51, %49 [%46, %50] : f32 into vector<2x8xf32>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %54 = vector.extract %45[0, 2] : f32 from vector<1x4xf32>
        %55 = vector.insert %54, %52 [%46, %53] : f32 into vector<2x8xf32>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %57 = vector.extract %45[0, 3] : f32 from vector<1x4xf32>
        %58 = vector.insert %57, %55 [%46, %56] : f32 into vector<2x8xf32>
        %59 = vector.extract %30#1[%46, 0] : f32 from vector<2x1xf32>
        %60 = vector.insert %59, %arg13 [%46, %47] : f32 into vector<2x8xf32>
        %61 = vector.insert %59, %60 [%46, %50] : f32 into vector<2x8xf32>
        %62 = vector.insert %59, %61 [%46, %53] : f32 into vector<2x8xf32>
        %63 = vector.insert %59, %62 [%46, %56] : f32 into vector<2x8xf32>
        %64 = vector.extract %arg9[%46, %47] : f32 from vector<2x8xf32>
        %65 = vector.extract %arg9[%46, %50] : f32 from vector<2x8xf32>
        %66 = vector.extract %arg9[%46, %53] : f32 from vector<2x8xf32>
        %67 = vector.extract %arg9[%46, %56] : f32 from vector<2x8xf32>
        %68 = vector.from_elements %64, %65, %66, %67 : vector<1x4xf32>
        %69 = vector.extract %63[%46, %47] : f32 from vector<2x8xf32>
        %70 = vector.extract %63[%46, %50] : f32 from vector<2x8xf32>
        %71 = vector.extract %63[%46, %53] : f32 from vector<2x8xf32>
        %72 = vector.extract %63[%46, %56] : f32 from vector<2x8xf32>
        %73 = vector.from_elements %69, %70, %71, %72 : vector<1x4xf32>
        %74 = arith.mulf %68, %73 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %75 = vector.extract %74[0, 0] : f32 from vector<1x4xf32>
        %76 = vector.insert %75, %arg14 [%46, %47] : f32 into vector<2x8xf32>
        %77 = vector.extract %74[0, 1] : f32 from vector<1x4xf32>
        %78 = vector.insert %77, %76 [%46, %50] : f32 into vector<2x8xf32>
        %79 = vector.extract %74[0, 2] : f32 from vector<1x4xf32>
        %80 = vector.insert %79, %78 [%46, %53] : f32 into vector<2x8xf32>
        %81 = vector.extract %74[0, 3] : f32 from vector<1x4xf32>
        %82 = vector.insert %81, %80 [%46, %56] : f32 into vector<2x8xf32>
        %83 = vector.extract %82[%46, %47] : f32 from vector<2x8xf32>
        %84 = vector.extract %82[%46, %50] : f32 from vector<2x8xf32>
        %85 = vector.extract %82[%46, %53] : f32 from vector<2x8xf32>
        %86 = vector.extract %82[%46, %56] : f32 from vector<2x8xf32>
        %87 = vector.from_elements %83, %84, %85, %86 : vector<1x4xf32>
        %88 = vector.extract %58[%46, %47] : f32 from vector<2x8xf32>
        %89 = vector.extract %58[%46, %50] : f32 from vector<2x8xf32>
        %90 = vector.extract %58[%46, %53] : f32 from vector<2x8xf32>
        %91 = vector.extract %58[%46, %56] : f32 from vector<2x8xf32>
        %92 = vector.from_elements %88, %89, %90, %91 : vector<1x4xf32>
        %93 = arith.addf %87, %92 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %94 = vector.extract %93[0, 0] : f32 from vector<1x4xf32>
        %95 = vector.insert %94, %arg15 [%46, %47] : f32 into vector<2x8xf32>
        %96 = vector.extract %93[0, 1] : f32 from vector<1x4xf32>
        %97 = vector.insert %96, %95 [%46, %50] : f32 into vector<2x8xf32>
        %98 = vector.extract %93[0, 2] : f32 from vector<1x4xf32>
        %99 = vector.insert %98, %97 [%46, %53] : f32 into vector<2x8xf32>
        %100 = vector.extract %93[0, 3] : f32 from vector<1x4xf32>
        %101 = vector.insert %100, %99 [%46, %56] : f32 into vector<2x8xf32>
        affine.yield %58, %63, %82, %101 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_16 = memref.subview %arg2[0, %34, %arg4, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_16 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_15 : memref<128x32xf16, 3>
      %41 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %45 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_10]
        %46 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_10]
        %47 = frisk.read_matrix_from_shm %alloc_15[%45, %46] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %48 = vector.bitcast %47 : vector<4xi32> to vector<8xf16>
        %49 = vector.extract %48[0] : f16 from vector<8xf16>
        %50 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_10]
        %51 = vector.extract %48[%50] : f16 from vector<8xf16>
        %52 = vector.extract %48[1] : f16 from vector<8xf16>
        %53 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_10]
        %54 = vector.extract %48[%53] : f16 from vector<8xf16>
        %55 = vector.extract %48[2] : f16 from vector<8xf16>
        %56 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_10]
        %57 = vector.extract %48[%56] : f16 from vector<8xf16>
        %58 = vector.extract %48[3] : f16 from vector<8xf16>
        %59 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_10]
        %60 = vector.extract %48[%59] : f16 from vector<8xf16>
        %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %62 = vector.insert %49, %arg12 [%61, 0] : f16 into vector<32x2xf16>
        %63 = vector.insert %51, %62 [%61, 1] : f16 into vector<32x2xf16>
        %64 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %65 = vector.insert %52, %63 [%64, 0] : f16 into vector<32x2xf16>
        %66 = vector.insert %54, %65 [%64, 1] : f16 into vector<32x2xf16>
        %67 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %68 = vector.insert %55, %66 [%67, 0] : f16 into vector<32x2xf16>
        %69 = vector.insert %57, %68 [%67, 1] : f16 into vector<32x2xf16>
        %70 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %71 = vector.insert %58, %69 [%70, 0] : f16 into vector<32x2xf16>
        %72 = vector.insert %60, %71 [%70, 1] : f16 into vector<32x2xf16>
        affine.yield %72 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %42:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_4) -> (vector<1x4xf32>) {
          %110 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %111 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %112 = vector.extract %33[%110, %111] : f16 from vector<2x32xf16>
          %113 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %114 = vector.extract %33[%110, %113] : f16 from vector<2x32xf16>
          %115 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %116 = vector.extract %33[%110, %115] : f16 from vector<2x32xf16>
          %117 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %118 = vector.extract %33[%110, %117] : f16 from vector<2x32xf16>
          %119 = vector.from_elements %112, %114, %116, %118 : vector<1x4xf16>
          %120 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %121 = vector.extract %41[%111, %120] : f16 from vector<32x2xf16>
          %122 = vector.extract %41[%113, %120] : f16 from vector<32x2xf16>
          %123 = vector.extract %41[%115, %120] : f16 from vector<32x2xf16>
          %124 = vector.extract %41[%117, %120] : f16 from vector<32x2xf16>
          %125 = vector.from_elements %121, %122, %123, %124 : vector<4x1xf16>
          %126 = frisk.warp_mma_rr %119, %125, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %126 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %46 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %47 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %48 = vector.extract %45[0, 0] : f32 from vector<1x4xf32>
        %49 = vector.insert %48, %arg12 [%46, %47] : f32 into vector<2x8xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %51 = vector.extract %45[0, 1] : f32 from vector<1x4xf32>
        %52 = vector.insert %51, %49 [%46, %50] : f32 into vector<2x8xf32>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %54 = vector.extract %45[0, 2] : f32 from vector<1x4xf32>
        %55 = vector.insert %54, %52 [%46, %53] : f32 into vector<2x8xf32>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %57 = vector.extract %45[0, 3] : f32 from vector<1x4xf32>
        %58 = vector.insert %57, %55 [%46, %56] : f32 into vector<2x8xf32>
        %59 = vector.extract %30#1[%46, 0] : f32 from vector<2x1xf32>
        %60 = vector.insert %59, %arg13 [%46, %47] : f32 into vector<2x8xf32>
        %61 = vector.insert %59, %60 [%46, %50] : f32 into vector<2x8xf32>
        %62 = vector.insert %59, %61 [%46, %53] : f32 into vector<2x8xf32>
        %63 = vector.insert %59, %62 [%46, %56] : f32 into vector<2x8xf32>
        %64 = vector.extract %arg10[%46, %47] : f32 from vector<2x8xf32>
        %65 = vector.extract %arg10[%46, %50] : f32 from vector<2x8xf32>
        %66 = vector.extract %arg10[%46, %53] : f32 from vector<2x8xf32>
        %67 = vector.extract %arg10[%46, %56] : f32 from vector<2x8xf32>
        %68 = vector.from_elements %64, %65, %66, %67 : vector<1x4xf32>
        %69 = vector.extract %63[%46, %47] : f32 from vector<2x8xf32>
        %70 = vector.extract %63[%46, %50] : f32 from vector<2x8xf32>
        %71 = vector.extract %63[%46, %53] : f32 from vector<2x8xf32>
        %72 = vector.extract %63[%46, %56] : f32 from vector<2x8xf32>
        %73 = vector.from_elements %69, %70, %71, %72 : vector<1x4xf32>
        %74 = arith.mulf %68, %73 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %75 = vector.extract %74[0, 0] : f32 from vector<1x4xf32>
        %76 = vector.insert %75, %arg14 [%46, %47] : f32 into vector<2x8xf32>
        %77 = vector.extract %74[0, 1] : f32 from vector<1x4xf32>
        %78 = vector.insert %77, %76 [%46, %50] : f32 into vector<2x8xf32>
        %79 = vector.extract %74[0, 2] : f32 from vector<1x4xf32>
        %80 = vector.insert %79, %78 [%46, %53] : f32 into vector<2x8xf32>
        %81 = vector.extract %74[0, 3] : f32 from vector<1x4xf32>
        %82 = vector.insert %81, %80 [%46, %56] : f32 into vector<2x8xf32>
        %83 = vector.extract %82[%46, %47] : f32 from vector<2x8xf32>
        %84 = vector.extract %82[%46, %50] : f32 from vector<2x8xf32>
        %85 = vector.extract %82[%46, %53] : f32 from vector<2x8xf32>
        %86 = vector.extract %82[%46, %56] : f32 from vector<2x8xf32>
        %87 = vector.from_elements %83, %84, %85, %86 : vector<1x4xf32>
        %88 = vector.extract %58[%46, %47] : f32 from vector<2x8xf32>
        %89 = vector.extract %58[%46, %50] : f32 from vector<2x8xf32>
        %90 = vector.extract %58[%46, %53] : f32 from vector<2x8xf32>
        %91 = vector.extract %58[%46, %56] : f32 from vector<2x8xf32>
        %92 = vector.from_elements %88, %89, %90, %91 : vector<1x4xf32>
        %93 = arith.addf %87, %92 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %94 = vector.extract %93[0, 0] : f32 from vector<1x4xf32>
        %95 = vector.insert %94, %arg15 [%46, %47] : f32 into vector<2x8xf32>
        %96 = vector.extract %93[0, 1] : f32 from vector<1x4xf32>
        %97 = vector.insert %96, %95 [%46, %50] : f32 into vector<2x8xf32>
        %98 = vector.extract %93[0, 2] : f32 from vector<1x4xf32>
        %99 = vector.insert %98, %97 [%46, %53] : f32 into vector<2x8xf32>
        %100 = vector.extract %93[0, 3] : f32 from vector<1x4xf32>
        %101 = vector.insert %100, %99 [%46, %56] : f32 into vector<2x8xf32>
        %102 = vector.extract %36#3[%46, %47] : f32 from vector<2x8xf32>
        %103 = vector.extract %36#3[%46, %50] : f32 from vector<2x8xf32>
        %104 = vector.extract %36#3[%46, %53] : f32 from vector<2x8xf32>
        %105 = vector.extract %36#3[%46, %56] : f32 from vector<2x8xf32>
        %106 = vector.insert %102, %arg16 [%46, %47] : f32 into vector<2x8xf32>
        %107 = vector.insert %103, %106 [%46, %50] : f32 into vector<2x8xf32>
        %108 = vector.insert %104, %107 [%46, %53] : f32 into vector<2x8xf32>
        %109 = vector.insert %105, %108 [%46, %56] : f32 into vector<2x8xf32>
        affine.yield %58, %63, %82, %101, %109 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %43:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_7, %arg13 = %cst_7) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %45 = vector.extract %32#1[%arg11, 0] : f32 from vector<2x1xf32>
        %46 = vector.insert %45, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %47 = vector.extract %28#1[%arg11, 0] : f32 from vector<2x1xf32>
        %48 = vector.insert %47, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %46, %48 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %44:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %46 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %47 = vector.extract %38#3[%45, %46] : f32 from vector<2x8xf32>
        %48 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %49 = vector.extract %38#3[%45, %48] : f32 from vector<2x8xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %51 = vector.extract %38#3[%45, %50] : f32 from vector<2x8xf32>
        %52 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %53 = vector.extract %38#3[%45, %52] : f32 from vector<2x8xf32>
        %54 = vector.insert %47, %arg12 [%45, %46] : f32 into vector<2x8xf32>
        %55 = vector.insert %49, %54 [%45, %48] : f32 into vector<2x8xf32>
        %56 = vector.insert %51, %55 [%45, %50] : f32 into vector<2x8xf32>
        %57 = vector.insert %53, %56 [%45, %52] : f32 into vector<2x8xf32>
        %58 = vector.extract %40#3[%45, %46] : f32 from vector<2x8xf32>
        %59 = vector.extract %40#3[%45, %48] : f32 from vector<2x8xf32>
        %60 = vector.extract %40#3[%45, %50] : f32 from vector<2x8xf32>
        %61 = vector.extract %40#3[%45, %52] : f32 from vector<2x8xf32>
        %62 = vector.insert %58, %arg13 [%45, %46] : f32 into vector<2x8xf32>
        %63 = vector.insert %59, %62 [%45, %48] : f32 into vector<2x8xf32>
        %64 = vector.insert %60, %63 [%45, %50] : f32 into vector<2x8xf32>
        %65 = vector.insert %61, %64 [%45, %52] : f32 into vector<2x8xf32>
        %66 = vector.extract %42#3[%45, %46] : f32 from vector<2x8xf32>
        %67 = vector.extract %42#3[%45, %48] : f32 from vector<2x8xf32>
        %68 = vector.extract %42#3[%45, %50] : f32 from vector<2x8xf32>
        %69 = vector.extract %42#3[%45, %52] : f32 from vector<2x8xf32>
        %70 = vector.insert %66, %arg14 [%45, %46] : f32 into vector<2x8xf32>
        %71 = vector.insert %67, %70 [%45, %48] : f32 into vector<2x8xf32>
        %72 = vector.insert %68, %71 [%45, %50] : f32 into vector<2x8xf32>
        %73 = vector.insert %69, %72 [%45, %52] : f32 into vector<2x8xf32>
        affine.yield %57, %65, %73 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %42#4, %43#0, %43#1, %44#0, %44#1, %44#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {local_yield_transformed = true}
    %thread_id_x = gpu.thread_id x
    %14 = arith.index_cast %thread_id_x : index to i32
    %15 = arith.andi %14, %c16_i32 : i32
    %16 = arith.cmpi ne, %15, %c0_i32 : i32
    %17 = arith.andi %14, %c32_i32 : i32
    %18 = arith.cmpi ne, %17, %c0_i32 : i32
    %19:8 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_0, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_6, %arg12 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %26 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %27 = vector.extract %13#5[%25, %26] : f32 from vector<2x8xf32>
      %28 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %29 = vector.extract %13#5[%25, %28] : f32 from vector<2x8xf32>
      %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %31 = vector.extract %13#5[%25, %30] : f32 from vector<2x8xf32>
      %32 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %33 = vector.extract %13#5[%25, %32] : f32 from vector<2x8xf32>
      %34 = arith.select %16, %27, %29 : f32
      %shuffleResult, %valid = gpu.shuffle xor %34, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %35 = arith.select %16, %shuffleResult, %27 : f32
      %36 = arith.select %16, %29, %shuffleResult : f32
      %37 = arith.select %16, %31, %33 : f32
      %shuffleResult_10, %valid_11 = gpu.shuffle xor %37, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %38 = arith.select %16, %shuffleResult_10, %31 : f32
      %39 = arith.select %16, %33, %shuffleResult_10 : f32
      %40 = arith.select %18, %35, %38 : f32
      %shuffleResult_12, %valid_13 = gpu.shuffle xor %40, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %41 = arith.select %18, %shuffleResult_12, %35 : f32
      %42 = arith.select %18, %38, %shuffleResult_12 : f32
      %43 = arith.select %18, %36, %39 : f32
      %shuffleResult_14, %valid_15 = gpu.shuffle xor %43, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %44 = arith.select %18, %shuffleResult_14, %36 : f32
      %45 = arith.select %18, %39, %shuffleResult_14 : f32
      %46 = vector.insert %41, %arg5 [%25, %26] : f32 into vector<2x8xf32>
      %47 = vector.insert %44, %46 [%25, %28] : f32 into vector<2x8xf32>
      %48 = vector.insert %42, %47 [%25, %30] : f32 into vector<2x8xf32>
      %49 = vector.insert %45, %48 [%25, %32] : f32 into vector<2x8xf32>
      %50 = vector.extract %13#4[%25, %26] : f32 from vector<2x8xf32>
      %51 = vector.extract %13#4[%25, %28] : f32 from vector<2x8xf32>
      %52 = vector.extract %13#4[%25, %30] : f32 from vector<2x8xf32>
      %53 = vector.extract %13#4[%25, %32] : f32 from vector<2x8xf32>
      %54 = arith.select %16, %50, %51 : f32
      %shuffleResult_16, %valid_17 = gpu.shuffle xor %54, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %55 = arith.select %16, %shuffleResult_16, %50 : f32
      %56 = arith.select %16, %51, %shuffleResult_16 : f32
      %57 = arith.select %16, %52, %53 : f32
      %shuffleResult_18, %valid_19 = gpu.shuffle xor %57, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %58 = arith.select %16, %shuffleResult_18, %52 : f32
      %59 = arith.select %16, %53, %shuffleResult_18 : f32
      %60 = arith.select %18, %55, %58 : f32
      %shuffleResult_20, %valid_21 = gpu.shuffle xor %60, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %61 = arith.select %18, %shuffleResult_20, %55 : f32
      %62 = arith.select %18, %58, %shuffleResult_20 : f32
      %63 = arith.select %18, %56, %59 : f32
      %shuffleResult_22, %valid_23 = gpu.shuffle xor %63, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %64 = arith.select %18, %shuffleResult_22, %56 : f32
      %65 = arith.select %18, %59, %shuffleResult_22 : f32
      %66 = vector.insert %61, %arg6 [%25, %26] : f32 into vector<2x8xf32>
      %67 = vector.insert %64, %66 [%25, %28] : f32 into vector<2x8xf32>
      %68 = vector.insert %62, %67 [%25, %30] : f32 into vector<2x8xf32>
      %69 = vector.insert %65, %68 [%25, %32] : f32 into vector<2x8xf32>
      %70 = vector.extract %13#3[%25, %26] : f32 from vector<2x8xf32>
      %71 = vector.extract %13#3[%25, %28] : f32 from vector<2x8xf32>
      %72 = vector.extract %13#3[%25, %30] : f32 from vector<2x8xf32>
      %73 = vector.extract %13#3[%25, %32] : f32 from vector<2x8xf32>
      %74 = arith.select %16, %70, %71 : f32
      %shuffleResult_24, %valid_25 = gpu.shuffle xor %74, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %75 = arith.select %16, %shuffleResult_24, %70 : f32
      %76 = arith.select %16, %71, %shuffleResult_24 : f32
      %77 = arith.select %16, %72, %73 : f32
      %shuffleResult_26, %valid_27 = gpu.shuffle xor %77, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %78 = arith.select %16, %shuffleResult_26, %72 : f32
      %79 = arith.select %16, %73, %shuffleResult_26 : f32
      %80 = arith.select %18, %75, %78 : f32
      %shuffleResult_28, %valid_29 = gpu.shuffle xor %80, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %81 = arith.select %18, %shuffleResult_28, %75 : f32
      %82 = arith.select %18, %78, %shuffleResult_28 : f32
      %83 = arith.select %18, %76, %79 : f32
      %shuffleResult_30, %valid_31 = gpu.shuffle xor %83, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %84 = arith.select %18, %shuffleResult_30, %76 : f32
      %85 = arith.select %18, %79, %shuffleResult_30 : f32
      %86 = vector.insert %81, %arg7 [%25, %26] : f32 into vector<2x8xf32>
      %87 = vector.insert %84, %86 [%25, %28] : f32 into vector<2x8xf32>
      %88 = vector.insert %82, %87 [%25, %30] : f32 into vector<2x8xf32>
      %89 = vector.insert %85, %88 [%25, %32] : f32 into vector<2x8xf32>
      %90 = vector.extract %13#0[%25, %26] : f32 from vector<2x8xf32>
      %91 = vector.extract %13#0[%25, %28] : f32 from vector<2x8xf32>
      %92 = vector.extract %13#0[%25, %30] : f32 from vector<2x8xf32>
      %93 = vector.extract %13#0[%25, %32] : f32 from vector<2x8xf32>
      %94 = arith.select %16, %90, %91 : f32
      %shuffleResult_32, %valid_33 = gpu.shuffle xor %94, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %95 = arith.select %16, %shuffleResult_32, %90 : f32
      %96 = arith.select %16, %91, %shuffleResult_32 : f32
      %97 = arith.select %16, %92, %93 : f32
      %shuffleResult_34, %valid_35 = gpu.shuffle xor %97, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %98 = arith.select %16, %shuffleResult_34, %92 : f32
      %99 = arith.select %16, %93, %shuffleResult_34 : f32
      %100 = arith.select %18, %95, %98 : f32
      %shuffleResult_36, %valid_37 = gpu.shuffle xor %100, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %101 = arith.select %18, %shuffleResult_36, %95 : f32
      %102 = arith.select %18, %98, %shuffleResult_36 : f32
      %103 = arith.select %18, %96, %99 : f32
      %shuffleResult_38, %valid_39 = gpu.shuffle xor %103, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %104 = arith.select %18, %shuffleResult_38, %96 : f32
      %105 = arith.select %18, %99, %shuffleResult_38 : f32
      %106 = vector.insert %101, %arg8 [%25, %26] : f32 into vector<2x8xf32>
      %107 = vector.insert %104, %106 [%25, %28] : f32 into vector<2x8xf32>
      %108 = vector.insert %102, %107 [%25, %30] : f32 into vector<2x8xf32>
      %109 = vector.insert %105, %108 [%25, %32] : f32 into vector<2x8xf32>
      %110 = vector.extract %13#1[%25, 0] : f32 from vector<2x1xf32>
      %111 = vector.insert %110, %arg9 [%25, %26] : f32 into vector<2x8xf32>
      %112 = vector.insert %110, %111 [%25, %28] : f32 into vector<2x8xf32>
      %113 = vector.insert %110, %112 [%25, %30] : f32 into vector<2x8xf32>
      %114 = vector.insert %110, %113 [%25, %32] : f32 into vector<2x8xf32>
      %115 = vector.extract %109[%25, %26] : f32 from vector<2x8xf32>
      %116 = vector.extract %109[%25, %28] : f32 from vector<2x8xf32>
      %117 = vector.extract %109[%25, %30] : f32 from vector<2x8xf32>
      %118 = vector.extract %109[%25, %32] : f32 from vector<2x8xf32>
      %119 = vector.from_elements %115, %116, %117, %118 : vector<1x4xf32>
      %120 = vector.extract %114[%25, %26] : f32 from vector<2x8xf32>
      %121 = vector.extract %114[%25, %28] : f32 from vector<2x8xf32>
      %122 = vector.extract %114[%25, %30] : f32 from vector<2x8xf32>
      %123 = vector.extract %114[%25, %32] : f32 from vector<2x8xf32>
      %124 = vector.from_elements %120, %121, %122, %123 : vector<1x4xf32>
      %125 = arith.divf %119, %124 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %126 = vector.extract %125[0, 0] : f32 from vector<1x4xf32>
      %127 = vector.insert %126, %arg10 [%25, %26] : f32 into vector<2x8xf32>
      %128 = vector.extract %125[0, 1] : f32 from vector<1x4xf32>
      %129 = vector.insert %128, %127 [%25, %28] : f32 into vector<2x8xf32>
      %130 = vector.extract %125[0, 2] : f32 from vector<1x4xf32>
      %131 = vector.insert %130, %129 [%25, %30] : f32 into vector<2x8xf32>
      %132 = vector.extract %125[0, 3] : f32 from vector<1x4xf32>
      %133 = vector.insert %132, %131 [%25, %32] : f32 into vector<2x8xf32>
      %134 = vector.extract %133[%25, %26] : f32 from vector<2x8xf32>
      %135 = vector.extract %133[%25, %28] : f32 from vector<2x8xf32>
      %136 = vector.extract %133[%25, %30] : f32 from vector<2x8xf32>
      %137 = vector.extract %133[%25, %32] : f32 from vector<2x8xf32>
      %138 = vector.from_elements %134, %135, %136, %137 : vector<1x4xf32>
      %139 = arith.truncf %138 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %140 = vector.extract %139[0, 0] : f16 from vector<1x4xf16>
      %141 = vector.insert %140, %arg11 [%25, %26] : f16 into vector<2x8xf16>
      %142 = vector.extract %139[0, 1] : f16 from vector<1x4xf16>
      %143 = vector.insert %142, %141 [%25, %28] : f16 into vector<2x8xf16>
      %144 = vector.extract %139[0, 2] : f16 from vector<1x4xf16>
      %145 = vector.insert %144, %143 [%25, %30] : f16 into vector<2x8xf16>
      %146 = vector.extract %139[0, 3] : f16 from vector<1x4xf16>
      %147 = vector.insert %146, %145 [%25, %32] : f16 into vector<2x8xf16>
      %148 = vector.extract %147[%25, %26] : f16 from vector<2x8xf16>
      %149 = vector.extract %147[%25, %28] : f16 from vector<2x8xf16>
      %150 = vector.extract %147[%25, %30] : f16 from vector<2x8xf16>
      %151 = vector.extract %147[%25, %32] : f16 from vector<2x8xf16>
      %152 = vector.insert %148, %arg12 [%25, %26] : f16 into vector<2x8xf16>
      %153 = vector.insert %149, %152 [%25, %28] : f16 into vector<2x8xf16>
      %154 = vector.insert %150, %153 [%25, %30] : f16 into vector<2x8xf16>
      %155 = vector.insert %151, %154 [%25, %32] : f16 into vector<2x8xf16>
      affine.yield %49, %69, %89, %109, %114, %133, %147, %155 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    %20 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %21 = arith.cmpi ult, %20, %c4 : index
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %26 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %27 = vector.extract %19#7[%25, %26] : f16 from vector<2x8xf16>
        %28 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %29 = vector.extract %19#7[%25, %28] : f16 from vector<2x8xf16>
        %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %31 = vector.extract %19#7[%25, %30] : f16 from vector<2x8xf16>
        %32 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %33 = vector.extract %19#7[%25, %32] : f16 from vector<2x8xf16>
        affine.store %27, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        affine.store %29, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        affine.store %31, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        affine.store %33, %arg3[0, (symbol(%7) + symbol(%block_id_z) * 32 + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %22:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %26 = vector.extract %13#1[%25, 0] : f32 from vector<2x1xf32>
      %27 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %28 = vector.insert %26, %arg5 [%25, %27] : f32 into vector<2x8xf32>
      %29 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %30 = vector.insert %26, %28 [%25, %29] : f32 into vector<2x8xf32>
      %31 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %32 = vector.insert %26, %30 [%25, %31] : f32 into vector<2x8xf32>
      %33 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %34 = vector.insert %26, %32 [%25, %33] : f32 into vector<2x8xf32>
      %35 = vector.extract %19#2[%25, %27] : f32 from vector<2x8xf32>
      %36 = vector.extract %19#2[%25, %29] : f32 from vector<2x8xf32>
      %37 = vector.extract %19#2[%25, %31] : f32 from vector<2x8xf32>
      %38 = vector.extract %19#2[%25, %33] : f32 from vector<2x8xf32>
      %39 = vector.from_elements %35, %36, %37, %38 : vector<1x4xf32>
      %40 = vector.extract %34[%25, %27] : f32 from vector<2x8xf32>
      %41 = vector.extract %34[%25, %29] : f32 from vector<2x8xf32>
      %42 = vector.extract %34[%25, %31] : f32 from vector<2x8xf32>
      %43 = vector.extract %34[%25, %33] : f32 from vector<2x8xf32>
      %44 = vector.from_elements %40, %41, %42, %43 : vector<1x4xf32>
      %45 = arith.divf %39, %44 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %46 = vector.extract %45[0, 0] : f32 from vector<1x4xf32>
      %47 = vector.insert %46, %arg6 [%25, %27] : f32 into vector<2x8xf32>
      %48 = vector.extract %45[0, 1] : f32 from vector<1x4xf32>
      %49 = vector.insert %48, %47 [%25, %29] : f32 into vector<2x8xf32>
      %50 = vector.extract %45[0, 2] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %49 [%25, %31] : f32 into vector<2x8xf32>
      %52 = vector.extract %45[0, 3] : f32 from vector<1x4xf32>
      %53 = vector.insert %52, %51 [%25, %33] : f32 into vector<2x8xf32>
      %54 = vector.extract %53[%25, %27] : f32 from vector<2x8xf32>
      %55 = vector.extract %53[%25, %29] : f32 from vector<2x8xf32>
      %56 = vector.extract %53[%25, %31] : f32 from vector<2x8xf32>
      %57 = vector.extract %53[%25, %33] : f32 from vector<2x8xf32>
      %58 = vector.from_elements %54, %55, %56, %57 : vector<1x4xf32>
      %59 = arith.truncf %58 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %60 = vector.extract %59[0, 0] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %arg7 [%25, %27] : f16 into vector<2x8xf16>
      %62 = vector.extract %59[0, 1] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%25, %29] : f16 into vector<2x8xf16>
      %64 = vector.extract %59[0, 2] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%25, %31] : f16 into vector<2x8xf16>
      %66 = vector.extract %59[0, 3] : f16 from vector<1x4xf16>
      %67 = vector.insert %66, %65 [%25, %33] : f16 into vector<2x8xf16>
      %68 = vector.extract %67[%25, %27] : f16 from vector<2x8xf16>
      %69 = vector.extract %67[%25, %29] : f16 from vector<2x8xf16>
      %70 = vector.extract %67[%25, %31] : f16 from vector<2x8xf16>
      %71 = vector.extract %67[%25, %33] : f16 from vector<2x8xf16>
      %72 = vector.insert %68, %arg8 [%25, %27] : f16 into vector<2x8xf16>
      %73 = vector.insert %69, %72 [%25, %29] : f16 into vector<2x8xf16>
      %74 = vector.insert %70, %73 [%25, %31] : f16 into vector<2x8xf16>
      %75 = vector.insert %71, %74 [%25, %33] : f16 into vector<2x8xf16>
      affine.yield %34, %53, %67, %75 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %26 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %27 = vector.extract %22#3[%25, %26] : f16 from vector<2x8xf16>
        %28 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %29 = vector.extract %22#3[%25, %28] : f16 from vector<2x8xf16>
        %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %31 = vector.extract %22#3[%25, %30] : f16 from vector<2x8xf16>
        %32 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %33 = vector.extract %22#3[%25, %32] : f16 from vector<2x8xf16>
        affine.store %27, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        affine.store %29, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        affine.store %31, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        affine.store %33, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %23:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %26 = vector.extract %13#1[%25, 0] : f32 from vector<2x1xf32>
      %27 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %28 = vector.insert %26, %arg5 [%25, %27] : f32 into vector<2x8xf32>
      %29 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %30 = vector.insert %26, %28 [%25, %29] : f32 into vector<2x8xf32>
      %31 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %32 = vector.insert %26, %30 [%25, %31] : f32 into vector<2x8xf32>
      %33 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %34 = vector.insert %26, %32 [%25, %33] : f32 into vector<2x8xf32>
      %35 = vector.extract %19#1[%25, %27] : f32 from vector<2x8xf32>
      %36 = vector.extract %19#1[%25, %29] : f32 from vector<2x8xf32>
      %37 = vector.extract %19#1[%25, %31] : f32 from vector<2x8xf32>
      %38 = vector.extract %19#1[%25, %33] : f32 from vector<2x8xf32>
      %39 = vector.from_elements %35, %36, %37, %38 : vector<1x4xf32>
      %40 = vector.extract %34[%25, %27] : f32 from vector<2x8xf32>
      %41 = vector.extract %34[%25, %29] : f32 from vector<2x8xf32>
      %42 = vector.extract %34[%25, %31] : f32 from vector<2x8xf32>
      %43 = vector.extract %34[%25, %33] : f32 from vector<2x8xf32>
      %44 = vector.from_elements %40, %41, %42, %43 : vector<1x4xf32>
      %45 = arith.divf %39, %44 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %46 = vector.extract %45[0, 0] : f32 from vector<1x4xf32>
      %47 = vector.insert %46, %arg6 [%25, %27] : f32 into vector<2x8xf32>
      %48 = vector.extract %45[0, 1] : f32 from vector<1x4xf32>
      %49 = vector.insert %48, %47 [%25, %29] : f32 into vector<2x8xf32>
      %50 = vector.extract %45[0, 2] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %49 [%25, %31] : f32 into vector<2x8xf32>
      %52 = vector.extract %45[0, 3] : f32 from vector<1x4xf32>
      %53 = vector.insert %52, %51 [%25, %33] : f32 into vector<2x8xf32>
      %54 = vector.extract %53[%25, %27] : f32 from vector<2x8xf32>
      %55 = vector.extract %53[%25, %29] : f32 from vector<2x8xf32>
      %56 = vector.extract %53[%25, %31] : f32 from vector<2x8xf32>
      %57 = vector.extract %53[%25, %33] : f32 from vector<2x8xf32>
      %58 = vector.from_elements %54, %55, %56, %57 : vector<1x4xf32>
      %59 = arith.truncf %58 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %60 = vector.extract %59[0, 0] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %arg7 [%25, %27] : f16 into vector<2x8xf16>
      %62 = vector.extract %59[0, 1] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%25, %29] : f16 into vector<2x8xf16>
      %64 = vector.extract %59[0, 2] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%25, %31] : f16 into vector<2x8xf16>
      %66 = vector.extract %59[0, 3] : f16 from vector<1x4xf16>
      %67 = vector.insert %66, %65 [%25, %33] : f16 into vector<2x8xf16>
      %68 = vector.extract %67[%25, %27] : f16 from vector<2x8xf16>
      %69 = vector.extract %67[%25, %29] : f16 from vector<2x8xf16>
      %70 = vector.extract %67[%25, %31] : f16 from vector<2x8xf16>
      %71 = vector.extract %67[%25, %33] : f16 from vector<2x8xf16>
      %72 = vector.insert %68, %arg8 [%25, %27] : f16 into vector<2x8xf16>
      %73 = vector.insert %69, %72 [%25, %29] : f16 into vector<2x8xf16>
      %74 = vector.insert %70, %73 [%25, %31] : f16 into vector<2x8xf16>
      %75 = vector.insert %71, %74 [%25, %33] : f16 into vector<2x8xf16>
      affine.yield %34, %53, %67, %75 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %26 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %27 = vector.extract %23#3[%25, %26] : f16 from vector<2x8xf16>
        %28 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %29 = vector.extract %23#3[%25, %28] : f16 from vector<2x8xf16>
        %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %31 = vector.extract %23#3[%25, %30] : f16 from vector<2x8xf16>
        %32 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %33 = vector.extract %23#3[%25, %32] : f16 from vector<2x8xf16>
        affine.store %27, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        affine.store %29, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        affine.store %31, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        affine.store %33, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %24:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %26 = vector.extract %13#1[%25, 0] : f32 from vector<2x1xf32>
      %27 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %28 = vector.insert %26, %arg5 [%25, %27] : f32 into vector<2x8xf32>
      %29 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %30 = vector.insert %26, %28 [%25, %29] : f32 into vector<2x8xf32>
      %31 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %32 = vector.insert %26, %30 [%25, %31] : f32 into vector<2x8xf32>
      %33 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %34 = vector.insert %26, %32 [%25, %33] : f32 into vector<2x8xf32>
      %35 = vector.extract %19#0[%25, %27] : f32 from vector<2x8xf32>
      %36 = vector.extract %19#0[%25, %29] : f32 from vector<2x8xf32>
      %37 = vector.extract %19#0[%25, %31] : f32 from vector<2x8xf32>
      %38 = vector.extract %19#0[%25, %33] : f32 from vector<2x8xf32>
      %39 = vector.from_elements %35, %36, %37, %38 : vector<1x4xf32>
      %40 = vector.extract %34[%25, %27] : f32 from vector<2x8xf32>
      %41 = vector.extract %34[%25, %29] : f32 from vector<2x8xf32>
      %42 = vector.extract %34[%25, %31] : f32 from vector<2x8xf32>
      %43 = vector.extract %34[%25, %33] : f32 from vector<2x8xf32>
      %44 = vector.from_elements %40, %41, %42, %43 : vector<1x4xf32>
      %45 = arith.divf %39, %44 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %46 = vector.extract %45[0, 0] : f32 from vector<1x4xf32>
      %47 = vector.insert %46, %arg6 [%25, %27] : f32 into vector<2x8xf32>
      %48 = vector.extract %45[0, 1] : f32 from vector<1x4xf32>
      %49 = vector.insert %48, %47 [%25, %29] : f32 into vector<2x8xf32>
      %50 = vector.extract %45[0, 2] : f32 from vector<1x4xf32>
      %51 = vector.insert %50, %49 [%25, %31] : f32 into vector<2x8xf32>
      %52 = vector.extract %45[0, 3] : f32 from vector<1x4xf32>
      %53 = vector.insert %52, %51 [%25, %33] : f32 into vector<2x8xf32>
      %54 = vector.extract %53[%25, %27] : f32 from vector<2x8xf32>
      %55 = vector.extract %53[%25, %29] : f32 from vector<2x8xf32>
      %56 = vector.extract %53[%25, %31] : f32 from vector<2x8xf32>
      %57 = vector.extract %53[%25, %33] : f32 from vector<2x8xf32>
      %58 = vector.from_elements %54, %55, %56, %57 : vector<1x4xf32>
      %59 = arith.truncf %58 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %60 = vector.extract %59[0, 0] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %arg7 [%25, %27] : f16 into vector<2x8xf16>
      %62 = vector.extract %59[0, 1] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%25, %29] : f16 into vector<2x8xf16>
      %64 = vector.extract %59[0, 2] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%25, %31] : f16 into vector<2x8xf16>
      %66 = vector.extract %59[0, 3] : f16 from vector<1x4xf16>
      %67 = vector.insert %66, %65 [%25, %33] : f16 into vector<2x8xf16>
      %68 = vector.extract %67[%25, %27] : f16 from vector<2x8xf16>
      %69 = vector.extract %67[%25, %29] : f16 from vector<2x8xf16>
      %70 = vector.extract %67[%25, %31] : f16 from vector<2x8xf16>
      %71 = vector.extract %67[%25, %33] : f16 from vector<2x8xf16>
      %72 = vector.insert %68, %arg8 [%25, %27] : f16 into vector<2x8xf16>
      %73 = vector.insert %69, %72 [%25, %29] : f16 into vector<2x8xf16>
      %74 = vector.insert %70, %73 [%25, %31] : f16 into vector<2x8xf16>
      %75 = vector.insert %71, %74 [%25, %33] : f16 into vector<2x8xf16>
      affine.yield %34, %53, %67, %75 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %26 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %27 = vector.extract %24#3[%25, %26] : f16 from vector<2x8xf16>
        %28 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %29 = vector.extract %24#3[%25, %28] : f16 from vector<2x8xf16>
        %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %31 = vector.extract %24#3[%25, %30] : f16 from vector<2x8xf16>
        %32 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %33 = vector.extract %24#3[%25, %32] : f16 from vector<2x8xf16>
        affine.store %27, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        affine.store %29, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        affine.store %31, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        affine.store %33, %arg3[0, (symbol(%7) + symbol(%6) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%6) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%6) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
