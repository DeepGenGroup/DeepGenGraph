module {
  func.func @Attn(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> {
    %cst = arith.constant dense<1.131250e+01> : tensor<1xf16>
    %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFC00 : f16
    %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16>
    %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16>
    %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf16>, tensor<1xf16>) -> tensor<1x32x4096x4096xf16>
    %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf16>, tensor<4096x4096xf16>) -> tensor<1x32x4096x4096xf16>
    %7 = deepgengraph.convert %6, type = f32 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32>
    %8 = deepgengraph.exp %7 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %9 = deepgengraph.reduce(%8), dim = -1, op = ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %10 = deepgengraph.div %8, %9 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %11 = deepgengraph.convert %10, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
    %12 = deepgengraph.dot %11, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16>
    %13 = deepgengraph.permute %12, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16>
    return %13 : tensor<1x4096x32x128xf16>
  }
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1>, %arg1: memref<1x32x4096x128xf16, 1>, %arg2: memref<1x32x4096x128xf16, 1>, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
    %cst = arith.constant 0.127531052 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %cst_2 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %c127 = arith.constant 127 : index
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_6 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() alignment = 16 {frisk.shm_pool, tiled = true} : memref<8192xi8, 3>
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_z = gpu.block_id z {range = 1 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %block_id_x = gpu.block_id x {range = 32 : index}
    %0 = arith.truncf %cst : f32 to f16
    %1 = arith.muli %block_id_y, %c128 : index
    %2 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8) -> (vector<2x8xf32>) {
      %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %37 = vector.insert %cst_1, %arg5 [%36, %35] : f32 into vector<2x8xf32>
      %38 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %39 = vector.insert %cst_1, %37 [%36, %38] : f32 into vector<2x8xf32>
      %40 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %41 = vector.insert %cst_1, %39 [%36, %40] : f32 into vector<2x8xf32>
      %42 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %43 = vector.insert %cst_1, %41 [%36, %42] : f32 into vector<2x8xf32>
      affine.yield %43 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %3 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_5) -> (vector<2x1xf32>) {
      %35 = vector.insert %cst_1, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %35 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    %4:3 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %37 = vector.insert %cst_1, %arg5 [%36, %35] : f32 into vector<2x8xf32>
      %38 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %39 = vector.insert %cst_1, %37 [%36, %38] : f32 into vector<2x8xf32>
      %40 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %41 = vector.insert %cst_1, %39 [%36, %40] : f32 into vector<2x8xf32>
      %42 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %43 = vector.insert %cst_1, %41 [%36, %42] : f32 into vector<2x8xf32>
      %44 = vector.insert %cst_1, %arg6 [%36, %35] : f32 into vector<2x8xf32>
      %45 = vector.insert %cst_1, %44 [%36, %38] : f32 into vector<2x8xf32>
      %46 = vector.insert %cst_1, %45 [%36, %40] : f32 into vector<2x8xf32>
      %47 = vector.insert %cst_1, %46 [%36, %42] : f32 into vector<2x8xf32>
      %48 = vector.insert %cst_1, %arg7 [%36, %35] : f32 into vector<2x8xf32>
      %49 = vector.insert %cst_1, %48 [%36, %38] : f32 into vector<2x8xf32>
      %50 = vector.insert %cst_1, %49 [%36, %40] : f32 into vector<2x8xf32>
      %51 = vector.insert %cst_1, %50 [%36, %42] : f32 into vector<2x8xf32>
      affine.yield %43, %47, %51 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %alloc_9 = memref.alloc() alignment = 16 {frisk.packed_axis = 1 : i64, frisk.packed_shape = array<i64: 128, 32>, frisk.packed_width = 8 : i64, tiled = true} : memref<4096xf16, 3>
    %alloc_10 = memref.alloc() alignment = 16 {frisk.packed_axis = 1 : i64, frisk.packed_shape = array<i64: 128, 32>, frisk.packed_width = 8 : i64, tiled = true} : memref<4096xf16, 3>
    %alloc_11 = memref.alloc() alignment = 16 {frisk.packed_axis = 0 : i64, frisk.packed_shape = array<i64: 32, 128>, frisk.packed_width = 8 : i64, tiled = true} : memref<4096xf16, 3>
    %alloc_12 = memref.alloc() alignment = 16 {frisk.packed_axis = 0 : i64, frisk.packed_shape = array<i64: 32, 128>, frisk.packed_width = 8 : i64, tiled = true} : memref<4096xf16, 3>
    %5 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 + s2 * 32 + s0 floordiv 32) mod 32)>()[%block_id_y, %block_id_x, %block_id_z]
    %6 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%block_id_y]
    %subview = memref.subview %arg0[0, %5, %6, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %7 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_3) -> (vector<2x32xf32>) {
      %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %36 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg4)
      %37 = vector.insert %cst_1, %arg5 [%36, %35] : f32 into vector<2x32xf32>
      %38 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %39 = vector.insert %cst_1, %37 [%36, %38] : f32 into vector<2x32xf32>
      %40 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %41 = vector.insert %cst_1, %39 [%36, %40] : f32 into vector<2x32xf32>
      %42 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %43 = vector.insert %cst_1, %41 [%36, %42] : f32 into vector<2x32xf32>
      affine.yield %43 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
    %thread_id_x = gpu.thread_id x
    %8 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128)>()[%thread_id_x]
    %9 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 1)>()[%thread_id_x]
    %10 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2)>()[%thread_id_x]
    %11 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 3)>()[%thread_id_x]
    %12 = vector.broadcast %0 : f16 to vector<1x4xf16>
    %view = memref.view %alloc[%c0][] : memref<8192xi8, 3> to memref<4096xf16, 3>
    %13 = arith.addi %1, %c127 : index
    %14 = arith.addi %1, %c1 : index
    %15 = arith.addi %1, %c128 : index
    %16 = arith.cmpi uge, %15, %13 : index
    %17 = arith.cmpi uge, %13, %1 : index
    %18 = arith.andi %17, %16 : i1
    %19 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
    %20 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
    %21 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
    %22 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
    %23:5 = affine.for %arg4 = 0 to affine_map<()[s0] -> (s0 + 1)>()[%block_id_y] iter_args(%arg5 = %2, %arg6 = %3, %arg7 = %4#0, %arg8 = %4#1, %arg9 = %4#2) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %35 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      frisk.async_copy_gm_to_shm %subview to %alloc_9 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %36 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 floordiv 4096 + s0 + s1 * 32) mod 32)>(%arg4)[%block_id_x, %block_id_z]
      %37 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 128)>(%arg4)
      %38 = affine.apply affine_map<(d0) -> (d0 * 128 - (d0 floordiv 32) * 4096)>(%arg4)
      %subview_13 = memref.subview %transpose[0, %36, %37, %38] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_13 to %alloc_11 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
      %39 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %7) -> (vector<2x32xf32>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg10)
        %62 = arith.divui %61, %c32 : index
        %63 = arith.remui %62, %c2 : index
        %64 = arith.cmpi eq, %63, %c0 : index
        %65 = arith.select %64, %alloc_9, %alloc_10 : memref<4096xf16, 3>
        %66 = arith.select %64, %alloc_11, %alloc_12 : memref<4096xf16, 3>
        frisk.wait_buffer_ready %65 : memref<4096xf16, 3>
        frisk.wait_buffer_ready %66 : memref<4096xf16, 3>
        frisk.sync_threads_in_block
        %67 = arith.subi %c128, %61 : index
        %68 = arith.cmpi ugt, %67, %c32 : index
        scf.if %68 {
          %90 = arith.addi %63, %c1 : index
          %91 = arith.remui %90, %c2 : index
          %92 = arith.cmpi eq, %91, %c0 : index
          %93 = arith.select %92, %alloc_9, %alloc_10 : memref<4096xf16, 3>
          %94 = affine.apply affine_map<(d0)[s0, s1, s2] -> (((d0 + s0 * 512 + s1 * 16384 + s2 * 524288 + 1) floordiv 16384) mod 32)>(%arg10)[%block_id_y, %block_id_x, %block_id_z]
          %95 = affine.apply affine_map<(d0)[s0, s1, s2] -> (((d0 + 1) floordiv 4 + s0 * 128 + s1 * 4096 + s2 * 131072) mod 4096)>(%arg10)[%block_id_y, %block_id_x, %block_id_z]
          %96 = affine.apply affine_map<(d0) -> (d0 * 32 - ((d0 + 1) floordiv 4) * 128 + 32)>(%arg10)
          %subview_18 = memref.subview %arg0[0, %94, %95, %96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
          frisk.async_copy_gm_to_shm %subview_18 to %93 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
          %97 = arith.select %92, %alloc_11, %alloc_12 : memref<4096xf16, 3>
          %98 = affine.apply affine_map<(d0, d1)[s0, s1] -> (((d0 + d1 * 1024 + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32)>(%arg4, %arg10)[%block_id_x, %block_id_z]
          %99 = affine.apply affine_map<(d0, d1)[s0, s1] -> (d1 * 32 + s0 * 128 + s1 * 4096 + d0 floordiv 32 - ((d1 * 32 + s0 * 128 + s1 * 4096 + d0 floordiv 32 + 32) floordiv 128) * 128 + 32)>(%arg4, %arg10)[%block_id_x, %block_id_z]
          %subview_19 = memref.subview %transpose[0, %98, %99, %38] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
          frisk.async_copy_gm_to_shm %subview_19 to %97 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3>
        }
        %69 = memref.load %65[%8] : memref<4096xf16, 3>
        %70 = memref.load %65[%9] : memref<4096xf16, 3>
        %71 = memref.load %65[%10] : memref<4096xf16, 3>
        %72 = memref.load %65[%11] : memref<4096xf16, 3>
        %73 = vector.from_elements %69, %70, %71, %72 : vector<1x4xf16>
        %74:3 = affine.for %arg12 = 0 to 3 iter_args(%arg13 = %cst_2, %arg14 = %cst_2, %arg15 = %73) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
          %90 = affine.apply affine_map<(d0)[s0] -> (d0 * 2048 + ((d0 + 1) floordiv 2) * -3584 + s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>(%arg12)[%thread_id_x]
          %91 = memref.load %65[%90] : memref<4096xf16, 3>
          %92 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 17)>(%arg12)[%thread_id_x]
          %93 = memref.load %65[%92] : memref<4096xf16, 3>
          %94 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 8 - ((d0 + 1) floordiv 2) * 16 + ((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 18)>(%arg12)[%thread_id_x]
          %95 = memref.load %65[%94] : memref<4096xf16, 3>
          %96 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((d0 + 1) floordiv 2) * 480 + s0 * 8 + (s0 floordiv 64) * 128 + ((d0 * 16 - ((d0 + 1) floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 + ((s0 mod 64) floordiv 16) * 4 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 19)>(%arg12)[%thread_id_x]
          %97 = memref.load %65[%96] : memref<4096xf16, 3>
          %98 = vector.from_elements %91, %93, %95, %97 : vector<1x4xf16>
          %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg12)
          %100 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg12)
          %101 = vector.extract %arg15[0, 0] : f16 from vector<1x4xf16>
          %102 = vector.insert %101, %arg13 [%100, %99] : f16 into vector<2x8xf16>
          %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg12)
          %104 = vector.extract %arg15[0, 1] : f16 from vector<1x4xf16>
          %105 = vector.insert %104, %102 [%100, %103] : f16 into vector<2x8xf16>
          %106 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg12)
          %107 = vector.extract %arg15[0, 2] : f16 from vector<1x4xf16>
          %108 = vector.insert %107, %105 [%100, %106] : f16 into vector<2x8xf16>
          %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg12)
          %110 = vector.extract %arg15[0, 3] : f16 from vector<1x4xf16>
          %111 = vector.insert %110, %108 [%100, %109] : f16 into vector<2x8xf16>
          %112 = vector.extract %111[%100, %99] : f16 from vector<2x8xf16>
          %113 = vector.extract %111[%100, %103] : f16 from vector<2x8xf16>
          %114 = vector.extract %111[%100, %106] : f16 from vector<2x8xf16>
          %115 = vector.extract %111[%100, %109] : f16 from vector<2x8xf16>
          %116 = vector.from_elements %112, %113, %114, %115 : vector<1x4xf16>
          %117 = arith.mulf %116, %12 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
          %118 = vector.extract %117[0, 0] : f16 from vector<1x4xf16>
          %119 = vector.insert %118, %arg14 [%100, %99] : f16 into vector<2x8xf16>
          %120 = vector.extract %117[0, 1] : f16 from vector<1x4xf16>
          %121 = vector.insert %120, %119 [%100, %103] : f16 into vector<2x8xf16>
          %122 = vector.extract %117[0, 2] : f16 from vector<1x4xf16>
          %123 = vector.insert %122, %121 [%100, %106] : f16 into vector<2x8xf16>
          %124 = vector.extract %117[0, 3] : f16 from vector<1x4xf16>
          %125 = vector.insert %124, %123 [%100, %109] : f16 into vector<2x8xf16>
          affine.yield %111, %125, %98 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
        %75 = vector.extract %74#2[0, 0] : f16 from vector<1x4xf16>
        %76 = vector.extract %74#2[0, 1] : f16 from vector<1x4xf16>
        %77 = vector.extract %74#2[0, 2] : f16 from vector<1x4xf16>
        %78 = vector.extract %74#2[0, 3] : f16 from vector<1x4xf16>
        %79 = vector.from_elements %75, %76, %77, %78 : vector<1x4xf16>
        %80 = arith.mulf %79, %12 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
        %81 = vector.extract %80[0, 0] : f16 from vector<1x4xf16>
        %82 = vector.insert %81, %74#1 [1, 4] : f16 into vector<2x8xf16>
        %83 = vector.extract %80[0, 1] : f16 from vector<1x4xf16>
        %84 = vector.insert %83, %82 [1, 5] : f16 into vector<2x8xf16>
        %85 = vector.extract %80[0, 2] : f16 from vector<1x4xf16>
        %86 = vector.insert %85, %84 [1, 6] : f16 into vector<2x8xf16>
        %87 = vector.extract %80[0, 3] : f16 from vector<1x4xf16>
        %88 = vector.insert %87, %86 [1, 7] : f16 into vector<2x8xf16>
        %89:3 = affine.for %arg12 = 0 to 16 iter_args(%arg13 = %cst_3, %arg14 = %cst_3, %arg15 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>, vector<2x32xf32>) {
          %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg12)
          %91 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg12)
          %92 = vector.extract %88[%91, 0] : f16 from vector<2x8xf16>
          %93 = vector.extract %88[%91, 1] : f16 from vector<2x8xf16>
          %94 = vector.extract %88[%91, 2] : f16 from vector<2x8xf16>
          %95 = vector.extract %88[%91, 3] : f16 from vector<2x8xf16>
          %96 = vector.from_elements %92, %93, %94, %95 : vector<1x4xf16>
          %97 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>(%arg12)[%thread_id_x]
          %98 = memref.load %66[%97] : memref<4096xf16, 3>
          %99 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 1)>(%arg12)[%thread_id_x]
          %100 = memref.load %66[%99] : memref<4096xf16, 3>
          %101 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 2)>(%arg12)[%thread_id_x]
          %102 = memref.load %66[%101] : memref<4096xf16, 3>
          %103 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 3)>(%arg12)[%thread_id_x]
          %104 = memref.load %66[%103] : memref<4096xf16, 3>
          %105 = vector.from_elements %98, %100, %102, %104 : vector<4x1xf16>
          %106 = vector.extract %88[%91, 4] : f16 from vector<2x8xf16>
          %107 = vector.extract %88[%91, 5] : f16 from vector<2x8xf16>
          %108 = vector.extract %88[%91, 6] : f16 from vector<2x8xf16>
          %109 = vector.extract %88[%91, 7] : f16 from vector<2x8xf16>
          %110 = vector.from_elements %106, %107, %108, %109 : vector<1x4xf16>
          %111 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + 2048)>(%arg12)[%thread_id_x]
          %112 = memref.load %66[%111] : memref<4096xf16, 3>
          %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 17) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + 17)>(%arg12)[%thread_id_x]
          %114 = memref.load %66[%113] : memref<4096xf16, 3>
          %115 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 + 9) floordiv 4) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + 18)>(%arg12)[%thread_id_x]
          %116 = memref.load %66[%115] : memref<4096xf16, 3>
          %117 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 8) * -256 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 512 + ((d0 mod 8) floordiv 4) * -128 + s0 * 128 + ((s0 mod 64) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 + 19) floordiv 8) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + 19)>(%arg12)[%thread_id_x]
          %118 = memref.load %66[%117] : memref<4096xf16, 3>
          %119 = vector.from_elements %112, %114, %116, %118 : vector<4x1xf16>
          %120 = frisk.warp_mma_rr %96, %105, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          %121 = frisk.warp_mma_rr %110, %119, %120 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          %122 = vector.extract %121[0, 0] : f32 from vector<1x4xf32>
          %123 = vector.insert %122, %arg13 [%91, %90] : f32 into vector<2x32xf32>
          %124 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg12)
          %125 = vector.extract %121[0, 1] : f32 from vector<1x4xf32>
          %126 = vector.insert %125, %123 [%91, %124] : f32 into vector<2x32xf32>
          %127 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg12)
          %128 = vector.extract %121[0, 2] : f32 from vector<1x4xf32>
          %129 = vector.insert %128, %126 [%91, %127] : f32 into vector<2x32xf32>
          %130 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg12)
          %131 = vector.extract %121[0, 3] : f32 from vector<1x4xf32>
          %132 = vector.insert %131, %129 [%91, %130] : f32 into vector<2x32xf32>
          %133 = vector.extract %arg11[%91, %90] : f32 from vector<2x32xf32>
          %134 = vector.extract %arg11[%91, %124] : f32 from vector<2x32xf32>
          %135 = vector.extract %arg11[%91, %127] : f32 from vector<2x32xf32>
          %136 = vector.extract %arg11[%91, %130] : f32 from vector<2x32xf32>
          %137 = vector.from_elements %133, %134, %135, %136 : vector<1x4xf32>
          %138 = vector.extract %132[%91, %90] : f32 from vector<2x32xf32>
          %139 = vector.extract %132[%91, %124] : f32 from vector<2x32xf32>
          %140 = vector.extract %132[%91, %127] : f32 from vector<2x32xf32>
          %141 = vector.extract %132[%91, %130] : f32 from vector<2x32xf32>
          %142 = vector.from_elements %138, %139, %140, %141 : vector<1x4xf32>
          %143 = arith.addf %137, %142 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          %144 = vector.extract %143[0, 0] : f32 from vector<1x4xf32>
          %145 = vector.insert %144, %arg14 [%91, %90] : f32 into vector<2x32xf32>
          %146 = vector.extract %143[0, 1] : f32 from vector<1x4xf32>
          %147 = vector.insert %146, %145 [%91, %124] : f32 into vector<2x32xf32>
          %148 = vector.extract %143[0, 2] : f32 from vector<1x4xf32>
          %149 = vector.insert %148, %147 [%91, %127] : f32 into vector<2x32xf32>
          %150 = vector.extract %143[0, 3] : f32 from vector<1x4xf32>
          %151 = vector.insert %150, %149 [%91, %130] : f32 into vector<2x32xf32>
          %152 = vector.extract %151[%91, %90] : f32 from vector<2x32xf32>
          %153 = vector.extract %151[%91, %124] : f32 from vector<2x32xf32>
          %154 = vector.extract %151[%91, %127] : f32 from vector<2x32xf32>
          %155 = vector.extract %151[%91, %130] : f32 from vector<2x32xf32>
          %156 = vector.insert %152, %arg15 [%91, %90] : f32 into vector<2x32xf32>
          %157 = vector.insert %153, %156 [%91, %124] : f32 into vector<2x32xf32>
          %158 = vector.insert %154, %157 [%91, %127] : f32 into vector<2x32xf32>
          %159 = vector.insert %155, %158 [%91, %130] : f32 into vector<2x32xf32>
          affine.yield %132, %151, %159 : vector<2x32xf32>, vector<2x32xf32>, vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
        gpu.barrier
        affine.yield %89#2 : vector<2x32xf32>
      } {local_yield_transformed = true, pipeline.num_stages = 2 : i64, pipeline.scheduled, pipeline.slots = 2 : i64}
      %40 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 floordiv 32 + s0 + s1 * 32) mod 32)>(%arg4)[%block_id_x, %block_id_z]
      %subview_14 = memref.subview %arg2[0, %40, %38, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_14 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %41 = arith.addi %35, %c127 : index
      %42 = arith.cmpi uge, %41, %35 : index
      %43 = arith.andi %18, %42 : i1
      %44 = arith.cmpi ugt, %14, %41 : index
      %45 = arith.andi %43, %44 : i1
      %46 = scf.if %45 -> (vector<2x32xf32>) {
        scf.yield %cst_3 : vector<2x32xf32>
      } else {
        %61 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_3) -> (vector<2x32xf32>) {
          %62 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
          %63 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg10)
          %64 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 floordiv 8) * 64 + s0 + s1 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%arg10)[%thread_id_x, %1]
          %65 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 * 128 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + ((s0 mod 64) floordiv 16) * 16)>(%arg10, %arg4)[%thread_id_x]
          %66 = arith.addi %64, %c1 : index
          %67 = arith.cmpi ule, %66, %65 : index
          %68 = arith.select %67, %cst_0, %cst_1 : f32
          %69 = affine.apply affine_map<(d0, d1)[s0] -> (d1 * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 4) floordiv 64) * 64 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 4) floordiv 16) * 4 + (((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 4) floordiv 16) floordiv 4) * -16 + ((s0 mod 64) floordiv 16) * 16 + 1)>(%arg10, %arg4)[%thread_id_x]
          %70 = arith.cmpi ule, %66, %69 : index
          %71 = arith.select %70, %cst_0, %cst_1 : f32
          %72 = affine.apply affine_map<(d0, d1)[s0] -> (d1 * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 8) floordiv 64) * 64 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 8) floordiv 16) * 4 + (((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 8) floordiv 16) floordiv 4) * -16 + ((s0 mod 64) floordiv 16) * 16 + 2)>(%arg10, %arg4)[%thread_id_x]
          %73 = arith.cmpi ule, %66, %72 : index
          %74 = arith.select %73, %cst_0, %cst_1 : f32
          %75 = affine.apply affine_map<(d0, d1)[s0] -> (d1 * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 12) floordiv 64) * 64 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 12) floordiv 16) * 4 + (((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 16 + 12) floordiv 16) floordiv 4) * -16 + ((s0 mod 64) floordiv 16) * 16 + 3)>(%arg10, %arg4)[%thread_id_x]
          %76 = arith.cmpi ule, %66, %75 : index
          %77 = arith.select %76, %cst_0, %cst_1 : f32
          %78 = vector.insert %68, %arg11 [%63, %62] : f32 into vector<2x32xf32>
          %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %80 = vector.insert %71, %78 [%63, %79] : f32 into vector<2x32xf32>
          %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %82 = vector.insert %74, %80 [%63, %81] : f32 into vector<2x32xf32>
          %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %84 = vector.insert %77, %82 [%63, %83] : f32 into vector<2x32xf32>
          affine.yield %84 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "mask_fragment", tiled = true}
        scf.yield %61 : vector<2x32xf32>
      } {frisk.uniform_mask}
      %47:2 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_3, %arg12 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg10)
        %63 = vector.extract %39[%62, %61] : f32 from vector<2x32xf32>
        %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %65 = vector.extract %39[%62, %64] : f32 from vector<2x32xf32>
        %66 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %67 = vector.extract %39[%62, %66] : f32 from vector<2x32xf32>
        %68 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %69 = vector.extract %39[%62, %68] : f32 from vector<2x32xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = vector.extract %46[%62, %61] : f32 from vector<2x32xf32>
        %72 = vector.extract %46[%62, %64] : f32 from vector<2x32xf32>
        %73 = vector.extract %46[%62, %66] : f32 from vector<2x32xf32>
        %74 = vector.extract %46[%62, %68] : f32 from vector<2x32xf32>
        %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
        %76 = arith.addf %70, %75 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
        %78 = vector.insert %77, %arg11 [%62, %61] : f32 into vector<2x32xf32>
        %79 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
        %80 = vector.insert %79, %78 [%62, %64] : f32 into vector<2x32xf32>
        %81 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
        %82 = vector.insert %81, %80 [%62, %66] : f32 into vector<2x32xf32>
        %83 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
        %84 = vector.insert %83, %82 [%62, %68] : f32 into vector<2x32xf32>
        %85 = vector.extract %84[%62, %61] : f32 from vector<2x32xf32>
        %86 = vector.extract %84[%62, %64] : f32 from vector<2x32xf32>
        %87 = vector.extract %84[%62, %66] : f32 from vector<2x32xf32>
        %88 = vector.extract %84[%62, %68] : f32 from vector<2x32xf32>
        %89 = vector.from_elements %85, %86, %87, %88 : vector<1x4xf32>
        %90 = math.exp2 %89 {tiled = true} : vector<1x4xf32>
        %91 = vector.extract %90[0, 0] : f32 from vector<1x4xf32>
        %92 = vector.insert %91, %arg12 [%62, %61] : f32 into vector<2x32xf32>
        %93 = vector.extract %90[0, 1] : f32 from vector<1x4xf32>
        %94 = vector.insert %93, %92 [%62, %64] : f32 into vector<2x32xf32>
        %95 = vector.extract %90[0, 2] : f32 from vector<1x4xf32>
        %96 = vector.insert %95, %94 [%62, %66] : f32 into vector<2x32xf32>
        %97 = vector.extract %90[0, 3] : f32 from vector<1x4xf32>
        %98 = vector.insert %97, %96 [%62, %68] : f32 into vector<2x32xf32>
        affine.yield %84, %98 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %48:2 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_5, %arg12 = %cst_5) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %61 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_1) -> (f32) {
          %72 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
          %73 = vector.extract %47#1[%arg10, %72] : f32 from vector<2x32xf32>
          %74 = arith.addf %arg14, %73 : f32
          %75 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
          %76 = vector.extract %47#1[%arg10, %75] : f32 from vector<2x32xf32>
          %77 = arith.addf %74, %76 : f32
          %78 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
          %79 = vector.extract %47#1[%arg10, %78] : f32 from vector<2x32xf32>
          %80 = arith.addf %77, %79 : f32
          %81 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
          %82 = vector.extract %47#1[%arg10, %81] : f32 from vector<2x32xf32>
          %83 = arith.addf %80, %82 : f32
          affine.yield %83 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %61, %c16_i32, %c64_i32 : f32
        %62 = arith.addf %61, %shuffleResult : f32
        %shuffleResult_18, %valid_19 = gpu.shuffle xor %62, %c32_i32, %c64_i32 : f32
        %63 = arith.addf %62, %shuffleResult_18 : f32
        %64 = vector.insert %63, %arg11 [%arg10, 0] : f32 into vector<2x1xf32>
        %65 = vector.extract %64[%arg10, 0] : f32 from vector<2x1xf32>
        %66 = vector.broadcast %65 : f32 to vector<1x1xf32>
        %67 = vector.extract %arg6[%arg10, 0] : f32 from vector<2x1xf32>
        %68 = vector.broadcast %67 : f32 to vector<1x1xf32>
        %69 = arith.addf %66, %68 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %70 = vector.extract %69[0, 0] : f32 from vector<1x1xf32>
        %71 = vector.insert %70, %arg12 [%arg10, 0] : f32 into vector<2x1xf32>
        affine.yield %64, %71 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %49 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_6) -> (vector<2x32xf16>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg10)
        %63 = vector.extract %47#1[%62, %61] : f32 from vector<2x32xf32>
        %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %65 = vector.extract %47#1[%62, %64] : f32 from vector<2x32xf32>
        %66 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %67 = vector.extract %47#1[%62, %66] : f32 from vector<2x32xf32>
        %68 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %69 = vector.extract %47#1[%62, %68] : f32 from vector<2x32xf32>
        %70 = vector.from_elements %63, %65, %67, %69 : vector<1x4xf32>
        %71 = arith.truncf %70 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %72 = vector.extract %71[0, 0] : f16 from vector<1x4xf16>
        %73 = vector.insert %72, %arg11 [%62, %61] : f16 into vector<2x32xf16>
        %74 = vector.extract %71[0, 1] : f16 from vector<1x4xf16>
        %75 = vector.insert %74, %73 [%62, %64] : f16 into vector<2x32xf16>
        %76 = vector.extract %71[0, 2] : f16 from vector<1x4xf16>
        %77 = vector.insert %76, %75 [%62, %66] : f16 into vector<2x32xf16>
        %78 = vector.extract %71[0, 3] : f16 from vector<1x4xf16>
        %79 = vector.insert %78, %77 [%62, %68] : f16 into vector<2x32xf16>
        affine.yield %79 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %50 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_7) -> (vector<32x2xf16>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg10)
        %62 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg10)[%thread_id_x]
        %63 = frisk.read_matrix_from_shm %view[%62] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %64 = vector.bitcast %63 : vector<4xi32> to vector<8xf16>
        %65 = vector.extract %64[0] : f16 from vector<8xf16>
        %66 = vector.extract %64[%19] : f16 from vector<8xf16>
        %67 = vector.extract %64[1] : f16 from vector<8xf16>
        %68 = vector.extract %64[%20] : f16 from vector<8xf16>
        %69 = vector.extract %64[2] : f16 from vector<8xf16>
        %70 = vector.extract %64[%21] : f16 from vector<8xf16>
        %71 = vector.extract %64[3] : f16 from vector<8xf16>
        %72 = vector.extract %64[%22] : f16 from vector<8xf16>
        %73 = vector.insert %65, %arg11 [%61, 0] : f16 into vector<32x2xf16>
        %74 = vector.insert %66, %73 [%61, 1] : f16 into vector<32x2xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg10)
        %76 = vector.insert %67, %74 [%75, 0] : f16 into vector<32x2xf16>
        %77 = vector.insert %68, %76 [%75, 1] : f16 into vector<32x2xf16>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg10)
        %79 = vector.insert %69, %77 [%78, 0] : f16 into vector<32x2xf16>
        %80 = vector.insert %70, %79 [%78, 1] : f16 into vector<32x2xf16>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg10)
        %82 = vector.insert %71, %80 [%81, 0] : f16 into vector<32x2xf16>
        %83 = vector.insert %72, %82 [%81, 1] : f16 into vector<32x2xf16>
        affine.yield %83 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %51 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 32 + s0) mod 32)>(%arg4)[%block_id_x]
      %subview_15 = memref.subview %arg2[0, %51, %38, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      frisk.async_copy_gm_to_shm %subview_15 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %52:2 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_8, %arg12 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg10)
        %63 = vector.extract %49[%62, 0] : f16 from vector<2x32xf16>
        %64 = vector.extract %49[%62, 1] : f16 from vector<2x32xf16>
        %65 = vector.extract %49[%62, 2] : f16 from vector<2x32xf16>
        %66 = vector.extract %49[%62, 3] : f16 from vector<2x32xf16>
        %67 = vector.from_elements %63, %64, %65, %66 : vector<1x4xf16>
        %68 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %69 = vector.extract %50[0, %68] : f16 from vector<32x2xf16>
        %70 = vector.extract %50[1, %68] : f16 from vector<32x2xf16>
        %71 = vector.extract %50[2, %68] : f16 from vector<32x2xf16>
        %72 = vector.extract %50[3, %68] : f16 from vector<32x2xf16>
        %73 = vector.from_elements %69, %70, %71, %72 : vector<4x1xf16>
        %74 = frisk.warp_mma_rr %67, %73, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %75 = vector.extract %49[%62, 4] : f16 from vector<2x32xf16>
        %76 = vector.extract %49[%62, 5] : f16 from vector<2x32xf16>
        %77 = vector.extract %49[%62, 6] : f16 from vector<2x32xf16>
        %78 = vector.extract %49[%62, 7] : f16 from vector<2x32xf16>
        %79 = vector.from_elements %75, %76, %77, %78 : vector<1x4xf16>
        %80 = vector.extract %50[4, %68] : f16 from vector<32x2xf16>
        %81 = vector.extract %50[5, %68] : f16 from vector<32x2xf16>
        %82 = vector.extract %50[6, %68] : f16 from vector<32x2xf16>
        %83 = vector.extract %50[7, %68] : f16 from vector<32x2xf16>
        %84 = vector.from_elements %80, %81, %82, %83 : vector<4x1xf16>
        %85 = frisk.warp_mma_rr %79, %84, %74 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %86 = vector.extract %49[%62, 8] : f16 from vector<2x32xf16>
        %87 = vector.extract %49[%62, 9] : f16 from vector<2x32xf16>
        %88 = vector.extract %49[%62, 10] : f16 from vector<2x32xf16>
        %89 = vector.extract %49[%62, 11] : f16 from vector<2x32xf16>
        %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf16>
        %91 = vector.extract %50[8, %68] : f16 from vector<32x2xf16>
        %92 = vector.extract %50[9, %68] : f16 from vector<32x2xf16>
        %93 = vector.extract %50[10, %68] : f16 from vector<32x2xf16>
        %94 = vector.extract %50[11, %68] : f16 from vector<32x2xf16>
        %95 = vector.from_elements %91, %92, %93, %94 : vector<4x1xf16>
        %96 = frisk.warp_mma_rr %90, %95, %85 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %97 = vector.extract %49[%62, 12] : f16 from vector<2x32xf16>
        %98 = vector.extract %49[%62, 13] : f16 from vector<2x32xf16>
        %99 = vector.extract %49[%62, 14] : f16 from vector<2x32xf16>
        %100 = vector.extract %49[%62, 15] : f16 from vector<2x32xf16>
        %101 = vector.from_elements %97, %98, %99, %100 : vector<1x4xf16>
        %102 = vector.extract %50[12, %68] : f16 from vector<32x2xf16>
        %103 = vector.extract %50[13, %68] : f16 from vector<32x2xf16>
        %104 = vector.extract %50[14, %68] : f16 from vector<32x2xf16>
        %105 = vector.extract %50[15, %68] : f16 from vector<32x2xf16>
        %106 = vector.from_elements %102, %103, %104, %105 : vector<4x1xf16>
        %107 = frisk.warp_mma_rr %101, %106, %96 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %108 = vector.extract %49[%62, 16] : f16 from vector<2x32xf16>
        %109 = vector.extract %49[%62, 17] : f16 from vector<2x32xf16>
        %110 = vector.extract %49[%62, 18] : f16 from vector<2x32xf16>
        %111 = vector.extract %49[%62, 19] : f16 from vector<2x32xf16>
        %112 = vector.from_elements %108, %109, %110, %111 : vector<1x4xf16>
        %113 = vector.extract %50[16, %68] : f16 from vector<32x2xf16>
        %114 = vector.extract %50[17, %68] : f16 from vector<32x2xf16>
        %115 = vector.extract %50[18, %68] : f16 from vector<32x2xf16>
        %116 = vector.extract %50[19, %68] : f16 from vector<32x2xf16>
        %117 = vector.from_elements %113, %114, %115, %116 : vector<4x1xf16>
        %118 = frisk.warp_mma_rr %112, %117, %107 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %119 = vector.extract %49[%62, 20] : f16 from vector<2x32xf16>
        %120 = vector.extract %49[%62, 21] : f16 from vector<2x32xf16>
        %121 = vector.extract %49[%62, 22] : f16 from vector<2x32xf16>
        %122 = vector.extract %49[%62, 23] : f16 from vector<2x32xf16>
        %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf16>
        %124 = vector.extract %50[20, %68] : f16 from vector<32x2xf16>
        %125 = vector.extract %50[21, %68] : f16 from vector<32x2xf16>
        %126 = vector.extract %50[22, %68] : f16 from vector<32x2xf16>
        %127 = vector.extract %50[23, %68] : f16 from vector<32x2xf16>
        %128 = vector.from_elements %124, %125, %126, %127 : vector<4x1xf16>
        %129 = frisk.warp_mma_rr %123, %128, %118 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %130 = vector.extract %49[%62, 24] : f16 from vector<2x32xf16>
        %131 = vector.extract %49[%62, 25] : f16 from vector<2x32xf16>
        %132 = vector.extract %49[%62, 26] : f16 from vector<2x32xf16>
        %133 = vector.extract %49[%62, 27] : f16 from vector<2x32xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf16>
        %135 = vector.extract %50[24, %68] : f16 from vector<32x2xf16>
        %136 = vector.extract %50[25, %68] : f16 from vector<32x2xf16>
        %137 = vector.extract %50[26, %68] : f16 from vector<32x2xf16>
        %138 = vector.extract %50[27, %68] : f16 from vector<32x2xf16>
        %139 = vector.from_elements %135, %136, %137, %138 : vector<4x1xf16>
        %140 = frisk.warp_mma_rr %134, %139, %129 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %141 = vector.extract %49[%62, 28] : f16 from vector<2x32xf16>
        %142 = vector.extract %49[%62, 29] : f16 from vector<2x32xf16>
        %143 = vector.extract %49[%62, 30] : f16 from vector<2x32xf16>
        %144 = vector.extract %49[%62, 31] : f16 from vector<2x32xf16>
        %145 = vector.from_elements %141, %142, %143, %144 : vector<1x4xf16>
        %146 = vector.extract %50[28, %68] : f16 from vector<32x2xf16>
        %147 = vector.extract %50[29, %68] : f16 from vector<32x2xf16>
        %148 = vector.extract %50[30, %68] : f16 from vector<32x2xf16>
        %149 = vector.extract %50[31, %68] : f16 from vector<32x2xf16>
        %150 = vector.from_elements %146, %147, %148, %149 : vector<4x1xf16>
        %151 = frisk.warp_mma_rr %145, %150, %140 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg11 [%62, %61] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%62, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%62, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%62, %160] : f32 into vector<2x8xf32>
        %163 = vector.extract %arg5[%62, %61] : f32 from vector<2x8xf32>
        %164 = vector.extract %arg5[%62, %154] : f32 from vector<2x8xf32>
        %165 = vector.extract %arg5[%62, %157] : f32 from vector<2x8xf32>
        %166 = vector.extract %arg5[%62, %160] : f32 from vector<2x8xf32>
        %167 = vector.from_elements %163, %164, %165, %166 : vector<1x4xf32>
        %168 = vector.extract %162[%62, %61] : f32 from vector<2x8xf32>
        %169 = vector.extract %162[%62, %154] : f32 from vector<2x8xf32>
        %170 = vector.extract %162[%62, %157] : f32 from vector<2x8xf32>
        %171 = vector.extract %162[%62, %160] : f32 from vector<2x8xf32>
        %172 = vector.from_elements %168, %169, %170, %171 : vector<1x4xf32>
        %173 = arith.addf %167, %172 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %174 = vector.extract %173[0, 0] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %arg12 [%62, %61] : f32 into vector<2x8xf32>
        %176 = vector.extract %173[0, 1] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%62, %154] : f32 into vector<2x8xf32>
        %178 = vector.extract %173[0, 2] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%62, %157] : f32 into vector<2x8xf32>
        %180 = vector.extract %173[0, 3] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%62, %160] : f32 into vector<2x8xf32>
        affine.yield %162, %181 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %53 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_7) -> (vector<32x2xf16>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg10)
        %62 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg10)[%thread_id_x]
        %63 = frisk.read_matrix_from_shm %view[%62] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %64 = vector.bitcast %63 : vector<4xi32> to vector<8xf16>
        %65 = vector.extract %64[0] : f16 from vector<8xf16>
        %66 = vector.extract %64[%19] : f16 from vector<8xf16>
        %67 = vector.extract %64[1] : f16 from vector<8xf16>
        %68 = vector.extract %64[%20] : f16 from vector<8xf16>
        %69 = vector.extract %64[2] : f16 from vector<8xf16>
        %70 = vector.extract %64[%21] : f16 from vector<8xf16>
        %71 = vector.extract %64[3] : f16 from vector<8xf16>
        %72 = vector.extract %64[%22] : f16 from vector<8xf16>
        %73 = vector.insert %65, %arg11 [%61, 0] : f16 into vector<32x2xf16>
        %74 = vector.insert %66, %73 [%61, 1] : f16 into vector<32x2xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg10)
        %76 = vector.insert %67, %74 [%75, 0] : f16 into vector<32x2xf16>
        %77 = vector.insert %68, %76 [%75, 1] : f16 into vector<32x2xf16>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg10)
        %79 = vector.insert %69, %77 [%78, 0] : f16 into vector<32x2xf16>
        %80 = vector.insert %70, %79 [%78, 1] : f16 into vector<32x2xf16>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg10)
        %82 = vector.insert %71, %80 [%81, 0] : f16 into vector<32x2xf16>
        %83 = vector.insert %72, %82 [%81, 1] : f16 into vector<32x2xf16>
        affine.yield %83 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_16 = memref.subview %arg2[0, %51, %38, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      frisk.async_copy_gm_to_shm %subview_16 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %54:2 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_8, %arg12 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg10)
        %63 = vector.extract %49[%62, 0] : f16 from vector<2x32xf16>
        %64 = vector.extract %49[%62, 1] : f16 from vector<2x32xf16>
        %65 = vector.extract %49[%62, 2] : f16 from vector<2x32xf16>
        %66 = vector.extract %49[%62, 3] : f16 from vector<2x32xf16>
        %67 = vector.from_elements %63, %64, %65, %66 : vector<1x4xf16>
        %68 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %69 = vector.extract %53[0, %68] : f16 from vector<32x2xf16>
        %70 = vector.extract %53[1, %68] : f16 from vector<32x2xf16>
        %71 = vector.extract %53[2, %68] : f16 from vector<32x2xf16>
        %72 = vector.extract %53[3, %68] : f16 from vector<32x2xf16>
        %73 = vector.from_elements %69, %70, %71, %72 : vector<4x1xf16>
        %74 = frisk.warp_mma_rr %67, %73, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %75 = vector.extract %49[%62, 4] : f16 from vector<2x32xf16>
        %76 = vector.extract %49[%62, 5] : f16 from vector<2x32xf16>
        %77 = vector.extract %49[%62, 6] : f16 from vector<2x32xf16>
        %78 = vector.extract %49[%62, 7] : f16 from vector<2x32xf16>
        %79 = vector.from_elements %75, %76, %77, %78 : vector<1x4xf16>
        %80 = vector.extract %53[4, %68] : f16 from vector<32x2xf16>
        %81 = vector.extract %53[5, %68] : f16 from vector<32x2xf16>
        %82 = vector.extract %53[6, %68] : f16 from vector<32x2xf16>
        %83 = vector.extract %53[7, %68] : f16 from vector<32x2xf16>
        %84 = vector.from_elements %80, %81, %82, %83 : vector<4x1xf16>
        %85 = frisk.warp_mma_rr %79, %84, %74 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %86 = vector.extract %49[%62, 8] : f16 from vector<2x32xf16>
        %87 = vector.extract %49[%62, 9] : f16 from vector<2x32xf16>
        %88 = vector.extract %49[%62, 10] : f16 from vector<2x32xf16>
        %89 = vector.extract %49[%62, 11] : f16 from vector<2x32xf16>
        %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf16>
        %91 = vector.extract %53[8, %68] : f16 from vector<32x2xf16>
        %92 = vector.extract %53[9, %68] : f16 from vector<32x2xf16>
        %93 = vector.extract %53[10, %68] : f16 from vector<32x2xf16>
        %94 = vector.extract %53[11, %68] : f16 from vector<32x2xf16>
        %95 = vector.from_elements %91, %92, %93, %94 : vector<4x1xf16>
        %96 = frisk.warp_mma_rr %90, %95, %85 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %97 = vector.extract %49[%62, 12] : f16 from vector<2x32xf16>
        %98 = vector.extract %49[%62, 13] : f16 from vector<2x32xf16>
        %99 = vector.extract %49[%62, 14] : f16 from vector<2x32xf16>
        %100 = vector.extract %49[%62, 15] : f16 from vector<2x32xf16>
        %101 = vector.from_elements %97, %98, %99, %100 : vector<1x4xf16>
        %102 = vector.extract %53[12, %68] : f16 from vector<32x2xf16>
        %103 = vector.extract %53[13, %68] : f16 from vector<32x2xf16>
        %104 = vector.extract %53[14, %68] : f16 from vector<32x2xf16>
        %105 = vector.extract %53[15, %68] : f16 from vector<32x2xf16>
        %106 = vector.from_elements %102, %103, %104, %105 : vector<4x1xf16>
        %107 = frisk.warp_mma_rr %101, %106, %96 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %108 = vector.extract %49[%62, 16] : f16 from vector<2x32xf16>
        %109 = vector.extract %49[%62, 17] : f16 from vector<2x32xf16>
        %110 = vector.extract %49[%62, 18] : f16 from vector<2x32xf16>
        %111 = vector.extract %49[%62, 19] : f16 from vector<2x32xf16>
        %112 = vector.from_elements %108, %109, %110, %111 : vector<1x4xf16>
        %113 = vector.extract %53[16, %68] : f16 from vector<32x2xf16>
        %114 = vector.extract %53[17, %68] : f16 from vector<32x2xf16>
        %115 = vector.extract %53[18, %68] : f16 from vector<32x2xf16>
        %116 = vector.extract %53[19, %68] : f16 from vector<32x2xf16>
        %117 = vector.from_elements %113, %114, %115, %116 : vector<4x1xf16>
        %118 = frisk.warp_mma_rr %112, %117, %107 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %119 = vector.extract %49[%62, 20] : f16 from vector<2x32xf16>
        %120 = vector.extract %49[%62, 21] : f16 from vector<2x32xf16>
        %121 = vector.extract %49[%62, 22] : f16 from vector<2x32xf16>
        %122 = vector.extract %49[%62, 23] : f16 from vector<2x32xf16>
        %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf16>
        %124 = vector.extract %53[20, %68] : f16 from vector<32x2xf16>
        %125 = vector.extract %53[21, %68] : f16 from vector<32x2xf16>
        %126 = vector.extract %53[22, %68] : f16 from vector<32x2xf16>
        %127 = vector.extract %53[23, %68] : f16 from vector<32x2xf16>
        %128 = vector.from_elements %124, %125, %126, %127 : vector<4x1xf16>
        %129 = frisk.warp_mma_rr %123, %128, %118 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %130 = vector.extract %49[%62, 24] : f16 from vector<2x32xf16>
        %131 = vector.extract %49[%62, 25] : f16 from vector<2x32xf16>
        %132 = vector.extract %49[%62, 26] : f16 from vector<2x32xf16>
        %133 = vector.extract %49[%62, 27] : f16 from vector<2x32xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf16>
        %135 = vector.extract %53[24, %68] : f16 from vector<32x2xf16>
        %136 = vector.extract %53[25, %68] : f16 from vector<32x2xf16>
        %137 = vector.extract %53[26, %68] : f16 from vector<32x2xf16>
        %138 = vector.extract %53[27, %68] : f16 from vector<32x2xf16>
        %139 = vector.from_elements %135, %136, %137, %138 : vector<4x1xf16>
        %140 = frisk.warp_mma_rr %134, %139, %129 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %141 = vector.extract %49[%62, 28] : f16 from vector<2x32xf16>
        %142 = vector.extract %49[%62, 29] : f16 from vector<2x32xf16>
        %143 = vector.extract %49[%62, 30] : f16 from vector<2x32xf16>
        %144 = vector.extract %49[%62, 31] : f16 from vector<2x32xf16>
        %145 = vector.from_elements %141, %142, %143, %144 : vector<1x4xf16>
        %146 = vector.extract %53[28, %68] : f16 from vector<32x2xf16>
        %147 = vector.extract %53[29, %68] : f16 from vector<32x2xf16>
        %148 = vector.extract %53[30, %68] : f16 from vector<32x2xf16>
        %149 = vector.extract %53[31, %68] : f16 from vector<32x2xf16>
        %150 = vector.from_elements %146, %147, %148, %149 : vector<4x1xf16>
        %151 = frisk.warp_mma_rr %145, %150, %140 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg11 [%62, %61] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%62, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%62, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%62, %160] : f32 into vector<2x8xf32>
        %163 = vector.extract %arg7[%62, %61] : f32 from vector<2x8xf32>
        %164 = vector.extract %arg7[%62, %154] : f32 from vector<2x8xf32>
        %165 = vector.extract %arg7[%62, %157] : f32 from vector<2x8xf32>
        %166 = vector.extract %arg7[%62, %160] : f32 from vector<2x8xf32>
        %167 = vector.from_elements %163, %164, %165, %166 : vector<1x4xf32>
        %168 = vector.extract %162[%62, %61] : f32 from vector<2x8xf32>
        %169 = vector.extract %162[%62, %154] : f32 from vector<2x8xf32>
        %170 = vector.extract %162[%62, %157] : f32 from vector<2x8xf32>
        %171 = vector.extract %162[%62, %160] : f32 from vector<2x8xf32>
        %172 = vector.from_elements %168, %169, %170, %171 : vector<1x4xf32>
        %173 = arith.addf %167, %172 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %174 = vector.extract %173[0, 0] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %arg12 [%62, %61] : f32 into vector<2x8xf32>
        %176 = vector.extract %173[0, 1] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%62, %154] : f32 into vector<2x8xf32>
        %178 = vector.extract %173[0, 2] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%62, %157] : f32 into vector<2x8xf32>
        %180 = vector.extract %173[0, 3] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%62, %160] : f32 into vector<2x8xf32>
        affine.yield %162, %181 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %55 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_7) -> (vector<32x2xf16>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg10)
        %62 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg10)[%thread_id_x]
        %63 = frisk.read_matrix_from_shm %view[%62] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %64 = vector.bitcast %63 : vector<4xi32> to vector<8xf16>
        %65 = vector.extract %64[0] : f16 from vector<8xf16>
        %66 = vector.extract %64[%19] : f16 from vector<8xf16>
        %67 = vector.extract %64[1] : f16 from vector<8xf16>
        %68 = vector.extract %64[%20] : f16 from vector<8xf16>
        %69 = vector.extract %64[2] : f16 from vector<8xf16>
        %70 = vector.extract %64[%21] : f16 from vector<8xf16>
        %71 = vector.extract %64[3] : f16 from vector<8xf16>
        %72 = vector.extract %64[%22] : f16 from vector<8xf16>
        %73 = vector.insert %65, %arg11 [%61, 0] : f16 into vector<32x2xf16>
        %74 = vector.insert %66, %73 [%61, 1] : f16 into vector<32x2xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg10)
        %76 = vector.insert %67, %74 [%75, 0] : f16 into vector<32x2xf16>
        %77 = vector.insert %68, %76 [%75, 1] : f16 into vector<32x2xf16>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg10)
        %79 = vector.insert %69, %77 [%78, 0] : f16 into vector<32x2xf16>
        %80 = vector.insert %70, %79 [%78, 1] : f16 into vector<32x2xf16>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg10)
        %82 = vector.insert %71, %80 [%81, 0] : f16 into vector<32x2xf16>
        %83 = vector.insert %72, %82 [%81, 1] : f16 into vector<32x2xf16>
        affine.yield %83 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %subview_17 = memref.subview %arg2[0, %51, %38, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      frisk.async_copy_gm_to_shm %subview_17 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3>
      %56:2 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_8, %arg12 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg10)
        %63 = vector.extract %49[%62, 0] : f16 from vector<2x32xf16>
        %64 = vector.extract %49[%62, 1] : f16 from vector<2x32xf16>
        %65 = vector.extract %49[%62, 2] : f16 from vector<2x32xf16>
        %66 = vector.extract %49[%62, 3] : f16 from vector<2x32xf16>
        %67 = vector.from_elements %63, %64, %65, %66 : vector<1x4xf16>
        %68 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %69 = vector.extract %55[0, %68] : f16 from vector<32x2xf16>
        %70 = vector.extract %55[1, %68] : f16 from vector<32x2xf16>
        %71 = vector.extract %55[2, %68] : f16 from vector<32x2xf16>
        %72 = vector.extract %55[3, %68] : f16 from vector<32x2xf16>
        %73 = vector.from_elements %69, %70, %71, %72 : vector<4x1xf16>
        %74 = frisk.warp_mma_rr %67, %73, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %75 = vector.extract %49[%62, 4] : f16 from vector<2x32xf16>
        %76 = vector.extract %49[%62, 5] : f16 from vector<2x32xf16>
        %77 = vector.extract %49[%62, 6] : f16 from vector<2x32xf16>
        %78 = vector.extract %49[%62, 7] : f16 from vector<2x32xf16>
        %79 = vector.from_elements %75, %76, %77, %78 : vector<1x4xf16>
        %80 = vector.extract %55[4, %68] : f16 from vector<32x2xf16>
        %81 = vector.extract %55[5, %68] : f16 from vector<32x2xf16>
        %82 = vector.extract %55[6, %68] : f16 from vector<32x2xf16>
        %83 = vector.extract %55[7, %68] : f16 from vector<32x2xf16>
        %84 = vector.from_elements %80, %81, %82, %83 : vector<4x1xf16>
        %85 = frisk.warp_mma_rr %79, %84, %74 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %86 = vector.extract %49[%62, 8] : f16 from vector<2x32xf16>
        %87 = vector.extract %49[%62, 9] : f16 from vector<2x32xf16>
        %88 = vector.extract %49[%62, 10] : f16 from vector<2x32xf16>
        %89 = vector.extract %49[%62, 11] : f16 from vector<2x32xf16>
        %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf16>
        %91 = vector.extract %55[8, %68] : f16 from vector<32x2xf16>
        %92 = vector.extract %55[9, %68] : f16 from vector<32x2xf16>
        %93 = vector.extract %55[10, %68] : f16 from vector<32x2xf16>
        %94 = vector.extract %55[11, %68] : f16 from vector<32x2xf16>
        %95 = vector.from_elements %91, %92, %93, %94 : vector<4x1xf16>
        %96 = frisk.warp_mma_rr %90, %95, %85 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %97 = vector.extract %49[%62, 12] : f16 from vector<2x32xf16>
        %98 = vector.extract %49[%62, 13] : f16 from vector<2x32xf16>
        %99 = vector.extract %49[%62, 14] : f16 from vector<2x32xf16>
        %100 = vector.extract %49[%62, 15] : f16 from vector<2x32xf16>
        %101 = vector.from_elements %97, %98, %99, %100 : vector<1x4xf16>
        %102 = vector.extract %55[12, %68] : f16 from vector<32x2xf16>
        %103 = vector.extract %55[13, %68] : f16 from vector<32x2xf16>
        %104 = vector.extract %55[14, %68] : f16 from vector<32x2xf16>
        %105 = vector.extract %55[15, %68] : f16 from vector<32x2xf16>
        %106 = vector.from_elements %102, %103, %104, %105 : vector<4x1xf16>
        %107 = frisk.warp_mma_rr %101, %106, %96 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %108 = vector.extract %49[%62, 16] : f16 from vector<2x32xf16>
        %109 = vector.extract %49[%62, 17] : f16 from vector<2x32xf16>
        %110 = vector.extract %49[%62, 18] : f16 from vector<2x32xf16>
        %111 = vector.extract %49[%62, 19] : f16 from vector<2x32xf16>
        %112 = vector.from_elements %108, %109, %110, %111 : vector<1x4xf16>
        %113 = vector.extract %55[16, %68] : f16 from vector<32x2xf16>
        %114 = vector.extract %55[17, %68] : f16 from vector<32x2xf16>
        %115 = vector.extract %55[18, %68] : f16 from vector<32x2xf16>
        %116 = vector.extract %55[19, %68] : f16 from vector<32x2xf16>
        %117 = vector.from_elements %113, %114, %115, %116 : vector<4x1xf16>
        %118 = frisk.warp_mma_rr %112, %117, %107 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %119 = vector.extract %49[%62, 20] : f16 from vector<2x32xf16>
        %120 = vector.extract %49[%62, 21] : f16 from vector<2x32xf16>
        %121 = vector.extract %49[%62, 22] : f16 from vector<2x32xf16>
        %122 = vector.extract %49[%62, 23] : f16 from vector<2x32xf16>
        %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf16>
        %124 = vector.extract %55[20, %68] : f16 from vector<32x2xf16>
        %125 = vector.extract %55[21, %68] : f16 from vector<32x2xf16>
        %126 = vector.extract %55[22, %68] : f16 from vector<32x2xf16>
        %127 = vector.extract %55[23, %68] : f16 from vector<32x2xf16>
        %128 = vector.from_elements %124, %125, %126, %127 : vector<4x1xf16>
        %129 = frisk.warp_mma_rr %123, %128, %118 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %130 = vector.extract %49[%62, 24] : f16 from vector<2x32xf16>
        %131 = vector.extract %49[%62, 25] : f16 from vector<2x32xf16>
        %132 = vector.extract %49[%62, 26] : f16 from vector<2x32xf16>
        %133 = vector.extract %49[%62, 27] : f16 from vector<2x32xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf16>
        %135 = vector.extract %55[24, %68] : f16 from vector<32x2xf16>
        %136 = vector.extract %55[25, %68] : f16 from vector<32x2xf16>
        %137 = vector.extract %55[26, %68] : f16 from vector<32x2xf16>
        %138 = vector.extract %55[27, %68] : f16 from vector<32x2xf16>
        %139 = vector.from_elements %135, %136, %137, %138 : vector<4x1xf16>
        %140 = frisk.warp_mma_rr %134, %139, %129 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %141 = vector.extract %49[%62, 28] : f16 from vector<2x32xf16>
        %142 = vector.extract %49[%62, 29] : f16 from vector<2x32xf16>
        %143 = vector.extract %49[%62, 30] : f16 from vector<2x32xf16>
        %144 = vector.extract %49[%62, 31] : f16 from vector<2x32xf16>
        %145 = vector.from_elements %141, %142, %143, %144 : vector<1x4xf16>
        %146 = vector.extract %55[28, %68] : f16 from vector<32x2xf16>
        %147 = vector.extract %55[29, %68] : f16 from vector<32x2xf16>
        %148 = vector.extract %55[30, %68] : f16 from vector<32x2xf16>
        %149 = vector.extract %55[31, %68] : f16 from vector<32x2xf16>
        %150 = vector.from_elements %146, %147, %148, %149 : vector<4x1xf16>
        %151 = frisk.warp_mma_rr %145, %150, %140 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg11 [%62, %61] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%62, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%62, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%62, %160] : f32 into vector<2x8xf32>
        %163 = vector.extract %arg8[%62, %61] : f32 from vector<2x8xf32>
        %164 = vector.extract %arg8[%62, %154] : f32 from vector<2x8xf32>
        %165 = vector.extract %arg8[%62, %157] : f32 from vector<2x8xf32>
        %166 = vector.extract %arg8[%62, %160] : f32 from vector<2x8xf32>
        %167 = vector.from_elements %163, %164, %165, %166 : vector<1x4xf32>
        %168 = vector.extract %162[%62, %61] : f32 from vector<2x8xf32>
        %169 = vector.extract %162[%62, %154] : f32 from vector<2x8xf32>
        %170 = vector.extract %162[%62, %157] : f32 from vector<2x8xf32>
        %171 = vector.extract %162[%62, %160] : f32 from vector<2x8xf32>
        %172 = vector.from_elements %168, %169, %170, %171 : vector<1x4xf32>
        %173 = arith.addf %167, %172 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %174 = vector.extract %173[0, 0] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %arg12 [%62, %61] : f32 into vector<2x8xf32>
        %176 = vector.extract %173[0, 1] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%62, %154] : f32 into vector<2x8xf32>
        %178 = vector.extract %173[0, 2] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%62, %157] : f32 into vector<2x8xf32>
        %180 = vector.extract %173[0, 3] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%62, %160] : f32 into vector<2x8xf32>
        affine.yield %162, %181 : vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %view : memref<4096xf16, 3>
      %57 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_7) -> (vector<32x2xf16>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg10)
        %62 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 8 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 2048 + (d0 floordiv 4) * -512 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>(%arg10)[%thread_id_x]
        %63 = frisk.read_matrix_from_shm %view[%62] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
        %64 = vector.bitcast %63 : vector<4xi32> to vector<8xf16>
        %65 = vector.extract %64[0] : f16 from vector<8xf16>
        %66 = vector.extract %64[%19] : f16 from vector<8xf16>
        %67 = vector.extract %64[1] : f16 from vector<8xf16>
        %68 = vector.extract %64[%20] : f16 from vector<8xf16>
        %69 = vector.extract %64[2] : f16 from vector<8xf16>
        %70 = vector.extract %64[%21] : f16 from vector<8xf16>
        %71 = vector.extract %64[3] : f16 from vector<8xf16>
        %72 = vector.extract %64[%22] : f16 from vector<8xf16>
        %73 = vector.insert %65, %arg11 [%61, 0] : f16 into vector<32x2xf16>
        %74 = vector.insert %66, %73 [%61, 1] : f16 into vector<32x2xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg10)
        %76 = vector.insert %67, %74 [%75, 0] : f16 into vector<32x2xf16>
        %77 = vector.insert %68, %76 [%75, 1] : f16 into vector<32x2xf16>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg10)
        %79 = vector.insert %69, %77 [%78, 0] : f16 into vector<32x2xf16>
        %80 = vector.insert %70, %79 [%78, 1] : f16 into vector<32x2xf16>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg10)
        %82 = vector.insert %71, %80 [%81, 0] : f16 into vector<32x2xf16>
        %83 = vector.insert %72, %82 [%81, 1] : f16 into vector<32x2xf16>
        affine.yield %83 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %58:3 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_8, %arg12 = %cst_8, %arg13 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg10)
        %63 = vector.extract %49[%62, 0] : f16 from vector<2x32xf16>
        %64 = vector.extract %49[%62, 1] : f16 from vector<2x32xf16>
        %65 = vector.extract %49[%62, 2] : f16 from vector<2x32xf16>
        %66 = vector.extract %49[%62, 3] : f16 from vector<2x32xf16>
        %67 = vector.from_elements %63, %64, %65, %66 : vector<1x4xf16>
        %68 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %69 = vector.extract %57[0, %68] : f16 from vector<32x2xf16>
        %70 = vector.extract %57[1, %68] : f16 from vector<32x2xf16>
        %71 = vector.extract %57[2, %68] : f16 from vector<32x2xf16>
        %72 = vector.extract %57[3, %68] : f16 from vector<32x2xf16>
        %73 = vector.from_elements %69, %70, %71, %72 : vector<4x1xf16>
        %74 = frisk.warp_mma_rr %67, %73, %cst_4 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %75 = vector.extract %49[%62, 4] : f16 from vector<2x32xf16>
        %76 = vector.extract %49[%62, 5] : f16 from vector<2x32xf16>
        %77 = vector.extract %49[%62, 6] : f16 from vector<2x32xf16>
        %78 = vector.extract %49[%62, 7] : f16 from vector<2x32xf16>
        %79 = vector.from_elements %75, %76, %77, %78 : vector<1x4xf16>
        %80 = vector.extract %57[4, %68] : f16 from vector<32x2xf16>
        %81 = vector.extract %57[5, %68] : f16 from vector<32x2xf16>
        %82 = vector.extract %57[6, %68] : f16 from vector<32x2xf16>
        %83 = vector.extract %57[7, %68] : f16 from vector<32x2xf16>
        %84 = vector.from_elements %80, %81, %82, %83 : vector<4x1xf16>
        %85 = frisk.warp_mma_rr %79, %84, %74 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %86 = vector.extract %49[%62, 8] : f16 from vector<2x32xf16>
        %87 = vector.extract %49[%62, 9] : f16 from vector<2x32xf16>
        %88 = vector.extract %49[%62, 10] : f16 from vector<2x32xf16>
        %89 = vector.extract %49[%62, 11] : f16 from vector<2x32xf16>
        %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf16>
        %91 = vector.extract %57[8, %68] : f16 from vector<32x2xf16>
        %92 = vector.extract %57[9, %68] : f16 from vector<32x2xf16>
        %93 = vector.extract %57[10, %68] : f16 from vector<32x2xf16>
        %94 = vector.extract %57[11, %68] : f16 from vector<32x2xf16>
        %95 = vector.from_elements %91, %92, %93, %94 : vector<4x1xf16>
        %96 = frisk.warp_mma_rr %90, %95, %85 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %97 = vector.extract %49[%62, 12] : f16 from vector<2x32xf16>
        %98 = vector.extract %49[%62, 13] : f16 from vector<2x32xf16>
        %99 = vector.extract %49[%62, 14] : f16 from vector<2x32xf16>
        %100 = vector.extract %49[%62, 15] : f16 from vector<2x32xf16>
        %101 = vector.from_elements %97, %98, %99, %100 : vector<1x4xf16>
        %102 = vector.extract %57[12, %68] : f16 from vector<32x2xf16>
        %103 = vector.extract %57[13, %68] : f16 from vector<32x2xf16>
        %104 = vector.extract %57[14, %68] : f16 from vector<32x2xf16>
        %105 = vector.extract %57[15, %68] : f16 from vector<32x2xf16>
        %106 = vector.from_elements %102, %103, %104, %105 : vector<4x1xf16>
        %107 = frisk.warp_mma_rr %101, %106, %96 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %108 = vector.extract %49[%62, 16] : f16 from vector<2x32xf16>
        %109 = vector.extract %49[%62, 17] : f16 from vector<2x32xf16>
        %110 = vector.extract %49[%62, 18] : f16 from vector<2x32xf16>
        %111 = vector.extract %49[%62, 19] : f16 from vector<2x32xf16>
        %112 = vector.from_elements %108, %109, %110, %111 : vector<1x4xf16>
        %113 = vector.extract %57[16, %68] : f16 from vector<32x2xf16>
        %114 = vector.extract %57[17, %68] : f16 from vector<32x2xf16>
        %115 = vector.extract %57[18, %68] : f16 from vector<32x2xf16>
        %116 = vector.extract %57[19, %68] : f16 from vector<32x2xf16>
        %117 = vector.from_elements %113, %114, %115, %116 : vector<4x1xf16>
        %118 = frisk.warp_mma_rr %112, %117, %107 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %119 = vector.extract %49[%62, 20] : f16 from vector<2x32xf16>
        %120 = vector.extract %49[%62, 21] : f16 from vector<2x32xf16>
        %121 = vector.extract %49[%62, 22] : f16 from vector<2x32xf16>
        %122 = vector.extract %49[%62, 23] : f16 from vector<2x32xf16>
        %123 = vector.from_elements %119, %120, %121, %122 : vector<1x4xf16>
        %124 = vector.extract %57[20, %68] : f16 from vector<32x2xf16>
        %125 = vector.extract %57[21, %68] : f16 from vector<32x2xf16>
        %126 = vector.extract %57[22, %68] : f16 from vector<32x2xf16>
        %127 = vector.extract %57[23, %68] : f16 from vector<32x2xf16>
        %128 = vector.from_elements %124, %125, %126, %127 : vector<4x1xf16>
        %129 = frisk.warp_mma_rr %123, %128, %118 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %130 = vector.extract %49[%62, 24] : f16 from vector<2x32xf16>
        %131 = vector.extract %49[%62, 25] : f16 from vector<2x32xf16>
        %132 = vector.extract %49[%62, 26] : f16 from vector<2x32xf16>
        %133 = vector.extract %49[%62, 27] : f16 from vector<2x32xf16>
        %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf16>
        %135 = vector.extract %57[24, %68] : f16 from vector<32x2xf16>
        %136 = vector.extract %57[25, %68] : f16 from vector<32x2xf16>
        %137 = vector.extract %57[26, %68] : f16 from vector<32x2xf16>
        %138 = vector.extract %57[27, %68] : f16 from vector<32x2xf16>
        %139 = vector.from_elements %135, %136, %137, %138 : vector<4x1xf16>
        %140 = frisk.warp_mma_rr %134, %139, %129 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %141 = vector.extract %49[%62, 28] : f16 from vector<2x32xf16>
        %142 = vector.extract %49[%62, 29] : f16 from vector<2x32xf16>
        %143 = vector.extract %49[%62, 30] : f16 from vector<2x32xf16>
        %144 = vector.extract %49[%62, 31] : f16 from vector<2x32xf16>
        %145 = vector.from_elements %141, %142, %143, %144 : vector<1x4xf16>
        %146 = vector.extract %57[28, %68] : f16 from vector<32x2xf16>
        %147 = vector.extract %57[29, %68] : f16 from vector<32x2xf16>
        %148 = vector.extract %57[30, %68] : f16 from vector<32x2xf16>
        %149 = vector.extract %57[31, %68] : f16 from vector<32x2xf16>
        %150 = vector.from_elements %146, %147, %148, %149 : vector<4x1xf16>
        %151 = frisk.warp_mma_rr %145, %150, %140 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg11 [%62, %61] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%62, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%62, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%62, %160] : f32 into vector<2x8xf32>
        %163 = vector.extract %arg9[%62, %61] : f32 from vector<2x8xf32>
        %164 = vector.extract %arg9[%62, %154] : f32 from vector<2x8xf32>
        %165 = vector.extract %arg9[%62, %157] : f32 from vector<2x8xf32>
        %166 = vector.extract %arg9[%62, %160] : f32 from vector<2x8xf32>
        %167 = vector.from_elements %163, %164, %165, %166 : vector<1x4xf32>
        %168 = vector.extract %162[%62, %61] : f32 from vector<2x8xf32>
        %169 = vector.extract %162[%62, %154] : f32 from vector<2x8xf32>
        %170 = vector.extract %162[%62, %157] : f32 from vector<2x8xf32>
        %171 = vector.extract %162[%62, %160] : f32 from vector<2x8xf32>
        %172 = vector.from_elements %168, %169, %170, %171 : vector<1x4xf32>
        %173 = arith.addf %167, %172 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %174 = vector.extract %173[0, 0] : f32 from vector<1x4xf32>
        %175 = vector.insert %174, %arg12 [%62, %61] : f32 into vector<2x8xf32>
        %176 = vector.extract %173[0, 1] : f32 from vector<1x4xf32>
        %177 = vector.insert %176, %175 [%62, %154] : f32 into vector<2x8xf32>
        %178 = vector.extract %173[0, 2] : f32 from vector<1x4xf32>
        %179 = vector.insert %178, %177 [%62, %157] : f32 into vector<2x8xf32>
        %180 = vector.extract %173[0, 3] : f32 from vector<1x4xf32>
        %181 = vector.insert %180, %179 [%62, %160] : f32 into vector<2x8xf32>
        %182 = vector.extract %52#1[%62, %61] : f32 from vector<2x8xf32>
        %183 = vector.extract %52#1[%62, %154] : f32 from vector<2x8xf32>
        %184 = vector.extract %52#1[%62, %157] : f32 from vector<2x8xf32>
        %185 = vector.extract %52#1[%62, %160] : f32 from vector<2x8xf32>
        %186 = vector.insert %182, %arg13 [%62, %61] : f32 into vector<2x8xf32>
        %187 = vector.insert %183, %186 [%62, %154] : f32 into vector<2x8xf32>
        %188 = vector.insert %184, %187 [%62, %157] : f32 into vector<2x8xf32>
        %189 = vector.insert %185, %188 [%62, %160] : f32 into vector<2x8xf32>
        affine.yield %162, %181, %189 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %59 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_5) -> (vector<2x1xf32>) {
        %61 = vector.extract %48#1[%arg10, 0] : f32 from vector<2x1xf32>
        %62 = vector.insert %61, %arg11 [%arg10, 0] : f32 into vector<2x1xf32>
        affine.yield %62 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %60:3 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_8, %arg12 = %cst_8, %arg13 = %cst_8) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %62 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg10)
        %63 = vector.extract %54#1[%62, %61] : f32 from vector<2x8xf32>
        %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %65 = vector.extract %54#1[%62, %64] : f32 from vector<2x8xf32>
        %66 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %67 = vector.extract %54#1[%62, %66] : f32 from vector<2x8xf32>
        %68 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %69 = vector.extract %54#1[%62, %68] : f32 from vector<2x8xf32>
        %70 = vector.insert %63, %arg11 [%62, %61] : f32 into vector<2x8xf32>
        %71 = vector.insert %65, %70 [%62, %64] : f32 into vector<2x8xf32>
        %72 = vector.insert %67, %71 [%62, %66] : f32 into vector<2x8xf32>
        %73 = vector.insert %69, %72 [%62, %68] : f32 into vector<2x8xf32>
        %74 = vector.extract %56#1[%62, %61] : f32 from vector<2x8xf32>
        %75 = vector.extract %56#1[%62, %64] : f32 from vector<2x8xf32>
        %76 = vector.extract %56#1[%62, %66] : f32 from vector<2x8xf32>
        %77 = vector.extract %56#1[%62, %68] : f32 from vector<2x8xf32>
        %78 = vector.insert %74, %arg12 [%62, %61] : f32 into vector<2x8xf32>
        %79 = vector.insert %75, %78 [%62, %64] : f32 into vector<2x8xf32>
        %80 = vector.insert %76, %79 [%62, %66] : f32 into vector<2x8xf32>
        %81 = vector.insert %77, %80 [%62, %68] : f32 into vector<2x8xf32>
        %82 = vector.extract %58#1[%62, %61] : f32 from vector<2x8xf32>
        %83 = vector.extract %58#1[%62, %64] : f32 from vector<2x8xf32>
        %84 = vector.extract %58#1[%62, %66] : f32 from vector<2x8xf32>
        %85 = vector.extract %58#1[%62, %68] : f32 from vector<2x8xf32>
        %86 = vector.insert %82, %arg13 [%62, %61] : f32 into vector<2x8xf32>
        %87 = vector.insert %83, %86 [%62, %64] : f32 into vector<2x8xf32>
        %88 = vector.insert %84, %87 [%62, %66] : f32 into vector<2x8xf32>
        %89 = vector.insert %85, %88 [%62, %68] : f32 into vector<2x8xf32>
        affine.yield %73, %81, %89 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      gpu.barrier
      affine.yield %58#2, %59, %60#0, %60#1, %60#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {local_yield_transformed = true}
    %24 = arith.index_cast %thread_id_x : index to i32
    %25 = arith.andi %24, %c16_i32 : i32
    %26 = arith.cmpi ne, %25, %c0_i32 : i32
    %27 = arith.andi %24, %c32_i32 : i32
    %28 = arith.cmpi ne, %27, %c0_i32 : i32
    %29:8 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_8, %arg8 = %cst_8, %arg9 = %cst_8, %arg10 = %cst_8, %arg11 = %cst_2, %arg12 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %37 = vector.extract %23#4[%36, %35] : f32 from vector<2x8xf32>
      %38 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %39 = vector.extract %23#4[%36, %38] : f32 from vector<2x8xf32>
      %40 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %41 = vector.extract %23#4[%36, %40] : f32 from vector<2x8xf32>
      %42 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %43 = vector.extract %23#4[%36, %42] : f32 from vector<2x8xf32>
      %44 = arith.select %26, %37, %39 : f32
      %shuffleResult, %valid = gpu.shuffle xor %44, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %45 = arith.select %26, %shuffleResult, %37 : f32
      %46 = arith.select %26, %39, %shuffleResult : f32
      %47 = arith.select %26, %41, %43 : f32
      %shuffleResult_13, %valid_14 = gpu.shuffle xor %47, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %48 = arith.select %26, %shuffleResult_13, %41 : f32
      %49 = arith.select %26, %43, %shuffleResult_13 : f32
      %50 = arith.select %28, %45, %48 : f32
      %shuffleResult_15, %valid_16 = gpu.shuffle xor %50, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %51 = arith.select %28, %shuffleResult_15, %45 : f32
      %52 = arith.select %28, %48, %shuffleResult_15 : f32
      %53 = arith.select %28, %46, %49 : f32
      %shuffleResult_17, %valid_18 = gpu.shuffle xor %53, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %54 = arith.select %28, %shuffleResult_17, %46 : f32
      %55 = arith.select %28, %49, %shuffleResult_17 : f32
      %56 = vector.insert %51, %arg5 [%36, %35] : f32 into vector<2x8xf32>
      %57 = vector.insert %54, %56 [%36, %38] : f32 into vector<2x8xf32>
      %58 = vector.insert %52, %57 [%36, %40] : f32 into vector<2x8xf32>
      %59 = vector.insert %55, %58 [%36, %42] : f32 into vector<2x8xf32>
      %60 = vector.extract %23#3[%36, %35] : f32 from vector<2x8xf32>
      %61 = vector.extract %23#3[%36, %38] : f32 from vector<2x8xf32>
      %62 = vector.extract %23#3[%36, %40] : f32 from vector<2x8xf32>
      %63 = vector.extract %23#3[%36, %42] : f32 from vector<2x8xf32>
      %64 = arith.select %26, %60, %61 : f32
      %shuffleResult_19, %valid_20 = gpu.shuffle xor %64, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %65 = arith.select %26, %shuffleResult_19, %60 : f32
      %66 = arith.select %26, %61, %shuffleResult_19 : f32
      %67 = arith.select %26, %62, %63 : f32
      %shuffleResult_21, %valid_22 = gpu.shuffle xor %67, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %68 = arith.select %26, %shuffleResult_21, %62 : f32
      %69 = arith.select %26, %63, %shuffleResult_21 : f32
      %70 = arith.select %28, %65, %68 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %70, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %71 = arith.select %28, %shuffleResult_23, %65 : f32
      %72 = arith.select %28, %68, %shuffleResult_23 : f32
      %73 = arith.select %28, %66, %69 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %73, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %74 = arith.select %28, %shuffleResult_25, %66 : f32
      %75 = arith.select %28, %69, %shuffleResult_25 : f32
      %76 = vector.insert %71, %arg6 [%36, %35] : f32 into vector<2x8xf32>
      %77 = vector.insert %74, %76 [%36, %38] : f32 into vector<2x8xf32>
      %78 = vector.insert %72, %77 [%36, %40] : f32 into vector<2x8xf32>
      %79 = vector.insert %75, %78 [%36, %42] : f32 into vector<2x8xf32>
      %80 = vector.extract %23#2[%36, %35] : f32 from vector<2x8xf32>
      %81 = vector.extract %23#2[%36, %38] : f32 from vector<2x8xf32>
      %82 = vector.extract %23#2[%36, %40] : f32 from vector<2x8xf32>
      %83 = vector.extract %23#2[%36, %42] : f32 from vector<2x8xf32>
      %84 = arith.select %26, %80, %81 : f32
      %shuffleResult_27, %valid_28 = gpu.shuffle xor %84, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %85 = arith.select %26, %shuffleResult_27, %80 : f32
      %86 = arith.select %26, %81, %shuffleResult_27 : f32
      %87 = arith.select %26, %82, %83 : f32
      %shuffleResult_29, %valid_30 = gpu.shuffle xor %87, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %88 = arith.select %26, %shuffleResult_29, %82 : f32
      %89 = arith.select %26, %83, %shuffleResult_29 : f32
      %90 = arith.select %28, %85, %88 : f32
      %shuffleResult_31, %valid_32 = gpu.shuffle xor %90, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %91 = arith.select %28, %shuffleResult_31, %85 : f32
      %92 = arith.select %28, %88, %shuffleResult_31 : f32
      %93 = arith.select %28, %86, %89 : f32
      %shuffleResult_33, %valid_34 = gpu.shuffle xor %93, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %94 = arith.select %28, %shuffleResult_33, %86 : f32
      %95 = arith.select %28, %89, %shuffleResult_33 : f32
      %96 = vector.insert %91, %arg7 [%36, %35] : f32 into vector<2x8xf32>
      %97 = vector.insert %94, %96 [%36, %38] : f32 into vector<2x8xf32>
      %98 = vector.insert %92, %97 [%36, %40] : f32 into vector<2x8xf32>
      %99 = vector.insert %95, %98 [%36, %42] : f32 into vector<2x8xf32>
      %100 = vector.extract %23#0[%36, %35] : f32 from vector<2x8xf32>
      %101 = vector.extract %23#0[%36, %38] : f32 from vector<2x8xf32>
      %102 = vector.extract %23#0[%36, %40] : f32 from vector<2x8xf32>
      %103 = vector.extract %23#0[%36, %42] : f32 from vector<2x8xf32>
      %104 = arith.select %26, %100, %101 : f32
      %shuffleResult_35, %valid_36 = gpu.shuffle xor %104, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %105 = arith.select %26, %shuffleResult_35, %100 : f32
      %106 = arith.select %26, %101, %shuffleResult_35 : f32
      %107 = arith.select %26, %102, %103 : f32
      %shuffleResult_37, %valid_38 = gpu.shuffle xor %107, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %108 = arith.select %26, %shuffleResult_37, %102 : f32
      %109 = arith.select %26, %103, %shuffleResult_37 : f32
      %110 = arith.select %28, %105, %108 : f32
      %shuffleResult_39, %valid_40 = gpu.shuffle xor %110, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %111 = arith.select %28, %shuffleResult_39, %105 : f32
      %112 = arith.select %28, %108, %shuffleResult_39 : f32
      %113 = arith.select %28, %106, %109 : f32
      %shuffleResult_41, %valid_42 = gpu.shuffle xor %113, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %114 = arith.select %28, %shuffleResult_41, %106 : f32
      %115 = arith.select %28, %109, %shuffleResult_41 : f32
      %116 = vector.insert %111, %arg8 [%36, %35] : f32 into vector<2x8xf32>
      %117 = vector.insert %114, %116 [%36, %38] : f32 into vector<2x8xf32>
      %118 = vector.insert %112, %117 [%36, %40] : f32 into vector<2x8xf32>
      %119 = vector.insert %115, %118 [%36, %42] : f32 into vector<2x8xf32>
      %120 = vector.extract %23#1[%36, 0] : f32 from vector<2x1xf32>
      %121 = vector.insert %120, %arg9 [%36, %35] : f32 into vector<2x8xf32>
      %122 = vector.insert %120, %121 [%36, %38] : f32 into vector<2x8xf32>
      %123 = vector.insert %120, %122 [%36, %40] : f32 into vector<2x8xf32>
      %124 = vector.insert %120, %123 [%36, %42] : f32 into vector<2x8xf32>
      %125 = vector.extract %119[%36, %35] : f32 from vector<2x8xf32>
      %126 = vector.extract %119[%36, %38] : f32 from vector<2x8xf32>
      %127 = vector.extract %119[%36, %40] : f32 from vector<2x8xf32>
      %128 = vector.extract %119[%36, %42] : f32 from vector<2x8xf32>
      %129 = vector.from_elements %125, %126, %127, %128 : vector<1x4xf32>
      %130 = vector.extract %124[%36, %35] : f32 from vector<2x8xf32>
      %131 = vector.extract %124[%36, %38] : f32 from vector<2x8xf32>
      %132 = vector.extract %124[%36, %40] : f32 from vector<2x8xf32>
      %133 = vector.extract %124[%36, %42] : f32 from vector<2x8xf32>
      %134 = vector.from_elements %130, %131, %132, %133 : vector<1x4xf32>
      %135 = arith.divf %129, %134 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %136 = vector.extract %135[0, 0] : f32 from vector<1x4xf32>
      %137 = vector.insert %136, %arg10 [%36, %35] : f32 into vector<2x8xf32>
      %138 = vector.extract %135[0, 1] : f32 from vector<1x4xf32>
      %139 = vector.insert %138, %137 [%36, %38] : f32 into vector<2x8xf32>
      %140 = vector.extract %135[0, 2] : f32 from vector<1x4xf32>
      %141 = vector.insert %140, %139 [%36, %40] : f32 into vector<2x8xf32>
      %142 = vector.extract %135[0, 3] : f32 from vector<1x4xf32>
      %143 = vector.insert %142, %141 [%36, %42] : f32 into vector<2x8xf32>
      %144 = vector.extract %143[%36, %35] : f32 from vector<2x8xf32>
      %145 = vector.extract %143[%36, %38] : f32 from vector<2x8xf32>
      %146 = vector.extract %143[%36, %40] : f32 from vector<2x8xf32>
      %147 = vector.extract %143[%36, %42] : f32 from vector<2x8xf32>
      %148 = vector.from_elements %144, %145, %146, %147 : vector<1x4xf32>
      %149 = arith.truncf %148 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %150 = vector.extract %149[0, 0] : f16 from vector<1x4xf16>
      %151 = vector.insert %150, %arg11 [%36, %35] : f16 into vector<2x8xf16>
      %152 = vector.extract %149[0, 1] : f16 from vector<1x4xf16>
      %153 = vector.insert %152, %151 [%36, %38] : f16 into vector<2x8xf16>
      %154 = vector.extract %149[0, 2] : f16 from vector<1x4xf16>
      %155 = vector.insert %154, %153 [%36, %40] : f16 into vector<2x8xf16>
      %156 = vector.extract %149[0, 3] : f16 from vector<1x4xf16>
      %157 = vector.insert %156, %155 [%36, %42] : f16 into vector<2x8xf16>
      %158 = vector.extract %157[%36, %35] : f16 from vector<2x8xf16>
      %159 = vector.extract %157[%36, %38] : f16 from vector<2x8xf16>
      %160 = vector.extract %157[%36, %40] : f16 from vector<2x8xf16>
      %161 = vector.extract %157[%36, %42] : f16 from vector<2x8xf16>
      %162 = vector.insert %158, %arg12 [%36, %35] : f16 into vector<2x8xf16>
      %163 = vector.insert %159, %162 [%36, %38] : f16 into vector<2x8xf16>
      %164 = vector.insert %160, %163 [%36, %40] : f16 into vector<2x8xf16>
      %165 = vector.insert %161, %164 [%36, %42] : f16 into vector<2x8xf16>
      affine.yield %59, %79, %99, %119, %124, %143, %157, %165 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    %30 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %31 = arith.cmpi ult, %30, %c4 : index
    scf.if %31 {
      affine.for %arg4 = 0 to 4 {
        %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %37 = vector.extract %29#7[%36, %35] : f16 from vector<2x8xf16>
        affine.store %37, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_z) * 32 + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
        %38 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %39 = vector.extract %29#7[%36, %38] : f16 from vector<2x8xf16>
        affine.store %39, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_z) * 32 + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
        %40 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %41 = vector.extract %29#7[%36, %40] : f16 from vector<2x8xf16>
        affine.store %41, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_z) * 32 + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
        %42 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %43 = vector.extract %29#7[%36, %42] : f16 from vector<2x8xf16>
        affine.store %43, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_z) * 32 + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %32:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %37 = vector.extract %23#1[%36, 0] : f32 from vector<2x1xf32>
      %38 = vector.insert %37, %arg5 [%36, %35] : f32 into vector<2x8xf32>
      %39 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %40 = vector.insert %37, %38 [%36, %39] : f32 into vector<2x8xf32>
      %41 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %42 = vector.insert %37, %40 [%36, %41] : f32 into vector<2x8xf32>
      %43 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %44 = vector.insert %37, %42 [%36, %43] : f32 into vector<2x8xf32>
      %45 = vector.extract %29#2[%36, %35] : f32 from vector<2x8xf32>
      %46 = vector.extract %29#2[%36, %39] : f32 from vector<2x8xf32>
      %47 = vector.extract %29#2[%36, %41] : f32 from vector<2x8xf32>
      %48 = vector.extract %29#2[%36, %43] : f32 from vector<2x8xf32>
      %49 = vector.from_elements %45, %46, %47, %48 : vector<1x4xf32>
      %50 = vector.extract %44[%36, %35] : f32 from vector<2x8xf32>
      %51 = vector.extract %44[%36, %39] : f32 from vector<2x8xf32>
      %52 = vector.extract %44[%36, %41] : f32 from vector<2x8xf32>
      %53 = vector.extract %44[%36, %43] : f32 from vector<2x8xf32>
      %54 = vector.from_elements %50, %51, %52, %53 : vector<1x4xf32>
      %55 = arith.divf %49, %54 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg6 [%36, %35] : f32 into vector<2x8xf32>
      %58 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %59 = vector.insert %58, %57 [%36, %39] : f32 into vector<2x8xf32>
      %60 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %61 = vector.insert %60, %59 [%36, %41] : f32 into vector<2x8xf32>
      %62 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %61 [%36, %43] : f32 into vector<2x8xf32>
      %64 = vector.extract %63[%36, %35] : f32 from vector<2x8xf32>
      %65 = vector.extract %63[%36, %39] : f32 from vector<2x8xf32>
      %66 = vector.extract %63[%36, %41] : f32 from vector<2x8xf32>
      %67 = vector.extract %63[%36, %43] : f32 from vector<2x8xf32>
      %68 = vector.from_elements %64, %65, %66, %67 : vector<1x4xf32>
      %69 = arith.truncf %68 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %70 = vector.extract %69[0, 0] : f16 from vector<1x4xf16>
      %71 = vector.insert %70, %arg7 [%36, %35] : f16 into vector<2x8xf16>
      %72 = vector.extract %69[0, 1] : f16 from vector<1x4xf16>
      %73 = vector.insert %72, %71 [%36, %39] : f16 into vector<2x8xf16>
      %74 = vector.extract %69[0, 2] : f16 from vector<1x4xf16>
      %75 = vector.insert %74, %73 [%36, %41] : f16 into vector<2x8xf16>
      %76 = vector.extract %69[0, 3] : f16 from vector<1x4xf16>
      %77 = vector.insert %76, %75 [%36, %43] : f16 into vector<2x8xf16>
      %78 = vector.extract %77[%36, %35] : f16 from vector<2x8xf16>
      %79 = vector.extract %77[%36, %39] : f16 from vector<2x8xf16>
      %80 = vector.extract %77[%36, %41] : f16 from vector<2x8xf16>
      %81 = vector.extract %77[%36, %43] : f16 from vector<2x8xf16>
      %82 = vector.insert %78, %arg8 [%36, %35] : f16 into vector<2x8xf16>
      %83 = vector.insert %79, %82 [%36, %39] : f16 into vector<2x8xf16>
      %84 = vector.insert %80, %83 [%36, %41] : f16 into vector<2x8xf16>
      %85 = vector.insert %81, %84 [%36, %43] : f16 into vector<2x8xf16>
      affine.yield %44, %63, %77, %85 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %31 {
      affine.for %arg4 = 0 to 4 {
        %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %37 = vector.extract %32#3[%36, %35] : f16 from vector<2x8xf16>
        affine.store %37, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
        %38 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %39 = vector.extract %32#3[%36, %38] : f16 from vector<2x8xf16>
        affine.store %39, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
        %40 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %41 = vector.extract %32#3[%36, %40] : f16 from vector<2x8xf16>
        affine.store %41, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
        %42 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %43 = vector.extract %32#3[%36, %42] : f16 from vector<2x8xf16>
        affine.store %43, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %33:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %37 = vector.extract %23#1[%36, 0] : f32 from vector<2x1xf32>
      %38 = vector.insert %37, %arg5 [%36, %35] : f32 into vector<2x8xf32>
      %39 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %40 = vector.insert %37, %38 [%36, %39] : f32 into vector<2x8xf32>
      %41 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %42 = vector.insert %37, %40 [%36, %41] : f32 into vector<2x8xf32>
      %43 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %44 = vector.insert %37, %42 [%36, %43] : f32 into vector<2x8xf32>
      %45 = vector.extract %29#1[%36, %35] : f32 from vector<2x8xf32>
      %46 = vector.extract %29#1[%36, %39] : f32 from vector<2x8xf32>
      %47 = vector.extract %29#1[%36, %41] : f32 from vector<2x8xf32>
      %48 = vector.extract %29#1[%36, %43] : f32 from vector<2x8xf32>
      %49 = vector.from_elements %45, %46, %47, %48 : vector<1x4xf32>
      %50 = vector.extract %44[%36, %35] : f32 from vector<2x8xf32>
      %51 = vector.extract %44[%36, %39] : f32 from vector<2x8xf32>
      %52 = vector.extract %44[%36, %41] : f32 from vector<2x8xf32>
      %53 = vector.extract %44[%36, %43] : f32 from vector<2x8xf32>
      %54 = vector.from_elements %50, %51, %52, %53 : vector<1x4xf32>
      %55 = arith.divf %49, %54 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg6 [%36, %35] : f32 into vector<2x8xf32>
      %58 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %59 = vector.insert %58, %57 [%36, %39] : f32 into vector<2x8xf32>
      %60 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %61 = vector.insert %60, %59 [%36, %41] : f32 into vector<2x8xf32>
      %62 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %61 [%36, %43] : f32 into vector<2x8xf32>
      %64 = vector.extract %63[%36, %35] : f32 from vector<2x8xf32>
      %65 = vector.extract %63[%36, %39] : f32 from vector<2x8xf32>
      %66 = vector.extract %63[%36, %41] : f32 from vector<2x8xf32>
      %67 = vector.extract %63[%36, %43] : f32 from vector<2x8xf32>
      %68 = vector.from_elements %64, %65, %66, %67 : vector<1x4xf32>
      %69 = arith.truncf %68 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %70 = vector.extract %69[0, 0] : f16 from vector<1x4xf16>
      %71 = vector.insert %70, %arg7 [%36, %35] : f16 into vector<2x8xf16>
      %72 = vector.extract %69[0, 1] : f16 from vector<1x4xf16>
      %73 = vector.insert %72, %71 [%36, %39] : f16 into vector<2x8xf16>
      %74 = vector.extract %69[0, 2] : f16 from vector<1x4xf16>
      %75 = vector.insert %74, %73 [%36, %41] : f16 into vector<2x8xf16>
      %76 = vector.extract %69[0, 3] : f16 from vector<1x4xf16>
      %77 = vector.insert %76, %75 [%36, %43] : f16 into vector<2x8xf16>
      %78 = vector.extract %77[%36, %35] : f16 from vector<2x8xf16>
      %79 = vector.extract %77[%36, %39] : f16 from vector<2x8xf16>
      %80 = vector.extract %77[%36, %41] : f16 from vector<2x8xf16>
      %81 = vector.extract %77[%36, %43] : f16 from vector<2x8xf16>
      %82 = vector.insert %78, %arg8 [%36, %35] : f16 into vector<2x8xf16>
      %83 = vector.insert %79, %82 [%36, %39] : f16 into vector<2x8xf16>
      %84 = vector.insert %80, %83 [%36, %41] : f16 into vector<2x8xf16>
      %85 = vector.insert %81, %84 [%36, %43] : f16 into vector<2x8xf16>
      affine.yield %44, %63, %77, %85 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %31 {
      affine.for %arg4 = 0 to 4 {
        %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %37 = vector.extract %33#3[%36, %35] : f16 from vector<2x8xf16>
        affine.store %37, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
        %38 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %39 = vector.extract %33#3[%36, %38] : f16 from vector<2x8xf16>
        affine.store %39, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
        %40 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %41 = vector.extract %33#3[%36, %40] : f16 from vector<2x8xf16>
        affine.store %41, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
        %42 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %43 = vector.extract %33#3[%36, %42] : f16 from vector<2x8xf16>
        affine.store %43, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %34:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_2, %arg8 = %cst_2) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %37 = vector.extract %23#1[%36, 0] : f32 from vector<2x1xf32>
      %38 = vector.insert %37, %arg5 [%36, %35] : f32 into vector<2x8xf32>
      %39 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %40 = vector.insert %37, %38 [%36, %39] : f32 into vector<2x8xf32>
      %41 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %42 = vector.insert %37, %40 [%36, %41] : f32 into vector<2x8xf32>
      %43 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %44 = vector.insert %37, %42 [%36, %43] : f32 into vector<2x8xf32>
      %45 = vector.extract %29#0[%36, %35] : f32 from vector<2x8xf32>
      %46 = vector.extract %29#0[%36, %39] : f32 from vector<2x8xf32>
      %47 = vector.extract %29#0[%36, %41] : f32 from vector<2x8xf32>
      %48 = vector.extract %29#0[%36, %43] : f32 from vector<2x8xf32>
      %49 = vector.from_elements %45, %46, %47, %48 : vector<1x4xf32>
      %50 = vector.extract %44[%36, %35] : f32 from vector<2x8xf32>
      %51 = vector.extract %44[%36, %39] : f32 from vector<2x8xf32>
      %52 = vector.extract %44[%36, %41] : f32 from vector<2x8xf32>
      %53 = vector.extract %44[%36, %43] : f32 from vector<2x8xf32>
      %54 = vector.from_elements %50, %51, %52, %53 : vector<1x4xf32>
      %55 = arith.divf %49, %54 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg6 [%36, %35] : f32 into vector<2x8xf32>
      %58 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %59 = vector.insert %58, %57 [%36, %39] : f32 into vector<2x8xf32>
      %60 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %61 = vector.insert %60, %59 [%36, %41] : f32 into vector<2x8xf32>
      %62 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %61 [%36, %43] : f32 into vector<2x8xf32>
      %64 = vector.extract %63[%36, %35] : f32 from vector<2x8xf32>
      %65 = vector.extract %63[%36, %39] : f32 from vector<2x8xf32>
      %66 = vector.extract %63[%36, %41] : f32 from vector<2x8xf32>
      %67 = vector.extract %63[%36, %43] : f32 from vector<2x8xf32>
      %68 = vector.from_elements %64, %65, %66, %67 : vector<1x4xf32>
      %69 = arith.truncf %68 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %70 = vector.extract %69[0, 0] : f16 from vector<1x4xf16>
      %71 = vector.insert %70, %arg7 [%36, %35] : f16 into vector<2x8xf16>
      %72 = vector.extract %69[0, 1] : f16 from vector<1x4xf16>
      %73 = vector.insert %72, %71 [%36, %39] : f16 into vector<2x8xf16>
      %74 = vector.extract %69[0, 2] : f16 from vector<1x4xf16>
      %75 = vector.insert %74, %73 [%36, %41] : f16 into vector<2x8xf16>
      %76 = vector.extract %69[0, 3] : f16 from vector<1x4xf16>
      %77 = vector.insert %76, %75 [%36, %43] : f16 into vector<2x8xf16>
      %78 = vector.extract %77[%36, %35] : f16 from vector<2x8xf16>
      %79 = vector.extract %77[%36, %39] : f16 from vector<2x8xf16>
      %80 = vector.extract %77[%36, %41] : f16 from vector<2x8xf16>
      %81 = vector.extract %77[%36, %43] : f16 from vector<2x8xf16>
      %82 = vector.insert %78, %arg8 [%36, %35] : f16 into vector<2x8xf16>
      %83 = vector.insert %79, %82 [%36, %39] : f16 into vector<2x8xf16>
      %84 = vector.insert %80, %83 [%36, %41] : f16 into vector<2x8xf16>
      %85 = vector.insert %81, %84 [%36, %43] : f16 into vector<2x8xf16>
      affine.yield %44, %63, %77, %85 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %31 {
      affine.for %arg4 = 0 to 4 {
        %35 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %36 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %37 = vector.extract %34#3[%36, %35] : f16 from vector<2x8xf16>
        affine.store %37, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
        %38 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %39 = vector.extract %34#3[%36, %38] : f16 from vector<2x8xf16>
        affine.store %39, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
        %40 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %41 = vector.extract %34#3[%36, %40] : f16 from vector<2x8xf16>
        affine.store %41, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
        %42 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %43 = vector.extract %34#3[%36, %42] : f16 from vector<2x8xf16>
        affine.store %43, %arg3[0, (symbol(%block_id_x) + symbol(%block_id_y) floordiv 32) mod 32, (%arg4 floordiv 2) * 64 + symbol(%block_id_y) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%block_id_y) floordiv 32) * 4096, %arg4 * 16 - (%arg4 floordiv 2) * 32 + ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
