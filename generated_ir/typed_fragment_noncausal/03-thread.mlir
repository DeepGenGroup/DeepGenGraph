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
    %cst_7 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_9 = arith.constant 0xFF800000 : f32
    %cst_10 = arith.constant 0.127531052 : f32
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
    %8 = arith.truncf %cst_10 : f32 to f16
    %9:3 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_8, %arg6 = %cst_8, %arg7 = %cst_8) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %25 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        frisk.fragment_yield %cst_7 : vector<1x1xf32>
      }
      %26 = vector.extract %25[0, 0] : f32 from vector<1x1xf32>
      %27 = vector.insert %26, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      %28 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %34 = vector.extract %27[%arg4, 0] : f32 from vector<2x1xf32>
        %35 = vector.broadcast %34 : f32 to vector<1x1xf32>
        %36 = arith.addf %35, %cst {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        frisk.fragment_yield %36 : vector<1x1xf32>
      }
      %29 = vector.extract %28[0, 0] : f32 from vector<1x1xf32>
      %30 = vector.insert %29, %arg6 [%arg4, 0] : f32 into vector<2x1xf32>
      %31 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %34 = vector.extract %30[%arg4, 0] : f32 from vector<2x1xf32>
        %35 = vector.broadcast %34 : f32 to vector<1x1xf32>
        frisk.fragment_yield %35 : vector<1x1xf32>
      }
      %32 = vector.extract %31[0, 0] : f32 from vector<1x1xf32>
      %33 = vector.insert %32, %arg7 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %27, %30, %33 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (vector<2x8xf32>) {
      %25 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %26 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %27 = frisk.fragment "load" -> (vector<1x4xf32>) {
        frisk.fragment_yield %cst_4 : vector<1x4xf32>
      }
      %28 = vector.extract %27[0, 0] : f32 from vector<1x4xf32>
      %29 = vector.insert %28, %arg5 [%26, %25] : f32 into vector<2x8xf32>
      %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %31 = vector.extract %27[0, 1] : f32 from vector<1x4xf32>
      %32 = vector.insert %31, %29 [%26, %30] : f32 into vector<2x8xf32>
      %33 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %34 = vector.extract %27[0, 2] : f32 from vector<1x4xf32>
      %35 = vector.insert %34, %32 [%26, %33] : f32 into vector<2x8xf32>
      %36 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %37 = vector.extract %27[0, 3] : f32 from vector<1x4xf32>
      %38 = vector.insert %37, %35 [%26, %36] : f32 into vector<2x8xf32>
      affine.yield %38 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_8) -> (vector<2x1xf32>) {
      %25 = frisk.fragment "load" -> (vector<1x1xf32>) {
        frisk.fragment_yield %cst_7 : vector<1x1xf32>
      }
      %26 = vector.extract %25[0, 0] : f32 from vector<1x1xf32>
      %27 = vector.insert %26, %arg5 [%arg4, 0] : f32 into vector<2x1xf32>
      affine.yield %27 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
    %12:3 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %25 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %26 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %27 = frisk.fragment "load" -> (vector<1x4xf32>) {
        frisk.fragment_yield %cst_4 : vector<1x4xf32>
      }
      %28 = vector.extract %27[0, 0] : f32 from vector<1x4xf32>
      %29 = vector.insert %28, %arg5 [%26, %25] : f32 into vector<2x8xf32>
      %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %31 = vector.extract %27[0, 1] : f32 from vector<1x4xf32>
      %32 = vector.insert %31, %29 [%26, %30] : f32 into vector<2x8xf32>
      %33 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %34 = vector.extract %27[0, 2] : f32 from vector<1x4xf32>
      %35 = vector.insert %34, %32 [%26, %33] : f32 into vector<2x8xf32>
      %36 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %37 = vector.extract %27[0, 3] : f32 from vector<1x4xf32>
      %38 = vector.insert %37, %35 [%26, %36] : f32 into vector<2x8xf32>
      %39 = vector.insert %28, %arg6 [%26, %25] : f32 into vector<2x8xf32>
      %40 = vector.insert %31, %39 [%26, %30] : f32 into vector<2x8xf32>
      %41 = vector.insert %34, %40 [%26, %33] : f32 into vector<2x8xf32>
      %42 = vector.insert %37, %41 [%26, %36] : f32 into vector<2x8xf32>
      %43 = vector.insert %28, %arg7 [%26, %25] : f32 into vector<2x8xf32>
      %44 = vector.insert %31, %43 [%26, %30] : f32 into vector<2x8xf32>
      %45 = vector.insert %34, %44 [%26, %33] : f32 into vector<2x8xf32>
      %46 = vector.insert %37, %45 [%26, %36] : f32 into vector<2x8xf32>
      affine.yield %38, %42, %46 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "load_fragment", tiled = true}
    %13:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#2, %arg8 = %12#0, %arg9 = %12#1, %arg10 = %12#2) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %25 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %45 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %46 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %47 = frisk.fragment "load" -> (vector<1x4xf32>) {
          frisk.fragment_yield %cst_4 : vector<1x4xf32>
        }
        %48 = vector.extract %47[0, 0] : f32 from vector<1x4xf32>
        %49 = vector.insert %48, %arg12 [%46, %45] : f32 into vector<2x32xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %51 = vector.extract %47[0, 1] : f32 from vector<1x4xf32>
        %52 = vector.insert %51, %49 [%46, %50] : f32 into vector<2x32xf32>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %54 = vector.extract %47[0, 2] : f32 from vector<1x4xf32>
        %55 = vector.insert %54, %52 [%46, %53] : f32 into vector<2x32xf32>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %57 = vector.extract %47[0, 3] : f32 from vector<1x4xf32>
        %58 = vector.insert %57, %55 [%46, %56] : f32 into vector<2x32xf32>
        affine.yield %58 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
      %26 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %25) -> (vector<2x32xf32>) {
        %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_19 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %45 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%6, %7]
        %46 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%6]
        %subview_20 = memref.subview %arg0[0, %45, %46, %arg11] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
        frisk.async_copy_gm_to_shm %subview_20 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
        frisk.wait_buffer_ready %alloc_18 : memref<128x32xf16, 3>
        %47 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
        %subview_21 = memref.subview %transpose[0, %47, %arg11, %arg4] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
        frisk.async_copy_gm_to_shm %subview_21 to %alloc_19 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<32x128xf16, 3>
        %thread_id_x_22 = gpu.thread_id x
        %c0 = arith.constant 0 : index
        %48 = frisk.fragment "load" attributes {frisk.pipeline_stage = "prologue"} -> (vector<1x4xf16>) {
          %73 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%c0)[%thread_id_x_22]
          %74 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4)>(%c0)[%thread_id_x_22]
          %75 = memref.load %alloc_18[%73, %74] : memref<128x32xf16, 3>
          %76 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 1)>(%c0)[%thread_id_x_22]
          %77 = memref.load %alloc_18[%73, %76] : memref<128x32xf16, 3>
          %78 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 2)>(%c0)[%thread_id_x_22]
          %79 = memref.load %alloc_18[%73, %78] : memref<128x32xf16, 3>
          %80 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 3)>(%c0)[%thread_id_x_22]
          %81 = memref.load %alloc_18[%73, %80] : memref<128x32xf16, 3>
          %82 = vector.from_elements %75, %77, %79, %81 : vector<1x4xf16>
          frisk.fragment_yield %82 : vector<1x4xf16>
        }
        %49:3 = affine.for %arg13 = 0 to 3 iter_args(%arg14 = %cst_6, %arg15 = %cst_6, %arg16 = %48) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
          %73 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg13)
          %74 = frisk.fragment "load" attributes {frisk.pipeline_stage = "prefetch"} -> (vector<1x4xf16>) {
            %97 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 2) * 64 + s0 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>(%73)[%thread_id_x_22]
            %98 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4)>(%73)[%thread_id_x_22]
            %99 = memref.load %alloc_18[%97, %98] : memref<128x32xf16, 3>
            %100 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 1)>(%73)[%thread_id_x_22]
            %101 = memref.load %alloc_18[%97, %100] : memref<128x32xf16, 3>
            %102 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 2)>(%73)[%thread_id_x_22]
            %103 = memref.load %alloc_18[%97, %102] : memref<128x32xf16, 3>
            %104 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 2) * 32 + ((s0 mod 64) floordiv 16) * 4 + 3)>(%73)[%thread_id_x_22]
            %105 = memref.load %alloc_18[%97, %104] : memref<128x32xf16, 3>
            %106 = vector.from_elements %99, %101, %103, %105 : vector<1x4xf16>
            frisk.fragment_yield %106 : vector<1x4xf16>
          }
          %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %76 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg13)
          %77 = vector.extract %arg16[0, 0] : f16 from vector<1x4xf16>
          %78 = vector.insert %77, %arg14 [%76, %75] : f16 into vector<2x8xf16>
          %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg13)
          %80 = vector.extract %arg16[0, 1] : f16 from vector<1x4xf16>
          %81 = vector.insert %80, %78 [%76, %79] : f16 into vector<2x8xf16>
          %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg13)
          %83 = vector.extract %arg16[0, 2] : f16 from vector<1x4xf16>
          %84 = vector.insert %83, %81 [%76, %82] : f16 into vector<2x8xf16>
          %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg13)
          %86 = vector.extract %arg16[0, 3] : f16 from vector<1x4xf16>
          %87 = vector.insert %86, %84 [%76, %85] : f16 into vector<2x8xf16>
          %88 = frisk.fragment "compute" -> (vector<1x4xf16>) {
            %97 = vector.extract %87[%76, %75] : f16 from vector<2x8xf16>
            %98 = vector.extract %87[%76, %79] : f16 from vector<2x8xf16>
            %99 = vector.extract %87[%76, %82] : f16 from vector<2x8xf16>
            %100 = vector.extract %87[%76, %85] : f16 from vector<2x8xf16>
            %101 = vector.from_elements %97, %98, %99, %100 : vector<1x4xf16>
            %102 = vector.broadcast %8 : f16 to vector<1x4xf16>
            %103 = arith.mulf %101, %102 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
            frisk.fragment_yield %103 : vector<1x4xf16>
          }
          %89 = vector.extract %88[0, 0] : f16 from vector<1x4xf16>
          %90 = vector.insert %89, %arg15 [%76, %75] : f16 into vector<2x8xf16>
          %91 = vector.extract %88[0, 1] : f16 from vector<1x4xf16>
          %92 = vector.insert %91, %90 [%76, %79] : f16 into vector<2x8xf16>
          %93 = vector.extract %88[0, 2] : f16 from vector<1x4xf16>
          %94 = vector.insert %93, %92 [%76, %82] : f16 into vector<2x8xf16>
          %95 = vector.extract %88[0, 3] : f16 from vector<1x4xf16>
          %96 = vector.insert %95, %94 [%76, %85] : f16 into vector<2x8xf16>
          affine.yield %87, %96, %74 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
        %c3 = arith.constant 3 : index
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%c3)
        %51 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%c3)
        %52 = vector.extract %49#2[0, 0] : f16 from vector<1x4xf16>
        %53 = vector.insert %52, %49#0 [%51, %50] : f16 into vector<2x8xf16>
        %54 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%c3)
        %55 = vector.extract %49#2[0, 1] : f16 from vector<1x4xf16>
        %56 = vector.insert %55, %53 [%51, %54] : f16 into vector<2x8xf16>
        %57 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%c3)
        %58 = vector.extract %49#2[0, 2] : f16 from vector<1x4xf16>
        %59 = vector.insert %58, %56 [%51, %57] : f16 into vector<2x8xf16>
        %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%c3)
        %61 = vector.extract %49#2[0, 3] : f16 from vector<1x4xf16>
        %62 = vector.insert %61, %59 [%51, %60] : f16 into vector<2x8xf16>
        %63 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %73 = vector.extract %62[%51, %50] : f16 from vector<2x8xf16>
          %74 = vector.extract %62[%51, %54] : f16 from vector<2x8xf16>
          %75 = vector.extract %62[%51, %57] : f16 from vector<2x8xf16>
          %76 = vector.extract %62[%51, %60] : f16 from vector<2x8xf16>
          %77 = vector.from_elements %73, %74, %75, %76 : vector<1x4xf16>
          %78 = vector.broadcast %8 : f16 to vector<1x4xf16>
          %79 = arith.mulf %77, %78 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
          frisk.fragment_yield %79 : vector<1x4xf16>
        }
        %64 = vector.extract %63[0, 0] : f16 from vector<1x4xf16>
        %65 = vector.insert %64, %49#1 [%51, %50] : f16 into vector<2x8xf16>
        %66 = vector.extract %63[0, 1] : f16 from vector<1x4xf16>
        %67 = vector.insert %66, %65 [%51, %54] : f16 into vector<2x8xf16>
        %68 = vector.extract %63[0, 2] : f16 from vector<1x4xf16>
        %69 = vector.insert %68, %67 [%51, %57] : f16 into vector<2x8xf16>
        %70 = vector.extract %63[0, 3] : f16 from vector<1x4xf16>
        %71 = vector.insert %70, %69 [%51, %60] : f16 into vector<2x8xf16>
        frisk.wait_buffer_ready %alloc_19 : memref<32x128xf16, 3>
        %72:3 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_5, %arg15 = %cst_5, %arg16 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>, vector<2x32xf32>) {
          %73 = frisk.fragment "load" attributes {frisk.pipeline_stage = "prologue"} -> (vector<1x4xf16>) {
            %108 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
            %109 = affine.apply affine_map<(d0) -> (d0 * 4)>(%c0)
            %110 = vector.extract %71[%108, %109] : f16 from vector<2x8xf16>
            %111 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%c0)
            %112 = vector.extract %71[%108, %111] : f16 from vector<2x8xf16>
            %113 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%c0)
            %114 = vector.extract %71[%108, %113] : f16 from vector<2x8xf16>
            %115 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%c0)
            %116 = vector.extract %71[%108, %115] : f16 from vector<2x8xf16>
            %117 = vector.from_elements %110, %112, %114, %116 : vector<1x4xf16>
            frisk.fragment_yield %117 : vector<1x4xf16>
          }
          %74 = frisk.fragment "load" attributes {frisk.pipeline_stage = "prologue"} -> (vector<4x1xf16>) {
            %108 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4)>(%c0)[%thread_id_x_22]
            %109 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg13)[%thread_id_x_22]
            %110 = memref.load %alloc_19[%108, %109] : memref<32x128xf16, 3>
            %111 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>(%c0)[%thread_id_x_22]
            %112 = memref.load %alloc_19[%111, %109] : memref<32x128xf16, 3>
            %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>(%c0)[%thread_id_x_22]
            %114 = memref.load %alloc_19[%113, %109] : memref<32x128xf16, 3>
            %115 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>(%c0)[%thread_id_x_22]
            %116 = memref.load %alloc_19[%115, %109] : memref<32x128xf16, 3>
            %117 = vector.from_elements %110, %112, %114, %116 : vector<4x1xf16>
            frisk.fragment_yield %117 : vector<4x1xf16>
          }
          %75:3 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_4, %arg19 = %73, %arg20 = %74) -> (vector<1x4xf32>, vector<1x4xf16>, vector<4x1xf16>) {
            %108 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg17)
            %109 = frisk.fragment "load" attributes {frisk.pipeline_stage = "prefetch"} -> (vector<1x4xf16>) {
              %112 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
              %113 = affine.apply affine_map<(d0) -> (d0 * 4)>(%108)
              %114 = vector.extract %71[%112, %113] : f16 from vector<2x8xf16>
              %115 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%108)
              %116 = vector.extract %71[%112, %115] : f16 from vector<2x8xf16>
              %117 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%108)
              %118 = vector.extract %71[%112, %117] : f16 from vector<2x8xf16>
              %119 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%108)
              %120 = vector.extract %71[%112, %119] : f16 from vector<2x8xf16>
              %121 = vector.from_elements %114, %116, %118, %120 : vector<1x4xf16>
              frisk.fragment_yield %121 : vector<1x4xf16>
            }
            %110 = frisk.fragment "load" attributes {frisk.pipeline_stage = "prefetch"} -> (vector<4x1xf16>) {
              %112 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4)>(%108)[%thread_id_x_22]
              %113 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + (d0 floordiv 8) * -32 + ((d0 * 16 + (d0 floordiv 8) * -128 + s0 mod 16) floordiv 64) * 64 + ((d0 mod 8) floordiv 4) * -16 + s0 * 16 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4)>(%arg13)[%thread_id_x_22]
              %114 = memref.load %alloc_19[%112, %113] : memref<32x128xf16, 3>
              %115 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>(%108)[%thread_id_x_22]
              %116 = memref.load %alloc_19[%115, %113] : memref<32x128xf16, 3>
              %117 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>(%108)[%thread_id_x_22]
              %118 = memref.load %alloc_19[%117, %113] : memref<32x128xf16, 3>
              %119 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>(%108)[%thread_id_x_22]
              %120 = memref.load %alloc_19[%119, %113] : memref<32x128xf16, 3>
              %121 = vector.from_elements %114, %116, %118, %120 : vector<4x1xf16>
              frisk.fragment_yield %121 : vector<4x1xf16>
            }
            %111 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %112 = frisk.warp_mma_rr %arg19, %arg20, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %112 : vector<1x4xf32>
            }
            affine.yield %111, %109, %110 : vector<1x4xf32>, vector<1x4xf16>, vector<4x1xf16>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
          %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %108 = frisk.warp_mma_rr %75#1, %75#2, %75#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %108 : vector<1x4xf32>
          }
          %77 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg13)
          %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %79 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
          %80 = vector.insert %79, %arg14 [%77, %78] : f32 into vector<2x32xf32>
          %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %82 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
          %83 = vector.insert %82, %80 [%77, %81] : f32 into vector<2x32xf32>
          %84 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %85 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
          %86 = vector.insert %85, %83 [%77, %84] : f32 into vector<2x32xf32>
          %87 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %88 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
          %89 = vector.insert %88, %86 [%77, %87] : f32 into vector<2x32xf32>
          %90 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %108 = vector.extract %arg12[%77, %78] : f32 from vector<2x32xf32>
            %109 = vector.extract %arg12[%77, %81] : f32 from vector<2x32xf32>
            %110 = vector.extract %arg12[%77, %84] : f32 from vector<2x32xf32>
            %111 = vector.extract %arg12[%77, %87] : f32 from vector<2x32xf32>
            %112 = vector.from_elements %108, %109, %110, %111 : vector<1x4xf32>
            %113 = vector.extract %89[%77, %78] : f32 from vector<2x32xf32>
            %114 = vector.extract %89[%77, %81] : f32 from vector<2x32xf32>
            %115 = vector.extract %89[%77, %84] : f32 from vector<2x32xf32>
            %116 = vector.extract %89[%77, %87] : f32 from vector<2x32xf32>
            %117 = vector.from_elements %113, %114, %115, %116 : vector<1x4xf32>
            %118 = arith.addf %112, %117 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %118 : vector<1x4xf32>
          }
          %91 = vector.extract %90[0, 0] : f32 from vector<1x4xf32>
          %92 = vector.insert %91, %arg15 [%77, %78] : f32 into vector<2x32xf32>
          %93 = vector.extract %90[0, 1] : f32 from vector<1x4xf32>
          %94 = vector.insert %93, %92 [%77, %81] : f32 into vector<2x32xf32>
          %95 = vector.extract %90[0, 2] : f32 from vector<1x4xf32>
          %96 = vector.insert %95, %94 [%77, %84] : f32 into vector<2x32xf32>
          %97 = vector.extract %90[0, 3] : f32 from vector<1x4xf32>
          %98 = vector.insert %97, %96 [%77, %87] : f32 into vector<2x32xf32>
          %99 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %108 = vector.extract %98[%77, %78] : f32 from vector<2x32xf32>
            %109 = vector.extract %98[%77, %81] : f32 from vector<2x32xf32>
            %110 = vector.extract %98[%77, %84] : f32 from vector<2x32xf32>
            %111 = vector.extract %98[%77, %87] : f32 from vector<2x32xf32>
            %112 = vector.from_elements %108, %109, %110, %111 : vector<1x4xf32>
            frisk.fragment_yield %112 : vector<1x4xf32>
          }
          %100 = vector.extract %99[0, 0] : f32 from vector<1x4xf32>
          %101 = vector.insert %100, %arg16 [%77, %78] : f32 into vector<2x32xf32>
          %102 = vector.extract %99[0, 1] : f32 from vector<1x4xf32>
          %103 = vector.insert %102, %101 [%77, %81] : f32 into vector<2x32xf32>
          %104 = vector.extract %99[0, 2] : f32 from vector<1x4xf32>
          %105 = vector.insert %104, %103 [%77, %84] : f32 into vector<2x32xf32>
          %106 = vector.extract %99[0, 3] : f32 from vector<1x4xf32>
          %107 = vector.insert %106, %105 [%77, %87] : f32 into vector<2x32xf32>
          affine.yield %89, %98, %107 : vector<2x32xf32>, vector<2x32xf32>, vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
        affine.yield %72#2 : vector<2x32xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %27 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %45 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %59 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
          %61 = vector.extract %26[%59, %60] : f32 from vector<2x32xf32>
          %62 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %63 = vector.extract %26[%59, %62] : f32 from vector<2x32xf32>
          %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %65 = vector.extract %26[%59, %64] : f32 from vector<2x32xf32>
          %66 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %67 = vector.extract %26[%59, %66] : f32 from vector<2x32xf32>
          %68 = vector.from_elements %61, %63, %65, %67 : vector<1x4xf32>
          frisk.fragment_yield %68 : vector<1x4xf32>
        }
        %46 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %47 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %48 = vector.extract %45[0, 0] : f32 from vector<1x4xf32>
        %49 = vector.insert %48, %arg12 [%46, %47] : f32 into vector<2x32xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %51 = vector.extract %45[0, 1] : f32 from vector<1x4xf32>
        %52 = vector.insert %51, %49 [%46, %50] : f32 into vector<2x32xf32>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %54 = vector.extract %45[0, 2] : f32 from vector<1x4xf32>
        %55 = vector.insert %54, %52 [%46, %53] : f32 into vector<2x32xf32>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %57 = vector.extract %45[0, 3] : f32 from vector<1x4xf32>
        %58 = vector.insert %57, %55 [%46, %56] : f32 into vector<2x32xf32>
        affine.yield %58 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %28:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_8, %arg13 = %cst_8) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %45 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %51 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_9) -> (f32) {
            %55 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
            %56 = frisk.fragment "compute" -> (f32) {
              %57 = vector.extract %27[%arg11, %55] : f32 from vector<2x32xf32>
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
              frisk.fragment_yield %67 : f32
            }
            affine.yield %56 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %shuffleResult, %valid = gpu.shuffle xor %51, %c16_i32, %c64_i32 : f32
          %52 = arith.maxnumf %51, %shuffleResult : f32
          %shuffleResult_18, %valid_19 = gpu.shuffle xor %52, %c32_i32, %c64_i32 : f32
          %53 = arith.maxnumf %52, %shuffleResult_18 : f32
          %54 = vector.broadcast %53 : f32 to vector<1x1xf32>
          frisk.fragment_yield %54 : vector<1x1xf32>
        }
        %46 = vector.extract %45[0, 0] : f32 from vector<1x1xf32>
        %47 = vector.insert %46, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %48 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %51 = vector.extract %47[%arg11, 0] : f32 from vector<2x1xf32>
          %52 = vector.broadcast %51 : f32 to vector<1x1xf32>
          %53 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
          %54 = vector.broadcast %53 : f32 to vector<1x1xf32>
          %55 = arith.maxnumf %52, %54 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %55 : vector<1x1xf32>
        }
        %49 = vector.extract %48[0, 0] : f32 from vector<1x1xf32>
        %50 = vector.insert %49, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %47, %50 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %29:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %45 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %46 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %68 = vector.extract %28#1[%45, 0] : f32 from vector<2x1xf32>
          %69 = vector.from_elements %68, %68, %68, %68 : vector<1x4xf32>
          frisk.fragment_yield %69 : vector<1x4xf32>
        }
        %47 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %48 = vector.extract %46[0, 0] : f32 from vector<1x4xf32>
        %49 = vector.insert %48, %arg12 [%45, %47] : f32 into vector<2x32xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %51 = vector.extract %46[0, 1] : f32 from vector<1x4xf32>
        %52 = vector.insert %51, %49 [%45, %50] : f32 into vector<2x32xf32>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %54 = vector.extract %46[0, 2] : f32 from vector<1x4xf32>
        %55 = vector.insert %54, %52 [%45, %53] : f32 into vector<2x32xf32>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %57 = vector.extract %46[0, 3] : f32 from vector<1x4xf32>
        %58 = vector.insert %57, %55 [%45, %56] : f32 into vector<2x32xf32>
        %59 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %68 = vector.extract %26[%45, %47] : f32 from vector<2x32xf32>
          %69 = vector.extract %26[%45, %50] : f32 from vector<2x32xf32>
          %70 = vector.extract %26[%45, %53] : f32 from vector<2x32xf32>
          %71 = vector.extract %26[%45, %56] : f32 from vector<2x32xf32>
          %72 = vector.from_elements %68, %69, %70, %71 : vector<1x4xf32>
          %73 = vector.extract %58[%45, %47] : f32 from vector<2x32xf32>
          %74 = vector.extract %58[%45, %50] : f32 from vector<2x32xf32>
          %75 = vector.extract %58[%45, %53] : f32 from vector<2x32xf32>
          %76 = vector.extract %58[%45, %56] : f32 from vector<2x32xf32>
          %77 = vector.from_elements %73, %74, %75, %76 : vector<1x4xf32>
          %78 = arith.subf %72, %77 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %78 : vector<1x4xf32>
        }
        %60 = vector.extract %59[0, 0] : f32 from vector<1x4xf32>
        %61 = vector.insert %60, %arg13 [%45, %47] : f32 into vector<2x32xf32>
        %62 = vector.extract %59[0, 1] : f32 from vector<1x4xf32>
        %63 = vector.insert %62, %61 [%45, %50] : f32 into vector<2x32xf32>
        %64 = vector.extract %59[0, 2] : f32 from vector<1x4xf32>
        %65 = vector.insert %64, %63 [%45, %53] : f32 into vector<2x32xf32>
        %66 = vector.extract %59[0, 3] : f32 from vector<1x4xf32>
        %67 = vector.insert %66, %65 [%45, %56] : f32 into vector<2x32xf32>
        affine.yield %58, %67 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>}
      %30:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_8, %arg13 = %cst_8, %arg14 = %cst_8) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %45 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %54 = vector.extract %arg7[%arg11, 0] : f32 from vector<2x1xf32>
          %55 = vector.broadcast %54 : f32 to vector<1x1xf32>
          %56 = vector.extract %28#1[%arg11, 0] : f32 from vector<2x1xf32>
          %57 = vector.broadcast %56 : f32 to vector<1x1xf32>
          %58 = arith.subf %55, %57 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %58 : vector<1x1xf32>
        }
        %46 = vector.extract %45[0, 0] : f32 from vector<1x1xf32>
        %47 = vector.insert %46, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %48 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %54 = vector.extract %47[%arg11, 0] : f32 from vector<2x1xf32>
          %55 = vector.broadcast %54 : f32 to vector<1x1xf32>
          %56 = math.exp2 %55 {tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %56 : vector<1x1xf32>
        }
        %49 = vector.extract %48[0, 0] : f32 from vector<1x1xf32>
        %50 = vector.insert %49, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        %51 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %54 = vector.extract %arg6[%arg11, 0] : f32 from vector<2x1xf32>
          %55 = vector.broadcast %54 : f32 to vector<1x1xf32>
          %56 = vector.extract %50[%arg11, 0] : f32 from vector<2x1xf32>
          %57 = vector.broadcast %56 : f32 to vector<1x1xf32>
          %58 = arith.mulf %55, %57 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %58 : vector<1x1xf32>
        }
        %52 = vector.extract %51[0, 0] : f32 from vector<1x1xf32>
        %53 = vector.insert %52, %arg14 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %47, %50, %53 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %31:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %45 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %68 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
          %70 = vector.extract %29#1[%68, %69] : f32 from vector<2x32xf32>
          %71 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %72 = vector.extract %29#1[%68, %71] : f32 from vector<2x32xf32>
          %73 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %74 = vector.extract %29#1[%68, %73] : f32 from vector<2x32xf32>
          %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %76 = vector.extract %29#1[%68, %75] : f32 from vector<2x32xf32>
          %77 = vector.from_elements %70, %72, %74, %76 : vector<1x4xf32>
          %78 = math.exp2 %77 {tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %78 : vector<1x4xf32>
        }
        %46 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %47 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %48 = vector.extract %45[0, 0] : f32 from vector<1x4xf32>
        %49 = vector.insert %48, %arg12 [%46, %47] : f32 into vector<2x32xf32>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %51 = vector.extract %45[0, 1] : f32 from vector<1x4xf32>
        %52 = vector.insert %51, %49 [%46, %50] : f32 into vector<2x32xf32>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %54 = vector.extract %45[0, 2] : f32 from vector<1x4xf32>
        %55 = vector.insert %54, %52 [%46, %53] : f32 into vector<2x32xf32>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %57 = vector.extract %45[0, 3] : f32 from vector<1x4xf32>
        %58 = vector.insert %57, %55 [%46, %56] : f32 into vector<2x32xf32>
        %59 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %68 = vector.extract %58[%46, %47] : f32 from vector<2x32xf32>
          %69 = vector.extract %58[%46, %50] : f32 from vector<2x32xf32>
          %70 = vector.extract %58[%46, %53] : f32 from vector<2x32xf32>
          %71 = vector.extract %58[%46, %56] : f32 from vector<2x32xf32>
          %72 = vector.from_elements %68, %69, %70, %71 : vector<1x4xf32>
          frisk.fragment_yield %72 : vector<1x4xf32>
        }
        %60 = vector.extract %59[0, 0] : f32 from vector<1x4xf32>
        %61 = vector.insert %60, %arg13 [%46, %47] : f32 into vector<2x32xf32>
        %62 = vector.extract %59[0, 1] : f32 from vector<1x4xf32>
        %63 = vector.insert %62, %61 [%46, %50] : f32 into vector<2x32xf32>
        %64 = vector.extract %59[0, 2] : f32 from vector<1x4xf32>
        %65 = vector.insert %64, %63 [%46, %53] : f32 into vector<2x32xf32>
        %66 = vector.extract %59[0, 3] : f32 from vector<1x4xf32>
        %67 = vector.insert %66, %65 [%46, %56] : f32 into vector<2x32xf32>
        affine.yield %58, %67 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %32:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_8, %arg13 = %cst_8) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %45 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %51 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_3) -> (f32) {
            %55 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
            %56 = frisk.fragment "compute" -> (f32) {
              %57 = vector.extract %31#1[%arg11, %55] : f32 from vector<2x32xf32>
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
              frisk.fragment_yield %67 : f32
            }
            affine.yield %56 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %shuffleResult, %valid = gpu.shuffle xor %51, %c16_i32, %c64_i32 : f32
          %52 = arith.addf %51, %shuffleResult : f32
          %shuffleResult_18, %valid_19 = gpu.shuffle xor %52, %c32_i32, %c64_i32 : f32
          %53 = arith.addf %52, %shuffleResult_18 : f32
          %54 = vector.broadcast %53 : f32 to vector<1x1xf32>
          frisk.fragment_yield %54 : vector<1x1xf32>
        }
        %46 = vector.extract %45[0, 0] : f32 from vector<1x1xf32>
        %47 = vector.insert %46, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %48 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %51 = vector.extract %47[%arg11, 0] : f32 from vector<2x1xf32>
          %52 = vector.broadcast %51 : f32 to vector<1x1xf32>
          %53 = vector.extract %30#2[%arg11, 0] : f32 from vector<2x1xf32>
          %54 = vector.broadcast %53 : f32 to vector<1x1xf32>
          %55 = arith.addf %52, %54 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %55 : vector<1x1xf32>
        }
        %49 = vector.extract %48[0, 0] : f32 from vector<1x1xf32>
        %50 = vector.insert %49, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %47, %50 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %33 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %45 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %59 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %60 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
          %61 = vector.extract %31#0[%59, %60] : f32 from vector<2x32xf32>
          %62 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %63 = vector.extract %31#0[%59, %62] : f32 from vector<2x32xf32>
          %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %65 = vector.extract %31#0[%59, %64] : f32 from vector<2x32xf32>
          %66 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %67 = vector.extract %31#0[%59, %66] : f32 from vector<2x32xf32>
          %68 = vector.from_elements %61, %63, %65, %67 : vector<1x4xf32>
          %69 = arith.truncf %68 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %69 : vector<1x4xf16>
        }
        %46 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %47 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %48 = vector.extract %45[0, 0] : f16 from vector<1x4xf16>
        %49 = vector.insert %48, %arg12 [%46, %47] : f16 into vector<2x32xf16>
        %50 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %51 = vector.extract %45[0, 1] : f16 from vector<1x4xf16>
        %52 = vector.insert %51, %49 [%46, %50] : f16 into vector<2x32xf16>
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %54 = vector.extract %45[0, 2] : f16 from vector<1x4xf16>
        %55 = vector.insert %54, %52 [%46, %53] : f16 into vector<2x32xf16>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %57 = vector.extract %45[0, 3] : f16 from vector<1x4xf16>
        %58 = vector.insert %57, %55 [%46, %56] : f16 into vector<2x32xf16>
        affine.yield %58 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %34 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%7]
      %subview = memref.subview %arg2[0, %34, %arg4, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_11 = gpu.thread_id x
      %35 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %45 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %66 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_11]
          %67 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_11]
          %68 = frisk.read_matrix_from_shm %alloc[%66, %67] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %69 = vector.bitcast %68 : vector<4xi32> to vector<8xf16>
          %70 = vector.extract %69[0] : f16 from vector<8xf16>
          %71 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_11]
          %72 = vector.extract %69[%71] : f16 from vector<8xf16>
          %73 = vector.extract %69[1] : f16 from vector<8xf16>
          %74 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_11]
          %75 = vector.extract %69[%74] : f16 from vector<8xf16>
          %76 = vector.extract %69[2] : f16 from vector<8xf16>
          %77 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_11]
          %78 = vector.extract %69[%77] : f16 from vector<8xf16>
          %79 = vector.extract %69[3] : f16 from vector<8xf16>
          %80 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_11]
          %81 = vector.extract %69[%80] : f16 from vector<8xf16>
          %82 = vector.from_elements %70, %72, %73, %75, %76, %78, %79, %81 : vector<4x2xf16>
          frisk.fragment_yield %82 : vector<4x2xf16>
        }
        %46 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %47 = vector.extract %45[0, 0] : f16 from vector<4x2xf16>
        %48 = vector.insert %47, %arg12 [%46, 0] : f16 into vector<32x2xf16>
        %49 = vector.extract %45[0, 1] : f16 from vector<4x2xf16>
        %50 = vector.insert %49, %48 [%46, 1] : f16 into vector<32x2xf16>
        %51 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %52 = vector.extract %45[1, 0] : f16 from vector<4x2xf16>
        %53 = vector.insert %52, %50 [%51, 0] : f16 into vector<32x2xf16>
        %54 = vector.extract %45[1, 1] : f16 from vector<4x2xf16>
        %55 = vector.insert %54, %53 [%51, 1] : f16 into vector<32x2xf16>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %57 = vector.extract %45[2, 0] : f16 from vector<4x2xf16>
        %58 = vector.insert %57, %55 [%56, 0] : f16 into vector<32x2xf16>
        %59 = vector.extract %45[2, 1] : f16 from vector<4x2xf16>
        %60 = vector.insert %59, %58 [%56, 1] : f16 into vector<32x2xf16>
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %62 = vector.extract %45[3, 0] : f16 from vector<4x2xf16>
        %63 = vector.insert %62, %60 [%61, 0] : f16 into vector<32x2xf16>
        %64 = vector.extract %45[3, 1] : f16 from vector<4x2xf16>
        %65 = vector.insert %64, %63 [%61, 1] : f16 into vector<32x2xf16>
        affine.yield %65 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %36:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_4) -> (vector<1x4xf32>) {
          %86 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %89 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
            %90 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
            %91 = vector.extract %33[%89, %90] : f16 from vector<2x32xf16>
            %92 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
            %93 = vector.extract %33[%89, %92] : f16 from vector<2x32xf16>
            %94 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
            %95 = vector.extract %33[%89, %94] : f16 from vector<2x32xf16>
            %96 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
            %97 = vector.extract %33[%89, %96] : f16 from vector<2x32xf16>
            %98 = vector.from_elements %91, %93, %95, %97 : vector<1x4xf16>
            frisk.fragment_yield %98 : vector<1x4xf16>
          }
          %87 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %89 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
            %90 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %91 = vector.extract %35[%89, %90] : f16 from vector<32x2xf16>
            %92 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
            %93 = vector.extract %35[%92, %90] : f16 from vector<32x2xf16>
            %94 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
            %95 = vector.extract %35[%94, %90] : f16 from vector<32x2xf16>
            %96 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
            %97 = vector.extract %35[%96, %90] : f16 from vector<32x2xf16>
            %98 = vector.from_elements %91, %93, %95, %97 : vector<4x1xf16>
            frisk.fragment_yield %98 : vector<4x1xf16>
          }
          %88 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %89 = frisk.warp_mma_rr %86, %87, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %89 : vector<1x4xf32>
          }
          affine.yield %88 : vector<1x4xf32>
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
        %59 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %86 = vector.extract %30#1[%46, 0] : f32 from vector<2x1xf32>
          %87 = vector.from_elements %86, %86, %86, %86 : vector<1x4xf32>
          frisk.fragment_yield %87 : vector<1x4xf32>
        }
        %60 = vector.extract %59[0, 0] : f32 from vector<1x4xf32>
        %61 = vector.insert %60, %arg13 [%46, %47] : f32 into vector<2x8xf32>
        %62 = vector.extract %59[0, 1] : f32 from vector<1x4xf32>
        %63 = vector.insert %62, %61 [%46, %50] : f32 into vector<2x8xf32>
        %64 = vector.extract %59[0, 2] : f32 from vector<1x4xf32>
        %65 = vector.insert %64, %63 [%46, %53] : f32 into vector<2x8xf32>
        %66 = vector.extract %59[0, 3] : f32 from vector<1x4xf32>
        %67 = vector.insert %66, %65 [%46, %56] : f32 into vector<2x8xf32>
        %68 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %86 = vector.extract %arg5[%46, %47] : f32 from vector<2x8xf32>
          %87 = vector.extract %arg5[%46, %50] : f32 from vector<2x8xf32>
          %88 = vector.extract %arg5[%46, %53] : f32 from vector<2x8xf32>
          %89 = vector.extract %arg5[%46, %56] : f32 from vector<2x8xf32>
          %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf32>
          %91 = vector.extract %67[%46, %47] : f32 from vector<2x8xf32>
          %92 = vector.extract %67[%46, %50] : f32 from vector<2x8xf32>
          %93 = vector.extract %67[%46, %53] : f32 from vector<2x8xf32>
          %94 = vector.extract %67[%46, %56] : f32 from vector<2x8xf32>
          %95 = vector.from_elements %91, %92, %93, %94 : vector<1x4xf32>
          %96 = arith.mulf %90, %95 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %96 : vector<1x4xf32>
        }
        %69 = vector.extract %68[0, 0] : f32 from vector<1x4xf32>
        %70 = vector.insert %69, %arg14 [%46, %47] : f32 into vector<2x8xf32>
        %71 = vector.extract %68[0, 1] : f32 from vector<1x4xf32>
        %72 = vector.insert %71, %70 [%46, %50] : f32 into vector<2x8xf32>
        %73 = vector.extract %68[0, 2] : f32 from vector<1x4xf32>
        %74 = vector.insert %73, %72 [%46, %53] : f32 into vector<2x8xf32>
        %75 = vector.extract %68[0, 3] : f32 from vector<1x4xf32>
        %76 = vector.insert %75, %74 [%46, %56] : f32 into vector<2x8xf32>
        %77 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %86 = vector.extract %76[%46, %47] : f32 from vector<2x8xf32>
          %87 = vector.extract %76[%46, %50] : f32 from vector<2x8xf32>
          %88 = vector.extract %76[%46, %53] : f32 from vector<2x8xf32>
          %89 = vector.extract %76[%46, %56] : f32 from vector<2x8xf32>
          %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf32>
          %91 = vector.extract %58[%46, %47] : f32 from vector<2x8xf32>
          %92 = vector.extract %58[%46, %50] : f32 from vector<2x8xf32>
          %93 = vector.extract %58[%46, %53] : f32 from vector<2x8xf32>
          %94 = vector.extract %58[%46, %56] : f32 from vector<2x8xf32>
          %95 = vector.from_elements %91, %92, %93, %94 : vector<1x4xf32>
          %96 = arith.addf %90, %95 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %96 : vector<1x4xf32>
        }
        %78 = vector.extract %77[0, 0] : f32 from vector<1x4xf32>
        %79 = vector.insert %78, %arg15 [%46, %47] : f32 into vector<2x8xf32>
        %80 = vector.extract %77[0, 1] : f32 from vector<1x4xf32>
        %81 = vector.insert %80, %79 [%46, %50] : f32 into vector<2x8xf32>
        %82 = vector.extract %77[0, 2] : f32 from vector<1x4xf32>
        %83 = vector.insert %82, %81 [%46, %53] : f32 into vector<2x8xf32>
        %84 = vector.extract %77[0, 3] : f32 from vector<1x4xf32>
        %85 = vector.insert %84, %83 [%46, %56] : f32 into vector<2x8xf32>
        affine.yield %58, %67, %76, %85 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_12 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_13 = memref.subview %arg2[0, %34, %arg4, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_13 to %alloc_12 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_12 : memref<128x32xf16, 3>
      %37 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %45 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %66 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_11]
          %67 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_11]
          %68 = frisk.read_matrix_from_shm %alloc_12[%66, %67] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %69 = vector.bitcast %68 : vector<4xi32> to vector<8xf16>
          %70 = vector.extract %69[0] : f16 from vector<8xf16>
          %71 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_11]
          %72 = vector.extract %69[%71] : f16 from vector<8xf16>
          %73 = vector.extract %69[1] : f16 from vector<8xf16>
          %74 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_11]
          %75 = vector.extract %69[%74] : f16 from vector<8xf16>
          %76 = vector.extract %69[2] : f16 from vector<8xf16>
          %77 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_11]
          %78 = vector.extract %69[%77] : f16 from vector<8xf16>
          %79 = vector.extract %69[3] : f16 from vector<8xf16>
          %80 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_11]
          %81 = vector.extract %69[%80] : f16 from vector<8xf16>
          %82 = vector.from_elements %70, %72, %73, %75, %76, %78, %79, %81 : vector<4x2xf16>
          frisk.fragment_yield %82 : vector<4x2xf16>
        }
        %46 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %47 = vector.extract %45[0, 0] : f16 from vector<4x2xf16>
        %48 = vector.insert %47, %arg12 [%46, 0] : f16 into vector<32x2xf16>
        %49 = vector.extract %45[0, 1] : f16 from vector<4x2xf16>
        %50 = vector.insert %49, %48 [%46, 1] : f16 into vector<32x2xf16>
        %51 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %52 = vector.extract %45[1, 0] : f16 from vector<4x2xf16>
        %53 = vector.insert %52, %50 [%51, 0] : f16 into vector<32x2xf16>
        %54 = vector.extract %45[1, 1] : f16 from vector<4x2xf16>
        %55 = vector.insert %54, %53 [%51, 1] : f16 into vector<32x2xf16>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %57 = vector.extract %45[2, 0] : f16 from vector<4x2xf16>
        %58 = vector.insert %57, %55 [%56, 0] : f16 into vector<32x2xf16>
        %59 = vector.extract %45[2, 1] : f16 from vector<4x2xf16>
        %60 = vector.insert %59, %58 [%56, 1] : f16 into vector<32x2xf16>
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %62 = vector.extract %45[3, 0] : f16 from vector<4x2xf16>
        %63 = vector.insert %62, %60 [%61, 0] : f16 into vector<32x2xf16>
        %64 = vector.extract %45[3, 1] : f16 from vector<4x2xf16>
        %65 = vector.insert %64, %63 [%61, 1] : f16 into vector<32x2xf16>
        affine.yield %65 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %38:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_4) -> (vector<1x4xf32>) {
          %86 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %89 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
            %90 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
            %91 = vector.extract %33[%89, %90] : f16 from vector<2x32xf16>
            %92 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
            %93 = vector.extract %33[%89, %92] : f16 from vector<2x32xf16>
            %94 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
            %95 = vector.extract %33[%89, %94] : f16 from vector<2x32xf16>
            %96 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
            %97 = vector.extract %33[%89, %96] : f16 from vector<2x32xf16>
            %98 = vector.from_elements %91, %93, %95, %97 : vector<1x4xf16>
            frisk.fragment_yield %98 : vector<1x4xf16>
          }
          %87 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %89 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
            %90 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %91 = vector.extract %37[%89, %90] : f16 from vector<32x2xf16>
            %92 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
            %93 = vector.extract %37[%92, %90] : f16 from vector<32x2xf16>
            %94 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
            %95 = vector.extract %37[%94, %90] : f16 from vector<32x2xf16>
            %96 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
            %97 = vector.extract %37[%96, %90] : f16 from vector<32x2xf16>
            %98 = vector.from_elements %91, %93, %95, %97 : vector<4x1xf16>
            frisk.fragment_yield %98 : vector<4x1xf16>
          }
          %88 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %89 = frisk.warp_mma_rr %86, %87, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %89 : vector<1x4xf32>
          }
          affine.yield %88 : vector<1x4xf32>
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
        %59 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %86 = vector.extract %30#1[%46, 0] : f32 from vector<2x1xf32>
          %87 = vector.from_elements %86, %86, %86, %86 : vector<1x4xf32>
          frisk.fragment_yield %87 : vector<1x4xf32>
        }
        %60 = vector.extract %59[0, 0] : f32 from vector<1x4xf32>
        %61 = vector.insert %60, %arg13 [%46, %47] : f32 into vector<2x8xf32>
        %62 = vector.extract %59[0, 1] : f32 from vector<1x4xf32>
        %63 = vector.insert %62, %61 [%46, %50] : f32 into vector<2x8xf32>
        %64 = vector.extract %59[0, 2] : f32 from vector<1x4xf32>
        %65 = vector.insert %64, %63 [%46, %53] : f32 into vector<2x8xf32>
        %66 = vector.extract %59[0, 3] : f32 from vector<1x4xf32>
        %67 = vector.insert %66, %65 [%46, %56] : f32 into vector<2x8xf32>
        %68 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %86 = vector.extract %arg8[%46, %47] : f32 from vector<2x8xf32>
          %87 = vector.extract %arg8[%46, %50] : f32 from vector<2x8xf32>
          %88 = vector.extract %arg8[%46, %53] : f32 from vector<2x8xf32>
          %89 = vector.extract %arg8[%46, %56] : f32 from vector<2x8xf32>
          %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf32>
          %91 = vector.extract %67[%46, %47] : f32 from vector<2x8xf32>
          %92 = vector.extract %67[%46, %50] : f32 from vector<2x8xf32>
          %93 = vector.extract %67[%46, %53] : f32 from vector<2x8xf32>
          %94 = vector.extract %67[%46, %56] : f32 from vector<2x8xf32>
          %95 = vector.from_elements %91, %92, %93, %94 : vector<1x4xf32>
          %96 = arith.mulf %90, %95 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %96 : vector<1x4xf32>
        }
        %69 = vector.extract %68[0, 0] : f32 from vector<1x4xf32>
        %70 = vector.insert %69, %arg14 [%46, %47] : f32 into vector<2x8xf32>
        %71 = vector.extract %68[0, 1] : f32 from vector<1x4xf32>
        %72 = vector.insert %71, %70 [%46, %50] : f32 into vector<2x8xf32>
        %73 = vector.extract %68[0, 2] : f32 from vector<1x4xf32>
        %74 = vector.insert %73, %72 [%46, %53] : f32 into vector<2x8xf32>
        %75 = vector.extract %68[0, 3] : f32 from vector<1x4xf32>
        %76 = vector.insert %75, %74 [%46, %56] : f32 into vector<2x8xf32>
        %77 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %86 = vector.extract %76[%46, %47] : f32 from vector<2x8xf32>
          %87 = vector.extract %76[%46, %50] : f32 from vector<2x8xf32>
          %88 = vector.extract %76[%46, %53] : f32 from vector<2x8xf32>
          %89 = vector.extract %76[%46, %56] : f32 from vector<2x8xf32>
          %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf32>
          %91 = vector.extract %58[%46, %47] : f32 from vector<2x8xf32>
          %92 = vector.extract %58[%46, %50] : f32 from vector<2x8xf32>
          %93 = vector.extract %58[%46, %53] : f32 from vector<2x8xf32>
          %94 = vector.extract %58[%46, %56] : f32 from vector<2x8xf32>
          %95 = vector.from_elements %91, %92, %93, %94 : vector<1x4xf32>
          %96 = arith.addf %90, %95 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %96 : vector<1x4xf32>
        }
        %78 = vector.extract %77[0, 0] : f32 from vector<1x4xf32>
        %79 = vector.insert %78, %arg15 [%46, %47] : f32 into vector<2x8xf32>
        %80 = vector.extract %77[0, 1] : f32 from vector<1x4xf32>
        %81 = vector.insert %80, %79 [%46, %50] : f32 into vector<2x8xf32>
        %82 = vector.extract %77[0, 2] : f32 from vector<1x4xf32>
        %83 = vector.insert %82, %81 [%46, %53] : f32 into vector<2x8xf32>
        %84 = vector.extract %77[0, 3] : f32 from vector<1x4xf32>
        %85 = vector.insert %84, %83 [%46, %56] : f32 into vector<2x8xf32>
        affine.yield %58, %67, %76, %85 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_14 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_15 = memref.subview %arg2[0, %34, %arg4, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_15 to %alloc_14 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_14 : memref<128x32xf16, 3>
      %39 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %45 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %66 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_11]
          %67 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_11]
          %68 = frisk.read_matrix_from_shm %alloc_14[%66, %67] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %69 = vector.bitcast %68 : vector<4xi32> to vector<8xf16>
          %70 = vector.extract %69[0] : f16 from vector<8xf16>
          %71 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_11]
          %72 = vector.extract %69[%71] : f16 from vector<8xf16>
          %73 = vector.extract %69[1] : f16 from vector<8xf16>
          %74 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_11]
          %75 = vector.extract %69[%74] : f16 from vector<8xf16>
          %76 = vector.extract %69[2] : f16 from vector<8xf16>
          %77 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_11]
          %78 = vector.extract %69[%77] : f16 from vector<8xf16>
          %79 = vector.extract %69[3] : f16 from vector<8xf16>
          %80 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_11]
          %81 = vector.extract %69[%80] : f16 from vector<8xf16>
          %82 = vector.from_elements %70, %72, %73, %75, %76, %78, %79, %81 : vector<4x2xf16>
          frisk.fragment_yield %82 : vector<4x2xf16>
        }
        %46 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %47 = vector.extract %45[0, 0] : f16 from vector<4x2xf16>
        %48 = vector.insert %47, %arg12 [%46, 0] : f16 into vector<32x2xf16>
        %49 = vector.extract %45[0, 1] : f16 from vector<4x2xf16>
        %50 = vector.insert %49, %48 [%46, 1] : f16 into vector<32x2xf16>
        %51 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %52 = vector.extract %45[1, 0] : f16 from vector<4x2xf16>
        %53 = vector.insert %52, %50 [%51, 0] : f16 into vector<32x2xf16>
        %54 = vector.extract %45[1, 1] : f16 from vector<4x2xf16>
        %55 = vector.insert %54, %53 [%51, 1] : f16 into vector<32x2xf16>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %57 = vector.extract %45[2, 0] : f16 from vector<4x2xf16>
        %58 = vector.insert %57, %55 [%56, 0] : f16 into vector<32x2xf16>
        %59 = vector.extract %45[2, 1] : f16 from vector<4x2xf16>
        %60 = vector.insert %59, %58 [%56, 1] : f16 into vector<32x2xf16>
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %62 = vector.extract %45[3, 0] : f16 from vector<4x2xf16>
        %63 = vector.insert %62, %60 [%61, 0] : f16 into vector<32x2xf16>
        %64 = vector.extract %45[3, 1] : f16 from vector<4x2xf16>
        %65 = vector.insert %64, %63 [%61, 1] : f16 into vector<32x2xf16>
        affine.yield %65 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %40:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_4) -> (vector<1x4xf32>) {
          %86 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %89 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
            %90 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
            %91 = vector.extract %33[%89, %90] : f16 from vector<2x32xf16>
            %92 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
            %93 = vector.extract %33[%89, %92] : f16 from vector<2x32xf16>
            %94 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
            %95 = vector.extract %33[%89, %94] : f16 from vector<2x32xf16>
            %96 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
            %97 = vector.extract %33[%89, %96] : f16 from vector<2x32xf16>
            %98 = vector.from_elements %91, %93, %95, %97 : vector<1x4xf16>
            frisk.fragment_yield %98 : vector<1x4xf16>
          }
          %87 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %89 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
            %90 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %91 = vector.extract %39[%89, %90] : f16 from vector<32x2xf16>
            %92 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
            %93 = vector.extract %39[%92, %90] : f16 from vector<32x2xf16>
            %94 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
            %95 = vector.extract %39[%94, %90] : f16 from vector<32x2xf16>
            %96 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
            %97 = vector.extract %39[%96, %90] : f16 from vector<32x2xf16>
            %98 = vector.from_elements %91, %93, %95, %97 : vector<4x1xf16>
            frisk.fragment_yield %98 : vector<4x1xf16>
          }
          %88 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %89 = frisk.warp_mma_rr %86, %87, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %89 : vector<1x4xf32>
          }
          affine.yield %88 : vector<1x4xf32>
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
        %59 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %86 = vector.extract %30#1[%46, 0] : f32 from vector<2x1xf32>
          %87 = vector.from_elements %86, %86, %86, %86 : vector<1x4xf32>
          frisk.fragment_yield %87 : vector<1x4xf32>
        }
        %60 = vector.extract %59[0, 0] : f32 from vector<1x4xf32>
        %61 = vector.insert %60, %arg13 [%46, %47] : f32 into vector<2x8xf32>
        %62 = vector.extract %59[0, 1] : f32 from vector<1x4xf32>
        %63 = vector.insert %62, %61 [%46, %50] : f32 into vector<2x8xf32>
        %64 = vector.extract %59[0, 2] : f32 from vector<1x4xf32>
        %65 = vector.insert %64, %63 [%46, %53] : f32 into vector<2x8xf32>
        %66 = vector.extract %59[0, 3] : f32 from vector<1x4xf32>
        %67 = vector.insert %66, %65 [%46, %56] : f32 into vector<2x8xf32>
        %68 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %86 = vector.extract %arg9[%46, %47] : f32 from vector<2x8xf32>
          %87 = vector.extract %arg9[%46, %50] : f32 from vector<2x8xf32>
          %88 = vector.extract %arg9[%46, %53] : f32 from vector<2x8xf32>
          %89 = vector.extract %arg9[%46, %56] : f32 from vector<2x8xf32>
          %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf32>
          %91 = vector.extract %67[%46, %47] : f32 from vector<2x8xf32>
          %92 = vector.extract %67[%46, %50] : f32 from vector<2x8xf32>
          %93 = vector.extract %67[%46, %53] : f32 from vector<2x8xf32>
          %94 = vector.extract %67[%46, %56] : f32 from vector<2x8xf32>
          %95 = vector.from_elements %91, %92, %93, %94 : vector<1x4xf32>
          %96 = arith.mulf %90, %95 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %96 : vector<1x4xf32>
        }
        %69 = vector.extract %68[0, 0] : f32 from vector<1x4xf32>
        %70 = vector.insert %69, %arg14 [%46, %47] : f32 into vector<2x8xf32>
        %71 = vector.extract %68[0, 1] : f32 from vector<1x4xf32>
        %72 = vector.insert %71, %70 [%46, %50] : f32 into vector<2x8xf32>
        %73 = vector.extract %68[0, 2] : f32 from vector<1x4xf32>
        %74 = vector.insert %73, %72 [%46, %53] : f32 into vector<2x8xf32>
        %75 = vector.extract %68[0, 3] : f32 from vector<1x4xf32>
        %76 = vector.insert %75, %74 [%46, %56] : f32 into vector<2x8xf32>
        %77 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %86 = vector.extract %76[%46, %47] : f32 from vector<2x8xf32>
          %87 = vector.extract %76[%46, %50] : f32 from vector<2x8xf32>
          %88 = vector.extract %76[%46, %53] : f32 from vector<2x8xf32>
          %89 = vector.extract %76[%46, %56] : f32 from vector<2x8xf32>
          %90 = vector.from_elements %86, %87, %88, %89 : vector<1x4xf32>
          %91 = vector.extract %58[%46, %47] : f32 from vector<2x8xf32>
          %92 = vector.extract %58[%46, %50] : f32 from vector<2x8xf32>
          %93 = vector.extract %58[%46, %53] : f32 from vector<2x8xf32>
          %94 = vector.extract %58[%46, %56] : f32 from vector<2x8xf32>
          %95 = vector.from_elements %91, %92, %93, %94 : vector<1x4xf32>
          %96 = arith.addf %90, %95 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %96 : vector<1x4xf32>
        }
        %78 = vector.extract %77[0, 0] : f32 from vector<1x4xf32>
        %79 = vector.insert %78, %arg15 [%46, %47] : f32 into vector<2x8xf32>
        %80 = vector.extract %77[0, 1] : f32 from vector<1x4xf32>
        %81 = vector.insert %80, %79 [%46, %50] : f32 into vector<2x8xf32>
        %82 = vector.extract %77[0, 2] : f32 from vector<1x4xf32>
        %83 = vector.insert %82, %81 [%46, %53] : f32 into vector<2x8xf32>
        %84 = vector.extract %77[0, 3] : f32 from vector<1x4xf32>
        %85 = vector.insert %84, %83 [%46, %56] : f32 into vector<2x8xf32>
        affine.yield %58, %67, %76, %85 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %subview_17 = memref.subview %arg2[0, %34, %arg4, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      frisk.async_copy_gm_to_shm %subview_17 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_16 : memref<128x32xf16, 3>
      %41 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %45 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %66 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_11]
          %67 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_11]
          %68 = frisk.read_matrix_from_shm %alloc_16[%66, %67] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %69 = vector.bitcast %68 : vector<4xi32> to vector<8xf16>
          %70 = vector.extract %69[0] : f16 from vector<8xf16>
          %71 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_11]
          %72 = vector.extract %69[%71] : f16 from vector<8xf16>
          %73 = vector.extract %69[1] : f16 from vector<8xf16>
          %74 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_11]
          %75 = vector.extract %69[%74] : f16 from vector<8xf16>
          %76 = vector.extract %69[2] : f16 from vector<8xf16>
          %77 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_11]
          %78 = vector.extract %69[%77] : f16 from vector<8xf16>
          %79 = vector.extract %69[3] : f16 from vector<8xf16>
          %80 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_11]
          %81 = vector.extract %69[%80] : f16 from vector<8xf16>
          %82 = vector.from_elements %70, %72, %73, %75, %76, %78, %79, %81 : vector<4x2xf16>
          frisk.fragment_yield %82 : vector<4x2xf16>
        }
        %46 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %47 = vector.extract %45[0, 0] : f16 from vector<4x2xf16>
        %48 = vector.insert %47, %arg12 [%46, 0] : f16 into vector<32x2xf16>
        %49 = vector.extract %45[0, 1] : f16 from vector<4x2xf16>
        %50 = vector.insert %49, %48 [%46, 1] : f16 into vector<32x2xf16>
        %51 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %52 = vector.extract %45[1, 0] : f16 from vector<4x2xf16>
        %53 = vector.insert %52, %50 [%51, 0] : f16 into vector<32x2xf16>
        %54 = vector.extract %45[1, 1] : f16 from vector<4x2xf16>
        %55 = vector.insert %54, %53 [%51, 1] : f16 into vector<32x2xf16>
        %56 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %57 = vector.extract %45[2, 0] : f16 from vector<4x2xf16>
        %58 = vector.insert %57, %55 [%56, 0] : f16 into vector<32x2xf16>
        %59 = vector.extract %45[2, 1] : f16 from vector<4x2xf16>
        %60 = vector.insert %59, %58 [%56, 1] : f16 into vector<32x2xf16>
        %61 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %62 = vector.extract %45[3, 0] : f16 from vector<4x2xf16>
        %63 = vector.insert %62, %60 [%61, 0] : f16 into vector<32x2xf16>
        %64 = vector.extract %45[3, 1] : f16 from vector<4x2xf16>
        %65 = vector.insert %64, %63 [%61, 1] : f16 into vector<32x2xf16>
        affine.yield %65 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %42:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_4) -> (vector<1x4xf32>) {
          %95 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %98 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
            %99 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
            %100 = vector.extract %33[%98, %99] : f16 from vector<2x32xf16>
            %101 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
            %102 = vector.extract %33[%98, %101] : f16 from vector<2x32xf16>
            %103 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
            %104 = vector.extract %33[%98, %103] : f16 from vector<2x32xf16>
            %105 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
            %106 = vector.extract %33[%98, %105] : f16 from vector<2x32xf16>
            %107 = vector.from_elements %100, %102, %104, %106 : vector<1x4xf16>
            frisk.fragment_yield %107 : vector<1x4xf16>
          }
          %96 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %98 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
            %99 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %100 = vector.extract %41[%98, %99] : f16 from vector<32x2xf16>
            %101 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
            %102 = vector.extract %41[%101, %99] : f16 from vector<32x2xf16>
            %103 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
            %104 = vector.extract %41[%103, %99] : f16 from vector<32x2xf16>
            %105 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
            %106 = vector.extract %41[%105, %99] : f16 from vector<32x2xf16>
            %107 = vector.from_elements %100, %102, %104, %106 : vector<4x1xf16>
            frisk.fragment_yield %107 : vector<4x1xf16>
          }
          %97 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %98 = frisk.warp_mma_rr %95, %96, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %98 : vector<1x4xf32>
          }
          affine.yield %97 : vector<1x4xf32>
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
        %59 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %95 = vector.extract %30#1[%46, 0] : f32 from vector<2x1xf32>
          %96 = vector.from_elements %95, %95, %95, %95 : vector<1x4xf32>
          frisk.fragment_yield %96 : vector<1x4xf32>
        }
        %60 = vector.extract %59[0, 0] : f32 from vector<1x4xf32>
        %61 = vector.insert %60, %arg13 [%46, %47] : f32 into vector<2x8xf32>
        %62 = vector.extract %59[0, 1] : f32 from vector<1x4xf32>
        %63 = vector.insert %62, %61 [%46, %50] : f32 into vector<2x8xf32>
        %64 = vector.extract %59[0, 2] : f32 from vector<1x4xf32>
        %65 = vector.insert %64, %63 [%46, %53] : f32 into vector<2x8xf32>
        %66 = vector.extract %59[0, 3] : f32 from vector<1x4xf32>
        %67 = vector.insert %66, %65 [%46, %56] : f32 into vector<2x8xf32>
        %68 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %95 = vector.extract %arg10[%46, %47] : f32 from vector<2x8xf32>
          %96 = vector.extract %arg10[%46, %50] : f32 from vector<2x8xf32>
          %97 = vector.extract %arg10[%46, %53] : f32 from vector<2x8xf32>
          %98 = vector.extract %arg10[%46, %56] : f32 from vector<2x8xf32>
          %99 = vector.from_elements %95, %96, %97, %98 : vector<1x4xf32>
          %100 = vector.extract %67[%46, %47] : f32 from vector<2x8xf32>
          %101 = vector.extract %67[%46, %50] : f32 from vector<2x8xf32>
          %102 = vector.extract %67[%46, %53] : f32 from vector<2x8xf32>
          %103 = vector.extract %67[%46, %56] : f32 from vector<2x8xf32>
          %104 = vector.from_elements %100, %101, %102, %103 : vector<1x4xf32>
          %105 = arith.mulf %99, %104 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %105 : vector<1x4xf32>
        }
        %69 = vector.extract %68[0, 0] : f32 from vector<1x4xf32>
        %70 = vector.insert %69, %arg14 [%46, %47] : f32 into vector<2x8xf32>
        %71 = vector.extract %68[0, 1] : f32 from vector<1x4xf32>
        %72 = vector.insert %71, %70 [%46, %50] : f32 into vector<2x8xf32>
        %73 = vector.extract %68[0, 2] : f32 from vector<1x4xf32>
        %74 = vector.insert %73, %72 [%46, %53] : f32 into vector<2x8xf32>
        %75 = vector.extract %68[0, 3] : f32 from vector<1x4xf32>
        %76 = vector.insert %75, %74 [%46, %56] : f32 into vector<2x8xf32>
        %77 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %95 = vector.extract %76[%46, %47] : f32 from vector<2x8xf32>
          %96 = vector.extract %76[%46, %50] : f32 from vector<2x8xf32>
          %97 = vector.extract %76[%46, %53] : f32 from vector<2x8xf32>
          %98 = vector.extract %76[%46, %56] : f32 from vector<2x8xf32>
          %99 = vector.from_elements %95, %96, %97, %98 : vector<1x4xf32>
          %100 = vector.extract %58[%46, %47] : f32 from vector<2x8xf32>
          %101 = vector.extract %58[%46, %50] : f32 from vector<2x8xf32>
          %102 = vector.extract %58[%46, %53] : f32 from vector<2x8xf32>
          %103 = vector.extract %58[%46, %56] : f32 from vector<2x8xf32>
          %104 = vector.from_elements %100, %101, %102, %103 : vector<1x4xf32>
          %105 = arith.addf %99, %104 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %105 : vector<1x4xf32>
        }
        %78 = vector.extract %77[0, 0] : f32 from vector<1x4xf32>
        %79 = vector.insert %78, %arg15 [%46, %47] : f32 into vector<2x8xf32>
        %80 = vector.extract %77[0, 1] : f32 from vector<1x4xf32>
        %81 = vector.insert %80, %79 [%46, %50] : f32 into vector<2x8xf32>
        %82 = vector.extract %77[0, 2] : f32 from vector<1x4xf32>
        %83 = vector.insert %82, %81 [%46, %53] : f32 into vector<2x8xf32>
        %84 = vector.extract %77[0, 3] : f32 from vector<1x4xf32>
        %85 = vector.insert %84, %83 [%46, %56] : f32 into vector<2x8xf32>
        %86 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %95 = vector.extract %36#3[%46, %47] : f32 from vector<2x8xf32>
          %96 = vector.extract %36#3[%46, %50] : f32 from vector<2x8xf32>
          %97 = vector.extract %36#3[%46, %53] : f32 from vector<2x8xf32>
          %98 = vector.extract %36#3[%46, %56] : f32 from vector<2x8xf32>
          %99 = vector.from_elements %95, %96, %97, %98 : vector<1x4xf32>
          frisk.fragment_yield %99 : vector<1x4xf32>
        }
        %87 = vector.extract %86[0, 0] : f32 from vector<1x4xf32>
        %88 = vector.insert %87, %arg16 [%46, %47] : f32 into vector<2x8xf32>
        %89 = vector.extract %86[0, 1] : f32 from vector<1x4xf32>
        %90 = vector.insert %89, %88 [%46, %50] : f32 into vector<2x8xf32>
        %91 = vector.extract %86[0, 2] : f32 from vector<1x4xf32>
        %92 = vector.insert %91, %90 [%46, %53] : f32 into vector<2x8xf32>
        %93 = vector.extract %86[0, 3] : f32 from vector<1x4xf32>
        %94 = vector.insert %93, %92 [%46, %56] : f32 into vector<2x8xf32>
        affine.yield %58, %67, %76, %85, %94 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %43:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_8, %arg13 = %cst_8) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %45 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %51 = vector.extract %32#1[%arg11, 0] : f32 from vector<2x1xf32>
          %52 = vector.broadcast %51 : f32 to vector<1x1xf32>
          frisk.fragment_yield %52 : vector<1x1xf32>
        }
        %46 = vector.extract %45[0, 0] : f32 from vector<1x1xf32>
        %47 = vector.insert %46, %arg12 [%arg11, 0] : f32 into vector<2x1xf32>
        %48 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %51 = vector.extract %28#1[%arg11, 0] : f32 from vector<2x1xf32>
          %52 = vector.broadcast %51 : f32 to vector<1x1xf32>
          frisk.fragment_yield %52 : vector<1x1xf32>
        }
        %49 = vector.extract %48[0, 0] : f32 from vector<1x1xf32>
        %50 = vector.insert %49, %arg13 [%arg11, 0] : f32 into vector<2x1xf32>
        affine.yield %47, %50 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %44:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %45 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %77 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
          %79 = vector.extract %38#3[%77, %78] : f32 from vector<2x8xf32>
          %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %81 = vector.extract %38#3[%77, %80] : f32 from vector<2x8xf32>
          %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %83 = vector.extract %38#3[%77, %82] : f32 from vector<2x8xf32>
          %84 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %85 = vector.extract %38#3[%77, %84] : f32 from vector<2x8xf32>
          %86 = vector.from_elements %79, %81, %83, %85 : vector<1x4xf32>
          frisk.fragment_yield %86 : vector<1x4xf32>
        }
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
        %59 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %77 = vector.extract %40#3[%46, %47] : f32 from vector<2x8xf32>
          %78 = vector.extract %40#3[%46, %50] : f32 from vector<2x8xf32>
          %79 = vector.extract %40#3[%46, %53] : f32 from vector<2x8xf32>
          %80 = vector.extract %40#3[%46, %56] : f32 from vector<2x8xf32>
          %81 = vector.from_elements %77, %78, %79, %80 : vector<1x4xf32>
          frisk.fragment_yield %81 : vector<1x4xf32>
        }
        %60 = vector.extract %59[0, 0] : f32 from vector<1x4xf32>
        %61 = vector.insert %60, %arg13 [%46, %47] : f32 into vector<2x8xf32>
        %62 = vector.extract %59[0, 1] : f32 from vector<1x4xf32>
        %63 = vector.insert %62, %61 [%46, %50] : f32 into vector<2x8xf32>
        %64 = vector.extract %59[0, 2] : f32 from vector<1x4xf32>
        %65 = vector.insert %64, %63 [%46, %53] : f32 into vector<2x8xf32>
        %66 = vector.extract %59[0, 3] : f32 from vector<1x4xf32>
        %67 = vector.insert %66, %65 [%46, %56] : f32 into vector<2x8xf32>
        %68 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %77 = vector.extract %42#3[%46, %47] : f32 from vector<2x8xf32>
          %78 = vector.extract %42#3[%46, %50] : f32 from vector<2x8xf32>
          %79 = vector.extract %42#3[%46, %53] : f32 from vector<2x8xf32>
          %80 = vector.extract %42#3[%46, %56] : f32 from vector<2x8xf32>
          %81 = vector.from_elements %77, %78, %79, %80 : vector<1x4xf32>
          frisk.fragment_yield %81 : vector<1x4xf32>
        }
        %69 = vector.extract %68[0, 0] : f32 from vector<1x4xf32>
        %70 = vector.insert %69, %arg14 [%46, %47] : f32 into vector<2x8xf32>
        %71 = vector.extract %68[0, 1] : f32 from vector<1x4xf32>
        %72 = vector.insert %71, %70 [%46, %50] : f32 into vector<2x8xf32>
        %73 = vector.extract %68[0, 2] : f32 from vector<1x4xf32>
        %74 = vector.insert %73, %72 [%46, %53] : f32 into vector<2x8xf32>
        %75 = vector.extract %68[0, 3] : f32 from vector<1x4xf32>
        %76 = vector.insert %75, %74 [%46, %56] : f32 into vector<2x8xf32>
        affine.yield %58, %67, %76 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
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
      %25 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %102 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %103 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %104 = vector.extract %13#5[%102, %103] : f32 from vector<2x8xf32>
        %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %106 = vector.extract %13#5[%102, %105] : f32 from vector<2x8xf32>
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %108 = vector.extract %13#5[%102, %107] : f32 from vector<2x8xf32>
        %109 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %110 = vector.extract %13#5[%102, %109] : f32 from vector<2x8xf32>
        %111 = arith.select %16, %104, %106 : f32
        %shuffleResult, %valid = gpu.shuffle xor %111, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %112 = arith.select %16, %shuffleResult, %104 : f32
        %113 = arith.select %16, %106, %shuffleResult : f32
        %114 = arith.select %16, %108, %110 : f32
        %shuffleResult_11, %valid_12 = gpu.shuffle xor %114, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %115 = arith.select %16, %shuffleResult_11, %108 : f32
        %116 = arith.select %16, %110, %shuffleResult_11 : f32
        %117 = arith.select %18, %112, %115 : f32
        %shuffleResult_13, %valid_14 = gpu.shuffle xor %117, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %118 = arith.select %18, %shuffleResult_13, %112 : f32
        %119 = arith.select %18, %115, %shuffleResult_13 : f32
        %120 = arith.select %18, %113, %116 : f32
        %shuffleResult_15, %valid_16 = gpu.shuffle xor %120, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %121 = arith.select %18, %shuffleResult_15, %113 : f32
        %122 = arith.select %18, %116, %shuffleResult_15 : f32
        %123 = vector.from_elements %118, %121, %119, %122 : vector<1x4xf32>
        frisk.fragment_yield %123 : vector<1x4xf32>
      }
      %26 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %27 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %28 = vector.extract %25[0, 0] : f32 from vector<1x4xf32>
      %29 = vector.insert %28, %arg5 [%26, %27] : f32 into vector<2x8xf32>
      %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %31 = vector.extract %25[0, 1] : f32 from vector<1x4xf32>
      %32 = vector.insert %31, %29 [%26, %30] : f32 into vector<2x8xf32>
      %33 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %34 = vector.extract %25[0, 2] : f32 from vector<1x4xf32>
      %35 = vector.insert %34, %32 [%26, %33] : f32 into vector<2x8xf32>
      %36 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %37 = vector.extract %25[0, 3] : f32 from vector<1x4xf32>
      %38 = vector.insert %37, %35 [%26, %36] : f32 into vector<2x8xf32>
      %39 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %102 = vector.extract %13#4[%26, %27] : f32 from vector<2x8xf32>
        %103 = vector.extract %13#4[%26, %30] : f32 from vector<2x8xf32>
        %104 = vector.extract %13#4[%26, %33] : f32 from vector<2x8xf32>
        %105 = vector.extract %13#4[%26, %36] : f32 from vector<2x8xf32>
        %106 = arith.select %16, %102, %103 : f32
        %shuffleResult, %valid = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %16, %shuffleResult, %102 : f32
        %108 = arith.select %16, %103, %shuffleResult : f32
        %109 = arith.select %16, %104, %105 : f32
        %shuffleResult_11, %valid_12 = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %16, %shuffleResult_11, %104 : f32
        %111 = arith.select %16, %105, %shuffleResult_11 : f32
        %112 = arith.select %18, %107, %110 : f32
        %shuffleResult_13, %valid_14 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %18, %shuffleResult_13, %107 : f32
        %114 = arith.select %18, %110, %shuffleResult_13 : f32
        %115 = arith.select %18, %108, %111 : f32
        %shuffleResult_15, %valid_16 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %116 = arith.select %18, %shuffleResult_15, %108 : f32
        %117 = arith.select %18, %111, %shuffleResult_15 : f32
        %118 = vector.from_elements %113, %116, %114, %117 : vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %40 = vector.extract %39[0, 0] : f32 from vector<1x4xf32>
      %41 = vector.insert %40, %arg6 [%26, %27] : f32 into vector<2x8xf32>
      %42 = vector.extract %39[0, 1] : f32 from vector<1x4xf32>
      %43 = vector.insert %42, %41 [%26, %30] : f32 into vector<2x8xf32>
      %44 = vector.extract %39[0, 2] : f32 from vector<1x4xf32>
      %45 = vector.insert %44, %43 [%26, %33] : f32 into vector<2x8xf32>
      %46 = vector.extract %39[0, 3] : f32 from vector<1x4xf32>
      %47 = vector.insert %46, %45 [%26, %36] : f32 into vector<2x8xf32>
      %48 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %102 = vector.extract %13#3[%26, %27] : f32 from vector<2x8xf32>
        %103 = vector.extract %13#3[%26, %30] : f32 from vector<2x8xf32>
        %104 = vector.extract %13#3[%26, %33] : f32 from vector<2x8xf32>
        %105 = vector.extract %13#3[%26, %36] : f32 from vector<2x8xf32>
        %106 = arith.select %16, %102, %103 : f32
        %shuffleResult, %valid = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %16, %shuffleResult, %102 : f32
        %108 = arith.select %16, %103, %shuffleResult : f32
        %109 = arith.select %16, %104, %105 : f32
        %shuffleResult_11, %valid_12 = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %16, %shuffleResult_11, %104 : f32
        %111 = arith.select %16, %105, %shuffleResult_11 : f32
        %112 = arith.select %18, %107, %110 : f32
        %shuffleResult_13, %valid_14 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %18, %shuffleResult_13, %107 : f32
        %114 = arith.select %18, %110, %shuffleResult_13 : f32
        %115 = arith.select %18, %108, %111 : f32
        %shuffleResult_15, %valid_16 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %116 = arith.select %18, %shuffleResult_15, %108 : f32
        %117 = arith.select %18, %111, %shuffleResult_15 : f32
        %118 = vector.from_elements %113, %116, %114, %117 : vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %49 = vector.extract %48[0, 0] : f32 from vector<1x4xf32>
      %50 = vector.insert %49, %arg7 [%26, %27] : f32 into vector<2x8xf32>
      %51 = vector.extract %48[0, 1] : f32 from vector<1x4xf32>
      %52 = vector.insert %51, %50 [%26, %30] : f32 into vector<2x8xf32>
      %53 = vector.extract %48[0, 2] : f32 from vector<1x4xf32>
      %54 = vector.insert %53, %52 [%26, %33] : f32 into vector<2x8xf32>
      %55 = vector.extract %48[0, 3] : f32 from vector<1x4xf32>
      %56 = vector.insert %55, %54 [%26, %36] : f32 into vector<2x8xf32>
      %57 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %102 = vector.extract %13#0[%26, %27] : f32 from vector<2x8xf32>
        %103 = vector.extract %13#0[%26, %30] : f32 from vector<2x8xf32>
        %104 = vector.extract %13#0[%26, %33] : f32 from vector<2x8xf32>
        %105 = vector.extract %13#0[%26, %36] : f32 from vector<2x8xf32>
        %106 = arith.select %16, %102, %103 : f32
        %shuffleResult, %valid = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %16, %shuffleResult, %102 : f32
        %108 = arith.select %16, %103, %shuffleResult : f32
        %109 = arith.select %16, %104, %105 : f32
        %shuffleResult_11, %valid_12 = gpu.shuffle xor %109, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %16, %shuffleResult_11, %104 : f32
        %111 = arith.select %16, %105, %shuffleResult_11 : f32
        %112 = arith.select %18, %107, %110 : f32
        %shuffleResult_13, %valid_14 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %18, %shuffleResult_13, %107 : f32
        %114 = arith.select %18, %110, %shuffleResult_13 : f32
        %115 = arith.select %18, %108, %111 : f32
        %shuffleResult_15, %valid_16 = gpu.shuffle xor %115, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %116 = arith.select %18, %shuffleResult_15, %108 : f32
        %117 = arith.select %18, %111, %shuffleResult_15 : f32
        %118 = vector.from_elements %113, %116, %114, %117 : vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %58 = vector.extract %57[0, 0] : f32 from vector<1x4xf32>
      %59 = vector.insert %58, %arg8 [%26, %27] : f32 into vector<2x8xf32>
      %60 = vector.extract %57[0, 1] : f32 from vector<1x4xf32>
      %61 = vector.insert %60, %59 [%26, %30] : f32 into vector<2x8xf32>
      %62 = vector.extract %57[0, 2] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %61 [%26, %33] : f32 into vector<2x8xf32>
      %64 = vector.extract %57[0, 3] : f32 from vector<1x4xf32>
      %65 = vector.insert %64, %63 [%26, %36] : f32 into vector<2x8xf32>
      %66 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %102 = vector.extract %13#1[%26, 0] : f32 from vector<2x1xf32>
        %103 = vector.from_elements %102, %102, %102, %102 : vector<1x4xf32>
        frisk.fragment_yield %103 : vector<1x4xf32>
      }
      %67 = vector.extract %66[0, 0] : f32 from vector<1x4xf32>
      %68 = vector.insert %67, %arg9 [%26, %27] : f32 into vector<2x8xf32>
      %69 = vector.extract %66[0, 1] : f32 from vector<1x4xf32>
      %70 = vector.insert %69, %68 [%26, %30] : f32 into vector<2x8xf32>
      %71 = vector.extract %66[0, 2] : f32 from vector<1x4xf32>
      %72 = vector.insert %71, %70 [%26, %33] : f32 into vector<2x8xf32>
      %73 = vector.extract %66[0, 3] : f32 from vector<1x4xf32>
      %74 = vector.insert %73, %72 [%26, %36] : f32 into vector<2x8xf32>
      %75 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %102 = vector.extract %65[%26, %27] : f32 from vector<2x8xf32>
        %103 = vector.extract %65[%26, %30] : f32 from vector<2x8xf32>
        %104 = vector.extract %65[%26, %33] : f32 from vector<2x8xf32>
        %105 = vector.extract %65[%26, %36] : f32 from vector<2x8xf32>
        %106 = vector.from_elements %102, %103, %104, %105 : vector<1x4xf32>
        %107 = vector.extract %74[%26, %27] : f32 from vector<2x8xf32>
        %108 = vector.extract %74[%26, %30] : f32 from vector<2x8xf32>
        %109 = vector.extract %74[%26, %33] : f32 from vector<2x8xf32>
        %110 = vector.extract %74[%26, %36] : f32 from vector<2x8xf32>
        %111 = vector.from_elements %107, %108, %109, %110 : vector<1x4xf32>
        %112 = arith.divf %106, %111 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %112 : vector<1x4xf32>
      }
      %76 = vector.extract %75[0, 0] : f32 from vector<1x4xf32>
      %77 = vector.insert %76, %arg10 [%26, %27] : f32 into vector<2x8xf32>
      %78 = vector.extract %75[0, 1] : f32 from vector<1x4xf32>
      %79 = vector.insert %78, %77 [%26, %30] : f32 into vector<2x8xf32>
      %80 = vector.extract %75[0, 2] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %79 [%26, %33] : f32 into vector<2x8xf32>
      %82 = vector.extract %75[0, 3] : f32 from vector<1x4xf32>
      %83 = vector.insert %82, %81 [%26, %36] : f32 into vector<2x8xf32>
      %84 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %102 = vector.extract %83[%26, %27] : f32 from vector<2x8xf32>
        %103 = vector.extract %83[%26, %30] : f32 from vector<2x8xf32>
        %104 = vector.extract %83[%26, %33] : f32 from vector<2x8xf32>
        %105 = vector.extract %83[%26, %36] : f32 from vector<2x8xf32>
        %106 = vector.from_elements %102, %103, %104, %105 : vector<1x4xf32>
        %107 = arith.truncf %106 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %107 : vector<1x4xf16>
      }
      %85 = vector.extract %84[0, 0] : f16 from vector<1x4xf16>
      %86 = vector.insert %85, %arg11 [%26, %27] : f16 into vector<2x8xf16>
      %87 = vector.extract %84[0, 1] : f16 from vector<1x4xf16>
      %88 = vector.insert %87, %86 [%26, %30] : f16 into vector<2x8xf16>
      %89 = vector.extract %84[0, 2] : f16 from vector<1x4xf16>
      %90 = vector.insert %89, %88 [%26, %33] : f16 into vector<2x8xf16>
      %91 = vector.extract %84[0, 3] : f16 from vector<1x4xf16>
      %92 = vector.insert %91, %90 [%26, %36] : f16 into vector<2x8xf16>
      %93 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %102 = vector.extract %92[%26, %27] : f16 from vector<2x8xf16>
        %103 = vector.extract %92[%26, %30] : f16 from vector<2x8xf16>
        %104 = vector.extract %92[%26, %33] : f16 from vector<2x8xf16>
        %105 = vector.extract %92[%26, %36] : f16 from vector<2x8xf16>
        %106 = vector.from_elements %102, %103, %104, %105 : vector<1x4xf16>
        frisk.fragment_yield %106 : vector<1x4xf16>
      }
      %94 = vector.extract %93[0, 0] : f16 from vector<1x4xf16>
      %95 = vector.insert %94, %arg12 [%26, %27] : f16 into vector<2x8xf16>
      %96 = vector.extract %93[0, 1] : f16 from vector<1x4xf16>
      %97 = vector.insert %96, %95 [%26, %30] : f16 into vector<2x8xf16>
      %98 = vector.extract %93[0, 2] : f16 from vector<1x4xf16>
      %99 = vector.insert %98, %97 [%26, %33] : f16 into vector<2x8xf16>
      %100 = vector.extract %93[0, 3] : f16 from vector<1x4xf16>
      %101 = vector.insert %100, %99 [%26, %36] : f16 into vector<2x8xf16>
      affine.yield %38, %47, %56, %65, %74, %83, %92, %101 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    %20 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %21 = arith.cmpi ult, %20, %c4 : index
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
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
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %22:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %26 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %66 = vector.extract %13#1[%25, 0] : f32 from vector<2x1xf32>
        %67 = vector.from_elements %66, %66, %66, %66 : vector<1x4xf32>
        frisk.fragment_yield %67 : vector<1x4xf32>
      }
      %27 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %28 = vector.extract %26[0, 0] : f32 from vector<1x4xf32>
      %29 = vector.insert %28, %arg5 [%25, %27] : f32 into vector<2x8xf32>
      %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %31 = vector.extract %26[0, 1] : f32 from vector<1x4xf32>
      %32 = vector.insert %31, %29 [%25, %30] : f32 into vector<2x8xf32>
      %33 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %34 = vector.extract %26[0, 2] : f32 from vector<1x4xf32>
      %35 = vector.insert %34, %32 [%25, %33] : f32 into vector<2x8xf32>
      %36 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %37 = vector.extract %26[0, 3] : f32 from vector<1x4xf32>
      %38 = vector.insert %37, %35 [%25, %36] : f32 into vector<2x8xf32>
      %39 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %66 = vector.extract %19#2[%25, %27] : f32 from vector<2x8xf32>
        %67 = vector.extract %19#2[%25, %30] : f32 from vector<2x8xf32>
        %68 = vector.extract %19#2[%25, %33] : f32 from vector<2x8xf32>
        %69 = vector.extract %19#2[%25, %36] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf32>
        %71 = vector.extract %38[%25, %27] : f32 from vector<2x8xf32>
        %72 = vector.extract %38[%25, %30] : f32 from vector<2x8xf32>
        %73 = vector.extract %38[%25, %33] : f32 from vector<2x8xf32>
        %74 = vector.extract %38[%25, %36] : f32 from vector<2x8xf32>
        %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
        %76 = arith.divf %70, %75 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %76 : vector<1x4xf32>
      }
      %40 = vector.extract %39[0, 0] : f32 from vector<1x4xf32>
      %41 = vector.insert %40, %arg6 [%25, %27] : f32 into vector<2x8xf32>
      %42 = vector.extract %39[0, 1] : f32 from vector<1x4xf32>
      %43 = vector.insert %42, %41 [%25, %30] : f32 into vector<2x8xf32>
      %44 = vector.extract %39[0, 2] : f32 from vector<1x4xf32>
      %45 = vector.insert %44, %43 [%25, %33] : f32 into vector<2x8xf32>
      %46 = vector.extract %39[0, 3] : f32 from vector<1x4xf32>
      %47 = vector.insert %46, %45 [%25, %36] : f32 into vector<2x8xf32>
      %48 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %66 = vector.extract %47[%25, %27] : f32 from vector<2x8xf32>
        %67 = vector.extract %47[%25, %30] : f32 from vector<2x8xf32>
        %68 = vector.extract %47[%25, %33] : f32 from vector<2x8xf32>
        %69 = vector.extract %47[%25, %36] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf32>
        %71 = arith.truncf %70 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %71 : vector<1x4xf16>
      }
      %49 = vector.extract %48[0, 0] : f16 from vector<1x4xf16>
      %50 = vector.insert %49, %arg7 [%25, %27] : f16 into vector<2x8xf16>
      %51 = vector.extract %48[0, 1] : f16 from vector<1x4xf16>
      %52 = vector.insert %51, %50 [%25, %30] : f16 into vector<2x8xf16>
      %53 = vector.extract %48[0, 2] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %52 [%25, %33] : f16 into vector<2x8xf16>
      %55 = vector.extract %48[0, 3] : f16 from vector<1x4xf16>
      %56 = vector.insert %55, %54 [%25, %36] : f16 into vector<2x8xf16>
      %57 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %66 = vector.extract %56[%25, %27] : f16 from vector<2x8xf16>
        %67 = vector.extract %56[%25, %30] : f16 from vector<2x8xf16>
        %68 = vector.extract %56[%25, %33] : f16 from vector<2x8xf16>
        %69 = vector.extract %56[%25, %36] : f16 from vector<2x8xf16>
        %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf16>
        frisk.fragment_yield %70 : vector<1x4xf16>
      }
      %58 = vector.extract %57[0, 0] : f16 from vector<1x4xf16>
      %59 = vector.insert %58, %arg8 [%25, %27] : f16 into vector<2x8xf16>
      %60 = vector.extract %57[0, 1] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %59 [%25, %30] : f16 into vector<2x8xf16>
      %62 = vector.extract %57[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%25, %33] : f16 into vector<2x8xf16>
      %64 = vector.extract %57[0, 3] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%25, %36] : f16 into vector<2x8xf16>
      affine.yield %38, %47, %56, %65 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
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
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %23:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %26 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %66 = vector.extract %13#1[%25, 0] : f32 from vector<2x1xf32>
        %67 = vector.from_elements %66, %66, %66, %66 : vector<1x4xf32>
        frisk.fragment_yield %67 : vector<1x4xf32>
      }
      %27 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %28 = vector.extract %26[0, 0] : f32 from vector<1x4xf32>
      %29 = vector.insert %28, %arg5 [%25, %27] : f32 into vector<2x8xf32>
      %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %31 = vector.extract %26[0, 1] : f32 from vector<1x4xf32>
      %32 = vector.insert %31, %29 [%25, %30] : f32 into vector<2x8xf32>
      %33 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %34 = vector.extract %26[0, 2] : f32 from vector<1x4xf32>
      %35 = vector.insert %34, %32 [%25, %33] : f32 into vector<2x8xf32>
      %36 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %37 = vector.extract %26[0, 3] : f32 from vector<1x4xf32>
      %38 = vector.insert %37, %35 [%25, %36] : f32 into vector<2x8xf32>
      %39 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %66 = vector.extract %19#1[%25, %27] : f32 from vector<2x8xf32>
        %67 = vector.extract %19#1[%25, %30] : f32 from vector<2x8xf32>
        %68 = vector.extract %19#1[%25, %33] : f32 from vector<2x8xf32>
        %69 = vector.extract %19#1[%25, %36] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf32>
        %71 = vector.extract %38[%25, %27] : f32 from vector<2x8xf32>
        %72 = vector.extract %38[%25, %30] : f32 from vector<2x8xf32>
        %73 = vector.extract %38[%25, %33] : f32 from vector<2x8xf32>
        %74 = vector.extract %38[%25, %36] : f32 from vector<2x8xf32>
        %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
        %76 = arith.divf %70, %75 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %76 : vector<1x4xf32>
      }
      %40 = vector.extract %39[0, 0] : f32 from vector<1x4xf32>
      %41 = vector.insert %40, %arg6 [%25, %27] : f32 into vector<2x8xf32>
      %42 = vector.extract %39[0, 1] : f32 from vector<1x4xf32>
      %43 = vector.insert %42, %41 [%25, %30] : f32 into vector<2x8xf32>
      %44 = vector.extract %39[0, 2] : f32 from vector<1x4xf32>
      %45 = vector.insert %44, %43 [%25, %33] : f32 into vector<2x8xf32>
      %46 = vector.extract %39[0, 3] : f32 from vector<1x4xf32>
      %47 = vector.insert %46, %45 [%25, %36] : f32 into vector<2x8xf32>
      %48 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %66 = vector.extract %47[%25, %27] : f32 from vector<2x8xf32>
        %67 = vector.extract %47[%25, %30] : f32 from vector<2x8xf32>
        %68 = vector.extract %47[%25, %33] : f32 from vector<2x8xf32>
        %69 = vector.extract %47[%25, %36] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf32>
        %71 = arith.truncf %70 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %71 : vector<1x4xf16>
      }
      %49 = vector.extract %48[0, 0] : f16 from vector<1x4xf16>
      %50 = vector.insert %49, %arg7 [%25, %27] : f16 into vector<2x8xf16>
      %51 = vector.extract %48[0, 1] : f16 from vector<1x4xf16>
      %52 = vector.insert %51, %50 [%25, %30] : f16 into vector<2x8xf16>
      %53 = vector.extract %48[0, 2] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %52 [%25, %33] : f16 into vector<2x8xf16>
      %55 = vector.extract %48[0, 3] : f16 from vector<1x4xf16>
      %56 = vector.insert %55, %54 [%25, %36] : f16 into vector<2x8xf16>
      %57 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %66 = vector.extract %56[%25, %27] : f16 from vector<2x8xf16>
        %67 = vector.extract %56[%25, %30] : f16 from vector<2x8xf16>
        %68 = vector.extract %56[%25, %33] : f16 from vector<2x8xf16>
        %69 = vector.extract %56[%25, %36] : f16 from vector<2x8xf16>
        %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf16>
        frisk.fragment_yield %70 : vector<1x4xf16>
      }
      %58 = vector.extract %57[0, 0] : f16 from vector<1x4xf16>
      %59 = vector.insert %58, %arg8 [%25, %27] : f16 into vector<2x8xf16>
      %60 = vector.extract %57[0, 1] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %59 [%25, %30] : f16 into vector<2x8xf16>
      %62 = vector.extract %57[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%25, %33] : f16 into vector<2x8xf16>
      %64 = vector.extract %57[0, 3] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%25, %36] : f16 into vector<2x8xf16>
      affine.yield %38, %47, %56, %65 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
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
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %24:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %25 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %26 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %66 = vector.extract %13#1[%25, 0] : f32 from vector<2x1xf32>
        %67 = vector.from_elements %66, %66, %66, %66 : vector<1x4xf32>
        frisk.fragment_yield %67 : vector<1x4xf32>
      }
      %27 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %28 = vector.extract %26[0, 0] : f32 from vector<1x4xf32>
      %29 = vector.insert %28, %arg5 [%25, %27] : f32 into vector<2x8xf32>
      %30 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %31 = vector.extract %26[0, 1] : f32 from vector<1x4xf32>
      %32 = vector.insert %31, %29 [%25, %30] : f32 into vector<2x8xf32>
      %33 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %34 = vector.extract %26[0, 2] : f32 from vector<1x4xf32>
      %35 = vector.insert %34, %32 [%25, %33] : f32 into vector<2x8xf32>
      %36 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %37 = vector.extract %26[0, 3] : f32 from vector<1x4xf32>
      %38 = vector.insert %37, %35 [%25, %36] : f32 into vector<2x8xf32>
      %39 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %66 = vector.extract %19#0[%25, %27] : f32 from vector<2x8xf32>
        %67 = vector.extract %19#0[%25, %30] : f32 from vector<2x8xf32>
        %68 = vector.extract %19#0[%25, %33] : f32 from vector<2x8xf32>
        %69 = vector.extract %19#0[%25, %36] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf32>
        %71 = vector.extract %38[%25, %27] : f32 from vector<2x8xf32>
        %72 = vector.extract %38[%25, %30] : f32 from vector<2x8xf32>
        %73 = vector.extract %38[%25, %33] : f32 from vector<2x8xf32>
        %74 = vector.extract %38[%25, %36] : f32 from vector<2x8xf32>
        %75 = vector.from_elements %71, %72, %73, %74 : vector<1x4xf32>
        %76 = arith.divf %70, %75 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %76 : vector<1x4xf32>
      }
      %40 = vector.extract %39[0, 0] : f32 from vector<1x4xf32>
      %41 = vector.insert %40, %arg6 [%25, %27] : f32 into vector<2x8xf32>
      %42 = vector.extract %39[0, 1] : f32 from vector<1x4xf32>
      %43 = vector.insert %42, %41 [%25, %30] : f32 into vector<2x8xf32>
      %44 = vector.extract %39[0, 2] : f32 from vector<1x4xf32>
      %45 = vector.insert %44, %43 [%25, %33] : f32 into vector<2x8xf32>
      %46 = vector.extract %39[0, 3] : f32 from vector<1x4xf32>
      %47 = vector.insert %46, %45 [%25, %36] : f32 into vector<2x8xf32>
      %48 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %66 = vector.extract %47[%25, %27] : f32 from vector<2x8xf32>
        %67 = vector.extract %47[%25, %30] : f32 from vector<2x8xf32>
        %68 = vector.extract %47[%25, %33] : f32 from vector<2x8xf32>
        %69 = vector.extract %47[%25, %36] : f32 from vector<2x8xf32>
        %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf32>
        %71 = arith.truncf %70 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %71 : vector<1x4xf16>
      }
      %49 = vector.extract %48[0, 0] : f16 from vector<1x4xf16>
      %50 = vector.insert %49, %arg7 [%25, %27] : f16 into vector<2x8xf16>
      %51 = vector.extract %48[0, 1] : f16 from vector<1x4xf16>
      %52 = vector.insert %51, %50 [%25, %30] : f16 into vector<2x8xf16>
      %53 = vector.extract %48[0, 2] : f16 from vector<1x4xf16>
      %54 = vector.insert %53, %52 [%25, %33] : f16 into vector<2x8xf16>
      %55 = vector.extract %48[0, 3] : f16 from vector<1x4xf16>
      %56 = vector.insert %55, %54 [%25, %36] : f16 into vector<2x8xf16>
      %57 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %66 = vector.extract %56[%25, %27] : f16 from vector<2x8xf16>
        %67 = vector.extract %56[%25, %30] : f16 from vector<2x8xf16>
        %68 = vector.extract %56[%25, %33] : f16 from vector<2x8xf16>
        %69 = vector.extract %56[%25, %36] : f16 from vector<2x8xf16>
        %70 = vector.from_elements %66, %67, %68, %69 : vector<1x4xf16>
        frisk.fragment_yield %70 : vector<1x4xf16>
      }
      %58 = vector.extract %57[0, 0] : f16 from vector<1x4xf16>
      %59 = vector.insert %58, %arg8 [%25, %27] : f16 into vector<2x8xf16>
      %60 = vector.extract %57[0, 1] : f16 from vector<1x4xf16>
      %61 = vector.insert %60, %59 [%25, %30] : f16 into vector<2x8xf16>
      %62 = vector.extract %57[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %61 [%25, %33] : f16 into vector<2x8xf16>
      %64 = vector.extract %57[0, 3] : f16 from vector<1x4xf16>
      %65 = vector.insert %64, %63 [%25, %36] : f16 into vector<2x8xf16>
      affine.yield %38, %47, %56, %65 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %21 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
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
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
