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
    %c96 = arith.constant 96 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0xFF800000> : vector<2x1xf32>
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
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_10 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_12 = arith.constant 0xFF800000 : f32
    %cst_13 = arith.constant 0.127531052 : f32
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
    %8 = arith.truncf %cst_13 : f32 to f16
    %9:3 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_11, %arg6 = %cst_11, %arg7 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %162 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %163 = affine.apply affine_map<() -> (0)>()
      %164 = vector.extract %cst_10[0, 0] : f32 from vector<1x1xf32>
      %165 = vector.insert %164, %arg5 [%162, %163] : f32 into vector<2x1xf32>
      %166 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %167 = affine.apply affine_map<() -> (0)>()
      %cst_22 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %168 = vector.extract %cst[%166, %167] : f32 from vector<2x1xf32>
      %169 = vector.insert %168, %cst_22 [0, 0] : f32 into vector<1x1xf32>
      %170 = arith.addf %cst_10, %169 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
      %171 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %172 = affine.apply affine_map<() -> (0)>()
      %173 = vector.extract %170[0, 0] : f32 from vector<1x1xf32>
      %174 = vector.insert %173, %arg6 [%171, %172] : f32 into vector<2x1xf32>
      %175 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %176 = affine.apply affine_map<() -> (0)>()
      %177 = vector.extract %170[0, 0] : f32 from vector<1x1xf32>
      %178 = vector.insert %177, %arg7 [%175, %176] : f32 into vector<2x1xf32>
      affine.yield %165, %174, %178 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = frisk.to_threadTile %cst_9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %11 = frisk.to_threadTile %cst_8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %12 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c0) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c0 floordiv 128) mod 4096, %c0 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %12 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
    %13 = vector.broadcast %8 : f16 to vector<2x8xf16>
    %14 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %15 = affine.apply affine_map<() -> (0)>()
    %16 = affine.apply affine_map<() -> (0)>()
    %cst_14 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %17 = vector.extract %14[%15, %16] : f16 from vector<2x8xf16>
    %18 = vector.insert %17, %cst_14 [0, 0] : f16 into vector<1x4xf16>
    %19 = affine.apply affine_map<() -> (1)>()
    %20 = vector.extract %14[%15, %19] : f16 from vector<2x8xf16>
    %21 = vector.insert %20, %18 [0, 1] : f16 into vector<1x4xf16>
    %22 = affine.apply affine_map<() -> (2)>()
    %23 = vector.extract %14[%15, %22] : f16 from vector<2x8xf16>
    %24 = vector.insert %23, %21 [0, 2] : f16 into vector<1x4xf16>
    %25 = affine.apply affine_map<() -> (3)>()
    %26 = vector.extract %14[%15, %25] : f16 from vector<2x8xf16>
    %27 = vector.insert %26, %24 [0, 3] : f16 into vector<1x4xf16>
    %28:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_6, %arg6 = %cst_6, %arg7 = %27) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %162 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %163 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %164 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %165 = vector.extract %162[%163, %164] : f16 from vector<2x8xf16>
      %166 = vector.insert %165, %cst_22 [0, 0] : f16 into vector<1x4xf16>
      %167 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %168 = vector.extract %162[%163, %167] : f16 from vector<2x8xf16>
      %169 = vector.insert %168, %166 [0, 1] : f16 into vector<1x4xf16>
      %170 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %171 = vector.extract %162[%163, %170] : f16 from vector<2x8xf16>
      %172 = vector.insert %171, %169 [0, 2] : f16 into vector<1x4xf16>
      %173 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %174 = vector.extract %162[%163, %173] : f16 from vector<2x8xf16>
      %175 = vector.insert %174, %172 [0, 3] : f16 into vector<1x4xf16>
      %176 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %178 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %179 = vector.insert %178, %arg5 [%176, %177] : f16 into vector<2x8xf16>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %181 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %182 = vector.insert %181, %179 [%176, %180] : f16 into vector<2x8xf16>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %184 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %185 = vector.insert %184, %182 [%176, %183] : f16 into vector<2x8xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %187 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %188 = vector.insert %187, %185 [%176, %186] : f16 into vector<2x8xf16>
      %189 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %191 = vector.extract %13[%189, %190] : f16 from vector<2x8xf16>
      %192 = vector.insert %191, %cst_23 [0, 0] : f16 into vector<1x4xf16>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %194 = vector.extract %13[%189, %193] : f16 from vector<2x8xf16>
      %195 = vector.insert %194, %192 [0, 1] : f16 into vector<1x4xf16>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %197 = vector.extract %13[%189, %196] : f16 from vector<2x8xf16>
      %198 = vector.insert %197, %195 [0, 2] : f16 into vector<1x4xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %200 = vector.extract %13[%189, %199] : f16 from vector<2x8xf16>
      %201 = vector.insert %200, %198 [0, 3] : f16 into vector<1x4xf16>
      %202 = arith.mulf %arg7, %201 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %203 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %205 = vector.extract %202[0, 0] : f16 from vector<1x4xf16>
      %206 = vector.insert %205, %arg6 [%203, %204] : f16 into vector<2x8xf16>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %208 = vector.extract %202[0, 1] : f16 from vector<1x4xf16>
      %209 = vector.insert %208, %206 [%203, %207] : f16 into vector<2x8xf16>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %211 = vector.extract %202[0, 2] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %209 [%203, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %214 = vector.extract %202[0, 3] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%203, %213] : f16 into vector<2x8xf16>
      affine.yield %188, %215, %175 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %29 = affine.apply affine_map<() -> (1)>()
    %30 = affine.apply affine_map<() -> (4)>()
    %cst_15 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %31 = vector.extract %13[%29, %30] : f16 from vector<2x8xf16>
    %32 = vector.insert %31, %cst_15 [0, 0] : f16 into vector<1x4xf16>
    %33 = affine.apply affine_map<() -> (5)>()
    %34 = vector.extract %13[%29, %33] : f16 from vector<2x8xf16>
    %35 = vector.insert %34, %32 [0, 1] : f16 into vector<1x4xf16>
    %36 = affine.apply affine_map<() -> (6)>()
    %37 = vector.extract %13[%29, %36] : f16 from vector<2x8xf16>
    %38 = vector.insert %37, %35 [0, 2] : f16 into vector<1x4xf16>
    %39 = affine.apply affine_map<() -> (7)>()
    %40 = vector.extract %13[%29, %39] : f16 from vector<2x8xf16>
    %41 = vector.insert %40, %38 [0, 3] : f16 into vector<1x4xf16>
    %42 = arith.mulf %28#2, %41 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %43 = affine.apply affine_map<() -> (1)>()
    %44 = affine.apply affine_map<() -> (4)>()
    %45 = vector.extract %42[0, 0] : f16 from vector<1x4xf16>
    %46 = vector.insert %45, %28#1 [%43, %44] : f16 into vector<2x8xf16>
    %47 = affine.apply affine_map<() -> (5)>()
    %48 = vector.extract %42[0, 1] : f16 from vector<1x4xf16>
    %49 = vector.insert %48, %46 [%43, %47] : f16 into vector<2x8xf16>
    %50 = affine.apply affine_map<() -> (6)>()
    %51 = vector.extract %42[0, 2] : f16 from vector<1x4xf16>
    %52 = vector.insert %51, %49 [%43, %50] : f16 into vector<2x8xf16>
    %53 = affine.apply affine_map<() -> (7)>()
    %54 = vector.extract %42[0, 3] : f16 from vector<1x4xf16>
    %55 = vector.insert %54, %52 [%43, %53] : f16 into vector<2x8xf16>
    %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %56 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c32) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c32 floordiv 128) mod 4096, %c32 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %56 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_16 : memref<128x32xf16, 3>
    %57 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %58 = affine.apply affine_map<() -> (0)>()
    %59 = affine.apply affine_map<() -> (0)>()
    %cst_17 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %60 = vector.extract %57[%58, %59] : f16 from vector<2x8xf16>
    %61 = vector.insert %60, %cst_17 [0, 0] : f16 into vector<1x4xf16>
    %62 = affine.apply affine_map<() -> (1)>()
    %63 = vector.extract %57[%58, %62] : f16 from vector<2x8xf16>
    %64 = vector.insert %63, %61 [0, 1] : f16 into vector<1x4xf16>
    %65 = affine.apply affine_map<() -> (2)>()
    %66 = vector.extract %57[%58, %65] : f16 from vector<2x8xf16>
    %67 = vector.insert %66, %64 [0, 2] : f16 into vector<1x4xf16>
    %68 = affine.apply affine_map<() -> (3)>()
    %69 = vector.extract %57[%58, %68] : f16 from vector<2x8xf16>
    %70 = vector.insert %69, %67 [0, 3] : f16 into vector<1x4xf16>
    %71:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_6, %arg6 = %cst_6, %arg7 = %70) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %162 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %163 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %164 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %165 = vector.extract %162[%163, %164] : f16 from vector<2x8xf16>
      %166 = vector.insert %165, %cst_22 [0, 0] : f16 into vector<1x4xf16>
      %167 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %168 = vector.extract %162[%163, %167] : f16 from vector<2x8xf16>
      %169 = vector.insert %168, %166 [0, 1] : f16 into vector<1x4xf16>
      %170 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %171 = vector.extract %162[%163, %170] : f16 from vector<2x8xf16>
      %172 = vector.insert %171, %169 [0, 2] : f16 into vector<1x4xf16>
      %173 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %174 = vector.extract %162[%163, %173] : f16 from vector<2x8xf16>
      %175 = vector.insert %174, %172 [0, 3] : f16 into vector<1x4xf16>
      %176 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %178 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %179 = vector.insert %178, %arg5 [%176, %177] : f16 into vector<2x8xf16>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %181 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %182 = vector.insert %181, %179 [%176, %180] : f16 into vector<2x8xf16>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %184 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %185 = vector.insert %184, %182 [%176, %183] : f16 into vector<2x8xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %187 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %188 = vector.insert %187, %185 [%176, %186] : f16 into vector<2x8xf16>
      %189 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %191 = vector.extract %13[%189, %190] : f16 from vector<2x8xf16>
      %192 = vector.insert %191, %cst_23 [0, 0] : f16 into vector<1x4xf16>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %194 = vector.extract %13[%189, %193] : f16 from vector<2x8xf16>
      %195 = vector.insert %194, %192 [0, 1] : f16 into vector<1x4xf16>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %197 = vector.extract %13[%189, %196] : f16 from vector<2x8xf16>
      %198 = vector.insert %197, %195 [0, 2] : f16 into vector<1x4xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %200 = vector.extract %13[%189, %199] : f16 from vector<2x8xf16>
      %201 = vector.insert %200, %198 [0, 3] : f16 into vector<1x4xf16>
      %202 = arith.mulf %arg7, %201 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %203 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %205 = vector.extract %202[0, 0] : f16 from vector<1x4xf16>
      %206 = vector.insert %205, %arg6 [%203, %204] : f16 into vector<2x8xf16>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %208 = vector.extract %202[0, 1] : f16 from vector<1x4xf16>
      %209 = vector.insert %208, %206 [%203, %207] : f16 into vector<2x8xf16>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %211 = vector.extract %202[0, 2] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %209 [%203, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %214 = vector.extract %202[0, 3] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%203, %213] : f16 into vector<2x8xf16>
      affine.yield %188, %215, %175 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %72 = arith.mulf %71#2, %41 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %73 = affine.apply affine_map<() -> (1)>()
    %74 = affine.apply affine_map<() -> (4)>()
    %75 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
    %76 = vector.insert %75, %71#1 [%73, %74] : f16 into vector<2x8xf16>
    %77 = affine.apply affine_map<() -> (5)>()
    %78 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
    %79 = vector.insert %78, %76 [%73, %77] : f16 into vector<2x8xf16>
    %80 = affine.apply affine_map<() -> (6)>()
    %81 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
    %82 = vector.insert %81, %79 [%73, %80] : f16 into vector<2x8xf16>
    %83 = affine.apply affine_map<() -> (7)>()
    %84 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
    %85 = vector.insert %84, %82 [%73, %83] : f16 into vector<2x8xf16>
    %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %86 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c64) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c64 floordiv 128) mod 4096, %c64 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %86 to %alloc_18 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_18 : memref<128x32xf16, 3>
    %87 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %88 = affine.apply affine_map<() -> (0)>()
    %89 = affine.apply affine_map<() -> (0)>()
    %cst_19 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %90 = vector.extract %87[%88, %89] : f16 from vector<2x8xf16>
    %91 = vector.insert %90, %cst_19 [0, 0] : f16 into vector<1x4xf16>
    %92 = affine.apply affine_map<() -> (1)>()
    %93 = vector.extract %87[%88, %92] : f16 from vector<2x8xf16>
    %94 = vector.insert %93, %91 [0, 1] : f16 into vector<1x4xf16>
    %95 = affine.apply affine_map<() -> (2)>()
    %96 = vector.extract %87[%88, %95] : f16 from vector<2x8xf16>
    %97 = vector.insert %96, %94 [0, 2] : f16 into vector<1x4xf16>
    %98 = affine.apply affine_map<() -> (3)>()
    %99 = vector.extract %87[%88, %98] : f16 from vector<2x8xf16>
    %100 = vector.insert %99, %97 [0, 3] : f16 into vector<1x4xf16>
    %101:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_6, %arg6 = %cst_6, %arg7 = %100) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %162 = frisk.to_threadTile %alloc_18 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %163 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %164 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %165 = vector.extract %162[%163, %164] : f16 from vector<2x8xf16>
      %166 = vector.insert %165, %cst_22 [0, 0] : f16 into vector<1x4xf16>
      %167 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %168 = vector.extract %162[%163, %167] : f16 from vector<2x8xf16>
      %169 = vector.insert %168, %166 [0, 1] : f16 into vector<1x4xf16>
      %170 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %171 = vector.extract %162[%163, %170] : f16 from vector<2x8xf16>
      %172 = vector.insert %171, %169 [0, 2] : f16 into vector<1x4xf16>
      %173 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %174 = vector.extract %162[%163, %173] : f16 from vector<2x8xf16>
      %175 = vector.insert %174, %172 [0, 3] : f16 into vector<1x4xf16>
      %176 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %178 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %179 = vector.insert %178, %arg5 [%176, %177] : f16 into vector<2x8xf16>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %181 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %182 = vector.insert %181, %179 [%176, %180] : f16 into vector<2x8xf16>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %184 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %185 = vector.insert %184, %182 [%176, %183] : f16 into vector<2x8xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %187 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %188 = vector.insert %187, %185 [%176, %186] : f16 into vector<2x8xf16>
      %189 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %191 = vector.extract %13[%189, %190] : f16 from vector<2x8xf16>
      %192 = vector.insert %191, %cst_23 [0, 0] : f16 into vector<1x4xf16>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %194 = vector.extract %13[%189, %193] : f16 from vector<2x8xf16>
      %195 = vector.insert %194, %192 [0, 1] : f16 into vector<1x4xf16>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %197 = vector.extract %13[%189, %196] : f16 from vector<2x8xf16>
      %198 = vector.insert %197, %195 [0, 2] : f16 into vector<1x4xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %200 = vector.extract %13[%189, %199] : f16 from vector<2x8xf16>
      %201 = vector.insert %200, %198 [0, 3] : f16 into vector<1x4xf16>
      %202 = arith.mulf %arg7, %201 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %203 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %205 = vector.extract %202[0, 0] : f16 from vector<1x4xf16>
      %206 = vector.insert %205, %arg6 [%203, %204] : f16 into vector<2x8xf16>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %208 = vector.extract %202[0, 1] : f16 from vector<1x4xf16>
      %209 = vector.insert %208, %206 [%203, %207] : f16 into vector<2x8xf16>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %211 = vector.extract %202[0, 2] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %209 [%203, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %214 = vector.extract %202[0, 3] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%203, %213] : f16 into vector<2x8xf16>
      affine.yield %188, %215, %175 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %102 = arith.mulf %101#2, %41 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %103 = affine.apply affine_map<() -> (1)>()
    %104 = affine.apply affine_map<() -> (4)>()
    %105 = vector.extract %102[0, 0] : f16 from vector<1x4xf16>
    %106 = vector.insert %105, %101#1 [%103, %104] : f16 into vector<2x8xf16>
    %107 = affine.apply affine_map<() -> (5)>()
    %108 = vector.extract %102[0, 1] : f16 from vector<1x4xf16>
    %109 = vector.insert %108, %106 [%103, %107] : f16 into vector<2x8xf16>
    %110 = affine.apply affine_map<() -> (6)>()
    %111 = vector.extract %102[0, 2] : f16 from vector<1x4xf16>
    %112 = vector.insert %111, %109 [%103, %110] : f16 into vector<2x8xf16>
    %113 = affine.apply affine_map<() -> (7)>()
    %114 = vector.extract %102[0, 3] : f16 from vector<1x4xf16>
    %115 = vector.insert %114, %112 [%103, %113] : f16 into vector<2x8xf16>
    %alloc_20 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %116 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c96) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c96 floordiv 128) mod 4096, %c96 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %116 to %alloc_20 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_20 : memref<128x32xf16, 3>
    %117 = frisk.to_threadTile %alloc_20 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %118 = affine.apply affine_map<() -> (0)>()
    %119 = affine.apply affine_map<() -> (0)>()
    %cst_21 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %120 = vector.extract %117[%118, %119] : f16 from vector<2x8xf16>
    %121 = vector.insert %120, %cst_21 [0, 0] : f16 into vector<1x4xf16>
    %122 = affine.apply affine_map<() -> (1)>()
    %123 = vector.extract %117[%118, %122] : f16 from vector<2x8xf16>
    %124 = vector.insert %123, %121 [0, 1] : f16 into vector<1x4xf16>
    %125 = affine.apply affine_map<() -> (2)>()
    %126 = vector.extract %117[%118, %125] : f16 from vector<2x8xf16>
    %127 = vector.insert %126, %124 [0, 2] : f16 into vector<1x4xf16>
    %128 = affine.apply affine_map<() -> (3)>()
    %129 = vector.extract %117[%118, %128] : f16 from vector<2x8xf16>
    %130 = vector.insert %129, %127 [0, 3] : f16 into vector<1x4xf16>
    %131:3 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_6, %arg6 = %cst_6, %arg7 = %130) -> (vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %162 = frisk.to_threadTile %alloc_20 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %163 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %164 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %165 = vector.extract %162[%163, %164] : f16 from vector<2x8xf16>
      %166 = vector.insert %165, %cst_22 [0, 0] : f16 into vector<1x4xf16>
      %167 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %168 = vector.extract %162[%163, %167] : f16 from vector<2x8xf16>
      %169 = vector.insert %168, %166 [0, 1] : f16 into vector<1x4xf16>
      %170 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %171 = vector.extract %162[%163, %170] : f16 from vector<2x8xf16>
      %172 = vector.insert %171, %169 [0, 2] : f16 into vector<1x4xf16>
      %173 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %174 = vector.extract %162[%163, %173] : f16 from vector<2x8xf16>
      %175 = vector.insert %174, %172 [0, 3] : f16 into vector<1x4xf16>
      %176 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %178 = vector.extract %arg7[0, 0] : f16 from vector<1x4xf16>
      %179 = vector.insert %178, %arg5 [%176, %177] : f16 into vector<2x8xf16>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %181 = vector.extract %arg7[0, 1] : f16 from vector<1x4xf16>
      %182 = vector.insert %181, %179 [%176, %180] : f16 into vector<2x8xf16>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %184 = vector.extract %arg7[0, 2] : f16 from vector<1x4xf16>
      %185 = vector.insert %184, %182 [%176, %183] : f16 into vector<2x8xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %187 = vector.extract %arg7[0, 3] : f16 from vector<1x4xf16>
      %188 = vector.insert %187, %185 [%176, %186] : f16 into vector<2x8xf16>
      %189 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %191 = vector.extract %13[%189, %190] : f16 from vector<2x8xf16>
      %192 = vector.insert %191, %cst_23 [0, 0] : f16 into vector<1x4xf16>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %194 = vector.extract %13[%189, %193] : f16 from vector<2x8xf16>
      %195 = vector.insert %194, %192 [0, 1] : f16 into vector<1x4xf16>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %197 = vector.extract %13[%189, %196] : f16 from vector<2x8xf16>
      %198 = vector.insert %197, %195 [0, 2] : f16 into vector<1x4xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %200 = vector.extract %13[%189, %199] : f16 from vector<2x8xf16>
      %201 = vector.insert %200, %198 [0, 3] : f16 into vector<1x4xf16>
      %202 = arith.mulf %arg7, %201 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %203 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %205 = vector.extract %202[0, 0] : f16 from vector<1x4xf16>
      %206 = vector.insert %205, %arg6 [%203, %204] : f16 into vector<2x8xf16>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %208 = vector.extract %202[0, 1] : f16 from vector<1x4xf16>
      %209 = vector.insert %208, %206 [%203, %207] : f16 into vector<2x8xf16>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %211 = vector.extract %202[0, 2] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %209 [%203, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %214 = vector.extract %202[0, 3] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%203, %213] : f16 into vector<2x8xf16>
      affine.yield %188, %215, %175 : vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %132 = arith.mulf %131#2, %41 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %133 = affine.apply affine_map<() -> (1)>()
    %134 = affine.apply affine_map<() -> (4)>()
    %135 = vector.extract %132[0, 0] : f16 from vector<1x4xf16>
    %136 = vector.insert %135, %131#1 [%133, %134] : f16 into vector<2x8xf16>
    %137 = affine.apply affine_map<() -> (5)>()
    %138 = vector.extract %132[0, 1] : f16 from vector<1x4xf16>
    %139 = vector.insert %138, %136 [%133, %137] : f16 into vector<2x8xf16>
    %140 = affine.apply affine_map<() -> (6)>()
    %141 = vector.extract %132[0, 2] : f16 from vector<1x4xf16>
    %142 = vector.insert %141, %139 [%133, %140] : f16 into vector<2x8xf16>
    %143 = affine.apply affine_map<() -> (7)>()
    %144 = vector.extract %132[0, 3] : f16 from vector<1x4xf16>
    %145 = vector.insert %144, %142 [%133, %143] : f16 into vector<2x8xf16>
    %146:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#2, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %162 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %alloc_22 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %163 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c0 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c0 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %163 to %alloc_22 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_22 : memref<32x128xf16, 3>
      %164:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %192 = frisk.to_threadTile %alloc_22 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %193 = affine.apply affine_map<() -> (0)>()
        %194 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %195 = vector.extract %192[%193, %194] : f16 from vector<8x8xf16>
        %196 = vector.insert %195, %cst_31 [0, 0] : f16 into vector<4x1xf16>
        %197 = affine.apply affine_map<() -> (1)>()
        %198 = vector.extract %192[%197, %194] : f16 from vector<8x8xf16>
        %199 = vector.insert %198, %196 [1, 0] : f16 into vector<4x1xf16>
        %200 = affine.apply affine_map<() -> (2)>()
        %201 = vector.extract %192[%200, %194] : f16 from vector<8x8xf16>
        %202 = vector.insert %201, %199 [2, 0] : f16 into vector<4x1xf16>
        %203 = affine.apply affine_map<() -> (3)>()
        %204 = vector.extract %192[%203, %194] : f16 from vector<8x8xf16>
        %205 = vector.insert %204, %202 [3, 0] : f16 into vector<4x1xf16>
        %206:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_4, %arg16 = %205) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %261 = frisk.to_threadTile %alloc_22 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %262 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %263 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_34 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %264 = vector.extract %261[%262, %263] : f16 from vector<8x8xf16>
          %265 = vector.insert %264, %cst_34 [0, 0] : f16 into vector<4x1xf16>
          %266 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %267 = vector.extract %261[%266, %263] : f16 from vector<8x8xf16>
          %268 = vector.insert %267, %265 [1, 0] : f16 into vector<4x1xf16>
          %269 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %270 = vector.extract %261[%269, %263] : f16 from vector<8x8xf16>
          %271 = vector.insert %270, %268 [2, 0] : f16 into vector<4x1xf16>
          %272 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %273 = vector.extract %261[%272, %263] : f16 from vector<8x8xf16>
          %274 = vector.insert %273, %271 [3, 0] : f16 into vector<4x1xf16>
          %275 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %276 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_35 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %277 = vector.extract %55[%275, %276] : f16 from vector<2x8xf16>
          %278 = vector.insert %277, %cst_35 [0, 0] : f16 into vector<1x4xf16>
          %279 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %280 = vector.extract %55[%275, %279] : f16 from vector<2x8xf16>
          %281 = vector.insert %280, %278 [0, 1] : f16 into vector<1x4xf16>
          %282 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %283 = vector.extract %55[%275, %282] : f16 from vector<2x8xf16>
          %284 = vector.insert %283, %281 [0, 2] : f16 into vector<1x4xf16>
          %285 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %286 = vector.extract %55[%275, %285] : f16 from vector<2x8xf16>
          %287 = vector.insert %286, %284 [0, 3] : f16 into vector<1x4xf16>
          %288 = frisk.warp_mma_rr %287, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %288, %274 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %207 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %208 = affine.apply affine_map<() -> (4)>()
        %cst_32 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %209 = vector.extract %55[%207, %208] : f16 from vector<2x8xf16>
        %210 = vector.insert %209, %cst_32 [0, 0] : f16 into vector<1x4xf16>
        %211 = affine.apply affine_map<() -> (5)>()
        %212 = vector.extract %55[%207, %211] : f16 from vector<2x8xf16>
        %213 = vector.insert %212, %210 [0, 1] : f16 into vector<1x4xf16>
        %214 = affine.apply affine_map<() -> (6)>()
        %215 = vector.extract %55[%207, %214] : f16 from vector<2x8xf16>
        %216 = vector.insert %215, %213 [0, 2] : f16 into vector<1x4xf16>
        %217 = affine.apply affine_map<() -> (7)>()
        %218 = vector.extract %55[%207, %217] : f16 from vector<2x8xf16>
        %219 = vector.insert %218, %216 [0, 3] : f16 into vector<1x4xf16>
        %220 = frisk.warp_mma_rr %219, %206#1, %206#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %221 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %223 = vector.extract %162[%221, %222] : f32 from vector<2x32xf32>
        %224 = vector.insert %223, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %226 = vector.extract %162[%221, %225] : f32 from vector<2x32xf32>
        %227 = vector.insert %226, %224 [0, 1] : f32 into vector<1x4xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %229 = vector.extract %162[%221, %228] : f32 from vector<2x32xf32>
        %230 = vector.insert %229, %227 [0, 2] : f32 into vector<1x4xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %232 = vector.extract %162[%221, %231] : f32 from vector<2x32xf32>
        %233 = vector.insert %232, %230 [0, 3] : f32 into vector<1x4xf32>
        %234 = arith.addf %233, %220 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %235 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %237 = vector.extract %234[0, 0] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %arg12 [%235, %236] : f32 into vector<2x32xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %240 = vector.extract %234[0, 1] : f32 from vector<1x4xf32>
        %241 = vector.insert %240, %238 [%235, %239] : f32 into vector<2x32xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %243 = vector.extract %234[0, 2] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %241 [%235, %242] : f32 into vector<2x32xf32>
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %246 = vector.extract %234[0, 3] : f32 from vector<1x4xf32>
        %247 = vector.insert %246, %244 [%235, %245] : f32 into vector<2x32xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %249 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %250 = vector.extract %234[0, 0] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %arg13 [%248, %249] : f32 into vector<2x32xf32>
        %252 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %253 = vector.extract %234[0, 1] : f32 from vector<1x4xf32>
        %254 = vector.insert %253, %251 [%248, %252] : f32 into vector<2x32xf32>
        %255 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %256 = vector.extract %234[0, 2] : f32 from vector<1x4xf32>
        %257 = vector.insert %256, %254 [%248, %255] : f32 into vector<2x32xf32>
        %258 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %259 = vector.extract %234[0, 3] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %257 [%248, %258] : f32 into vector<2x32xf32>
        affine.yield %247, %260 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_23 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %165 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c32 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c32 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %165 to %alloc_23 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_23 : memref<32x128xf16, 3>
      %166:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %192 = frisk.to_threadTile %alloc_23 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %193 = affine.apply affine_map<() -> (0)>()
        %194 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %195 = vector.extract %192[%193, %194] : f16 from vector<8x8xf16>
        %196 = vector.insert %195, %cst_31 [0, 0] : f16 into vector<4x1xf16>
        %197 = affine.apply affine_map<() -> (1)>()
        %198 = vector.extract %192[%197, %194] : f16 from vector<8x8xf16>
        %199 = vector.insert %198, %196 [1, 0] : f16 into vector<4x1xf16>
        %200 = affine.apply affine_map<() -> (2)>()
        %201 = vector.extract %192[%200, %194] : f16 from vector<8x8xf16>
        %202 = vector.insert %201, %199 [2, 0] : f16 into vector<4x1xf16>
        %203 = affine.apply affine_map<() -> (3)>()
        %204 = vector.extract %192[%203, %194] : f16 from vector<8x8xf16>
        %205 = vector.insert %204, %202 [3, 0] : f16 into vector<4x1xf16>
        %206:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_4, %arg16 = %205) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %261 = frisk.to_threadTile %alloc_23 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %262 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %263 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_34 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %264 = vector.extract %261[%262, %263] : f16 from vector<8x8xf16>
          %265 = vector.insert %264, %cst_34 [0, 0] : f16 into vector<4x1xf16>
          %266 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %267 = vector.extract %261[%266, %263] : f16 from vector<8x8xf16>
          %268 = vector.insert %267, %265 [1, 0] : f16 into vector<4x1xf16>
          %269 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %270 = vector.extract %261[%269, %263] : f16 from vector<8x8xf16>
          %271 = vector.insert %270, %268 [2, 0] : f16 into vector<4x1xf16>
          %272 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %273 = vector.extract %261[%272, %263] : f16 from vector<8x8xf16>
          %274 = vector.insert %273, %271 [3, 0] : f16 into vector<4x1xf16>
          %275 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %276 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_35 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %277 = vector.extract %85[%275, %276] : f16 from vector<2x8xf16>
          %278 = vector.insert %277, %cst_35 [0, 0] : f16 into vector<1x4xf16>
          %279 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %280 = vector.extract %85[%275, %279] : f16 from vector<2x8xf16>
          %281 = vector.insert %280, %278 [0, 1] : f16 into vector<1x4xf16>
          %282 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %283 = vector.extract %85[%275, %282] : f16 from vector<2x8xf16>
          %284 = vector.insert %283, %281 [0, 2] : f16 into vector<1x4xf16>
          %285 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %286 = vector.extract %85[%275, %285] : f16 from vector<2x8xf16>
          %287 = vector.insert %286, %284 [0, 3] : f16 into vector<1x4xf16>
          %288 = frisk.warp_mma_rr %287, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %288, %274 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %207 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %208 = affine.apply affine_map<() -> (4)>()
        %cst_32 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %209 = vector.extract %85[%207, %208] : f16 from vector<2x8xf16>
        %210 = vector.insert %209, %cst_32 [0, 0] : f16 into vector<1x4xf16>
        %211 = affine.apply affine_map<() -> (5)>()
        %212 = vector.extract %85[%207, %211] : f16 from vector<2x8xf16>
        %213 = vector.insert %212, %210 [0, 1] : f16 into vector<1x4xf16>
        %214 = affine.apply affine_map<() -> (6)>()
        %215 = vector.extract %85[%207, %214] : f16 from vector<2x8xf16>
        %216 = vector.insert %215, %213 [0, 2] : f16 into vector<1x4xf16>
        %217 = affine.apply affine_map<() -> (7)>()
        %218 = vector.extract %85[%207, %217] : f16 from vector<2x8xf16>
        %219 = vector.insert %218, %216 [0, 3] : f16 into vector<1x4xf16>
        %220 = frisk.warp_mma_rr %219, %206#1, %206#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %221 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %223 = vector.extract %164#1[%221, %222] : f32 from vector<2x32xf32>
        %224 = vector.insert %223, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %226 = vector.extract %164#1[%221, %225] : f32 from vector<2x32xf32>
        %227 = vector.insert %226, %224 [0, 1] : f32 into vector<1x4xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %229 = vector.extract %164#1[%221, %228] : f32 from vector<2x32xf32>
        %230 = vector.insert %229, %227 [0, 2] : f32 into vector<1x4xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %232 = vector.extract %164#1[%221, %231] : f32 from vector<2x32xf32>
        %233 = vector.insert %232, %230 [0, 3] : f32 into vector<1x4xf32>
        %234 = arith.addf %233, %220 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %235 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %237 = vector.extract %234[0, 0] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %arg12 [%235, %236] : f32 into vector<2x32xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %240 = vector.extract %234[0, 1] : f32 from vector<1x4xf32>
        %241 = vector.insert %240, %238 [%235, %239] : f32 into vector<2x32xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %243 = vector.extract %234[0, 2] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %241 [%235, %242] : f32 into vector<2x32xf32>
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %246 = vector.extract %234[0, 3] : f32 from vector<1x4xf32>
        %247 = vector.insert %246, %244 [%235, %245] : f32 into vector<2x32xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %249 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %250 = vector.extract %234[0, 0] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %arg13 [%248, %249] : f32 into vector<2x32xf32>
        %252 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %253 = vector.extract %234[0, 1] : f32 from vector<1x4xf32>
        %254 = vector.insert %253, %251 [%248, %252] : f32 into vector<2x32xf32>
        %255 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %256 = vector.extract %234[0, 2] : f32 from vector<1x4xf32>
        %257 = vector.insert %256, %254 [%248, %255] : f32 into vector<2x32xf32>
        %258 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %259 = vector.extract %234[0, 3] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %257 [%248, %258] : f32 into vector<2x32xf32>
        affine.yield %247, %260 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_24 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %167 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c64 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c64 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %167 to %alloc_24 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_24 : memref<32x128xf16, 3>
      %168:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %192 = frisk.to_threadTile %alloc_24 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %193 = affine.apply affine_map<() -> (0)>()
        %194 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %195 = vector.extract %192[%193, %194] : f16 from vector<8x8xf16>
        %196 = vector.insert %195, %cst_31 [0, 0] : f16 into vector<4x1xf16>
        %197 = affine.apply affine_map<() -> (1)>()
        %198 = vector.extract %192[%197, %194] : f16 from vector<8x8xf16>
        %199 = vector.insert %198, %196 [1, 0] : f16 into vector<4x1xf16>
        %200 = affine.apply affine_map<() -> (2)>()
        %201 = vector.extract %192[%200, %194] : f16 from vector<8x8xf16>
        %202 = vector.insert %201, %199 [2, 0] : f16 into vector<4x1xf16>
        %203 = affine.apply affine_map<() -> (3)>()
        %204 = vector.extract %192[%203, %194] : f16 from vector<8x8xf16>
        %205 = vector.insert %204, %202 [3, 0] : f16 into vector<4x1xf16>
        %206:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_4, %arg16 = %205) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %261 = frisk.to_threadTile %alloc_24 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %262 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %263 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_34 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %264 = vector.extract %261[%262, %263] : f16 from vector<8x8xf16>
          %265 = vector.insert %264, %cst_34 [0, 0] : f16 into vector<4x1xf16>
          %266 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %267 = vector.extract %261[%266, %263] : f16 from vector<8x8xf16>
          %268 = vector.insert %267, %265 [1, 0] : f16 into vector<4x1xf16>
          %269 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %270 = vector.extract %261[%269, %263] : f16 from vector<8x8xf16>
          %271 = vector.insert %270, %268 [2, 0] : f16 into vector<4x1xf16>
          %272 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %273 = vector.extract %261[%272, %263] : f16 from vector<8x8xf16>
          %274 = vector.insert %273, %271 [3, 0] : f16 into vector<4x1xf16>
          %275 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %276 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_35 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %277 = vector.extract %115[%275, %276] : f16 from vector<2x8xf16>
          %278 = vector.insert %277, %cst_35 [0, 0] : f16 into vector<1x4xf16>
          %279 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %280 = vector.extract %115[%275, %279] : f16 from vector<2x8xf16>
          %281 = vector.insert %280, %278 [0, 1] : f16 into vector<1x4xf16>
          %282 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %283 = vector.extract %115[%275, %282] : f16 from vector<2x8xf16>
          %284 = vector.insert %283, %281 [0, 2] : f16 into vector<1x4xf16>
          %285 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %286 = vector.extract %115[%275, %285] : f16 from vector<2x8xf16>
          %287 = vector.insert %286, %284 [0, 3] : f16 into vector<1x4xf16>
          %288 = frisk.warp_mma_rr %287, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %288, %274 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %207 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %208 = affine.apply affine_map<() -> (4)>()
        %cst_32 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %209 = vector.extract %115[%207, %208] : f16 from vector<2x8xf16>
        %210 = vector.insert %209, %cst_32 [0, 0] : f16 into vector<1x4xf16>
        %211 = affine.apply affine_map<() -> (5)>()
        %212 = vector.extract %115[%207, %211] : f16 from vector<2x8xf16>
        %213 = vector.insert %212, %210 [0, 1] : f16 into vector<1x4xf16>
        %214 = affine.apply affine_map<() -> (6)>()
        %215 = vector.extract %115[%207, %214] : f16 from vector<2x8xf16>
        %216 = vector.insert %215, %213 [0, 2] : f16 into vector<1x4xf16>
        %217 = affine.apply affine_map<() -> (7)>()
        %218 = vector.extract %115[%207, %217] : f16 from vector<2x8xf16>
        %219 = vector.insert %218, %216 [0, 3] : f16 into vector<1x4xf16>
        %220 = frisk.warp_mma_rr %219, %206#1, %206#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %221 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %223 = vector.extract %166#1[%221, %222] : f32 from vector<2x32xf32>
        %224 = vector.insert %223, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %226 = vector.extract %166#1[%221, %225] : f32 from vector<2x32xf32>
        %227 = vector.insert %226, %224 [0, 1] : f32 into vector<1x4xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %229 = vector.extract %166#1[%221, %228] : f32 from vector<2x32xf32>
        %230 = vector.insert %229, %227 [0, 2] : f32 into vector<1x4xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %232 = vector.extract %166#1[%221, %231] : f32 from vector<2x32xf32>
        %233 = vector.insert %232, %230 [0, 3] : f32 into vector<1x4xf32>
        %234 = arith.addf %233, %220 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %235 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %237 = vector.extract %234[0, 0] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %arg12 [%235, %236] : f32 into vector<2x32xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %240 = vector.extract %234[0, 1] : f32 from vector<1x4xf32>
        %241 = vector.insert %240, %238 [%235, %239] : f32 into vector<2x32xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %243 = vector.extract %234[0, 2] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %241 [%235, %242] : f32 into vector<2x32xf32>
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %246 = vector.extract %234[0, 3] : f32 from vector<1x4xf32>
        %247 = vector.insert %246, %244 [%235, %245] : f32 into vector<2x32xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %249 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %250 = vector.extract %234[0, 0] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %arg13 [%248, %249] : f32 into vector<2x32xf32>
        %252 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %253 = vector.extract %234[0, 1] : f32 from vector<1x4xf32>
        %254 = vector.insert %253, %251 [%248, %252] : f32 into vector<2x32xf32>
        %255 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %256 = vector.extract %234[0, 2] : f32 from vector<1x4xf32>
        %257 = vector.insert %256, %254 [%248, %255] : f32 into vector<2x32xf32>
        %258 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %259 = vector.extract %234[0, 3] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %257 [%248, %258] : f32 into vector<2x32xf32>
        affine.yield %247, %260 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_25 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %169 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c96 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c96 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %169 to %alloc_25 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_25 : memref<32x128xf16, 3>
      %170:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %192 = frisk.to_threadTile %alloc_25 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %193 = affine.apply affine_map<() -> (0)>()
        %194 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %195 = vector.extract %192[%193, %194] : f16 from vector<8x8xf16>
        %196 = vector.insert %195, %cst_31 [0, 0] : f16 into vector<4x1xf16>
        %197 = affine.apply affine_map<() -> (1)>()
        %198 = vector.extract %192[%197, %194] : f16 from vector<8x8xf16>
        %199 = vector.insert %198, %196 [1, 0] : f16 into vector<4x1xf16>
        %200 = affine.apply affine_map<() -> (2)>()
        %201 = vector.extract %192[%200, %194] : f16 from vector<8x8xf16>
        %202 = vector.insert %201, %199 [2, 0] : f16 into vector<4x1xf16>
        %203 = affine.apply affine_map<() -> (3)>()
        %204 = vector.extract %192[%203, %194] : f16 from vector<8x8xf16>
        %205 = vector.insert %204, %202 [3, 0] : f16 into vector<4x1xf16>
        %206:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_4, %arg16 = %205) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %261 = frisk.to_threadTile %alloc_25 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %262 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %263 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_34 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %264 = vector.extract %261[%262, %263] : f16 from vector<8x8xf16>
          %265 = vector.insert %264, %cst_34 [0, 0] : f16 into vector<4x1xf16>
          %266 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %267 = vector.extract %261[%266, %263] : f16 from vector<8x8xf16>
          %268 = vector.insert %267, %265 [1, 0] : f16 into vector<4x1xf16>
          %269 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %270 = vector.extract %261[%269, %263] : f16 from vector<8x8xf16>
          %271 = vector.insert %270, %268 [2, 0] : f16 into vector<4x1xf16>
          %272 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %273 = vector.extract %261[%272, %263] : f16 from vector<8x8xf16>
          %274 = vector.insert %273, %271 [3, 0] : f16 into vector<4x1xf16>
          %275 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %276 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_35 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %277 = vector.extract %145[%275, %276] : f16 from vector<2x8xf16>
          %278 = vector.insert %277, %cst_35 [0, 0] : f16 into vector<1x4xf16>
          %279 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %280 = vector.extract %145[%275, %279] : f16 from vector<2x8xf16>
          %281 = vector.insert %280, %278 [0, 1] : f16 into vector<1x4xf16>
          %282 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %283 = vector.extract %145[%275, %282] : f16 from vector<2x8xf16>
          %284 = vector.insert %283, %281 [0, 2] : f16 into vector<1x4xf16>
          %285 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %286 = vector.extract %145[%275, %285] : f16 from vector<2x8xf16>
          %287 = vector.insert %286, %284 [0, 3] : f16 into vector<1x4xf16>
          %288 = frisk.warp_mma_rr %287, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %288, %274 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %207 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %208 = affine.apply affine_map<() -> (4)>()
        %cst_32 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %209 = vector.extract %145[%207, %208] : f16 from vector<2x8xf16>
        %210 = vector.insert %209, %cst_32 [0, 0] : f16 into vector<1x4xf16>
        %211 = affine.apply affine_map<() -> (5)>()
        %212 = vector.extract %145[%207, %211] : f16 from vector<2x8xf16>
        %213 = vector.insert %212, %210 [0, 1] : f16 into vector<1x4xf16>
        %214 = affine.apply affine_map<() -> (6)>()
        %215 = vector.extract %145[%207, %214] : f16 from vector<2x8xf16>
        %216 = vector.insert %215, %213 [0, 2] : f16 into vector<1x4xf16>
        %217 = affine.apply affine_map<() -> (7)>()
        %218 = vector.extract %145[%207, %217] : f16 from vector<2x8xf16>
        %219 = vector.insert %218, %216 [0, 3] : f16 into vector<1x4xf16>
        %220 = frisk.warp_mma_rr %219, %206#1, %206#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %221 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %223 = vector.extract %168#1[%221, %222] : f32 from vector<2x32xf32>
        %224 = vector.insert %223, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %226 = vector.extract %168#1[%221, %225] : f32 from vector<2x32xf32>
        %227 = vector.insert %226, %224 [0, 1] : f32 into vector<1x4xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %229 = vector.extract %168#1[%221, %228] : f32 from vector<2x32xf32>
        %230 = vector.insert %229, %227 [0, 2] : f32 into vector<1x4xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %232 = vector.extract %168#1[%221, %231] : f32 from vector<2x32xf32>
        %233 = vector.insert %232, %230 [0, 3] : f32 into vector<1x4xf32>
        %234 = arith.addf %233, %220 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %235 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %237 = vector.extract %234[0, 0] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %arg12 [%235, %236] : f32 into vector<2x32xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %240 = vector.extract %234[0, 1] : f32 from vector<1x4xf32>
        %241 = vector.insert %240, %238 [%235, %239] : f32 into vector<2x32xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %243 = vector.extract %234[0, 2] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %241 [%235, %242] : f32 into vector<2x32xf32>
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %246 = vector.extract %234[0, 3] : f32 from vector<1x4xf32>
        %247 = vector.insert %246, %244 [%235, %245] : f32 into vector<2x32xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %249 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %250 = vector.extract %234[0, 0] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %arg13 [%248, %249] : f32 into vector<2x32xf32>
        %252 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %253 = vector.extract %234[0, 1] : f32 from vector<1x4xf32>
        %254 = vector.insert %253, %251 [%248, %252] : f32 into vector<2x32xf32>
        %255 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %256 = vector.extract %234[0, 2] : f32 from vector<1x4xf32>
        %257 = vector.insert %256, %254 [%248, %255] : f32 into vector<2x32xf32>
        %258 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %259 = vector.extract %234[0, 3] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %257 [%248, %258] : f32 into vector<2x32xf32>
        affine.yield %247, %260 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %171 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %194 = vector.extract %170#1[%192, %193] : f32 from vector<2x32xf32>
        %195 = vector.insert %194, %cst_31 [0, 0] : f32 into vector<1x4xf32>
        %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %197 = vector.extract %170#1[%192, %196] : f32 from vector<2x32xf32>
        %198 = vector.insert %197, %195 [0, 1] : f32 into vector<1x4xf32>
        %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %200 = vector.extract %170#1[%192, %199] : f32 from vector<2x32xf32>
        %201 = vector.insert %200, %198 [0, 2] : f32 into vector<1x4xf32>
        %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %203 = vector.extract %170#1[%192, %202] : f32 from vector<2x32xf32>
        %204 = vector.insert %203, %201 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        %205 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %207 = vector.extract %204[0, 0] : f32 from vector<1x4xf32>
        %208 = vector.insert %207, %arg12 [%205, %206] : f32 into vector<2x32xf32>
        %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %210 = vector.extract %204[0, 1] : f32 from vector<1x4xf32>
        %211 = vector.insert %210, %208 [%205, %209] : f32 into vector<2x32xf32>
        %212 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %213 = vector.extract %204[0, 2] : f32 from vector<1x4xf32>
        %214 = vector.insert %213, %211 [%205, %212] : f32 into vector<2x32xf32>
        %215 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %216 = vector.extract %204[0, 3] : f32 from vector<1x4xf32>
        %217 = vector.insert %216, %214 [%205, %215] : f32 into vector<2x32xf32>
        affine.yield %217 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %172:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11, %arg13 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %192 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_12) -> (f32) {
          %209 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %210 = vector.extract %171[%arg11, %209] : f32 from vector<2x32xf32>
          %211 = arith.maxnumf %arg15, %210 : f32
          %212 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %213 = vector.extract %171[%arg11, %212] : f32 from vector<2x32xf32>
          %214 = arith.maxnumf %211, %213 : f32
          %215 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %216 = vector.extract %171[%arg11, %215] : f32 from vector<2x32xf32>
          %217 = arith.maxnumf %214, %216 : f32
          %218 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %219 = vector.extract %171[%arg11, %218] : f32 from vector<2x32xf32>
          %220 = arith.maxnumf %217, %219 : f32
          affine.yield %220 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %192, %c16_i32, %c64_i32 : f32
        %193 = arith.maxnumf %192, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %193, %c32_i32, %c64_i32 : f32
        %194 = arith.maxnumf %193, %shuffleResult_31 : f32
        %195 = vector.broadcast %194 : f32 to vector<1x1xf32>
        %196 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %197 = affine.apply affine_map<() -> (0)>()
        %198 = vector.extract %195[0, 0] : f32 from vector<1x1xf32>
        %199 = vector.insert %198, %arg12 [%196, %197] : f32 into vector<2x1xf32>
        %200 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %201 = affine.apply affine_map<() -> (0)>()
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %202 = vector.extract %arg7[%200, %201] : f32 from vector<2x1xf32>
        %203 = vector.insert %202, %cst_33 [0, 0] : f32 into vector<1x1xf32>
        %204 = arith.maxnumf %195, %203 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %205 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %206 = affine.apply affine_map<() -> (0)>()
        %207 = vector.extract %204[0, 0] : f32 from vector<1x1xf32>
        %208 = vector.insert %207, %arg13 [%205, %206] : f32 into vector<2x1xf32>
        affine.yield %199, %208 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %173:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %193 = vector.extract %172#1[%192, 0] : f32 from vector<2x1xf32>
        %194 = vector.broadcast %193 : f32 to vector<1x4xf32>
        %195 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %197 = vector.extract %194[0, 0] : f32 from vector<1x4xf32>
        %198 = vector.insert %197, %arg12 [%195, %196] : f32 into vector<2x32xf32>
        %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %200 = vector.extract %194[0, 1] : f32 from vector<1x4xf32>
        %201 = vector.insert %200, %198 [%195, %199] : f32 into vector<2x32xf32>
        %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %203 = vector.extract %194[0, 2] : f32 from vector<1x4xf32>
        %204 = vector.insert %203, %201 [%195, %202] : f32 into vector<2x32xf32>
        %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %206 = vector.extract %194[0, 3] : f32 from vector<1x4xf32>
        %207 = vector.insert %206, %204 [%195, %205] : f32 into vector<2x32xf32>
        %208 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %210 = vector.extract %170#1[%208, %209] : f32 from vector<2x32xf32>
        %211 = vector.insert %210, %cst_31 [0, 0] : f32 into vector<1x4xf32>
        %212 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %213 = vector.extract %170#1[%208, %212] : f32 from vector<2x32xf32>
        %214 = vector.insert %213, %211 [0, 1] : f32 into vector<1x4xf32>
        %215 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %216 = vector.extract %170#1[%208, %215] : f32 from vector<2x32xf32>
        %217 = vector.insert %216, %214 [0, 2] : f32 into vector<1x4xf32>
        %218 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %219 = vector.extract %170#1[%208, %218] : f32 from vector<2x32xf32>
        %220 = vector.insert %219, %217 [0, 3] : f32 into vector<1x4xf32>
        %221 = arith.subf %220, %194 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %224 = vector.extract %221[0, 0] : f32 from vector<1x4xf32>
        %225 = vector.insert %224, %arg13 [%222, %223] : f32 into vector<2x32xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %227 = vector.extract %221[0, 1] : f32 from vector<1x4xf32>
        %228 = vector.insert %227, %225 [%222, %226] : f32 into vector<2x32xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %230 = vector.extract %221[0, 2] : f32 from vector<1x4xf32>
        %231 = vector.insert %230, %228 [%222, %229] : f32 into vector<2x32xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %233 = vector.extract %221[0, 3] : f32 from vector<1x4xf32>
        %234 = vector.insert %233, %231 [%222, %232] : f32 into vector<2x32xf32>
        affine.yield %207, %234 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>}
      %174:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11, %arg13 = %cst_11, %arg14 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %193 = affine.apply affine_map<() -> (0)>()
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %194 = vector.extract %arg7[%192, %193] : f32 from vector<2x1xf32>
        %195 = vector.insert %194, %cst_31 [0, 0] : f32 into vector<1x1xf32>
        %196 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %197 = affine.apply affine_map<() -> (0)>()
        %cst_32 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %198 = vector.extract %172#1[%196, %197] : f32 from vector<2x1xf32>
        %199 = vector.insert %198, %cst_32 [0, 0] : f32 into vector<1x1xf32>
        %200 = arith.subf %195, %199 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %201 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %202 = affine.apply affine_map<() -> (0)>()
        %203 = vector.extract %200[0, 0] : f32 from vector<1x1xf32>
        %204 = vector.insert %203, %arg12 [%201, %202] : f32 into vector<2x1xf32>
        %205 = math.exp2 %200 {tiled = true} : vector<1x1xf32>
        %206 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %207 = affine.apply affine_map<() -> (0)>()
        %208 = vector.extract %205[0, 0] : f32 from vector<1x1xf32>
        %209 = vector.insert %208, %arg13 [%206, %207] : f32 into vector<2x1xf32>
        %210 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %211 = affine.apply affine_map<() -> (0)>()
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %212 = vector.extract %arg6[%210, %211] : f32 from vector<2x1xf32>
        %213 = vector.insert %212, %cst_33 [0, 0] : f32 into vector<1x1xf32>
        %214 = arith.mulf %213, %205 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %215 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %216 = affine.apply affine_map<() -> (0)>()
        %217 = vector.extract %214[0, 0] : f32 from vector<1x1xf32>
        %218 = vector.insert %217, %arg14 [%215, %216] : f32 into vector<2x1xf32>
        affine.yield %204, %209, %218 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %175:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5, %arg13 = %cst_5) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %194 = vector.extract %173#1[%192, %193] : f32 from vector<2x32xf32>
        %195 = vector.insert %194, %cst_31 [0, 0] : f32 into vector<1x4xf32>
        %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %197 = vector.extract %173#1[%192, %196] : f32 from vector<2x32xf32>
        %198 = vector.insert %197, %195 [0, 1] : f32 into vector<1x4xf32>
        %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %200 = vector.extract %173#1[%192, %199] : f32 from vector<2x32xf32>
        %201 = vector.insert %200, %198 [0, 2] : f32 into vector<1x4xf32>
        %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %203 = vector.extract %173#1[%192, %202] : f32 from vector<2x32xf32>
        %204 = vector.insert %203, %201 [0, 3] : f32 into vector<1x4xf32>
        %205 = math.exp2 %204 {tiled = true} : vector<1x4xf32>
        %206 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %208 = vector.extract %205[0, 0] : f32 from vector<1x4xf32>
        %209 = vector.insert %208, %arg12 [%206, %207] : f32 into vector<2x32xf32>
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %211 = vector.extract %205[0, 1] : f32 from vector<1x4xf32>
        %212 = vector.insert %211, %209 [%206, %210] : f32 into vector<2x32xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %214 = vector.extract %205[0, 2] : f32 from vector<1x4xf32>
        %215 = vector.insert %214, %212 [%206, %213] : f32 into vector<2x32xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %217 = vector.extract %205[0, 3] : f32 from vector<1x4xf32>
        %218 = vector.insert %217, %215 [%206, %216] : f32 into vector<2x32xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %221 = vector.extract %205[0, 0] : f32 from vector<1x4xf32>
        %222 = vector.insert %221, %arg13 [%219, %220] : f32 into vector<2x32xf32>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %224 = vector.extract %205[0, 1] : f32 from vector<1x4xf32>
        %225 = vector.insert %224, %222 [%219, %223] : f32 into vector<2x32xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %227 = vector.extract %205[0, 2] : f32 from vector<1x4xf32>
        %228 = vector.insert %227, %225 [%219, %226] : f32 into vector<2x32xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %230 = vector.extract %205[0, 3] : f32 from vector<1x4xf32>
        %231 = vector.insert %230, %228 [%219, %229] : f32 into vector<2x32xf32>
        affine.yield %218, %231 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %176:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11, %arg13 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %192 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst_3) -> (f32) {
          %209 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %210 = vector.extract %175#1[%arg11, %209] : f32 from vector<2x32xf32>
          %211 = arith.addf %arg15, %210 : f32
          %212 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %213 = vector.extract %175#1[%arg11, %212] : f32 from vector<2x32xf32>
          %214 = arith.addf %211, %213 : f32
          %215 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %216 = vector.extract %175#1[%arg11, %215] : f32 from vector<2x32xf32>
          %217 = arith.addf %214, %216 : f32
          %218 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %219 = vector.extract %175#1[%arg11, %218] : f32 from vector<2x32xf32>
          %220 = arith.addf %217, %219 : f32
          affine.yield %220 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %shuffleResult, %valid = gpu.shuffle xor %192, %c16_i32, %c64_i32 : f32
        %193 = arith.addf %192, %shuffleResult : f32
        %shuffleResult_31, %valid_32 = gpu.shuffle xor %193, %c32_i32, %c64_i32 : f32
        %194 = arith.addf %193, %shuffleResult_31 : f32
        %195 = vector.broadcast %194 : f32 to vector<1x1xf32>
        %196 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %197 = affine.apply affine_map<() -> (0)>()
        %198 = vector.extract %195[0, 0] : f32 from vector<1x1xf32>
        %199 = vector.insert %198, %arg12 [%196, %197] : f32 into vector<2x1xf32>
        %200 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %201 = affine.apply affine_map<() -> (0)>()
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %202 = vector.extract %174#2[%200, %201] : f32 from vector<2x1xf32>
        %203 = vector.insert %202, %cst_33 [0, 0] : f32 into vector<1x1xf32>
        %204 = arith.addf %195, %203 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %205 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %206 = affine.apply affine_map<() -> (0)>()
        %207 = vector.extract %204[0, 0] : f32 from vector<1x1xf32>
        %208 = vector.insert %207, %arg13 [%205, %206] : f32 into vector<2x1xf32>
        affine.yield %199, %208 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, tiled = true}
      %177 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_2) -> (vector<2x32xf16>) {
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %194 = vector.extract %175#0[%192, %193] : f32 from vector<2x32xf32>
        %195 = vector.insert %194, %cst_31 [0, 0] : f32 into vector<1x4xf32>
        %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %197 = vector.extract %175#0[%192, %196] : f32 from vector<2x32xf32>
        %198 = vector.insert %197, %195 [0, 1] : f32 into vector<1x4xf32>
        %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %200 = vector.extract %175#0[%192, %199] : f32 from vector<2x32xf32>
        %201 = vector.insert %200, %198 [0, 2] : f32 into vector<1x4xf32>
        %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %203 = vector.extract %175#0[%192, %202] : f32 from vector<2x32xf32>
        %204 = vector.insert %203, %201 [0, 3] : f32 into vector<1x4xf32>
        %205 = arith.truncf %204 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %206 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %208 = vector.extract %205[0, 0] : f16 from vector<1x4xf16>
        %209 = vector.insert %208, %arg12 [%206, %207] : f16 into vector<2x32xf16>
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %211 = vector.extract %205[0, 1] : f16 from vector<1x4xf16>
        %212 = vector.insert %211, %209 [%206, %210] : f16 into vector<2x32xf16>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %214 = vector.extract %205[0, 2] : f16 from vector<1x4xf16>
        %215 = vector.insert %214, %212 [%206, %213] : f16 into vector<2x32xf16>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %217 = vector.extract %205[0, 3] : f16 from vector<1x4xf16>
        %218 = vector.insert %217, %215 [%206, %216] : f16 into vector<2x32xf16>
        affine.yield %218 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_26 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %178 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %178 to %alloc_26 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_26 : memref<128x32xf16, 3>
      %thread_id_x_27 = gpu.thread_id x
      %179 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %192 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_27]
        %193 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_27]
        %194 = frisk.read_matrix_from_shm %alloc_26[%192, %193] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %195 = vector.bitcast %194 : vector<4xi32> to vector<8xf16>
        %196 = vector.extract %195[0] : f16 from vector<8xf16>
        %197 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_27]
        %198 = vector.extract %195[%197] : f16 from vector<8xf16>
        %199 = vector.extract %195[1] : f16 from vector<8xf16>
        %200 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_27]
        %201 = vector.extract %195[%200] : f16 from vector<8xf16>
        %202 = vector.extract %195[2] : f16 from vector<8xf16>
        %203 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_27]
        %204 = vector.extract %195[%203] : f16 from vector<8xf16>
        %205 = vector.extract %195[3] : f16 from vector<8xf16>
        %206 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_27]
        %207 = vector.extract %195[%206] : f16 from vector<8xf16>
        %208 = vector.from_elements %196, %198, %199, %201, %202, %204, %205, %207 : vector<4x2xf16>
        %209 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %210 = affine.apply affine_map<() -> (0)>()
        %211 = vector.extract %208[0, 0] : f16 from vector<4x2xf16>
        %212 = vector.insert %211, %arg12 [%209, %210] : f16 into vector<32x2xf16>
        %213 = affine.apply affine_map<() -> (1)>()
        %214 = vector.extract %208[0, 1] : f16 from vector<4x2xf16>
        %215 = vector.insert %214, %212 [%209, %213] : f16 into vector<32x2xf16>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %217 = vector.extract %208[1, 0] : f16 from vector<4x2xf16>
        %218 = vector.insert %217, %215 [%216, %210] : f16 into vector<32x2xf16>
        %219 = affine.apply affine_map<() -> (1)>()
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %221 = vector.extract %208[1, 1] : f16 from vector<4x2xf16>
        %222 = vector.insert %221, %218 [%220, %219] : f16 into vector<32x2xf16>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %224 = vector.extract %208[2, 0] : f16 from vector<4x2xf16>
        %225 = vector.insert %224, %222 [%223, %210] : f16 into vector<32x2xf16>
        %226 = affine.apply affine_map<() -> (1)>()
        %227 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %228 = vector.extract %208[2, 1] : f16 from vector<4x2xf16>
        %229 = vector.insert %228, %225 [%227, %226] : f16 into vector<32x2xf16>
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %231 = vector.extract %208[3, 0] : f16 from vector<4x2xf16>
        %232 = vector.insert %231, %229 [%230, %210] : f16 into vector<32x2xf16>
        %233 = affine.apply affine_map<() -> (1)>()
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %235 = vector.extract %208[3, 1] : f16 from vector<4x2xf16>
        %236 = vector.insert %235, %232 [%234, %233] : f16 into vector<32x2xf16>
        affine.yield %236 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %180:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %193 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_4) -> (vector<1x4xf32>) {
          %263 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %264 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %cst_32 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %265 = vector.extract %177[%263, %264] : f16 from vector<2x32xf16>
          %266 = vector.insert %265, %cst_32 [0, 0] : f16 into vector<1x4xf16>
          %267 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %268 = vector.extract %177[%263, %267] : f16 from vector<2x32xf16>
          %269 = vector.insert %268, %266 [0, 1] : f16 into vector<1x4xf16>
          %270 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %271 = vector.extract %177[%263, %270] : f16 from vector<2x32xf16>
          %272 = vector.insert %271, %269 [0, 2] : f16 into vector<1x4xf16>
          %273 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %274 = vector.extract %177[%263, %273] : f16 from vector<2x32xf16>
          %275 = vector.insert %274, %272 [0, 3] : f16 into vector<1x4xf16>
          %276 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %277 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_33 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %278 = vector.extract %179[%276, %277] : f16 from vector<32x2xf16>
          %279 = vector.insert %278, %cst_33 [0, 0] : f16 into vector<4x1xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %281 = vector.extract %179[%280, %277] : f16 from vector<32x2xf16>
          %282 = vector.insert %281, %279 [1, 0] : f16 into vector<4x1xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %284 = vector.extract %179[%283, %277] : f16 from vector<32x2xf16>
          %285 = vector.insert %284, %282 [2, 0] : f16 into vector<4x1xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %287 = vector.extract %179[%286, %277] : f16 from vector<32x2xf16>
          %288 = vector.insert %287, %285 [3, 0] : f16 into vector<4x1xf16>
          %289 = frisk.warp_mma_rr %275, %288, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %289 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %194 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %195 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %196 = vector.extract %193[0, 0] : f32 from vector<1x4xf32>
        %197 = vector.insert %196, %arg12 [%194, %195] : f32 into vector<2x8xf32>
        %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %199 = vector.extract %193[0, 1] : f32 from vector<1x4xf32>
        %200 = vector.insert %199, %197 [%194, %198] : f32 into vector<2x8xf32>
        %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %202 = vector.extract %193[0, 2] : f32 from vector<1x4xf32>
        %203 = vector.insert %202, %200 [%194, %201] : f32 into vector<2x8xf32>
        %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %205 = vector.extract %193[0, 3] : f32 from vector<1x4xf32>
        %206 = vector.insert %205, %203 [%194, %204] : f32 into vector<2x8xf32>
        %207 = vector.extract %174#1[%192, 0] : f32 from vector<2x1xf32>
        %208 = vector.broadcast %207 : f32 to vector<1x4xf32>
        %209 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %211 = vector.extract %208[0, 0] : f32 from vector<1x4xf32>
        %212 = vector.insert %211, %arg13 [%209, %210] : f32 into vector<2x8xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %214 = vector.extract %208[0, 1] : f32 from vector<1x4xf32>
        %215 = vector.insert %214, %212 [%209, %213] : f32 into vector<2x8xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %217 = vector.extract %208[0, 2] : f32 from vector<1x4xf32>
        %218 = vector.insert %217, %215 [%209, %216] : f32 into vector<2x8xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %220 = vector.extract %208[0, 3] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %218 [%209, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %224 = vector.extract %arg5[%222, %223] : f32 from vector<2x8xf32>
        %225 = vector.insert %224, %cst_31 [0, 0] : f32 into vector<1x4xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %227 = vector.extract %arg5[%222, %226] : f32 from vector<2x8xf32>
        %228 = vector.insert %227, %225 [0, 1] : f32 into vector<1x4xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %230 = vector.extract %arg5[%222, %229] : f32 from vector<2x8xf32>
        %231 = vector.insert %230, %228 [0, 2] : f32 into vector<1x4xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %233 = vector.extract %arg5[%222, %232] : f32 from vector<2x8xf32>
        %234 = vector.insert %233, %231 [0, 3] : f32 into vector<1x4xf32>
        %235 = arith.mulf %234, %208 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %238 = vector.extract %235[0, 0] : f32 from vector<1x4xf32>
        %239 = vector.insert %238, %arg14 [%236, %237] : f32 into vector<2x8xf32>
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %241 = vector.extract %235[0, 1] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %239 [%236, %240] : f32 into vector<2x8xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %244 = vector.extract %235[0, 2] : f32 from vector<1x4xf32>
        %245 = vector.insert %244, %242 [%236, %243] : f32 into vector<2x8xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %247 = vector.extract %235[0, 3] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %245 [%236, %246] : f32 into vector<2x8xf32>
        %249 = arith.addf %235, %193 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %252 = vector.extract %249[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg15 [%250, %251] : f32 into vector<2x8xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %255 = vector.extract %249[0, 1] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %253 [%250, %254] : f32 into vector<2x8xf32>
        %257 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %258 = vector.extract %249[0, 2] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %256 [%250, %257] : f32 into vector<2x8xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %261 = vector.extract %249[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%250, %260] : f32 into vector<2x8xf32>
        affine.yield %206, %221, %248, %262 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_28 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %181 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %181 to %alloc_28 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_28 : memref<128x32xf16, 3>
      %182 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %192 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_27]
        %193 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_27]
        %194 = frisk.read_matrix_from_shm %alloc_28[%192, %193] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %195 = vector.bitcast %194 : vector<4xi32> to vector<8xf16>
        %196 = vector.extract %195[0] : f16 from vector<8xf16>
        %197 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_27]
        %198 = vector.extract %195[%197] : f16 from vector<8xf16>
        %199 = vector.extract %195[1] : f16 from vector<8xf16>
        %200 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_27]
        %201 = vector.extract %195[%200] : f16 from vector<8xf16>
        %202 = vector.extract %195[2] : f16 from vector<8xf16>
        %203 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_27]
        %204 = vector.extract %195[%203] : f16 from vector<8xf16>
        %205 = vector.extract %195[3] : f16 from vector<8xf16>
        %206 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_27]
        %207 = vector.extract %195[%206] : f16 from vector<8xf16>
        %208 = vector.from_elements %196, %198, %199, %201, %202, %204, %205, %207 : vector<4x2xf16>
        %209 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %210 = affine.apply affine_map<() -> (0)>()
        %211 = vector.extract %208[0, 0] : f16 from vector<4x2xf16>
        %212 = vector.insert %211, %arg12 [%209, %210] : f16 into vector<32x2xf16>
        %213 = affine.apply affine_map<() -> (1)>()
        %214 = vector.extract %208[0, 1] : f16 from vector<4x2xf16>
        %215 = vector.insert %214, %212 [%209, %213] : f16 into vector<32x2xf16>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %217 = vector.extract %208[1, 0] : f16 from vector<4x2xf16>
        %218 = vector.insert %217, %215 [%216, %210] : f16 into vector<32x2xf16>
        %219 = affine.apply affine_map<() -> (1)>()
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %221 = vector.extract %208[1, 1] : f16 from vector<4x2xf16>
        %222 = vector.insert %221, %218 [%220, %219] : f16 into vector<32x2xf16>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %224 = vector.extract %208[2, 0] : f16 from vector<4x2xf16>
        %225 = vector.insert %224, %222 [%223, %210] : f16 into vector<32x2xf16>
        %226 = affine.apply affine_map<() -> (1)>()
        %227 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %228 = vector.extract %208[2, 1] : f16 from vector<4x2xf16>
        %229 = vector.insert %228, %225 [%227, %226] : f16 into vector<32x2xf16>
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %231 = vector.extract %208[3, 0] : f16 from vector<4x2xf16>
        %232 = vector.insert %231, %229 [%230, %210] : f16 into vector<32x2xf16>
        %233 = affine.apply affine_map<() -> (1)>()
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %235 = vector.extract %208[3, 1] : f16 from vector<4x2xf16>
        %236 = vector.insert %235, %232 [%234, %233] : f16 into vector<32x2xf16>
        affine.yield %236 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %183:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %193 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_4) -> (vector<1x4xf32>) {
          %263 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %264 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %cst_32 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %265 = vector.extract %177[%263, %264] : f16 from vector<2x32xf16>
          %266 = vector.insert %265, %cst_32 [0, 0] : f16 into vector<1x4xf16>
          %267 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %268 = vector.extract %177[%263, %267] : f16 from vector<2x32xf16>
          %269 = vector.insert %268, %266 [0, 1] : f16 into vector<1x4xf16>
          %270 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %271 = vector.extract %177[%263, %270] : f16 from vector<2x32xf16>
          %272 = vector.insert %271, %269 [0, 2] : f16 into vector<1x4xf16>
          %273 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %274 = vector.extract %177[%263, %273] : f16 from vector<2x32xf16>
          %275 = vector.insert %274, %272 [0, 3] : f16 into vector<1x4xf16>
          %276 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %277 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_33 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %278 = vector.extract %182[%276, %277] : f16 from vector<32x2xf16>
          %279 = vector.insert %278, %cst_33 [0, 0] : f16 into vector<4x1xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %281 = vector.extract %182[%280, %277] : f16 from vector<32x2xf16>
          %282 = vector.insert %281, %279 [1, 0] : f16 into vector<4x1xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %284 = vector.extract %182[%283, %277] : f16 from vector<32x2xf16>
          %285 = vector.insert %284, %282 [2, 0] : f16 into vector<4x1xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %287 = vector.extract %182[%286, %277] : f16 from vector<32x2xf16>
          %288 = vector.insert %287, %285 [3, 0] : f16 into vector<4x1xf16>
          %289 = frisk.warp_mma_rr %275, %288, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %289 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %194 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %195 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %196 = vector.extract %193[0, 0] : f32 from vector<1x4xf32>
        %197 = vector.insert %196, %arg12 [%194, %195] : f32 into vector<2x8xf32>
        %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %199 = vector.extract %193[0, 1] : f32 from vector<1x4xf32>
        %200 = vector.insert %199, %197 [%194, %198] : f32 into vector<2x8xf32>
        %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %202 = vector.extract %193[0, 2] : f32 from vector<1x4xf32>
        %203 = vector.insert %202, %200 [%194, %201] : f32 into vector<2x8xf32>
        %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %205 = vector.extract %193[0, 3] : f32 from vector<1x4xf32>
        %206 = vector.insert %205, %203 [%194, %204] : f32 into vector<2x8xf32>
        %207 = vector.extract %174#1[%192, 0] : f32 from vector<2x1xf32>
        %208 = vector.broadcast %207 : f32 to vector<1x4xf32>
        %209 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %211 = vector.extract %208[0, 0] : f32 from vector<1x4xf32>
        %212 = vector.insert %211, %arg13 [%209, %210] : f32 into vector<2x8xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %214 = vector.extract %208[0, 1] : f32 from vector<1x4xf32>
        %215 = vector.insert %214, %212 [%209, %213] : f32 into vector<2x8xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %217 = vector.extract %208[0, 2] : f32 from vector<1x4xf32>
        %218 = vector.insert %217, %215 [%209, %216] : f32 into vector<2x8xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %220 = vector.extract %208[0, 3] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %218 [%209, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %224 = vector.extract %arg8[%222, %223] : f32 from vector<2x8xf32>
        %225 = vector.insert %224, %cst_31 [0, 0] : f32 into vector<1x4xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %227 = vector.extract %arg8[%222, %226] : f32 from vector<2x8xf32>
        %228 = vector.insert %227, %225 [0, 1] : f32 into vector<1x4xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %230 = vector.extract %arg8[%222, %229] : f32 from vector<2x8xf32>
        %231 = vector.insert %230, %228 [0, 2] : f32 into vector<1x4xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %233 = vector.extract %arg8[%222, %232] : f32 from vector<2x8xf32>
        %234 = vector.insert %233, %231 [0, 3] : f32 into vector<1x4xf32>
        %235 = arith.mulf %234, %208 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %238 = vector.extract %235[0, 0] : f32 from vector<1x4xf32>
        %239 = vector.insert %238, %arg14 [%236, %237] : f32 into vector<2x8xf32>
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %241 = vector.extract %235[0, 1] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %239 [%236, %240] : f32 into vector<2x8xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %244 = vector.extract %235[0, 2] : f32 from vector<1x4xf32>
        %245 = vector.insert %244, %242 [%236, %243] : f32 into vector<2x8xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %247 = vector.extract %235[0, 3] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %245 [%236, %246] : f32 into vector<2x8xf32>
        %249 = arith.addf %235, %193 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %252 = vector.extract %249[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg15 [%250, %251] : f32 into vector<2x8xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %255 = vector.extract %249[0, 1] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %253 [%250, %254] : f32 into vector<2x8xf32>
        %257 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %258 = vector.extract %249[0, 2] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %256 [%250, %257] : f32 into vector<2x8xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %261 = vector.extract %249[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%250, %260] : f32 into vector<2x8xf32>
        affine.yield %206, %221, %248, %262 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_29 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %184 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %184 to %alloc_29 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_29 : memref<128x32xf16, 3>
      %185 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %192 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_27]
        %193 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_27]
        %194 = frisk.read_matrix_from_shm %alloc_29[%192, %193] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %195 = vector.bitcast %194 : vector<4xi32> to vector<8xf16>
        %196 = vector.extract %195[0] : f16 from vector<8xf16>
        %197 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_27]
        %198 = vector.extract %195[%197] : f16 from vector<8xf16>
        %199 = vector.extract %195[1] : f16 from vector<8xf16>
        %200 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_27]
        %201 = vector.extract %195[%200] : f16 from vector<8xf16>
        %202 = vector.extract %195[2] : f16 from vector<8xf16>
        %203 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_27]
        %204 = vector.extract %195[%203] : f16 from vector<8xf16>
        %205 = vector.extract %195[3] : f16 from vector<8xf16>
        %206 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_27]
        %207 = vector.extract %195[%206] : f16 from vector<8xf16>
        %208 = vector.from_elements %196, %198, %199, %201, %202, %204, %205, %207 : vector<4x2xf16>
        %209 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %210 = affine.apply affine_map<() -> (0)>()
        %211 = vector.extract %208[0, 0] : f16 from vector<4x2xf16>
        %212 = vector.insert %211, %arg12 [%209, %210] : f16 into vector<32x2xf16>
        %213 = affine.apply affine_map<() -> (1)>()
        %214 = vector.extract %208[0, 1] : f16 from vector<4x2xf16>
        %215 = vector.insert %214, %212 [%209, %213] : f16 into vector<32x2xf16>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %217 = vector.extract %208[1, 0] : f16 from vector<4x2xf16>
        %218 = vector.insert %217, %215 [%216, %210] : f16 into vector<32x2xf16>
        %219 = affine.apply affine_map<() -> (1)>()
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %221 = vector.extract %208[1, 1] : f16 from vector<4x2xf16>
        %222 = vector.insert %221, %218 [%220, %219] : f16 into vector<32x2xf16>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %224 = vector.extract %208[2, 0] : f16 from vector<4x2xf16>
        %225 = vector.insert %224, %222 [%223, %210] : f16 into vector<32x2xf16>
        %226 = affine.apply affine_map<() -> (1)>()
        %227 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %228 = vector.extract %208[2, 1] : f16 from vector<4x2xf16>
        %229 = vector.insert %228, %225 [%227, %226] : f16 into vector<32x2xf16>
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %231 = vector.extract %208[3, 0] : f16 from vector<4x2xf16>
        %232 = vector.insert %231, %229 [%230, %210] : f16 into vector<32x2xf16>
        %233 = affine.apply affine_map<() -> (1)>()
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %235 = vector.extract %208[3, 1] : f16 from vector<4x2xf16>
        %236 = vector.insert %235, %232 [%234, %233] : f16 into vector<32x2xf16>
        affine.yield %236 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %186:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %193 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_4) -> (vector<1x4xf32>) {
          %263 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %264 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %cst_32 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %265 = vector.extract %177[%263, %264] : f16 from vector<2x32xf16>
          %266 = vector.insert %265, %cst_32 [0, 0] : f16 into vector<1x4xf16>
          %267 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %268 = vector.extract %177[%263, %267] : f16 from vector<2x32xf16>
          %269 = vector.insert %268, %266 [0, 1] : f16 into vector<1x4xf16>
          %270 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %271 = vector.extract %177[%263, %270] : f16 from vector<2x32xf16>
          %272 = vector.insert %271, %269 [0, 2] : f16 into vector<1x4xf16>
          %273 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %274 = vector.extract %177[%263, %273] : f16 from vector<2x32xf16>
          %275 = vector.insert %274, %272 [0, 3] : f16 into vector<1x4xf16>
          %276 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %277 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_33 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %278 = vector.extract %185[%276, %277] : f16 from vector<32x2xf16>
          %279 = vector.insert %278, %cst_33 [0, 0] : f16 into vector<4x1xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %281 = vector.extract %185[%280, %277] : f16 from vector<32x2xf16>
          %282 = vector.insert %281, %279 [1, 0] : f16 into vector<4x1xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %284 = vector.extract %185[%283, %277] : f16 from vector<32x2xf16>
          %285 = vector.insert %284, %282 [2, 0] : f16 into vector<4x1xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %287 = vector.extract %185[%286, %277] : f16 from vector<32x2xf16>
          %288 = vector.insert %287, %285 [3, 0] : f16 into vector<4x1xf16>
          %289 = frisk.warp_mma_rr %275, %288, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %289 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %194 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %195 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %196 = vector.extract %193[0, 0] : f32 from vector<1x4xf32>
        %197 = vector.insert %196, %arg12 [%194, %195] : f32 into vector<2x8xf32>
        %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %199 = vector.extract %193[0, 1] : f32 from vector<1x4xf32>
        %200 = vector.insert %199, %197 [%194, %198] : f32 into vector<2x8xf32>
        %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %202 = vector.extract %193[0, 2] : f32 from vector<1x4xf32>
        %203 = vector.insert %202, %200 [%194, %201] : f32 into vector<2x8xf32>
        %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %205 = vector.extract %193[0, 3] : f32 from vector<1x4xf32>
        %206 = vector.insert %205, %203 [%194, %204] : f32 into vector<2x8xf32>
        %207 = vector.extract %174#1[%192, 0] : f32 from vector<2x1xf32>
        %208 = vector.broadcast %207 : f32 to vector<1x4xf32>
        %209 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %211 = vector.extract %208[0, 0] : f32 from vector<1x4xf32>
        %212 = vector.insert %211, %arg13 [%209, %210] : f32 into vector<2x8xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %214 = vector.extract %208[0, 1] : f32 from vector<1x4xf32>
        %215 = vector.insert %214, %212 [%209, %213] : f32 into vector<2x8xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %217 = vector.extract %208[0, 2] : f32 from vector<1x4xf32>
        %218 = vector.insert %217, %215 [%209, %216] : f32 into vector<2x8xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %220 = vector.extract %208[0, 3] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %218 [%209, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %224 = vector.extract %arg9[%222, %223] : f32 from vector<2x8xf32>
        %225 = vector.insert %224, %cst_31 [0, 0] : f32 into vector<1x4xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %227 = vector.extract %arg9[%222, %226] : f32 from vector<2x8xf32>
        %228 = vector.insert %227, %225 [0, 1] : f32 into vector<1x4xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %230 = vector.extract %arg9[%222, %229] : f32 from vector<2x8xf32>
        %231 = vector.insert %230, %228 [0, 2] : f32 into vector<1x4xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %233 = vector.extract %arg9[%222, %232] : f32 from vector<2x8xf32>
        %234 = vector.insert %233, %231 [0, 3] : f32 into vector<1x4xf32>
        %235 = arith.mulf %234, %208 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %238 = vector.extract %235[0, 0] : f32 from vector<1x4xf32>
        %239 = vector.insert %238, %arg14 [%236, %237] : f32 into vector<2x8xf32>
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %241 = vector.extract %235[0, 1] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %239 [%236, %240] : f32 into vector<2x8xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %244 = vector.extract %235[0, 2] : f32 from vector<1x4xf32>
        %245 = vector.insert %244, %242 [%236, %243] : f32 into vector<2x8xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %247 = vector.extract %235[0, 3] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %245 [%236, %246] : f32 into vector<2x8xf32>
        %249 = arith.addf %235, %193 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %252 = vector.extract %249[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg15 [%250, %251] : f32 into vector<2x8xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %255 = vector.extract %249[0, 1] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %253 [%250, %254] : f32 into vector<2x8xf32>
        %257 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %258 = vector.extract %249[0, 2] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %256 [%250, %257] : f32 into vector<2x8xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %261 = vector.extract %249[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%250, %260] : f32 into vector<2x8xf32>
        affine.yield %206, %221, %248, %262 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %alloc_30 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %187 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %187 to %alloc_30 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_30 : memref<128x32xf16, 3>
      %188 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_1) -> (vector<32x2xf16>) {
        %192 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_27]
        %193 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_27]
        %194 = frisk.read_matrix_from_shm %alloc_30[%192, %193] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %195 = vector.bitcast %194 : vector<4xi32> to vector<8xf16>
        %196 = vector.extract %195[0] : f16 from vector<8xf16>
        %197 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_27]
        %198 = vector.extract %195[%197] : f16 from vector<8xf16>
        %199 = vector.extract %195[1] : f16 from vector<8xf16>
        %200 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_27]
        %201 = vector.extract %195[%200] : f16 from vector<8xf16>
        %202 = vector.extract %195[2] : f16 from vector<8xf16>
        %203 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_27]
        %204 = vector.extract %195[%203] : f16 from vector<8xf16>
        %205 = vector.extract %195[3] : f16 from vector<8xf16>
        %206 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_27]
        %207 = vector.extract %195[%206] : f16 from vector<8xf16>
        %208 = vector.from_elements %196, %198, %199, %201, %202, %204, %205, %207 : vector<4x2xf16>
        %209 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %210 = affine.apply affine_map<() -> (0)>()
        %211 = vector.extract %208[0, 0] : f16 from vector<4x2xf16>
        %212 = vector.insert %211, %arg12 [%209, %210] : f16 into vector<32x2xf16>
        %213 = affine.apply affine_map<() -> (1)>()
        %214 = vector.extract %208[0, 1] : f16 from vector<4x2xf16>
        %215 = vector.insert %214, %212 [%209, %213] : f16 into vector<32x2xf16>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %217 = vector.extract %208[1, 0] : f16 from vector<4x2xf16>
        %218 = vector.insert %217, %215 [%216, %210] : f16 into vector<32x2xf16>
        %219 = affine.apply affine_map<() -> (1)>()
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %221 = vector.extract %208[1, 1] : f16 from vector<4x2xf16>
        %222 = vector.insert %221, %218 [%220, %219] : f16 into vector<32x2xf16>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %224 = vector.extract %208[2, 0] : f16 from vector<4x2xf16>
        %225 = vector.insert %224, %222 [%223, %210] : f16 into vector<32x2xf16>
        %226 = affine.apply affine_map<() -> (1)>()
        %227 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %228 = vector.extract %208[2, 1] : f16 from vector<4x2xf16>
        %229 = vector.insert %228, %225 [%227, %226] : f16 into vector<32x2xf16>
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %231 = vector.extract %208[3, 0] : f16 from vector<4x2xf16>
        %232 = vector.insert %231, %229 [%230, %210] : f16 into vector<32x2xf16>
        %233 = affine.apply affine_map<() -> (1)>()
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %235 = vector.extract %208[3, 1] : f16 from vector<4x2xf16>
        %236 = vector.insert %235, %232 [%234, %233] : f16 into vector<32x2xf16>
        affine.yield %236 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %189:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %193 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_4) -> (vector<1x4xf32>) {
          %289 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %290 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %291 = vector.extract %177[%289, %290] : f16 from vector<2x32xf16>
          %292 = vector.insert %291, %cst_33 [0, 0] : f16 into vector<1x4xf16>
          %293 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %294 = vector.extract %177[%289, %293] : f16 from vector<2x32xf16>
          %295 = vector.insert %294, %292 [0, 1] : f16 into vector<1x4xf16>
          %296 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %297 = vector.extract %177[%289, %296] : f16 from vector<2x32xf16>
          %298 = vector.insert %297, %295 [0, 2] : f16 into vector<1x4xf16>
          %299 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %300 = vector.extract %177[%289, %299] : f16 from vector<2x32xf16>
          %301 = vector.insert %300, %298 [0, 3] : f16 into vector<1x4xf16>
          %302 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %303 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_34 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %304 = vector.extract %188[%302, %303] : f16 from vector<32x2xf16>
          %305 = vector.insert %304, %cst_34 [0, 0] : f16 into vector<4x1xf16>
          %306 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %307 = vector.extract %188[%306, %303] : f16 from vector<32x2xf16>
          %308 = vector.insert %307, %305 [1, 0] : f16 into vector<4x1xf16>
          %309 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %310 = vector.extract %188[%309, %303] : f16 from vector<32x2xf16>
          %311 = vector.insert %310, %308 [2, 0] : f16 into vector<4x1xf16>
          %312 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %313 = vector.extract %188[%312, %303] : f16 from vector<32x2xf16>
          %314 = vector.insert %313, %311 [3, 0] : f16 into vector<4x1xf16>
          %315 = frisk.warp_mma_rr %301, %314, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %315 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %194 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %195 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %196 = vector.extract %193[0, 0] : f32 from vector<1x4xf32>
        %197 = vector.insert %196, %arg12 [%194, %195] : f32 into vector<2x8xf32>
        %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %199 = vector.extract %193[0, 1] : f32 from vector<1x4xf32>
        %200 = vector.insert %199, %197 [%194, %198] : f32 into vector<2x8xf32>
        %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %202 = vector.extract %193[0, 2] : f32 from vector<1x4xf32>
        %203 = vector.insert %202, %200 [%194, %201] : f32 into vector<2x8xf32>
        %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %205 = vector.extract %193[0, 3] : f32 from vector<1x4xf32>
        %206 = vector.insert %205, %203 [%194, %204] : f32 into vector<2x8xf32>
        %207 = vector.extract %174#1[%192, 0] : f32 from vector<2x1xf32>
        %208 = vector.broadcast %207 : f32 to vector<1x4xf32>
        %209 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %211 = vector.extract %208[0, 0] : f32 from vector<1x4xf32>
        %212 = vector.insert %211, %arg13 [%209, %210] : f32 into vector<2x8xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %214 = vector.extract %208[0, 1] : f32 from vector<1x4xf32>
        %215 = vector.insert %214, %212 [%209, %213] : f32 into vector<2x8xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %217 = vector.extract %208[0, 2] : f32 from vector<1x4xf32>
        %218 = vector.insert %217, %215 [%209, %216] : f32 into vector<2x8xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %220 = vector.extract %208[0, 3] : f32 from vector<1x4xf32>
        %221 = vector.insert %220, %218 [%209, %219] : f32 into vector<2x8xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %224 = vector.extract %arg10[%222, %223] : f32 from vector<2x8xf32>
        %225 = vector.insert %224, %cst_31 [0, 0] : f32 into vector<1x4xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %227 = vector.extract %arg10[%222, %226] : f32 from vector<2x8xf32>
        %228 = vector.insert %227, %225 [0, 1] : f32 into vector<1x4xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %230 = vector.extract %arg10[%222, %229] : f32 from vector<2x8xf32>
        %231 = vector.insert %230, %228 [0, 2] : f32 into vector<1x4xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %233 = vector.extract %arg10[%222, %232] : f32 from vector<2x8xf32>
        %234 = vector.insert %233, %231 [0, 3] : f32 into vector<1x4xf32>
        %235 = arith.mulf %234, %208 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %238 = vector.extract %235[0, 0] : f32 from vector<1x4xf32>
        %239 = vector.insert %238, %arg14 [%236, %237] : f32 into vector<2x8xf32>
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %241 = vector.extract %235[0, 1] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %239 [%236, %240] : f32 into vector<2x8xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %244 = vector.extract %235[0, 2] : f32 from vector<1x4xf32>
        %245 = vector.insert %244, %242 [%236, %243] : f32 into vector<2x8xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %247 = vector.extract %235[0, 3] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %245 [%236, %246] : f32 into vector<2x8xf32>
        %249 = arith.addf %235, %193 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %252 = vector.extract %249[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg15 [%250, %251] : f32 into vector<2x8xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %255 = vector.extract %249[0, 1] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %253 [%250, %254] : f32 into vector<2x8xf32>
        %257 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %258 = vector.extract %249[0, 2] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %256 [%250, %257] : f32 into vector<2x8xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %261 = vector.extract %249[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%250, %260] : f32 into vector<2x8xf32>
        %263 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %264 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_32 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %265 = vector.extract %180#3[%263, %264] : f32 from vector<2x8xf32>
        %266 = vector.insert %265, %cst_32 [0, 0] : f32 into vector<1x4xf32>
        %267 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %268 = vector.extract %180#3[%263, %267] : f32 from vector<2x8xf32>
        %269 = vector.insert %268, %266 [0, 1] : f32 into vector<1x4xf32>
        %270 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %271 = vector.extract %180#3[%263, %270] : f32 from vector<2x8xf32>
        %272 = vector.insert %271, %269 [0, 2] : f32 into vector<1x4xf32>
        %273 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %274 = vector.extract %180#3[%263, %273] : f32 from vector<2x8xf32>
        %275 = vector.insert %274, %272 [0, 3] : f32 into vector<1x4xf32>
        %276 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %277 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %278 = vector.extract %275[0, 0] : f32 from vector<1x4xf32>
        %279 = vector.insert %278, %arg16 [%276, %277] : f32 into vector<2x8xf32>
        %280 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %281 = vector.extract %275[0, 1] : f32 from vector<1x4xf32>
        %282 = vector.insert %281, %279 [%276, %280] : f32 into vector<2x8xf32>
        %283 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %284 = vector.extract %275[0, 2] : f32 from vector<1x4xf32>
        %285 = vector.insert %284, %282 [%276, %283] : f32 into vector<2x8xf32>
        %286 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %287 = vector.extract %275[0, 3] : f32 from vector<1x4xf32>
        %288 = vector.insert %287, %285 [%276, %286] : f32 into vector<2x8xf32>
        affine.yield %206, %221, %248, %262, %288 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
      %190:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_11, %arg13 = %cst_11) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %193 = affine.apply affine_map<() -> (0)>()
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %194 = vector.extract %176#1[%192, %193] : f32 from vector<2x1xf32>
        %195 = vector.insert %194, %cst_31 [0, 0] : f32 into vector<1x1xf32>
        %196 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %197 = affine.apply affine_map<() -> (0)>()
        %198 = vector.extract %195[0, 0] : f32 from vector<1x1xf32>
        %199 = vector.insert %198, %arg12 [%196, %197] : f32 into vector<2x1xf32>
        %200 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %201 = affine.apply affine_map<() -> (0)>()
        %cst_32 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %202 = vector.extract %172#1[%200, %201] : f32 from vector<2x1xf32>
        %203 = vector.insert %202, %cst_32 [0, 0] : f32 into vector<1x1xf32>
        %204 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %205 = affine.apply affine_map<() -> (0)>()
        %206 = vector.extract %203[0, 0] : f32 from vector<1x1xf32>
        %207 = vector.insert %206, %arg13 [%204, %205] : f32 into vector<2x1xf32>
        affine.yield %199, %207 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %191:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_31 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %194 = vector.extract %183#3[%192, %193] : f32 from vector<2x8xf32>
        %195 = vector.insert %194, %cst_31 [0, 0] : f32 into vector<1x4xf32>
        %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %197 = vector.extract %183#3[%192, %196] : f32 from vector<2x8xf32>
        %198 = vector.insert %197, %195 [0, 1] : f32 into vector<1x4xf32>
        %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %200 = vector.extract %183#3[%192, %199] : f32 from vector<2x8xf32>
        %201 = vector.insert %200, %198 [0, 2] : f32 into vector<1x4xf32>
        %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %203 = vector.extract %183#3[%192, %202] : f32 from vector<2x8xf32>
        %204 = vector.insert %203, %201 [0, 3] : f32 into vector<1x4xf32>
        %205 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %207 = vector.extract %204[0, 0] : f32 from vector<1x4xf32>
        %208 = vector.insert %207, %arg12 [%205, %206] : f32 into vector<2x8xf32>
        %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %210 = vector.extract %204[0, 1] : f32 from vector<1x4xf32>
        %211 = vector.insert %210, %208 [%205, %209] : f32 into vector<2x8xf32>
        %212 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %213 = vector.extract %204[0, 2] : f32 from vector<1x4xf32>
        %214 = vector.insert %213, %211 [%205, %212] : f32 into vector<2x8xf32>
        %215 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %216 = vector.extract %204[0, 3] : f32 from vector<1x4xf32>
        %217 = vector.insert %216, %214 [%205, %215] : f32 into vector<2x8xf32>
        %218 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_32 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %220 = vector.extract %186#3[%218, %219] : f32 from vector<2x8xf32>
        %221 = vector.insert %220, %cst_32 [0, 0] : f32 into vector<1x4xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %223 = vector.extract %186#3[%218, %222] : f32 from vector<2x8xf32>
        %224 = vector.insert %223, %221 [0, 1] : f32 into vector<1x4xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %226 = vector.extract %186#3[%218, %225] : f32 from vector<2x8xf32>
        %227 = vector.insert %226, %224 [0, 2] : f32 into vector<1x4xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %229 = vector.extract %186#3[%218, %228] : f32 from vector<2x8xf32>
        %230 = vector.insert %229, %227 [0, 3] : f32 into vector<1x4xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %233 = vector.extract %230[0, 0] : f32 from vector<1x4xf32>
        %234 = vector.insert %233, %arg13 [%231, %232] : f32 into vector<2x8xf32>
        %235 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %236 = vector.extract %230[0, 1] : f32 from vector<1x4xf32>
        %237 = vector.insert %236, %234 [%231, %235] : f32 into vector<2x8xf32>
        %238 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %239 = vector.extract %230[0, 2] : f32 from vector<1x4xf32>
        %240 = vector.insert %239, %237 [%231, %238] : f32 into vector<2x8xf32>
        %241 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %242 = vector.extract %230[0, 3] : f32 from vector<1x4xf32>
        %243 = vector.insert %242, %240 [%231, %241] : f32 into vector<2x8xf32>
        %244 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %246 = vector.extract %189#3[%244, %245] : f32 from vector<2x8xf32>
        %247 = vector.insert %246, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %249 = vector.extract %189#3[%244, %248] : f32 from vector<2x8xf32>
        %250 = vector.insert %249, %247 [0, 1] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %252 = vector.extract %189#3[%244, %251] : f32 from vector<2x8xf32>
        %253 = vector.insert %252, %250 [0, 2] : f32 into vector<1x4xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %255 = vector.extract %189#3[%244, %254] : f32 from vector<2x8xf32>
        %256 = vector.insert %255, %253 [0, 3] : f32 into vector<1x4xf32>
        %257 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %258 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %259 = vector.extract %256[0, 0] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %arg14 [%257, %258] : f32 into vector<2x8xf32>
        %261 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %262 = vector.extract %256[0, 1] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %260 [%257, %261] : f32 into vector<2x8xf32>
        %264 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %265 = vector.extract %256[0, 2] : f32 from vector<1x4xf32>
        %266 = vector.insert %265, %263 [%257, %264] : f32 into vector<2x8xf32>
        %267 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %268 = vector.extract %256[0, 3] : f32 from vector<1x4xf32>
        %269 = vector.insert %268, %266 [%257, %267] : f32 into vector<2x8xf32>
        affine.yield %217, %243, %269 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %189#4, %190#0, %190#1, %191#0, %191#1, %191#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %thread_id_x = gpu.thread_id x
    %147 = arith.index_cast %thread_id_x : index to i32
    %148 = arith.andi %147, %c16_i32 : i32
    %149 = arith.cmpi ne, %148, %c0_i32 : i32
    %150 = arith.andi %147, %c32_i32 : i32
    %151 = arith.cmpi ne, %150, %c0_i32 : i32
    %152 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %153:8 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_0, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_6, %arg12 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %162 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %163 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %165 = vector.extract %146#5[%163, %164] : f32 from vector<2x8xf32>
      %166 = vector.insert %165, %cst_22 [0, 0] : f32 into vector<1x4xf32>
      %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %168 = vector.extract %146#5[%163, %167] : f32 from vector<2x8xf32>
      %169 = vector.insert %168, %166 [0, 1] : f32 into vector<1x4xf32>
      %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %171 = vector.extract %146#5[%163, %170] : f32 from vector<2x8xf32>
      %172 = vector.insert %171, %169 [0, 2] : f32 into vector<1x4xf32>
      %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %174 = vector.extract %146#5[%163, %173] : f32 from vector<2x8xf32>
      %175 = vector.insert %174, %172 [0, 3] : f32 into vector<1x4xf32>
      %176 = vector.extract %175[0, 0] : f32 from vector<1x4xf32>
      %177 = vector.extract %175[0, 1] : f32 from vector<1x4xf32>
      %178 = vector.extract %175[0, 2] : f32 from vector<1x4xf32>
      %179 = vector.extract %175[0, 3] : f32 from vector<1x4xf32>
      %180 = arith.select %149, %176, %177 : f32
      %shuffleResult, %valid = gpu.shuffle xor %180, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %181 = arith.select %149, %shuffleResult, %176 : f32
      %182 = arith.select %149, %177, %shuffleResult : f32
      %183 = arith.select %149, %178, %179 : f32
      %shuffleResult_23, %valid_24 = gpu.shuffle xor %183, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %184 = arith.select %149, %shuffleResult_23, %178 : f32
      %185 = arith.select %149, %179, %shuffleResult_23 : f32
      %186 = arith.select %151, %181, %184 : f32
      %shuffleResult_25, %valid_26 = gpu.shuffle xor %186, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %187 = arith.select %151, %shuffleResult_25, %181 : f32
      %188 = arith.select %151, %184, %shuffleResult_25 : f32
      %189 = arith.select %151, %182, %185 : f32
      %shuffleResult_27, %valid_28 = gpu.shuffle xor %189, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %190 = arith.select %151, %shuffleResult_27, %182 : f32
      %191 = arith.select %151, %185, %shuffleResult_27 : f32
      %192 = vector.from_elements %187, %190, %188, %191 : vector<1x4xf32>
      %193 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %195 = vector.extract %192[0, 0] : f32 from vector<1x4xf32>
      %196 = vector.insert %195, %arg5 [%193, %194] : f32 into vector<2x8xf32>
      %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %198 = vector.extract %192[0, 1] : f32 from vector<1x4xf32>
      %199 = vector.insert %198, %196 [%193, %197] : f32 into vector<2x8xf32>
      %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %201 = vector.extract %192[0, 2] : f32 from vector<1x4xf32>
      %202 = vector.insert %201, %199 [%193, %200] : f32 into vector<2x8xf32>
      %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %204 = vector.extract %192[0, 3] : f32 from vector<1x4xf32>
      %205 = vector.insert %204, %202 [%193, %203] : f32 into vector<2x8xf32>
      %206 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_29 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %208 = vector.extract %146#4[%206, %207] : f32 from vector<2x8xf32>
      %209 = vector.insert %208, %cst_29 [0, 0] : f32 into vector<1x4xf32>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %211 = vector.extract %146#4[%206, %210] : f32 from vector<2x8xf32>
      %212 = vector.insert %211, %209 [0, 1] : f32 into vector<1x4xf32>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %214 = vector.extract %146#4[%206, %213] : f32 from vector<2x8xf32>
      %215 = vector.insert %214, %212 [0, 2] : f32 into vector<1x4xf32>
      %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %217 = vector.extract %146#4[%206, %216] : f32 from vector<2x8xf32>
      %218 = vector.insert %217, %215 [0, 3] : f32 into vector<1x4xf32>
      %219 = vector.extract %218[0, 0] : f32 from vector<1x4xf32>
      %220 = vector.extract %218[0, 1] : f32 from vector<1x4xf32>
      %221 = vector.extract %218[0, 2] : f32 from vector<1x4xf32>
      %222 = vector.extract %218[0, 3] : f32 from vector<1x4xf32>
      %223 = arith.select %149, %219, %220 : f32
      %shuffleResult_30, %valid_31 = gpu.shuffle xor %223, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %224 = arith.select %149, %shuffleResult_30, %219 : f32
      %225 = arith.select %149, %220, %shuffleResult_30 : f32
      %226 = arith.select %149, %221, %222 : f32
      %shuffleResult_32, %valid_33 = gpu.shuffle xor %226, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %227 = arith.select %149, %shuffleResult_32, %221 : f32
      %228 = arith.select %149, %222, %shuffleResult_32 : f32
      %229 = arith.select %151, %224, %227 : f32
      %shuffleResult_34, %valid_35 = gpu.shuffle xor %229, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %230 = arith.select %151, %shuffleResult_34, %224 : f32
      %231 = arith.select %151, %227, %shuffleResult_34 : f32
      %232 = arith.select %151, %225, %228 : f32
      %shuffleResult_36, %valid_37 = gpu.shuffle xor %232, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %233 = arith.select %151, %shuffleResult_36, %225 : f32
      %234 = arith.select %151, %228, %shuffleResult_36 : f32
      %235 = vector.from_elements %230, %233, %231, %234 : vector<1x4xf32>
      %236 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %238 = vector.extract %235[0, 0] : f32 from vector<1x4xf32>
      %239 = vector.insert %238, %arg6 [%236, %237] : f32 into vector<2x8xf32>
      %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %241 = vector.extract %235[0, 1] : f32 from vector<1x4xf32>
      %242 = vector.insert %241, %239 [%236, %240] : f32 into vector<2x8xf32>
      %243 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %244 = vector.extract %235[0, 2] : f32 from vector<1x4xf32>
      %245 = vector.insert %244, %242 [%236, %243] : f32 into vector<2x8xf32>
      %246 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %247 = vector.extract %235[0, 3] : f32 from vector<1x4xf32>
      %248 = vector.insert %247, %245 [%236, %246] : f32 into vector<2x8xf32>
      %249 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %250 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_38 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %251 = vector.extract %146#3[%249, %250] : f32 from vector<2x8xf32>
      %252 = vector.insert %251, %cst_38 [0, 0] : f32 into vector<1x4xf32>
      %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %254 = vector.extract %146#3[%249, %253] : f32 from vector<2x8xf32>
      %255 = vector.insert %254, %252 [0, 1] : f32 into vector<1x4xf32>
      %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %257 = vector.extract %146#3[%249, %256] : f32 from vector<2x8xf32>
      %258 = vector.insert %257, %255 [0, 2] : f32 into vector<1x4xf32>
      %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %260 = vector.extract %146#3[%249, %259] : f32 from vector<2x8xf32>
      %261 = vector.insert %260, %258 [0, 3] : f32 into vector<1x4xf32>
      %262 = vector.extract %261[0, 0] : f32 from vector<1x4xf32>
      %263 = vector.extract %261[0, 1] : f32 from vector<1x4xf32>
      %264 = vector.extract %261[0, 2] : f32 from vector<1x4xf32>
      %265 = vector.extract %261[0, 3] : f32 from vector<1x4xf32>
      %266 = arith.select %149, %262, %263 : f32
      %shuffleResult_39, %valid_40 = gpu.shuffle xor %266, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %267 = arith.select %149, %shuffleResult_39, %262 : f32
      %268 = arith.select %149, %263, %shuffleResult_39 : f32
      %269 = arith.select %149, %264, %265 : f32
      %shuffleResult_41, %valid_42 = gpu.shuffle xor %269, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %270 = arith.select %149, %shuffleResult_41, %264 : f32
      %271 = arith.select %149, %265, %shuffleResult_41 : f32
      %272 = arith.select %151, %267, %270 : f32
      %shuffleResult_43, %valid_44 = gpu.shuffle xor %272, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %273 = arith.select %151, %shuffleResult_43, %267 : f32
      %274 = arith.select %151, %270, %shuffleResult_43 : f32
      %275 = arith.select %151, %268, %271 : f32
      %shuffleResult_45, %valid_46 = gpu.shuffle xor %275, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %276 = arith.select %151, %shuffleResult_45, %268 : f32
      %277 = arith.select %151, %271, %shuffleResult_45 : f32
      %278 = vector.from_elements %273, %276, %274, %277 : vector<1x4xf32>
      %279 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %280 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %281 = vector.extract %278[0, 0] : f32 from vector<1x4xf32>
      %282 = vector.insert %281, %arg7 [%279, %280] : f32 into vector<2x8xf32>
      %283 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %284 = vector.extract %278[0, 1] : f32 from vector<1x4xf32>
      %285 = vector.insert %284, %282 [%279, %283] : f32 into vector<2x8xf32>
      %286 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %287 = vector.extract %278[0, 2] : f32 from vector<1x4xf32>
      %288 = vector.insert %287, %285 [%279, %286] : f32 into vector<2x8xf32>
      %289 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %290 = vector.extract %278[0, 3] : f32 from vector<1x4xf32>
      %291 = vector.insert %290, %288 [%279, %289] : f32 into vector<2x8xf32>
      %292 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %293 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %294 = vector.extract %146#0[%292, %293] : f32 from vector<2x8xf32>
      %295 = vector.insert %294, %cst_47 [0, 0] : f32 into vector<1x4xf32>
      %296 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %297 = vector.extract %146#0[%292, %296] : f32 from vector<2x8xf32>
      %298 = vector.insert %297, %295 [0, 1] : f32 into vector<1x4xf32>
      %299 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %300 = vector.extract %146#0[%292, %299] : f32 from vector<2x8xf32>
      %301 = vector.insert %300, %298 [0, 2] : f32 into vector<1x4xf32>
      %302 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %303 = vector.extract %146#0[%292, %302] : f32 from vector<2x8xf32>
      %304 = vector.insert %303, %301 [0, 3] : f32 into vector<1x4xf32>
      %305 = vector.extract %304[0, 0] : f32 from vector<1x4xf32>
      %306 = vector.extract %304[0, 1] : f32 from vector<1x4xf32>
      %307 = vector.extract %304[0, 2] : f32 from vector<1x4xf32>
      %308 = vector.extract %304[0, 3] : f32 from vector<1x4xf32>
      %309 = arith.select %149, %305, %306 : f32
      %shuffleResult_48, %valid_49 = gpu.shuffle xor %309, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %310 = arith.select %149, %shuffleResult_48, %305 : f32
      %311 = arith.select %149, %306, %shuffleResult_48 : f32
      %312 = arith.select %149, %307, %308 : f32
      %shuffleResult_50, %valid_51 = gpu.shuffle xor %312, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %313 = arith.select %149, %shuffleResult_50, %307 : f32
      %314 = arith.select %149, %308, %shuffleResult_50 : f32
      %315 = arith.select %151, %310, %313 : f32
      %shuffleResult_52, %valid_53 = gpu.shuffle xor %315, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %316 = arith.select %151, %shuffleResult_52, %310 : f32
      %317 = arith.select %151, %313, %shuffleResult_52 : f32
      %318 = arith.select %151, %311, %314 : f32
      %shuffleResult_54, %valid_55 = gpu.shuffle xor %318, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %319 = arith.select %151, %shuffleResult_54, %311 : f32
      %320 = arith.select %151, %314, %shuffleResult_54 : f32
      %321 = vector.from_elements %316, %319, %317, %320 : vector<1x4xf32>
      %322 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %323 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %324 = vector.extract %321[0, 0] : f32 from vector<1x4xf32>
      %325 = vector.insert %324, %arg8 [%322, %323] : f32 into vector<2x8xf32>
      %326 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %327 = vector.extract %321[0, 1] : f32 from vector<1x4xf32>
      %328 = vector.insert %327, %325 [%322, %326] : f32 into vector<2x8xf32>
      %329 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %330 = vector.extract %321[0, 2] : f32 from vector<1x4xf32>
      %331 = vector.insert %330, %328 [%322, %329] : f32 into vector<2x8xf32>
      %332 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %333 = vector.extract %321[0, 3] : f32 from vector<1x4xf32>
      %334 = vector.insert %333, %331 [%322, %332] : f32 into vector<2x8xf32>
      %335 = vector.extract %146#1[%162, 0] : f32 from vector<2x1xf32>
      %336 = vector.broadcast %335 : f32 to vector<1x4xf32>
      %337 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %338 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %339 = vector.extract %336[0, 0] : f32 from vector<1x4xf32>
      %340 = vector.insert %339, %arg9 [%337, %338] : f32 into vector<2x8xf32>
      %341 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %342 = vector.extract %336[0, 1] : f32 from vector<1x4xf32>
      %343 = vector.insert %342, %340 [%337, %341] : f32 into vector<2x8xf32>
      %344 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %345 = vector.extract %336[0, 2] : f32 from vector<1x4xf32>
      %346 = vector.insert %345, %343 [%337, %344] : f32 into vector<2x8xf32>
      %347 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %348 = vector.extract %336[0, 3] : f32 from vector<1x4xf32>
      %349 = vector.insert %348, %346 [%337, %347] : f32 into vector<2x8xf32>
      %350 = arith.divf %321, %336 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %351 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %352 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %353 = vector.extract %350[0, 0] : f32 from vector<1x4xf32>
      %354 = vector.insert %353, %arg10 [%351, %352] : f32 into vector<2x8xf32>
      %355 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %356 = vector.extract %350[0, 1] : f32 from vector<1x4xf32>
      %357 = vector.insert %356, %354 [%351, %355] : f32 into vector<2x8xf32>
      %358 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %359 = vector.extract %350[0, 2] : f32 from vector<1x4xf32>
      %360 = vector.insert %359, %357 [%351, %358] : f32 into vector<2x8xf32>
      %361 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %362 = vector.extract %350[0, 3] : f32 from vector<1x4xf32>
      %363 = vector.insert %362, %360 [%351, %361] : f32 into vector<2x8xf32>
      %364 = arith.truncf %350 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %365 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %366 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %367 = vector.extract %364[0, 0] : f16 from vector<1x4xf16>
      %368 = vector.insert %367, %arg11 [%365, %366] : f16 into vector<2x8xf16>
      %369 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %370 = vector.extract %364[0, 1] : f16 from vector<1x4xf16>
      %371 = vector.insert %370, %368 [%365, %369] : f16 into vector<2x8xf16>
      %372 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %373 = vector.extract %364[0, 2] : f16 from vector<1x4xf16>
      %374 = vector.insert %373, %371 [%365, %372] : f16 into vector<2x8xf16>
      %375 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %376 = vector.extract %364[0, 3] : f16 from vector<1x4xf16>
      %377 = vector.insert %376, %374 [%365, %375] : f16 into vector<2x8xf16>
      %378 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %379 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %380 = vector.extract %364[0, 0] : f16 from vector<1x4xf16>
      %381 = vector.insert %380, %arg12 [%378, %379] : f16 into vector<2x8xf16>
      %382 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %383 = vector.extract %364[0, 1] : f16 from vector<1x4xf16>
      %384 = vector.insert %383, %381 [%378, %382] : f16 into vector<2x8xf16>
      %385 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %386 = vector.extract %364[0, 2] : f16 from vector<1x4xf16>
      %387 = vector.insert %386, %384 [%378, %385] : f16 into vector<2x8xf16>
      %388 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %389 = vector.extract %364[0, 3] : f16 from vector<1x4xf16>
      %390 = vector.insert %389, %387 [%378, %388] : f16 into vector<2x8xf16>
      affine.yield %205, %248, %291, %334, %349, %363, %377, %390 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    %154 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %155 = arith.cmpi ult, %154, %c4 : index
    scf.if %155 {
      affine.for %arg4 = 0 to 4 {
        %162 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %164 = vector.extract %153#7[%162, %163] : f16 from vector<2x8xf16>
        %165 = vector.insert %164, %cst_22 [0, 0] : f16 into vector<1x4xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %167 = vector.extract %153#7[%162, %166] : f16 from vector<2x8xf16>
        %168 = vector.insert %167, %165 [0, 1] : f16 into vector<1x4xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %170 = vector.extract %153#7[%162, %169] : f16 from vector<2x8xf16>
        %171 = vector.insert %170, %168 [0, 2] : f16 into vector<1x4xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %173 = vector.extract %153#7[%162, %172] : f16 from vector<2x8xf16>
        %174 = vector.insert %173, %171 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_23 = gpu.thread_id x
        %175 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %177 = vector.extract %174[0, 0] : f16 from vector<1x4xf16>
        %c0_24 = arith.constant 0 : index
        %c0_25 = arith.constant 0 : index
        %178 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%176)
        %c0_26 = arith.constant 0 : index
        %179 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%176)
        %c0_27 = arith.constant 0 : index
        %180 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_25, %179)
        %181 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %178, %c0_24, %c0_26, %c0_27, %180)
        %182 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %178, %c0_24, %c0_26, %c0_27, %180)
        affine.store %177, %152[%181, %182] {tiled = true} : memref<128x32xf16, 1>
        %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %184 = vector.extract %174[0, 1] : f16 from vector<1x4xf16>
        %c0_28 = arith.constant 0 : index
        %c0_29 = arith.constant 0 : index
        %185 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%183)
        %c0_30 = arith.constant 0 : index
        %186 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%183)
        %c0_31 = arith.constant 0 : index
        %187 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_29, %186)
        %188 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %185, %c0_28, %c0_30, %c0_31, %187)
        %189 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %185, %c0_28, %c0_30, %c0_31, %187)
        affine.store %184, %152[%188, %189] {tiled = true} : memref<128x32xf16, 1>
        %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %191 = vector.extract %174[0, 2] : f16 from vector<1x4xf16>
        %c0_32 = arith.constant 0 : index
        %c0_33 = arith.constant 0 : index
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%190)
        %c0_34 = arith.constant 0 : index
        %193 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%190)
        %c0_35 = arith.constant 0 : index
        %194 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_33, %193)
        %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %192, %c0_32, %c0_34, %c0_35, %194)
        %196 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %192, %c0_32, %c0_34, %c0_35, %194)
        affine.store %191, %152[%195, %196] {tiled = true} : memref<128x32xf16, 1>
        %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %198 = vector.extract %174[0, 3] : f16 from vector<1x4xf16>
        %c0_36 = arith.constant 0 : index
        %c0_37 = arith.constant 0 : index
        %199 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%197)
        %c0_38 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%197)
        %c0_39 = arith.constant 0 : index
        %201 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_37, %200)
        %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %199, %c0_36, %c0_38, %c0_39, %201)
        %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %199, %c0_36, %c0_38, %c0_39, %201)
        affine.store %198, %152[%202, %203] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %156 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %157:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %162 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %163 = vector.extract %146#1[%162, 0] : f32 from vector<2x1xf32>
      %164 = vector.broadcast %163 : f32 to vector<1x4xf32>
      %165 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %167 = vector.extract %164[0, 0] : f32 from vector<1x4xf32>
      %168 = vector.insert %167, %arg5 [%165, %166] : f32 into vector<2x8xf32>
      %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %170 = vector.extract %164[0, 1] : f32 from vector<1x4xf32>
      %171 = vector.insert %170, %168 [%165, %169] : f32 into vector<2x8xf32>
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %173 = vector.extract %164[0, 2] : f32 from vector<1x4xf32>
      %174 = vector.insert %173, %171 [%165, %172] : f32 into vector<2x8xf32>
      %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %176 = vector.extract %164[0, 3] : f32 from vector<1x4xf32>
      %177 = vector.insert %176, %174 [%165, %175] : f32 into vector<2x8xf32>
      %178 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %180 = vector.extract %153#2[%178, %179] : f32 from vector<2x8xf32>
      %181 = vector.insert %180, %cst_22 [0, 0] : f32 into vector<1x4xf32>
      %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %183 = vector.extract %153#2[%178, %182] : f32 from vector<2x8xf32>
      %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
      %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %186 = vector.extract %153#2[%178, %185] : f32 from vector<2x8xf32>
      %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
      %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %189 = vector.extract %153#2[%178, %188] : f32 from vector<2x8xf32>
      %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
      %191 = arith.divf %190, %164 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %192 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %194 = vector.extract %191[0, 0] : f32 from vector<1x4xf32>
      %195 = vector.insert %194, %arg6 [%192, %193] : f32 into vector<2x8xf32>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %197 = vector.extract %191[0, 1] : f32 from vector<1x4xf32>
      %198 = vector.insert %197, %195 [%192, %196] : f32 into vector<2x8xf32>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %200 = vector.extract %191[0, 2] : f32 from vector<1x4xf32>
      %201 = vector.insert %200, %198 [%192, %199] : f32 into vector<2x8xf32>
      %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %203 = vector.extract %191[0, 3] : f32 from vector<1x4xf32>
      %204 = vector.insert %203, %201 [%192, %202] : f32 into vector<2x8xf32>
      %205 = arith.truncf %191 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %206 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %208 = vector.extract %205[0, 0] : f16 from vector<1x4xf16>
      %209 = vector.insert %208, %arg7 [%206, %207] : f16 into vector<2x8xf16>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %211 = vector.extract %205[0, 1] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %209 [%206, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %214 = vector.extract %205[0, 2] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%206, %213] : f16 into vector<2x8xf16>
      %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %217 = vector.extract %205[0, 3] : f16 from vector<1x4xf16>
      %218 = vector.insert %217, %215 [%206, %216] : f16 into vector<2x8xf16>
      %219 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %221 = vector.extract %205[0, 0] : f16 from vector<1x4xf16>
      %222 = vector.insert %221, %arg8 [%219, %220] : f16 into vector<2x8xf16>
      %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %224 = vector.extract %205[0, 1] : f16 from vector<1x4xf16>
      %225 = vector.insert %224, %222 [%219, %223] : f16 into vector<2x8xf16>
      %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %227 = vector.extract %205[0, 2] : f16 from vector<1x4xf16>
      %228 = vector.insert %227, %225 [%219, %226] : f16 into vector<2x8xf16>
      %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %230 = vector.extract %205[0, 3] : f16 from vector<1x4xf16>
      %231 = vector.insert %230, %228 [%219, %229] : f16 into vector<2x8xf16>
      affine.yield %177, %204, %218, %231 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %155 {
      affine.for %arg4 = 0 to 4 {
        %162 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %164 = vector.extract %157#3[%162, %163] : f16 from vector<2x8xf16>
        %165 = vector.insert %164, %cst_22 [0, 0] : f16 into vector<1x4xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %167 = vector.extract %157#3[%162, %166] : f16 from vector<2x8xf16>
        %168 = vector.insert %167, %165 [0, 1] : f16 into vector<1x4xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %170 = vector.extract %157#3[%162, %169] : f16 from vector<2x8xf16>
        %171 = vector.insert %170, %168 [0, 2] : f16 into vector<1x4xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %173 = vector.extract %157#3[%162, %172] : f16 from vector<2x8xf16>
        %174 = vector.insert %173, %171 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_23 = gpu.thread_id x
        %175 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %177 = vector.extract %174[0, 0] : f16 from vector<1x4xf16>
        %c0_24 = arith.constant 0 : index
        %c0_25 = arith.constant 0 : index
        %178 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%176)
        %c0_26 = arith.constant 0 : index
        %179 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%176)
        %c0_27 = arith.constant 0 : index
        %180 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_25, %179)
        %181 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %178, %c0_24, %c0_26, %c0_27, %180)
        %182 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %178, %c0_24, %c0_26, %c0_27, %180)
        affine.store %177, %156[%181, %182] {tiled = true} : memref<128x32xf16, 1>
        %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %184 = vector.extract %174[0, 1] : f16 from vector<1x4xf16>
        %c0_28 = arith.constant 0 : index
        %c0_29 = arith.constant 0 : index
        %185 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%183)
        %c0_30 = arith.constant 0 : index
        %186 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%183)
        %c0_31 = arith.constant 0 : index
        %187 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_29, %186)
        %188 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %185, %c0_28, %c0_30, %c0_31, %187)
        %189 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %185, %c0_28, %c0_30, %c0_31, %187)
        affine.store %184, %156[%188, %189] {tiled = true} : memref<128x32xf16, 1>
        %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %191 = vector.extract %174[0, 2] : f16 from vector<1x4xf16>
        %c0_32 = arith.constant 0 : index
        %c0_33 = arith.constant 0 : index
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%190)
        %c0_34 = arith.constant 0 : index
        %193 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%190)
        %c0_35 = arith.constant 0 : index
        %194 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_33, %193)
        %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %192, %c0_32, %c0_34, %c0_35, %194)
        %196 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %192, %c0_32, %c0_34, %c0_35, %194)
        affine.store %191, %156[%195, %196] {tiled = true} : memref<128x32xf16, 1>
        %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %198 = vector.extract %174[0, 3] : f16 from vector<1x4xf16>
        %c0_36 = arith.constant 0 : index
        %c0_37 = arith.constant 0 : index
        %199 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%197)
        %c0_38 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%197)
        %c0_39 = arith.constant 0 : index
        %201 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_37, %200)
        %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %199, %c0_36, %c0_38, %c0_39, %201)
        %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %199, %c0_36, %c0_38, %c0_39, %201)
        affine.store %198, %156[%202, %203] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %158 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %159:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %162 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %163 = vector.extract %146#1[%162, 0] : f32 from vector<2x1xf32>
      %164 = vector.broadcast %163 : f32 to vector<1x4xf32>
      %165 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %167 = vector.extract %164[0, 0] : f32 from vector<1x4xf32>
      %168 = vector.insert %167, %arg5 [%165, %166] : f32 into vector<2x8xf32>
      %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %170 = vector.extract %164[0, 1] : f32 from vector<1x4xf32>
      %171 = vector.insert %170, %168 [%165, %169] : f32 into vector<2x8xf32>
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %173 = vector.extract %164[0, 2] : f32 from vector<1x4xf32>
      %174 = vector.insert %173, %171 [%165, %172] : f32 into vector<2x8xf32>
      %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %176 = vector.extract %164[0, 3] : f32 from vector<1x4xf32>
      %177 = vector.insert %176, %174 [%165, %175] : f32 into vector<2x8xf32>
      %178 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %180 = vector.extract %153#1[%178, %179] : f32 from vector<2x8xf32>
      %181 = vector.insert %180, %cst_22 [0, 0] : f32 into vector<1x4xf32>
      %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %183 = vector.extract %153#1[%178, %182] : f32 from vector<2x8xf32>
      %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
      %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %186 = vector.extract %153#1[%178, %185] : f32 from vector<2x8xf32>
      %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
      %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %189 = vector.extract %153#1[%178, %188] : f32 from vector<2x8xf32>
      %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
      %191 = arith.divf %190, %164 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %192 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %194 = vector.extract %191[0, 0] : f32 from vector<1x4xf32>
      %195 = vector.insert %194, %arg6 [%192, %193] : f32 into vector<2x8xf32>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %197 = vector.extract %191[0, 1] : f32 from vector<1x4xf32>
      %198 = vector.insert %197, %195 [%192, %196] : f32 into vector<2x8xf32>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %200 = vector.extract %191[0, 2] : f32 from vector<1x4xf32>
      %201 = vector.insert %200, %198 [%192, %199] : f32 into vector<2x8xf32>
      %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %203 = vector.extract %191[0, 3] : f32 from vector<1x4xf32>
      %204 = vector.insert %203, %201 [%192, %202] : f32 into vector<2x8xf32>
      %205 = arith.truncf %191 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %206 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %208 = vector.extract %205[0, 0] : f16 from vector<1x4xf16>
      %209 = vector.insert %208, %arg7 [%206, %207] : f16 into vector<2x8xf16>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %211 = vector.extract %205[0, 1] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %209 [%206, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %214 = vector.extract %205[0, 2] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%206, %213] : f16 into vector<2x8xf16>
      %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %217 = vector.extract %205[0, 3] : f16 from vector<1x4xf16>
      %218 = vector.insert %217, %215 [%206, %216] : f16 into vector<2x8xf16>
      %219 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %221 = vector.extract %205[0, 0] : f16 from vector<1x4xf16>
      %222 = vector.insert %221, %arg8 [%219, %220] : f16 into vector<2x8xf16>
      %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %224 = vector.extract %205[0, 1] : f16 from vector<1x4xf16>
      %225 = vector.insert %224, %222 [%219, %223] : f16 into vector<2x8xf16>
      %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %227 = vector.extract %205[0, 2] : f16 from vector<1x4xf16>
      %228 = vector.insert %227, %225 [%219, %226] : f16 into vector<2x8xf16>
      %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %230 = vector.extract %205[0, 3] : f16 from vector<1x4xf16>
      %231 = vector.insert %230, %228 [%219, %229] : f16 into vector<2x8xf16>
      affine.yield %177, %204, %218, %231 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %155 {
      affine.for %arg4 = 0 to 4 {
        %162 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %164 = vector.extract %159#3[%162, %163] : f16 from vector<2x8xf16>
        %165 = vector.insert %164, %cst_22 [0, 0] : f16 into vector<1x4xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %167 = vector.extract %159#3[%162, %166] : f16 from vector<2x8xf16>
        %168 = vector.insert %167, %165 [0, 1] : f16 into vector<1x4xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %170 = vector.extract %159#3[%162, %169] : f16 from vector<2x8xf16>
        %171 = vector.insert %170, %168 [0, 2] : f16 into vector<1x4xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %173 = vector.extract %159#3[%162, %172] : f16 from vector<2x8xf16>
        %174 = vector.insert %173, %171 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_23 = gpu.thread_id x
        %175 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %177 = vector.extract %174[0, 0] : f16 from vector<1x4xf16>
        %c0_24 = arith.constant 0 : index
        %c0_25 = arith.constant 0 : index
        %178 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%176)
        %c0_26 = arith.constant 0 : index
        %179 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%176)
        %c0_27 = arith.constant 0 : index
        %180 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_25, %179)
        %181 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %178, %c0_24, %c0_26, %c0_27, %180)
        %182 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %178, %c0_24, %c0_26, %c0_27, %180)
        affine.store %177, %158[%181, %182] {tiled = true} : memref<128x32xf16, 1>
        %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %184 = vector.extract %174[0, 1] : f16 from vector<1x4xf16>
        %c0_28 = arith.constant 0 : index
        %c0_29 = arith.constant 0 : index
        %185 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%183)
        %c0_30 = arith.constant 0 : index
        %186 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%183)
        %c0_31 = arith.constant 0 : index
        %187 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_29, %186)
        %188 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %185, %c0_28, %c0_30, %c0_31, %187)
        %189 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %185, %c0_28, %c0_30, %c0_31, %187)
        affine.store %184, %158[%188, %189] {tiled = true} : memref<128x32xf16, 1>
        %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %191 = vector.extract %174[0, 2] : f16 from vector<1x4xf16>
        %c0_32 = arith.constant 0 : index
        %c0_33 = arith.constant 0 : index
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%190)
        %c0_34 = arith.constant 0 : index
        %193 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%190)
        %c0_35 = arith.constant 0 : index
        %194 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_33, %193)
        %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %192, %c0_32, %c0_34, %c0_35, %194)
        %196 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %192, %c0_32, %c0_34, %c0_35, %194)
        affine.store %191, %158[%195, %196] {tiled = true} : memref<128x32xf16, 1>
        %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %198 = vector.extract %174[0, 3] : f16 from vector<1x4xf16>
        %c0_36 = arith.constant 0 : index
        %c0_37 = arith.constant 0 : index
        %199 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%197)
        %c0_38 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%197)
        %c0_39 = arith.constant 0 : index
        %201 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_37, %200)
        %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %199, %c0_36, %c0_38, %c0_39, %201)
        %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %199, %c0_36, %c0_38, %c0_39, %201)
        affine.store %198, %158[%202, %203] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %160 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %161:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0, %arg6 = %cst_0, %arg7 = %cst_6, %arg8 = %cst_6) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %162 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %163 = vector.extract %146#1[%162, 0] : f32 from vector<2x1xf32>
      %164 = vector.broadcast %163 : f32 to vector<1x4xf32>
      %165 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %167 = vector.extract %164[0, 0] : f32 from vector<1x4xf32>
      %168 = vector.insert %167, %arg5 [%165, %166] : f32 into vector<2x8xf32>
      %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %170 = vector.extract %164[0, 1] : f32 from vector<1x4xf32>
      %171 = vector.insert %170, %168 [%165, %169] : f32 into vector<2x8xf32>
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %173 = vector.extract %164[0, 2] : f32 from vector<1x4xf32>
      %174 = vector.insert %173, %171 [%165, %172] : f32 into vector<2x8xf32>
      %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %176 = vector.extract %164[0, 3] : f32 from vector<1x4xf32>
      %177 = vector.insert %176, %174 [%165, %175] : f32 into vector<2x8xf32>
      %178 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %180 = vector.extract %153#0[%178, %179] : f32 from vector<2x8xf32>
      %181 = vector.insert %180, %cst_22 [0, 0] : f32 into vector<1x4xf32>
      %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %183 = vector.extract %153#0[%178, %182] : f32 from vector<2x8xf32>
      %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
      %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %186 = vector.extract %153#0[%178, %185] : f32 from vector<2x8xf32>
      %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
      %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %189 = vector.extract %153#0[%178, %188] : f32 from vector<2x8xf32>
      %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
      %191 = arith.divf %190, %164 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %192 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %194 = vector.extract %191[0, 0] : f32 from vector<1x4xf32>
      %195 = vector.insert %194, %arg6 [%192, %193] : f32 into vector<2x8xf32>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %197 = vector.extract %191[0, 1] : f32 from vector<1x4xf32>
      %198 = vector.insert %197, %195 [%192, %196] : f32 into vector<2x8xf32>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %200 = vector.extract %191[0, 2] : f32 from vector<1x4xf32>
      %201 = vector.insert %200, %198 [%192, %199] : f32 into vector<2x8xf32>
      %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %203 = vector.extract %191[0, 3] : f32 from vector<1x4xf32>
      %204 = vector.insert %203, %201 [%192, %202] : f32 into vector<2x8xf32>
      %205 = arith.truncf %191 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %206 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %208 = vector.extract %205[0, 0] : f16 from vector<1x4xf16>
      %209 = vector.insert %208, %arg7 [%206, %207] : f16 into vector<2x8xf16>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %211 = vector.extract %205[0, 1] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %209 [%206, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %214 = vector.extract %205[0, 2] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%206, %213] : f16 into vector<2x8xf16>
      %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %217 = vector.extract %205[0, 3] : f16 from vector<1x4xf16>
      %218 = vector.insert %217, %215 [%206, %216] : f16 into vector<2x8xf16>
      %219 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %221 = vector.extract %205[0, 0] : f16 from vector<1x4xf16>
      %222 = vector.insert %221, %arg8 [%219, %220] : f16 into vector<2x8xf16>
      %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %224 = vector.extract %205[0, 1] : f16 from vector<1x4xf16>
      %225 = vector.insert %224, %222 [%219, %223] : f16 into vector<2x8xf16>
      %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %227 = vector.extract %205[0, 2] : f16 from vector<1x4xf16>
      %228 = vector.insert %227, %225 [%219, %226] : f16 into vector<2x8xf16>
      %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %230 = vector.extract %205[0, 3] : f16 from vector<1x4xf16>
      %231 = vector.insert %230, %228 [%219, %229] : f16 into vector<2x8xf16>
      affine.yield %177, %204, %218, %231 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>}
    scf.if %155 {
      affine.for %arg4 = 0 to 4 {
        %162 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_22 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %164 = vector.extract %161#3[%162, %163] : f16 from vector<2x8xf16>
        %165 = vector.insert %164, %cst_22 [0, 0] : f16 into vector<1x4xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %167 = vector.extract %161#3[%162, %166] : f16 from vector<2x8xf16>
        %168 = vector.insert %167, %165 [0, 1] : f16 into vector<1x4xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %170 = vector.extract %161#3[%162, %169] : f16 from vector<2x8xf16>
        %171 = vector.insert %170, %168 [0, 2] : f16 into vector<1x4xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %173 = vector.extract %161#3[%162, %172] : f16 from vector<2x8xf16>
        %174 = vector.insert %173, %171 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_23 = gpu.thread_id x
        %175 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %177 = vector.extract %174[0, 0] : f16 from vector<1x4xf16>
        %c0_24 = arith.constant 0 : index
        %c0_25 = arith.constant 0 : index
        %178 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%176)
        %c0_26 = arith.constant 0 : index
        %179 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%176)
        %c0_27 = arith.constant 0 : index
        %180 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_25, %179)
        %181 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %178, %c0_24, %c0_26, %c0_27, %180)
        %182 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %178, %c0_24, %c0_26, %c0_27, %180)
        affine.store %177, %160[%181, %182] {tiled = true} : memref<128x32xf16, 1>
        %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %184 = vector.extract %174[0, 1] : f16 from vector<1x4xf16>
        %c0_28 = arith.constant 0 : index
        %c0_29 = arith.constant 0 : index
        %185 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%183)
        %c0_30 = arith.constant 0 : index
        %186 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%183)
        %c0_31 = arith.constant 0 : index
        %187 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_29, %186)
        %188 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %185, %c0_28, %c0_30, %c0_31, %187)
        %189 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %185, %c0_28, %c0_30, %c0_31, %187)
        affine.store %184, %160[%188, %189] {tiled = true} : memref<128x32xf16, 1>
        %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %191 = vector.extract %174[0, 2] : f16 from vector<1x4xf16>
        %c0_32 = arith.constant 0 : index
        %c0_33 = arith.constant 0 : index
        %192 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%190)
        %c0_34 = arith.constant 0 : index
        %193 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%190)
        %c0_35 = arith.constant 0 : index
        %194 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_33, %193)
        %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %192, %c0_32, %c0_34, %c0_35, %194)
        %196 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %192, %c0_32, %c0_34, %c0_35, %194)
        affine.store %191, %160[%195, %196] {tiled = true} : memref<128x32xf16, 1>
        %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %198 = vector.extract %174[0, 3] : f16 from vector<1x4xf16>
        %c0_36 = arith.constant 0 : index
        %c0_37 = arith.constant 0 : index
        %199 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%197)
        %c0_38 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%197)
        %c0_39 = arith.constant 0 : index
        %201 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_37, %200)
        %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_23, %175, %199, %c0_36, %c0_38, %c0_39, %201)
        %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_23, %175, %199, %c0_36, %c0_38, %c0_39, %201)
        affine.store %198, %160[%202, %203] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
