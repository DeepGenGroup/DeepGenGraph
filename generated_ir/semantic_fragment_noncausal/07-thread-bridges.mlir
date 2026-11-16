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
    %c4 = arith.constant 4 : index
    %cst = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_8 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_10 = arith.constant 0xFF800000 : f32
    %cst_11 = arith.constant 0.127531052 : f32
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
    %8 = arith.truncf %cst_11 : f32 to f16
    %9:4 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_9, %arg6 = %cst_9, %arg7 = %cst_9, %arg8 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
      %168 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %169 = affine.apply affine_map<() -> (0)>()
      %170 = vector.extract %cst_8[0, 0] : f32 from vector<1x1xf32>
      %171 = vector.insert %170, %arg5 [%168, %169] : f32 into vector<2x1xf32>
      %172 = vector.broadcast %cst_10 : f32 to vector<1x1xf32>
      %173 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %174 = affine.apply affine_map<() -> (0)>()
      %175 = vector.extract %172[0, 0] : f32 from vector<1x1xf32>
      %176 = vector.insert %175, %arg6 [%173, %174] : f32 into vector<2x1xf32>
      %177 = arith.addf %cst_8, %172 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
      %178 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %179 = affine.apply affine_map<() -> (0)>()
      %180 = vector.extract %177[0, 0] : f32 from vector<1x1xf32>
      %181 = vector.insert %180, %arg7 [%178, %179] : f32 into vector<2x1xf32>
      %182 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %183 = affine.apply affine_map<() -> (0)>()
      %184 = vector.extract %177[0, 0] : f32 from vector<1x1xf32>
      %185 = vector.insert %184, %arg8 [%182, %183] : f32 into vector<2x1xf32>
      affine.yield %171, %176, %181, %185 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %11 = frisk.to_threadTile %cst_6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %12 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c0) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c0 floordiv 128) mod 4096, %c0 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %12 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
    %13 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %14 = affine.apply affine_map<() -> (0)>()
    %15 = affine.apply affine_map<() -> (0)>()
    %cst_12 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %16 = vector.extract %13[%14, %15] : f16 from vector<2x8xf16>
    %17 = vector.insert %16, %cst_12 [0, 0] : f16 into vector<1x4xf16>
    %18 = affine.apply affine_map<() -> (1)>()
    %19 = vector.extract %13[%14, %18] : f16 from vector<2x8xf16>
    %20 = vector.insert %19, %17 [0, 1] : f16 into vector<1x4xf16>
    %21 = affine.apply affine_map<() -> (2)>()
    %22 = vector.extract %13[%14, %21] : f16 from vector<2x8xf16>
    %23 = vector.insert %22, %20 [0, 2] : f16 into vector<1x4xf16>
    %24 = affine.apply affine_map<() -> (3)>()
    %25 = vector.extract %13[%14, %24] : f16 from vector<2x8xf16>
    %26 = vector.insert %25, %23 [0, 3] : f16 into vector<1x4xf16>
    %27:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_4, %arg6 = %cst_4, %arg7 = %cst_4, %arg8 = %26) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %168 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %169 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %170 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %171 = vector.extract %168[%169, %170] : f16 from vector<2x8xf16>
      %172 = vector.insert %171, %cst_39 [0, 0] : f16 into vector<1x4xf16>
      %173 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %174 = vector.extract %168[%169, %173] : f16 from vector<2x8xf16>
      %175 = vector.insert %174, %172 [0, 1] : f16 into vector<1x4xf16>
      %176 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %177 = vector.extract %168[%169, %176] : f16 from vector<2x8xf16>
      %178 = vector.insert %177, %175 [0, 2] : f16 into vector<1x4xf16>
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %180 = vector.extract %168[%169, %179] : f16 from vector<2x8xf16>
      %181 = vector.insert %180, %178 [0, 3] : f16 into vector<1x4xf16>
      %182 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %184 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %185 = vector.insert %184, %arg5 [%182, %183] : f16 into vector<2x8xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %187 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %188 = vector.insert %187, %185 [%182, %186] : f16 into vector<2x8xf16>
      %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %190 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %191 = vector.insert %190, %188 [%182, %189] : f16 into vector<2x8xf16>
      %192 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %193 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %194 = vector.insert %193, %191 [%182, %192] : f16 into vector<2x8xf16>
      %195 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %196 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %198 = vector.extract %195[0, 0] : f16 from vector<1x4xf16>
      %199 = vector.insert %198, %arg6 [%196, %197] : f16 into vector<2x8xf16>
      %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %201 = vector.extract %195[0, 1] : f16 from vector<1x4xf16>
      %202 = vector.insert %201, %199 [%196, %200] : f16 into vector<2x8xf16>
      %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %204 = vector.extract %195[0, 2] : f16 from vector<1x4xf16>
      %205 = vector.insert %204, %202 [%196, %203] : f16 into vector<2x8xf16>
      %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %207 = vector.extract %195[0, 3] : f16 from vector<1x4xf16>
      %208 = vector.insert %207, %205 [%196, %206] : f16 into vector<2x8xf16>
      %209 = arith.mulf %arg8, %195 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %210 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %212 = vector.extract %209[0, 0] : f16 from vector<1x4xf16>
      %213 = vector.insert %212, %arg7 [%210, %211] : f16 into vector<2x8xf16>
      %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %215 = vector.extract %209[0, 1] : f16 from vector<1x4xf16>
      %216 = vector.insert %215, %213 [%210, %214] : f16 into vector<2x8xf16>
      %217 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %218 = vector.extract %209[0, 2] : f16 from vector<1x4xf16>
      %219 = vector.insert %218, %216 [%210, %217] : f16 into vector<2x8xf16>
      %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %221 = vector.extract %209[0, 3] : f16 from vector<1x4xf16>
      %222 = vector.insert %221, %219 [%210, %220] : f16 into vector<2x8xf16>
      affine.yield %194, %208, %222, %181 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %28 = vector.broadcast %8 : f16 to vector<1x4xf16>
    %29 = arith.mulf %27#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %30 = affine.apply affine_map<() -> (1)>()
    %31 = affine.apply affine_map<() -> (4)>()
    %32 = vector.extract %29[0, 0] : f16 from vector<1x4xf16>
    %33 = vector.insert %32, %27#2 [%30, %31] : f16 into vector<2x8xf16>
    %34 = affine.apply affine_map<() -> (5)>()
    %35 = vector.extract %29[0, 1] : f16 from vector<1x4xf16>
    %36 = vector.insert %35, %33 [%30, %34] : f16 into vector<2x8xf16>
    %37 = affine.apply affine_map<() -> (6)>()
    %38 = vector.extract %29[0, 2] : f16 from vector<1x4xf16>
    %39 = vector.insert %38, %36 [%30, %37] : f16 into vector<2x8xf16>
    %40 = affine.apply affine_map<() -> (7)>()
    %41 = vector.extract %29[0, 3] : f16 from vector<1x4xf16>
    %42 = vector.insert %41, %39 [%30, %40] : f16 into vector<2x8xf16>
    %alloc_13 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %43 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c32) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c32 floordiv 128) mod 4096, %c32 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %43 to %alloc_13 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_13 : memref<128x32xf16, 3>
    %44 = frisk.to_threadTile %alloc_13 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %45 = affine.apply affine_map<() -> (0)>()
    %46 = affine.apply affine_map<() -> (0)>()
    %cst_14 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %47 = vector.extract %44[%45, %46] : f16 from vector<2x8xf16>
    %48 = vector.insert %47, %cst_14 [0, 0] : f16 into vector<1x4xf16>
    %49 = affine.apply affine_map<() -> (1)>()
    %50 = vector.extract %44[%45, %49] : f16 from vector<2x8xf16>
    %51 = vector.insert %50, %48 [0, 1] : f16 into vector<1x4xf16>
    %52 = affine.apply affine_map<() -> (2)>()
    %53 = vector.extract %44[%45, %52] : f16 from vector<2x8xf16>
    %54 = vector.insert %53, %51 [0, 2] : f16 into vector<1x4xf16>
    %55 = affine.apply affine_map<() -> (3)>()
    %56 = vector.extract %44[%45, %55] : f16 from vector<2x8xf16>
    %57 = vector.insert %56, %54 [0, 3] : f16 into vector<1x4xf16>
    %58:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_4, %arg6 = %cst_4, %arg7 = %cst_4, %arg8 = %57) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %168 = frisk.to_threadTile %alloc_13 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %169 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %170 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %171 = vector.extract %168[%169, %170] : f16 from vector<2x8xf16>
      %172 = vector.insert %171, %cst_39 [0, 0] : f16 into vector<1x4xf16>
      %173 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %174 = vector.extract %168[%169, %173] : f16 from vector<2x8xf16>
      %175 = vector.insert %174, %172 [0, 1] : f16 into vector<1x4xf16>
      %176 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %177 = vector.extract %168[%169, %176] : f16 from vector<2x8xf16>
      %178 = vector.insert %177, %175 [0, 2] : f16 into vector<1x4xf16>
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %180 = vector.extract %168[%169, %179] : f16 from vector<2x8xf16>
      %181 = vector.insert %180, %178 [0, 3] : f16 into vector<1x4xf16>
      %182 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %184 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %185 = vector.insert %184, %arg5 [%182, %183] : f16 into vector<2x8xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %187 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %188 = vector.insert %187, %185 [%182, %186] : f16 into vector<2x8xf16>
      %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %190 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %191 = vector.insert %190, %188 [%182, %189] : f16 into vector<2x8xf16>
      %192 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %193 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %194 = vector.insert %193, %191 [%182, %192] : f16 into vector<2x8xf16>
      %195 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %197 = vector.extract %28[0, 0] : f16 from vector<1x4xf16>
      %198 = vector.insert %197, %arg6 [%195, %196] : f16 into vector<2x8xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %200 = vector.extract %28[0, 1] : f16 from vector<1x4xf16>
      %201 = vector.insert %200, %198 [%195, %199] : f16 into vector<2x8xf16>
      %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %203 = vector.extract %28[0, 2] : f16 from vector<1x4xf16>
      %204 = vector.insert %203, %201 [%195, %202] : f16 into vector<2x8xf16>
      %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %206 = vector.extract %28[0, 3] : f16 from vector<1x4xf16>
      %207 = vector.insert %206, %204 [%195, %205] : f16 into vector<2x8xf16>
      %208 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %209 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %211 = vector.extract %208[0, 0] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %arg7 [%209, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %214 = vector.extract %208[0, 1] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%209, %213] : f16 into vector<2x8xf16>
      %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %217 = vector.extract %208[0, 2] : f16 from vector<1x4xf16>
      %218 = vector.insert %217, %215 [%209, %216] : f16 into vector<2x8xf16>
      %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %220 = vector.extract %208[0, 3] : f16 from vector<1x4xf16>
      %221 = vector.insert %220, %218 [%209, %219] : f16 into vector<2x8xf16>
      affine.yield %194, %207, %221, %181 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %59 = arith.mulf %58#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %60 = affine.apply affine_map<() -> (1)>()
    %61 = affine.apply affine_map<() -> (4)>()
    %62 = vector.extract %59[0, 0] : f16 from vector<1x4xf16>
    %63 = vector.insert %62, %58#2 [%60, %61] : f16 into vector<2x8xf16>
    %64 = affine.apply affine_map<() -> (5)>()
    %65 = vector.extract %59[0, 1] : f16 from vector<1x4xf16>
    %66 = vector.insert %65, %63 [%60, %64] : f16 into vector<2x8xf16>
    %67 = affine.apply affine_map<() -> (6)>()
    %68 = vector.extract %59[0, 2] : f16 from vector<1x4xf16>
    %69 = vector.insert %68, %66 [%60, %67] : f16 into vector<2x8xf16>
    %70 = affine.apply affine_map<() -> (7)>()
    %71 = vector.extract %59[0, 3] : f16 from vector<1x4xf16>
    %72 = vector.insert %71, %69 [%60, %70] : f16 into vector<2x8xf16>
    %alloc_15 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %73 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c64) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c64 floordiv 128) mod 4096, %c64 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %73 to %alloc_15 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_15 : memref<128x32xf16, 3>
    %74 = frisk.to_threadTile %alloc_15 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %75 = affine.apply affine_map<() -> (0)>()
    %76 = affine.apply affine_map<() -> (0)>()
    %cst_16 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %77 = vector.extract %74[%75, %76] : f16 from vector<2x8xf16>
    %78 = vector.insert %77, %cst_16 [0, 0] : f16 into vector<1x4xf16>
    %79 = affine.apply affine_map<() -> (1)>()
    %80 = vector.extract %74[%75, %79] : f16 from vector<2x8xf16>
    %81 = vector.insert %80, %78 [0, 1] : f16 into vector<1x4xf16>
    %82 = affine.apply affine_map<() -> (2)>()
    %83 = vector.extract %74[%75, %82] : f16 from vector<2x8xf16>
    %84 = vector.insert %83, %81 [0, 2] : f16 into vector<1x4xf16>
    %85 = affine.apply affine_map<() -> (3)>()
    %86 = vector.extract %74[%75, %85] : f16 from vector<2x8xf16>
    %87 = vector.insert %86, %84 [0, 3] : f16 into vector<1x4xf16>
    %88:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_4, %arg6 = %cst_4, %arg7 = %cst_4, %arg8 = %87) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %168 = frisk.to_threadTile %alloc_15 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %169 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %170 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %171 = vector.extract %168[%169, %170] : f16 from vector<2x8xf16>
      %172 = vector.insert %171, %cst_39 [0, 0] : f16 into vector<1x4xf16>
      %173 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %174 = vector.extract %168[%169, %173] : f16 from vector<2x8xf16>
      %175 = vector.insert %174, %172 [0, 1] : f16 into vector<1x4xf16>
      %176 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %177 = vector.extract %168[%169, %176] : f16 from vector<2x8xf16>
      %178 = vector.insert %177, %175 [0, 2] : f16 into vector<1x4xf16>
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %180 = vector.extract %168[%169, %179] : f16 from vector<2x8xf16>
      %181 = vector.insert %180, %178 [0, 3] : f16 into vector<1x4xf16>
      %182 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %184 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %185 = vector.insert %184, %arg5 [%182, %183] : f16 into vector<2x8xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %187 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %188 = vector.insert %187, %185 [%182, %186] : f16 into vector<2x8xf16>
      %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %190 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %191 = vector.insert %190, %188 [%182, %189] : f16 into vector<2x8xf16>
      %192 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %193 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %194 = vector.insert %193, %191 [%182, %192] : f16 into vector<2x8xf16>
      %195 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %197 = vector.extract %28[0, 0] : f16 from vector<1x4xf16>
      %198 = vector.insert %197, %arg6 [%195, %196] : f16 into vector<2x8xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %200 = vector.extract %28[0, 1] : f16 from vector<1x4xf16>
      %201 = vector.insert %200, %198 [%195, %199] : f16 into vector<2x8xf16>
      %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %203 = vector.extract %28[0, 2] : f16 from vector<1x4xf16>
      %204 = vector.insert %203, %201 [%195, %202] : f16 into vector<2x8xf16>
      %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %206 = vector.extract %28[0, 3] : f16 from vector<1x4xf16>
      %207 = vector.insert %206, %204 [%195, %205] : f16 into vector<2x8xf16>
      %208 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %209 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %211 = vector.extract %208[0, 0] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %arg7 [%209, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %214 = vector.extract %208[0, 1] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%209, %213] : f16 into vector<2x8xf16>
      %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %217 = vector.extract %208[0, 2] : f16 from vector<1x4xf16>
      %218 = vector.insert %217, %215 [%209, %216] : f16 into vector<2x8xf16>
      %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %220 = vector.extract %208[0, 3] : f16 from vector<1x4xf16>
      %221 = vector.insert %220, %218 [%209, %219] : f16 into vector<2x8xf16>
      affine.yield %194, %207, %221, %181 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %89 = arith.mulf %88#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %90 = affine.apply affine_map<() -> (1)>()
    %91 = affine.apply affine_map<() -> (4)>()
    %92 = vector.extract %89[0, 0] : f16 from vector<1x4xf16>
    %93 = vector.insert %92, %88#2 [%90, %91] : f16 into vector<2x8xf16>
    %94 = affine.apply affine_map<() -> (5)>()
    %95 = vector.extract %89[0, 1] : f16 from vector<1x4xf16>
    %96 = vector.insert %95, %93 [%90, %94] : f16 into vector<2x8xf16>
    %97 = affine.apply affine_map<() -> (6)>()
    %98 = vector.extract %89[0, 2] : f16 from vector<1x4xf16>
    %99 = vector.insert %98, %96 [%90, %97] : f16 into vector<2x8xf16>
    %100 = affine.apply affine_map<() -> (7)>()
    %101 = vector.extract %89[0, 3] : f16 from vector<1x4xf16>
    %102 = vector.insert %101, %99 [%90, %100] : f16 into vector<2x8xf16>
    %alloc_17 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %103 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c96) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c96 floordiv 128) mod 4096, %c96 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    frisk.async_copy_gm_to_shm %103 to %alloc_17 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
    frisk.wait_buffer_ready %alloc_17 : memref<128x32xf16, 3>
    %104 = frisk.to_threadTile %alloc_17 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %105 = affine.apply affine_map<() -> (0)>()
    %106 = affine.apply affine_map<() -> (0)>()
    %cst_18 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %107 = vector.extract %104[%105, %106] : f16 from vector<2x8xf16>
    %108 = vector.insert %107, %cst_18 [0, 0] : f16 into vector<1x4xf16>
    %109 = affine.apply affine_map<() -> (1)>()
    %110 = vector.extract %104[%105, %109] : f16 from vector<2x8xf16>
    %111 = vector.insert %110, %108 [0, 1] : f16 into vector<1x4xf16>
    %112 = affine.apply affine_map<() -> (2)>()
    %113 = vector.extract %104[%105, %112] : f16 from vector<2x8xf16>
    %114 = vector.insert %113, %111 [0, 2] : f16 into vector<1x4xf16>
    %115 = affine.apply affine_map<() -> (3)>()
    %116 = vector.extract %104[%105, %115] : f16 from vector<2x8xf16>
    %117 = vector.insert %116, %114 [0, 3] : f16 into vector<1x4xf16>
    %118:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_4, %arg6 = %cst_4, %arg7 = %cst_4, %arg8 = %117) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %168 = frisk.to_threadTile %alloc_17 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %169 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %170 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %171 = vector.extract %168[%169, %170] : f16 from vector<2x8xf16>
      %172 = vector.insert %171, %cst_39 [0, 0] : f16 into vector<1x4xf16>
      %173 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %174 = vector.extract %168[%169, %173] : f16 from vector<2x8xf16>
      %175 = vector.insert %174, %172 [0, 1] : f16 into vector<1x4xf16>
      %176 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %177 = vector.extract %168[%169, %176] : f16 from vector<2x8xf16>
      %178 = vector.insert %177, %175 [0, 2] : f16 into vector<1x4xf16>
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %180 = vector.extract %168[%169, %179] : f16 from vector<2x8xf16>
      %181 = vector.insert %180, %178 [0, 3] : f16 into vector<1x4xf16>
      %182 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %184 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %185 = vector.insert %184, %arg5 [%182, %183] : f16 into vector<2x8xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %187 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %188 = vector.insert %187, %185 [%182, %186] : f16 into vector<2x8xf16>
      %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %190 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %191 = vector.insert %190, %188 [%182, %189] : f16 into vector<2x8xf16>
      %192 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %193 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %194 = vector.insert %193, %191 [%182, %192] : f16 into vector<2x8xf16>
      %195 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %197 = vector.extract %28[0, 0] : f16 from vector<1x4xf16>
      %198 = vector.insert %197, %arg6 [%195, %196] : f16 into vector<2x8xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %200 = vector.extract %28[0, 1] : f16 from vector<1x4xf16>
      %201 = vector.insert %200, %198 [%195, %199] : f16 into vector<2x8xf16>
      %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %203 = vector.extract %28[0, 2] : f16 from vector<1x4xf16>
      %204 = vector.insert %203, %201 [%195, %202] : f16 into vector<2x8xf16>
      %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %206 = vector.extract %28[0, 3] : f16 from vector<1x4xf16>
      %207 = vector.insert %206, %204 [%195, %205] : f16 into vector<2x8xf16>
      %208 = arith.mulf %arg8, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %209 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %211 = vector.extract %208[0, 0] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %arg7 [%209, %210] : f16 into vector<2x8xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %214 = vector.extract %208[0, 1] : f16 from vector<1x4xf16>
      %215 = vector.insert %214, %212 [%209, %213] : f16 into vector<2x8xf16>
      %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %217 = vector.extract %208[0, 2] : f16 from vector<1x4xf16>
      %218 = vector.insert %217, %215 [%209, %216] : f16 into vector<2x8xf16>
      %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %220 = vector.extract %208[0, 3] : f16 from vector<1x4xf16>
      %221 = vector.insert %220, %218 [%209, %219] : f16 into vector<2x8xf16>
      affine.yield %194, %207, %221, %181 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %119 = arith.mulf %118#3, %28 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %120 = affine.apply affine_map<() -> (1)>()
    %121 = affine.apply affine_map<() -> (4)>()
    %122 = vector.extract %119[0, 0] : f16 from vector<1x4xf16>
    %123 = vector.insert %122, %118#2 [%120, %121] : f16 into vector<2x8xf16>
    %124 = affine.apply affine_map<() -> (5)>()
    %125 = vector.extract %119[0, 1] : f16 from vector<1x4xf16>
    %126 = vector.insert %125, %123 [%120, %124] : f16 into vector<2x8xf16>
    %127 = affine.apply affine_map<() -> (6)>()
    %128 = vector.extract %119[0, 2] : f16 from vector<1x4xf16>
    %129 = vector.insert %128, %126 [%120, %127] : f16 into vector<2x8xf16>
    %130 = affine.apply affine_map<() -> (7)>()
    %131 = vector.extract %119[0, 3] : f16 from vector<1x4xf16>
    %132 = vector.insert %131, %129 [%120, %130] : f16 into vector<2x8xf16>
    %133:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#3, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %168 = frisk.to_threadTile %cst_5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %alloc_39 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %169 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c0 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c0 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %169 to %alloc_39 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_39 : memref<32x128xf16, 3>
      %170:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %200 = frisk.to_threadTile %alloc_39 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %201 = affine.apply affine_map<() -> (0)>()
        %202 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %203 = vector.extract %200[%201, %202] : f16 from vector<8x8xf16>
        %204 = vector.insert %203, %cst_52 [0, 0] : f16 into vector<4x1xf16>
        %205 = affine.apply affine_map<() -> (1)>()
        %206 = vector.extract %200[%205, %202] : f16 from vector<8x8xf16>
        %207 = vector.insert %206, %204 [1, 0] : f16 into vector<4x1xf16>
        %208 = affine.apply affine_map<() -> (2)>()
        %209 = vector.extract %200[%208, %202] : f16 from vector<8x8xf16>
        %210 = vector.insert %209, %207 [2, 0] : f16 into vector<4x1xf16>
        %211 = affine.apply affine_map<() -> (3)>()
        %212 = vector.extract %200[%211, %202] : f16 from vector<8x8xf16>
        %213 = vector.insert %212, %210 [3, 0] : f16 into vector<4x1xf16>
        %214:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_2, %arg16 = %213) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %269 = frisk.to_threadTile %alloc_39 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %270 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %271 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %272 = vector.extract %269[%270, %271] : f16 from vector<8x8xf16>
          %273 = vector.insert %272, %cst_55 [0, 0] : f16 into vector<4x1xf16>
          %274 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %275 = vector.extract %269[%274, %271] : f16 from vector<8x8xf16>
          %276 = vector.insert %275, %273 [1, 0] : f16 into vector<4x1xf16>
          %277 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %278 = vector.extract %269[%277, %271] : f16 from vector<8x8xf16>
          %279 = vector.insert %278, %276 [2, 0] : f16 into vector<4x1xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %281 = vector.extract %269[%280, %271] : f16 from vector<8x8xf16>
          %282 = vector.insert %281, %279 [3, 0] : f16 into vector<4x1xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %284 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %285 = vector.extract %42[%283, %284] : f16 from vector<2x8xf16>
          %286 = vector.insert %285, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %287 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %288 = vector.extract %42[%283, %287] : f16 from vector<2x8xf16>
          %289 = vector.insert %288, %286 [0, 1] : f16 into vector<1x4xf16>
          %290 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %291 = vector.extract %42[%283, %290] : f16 from vector<2x8xf16>
          %292 = vector.insert %291, %289 [0, 2] : f16 into vector<1x4xf16>
          %293 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %294 = vector.extract %42[%283, %293] : f16 from vector<2x8xf16>
          %295 = vector.insert %294, %292 [0, 3] : f16 into vector<1x4xf16>
          %296 = frisk.warp_mma_rr %295, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %296, %282 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %215 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %216 = affine.apply affine_map<() -> (4)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %217 = vector.extract %42[%215, %216] : f16 from vector<2x8xf16>
        %218 = vector.insert %217, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %219 = affine.apply affine_map<() -> (5)>()
        %220 = vector.extract %42[%215, %219] : f16 from vector<2x8xf16>
        %221 = vector.insert %220, %218 [0, 1] : f16 into vector<1x4xf16>
        %222 = affine.apply affine_map<() -> (6)>()
        %223 = vector.extract %42[%215, %222] : f16 from vector<2x8xf16>
        %224 = vector.insert %223, %221 [0, 2] : f16 into vector<1x4xf16>
        %225 = affine.apply affine_map<() -> (7)>()
        %226 = vector.extract %42[%215, %225] : f16 from vector<2x8xf16>
        %227 = vector.insert %226, %224 [0, 3] : f16 into vector<1x4xf16>
        %228 = frisk.warp_mma_rr %227, %214#1, %214#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %231 = vector.extract %168[%229, %230] : f32 from vector<2x32xf32>
        %232 = vector.insert %231, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %234 = vector.extract %168[%229, %233] : f32 from vector<2x32xf32>
        %235 = vector.insert %234, %232 [0, 1] : f32 into vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %237 = vector.extract %168[%229, %236] : f32 from vector<2x32xf32>
        %238 = vector.insert %237, %235 [0, 2] : f32 into vector<1x4xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %240 = vector.extract %168[%229, %239] : f32 from vector<2x32xf32>
        %241 = vector.insert %240, %238 [0, 3] : f32 into vector<1x4xf32>
        %242 = arith.addf %241, %228 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %244 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %245 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %246 = vector.insert %245, %arg12 [%243, %244] : f32 into vector<2x32xf32>
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %248 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %246 [%243, %247] : f32 into vector<2x32xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %251 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%243, %250] : f32 into vector<2x32xf32>
        %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %254 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%243, %253] : f32 into vector<2x32xf32>
        %256 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %257 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %258 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %arg13 [%256, %257] : f32 into vector<2x32xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %261 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%256, %260] : f32 into vector<2x32xf32>
        %263 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %264 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %262 [%256, %263] : f32 into vector<2x32xf32>
        %266 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %267 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %268 = vector.insert %267, %265 [%256, %266] : f32 into vector<2x32xf32>
        affine.yield %255, %268 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_40 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %171 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c32 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c32 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %171 to %alloc_40 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_40 : memref<32x128xf16, 3>
      %172:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %200 = frisk.to_threadTile %alloc_40 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %201 = affine.apply affine_map<() -> (0)>()
        %202 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %203 = vector.extract %200[%201, %202] : f16 from vector<8x8xf16>
        %204 = vector.insert %203, %cst_52 [0, 0] : f16 into vector<4x1xf16>
        %205 = affine.apply affine_map<() -> (1)>()
        %206 = vector.extract %200[%205, %202] : f16 from vector<8x8xf16>
        %207 = vector.insert %206, %204 [1, 0] : f16 into vector<4x1xf16>
        %208 = affine.apply affine_map<() -> (2)>()
        %209 = vector.extract %200[%208, %202] : f16 from vector<8x8xf16>
        %210 = vector.insert %209, %207 [2, 0] : f16 into vector<4x1xf16>
        %211 = affine.apply affine_map<() -> (3)>()
        %212 = vector.extract %200[%211, %202] : f16 from vector<8x8xf16>
        %213 = vector.insert %212, %210 [3, 0] : f16 into vector<4x1xf16>
        %214:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_2, %arg16 = %213) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %269 = frisk.to_threadTile %alloc_40 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %270 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %271 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %272 = vector.extract %269[%270, %271] : f16 from vector<8x8xf16>
          %273 = vector.insert %272, %cst_55 [0, 0] : f16 into vector<4x1xf16>
          %274 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %275 = vector.extract %269[%274, %271] : f16 from vector<8x8xf16>
          %276 = vector.insert %275, %273 [1, 0] : f16 into vector<4x1xf16>
          %277 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %278 = vector.extract %269[%277, %271] : f16 from vector<8x8xf16>
          %279 = vector.insert %278, %276 [2, 0] : f16 into vector<4x1xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %281 = vector.extract %269[%280, %271] : f16 from vector<8x8xf16>
          %282 = vector.insert %281, %279 [3, 0] : f16 into vector<4x1xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %284 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %285 = vector.extract %72[%283, %284] : f16 from vector<2x8xf16>
          %286 = vector.insert %285, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %287 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %288 = vector.extract %72[%283, %287] : f16 from vector<2x8xf16>
          %289 = vector.insert %288, %286 [0, 1] : f16 into vector<1x4xf16>
          %290 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %291 = vector.extract %72[%283, %290] : f16 from vector<2x8xf16>
          %292 = vector.insert %291, %289 [0, 2] : f16 into vector<1x4xf16>
          %293 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %294 = vector.extract %72[%283, %293] : f16 from vector<2x8xf16>
          %295 = vector.insert %294, %292 [0, 3] : f16 into vector<1x4xf16>
          %296 = frisk.warp_mma_rr %295, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %296, %282 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %215 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %216 = affine.apply affine_map<() -> (4)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %217 = vector.extract %72[%215, %216] : f16 from vector<2x8xf16>
        %218 = vector.insert %217, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %219 = affine.apply affine_map<() -> (5)>()
        %220 = vector.extract %72[%215, %219] : f16 from vector<2x8xf16>
        %221 = vector.insert %220, %218 [0, 1] : f16 into vector<1x4xf16>
        %222 = affine.apply affine_map<() -> (6)>()
        %223 = vector.extract %72[%215, %222] : f16 from vector<2x8xf16>
        %224 = vector.insert %223, %221 [0, 2] : f16 into vector<1x4xf16>
        %225 = affine.apply affine_map<() -> (7)>()
        %226 = vector.extract %72[%215, %225] : f16 from vector<2x8xf16>
        %227 = vector.insert %226, %224 [0, 3] : f16 into vector<1x4xf16>
        %228 = frisk.warp_mma_rr %227, %214#1, %214#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %231 = vector.extract %170#1[%229, %230] : f32 from vector<2x32xf32>
        %232 = vector.insert %231, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %234 = vector.extract %170#1[%229, %233] : f32 from vector<2x32xf32>
        %235 = vector.insert %234, %232 [0, 1] : f32 into vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %237 = vector.extract %170#1[%229, %236] : f32 from vector<2x32xf32>
        %238 = vector.insert %237, %235 [0, 2] : f32 into vector<1x4xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %240 = vector.extract %170#1[%229, %239] : f32 from vector<2x32xf32>
        %241 = vector.insert %240, %238 [0, 3] : f32 into vector<1x4xf32>
        %242 = arith.addf %241, %228 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %244 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %245 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %246 = vector.insert %245, %arg12 [%243, %244] : f32 into vector<2x32xf32>
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %248 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %246 [%243, %247] : f32 into vector<2x32xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %251 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%243, %250] : f32 into vector<2x32xf32>
        %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %254 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%243, %253] : f32 into vector<2x32xf32>
        %256 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %257 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %258 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %arg13 [%256, %257] : f32 into vector<2x32xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %261 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%256, %260] : f32 into vector<2x32xf32>
        %263 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %264 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %262 [%256, %263] : f32 into vector<2x32xf32>
        %266 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %267 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %268 = vector.insert %267, %265 [%256, %266] : f32 into vector<2x32xf32>
        affine.yield %255, %268 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_41 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %173 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c64 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c64 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %173 to %alloc_41 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_41 : memref<32x128xf16, 3>
      %174:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %200 = frisk.to_threadTile %alloc_41 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %201 = affine.apply affine_map<() -> (0)>()
        %202 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %203 = vector.extract %200[%201, %202] : f16 from vector<8x8xf16>
        %204 = vector.insert %203, %cst_52 [0, 0] : f16 into vector<4x1xf16>
        %205 = affine.apply affine_map<() -> (1)>()
        %206 = vector.extract %200[%205, %202] : f16 from vector<8x8xf16>
        %207 = vector.insert %206, %204 [1, 0] : f16 into vector<4x1xf16>
        %208 = affine.apply affine_map<() -> (2)>()
        %209 = vector.extract %200[%208, %202] : f16 from vector<8x8xf16>
        %210 = vector.insert %209, %207 [2, 0] : f16 into vector<4x1xf16>
        %211 = affine.apply affine_map<() -> (3)>()
        %212 = vector.extract %200[%211, %202] : f16 from vector<8x8xf16>
        %213 = vector.insert %212, %210 [3, 0] : f16 into vector<4x1xf16>
        %214:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_2, %arg16 = %213) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %269 = frisk.to_threadTile %alloc_41 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %270 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %271 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %272 = vector.extract %269[%270, %271] : f16 from vector<8x8xf16>
          %273 = vector.insert %272, %cst_55 [0, 0] : f16 into vector<4x1xf16>
          %274 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %275 = vector.extract %269[%274, %271] : f16 from vector<8x8xf16>
          %276 = vector.insert %275, %273 [1, 0] : f16 into vector<4x1xf16>
          %277 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %278 = vector.extract %269[%277, %271] : f16 from vector<8x8xf16>
          %279 = vector.insert %278, %276 [2, 0] : f16 into vector<4x1xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %281 = vector.extract %269[%280, %271] : f16 from vector<8x8xf16>
          %282 = vector.insert %281, %279 [3, 0] : f16 into vector<4x1xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %284 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %285 = vector.extract %102[%283, %284] : f16 from vector<2x8xf16>
          %286 = vector.insert %285, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %287 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %288 = vector.extract %102[%283, %287] : f16 from vector<2x8xf16>
          %289 = vector.insert %288, %286 [0, 1] : f16 into vector<1x4xf16>
          %290 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %291 = vector.extract %102[%283, %290] : f16 from vector<2x8xf16>
          %292 = vector.insert %291, %289 [0, 2] : f16 into vector<1x4xf16>
          %293 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %294 = vector.extract %102[%283, %293] : f16 from vector<2x8xf16>
          %295 = vector.insert %294, %292 [0, 3] : f16 into vector<1x4xf16>
          %296 = frisk.warp_mma_rr %295, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %296, %282 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %215 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %216 = affine.apply affine_map<() -> (4)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %217 = vector.extract %102[%215, %216] : f16 from vector<2x8xf16>
        %218 = vector.insert %217, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %219 = affine.apply affine_map<() -> (5)>()
        %220 = vector.extract %102[%215, %219] : f16 from vector<2x8xf16>
        %221 = vector.insert %220, %218 [0, 1] : f16 into vector<1x4xf16>
        %222 = affine.apply affine_map<() -> (6)>()
        %223 = vector.extract %102[%215, %222] : f16 from vector<2x8xf16>
        %224 = vector.insert %223, %221 [0, 2] : f16 into vector<1x4xf16>
        %225 = affine.apply affine_map<() -> (7)>()
        %226 = vector.extract %102[%215, %225] : f16 from vector<2x8xf16>
        %227 = vector.insert %226, %224 [0, 3] : f16 into vector<1x4xf16>
        %228 = frisk.warp_mma_rr %227, %214#1, %214#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %231 = vector.extract %172#1[%229, %230] : f32 from vector<2x32xf32>
        %232 = vector.insert %231, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %234 = vector.extract %172#1[%229, %233] : f32 from vector<2x32xf32>
        %235 = vector.insert %234, %232 [0, 1] : f32 into vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %237 = vector.extract %172#1[%229, %236] : f32 from vector<2x32xf32>
        %238 = vector.insert %237, %235 [0, 2] : f32 into vector<1x4xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %240 = vector.extract %172#1[%229, %239] : f32 from vector<2x32xf32>
        %241 = vector.insert %240, %238 [0, 3] : f32 into vector<1x4xf32>
        %242 = arith.addf %241, %228 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %244 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %245 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %246 = vector.insert %245, %arg12 [%243, %244] : f32 into vector<2x32xf32>
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %248 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %246 [%243, %247] : f32 into vector<2x32xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %251 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%243, %250] : f32 into vector<2x32xf32>
        %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %254 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%243, %253] : f32 into vector<2x32xf32>
        %256 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %257 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %258 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %arg13 [%256, %257] : f32 into vector<2x32xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %261 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%256, %260] : f32 into vector<2x32xf32>
        %263 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %264 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %262 [%256, %263] : f32 into vector<2x32xf32>
        %266 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %267 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %268 = vector.insert %267, %265 [%256, %266] : f32 into vector<2x32xf32>
        affine.yield %255, %268 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %alloc_42 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %175 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c96 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c96 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      frisk.async_copy_gm_to_shm %175 to %alloc_42 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      frisk.wait_buffer_ready %alloc_42 : memref<32x128xf16, 3>
      %176:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %200 = frisk.to_threadTile %alloc_42 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %201 = affine.apply affine_map<() -> (0)>()
        %202 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %203 = vector.extract %200[%201, %202] : f16 from vector<8x8xf16>
        %204 = vector.insert %203, %cst_52 [0, 0] : f16 into vector<4x1xf16>
        %205 = affine.apply affine_map<() -> (1)>()
        %206 = vector.extract %200[%205, %202] : f16 from vector<8x8xf16>
        %207 = vector.insert %206, %204 [1, 0] : f16 into vector<4x1xf16>
        %208 = affine.apply affine_map<() -> (2)>()
        %209 = vector.extract %200[%208, %202] : f16 from vector<8x8xf16>
        %210 = vector.insert %209, %207 [2, 0] : f16 into vector<4x1xf16>
        %211 = affine.apply affine_map<() -> (3)>()
        %212 = vector.extract %200[%211, %202] : f16 from vector<8x8xf16>
        %213 = vector.insert %212, %210 [3, 0] : f16 into vector<4x1xf16>
        %214:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_2, %arg16 = %213) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %269 = frisk.to_threadTile %alloc_42 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %270 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %271 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %272 = vector.extract %269[%270, %271] : f16 from vector<8x8xf16>
          %273 = vector.insert %272, %cst_55 [0, 0] : f16 into vector<4x1xf16>
          %274 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %275 = vector.extract %269[%274, %271] : f16 from vector<8x8xf16>
          %276 = vector.insert %275, %273 [1, 0] : f16 into vector<4x1xf16>
          %277 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %278 = vector.extract %269[%277, %271] : f16 from vector<8x8xf16>
          %279 = vector.insert %278, %276 [2, 0] : f16 into vector<4x1xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %281 = vector.extract %269[%280, %271] : f16 from vector<8x8xf16>
          %282 = vector.insert %281, %279 [3, 0] : f16 into vector<4x1xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %284 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %285 = vector.extract %132[%283, %284] : f16 from vector<2x8xf16>
          %286 = vector.insert %285, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %287 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %288 = vector.extract %132[%283, %287] : f16 from vector<2x8xf16>
          %289 = vector.insert %288, %286 [0, 1] : f16 into vector<1x4xf16>
          %290 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %291 = vector.extract %132[%283, %290] : f16 from vector<2x8xf16>
          %292 = vector.insert %291, %289 [0, 2] : f16 into vector<1x4xf16>
          %293 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %294 = vector.extract %132[%283, %293] : f16 from vector<2x8xf16>
          %295 = vector.insert %294, %292 [0, 3] : f16 into vector<1x4xf16>
          %296 = frisk.warp_mma_rr %295, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %296, %282 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %215 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %216 = affine.apply affine_map<() -> (4)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %217 = vector.extract %132[%215, %216] : f16 from vector<2x8xf16>
        %218 = vector.insert %217, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %219 = affine.apply affine_map<() -> (5)>()
        %220 = vector.extract %132[%215, %219] : f16 from vector<2x8xf16>
        %221 = vector.insert %220, %218 [0, 1] : f16 into vector<1x4xf16>
        %222 = affine.apply affine_map<() -> (6)>()
        %223 = vector.extract %132[%215, %222] : f16 from vector<2x8xf16>
        %224 = vector.insert %223, %221 [0, 2] : f16 into vector<1x4xf16>
        %225 = affine.apply affine_map<() -> (7)>()
        %226 = vector.extract %132[%215, %225] : f16 from vector<2x8xf16>
        %227 = vector.insert %226, %224 [0, 3] : f16 into vector<1x4xf16>
        %228 = frisk.warp_mma_rr %227, %214#1, %214#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %231 = vector.extract %174#1[%229, %230] : f32 from vector<2x32xf32>
        %232 = vector.insert %231, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %234 = vector.extract %174#1[%229, %233] : f32 from vector<2x32xf32>
        %235 = vector.insert %234, %232 [0, 1] : f32 into vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %237 = vector.extract %174#1[%229, %236] : f32 from vector<2x32xf32>
        %238 = vector.insert %237, %235 [0, 2] : f32 into vector<1x4xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %240 = vector.extract %174#1[%229, %239] : f32 from vector<2x32xf32>
        %241 = vector.insert %240, %238 [0, 3] : f32 into vector<1x4xf32>
        %242 = arith.addf %241, %228 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %244 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %245 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %246 = vector.insert %245, %arg12 [%243, %244] : f32 into vector<2x32xf32>
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %248 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %246 [%243, %247] : f32 into vector<2x32xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %251 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%243, %250] : f32 into vector<2x32xf32>
        %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %254 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%243, %253] : f32 into vector<2x32xf32>
        %256 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %257 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %258 = vector.extract %242[0, 0] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %arg13 [%256, %257] : f32 into vector<2x32xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %261 = vector.extract %242[0, 1] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%256, %260] : f32 into vector<2x32xf32>
        %263 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %264 = vector.extract %242[0, 2] : f32 from vector<1x4xf32>
        %265 = vector.insert %264, %262 [%256, %263] : f32 into vector<2x32xf32>
        %266 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %267 = vector.extract %242[0, 3] : f32 from vector<1x4xf32>
        %268 = vector.insert %267, %265 [%256, %266] : f32 into vector<2x32xf32>
        affine.yield %255, %268 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %177 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %200 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %202 = vector.extract %176#1[%200, %201] : f32 from vector<2x32xf32>
        %203 = vector.insert %202, %cst_52 [0, 0] : f32 into vector<1x4xf32>
        %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %205 = vector.extract %176#1[%200, %204] : f32 from vector<2x32xf32>
        %206 = vector.insert %205, %203 [0, 1] : f32 into vector<1x4xf32>
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %208 = vector.extract %176#1[%200, %207] : f32 from vector<2x32xf32>
        %209 = vector.insert %208, %206 [0, 2] : f32 into vector<1x4xf32>
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %211 = vector.extract %176#1[%200, %210] : f32 from vector<2x32xf32>
        %212 = vector.insert %211, %209 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %215 = vector.extract %212[0, 0] : f32 from vector<1x4xf32>
        %216 = vector.insert %215, %arg12 [%213, %214] : f32 into vector<2x32xf32>
        %217 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %218 = vector.extract %212[0, 1] : f32 from vector<1x4xf32>
        %219 = vector.insert %218, %216 [%213, %217] : f32 into vector<2x32xf32>
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %221 = vector.extract %212[0, 2] : f32 from vector<1x4xf32>
        %222 = vector.insert %221, %219 [%213, %220] : f32 into vector<2x32xf32>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %224 = vector.extract %212[0, 3] : f32 from vector<1x4xf32>
        %225 = vector.insert %224, %222 [%213, %223] : f32 into vector<2x32xf32>
        affine.yield %225 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_43 = arith.constant 0xFF800000 : f32
      %cst_44 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %178 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_44) -> (vector<2x1xf32>) {
        %c0_52 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %201 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_43) -> (f32) {
          %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %208 = vector.extract %177[%200, %207] : f32 from vector<2x32xf32>
          %209 = arith.maxnumf %arg14, %208 : f32
          %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %211 = vector.extract %177[%200, %210] : f32 from vector<2x32xf32>
          %212 = arith.maxnumf %209, %211 : f32
          %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %214 = vector.extract %177[%200, %213] : f32 from vector<2x32xf32>
          %215 = arith.maxnumf %212, %214 : f32
          %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %217 = vector.extract %177[%200, %216] : f32 from vector<2x32xf32>
          %218 = arith.maxnumf %215, %217 : f32
          affine.yield %218 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %c64_i32_54 = arith.constant 64 : i32
        %c16_i32_55 = arith.constant 16 : i32
        %shuffleResult, %valid = gpu.shuffle xor %201, %c16_i32_55, %c64_i32_54 : f32
        %202 = arith.maxnumf %201, %shuffleResult : f32
        %c64_i32_56 = arith.constant 64 : i32
        %c32_i32_57 = arith.constant 32 : i32
        %shuffleResult_58, %valid_59 = gpu.shuffle xor %202, %c32_i32_57, %c64_i32_56 : f32
        %203 = arith.maxnumf %202, %shuffleResult_58 : f32
        %204 = vector.insert %203, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %205 = vector.extract %204[0, 0] : f32 from vector<1x1xf32>
        %206 = vector.insert %205, %arg12 [%200, %c0_52] : f32 into vector<2x1xf32>
        affine.yield %206 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %179 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %200 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %201 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %202 = vector.extract %178[%200, %201] : f32 from vector<2x1xf32>
        %203 = vector.insert %202, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %204 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %205 = affine.apply affine_map<() -> (0)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %206 = vector.extract %arg7[%204, %205] : f32 from vector<2x1xf32>
        %207 = vector.insert %206, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %208 = arith.maxnumf %203, %207 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %209 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %210 = affine.apply affine_map<() -> (0)>()
        %211 = vector.extract %208[0, 0] : f32 from vector<1x1xf32>
        %212 = vector.insert %211, %arg12 [%209, %210] : f32 into vector<2x1xf32>
        affine.yield %212 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %180:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %200 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %201 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %202 = vector.extract %179[%200, %201] : f32 from vector<2x1xf32>
        %203 = vector.insert %202, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %204 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_53) -> (vector<1x4xf32>) {
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %245 = vector.extract %203[%c0_58, %c0_59] : f32 from vector<1x1xf32>
          %246 = vector.insert %245, %cst_57 [0, 0] : f32 into vector<1x4xf32>
          %c0_60 = arith.constant 0 : index
          %c0_61 = arith.constant 0 : index
          %247 = vector.extract %203[%c0_60, %c0_61] : f32 from vector<1x1xf32>
          %248 = vector.insert %247, %246 [0, 1] : f32 into vector<1x4xf32>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %249 = vector.extract %203[%c0_62, %c0_63] : f32 from vector<1x1xf32>
          %250 = vector.insert %249, %248 [0, 2] : f32 into vector<1x4xf32>
          %c0_64 = arith.constant 0 : index
          %c0_65 = arith.constant 0 : index
          %251 = vector.extract %203[%c0_64, %c0_65] : f32 from vector<1x1xf32>
          %252 = vector.insert %251, %250 [0, 3] : f32 into vector<1x4xf32>
          %253 = vector.extract %252[0, 0] : f32 from vector<1x4xf32>
          %254 = vector.insert %253, %arg15 [%c0_56, %c0_55] : f32 into vector<1x4xf32>
          %255 = affine.apply affine_map<() -> (1)>()
          %256 = vector.extract %252[0, 1] : f32 from vector<1x4xf32>
          %257 = vector.insert %256, %254 [%c0_56, %255] : f32 into vector<1x4xf32>
          %258 = affine.apply affine_map<() -> (2)>()
          %259 = vector.extract %252[0, 2] : f32 from vector<1x4xf32>
          %260 = vector.insert %259, %257 [%c0_56, %258] : f32 into vector<1x4xf32>
          %261 = affine.apply affine_map<() -> (3)>()
          %262 = vector.extract %252[0, 3] : f32 from vector<1x4xf32>
          %263 = vector.insert %262, %260 [%c0_56, %261] : f32 into vector<1x4xf32>
          affine.yield %263 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
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
        %218 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %220 = vector.extract %176#1[%218, %219] : f32 from vector<2x32xf32>
        %221 = vector.insert %220, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %223 = vector.extract %176#1[%218, %222] : f32 from vector<2x32xf32>
        %224 = vector.insert %223, %221 [0, 1] : f32 into vector<1x4xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %226 = vector.extract %176#1[%218, %225] : f32 from vector<2x32xf32>
        %227 = vector.insert %226, %224 [0, 2] : f32 into vector<1x4xf32>
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %229 = vector.extract %176#1[%218, %228] : f32 from vector<2x32xf32>
        %230 = vector.insert %229, %227 [0, 3] : f32 into vector<1x4xf32>
        %231 = arith.subf %230, %204 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %234 = vector.extract %231[0, 0] : f32 from vector<1x4xf32>
        %235 = vector.insert %234, %arg13 [%232, %233] : f32 into vector<2x32xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %237 = vector.extract %231[0, 1] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %235 [%232, %236] : f32 into vector<2x32xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %240 = vector.extract %231[0, 2] : f32 from vector<1x4xf32>
        %241 = vector.insert %240, %238 [%232, %239] : f32 into vector<2x32xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %243 = vector.extract %231[0, 3] : f32 from vector<1x4xf32>
        %244 = vector.insert %243, %241 [%232, %242] : f32 into vector<2x32xf32>
        affine.yield %217, %244 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %181:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9, %arg14 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %200 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %201 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %202 = vector.extract %arg7[%200, %201] : f32 from vector<2x1xf32>
        %203 = vector.insert %202, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %204 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %205 = affine.apply affine_map<() -> (0)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %206 = vector.extract %179[%204, %205] : f32 from vector<2x1xf32>
        %207 = vector.insert %206, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %208 = arith.subf %203, %207 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %209 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %210 = affine.apply affine_map<() -> (0)>()
        %211 = vector.extract %208[0, 0] : f32 from vector<1x1xf32>
        %212 = vector.insert %211, %arg12 [%209, %210] : f32 into vector<2x1xf32>
        %213 = math.exp2 %208 {tiled = true} : vector<1x1xf32>
        %214 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %215 = affine.apply affine_map<() -> (0)>()
        %216 = vector.extract %213[0, 0] : f32 from vector<1x1xf32>
        %217 = vector.insert %216, %arg13 [%214, %215] : f32 into vector<2x1xf32>
        %218 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %219 = affine.apply affine_map<() -> (0)>()
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %220 = vector.extract %arg6[%218, %219] : f32 from vector<2x1xf32>
        %221 = vector.insert %220, %cst_54 [0, 0] : f32 into vector<1x1xf32>
        %222 = arith.mulf %221, %213 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %223 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %224 = affine.apply affine_map<() -> (0)>()
        %225 = vector.extract %222[0, 0] : f32 from vector<1x1xf32>
        %226 = vector.insert %225, %arg14 [%223, %224] : f32 into vector<2x1xf32>
        affine.yield %212, %217, %226 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %182:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %200 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %202 = vector.extract %180#1[%200, %201] : f32 from vector<2x32xf32>
        %203 = vector.insert %202, %cst_52 [0, 0] : f32 into vector<1x4xf32>
        %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %205 = vector.extract %180#1[%200, %204] : f32 from vector<2x32xf32>
        %206 = vector.insert %205, %203 [0, 1] : f32 into vector<1x4xf32>
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %208 = vector.extract %180#1[%200, %207] : f32 from vector<2x32xf32>
        %209 = vector.insert %208, %206 [0, 2] : f32 into vector<1x4xf32>
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %211 = vector.extract %180#1[%200, %210] : f32 from vector<2x32xf32>
        %212 = vector.insert %211, %209 [0, 3] : f32 into vector<1x4xf32>
        %213 = math.exp2 %212 {tiled = true} : vector<1x4xf32>
        %214 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %215 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %216 = vector.extract %213[0, 0] : f32 from vector<1x4xf32>
        %217 = vector.insert %216, %arg12 [%214, %215] : f32 into vector<2x32xf32>
        %218 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %219 = vector.extract %213[0, 1] : f32 from vector<1x4xf32>
        %220 = vector.insert %219, %217 [%214, %218] : f32 into vector<2x32xf32>
        %221 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %222 = vector.extract %213[0, 2] : f32 from vector<1x4xf32>
        %223 = vector.insert %222, %220 [%214, %221] : f32 into vector<2x32xf32>
        %224 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %225 = vector.extract %213[0, 3] : f32 from vector<1x4xf32>
        %226 = vector.insert %225, %223 [%214, %224] : f32 into vector<2x32xf32>
        %227 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %229 = vector.extract %213[0, 0] : f32 from vector<1x4xf32>
        %230 = vector.insert %229, %arg13 [%227, %228] : f32 into vector<2x32xf32>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %232 = vector.extract %213[0, 1] : f32 from vector<1x4xf32>
        %233 = vector.insert %232, %230 [%227, %231] : f32 into vector<2x32xf32>
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %235 = vector.extract %213[0, 2] : f32 from vector<1x4xf32>
        %236 = vector.insert %235, %233 [%227, %234] : f32 into vector<2x32xf32>
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %238 = vector.extract %213[0, 3] : f32 from vector<1x4xf32>
        %239 = vector.insert %238, %236 [%227, %237] : f32 into vector<2x32xf32>
        affine.yield %226, %239 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_45 = arith.constant 0.000000e+00 : f32
      %cst_46 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %183 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_46) -> (vector<2x1xf32>) {
        %c0_52 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %201 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_45) -> (f32) {
          %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %208 = vector.extract %182#1[%200, %207] : f32 from vector<2x32xf32>
          %209 = arith.addf %arg14, %208 : f32
          %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %211 = vector.extract %182#1[%200, %210] : f32 from vector<2x32xf32>
          %212 = arith.addf %209, %211 : f32
          %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %214 = vector.extract %182#1[%200, %213] : f32 from vector<2x32xf32>
          %215 = arith.addf %212, %214 : f32
          %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %217 = vector.extract %182#1[%200, %216] : f32 from vector<2x32xf32>
          %218 = arith.addf %215, %217 : f32
          affine.yield %218 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %c64_i32_54 = arith.constant 64 : i32
        %c16_i32_55 = arith.constant 16 : i32
        %shuffleResult, %valid = gpu.shuffle xor %201, %c16_i32_55, %c64_i32_54 : f32
        %202 = arith.addf %201, %shuffleResult : f32
        %c64_i32_56 = arith.constant 64 : i32
        %c32_i32_57 = arith.constant 32 : i32
        %shuffleResult_58, %valid_59 = gpu.shuffle xor %202, %c32_i32_57, %c64_i32_56 : f32
        %203 = arith.addf %202, %shuffleResult_58 : f32
        %204 = vector.insert %203, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %205 = vector.extract %204[0, 0] : f32 from vector<1x1xf32>
        %206 = vector.insert %205, %arg12 [%200, %c0_52] : f32 into vector<2x1xf32>
        affine.yield %206 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %184 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %200 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %201 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %202 = vector.extract %183[%200, %201] : f32 from vector<2x1xf32>
        %203 = vector.insert %202, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %204 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %205 = affine.apply affine_map<() -> (0)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %206 = vector.extract %181#2[%204, %205] : f32 from vector<2x1xf32>
        %207 = vector.insert %206, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %208 = arith.addf %203, %207 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %209 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %210 = affine.apply affine_map<() -> (0)>()
        %211 = vector.extract %208[0, 0] : f32 from vector<1x1xf32>
        %212 = vector.insert %211, %arg12 [%209, %210] : f32 into vector<2x1xf32>
        affine.yield %212 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %185 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_1) -> (vector<2x32xf16>) {
        %200 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %202 = vector.extract %182#0[%200, %201] : f32 from vector<2x32xf32>
        %203 = vector.insert %202, %cst_52 [0, 0] : f32 into vector<1x4xf32>
        %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %205 = vector.extract %182#0[%200, %204] : f32 from vector<2x32xf32>
        %206 = vector.insert %205, %203 [0, 1] : f32 into vector<1x4xf32>
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %208 = vector.extract %182#0[%200, %207] : f32 from vector<2x32xf32>
        %209 = vector.insert %208, %206 [0, 2] : f32 into vector<1x4xf32>
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %211 = vector.extract %182#0[%200, %210] : f32 from vector<2x32xf32>
        %212 = vector.insert %211, %209 [0, 3] : f32 into vector<1x4xf32>
        %213 = arith.truncf %212 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %214 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %215 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %216 = vector.extract %213[0, 0] : f16 from vector<1x4xf16>
        %217 = vector.insert %216, %arg12 [%214, %215] : f16 into vector<2x32xf16>
        %218 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %219 = vector.extract %213[0, 1] : f16 from vector<1x4xf16>
        %220 = vector.insert %219, %217 [%214, %218] : f16 into vector<2x32xf16>
        %221 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %222 = vector.extract %213[0, 2] : f16 from vector<1x4xf16>
        %223 = vector.insert %222, %220 [%214, %221] : f16 into vector<2x32xf16>
        %224 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %225 = vector.extract %213[0, 3] : f16 from vector<1x4xf16>
        %226 = vector.insert %225, %223 [%214, %224] : f16 into vector<2x32xf16>
        affine.yield %226 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_47 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %186 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %186 to %alloc_47 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_47 : memref<128x32xf16, 3>
      %thread_id_x_48 = gpu.thread_id x
      %187 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %200 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_48]
        %201 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_48]
        %202 = frisk.read_matrix_from_shm %alloc_47[%200, %201] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %203 = vector.bitcast %202 : vector<4xi32> to vector<8xf16>
        %204 = vector.extract %203[0] : f16 from vector<8xf16>
        %205 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_48]
        %206 = vector.extract %203[%205] : f16 from vector<8xf16>
        %207 = vector.extract %203[1] : f16 from vector<8xf16>
        %208 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_48]
        %209 = vector.extract %203[%208] : f16 from vector<8xf16>
        %210 = vector.extract %203[2] : f16 from vector<8xf16>
        %211 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_48]
        %212 = vector.extract %203[%211] : f16 from vector<8xf16>
        %213 = vector.extract %203[3] : f16 from vector<8xf16>
        %214 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_48]
        %215 = vector.extract %203[%214] : f16 from vector<8xf16>
        %216 = vector.from_elements %204, %206, %207, %209, %210, %212, %213, %215 : vector<4x2xf16>
        %217 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %218 = affine.apply affine_map<() -> (0)>()
        %219 = vector.extract %216[0, 0] : f16 from vector<4x2xf16>
        %220 = vector.insert %219, %arg12 [%217, %218] : f16 into vector<32x2xf16>
        %221 = affine.apply affine_map<() -> (1)>()
        %222 = vector.extract %216[0, 1] : f16 from vector<4x2xf16>
        %223 = vector.insert %222, %220 [%217, %221] : f16 into vector<32x2xf16>
        %224 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %225 = vector.extract %216[1, 0] : f16 from vector<4x2xf16>
        %226 = vector.insert %225, %223 [%224, %218] : f16 into vector<32x2xf16>
        %227 = affine.apply affine_map<() -> (1)>()
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %229 = vector.extract %216[1, 1] : f16 from vector<4x2xf16>
        %230 = vector.insert %229, %226 [%228, %227] : f16 into vector<32x2xf16>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %232 = vector.extract %216[2, 0] : f16 from vector<4x2xf16>
        %233 = vector.insert %232, %230 [%231, %218] : f16 into vector<32x2xf16>
        %234 = affine.apply affine_map<() -> (1)>()
        %235 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %236 = vector.extract %216[2, 1] : f16 from vector<4x2xf16>
        %237 = vector.insert %236, %233 [%235, %234] : f16 into vector<32x2xf16>
        %238 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %239 = vector.extract %216[3, 0] : f16 from vector<4x2xf16>
        %240 = vector.insert %239, %237 [%238, %218] : f16 into vector<32x2xf16>
        %241 = affine.apply affine_map<() -> (1)>()
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %243 = vector.extract %216[3, 1] : f16 from vector<4x2xf16>
        %244 = vector.insert %243, %240 [%242, %241] : f16 into vector<32x2xf16>
        affine.yield %244 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %188:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %200 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_2) -> (vector<1x4xf32>) {
          %273 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %274 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %275 = vector.extract %185[%273, %274] : f16 from vector<2x32xf16>
          %276 = vector.insert %275, %cst_55 [0, 0] : f16 into vector<1x4xf16>
          %277 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %278 = vector.extract %185[%273, %277] : f16 from vector<2x32xf16>
          %279 = vector.insert %278, %276 [0, 1] : f16 into vector<1x4xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %281 = vector.extract %185[%273, %280] : f16 from vector<2x32xf16>
          %282 = vector.insert %281, %279 [0, 2] : f16 into vector<1x4xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %284 = vector.extract %185[%273, %283] : f16 from vector<2x32xf16>
          %285 = vector.insert %284, %282 [0, 3] : f16 into vector<1x4xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %287 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %288 = vector.extract %187[%286, %287] : f16 from vector<32x2xf16>
          %289 = vector.insert %288, %cst_56 [0, 0] : f16 into vector<4x1xf16>
          %290 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %291 = vector.extract %187[%290, %287] : f16 from vector<32x2xf16>
          %292 = vector.insert %291, %289 [1, 0] : f16 into vector<4x1xf16>
          %293 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %294 = vector.extract %187[%293, %287] : f16 from vector<32x2xf16>
          %295 = vector.insert %294, %292 [2, 0] : f16 into vector<4x1xf16>
          %296 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %297 = vector.extract %187[%296, %287] : f16 from vector<32x2xf16>
          %298 = vector.insert %297, %295 [3, 0] : f16 into vector<4x1xf16>
          %299 = frisk.warp_mma_rr %285, %298, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %299 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %201 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %203 = vector.extract %200[0, 0] : f32 from vector<1x4xf32>
        %204 = vector.insert %203, %arg12 [%201, %202] : f32 into vector<2x8xf32>
        %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %206 = vector.extract %200[0, 1] : f32 from vector<1x4xf32>
        %207 = vector.insert %206, %204 [%201, %205] : f32 into vector<2x8xf32>
        %208 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %209 = vector.extract %200[0, 2] : f32 from vector<1x4xf32>
        %210 = vector.insert %209, %207 [%201, %208] : f32 into vector<2x8xf32>
        %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %212 = vector.extract %200[0, 3] : f32 from vector<1x4xf32>
        %213 = vector.insert %212, %210 [%201, %211] : f32 into vector<2x8xf32>
        %214 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %215 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %216 = vector.extract %181#1[%214, %215] : f32 from vector<2x1xf32>
        %217 = vector.insert %216, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %218 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_53) -> (vector<1x4xf32>) {
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %273 = vector.extract %217[%c0_58, %c0_59] : f32 from vector<1x1xf32>
          %274 = vector.insert %273, %cst_57 [0, 0] : f32 into vector<1x4xf32>
          %c0_60 = arith.constant 0 : index
          %c0_61 = arith.constant 0 : index
          %275 = vector.extract %217[%c0_60, %c0_61] : f32 from vector<1x1xf32>
          %276 = vector.insert %275, %274 [0, 1] : f32 into vector<1x4xf32>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %277 = vector.extract %217[%c0_62, %c0_63] : f32 from vector<1x1xf32>
          %278 = vector.insert %277, %276 [0, 2] : f32 into vector<1x4xf32>
          %c0_64 = arith.constant 0 : index
          %c0_65 = arith.constant 0 : index
          %279 = vector.extract %217[%c0_64, %c0_65] : f32 from vector<1x1xf32>
          %280 = vector.insert %279, %278 [0, 3] : f32 into vector<1x4xf32>
          %281 = vector.extract %280[0, 0] : f32 from vector<1x4xf32>
          %282 = vector.insert %281, %arg17 [%c0_56, %c0_55] : f32 into vector<1x4xf32>
          %283 = affine.apply affine_map<() -> (1)>()
          %284 = vector.extract %280[0, 1] : f32 from vector<1x4xf32>
          %285 = vector.insert %284, %282 [%c0_56, %283] : f32 into vector<1x4xf32>
          %286 = affine.apply affine_map<() -> (2)>()
          %287 = vector.extract %280[0, 2] : f32 from vector<1x4xf32>
          %288 = vector.insert %287, %285 [%c0_56, %286] : f32 into vector<1x4xf32>
          %289 = affine.apply affine_map<() -> (3)>()
          %290 = vector.extract %280[0, 3] : f32 from vector<1x4xf32>
          %291 = vector.insert %290, %288 [%c0_56, %289] : f32 into vector<1x4xf32>
          affine.yield %291 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %219 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %221 = vector.extract %218[0, 0] : f32 from vector<1x4xf32>
        %222 = vector.insert %221, %arg13 [%219, %220] : f32 into vector<2x8xf32>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %224 = vector.extract %218[0, 1] : f32 from vector<1x4xf32>
        %225 = vector.insert %224, %222 [%219, %223] : f32 into vector<2x8xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %227 = vector.extract %218[0, 2] : f32 from vector<1x4xf32>
        %228 = vector.insert %227, %225 [%219, %226] : f32 into vector<2x8xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %230 = vector.extract %218[0, 3] : f32 from vector<1x4xf32>
        %231 = vector.insert %230, %228 [%219, %229] : f32 into vector<2x8xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %234 = vector.extract %arg5[%232, %233] : f32 from vector<2x8xf32>
        %235 = vector.insert %234, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %237 = vector.extract %arg5[%232, %236] : f32 from vector<2x8xf32>
        %238 = vector.insert %237, %235 [0, 1] : f32 into vector<1x4xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %240 = vector.extract %arg5[%232, %239] : f32 from vector<2x8xf32>
        %241 = vector.insert %240, %238 [0, 2] : f32 into vector<1x4xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %243 = vector.extract %arg5[%232, %242] : f32 from vector<2x8xf32>
        %244 = vector.insert %243, %241 [0, 3] : f32 into vector<1x4xf32>
        %245 = arith.mulf %244, %218 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %248 = vector.extract %245[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg14 [%246, %247] : f32 into vector<2x8xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %251 = vector.extract %245[0, 1] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%246, %250] : f32 into vector<2x8xf32>
        %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %254 = vector.extract %245[0, 2] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%246, %253] : f32 into vector<2x8xf32>
        %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %257 = vector.extract %245[0, 3] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %255 [%246, %256] : f32 into vector<2x8xf32>
        %259 = arith.addf %245, %200 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %261 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %262 = vector.extract %259[0, 0] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %arg15 [%260, %261] : f32 into vector<2x8xf32>
        %264 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %265 = vector.extract %259[0, 1] : f32 from vector<1x4xf32>
        %266 = vector.insert %265, %263 [%260, %264] : f32 into vector<2x8xf32>
        %267 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %268 = vector.extract %259[0, 2] : f32 from vector<1x4xf32>
        %269 = vector.insert %268, %266 [%260, %267] : f32 into vector<2x8xf32>
        %270 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %271 = vector.extract %259[0, 3] : f32 from vector<1x4xf32>
        %272 = vector.insert %271, %269 [%260, %270] : f32 into vector<2x8xf32>
        affine.yield %213, %231, %258, %272 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_49 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %189 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %189 to %alloc_49 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_49 : memref<128x32xf16, 3>
      %190 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %200 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_48]
        %201 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_48]
        %202 = frisk.read_matrix_from_shm %alloc_49[%200, %201] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %203 = vector.bitcast %202 : vector<4xi32> to vector<8xf16>
        %204 = vector.extract %203[0] : f16 from vector<8xf16>
        %205 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_48]
        %206 = vector.extract %203[%205] : f16 from vector<8xf16>
        %207 = vector.extract %203[1] : f16 from vector<8xf16>
        %208 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_48]
        %209 = vector.extract %203[%208] : f16 from vector<8xf16>
        %210 = vector.extract %203[2] : f16 from vector<8xf16>
        %211 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_48]
        %212 = vector.extract %203[%211] : f16 from vector<8xf16>
        %213 = vector.extract %203[3] : f16 from vector<8xf16>
        %214 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_48]
        %215 = vector.extract %203[%214] : f16 from vector<8xf16>
        %216 = vector.from_elements %204, %206, %207, %209, %210, %212, %213, %215 : vector<4x2xf16>
        %217 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %218 = affine.apply affine_map<() -> (0)>()
        %219 = vector.extract %216[0, 0] : f16 from vector<4x2xf16>
        %220 = vector.insert %219, %arg12 [%217, %218] : f16 into vector<32x2xf16>
        %221 = affine.apply affine_map<() -> (1)>()
        %222 = vector.extract %216[0, 1] : f16 from vector<4x2xf16>
        %223 = vector.insert %222, %220 [%217, %221] : f16 into vector<32x2xf16>
        %224 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %225 = vector.extract %216[1, 0] : f16 from vector<4x2xf16>
        %226 = vector.insert %225, %223 [%224, %218] : f16 into vector<32x2xf16>
        %227 = affine.apply affine_map<() -> (1)>()
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %229 = vector.extract %216[1, 1] : f16 from vector<4x2xf16>
        %230 = vector.insert %229, %226 [%228, %227] : f16 into vector<32x2xf16>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %232 = vector.extract %216[2, 0] : f16 from vector<4x2xf16>
        %233 = vector.insert %232, %230 [%231, %218] : f16 into vector<32x2xf16>
        %234 = affine.apply affine_map<() -> (1)>()
        %235 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %236 = vector.extract %216[2, 1] : f16 from vector<4x2xf16>
        %237 = vector.insert %236, %233 [%235, %234] : f16 into vector<32x2xf16>
        %238 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %239 = vector.extract %216[3, 0] : f16 from vector<4x2xf16>
        %240 = vector.insert %239, %237 [%238, %218] : f16 into vector<32x2xf16>
        %241 = affine.apply affine_map<() -> (1)>()
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %243 = vector.extract %216[3, 1] : f16 from vector<4x2xf16>
        %244 = vector.insert %243, %240 [%242, %241] : f16 into vector<32x2xf16>
        affine.yield %244 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %191:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %200 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_2) -> (vector<1x4xf32>) {
          %273 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %274 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %275 = vector.extract %185[%273, %274] : f16 from vector<2x32xf16>
          %276 = vector.insert %275, %cst_55 [0, 0] : f16 into vector<1x4xf16>
          %277 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %278 = vector.extract %185[%273, %277] : f16 from vector<2x32xf16>
          %279 = vector.insert %278, %276 [0, 1] : f16 into vector<1x4xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %281 = vector.extract %185[%273, %280] : f16 from vector<2x32xf16>
          %282 = vector.insert %281, %279 [0, 2] : f16 into vector<1x4xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %284 = vector.extract %185[%273, %283] : f16 from vector<2x32xf16>
          %285 = vector.insert %284, %282 [0, 3] : f16 into vector<1x4xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %287 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %288 = vector.extract %190[%286, %287] : f16 from vector<32x2xf16>
          %289 = vector.insert %288, %cst_56 [0, 0] : f16 into vector<4x1xf16>
          %290 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %291 = vector.extract %190[%290, %287] : f16 from vector<32x2xf16>
          %292 = vector.insert %291, %289 [1, 0] : f16 into vector<4x1xf16>
          %293 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %294 = vector.extract %190[%293, %287] : f16 from vector<32x2xf16>
          %295 = vector.insert %294, %292 [2, 0] : f16 into vector<4x1xf16>
          %296 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %297 = vector.extract %190[%296, %287] : f16 from vector<32x2xf16>
          %298 = vector.insert %297, %295 [3, 0] : f16 into vector<4x1xf16>
          %299 = frisk.warp_mma_rr %285, %298, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %299 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %201 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %203 = vector.extract %200[0, 0] : f32 from vector<1x4xf32>
        %204 = vector.insert %203, %arg12 [%201, %202] : f32 into vector<2x8xf32>
        %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %206 = vector.extract %200[0, 1] : f32 from vector<1x4xf32>
        %207 = vector.insert %206, %204 [%201, %205] : f32 into vector<2x8xf32>
        %208 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %209 = vector.extract %200[0, 2] : f32 from vector<1x4xf32>
        %210 = vector.insert %209, %207 [%201, %208] : f32 into vector<2x8xf32>
        %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %212 = vector.extract %200[0, 3] : f32 from vector<1x4xf32>
        %213 = vector.insert %212, %210 [%201, %211] : f32 into vector<2x8xf32>
        %214 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %215 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %216 = vector.extract %181#1[%214, %215] : f32 from vector<2x1xf32>
        %217 = vector.insert %216, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %218 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_53) -> (vector<1x4xf32>) {
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %273 = vector.extract %217[%c0_58, %c0_59] : f32 from vector<1x1xf32>
          %274 = vector.insert %273, %cst_57 [0, 0] : f32 into vector<1x4xf32>
          %c0_60 = arith.constant 0 : index
          %c0_61 = arith.constant 0 : index
          %275 = vector.extract %217[%c0_60, %c0_61] : f32 from vector<1x1xf32>
          %276 = vector.insert %275, %274 [0, 1] : f32 into vector<1x4xf32>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %277 = vector.extract %217[%c0_62, %c0_63] : f32 from vector<1x1xf32>
          %278 = vector.insert %277, %276 [0, 2] : f32 into vector<1x4xf32>
          %c0_64 = arith.constant 0 : index
          %c0_65 = arith.constant 0 : index
          %279 = vector.extract %217[%c0_64, %c0_65] : f32 from vector<1x1xf32>
          %280 = vector.insert %279, %278 [0, 3] : f32 into vector<1x4xf32>
          %281 = vector.extract %280[0, 0] : f32 from vector<1x4xf32>
          %282 = vector.insert %281, %arg17 [%c0_56, %c0_55] : f32 into vector<1x4xf32>
          %283 = affine.apply affine_map<() -> (1)>()
          %284 = vector.extract %280[0, 1] : f32 from vector<1x4xf32>
          %285 = vector.insert %284, %282 [%c0_56, %283] : f32 into vector<1x4xf32>
          %286 = affine.apply affine_map<() -> (2)>()
          %287 = vector.extract %280[0, 2] : f32 from vector<1x4xf32>
          %288 = vector.insert %287, %285 [%c0_56, %286] : f32 into vector<1x4xf32>
          %289 = affine.apply affine_map<() -> (3)>()
          %290 = vector.extract %280[0, 3] : f32 from vector<1x4xf32>
          %291 = vector.insert %290, %288 [%c0_56, %289] : f32 into vector<1x4xf32>
          affine.yield %291 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %219 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %221 = vector.extract %218[0, 0] : f32 from vector<1x4xf32>
        %222 = vector.insert %221, %arg13 [%219, %220] : f32 into vector<2x8xf32>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %224 = vector.extract %218[0, 1] : f32 from vector<1x4xf32>
        %225 = vector.insert %224, %222 [%219, %223] : f32 into vector<2x8xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %227 = vector.extract %218[0, 2] : f32 from vector<1x4xf32>
        %228 = vector.insert %227, %225 [%219, %226] : f32 into vector<2x8xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %230 = vector.extract %218[0, 3] : f32 from vector<1x4xf32>
        %231 = vector.insert %230, %228 [%219, %229] : f32 into vector<2x8xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %234 = vector.extract %arg8[%232, %233] : f32 from vector<2x8xf32>
        %235 = vector.insert %234, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %237 = vector.extract %arg8[%232, %236] : f32 from vector<2x8xf32>
        %238 = vector.insert %237, %235 [0, 1] : f32 into vector<1x4xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %240 = vector.extract %arg8[%232, %239] : f32 from vector<2x8xf32>
        %241 = vector.insert %240, %238 [0, 2] : f32 into vector<1x4xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %243 = vector.extract %arg8[%232, %242] : f32 from vector<2x8xf32>
        %244 = vector.insert %243, %241 [0, 3] : f32 into vector<1x4xf32>
        %245 = arith.mulf %244, %218 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %248 = vector.extract %245[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg14 [%246, %247] : f32 into vector<2x8xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %251 = vector.extract %245[0, 1] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%246, %250] : f32 into vector<2x8xf32>
        %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %254 = vector.extract %245[0, 2] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%246, %253] : f32 into vector<2x8xf32>
        %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %257 = vector.extract %245[0, 3] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %255 [%246, %256] : f32 into vector<2x8xf32>
        %259 = arith.addf %245, %200 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %261 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %262 = vector.extract %259[0, 0] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %arg15 [%260, %261] : f32 into vector<2x8xf32>
        %264 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %265 = vector.extract %259[0, 1] : f32 from vector<1x4xf32>
        %266 = vector.insert %265, %263 [%260, %264] : f32 into vector<2x8xf32>
        %267 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %268 = vector.extract %259[0, 2] : f32 from vector<1x4xf32>
        %269 = vector.insert %268, %266 [%260, %267] : f32 into vector<2x8xf32>
        %270 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %271 = vector.extract %259[0, 3] : f32 from vector<1x4xf32>
        %272 = vector.insert %271, %269 [%260, %270] : f32 into vector<2x8xf32>
        affine.yield %213, %231, %258, %272 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_50 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %192 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %192 to %alloc_50 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_50 : memref<128x32xf16, 3>
      %193 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %200 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_48]
        %201 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_48]
        %202 = frisk.read_matrix_from_shm %alloc_50[%200, %201] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %203 = vector.bitcast %202 : vector<4xi32> to vector<8xf16>
        %204 = vector.extract %203[0] : f16 from vector<8xf16>
        %205 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_48]
        %206 = vector.extract %203[%205] : f16 from vector<8xf16>
        %207 = vector.extract %203[1] : f16 from vector<8xf16>
        %208 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_48]
        %209 = vector.extract %203[%208] : f16 from vector<8xf16>
        %210 = vector.extract %203[2] : f16 from vector<8xf16>
        %211 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_48]
        %212 = vector.extract %203[%211] : f16 from vector<8xf16>
        %213 = vector.extract %203[3] : f16 from vector<8xf16>
        %214 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_48]
        %215 = vector.extract %203[%214] : f16 from vector<8xf16>
        %216 = vector.from_elements %204, %206, %207, %209, %210, %212, %213, %215 : vector<4x2xf16>
        %217 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %218 = affine.apply affine_map<() -> (0)>()
        %219 = vector.extract %216[0, 0] : f16 from vector<4x2xf16>
        %220 = vector.insert %219, %arg12 [%217, %218] : f16 into vector<32x2xf16>
        %221 = affine.apply affine_map<() -> (1)>()
        %222 = vector.extract %216[0, 1] : f16 from vector<4x2xf16>
        %223 = vector.insert %222, %220 [%217, %221] : f16 into vector<32x2xf16>
        %224 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %225 = vector.extract %216[1, 0] : f16 from vector<4x2xf16>
        %226 = vector.insert %225, %223 [%224, %218] : f16 into vector<32x2xf16>
        %227 = affine.apply affine_map<() -> (1)>()
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %229 = vector.extract %216[1, 1] : f16 from vector<4x2xf16>
        %230 = vector.insert %229, %226 [%228, %227] : f16 into vector<32x2xf16>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %232 = vector.extract %216[2, 0] : f16 from vector<4x2xf16>
        %233 = vector.insert %232, %230 [%231, %218] : f16 into vector<32x2xf16>
        %234 = affine.apply affine_map<() -> (1)>()
        %235 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %236 = vector.extract %216[2, 1] : f16 from vector<4x2xf16>
        %237 = vector.insert %236, %233 [%235, %234] : f16 into vector<32x2xf16>
        %238 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %239 = vector.extract %216[3, 0] : f16 from vector<4x2xf16>
        %240 = vector.insert %239, %237 [%238, %218] : f16 into vector<32x2xf16>
        %241 = affine.apply affine_map<() -> (1)>()
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %243 = vector.extract %216[3, 1] : f16 from vector<4x2xf16>
        %244 = vector.insert %243, %240 [%242, %241] : f16 into vector<32x2xf16>
        affine.yield %244 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %194:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %200 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_2) -> (vector<1x4xf32>) {
          %273 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %274 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %275 = vector.extract %185[%273, %274] : f16 from vector<2x32xf16>
          %276 = vector.insert %275, %cst_55 [0, 0] : f16 into vector<1x4xf16>
          %277 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %278 = vector.extract %185[%273, %277] : f16 from vector<2x32xf16>
          %279 = vector.insert %278, %276 [0, 1] : f16 into vector<1x4xf16>
          %280 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %281 = vector.extract %185[%273, %280] : f16 from vector<2x32xf16>
          %282 = vector.insert %281, %279 [0, 2] : f16 into vector<1x4xf16>
          %283 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %284 = vector.extract %185[%273, %283] : f16 from vector<2x32xf16>
          %285 = vector.insert %284, %282 [0, 3] : f16 into vector<1x4xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %287 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %288 = vector.extract %193[%286, %287] : f16 from vector<32x2xf16>
          %289 = vector.insert %288, %cst_56 [0, 0] : f16 into vector<4x1xf16>
          %290 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %291 = vector.extract %193[%290, %287] : f16 from vector<32x2xf16>
          %292 = vector.insert %291, %289 [1, 0] : f16 into vector<4x1xf16>
          %293 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %294 = vector.extract %193[%293, %287] : f16 from vector<32x2xf16>
          %295 = vector.insert %294, %292 [2, 0] : f16 into vector<4x1xf16>
          %296 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %297 = vector.extract %193[%296, %287] : f16 from vector<32x2xf16>
          %298 = vector.insert %297, %295 [3, 0] : f16 into vector<4x1xf16>
          %299 = frisk.warp_mma_rr %285, %298, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %299 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %201 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %203 = vector.extract %200[0, 0] : f32 from vector<1x4xf32>
        %204 = vector.insert %203, %arg12 [%201, %202] : f32 into vector<2x8xf32>
        %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %206 = vector.extract %200[0, 1] : f32 from vector<1x4xf32>
        %207 = vector.insert %206, %204 [%201, %205] : f32 into vector<2x8xf32>
        %208 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %209 = vector.extract %200[0, 2] : f32 from vector<1x4xf32>
        %210 = vector.insert %209, %207 [%201, %208] : f32 into vector<2x8xf32>
        %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %212 = vector.extract %200[0, 3] : f32 from vector<1x4xf32>
        %213 = vector.insert %212, %210 [%201, %211] : f32 into vector<2x8xf32>
        %214 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %215 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %216 = vector.extract %181#1[%214, %215] : f32 from vector<2x1xf32>
        %217 = vector.insert %216, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %218 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_53) -> (vector<1x4xf32>) {
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %273 = vector.extract %217[%c0_58, %c0_59] : f32 from vector<1x1xf32>
          %274 = vector.insert %273, %cst_57 [0, 0] : f32 into vector<1x4xf32>
          %c0_60 = arith.constant 0 : index
          %c0_61 = arith.constant 0 : index
          %275 = vector.extract %217[%c0_60, %c0_61] : f32 from vector<1x1xf32>
          %276 = vector.insert %275, %274 [0, 1] : f32 into vector<1x4xf32>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %277 = vector.extract %217[%c0_62, %c0_63] : f32 from vector<1x1xf32>
          %278 = vector.insert %277, %276 [0, 2] : f32 into vector<1x4xf32>
          %c0_64 = arith.constant 0 : index
          %c0_65 = arith.constant 0 : index
          %279 = vector.extract %217[%c0_64, %c0_65] : f32 from vector<1x1xf32>
          %280 = vector.insert %279, %278 [0, 3] : f32 into vector<1x4xf32>
          %281 = vector.extract %280[0, 0] : f32 from vector<1x4xf32>
          %282 = vector.insert %281, %arg17 [%c0_56, %c0_55] : f32 into vector<1x4xf32>
          %283 = affine.apply affine_map<() -> (1)>()
          %284 = vector.extract %280[0, 1] : f32 from vector<1x4xf32>
          %285 = vector.insert %284, %282 [%c0_56, %283] : f32 into vector<1x4xf32>
          %286 = affine.apply affine_map<() -> (2)>()
          %287 = vector.extract %280[0, 2] : f32 from vector<1x4xf32>
          %288 = vector.insert %287, %285 [%c0_56, %286] : f32 into vector<1x4xf32>
          %289 = affine.apply affine_map<() -> (3)>()
          %290 = vector.extract %280[0, 3] : f32 from vector<1x4xf32>
          %291 = vector.insert %290, %288 [%c0_56, %289] : f32 into vector<1x4xf32>
          affine.yield %291 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %219 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %221 = vector.extract %218[0, 0] : f32 from vector<1x4xf32>
        %222 = vector.insert %221, %arg13 [%219, %220] : f32 into vector<2x8xf32>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %224 = vector.extract %218[0, 1] : f32 from vector<1x4xf32>
        %225 = vector.insert %224, %222 [%219, %223] : f32 into vector<2x8xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %227 = vector.extract %218[0, 2] : f32 from vector<1x4xf32>
        %228 = vector.insert %227, %225 [%219, %226] : f32 into vector<2x8xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %230 = vector.extract %218[0, 3] : f32 from vector<1x4xf32>
        %231 = vector.insert %230, %228 [%219, %229] : f32 into vector<2x8xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %234 = vector.extract %arg9[%232, %233] : f32 from vector<2x8xf32>
        %235 = vector.insert %234, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %237 = vector.extract %arg9[%232, %236] : f32 from vector<2x8xf32>
        %238 = vector.insert %237, %235 [0, 1] : f32 into vector<1x4xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %240 = vector.extract %arg9[%232, %239] : f32 from vector<2x8xf32>
        %241 = vector.insert %240, %238 [0, 2] : f32 into vector<1x4xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %243 = vector.extract %arg9[%232, %242] : f32 from vector<2x8xf32>
        %244 = vector.insert %243, %241 [0, 3] : f32 into vector<1x4xf32>
        %245 = arith.mulf %244, %218 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %248 = vector.extract %245[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg14 [%246, %247] : f32 into vector<2x8xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %251 = vector.extract %245[0, 1] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%246, %250] : f32 into vector<2x8xf32>
        %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %254 = vector.extract %245[0, 2] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%246, %253] : f32 into vector<2x8xf32>
        %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %257 = vector.extract %245[0, 3] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %255 [%246, %256] : f32 into vector<2x8xf32>
        %259 = arith.addf %245, %200 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %261 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %262 = vector.extract %259[0, 0] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %arg15 [%260, %261] : f32 into vector<2x8xf32>
        %264 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %265 = vector.extract %259[0, 1] : f32 from vector<1x4xf32>
        %266 = vector.insert %265, %263 [%260, %264] : f32 into vector<2x8xf32>
        %267 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %268 = vector.extract %259[0, 2] : f32 from vector<1x4xf32>
        %269 = vector.insert %268, %266 [%260, %267] : f32 into vector<2x8xf32>
        %270 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %271 = vector.extract %259[0, 3] : f32 from vector<1x4xf32>
        %272 = vector.insert %271, %269 [%260, %270] : f32 into vector<2x8xf32>
        affine.yield %213, %231, %258, %272 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %alloc_51 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %195 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %195 to %alloc_51 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_51 : memref<128x32xf16, 3>
      %196 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %200 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_48]
        %201 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_48]
        %202 = frisk.read_matrix_from_shm %alloc_51[%200, %201] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %203 = vector.bitcast %202 : vector<4xi32> to vector<8xf16>
        %204 = vector.extract %203[0] : f16 from vector<8xf16>
        %205 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_48]
        %206 = vector.extract %203[%205] : f16 from vector<8xf16>
        %207 = vector.extract %203[1] : f16 from vector<8xf16>
        %208 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_48]
        %209 = vector.extract %203[%208] : f16 from vector<8xf16>
        %210 = vector.extract %203[2] : f16 from vector<8xf16>
        %211 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_48]
        %212 = vector.extract %203[%211] : f16 from vector<8xf16>
        %213 = vector.extract %203[3] : f16 from vector<8xf16>
        %214 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_48]
        %215 = vector.extract %203[%214] : f16 from vector<8xf16>
        %216 = vector.from_elements %204, %206, %207, %209, %210, %212, %213, %215 : vector<4x2xf16>
        %217 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %218 = affine.apply affine_map<() -> (0)>()
        %219 = vector.extract %216[0, 0] : f16 from vector<4x2xf16>
        %220 = vector.insert %219, %arg12 [%217, %218] : f16 into vector<32x2xf16>
        %221 = affine.apply affine_map<() -> (1)>()
        %222 = vector.extract %216[0, 1] : f16 from vector<4x2xf16>
        %223 = vector.insert %222, %220 [%217, %221] : f16 into vector<32x2xf16>
        %224 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %225 = vector.extract %216[1, 0] : f16 from vector<4x2xf16>
        %226 = vector.insert %225, %223 [%224, %218] : f16 into vector<32x2xf16>
        %227 = affine.apply affine_map<() -> (1)>()
        %228 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %229 = vector.extract %216[1, 1] : f16 from vector<4x2xf16>
        %230 = vector.insert %229, %226 [%228, %227] : f16 into vector<32x2xf16>
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %232 = vector.extract %216[2, 0] : f16 from vector<4x2xf16>
        %233 = vector.insert %232, %230 [%231, %218] : f16 into vector<32x2xf16>
        %234 = affine.apply affine_map<() -> (1)>()
        %235 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %236 = vector.extract %216[2, 1] : f16 from vector<4x2xf16>
        %237 = vector.insert %236, %233 [%235, %234] : f16 into vector<32x2xf16>
        %238 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %239 = vector.extract %216[3, 0] : f16 from vector<4x2xf16>
        %240 = vector.insert %239, %237 [%238, %218] : f16 into vector<32x2xf16>
        %241 = affine.apply affine_map<() -> (1)>()
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %243 = vector.extract %216[3, 1] : f16 from vector<4x2xf16>
        %244 = vector.insert %243, %240 [%242, %241] : f16 into vector<32x2xf16>
        affine.yield %244 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %197:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %200 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_2) -> (vector<1x4xf32>) {
          %299 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %300 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %301 = vector.extract %185[%299, %300] : f16 from vector<2x32xf16>
          %302 = vector.insert %301, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %303 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %304 = vector.extract %185[%299, %303] : f16 from vector<2x32xf16>
          %305 = vector.insert %304, %302 [0, 1] : f16 into vector<1x4xf16>
          %306 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %307 = vector.extract %185[%299, %306] : f16 from vector<2x32xf16>
          %308 = vector.insert %307, %305 [0, 2] : f16 into vector<1x4xf16>
          %309 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %310 = vector.extract %185[%299, %309] : f16 from vector<2x32xf16>
          %311 = vector.insert %310, %308 [0, 3] : f16 into vector<1x4xf16>
          %312 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %313 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_57 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %314 = vector.extract %196[%312, %313] : f16 from vector<32x2xf16>
          %315 = vector.insert %314, %cst_57 [0, 0] : f16 into vector<4x1xf16>
          %316 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %317 = vector.extract %196[%316, %313] : f16 from vector<32x2xf16>
          %318 = vector.insert %317, %315 [1, 0] : f16 into vector<4x1xf16>
          %319 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %320 = vector.extract %196[%319, %313] : f16 from vector<32x2xf16>
          %321 = vector.insert %320, %318 [2, 0] : f16 into vector<4x1xf16>
          %322 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %323 = vector.extract %196[%322, %313] : f16 from vector<32x2xf16>
          %324 = vector.insert %323, %321 [3, 0] : f16 into vector<4x1xf16>
          %325 = frisk.warp_mma_rr %311, %324, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %325 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %201 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %202 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %203 = vector.extract %200[0, 0] : f32 from vector<1x4xf32>
        %204 = vector.insert %203, %arg12 [%201, %202] : f32 into vector<2x8xf32>
        %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %206 = vector.extract %200[0, 1] : f32 from vector<1x4xf32>
        %207 = vector.insert %206, %204 [%201, %205] : f32 into vector<2x8xf32>
        %208 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %209 = vector.extract %200[0, 2] : f32 from vector<1x4xf32>
        %210 = vector.insert %209, %207 [%201, %208] : f32 into vector<2x8xf32>
        %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %212 = vector.extract %200[0, 3] : f32 from vector<1x4xf32>
        %213 = vector.insert %212, %210 [%201, %211] : f32 into vector<2x8xf32>
        %214 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %215 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %216 = vector.extract %181#1[%214, %215] : f32 from vector<2x1xf32>
        %217 = vector.insert %216, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %218 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_53) -> (vector<1x4xf32>) {
          %c0_56 = arith.constant 0 : index
          %c0_57 = arith.constant 0 : index
          %cst_58 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_59 = arith.constant 0 : index
          %c0_60 = arith.constant 0 : index
          %299 = vector.extract %217[%c0_59, %c0_60] : f32 from vector<1x1xf32>
          %300 = vector.insert %299, %cst_58 [0, 0] : f32 into vector<1x4xf32>
          %c0_61 = arith.constant 0 : index
          %c0_62 = arith.constant 0 : index
          %301 = vector.extract %217[%c0_61, %c0_62] : f32 from vector<1x1xf32>
          %302 = vector.insert %301, %300 [0, 1] : f32 into vector<1x4xf32>
          %c0_63 = arith.constant 0 : index
          %c0_64 = arith.constant 0 : index
          %303 = vector.extract %217[%c0_63, %c0_64] : f32 from vector<1x1xf32>
          %304 = vector.insert %303, %302 [0, 2] : f32 into vector<1x4xf32>
          %c0_65 = arith.constant 0 : index
          %c0_66 = arith.constant 0 : index
          %305 = vector.extract %217[%c0_65, %c0_66] : f32 from vector<1x1xf32>
          %306 = vector.insert %305, %304 [0, 3] : f32 into vector<1x4xf32>
          %307 = vector.extract %306[0, 0] : f32 from vector<1x4xf32>
          %308 = vector.insert %307, %arg18 [%c0_57, %c0_56] : f32 into vector<1x4xf32>
          %309 = affine.apply affine_map<() -> (1)>()
          %310 = vector.extract %306[0, 1] : f32 from vector<1x4xf32>
          %311 = vector.insert %310, %308 [%c0_57, %309] : f32 into vector<1x4xf32>
          %312 = affine.apply affine_map<() -> (2)>()
          %313 = vector.extract %306[0, 2] : f32 from vector<1x4xf32>
          %314 = vector.insert %313, %311 [%c0_57, %312] : f32 into vector<1x4xf32>
          %315 = affine.apply affine_map<() -> (3)>()
          %316 = vector.extract %306[0, 3] : f32 from vector<1x4xf32>
          %317 = vector.insert %316, %314 [%c0_57, %315] : f32 into vector<1x4xf32>
          affine.yield %317 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %219 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %221 = vector.extract %218[0, 0] : f32 from vector<1x4xf32>
        %222 = vector.insert %221, %arg13 [%219, %220] : f32 into vector<2x8xf32>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %224 = vector.extract %218[0, 1] : f32 from vector<1x4xf32>
        %225 = vector.insert %224, %222 [%219, %223] : f32 into vector<2x8xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %227 = vector.extract %218[0, 2] : f32 from vector<1x4xf32>
        %228 = vector.insert %227, %225 [%219, %226] : f32 into vector<2x8xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %230 = vector.extract %218[0, 3] : f32 from vector<1x4xf32>
        %231 = vector.insert %230, %228 [%219, %229] : f32 into vector<2x8xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %234 = vector.extract %arg10[%232, %233] : f32 from vector<2x8xf32>
        %235 = vector.insert %234, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %237 = vector.extract %arg10[%232, %236] : f32 from vector<2x8xf32>
        %238 = vector.insert %237, %235 [0, 1] : f32 into vector<1x4xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %240 = vector.extract %arg10[%232, %239] : f32 from vector<2x8xf32>
        %241 = vector.insert %240, %238 [0, 2] : f32 into vector<1x4xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %243 = vector.extract %arg10[%232, %242] : f32 from vector<2x8xf32>
        %244 = vector.insert %243, %241 [0, 3] : f32 into vector<1x4xf32>
        %245 = arith.mulf %244, %218 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %248 = vector.extract %245[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg14 [%246, %247] : f32 into vector<2x8xf32>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %251 = vector.extract %245[0, 1] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%246, %250] : f32 into vector<2x8xf32>
        %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %254 = vector.extract %245[0, 2] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%246, %253] : f32 into vector<2x8xf32>
        %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %257 = vector.extract %245[0, 3] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %255 [%246, %256] : f32 into vector<2x8xf32>
        %259 = arith.addf %245, %200 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %260 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %261 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %262 = vector.extract %259[0, 0] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %arg15 [%260, %261] : f32 into vector<2x8xf32>
        %264 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %265 = vector.extract %259[0, 1] : f32 from vector<1x4xf32>
        %266 = vector.insert %265, %263 [%260, %264] : f32 into vector<2x8xf32>
        %267 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %268 = vector.extract %259[0, 2] : f32 from vector<1x4xf32>
        %269 = vector.insert %268, %266 [%260, %267] : f32 into vector<2x8xf32>
        %270 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %271 = vector.extract %259[0, 3] : f32 from vector<1x4xf32>
        %272 = vector.insert %271, %269 [%260, %270] : f32 into vector<2x8xf32>
        %273 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %274 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_55 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %275 = vector.extract %188#3[%273, %274] : f32 from vector<2x8xf32>
        %276 = vector.insert %275, %cst_55 [0, 0] : f32 into vector<1x4xf32>
        %277 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %278 = vector.extract %188#3[%273, %277] : f32 from vector<2x8xf32>
        %279 = vector.insert %278, %276 [0, 1] : f32 into vector<1x4xf32>
        %280 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %281 = vector.extract %188#3[%273, %280] : f32 from vector<2x8xf32>
        %282 = vector.insert %281, %279 [0, 2] : f32 into vector<1x4xf32>
        %283 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %284 = vector.extract %188#3[%273, %283] : f32 from vector<2x8xf32>
        %285 = vector.insert %284, %282 [0, 3] : f32 into vector<1x4xf32>
        %286 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %287 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %288 = vector.extract %285[0, 0] : f32 from vector<1x4xf32>
        %289 = vector.insert %288, %arg16 [%286, %287] : f32 into vector<2x8xf32>
        %290 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %291 = vector.extract %285[0, 1] : f32 from vector<1x4xf32>
        %292 = vector.insert %291, %289 [%286, %290] : f32 into vector<2x8xf32>
        %293 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %294 = vector.extract %285[0, 2] : f32 from vector<1x4xf32>
        %295 = vector.insert %294, %292 [%286, %293] : f32 into vector<2x8xf32>
        %296 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %297 = vector.extract %285[0, 3] : f32 from vector<1x4xf32>
        %298 = vector.insert %297, %295 [%286, %296] : f32 into vector<2x8xf32>
        affine.yield %213, %231, %258, %272, %298 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %198:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %200 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %201 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %202 = vector.extract %184[%200, %201] : f32 from vector<2x1xf32>
        %203 = vector.insert %202, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %204 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %205 = affine.apply affine_map<() -> (0)>()
        %206 = vector.extract %203[0, 0] : f32 from vector<1x1xf32>
        %207 = vector.insert %206, %arg12 [%204, %205] : f32 into vector<2x1xf32>
        %208 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %209 = affine.apply affine_map<() -> (0)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %210 = vector.extract %179[%208, %209] : f32 from vector<2x1xf32>
        %211 = vector.insert %210, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %212 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %213 = affine.apply affine_map<() -> (0)>()
        %214 = vector.extract %211[0, 0] : f32 from vector<1x1xf32>
        %215 = vector.insert %214, %arg13 [%212, %213] : f32 into vector<2x1xf32>
        affine.yield %207, %215 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %199:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %200 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %202 = vector.extract %191#3[%200, %201] : f32 from vector<2x8xf32>
        %203 = vector.insert %202, %cst_52 [0, 0] : f32 into vector<1x4xf32>
        %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %205 = vector.extract %191#3[%200, %204] : f32 from vector<2x8xf32>
        %206 = vector.insert %205, %203 [0, 1] : f32 into vector<1x4xf32>
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %208 = vector.extract %191#3[%200, %207] : f32 from vector<2x8xf32>
        %209 = vector.insert %208, %206 [0, 2] : f32 into vector<1x4xf32>
        %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %211 = vector.extract %191#3[%200, %210] : f32 from vector<2x8xf32>
        %212 = vector.insert %211, %209 [0, 3] : f32 into vector<1x4xf32>
        %213 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %215 = vector.extract %212[0, 0] : f32 from vector<1x4xf32>
        %216 = vector.insert %215, %arg12 [%213, %214] : f32 into vector<2x8xf32>
        %217 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %218 = vector.extract %212[0, 1] : f32 from vector<1x4xf32>
        %219 = vector.insert %218, %216 [%213, %217] : f32 into vector<2x8xf32>
        %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %221 = vector.extract %212[0, 2] : f32 from vector<1x4xf32>
        %222 = vector.insert %221, %219 [%213, %220] : f32 into vector<2x8xf32>
        %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %224 = vector.extract %212[0, 3] : f32 from vector<1x4xf32>
        %225 = vector.insert %224, %222 [%213, %223] : f32 into vector<2x8xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %227 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %228 = vector.extract %194#3[%226, %227] : f32 from vector<2x8xf32>
        %229 = vector.insert %228, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %231 = vector.extract %194#3[%226, %230] : f32 from vector<2x8xf32>
        %232 = vector.insert %231, %229 [0, 1] : f32 into vector<1x4xf32>
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %234 = vector.extract %194#3[%226, %233] : f32 from vector<2x8xf32>
        %235 = vector.insert %234, %232 [0, 2] : f32 into vector<1x4xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %237 = vector.extract %194#3[%226, %236] : f32 from vector<2x8xf32>
        %238 = vector.insert %237, %235 [0, 3] : f32 into vector<1x4xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %241 = vector.extract %238[0, 0] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %arg13 [%239, %240] : f32 into vector<2x8xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %244 = vector.extract %238[0, 1] : f32 from vector<1x4xf32>
        %245 = vector.insert %244, %242 [%239, %243] : f32 into vector<2x8xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %247 = vector.extract %238[0, 2] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %245 [%239, %246] : f32 into vector<2x8xf32>
        %249 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %250 = vector.extract %238[0, 3] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %248 [%239, %249] : f32 into vector<2x8xf32>
        %252 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %253 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %254 = vector.extract %197#3[%252, %253] : f32 from vector<2x8xf32>
        %255 = vector.insert %254, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %257 = vector.extract %197#3[%252, %256] : f32 from vector<2x8xf32>
        %258 = vector.insert %257, %255 [0, 1] : f32 into vector<1x4xf32>
        %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %260 = vector.extract %197#3[%252, %259] : f32 from vector<2x8xf32>
        %261 = vector.insert %260, %258 [0, 2] : f32 into vector<1x4xf32>
        %262 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %263 = vector.extract %197#3[%252, %262] : f32 from vector<2x8xf32>
        %264 = vector.insert %263, %261 [0, 3] : f32 into vector<1x4xf32>
        %265 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %266 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %267 = vector.extract %264[0, 0] : f32 from vector<1x4xf32>
        %268 = vector.insert %267, %arg14 [%265, %266] : f32 into vector<2x8xf32>
        %269 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %270 = vector.extract %264[0, 1] : f32 from vector<1x4xf32>
        %271 = vector.insert %270, %268 [%265, %269] : f32 into vector<2x8xf32>
        %272 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %273 = vector.extract %264[0, 2] : f32 from vector<1x4xf32>
        %274 = vector.insert %273, %271 [%265, %272] : f32 into vector<2x8xf32>
        %275 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %276 = vector.extract %264[0, 3] : f32 from vector<1x4xf32>
        %277 = vector.insert %276, %274 [%265, %275] : f32 into vector<2x8xf32>
        affine.yield %225, %251, %277 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %197#4, %198#0, %198#1, %199#0, %199#1, %199#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %thread_id_x = gpu.thread_id x
    %134 = arith.index_cast %thread_id_x : index to i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %135 = arith.andi %134, %c16_i32 : i32
    %136 = arith.cmpi ne, %135, %c0_i32 : i32
    %c32_i32 = arith.constant 32 : i32
    %137 = arith.andi %134, %c32_i32 : i32
    %138 = arith.cmpi ne, %137, %c0_i32 : i32
    %cst_19 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %139 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_19) -> (vector<2x8xf32>) {
      %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %169 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %170 = vector.extract %133#5[%169, %168] : f32 from vector<2x8xf32>
      %171 = vector.insert %170, %cst_39 [0, 0] : f32 into vector<1x4xf32>
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %173 = vector.extract %133#5[%169, %172] : f32 from vector<2x8xf32>
      %174 = vector.insert %173, %171 [0, 1] : f32 into vector<1x4xf32>
      %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %176 = vector.extract %133#5[%169, %175] : f32 from vector<2x8xf32>
      %177 = vector.insert %176, %174 [0, 2] : f32 into vector<1x4xf32>
      %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %179 = vector.extract %133#5[%169, %178] : f32 from vector<2x8xf32>
      %180 = vector.insert %179, %177 [0, 3] : f32 into vector<1x4xf32>
      %181 = vector.extract %180[0, 0] : f32 from vector<1x4xf32>
      %182 = vector.extract %180[0, 1] : f32 from vector<1x4xf32>
      %183 = vector.extract %180[0, 2] : f32 from vector<1x4xf32>
      %184 = vector.extract %180[0, 3] : f32 from vector<1x4xf32>
      %185 = arith.select %136, %181, %182 : f32
      %shuffleResult, %valid = gpu.shuffle xor %185, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %186 = arith.select %136, %shuffleResult, %181 : f32
      %187 = arith.select %136, %182, %shuffleResult : f32
      %188 = arith.select %136, %183, %184 : f32
      %shuffleResult_40, %valid_41 = gpu.shuffle xor %188, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %189 = arith.select %136, %shuffleResult_40, %183 : f32
      %190 = arith.select %136, %184, %shuffleResult_40 : f32
      %191 = arith.select %138, %186, %189 : f32
      %shuffleResult_42, %valid_43 = gpu.shuffle xor %191, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %192 = arith.select %138, %shuffleResult_42, %186 : f32
      %193 = arith.select %138, %189, %shuffleResult_42 : f32
      %194 = arith.select %138, %187, %190 : f32
      %shuffleResult_44, %valid_45 = gpu.shuffle xor %194, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %195 = arith.select %138, %shuffleResult_44, %187 : f32
      %196 = arith.select %138, %190, %shuffleResult_44 : f32
      %197 = vector.insert %192, %180 [0, 0] : f32 into vector<1x4xf32>
      %198 = vector.insert %195, %197 [0, 1] : f32 into vector<1x4xf32>
      %199 = vector.insert %193, %198 [0, 2] : f32 into vector<1x4xf32>
      %200 = vector.insert %196, %199 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
      %201 = vector.extract %200[0, 0] : f32 from vector<1x4xf32>
      %202 = vector.insert %201, %arg5 [%169, %168] : f32 into vector<2x8xf32>
      %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %204 = vector.extract %200[0, 1] : f32 from vector<1x4xf32>
      %205 = vector.insert %204, %202 [%169, %203] : f32 into vector<2x8xf32>
      %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %207 = vector.extract %200[0, 2] : f32 from vector<1x4xf32>
      %208 = vector.insert %207, %205 [%169, %206] : f32 into vector<2x8xf32>
      %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %210 = vector.extract %200[0, 3] : f32 from vector<1x4xf32>
      %211 = vector.insert %210, %208 [%169, %209] : f32 into vector<2x8xf32>
      affine.yield %211 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_20 = gpu.thread_id x
    %140 = arith.index_cast %thread_id_x_20 : index to i32
    %c0_i32_21 = arith.constant 0 : i32
    %c64_i32_22 = arith.constant 64 : i32
    %c16_i32_23 = arith.constant 16 : i32
    %141 = arith.andi %140, %c16_i32_23 : i32
    %142 = arith.cmpi ne, %141, %c0_i32_21 : i32
    %c32_i32_24 = arith.constant 32 : i32
    %143 = arith.andi %140, %c32_i32_24 : i32
    %144 = arith.cmpi ne, %143, %c0_i32_21 : i32
    %cst_25 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %145 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_25) -> (vector<2x8xf32>) {
      %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %169 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %170 = vector.extract %133#4[%169, %168] : f32 from vector<2x8xf32>
      %171 = vector.insert %170, %cst_39 [0, 0] : f32 into vector<1x4xf32>
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %173 = vector.extract %133#4[%169, %172] : f32 from vector<2x8xf32>
      %174 = vector.insert %173, %171 [0, 1] : f32 into vector<1x4xf32>
      %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %176 = vector.extract %133#4[%169, %175] : f32 from vector<2x8xf32>
      %177 = vector.insert %176, %174 [0, 2] : f32 into vector<1x4xf32>
      %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %179 = vector.extract %133#4[%169, %178] : f32 from vector<2x8xf32>
      %180 = vector.insert %179, %177 [0, 3] : f32 into vector<1x4xf32>
      %181 = vector.extract %180[0, 0] : f32 from vector<1x4xf32>
      %182 = vector.extract %180[0, 1] : f32 from vector<1x4xf32>
      %183 = vector.extract %180[0, 2] : f32 from vector<1x4xf32>
      %184 = vector.extract %180[0, 3] : f32 from vector<1x4xf32>
      %185 = arith.select %142, %181, %182 : f32
      %shuffleResult, %valid = gpu.shuffle xor %185, %c16_i32_23, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
      %186 = arith.select %142, %shuffleResult, %181 : f32
      %187 = arith.select %142, %182, %shuffleResult : f32
      %188 = arith.select %142, %183, %184 : f32
      %shuffleResult_40, %valid_41 = gpu.shuffle xor %188, %c16_i32_23, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
      %189 = arith.select %142, %shuffleResult_40, %183 : f32
      %190 = arith.select %142, %184, %shuffleResult_40 : f32
      %191 = arith.select %144, %186, %189 : f32
      %shuffleResult_42, %valid_43 = gpu.shuffle xor %191, %c32_i32_24, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
      %192 = arith.select %144, %shuffleResult_42, %186 : f32
      %193 = arith.select %144, %189, %shuffleResult_42 : f32
      %194 = arith.select %144, %187, %190 : f32
      %shuffleResult_44, %valid_45 = gpu.shuffle xor %194, %c32_i32_24, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
      %195 = arith.select %144, %shuffleResult_44, %187 : f32
      %196 = arith.select %144, %190, %shuffleResult_44 : f32
      %197 = vector.insert %192, %180 [0, 0] : f32 into vector<1x4xf32>
      %198 = vector.insert %195, %197 [0, 1] : f32 into vector<1x4xf32>
      %199 = vector.insert %193, %198 [0, 2] : f32 into vector<1x4xf32>
      %200 = vector.insert %196, %199 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
      %201 = vector.extract %200[0, 0] : f32 from vector<1x4xf32>
      %202 = vector.insert %201, %arg5 [%169, %168] : f32 into vector<2x8xf32>
      %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %204 = vector.extract %200[0, 1] : f32 from vector<1x4xf32>
      %205 = vector.insert %204, %202 [%169, %203] : f32 into vector<2x8xf32>
      %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %207 = vector.extract %200[0, 2] : f32 from vector<1x4xf32>
      %208 = vector.insert %207, %205 [%169, %206] : f32 into vector<2x8xf32>
      %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %210 = vector.extract %200[0, 3] : f32 from vector<1x4xf32>
      %211 = vector.insert %210, %208 [%169, %209] : f32 into vector<2x8xf32>
      affine.yield %211 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_26 = gpu.thread_id x
    %146 = arith.index_cast %thread_id_x_26 : index to i32
    %c0_i32_27 = arith.constant 0 : i32
    %c64_i32_28 = arith.constant 64 : i32
    %c16_i32_29 = arith.constant 16 : i32
    %147 = arith.andi %146, %c16_i32_29 : i32
    %148 = arith.cmpi ne, %147, %c0_i32_27 : i32
    %c32_i32_30 = arith.constant 32 : i32
    %149 = arith.andi %146, %c32_i32_30 : i32
    %150 = arith.cmpi ne, %149, %c0_i32_27 : i32
    %cst_31 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %151 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_31) -> (vector<2x8xf32>) {
      %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %169 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %170 = vector.extract %133#3[%169, %168] : f32 from vector<2x8xf32>
      %171 = vector.insert %170, %cst_39 [0, 0] : f32 into vector<1x4xf32>
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %173 = vector.extract %133#3[%169, %172] : f32 from vector<2x8xf32>
      %174 = vector.insert %173, %171 [0, 1] : f32 into vector<1x4xf32>
      %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %176 = vector.extract %133#3[%169, %175] : f32 from vector<2x8xf32>
      %177 = vector.insert %176, %174 [0, 2] : f32 into vector<1x4xf32>
      %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %179 = vector.extract %133#3[%169, %178] : f32 from vector<2x8xf32>
      %180 = vector.insert %179, %177 [0, 3] : f32 into vector<1x4xf32>
      %181 = vector.extract %180[0, 0] : f32 from vector<1x4xf32>
      %182 = vector.extract %180[0, 1] : f32 from vector<1x4xf32>
      %183 = vector.extract %180[0, 2] : f32 from vector<1x4xf32>
      %184 = vector.extract %180[0, 3] : f32 from vector<1x4xf32>
      %185 = arith.select %148, %181, %182 : f32
      %shuffleResult, %valid = gpu.shuffle xor %185, %c16_i32_29, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
      %186 = arith.select %148, %shuffleResult, %181 : f32
      %187 = arith.select %148, %182, %shuffleResult : f32
      %188 = arith.select %148, %183, %184 : f32
      %shuffleResult_40, %valid_41 = gpu.shuffle xor %188, %c16_i32_29, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
      %189 = arith.select %148, %shuffleResult_40, %183 : f32
      %190 = arith.select %148, %184, %shuffleResult_40 : f32
      %191 = arith.select %150, %186, %189 : f32
      %shuffleResult_42, %valid_43 = gpu.shuffle xor %191, %c32_i32_30, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
      %192 = arith.select %150, %shuffleResult_42, %186 : f32
      %193 = arith.select %150, %189, %shuffleResult_42 : f32
      %194 = arith.select %150, %187, %190 : f32
      %shuffleResult_44, %valid_45 = gpu.shuffle xor %194, %c32_i32_30, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
      %195 = arith.select %150, %shuffleResult_44, %187 : f32
      %196 = arith.select %150, %190, %shuffleResult_44 : f32
      %197 = vector.insert %192, %180 [0, 0] : f32 into vector<1x4xf32>
      %198 = vector.insert %195, %197 [0, 1] : f32 into vector<1x4xf32>
      %199 = vector.insert %193, %198 [0, 2] : f32 into vector<1x4xf32>
      %200 = vector.insert %196, %199 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
      %201 = vector.extract %200[0, 0] : f32 from vector<1x4xf32>
      %202 = vector.insert %201, %arg5 [%169, %168] : f32 into vector<2x8xf32>
      %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %204 = vector.extract %200[0, 1] : f32 from vector<1x4xf32>
      %205 = vector.insert %204, %202 [%169, %203] : f32 into vector<2x8xf32>
      %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %207 = vector.extract %200[0, 2] : f32 from vector<1x4xf32>
      %208 = vector.insert %207, %205 [%169, %206] : f32 into vector<2x8xf32>
      %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %210 = vector.extract %200[0, 3] : f32 from vector<1x4xf32>
      %211 = vector.insert %210, %208 [%169, %209] : f32 into vector<2x8xf32>
      affine.yield %211 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_32 = gpu.thread_id x
    %152 = arith.index_cast %thread_id_x_32 : index to i32
    %c0_i32_33 = arith.constant 0 : i32
    %c64_i32_34 = arith.constant 64 : i32
    %c16_i32_35 = arith.constant 16 : i32
    %153 = arith.andi %152, %c16_i32_35 : i32
    %154 = arith.cmpi ne, %153, %c0_i32_33 : i32
    %c32_i32_36 = arith.constant 32 : i32
    %155 = arith.andi %152, %c32_i32_36 : i32
    %156 = arith.cmpi ne, %155, %c0_i32_33 : i32
    %cst_37 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %157 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_37) -> (vector<2x8xf32>) {
      %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %169 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %170 = vector.extract %133#0[%169, %168] : f32 from vector<2x8xf32>
      %171 = vector.insert %170, %cst_39 [0, 0] : f32 into vector<1x4xf32>
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %173 = vector.extract %133#0[%169, %172] : f32 from vector<2x8xf32>
      %174 = vector.insert %173, %171 [0, 1] : f32 into vector<1x4xf32>
      %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %176 = vector.extract %133#0[%169, %175] : f32 from vector<2x8xf32>
      %177 = vector.insert %176, %174 [0, 2] : f32 into vector<1x4xf32>
      %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %179 = vector.extract %133#0[%169, %178] : f32 from vector<2x8xf32>
      %180 = vector.insert %179, %177 [0, 3] : f32 into vector<1x4xf32>
      %181 = vector.extract %180[0, 0] : f32 from vector<1x4xf32>
      %182 = vector.extract %180[0, 1] : f32 from vector<1x4xf32>
      %183 = vector.extract %180[0, 2] : f32 from vector<1x4xf32>
      %184 = vector.extract %180[0, 3] : f32 from vector<1x4xf32>
      %185 = arith.select %154, %181, %182 : f32
      %shuffleResult, %valid = gpu.shuffle xor %185, %c16_i32_35, %c64_i32_34 {frisk.layout_role = "wmmaC"} : f32
      %186 = arith.select %154, %shuffleResult, %181 : f32
      %187 = arith.select %154, %182, %shuffleResult : f32
      %188 = arith.select %154, %183, %184 : f32
      %shuffleResult_40, %valid_41 = gpu.shuffle xor %188, %c16_i32_35, %c64_i32_34 {frisk.layout_role = "wmmaC"} : f32
      %189 = arith.select %154, %shuffleResult_40, %183 : f32
      %190 = arith.select %154, %184, %shuffleResult_40 : f32
      %191 = arith.select %156, %186, %189 : f32
      %shuffleResult_42, %valid_43 = gpu.shuffle xor %191, %c32_i32_36, %c64_i32_34 {frisk.layout_role = "wmmaC"} : f32
      %192 = arith.select %156, %shuffleResult_42, %186 : f32
      %193 = arith.select %156, %189, %shuffleResult_42 : f32
      %194 = arith.select %156, %187, %190 : f32
      %shuffleResult_44, %valid_45 = gpu.shuffle xor %194, %c32_i32_36, %c64_i32_34 {frisk.layout_role = "wmmaC"} : f32
      %195 = arith.select %156, %shuffleResult_44, %187 : f32
      %196 = arith.select %156, %190, %shuffleResult_44 : f32
      %197 = vector.insert %192, %180 [0, 0] : f32 into vector<1x4xf32>
      %198 = vector.insert %195, %197 [0, 1] : f32 into vector<1x4xf32>
      %199 = vector.insert %193, %198 [0, 2] : f32 into vector<1x4xf32>
      %200 = vector.insert %196, %199 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
      %201 = vector.extract %200[0, 0] : f32 from vector<1x4xf32>
      %202 = vector.insert %201, %arg5 [%169, %168] : f32 into vector<2x8xf32>
      %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %204 = vector.extract %200[0, 1] : f32 from vector<1x4xf32>
      %205 = vector.insert %204, %202 [%169, %203] : f32 into vector<2x8xf32>
      %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %207 = vector.extract %200[0, 2] : f32 from vector<1x4xf32>
      %208 = vector.insert %207, %205 [%169, %206] : f32 into vector<2x8xf32>
      %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %210 = vector.extract %200[0, 3] : f32 from vector<1x4xf32>
      %211 = vector.insert %210, %208 [%169, %209] : f32 into vector<2x8xf32>
      affine.yield %211 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %158 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %159:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %168 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %169 = affine.apply affine_map<() -> (0)>()
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %170 = vector.extract %133#1[%168, %169] : f32 from vector<2x1xf32>
      %171 = vector.insert %170, %cst_39 [0, 0] : f32 into vector<1x1xf32>
      %cst_40 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %172 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_40) -> (vector<1x4xf32>) {
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %cst_44 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %240 = vector.extract %171[%c0_45, %c0_46] : f32 from vector<1x1xf32>
        %241 = vector.insert %240, %cst_44 [0, 0] : f32 into vector<1x4xf32>
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %242 = vector.extract %171[%c0_47, %c0_48] : f32 from vector<1x1xf32>
        %243 = vector.insert %242, %241 [0, 1] : f32 into vector<1x4xf32>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %244 = vector.extract %171[%c0_49, %c0_50] : f32 from vector<1x1xf32>
        %245 = vector.insert %244, %243 [0, 2] : f32 into vector<1x4xf32>
        %c0_51 = arith.constant 0 : index
        %c0_52 = arith.constant 0 : index
        %246 = vector.extract %171[%c0_51, %c0_52] : f32 from vector<1x1xf32>
        %247 = vector.insert %246, %245 [0, 3] : f32 into vector<1x4xf32>
        %248 = vector.extract %247[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg10 [%c0_43, %c0_42] : f32 into vector<1x4xf32>
        %250 = affine.apply affine_map<() -> (1)>()
        %251 = vector.extract %247[0, 1] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%c0_43, %250] : f32 into vector<1x4xf32>
        %253 = affine.apply affine_map<() -> (2)>()
        %254 = vector.extract %247[0, 2] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%c0_43, %253] : f32 into vector<1x4xf32>
        %256 = affine.apply affine_map<() -> (3)>()
        %257 = vector.extract %247[0, 3] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %255 [%c0_43, %256] : f32 into vector<1x4xf32>
        affine.yield %258 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %173 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %175 = vector.extract %172[0, 0] : f32 from vector<1x4xf32>
      %176 = vector.insert %175, %arg5 [%173, %174] : f32 into vector<2x8xf32>
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %178 = vector.extract %172[0, 1] : f32 from vector<1x4xf32>
      %179 = vector.insert %178, %176 [%173, %177] : f32 into vector<2x8xf32>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %181 = vector.extract %172[0, 2] : f32 from vector<1x4xf32>
      %182 = vector.insert %181, %179 [%173, %180] : f32 into vector<2x8xf32>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %184 = vector.extract %172[0, 3] : f32 from vector<1x4xf32>
      %185 = vector.insert %184, %182 [%173, %183] : f32 into vector<2x8xf32>
      %186 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_41 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %188 = vector.extract %157[%186, %187] : f32 from vector<2x8xf32>
      %189 = vector.insert %188, %cst_41 [0, 0] : f32 into vector<1x4xf32>
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %191 = vector.extract %157[%186, %190] : f32 from vector<2x8xf32>
      %192 = vector.insert %191, %189 [0, 1] : f32 into vector<1x4xf32>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %194 = vector.extract %157[%186, %193] : f32 from vector<2x8xf32>
      %195 = vector.insert %194, %192 [0, 2] : f32 into vector<1x4xf32>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %197 = vector.extract %157[%186, %196] : f32 from vector<2x8xf32>
      %198 = vector.insert %197, %195 [0, 3] : f32 into vector<1x4xf32>
      %199 = arith.divf %198, %172 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %200 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %202 = vector.extract %199[0, 0] : f32 from vector<1x4xf32>
      %203 = vector.insert %202, %arg6 [%200, %201] : f32 into vector<2x8xf32>
      %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %205 = vector.extract %199[0, 1] : f32 from vector<1x4xf32>
      %206 = vector.insert %205, %203 [%200, %204] : f32 into vector<2x8xf32>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %208 = vector.extract %199[0, 2] : f32 from vector<1x4xf32>
      %209 = vector.insert %208, %206 [%200, %207] : f32 into vector<2x8xf32>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %211 = vector.extract %199[0, 3] : f32 from vector<1x4xf32>
      %212 = vector.insert %211, %209 [%200, %210] : f32 into vector<2x8xf32>
      %213 = arith.truncf %199 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %214 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %215 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %216 = vector.extract %213[0, 0] : f16 from vector<1x4xf16>
      %217 = vector.insert %216, %arg7 [%214, %215] : f16 into vector<2x8xf16>
      %218 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %219 = vector.extract %213[0, 1] : f16 from vector<1x4xf16>
      %220 = vector.insert %219, %217 [%214, %218] : f16 into vector<2x8xf16>
      %221 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %222 = vector.extract %213[0, 2] : f16 from vector<1x4xf16>
      %223 = vector.insert %222, %220 [%214, %221] : f16 into vector<2x8xf16>
      %224 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %225 = vector.extract %213[0, 3] : f16 from vector<1x4xf16>
      %226 = vector.insert %225, %223 [%214, %224] : f16 into vector<2x8xf16>
      %227 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %229 = vector.extract %213[0, 0] : f16 from vector<1x4xf16>
      %230 = vector.insert %229, %arg8 [%227, %228] : f16 into vector<2x8xf16>
      %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %232 = vector.extract %213[0, 1] : f16 from vector<1x4xf16>
      %233 = vector.insert %232, %230 [%227, %231] : f16 into vector<2x8xf16>
      %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %235 = vector.extract %213[0, 2] : f16 from vector<1x4xf16>
      %236 = vector.insert %235, %233 [%227, %234] : f16 into vector<2x8xf16>
      %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %238 = vector.extract %213[0, 3] : f16 from vector<1x4xf16>
      %239 = vector.insert %238, %236 [%227, %237] : f16 into vector<2x8xf16>
      affine.yield %185, %212, %226, %239 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_38 = gpu.thread_id x
    %160 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x_38]
    %161 = arith.cmpi ult, %160, %c4 : index
    scf.if %161 {
      affine.for %arg4 = 0 to 4 {
        %168 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %170 = vector.extract %159#3[%168, %169] : f16 from vector<2x8xf16>
        %171 = vector.insert %170, %cst_39 [0, 0] : f16 into vector<1x4xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %173 = vector.extract %159#3[%168, %172] : f16 from vector<2x8xf16>
        %174 = vector.insert %173, %171 [0, 1] : f16 into vector<1x4xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %176 = vector.extract %159#3[%168, %175] : f16 from vector<2x8xf16>
        %177 = vector.insert %176, %174 [0, 2] : f16 into vector<1x4xf16>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %179 = vector.extract %159#3[%168, %178] : f16 from vector<2x8xf16>
        %180 = vector.insert %179, %177 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_40 = gpu.thread_id x
        %181 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %183 = vector.extract %180[0, 0] : f16 from vector<1x4xf16>
        %c0_41 = arith.constant 0 : index
        %c0_42 = arith.constant 0 : index
        %184 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%182)
        %c0_43 = arith.constant 0 : index
        %185 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%182)
        %c0_44 = arith.constant 0 : index
        %186 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_42, %185)
        %187 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %184, %c0_41, %c0_43, %c0_44, %186)
        %188 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %184, %c0_41, %c0_43, %c0_44, %186)
        affine.store %183, %158[%187, %188] {tiled = true} : memref<128x32xf16, 1>
        %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %190 = vector.extract %180[0, 1] : f16 from vector<1x4xf16>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %191 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%189)
        %c0_47 = arith.constant 0 : index
        %192 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%189)
        %c0_48 = arith.constant 0 : index
        %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_46, %192)
        %194 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %191, %c0_45, %c0_47, %c0_48, %193)
        %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %191, %c0_45, %c0_47, %c0_48, %193)
        affine.store %190, %158[%194, %195] {tiled = true} : memref<128x32xf16, 1>
        %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %197 = vector.extract %180[0, 2] : f16 from vector<1x4xf16>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %198 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%196)
        %c0_51 = arith.constant 0 : index
        %199 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%196)
        %c0_52 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_50, %199)
        %201 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %198, %c0_49, %c0_51, %c0_52, %200)
        %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %198, %c0_49, %c0_51, %c0_52, %200)
        affine.store %197, %158[%201, %202] {tiled = true} : memref<128x32xf16, 1>
        %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %204 = vector.extract %180[0, 3] : f16 from vector<1x4xf16>
        %c0_53 = arith.constant 0 : index
        %c0_54 = arith.constant 0 : index
        %205 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%203)
        %c0_55 = arith.constant 0 : index
        %206 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%203)
        %c0_56 = arith.constant 0 : index
        %207 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_54, %206)
        %208 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %205, %c0_53, %c0_55, %c0_56, %207)
        %209 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %205, %c0_53, %c0_55, %c0_56, %207)
        affine.store %204, %158[%208, %209] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %162 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %163:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %168 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %169 = affine.apply affine_map<() -> (0)>()
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %170 = vector.extract %133#1[%168, %169] : f32 from vector<2x1xf32>
      %171 = vector.insert %170, %cst_39 [0, 0] : f32 into vector<1x1xf32>
      %cst_40 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %172 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_40) -> (vector<1x4xf32>) {
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %cst_44 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %240 = vector.extract %171[%c0_45, %c0_46] : f32 from vector<1x1xf32>
        %241 = vector.insert %240, %cst_44 [0, 0] : f32 into vector<1x4xf32>
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %242 = vector.extract %171[%c0_47, %c0_48] : f32 from vector<1x1xf32>
        %243 = vector.insert %242, %241 [0, 1] : f32 into vector<1x4xf32>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %244 = vector.extract %171[%c0_49, %c0_50] : f32 from vector<1x1xf32>
        %245 = vector.insert %244, %243 [0, 2] : f32 into vector<1x4xf32>
        %c0_51 = arith.constant 0 : index
        %c0_52 = arith.constant 0 : index
        %246 = vector.extract %171[%c0_51, %c0_52] : f32 from vector<1x1xf32>
        %247 = vector.insert %246, %245 [0, 3] : f32 into vector<1x4xf32>
        %248 = vector.extract %247[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg10 [%c0_43, %c0_42] : f32 into vector<1x4xf32>
        %250 = affine.apply affine_map<() -> (1)>()
        %251 = vector.extract %247[0, 1] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%c0_43, %250] : f32 into vector<1x4xf32>
        %253 = affine.apply affine_map<() -> (2)>()
        %254 = vector.extract %247[0, 2] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%c0_43, %253] : f32 into vector<1x4xf32>
        %256 = affine.apply affine_map<() -> (3)>()
        %257 = vector.extract %247[0, 3] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %255 [%c0_43, %256] : f32 into vector<1x4xf32>
        affine.yield %258 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %173 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %175 = vector.extract %172[0, 0] : f32 from vector<1x4xf32>
      %176 = vector.insert %175, %arg5 [%173, %174] : f32 into vector<2x8xf32>
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %178 = vector.extract %172[0, 1] : f32 from vector<1x4xf32>
      %179 = vector.insert %178, %176 [%173, %177] : f32 into vector<2x8xf32>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %181 = vector.extract %172[0, 2] : f32 from vector<1x4xf32>
      %182 = vector.insert %181, %179 [%173, %180] : f32 into vector<2x8xf32>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %184 = vector.extract %172[0, 3] : f32 from vector<1x4xf32>
      %185 = vector.insert %184, %182 [%173, %183] : f32 into vector<2x8xf32>
      %186 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_41 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %188 = vector.extract %151[%186, %187] : f32 from vector<2x8xf32>
      %189 = vector.insert %188, %cst_41 [0, 0] : f32 into vector<1x4xf32>
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %191 = vector.extract %151[%186, %190] : f32 from vector<2x8xf32>
      %192 = vector.insert %191, %189 [0, 1] : f32 into vector<1x4xf32>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %194 = vector.extract %151[%186, %193] : f32 from vector<2x8xf32>
      %195 = vector.insert %194, %192 [0, 2] : f32 into vector<1x4xf32>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %197 = vector.extract %151[%186, %196] : f32 from vector<2x8xf32>
      %198 = vector.insert %197, %195 [0, 3] : f32 into vector<1x4xf32>
      %199 = arith.divf %198, %172 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %200 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %202 = vector.extract %199[0, 0] : f32 from vector<1x4xf32>
      %203 = vector.insert %202, %arg6 [%200, %201] : f32 into vector<2x8xf32>
      %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %205 = vector.extract %199[0, 1] : f32 from vector<1x4xf32>
      %206 = vector.insert %205, %203 [%200, %204] : f32 into vector<2x8xf32>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %208 = vector.extract %199[0, 2] : f32 from vector<1x4xf32>
      %209 = vector.insert %208, %206 [%200, %207] : f32 into vector<2x8xf32>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %211 = vector.extract %199[0, 3] : f32 from vector<1x4xf32>
      %212 = vector.insert %211, %209 [%200, %210] : f32 into vector<2x8xf32>
      %213 = arith.truncf %199 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %214 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %215 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %216 = vector.extract %213[0, 0] : f16 from vector<1x4xf16>
      %217 = vector.insert %216, %arg7 [%214, %215] : f16 into vector<2x8xf16>
      %218 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %219 = vector.extract %213[0, 1] : f16 from vector<1x4xf16>
      %220 = vector.insert %219, %217 [%214, %218] : f16 into vector<2x8xf16>
      %221 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %222 = vector.extract %213[0, 2] : f16 from vector<1x4xf16>
      %223 = vector.insert %222, %220 [%214, %221] : f16 into vector<2x8xf16>
      %224 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %225 = vector.extract %213[0, 3] : f16 from vector<1x4xf16>
      %226 = vector.insert %225, %223 [%214, %224] : f16 into vector<2x8xf16>
      %227 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %229 = vector.extract %213[0, 0] : f16 from vector<1x4xf16>
      %230 = vector.insert %229, %arg8 [%227, %228] : f16 into vector<2x8xf16>
      %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %232 = vector.extract %213[0, 1] : f16 from vector<1x4xf16>
      %233 = vector.insert %232, %230 [%227, %231] : f16 into vector<2x8xf16>
      %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %235 = vector.extract %213[0, 2] : f16 from vector<1x4xf16>
      %236 = vector.insert %235, %233 [%227, %234] : f16 into vector<2x8xf16>
      %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %238 = vector.extract %213[0, 3] : f16 from vector<1x4xf16>
      %239 = vector.insert %238, %236 [%227, %237] : f16 into vector<2x8xf16>
      affine.yield %185, %212, %226, %239 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %161 {
      affine.for %arg4 = 0 to 4 {
        %168 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %170 = vector.extract %163#3[%168, %169] : f16 from vector<2x8xf16>
        %171 = vector.insert %170, %cst_39 [0, 0] : f16 into vector<1x4xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %173 = vector.extract %163#3[%168, %172] : f16 from vector<2x8xf16>
        %174 = vector.insert %173, %171 [0, 1] : f16 into vector<1x4xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %176 = vector.extract %163#3[%168, %175] : f16 from vector<2x8xf16>
        %177 = vector.insert %176, %174 [0, 2] : f16 into vector<1x4xf16>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %179 = vector.extract %163#3[%168, %178] : f16 from vector<2x8xf16>
        %180 = vector.insert %179, %177 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_40 = gpu.thread_id x
        %181 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %183 = vector.extract %180[0, 0] : f16 from vector<1x4xf16>
        %c0_41 = arith.constant 0 : index
        %c0_42 = arith.constant 0 : index
        %184 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%182)
        %c0_43 = arith.constant 0 : index
        %185 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%182)
        %c0_44 = arith.constant 0 : index
        %186 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_42, %185)
        %187 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %184, %c0_41, %c0_43, %c0_44, %186)
        %188 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %184, %c0_41, %c0_43, %c0_44, %186)
        affine.store %183, %162[%187, %188] {tiled = true} : memref<128x32xf16, 1>
        %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %190 = vector.extract %180[0, 1] : f16 from vector<1x4xf16>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %191 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%189)
        %c0_47 = arith.constant 0 : index
        %192 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%189)
        %c0_48 = arith.constant 0 : index
        %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_46, %192)
        %194 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %191, %c0_45, %c0_47, %c0_48, %193)
        %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %191, %c0_45, %c0_47, %c0_48, %193)
        affine.store %190, %162[%194, %195] {tiled = true} : memref<128x32xf16, 1>
        %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %197 = vector.extract %180[0, 2] : f16 from vector<1x4xf16>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %198 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%196)
        %c0_51 = arith.constant 0 : index
        %199 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%196)
        %c0_52 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_50, %199)
        %201 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %198, %c0_49, %c0_51, %c0_52, %200)
        %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %198, %c0_49, %c0_51, %c0_52, %200)
        affine.store %197, %162[%201, %202] {tiled = true} : memref<128x32xf16, 1>
        %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %204 = vector.extract %180[0, 3] : f16 from vector<1x4xf16>
        %c0_53 = arith.constant 0 : index
        %c0_54 = arith.constant 0 : index
        %205 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%203)
        %c0_55 = arith.constant 0 : index
        %206 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%203)
        %c0_56 = arith.constant 0 : index
        %207 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_54, %206)
        %208 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %205, %c0_53, %c0_55, %c0_56, %207)
        %209 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %205, %c0_53, %c0_55, %c0_56, %207)
        affine.store %204, %162[%208, %209] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %164 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %165:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %168 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %169 = affine.apply affine_map<() -> (0)>()
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %170 = vector.extract %133#1[%168, %169] : f32 from vector<2x1xf32>
      %171 = vector.insert %170, %cst_39 [0, 0] : f32 into vector<1x1xf32>
      %cst_40 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %172 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_40) -> (vector<1x4xf32>) {
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %cst_44 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %240 = vector.extract %171[%c0_45, %c0_46] : f32 from vector<1x1xf32>
        %241 = vector.insert %240, %cst_44 [0, 0] : f32 into vector<1x4xf32>
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %242 = vector.extract %171[%c0_47, %c0_48] : f32 from vector<1x1xf32>
        %243 = vector.insert %242, %241 [0, 1] : f32 into vector<1x4xf32>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %244 = vector.extract %171[%c0_49, %c0_50] : f32 from vector<1x1xf32>
        %245 = vector.insert %244, %243 [0, 2] : f32 into vector<1x4xf32>
        %c0_51 = arith.constant 0 : index
        %c0_52 = arith.constant 0 : index
        %246 = vector.extract %171[%c0_51, %c0_52] : f32 from vector<1x1xf32>
        %247 = vector.insert %246, %245 [0, 3] : f32 into vector<1x4xf32>
        %248 = vector.extract %247[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg10 [%c0_43, %c0_42] : f32 into vector<1x4xf32>
        %250 = affine.apply affine_map<() -> (1)>()
        %251 = vector.extract %247[0, 1] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%c0_43, %250] : f32 into vector<1x4xf32>
        %253 = affine.apply affine_map<() -> (2)>()
        %254 = vector.extract %247[0, 2] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%c0_43, %253] : f32 into vector<1x4xf32>
        %256 = affine.apply affine_map<() -> (3)>()
        %257 = vector.extract %247[0, 3] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %255 [%c0_43, %256] : f32 into vector<1x4xf32>
        affine.yield %258 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %173 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %175 = vector.extract %172[0, 0] : f32 from vector<1x4xf32>
      %176 = vector.insert %175, %arg5 [%173, %174] : f32 into vector<2x8xf32>
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %178 = vector.extract %172[0, 1] : f32 from vector<1x4xf32>
      %179 = vector.insert %178, %176 [%173, %177] : f32 into vector<2x8xf32>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %181 = vector.extract %172[0, 2] : f32 from vector<1x4xf32>
      %182 = vector.insert %181, %179 [%173, %180] : f32 into vector<2x8xf32>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %184 = vector.extract %172[0, 3] : f32 from vector<1x4xf32>
      %185 = vector.insert %184, %182 [%173, %183] : f32 into vector<2x8xf32>
      %186 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_41 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %188 = vector.extract %145[%186, %187] : f32 from vector<2x8xf32>
      %189 = vector.insert %188, %cst_41 [0, 0] : f32 into vector<1x4xf32>
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %191 = vector.extract %145[%186, %190] : f32 from vector<2x8xf32>
      %192 = vector.insert %191, %189 [0, 1] : f32 into vector<1x4xf32>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %194 = vector.extract %145[%186, %193] : f32 from vector<2x8xf32>
      %195 = vector.insert %194, %192 [0, 2] : f32 into vector<1x4xf32>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %197 = vector.extract %145[%186, %196] : f32 from vector<2x8xf32>
      %198 = vector.insert %197, %195 [0, 3] : f32 into vector<1x4xf32>
      %199 = arith.divf %198, %172 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %200 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %202 = vector.extract %199[0, 0] : f32 from vector<1x4xf32>
      %203 = vector.insert %202, %arg6 [%200, %201] : f32 into vector<2x8xf32>
      %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %205 = vector.extract %199[0, 1] : f32 from vector<1x4xf32>
      %206 = vector.insert %205, %203 [%200, %204] : f32 into vector<2x8xf32>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %208 = vector.extract %199[0, 2] : f32 from vector<1x4xf32>
      %209 = vector.insert %208, %206 [%200, %207] : f32 into vector<2x8xf32>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %211 = vector.extract %199[0, 3] : f32 from vector<1x4xf32>
      %212 = vector.insert %211, %209 [%200, %210] : f32 into vector<2x8xf32>
      %213 = arith.truncf %199 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %214 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %215 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %216 = vector.extract %213[0, 0] : f16 from vector<1x4xf16>
      %217 = vector.insert %216, %arg7 [%214, %215] : f16 into vector<2x8xf16>
      %218 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %219 = vector.extract %213[0, 1] : f16 from vector<1x4xf16>
      %220 = vector.insert %219, %217 [%214, %218] : f16 into vector<2x8xf16>
      %221 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %222 = vector.extract %213[0, 2] : f16 from vector<1x4xf16>
      %223 = vector.insert %222, %220 [%214, %221] : f16 into vector<2x8xf16>
      %224 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %225 = vector.extract %213[0, 3] : f16 from vector<1x4xf16>
      %226 = vector.insert %225, %223 [%214, %224] : f16 into vector<2x8xf16>
      %227 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %229 = vector.extract %213[0, 0] : f16 from vector<1x4xf16>
      %230 = vector.insert %229, %arg8 [%227, %228] : f16 into vector<2x8xf16>
      %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %232 = vector.extract %213[0, 1] : f16 from vector<1x4xf16>
      %233 = vector.insert %232, %230 [%227, %231] : f16 into vector<2x8xf16>
      %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %235 = vector.extract %213[0, 2] : f16 from vector<1x4xf16>
      %236 = vector.insert %235, %233 [%227, %234] : f16 into vector<2x8xf16>
      %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %238 = vector.extract %213[0, 3] : f16 from vector<1x4xf16>
      %239 = vector.insert %238, %236 [%227, %237] : f16 into vector<2x8xf16>
      affine.yield %185, %212, %226, %239 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %161 {
      affine.for %arg4 = 0 to 4 {
        %168 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %170 = vector.extract %165#3[%168, %169] : f16 from vector<2x8xf16>
        %171 = vector.insert %170, %cst_39 [0, 0] : f16 into vector<1x4xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %173 = vector.extract %165#3[%168, %172] : f16 from vector<2x8xf16>
        %174 = vector.insert %173, %171 [0, 1] : f16 into vector<1x4xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %176 = vector.extract %165#3[%168, %175] : f16 from vector<2x8xf16>
        %177 = vector.insert %176, %174 [0, 2] : f16 into vector<1x4xf16>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %179 = vector.extract %165#3[%168, %178] : f16 from vector<2x8xf16>
        %180 = vector.insert %179, %177 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_40 = gpu.thread_id x
        %181 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %183 = vector.extract %180[0, 0] : f16 from vector<1x4xf16>
        %c0_41 = arith.constant 0 : index
        %c0_42 = arith.constant 0 : index
        %184 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%182)
        %c0_43 = arith.constant 0 : index
        %185 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%182)
        %c0_44 = arith.constant 0 : index
        %186 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_42, %185)
        %187 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %184, %c0_41, %c0_43, %c0_44, %186)
        %188 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %184, %c0_41, %c0_43, %c0_44, %186)
        affine.store %183, %164[%187, %188] {tiled = true} : memref<128x32xf16, 1>
        %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %190 = vector.extract %180[0, 1] : f16 from vector<1x4xf16>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %191 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%189)
        %c0_47 = arith.constant 0 : index
        %192 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%189)
        %c0_48 = arith.constant 0 : index
        %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_46, %192)
        %194 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %191, %c0_45, %c0_47, %c0_48, %193)
        %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %191, %c0_45, %c0_47, %c0_48, %193)
        affine.store %190, %164[%194, %195] {tiled = true} : memref<128x32xf16, 1>
        %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %197 = vector.extract %180[0, 2] : f16 from vector<1x4xf16>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %198 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%196)
        %c0_51 = arith.constant 0 : index
        %199 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%196)
        %c0_52 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_50, %199)
        %201 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %198, %c0_49, %c0_51, %c0_52, %200)
        %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %198, %c0_49, %c0_51, %c0_52, %200)
        affine.store %197, %164[%201, %202] {tiled = true} : memref<128x32xf16, 1>
        %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %204 = vector.extract %180[0, 3] : f16 from vector<1x4xf16>
        %c0_53 = arith.constant 0 : index
        %c0_54 = arith.constant 0 : index
        %205 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%203)
        %c0_55 = arith.constant 0 : index
        %206 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%203)
        %c0_56 = arith.constant 0 : index
        %207 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_54, %206)
        %208 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %205, %c0_53, %c0_55, %c0_56, %207)
        %209 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %205, %c0_53, %c0_55, %c0_56, %207)
        affine.store %204, %164[%208, %209] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %166 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %167:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %168 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %169 = affine.apply affine_map<() -> (0)>()
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %170 = vector.extract %133#1[%168, %169] : f32 from vector<2x1xf32>
      %171 = vector.insert %170, %cst_39 [0, 0] : f32 into vector<1x1xf32>
      %cst_40 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %172 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_40) -> (vector<1x4xf32>) {
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %cst_44 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %240 = vector.extract %171[%c0_45, %c0_46] : f32 from vector<1x1xf32>
        %241 = vector.insert %240, %cst_44 [0, 0] : f32 into vector<1x4xf32>
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %242 = vector.extract %171[%c0_47, %c0_48] : f32 from vector<1x1xf32>
        %243 = vector.insert %242, %241 [0, 1] : f32 into vector<1x4xf32>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %244 = vector.extract %171[%c0_49, %c0_50] : f32 from vector<1x1xf32>
        %245 = vector.insert %244, %243 [0, 2] : f32 into vector<1x4xf32>
        %c0_51 = arith.constant 0 : index
        %c0_52 = arith.constant 0 : index
        %246 = vector.extract %171[%c0_51, %c0_52] : f32 from vector<1x1xf32>
        %247 = vector.insert %246, %245 [0, 3] : f32 into vector<1x4xf32>
        %248 = vector.extract %247[0, 0] : f32 from vector<1x4xf32>
        %249 = vector.insert %248, %arg10 [%c0_43, %c0_42] : f32 into vector<1x4xf32>
        %250 = affine.apply affine_map<() -> (1)>()
        %251 = vector.extract %247[0, 1] : f32 from vector<1x4xf32>
        %252 = vector.insert %251, %249 [%c0_43, %250] : f32 into vector<1x4xf32>
        %253 = affine.apply affine_map<() -> (2)>()
        %254 = vector.extract %247[0, 2] : f32 from vector<1x4xf32>
        %255 = vector.insert %254, %252 [%c0_43, %253] : f32 into vector<1x4xf32>
        %256 = affine.apply affine_map<() -> (3)>()
        %257 = vector.extract %247[0, 3] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %255 [%c0_43, %256] : f32 into vector<1x4xf32>
        affine.yield %258 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %173 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %175 = vector.extract %172[0, 0] : f32 from vector<1x4xf32>
      %176 = vector.insert %175, %arg5 [%173, %174] : f32 into vector<2x8xf32>
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %178 = vector.extract %172[0, 1] : f32 from vector<1x4xf32>
      %179 = vector.insert %178, %176 [%173, %177] : f32 into vector<2x8xf32>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %181 = vector.extract %172[0, 2] : f32 from vector<1x4xf32>
      %182 = vector.insert %181, %179 [%173, %180] : f32 into vector<2x8xf32>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %184 = vector.extract %172[0, 3] : f32 from vector<1x4xf32>
      %185 = vector.insert %184, %182 [%173, %183] : f32 into vector<2x8xf32>
      %186 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_41 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %188 = vector.extract %139[%186, %187] : f32 from vector<2x8xf32>
      %189 = vector.insert %188, %cst_41 [0, 0] : f32 into vector<1x4xf32>
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %191 = vector.extract %139[%186, %190] : f32 from vector<2x8xf32>
      %192 = vector.insert %191, %189 [0, 1] : f32 into vector<1x4xf32>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %194 = vector.extract %139[%186, %193] : f32 from vector<2x8xf32>
      %195 = vector.insert %194, %192 [0, 2] : f32 into vector<1x4xf32>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %197 = vector.extract %139[%186, %196] : f32 from vector<2x8xf32>
      %198 = vector.insert %197, %195 [0, 3] : f32 into vector<1x4xf32>
      %199 = arith.divf %198, %172 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %200 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %202 = vector.extract %199[0, 0] : f32 from vector<1x4xf32>
      %203 = vector.insert %202, %arg6 [%200, %201] : f32 into vector<2x8xf32>
      %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %205 = vector.extract %199[0, 1] : f32 from vector<1x4xf32>
      %206 = vector.insert %205, %203 [%200, %204] : f32 into vector<2x8xf32>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %208 = vector.extract %199[0, 2] : f32 from vector<1x4xf32>
      %209 = vector.insert %208, %206 [%200, %207] : f32 into vector<2x8xf32>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %211 = vector.extract %199[0, 3] : f32 from vector<1x4xf32>
      %212 = vector.insert %211, %209 [%200, %210] : f32 into vector<2x8xf32>
      %213 = arith.truncf %199 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %214 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %215 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %216 = vector.extract %213[0, 0] : f16 from vector<1x4xf16>
      %217 = vector.insert %216, %arg7 [%214, %215] : f16 into vector<2x8xf16>
      %218 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %219 = vector.extract %213[0, 1] : f16 from vector<1x4xf16>
      %220 = vector.insert %219, %217 [%214, %218] : f16 into vector<2x8xf16>
      %221 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %222 = vector.extract %213[0, 2] : f16 from vector<1x4xf16>
      %223 = vector.insert %222, %220 [%214, %221] : f16 into vector<2x8xf16>
      %224 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %225 = vector.extract %213[0, 3] : f16 from vector<1x4xf16>
      %226 = vector.insert %225, %223 [%214, %224] : f16 into vector<2x8xf16>
      %227 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %229 = vector.extract %213[0, 0] : f16 from vector<1x4xf16>
      %230 = vector.insert %229, %arg8 [%227, %228] : f16 into vector<2x8xf16>
      %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %232 = vector.extract %213[0, 1] : f16 from vector<1x4xf16>
      %233 = vector.insert %232, %230 [%227, %231] : f16 into vector<2x8xf16>
      %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %235 = vector.extract %213[0, 2] : f16 from vector<1x4xf16>
      %236 = vector.insert %235, %233 [%227, %234] : f16 into vector<2x8xf16>
      %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %238 = vector.extract %213[0, 3] : f16 from vector<1x4xf16>
      %239 = vector.insert %238, %236 [%227, %237] : f16 into vector<2x8xf16>
      affine.yield %185, %212, %226, %239 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %161 {
      affine.for %arg4 = 0 to 4 {
        %168 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %170 = vector.extract %167#3[%168, %169] : f16 from vector<2x8xf16>
        %171 = vector.insert %170, %cst_39 [0, 0] : f16 into vector<1x4xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %173 = vector.extract %167#3[%168, %172] : f16 from vector<2x8xf16>
        %174 = vector.insert %173, %171 [0, 1] : f16 into vector<1x4xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %176 = vector.extract %167#3[%168, %175] : f16 from vector<2x8xf16>
        %177 = vector.insert %176, %174 [0, 2] : f16 into vector<1x4xf16>
        %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %179 = vector.extract %167#3[%168, %178] : f16 from vector<2x8xf16>
        %180 = vector.insert %179, %177 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_40 = gpu.thread_id x
        %181 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %183 = vector.extract %180[0, 0] : f16 from vector<1x4xf16>
        %c0_41 = arith.constant 0 : index
        %c0_42 = arith.constant 0 : index
        %184 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%182)
        %c0_43 = arith.constant 0 : index
        %185 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%182)
        %c0_44 = arith.constant 0 : index
        %186 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_42, %185)
        %187 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %184, %c0_41, %c0_43, %c0_44, %186)
        %188 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %184, %c0_41, %c0_43, %c0_44, %186)
        affine.store %183, %166[%187, %188] {tiled = true} : memref<128x32xf16, 1>
        %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %190 = vector.extract %180[0, 1] : f16 from vector<1x4xf16>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %191 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%189)
        %c0_47 = arith.constant 0 : index
        %192 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%189)
        %c0_48 = arith.constant 0 : index
        %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_46, %192)
        %194 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %191, %c0_45, %c0_47, %c0_48, %193)
        %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %191, %c0_45, %c0_47, %c0_48, %193)
        affine.store %190, %166[%194, %195] {tiled = true} : memref<128x32xf16, 1>
        %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %197 = vector.extract %180[0, 2] : f16 from vector<1x4xf16>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %198 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%196)
        %c0_51 = arith.constant 0 : index
        %199 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%196)
        %c0_52 = arith.constant 0 : index
        %200 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_50, %199)
        %201 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %198, %c0_49, %c0_51, %c0_52, %200)
        %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %198, %c0_49, %c0_51, %c0_52, %200)
        affine.store %197, %166[%201, %202] {tiled = true} : memref<128x32xf16, 1>
        %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %204 = vector.extract %180[0, 3] : f16 from vector<1x4xf16>
        %c0_53 = arith.constant 0 : index
        %c0_54 = arith.constant 0 : index
        %205 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%203)
        %c0_55 = arith.constant 0 : index
        %206 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%203)
        %c0_56 = arith.constant 0 : index
        %207 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_54, %206)
        %208 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %181, %205, %c0_53, %c0_55, %c0_56, %207)
        %209 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %181, %205, %c0_53, %c0_55, %c0_56, %207)
        affine.store %204, %166[%208, %209] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
