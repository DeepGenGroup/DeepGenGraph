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
      %172 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %173 = affine.apply affine_map<() -> (0)>()
      %174 = vector.extract %cst_8[0, 0] : f32 from vector<1x1xf32>
      %175 = vector.insert %174, %arg5 [%172, %173] : f32 into vector<2x1xf32>
      %176 = vector.broadcast %cst_10 : f32 to vector<1x1xf32>
      %177 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %178 = affine.apply affine_map<() -> (0)>()
      %179 = vector.extract %176[0, 0] : f32 from vector<1x1xf32>
      %180 = vector.insert %179, %arg6 [%177, %178] : f32 into vector<2x1xf32>
      %181 = arith.addf %cst_8, %176 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
      %182 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %183 = affine.apply affine_map<() -> (0)>()
      %184 = vector.extract %181[0, 0] : f32 from vector<1x1xf32>
      %185 = vector.insert %184, %arg7 [%182, %183] : f32 into vector<2x1xf32>
      %186 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %187 = affine.apply affine_map<() -> (0)>()
      %188 = vector.extract %181[0, 0] : f32 from vector<1x1xf32>
      %189 = vector.insert %188, %arg8 [%186, %187] : f32 into vector<2x1xf32>
      affine.yield %175, %180, %185, %189 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = frisk.to_threadTile %cst_7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %11 = frisk.to_threadTile %cst_6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %12 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c0) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c0 floordiv 128) mod 4096, %c0 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %13 = frisk.async_copy_gm_to_shm %12 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    frisk.wait_buffer_ready %13 : !frisk.async_task
    %14 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c32) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c32 floordiv 128) mod 4096, %c32 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %alloc_12 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %15 = frisk.async_copy_gm_to_shm %14 to %alloc_12 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %16 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %17 = affine.apply affine_map<() -> (0)>()
    %18 = affine.apply affine_map<() -> (0)>()
    %cst_13 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %19 = vector.extract %16[%17, %18] : f16 from vector<2x8xf16>
    %20 = vector.insert %19, %cst_13 [0, 0] : f16 into vector<1x4xf16>
    %21 = affine.apply affine_map<() -> (1)>()
    %22 = vector.extract %16[%17, %21] : f16 from vector<2x8xf16>
    %23 = vector.insert %22, %20 [0, 1] : f16 into vector<1x4xf16>
    %24 = affine.apply affine_map<() -> (2)>()
    %25 = vector.extract %16[%17, %24] : f16 from vector<2x8xf16>
    %26 = vector.insert %25, %23 [0, 2] : f16 into vector<1x4xf16>
    %27 = affine.apply affine_map<() -> (3)>()
    %28 = vector.extract %16[%17, %27] : f16 from vector<2x8xf16>
    %29 = vector.insert %28, %26 [0, 3] : f16 into vector<1x4xf16>
    %30:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_4, %arg6 = %cst_4, %arg7 = %cst_4, %arg8 = %29) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %172 = frisk.to_threadTile %alloc {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %173 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %174 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %175 = vector.extract %172[%173, %174] : f16 from vector<2x8xf16>
      %176 = vector.insert %175, %cst_39 [0, 0] : f16 into vector<1x4xf16>
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %178 = vector.extract %172[%173, %177] : f16 from vector<2x8xf16>
      %179 = vector.insert %178, %176 [0, 1] : f16 into vector<1x4xf16>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %181 = vector.extract %172[%173, %180] : f16 from vector<2x8xf16>
      %182 = vector.insert %181, %179 [0, 2] : f16 into vector<1x4xf16>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %184 = vector.extract %172[%173, %183] : f16 from vector<2x8xf16>
      %185 = vector.insert %184, %182 [0, 3] : f16 into vector<1x4xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %188 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %189 = vector.insert %188, %arg5 [%186, %187] : f16 into vector<2x8xf16>
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %191 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %192 = vector.insert %191, %189 [%186, %190] : f16 into vector<2x8xf16>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %194 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %195 = vector.insert %194, %192 [%186, %193] : f16 into vector<2x8xf16>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %197 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %198 = vector.insert %197, %195 [%186, %196] : f16 into vector<2x8xf16>
      %199 = vector.broadcast %8 : f16 to vector<1x4xf16>
      %200 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %202 = vector.extract %199[0, 0] : f16 from vector<1x4xf16>
      %203 = vector.insert %202, %arg6 [%200, %201] : f16 into vector<2x8xf16>
      %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %205 = vector.extract %199[0, 1] : f16 from vector<1x4xf16>
      %206 = vector.insert %205, %203 [%200, %204] : f16 into vector<2x8xf16>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %208 = vector.extract %199[0, 2] : f16 from vector<1x4xf16>
      %209 = vector.insert %208, %206 [%200, %207] : f16 into vector<2x8xf16>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %211 = vector.extract %199[0, 3] : f16 from vector<1x4xf16>
      %212 = vector.insert %211, %209 [%200, %210] : f16 into vector<2x8xf16>
      %213 = arith.mulf %arg8, %199 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
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
      affine.yield %198, %212, %226, %185 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %31 = vector.broadcast %8 : f16 to vector<1x4xf16>
    %32 = arith.mulf %30#3, %31 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %33 = affine.apply affine_map<() -> (1)>()
    %34 = affine.apply affine_map<() -> (4)>()
    %35 = vector.extract %32[0, 0] : f16 from vector<1x4xf16>
    %36 = vector.insert %35, %30#2 [%33, %34] : f16 into vector<2x8xf16>
    %37 = affine.apply affine_map<() -> (5)>()
    %38 = vector.extract %32[0, 1] : f16 from vector<1x4xf16>
    %39 = vector.insert %38, %36 [%33, %37] : f16 into vector<2x8xf16>
    %40 = affine.apply affine_map<() -> (6)>()
    %41 = vector.extract %32[0, 2] : f16 from vector<1x4xf16>
    %42 = vector.insert %41, %39 [%33, %40] : f16 into vector<2x8xf16>
    %43 = affine.apply affine_map<() -> (7)>()
    %44 = vector.extract %32[0, 3] : f16 from vector<1x4xf16>
    %45 = vector.insert %44, %42 [%33, %43] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %15 : !frisk.async_task
    %46 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c64) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c64 floordiv 128) mod 4096, %c64 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %alloc_14 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %47 = frisk.async_copy_gm_to_shm %46 to %alloc_14 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %48 = frisk.to_threadTile %alloc_12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %49 = affine.apply affine_map<() -> (0)>()
    %50 = affine.apply affine_map<() -> (0)>()
    %cst_15 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %51 = vector.extract %48[%49, %50] : f16 from vector<2x8xf16>
    %52 = vector.insert %51, %cst_15 [0, 0] : f16 into vector<1x4xf16>
    %53 = affine.apply affine_map<() -> (1)>()
    %54 = vector.extract %48[%49, %53] : f16 from vector<2x8xf16>
    %55 = vector.insert %54, %52 [0, 1] : f16 into vector<1x4xf16>
    %56 = affine.apply affine_map<() -> (2)>()
    %57 = vector.extract %48[%49, %56] : f16 from vector<2x8xf16>
    %58 = vector.insert %57, %55 [0, 2] : f16 into vector<1x4xf16>
    %59 = affine.apply affine_map<() -> (3)>()
    %60 = vector.extract %48[%49, %59] : f16 from vector<2x8xf16>
    %61 = vector.insert %60, %58 [0, 3] : f16 into vector<1x4xf16>
    %62:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_4, %arg6 = %cst_4, %arg7 = %cst_4, %arg8 = %61) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %172 = frisk.to_threadTile %alloc_12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %173 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %174 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %175 = vector.extract %172[%173, %174] : f16 from vector<2x8xf16>
      %176 = vector.insert %175, %cst_39 [0, 0] : f16 into vector<1x4xf16>
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %178 = vector.extract %172[%173, %177] : f16 from vector<2x8xf16>
      %179 = vector.insert %178, %176 [0, 1] : f16 into vector<1x4xf16>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %181 = vector.extract %172[%173, %180] : f16 from vector<2x8xf16>
      %182 = vector.insert %181, %179 [0, 2] : f16 into vector<1x4xf16>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %184 = vector.extract %172[%173, %183] : f16 from vector<2x8xf16>
      %185 = vector.insert %184, %182 [0, 3] : f16 into vector<1x4xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %188 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %189 = vector.insert %188, %arg5 [%186, %187] : f16 into vector<2x8xf16>
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %191 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %192 = vector.insert %191, %189 [%186, %190] : f16 into vector<2x8xf16>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %194 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %195 = vector.insert %194, %192 [%186, %193] : f16 into vector<2x8xf16>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %197 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %198 = vector.insert %197, %195 [%186, %196] : f16 into vector<2x8xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %201 = vector.extract %31[0, 0] : f16 from vector<1x4xf16>
      %202 = vector.insert %201, %arg6 [%199, %200] : f16 into vector<2x8xf16>
      %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %204 = vector.extract %31[0, 1] : f16 from vector<1x4xf16>
      %205 = vector.insert %204, %202 [%199, %203] : f16 into vector<2x8xf16>
      %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %207 = vector.extract %31[0, 2] : f16 from vector<1x4xf16>
      %208 = vector.insert %207, %205 [%199, %206] : f16 into vector<2x8xf16>
      %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %210 = vector.extract %31[0, 3] : f16 from vector<1x4xf16>
      %211 = vector.insert %210, %208 [%199, %209] : f16 into vector<2x8xf16>
      %212 = arith.mulf %arg8, %31 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %215 = vector.extract %212[0, 0] : f16 from vector<1x4xf16>
      %216 = vector.insert %215, %arg7 [%213, %214] : f16 into vector<2x8xf16>
      %217 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %218 = vector.extract %212[0, 1] : f16 from vector<1x4xf16>
      %219 = vector.insert %218, %216 [%213, %217] : f16 into vector<2x8xf16>
      %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %221 = vector.extract %212[0, 2] : f16 from vector<1x4xf16>
      %222 = vector.insert %221, %219 [%213, %220] : f16 into vector<2x8xf16>
      %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %224 = vector.extract %212[0, 3] : f16 from vector<1x4xf16>
      %225 = vector.insert %224, %222 [%213, %223] : f16 into vector<2x8xf16>
      affine.yield %198, %211, %225, %185 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %63 = arith.mulf %62#3, %31 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %64 = affine.apply affine_map<() -> (1)>()
    %65 = affine.apply affine_map<() -> (4)>()
    %66 = vector.extract %63[0, 0] : f16 from vector<1x4xf16>
    %67 = vector.insert %66, %62#2 [%64, %65] : f16 into vector<2x8xf16>
    %68 = affine.apply affine_map<() -> (5)>()
    %69 = vector.extract %63[0, 1] : f16 from vector<1x4xf16>
    %70 = vector.insert %69, %67 [%64, %68] : f16 into vector<2x8xf16>
    %71 = affine.apply affine_map<() -> (6)>()
    %72 = vector.extract %63[0, 2] : f16 from vector<1x4xf16>
    %73 = vector.insert %72, %70 [%64, %71] : f16 into vector<2x8xf16>
    %74 = affine.apply affine_map<() -> (7)>()
    %75 = vector.extract %63[0, 3] : f16 from vector<1x4xf16>
    %76 = vector.insert %75, %73 [%64, %74] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %47 : !frisk.async_task
    %77 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %c96) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %c96 floordiv 128) mod 4096, %c96 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %alloc_16 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
    %78 = frisk.async_copy_gm_to_shm %77 to %alloc_16 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
    %79 = frisk.to_threadTile %alloc_14 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %80 = affine.apply affine_map<() -> (0)>()
    %81 = affine.apply affine_map<() -> (0)>()
    %cst_17 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %82 = vector.extract %79[%80, %81] : f16 from vector<2x8xf16>
    %83 = vector.insert %82, %cst_17 [0, 0] : f16 into vector<1x4xf16>
    %84 = affine.apply affine_map<() -> (1)>()
    %85 = vector.extract %79[%80, %84] : f16 from vector<2x8xf16>
    %86 = vector.insert %85, %83 [0, 1] : f16 into vector<1x4xf16>
    %87 = affine.apply affine_map<() -> (2)>()
    %88 = vector.extract %79[%80, %87] : f16 from vector<2x8xf16>
    %89 = vector.insert %88, %86 [0, 2] : f16 into vector<1x4xf16>
    %90 = affine.apply affine_map<() -> (3)>()
    %91 = vector.extract %79[%80, %90] : f16 from vector<2x8xf16>
    %92 = vector.insert %91, %89 [0, 3] : f16 into vector<1x4xf16>
    %93:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_4, %arg6 = %cst_4, %arg7 = %cst_4, %arg8 = %92) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %172 = frisk.to_threadTile %alloc_14 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %173 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %174 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %175 = vector.extract %172[%173, %174] : f16 from vector<2x8xf16>
      %176 = vector.insert %175, %cst_39 [0, 0] : f16 into vector<1x4xf16>
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %178 = vector.extract %172[%173, %177] : f16 from vector<2x8xf16>
      %179 = vector.insert %178, %176 [0, 1] : f16 into vector<1x4xf16>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %181 = vector.extract %172[%173, %180] : f16 from vector<2x8xf16>
      %182 = vector.insert %181, %179 [0, 2] : f16 into vector<1x4xf16>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %184 = vector.extract %172[%173, %183] : f16 from vector<2x8xf16>
      %185 = vector.insert %184, %182 [0, 3] : f16 into vector<1x4xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %188 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %189 = vector.insert %188, %arg5 [%186, %187] : f16 into vector<2x8xf16>
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %191 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %192 = vector.insert %191, %189 [%186, %190] : f16 into vector<2x8xf16>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %194 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %195 = vector.insert %194, %192 [%186, %193] : f16 into vector<2x8xf16>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %197 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %198 = vector.insert %197, %195 [%186, %196] : f16 into vector<2x8xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %201 = vector.extract %31[0, 0] : f16 from vector<1x4xf16>
      %202 = vector.insert %201, %arg6 [%199, %200] : f16 into vector<2x8xf16>
      %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %204 = vector.extract %31[0, 1] : f16 from vector<1x4xf16>
      %205 = vector.insert %204, %202 [%199, %203] : f16 into vector<2x8xf16>
      %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %207 = vector.extract %31[0, 2] : f16 from vector<1x4xf16>
      %208 = vector.insert %207, %205 [%199, %206] : f16 into vector<2x8xf16>
      %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %210 = vector.extract %31[0, 3] : f16 from vector<1x4xf16>
      %211 = vector.insert %210, %208 [%199, %209] : f16 into vector<2x8xf16>
      %212 = arith.mulf %arg8, %31 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %215 = vector.extract %212[0, 0] : f16 from vector<1x4xf16>
      %216 = vector.insert %215, %arg7 [%213, %214] : f16 into vector<2x8xf16>
      %217 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %218 = vector.extract %212[0, 1] : f16 from vector<1x4xf16>
      %219 = vector.insert %218, %216 [%213, %217] : f16 into vector<2x8xf16>
      %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %221 = vector.extract %212[0, 2] : f16 from vector<1x4xf16>
      %222 = vector.insert %221, %219 [%213, %220] : f16 into vector<2x8xf16>
      %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %224 = vector.extract %212[0, 3] : f16 from vector<1x4xf16>
      %225 = vector.insert %224, %222 [%213, %223] : f16 into vector<2x8xf16>
      affine.yield %198, %211, %225, %185 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %94 = arith.mulf %93#3, %31 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %95 = affine.apply affine_map<() -> (1)>()
    %96 = affine.apply affine_map<() -> (4)>()
    %97 = vector.extract %94[0, 0] : f16 from vector<1x4xf16>
    %98 = vector.insert %97, %93#2 [%95, %96] : f16 into vector<2x8xf16>
    %99 = affine.apply affine_map<() -> (5)>()
    %100 = vector.extract %94[0, 1] : f16 from vector<1x4xf16>
    %101 = vector.insert %100, %98 [%95, %99] : f16 into vector<2x8xf16>
    %102 = affine.apply affine_map<() -> (6)>()
    %103 = vector.extract %94[0, 2] : f16 from vector<1x4xf16>
    %104 = vector.insert %103, %101 [%95, %102] : f16 into vector<2x8xf16>
    %105 = affine.apply affine_map<() -> (7)>()
    %106 = vector.extract %94[0, 3] : f16 from vector<1x4xf16>
    %107 = vector.insert %106, %104 [%95, %105] : f16 into vector<2x8xf16>
    frisk.wait_buffer_ready %78 : !frisk.async_task
    %108 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %109 = affine.apply affine_map<() -> (0)>()
    %110 = affine.apply affine_map<() -> (0)>()
    %cst_18 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %111 = vector.extract %108[%109, %110] : f16 from vector<2x8xf16>
    %112 = vector.insert %111, %cst_18 [0, 0] : f16 into vector<1x4xf16>
    %113 = affine.apply affine_map<() -> (1)>()
    %114 = vector.extract %108[%109, %113] : f16 from vector<2x8xf16>
    %115 = vector.insert %114, %112 [0, 1] : f16 into vector<1x4xf16>
    %116 = affine.apply affine_map<() -> (2)>()
    %117 = vector.extract %108[%109, %116] : f16 from vector<2x8xf16>
    %118 = vector.insert %117, %115 [0, 2] : f16 into vector<1x4xf16>
    %119 = affine.apply affine_map<() -> (3)>()
    %120 = vector.extract %108[%109, %119] : f16 from vector<2x8xf16>
    %121 = vector.insert %120, %118 [0, 3] : f16 into vector<1x4xf16>
    %122:4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %cst_4, %arg6 = %cst_4, %arg7 = %cst_4, %arg8 = %121) -> (vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>) {
      %172 = frisk.to_threadTile %alloc_16 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
      %173 = affine.apply affine_map<(d0) -> ((d0 + 1) floordiv 2)>(%arg4)
      %174 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 4)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
      %175 = vector.extract %172[%173, %174] : f16 from vector<2x8xf16>
      %176 = vector.insert %175, %cst_39 [0, 0] : f16 into vector<1x4xf16>
      %177 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 5)>(%arg4)
      %178 = vector.extract %172[%173, %177] : f16 from vector<2x8xf16>
      %179 = vector.insert %178, %176 [0, 1] : f16 into vector<1x4xf16>
      %180 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 6)>(%arg4)
      %181 = vector.extract %172[%173, %180] : f16 from vector<2x8xf16>
      %182 = vector.insert %181, %179 [0, 2] : f16 into vector<1x4xf16>
      %183 = affine.apply affine_map<(d0) -> (d0 * 4 - ((d0 + 1) floordiv 2) * 8 + 7)>(%arg4)
      %184 = vector.extract %172[%173, %183] : f16 from vector<2x8xf16>
      %185 = vector.insert %184, %182 [0, 3] : f16 into vector<1x4xf16>
      %186 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %188 = vector.extract %arg8[0, 0] : f16 from vector<1x4xf16>
      %189 = vector.insert %188, %arg5 [%186, %187] : f16 into vector<2x8xf16>
      %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %191 = vector.extract %arg8[0, 1] : f16 from vector<1x4xf16>
      %192 = vector.insert %191, %189 [%186, %190] : f16 into vector<2x8xf16>
      %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %194 = vector.extract %arg8[0, 2] : f16 from vector<1x4xf16>
      %195 = vector.insert %194, %192 [%186, %193] : f16 into vector<2x8xf16>
      %196 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %197 = vector.extract %arg8[0, 3] : f16 from vector<1x4xf16>
      %198 = vector.insert %197, %195 [%186, %196] : f16 into vector<2x8xf16>
      %199 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %201 = vector.extract %31[0, 0] : f16 from vector<1x4xf16>
      %202 = vector.insert %201, %arg6 [%199, %200] : f16 into vector<2x8xf16>
      %203 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %204 = vector.extract %31[0, 1] : f16 from vector<1x4xf16>
      %205 = vector.insert %204, %202 [%199, %203] : f16 into vector<2x8xf16>
      %206 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %207 = vector.extract %31[0, 2] : f16 from vector<1x4xf16>
      %208 = vector.insert %207, %205 [%199, %206] : f16 into vector<2x8xf16>
      %209 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %210 = vector.extract %31[0, 3] : f16 from vector<1x4xf16>
      %211 = vector.insert %210, %208 [%199, %209] : f16 into vector<2x8xf16>
      %212 = arith.mulf %arg8, %31 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
      %213 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %215 = vector.extract %212[0, 0] : f16 from vector<1x4xf16>
      %216 = vector.insert %215, %arg7 [%213, %214] : f16 into vector<2x8xf16>
      %217 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %218 = vector.extract %212[0, 1] : f16 from vector<1x4xf16>
      %219 = vector.insert %218, %216 [%213, %217] : f16 into vector<2x8xf16>
      %220 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %221 = vector.extract %212[0, 2] : f16 from vector<1x4xf16>
      %222 = vector.insert %221, %219 [%213, %220] : f16 into vector<2x8xf16>
      %223 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %224 = vector.extract %212[0, 3] : f16 from vector<1x4xf16>
      %225 = vector.insert %224, %222 [%213, %223] : f16 into vector<2x8xf16>
      affine.yield %198, %211, %225, %185 : vector<2x8xf16>, vector<2x8xf16>, vector<2x8xf16>, vector<1x4xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.pipelined, frisk.tile_shape = array<i64: 2, 8>}
    %123 = arith.mulf %122#3, %31 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
    %124 = affine.apply affine_map<() -> (1)>()
    %125 = affine.apply affine_map<() -> (4)>()
    %126 = vector.extract %123[0, 0] : f16 from vector<1x4xf16>
    %127 = vector.insert %126, %122#2 [%124, %125] : f16 into vector<2x8xf16>
    %128 = affine.apply affine_map<() -> (5)>()
    %129 = vector.extract %123[0, 1] : f16 from vector<1x4xf16>
    %130 = vector.insert %129, %127 [%124, %128] : f16 into vector<2x8xf16>
    %131 = affine.apply affine_map<() -> (6)>()
    %132 = vector.extract %123[0, 2] : f16 from vector<1x4xf16>
    %133 = vector.insert %132, %130 [%124, %131] : f16 into vector<2x8xf16>
    %134 = affine.apply affine_map<() -> (7)>()
    %135 = vector.extract %123[0, 3] : f16 from vector<1x4xf16>
    %136 = vector.insert %135, %133 [%124, %134] : f16 into vector<2x8xf16>
    %137:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %10, %arg6 = %11, %arg7 = %9#3, %arg8 = %10, %arg9 = %10, %arg10 = %10) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %172 = frisk.to_threadTile %cst_5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %alloc_39 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %173 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c0 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c0 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      %174 = frisk.async_copy_gm_to_shm %173 to %alloc_39 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %174 : !frisk.async_task
      %175 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c32 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c32 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      %alloc_40 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %176 = frisk.async_copy_gm_to_shm %175 to %alloc_40 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %177:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %212 = frisk.to_threadTile %alloc_39 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %213 = affine.apply affine_map<() -> (0)>()
        %214 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %215 = vector.extract %212[%213, %214] : f16 from vector<8x8xf16>
        %216 = vector.insert %215, %cst_52 [0, 0] : f16 into vector<4x1xf16>
        %217 = affine.apply affine_map<() -> (1)>()
        %218 = vector.extract %212[%217, %214] : f16 from vector<8x8xf16>
        %219 = vector.insert %218, %216 [1, 0] : f16 into vector<4x1xf16>
        %220 = affine.apply affine_map<() -> (2)>()
        %221 = vector.extract %212[%220, %214] : f16 from vector<8x8xf16>
        %222 = vector.insert %221, %219 [2, 0] : f16 into vector<4x1xf16>
        %223 = affine.apply affine_map<() -> (3)>()
        %224 = vector.extract %212[%223, %214] : f16 from vector<8x8xf16>
        %225 = vector.insert %224, %222 [3, 0] : f16 into vector<4x1xf16>
        %226:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_2, %arg16 = %225) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %281 = frisk.to_threadTile %alloc_39 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %282 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %283 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %284 = vector.extract %281[%282, %283] : f16 from vector<8x8xf16>
          %285 = vector.insert %284, %cst_55 [0, 0] : f16 into vector<4x1xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %287 = vector.extract %281[%286, %283] : f16 from vector<8x8xf16>
          %288 = vector.insert %287, %285 [1, 0] : f16 into vector<4x1xf16>
          %289 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %290 = vector.extract %281[%289, %283] : f16 from vector<8x8xf16>
          %291 = vector.insert %290, %288 [2, 0] : f16 into vector<4x1xf16>
          %292 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %293 = vector.extract %281[%292, %283] : f16 from vector<8x8xf16>
          %294 = vector.insert %293, %291 [3, 0] : f16 into vector<4x1xf16>
          %295 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %296 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %297 = vector.extract %45[%295, %296] : f16 from vector<2x8xf16>
          %298 = vector.insert %297, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %299 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %300 = vector.extract %45[%295, %299] : f16 from vector<2x8xf16>
          %301 = vector.insert %300, %298 [0, 1] : f16 into vector<1x4xf16>
          %302 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %303 = vector.extract %45[%295, %302] : f16 from vector<2x8xf16>
          %304 = vector.insert %303, %301 [0, 2] : f16 into vector<1x4xf16>
          %305 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %306 = vector.extract %45[%295, %305] : f16 from vector<2x8xf16>
          %307 = vector.insert %306, %304 [0, 3] : f16 into vector<1x4xf16>
          %308 = frisk.warp_mma_rr %307, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %308, %294 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %227 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %228 = affine.apply affine_map<() -> (4)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %229 = vector.extract %45[%227, %228] : f16 from vector<2x8xf16>
        %230 = vector.insert %229, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %231 = affine.apply affine_map<() -> (5)>()
        %232 = vector.extract %45[%227, %231] : f16 from vector<2x8xf16>
        %233 = vector.insert %232, %230 [0, 1] : f16 into vector<1x4xf16>
        %234 = affine.apply affine_map<() -> (6)>()
        %235 = vector.extract %45[%227, %234] : f16 from vector<2x8xf16>
        %236 = vector.insert %235, %233 [0, 2] : f16 into vector<1x4xf16>
        %237 = affine.apply affine_map<() -> (7)>()
        %238 = vector.extract %45[%227, %237] : f16 from vector<2x8xf16>
        %239 = vector.insert %238, %236 [0, 3] : f16 into vector<1x4xf16>
        %240 = frisk.warp_mma_rr %239, %226#1, %226#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %241 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %243 = vector.extract %172[%241, %242] : f32 from vector<2x32xf32>
        %244 = vector.insert %243, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %246 = vector.extract %172[%241, %245] : f32 from vector<2x32xf32>
        %247 = vector.insert %246, %244 [0, 1] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %249 = vector.extract %172[%241, %248] : f32 from vector<2x32xf32>
        %250 = vector.insert %249, %247 [0, 2] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %252 = vector.extract %172[%241, %251] : f32 from vector<2x32xf32>
        %253 = vector.insert %252, %250 [0, 3] : f32 into vector<1x4xf32>
        %254 = arith.addf %253, %240 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %255 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %257 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %arg12 [%255, %256] : f32 into vector<2x32xf32>
        %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %260 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %258 [%255, %259] : f32 into vector<2x32xf32>
        %262 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %263 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %261 [%255, %262] : f32 into vector<2x32xf32>
        %265 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %266 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %264 [%255, %265] : f32 into vector<2x32xf32>
        %268 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %269 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %270 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %271 = vector.insert %270, %arg13 [%268, %269] : f32 into vector<2x32xf32>
        %272 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %273 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %274 = vector.insert %273, %271 [%268, %272] : f32 into vector<2x32xf32>
        %275 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %276 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %277 = vector.insert %276, %274 [%268, %275] : f32 into vector<2x32xf32>
        %278 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %279 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %280 = vector.insert %279, %277 [%268, %278] : f32 into vector<2x32xf32>
        affine.yield %267, %280 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %176 : !frisk.async_task
      %178 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c64 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c64 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      %alloc_41 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %179 = frisk.async_copy_gm_to_shm %178 to %alloc_41 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %180:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %212 = frisk.to_threadTile %alloc_40 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %213 = affine.apply affine_map<() -> (0)>()
        %214 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %215 = vector.extract %212[%213, %214] : f16 from vector<8x8xf16>
        %216 = vector.insert %215, %cst_52 [0, 0] : f16 into vector<4x1xf16>
        %217 = affine.apply affine_map<() -> (1)>()
        %218 = vector.extract %212[%217, %214] : f16 from vector<8x8xf16>
        %219 = vector.insert %218, %216 [1, 0] : f16 into vector<4x1xf16>
        %220 = affine.apply affine_map<() -> (2)>()
        %221 = vector.extract %212[%220, %214] : f16 from vector<8x8xf16>
        %222 = vector.insert %221, %219 [2, 0] : f16 into vector<4x1xf16>
        %223 = affine.apply affine_map<() -> (3)>()
        %224 = vector.extract %212[%223, %214] : f16 from vector<8x8xf16>
        %225 = vector.insert %224, %222 [3, 0] : f16 into vector<4x1xf16>
        %226:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_2, %arg16 = %225) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %281 = frisk.to_threadTile %alloc_40 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %282 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %283 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %284 = vector.extract %281[%282, %283] : f16 from vector<8x8xf16>
          %285 = vector.insert %284, %cst_55 [0, 0] : f16 into vector<4x1xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %287 = vector.extract %281[%286, %283] : f16 from vector<8x8xf16>
          %288 = vector.insert %287, %285 [1, 0] : f16 into vector<4x1xf16>
          %289 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %290 = vector.extract %281[%289, %283] : f16 from vector<8x8xf16>
          %291 = vector.insert %290, %288 [2, 0] : f16 into vector<4x1xf16>
          %292 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %293 = vector.extract %281[%292, %283] : f16 from vector<8x8xf16>
          %294 = vector.insert %293, %291 [3, 0] : f16 into vector<4x1xf16>
          %295 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %296 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %297 = vector.extract %76[%295, %296] : f16 from vector<2x8xf16>
          %298 = vector.insert %297, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %299 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %300 = vector.extract %76[%295, %299] : f16 from vector<2x8xf16>
          %301 = vector.insert %300, %298 [0, 1] : f16 into vector<1x4xf16>
          %302 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %303 = vector.extract %76[%295, %302] : f16 from vector<2x8xf16>
          %304 = vector.insert %303, %301 [0, 2] : f16 into vector<1x4xf16>
          %305 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %306 = vector.extract %76[%295, %305] : f16 from vector<2x8xf16>
          %307 = vector.insert %306, %304 [0, 3] : f16 into vector<1x4xf16>
          %308 = frisk.warp_mma_rr %307, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %308, %294 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %227 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %228 = affine.apply affine_map<() -> (4)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %229 = vector.extract %76[%227, %228] : f16 from vector<2x8xf16>
        %230 = vector.insert %229, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %231 = affine.apply affine_map<() -> (5)>()
        %232 = vector.extract %76[%227, %231] : f16 from vector<2x8xf16>
        %233 = vector.insert %232, %230 [0, 1] : f16 into vector<1x4xf16>
        %234 = affine.apply affine_map<() -> (6)>()
        %235 = vector.extract %76[%227, %234] : f16 from vector<2x8xf16>
        %236 = vector.insert %235, %233 [0, 2] : f16 into vector<1x4xf16>
        %237 = affine.apply affine_map<() -> (7)>()
        %238 = vector.extract %76[%227, %237] : f16 from vector<2x8xf16>
        %239 = vector.insert %238, %236 [0, 3] : f16 into vector<1x4xf16>
        %240 = frisk.warp_mma_rr %239, %226#1, %226#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %241 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %243 = vector.extract %177#1[%241, %242] : f32 from vector<2x32xf32>
        %244 = vector.insert %243, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %246 = vector.extract %177#1[%241, %245] : f32 from vector<2x32xf32>
        %247 = vector.insert %246, %244 [0, 1] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %249 = vector.extract %177#1[%241, %248] : f32 from vector<2x32xf32>
        %250 = vector.insert %249, %247 [0, 2] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %252 = vector.extract %177#1[%241, %251] : f32 from vector<2x32xf32>
        %253 = vector.insert %252, %250 [0, 3] : f32 into vector<1x4xf32>
        %254 = arith.addf %253, %240 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %255 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %257 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %arg12 [%255, %256] : f32 into vector<2x32xf32>
        %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %260 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %258 [%255, %259] : f32 into vector<2x32xf32>
        %262 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %263 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %261 [%255, %262] : f32 into vector<2x32xf32>
        %265 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %266 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %264 [%255, %265] : f32 into vector<2x32xf32>
        %268 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %269 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %270 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %271 = vector.insert %270, %arg13 [%268, %269] : f32 into vector<2x32xf32>
        %272 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %273 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %274 = vector.insert %273, %271 [%268, %272] : f32 into vector<2x32xf32>
        %275 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %276 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %277 = vector.insert %276, %274 [%268, %275] : f32 into vector<2x32xf32>
        %278 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %279 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %280 = vector.insert %279, %277 [%268, %278] : f32 into vector<2x32xf32>
        affine.yield %267, %280 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %179 : !frisk.async_task
      %181 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %c96 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %c96 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
      %alloc_42 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
      %182 = frisk.async_copy_gm_to_shm %181 to %alloc_42 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> !frisk.async_task
      %183:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %212 = frisk.to_threadTile %alloc_41 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %213 = affine.apply affine_map<() -> (0)>()
        %214 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %215 = vector.extract %212[%213, %214] : f16 from vector<8x8xf16>
        %216 = vector.insert %215, %cst_52 [0, 0] : f16 into vector<4x1xf16>
        %217 = affine.apply affine_map<() -> (1)>()
        %218 = vector.extract %212[%217, %214] : f16 from vector<8x8xf16>
        %219 = vector.insert %218, %216 [1, 0] : f16 into vector<4x1xf16>
        %220 = affine.apply affine_map<() -> (2)>()
        %221 = vector.extract %212[%220, %214] : f16 from vector<8x8xf16>
        %222 = vector.insert %221, %219 [2, 0] : f16 into vector<4x1xf16>
        %223 = affine.apply affine_map<() -> (3)>()
        %224 = vector.extract %212[%223, %214] : f16 from vector<8x8xf16>
        %225 = vector.insert %224, %222 [3, 0] : f16 into vector<4x1xf16>
        %226:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_2, %arg16 = %225) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %281 = frisk.to_threadTile %alloc_41 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %282 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %283 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %284 = vector.extract %281[%282, %283] : f16 from vector<8x8xf16>
          %285 = vector.insert %284, %cst_55 [0, 0] : f16 into vector<4x1xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %287 = vector.extract %281[%286, %283] : f16 from vector<8x8xf16>
          %288 = vector.insert %287, %285 [1, 0] : f16 into vector<4x1xf16>
          %289 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %290 = vector.extract %281[%289, %283] : f16 from vector<8x8xf16>
          %291 = vector.insert %290, %288 [2, 0] : f16 into vector<4x1xf16>
          %292 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %293 = vector.extract %281[%292, %283] : f16 from vector<8x8xf16>
          %294 = vector.insert %293, %291 [3, 0] : f16 into vector<4x1xf16>
          %295 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %296 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %297 = vector.extract %107[%295, %296] : f16 from vector<2x8xf16>
          %298 = vector.insert %297, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %299 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %300 = vector.extract %107[%295, %299] : f16 from vector<2x8xf16>
          %301 = vector.insert %300, %298 [0, 1] : f16 into vector<1x4xf16>
          %302 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %303 = vector.extract %107[%295, %302] : f16 from vector<2x8xf16>
          %304 = vector.insert %303, %301 [0, 2] : f16 into vector<1x4xf16>
          %305 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %306 = vector.extract %107[%295, %305] : f16 from vector<2x8xf16>
          %307 = vector.insert %306, %304 [0, 3] : f16 into vector<1x4xf16>
          %308 = frisk.warp_mma_rr %307, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %308, %294 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %227 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %228 = affine.apply affine_map<() -> (4)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %229 = vector.extract %107[%227, %228] : f16 from vector<2x8xf16>
        %230 = vector.insert %229, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %231 = affine.apply affine_map<() -> (5)>()
        %232 = vector.extract %107[%227, %231] : f16 from vector<2x8xf16>
        %233 = vector.insert %232, %230 [0, 1] : f16 into vector<1x4xf16>
        %234 = affine.apply affine_map<() -> (6)>()
        %235 = vector.extract %107[%227, %234] : f16 from vector<2x8xf16>
        %236 = vector.insert %235, %233 [0, 2] : f16 into vector<1x4xf16>
        %237 = affine.apply affine_map<() -> (7)>()
        %238 = vector.extract %107[%227, %237] : f16 from vector<2x8xf16>
        %239 = vector.insert %238, %236 [0, 3] : f16 into vector<1x4xf16>
        %240 = frisk.warp_mma_rr %239, %226#1, %226#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %241 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %243 = vector.extract %180#1[%241, %242] : f32 from vector<2x32xf32>
        %244 = vector.insert %243, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %246 = vector.extract %180#1[%241, %245] : f32 from vector<2x32xf32>
        %247 = vector.insert %246, %244 [0, 1] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %249 = vector.extract %180#1[%241, %248] : f32 from vector<2x32xf32>
        %250 = vector.insert %249, %247 [0, 2] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %252 = vector.extract %180#1[%241, %251] : f32 from vector<2x32xf32>
        %253 = vector.insert %252, %250 [0, 3] : f32 into vector<1x4xf32>
        %254 = arith.addf %253, %240 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %255 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %257 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %arg12 [%255, %256] : f32 into vector<2x32xf32>
        %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %260 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %258 [%255, %259] : f32 into vector<2x32xf32>
        %262 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %263 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %261 [%255, %262] : f32 into vector<2x32xf32>
        %265 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %266 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %264 [%255, %265] : f32 into vector<2x32xf32>
        %268 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %269 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %270 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %271 = vector.insert %270, %arg13 [%268, %269] : f32 into vector<2x32xf32>
        %272 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %273 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %274 = vector.insert %273, %271 [%268, %272] : f32 into vector<2x32xf32>
        %275 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %276 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %277 = vector.insert %276, %274 [%268, %275] : f32 into vector<2x32xf32>
        %278 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %279 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %280 = vector.insert %279, %277 [%268, %278] : f32 into vector<2x32xf32>
        affine.yield %267, %280 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      frisk.wait_buffer_ready %182 : !frisk.async_task
      %184 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %alloc_43 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %185 = frisk.async_copy_gm_to_shm %184 to %alloc_43 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %186:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %212 = frisk.to_threadTile %alloc_42 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prologue", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
        %213 = affine.apply affine_map<() -> (0)>()
        %214 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
        %215 = vector.extract %212[%213, %214] : f16 from vector<8x8xf16>
        %216 = vector.insert %215, %cst_52 [0, 0] : f16 into vector<4x1xf16>
        %217 = affine.apply affine_map<() -> (1)>()
        %218 = vector.extract %212[%217, %214] : f16 from vector<8x8xf16>
        %219 = vector.insert %218, %216 [1, 0] : f16 into vector<4x1xf16>
        %220 = affine.apply affine_map<() -> (2)>()
        %221 = vector.extract %212[%220, %214] : f16 from vector<8x8xf16>
        %222 = vector.insert %221, %219 [2, 0] : f16 into vector<4x1xf16>
        %223 = affine.apply affine_map<() -> (3)>()
        %224 = vector.extract %212[%223, %214] : f16 from vector<8x8xf16>
        %225 = vector.insert %224, %222 [3, 0] : f16 into vector<4x1xf16>
        %226:2 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_2, %arg16 = %225) -> (vector<1x4xf32>, vector<4x1xf16>) {
          %281 = frisk.to_threadTile %alloc_42 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, frisk.pipeline_stage = "prefetch", ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
          %282 = affine.apply affine_map<(d0) -> (d0 * 4 + 4)>(%arg14)
          %283 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %284 = vector.extract %281[%282, %283] : f16 from vector<8x8xf16>
          %285 = vector.insert %284, %cst_55 [0, 0] : f16 into vector<4x1xf16>
          %286 = affine.apply affine_map<(d0) -> (d0 * 4 + 5)>(%arg14)
          %287 = vector.extract %281[%286, %283] : f16 from vector<8x8xf16>
          %288 = vector.insert %287, %285 [1, 0] : f16 into vector<4x1xf16>
          %289 = affine.apply affine_map<(d0) -> (d0 * 4 + 6)>(%arg14)
          %290 = vector.extract %281[%289, %283] : f16 from vector<8x8xf16>
          %291 = vector.insert %290, %288 [2, 0] : f16 into vector<4x1xf16>
          %292 = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%arg14)
          %293 = vector.extract %281[%292, %283] : f16 from vector<8x8xf16>
          %294 = vector.insert %293, %291 [3, 0] : f16 into vector<4x1xf16>
          %295 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
          %296 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %297 = vector.extract %136[%295, %296] : f16 from vector<2x8xf16>
          %298 = vector.insert %297, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %299 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
          %300 = vector.extract %136[%295, %299] : f16 from vector<2x8xf16>
          %301 = vector.insert %300, %298 [0, 1] : f16 into vector<1x4xf16>
          %302 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
          %303 = vector.extract %136[%295, %302] : f16 from vector<2x8xf16>
          %304 = vector.insert %303, %301 [0, 2] : f16 into vector<1x4xf16>
          %305 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
          %306 = vector.extract %136[%295, %305] : f16 from vector<2x8xf16>
          %307 = vector.insert %306, %304 [0, 3] : f16 into vector<1x4xf16>
          %308 = frisk.warp_mma_rr %307, %arg16, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %308, %294 : vector<1x4xf32>, vector<4x1xf16>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, frisk.pipelined, iterLabel = "gemm_k"}
        %227 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %228 = affine.apply affine_map<() -> (4)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %229 = vector.extract %136[%227, %228] : f16 from vector<2x8xf16>
        %230 = vector.insert %229, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %231 = affine.apply affine_map<() -> (5)>()
        %232 = vector.extract %136[%227, %231] : f16 from vector<2x8xf16>
        %233 = vector.insert %232, %230 [0, 1] : f16 into vector<1x4xf16>
        %234 = affine.apply affine_map<() -> (6)>()
        %235 = vector.extract %136[%227, %234] : f16 from vector<2x8xf16>
        %236 = vector.insert %235, %233 [0, 2] : f16 into vector<1x4xf16>
        %237 = affine.apply affine_map<() -> (7)>()
        %238 = vector.extract %136[%227, %237] : f16 from vector<2x8xf16>
        %239 = vector.insert %238, %236 [0, 3] : f16 into vector<1x4xf16>
        %240 = frisk.warp_mma_rr %239, %226#1, %226#0 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
        %241 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %243 = vector.extract %183#1[%241, %242] : f32 from vector<2x32xf32>
        %244 = vector.insert %243, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %246 = vector.extract %183#1[%241, %245] : f32 from vector<2x32xf32>
        %247 = vector.insert %246, %244 [0, 1] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %249 = vector.extract %183#1[%241, %248] : f32 from vector<2x32xf32>
        %250 = vector.insert %249, %247 [0, 2] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %252 = vector.extract %183#1[%241, %251] : f32 from vector<2x32xf32>
        %253 = vector.insert %252, %250 [0, 3] : f32 into vector<1x4xf32>
        %254 = arith.addf %253, %240 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %255 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %256 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %257 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %258 = vector.insert %257, %arg12 [%255, %256] : f32 into vector<2x32xf32>
        %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %260 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %258 [%255, %259] : f32 into vector<2x32xf32>
        %262 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %263 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %261 [%255, %262] : f32 into vector<2x32xf32>
        %265 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %266 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %264 [%255, %265] : f32 into vector<2x32xf32>
        %268 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %269 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %270 = vector.extract %254[0, 0] : f32 from vector<1x4xf32>
        %271 = vector.insert %270, %arg13 [%268, %269] : f32 into vector<2x32xf32>
        %272 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %273 = vector.extract %254[0, 1] : f32 from vector<1x4xf32>
        %274 = vector.insert %273, %271 [%268, %272] : f32 into vector<2x32xf32>
        %275 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %276 = vector.extract %254[0, 2] : f32 from vector<1x4xf32>
        %277 = vector.insert %276, %274 [%268, %275] : f32 into vector<2x32xf32>
        %278 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %279 = vector.extract %254[0, 3] : f32 from vector<1x4xf32>
        %280 = vector.insert %279, %277 [%268, %278] : f32 into vector<2x32xf32>
        affine.yield %267, %280 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, tiled = true}
      %187 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3) -> (vector<2x32xf32>) {
        %212 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %214 = vector.extract %186#1[%212, %213] : f32 from vector<2x32xf32>
        %215 = vector.insert %214, %cst_52 [0, 0] : f32 into vector<1x4xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %217 = vector.extract %186#1[%212, %216] : f32 from vector<2x32xf32>
        %218 = vector.insert %217, %215 [0, 1] : f32 into vector<1x4xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %220 = vector.extract %186#1[%212, %219] : f32 from vector<2x32xf32>
        %221 = vector.insert %220, %218 [0, 2] : f32 into vector<1x4xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %223 = vector.extract %186#1[%212, %222] : f32 from vector<2x32xf32>
        %224 = vector.insert %223, %221 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %227 = vector.extract %224[0, 0] : f32 from vector<1x4xf32>
        %228 = vector.insert %227, %arg12 [%225, %226] : f32 into vector<2x32xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %230 = vector.extract %224[0, 1] : f32 from vector<1x4xf32>
        %231 = vector.insert %230, %228 [%225, %229] : f32 into vector<2x32xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %233 = vector.extract %224[0, 2] : f32 from vector<1x4xf32>
        %234 = vector.insert %233, %231 [%225, %232] : f32 into vector<2x32xf32>
        %235 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %236 = vector.extract %224[0, 3] : f32 from vector<1x4xf32>
        %237 = vector.insert %236, %234 [%225, %235] : f32 into vector<2x32xf32>
        affine.yield %237 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_44 = arith.constant 0xFF800000 : f32
      %cst_45 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %188 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_45) -> (vector<2x1xf32>) {
        %c0_52 = arith.constant 0 : index
        %212 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %213 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_44) -> (f32) {
          %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %220 = vector.extract %187[%212, %219] : f32 from vector<2x32xf32>
          %221 = arith.maxnumf %arg14, %220 : f32
          %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %223 = vector.extract %187[%212, %222] : f32 from vector<2x32xf32>
          %224 = arith.maxnumf %221, %223 : f32
          %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %226 = vector.extract %187[%212, %225] : f32 from vector<2x32xf32>
          %227 = arith.maxnumf %224, %226 : f32
          %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %229 = vector.extract %187[%212, %228] : f32 from vector<2x32xf32>
          %230 = arith.maxnumf %227, %229 : f32
          affine.yield %230 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %c64_i32_54 = arith.constant 64 : i32
        %c16_i32_55 = arith.constant 16 : i32
        %shuffleResult, %valid = gpu.shuffle xor %213, %c16_i32_55, %c64_i32_54 : f32
        %214 = arith.maxnumf %213, %shuffleResult : f32
        %c64_i32_56 = arith.constant 64 : i32
        %c32_i32_57 = arith.constant 32 : i32
        %shuffleResult_58, %valid_59 = gpu.shuffle xor %214, %c32_i32_57, %c64_i32_56 : f32
        %215 = arith.maxnumf %214, %shuffleResult_58 : f32
        %216 = vector.insert %215, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %217 = vector.extract %216[0, 0] : f32 from vector<1x1xf32>
        %218 = vector.insert %217, %arg12 [%212, %c0_52] : f32 into vector<2x1xf32>
        affine.yield %218 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %189 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %212 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %213 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %214 = vector.extract %188[%212, %213] : f32 from vector<2x1xf32>
        %215 = vector.insert %214, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %216 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %217 = affine.apply affine_map<() -> (0)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %218 = vector.extract %arg7[%216, %217] : f32 from vector<2x1xf32>
        %219 = vector.insert %218, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %220 = arith.maxnumf %215, %219 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
        %221 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %222 = affine.apply affine_map<() -> (0)>()
        %223 = vector.extract %220[0, 0] : f32 from vector<1x1xf32>
        %224 = vector.insert %223, %arg12 [%221, %222] : f32 into vector<2x1xf32>
        affine.yield %224 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %190:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %212 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %213 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %214 = vector.extract %189[%212, %213] : f32 from vector<2x1xf32>
        %215 = vector.insert %214, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %216 = affine.for %arg14 = 0 to 1 iter_args(%arg15 = %cst_53) -> (vector<1x4xf32>) {
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %257 = vector.extract %215[%c0_58, %c0_59] : f32 from vector<1x1xf32>
          %258 = vector.insert %257, %cst_57 [0, 0] : f32 into vector<1x4xf32>
          %c0_60 = arith.constant 0 : index
          %c0_61 = arith.constant 0 : index
          %259 = vector.extract %215[%c0_60, %c0_61] : f32 from vector<1x1xf32>
          %260 = vector.insert %259, %258 [0, 1] : f32 into vector<1x4xf32>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %261 = vector.extract %215[%c0_62, %c0_63] : f32 from vector<1x1xf32>
          %262 = vector.insert %261, %260 [0, 2] : f32 into vector<1x4xf32>
          %c0_64 = arith.constant 0 : index
          %c0_65 = arith.constant 0 : index
          %263 = vector.extract %215[%c0_64, %c0_65] : f32 from vector<1x1xf32>
          %264 = vector.insert %263, %262 [0, 3] : f32 into vector<1x4xf32>
          %265 = vector.extract %264[0, 0] : f32 from vector<1x4xf32>
          %266 = vector.insert %265, %arg15 [%c0_56, %c0_55] : f32 into vector<1x4xf32>
          %267 = affine.apply affine_map<() -> (1)>()
          %268 = vector.extract %264[0, 1] : f32 from vector<1x4xf32>
          %269 = vector.insert %268, %266 [%c0_56, %267] : f32 into vector<1x4xf32>
          %270 = affine.apply affine_map<() -> (2)>()
          %271 = vector.extract %264[0, 2] : f32 from vector<1x4xf32>
          %272 = vector.insert %271, %269 [%c0_56, %270] : f32 into vector<1x4xf32>
          %273 = affine.apply affine_map<() -> (3)>()
          %274 = vector.extract %264[0, 3] : f32 from vector<1x4xf32>
          %275 = vector.insert %274, %272 [%c0_56, %273] : f32 into vector<1x4xf32>
          affine.yield %275 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
        %217 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %218 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %219 = vector.extract %216[0, 0] : f32 from vector<1x4xf32>
        %220 = vector.insert %219, %arg12 [%217, %218] : f32 into vector<2x32xf32>
        %221 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %222 = vector.extract %216[0, 1] : f32 from vector<1x4xf32>
        %223 = vector.insert %222, %220 [%217, %221] : f32 into vector<2x32xf32>
        %224 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %225 = vector.extract %216[0, 2] : f32 from vector<1x4xf32>
        %226 = vector.insert %225, %223 [%217, %224] : f32 into vector<2x32xf32>
        %227 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %228 = vector.extract %216[0, 3] : f32 from vector<1x4xf32>
        %229 = vector.insert %228, %226 [%217, %227] : f32 into vector<2x32xf32>
        %230 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %231 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %232 = vector.extract %186#1[%230, %231] : f32 from vector<2x32xf32>
        %233 = vector.insert %232, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %234 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %235 = vector.extract %186#1[%230, %234] : f32 from vector<2x32xf32>
        %236 = vector.insert %235, %233 [0, 1] : f32 into vector<1x4xf32>
        %237 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %238 = vector.extract %186#1[%230, %237] : f32 from vector<2x32xf32>
        %239 = vector.insert %238, %236 [0, 2] : f32 into vector<1x4xf32>
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %241 = vector.extract %186#1[%230, %240] : f32 from vector<2x32xf32>
        %242 = vector.insert %241, %239 [0, 3] : f32 into vector<1x4xf32>
        %243 = arith.subf %242, %216 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
        %244 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %246 = vector.extract %243[0, 0] : f32 from vector<1x4xf32>
        %247 = vector.insert %246, %arg13 [%244, %245] : f32 into vector<2x32xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %249 = vector.extract %243[0, 1] : f32 from vector<1x4xf32>
        %250 = vector.insert %249, %247 [%244, %248] : f32 into vector<2x32xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %252 = vector.extract %243[0, 2] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %250 [%244, %251] : f32 into vector<2x32xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %255 = vector.extract %243[0, 3] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %253 [%244, %254] : f32 into vector<2x32xf32>
        affine.yield %229, %256 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %191:3 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9, %arg14 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>) {
        %212 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %213 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %214 = vector.extract %arg7[%212, %213] : f32 from vector<2x1xf32>
        %215 = vector.insert %214, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %216 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %217 = affine.apply affine_map<() -> (0)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %218 = vector.extract %189[%216, %217] : f32 from vector<2x1xf32>
        %219 = vector.insert %218, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %220 = arith.subf %215, %219 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
        %221 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %222 = affine.apply affine_map<() -> (0)>()
        %223 = vector.extract %220[0, 0] : f32 from vector<1x1xf32>
        %224 = vector.insert %223, %arg12 [%221, %222] : f32 into vector<2x1xf32>
        %225 = math.exp2 %220 {tiled = true} : vector<1x1xf32>
        %226 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %227 = affine.apply affine_map<() -> (0)>()
        %228 = vector.extract %225[0, 0] : f32 from vector<1x1xf32>
        %229 = vector.insert %228, %arg13 [%226, %227] : f32 into vector<2x1xf32>
        %230 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %231 = affine.apply affine_map<() -> (0)>()
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %232 = vector.extract %arg6[%230, %231] : f32 from vector<2x1xf32>
        %233 = vector.insert %232, %cst_54 [0, 0] : f32 into vector<1x1xf32>
        %234 = arith.mulf %233, %225 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
        %235 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %236 = affine.apply affine_map<() -> (0)>()
        %237 = vector.extract %234[0, 0] : f32 from vector<1x1xf32>
        %238 = vector.insert %237, %arg14 [%235, %236] : f32 into vector<2x1xf32>
        affine.yield %224, %229, %238 : vector<2x1xf32>, vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %192:2 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (vector<2x32xf32>, vector<2x32xf32>) {
        %212 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %214 = vector.extract %190#1[%212, %213] : f32 from vector<2x32xf32>
        %215 = vector.insert %214, %cst_52 [0, 0] : f32 into vector<1x4xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %217 = vector.extract %190#1[%212, %216] : f32 from vector<2x32xf32>
        %218 = vector.insert %217, %215 [0, 1] : f32 into vector<1x4xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %220 = vector.extract %190#1[%212, %219] : f32 from vector<2x32xf32>
        %221 = vector.insert %220, %218 [0, 2] : f32 into vector<1x4xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %223 = vector.extract %190#1[%212, %222] : f32 from vector<2x32xf32>
        %224 = vector.insert %223, %221 [0, 3] : f32 into vector<1x4xf32>
        %225 = math.exp2 %224 {tiled = true} : vector<1x4xf32>
        %226 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %227 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %228 = vector.extract %225[0, 0] : f32 from vector<1x4xf32>
        %229 = vector.insert %228, %arg12 [%226, %227] : f32 into vector<2x32xf32>
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %231 = vector.extract %225[0, 1] : f32 from vector<1x4xf32>
        %232 = vector.insert %231, %229 [%226, %230] : f32 into vector<2x32xf32>
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %234 = vector.extract %225[0, 2] : f32 from vector<1x4xf32>
        %235 = vector.insert %234, %232 [%226, %233] : f32 into vector<2x32xf32>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %237 = vector.extract %225[0, 3] : f32 from vector<1x4xf32>
        %238 = vector.insert %237, %235 [%226, %236] : f32 into vector<2x32xf32>
        %239 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %241 = vector.extract %225[0, 0] : f32 from vector<1x4xf32>
        %242 = vector.insert %241, %arg13 [%239, %240] : f32 into vector<2x32xf32>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %244 = vector.extract %225[0, 1] : f32 from vector<1x4xf32>
        %245 = vector.insert %244, %242 [%239, %243] : f32 into vector<2x32xf32>
        %246 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %247 = vector.extract %225[0, 2] : f32 from vector<1x4xf32>
        %248 = vector.insert %247, %245 [%239, %246] : f32 into vector<2x32xf32>
        %249 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %250 = vector.extract %225[0, 3] : f32 from vector<1x4xf32>
        %251 = vector.insert %250, %248 [%239, %249] : f32 into vector<2x32xf32>
        affine.yield %238, %251 : vector<2x32xf32>, vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_46 = arith.constant 0.000000e+00 : f32
      %cst_47 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %193 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_47) -> (vector<2x1xf32>) {
        %c0_52 = arith.constant 0 : index
        %212 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %213 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_46) -> (f32) {
          %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %220 = vector.extract %192#1[%212, %219] : f32 from vector<2x32xf32>
          %221 = arith.addf %arg14, %220 : f32
          %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %223 = vector.extract %192#1[%212, %222] : f32 from vector<2x32xf32>
          %224 = arith.addf %221, %223 : f32
          %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %226 = vector.extract %192#1[%212, %225] : f32 from vector<2x32xf32>
          %227 = arith.addf %224, %226 : f32
          %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %229 = vector.extract %192#1[%212, %228] : f32 from vector<2x32xf32>
          %230 = arith.addf %227, %229 : f32
          affine.yield %230 : f32
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
        %c64_i32_54 = arith.constant 64 : i32
        %c16_i32_55 = arith.constant 16 : i32
        %shuffleResult, %valid = gpu.shuffle xor %213, %c16_i32_55, %c64_i32_54 : f32
        %214 = arith.addf %213, %shuffleResult : f32
        %c64_i32_56 = arith.constant 64 : i32
        %c32_i32_57 = arith.constant 32 : i32
        %shuffleResult_58, %valid_59 = gpu.shuffle xor %214, %c32_i32_57, %c64_i32_56 : f32
        %215 = arith.addf %214, %shuffleResult_58 : f32
        %216 = vector.insert %215, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %217 = vector.extract %216[0, 0] : f32 from vector<1x1xf32>
        %218 = vector.insert %217, %arg12 [%212, %c0_52] : f32 into vector<2x1xf32>
        affine.yield %218 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %194 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x1xf32>) {
        %212 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %213 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %214 = vector.extract %193[%212, %213] : f32 from vector<2x1xf32>
        %215 = vector.insert %214, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %216 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %217 = affine.apply affine_map<() -> (0)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %218 = vector.extract %191#2[%216, %217] : f32 from vector<2x1xf32>
        %219 = vector.insert %218, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %220 = arith.addf %215, %219 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        %221 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %222 = affine.apply affine_map<() -> (0)>()
        %223 = vector.extract %220[0, 0] : f32 from vector<1x1xf32>
        %224 = vector.insert %223, %arg12 [%221, %222] : f32 into vector<2x1xf32>
        affine.yield %224 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %195 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_1) -> (vector<2x32xf16>) {
        %212 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %214 = vector.extract %192#0[%212, %213] : f32 from vector<2x32xf32>
        %215 = vector.insert %214, %cst_52 [0, 0] : f32 into vector<1x4xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %217 = vector.extract %192#0[%212, %216] : f32 from vector<2x32xf32>
        %218 = vector.insert %217, %215 [0, 1] : f32 into vector<1x4xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %220 = vector.extract %192#0[%212, %219] : f32 from vector<2x32xf32>
        %221 = vector.insert %220, %218 [0, 2] : f32 into vector<1x4xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %223 = vector.extract %192#0[%212, %222] : f32 from vector<2x32xf32>
        %224 = vector.insert %223, %221 [0, 3] : f32 into vector<1x4xf32>
        %225 = arith.truncf %224 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        %226 = affine.apply affine_map<(d0) -> (d0 floordiv 8)>(%arg11)
        %227 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %228 = vector.extract %225[0, 0] : f16 from vector<1x4xf16>
        %229 = vector.insert %228, %arg12 [%226, %227] : f16 into vector<2x32xf16>
        %230 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %231 = vector.extract %225[0, 1] : f16 from vector<1x4xf16>
        %232 = vector.insert %231, %229 [%226, %230] : f16 into vector<2x32xf16>
        %233 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %234 = vector.extract %225[0, 2] : f16 from vector<1x4xf16>
        %235 = vector.insert %234, %232 [%226, %233] : f16 into vector<2x32xf16>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %237 = vector.extract %225[0, 3] : f16 from vector<1x4xf16>
        %238 = vector.insert %237, %235 [%226, %236] : f16 into vector<2x32xf16>
        affine.yield %238 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %thread_id_x_48 = gpu.thread_id x
      frisk.wait_buffer_ready %185 : !frisk.async_task
      %196 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %212 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_48]
        %213 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_48]
        %214 = frisk.read_matrix_from_shm %alloc_43[%212, %213] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %215 = vector.bitcast %214 : vector<4xi32> to vector<8xf16>
        %216 = vector.extract %215[0] : f16 from vector<8xf16>
        %217 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_48]
        %218 = vector.extract %215[%217] : f16 from vector<8xf16>
        %219 = vector.extract %215[1] : f16 from vector<8xf16>
        %220 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_48]
        %221 = vector.extract %215[%220] : f16 from vector<8xf16>
        %222 = vector.extract %215[2] : f16 from vector<8xf16>
        %223 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_48]
        %224 = vector.extract %215[%223] : f16 from vector<8xf16>
        %225 = vector.extract %215[3] : f16 from vector<8xf16>
        %226 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_48]
        %227 = vector.extract %215[%226] : f16 from vector<8xf16>
        %228 = vector.from_elements %216, %218, %219, %221, %222, %224, %225, %227 : vector<4x2xf16>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %230 = affine.apply affine_map<() -> (0)>()
        %231 = vector.extract %228[0, 0] : f16 from vector<4x2xf16>
        %232 = vector.insert %231, %arg12 [%229, %230] : f16 into vector<32x2xf16>
        %233 = affine.apply affine_map<() -> (1)>()
        %234 = vector.extract %228[0, 1] : f16 from vector<4x2xf16>
        %235 = vector.insert %234, %232 [%229, %233] : f16 into vector<32x2xf16>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %237 = vector.extract %228[1, 0] : f16 from vector<4x2xf16>
        %238 = vector.insert %237, %235 [%236, %230] : f16 into vector<32x2xf16>
        %239 = affine.apply affine_map<() -> (1)>()
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %241 = vector.extract %228[1, 1] : f16 from vector<4x2xf16>
        %242 = vector.insert %241, %238 [%240, %239] : f16 into vector<32x2xf16>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %244 = vector.extract %228[2, 0] : f16 from vector<4x2xf16>
        %245 = vector.insert %244, %242 [%243, %230] : f16 into vector<32x2xf16>
        %246 = affine.apply affine_map<() -> (1)>()
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %248 = vector.extract %228[2, 1] : f16 from vector<4x2xf16>
        %249 = vector.insert %248, %245 [%247, %246] : f16 into vector<32x2xf16>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %251 = vector.extract %228[3, 0] : f16 from vector<4x2xf16>
        %252 = vector.insert %251, %249 [%250, %230] : f16 into vector<32x2xf16>
        %253 = affine.apply affine_map<() -> (1)>()
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %255 = vector.extract %228[3, 1] : f16 from vector<4x2xf16>
        %256 = vector.insert %255, %252 [%254, %253] : f16 into vector<32x2xf16>
        affine.yield %256 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %197 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %alloc_49 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %198 = frisk.async_copy_gm_to_shm %197 to %alloc_49 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %199:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %212 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_2) -> (vector<1x4xf32>) {
          %285 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %286 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %287 = vector.extract %195[%285, %286] : f16 from vector<2x32xf16>
          %288 = vector.insert %287, %cst_55 [0, 0] : f16 into vector<1x4xf16>
          %289 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %290 = vector.extract %195[%285, %289] : f16 from vector<2x32xf16>
          %291 = vector.insert %290, %288 [0, 1] : f16 into vector<1x4xf16>
          %292 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %293 = vector.extract %195[%285, %292] : f16 from vector<2x32xf16>
          %294 = vector.insert %293, %291 [0, 2] : f16 into vector<1x4xf16>
          %295 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %296 = vector.extract %195[%285, %295] : f16 from vector<2x32xf16>
          %297 = vector.insert %296, %294 [0, 3] : f16 into vector<1x4xf16>
          %298 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %299 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %300 = vector.extract %196[%298, %299] : f16 from vector<32x2xf16>
          %301 = vector.insert %300, %cst_56 [0, 0] : f16 into vector<4x1xf16>
          %302 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %303 = vector.extract %196[%302, %299] : f16 from vector<32x2xf16>
          %304 = vector.insert %303, %301 [1, 0] : f16 into vector<4x1xf16>
          %305 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %306 = vector.extract %196[%305, %299] : f16 from vector<32x2xf16>
          %307 = vector.insert %306, %304 [2, 0] : f16 into vector<4x1xf16>
          %308 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %309 = vector.extract %196[%308, %299] : f16 from vector<32x2xf16>
          %310 = vector.insert %309, %307 [3, 0] : f16 into vector<4x1xf16>
          %311 = frisk.warp_mma_rr %297, %310, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %311 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
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
        %227 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %228 = vector.extract %191#1[%226, %227] : f32 from vector<2x1xf32>
        %229 = vector.insert %228, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %230 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_53) -> (vector<1x4xf32>) {
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %285 = vector.extract %229[%c0_58, %c0_59] : f32 from vector<1x1xf32>
          %286 = vector.insert %285, %cst_57 [0, 0] : f32 into vector<1x4xf32>
          %c0_60 = arith.constant 0 : index
          %c0_61 = arith.constant 0 : index
          %287 = vector.extract %229[%c0_60, %c0_61] : f32 from vector<1x1xf32>
          %288 = vector.insert %287, %286 [0, 1] : f32 into vector<1x4xf32>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %289 = vector.extract %229[%c0_62, %c0_63] : f32 from vector<1x1xf32>
          %290 = vector.insert %289, %288 [0, 2] : f32 into vector<1x4xf32>
          %c0_64 = arith.constant 0 : index
          %c0_65 = arith.constant 0 : index
          %291 = vector.extract %229[%c0_64, %c0_65] : f32 from vector<1x1xf32>
          %292 = vector.insert %291, %290 [0, 3] : f32 into vector<1x4xf32>
          %293 = vector.extract %292[0, 0] : f32 from vector<1x4xf32>
          %294 = vector.insert %293, %arg17 [%c0_56, %c0_55] : f32 into vector<1x4xf32>
          %295 = affine.apply affine_map<() -> (1)>()
          %296 = vector.extract %292[0, 1] : f32 from vector<1x4xf32>
          %297 = vector.insert %296, %294 [%c0_56, %295] : f32 into vector<1x4xf32>
          %298 = affine.apply affine_map<() -> (2)>()
          %299 = vector.extract %292[0, 2] : f32 from vector<1x4xf32>
          %300 = vector.insert %299, %297 [%c0_56, %298] : f32 into vector<1x4xf32>
          %301 = affine.apply affine_map<() -> (3)>()
          %302 = vector.extract %292[0, 3] : f32 from vector<1x4xf32>
          %303 = vector.insert %302, %300 [%c0_56, %301] : f32 into vector<1x4xf32>
          affine.yield %303 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
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
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %246 = vector.extract %arg5[%244, %245] : f32 from vector<2x8xf32>
        %247 = vector.insert %246, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %249 = vector.extract %arg5[%244, %248] : f32 from vector<2x8xf32>
        %250 = vector.insert %249, %247 [0, 1] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %252 = vector.extract %arg5[%244, %251] : f32 from vector<2x8xf32>
        %253 = vector.insert %252, %250 [0, 2] : f32 into vector<1x4xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %255 = vector.extract %arg5[%244, %254] : f32 from vector<2x8xf32>
        %256 = vector.insert %255, %253 [0, 3] : f32 into vector<1x4xf32>
        %257 = arith.mulf %256, %230 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %258 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %260 = vector.extract %257[0, 0] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %arg14 [%258, %259] : f32 into vector<2x8xf32>
        %262 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %263 = vector.extract %257[0, 1] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %261 [%258, %262] : f32 into vector<2x8xf32>
        %265 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %266 = vector.extract %257[0, 2] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %264 [%258, %265] : f32 into vector<2x8xf32>
        %268 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %269 = vector.extract %257[0, 3] : f32 from vector<1x4xf32>
        %270 = vector.insert %269, %267 [%258, %268] : f32 into vector<2x8xf32>
        %271 = arith.addf %257, %212 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %272 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %273 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %274 = vector.extract %271[0, 0] : f32 from vector<1x4xf32>
        %275 = vector.insert %274, %arg15 [%272, %273] : f32 into vector<2x8xf32>
        %276 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %277 = vector.extract %271[0, 1] : f32 from vector<1x4xf32>
        %278 = vector.insert %277, %275 [%272, %276] : f32 into vector<2x8xf32>
        %279 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %280 = vector.extract %271[0, 2] : f32 from vector<1x4xf32>
        %281 = vector.insert %280, %278 [%272, %279] : f32 into vector<2x8xf32>
        %282 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %283 = vector.extract %271[0, 3] : f32 from vector<1x4xf32>
        %284 = vector.insert %283, %281 [%272, %282] : f32 into vector<2x8xf32>
        affine.yield %225, %243, %270, %284 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %198 : !frisk.async_task
      %200 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %212 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_48]
        %213 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_48]
        %214 = frisk.read_matrix_from_shm %alloc_49[%212, %213] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %215 = vector.bitcast %214 : vector<4xi32> to vector<8xf16>
        %216 = vector.extract %215[0] : f16 from vector<8xf16>
        %217 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_48]
        %218 = vector.extract %215[%217] : f16 from vector<8xf16>
        %219 = vector.extract %215[1] : f16 from vector<8xf16>
        %220 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_48]
        %221 = vector.extract %215[%220] : f16 from vector<8xf16>
        %222 = vector.extract %215[2] : f16 from vector<8xf16>
        %223 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_48]
        %224 = vector.extract %215[%223] : f16 from vector<8xf16>
        %225 = vector.extract %215[3] : f16 from vector<8xf16>
        %226 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_48]
        %227 = vector.extract %215[%226] : f16 from vector<8xf16>
        %228 = vector.from_elements %216, %218, %219, %221, %222, %224, %225, %227 : vector<4x2xf16>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %230 = affine.apply affine_map<() -> (0)>()
        %231 = vector.extract %228[0, 0] : f16 from vector<4x2xf16>
        %232 = vector.insert %231, %arg12 [%229, %230] : f16 into vector<32x2xf16>
        %233 = affine.apply affine_map<() -> (1)>()
        %234 = vector.extract %228[0, 1] : f16 from vector<4x2xf16>
        %235 = vector.insert %234, %232 [%229, %233] : f16 into vector<32x2xf16>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %237 = vector.extract %228[1, 0] : f16 from vector<4x2xf16>
        %238 = vector.insert %237, %235 [%236, %230] : f16 into vector<32x2xf16>
        %239 = affine.apply affine_map<() -> (1)>()
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %241 = vector.extract %228[1, 1] : f16 from vector<4x2xf16>
        %242 = vector.insert %241, %238 [%240, %239] : f16 into vector<32x2xf16>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %244 = vector.extract %228[2, 0] : f16 from vector<4x2xf16>
        %245 = vector.insert %244, %242 [%243, %230] : f16 into vector<32x2xf16>
        %246 = affine.apply affine_map<() -> (1)>()
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %248 = vector.extract %228[2, 1] : f16 from vector<4x2xf16>
        %249 = vector.insert %248, %245 [%247, %246] : f16 into vector<32x2xf16>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %251 = vector.extract %228[3, 0] : f16 from vector<4x2xf16>
        %252 = vector.insert %251, %249 [%250, %230] : f16 into vector<32x2xf16>
        %253 = affine.apply affine_map<() -> (1)>()
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %255 = vector.extract %228[3, 1] : f16 from vector<4x2xf16>
        %256 = vector.insert %255, %252 [%254, %253] : f16 into vector<32x2xf16>
        affine.yield %256 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %201 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %alloc_50 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %202 = frisk.async_copy_gm_to_shm %201 to %alloc_50 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %203:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %212 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_2) -> (vector<1x4xf32>) {
          %285 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %286 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %287 = vector.extract %195[%285, %286] : f16 from vector<2x32xf16>
          %288 = vector.insert %287, %cst_55 [0, 0] : f16 into vector<1x4xf16>
          %289 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %290 = vector.extract %195[%285, %289] : f16 from vector<2x32xf16>
          %291 = vector.insert %290, %288 [0, 1] : f16 into vector<1x4xf16>
          %292 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %293 = vector.extract %195[%285, %292] : f16 from vector<2x32xf16>
          %294 = vector.insert %293, %291 [0, 2] : f16 into vector<1x4xf16>
          %295 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %296 = vector.extract %195[%285, %295] : f16 from vector<2x32xf16>
          %297 = vector.insert %296, %294 [0, 3] : f16 into vector<1x4xf16>
          %298 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %299 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %300 = vector.extract %200[%298, %299] : f16 from vector<32x2xf16>
          %301 = vector.insert %300, %cst_56 [0, 0] : f16 into vector<4x1xf16>
          %302 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %303 = vector.extract %200[%302, %299] : f16 from vector<32x2xf16>
          %304 = vector.insert %303, %301 [1, 0] : f16 into vector<4x1xf16>
          %305 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %306 = vector.extract %200[%305, %299] : f16 from vector<32x2xf16>
          %307 = vector.insert %306, %304 [2, 0] : f16 into vector<4x1xf16>
          %308 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %309 = vector.extract %200[%308, %299] : f16 from vector<32x2xf16>
          %310 = vector.insert %309, %307 [3, 0] : f16 into vector<4x1xf16>
          %311 = frisk.warp_mma_rr %297, %310, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %311 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
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
        %227 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %228 = vector.extract %191#1[%226, %227] : f32 from vector<2x1xf32>
        %229 = vector.insert %228, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %230 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_53) -> (vector<1x4xf32>) {
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %285 = vector.extract %229[%c0_58, %c0_59] : f32 from vector<1x1xf32>
          %286 = vector.insert %285, %cst_57 [0, 0] : f32 into vector<1x4xf32>
          %c0_60 = arith.constant 0 : index
          %c0_61 = arith.constant 0 : index
          %287 = vector.extract %229[%c0_60, %c0_61] : f32 from vector<1x1xf32>
          %288 = vector.insert %287, %286 [0, 1] : f32 into vector<1x4xf32>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %289 = vector.extract %229[%c0_62, %c0_63] : f32 from vector<1x1xf32>
          %290 = vector.insert %289, %288 [0, 2] : f32 into vector<1x4xf32>
          %c0_64 = arith.constant 0 : index
          %c0_65 = arith.constant 0 : index
          %291 = vector.extract %229[%c0_64, %c0_65] : f32 from vector<1x1xf32>
          %292 = vector.insert %291, %290 [0, 3] : f32 into vector<1x4xf32>
          %293 = vector.extract %292[0, 0] : f32 from vector<1x4xf32>
          %294 = vector.insert %293, %arg17 [%c0_56, %c0_55] : f32 into vector<1x4xf32>
          %295 = affine.apply affine_map<() -> (1)>()
          %296 = vector.extract %292[0, 1] : f32 from vector<1x4xf32>
          %297 = vector.insert %296, %294 [%c0_56, %295] : f32 into vector<1x4xf32>
          %298 = affine.apply affine_map<() -> (2)>()
          %299 = vector.extract %292[0, 2] : f32 from vector<1x4xf32>
          %300 = vector.insert %299, %297 [%c0_56, %298] : f32 into vector<1x4xf32>
          %301 = affine.apply affine_map<() -> (3)>()
          %302 = vector.extract %292[0, 3] : f32 from vector<1x4xf32>
          %303 = vector.insert %302, %300 [%c0_56, %301] : f32 into vector<1x4xf32>
          affine.yield %303 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
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
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %246 = vector.extract %arg8[%244, %245] : f32 from vector<2x8xf32>
        %247 = vector.insert %246, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %249 = vector.extract %arg8[%244, %248] : f32 from vector<2x8xf32>
        %250 = vector.insert %249, %247 [0, 1] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %252 = vector.extract %arg8[%244, %251] : f32 from vector<2x8xf32>
        %253 = vector.insert %252, %250 [0, 2] : f32 into vector<1x4xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %255 = vector.extract %arg8[%244, %254] : f32 from vector<2x8xf32>
        %256 = vector.insert %255, %253 [0, 3] : f32 into vector<1x4xf32>
        %257 = arith.mulf %256, %230 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %258 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %260 = vector.extract %257[0, 0] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %arg14 [%258, %259] : f32 into vector<2x8xf32>
        %262 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %263 = vector.extract %257[0, 1] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %261 [%258, %262] : f32 into vector<2x8xf32>
        %265 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %266 = vector.extract %257[0, 2] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %264 [%258, %265] : f32 into vector<2x8xf32>
        %268 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %269 = vector.extract %257[0, 3] : f32 from vector<1x4xf32>
        %270 = vector.insert %269, %267 [%258, %268] : f32 into vector<2x8xf32>
        %271 = arith.addf %257, %212 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %272 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %273 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %274 = vector.extract %271[0, 0] : f32 from vector<1x4xf32>
        %275 = vector.insert %274, %arg15 [%272, %273] : f32 into vector<2x8xf32>
        %276 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %277 = vector.extract %271[0, 1] : f32 from vector<1x4xf32>
        %278 = vector.insert %277, %275 [%272, %276] : f32 into vector<2x8xf32>
        %279 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %280 = vector.extract %271[0, 2] : f32 from vector<1x4xf32>
        %281 = vector.insert %280, %278 [%272, %279] : f32 into vector<2x8xf32>
        %282 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %283 = vector.extract %271[0, 3] : f32 from vector<1x4xf32>
        %284 = vector.insert %283, %281 [%272, %282] : f32 into vector<2x8xf32>
        affine.yield %225, %243, %270, %284 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %202 : !frisk.async_task
      %204 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %212 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_48]
        %213 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_48]
        %214 = frisk.read_matrix_from_shm %alloc_50[%212, %213] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %215 = vector.bitcast %214 : vector<4xi32> to vector<8xf16>
        %216 = vector.extract %215[0] : f16 from vector<8xf16>
        %217 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_48]
        %218 = vector.extract %215[%217] : f16 from vector<8xf16>
        %219 = vector.extract %215[1] : f16 from vector<8xf16>
        %220 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_48]
        %221 = vector.extract %215[%220] : f16 from vector<8xf16>
        %222 = vector.extract %215[2] : f16 from vector<8xf16>
        %223 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_48]
        %224 = vector.extract %215[%223] : f16 from vector<8xf16>
        %225 = vector.extract %215[3] : f16 from vector<8xf16>
        %226 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_48]
        %227 = vector.extract %215[%226] : f16 from vector<8xf16>
        %228 = vector.from_elements %216, %218, %219, %221, %222, %224, %225, %227 : vector<4x2xf16>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %230 = affine.apply affine_map<() -> (0)>()
        %231 = vector.extract %228[0, 0] : f16 from vector<4x2xf16>
        %232 = vector.insert %231, %arg12 [%229, %230] : f16 into vector<32x2xf16>
        %233 = affine.apply affine_map<() -> (1)>()
        %234 = vector.extract %228[0, 1] : f16 from vector<4x2xf16>
        %235 = vector.insert %234, %232 [%229, %233] : f16 into vector<32x2xf16>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %237 = vector.extract %228[1, 0] : f16 from vector<4x2xf16>
        %238 = vector.insert %237, %235 [%236, %230] : f16 into vector<32x2xf16>
        %239 = affine.apply affine_map<() -> (1)>()
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %241 = vector.extract %228[1, 1] : f16 from vector<4x2xf16>
        %242 = vector.insert %241, %238 [%240, %239] : f16 into vector<32x2xf16>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %244 = vector.extract %228[2, 0] : f16 from vector<4x2xf16>
        %245 = vector.insert %244, %242 [%243, %230] : f16 into vector<32x2xf16>
        %246 = affine.apply affine_map<() -> (1)>()
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %248 = vector.extract %228[2, 1] : f16 from vector<4x2xf16>
        %249 = vector.insert %248, %245 [%247, %246] : f16 into vector<32x2xf16>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %251 = vector.extract %228[3, 0] : f16 from vector<4x2xf16>
        %252 = vector.insert %251, %249 [%250, %230] : f16 into vector<32x2xf16>
        %253 = affine.apply affine_map<() -> (1)>()
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %255 = vector.extract %228[3, 1] : f16 from vector<4x2xf16>
        %256 = vector.insert %255, %252 [%254, %253] : f16 into vector<32x2xf16>
        affine.yield %256 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %205 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      %alloc_51 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %206 = frisk.async_copy_gm_to_shm %205 to %alloc_51 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> !frisk.async_task
      %207:4 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %212 = affine.for %arg16 = 0 to 8 iter_args(%arg17 = %cst_2) -> (vector<1x4xf32>) {
          %285 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %286 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %cst_55 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %287 = vector.extract %195[%285, %286] : f16 from vector<2x32xf16>
          %288 = vector.insert %287, %cst_55 [0, 0] : f16 into vector<1x4xf16>
          %289 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %290 = vector.extract %195[%285, %289] : f16 from vector<2x32xf16>
          %291 = vector.insert %290, %288 [0, 1] : f16 into vector<1x4xf16>
          %292 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %293 = vector.extract %195[%285, %292] : f16 from vector<2x32xf16>
          %294 = vector.insert %293, %291 [0, 2] : f16 into vector<1x4xf16>
          %295 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %296 = vector.extract %195[%285, %295] : f16 from vector<2x32xf16>
          %297 = vector.insert %296, %294 [0, 3] : f16 into vector<1x4xf16>
          %298 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg16)
          %299 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %300 = vector.extract %204[%298, %299] : f16 from vector<32x2xf16>
          %301 = vector.insert %300, %cst_56 [0, 0] : f16 into vector<4x1xf16>
          %302 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg16)
          %303 = vector.extract %204[%302, %299] : f16 from vector<32x2xf16>
          %304 = vector.insert %303, %301 [1, 0] : f16 into vector<4x1xf16>
          %305 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg16)
          %306 = vector.extract %204[%305, %299] : f16 from vector<32x2xf16>
          %307 = vector.insert %306, %304 [2, 0] : f16 into vector<4x1xf16>
          %308 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg16)
          %309 = vector.extract %204[%308, %299] : f16 from vector<32x2xf16>
          %310 = vector.insert %309, %307 [3, 0] : f16 into vector<4x1xf16>
          %311 = frisk.warp_mma_rr %297, %310, %arg17 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %311 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
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
        %227 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %228 = vector.extract %191#1[%226, %227] : f32 from vector<2x1xf32>
        %229 = vector.insert %228, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %230 = affine.for %arg16 = 0 to 1 iter_args(%arg17 = %cst_53) -> (vector<1x4xf32>) {
          %c0_55 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %285 = vector.extract %229[%c0_58, %c0_59] : f32 from vector<1x1xf32>
          %286 = vector.insert %285, %cst_57 [0, 0] : f32 into vector<1x4xf32>
          %c0_60 = arith.constant 0 : index
          %c0_61 = arith.constant 0 : index
          %287 = vector.extract %229[%c0_60, %c0_61] : f32 from vector<1x1xf32>
          %288 = vector.insert %287, %286 [0, 1] : f32 into vector<1x4xf32>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %289 = vector.extract %229[%c0_62, %c0_63] : f32 from vector<1x1xf32>
          %290 = vector.insert %289, %288 [0, 2] : f32 into vector<1x4xf32>
          %c0_64 = arith.constant 0 : index
          %c0_65 = arith.constant 0 : index
          %291 = vector.extract %229[%c0_64, %c0_65] : f32 from vector<1x1xf32>
          %292 = vector.insert %291, %290 [0, 3] : f32 into vector<1x4xf32>
          %293 = vector.extract %292[0, 0] : f32 from vector<1x4xf32>
          %294 = vector.insert %293, %arg17 [%c0_56, %c0_55] : f32 into vector<1x4xf32>
          %295 = affine.apply affine_map<() -> (1)>()
          %296 = vector.extract %292[0, 1] : f32 from vector<1x4xf32>
          %297 = vector.insert %296, %294 [%c0_56, %295] : f32 into vector<1x4xf32>
          %298 = affine.apply affine_map<() -> (2)>()
          %299 = vector.extract %292[0, 2] : f32 from vector<1x4xf32>
          %300 = vector.insert %299, %297 [%c0_56, %298] : f32 into vector<1x4xf32>
          %301 = affine.apply affine_map<() -> (3)>()
          %302 = vector.extract %292[0, 3] : f32 from vector<1x4xf32>
          %303 = vector.insert %302, %300 [%c0_56, %301] : f32 into vector<1x4xf32>
          affine.yield %303 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
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
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %246 = vector.extract %arg9[%244, %245] : f32 from vector<2x8xf32>
        %247 = vector.insert %246, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %249 = vector.extract %arg9[%244, %248] : f32 from vector<2x8xf32>
        %250 = vector.insert %249, %247 [0, 1] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %252 = vector.extract %arg9[%244, %251] : f32 from vector<2x8xf32>
        %253 = vector.insert %252, %250 [0, 2] : f32 into vector<1x4xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %255 = vector.extract %arg9[%244, %254] : f32 from vector<2x8xf32>
        %256 = vector.insert %255, %253 [0, 3] : f32 into vector<1x4xf32>
        %257 = arith.mulf %256, %230 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %258 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %260 = vector.extract %257[0, 0] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %arg14 [%258, %259] : f32 into vector<2x8xf32>
        %262 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %263 = vector.extract %257[0, 1] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %261 [%258, %262] : f32 into vector<2x8xf32>
        %265 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %266 = vector.extract %257[0, 2] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %264 [%258, %265] : f32 into vector<2x8xf32>
        %268 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %269 = vector.extract %257[0, 3] : f32 from vector<1x4xf32>
        %270 = vector.insert %269, %267 [%258, %268] : f32 into vector<2x8xf32>
        %271 = arith.addf %257, %212 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %272 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %273 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %274 = vector.extract %271[0, 0] : f32 from vector<1x4xf32>
        %275 = vector.insert %274, %arg15 [%272, %273] : f32 into vector<2x8xf32>
        %276 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %277 = vector.extract %271[0, 1] : f32 from vector<1x4xf32>
        %278 = vector.insert %277, %275 [%272, %276] : f32 into vector<2x8xf32>
        %279 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %280 = vector.extract %271[0, 2] : f32 from vector<1x4xf32>
        %281 = vector.insert %280, %278 [%272, %279] : f32 into vector<2x8xf32>
        %282 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %283 = vector.extract %271[0, 3] : f32 from vector<1x4xf32>
        %284 = vector.insert %283, %281 [%272, %282] : f32 into vector<2x8xf32>
        affine.yield %225, %243, %270, %284 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      frisk.wait_buffer_ready %206 : !frisk.async_task
      %208 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_0) -> (vector<32x2xf16>) {
        %212 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + ((d0 * 16 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (d0 floordiv 4) * -16 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12)>(%arg11)[%thread_id_x_48]
        %213 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_48]
        %214 = frisk.read_matrix_from_shm %alloc_51[%212, %213] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
        %215 = vector.bitcast %214 : vector<4xi32> to vector<8xf16>
        %216 = vector.extract %215[0] : f16 from vector<8xf16>
        %217 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x_48]
        %218 = vector.extract %215[%217] : f16 from vector<8xf16>
        %219 = vector.extract %215[1] : f16 from vector<8xf16>
        %220 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x_48]
        %221 = vector.extract %215[%220] : f16 from vector<8xf16>
        %222 = vector.extract %215[2] : f16 from vector<8xf16>
        %223 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x_48]
        %224 = vector.extract %215[%223] : f16 from vector<8xf16>
        %225 = vector.extract %215[3] : f16 from vector<8xf16>
        %226 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x_48]
        %227 = vector.extract %215[%226] : f16 from vector<8xf16>
        %228 = vector.from_elements %216, %218, %219, %221, %222, %224, %225, %227 : vector<4x2xf16>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
        %230 = affine.apply affine_map<() -> (0)>()
        %231 = vector.extract %228[0, 0] : f16 from vector<4x2xf16>
        %232 = vector.insert %231, %arg12 [%229, %230] : f16 into vector<32x2xf16>
        %233 = affine.apply affine_map<() -> (1)>()
        %234 = vector.extract %228[0, 1] : f16 from vector<4x2xf16>
        %235 = vector.insert %234, %232 [%229, %233] : f16 into vector<32x2xf16>
        %236 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %237 = vector.extract %228[1, 0] : f16 from vector<4x2xf16>
        %238 = vector.insert %237, %235 [%236, %230] : f16 into vector<32x2xf16>
        %239 = affine.apply affine_map<() -> (1)>()
        %240 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
        %241 = vector.extract %228[1, 1] : f16 from vector<4x2xf16>
        %242 = vector.insert %241, %238 [%240, %239] : f16 into vector<32x2xf16>
        %243 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %244 = vector.extract %228[2, 0] : f16 from vector<4x2xf16>
        %245 = vector.insert %244, %242 [%243, %230] : f16 into vector<32x2xf16>
        %246 = affine.apply affine_map<() -> (1)>()
        %247 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
        %248 = vector.extract %228[2, 1] : f16 from vector<4x2xf16>
        %249 = vector.insert %248, %245 [%247, %246] : f16 into vector<32x2xf16>
        %250 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %251 = vector.extract %228[3, 0] : f16 from vector<4x2xf16>
        %252 = vector.insert %251, %249 [%250, %230] : f16 into vector<32x2xf16>
        %253 = affine.apply affine_map<() -> (1)>()
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
        %255 = vector.extract %228[3, 1] : f16 from vector<4x2xf16>
        %256 = vector.insert %255, %252 [%254, %253] : f16 into vector<32x2xf16>
        affine.yield %256 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %209:5 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %212 = affine.for %arg17 = 0 to 8 iter_args(%arg18 = %cst_2) -> (vector<1x4xf32>) {
          %311 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
          %312 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %313 = vector.extract %195[%311, %312] : f16 from vector<2x32xf16>
          %314 = vector.insert %313, %cst_56 [0, 0] : f16 into vector<1x4xf16>
          %315 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %316 = vector.extract %195[%311, %315] : f16 from vector<2x32xf16>
          %317 = vector.insert %316, %314 [0, 1] : f16 into vector<1x4xf16>
          %318 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %319 = vector.extract %195[%311, %318] : f16 from vector<2x32xf16>
          %320 = vector.insert %319, %317 [0, 2] : f16 into vector<1x4xf16>
          %321 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %322 = vector.extract %195[%311, %321] : f16 from vector<2x32xf16>
          %323 = vector.insert %322, %320 [0, 3] : f16 into vector<1x4xf16>
          %324 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg17)
          %325 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
          %cst_57 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
          %326 = vector.extract %208[%324, %325] : f16 from vector<32x2xf16>
          %327 = vector.insert %326, %cst_57 [0, 0] : f16 into vector<4x1xf16>
          %328 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg17)
          %329 = vector.extract %208[%328, %325] : f16 from vector<32x2xf16>
          %330 = vector.insert %329, %327 [1, 0] : f16 into vector<4x1xf16>
          %331 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg17)
          %332 = vector.extract %208[%331, %325] : f16 from vector<32x2xf16>
          %333 = vector.insert %332, %330 [2, 0] : f16 into vector<4x1xf16>
          %334 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg17)
          %335 = vector.extract %208[%334, %325] : f16 from vector<32x2xf16>
          %336 = vector.insert %335, %333 [3, 0] : f16 into vector<4x1xf16>
          %337 = frisk.warp_mma_rr %323, %336, %arg18 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
          affine.yield %337 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
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
        %227 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %228 = vector.extract %191#1[%226, %227] : f32 from vector<2x1xf32>
        %229 = vector.insert %228, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %230 = affine.for %arg17 = 0 to 1 iter_args(%arg18 = %cst_53) -> (vector<1x4xf32>) {
          %c0_56 = arith.constant 0 : index
          %c0_57 = arith.constant 0 : index
          %cst_58 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_59 = arith.constant 0 : index
          %c0_60 = arith.constant 0 : index
          %311 = vector.extract %229[%c0_59, %c0_60] : f32 from vector<1x1xf32>
          %312 = vector.insert %311, %cst_58 [0, 0] : f32 into vector<1x4xf32>
          %c0_61 = arith.constant 0 : index
          %c0_62 = arith.constant 0 : index
          %313 = vector.extract %229[%c0_61, %c0_62] : f32 from vector<1x1xf32>
          %314 = vector.insert %313, %312 [0, 1] : f32 into vector<1x4xf32>
          %c0_63 = arith.constant 0 : index
          %c0_64 = arith.constant 0 : index
          %315 = vector.extract %229[%c0_63, %c0_64] : f32 from vector<1x1xf32>
          %316 = vector.insert %315, %314 [0, 2] : f32 into vector<1x4xf32>
          %c0_65 = arith.constant 0 : index
          %c0_66 = arith.constant 0 : index
          %317 = vector.extract %229[%c0_65, %c0_66] : f32 from vector<1x1xf32>
          %318 = vector.insert %317, %316 [0, 3] : f32 into vector<1x4xf32>
          %319 = vector.extract %318[0, 0] : f32 from vector<1x4xf32>
          %320 = vector.insert %319, %arg18 [%c0_57, %c0_56] : f32 into vector<1x4xf32>
          %321 = affine.apply affine_map<() -> (1)>()
          %322 = vector.extract %318[0, 1] : f32 from vector<1x4xf32>
          %323 = vector.insert %322, %320 [%c0_57, %321] : f32 into vector<1x4xf32>
          %324 = affine.apply affine_map<() -> (2)>()
          %325 = vector.extract %318[0, 2] : f32 from vector<1x4xf32>
          %326 = vector.insert %325, %323 [%c0_57, %324] : f32 into vector<1x4xf32>
          %327 = affine.apply affine_map<() -> (3)>()
          %328 = vector.extract %318[0, 3] : f32 from vector<1x4xf32>
          %329 = vector.insert %328, %326 [%c0_57, %327] : f32 into vector<1x4xf32>
          affine.yield %329 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
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
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %246 = vector.extract %arg10[%244, %245] : f32 from vector<2x8xf32>
        %247 = vector.insert %246, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %249 = vector.extract %arg10[%244, %248] : f32 from vector<2x8xf32>
        %250 = vector.insert %249, %247 [0, 1] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %252 = vector.extract %arg10[%244, %251] : f32 from vector<2x8xf32>
        %253 = vector.insert %252, %250 [0, 2] : f32 into vector<1x4xf32>
        %254 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %255 = vector.extract %arg10[%244, %254] : f32 from vector<2x8xf32>
        %256 = vector.insert %255, %253 [0, 3] : f32 into vector<1x4xf32>
        %257 = arith.mulf %256, %230 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
        %258 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %259 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %260 = vector.extract %257[0, 0] : f32 from vector<1x4xf32>
        %261 = vector.insert %260, %arg14 [%258, %259] : f32 into vector<2x8xf32>
        %262 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %263 = vector.extract %257[0, 1] : f32 from vector<1x4xf32>
        %264 = vector.insert %263, %261 [%258, %262] : f32 into vector<2x8xf32>
        %265 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %266 = vector.extract %257[0, 2] : f32 from vector<1x4xf32>
        %267 = vector.insert %266, %264 [%258, %265] : f32 into vector<2x8xf32>
        %268 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %269 = vector.extract %257[0, 3] : f32 from vector<1x4xf32>
        %270 = vector.insert %269, %267 [%258, %268] : f32 into vector<2x8xf32>
        %271 = arith.addf %257, %212 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
        %272 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %273 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %274 = vector.extract %271[0, 0] : f32 from vector<1x4xf32>
        %275 = vector.insert %274, %arg15 [%272, %273] : f32 into vector<2x8xf32>
        %276 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %277 = vector.extract %271[0, 1] : f32 from vector<1x4xf32>
        %278 = vector.insert %277, %275 [%272, %276] : f32 into vector<2x8xf32>
        %279 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %280 = vector.extract %271[0, 2] : f32 from vector<1x4xf32>
        %281 = vector.insert %280, %278 [%272, %279] : f32 into vector<2x8xf32>
        %282 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %283 = vector.extract %271[0, 3] : f32 from vector<1x4xf32>
        %284 = vector.insert %283, %281 [%272, %282] : f32 into vector<2x8xf32>
        %285 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %286 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_55 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %287 = vector.extract %199#3[%285, %286] : f32 from vector<2x8xf32>
        %288 = vector.insert %287, %cst_55 [0, 0] : f32 into vector<1x4xf32>
        %289 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %290 = vector.extract %199#3[%285, %289] : f32 from vector<2x8xf32>
        %291 = vector.insert %290, %288 [0, 1] : f32 into vector<1x4xf32>
        %292 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %293 = vector.extract %199#3[%285, %292] : f32 from vector<2x8xf32>
        %294 = vector.insert %293, %291 [0, 2] : f32 into vector<1x4xf32>
        %295 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %296 = vector.extract %199#3[%285, %295] : f32 from vector<2x8xf32>
        %297 = vector.insert %296, %294 [0, 3] : f32 into vector<1x4xf32>
        %298 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %299 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %300 = vector.extract %297[0, 0] : f32 from vector<1x4xf32>
        %301 = vector.insert %300, %arg16 [%298, %299] : f32 into vector<2x8xf32>
        %302 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %303 = vector.extract %297[0, 1] : f32 from vector<1x4xf32>
        %304 = vector.insert %303, %301 [%298, %302] : f32 into vector<2x8xf32>
        %305 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %306 = vector.extract %297[0, 2] : f32 from vector<1x4xf32>
        %307 = vector.insert %306, %304 [%298, %305] : f32 into vector<2x8xf32>
        %308 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %309 = vector.extract %297[0, 3] : f32 from vector<1x4xf32>
        %310 = vector.insert %309, %307 [%298, %308] : f32 into vector<2x8xf32>
        affine.yield %225, %243, %270, %284, %310 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, tiled = true}
      %210:2 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9, %arg13 = %cst_9) -> (vector<2x1xf32>, vector<2x1xf32>) {
        %212 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %213 = affine.apply affine_map<() -> (0)>()
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %214 = vector.extract %194[%212, %213] : f32 from vector<2x1xf32>
        %215 = vector.insert %214, %cst_52 [0, 0] : f32 into vector<1x1xf32>
        %216 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %217 = affine.apply affine_map<() -> (0)>()
        %218 = vector.extract %215[0, 0] : f32 from vector<1x1xf32>
        %219 = vector.insert %218, %arg12 [%216, %217] : f32 into vector<2x1xf32>
        %220 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %221 = affine.apply affine_map<() -> (0)>()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %222 = vector.extract %189[%220, %221] : f32 from vector<2x1xf32>
        %223 = vector.insert %222, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %224 = affine.apply affine_map<(d0) -> (d0)>(%arg11)
        %225 = affine.apply affine_map<() -> (0)>()
        %226 = vector.extract %223[0, 0] : f32 from vector<1x1xf32>
        %227 = vector.insert %226, %arg13 [%224, %225] : f32 into vector<2x1xf32>
        affine.yield %219, %227 : vector<2x1xf32>, vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.fused, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %211:3 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst, %arg13 = %cst, %arg14 = %cst) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
        %212 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_52 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %214 = vector.extract %203#3[%212, %213] : f32 from vector<2x8xf32>
        %215 = vector.insert %214, %cst_52 [0, 0] : f32 into vector<1x4xf32>
        %216 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %217 = vector.extract %203#3[%212, %216] : f32 from vector<2x8xf32>
        %218 = vector.insert %217, %215 [0, 1] : f32 into vector<1x4xf32>
        %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %220 = vector.extract %203#3[%212, %219] : f32 from vector<2x8xf32>
        %221 = vector.insert %220, %218 [0, 2] : f32 into vector<1x4xf32>
        %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %223 = vector.extract %203#3[%212, %222] : f32 from vector<2x8xf32>
        %224 = vector.insert %223, %221 [0, 3] : f32 into vector<1x4xf32>
        %225 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %226 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %227 = vector.extract %224[0, 0] : f32 from vector<1x4xf32>
        %228 = vector.insert %227, %arg12 [%225, %226] : f32 into vector<2x8xf32>
        %229 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %230 = vector.extract %224[0, 1] : f32 from vector<1x4xf32>
        %231 = vector.insert %230, %228 [%225, %229] : f32 into vector<2x8xf32>
        %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %233 = vector.extract %224[0, 2] : f32 from vector<1x4xf32>
        %234 = vector.insert %233, %231 [%225, %232] : f32 into vector<2x8xf32>
        %235 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %236 = vector.extract %224[0, 3] : f32 from vector<1x4xf32>
        %237 = vector.insert %236, %234 [%225, %235] : f32 into vector<2x8xf32>
        %238 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %239 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %240 = vector.extract %207#3[%238, %239] : f32 from vector<2x8xf32>
        %241 = vector.insert %240, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %242 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %243 = vector.extract %207#3[%238, %242] : f32 from vector<2x8xf32>
        %244 = vector.insert %243, %241 [0, 1] : f32 into vector<1x4xf32>
        %245 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %246 = vector.extract %207#3[%238, %245] : f32 from vector<2x8xf32>
        %247 = vector.insert %246, %244 [0, 2] : f32 into vector<1x4xf32>
        %248 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %249 = vector.extract %207#3[%238, %248] : f32 from vector<2x8xf32>
        %250 = vector.insert %249, %247 [0, 3] : f32 into vector<1x4xf32>
        %251 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %252 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %253 = vector.extract %250[0, 0] : f32 from vector<1x4xf32>
        %254 = vector.insert %253, %arg13 [%251, %252] : f32 into vector<2x8xf32>
        %255 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %256 = vector.extract %250[0, 1] : f32 from vector<1x4xf32>
        %257 = vector.insert %256, %254 [%251, %255] : f32 into vector<2x8xf32>
        %258 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %259 = vector.extract %250[0, 2] : f32 from vector<1x4xf32>
        %260 = vector.insert %259, %257 [%251, %258] : f32 into vector<2x8xf32>
        %261 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %262 = vector.extract %250[0, 3] : f32 from vector<1x4xf32>
        %263 = vector.insert %262, %260 [%251, %261] : f32 into vector<2x8xf32>
        %264 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %265 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %266 = vector.extract %209#3[%264, %265] : f32 from vector<2x8xf32>
        %267 = vector.insert %266, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %268 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %269 = vector.extract %209#3[%264, %268] : f32 from vector<2x8xf32>
        %270 = vector.insert %269, %267 [0, 1] : f32 into vector<1x4xf32>
        %271 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %272 = vector.extract %209#3[%264, %271] : f32 from vector<2x8xf32>
        %273 = vector.insert %272, %270 [0, 2] : f32 into vector<1x4xf32>
        %274 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %275 = vector.extract %209#3[%264, %274] : f32 from vector<2x8xf32>
        %276 = vector.insert %275, %273 [0, 3] : f32 into vector<1x4xf32>
        %277 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg11)
        %278 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %279 = vector.extract %276[0, 0] : f32 from vector<1x4xf32>
        %280 = vector.insert %279, %arg14 [%277, %278] : f32 into vector<2x8xf32>
        %281 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %282 = vector.extract %276[0, 1] : f32 from vector<1x4xf32>
        %283 = vector.insert %282, %280 [%277, %281] : f32 into vector<2x8xf32>
        %284 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %285 = vector.extract %276[0, 2] : f32 from vector<1x4xf32>
        %286 = vector.insert %285, %283 [%277, %284] : f32 into vector<2x8xf32>
        %287 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %288 = vector.extract %276[0, 3] : f32 from vector<1x4xf32>
        %289 = vector.insert %288, %286 [%277, %287] : f32 into vector<2x8xf32>
        affine.yield %237, %263, %289 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      affine.yield %209#4, %210#0, %210#1, %211#0, %211#1, %211#2 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %thread_id_x = gpu.thread_id x
    %138 = arith.index_cast %thread_id_x : index to i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %139 = arith.andi %138, %c16_i32 : i32
    %140 = arith.cmpi ne, %139, %c0_i32 : i32
    %c32_i32 = arith.constant 32 : i32
    %141 = arith.andi %138, %c32_i32 : i32
    %142 = arith.cmpi ne, %141, %c0_i32 : i32
    %cst_19 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %143 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_19) -> (vector<2x8xf32>) {
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %173 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %174 = vector.extract %137#5[%173, %172] : f32 from vector<2x8xf32>
      %175 = vector.insert %174, %cst_39 [0, 0] : f32 into vector<1x4xf32>
      %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %177 = vector.extract %137#5[%173, %176] : f32 from vector<2x8xf32>
      %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %180 = vector.extract %137#5[%173, %179] : f32 from vector<2x8xf32>
      %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
      %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %183 = vector.extract %137#5[%173, %182] : f32 from vector<2x8xf32>
      %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
      %185 = vector.extract %184[0, 0] : f32 from vector<1x4xf32>
      %186 = vector.extract %184[0, 1] : f32 from vector<1x4xf32>
      %187 = vector.extract %184[0, 2] : f32 from vector<1x4xf32>
      %188 = vector.extract %184[0, 3] : f32 from vector<1x4xf32>
      %189 = arith.select %140, %185, %186 : f32
      %shuffleResult, %valid = gpu.shuffle xor %189, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %190 = arith.select %140, %shuffleResult, %185 : f32
      %191 = arith.select %140, %186, %shuffleResult : f32
      %192 = arith.select %140, %187, %188 : f32
      %shuffleResult_40, %valid_41 = gpu.shuffle xor %192, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %193 = arith.select %140, %shuffleResult_40, %187 : f32
      %194 = arith.select %140, %188, %shuffleResult_40 : f32
      %195 = arith.select %142, %190, %193 : f32
      %shuffleResult_42, %valid_43 = gpu.shuffle xor %195, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %196 = arith.select %142, %shuffleResult_42, %190 : f32
      %197 = arith.select %142, %193, %shuffleResult_42 : f32
      %198 = arith.select %142, %191, %194 : f32
      %shuffleResult_44, %valid_45 = gpu.shuffle xor %198, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
      %199 = arith.select %142, %shuffleResult_44, %191 : f32
      %200 = arith.select %142, %194, %shuffleResult_44 : f32
      %201 = vector.insert %196, %184 [0, 0] : f32 into vector<1x4xf32>
      %202 = vector.insert %199, %201 [0, 1] : f32 into vector<1x4xf32>
      %203 = vector.insert %197, %202 [0, 2] : f32 into vector<1x4xf32>
      %204 = vector.insert %200, %203 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
      %205 = vector.extract %204[0, 0] : f32 from vector<1x4xf32>
      %206 = vector.insert %205, %arg5 [%173, %172] : f32 into vector<2x8xf32>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %208 = vector.extract %204[0, 1] : f32 from vector<1x4xf32>
      %209 = vector.insert %208, %206 [%173, %207] : f32 into vector<2x8xf32>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %211 = vector.extract %204[0, 2] : f32 from vector<1x4xf32>
      %212 = vector.insert %211, %209 [%173, %210] : f32 into vector<2x8xf32>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %214 = vector.extract %204[0, 3] : f32 from vector<1x4xf32>
      %215 = vector.insert %214, %212 [%173, %213] : f32 into vector<2x8xf32>
      affine.yield %215 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_20 = gpu.thread_id x
    %144 = arith.index_cast %thread_id_x_20 : index to i32
    %c0_i32_21 = arith.constant 0 : i32
    %c64_i32_22 = arith.constant 64 : i32
    %c16_i32_23 = arith.constant 16 : i32
    %145 = arith.andi %144, %c16_i32_23 : i32
    %146 = arith.cmpi ne, %145, %c0_i32_21 : i32
    %c32_i32_24 = arith.constant 32 : i32
    %147 = arith.andi %144, %c32_i32_24 : i32
    %148 = arith.cmpi ne, %147, %c0_i32_21 : i32
    %cst_25 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %149 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_25) -> (vector<2x8xf32>) {
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %173 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %174 = vector.extract %137#4[%173, %172] : f32 from vector<2x8xf32>
      %175 = vector.insert %174, %cst_39 [0, 0] : f32 into vector<1x4xf32>
      %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %177 = vector.extract %137#4[%173, %176] : f32 from vector<2x8xf32>
      %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %180 = vector.extract %137#4[%173, %179] : f32 from vector<2x8xf32>
      %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
      %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %183 = vector.extract %137#4[%173, %182] : f32 from vector<2x8xf32>
      %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
      %185 = vector.extract %184[0, 0] : f32 from vector<1x4xf32>
      %186 = vector.extract %184[0, 1] : f32 from vector<1x4xf32>
      %187 = vector.extract %184[0, 2] : f32 from vector<1x4xf32>
      %188 = vector.extract %184[0, 3] : f32 from vector<1x4xf32>
      %189 = arith.select %146, %185, %186 : f32
      %shuffleResult, %valid = gpu.shuffle xor %189, %c16_i32_23, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
      %190 = arith.select %146, %shuffleResult, %185 : f32
      %191 = arith.select %146, %186, %shuffleResult : f32
      %192 = arith.select %146, %187, %188 : f32
      %shuffleResult_40, %valid_41 = gpu.shuffle xor %192, %c16_i32_23, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
      %193 = arith.select %146, %shuffleResult_40, %187 : f32
      %194 = arith.select %146, %188, %shuffleResult_40 : f32
      %195 = arith.select %148, %190, %193 : f32
      %shuffleResult_42, %valid_43 = gpu.shuffle xor %195, %c32_i32_24, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
      %196 = arith.select %148, %shuffleResult_42, %190 : f32
      %197 = arith.select %148, %193, %shuffleResult_42 : f32
      %198 = arith.select %148, %191, %194 : f32
      %shuffleResult_44, %valid_45 = gpu.shuffle xor %198, %c32_i32_24, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
      %199 = arith.select %148, %shuffleResult_44, %191 : f32
      %200 = arith.select %148, %194, %shuffleResult_44 : f32
      %201 = vector.insert %196, %184 [0, 0] : f32 into vector<1x4xf32>
      %202 = vector.insert %199, %201 [0, 1] : f32 into vector<1x4xf32>
      %203 = vector.insert %197, %202 [0, 2] : f32 into vector<1x4xf32>
      %204 = vector.insert %200, %203 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
      %205 = vector.extract %204[0, 0] : f32 from vector<1x4xf32>
      %206 = vector.insert %205, %arg5 [%173, %172] : f32 into vector<2x8xf32>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %208 = vector.extract %204[0, 1] : f32 from vector<1x4xf32>
      %209 = vector.insert %208, %206 [%173, %207] : f32 into vector<2x8xf32>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %211 = vector.extract %204[0, 2] : f32 from vector<1x4xf32>
      %212 = vector.insert %211, %209 [%173, %210] : f32 into vector<2x8xf32>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %214 = vector.extract %204[0, 3] : f32 from vector<1x4xf32>
      %215 = vector.insert %214, %212 [%173, %213] : f32 into vector<2x8xf32>
      affine.yield %215 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_26 = gpu.thread_id x
    %150 = arith.index_cast %thread_id_x_26 : index to i32
    %c0_i32_27 = arith.constant 0 : i32
    %c64_i32_28 = arith.constant 64 : i32
    %c16_i32_29 = arith.constant 16 : i32
    %151 = arith.andi %150, %c16_i32_29 : i32
    %152 = arith.cmpi ne, %151, %c0_i32_27 : i32
    %c32_i32_30 = arith.constant 32 : i32
    %153 = arith.andi %150, %c32_i32_30 : i32
    %154 = arith.cmpi ne, %153, %c0_i32_27 : i32
    %cst_31 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %155 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_31) -> (vector<2x8xf32>) {
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %173 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %174 = vector.extract %137#3[%173, %172] : f32 from vector<2x8xf32>
      %175 = vector.insert %174, %cst_39 [0, 0] : f32 into vector<1x4xf32>
      %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %177 = vector.extract %137#3[%173, %176] : f32 from vector<2x8xf32>
      %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %180 = vector.extract %137#3[%173, %179] : f32 from vector<2x8xf32>
      %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
      %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %183 = vector.extract %137#3[%173, %182] : f32 from vector<2x8xf32>
      %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
      %185 = vector.extract %184[0, 0] : f32 from vector<1x4xf32>
      %186 = vector.extract %184[0, 1] : f32 from vector<1x4xf32>
      %187 = vector.extract %184[0, 2] : f32 from vector<1x4xf32>
      %188 = vector.extract %184[0, 3] : f32 from vector<1x4xf32>
      %189 = arith.select %152, %185, %186 : f32
      %shuffleResult, %valid = gpu.shuffle xor %189, %c16_i32_29, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
      %190 = arith.select %152, %shuffleResult, %185 : f32
      %191 = arith.select %152, %186, %shuffleResult : f32
      %192 = arith.select %152, %187, %188 : f32
      %shuffleResult_40, %valid_41 = gpu.shuffle xor %192, %c16_i32_29, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
      %193 = arith.select %152, %shuffleResult_40, %187 : f32
      %194 = arith.select %152, %188, %shuffleResult_40 : f32
      %195 = arith.select %154, %190, %193 : f32
      %shuffleResult_42, %valid_43 = gpu.shuffle xor %195, %c32_i32_30, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
      %196 = arith.select %154, %shuffleResult_42, %190 : f32
      %197 = arith.select %154, %193, %shuffleResult_42 : f32
      %198 = arith.select %154, %191, %194 : f32
      %shuffleResult_44, %valid_45 = gpu.shuffle xor %198, %c32_i32_30, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
      %199 = arith.select %154, %shuffleResult_44, %191 : f32
      %200 = arith.select %154, %194, %shuffleResult_44 : f32
      %201 = vector.insert %196, %184 [0, 0] : f32 into vector<1x4xf32>
      %202 = vector.insert %199, %201 [0, 1] : f32 into vector<1x4xf32>
      %203 = vector.insert %197, %202 [0, 2] : f32 into vector<1x4xf32>
      %204 = vector.insert %200, %203 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
      %205 = vector.extract %204[0, 0] : f32 from vector<1x4xf32>
      %206 = vector.insert %205, %arg5 [%173, %172] : f32 into vector<2x8xf32>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %208 = vector.extract %204[0, 1] : f32 from vector<1x4xf32>
      %209 = vector.insert %208, %206 [%173, %207] : f32 into vector<2x8xf32>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %211 = vector.extract %204[0, 2] : f32 from vector<1x4xf32>
      %212 = vector.insert %211, %209 [%173, %210] : f32 into vector<2x8xf32>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %214 = vector.extract %204[0, 3] : f32 from vector<1x4xf32>
      %215 = vector.insert %214, %212 [%173, %213] : f32 into vector<2x8xf32>
      affine.yield %215 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_32 = gpu.thread_id x
    %156 = arith.index_cast %thread_id_x_32 : index to i32
    %c0_i32_33 = arith.constant 0 : i32
    %c64_i32_34 = arith.constant 64 : i32
    %c16_i32_35 = arith.constant 16 : i32
    %157 = arith.andi %156, %c16_i32_35 : i32
    %158 = arith.cmpi ne, %157, %c0_i32_33 : i32
    %c32_i32_36 = arith.constant 32 : i32
    %159 = arith.andi %156, %c32_i32_36 : i32
    %160 = arith.cmpi ne, %159, %c0_i32_33 : i32
    %cst_37 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %161 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_37) -> (vector<2x8xf32>) {
      %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %173 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %174 = vector.extract %137#0[%173, %172] : f32 from vector<2x8xf32>
      %175 = vector.insert %174, %cst_39 [0, 0] : f32 into vector<1x4xf32>
      %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %177 = vector.extract %137#0[%173, %176] : f32 from vector<2x8xf32>
      %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
      %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %180 = vector.extract %137#0[%173, %179] : f32 from vector<2x8xf32>
      %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
      %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %183 = vector.extract %137#0[%173, %182] : f32 from vector<2x8xf32>
      %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
      %185 = vector.extract %184[0, 0] : f32 from vector<1x4xf32>
      %186 = vector.extract %184[0, 1] : f32 from vector<1x4xf32>
      %187 = vector.extract %184[0, 2] : f32 from vector<1x4xf32>
      %188 = vector.extract %184[0, 3] : f32 from vector<1x4xf32>
      %189 = arith.select %158, %185, %186 : f32
      %shuffleResult, %valid = gpu.shuffle xor %189, %c16_i32_35, %c64_i32_34 {frisk.layout_role = "wmmaC"} : f32
      %190 = arith.select %158, %shuffleResult, %185 : f32
      %191 = arith.select %158, %186, %shuffleResult : f32
      %192 = arith.select %158, %187, %188 : f32
      %shuffleResult_40, %valid_41 = gpu.shuffle xor %192, %c16_i32_35, %c64_i32_34 {frisk.layout_role = "wmmaC"} : f32
      %193 = arith.select %158, %shuffleResult_40, %187 : f32
      %194 = arith.select %158, %188, %shuffleResult_40 : f32
      %195 = arith.select %160, %190, %193 : f32
      %shuffleResult_42, %valid_43 = gpu.shuffle xor %195, %c32_i32_36, %c64_i32_34 {frisk.layout_role = "wmmaC"} : f32
      %196 = arith.select %160, %shuffleResult_42, %190 : f32
      %197 = arith.select %160, %193, %shuffleResult_42 : f32
      %198 = arith.select %160, %191, %194 : f32
      %shuffleResult_44, %valid_45 = gpu.shuffle xor %198, %c32_i32_36, %c64_i32_34 {frisk.layout_role = "wmmaC"} : f32
      %199 = arith.select %160, %shuffleResult_44, %191 : f32
      %200 = arith.select %160, %194, %shuffleResult_44 : f32
      %201 = vector.insert %196, %184 [0, 0] : f32 into vector<1x4xf32>
      %202 = vector.insert %199, %201 [0, 1] : f32 into vector<1x4xf32>
      %203 = vector.insert %197, %202 [0, 2] : f32 into vector<1x4xf32>
      %204 = vector.insert %200, %203 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
      %205 = vector.extract %204[0, 0] : f32 from vector<1x4xf32>
      %206 = vector.insert %205, %arg5 [%173, %172] : f32 into vector<2x8xf32>
      %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %208 = vector.extract %204[0, 1] : f32 from vector<1x4xf32>
      %209 = vector.insert %208, %206 [%173, %207] : f32 into vector<2x8xf32>
      %210 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %211 = vector.extract %204[0, 2] : f32 from vector<1x4xf32>
      %212 = vector.insert %211, %209 [%173, %210] : f32 into vector<2x8xf32>
      %213 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %214 = vector.extract %204[0, 3] : f32 from vector<1x4xf32>
      %215 = vector.insert %214, %212 [%173, %213] : f32 into vector<2x8xf32>
      affine.yield %215 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %162 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %163:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %173 = affine.apply affine_map<() -> (0)>()
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %174 = vector.extract %137#1[%172, %173] : f32 from vector<2x1xf32>
      %175 = vector.insert %174, %cst_39 [0, 0] : f32 into vector<1x1xf32>
      %cst_40 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %176 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_40) -> (vector<1x4xf32>) {
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %cst_44 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %244 = vector.extract %175[%c0_45, %c0_46] : f32 from vector<1x1xf32>
        %245 = vector.insert %244, %cst_44 [0, 0] : f32 into vector<1x4xf32>
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %246 = vector.extract %175[%c0_47, %c0_48] : f32 from vector<1x1xf32>
        %247 = vector.insert %246, %245 [0, 1] : f32 into vector<1x4xf32>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %248 = vector.extract %175[%c0_49, %c0_50] : f32 from vector<1x1xf32>
        %249 = vector.insert %248, %247 [0, 2] : f32 into vector<1x4xf32>
        %c0_51 = arith.constant 0 : index
        %c0_52 = arith.constant 0 : index
        %250 = vector.extract %175[%c0_51, %c0_52] : f32 from vector<1x1xf32>
        %251 = vector.insert %250, %249 [0, 3] : f32 into vector<1x4xf32>
        %252 = vector.extract %251[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg10 [%c0_43, %c0_42] : f32 into vector<1x4xf32>
        %254 = affine.apply affine_map<() -> (1)>()
        %255 = vector.extract %251[0, 1] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %253 [%c0_43, %254] : f32 into vector<1x4xf32>
        %257 = affine.apply affine_map<() -> (2)>()
        %258 = vector.extract %251[0, 2] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %256 [%c0_43, %257] : f32 into vector<1x4xf32>
        %260 = affine.apply affine_map<() -> (3)>()
        %261 = vector.extract %251[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%c0_43, %260] : f32 into vector<1x4xf32>
        affine.yield %262 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %177 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %179 = vector.extract %176[0, 0] : f32 from vector<1x4xf32>
      %180 = vector.insert %179, %arg5 [%177, %178] : f32 into vector<2x8xf32>
      %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %182 = vector.extract %176[0, 1] : f32 from vector<1x4xf32>
      %183 = vector.insert %182, %180 [%177, %181] : f32 into vector<2x8xf32>
      %184 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %185 = vector.extract %176[0, 2] : f32 from vector<1x4xf32>
      %186 = vector.insert %185, %183 [%177, %184] : f32 into vector<2x8xf32>
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %188 = vector.extract %176[0, 3] : f32 from vector<1x4xf32>
      %189 = vector.insert %188, %186 [%177, %187] : f32 into vector<2x8xf32>
      %190 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_41 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %192 = vector.extract %161[%190, %191] : f32 from vector<2x8xf32>
      %193 = vector.insert %192, %cst_41 [0, 0] : f32 into vector<1x4xf32>
      %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %195 = vector.extract %161[%190, %194] : f32 from vector<2x8xf32>
      %196 = vector.insert %195, %193 [0, 1] : f32 into vector<1x4xf32>
      %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %198 = vector.extract %161[%190, %197] : f32 from vector<2x8xf32>
      %199 = vector.insert %198, %196 [0, 2] : f32 into vector<1x4xf32>
      %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %201 = vector.extract %161[%190, %200] : f32 from vector<2x8xf32>
      %202 = vector.insert %201, %199 [0, 3] : f32 into vector<1x4xf32>
      %203 = arith.divf %202, %176 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %204 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %206 = vector.extract %203[0, 0] : f32 from vector<1x4xf32>
      %207 = vector.insert %206, %arg6 [%204, %205] : f32 into vector<2x8xf32>
      %208 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %209 = vector.extract %203[0, 1] : f32 from vector<1x4xf32>
      %210 = vector.insert %209, %207 [%204, %208] : f32 into vector<2x8xf32>
      %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %212 = vector.extract %203[0, 2] : f32 from vector<1x4xf32>
      %213 = vector.insert %212, %210 [%204, %211] : f32 into vector<2x8xf32>
      %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %215 = vector.extract %203[0, 3] : f32 from vector<1x4xf32>
      %216 = vector.insert %215, %213 [%204, %214] : f32 into vector<2x8xf32>
      %217 = arith.truncf %203 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %218 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %220 = vector.extract %217[0, 0] : f16 from vector<1x4xf16>
      %221 = vector.insert %220, %arg7 [%218, %219] : f16 into vector<2x8xf16>
      %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %223 = vector.extract %217[0, 1] : f16 from vector<1x4xf16>
      %224 = vector.insert %223, %221 [%218, %222] : f16 into vector<2x8xf16>
      %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %226 = vector.extract %217[0, 2] : f16 from vector<1x4xf16>
      %227 = vector.insert %226, %224 [%218, %225] : f16 into vector<2x8xf16>
      %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %229 = vector.extract %217[0, 3] : f16 from vector<1x4xf16>
      %230 = vector.insert %229, %227 [%218, %228] : f16 into vector<2x8xf16>
      %231 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %233 = vector.extract %217[0, 0] : f16 from vector<1x4xf16>
      %234 = vector.insert %233, %arg8 [%231, %232] : f16 into vector<2x8xf16>
      %235 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %236 = vector.extract %217[0, 1] : f16 from vector<1x4xf16>
      %237 = vector.insert %236, %234 [%231, %235] : f16 into vector<2x8xf16>
      %238 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %239 = vector.extract %217[0, 2] : f16 from vector<1x4xf16>
      %240 = vector.insert %239, %237 [%231, %238] : f16 into vector<2x8xf16>
      %241 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %242 = vector.extract %217[0, 3] : f16 from vector<1x4xf16>
      %243 = vector.insert %242, %240 [%231, %241] : f16 into vector<2x8xf16>
      affine.yield %189, %216, %230, %243 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_38 = gpu.thread_id x
    %164 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x_38]
    %165 = arith.cmpi ult, %164, %c4 : index
    scf.if %165 {
      affine.for %arg4 = 0 to 4 {
        %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %174 = vector.extract %163#3[%172, %173] : f16 from vector<2x8xf16>
        %175 = vector.insert %174, %cst_39 [0, 0] : f16 into vector<1x4xf16>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %177 = vector.extract %163#3[%172, %176] : f16 from vector<2x8xf16>
        %178 = vector.insert %177, %175 [0, 1] : f16 into vector<1x4xf16>
        %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %180 = vector.extract %163#3[%172, %179] : f16 from vector<2x8xf16>
        %181 = vector.insert %180, %178 [0, 2] : f16 into vector<1x4xf16>
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %183 = vector.extract %163#3[%172, %182] : f16 from vector<2x8xf16>
        %184 = vector.insert %183, %181 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_40 = gpu.thread_id x
        %185 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %187 = vector.extract %184[0, 0] : f16 from vector<1x4xf16>
        %c0_41 = arith.constant 0 : index
        %c0_42 = arith.constant 0 : index
        %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%186)
        %c0_43 = arith.constant 0 : index
        %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%186)
        %c0_44 = arith.constant 0 : index
        %190 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_42, %189)
        %191 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %188, %c0_41, %c0_43, %c0_44, %190)
        %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %188, %c0_41, %c0_43, %c0_44, %190)
        affine.store %187, %162[%191, %192] {tiled = true} : memref<128x32xf16, 1>
        %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %194 = vector.extract %184[0, 1] : f16 from vector<1x4xf16>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %195 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%193)
        %c0_47 = arith.constant 0 : index
        %196 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%193)
        %c0_48 = arith.constant 0 : index
        %197 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_46, %196)
        %198 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %195, %c0_45, %c0_47, %c0_48, %197)
        %199 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %195, %c0_45, %c0_47, %c0_48, %197)
        affine.store %194, %162[%198, %199] {tiled = true} : memref<128x32xf16, 1>
        %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %201 = vector.extract %184[0, 2] : f16 from vector<1x4xf16>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %202 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%200)
        %c0_51 = arith.constant 0 : index
        %203 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%200)
        %c0_52 = arith.constant 0 : index
        %204 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_50, %203)
        %205 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %202, %c0_49, %c0_51, %c0_52, %204)
        %206 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %202, %c0_49, %c0_51, %c0_52, %204)
        affine.store %201, %162[%205, %206] {tiled = true} : memref<128x32xf16, 1>
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %208 = vector.extract %184[0, 3] : f16 from vector<1x4xf16>
        %c0_53 = arith.constant 0 : index
        %c0_54 = arith.constant 0 : index
        %209 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%207)
        %c0_55 = arith.constant 0 : index
        %210 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%207)
        %c0_56 = arith.constant 0 : index
        %211 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_54, %210)
        %212 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %209, %c0_53, %c0_55, %c0_56, %211)
        %213 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %209, %c0_53, %c0_55, %c0_56, %211)
        affine.store %208, %162[%212, %213] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %166 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %167:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %173 = affine.apply affine_map<() -> (0)>()
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %174 = vector.extract %137#1[%172, %173] : f32 from vector<2x1xf32>
      %175 = vector.insert %174, %cst_39 [0, 0] : f32 into vector<1x1xf32>
      %cst_40 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %176 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_40) -> (vector<1x4xf32>) {
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %cst_44 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %244 = vector.extract %175[%c0_45, %c0_46] : f32 from vector<1x1xf32>
        %245 = vector.insert %244, %cst_44 [0, 0] : f32 into vector<1x4xf32>
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %246 = vector.extract %175[%c0_47, %c0_48] : f32 from vector<1x1xf32>
        %247 = vector.insert %246, %245 [0, 1] : f32 into vector<1x4xf32>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %248 = vector.extract %175[%c0_49, %c0_50] : f32 from vector<1x1xf32>
        %249 = vector.insert %248, %247 [0, 2] : f32 into vector<1x4xf32>
        %c0_51 = arith.constant 0 : index
        %c0_52 = arith.constant 0 : index
        %250 = vector.extract %175[%c0_51, %c0_52] : f32 from vector<1x1xf32>
        %251 = vector.insert %250, %249 [0, 3] : f32 into vector<1x4xf32>
        %252 = vector.extract %251[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg10 [%c0_43, %c0_42] : f32 into vector<1x4xf32>
        %254 = affine.apply affine_map<() -> (1)>()
        %255 = vector.extract %251[0, 1] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %253 [%c0_43, %254] : f32 into vector<1x4xf32>
        %257 = affine.apply affine_map<() -> (2)>()
        %258 = vector.extract %251[0, 2] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %256 [%c0_43, %257] : f32 into vector<1x4xf32>
        %260 = affine.apply affine_map<() -> (3)>()
        %261 = vector.extract %251[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%c0_43, %260] : f32 into vector<1x4xf32>
        affine.yield %262 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %177 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %179 = vector.extract %176[0, 0] : f32 from vector<1x4xf32>
      %180 = vector.insert %179, %arg5 [%177, %178] : f32 into vector<2x8xf32>
      %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %182 = vector.extract %176[0, 1] : f32 from vector<1x4xf32>
      %183 = vector.insert %182, %180 [%177, %181] : f32 into vector<2x8xf32>
      %184 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %185 = vector.extract %176[0, 2] : f32 from vector<1x4xf32>
      %186 = vector.insert %185, %183 [%177, %184] : f32 into vector<2x8xf32>
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %188 = vector.extract %176[0, 3] : f32 from vector<1x4xf32>
      %189 = vector.insert %188, %186 [%177, %187] : f32 into vector<2x8xf32>
      %190 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_41 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %192 = vector.extract %155[%190, %191] : f32 from vector<2x8xf32>
      %193 = vector.insert %192, %cst_41 [0, 0] : f32 into vector<1x4xf32>
      %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %195 = vector.extract %155[%190, %194] : f32 from vector<2x8xf32>
      %196 = vector.insert %195, %193 [0, 1] : f32 into vector<1x4xf32>
      %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %198 = vector.extract %155[%190, %197] : f32 from vector<2x8xf32>
      %199 = vector.insert %198, %196 [0, 2] : f32 into vector<1x4xf32>
      %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %201 = vector.extract %155[%190, %200] : f32 from vector<2x8xf32>
      %202 = vector.insert %201, %199 [0, 3] : f32 into vector<1x4xf32>
      %203 = arith.divf %202, %176 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %204 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %206 = vector.extract %203[0, 0] : f32 from vector<1x4xf32>
      %207 = vector.insert %206, %arg6 [%204, %205] : f32 into vector<2x8xf32>
      %208 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %209 = vector.extract %203[0, 1] : f32 from vector<1x4xf32>
      %210 = vector.insert %209, %207 [%204, %208] : f32 into vector<2x8xf32>
      %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %212 = vector.extract %203[0, 2] : f32 from vector<1x4xf32>
      %213 = vector.insert %212, %210 [%204, %211] : f32 into vector<2x8xf32>
      %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %215 = vector.extract %203[0, 3] : f32 from vector<1x4xf32>
      %216 = vector.insert %215, %213 [%204, %214] : f32 into vector<2x8xf32>
      %217 = arith.truncf %203 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %218 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %220 = vector.extract %217[0, 0] : f16 from vector<1x4xf16>
      %221 = vector.insert %220, %arg7 [%218, %219] : f16 into vector<2x8xf16>
      %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %223 = vector.extract %217[0, 1] : f16 from vector<1x4xf16>
      %224 = vector.insert %223, %221 [%218, %222] : f16 into vector<2x8xf16>
      %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %226 = vector.extract %217[0, 2] : f16 from vector<1x4xf16>
      %227 = vector.insert %226, %224 [%218, %225] : f16 into vector<2x8xf16>
      %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %229 = vector.extract %217[0, 3] : f16 from vector<1x4xf16>
      %230 = vector.insert %229, %227 [%218, %228] : f16 into vector<2x8xf16>
      %231 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %233 = vector.extract %217[0, 0] : f16 from vector<1x4xf16>
      %234 = vector.insert %233, %arg8 [%231, %232] : f16 into vector<2x8xf16>
      %235 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %236 = vector.extract %217[0, 1] : f16 from vector<1x4xf16>
      %237 = vector.insert %236, %234 [%231, %235] : f16 into vector<2x8xf16>
      %238 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %239 = vector.extract %217[0, 2] : f16 from vector<1x4xf16>
      %240 = vector.insert %239, %237 [%231, %238] : f16 into vector<2x8xf16>
      %241 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %242 = vector.extract %217[0, 3] : f16 from vector<1x4xf16>
      %243 = vector.insert %242, %240 [%231, %241] : f16 into vector<2x8xf16>
      affine.yield %189, %216, %230, %243 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %165 {
      affine.for %arg4 = 0 to 4 {
        %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %174 = vector.extract %167#3[%172, %173] : f16 from vector<2x8xf16>
        %175 = vector.insert %174, %cst_39 [0, 0] : f16 into vector<1x4xf16>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %177 = vector.extract %167#3[%172, %176] : f16 from vector<2x8xf16>
        %178 = vector.insert %177, %175 [0, 1] : f16 into vector<1x4xf16>
        %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %180 = vector.extract %167#3[%172, %179] : f16 from vector<2x8xf16>
        %181 = vector.insert %180, %178 [0, 2] : f16 into vector<1x4xf16>
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %183 = vector.extract %167#3[%172, %182] : f16 from vector<2x8xf16>
        %184 = vector.insert %183, %181 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_40 = gpu.thread_id x
        %185 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %187 = vector.extract %184[0, 0] : f16 from vector<1x4xf16>
        %c0_41 = arith.constant 0 : index
        %c0_42 = arith.constant 0 : index
        %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%186)
        %c0_43 = arith.constant 0 : index
        %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%186)
        %c0_44 = arith.constant 0 : index
        %190 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_42, %189)
        %191 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %188, %c0_41, %c0_43, %c0_44, %190)
        %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %188, %c0_41, %c0_43, %c0_44, %190)
        affine.store %187, %166[%191, %192] {tiled = true} : memref<128x32xf16, 1>
        %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %194 = vector.extract %184[0, 1] : f16 from vector<1x4xf16>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %195 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%193)
        %c0_47 = arith.constant 0 : index
        %196 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%193)
        %c0_48 = arith.constant 0 : index
        %197 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_46, %196)
        %198 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %195, %c0_45, %c0_47, %c0_48, %197)
        %199 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %195, %c0_45, %c0_47, %c0_48, %197)
        affine.store %194, %166[%198, %199] {tiled = true} : memref<128x32xf16, 1>
        %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %201 = vector.extract %184[0, 2] : f16 from vector<1x4xf16>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %202 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%200)
        %c0_51 = arith.constant 0 : index
        %203 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%200)
        %c0_52 = arith.constant 0 : index
        %204 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_50, %203)
        %205 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %202, %c0_49, %c0_51, %c0_52, %204)
        %206 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %202, %c0_49, %c0_51, %c0_52, %204)
        affine.store %201, %166[%205, %206] {tiled = true} : memref<128x32xf16, 1>
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %208 = vector.extract %184[0, 3] : f16 from vector<1x4xf16>
        %c0_53 = arith.constant 0 : index
        %c0_54 = arith.constant 0 : index
        %209 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%207)
        %c0_55 = arith.constant 0 : index
        %210 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%207)
        %c0_56 = arith.constant 0 : index
        %211 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_54, %210)
        %212 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %209, %c0_53, %c0_55, %c0_56, %211)
        %213 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %209, %c0_53, %c0_55, %c0_56, %211)
        affine.store %208, %166[%212, %213] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %168 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %169:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %173 = affine.apply affine_map<() -> (0)>()
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %174 = vector.extract %137#1[%172, %173] : f32 from vector<2x1xf32>
      %175 = vector.insert %174, %cst_39 [0, 0] : f32 into vector<1x1xf32>
      %cst_40 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %176 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_40) -> (vector<1x4xf32>) {
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %cst_44 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %244 = vector.extract %175[%c0_45, %c0_46] : f32 from vector<1x1xf32>
        %245 = vector.insert %244, %cst_44 [0, 0] : f32 into vector<1x4xf32>
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %246 = vector.extract %175[%c0_47, %c0_48] : f32 from vector<1x1xf32>
        %247 = vector.insert %246, %245 [0, 1] : f32 into vector<1x4xf32>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %248 = vector.extract %175[%c0_49, %c0_50] : f32 from vector<1x1xf32>
        %249 = vector.insert %248, %247 [0, 2] : f32 into vector<1x4xf32>
        %c0_51 = arith.constant 0 : index
        %c0_52 = arith.constant 0 : index
        %250 = vector.extract %175[%c0_51, %c0_52] : f32 from vector<1x1xf32>
        %251 = vector.insert %250, %249 [0, 3] : f32 into vector<1x4xf32>
        %252 = vector.extract %251[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg10 [%c0_43, %c0_42] : f32 into vector<1x4xf32>
        %254 = affine.apply affine_map<() -> (1)>()
        %255 = vector.extract %251[0, 1] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %253 [%c0_43, %254] : f32 into vector<1x4xf32>
        %257 = affine.apply affine_map<() -> (2)>()
        %258 = vector.extract %251[0, 2] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %256 [%c0_43, %257] : f32 into vector<1x4xf32>
        %260 = affine.apply affine_map<() -> (3)>()
        %261 = vector.extract %251[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%c0_43, %260] : f32 into vector<1x4xf32>
        affine.yield %262 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %177 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %179 = vector.extract %176[0, 0] : f32 from vector<1x4xf32>
      %180 = vector.insert %179, %arg5 [%177, %178] : f32 into vector<2x8xf32>
      %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %182 = vector.extract %176[0, 1] : f32 from vector<1x4xf32>
      %183 = vector.insert %182, %180 [%177, %181] : f32 into vector<2x8xf32>
      %184 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %185 = vector.extract %176[0, 2] : f32 from vector<1x4xf32>
      %186 = vector.insert %185, %183 [%177, %184] : f32 into vector<2x8xf32>
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %188 = vector.extract %176[0, 3] : f32 from vector<1x4xf32>
      %189 = vector.insert %188, %186 [%177, %187] : f32 into vector<2x8xf32>
      %190 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_41 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %192 = vector.extract %149[%190, %191] : f32 from vector<2x8xf32>
      %193 = vector.insert %192, %cst_41 [0, 0] : f32 into vector<1x4xf32>
      %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %195 = vector.extract %149[%190, %194] : f32 from vector<2x8xf32>
      %196 = vector.insert %195, %193 [0, 1] : f32 into vector<1x4xf32>
      %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %198 = vector.extract %149[%190, %197] : f32 from vector<2x8xf32>
      %199 = vector.insert %198, %196 [0, 2] : f32 into vector<1x4xf32>
      %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %201 = vector.extract %149[%190, %200] : f32 from vector<2x8xf32>
      %202 = vector.insert %201, %199 [0, 3] : f32 into vector<1x4xf32>
      %203 = arith.divf %202, %176 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %204 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %206 = vector.extract %203[0, 0] : f32 from vector<1x4xf32>
      %207 = vector.insert %206, %arg6 [%204, %205] : f32 into vector<2x8xf32>
      %208 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %209 = vector.extract %203[0, 1] : f32 from vector<1x4xf32>
      %210 = vector.insert %209, %207 [%204, %208] : f32 into vector<2x8xf32>
      %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %212 = vector.extract %203[0, 2] : f32 from vector<1x4xf32>
      %213 = vector.insert %212, %210 [%204, %211] : f32 into vector<2x8xf32>
      %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %215 = vector.extract %203[0, 3] : f32 from vector<1x4xf32>
      %216 = vector.insert %215, %213 [%204, %214] : f32 into vector<2x8xf32>
      %217 = arith.truncf %203 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %218 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %220 = vector.extract %217[0, 0] : f16 from vector<1x4xf16>
      %221 = vector.insert %220, %arg7 [%218, %219] : f16 into vector<2x8xf16>
      %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %223 = vector.extract %217[0, 1] : f16 from vector<1x4xf16>
      %224 = vector.insert %223, %221 [%218, %222] : f16 into vector<2x8xf16>
      %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %226 = vector.extract %217[0, 2] : f16 from vector<1x4xf16>
      %227 = vector.insert %226, %224 [%218, %225] : f16 into vector<2x8xf16>
      %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %229 = vector.extract %217[0, 3] : f16 from vector<1x4xf16>
      %230 = vector.insert %229, %227 [%218, %228] : f16 into vector<2x8xf16>
      %231 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %233 = vector.extract %217[0, 0] : f16 from vector<1x4xf16>
      %234 = vector.insert %233, %arg8 [%231, %232] : f16 into vector<2x8xf16>
      %235 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %236 = vector.extract %217[0, 1] : f16 from vector<1x4xf16>
      %237 = vector.insert %236, %234 [%231, %235] : f16 into vector<2x8xf16>
      %238 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %239 = vector.extract %217[0, 2] : f16 from vector<1x4xf16>
      %240 = vector.insert %239, %237 [%231, %238] : f16 into vector<2x8xf16>
      %241 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %242 = vector.extract %217[0, 3] : f16 from vector<1x4xf16>
      %243 = vector.insert %242, %240 [%231, %241] : f16 into vector<2x8xf16>
      affine.yield %189, %216, %230, %243 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %165 {
      affine.for %arg4 = 0 to 4 {
        %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %174 = vector.extract %169#3[%172, %173] : f16 from vector<2x8xf16>
        %175 = vector.insert %174, %cst_39 [0, 0] : f16 into vector<1x4xf16>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %177 = vector.extract %169#3[%172, %176] : f16 from vector<2x8xf16>
        %178 = vector.insert %177, %175 [0, 1] : f16 into vector<1x4xf16>
        %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %180 = vector.extract %169#3[%172, %179] : f16 from vector<2x8xf16>
        %181 = vector.insert %180, %178 [0, 2] : f16 into vector<1x4xf16>
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %183 = vector.extract %169#3[%172, %182] : f16 from vector<2x8xf16>
        %184 = vector.insert %183, %181 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_40 = gpu.thread_id x
        %185 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %187 = vector.extract %184[0, 0] : f16 from vector<1x4xf16>
        %c0_41 = arith.constant 0 : index
        %c0_42 = arith.constant 0 : index
        %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%186)
        %c0_43 = arith.constant 0 : index
        %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%186)
        %c0_44 = arith.constant 0 : index
        %190 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_42, %189)
        %191 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %188, %c0_41, %c0_43, %c0_44, %190)
        %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %188, %c0_41, %c0_43, %c0_44, %190)
        affine.store %187, %168[%191, %192] {tiled = true} : memref<128x32xf16, 1>
        %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %194 = vector.extract %184[0, 1] : f16 from vector<1x4xf16>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %195 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%193)
        %c0_47 = arith.constant 0 : index
        %196 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%193)
        %c0_48 = arith.constant 0 : index
        %197 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_46, %196)
        %198 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %195, %c0_45, %c0_47, %c0_48, %197)
        %199 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %195, %c0_45, %c0_47, %c0_48, %197)
        affine.store %194, %168[%198, %199] {tiled = true} : memref<128x32xf16, 1>
        %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %201 = vector.extract %184[0, 2] : f16 from vector<1x4xf16>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %202 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%200)
        %c0_51 = arith.constant 0 : index
        %203 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%200)
        %c0_52 = arith.constant 0 : index
        %204 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_50, %203)
        %205 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %202, %c0_49, %c0_51, %c0_52, %204)
        %206 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %202, %c0_49, %c0_51, %c0_52, %204)
        affine.store %201, %168[%205, %206] {tiled = true} : memref<128x32xf16, 1>
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %208 = vector.extract %184[0, 3] : f16 from vector<1x4xf16>
        %c0_53 = arith.constant 0 : index
        %c0_54 = arith.constant 0 : index
        %209 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%207)
        %c0_55 = arith.constant 0 : index
        %210 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%207)
        %c0_56 = arith.constant 0 : index
        %211 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_54, %210)
        %212 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %209, %c0_53, %c0_55, %c0_56, %211)
        %213 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %209, %c0_53, %c0_55, %c0_56, %211)
        affine.store %208, %168[%212, %213] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %170 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %171:4 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst, %arg6 = %cst, %arg7 = %cst_4, %arg8 = %cst_4) -> (vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>) {
      %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %173 = affine.apply affine_map<() -> (0)>()
      %cst_39 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %174 = vector.extract %137#1[%172, %173] : f32 from vector<2x1xf32>
      %175 = vector.insert %174, %cst_39 [0, 0] : f32 into vector<1x1xf32>
      %cst_40 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %176 = affine.for %arg9 = 0 to 1 iter_args(%arg10 = %cst_40) -> (vector<1x4xf32>) {
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %cst_44 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %244 = vector.extract %175[%c0_45, %c0_46] : f32 from vector<1x1xf32>
        %245 = vector.insert %244, %cst_44 [0, 0] : f32 into vector<1x4xf32>
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %246 = vector.extract %175[%c0_47, %c0_48] : f32 from vector<1x1xf32>
        %247 = vector.insert %246, %245 [0, 1] : f32 into vector<1x4xf32>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %248 = vector.extract %175[%c0_49, %c0_50] : f32 from vector<1x1xf32>
        %249 = vector.insert %248, %247 [0, 2] : f32 into vector<1x4xf32>
        %c0_51 = arith.constant 0 : index
        %c0_52 = arith.constant 0 : index
        %250 = vector.extract %175[%c0_51, %c0_52] : f32 from vector<1x1xf32>
        %251 = vector.insert %250, %249 [0, 3] : f32 into vector<1x4xf32>
        %252 = vector.extract %251[0, 0] : f32 from vector<1x4xf32>
        %253 = vector.insert %252, %arg10 [%c0_43, %c0_42] : f32 into vector<1x4xf32>
        %254 = affine.apply affine_map<() -> (1)>()
        %255 = vector.extract %251[0, 1] : f32 from vector<1x4xf32>
        %256 = vector.insert %255, %253 [%c0_43, %254] : f32 into vector<1x4xf32>
        %257 = affine.apply affine_map<() -> (2)>()
        %258 = vector.extract %251[0, 2] : f32 from vector<1x4xf32>
        %259 = vector.insert %258, %256 [%c0_43, %257] : f32 into vector<1x4xf32>
        %260 = affine.apply affine_map<() -> (3)>()
        %261 = vector.extract %251[0, 3] : f32 from vector<1x4xf32>
        %262 = vector.insert %261, %259 [%c0_43, %260] : f32 into vector<1x4xf32>
        affine.yield %262 : vector<1x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 1, 4>}
      %177 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %179 = vector.extract %176[0, 0] : f32 from vector<1x4xf32>
      %180 = vector.insert %179, %arg5 [%177, %178] : f32 into vector<2x8xf32>
      %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %182 = vector.extract %176[0, 1] : f32 from vector<1x4xf32>
      %183 = vector.insert %182, %180 [%177, %181] : f32 into vector<2x8xf32>
      %184 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %185 = vector.extract %176[0, 2] : f32 from vector<1x4xf32>
      %186 = vector.insert %185, %183 [%177, %184] : f32 into vector<2x8xf32>
      %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %188 = vector.extract %176[0, 3] : f32 from vector<1x4xf32>
      %189 = vector.insert %188, %186 [%177, %187] : f32 into vector<2x8xf32>
      %190 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %cst_41 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %192 = vector.extract %143[%190, %191] : f32 from vector<2x8xf32>
      %193 = vector.insert %192, %cst_41 [0, 0] : f32 into vector<1x4xf32>
      %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %195 = vector.extract %143[%190, %194] : f32 from vector<2x8xf32>
      %196 = vector.insert %195, %193 [0, 1] : f32 into vector<1x4xf32>
      %197 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %198 = vector.extract %143[%190, %197] : f32 from vector<2x8xf32>
      %199 = vector.insert %198, %196 [0, 2] : f32 into vector<1x4xf32>
      %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %201 = vector.extract %143[%190, %200] : f32 from vector<2x8xf32>
      %202 = vector.insert %201, %199 [0, 3] : f32 into vector<1x4xf32>
      %203 = arith.divf %202, %176 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
      %204 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %205 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %206 = vector.extract %203[0, 0] : f32 from vector<1x4xf32>
      %207 = vector.insert %206, %arg6 [%204, %205] : f32 into vector<2x8xf32>
      %208 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %209 = vector.extract %203[0, 1] : f32 from vector<1x4xf32>
      %210 = vector.insert %209, %207 [%204, %208] : f32 into vector<2x8xf32>
      %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %212 = vector.extract %203[0, 2] : f32 from vector<1x4xf32>
      %213 = vector.insert %212, %210 [%204, %211] : f32 into vector<2x8xf32>
      %214 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %215 = vector.extract %203[0, 3] : f32 from vector<1x4xf32>
      %216 = vector.insert %215, %213 [%204, %214] : f32 into vector<2x8xf32>
      %217 = arith.truncf %203 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
      %218 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %219 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %220 = vector.extract %217[0, 0] : f16 from vector<1x4xf16>
      %221 = vector.insert %220, %arg7 [%218, %219] : f16 into vector<2x8xf16>
      %222 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %223 = vector.extract %217[0, 1] : f16 from vector<1x4xf16>
      %224 = vector.insert %223, %221 [%218, %222] : f16 into vector<2x8xf16>
      %225 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %226 = vector.extract %217[0, 2] : f16 from vector<1x4xf16>
      %227 = vector.insert %226, %224 [%218, %225] : f16 into vector<2x8xf16>
      %228 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %229 = vector.extract %217[0, 3] : f16 from vector<1x4xf16>
      %230 = vector.insert %229, %227 [%218, %228] : f16 into vector<2x8xf16>
      %231 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
      %232 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %233 = vector.extract %217[0, 0] : f16 from vector<1x4xf16>
      %234 = vector.insert %233, %arg8 [%231, %232] : f16 into vector<2x8xf16>
      %235 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %236 = vector.extract %217[0, 1] : f16 from vector<1x4xf16>
      %237 = vector.insert %236, %234 [%231, %235] : f16 into vector<2x8xf16>
      %238 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %239 = vector.extract %217[0, 2] : f16 from vector<1x4xf16>
      %240 = vector.insert %239, %237 [%231, %238] : f16 into vector<2x8xf16>
      %241 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %242 = vector.extract %217[0, 3] : f16 from vector<1x4xf16>
      %243 = vector.insert %242, %240 [%231, %241] : f16 into vector<2x8xf16>
      affine.yield %189, %216, %230, %243 : vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf16>, vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.fused, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    scf.if %165 {
      affine.for %arg4 = 0 to 4 {
        %172 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %cst_39 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %174 = vector.extract %171#3[%172, %173] : f16 from vector<2x8xf16>
        %175 = vector.insert %174, %cst_39 [0, 0] : f16 into vector<1x4xf16>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %177 = vector.extract %171#3[%172, %176] : f16 from vector<2x8xf16>
        %178 = vector.insert %177, %175 [0, 1] : f16 into vector<1x4xf16>
        %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %180 = vector.extract %171#3[%172, %179] : f16 from vector<2x8xf16>
        %181 = vector.insert %180, %178 [0, 2] : f16 into vector<1x4xf16>
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %183 = vector.extract %171#3[%172, %182] : f16 from vector<2x8xf16>
        %184 = vector.insert %183, %181 [0, 3] : f16 into vector<1x4xf16>
        %thread_id_x_40 = gpu.thread_id x
        %185 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg4)
        %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %187 = vector.extract %184[0, 0] : f16 from vector<1x4xf16>
        %c0_41 = arith.constant 0 : index
        %c0_42 = arith.constant 0 : index
        %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%186)
        %c0_43 = arith.constant 0 : index
        %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%186)
        %c0_44 = arith.constant 0 : index
        %190 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_42, %189)
        %191 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %188, %c0_41, %c0_43, %c0_44, %190)
        %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %188, %c0_41, %c0_43, %c0_44, %190)
        affine.store %187, %170[%191, %192] {tiled = true} : memref<128x32xf16, 1>
        %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %194 = vector.extract %184[0, 1] : f16 from vector<1x4xf16>
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %195 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%193)
        %c0_47 = arith.constant 0 : index
        %196 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%193)
        %c0_48 = arith.constant 0 : index
        %197 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_46, %196)
        %198 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %195, %c0_45, %c0_47, %c0_48, %197)
        %199 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %195, %c0_45, %c0_47, %c0_48, %197)
        affine.store %194, %170[%198, %199] {tiled = true} : memref<128x32xf16, 1>
        %200 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %201 = vector.extract %184[0, 2] : f16 from vector<1x4xf16>
        %c0_49 = arith.constant 0 : index
        %c0_50 = arith.constant 0 : index
        %202 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%200)
        %c0_51 = arith.constant 0 : index
        %203 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%200)
        %c0_52 = arith.constant 0 : index
        %204 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_50, %203)
        %205 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %202, %c0_49, %c0_51, %c0_52, %204)
        %206 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %202, %c0_49, %c0_51, %c0_52, %204)
        affine.store %201, %170[%205, %206] {tiled = true} : memref<128x32xf16, 1>
        %207 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %208 = vector.extract %184[0, 3] : f16 from vector<1x4xf16>
        %c0_53 = arith.constant 0 : index
        %c0_54 = arith.constant 0 : index
        %209 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%207)
        %c0_55 = arith.constant 0 : index
        %210 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%207)
        %c0_56 = arith.constant 0 : index
        %211 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_54, %210)
        %212 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_40, %185, %209, %c0_53, %c0_55, %c0_56, %211)
        %213 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_40, %185, %209, %c0_53, %c0_55, %c0_56, %211)
        affine.store %208, %170[%212, %213] {tiled = true} : memref<128x32xf16, 1>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
