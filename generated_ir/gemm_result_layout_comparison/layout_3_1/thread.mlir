module {
  func.func @Attn(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> {
    %cst = arith.constant dense<1.131250e+01> : tensor<1xf32>
    %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFF800000 : f32
    %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16>
    %4 = deepgengraph.precise_dot_op %1, %3, acc = f32 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf32>
    %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf32>, tensor<1xf32>) -> tensor<1x32x4096x4096xf32>
    %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %7 = deepgengraph.reduce(%6), dim = -1, op = MAX, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %8 = deepgengraph.sub %6, %7 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %9 = deepgengraph.exp %8 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %10 = deepgengraph.reduce(%9), dim = -1, op = ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %11 = deepgengraph.div %9, %10 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %12 = deepgengraph.convert %11, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
    %13 = deepgengraph.dot %12, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16>
    %14 = deepgengraph.permute %13, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16>
    return %14 : tensor<1x4096x32x128xf16>
  }
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 4, 0, 1, 1>, thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %c32 = arith.constant 32 : index
    %0 = arith.muli %block_id_y, %c32 : index
    %1 = arith.addi %block_id_x, %0 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %2 = arith.divui %1, %c128 : index
    %3 = arith.muli %2, %c4 : index
    %c4_0 = arith.constant 4 : index
    %4 = arith.remui %1, %c128 : index
    %5 = arith.remui %4, %c4_0 : index
    %6 = arith.addi %3, %5 : index
    %7 = arith.divui %4, %c4_0 : index
    %c31 = arith.constant 31 : index
    %8 = arith.subi %c31, %6 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c128_3 = arith.constant 128 : index
    %cst_4 = arith.constant 0xFF800000 : f32
    %9 = arith.truncf %cst : f32 to f16
    %10 = arith.muli %8, %c128_3 : index
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_5) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %53 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %54 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_36 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
        frisk.fragment_yield %cst_36 : vector<1x1xf32>
      }
      %55 = vector.extract %54[0, 0] : f32 from vector<1x1xf32>
      %56 = vector.insert %55, %arg5 [%53, %c0] : f32 into vector<2x1xf32>
      affine.yield %56 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %12 = vector.broadcast %cst_4 : f32 to vector<2x1xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %13 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %53 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %54 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %57 = vector.extract %11[%53, %c0] : f32 from vector<2x1xf32>
        %58 = vector.insert %57, %cst_36 [0, 0] : f32 into vector<1x1xf32>
        %cst_37 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %59 = vector.extract %12[%53, %c0] : f32 from vector<2x1xf32>
        %60 = vector.insert %59, %cst_37 [0, 0] : f32 into vector<1x1xf32>
        %61 = arith.addf %58, %60 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        frisk.fragment_yield %61 : vector<1x1xf32>
      }
      %55 = vector.extract %54[0, 0] : f32 from vector<1x1xf32>
      %56 = vector.insert %55, %arg5 [%53, %c0] : f32 into vector<2x1xf32>
      affine.yield %56 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %14 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_9) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %53 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %54 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %57 = vector.extract %13[%53, %c0] : f32 from vector<2x1xf32>
        %58 = vector.insert %57, %cst_36 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
        frisk.fragment_yield %58 : vector<1x1xf32>
      }
      %55 = vector.extract %54[0, 0] : f32 from vector<1x1xf32>
      %56 = vector.insert %55, %arg5 [%53, %c0] : f32 into vector<2x1xf32>
      affine.yield %56 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %15 = frisk.from_threadTile %14 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %16:6 = affine.for %arg4 = 0 to affine_map<()[s0] -> (s0 * 128 + 128)>()[%8] step 128 iter_args(%arg5 = %cst_7, %arg6 = %cst_8, %arg7 = %15, %arg8 = %cst_10, %arg9 = %cst_11, %arg10 = %cst_12) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %cst_36 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %53 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_36) -> (vector<128x128xf32>) {
        %alloc_119 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_120 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %155 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %8 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %155 to %alloc_119 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %156 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_119 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %156 to %alloc_120 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %157 = frisk.to_threadTile %alloc_119 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
        %158 = vector.broadcast %9 : f16 to vector<2x8xf16>
        %cst_121 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
        %159 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_121) -> (vector<2x8xf16>) {
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %166 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg13)
          %167 = frisk.fragment "compute" -> (vector<1x4xf16>) {
            %cst_126 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %179 = vector.extract %157[%166, %165] : f16 from vector<2x8xf16>
            %180 = vector.insert %179, %cst_126 [0, 0] : f16 into vector<1x4xf16>
            %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg13)
            %182 = vector.extract %157[%166, %181] : f16 from vector<2x8xf16>
            %183 = vector.insert %182, %180 [0, 1] : f16 into vector<1x4xf16>
            %184 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg13)
            %185 = vector.extract %157[%166, %184] : f16 from vector<2x8xf16>
            %186 = vector.insert %185, %183 [0, 2] : f16 into vector<1x4xf16>
            %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg13)
            %188 = vector.extract %157[%166, %187] : f16 from vector<2x8xf16>
            %189 = vector.insert %188, %186 [0, 3] : f16 into vector<1x4xf16>
            %cst_127 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %190 = vector.extract %158[%166, %165] : f16 from vector<2x8xf16>
            %191 = vector.insert %190, %cst_127 [0, 0] : f16 into vector<1x4xf16>
            %192 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg13)
            %193 = vector.extract %158[%166, %192] : f16 from vector<2x8xf16>
            %194 = vector.insert %193, %191 [0, 1] : f16 into vector<1x4xf16>
            %195 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg13)
            %196 = vector.extract %158[%166, %195] : f16 from vector<2x8xf16>
            %197 = vector.insert %196, %194 [0, 2] : f16 into vector<1x4xf16>
            %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg13)
            %199 = vector.extract %158[%166, %198] : f16 from vector<2x8xf16>
            %200 = vector.insert %199, %197 [0, 3] : f16 into vector<1x4xf16>
            %201 = arith.mulf %189, %200 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
            frisk.fragment_yield %201 : vector<1x4xf16>
          }
          %168 = vector.extract %167[0, 0] : f16 from vector<1x4xf16>
          %169 = vector.insert %168, %arg14 [%166, %165] : f16 into vector<2x8xf16>
          %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg13)
          %171 = vector.extract %167[0, 1] : f16 from vector<1x4xf16>
          %172 = vector.insert %171, %169 [%166, %170] : f16 into vector<2x8xf16>
          %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg13)
          %174 = vector.extract %167[0, 2] : f16 from vector<1x4xf16>
          %175 = vector.insert %174, %172 [%166, %173] : f16 into vector<2x8xf16>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg13)
          %177 = vector.extract %167[0, 3] : f16 from vector<1x4xf16>
          %178 = vector.insert %177, %175 [%166, %176] : f16 into vector<2x8xf16>
          affine.yield %178 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        frisk.wait_buffer_ready %alloc_120 : memref<32x128xf16, 3>
        %cst_122 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %cst_123 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %160 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_122) -> (vector<2x32xf32>) {
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %166 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %167 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %cst_123) -> (vector<1x4xf32>) {
            %179 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %182 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
              %183 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg15)
              %cst_126 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
              %184 = vector.extract %159[%182, %183] : f16 from vector<2x8xf16>
              %185 = vector.insert %184, %cst_126 [0, 0] : f16 into vector<1x4xf16>
              %186 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg15)
              %187 = vector.extract %159[%182, %186] : f16 from vector<2x8xf16>
              %188 = vector.insert %187, %185 [0, 1] : f16 into vector<1x4xf16>
              %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg15)
              %190 = vector.extract %159[%182, %189] : f16 from vector<2x8xf16>
              %191 = vector.insert %190, %188 [0, 2] : f16 into vector<1x4xf16>
              %192 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg15)
              %193 = vector.extract %159[%182, %192] : f16 from vector<2x8xf16>
              %194 = vector.insert %193, %191 [0, 3] : f16 into vector<1x4xf16>
              frisk.fragment_yield %194 : vector<1x4xf16>
            }
            %180 = frisk.fragment "load" -> (vector<4x1xf16>) {
              %182 = frisk.to_threadTile %alloc_120 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
              %183 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg15)
              %184 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
              %cst_126 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
              %185 = vector.extract %182[%183, %184] : f16 from vector<8x8xf16>
              %186 = vector.insert %185, %cst_126 [0, 0] : f16 into vector<4x1xf16>
              %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg15)
              %188 = vector.extract %182[%187, %184] : f16 from vector<8x8xf16>
              %189 = vector.insert %188, %186 [1, 0] : f16 into vector<4x1xf16>
              %190 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg15)
              %191 = vector.extract %182[%190, %184] : f16 from vector<8x8xf16>
              %192 = vector.insert %191, %189 [2, 0] : f16 into vector<4x1xf16>
              %193 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg15)
              %194 = vector.extract %182[%193, %184] : f16 from vector<8x8xf16>
              %195 = vector.insert %194, %192 [3, 0] : f16 into vector<4x1xf16>
              frisk.fragment_yield %195 : vector<4x1xf16>
            }
            %181 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %182 = frisk.warp_mma_rr %179, %180, %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %182 : vector<1x4xf32>
            }
            affine.yield %181 : vector<1x4xf32>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %168 = vector.extract %167[0, 0] : f32 from vector<1x4xf32>
          %169 = vector.insert %168, %arg14 [%166, %165] : f32 into vector<2x32xf32>
          %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %171 = vector.extract %167[0, 1] : f32 from vector<1x4xf32>
          %172 = vector.insert %171, %169 [%166, %170] : f32 into vector<2x32xf32>
          %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %174 = vector.extract %167[0, 2] : f32 from vector<1x4xf32>
          %175 = vector.insert %174, %172 [%166, %173] : f32 into vector<2x32xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %177 = vector.extract %167[0, 3] : f32 from vector<1x4xf32>
          %178 = vector.insert %177, %175 [%166, %176] : f32 into vector<2x32xf32>
          affine.yield %178 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %161 = frisk.to_threadTile %arg12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
        %cst_124 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %162 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_124) -> (vector<2x32xf32>) {
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %166 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %167 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_126 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %179 = vector.extract %161[%166, %165] : f32 from vector<2x32xf32>
            %180 = vector.insert %179, %cst_126 [0, 0] : f32 into vector<1x4xf32>
            %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
            %182 = vector.extract %161[%166, %181] : f32 from vector<2x32xf32>
            %183 = vector.insert %182, %180 [0, 1] : f32 into vector<1x4xf32>
            %184 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
            %185 = vector.extract %161[%166, %184] : f32 from vector<2x32xf32>
            %186 = vector.insert %185, %183 [0, 2] : f32 into vector<1x4xf32>
            %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
            %188 = vector.extract %161[%166, %187] : f32 from vector<2x32xf32>
            %189 = vector.insert %188, %186 [0, 3] : f32 into vector<1x4xf32>
            %cst_127 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %190 = vector.extract %160[%166, %165] : f32 from vector<2x32xf32>
            %191 = vector.insert %190, %cst_127 [0, 0] : f32 into vector<1x4xf32>
            %192 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
            %193 = vector.extract %160[%166, %192] : f32 from vector<2x32xf32>
            %194 = vector.insert %193, %191 [0, 1] : f32 into vector<1x4xf32>
            %195 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
            %196 = vector.extract %160[%166, %195] : f32 from vector<2x32xf32>
            %197 = vector.insert %196, %194 [0, 2] : f32 into vector<1x4xf32>
            %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
            %199 = vector.extract %160[%166, %198] : f32 from vector<2x32xf32>
            %200 = vector.insert %199, %197 [0, 3] : f32 into vector<1x4xf32>
            %201 = arith.addf %189, %200 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %201 : vector<1x4xf32>
          }
          %168 = vector.extract %167[0, 0] : f32 from vector<1x4xf32>
          %169 = vector.insert %168, %arg14 [%166, %165] : f32 into vector<2x32xf32>
          %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %171 = vector.extract %167[0, 1] : f32 from vector<1x4xf32>
          %172 = vector.insert %171, %169 [%166, %170] : f32 into vector<2x32xf32>
          %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %174 = vector.extract %167[0, 2] : f32 from vector<1x4xf32>
          %175 = vector.insert %174, %172 [%166, %173] : f32 into vector<2x32xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %177 = vector.extract %167[0, 3] : f32 from vector<1x4xf32>
          %178 = vector.insert %177, %175 [%166, %176] : f32 into vector<2x32xf32>
          affine.yield %178 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %cst_125 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %163 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_125) -> (vector<2x32xf32>) {
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %166 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %167 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_126 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %179 = vector.extract %162[%166, %165] : f32 from vector<2x32xf32>
            %180 = vector.insert %179, %cst_126 [0, 0] : f32 into vector<1x4xf32>
            %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
            %182 = vector.extract %162[%166, %181] : f32 from vector<2x32xf32>
            %183 = vector.insert %182, %180 [0, 1] : f32 into vector<1x4xf32>
            %184 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
            %185 = vector.extract %162[%166, %184] : f32 from vector<2x32xf32>
            %186 = vector.insert %185, %183 [0, 2] : f32 into vector<1x4xf32>
            %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
            %188 = vector.extract %162[%166, %187] : f32 from vector<2x32xf32>
            %189 = vector.insert %188, %186 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
            frisk.fragment_yield %189 : vector<1x4xf32>
          }
          %168 = vector.extract %167[0, 0] : f32 from vector<1x4xf32>
          %169 = vector.insert %168, %arg14 [%166, %165] : f32 into vector<2x32xf32>
          %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %171 = vector.extract %167[0, 1] : f32 from vector<1x4xf32>
          %172 = vector.insert %171, %169 [%166, %170] : f32 into vector<2x32xf32>
          %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %174 = vector.extract %167[0, 2] : f32 from vector<1x4xf32>
          %175 = vector.insert %174, %172 [%166, %173] : f32 into vector<2x32xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %177 = vector.extract %167[0, 3] : f32 from vector<1x4xf32>
          %178 = vector.insert %177, %175 [%166, %176] : f32 into vector<2x32xf32>
          affine.yield %178 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %164 = frisk.from_threadTile %163 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
        affine.yield %164 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %54 = frisk.to_threadTile %53 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %alloc = memref.alloc() {tiled = true} : memref<128x128xf32, 3>
      %thread_id_x_37 = gpu.thread_id x
      %thread_id_x_38 = gpu.thread_id x
      %55 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_38)
      %c4_39 = arith.constant 4 : index
      %56 = arith.cmpi ult, %55, %c4_39 : index
      scf.if %56 {
        affine.for %arg11 = 0 to 16 {
          %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
          %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
          frisk.fragment "store" {
            %157 = vector.extract %54[%156, %155] : f32 from vector<2x32xf32>
            %c0 = arith.constant 0 : index
            %c0_119 = arith.constant 0 : index
            %158 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%155)
            %c0_120 = arith.constant 0 : index
            %159 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%155)
            %160 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_119, %159)
            %c0_121 = arith.constant 0 : index
            %161 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_38, %156, %158, %c0, %c0_120, %160, %c0_121)
            %162 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>(%thread_id_x_38, %156, %158, %c0, %c0_120, %160, %c0_121)
            affine.store %157, %alloc[%161, %162] {tiled = true} : memref<128x128xf32, 3>
            %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
            %164 = vector.extract %54[%156, %163] : f32 from vector<2x32xf32>
            %c0_122 = arith.constant 0 : index
            %c0_123 = arith.constant 0 : index
            %165 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%163)
            %c0_124 = arith.constant 0 : index
            %166 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%163)
            %167 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_123, %166)
            %c0_125 = arith.constant 0 : index
            %168 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_38, %156, %165, %c0_122, %c0_124, %167, %c0_125)
            %169 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>(%thread_id_x_38, %156, %165, %c0_122, %c0_124, %167, %c0_125)
            affine.store %164, %alloc[%168, %169] {tiled = true} : memref<128x128xf32, 3>
            %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
            %171 = vector.extract %54[%156, %170] : f32 from vector<2x32xf32>
            %c0_126 = arith.constant 0 : index
            %c0_127 = arith.constant 0 : index
            %172 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%170)
            %c0_128 = arith.constant 0 : index
            %173 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%170)
            %174 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_127, %173)
            %c0_129 = arith.constant 0 : index
            %175 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_38, %156, %172, %c0_126, %c0_128, %174, %c0_129)
            %176 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>(%thread_id_x_38, %156, %172, %c0_126, %c0_128, %174, %c0_129)
            affine.store %171, %alloc[%175, %176] {tiled = true} : memref<128x128xf32, 3>
            %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
            %178 = vector.extract %54[%156, %177] : f32 from vector<2x32xf32>
            %c0_130 = arith.constant 0 : index
            %c0_131 = arith.constant 0 : index
            %179 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%177)
            %c0_132 = arith.constant 0 : index
            %180 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%177)
            %181 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_131, %180)
            %c0_133 = arith.constant 0 : index
            %182 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_38, %156, %179, %c0_130, %c0_132, %181, %c0_133)
            %183 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>(%thread_id_x_38, %156, %179, %c0_130, %c0_132, %181, %c0_133)
            affine.store %178, %alloc[%182, %183] {tiled = true} : memref<128x128xf32, 3>
            frisk.fragment_yield
          }
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "store", tiled = true}
      }
      gpu.barrier {tiled = true}
      %cst_40 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %57 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_40) -> (vector<2x32xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %157 = frisk.fragment "load" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %c0_120 = arith.constant 0 : index
          %169 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%155)
          %c0_121 = arith.constant 0 : index
          %170 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%155)
          %171 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%170)
          %c0_122 = arith.constant 0 : index
          %172 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_120, %171)
          %173 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_37, %156, %169, %c0, %c0_121, %c0_122, %172)
          %174 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_37, %156, %169, %c0, %c0_121, %c0_122, %172)
          %175 = affine.load %alloc[%173, %174] : memref<128x128xf32, 3>
          %176 = vector.insert %175, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %c0_123 = arith.constant 0 : index
          %c0_124 = arith.constant 0 : index
          %178 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%177)
          %c0_125 = arith.constant 0 : index
          %179 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%177)
          %180 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%179)
          %c0_126 = arith.constant 0 : index
          %181 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_124, %180)
          %182 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_37, %156, %178, %c0_123, %c0_125, %c0_126, %181)
          %183 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_37, %156, %178, %c0_123, %c0_125, %c0_126, %181)
          %184 = affine.load %alloc[%182, %183] : memref<128x128xf32, 3>
          %185 = vector.insert %184, %176 [0, 1] : f32 into vector<1x4xf32>
          %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %c0_127 = arith.constant 0 : index
          %c0_128 = arith.constant 0 : index
          %187 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%186)
          %c0_129 = arith.constant 0 : index
          %188 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%186)
          %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%188)
          %c0_130 = arith.constant 0 : index
          %190 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_128, %189)
          %191 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_37, %156, %187, %c0_127, %c0_129, %c0_130, %190)
          %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_37, %156, %187, %c0_127, %c0_129, %c0_130, %190)
          %193 = affine.load %alloc[%191, %192] : memref<128x128xf32, 3>
          %194 = vector.insert %193, %185 [0, 2] : f32 into vector<1x4xf32>
          %195 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %c0_131 = arith.constant 0 : index
          %c0_132 = arith.constant 0 : index
          %196 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%195)
          %c0_133 = arith.constant 0 : index
          %197 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%195)
          %198 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%197)
          %c0_134 = arith.constant 0 : index
          %199 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_132, %198)
          %200 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_37, %156, %196, %c0_131, %c0_133, %c0_134, %199)
          %201 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_37, %156, %196, %c0_131, %c0_133, %c0_134, %199)
          %202 = affine.load %alloc[%200, %201] : memref<128x128xf32, 3>
          %203 = vector.insert %202, %194 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %203 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x32xf32>
        affine.yield %168 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "layout_fragment", tiled = true}
      frisk.sync_threads_in_block {tiled = true}
      %cst_41 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %thread_id_x_42 = gpu.thread_id x
      %c127 = arith.constant 127 : index
      %58 = arith.addi %10, %c127 : index
      %c127_43 = arith.constant 127 : index
      %59 = arith.addi %arg4, %c127_43 : index
      %c1_44 = arith.constant 1 : index
      %60 = arith.addi %10, %c1_44 : index
      %c1_45 = arith.constant 1 : index
      %61 = arith.addi %58, %c1_45 : index
      %62 = arith.cmpi uge, %61, %58 : index
      %63 = arith.cmpi uge, %58, %10 : index
      %64 = arith.andi %63, %62 : i1
      %65 = arith.cmpi uge, %59, %arg4 : index
      %66 = arith.andi %64, %65 : i1
      %67 = arith.cmpi ugt, %60, %59 : index
      %68 = arith.andi %66, %67 : i1
      %cst_46 = arith.constant 0.000000e+00 : f32
      %69 = scf.if %68 -> (vector<2x32xf32>) {
        %155 = vector.broadcast %cst_46 : f32 to vector<2x32xf32>
        scf.yield %155 : vector<2x32xf32>
      } else {
        %155 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_41) -> (vector<2x32xf32>) {
          %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
          %157 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
          %158 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %c0 = arith.constant 0 : index
            %c0_120 = arith.constant 0 : index
            %170 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%156)
            %c0_121 = arith.constant 0 : index
            %171 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%156)
            %172 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%171)
            %c0_122 = arith.constant 0 : index
            %173 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_120, %172)
            %174 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_42, %157, %170, %c0, %c0_121, %c0_122, %173)
            %175 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_42, %157, %170, %c0, %c0_121, %c0_122, %173)
            %176 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %174)
            %177 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %175)
            %178 = arith.addi %176, %c1 : index
            %179 = arith.cmpi ule, %178, %177 : index
            %180 = arith.select %179, %cst_1, %cst_2 : f32
            %181 = vector.insert %180, %cst_119 [0, 0] : f32 into vector<1x4xf32>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
            %c0_123 = arith.constant 0 : index
            %c0_124 = arith.constant 0 : index
            %183 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%182)
            %c0_125 = arith.constant 0 : index
            %184 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%182)
            %185 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%184)
            %c0_126 = arith.constant 0 : index
            %186 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_124, %185)
            %187 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_42, %157, %183, %c0_123, %c0_125, %c0_126, %186)
            %188 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_42, %157, %183, %c0_123, %c0_125, %c0_126, %186)
            %189 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %187)
            %190 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %188)
            %191 = arith.addi %189, %c1 : index
            %192 = arith.cmpi ule, %191, %190 : index
            %193 = arith.select %192, %cst_1, %cst_2 : f32
            %194 = vector.insert %193, %181 [0, 1] : f32 into vector<1x4xf32>
            %195 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
            %c0_127 = arith.constant 0 : index
            %c0_128 = arith.constant 0 : index
            %196 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%195)
            %c0_129 = arith.constant 0 : index
            %197 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%195)
            %198 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%197)
            %c0_130 = arith.constant 0 : index
            %199 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_128, %198)
            %200 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_42, %157, %196, %c0_127, %c0_129, %c0_130, %199)
            %201 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_42, %157, %196, %c0_127, %c0_129, %c0_130, %199)
            %202 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %200)
            %203 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %201)
            %204 = arith.addi %202, %c1 : index
            %205 = arith.cmpi ule, %204, %203 : index
            %206 = arith.select %205, %cst_1, %cst_2 : f32
            %207 = vector.insert %206, %194 [0, 2] : f32 into vector<1x4xf32>
            %208 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
            %c0_131 = arith.constant 0 : index
            %c0_132 = arith.constant 0 : index
            %209 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%208)
            %c0_133 = arith.constant 0 : index
            %210 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%208)
            %211 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%210)
            %c0_134 = arith.constant 0 : index
            %212 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_132, %211)
            %213 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_42, %157, %209, %c0_131, %c0_133, %c0_134, %212)
            %214 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_42, %157, %209, %c0_131, %c0_133, %c0_134, %212)
            %215 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %213)
            %216 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %214)
            %217 = arith.addi %215, %c1 : index
            %218 = arith.cmpi ule, %217, %216 : index
            %219 = arith.select %218, %cst_1, %cst_2 : f32
            %220 = vector.insert %219, %207 [0, 3] : f32 into vector<1x4xf32>
            frisk.fragment_yield %220 : vector<1x4xf32>
          }
          %159 = vector.extract %158[0, 0] : f32 from vector<1x4xf32>
          %160 = vector.insert %159, %arg12 [%157, %156] : f32 into vector<2x32xf32>
          %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %162 = vector.extract %158[0, 1] : f32 from vector<1x4xf32>
          %163 = vector.insert %162, %160 [%157, %161] : f32 into vector<2x32xf32>
          %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %165 = vector.extract %158[0, 2] : f32 from vector<1x4xf32>
          %166 = vector.insert %165, %163 [%157, %164] : f32 into vector<2x32xf32>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %168 = vector.extract %158[0, 3] : f32 from vector<1x4xf32>
          %169 = vector.insert %168, %166 [%157, %167] : f32 into vector<2x32xf32>
          affine.yield %169 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "mask_fragment", tiled = true}
        scf.yield %155 : vector<2x32xf32>
      } {frisk.uniform_mask}
      %cst_47 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %70 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_47) -> (vector<2x32xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %57[%156, %155] : f32 from vector<2x32xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %172 = vector.extract %57[%156, %171] : f32 from vector<2x32xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %175 = vector.extract %57[%156, %174] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %178 = vector.extract %57[%156, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %69[%156, %155] : f32 from vector<2x32xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %183 = vector.extract %69[%156, %182] : f32 from vector<2x32xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %186 = vector.extract %69[%156, %185] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %189 = vector.extract %69[%156, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.addf %179, %190 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x32xf32>
        affine.yield %168 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_48 = arith.constant 0xFF800000 : f32
      %cst_49 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %71 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_49) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %156 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %159 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_48) -> (f32) {
            %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
            %164 = frisk.fragment "compute" -> (f32) {
              %165 = vector.extract %70[%155, %163] : f32 from vector<2x32xf32>
              %166 = arith.maxnumf %arg14, %165 : f32
              %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
              %168 = vector.extract %70[%155, %167] : f32 from vector<2x32xf32>
              %169 = arith.maxnumf %166, %168 : f32
              %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
              %171 = vector.extract %70[%155, %170] : f32 from vector<2x32xf32>
              %172 = arith.maxnumf %169, %171 : f32
              %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
              %174 = vector.extract %70[%155, %173] : f32 from vector<2x32xf32>
              %175 = arith.maxnumf %172, %174 : f32
              frisk.fragment_yield %175 : f32
            }
            affine.yield %164 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_120 = arith.constant 64 : i32
          %c16_i32_121 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %159, %c16_i32_121, %c64_i32_120 : f32
          %160 = arith.maxnumf %159, %shuffleResult : f32
          %c64_i32_122 = arith.constant 64 : i32
          %c32_i32_123 = arith.constant 32 : i32
          %shuffleResult_124, %valid_125 = gpu.shuffle xor %160, %c32_i32_123, %c64_i32_122 : f32
          %161 = arith.maxnumf %160, %shuffleResult_124 : f32
          %162 = vector.insert %161, %cst_119 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %162 : vector<1x1xf32>
        }
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f32 into vector<2x1xf32>
        affine.yield %158 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %72 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_50 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %73 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_50) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %156 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %159 = vector.extract %71[%155, %c0] : f32 from vector<2x1xf32>
          %160 = vector.insert %159, %cst_119 [0, 0] : f32 into vector<1x1xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %161 = vector.extract %72[%155, %c0] : f32 from vector<2x1xf32>
          %162 = vector.insert %161, %cst_120 [0, 0] : f32 into vector<1x1xf32>
          %163 = arith.maxnumf %160, %162 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %163 : vector<1x1xf32>
        }
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f32 into vector<2x1xf32>
        affine.yield %158 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_51 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %74 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_51) -> (vector<2x32xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %169 = vector.extract %73[%156, %c0] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %c0_120 = arith.constant 0 : index
          %171 = vector.extract %73[%156, %c0_120] : f32 from vector<2x1xf32>
          %172 = vector.insert %171, %170 [0, 1] : f32 into vector<1x4xf32>
          %c0_121 = arith.constant 0 : index
          %173 = vector.extract %73[%156, %c0_121] : f32 from vector<2x1xf32>
          %174 = vector.insert %173, %172 [0, 2] : f32 into vector<1x4xf32>
          %c0_122 = arith.constant 0 : index
          %175 = vector.extract %73[%156, %c0_122] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %174 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %176 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x32xf32>
        affine.yield %168 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %cst_52 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %75 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_52) -> (vector<2x32xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %70[%156, %155] : f32 from vector<2x32xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %172 = vector.extract %70[%156, %171] : f32 from vector<2x32xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %175 = vector.extract %70[%156, %174] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %178 = vector.extract %70[%156, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %74[%156, %155] : f32 from vector<2x32xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %183 = vector.extract %74[%156, %182] : f32 from vector<2x32xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %186 = vector.extract %74[%156, %185] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %189 = vector.extract %74[%156, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.subf %179, %190 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x32xf32>
        affine.yield %168 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %76 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_53 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %77 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_53) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %156 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %159 = vector.extract %76[%155, %c0] : f32 from vector<2x1xf32>
          %160 = vector.insert %159, %cst_119 [0, 0] : f32 into vector<1x1xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %161 = vector.extract %73[%155, %c0] : f32 from vector<2x1xf32>
          %162 = vector.insert %161, %cst_120 [0, 0] : f32 into vector<1x1xf32>
          %163 = arith.subf %160, %162 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %163 : vector<1x1xf32>
        }
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f32 into vector<2x1xf32>
        affine.yield %158 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_54 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %78 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_54) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %156 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %159 = vector.extract %77[%155, %c0] : f32 from vector<2x1xf32>
          %160 = vector.insert %159, %cst_119 [0, 0] : f32 into vector<1x1xf32>
          %161 = math.exp2 %160 {tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %161 : vector<1x1xf32>
        }
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f32 into vector<2x1xf32>
        affine.yield %158 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %79 = frisk.from_threadTile %78 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 1, 1>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 128 + ((d0 floordiv 64) mod 4) * 32 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 2, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> memref<128x1xf32>
      %80 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %81 = frisk.to_threadTile %79 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_55 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %82 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_55) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %156 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %159 = vector.extract %80[%155, %c0] : f32 from vector<2x1xf32>
          %160 = vector.insert %159, %cst_119 [0, 0] : f32 into vector<1x1xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %161 = vector.extract %81[%155, %c0] : f32 from vector<2x1xf32>
          %162 = vector.insert %161, %cst_120 [0, 0] : f32 into vector<1x1xf32>
          %163 = arith.mulf %160, %162 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %163 : vector<1x1xf32>
        }
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f32 into vector<2x1xf32>
        affine.yield %158 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_56 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %83 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_56) -> (vector<2x32xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %75[%156, %155] : f32 from vector<2x32xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %172 = vector.extract %75[%156, %171] : f32 from vector<2x32xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %175 = vector.extract %75[%156, %174] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %178 = vector.extract %75[%156, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %180 = math.exp2 %179 {tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %180 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x32xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x32xf32>
        affine.yield %168 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_57 = arith.constant 0.000000e+00 : f32
      %cst_58 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %84 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_58) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %156 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %159 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_57) -> (f32) {
            %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
            %164 = frisk.fragment "compute" -> (f32) {
              %165 = vector.extract %83[%155, %163] : f32 from vector<2x32xf32>
              %166 = arith.addf %arg14, %165 : f32
              %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
              %168 = vector.extract %83[%155, %167] : f32 from vector<2x32xf32>
              %169 = arith.addf %166, %168 : f32
              %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
              %171 = vector.extract %83[%155, %170] : f32 from vector<2x32xf32>
              %172 = arith.addf %169, %171 : f32
              %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
              %174 = vector.extract %83[%155, %173] : f32 from vector<2x32xf32>
              %175 = arith.addf %172, %174 : f32
              frisk.fragment_yield %175 : f32
            }
            affine.yield %164 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_120 = arith.constant 64 : i32
          %c16_i32_121 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %159, %c16_i32_121, %c64_i32_120 : f32
          %160 = arith.addf %159, %shuffleResult : f32
          %c64_i32_122 = arith.constant 64 : i32
          %c32_i32_123 = arith.constant 32 : i32
          %shuffleResult_124, %valid_125 = gpu.shuffle xor %160, %c32_i32_123, %c64_i32_122 : f32
          %161 = arith.addf %160, %shuffleResult_124 : f32
          %162 = vector.insert %161, %cst_119 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %162 : vector<1x1xf32>
        }
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f32 into vector<2x1xf32>
        affine.yield %158 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %cst_59 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %85 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_59) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %156 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %159 = vector.extract %84[%155, %c0] : f32 from vector<2x1xf32>
          %160 = vector.insert %159, %cst_119 [0, 0] : f32 into vector<1x1xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %161 = vector.extract %82[%155, %c0] : f32 from vector<2x1xf32>
          %162 = vector.insert %161, %cst_120 [0, 0] : f32 into vector<1x1xf32>
          %163 = arith.addf %160, %162 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %163 : vector<1x1xf32>
        }
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f32 into vector<2x1xf32>
        affine.yield %158 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_60 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
      %86 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_60) -> (vector<2x32xf16>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %83[%156, %155] : f32 from vector<2x32xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %172 = vector.extract %83[%156, %171] : f32 from vector<2x32xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %175 = vector.extract %83[%156, %174] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %178 = vector.extract %83[%156, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %180 = arith.truncf %179 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %180 : vector<1x4xf16>
        }
        %158 = vector.extract %157[0, 0] : f16 from vector<1x4xf16>
        %159 = vector.insert %158, %arg12 [%156, %155] : f16 into vector<2x32xf16>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f16 from vector<1x4xf16>
        %162 = vector.insert %161, %159 [%156, %160] : f16 into vector<2x32xf16>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f16 from vector<1x4xf16>
        %165 = vector.insert %164, %162 [%156, %163] : f16 into vector<2x32xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f16 from vector<1x4xf16>
        %168 = vector.insert %167, %165 [%156, %166] : f16 into vector<2x32xf16>
        affine.yield %168 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_61 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %87 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %87 to %alloc_61 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_61 : memref<128x32xf16, 3>
      %thread_id_x_62 = gpu.thread_id x
      %cst_63 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %88 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_63) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %183 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4)>(%arg11)[%thread_id_x_62]
          %184 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_62]
          %185 = frisk.read_matrix_from_shm %alloc_61[%183, %184] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %186 = vector.bitcast %185 : vector<4xi32> to vector<8xf16>
          %cst_119 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %187 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_62]
          %188 = vector.extract %186[%187] : f16 from vector<8xf16>
          %189 = vector.insert %188, %cst_119 [0, 0] : f16 into vector<4x2xf16>
          %190 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_62]
          %191 = vector.extract %186[%190] : f16 from vector<8xf16>
          %192 = vector.insert %191, %189 [0, 1] : f16 into vector<4x2xf16>
          %193 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_62]
          %194 = vector.extract %186[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %192 [1, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_62]
          %197 = vector.extract %186[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [1, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_62]
          %200 = vector.extract %186[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [2, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_62]
          %203 = vector.extract %186[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [2, 1] : f16 into vector<4x2xf16>
          %205 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_62]
          %206 = vector.extract %186[%205] : f16 from vector<8xf16>
          %207 = vector.insert %206, %204 [3, 0] : f16 into vector<4x2xf16>
          %208 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_62]
          %209 = vector.extract %186[%208] : f16 from vector<8xf16>
          %210 = vector.insert %209, %207 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %210 : vector<4x2xf16>
        }
        %157 = vector.extract %156[0, 0] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<() -> (1)>()
        %160 = vector.extract %156[0, 1] : f16 from vector<4x2xf16>
        %161 = vector.insert %160, %158 [%155, %159] : f16 into vector<32x2xf16>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %163 = vector.extract %156[1, 0] : f16 from vector<4x2xf16>
        %164 = vector.insert %163, %161 [%162, %c0] : f16 into vector<32x2xf16>
        %165 = affine.apply affine_map<() -> (1)>()
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %167 = vector.extract %156[1, 1] : f16 from vector<4x2xf16>
        %168 = vector.insert %167, %164 [%166, %165] : f16 into vector<32x2xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %170 = vector.extract %156[2, 0] : f16 from vector<4x2xf16>
        %171 = vector.insert %170, %168 [%169, %c0] : f16 into vector<32x2xf16>
        %172 = affine.apply affine_map<() -> (1)>()
        %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %174 = vector.extract %156[2, 1] : f16 from vector<4x2xf16>
        %175 = vector.insert %174, %171 [%173, %172] : f16 into vector<32x2xf16>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %177 = vector.extract %156[3, 0] : f16 from vector<4x2xf16>
        %178 = vector.insert %177, %175 [%176, %c0] : f16 into vector<32x2xf16>
        %179 = affine.apply affine_map<() -> (1)>()
        %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %181 = vector.extract %156[3, 1] : f16 from vector<4x2xf16>
        %182 = vector.insert %181, %178 [%180, %179] : f16 into vector<32x2xf16>
        affine.yield %182 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_64 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_65 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %89 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_64) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_65) -> (vector<1x4xf32>) {
          %169 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %172 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
            %173 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %174 = vector.extract %86[%172, %173] : f16 from vector<2x32xf16>
            %175 = vector.insert %174, %cst_119 [0, 0] : f16 into vector<1x4xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %177 = vector.extract %86[%172, %176] : f16 from vector<2x32xf16>
            %178 = vector.insert %177, %175 [0, 1] : f16 into vector<1x4xf16>
            %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %180 = vector.extract %86[%172, %179] : f16 from vector<2x32xf16>
            %181 = vector.insert %180, %178 [0, 2] : f16 into vector<1x4xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %183 = vector.extract %86[%172, %182] : f16 from vector<2x32xf16>
            %184 = vector.insert %183, %181 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %184 : vector<1x4xf16>
          }
          %170 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %172 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %173 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %cst_119 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %174 = vector.extract %88[%172, %173] : f16 from vector<32x2xf16>
            %175 = vector.insert %174, %cst_119 [0, 0] : f16 into vector<4x1xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %177 = vector.extract %88[%176, %173] : f16 from vector<32x2xf16>
            %178 = vector.insert %177, %175 [1, 0] : f16 into vector<4x1xf16>
            %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %180 = vector.extract %88[%179, %173] : f16 from vector<32x2xf16>
            %181 = vector.insert %180, %178 [2, 0] : f16 into vector<4x1xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %183 = vector.extract %88[%182, %173] : f16 from vector<32x2xf16>
            %184 = vector.insert %183, %181 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %184 : vector<4x1xf16>
          }
          %171 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %172 = frisk.warp_mma_rr %169, %170, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %172 : vector<1x4xf32>
          }
          affine.yield %171 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %thread_id_x_66 = gpu.thread_id x
      %90 = arith.index_cast %thread_id_x_66 : index to i32
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %c16_i32 = arith.constant 16 : i32
      %91 = arith.andi %90, %c16_i32 : i32
      %92 = arith.cmpi ne, %91, %c0_i32 : i32
      %c32_i32 = arith.constant 32 : i32
      %93 = arith.andi %90, %c32_i32 : i32
      %94 = arith.cmpi ne, %93, %c0_i32 : i32
      %cst_67 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %95 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_67) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %89[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %89[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %89[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %89[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %180 = vector.extract %179[0, 0] : f32 from vector<1x4xf32>
          %181 = vector.extract %179[0, 1] : f32 from vector<1x4xf32>
          %182 = vector.extract %179[0, 2] : f32 from vector<1x4xf32>
          %183 = vector.extract %179[0, 3] : f32 from vector<1x4xf32>
          %184 = arith.select %92, %180, %181 : f32
          %shuffleResult, %valid = gpu.shuffle xor %184, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %185 = arith.select %92, %shuffleResult, %180 : f32
          %186 = arith.select %92, %181, %shuffleResult : f32
          %187 = arith.select %92, %182, %183 : f32
          %shuffleResult_120, %valid_121 = gpu.shuffle xor %187, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %188 = arith.select %92, %shuffleResult_120, %182 : f32
          %189 = arith.select %92, %183, %shuffleResult_120 : f32
          %190 = arith.select %94, %185, %188 : f32
          %shuffleResult_122, %valid_123 = gpu.shuffle xor %190, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %191 = arith.select %94, %shuffleResult_122, %185 : f32
          %192 = arith.select %94, %188, %shuffleResult_122 : f32
          %193 = arith.select %94, %186, %189 : f32
          %shuffleResult_124, %valid_125 = gpu.shuffle xor %193, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %194 = arith.select %94, %shuffleResult_124, %186 : f32
          %195 = arith.select %94, %189, %shuffleResult_124 : f32
          %196 = vector.insert %191, %179 [0, 0] : f32 into vector<1x4xf32>
          %197 = vector.insert %194, %196 [0, 1] : f32 into vector<1x4xf32>
          %198 = vector.insert %192, %197 [0, 2] : f32 into vector<1x4xf32>
          %199 = vector.insert %195, %198 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %199 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %96 = frisk.to_threadTile %arg5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %97 = frisk.to_threadTile %79 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_68 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %98 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_68) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %169 = vector.extract %97[%156, %c0] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %c0_120 = arith.constant 0 : index
          %171 = vector.extract %97[%156, %c0_120] : f32 from vector<2x1xf32>
          %172 = vector.insert %171, %170 [0, 1] : f32 into vector<1x4xf32>
          %c0_121 = arith.constant 0 : index
          %173 = vector.extract %97[%156, %c0_121] : f32 from vector<2x1xf32>
          %174 = vector.insert %173, %172 [0, 2] : f32 into vector<1x4xf32>
          %c0_122 = arith.constant 0 : index
          %175 = vector.extract %97[%156, %c0_122] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %174 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %176 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_69 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %99 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_69) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %96[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %96[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %96[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %96[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %98[%156, %155] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %183 = vector.extract %98[%156, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %186 = vector.extract %98[%156, %185] : f32 from vector<2x8xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %189 = vector.extract %98[%156, %188] : f32 from vector<2x8xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.mulf %179, %190 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_70 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %100 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_70) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %99[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %99[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %99[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %99[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %95[%156, %155] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %183 = vector.extract %95[%156, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %186 = vector.extract %95[%156, %185] : f32 from vector<2x8xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %189 = vector.extract %95[%156, %188] : f32 from vector<2x8xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.addf %179, %190 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_71 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %101 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %101 to %alloc_71 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_71 : memref<128x32xf16, 3>
      %thread_id_x_72 = gpu.thread_id x
      %cst_73 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %102 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_73) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %183 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4)>(%arg11)[%thread_id_x_72]
          %184 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_72]
          %185 = frisk.read_matrix_from_shm %alloc_71[%183, %184] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %186 = vector.bitcast %185 : vector<4xi32> to vector<8xf16>
          %cst_119 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %187 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_72]
          %188 = vector.extract %186[%187] : f16 from vector<8xf16>
          %189 = vector.insert %188, %cst_119 [0, 0] : f16 into vector<4x2xf16>
          %190 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_72]
          %191 = vector.extract %186[%190] : f16 from vector<8xf16>
          %192 = vector.insert %191, %189 [0, 1] : f16 into vector<4x2xf16>
          %193 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_72]
          %194 = vector.extract %186[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %192 [1, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_72]
          %197 = vector.extract %186[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [1, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_72]
          %200 = vector.extract %186[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [2, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_72]
          %203 = vector.extract %186[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [2, 1] : f16 into vector<4x2xf16>
          %205 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_72]
          %206 = vector.extract %186[%205] : f16 from vector<8xf16>
          %207 = vector.insert %206, %204 [3, 0] : f16 into vector<4x2xf16>
          %208 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_72]
          %209 = vector.extract %186[%208] : f16 from vector<8xf16>
          %210 = vector.insert %209, %207 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %210 : vector<4x2xf16>
        }
        %157 = vector.extract %156[0, 0] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<() -> (1)>()
        %160 = vector.extract %156[0, 1] : f16 from vector<4x2xf16>
        %161 = vector.insert %160, %158 [%155, %159] : f16 into vector<32x2xf16>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %163 = vector.extract %156[1, 0] : f16 from vector<4x2xf16>
        %164 = vector.insert %163, %161 [%162, %c0] : f16 into vector<32x2xf16>
        %165 = affine.apply affine_map<() -> (1)>()
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %167 = vector.extract %156[1, 1] : f16 from vector<4x2xf16>
        %168 = vector.insert %167, %164 [%166, %165] : f16 into vector<32x2xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %170 = vector.extract %156[2, 0] : f16 from vector<4x2xf16>
        %171 = vector.insert %170, %168 [%169, %c0] : f16 into vector<32x2xf16>
        %172 = affine.apply affine_map<() -> (1)>()
        %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %174 = vector.extract %156[2, 1] : f16 from vector<4x2xf16>
        %175 = vector.insert %174, %171 [%173, %172] : f16 into vector<32x2xf16>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %177 = vector.extract %156[3, 0] : f16 from vector<4x2xf16>
        %178 = vector.insert %177, %175 [%176, %c0] : f16 into vector<32x2xf16>
        %179 = affine.apply affine_map<() -> (1)>()
        %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %181 = vector.extract %156[3, 1] : f16 from vector<4x2xf16>
        %182 = vector.insert %181, %178 [%180, %179] : f16 into vector<32x2xf16>
        affine.yield %182 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_74 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_75 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %103 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_74) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_75) -> (vector<1x4xf32>) {
          %169 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %172 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
            %173 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %174 = vector.extract %86[%172, %173] : f16 from vector<2x32xf16>
            %175 = vector.insert %174, %cst_119 [0, 0] : f16 into vector<1x4xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %177 = vector.extract %86[%172, %176] : f16 from vector<2x32xf16>
            %178 = vector.insert %177, %175 [0, 1] : f16 into vector<1x4xf16>
            %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %180 = vector.extract %86[%172, %179] : f16 from vector<2x32xf16>
            %181 = vector.insert %180, %178 [0, 2] : f16 into vector<1x4xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %183 = vector.extract %86[%172, %182] : f16 from vector<2x32xf16>
            %184 = vector.insert %183, %181 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %184 : vector<1x4xf16>
          }
          %170 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %172 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %173 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %cst_119 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %174 = vector.extract %102[%172, %173] : f16 from vector<32x2xf16>
            %175 = vector.insert %174, %cst_119 [0, 0] : f16 into vector<4x1xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %177 = vector.extract %102[%176, %173] : f16 from vector<32x2xf16>
            %178 = vector.insert %177, %175 [1, 0] : f16 into vector<4x1xf16>
            %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %180 = vector.extract %102[%179, %173] : f16 from vector<32x2xf16>
            %181 = vector.insert %180, %178 [2, 0] : f16 into vector<4x1xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %183 = vector.extract %102[%182, %173] : f16 from vector<32x2xf16>
            %184 = vector.insert %183, %181 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %184 : vector<4x1xf16>
          }
          %171 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %172 = frisk.warp_mma_rr %169, %170, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %172 : vector<1x4xf32>
          }
          affine.yield %171 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %thread_id_x_76 = gpu.thread_id x
      %104 = arith.index_cast %thread_id_x_76 : index to i32
      %c0_i32_77 = arith.constant 0 : i32
      %c64_i32_78 = arith.constant 64 : i32
      %c16_i32_79 = arith.constant 16 : i32
      %105 = arith.andi %104, %c16_i32_79 : i32
      %106 = arith.cmpi ne, %105, %c0_i32_77 : i32
      %c32_i32_80 = arith.constant 32 : i32
      %107 = arith.andi %104, %c32_i32_80 : i32
      %108 = arith.cmpi ne, %107, %c0_i32_77 : i32
      %cst_81 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %109 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_81) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %103[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %103[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %103[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %103[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %180 = vector.extract %179[0, 0] : f32 from vector<1x4xf32>
          %181 = vector.extract %179[0, 1] : f32 from vector<1x4xf32>
          %182 = vector.extract %179[0, 2] : f32 from vector<1x4xf32>
          %183 = vector.extract %179[0, 3] : f32 from vector<1x4xf32>
          %184 = arith.select %106, %180, %181 : f32
          %shuffleResult, %valid = gpu.shuffle xor %184, %c16_i32_79, %c64_i32_78 {frisk.layout_role = "wmmaC"} : f32
          %185 = arith.select %106, %shuffleResult, %180 : f32
          %186 = arith.select %106, %181, %shuffleResult : f32
          %187 = arith.select %106, %182, %183 : f32
          %shuffleResult_120, %valid_121 = gpu.shuffle xor %187, %c16_i32_79, %c64_i32_78 {frisk.layout_role = "wmmaC"} : f32
          %188 = arith.select %106, %shuffleResult_120, %182 : f32
          %189 = arith.select %106, %183, %shuffleResult_120 : f32
          %190 = arith.select %108, %185, %188 : f32
          %shuffleResult_122, %valid_123 = gpu.shuffle xor %190, %c32_i32_80, %c64_i32_78 {frisk.layout_role = "wmmaC"} : f32
          %191 = arith.select %108, %shuffleResult_122, %185 : f32
          %192 = arith.select %108, %188, %shuffleResult_122 : f32
          %193 = arith.select %108, %186, %189 : f32
          %shuffleResult_124, %valid_125 = gpu.shuffle xor %193, %c32_i32_80, %c64_i32_78 {frisk.layout_role = "wmmaC"} : f32
          %194 = arith.select %108, %shuffleResult_124, %186 : f32
          %195 = arith.select %108, %189, %shuffleResult_124 : f32
          %196 = vector.insert %191, %179 [0, 0] : f32 into vector<1x4xf32>
          %197 = vector.insert %194, %196 [0, 1] : f32 into vector<1x4xf32>
          %198 = vector.insert %192, %197 [0, 2] : f32 into vector<1x4xf32>
          %199 = vector.insert %195, %198 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %199 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %110 = frisk.to_threadTile %arg8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %111 = frisk.to_threadTile %79 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_82 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %112 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_82) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %169 = vector.extract %111[%156, %c0] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %c0_120 = arith.constant 0 : index
          %171 = vector.extract %111[%156, %c0_120] : f32 from vector<2x1xf32>
          %172 = vector.insert %171, %170 [0, 1] : f32 into vector<1x4xf32>
          %c0_121 = arith.constant 0 : index
          %173 = vector.extract %111[%156, %c0_121] : f32 from vector<2x1xf32>
          %174 = vector.insert %173, %172 [0, 2] : f32 into vector<1x4xf32>
          %c0_122 = arith.constant 0 : index
          %175 = vector.extract %111[%156, %c0_122] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %174 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %176 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_83 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %113 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_83) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %110[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %110[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %110[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %110[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %112[%156, %155] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %183 = vector.extract %112[%156, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %186 = vector.extract %112[%156, %185] : f32 from vector<2x8xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %189 = vector.extract %112[%156, %188] : f32 from vector<2x8xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.mulf %179, %190 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_84 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %114 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_84) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %113[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %113[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %113[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %113[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %109[%156, %155] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %183 = vector.extract %109[%156, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %186 = vector.extract %109[%156, %185] : f32 from vector<2x8xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %189 = vector.extract %109[%156, %188] : f32 from vector<2x8xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.addf %179, %190 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_85 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %115 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %115 to %alloc_85 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_85 : memref<128x32xf16, 3>
      %thread_id_x_86 = gpu.thread_id x
      %cst_87 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %116 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_87) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %183 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4)>(%arg11)[%thread_id_x_86]
          %184 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_86]
          %185 = frisk.read_matrix_from_shm %alloc_85[%183, %184] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %186 = vector.bitcast %185 : vector<4xi32> to vector<8xf16>
          %cst_119 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %187 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_86]
          %188 = vector.extract %186[%187] : f16 from vector<8xf16>
          %189 = vector.insert %188, %cst_119 [0, 0] : f16 into vector<4x2xf16>
          %190 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_86]
          %191 = vector.extract %186[%190] : f16 from vector<8xf16>
          %192 = vector.insert %191, %189 [0, 1] : f16 into vector<4x2xf16>
          %193 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_86]
          %194 = vector.extract %186[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %192 [1, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_86]
          %197 = vector.extract %186[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [1, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_86]
          %200 = vector.extract %186[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [2, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_86]
          %203 = vector.extract %186[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [2, 1] : f16 into vector<4x2xf16>
          %205 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_86]
          %206 = vector.extract %186[%205] : f16 from vector<8xf16>
          %207 = vector.insert %206, %204 [3, 0] : f16 into vector<4x2xf16>
          %208 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_86]
          %209 = vector.extract %186[%208] : f16 from vector<8xf16>
          %210 = vector.insert %209, %207 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %210 : vector<4x2xf16>
        }
        %157 = vector.extract %156[0, 0] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<() -> (1)>()
        %160 = vector.extract %156[0, 1] : f16 from vector<4x2xf16>
        %161 = vector.insert %160, %158 [%155, %159] : f16 into vector<32x2xf16>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %163 = vector.extract %156[1, 0] : f16 from vector<4x2xf16>
        %164 = vector.insert %163, %161 [%162, %c0] : f16 into vector<32x2xf16>
        %165 = affine.apply affine_map<() -> (1)>()
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %167 = vector.extract %156[1, 1] : f16 from vector<4x2xf16>
        %168 = vector.insert %167, %164 [%166, %165] : f16 into vector<32x2xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %170 = vector.extract %156[2, 0] : f16 from vector<4x2xf16>
        %171 = vector.insert %170, %168 [%169, %c0] : f16 into vector<32x2xf16>
        %172 = affine.apply affine_map<() -> (1)>()
        %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %174 = vector.extract %156[2, 1] : f16 from vector<4x2xf16>
        %175 = vector.insert %174, %171 [%173, %172] : f16 into vector<32x2xf16>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %177 = vector.extract %156[3, 0] : f16 from vector<4x2xf16>
        %178 = vector.insert %177, %175 [%176, %c0] : f16 into vector<32x2xf16>
        %179 = affine.apply affine_map<() -> (1)>()
        %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %181 = vector.extract %156[3, 1] : f16 from vector<4x2xf16>
        %182 = vector.insert %181, %178 [%180, %179] : f16 into vector<32x2xf16>
        affine.yield %182 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_88 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_89 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %117 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_88) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_89) -> (vector<1x4xf32>) {
          %169 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %172 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
            %173 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %174 = vector.extract %86[%172, %173] : f16 from vector<2x32xf16>
            %175 = vector.insert %174, %cst_119 [0, 0] : f16 into vector<1x4xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %177 = vector.extract %86[%172, %176] : f16 from vector<2x32xf16>
            %178 = vector.insert %177, %175 [0, 1] : f16 into vector<1x4xf16>
            %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %180 = vector.extract %86[%172, %179] : f16 from vector<2x32xf16>
            %181 = vector.insert %180, %178 [0, 2] : f16 into vector<1x4xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %183 = vector.extract %86[%172, %182] : f16 from vector<2x32xf16>
            %184 = vector.insert %183, %181 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %184 : vector<1x4xf16>
          }
          %170 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %172 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %173 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %cst_119 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %174 = vector.extract %116[%172, %173] : f16 from vector<32x2xf16>
            %175 = vector.insert %174, %cst_119 [0, 0] : f16 into vector<4x1xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %177 = vector.extract %116[%176, %173] : f16 from vector<32x2xf16>
            %178 = vector.insert %177, %175 [1, 0] : f16 into vector<4x1xf16>
            %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %180 = vector.extract %116[%179, %173] : f16 from vector<32x2xf16>
            %181 = vector.insert %180, %178 [2, 0] : f16 into vector<4x1xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %183 = vector.extract %116[%182, %173] : f16 from vector<32x2xf16>
            %184 = vector.insert %183, %181 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %184 : vector<4x1xf16>
          }
          %171 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %172 = frisk.warp_mma_rr %169, %170, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %172 : vector<1x4xf32>
          }
          affine.yield %171 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %thread_id_x_90 = gpu.thread_id x
      %118 = arith.index_cast %thread_id_x_90 : index to i32
      %c0_i32_91 = arith.constant 0 : i32
      %c64_i32_92 = arith.constant 64 : i32
      %c16_i32_93 = arith.constant 16 : i32
      %119 = arith.andi %118, %c16_i32_93 : i32
      %120 = arith.cmpi ne, %119, %c0_i32_91 : i32
      %c32_i32_94 = arith.constant 32 : i32
      %121 = arith.andi %118, %c32_i32_94 : i32
      %122 = arith.cmpi ne, %121, %c0_i32_91 : i32
      %cst_95 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %123 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_95) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %117[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %117[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %117[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %117[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %180 = vector.extract %179[0, 0] : f32 from vector<1x4xf32>
          %181 = vector.extract %179[0, 1] : f32 from vector<1x4xf32>
          %182 = vector.extract %179[0, 2] : f32 from vector<1x4xf32>
          %183 = vector.extract %179[0, 3] : f32 from vector<1x4xf32>
          %184 = arith.select %120, %180, %181 : f32
          %shuffleResult, %valid = gpu.shuffle xor %184, %c16_i32_93, %c64_i32_92 {frisk.layout_role = "wmmaC"} : f32
          %185 = arith.select %120, %shuffleResult, %180 : f32
          %186 = arith.select %120, %181, %shuffleResult : f32
          %187 = arith.select %120, %182, %183 : f32
          %shuffleResult_120, %valid_121 = gpu.shuffle xor %187, %c16_i32_93, %c64_i32_92 {frisk.layout_role = "wmmaC"} : f32
          %188 = arith.select %120, %shuffleResult_120, %182 : f32
          %189 = arith.select %120, %183, %shuffleResult_120 : f32
          %190 = arith.select %122, %185, %188 : f32
          %shuffleResult_122, %valid_123 = gpu.shuffle xor %190, %c32_i32_94, %c64_i32_92 {frisk.layout_role = "wmmaC"} : f32
          %191 = arith.select %122, %shuffleResult_122, %185 : f32
          %192 = arith.select %122, %188, %shuffleResult_122 : f32
          %193 = arith.select %122, %186, %189 : f32
          %shuffleResult_124, %valid_125 = gpu.shuffle xor %193, %c32_i32_94, %c64_i32_92 {frisk.layout_role = "wmmaC"} : f32
          %194 = arith.select %122, %shuffleResult_124, %186 : f32
          %195 = arith.select %122, %189, %shuffleResult_124 : f32
          %196 = vector.insert %191, %179 [0, 0] : f32 into vector<1x4xf32>
          %197 = vector.insert %194, %196 [0, 1] : f32 into vector<1x4xf32>
          %198 = vector.insert %192, %197 [0, 2] : f32 into vector<1x4xf32>
          %199 = vector.insert %195, %198 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %199 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %124 = frisk.to_threadTile %arg9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %125 = frisk.to_threadTile %79 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_96 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %126 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_96) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %169 = vector.extract %125[%156, %c0] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %c0_120 = arith.constant 0 : index
          %171 = vector.extract %125[%156, %c0_120] : f32 from vector<2x1xf32>
          %172 = vector.insert %171, %170 [0, 1] : f32 into vector<1x4xf32>
          %c0_121 = arith.constant 0 : index
          %173 = vector.extract %125[%156, %c0_121] : f32 from vector<2x1xf32>
          %174 = vector.insert %173, %172 [0, 2] : f32 into vector<1x4xf32>
          %c0_122 = arith.constant 0 : index
          %175 = vector.extract %125[%156, %c0_122] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %174 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %176 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_97 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %127 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_97) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %124[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %124[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %124[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %124[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %126[%156, %155] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %183 = vector.extract %126[%156, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %186 = vector.extract %126[%156, %185] : f32 from vector<2x8xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %189 = vector.extract %126[%156, %188] : f32 from vector<2x8xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.mulf %179, %190 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_98 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %128 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_98) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %127[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %127[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %127[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %127[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %123[%156, %155] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %183 = vector.extract %123[%156, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %186 = vector.extract %123[%156, %185] : f32 from vector<2x8xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %189 = vector.extract %123[%156, %188] : f32 from vector<2x8xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.addf %179, %190 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_99 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %129 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %129 to %alloc_99 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_99 : memref<128x32xf16, 3>
      %thread_id_x_100 = gpu.thread_id x
      %cst_101 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %130 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_101) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %156 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %183 = affine.apply affine_map<(d0)[s0] -> (d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4)>(%arg11)[%thread_id_x_100]
          %184 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_100]
          %185 = frisk.read_matrix_from_shm %alloc_99[%183, %184] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %186 = vector.bitcast %185 : vector<4xi32> to vector<8xf16>
          %cst_119 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %187 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_100]
          %188 = vector.extract %186[%187] : f16 from vector<8xf16>
          %189 = vector.insert %188, %cst_119 [0, 0] : f16 into vector<4x2xf16>
          %190 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_100]
          %191 = vector.extract %186[%190] : f16 from vector<8xf16>
          %192 = vector.insert %191, %189 [0, 1] : f16 into vector<4x2xf16>
          %193 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_100]
          %194 = vector.extract %186[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %192 [1, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_100]
          %197 = vector.extract %186[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [1, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_100]
          %200 = vector.extract %186[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [2, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_100]
          %203 = vector.extract %186[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [2, 1] : f16 into vector<4x2xf16>
          %205 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_100]
          %206 = vector.extract %186[%205] : f16 from vector<8xf16>
          %207 = vector.insert %206, %204 [3, 0] : f16 into vector<4x2xf16>
          %208 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_100]
          %209 = vector.extract %186[%208] : f16 from vector<8xf16>
          %210 = vector.insert %209, %207 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %210 : vector<4x2xf16>
        }
        %157 = vector.extract %156[0, 0] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<() -> (1)>()
        %160 = vector.extract %156[0, 1] : f16 from vector<4x2xf16>
        %161 = vector.insert %160, %158 [%155, %159] : f16 into vector<32x2xf16>
        %162 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %163 = vector.extract %156[1, 0] : f16 from vector<4x2xf16>
        %164 = vector.insert %163, %161 [%162, %c0] : f16 into vector<32x2xf16>
        %165 = affine.apply affine_map<() -> (1)>()
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %167 = vector.extract %156[1, 1] : f16 from vector<4x2xf16>
        %168 = vector.insert %167, %164 [%166, %165] : f16 into vector<32x2xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %170 = vector.extract %156[2, 0] : f16 from vector<4x2xf16>
        %171 = vector.insert %170, %168 [%169, %c0] : f16 into vector<32x2xf16>
        %172 = affine.apply affine_map<() -> (1)>()
        %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %174 = vector.extract %156[2, 1] : f16 from vector<4x2xf16>
        %175 = vector.insert %174, %171 [%173, %172] : f16 into vector<32x2xf16>
        %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %177 = vector.extract %156[3, 0] : f16 from vector<4x2xf16>
        %178 = vector.insert %177, %175 [%176, %c0] : f16 into vector<32x2xf16>
        %179 = affine.apply affine_map<() -> (1)>()
        %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %181 = vector.extract %156[3, 1] : f16 from vector<4x2xf16>
        %182 = vector.insert %181, %178 [%180, %179] : f16 into vector<32x2xf16>
        affine.yield %182 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_102 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_103 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %131 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_102) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_103) -> (vector<1x4xf32>) {
          %169 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %172 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
            %173 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %174 = vector.extract %86[%172, %173] : f16 from vector<2x32xf16>
            %175 = vector.insert %174, %cst_119 [0, 0] : f16 into vector<1x4xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %177 = vector.extract %86[%172, %176] : f16 from vector<2x32xf16>
            %178 = vector.insert %177, %175 [0, 1] : f16 into vector<1x4xf16>
            %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %180 = vector.extract %86[%172, %179] : f16 from vector<2x32xf16>
            %181 = vector.insert %180, %178 [0, 2] : f16 into vector<1x4xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %183 = vector.extract %86[%172, %182] : f16 from vector<2x32xf16>
            %184 = vector.insert %183, %181 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %184 : vector<1x4xf16>
          }
          %170 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %172 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %173 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %cst_119 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %174 = vector.extract %130[%172, %173] : f16 from vector<32x2xf16>
            %175 = vector.insert %174, %cst_119 [0, 0] : f16 into vector<4x1xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %177 = vector.extract %130[%176, %173] : f16 from vector<32x2xf16>
            %178 = vector.insert %177, %175 [1, 0] : f16 into vector<4x1xf16>
            %179 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %180 = vector.extract %130[%179, %173] : f16 from vector<32x2xf16>
            %181 = vector.insert %180, %178 [2, 0] : f16 into vector<4x1xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %183 = vector.extract %130[%182, %173] : f16 from vector<32x2xf16>
            %184 = vector.insert %183, %181 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %184 : vector<4x1xf16>
          }
          %171 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %172 = frisk.warp_mma_rr %169, %170, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %172 : vector<1x4xf32>
          }
          affine.yield %171 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %thread_id_x_104 = gpu.thread_id x
      %132 = arith.index_cast %thread_id_x_104 : index to i32
      %c0_i32_105 = arith.constant 0 : i32
      %c64_i32_106 = arith.constant 64 : i32
      %c16_i32_107 = arith.constant 16 : i32
      %133 = arith.andi %132, %c16_i32_107 : i32
      %134 = arith.cmpi ne, %133, %c0_i32_105 : i32
      %c32_i32_108 = arith.constant 32 : i32
      %135 = arith.andi %132, %c32_i32_108 : i32
      %136 = arith.cmpi ne, %135, %c0_i32_105 : i32
      %cst_109 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %137 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_109) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %131[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %131[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %131[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %131[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %180 = vector.extract %179[0, 0] : f32 from vector<1x4xf32>
          %181 = vector.extract %179[0, 1] : f32 from vector<1x4xf32>
          %182 = vector.extract %179[0, 2] : f32 from vector<1x4xf32>
          %183 = vector.extract %179[0, 3] : f32 from vector<1x4xf32>
          %184 = arith.select %134, %180, %181 : f32
          %shuffleResult, %valid = gpu.shuffle xor %184, %c16_i32_107, %c64_i32_106 {frisk.layout_role = "wmmaC"} : f32
          %185 = arith.select %134, %shuffleResult, %180 : f32
          %186 = arith.select %134, %181, %shuffleResult : f32
          %187 = arith.select %134, %182, %183 : f32
          %shuffleResult_120, %valid_121 = gpu.shuffle xor %187, %c16_i32_107, %c64_i32_106 {frisk.layout_role = "wmmaC"} : f32
          %188 = arith.select %134, %shuffleResult_120, %182 : f32
          %189 = arith.select %134, %183, %shuffleResult_120 : f32
          %190 = arith.select %136, %185, %188 : f32
          %shuffleResult_122, %valid_123 = gpu.shuffle xor %190, %c32_i32_108, %c64_i32_106 {frisk.layout_role = "wmmaC"} : f32
          %191 = arith.select %136, %shuffleResult_122, %185 : f32
          %192 = arith.select %136, %188, %shuffleResult_122 : f32
          %193 = arith.select %136, %186, %189 : f32
          %shuffleResult_124, %valid_125 = gpu.shuffle xor %193, %c32_i32_108, %c64_i32_106 {frisk.layout_role = "wmmaC"} : f32
          %194 = arith.select %136, %shuffleResult_124, %186 : f32
          %195 = arith.select %136, %189, %shuffleResult_124 : f32
          %196 = vector.insert %191, %179 [0, 0] : f32 into vector<1x4xf32>
          %197 = vector.insert %194, %196 [0, 1] : f32 into vector<1x4xf32>
          %198 = vector.insert %192, %197 [0, 2] : f32 into vector<1x4xf32>
          %199 = vector.insert %195, %198 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %199 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %138 = frisk.to_threadTile %arg10 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %139 = frisk.to_threadTile %79 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_110 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %140 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_110) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %169 = vector.extract %139[%156, %c0] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %c0_120 = arith.constant 0 : index
          %171 = vector.extract %139[%156, %c0_120] : f32 from vector<2x1xf32>
          %172 = vector.insert %171, %170 [0, 1] : f32 into vector<1x4xf32>
          %c0_121 = arith.constant 0 : index
          %173 = vector.extract %139[%156, %c0_121] : f32 from vector<2x1xf32>
          %174 = vector.insert %173, %172 [0, 2] : f32 into vector<1x4xf32>
          %c0_122 = arith.constant 0 : index
          %175 = vector.extract %139[%156, %c0_122] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %174 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %176 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_111 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %141 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_111) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %138[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %138[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %138[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %138[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %140[%156, %155] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %183 = vector.extract %140[%156, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %186 = vector.extract %140[%156, %185] : f32 from vector<2x8xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %189 = vector.extract %140[%156, %188] : f32 from vector<2x8xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.mulf %179, %190 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_112 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %142 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_112) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %141[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %141[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %141[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %141[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] : f32 into vector<1x4xf32>
          %cst_120 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %180 = vector.extract %137[%156, %155] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %cst_120 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %183 = vector.extract %137[%156, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 1] : f32 into vector<1x4xf32>
          %185 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %186 = vector.extract %137[%156, %185] : f32 from vector<2x8xf32>
          %187 = vector.insert %186, %184 [0, 2] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %189 = vector.extract %137[%156, %188] : f32 from vector<2x8xf32>
          %190 = vector.insert %189, %187 [0, 3] : f32 into vector<1x4xf32>
          %191 = arith.addf %179, %190 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %191 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_113 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %143 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_113) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %100[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %100[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %100[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %100[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %179 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %144 = frisk.from_threadTile %143 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_114 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %145 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_114) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %156 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %159 = vector.extract %85[%155, %c0] : f32 from vector<2x1xf32>
          %160 = vector.insert %159, %cst_119 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
          frisk.fragment_yield %160 : vector<1x1xf32>
        }
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f32 into vector<2x1xf32>
        affine.yield %158 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %146 = frisk.from_threadTile %145 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_115 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %147 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_115) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %155 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %156 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %159 = vector.extract %73[%155, %c0] : f32 from vector<2x1xf32>
          %160 = vector.insert %159, %cst_119 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
          frisk.fragment_yield %160 : vector<1x1xf32>
        }
        %157 = vector.extract %156[0, 0] : f32 from vector<1x1xf32>
        %158 = vector.insert %157, %arg12 [%155, %c0] : f32 into vector<2x1xf32>
        affine.yield %158 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %148 = frisk.from_threadTile %147 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_116 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %149 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_116) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %114[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %114[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %114[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %114[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %179 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %150 = frisk.from_threadTile %149 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_117 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %151 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_117) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %128[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %128[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %128[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %128[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %179 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %152 = frisk.from_threadTile %151 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_118 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %153 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_118) -> (vector<2x8xf32>) {
        %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %156 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %157 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_119 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %169 = vector.extract %142[%156, %155] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %cst_119 [0, 0] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %172 = vector.extract %142[%156, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 1] : f32 into vector<1x4xf32>
          %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %175 = vector.extract %142[%156, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 2] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %178 = vector.extract %142[%156, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %179 : vector<1x4xf32>
        }
        %158 = vector.extract %157[0, 0] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %arg12 [%156, %155] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %161 = vector.extract %157[0, 1] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%156, %160] : f32 into vector<2x8xf32>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %164 = vector.extract %157[0, 2] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %162 [%156, %163] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %167 = vector.extract %157[0, 3] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%156, %166] : f32 into vector<2x8xf32>
        affine.yield %168 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %154 = frisk.from_threadTile %153 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      affine.yield %144, %146, %148, %150, %152, %154 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %17 = frisk.to_threadTile %16#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %18 = frisk.to_threadTile %16#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_13 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %19 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_13) -> (vector<2x8xf32>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %67 = vector.extract %18[%54, %c0] : f32 from vector<2x1xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %c0_37 = arith.constant 0 : index
        %69 = vector.extract %18[%54, %c0_37] : f32 from vector<2x1xf32>
        %70 = vector.insert %69, %68 [0, 1] : f32 into vector<1x4xf32>
        %c0_38 = arith.constant 0 : index
        %71 = vector.extract %18[%54, %c0_38] : f32 from vector<2x1xf32>
        %72 = vector.insert %71, %70 [0, 2] : f32 into vector<1x4xf32>
        %c0_39 = arith.constant 0 : index
        %73 = vector.extract %18[%54, %c0_39] : f32 from vector<2x1xf32>
        %74 = vector.insert %73, %72 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %74 : vector<1x4xf32>
      }
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg5 [%54, %53] : f32 into vector<2x8xf32>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%54, %58] : f32 into vector<2x8xf32>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %60 [%54, %61] : f32 into vector<2x8xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%54, %64] : f32 into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_14 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %20 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_14) -> (vector<2x8xf32>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %67 = vector.extract %17[%54, %53] : f32 from vector<2x8xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %17[%54, %69] : f32 from vector<2x8xf32>
        %71 = vector.insert %70, %68 [0, 1] : f32 into vector<1x4xf32>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %17[%54, %72] : f32 from vector<2x8xf32>
        %74 = vector.insert %73, %71 [0, 2] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %17[%54, %75] : f32 from vector<2x8xf32>
        %77 = vector.insert %76, %74 [0, 3] : f32 into vector<1x4xf32>
        %cst_37 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %78 = vector.extract %19[%54, %53] : f32 from vector<2x8xf32>
        %79 = vector.insert %78, %cst_37 [0, 0] : f32 into vector<1x4xf32>
        %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %81 = vector.extract %19[%54, %80] : f32 from vector<2x8xf32>
        %82 = vector.insert %81, %79 [0, 1] : f32 into vector<1x4xf32>
        %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %84 = vector.extract %19[%54, %83] : f32 from vector<2x8xf32>
        %85 = vector.insert %84, %82 [0, 2] : f32 into vector<1x4xf32>
        %86 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %87 = vector.extract %19[%54, %86] : f32 from vector<2x8xf32>
        %88 = vector.insert %87, %85 [0, 3] : f32 into vector<1x4xf32>
        %89 = arith.divf %77, %88 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %89 : vector<1x4xf32>
      }
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg5 [%54, %53] : f32 into vector<2x8xf32>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%54, %58] : f32 into vector<2x8xf32>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %60 [%54, %61] : f32 into vector<2x8xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%54, %64] : f32 into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_15 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %21 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_15) -> (vector<2x8xf16>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %67 = vector.extract %20[%54, %53] : f32 from vector<2x8xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %20[%54, %69] : f32 from vector<2x8xf32>
        %71 = vector.insert %70, %68 [0, 1] : f32 into vector<1x4xf32>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %20[%54, %72] : f32 from vector<2x8xf32>
        %74 = vector.insert %73, %71 [0, 2] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %20[%54, %75] : f32 from vector<2x8xf32>
        %77 = vector.insert %76, %74 [0, 3] : f32 into vector<1x4xf32>
        %78 = arith.truncf %77 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %78 : vector<1x4xf16>
      }
      %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %arg5 [%54, %53] : f16 into vector<2x8xf16>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%54, %58] : f16 into vector<2x8xf16>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %60 [%54, %61] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%54, %64] : f16 into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %22 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%8 * 16384) floordiv 524288) mod 32, (%8 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_16 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %23 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_16) -> (vector<2x8xf16>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %67 = vector.extract %21[%54, %53] : f16 from vector<2x8xf16>
        %68 = vector.insert %67, %cst_36 [0, 0] : f16 into vector<1x4xf16>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %21[%54, %69] : f16 from vector<2x8xf16>
        %71 = vector.insert %70, %68 [0, 1] : f16 into vector<1x4xf16>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %21[%54, %72] : f16 from vector<2x8xf16>
        %74 = vector.insert %73, %71 [0, 2] : f16 into vector<1x4xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %21[%54, %75] : f16 from vector<2x8xf16>
        %77 = vector.insert %76, %74 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %77 : vector<1x4xf16>
      }
      %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %arg5 [%54, %53] : f16 into vector<2x8xf16>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%54, %58] : f16 into vector<2x8xf16>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %60 [%54, %61] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%54, %64] : f16 into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x = gpu.thread_id x
    %24 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x)
    %c4_17 = arith.constant 4 : index
    %25 = arith.cmpi ult, %24, %c4_17 : index
    scf.if %25 {
      affine.for %arg4 = 0 to 4 {
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %55 = vector.extract %23[%54, %53] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_36 = arith.constant 0 : index
          %56 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%53)
          %c0_37 = arith.constant 0 : index
          %57 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%53)
          %58 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%57)
          %c0_38 = arith.constant 0 : index
          %59 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_36, %58)
          %60 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x, %54, %56, %c0, %c0_37, %c0_38, %59)
          %61 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x, %54, %56, %c0, %c0_37, %c0_38, %59)
          affine.store %55, %22[%60, %61] {tiled = true} : memref<128x32xf16, 1>
          %62 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %63 = vector.extract %23[%54, %62] : f16 from vector<2x8xf16>
          %c0_39 = arith.constant 0 : index
          %c0_40 = arith.constant 0 : index
          %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%62)
          %c0_41 = arith.constant 0 : index
          %65 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%62)
          %66 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%65)
          %c0_42 = arith.constant 0 : index
          %67 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_40, %66)
          %68 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x, %54, %64, %c0_39, %c0_41, %c0_42, %67)
          %69 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x, %54, %64, %c0_39, %c0_41, %c0_42, %67)
          affine.store %63, %22[%68, %69] {tiled = true} : memref<128x32xf16, 1>
          %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %71 = vector.extract %23[%54, %70] : f16 from vector<2x8xf16>
          %c0_43 = arith.constant 0 : index
          %c0_44 = arith.constant 0 : index
          %72 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%70)
          %c0_45 = arith.constant 0 : index
          %73 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%70)
          %74 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%73)
          %c0_46 = arith.constant 0 : index
          %75 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_44, %74)
          %76 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x, %54, %72, %c0_43, %c0_45, %c0_46, %75)
          %77 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x, %54, %72, %c0_43, %c0_45, %c0_46, %75)
          affine.store %71, %22[%76, %77] {tiled = true} : memref<128x32xf16, 1>
          %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %79 = vector.extract %23[%54, %78] : f16 from vector<2x8xf16>
          %c0_47 = arith.constant 0 : index
          %c0_48 = arith.constant 0 : index
          %80 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%78)
          %c0_49 = arith.constant 0 : index
          %81 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%78)
          %82 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%81)
          %c0_50 = arith.constant 0 : index
          %83 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_48, %82)
          %84 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x, %54, %80, %c0_47, %c0_49, %c0_50, %83)
          %85 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x, %54, %80, %c0_47, %c0_49, %c0_50, %83)
          affine.store %79, %22[%84, %85] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %26 = frisk.to_threadTile %16#3 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %27 = frisk.to_threadTile %16#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_18 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %28 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_18) -> (vector<2x8xf32>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %67 = vector.extract %27[%54, %c0] : f32 from vector<2x1xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %c0_37 = arith.constant 0 : index
        %69 = vector.extract %27[%54, %c0_37] : f32 from vector<2x1xf32>
        %70 = vector.insert %69, %68 [0, 1] : f32 into vector<1x4xf32>
        %c0_38 = arith.constant 0 : index
        %71 = vector.extract %27[%54, %c0_38] : f32 from vector<2x1xf32>
        %72 = vector.insert %71, %70 [0, 2] : f32 into vector<1x4xf32>
        %c0_39 = arith.constant 0 : index
        %73 = vector.extract %27[%54, %c0_39] : f32 from vector<2x1xf32>
        %74 = vector.insert %73, %72 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %74 : vector<1x4xf32>
      }
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg5 [%54, %53] : f32 into vector<2x8xf32>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%54, %58] : f32 into vector<2x8xf32>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %60 [%54, %61] : f32 into vector<2x8xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%54, %64] : f32 into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_19 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %29 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_19) -> (vector<2x8xf32>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %67 = vector.extract %26[%54, %53] : f32 from vector<2x8xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %26[%54, %69] : f32 from vector<2x8xf32>
        %71 = vector.insert %70, %68 [0, 1] : f32 into vector<1x4xf32>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %26[%54, %72] : f32 from vector<2x8xf32>
        %74 = vector.insert %73, %71 [0, 2] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %26[%54, %75] : f32 from vector<2x8xf32>
        %77 = vector.insert %76, %74 [0, 3] : f32 into vector<1x4xf32>
        %cst_37 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %78 = vector.extract %28[%54, %53] : f32 from vector<2x8xf32>
        %79 = vector.insert %78, %cst_37 [0, 0] : f32 into vector<1x4xf32>
        %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %81 = vector.extract %28[%54, %80] : f32 from vector<2x8xf32>
        %82 = vector.insert %81, %79 [0, 1] : f32 into vector<1x4xf32>
        %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %84 = vector.extract %28[%54, %83] : f32 from vector<2x8xf32>
        %85 = vector.insert %84, %82 [0, 2] : f32 into vector<1x4xf32>
        %86 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %87 = vector.extract %28[%54, %86] : f32 from vector<2x8xf32>
        %88 = vector.insert %87, %85 [0, 3] : f32 into vector<1x4xf32>
        %89 = arith.divf %77, %88 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %89 : vector<1x4xf32>
      }
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg5 [%54, %53] : f32 into vector<2x8xf32>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%54, %58] : f32 into vector<2x8xf32>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %60 [%54, %61] : f32 into vector<2x8xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%54, %64] : f32 into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_20 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %30 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_20) -> (vector<2x8xf16>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %67 = vector.extract %29[%54, %53] : f32 from vector<2x8xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %29[%54, %69] : f32 from vector<2x8xf32>
        %71 = vector.insert %70, %68 [0, 1] : f32 into vector<1x4xf32>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %29[%54, %72] : f32 from vector<2x8xf32>
        %74 = vector.insert %73, %71 [0, 2] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %29[%54, %75] : f32 from vector<2x8xf32>
        %77 = vector.insert %76, %74 [0, 3] : f32 into vector<1x4xf32>
        %78 = arith.truncf %77 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %78 : vector<1x4xf16>
      }
      %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %arg5 [%54, %53] : f16 into vector<2x8xf16>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%54, %58] : f16 into vector<2x8xf16>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %60 [%54, %61] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%54, %64] : f16 into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %31 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + 32) floordiv 524288) mod 32, (%8 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_21 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %32 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_21) -> (vector<2x8xf16>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %67 = vector.extract %30[%54, %53] : f16 from vector<2x8xf16>
        %68 = vector.insert %67, %cst_36 [0, 0] : f16 into vector<1x4xf16>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %30[%54, %69] : f16 from vector<2x8xf16>
        %71 = vector.insert %70, %68 [0, 1] : f16 into vector<1x4xf16>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %30[%54, %72] : f16 from vector<2x8xf16>
        %74 = vector.insert %73, %71 [0, 2] : f16 into vector<1x4xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %30[%54, %75] : f16 from vector<2x8xf16>
        %77 = vector.insert %76, %74 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %77 : vector<1x4xf16>
      }
      %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %arg5 [%54, %53] : f16 into vector<2x8xf16>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%54, %58] : f16 into vector<2x8xf16>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %60 [%54, %61] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%54, %64] : f16 into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_22 = gpu.thread_id x
    %33 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_22)
    %c4_23 = arith.constant 4 : index
    %34 = arith.cmpi ult, %33, %c4_23 : index
    scf.if %34 {
      affine.for %arg4 = 0 to 4 {
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %55 = vector.extract %32[%54, %53] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_36 = arith.constant 0 : index
          %56 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%53)
          %c0_37 = arith.constant 0 : index
          %57 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%53)
          %58 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%57)
          %c0_38 = arith.constant 0 : index
          %59 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_36, %58)
          %60 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_22, %54, %56, %c0, %c0_37, %c0_38, %59)
          %61 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_22, %54, %56, %c0, %c0_37, %c0_38, %59)
          affine.store %55, %31[%60, %61] {tiled = true} : memref<128x32xf16, 1>
          %62 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %63 = vector.extract %32[%54, %62] : f16 from vector<2x8xf16>
          %c0_39 = arith.constant 0 : index
          %c0_40 = arith.constant 0 : index
          %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%62)
          %c0_41 = arith.constant 0 : index
          %65 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%62)
          %66 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%65)
          %c0_42 = arith.constant 0 : index
          %67 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_40, %66)
          %68 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_22, %54, %64, %c0_39, %c0_41, %c0_42, %67)
          %69 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_22, %54, %64, %c0_39, %c0_41, %c0_42, %67)
          affine.store %63, %31[%68, %69] {tiled = true} : memref<128x32xf16, 1>
          %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %71 = vector.extract %32[%54, %70] : f16 from vector<2x8xf16>
          %c0_43 = arith.constant 0 : index
          %c0_44 = arith.constant 0 : index
          %72 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%70)
          %c0_45 = arith.constant 0 : index
          %73 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%70)
          %74 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%73)
          %c0_46 = arith.constant 0 : index
          %75 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_44, %74)
          %76 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_22, %54, %72, %c0_43, %c0_45, %c0_46, %75)
          %77 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_22, %54, %72, %c0_43, %c0_45, %c0_46, %75)
          affine.store %71, %31[%76, %77] {tiled = true} : memref<128x32xf16, 1>
          %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %79 = vector.extract %32[%54, %78] : f16 from vector<2x8xf16>
          %c0_47 = arith.constant 0 : index
          %c0_48 = arith.constant 0 : index
          %80 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%78)
          %c0_49 = arith.constant 0 : index
          %81 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%78)
          %82 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%81)
          %c0_50 = arith.constant 0 : index
          %83 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_48, %82)
          %84 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_22, %54, %80, %c0_47, %c0_49, %c0_50, %83)
          %85 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_22, %54, %80, %c0_47, %c0_49, %c0_50, %83)
          affine.store %79, %31[%84, %85] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %35 = frisk.to_threadTile %16#4 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %36 = frisk.to_threadTile %16#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_24 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %37 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_24) -> (vector<2x8xf32>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %67 = vector.extract %36[%54, %c0] : f32 from vector<2x1xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %c0_37 = arith.constant 0 : index
        %69 = vector.extract %36[%54, %c0_37] : f32 from vector<2x1xf32>
        %70 = vector.insert %69, %68 [0, 1] : f32 into vector<1x4xf32>
        %c0_38 = arith.constant 0 : index
        %71 = vector.extract %36[%54, %c0_38] : f32 from vector<2x1xf32>
        %72 = vector.insert %71, %70 [0, 2] : f32 into vector<1x4xf32>
        %c0_39 = arith.constant 0 : index
        %73 = vector.extract %36[%54, %c0_39] : f32 from vector<2x1xf32>
        %74 = vector.insert %73, %72 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %74 : vector<1x4xf32>
      }
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg5 [%54, %53] : f32 into vector<2x8xf32>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%54, %58] : f32 into vector<2x8xf32>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %60 [%54, %61] : f32 into vector<2x8xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%54, %64] : f32 into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_25 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %38 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_25) -> (vector<2x8xf32>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %67 = vector.extract %35[%54, %53] : f32 from vector<2x8xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %35[%54, %69] : f32 from vector<2x8xf32>
        %71 = vector.insert %70, %68 [0, 1] : f32 into vector<1x4xf32>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %35[%54, %72] : f32 from vector<2x8xf32>
        %74 = vector.insert %73, %71 [0, 2] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %35[%54, %75] : f32 from vector<2x8xf32>
        %77 = vector.insert %76, %74 [0, 3] : f32 into vector<1x4xf32>
        %cst_37 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %78 = vector.extract %37[%54, %53] : f32 from vector<2x8xf32>
        %79 = vector.insert %78, %cst_37 [0, 0] : f32 into vector<1x4xf32>
        %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %81 = vector.extract %37[%54, %80] : f32 from vector<2x8xf32>
        %82 = vector.insert %81, %79 [0, 1] : f32 into vector<1x4xf32>
        %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %84 = vector.extract %37[%54, %83] : f32 from vector<2x8xf32>
        %85 = vector.insert %84, %82 [0, 2] : f32 into vector<1x4xf32>
        %86 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %87 = vector.extract %37[%54, %86] : f32 from vector<2x8xf32>
        %88 = vector.insert %87, %85 [0, 3] : f32 into vector<1x4xf32>
        %89 = arith.divf %77, %88 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %89 : vector<1x4xf32>
      }
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg5 [%54, %53] : f32 into vector<2x8xf32>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%54, %58] : f32 into vector<2x8xf32>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %60 [%54, %61] : f32 into vector<2x8xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%54, %64] : f32 into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_26 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %39 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_26) -> (vector<2x8xf16>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %67 = vector.extract %38[%54, %53] : f32 from vector<2x8xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %38[%54, %69] : f32 from vector<2x8xf32>
        %71 = vector.insert %70, %68 [0, 1] : f32 into vector<1x4xf32>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %38[%54, %72] : f32 from vector<2x8xf32>
        %74 = vector.insert %73, %71 [0, 2] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %38[%54, %75] : f32 from vector<2x8xf32>
        %77 = vector.insert %76, %74 [0, 3] : f32 into vector<1x4xf32>
        %78 = arith.truncf %77 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %78 : vector<1x4xf16>
      }
      %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %arg5 [%54, %53] : f16 into vector<2x8xf16>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%54, %58] : f16 into vector<2x8xf16>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %60 [%54, %61] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%54, %64] : f16 into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %40 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + 64) floordiv 524288) mod 32, (%8 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_27 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %41 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_27) -> (vector<2x8xf16>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %67 = vector.extract %39[%54, %53] : f16 from vector<2x8xf16>
        %68 = vector.insert %67, %cst_36 [0, 0] : f16 into vector<1x4xf16>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %39[%54, %69] : f16 from vector<2x8xf16>
        %71 = vector.insert %70, %68 [0, 1] : f16 into vector<1x4xf16>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %39[%54, %72] : f16 from vector<2x8xf16>
        %74 = vector.insert %73, %71 [0, 2] : f16 into vector<1x4xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %39[%54, %75] : f16 from vector<2x8xf16>
        %77 = vector.insert %76, %74 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %77 : vector<1x4xf16>
      }
      %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %arg5 [%54, %53] : f16 into vector<2x8xf16>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%54, %58] : f16 into vector<2x8xf16>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %60 [%54, %61] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%54, %64] : f16 into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_28 = gpu.thread_id x
    %42 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_28)
    %c4_29 = arith.constant 4 : index
    %43 = arith.cmpi ult, %42, %c4_29 : index
    scf.if %43 {
      affine.for %arg4 = 0 to 4 {
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %55 = vector.extract %41[%54, %53] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_36 = arith.constant 0 : index
          %56 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%53)
          %c0_37 = arith.constant 0 : index
          %57 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%53)
          %58 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%57)
          %c0_38 = arith.constant 0 : index
          %59 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_36, %58)
          %60 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_28, %54, %56, %c0, %c0_37, %c0_38, %59)
          %61 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_28, %54, %56, %c0, %c0_37, %c0_38, %59)
          affine.store %55, %40[%60, %61] {tiled = true} : memref<128x32xf16, 1>
          %62 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %63 = vector.extract %41[%54, %62] : f16 from vector<2x8xf16>
          %c0_39 = arith.constant 0 : index
          %c0_40 = arith.constant 0 : index
          %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%62)
          %c0_41 = arith.constant 0 : index
          %65 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%62)
          %66 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%65)
          %c0_42 = arith.constant 0 : index
          %67 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_40, %66)
          %68 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_28, %54, %64, %c0_39, %c0_41, %c0_42, %67)
          %69 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_28, %54, %64, %c0_39, %c0_41, %c0_42, %67)
          affine.store %63, %40[%68, %69] {tiled = true} : memref<128x32xf16, 1>
          %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %71 = vector.extract %41[%54, %70] : f16 from vector<2x8xf16>
          %c0_43 = arith.constant 0 : index
          %c0_44 = arith.constant 0 : index
          %72 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%70)
          %c0_45 = arith.constant 0 : index
          %73 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%70)
          %74 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%73)
          %c0_46 = arith.constant 0 : index
          %75 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_44, %74)
          %76 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_28, %54, %72, %c0_43, %c0_45, %c0_46, %75)
          %77 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_28, %54, %72, %c0_43, %c0_45, %c0_46, %75)
          affine.store %71, %40[%76, %77] {tiled = true} : memref<128x32xf16, 1>
          %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %79 = vector.extract %41[%54, %78] : f16 from vector<2x8xf16>
          %c0_47 = arith.constant 0 : index
          %c0_48 = arith.constant 0 : index
          %80 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%78)
          %c0_49 = arith.constant 0 : index
          %81 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%78)
          %82 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%81)
          %c0_50 = arith.constant 0 : index
          %83 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_48, %82)
          %84 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_28, %54, %80, %c0_47, %c0_49, %c0_50, %83)
          %85 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_28, %54, %80, %c0_47, %c0_49, %c0_50, %83)
          affine.store %79, %40[%84, %85] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %44 = frisk.to_threadTile %16#5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %45 = frisk.to_threadTile %16#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %46 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_30) -> (vector<2x8xf32>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %67 = vector.extract %45[%54, %c0] : f32 from vector<2x1xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %c0_37 = arith.constant 0 : index
        %69 = vector.extract %45[%54, %c0_37] : f32 from vector<2x1xf32>
        %70 = vector.insert %69, %68 [0, 1] : f32 into vector<1x4xf32>
        %c0_38 = arith.constant 0 : index
        %71 = vector.extract %45[%54, %c0_38] : f32 from vector<2x1xf32>
        %72 = vector.insert %71, %70 [0, 2] : f32 into vector<1x4xf32>
        %c0_39 = arith.constant 0 : index
        %73 = vector.extract %45[%54, %c0_39] : f32 from vector<2x1xf32>
        %74 = vector.insert %73, %72 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %74 : vector<1x4xf32>
      }
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg5 [%54, %53] : f32 into vector<2x8xf32>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%54, %58] : f32 into vector<2x8xf32>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %60 [%54, %61] : f32 into vector<2x8xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%54, %64] : f32 into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_31 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %47 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_31) -> (vector<2x8xf32>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %67 = vector.extract %44[%54, %53] : f32 from vector<2x8xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %44[%54, %69] : f32 from vector<2x8xf32>
        %71 = vector.insert %70, %68 [0, 1] : f32 into vector<1x4xf32>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %44[%54, %72] : f32 from vector<2x8xf32>
        %74 = vector.insert %73, %71 [0, 2] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %44[%54, %75] : f32 from vector<2x8xf32>
        %77 = vector.insert %76, %74 [0, 3] : f32 into vector<1x4xf32>
        %cst_37 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %78 = vector.extract %46[%54, %53] : f32 from vector<2x8xf32>
        %79 = vector.insert %78, %cst_37 [0, 0] : f32 into vector<1x4xf32>
        %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %81 = vector.extract %46[%54, %80] : f32 from vector<2x8xf32>
        %82 = vector.insert %81, %79 [0, 1] : f32 into vector<1x4xf32>
        %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %84 = vector.extract %46[%54, %83] : f32 from vector<2x8xf32>
        %85 = vector.insert %84, %82 [0, 2] : f32 into vector<1x4xf32>
        %86 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %87 = vector.extract %46[%54, %86] : f32 from vector<2x8xf32>
        %88 = vector.insert %87, %85 [0, 3] : f32 into vector<1x4xf32>
        %89 = arith.divf %77, %88 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %89 : vector<1x4xf32>
      }
      %56 = vector.extract %55[0, 0] : f32 from vector<1x4xf32>
      %57 = vector.insert %56, %arg5 [%54, %53] : f32 into vector<2x8xf32>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f32 from vector<1x4xf32>
      %60 = vector.insert %59, %57 [%54, %58] : f32 into vector<2x8xf32>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %60 [%54, %61] : f32 into vector<2x8xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%54, %64] : f32 into vector<2x8xf32>
      affine.yield %66 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_32 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %48 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_32) -> (vector<2x8xf16>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %67 = vector.extract %47[%54, %53] : f32 from vector<2x8xf32>
        %68 = vector.insert %67, %cst_36 [0, 0] : f32 into vector<1x4xf32>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %47[%54, %69] : f32 from vector<2x8xf32>
        %71 = vector.insert %70, %68 [0, 1] : f32 into vector<1x4xf32>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %47[%54, %72] : f32 from vector<2x8xf32>
        %74 = vector.insert %73, %71 [0, 2] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %47[%54, %75] : f32 from vector<2x8xf32>
        %77 = vector.insert %76, %74 [0, 3] : f32 into vector<1x4xf32>
        %78 = arith.truncf %77 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %78 : vector<1x4xf16>
      }
      %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %arg5 [%54, %53] : f16 into vector<2x8xf16>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%54, %58] : f16 into vector<2x8xf16>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %60 [%54, %61] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%54, %64] : f16 into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %49 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + 96) floordiv 524288) mod 32, (%8 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_33 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %50 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_33) -> (vector<2x8xf16>) {
      %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %55 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_36 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %67 = vector.extract %48[%54, %53] : f16 from vector<2x8xf16>
        %68 = vector.insert %67, %cst_36 [0, 0] : f16 into vector<1x4xf16>
        %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %70 = vector.extract %48[%54, %69] : f16 from vector<2x8xf16>
        %71 = vector.insert %70, %68 [0, 1] : f16 into vector<1x4xf16>
        %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %73 = vector.extract %48[%54, %72] : f16 from vector<2x8xf16>
        %74 = vector.insert %73, %71 [0, 2] : f16 into vector<1x4xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %76 = vector.extract %48[%54, %75] : f16 from vector<2x8xf16>
        %77 = vector.insert %76, %74 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %77 : vector<1x4xf16>
      }
      %56 = vector.extract %55[0, 0] : f16 from vector<1x4xf16>
      %57 = vector.insert %56, %arg5 [%54, %53] : f16 into vector<2x8xf16>
      %58 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %59 = vector.extract %55[0, 1] : f16 from vector<1x4xf16>
      %60 = vector.insert %59, %57 [%54, %58] : f16 into vector<2x8xf16>
      %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %62 = vector.extract %55[0, 2] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %60 [%54, %61] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %65 = vector.extract %55[0, 3] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%54, %64] : f16 into vector<2x8xf16>
      affine.yield %66 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_34 = gpu.thread_id x
    %51 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_34)
    %c4_35 = arith.constant 4 : index
    %52 = arith.cmpi ult, %51, %c4_35 : index
    scf.if %52 {
      affine.for %arg4 = 0 to 4 {
        %53 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %54 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %55 = vector.extract %50[%54, %53] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_36 = arith.constant 0 : index
          %56 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%53)
          %c0_37 = arith.constant 0 : index
          %57 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%53)
          %58 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%57)
          %c0_38 = arith.constant 0 : index
          %59 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_36, %58)
          %60 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_34, %54, %56, %c0, %c0_37, %c0_38, %59)
          %61 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_34, %54, %56, %c0, %c0_37, %c0_38, %59)
          affine.store %55, %49[%60, %61] {tiled = true} : memref<128x32xf16, 1>
          %62 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %63 = vector.extract %50[%54, %62] : f16 from vector<2x8xf16>
          %c0_39 = arith.constant 0 : index
          %c0_40 = arith.constant 0 : index
          %64 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%62)
          %c0_41 = arith.constant 0 : index
          %65 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%62)
          %66 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%65)
          %c0_42 = arith.constant 0 : index
          %67 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_40, %66)
          %68 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_34, %54, %64, %c0_39, %c0_41, %c0_42, %67)
          %69 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_34, %54, %64, %c0_39, %c0_41, %c0_42, %67)
          affine.store %63, %49[%68, %69] {tiled = true} : memref<128x32xf16, 1>
          %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %71 = vector.extract %50[%54, %70] : f16 from vector<2x8xf16>
          %c0_43 = arith.constant 0 : index
          %c0_44 = arith.constant 0 : index
          %72 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%70)
          %c0_45 = arith.constant 0 : index
          %73 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%70)
          %74 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%73)
          %c0_46 = arith.constant 0 : index
          %75 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_44, %74)
          %76 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_34, %54, %72, %c0_43, %c0_45, %c0_46, %75)
          %77 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_34, %54, %72, %c0_43, %c0_45, %c0_46, %75)
          affine.store %71, %49[%76, %77] {tiled = true} : memref<128x32xf16, 1>
          %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %79 = vector.extract %50[%54, %78] : f16 from vector<2x8xf16>
          %c0_47 = arith.constant 0 : index
          %c0_48 = arith.constant 0 : index
          %80 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%78)
          %c0_49 = arith.constant 0 : index
          %81 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%78)
          %82 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%81)
          %c0_50 = arith.constant 0 : index
          %83 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_48, %82)
          %84 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_34, %54, %80, %c0_47, %c0_49, %c0_50, %83)
          %85 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_34, %54, %80, %c0_47, %c0_49, %c0_50, %83)
          affine.store %79, %49[%84, %85] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
