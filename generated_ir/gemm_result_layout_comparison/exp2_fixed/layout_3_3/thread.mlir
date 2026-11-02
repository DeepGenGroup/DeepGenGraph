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
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 4, 0, 1, 1>, thread_num = 256 : i32, warp_layout = array<i64: 4, 16>} {
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
      %74 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %75 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_56 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
        frisk.fragment_yield %cst_56 : vector<1x1xf32>
      }
      %76 = vector.extract %75[0, 0] : f32 from vector<1x1xf32>
      %77 = vector.insert %76, %arg5 [%74, %c0] : f32 into vector<2x1xf32>
      affine.yield %77 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %12 = vector.broadcast %cst_4 : f32 to vector<2x1xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %13 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_6) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %74 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %75 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %78 = vector.extract %11[%74, %c0] : f32 from vector<2x1xf32>
        %79 = vector.insert %78, %cst_56 [0, 0] : f32 into vector<1x1xf32>
        %cst_57 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %80 = vector.extract %12[%74, %c0] : f32 from vector<2x1xf32>
        %81 = vector.insert %80, %cst_57 [0, 0] : f32 into vector<1x1xf32>
        %82 = arith.addf %79, %81 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        frisk.fragment_yield %82 : vector<1x1xf32>
      }
      %76 = vector.extract %75[0, 0] : f32 from vector<1x1xf32>
      %77 = vector.insert %76, %arg5 [%74, %c0] : f32 into vector<2x1xf32>
      affine.yield %77 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %14 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_9) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %74 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %75 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %78 = vector.extract %13[%74, %c0] : f32 from vector<2x1xf32>
        %79 = vector.insert %78, %cst_56 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
        frisk.fragment_yield %79 : vector<1x1xf32>
      }
      %76 = vector.extract %75[0, 0] : f32 from vector<1x1xf32>
      %77 = vector.insert %76, %arg5 [%74, %c0] : f32 into vector<2x1xf32>
      affine.yield %77 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %15 = frisk.from_threadTile %14 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %16:6 = affine.for %arg4 = 0 to affine_map<()[s0] -> (s0 * 128 + 128)>()[%8] step 128 iter_args(%arg5 = %cst_7, %arg6 = %cst_8, %arg7 = %15, %arg8 = %cst_10, %arg9 = %cst_11, %arg10 = %cst_12) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %cst_56 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %74 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_56) -> (vector<128x128xf32>) {
        %alloc_114 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_115 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %149 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %8 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %149 to %alloc_114 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %150 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_114 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %150 to %alloc_115 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %151 = frisk.to_threadTile %alloc_114 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
        %152 = vector.broadcast %9 : f16 to vector<2x8xf16>
        %cst_116 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
        %153 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_116) -> (vector<2x8xf16>) {
          %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg13)
          %160 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg13)
          %161 = frisk.fragment "compute" -> (vector<1x4xf16>) {
            %cst_121 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %173 = vector.extract %151[%160, %159] : f16 from vector<2x8xf16>
            %174 = vector.insert %173, %cst_121 [0, 0] : f16 into vector<1x4xf16>
            %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg13)
            %176 = vector.extract %151[%160, %175] : f16 from vector<2x8xf16>
            %177 = vector.insert %176, %174 [0, 1] : f16 into vector<1x4xf16>
            %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg13)
            %179 = vector.extract %151[%160, %178] : f16 from vector<2x8xf16>
            %180 = vector.insert %179, %177 [0, 2] : f16 into vector<1x4xf16>
            %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg13)
            %182 = vector.extract %151[%160, %181] : f16 from vector<2x8xf16>
            %183 = vector.insert %182, %180 [0, 3] : f16 into vector<1x4xf16>
            %cst_122 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %184 = vector.extract %152[%160, %159] : f16 from vector<2x8xf16>
            %185 = vector.insert %184, %cst_122 [0, 0] : f16 into vector<1x4xf16>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg13)
            %187 = vector.extract %152[%160, %186] : f16 from vector<2x8xf16>
            %188 = vector.insert %187, %185 [0, 1] : f16 into vector<1x4xf16>
            %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg13)
            %190 = vector.extract %152[%160, %189] : f16 from vector<2x8xf16>
            %191 = vector.insert %190, %188 [0, 2] : f16 into vector<1x4xf16>
            %192 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg13)
            %193 = vector.extract %152[%160, %192] : f16 from vector<2x8xf16>
            %194 = vector.insert %193, %191 [0, 3] : f16 into vector<1x4xf16>
            %195 = arith.mulf %183, %194 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
            frisk.fragment_yield %195 : vector<1x4xf16>
          }
          %162 = vector.extract %161[0, 0] : f16 from vector<1x4xf16>
          %163 = vector.insert %162, %arg14 [%160, %159] : f16 into vector<2x8xf16>
          %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg13)
          %165 = vector.extract %161[0, 1] : f16 from vector<1x4xf16>
          %166 = vector.insert %165, %163 [%160, %164] : f16 into vector<2x8xf16>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg13)
          %168 = vector.extract %161[0, 2] : f16 from vector<1x4xf16>
          %169 = vector.insert %168, %166 [%160, %167] : f16 into vector<2x8xf16>
          %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg13)
          %171 = vector.extract %161[0, 3] : f16 from vector<1x4xf16>
          %172 = vector.insert %171, %169 [%160, %170] : f16 into vector<2x8xf16>
          affine.yield %172 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        frisk.wait_buffer_ready %alloc_115 : memref<32x128xf16, 3>
        %cst_117 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %cst_118 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %154 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_117) -> (vector<2x32xf32>) {
          %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %160 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %161 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %cst_118) -> (vector<1x4xf32>) {
            %173 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %176 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
              %177 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg15)
              %cst_121 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
              %178 = vector.extract %153[%176, %177] : f16 from vector<2x8xf16>
              %179 = vector.insert %178, %cst_121 [0, 0] : f16 into vector<1x4xf16>
              %180 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg15)
              %181 = vector.extract %153[%176, %180] : f16 from vector<2x8xf16>
              %182 = vector.insert %181, %179 [0, 1] : f16 into vector<1x4xf16>
              %183 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg15)
              %184 = vector.extract %153[%176, %183] : f16 from vector<2x8xf16>
              %185 = vector.insert %184, %182 [0, 2] : f16 into vector<1x4xf16>
              %186 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg15)
              %187 = vector.extract %153[%176, %186] : f16 from vector<2x8xf16>
              %188 = vector.insert %187, %185 [0, 3] : f16 into vector<1x4xf16>
              frisk.fragment_yield %188 : vector<1x4xf16>
            }
            %174 = frisk.fragment "load" -> (vector<4x1xf16>) {
              %176 = frisk.to_threadTile %alloc_115 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
              %177 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg15)
              %178 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg13)
              %cst_121 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
              %179 = vector.extract %176[%177, %178] : f16 from vector<8x8xf16>
              %180 = vector.insert %179, %cst_121 [0, 0] : f16 into vector<4x1xf16>
              %181 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg15)
              %182 = vector.extract %176[%181, %178] : f16 from vector<8x8xf16>
              %183 = vector.insert %182, %180 [1, 0] : f16 into vector<4x1xf16>
              %184 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg15)
              %185 = vector.extract %176[%184, %178] : f16 from vector<8x8xf16>
              %186 = vector.insert %185, %183 [2, 0] : f16 into vector<4x1xf16>
              %187 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg15)
              %188 = vector.extract %176[%187, %178] : f16 from vector<8x8xf16>
              %189 = vector.insert %188, %186 [3, 0] : f16 into vector<4x1xf16>
              frisk.fragment_yield %189 : vector<4x1xf16>
            }
            %175 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %176 = frisk.warp_mma_rr %173, %174, %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %176 : vector<1x4xf32>
            }
            affine.yield %175 : vector<1x4xf32>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %162 = vector.extract %161[0, 0] : f32 from vector<1x4xf32>
          %163 = vector.insert %162, %arg14 [%160, %159] : f32 into vector<2x32xf32>
          %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %165 = vector.extract %161[0, 1] : f32 from vector<1x4xf32>
          %166 = vector.insert %165, %163 [%160, %164] : f32 into vector<2x32xf32>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %168 = vector.extract %161[0, 2] : f32 from vector<1x4xf32>
          %169 = vector.insert %168, %166 [%160, %167] : f32 into vector<2x32xf32>
          %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %171 = vector.extract %161[0, 3] : f32 from vector<1x4xf32>
          %172 = vector.insert %171, %169 [%160, %170] : f32 into vector<2x32xf32>
          affine.yield %172 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %155 = frisk.to_threadTile %arg12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
        %cst_119 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %156 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_119) -> (vector<2x32xf32>) {
          %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %160 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %161 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_121 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %173 = vector.extract %155[%160, %159] : f32 from vector<2x32xf32>
            %174 = vector.insert %173, %cst_121 [0, 0] : f32 into vector<1x4xf32>
            %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
            %176 = vector.extract %155[%160, %175] : f32 from vector<2x32xf32>
            %177 = vector.insert %176, %174 [0, 1] : f32 into vector<1x4xf32>
            %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
            %179 = vector.extract %155[%160, %178] : f32 from vector<2x32xf32>
            %180 = vector.insert %179, %177 [0, 2] : f32 into vector<1x4xf32>
            %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
            %182 = vector.extract %155[%160, %181] : f32 from vector<2x32xf32>
            %183 = vector.insert %182, %180 [0, 3] : f32 into vector<1x4xf32>
            %cst_122 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %184 = vector.extract %154[%160, %159] : f32 from vector<2x32xf32>
            %185 = vector.insert %184, %cst_122 [0, 0] : f32 into vector<1x4xf32>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
            %187 = vector.extract %154[%160, %186] : f32 from vector<2x32xf32>
            %188 = vector.insert %187, %185 [0, 1] : f32 into vector<1x4xf32>
            %189 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
            %190 = vector.extract %154[%160, %189] : f32 from vector<2x32xf32>
            %191 = vector.insert %190, %188 [0, 2] : f32 into vector<1x4xf32>
            %192 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
            %193 = vector.extract %154[%160, %192] : f32 from vector<2x32xf32>
            %194 = vector.insert %193, %191 [0, 3] : f32 into vector<1x4xf32>
            %195 = arith.addf %183, %194 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %195 : vector<1x4xf32>
          }
          %162 = vector.extract %161[0, 0] : f32 from vector<1x4xf32>
          %163 = vector.insert %162, %arg14 [%160, %159] : f32 into vector<2x32xf32>
          %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %165 = vector.extract %161[0, 1] : f32 from vector<1x4xf32>
          %166 = vector.insert %165, %163 [%160, %164] : f32 into vector<2x32xf32>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %168 = vector.extract %161[0, 2] : f32 from vector<1x4xf32>
          %169 = vector.insert %168, %166 [%160, %167] : f32 into vector<2x32xf32>
          %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %171 = vector.extract %161[0, 3] : f32 from vector<1x4xf32>
          %172 = vector.insert %171, %169 [%160, %170] : f32 into vector<2x32xf32>
          affine.yield %172 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %cst_120 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %157 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_120) -> (vector<2x32xf32>) {
          %159 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
          %160 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg13)
          %161 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_121 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %173 = vector.extract %156[%160, %159] : f32 from vector<2x32xf32>
            %174 = vector.insert %173, %cst_121 [0, 0] : f32 into vector<1x4xf32>
            %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
            %176 = vector.extract %156[%160, %175] : f32 from vector<2x32xf32>
            %177 = vector.insert %176, %174 [0, 1] : f32 into vector<1x4xf32>
            %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
            %179 = vector.extract %156[%160, %178] : f32 from vector<2x32xf32>
            %180 = vector.insert %179, %177 [0, 2] : f32 into vector<1x4xf32>
            %181 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
            %182 = vector.extract %156[%160, %181] : f32 from vector<2x32xf32>
            %183 = vector.insert %182, %180 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
            frisk.fragment_yield %183 : vector<1x4xf32>
          }
          %162 = vector.extract %161[0, 0] : f32 from vector<1x4xf32>
          %163 = vector.insert %162, %arg14 [%160, %159] : f32 into vector<2x32xf32>
          %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
          %165 = vector.extract %161[0, 1] : f32 from vector<1x4xf32>
          %166 = vector.insert %165, %163 [%160, %164] : f32 into vector<2x32xf32>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
          %168 = vector.extract %161[0, 2] : f32 from vector<1x4xf32>
          %169 = vector.insert %168, %166 [%160, %167] : f32 into vector<2x32xf32>
          %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
          %171 = vector.extract %161[0, 3] : f32 from vector<1x4xf32>
          %172 = vector.insert %171, %169 [%160, %170] : f32 into vector<2x32xf32>
          affine.yield %172 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %158 = frisk.from_threadTile %157 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
        affine.yield %158 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %cst_57 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %thread_id_x_58 = gpu.thread_id x
      %c127 = arith.constant 127 : index
      %75 = arith.addi %10, %c127 : index
      %c127_59 = arith.constant 127 : index
      %76 = arith.addi %arg4, %c127_59 : index
      %c1_60 = arith.constant 1 : index
      %77 = arith.addi %10, %c1_60 : index
      %c1_61 = arith.constant 1 : index
      %78 = arith.addi %75, %c1_61 : index
      %79 = arith.cmpi uge, %78, %75 : index
      %80 = arith.cmpi uge, %75, %10 : index
      %81 = arith.andi %80, %79 : i1
      %82 = arith.cmpi uge, %76, %arg4 : index
      %83 = arith.andi %81, %82 : i1
      %84 = arith.cmpi ugt, %77, %76 : index
      %85 = arith.andi %83, %84 : i1
      %cst_62 = arith.constant 0.000000e+00 : f32
      %86 = scf.if %85 -> (vector<2x32xf32>) {
        %149 = vector.broadcast %cst_62 : f32 to vector<2x32xf32>
        scf.yield %149 : vector<2x32xf32>
      } else {
        %149 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_57) -> (vector<2x32xf32>) {
          %150 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
          %151 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
          %152 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %c0 = arith.constant 0 : index
            %c0_115 = arith.constant 0 : index
            %164 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%150)
            %c0_116 = arith.constant 0 : index
            %165 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%150)
            %166 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_115, %165)
            %c0_117 = arith.constant 0 : index
            %167 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_58, %151, %164, %c0, %c0_116, %166, %c0_117)
            %168 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_58, %151, %164, %c0, %c0_116, %166, %c0_117)
            %169 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %167)
            %170 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %168)
            %171 = arith.addi %169, %c1 : index
            %172 = arith.cmpi ule, %171, %170 : index
            %173 = arith.select %172, %cst_1, %cst_2 : f32
            %174 = vector.insert %173, %cst_114 [0, 0] : f32 into vector<1x4xf32>
            %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
            %c0_118 = arith.constant 0 : index
            %c0_119 = arith.constant 0 : index
            %176 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%175)
            %c0_120 = arith.constant 0 : index
            %177 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%175)
            %178 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_119, %177)
            %c0_121 = arith.constant 0 : index
            %179 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_58, %151, %176, %c0_118, %c0_120, %178, %c0_121)
            %180 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_58, %151, %176, %c0_118, %c0_120, %178, %c0_121)
            %181 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %179)
            %182 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %180)
            %183 = arith.addi %181, %c1 : index
            %184 = arith.cmpi ule, %183, %182 : index
            %185 = arith.select %184, %cst_1, %cst_2 : f32
            %186 = vector.insert %185, %174 [0, 1] : f32 into vector<1x4xf32>
            %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
            %c0_122 = arith.constant 0 : index
            %c0_123 = arith.constant 0 : index
            %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%187)
            %c0_124 = arith.constant 0 : index
            %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%187)
            %190 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_123, %189)
            %c0_125 = arith.constant 0 : index
            %191 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_58, %151, %188, %c0_122, %c0_124, %190, %c0_125)
            %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_58, %151, %188, %c0_122, %c0_124, %190, %c0_125)
            %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %191)
            %194 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %192)
            %195 = arith.addi %193, %c1 : index
            %196 = arith.cmpi ule, %195, %194 : index
            %197 = arith.select %196, %cst_1, %cst_2 : f32
            %198 = vector.insert %197, %186 [0, 2] : f32 into vector<1x4xf32>
            %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
            %c0_126 = arith.constant 0 : index
            %c0_127 = arith.constant 0 : index
            %200 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%199)
            %c0_128 = arith.constant 0 : index
            %201 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%199)
            %202 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_127, %201)
            %c0_129 = arith.constant 0 : index
            %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_58, %151, %200, %c0_126, %c0_128, %202, %c0_129)
            %204 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_58, %151, %200, %c0_126, %c0_128, %202, %c0_129)
            %205 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %203)
            %206 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %204)
            %207 = arith.addi %205, %c1 : index
            %208 = arith.cmpi ule, %207, %206 : index
            %209 = arith.select %208, %cst_1, %cst_2 : f32
            %210 = vector.insert %209, %198 [0, 3] : f32 into vector<1x4xf32>
            frisk.fragment_yield %210 : vector<1x4xf32>
          }
          %153 = vector.extract %152[0, 0] : f32 from vector<1x4xf32>
          %154 = vector.insert %153, %arg12 [%151, %150] : f32 into vector<2x32xf32>
          %155 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %156 = vector.extract %152[0, 1] : f32 from vector<1x4xf32>
          %157 = vector.insert %156, %154 [%151, %155] : f32 into vector<2x32xf32>
          %158 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %159 = vector.extract %152[0, 2] : f32 from vector<1x4xf32>
          %160 = vector.insert %159, %157 [%151, %158] : f32 into vector<2x32xf32>
          %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %162 = vector.extract %152[0, 3] : f32 from vector<1x4xf32>
          %163 = vector.insert %162, %160 [%151, %161] : f32 into vector<2x32xf32>
          affine.yield %163 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "mask_fragment", tiled = true}
        scf.yield %149 : vector<2x32xf32>
      } {frisk.uniform_mask}
      %87 = frisk.to_threadTile %74 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_63 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %88 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_63) -> (vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %87[%150, %149] : f32 from vector<2x32xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %166 = vector.extract %87[%150, %165] : f32 from vector<2x32xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %169 = vector.extract %87[%150, %168] : f32 from vector<2x32xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %172 = vector.extract %87[%150, %171] : f32 from vector<2x32xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %86[%150, %149] : f32 from vector<2x32xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %177 = vector.extract %86[%150, %176] : f32 from vector<2x32xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %180 = vector.extract %86[%150, %179] : f32 from vector<2x32xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %183 = vector.extract %86[%150, %182] : f32 from vector<2x32xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.addf %173, %184 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x32xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x32xf32>
        affine.yield %162 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_64 = arith.constant 0xFF800000 : f32
      %cst_65 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %89 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_65) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %150 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %153 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_64) -> (f32) {
            %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
            %158 = frisk.fragment "compute" -> (f32) {
              %159 = vector.extract %88[%149, %157] : f32 from vector<2x32xf32>
              %160 = arith.maxnumf %arg14, %159 : f32
              %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
              %162 = vector.extract %88[%149, %161] : f32 from vector<2x32xf32>
              %163 = arith.maxnumf %160, %162 : f32
              %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
              %165 = vector.extract %88[%149, %164] : f32 from vector<2x32xf32>
              %166 = arith.maxnumf %163, %165 : f32
              %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
              %168 = vector.extract %88[%149, %167] : f32 from vector<2x32xf32>
              %169 = arith.maxnumf %166, %168 : f32
              frisk.fragment_yield %169 : f32
            }
            affine.yield %158 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_115 = arith.constant 64 : i32
          %c16_i32_116 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %153, %c16_i32_116, %c64_i32_115 : f32
          %154 = arith.maxnumf %153, %shuffleResult : f32
          %c64_i32_117 = arith.constant 64 : i32
          %c32_i32_118 = arith.constant 32 : i32
          %shuffleResult_119, %valid_120 = gpu.shuffle xor %154, %c32_i32_118, %c64_i32_117 : f32
          %155 = arith.maxnumf %154, %shuffleResult_119 : f32
          %156 = vector.insert %155, %cst_114 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %156 : vector<1x1xf32>
        }
        %151 = vector.extract %150[0, 0] : f32 from vector<1x1xf32>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %90 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_66 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %91 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_66) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %150 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %153 = vector.extract %89[%149, %c0] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %cst_114 [0, 0] : f32 into vector<1x1xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %155 = vector.extract %90[%149, %c0] : f32 from vector<2x1xf32>
          %156 = vector.insert %155, %cst_115 [0, 0] : f32 into vector<1x1xf32>
          %157 = arith.maxnumf %154, %156 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %157 : vector<1x1xf32>
        }
        %151 = vector.extract %150[0, 0] : f32 from vector<1x1xf32>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_67 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %92 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_67) -> (vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %163 = vector.extract %91[%150, %c0] : f32 from vector<2x1xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %c0_115 = arith.constant 0 : index
          %165 = vector.extract %91[%150, %c0_115] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %164 [0, 1] : f32 into vector<1x4xf32>
          %c0_116 = arith.constant 0 : index
          %167 = vector.extract %91[%150, %c0_116] : f32 from vector<2x1xf32>
          %168 = vector.insert %167, %166 [0, 2] : f32 into vector<1x4xf32>
          %c0_117 = arith.constant 0 : index
          %169 = vector.extract %91[%150, %c0_117] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %168 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %170 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x32xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x32xf32>
        affine.yield %162 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %cst_68 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %93 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_68) -> (vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %88[%150, %149] : f32 from vector<2x32xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %166 = vector.extract %88[%150, %165] : f32 from vector<2x32xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %169 = vector.extract %88[%150, %168] : f32 from vector<2x32xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %172 = vector.extract %88[%150, %171] : f32 from vector<2x32xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %92[%150, %149] : f32 from vector<2x32xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %177 = vector.extract %92[%150, %176] : f32 from vector<2x32xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %180 = vector.extract %92[%150, %179] : f32 from vector<2x32xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %183 = vector.extract %92[%150, %182] : f32 from vector<2x32xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.subf %173, %184 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x32xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x32xf32>
        affine.yield %162 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %94 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_69 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %95 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_69) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %150 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %153 = vector.extract %94[%149, %c0] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %cst_114 [0, 0] : f32 into vector<1x1xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %155 = vector.extract %91[%149, %c0] : f32 from vector<2x1xf32>
          %156 = vector.insert %155, %cst_115 [0, 0] : f32 into vector<1x1xf32>
          %157 = arith.subf %154, %156 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %157 : vector<1x1xf32>
        }
        %151 = vector.extract %150[0, 0] : f32 from vector<1x1xf32>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_70 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %96 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_70) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %150 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %153 = vector.extract %95[%149, %c0] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %cst_114 [0, 0] : f32 into vector<1x1xf32>
          %155 = math.exp2 %154 {tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %155 : vector<1x1xf32>
        }
        %151 = vector.extract %150[0, 0] : f32 from vector<1x1xf32>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %97 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_71 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %98 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_71) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %150 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %153 = vector.extract %97[%149, %c0] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %cst_114 [0, 0] : f32 into vector<1x1xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %155 = vector.extract %96[%149, %c0] : f32 from vector<2x1xf32>
          %156 = vector.insert %155, %cst_115 [0, 0] : f32 into vector<1x1xf32>
          %157 = arith.mulf %154, %156 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %157 : vector<1x1xf32>
        }
        %151 = vector.extract %150[0, 0] : f32 from vector<1x1xf32>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_72 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %99 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_72) -> (vector<2x32xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %93[%150, %149] : f32 from vector<2x32xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %166 = vector.extract %93[%150, %165] : f32 from vector<2x32xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %169 = vector.extract %93[%150, %168] : f32 from vector<2x32xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %172 = vector.extract %93[%150, %171] : f32 from vector<2x32xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %174 = math.exp2 %173 {tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %174 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x32xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x32xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x32xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x32xf32>
        affine.yield %162 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_73 = arith.constant 0.000000e+00 : f32
      %cst_74 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %100 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_74) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %150 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %153 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_73) -> (f32) {
            %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg13)
            %158 = frisk.fragment "compute" -> (f32) {
              %159 = vector.extract %99[%149, %157] : f32 from vector<2x32xf32>
              %160 = arith.addf %arg14, %159 : f32
              %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg13)
              %162 = vector.extract %99[%149, %161] : f32 from vector<2x32xf32>
              %163 = arith.addf %160, %162 : f32
              %164 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg13)
              %165 = vector.extract %99[%149, %164] : f32 from vector<2x32xf32>
              %166 = arith.addf %163, %165 : f32
              %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg13)
              %168 = vector.extract %99[%149, %167] : f32 from vector<2x32xf32>
              %169 = arith.addf %166, %168 : f32
              frisk.fragment_yield %169 : f32
            }
            affine.yield %158 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_115 = arith.constant 64 : i32
          %c16_i32_116 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %153, %c16_i32_116, %c64_i32_115 : f32
          %154 = arith.addf %153, %shuffleResult : f32
          %c64_i32_117 = arith.constant 64 : i32
          %c32_i32_118 = arith.constant 32 : i32
          %shuffleResult_119, %valid_120 = gpu.shuffle xor %154, %c32_i32_118, %c64_i32_117 : f32
          %155 = arith.addf %154, %shuffleResult_119 : f32
          %156 = vector.insert %155, %cst_114 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %156 : vector<1x1xf32>
        }
        %151 = vector.extract %150[0, 0] : f32 from vector<1x1xf32>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %cst_75 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %101 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_75) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %150 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %153 = vector.extract %100[%149, %c0] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %cst_114 [0, 0] : f32 into vector<1x1xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %155 = vector.extract %98[%149, %c0] : f32 from vector<2x1xf32>
          %156 = vector.insert %155, %cst_115 [0, 0] : f32 into vector<1x1xf32>
          %157 = arith.addf %154, %156 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %157 : vector<1x1xf32>
        }
        %151 = vector.extract %150[0, 0] : f32 from vector<1x1xf32>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_76 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
      %102 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_76) -> (vector<2x32xf16>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %99[%150, %149] : f32 from vector<2x32xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
          %166 = vector.extract %99[%150, %165] : f32 from vector<2x32xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
          %169 = vector.extract %99[%150, %168] : f32 from vector<2x32xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
          %172 = vector.extract %99[%150, %171] : f32 from vector<2x32xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %174 = arith.truncf %173 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %174 : vector<1x4xf16>
        }
        %152 = vector.extract %151[0, 0] : f16 from vector<1x4xf16>
        %153 = vector.insert %152, %arg12 [%150, %149] : f16 into vector<2x32xf16>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f16 from vector<1x4xf16>
        %156 = vector.insert %155, %153 [%150, %154] : f16 into vector<2x32xf16>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f16 from vector<1x4xf16>
        %159 = vector.insert %158, %156 [%150, %157] : f16 into vector<2x32xf16>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f16 from vector<1x4xf16>
        %162 = vector.insert %161, %159 [%150, %160] : f16 into vector<2x32xf16>
        affine.yield %162 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %103 = frisk.from_threadTile %102 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x128xf16, 3>
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %104 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %104 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_77 = gpu.thread_id x
      %cst_78 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %105 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_78) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %150 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_77]
          %178 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_77]
          %179 = frisk.read_matrix_from_shm %alloc[%177, %178] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %180 = vector.bitcast %179 : vector<4xi32> to vector<8xf16>
          %cst_114 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %181 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_77]
          %182 = vector.extract %180[%181] : f16 from vector<8xf16>
          %183 = vector.insert %182, %cst_114 [0, 0] : f16 into vector<4x2xf16>
          %184 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_77]
          %185 = vector.extract %180[%184] : f16 from vector<8xf16>
          %186 = vector.insert %185, %183 [0, 1] : f16 into vector<4x2xf16>
          %187 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_77]
          %188 = vector.extract %180[%187] : f16 from vector<8xf16>
          %189 = vector.insert %188, %186 [1, 0] : f16 into vector<4x2xf16>
          %190 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_77]
          %191 = vector.extract %180[%190] : f16 from vector<8xf16>
          %192 = vector.insert %191, %189 [1, 1] : f16 into vector<4x2xf16>
          %193 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_77]
          %194 = vector.extract %180[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %192 [2, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_77]
          %197 = vector.extract %180[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [2, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_77]
          %200 = vector.extract %180[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [3, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_77]
          %203 = vector.extract %180[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %204 : vector<4x2xf16>
        }
        %151 = vector.extract %150[0, 0] : f16 from vector<4x2xf16>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f16 into vector<32x2xf16>
        %153 = affine.apply affine_map<() -> (1)>()
        %154 = vector.extract %150[0, 1] : f16 from vector<4x2xf16>
        %155 = vector.insert %154, %152 [%149, %153] : f16 into vector<32x2xf16>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %157 = vector.extract %150[1, 0] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %155 [%156, %c0] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<() -> (1)>()
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %150[1, 1] : f16 from vector<4x2xf16>
        %162 = vector.insert %161, %158 [%160, %159] : f16 into vector<32x2xf16>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %150[2, 0] : f16 from vector<4x2xf16>
        %165 = vector.insert %164, %162 [%163, %c0] : f16 into vector<32x2xf16>
        %166 = affine.apply affine_map<() -> (1)>()
        %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %168 = vector.extract %150[2, 1] : f16 from vector<4x2xf16>
        %169 = vector.insert %168, %165 [%167, %166] : f16 into vector<32x2xf16>
        %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %171 = vector.extract %150[3, 0] : f16 from vector<4x2xf16>
        %172 = vector.insert %171, %169 [%170, %c0] : f16 into vector<32x2xf16>
        %173 = affine.apply affine_map<() -> (1)>()
        %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %175 = vector.extract %150[3, 1] : f16 from vector<4x2xf16>
        %176 = vector.insert %175, %172 [%174, %173] : f16 into vector<32x2xf16>
        affine.yield %176 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_79 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_80 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %106 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_79) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_80) -> (vector<1x4xf32>) {
          %163 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %166 = frisk.to_threadTile %103 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %167 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
            %168 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %169 = vector.extract %166[%167, %168] : f16 from vector<2x32xf16>
            %170 = vector.insert %169, %cst_114 [0, 0] : f16 into vector<1x4xf16>
            %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %172 = vector.extract %166[%167, %171] : f16 from vector<2x32xf16>
            %173 = vector.insert %172, %170 [0, 1] : f16 into vector<1x4xf16>
            %174 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %175 = vector.extract %166[%167, %174] : f16 from vector<2x32xf16>
            %176 = vector.insert %175, %173 [0, 2] : f16 into vector<1x4xf16>
            %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %178 = vector.extract %166[%167, %177] : f16 from vector<2x32xf16>
            %179 = vector.insert %178, %176 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %179 : vector<1x4xf16>
          }
          %164 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %166 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %167 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %168 = vector.extract %105[%166, %167] : f16 from vector<32x2xf16>
            %169 = vector.insert %168, %cst_114 [0, 0] : f16 into vector<4x1xf16>
            %170 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %171 = vector.extract %105[%170, %167] : f16 from vector<32x2xf16>
            %172 = vector.insert %171, %169 [1, 0] : f16 into vector<4x1xf16>
            %173 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %174 = vector.extract %105[%173, %167] : f16 from vector<32x2xf16>
            %175 = vector.insert %174, %172 [2, 0] : f16 into vector<4x1xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %177 = vector.extract %105[%176, %167] : f16 from vector<32x2xf16>
            %178 = vector.insert %177, %175 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %178 : vector<4x1xf16>
          }
          %165 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %166 = frisk.warp_mma_rr %163, %164, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %166 : vector<1x4xf32>
          }
          affine.yield %165 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %107 = frisk.from_threadTile %96 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> memref<128x1xf32>
      %108 = frisk.to_threadTile %arg5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %109 = frisk.to_threadTile %107 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_81 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %110 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_81) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %163 = vector.extract %109[%150, %c0] : f32 from vector<2x1xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %c0_115 = arith.constant 0 : index
          %165 = vector.extract %109[%150, %c0_115] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %164 [0, 1] : f32 into vector<1x4xf32>
          %c0_116 = arith.constant 0 : index
          %167 = vector.extract %109[%150, %c0_116] : f32 from vector<2x1xf32>
          %168 = vector.insert %167, %166 [0, 2] : f32 into vector<1x4xf32>
          %c0_117 = arith.constant 0 : index
          %169 = vector.extract %109[%150, %c0_117] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %168 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %170 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_82 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %111 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_82) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %108[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %108[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %108[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %108[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %110[%150, %149] : f32 from vector<2x8xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %177 = vector.extract %110[%150, %176] : f32 from vector<2x8xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %180 = vector.extract %110[%150, %179] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %183 = vector.extract %110[%150, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.mulf %173, %184 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_83 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %112 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_83) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %111[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %111[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %111[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %111[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %106[%150, %149] : f32 from vector<2x8xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %177 = vector.extract %106[%150, %176] : f32 from vector<2x8xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %180 = vector.extract %106[%150, %179] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %183 = vector.extract %106[%150, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.addf %173, %184 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_84 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %113 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %113 to %alloc_84 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_84 : memref<128x32xf16, 3>
      %thread_id_x_85 = gpu.thread_id x
      %cst_86 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %114 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_86) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %150 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_85]
          %178 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_85]
          %179 = frisk.read_matrix_from_shm %alloc_84[%177, %178] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %180 = vector.bitcast %179 : vector<4xi32> to vector<8xf16>
          %cst_114 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %181 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_85]
          %182 = vector.extract %180[%181] : f16 from vector<8xf16>
          %183 = vector.insert %182, %cst_114 [0, 0] : f16 into vector<4x2xf16>
          %184 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_85]
          %185 = vector.extract %180[%184] : f16 from vector<8xf16>
          %186 = vector.insert %185, %183 [0, 1] : f16 into vector<4x2xf16>
          %187 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_85]
          %188 = vector.extract %180[%187] : f16 from vector<8xf16>
          %189 = vector.insert %188, %186 [1, 0] : f16 into vector<4x2xf16>
          %190 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_85]
          %191 = vector.extract %180[%190] : f16 from vector<8xf16>
          %192 = vector.insert %191, %189 [1, 1] : f16 into vector<4x2xf16>
          %193 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_85]
          %194 = vector.extract %180[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %192 [2, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_85]
          %197 = vector.extract %180[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [2, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_85]
          %200 = vector.extract %180[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [3, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_85]
          %203 = vector.extract %180[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %204 : vector<4x2xf16>
        }
        %151 = vector.extract %150[0, 0] : f16 from vector<4x2xf16>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f16 into vector<32x2xf16>
        %153 = affine.apply affine_map<() -> (1)>()
        %154 = vector.extract %150[0, 1] : f16 from vector<4x2xf16>
        %155 = vector.insert %154, %152 [%149, %153] : f16 into vector<32x2xf16>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %157 = vector.extract %150[1, 0] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %155 [%156, %c0] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<() -> (1)>()
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %150[1, 1] : f16 from vector<4x2xf16>
        %162 = vector.insert %161, %158 [%160, %159] : f16 into vector<32x2xf16>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %150[2, 0] : f16 from vector<4x2xf16>
        %165 = vector.insert %164, %162 [%163, %c0] : f16 into vector<32x2xf16>
        %166 = affine.apply affine_map<() -> (1)>()
        %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %168 = vector.extract %150[2, 1] : f16 from vector<4x2xf16>
        %169 = vector.insert %168, %165 [%167, %166] : f16 into vector<32x2xf16>
        %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %171 = vector.extract %150[3, 0] : f16 from vector<4x2xf16>
        %172 = vector.insert %171, %169 [%170, %c0] : f16 into vector<32x2xf16>
        %173 = affine.apply affine_map<() -> (1)>()
        %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %175 = vector.extract %150[3, 1] : f16 from vector<4x2xf16>
        %176 = vector.insert %175, %172 [%174, %173] : f16 into vector<32x2xf16>
        affine.yield %176 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_87 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_88 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %115 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_87) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_88) -> (vector<1x4xf32>) {
          %163 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %166 = frisk.to_threadTile %103 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %167 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
            %168 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %169 = vector.extract %166[%167, %168] : f16 from vector<2x32xf16>
            %170 = vector.insert %169, %cst_114 [0, 0] : f16 into vector<1x4xf16>
            %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %172 = vector.extract %166[%167, %171] : f16 from vector<2x32xf16>
            %173 = vector.insert %172, %170 [0, 1] : f16 into vector<1x4xf16>
            %174 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %175 = vector.extract %166[%167, %174] : f16 from vector<2x32xf16>
            %176 = vector.insert %175, %173 [0, 2] : f16 into vector<1x4xf16>
            %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %178 = vector.extract %166[%167, %177] : f16 from vector<2x32xf16>
            %179 = vector.insert %178, %176 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %179 : vector<1x4xf16>
          }
          %164 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %166 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %167 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %168 = vector.extract %114[%166, %167] : f16 from vector<32x2xf16>
            %169 = vector.insert %168, %cst_114 [0, 0] : f16 into vector<4x1xf16>
            %170 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %171 = vector.extract %114[%170, %167] : f16 from vector<32x2xf16>
            %172 = vector.insert %171, %169 [1, 0] : f16 into vector<4x1xf16>
            %173 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %174 = vector.extract %114[%173, %167] : f16 from vector<32x2xf16>
            %175 = vector.insert %174, %172 [2, 0] : f16 into vector<4x1xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %177 = vector.extract %114[%176, %167] : f16 from vector<32x2xf16>
            %178 = vector.insert %177, %175 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %178 : vector<4x1xf16>
          }
          %165 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %166 = frisk.warp_mma_rr %163, %164, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %166 : vector<1x4xf32>
          }
          affine.yield %165 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %116 = frisk.to_threadTile %arg8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %117 = frisk.to_threadTile %107 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_89 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %118 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_89) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %163 = vector.extract %117[%150, %c0] : f32 from vector<2x1xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %c0_115 = arith.constant 0 : index
          %165 = vector.extract %117[%150, %c0_115] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %164 [0, 1] : f32 into vector<1x4xf32>
          %c0_116 = arith.constant 0 : index
          %167 = vector.extract %117[%150, %c0_116] : f32 from vector<2x1xf32>
          %168 = vector.insert %167, %166 [0, 2] : f32 into vector<1x4xf32>
          %c0_117 = arith.constant 0 : index
          %169 = vector.extract %117[%150, %c0_117] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %168 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %170 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_90 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %119 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_90) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %116[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %116[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %116[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %116[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %118[%150, %149] : f32 from vector<2x8xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %177 = vector.extract %118[%150, %176] : f32 from vector<2x8xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %180 = vector.extract %118[%150, %179] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %183 = vector.extract %118[%150, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.mulf %173, %184 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_91 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %120 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_91) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %119[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %119[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %119[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %119[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %115[%150, %149] : f32 from vector<2x8xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %177 = vector.extract %115[%150, %176] : f32 from vector<2x8xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %180 = vector.extract %115[%150, %179] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %183 = vector.extract %115[%150, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.addf %173, %184 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_92 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %121 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %121 to %alloc_92 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_92 : memref<128x32xf16, 3>
      %thread_id_x_93 = gpu.thread_id x
      %cst_94 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %122 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_94) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %150 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_93]
          %178 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_93]
          %179 = frisk.read_matrix_from_shm %alloc_92[%177, %178] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %180 = vector.bitcast %179 : vector<4xi32> to vector<8xf16>
          %cst_114 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %181 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_93]
          %182 = vector.extract %180[%181] : f16 from vector<8xf16>
          %183 = vector.insert %182, %cst_114 [0, 0] : f16 into vector<4x2xf16>
          %184 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_93]
          %185 = vector.extract %180[%184] : f16 from vector<8xf16>
          %186 = vector.insert %185, %183 [0, 1] : f16 into vector<4x2xf16>
          %187 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_93]
          %188 = vector.extract %180[%187] : f16 from vector<8xf16>
          %189 = vector.insert %188, %186 [1, 0] : f16 into vector<4x2xf16>
          %190 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_93]
          %191 = vector.extract %180[%190] : f16 from vector<8xf16>
          %192 = vector.insert %191, %189 [1, 1] : f16 into vector<4x2xf16>
          %193 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_93]
          %194 = vector.extract %180[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %192 [2, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_93]
          %197 = vector.extract %180[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [2, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_93]
          %200 = vector.extract %180[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [3, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_93]
          %203 = vector.extract %180[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %204 : vector<4x2xf16>
        }
        %151 = vector.extract %150[0, 0] : f16 from vector<4x2xf16>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f16 into vector<32x2xf16>
        %153 = affine.apply affine_map<() -> (1)>()
        %154 = vector.extract %150[0, 1] : f16 from vector<4x2xf16>
        %155 = vector.insert %154, %152 [%149, %153] : f16 into vector<32x2xf16>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %157 = vector.extract %150[1, 0] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %155 [%156, %c0] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<() -> (1)>()
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %150[1, 1] : f16 from vector<4x2xf16>
        %162 = vector.insert %161, %158 [%160, %159] : f16 into vector<32x2xf16>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %150[2, 0] : f16 from vector<4x2xf16>
        %165 = vector.insert %164, %162 [%163, %c0] : f16 into vector<32x2xf16>
        %166 = affine.apply affine_map<() -> (1)>()
        %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %168 = vector.extract %150[2, 1] : f16 from vector<4x2xf16>
        %169 = vector.insert %168, %165 [%167, %166] : f16 into vector<32x2xf16>
        %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %171 = vector.extract %150[3, 0] : f16 from vector<4x2xf16>
        %172 = vector.insert %171, %169 [%170, %c0] : f16 into vector<32x2xf16>
        %173 = affine.apply affine_map<() -> (1)>()
        %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %175 = vector.extract %150[3, 1] : f16 from vector<4x2xf16>
        %176 = vector.insert %175, %172 [%174, %173] : f16 into vector<32x2xf16>
        affine.yield %176 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_95 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_96 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %123 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_95) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_96) -> (vector<1x4xf32>) {
          %163 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %166 = frisk.to_threadTile %103 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %167 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
            %168 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %169 = vector.extract %166[%167, %168] : f16 from vector<2x32xf16>
            %170 = vector.insert %169, %cst_114 [0, 0] : f16 into vector<1x4xf16>
            %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %172 = vector.extract %166[%167, %171] : f16 from vector<2x32xf16>
            %173 = vector.insert %172, %170 [0, 1] : f16 into vector<1x4xf16>
            %174 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %175 = vector.extract %166[%167, %174] : f16 from vector<2x32xf16>
            %176 = vector.insert %175, %173 [0, 2] : f16 into vector<1x4xf16>
            %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %178 = vector.extract %166[%167, %177] : f16 from vector<2x32xf16>
            %179 = vector.insert %178, %176 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %179 : vector<1x4xf16>
          }
          %164 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %166 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %167 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %168 = vector.extract %122[%166, %167] : f16 from vector<32x2xf16>
            %169 = vector.insert %168, %cst_114 [0, 0] : f16 into vector<4x1xf16>
            %170 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %171 = vector.extract %122[%170, %167] : f16 from vector<32x2xf16>
            %172 = vector.insert %171, %169 [1, 0] : f16 into vector<4x1xf16>
            %173 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %174 = vector.extract %122[%173, %167] : f16 from vector<32x2xf16>
            %175 = vector.insert %174, %172 [2, 0] : f16 into vector<4x1xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %177 = vector.extract %122[%176, %167] : f16 from vector<32x2xf16>
            %178 = vector.insert %177, %175 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %178 : vector<4x1xf16>
          }
          %165 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %166 = frisk.warp_mma_rr %163, %164, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %166 : vector<1x4xf32>
          }
          affine.yield %165 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %124 = frisk.to_threadTile %arg9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %125 = frisk.to_threadTile %107 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_97 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %126 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_97) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %163 = vector.extract %125[%150, %c0] : f32 from vector<2x1xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %c0_115 = arith.constant 0 : index
          %165 = vector.extract %125[%150, %c0_115] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %164 [0, 1] : f32 into vector<1x4xf32>
          %c0_116 = arith.constant 0 : index
          %167 = vector.extract %125[%150, %c0_116] : f32 from vector<2x1xf32>
          %168 = vector.insert %167, %166 [0, 2] : f32 into vector<1x4xf32>
          %c0_117 = arith.constant 0 : index
          %169 = vector.extract %125[%150, %c0_117] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %168 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %170 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_98 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %127 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_98) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %124[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %124[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %124[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %124[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %126[%150, %149] : f32 from vector<2x8xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %177 = vector.extract %126[%150, %176] : f32 from vector<2x8xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %180 = vector.extract %126[%150, %179] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %183 = vector.extract %126[%150, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.mulf %173, %184 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_99 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %128 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_99) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %127[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %127[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %127[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %127[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %123[%150, %149] : f32 from vector<2x8xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %177 = vector.extract %123[%150, %176] : f32 from vector<2x8xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %180 = vector.extract %123[%150, %179] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %183 = vector.extract %123[%150, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.addf %173, %184 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_100 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %129 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %129 to %alloc_100 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_100 : memref<128x32xf16, 3>
      %thread_id_x_101 = gpu.thread_id x
      %cst_102 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %130 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_102) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
        %150 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %177 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg11)[%thread_id_x_101]
          %178 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_101]
          %179 = frisk.read_matrix_from_shm %alloc_100[%177, %178] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %180 = vector.bitcast %179 : vector<4xi32> to vector<8xf16>
          %cst_114 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %181 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_101]
          %182 = vector.extract %180[%181] : f16 from vector<8xf16>
          %183 = vector.insert %182, %cst_114 [0, 0] : f16 into vector<4x2xf16>
          %184 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_101]
          %185 = vector.extract %180[%184] : f16 from vector<8xf16>
          %186 = vector.insert %185, %183 [0, 1] : f16 into vector<4x2xf16>
          %187 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg11)[%thread_id_x_101]
          %188 = vector.extract %180[%187] : f16 from vector<8xf16>
          %189 = vector.insert %188, %186 [1, 0] : f16 into vector<4x2xf16>
          %190 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg11)[%thread_id_x_101]
          %191 = vector.extract %180[%190] : f16 from vector<8xf16>
          %192 = vector.insert %191, %189 [1, 1] : f16 into vector<4x2xf16>
          %193 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg11)[%thread_id_x_101]
          %194 = vector.extract %180[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %192 [2, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg11)[%thread_id_x_101]
          %197 = vector.extract %180[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [2, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg11)[%thread_id_x_101]
          %200 = vector.extract %180[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [3, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg11)[%thread_id_x_101]
          %203 = vector.extract %180[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %204 : vector<4x2xf16>
        }
        %151 = vector.extract %150[0, 0] : f16 from vector<4x2xf16>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f16 into vector<32x2xf16>
        %153 = affine.apply affine_map<() -> (1)>()
        %154 = vector.extract %150[0, 1] : f16 from vector<4x2xf16>
        %155 = vector.insert %154, %152 [%149, %153] : f16 into vector<32x2xf16>
        %156 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %157 = vector.extract %150[1, 0] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %155 [%156, %c0] : f16 into vector<32x2xf16>
        %159 = affine.apply affine_map<() -> (1)>()
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
        %161 = vector.extract %150[1, 1] : f16 from vector<4x2xf16>
        %162 = vector.insert %161, %158 [%160, %159] : f16 into vector<32x2xf16>
        %163 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %164 = vector.extract %150[2, 0] : f16 from vector<4x2xf16>
        %165 = vector.insert %164, %162 [%163, %c0] : f16 into vector<32x2xf16>
        %166 = affine.apply affine_map<() -> (1)>()
        %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
        %168 = vector.extract %150[2, 1] : f16 from vector<4x2xf16>
        %169 = vector.insert %168, %165 [%167, %166] : f16 into vector<32x2xf16>
        %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %171 = vector.extract %150[3, 0] : f16 from vector<4x2xf16>
        %172 = vector.insert %171, %169 [%170, %c0] : f16 into vector<32x2xf16>
        %173 = affine.apply affine_map<() -> (1)>()
        %174 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
        %175 = vector.extract %150[3, 1] : f16 from vector<4x2xf16>
        %176 = vector.insert %175, %172 [%174, %173] : f16 into vector<32x2xf16>
        affine.yield %176 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_103 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_104 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %131 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_103) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_104) -> (vector<1x4xf32>) {
          %163 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %166 = frisk.to_threadTile %103 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %167 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
            %168 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %169 = vector.extract %166[%167, %168] : f16 from vector<2x32xf16>
            %170 = vector.insert %169, %cst_114 [0, 0] : f16 into vector<1x4xf16>
            %171 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %172 = vector.extract %166[%167, %171] : f16 from vector<2x32xf16>
            %173 = vector.insert %172, %170 [0, 1] : f16 into vector<1x4xf16>
            %174 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %175 = vector.extract %166[%167, %174] : f16 from vector<2x32xf16>
            %176 = vector.insert %175, %173 [0, 2] : f16 into vector<1x4xf16>
            %177 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %178 = vector.extract %166[%167, %177] : f16 from vector<2x32xf16>
            %179 = vector.insert %178, %176 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %179 : vector<1x4xf16>
          }
          %164 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %166 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
            %167 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %168 = vector.extract %130[%166, %167] : f16 from vector<32x2xf16>
            %169 = vector.insert %168, %cst_114 [0, 0] : f16 into vector<4x1xf16>
            %170 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
            %171 = vector.extract %130[%170, %167] : f16 from vector<32x2xf16>
            %172 = vector.insert %171, %169 [1, 0] : f16 into vector<4x1xf16>
            %173 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
            %174 = vector.extract %130[%173, %167] : f16 from vector<32x2xf16>
            %175 = vector.insert %174, %172 [2, 0] : f16 into vector<4x1xf16>
            %176 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
            %177 = vector.extract %130[%176, %167] : f16 from vector<32x2xf16>
            %178 = vector.insert %177, %175 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %178 : vector<4x1xf16>
          }
          %165 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %166 = frisk.warp_mma_rr %163, %164, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %166 : vector<1x4xf32>
          }
          affine.yield %165 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %132 = frisk.to_threadTile %arg10 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %133 = frisk.to_threadTile %107 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_105 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %134 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_105) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %163 = vector.extract %133[%150, %c0] : f32 from vector<2x1xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %c0_115 = arith.constant 0 : index
          %165 = vector.extract %133[%150, %c0_115] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %164 [0, 1] : f32 into vector<1x4xf32>
          %c0_116 = arith.constant 0 : index
          %167 = vector.extract %133[%150, %c0_116] : f32 from vector<2x1xf32>
          %168 = vector.insert %167, %166 [0, 2] : f32 into vector<1x4xf32>
          %c0_117 = arith.constant 0 : index
          %169 = vector.extract %133[%150, %c0_117] : f32 from vector<2x1xf32>
          %170 = vector.insert %169, %168 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %170 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_106 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %135 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_106) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %132[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %132[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %132[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %132[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %134[%150, %149] : f32 from vector<2x8xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %177 = vector.extract %134[%150, %176] : f32 from vector<2x8xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %180 = vector.extract %134[%150, %179] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %183 = vector.extract %134[%150, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.mulf %173, %184 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_107 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %136 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_107) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %135[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %135[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %135[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %135[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
          %cst_115 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %174 = vector.extract %131[%150, %149] : f32 from vector<2x8xf32>
          %175 = vector.insert %174, %cst_115 [0, 0] : f32 into vector<1x4xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %177 = vector.extract %131[%150, %176] : f32 from vector<2x8xf32>
          %178 = vector.insert %177, %175 [0, 1] : f32 into vector<1x4xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %180 = vector.extract %131[%150, %179] : f32 from vector<2x8xf32>
          %181 = vector.insert %180, %178 [0, 2] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %183 = vector.extract %131[%150, %182] : f32 from vector<2x8xf32>
          %184 = vector.insert %183, %181 [0, 3] : f32 into vector<1x4xf32>
          %185 = arith.addf %173, %184 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_108 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %137 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_108) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %112[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %112[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %112[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %112[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %173 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %138 = frisk.from_threadTile %137 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_109 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %139 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_109) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %150 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %153 = vector.extract %101[%149, %c0] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %cst_114 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
          frisk.fragment_yield %154 : vector<1x1xf32>
        }
        %151 = vector.extract %150[0, 0] : f32 from vector<1x1xf32>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %140 = frisk.from_threadTile %139 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_110 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %141 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_110) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %149 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
        %150 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %153 = vector.extract %91[%149, %c0] : f32 from vector<2x1xf32>
          %154 = vector.insert %153, %cst_114 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
          frisk.fragment_yield %154 : vector<1x1xf32>
        }
        %151 = vector.extract %150[0, 0] : f32 from vector<1x1xf32>
        %152 = vector.insert %151, %arg12 [%149, %c0] : f32 into vector<2x1xf32>
        affine.yield %152 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %142 = frisk.from_threadTile %141 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_111 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %143 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_111) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %120[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %120[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %120[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %120[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %173 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %144 = frisk.from_threadTile %143 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_112 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %145 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_112) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %128[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %128[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %128[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %128[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %173 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %146 = frisk.from_threadTile %145 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_113 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %147 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_113) -> (vector<2x8xf32>) {
        %149 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg11)
        %150 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg11)
        %151 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %163 = vector.extract %136[%150, %149] : f32 from vector<2x8xf32>
          %164 = vector.insert %163, %cst_114 [0, 0] : f32 into vector<1x4xf32>
          %165 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
          %166 = vector.extract %136[%150, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
          %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
          %169 = vector.extract %136[%150, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
          %172 = vector.extract %136[%150, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %173 : vector<1x4xf32>
        }
        %152 = vector.extract %151[0, 0] : f32 from vector<1x4xf32>
        %153 = vector.insert %152, %arg12 [%150, %149] : f32 into vector<2x8xf32>
        %154 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg11)
        %155 = vector.extract %151[0, 1] : f32 from vector<1x4xf32>
        %156 = vector.insert %155, %153 [%150, %154] : f32 into vector<2x8xf32>
        %157 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg11)
        %158 = vector.extract %151[0, 2] : f32 from vector<1x4xf32>
        %159 = vector.insert %158, %156 [%150, %157] : f32 into vector<2x8xf32>
        %160 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg11)
        %161 = vector.extract %151[0, 3] : f32 from vector<1x4xf32>
        %162 = vector.insert %161, %159 [%150, %160] : f32 into vector<2x8xf32>
        affine.yield %162 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %148 = frisk.from_threadTile %147 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      affine.yield %138, %140, %142, %144, %146, %148 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %17 = frisk.to_threadTile %16#5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x = gpu.thread_id x
    %18 = arith.index_cast %thread_id_x : index to i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %19 = arith.andi %18, %c16_i32 : i32
    %20 = arith.cmpi ne, %19, %c0_i32 : i32
    %c32_i32 = arith.constant 32 : i32
    %21 = arith.andi %18, %c32_i32 : i32
    %22 = arith.cmpi ne, %21, %c0_i32 : i32
    %cst_13 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %23 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_13) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %17[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %17[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %17[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %17[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
        %100 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
        %101 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
        %102 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
        %103 = arith.select %20, %99, %100 : f32
        %shuffleResult, %valid = gpu.shuffle xor %103, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %104 = arith.select %20, %shuffleResult, %99 : f32
        %105 = arith.select %20, %100, %shuffleResult : f32
        %106 = arith.select %20, %101, %102 : f32
        %shuffleResult_57, %valid_58 = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %20, %shuffleResult_57, %101 : f32
        %108 = arith.select %20, %102, %shuffleResult_57 : f32
        %109 = arith.select %22, %104, %107 : f32
        %shuffleResult_59, %valid_60 = gpu.shuffle xor %109, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %22, %shuffleResult_59, %104 : f32
        %111 = arith.select %22, %107, %shuffleResult_59 : f32
        %112 = arith.select %22, %105, %108 : f32
        %shuffleResult_61, %valid_62 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %22, %shuffleResult_61, %105 : f32
        %114 = arith.select %22, %108, %shuffleResult_61 : f32
        %115 = vector.insert %110, %98 [0, 0] : f32 into vector<1x4xf32>
        %116 = vector.insert %113, %115 [0, 1] : f32 into vector<1x4xf32>
        %117 = vector.insert %111, %116 [0, 2] : f32 into vector<1x4xf32>
        %118 = vector.insert %114, %117 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %24 = frisk.to_threadTile %16#4 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x_14 = gpu.thread_id x
    %25 = arith.index_cast %thread_id_x_14 : index to i32
    %c0_i32_15 = arith.constant 0 : i32
    %c64_i32_16 = arith.constant 64 : i32
    %c16_i32_17 = arith.constant 16 : i32
    %26 = arith.andi %25, %c16_i32_17 : i32
    %27 = arith.cmpi ne, %26, %c0_i32_15 : i32
    %c32_i32_18 = arith.constant 32 : i32
    %28 = arith.andi %25, %c32_i32_18 : i32
    %29 = arith.cmpi ne, %28, %c0_i32_15 : i32
    %cst_19 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %30 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_19) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %24[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %24[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %24[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %24[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
        %100 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
        %101 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
        %102 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
        %103 = arith.select %27, %99, %100 : f32
        %shuffleResult, %valid = gpu.shuffle xor %103, %c16_i32_17, %c64_i32_16 {frisk.layout_role = "wmmaC"} : f32
        %104 = arith.select %27, %shuffleResult, %99 : f32
        %105 = arith.select %27, %100, %shuffleResult : f32
        %106 = arith.select %27, %101, %102 : f32
        %shuffleResult_57, %valid_58 = gpu.shuffle xor %106, %c16_i32_17, %c64_i32_16 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %27, %shuffleResult_57, %101 : f32
        %108 = arith.select %27, %102, %shuffleResult_57 : f32
        %109 = arith.select %29, %104, %107 : f32
        %shuffleResult_59, %valid_60 = gpu.shuffle xor %109, %c32_i32_18, %c64_i32_16 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %29, %shuffleResult_59, %104 : f32
        %111 = arith.select %29, %107, %shuffleResult_59 : f32
        %112 = arith.select %29, %105, %108 : f32
        %shuffleResult_61, %valid_62 = gpu.shuffle xor %112, %c32_i32_18, %c64_i32_16 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %29, %shuffleResult_61, %105 : f32
        %114 = arith.select %29, %108, %shuffleResult_61 : f32
        %115 = vector.insert %110, %98 [0, 0] : f32 into vector<1x4xf32>
        %116 = vector.insert %113, %115 [0, 1] : f32 into vector<1x4xf32>
        %117 = vector.insert %111, %116 [0, 2] : f32 into vector<1x4xf32>
        %118 = vector.insert %114, %117 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %31 = frisk.to_threadTile %16#3 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x_20 = gpu.thread_id x
    %32 = arith.index_cast %thread_id_x_20 : index to i32
    %c0_i32_21 = arith.constant 0 : i32
    %c64_i32_22 = arith.constant 64 : i32
    %c16_i32_23 = arith.constant 16 : i32
    %33 = arith.andi %32, %c16_i32_23 : i32
    %34 = arith.cmpi ne, %33, %c0_i32_21 : i32
    %c32_i32_24 = arith.constant 32 : i32
    %35 = arith.andi %32, %c32_i32_24 : i32
    %36 = arith.cmpi ne, %35, %c0_i32_21 : i32
    %cst_25 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %37 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_25) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %31[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %31[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %31[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %31[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
        %100 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
        %101 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
        %102 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
        %103 = arith.select %34, %99, %100 : f32
        %shuffleResult, %valid = gpu.shuffle xor %103, %c16_i32_23, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
        %104 = arith.select %34, %shuffleResult, %99 : f32
        %105 = arith.select %34, %100, %shuffleResult : f32
        %106 = arith.select %34, %101, %102 : f32
        %shuffleResult_57, %valid_58 = gpu.shuffle xor %106, %c16_i32_23, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %34, %shuffleResult_57, %101 : f32
        %108 = arith.select %34, %102, %shuffleResult_57 : f32
        %109 = arith.select %36, %104, %107 : f32
        %shuffleResult_59, %valid_60 = gpu.shuffle xor %109, %c32_i32_24, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %36, %shuffleResult_59, %104 : f32
        %111 = arith.select %36, %107, %shuffleResult_59 : f32
        %112 = arith.select %36, %105, %108 : f32
        %shuffleResult_61, %valid_62 = gpu.shuffle xor %112, %c32_i32_24, %c64_i32_22 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %36, %shuffleResult_61, %105 : f32
        %114 = arith.select %36, %108, %shuffleResult_61 : f32
        %115 = vector.insert %110, %98 [0, 0] : f32 into vector<1x4xf32>
        %116 = vector.insert %113, %115 [0, 1] : f32 into vector<1x4xf32>
        %117 = vector.insert %111, %116 [0, 2] : f32 into vector<1x4xf32>
        %118 = vector.insert %114, %117 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %38 = frisk.to_threadTile %16#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x_26 = gpu.thread_id x
    %39 = arith.index_cast %thread_id_x_26 : index to i32
    %c0_i32_27 = arith.constant 0 : i32
    %c64_i32_28 = arith.constant 64 : i32
    %c16_i32_29 = arith.constant 16 : i32
    %40 = arith.andi %39, %c16_i32_29 : i32
    %41 = arith.cmpi ne, %40, %c0_i32_27 : i32
    %c32_i32_30 = arith.constant 32 : i32
    %42 = arith.andi %39, %c32_i32_30 : i32
    %43 = arith.cmpi ne, %42, %c0_i32_27 : i32
    %cst_31 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %44 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_31) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %38[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %38[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %38[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %38[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
        %100 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
        %101 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
        %102 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
        %103 = arith.select %41, %99, %100 : f32
        %shuffleResult, %valid = gpu.shuffle xor %103, %c16_i32_29, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
        %104 = arith.select %41, %shuffleResult, %99 : f32
        %105 = arith.select %41, %100, %shuffleResult : f32
        %106 = arith.select %41, %101, %102 : f32
        %shuffleResult_57, %valid_58 = gpu.shuffle xor %106, %c16_i32_29, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %41, %shuffleResult_57, %101 : f32
        %108 = arith.select %41, %102, %shuffleResult_57 : f32
        %109 = arith.select %43, %104, %107 : f32
        %shuffleResult_59, %valid_60 = gpu.shuffle xor %109, %c32_i32_30, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %43, %shuffleResult_59, %104 : f32
        %111 = arith.select %43, %107, %shuffleResult_59 : f32
        %112 = arith.select %43, %105, %108 : f32
        %shuffleResult_61, %valid_62 = gpu.shuffle xor %112, %c32_i32_30, %c64_i32_28 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %43, %shuffleResult_61, %105 : f32
        %114 = arith.select %43, %108, %shuffleResult_61 : f32
        %115 = vector.insert %110, %98 [0, 0] : f32 into vector<1x4xf32>
        %116 = vector.insert %113, %115 [0, 1] : f32 into vector<1x4xf32>
        %117 = vector.insert %111, %116 [0, 2] : f32 into vector<1x4xf32>
        %118 = vector.insert %114, %117 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %45 = frisk.to_threadTile %16#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_32 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %46 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_32) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %88 = vector.extract %45[%75, %c0] : f32 from vector<2x1xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %c0_57 = arith.constant 0 : index
        %90 = vector.extract %45[%75, %c0_57] : f32 from vector<2x1xf32>
        %91 = vector.insert %90, %89 [0, 1] : f32 into vector<1x4xf32>
        %c0_58 = arith.constant 0 : index
        %92 = vector.extract %45[%75, %c0_58] : f32 from vector<2x1xf32>
        %93 = vector.insert %92, %91 [0, 2] : f32 into vector<1x4xf32>
        %c0_59 = arith.constant 0 : index
        %94 = vector.extract %45[%75, %c0_59] : f32 from vector<2x1xf32>
        %95 = vector.insert %94, %93 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %95 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_33 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %47 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_33) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %44[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %44[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %44[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %44[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %99 = vector.extract %46[%75, %74] : f32 from vector<2x8xf32>
        %100 = vector.insert %99, %cst_57 [0, 0] : f32 into vector<1x4xf32>
        %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %102 = vector.extract %46[%75, %101] : f32 from vector<2x8xf32>
        %103 = vector.insert %102, %100 [0, 1] : f32 into vector<1x4xf32>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %105 = vector.extract %46[%75, %104] : f32 from vector<2x8xf32>
        %106 = vector.insert %105, %103 [0, 2] : f32 into vector<1x4xf32>
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %108 = vector.extract %46[%75, %107] : f32 from vector<2x8xf32>
        %109 = vector.insert %108, %106 [0, 3] : f32 into vector<1x4xf32>
        %110 = arith.divf %98, %109 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %110 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_34 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %48 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_34) -> (vector<2x8xf16>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %47[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %47[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %47[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %47[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = arith.truncf %98 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %99 : vector<1x4xf16>
      }
      %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
      %78 = vector.insert %77, %arg5 [%75, %74] : f16 into vector<2x8xf16>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
      %81 = vector.insert %80, %78 [%75, %79] : f16 into vector<2x8xf16>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
      %84 = vector.insert %83, %81 [%75, %82] : f16 into vector<2x8xf16>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
      %87 = vector.insert %86, %84 [%75, %85] : f16 into vector<2x8xf16>
      affine.yield %87 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %49 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%8 * 16384) floordiv 524288) mod 32, (%8 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_35 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %50 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_35) -> (vector<2x8xf16>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %88 = vector.extract %48[%75, %74] : f16 from vector<2x8xf16>
        %89 = vector.insert %88, %cst_56 [0, 0] : f16 into vector<1x4xf16>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %48[%75, %90] : f16 from vector<2x8xf16>
        %92 = vector.insert %91, %89 [0, 1] : f16 into vector<1x4xf16>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %48[%75, %93] : f16 from vector<2x8xf16>
        %95 = vector.insert %94, %92 [0, 2] : f16 into vector<1x4xf16>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %48[%75, %96] : f16 from vector<2x8xf16>
        %98 = vector.insert %97, %95 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %98 : vector<1x4xf16>
      }
      %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
      %78 = vector.insert %77, %arg5 [%75, %74] : f16 into vector<2x8xf16>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
      %81 = vector.insert %80, %78 [%75, %79] : f16 into vector<2x8xf16>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
      %84 = vector.insert %83, %81 [%75, %82] : f16 into vector<2x8xf16>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
      %87 = vector.insert %86, %84 [%75, %85] : f16 into vector<2x8xf16>
      affine.yield %87 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_36 = gpu.thread_id x
    %51 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_36)
    %c4_37 = arith.constant 4 : index
    %52 = arith.cmpi ult, %51, %c4_37 : index
    scf.if %52 {
      affine.for %arg4 = 0 to 4 {
        %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %76 = vector.extract %50[%75, %74] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %77 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%74)
          %c0_57 = arith.constant 0 : index
          %78 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%74)
          %79 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%78)
          %c0_58 = arith.constant 0 : index
          %80 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_56, %79)
          %81 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_36, %75, %77, %c0, %c0_57, %c0_58, %80)
          %82 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_36, %75, %77, %c0, %c0_57, %c0_58, %80)
          affine.store %76, %49[%81, %82] {tiled = true} : memref<128x32xf16, 1>
          %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %84 = vector.extract %50[%75, %83] : f16 from vector<2x8xf16>
          %c0_59 = arith.constant 0 : index
          %c0_60 = arith.constant 0 : index
          %85 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%83)
          %c0_61 = arith.constant 0 : index
          %86 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%83)
          %87 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%86)
          %c0_62 = arith.constant 0 : index
          %88 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_60, %87)
          %89 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_36, %75, %85, %c0_59, %c0_61, %c0_62, %88)
          %90 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_36, %75, %85, %c0_59, %c0_61, %c0_62, %88)
          affine.store %84, %49[%89, %90] {tiled = true} : memref<128x32xf16, 1>
          %91 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %92 = vector.extract %50[%75, %91] : f16 from vector<2x8xf16>
          %c0_63 = arith.constant 0 : index
          %c0_64 = arith.constant 0 : index
          %93 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%91)
          %c0_65 = arith.constant 0 : index
          %94 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%91)
          %95 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%94)
          %c0_66 = arith.constant 0 : index
          %96 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_64, %95)
          %97 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_36, %75, %93, %c0_63, %c0_65, %c0_66, %96)
          %98 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_36, %75, %93, %c0_63, %c0_65, %c0_66, %96)
          affine.store %92, %49[%97, %98] {tiled = true} : memref<128x32xf16, 1>
          %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %100 = vector.extract %50[%75, %99] : f16 from vector<2x8xf16>
          %c0_67 = arith.constant 0 : index
          %c0_68 = arith.constant 0 : index
          %101 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%99)
          %c0_69 = arith.constant 0 : index
          %102 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%99)
          %103 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%102)
          %c0_70 = arith.constant 0 : index
          %104 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_68, %103)
          %105 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_36, %75, %101, %c0_67, %c0_69, %c0_70, %104)
          %106 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_36, %75, %101, %c0_67, %c0_69, %c0_70, %104)
          affine.store %100, %49[%105, %106] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_38 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %53 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_38) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %88 = vector.extract %45[%75, %c0] : f32 from vector<2x1xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %c0_57 = arith.constant 0 : index
        %90 = vector.extract %45[%75, %c0_57] : f32 from vector<2x1xf32>
        %91 = vector.insert %90, %89 [0, 1] : f32 into vector<1x4xf32>
        %c0_58 = arith.constant 0 : index
        %92 = vector.extract %45[%75, %c0_58] : f32 from vector<2x1xf32>
        %93 = vector.insert %92, %91 [0, 2] : f32 into vector<1x4xf32>
        %c0_59 = arith.constant 0 : index
        %94 = vector.extract %45[%75, %c0_59] : f32 from vector<2x1xf32>
        %95 = vector.insert %94, %93 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %95 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_39 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %54 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_39) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %37[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %37[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %37[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %37[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %99 = vector.extract %53[%75, %74] : f32 from vector<2x8xf32>
        %100 = vector.insert %99, %cst_57 [0, 0] : f32 into vector<1x4xf32>
        %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %102 = vector.extract %53[%75, %101] : f32 from vector<2x8xf32>
        %103 = vector.insert %102, %100 [0, 1] : f32 into vector<1x4xf32>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %105 = vector.extract %53[%75, %104] : f32 from vector<2x8xf32>
        %106 = vector.insert %105, %103 [0, 2] : f32 into vector<1x4xf32>
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %108 = vector.extract %53[%75, %107] : f32 from vector<2x8xf32>
        %109 = vector.insert %108, %106 [0, 3] : f32 into vector<1x4xf32>
        %110 = arith.divf %98, %109 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %110 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_40 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %55 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_40) -> (vector<2x8xf16>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %54[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %54[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %54[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %54[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = arith.truncf %98 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %99 : vector<1x4xf16>
      }
      %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
      %78 = vector.insert %77, %arg5 [%75, %74] : f16 into vector<2x8xf16>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
      %81 = vector.insert %80, %78 [%75, %79] : f16 into vector<2x8xf16>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
      %84 = vector.insert %83, %81 [%75, %82] : f16 into vector<2x8xf16>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
      %87 = vector.insert %86, %84 [%75, %85] : f16 into vector<2x8xf16>
      affine.yield %87 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %56 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + 32) floordiv 524288) mod 32, (%8 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_41 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %57 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_41) -> (vector<2x8xf16>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %88 = vector.extract %55[%75, %74] : f16 from vector<2x8xf16>
        %89 = vector.insert %88, %cst_56 [0, 0] : f16 into vector<1x4xf16>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %55[%75, %90] : f16 from vector<2x8xf16>
        %92 = vector.insert %91, %89 [0, 1] : f16 into vector<1x4xf16>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %55[%75, %93] : f16 from vector<2x8xf16>
        %95 = vector.insert %94, %92 [0, 2] : f16 into vector<1x4xf16>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %55[%75, %96] : f16 from vector<2x8xf16>
        %98 = vector.insert %97, %95 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %98 : vector<1x4xf16>
      }
      %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
      %78 = vector.insert %77, %arg5 [%75, %74] : f16 into vector<2x8xf16>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
      %81 = vector.insert %80, %78 [%75, %79] : f16 into vector<2x8xf16>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
      %84 = vector.insert %83, %81 [%75, %82] : f16 into vector<2x8xf16>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
      %87 = vector.insert %86, %84 [%75, %85] : f16 into vector<2x8xf16>
      affine.yield %87 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_42 = gpu.thread_id x
    %58 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_42)
    %c4_43 = arith.constant 4 : index
    %59 = arith.cmpi ult, %58, %c4_43 : index
    scf.if %59 {
      affine.for %arg4 = 0 to 4 {
        %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %76 = vector.extract %57[%75, %74] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %77 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%74)
          %c0_57 = arith.constant 0 : index
          %78 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%74)
          %79 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%78)
          %c0_58 = arith.constant 0 : index
          %80 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_56, %79)
          %81 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_42, %75, %77, %c0, %c0_57, %c0_58, %80)
          %82 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_42, %75, %77, %c0, %c0_57, %c0_58, %80)
          affine.store %76, %56[%81, %82] {tiled = true} : memref<128x32xf16, 1>
          %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %84 = vector.extract %57[%75, %83] : f16 from vector<2x8xf16>
          %c0_59 = arith.constant 0 : index
          %c0_60 = arith.constant 0 : index
          %85 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%83)
          %c0_61 = arith.constant 0 : index
          %86 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%83)
          %87 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%86)
          %c0_62 = arith.constant 0 : index
          %88 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_60, %87)
          %89 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_42, %75, %85, %c0_59, %c0_61, %c0_62, %88)
          %90 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_42, %75, %85, %c0_59, %c0_61, %c0_62, %88)
          affine.store %84, %56[%89, %90] {tiled = true} : memref<128x32xf16, 1>
          %91 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %92 = vector.extract %57[%75, %91] : f16 from vector<2x8xf16>
          %c0_63 = arith.constant 0 : index
          %c0_64 = arith.constant 0 : index
          %93 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%91)
          %c0_65 = arith.constant 0 : index
          %94 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%91)
          %95 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%94)
          %c0_66 = arith.constant 0 : index
          %96 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_64, %95)
          %97 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_42, %75, %93, %c0_63, %c0_65, %c0_66, %96)
          %98 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_42, %75, %93, %c0_63, %c0_65, %c0_66, %96)
          affine.store %92, %56[%97, %98] {tiled = true} : memref<128x32xf16, 1>
          %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %100 = vector.extract %57[%75, %99] : f16 from vector<2x8xf16>
          %c0_67 = arith.constant 0 : index
          %c0_68 = arith.constant 0 : index
          %101 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%99)
          %c0_69 = arith.constant 0 : index
          %102 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%99)
          %103 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%102)
          %c0_70 = arith.constant 0 : index
          %104 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_68, %103)
          %105 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_42, %75, %101, %c0_67, %c0_69, %c0_70, %104)
          %106 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_42, %75, %101, %c0_67, %c0_69, %c0_70, %104)
          affine.store %100, %56[%105, %106] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_44 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %60 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_44) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %88 = vector.extract %45[%75, %c0] : f32 from vector<2x1xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %c0_57 = arith.constant 0 : index
        %90 = vector.extract %45[%75, %c0_57] : f32 from vector<2x1xf32>
        %91 = vector.insert %90, %89 [0, 1] : f32 into vector<1x4xf32>
        %c0_58 = arith.constant 0 : index
        %92 = vector.extract %45[%75, %c0_58] : f32 from vector<2x1xf32>
        %93 = vector.insert %92, %91 [0, 2] : f32 into vector<1x4xf32>
        %c0_59 = arith.constant 0 : index
        %94 = vector.extract %45[%75, %c0_59] : f32 from vector<2x1xf32>
        %95 = vector.insert %94, %93 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %95 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_45 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %61 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_45) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %30[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %30[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %30[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %30[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %99 = vector.extract %60[%75, %74] : f32 from vector<2x8xf32>
        %100 = vector.insert %99, %cst_57 [0, 0] : f32 into vector<1x4xf32>
        %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %102 = vector.extract %60[%75, %101] : f32 from vector<2x8xf32>
        %103 = vector.insert %102, %100 [0, 1] : f32 into vector<1x4xf32>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %105 = vector.extract %60[%75, %104] : f32 from vector<2x8xf32>
        %106 = vector.insert %105, %103 [0, 2] : f32 into vector<1x4xf32>
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %108 = vector.extract %60[%75, %107] : f32 from vector<2x8xf32>
        %109 = vector.insert %108, %106 [0, 3] : f32 into vector<1x4xf32>
        %110 = arith.divf %98, %109 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %110 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_46 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %62 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_46) -> (vector<2x8xf16>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %61[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %61[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %61[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %61[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = arith.truncf %98 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %99 : vector<1x4xf16>
      }
      %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
      %78 = vector.insert %77, %arg5 [%75, %74] : f16 into vector<2x8xf16>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
      %81 = vector.insert %80, %78 [%75, %79] : f16 into vector<2x8xf16>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
      %84 = vector.insert %83, %81 [%75, %82] : f16 into vector<2x8xf16>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
      %87 = vector.insert %86, %84 [%75, %85] : f16 into vector<2x8xf16>
      affine.yield %87 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %63 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + 64) floordiv 524288) mod 32, (%8 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_47 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %64 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_47) -> (vector<2x8xf16>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %88 = vector.extract %62[%75, %74] : f16 from vector<2x8xf16>
        %89 = vector.insert %88, %cst_56 [0, 0] : f16 into vector<1x4xf16>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %62[%75, %90] : f16 from vector<2x8xf16>
        %92 = vector.insert %91, %89 [0, 1] : f16 into vector<1x4xf16>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %62[%75, %93] : f16 from vector<2x8xf16>
        %95 = vector.insert %94, %92 [0, 2] : f16 into vector<1x4xf16>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %62[%75, %96] : f16 from vector<2x8xf16>
        %98 = vector.insert %97, %95 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %98 : vector<1x4xf16>
      }
      %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
      %78 = vector.insert %77, %arg5 [%75, %74] : f16 into vector<2x8xf16>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
      %81 = vector.insert %80, %78 [%75, %79] : f16 into vector<2x8xf16>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
      %84 = vector.insert %83, %81 [%75, %82] : f16 into vector<2x8xf16>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
      %87 = vector.insert %86, %84 [%75, %85] : f16 into vector<2x8xf16>
      affine.yield %87 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_48 = gpu.thread_id x
    %65 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_48)
    %c4_49 = arith.constant 4 : index
    %66 = arith.cmpi ult, %65, %c4_49 : index
    scf.if %66 {
      affine.for %arg4 = 0 to 4 {
        %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %76 = vector.extract %64[%75, %74] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %77 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%74)
          %c0_57 = arith.constant 0 : index
          %78 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%74)
          %79 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%78)
          %c0_58 = arith.constant 0 : index
          %80 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_56, %79)
          %81 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_48, %75, %77, %c0, %c0_57, %c0_58, %80)
          %82 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_48, %75, %77, %c0, %c0_57, %c0_58, %80)
          affine.store %76, %63[%81, %82] {tiled = true} : memref<128x32xf16, 1>
          %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %84 = vector.extract %64[%75, %83] : f16 from vector<2x8xf16>
          %c0_59 = arith.constant 0 : index
          %c0_60 = arith.constant 0 : index
          %85 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%83)
          %c0_61 = arith.constant 0 : index
          %86 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%83)
          %87 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%86)
          %c0_62 = arith.constant 0 : index
          %88 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_60, %87)
          %89 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_48, %75, %85, %c0_59, %c0_61, %c0_62, %88)
          %90 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_48, %75, %85, %c0_59, %c0_61, %c0_62, %88)
          affine.store %84, %63[%89, %90] {tiled = true} : memref<128x32xf16, 1>
          %91 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %92 = vector.extract %64[%75, %91] : f16 from vector<2x8xf16>
          %c0_63 = arith.constant 0 : index
          %c0_64 = arith.constant 0 : index
          %93 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%91)
          %c0_65 = arith.constant 0 : index
          %94 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%91)
          %95 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%94)
          %c0_66 = arith.constant 0 : index
          %96 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_64, %95)
          %97 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_48, %75, %93, %c0_63, %c0_65, %c0_66, %96)
          %98 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_48, %75, %93, %c0_63, %c0_65, %c0_66, %96)
          affine.store %92, %63[%97, %98] {tiled = true} : memref<128x32xf16, 1>
          %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %100 = vector.extract %64[%75, %99] : f16 from vector<2x8xf16>
          %c0_67 = arith.constant 0 : index
          %c0_68 = arith.constant 0 : index
          %101 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%99)
          %c0_69 = arith.constant 0 : index
          %102 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%99)
          %103 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%102)
          %c0_70 = arith.constant 0 : index
          %104 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_68, %103)
          %105 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_48, %75, %101, %c0_67, %c0_69, %c0_70, %104)
          %106 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_48, %75, %101, %c0_67, %c0_69, %c0_70, %104)
          affine.store %100, %63[%105, %106] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_50 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %67 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_50) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %88 = vector.extract %45[%75, %c0] : f32 from vector<2x1xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %c0_57 = arith.constant 0 : index
        %90 = vector.extract %45[%75, %c0_57] : f32 from vector<2x1xf32>
        %91 = vector.insert %90, %89 [0, 1] : f32 into vector<1x4xf32>
        %c0_58 = arith.constant 0 : index
        %92 = vector.extract %45[%75, %c0_58] : f32 from vector<2x1xf32>
        %93 = vector.insert %92, %91 [0, 2] : f32 into vector<1x4xf32>
        %c0_59 = arith.constant 0 : index
        %94 = vector.extract %45[%75, %c0_59] : f32 from vector<2x1xf32>
        %95 = vector.insert %94, %93 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %95 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_51 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %68 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_51) -> (vector<2x8xf32>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %23[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %23[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %23[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %23[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %99 = vector.extract %67[%75, %74] : f32 from vector<2x8xf32>
        %100 = vector.insert %99, %cst_57 [0, 0] : f32 into vector<1x4xf32>
        %101 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %102 = vector.extract %67[%75, %101] : f32 from vector<2x8xf32>
        %103 = vector.insert %102, %100 [0, 1] : f32 into vector<1x4xf32>
        %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %105 = vector.extract %67[%75, %104] : f32 from vector<2x8xf32>
        %106 = vector.insert %105, %103 [0, 2] : f32 into vector<1x4xf32>
        %107 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %108 = vector.extract %67[%75, %107] : f32 from vector<2x8xf32>
        %109 = vector.insert %108, %106 [0, 3] : f32 into vector<1x4xf32>
        %110 = arith.divf %98, %109 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %110 : vector<1x4xf32>
      }
      %77 = vector.extract %76[0, 0] : f32 from vector<1x4xf32>
      %78 = vector.insert %77, %arg5 [%75, %74] : f32 into vector<2x8xf32>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f32 from vector<1x4xf32>
      %81 = vector.insert %80, %78 [%75, %79] : f32 into vector<2x8xf32>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f32 from vector<1x4xf32>
      %84 = vector.insert %83, %81 [%75, %82] : f32 into vector<2x8xf32>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f32 from vector<1x4xf32>
      %87 = vector.insert %86, %84 [%75, %85] : f32 into vector<2x8xf32>
      affine.yield %87 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_52 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %69 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_52) -> (vector<2x8xf16>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %68[%75, %74] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_56 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %68[%75, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %68[%75, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %68[%75, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = arith.truncf %98 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %99 : vector<1x4xf16>
      }
      %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
      %78 = vector.insert %77, %arg5 [%75, %74] : f16 into vector<2x8xf16>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
      %81 = vector.insert %80, %78 [%75, %79] : f16 into vector<2x8xf16>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
      %84 = vector.insert %83, %81 [%75, %82] : f16 into vector<2x8xf16>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
      %87 = vector.insert %86, %84 [%75, %85] : f16 into vector<2x8xf16>
      affine.yield %87 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %70 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + 96) floordiv 524288) mod 32, (%8 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_53 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %71 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_53) -> (vector<2x8xf16>) {
      %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %76 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %88 = vector.extract %69[%75, %74] : f16 from vector<2x8xf16>
        %89 = vector.insert %88, %cst_56 [0, 0] : f16 into vector<1x4xf16>
        %90 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %91 = vector.extract %69[%75, %90] : f16 from vector<2x8xf16>
        %92 = vector.insert %91, %89 [0, 1] : f16 into vector<1x4xf16>
        %93 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %94 = vector.extract %69[%75, %93] : f16 from vector<2x8xf16>
        %95 = vector.insert %94, %92 [0, 2] : f16 into vector<1x4xf16>
        %96 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %97 = vector.extract %69[%75, %96] : f16 from vector<2x8xf16>
        %98 = vector.insert %97, %95 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %98 : vector<1x4xf16>
      }
      %77 = vector.extract %76[0, 0] : f16 from vector<1x4xf16>
      %78 = vector.insert %77, %arg5 [%75, %74] : f16 into vector<2x8xf16>
      %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %80 = vector.extract %76[0, 1] : f16 from vector<1x4xf16>
      %81 = vector.insert %80, %78 [%75, %79] : f16 into vector<2x8xf16>
      %82 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %83 = vector.extract %76[0, 2] : f16 from vector<1x4xf16>
      %84 = vector.insert %83, %81 [%75, %82] : f16 into vector<2x8xf16>
      %85 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %86 = vector.extract %76[0, 3] : f16 from vector<1x4xf16>
      %87 = vector.insert %86, %84 [%75, %85] : f16 into vector<2x8xf16>
      affine.yield %87 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_54 = gpu.thread_id x
    %72 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_54)
    %c4_55 = arith.constant 4 : index
    %73 = arith.cmpi ult, %72, %c4_55 : index
    scf.if %73 {
      affine.for %arg4 = 0 to 4 {
        %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %75 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %76 = vector.extract %71[%75, %74] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_56 = arith.constant 0 : index
          %77 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%74)
          %c0_57 = arith.constant 0 : index
          %78 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%74)
          %79 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%78)
          %c0_58 = arith.constant 0 : index
          %80 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_56, %79)
          %81 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_54, %75, %77, %c0, %c0_57, %c0_58, %80)
          %82 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_54, %75, %77, %c0, %c0_57, %c0_58, %80)
          affine.store %76, %70[%81, %82] {tiled = true} : memref<128x32xf16, 1>
          %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %84 = vector.extract %71[%75, %83] : f16 from vector<2x8xf16>
          %c0_59 = arith.constant 0 : index
          %c0_60 = arith.constant 0 : index
          %85 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%83)
          %c0_61 = arith.constant 0 : index
          %86 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%83)
          %87 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%86)
          %c0_62 = arith.constant 0 : index
          %88 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_60, %87)
          %89 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_54, %75, %85, %c0_59, %c0_61, %c0_62, %88)
          %90 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_54, %75, %85, %c0_59, %c0_61, %c0_62, %88)
          affine.store %84, %70[%89, %90] {tiled = true} : memref<128x32xf16, 1>
          %91 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %92 = vector.extract %71[%75, %91] : f16 from vector<2x8xf16>
          %c0_63 = arith.constant 0 : index
          %c0_64 = arith.constant 0 : index
          %93 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%91)
          %c0_65 = arith.constant 0 : index
          %94 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%91)
          %95 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%94)
          %c0_66 = arith.constant 0 : index
          %96 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_64, %95)
          %97 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_54, %75, %93, %c0_63, %c0_65, %c0_66, %96)
          %98 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_54, %75, %93, %c0_63, %c0_65, %c0_66, %96)
          affine.store %92, %70[%97, %98] {tiled = true} : memref<128x32xf16, 1>
          %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %100 = vector.extract %71[%75, %99] : f16 from vector<2x8xf16>
          %c0_67 = arith.constant 0 : index
          %c0_68 = arith.constant 0 : index
          %101 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%99)
          %c0_69 = arith.constant 0 : index
          %102 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%99)
          %103 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%102)
          %c0_70 = arith.constant 0 : index
          %104 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_68, %103)
          %105 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_54, %75, %101, %c0_67, %c0_69, %c0_70, %104)
          %106 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_54, %75, %101, %c0_67, %c0_69, %c0_70, %104)
          affine.store %100, %70[%105, %106] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
