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
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 8, 0, 1, 1>, thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %c32 = arith.constant 32 : index
    %0 = arith.muli %block_id_y, %c32 : index
    %1 = arith.addi %block_id_x, %0 : index
    %c256 = arith.constant 256 : index
    %c8 = arith.constant 8 : index
    %2 = arith.divui %1, %c256 : index
    %3 = arith.muli %2, %c8 : index
    %c8_0 = arith.constant 8 : index
    %4 = arith.remui %1, %c256 : index
    %5 = arith.remui %4, %c8_0 : index
    %6 = arith.addi %3, %5 : index
    %7 = arith.divui %4, %c8_0 : index
    %c31 = arith.constant 31 : index
    %8 = arith.subi %c31, %6 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst_3 = arith.constant 0xFF800000 : f32
    %9 = arith.truncf %cst : f32 to f16
    %10 = arith.muli %8, %c128 : index
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_4) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %59 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %60 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_33 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
        frisk.fragment_yield %cst_33 : vector<1x1xf32>
      }
      %61 = vector.extract %60[0, 0] : f32 from vector<1x1xf32>
      %62 = vector.insert %61, %arg5 [%59, %c0] : f32 into vector<2x1xf32>
      affine.yield %62 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %12 = vector.broadcast %cst_3 : f32 to vector<2x1xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %13 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_5) -> (vector<2x1xf32>) {
      %c0 = arith.constant 0 : index
      %59 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg4)
      %60 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %63 = vector.extract %11[%59, %c0] : f32 from vector<2x1xf32>
        %64 = vector.insert %63, %cst_33 [0, 0] : f32 into vector<1x1xf32>
        %cst_34 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %65 = vector.extract %12[%59, %c0] : f32 from vector<2x1xf32>
        %66 = vector.insert %65, %cst_34 [0, 0] : f32 into vector<1x1xf32>
        %67 = arith.addf %64, %66 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        frisk.fragment_yield %67 : vector<1x1xf32>
      }
      %61 = vector.extract %60[0, 0] : f32 from vector<1x1xf32>
      %62 = vector.insert %61, %arg5 [%59, %c0] : f32 into vector<2x1xf32>
      affine.yield %62 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %14:5 = affine.for %arg4 = 0 to affine_map<()[s0] -> (s0 * 128 + 128)>()[%8] step 128 iter_args(%arg5 = %cst_6, %arg6 = %cst_7, %arg7 = %cst_8, %arg8 = %cst_9, %arg9 = %cst_10) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %cst_33 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %59 = affine.for %arg10 = 0 to 128 step 32 iter_args(%arg11 = %cst_33) -> (vector<128x128xf32>) {
        %alloc_130 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_131 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %161 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + %arg10) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %8 * 128 + %arg10 floordiv 128) mod 4096, %arg10 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %161 to %alloc_130 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %162 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg10 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg10 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_130 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %162 to %alloc_131 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %163 = frisk.to_threadTile %alloc_130 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
        %164 = vector.broadcast %9 : f16 to vector<2x8xf16>
        %cst_132 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
        %165 = affine.for %arg12 = 0 to 4 iter_args(%arg13 = %cst_132) -> (vector<2x8xf16>) {
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg12)
          %172 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg12)
          %173 = frisk.fragment "compute" -> (vector<1x4xf16>) {
            %cst_137 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %185 = vector.extract %163[%172, %171] : f16 from vector<2x8xf16>
            %186 = vector.insert %185, %cst_137 [0, 0] : f16 into vector<1x4xf16>
            %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg12)
            %188 = vector.extract %163[%172, %187] : f16 from vector<2x8xf16>
            %189 = vector.insert %188, %186 [0, 1] : f16 into vector<1x4xf16>
            %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg12)
            %191 = vector.extract %163[%172, %190] : f16 from vector<2x8xf16>
            %192 = vector.insert %191, %189 [0, 2] : f16 into vector<1x4xf16>
            %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg12)
            %194 = vector.extract %163[%172, %193] : f16 from vector<2x8xf16>
            %195 = vector.insert %194, %192 [0, 3] : f16 into vector<1x4xf16>
            %cst_138 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %196 = vector.extract %164[%172, %171] : f16 from vector<2x8xf16>
            %197 = vector.insert %196, %cst_138 [0, 0] : f16 into vector<1x4xf16>
            %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg12)
            %199 = vector.extract %164[%172, %198] : f16 from vector<2x8xf16>
            %200 = vector.insert %199, %197 [0, 1] : f16 into vector<1x4xf16>
            %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg12)
            %202 = vector.extract %164[%172, %201] : f16 from vector<2x8xf16>
            %203 = vector.insert %202, %200 [0, 2] : f16 into vector<1x4xf16>
            %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg12)
            %205 = vector.extract %164[%172, %204] : f16 from vector<2x8xf16>
            %206 = vector.insert %205, %203 [0, 3] : f16 into vector<1x4xf16>
            %207 = arith.mulf %195, %206 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
            frisk.fragment_yield %207 : vector<1x4xf16>
          }
          %174 = vector.extract %173[0, 0] : f16 from vector<1x4xf16>
          %175 = vector.insert %174, %arg13 [%172, %171] : f16 into vector<2x8xf16>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg12)
          %177 = vector.extract %173[0, 1] : f16 from vector<1x4xf16>
          %178 = vector.insert %177, %175 [%172, %176] : f16 into vector<2x8xf16>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg12)
          %180 = vector.extract %173[0, 2] : f16 from vector<1x4xf16>
          %181 = vector.insert %180, %178 [%172, %179] : f16 into vector<2x8xf16>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg12)
          %183 = vector.extract %173[0, 3] : f16 from vector<1x4xf16>
          %184 = vector.insert %183, %181 [%172, %182] : f16 into vector<2x8xf16>
          affine.yield %184 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        frisk.wait_buffer_ready %alloc_131 : memref<32x128xf16, 3>
        %cst_133 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %cst_134 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %166 = affine.for %arg12 = 0 to 16 iter_args(%arg13 = %cst_133) -> (vector<2x32xf32>) {
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg12)
          %172 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg12)
          %173 = affine.for %arg14 = 0 to 2 iter_args(%arg15 = %cst_134) -> (vector<1x4xf32>) {
            %185 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %188 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg12)
              %189 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
              %cst_137 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
              %190 = vector.extract %165[%188, %189] : f16 from vector<2x8xf16>
              %191 = vector.insert %190, %cst_137 [0, 0] : f16 into vector<1x4xf16>
              %192 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
              %193 = vector.extract %165[%188, %192] : f16 from vector<2x8xf16>
              %194 = vector.insert %193, %191 [0, 1] : f16 into vector<1x4xf16>
              %195 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
              %196 = vector.extract %165[%188, %195] : f16 from vector<2x8xf16>
              %197 = vector.insert %196, %194 [0, 2] : f16 into vector<1x4xf16>
              %198 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
              %199 = vector.extract %165[%188, %198] : f16 from vector<2x8xf16>
              %200 = vector.insert %199, %197 [0, 3] : f16 into vector<1x4xf16>
              frisk.fragment_yield %200 : vector<1x4xf16>
            }
            %186 = frisk.fragment "load" -> (vector<4x1xf16>) {
              %188 = frisk.to_threadTile %alloc_131 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
              %189 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg14)
              %190 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg12)
              %cst_137 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
              %191 = vector.extract %188[%189, %190] : f16 from vector<8x8xf16>
              %192 = vector.insert %191, %cst_137 [0, 0] : f16 into vector<4x1xf16>
              %193 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg14)
              %194 = vector.extract %188[%193, %190] : f16 from vector<8x8xf16>
              %195 = vector.insert %194, %192 [1, 0] : f16 into vector<4x1xf16>
              %196 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg14)
              %197 = vector.extract %188[%196, %190] : f16 from vector<8x8xf16>
              %198 = vector.insert %197, %195 [2, 0] : f16 into vector<4x1xf16>
              %199 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg14)
              %200 = vector.extract %188[%199, %190] : f16 from vector<8x8xf16>
              %201 = vector.insert %200, %198 [3, 0] : f16 into vector<4x1xf16>
              frisk.fragment_yield %201 : vector<4x1xf16>
            }
            %187 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %188 = frisk.warp_mma_rr %185, %186, %arg15 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %188 : vector<1x4xf32>
            }
            affine.yield %187 : vector<1x4xf32>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %174 = vector.extract %173[0, 0] : f32 from vector<1x4xf32>
          %175 = vector.insert %174, %arg13 [%172, %171] : f32 into vector<2x32xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg12)
          %177 = vector.extract %173[0, 1] : f32 from vector<1x4xf32>
          %178 = vector.insert %177, %175 [%172, %176] : f32 into vector<2x32xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg12)
          %180 = vector.extract %173[0, 2] : f32 from vector<1x4xf32>
          %181 = vector.insert %180, %178 [%172, %179] : f32 into vector<2x32xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg12)
          %183 = vector.extract %173[0, 3] : f32 from vector<1x4xf32>
          %184 = vector.insert %183, %181 [%172, %182] : f32 into vector<2x32xf32>
          affine.yield %184 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %167 = frisk.to_threadTile %arg11 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
        %cst_135 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %168 = affine.for %arg12 = 0 to 16 iter_args(%arg13 = %cst_135) -> (vector<2x32xf32>) {
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg12)
          %172 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg12)
          %173 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_137 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %185 = vector.extract %167[%172, %171] : f32 from vector<2x32xf32>
            %186 = vector.insert %185, %cst_137 [0, 0] : f32 into vector<1x4xf32>
            %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg12)
            %188 = vector.extract %167[%172, %187] : f32 from vector<2x32xf32>
            %189 = vector.insert %188, %186 [0, 1] : f32 into vector<1x4xf32>
            %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg12)
            %191 = vector.extract %167[%172, %190] : f32 from vector<2x32xf32>
            %192 = vector.insert %191, %189 [0, 2] : f32 into vector<1x4xf32>
            %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg12)
            %194 = vector.extract %167[%172, %193] : f32 from vector<2x32xf32>
            %195 = vector.insert %194, %192 [0, 3] : f32 into vector<1x4xf32>
            %cst_138 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %196 = vector.extract %166[%172, %171] : f32 from vector<2x32xf32>
            %197 = vector.insert %196, %cst_138 [0, 0] : f32 into vector<1x4xf32>
            %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg12)
            %199 = vector.extract %166[%172, %198] : f32 from vector<2x32xf32>
            %200 = vector.insert %199, %197 [0, 1] : f32 into vector<1x4xf32>
            %201 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg12)
            %202 = vector.extract %166[%172, %201] : f32 from vector<2x32xf32>
            %203 = vector.insert %202, %200 [0, 2] : f32 into vector<1x4xf32>
            %204 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg12)
            %205 = vector.extract %166[%172, %204] : f32 from vector<2x32xf32>
            %206 = vector.insert %205, %203 [0, 3] : f32 into vector<1x4xf32>
            %207 = arith.addf %195, %206 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %207 : vector<1x4xf32>
          }
          %174 = vector.extract %173[0, 0] : f32 from vector<1x4xf32>
          %175 = vector.insert %174, %arg13 [%172, %171] : f32 into vector<2x32xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg12)
          %177 = vector.extract %173[0, 1] : f32 from vector<1x4xf32>
          %178 = vector.insert %177, %175 [%172, %176] : f32 into vector<2x32xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg12)
          %180 = vector.extract %173[0, 2] : f32 from vector<1x4xf32>
          %181 = vector.insert %180, %178 [%172, %179] : f32 into vector<2x32xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg12)
          %183 = vector.extract %173[0, 3] : f32 from vector<1x4xf32>
          %184 = vector.insert %183, %181 [%172, %182] : f32 into vector<2x32xf32>
          affine.yield %184 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %cst_136 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %169 = affine.for %arg12 = 0 to 16 iter_args(%arg13 = %cst_136) -> (vector<2x32xf32>) {
          %171 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg12)
          %172 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg12)
          %173 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_137 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %185 = vector.extract %168[%172, %171] : f32 from vector<2x32xf32>
            %186 = vector.insert %185, %cst_137 [0, 0] : f32 into vector<1x4xf32>
            %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg12)
            %188 = vector.extract %168[%172, %187] : f32 from vector<2x32xf32>
            %189 = vector.insert %188, %186 [0, 1] : f32 into vector<1x4xf32>
            %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg12)
            %191 = vector.extract %168[%172, %190] : f32 from vector<2x32xf32>
            %192 = vector.insert %191, %189 [0, 2] : f32 into vector<1x4xf32>
            %193 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg12)
            %194 = vector.extract %168[%172, %193] : f32 from vector<2x32xf32>
            %195 = vector.insert %194, %192 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
            frisk.fragment_yield %195 : vector<1x4xf32>
          }
          %174 = vector.extract %173[0, 0] : f32 from vector<1x4xf32>
          %175 = vector.insert %174, %arg13 [%172, %171] : f32 into vector<2x32xf32>
          %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg12)
          %177 = vector.extract %173[0, 1] : f32 from vector<1x4xf32>
          %178 = vector.insert %177, %175 [%172, %176] : f32 into vector<2x32xf32>
          %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg12)
          %180 = vector.extract %173[0, 2] : f32 from vector<1x4xf32>
          %181 = vector.insert %180, %178 [%172, %179] : f32 into vector<2x32xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg12)
          %183 = vector.extract %173[0, 3] : f32 from vector<1x4xf32>
          %184 = vector.insert %183, %181 [%172, %182] : f32 into vector<2x32xf32>
          affine.yield %184 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %170 = frisk.from_threadTile %169 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
        affine.yield %170 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %cst_34 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %thread_id_x_35 = gpu.thread_id x
      %c127 = arith.constant 127 : index
      %60 = arith.addi %10, %c127 : index
      %c127_36 = arith.constant 127 : index
      %61 = arith.addi %arg4, %c127_36 : index
      %c1_37 = arith.constant 1 : index
      %62 = arith.addi %10, %c1_37 : index
      %c1_38 = arith.constant 1 : index
      %63 = arith.addi %60, %c1_38 : index
      %64 = arith.cmpi uge, %63, %60 : index
      %65 = arith.cmpi uge, %60, %10 : index
      %66 = arith.andi %65, %64 : i1
      %67 = arith.cmpi uge, %61, %arg4 : index
      %68 = arith.andi %66, %67 : i1
      %69 = arith.cmpi ugt, %62, %61 : index
      %70 = arith.andi %68, %69 : i1
      %cst_39 = arith.constant 0.000000e+00 : f32
      %71 = scf.if %70 -> (vector<2x32xf32>) {
        %161 = vector.broadcast %cst_39 : f32 to vector<2x32xf32>
        scf.yield %161 : vector<2x32xf32>
      } else {
        %161 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_34) -> (vector<2x32xf32>) {
          %162 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
          %163 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
          %164 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %c0 = arith.constant 0 : index
            %c0_131 = arith.constant 0 : index
            %176 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%162)
            %c0_132 = arith.constant 0 : index
            %177 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%162)
            %178 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_131, %177)
            %c0_133 = arith.constant 0 : index
            %179 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_35, %163, %176, %c0, %c0_132, %178, %c0_133)
            %180 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_35, %163, %176, %c0, %c0_132, %178, %c0_133)
            %181 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %179)
            %182 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %180)
            %183 = arith.addi %181, %c1 : index
            %184 = arith.cmpi ule, %183, %182 : index
            %185 = arith.select %184, %cst_1, %cst_2 : f32
            %186 = vector.insert %185, %cst_130 [0, 0] : f32 into vector<1x4xf32>
            %187 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
            %c0_134 = arith.constant 0 : index
            %c0_135 = arith.constant 0 : index
            %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%187)
            %c0_136 = arith.constant 0 : index
            %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%187)
            %190 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_135, %189)
            %c0_137 = arith.constant 0 : index
            %191 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_35, %163, %188, %c0_134, %c0_136, %190, %c0_137)
            %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_35, %163, %188, %c0_134, %c0_136, %190, %c0_137)
            %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %191)
            %194 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %192)
            %195 = arith.addi %193, %c1 : index
            %196 = arith.cmpi ule, %195, %194 : index
            %197 = arith.select %196, %cst_1, %cst_2 : f32
            %198 = vector.insert %197, %186 [0, 1] : f32 into vector<1x4xf32>
            %199 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
            %c0_138 = arith.constant 0 : index
            %c0_139 = arith.constant 0 : index
            %200 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%199)
            %c0_140 = arith.constant 0 : index
            %201 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%199)
            %202 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_139, %201)
            %c0_141 = arith.constant 0 : index
            %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_35, %163, %200, %c0_138, %c0_140, %202, %c0_141)
            %204 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_35, %163, %200, %c0_138, %c0_140, %202, %c0_141)
            %205 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %203)
            %206 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %204)
            %207 = arith.addi %205, %c1 : index
            %208 = arith.cmpi ule, %207, %206 : index
            %209 = arith.select %208, %cst_1, %cst_2 : f32
            %210 = vector.insert %209, %198 [0, 2] : f32 into vector<1x4xf32>
            %211 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
            %c0_142 = arith.constant 0 : index
            %c0_143 = arith.constant 0 : index
            %212 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%211)
            %c0_144 = arith.constant 0 : index
            %213 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%211)
            %214 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_143, %213)
            %c0_145 = arith.constant 0 : index
            %215 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_35, %163, %212, %c0_142, %c0_144, %214, %c0_145)
            %216 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_35, %163, %212, %c0_142, %c0_144, %214, %c0_145)
            %217 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %215)
            %218 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %216)
            %219 = arith.addi %217, %c1 : index
            %220 = arith.cmpi ule, %219, %218 : index
            %221 = arith.select %220, %cst_1, %cst_2 : f32
            %222 = vector.insert %221, %210 [0, 3] : f32 into vector<1x4xf32>
            frisk.fragment_yield %222 : vector<1x4xf32>
          }
          %165 = vector.extract %164[0, 0] : f32 from vector<1x4xf32>
          %166 = vector.insert %165, %arg11 [%163, %162] : f32 into vector<2x32xf32>
          %167 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %168 = vector.extract %164[0, 1] : f32 from vector<1x4xf32>
          %169 = vector.insert %168, %166 [%163, %167] : f32 into vector<2x32xf32>
          %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %171 = vector.extract %164[0, 2] : f32 from vector<1x4xf32>
          %172 = vector.insert %171, %169 [%163, %170] : f32 into vector<2x32xf32>
          %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %174 = vector.extract %164[0, 3] : f32 from vector<1x4xf32>
          %175 = vector.insert %174, %172 [%163, %173] : f32 into vector<2x32xf32>
          affine.yield %175 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "mask_fragment", tiled = true}
        scf.yield %161 : vector<2x32xf32>
      } {frisk.uniform_mask}
      %72 = frisk.to_threadTile %59 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_40 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %73 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_40) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %72[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %72[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %72[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %72[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %71[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %71[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %71[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %71[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.addf %185, %196 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_41 = arith.constant 0xFF800000 : f32
      %cst_42 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %74 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_42) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %162 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %165 = affine.for %arg12 = 0 to 8 iter_args(%arg13 = %cst_41) -> (f32) {
            %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg12)
            %170 = frisk.fragment "compute" -> (f32) {
              %171 = vector.extract %73[%161, %169] : f32 from vector<2x32xf32>
              %172 = arith.maxnumf %arg13, %171 : f32
              %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg12)
              %174 = vector.extract %73[%161, %173] : f32 from vector<2x32xf32>
              %175 = arith.maxnumf %172, %174 : f32
              %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg12)
              %177 = vector.extract %73[%161, %176] : f32 from vector<2x32xf32>
              %178 = arith.maxnumf %175, %177 : f32
              %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg12)
              %180 = vector.extract %73[%161, %179] : f32 from vector<2x32xf32>
              %181 = arith.maxnumf %178, %180 : f32
              frisk.fragment_yield %181 : f32
            }
            affine.yield %170 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_131 = arith.constant 64 : i32
          %c16_i32_132 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %165, %c16_i32_132, %c64_i32_131 : f32
          %166 = arith.maxnumf %165, %shuffleResult : f32
          %c64_i32_133 = arith.constant 64 : i32
          %c32_i32_134 = arith.constant 32 : i32
          %shuffleResult_135, %valid_136 = gpu.shuffle xor %166, %c32_i32_134, %c64_i32_133 : f32
          %167 = arith.maxnumf %166, %shuffleResult_135 : f32
          %168 = vector.insert %167, %cst_130 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %168 : vector<1x1xf32>
        }
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %cst_43 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %75 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_43) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %162 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %165 = vector.extract %74[%161, %c0] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %cst_130 [0, 0] : f32 into vector<1x1xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %167 = vector.extract %13[%161, %c0] : f32 from vector<2x1xf32>
          %168 = vector.insert %167, %cst_131 [0, 0] : f32 into vector<1x1xf32>
          %169 = arith.maxnumf %166, %168 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %169 : vector<1x1xf32>
        }
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_44 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %76 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_44) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %175 = vector.extract %75[%162, %c0] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %c0_131 = arith.constant 0 : index
          %177 = vector.extract %75[%162, %c0_131] : f32 from vector<2x1xf32>
          %178 = vector.insert %177, %176 [0, 1] : f32 into vector<1x4xf32>
          %c0_132 = arith.constant 0 : index
          %179 = vector.extract %75[%162, %c0_132] : f32 from vector<2x1xf32>
          %180 = vector.insert %179, %178 [0, 2] : f32 into vector<1x4xf32>
          %c0_133 = arith.constant 0 : index
          %181 = vector.extract %75[%162, %c0_133] : f32 from vector<2x1xf32>
          %182 = vector.insert %181, %180 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %182 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %cst_45 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %77 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_45) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %73[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %73[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %73[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %73[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %76[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %76[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %76[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %76[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.subf %185, %196 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_46 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %78 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_46) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %162 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %165 = vector.extract %13[%161, %c0] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %cst_130 [0, 0] : f32 into vector<1x1xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %167 = vector.extract %75[%161, %c0] : f32 from vector<2x1xf32>
          %168 = vector.insert %167, %cst_131 [0, 0] : f32 into vector<1x1xf32>
          %169 = arith.subf %166, %168 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %169 : vector<1x1xf32>
        }
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_47 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %79 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_47) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %162 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %165 = vector.extract %78[%161, %c0] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %cst_130 [0, 0] : f32 into vector<1x1xf32>
          %167 = math.exp2 %166 {tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %167 : vector<1x1xf32>
        }
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %80 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_48 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %81 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_48) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %162 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %165 = vector.extract %80[%161, %c0] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %cst_130 [0, 0] : f32 into vector<1x1xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %167 = vector.extract %79[%161, %c0] : f32 from vector<2x1xf32>
          %168 = vector.insert %167, %cst_131 [0, 0] : f32 into vector<1x1xf32>
          %169 = arith.mulf %166, %168 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %169 : vector<1x1xf32>
        }
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_49 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %82 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_49) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %77[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %77[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %77[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %77[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %186 = math.exp2 %185 {tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %186 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_50 = arith.constant 0.000000e+00 : f32
      %cst_51 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %83 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_51) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %162 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %165 = affine.for %arg12 = 0 to 8 iter_args(%arg13 = %cst_50) -> (f32) {
            %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg12)
            %170 = frisk.fragment "compute" -> (f32) {
              %171 = vector.extract %82[%161, %169] : f32 from vector<2x32xf32>
              %172 = arith.addf %arg13, %171 : f32
              %173 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg12)
              %174 = vector.extract %82[%161, %173] : f32 from vector<2x32xf32>
              %175 = arith.addf %172, %174 : f32
              %176 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg12)
              %177 = vector.extract %82[%161, %176] : f32 from vector<2x32xf32>
              %178 = arith.addf %175, %177 : f32
              %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg12)
              %180 = vector.extract %82[%161, %179] : f32 from vector<2x32xf32>
              %181 = arith.addf %178, %180 : f32
              frisk.fragment_yield %181 : f32
            }
            affine.yield %170 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_131 = arith.constant 64 : i32
          %c16_i32_132 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %165, %c16_i32_132, %c64_i32_131 : f32
          %166 = arith.addf %165, %shuffleResult : f32
          %c64_i32_133 = arith.constant 64 : i32
          %c32_i32_134 = arith.constant 32 : i32
          %shuffleResult_135, %valid_136 = gpu.shuffle xor %166, %c32_i32_134, %c64_i32_133 : f32
          %167 = arith.addf %166, %shuffleResult_135 : f32
          %168 = vector.insert %167, %cst_130 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %168 : vector<1x1xf32>
        }
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %cst_52 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %84 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_52) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %162 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %165 = vector.extract %83[%161, %c0] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %cst_130 [0, 0] : f32 into vector<1x1xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %167 = vector.extract %81[%161, %c0] : f32 from vector<2x1xf32>
          %168 = vector.insert %167, %cst_131 [0, 0] : f32 into vector<1x1xf32>
          %169 = arith.addf %166, %168 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %169 : vector<1x1xf32>
        }
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_53 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
      %85 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_53) -> (vector<2x32xf16>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %82[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %82[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %82[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %82[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %186 = arith.truncf %185 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %186 : vector<1x4xf16>
        }
        %164 = vector.extract %163[0, 0] : f16 from vector<1x4xf16>
        %165 = vector.insert %164, %arg11 [%162, %161] : f16 into vector<2x32xf16>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f16 from vector<1x4xf16>
        %168 = vector.insert %167, %165 [%162, %166] : f16 into vector<2x32xf16>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f16 from vector<1x4xf16>
        %171 = vector.insert %170, %168 [%162, %169] : f16 into vector<2x32xf16>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f16 from vector<1x4xf16>
        %174 = vector.insert %173, %171 [%162, %172] : f16 into vector<2x32xf16>
        affine.yield %174 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %86 = frisk.from_threadTile %85 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x128xf16, 3>
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %87 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %87 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_54 = gpu.thread_id x
      %cst_55 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %88 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_55) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %189 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg10)[%thread_id_x_54]
          %190 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_54]
          %191 = frisk.read_matrix_from_shm %alloc[%189, %190] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %192 = vector.bitcast %191 : vector<4xi32> to vector<8xf16>
          %cst_130 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %193 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_54]
          %194 = vector.extract %192[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %cst_130 [0, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_54]
          %197 = vector.extract %192[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [0, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg10)[%thread_id_x_54]
          %200 = vector.extract %192[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [1, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg10)[%thread_id_x_54]
          %203 = vector.extract %192[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [1, 1] : f16 into vector<4x2xf16>
          %205 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg10)[%thread_id_x_54]
          %206 = vector.extract %192[%205] : f16 from vector<8xf16>
          %207 = vector.insert %206, %204 [2, 0] : f16 into vector<4x2xf16>
          %208 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg10)[%thread_id_x_54]
          %209 = vector.extract %192[%208] : f16 from vector<8xf16>
          %210 = vector.insert %209, %207 [2, 1] : f16 into vector<4x2xf16>
          %211 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg10)[%thread_id_x_54]
          %212 = vector.extract %192[%211] : f16 from vector<8xf16>
          %213 = vector.insert %212, %210 [3, 0] : f16 into vector<4x2xf16>
          %214 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg10)[%thread_id_x_54]
          %215 = vector.extract %192[%214] : f16 from vector<8xf16>
          %216 = vector.insert %215, %213 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %216 : vector<4x2xf16>
        }
        %163 = vector.extract %162[0, 0] : f16 from vector<4x2xf16>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f16 into vector<32x2xf16>
        %165 = affine.apply affine_map<() -> (1)>()
        %166 = vector.extract %162[0, 1] : f16 from vector<4x2xf16>
        %167 = vector.insert %166, %164 [%161, %165] : f16 into vector<32x2xf16>
        %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %169 = vector.extract %162[1, 0] : f16 from vector<4x2xf16>
        %170 = vector.insert %169, %167 [%168, %c0] : f16 into vector<32x2xf16>
        %171 = affine.apply affine_map<() -> (1)>()
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %173 = vector.extract %162[1, 1] : f16 from vector<4x2xf16>
        %174 = vector.insert %173, %170 [%172, %171] : f16 into vector<32x2xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %176 = vector.extract %162[2, 0] : f16 from vector<4x2xf16>
        %177 = vector.insert %176, %174 [%175, %c0] : f16 into vector<32x2xf16>
        %178 = affine.apply affine_map<() -> (1)>()
        %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %180 = vector.extract %162[2, 1] : f16 from vector<4x2xf16>
        %181 = vector.insert %180, %177 [%179, %178] : f16 into vector<32x2xf16>
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %183 = vector.extract %162[3, 0] : f16 from vector<4x2xf16>
        %184 = vector.insert %183, %181 [%182, %c0] : f16 into vector<32x2xf16>
        %185 = affine.apply affine_map<() -> (1)>()
        %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %187 = vector.extract %162[3, 1] : f16 from vector<4x2xf16>
        %188 = vector.insert %187, %184 [%186, %185] : f16 into vector<32x2xf16>
        affine.yield %188 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_56 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %89 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_56) -> (vector<2x8xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
        %163 = affine.for %arg12 = 0 to 8 iter_args(%arg13 = %cst_57) -> (vector<1x4xf32>) {
          %175 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %178 = frisk.to_threadTile %86 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %179 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
            %180 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg12)
            %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %181 = vector.extract %178[%179, %180] : f16 from vector<2x32xf16>
            %182 = vector.insert %181, %cst_130 [0, 0] : f16 into vector<1x4xf16>
            %183 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg12)
            %184 = vector.extract %178[%179, %183] : f16 from vector<2x32xf16>
            %185 = vector.insert %184, %182 [0, 1] : f16 into vector<1x4xf16>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg12)
            %187 = vector.extract %178[%179, %186] : f16 from vector<2x32xf16>
            %188 = vector.insert %187, %185 [0, 2] : f16 into vector<1x4xf16>
            %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg12)
            %190 = vector.extract %178[%179, %189] : f16 from vector<2x32xf16>
            %191 = vector.insert %190, %188 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %191 : vector<1x4xf16>
          }
          %176 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %178 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg12)
            %179 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
            %cst_130 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %180 = vector.extract %88[%178, %179] : f16 from vector<32x2xf16>
            %181 = vector.insert %180, %cst_130 [0, 0] : f16 into vector<4x1xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg12)
            %183 = vector.extract %88[%182, %179] : f16 from vector<32x2xf16>
            %184 = vector.insert %183, %181 [1, 0] : f16 into vector<4x1xf16>
            %185 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg12)
            %186 = vector.extract %88[%185, %179] : f16 from vector<32x2xf16>
            %187 = vector.insert %186, %184 [2, 0] : f16 into vector<4x1xf16>
            %188 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg12)
            %189 = vector.extract %88[%188, %179] : f16 from vector<32x2xf16>
            %190 = vector.insert %189, %187 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %190 : vector<4x1xf16>
          }
          %177 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %178 = frisk.warp_mma_rr %175, %176, %arg13 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %178 : vector<1x4xf32>
          }
          affine.yield %177 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x8xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x8xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x8xf32>
        affine.yield %174 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %thread_id_x_58 = gpu.thread_id x
      %90 = arith.index_cast %thread_id_x_58 : index to i32
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %c16_i32 = arith.constant 16 : i32
      %91 = arith.andi %90, %c16_i32 : i32
      %92 = arith.cmpi ne, %91, %c0_i32 : i32
      %c32_i32 = arith.constant 32 : i32
      %93 = arith.andi %90, %c32_i32 : i32
      %94 = arith.cmpi ne, %93, %c0_i32 : i32
      %cst_59 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %95 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_59) -> (vector<2x8xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %89[%162, %161] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
          %178 = vector.extract %89[%162, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
          %181 = vector.extract %89[%162, %180] : f32 from vector<2x8xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
          %184 = vector.extract %89[%162, %183] : f32 from vector<2x8xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %186 = vector.extract %185[0, 0] : f32 from vector<1x4xf32>
          %187 = vector.extract %185[0, 1] : f32 from vector<1x4xf32>
          %188 = vector.extract %185[0, 2] : f32 from vector<1x4xf32>
          %189 = vector.extract %185[0, 3] : f32 from vector<1x4xf32>
          %190 = arith.select %92, %186, %187 : f32
          %shuffleResult, %valid = gpu.shuffle xor %190, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %191 = arith.select %92, %shuffleResult, %186 : f32
          %192 = arith.select %92, %187, %shuffleResult : f32
          %193 = arith.select %92, %188, %189 : f32
          %shuffleResult_131, %valid_132 = gpu.shuffle xor %193, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %194 = arith.select %92, %shuffleResult_131, %188 : f32
          %195 = arith.select %92, %189, %shuffleResult_131 : f32
          %196 = arith.select %94, %191, %194 : f32
          %shuffleResult_133, %valid_134 = gpu.shuffle xor %196, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %197 = arith.select %94, %shuffleResult_133, %191 : f32
          %198 = arith.select %94, %194, %shuffleResult_133 : f32
          %199 = arith.select %94, %192, %195 : f32
          %shuffleResult_135, %valid_136 = gpu.shuffle xor %199, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %200 = arith.select %94, %shuffleResult_135, %192 : f32
          %201 = arith.select %94, %195, %shuffleResult_135 : f32
          %202 = vector.insert %197, %185 [0, 0] : f32 into vector<1x4xf32>
          %203 = vector.insert %200, %202 [0, 1] : f32 into vector<1x4xf32>
          %204 = vector.insert %198, %203 [0, 2] : f32 into vector<1x4xf32>
          %205 = vector.insert %201, %204 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %205 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x8xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x8xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x8xf32>
        affine.yield %174 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %96 = frisk.to_threadTile %arg5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x32xf32>
      %cst_60 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %97 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_60) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %175 = vector.extract %79[%162, %c0] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %c0_131 = arith.constant 0 : index
          %177 = vector.extract %79[%162, %c0_131] : f32 from vector<2x1xf32>
          %178 = vector.insert %177, %176 [0, 1] : f32 into vector<1x4xf32>
          %c0_132 = arith.constant 0 : index
          %179 = vector.extract %79[%162, %c0_132] : f32 from vector<2x1xf32>
          %180 = vector.insert %179, %178 [0, 2] : f32 into vector<1x4xf32>
          %c0_133 = arith.constant 0 : index
          %181 = vector.extract %79[%162, %c0_133] : f32 from vector<2x1xf32>
          %182 = vector.insert %181, %180 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %182 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %cst_61 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %98 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_61) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %96[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %96[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %96[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %96[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %97[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %97[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %97[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %97[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.mulf %185, %196 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_62 = memref.alloc() {tiled = true} : memref<128x32xf32, 3>
      %thread_id_x_63 = gpu.thread_id x
      %thread_id_x_64 = gpu.thread_id x
      %99 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_64)
      %c4_65 = arith.constant 4 : index
      %100 = arith.cmpi ult, %99, %c4_65 : index
      scf.if %100 {
        affine.for %arg10 = 0 to 4 {
          %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
          %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
          frisk.fragment "store" {
            %163 = vector.extract %95[%162, %161] : f32 from vector<2x8xf32>
            %c0 = arith.constant 0 : index
            %c0_130 = arith.constant 0 : index
            %164 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%161)
            %c0_131 = arith.constant 0 : index
            %165 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%161)
            %166 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%165)
            %c0_132 = arith.constant 0 : index
            %167 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_130, %166)
            %168 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_64, %162, %164, %c0, %c0_131, %c0_132, %167)
            %169 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_64, %162, %164, %c0, %c0_131, %c0_132, %167)
            affine.store %163, %alloc_62[%168, %169] {tiled = true} : memref<128x32xf32, 3>
            %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
            %171 = vector.extract %95[%162, %170] : f32 from vector<2x8xf32>
            %c0_133 = arith.constant 0 : index
            %c0_134 = arith.constant 0 : index
            %172 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%170)
            %c0_135 = arith.constant 0 : index
            %173 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%170)
            %174 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%173)
            %c0_136 = arith.constant 0 : index
            %175 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_134, %174)
            %176 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_64, %162, %172, %c0_133, %c0_135, %c0_136, %175)
            %177 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_64, %162, %172, %c0_133, %c0_135, %c0_136, %175)
            affine.store %171, %alloc_62[%176, %177] {tiled = true} : memref<128x32xf32, 3>
            %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
            %179 = vector.extract %95[%162, %178] : f32 from vector<2x8xf32>
            %c0_137 = arith.constant 0 : index
            %c0_138 = arith.constant 0 : index
            %180 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%178)
            %c0_139 = arith.constant 0 : index
            %181 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%178)
            %182 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%181)
            %c0_140 = arith.constant 0 : index
            %183 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_138, %182)
            %184 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_64, %162, %180, %c0_137, %c0_139, %c0_140, %183)
            %185 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_64, %162, %180, %c0_137, %c0_139, %c0_140, %183)
            affine.store %179, %alloc_62[%184, %185] {tiled = true} : memref<128x32xf32, 3>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
            %187 = vector.extract %95[%162, %186] : f32 from vector<2x8xf32>
            %c0_141 = arith.constant 0 : index
            %c0_142 = arith.constant 0 : index
            %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%186)
            %c0_143 = arith.constant 0 : index
            %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%186)
            %190 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%189)
            %c0_144 = arith.constant 0 : index
            %191 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_142, %190)
            %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_64, %162, %188, %c0_141, %c0_143, %c0_144, %191)
            %193 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_64, %162, %188, %c0_141, %c0_143, %c0_144, %191)
            affine.store %187, %alloc_62[%192, %193] {tiled = true} : memref<128x32xf32, 3>
            frisk.fragment_yield
          }
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
      }
      gpu.barrier {tiled = true}
      %cst_66 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %101 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_66) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "load" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %c0_131 = arith.constant 0 : index
          %175 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%161)
          %c0_132 = arith.constant 0 : index
          %176 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%161)
          %177 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_131, %176)
          %c0_133 = arith.constant 0 : index
          %178 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_63, %162, %175, %c0, %c0_132, %177, %c0_133)
          %179 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_63, %162, %175, %c0, %c0_132, %177, %c0_133)
          %180 = affine.load %alloc_62[%178, %179] : memref<128x32xf32, 3>
          %181 = vector.insert %180, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %c0_134 = arith.constant 0 : index
          %c0_135 = arith.constant 0 : index
          %183 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%182)
          %c0_136 = arith.constant 0 : index
          %184 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%182)
          %185 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_135, %184)
          %c0_137 = arith.constant 0 : index
          %186 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_63, %162, %183, %c0_134, %c0_136, %185, %c0_137)
          %187 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_63, %162, %183, %c0_134, %c0_136, %185, %c0_137)
          %188 = affine.load %alloc_62[%186, %187] : memref<128x32xf32, 3>
          %189 = vector.insert %188, %181 [0, 1] : f32 into vector<1x4xf32>
          %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %c0_138 = arith.constant 0 : index
          %c0_139 = arith.constant 0 : index
          %191 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%190)
          %c0_140 = arith.constant 0 : index
          %192 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%190)
          %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_139, %192)
          %c0_141 = arith.constant 0 : index
          %194 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_63, %162, %191, %c0_138, %c0_140, %193, %c0_141)
          %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_63, %162, %191, %c0_138, %c0_140, %193, %c0_141)
          %196 = affine.load %alloc_62[%194, %195] : memref<128x32xf32, 3>
          %197 = vector.insert %196, %189 [0, 2] : f32 into vector<1x4xf32>
          %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %c0_142 = arith.constant 0 : index
          %c0_143 = arith.constant 0 : index
          %199 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%198)
          %c0_144 = arith.constant 0 : index
          %200 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%198)
          %201 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_143, %200)
          %c0_145 = arith.constant 0 : index
          %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_63, %162, %199, %c0_142, %c0_144, %201, %c0_145)
          %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_63, %162, %199, %c0_142, %c0_144, %201, %c0_145)
          %204 = affine.load %alloc_62[%202, %203] : memref<128x32xf32, 3>
          %205 = vector.insert %204, %197 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %205 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "layout_fragment", tiled = true}
      frisk.sync_threads_in_block {tiled = true}
      %cst_67 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %102 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_67) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %98[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %98[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %98[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %98[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %101[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %101[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %101[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %101[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.addf %185, %196 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_68 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %103 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %103 to %alloc_68 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_68 : memref<128x32xf16, 3>
      %thread_id_x_69 = gpu.thread_id x
      %cst_70 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %104 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_70) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %189 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg10)[%thread_id_x_69]
          %190 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_69]
          %191 = frisk.read_matrix_from_shm %alloc_68[%189, %190] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %192 = vector.bitcast %191 : vector<4xi32> to vector<8xf16>
          %cst_130 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %193 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_69]
          %194 = vector.extract %192[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %cst_130 [0, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_69]
          %197 = vector.extract %192[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [0, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg10)[%thread_id_x_69]
          %200 = vector.extract %192[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [1, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg10)[%thread_id_x_69]
          %203 = vector.extract %192[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [1, 1] : f16 into vector<4x2xf16>
          %205 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg10)[%thread_id_x_69]
          %206 = vector.extract %192[%205] : f16 from vector<8xf16>
          %207 = vector.insert %206, %204 [2, 0] : f16 into vector<4x2xf16>
          %208 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg10)[%thread_id_x_69]
          %209 = vector.extract %192[%208] : f16 from vector<8xf16>
          %210 = vector.insert %209, %207 [2, 1] : f16 into vector<4x2xf16>
          %211 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg10)[%thread_id_x_69]
          %212 = vector.extract %192[%211] : f16 from vector<8xf16>
          %213 = vector.insert %212, %210 [3, 0] : f16 into vector<4x2xf16>
          %214 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg10)[%thread_id_x_69]
          %215 = vector.extract %192[%214] : f16 from vector<8xf16>
          %216 = vector.insert %215, %213 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %216 : vector<4x2xf16>
        }
        %163 = vector.extract %162[0, 0] : f16 from vector<4x2xf16>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f16 into vector<32x2xf16>
        %165 = affine.apply affine_map<() -> (1)>()
        %166 = vector.extract %162[0, 1] : f16 from vector<4x2xf16>
        %167 = vector.insert %166, %164 [%161, %165] : f16 into vector<32x2xf16>
        %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %169 = vector.extract %162[1, 0] : f16 from vector<4x2xf16>
        %170 = vector.insert %169, %167 [%168, %c0] : f16 into vector<32x2xf16>
        %171 = affine.apply affine_map<() -> (1)>()
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %173 = vector.extract %162[1, 1] : f16 from vector<4x2xf16>
        %174 = vector.insert %173, %170 [%172, %171] : f16 into vector<32x2xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %176 = vector.extract %162[2, 0] : f16 from vector<4x2xf16>
        %177 = vector.insert %176, %174 [%175, %c0] : f16 into vector<32x2xf16>
        %178 = affine.apply affine_map<() -> (1)>()
        %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %180 = vector.extract %162[2, 1] : f16 from vector<4x2xf16>
        %181 = vector.insert %180, %177 [%179, %178] : f16 into vector<32x2xf16>
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %183 = vector.extract %162[3, 0] : f16 from vector<4x2xf16>
        %184 = vector.insert %183, %181 [%182, %c0] : f16 into vector<32x2xf16>
        %185 = affine.apply affine_map<() -> (1)>()
        %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %187 = vector.extract %162[3, 1] : f16 from vector<4x2xf16>
        %188 = vector.insert %187, %184 [%186, %185] : f16 into vector<32x2xf16>
        affine.yield %188 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_71 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_72 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %105 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_71) -> (vector<2x8xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
        %163 = affine.for %arg12 = 0 to 8 iter_args(%arg13 = %cst_72) -> (vector<1x4xf32>) {
          %175 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %178 = frisk.to_threadTile %86 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %179 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
            %180 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg12)
            %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %181 = vector.extract %178[%179, %180] : f16 from vector<2x32xf16>
            %182 = vector.insert %181, %cst_130 [0, 0] : f16 into vector<1x4xf16>
            %183 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg12)
            %184 = vector.extract %178[%179, %183] : f16 from vector<2x32xf16>
            %185 = vector.insert %184, %182 [0, 1] : f16 into vector<1x4xf16>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg12)
            %187 = vector.extract %178[%179, %186] : f16 from vector<2x32xf16>
            %188 = vector.insert %187, %185 [0, 2] : f16 into vector<1x4xf16>
            %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg12)
            %190 = vector.extract %178[%179, %189] : f16 from vector<2x32xf16>
            %191 = vector.insert %190, %188 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %191 : vector<1x4xf16>
          }
          %176 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %178 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg12)
            %179 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
            %cst_130 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %180 = vector.extract %104[%178, %179] : f16 from vector<32x2xf16>
            %181 = vector.insert %180, %cst_130 [0, 0] : f16 into vector<4x1xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg12)
            %183 = vector.extract %104[%182, %179] : f16 from vector<32x2xf16>
            %184 = vector.insert %183, %181 [1, 0] : f16 into vector<4x1xf16>
            %185 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg12)
            %186 = vector.extract %104[%185, %179] : f16 from vector<32x2xf16>
            %187 = vector.insert %186, %184 [2, 0] : f16 into vector<4x1xf16>
            %188 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg12)
            %189 = vector.extract %104[%188, %179] : f16 from vector<32x2xf16>
            %190 = vector.insert %189, %187 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %190 : vector<4x1xf16>
          }
          %177 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %178 = frisk.warp_mma_rr %175, %176, %arg13 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %178 : vector<1x4xf32>
          }
          affine.yield %177 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x8xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x8xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x8xf32>
        affine.yield %174 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %thread_id_x_73 = gpu.thread_id x
      %106 = arith.index_cast %thread_id_x_73 : index to i32
      %c0_i32_74 = arith.constant 0 : i32
      %c64_i32_75 = arith.constant 64 : i32
      %c16_i32_76 = arith.constant 16 : i32
      %107 = arith.andi %106, %c16_i32_76 : i32
      %108 = arith.cmpi ne, %107, %c0_i32_74 : i32
      %c32_i32_77 = arith.constant 32 : i32
      %109 = arith.andi %106, %c32_i32_77 : i32
      %110 = arith.cmpi ne, %109, %c0_i32_74 : i32
      %cst_78 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %111 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_78) -> (vector<2x8xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %105[%162, %161] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
          %178 = vector.extract %105[%162, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
          %181 = vector.extract %105[%162, %180] : f32 from vector<2x8xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
          %184 = vector.extract %105[%162, %183] : f32 from vector<2x8xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %186 = vector.extract %185[0, 0] : f32 from vector<1x4xf32>
          %187 = vector.extract %185[0, 1] : f32 from vector<1x4xf32>
          %188 = vector.extract %185[0, 2] : f32 from vector<1x4xf32>
          %189 = vector.extract %185[0, 3] : f32 from vector<1x4xf32>
          %190 = arith.select %108, %186, %187 : f32
          %shuffleResult, %valid = gpu.shuffle xor %190, %c16_i32_76, %c64_i32_75 {frisk.layout_role = "wmmaC"} : f32
          %191 = arith.select %108, %shuffleResult, %186 : f32
          %192 = arith.select %108, %187, %shuffleResult : f32
          %193 = arith.select %108, %188, %189 : f32
          %shuffleResult_131, %valid_132 = gpu.shuffle xor %193, %c16_i32_76, %c64_i32_75 {frisk.layout_role = "wmmaC"} : f32
          %194 = arith.select %108, %shuffleResult_131, %188 : f32
          %195 = arith.select %108, %189, %shuffleResult_131 : f32
          %196 = arith.select %110, %191, %194 : f32
          %shuffleResult_133, %valid_134 = gpu.shuffle xor %196, %c32_i32_77, %c64_i32_75 {frisk.layout_role = "wmmaC"} : f32
          %197 = arith.select %110, %shuffleResult_133, %191 : f32
          %198 = arith.select %110, %194, %shuffleResult_133 : f32
          %199 = arith.select %110, %192, %195 : f32
          %shuffleResult_135, %valid_136 = gpu.shuffle xor %199, %c32_i32_77, %c64_i32_75 {frisk.layout_role = "wmmaC"} : f32
          %200 = arith.select %110, %shuffleResult_135, %192 : f32
          %201 = arith.select %110, %195, %shuffleResult_135 : f32
          %202 = vector.insert %197, %185 [0, 0] : f32 into vector<1x4xf32>
          %203 = vector.insert %200, %202 [0, 1] : f32 into vector<1x4xf32>
          %204 = vector.insert %198, %203 [0, 2] : f32 into vector<1x4xf32>
          %205 = vector.insert %201, %204 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %205 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x8xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x8xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x8xf32>
        affine.yield %174 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %112 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x32xf32>
      %cst_79 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %113 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_79) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %175 = vector.extract %79[%162, %c0] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %c0_131 = arith.constant 0 : index
          %177 = vector.extract %79[%162, %c0_131] : f32 from vector<2x1xf32>
          %178 = vector.insert %177, %176 [0, 1] : f32 into vector<1x4xf32>
          %c0_132 = arith.constant 0 : index
          %179 = vector.extract %79[%162, %c0_132] : f32 from vector<2x1xf32>
          %180 = vector.insert %179, %178 [0, 2] : f32 into vector<1x4xf32>
          %c0_133 = arith.constant 0 : index
          %181 = vector.extract %79[%162, %c0_133] : f32 from vector<2x1xf32>
          %182 = vector.insert %181, %180 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %182 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %cst_80 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %114 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_80) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %112[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %112[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %112[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %112[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %113[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %113[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %113[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %113[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.mulf %185, %196 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_81 = memref.alloc() {tiled = true} : memref<128x32xf32, 3>
      %thread_id_x_82 = gpu.thread_id x
      %thread_id_x_83 = gpu.thread_id x
      %115 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_83)
      %c4_84 = arith.constant 4 : index
      %116 = arith.cmpi ult, %115, %c4_84 : index
      scf.if %116 {
        affine.for %arg10 = 0 to 4 {
          %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
          %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
          frisk.fragment "store" {
            %163 = vector.extract %111[%162, %161] : f32 from vector<2x8xf32>
            %c0 = arith.constant 0 : index
            %c0_130 = arith.constant 0 : index
            %164 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%161)
            %c0_131 = arith.constant 0 : index
            %165 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%161)
            %166 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%165)
            %c0_132 = arith.constant 0 : index
            %167 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_130, %166)
            %168 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_83, %162, %164, %c0, %c0_131, %c0_132, %167)
            %169 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_83, %162, %164, %c0, %c0_131, %c0_132, %167)
            affine.store %163, %alloc_81[%168, %169] {tiled = true} : memref<128x32xf32, 3>
            %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
            %171 = vector.extract %111[%162, %170] : f32 from vector<2x8xf32>
            %c0_133 = arith.constant 0 : index
            %c0_134 = arith.constant 0 : index
            %172 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%170)
            %c0_135 = arith.constant 0 : index
            %173 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%170)
            %174 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%173)
            %c0_136 = arith.constant 0 : index
            %175 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_134, %174)
            %176 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_83, %162, %172, %c0_133, %c0_135, %c0_136, %175)
            %177 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_83, %162, %172, %c0_133, %c0_135, %c0_136, %175)
            affine.store %171, %alloc_81[%176, %177] {tiled = true} : memref<128x32xf32, 3>
            %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
            %179 = vector.extract %111[%162, %178] : f32 from vector<2x8xf32>
            %c0_137 = arith.constant 0 : index
            %c0_138 = arith.constant 0 : index
            %180 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%178)
            %c0_139 = arith.constant 0 : index
            %181 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%178)
            %182 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%181)
            %c0_140 = arith.constant 0 : index
            %183 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_138, %182)
            %184 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_83, %162, %180, %c0_137, %c0_139, %c0_140, %183)
            %185 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_83, %162, %180, %c0_137, %c0_139, %c0_140, %183)
            affine.store %179, %alloc_81[%184, %185] {tiled = true} : memref<128x32xf32, 3>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
            %187 = vector.extract %111[%162, %186] : f32 from vector<2x8xf32>
            %c0_141 = arith.constant 0 : index
            %c0_142 = arith.constant 0 : index
            %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%186)
            %c0_143 = arith.constant 0 : index
            %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%186)
            %190 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%189)
            %c0_144 = arith.constant 0 : index
            %191 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_142, %190)
            %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_83, %162, %188, %c0_141, %c0_143, %c0_144, %191)
            %193 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_83, %162, %188, %c0_141, %c0_143, %c0_144, %191)
            affine.store %187, %alloc_81[%192, %193] {tiled = true} : memref<128x32xf32, 3>
            frisk.fragment_yield
          }
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
      }
      gpu.barrier {tiled = true}
      %cst_85 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %117 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_85) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "load" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %c0_131 = arith.constant 0 : index
          %175 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%161)
          %c0_132 = arith.constant 0 : index
          %176 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%161)
          %177 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_131, %176)
          %c0_133 = arith.constant 0 : index
          %178 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_82, %162, %175, %c0, %c0_132, %177, %c0_133)
          %179 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_82, %162, %175, %c0, %c0_132, %177, %c0_133)
          %180 = affine.load %alloc_81[%178, %179] : memref<128x32xf32, 3>
          %181 = vector.insert %180, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %c0_134 = arith.constant 0 : index
          %c0_135 = arith.constant 0 : index
          %183 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%182)
          %c0_136 = arith.constant 0 : index
          %184 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%182)
          %185 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_135, %184)
          %c0_137 = arith.constant 0 : index
          %186 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_82, %162, %183, %c0_134, %c0_136, %185, %c0_137)
          %187 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_82, %162, %183, %c0_134, %c0_136, %185, %c0_137)
          %188 = affine.load %alloc_81[%186, %187] : memref<128x32xf32, 3>
          %189 = vector.insert %188, %181 [0, 1] : f32 into vector<1x4xf32>
          %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %c0_138 = arith.constant 0 : index
          %c0_139 = arith.constant 0 : index
          %191 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%190)
          %c0_140 = arith.constant 0 : index
          %192 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%190)
          %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_139, %192)
          %c0_141 = arith.constant 0 : index
          %194 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_82, %162, %191, %c0_138, %c0_140, %193, %c0_141)
          %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_82, %162, %191, %c0_138, %c0_140, %193, %c0_141)
          %196 = affine.load %alloc_81[%194, %195] : memref<128x32xf32, 3>
          %197 = vector.insert %196, %189 [0, 2] : f32 into vector<1x4xf32>
          %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %c0_142 = arith.constant 0 : index
          %c0_143 = arith.constant 0 : index
          %199 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%198)
          %c0_144 = arith.constant 0 : index
          %200 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%198)
          %201 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_143, %200)
          %c0_145 = arith.constant 0 : index
          %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_82, %162, %199, %c0_142, %c0_144, %201, %c0_145)
          %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_82, %162, %199, %c0_142, %c0_144, %201, %c0_145)
          %204 = affine.load %alloc_81[%202, %203] : memref<128x32xf32, 3>
          %205 = vector.insert %204, %197 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %205 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "layout_fragment", tiled = true}
      frisk.sync_threads_in_block {tiled = true}
      %cst_86 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %118 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_86) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %114[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %114[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %114[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %114[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %117[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %117[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %117[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %117[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.addf %185, %196 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_87 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %119 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %119 to %alloc_87 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_87 : memref<128x32xf16, 3>
      %thread_id_x_88 = gpu.thread_id x
      %cst_89 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %120 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_89) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %189 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg10)[%thread_id_x_88]
          %190 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_88]
          %191 = frisk.read_matrix_from_shm %alloc_87[%189, %190] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %192 = vector.bitcast %191 : vector<4xi32> to vector<8xf16>
          %cst_130 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %193 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_88]
          %194 = vector.extract %192[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %cst_130 [0, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_88]
          %197 = vector.extract %192[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [0, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg10)[%thread_id_x_88]
          %200 = vector.extract %192[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [1, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg10)[%thread_id_x_88]
          %203 = vector.extract %192[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [1, 1] : f16 into vector<4x2xf16>
          %205 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg10)[%thread_id_x_88]
          %206 = vector.extract %192[%205] : f16 from vector<8xf16>
          %207 = vector.insert %206, %204 [2, 0] : f16 into vector<4x2xf16>
          %208 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg10)[%thread_id_x_88]
          %209 = vector.extract %192[%208] : f16 from vector<8xf16>
          %210 = vector.insert %209, %207 [2, 1] : f16 into vector<4x2xf16>
          %211 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg10)[%thread_id_x_88]
          %212 = vector.extract %192[%211] : f16 from vector<8xf16>
          %213 = vector.insert %212, %210 [3, 0] : f16 into vector<4x2xf16>
          %214 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg10)[%thread_id_x_88]
          %215 = vector.extract %192[%214] : f16 from vector<8xf16>
          %216 = vector.insert %215, %213 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %216 : vector<4x2xf16>
        }
        %163 = vector.extract %162[0, 0] : f16 from vector<4x2xf16>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f16 into vector<32x2xf16>
        %165 = affine.apply affine_map<() -> (1)>()
        %166 = vector.extract %162[0, 1] : f16 from vector<4x2xf16>
        %167 = vector.insert %166, %164 [%161, %165] : f16 into vector<32x2xf16>
        %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %169 = vector.extract %162[1, 0] : f16 from vector<4x2xf16>
        %170 = vector.insert %169, %167 [%168, %c0] : f16 into vector<32x2xf16>
        %171 = affine.apply affine_map<() -> (1)>()
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %173 = vector.extract %162[1, 1] : f16 from vector<4x2xf16>
        %174 = vector.insert %173, %170 [%172, %171] : f16 into vector<32x2xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %176 = vector.extract %162[2, 0] : f16 from vector<4x2xf16>
        %177 = vector.insert %176, %174 [%175, %c0] : f16 into vector<32x2xf16>
        %178 = affine.apply affine_map<() -> (1)>()
        %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %180 = vector.extract %162[2, 1] : f16 from vector<4x2xf16>
        %181 = vector.insert %180, %177 [%179, %178] : f16 into vector<32x2xf16>
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %183 = vector.extract %162[3, 0] : f16 from vector<4x2xf16>
        %184 = vector.insert %183, %181 [%182, %c0] : f16 into vector<32x2xf16>
        %185 = affine.apply affine_map<() -> (1)>()
        %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %187 = vector.extract %162[3, 1] : f16 from vector<4x2xf16>
        %188 = vector.insert %187, %184 [%186, %185] : f16 into vector<32x2xf16>
        affine.yield %188 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_90 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_91 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %121 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_90) -> (vector<2x8xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
        %163 = affine.for %arg12 = 0 to 8 iter_args(%arg13 = %cst_91) -> (vector<1x4xf32>) {
          %175 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %178 = frisk.to_threadTile %86 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %179 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
            %180 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg12)
            %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %181 = vector.extract %178[%179, %180] : f16 from vector<2x32xf16>
            %182 = vector.insert %181, %cst_130 [0, 0] : f16 into vector<1x4xf16>
            %183 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg12)
            %184 = vector.extract %178[%179, %183] : f16 from vector<2x32xf16>
            %185 = vector.insert %184, %182 [0, 1] : f16 into vector<1x4xf16>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg12)
            %187 = vector.extract %178[%179, %186] : f16 from vector<2x32xf16>
            %188 = vector.insert %187, %185 [0, 2] : f16 into vector<1x4xf16>
            %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg12)
            %190 = vector.extract %178[%179, %189] : f16 from vector<2x32xf16>
            %191 = vector.insert %190, %188 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %191 : vector<1x4xf16>
          }
          %176 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %178 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg12)
            %179 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
            %cst_130 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %180 = vector.extract %120[%178, %179] : f16 from vector<32x2xf16>
            %181 = vector.insert %180, %cst_130 [0, 0] : f16 into vector<4x1xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg12)
            %183 = vector.extract %120[%182, %179] : f16 from vector<32x2xf16>
            %184 = vector.insert %183, %181 [1, 0] : f16 into vector<4x1xf16>
            %185 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg12)
            %186 = vector.extract %120[%185, %179] : f16 from vector<32x2xf16>
            %187 = vector.insert %186, %184 [2, 0] : f16 into vector<4x1xf16>
            %188 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg12)
            %189 = vector.extract %120[%188, %179] : f16 from vector<32x2xf16>
            %190 = vector.insert %189, %187 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %190 : vector<4x1xf16>
          }
          %177 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %178 = frisk.warp_mma_rr %175, %176, %arg13 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %178 : vector<1x4xf32>
          }
          affine.yield %177 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x8xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x8xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x8xf32>
        affine.yield %174 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %thread_id_x_92 = gpu.thread_id x
      %122 = arith.index_cast %thread_id_x_92 : index to i32
      %c0_i32_93 = arith.constant 0 : i32
      %c64_i32_94 = arith.constant 64 : i32
      %c16_i32_95 = arith.constant 16 : i32
      %123 = arith.andi %122, %c16_i32_95 : i32
      %124 = arith.cmpi ne, %123, %c0_i32_93 : i32
      %c32_i32_96 = arith.constant 32 : i32
      %125 = arith.andi %122, %c32_i32_96 : i32
      %126 = arith.cmpi ne, %125, %c0_i32_93 : i32
      %cst_97 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %127 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_97) -> (vector<2x8xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %121[%162, %161] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
          %178 = vector.extract %121[%162, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
          %181 = vector.extract %121[%162, %180] : f32 from vector<2x8xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
          %184 = vector.extract %121[%162, %183] : f32 from vector<2x8xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %186 = vector.extract %185[0, 0] : f32 from vector<1x4xf32>
          %187 = vector.extract %185[0, 1] : f32 from vector<1x4xf32>
          %188 = vector.extract %185[0, 2] : f32 from vector<1x4xf32>
          %189 = vector.extract %185[0, 3] : f32 from vector<1x4xf32>
          %190 = arith.select %124, %186, %187 : f32
          %shuffleResult, %valid = gpu.shuffle xor %190, %c16_i32_95, %c64_i32_94 {frisk.layout_role = "wmmaC"} : f32
          %191 = arith.select %124, %shuffleResult, %186 : f32
          %192 = arith.select %124, %187, %shuffleResult : f32
          %193 = arith.select %124, %188, %189 : f32
          %shuffleResult_131, %valid_132 = gpu.shuffle xor %193, %c16_i32_95, %c64_i32_94 {frisk.layout_role = "wmmaC"} : f32
          %194 = arith.select %124, %shuffleResult_131, %188 : f32
          %195 = arith.select %124, %189, %shuffleResult_131 : f32
          %196 = arith.select %126, %191, %194 : f32
          %shuffleResult_133, %valid_134 = gpu.shuffle xor %196, %c32_i32_96, %c64_i32_94 {frisk.layout_role = "wmmaC"} : f32
          %197 = arith.select %126, %shuffleResult_133, %191 : f32
          %198 = arith.select %126, %194, %shuffleResult_133 : f32
          %199 = arith.select %126, %192, %195 : f32
          %shuffleResult_135, %valid_136 = gpu.shuffle xor %199, %c32_i32_96, %c64_i32_94 {frisk.layout_role = "wmmaC"} : f32
          %200 = arith.select %126, %shuffleResult_135, %192 : f32
          %201 = arith.select %126, %195, %shuffleResult_135 : f32
          %202 = vector.insert %197, %185 [0, 0] : f32 into vector<1x4xf32>
          %203 = vector.insert %200, %202 [0, 1] : f32 into vector<1x4xf32>
          %204 = vector.insert %198, %203 [0, 2] : f32 into vector<1x4xf32>
          %205 = vector.insert %201, %204 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %205 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x8xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x8xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x8xf32>
        affine.yield %174 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %128 = frisk.to_threadTile %arg8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x32xf32>
      %cst_98 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %129 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_98) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %175 = vector.extract %79[%162, %c0] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %c0_131 = arith.constant 0 : index
          %177 = vector.extract %79[%162, %c0_131] : f32 from vector<2x1xf32>
          %178 = vector.insert %177, %176 [0, 1] : f32 into vector<1x4xf32>
          %c0_132 = arith.constant 0 : index
          %179 = vector.extract %79[%162, %c0_132] : f32 from vector<2x1xf32>
          %180 = vector.insert %179, %178 [0, 2] : f32 into vector<1x4xf32>
          %c0_133 = arith.constant 0 : index
          %181 = vector.extract %79[%162, %c0_133] : f32 from vector<2x1xf32>
          %182 = vector.insert %181, %180 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %182 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %cst_99 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %130 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_99) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %128[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %128[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %128[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %128[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %129[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %129[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %129[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %129[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.mulf %185, %196 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_100 = memref.alloc() {tiled = true} : memref<128x32xf32, 3>
      %thread_id_x_101 = gpu.thread_id x
      %thread_id_x_102 = gpu.thread_id x
      %131 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_102)
      %c4_103 = arith.constant 4 : index
      %132 = arith.cmpi ult, %131, %c4_103 : index
      scf.if %132 {
        affine.for %arg10 = 0 to 4 {
          %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
          %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
          frisk.fragment "store" {
            %163 = vector.extract %127[%162, %161] : f32 from vector<2x8xf32>
            %c0 = arith.constant 0 : index
            %c0_130 = arith.constant 0 : index
            %164 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%161)
            %c0_131 = arith.constant 0 : index
            %165 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%161)
            %166 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%165)
            %c0_132 = arith.constant 0 : index
            %167 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_130, %166)
            %168 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_102, %162, %164, %c0, %c0_131, %c0_132, %167)
            %169 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_102, %162, %164, %c0, %c0_131, %c0_132, %167)
            affine.store %163, %alloc_100[%168, %169] {tiled = true} : memref<128x32xf32, 3>
            %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
            %171 = vector.extract %127[%162, %170] : f32 from vector<2x8xf32>
            %c0_133 = arith.constant 0 : index
            %c0_134 = arith.constant 0 : index
            %172 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%170)
            %c0_135 = arith.constant 0 : index
            %173 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%170)
            %174 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%173)
            %c0_136 = arith.constant 0 : index
            %175 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_134, %174)
            %176 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_102, %162, %172, %c0_133, %c0_135, %c0_136, %175)
            %177 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_102, %162, %172, %c0_133, %c0_135, %c0_136, %175)
            affine.store %171, %alloc_100[%176, %177] {tiled = true} : memref<128x32xf32, 3>
            %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
            %179 = vector.extract %127[%162, %178] : f32 from vector<2x8xf32>
            %c0_137 = arith.constant 0 : index
            %c0_138 = arith.constant 0 : index
            %180 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%178)
            %c0_139 = arith.constant 0 : index
            %181 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%178)
            %182 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%181)
            %c0_140 = arith.constant 0 : index
            %183 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_138, %182)
            %184 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_102, %162, %180, %c0_137, %c0_139, %c0_140, %183)
            %185 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_102, %162, %180, %c0_137, %c0_139, %c0_140, %183)
            affine.store %179, %alloc_100[%184, %185] {tiled = true} : memref<128x32xf32, 3>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
            %187 = vector.extract %127[%162, %186] : f32 from vector<2x8xf32>
            %c0_141 = arith.constant 0 : index
            %c0_142 = arith.constant 0 : index
            %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%186)
            %c0_143 = arith.constant 0 : index
            %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%186)
            %190 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%189)
            %c0_144 = arith.constant 0 : index
            %191 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_142, %190)
            %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_102, %162, %188, %c0_141, %c0_143, %c0_144, %191)
            %193 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_102, %162, %188, %c0_141, %c0_143, %c0_144, %191)
            affine.store %187, %alloc_100[%192, %193] {tiled = true} : memref<128x32xf32, 3>
            frisk.fragment_yield
          }
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
      }
      gpu.barrier {tiled = true}
      %cst_104 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %133 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_104) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "load" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %c0_131 = arith.constant 0 : index
          %175 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%161)
          %c0_132 = arith.constant 0 : index
          %176 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%161)
          %177 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_131, %176)
          %c0_133 = arith.constant 0 : index
          %178 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_101, %162, %175, %c0, %c0_132, %177, %c0_133)
          %179 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_101, %162, %175, %c0, %c0_132, %177, %c0_133)
          %180 = affine.load %alloc_100[%178, %179] : memref<128x32xf32, 3>
          %181 = vector.insert %180, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %c0_134 = arith.constant 0 : index
          %c0_135 = arith.constant 0 : index
          %183 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%182)
          %c0_136 = arith.constant 0 : index
          %184 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%182)
          %185 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_135, %184)
          %c0_137 = arith.constant 0 : index
          %186 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_101, %162, %183, %c0_134, %c0_136, %185, %c0_137)
          %187 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_101, %162, %183, %c0_134, %c0_136, %185, %c0_137)
          %188 = affine.load %alloc_100[%186, %187] : memref<128x32xf32, 3>
          %189 = vector.insert %188, %181 [0, 1] : f32 into vector<1x4xf32>
          %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %c0_138 = arith.constant 0 : index
          %c0_139 = arith.constant 0 : index
          %191 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%190)
          %c0_140 = arith.constant 0 : index
          %192 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%190)
          %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_139, %192)
          %c0_141 = arith.constant 0 : index
          %194 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_101, %162, %191, %c0_138, %c0_140, %193, %c0_141)
          %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_101, %162, %191, %c0_138, %c0_140, %193, %c0_141)
          %196 = affine.load %alloc_100[%194, %195] : memref<128x32xf32, 3>
          %197 = vector.insert %196, %189 [0, 2] : f32 into vector<1x4xf32>
          %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %c0_142 = arith.constant 0 : index
          %c0_143 = arith.constant 0 : index
          %199 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%198)
          %c0_144 = arith.constant 0 : index
          %200 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%198)
          %201 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_143, %200)
          %c0_145 = arith.constant 0 : index
          %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_101, %162, %199, %c0_142, %c0_144, %201, %c0_145)
          %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_101, %162, %199, %c0_142, %c0_144, %201, %c0_145)
          %204 = affine.load %alloc_100[%202, %203] : memref<128x32xf32, 3>
          %205 = vector.insert %204, %197 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %205 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "layout_fragment", tiled = true}
      frisk.sync_threads_in_block {tiled = true}
      %cst_105 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %134 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_105) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %130[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %130[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %130[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %130[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %133[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %133[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %133[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %133[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.addf %185, %196 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_106 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %135 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %135 to %alloc_106 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_106 : memref<128x32xf16, 3>
      %thread_id_x_107 = gpu.thread_id x
      %cst_108 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %136 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_108) -> (vector<32x2xf16>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %189 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>(%arg10)[%thread_id_x_107]
          %190 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%thread_id_x_107]
          %191 = frisk.read_matrix_from_shm %alloc_106[%189, %190] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %192 = vector.bitcast %191 : vector<4xi32> to vector<8xf16>
          %cst_130 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %193 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_107]
          %194 = vector.extract %192[%193] : f16 from vector<8xf16>
          %195 = vector.insert %194, %cst_130 [0, 0] : f16 into vector<4x2xf16>
          %196 = affine.apply affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_107]
          %197 = vector.extract %192[%196] : f16 from vector<8xf16>
          %198 = vector.insert %197, %195 [0, 1] : f16 into vector<4x2xf16>
          %199 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>(%arg10)[%thread_id_x_107]
          %200 = vector.extract %192[%199] : f16 from vector<8xf16>
          %201 = vector.insert %200, %198 [1, 0] : f16 into vector<4x2xf16>
          %202 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>(%arg10)[%thread_id_x_107]
          %203 = vector.extract %192[%202] : f16 from vector<8xf16>
          %204 = vector.insert %203, %201 [1, 1] : f16 into vector<4x2xf16>
          %205 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>(%arg10)[%thread_id_x_107]
          %206 = vector.extract %192[%205] : f16 from vector<8xf16>
          %207 = vector.insert %206, %204 [2, 0] : f16 into vector<4x2xf16>
          %208 = affine.apply affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 + ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * -8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>(%arg10)[%thread_id_x_107]
          %209 = vector.extract %192[%208] : f16 from vector<8xf16>
          %210 = vector.insert %209, %207 [2, 1] : f16 into vector<4x2xf16>
          %211 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>(%arg10)[%thread_id_x_107]
          %212 = vector.extract %192[%211] : f16 from vector<8xf16>
          %213 = vector.insert %212, %210 [3, 0] : f16 into vector<4x2xf16>
          %214 = affine.apply affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 + ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * -16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>(%arg10)[%thread_id_x_107]
          %215 = vector.extract %192[%214] : f16 from vector<8xf16>
          %216 = vector.insert %215, %213 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %216 : vector<4x2xf16>
        }
        %163 = vector.extract %162[0, 0] : f16 from vector<4x2xf16>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f16 into vector<32x2xf16>
        %165 = affine.apply affine_map<() -> (1)>()
        %166 = vector.extract %162[0, 1] : f16 from vector<4x2xf16>
        %167 = vector.insert %166, %164 [%161, %165] : f16 into vector<32x2xf16>
        %168 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %169 = vector.extract %162[1, 0] : f16 from vector<4x2xf16>
        %170 = vector.insert %169, %167 [%168, %c0] : f16 into vector<32x2xf16>
        %171 = affine.apply affine_map<() -> (1)>()
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %173 = vector.extract %162[1, 1] : f16 from vector<4x2xf16>
        %174 = vector.insert %173, %170 [%172, %171] : f16 into vector<32x2xf16>
        %175 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %176 = vector.extract %162[2, 0] : f16 from vector<4x2xf16>
        %177 = vector.insert %176, %174 [%175, %c0] : f16 into vector<32x2xf16>
        %178 = affine.apply affine_map<() -> (1)>()
        %179 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %180 = vector.extract %162[2, 1] : f16 from vector<4x2xf16>
        %181 = vector.insert %180, %177 [%179, %178] : f16 into vector<32x2xf16>
        %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %183 = vector.extract %162[3, 0] : f16 from vector<4x2xf16>
        %184 = vector.insert %183, %181 [%182, %c0] : f16 into vector<32x2xf16>
        %185 = affine.apply affine_map<() -> (1)>()
        %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %187 = vector.extract %162[3, 1] : f16 from vector<4x2xf16>
        %188 = vector.insert %187, %184 [%186, %185] : f16 into vector<32x2xf16>
        affine.yield %188 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_109 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_110 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %137 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_109) -> (vector<2x8xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
        %163 = affine.for %arg12 = 0 to 8 iter_args(%arg13 = %cst_110) -> (vector<1x4xf32>) {
          %175 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %178 = frisk.to_threadTile %86 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %179 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
            %180 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg12)
            %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %181 = vector.extract %178[%179, %180] : f16 from vector<2x32xf16>
            %182 = vector.insert %181, %cst_130 [0, 0] : f16 into vector<1x4xf16>
            %183 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg12)
            %184 = vector.extract %178[%179, %183] : f16 from vector<2x32xf16>
            %185 = vector.insert %184, %182 [0, 1] : f16 into vector<1x4xf16>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg12)
            %187 = vector.extract %178[%179, %186] : f16 from vector<2x32xf16>
            %188 = vector.insert %187, %185 [0, 2] : f16 into vector<1x4xf16>
            %189 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg12)
            %190 = vector.extract %178[%179, %189] : f16 from vector<2x32xf16>
            %191 = vector.insert %190, %188 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %191 : vector<1x4xf16>
          }
          %176 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %178 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg12)
            %179 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
            %cst_130 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %180 = vector.extract %136[%178, %179] : f16 from vector<32x2xf16>
            %181 = vector.insert %180, %cst_130 [0, 0] : f16 into vector<4x1xf16>
            %182 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg12)
            %183 = vector.extract %136[%182, %179] : f16 from vector<32x2xf16>
            %184 = vector.insert %183, %181 [1, 0] : f16 into vector<4x1xf16>
            %185 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg12)
            %186 = vector.extract %136[%185, %179] : f16 from vector<32x2xf16>
            %187 = vector.insert %186, %184 [2, 0] : f16 into vector<4x1xf16>
            %188 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg12)
            %189 = vector.extract %136[%188, %179] : f16 from vector<32x2xf16>
            %190 = vector.insert %189, %187 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %190 : vector<4x1xf16>
          }
          %177 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %178 = frisk.warp_mma_rr %175, %176, %arg13 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %178 : vector<1x4xf32>
          }
          affine.yield %177 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x8xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x8xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x8xf32>
        affine.yield %174 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %thread_id_x_111 = gpu.thread_id x
      %138 = arith.index_cast %thread_id_x_111 : index to i32
      %c0_i32_112 = arith.constant 0 : i32
      %c64_i32_113 = arith.constant 64 : i32
      %c16_i32_114 = arith.constant 16 : i32
      %139 = arith.andi %138, %c16_i32_114 : i32
      %140 = arith.cmpi ne, %139, %c0_i32_112 : i32
      %c32_i32_115 = arith.constant 32 : i32
      %141 = arith.andi %138, %c32_i32_115 : i32
      %142 = arith.cmpi ne, %141, %c0_i32_112 : i32
      %cst_116 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %143 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_116) -> (vector<2x8xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %137[%162, %161] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
          %178 = vector.extract %137[%162, %177] : f32 from vector<2x8xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
          %181 = vector.extract %137[%162, %180] : f32 from vector<2x8xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
          %184 = vector.extract %137[%162, %183] : f32 from vector<2x8xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %186 = vector.extract %185[0, 0] : f32 from vector<1x4xf32>
          %187 = vector.extract %185[0, 1] : f32 from vector<1x4xf32>
          %188 = vector.extract %185[0, 2] : f32 from vector<1x4xf32>
          %189 = vector.extract %185[0, 3] : f32 from vector<1x4xf32>
          %190 = arith.select %140, %186, %187 : f32
          %shuffleResult, %valid = gpu.shuffle xor %190, %c16_i32_114, %c64_i32_113 {frisk.layout_role = "wmmaC"} : f32
          %191 = arith.select %140, %shuffleResult, %186 : f32
          %192 = arith.select %140, %187, %shuffleResult : f32
          %193 = arith.select %140, %188, %189 : f32
          %shuffleResult_131, %valid_132 = gpu.shuffle xor %193, %c16_i32_114, %c64_i32_113 {frisk.layout_role = "wmmaC"} : f32
          %194 = arith.select %140, %shuffleResult_131, %188 : f32
          %195 = arith.select %140, %189, %shuffleResult_131 : f32
          %196 = arith.select %142, %191, %194 : f32
          %shuffleResult_133, %valid_134 = gpu.shuffle xor %196, %c32_i32_115, %c64_i32_113 {frisk.layout_role = "wmmaC"} : f32
          %197 = arith.select %142, %shuffleResult_133, %191 : f32
          %198 = arith.select %142, %194, %shuffleResult_133 : f32
          %199 = arith.select %142, %192, %195 : f32
          %shuffleResult_135, %valid_136 = gpu.shuffle xor %199, %c32_i32_115, %c64_i32_113 {frisk.layout_role = "wmmaC"} : f32
          %200 = arith.select %142, %shuffleResult_135, %192 : f32
          %201 = arith.select %142, %195, %shuffleResult_135 : f32
          %202 = vector.insert %197, %185 [0, 0] : f32 into vector<1x4xf32>
          %203 = vector.insert %200, %202 [0, 1] : f32 into vector<1x4xf32>
          %204 = vector.insert %198, %203 [0, 2] : f32 into vector<1x4xf32>
          %205 = vector.insert %201, %204 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %205 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x8xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x8xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x8xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x8xf32>
        affine.yield %174 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %144 = frisk.to_threadTile %arg9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x32xf32>
      %cst_117 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %145 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_117) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %175 = vector.extract %79[%162, %c0] : f32 from vector<2x1xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %c0_131 = arith.constant 0 : index
          %177 = vector.extract %79[%162, %c0_131] : f32 from vector<2x1xf32>
          %178 = vector.insert %177, %176 [0, 1] : f32 into vector<1x4xf32>
          %c0_132 = arith.constant 0 : index
          %179 = vector.extract %79[%162, %c0_132] : f32 from vector<2x1xf32>
          %180 = vector.insert %179, %178 [0, 2] : f32 into vector<1x4xf32>
          %c0_133 = arith.constant 0 : index
          %181 = vector.extract %79[%162, %c0_133] : f32 from vector<2x1xf32>
          %182 = vector.insert %181, %180 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %182 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %cst_118 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %146 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_118) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %144[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %144[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %144[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %144[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %145[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %145[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %145[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %145[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.mulf %185, %196 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %alloc_119 = memref.alloc() {tiled = true} : memref<128x32xf32, 3>
      %thread_id_x_120 = gpu.thread_id x
      %thread_id_x_121 = gpu.thread_id x
      %147 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_121)
      %c4_122 = arith.constant 4 : index
      %148 = arith.cmpi ult, %147, %c4_122 : index
      scf.if %148 {
        affine.for %arg10 = 0 to 4 {
          %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg10)
          %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg10)
          frisk.fragment "store" {
            %163 = vector.extract %143[%162, %161] : f32 from vector<2x8xf32>
            %c0 = arith.constant 0 : index
            %c0_130 = arith.constant 0 : index
            %164 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%161)
            %c0_131 = arith.constant 0 : index
            %165 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%161)
            %166 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%165)
            %c0_132 = arith.constant 0 : index
            %167 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_130, %166)
            %168 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_121, %162, %164, %c0, %c0_131, %c0_132, %167)
            %169 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_121, %162, %164, %c0, %c0_131, %c0_132, %167)
            affine.store %163, %alloc_119[%168, %169] {tiled = true} : memref<128x32xf32, 3>
            %170 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg10)
            %171 = vector.extract %143[%162, %170] : f32 from vector<2x8xf32>
            %c0_133 = arith.constant 0 : index
            %c0_134 = arith.constant 0 : index
            %172 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%170)
            %c0_135 = arith.constant 0 : index
            %173 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%170)
            %174 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%173)
            %c0_136 = arith.constant 0 : index
            %175 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_134, %174)
            %176 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_121, %162, %172, %c0_133, %c0_135, %c0_136, %175)
            %177 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_121, %162, %172, %c0_133, %c0_135, %c0_136, %175)
            affine.store %171, %alloc_119[%176, %177] {tiled = true} : memref<128x32xf32, 3>
            %178 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg10)
            %179 = vector.extract %143[%162, %178] : f32 from vector<2x8xf32>
            %c0_137 = arith.constant 0 : index
            %c0_138 = arith.constant 0 : index
            %180 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%178)
            %c0_139 = arith.constant 0 : index
            %181 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%178)
            %182 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%181)
            %c0_140 = arith.constant 0 : index
            %183 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_138, %182)
            %184 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_121, %162, %180, %c0_137, %c0_139, %c0_140, %183)
            %185 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_121, %162, %180, %c0_137, %c0_139, %c0_140, %183)
            affine.store %179, %alloc_119[%184, %185] {tiled = true} : memref<128x32xf32, 3>
            %186 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg10)
            %187 = vector.extract %143[%162, %186] : f32 from vector<2x8xf32>
            %c0_141 = arith.constant 0 : index
            %c0_142 = arith.constant 0 : index
            %188 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%186)
            %c0_143 = arith.constant 0 : index
            %189 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%186)
            %190 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%189)
            %c0_144 = arith.constant 0 : index
            %191 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_142, %190)
            %192 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_121, %162, %188, %c0_141, %c0_143, %c0_144, %191)
            %193 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_121, %162, %188, %c0_141, %c0_143, %c0_144, %191)
            affine.store %187, %alloc_119[%192, %193] {tiled = true} : memref<128x32xf32, 3>
            frisk.fragment_yield
          }
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
      }
      gpu.barrier {tiled = true}
      %cst_123 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %149 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_123) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "load" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %c0_131 = arith.constant 0 : index
          %175 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%161)
          %c0_132 = arith.constant 0 : index
          %176 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%161)
          %177 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_131, %176)
          %c0_133 = arith.constant 0 : index
          %178 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_120, %162, %175, %c0, %c0_132, %177, %c0_133)
          %179 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_120, %162, %175, %c0, %c0_132, %177, %c0_133)
          %180 = affine.load %alloc_119[%178, %179] : memref<128x32xf32, 3>
          %181 = vector.insert %180, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %182 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %c0_134 = arith.constant 0 : index
          %c0_135 = arith.constant 0 : index
          %183 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%182)
          %c0_136 = arith.constant 0 : index
          %184 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%182)
          %185 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_135, %184)
          %c0_137 = arith.constant 0 : index
          %186 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_120, %162, %183, %c0_134, %c0_136, %185, %c0_137)
          %187 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_120, %162, %183, %c0_134, %c0_136, %185, %c0_137)
          %188 = affine.load %alloc_119[%186, %187] : memref<128x32xf32, 3>
          %189 = vector.insert %188, %181 [0, 1] : f32 into vector<1x4xf32>
          %190 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %c0_138 = arith.constant 0 : index
          %c0_139 = arith.constant 0 : index
          %191 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%190)
          %c0_140 = arith.constant 0 : index
          %192 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%190)
          %193 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_139, %192)
          %c0_141 = arith.constant 0 : index
          %194 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_120, %162, %191, %c0_138, %c0_140, %193, %c0_141)
          %195 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_120, %162, %191, %c0_138, %c0_140, %193, %c0_141)
          %196 = affine.load %alloc_119[%194, %195] : memref<128x32xf32, 3>
          %197 = vector.insert %196, %189 [0, 2] : f32 into vector<1x4xf32>
          %198 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %c0_142 = arith.constant 0 : index
          %c0_143 = arith.constant 0 : index
          %199 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%198)
          %c0_144 = arith.constant 0 : index
          %200 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%198)
          %201 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_143, %200)
          %c0_145 = arith.constant 0 : index
          %202 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_120, %162, %199, %c0_142, %c0_144, %201, %c0_145)
          %203 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_120, %162, %199, %c0_142, %c0_144, %201, %c0_145)
          %204 = affine.load %alloc_119[%202, %203] : memref<128x32xf32, 3>
          %205 = vector.insert %204, %197 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %205 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "layout_fragment", tiled = true}
      frisk.sync_threads_in_block {tiled = true}
      %cst_124 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %150 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_124) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %146[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %146[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %146[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %146[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] : f32 into vector<1x4xf32>
          %cst_131 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %186 = vector.extract %149[%162, %161] : f32 from vector<2x32xf32>
          %187 = vector.insert %186, %cst_131 [0, 0] : f32 into vector<1x4xf32>
          %188 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %189 = vector.extract %149[%162, %188] : f32 from vector<2x32xf32>
          %190 = vector.insert %189, %187 [0, 1] : f32 into vector<1x4xf32>
          %191 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %192 = vector.extract %149[%162, %191] : f32 from vector<2x32xf32>
          %193 = vector.insert %192, %190 [0, 2] : f32 into vector<1x4xf32>
          %194 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %195 = vector.extract %149[%162, %194] : f32 from vector<2x32xf32>
          %196 = vector.insert %195, %193 [0, 3] : f32 into vector<1x4xf32>
          %197 = arith.addf %185, %196 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %197 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_125 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %151 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_125) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %102[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %102[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %102[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %102[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %152 = frisk.from_threadTile %151 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x32xf32>
      %cst_126 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %153 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %cst_126) -> (vector<2x1xf32>) {
        %c0 = arith.constant 0 : index
        %161 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg10)
        %162 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %165 = vector.extract %84[%161, %c0] : f32 from vector<2x1xf32>
          %166 = vector.insert %165, %cst_130 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
          frisk.fragment_yield %166 : vector<1x1xf32>
        }
        %163 = vector.extract %162[0, 0] : f32 from vector<1x1xf32>
        %164 = vector.insert %163, %arg11 [%161, %c0] : f32 into vector<2x1xf32>
        affine.yield %164 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %154 = frisk.from_threadTile %153 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_127 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %155 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_127) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %118[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %118[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %118[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %118[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %156 = frisk.from_threadTile %155 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x32xf32>
      %cst_128 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %157 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_128) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %134[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %134[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %134[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %134[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %158 = frisk.from_threadTile %157 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x32xf32>
      %cst_129 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %159 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_129) -> (vector<2x32xf32>) {
        %161 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg10)
        %162 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg10)
        %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_130 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %175 = vector.extract %150[%162, %161] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %cst_130 [0, 0] : f32 into vector<1x4xf32>
          %177 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
          %178 = vector.extract %150[%162, %177] : f32 from vector<2x32xf32>
          %179 = vector.insert %178, %176 [0, 1] : f32 into vector<1x4xf32>
          %180 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
          %181 = vector.extract %150[%162, %180] : f32 from vector<2x32xf32>
          %182 = vector.insert %181, %179 [0, 2] : f32 into vector<1x4xf32>
          %183 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
          %184 = vector.extract %150[%162, %183] : f32 from vector<2x32xf32>
          %185 = vector.insert %184, %182 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %185 : vector<1x4xf32>
        }
        %164 = vector.extract %163[0, 0] : f32 from vector<1x4xf32>
        %165 = vector.insert %164, %arg11 [%162, %161] : f32 into vector<2x32xf32>
        %166 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg10)
        %167 = vector.extract %163[0, 1] : f32 from vector<1x4xf32>
        %168 = vector.insert %167, %165 [%162, %166] : f32 into vector<2x32xf32>
        %169 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg10)
        %170 = vector.extract %163[0, 2] : f32 from vector<1x4xf32>
        %171 = vector.insert %170, %168 [%162, %169] : f32 into vector<2x32xf32>
        %172 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg10)
        %173 = vector.extract %163[0, 3] : f32 from vector<1x4xf32>
        %174 = vector.insert %173, %171 [%162, %172] : f32 into vector<2x32xf32>
        affine.yield %174 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %160 = frisk.from_threadTile %159 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x32xf32>
      affine.yield %152, %154, %156, %158, %160 : vector<128x32xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %15 = frisk.to_threadTile %14#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x32xf32>
    %16 = frisk.to_threadTile %14#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %17 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_11) -> (vector<2x32xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %73 = vector.extract %16[%60, %c0] : f32 from vector<2x1xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %c0_34 = arith.constant 0 : index
        %75 = vector.extract %16[%60, %c0_34] : f32 from vector<2x1xf32>
        %76 = vector.insert %75, %74 [0, 1] : f32 into vector<1x4xf32>
        %c0_35 = arith.constant 0 : index
        %77 = vector.extract %16[%60, %c0_35] : f32 from vector<2x1xf32>
        %78 = vector.insert %77, %76 [0, 2] : f32 into vector<1x4xf32>
        %c0_36 = arith.constant 0 : index
        %79 = vector.extract %16[%60, %c0_36] : f32 from vector<2x1xf32>
        %80 = vector.insert %79, %78 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %80 : vector<1x4xf32>
      }
      %62 = vector.extract %61[0, 0] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %arg5 [%60, %59] : f32 into vector<2x32xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%60, %64] : f32 into vector<2x32xf32>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f32 from vector<1x4xf32>
      %69 = vector.insert %68, %66 [%60, %67] : f32 into vector<2x32xf32>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f32 from vector<1x4xf32>
      %72 = vector.insert %71, %69 [%60, %70] : f32 into vector<2x32xf32>
      affine.yield %72 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
    %cst_12 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %18 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_12) -> (vector<2x32xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %73 = vector.extract %15[%60, %59] : f32 from vector<2x32xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %76 = vector.extract %15[%60, %75] : f32 from vector<2x32xf32>
        %77 = vector.insert %76, %74 [0, 1] : f32 into vector<1x4xf32>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %79 = vector.extract %15[%60, %78] : f32 from vector<2x32xf32>
        %80 = vector.insert %79, %77 [0, 2] : f32 into vector<1x4xf32>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %82 = vector.extract %15[%60, %81] : f32 from vector<2x32xf32>
        %83 = vector.insert %82, %80 [0, 3] : f32 into vector<1x4xf32>
        %cst_34 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %84 = vector.extract %17[%60, %59] : f32 from vector<2x32xf32>
        %85 = vector.insert %84, %cst_34 [0, 0] : f32 into vector<1x4xf32>
        %86 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %87 = vector.extract %17[%60, %86] : f32 from vector<2x32xf32>
        %88 = vector.insert %87, %85 [0, 1] : f32 into vector<1x4xf32>
        %89 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %90 = vector.extract %17[%60, %89] : f32 from vector<2x32xf32>
        %91 = vector.insert %90, %88 [0, 2] : f32 into vector<1x4xf32>
        %92 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %93 = vector.extract %17[%60, %92] : f32 from vector<2x32xf32>
        %94 = vector.insert %93, %91 [0, 3] : f32 into vector<1x4xf32>
        %95 = arith.divf %83, %94 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %95 : vector<1x4xf32>
      }
      %62 = vector.extract %61[0, 0] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %arg5 [%60, %59] : f32 into vector<2x32xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%60, %64] : f32 into vector<2x32xf32>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f32 from vector<1x4xf32>
      %69 = vector.insert %68, %66 [%60, %67] : f32 into vector<2x32xf32>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f32 from vector<1x4xf32>
      %72 = vector.insert %71, %69 [%60, %70] : f32 into vector<2x32xf32>
      affine.yield %72 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %cst_13 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %19 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_13) -> (vector<2x32xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %73 = vector.extract %18[%60, %59] : f32 from vector<2x32xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %76 = vector.extract %18[%60, %75] : f32 from vector<2x32xf32>
        %77 = vector.insert %76, %74 [0, 1] : f32 into vector<1x4xf32>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %79 = vector.extract %18[%60, %78] : f32 from vector<2x32xf32>
        %80 = vector.insert %79, %77 [0, 2] : f32 into vector<1x4xf32>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %82 = vector.extract %18[%60, %81] : f32 from vector<2x32xf32>
        %83 = vector.insert %82, %80 [0, 3] : f32 into vector<1x4xf32>
        %84 = arith.truncf %83 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %84 : vector<1x4xf16>
      }
      %62 = vector.extract %61[0, 0] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %arg5 [%60, %59] : f16 into vector<2x32xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%60, %64] : f16 into vector<2x32xf16>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f16 from vector<1x4xf16>
      %69 = vector.insert %68, %66 [%60, %67] : f16 into vector<2x32xf16>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f16 from vector<1x4xf16>
      %72 = vector.insert %71, %69 [%60, %70] : f16 into vector<2x32xf16>
      affine.yield %72 : vector<2x32xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %20 = frisk.from_threadTile %19 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x32xf16, 3>
    %21 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%8 * 16384) floordiv 524288) mod 32, (%8 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %22 = frisk.to_threadTile %20 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %cst_14 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %23 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_14) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %73 = vector.extract %22[%60, %59] : f16 from vector<2x8xf16>
        %74 = vector.insert %73, %cst_33 [0, 0] : f16 into vector<1x4xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %76 = vector.extract %22[%60, %75] : f16 from vector<2x8xf16>
        %77 = vector.insert %76, %74 [0, 1] : f16 into vector<1x4xf16>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %79 = vector.extract %22[%60, %78] : f16 from vector<2x8xf16>
        %80 = vector.insert %79, %77 [0, 2] : f16 into vector<1x4xf16>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %82 = vector.extract %22[%60, %81] : f16 from vector<2x8xf16>
        %83 = vector.insert %82, %80 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %83 : vector<1x4xf16>
      }
      %62 = vector.extract %61[0, 0] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %arg5 [%60, %59] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%60, %64] : f16 into vector<2x8xf16>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f16 from vector<1x4xf16>
      %69 = vector.insert %68, %66 [%60, %67] : f16 into vector<2x8xf16>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f16 from vector<1x4xf16>
      %72 = vector.insert %71, %69 [%60, %70] : f16 into vector<2x8xf16>
      affine.yield %72 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x = gpu.thread_id x
    %24 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x)
    %c4 = arith.constant 4 : index
    %25 = arith.cmpi ult, %24, %c4 : index
    scf.if %25 {
      affine.for %arg4 = 0 to 4 {
        %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %61 = vector.extract %23[%60, %59] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_33 = arith.constant 0 : index
          %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%59)
          %c0_34 = arith.constant 0 : index
          %63 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%59)
          %64 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_33, %63)
          %c0_35 = arith.constant 0 : index
          %65 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x, %60, %62, %c0, %c0_34, %64, %c0_35)
          %66 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x, %60, %62, %c0, %c0_34, %64, %c0_35)
          affine.store %61, %21[%65, %66] {tiled = true} : memref<128x32xf16, 1>
          %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %68 = vector.extract %23[%60, %67] : f16 from vector<2x8xf16>
          %c0_36 = arith.constant 0 : index
          %c0_37 = arith.constant 0 : index
          %69 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%67)
          %c0_38 = arith.constant 0 : index
          %70 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%67)
          %71 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_37, %70)
          %c0_39 = arith.constant 0 : index
          %72 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x, %60, %69, %c0_36, %c0_38, %71, %c0_39)
          %73 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x, %60, %69, %c0_36, %c0_38, %71, %c0_39)
          affine.store %68, %21[%72, %73] {tiled = true} : memref<128x32xf16, 1>
          %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %75 = vector.extract %23[%60, %74] : f16 from vector<2x8xf16>
          %c0_40 = arith.constant 0 : index
          %c0_41 = arith.constant 0 : index
          %76 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%74)
          %c0_42 = arith.constant 0 : index
          %77 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%74)
          %78 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_41, %77)
          %c0_43 = arith.constant 0 : index
          %79 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x, %60, %76, %c0_40, %c0_42, %78, %c0_43)
          %80 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x, %60, %76, %c0_40, %c0_42, %78, %c0_43)
          affine.store %75, %21[%79, %80] {tiled = true} : memref<128x32xf16, 1>
          %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %82 = vector.extract %23[%60, %81] : f16 from vector<2x8xf16>
          %c0_44 = arith.constant 0 : index
          %c0_45 = arith.constant 0 : index
          %83 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%81)
          %c0_46 = arith.constant 0 : index
          %84 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%81)
          %85 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_45, %84)
          %c0_47 = arith.constant 0 : index
          %86 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x, %60, %83, %c0_44, %c0_46, %85, %c0_47)
          %87 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x, %60, %83, %c0_44, %c0_46, %85, %c0_47)
          affine.store %82, %21[%86, %87] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %26 = frisk.to_threadTile %14#2 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x32xf32>
    %27 = frisk.to_threadTile %14#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_15 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %28 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_15) -> (vector<2x32xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %73 = vector.extract %27[%60, %c0] : f32 from vector<2x1xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %c0_34 = arith.constant 0 : index
        %75 = vector.extract %27[%60, %c0_34] : f32 from vector<2x1xf32>
        %76 = vector.insert %75, %74 [0, 1] : f32 into vector<1x4xf32>
        %c0_35 = arith.constant 0 : index
        %77 = vector.extract %27[%60, %c0_35] : f32 from vector<2x1xf32>
        %78 = vector.insert %77, %76 [0, 2] : f32 into vector<1x4xf32>
        %c0_36 = arith.constant 0 : index
        %79 = vector.extract %27[%60, %c0_36] : f32 from vector<2x1xf32>
        %80 = vector.insert %79, %78 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %80 : vector<1x4xf32>
      }
      %62 = vector.extract %61[0, 0] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %arg5 [%60, %59] : f32 into vector<2x32xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%60, %64] : f32 into vector<2x32xf32>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f32 from vector<1x4xf32>
      %69 = vector.insert %68, %66 [%60, %67] : f32 into vector<2x32xf32>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f32 from vector<1x4xf32>
      %72 = vector.insert %71, %69 [%60, %70] : f32 into vector<2x32xf32>
      affine.yield %72 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
    %cst_16 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %29 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_16) -> (vector<2x32xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %73 = vector.extract %26[%60, %59] : f32 from vector<2x32xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %76 = vector.extract %26[%60, %75] : f32 from vector<2x32xf32>
        %77 = vector.insert %76, %74 [0, 1] : f32 into vector<1x4xf32>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %79 = vector.extract %26[%60, %78] : f32 from vector<2x32xf32>
        %80 = vector.insert %79, %77 [0, 2] : f32 into vector<1x4xf32>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %82 = vector.extract %26[%60, %81] : f32 from vector<2x32xf32>
        %83 = vector.insert %82, %80 [0, 3] : f32 into vector<1x4xf32>
        %cst_34 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %84 = vector.extract %28[%60, %59] : f32 from vector<2x32xf32>
        %85 = vector.insert %84, %cst_34 [0, 0] : f32 into vector<1x4xf32>
        %86 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %87 = vector.extract %28[%60, %86] : f32 from vector<2x32xf32>
        %88 = vector.insert %87, %85 [0, 1] : f32 into vector<1x4xf32>
        %89 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %90 = vector.extract %28[%60, %89] : f32 from vector<2x32xf32>
        %91 = vector.insert %90, %88 [0, 2] : f32 into vector<1x4xf32>
        %92 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %93 = vector.extract %28[%60, %92] : f32 from vector<2x32xf32>
        %94 = vector.insert %93, %91 [0, 3] : f32 into vector<1x4xf32>
        %95 = arith.divf %83, %94 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %95 : vector<1x4xf32>
      }
      %62 = vector.extract %61[0, 0] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %arg5 [%60, %59] : f32 into vector<2x32xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%60, %64] : f32 into vector<2x32xf32>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f32 from vector<1x4xf32>
      %69 = vector.insert %68, %66 [%60, %67] : f32 into vector<2x32xf32>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f32 from vector<1x4xf32>
      %72 = vector.insert %71, %69 [%60, %70] : f32 into vector<2x32xf32>
      affine.yield %72 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %cst_17 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %30 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_17) -> (vector<2x32xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %73 = vector.extract %29[%60, %59] : f32 from vector<2x32xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %76 = vector.extract %29[%60, %75] : f32 from vector<2x32xf32>
        %77 = vector.insert %76, %74 [0, 1] : f32 into vector<1x4xf32>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %79 = vector.extract %29[%60, %78] : f32 from vector<2x32xf32>
        %80 = vector.insert %79, %77 [0, 2] : f32 into vector<1x4xf32>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %82 = vector.extract %29[%60, %81] : f32 from vector<2x32xf32>
        %83 = vector.insert %82, %80 [0, 3] : f32 into vector<1x4xf32>
        %84 = arith.truncf %83 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %84 : vector<1x4xf16>
      }
      %62 = vector.extract %61[0, 0] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %arg5 [%60, %59] : f16 into vector<2x32xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%60, %64] : f16 into vector<2x32xf16>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f16 from vector<1x4xf16>
      %69 = vector.insert %68, %66 [%60, %67] : f16 into vector<2x32xf16>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f16 from vector<1x4xf16>
      %72 = vector.insert %71, %69 [%60, %70] : f16 into vector<2x32xf16>
      affine.yield %72 : vector<2x32xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %31 = frisk.from_threadTile %30 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x32xf16, 3>
    %32 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + 32) floordiv 524288) mod 32, (%8 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %33 = frisk.to_threadTile %31 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %cst_18 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %34 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_18) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %73 = vector.extract %33[%60, %59] : f16 from vector<2x8xf16>
        %74 = vector.insert %73, %cst_33 [0, 0] : f16 into vector<1x4xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %76 = vector.extract %33[%60, %75] : f16 from vector<2x8xf16>
        %77 = vector.insert %76, %74 [0, 1] : f16 into vector<1x4xf16>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %79 = vector.extract %33[%60, %78] : f16 from vector<2x8xf16>
        %80 = vector.insert %79, %77 [0, 2] : f16 into vector<1x4xf16>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %82 = vector.extract %33[%60, %81] : f16 from vector<2x8xf16>
        %83 = vector.insert %82, %80 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %83 : vector<1x4xf16>
      }
      %62 = vector.extract %61[0, 0] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %arg5 [%60, %59] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%60, %64] : f16 into vector<2x8xf16>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f16 from vector<1x4xf16>
      %69 = vector.insert %68, %66 [%60, %67] : f16 into vector<2x8xf16>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f16 from vector<1x4xf16>
      %72 = vector.insert %71, %69 [%60, %70] : f16 into vector<2x8xf16>
      affine.yield %72 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_19 = gpu.thread_id x
    %35 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_19)
    %c4_20 = arith.constant 4 : index
    %36 = arith.cmpi ult, %35, %c4_20 : index
    scf.if %36 {
      affine.for %arg4 = 0 to 4 {
        %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %61 = vector.extract %34[%60, %59] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_33 = arith.constant 0 : index
          %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%59)
          %c0_34 = arith.constant 0 : index
          %63 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%59)
          %64 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_33, %63)
          %c0_35 = arith.constant 0 : index
          %65 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_19, %60, %62, %c0, %c0_34, %64, %c0_35)
          %66 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_19, %60, %62, %c0, %c0_34, %64, %c0_35)
          affine.store %61, %32[%65, %66] {tiled = true} : memref<128x32xf16, 1>
          %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %68 = vector.extract %34[%60, %67] : f16 from vector<2x8xf16>
          %c0_36 = arith.constant 0 : index
          %c0_37 = arith.constant 0 : index
          %69 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%67)
          %c0_38 = arith.constant 0 : index
          %70 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%67)
          %71 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_37, %70)
          %c0_39 = arith.constant 0 : index
          %72 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_19, %60, %69, %c0_36, %c0_38, %71, %c0_39)
          %73 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_19, %60, %69, %c0_36, %c0_38, %71, %c0_39)
          affine.store %68, %32[%72, %73] {tiled = true} : memref<128x32xf16, 1>
          %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %75 = vector.extract %34[%60, %74] : f16 from vector<2x8xf16>
          %c0_40 = arith.constant 0 : index
          %c0_41 = arith.constant 0 : index
          %76 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%74)
          %c0_42 = arith.constant 0 : index
          %77 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%74)
          %78 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_41, %77)
          %c0_43 = arith.constant 0 : index
          %79 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_19, %60, %76, %c0_40, %c0_42, %78, %c0_43)
          %80 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_19, %60, %76, %c0_40, %c0_42, %78, %c0_43)
          affine.store %75, %32[%79, %80] {tiled = true} : memref<128x32xf16, 1>
          %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %82 = vector.extract %34[%60, %81] : f16 from vector<2x8xf16>
          %c0_44 = arith.constant 0 : index
          %c0_45 = arith.constant 0 : index
          %83 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%81)
          %c0_46 = arith.constant 0 : index
          %84 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%81)
          %85 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_45, %84)
          %c0_47 = arith.constant 0 : index
          %86 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_19, %60, %83, %c0_44, %c0_46, %85, %c0_47)
          %87 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_19, %60, %83, %c0_44, %c0_46, %85, %c0_47)
          affine.store %82, %32[%86, %87] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %37 = frisk.to_threadTile %14#3 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x32xf32>
    %38 = frisk.to_threadTile %14#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_21 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %39 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_21) -> (vector<2x32xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %73 = vector.extract %38[%60, %c0] : f32 from vector<2x1xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %c0_34 = arith.constant 0 : index
        %75 = vector.extract %38[%60, %c0_34] : f32 from vector<2x1xf32>
        %76 = vector.insert %75, %74 [0, 1] : f32 into vector<1x4xf32>
        %c0_35 = arith.constant 0 : index
        %77 = vector.extract %38[%60, %c0_35] : f32 from vector<2x1xf32>
        %78 = vector.insert %77, %76 [0, 2] : f32 into vector<1x4xf32>
        %c0_36 = arith.constant 0 : index
        %79 = vector.extract %38[%60, %c0_36] : f32 from vector<2x1xf32>
        %80 = vector.insert %79, %78 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %80 : vector<1x4xf32>
      }
      %62 = vector.extract %61[0, 0] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %arg5 [%60, %59] : f32 into vector<2x32xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%60, %64] : f32 into vector<2x32xf32>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f32 from vector<1x4xf32>
      %69 = vector.insert %68, %66 [%60, %67] : f32 into vector<2x32xf32>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f32 from vector<1x4xf32>
      %72 = vector.insert %71, %69 [%60, %70] : f32 into vector<2x32xf32>
      affine.yield %72 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
    %cst_22 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %40 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_22) -> (vector<2x32xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %73 = vector.extract %37[%60, %59] : f32 from vector<2x32xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %76 = vector.extract %37[%60, %75] : f32 from vector<2x32xf32>
        %77 = vector.insert %76, %74 [0, 1] : f32 into vector<1x4xf32>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %79 = vector.extract %37[%60, %78] : f32 from vector<2x32xf32>
        %80 = vector.insert %79, %77 [0, 2] : f32 into vector<1x4xf32>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %82 = vector.extract %37[%60, %81] : f32 from vector<2x32xf32>
        %83 = vector.insert %82, %80 [0, 3] : f32 into vector<1x4xf32>
        %cst_34 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %84 = vector.extract %39[%60, %59] : f32 from vector<2x32xf32>
        %85 = vector.insert %84, %cst_34 [0, 0] : f32 into vector<1x4xf32>
        %86 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %87 = vector.extract %39[%60, %86] : f32 from vector<2x32xf32>
        %88 = vector.insert %87, %85 [0, 1] : f32 into vector<1x4xf32>
        %89 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %90 = vector.extract %39[%60, %89] : f32 from vector<2x32xf32>
        %91 = vector.insert %90, %88 [0, 2] : f32 into vector<1x4xf32>
        %92 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %93 = vector.extract %39[%60, %92] : f32 from vector<2x32xf32>
        %94 = vector.insert %93, %91 [0, 3] : f32 into vector<1x4xf32>
        %95 = arith.divf %83, %94 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %95 : vector<1x4xf32>
      }
      %62 = vector.extract %61[0, 0] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %arg5 [%60, %59] : f32 into vector<2x32xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%60, %64] : f32 into vector<2x32xf32>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f32 from vector<1x4xf32>
      %69 = vector.insert %68, %66 [%60, %67] : f32 into vector<2x32xf32>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f32 from vector<1x4xf32>
      %72 = vector.insert %71, %69 [%60, %70] : f32 into vector<2x32xf32>
      affine.yield %72 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %cst_23 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %41 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_23) -> (vector<2x32xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %73 = vector.extract %40[%60, %59] : f32 from vector<2x32xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %76 = vector.extract %40[%60, %75] : f32 from vector<2x32xf32>
        %77 = vector.insert %76, %74 [0, 1] : f32 into vector<1x4xf32>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %79 = vector.extract %40[%60, %78] : f32 from vector<2x32xf32>
        %80 = vector.insert %79, %77 [0, 2] : f32 into vector<1x4xf32>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %82 = vector.extract %40[%60, %81] : f32 from vector<2x32xf32>
        %83 = vector.insert %82, %80 [0, 3] : f32 into vector<1x4xf32>
        %84 = arith.truncf %83 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %84 : vector<1x4xf16>
      }
      %62 = vector.extract %61[0, 0] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %arg5 [%60, %59] : f16 into vector<2x32xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%60, %64] : f16 into vector<2x32xf16>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f16 from vector<1x4xf16>
      %69 = vector.insert %68, %66 [%60, %67] : f16 into vector<2x32xf16>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f16 from vector<1x4xf16>
      %72 = vector.insert %71, %69 [%60, %70] : f16 into vector<2x32xf16>
      affine.yield %72 : vector<2x32xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %42 = frisk.from_threadTile %41 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x32xf16, 3>
    %43 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + 64) floordiv 524288) mod 32, (%8 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %44 = frisk.to_threadTile %42 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %cst_24 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %45 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_24) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %73 = vector.extract %44[%60, %59] : f16 from vector<2x8xf16>
        %74 = vector.insert %73, %cst_33 [0, 0] : f16 into vector<1x4xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %76 = vector.extract %44[%60, %75] : f16 from vector<2x8xf16>
        %77 = vector.insert %76, %74 [0, 1] : f16 into vector<1x4xf16>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %79 = vector.extract %44[%60, %78] : f16 from vector<2x8xf16>
        %80 = vector.insert %79, %77 [0, 2] : f16 into vector<1x4xf16>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %82 = vector.extract %44[%60, %81] : f16 from vector<2x8xf16>
        %83 = vector.insert %82, %80 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %83 : vector<1x4xf16>
      }
      %62 = vector.extract %61[0, 0] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %arg5 [%60, %59] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%60, %64] : f16 into vector<2x8xf16>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f16 from vector<1x4xf16>
      %69 = vector.insert %68, %66 [%60, %67] : f16 into vector<2x8xf16>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f16 from vector<1x4xf16>
      %72 = vector.insert %71, %69 [%60, %70] : f16 into vector<2x8xf16>
      affine.yield %72 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_25 = gpu.thread_id x
    %46 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_25)
    %c4_26 = arith.constant 4 : index
    %47 = arith.cmpi ult, %46, %c4_26 : index
    scf.if %47 {
      affine.for %arg4 = 0 to 4 {
        %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %61 = vector.extract %45[%60, %59] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_33 = arith.constant 0 : index
          %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%59)
          %c0_34 = arith.constant 0 : index
          %63 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%59)
          %64 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_33, %63)
          %c0_35 = arith.constant 0 : index
          %65 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_25, %60, %62, %c0, %c0_34, %64, %c0_35)
          %66 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_25, %60, %62, %c0, %c0_34, %64, %c0_35)
          affine.store %61, %43[%65, %66] {tiled = true} : memref<128x32xf16, 1>
          %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %68 = vector.extract %45[%60, %67] : f16 from vector<2x8xf16>
          %c0_36 = arith.constant 0 : index
          %c0_37 = arith.constant 0 : index
          %69 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%67)
          %c0_38 = arith.constant 0 : index
          %70 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%67)
          %71 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_37, %70)
          %c0_39 = arith.constant 0 : index
          %72 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_25, %60, %69, %c0_36, %c0_38, %71, %c0_39)
          %73 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_25, %60, %69, %c0_36, %c0_38, %71, %c0_39)
          affine.store %68, %43[%72, %73] {tiled = true} : memref<128x32xf16, 1>
          %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %75 = vector.extract %45[%60, %74] : f16 from vector<2x8xf16>
          %c0_40 = arith.constant 0 : index
          %c0_41 = arith.constant 0 : index
          %76 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%74)
          %c0_42 = arith.constant 0 : index
          %77 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%74)
          %78 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_41, %77)
          %c0_43 = arith.constant 0 : index
          %79 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_25, %60, %76, %c0_40, %c0_42, %78, %c0_43)
          %80 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_25, %60, %76, %c0_40, %c0_42, %78, %c0_43)
          affine.store %75, %43[%79, %80] {tiled = true} : memref<128x32xf16, 1>
          %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %82 = vector.extract %45[%60, %81] : f16 from vector<2x8xf16>
          %c0_44 = arith.constant 0 : index
          %c0_45 = arith.constant 0 : index
          %83 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%81)
          %c0_46 = arith.constant 0 : index
          %84 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%81)
          %85 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_45, %84)
          %c0_47 = arith.constant 0 : index
          %86 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_25, %60, %83, %c0_44, %c0_46, %85, %c0_47)
          %87 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_25, %60, %83, %c0_44, %c0_46, %85, %c0_47)
          affine.store %82, %43[%86, %87] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %48 = frisk.to_threadTile %14#4 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x32xf32>
    %49 = frisk.to_threadTile %14#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_27 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %50 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_27) -> (vector<2x32xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %73 = vector.extract %49[%60, %c0] : f32 from vector<2x1xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %c0_34 = arith.constant 0 : index
        %75 = vector.extract %49[%60, %c0_34] : f32 from vector<2x1xf32>
        %76 = vector.insert %75, %74 [0, 1] : f32 into vector<1x4xf32>
        %c0_35 = arith.constant 0 : index
        %77 = vector.extract %49[%60, %c0_35] : f32 from vector<2x1xf32>
        %78 = vector.insert %77, %76 [0, 2] : f32 into vector<1x4xf32>
        %c0_36 = arith.constant 0 : index
        %79 = vector.extract %49[%60, %c0_36] : f32 from vector<2x1xf32>
        %80 = vector.insert %79, %78 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %80 : vector<1x4xf32>
      }
      %62 = vector.extract %61[0, 0] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %arg5 [%60, %59] : f32 into vector<2x32xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%60, %64] : f32 into vector<2x32xf32>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f32 from vector<1x4xf32>
      %69 = vector.insert %68, %66 [%60, %67] : f32 into vector<2x32xf32>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f32 from vector<1x4xf32>
      %72 = vector.insert %71, %69 [%60, %70] : f32 into vector<2x32xf32>
      affine.yield %72 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
    %cst_28 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %51 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_28) -> (vector<2x32xf32>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %73 = vector.extract %48[%60, %59] : f32 from vector<2x32xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %76 = vector.extract %48[%60, %75] : f32 from vector<2x32xf32>
        %77 = vector.insert %76, %74 [0, 1] : f32 into vector<1x4xf32>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %79 = vector.extract %48[%60, %78] : f32 from vector<2x32xf32>
        %80 = vector.insert %79, %77 [0, 2] : f32 into vector<1x4xf32>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %82 = vector.extract %48[%60, %81] : f32 from vector<2x32xf32>
        %83 = vector.insert %82, %80 [0, 3] : f32 into vector<1x4xf32>
        %cst_34 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %84 = vector.extract %50[%60, %59] : f32 from vector<2x32xf32>
        %85 = vector.insert %84, %cst_34 [0, 0] : f32 into vector<1x4xf32>
        %86 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %87 = vector.extract %50[%60, %86] : f32 from vector<2x32xf32>
        %88 = vector.insert %87, %85 [0, 1] : f32 into vector<1x4xf32>
        %89 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %90 = vector.extract %50[%60, %89] : f32 from vector<2x32xf32>
        %91 = vector.insert %90, %88 [0, 2] : f32 into vector<1x4xf32>
        %92 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %93 = vector.extract %50[%60, %92] : f32 from vector<2x32xf32>
        %94 = vector.insert %93, %91 [0, 3] : f32 into vector<1x4xf32>
        %95 = arith.divf %83, %94 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %95 : vector<1x4xf32>
      }
      %62 = vector.extract %61[0, 0] : f32 from vector<1x4xf32>
      %63 = vector.insert %62, %arg5 [%60, %59] : f32 into vector<2x32xf32>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f32 from vector<1x4xf32>
      %66 = vector.insert %65, %63 [%60, %64] : f32 into vector<2x32xf32>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f32 from vector<1x4xf32>
      %69 = vector.insert %68, %66 [%60, %67] : f32 into vector<2x32xf32>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f32 from vector<1x4xf32>
      %72 = vector.insert %71, %69 [%60, %70] : f32 into vector<2x32xf32>
      affine.yield %72 : vector<2x32xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %cst_29 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %52 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %cst_29) -> (vector<2x32xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %73 = vector.extract %51[%60, %59] : f32 from vector<2x32xf32>
        %74 = vector.insert %73, %cst_33 [0, 0] : f32 into vector<1x4xf32>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
        %76 = vector.extract %51[%60, %75] : f32 from vector<2x32xf32>
        %77 = vector.insert %76, %74 [0, 1] : f32 into vector<1x4xf32>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
        %79 = vector.extract %51[%60, %78] : f32 from vector<2x32xf32>
        %80 = vector.insert %79, %77 [0, 2] : f32 into vector<1x4xf32>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
        %82 = vector.extract %51[%60, %81] : f32 from vector<2x32xf32>
        %83 = vector.insert %82, %80 [0, 3] : f32 into vector<1x4xf32>
        %84 = arith.truncf %83 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %84 : vector<1x4xf16>
      }
      %62 = vector.extract %61[0, 0] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %arg5 [%60, %59] : f16 into vector<2x32xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%60, %64] : f16 into vector<2x32xf16>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f16 from vector<1x4xf16>
      %69 = vector.insert %68, %66 [%60, %67] : f16 into vector<2x32xf16>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f16 from vector<1x4xf16>
      %72 = vector.insert %71, %69 [%60, %70] : f16 into vector<2x32xf16>
      affine.yield %72 : vector<2x32xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
    %53 = frisk.from_threadTile %52 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x32xf16, 3>
    %54 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %8 * 16384 + 96) floordiv 524288) mod 32, (%8 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %55 = frisk.to_threadTile %53 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, logical_transform = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
    %cst_30 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %56 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_30) -> (vector<2x8xf16>) {
      %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
      %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
      %61 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %cst_33 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %73 = vector.extract %55[%60, %59] : f16 from vector<2x8xf16>
        %74 = vector.insert %73, %cst_33 [0, 0] : f16 into vector<1x4xf16>
        %75 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
        %76 = vector.extract %55[%60, %75] : f16 from vector<2x8xf16>
        %77 = vector.insert %76, %74 [0, 1] : f16 into vector<1x4xf16>
        %78 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
        %79 = vector.extract %55[%60, %78] : f16 from vector<2x8xf16>
        %80 = vector.insert %79, %77 [0, 2] : f16 into vector<1x4xf16>
        %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
        %82 = vector.extract %55[%60, %81] : f16 from vector<2x8xf16>
        %83 = vector.insert %82, %80 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
        frisk.fragment_yield %83 : vector<1x4xf16>
      }
      %62 = vector.extract %61[0, 0] : f16 from vector<1x4xf16>
      %63 = vector.insert %62, %arg5 [%60, %59] : f16 into vector<2x8xf16>
      %64 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
      %65 = vector.extract %61[0, 1] : f16 from vector<1x4xf16>
      %66 = vector.insert %65, %63 [%60, %64] : f16 into vector<2x8xf16>
      %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
      %68 = vector.extract %61[0, 2] : f16 from vector<1x4xf16>
      %69 = vector.insert %68, %66 [%60, %67] : f16 into vector<2x8xf16>
      %70 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
      %71 = vector.extract %61[0, 3] : f16 from vector<1x4xf16>
      %72 = vector.insert %71, %69 [%60, %70] : f16 into vector<2x8xf16>
      affine.yield %72 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_31 = gpu.thread_id x
    %57 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_31)
    %c4_32 = arith.constant 4 : index
    %58 = arith.cmpi ult, %57, %c4_32 : index
    scf.if %58 {
      affine.for %arg4 = 0 to 4 {
        %59 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>(%arg4)
        %60 = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 2)>(%arg4)
        frisk.fragment "store" {
          %61 = vector.extract %56[%60, %59] : f16 from vector<2x8xf16>
          %c0 = arith.constant 0 : index
          %c0_33 = arith.constant 0 : index
          %62 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%59)
          %c0_34 = arith.constant 0 : index
          %63 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%59)
          %64 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_33, %63)
          %c0_35 = arith.constant 0 : index
          %65 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_31, %60, %62, %c0, %c0_34, %64, %c0_35)
          %66 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_31, %60, %62, %c0, %c0_34, %64, %c0_35)
          affine.store %61, %54[%65, %66] {tiled = true} : memref<128x32xf16, 1>
          %67 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>(%arg4)
          %68 = vector.extract %56[%60, %67] : f16 from vector<2x8xf16>
          %c0_36 = arith.constant 0 : index
          %c0_37 = arith.constant 0 : index
          %69 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%67)
          %c0_38 = arith.constant 0 : index
          %70 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%67)
          %71 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_37, %70)
          %c0_39 = arith.constant 0 : index
          %72 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_31, %60, %69, %c0_36, %c0_38, %71, %c0_39)
          %73 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_31, %60, %69, %c0_36, %c0_38, %71, %c0_39)
          affine.store %68, %54[%72, %73] {tiled = true} : memref<128x32xf16, 1>
          %74 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>(%arg4)
          %75 = vector.extract %56[%60, %74] : f16 from vector<2x8xf16>
          %c0_40 = arith.constant 0 : index
          %c0_41 = arith.constant 0 : index
          %76 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%74)
          %c0_42 = arith.constant 0 : index
          %77 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%74)
          %78 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_41, %77)
          %c0_43 = arith.constant 0 : index
          %79 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_31, %60, %76, %c0_40, %c0_42, %78, %c0_43)
          %80 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_31, %60, %76, %c0_40, %c0_42, %78, %c0_43)
          affine.store %75, %54[%79, %80] {tiled = true} : memref<128x32xf16, 1>
          %81 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>(%arg4)
          %82 = vector.extract %56[%60, %81] : f16 from vector<2x8xf16>
          %c0_44 = arith.constant 0 : index
          %c0_45 = arith.constant 0 : index
          %83 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%81)
          %c0_46 = arith.constant 0 : index
          %84 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%81)
          %85 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_45, %84)
          %c0_47 = arith.constant 0 : index
          %86 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_31, %60, %83, %c0_44, %c0_46, %85, %c0_47)
          %87 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>(%thread_id_x_31, %60, %83, %c0_44, %c0_46, %85, %c0_47)
          affine.store %82, %54[%86, %87] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}
