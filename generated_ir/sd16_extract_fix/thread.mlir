
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
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_z = gpu.block_id z {range = 1 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %block_id_x = gpu.block_id x {range = 32 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %0 = arith.truncf %cst : f32 to f16
    %1 = arith.muli %block_id_y, %c128 : index
    affine.for %arg4 = 0 to 128 step 32 {
      %cst_2 = arith.constant dense<0.000000e+00> : vector<128x16xf32>
      %cst_3 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
      %cst_4 = arith.constant dense<0.000000e+00> : vector<128x16xf32>
      %2:3 = affine.for %arg5 = 0 to affine_map<()[s0] -> (s0 * 128 + 128)>()[%block_id_y] step 128 iter_args(%arg6 = %cst_2, %arg7 = %cst_3, %arg8 = %cst_4) -> (vector<128x16xf32>, vector<128x1xf32>, vector<128x16xf32>) {
        %cst_23 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
        %33 = affine.for %arg9 = 0 to 128 step 16 iter_args(%arg10 = %cst_23) -> (vector<128x128xf32>) {
          %alloc_47 = memref.alloc() alignment = 16 {frisk.shared_pack = 4 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x16xf16, 3>
          %alloc_48 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 2 : i64, tiled = true} : memref<16x128xf16, 3>
          %61 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %block_id_y * 16384 + %arg9) floordiv 524288) mod 32, (%block_id_z * 131072 + %block_id_x * 4096 + %block_id_y * 128 + %arg9 floordiv 128) mod 4096, %arg9 mod 128], ranges = [128, 16] : memref<1x32x4096x128xf16, 1> -> memref<128x16xf16, 1>
          frisk.async_copy_gm_to_shm %61 to %alloc_47 {alignment = 16 : i64} : memref<128x16xf16, 1>, memref<128x16xf16, 3>
          %62 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %arg9 * 4096 + %arg5) floordiv 524288) mod 32, (%block_id_z * 4096 + %block_id_x * 128 + %arg9 + %arg5 floordiv 4096) mod 128, %arg5 mod 4096], ranges = [16, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<16x128xf16, 1>
          frisk.wait_buffer_ready %alloc_47 : memref<128x16xf16, 3>
          frisk.async_copy_gm_to_shm %62 to %alloc_48 {alignment = 16 : i64} : memref<16x128xf16, 1>, memref<16x128xf16, 3>
          %63 = frisk.to_threadTile %alloc_47 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 1>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x16xf16, 3> -> vector<2x4xf16>
          %64 = vector.broadcast %0 : f16 to vector<2x4xf16>
          %cst_49 = arith.constant dense<0.000000e+00> : vector<2x4xf16>
          %65 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_49) -> (vector<2x4xf16>) {
            %c0 = arith.constant 0 : index
            %72 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg11)
            %73 = frisk.fragment "compute" -> (vector<1x4xf16>) {
              %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
              %85 = vector.extract %63[%72, %c0] : f16 from vector<2x4xf16>
              %86 = vector.insert %85, %cst_56 [0, 0] : f16 into vector<1x4xf16>
              %87 = affine.apply affine_map<() -> (1)>()
              %88 = vector.extract %63[%72, %87] : f16 from vector<2x4xf16>
              %89 = vector.insert %88, %86 [0, 1] : f16 into vector<1x4xf16>
              %90 = affine.apply affine_map<() -> (2)>()
              %91 = vector.extract %63[%72, %90] : f16 from vector<2x4xf16>
              %92 = vector.insert %91, %89 [0, 2] : f16 into vector<1x4xf16>
              %93 = affine.apply affine_map<() -> (3)>()
              %94 = vector.extract %63[%72, %93] : f16 from vector<2x4xf16>
              %95 = vector.insert %94, %92 [0, 3] : f16 into vector<1x4xf16>
              %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
              %96 = vector.extract %64[%72, %c0] : f16 from vector<2x4xf16>
              %97 = vector.insert %96, %cst_57 [0, 0] : f16 into vector<1x4xf16>
              %98 = affine.apply affine_map<() -> (1)>()
              %99 = vector.extract %64[%72, %98] : f16 from vector<2x4xf16>
              %100 = vector.insert %99, %97 [0, 1] : f16 into vector<1x4xf16>
              %101 = affine.apply affine_map<() -> (2)>()
              %102 = vector.extract %64[%72, %101] : f16 from vector<2x4xf16>
              %103 = vector.insert %102, %100 [0, 2] : f16 into vector<1x4xf16>
              %104 = affine.apply affine_map<() -> (3)>()
              %105 = vector.extract %64[%72, %104] : f16 from vector<2x4xf16>
              %106 = vector.insert %105, %103 [0, 3] : f16 into vector<1x4xf16>
              %107 = arith.mulf %95, %106 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
              frisk.fragment_yield %107 : vector<1x4xf16>
            }
            %74 = vector.extract %73[0, 0] : f16 from vector<1x4xf16>
            %75 = vector.insert %74, %arg12 [%72, %c0] : f16 into vector<2x4xf16>
            %76 = affine.apply affine_map<() -> (1)>()
            %77 = vector.extract %73[0, 1] : f16 from vector<1x4xf16>
            %78 = vector.insert %77, %75 [%72, %76] : f16 into vector<2x4xf16>
            %79 = affine.apply affine_map<() -> (2)>()
            %80 = vector.extract %73[0, 2] : f16 from vector<1x4xf16>
            %81 = vector.insert %80, %78 [%72, %79] : f16 into vector<2x4xf16>
            %82 = affine.apply affine_map<() -> (3)>()
            %83 = vector.extract %73[0, 3] : f16 from vector<1x4xf16>
            %84 = vector.insert %83, %81 [%72, %82] : f16 into vector<2x4xf16>
            affine.yield %84 : vector<2x4xf16>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
          frisk.wait_buffer_ready %alloc_48 : memref<16x128xf16, 3>
          %thread_id_x_50 = gpu.thread_id x
          %cst_51 = arith.constant dense<0.000000e+00> : vector<4x8xf16>
          %66 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_51) -> (vector<4x8xf16>) {
            %72 = affine.apply affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8)>(%arg11)
            %c0 = arith.constant 0 : index
            %73 = frisk.fragment "load" -> (vector<4x2xf16>) {
              %100 = affine.apply affine_map<()[s0] -> ((s0 mod 64) floordiv 4)>()[%thread_id_x_50]
              %101 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + (d0 floordiv 4) * -128 + s0 * 8 - (s0 floordiv 4) * 32)>(%arg11)[%thread_id_x_50]
              %102 = frisk.read_matrix_from_shm %alloc_48[%100, %101] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<16x128xf16, 3> -> vector<4xi32>
              %103 = vector.bitcast %102 : vector<4xi32> to vector<8xf16>
              %cst_56 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
              %104 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>()[%thread_id_x_50]
              %105 = vector.extract %103[%104] : f16 from vector<8xf16>
              %106 = vector.insert %105, %cst_56 [0, 0] : f16 into vector<4x2xf16>
              %107 = affine.apply affine_map<(d0)[s0] -> ((d0 * 2 + (d0 floordiv 4) * -8 + ((d0 * 32 + (d0 floordiv 4) * -128 + s0 mod 16 + 16) floordiv 32) * -2 + (s0 mod 16) floordiv 16 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>(%arg11)[%thread_id_x_50]
              %108 = vector.extract %103[%107] : f16 from vector<8xf16>
              %109 = vector.insert %108, %106 [0, 1] : f16 into vector<4x2xf16>
              %110 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 16 + 1) floordiv 16) * 8 + 1)>()[%thread_id_x_50]
              %111 = vector.extract %103[%110] : f16 from vector<8xf16>
              %112 = vector.insert %111, %109 [1, 0] : f16 into vector<4x2xf16>
              %113 = affine.apply affine_map<(d0)[s0] -> ((d0 * 2 + (d0 floordiv 4) * -8 + ((d0 * 32 + (d0 floordiv 4) * -128 + s0 mod 16 + 16) floordiv 32) * -2 + (s0 mod 16) floordiv 16 + 1) * 4 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 16 + 1) floordiv 16) * 8 + 1)>(%arg11)[%thread_id_x_50]
              %114 = vector.extract %103[%113] : f16 from vector<8xf16>
              %115 = vector.insert %114, %112 [1, 1] : f16 into vector<4x2xf16>
              %116 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + 1) floordiv 8) * 8 + 2)>()[%thread_id_x_50]
              %117 = vector.extract %103[%116] : f16 from vector<8xf16>
              %118 = vector.insert %117, %115 [2, 0] : f16 into vector<4x2xf16>
              %119 = affine.apply affine_map<(d0)[s0] -> ((d0 * 2 + (d0 floordiv 4) * -8 + ((d0 * 32 + (d0 floordiv 4) * -128 + s0 mod 16 + 16) floordiv 32) * -2 + (s0 mod 16) floordiv 16 + 1) * 4 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + 1) floordiv 8) * 8 + 2)>(%arg11)[%thread_id_x_50]
              %120 = vector.extract %103[%119] : f16 from vector<8xf16>
              %121 = vector.insert %120, %118 [2, 1] : f16 into vector<4x2xf16>
              %122 = affine.apply affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 16 + 3) floordiv 16) * 8 + 3)>()[%thread_id_x_50]
              %123 = vector.extract %103[%122] : f16 from vector<8xf16>
              %124 = vector.insert %123, %121 [3, 0] : f16 into vector<4x2xf16>
              %125 = affine.apply affine_map<(d0)[s0] -> ((d0 * 2 + (d0 floordiv 4) * -8 + ((d0 * 32 + (d0 floordiv 4) * -128 + s0 mod 16 + 16) floordiv 32) * -2 + (s0 mod 16) floordiv 16 + 1) * 4 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 16 + 3) floordiv 16) * 8 + 3)>(%arg11)[%thread_id_x_50]
              %126 = vector.extract %103[%125] : f16 from vector<8xf16>
              %127 = vector.insert %126, %124 [3, 1] : f16 into vector<4x2xf16>
              frisk.fragment_yield %127 : vector<4x2xf16>
            }
            %74 = vector.extract %73[0, 0] : f16 from vector<4x2xf16>
            %75 = vector.insert %74, %arg12 [%c0, %72] : f16 into vector<4x8xf16>
            %76 = affine.apply affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8 + 1)>(%arg11)
            %77 = vector.extract %73[0, 1] : f16 from vector<4x2xf16>
            %78 = vector.insert %77, %75 [%c0, %76] : f16 into vector<4x8xf16>
            %79 = affine.apply affine_map<() -> (1)>()
            %80 = vector.extract %73[1, 0] : f16 from vector<4x2xf16>
            %81 = vector.insert %80, %78 [%79, %72] : f16 into vector<4x8xf16>
            %82 = affine.apply affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8 + 1)>(%arg11)
            %83 = affine.apply affine_map<() -> (1)>()
            %84 = vector.extract %73[1, 1] : f16 from vector<4x2xf16>
            %85 = vector.insert %84, %81 [%83, %82] : f16 into vector<4x8xf16>
            %86 = affine.apply affine_map<() -> (2)>()
            %87 = vector.extract %73[2, 0] : f16 from vector<4x2xf16>
            %88 = vector.insert %87, %85 [%86, %72] : f16 into vector<4x8xf16>
            %89 = affine.apply affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8 + 1)>(%arg11)
            %90 = affine.apply affine_map<() -> (2)>()
            %91 = vector.extract %73[2, 1] : f16 from vector<4x2xf16>
            %92 = vector.insert %91, %88 [%90, %89] : f16 into vector<4x8xf16>
            %93 = affine.apply affine_map<() -> (3)>()
            %94 = vector.extract %73[3, 0] : f16 from vector<4x2xf16>
            %95 = vector.insert %94, %92 [%93, %72] : f16 into vector<4x8xf16>
            %96 = affine.apply affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8 + 1)>(%arg11)
            %97 = affine.apply affine_map<() -> (3)>()
            %98 = vector.extract %73[3, 1] : f16 from vector<4x2xf16>
            %99 = vector.insert %98, %95 [%97, %96] : f16 into vector<4x8xf16>
            affine.yield %99 : vector<4x8xf16>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 4, 8>, iterLabel = "matrix_load", tiled = true}
          %cst_52 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
          %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %67 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_52) -> (vector<2x32xf32>) {
            %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
            %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
            %74 = affine.for %arg13 = 0 to 1 iter_args(%arg14 = %cst_53) -> (vector<1x4xf32>) {
              %86 = frisk.fragment "load" -> (vector<1x4xf16>) {
                %89 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
                %90 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
                %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
                %91 = vector.extract %65[%89, %90] : f16 from vector<2x4xf16>
                %92 = vector.insert %91, %cst_56 [0, 0] : f16 into vector<1x4xf16>
                %93 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
                %94 = vector.extract %65[%89, %93] : f16 from vector<2x4xf16>
                %95 = vector.insert %94, %92 [0, 1] : f16 into vector<1x4xf16>
                %96 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
                %97 = vector.extract %65[%89, %96] : f16 from vector<2x4xf16>
                %98 = vector.insert %97, %95 [0, 2] : f16 into vector<1x4xf16>
                %99 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
                %100 = vector.extract %65[%89, %99] : f16 from vector<2x4xf16>
                %101 = vector.insert %100, %98 [0, 3] : f16 into vector<1x4xf16>
                frisk.fragment_yield %101 : vector<1x4xf16>
              }
              %87 = frisk.fragment "load" -> (vector<4x1xf16>) {
                %89 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg13)
                %90 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg11)
                %cst_56 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
                %91 = vector.extract %66[%89, %90] : f16 from vector<4x8xf16>
                %92 = vector.insert %91, %cst_56 [0, 0] : f16 into vector<4x1xf16>
                %93 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg13)
                %94 = vector.extract %66[%93, %90] : f16 from vector<4x8xf16>
                %95 = vector.insert %94, %92 [1, 0] : f16 into vector<4x1xf16>
                %96 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg13)
                %97 = vector.extract %66[%96, %90] : f16 from vector<4x8xf16>
                %98 = vector.insert %97, %95 [2, 0] : f16 into vector<4x1xf16>
                %99 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg13)
                %100 = vector.extract %66[%99, %90] : f16 from vector<4x8xf16>
                %101 = vector.insert %100, %98 [3, 0] : f16 into vector<4x1xf16>
                frisk.fragment_yield %101 : vector<4x1xf16>
              }
              %88 = frisk.fragment "compute" -> (vector<1x4xf32>) {
                %89 = frisk.warp_mma_rr %86, %87, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
                frisk.fragment_yield %89 : vector<1x4xf32>
              }
              affine.yield %88 : vector<1x4xf32>
            } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
            %75 = vector.extract %74[0, 0] : f32 from vector<1x4xf32>
            %76 = vector.insert %75, %arg12 [%73, %72] : f32 into vector<2x32xf32>
            %77 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
            %78 = vector.extract %74[0, 1] : f32 from vector<1x4xf32>
            %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x32xf32>
            %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
            %81 = vector.extract %74[0, 2] : f32 from vector<1x4xf32>
            %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x32xf32>
            %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
            %84 = vector.extract %74[0, 3] : f32 from vector<1x4xf32>
            %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x32xf32>
            affine.yield %85 : vector<2x32xf32>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
          %68 = frisk.to_threadTile %arg10 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
          %cst_54 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
          %69 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_54) -> (vector<2x32xf32>) {
            %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
            %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
            %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
              %86 = vector.extract %68[%73, %72] : f32 from vector<2x32xf32>
              %87 = vector.insert %86, %cst_56 [0, 0] : f32 into vector<1x4xf32>
              %88 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
              %89 = vector.extract %68[%73, %88] : f32 from vector<2x32xf32>
              %90 = vector.insert %89, %87 [0, 1] : f32 into vector<1x4xf32>
              %91 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
              %92 = vector.extract %68[%73, %91] : f32 from vector<2x32xf32>
              %93 = vector.insert %92, %90 [0, 2] : f32 into vector<1x4xf32>
              %94 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
              %95 = vector.extract %68[%73, %94] : f32 from vector<2x32xf32>
              %96 = vector.insert %95, %93 [0, 3] : f32 into vector<1x4xf32>
              %cst_57 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
              %97 = vector.extract %67[%73, %72] : f32 from vector<2x32xf32>
              %98 = vector.insert %97, %cst_57 [0, 0] : f32 into vector<1x4xf32>
              %99 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
              %100 = vector.extract %67[%73, %99] : f32 from vector<2x32xf32>
              %101 = vector.insert %100, %98 [0, 1] : f32 into vector<1x4xf32>
              %102 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
              %103 = vector.extract %67[%73, %102] : f32 from vector<2x32xf32>
              %104 = vector.insert %103, %101 [0, 2] : f32 into vector<1x4xf32>
              %105 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
              %106 = vector.extract %67[%73, %105] : f32 from vector<2x32xf32>
              %107 = vector.insert %106, %104 [0, 3] : f32 into vector<1x4xf32>
              %108 = arith.addf %96, %107 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
              frisk.fragment_yield %108 : vector<1x4xf32>
            }
            %75 = vector.extract %74[0, 0] : f32 from vector<1x4xf32>
            %76 = vector.insert %75, %arg12 [%73, %72] : f32 into vector<2x32xf32>
            %77 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
            %78 = vector.extract %74[0, 1] : f32 from vector<1x4xf32>
            %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x32xf32>
            %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
            %81 = vector.extract %74[0, 2] : f32 from vector<1x4xf32>
            %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x32xf32>
            %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
            %84 = vector.extract %74[0, 3] : f32 from vector<1x4xf32>
            %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x32xf32>
            affine.yield %85 : vector<2x32xf32>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
          %cst_55 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
          %70 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_55) -> (vector<2x32xf32>) {
            %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
            %73 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg11)
            %74 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %cst_56 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
              %86 = vector.extract %69[%73, %72] : f32 from vector<2x32xf32>
              %87 = vector.insert %86, %cst_56 [0, 0] : f32 into vector<1x4xf32>
              %88 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
              %89 = vector.extract %69[%73, %88] : f32 from vector<2x32xf32>
              %90 = vector.insert %89, %87 [0, 1] : f32 into vector<1x4xf32>
              %91 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
              %92 = vector.extract %69[%73, %91] : f32 from vector<2x32xf32>
              %93 = vector.insert %92, %90 [0, 2] : f32 into vector<1x4xf32>
              %94 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
              %95 = vector.extract %69[%73, %94] : f32 from vector<2x32xf32>
              %96 = vector.insert %95, %93 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
              frisk.fragment_yield %96 : vector<1x4xf32>
            }
            %75 = vector.extract %74[0, 0] : f32 from vector<1x4xf32>
            %76 = vector.insert %75, %arg12 [%73, %72] : f32 into vector<2x32xf32>
            %77 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
            %78 = vector.extract %74[0, 1] : f32 from vector<1x4xf32>
            %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x32xf32>
            %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
            %81 = vector.extract %74[0, 2] : f32 from vector<1x4xf32>
            %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x32xf32>
            %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
            %84 = vector.extract %74[0, 3] : f32 from vector<1x4xf32>
            %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x32xf32>
            affine.yield %85 : vector<2x32xf32>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
          %71 = frisk.from_threadTile %70 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
          affine.yield %71 : vector<128x128xf32>
        } {local_yield_transformed = true}
        %34 = frisk.to_threadTile %33 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
        %thread_id_x_24 = gpu.thread_id x
        %35 = arith.index_cast %thread_id_x_24 : index to i32
        %c0_i32_25 = arith.constant 0 : i32
        %c64_i32_26 = arith.constant 64 : i32
        %c16_i32_27 = arith.constant 16 : i32
        %36 = arith.andi %35, %c16_i32_27 : i32
        %37 = arith.cmpi ne, %36, %c0_i32_25 : i32
        %c32_i32_28 = arith.constant 32 : i32
        %38 = arith.andi %35, %c32_i32_28 : i32
        %39 = arith.cmpi ne, %38, %c0_i32_25 : i32
        %cst_29 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %40 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_29) -> (vector<2x32xf32>) {
          %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg9)
          %62 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg9)
          %63 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %75 = vector.extract %34[%62, %61] : f32 from vector<2x32xf32>
            %76 = vector.insert %75, %cst_47 [0, 0] : f32 into vector<1x4xf32>
            %77 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg9)
            %78 = vector.extract %34[%62, %77] : f32 from vector<2x32xf32>
            %79 = vector.insert %78, %76 [0, 1] : f32 into vector<1x4xf32>
            %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg9)
            %81 = vector.extract %34[%62, %80] : f32 from vector<2x32xf32>
            %82 = vector.insert %81, %79 [0, 2] : f32 into vector<1x4xf32>
            %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg9)
            %84 = vector.extract %34[%62, %83] : f32 from vector<2x32xf32>
            %85 = vector.insert %84, %82 [0, 3] : f32 into vector<1x4xf32>
            %86 = vector.extract %85[0, 0] : f32 from vector<1x4xf32>
            %87 = vector.extract %85[0, 1] : f32 from vector<1x4xf32>
            %88 = vector.extract %85[0, 2] : f32 from vector<1x4xf32>
            %89 = vector.extract %85[0, 3] : f32 from vector<1x4xf32>
            %90 = arith.select %37, %86, %87 : f32
            %shuffleResult, %valid = gpu.shuffle xor %90, %c16_i32_27, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
            %91 = arith.select %37, %shuffleResult, %86 : f32
            %92 = arith.select %37, %87, %shuffleResult : f32
            %93 = arith.select %37, %88, %89 : f32
            %shuffleResult_48, %valid_49 = gpu.shuffle xor %93, %c16_i32_27, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
            %94 = arith.select %37, %shuffleResult_48, %88 : f32
            %95 = arith.select %37, %89, %shuffleResult_48 : f32
            %96 = arith.select %39, %91, %94 : f32
            %shuffleResult_50, %valid_51 = gpu.shuffle xor %96, %c32_i32_28, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
            %97 = arith.select %39, %shuffleResult_50, %91 : f32
            %98 = arith.select %39, %94, %shuffleResult_50 : f32
            %99 = arith.select %39, %92, %95 : f32
            %shuffleResult_52, %valid_53 = gpu.shuffle xor %99, %c32_i32_28, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
            %100 = arith.select %39, %shuffleResult_52, %92 : f32
            %101 = arith.select %39, %95, %shuffleResult_52 : f32
            %102 = vector.insert %97, %85 [0, 0] : f32 into vector<1x4xf32>
            %103 = vector.insert %100, %102 [0, 1] : f32 into vector<1x4xf32>
            %104 = vector.insert %98, %103 [0, 2] : f32 into vector<1x4xf32>
            %105 = vector.insert %101, %104 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
            frisk.fragment_yield %105 : vector<1x4xf32>
          }
          %64 = vector.extract %63[0, 0] : f32 from vector<1x4xf32>
          %65 = vector.insert %64, %arg10 [%62, %61] : f32 into vector<2x32xf32>
          %66 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg9)
          %67 = vector.extract %63[0, 1] : f32 from vector<1x4xf32>
          %68 = vector.insert %67, %65 [%62, %66] : f32 into vector<2x32xf32>
          %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg9)
          %70 = vector.extract %63[0, 2] : f32 from vector<1x4xf32>
          %71 = vector.insert %70, %68 [%62, %69] : f32 into vector<2x32xf32>
          %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg9)
          %73 = vector.extract %63[0, 3] : f32 from vector<1x4xf32>
          %74 = vector.insert %73, %71 [%62, %72] : f32 into vector<2x32xf32>
          affine.yield %74 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %cst_30 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %thread_id_x_31 = gpu.thread_id x
        %41 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_30) -> (vector<2x32xf32>) {
          %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg9)
          %62 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg9)
          %63 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %c0 = arith.constant 0 : index
            %c0_48 = arith.constant 0 : index
            %75 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%61)
            %c0_49 = arith.constant 0 : index
            %76 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%61)
            %77 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%76)
            %c0_50 = arith.constant 0 : index
            %78 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_48, %77)
            %79 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_31, %62, %75, %c0, %c0_49, %c0_50, %78)
            %80 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_31, %62, %75, %c0, %c0_49, %c0_50, %78)
            %81 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%1, %79)
            %82 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %80)
            %83 = arith.addi %81, %c1 : index
            %84 = arith.cmpi ule, %83, %82 : index
            %85 = arith.select %84, %cst_0, %cst_1 : f32
            %86 = vector.extract %40[%62, %61] : f32 from vector<2x32xf32>
            %87 = arith.addf %86, %85 {frisk.fused_mask_add} : f32
            %88 = vector.insert %87, %cst_47 [0, 0] : f32 into vector<1x4xf32>
            %89 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg9)
            %c0_51 = arith.constant 0 : index
            %c0_52 = arith.constant 0 : index
            %90 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%89)
            %c0_53 = arith.constant 0 : index
            %91 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%89)
            %92 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%91)
            %c0_54 = arith.constant 0 : index
            %93 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_52, %92)
            %94 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_31, %62, %90, %c0_51, %c0_53, %c0_54, %93)
            %95 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_31, %62, %90, %c0_51, %c0_53, %c0_54, %93)
            %96 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%1, %94)
            %97 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %95)
            %98 = arith.addi %96, %c1 : index
            %99 = arith.cmpi ule, %98, %97 : index
            %100 = arith.select %99, %cst_0, %cst_1 : f32
            %101 = vector.extract %40[%62, %89] : f32 from vector<2x32xf32>
            %102 = arith.addf %101, %100 {frisk.fused_mask_add} : f32
            %103 = vector.insert %102, %88 [0, 1] : f32 into vector<1x4xf32>
            %104 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg9)
            %c0_55 = arith.constant 0 : index
            %c0_56 = arith.constant 0 : index
            %105 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%104)
            %c0_57 = arith.constant 0 : index
            %106 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%104)
            %107 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%106)
            %c0_58 = arith.constant 0 : index
            %108 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_56, %107)
            %109 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_31, %62, %105, %c0_55, %c0_57, %c0_58, %108)
            %110 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_31, %62, %105, %c0_55, %c0_57, %c0_58, %108)
            %111 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%1, %109)
            %112 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %110)
            %113 = arith.addi %111, %c1 : index
            %114 = arith.cmpi ule, %113, %112 : index
            %115 = arith.select %114, %cst_0, %cst_1 : f32
            %116 = vector.extract %40[%62, %104] : f32 from vector<2x32xf32>
            %117 = arith.addf %116, %115 {frisk.fused_mask_add} : f32
            %118 = vector.insert %117, %103 [0, 2] : f32 into vector<1x4xf32>
            %119 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg9)
            %c0_59 = arith.constant 0 : index
            %c0_60 = arith.constant 0 : index
            %120 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%119)
            %c0_61 = arith.constant 0 : index
            %121 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%119)
            %122 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%121)
            %c0_62 = arith.constant 0 : index
            %123 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_60, %122)
            %124 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_31, %62, %120, %c0_59, %c0_61, %c0_62, %123)
            %125 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_31, %62, %120, %c0_59, %c0_61, %c0_62, %123)
            %126 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%1, %124)
            %127 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg5, %125)
            %128 = arith.addi %126, %c1 : index
            %129 = arith.cmpi ule, %128, %127 : index
            %130 = arith.select %129, %cst_0, %cst_1 : f32
            %131 = vector.extract %40[%62, %119] : f32 from vector<2x32xf32>
            %132 = arith.addf %131, %130 {frisk.fused_mask_add} : f32
            %133 = vector.insert %132, %118 [0, 3] : f32 into vector<1x4xf32>
            frisk.fragment_yield %133 : vector<1x4xf32>
          }
          %64 = vector.extract %63[0, 0] : f32 from vector<1x4xf32>
          %65 = vector.insert %64, %arg10 [%62, %61] : f32 into vector<2x32xf32>
          %66 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg9)
          %67 = vector.extract %63[0, 1] : f32 from vector<1x4xf32>
          %68 = vector.insert %67, %65 [%62, %66] : f32 into vector<2x32xf32>
          %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg9)
          %70 = vector.extract %63[0, 2] : f32 from vector<1x4xf32>
          %71 = vector.insert %70, %68 [%62, %69] : f32 into vector<2x32xf32>
          %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg9)
          %73 = vector.extract %63[0, 3] : f32 from vector<1x4xf32>
          %74 = vector.insert %73, %71 [%62, %72] : f32 into vector<2x32xf32>
          affine.yield %74 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "mask_fragment", tiled = true}
        %cst_32 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %42 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_32) -> (vector<2x32xf32>) {
          %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg9)
          %62 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg9)
          %63 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %75 = vector.extract %41[%62, %61] : f32 from vector<2x32xf32>
            %76 = vector.insert %75, %cst_47 [0, 0] : f32 into vector<1x4xf32>
            %77 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg9)
            %78 = vector.extract %41[%62, %77] : f32 from vector<2x32xf32>
            %79 = vector.insert %78, %76 [0, 1] : f32 into vector<1x4xf32>
            %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg9)
            %81 = vector.extract %41[%62, %80] : f32 from vector<2x32xf32>
            %82 = vector.insert %81, %79 [0, 2] : f32 into vector<1x4xf32>
            %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg9)
            %84 = vector.extract %41[%62, %83] : f32 from vector<2x32xf32>
            %85 = vector.insert %84, %82 [0, 3] : f32 into vector<1x4xf32>
            %86 = math.exp2 %85 {tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %86 : vector<1x4xf32>
          }
          %64 = vector.extract %63[0, 0] : f32 from vector<1x4xf32>
          %65 = vector.insert %64, %arg10 [%62, %61] : f32 into vector<2x32xf32>
          %66 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg9)
          %67 = vector.extract %63[0, 1] : f32 from vector<1x4xf32>
          %68 = vector.insert %67, %65 [%62, %66] : f32 into vector<2x32xf32>
          %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg9)
          %70 = vector.extract %63[0, 2] : f32 from vector<1x4xf32>
          %71 = vector.insert %70, %68 [%62, %69] : f32 into vector<2x32xf32>
          %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg9)
          %73 = vector.extract %63[0, 3] : f32 from vector<1x4xf32>
          %74 = vector.insert %73, %71 [%62, %72] : f32 into vector<2x32xf32>
          affine.yield %74 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %cst_33 = arith.constant 0.000000e+00 : f32
        %cst_34 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
        %43 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_34) -> (vector<2x1xf32>) {
          %c0 = arith.constant 0 : index
          %61 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
          %62 = frisk.fragment "compute" -> (vector<1x1xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
            %65 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_33) -> (f32) {
              %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg11)
              %70 = frisk.fragment "compute" -> (f32) {
                %71 = vector.extract %42[%61, %69] : f32 from vector<2x32xf32>
                %72 = arith.addf %arg12, %71 : f32
                %73 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg11)
                %74 = vector.extract %42[%61, %73] : f32 from vector<2x32xf32>
                %75 = arith.addf %72, %74 : f32
                %76 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg11)
                %77 = vector.extract %42[%61, %76] : f32 from vector<2x32xf32>
                %78 = arith.addf %75, %77 : f32
                %79 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg11)
                %80 = vector.extract %42[%61, %79] : f32 from vector<2x32xf32>
                %81 = arith.addf %78, %80 : f32
                frisk.fragment_yield %81 : f32
              }
              affine.yield %70 : f32
            } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
            %c64_i32_48 = arith.constant 64 : i32
            %c16_i32_49 = arith.constant 16 : i32
            %shuffleResult, %valid = gpu.shuffle xor %65, %c16_i32_49, %c64_i32_48 : f32
            %66 = arith.addf %65, %shuffleResult : f32
            %c64_i32_50 = arith.constant 64 : i32
            %c32_i32_51 = arith.constant 32 : i32
            %shuffleResult_52, %valid_53 = gpu.shuffle xor %66, %c32_i32_51, %c64_i32_50 : f32
            %67 = arith.addf %66, %shuffleResult_52 : f32
            %68 = vector.insert %67, %cst_47 [0, 0] : f32 into vector<1x1xf32>
            frisk.fragment_yield %68 : vector<1x1xf32>
          }
          %63 = vector.extract %62[0, 0] : f32 from vector<1x1xf32>
          %64 = vector.insert %63, %arg10 [%61, %c0] : f32 into vector<2x1xf32>
          affine.yield %64 : vector<2x1xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
        %44 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
        %cst_35 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
        %45 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_35) -> (vector<2x1xf32>) {
          %c0 = arith.constant 0 : index
          %61 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
          %62 = frisk.fragment "compute" -> (vector<1x1xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
            %65 = vector.extract %43[%61, %c0] : f32 from vector<2x1xf32>
            %66 = vector.insert %65, %cst_47 [0, 0] : f32 into vector<1x1xf32>
            %cst_48 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
            %67 = vector.extract %44[%61, %c0] : f32 from vector<2x1xf32>
            %68 = vector.insert %67, %cst_48 [0, 0] : f32 into vector<1x1xf32>
            %69 = arith.addf %66, %68 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
            frisk.fragment_yield %69 : vector<1x1xf32>
          }
          %63 = vector.extract %62[0, 0] : f32 from vector<1x1xf32>
          %64 = vector.insert %63, %arg10 [%61, %c0] : f32 into vector<2x1xf32>
          affine.yield %64 : vector<2x1xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
        %cst_36 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
        %46 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_36) -> (vector<2x32xf16>) {
          %61 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>(%arg9)
          %62 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 2)>(%arg9)
          %63 = frisk.fragment "compute" -> (vector<1x4xf16>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %75 = vector.extract %42[%62, %61] : f32 from vector<2x32xf32>
            %76 = vector.insert %75, %cst_47 [0, 0] : f32 into vector<1x4xf32>
            %77 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg9)
            %78 = vector.extract %42[%62, %77] : f32 from vector<2x32xf32>
            %79 = vector.insert %78, %76 [0, 1] : f32 into vector<1x4xf32>
            %80 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg9)
            %81 = vector.extract %42[%62, %80] : f32 from vector<2x32xf32>
            %82 = vector.insert %81, %79 [0, 2] : f32 into vector<1x4xf32>
            %83 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg9)
            %84 = vector.extract %42[%62, %83] : f32 from vector<2x32xf32>
            %85 = vector.insert %84, %82 [0, 3] : f32 into vector<1x4xf32>
            %86 = arith.truncf %85 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
            frisk.fragment_yield %86 : vector<1x4xf16>
          }
          %64 = vector.extract %63[0, 0] : f16 from vector<1x4xf16>
          %65 = vector.insert %64, %arg10 [%62, %61] : f16 into vector<2x32xf16>
          %66 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>(%arg9)
          %67 = vector.extract %63[0, 1] : f16 from vector<1x4xf16>
          %68 = vector.insert %67, %65 [%62, %66] : f16 into vector<2x32xf16>
          %69 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>(%arg9)
          %70 = vector.extract %63[0, 2] : f16 from vector<1x4xf16>
          %71 = vector.insert %70, %68 [%62, %69] : f16 into vector<2x32xf16>
          %72 = affine.apply affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>(%arg9)
          %73 = vector.extract %63[0, 3] : f16 from vector<1x4xf16>
          %74 = vector.insert %73, %71 [%62, %72] : f16 into vector<2x32xf16>
          affine.yield %74 : vector<2x32xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 4 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_pair_stride = 16 : i64, tiled = true} : memref<128x16xf16, 3>
        %47 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %arg5 * 128 + %arg4) floordiv 524288) mod 32, (%block_id_z * 131072 + %block_id_x * 4096 + %arg5 + %arg4 floordiv 128) mod 4096, %arg4 mod 128], ranges = [128, 16] : memref<1x32x4096x128xf16, 1> -> memref<128x16xf16, 1>
        frisk.async_copy_gm_to_shm %47 to %alloc {alignment = 16 : i64} : memref<128x16xf16, 1>, memref<128x16xf16, 3>
        frisk.wait_buffer_ready %alloc : memref<128x16xf16, 3>
        %cst_37 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
        %cst_38 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %48 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_37) -> (vector<2x4xf32>) {
          %c0 = arith.constant 0 : index
          %61 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
          %62 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_38) -> (vector<1x4xf32>) {
            %74 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %77 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
              %78 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
              %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
              %79 = vector.extract %46[%77, %78] : f16 from vector<2x32xf16>
              %80 = vector.insert %79, %cst_47 [0, 0] : f16 into vector<1x4xf16>
              %81 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
              %82 = vector.extract %46[%77, %81] : f16 from vector<2x32xf16>
              %83 = vector.insert %82, %80 [0, 1] : f16 into vector<1x4xf16>
              %84 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
              %85 = vector.extract %46[%77, %84] : f16 from vector<2x32xf16>
              %86 = vector.insert %85, %83 [0, 2] : f16 into vector<1x4xf16>
              %87 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
              %88 = vector.extract %46[%77, %87] : f16 from vector<2x32xf16>
              %89 = vector.insert %88, %86 [0, 3] : f16 into vector<1x4xf16>
              frisk.fragment_yield %89 : vector<1x4xf16>
            }
            %75 = frisk.fragment "load" -> (vector<4x1xf16>) {
              %77 = frisk.to_threadTile %alloc {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 8, 1>, frisk.fragment_source, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x16xf16, 3> -> vector<32x1xf16>
              %78 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
              %79 = affine.apply affine_map<() -> (0)>()
              %cst_47 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
              %80 = vector.extract %77[%78, %79] : f16 from vector<32x1xf16>
              %81 = vector.insert %80, %cst_47 [0, 0] : f16 into vector<4x1xf16>
              %82 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
              %83 = vector.extract %77[%82, %79] : f16 from vector<32x1xf16>
              %84 = vector.insert %83, %81 [1, 0] : f16 into vector<4x1xf16>
              %85 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
              %86 = vector.extract %77[%85, %79] : f16 from vector<32x1xf16>
              %87 = vector.insert %86, %84 [2, 0] : f16 into vector<4x1xf16>
              %88 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
              %89 = vector.extract %77[%88, %79] : f16 from vector<32x1xf16>
              %90 = vector.insert %89, %87 [3, 0] : f16 into vector<4x1xf16>
              frisk.fragment_yield %90 : vector<4x1xf16>
            }
            %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %77 = frisk.warp_mma_rr %74, %75, %arg12 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %77 : vector<1x4xf32>
            }
            affine.yield %76 : vector<1x4xf32>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
          %64 = vector.insert %63, %arg10 [%61, %c0] : f32 into vector<2x4xf32>
          %65 = affine.apply affine_map<() -> (1)>()
          %66 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
          %67 = vector.insert %66, %64 [%61, %65] : f32 into vector<2x4xf32>
          %68 = affine.apply affine_map<() -> (2)>()
          %69 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
          %70 = vector.insert %69, %67 [%61, %68] : f32 into vector<2x4xf32>
          %71 = affine.apply affine_map<() -> (3)>()
          %72 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
          %73 = vector.insert %72, %70 [%61, %71] : f32 into vector<2x4xf32>
          affine.yield %73 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "gemm_mn", tiled = true}
        %49 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 1>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x16xf32> -> vector<2x4xf32>
        %cst_39 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
        %50 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_39) -> (vector<2x4xf32>) {
          %c0 = arith.constant 0 : index
          %61 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
          %62 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %74 = vector.extract %49[%61, %c0] : f32 from vector<2x4xf32>
            %75 = vector.insert %74, %cst_47 [0, 0] : f32 into vector<1x4xf32>
            %76 = affine.apply affine_map<() -> (1)>()
            %77 = vector.extract %49[%61, %76] : f32 from vector<2x4xf32>
            %78 = vector.insert %77, %75 [0, 1] : f32 into vector<1x4xf32>
            %79 = affine.apply affine_map<() -> (2)>()
            %80 = vector.extract %49[%61, %79] : f32 from vector<2x4xf32>
            %81 = vector.insert %80, %78 [0, 2] : f32 into vector<1x4xf32>
            %82 = affine.apply affine_map<() -> (3)>()
            %83 = vector.extract %49[%61, %82] : f32 from vector<2x4xf32>
            %84 = vector.insert %83, %81 [0, 3] : f32 into vector<1x4xf32>
            %cst_48 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %85 = vector.extract %48[%61, %c0] : f32 from vector<2x4xf32>
            %86 = vector.insert %85, %cst_48 [0, 0] : f32 into vector<1x4xf32>
            %87 = affine.apply affine_map<() -> (1)>()
            %88 = vector.extract %48[%61, %87] : f32 from vector<2x4xf32>
            %89 = vector.insert %88, %86 [0, 1] : f32 into vector<1x4xf32>
            %90 = affine.apply affine_map<() -> (2)>()
            %91 = vector.extract %48[%61, %90] : f32 from vector<2x4xf32>
            %92 = vector.insert %91, %89 [0, 2] : f32 into vector<1x4xf32>
            %93 = affine.apply affine_map<() -> (3)>()
            %94 = vector.extract %48[%61, %93] : f32 from vector<2x4xf32>
            %95 = vector.insert %94, %92 [0, 3] : f32 into vector<1x4xf32>
            %96 = arith.addf %84, %95 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %96 : vector<1x4xf32>
          }
          %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
          %64 = vector.insert %63, %arg10 [%61, %c0] : f32 into vector<2x4xf32>
          %65 = affine.apply affine_map<() -> (1)>()
          %66 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
          %67 = vector.insert %66, %64 [%61, %65] : f32 into vector<2x4xf32>
          %68 = affine.apply affine_map<() -> (2)>()
          %69 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
          %70 = vector.insert %69, %67 [%61, %68] : f32 into vector<2x4xf32>
          %71 = affine.apply affine_map<() -> (3)>()
          %72 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
          %73 = vector.insert %72, %70 [%61, %71] : f32 into vector<2x4xf32>
          affine.yield %73 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
        %alloc_40 = memref.alloc() alignment = 16 {frisk.shared_pack = 4 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_pair_stride = 16 : i64, tiled = true} : memref<128x16xf16, 3>
        %51 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %arg5 * 128 + %arg4 + 16) floordiv 524288) mod 32, (%block_id_z * 131072 + %block_id_x * 4096 + %arg5 + (%arg4 + 16) floordiv 128) mod 4096, (%arg4 + 16) mod 128], ranges = [128, 16] : memref<1x32x4096x128xf16, 1> -> memref<128x16xf16, 1>
        frisk.async_copy_gm_to_shm %51 to %alloc_40 {alignment = 16 : i64} : memref<128x16xf16, 1>, memref<128x16xf16, 3>
        frisk.wait_buffer_ready %alloc_40 : memref<128x16xf16, 3>
        %cst_41 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
        %cst_42 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %52 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_41) -> (vector<2x4xf32>) {
          %c0 = arith.constant 0 : index
          %61 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
          %62 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_42) -> (vector<1x4xf32>) {
            %74 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %77 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
              %78 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
              %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
              %79 = vector.extract %46[%77, %78] : f16 from vector<2x32xf16>
              %80 = vector.insert %79, %cst_47 [0, 0] : f16 into vector<1x4xf16>
              %81 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
              %82 = vector.extract %46[%77, %81] : f16 from vector<2x32xf16>
              %83 = vector.insert %82, %80 [0, 1] : f16 into vector<1x4xf16>
              %84 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
              %85 = vector.extract %46[%77, %84] : f16 from vector<2x32xf16>
              %86 = vector.insert %85, %83 [0, 2] : f16 into vector<1x4xf16>
              %87 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
              %88 = vector.extract %46[%77, %87] : f16 from vector<2x32xf16>
              %89 = vector.insert %88, %86 [0, 3] : f16 into vector<1x4xf16>
              frisk.fragment_yield %89 : vector<1x4xf16>
            }
            %75 = frisk.fragment "load" -> (vector<4x1xf16>) {
              %77 = frisk.to_threadTile %alloc_40 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 8, 1>, frisk.fragment_source, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4)>, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x16xf16, 3> -> vector<32x1xf16>
              %78 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg11)
              %79 = affine.apply affine_map<() -> (0)>()
              %cst_47 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
              %80 = vector.extract %77[%78, %79] : f16 from vector<32x1xf16>
              %81 = vector.insert %80, %cst_47 [0, 0] : f16 into vector<4x1xf16>
              %82 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg11)
              %83 = vector.extract %77[%82, %79] : f16 from vector<32x1xf16>
              %84 = vector.insert %83, %81 [1, 0] : f16 into vector<4x1xf16>
              %85 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg11)
              %86 = vector.extract %77[%85, %79] : f16 from vector<32x1xf16>
              %87 = vector.insert %86, %84 [2, 0] : f16 into vector<4x1xf16>
              %88 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg11)
              %89 = vector.extract %77[%88, %79] : f16 from vector<32x1xf16>
              %90 = vector.insert %89, %87 [3, 0] : f16 into vector<4x1xf16>
              frisk.fragment_yield %90 : vector<4x1xf16>
            }
            %76 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %77 = frisk.warp_mma_rr %74, %75, %arg12 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %77 : vector<1x4xf32>
            }
            affine.yield %76 : vector<1x4xf32>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
          %64 = vector.insert %63, %arg10 [%61, %c0] : f32 into vector<2x4xf32>
          %65 = affine.apply affine_map<() -> (1)>()
          %66 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
          %67 = vector.insert %66, %64 [%61, %65] : f32 into vector<2x4xf32>
          %68 = affine.apply affine_map<() -> (2)>()
          %69 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
          %70 = vector.insert %69, %67 [%61, %68] : f32 into vector<2x4xf32>
          %71 = affine.apply affine_map<() -> (3)>()
          %72 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
          %73 = vector.insert %72, %70 [%61, %71] : f32 into vector<2x4xf32>
          affine.yield %73 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "gemm_mn", tiled = true}
        %53 = frisk.to_threadTile %arg8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 1>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x16xf32> -> vector<2x4xf32>
        %cst_43 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
        %54 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_43) -> (vector<2x4xf32>) {
          %c0 = arith.constant 0 : index
          %61 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
          %62 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %74 = vector.extract %53[%61, %c0] : f32 from vector<2x4xf32>
            %75 = vector.insert %74, %cst_47 [0, 0] : f32 into vector<1x4xf32>
            %76 = affine.apply affine_map<() -> (1)>()
            %77 = vector.extract %53[%61, %76] : f32 from vector<2x4xf32>
            %78 = vector.insert %77, %75 [0, 1] : f32 into vector<1x4xf32>
            %79 = affine.apply affine_map<() -> (2)>()
            %80 = vector.extract %53[%61, %79] : f32 from vector<2x4xf32>
            %81 = vector.insert %80, %78 [0, 2] : f32 into vector<1x4xf32>
            %82 = affine.apply affine_map<() -> (3)>()
            %83 = vector.extract %53[%61, %82] : f32 from vector<2x4xf32>
            %84 = vector.insert %83, %81 [0, 3] : f32 into vector<1x4xf32>
            %cst_48 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %85 = vector.extract %52[%61, %c0] : f32 from vector<2x4xf32>
            %86 = vector.insert %85, %cst_48 [0, 0] : f32 into vector<1x4xf32>
            %87 = affine.apply affine_map<() -> (1)>()
            %88 = vector.extract %52[%61, %87] : f32 from vector<2x4xf32>
            %89 = vector.insert %88, %86 [0, 1] : f32 into vector<1x4xf32>
            %90 = affine.apply affine_map<() -> (2)>()
            %91 = vector.extract %52[%61, %90] : f32 from vector<2x4xf32>
            %92 = vector.insert %91, %89 [0, 2] : f32 into vector<1x4xf32>
            %93 = affine.apply affine_map<() -> (3)>()
            %94 = vector.extract %52[%61, %93] : f32 from vector<2x4xf32>
            %95 = vector.insert %94, %92 [0, 3] : f32 into vector<1x4xf32>
            %96 = arith.addf %84, %95 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %96 : vector<1x4xf32>
          }
          %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
          %64 = vector.insert %63, %arg10 [%61, %c0] : f32 into vector<2x4xf32>
          %65 = affine.apply affine_map<() -> (1)>()
          %66 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
          %67 = vector.insert %66, %64 [%61, %65] : f32 into vector<2x4xf32>
          %68 = affine.apply affine_map<() -> (2)>()
          %69 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
          %70 = vector.insert %69, %67 [%61, %68] : f32 into vector<2x4xf32>
          %71 = affine.apply affine_map<() -> (3)>()
          %72 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
          %73 = vector.insert %72, %70 [%61, %71] : f32 into vector<2x4xf32>
          affine.yield %73 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
        %cst_44 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
        %55 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_44) -> (vector<2x4xf32>) {
          %c0 = arith.constant 0 : index
          %61 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
          %62 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %74 = vector.extract %50[%61, %c0] : f32 from vector<2x4xf32>
            %75 = vector.insert %74, %cst_47 [0, 0] : f32 into vector<1x4xf32>
            %76 = affine.apply affine_map<() -> (1)>()
            %77 = vector.extract %50[%61, %76] : f32 from vector<2x4xf32>
            %78 = vector.insert %77, %75 [0, 1] : f32 into vector<1x4xf32>
            %79 = affine.apply affine_map<() -> (2)>()
            %80 = vector.extract %50[%61, %79] : f32 from vector<2x4xf32>
            %81 = vector.insert %80, %78 [0, 2] : f32 into vector<1x4xf32>
            %82 = affine.apply affine_map<() -> (3)>()
            %83 = vector.extract %50[%61, %82] : f32 from vector<2x4xf32>
            %84 = vector.insert %83, %81 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
            frisk.fragment_yield %84 : vector<1x4xf32>
          }
          %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
          %64 = vector.insert %63, %arg10 [%61, %c0] : f32 into vector<2x4xf32>
          %65 = affine.apply affine_map<() -> (1)>()
          %66 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
          %67 = vector.insert %66, %64 [%61, %65] : f32 into vector<2x4xf32>
          %68 = affine.apply affine_map<() -> (2)>()
          %69 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
          %70 = vector.insert %69, %67 [%61, %68] : f32 into vector<2x4xf32>
          %71 = affine.apply affine_map<() -> (3)>()
          %72 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
          %73 = vector.insert %72, %70 [%61, %71] : f32 into vector<2x4xf32>
          affine.yield %73 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
        %56 = frisk.from_threadTile %55 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 1>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x4xf32> -> vector<128x16xf32>
        %cst_45 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
        %57 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_45) -> (vector<2x1xf32>) {
          %c0 = arith.constant 0 : index
          %61 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
          %62 = frisk.fragment "compute" -> (vector<1x1xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
            %65 = vector.extract %45[%61, %c0] : f32 from vector<2x1xf32>
            %66 = vector.insert %65, %cst_47 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
            frisk.fragment_yield %66 : vector<1x1xf32>
          }
          %63 = vector.extract %62[0, 0] : f32 from vector<1x1xf32>
          %64 = vector.insert %63, %arg10 [%61, %c0] : f32 into vector<2x1xf32>
          affine.yield %64 : vector<2x1xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
        %58 = frisk.from_threadTile %57 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
        %cst_46 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
        %59 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_46) -> (vector<2x4xf32>) {
          %c0 = arith.constant 0 : index
          %61 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg9)
          %62 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %cst_47 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %74 = vector.extract %54[%61, %c0] : f32 from vector<2x4xf32>
            %75 = vector.insert %74, %cst_47 [0, 0] : f32 into vector<1x4xf32>
            %76 = affine.apply affine_map<() -> (1)>()
            %77 = vector.extract %54[%61, %76] : f32 from vector<2x4xf32>
            %78 = vector.insert %77, %75 [0, 1] : f32 into vector<1x4xf32>
            %79 = affine.apply affine_map<() -> (2)>()
            %80 = vector.extract %54[%61, %79] : f32 from vector<2x4xf32>
            %81 = vector.insert %80, %78 [0, 2] : f32 into vector<1x4xf32>
            %82 = affine.apply affine_map<() -> (3)>()
            %83 = vector.extract %54[%61, %82] : f32 from vector<2x4xf32>
            %84 = vector.insert %83, %81 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
            frisk.fragment_yield %84 : vector<1x4xf32>
          }
          %63 = vector.extract %62[0, 0] : f32 from vector<1x4xf32>
          %64 = vector.insert %63, %arg10 [%61, %c0] : f32 into vector<2x4xf32>
          %65 = affine.apply affine_map<() -> (1)>()
          %66 = vector.extract %62[0, 1] : f32 from vector<1x4xf32>
          %67 = vector.insert %66, %64 [%61, %65] : f32 into vector<2x4xf32>
          %68 = affine.apply affine_map<() -> (2)>()
          %69 = vector.extract %62[0, 2] : f32 from vector<1x4xf32>
          %70 = vector.insert %69, %67 [%61, %68] : f32 into vector<2x4xf32>
          %71 = affine.apply affine_map<() -> (3)>()
          %72 = vector.extract %62[0, 3] : f32 from vector<1x4xf32>
          %73 = vector.insert %72, %70 [%61, %71] : f32 into vector<2x4xf32>
          affine.yield %73 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
        %60 = frisk.from_threadTile %59 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 1>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x4xf32> -> vector<128x16xf32>
        affine.yield %56, %58, %60 : vector<128x16xf32>, vector<128x1xf32>, vector<128x16xf32>
      } {local_yield_transformed = true}
      %3 = frisk.to_threadTile %2#2 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 1>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x16xf32> -> vector<2x4xf32>
      %thread_id_x = gpu.thread_id x
      %4 = arith.index_cast %thread_id_x : index to i32
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %c16_i32 = arith.constant 16 : i32
      %5 = arith.andi %4, %c16_i32 : i32
      %6 = arith.cmpi ne, %5, %c0_i32 : i32
      %c32_i32 = arith.constant 32 : i32
      %7 = arith.andi %4, %c32_i32 : i32
      %8 = arith.cmpi ne, %7, %c0_i32 : i32
      %cst_5 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
      %9 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_5) -> (vector<2x4xf32>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %46 = vector.extract %3[%33, %c0] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %cst_23 [0, 0] : f32 into vector<1x4xf32>
          %48 = affine.apply affine_map<() -> (1)>()
          %49 = vector.extract %3[%33, %48] : f32 from vector<2x4xf32>
          %50 = vector.insert %49, %47 [0, 1] : f32 into vector<1x4xf32>
          %51 = affine.apply affine_map<() -> (2)>()
          %52 = vector.extract %3[%33, %51] : f32 from vector<2x4xf32>
          %53 = vector.insert %52, %50 [0, 2] : f32 into vector<1x4xf32>
          %54 = affine.apply affine_map<() -> (3)>()
          %55 = vector.extract %3[%33, %54] : f32 from vector<2x4xf32>
          %56 = vector.insert %55, %53 [0, 3] : f32 into vector<1x4xf32>
          %57 = vector.extract %56[0, 0] : f32 from vector<1x4xf32>
          %58 = vector.extract %56[0, 1] : f32 from vector<1x4xf32>
          %59 = vector.extract %56[0, 2] : f32 from vector<1x4xf32>
          %60 = vector.extract %56[0, 3] : f32 from vector<1x4xf32>
          %61 = arith.select %6, %57, %58 : f32
          %shuffleResult, %valid = gpu.shuffle xor %61, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %62 = arith.select %6, %shuffleResult, %57 : f32
          %63 = arith.select %6, %58, %shuffleResult : f32
          %64 = arith.select %6, %59, %60 : f32
          %shuffleResult_24, %valid_25 = gpu.shuffle xor %64, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %65 = arith.select %6, %shuffleResult_24, %59 : f32
          %66 = arith.select %6, %60, %shuffleResult_24 : f32
          %67 = arith.select %8, %62, %65 : f32
          %shuffleResult_26, %valid_27 = gpu.shuffle xor %67, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %68 = arith.select %8, %shuffleResult_26, %62 : f32
          %69 = arith.select %8, %65, %shuffleResult_26 : f32
          %70 = arith.select %8, %63, %66 : f32
          %shuffleResult_28, %valid_29 = gpu.shuffle xor %70, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %71 = arith.select %8, %shuffleResult_28, %63 : f32
          %72 = arith.select %8, %66, %shuffleResult_28 : f32
          %73 = vector.insert %68, %56 [0, 0] : f32 into vector<1x4xf32>
          %74 = vector.insert %71, %73 [0, 1] : f32 into vector<1x4xf32>
          %75 = vector.insert %69, %74 [0, 2] : f32 into vector<1x4xf32>
          %76 = vector.insert %72, %75 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %76 : vector<1x4xf32>
        }
        %35 = vector.extract %34[0, 0] : f32 from vector<1x4xf32>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f32 into vector<2x4xf32>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %36 [%33, %37] : f32 into vector<2x4xf32>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f32 from vector<1x4xf32>
        %42 = vector.insert %41, %39 [%33, %40] : f32 into vector<2x4xf32>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f32 from vector<1x4xf32>
        %45 = vector.insert %44, %42 [%33, %43] : f32 into vector<2x4xf32>
        affine.yield %45 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %10 = frisk.to_threadTile %2#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 1>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x16xf32> -> vector<2x4xf32>
      %thread_id_x_6 = gpu.thread_id x
      %11 = arith.index_cast %thread_id_x_6 : index to i32
      %c0_i32_7 = arith.constant 0 : i32
      %c64_i32_8 = arith.constant 64 : i32
      %c16_i32_9 = arith.constant 16 : i32
      %12 = arith.andi %11, %c16_i32_9 : i32
      %13 = arith.cmpi ne, %12, %c0_i32_7 : i32
      %c32_i32_10 = arith.constant 32 : i32
      %14 = arith.andi %11, %c32_i32_10 : i32
      %15 = arith.cmpi ne, %14, %c0_i32_7 : i32
      %cst_11 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
      %16 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_11) -> (vector<2x4xf32>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %46 = vector.extract %10[%33, %c0] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %cst_23 [0, 0] : f32 into vector<1x4xf32>
          %48 = affine.apply affine_map<() -> (1)>()
          %49 = vector.extract %10[%33, %48] : f32 from vector<2x4xf32>
          %50 = vector.insert %49, %47 [0, 1] : f32 into vector<1x4xf32>
          %51 = affine.apply affine_map<() -> (2)>()
          %52 = vector.extract %10[%33, %51] : f32 from vector<2x4xf32>
          %53 = vector.insert %52, %50 [0, 2] : f32 into vector<1x4xf32>
          %54 = affine.apply affine_map<() -> (3)>()
          %55 = vector.extract %10[%33, %54] : f32 from vector<2x4xf32>
          %56 = vector.insert %55, %53 [0, 3] : f32 into vector<1x4xf32>
          %57 = vector.extract %56[0, 0] : f32 from vector<1x4xf32>
          %58 = vector.extract %56[0, 1] : f32 from vector<1x4xf32>
          %59 = vector.extract %56[0, 2] : f32 from vector<1x4xf32>
          %60 = vector.extract %56[0, 3] : f32 from vector<1x4xf32>
          %61 = arith.select %13, %57, %58 : f32
          %shuffleResult, %valid = gpu.shuffle xor %61, %c16_i32_9, %c64_i32_8 {frisk.layout_role = "wmmaC"} : f32
          %62 = arith.select %13, %shuffleResult, %57 : f32
          %63 = arith.select %13, %58, %shuffleResult : f32
          %64 = arith.select %13, %59, %60 : f32
          %shuffleResult_24, %valid_25 = gpu.shuffle xor %64, %c16_i32_9, %c64_i32_8 {frisk.layout_role = "wmmaC"} : f32
          %65 = arith.select %13, %shuffleResult_24, %59 : f32
          %66 = arith.select %13, %60, %shuffleResult_24 : f32
          %67 = arith.select %15, %62, %65 : f32
          %shuffleResult_26, %valid_27 = gpu.shuffle xor %67, %c32_i32_10, %c64_i32_8 {frisk.layout_role = "wmmaC"} : f32
          %68 = arith.select %15, %shuffleResult_26, %62 : f32
          %69 = arith.select %15, %65, %shuffleResult_26 : f32
          %70 = arith.select %15, %63, %66 : f32
          %shuffleResult_28, %valid_29 = gpu.shuffle xor %70, %c32_i32_10, %c64_i32_8 {frisk.layout_role = "wmmaC"} : f32
          %71 = arith.select %15, %shuffleResult_28, %63 : f32
          %72 = arith.select %15, %66, %shuffleResult_28 : f32
          %73 = vector.insert %68, %56 [0, 0] : f32 into vector<1x4xf32>
          %74 = vector.insert %71, %73 [0, 1] : f32 into vector<1x4xf32>
          %75 = vector.insert %69, %74 [0, 2] : f32 into vector<1x4xf32>
          %76 = vector.insert %72, %75 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %76 : vector<1x4xf32>
        }
        %35 = vector.extract %34[0, 0] : f32 from vector<1x4xf32>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f32 into vector<2x4xf32>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %36 [%33, %37] : f32 into vector<2x4xf32>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f32 from vector<1x4xf32>
        %42 = vector.insert %41, %39 [%33, %40] : f32 into vector<2x4xf32>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f32 from vector<1x4xf32>
        %45 = vector.insert %44, %42 [%33, %43] : f32 into vector<2x4xf32>
        affine.yield %45 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %17 = frisk.to_threadTile %2#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 1>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_12 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
      %18 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_12) -> (vector<2x4xf32>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_24 = arith.constant 0 : index
          %46 = vector.extract %17[%33, %c0_24] : f32 from vector<2x1xf32>
          %47 = vector.insert %46, %cst_23 [0, 0] : f32 into vector<1x4xf32>
          %c0_25 = arith.constant 0 : index
          %48 = vector.extract %17[%33, %c0_25] : f32 from vector<2x1xf32>
          %49 = vector.insert %48, %47 [0, 1] : f32 into vector<1x4xf32>
          %c0_26 = arith.constant 0 : index
          %50 = vector.extract %17[%33, %c0_26] : f32 from vector<2x1xf32>
          %51 = vector.insert %50, %49 [0, 2] : f32 into vector<1x4xf32>
          %c0_27 = arith.constant 0 : index
          %52 = vector.extract %17[%33, %c0_27] : f32 from vector<2x1xf32>
          %53 = vector.insert %52, %51 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %53 : vector<1x4xf32>
        }
        %35 = vector.extract %34[0, 0] : f32 from vector<1x4xf32>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f32 into vector<2x4xf32>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %36 [%33, %37] : f32 into vector<2x4xf32>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f32 from vector<1x4xf32>
        %42 = vector.insert %41, %39 [%33, %40] : f32 into vector<2x4xf32>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f32 from vector<1x4xf32>
        %45 = vector.insert %44, %42 [%33, %43] : f32 into vector<2x4xf32>
        affine.yield %45 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>}
      %cst_13 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
      %19 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_13) -> (vector<2x4xf32>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %46 = vector.extract %16[%33, %c0] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %cst_23 [0, 0] : f32 into vector<1x4xf32>
          %48 = affine.apply affine_map<() -> (1)>()
          %49 = vector.extract %16[%33, %48] : f32 from vector<2x4xf32>
          %50 = vector.insert %49, %47 [0, 1] : f32 into vector<1x4xf32>
          %51 = affine.apply affine_map<() -> (2)>()
          %52 = vector.extract %16[%33, %51] : f32 from vector<2x4xf32>
          %53 = vector.insert %52, %50 [0, 2] : f32 into vector<1x4xf32>
          %54 = affine.apply affine_map<() -> (3)>()
          %55 = vector.extract %16[%33, %54] : f32 from vector<2x4xf32>
          %56 = vector.insert %55, %53 [0, 3] : f32 into vector<1x4xf32>
          %cst_24 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %57 = vector.extract %18[%33, %c0] : f32 from vector<2x4xf32>
          %58 = vector.insert %57, %cst_24 [0, 0] : f32 into vector<1x4xf32>
          %59 = affine.apply affine_map<() -> (1)>()
          %60 = vector.extract %18[%33, %59] : f32 from vector<2x4xf32>
          %61 = vector.insert %60, %58 [0, 1] : f32 into vector<1x4xf32>
          %62 = affine.apply affine_map<() -> (2)>()
          %63 = vector.extract %18[%33, %62] : f32 from vector<2x4xf32>
          %64 = vector.insert %63, %61 [0, 2] : f32 into vector<1x4xf32>
          %65 = affine.apply affine_map<() -> (3)>()
          %66 = vector.extract %18[%33, %65] : f32 from vector<2x4xf32>
          %67 = vector.insert %66, %64 [0, 3] : f32 into vector<1x4xf32>
          %68 = arith.divf %56, %67 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %68 : vector<1x4xf32>
        }
        %35 = vector.extract %34[0, 0] : f32 from vector<1x4xf32>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f32 into vector<2x4xf32>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %36 [%33, %37] : f32 into vector<2x4xf32>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f32 from vector<1x4xf32>
        %42 = vector.insert %41, %39 [%33, %40] : f32 into vector<2x4xf32>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f32 from vector<1x4xf32>
        %45 = vector.insert %44, %42 [%33, %43] : f32 into vector<2x4xf32>
        affine.yield %45 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %cst_14 = arith.constant dense<0.000000e+00> : vector<2x4xf16>
      %20 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_14) -> (vector<2x4xf16>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %46 = vector.extract %19[%33, %c0] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %cst_23 [0, 0] : f32 into vector<1x4xf32>
          %48 = affine.apply affine_map<() -> (1)>()
          %49 = vector.extract %19[%33, %48] : f32 from vector<2x4xf32>
          %50 = vector.insert %49, %47 [0, 1] : f32 into vector<1x4xf32>
          %51 = affine.apply affine_map<() -> (2)>()
          %52 = vector.extract %19[%33, %51] : f32 from vector<2x4xf32>
          %53 = vector.insert %52, %50 [0, 2] : f32 into vector<1x4xf32>
          %54 = affine.apply affine_map<() -> (3)>()
          %55 = vector.extract %19[%33, %54] : f32 from vector<2x4xf32>
          %56 = vector.insert %55, %53 [0, 3] : f32 into vector<1x4xf32>
          %57 = arith.truncf %56 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %57 : vector<1x4xf16>
        }
        %35 = vector.extract %34[0, 0] : f16 from vector<1x4xf16>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f16 into vector<2x4xf16>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f16 from vector<1x4xf16>
        %39 = vector.insert %38, %36 [%33, %37] : f16 into vector<2x4xf16>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f16 from vector<1x4xf16>
        %42 = vector.insert %41, %39 [%33, %40] : f16 into vector<2x4xf16>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f16 from vector<1x4xf16>
        %45 = vector.insert %44, %42 [%33, %43] : f16 into vector<2x4xf16>
        affine.yield %45 : vector<2x4xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %21 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %block_id_y * 16384 + %arg4) floordiv 524288) mod 32, (%block_id_z * 131072 + %block_id_x * 4096 + %block_id_y * 128 + %arg4 floordiv 128) mod 4096, %arg4 mod 128], ranges = [128, 16] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x16xf16, 1>
      %cst_15 = arith.constant dense<0.000000e+00> : vector<2x4xf16>
      %22 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_15) -> (vector<2x4xf16>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %46 = vector.extract %20[%33, %c0] : f16 from vector<2x4xf16>
          %47 = vector.insert %46, %cst_23 [0, 0] : f16 into vector<1x4xf16>
          %48 = affine.apply affine_map<() -> (1)>()
          %49 = vector.extract %20[%33, %48] : f16 from vector<2x4xf16>
          %50 = vector.insert %49, %47 [0, 1] : f16 into vector<1x4xf16>
          %51 = affine.apply affine_map<() -> (2)>()
          %52 = vector.extract %20[%33, %51] : f16 from vector<2x4xf16>
          %53 = vector.insert %52, %50 [0, 2] : f16 into vector<1x4xf16>
          %54 = affine.apply affine_map<() -> (3)>()
          %55 = vector.extract %20[%33, %54] : f16 from vector<2x4xf16>
          %56 = vector.insert %55, %53 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
          frisk.fragment_yield %56 : vector<1x4xf16>
        }
        %35 = vector.extract %34[0, 0] : f16 from vector<1x4xf16>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f16 into vector<2x4xf16>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f16 from vector<1x4xf16>
        %39 = vector.insert %38, %36 [%33, %37] : f16 into vector<2x4xf16>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f16 from vector<1x4xf16>
        %42 = vector.insert %41, %39 [%33, %40] : f16 into vector<2x4xf16>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f16 from vector<1x4xf16>
        %45 = vector.insert %44, %42 [%33, %43] : f16 into vector<2x4xf16>
        affine.yield %45 : vector<2x4xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %thread_id_x_16 = gpu.thread_id x
      %23 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_16)
      %c4 = arith.constant 4 : index
      %24 = arith.cmpi ult, %23, %c4 : index
      scf.if %24 {
        affine.for %arg5 = 0 to 2 {
          %c0 = arith.constant 0 : index
          %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
          frisk.fragment "store" {
            %34 = vector.extract %22[%33, %c0] : f16 from vector<2x4xf16>
            %c0_23 = arith.constant 0 : index
            %c0_24 = arith.constant 0 : index
            %35 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%c0)
            %c0_25 = arith.constant 0 : index
            %36 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%c0)
            %37 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%36)
            %c0_26 = arith.constant 0 : index
            %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_24, %37)
            %39 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_16, %33, %35, %c0_23, %c0_25, %c0_26, %38)
            %40 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_16, %33, %35, %c0_23, %c0_25, %c0_26, %38)
            affine.store %34, %21[%39, %40] {tiled = true} : memref<128x16xf16, 1>
            %41 = affine.apply affine_map<() -> (1)>()
            %42 = vector.extract %22[%33, %41] : f16 from vector<2x4xf16>
            %c0_27 = arith.constant 0 : index
            %c0_28 = arith.constant 0 : index
            %43 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%41)
            %c0_29 = arith.constant 0 : index
            %44 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%41)
            %45 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%44)
            %c0_30 = arith.constant 0 : index
            %46 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_28, %45)
            %47 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_16, %33, %43, %c0_27, %c0_29, %c0_30, %46)
            %48 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_16, %33, %43, %c0_27, %c0_29, %c0_30, %46)
            affine.store %42, %21[%47, %48] {tiled = true} : memref<128x16xf16, 1>
            %49 = affine.apply affine_map<() -> (2)>()
            %50 = vector.extract %22[%33, %49] : f16 from vector<2x4xf16>
            %c0_31 = arith.constant 0 : index
            %c0_32 = arith.constant 0 : index
            %51 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%49)
            %c0_33 = arith.constant 0 : index
            %52 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%49)
            %53 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%52)
            %c0_34 = arith.constant 0 : index
            %54 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_32, %53)
            %55 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_16, %33, %51, %c0_31, %c0_33, %c0_34, %54)
            %56 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_16, %33, %51, %c0_31, %c0_33, %c0_34, %54)
            affine.store %50, %21[%55, %56] {tiled = true} : memref<128x16xf16, 1>
            %57 = affine.apply affine_map<() -> (3)>()
            %58 = vector.extract %22[%33, %57] : f16 from vector<2x4xf16>
            %c0_35 = arith.constant 0 : index
            %c0_36 = arith.constant 0 : index
            %59 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%57)
            %c0_37 = arith.constant 0 : index
            %60 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%57)
            %61 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%60)
            %c0_38 = arith.constant 0 : index
            %62 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_36, %61)
            %63 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_16, %33, %59, %c0_35, %c0_37, %c0_38, %62)
            %64 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_16, %33, %59, %c0_35, %c0_37, %c0_38, %62)
            affine.store %58, %21[%63, %64] {tiled = true} : memref<128x16xf16, 1>
            frisk.fragment_yield
          }
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "store", tiled = true}
      }
      %25 = frisk.to_threadTile %2#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 1>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_17 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
      %26 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_17) -> (vector<2x4xf32>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0_24 = arith.constant 0 : index
          %46 = vector.extract %25[%33, %c0_24] : f32 from vector<2x1xf32>
          %47 = vector.insert %46, %cst_23 [0, 0] : f32 into vector<1x4xf32>
          %c0_25 = arith.constant 0 : index
          %48 = vector.extract %25[%33, %c0_25] : f32 from vector<2x1xf32>
          %49 = vector.insert %48, %47 [0, 1] : f32 into vector<1x4xf32>
          %c0_26 = arith.constant 0 : index
          %50 = vector.extract %25[%33, %c0_26] : f32 from vector<2x1xf32>
          %51 = vector.insert %50, %49 [0, 2] : f32 into vector<1x4xf32>
          %c0_27 = arith.constant 0 : index
          %52 = vector.extract %25[%33, %c0_27] : f32 from vector<2x1xf32>
          %53 = vector.insert %52, %51 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %53 : vector<1x4xf32>
        }
        %35 = vector.extract %34[0, 0] : f32 from vector<1x4xf32>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f32 into vector<2x4xf32>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %36 [%33, %37] : f32 into vector<2x4xf32>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f32 from vector<1x4xf32>
        %42 = vector.insert %41, %39 [%33, %40] : f32 into vector<2x4xf32>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f32 from vector<1x4xf32>
        %45 = vector.insert %44, %42 [%33, %43] : f32 into vector<2x4xf32>
        affine.yield %45 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>}
      %cst_18 = arith.constant dense<0.000000e+00> : vector<2x4xf32>
      %27 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_18) -> (vector<2x4xf32>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %46 = vector.extract %9[%33, %c0] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %cst_23 [0, 0] : f32 into vector<1x4xf32>
          %48 = affine.apply affine_map<() -> (1)>()
          %49 = vector.extract %9[%33, %48] : f32 from vector<2x4xf32>
          %50 = vector.insert %49, %47 [0, 1] : f32 into vector<1x4xf32>
          %51 = affine.apply affine_map<() -> (2)>()
          %52 = vector.extract %9[%33, %51] : f32 from vector<2x4xf32>
          %53 = vector.insert %52, %50 [0, 2] : f32 into vector<1x4xf32>
          %54 = affine.apply affine_map<() -> (3)>()
          %55 = vector.extract %9[%33, %54] : f32 from vector<2x4xf32>
          %56 = vector.insert %55, %53 [0, 3] : f32 into vector<1x4xf32>
          %cst_24 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %57 = vector.extract %26[%33, %c0] : f32 from vector<2x4xf32>
          %58 = vector.insert %57, %cst_24 [0, 0] : f32 into vector<1x4xf32>
          %59 = affine.apply affine_map<() -> (1)>()
          %60 = vector.extract %26[%33, %59] : f32 from vector<2x4xf32>
          %61 = vector.insert %60, %58 [0, 1] : f32 into vector<1x4xf32>
          %62 = affine.apply affine_map<() -> (2)>()
          %63 = vector.extract %26[%33, %62] : f32 from vector<2x4xf32>
          %64 = vector.insert %63, %61 [0, 2] : f32 into vector<1x4xf32>
          %65 = affine.apply affine_map<() -> (3)>()
          %66 = vector.extract %26[%33, %65] : f32 from vector<2x4xf32>
          %67 = vector.insert %66, %64 [0, 3] : f32 into vector<1x4xf32>
          %68 = arith.divf %56, %67 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %68 : vector<1x4xf32>
        }
        %35 = vector.extract %34[0, 0] : f32 from vector<1x4xf32>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f32 into vector<2x4xf32>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %36 [%33, %37] : f32 into vector<2x4xf32>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f32 from vector<1x4xf32>
        %42 = vector.insert %41, %39 [%33, %40] : f32 into vector<2x4xf32>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f32 from vector<1x4xf32>
        %45 = vector.insert %44, %42 [%33, %43] : f32 into vector<2x4xf32>
        affine.yield %45 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %cst_19 = arith.constant dense<0.000000e+00> : vector<2x4xf16>
      %28 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_19) -> (vector<2x4xf16>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %46 = vector.extract %27[%33, %c0] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %cst_23 [0, 0] : f32 into vector<1x4xf32>
          %48 = affine.apply affine_map<() -> (1)>()
          %49 = vector.extract %27[%33, %48] : f32 from vector<2x4xf32>
          %50 = vector.insert %49, %47 [0, 1] : f32 into vector<1x4xf32>
          %51 = affine.apply affine_map<() -> (2)>()
          %52 = vector.extract %27[%33, %51] : f32 from vector<2x4xf32>
          %53 = vector.insert %52, %50 [0, 2] : f32 into vector<1x4xf32>
          %54 = affine.apply affine_map<() -> (3)>()
          %55 = vector.extract %27[%33, %54] : f32 from vector<2x4xf32>
          %56 = vector.insert %55, %53 [0, 3] : f32 into vector<1x4xf32>
          %57 = arith.truncf %56 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %57 : vector<1x4xf16>
        }
        %35 = vector.extract %34[0, 0] : f16 from vector<1x4xf16>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f16 into vector<2x4xf16>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f16 from vector<1x4xf16>
        %39 = vector.insert %38, %36 [%33, %37] : f16 into vector<2x4xf16>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f16 from vector<1x4xf16>
        %42 = vector.insert %41, %39 [%33, %40] : f16 into vector<2x4xf16>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f16 from vector<1x4xf16>
        %45 = vector.insert %44, %42 [%33, %43] : f16 into vector<2x4xf16>
        affine.yield %45 : vector<2x4xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %29 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %block_id_y * 16384 + %arg4 + 16) floordiv 524288) mod 32, (%block_id_z * 131072 + %block_id_x * 4096 + %block_id_y * 128 + (%arg4 + 16) floordiv 128) mod 4096, (%arg4 + 16) mod 128], ranges = [128, 16] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x16xf16, 1>
      %cst_20 = arith.constant dense<0.000000e+00> : vector<2x4xf16>
      %30 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_20) -> (vector<2x4xf16>) {
        %c0 = arith.constant 0 : index
        %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
        %34 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %cst_23 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %46 = vector.extract %28[%33, %c0] : f16 from vector<2x4xf16>
          %47 = vector.insert %46, %cst_23 [0, 0] : f16 into vector<1x4xf16>
          %48 = affine.apply affine_map<() -> (1)>()
          %49 = vector.extract %28[%33, %48] : f16 from vector<2x4xf16>
          %50 = vector.insert %49, %47 [0, 1] : f16 into vector<1x4xf16>
          %51 = affine.apply affine_map<() -> (2)>()
          %52 = vector.extract %28[%33, %51] : f16 from vector<2x4xf16>
          %53 = vector.insert %52, %50 [0, 2] : f16 into vector<1x4xf16>
          %54 = affine.apply affine_map<() -> (3)>()
          %55 = vector.extract %28[%33, %54] : f16 from vector<2x4xf16>
          %56 = vector.insert %55, %53 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
          frisk.fragment_yield %56 : vector<1x4xf16>
        }
        %35 = vector.extract %34[0, 0] : f16 from vector<1x4xf16>
        %36 = vector.insert %35, %arg6 [%33, %c0] : f16 into vector<2x4xf16>
        %37 = affine.apply affine_map<() -> (1)>()
        %38 = vector.extract %34[0, 1] : f16 from vector<1x4xf16>
        %39 = vector.insert %38, %36 [%33, %37] : f16 into vector<2x4xf16>
        %40 = affine.apply affine_map<() -> (2)>()
        %41 = vector.extract %34[0, 2] : f16 from vector<1x4xf16>
        %42 = vector.insert %41, %39 [%33, %40] : f16 into vector<2x4xf16>
        %43 = affine.apply affine_map<() -> (3)>()
        %44 = vector.extract %34[0, 3] : f16 from vector<1x4xf16>
        %45 = vector.insert %44, %42 [%33, %43] : f16 into vector<2x4xf16>
        affine.yield %45 : vector<2x4xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %thread_id_x_21 = gpu.thread_id x
      %31 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%thread_id_x_21)
      %c4_22 = arith.constant 4 : index
      %32 = arith.cmpi ult, %31, %c4_22 : index
      scf.if %32 {
        affine.for %arg5 = 0 to 2 {
          %c0 = arith.constant 0 : index
          %33 = affine.apply affine_map<(d0) -> (d0 mod 2)>(%arg5)
          frisk.fragment "store" {
            %34 = vector.extract %30[%33, %c0] : f16 from vector<2x4xf16>
            %c0_23 = arith.constant 0 : index
            %c0_24 = arith.constant 0 : index
            %35 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%c0)
            %c0_25 = arith.constant 0 : index
            %36 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%c0)
            %37 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%36)
            %c0_26 = arith.constant 0 : index
            %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_24, %37)
            %39 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_21, %33, %35, %c0_23, %c0_25, %c0_26, %38)
            %40 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_21, %33, %35, %c0_23, %c0_25, %c0_26, %38)
            affine.store %34, %29[%39, %40] {tiled = true} : memref<128x16xf16, 1>
            %41 = affine.apply affine_map<() -> (1)>()
            %42 = vector.extract %30[%33, %41] : f16 from vector<2x4xf16>
            %c0_27 = arith.constant 0 : index
            %c0_28 = arith.constant 0 : index
            %43 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%41)
            %c0_29 = arith.constant 0 : index
            %44 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%41)
            %45 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%44)
            %c0_30 = arith.constant 0 : index
            %46 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_28, %45)
            %47 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_21, %33, %43, %c0_27, %c0_29, %c0_30, %46)
            %48 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_21, %33, %43, %c0_27, %c0_29, %c0_30, %46)
            affine.store %42, %29[%47, %48] {tiled = true} : memref<128x16xf16, 1>
            %49 = affine.apply affine_map<() -> (2)>()
            %50 = vector.extract %30[%33, %49] : f16 from vector<2x4xf16>
            %c0_31 = arith.constant 0 : index
            %c0_32 = arith.constant 0 : index
            %51 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%49)
            %c0_33 = arith.constant 0 : index
            %52 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%49)
            %53 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%52)
            %c0_34 = arith.constant 0 : index
            %54 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_32, %53)
            %55 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_21, %33, %51, %c0_31, %c0_33, %c0_34, %54)
            %56 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_21, %33, %51, %c0_31, %c0_33, %c0_34, %54)
            affine.store %50, %29[%55, %56] {tiled = true} : memref<128x16xf16, 1>
            %57 = affine.apply affine_map<() -> (3)>()
            %58 = vector.extract %30[%33, %57] : f16 from vector<2x4xf16>
            %c0_35 = arith.constant 0 : index
            %c0_36 = arith.constant 0 : index
            %59 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%57)
            %c0_37 = arith.constant 0 : index
            %60 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%57)
            %61 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%60)
            %c0_38 = arith.constant 0 : index
            %62 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c0_36, %61)
            %63 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>(%thread_id_x_21, %33, %59, %c0_35, %c0_37, %c0_38, %62)
            %64 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>(%thread_id_x_21, %33, %59, %c0_35, %c0_37, %c0_38, %62)
            affine.store %58, %29[%63, %64] {tiled = true} : memref<128x16xf16, 1>
            frisk.fragment_yield
          }
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "store", tiled = true}
      }
    } {local_yield_transformed = true}
    return
  }
}