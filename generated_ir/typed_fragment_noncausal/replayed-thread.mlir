#map = affine_map<(d0) -> (d0 mod 2)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + (d1 mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + (d1 floordiv 4) mod 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 64) * 64 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 16) mod 4) * 4 + ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6) floordiv 4) mod 4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>
#map5 = affine_map<(d0) -> ((d0 floordiv 2) mod 2)>
#map6 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8)>
#map7 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 1)>
#map8 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 2)>
#map9 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 2) * 8 + 3)>
#map10 = affine_map<(d0) -> ((d0 floordiv 8) mod 2)>
#map11 = affine_map<(d0) -> (d0 * 4)>
#map12 = affine_map<(d0) -> (d0 * 4 + 1)>
#map13 = affine_map<(d0) -> (d0 * 4 + 2)>
#map14 = affine_map<(d0) -> (d0 * 4 + 3)>
#map15 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4, ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 64) * 64 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) mod 4) * 16 + (((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 16) mod 4) * 4 + ((d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4) floordiv 4) mod 4)>
#map16 = affine_map<(d0) -> (d0 mod 8)>
#map17 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>
#map18 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>
#map19 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>
#map20 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>
#map21 = affine_map<(d0)[s0] -> (d0 * 4 - (d0 floordiv 8) * 32 + ((d0 * 16 - (d0 floordiv 8) * 128 + (s0 mod 64) floordiv 4) floordiv 64) * 64 + (s0 mod 64) floordiv 4 + (((s0 mod 64) floordiv 4) floordiv 4) * 12 + ((d0 mod 8 + ((s0 mod 64) floordiv 4) floordiv 16) floordiv 4) * -16 - ((((s0 mod 64) floordiv 4) floordiv 4) floordiv 4) * 64 + (((s0 mod 64) floordiv 4) floordiv 16) * 4)>
#map22 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
#map23 = affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>
#map24 = affine_map<()[s0] -> (((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>
#map25 = affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 - ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 1)>
#map26 = affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 - ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 1)>
#map27 = affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 - ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 2)>
#map28 = affine_map<(d0)[s0] -> (((d0 * 8 - (d0 floordiv 8) * 64 - ((d0 * 8 - (d0 floordiv 8) * 64 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 2)>
#map29 = affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 - ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + (((s0 mod 16) mod 32) floordiv 16) * 4 + 3)>
#map30 = affine_map<(d0)[s0] -> (((d0 * 16 - (d0 floordiv 8) * 128 - ((d0 * 16 - (d0 floordiv 8) * 128 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 8 + ((s0 mod 16) floordiv 16 - ((s0 mod 16 + 16) floordiv 32) * 2 + 1) * 4 + 3)>
#map31 = affine_map<() -> (1)>
#map32 = affine_map<(d0, d1) -> (d0, (d1 floordiv 64) * 64 + ((d1 floordiv 4) mod 4) * 16 + ((d1 floordiv 16) mod 4) * 4 + d1 mod 4)>
#map33 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4, ((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 64) * 64 + ((d2 * 4 + (((d0 floordiv 64) mod 4) floordiv 4) * 4 + d4 * 4 + d5 * 4 + (d0 mod 64) floordiv 16 + (d6 mod 4) floordiv 4) mod 4) * 16 + (((d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4) floordiv 16) mod 4) * 4 + d6 mod 4)>
#map34 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>
#map35 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16, d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 4 + (d0 mod 64) floordiv 16 + d6)>
#map36 = affine_map<(d0) -> (d0 floordiv 64)>
#map37 = affine_map<(d0) -> (d0 floordiv 4)>
#map38 = affine_map<(d0) -> (d0 mod 4)>
#map39 = affine_map<(d0, d1) -> (d0 + d1)>
#map40 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>
#map41 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>
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
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %c32 = arith.constant 32 : index
    %0 = arith.muli %block_id_y, %c32 : index
    %1 = arith.addi %block_id_x, %0 : index
    %c1024 = arith.constant 1024 : index
    %c32_0 = arith.constant 32 : index
    %2 = arith.divui %1, %c1024 : index
    %3 = arith.muli %2, %c32_0 : index
    %c32_1 = arith.constant 32 : index
    %4 = arith.remui %1, %c1024 : index
    %5 = arith.remui %4, %c32_1 : index
    %6 = arith.addi %3, %5 : index
    %7 = arith.divui %4, %c32_1 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %8 = arith.truncf %cst : f32 to f16
    %cst_3 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %9 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_3) -> (vector<2x1xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %cst_53 = arith.constant {tiled = true} dense<0.000000e+00> : vector<1x1xf32>
        frisk.fragment_yield %cst_53 : vector<1x1xf32>
      }
      %73 = affine.apply #map(%arg4)
      %74 = affine.apply #map1()
      %75 = vector.extract %72[0, 0] : f32 from vector<1x1xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x1xf32>
      affine.yield %76 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %10 = vector.broadcast %cst_2 : f32 to vector<2x1xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %11 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_4) -> (vector<2x1xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %77 = affine.apply #map(%arg4)
        %78 = affine.apply #map1()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %79 = vector.extract %9[%77, %78] : f32 from vector<2x1xf32>
        %80 = vector.insert %79, %cst_53 [0, 0] : f32 into vector<1x1xf32>
        %81 = affine.apply #map(%arg4)
        %82 = affine.apply #map1()
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %83 = vector.extract %10[%81, %82] : f32 from vector<2x1xf32>
        %84 = vector.insert %83, %cst_54 [0, 0] : f32 into vector<1x1xf32>
        %85 = arith.addf %80, %84 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
        frisk.fragment_yield %85 : vector<1x1xf32>
      }
      %73 = affine.apply #map(%arg4)
      %74 = affine.apply #map1()
      %75 = vector.extract %72[0, 0] : f32 from vector<1x1xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x1xf32>
      affine.yield %76 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %12 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_7) -> (vector<2x1xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x1xf32>) {
        %77 = affine.apply #map(%arg4)
        %78 = affine.apply #map1()
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
        %79 = vector.extract %11[%77, %78] : f32 from vector<2x1xf32>
        %80 = vector.insert %79, %cst_53 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
        frisk.fragment_yield %80 : vector<1x1xf32>
      }
      %73 = affine.apply #map(%arg4)
      %74 = affine.apply #map1()
      %75 = vector.extract %72[0, 0] : f32 from vector<1x1xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x1xf32>
      affine.yield %76 : vector<2x1xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
    %13 = frisk.from_threadTile %12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %14:6 = affine.for %arg4 = 0 to 4096 step 128 iter_args(%arg5 = %cst_5, %arg6 = %cst_6, %arg7 = %13, %arg8 = %cst_8, %arg9 = %cst_9, %arg10 = %cst_10) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %cst_53 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %72 = affine.for %arg11 = 0 to 128 step 32 iter_args(%arg12 = %cst_53) -> (vector<128x128xf32>) {
        %alloc_106 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x32xf16, 3>
        %alloc_107 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_swizzle_dst = 6 : i64, frisk.shared_swizzle_src = 8 : i64, tiled = true} : memref<32x128xf16, 3>
        %137 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + %arg11) floordiv 524288) mod 32, (%block_id_z * 131072 + %7 * 4096 + %6 * 128 + %arg11 floordiv 128) mod 4096, %arg11 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        frisk.async_copy_gm_to_shm %137 to %alloc_106 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        %138 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg11 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %7 * 128 + %arg11 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.wait_buffer_ready %alloc_106 : memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %138 to %alloc_107 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        %139 = frisk.to_threadTile %alloc_106 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map4, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x32xf16, 3> -> vector<2x8xf16>
        %140 = vector.broadcast %8 : f16 to vector<2x8xf16>
        %cst_108 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
        %141 = affine.for %arg13 = 0 to 4 iter_args(%arg14 = %cst_108) -> (vector<2x8xf16>) {
          %147 = frisk.fragment "compute" -> (vector<1x4xf16>) {
            %161 = affine.apply #map5(%arg13)
            %162 = affine.apply #map6(%arg13)
            %cst_113 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %163 = vector.extract %139[%161, %162] : f16 from vector<2x8xf16>
            %164 = vector.insert %163, %cst_113 [0, 0] : f16 into vector<1x4xf16>
            %165 = affine.apply #map7(%arg13)
            %166 = vector.extract %139[%161, %165] : f16 from vector<2x8xf16>
            %167 = vector.insert %166, %164 [0, 1] : f16 into vector<1x4xf16>
            %168 = affine.apply #map8(%arg13)
            %169 = vector.extract %139[%161, %168] : f16 from vector<2x8xf16>
            %170 = vector.insert %169, %167 [0, 2] : f16 into vector<1x4xf16>
            %171 = affine.apply #map9(%arg13)
            %172 = vector.extract %139[%161, %171] : f16 from vector<2x8xf16>
            %173 = vector.insert %172, %170 [0, 3] : f16 into vector<1x4xf16>
            %174 = affine.apply #map5(%arg13)
            %175 = affine.apply #map6(%arg13)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %176 = vector.extract %140[%174, %175] : f16 from vector<2x8xf16>
            %177 = vector.insert %176, %cst_114 [0, 0] : f16 into vector<1x4xf16>
            %178 = affine.apply #map7(%arg13)
            %179 = vector.extract %140[%174, %178] : f16 from vector<2x8xf16>
            %180 = vector.insert %179, %177 [0, 1] : f16 into vector<1x4xf16>
            %181 = affine.apply #map8(%arg13)
            %182 = vector.extract %140[%174, %181] : f16 from vector<2x8xf16>
            %183 = vector.insert %182, %180 [0, 2] : f16 into vector<1x4xf16>
            %184 = affine.apply #map9(%arg13)
            %185 = vector.extract %140[%174, %184] : f16 from vector<2x8xf16>
            %186 = vector.insert %185, %183 [0, 3] : f16 into vector<1x4xf16>
            %187 = arith.mulf %173, %186 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
            frisk.fragment_yield %187 : vector<1x4xf16>
          }
          %148 = affine.apply #map5(%arg13)
          %149 = affine.apply #map6(%arg13)
          %150 = vector.extract %147[0, 0] : f16 from vector<1x4xf16>
          %151 = vector.insert %150, %arg14 [%148, %149] : f16 into vector<2x8xf16>
          %152 = affine.apply #map7(%arg13)
          %153 = vector.extract %147[0, 1] : f16 from vector<1x4xf16>
          %154 = vector.insert %153, %151 [%148, %152] : f16 into vector<2x8xf16>
          %155 = affine.apply #map8(%arg13)
          %156 = vector.extract %147[0, 2] : f16 from vector<1x4xf16>
          %157 = vector.insert %156, %154 [%148, %155] : f16 into vector<2x8xf16>
          %158 = affine.apply #map9(%arg13)
          %159 = vector.extract %147[0, 3] : f16 from vector<1x4xf16>
          %160 = vector.insert %159, %157 [%148, %158] : f16 into vector<2x8xf16>
          affine.yield %160 : vector<2x8xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
        frisk.wait_buffer_ready %alloc_107 : memref<32x128xf16, 3>
        %cst_109 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %cst_110 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %142 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_109) -> (vector<2x32xf32>) {
          %147 = affine.for %arg15 = 0 to 2 iter_args(%arg16 = %cst_110) -> (vector<1x4xf32>) {
            %161 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %164 = affine.apply #map10(%arg13)
              %165 = affine.apply #map11(%arg15)
              %cst_113 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
              %166 = vector.extract %141[%164, %165] : f16 from vector<2x8xf16>
              %167 = vector.insert %166, %cst_113 [0, 0] : f16 into vector<1x4xf16>
              %168 = affine.apply #map12(%arg15)
              %169 = vector.extract %141[%164, %168] : f16 from vector<2x8xf16>
              %170 = vector.insert %169, %167 [0, 1] : f16 into vector<1x4xf16>
              %171 = affine.apply #map13(%arg15)
              %172 = vector.extract %141[%164, %171] : f16 from vector<2x8xf16>
              %173 = vector.insert %172, %170 [0, 2] : f16 into vector<1x4xf16>
              %174 = affine.apply #map14(%arg15)
              %175 = vector.extract %141[%164, %174] : f16 from vector<2x8xf16>
              %176 = vector.insert %175, %173 [0, 3] : f16 into vector<1x4xf16>
              frisk.fragment_yield %176 : vector<1x4xf16>
            }
            %162 = frisk.fragment "load" -> (vector<4x1xf16>) {
              %164 = frisk.to_threadTile %alloc_107 {block_layout = array<i64: 1, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 4, 1>, tile_layout = #map15, tiled = true, warp_id_divisor = 4 : i64, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 4, 16>, warp_layout_order = array<i64: 1, 0>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<32x128xf16, 3> -> vector<8x8xf16>
              %165 = affine.apply #map11(%arg15)
              %166 = affine.apply #map16(%arg13)
              %cst_113 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
              %167 = vector.extract %164[%165, %166] : f16 from vector<8x8xf16>
              %168 = vector.insert %167, %cst_113 [0, 0] : f16 into vector<4x1xf16>
              %169 = affine.apply #map12(%arg15)
              %170 = vector.extract %164[%169, %166] : f16 from vector<8x8xf16>
              %171 = vector.insert %170, %168 [1, 0] : f16 into vector<4x1xf16>
              %172 = affine.apply #map13(%arg15)
              %173 = vector.extract %164[%172, %166] : f16 from vector<8x8xf16>
              %174 = vector.insert %173, %171 [2, 0] : f16 into vector<4x1xf16>
              %175 = affine.apply #map14(%arg15)
              %176 = vector.extract %164[%175, %166] : f16 from vector<8x8xf16>
              %177 = vector.insert %176, %174 [3, 0] : f16 into vector<4x1xf16>
              frisk.fragment_yield %177 : vector<4x1xf16>
            }
            %163 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %164 = frisk.warp_mma_rr %161, %162, %arg16 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %164 : vector<1x4xf32>
            }
            affine.yield %163 : vector<1x4xf32>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %148 = affine.apply #map10(%arg13)
          %149 = affine.apply #map17(%arg13)
          %150 = vector.extract %147[0, 0] : f32 from vector<1x4xf32>
          %151 = vector.insert %150, %arg14 [%148, %149] : f32 into vector<2x32xf32>
          %152 = affine.apply #map18(%arg13)
          %153 = vector.extract %147[0, 1] : f32 from vector<1x4xf32>
          %154 = vector.insert %153, %151 [%148, %152] : f32 into vector<2x32xf32>
          %155 = affine.apply #map19(%arg13)
          %156 = vector.extract %147[0, 2] : f32 from vector<1x4xf32>
          %157 = vector.insert %156, %154 [%148, %155] : f32 into vector<2x32xf32>
          %158 = affine.apply #map20(%arg13)
          %159 = vector.extract %147[0, 3] : f32 from vector<1x4xf32>
          %160 = vector.insert %159, %157 [%148, %158] : f32 into vector<2x32xf32>
          affine.yield %160 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
        %143 = frisk.to_threadTile %arg12 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
        %cst_111 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %144 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_111) -> (vector<2x32xf32>) {
          %147 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %161 = affine.apply #map10(%arg13)
            %162 = affine.apply #map17(%arg13)
            %cst_113 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %163 = vector.extract %143[%161, %162] : f32 from vector<2x32xf32>
            %164 = vector.insert %163, %cst_113 [0, 0] : f32 into vector<1x4xf32>
            %165 = affine.apply #map18(%arg13)
            %166 = vector.extract %143[%161, %165] : f32 from vector<2x32xf32>
            %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
            %168 = affine.apply #map19(%arg13)
            %169 = vector.extract %143[%161, %168] : f32 from vector<2x32xf32>
            %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
            %171 = affine.apply #map20(%arg13)
            %172 = vector.extract %143[%161, %171] : f32 from vector<2x32xf32>
            %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
            %174 = affine.apply #map10(%arg13)
            %175 = affine.apply #map17(%arg13)
            %cst_114 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %176 = vector.extract %142[%174, %175] : f32 from vector<2x32xf32>
            %177 = vector.insert %176, %cst_114 [0, 0] : f32 into vector<1x4xf32>
            %178 = affine.apply #map18(%arg13)
            %179 = vector.extract %142[%174, %178] : f32 from vector<2x32xf32>
            %180 = vector.insert %179, %177 [0, 1] : f32 into vector<1x4xf32>
            %181 = affine.apply #map19(%arg13)
            %182 = vector.extract %142[%174, %181] : f32 from vector<2x32xf32>
            %183 = vector.insert %182, %180 [0, 2] : f32 into vector<1x4xf32>
            %184 = affine.apply #map20(%arg13)
            %185 = vector.extract %142[%174, %184] : f32 from vector<2x32xf32>
            %186 = vector.insert %185, %183 [0, 3] : f32 into vector<1x4xf32>
            %187 = arith.addf %173, %186 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %187 : vector<1x4xf32>
          }
          %148 = affine.apply #map10(%arg13)
          %149 = affine.apply #map17(%arg13)
          %150 = vector.extract %147[0, 0] : f32 from vector<1x4xf32>
          %151 = vector.insert %150, %arg14 [%148, %149] : f32 into vector<2x32xf32>
          %152 = affine.apply #map18(%arg13)
          %153 = vector.extract %147[0, 1] : f32 from vector<1x4xf32>
          %154 = vector.insert %153, %151 [%148, %152] : f32 into vector<2x32xf32>
          %155 = affine.apply #map19(%arg13)
          %156 = vector.extract %147[0, 2] : f32 from vector<1x4xf32>
          %157 = vector.insert %156, %154 [%148, %155] : f32 into vector<2x32xf32>
          %158 = affine.apply #map20(%arg13)
          %159 = vector.extract %147[0, 3] : f32 from vector<1x4xf32>
          %160 = vector.insert %159, %157 [%148, %158] : f32 into vector<2x32xf32>
          affine.yield %160 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %cst_112 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
        %145 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst_112) -> (vector<2x32xf32>) {
          %147 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %161 = affine.apply #map10(%arg13)
            %162 = affine.apply #map17(%arg13)
            %cst_113 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
            %163 = vector.extract %144[%161, %162] : f32 from vector<2x32xf32>
            %164 = vector.insert %163, %cst_113 [0, 0] : f32 into vector<1x4xf32>
            %165 = affine.apply #map18(%arg13)
            %166 = vector.extract %144[%161, %165] : f32 from vector<2x32xf32>
            %167 = vector.insert %166, %164 [0, 1] : f32 into vector<1x4xf32>
            %168 = affine.apply #map19(%arg13)
            %169 = vector.extract %144[%161, %168] : f32 from vector<2x32xf32>
            %170 = vector.insert %169, %167 [0, 2] : f32 into vector<1x4xf32>
            %171 = affine.apply #map20(%arg13)
            %172 = vector.extract %144[%161, %171] : f32 from vector<2x32xf32>
            %173 = vector.insert %172, %170 [0, 3] : f32 into vector<1x4xf32>
            frisk.fragment_yield %173 : vector<1x4xf32>
          }
          %148 = affine.apply #map10(%arg13)
          %149 = affine.apply #map17(%arg13)
          %150 = vector.extract %147[0, 0] : f32 from vector<1x4xf32>
          %151 = vector.insert %150, %arg14 [%148, %149] : f32 into vector<2x32xf32>
          %152 = affine.apply #map18(%arg13)
          %153 = vector.extract %147[0, 1] : f32 from vector<1x4xf32>
          %154 = vector.insert %153, %151 [%148, %152] : f32 into vector<2x32xf32>
          %155 = affine.apply #map19(%arg13)
          %156 = vector.extract %147[0, 2] : f32 from vector<1x4xf32>
          %157 = vector.insert %156, %154 [%148, %155] : f32 into vector<2x32xf32>
          %158 = affine.apply #map20(%arg13)
          %159 = vector.extract %147[0, 3] : f32 from vector<1x4xf32>
          %160 = vector.insert %159, %157 [%148, %158] : f32 into vector<2x32xf32>
          affine.yield %160 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %146 = frisk.from_threadTile %145 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf32> -> vector<128x128xf32>
        affine.yield %146 : vector<128x128xf32>
      } {frisk.simplify_loop, local_yield_transformed = true}
      %73 = frisk.to_threadTile %72 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_54 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %74 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_54) -> (vector<2x32xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map10(%arg11)
          %152 = affine.apply #map17(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %73[%151, %152] : f32 from vector<2x32xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map18(%arg11)
          %156 = vector.extract %73[%151, %155] : f32 from vector<2x32xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map19(%arg11)
          %159 = vector.extract %73[%151, %158] : f32 from vector<2x32xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map20(%arg11)
          %162 = vector.extract %73[%151, %161] : f32 from vector<2x32xf32>
          %163 = vector.insert %162, %160 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %163 : vector<1x4xf32>
        }
        %138 = affine.apply #map10(%arg11)
        %139 = affine.apply #map17(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x32xf32>
        %142 = affine.apply #map18(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x32xf32>
        %145 = affine.apply #map19(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x32xf32>
        %148 = affine.apply #map20(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x32xf32>
        affine.yield %150 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_55 = arith.constant 0xFF800000 : f32
      %cst_56 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %75 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_56) -> (vector<2x1xf32>) {
        %137 = affine.apply #map(%arg11)
        %138 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %143 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_55) -> (f32) {
            %147 = affine.apply #map17(%arg13)
            %148 = frisk.fragment "compute" -> (f32) {
              %149 = vector.extract %74[%137, %147] : f32 from vector<2x32xf32>
              %150 = arith.maxnumf %arg14, %149 : f32
              %151 = affine.apply #map18(%arg13)
              %152 = vector.extract %74[%137, %151] : f32 from vector<2x32xf32>
              %153 = arith.maxnumf %150, %152 : f32
              %154 = affine.apply #map19(%arg13)
              %155 = vector.extract %74[%137, %154] : f32 from vector<2x32xf32>
              %156 = arith.maxnumf %153, %155 : f32
              %157 = affine.apply #map20(%arg13)
              %158 = vector.extract %74[%137, %157] : f32 from vector<2x32xf32>
              %159 = arith.maxnumf %156, %158 : f32
              frisk.fragment_yield %159 : f32
            }
            affine.yield %148 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_107 = arith.constant 64 : i32
          %c16_i32_108 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %143, %c16_i32_108, %c64_i32_107 : f32
          %144 = arith.maxnumf %143, %shuffleResult : f32
          %c64_i32_109 = arith.constant 64 : i32
          %c32_i32_110 = arith.constant 32 : i32
          %shuffleResult_111, %valid_112 = gpu.shuffle xor %144, %c32_i32_110, %c64_i32_109 : f32
          %145 = arith.maxnumf %144, %shuffleResult_111 : f32
          %146 = vector.insert %145, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %146 : vector<1x1xf32>
        }
        %139 = affine.apply #map(%arg11)
        %140 = affine.apply #map1()
        %141 = vector.extract %138[0, 0] : f32 from vector<1x1xf32>
        %142 = vector.insert %141, %arg12 [%139, %140] : f32 into vector<2x1xf32>
        affine.yield %142 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %76 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_57 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %77 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_57) -> (vector<2x1xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %142 = affine.apply #map(%arg11)
          %143 = affine.apply #map1()
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %144 = vector.extract %75[%142, %143] : f32 from vector<2x1xf32>
          %145 = vector.insert %144, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          %146 = affine.apply #map(%arg11)
          %147 = affine.apply #map1()
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %148 = vector.extract %76[%146, %147] : f32 from vector<2x1xf32>
          %149 = vector.insert %148, %cst_107 [0, 0] : f32 into vector<1x1xf32>
          %150 = arith.maxnumf %145, %149 {origin = "frisk.max", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %150 : vector<1x1xf32>
        }
        %138 = affine.apply #map(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %78 = frisk.to_threadTile %72 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x128xf32> -> vector<2x32xf32>
      %cst_58 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %79 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_58) -> (vector<2x32xf32>) {
        %137 = affine.apply #map10(%arg11)
        %138 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %152 = vector.extract %77[%137, %c0] : f32 from vector<2x1xf32>
          %153 = vector.insert %152, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %154 = vector.extract %77[%137, %c0_107] : f32 from vector<2x1xf32>
          %155 = vector.insert %154, %153 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %156 = vector.extract %77[%137, %c0_108] : f32 from vector<2x1xf32>
          %157 = vector.insert %156, %155 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %158 = vector.extract %77[%137, %c0_109] : f32 from vector<2x1xf32>
          %159 = vector.insert %158, %157 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %159 : vector<1x4xf32>
        }
        %139 = affine.apply #map10(%arg11)
        %140 = affine.apply #map17(%arg11)
        %141 = vector.extract %138[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg12 [%139, %140] : f32 into vector<2x32xf32>
        %143 = affine.apply #map18(%arg11)
        %144 = vector.extract %138[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %142 [%139, %143] : f32 into vector<2x32xf32>
        %146 = affine.apply #map19(%arg11)
        %147 = vector.extract %138[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %145 [%139, %146] : f32 into vector<2x32xf32>
        %149 = affine.apply #map20(%arg11)
        %150 = vector.extract %138[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %148 [%139, %149] : f32 into vector<2x32xf32>
        affine.yield %151 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>}
      %cst_59 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %80 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_59) -> (vector<2x32xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map10(%arg11)
          %152 = affine.apply #map17(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %78[%151, %152] : f32 from vector<2x32xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map18(%arg11)
          %156 = vector.extract %78[%151, %155] : f32 from vector<2x32xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map19(%arg11)
          %159 = vector.extract %78[%151, %158] : f32 from vector<2x32xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map20(%arg11)
          %162 = vector.extract %78[%151, %161] : f32 from vector<2x32xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = affine.apply #map10(%arg11)
          %165 = affine.apply #map17(%arg11)
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %166 = vector.extract %79[%164, %165] : f32 from vector<2x32xf32>
          %167 = vector.insert %166, %cst_107 [0, 0] : f32 into vector<1x4xf32>
          %168 = affine.apply #map18(%arg11)
          %169 = vector.extract %79[%164, %168] : f32 from vector<2x32xf32>
          %170 = vector.insert %169, %167 [0, 1] : f32 into vector<1x4xf32>
          %171 = affine.apply #map19(%arg11)
          %172 = vector.extract %79[%164, %171] : f32 from vector<2x32xf32>
          %173 = vector.insert %172, %170 [0, 2] : f32 into vector<1x4xf32>
          %174 = affine.apply #map20(%arg11)
          %175 = vector.extract %79[%164, %174] : f32 from vector<2x32xf32>
          %176 = vector.insert %175, %173 [0, 3] : f32 into vector<1x4xf32>
          %177 = arith.subf %163, %176 {origin = "frisk.sub", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %177 : vector<1x4xf32>
        }
        %138 = affine.apply #map10(%arg11)
        %139 = affine.apply #map17(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x32xf32>
        %142 = affine.apply #map18(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x32xf32>
        %145 = affine.apply #map19(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x32xf32>
        %148 = affine.apply #map20(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x32xf32>
        affine.yield %150 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %81 = frisk.to_threadTile %arg7 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_60 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %82 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_60) -> (vector<2x1xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %142 = affine.apply #map(%arg11)
          %143 = affine.apply #map1()
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %144 = vector.extract %81[%142, %143] : f32 from vector<2x1xf32>
          %145 = vector.insert %144, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          %146 = affine.apply #map(%arg11)
          %147 = affine.apply #map1()
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %148 = vector.extract %77[%146, %147] : f32 from vector<2x1xf32>
          %149 = vector.insert %148, %cst_107 [0, 0] : f32 into vector<1x1xf32>
          %150 = arith.subf %145, %149 {origin = "frisk.sub", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %150 : vector<1x1xf32>
        }
        %138 = affine.apply #map(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_61 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %83 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_61) -> (vector<2x1xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %142 = affine.apply #map(%arg11)
          %143 = affine.apply #map1()
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %144 = vector.extract %82[%142, %143] : f32 from vector<2x1xf32>
          %145 = vector.insert %144, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          %146 = math.exp2 %145 {tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %146 : vector<1x1xf32>
        }
        %138 = affine.apply #map(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %84 = frisk.to_threadTile %arg6 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
      %cst_62 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %85 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_62) -> (vector<2x1xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %142 = affine.apply #map(%arg11)
          %143 = affine.apply #map1()
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %144 = vector.extract %84[%142, %143] : f32 from vector<2x1xf32>
          %145 = vector.insert %144, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          %146 = affine.apply #map(%arg11)
          %147 = affine.apply #map1()
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %148 = vector.extract %83[%146, %147] : f32 from vector<2x1xf32>
          %149 = vector.insert %148, %cst_107 [0, 0] : f32 into vector<1x1xf32>
          %150 = arith.mulf %145, %149 {origin = "frisk.mul", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %150 : vector<1x1xf32>
        }
        %138 = affine.apply #map(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_63 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %86 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_63) -> (vector<2x32xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map10(%arg11)
          %152 = affine.apply #map17(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %80[%151, %152] : f32 from vector<2x32xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map18(%arg11)
          %156 = vector.extract %80[%151, %155] : f32 from vector<2x32xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map19(%arg11)
          %159 = vector.extract %80[%151, %158] : f32 from vector<2x32xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map20(%arg11)
          %162 = vector.extract %80[%151, %161] : f32 from vector<2x32xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = math.exp2 %163 {tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %164 : vector<1x4xf32>
        }
        %138 = affine.apply #map10(%arg11)
        %139 = affine.apply #map17(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x32xf32>
        %142 = affine.apply #map18(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x32xf32>
        %145 = affine.apply #map19(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x32xf32>
        %148 = affine.apply #map20(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x32xf32>
        affine.yield %150 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_64 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
      %87 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_64) -> (vector<2x32xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map10(%arg11)
          %152 = affine.apply #map17(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %86[%151, %152] : f32 from vector<2x32xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map18(%arg11)
          %156 = vector.extract %86[%151, %155] : f32 from vector<2x32xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map19(%arg11)
          %159 = vector.extract %86[%151, %158] : f32 from vector<2x32xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map20(%arg11)
          %162 = vector.extract %86[%151, %161] : f32 from vector<2x32xf32>
          %163 = vector.insert %162, %160 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %163 : vector<1x4xf32>
        }
        %138 = affine.apply #map10(%arg11)
        %139 = affine.apply #map17(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x32xf32>
        %142 = affine.apply #map18(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x32xf32>
        %145 = affine.apply #map19(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x32xf32>
        %148 = affine.apply #map20(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x32xf32>
        affine.yield %150 : vector<2x32xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %cst_65 = arith.constant 0.000000e+00 : f32
      %cst_66 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %88 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_66) -> (vector<2x1xf32>) {
        %137 = affine.apply #map(%arg11)
        %138 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %143 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_65) -> (f32) {
            %147 = affine.apply #map17(%arg13)
            %148 = frisk.fragment "compute" -> (f32) {
              %149 = vector.extract %87[%137, %147] : f32 from vector<2x32xf32>
              %150 = arith.addf %arg14, %149 : f32
              %151 = affine.apply #map18(%arg13)
              %152 = vector.extract %87[%137, %151] : f32 from vector<2x32xf32>
              %153 = arith.addf %150, %152 : f32
              %154 = affine.apply #map19(%arg13)
              %155 = vector.extract %87[%137, %154] : f32 from vector<2x32xf32>
              %156 = arith.addf %153, %155 : f32
              %157 = affine.apply #map20(%arg13)
              %158 = vector.extract %87[%137, %157] : f32 from vector<2x32xf32>
              %159 = arith.addf %156, %158 : f32
              frisk.fragment_yield %159 : f32
            }
            affine.yield %148 : f32
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
          %c64_i32_107 = arith.constant 64 : i32
          %c16_i32_108 = arith.constant 16 : i32
          %shuffleResult, %valid = gpu.shuffle xor %143, %c16_i32_108, %c64_i32_107 : f32
          %144 = arith.addf %143, %shuffleResult : f32
          %c64_i32_109 = arith.constant 64 : i32
          %c32_i32_110 = arith.constant 32 : i32
          %shuffleResult_111, %valid_112 = gpu.shuffle xor %144, %c32_i32_110, %c64_i32_109 : f32
          %145 = arith.addf %144, %shuffleResult_111 : f32
          %146 = vector.insert %145, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %146 : vector<1x1xf32>
        }
        %139 = affine.apply #map(%arg11)
        %140 = affine.apply #map1()
        %141 = vector.extract %138[0, 0] : f32 from vector<1x1xf32>
        %142 = vector.insert %141, %arg12 [%139, %140] : f32 into vector<2x1xf32>
        affine.yield %142 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
      %cst_67 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %89 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_67) -> (vector<2x1xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %142 = affine.apply #map(%arg11)
          %143 = affine.apply #map1()
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %144 = vector.extract %88[%142, %143] : f32 from vector<2x1xf32>
          %145 = vector.insert %144, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          %146 = affine.apply #map(%arg11)
          %147 = affine.apply #map1()
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %148 = vector.extract %85[%146, %147] : f32 from vector<2x1xf32>
          %149 = vector.insert %148, %cst_107 [0, 0] : f32 into vector<1x1xf32>
          %150 = arith.addf %145, %149 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
          frisk.fragment_yield %150 : vector<1x1xf32>
        }
        %138 = affine.apply #map(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %cst_68 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
      %90 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_68) -> (vector<2x32xf16>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %151 = affine.apply #map10(%arg11)
          %152 = affine.apply #map17(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %86[%151, %152] : f32 from vector<2x32xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map18(%arg11)
          %156 = vector.extract %86[%151, %155] : f32 from vector<2x32xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map19(%arg11)
          %159 = vector.extract %86[%151, %158] : f32 from vector<2x32xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map20(%arg11)
          %162 = vector.extract %86[%151, %161] : f32 from vector<2x32xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = arith.truncf %163 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %164 : vector<1x4xf16>
        }
        %138 = affine.apply #map10(%arg11)
        %139 = affine.apply #map17(%arg11)
        %140 = vector.extract %137[0, 0] : f16 from vector<1x4xf16>
        %141 = vector.insert %140, %arg12 [%138, %139] : f16 into vector<2x32xf16>
        %142 = affine.apply #map18(%arg11)
        %143 = vector.extract %137[0, 1] : f16 from vector<1x4xf16>
        %144 = vector.insert %143, %141 [%138, %142] : f16 into vector<2x32xf16>
        %145 = affine.apply #map19(%arg11)
        %146 = vector.extract %137[0, 2] : f16 from vector<1x4xf16>
        %147 = vector.insert %146, %144 [%138, %145] : f16 into vector<2x32xf16>
        %148 = affine.apply #map20(%arg11)
        %149 = vector.extract %137[0, 3] : f16 from vector<1x4xf16>
        %150 = vector.insert %149, %147 [%138, %148] : f16 into vector<2x32xf16>
        affine.yield %150 : vector<2x32xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
      %91 = frisk.from_threadTile %90 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = -1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x32xf16> -> memref<128x128xf16, 3>
      %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %92 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %7 + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %92 to %alloc {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc : memref<128x32xf16, 3>
      %thread_id_x_69 = gpu.thread_id x
      %cst_70 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %93 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_70) -> (vector<32x2xf16>) {
        %137 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %166 = affine.apply #map21(%arg11)[%thread_id_x_69]
          %167 = affine.apply #map22()[%thread_id_x_69]
          %168 = frisk.read_matrix_from_shm %alloc[%166, %167] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %169 = vector.bitcast %168 : vector<4xi32> to vector<8xf16>
          %cst_106 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %170 = affine.apply #map23()[%thread_id_x_69]
          %171 = vector.extract %169[%170] : f16 from vector<8xf16>
          %172 = vector.insert %171, %cst_106 [0, 0] : f16 into vector<4x2xf16>
          %173 = affine.apply #map24()[%thread_id_x_69]
          %174 = vector.extract %169[%173] : f16 from vector<8xf16>
          %175 = vector.insert %174, %172 [0, 1] : f16 into vector<4x2xf16>
          %176 = affine.apply #map25(%arg11)[%thread_id_x_69]
          %177 = vector.extract %169[%176] : f16 from vector<8xf16>
          %178 = vector.insert %177, %175 [1, 0] : f16 into vector<4x2xf16>
          %179 = affine.apply #map26(%arg11)[%thread_id_x_69]
          %180 = vector.extract %169[%179] : f16 from vector<8xf16>
          %181 = vector.insert %180, %178 [1, 1] : f16 into vector<4x2xf16>
          %182 = affine.apply #map27(%arg11)[%thread_id_x_69]
          %183 = vector.extract %169[%182] : f16 from vector<8xf16>
          %184 = vector.insert %183, %181 [2, 0] : f16 into vector<4x2xf16>
          %185 = affine.apply #map28(%arg11)[%thread_id_x_69]
          %186 = vector.extract %169[%185] : f16 from vector<8xf16>
          %187 = vector.insert %186, %184 [2, 1] : f16 into vector<4x2xf16>
          %188 = affine.apply #map29(%arg11)[%thread_id_x_69]
          %189 = vector.extract %169[%188] : f16 from vector<8xf16>
          %190 = vector.insert %189, %187 [3, 0] : f16 into vector<4x2xf16>
          %191 = affine.apply #map30(%arg11)[%thread_id_x_69]
          %192 = vector.extract %169[%191] : f16 from vector<8xf16>
          %193 = vector.insert %192, %190 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %193 : vector<4x2xf16>
        }
        %138 = affine.apply #map17(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f16 from vector<4x2xf16>
        %141 = vector.insert %140, %arg12 [%138, %139] : f16 into vector<32x2xf16>
        %142 = affine.apply #map31()
        %143 = vector.extract %137[0, 1] : f16 from vector<4x2xf16>
        %144 = vector.insert %143, %141 [%138, %142] : f16 into vector<32x2xf16>
        %145 = affine.apply #map18(%arg11)
        %146 = vector.extract %137[1, 0] : f16 from vector<4x2xf16>
        %147 = vector.insert %146, %144 [%145, %139] : f16 into vector<32x2xf16>
        %148 = affine.apply #map31()
        %149 = affine.apply #map18(%arg11)
        %150 = vector.extract %137[1, 1] : f16 from vector<4x2xf16>
        %151 = vector.insert %150, %147 [%149, %148] : f16 into vector<32x2xf16>
        %152 = affine.apply #map19(%arg11)
        %153 = vector.extract %137[2, 0] : f16 from vector<4x2xf16>
        %154 = vector.insert %153, %151 [%152, %139] : f16 into vector<32x2xf16>
        %155 = affine.apply #map31()
        %156 = affine.apply #map19(%arg11)
        %157 = vector.extract %137[2, 1] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %154 [%156, %155] : f16 into vector<32x2xf16>
        %159 = affine.apply #map20(%arg11)
        %160 = vector.extract %137[3, 0] : f16 from vector<4x2xf16>
        %161 = vector.insert %160, %158 [%159, %139] : f16 into vector<32x2xf16>
        %162 = affine.apply #map31()
        %163 = affine.apply #map20(%arg11)
        %164 = vector.extract %137[3, 1] : f16 from vector<4x2xf16>
        %165 = vector.insert %164, %161 [%163, %162] : f16 into vector<32x2xf16>
        affine.yield %165 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_71 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_72 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %94 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_71) -> (vector<2x8xf32>) {
        %137 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_72) -> (vector<1x4xf32>) {
          %151 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %154 = frisk.to_threadTile %91 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map32, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map33, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %155 = affine.apply #map5(%arg11)
            %156 = affine.apply #map11(%arg13)
            %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %157 = vector.extract %154[%155, %156] : f16 from vector<2x32xf16>
            %158 = vector.insert %157, %cst_106 [0, 0] : f16 into vector<1x4xf16>
            %159 = affine.apply #map12(%arg13)
            %160 = vector.extract %154[%155, %159] : f16 from vector<2x32xf16>
            %161 = vector.insert %160, %158 [0, 1] : f16 into vector<1x4xf16>
            %162 = affine.apply #map13(%arg13)
            %163 = vector.extract %154[%155, %162] : f16 from vector<2x32xf16>
            %164 = vector.insert %163, %161 [0, 2] : f16 into vector<1x4xf16>
            %165 = affine.apply #map14(%arg13)
            %166 = vector.extract %154[%155, %165] : f16 from vector<2x32xf16>
            %167 = vector.insert %166, %164 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %167 : vector<1x4xf16>
          }
          %152 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %154 = affine.apply #map11(%arg13)
            %155 = affine.apply #map(%arg11)
            %cst_106 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %156 = vector.extract %93[%154, %155] : f16 from vector<32x2xf16>
            %157 = vector.insert %156, %cst_106 [0, 0] : f16 into vector<4x1xf16>
            %158 = affine.apply #map12(%arg13)
            %159 = vector.extract %93[%158, %155] : f16 from vector<32x2xf16>
            %160 = vector.insert %159, %157 [1, 0] : f16 into vector<4x1xf16>
            %161 = affine.apply #map13(%arg13)
            %162 = vector.extract %93[%161, %155] : f16 from vector<32x2xf16>
            %163 = vector.insert %162, %160 [2, 0] : f16 into vector<4x1xf16>
            %164 = affine.apply #map14(%arg13)
            %165 = vector.extract %93[%164, %155] : f16 from vector<32x2xf16>
            %166 = vector.insert %165, %163 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %166 : vector<4x1xf16>
          }
          %153 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %154 = frisk.warp_mma_rr %151, %152, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %154 : vector<1x4xf32>
          }
          affine.yield %153 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %95 = frisk.from_threadTile %83 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 4>, tile_layout = #map34, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> memref<128x1xf32>
      %96 = frisk.to_threadTile %arg5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %97 = frisk.to_threadTile %95 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_73 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %98 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_73) -> (vector<2x8xf32>) {
        %137 = affine.apply #map5(%arg11)
        %138 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %152 = vector.extract %97[%137, %c0] : f32 from vector<2x1xf32>
          %153 = vector.insert %152, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %154 = vector.extract %97[%137, %c0_107] : f32 from vector<2x1xf32>
          %155 = vector.insert %154, %153 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %156 = vector.extract %97[%137, %c0_108] : f32 from vector<2x1xf32>
          %157 = vector.insert %156, %155 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %158 = vector.extract %97[%137, %c0_109] : f32 from vector<2x1xf32>
          %159 = vector.insert %158, %157 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %159 : vector<1x4xf32>
        }
        %139 = affine.apply #map5(%arg11)
        %140 = affine.apply #map6(%arg11)
        %141 = vector.extract %138[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg12 [%139, %140] : f32 into vector<2x8xf32>
        %143 = affine.apply #map7(%arg11)
        %144 = vector.extract %138[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %142 [%139, %143] : f32 into vector<2x8xf32>
        %146 = affine.apply #map8(%arg11)
        %147 = vector.extract %138[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %145 [%139, %146] : f32 into vector<2x8xf32>
        %149 = affine.apply #map9(%arg11)
        %150 = vector.extract %138[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %148 [%139, %149] : f32 into vector<2x8xf32>
        affine.yield %151 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_74 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %99 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_74) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %96[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %96[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %96[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %96[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = affine.apply #map5(%arg11)
          %165 = affine.apply #map6(%arg11)
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %166 = vector.extract %98[%164, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %cst_107 [0, 0] : f32 into vector<1x4xf32>
          %168 = affine.apply #map7(%arg11)
          %169 = vector.extract %98[%164, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 1] : f32 into vector<1x4xf32>
          %171 = affine.apply #map8(%arg11)
          %172 = vector.extract %98[%164, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 2] : f32 into vector<1x4xf32>
          %174 = affine.apply #map9(%arg11)
          %175 = vector.extract %98[%164, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 3] : f32 into vector<1x4xf32>
          %177 = arith.mulf %163, %176 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %177 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_75 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %100 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_75) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %99[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %99[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %99[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %99[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = affine.apply #map5(%arg11)
          %165 = affine.apply #map6(%arg11)
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %166 = vector.extract %94[%164, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %cst_107 [0, 0] : f32 into vector<1x4xf32>
          %168 = affine.apply #map7(%arg11)
          %169 = vector.extract %94[%164, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 1] : f32 into vector<1x4xf32>
          %171 = affine.apply #map8(%arg11)
          %172 = vector.extract %94[%164, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 2] : f32 into vector<1x4xf32>
          %174 = affine.apply #map9(%arg11)
          %175 = vector.extract %94[%164, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 3] : f32 into vector<1x4xf32>
          %177 = arith.addf %163, %176 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %177 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_76 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %101 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %101 to %alloc_76 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_76 : memref<128x32xf16, 3>
      %thread_id_x_77 = gpu.thread_id x
      %cst_78 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %102 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_78) -> (vector<32x2xf16>) {
        %137 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %166 = affine.apply #map21(%arg11)[%thread_id_x_77]
          %167 = affine.apply #map22()[%thread_id_x_77]
          %168 = frisk.read_matrix_from_shm %alloc_76[%166, %167] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %169 = vector.bitcast %168 : vector<4xi32> to vector<8xf16>
          %cst_106 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %170 = affine.apply #map23()[%thread_id_x_77]
          %171 = vector.extract %169[%170] : f16 from vector<8xf16>
          %172 = vector.insert %171, %cst_106 [0, 0] : f16 into vector<4x2xf16>
          %173 = affine.apply #map24()[%thread_id_x_77]
          %174 = vector.extract %169[%173] : f16 from vector<8xf16>
          %175 = vector.insert %174, %172 [0, 1] : f16 into vector<4x2xf16>
          %176 = affine.apply #map25(%arg11)[%thread_id_x_77]
          %177 = vector.extract %169[%176] : f16 from vector<8xf16>
          %178 = vector.insert %177, %175 [1, 0] : f16 into vector<4x2xf16>
          %179 = affine.apply #map26(%arg11)[%thread_id_x_77]
          %180 = vector.extract %169[%179] : f16 from vector<8xf16>
          %181 = vector.insert %180, %178 [1, 1] : f16 into vector<4x2xf16>
          %182 = affine.apply #map27(%arg11)[%thread_id_x_77]
          %183 = vector.extract %169[%182] : f16 from vector<8xf16>
          %184 = vector.insert %183, %181 [2, 0] : f16 into vector<4x2xf16>
          %185 = affine.apply #map28(%arg11)[%thread_id_x_77]
          %186 = vector.extract %169[%185] : f16 from vector<8xf16>
          %187 = vector.insert %186, %184 [2, 1] : f16 into vector<4x2xf16>
          %188 = affine.apply #map29(%arg11)[%thread_id_x_77]
          %189 = vector.extract %169[%188] : f16 from vector<8xf16>
          %190 = vector.insert %189, %187 [3, 0] : f16 into vector<4x2xf16>
          %191 = affine.apply #map30(%arg11)[%thread_id_x_77]
          %192 = vector.extract %169[%191] : f16 from vector<8xf16>
          %193 = vector.insert %192, %190 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %193 : vector<4x2xf16>
        }
        %138 = affine.apply #map17(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f16 from vector<4x2xf16>
        %141 = vector.insert %140, %arg12 [%138, %139] : f16 into vector<32x2xf16>
        %142 = affine.apply #map31()
        %143 = vector.extract %137[0, 1] : f16 from vector<4x2xf16>
        %144 = vector.insert %143, %141 [%138, %142] : f16 into vector<32x2xf16>
        %145 = affine.apply #map18(%arg11)
        %146 = vector.extract %137[1, 0] : f16 from vector<4x2xf16>
        %147 = vector.insert %146, %144 [%145, %139] : f16 into vector<32x2xf16>
        %148 = affine.apply #map31()
        %149 = affine.apply #map18(%arg11)
        %150 = vector.extract %137[1, 1] : f16 from vector<4x2xf16>
        %151 = vector.insert %150, %147 [%149, %148] : f16 into vector<32x2xf16>
        %152 = affine.apply #map19(%arg11)
        %153 = vector.extract %137[2, 0] : f16 from vector<4x2xf16>
        %154 = vector.insert %153, %151 [%152, %139] : f16 into vector<32x2xf16>
        %155 = affine.apply #map31()
        %156 = affine.apply #map19(%arg11)
        %157 = vector.extract %137[2, 1] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %154 [%156, %155] : f16 into vector<32x2xf16>
        %159 = affine.apply #map20(%arg11)
        %160 = vector.extract %137[3, 0] : f16 from vector<4x2xf16>
        %161 = vector.insert %160, %158 [%159, %139] : f16 into vector<32x2xf16>
        %162 = affine.apply #map31()
        %163 = affine.apply #map20(%arg11)
        %164 = vector.extract %137[3, 1] : f16 from vector<4x2xf16>
        %165 = vector.insert %164, %161 [%163, %162] : f16 into vector<32x2xf16>
        affine.yield %165 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_79 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_80 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %103 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_79) -> (vector<2x8xf32>) {
        %137 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_80) -> (vector<1x4xf32>) {
          %151 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %154 = frisk.to_threadTile %91 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map32, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map33, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %155 = affine.apply #map5(%arg11)
            %156 = affine.apply #map11(%arg13)
            %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %157 = vector.extract %154[%155, %156] : f16 from vector<2x32xf16>
            %158 = vector.insert %157, %cst_106 [0, 0] : f16 into vector<1x4xf16>
            %159 = affine.apply #map12(%arg13)
            %160 = vector.extract %154[%155, %159] : f16 from vector<2x32xf16>
            %161 = vector.insert %160, %158 [0, 1] : f16 into vector<1x4xf16>
            %162 = affine.apply #map13(%arg13)
            %163 = vector.extract %154[%155, %162] : f16 from vector<2x32xf16>
            %164 = vector.insert %163, %161 [0, 2] : f16 into vector<1x4xf16>
            %165 = affine.apply #map14(%arg13)
            %166 = vector.extract %154[%155, %165] : f16 from vector<2x32xf16>
            %167 = vector.insert %166, %164 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %167 : vector<1x4xf16>
          }
          %152 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %154 = affine.apply #map11(%arg13)
            %155 = affine.apply #map(%arg11)
            %cst_106 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %156 = vector.extract %102[%154, %155] : f16 from vector<32x2xf16>
            %157 = vector.insert %156, %cst_106 [0, 0] : f16 into vector<4x1xf16>
            %158 = affine.apply #map12(%arg13)
            %159 = vector.extract %102[%158, %155] : f16 from vector<32x2xf16>
            %160 = vector.insert %159, %157 [1, 0] : f16 into vector<4x1xf16>
            %161 = affine.apply #map13(%arg13)
            %162 = vector.extract %102[%161, %155] : f16 from vector<32x2xf16>
            %163 = vector.insert %162, %160 [2, 0] : f16 into vector<4x1xf16>
            %164 = affine.apply #map14(%arg13)
            %165 = vector.extract %102[%164, %155] : f16 from vector<32x2xf16>
            %166 = vector.insert %165, %163 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %166 : vector<4x1xf16>
          }
          %153 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %154 = frisk.warp_mma_rr %151, %152, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %154 : vector<1x4xf32>
          }
          affine.yield %153 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %104 = frisk.to_threadTile %arg8 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %105 = frisk.to_threadTile %95 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_81 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %106 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_81) -> (vector<2x8xf32>) {
        %137 = affine.apply #map5(%arg11)
        %138 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %152 = vector.extract %105[%137, %c0] : f32 from vector<2x1xf32>
          %153 = vector.insert %152, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %154 = vector.extract %105[%137, %c0_107] : f32 from vector<2x1xf32>
          %155 = vector.insert %154, %153 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %156 = vector.extract %105[%137, %c0_108] : f32 from vector<2x1xf32>
          %157 = vector.insert %156, %155 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %158 = vector.extract %105[%137, %c0_109] : f32 from vector<2x1xf32>
          %159 = vector.insert %158, %157 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %159 : vector<1x4xf32>
        }
        %139 = affine.apply #map5(%arg11)
        %140 = affine.apply #map6(%arg11)
        %141 = vector.extract %138[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg12 [%139, %140] : f32 into vector<2x8xf32>
        %143 = affine.apply #map7(%arg11)
        %144 = vector.extract %138[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %142 [%139, %143] : f32 into vector<2x8xf32>
        %146 = affine.apply #map8(%arg11)
        %147 = vector.extract %138[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %145 [%139, %146] : f32 into vector<2x8xf32>
        %149 = affine.apply #map9(%arg11)
        %150 = vector.extract %138[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %148 [%139, %149] : f32 into vector<2x8xf32>
        affine.yield %151 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_82 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %107 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_82) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %104[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %104[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %104[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %104[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = affine.apply #map5(%arg11)
          %165 = affine.apply #map6(%arg11)
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %166 = vector.extract %106[%164, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %cst_107 [0, 0] : f32 into vector<1x4xf32>
          %168 = affine.apply #map7(%arg11)
          %169 = vector.extract %106[%164, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 1] : f32 into vector<1x4xf32>
          %171 = affine.apply #map8(%arg11)
          %172 = vector.extract %106[%164, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 2] : f32 into vector<1x4xf32>
          %174 = affine.apply #map9(%arg11)
          %175 = vector.extract %106[%164, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 3] : f32 into vector<1x4xf32>
          %177 = arith.mulf %163, %176 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %177 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_83 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %108 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_83) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %107[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %107[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %107[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %107[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = affine.apply #map5(%arg11)
          %165 = affine.apply #map6(%arg11)
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %166 = vector.extract %103[%164, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %cst_107 [0, 0] : f32 into vector<1x4xf32>
          %168 = affine.apply #map7(%arg11)
          %169 = vector.extract %103[%164, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 1] : f32 into vector<1x4xf32>
          %171 = affine.apply #map8(%arg11)
          %172 = vector.extract %103[%164, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 2] : f32 into vector<1x4xf32>
          %174 = affine.apply #map9(%arg11)
          %175 = vector.extract %103[%164, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 3] : f32 into vector<1x4xf32>
          %177 = arith.addf %163, %176 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %177 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_84 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %109 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %109 to %alloc_84 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_84 : memref<128x32xf16, 3>
      %thread_id_x_85 = gpu.thread_id x
      %cst_86 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %110 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_86) -> (vector<32x2xf16>) {
        %137 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %166 = affine.apply #map21(%arg11)[%thread_id_x_85]
          %167 = affine.apply #map22()[%thread_id_x_85]
          %168 = frisk.read_matrix_from_shm %alloc_84[%166, %167] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %169 = vector.bitcast %168 : vector<4xi32> to vector<8xf16>
          %cst_106 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %170 = affine.apply #map23()[%thread_id_x_85]
          %171 = vector.extract %169[%170] : f16 from vector<8xf16>
          %172 = vector.insert %171, %cst_106 [0, 0] : f16 into vector<4x2xf16>
          %173 = affine.apply #map24()[%thread_id_x_85]
          %174 = vector.extract %169[%173] : f16 from vector<8xf16>
          %175 = vector.insert %174, %172 [0, 1] : f16 into vector<4x2xf16>
          %176 = affine.apply #map25(%arg11)[%thread_id_x_85]
          %177 = vector.extract %169[%176] : f16 from vector<8xf16>
          %178 = vector.insert %177, %175 [1, 0] : f16 into vector<4x2xf16>
          %179 = affine.apply #map26(%arg11)[%thread_id_x_85]
          %180 = vector.extract %169[%179] : f16 from vector<8xf16>
          %181 = vector.insert %180, %178 [1, 1] : f16 into vector<4x2xf16>
          %182 = affine.apply #map27(%arg11)[%thread_id_x_85]
          %183 = vector.extract %169[%182] : f16 from vector<8xf16>
          %184 = vector.insert %183, %181 [2, 0] : f16 into vector<4x2xf16>
          %185 = affine.apply #map28(%arg11)[%thread_id_x_85]
          %186 = vector.extract %169[%185] : f16 from vector<8xf16>
          %187 = vector.insert %186, %184 [2, 1] : f16 into vector<4x2xf16>
          %188 = affine.apply #map29(%arg11)[%thread_id_x_85]
          %189 = vector.extract %169[%188] : f16 from vector<8xf16>
          %190 = vector.insert %189, %187 [3, 0] : f16 into vector<4x2xf16>
          %191 = affine.apply #map30(%arg11)[%thread_id_x_85]
          %192 = vector.extract %169[%191] : f16 from vector<8xf16>
          %193 = vector.insert %192, %190 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %193 : vector<4x2xf16>
        }
        %138 = affine.apply #map17(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f16 from vector<4x2xf16>
        %141 = vector.insert %140, %arg12 [%138, %139] : f16 into vector<32x2xf16>
        %142 = affine.apply #map31()
        %143 = vector.extract %137[0, 1] : f16 from vector<4x2xf16>
        %144 = vector.insert %143, %141 [%138, %142] : f16 into vector<32x2xf16>
        %145 = affine.apply #map18(%arg11)
        %146 = vector.extract %137[1, 0] : f16 from vector<4x2xf16>
        %147 = vector.insert %146, %144 [%145, %139] : f16 into vector<32x2xf16>
        %148 = affine.apply #map31()
        %149 = affine.apply #map18(%arg11)
        %150 = vector.extract %137[1, 1] : f16 from vector<4x2xf16>
        %151 = vector.insert %150, %147 [%149, %148] : f16 into vector<32x2xf16>
        %152 = affine.apply #map19(%arg11)
        %153 = vector.extract %137[2, 0] : f16 from vector<4x2xf16>
        %154 = vector.insert %153, %151 [%152, %139] : f16 into vector<32x2xf16>
        %155 = affine.apply #map31()
        %156 = affine.apply #map19(%arg11)
        %157 = vector.extract %137[2, 1] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %154 [%156, %155] : f16 into vector<32x2xf16>
        %159 = affine.apply #map20(%arg11)
        %160 = vector.extract %137[3, 0] : f16 from vector<4x2xf16>
        %161 = vector.insert %160, %158 [%159, %139] : f16 into vector<32x2xf16>
        %162 = affine.apply #map31()
        %163 = affine.apply #map20(%arg11)
        %164 = vector.extract %137[3, 1] : f16 from vector<4x2xf16>
        %165 = vector.insert %164, %161 [%163, %162] : f16 into vector<32x2xf16>
        affine.yield %165 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_87 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_88 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %111 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_87) -> (vector<2x8xf32>) {
        %137 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_88) -> (vector<1x4xf32>) {
          %151 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %154 = frisk.to_threadTile %91 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map32, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map33, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %155 = affine.apply #map5(%arg11)
            %156 = affine.apply #map11(%arg13)
            %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %157 = vector.extract %154[%155, %156] : f16 from vector<2x32xf16>
            %158 = vector.insert %157, %cst_106 [0, 0] : f16 into vector<1x4xf16>
            %159 = affine.apply #map12(%arg13)
            %160 = vector.extract %154[%155, %159] : f16 from vector<2x32xf16>
            %161 = vector.insert %160, %158 [0, 1] : f16 into vector<1x4xf16>
            %162 = affine.apply #map13(%arg13)
            %163 = vector.extract %154[%155, %162] : f16 from vector<2x32xf16>
            %164 = vector.insert %163, %161 [0, 2] : f16 into vector<1x4xf16>
            %165 = affine.apply #map14(%arg13)
            %166 = vector.extract %154[%155, %165] : f16 from vector<2x32xf16>
            %167 = vector.insert %166, %164 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %167 : vector<1x4xf16>
          }
          %152 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %154 = affine.apply #map11(%arg13)
            %155 = affine.apply #map(%arg11)
            %cst_106 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %156 = vector.extract %110[%154, %155] : f16 from vector<32x2xf16>
            %157 = vector.insert %156, %cst_106 [0, 0] : f16 into vector<4x1xf16>
            %158 = affine.apply #map12(%arg13)
            %159 = vector.extract %110[%158, %155] : f16 from vector<32x2xf16>
            %160 = vector.insert %159, %157 [1, 0] : f16 into vector<4x1xf16>
            %161 = affine.apply #map13(%arg13)
            %162 = vector.extract %110[%161, %155] : f16 from vector<32x2xf16>
            %163 = vector.insert %162, %160 [2, 0] : f16 into vector<4x1xf16>
            %164 = affine.apply #map14(%arg13)
            %165 = vector.extract %110[%164, %155] : f16 from vector<32x2xf16>
            %166 = vector.insert %165, %163 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %166 : vector<4x1xf16>
          }
          %153 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %154 = frisk.warp_mma_rr %151, %152, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %154 : vector<1x4xf32>
          }
          affine.yield %153 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %112 = frisk.to_threadTile %arg9 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %113 = frisk.to_threadTile %95 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_89 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %114 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_89) -> (vector<2x8xf32>) {
        %137 = affine.apply #map5(%arg11)
        %138 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %152 = vector.extract %113[%137, %c0] : f32 from vector<2x1xf32>
          %153 = vector.insert %152, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %154 = vector.extract %113[%137, %c0_107] : f32 from vector<2x1xf32>
          %155 = vector.insert %154, %153 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %156 = vector.extract %113[%137, %c0_108] : f32 from vector<2x1xf32>
          %157 = vector.insert %156, %155 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %158 = vector.extract %113[%137, %c0_109] : f32 from vector<2x1xf32>
          %159 = vector.insert %158, %157 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %159 : vector<1x4xf32>
        }
        %139 = affine.apply #map5(%arg11)
        %140 = affine.apply #map6(%arg11)
        %141 = vector.extract %138[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg12 [%139, %140] : f32 into vector<2x8xf32>
        %143 = affine.apply #map7(%arg11)
        %144 = vector.extract %138[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %142 [%139, %143] : f32 into vector<2x8xf32>
        %146 = affine.apply #map8(%arg11)
        %147 = vector.extract %138[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %145 [%139, %146] : f32 into vector<2x8xf32>
        %149 = affine.apply #map9(%arg11)
        %150 = vector.extract %138[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %148 [%139, %149] : f32 into vector<2x8xf32>
        affine.yield %151 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_90 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %115 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_90) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %112[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %112[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %112[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %112[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = affine.apply #map5(%arg11)
          %165 = affine.apply #map6(%arg11)
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %166 = vector.extract %114[%164, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %cst_107 [0, 0] : f32 into vector<1x4xf32>
          %168 = affine.apply #map7(%arg11)
          %169 = vector.extract %114[%164, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 1] : f32 into vector<1x4xf32>
          %171 = affine.apply #map8(%arg11)
          %172 = vector.extract %114[%164, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 2] : f32 into vector<1x4xf32>
          %174 = affine.apply #map9(%arg11)
          %175 = vector.extract %114[%164, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 3] : f32 into vector<1x4xf32>
          %177 = arith.mulf %163, %176 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %177 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_91 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %116 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_91) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %115[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %115[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %115[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %115[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = affine.apply #map5(%arg11)
          %165 = affine.apply #map6(%arg11)
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %166 = vector.extract %111[%164, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %cst_107 [0, 0] : f32 into vector<1x4xf32>
          %168 = affine.apply #map7(%arg11)
          %169 = vector.extract %111[%164, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 1] : f32 into vector<1x4xf32>
          %171 = affine.apply #map8(%arg11)
          %172 = vector.extract %111[%164, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 2] : f32 into vector<1x4xf32>
          %174 = affine.apply #map9(%arg11)
          %175 = vector.extract %111[%164, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 3] : f32 into vector<1x4xf32>
          %177 = arith.addf %163, %176 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %177 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %alloc_92 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 1 : i64, tiled = true} : memref<128x32xf16, 3>
      %117 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %7 * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %117 to %alloc_92 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %alloc_92 : memref<128x32xf16, 3>
      %thread_id_x_93 = gpu.thread_id x
      %cst_94 = arith.constant dense<0.000000e+00> : vector<32x2xf16>
      %118 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_94) -> (vector<32x2xf16>) {
        %137 = frisk.fragment "load" -> (vector<4x2xf16>) {
          %166 = affine.apply #map21(%arg11)[%thread_id_x_93]
          %167 = affine.apply #map22()[%thread_id_x_93]
          %168 = frisk.read_matrix_from_shm %alloc_92[%166, %167] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<128x32xf16, 3> -> vector<4xi32>
          %169 = vector.bitcast %168 : vector<4xi32> to vector<8xf16>
          %cst_106 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
          %170 = affine.apply #map23()[%thread_id_x_93]
          %171 = vector.extract %169[%170] : f16 from vector<8xf16>
          %172 = vector.insert %171, %cst_106 [0, 0] : f16 into vector<4x2xf16>
          %173 = affine.apply #map24()[%thread_id_x_93]
          %174 = vector.extract %169[%173] : f16 from vector<8xf16>
          %175 = vector.insert %174, %172 [0, 1] : f16 into vector<4x2xf16>
          %176 = affine.apply #map25(%arg11)[%thread_id_x_93]
          %177 = vector.extract %169[%176] : f16 from vector<8xf16>
          %178 = vector.insert %177, %175 [1, 0] : f16 into vector<4x2xf16>
          %179 = affine.apply #map26(%arg11)[%thread_id_x_93]
          %180 = vector.extract %169[%179] : f16 from vector<8xf16>
          %181 = vector.insert %180, %178 [1, 1] : f16 into vector<4x2xf16>
          %182 = affine.apply #map27(%arg11)[%thread_id_x_93]
          %183 = vector.extract %169[%182] : f16 from vector<8xf16>
          %184 = vector.insert %183, %181 [2, 0] : f16 into vector<4x2xf16>
          %185 = affine.apply #map28(%arg11)[%thread_id_x_93]
          %186 = vector.extract %169[%185] : f16 from vector<8xf16>
          %187 = vector.insert %186, %184 [2, 1] : f16 into vector<4x2xf16>
          %188 = affine.apply #map29(%arg11)[%thread_id_x_93]
          %189 = vector.extract %169[%188] : f16 from vector<8xf16>
          %190 = vector.insert %189, %187 [3, 0] : f16 into vector<4x2xf16>
          %191 = affine.apply #map30(%arg11)[%thread_id_x_93]
          %192 = vector.extract %169[%191] : f16 from vector<8xf16>
          %193 = vector.insert %192, %190 [3, 1] : f16 into vector<4x2xf16>
          frisk.fragment_yield %193 : vector<4x2xf16>
        }
        %138 = affine.apply #map17(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f16 from vector<4x2xf16>
        %141 = vector.insert %140, %arg12 [%138, %139] : f16 into vector<32x2xf16>
        %142 = affine.apply #map31()
        %143 = vector.extract %137[0, 1] : f16 from vector<4x2xf16>
        %144 = vector.insert %143, %141 [%138, %142] : f16 into vector<32x2xf16>
        %145 = affine.apply #map18(%arg11)
        %146 = vector.extract %137[1, 0] : f16 from vector<4x2xf16>
        %147 = vector.insert %146, %144 [%145, %139] : f16 into vector<32x2xf16>
        %148 = affine.apply #map31()
        %149 = affine.apply #map18(%arg11)
        %150 = vector.extract %137[1, 1] : f16 from vector<4x2xf16>
        %151 = vector.insert %150, %147 [%149, %148] : f16 into vector<32x2xf16>
        %152 = affine.apply #map19(%arg11)
        %153 = vector.extract %137[2, 0] : f16 from vector<4x2xf16>
        %154 = vector.insert %153, %151 [%152, %139] : f16 into vector<32x2xf16>
        %155 = affine.apply #map31()
        %156 = affine.apply #map19(%arg11)
        %157 = vector.extract %137[2, 1] : f16 from vector<4x2xf16>
        %158 = vector.insert %157, %154 [%156, %155] : f16 into vector<32x2xf16>
        %159 = affine.apply #map20(%arg11)
        %160 = vector.extract %137[3, 0] : f16 from vector<4x2xf16>
        %161 = vector.insert %160, %158 [%159, %139] : f16 into vector<32x2xf16>
        %162 = affine.apply #map31()
        %163 = affine.apply #map20(%arg11)
        %164 = vector.extract %137[3, 1] : f16 from vector<4x2xf16>
        %165 = vector.insert %164, %161 [%163, %162] : f16 into vector<32x2xf16>
        affine.yield %165 : vector<32x2xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 32, 2>, iterLabel = "matrix_load", tiled = true}
      %cst_95 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %cst_96 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
      %119 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_95) -> (vector<2x8xf32>) {
        %137 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst_96) -> (vector<1x4xf32>) {
          %151 = frisk.fragment "load" -> (vector<1x4xf16>) {
            %154 = frisk.to_threadTile %91 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, frisk.fragment_source, ignore_dim = -1 : i64, logical_transform = #map32, thread_creg_order = array<i64: 1, 0>, thread_widths = array<i64: 1, 4>, tile_layout = #map33, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 1>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x128xf16, 3> -> vector<2x32xf16>
            %155 = affine.apply #map5(%arg11)
            %156 = affine.apply #map11(%arg13)
            %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
            %157 = vector.extract %154[%155, %156] : f16 from vector<2x32xf16>
            %158 = vector.insert %157, %cst_106 [0, 0] : f16 into vector<1x4xf16>
            %159 = affine.apply #map12(%arg13)
            %160 = vector.extract %154[%155, %159] : f16 from vector<2x32xf16>
            %161 = vector.insert %160, %158 [0, 1] : f16 into vector<1x4xf16>
            %162 = affine.apply #map13(%arg13)
            %163 = vector.extract %154[%155, %162] : f16 from vector<2x32xf16>
            %164 = vector.insert %163, %161 [0, 2] : f16 into vector<1x4xf16>
            %165 = affine.apply #map14(%arg13)
            %166 = vector.extract %154[%155, %165] : f16 from vector<2x32xf16>
            %167 = vector.insert %166, %164 [0, 3] : f16 into vector<1x4xf16>
            frisk.fragment_yield %167 : vector<1x4xf16>
          }
          %152 = frisk.fragment "load" -> (vector<4x1xf16>) {
            %154 = affine.apply #map11(%arg13)
            %155 = affine.apply #map(%arg11)
            %cst_106 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
            %156 = vector.extract %118[%154, %155] : f16 from vector<32x2xf16>
            %157 = vector.insert %156, %cst_106 [0, 0] : f16 into vector<4x1xf16>
            %158 = affine.apply #map12(%arg13)
            %159 = vector.extract %118[%158, %155] : f16 from vector<32x2xf16>
            %160 = vector.insert %159, %157 [1, 0] : f16 into vector<4x1xf16>
            %161 = affine.apply #map13(%arg13)
            %162 = vector.extract %118[%161, %155] : f16 from vector<32x2xf16>
            %163 = vector.insert %162, %160 [2, 0] : f16 into vector<4x1xf16>
            %164 = affine.apply #map14(%arg13)
            %165 = vector.extract %118[%164, %155] : f16 from vector<32x2xf16>
            %166 = vector.insert %165, %163 [3, 0] : f16 into vector<4x1xf16>
            frisk.fragment_yield %166 : vector<4x1xf16>
          }
          %153 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %154 = frisk.warp_mma_rr %151, %152, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
            frisk.fragment_yield %154 : vector<1x4xf32>
          }
          affine.yield %153 : vector<1x4xf32>
        } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "gemm_mn", tiled = true}
      %120 = frisk.to_threadTile %arg10 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
      %121 = frisk.to_threadTile %95 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = 1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : memref<128x1xf32> -> vector<2x1xf32>
      %cst_97 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %122 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_97) -> (vector<2x8xf32>) {
        %137 = affine.apply #map5(%arg11)
        %138 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %c0 = arith.constant 0 : index
          %152 = vector.extract %121[%137, %c0] : f32 from vector<2x1xf32>
          %153 = vector.insert %152, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %c0_107 = arith.constant 0 : index
          %154 = vector.extract %121[%137, %c0_107] : f32 from vector<2x1xf32>
          %155 = vector.insert %154, %153 [0, 1] : f32 into vector<1x4xf32>
          %c0_108 = arith.constant 0 : index
          %156 = vector.extract %121[%137, %c0_108] : f32 from vector<2x1xf32>
          %157 = vector.insert %156, %155 [0, 2] : f32 into vector<1x4xf32>
          %c0_109 = arith.constant 0 : index
          %158 = vector.extract %121[%137, %c0_109] : f32 from vector<2x1xf32>
          %159 = vector.insert %158, %157 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %159 : vector<1x4xf32>
        }
        %139 = affine.apply #map5(%arg11)
        %140 = affine.apply #map6(%arg11)
        %141 = vector.extract %138[0, 0] : f32 from vector<1x4xf32>
        %142 = vector.insert %141, %arg12 [%139, %140] : f32 into vector<2x8xf32>
        %143 = affine.apply #map7(%arg11)
        %144 = vector.extract %138[0, 1] : f32 from vector<1x4xf32>
        %145 = vector.insert %144, %142 [%139, %143] : f32 into vector<2x8xf32>
        %146 = affine.apply #map8(%arg11)
        %147 = vector.extract %138[0, 2] : f32 from vector<1x4xf32>
        %148 = vector.insert %147, %145 [%139, %146] : f32 into vector<2x8xf32>
        %149 = affine.apply #map9(%arg11)
        %150 = vector.extract %138[0, 3] : f32 from vector<1x4xf32>
        %151 = vector.insert %150, %148 [%139, %149] : f32 into vector<2x8xf32>
        affine.yield %151 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
      %cst_98 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %123 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_98) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %120[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %120[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %120[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %120[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = affine.apply #map5(%arg11)
          %165 = affine.apply #map6(%arg11)
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %166 = vector.extract %122[%164, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %cst_107 [0, 0] : f32 into vector<1x4xf32>
          %168 = affine.apply #map7(%arg11)
          %169 = vector.extract %122[%164, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 1] : f32 into vector<1x4xf32>
          %171 = affine.apply #map8(%arg11)
          %172 = vector.extract %122[%164, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 2] : f32 into vector<1x4xf32>
          %174 = affine.apply #map9(%arg11)
          %175 = vector.extract %122[%164, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 3] : f32 into vector<1x4xf32>
          %177 = arith.mulf %163, %176 {origin = "frisk.mul", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %177 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_99 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %124 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_99) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %123[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %123[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %123[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %123[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          %164 = affine.apply #map5(%arg11)
          %165 = affine.apply #map6(%arg11)
          %cst_107 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %166 = vector.extract %119[%164, %165] : f32 from vector<2x8xf32>
          %167 = vector.insert %166, %cst_107 [0, 0] : f32 into vector<1x4xf32>
          %168 = affine.apply #map7(%arg11)
          %169 = vector.extract %119[%164, %168] : f32 from vector<2x8xf32>
          %170 = vector.insert %169, %167 [0, 1] : f32 into vector<1x4xf32>
          %171 = affine.apply #map8(%arg11)
          %172 = vector.extract %119[%164, %171] : f32 from vector<2x8xf32>
          %173 = vector.insert %172, %170 [0, 2] : f32 into vector<1x4xf32>
          %174 = affine.apply #map9(%arg11)
          %175 = vector.extract %119[%164, %174] : f32 from vector<2x8xf32>
          %176 = vector.insert %175, %173 [0, 3] : f32 into vector<1x4xf32>
          %177 = arith.addf %163, %176 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %177 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %cst_100 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %125 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_100) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %100[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %100[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %100[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %100[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %163 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %126 = frisk.from_threadTile %125 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_101 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %127 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_101) -> (vector<2x1xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %142 = affine.apply #map(%arg11)
          %143 = affine.apply #map1()
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %144 = vector.extract %89[%142, %143] : f32 from vector<2x1xf32>
          %145 = vector.insert %144, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %145 : vector<1x1xf32>
        }
        %138 = affine.apply #map(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %128 = frisk.from_threadTile %127 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_102 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
      %129 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_102) -> (vector<2x1xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x1xf32>) {
          %142 = affine.apply #map(%arg11)
          %143 = affine.apply #map1()
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
          %144 = vector.extract %77[%142, %143] : f32 from vector<2x1xf32>
          %145 = vector.insert %144, %cst_106 [0, 0] : f32 into vector<1x1xf32>
          frisk.fragment_yield %145 : vector<1x1xf32>
        }
        %138 = affine.apply #map(%arg11)
        %139 = affine.apply #map1()
        %140 = vector.extract %137[0, 0] : f32 from vector<1x1xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x1xf32>
        affine.yield %141 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
      %130 = frisk.from_threadTile %129 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x1xf32> -> vector<128x1xf32>
      %cst_103 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %131 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_103) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %108[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %108[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %108[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %108[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %163 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %132 = frisk.from_threadTile %131 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_104 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %133 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_104) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %116[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %116[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %116[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %116[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %163 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %134 = frisk.from_threadTile %133 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      %cst_105 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
      %135 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_105) -> (vector<2x8xf32>) {
        %137 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %151 = affine.apply #map5(%arg11)
          %152 = affine.apply #map6(%arg11)
          %cst_106 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
          %153 = vector.extract %124[%151, %152] : f32 from vector<2x8xf32>
          %154 = vector.insert %153, %cst_106 [0, 0] : f32 into vector<1x4xf32>
          %155 = affine.apply #map7(%arg11)
          %156 = vector.extract %124[%151, %155] : f32 from vector<2x8xf32>
          %157 = vector.insert %156, %154 [0, 1] : f32 into vector<1x4xf32>
          %158 = affine.apply #map8(%arg11)
          %159 = vector.extract %124[%151, %158] : f32 from vector<2x8xf32>
          %160 = vector.insert %159, %157 [0, 2] : f32 into vector<1x4xf32>
          %161 = affine.apply #map9(%arg11)
          %162 = vector.extract %124[%151, %161] : f32 from vector<2x8xf32>
          %163 = vector.insert %162, %160 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %163 : vector<1x4xf32>
        }
        %138 = affine.apply #map5(%arg11)
        %139 = affine.apply #map6(%arg11)
        %140 = vector.extract %137[0, 0] : f32 from vector<1x4xf32>
        %141 = vector.insert %140, %arg12 [%138, %139] : f32 into vector<2x8xf32>
        %142 = affine.apply #map7(%arg11)
        %143 = vector.extract %137[0, 1] : f32 from vector<1x4xf32>
        %144 = vector.insert %143, %141 [%138, %142] : f32 into vector<2x8xf32>
        %145 = affine.apply #map8(%arg11)
        %146 = vector.extract %137[0, 2] : f32 from vector<1x4xf32>
        %147 = vector.insert %146, %144 [%138, %145] : f32 into vector<2x8xf32>
        %148 = affine.apply #map9(%arg11)
        %149 = vector.extract %137[0, 3] : f32 from vector<1x4xf32>
        %150 = vector.insert %149, %147 [%138, %148] : f32 into vector<2x8xf32>
        affine.yield %150 : vector<2x8xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
      %136 = frisk.from_threadTile %135 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<2x8xf32> -> vector<128x32xf32>
      affine.yield %126, %128, %130, %132, %134, %136 : vector<128x32xf32>, vector<128x1xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %15 = frisk.to_threadTile %14#5 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x = gpu.thread_id x
    %16 = arith.index_cast %thread_id_x : index to i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %17 = arith.andi %16, %c16_i32 : i32
    %18 = arith.cmpi ne, %17, %c0_i32 : i32
    %c32_i32 = arith.constant 32 : i32
    %19 = arith.andi %16, %c32_i32 : i32
    %20 = arith.cmpi ne, %19, %c0_i32 : i32
    %cst_11 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %21 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_11) -> (vector<2x8xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %15[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %15[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %15[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %15[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
        %100 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
        %101 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
        %102 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
        %103 = arith.select %18, %99, %100 : f32
        %shuffleResult, %valid = gpu.shuffle xor %103, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %104 = arith.select %18, %shuffleResult, %99 : f32
        %105 = arith.select %18, %100, %shuffleResult : f32
        %106 = arith.select %18, %101, %102 : f32
        %shuffleResult_54, %valid_55 = gpu.shuffle xor %106, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %18, %shuffleResult_54, %101 : f32
        %108 = arith.select %18, %102, %shuffleResult_54 : f32
        %109 = arith.select %20, %104, %107 : f32
        %shuffleResult_56, %valid_57 = gpu.shuffle xor %109, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %20, %shuffleResult_56, %104 : f32
        %111 = arith.select %20, %107, %shuffleResult_56 : f32
        %112 = arith.select %20, %105, %108 : f32
        %shuffleResult_58, %valid_59 = gpu.shuffle xor %112, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %20, %shuffleResult_58, %105 : f32
        %114 = arith.select %20, %108, %shuffleResult_58 : f32
        %115 = vector.insert %110, %98 [0, 0] : f32 into vector<1x4xf32>
        %116 = vector.insert %113, %115 [0, 1] : f32 into vector<1x4xf32>
        %117 = vector.insert %111, %116 [0, 2] : f32 into vector<1x4xf32>
        %118 = vector.insert %114, %117 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f32 from vector<1x4xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x8xf32>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f32 from vector<1x4xf32>
      %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x8xf32>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f32 from vector<1x4xf32>
      %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x8xf32>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f32 from vector<1x4xf32>
      %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x8xf32>
      affine.yield %85 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %22 = frisk.to_threadTile %14#4 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x_12 = gpu.thread_id x
    %23 = arith.index_cast %thread_id_x_12 : index to i32
    %c0_i32_13 = arith.constant 0 : i32
    %c64_i32_14 = arith.constant 64 : i32
    %c16_i32_15 = arith.constant 16 : i32
    %24 = arith.andi %23, %c16_i32_15 : i32
    %25 = arith.cmpi ne, %24, %c0_i32_13 : i32
    %c32_i32_16 = arith.constant 32 : i32
    %26 = arith.andi %23, %c32_i32_16 : i32
    %27 = arith.cmpi ne, %26, %c0_i32_13 : i32
    %cst_17 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %28 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_17) -> (vector<2x8xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %22[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %22[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %22[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %22[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
        %100 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
        %101 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
        %102 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
        %103 = arith.select %25, %99, %100 : f32
        %shuffleResult, %valid = gpu.shuffle xor %103, %c16_i32_15, %c64_i32_14 {frisk.layout_role = "wmmaC"} : f32
        %104 = arith.select %25, %shuffleResult, %99 : f32
        %105 = arith.select %25, %100, %shuffleResult : f32
        %106 = arith.select %25, %101, %102 : f32
        %shuffleResult_54, %valid_55 = gpu.shuffle xor %106, %c16_i32_15, %c64_i32_14 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %25, %shuffleResult_54, %101 : f32
        %108 = arith.select %25, %102, %shuffleResult_54 : f32
        %109 = arith.select %27, %104, %107 : f32
        %shuffleResult_56, %valid_57 = gpu.shuffle xor %109, %c32_i32_16, %c64_i32_14 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %27, %shuffleResult_56, %104 : f32
        %111 = arith.select %27, %107, %shuffleResult_56 : f32
        %112 = arith.select %27, %105, %108 : f32
        %shuffleResult_58, %valid_59 = gpu.shuffle xor %112, %c32_i32_16, %c64_i32_14 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %27, %shuffleResult_58, %105 : f32
        %114 = arith.select %27, %108, %shuffleResult_58 : f32
        %115 = vector.insert %110, %98 [0, 0] : f32 into vector<1x4xf32>
        %116 = vector.insert %113, %115 [0, 1] : f32 into vector<1x4xf32>
        %117 = vector.insert %111, %116 [0, 2] : f32 into vector<1x4xf32>
        %118 = vector.insert %114, %117 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f32 from vector<1x4xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x8xf32>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f32 from vector<1x4xf32>
      %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x8xf32>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f32 from vector<1x4xf32>
      %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x8xf32>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f32 from vector<1x4xf32>
      %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x8xf32>
      affine.yield %85 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %29 = frisk.to_threadTile %14#3 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x_18 = gpu.thread_id x
    %30 = arith.index_cast %thread_id_x_18 : index to i32
    %c0_i32_19 = arith.constant 0 : i32
    %c64_i32_20 = arith.constant 64 : i32
    %c16_i32_21 = arith.constant 16 : i32
    %31 = arith.andi %30, %c16_i32_21 : i32
    %32 = arith.cmpi ne, %31, %c0_i32_19 : i32
    %c32_i32_22 = arith.constant 32 : i32
    %33 = arith.andi %30, %c32_i32_22 : i32
    %34 = arith.cmpi ne, %33, %c0_i32_19 : i32
    %cst_23 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %35 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_23) -> (vector<2x8xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %29[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %29[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %29[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %29[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
        %100 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
        %101 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
        %102 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
        %103 = arith.select %32, %99, %100 : f32
        %shuffleResult, %valid = gpu.shuffle xor %103, %c16_i32_21, %c64_i32_20 {frisk.layout_role = "wmmaC"} : f32
        %104 = arith.select %32, %shuffleResult, %99 : f32
        %105 = arith.select %32, %100, %shuffleResult : f32
        %106 = arith.select %32, %101, %102 : f32
        %shuffleResult_54, %valid_55 = gpu.shuffle xor %106, %c16_i32_21, %c64_i32_20 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %32, %shuffleResult_54, %101 : f32
        %108 = arith.select %32, %102, %shuffleResult_54 : f32
        %109 = arith.select %34, %104, %107 : f32
        %shuffleResult_56, %valid_57 = gpu.shuffle xor %109, %c32_i32_22, %c64_i32_20 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %34, %shuffleResult_56, %104 : f32
        %111 = arith.select %34, %107, %shuffleResult_56 : f32
        %112 = arith.select %34, %105, %108 : f32
        %shuffleResult_58, %valid_59 = gpu.shuffle xor %112, %c32_i32_22, %c64_i32_20 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %34, %shuffleResult_58, %105 : f32
        %114 = arith.select %34, %108, %shuffleResult_58 : f32
        %115 = vector.insert %110, %98 [0, 0] : f32 into vector<1x4xf32>
        %116 = vector.insert %113, %115 [0, 1] : f32 into vector<1x4xf32>
        %117 = vector.insert %111, %116 [0, 2] : f32 into vector<1x4xf32>
        %118 = vector.insert %114, %117 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f32 from vector<1x4xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x8xf32>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f32 from vector<1x4xf32>
      %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x8xf32>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f32 from vector<1x4xf32>
      %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x8xf32>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f32 from vector<1x4xf32>
      %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x8xf32>
      affine.yield %85 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %36 = frisk.to_threadTile %14#0 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 2>, ignore_dim = -1 : i64, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map35, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x32xf32> -> vector<2x8xf32>
    %thread_id_x_24 = gpu.thread_id x
    %37 = arith.index_cast %thread_id_x_24 : index to i32
    %c0_i32_25 = arith.constant 0 : i32
    %c64_i32_26 = arith.constant 64 : i32
    %c16_i32_27 = arith.constant 16 : i32
    %38 = arith.andi %37, %c16_i32_27 : i32
    %39 = arith.cmpi ne, %38, %c0_i32_25 : i32
    %c32_i32_28 = arith.constant 32 : i32
    %40 = arith.andi %37, %c32_i32_28 : i32
    %41 = arith.cmpi ne, %40, %c0_i32_25 : i32
    %cst_29 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %42 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_29) -> (vector<2x8xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %36[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %36[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %36[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %36[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = vector.extract %98[0, 0] : f32 from vector<1x4xf32>
        %100 = vector.extract %98[0, 1] : f32 from vector<1x4xf32>
        %101 = vector.extract %98[0, 2] : f32 from vector<1x4xf32>
        %102 = vector.extract %98[0, 3] : f32 from vector<1x4xf32>
        %103 = arith.select %39, %99, %100 : f32
        %shuffleResult, %valid = gpu.shuffle xor %103, %c16_i32_27, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
        %104 = arith.select %39, %shuffleResult, %99 : f32
        %105 = arith.select %39, %100, %shuffleResult : f32
        %106 = arith.select %39, %101, %102 : f32
        %shuffleResult_54, %valid_55 = gpu.shuffle xor %106, %c16_i32_27, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
        %107 = arith.select %39, %shuffleResult_54, %101 : f32
        %108 = arith.select %39, %102, %shuffleResult_54 : f32
        %109 = arith.select %41, %104, %107 : f32
        %shuffleResult_56, %valid_57 = gpu.shuffle xor %109, %c32_i32_28, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
        %110 = arith.select %41, %shuffleResult_56, %104 : f32
        %111 = arith.select %41, %107, %shuffleResult_56 : f32
        %112 = arith.select %41, %105, %108 : f32
        %shuffleResult_58, %valid_59 = gpu.shuffle xor %112, %c32_i32_28, %c64_i32_26 {frisk.layout_role = "wmmaC"} : f32
        %113 = arith.select %41, %shuffleResult_58, %105 : f32
        %114 = arith.select %41, %108, %shuffleResult_58 : f32
        %115 = vector.insert %110, %98 [0, 0] : f32 into vector<1x4xf32>
        %116 = vector.insert %113, %115 [0, 1] : f32 into vector<1x4xf32>
        %117 = vector.insert %111, %116 [0, 2] : f32 into vector<1x4xf32>
        %118 = vector.insert %114, %117 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
        frisk.fragment_yield %118 : vector<1x4xf32>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f32 from vector<1x4xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x8xf32>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f32 from vector<1x4xf32>
      %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x8xf32>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f32 from vector<1x4xf32>
      %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x8xf32>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f32 from vector<1x4xf32>
      %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x8xf32>
      affine.yield %85 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %43 = frisk.to_threadTile %14#1 {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, block_repeat = array<i64: 2, 8>, ignore_dim = 1 : i64, logical_transform = #map2, thread_creg_order = array<i64: 0, 1>, thread_widths = array<i64: 1, 1>, tile_layout = #map3, tiled = true, warp_inst_unroll = array<i64: 1, 1>, warp_layout = array<i64: 16, 4>, warp_layout_order = array<i64: 0, 1>, warp_repeat = array<i64: 1, 4>, warp_repeat_order = array<i64: 0, 1>, warp_threads = 64 : i64} : vector<128x1xf32> -> vector<2x1xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %44 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_30) -> (vector<2x8xf32>) {
      %72 = affine.apply #map5(%arg4)
      %73 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %87 = vector.extract %43[%72, %c0] : f32 from vector<2x1xf32>
        %88 = vector.insert %87, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %c0_54 = arith.constant 0 : index
        %89 = vector.extract %43[%72, %c0_54] : f32 from vector<2x1xf32>
        %90 = vector.insert %89, %88 [0, 1] : f32 into vector<1x4xf32>
        %c0_55 = arith.constant 0 : index
        %91 = vector.extract %43[%72, %c0_55] : f32 from vector<2x1xf32>
        %92 = vector.insert %91, %90 [0, 2] : f32 into vector<1x4xf32>
        %c0_56 = arith.constant 0 : index
        %93 = vector.extract %43[%72, %c0_56] : f32 from vector<2x1xf32>
        %94 = vector.insert %93, %92 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %94 : vector<1x4xf32>
      }
      %74 = affine.apply #map5(%arg4)
      %75 = affine.apply #map6(%arg4)
      %76 = vector.extract %73[0, 0] : f32 from vector<1x4xf32>
      %77 = vector.insert %76, %arg5 [%74, %75] : f32 into vector<2x8xf32>
      %78 = affine.apply #map7(%arg4)
      %79 = vector.extract %73[0, 1] : f32 from vector<1x4xf32>
      %80 = vector.insert %79, %77 [%74, %78] : f32 into vector<2x8xf32>
      %81 = affine.apply #map8(%arg4)
      %82 = vector.extract %73[0, 2] : f32 from vector<1x4xf32>
      %83 = vector.insert %82, %80 [%74, %81] : f32 into vector<2x8xf32>
      %84 = affine.apply #map9(%arg4)
      %85 = vector.extract %73[0, 3] : f32 from vector<1x4xf32>
      %86 = vector.insert %85, %83 [%74, %84] : f32 into vector<2x8xf32>
      affine.yield %86 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_31 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %45 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_31) -> (vector<2x8xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %42[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %42[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %42[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %42[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = affine.apply #map5(%arg4)
        %100 = affine.apply #map6(%arg4)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %101 = vector.extract %44[%99, %100] : f32 from vector<2x8xf32>
        %102 = vector.insert %101, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %103 = affine.apply #map7(%arg4)
        %104 = vector.extract %44[%99, %103] : f32 from vector<2x8xf32>
        %105 = vector.insert %104, %102 [0, 1] : f32 into vector<1x4xf32>
        %106 = affine.apply #map8(%arg4)
        %107 = vector.extract %44[%99, %106] : f32 from vector<2x8xf32>
        %108 = vector.insert %107, %105 [0, 2] : f32 into vector<1x4xf32>
        %109 = affine.apply #map9(%arg4)
        %110 = vector.extract %44[%99, %109] : f32 from vector<2x8xf32>
        %111 = vector.insert %110, %108 [0, 3] : f32 into vector<1x4xf32>
        %112 = arith.divf %98, %111 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %112 : vector<1x4xf32>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f32 from vector<1x4xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x8xf32>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f32 from vector<1x4xf32>
      %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x8xf32>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f32 from vector<1x4xf32>
      %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x8xf32>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f32 from vector<1x4xf32>
      %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x8xf32>
      affine.yield %85 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_32 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %46 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_32) -> (vector<2x8xf16>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %45[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %45[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %45[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %45[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = arith.truncf %98 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %99 : vector<1x4xf16>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
      %76 = vector.insert %75, %arg5 [%73, %74] : f16 into vector<2x8xf16>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
      %79 = vector.insert %78, %76 [%73, %77] : f16 into vector<2x8xf16>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
      %82 = vector.insert %81, %79 [%73, %80] : f16 into vector<2x8xf16>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
      %85 = vector.insert %84, %82 [%73, %83] : f16 into vector<2x8xf16>
      affine.yield %85 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %47 = frisk.buffer_view %arg3[0, (%block_id_z * 32 + %7 + (%6 * 16384) floordiv 524288) mod 32, (%6 * 128) mod 4096, 0], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_33 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %48 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_33) -> (vector<2x8xf16>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %88 = vector.extract %46[%86, %87] : f16 from vector<2x8xf16>
        %89 = vector.insert %88, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %46[%86, %90] : f16 from vector<2x8xf16>
        %92 = vector.insert %91, %89 [0, 1] : f16 into vector<1x4xf16>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %46[%86, %93] : f16 from vector<2x8xf16>
        %95 = vector.insert %94, %92 [0, 2] : f16 into vector<1x4xf16>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %46[%86, %96] : f16 from vector<2x8xf16>
        %98 = vector.insert %97, %95 [0, 3] : f16 into vector<1x4xf16>
        frisk.fragment_yield %98 : vector<1x4xf16>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
      %76 = vector.insert %75, %arg5 [%73, %74] : f16 into vector<2x8xf16>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
      %79 = vector.insert %78, %76 [%73, %77] : f16 into vector<2x8xf16>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
      %82 = vector.insert %81, %79 [%73, %80] : f16 into vector<2x8xf16>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
      %85 = vector.insert %84, %82 [%73, %83] : f16 into vector<2x8xf16>
      affine.yield %85 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_34 = gpu.thread_id x
    %49 = affine.apply #map36(%thread_id_x_34)
    %c4 = arith.constant 4 : index
    %50 = arith.cmpi ult, %49, %c4 : index
    scf.if %50 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
          %72 = affine.apply #map5(%arg4)
          %73 = affine.apply #map6(%arg4)
          %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %74 = vector.extract %48[%72, %73] : f16 from vector<2x8xf16>
          %75 = vector.insert %74, %cst_53 [0, 0] : f16 into vector<1x4xf16>
          %76 = affine.apply #map7(%arg4)
          %77 = vector.extract %48[%72, %76] : f16 from vector<2x8xf16>
          %78 = vector.insert %77, %75 [0, 1] : f16 into vector<1x4xf16>
          %79 = affine.apply #map8(%arg4)
          %80 = vector.extract %48[%72, %79] : f16 from vector<2x8xf16>
          %81 = vector.insert %80, %78 [0, 2] : f16 into vector<1x4xf16>
          %82 = affine.apply #map9(%arg4)
          %83 = vector.extract %48[%72, %82] : f16 from vector<2x8xf16>
          %84 = vector.insert %83, %81 [0, 3] : f16 into vector<1x4xf16>
          %thread_id_x_54 = gpu.thread_id x
          %85 = affine.apply #map5(%arg4)
          %86 = affine.apply #map6(%arg4)
          %87 = vector.extract %84[0, 0] : f16 from vector<1x4xf16>
          %c0 = arith.constant 0 : index
          %c0_55 = arith.constant 0 : index
          %88 = affine.apply #map37(%86)
          %c0_56 = arith.constant 0 : index
          %89 = affine.apply #map38(%86)
          %c0_57 = arith.constant 0 : index
          %90 = affine.apply #map39(%c0_55, %89)
          %91 = affine.apply #map40(%thread_id_x_54, %85, %88, %c0, %c0_56, %c0_57, %90)
          %92 = affine.apply #map41(%thread_id_x_54, %85, %88, %c0, %c0_56, %c0_57, %90)
          affine.store %87, %47[%91, %92] {tiled = true} : memref<128x32xf16, 1>
          %93 = affine.apply #map7(%arg4)
          %94 = vector.extract %84[0, 1] : f16 from vector<1x4xf16>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %95 = affine.apply #map37(%93)
          %c0_60 = arith.constant 0 : index
          %96 = affine.apply #map38(%93)
          %c0_61 = arith.constant 0 : index
          %97 = affine.apply #map39(%c0_59, %96)
          %98 = affine.apply #map40(%thread_id_x_54, %85, %95, %c0_58, %c0_60, %c0_61, %97)
          %99 = affine.apply #map41(%thread_id_x_54, %85, %95, %c0_58, %c0_60, %c0_61, %97)
          affine.store %94, %47[%98, %99] {tiled = true} : memref<128x32xf16, 1>
          %100 = affine.apply #map8(%arg4)
          %101 = vector.extract %84[0, 2] : f16 from vector<1x4xf16>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %102 = affine.apply #map37(%100)
          %c0_64 = arith.constant 0 : index
          %103 = affine.apply #map38(%100)
          %c0_65 = arith.constant 0 : index
          %104 = affine.apply #map39(%c0_63, %103)
          %105 = affine.apply #map40(%thread_id_x_54, %85, %102, %c0_62, %c0_64, %c0_65, %104)
          %106 = affine.apply #map41(%thread_id_x_54, %85, %102, %c0_62, %c0_64, %c0_65, %104)
          affine.store %101, %47[%105, %106] {tiled = true} : memref<128x32xf16, 1>
          %107 = affine.apply #map9(%arg4)
          %108 = vector.extract %84[0, 3] : f16 from vector<1x4xf16>
          %c0_66 = arith.constant 0 : index
          %c0_67 = arith.constant 0 : index
          %109 = affine.apply #map37(%107)
          %c0_68 = arith.constant 0 : index
          %110 = affine.apply #map38(%107)
          %c0_69 = arith.constant 0 : index
          %111 = affine.apply #map39(%c0_67, %110)
          %112 = affine.apply #map40(%thread_id_x_54, %85, %109, %c0_66, %c0_68, %c0_69, %111)
          %113 = affine.apply #map41(%thread_id_x_54, %85, %109, %c0_66, %c0_68, %c0_69, %111)
          affine.store %108, %47[%112, %113] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_35 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %51 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_35) -> (vector<2x8xf32>) {
      %72 = affine.apply #map5(%arg4)
      %73 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %87 = vector.extract %43[%72, %c0] : f32 from vector<2x1xf32>
        %88 = vector.insert %87, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %c0_54 = arith.constant 0 : index
        %89 = vector.extract %43[%72, %c0_54] : f32 from vector<2x1xf32>
        %90 = vector.insert %89, %88 [0, 1] : f32 into vector<1x4xf32>
        %c0_55 = arith.constant 0 : index
        %91 = vector.extract %43[%72, %c0_55] : f32 from vector<2x1xf32>
        %92 = vector.insert %91, %90 [0, 2] : f32 into vector<1x4xf32>
        %c0_56 = arith.constant 0 : index
        %93 = vector.extract %43[%72, %c0_56] : f32 from vector<2x1xf32>
        %94 = vector.insert %93, %92 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %94 : vector<1x4xf32>
      }
      %74 = affine.apply #map5(%arg4)
      %75 = affine.apply #map6(%arg4)
      %76 = vector.extract %73[0, 0] : f32 from vector<1x4xf32>
      %77 = vector.insert %76, %arg5 [%74, %75] : f32 into vector<2x8xf32>
      %78 = affine.apply #map7(%arg4)
      %79 = vector.extract %73[0, 1] : f32 from vector<1x4xf32>
      %80 = vector.insert %79, %77 [%74, %78] : f32 into vector<2x8xf32>
      %81 = affine.apply #map8(%arg4)
      %82 = vector.extract %73[0, 2] : f32 from vector<1x4xf32>
      %83 = vector.insert %82, %80 [%74, %81] : f32 into vector<2x8xf32>
      %84 = affine.apply #map9(%arg4)
      %85 = vector.extract %73[0, 3] : f32 from vector<1x4xf32>
      %86 = vector.insert %85, %83 [%74, %84] : f32 into vector<2x8xf32>
      affine.yield %86 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_36 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %52 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_36) -> (vector<2x8xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %35[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %35[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %35[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %35[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = affine.apply #map5(%arg4)
        %100 = affine.apply #map6(%arg4)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %101 = vector.extract %51[%99, %100] : f32 from vector<2x8xf32>
        %102 = vector.insert %101, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %103 = affine.apply #map7(%arg4)
        %104 = vector.extract %51[%99, %103] : f32 from vector<2x8xf32>
        %105 = vector.insert %104, %102 [0, 1] : f32 into vector<1x4xf32>
        %106 = affine.apply #map8(%arg4)
        %107 = vector.extract %51[%99, %106] : f32 from vector<2x8xf32>
        %108 = vector.insert %107, %105 [0, 2] : f32 into vector<1x4xf32>
        %109 = affine.apply #map9(%arg4)
        %110 = vector.extract %51[%99, %109] : f32 from vector<2x8xf32>
        %111 = vector.insert %110, %108 [0, 3] : f32 into vector<1x4xf32>
        %112 = arith.divf %98, %111 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %112 : vector<1x4xf32>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f32 from vector<1x4xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x8xf32>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f32 from vector<1x4xf32>
      %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x8xf32>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f32 from vector<1x4xf32>
      %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x8xf32>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f32 from vector<1x4xf32>
      %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x8xf32>
      affine.yield %85 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_37 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %53 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_37) -> (vector<2x8xf16>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %52[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %52[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %52[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %52[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = arith.truncf %98 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %99 : vector<1x4xf16>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
      %76 = vector.insert %75, %arg5 [%73, %74] : f16 into vector<2x8xf16>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
      %79 = vector.insert %78, %76 [%73, %77] : f16 into vector<2x8xf16>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
      %82 = vector.insert %81, %79 [%73, %80] : f16 into vector<2x8xf16>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
      %85 = vector.insert %84, %82 [%73, %83] : f16 into vector<2x8xf16>
      affine.yield %85 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %54 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 32) floordiv 524288) mod 32, (%6 * 128) mod 4096, 32], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_38 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %55 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_38) -> (vector<2x8xf16>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %88 = vector.extract %53[%86, %87] : f16 from vector<2x8xf16>
        %89 = vector.insert %88, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %53[%86, %90] : f16 from vector<2x8xf16>
        %92 = vector.insert %91, %89 [0, 1] : f16 into vector<1x4xf16>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %53[%86, %93] : f16 from vector<2x8xf16>
        %95 = vector.insert %94, %92 [0, 2] : f16 into vector<1x4xf16>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %53[%86, %96] : f16 from vector<2x8xf16>
        %98 = vector.insert %97, %95 [0, 3] : f16 into vector<1x4xf16>
        frisk.fragment_yield %98 : vector<1x4xf16>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
      %76 = vector.insert %75, %arg5 [%73, %74] : f16 into vector<2x8xf16>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
      %79 = vector.insert %78, %76 [%73, %77] : f16 into vector<2x8xf16>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
      %82 = vector.insert %81, %79 [%73, %80] : f16 into vector<2x8xf16>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
      %85 = vector.insert %84, %82 [%73, %83] : f16 into vector<2x8xf16>
      affine.yield %85 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_39 = gpu.thread_id x
    %56 = affine.apply #map36(%thread_id_x_39)
    %c4_40 = arith.constant 4 : index
    %57 = arith.cmpi ult, %56, %c4_40 : index
    scf.if %57 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
          %72 = affine.apply #map5(%arg4)
          %73 = affine.apply #map6(%arg4)
          %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %74 = vector.extract %55[%72, %73] : f16 from vector<2x8xf16>
          %75 = vector.insert %74, %cst_53 [0, 0] : f16 into vector<1x4xf16>
          %76 = affine.apply #map7(%arg4)
          %77 = vector.extract %55[%72, %76] : f16 from vector<2x8xf16>
          %78 = vector.insert %77, %75 [0, 1] : f16 into vector<1x4xf16>
          %79 = affine.apply #map8(%arg4)
          %80 = vector.extract %55[%72, %79] : f16 from vector<2x8xf16>
          %81 = vector.insert %80, %78 [0, 2] : f16 into vector<1x4xf16>
          %82 = affine.apply #map9(%arg4)
          %83 = vector.extract %55[%72, %82] : f16 from vector<2x8xf16>
          %84 = vector.insert %83, %81 [0, 3] : f16 into vector<1x4xf16>
          %thread_id_x_54 = gpu.thread_id x
          %85 = affine.apply #map5(%arg4)
          %86 = affine.apply #map6(%arg4)
          %87 = vector.extract %84[0, 0] : f16 from vector<1x4xf16>
          %c0 = arith.constant 0 : index
          %c0_55 = arith.constant 0 : index
          %88 = affine.apply #map37(%86)
          %c0_56 = arith.constant 0 : index
          %89 = affine.apply #map38(%86)
          %c0_57 = arith.constant 0 : index
          %90 = affine.apply #map39(%c0_55, %89)
          %91 = affine.apply #map40(%thread_id_x_54, %85, %88, %c0, %c0_56, %c0_57, %90)
          %92 = affine.apply #map41(%thread_id_x_54, %85, %88, %c0, %c0_56, %c0_57, %90)
          affine.store %87, %54[%91, %92] {tiled = true} : memref<128x32xf16, 1>
          %93 = affine.apply #map7(%arg4)
          %94 = vector.extract %84[0, 1] : f16 from vector<1x4xf16>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %95 = affine.apply #map37(%93)
          %c0_60 = arith.constant 0 : index
          %96 = affine.apply #map38(%93)
          %c0_61 = arith.constant 0 : index
          %97 = affine.apply #map39(%c0_59, %96)
          %98 = affine.apply #map40(%thread_id_x_54, %85, %95, %c0_58, %c0_60, %c0_61, %97)
          %99 = affine.apply #map41(%thread_id_x_54, %85, %95, %c0_58, %c0_60, %c0_61, %97)
          affine.store %94, %54[%98, %99] {tiled = true} : memref<128x32xf16, 1>
          %100 = affine.apply #map8(%arg4)
          %101 = vector.extract %84[0, 2] : f16 from vector<1x4xf16>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %102 = affine.apply #map37(%100)
          %c0_64 = arith.constant 0 : index
          %103 = affine.apply #map38(%100)
          %c0_65 = arith.constant 0 : index
          %104 = affine.apply #map39(%c0_63, %103)
          %105 = affine.apply #map40(%thread_id_x_54, %85, %102, %c0_62, %c0_64, %c0_65, %104)
          %106 = affine.apply #map41(%thread_id_x_54, %85, %102, %c0_62, %c0_64, %c0_65, %104)
          affine.store %101, %54[%105, %106] {tiled = true} : memref<128x32xf16, 1>
          %107 = affine.apply #map9(%arg4)
          %108 = vector.extract %84[0, 3] : f16 from vector<1x4xf16>
          %c0_66 = arith.constant 0 : index
          %c0_67 = arith.constant 0 : index
          %109 = affine.apply #map37(%107)
          %c0_68 = arith.constant 0 : index
          %110 = affine.apply #map38(%107)
          %c0_69 = arith.constant 0 : index
          %111 = affine.apply #map39(%c0_67, %110)
          %112 = affine.apply #map40(%thread_id_x_54, %85, %109, %c0_66, %c0_68, %c0_69, %111)
          %113 = affine.apply #map41(%thread_id_x_54, %85, %109, %c0_66, %c0_68, %c0_69, %111)
          affine.store %108, %54[%112, %113] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_41 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %58 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_41) -> (vector<2x8xf32>) {
      %72 = affine.apply #map5(%arg4)
      %73 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %87 = vector.extract %43[%72, %c0] : f32 from vector<2x1xf32>
        %88 = vector.insert %87, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %c0_54 = arith.constant 0 : index
        %89 = vector.extract %43[%72, %c0_54] : f32 from vector<2x1xf32>
        %90 = vector.insert %89, %88 [0, 1] : f32 into vector<1x4xf32>
        %c0_55 = arith.constant 0 : index
        %91 = vector.extract %43[%72, %c0_55] : f32 from vector<2x1xf32>
        %92 = vector.insert %91, %90 [0, 2] : f32 into vector<1x4xf32>
        %c0_56 = arith.constant 0 : index
        %93 = vector.extract %43[%72, %c0_56] : f32 from vector<2x1xf32>
        %94 = vector.insert %93, %92 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %94 : vector<1x4xf32>
      }
      %74 = affine.apply #map5(%arg4)
      %75 = affine.apply #map6(%arg4)
      %76 = vector.extract %73[0, 0] : f32 from vector<1x4xf32>
      %77 = vector.insert %76, %arg5 [%74, %75] : f32 into vector<2x8xf32>
      %78 = affine.apply #map7(%arg4)
      %79 = vector.extract %73[0, 1] : f32 from vector<1x4xf32>
      %80 = vector.insert %79, %77 [%74, %78] : f32 into vector<2x8xf32>
      %81 = affine.apply #map8(%arg4)
      %82 = vector.extract %73[0, 2] : f32 from vector<1x4xf32>
      %83 = vector.insert %82, %80 [%74, %81] : f32 into vector<2x8xf32>
      %84 = affine.apply #map9(%arg4)
      %85 = vector.extract %73[0, 3] : f32 from vector<1x4xf32>
      %86 = vector.insert %85, %83 [%74, %84] : f32 into vector<2x8xf32>
      affine.yield %86 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_42 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %59 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_42) -> (vector<2x8xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %28[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %28[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %28[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %28[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = affine.apply #map5(%arg4)
        %100 = affine.apply #map6(%arg4)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %101 = vector.extract %58[%99, %100] : f32 from vector<2x8xf32>
        %102 = vector.insert %101, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %103 = affine.apply #map7(%arg4)
        %104 = vector.extract %58[%99, %103] : f32 from vector<2x8xf32>
        %105 = vector.insert %104, %102 [0, 1] : f32 into vector<1x4xf32>
        %106 = affine.apply #map8(%arg4)
        %107 = vector.extract %58[%99, %106] : f32 from vector<2x8xf32>
        %108 = vector.insert %107, %105 [0, 2] : f32 into vector<1x4xf32>
        %109 = affine.apply #map9(%arg4)
        %110 = vector.extract %58[%99, %109] : f32 from vector<2x8xf32>
        %111 = vector.insert %110, %108 [0, 3] : f32 into vector<1x4xf32>
        %112 = arith.divf %98, %111 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %112 : vector<1x4xf32>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f32 from vector<1x4xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x8xf32>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f32 from vector<1x4xf32>
      %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x8xf32>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f32 from vector<1x4xf32>
      %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x8xf32>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f32 from vector<1x4xf32>
      %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x8xf32>
      affine.yield %85 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_43 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %60 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_43) -> (vector<2x8xf16>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %59[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %59[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %59[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %59[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = arith.truncf %98 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %99 : vector<1x4xf16>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
      %76 = vector.insert %75, %arg5 [%73, %74] : f16 into vector<2x8xf16>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
      %79 = vector.insert %78, %76 [%73, %77] : f16 into vector<2x8xf16>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
      %82 = vector.insert %81, %79 [%73, %80] : f16 into vector<2x8xf16>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
      %85 = vector.insert %84, %82 [%73, %83] : f16 into vector<2x8xf16>
      affine.yield %85 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %61 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 64) floordiv 524288) mod 32, (%6 * 128) mod 4096, 64], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_44 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %62 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_44) -> (vector<2x8xf16>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %88 = vector.extract %60[%86, %87] : f16 from vector<2x8xf16>
        %89 = vector.insert %88, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %60[%86, %90] : f16 from vector<2x8xf16>
        %92 = vector.insert %91, %89 [0, 1] : f16 into vector<1x4xf16>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %60[%86, %93] : f16 from vector<2x8xf16>
        %95 = vector.insert %94, %92 [0, 2] : f16 into vector<1x4xf16>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %60[%86, %96] : f16 from vector<2x8xf16>
        %98 = vector.insert %97, %95 [0, 3] : f16 into vector<1x4xf16>
        frisk.fragment_yield %98 : vector<1x4xf16>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
      %76 = vector.insert %75, %arg5 [%73, %74] : f16 into vector<2x8xf16>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
      %79 = vector.insert %78, %76 [%73, %77] : f16 into vector<2x8xf16>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
      %82 = vector.insert %81, %79 [%73, %80] : f16 into vector<2x8xf16>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
      %85 = vector.insert %84, %82 [%73, %83] : f16 into vector<2x8xf16>
      affine.yield %85 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_45 = gpu.thread_id x
    %63 = affine.apply #map36(%thread_id_x_45)
    %c4_46 = arith.constant 4 : index
    %64 = arith.cmpi ult, %63, %c4_46 : index
    scf.if %64 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
          %72 = affine.apply #map5(%arg4)
          %73 = affine.apply #map6(%arg4)
          %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %74 = vector.extract %62[%72, %73] : f16 from vector<2x8xf16>
          %75 = vector.insert %74, %cst_53 [0, 0] : f16 into vector<1x4xf16>
          %76 = affine.apply #map7(%arg4)
          %77 = vector.extract %62[%72, %76] : f16 from vector<2x8xf16>
          %78 = vector.insert %77, %75 [0, 1] : f16 into vector<1x4xf16>
          %79 = affine.apply #map8(%arg4)
          %80 = vector.extract %62[%72, %79] : f16 from vector<2x8xf16>
          %81 = vector.insert %80, %78 [0, 2] : f16 into vector<1x4xf16>
          %82 = affine.apply #map9(%arg4)
          %83 = vector.extract %62[%72, %82] : f16 from vector<2x8xf16>
          %84 = vector.insert %83, %81 [0, 3] : f16 into vector<1x4xf16>
          %thread_id_x_54 = gpu.thread_id x
          %85 = affine.apply #map5(%arg4)
          %86 = affine.apply #map6(%arg4)
          %87 = vector.extract %84[0, 0] : f16 from vector<1x4xf16>
          %c0 = arith.constant 0 : index
          %c0_55 = arith.constant 0 : index
          %88 = affine.apply #map37(%86)
          %c0_56 = arith.constant 0 : index
          %89 = affine.apply #map38(%86)
          %c0_57 = arith.constant 0 : index
          %90 = affine.apply #map39(%c0_55, %89)
          %91 = affine.apply #map40(%thread_id_x_54, %85, %88, %c0, %c0_56, %c0_57, %90)
          %92 = affine.apply #map41(%thread_id_x_54, %85, %88, %c0, %c0_56, %c0_57, %90)
          affine.store %87, %61[%91, %92] {tiled = true} : memref<128x32xf16, 1>
          %93 = affine.apply #map7(%arg4)
          %94 = vector.extract %84[0, 1] : f16 from vector<1x4xf16>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %95 = affine.apply #map37(%93)
          %c0_60 = arith.constant 0 : index
          %96 = affine.apply #map38(%93)
          %c0_61 = arith.constant 0 : index
          %97 = affine.apply #map39(%c0_59, %96)
          %98 = affine.apply #map40(%thread_id_x_54, %85, %95, %c0_58, %c0_60, %c0_61, %97)
          %99 = affine.apply #map41(%thread_id_x_54, %85, %95, %c0_58, %c0_60, %c0_61, %97)
          affine.store %94, %61[%98, %99] {tiled = true} : memref<128x32xf16, 1>
          %100 = affine.apply #map8(%arg4)
          %101 = vector.extract %84[0, 2] : f16 from vector<1x4xf16>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %102 = affine.apply #map37(%100)
          %c0_64 = arith.constant 0 : index
          %103 = affine.apply #map38(%100)
          %c0_65 = arith.constant 0 : index
          %104 = affine.apply #map39(%c0_63, %103)
          %105 = affine.apply #map40(%thread_id_x_54, %85, %102, %c0_62, %c0_64, %c0_65, %104)
          %106 = affine.apply #map41(%thread_id_x_54, %85, %102, %c0_62, %c0_64, %c0_65, %104)
          affine.store %101, %61[%105, %106] {tiled = true} : memref<128x32xf16, 1>
          %107 = affine.apply #map9(%arg4)
          %108 = vector.extract %84[0, 3] : f16 from vector<1x4xf16>
          %c0_66 = arith.constant 0 : index
          %c0_67 = arith.constant 0 : index
          %109 = affine.apply #map37(%107)
          %c0_68 = arith.constant 0 : index
          %110 = affine.apply #map38(%107)
          %c0_69 = arith.constant 0 : index
          %111 = affine.apply #map39(%c0_67, %110)
          %112 = affine.apply #map40(%thread_id_x_54, %85, %109, %c0_66, %c0_68, %c0_69, %111)
          %113 = affine.apply #map41(%thread_id_x_54, %85, %109, %c0_66, %c0_68, %c0_69, %111)
          affine.store %108, %61[%112, %113] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    %cst_47 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %65 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_47) -> (vector<2x8xf32>) {
      %72 = affine.apply #map5(%arg4)
      %73 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %c0 = arith.constant 0 : index
        %87 = vector.extract %43[%72, %c0] : f32 from vector<2x1xf32>
        %88 = vector.insert %87, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %c0_54 = arith.constant 0 : index
        %89 = vector.extract %43[%72, %c0_54] : f32 from vector<2x1xf32>
        %90 = vector.insert %89, %88 [0, 1] : f32 into vector<1x4xf32>
        %c0_55 = arith.constant 0 : index
        %91 = vector.extract %43[%72, %c0_55] : f32 from vector<2x1xf32>
        %92 = vector.insert %91, %90 [0, 2] : f32 into vector<1x4xf32>
        %c0_56 = arith.constant 0 : index
        %93 = vector.extract %43[%72, %c0_56] : f32 from vector<2x1xf32>
        %94 = vector.insert %93, %92 [0, 3] : f32 into vector<1x4xf32>
        frisk.fragment_yield %94 : vector<1x4xf32>
      }
      %74 = affine.apply #map5(%arg4)
      %75 = affine.apply #map6(%arg4)
      %76 = vector.extract %73[0, 0] : f32 from vector<1x4xf32>
      %77 = vector.insert %76, %arg5 [%74, %75] : f32 into vector<2x8xf32>
      %78 = affine.apply #map7(%arg4)
      %79 = vector.extract %73[0, 1] : f32 from vector<1x4xf32>
      %80 = vector.insert %79, %77 [%74, %78] : f32 into vector<2x8xf32>
      %81 = affine.apply #map8(%arg4)
      %82 = vector.extract %73[0, 2] : f32 from vector<1x4xf32>
      %83 = vector.insert %82, %80 [%74, %81] : f32 into vector<2x8xf32>
      %84 = affine.apply #map9(%arg4)
      %85 = vector.extract %73[0, 3] : f32 from vector<1x4xf32>
      %86 = vector.insert %85, %83 [%74, %84] : f32 into vector<2x8xf32>
      affine.yield %86 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>}
    %cst_48 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %66 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_48) -> (vector<2x8xf32>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf32>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %21[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %21[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %21[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %21[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = affine.apply #map5(%arg4)
        %100 = affine.apply #map6(%arg4)
        %cst_54 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %101 = vector.extract %65[%99, %100] : f32 from vector<2x8xf32>
        %102 = vector.insert %101, %cst_54 [0, 0] : f32 into vector<1x4xf32>
        %103 = affine.apply #map7(%arg4)
        %104 = vector.extract %65[%99, %103] : f32 from vector<2x8xf32>
        %105 = vector.insert %104, %102 [0, 1] : f32 into vector<1x4xf32>
        %106 = affine.apply #map8(%arg4)
        %107 = vector.extract %65[%99, %106] : f32 from vector<2x8xf32>
        %108 = vector.insert %107, %105 [0, 2] : f32 into vector<1x4xf32>
        %109 = affine.apply #map9(%arg4)
        %110 = vector.extract %65[%99, %109] : f32 from vector<2x8xf32>
        %111 = vector.insert %110, %108 [0, 3] : f32 into vector<1x4xf32>
        %112 = arith.divf %98, %111 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
        frisk.fragment_yield %112 : vector<1x4xf32>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f32 from vector<1x4xf32>
      %76 = vector.insert %75, %arg5 [%73, %74] : f32 into vector<2x8xf32>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f32 from vector<1x4xf32>
      %79 = vector.insert %78, %76 [%73, %77] : f32 into vector<2x8xf32>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f32 from vector<1x4xf32>
      %82 = vector.insert %81, %79 [%73, %80] : f32 into vector<2x8xf32>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f32 from vector<1x4xf32>
      %85 = vector.insert %84, %82 [%73, %83] : f32 into vector<2x8xf32>
      affine.yield %85 : vector<2x8xf32>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %cst_49 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %67 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_49) -> (vector<2x8xf16>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
        %88 = vector.extract %66[%86, %87] : f32 from vector<2x8xf32>
        %89 = vector.insert %88, %cst_53 [0, 0] : f32 into vector<1x4xf32>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %66[%86, %90] : f32 from vector<2x8xf32>
        %92 = vector.insert %91, %89 [0, 1] : f32 into vector<1x4xf32>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %66[%86, %93] : f32 from vector<2x8xf32>
        %95 = vector.insert %94, %92 [0, 2] : f32 into vector<1x4xf32>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %66[%86, %96] : f32 from vector<2x8xf32>
        %98 = vector.insert %97, %95 [0, 3] : f32 into vector<1x4xf32>
        %99 = arith.truncf %98 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
        frisk.fragment_yield %99 : vector<1x4xf16>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
      %76 = vector.insert %75, %arg5 [%73, %74] : f16 into vector<2x8xf16>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
      %79 = vector.insert %78, %76 [%73, %77] : f16 into vector<2x8xf16>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
      %82 = vector.insert %81, %79 [%73, %80] : f16 into vector<2x8xf16>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
      %85 = vector.insert %84, %82 [%73, %83] : f16 into vector<2x8xf16>
      affine.yield %85 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %68 = frisk.buffer_view %arg3[0, ((%block_id_z * 16777216 + %7 * 524288 + %6 * 16384 + 96) floordiv 524288) mod 32, (%6 * 128) mod 4096, 96], ranges = [128, 32] {tiled = true} : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
    %cst_50 = arith.constant dense<0.000000e+00> : vector<2x8xf16>
    %69 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_50) -> (vector<2x8xf16>) {
      %72 = frisk.fragment "compute" -> (vector<1x4xf16>) {
        %86 = affine.apply #map5(%arg4)
        %87 = affine.apply #map6(%arg4)
        %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
        %88 = vector.extract %67[%86, %87] : f16 from vector<2x8xf16>
        %89 = vector.insert %88, %cst_53 [0, 0] : f16 into vector<1x4xf16>
        %90 = affine.apply #map7(%arg4)
        %91 = vector.extract %67[%86, %90] : f16 from vector<2x8xf16>
        %92 = vector.insert %91, %89 [0, 1] : f16 into vector<1x4xf16>
        %93 = affine.apply #map8(%arg4)
        %94 = vector.extract %67[%86, %93] : f16 from vector<2x8xf16>
        %95 = vector.insert %94, %92 [0, 2] : f16 into vector<1x4xf16>
        %96 = affine.apply #map9(%arg4)
        %97 = vector.extract %67[%86, %96] : f16 from vector<2x8xf16>
        %98 = vector.insert %97, %95 [0, 3] : f16 into vector<1x4xf16>
        frisk.fragment_yield %98 : vector<1x4xf16>
      }
      %73 = affine.apply #map5(%arg4)
      %74 = affine.apply #map6(%arg4)
      %75 = vector.extract %72[0, 0] : f16 from vector<1x4xf16>
      %76 = vector.insert %75, %arg5 [%73, %74] : f16 into vector<2x8xf16>
      %77 = affine.apply #map7(%arg4)
      %78 = vector.extract %72[0, 1] : f16 from vector<1x4xf16>
      %79 = vector.insert %78, %76 [%73, %77] : f16 into vector<2x8xf16>
      %80 = affine.apply #map8(%arg4)
      %81 = vector.extract %72[0, 2] : f16 from vector<1x4xf16>
      %82 = vector.insert %81, %79 [%73, %80] : f16 into vector<2x8xf16>
      %83 = affine.apply #map9(%arg4)
      %84 = vector.extract %72[0, 3] : f16 from vector<1x4xf16>
      %85 = vector.insert %84, %82 [%73, %83] : f16 into vector<2x8xf16>
      affine.yield %85 : vector<2x8xf16>
    } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "fragment", tiled = true}
    %thread_id_x_51 = gpu.thread_id x
    %70 = affine.apply #map36(%thread_id_x_51)
    %c4_52 = arith.constant 4 : index
    %71 = arith.cmpi ult, %70, %c4_52 : index
    scf.if %71 {
      affine.for %arg4 = 0 to 4 {
        frisk.fragment "store" {
          %72 = affine.apply #map5(%arg4)
          %73 = affine.apply #map6(%arg4)
          %cst_53 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
          %74 = vector.extract %69[%72, %73] : f16 from vector<2x8xf16>
          %75 = vector.insert %74, %cst_53 [0, 0] : f16 into vector<1x4xf16>
          %76 = affine.apply #map7(%arg4)
          %77 = vector.extract %69[%72, %76] : f16 from vector<2x8xf16>
          %78 = vector.insert %77, %75 [0, 1] : f16 into vector<1x4xf16>
          %79 = affine.apply #map8(%arg4)
          %80 = vector.extract %69[%72, %79] : f16 from vector<2x8xf16>
          %81 = vector.insert %80, %78 [0, 2] : f16 into vector<1x4xf16>
          %82 = affine.apply #map9(%arg4)
          %83 = vector.extract %69[%72, %82] : f16 from vector<2x8xf16>
          %84 = vector.insert %83, %81 [0, 3] : f16 into vector<1x4xf16>
          %thread_id_x_54 = gpu.thread_id x
          %85 = affine.apply #map5(%arg4)
          %86 = affine.apply #map6(%arg4)
          %87 = vector.extract %84[0, 0] : f16 from vector<1x4xf16>
          %c0 = arith.constant 0 : index
          %c0_55 = arith.constant 0 : index
          %88 = affine.apply #map37(%86)
          %c0_56 = arith.constant 0 : index
          %89 = affine.apply #map38(%86)
          %c0_57 = arith.constant 0 : index
          %90 = affine.apply #map39(%c0_55, %89)
          %91 = affine.apply #map40(%thread_id_x_54, %85, %88, %c0, %c0_56, %c0_57, %90)
          %92 = affine.apply #map41(%thread_id_x_54, %85, %88, %c0, %c0_56, %c0_57, %90)
          affine.store %87, %68[%91, %92] {tiled = true} : memref<128x32xf16, 1>
          %93 = affine.apply #map7(%arg4)
          %94 = vector.extract %84[0, 1] : f16 from vector<1x4xf16>
          %c0_58 = arith.constant 0 : index
          %c0_59 = arith.constant 0 : index
          %95 = affine.apply #map37(%93)
          %c0_60 = arith.constant 0 : index
          %96 = affine.apply #map38(%93)
          %c0_61 = arith.constant 0 : index
          %97 = affine.apply #map39(%c0_59, %96)
          %98 = affine.apply #map40(%thread_id_x_54, %85, %95, %c0_58, %c0_60, %c0_61, %97)
          %99 = affine.apply #map41(%thread_id_x_54, %85, %95, %c0_58, %c0_60, %c0_61, %97)
          affine.store %94, %68[%98, %99] {tiled = true} : memref<128x32xf16, 1>
          %100 = affine.apply #map8(%arg4)
          %101 = vector.extract %84[0, 2] : f16 from vector<1x4xf16>
          %c0_62 = arith.constant 0 : index
          %c0_63 = arith.constant 0 : index
          %102 = affine.apply #map37(%100)
          %c0_64 = arith.constant 0 : index
          %103 = affine.apply #map38(%100)
          %c0_65 = arith.constant 0 : index
          %104 = affine.apply #map39(%c0_63, %103)
          %105 = affine.apply #map40(%thread_id_x_54, %85, %102, %c0_62, %c0_64, %c0_65, %104)
          %106 = affine.apply #map41(%thread_id_x_54, %85, %102, %c0_62, %c0_64, %c0_65, %104)
          affine.store %101, %68[%105, %106] {tiled = true} : memref<128x32xf16, 1>
          %107 = affine.apply #map9(%arg4)
          %108 = vector.extract %84[0, 3] : f16 from vector<1x4xf16>
          %c0_66 = arith.constant 0 : index
          %c0_67 = arith.constant 0 : index
          %109 = affine.apply #map37(%107)
          %c0_68 = arith.constant 0 : index
          %110 = affine.apply #map38(%107)
          %c0_69 = arith.constant 0 : index
          %111 = affine.apply #map39(%c0_67, %110)
          %112 = affine.apply #map40(%thread_id_x_54, %85, %109, %c0_66, %c0_68, %c0_69, %111)
          %113 = affine.apply #map41(%thread_id_x_54, %85, %109, %c0_66, %c0_68, %c0_69, %111)
          affine.store %108, %68[%112, %113] {tiled = true} : memref<128x32xf16, 1>
          frisk.fragment_yield
        }
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 8>, iterLabel = "store", tiled = true}
    }
    return
  }
}

