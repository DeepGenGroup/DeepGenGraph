#map = affine_map<(d0) -> (d0 mod 2)>
#map1 = affine_map<()[s0] -> (s0 * 128 + 128)>
#map2 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>
#map3 = affine_map<(d0) -> ((d0 floordiv 8) mod 2)>
#map4 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 1)>
#map5 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 2)>
#map6 = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32 + 3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (((d0 * 16777216 + d1 * 524288 + d2 * 16384 + d3) floordiv 524288) mod 32)>
#map8 = affine_map<(d0, d1, d2, d3) -> ((d0 * 131072 + d1 * 4096 + d2 * 128 + d3 floordiv 128) mod 4096)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 mod 128)>
#map10 = affine_map<(d0, d1, d2, d3) -> (((d0 * 16777216 + d1 * 524288 + d2 * 4096 + d3) floordiv 524288) mod 32)>
#map11 = affine_map<(d0, d1, d2, d3) -> ((d0 * 4096 + d1 * 128 + d2 + d3 floordiv 4096) mod 128)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d3 mod 4096)>
#map13 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16 + d6 floordiv 4)>
#map14 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>
#map15 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8)>
#map16 = affine_map<()[s0] -> ((s0 mod 64) floordiv 4)>
#map17 = affine_map<(d0)[s0] -> (d0 * 32 - (d0 floordiv 4) * 128 + s0 * 8 - (s0 floordiv 4) * 32)>
#map18 = affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>
#map19 = affine_map<(d0)[s0] -> ((d0 * 2 - (d0 floordiv 4) * 8 - ((d0 * 32 - (d0 floordiv 4) * 128 + s0 mod 16 + 16) floordiv 32) * 2 + (s0 mod 16) floordiv 16 + 1) * 4 + ((((s0 mod 64) floordiv 16) mod 4) floordiv 4) * 8)>
#map20 = affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 16 + 1) floordiv 16) * 8 + 1)>
#map21 = affine_map<(d0)[s0] -> ((d0 * 2 - (d0 floordiv 4) * 8 - ((d0 * 32 - (d0 floordiv 4) * 128 + s0 mod 16 + 16) floordiv 32) * 2 + (s0 mod 16) floordiv 16 + 1) * 4 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 1) floordiv 16) * 16 + 1) floordiv 16) * 8 + 1)>
#map22 = affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + 1) floordiv 8) * 8 + 2)>
#map23 = affine_map<(d0)[s0] -> ((d0 * 2 - (d0 floordiv 4) * 8 - ((d0 * 32 - (d0 floordiv 4) * 128 + s0 mod 16 + 16) floordiv 32) * 2 + (s0 mod 16) floordiv 16 + 1) * 4 + ((((s0 mod 64) floordiv 16) * 2 - ((((s0 mod 64) floordiv 16) * 2 + 1) floordiv 8) * 8 + 1) floordiv 8) * 8 + 2)>
#map24 = affine_map<()[s0] -> ((((s0 mod 16) mod 32) floordiv 16) * 4 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 16 + 3) floordiv 16) * 8 + 3)>
#map25 = affine_map<(d0)[s0] -> ((d0 * 2 - (d0 floordiv 4) * 8 - ((d0 * 32 - (d0 floordiv 4) * 128 + s0 mod 16 + 16) floordiv 32) * 2 + (s0 mod 16) floordiv 16 + 1) * 4 + ((((s0 mod 64) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16) * 4 + 3) floordiv 16) * 16 + 3) floordiv 16) * 8 + 3)>
#map26 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 4) * 8 + 1)>
#map27 = affine_map<(d0) -> (d0 * 4)>
#map28 = affine_map<(d0) -> (d0 * 4 + 1)>
#map29 = affine_map<(d0) -> (d0 * 4 + 2)>
#map30 = affine_map<(d0) -> (d0 * 4 + 3)>
#map31 = affine_map<(d0) -> (d0 mod 8)>
#map32 = affine_map<(d0) -> (d0 floordiv 4)>
#map33 = affine_map<(d0) -> (d0 mod 4)>
#map34 = affine_map<(d0, d1) -> (d0 + d1)>
#map35 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 64 + ((d0 floordiv 64) mod 4) * 16 + d3 * 16 + d0 mod 16)>
#map36 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + (((d0 floordiv 64) mod 4) floordiv 4) * 16 + d4 * 16 + d5 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6)>
#map37 = affine_map<(d0, d1, d2, d3) -> (((d0 * 16777216 + d1 * 524288 + d2 * 128 + d3) floordiv 524288) mod 32)>
#map38 = affine_map<(d0, d1, d2, d3) -> ((d0 * 131072 + d1 * 4096 + d2 + d3 floordiv 128) mod 4096)>
#map39 = affine_map<(d0, d1) -> (d0 + d1 * 4)>
#map40 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 * 16 + d3 * 16 + ((d0 mod 64) floordiv 16) * 4 + d6 mod 4)>
#map41 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2 * 16 + d4 * 16 + d5 * 16 + d0 mod 16 + d6 floordiv 4)>
#map42 = affine_map<(d0, d1, d2, d3) -> (((d0 * 16777216 + d1 * 524288 + d2 * 128 + d3 + 16) floordiv 524288) mod 32)>
#map43 = affine_map<(d0, d1, d2, d3) -> ((d0 * 131072 + d1 * 4096 + d2 + (d3 + 16) floordiv 128) mod 4096)>
#map44 = affine_map<(d0, d1, d2, d3) -> ((d3 + 16) mod 128)>
#map45 = affine_map<(d0) -> (d0 floordiv 64)>
#map46 = affine_map<(d0)[s0, s1, s2] -> (((d0 + s0 * 16777216 + s1 * 524288 + s2 * 16384) floordiv 524288) mod 32)>
#map47 = affine_map<(d0, d1)[s0, s1, s2, s3] -> (d1 * 64 + s0 * 131072 + s1 * 4096 + s2 * 128 + s3 + d0 floordiv 128 - ((d0 floordiv 128 + s0 * 131072 + s1 * 4096 + s2 * 128) floordiv 4096) * 4096 - (d1 floordiv 2) * 128 + (s3 floordiv 64) * 16 - ((s3 floordiv 64) floordiv 4) * 64 - (s3 floordiv 16) * 16)>
#map48 = affine_map<(d0)[s0] -> (d0 mod 128 + (((s0 floordiv 64) mod 4) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 4)>
#map49 = affine_map<(d0)[s0] -> (d0 mod 128 + (((s0 floordiv 64) mod 4) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>
#map50 = affine_map<(d0)[s0] -> (d0 mod 128 + (((s0 floordiv 64) mod 4) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>
#map51 = affine_map<(d0)[s0] -> (d0 mod 128 + (((s0 floordiv 64) mod 4) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>
#map52 = affine_map<(d0)[s0, s1, s2] -> (((d0 + s0 * 16777216 + s1 * 524288 + s2 * 16384 + 16) floordiv 524288) mod 32)>
#map53 = affine_map<(d0, d1)[s0, s1, s2, s3] -> (d1 * 64 + s0 * 131072 + s1 * 4096 + s2 * 128 + s3 + (d0 + 16) floordiv 128 - (((d0 + 16) floordiv 128 + s0 * 131072 + s1 * 4096 + s2 * 128) floordiv 4096) * 4096 - (d1 floordiv 2) * 128 + (s3 floordiv 64) * 16 - ((s3 floordiv 64) floordiv 4) * 64 - (s3 floordiv 16) * 16)>
#map54 = affine_map<(d0)[s0] -> (d0 - ((d0 + 16) floordiv 128) * 128 + (((s0 floordiv 64) mod 4) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 4 + 16)>
#map55 = affine_map<(d0)[s0] -> (d0 - ((d0 + 16) floordiv 128) * 128 + (((s0 floordiv 64) mod 4) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 4 + 17)>
#map56 = affine_map<(d0)[s0] -> (d0 - ((d0 + 16) floordiv 128) * 128 + (((s0 floordiv 64) mod 4) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 4 + 18)>
#map57 = affine_map<(d0)[s0] -> (d0 - ((d0 + 16) floordiv 128) * 128 + (((s0 floordiv 64) mod 4) floordiv 4) * 16 + ((s0 mod 64) floordiv 16) * 4 + 19)>
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
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant dense<0.000000e+00> : vector<2x4xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<2x32xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<1x1xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : vector<4x1xf16>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<4x2xf16>
    %cst_7 = arith.constant dense<0.000000e+00> : vector<4x8xf16>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<1x4xf16>
    %c0 = arith.constant 0 : index
    %cst_9 = arith.constant dense<0.000000e+00> : vector<2x4xf16>
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst_10 = arith.constant 0.000000e+00 : f32
    %cst_11 = arith.constant 0xFF800000 : f32
    %cst_12 = arith.constant 0.127531052 : f32
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_z = gpu.block_id z {range = 1 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %block_id_x = gpu.block_id x {range = 32 : index}
    %0 = arith.truncf %cst_12 : f32 to f16
    %1 = arith.muli %block_id_y, %c128 : index
    affine.for %arg4 = 0 to 128 step 32 {
      %2 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst) -> (vector<2x4xf32>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "load" -> (vector<1x4xf32>) {
          frisk.fragment_yield %cst_4 : vector<1x4xf32>
        }
        %32 = vector.extract %31[0, 0] : f32 from vector<1x4xf32>
        %33 = vector.insert %32, %arg6 [%30, 0] : f32 into vector<2x4xf32>
        %34 = vector.extract %31[0, 1] : f32 from vector<1x4xf32>
        %35 = vector.insert %34, %33 [%30, 1] : f32 into vector<2x4xf32>
        %36 = vector.extract %31[0, 2] : f32 from vector<1x4xf32>
        %37 = vector.insert %36, %35 [%30, 2] : f32 into vector<2x4xf32>
        %38 = vector.extract %31[0, 3] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %37 [%30, 3] : f32 into vector<2x4xf32>
        affine.yield %39 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "load_fragment", tiled = true}
      %3 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_2) -> (vector<2x1xf32>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "load" -> (vector<1x1xf32>) {
          frisk.fragment_yield %cst_1 : vector<1x1xf32>
        }
        %32 = vector.extract %31[0, 0] : f32 from vector<1x1xf32>
        %33 = vector.insert %32, %arg6 [%30, 0] : f32 into vector<2x1xf32>
        affine.yield %33 : vector<2x1xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "load_fragment", tiled = true}
      %4 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst) -> (vector<2x4xf32>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "load" -> (vector<1x4xf32>) {
          frisk.fragment_yield %cst_4 : vector<1x4xf32>
        }
        %32 = vector.extract %31[0, 0] : f32 from vector<1x4xf32>
        %33 = vector.insert %32, %arg6 [%30, 0] : f32 into vector<2x4xf32>
        %34 = vector.extract %31[0, 1] : f32 from vector<1x4xf32>
        %35 = vector.insert %34, %33 [%30, 1] : f32 into vector<2x4xf32>
        %36 = vector.extract %31[0, 2] : f32 from vector<1x4xf32>
        %37 = vector.insert %36, %35 [%30, 2] : f32 into vector<2x4xf32>
        %38 = vector.extract %31[0, 3] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %37 [%30, 3] : f32 into vector<2x4xf32>
        affine.yield %39 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "load_fragment", tiled = true}
      %5:3 = affine.for %arg5 = 0 to #map1()[%block_id_y] step 128 iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4) -> (vector<2x4xf32>, vector<2x1xf32>, vector<2x4xf32>) {
        %30 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_5) -> (vector<2x32xf32>) {
          %56 = affine.apply #map2(%arg9)
          %57 = affine.apply #map3(%arg9)
          %58 = frisk.fragment "load" -> (vector<1x4xf32>) {
            frisk.fragment_yield %cst_4 : vector<1x4xf32>
          }
          %59 = vector.extract %58[0, 0] : f32 from vector<1x4xf32>
          %60 = vector.insert %59, %arg10 [%57, %56] : f32 into vector<2x32xf32>
          %61 = affine.apply #map4(%arg9)
          %62 = vector.extract %58[0, 1] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %60 [%57, %61] : f32 into vector<2x32xf32>
          %64 = affine.apply #map5(%arg9)
          %65 = vector.extract %58[0, 2] : f32 from vector<1x4xf32>
          %66 = vector.insert %65, %63 [%57, %64] : f32 into vector<2x32xf32>
          %67 = affine.apply #map6(%arg9)
          %68 = vector.extract %58[0, 3] : f32 from vector<1x4xf32>
          %69 = vector.insert %68, %66 [%57, %67] : f32 into vector<2x32xf32>
          affine.yield %69 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "load_fragment", tiled = true}
        %31 = affine.for %arg9 = 0 to 128 step 16 iter_args(%arg10 = %30) -> (vector<2x32xf32>) {
          %alloc_20 = memref.alloc() alignment = 16 {frisk.shared_pack = 4 : i64, frisk.shared_pack_axis = 1 : i64, tiled = true} : memref<128x16xf16, 3>
          %alloc_21 = memref.alloc() alignment = 16 {frisk.shared_pack = 8 : i64, frisk.shared_pack_axis = 1 : i64, frisk.shared_packet_rows = 2 : i64, tiled = true} : memref<16x128xf16, 3>
          %56 = affine.apply #map7(%block_id_z, %block_id_x, %block_id_y, %arg9)
          %57 = affine.apply #map8(%block_id_z, %block_id_x, %block_id_y, %arg9)
          %58 = affine.apply #map9(%block_id_z, %block_id_x, %block_id_y, %arg9)
          %subview_22 = memref.subview %arg0[%c0, %56, %57, %58] [1, 1, 128, 16] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x16xf16, strided<[128, 1], offset: ?>, 1>
          frisk.async_copy_gm_to_shm %subview_22 to %alloc_20 {alignment = 16 : i64} : memref<128x16xf16, strided<[128, 1], offset: ?>, 1>, memref<128x16xf16, 3>
          frisk.wait_buffer_ready %alloc_20 : memref<128x16xf16, 3>
          %59 = affine.apply #map10(%block_id_z, %block_id_x, %arg9, %arg5)
          %60 = affine.apply #map11(%block_id_z, %block_id_x, %arg9, %arg5)
          %61 = affine.apply #map12(%block_id_z, %block_id_x, %arg9, %arg5)
          %subview_23 = memref.subview %transpose[%c0, %59, %60, %61] [1, 1, 16, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<16x128xf16, strided<[1, 128], offset: ?>, 1>
          frisk.async_copy_gm_to_shm %subview_23 to %alloc_21 {alignment = 16 : i64} : memref<16x128xf16, strided<[1, 128], offset: ?>, 1>, memref<16x128xf16, 3>
          %thread_id_x_24 = gpu.thread_id x
          %62 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x4xf16>) {
            %68 = affine.apply #map(%arg11)
            %69 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %78 = affine.apply #map13(%thread_id_x_24, %68, %c0, %c0, %c0, %c0, %c0)
              %79 = affine.apply #map14(%thread_id_x_24, %68, %c0, %c0, %c0, %c0, %c0)
              %80 = memref.load %alloc_20[%78, %79] : memref<128x16xf16, 3>
              %81 = vector.insert %80, %cst_8 [0, 0] : f16 into vector<1x4xf16>
              %82 = affine.apply #map13(%thread_id_x_24, %68, %c0, %c0, %c0, %c0, %c1)
              %83 = affine.apply #map14(%thread_id_x_24, %68, %c0, %c0, %c0, %c0, %c1)
              %84 = memref.load %alloc_20[%82, %83] : memref<128x16xf16, 3>
              %85 = vector.insert %84, %81 [0, 1] : f16 into vector<1x4xf16>
              %86 = affine.apply #map13(%thread_id_x_24, %68, %c0, %c0, %c0, %c0, %c2)
              %87 = affine.apply #map14(%thread_id_x_24, %68, %c0, %c0, %c0, %c0, %c2)
              %88 = memref.load %alloc_20[%86, %87] : memref<128x16xf16, 3>
              %89 = vector.insert %88, %85 [0, 2] : f16 into vector<1x4xf16>
              %90 = affine.apply #map13(%thread_id_x_24, %68, %c0, %c0, %c0, %c0, %c3)
              %91 = affine.apply #map14(%thread_id_x_24, %68, %c0, %c0, %c0, %c0, %c3)
              %92 = memref.load %alloc_20[%90, %91] : memref<128x16xf16, 3>
              %93 = vector.insert %92, %89 [0, 3] : f16 into vector<1x4xf16>
              frisk.fragment_yield %93 : vector<1x4xf16>
            }
            %70 = vector.extract %69[0, 0] : f16 from vector<1x4xf16>
            %71 = vector.insert %70, %arg12 [%68, 0] : f16 into vector<2x4xf16>
            %72 = vector.extract %69[0, 1] : f16 from vector<1x4xf16>
            %73 = vector.insert %72, %71 [%68, 1] : f16 into vector<2x4xf16>
            %74 = vector.extract %69[0, 2] : f16 from vector<1x4xf16>
            %75 = vector.insert %74, %73 [%68, 2] : f16 into vector<2x4xf16>
            %76 = vector.extract %69[0, 3] : f16 from vector<1x4xf16>
            %77 = vector.insert %76, %75 [%68, 3] : f16 into vector<2x4xf16>
            affine.yield %77 : vector<2x4xf16>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "load_fragment", tiled = true}
          %63 = affine.for %arg11 = 0 to 2 iter_args(%arg12 = %cst_9) -> (vector<2x4xf16>) {
            %68 = affine.apply #map(%arg11)
            %69 = frisk.fragment "compute" -> (vector<1x4xf16>) {
              %78 = vector.extract %62[%68, 0] : f16 from vector<2x4xf16>
              %79 = vector.insert %78, %cst_8 [0, 0] : f16 into vector<1x4xf16>
              %80 = vector.extract %62[%68, 1] : f16 from vector<2x4xf16>
              %81 = vector.insert %80, %79 [0, 1] : f16 into vector<1x4xf16>
              %82 = vector.extract %62[%68, 2] : f16 from vector<2x4xf16>
              %83 = vector.insert %82, %81 [0, 2] : f16 into vector<1x4xf16>
              %84 = vector.extract %62[%68, 3] : f16 from vector<2x4xf16>
              %85 = vector.insert %84, %83 [0, 3] : f16 into vector<1x4xf16>
              %86 = vector.insert %0, %cst_8 [0, 0] : f16 into vector<1x4xf16>
              %87 = vector.insert %0, %86 [0, 1] : f16 into vector<1x4xf16>
              %88 = vector.insert %0, %87 [0, 2] : f16 into vector<1x4xf16>
              %89 = vector.insert %0, %88 [0, 3] : f16 into vector<1x4xf16>
              %90 = arith.mulf %85, %89 {origin = "frisk.mul", tiled = true} : vector<1x4xf16>
              frisk.fragment_yield %90 : vector<1x4xf16>
            }
            %70 = vector.extract %69[0, 0] : f16 from vector<1x4xf16>
            %71 = vector.insert %70, %arg12 [%68, 0] : f16 into vector<2x4xf16>
            %72 = vector.extract %69[0, 1] : f16 from vector<1x4xf16>
            %73 = vector.insert %72, %71 [%68, 1] : f16 into vector<2x4xf16>
            %74 = vector.extract %69[0, 2] : f16 from vector<1x4xf16>
            %75 = vector.insert %74, %73 [%68, 2] : f16 into vector<2x4xf16>
            %76 = vector.extract %69[0, 3] : f16 from vector<1x4xf16>
            %77 = vector.insert %76, %75 [%68, 3] : f16 into vector<2x4xf16>
            affine.yield %77 : vector<2x4xf16>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
          frisk.wait_buffer_ready %alloc_21 : memref<16x128xf16, 3>
          %thread_id_x_25 = gpu.thread_id x
          %64 = affine.for %arg11 = 0 to 4 iter_args(%arg12 = %cst_7) -> (vector<4x8xf16>) {
            %68 = affine.apply #map15(%arg11)
            %69 = frisk.fragment "load" -> (vector<4x2xf16>) {
              %90 = affine.apply #map16()[%thread_id_x_25]
              %91 = affine.apply #map17(%arg11)[%thread_id_x_25]
              %92 = frisk.read_matrix_from_shm %alloc_21[%90, %91] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64, tiled = true} : memref<16x128xf16, 3> -> vector<4xi32>
              %93 = vector.bitcast %92 : vector<4xi32> to vector<8xf16>
              %94 = affine.apply #map18()[%thread_id_x_25]
              %95 = vector.extract %93[%94] : f16 from vector<8xf16>
              %96 = vector.insert %95, %cst_6 [0, 0] : f16 into vector<4x2xf16>
              %97 = affine.apply #map19(%arg11)[%thread_id_x_25]
              %98 = vector.extract %93[%97] : f16 from vector<8xf16>
              %99 = vector.insert %98, %96 [0, 1] : f16 into vector<4x2xf16>
              %100 = affine.apply #map20()[%thread_id_x_25]
              %101 = vector.extract %93[%100] : f16 from vector<8xf16>
              %102 = vector.insert %101, %99 [1, 0] : f16 into vector<4x2xf16>
              %103 = affine.apply #map21(%arg11)[%thread_id_x_25]
              %104 = vector.extract %93[%103] : f16 from vector<8xf16>
              %105 = vector.insert %104, %102 [1, 1] : f16 into vector<4x2xf16>
              %106 = affine.apply #map22()[%thread_id_x_25]
              %107 = vector.extract %93[%106] : f16 from vector<8xf16>
              %108 = vector.insert %107, %105 [2, 0] : f16 into vector<4x2xf16>
              %109 = affine.apply #map23(%arg11)[%thread_id_x_25]
              %110 = vector.extract %93[%109] : f16 from vector<8xf16>
              %111 = vector.insert %110, %108 [2, 1] : f16 into vector<4x2xf16>
              %112 = affine.apply #map24()[%thread_id_x_25]
              %113 = vector.extract %93[%112] : f16 from vector<8xf16>
              %114 = vector.insert %113, %111 [3, 0] : f16 into vector<4x2xf16>
              %115 = affine.apply #map25(%arg11)[%thread_id_x_25]
              %116 = vector.extract %93[%115] : f16 from vector<8xf16>
              %117 = vector.insert %116, %114 [3, 1] : f16 into vector<4x2xf16>
              frisk.fragment_yield %117 : vector<4x2xf16>
            }
            %70 = vector.extract %69[0, 0] : f16 from vector<4x2xf16>
            %71 = vector.insert %70, %arg12 [0, %68] : f16 into vector<4x8xf16>
            %72 = affine.apply #map26(%arg11)
            %73 = vector.extract %69[0, 1] : f16 from vector<4x2xf16>
            %74 = vector.insert %73, %71 [0, %72] : f16 into vector<4x8xf16>
            %75 = vector.extract %69[1, 0] : f16 from vector<4x2xf16>
            %76 = vector.insert %75, %74 [1, %68] : f16 into vector<4x8xf16>
            %77 = affine.apply #map26(%arg11)
            %78 = vector.extract %69[1, 1] : f16 from vector<4x2xf16>
            %79 = vector.insert %78, %76 [1, %77] : f16 into vector<4x8xf16>
            %80 = vector.extract %69[2, 0] : f16 from vector<4x2xf16>
            %81 = vector.insert %80, %79 [2, %68] : f16 into vector<4x8xf16>
            %82 = affine.apply #map26(%arg11)
            %83 = vector.extract %69[2, 1] : f16 from vector<4x2xf16>
            %84 = vector.insert %83, %81 [2, %82] : f16 into vector<4x8xf16>
            %85 = vector.extract %69[3, 0] : f16 from vector<4x2xf16>
            %86 = vector.insert %85, %84 [3, %68] : f16 into vector<4x8xf16>
            %87 = affine.apply #map26(%arg11)
            %88 = vector.extract %69[3, 1] : f16 from vector<4x2xf16>
            %89 = vector.insert %88, %86 [3, %87] : f16 into vector<4x8xf16>
            affine.yield %89 : vector<4x8xf16>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4, 2>, frisk.tile_shape = array<i64: 4, 8>, iterLabel = "matrix_load", tiled = true}
          %65 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
            %68 = affine.apply #map2(%arg11)
            %69 = affine.apply #map3(%arg11)
            %70 = affine.for %arg13 = 0 to 1 iter_args(%arg14 = %cst_4) -> (vector<1x4xf32>) {
              %82 = frisk.fragment "load" -> (vector<1x4xf16>) {
                %85 = affine.apply #map3(%arg11)
                %86 = affine.apply #map27(%arg13)
                %87 = vector.extract %63[%85, %86] : f16 from vector<2x4xf16>
                %88 = vector.insert %87, %cst_8 [0, 0] : f16 into vector<1x4xf16>
                %89 = affine.apply #map28(%arg13)
                %90 = vector.extract %63[%85, %89] : f16 from vector<2x4xf16>
                %91 = vector.insert %90, %88 [0, 1] : f16 into vector<1x4xf16>
                %92 = affine.apply #map29(%arg13)
                %93 = vector.extract %63[%85, %92] : f16 from vector<2x4xf16>
                %94 = vector.insert %93, %91 [0, 2] : f16 into vector<1x4xf16>
                %95 = affine.apply #map30(%arg13)
                %96 = vector.extract %63[%85, %95] : f16 from vector<2x4xf16>
                %97 = vector.insert %96, %94 [0, 3] : f16 into vector<1x4xf16>
                frisk.fragment_yield %97 : vector<1x4xf16>
              }
              %83 = frisk.fragment "load" -> (vector<4x1xf16>) {
                %85 = affine.apply #map27(%arg13)
                %86 = affine.apply #map31(%arg11)
                %87 = vector.extract %64[%85, %86] : f16 from vector<4x8xf16>
                %88 = vector.insert %87, %cst_3 [0, 0] : f16 into vector<4x1xf16>
                %89 = affine.apply #map28(%arg13)
                %90 = vector.extract %64[%89, %86] : f16 from vector<4x8xf16>
                %91 = vector.insert %90, %88 [1, 0] : f16 into vector<4x1xf16>
                %92 = affine.apply #map29(%arg13)
                %93 = vector.extract %64[%92, %86] : f16 from vector<4x8xf16>
                %94 = vector.insert %93, %91 [2, 0] : f16 into vector<4x1xf16>
                %95 = affine.apply #map30(%arg13)
                %96 = vector.extract %64[%95, %86] : f16 from vector<4x8xf16>
                %97 = vector.insert %96, %94 [3, 0] : f16 into vector<4x1xf16>
                frisk.fragment_yield %97 : vector<4x1xf16>
              }
              %84 = frisk.fragment "compute" -> (vector<1x4xf32>) {
                %85 = frisk.warp_mma_rr %82, %83, %arg14 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
                frisk.fragment_yield %85 : vector<1x4xf32>
              }
              affine.yield %84 : vector<1x4xf32>
            } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
            %71 = vector.extract %70[0, 0] : f32 from vector<1x4xf32>
            %72 = vector.insert %71, %arg12 [%69, %68] : f32 into vector<2x32xf32>
            %73 = affine.apply #map4(%arg11)
            %74 = vector.extract %70[0, 1] : f32 from vector<1x4xf32>
            %75 = vector.insert %74, %72 [%69, %73] : f32 into vector<2x32xf32>
            %76 = affine.apply #map5(%arg11)
            %77 = vector.extract %70[0, 2] : f32 from vector<1x4xf32>
            %78 = vector.insert %77, %75 [%69, %76] : f32 into vector<2x32xf32>
            %79 = affine.apply #map6(%arg11)
            %80 = vector.extract %70[0, 3] : f32 from vector<1x4xf32>
            %81 = vector.insert %80, %78 [%69, %79] : f32 into vector<2x32xf32>
            affine.yield %81 : vector<2x32xf32>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "gemm_mn", tiled = true}
          %66 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
            %68 = affine.apply #map2(%arg11)
            %69 = affine.apply #map3(%arg11)
            %70 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %82 = vector.extract %arg10[%69, %68] : f32 from vector<2x32xf32>
              %83 = vector.insert %82, %cst_4 [0, 0] : f32 into vector<1x4xf32>
              %84 = affine.apply #map4(%arg11)
              %85 = vector.extract %arg10[%69, %84] : f32 from vector<2x32xf32>
              %86 = vector.insert %85, %83 [0, 1] : f32 into vector<1x4xf32>
              %87 = affine.apply #map5(%arg11)
              %88 = vector.extract %arg10[%69, %87] : f32 from vector<2x32xf32>
              %89 = vector.insert %88, %86 [0, 2] : f32 into vector<1x4xf32>
              %90 = affine.apply #map6(%arg11)
              %91 = vector.extract %arg10[%69, %90] : f32 from vector<2x32xf32>
              %92 = vector.insert %91, %89 [0, 3] : f32 into vector<1x4xf32>
              %93 = vector.extract %65[%69, %68] : f32 from vector<2x32xf32>
              %94 = vector.insert %93, %cst_4 [0, 0] : f32 into vector<1x4xf32>
              %95 = affine.apply #map4(%arg11)
              %96 = vector.extract %65[%69, %95] : f32 from vector<2x32xf32>
              %97 = vector.insert %96, %94 [0, 1] : f32 into vector<1x4xf32>
              %98 = affine.apply #map5(%arg11)
              %99 = vector.extract %65[%69, %98] : f32 from vector<2x32xf32>
              %100 = vector.insert %99, %97 [0, 2] : f32 into vector<1x4xf32>
              %101 = affine.apply #map6(%arg11)
              %102 = vector.extract %65[%69, %101] : f32 from vector<2x32xf32>
              %103 = vector.insert %102, %100 [0, 3] : f32 into vector<1x4xf32>
              %104 = arith.addf %92, %103 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
              frisk.fragment_yield %104 : vector<1x4xf32>
            }
            %71 = vector.extract %70[0, 0] : f32 from vector<1x4xf32>
            %72 = vector.insert %71, %arg12 [%69, %68] : f32 into vector<2x32xf32>
            %73 = affine.apply #map4(%arg11)
            %74 = vector.extract %70[0, 1] : f32 from vector<1x4xf32>
            %75 = vector.insert %74, %72 [%69, %73] : f32 into vector<2x32xf32>
            %76 = affine.apply #map5(%arg11)
            %77 = vector.extract %70[0, 2] : f32 from vector<1x4xf32>
            %78 = vector.insert %77, %75 [%69, %76] : f32 into vector<2x32xf32>
            %79 = affine.apply #map6(%arg11)
            %80 = vector.extract %70[0, 3] : f32 from vector<1x4xf32>
            %81 = vector.insert %80, %78 [%69, %79] : f32 into vector<2x32xf32>
            affine.yield %81 : vector<2x32xf32>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
          %67 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %cst_5) -> (vector<2x32xf32>) {
            %68 = affine.apply #map2(%arg11)
            %69 = affine.apply #map3(%arg11)
            %70 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %82 = vector.extract %66[%69, %68] : f32 from vector<2x32xf32>
              %83 = vector.insert %82, %cst_4 [0, 0] : f32 into vector<1x4xf32>
              %84 = affine.apply #map4(%arg11)
              %85 = vector.extract %66[%69, %84] : f32 from vector<2x32xf32>
              %86 = vector.insert %85, %83 [0, 1] : f32 into vector<1x4xf32>
              %87 = affine.apply #map5(%arg11)
              %88 = vector.extract %66[%69, %87] : f32 from vector<2x32xf32>
              %89 = vector.insert %88, %86 [0, 2] : f32 into vector<1x4xf32>
              %90 = affine.apply #map6(%arg11)
              %91 = vector.extract %66[%69, %90] : f32 from vector<2x32xf32>
              %92 = vector.insert %91, %89 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
              frisk.fragment_yield %92 : vector<1x4xf32>
            }
            %71 = vector.extract %70[0, 0] : f32 from vector<1x4xf32>
            %72 = vector.insert %71, %arg12 [%69, %68] : f32 into vector<2x32xf32>
            %73 = affine.apply #map4(%arg11)
            %74 = vector.extract %70[0, 1] : f32 from vector<1x4xf32>
            %75 = vector.insert %74, %72 [%69, %73] : f32 into vector<2x32xf32>
            %76 = affine.apply #map5(%arg11)
            %77 = vector.extract %70[0, 2] : f32 from vector<1x4xf32>
            %78 = vector.insert %77, %75 [%69, %76] : f32 into vector<2x32xf32>
            %79 = affine.apply #map6(%arg11)
            %80 = vector.extract %70[0, 3] : f32 from vector<1x4xf32>
            %81 = vector.insert %80, %78 [%69, %79] : f32 into vector<2x32xf32>
            affine.yield %81 : vector<2x32xf32>
          } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
          affine.yield %67 : vector<2x32xf32>
        } {local_yield_transformed = true}
        %thread_id_x_16 = gpu.thread_id x
        %32 = arith.index_cast %thread_id_x_16 : index to i32
        %33 = arith.andi %32, %c16_i32 : i32
        %34 = arith.cmpi ne, %33, %c0_i32 : i32
        %35 = arith.andi %32, %c32_i32 : i32
        %36 = arith.cmpi ne, %35, %c0_i32 : i32
        %37 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_5) -> (vector<2x32xf32>) {
          %56 = affine.apply #map2(%arg9)
          %57 = affine.apply #map3(%arg9)
          %58 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %70 = vector.extract %31[%57, %56] : f32 from vector<2x32xf32>
            %71 = vector.insert %70, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %72 = affine.apply #map4(%arg9)
            %73 = vector.extract %31[%57, %72] : f32 from vector<2x32xf32>
            %74 = vector.insert %73, %71 [0, 1] : f32 into vector<1x4xf32>
            %75 = affine.apply #map5(%arg9)
            %76 = vector.extract %31[%57, %75] : f32 from vector<2x32xf32>
            %77 = vector.insert %76, %74 [0, 2] : f32 into vector<1x4xf32>
            %78 = affine.apply #map6(%arg9)
            %79 = vector.extract %31[%57, %78] : f32 from vector<2x32xf32>
            %80 = vector.insert %79, %77 [0, 3] : f32 into vector<1x4xf32>
            %81 = arith.select %34, %70, %73 : f32
            %shuffleResult, %valid = gpu.shuffle xor %81, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
            %82 = arith.select %34, %shuffleResult, %70 : f32
            %83 = arith.select %34, %73, %shuffleResult : f32
            %84 = arith.select %34, %76, %79 : f32
            %shuffleResult_20, %valid_21 = gpu.shuffle xor %84, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
            %85 = arith.select %34, %shuffleResult_20, %76 : f32
            %86 = arith.select %34, %79, %shuffleResult_20 : f32
            %87 = arith.select %36, %82, %85 : f32
            %shuffleResult_22, %valid_23 = gpu.shuffle xor %87, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
            %88 = arith.select %36, %shuffleResult_22, %82 : f32
            %89 = arith.select %36, %85, %shuffleResult_22 : f32
            %90 = arith.select %36, %83, %86 : f32
            %shuffleResult_24, %valid_25 = gpu.shuffle xor %90, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
            %91 = arith.select %36, %shuffleResult_24, %83 : f32
            %92 = arith.select %36, %86, %shuffleResult_24 : f32
            %93 = vector.insert %88, %80 [0, 0] : f32 into vector<1x4xf32>
            %94 = vector.insert %91, %93 [0, 1] : f32 into vector<1x4xf32>
            %95 = vector.insert %89, %94 [0, 2] : f32 into vector<1x4xf32>
            %96 = vector.insert %92, %95 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
            frisk.fragment_yield %96 : vector<1x4xf32>
          }
          %59 = vector.extract %58[0, 0] : f32 from vector<1x4xf32>
          %60 = vector.insert %59, %arg10 [%57, %56] : f32 into vector<2x32xf32>
          %61 = affine.apply #map4(%arg9)
          %62 = vector.extract %58[0, 1] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %60 [%57, %61] : f32 into vector<2x32xf32>
          %64 = affine.apply #map5(%arg9)
          %65 = vector.extract %58[0, 2] : f32 from vector<1x4xf32>
          %66 = vector.insert %65, %63 [%57, %64] : f32 into vector<2x32xf32>
          %67 = affine.apply #map6(%arg9)
          %68 = vector.extract %58[0, 3] : f32 from vector<1x4xf32>
          %69 = vector.insert %68, %66 [%57, %67] : f32 into vector<2x32xf32>
          affine.yield %69 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %thread_id_x_17 = gpu.thread_id x
        %38 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_5) -> (vector<2x32xf32>) {
          %56 = affine.apply #map2(%arg9)
          %57 = affine.apply #map3(%arg9)
          %58 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %70 = affine.apply #map32(%56)
            %71 = affine.apply #map33(%56)
            %72 = affine.apply #map33(%71)
            %73 = affine.apply #map34(%c0, %72)
            %74 = affine.apply #map35(%thread_id_x_17, %57, %70, %c0, %c0, %c0, %73)
            %75 = affine.apply #map36(%thread_id_x_17, %57, %70, %c0, %c0, %c0, %73)
            %76 = affine.apply #map34(%1, %74)
            %77 = affine.apply #map34(%arg5, %75)
            %78 = arith.addi %76, %c1 : index
            %79 = arith.cmpi ule, %78, %77 : index
            %80 = arith.select %79, %cst_11, %cst_10 : f32
            %81 = vector.extract %37[%57, %56] : f32 from vector<2x32xf32>
            %82 = arith.addf %81, %80 {frisk.fused_mask_add} : f32
            %83 = vector.insert %82, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %84 = affine.apply #map4(%arg9)
            %85 = affine.apply #map32(%84)
            %86 = affine.apply #map33(%84)
            %87 = affine.apply #map33(%86)
            %88 = affine.apply #map34(%c0, %87)
            %89 = affine.apply #map35(%thread_id_x_17, %57, %85, %c0, %c0, %c0, %88)
            %90 = affine.apply #map36(%thread_id_x_17, %57, %85, %c0, %c0, %c0, %88)
            %91 = affine.apply #map34(%1, %89)
            %92 = affine.apply #map34(%arg5, %90)
            %93 = arith.addi %91, %c1 : index
            %94 = arith.cmpi ule, %93, %92 : index
            %95 = arith.select %94, %cst_11, %cst_10 : f32
            %96 = vector.extract %37[%57, %84] : f32 from vector<2x32xf32>
            %97 = arith.addf %96, %95 {frisk.fused_mask_add} : f32
            %98 = vector.insert %97, %83 [0, 1] : f32 into vector<1x4xf32>
            %99 = affine.apply #map5(%arg9)
            %100 = affine.apply #map32(%99)
            %101 = affine.apply #map33(%99)
            %102 = affine.apply #map33(%101)
            %103 = affine.apply #map34(%c0, %102)
            %104 = affine.apply #map35(%thread_id_x_17, %57, %100, %c0, %c0, %c0, %103)
            %105 = affine.apply #map36(%thread_id_x_17, %57, %100, %c0, %c0, %c0, %103)
            %106 = affine.apply #map34(%1, %104)
            %107 = affine.apply #map34(%arg5, %105)
            %108 = arith.addi %106, %c1 : index
            %109 = arith.cmpi ule, %108, %107 : index
            %110 = arith.select %109, %cst_11, %cst_10 : f32
            %111 = vector.extract %37[%57, %99] : f32 from vector<2x32xf32>
            %112 = arith.addf %111, %110 {frisk.fused_mask_add} : f32
            %113 = vector.insert %112, %98 [0, 2] : f32 into vector<1x4xf32>
            %114 = affine.apply #map6(%arg9)
            %115 = affine.apply #map32(%114)
            %116 = affine.apply #map33(%114)
            %117 = affine.apply #map33(%116)
            %118 = affine.apply #map34(%c0, %117)
            %119 = affine.apply #map35(%thread_id_x_17, %57, %115, %c0, %c0, %c0, %118)
            %120 = affine.apply #map36(%thread_id_x_17, %57, %115, %c0, %c0, %c0, %118)
            %121 = affine.apply #map34(%1, %119)
            %122 = affine.apply #map34(%arg5, %120)
            %123 = arith.addi %121, %c1 : index
            %124 = arith.cmpi ule, %123, %122 : index
            %125 = arith.select %124, %cst_11, %cst_10 : f32
            %126 = vector.extract %37[%57, %114] : f32 from vector<2x32xf32>
            %127 = arith.addf %126, %125 {frisk.fused_mask_add} : f32
            %128 = vector.insert %127, %113 [0, 3] : f32 into vector<1x4xf32>
            frisk.fragment_yield %128 : vector<1x4xf32>
          }
          %59 = vector.extract %58[0, 0] : f32 from vector<1x4xf32>
          %60 = vector.insert %59, %arg10 [%57, %56] : f32 into vector<2x32xf32>
          %61 = affine.apply #map4(%arg9)
          %62 = vector.extract %58[0, 1] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %60 [%57, %61] : f32 into vector<2x32xf32>
          %64 = affine.apply #map5(%arg9)
          %65 = vector.extract %58[0, 2] : f32 from vector<1x4xf32>
          %66 = vector.insert %65, %63 [%57, %64] : f32 into vector<2x32xf32>
          %67 = affine.apply #map6(%arg9)
          %68 = vector.extract %58[0, 3] : f32 from vector<1x4xf32>
          %69 = vector.insert %68, %66 [%57, %67] : f32 into vector<2x32xf32>
          affine.yield %69 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "mask_fragment", tiled = true}
        %39 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_5) -> (vector<2x32xf32>) {
          %56 = affine.apply #map2(%arg9)
          %57 = affine.apply #map3(%arg9)
          %58 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %70 = vector.extract %38[%57, %56] : f32 from vector<2x32xf32>
            %71 = vector.insert %70, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %72 = affine.apply #map4(%arg9)
            %73 = vector.extract %38[%57, %72] : f32 from vector<2x32xf32>
            %74 = vector.insert %73, %71 [0, 1] : f32 into vector<1x4xf32>
            %75 = affine.apply #map5(%arg9)
            %76 = vector.extract %38[%57, %75] : f32 from vector<2x32xf32>
            %77 = vector.insert %76, %74 [0, 2] : f32 into vector<1x4xf32>
            %78 = affine.apply #map6(%arg9)
            %79 = vector.extract %38[%57, %78] : f32 from vector<2x32xf32>
            %80 = vector.insert %79, %77 [0, 3] : f32 into vector<1x4xf32>
            %81 = math.exp2 %80 {tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %81 : vector<1x4xf32>
          }
          %59 = vector.extract %58[0, 0] : f32 from vector<1x4xf32>
          %60 = vector.insert %59, %arg10 [%57, %56] : f32 into vector<2x32xf32>
          %61 = affine.apply #map4(%arg9)
          %62 = vector.extract %58[0, 1] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %60 [%57, %61] : f32 into vector<2x32xf32>
          %64 = affine.apply #map5(%arg9)
          %65 = vector.extract %58[0, 2] : f32 from vector<1x4xf32>
          %66 = vector.insert %65, %63 [%57, %64] : f32 into vector<2x32xf32>
          %67 = affine.apply #map6(%arg9)
          %68 = vector.extract %58[0, 3] : f32 from vector<1x4xf32>
          %69 = vector.insert %68, %66 [%57, %67] : f32 into vector<2x32xf32>
          affine.yield %69 : vector<2x32xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %40 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_2) -> (vector<2x1xf32>) {
          %56 = affine.apply #map(%arg9)
          %57 = frisk.fragment "compute" -> (vector<1x1xf32>) {
            %60 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_10) -> (f32) {
              %64 = affine.apply #map2(%arg11)
              %65 = frisk.fragment "compute" -> (f32) {
                %66 = vector.extract %39[%56, %64] : f32 from vector<2x32xf32>
                %67 = arith.addf %arg12, %66 : f32
                %68 = affine.apply #map4(%arg11)
                %69 = vector.extract %39[%56, %68] : f32 from vector<2x32xf32>
                %70 = arith.addf %67, %69 : f32
                %71 = affine.apply #map5(%arg11)
                %72 = vector.extract %39[%56, %71] : f32 from vector<2x32xf32>
                %73 = arith.addf %70, %72 : f32
                %74 = affine.apply #map6(%arg11)
                %75 = vector.extract %39[%56, %74] : f32 from vector<2x32xf32>
                %76 = arith.addf %73, %75 : f32
                frisk.fragment_yield %76 : f32
              }
              affine.yield %65 : f32
            } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 4>, frisk.tile_shape = array<i64: 32>, iterLabel = "reduce_axis_fragment", tiled = true}
            %shuffleResult, %valid = gpu.shuffle xor %60, %c16_i32, %c64_i32 : f32
            %61 = arith.addf %60, %shuffleResult : f32
            %shuffleResult_20, %valid_21 = gpu.shuffle xor %61, %c32_i32, %c64_i32 : f32
            %62 = arith.addf %61, %shuffleResult_20 : f32
            %63 = vector.insert %62, %cst_1 [0, 0] : f32 into vector<1x1xf32>
            frisk.fragment_yield %63 : vector<1x1xf32>
          }
          %58 = vector.extract %57[0, 0] : f32 from vector<1x1xf32>
          %59 = vector.insert %58, %arg10 [%56, 0] : f32 into vector<2x1xf32>
          affine.yield %59 : vector<2x1xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "reduce_fragment", tiled = true}
        %41 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_2) -> (vector<2x1xf32>) {
          %56 = affine.apply #map(%arg9)
          %57 = frisk.fragment "compute" -> (vector<1x1xf32>) {
            %60 = vector.extract %40[%56, 0] : f32 from vector<2x1xf32>
            %61 = vector.insert %60, %cst_1 [0, 0] : f32 into vector<1x1xf32>
            %62 = vector.extract %arg7[%56, 0] : f32 from vector<2x1xf32>
            %63 = vector.insert %62, %cst_1 [0, 0] : f32 into vector<1x1xf32>
            %64 = arith.addf %61, %63 {origin = "frisk.add", tiled = true} : vector<1x1xf32>
            frisk.fragment_yield %64 : vector<1x1xf32>
          }
          %58 = vector.extract %57[0, 0] : f32 from vector<1x1xf32>
          %59 = vector.insert %58, %arg10 [%56, 0] : f32 into vector<2x1xf32>
          affine.yield %59 : vector<2x1xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
        %42 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_0) -> (vector<2x32xf16>) {
          %56 = affine.apply #map2(%arg9)
          %57 = affine.apply #map3(%arg9)
          %58 = frisk.fragment "compute" -> (vector<1x4xf16>) {
            %70 = vector.extract %39[%57, %56] : f32 from vector<2x32xf32>
            %71 = vector.insert %70, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %72 = affine.apply #map4(%arg9)
            %73 = vector.extract %39[%57, %72] : f32 from vector<2x32xf32>
            %74 = vector.insert %73, %71 [0, 1] : f32 into vector<1x4xf32>
            %75 = affine.apply #map5(%arg9)
            %76 = vector.extract %39[%57, %75] : f32 from vector<2x32xf32>
            %77 = vector.insert %76, %74 [0, 2] : f32 into vector<1x4xf32>
            %78 = affine.apply #map6(%arg9)
            %79 = vector.extract %39[%57, %78] : f32 from vector<2x32xf32>
            %80 = vector.insert %79, %77 [0, 3] : f32 into vector<1x4xf32>
            %81 = arith.truncf %80 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
            frisk.fragment_yield %81 : vector<1x4xf16>
          }
          %59 = vector.extract %58[0, 0] : f16 from vector<1x4xf16>
          %60 = vector.insert %59, %arg10 [%57, %56] : f16 into vector<2x32xf16>
          %61 = affine.apply #map4(%arg9)
          %62 = vector.extract %58[0, 1] : f16 from vector<1x4xf16>
          %63 = vector.insert %62, %60 [%57, %61] : f16 into vector<2x32xf16>
          %64 = affine.apply #map5(%arg9)
          %65 = vector.extract %58[0, 2] : f16 from vector<1x4xf16>
          %66 = vector.insert %65, %63 [%57, %64] : f16 into vector<2x32xf16>
          %67 = affine.apply #map6(%arg9)
          %68 = vector.extract %58[0, 3] : f16 from vector<1x4xf16>
          %69 = vector.insert %68, %66 [%57, %67] : f16 into vector<2x32xf16>
          affine.yield %69 : vector<2x32xf16>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 32>, iterLabel = "fragment", tiled = true}
        %alloc = memref.alloc() alignment = 16 {frisk.shared_pack = 4 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_pair_stride = 16 : i64, tiled = true} : memref<128x16xf16, 3>
        %43 = affine.apply #map37(%block_id_z, %block_id_x, %arg5, %arg4)
        %44 = affine.apply #map38(%block_id_z, %block_id_x, %arg5, %arg4)
        %45 = affine.apply #map9(%block_id_z, %block_id_x, %arg5, %arg4)
        %subview = memref.subview %arg2[%c0, %43, %44, %45] [1, 1, 128, 16] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x16xf16, strided<[128, 1], offset: ?>, 1>
        frisk.async_copy_gm_to_shm %subview to %alloc {alignment = 16 : i64} : memref<128x16xf16, strided<[128, 1], offset: ?>, 1>, memref<128x16xf16, 3>
        frisk.wait_buffer_ready %alloc : memref<128x16xf16, 3>
        %46 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst) -> (vector<2x4xf32>) {
          %56 = affine.apply #map(%arg9)
          %57 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_4) -> (vector<1x4xf32>) {
            %66 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %69 = affine.apply #map(%arg9)
              %70 = affine.apply #map27(%arg11)
              %71 = vector.extract %42[%69, %70] : f16 from vector<2x32xf16>
              %72 = vector.insert %71, %cst_8 [0, 0] : f16 into vector<1x4xf16>
              %73 = affine.apply #map28(%arg11)
              %74 = vector.extract %42[%69, %73] : f16 from vector<2x32xf16>
              %75 = vector.insert %74, %72 [0, 1] : f16 into vector<1x4xf16>
              %76 = affine.apply #map29(%arg11)
              %77 = vector.extract %42[%69, %76] : f16 from vector<2x32xf16>
              %78 = vector.insert %77, %75 [0, 2] : f16 into vector<1x4xf16>
              %79 = affine.apply #map30(%arg11)
              %80 = vector.extract %42[%69, %79] : f16 from vector<2x32xf16>
              %81 = vector.insert %80, %78 [0, 3] : f16 into vector<1x4xf16>
              frisk.fragment_yield %81 : vector<1x4xf16>
            }
            %67 = frisk.fragment "load" -> (vector<4x1xf16>) {
              %thread_id_x_20 = gpu.thread_id x
              %69 = affine.apply #map27(%arg11)
              %70 = affine.apply #map32(%69)
              %71 = affine.apply #map33(%69)
              %72 = affine.apply #map39(%71, %c0)
              %73 = affine.apply #map40(%thread_id_x_20, %70, %c0, %c0, %c0, %c0, %72)
              %74 = affine.apply #map41(%thread_id_x_20, %70, %c0, %c0, %c0, %c0, %72)
              %75 = memref.load %alloc[%73, %74] : memref<128x16xf16, 3>
              %76 = vector.insert %75, %cst_3 [0, 0] : f16 into vector<4x1xf16>
              %77 = affine.apply #map28(%arg11)
              %78 = affine.apply #map32(%77)
              %79 = affine.apply #map33(%77)
              %80 = affine.apply #map39(%79, %c0)
              %81 = affine.apply #map40(%thread_id_x_20, %78, %c0, %c0, %c0, %c0, %80)
              %82 = affine.apply #map41(%thread_id_x_20, %78, %c0, %c0, %c0, %c0, %80)
              %83 = memref.load %alloc[%81, %82] : memref<128x16xf16, 3>
              %84 = vector.insert %83, %76 [1, 0] : f16 into vector<4x1xf16>
              %85 = affine.apply #map29(%arg11)
              %86 = affine.apply #map32(%85)
              %87 = affine.apply #map33(%85)
              %88 = affine.apply #map39(%87, %c0)
              %89 = affine.apply #map40(%thread_id_x_20, %86, %c0, %c0, %c0, %c0, %88)
              %90 = affine.apply #map41(%thread_id_x_20, %86, %c0, %c0, %c0, %c0, %88)
              %91 = memref.load %alloc[%89, %90] : memref<128x16xf16, 3>
              %92 = vector.insert %91, %84 [2, 0] : f16 into vector<4x1xf16>
              %93 = affine.apply #map30(%arg11)
              %94 = affine.apply #map32(%93)
              %95 = affine.apply #map33(%93)
              %96 = affine.apply #map39(%95, %c0)
              %97 = affine.apply #map40(%thread_id_x_20, %94, %c0, %c0, %c0, %c0, %96)
              %98 = affine.apply #map41(%thread_id_x_20, %94, %c0, %c0, %c0, %c0, %96)
              %99 = memref.load %alloc[%97, %98] : memref<128x16xf16, 3>
              %100 = vector.insert %99, %92 [3, 0] : f16 into vector<4x1xf16>
              frisk.fragment_yield %100 : vector<4x1xf16>
            }
            %68 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %69 = frisk.warp_mma_rr %66, %67, %arg12 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %69 : vector<1x4xf32>
            }
            affine.yield %68 : vector<1x4xf32>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %58 = vector.extract %57[0, 0] : f32 from vector<1x4xf32>
          %59 = vector.insert %58, %arg10 [%56, 0] : f32 into vector<2x4xf32>
          %60 = vector.extract %57[0, 1] : f32 from vector<1x4xf32>
          %61 = vector.insert %60, %59 [%56, 1] : f32 into vector<2x4xf32>
          %62 = vector.extract %57[0, 2] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %61 [%56, 2] : f32 into vector<2x4xf32>
          %64 = vector.extract %57[0, 3] : f32 from vector<1x4xf32>
          %65 = vector.insert %64, %63 [%56, 3] : f32 into vector<2x4xf32>
          affine.yield %65 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "gemm_mn", tiled = true}
        %47 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst) -> (vector<2x4xf32>) {
          %56 = affine.apply #map(%arg9)
          %57 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %66 = vector.extract %arg6[%56, 0] : f32 from vector<2x4xf32>
            %67 = vector.insert %66, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %68 = vector.extract %arg6[%56, 1] : f32 from vector<2x4xf32>
            %69 = vector.insert %68, %67 [0, 1] : f32 into vector<1x4xf32>
            %70 = vector.extract %arg6[%56, 2] : f32 from vector<2x4xf32>
            %71 = vector.insert %70, %69 [0, 2] : f32 into vector<1x4xf32>
            %72 = vector.extract %arg6[%56, 3] : f32 from vector<2x4xf32>
            %73 = vector.insert %72, %71 [0, 3] : f32 into vector<1x4xf32>
            %74 = vector.extract %46[%56, 0] : f32 from vector<2x4xf32>
            %75 = vector.insert %74, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %76 = vector.extract %46[%56, 1] : f32 from vector<2x4xf32>
            %77 = vector.insert %76, %75 [0, 1] : f32 into vector<1x4xf32>
            %78 = vector.extract %46[%56, 2] : f32 from vector<2x4xf32>
            %79 = vector.insert %78, %77 [0, 2] : f32 into vector<1x4xf32>
            %80 = vector.extract %46[%56, 3] : f32 from vector<2x4xf32>
            %81 = vector.insert %80, %79 [0, 3] : f32 into vector<1x4xf32>
            %82 = arith.addf %73, %81 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %82 : vector<1x4xf32>
          }
          %58 = vector.extract %57[0, 0] : f32 from vector<1x4xf32>
          %59 = vector.insert %58, %arg10 [%56, 0] : f32 into vector<2x4xf32>
          %60 = vector.extract %57[0, 1] : f32 from vector<1x4xf32>
          %61 = vector.insert %60, %59 [%56, 1] : f32 into vector<2x4xf32>
          %62 = vector.extract %57[0, 2] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %61 [%56, 2] : f32 into vector<2x4xf32>
          %64 = vector.extract %57[0, 3] : f32 from vector<1x4xf32>
          %65 = vector.insert %64, %63 [%56, 3] : f32 into vector<2x4xf32>
          affine.yield %65 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
        %alloc_18 = memref.alloc() alignment = 16 {frisk.shared_pack = 4 : i64, frisk.shared_pack_axis = 0 : i64, frisk.shared_pair_stride = 16 : i64, tiled = true} : memref<128x16xf16, 3>
        %48 = affine.apply #map42(%block_id_z, %block_id_x, %arg5, %arg4)
        %49 = affine.apply #map43(%block_id_z, %block_id_x, %arg5, %arg4)
        %50 = affine.apply #map44(%block_id_z, %block_id_x, %arg5, %arg4)
        %subview_19 = memref.subview %arg2[%c0, %48, %49, %50] [1, 1, 128, 16] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x16xf16, strided<[128, 1], offset: ?>, 1>
        frisk.async_copy_gm_to_shm %subview_19 to %alloc_18 {alignment = 16 : i64} : memref<128x16xf16, strided<[128, 1], offset: ?>, 1>, memref<128x16xf16, 3>
        frisk.wait_buffer_ready %alloc_18 : memref<128x16xf16, 3>
        %51 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst) -> (vector<2x4xf32>) {
          %56 = affine.apply #map(%arg9)
          %57 = affine.for %arg11 = 0 to 8 iter_args(%arg12 = %cst_4) -> (vector<1x4xf32>) {
            %66 = frisk.fragment "load" -> (vector<1x4xf16>) {
              %69 = affine.apply #map(%arg9)
              %70 = affine.apply #map27(%arg11)
              %71 = vector.extract %42[%69, %70] : f16 from vector<2x32xf16>
              %72 = vector.insert %71, %cst_8 [0, 0] : f16 into vector<1x4xf16>
              %73 = affine.apply #map28(%arg11)
              %74 = vector.extract %42[%69, %73] : f16 from vector<2x32xf16>
              %75 = vector.insert %74, %72 [0, 1] : f16 into vector<1x4xf16>
              %76 = affine.apply #map29(%arg11)
              %77 = vector.extract %42[%69, %76] : f16 from vector<2x32xf16>
              %78 = vector.insert %77, %75 [0, 2] : f16 into vector<1x4xf16>
              %79 = affine.apply #map30(%arg11)
              %80 = vector.extract %42[%69, %79] : f16 from vector<2x32xf16>
              %81 = vector.insert %80, %78 [0, 3] : f16 into vector<1x4xf16>
              frisk.fragment_yield %81 : vector<1x4xf16>
            }
            %67 = frisk.fragment "load" -> (vector<4x1xf16>) {
              %thread_id_x_20 = gpu.thread_id x
              %69 = affine.apply #map27(%arg11)
              %70 = affine.apply #map32(%69)
              %71 = affine.apply #map33(%69)
              %72 = affine.apply #map39(%71, %c0)
              %73 = affine.apply #map40(%thread_id_x_20, %70, %c0, %c0, %c0, %c0, %72)
              %74 = affine.apply #map41(%thread_id_x_20, %70, %c0, %c0, %c0, %c0, %72)
              %75 = memref.load %alloc_18[%73, %74] : memref<128x16xf16, 3>
              %76 = vector.insert %75, %cst_3 [0, 0] : f16 into vector<4x1xf16>
              %77 = affine.apply #map28(%arg11)
              %78 = affine.apply #map32(%77)
              %79 = affine.apply #map33(%77)
              %80 = affine.apply #map39(%79, %c0)
              %81 = affine.apply #map40(%thread_id_x_20, %78, %c0, %c0, %c0, %c0, %80)
              %82 = affine.apply #map41(%thread_id_x_20, %78, %c0, %c0, %c0, %c0, %80)
              %83 = memref.load %alloc_18[%81, %82] : memref<128x16xf16, 3>
              %84 = vector.insert %83, %76 [1, 0] : f16 into vector<4x1xf16>
              %85 = affine.apply #map29(%arg11)
              %86 = affine.apply #map32(%85)
              %87 = affine.apply #map33(%85)
              %88 = affine.apply #map39(%87, %c0)
              %89 = affine.apply #map40(%thread_id_x_20, %86, %c0, %c0, %c0, %c0, %88)
              %90 = affine.apply #map41(%thread_id_x_20, %86, %c0, %c0, %c0, %c0, %88)
              %91 = memref.load %alloc_18[%89, %90] : memref<128x16xf16, 3>
              %92 = vector.insert %91, %84 [2, 0] : f16 into vector<4x1xf16>
              %93 = affine.apply #map30(%arg11)
              %94 = affine.apply #map32(%93)
              %95 = affine.apply #map33(%93)
              %96 = affine.apply #map39(%95, %c0)
              %97 = affine.apply #map40(%thread_id_x_20, %94, %c0, %c0, %c0, %c0, %96)
              %98 = affine.apply #map41(%thread_id_x_20, %94, %c0, %c0, %c0, %c0, %96)
              %99 = memref.load %alloc_18[%97, %98] : memref<128x16xf16, 3>
              %100 = vector.insert %99, %92 [3, 0] : f16 into vector<4x1xf16>
              frisk.fragment_yield %100 : vector<4x1xf16>
            }
            %68 = frisk.fragment "compute" -> (vector<1x4xf32>) {
              %69 = frisk.warp_mma_rr %66, %67, %arg12 {inst_constraints = "=v,v,v,0", inst_name = "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3", tiled = true} : vector<1x4xf16>, vector<4x1xf16>, vector<1x4xf32> -> vector<1x4xf32>
              frisk.fragment_yield %69 : vector<1x4xf32>
            }
            affine.yield %68 : vector<1x4xf32>
          } {frisk.fragment_loop, frisk.loopUnrollFull = true, iterLabel = "gemm_k"}
          %58 = vector.extract %57[0, 0] : f32 from vector<1x4xf32>
          %59 = vector.insert %58, %arg10 [%56, 0] : f32 into vector<2x4xf32>
          %60 = vector.extract %57[0, 1] : f32 from vector<1x4xf32>
          %61 = vector.insert %60, %59 [%56, 1] : f32 into vector<2x4xf32>
          %62 = vector.extract %57[0, 2] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %61 [%56, 2] : f32 into vector<2x4xf32>
          %64 = vector.extract %57[0, 3] : f32 from vector<1x4xf32>
          %65 = vector.insert %64, %63 [%56, 3] : f32 into vector<2x4xf32>
          affine.yield %65 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "gemm_mn", tiled = true}
        %52 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst) -> (vector<2x4xf32>) {
          %56 = affine.apply #map(%arg9)
          %57 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %66 = vector.extract %arg8[%56, 0] : f32 from vector<2x4xf32>
            %67 = vector.insert %66, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %68 = vector.extract %arg8[%56, 1] : f32 from vector<2x4xf32>
            %69 = vector.insert %68, %67 [0, 1] : f32 into vector<1x4xf32>
            %70 = vector.extract %arg8[%56, 2] : f32 from vector<2x4xf32>
            %71 = vector.insert %70, %69 [0, 2] : f32 into vector<1x4xf32>
            %72 = vector.extract %arg8[%56, 3] : f32 from vector<2x4xf32>
            %73 = vector.insert %72, %71 [0, 3] : f32 into vector<1x4xf32>
            %74 = vector.extract %51[%56, 0] : f32 from vector<2x4xf32>
            %75 = vector.insert %74, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %76 = vector.extract %51[%56, 1] : f32 from vector<2x4xf32>
            %77 = vector.insert %76, %75 [0, 1] : f32 into vector<1x4xf32>
            %78 = vector.extract %51[%56, 2] : f32 from vector<2x4xf32>
            %79 = vector.insert %78, %77 [0, 2] : f32 into vector<1x4xf32>
            %80 = vector.extract %51[%56, 3] : f32 from vector<2x4xf32>
            %81 = vector.insert %80, %79 [0, 3] : f32 into vector<1x4xf32>
            %82 = arith.addf %73, %81 {origin = "frisk.add", tiled = true} : vector<1x4xf32>
            frisk.fragment_yield %82 : vector<1x4xf32>
          }
          %58 = vector.extract %57[0, 0] : f32 from vector<1x4xf32>
          %59 = vector.insert %58, %arg10 [%56, 0] : f32 into vector<2x4xf32>
          %60 = vector.extract %57[0, 1] : f32 from vector<1x4xf32>
          %61 = vector.insert %60, %59 [%56, 1] : f32 into vector<2x4xf32>
          %62 = vector.extract %57[0, 2] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %61 [%56, 2] : f32 into vector<2x4xf32>
          %64 = vector.extract %57[0, 3] : f32 from vector<1x4xf32>
          %65 = vector.insert %64, %63 [%56, 3] : f32 into vector<2x4xf32>
          affine.yield %65 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
        %53 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst) -> (vector<2x4xf32>) {
          %56 = affine.apply #map(%arg9)
          %57 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %66 = vector.extract %47[%56, 0] : f32 from vector<2x4xf32>
            %67 = vector.insert %66, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %68 = vector.extract %47[%56, 1] : f32 from vector<2x4xf32>
            %69 = vector.insert %68, %67 [0, 1] : f32 into vector<1x4xf32>
            %70 = vector.extract %47[%56, 2] : f32 from vector<2x4xf32>
            %71 = vector.insert %70, %69 [0, 2] : f32 into vector<1x4xf32>
            %72 = vector.extract %47[%56, 3] : f32 from vector<2x4xf32>
            %73 = vector.insert %72, %71 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
            frisk.fragment_yield %73 : vector<1x4xf32>
          }
          %58 = vector.extract %57[0, 0] : f32 from vector<1x4xf32>
          %59 = vector.insert %58, %arg10 [%56, 0] : f32 into vector<2x4xf32>
          %60 = vector.extract %57[0, 1] : f32 from vector<1x4xf32>
          %61 = vector.insert %60, %59 [%56, 1] : f32 into vector<2x4xf32>
          %62 = vector.extract %57[0, 2] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %61 [%56, 2] : f32 into vector<2x4xf32>
          %64 = vector.extract %57[0, 3] : f32 from vector<1x4xf32>
          %65 = vector.insert %64, %63 [%56, 3] : f32 into vector<2x4xf32>
          affine.yield %65 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
        %54 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst_2) -> (vector<2x1xf32>) {
          %56 = affine.apply #map(%arg9)
          %57 = frisk.fragment "compute" -> (vector<1x1xf32>) {
            %60 = vector.extract %41[%56, 0] : f32 from vector<2x1xf32>
            %61 = vector.insert %60, %cst_1 [0, 0] {tiled = true} : f32 into vector<1x1xf32>
            frisk.fragment_yield %61 : vector<1x1xf32>
          }
          %58 = vector.extract %57[0, 0] : f32 from vector<1x1xf32>
          %59 = vector.insert %58, %arg10 [%56, 0] : f32 into vector<2x1xf32>
          affine.yield %59 : vector<2x1xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 1>, frisk.tile_shape = array<i64: 2, 1>, iterLabel = "fragment", tiled = true}
        %55 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %cst) -> (vector<2x4xf32>) {
          %56 = affine.apply #map(%arg9)
          %57 = frisk.fragment "compute" -> (vector<1x4xf32>) {
            %66 = vector.extract %52[%56, 0] : f32 from vector<2x4xf32>
            %67 = vector.insert %66, %cst_4 [0, 0] : f32 into vector<1x4xf32>
            %68 = vector.extract %52[%56, 1] : f32 from vector<2x4xf32>
            %69 = vector.insert %68, %67 [0, 1] : f32 into vector<1x4xf32>
            %70 = vector.extract %52[%56, 2] : f32 from vector<2x4xf32>
            %71 = vector.insert %70, %69 [0, 2] : f32 into vector<1x4xf32>
            %72 = vector.extract %52[%56, 3] : f32 from vector<2x4xf32>
            %73 = vector.insert %72, %71 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
            frisk.fragment_yield %73 : vector<1x4xf32>
          }
          %58 = vector.extract %57[0, 0] : f32 from vector<1x4xf32>
          %59 = vector.insert %58, %arg10 [%56, 0] : f32 into vector<2x4xf32>
          %60 = vector.extract %57[0, 1] : f32 from vector<1x4xf32>
          %61 = vector.insert %60, %59 [%56, 1] : f32 into vector<2x4xf32>
          %62 = vector.extract %57[0, 2] : f32 from vector<1x4xf32>
          %63 = vector.insert %62, %61 [%56, 2] : f32 into vector<2x4xf32>
          %64 = vector.extract %57[0, 3] : f32 from vector<1x4xf32>
          %65 = vector.insert %64, %63 [%56, 3] : f32 into vector<2x4xf32>
          affine.yield %65 : vector<2x4xf32>
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
        affine.yield %53, %54, %55 : vector<2x4xf32>, vector<2x1xf32>, vector<2x4xf32>
      } {local_yield_transformed = true}
      %thread_id_x = gpu.thread_id x
      %6 = arith.index_cast %thread_id_x : index to i32
      %7 = arith.andi %6, %c16_i32 : i32
      %8 = arith.cmpi ne, %7, %c0_i32 : i32
      %9 = arith.andi %6, %c32_i32 : i32
      %10 = arith.cmpi ne, %9, %c0_i32 : i32
      %11 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst) -> (vector<2x4xf32>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %40 = vector.extract %5#2[%30, 0] : f32 from vector<2x4xf32>
          %41 = vector.insert %40, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %42 = vector.extract %5#2[%30, 1] : f32 from vector<2x4xf32>
          %43 = vector.insert %42, %41 [0, 1] : f32 into vector<1x4xf32>
          %44 = vector.extract %5#2[%30, 2] : f32 from vector<2x4xf32>
          %45 = vector.insert %44, %43 [0, 2] : f32 into vector<1x4xf32>
          %46 = vector.extract %5#2[%30, 3] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %45 [0, 3] : f32 into vector<1x4xf32>
          %48 = arith.select %8, %40, %42 : f32
          %shuffleResult, %valid = gpu.shuffle xor %48, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %49 = arith.select %8, %shuffleResult, %40 : f32
          %50 = arith.select %8, %42, %shuffleResult : f32
          %51 = arith.select %8, %44, %46 : f32
          %shuffleResult_16, %valid_17 = gpu.shuffle xor %51, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %52 = arith.select %8, %shuffleResult_16, %44 : f32
          %53 = arith.select %8, %46, %shuffleResult_16 : f32
          %54 = arith.select %10, %49, %52 : f32
          %shuffleResult_18, %valid_19 = gpu.shuffle xor %54, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %55 = arith.select %10, %shuffleResult_18, %49 : f32
          %56 = arith.select %10, %52, %shuffleResult_18 : f32
          %57 = arith.select %10, %50, %53 : f32
          %shuffleResult_20, %valid_21 = gpu.shuffle xor %57, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %58 = arith.select %10, %shuffleResult_20, %50 : f32
          %59 = arith.select %10, %53, %shuffleResult_20 : f32
          %60 = vector.insert %55, %47 [0, 0] : f32 into vector<1x4xf32>
          %61 = vector.insert %58, %60 [0, 1] : f32 into vector<1x4xf32>
          %62 = vector.insert %56, %61 [0, 2] : f32 into vector<1x4xf32>
          %63 = vector.insert %59, %62 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %63 : vector<1x4xf32>
        }
        %32 = vector.extract %31[0, 0] : f32 from vector<1x4xf32>
        %33 = vector.insert %32, %arg6 [%30, 0] : f32 into vector<2x4xf32>
        %34 = vector.extract %31[0, 1] : f32 from vector<1x4xf32>
        %35 = vector.insert %34, %33 [%30, 1] : f32 into vector<2x4xf32>
        %36 = vector.extract %31[0, 2] : f32 from vector<1x4xf32>
        %37 = vector.insert %36, %35 [%30, 2] : f32 into vector<2x4xf32>
        %38 = vector.extract %31[0, 3] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %37 [%30, 3] : f32 into vector<2x4xf32>
        affine.yield %39 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %thread_id_x_13 = gpu.thread_id x
      %12 = arith.index_cast %thread_id_x_13 : index to i32
      %13 = arith.andi %12, %c16_i32 : i32
      %14 = arith.cmpi ne, %13, %c0_i32 : i32
      %15 = arith.andi %12, %c32_i32 : i32
      %16 = arith.cmpi ne, %15, %c0_i32 : i32
      %17 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst) -> (vector<2x4xf32>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %40 = vector.extract %5#0[%30, 0] : f32 from vector<2x4xf32>
          %41 = vector.insert %40, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %42 = vector.extract %5#0[%30, 1] : f32 from vector<2x4xf32>
          %43 = vector.insert %42, %41 [0, 1] : f32 into vector<1x4xf32>
          %44 = vector.extract %5#0[%30, 2] : f32 from vector<2x4xf32>
          %45 = vector.insert %44, %43 [0, 2] : f32 into vector<1x4xf32>
          %46 = vector.extract %5#0[%30, 3] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %45 [0, 3] : f32 into vector<1x4xf32>
          %48 = arith.select %14, %40, %42 : f32
          %shuffleResult, %valid = gpu.shuffle xor %48, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %49 = arith.select %14, %shuffleResult, %40 : f32
          %50 = arith.select %14, %42, %shuffleResult : f32
          %51 = arith.select %14, %44, %46 : f32
          %shuffleResult_16, %valid_17 = gpu.shuffle xor %51, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %52 = arith.select %14, %shuffleResult_16, %44 : f32
          %53 = arith.select %14, %46, %shuffleResult_16 : f32
          %54 = arith.select %16, %49, %52 : f32
          %shuffleResult_18, %valid_19 = gpu.shuffle xor %54, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %55 = arith.select %16, %shuffleResult_18, %49 : f32
          %56 = arith.select %16, %52, %shuffleResult_18 : f32
          %57 = arith.select %16, %50, %53 : f32
          %shuffleResult_20, %valid_21 = gpu.shuffle xor %57, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
          %58 = arith.select %16, %shuffleResult_20, %50 : f32
          %59 = arith.select %16, %53, %shuffleResult_20 : f32
          %60 = vector.insert %55, %47 [0, 0] : f32 into vector<1x4xf32>
          %61 = vector.insert %58, %60 [0, 1] : f32 into vector<1x4xf32>
          %62 = vector.insert %56, %61 [0, 2] : f32 into vector<1x4xf32>
          %63 = vector.insert %59, %62 [0, 3] {tiled = true} : f32 into vector<1x4xf32>
          frisk.fragment_yield %63 : vector<1x4xf32>
        }
        %32 = vector.extract %31[0, 0] : f32 from vector<1x4xf32>
        %33 = vector.insert %32, %arg6 [%30, 0] : f32 into vector<2x4xf32>
        %34 = vector.extract %31[0, 1] : f32 from vector<1x4xf32>
        %35 = vector.insert %34, %33 [%30, 1] : f32 into vector<2x4xf32>
        %36 = vector.extract %31[0, 2] : f32 from vector<1x4xf32>
        %37 = vector.insert %36, %35 [%30, 2] : f32 into vector<2x4xf32>
        %38 = vector.extract %31[0, 3] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %37 [%30, 3] : f32 into vector<2x4xf32>
        affine.yield %39 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %18 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst) -> (vector<2x4xf32>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %40 = vector.extract %5#1[%30, 0] : f32 from vector<2x1xf32>
          %41 = vector.insert %40, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %42 = vector.extract %5#1[%30, 0] : f32 from vector<2x1xf32>
          %43 = vector.insert %42, %41 [0, 1] : f32 into vector<1x4xf32>
          %44 = vector.extract %5#1[%30, 0] : f32 from vector<2x1xf32>
          %45 = vector.insert %44, %43 [0, 2] : f32 into vector<1x4xf32>
          %46 = vector.extract %5#1[%30, 0] : f32 from vector<2x1xf32>
          %47 = vector.insert %46, %45 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %47 : vector<1x4xf32>
        }
        %32 = vector.extract %31[0, 0] : f32 from vector<1x4xf32>
        %33 = vector.insert %32, %arg6 [%30, 0] : f32 into vector<2x4xf32>
        %34 = vector.extract %31[0, 1] : f32 from vector<1x4xf32>
        %35 = vector.insert %34, %33 [%30, 1] : f32 into vector<2x4xf32>
        %36 = vector.extract %31[0, 2] : f32 from vector<1x4xf32>
        %37 = vector.insert %36, %35 [%30, 2] : f32 into vector<2x4xf32>
        %38 = vector.extract %31[0, 3] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %37 [%30, 3] : f32 into vector<2x4xf32>
        affine.yield %39 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>}
      %19 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst) -> (vector<2x4xf32>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %40 = vector.extract %17[%30, 0] : f32 from vector<2x4xf32>
          %41 = vector.insert %40, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %42 = vector.extract %17[%30, 1] : f32 from vector<2x4xf32>
          %43 = vector.insert %42, %41 [0, 1] : f32 into vector<1x4xf32>
          %44 = vector.extract %17[%30, 2] : f32 from vector<2x4xf32>
          %45 = vector.insert %44, %43 [0, 2] : f32 into vector<1x4xf32>
          %46 = vector.extract %17[%30, 3] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %45 [0, 3] : f32 into vector<1x4xf32>
          %48 = vector.extract %18[%30, 0] : f32 from vector<2x4xf32>
          %49 = vector.insert %48, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %50 = vector.extract %18[%30, 1] : f32 from vector<2x4xf32>
          %51 = vector.insert %50, %49 [0, 1] : f32 into vector<1x4xf32>
          %52 = vector.extract %18[%30, 2] : f32 from vector<2x4xf32>
          %53 = vector.insert %52, %51 [0, 2] : f32 into vector<1x4xf32>
          %54 = vector.extract %18[%30, 3] : f32 from vector<2x4xf32>
          %55 = vector.insert %54, %53 [0, 3] : f32 into vector<1x4xf32>
          %56 = arith.divf %47, %55 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %56 : vector<1x4xf32>
        }
        %32 = vector.extract %31[0, 0] : f32 from vector<1x4xf32>
        %33 = vector.insert %32, %arg6 [%30, 0] : f32 into vector<2x4xf32>
        %34 = vector.extract %31[0, 1] : f32 from vector<1x4xf32>
        %35 = vector.insert %34, %33 [%30, 1] : f32 into vector<2x4xf32>
        %36 = vector.extract %31[0, 2] : f32 from vector<1x4xf32>
        %37 = vector.insert %36, %35 [%30, 2] : f32 into vector<2x4xf32>
        %38 = vector.extract %31[0, 3] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %37 [%30, 3] : f32 into vector<2x4xf32>
        affine.yield %39 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %20 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_9) -> (vector<2x4xf16>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %40 = vector.extract %19[%30, 0] : f32 from vector<2x4xf32>
          %41 = vector.insert %40, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %42 = vector.extract %19[%30, 1] : f32 from vector<2x4xf32>
          %43 = vector.insert %42, %41 [0, 1] : f32 into vector<1x4xf32>
          %44 = vector.extract %19[%30, 2] : f32 from vector<2x4xf32>
          %45 = vector.insert %44, %43 [0, 2] : f32 into vector<1x4xf32>
          %46 = vector.extract %19[%30, 3] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %45 [0, 3] : f32 into vector<1x4xf32>
          %48 = arith.truncf %47 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %48 : vector<1x4xf16>
        }
        %32 = vector.extract %31[0, 0] : f16 from vector<1x4xf16>
        %33 = vector.insert %32, %arg6 [%30, 0] : f16 into vector<2x4xf16>
        %34 = vector.extract %31[0, 1] : f16 from vector<1x4xf16>
        %35 = vector.insert %34, %33 [%30, 1] : f16 into vector<2x4xf16>
        %36 = vector.extract %31[0, 2] : f16 from vector<1x4xf16>
        %37 = vector.insert %36, %35 [%30, 2] : f16 into vector<2x4xf16>
        %38 = vector.extract %31[0, 3] : f16 from vector<1x4xf16>
        %39 = vector.insert %38, %37 [%30, 3] : f16 into vector<2x4xf16>
        affine.yield %39 : vector<2x4xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %21 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_9) -> (vector<2x4xf16>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %40 = vector.extract %20[%30, 0] : f16 from vector<2x4xf16>
          %41 = vector.insert %40, %cst_8 [0, 0] : f16 into vector<1x4xf16>
          %42 = vector.extract %20[%30, 1] : f16 from vector<2x4xf16>
          %43 = vector.insert %42, %41 [0, 1] : f16 into vector<1x4xf16>
          %44 = vector.extract %20[%30, 2] : f16 from vector<2x4xf16>
          %45 = vector.insert %44, %43 [0, 2] : f16 into vector<1x4xf16>
          %46 = vector.extract %20[%30, 3] : f16 from vector<2x4xf16>
          %47 = vector.insert %46, %45 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
          frisk.fragment_yield %47 : vector<1x4xf16>
        }
        %32 = vector.extract %31[0, 0] : f16 from vector<1x4xf16>
        %33 = vector.insert %32, %arg6 [%30, 0] : f16 into vector<2x4xf16>
        %34 = vector.extract %31[0, 1] : f16 from vector<1x4xf16>
        %35 = vector.insert %34, %33 [%30, 1] : f16 into vector<2x4xf16>
        %36 = vector.extract %31[0, 2] : f16 from vector<1x4xf16>
        %37 = vector.insert %36, %35 [%30, 2] : f16 into vector<2x4xf16>
        %38 = vector.extract %31[0, 3] : f16 from vector<1x4xf16>
        %39 = vector.insert %38, %37 [%30, 3] : f16 into vector<2x4xf16>
        affine.yield %39 : vector<2x4xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %thread_id_x_14 = gpu.thread_id x
      %22 = affine.apply #map45(%thread_id_x_14)
      %23 = arith.cmpi ult, %22, %c4 : index
      scf.if %23 {
        affine.for %arg5 = 0 to 2 {
          %30 = affine.apply #map(%arg5)
          frisk.fragment "store" {
            %31 = vector.extract %21[%30, 0] : f16 from vector<2x4xf16>
            %32 = affine.apply #map46(%arg4)[%block_id_z, %block_id_x, %block_id_y]
            %33 = affine.apply #map47(%arg4, %arg5)[%block_id_z, %block_id_x, %block_id_y, %thread_id_x_14]
            %34 = affine.apply #map48(%arg4)[%thread_id_x_14]
            affine.store %31, %arg3[%c0, %32, %33, %34] : memref<1x32x4096x128xf16, 1>
            %35 = vector.extract %21[%30, 1] : f16 from vector<2x4xf16>
            %36 = affine.apply #map46(%arg4)[%block_id_z, %block_id_x, %block_id_y]
            %37 = affine.apply #map47(%arg4, %arg5)[%block_id_z, %block_id_x, %block_id_y, %thread_id_x_14]
            %38 = affine.apply #map49(%arg4)[%thread_id_x_14]
            affine.store %35, %arg3[%c0, %36, %37, %38] : memref<1x32x4096x128xf16, 1>
            %39 = vector.extract %21[%30, 2] : f16 from vector<2x4xf16>
            %40 = affine.apply #map46(%arg4)[%block_id_z, %block_id_x, %block_id_y]
            %41 = affine.apply #map47(%arg4, %arg5)[%block_id_z, %block_id_x, %block_id_y, %thread_id_x_14]
            %42 = affine.apply #map50(%arg4)[%thread_id_x_14]
            affine.store %39, %arg3[%c0, %40, %41, %42] : memref<1x32x4096x128xf16, 1>
            %43 = vector.extract %21[%30, 3] : f16 from vector<2x4xf16>
            %44 = affine.apply #map46(%arg4)[%block_id_z, %block_id_x, %block_id_y]
            %45 = affine.apply #map47(%arg4, %arg5)[%block_id_z, %block_id_x, %block_id_y, %thread_id_x_14]
            %46 = affine.apply #map51(%arg4)[%thread_id_x_14]
            affine.store %43, %arg3[%c0, %44, %45, %46] : memref<1x32x4096x128xf16, 1>
            frisk.fragment_yield
          }
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "store", tiled = true}
      }
      %24 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst) -> (vector<2x4xf32>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %40 = vector.extract %5#1[%30, 0] : f32 from vector<2x1xf32>
          %41 = vector.insert %40, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %42 = vector.extract %5#1[%30, 0] : f32 from vector<2x1xf32>
          %43 = vector.insert %42, %41 [0, 1] : f32 into vector<1x4xf32>
          %44 = vector.extract %5#1[%30, 0] : f32 from vector<2x1xf32>
          %45 = vector.insert %44, %43 [0, 2] : f32 into vector<1x4xf32>
          %46 = vector.extract %5#1[%30, 0] : f32 from vector<2x1xf32>
          %47 = vector.insert %46, %45 [0, 3] : f32 into vector<1x4xf32>
          frisk.fragment_yield %47 : vector<1x4xf32>
        }
        %32 = vector.extract %31[0, 0] : f32 from vector<1x4xf32>
        %33 = vector.insert %32, %arg6 [%30, 0] : f32 into vector<2x4xf32>
        %34 = vector.extract %31[0, 1] : f32 from vector<1x4xf32>
        %35 = vector.insert %34, %33 [%30, 1] : f32 into vector<2x4xf32>
        %36 = vector.extract %31[0, 2] : f32 from vector<1x4xf32>
        %37 = vector.insert %36, %35 [%30, 2] : f32 into vector<2x4xf32>
        %38 = vector.extract %31[0, 3] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %37 [%30, 3] : f32 into vector<2x4xf32>
        affine.yield %39 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>}
      %25 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst) -> (vector<2x4xf32>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf32>) {
          %40 = vector.extract %11[%30, 0] : f32 from vector<2x4xf32>
          %41 = vector.insert %40, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %42 = vector.extract %11[%30, 1] : f32 from vector<2x4xf32>
          %43 = vector.insert %42, %41 [0, 1] : f32 into vector<1x4xf32>
          %44 = vector.extract %11[%30, 2] : f32 from vector<2x4xf32>
          %45 = vector.insert %44, %43 [0, 2] : f32 into vector<1x4xf32>
          %46 = vector.extract %11[%30, 3] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %45 [0, 3] : f32 into vector<1x4xf32>
          %48 = vector.extract %24[%30, 0] : f32 from vector<2x4xf32>
          %49 = vector.insert %48, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %50 = vector.extract %24[%30, 1] : f32 from vector<2x4xf32>
          %51 = vector.insert %50, %49 [0, 1] : f32 into vector<1x4xf32>
          %52 = vector.extract %24[%30, 2] : f32 from vector<2x4xf32>
          %53 = vector.insert %52, %51 [0, 2] : f32 into vector<1x4xf32>
          %54 = vector.extract %24[%30, 3] : f32 from vector<2x4xf32>
          %55 = vector.insert %54, %53 [0, 3] : f32 into vector<1x4xf32>
          %56 = arith.divf %47, %55 {origin = "frisk.div", tiled = true} : vector<1x4xf32>
          frisk.fragment_yield %56 : vector<1x4xf32>
        }
        %32 = vector.extract %31[0, 0] : f32 from vector<1x4xf32>
        %33 = vector.insert %32, %arg6 [%30, 0] : f32 into vector<2x4xf32>
        %34 = vector.extract %31[0, 1] : f32 from vector<1x4xf32>
        %35 = vector.insert %34, %33 [%30, 1] : f32 into vector<2x4xf32>
        %36 = vector.extract %31[0, 2] : f32 from vector<1x4xf32>
        %37 = vector.insert %36, %35 [%30, 2] : f32 into vector<2x4xf32>
        %38 = vector.extract %31[0, 3] : f32 from vector<1x4xf32>
        %39 = vector.insert %38, %37 [%30, 3] : f32 into vector<2x4xf32>
        affine.yield %39 : vector<2x4xf32>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %26 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_9) -> (vector<2x4xf16>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %40 = vector.extract %25[%30, 0] : f32 from vector<2x4xf32>
          %41 = vector.insert %40, %cst_4 [0, 0] : f32 into vector<1x4xf32>
          %42 = vector.extract %25[%30, 1] : f32 from vector<2x4xf32>
          %43 = vector.insert %42, %41 [0, 1] : f32 into vector<1x4xf32>
          %44 = vector.extract %25[%30, 2] : f32 from vector<2x4xf32>
          %45 = vector.insert %44, %43 [0, 2] : f32 into vector<1x4xf32>
          %46 = vector.extract %25[%30, 3] : f32 from vector<2x4xf32>
          %47 = vector.insert %46, %45 [0, 3] : f32 into vector<1x4xf32>
          %48 = arith.truncf %47 {tiled = true} : vector<1x4xf32> to vector<1x4xf16>
          frisk.fragment_yield %48 : vector<1x4xf16>
        }
        %32 = vector.extract %31[0, 0] : f16 from vector<1x4xf16>
        %33 = vector.insert %32, %arg6 [%30, 0] : f16 into vector<2x4xf16>
        %34 = vector.extract %31[0, 1] : f16 from vector<1x4xf16>
        %35 = vector.insert %34, %33 [%30, 1] : f16 into vector<2x4xf16>
        %36 = vector.extract %31[0, 2] : f16 from vector<1x4xf16>
        %37 = vector.insert %36, %35 [%30, 2] : f16 into vector<2x4xf16>
        %38 = vector.extract %31[0, 3] : f16 from vector<1x4xf16>
        %39 = vector.insert %38, %37 [%30, 3] : f16 into vector<2x4xf16>
        affine.yield %39 : vector<2x4xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %27 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst_9) -> (vector<2x4xf16>) {
        %30 = affine.apply #map(%arg5)
        %31 = frisk.fragment "compute" -> (vector<1x4xf16>) {
          %40 = vector.extract %26[%30, 0] : f16 from vector<2x4xf16>
          %41 = vector.insert %40, %cst_8 [0, 0] : f16 into vector<1x4xf16>
          %42 = vector.extract %26[%30, 1] : f16 from vector<2x4xf16>
          %43 = vector.insert %42, %41 [0, 1] : f16 into vector<1x4xf16>
          %44 = vector.extract %26[%30, 2] : f16 from vector<2x4xf16>
          %45 = vector.insert %44, %43 [0, 2] : f16 into vector<1x4xf16>
          %46 = vector.extract %26[%30, 3] : f16 from vector<2x4xf16>
          %47 = vector.insert %46, %45 [0, 3] {tiled = true} : f16 into vector<1x4xf16>
          frisk.fragment_yield %47 : vector<1x4xf16>
        }
        %32 = vector.extract %31[0, 0] : f16 from vector<1x4xf16>
        %33 = vector.insert %32, %arg6 [%30, 0] : f16 into vector<2x4xf16>
        %34 = vector.extract %31[0, 1] : f16 from vector<1x4xf16>
        %35 = vector.insert %34, %33 [%30, 1] : f16 into vector<2x4xf16>
        %36 = vector.extract %31[0, 2] : f16 from vector<1x4xf16>
        %37 = vector.insert %36, %35 [%30, 2] : f16 into vector<2x4xf16>
        %38 = vector.extract %31[0, 3] : f16 from vector<1x4xf16>
        %39 = vector.insert %38, %37 [%30, 3] : f16 into vector<2x4xf16>
        affine.yield %39 : vector<2x4xf16>
      } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "fragment", tiled = true}
      %thread_id_x_15 = gpu.thread_id x
      %28 = affine.apply #map45(%thread_id_x_15)
      %29 = arith.cmpi ult, %28, %c4 : index
      scf.if %29 {
        affine.for %arg5 = 0 to 2 {
          %30 = affine.apply #map(%arg5)
          frisk.fragment "store" {
            %31 = vector.extract %27[%30, 0] : f16 from vector<2x4xf16>
            %32 = affine.apply #map52(%arg4)[%block_id_z, %block_id_x, %block_id_y]
            %33 = affine.apply #map53(%arg4, %arg5)[%block_id_z, %block_id_x, %block_id_y, %thread_id_x_15]
            %34 = affine.apply #map54(%arg4)[%thread_id_x_15]
            affine.store %31, %arg3[%c0, %32, %33, %34] : memref<1x32x4096x128xf16, 1>
            %35 = vector.extract %27[%30, 1] : f16 from vector<2x4xf16>
            %36 = affine.apply #map52(%arg4)[%block_id_z, %block_id_x, %block_id_y]
            %37 = affine.apply #map53(%arg4, %arg5)[%block_id_z, %block_id_x, %block_id_y, %thread_id_x_15]
            %38 = affine.apply #map55(%arg4)[%thread_id_x_15]
            affine.store %35, %arg3[%c0, %36, %37, %38] : memref<1x32x4096x128xf16, 1>
            %39 = vector.extract %27[%30, 2] : f16 from vector<2x4xf16>
            %40 = affine.apply #map52(%arg4)[%block_id_z, %block_id_x, %block_id_y]
            %41 = affine.apply #map53(%arg4, %arg5)[%block_id_z, %block_id_x, %block_id_y, %thread_id_x_15]
            %42 = affine.apply #map56(%arg4)[%thread_id_x_15]
            affine.store %39, %arg3[%c0, %40, %41, %42] : memref<1x32x4096x128xf16, 1>
            %43 = vector.extract %27[%30, 3] : f16 from vector<2x4xf16>
            %44 = affine.apply #map52(%arg4)[%block_id_z, %block_id_x, %block_id_y]
            %45 = affine.apply #map53(%arg4, %arg5)[%block_id_z, %block_id_x, %block_id_y, %thread_id_x_15]
            %46 = affine.apply #map57(%arg4)[%thread_id_x_15]
            affine.store %43, %arg3[%c0, %44, %45, %46] : memref<1x32x4096x128xf16, 1>
            frisk.fragment_yield
          }
        } {frisk.fragment_loop, frisk.fragment_shape = array<i64: 1, 4>, frisk.tile_shape = array<i64: 2, 4>, iterLabel = "store", tiled = true}
      }
    } {local_yield_transformed = true}
    return
  }
}

