module {
  memref.global "public" @shm_0 : memref<16384xi8, 3> = uninitialized alignment = 16
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 32, 0, 1, 0>, gridDim = [1 : i32, 32 : i32, 32 : i32], thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
    %cst = arith.constant dense<1.000000e+00> : vector<4xf32>
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<2x1xf32>
    %c8192 = arith.constant 8192 : index
    %cst_2 = arith.constant 0.127531052 : f32
    %cst_3 = arith.constant 0xFF800000 : f32
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %c4 = arith.constant 4 : index
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_6 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %0 = memref.get_global @shm_0 : memref<16384xi8, 3>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %1 = arith.muli %block_id_y, %c32 : index
    %2 = arith.addi %block_id_x, %1 : index
    %3 = arith.divui %2, %c1024 : index
    %4 = arith.muli %3, %c32 : index
    %5 = arith.remui %2, %c1024 : index
    %6 = arith.remui %5, %c32 : index
    %7 = arith.addi %4, %6 : index
    %8 = arith.divui %5, %c32 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %9 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 + s2 * 32 + s0 floordiv 32) mod 32)>()[%7, %8, %block_id_z]
    %10 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%7]
    %subview = memref.subview %arg0[0, %9, %10, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %view = memref.view %0[%c0][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %11 = frisk.async_copy_gm_to_shm %subview to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %12 = arith.truncf %cst_2 : f32 to f16
    frisk.wait_buffer_ready %11 : !frisk.async_task
    %view_7 = memref.view %0[%c8192][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %13 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%7, %8]
    %subview_8 = memref.subview %arg0[0, %13, %10, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %14 = frisk.async_copy_gm_to_shm %subview_8 to %view_7 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %thread_id_x = gpu.thread_id x
    %15 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128)>()[%thread_id_x]
    %16 = vector.load %view[%15] : memref<4096xf16, 3>, vector<4xf16>
    %17 = vector.broadcast %12 : f16 to vector<4xf16>
    %18 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>()[%thread_id_x]
    %19 = vector.load %view[%18] : memref<4096xf16, 3>, vector<4xf16>
    %20 = vector.extract %16[0] : f16 from vector<4xf16>
    %21 = vector.extract %16[1] : f16 from vector<4xf16>
    %22 = vector.extract %16[2] : f16 from vector<4xf16>
    %23 = vector.extract %16[3] : f16 from vector<4xf16>
    %24 = vector.extract %19[0] : f16 from vector<4xf16>
    %25 = vector.extract %19[1] : f16 from vector<4xf16>
    %26 = vector.extract %19[2] : f16 from vector<4xf16>
    %27 = vector.extract %19[3] : f16 from vector<4xf16>
    %28 = vector.from_elements %20, %21, %22, %23 : vector<4xf16>
    %29 = arith.mulf %28, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %30 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 512)>()[%thread_id_x]
    %31 = vector.load %view[%30] : memref<4096xf16, 3>, vector<4xf16>
    %32 = vector.from_elements %24, %25, %26, %27 : vector<4xf16>
    %33 = arith.mulf %32, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %34 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2560)>()[%thread_id_x]
    %35 = vector.load %view[%34] : memref<4096xf16, 3>, vector<4xf16>
    %36 = vector.extract %29[0] : f16 from vector<4xf16>
    %37 = vector.extract %31[0] : f16 from vector<4xf16>
    %38 = vector.extract %31[1] : f16 from vector<4xf16>
    %39 = vector.extract %31[2] : f16 from vector<4xf16>
    %40 = vector.extract %31[3] : f16 from vector<4xf16>
    %41 = vector.extract %35[0] : f16 from vector<4xf16>
    %42 = vector.extract %35[1] : f16 from vector<4xf16>
    %43 = vector.extract %35[2] : f16 from vector<4xf16>
    %44 = vector.extract %35[3] : f16 from vector<4xf16>
    %45 = vector.from_elements %37, %38, %39, %40 : vector<4xf16>
    %46 = arith.mulf %45, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %47 = vector.extract %29[1] : f16 from vector<4xf16>
    %48 = vector.extract %29[2] : f16 from vector<4xf16>
    %49 = vector.extract %29[3] : f16 from vector<4xf16>
    %50 = vector.extract %33[0] : f16 from vector<4xf16>
    %51 = vector.extract %33[1] : f16 from vector<4xf16>
    %52 = vector.extract %33[2] : f16 from vector<4xf16>
    %53 = vector.extract %33[3] : f16 from vector<4xf16>
    %54 = vector.extract %46[0] : f16 from vector<4xf16>
    %55 = vector.extract %46[1] : f16 from vector<4xf16>
    %56 = vector.extract %46[2] : f16 from vector<4xf16>
    %57 = vector.extract %46[3] : f16 from vector<4xf16>
    %58 = vector.from_elements %41, %42, %43, %44 : vector<4xf16>
    %59 = arith.mulf %58, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %60 = vector.extract %59[0] : f16 from vector<4xf16>
    %61 = vector.extract %59[1] : f16 from vector<4xf16>
    %62 = vector.extract %59[2] : f16 from vector<4xf16>
    %63 = vector.extract %59[3] : f16 from vector<4xf16>
    frisk.wait_buffer_ready %14 : !frisk.async_task
    %subview_9 = memref.subview %arg0[0, %13, %10, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %64 = frisk.async_copy_gm_to_shm %subview_9 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %65 = vector.load %view_7[%15] : memref<4096xf16, 3>, vector<4xf16>
    %66 = vector.load %view_7[%18] : memref<4096xf16, 3>, vector<4xf16>
    %67 = vector.load %view_7[%30] : memref<4096xf16, 3>, vector<4xf16>
    %68 = vector.extract %65[0] : f16 from vector<4xf16>
    %69 = vector.extract %65[1] : f16 from vector<4xf16>
    %70 = vector.extract %65[2] : f16 from vector<4xf16>
    %71 = vector.extract %65[3] : f16 from vector<4xf16>
    %72 = vector.extract %66[0] : f16 from vector<4xf16>
    %73 = vector.extract %66[1] : f16 from vector<4xf16>
    %74 = vector.extract %66[2] : f16 from vector<4xf16>
    %75 = vector.extract %66[3] : f16 from vector<4xf16>
    %76 = vector.from_elements %68, %69, %70, %71 : vector<4xf16>
    %77 = arith.mulf %76, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %78 = vector.load %view_7[%34] : memref<4096xf16, 3>, vector<4xf16>
    %79 = vector.extract %67[0] : f16 from vector<4xf16>
    %80 = vector.extract %67[1] : f16 from vector<4xf16>
    %81 = vector.extract %67[2] : f16 from vector<4xf16>
    %82 = vector.extract %67[3] : f16 from vector<4xf16>
    %83 = vector.from_elements %72, %73, %74, %75 : vector<4xf16>
    %84 = arith.mulf %83, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %85 = vector.extract %77[0] : f16 from vector<4xf16>
    %86 = vector.extract %77[1] : f16 from vector<4xf16>
    %87 = vector.extract %77[2] : f16 from vector<4xf16>
    %88 = vector.extract %77[3] : f16 from vector<4xf16>
    %89 = vector.extract %84[0] : f16 from vector<4xf16>
    %90 = vector.extract %84[1] : f16 from vector<4xf16>
    %91 = vector.extract %84[2] : f16 from vector<4xf16>
    %92 = vector.extract %84[3] : f16 from vector<4xf16>
    %93 = vector.extract %78[0] : f16 from vector<4xf16>
    %94 = vector.extract %78[1] : f16 from vector<4xf16>
    %95 = vector.extract %78[2] : f16 from vector<4xf16>
    %96 = vector.extract %78[3] : f16 from vector<4xf16>
    %97 = vector.from_elements %79, %80, %81, %82 : vector<4xf16>
    %98 = arith.mulf %97, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %99 = vector.from_elements %93, %94, %95, %96 : vector<4xf16>
    %100 = arith.mulf %99, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %101 = vector.extract %98[0] : f16 from vector<4xf16>
    %102 = vector.extract %98[1] : f16 from vector<4xf16>
    %103 = vector.extract %98[2] : f16 from vector<4xf16>
    %104 = vector.extract %98[3] : f16 from vector<4xf16>
    %105 = vector.extract %100[0] : f16 from vector<4xf16>
    %106 = vector.extract %100[1] : f16 from vector<4xf16>
    %107 = vector.extract %100[2] : f16 from vector<4xf16>
    %108 = vector.extract %100[3] : f16 from vector<4xf16>
    frisk.wait_buffer_ready %64 : !frisk.async_task
    %subview_10 = memref.subview %arg0[0, %13, %10, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %109 = frisk.async_copy_gm_to_shm %subview_10 to %view_7 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %110 = vector.load %view[%15] : memref<4096xf16, 3>, vector<4xf16>
    %111 = vector.load %view[%18] : memref<4096xf16, 3>, vector<4xf16>
    %112 = vector.load %view[%30] : memref<4096xf16, 3>, vector<4xf16>
    %113 = vector.extract %110[0] : f16 from vector<4xf16>
    %114 = vector.extract %110[1] : f16 from vector<4xf16>
    %115 = vector.extract %110[2] : f16 from vector<4xf16>
    %116 = vector.extract %110[3] : f16 from vector<4xf16>
    %117 = vector.extract %111[0] : f16 from vector<4xf16>
    %118 = vector.extract %111[1] : f16 from vector<4xf16>
    %119 = vector.extract %111[2] : f16 from vector<4xf16>
    %120 = vector.extract %111[3] : f16 from vector<4xf16>
    %121 = vector.from_elements %113, %114, %115, %116 : vector<4xf16>
    %122 = arith.mulf %121, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %123 = vector.load %view[%34] : memref<4096xf16, 3>, vector<4xf16>
    %124 = vector.extract %112[0] : f16 from vector<4xf16>
    %125 = vector.extract %112[1] : f16 from vector<4xf16>
    %126 = vector.extract %112[2] : f16 from vector<4xf16>
    %127 = vector.extract %112[3] : f16 from vector<4xf16>
    %128 = vector.from_elements %117, %118, %119, %120 : vector<4xf16>
    %129 = arith.mulf %128, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %130 = vector.extract %122[0] : f16 from vector<4xf16>
    %131 = vector.extract %122[1] : f16 from vector<4xf16>
    %132 = vector.extract %122[2] : f16 from vector<4xf16>
    %133 = vector.extract %122[3] : f16 from vector<4xf16>
    %134 = vector.extract %129[0] : f16 from vector<4xf16>
    %135 = vector.extract %129[1] : f16 from vector<4xf16>
    %136 = vector.extract %129[2] : f16 from vector<4xf16>
    %137 = vector.extract %129[3] : f16 from vector<4xf16>
    %138 = vector.extract %123[0] : f16 from vector<4xf16>
    %139 = vector.extract %123[1] : f16 from vector<4xf16>
    %140 = vector.extract %123[2] : f16 from vector<4xf16>
    %141 = vector.extract %123[3] : f16 from vector<4xf16>
    %142 = vector.from_elements %124, %125, %126, %127 : vector<4xf16>
    %143 = arith.mulf %142, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %144 = vector.from_elements %138, %139, %140, %141 : vector<4xf16>
    %145 = arith.mulf %144, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %146 = vector.extract %143[0] : f16 from vector<4xf16>
    %147 = vector.extract %143[1] : f16 from vector<4xf16>
    %148 = vector.extract %143[2] : f16 from vector<4xf16>
    %149 = vector.extract %143[3] : f16 from vector<4xf16>
    %150 = vector.extract %145[0] : f16 from vector<4xf16>
    %151 = vector.extract %145[1] : f16 from vector<4xf16>
    %152 = vector.extract %145[2] : f16 from vector<4xf16>
    %153 = vector.extract %145[3] : f16 from vector<4xf16>
    frisk.wait_buffer_ready %109 : !frisk.async_task
    %154 = vector.load %view_7[%15] : memref<4096xf16, 3>, vector<4xf16>
    %155 = vector.load %view_7[%18] : memref<4096xf16, 3>, vector<4xf16>
    %156 = vector.load %view_7[%30] : memref<4096xf16, 3>, vector<4xf16>
    %157 = vector.extract %154[0] : f16 from vector<4xf16>
    %158 = vector.extract %154[1] : f16 from vector<4xf16>
    %159 = vector.extract %154[2] : f16 from vector<4xf16>
    %160 = vector.extract %154[3] : f16 from vector<4xf16>
    %161 = vector.extract %155[0] : f16 from vector<4xf16>
    %162 = vector.extract %155[1] : f16 from vector<4xf16>
    %163 = vector.extract %155[2] : f16 from vector<4xf16>
    %164 = vector.extract %155[3] : f16 from vector<4xf16>
    %165 = vector.from_elements %157, %158, %159, %160 : vector<4xf16>
    %166 = arith.mulf %165, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %167 = vector.load %view_7[%34] : memref<4096xf16, 3>, vector<4xf16>
    %168 = vector.extract %156[0] : f16 from vector<4xf16>
    %169 = vector.extract %156[1] : f16 from vector<4xf16>
    %170 = vector.extract %156[2] : f16 from vector<4xf16>
    %171 = vector.extract %156[3] : f16 from vector<4xf16>
    %172 = vector.from_elements %161, %162, %163, %164 : vector<4xf16>
    %173 = arith.mulf %172, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %174 = vector.extract %166[0] : f16 from vector<4xf16>
    %175 = vector.extract %166[1] : f16 from vector<4xf16>
    %176 = vector.extract %166[2] : f16 from vector<4xf16>
    %177 = vector.extract %166[3] : f16 from vector<4xf16>
    %178 = vector.extract %173[0] : f16 from vector<4xf16>
    %179 = vector.extract %173[1] : f16 from vector<4xf16>
    %180 = vector.extract %173[2] : f16 from vector<4xf16>
    %181 = vector.extract %173[3] : f16 from vector<4xf16>
    %182 = vector.extract %167[0] : f16 from vector<4xf16>
    %183 = vector.extract %167[1] : f16 from vector<4xf16>
    %184 = vector.extract %167[2] : f16 from vector<4xf16>
    %185 = vector.extract %167[3] : f16 from vector<4xf16>
    %186 = vector.from_elements %168, %169, %170, %171 : vector<4xf16>
    %187 = arith.mulf %186, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %188 = vector.from_elements %182, %183, %184, %185 : vector<4xf16>
    %189 = arith.mulf %188, %17 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %190 = vector.extract %187[0] : f16 from vector<4xf16>
    %191 = vector.extract %187[1] : f16 from vector<4xf16>
    %192 = vector.extract %187[2] : f16 from vector<4xf16>
    %193 = vector.extract %187[3] : f16 from vector<4xf16>
    %194 = vector.extract %189[0] : f16 from vector<4xf16>
    %195 = vector.extract %189[1] : f16 from vector<4xf16>
    %196 = vector.extract %189[2] : f16 from vector<4xf16>
    %197 = vector.extract %189[3] : f16 from vector<4xf16>
    gpu.barrier
    %198 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%8]
    %199 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
    %200 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
    %201 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
    %202 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
    %203:6 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %cst_5, %arg6 = %cst_4, %arg7 = %cst_1, %arg8 = %cst_5, %arg9 = %cst_5, %arg10 = %cst_5) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %585 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %subview_137 = memref.subview %transpose[0, %198, 0, %585] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %586 = frisk.async_copy_gm_to_shm %subview_137 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %586 : !frisk.async_task
      %subview_138 = memref.subview %transpose[0, %198, 32, %585] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %587 = frisk.async_copy_gm_to_shm %subview_138 to %view_7 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %588 = affine.apply affine_map<()[s0] -> (s0 * 160 - (s0 floordiv 4) * 640 + ((s0 mod 16) floordiv 4) * 8 - ((s0 mod 4) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>()[%thread_id_x]
      %589 = vector.load %view[%588] : memref<4096xf16, 3>, vector<4xf16>
      %590 = vector.from_elements %36, %47, %48, %49 : vector<4xf16>
      %591 = vector.from_elements %50, %51, %52, %53 : vector<4xf16>
      %592 = vector.extract %589[0] : f16 from vector<4xf16>
      %593 = vector.extract %589[1] : f16 from vector<4xf16>
      %594 = vector.extract %589[2] : f16 from vector<4xf16>
      %595 = vector.extract %589[3] : f16 from vector<4xf16>
      %596 = vector.from_elements %592, %593, %594, %595 : vector<4xf16>
      %597 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %590, %596, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %598 = vector.from_elements %54, %55, %56, %57 : vector<4xf16>
      %599 = vector.from_elements %60, %61, %62, %63 : vector<4xf16>
      %600 = affine.apply affine_map<()[s0] -> (s0 * 160 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 - (s0 floordiv 2) * 64 + 2048)>()[%thread_id_x]
      %601 = vector.load %view[%600] : memref<4096xf16, 3>, vector<4xf16>
      %602 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 16) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 16) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 32)>()[%thread_id_x]
      %603 = vector.load %view[%602] : memref<4096xf16, 3>, vector<4xf16>
      %604 = vector.extract %601[0] : f16 from vector<4xf16>
      %605 = vector.extract %601[1] : f16 from vector<4xf16>
      %606 = vector.extract %601[2] : f16 from vector<4xf16>
      %607 = vector.extract %601[3] : f16 from vector<4xf16>
      %608 = vector.from_elements %604, %605, %606, %607 : vector<4xf16>
      %609 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %591, %608, %597 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %610 = vector.extract %603[0] : f16 from vector<4xf16>
      %611 = vector.extract %603[1] : f16 from vector<4xf16>
      %612 = vector.extract %603[2] : f16 from vector<4xf16>
      %613 = vector.extract %603[3] : f16 from vector<4xf16>
      %614 = vector.from_elements %610, %611, %612, %613 : vector<4xf16>
      %615 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %590, %614, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %616 = vector.extract %609[0] : f32 from vector<4xf32>
      %617 = vector.extract %609[1] : f32 from vector<4xf32>
      %618 = vector.extract %609[2] : f32 from vector<4xf32>
      %619 = vector.extract %609[3] : f32 from vector<4xf32>
      %620 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 16) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 16) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + 2080)>()[%thread_id_x]
      %621 = vector.load %view[%620] : memref<4096xf16, 3>, vector<4xf16>
      %622 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 32) floordiv 64) * 512 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 64)>()[%thread_id_x]
      %623 = vector.load %view[%622] : memref<4096xf16, 3>, vector<4xf16>
      %624 = vector.extract %621[0] : f16 from vector<4xf16>
      %625 = vector.extract %621[1] : f16 from vector<4xf16>
      %626 = vector.extract %621[2] : f16 from vector<4xf16>
      %627 = vector.extract %621[3] : f16 from vector<4xf16>
      %628 = vector.from_elements %624, %625, %626, %627 : vector<4xf16>
      %629 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %591, %628, %615 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %630 = vector.extract %623[0] : f16 from vector<4xf16>
      %631 = vector.extract %623[1] : f16 from vector<4xf16>
      %632 = vector.extract %623[2] : f16 from vector<4xf16>
      %633 = vector.extract %623[3] : f16 from vector<4xf16>
      %634 = vector.from_elements %630, %631, %632, %633 : vector<4xf16>
      %635 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %590, %634, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %636 = vector.extract %629[0] : f32 from vector<4xf32>
      %637 = vector.extract %629[1] : f32 from vector<4xf32>
      %638 = vector.extract %629[2] : f32 from vector<4xf32>
      %639 = vector.extract %629[3] : f32 from vector<4xf32>
      %640 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 32) floordiv 64) * 512 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + 2112)>()[%thread_id_x]
      %641 = vector.load %view[%640] : memref<4096xf16, 3>, vector<4xf16>
      %642 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 48) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 48) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 160)>()[%thread_id_x]
      %643 = vector.load %view[%642] : memref<4096xf16, 3>, vector<4xf16>
      %644 = vector.extract %641[0] : f16 from vector<4xf16>
      %645 = vector.extract %641[1] : f16 from vector<4xf16>
      %646 = vector.extract %641[2] : f16 from vector<4xf16>
      %647 = vector.extract %641[3] : f16 from vector<4xf16>
      %648 = vector.from_elements %644, %645, %646, %647 : vector<4xf16>
      %649 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %591, %648, %635 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %650 = vector.extract %643[0] : f16 from vector<4xf16>
      %651 = vector.extract %643[1] : f16 from vector<4xf16>
      %652 = vector.extract %643[2] : f16 from vector<4xf16>
      %653 = vector.extract %643[3] : f16 from vector<4xf16>
      %654 = vector.from_elements %650, %651, %652, %653 : vector<4xf16>
      %655 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %590, %654, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %656 = vector.extract %649[0] : f32 from vector<4xf32>
      %657 = vector.extract %649[1] : f32 from vector<4xf32>
      %658 = vector.extract %649[2] : f32 from vector<4xf32>
      %659 = vector.extract %649[3] : f32 from vector<4xf32>
      %660 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 48) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 48) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + 2208)>()[%thread_id_x]
      %661 = vector.load %view[%660] : memref<4096xf16, 3>, vector<4xf16>
      %662 = affine.apply affine_map<()[s0] -> (s0 * 160 - (s0 floordiv 4) * 640 + ((s0 mod 16) floordiv 4) * 8 - ((s0 mod 4) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 512)>()[%thread_id_x]
      %663 = vector.load %view[%662] : memref<4096xf16, 3>, vector<4xf16>
      %664 = vector.extract %661[0] : f16 from vector<4xf16>
      %665 = vector.extract %661[1] : f16 from vector<4xf16>
      %666 = vector.extract %661[2] : f16 from vector<4xf16>
      %667 = vector.extract %661[3] : f16 from vector<4xf16>
      %668 = vector.from_elements %664, %665, %666, %667 : vector<4xf16>
      %669 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %591, %668, %655 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %670 = vector.extract %663[0] : f16 from vector<4xf16>
      %671 = vector.extract %663[1] : f16 from vector<4xf16>
      %672 = vector.extract %663[2] : f16 from vector<4xf16>
      %673 = vector.extract %663[3] : f16 from vector<4xf16>
      %674 = vector.from_elements %670, %671, %672, %673 : vector<4xf16>
      %675 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %590, %674, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %676 = vector.extract %669[0] : f32 from vector<4xf32>
      %677 = vector.extract %669[1] : f32 from vector<4xf32>
      %678 = vector.extract %669[2] : f32 from vector<4xf32>
      %679 = vector.extract %669[3] : f32 from vector<4xf32>
      %680 = affine.apply affine_map<()[s0] -> (s0 * 160 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 - (s0 floordiv 2) * 64 + 2560)>()[%thread_id_x]
      %681 = vector.load %view[%680] : memref<4096xf16, 3>, vector<4xf16>
      %682 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 80) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 80) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 32)>()[%thread_id_x]
      %683 = vector.load %view[%682] : memref<4096xf16, 3>, vector<4xf16>
      %684 = vector.extract %681[0] : f16 from vector<4xf16>
      %685 = vector.extract %681[1] : f16 from vector<4xf16>
      %686 = vector.extract %681[2] : f16 from vector<4xf16>
      %687 = vector.extract %681[3] : f16 from vector<4xf16>
      %688 = vector.from_elements %684, %685, %686, %687 : vector<4xf16>
      %689 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %591, %688, %675 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %690 = vector.extract %683[0] : f16 from vector<4xf16>
      %691 = vector.extract %683[1] : f16 from vector<4xf16>
      %692 = vector.extract %683[2] : f16 from vector<4xf16>
      %693 = vector.extract %683[3] : f16 from vector<4xf16>
      %694 = vector.from_elements %690, %691, %692, %693 : vector<4xf16>
      %695 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %590, %694, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %696 = vector.extract %689[0] : f32 from vector<4xf32>
      %697 = vector.extract %689[1] : f32 from vector<4xf32>
      %698 = vector.extract %689[2] : f32 from vector<4xf32>
      %699 = vector.extract %689[3] : f32 from vector<4xf32>
      %700 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 80) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 80) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + 2080)>()[%thread_id_x]
      %701 = vector.load %view[%700] : memref<4096xf16, 3>, vector<4xf16>
      %702 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 96) floordiv 64) * 512 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 64)>()[%thread_id_x]
      %703 = vector.load %view[%702] : memref<4096xf16, 3>, vector<4xf16>
      %704 = vector.extract %701[0] : f16 from vector<4xf16>
      %705 = vector.extract %701[1] : f16 from vector<4xf16>
      %706 = vector.extract %701[2] : f16 from vector<4xf16>
      %707 = vector.extract %701[3] : f16 from vector<4xf16>
      %708 = vector.from_elements %704, %705, %706, %707 : vector<4xf16>
      %709 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %591, %708, %695 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %710 = vector.extract %703[0] : f16 from vector<4xf16>
      %711 = vector.extract %703[1] : f16 from vector<4xf16>
      %712 = vector.extract %703[2] : f16 from vector<4xf16>
      %713 = vector.extract %703[3] : f16 from vector<4xf16>
      %714 = vector.from_elements %710, %711, %712, %713 : vector<4xf16>
      %715 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %590, %714, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %716 = vector.extract %709[0] : f32 from vector<4xf32>
      %717 = vector.extract %709[1] : f32 from vector<4xf32>
      %718 = vector.extract %709[2] : f32 from vector<4xf32>
      %719 = vector.extract %709[3] : f32 from vector<4xf32>
      %720 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 96) floordiv 64) * 512 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + 2112)>()[%thread_id_x]
      %721 = vector.load %view[%720] : memref<4096xf16, 3>, vector<4xf16>
      %722 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 112) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 112) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 160)>()[%thread_id_x]
      %723 = vector.load %view[%722] : memref<4096xf16, 3>, vector<4xf16>
      %724 = vector.extract %721[0] : f16 from vector<4xf16>
      %725 = vector.extract %721[1] : f16 from vector<4xf16>
      %726 = vector.extract %721[2] : f16 from vector<4xf16>
      %727 = vector.extract %721[3] : f16 from vector<4xf16>
      %728 = vector.from_elements %724, %725, %726, %727 : vector<4xf16>
      %729 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %591, %728, %715 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %730 = vector.extract %723[0] : f16 from vector<4xf16>
      %731 = vector.extract %723[1] : f16 from vector<4xf16>
      %732 = vector.extract %723[2] : f16 from vector<4xf16>
      %733 = vector.extract %723[3] : f16 from vector<4xf16>
      %734 = vector.from_elements %730, %731, %732, %733 : vector<4xf16>
      %735 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %590, %734, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %736 = vector.extract %729[0] : f32 from vector<4xf32>
      %737 = vector.extract %729[1] : f32 from vector<4xf32>
      %738 = vector.extract %729[2] : f32 from vector<4xf32>
      %739 = vector.extract %729[3] : f32 from vector<4xf32>
      %740 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 112) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 112) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + 2208)>()[%thread_id_x]
      %741 = vector.load %view[%740] : memref<4096xf16, 3>, vector<4xf16>
      %742 = vector.load %view[%588] : memref<4096xf16, 3>, vector<4xf16>
      %743 = vector.extract %741[0] : f16 from vector<4xf16>
      %744 = vector.extract %741[1] : f16 from vector<4xf16>
      %745 = vector.extract %741[2] : f16 from vector<4xf16>
      %746 = vector.extract %741[3] : f16 from vector<4xf16>
      %747 = vector.from_elements %743, %744, %745, %746 : vector<4xf16>
      %748 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %591, %747, %735 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %749 = vector.extract %742[0] : f16 from vector<4xf16>
      %750 = vector.extract %742[1] : f16 from vector<4xf16>
      %751 = vector.extract %742[2] : f16 from vector<4xf16>
      %752 = vector.extract %742[3] : f16 from vector<4xf16>
      %753 = vector.from_elements %749, %750, %751, %752 : vector<4xf16>
      %754 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %598, %753, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %755 = vector.extract %748[0] : f32 from vector<4xf32>
      %756 = vector.extract %748[1] : f32 from vector<4xf32>
      %757 = vector.extract %748[2] : f32 from vector<4xf32>
      %758 = vector.extract %748[3] : f32 from vector<4xf32>
      %759 = vector.load %view[%600] : memref<4096xf16, 3>, vector<4xf16>
      %760 = vector.load %view[%602] : memref<4096xf16, 3>, vector<4xf16>
      %761 = vector.extract %759[0] : f16 from vector<4xf16>
      %762 = vector.extract %759[1] : f16 from vector<4xf16>
      %763 = vector.extract %759[2] : f16 from vector<4xf16>
      %764 = vector.extract %759[3] : f16 from vector<4xf16>
      %765 = vector.from_elements %761, %762, %763, %764 : vector<4xf16>
      %766 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %599, %765, %754 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %767 = vector.extract %760[0] : f16 from vector<4xf16>
      %768 = vector.extract %760[1] : f16 from vector<4xf16>
      %769 = vector.extract %760[2] : f16 from vector<4xf16>
      %770 = vector.extract %760[3] : f16 from vector<4xf16>
      %771 = vector.from_elements %767, %768, %769, %770 : vector<4xf16>
      %772 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %598, %771, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %773 = vector.extract %766[0] : f32 from vector<4xf32>
      %774 = vector.extract %766[1] : f32 from vector<4xf32>
      %775 = vector.extract %766[2] : f32 from vector<4xf32>
      %776 = vector.extract %766[3] : f32 from vector<4xf32>
      %777 = vector.load %view[%620] : memref<4096xf16, 3>, vector<4xf16>
      %778 = vector.load %view[%622] : memref<4096xf16, 3>, vector<4xf16>
      %779 = vector.extract %777[0] : f16 from vector<4xf16>
      %780 = vector.extract %777[1] : f16 from vector<4xf16>
      %781 = vector.extract %777[2] : f16 from vector<4xf16>
      %782 = vector.extract %777[3] : f16 from vector<4xf16>
      %783 = vector.from_elements %779, %780, %781, %782 : vector<4xf16>
      %784 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %599, %783, %772 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %785 = vector.extract %778[0] : f16 from vector<4xf16>
      %786 = vector.extract %778[1] : f16 from vector<4xf16>
      %787 = vector.extract %778[2] : f16 from vector<4xf16>
      %788 = vector.extract %778[3] : f16 from vector<4xf16>
      %789 = vector.from_elements %785, %786, %787, %788 : vector<4xf16>
      %790 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %598, %789, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %791 = vector.extract %784[0] : f32 from vector<4xf32>
      %792 = vector.extract %784[1] : f32 from vector<4xf32>
      %793 = vector.extract %784[2] : f32 from vector<4xf32>
      %794 = vector.extract %784[3] : f32 from vector<4xf32>
      %795 = vector.load %view[%640] : memref<4096xf16, 3>, vector<4xf16>
      %796 = vector.load %view[%642] : memref<4096xf16, 3>, vector<4xf16>
      %797 = vector.extract %795[0] : f16 from vector<4xf16>
      %798 = vector.extract %795[1] : f16 from vector<4xf16>
      %799 = vector.extract %795[2] : f16 from vector<4xf16>
      %800 = vector.extract %795[3] : f16 from vector<4xf16>
      %801 = vector.from_elements %797, %798, %799, %800 : vector<4xf16>
      %802 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %599, %801, %790 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %803 = vector.extract %796[0] : f16 from vector<4xf16>
      %804 = vector.extract %796[1] : f16 from vector<4xf16>
      %805 = vector.extract %796[2] : f16 from vector<4xf16>
      %806 = vector.extract %796[3] : f16 from vector<4xf16>
      %807 = vector.from_elements %803, %804, %805, %806 : vector<4xf16>
      %808 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %598, %807, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %809 = vector.extract %802[0] : f32 from vector<4xf32>
      %810 = vector.extract %802[1] : f32 from vector<4xf32>
      %811 = vector.extract %802[2] : f32 from vector<4xf32>
      %812 = vector.extract %802[3] : f32 from vector<4xf32>
      %813 = vector.load %view[%660] : memref<4096xf16, 3>, vector<4xf16>
      %814 = vector.load %view[%662] : memref<4096xf16, 3>, vector<4xf16>
      %815 = vector.extract %813[0] : f16 from vector<4xf16>
      %816 = vector.extract %813[1] : f16 from vector<4xf16>
      %817 = vector.extract %813[2] : f16 from vector<4xf16>
      %818 = vector.extract %813[3] : f16 from vector<4xf16>
      %819 = vector.from_elements %815, %816, %817, %818 : vector<4xf16>
      %820 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %599, %819, %808 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %821 = vector.extract %814[0] : f16 from vector<4xf16>
      %822 = vector.extract %814[1] : f16 from vector<4xf16>
      %823 = vector.extract %814[2] : f16 from vector<4xf16>
      %824 = vector.extract %814[3] : f16 from vector<4xf16>
      %825 = vector.from_elements %821, %822, %823, %824 : vector<4xf16>
      %826 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %598, %825, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %827 = vector.extract %820[0] : f32 from vector<4xf32>
      %828 = vector.extract %820[1] : f32 from vector<4xf32>
      %829 = vector.extract %820[2] : f32 from vector<4xf32>
      %830 = vector.extract %820[3] : f32 from vector<4xf32>
      %831 = vector.load %view[%680] : memref<4096xf16, 3>, vector<4xf16>
      %832 = vector.load %view[%682] : memref<4096xf16, 3>, vector<4xf16>
      %833 = vector.extract %831[0] : f16 from vector<4xf16>
      %834 = vector.extract %831[1] : f16 from vector<4xf16>
      %835 = vector.extract %831[2] : f16 from vector<4xf16>
      %836 = vector.extract %831[3] : f16 from vector<4xf16>
      %837 = vector.from_elements %833, %834, %835, %836 : vector<4xf16>
      %838 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %599, %837, %826 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %839 = vector.extract %832[0] : f16 from vector<4xf16>
      %840 = vector.extract %832[1] : f16 from vector<4xf16>
      %841 = vector.extract %832[2] : f16 from vector<4xf16>
      %842 = vector.extract %832[3] : f16 from vector<4xf16>
      %843 = vector.from_elements %839, %840, %841, %842 : vector<4xf16>
      %844 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %598, %843, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %845 = vector.extract %838[0] : f32 from vector<4xf32>
      %846 = vector.extract %838[1] : f32 from vector<4xf32>
      %847 = vector.extract %838[2] : f32 from vector<4xf32>
      %848 = vector.extract %838[3] : f32 from vector<4xf32>
      %849 = vector.load %view[%700] : memref<4096xf16, 3>, vector<4xf16>
      %850 = vector.load %view[%702] : memref<4096xf16, 3>, vector<4xf16>
      %851 = vector.extract %849[0] : f16 from vector<4xf16>
      %852 = vector.extract %849[1] : f16 from vector<4xf16>
      %853 = vector.extract %849[2] : f16 from vector<4xf16>
      %854 = vector.extract %849[3] : f16 from vector<4xf16>
      %855 = vector.from_elements %851, %852, %853, %854 : vector<4xf16>
      %856 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %599, %855, %844 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %857 = vector.extract %850[0] : f16 from vector<4xf16>
      %858 = vector.extract %850[1] : f16 from vector<4xf16>
      %859 = vector.extract %850[2] : f16 from vector<4xf16>
      %860 = vector.extract %850[3] : f16 from vector<4xf16>
      %861 = vector.from_elements %857, %858, %859, %860 : vector<4xf16>
      %862 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %598, %861, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %863 = vector.extract %856[0] : f32 from vector<4xf32>
      %864 = vector.extract %856[1] : f32 from vector<4xf32>
      %865 = vector.extract %856[2] : f32 from vector<4xf32>
      %866 = vector.extract %856[3] : f32 from vector<4xf32>
      %867 = vector.load %view[%720] : memref<4096xf16, 3>, vector<4xf16>
      %868 = vector.load %view[%722] : memref<4096xf16, 3>, vector<4xf16>
      %869 = vector.extract %867[0] : f16 from vector<4xf16>
      %870 = vector.extract %867[1] : f16 from vector<4xf16>
      %871 = vector.extract %867[2] : f16 from vector<4xf16>
      %872 = vector.extract %867[3] : f16 from vector<4xf16>
      %873 = vector.from_elements %869, %870, %871, %872 : vector<4xf16>
      %874 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %599, %873, %862 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %875 = vector.extract %868[0] : f16 from vector<4xf16>
      %876 = vector.extract %868[1] : f16 from vector<4xf16>
      %877 = vector.extract %868[2] : f16 from vector<4xf16>
      %878 = vector.extract %868[3] : f16 from vector<4xf16>
      %879 = vector.from_elements %875, %876, %877, %878 : vector<4xf16>
      %880 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %598, %879, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %881 = vector.extract %874[0] : f32 from vector<4xf32>
      %882 = vector.extract %874[1] : f32 from vector<4xf32>
      %883 = vector.extract %874[2] : f32 from vector<4xf32>
      %884 = vector.extract %874[3] : f32 from vector<4xf32>
      %885 = vector.load %view[%740] : memref<4096xf16, 3>, vector<4xf16>
      %886 = vector.extract %885[0] : f16 from vector<4xf16>
      %887 = vector.extract %885[1] : f16 from vector<4xf16>
      %888 = vector.extract %885[2] : f16 from vector<4xf16>
      %889 = vector.extract %885[3] : f16 from vector<4xf16>
      %890 = vector.from_elements %886, %887, %888, %889 : vector<4xf16>
      %891 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %599, %890, %880 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %892 = vector.extract %891[0] : f32 from vector<4xf32>
      %893 = vector.extract %891[1] : f32 from vector<4xf32>
      %894 = vector.extract %891[2] : f32 from vector<4xf32>
      %895 = vector.extract %891[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %587 : !frisk.async_task
      %subview_139 = memref.subview %transpose[0, %198, 64, %585] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %896 = frisk.async_copy_gm_to_shm %subview_139 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %897 = vector.load %view_7[%588] : memref<4096xf16, 3>, vector<4xf16>
      %898 = vector.from_elements %85, %86, %87, %88 : vector<4xf16>
      %899 = vector.from_elements %616, %617, %618, %619 : vector<4xf32>
      %900 = vector.from_elements %89, %90, %91, %92 : vector<4xf16>
      %901 = vector.from_elements %636, %637, %638, %639 : vector<4xf32>
      %902 = vector.extract %897[0] : f16 from vector<4xf16>
      %903 = vector.extract %897[1] : f16 from vector<4xf16>
      %904 = vector.extract %897[2] : f16 from vector<4xf16>
      %905 = vector.extract %897[3] : f16 from vector<4xf16>
      %906 = vector.from_elements %902, %903, %904, %905 : vector<4xf16>
      %907 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %906, %899 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %908 = vector.from_elements %656, %657, %658, %659 : vector<4xf32>
      %909 = vector.from_elements %676, %677, %678, %679 : vector<4xf32>
      %910 = vector.from_elements %696, %697, %698, %699 : vector<4xf32>
      %911 = vector.from_elements %716, %717, %718, %719 : vector<4xf32>
      %912 = vector.from_elements %736, %737, %738, %739 : vector<4xf32>
      %913 = vector.from_elements %755, %756, %757, %758 : vector<4xf32>
      %914 = vector.from_elements %101, %102, %103, %104 : vector<4xf16>
      %915 = vector.from_elements %773, %774, %775, %776 : vector<4xf32>
      %916 = vector.from_elements %105, %106, %107, %108 : vector<4xf16>
      %917 = vector.from_elements %791, %792, %793, %794 : vector<4xf32>
      %918 = vector.from_elements %809, %810, %811, %812 : vector<4xf32>
      %919 = vector.from_elements %827, %828, %829, %830 : vector<4xf32>
      %920 = vector.from_elements %845, %846, %847, %848 : vector<4xf32>
      %921 = vector.from_elements %863, %864, %865, %866 : vector<4xf32>
      %922 = vector.from_elements %881, %882, %883, %884 : vector<4xf32>
      %923 = vector.from_elements %892, %893, %894, %895 : vector<4xf32>
      %924 = vector.load %view_7[%600] : memref<4096xf16, 3>, vector<4xf16>
      %925 = vector.load %view_7[%602] : memref<4096xf16, 3>, vector<4xf16>
      %926 = vector.extract %924[0] : f16 from vector<4xf16>
      %927 = vector.extract %924[1] : f16 from vector<4xf16>
      %928 = vector.extract %924[2] : f16 from vector<4xf16>
      %929 = vector.extract %924[3] : f16 from vector<4xf16>
      %930 = vector.from_elements %926, %927, %928, %929 : vector<4xf16>
      %931 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %900, %930, %907 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %932 = vector.extract %925[0] : f16 from vector<4xf16>
      %933 = vector.extract %925[1] : f16 from vector<4xf16>
      %934 = vector.extract %925[2] : f16 from vector<4xf16>
      %935 = vector.extract %925[3] : f16 from vector<4xf16>
      %936 = vector.from_elements %932, %933, %934, %935 : vector<4xf16>
      %937 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %936, %901 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %938 = vector.extract %931[0] : f32 from vector<4xf32>
      %939 = vector.extract %931[1] : f32 from vector<4xf32>
      %940 = vector.extract %931[2] : f32 from vector<4xf32>
      %941 = vector.extract %931[3] : f32 from vector<4xf32>
      %942 = vector.load %view_7[%620] : memref<4096xf16, 3>, vector<4xf16>
      %943 = vector.load %view_7[%622] : memref<4096xf16, 3>, vector<4xf16>
      %944 = vector.extract %942[0] : f16 from vector<4xf16>
      %945 = vector.extract %942[1] : f16 from vector<4xf16>
      %946 = vector.extract %942[2] : f16 from vector<4xf16>
      %947 = vector.extract %942[3] : f16 from vector<4xf16>
      %948 = vector.from_elements %944, %945, %946, %947 : vector<4xf16>
      %949 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %900, %948, %937 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %950 = vector.extract %943[0] : f16 from vector<4xf16>
      %951 = vector.extract %943[1] : f16 from vector<4xf16>
      %952 = vector.extract %943[2] : f16 from vector<4xf16>
      %953 = vector.extract %943[3] : f16 from vector<4xf16>
      %954 = vector.from_elements %950, %951, %952, %953 : vector<4xf16>
      %955 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %954, %908 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %956 = vector.extract %949[0] : f32 from vector<4xf32>
      %957 = vector.extract %949[1] : f32 from vector<4xf32>
      %958 = vector.extract %949[2] : f32 from vector<4xf32>
      %959 = vector.extract %949[3] : f32 from vector<4xf32>
      %960 = vector.load %view_7[%640] : memref<4096xf16, 3>, vector<4xf16>
      %961 = vector.load %view_7[%642] : memref<4096xf16, 3>, vector<4xf16>
      %962 = vector.extract %960[0] : f16 from vector<4xf16>
      %963 = vector.extract %960[1] : f16 from vector<4xf16>
      %964 = vector.extract %960[2] : f16 from vector<4xf16>
      %965 = vector.extract %960[3] : f16 from vector<4xf16>
      %966 = vector.from_elements %962, %963, %964, %965 : vector<4xf16>
      %967 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %900, %966, %955 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %968 = vector.extract %961[0] : f16 from vector<4xf16>
      %969 = vector.extract %961[1] : f16 from vector<4xf16>
      %970 = vector.extract %961[2] : f16 from vector<4xf16>
      %971 = vector.extract %961[3] : f16 from vector<4xf16>
      %972 = vector.from_elements %968, %969, %970, %971 : vector<4xf16>
      %973 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %972, %909 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %974 = vector.extract %967[0] : f32 from vector<4xf32>
      %975 = vector.extract %967[1] : f32 from vector<4xf32>
      %976 = vector.extract %967[2] : f32 from vector<4xf32>
      %977 = vector.extract %967[3] : f32 from vector<4xf32>
      %978 = vector.load %view_7[%660] : memref<4096xf16, 3>, vector<4xf16>
      %979 = vector.load %view_7[%662] : memref<4096xf16, 3>, vector<4xf16>
      %980 = vector.extract %978[0] : f16 from vector<4xf16>
      %981 = vector.extract %978[1] : f16 from vector<4xf16>
      %982 = vector.extract %978[2] : f16 from vector<4xf16>
      %983 = vector.extract %978[3] : f16 from vector<4xf16>
      %984 = vector.from_elements %980, %981, %982, %983 : vector<4xf16>
      %985 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %900, %984, %973 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %986 = vector.extract %979[0] : f16 from vector<4xf16>
      %987 = vector.extract %979[1] : f16 from vector<4xf16>
      %988 = vector.extract %979[2] : f16 from vector<4xf16>
      %989 = vector.extract %979[3] : f16 from vector<4xf16>
      %990 = vector.from_elements %986, %987, %988, %989 : vector<4xf16>
      %991 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %990, %910 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %992 = vector.extract %985[0] : f32 from vector<4xf32>
      %993 = vector.extract %985[1] : f32 from vector<4xf32>
      %994 = vector.extract %985[2] : f32 from vector<4xf32>
      %995 = vector.extract %985[3] : f32 from vector<4xf32>
      %996 = vector.load %view_7[%680] : memref<4096xf16, 3>, vector<4xf16>
      %997 = vector.load %view_7[%682] : memref<4096xf16, 3>, vector<4xf16>
      %998 = vector.extract %996[0] : f16 from vector<4xf16>
      %999 = vector.extract %996[1] : f16 from vector<4xf16>
      %1000 = vector.extract %996[2] : f16 from vector<4xf16>
      %1001 = vector.extract %996[3] : f16 from vector<4xf16>
      %1002 = vector.from_elements %998, %999, %1000, %1001 : vector<4xf16>
      %1003 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %900, %1002, %991 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1004 = vector.extract %997[0] : f16 from vector<4xf16>
      %1005 = vector.extract %997[1] : f16 from vector<4xf16>
      %1006 = vector.extract %997[2] : f16 from vector<4xf16>
      %1007 = vector.extract %997[3] : f16 from vector<4xf16>
      %1008 = vector.from_elements %1004, %1005, %1006, %1007 : vector<4xf16>
      %1009 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %1008, %911 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1010 = vector.extract %1003[0] : f32 from vector<4xf32>
      %1011 = vector.extract %1003[1] : f32 from vector<4xf32>
      %1012 = vector.extract %1003[2] : f32 from vector<4xf32>
      %1013 = vector.extract %1003[3] : f32 from vector<4xf32>
      %1014 = vector.load %view_7[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1015 = vector.load %view_7[%702] : memref<4096xf16, 3>, vector<4xf16>
      %1016 = vector.extract %1014[0] : f16 from vector<4xf16>
      %1017 = vector.extract %1014[1] : f16 from vector<4xf16>
      %1018 = vector.extract %1014[2] : f16 from vector<4xf16>
      %1019 = vector.extract %1014[3] : f16 from vector<4xf16>
      %1020 = vector.from_elements %1016, %1017, %1018, %1019 : vector<4xf16>
      %1021 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %900, %1020, %1009 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1022 = vector.extract %1015[0] : f16 from vector<4xf16>
      %1023 = vector.extract %1015[1] : f16 from vector<4xf16>
      %1024 = vector.extract %1015[2] : f16 from vector<4xf16>
      %1025 = vector.extract %1015[3] : f16 from vector<4xf16>
      %1026 = vector.from_elements %1022, %1023, %1024, %1025 : vector<4xf16>
      %1027 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %1026, %912 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1028 = vector.extract %1021[0] : f32 from vector<4xf32>
      %1029 = vector.extract %1021[1] : f32 from vector<4xf32>
      %1030 = vector.extract %1021[2] : f32 from vector<4xf32>
      %1031 = vector.extract %1021[3] : f32 from vector<4xf32>
      %1032 = vector.load %view_7[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1033 = vector.load %view_7[%722] : memref<4096xf16, 3>, vector<4xf16>
      %1034 = vector.extract %1032[0] : f16 from vector<4xf16>
      %1035 = vector.extract %1032[1] : f16 from vector<4xf16>
      %1036 = vector.extract %1032[2] : f16 from vector<4xf16>
      %1037 = vector.extract %1032[3] : f16 from vector<4xf16>
      %1038 = vector.from_elements %1034, %1035, %1036, %1037 : vector<4xf16>
      %1039 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %900, %1038, %1027 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1040 = vector.extract %1033[0] : f16 from vector<4xf16>
      %1041 = vector.extract %1033[1] : f16 from vector<4xf16>
      %1042 = vector.extract %1033[2] : f16 from vector<4xf16>
      %1043 = vector.extract %1033[3] : f16 from vector<4xf16>
      %1044 = vector.from_elements %1040, %1041, %1042, %1043 : vector<4xf16>
      %1045 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %1044, %913 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1046 = vector.extract %1039[0] : f32 from vector<4xf32>
      %1047 = vector.extract %1039[1] : f32 from vector<4xf32>
      %1048 = vector.extract %1039[2] : f32 from vector<4xf32>
      %1049 = vector.extract %1039[3] : f32 from vector<4xf32>
      %1050 = vector.load %view_7[%740] : memref<4096xf16, 3>, vector<4xf16>
      %1051 = vector.load %view_7[%588] : memref<4096xf16, 3>, vector<4xf16>
      %1052 = vector.extract %1050[0] : f16 from vector<4xf16>
      %1053 = vector.extract %1050[1] : f16 from vector<4xf16>
      %1054 = vector.extract %1050[2] : f16 from vector<4xf16>
      %1055 = vector.extract %1050[3] : f16 from vector<4xf16>
      %1056 = vector.from_elements %1052, %1053, %1054, %1055 : vector<4xf16>
      %1057 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %900, %1056, %1045 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1058 = vector.extract %1051[0] : f16 from vector<4xf16>
      %1059 = vector.extract %1051[1] : f16 from vector<4xf16>
      %1060 = vector.extract %1051[2] : f16 from vector<4xf16>
      %1061 = vector.extract %1051[3] : f16 from vector<4xf16>
      %1062 = vector.from_elements %1058, %1059, %1060, %1061 : vector<4xf16>
      %1063 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1062, %915 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1064 = vector.extract %1057[0] : f32 from vector<4xf32>
      %1065 = vector.extract %1057[1] : f32 from vector<4xf32>
      %1066 = vector.extract %1057[2] : f32 from vector<4xf32>
      %1067 = vector.extract %1057[3] : f32 from vector<4xf32>
      %1068 = vector.load %view_7[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1069 = vector.load %view_7[%602] : memref<4096xf16, 3>, vector<4xf16>
      %1070 = vector.extract %1068[0] : f16 from vector<4xf16>
      %1071 = vector.extract %1068[1] : f16 from vector<4xf16>
      %1072 = vector.extract %1068[2] : f16 from vector<4xf16>
      %1073 = vector.extract %1068[3] : f16 from vector<4xf16>
      %1074 = vector.from_elements %1070, %1071, %1072, %1073 : vector<4xf16>
      %1075 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %916, %1074, %1063 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1076 = vector.extract %1069[0] : f16 from vector<4xf16>
      %1077 = vector.extract %1069[1] : f16 from vector<4xf16>
      %1078 = vector.extract %1069[2] : f16 from vector<4xf16>
      %1079 = vector.extract %1069[3] : f16 from vector<4xf16>
      %1080 = vector.from_elements %1076, %1077, %1078, %1079 : vector<4xf16>
      %1081 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1080, %917 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1082 = vector.extract %1075[0] : f32 from vector<4xf32>
      %1083 = vector.extract %1075[1] : f32 from vector<4xf32>
      %1084 = vector.extract %1075[2] : f32 from vector<4xf32>
      %1085 = vector.extract %1075[3] : f32 from vector<4xf32>
      %1086 = vector.load %view_7[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1087 = vector.load %view_7[%622] : memref<4096xf16, 3>, vector<4xf16>
      %1088 = vector.extract %1086[0] : f16 from vector<4xf16>
      %1089 = vector.extract %1086[1] : f16 from vector<4xf16>
      %1090 = vector.extract %1086[2] : f16 from vector<4xf16>
      %1091 = vector.extract %1086[3] : f16 from vector<4xf16>
      %1092 = vector.from_elements %1088, %1089, %1090, %1091 : vector<4xf16>
      %1093 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %916, %1092, %1081 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1094 = vector.extract %1087[0] : f16 from vector<4xf16>
      %1095 = vector.extract %1087[1] : f16 from vector<4xf16>
      %1096 = vector.extract %1087[2] : f16 from vector<4xf16>
      %1097 = vector.extract %1087[3] : f16 from vector<4xf16>
      %1098 = vector.from_elements %1094, %1095, %1096, %1097 : vector<4xf16>
      %1099 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1098, %918 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1100 = vector.extract %1093[0] : f32 from vector<4xf32>
      %1101 = vector.extract %1093[1] : f32 from vector<4xf32>
      %1102 = vector.extract %1093[2] : f32 from vector<4xf32>
      %1103 = vector.extract %1093[3] : f32 from vector<4xf32>
      %1104 = vector.load %view_7[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1105 = vector.load %view_7[%642] : memref<4096xf16, 3>, vector<4xf16>
      %1106 = vector.extract %1104[0] : f16 from vector<4xf16>
      %1107 = vector.extract %1104[1] : f16 from vector<4xf16>
      %1108 = vector.extract %1104[2] : f16 from vector<4xf16>
      %1109 = vector.extract %1104[3] : f16 from vector<4xf16>
      %1110 = vector.from_elements %1106, %1107, %1108, %1109 : vector<4xf16>
      %1111 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %916, %1110, %1099 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1112 = vector.extract %1105[0] : f16 from vector<4xf16>
      %1113 = vector.extract %1105[1] : f16 from vector<4xf16>
      %1114 = vector.extract %1105[2] : f16 from vector<4xf16>
      %1115 = vector.extract %1105[3] : f16 from vector<4xf16>
      %1116 = vector.from_elements %1112, %1113, %1114, %1115 : vector<4xf16>
      %1117 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1116, %919 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1118 = vector.extract %1111[0] : f32 from vector<4xf32>
      %1119 = vector.extract %1111[1] : f32 from vector<4xf32>
      %1120 = vector.extract %1111[2] : f32 from vector<4xf32>
      %1121 = vector.extract %1111[3] : f32 from vector<4xf32>
      %1122 = vector.load %view_7[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1123 = vector.load %view_7[%662] : memref<4096xf16, 3>, vector<4xf16>
      %1124 = vector.extract %1122[0] : f16 from vector<4xf16>
      %1125 = vector.extract %1122[1] : f16 from vector<4xf16>
      %1126 = vector.extract %1122[2] : f16 from vector<4xf16>
      %1127 = vector.extract %1122[3] : f16 from vector<4xf16>
      %1128 = vector.from_elements %1124, %1125, %1126, %1127 : vector<4xf16>
      %1129 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %916, %1128, %1117 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1130 = vector.extract %1123[0] : f16 from vector<4xf16>
      %1131 = vector.extract %1123[1] : f16 from vector<4xf16>
      %1132 = vector.extract %1123[2] : f16 from vector<4xf16>
      %1133 = vector.extract %1123[3] : f16 from vector<4xf16>
      %1134 = vector.from_elements %1130, %1131, %1132, %1133 : vector<4xf16>
      %1135 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1134, %920 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1136 = vector.extract %1129[0] : f32 from vector<4xf32>
      %1137 = vector.extract %1129[1] : f32 from vector<4xf32>
      %1138 = vector.extract %1129[2] : f32 from vector<4xf32>
      %1139 = vector.extract %1129[3] : f32 from vector<4xf32>
      %1140 = vector.load %view_7[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1141 = vector.load %view_7[%682] : memref<4096xf16, 3>, vector<4xf16>
      %1142 = vector.extract %1140[0] : f16 from vector<4xf16>
      %1143 = vector.extract %1140[1] : f16 from vector<4xf16>
      %1144 = vector.extract %1140[2] : f16 from vector<4xf16>
      %1145 = vector.extract %1140[3] : f16 from vector<4xf16>
      %1146 = vector.from_elements %1142, %1143, %1144, %1145 : vector<4xf16>
      %1147 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %916, %1146, %1135 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1148 = vector.extract %1141[0] : f16 from vector<4xf16>
      %1149 = vector.extract %1141[1] : f16 from vector<4xf16>
      %1150 = vector.extract %1141[2] : f16 from vector<4xf16>
      %1151 = vector.extract %1141[3] : f16 from vector<4xf16>
      %1152 = vector.from_elements %1148, %1149, %1150, %1151 : vector<4xf16>
      %1153 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1152, %921 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1154 = vector.extract %1147[0] : f32 from vector<4xf32>
      %1155 = vector.extract %1147[1] : f32 from vector<4xf32>
      %1156 = vector.extract %1147[2] : f32 from vector<4xf32>
      %1157 = vector.extract %1147[3] : f32 from vector<4xf32>
      %1158 = vector.load %view_7[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1159 = vector.load %view_7[%702] : memref<4096xf16, 3>, vector<4xf16>
      %1160 = vector.extract %1158[0] : f16 from vector<4xf16>
      %1161 = vector.extract %1158[1] : f16 from vector<4xf16>
      %1162 = vector.extract %1158[2] : f16 from vector<4xf16>
      %1163 = vector.extract %1158[3] : f16 from vector<4xf16>
      %1164 = vector.from_elements %1160, %1161, %1162, %1163 : vector<4xf16>
      %1165 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %916, %1164, %1153 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1166 = vector.extract %1159[0] : f16 from vector<4xf16>
      %1167 = vector.extract %1159[1] : f16 from vector<4xf16>
      %1168 = vector.extract %1159[2] : f16 from vector<4xf16>
      %1169 = vector.extract %1159[3] : f16 from vector<4xf16>
      %1170 = vector.from_elements %1166, %1167, %1168, %1169 : vector<4xf16>
      %1171 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1170, %922 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1172 = vector.extract %1165[0] : f32 from vector<4xf32>
      %1173 = vector.extract %1165[1] : f32 from vector<4xf32>
      %1174 = vector.extract %1165[2] : f32 from vector<4xf32>
      %1175 = vector.extract %1165[3] : f32 from vector<4xf32>
      %1176 = vector.load %view_7[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1177 = vector.load %view_7[%722] : memref<4096xf16, 3>, vector<4xf16>
      %1178 = vector.extract %1176[0] : f16 from vector<4xf16>
      %1179 = vector.extract %1176[1] : f16 from vector<4xf16>
      %1180 = vector.extract %1176[2] : f16 from vector<4xf16>
      %1181 = vector.extract %1176[3] : f16 from vector<4xf16>
      %1182 = vector.from_elements %1178, %1179, %1180, %1181 : vector<4xf16>
      %1183 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %916, %1182, %1171 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1184 = vector.extract %1177[0] : f16 from vector<4xf16>
      %1185 = vector.extract %1177[1] : f16 from vector<4xf16>
      %1186 = vector.extract %1177[2] : f16 from vector<4xf16>
      %1187 = vector.extract %1177[3] : f16 from vector<4xf16>
      %1188 = vector.from_elements %1184, %1185, %1186, %1187 : vector<4xf16>
      %1189 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1188, %923 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1190 = vector.extract %1183[0] : f32 from vector<4xf32>
      %1191 = vector.extract %1183[1] : f32 from vector<4xf32>
      %1192 = vector.extract %1183[2] : f32 from vector<4xf32>
      %1193 = vector.extract %1183[3] : f32 from vector<4xf32>
      %1194 = vector.load %view_7[%740] : memref<4096xf16, 3>, vector<4xf16>
      %1195 = vector.extract %1194[0] : f16 from vector<4xf16>
      %1196 = vector.extract %1194[1] : f16 from vector<4xf16>
      %1197 = vector.extract %1194[2] : f16 from vector<4xf16>
      %1198 = vector.extract %1194[3] : f16 from vector<4xf16>
      %1199 = vector.from_elements %1195, %1196, %1197, %1198 : vector<4xf16>
      %1200 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %916, %1199, %1189 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1201 = vector.extract %1200[0] : f32 from vector<4xf32>
      %1202 = vector.extract %1200[1] : f32 from vector<4xf32>
      %1203 = vector.extract %1200[2] : f32 from vector<4xf32>
      %1204 = vector.extract %1200[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %896 : !frisk.async_task
      %subview_140 = memref.subview %transpose[0, %198, 96, %585] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %1205 = frisk.async_copy_gm_to_shm %subview_140 to %view_7 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %1206 = vector.load %view[%588] : memref<4096xf16, 3>, vector<4xf16>
      %1207 = vector.from_elements %130, %131, %132, %133 : vector<4xf16>
      %1208 = vector.from_elements %938, %939, %940, %941 : vector<4xf32>
      %1209 = vector.from_elements %134, %135, %136, %137 : vector<4xf16>
      %1210 = vector.from_elements %956, %957, %958, %959 : vector<4xf32>
      %1211 = vector.extract %1206[0] : f16 from vector<4xf16>
      %1212 = vector.extract %1206[1] : f16 from vector<4xf16>
      %1213 = vector.extract %1206[2] : f16 from vector<4xf16>
      %1214 = vector.extract %1206[3] : f16 from vector<4xf16>
      %1215 = vector.from_elements %1211, %1212, %1213, %1214 : vector<4xf16>
      %1216 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1215, %1208 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1217 = vector.from_elements %974, %975, %976, %977 : vector<4xf32>
      %1218 = vector.from_elements %992, %993, %994, %995 : vector<4xf32>
      %1219 = vector.from_elements %1010, %1011, %1012, %1013 : vector<4xf32>
      %1220 = vector.from_elements %1028, %1029, %1030, %1031 : vector<4xf32>
      %1221 = vector.from_elements %1046, %1047, %1048, %1049 : vector<4xf32>
      %1222 = vector.from_elements %1064, %1065, %1066, %1067 : vector<4xf32>
      %1223 = vector.from_elements %146, %147, %148, %149 : vector<4xf16>
      %1224 = vector.from_elements %1082, %1083, %1084, %1085 : vector<4xf32>
      %1225 = vector.from_elements %150, %151, %152, %153 : vector<4xf16>
      %1226 = vector.from_elements %1100, %1101, %1102, %1103 : vector<4xf32>
      %1227 = vector.from_elements %1118, %1119, %1120, %1121 : vector<4xf32>
      %1228 = vector.from_elements %1136, %1137, %1138, %1139 : vector<4xf32>
      %1229 = vector.from_elements %1154, %1155, %1156, %1157 : vector<4xf32>
      %1230 = vector.from_elements %1172, %1173, %1174, %1175 : vector<4xf32>
      %1231 = vector.from_elements %1190, %1191, %1192, %1193 : vector<4xf32>
      %1232 = vector.from_elements %1201, %1202, %1203, %1204 : vector<4xf32>
      %1233 = vector.load %view[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1234 = vector.load %view[%602] : memref<4096xf16, 3>, vector<4xf16>
      %1235 = vector.extract %1233[0] : f16 from vector<4xf16>
      %1236 = vector.extract %1233[1] : f16 from vector<4xf16>
      %1237 = vector.extract %1233[2] : f16 from vector<4xf16>
      %1238 = vector.extract %1233[3] : f16 from vector<4xf16>
      %1239 = vector.from_elements %1235, %1236, %1237, %1238 : vector<4xf16>
      %1240 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1209, %1239, %1216 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1241 = vector.extract %1234[0] : f16 from vector<4xf16>
      %1242 = vector.extract %1234[1] : f16 from vector<4xf16>
      %1243 = vector.extract %1234[2] : f16 from vector<4xf16>
      %1244 = vector.extract %1234[3] : f16 from vector<4xf16>
      %1245 = vector.from_elements %1241, %1242, %1243, %1244 : vector<4xf16>
      %1246 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1245, %1210 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1247 = vector.extract %1240[0] : f32 from vector<4xf32>
      %1248 = vector.extract %1240[1] : f32 from vector<4xf32>
      %1249 = vector.extract %1240[2] : f32 from vector<4xf32>
      %1250 = vector.extract %1240[3] : f32 from vector<4xf32>
      %1251 = vector.load %view[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1252 = vector.load %view[%622] : memref<4096xf16, 3>, vector<4xf16>
      %1253 = vector.extract %1251[0] : f16 from vector<4xf16>
      %1254 = vector.extract %1251[1] : f16 from vector<4xf16>
      %1255 = vector.extract %1251[2] : f16 from vector<4xf16>
      %1256 = vector.extract %1251[3] : f16 from vector<4xf16>
      %1257 = vector.from_elements %1253, %1254, %1255, %1256 : vector<4xf16>
      %1258 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1209, %1257, %1246 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1259 = vector.extract %1252[0] : f16 from vector<4xf16>
      %1260 = vector.extract %1252[1] : f16 from vector<4xf16>
      %1261 = vector.extract %1252[2] : f16 from vector<4xf16>
      %1262 = vector.extract %1252[3] : f16 from vector<4xf16>
      %1263 = vector.from_elements %1259, %1260, %1261, %1262 : vector<4xf16>
      %1264 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1263, %1217 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1265 = vector.extract %1258[0] : f32 from vector<4xf32>
      %1266 = vector.extract %1258[1] : f32 from vector<4xf32>
      %1267 = vector.extract %1258[2] : f32 from vector<4xf32>
      %1268 = vector.extract %1258[3] : f32 from vector<4xf32>
      %1269 = vector.load %view[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1270 = vector.load %view[%642] : memref<4096xf16, 3>, vector<4xf16>
      %1271 = vector.extract %1269[0] : f16 from vector<4xf16>
      %1272 = vector.extract %1269[1] : f16 from vector<4xf16>
      %1273 = vector.extract %1269[2] : f16 from vector<4xf16>
      %1274 = vector.extract %1269[3] : f16 from vector<4xf16>
      %1275 = vector.from_elements %1271, %1272, %1273, %1274 : vector<4xf16>
      %1276 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1209, %1275, %1264 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1277 = vector.extract %1270[0] : f16 from vector<4xf16>
      %1278 = vector.extract %1270[1] : f16 from vector<4xf16>
      %1279 = vector.extract %1270[2] : f16 from vector<4xf16>
      %1280 = vector.extract %1270[3] : f16 from vector<4xf16>
      %1281 = vector.from_elements %1277, %1278, %1279, %1280 : vector<4xf16>
      %1282 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1281, %1218 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1283 = vector.extract %1276[0] : f32 from vector<4xf32>
      %1284 = vector.extract %1276[1] : f32 from vector<4xf32>
      %1285 = vector.extract %1276[2] : f32 from vector<4xf32>
      %1286 = vector.extract %1276[3] : f32 from vector<4xf32>
      %1287 = vector.load %view[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1288 = vector.load %view[%662] : memref<4096xf16, 3>, vector<4xf16>
      %1289 = vector.extract %1287[0] : f16 from vector<4xf16>
      %1290 = vector.extract %1287[1] : f16 from vector<4xf16>
      %1291 = vector.extract %1287[2] : f16 from vector<4xf16>
      %1292 = vector.extract %1287[3] : f16 from vector<4xf16>
      %1293 = vector.from_elements %1289, %1290, %1291, %1292 : vector<4xf16>
      %1294 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1209, %1293, %1282 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1295 = vector.extract %1288[0] : f16 from vector<4xf16>
      %1296 = vector.extract %1288[1] : f16 from vector<4xf16>
      %1297 = vector.extract %1288[2] : f16 from vector<4xf16>
      %1298 = vector.extract %1288[3] : f16 from vector<4xf16>
      %1299 = vector.from_elements %1295, %1296, %1297, %1298 : vector<4xf16>
      %1300 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1299, %1219 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1301 = vector.extract %1294[0] : f32 from vector<4xf32>
      %1302 = vector.extract %1294[1] : f32 from vector<4xf32>
      %1303 = vector.extract %1294[2] : f32 from vector<4xf32>
      %1304 = vector.extract %1294[3] : f32 from vector<4xf32>
      %1305 = vector.load %view[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1306 = vector.load %view[%682] : memref<4096xf16, 3>, vector<4xf16>
      %1307 = vector.extract %1305[0] : f16 from vector<4xf16>
      %1308 = vector.extract %1305[1] : f16 from vector<4xf16>
      %1309 = vector.extract %1305[2] : f16 from vector<4xf16>
      %1310 = vector.extract %1305[3] : f16 from vector<4xf16>
      %1311 = vector.from_elements %1307, %1308, %1309, %1310 : vector<4xf16>
      %1312 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1209, %1311, %1300 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1313 = vector.extract %1306[0] : f16 from vector<4xf16>
      %1314 = vector.extract %1306[1] : f16 from vector<4xf16>
      %1315 = vector.extract %1306[2] : f16 from vector<4xf16>
      %1316 = vector.extract %1306[3] : f16 from vector<4xf16>
      %1317 = vector.from_elements %1313, %1314, %1315, %1316 : vector<4xf16>
      %1318 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1317, %1220 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1319 = vector.extract %1312[0] : f32 from vector<4xf32>
      %1320 = vector.extract %1312[1] : f32 from vector<4xf32>
      %1321 = vector.extract %1312[2] : f32 from vector<4xf32>
      %1322 = vector.extract %1312[3] : f32 from vector<4xf32>
      %1323 = vector.load %view[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1324 = vector.load %view[%702] : memref<4096xf16, 3>, vector<4xf16>
      %1325 = vector.extract %1323[0] : f16 from vector<4xf16>
      %1326 = vector.extract %1323[1] : f16 from vector<4xf16>
      %1327 = vector.extract %1323[2] : f16 from vector<4xf16>
      %1328 = vector.extract %1323[3] : f16 from vector<4xf16>
      %1329 = vector.from_elements %1325, %1326, %1327, %1328 : vector<4xf16>
      %1330 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1209, %1329, %1318 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1331 = vector.extract %1324[0] : f16 from vector<4xf16>
      %1332 = vector.extract %1324[1] : f16 from vector<4xf16>
      %1333 = vector.extract %1324[2] : f16 from vector<4xf16>
      %1334 = vector.extract %1324[3] : f16 from vector<4xf16>
      %1335 = vector.from_elements %1331, %1332, %1333, %1334 : vector<4xf16>
      %1336 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1335, %1221 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1337 = vector.extract %1330[0] : f32 from vector<4xf32>
      %1338 = vector.extract %1330[1] : f32 from vector<4xf32>
      %1339 = vector.extract %1330[2] : f32 from vector<4xf32>
      %1340 = vector.extract %1330[3] : f32 from vector<4xf32>
      %1341 = vector.load %view[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1342 = vector.load %view[%722] : memref<4096xf16, 3>, vector<4xf16>
      %1343 = vector.extract %1341[0] : f16 from vector<4xf16>
      %1344 = vector.extract %1341[1] : f16 from vector<4xf16>
      %1345 = vector.extract %1341[2] : f16 from vector<4xf16>
      %1346 = vector.extract %1341[3] : f16 from vector<4xf16>
      %1347 = vector.from_elements %1343, %1344, %1345, %1346 : vector<4xf16>
      %1348 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1209, %1347, %1336 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1349 = vector.extract %1342[0] : f16 from vector<4xf16>
      %1350 = vector.extract %1342[1] : f16 from vector<4xf16>
      %1351 = vector.extract %1342[2] : f16 from vector<4xf16>
      %1352 = vector.extract %1342[3] : f16 from vector<4xf16>
      %1353 = vector.from_elements %1349, %1350, %1351, %1352 : vector<4xf16>
      %1354 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1353, %1222 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1355 = vector.extract %1348[0] : f32 from vector<4xf32>
      %1356 = vector.extract %1348[1] : f32 from vector<4xf32>
      %1357 = vector.extract %1348[2] : f32 from vector<4xf32>
      %1358 = vector.extract %1348[3] : f32 from vector<4xf32>
      %1359 = vector.load %view[%740] : memref<4096xf16, 3>, vector<4xf16>
      %1360 = vector.load %view[%588] : memref<4096xf16, 3>, vector<4xf16>
      %1361 = vector.extract %1359[0] : f16 from vector<4xf16>
      %1362 = vector.extract %1359[1] : f16 from vector<4xf16>
      %1363 = vector.extract %1359[2] : f16 from vector<4xf16>
      %1364 = vector.extract %1359[3] : f16 from vector<4xf16>
      %1365 = vector.from_elements %1361, %1362, %1363, %1364 : vector<4xf16>
      %1366 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1209, %1365, %1354 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1367 = vector.extract %1360[0] : f16 from vector<4xf16>
      %1368 = vector.extract %1360[1] : f16 from vector<4xf16>
      %1369 = vector.extract %1360[2] : f16 from vector<4xf16>
      %1370 = vector.extract %1360[3] : f16 from vector<4xf16>
      %1371 = vector.from_elements %1367, %1368, %1369, %1370 : vector<4xf16>
      %1372 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1371, %1224 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1373 = vector.extract %1366[0] : f32 from vector<4xf32>
      %1374 = vector.extract %1366[1] : f32 from vector<4xf32>
      %1375 = vector.extract %1366[2] : f32 from vector<4xf32>
      %1376 = vector.extract %1366[3] : f32 from vector<4xf32>
      %1377 = vector.load %view[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1378 = vector.load %view[%602] : memref<4096xf16, 3>, vector<4xf16>
      %1379 = vector.extract %1377[0] : f16 from vector<4xf16>
      %1380 = vector.extract %1377[1] : f16 from vector<4xf16>
      %1381 = vector.extract %1377[2] : f16 from vector<4xf16>
      %1382 = vector.extract %1377[3] : f16 from vector<4xf16>
      %1383 = vector.from_elements %1379, %1380, %1381, %1382 : vector<4xf16>
      %1384 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1225, %1383, %1372 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1385 = vector.extract %1378[0] : f16 from vector<4xf16>
      %1386 = vector.extract %1378[1] : f16 from vector<4xf16>
      %1387 = vector.extract %1378[2] : f16 from vector<4xf16>
      %1388 = vector.extract %1378[3] : f16 from vector<4xf16>
      %1389 = vector.from_elements %1385, %1386, %1387, %1388 : vector<4xf16>
      %1390 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1389, %1226 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1391 = vector.extract %1384[0] : f32 from vector<4xf32>
      %1392 = vector.extract %1384[1] : f32 from vector<4xf32>
      %1393 = vector.extract %1384[2] : f32 from vector<4xf32>
      %1394 = vector.extract %1384[3] : f32 from vector<4xf32>
      %1395 = vector.load %view[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1396 = vector.load %view[%622] : memref<4096xf16, 3>, vector<4xf16>
      %1397 = vector.extract %1395[0] : f16 from vector<4xf16>
      %1398 = vector.extract %1395[1] : f16 from vector<4xf16>
      %1399 = vector.extract %1395[2] : f16 from vector<4xf16>
      %1400 = vector.extract %1395[3] : f16 from vector<4xf16>
      %1401 = vector.from_elements %1397, %1398, %1399, %1400 : vector<4xf16>
      %1402 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1225, %1401, %1390 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1403 = vector.extract %1396[0] : f16 from vector<4xf16>
      %1404 = vector.extract %1396[1] : f16 from vector<4xf16>
      %1405 = vector.extract %1396[2] : f16 from vector<4xf16>
      %1406 = vector.extract %1396[3] : f16 from vector<4xf16>
      %1407 = vector.from_elements %1403, %1404, %1405, %1406 : vector<4xf16>
      %1408 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1407, %1227 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1409 = vector.extract %1402[0] : f32 from vector<4xf32>
      %1410 = vector.extract %1402[1] : f32 from vector<4xf32>
      %1411 = vector.extract %1402[2] : f32 from vector<4xf32>
      %1412 = vector.extract %1402[3] : f32 from vector<4xf32>
      %1413 = vector.load %view[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1414 = vector.load %view[%642] : memref<4096xf16, 3>, vector<4xf16>
      %1415 = vector.extract %1413[0] : f16 from vector<4xf16>
      %1416 = vector.extract %1413[1] : f16 from vector<4xf16>
      %1417 = vector.extract %1413[2] : f16 from vector<4xf16>
      %1418 = vector.extract %1413[3] : f16 from vector<4xf16>
      %1419 = vector.from_elements %1415, %1416, %1417, %1418 : vector<4xf16>
      %1420 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1225, %1419, %1408 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1421 = vector.extract %1414[0] : f16 from vector<4xf16>
      %1422 = vector.extract %1414[1] : f16 from vector<4xf16>
      %1423 = vector.extract %1414[2] : f16 from vector<4xf16>
      %1424 = vector.extract %1414[3] : f16 from vector<4xf16>
      %1425 = vector.from_elements %1421, %1422, %1423, %1424 : vector<4xf16>
      %1426 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1425, %1228 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1427 = vector.extract %1420[0] : f32 from vector<4xf32>
      %1428 = vector.extract %1420[1] : f32 from vector<4xf32>
      %1429 = vector.extract %1420[2] : f32 from vector<4xf32>
      %1430 = vector.extract %1420[3] : f32 from vector<4xf32>
      %1431 = vector.load %view[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1432 = vector.load %view[%662] : memref<4096xf16, 3>, vector<4xf16>
      %1433 = vector.extract %1431[0] : f16 from vector<4xf16>
      %1434 = vector.extract %1431[1] : f16 from vector<4xf16>
      %1435 = vector.extract %1431[2] : f16 from vector<4xf16>
      %1436 = vector.extract %1431[3] : f16 from vector<4xf16>
      %1437 = vector.from_elements %1433, %1434, %1435, %1436 : vector<4xf16>
      %1438 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1225, %1437, %1426 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1439 = vector.extract %1432[0] : f16 from vector<4xf16>
      %1440 = vector.extract %1432[1] : f16 from vector<4xf16>
      %1441 = vector.extract %1432[2] : f16 from vector<4xf16>
      %1442 = vector.extract %1432[3] : f16 from vector<4xf16>
      %1443 = vector.from_elements %1439, %1440, %1441, %1442 : vector<4xf16>
      %1444 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1443, %1229 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1445 = vector.extract %1438[0] : f32 from vector<4xf32>
      %1446 = vector.extract %1438[1] : f32 from vector<4xf32>
      %1447 = vector.extract %1438[2] : f32 from vector<4xf32>
      %1448 = vector.extract %1438[3] : f32 from vector<4xf32>
      %1449 = vector.load %view[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1450 = vector.load %view[%682] : memref<4096xf16, 3>, vector<4xf16>
      %1451 = vector.extract %1449[0] : f16 from vector<4xf16>
      %1452 = vector.extract %1449[1] : f16 from vector<4xf16>
      %1453 = vector.extract %1449[2] : f16 from vector<4xf16>
      %1454 = vector.extract %1449[3] : f16 from vector<4xf16>
      %1455 = vector.from_elements %1451, %1452, %1453, %1454 : vector<4xf16>
      %1456 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1225, %1455, %1444 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1457 = vector.extract %1450[0] : f16 from vector<4xf16>
      %1458 = vector.extract %1450[1] : f16 from vector<4xf16>
      %1459 = vector.extract %1450[2] : f16 from vector<4xf16>
      %1460 = vector.extract %1450[3] : f16 from vector<4xf16>
      %1461 = vector.from_elements %1457, %1458, %1459, %1460 : vector<4xf16>
      %1462 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1461, %1230 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1463 = vector.extract %1456[0] : f32 from vector<4xf32>
      %1464 = vector.extract %1456[1] : f32 from vector<4xf32>
      %1465 = vector.extract %1456[2] : f32 from vector<4xf32>
      %1466 = vector.extract %1456[3] : f32 from vector<4xf32>
      %1467 = vector.load %view[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1468 = vector.load %view[%702] : memref<4096xf16, 3>, vector<4xf16>
      %1469 = vector.extract %1467[0] : f16 from vector<4xf16>
      %1470 = vector.extract %1467[1] : f16 from vector<4xf16>
      %1471 = vector.extract %1467[2] : f16 from vector<4xf16>
      %1472 = vector.extract %1467[3] : f16 from vector<4xf16>
      %1473 = vector.from_elements %1469, %1470, %1471, %1472 : vector<4xf16>
      %1474 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1225, %1473, %1462 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1475 = vector.extract %1468[0] : f16 from vector<4xf16>
      %1476 = vector.extract %1468[1] : f16 from vector<4xf16>
      %1477 = vector.extract %1468[2] : f16 from vector<4xf16>
      %1478 = vector.extract %1468[3] : f16 from vector<4xf16>
      %1479 = vector.from_elements %1475, %1476, %1477, %1478 : vector<4xf16>
      %1480 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1479, %1231 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1481 = vector.extract %1474[0] : f32 from vector<4xf32>
      %1482 = vector.extract %1474[1] : f32 from vector<4xf32>
      %1483 = vector.extract %1474[2] : f32 from vector<4xf32>
      %1484 = vector.extract %1474[3] : f32 from vector<4xf32>
      %1485 = vector.load %view[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1486 = vector.load %view[%722] : memref<4096xf16, 3>, vector<4xf16>
      %1487 = vector.extract %1485[0] : f16 from vector<4xf16>
      %1488 = vector.extract %1485[1] : f16 from vector<4xf16>
      %1489 = vector.extract %1485[2] : f16 from vector<4xf16>
      %1490 = vector.extract %1485[3] : f16 from vector<4xf16>
      %1491 = vector.from_elements %1487, %1488, %1489, %1490 : vector<4xf16>
      %1492 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1225, %1491, %1480 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1493 = vector.extract %1486[0] : f16 from vector<4xf16>
      %1494 = vector.extract %1486[1] : f16 from vector<4xf16>
      %1495 = vector.extract %1486[2] : f16 from vector<4xf16>
      %1496 = vector.extract %1486[3] : f16 from vector<4xf16>
      %1497 = vector.from_elements %1493, %1494, %1495, %1496 : vector<4xf16>
      %1498 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1497, %1232 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1499 = vector.extract %1492[0] : f32 from vector<4xf32>
      %1500 = vector.extract %1492[1] : f32 from vector<4xf32>
      %1501 = vector.extract %1492[2] : f32 from vector<4xf32>
      %1502 = vector.extract %1492[3] : f32 from vector<4xf32>
      %1503 = vector.load %view[%740] : memref<4096xf16, 3>, vector<4xf16>
      %1504 = vector.extract %1503[0] : f16 from vector<4xf16>
      %1505 = vector.extract %1503[1] : f16 from vector<4xf16>
      %1506 = vector.extract %1503[2] : f16 from vector<4xf16>
      %1507 = vector.extract %1503[3] : f16 from vector<4xf16>
      %1508 = vector.from_elements %1504, %1505, %1506, %1507 : vector<4xf16>
      %1509 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1225, %1508, %1498 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1510 = vector.extract %1509[0] : f32 from vector<4xf32>
      %1511 = vector.extract %1509[1] : f32 from vector<4xf32>
      %1512 = vector.extract %1509[2] : f32 from vector<4xf32>
      %1513 = vector.extract %1509[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %1205 : !frisk.async_task
      %subview_141 = memref.subview %arg2[0, %198, %585, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %1514 = frisk.async_copy_gm_to_shm %subview_141 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %1515 = vector.load %view_7[%588] : memref<4096xf16, 3>, vector<4xf16>
      %1516 = vector.from_elements %174, %175, %176, %177 : vector<4xf16>
      %1517 = vector.from_elements %1247, %1248, %1249, %1250 : vector<4xf32>
      %1518 = vector.from_elements %178, %179, %180, %181 : vector<4xf16>
      %1519 = vector.from_elements %1265, %1266, %1267, %1268 : vector<4xf32>
      %1520 = vector.extract %1515[0] : f16 from vector<4xf16>
      %1521 = vector.extract %1515[1] : f16 from vector<4xf16>
      %1522 = vector.extract %1515[2] : f16 from vector<4xf16>
      %1523 = vector.extract %1515[3] : f16 from vector<4xf16>
      %1524 = vector.from_elements %1520, %1521, %1522, %1523 : vector<4xf16>
      %1525 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1524, %1517 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1526 = vector.from_elements %1283, %1284, %1285, %1286 : vector<4xf32>
      %1527 = vector.from_elements %1301, %1302, %1303, %1304 : vector<4xf32>
      %1528 = vector.from_elements %1319, %1320, %1321, %1322 : vector<4xf32>
      %1529 = vector.from_elements %1337, %1338, %1339, %1340 : vector<4xf32>
      %1530 = vector.from_elements %1355, %1356, %1357, %1358 : vector<4xf32>
      %1531 = vector.from_elements %1373, %1374, %1375, %1376 : vector<4xf32>
      %1532 = vector.from_elements %190, %191, %192, %193 : vector<4xf16>
      %1533 = vector.from_elements %1391, %1392, %1393, %1394 : vector<4xf32>
      %1534 = vector.from_elements %194, %195, %196, %197 : vector<4xf16>
      %1535 = vector.from_elements %1409, %1410, %1411, %1412 : vector<4xf32>
      %1536 = vector.from_elements %1427, %1428, %1429, %1430 : vector<4xf32>
      %1537 = vector.from_elements %1445, %1446, %1447, %1448 : vector<4xf32>
      %1538 = vector.from_elements %1463, %1464, %1465, %1466 : vector<4xf32>
      %1539 = vector.from_elements %1481, %1482, %1483, %1484 : vector<4xf32>
      %1540 = vector.from_elements %1499, %1500, %1501, %1502 : vector<4xf32>
      %1541 = vector.from_elements %1510, %1511, %1512, %1513 : vector<4xf32>
      %1542 = vector.load %view_7[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1543 = vector.load %view_7[%602] : memref<4096xf16, 3>, vector<4xf16>
      %1544 = vector.extract %1542[0] : f16 from vector<4xf16>
      %1545 = vector.extract %1542[1] : f16 from vector<4xf16>
      %1546 = vector.extract %1542[2] : f16 from vector<4xf16>
      %1547 = vector.extract %1542[3] : f16 from vector<4xf16>
      %1548 = vector.from_elements %1544, %1545, %1546, %1547 : vector<4xf16>
      %1549 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1518, %1548, %1525 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1550 = vector.extract %1543[0] : f16 from vector<4xf16>
      %1551 = vector.extract %1543[1] : f16 from vector<4xf16>
      %1552 = vector.extract %1543[2] : f16 from vector<4xf16>
      %1553 = vector.extract %1543[3] : f16 from vector<4xf16>
      %1554 = vector.from_elements %1550, %1551, %1552, %1553 : vector<4xf16>
      %1555 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1554, %1519 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1556 = vector.extract %1549[0] : f32 from vector<4xf32>
      %1557 = arith.maxnumf %1556, %cst_3 : f32
      %1558 = vector.extract %1549[1] : f32 from vector<4xf32>
      %1559 = arith.maxnumf %1557, %1558 : f32
      %1560 = vector.extract %1549[2] : f32 from vector<4xf32>
      %1561 = arith.maxnumf %1559, %1560 : f32
      %1562 = vector.extract %1549[3] : f32 from vector<4xf32>
      %1563 = arith.maxnumf %1561, %1562 : f32
      %1564 = vector.load %view_7[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1565 = vector.load %view_7[%622] : memref<4096xf16, 3>, vector<4xf16>
      %1566 = vector.extract %1564[0] : f16 from vector<4xf16>
      %1567 = vector.extract %1564[1] : f16 from vector<4xf16>
      %1568 = vector.extract %1564[2] : f16 from vector<4xf16>
      %1569 = vector.extract %1564[3] : f16 from vector<4xf16>
      %1570 = vector.from_elements %1566, %1567, %1568, %1569 : vector<4xf16>
      %1571 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1518, %1570, %1555 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1572 = vector.extract %1565[0] : f16 from vector<4xf16>
      %1573 = vector.extract %1565[1] : f16 from vector<4xf16>
      %1574 = vector.extract %1565[2] : f16 from vector<4xf16>
      %1575 = vector.extract %1565[3] : f16 from vector<4xf16>
      %1576 = vector.from_elements %1572, %1573, %1574, %1575 : vector<4xf16>
      %1577 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1576, %1526 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1578 = vector.extract %1571[0] : f32 from vector<4xf32>
      %1579 = arith.maxnumf %1563, %1578 : f32
      %1580 = vector.extract %1571[1] : f32 from vector<4xf32>
      %1581 = arith.maxnumf %1579, %1580 : f32
      %1582 = vector.extract %1571[2] : f32 from vector<4xf32>
      %1583 = arith.maxnumf %1581, %1582 : f32
      %1584 = vector.extract %1571[3] : f32 from vector<4xf32>
      %1585 = arith.maxnumf %1583, %1584 : f32
      %1586 = vector.load %view_7[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1587 = vector.load %view_7[%642] : memref<4096xf16, 3>, vector<4xf16>
      %1588 = vector.extract %1586[0] : f16 from vector<4xf16>
      %1589 = vector.extract %1586[1] : f16 from vector<4xf16>
      %1590 = vector.extract %1586[2] : f16 from vector<4xf16>
      %1591 = vector.extract %1586[3] : f16 from vector<4xf16>
      %1592 = vector.from_elements %1588, %1589, %1590, %1591 : vector<4xf16>
      %1593 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1518, %1592, %1577 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1594 = vector.extract %1587[0] : f16 from vector<4xf16>
      %1595 = vector.extract %1587[1] : f16 from vector<4xf16>
      %1596 = vector.extract %1587[2] : f16 from vector<4xf16>
      %1597 = vector.extract %1587[3] : f16 from vector<4xf16>
      %1598 = vector.from_elements %1594, %1595, %1596, %1597 : vector<4xf16>
      %1599 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1598, %1527 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1600 = vector.extract %1593[0] : f32 from vector<4xf32>
      %1601 = arith.maxnumf %1585, %1600 : f32
      %1602 = vector.extract %1593[1] : f32 from vector<4xf32>
      %1603 = arith.maxnumf %1601, %1602 : f32
      %1604 = vector.extract %1593[2] : f32 from vector<4xf32>
      %1605 = arith.maxnumf %1603, %1604 : f32
      %1606 = vector.extract %1593[3] : f32 from vector<4xf32>
      %1607 = arith.maxnumf %1605, %1606 : f32
      %1608 = vector.load %view_7[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1609 = vector.load %view_7[%662] : memref<4096xf16, 3>, vector<4xf16>
      %1610 = vector.extract %1608[0] : f16 from vector<4xf16>
      %1611 = vector.extract %1608[1] : f16 from vector<4xf16>
      %1612 = vector.extract %1608[2] : f16 from vector<4xf16>
      %1613 = vector.extract %1608[3] : f16 from vector<4xf16>
      %1614 = vector.from_elements %1610, %1611, %1612, %1613 : vector<4xf16>
      %1615 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1518, %1614, %1599 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1616 = vector.extract %1609[0] : f16 from vector<4xf16>
      %1617 = vector.extract %1609[1] : f16 from vector<4xf16>
      %1618 = vector.extract %1609[2] : f16 from vector<4xf16>
      %1619 = vector.extract %1609[3] : f16 from vector<4xf16>
      %1620 = vector.from_elements %1616, %1617, %1618, %1619 : vector<4xf16>
      %1621 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1620, %1528 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1622 = vector.extract %1615[0] : f32 from vector<4xf32>
      %1623 = arith.maxnumf %1607, %1622 : f32
      %1624 = vector.extract %1615[1] : f32 from vector<4xf32>
      %1625 = arith.maxnumf %1623, %1624 : f32
      %1626 = vector.extract %1615[2] : f32 from vector<4xf32>
      %1627 = arith.maxnumf %1625, %1626 : f32
      %1628 = vector.extract %1615[3] : f32 from vector<4xf32>
      %1629 = arith.maxnumf %1627, %1628 : f32
      %1630 = vector.load %view_7[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1631 = vector.load %view_7[%682] : memref<4096xf16, 3>, vector<4xf16>
      %1632 = vector.extract %1630[0] : f16 from vector<4xf16>
      %1633 = vector.extract %1630[1] : f16 from vector<4xf16>
      %1634 = vector.extract %1630[2] : f16 from vector<4xf16>
      %1635 = vector.extract %1630[3] : f16 from vector<4xf16>
      %1636 = vector.from_elements %1632, %1633, %1634, %1635 : vector<4xf16>
      %1637 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1518, %1636, %1621 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1638 = vector.extract %1631[0] : f16 from vector<4xf16>
      %1639 = vector.extract %1631[1] : f16 from vector<4xf16>
      %1640 = vector.extract %1631[2] : f16 from vector<4xf16>
      %1641 = vector.extract %1631[3] : f16 from vector<4xf16>
      %1642 = vector.from_elements %1638, %1639, %1640, %1641 : vector<4xf16>
      %1643 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1642, %1529 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1644 = vector.extract %1637[0] : f32 from vector<4xf32>
      %1645 = arith.maxnumf %1629, %1644 : f32
      %1646 = vector.extract %1637[1] : f32 from vector<4xf32>
      %1647 = arith.maxnumf %1645, %1646 : f32
      %1648 = vector.extract %1637[2] : f32 from vector<4xf32>
      %1649 = arith.maxnumf %1647, %1648 : f32
      %1650 = vector.extract %1637[3] : f32 from vector<4xf32>
      %1651 = arith.maxnumf %1649, %1650 : f32
      %1652 = vector.load %view_7[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1653 = vector.load %view_7[%702] : memref<4096xf16, 3>, vector<4xf16>
      %1654 = vector.extract %1652[0] : f16 from vector<4xf16>
      %1655 = vector.extract %1652[1] : f16 from vector<4xf16>
      %1656 = vector.extract %1652[2] : f16 from vector<4xf16>
      %1657 = vector.extract %1652[3] : f16 from vector<4xf16>
      %1658 = vector.from_elements %1654, %1655, %1656, %1657 : vector<4xf16>
      %1659 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1518, %1658, %1643 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1660 = vector.extract %1653[0] : f16 from vector<4xf16>
      %1661 = vector.extract %1653[1] : f16 from vector<4xf16>
      %1662 = vector.extract %1653[2] : f16 from vector<4xf16>
      %1663 = vector.extract %1653[3] : f16 from vector<4xf16>
      %1664 = vector.from_elements %1660, %1661, %1662, %1663 : vector<4xf16>
      %1665 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1664, %1530 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1666 = vector.extract %1659[0] : f32 from vector<4xf32>
      %1667 = arith.maxnumf %1651, %1666 : f32
      %1668 = vector.extract %1659[1] : f32 from vector<4xf32>
      %1669 = arith.maxnumf %1667, %1668 : f32
      %1670 = vector.extract %1659[2] : f32 from vector<4xf32>
      %1671 = arith.maxnumf %1669, %1670 : f32
      %1672 = vector.extract %1659[3] : f32 from vector<4xf32>
      %1673 = arith.maxnumf %1671, %1672 : f32
      %1674 = vector.load %view_7[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1675 = vector.load %view_7[%722] : memref<4096xf16, 3>, vector<4xf16>
      %1676 = vector.extract %1674[0] : f16 from vector<4xf16>
      %1677 = vector.extract %1674[1] : f16 from vector<4xf16>
      %1678 = vector.extract %1674[2] : f16 from vector<4xf16>
      %1679 = vector.extract %1674[3] : f16 from vector<4xf16>
      %1680 = vector.from_elements %1676, %1677, %1678, %1679 : vector<4xf16>
      %1681 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1518, %1680, %1665 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1682 = vector.extract %1675[0] : f16 from vector<4xf16>
      %1683 = vector.extract %1675[1] : f16 from vector<4xf16>
      %1684 = vector.extract %1675[2] : f16 from vector<4xf16>
      %1685 = vector.extract %1675[3] : f16 from vector<4xf16>
      %1686 = vector.from_elements %1682, %1683, %1684, %1685 : vector<4xf16>
      %1687 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1686, %1531 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1688 = vector.extract %1681[0] : f32 from vector<4xf32>
      %1689 = arith.maxnumf %1673, %1688 : f32
      %1690 = vector.extract %1681[1] : f32 from vector<4xf32>
      %1691 = arith.maxnumf %1689, %1690 : f32
      %1692 = vector.extract %1681[2] : f32 from vector<4xf32>
      %1693 = arith.maxnumf %1691, %1692 : f32
      %1694 = vector.extract %1681[3] : f32 from vector<4xf32>
      %1695 = arith.maxnumf %1693, %1694 : f32
      %1696 = vector.load %view_7[%740] : memref<4096xf16, 3>, vector<4xf16>
      %1697 = vector.load %view_7[%588] : memref<4096xf16, 3>, vector<4xf16>
      %1698 = vector.extract %1696[0] : f16 from vector<4xf16>
      %1699 = vector.extract %1696[1] : f16 from vector<4xf16>
      %1700 = vector.extract %1696[2] : f16 from vector<4xf16>
      %1701 = vector.extract %1696[3] : f16 from vector<4xf16>
      %1702 = vector.from_elements %1698, %1699, %1700, %1701 : vector<4xf16>
      %1703 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1518, %1702, %1687 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1704 = vector.extract %1697[0] : f16 from vector<4xf16>
      %1705 = vector.extract %1697[1] : f16 from vector<4xf16>
      %1706 = vector.extract %1697[2] : f16 from vector<4xf16>
      %1707 = vector.extract %1697[3] : f16 from vector<4xf16>
      %1708 = vector.from_elements %1704, %1705, %1706, %1707 : vector<4xf16>
      %1709 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1708, %1533 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1710 = vector.extract %1703[0] : f32 from vector<4xf32>
      %1711 = arith.maxnumf %1695, %1710 : f32
      %1712 = vector.extract %1703[1] : f32 from vector<4xf32>
      %1713 = arith.maxnumf %1711, %1712 : f32
      %1714 = vector.extract %1703[2] : f32 from vector<4xf32>
      %1715 = arith.maxnumf %1713, %1714 : f32
      %1716 = vector.extract %1703[3] : f32 from vector<4xf32>
      %1717 = arith.maxnumf %1715, %1716 : f32
      %1718 = vector.load %view_7[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1719 = vector.load %view_7[%602] : memref<4096xf16, 3>, vector<4xf16>
      %1720 = vector.extract %1718[0] : f16 from vector<4xf16>
      %1721 = vector.extract %1718[1] : f16 from vector<4xf16>
      %1722 = vector.extract %1718[2] : f16 from vector<4xf16>
      %1723 = vector.extract %1718[3] : f16 from vector<4xf16>
      %1724 = vector.from_elements %1720, %1721, %1722, %1723 : vector<4xf16>
      %1725 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1534, %1724, %1709 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1726 = vector.extract %1719[0] : f16 from vector<4xf16>
      %1727 = vector.extract %1719[1] : f16 from vector<4xf16>
      %1728 = vector.extract %1719[2] : f16 from vector<4xf16>
      %1729 = vector.extract %1719[3] : f16 from vector<4xf16>
      %1730 = vector.from_elements %1726, %1727, %1728, %1729 : vector<4xf16>
      %1731 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1730, %1535 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1732 = vector.extract %1725[0] : f32 from vector<4xf32>
      %1733 = vector.extract %1725[1] : f32 from vector<4xf32>
      %1734 = vector.extract %1725[2] : f32 from vector<4xf32>
      %1735 = vector.extract %1725[3] : f32 from vector<4xf32>
      %1736 = vector.load %view_7[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1737 = vector.load %view_7[%622] : memref<4096xf16, 3>, vector<4xf16>
      %1738 = vector.extract %1736[0] : f16 from vector<4xf16>
      %1739 = vector.extract %1736[1] : f16 from vector<4xf16>
      %1740 = vector.extract %1736[2] : f16 from vector<4xf16>
      %1741 = vector.extract %1736[3] : f16 from vector<4xf16>
      %1742 = vector.from_elements %1738, %1739, %1740, %1741 : vector<4xf16>
      %1743 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1534, %1742, %1731 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1744 = vector.extract %1737[0] : f16 from vector<4xf16>
      %1745 = vector.extract %1737[1] : f16 from vector<4xf16>
      %1746 = vector.extract %1737[2] : f16 from vector<4xf16>
      %1747 = vector.extract %1737[3] : f16 from vector<4xf16>
      %1748 = vector.from_elements %1744, %1745, %1746, %1747 : vector<4xf16>
      %1749 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1748, %1536 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1750 = vector.extract %1743[0] : f32 from vector<4xf32>
      %1751 = vector.extract %1743[1] : f32 from vector<4xf32>
      %1752 = vector.extract %1743[2] : f32 from vector<4xf32>
      %1753 = vector.extract %1743[3] : f32 from vector<4xf32>
      %1754 = vector.load %view_7[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1755 = vector.load %view_7[%642] : memref<4096xf16, 3>, vector<4xf16>
      %1756 = vector.extract %1754[0] : f16 from vector<4xf16>
      %1757 = vector.extract %1754[1] : f16 from vector<4xf16>
      %1758 = vector.extract %1754[2] : f16 from vector<4xf16>
      %1759 = vector.extract %1754[3] : f16 from vector<4xf16>
      %1760 = vector.from_elements %1756, %1757, %1758, %1759 : vector<4xf16>
      %1761 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1534, %1760, %1749 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1762 = vector.extract %1755[0] : f16 from vector<4xf16>
      %1763 = vector.extract %1755[1] : f16 from vector<4xf16>
      %1764 = vector.extract %1755[2] : f16 from vector<4xf16>
      %1765 = vector.extract %1755[3] : f16 from vector<4xf16>
      %1766 = vector.from_elements %1762, %1763, %1764, %1765 : vector<4xf16>
      %1767 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1766, %1537 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1768 = vector.extract %1761[0] : f32 from vector<4xf32>
      %1769 = vector.extract %1761[1] : f32 from vector<4xf32>
      %1770 = vector.extract %1761[2] : f32 from vector<4xf32>
      %1771 = vector.extract %1761[3] : f32 from vector<4xf32>
      %1772 = vector.load %view_7[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1773 = vector.load %view_7[%662] : memref<4096xf16, 3>, vector<4xf16>
      %1774 = vector.extract %1772[0] : f16 from vector<4xf16>
      %1775 = vector.extract %1772[1] : f16 from vector<4xf16>
      %1776 = vector.extract %1772[2] : f16 from vector<4xf16>
      %1777 = vector.extract %1772[3] : f16 from vector<4xf16>
      %1778 = vector.from_elements %1774, %1775, %1776, %1777 : vector<4xf16>
      %1779 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1534, %1778, %1767 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1780 = vector.extract %1773[0] : f16 from vector<4xf16>
      %1781 = vector.extract %1773[1] : f16 from vector<4xf16>
      %1782 = vector.extract %1773[2] : f16 from vector<4xf16>
      %1783 = vector.extract %1773[3] : f16 from vector<4xf16>
      %1784 = vector.from_elements %1780, %1781, %1782, %1783 : vector<4xf16>
      %1785 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1784, %1538 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1786 = vector.extract %1779[0] : f32 from vector<4xf32>
      %1787 = vector.extract %1779[1] : f32 from vector<4xf32>
      %1788 = vector.extract %1779[2] : f32 from vector<4xf32>
      %1789 = vector.extract %1779[3] : f32 from vector<4xf32>
      %1790 = vector.load %view_7[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1791 = vector.load %view_7[%682] : memref<4096xf16, 3>, vector<4xf16>
      %1792 = vector.extract %1790[0] : f16 from vector<4xf16>
      %1793 = vector.extract %1790[1] : f16 from vector<4xf16>
      %1794 = vector.extract %1790[2] : f16 from vector<4xf16>
      %1795 = vector.extract %1790[3] : f16 from vector<4xf16>
      %1796 = vector.from_elements %1792, %1793, %1794, %1795 : vector<4xf16>
      %1797 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1534, %1796, %1785 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1798 = vector.extract %1791[0] : f16 from vector<4xf16>
      %1799 = vector.extract %1791[1] : f16 from vector<4xf16>
      %1800 = vector.extract %1791[2] : f16 from vector<4xf16>
      %1801 = vector.extract %1791[3] : f16 from vector<4xf16>
      %1802 = vector.from_elements %1798, %1799, %1800, %1801 : vector<4xf16>
      %1803 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1802, %1539 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1804 = vector.extract %1797[0] : f32 from vector<4xf32>
      %1805 = vector.extract %1797[1] : f32 from vector<4xf32>
      %1806 = vector.extract %1797[2] : f32 from vector<4xf32>
      %1807 = vector.extract %1797[3] : f32 from vector<4xf32>
      %1808 = vector.load %view_7[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1809 = vector.load %view_7[%702] : memref<4096xf16, 3>, vector<4xf16>
      %1810 = vector.extract %1808[0] : f16 from vector<4xf16>
      %1811 = vector.extract %1808[1] : f16 from vector<4xf16>
      %1812 = vector.extract %1808[2] : f16 from vector<4xf16>
      %1813 = vector.extract %1808[3] : f16 from vector<4xf16>
      %1814 = vector.from_elements %1810, %1811, %1812, %1813 : vector<4xf16>
      %1815 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1534, %1814, %1803 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1816 = vector.extract %1809[0] : f16 from vector<4xf16>
      %1817 = vector.extract %1809[1] : f16 from vector<4xf16>
      %1818 = vector.extract %1809[2] : f16 from vector<4xf16>
      %1819 = vector.extract %1809[3] : f16 from vector<4xf16>
      %1820 = vector.from_elements %1816, %1817, %1818, %1819 : vector<4xf16>
      %1821 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1820, %1540 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1822 = vector.extract %1815[0] : f32 from vector<4xf32>
      %1823 = vector.extract %1815[1] : f32 from vector<4xf32>
      %1824 = vector.extract %1815[2] : f32 from vector<4xf32>
      %1825 = vector.extract %1815[3] : f32 from vector<4xf32>
      %1826 = vector.load %view_7[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1827 = vector.load %view_7[%722] : memref<4096xf16, 3>, vector<4xf16>
      %1828 = vector.extract %1826[0] : f16 from vector<4xf16>
      %1829 = vector.extract %1826[1] : f16 from vector<4xf16>
      %1830 = vector.extract %1826[2] : f16 from vector<4xf16>
      %1831 = vector.extract %1826[3] : f16 from vector<4xf16>
      %1832 = vector.from_elements %1828, %1829, %1830, %1831 : vector<4xf16>
      %1833 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1534, %1832, %1821 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1834 = vector.extract %1827[0] : f16 from vector<4xf16>
      %1835 = vector.extract %1827[1] : f16 from vector<4xf16>
      %1836 = vector.extract %1827[2] : f16 from vector<4xf16>
      %1837 = vector.extract %1827[3] : f16 from vector<4xf16>
      %1838 = vector.from_elements %1834, %1835, %1836, %1837 : vector<4xf16>
      %1839 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1838, %1541 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1840 = vector.extract %1833[0] : f32 from vector<4xf32>
      %1841 = vector.extract %1833[1] : f32 from vector<4xf32>
      %1842 = vector.extract %1833[2] : f32 from vector<4xf32>
      %1843 = vector.extract %1833[3] : f32 from vector<4xf32>
      %1844 = vector.load %view_7[%740] : memref<4096xf16, 3>, vector<4xf16>
      %1845 = vector.extract %1844[0] : f16 from vector<4xf16>
      %1846 = vector.extract %1844[1] : f16 from vector<4xf16>
      %1847 = vector.extract %1844[2] : f16 from vector<4xf16>
      %1848 = vector.extract %1844[3] : f16 from vector<4xf16>
      %1849 = vector.from_elements %1845, %1846, %1847, %1848 : vector<4xf16>
      %1850 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1534, %1849, %1839 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1851 = vector.extract %1850[0] : f32 from vector<4xf32>
      %1852 = vector.extract %1850[1] : f32 from vector<4xf32>
      %1853 = vector.extract %1850[2] : f32 from vector<4xf32>
      %1854 = vector.extract %1850[3] : f32 from vector<4xf32>
      %shuffleResult_142, %valid_143 = gpu.shuffle xor %1717, %c16_i32, %c64_i32 : f32
      %1855 = arith.maxnumf %1717, %shuffleResult_142 : f32
      %shuffleResult_144, %valid_145 = gpu.shuffle xor %1855, %c32_i32, %c64_i32 : f32
      %1856 = arith.maxnumf %1855, %shuffleResult_144 : f32
      %1857 = arith.maxnumf %1732, %cst_3 : f32
      %1858 = arith.maxnumf %1857, %1733 : f32
      %1859 = arith.maxnumf %1858, %1734 : f32
      %1860 = arith.maxnumf %1859, %1735 : f32
      %1861 = arith.maxnumf %1860, %1750 : f32
      %1862 = arith.maxnumf %1861, %1751 : f32
      %1863 = arith.maxnumf %1862, %1752 : f32
      %1864 = arith.maxnumf %1863, %1753 : f32
      %1865 = arith.maxnumf %1864, %1768 : f32
      %1866 = arith.maxnumf %1865, %1769 : f32
      %1867 = arith.maxnumf %1866, %1770 : f32
      %1868 = arith.maxnumf %1867, %1771 : f32
      %1869 = arith.maxnumf %1868, %1786 : f32
      %1870 = arith.maxnumf %1869, %1787 : f32
      %1871 = arith.maxnumf %1870, %1788 : f32
      %1872 = arith.maxnumf %1871, %1789 : f32
      %1873 = arith.maxnumf %1872, %1804 : f32
      %1874 = arith.maxnumf %1873, %1805 : f32
      %1875 = arith.maxnumf %1874, %1806 : f32
      %1876 = arith.maxnumf %1875, %1807 : f32
      %1877 = arith.maxnumf %1876, %1822 : f32
      %1878 = arith.maxnumf %1877, %1823 : f32
      %1879 = arith.maxnumf %1878, %1824 : f32
      %1880 = arith.maxnumf %1879, %1825 : f32
      %1881 = arith.maxnumf %1880, %1840 : f32
      %1882 = arith.maxnumf %1881, %1841 : f32
      %1883 = arith.maxnumf %1882, %1842 : f32
      %1884 = arith.maxnumf %1883, %1843 : f32
      %1885 = arith.maxnumf %1884, %1851 : f32
      %1886 = arith.maxnumf %1885, %1852 : f32
      %1887 = arith.maxnumf %1886, %1853 : f32
      %1888 = arith.maxnumf %1887, %1854 : f32
      %shuffleResult_146, %valid_147 = gpu.shuffle xor %1888, %c16_i32, %c64_i32 : f32
      %1889 = arith.maxnumf %1888, %shuffleResult_146 : f32
      %shuffleResult_148, %valid_149 = gpu.shuffle xor %1889, %c32_i32, %c64_i32 : f32
      %1890 = arith.maxnumf %1889, %shuffleResult_148 : f32
      %1891 = vector.extract %arg7[0, 0] : f32 from vector<2x1xf32>
      %1892 = vector.broadcast %1856 : f32 to vector<1xf32>
      %1893 = vector.broadcast %1891 : f32 to vector<1xf32>
      %1894 = arith.maxnumf %1892, %1893 {origin = "frisk.max", tiled = true} : vector<1xf32>
      %1895 = vector.extract %1894[0] : f32 from vector<1xf32>
      %1896 = vector.extract %arg7[1, 0] : f32 from vector<2x1xf32>
      %1897 = vector.broadcast %1890 : f32 to vector<1xf32>
      %1898 = vector.broadcast %1896 : f32 to vector<1xf32>
      %1899 = arith.maxnumf %1897, %1898 {origin = "frisk.max", tiled = true} : vector<1xf32>
      %1900 = vector.extract %1899[0] : f32 from vector<1xf32>
      %1901 = vector.from_elements %1556, %1558, %1560, %1562 : vector<4xf32>
      %1902 = vector.broadcast %1895 : f32 to vector<4xf32>
      %1903 = arith.subf %1901, %1902 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1904 = vector.extract %1903[0] : f32 from vector<4xf32>
      %1905 = vector.extract %1903[1] : f32 from vector<4xf32>
      %1906 = vector.extract %1903[2] : f32 from vector<4xf32>
      %1907 = vector.extract %1903[3] : f32 from vector<4xf32>
      %1908 = vector.from_elements %1578, %1580, %1582, %1584 : vector<4xf32>
      %1909 = arith.subf %1908, %1902 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1910 = vector.extract %1909[0] : f32 from vector<4xf32>
      %1911 = vector.extract %1909[1] : f32 from vector<4xf32>
      %1912 = vector.extract %1909[2] : f32 from vector<4xf32>
      %1913 = vector.extract %1909[3] : f32 from vector<4xf32>
      %1914 = vector.from_elements %1600, %1602, %1604, %1606 : vector<4xf32>
      %1915 = arith.subf %1914, %1902 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1916 = vector.extract %1915[0] : f32 from vector<4xf32>
      %1917 = vector.extract %1915[1] : f32 from vector<4xf32>
      %1918 = vector.extract %1915[2] : f32 from vector<4xf32>
      %1919 = vector.extract %1915[3] : f32 from vector<4xf32>
      %1920 = vector.from_elements %1622, %1624, %1626, %1628 : vector<4xf32>
      %1921 = arith.subf %1920, %1902 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1922 = vector.extract %1921[0] : f32 from vector<4xf32>
      %1923 = vector.extract %1921[1] : f32 from vector<4xf32>
      %1924 = vector.extract %1921[2] : f32 from vector<4xf32>
      %1925 = vector.extract %1921[3] : f32 from vector<4xf32>
      %1926 = vector.from_elements %1644, %1646, %1648, %1650 : vector<4xf32>
      %1927 = arith.subf %1926, %1902 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1928 = vector.extract %1927[0] : f32 from vector<4xf32>
      %1929 = vector.extract %1927[1] : f32 from vector<4xf32>
      %1930 = vector.extract %1927[2] : f32 from vector<4xf32>
      %1931 = vector.extract %1927[3] : f32 from vector<4xf32>
      %1932 = vector.from_elements %1666, %1668, %1670, %1672 : vector<4xf32>
      %1933 = arith.subf %1932, %1902 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1934 = vector.extract %1933[0] : f32 from vector<4xf32>
      %1935 = vector.extract %1933[1] : f32 from vector<4xf32>
      %1936 = vector.extract %1933[2] : f32 from vector<4xf32>
      %1937 = vector.extract %1933[3] : f32 from vector<4xf32>
      %1938 = vector.from_elements %1688, %1690, %1692, %1694 : vector<4xf32>
      %1939 = arith.subf %1938, %1902 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1940 = vector.extract %1939[0] : f32 from vector<4xf32>
      %1941 = vector.extract %1939[1] : f32 from vector<4xf32>
      %1942 = vector.extract %1939[2] : f32 from vector<4xf32>
      %1943 = vector.extract %1939[3] : f32 from vector<4xf32>
      %1944 = vector.from_elements %1710, %1712, %1714, %1716 : vector<4xf32>
      %1945 = arith.subf %1944, %1902 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1946 = vector.extract %1945[0] : f32 from vector<4xf32>
      %1947 = vector.extract %1945[1] : f32 from vector<4xf32>
      %1948 = vector.extract %1945[2] : f32 from vector<4xf32>
      %1949 = vector.extract %1945[3] : f32 from vector<4xf32>
      %1950 = vector.from_elements %1732, %1733, %1734, %1735 : vector<4xf32>
      %1951 = vector.broadcast %1900 : f32 to vector<4xf32>
      %1952 = arith.subf %1950, %1951 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1953 = vector.extract %1952[0] : f32 from vector<4xf32>
      %1954 = vector.extract %1952[1] : f32 from vector<4xf32>
      %1955 = vector.extract %1952[2] : f32 from vector<4xf32>
      %1956 = vector.extract %1952[3] : f32 from vector<4xf32>
      %1957 = vector.from_elements %1750, %1751, %1752, %1753 : vector<4xf32>
      %1958 = arith.subf %1957, %1951 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1959 = vector.extract %1958[0] : f32 from vector<4xf32>
      %1960 = vector.extract %1958[1] : f32 from vector<4xf32>
      %1961 = vector.extract %1958[2] : f32 from vector<4xf32>
      %1962 = vector.extract %1958[3] : f32 from vector<4xf32>
      %1963 = vector.from_elements %1768, %1769, %1770, %1771 : vector<4xf32>
      %1964 = arith.subf %1963, %1951 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1965 = vector.extract %1964[0] : f32 from vector<4xf32>
      %1966 = vector.extract %1964[1] : f32 from vector<4xf32>
      %1967 = vector.extract %1964[2] : f32 from vector<4xf32>
      %1968 = vector.extract %1964[3] : f32 from vector<4xf32>
      %1969 = vector.from_elements %1786, %1787, %1788, %1789 : vector<4xf32>
      %1970 = arith.subf %1969, %1951 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1971 = vector.extract %1970[0] : f32 from vector<4xf32>
      %1972 = vector.extract %1970[1] : f32 from vector<4xf32>
      %1973 = vector.extract %1970[2] : f32 from vector<4xf32>
      %1974 = vector.extract %1970[3] : f32 from vector<4xf32>
      %1975 = vector.from_elements %1804, %1805, %1806, %1807 : vector<4xf32>
      %1976 = arith.subf %1975, %1951 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1977 = vector.extract %1976[0] : f32 from vector<4xf32>
      %1978 = vector.extract %1976[1] : f32 from vector<4xf32>
      %1979 = vector.extract %1976[2] : f32 from vector<4xf32>
      %1980 = vector.extract %1976[3] : f32 from vector<4xf32>
      %1981 = vector.from_elements %1822, %1823, %1824, %1825 : vector<4xf32>
      %1982 = arith.subf %1981, %1951 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1983 = vector.extract %1982[0] : f32 from vector<4xf32>
      %1984 = vector.extract %1982[1] : f32 from vector<4xf32>
      %1985 = vector.extract %1982[2] : f32 from vector<4xf32>
      %1986 = vector.extract %1982[3] : f32 from vector<4xf32>
      %1987 = vector.from_elements %1840, %1841, %1842, %1843 : vector<4xf32>
      %1988 = arith.subf %1987, %1951 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1989 = vector.extract %1988[0] : f32 from vector<4xf32>
      %1990 = vector.extract %1988[1] : f32 from vector<4xf32>
      %1991 = vector.extract %1988[2] : f32 from vector<4xf32>
      %1992 = vector.extract %1988[3] : f32 from vector<4xf32>
      %1993 = vector.from_elements %1851, %1852, %1853, %1854 : vector<4xf32>
      %1994 = arith.subf %1993, %1951 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1995 = vector.extract %1994[0] : f32 from vector<4xf32>
      %1996 = vector.extract %1994[1] : f32 from vector<4xf32>
      %1997 = vector.extract %1994[2] : f32 from vector<4xf32>
      %1998 = vector.extract %1994[3] : f32 from vector<4xf32>
      %1999 = vector.broadcast %1895 : f32 to vector<1xf32>
      %2000 = arith.subf %1893, %1999 {origin = "frisk.sub", tiled = true} : vector<1xf32>
      %2001 = math.exp2 %2000 {tiled = true} : vector<1xf32>
      %2002 = vector.extract %2001[0] : f32 from vector<1xf32>
      %2003 = vector.extract %arg6[0, 0] : f32 from vector<2x1xf32>
      %2004 = vector.broadcast %2003 : f32 to vector<1xf32>
      %2005 = arith.mulf %2004, %2001 {origin = "frisk.mul", tiled = true} : vector<1xf32>
      %2006 = vector.extract %2005[0] : f32 from vector<1xf32>
      %2007 = vector.broadcast %1900 : f32 to vector<1xf32>
      %2008 = arith.subf %1898, %2007 {origin = "frisk.sub", tiled = true} : vector<1xf32>
      %2009 = math.exp2 %2008 {tiled = true} : vector<1xf32>
      %2010 = vector.extract %2009[0] : f32 from vector<1xf32>
      %2011 = vector.extract %arg6[1, 0] : f32 from vector<2x1xf32>
      %2012 = vector.broadcast %2011 : f32 to vector<1xf32>
      %2013 = arith.mulf %2012, %2009 {origin = "frisk.mul", tiled = true} : vector<1xf32>
      %2014 = vector.extract %2013[0] : f32 from vector<1xf32>
      %2015 = vector.from_elements %1904, %1905, %1906, %1907 : vector<4xf32>
      %2016 = math.exp2 %2015 {tiled = true} : vector<4xf32>
      %2017 = vector.extract %2016[0] : f32 from vector<4xf32>
      %2018 = vector.extract %2016[1] : f32 from vector<4xf32>
      %2019 = vector.extract %2016[2] : f32 from vector<4xf32>
      %2020 = vector.extract %2016[3] : f32 from vector<4xf32>
      %2021 = vector.from_elements %1910, %1911, %1912, %1913 : vector<4xf32>
      %2022 = math.exp2 %2021 {tiled = true} : vector<4xf32>
      %2023 = vector.extract %2022[0] : f32 from vector<4xf32>
      %2024 = vector.extract %2022[1] : f32 from vector<4xf32>
      %2025 = vector.extract %2022[2] : f32 from vector<4xf32>
      %2026 = vector.extract %2022[3] : f32 from vector<4xf32>
      %2027 = vector.from_elements %1916, %1917, %1918, %1919 : vector<4xf32>
      %2028 = math.exp2 %2027 {tiled = true} : vector<4xf32>
      %2029 = vector.extract %2028[0] : f32 from vector<4xf32>
      %2030 = vector.extract %2028[1] : f32 from vector<4xf32>
      %2031 = vector.extract %2028[2] : f32 from vector<4xf32>
      %2032 = vector.extract %2028[3] : f32 from vector<4xf32>
      %2033 = vector.from_elements %1922, %1923, %1924, %1925 : vector<4xf32>
      %2034 = math.exp2 %2033 {tiled = true} : vector<4xf32>
      %2035 = vector.extract %2034[0] : f32 from vector<4xf32>
      %2036 = vector.extract %2034[1] : f32 from vector<4xf32>
      %2037 = vector.extract %2034[2] : f32 from vector<4xf32>
      %2038 = vector.extract %2034[3] : f32 from vector<4xf32>
      %2039 = vector.from_elements %1928, %1929, %1930, %1931 : vector<4xf32>
      %2040 = math.exp2 %2039 {tiled = true} : vector<4xf32>
      %2041 = vector.extract %2040[0] : f32 from vector<4xf32>
      %2042 = vector.extract %2040[1] : f32 from vector<4xf32>
      %2043 = vector.extract %2040[2] : f32 from vector<4xf32>
      %2044 = vector.extract %2040[3] : f32 from vector<4xf32>
      %2045 = vector.from_elements %1934, %1935, %1936, %1937 : vector<4xf32>
      %2046 = math.exp2 %2045 {tiled = true} : vector<4xf32>
      %2047 = vector.extract %2046[0] : f32 from vector<4xf32>
      %2048 = vector.extract %2046[1] : f32 from vector<4xf32>
      %2049 = vector.extract %2046[2] : f32 from vector<4xf32>
      %2050 = vector.extract %2046[3] : f32 from vector<4xf32>
      %2051 = vector.from_elements %1940, %1941, %1942, %1943 : vector<4xf32>
      %2052 = math.exp2 %2051 {tiled = true} : vector<4xf32>
      %2053 = vector.extract %2052[0] : f32 from vector<4xf32>
      %2054 = vector.extract %2052[1] : f32 from vector<4xf32>
      %2055 = vector.extract %2052[2] : f32 from vector<4xf32>
      %2056 = vector.extract %2052[3] : f32 from vector<4xf32>
      %2057 = vector.from_elements %1946, %1947, %1948, %1949 : vector<4xf32>
      %2058 = math.exp2 %2057 {tiled = true} : vector<4xf32>
      %2059 = vector.extract %2058[0] : f32 from vector<4xf32>
      %2060 = vector.extract %2058[1] : f32 from vector<4xf32>
      %2061 = vector.extract %2058[2] : f32 from vector<4xf32>
      %2062 = vector.extract %2058[3] : f32 from vector<4xf32>
      %2063 = vector.from_elements %1953, %1954, %1955, %1956 : vector<4xf32>
      %2064 = math.exp2 %2063 {tiled = true} : vector<4xf32>
      %2065 = vector.extract %2064[0] : f32 from vector<4xf32>
      %2066 = vector.extract %2064[1] : f32 from vector<4xf32>
      %2067 = vector.extract %2064[2] : f32 from vector<4xf32>
      %2068 = vector.extract %2064[3] : f32 from vector<4xf32>
      %2069 = vector.from_elements %1959, %1960, %1961, %1962 : vector<4xf32>
      %2070 = math.exp2 %2069 {tiled = true} : vector<4xf32>
      %2071 = vector.extract %2070[0] : f32 from vector<4xf32>
      %2072 = vector.extract %2070[1] : f32 from vector<4xf32>
      %2073 = vector.extract %2070[2] : f32 from vector<4xf32>
      %2074 = vector.extract %2070[3] : f32 from vector<4xf32>
      %2075 = vector.from_elements %1965, %1966, %1967, %1968 : vector<4xf32>
      %2076 = math.exp2 %2075 {tiled = true} : vector<4xf32>
      %2077 = vector.extract %2076[0] : f32 from vector<4xf32>
      %2078 = vector.extract %2076[1] : f32 from vector<4xf32>
      %2079 = vector.extract %2076[2] : f32 from vector<4xf32>
      %2080 = vector.extract %2076[3] : f32 from vector<4xf32>
      %2081 = vector.from_elements %1971, %1972, %1973, %1974 : vector<4xf32>
      %2082 = math.exp2 %2081 {tiled = true} : vector<4xf32>
      %2083 = vector.extract %2082[0] : f32 from vector<4xf32>
      %2084 = vector.extract %2082[1] : f32 from vector<4xf32>
      %2085 = vector.extract %2082[2] : f32 from vector<4xf32>
      %2086 = vector.extract %2082[3] : f32 from vector<4xf32>
      %2087 = vector.from_elements %1977, %1978, %1979, %1980 : vector<4xf32>
      %2088 = math.exp2 %2087 {tiled = true} : vector<4xf32>
      %2089 = vector.extract %2088[0] : f32 from vector<4xf32>
      %2090 = vector.extract %2088[1] : f32 from vector<4xf32>
      %2091 = vector.extract %2088[2] : f32 from vector<4xf32>
      %2092 = vector.extract %2088[3] : f32 from vector<4xf32>
      %2093 = vector.from_elements %1983, %1984, %1985, %1986 : vector<4xf32>
      %2094 = math.exp2 %2093 {tiled = true} : vector<4xf32>
      %2095 = vector.extract %2094[0] : f32 from vector<4xf32>
      %2096 = vector.extract %2094[1] : f32 from vector<4xf32>
      %2097 = vector.extract %2094[2] : f32 from vector<4xf32>
      %2098 = vector.extract %2094[3] : f32 from vector<4xf32>
      %2099 = vector.from_elements %1989, %1990, %1991, %1992 : vector<4xf32>
      %2100 = math.exp2 %2099 {tiled = true} : vector<4xf32>
      %2101 = vector.extract %2100[0] : f32 from vector<4xf32>
      %2102 = vector.extract %2100[1] : f32 from vector<4xf32>
      %2103 = vector.extract %2100[2] : f32 from vector<4xf32>
      %2104 = vector.extract %2100[3] : f32 from vector<4xf32>
      %2105 = vector.from_elements %1995, %1996, %1997, %1998 : vector<4xf32>
      %2106 = math.exp2 %2105 {tiled = true} : vector<4xf32>
      %2107 = vector.extract %2106[0] : f32 from vector<4xf32>
      %2108 = vector.extract %2106[1] : f32 from vector<4xf32>
      %2109 = vector.extract %2106[2] : f32 from vector<4xf32>
      %2110 = vector.extract %2106[3] : f32 from vector<4xf32>
      %2111 = arith.addf %2017, %cst_6 : f32
      %2112 = arith.addf %2111, %2018 : f32
      %2113 = arith.addf %2112, %2019 : f32
      %2114 = arith.addf %2113, %2020 : f32
      %2115 = arith.addf %2114, %2023 : f32
      %2116 = arith.addf %2115, %2024 : f32
      %2117 = arith.addf %2116, %2025 : f32
      %2118 = arith.addf %2117, %2026 : f32
      %2119 = arith.addf %2118, %2029 : f32
      %2120 = arith.addf %2119, %2030 : f32
      %2121 = arith.addf %2120, %2031 : f32
      %2122 = arith.addf %2121, %2032 : f32
      %2123 = arith.addf %2122, %2035 : f32
      %2124 = arith.addf %2123, %2036 : f32
      %2125 = arith.addf %2124, %2037 : f32
      %2126 = arith.addf %2125, %2038 : f32
      %2127 = arith.addf %2126, %2041 : f32
      %2128 = arith.addf %2127, %2042 : f32
      %2129 = arith.addf %2128, %2043 : f32
      %2130 = arith.addf %2129, %2044 : f32
      %2131 = arith.addf %2130, %2047 : f32
      %2132 = arith.addf %2131, %2048 : f32
      %2133 = arith.addf %2132, %2049 : f32
      %2134 = arith.addf %2133, %2050 : f32
      %2135 = arith.addf %2134, %2053 : f32
      %2136 = arith.addf %2135, %2054 : f32
      %2137 = arith.addf %2136, %2055 : f32
      %2138 = arith.addf %2137, %2056 : f32
      %2139 = arith.addf %2138, %2059 : f32
      %2140 = arith.addf %2139, %2060 : f32
      %2141 = arith.addf %2140, %2061 : f32
      %2142 = arith.addf %2141, %2062 : f32
      %shuffleResult_150, %valid_151 = gpu.shuffle xor %2142, %c16_i32, %c64_i32 : f32
      %2143 = arith.addf %2142, %shuffleResult_150 : f32
      %shuffleResult_152, %valid_153 = gpu.shuffle xor %2143, %c32_i32, %c64_i32 : f32
      %2144 = arith.addf %2143, %shuffleResult_152 : f32
      %2145 = arith.addf %2065, %cst_6 : f32
      %2146 = arith.addf %2145, %2066 : f32
      %2147 = arith.addf %2146, %2067 : f32
      %2148 = arith.addf %2147, %2068 : f32
      %2149 = arith.addf %2148, %2071 : f32
      %2150 = arith.addf %2149, %2072 : f32
      %2151 = arith.addf %2150, %2073 : f32
      %2152 = arith.addf %2151, %2074 : f32
      %2153 = arith.addf %2152, %2077 : f32
      %2154 = arith.addf %2153, %2078 : f32
      %2155 = arith.addf %2154, %2079 : f32
      %2156 = arith.addf %2155, %2080 : f32
      %2157 = arith.addf %2156, %2083 : f32
      %2158 = arith.addf %2157, %2084 : f32
      %2159 = arith.addf %2158, %2085 : f32
      %2160 = arith.addf %2159, %2086 : f32
      %2161 = arith.addf %2160, %2089 : f32
      %2162 = arith.addf %2161, %2090 : f32
      %2163 = arith.addf %2162, %2091 : f32
      %2164 = arith.addf %2163, %2092 : f32
      %2165 = arith.addf %2164, %2095 : f32
      %2166 = arith.addf %2165, %2096 : f32
      %2167 = arith.addf %2166, %2097 : f32
      %2168 = arith.addf %2167, %2098 : f32
      %2169 = arith.addf %2168, %2101 : f32
      %2170 = arith.addf %2169, %2102 : f32
      %2171 = arith.addf %2170, %2103 : f32
      %2172 = arith.addf %2171, %2104 : f32
      %2173 = arith.addf %2172, %2107 : f32
      %2174 = arith.addf %2173, %2108 : f32
      %2175 = arith.addf %2174, %2109 : f32
      %2176 = arith.addf %2175, %2110 : f32
      %shuffleResult_154, %valid_155 = gpu.shuffle xor %2176, %c16_i32, %c64_i32 : f32
      %2177 = arith.addf %2176, %shuffleResult_154 : f32
      %shuffleResult_156, %valid_157 = gpu.shuffle xor %2177, %c32_i32, %c64_i32 : f32
      %2178 = arith.addf %2177, %shuffleResult_156 : f32
      %2179 = vector.broadcast %2144 : f32 to vector<1xf32>
      %2180 = vector.broadcast %2006 : f32 to vector<1xf32>
      %2181 = arith.addf %2179, %2180 {origin = "frisk.add", tiled = true} : vector<1xf32>
      %2182 = vector.extract %2181[0] : f32 from vector<1xf32>
      %2183 = vector.broadcast %2178 : f32 to vector<1xf32>
      %2184 = vector.broadcast %2014 : f32 to vector<1xf32>
      %2185 = arith.addf %2183, %2184 {origin = "frisk.add", tiled = true} : vector<1xf32>
      %2186 = vector.extract %2185[0] : f32 from vector<1xf32>
      %2187 = vector.from_elements %2017, %2018, %2019, %2020 : vector<4xf32>
      %2188 = arith.truncf %2187 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2189 = vector.extract %2188[0] : f16 from vector<4xf16>
      %2190 = vector.extract %2188[1] : f16 from vector<4xf16>
      %2191 = vector.extract %2188[2] : f16 from vector<4xf16>
      %2192 = vector.extract %2188[3] : f16 from vector<4xf16>
      %2193 = vector.from_elements %2023, %2024, %2025, %2026 : vector<4xf32>
      %2194 = arith.truncf %2193 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2195 = vector.extract %2194[0] : f16 from vector<4xf16>
      %2196 = vector.extract %2194[1] : f16 from vector<4xf16>
      %2197 = vector.extract %2194[2] : f16 from vector<4xf16>
      %2198 = vector.extract %2194[3] : f16 from vector<4xf16>
      %2199 = vector.from_elements %2029, %2030, %2031, %2032 : vector<4xf32>
      %2200 = arith.truncf %2199 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2201 = vector.extract %2200[0] : f16 from vector<4xf16>
      %2202 = vector.extract %2200[1] : f16 from vector<4xf16>
      %2203 = vector.extract %2200[2] : f16 from vector<4xf16>
      %2204 = vector.extract %2200[3] : f16 from vector<4xf16>
      %2205 = vector.from_elements %2035, %2036, %2037, %2038 : vector<4xf32>
      %2206 = arith.truncf %2205 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2207 = vector.extract %2206[0] : f16 from vector<4xf16>
      %2208 = vector.extract %2206[1] : f16 from vector<4xf16>
      %2209 = vector.extract %2206[2] : f16 from vector<4xf16>
      %2210 = vector.extract %2206[3] : f16 from vector<4xf16>
      %2211 = vector.from_elements %2041, %2042, %2043, %2044 : vector<4xf32>
      %2212 = arith.truncf %2211 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2213 = vector.extract %2212[0] : f16 from vector<4xf16>
      %2214 = vector.extract %2212[1] : f16 from vector<4xf16>
      %2215 = vector.extract %2212[2] : f16 from vector<4xf16>
      %2216 = vector.extract %2212[3] : f16 from vector<4xf16>
      %2217 = vector.from_elements %2047, %2048, %2049, %2050 : vector<4xf32>
      %2218 = arith.truncf %2217 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2219 = vector.extract %2218[0] : f16 from vector<4xf16>
      %2220 = vector.extract %2218[1] : f16 from vector<4xf16>
      %2221 = vector.extract %2218[2] : f16 from vector<4xf16>
      %2222 = vector.extract %2218[3] : f16 from vector<4xf16>
      %2223 = vector.from_elements %2053, %2054, %2055, %2056 : vector<4xf32>
      %2224 = arith.truncf %2223 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2225 = vector.extract %2224[0] : f16 from vector<4xf16>
      %2226 = vector.extract %2224[1] : f16 from vector<4xf16>
      %2227 = vector.extract %2224[2] : f16 from vector<4xf16>
      %2228 = vector.extract %2224[3] : f16 from vector<4xf16>
      %2229 = vector.from_elements %2059, %2060, %2061, %2062 : vector<4xf32>
      %2230 = arith.truncf %2229 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2231 = vector.extract %2230[0] : f16 from vector<4xf16>
      %2232 = vector.extract %2230[1] : f16 from vector<4xf16>
      %2233 = vector.extract %2230[2] : f16 from vector<4xf16>
      %2234 = vector.extract %2230[3] : f16 from vector<4xf16>
      %2235 = vector.from_elements %2065, %2066, %2067, %2068 : vector<4xf32>
      %2236 = arith.truncf %2235 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2237 = vector.extract %2236[0] : f16 from vector<4xf16>
      %2238 = vector.extract %2236[1] : f16 from vector<4xf16>
      %2239 = vector.extract %2236[2] : f16 from vector<4xf16>
      %2240 = vector.extract %2236[3] : f16 from vector<4xf16>
      %2241 = vector.from_elements %2071, %2072, %2073, %2074 : vector<4xf32>
      %2242 = arith.truncf %2241 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2243 = vector.extract %2242[0] : f16 from vector<4xf16>
      %2244 = vector.extract %2242[1] : f16 from vector<4xf16>
      %2245 = vector.extract %2242[2] : f16 from vector<4xf16>
      %2246 = vector.extract %2242[3] : f16 from vector<4xf16>
      %2247 = vector.from_elements %2077, %2078, %2079, %2080 : vector<4xf32>
      %2248 = arith.truncf %2247 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2249 = vector.extract %2248[0] : f16 from vector<4xf16>
      %2250 = vector.extract %2248[1] : f16 from vector<4xf16>
      %2251 = vector.extract %2248[2] : f16 from vector<4xf16>
      %2252 = vector.extract %2248[3] : f16 from vector<4xf16>
      %2253 = vector.from_elements %2083, %2084, %2085, %2086 : vector<4xf32>
      %2254 = arith.truncf %2253 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2255 = vector.extract %2254[0] : f16 from vector<4xf16>
      %2256 = vector.extract %2254[1] : f16 from vector<4xf16>
      %2257 = vector.extract %2254[2] : f16 from vector<4xf16>
      %2258 = vector.extract %2254[3] : f16 from vector<4xf16>
      %2259 = vector.from_elements %2089, %2090, %2091, %2092 : vector<4xf32>
      %2260 = arith.truncf %2259 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2261 = vector.extract %2260[0] : f16 from vector<4xf16>
      %2262 = vector.extract %2260[1] : f16 from vector<4xf16>
      %2263 = vector.extract %2260[2] : f16 from vector<4xf16>
      %2264 = vector.extract %2260[3] : f16 from vector<4xf16>
      %2265 = vector.from_elements %2095, %2096, %2097, %2098 : vector<4xf32>
      %2266 = arith.truncf %2265 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2267 = vector.extract %2266[0] : f16 from vector<4xf16>
      %2268 = vector.extract %2266[1] : f16 from vector<4xf16>
      %2269 = vector.extract %2266[2] : f16 from vector<4xf16>
      %2270 = vector.extract %2266[3] : f16 from vector<4xf16>
      %2271 = vector.from_elements %2101, %2102, %2103, %2104 : vector<4xf32>
      %2272 = arith.truncf %2271 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2273 = vector.extract %2272[0] : f16 from vector<4xf16>
      %2274 = vector.extract %2272[1] : f16 from vector<4xf16>
      %2275 = vector.extract %2272[2] : f16 from vector<4xf16>
      %2276 = vector.extract %2272[3] : f16 from vector<4xf16>
      %2277 = vector.from_elements %2107, %2108, %2109, %2110 : vector<4xf32>
      %2278 = arith.truncf %2277 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2279 = vector.extract %2278[0] : f16 from vector<4xf16>
      %2280 = vector.extract %2278[1] : f16 from vector<4xf16>
      %2281 = vector.extract %2278[2] : f16 from vector<4xf16>
      %2282 = vector.extract %2278[3] : f16 from vector<4xf16>
      %2283 = vector.extract %arg5[0, 0] : f32 from vector<2x8xf32>
      %2284 = vector.extract %arg5[0, 1] : f32 from vector<2x8xf32>
      %2285 = vector.extract %arg5[0, 2] : f32 from vector<2x8xf32>
      %2286 = vector.extract %arg5[0, 3] : f32 from vector<2x8xf32>
      %2287 = vector.from_elements %2283, %2284, %2285, %2286 : vector<4xf32>
      %2288 = vector.broadcast %2002 : f32 to vector<4xf32>
      %2289 = arith.mulf %2287, %2288 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2290 = vector.extract %2289[0] : f32 from vector<4xf32>
      %2291 = vector.extract %2289[1] : f32 from vector<4xf32>
      %2292 = vector.extract %2289[2] : f32 from vector<4xf32>
      %2293 = vector.extract %2289[3] : f32 from vector<4xf32>
      %2294 = vector.extract %arg5[0, 4] : f32 from vector<2x8xf32>
      %2295 = vector.extract %arg5[0, 5] : f32 from vector<2x8xf32>
      %2296 = vector.extract %arg5[0, 6] : f32 from vector<2x8xf32>
      %2297 = vector.extract %arg5[0, 7] : f32 from vector<2x8xf32>
      %2298 = vector.from_elements %2294, %2295, %2296, %2297 : vector<4xf32>
      %2299 = arith.mulf %2298, %2288 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2300 = vector.extract %2299[0] : f32 from vector<4xf32>
      %2301 = vector.extract %2299[1] : f32 from vector<4xf32>
      %2302 = vector.extract %2299[2] : f32 from vector<4xf32>
      %2303 = vector.extract %2299[3] : f32 from vector<4xf32>
      %2304 = vector.extract %arg5[1, 0] : f32 from vector<2x8xf32>
      %2305 = vector.extract %arg5[1, 1] : f32 from vector<2x8xf32>
      %2306 = vector.extract %arg5[1, 2] : f32 from vector<2x8xf32>
      %2307 = vector.extract %arg5[1, 3] : f32 from vector<2x8xf32>
      %2308 = vector.from_elements %2304, %2305, %2306, %2307 : vector<4xf32>
      %2309 = vector.broadcast %2010 : f32 to vector<4xf32>
      %2310 = arith.mulf %2308, %2309 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2311 = vector.extract %2310[0] : f32 from vector<4xf32>
      %2312 = vector.extract %2310[1] : f32 from vector<4xf32>
      %2313 = vector.extract %2310[2] : f32 from vector<4xf32>
      %2314 = vector.extract %2310[3] : f32 from vector<4xf32>
      %2315 = vector.extract %arg5[1, 4] : f32 from vector<2x8xf32>
      %2316 = vector.extract %arg5[1, 5] : f32 from vector<2x8xf32>
      %2317 = vector.extract %arg5[1, 6] : f32 from vector<2x8xf32>
      %2318 = vector.extract %arg5[1, 7] : f32 from vector<2x8xf32>
      %2319 = vector.from_elements %2315, %2316, %2317, %2318 : vector<4xf32>
      %2320 = arith.mulf %2319, %2309 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2321 = vector.extract %2320[0] : f32 from vector<4xf32>
      %2322 = vector.extract %2320[1] : f32 from vector<4xf32>
      %2323 = vector.extract %2320[2] : f32 from vector<4xf32>
      %2324 = vector.extract %2320[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %1514 : !frisk.async_task
      %2325 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>()[%thread_id_x]
      %2326 = frisk.read_matrix_from_shm %view[%2325] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2327 = vector.bitcast %2326 : vector<4xi32> to vector<8xf16>
      %2328 = vector.extract %2327[0] : f16 from vector<8xf16>
      %2329 = vector.extract %2327[%199] : f16 from vector<8xf16>
      %2330 = vector.extract %2327[1] : f16 from vector<8xf16>
      %2331 = vector.extract %2327[%200] : f16 from vector<8xf16>
      %2332 = vector.extract %2327[2] : f16 from vector<8xf16>
      %2333 = vector.extract %2327[%201] : f16 from vector<8xf16>
      %2334 = vector.extract %2327[3] : f16 from vector<8xf16>
      %2335 = vector.extract %2327[%202] : f16 from vector<8xf16>
      %2336 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 16) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 128)>()[%thread_id_x]
      %2337 = frisk.read_matrix_from_shm %view[%2336] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2338 = vector.bitcast %2337 : vector<4xi32> to vector<8xf16>
      %2339 = vector.extract %2338[0] : f16 from vector<8xf16>
      %2340 = vector.extract %2338[%199] : f16 from vector<8xf16>
      %2341 = vector.extract %2338[1] : f16 from vector<8xf16>
      %2342 = vector.extract %2338[%200] : f16 from vector<8xf16>
      %2343 = vector.extract %2338[2] : f16 from vector<8xf16>
      %2344 = vector.extract %2338[%201] : f16 from vector<8xf16>
      %2345 = vector.extract %2338[3] : f16 from vector<8xf16>
      %2346 = vector.extract %2338[%202] : f16 from vector<8xf16>
      %2347 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 32) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 256)>()[%thread_id_x]
      %2348 = frisk.read_matrix_from_shm %view[%2347] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2349 = vector.bitcast %2348 : vector<4xi32> to vector<8xf16>
      %2350 = vector.extract %2349[0] : f16 from vector<8xf16>
      %2351 = vector.extract %2349[%199] : f16 from vector<8xf16>
      %2352 = vector.extract %2349[1] : f16 from vector<8xf16>
      %2353 = vector.extract %2349[%200] : f16 from vector<8xf16>
      %2354 = vector.extract %2349[2] : f16 from vector<8xf16>
      %2355 = vector.extract %2349[%201] : f16 from vector<8xf16>
      %2356 = vector.extract %2349[3] : f16 from vector<8xf16>
      %2357 = vector.extract %2349[%202] : f16 from vector<8xf16>
      %2358 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 48) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 384)>()[%thread_id_x]
      %2359 = frisk.read_matrix_from_shm %view[%2358] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2360 = vector.bitcast %2359 : vector<4xi32> to vector<8xf16>
      %2361 = vector.extract %2360[0] : f16 from vector<8xf16>
      %2362 = vector.extract %2360[%199] : f16 from vector<8xf16>
      %2363 = vector.extract %2360[1] : f16 from vector<8xf16>
      %2364 = vector.extract %2360[%200] : f16 from vector<8xf16>
      %2365 = vector.extract %2360[2] : f16 from vector<8xf16>
      %2366 = vector.extract %2360[%201] : f16 from vector<8xf16>
      %2367 = vector.extract %2360[3] : f16 from vector<8xf16>
      %2368 = vector.extract %2360[%202] : f16 from vector<8xf16>
      %2369 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 2048)>()[%thread_id_x]
      %2370 = frisk.read_matrix_from_shm %view[%2369] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2371 = vector.bitcast %2370 : vector<4xi32> to vector<8xf16>
      %2372 = vector.extract %2371[0] : f16 from vector<8xf16>
      %2373 = vector.extract %2371[%199] : f16 from vector<8xf16>
      %2374 = vector.extract %2371[1] : f16 from vector<8xf16>
      %2375 = vector.extract %2371[%200] : f16 from vector<8xf16>
      %2376 = vector.extract %2371[2] : f16 from vector<8xf16>
      %2377 = vector.extract %2371[%201] : f16 from vector<8xf16>
      %2378 = vector.extract %2371[3] : f16 from vector<8xf16>
      %2379 = vector.extract %2371[%202] : f16 from vector<8xf16>
      %2380 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 80) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 128)>()[%thread_id_x]
      %2381 = frisk.read_matrix_from_shm %view[%2380] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2382 = vector.bitcast %2381 : vector<4xi32> to vector<8xf16>
      %2383 = vector.extract %2382[0] : f16 from vector<8xf16>
      %2384 = vector.extract %2382[%199] : f16 from vector<8xf16>
      %2385 = vector.extract %2382[1] : f16 from vector<8xf16>
      %2386 = vector.extract %2382[%200] : f16 from vector<8xf16>
      %2387 = vector.extract %2382[2] : f16 from vector<8xf16>
      %2388 = vector.extract %2382[%201] : f16 from vector<8xf16>
      %2389 = vector.extract %2382[3] : f16 from vector<8xf16>
      %2390 = vector.extract %2382[%202] : f16 from vector<8xf16>
      %2391 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 96) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 256)>()[%thread_id_x]
      %2392 = frisk.read_matrix_from_shm %view[%2391] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2393 = vector.bitcast %2392 : vector<4xi32> to vector<8xf16>
      %2394 = vector.extract %2393[0] : f16 from vector<8xf16>
      %2395 = vector.extract %2393[%199] : f16 from vector<8xf16>
      %2396 = vector.extract %2393[1] : f16 from vector<8xf16>
      %2397 = vector.extract %2393[%200] : f16 from vector<8xf16>
      %2398 = vector.extract %2393[2] : f16 from vector<8xf16>
      %2399 = vector.extract %2393[%201] : f16 from vector<8xf16>
      %2400 = vector.extract %2393[3] : f16 from vector<8xf16>
      %2401 = vector.extract %2393[%202] : f16 from vector<8xf16>
      %2402 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 112) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 384)>()[%thread_id_x]
      %2403 = frisk.read_matrix_from_shm %view[%2402] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2404 = vector.bitcast %2403 : vector<4xi32> to vector<8xf16>
      %2405 = vector.extract %2404[0] : f16 from vector<8xf16>
      %2406 = vector.extract %2404[%199] : f16 from vector<8xf16>
      %2407 = vector.extract %2404[1] : f16 from vector<8xf16>
      %2408 = vector.extract %2404[%200] : f16 from vector<8xf16>
      %2409 = vector.extract %2404[2] : f16 from vector<8xf16>
      %2410 = vector.extract %2404[%201] : f16 from vector<8xf16>
      %2411 = vector.extract %2404[3] : f16 from vector<8xf16>
      %2412 = vector.extract %2404[%202] : f16 from vector<8xf16>
      %subview_158 = memref.subview %arg2[0, %198, %585, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      %2413 = frisk.async_copy_gm_to_shm %subview_158 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %2414 = vector.from_elements %2189, %2190, %2191, %2192 : vector<4xf16>
      %2415 = vector.from_elements %2328, %2330, %2332, %2334 : vector<4xf16>
      %2416 = vector.from_elements %2290, %2291, %2292, %2293 : vector<4xf32>
      %2417 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2414, %2415, %2416 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2418 = vector.from_elements %2195, %2196, %2197, %2198 : vector<4xf16>
      %2419 = vector.from_elements %2339, %2341, %2343, %2345 : vector<4xf16>
      %2420 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2418, %2419, %2417 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2421 = vector.from_elements %2201, %2202, %2203, %2204 : vector<4xf16>
      %2422 = vector.from_elements %2350, %2352, %2354, %2356 : vector<4xf16>
      %2423 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2421, %2422, %2420 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2424 = vector.from_elements %2207, %2208, %2209, %2210 : vector<4xf16>
      %2425 = vector.from_elements %2361, %2363, %2365, %2367 : vector<4xf16>
      %2426 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2424, %2425, %2423 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2427 = vector.from_elements %2213, %2214, %2215, %2216 : vector<4xf16>
      %2428 = vector.from_elements %2372, %2374, %2376, %2378 : vector<4xf16>
      %2429 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2427, %2428, %2426 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2430 = vector.from_elements %2219, %2220, %2221, %2222 : vector<4xf16>
      %2431 = vector.from_elements %2383, %2385, %2387, %2389 : vector<4xf16>
      %2432 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2430, %2431, %2429 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2433 = vector.from_elements %2225, %2226, %2227, %2228 : vector<4xf16>
      %2434 = vector.from_elements %2394, %2396, %2398, %2400 : vector<4xf16>
      %2435 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2433, %2434, %2432 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2436 = vector.from_elements %2231, %2232, %2233, %2234 : vector<4xf16>
      %2437 = vector.from_elements %2405, %2407, %2409, %2411 : vector<4xf16>
      %2438 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2436, %2437, %2435 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2439 = vector.extract %2438[0] : f32 from vector<4xf32>
      %2440 = vector.extract %2438[1] : f32 from vector<4xf32>
      %2441 = vector.extract %2438[2] : f32 from vector<4xf32>
      %2442 = vector.extract %2438[3] : f32 from vector<4xf32>
      %2443 = vector.from_elements %2329, %2331, %2333, %2335 : vector<4xf16>
      %2444 = vector.from_elements %2300, %2301, %2302, %2303 : vector<4xf32>
      %2445 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2414, %2443, %2444 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2446 = vector.from_elements %2340, %2342, %2344, %2346 : vector<4xf16>
      %2447 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2418, %2446, %2445 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2448 = vector.from_elements %2351, %2353, %2355, %2357 : vector<4xf16>
      %2449 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2421, %2448, %2447 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2450 = vector.from_elements %2362, %2364, %2366, %2368 : vector<4xf16>
      %2451 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2424, %2450, %2449 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2452 = vector.from_elements %2373, %2375, %2377, %2379 : vector<4xf16>
      %2453 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2427, %2452, %2451 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2454 = vector.from_elements %2384, %2386, %2388, %2390 : vector<4xf16>
      %2455 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2430, %2454, %2453 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2456 = vector.from_elements %2395, %2397, %2399, %2401 : vector<4xf16>
      %2457 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2433, %2456, %2455 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2458 = vector.from_elements %2406, %2408, %2410, %2412 : vector<4xf16>
      %2459 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2436, %2458, %2457 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2460 = vector.extract %2459[0] : f32 from vector<4xf32>
      %2461 = vector.extract %2459[1] : f32 from vector<4xf32>
      %2462 = vector.extract %2459[2] : f32 from vector<4xf32>
      %2463 = vector.extract %2459[3] : f32 from vector<4xf32>
      %2464 = vector.from_elements %2237, %2238, %2239, %2240 : vector<4xf16>
      %2465 = vector.from_elements %2311, %2312, %2313, %2314 : vector<4xf32>
      %2466 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2464, %2415, %2465 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2467 = vector.from_elements %2243, %2244, %2245, %2246 : vector<4xf16>
      %2468 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2419, %2466 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2469 = vector.from_elements %2249, %2250, %2251, %2252 : vector<4xf16>
      %2470 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2422, %2468 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2471 = vector.from_elements %2255, %2256, %2257, %2258 : vector<4xf16>
      %2472 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2425, %2470 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2473 = vector.from_elements %2261, %2262, %2263, %2264 : vector<4xf16>
      %2474 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2428, %2472 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2475 = vector.from_elements %2267, %2268, %2269, %2270 : vector<4xf16>
      %2476 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2431, %2474 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2477 = vector.from_elements %2273, %2274, %2275, %2276 : vector<4xf16>
      %2478 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2434, %2476 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2479 = vector.from_elements %2279, %2280, %2281, %2282 : vector<4xf16>
      %2480 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2479, %2437, %2478 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2481 = vector.extract %2480[0] : f32 from vector<4xf32>
      %2482 = vector.extract %2480[1] : f32 from vector<4xf32>
      %2483 = vector.extract %2480[2] : f32 from vector<4xf32>
      %2484 = vector.extract %2480[3] : f32 from vector<4xf32>
      %2485 = vector.from_elements %2321, %2322, %2323, %2324 : vector<4xf32>
      %2486 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2464, %2443, %2485 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2487 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2446, %2486 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2488 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2448, %2487 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2489 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2450, %2488 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2490 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2452, %2489 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2491 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2454, %2490 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2492 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2456, %2491 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2493 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2479, %2458, %2492 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2494 = vector.extract %2493[0] : f32 from vector<4xf32>
      %2495 = vector.extract %2493[1] : f32 from vector<4xf32>
      %2496 = vector.extract %2493[2] : f32 from vector<4xf32>
      %2497 = vector.extract %2493[3] : f32 from vector<4xf32>
      %2498 = vector.extract %arg8[0, 0] : f32 from vector<2x8xf32>
      %2499 = vector.extract %arg8[0, 1] : f32 from vector<2x8xf32>
      %2500 = vector.extract %arg8[0, 2] : f32 from vector<2x8xf32>
      %2501 = vector.extract %arg8[0, 3] : f32 from vector<2x8xf32>
      %2502 = vector.from_elements %2498, %2499, %2500, %2501 : vector<4xf32>
      %2503 = arith.mulf %2502, %2288 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2504 = vector.extract %2503[0] : f32 from vector<4xf32>
      %2505 = vector.extract %2503[1] : f32 from vector<4xf32>
      %2506 = vector.extract %2503[2] : f32 from vector<4xf32>
      %2507 = vector.extract %2503[3] : f32 from vector<4xf32>
      %2508 = vector.extract %arg8[0, 4] : f32 from vector<2x8xf32>
      %2509 = vector.extract %arg8[0, 5] : f32 from vector<2x8xf32>
      %2510 = vector.extract %arg8[0, 6] : f32 from vector<2x8xf32>
      %2511 = vector.extract %arg8[0, 7] : f32 from vector<2x8xf32>
      %2512 = vector.from_elements %2508, %2509, %2510, %2511 : vector<4xf32>
      %2513 = arith.mulf %2512, %2288 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2514 = vector.extract %2513[0] : f32 from vector<4xf32>
      %2515 = vector.extract %2513[1] : f32 from vector<4xf32>
      %2516 = vector.extract %2513[2] : f32 from vector<4xf32>
      %2517 = vector.extract %2513[3] : f32 from vector<4xf32>
      %2518 = vector.extract %arg8[1, 0] : f32 from vector<2x8xf32>
      %2519 = vector.extract %arg8[1, 1] : f32 from vector<2x8xf32>
      %2520 = vector.extract %arg8[1, 2] : f32 from vector<2x8xf32>
      %2521 = vector.extract %arg8[1, 3] : f32 from vector<2x8xf32>
      %2522 = vector.from_elements %2518, %2519, %2520, %2521 : vector<4xf32>
      %2523 = arith.mulf %2522, %2309 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2524 = vector.extract %2523[0] : f32 from vector<4xf32>
      %2525 = vector.extract %2523[1] : f32 from vector<4xf32>
      %2526 = vector.extract %2523[2] : f32 from vector<4xf32>
      %2527 = vector.extract %2523[3] : f32 from vector<4xf32>
      %2528 = vector.extract %arg8[1, 4] : f32 from vector<2x8xf32>
      %2529 = vector.extract %arg8[1, 5] : f32 from vector<2x8xf32>
      %2530 = vector.extract %arg8[1, 6] : f32 from vector<2x8xf32>
      %2531 = vector.extract %arg8[1, 7] : f32 from vector<2x8xf32>
      %2532 = vector.from_elements %2528, %2529, %2530, %2531 : vector<4xf32>
      %2533 = arith.mulf %2532, %2309 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2534 = vector.extract %2533[0] : f32 from vector<4xf32>
      %2535 = vector.extract %2533[1] : f32 from vector<4xf32>
      %2536 = vector.extract %2533[2] : f32 from vector<4xf32>
      %2537 = vector.extract %2533[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %2413 : !frisk.async_task
      %2538 = frisk.read_matrix_from_shm %view[%2325] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2539 = vector.bitcast %2538 : vector<4xi32> to vector<8xf16>
      %2540 = vector.extract %2539[0] : f16 from vector<8xf16>
      %2541 = vector.extract %2539[%199] : f16 from vector<8xf16>
      %2542 = vector.extract %2539[1] : f16 from vector<8xf16>
      %2543 = vector.extract %2539[%200] : f16 from vector<8xf16>
      %2544 = vector.extract %2539[2] : f16 from vector<8xf16>
      %2545 = vector.extract %2539[%201] : f16 from vector<8xf16>
      %2546 = vector.extract %2539[3] : f16 from vector<8xf16>
      %2547 = vector.extract %2539[%202] : f16 from vector<8xf16>
      %2548 = frisk.read_matrix_from_shm %view[%2336] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2549 = vector.bitcast %2548 : vector<4xi32> to vector<8xf16>
      %2550 = vector.extract %2549[0] : f16 from vector<8xf16>
      %2551 = vector.extract %2549[%199] : f16 from vector<8xf16>
      %2552 = vector.extract %2549[1] : f16 from vector<8xf16>
      %2553 = vector.extract %2549[%200] : f16 from vector<8xf16>
      %2554 = vector.extract %2549[2] : f16 from vector<8xf16>
      %2555 = vector.extract %2549[%201] : f16 from vector<8xf16>
      %2556 = vector.extract %2549[3] : f16 from vector<8xf16>
      %2557 = vector.extract %2549[%202] : f16 from vector<8xf16>
      %2558 = frisk.read_matrix_from_shm %view[%2347] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2559 = vector.bitcast %2558 : vector<4xi32> to vector<8xf16>
      %2560 = vector.extract %2559[0] : f16 from vector<8xf16>
      %2561 = vector.extract %2559[%199] : f16 from vector<8xf16>
      %2562 = vector.extract %2559[1] : f16 from vector<8xf16>
      %2563 = vector.extract %2559[%200] : f16 from vector<8xf16>
      %2564 = vector.extract %2559[2] : f16 from vector<8xf16>
      %2565 = vector.extract %2559[%201] : f16 from vector<8xf16>
      %2566 = vector.extract %2559[3] : f16 from vector<8xf16>
      %2567 = vector.extract %2559[%202] : f16 from vector<8xf16>
      %2568 = frisk.read_matrix_from_shm %view[%2358] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2569 = vector.bitcast %2568 : vector<4xi32> to vector<8xf16>
      %2570 = vector.extract %2569[0] : f16 from vector<8xf16>
      %2571 = vector.extract %2569[%199] : f16 from vector<8xf16>
      %2572 = vector.extract %2569[1] : f16 from vector<8xf16>
      %2573 = vector.extract %2569[%200] : f16 from vector<8xf16>
      %2574 = vector.extract %2569[2] : f16 from vector<8xf16>
      %2575 = vector.extract %2569[%201] : f16 from vector<8xf16>
      %2576 = vector.extract %2569[3] : f16 from vector<8xf16>
      %2577 = vector.extract %2569[%202] : f16 from vector<8xf16>
      %2578 = frisk.read_matrix_from_shm %view[%2369] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2579 = vector.bitcast %2578 : vector<4xi32> to vector<8xf16>
      %2580 = vector.extract %2579[0] : f16 from vector<8xf16>
      %2581 = vector.extract %2579[%199] : f16 from vector<8xf16>
      %2582 = vector.extract %2579[1] : f16 from vector<8xf16>
      %2583 = vector.extract %2579[%200] : f16 from vector<8xf16>
      %2584 = vector.extract %2579[2] : f16 from vector<8xf16>
      %2585 = vector.extract %2579[%201] : f16 from vector<8xf16>
      %2586 = vector.extract %2579[3] : f16 from vector<8xf16>
      %2587 = vector.extract %2579[%202] : f16 from vector<8xf16>
      %2588 = frisk.read_matrix_from_shm %view[%2380] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2589 = vector.bitcast %2588 : vector<4xi32> to vector<8xf16>
      %2590 = vector.extract %2589[0] : f16 from vector<8xf16>
      %2591 = vector.extract %2589[%199] : f16 from vector<8xf16>
      %2592 = vector.extract %2589[1] : f16 from vector<8xf16>
      %2593 = vector.extract %2589[%200] : f16 from vector<8xf16>
      %2594 = vector.extract %2589[2] : f16 from vector<8xf16>
      %2595 = vector.extract %2589[%201] : f16 from vector<8xf16>
      %2596 = vector.extract %2589[3] : f16 from vector<8xf16>
      %2597 = vector.extract %2589[%202] : f16 from vector<8xf16>
      %2598 = frisk.read_matrix_from_shm %view[%2391] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2599 = vector.bitcast %2598 : vector<4xi32> to vector<8xf16>
      %2600 = vector.extract %2599[0] : f16 from vector<8xf16>
      %2601 = vector.extract %2599[%199] : f16 from vector<8xf16>
      %2602 = vector.extract %2599[1] : f16 from vector<8xf16>
      %2603 = vector.extract %2599[%200] : f16 from vector<8xf16>
      %2604 = vector.extract %2599[2] : f16 from vector<8xf16>
      %2605 = vector.extract %2599[%201] : f16 from vector<8xf16>
      %2606 = vector.extract %2599[3] : f16 from vector<8xf16>
      %2607 = vector.extract %2599[%202] : f16 from vector<8xf16>
      %2608 = frisk.read_matrix_from_shm %view[%2402] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2609 = vector.bitcast %2608 : vector<4xi32> to vector<8xf16>
      %2610 = vector.extract %2609[0] : f16 from vector<8xf16>
      %2611 = vector.extract %2609[%199] : f16 from vector<8xf16>
      %2612 = vector.extract %2609[1] : f16 from vector<8xf16>
      %2613 = vector.extract %2609[%200] : f16 from vector<8xf16>
      %2614 = vector.extract %2609[2] : f16 from vector<8xf16>
      %2615 = vector.extract %2609[%201] : f16 from vector<8xf16>
      %2616 = vector.extract %2609[3] : f16 from vector<8xf16>
      %2617 = vector.extract %2609[%202] : f16 from vector<8xf16>
      %subview_159 = memref.subview %arg2[0, %198, %585, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      %2618 = frisk.async_copy_gm_to_shm %subview_159 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %2619 = vector.from_elements %2540, %2542, %2544, %2546 : vector<4xf16>
      %2620 = vector.from_elements %2504, %2505, %2506, %2507 : vector<4xf32>
      %2621 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2414, %2619, %2620 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2622 = vector.from_elements %2550, %2552, %2554, %2556 : vector<4xf16>
      %2623 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2418, %2622, %2621 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2624 = vector.from_elements %2560, %2562, %2564, %2566 : vector<4xf16>
      %2625 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2421, %2624, %2623 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2626 = vector.from_elements %2570, %2572, %2574, %2576 : vector<4xf16>
      %2627 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2424, %2626, %2625 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2628 = vector.from_elements %2580, %2582, %2584, %2586 : vector<4xf16>
      %2629 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2427, %2628, %2627 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2630 = vector.from_elements %2590, %2592, %2594, %2596 : vector<4xf16>
      %2631 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2430, %2630, %2629 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2632 = vector.from_elements %2600, %2602, %2604, %2606 : vector<4xf16>
      %2633 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2433, %2632, %2631 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2634 = vector.from_elements %2610, %2612, %2614, %2616 : vector<4xf16>
      %2635 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2436, %2634, %2633 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2636 = vector.extract %2635[0] : f32 from vector<4xf32>
      %2637 = vector.extract %2635[1] : f32 from vector<4xf32>
      %2638 = vector.extract %2635[2] : f32 from vector<4xf32>
      %2639 = vector.extract %2635[3] : f32 from vector<4xf32>
      %2640 = vector.from_elements %2541, %2543, %2545, %2547 : vector<4xf16>
      %2641 = vector.from_elements %2514, %2515, %2516, %2517 : vector<4xf32>
      %2642 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2414, %2640, %2641 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2643 = vector.from_elements %2551, %2553, %2555, %2557 : vector<4xf16>
      %2644 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2418, %2643, %2642 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2645 = vector.from_elements %2561, %2563, %2565, %2567 : vector<4xf16>
      %2646 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2421, %2645, %2644 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2647 = vector.from_elements %2571, %2573, %2575, %2577 : vector<4xf16>
      %2648 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2424, %2647, %2646 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2649 = vector.from_elements %2581, %2583, %2585, %2587 : vector<4xf16>
      %2650 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2427, %2649, %2648 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2651 = vector.from_elements %2591, %2593, %2595, %2597 : vector<4xf16>
      %2652 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2430, %2651, %2650 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2653 = vector.from_elements %2601, %2603, %2605, %2607 : vector<4xf16>
      %2654 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2433, %2653, %2652 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2655 = vector.from_elements %2611, %2613, %2615, %2617 : vector<4xf16>
      %2656 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2436, %2655, %2654 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2657 = vector.extract %2656[0] : f32 from vector<4xf32>
      %2658 = vector.extract %2656[1] : f32 from vector<4xf32>
      %2659 = vector.extract %2656[2] : f32 from vector<4xf32>
      %2660 = vector.extract %2656[3] : f32 from vector<4xf32>
      %2661 = vector.from_elements %2524, %2525, %2526, %2527 : vector<4xf32>
      %2662 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2464, %2619, %2661 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2663 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2622, %2662 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2664 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2624, %2663 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2665 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2626, %2664 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2666 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2628, %2665 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2667 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2630, %2666 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2668 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2632, %2667 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2669 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2479, %2634, %2668 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2670 = vector.extract %2669[0] : f32 from vector<4xf32>
      %2671 = vector.extract %2669[1] : f32 from vector<4xf32>
      %2672 = vector.extract %2669[2] : f32 from vector<4xf32>
      %2673 = vector.extract %2669[3] : f32 from vector<4xf32>
      %2674 = vector.from_elements %2534, %2535, %2536, %2537 : vector<4xf32>
      %2675 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2464, %2640, %2674 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2676 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2643, %2675 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2677 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2645, %2676 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2678 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2647, %2677 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2679 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2649, %2678 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2680 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2651, %2679 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2681 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2653, %2680 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2682 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2479, %2655, %2681 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2683 = vector.extract %2682[0] : f32 from vector<4xf32>
      %2684 = vector.extract %2682[1] : f32 from vector<4xf32>
      %2685 = vector.extract %2682[2] : f32 from vector<4xf32>
      %2686 = vector.extract %2682[3] : f32 from vector<4xf32>
      %2687 = vector.extract %arg9[0, 0] : f32 from vector<2x8xf32>
      %2688 = vector.extract %arg9[0, 1] : f32 from vector<2x8xf32>
      %2689 = vector.extract %arg9[0, 2] : f32 from vector<2x8xf32>
      %2690 = vector.extract %arg9[0, 3] : f32 from vector<2x8xf32>
      %2691 = vector.from_elements %2687, %2688, %2689, %2690 : vector<4xf32>
      %2692 = arith.mulf %2691, %2288 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2693 = vector.extract %2692[0] : f32 from vector<4xf32>
      %2694 = vector.extract %2692[1] : f32 from vector<4xf32>
      %2695 = vector.extract %2692[2] : f32 from vector<4xf32>
      %2696 = vector.extract %2692[3] : f32 from vector<4xf32>
      %2697 = vector.extract %arg9[0, 4] : f32 from vector<2x8xf32>
      %2698 = vector.extract %arg9[0, 5] : f32 from vector<2x8xf32>
      %2699 = vector.extract %arg9[0, 6] : f32 from vector<2x8xf32>
      %2700 = vector.extract %arg9[0, 7] : f32 from vector<2x8xf32>
      %2701 = vector.from_elements %2697, %2698, %2699, %2700 : vector<4xf32>
      %2702 = arith.mulf %2701, %2288 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2703 = vector.extract %2702[0] : f32 from vector<4xf32>
      %2704 = vector.extract %2702[1] : f32 from vector<4xf32>
      %2705 = vector.extract %2702[2] : f32 from vector<4xf32>
      %2706 = vector.extract %2702[3] : f32 from vector<4xf32>
      %2707 = vector.extract %arg9[1, 0] : f32 from vector<2x8xf32>
      %2708 = vector.extract %arg9[1, 1] : f32 from vector<2x8xf32>
      %2709 = vector.extract %arg9[1, 2] : f32 from vector<2x8xf32>
      %2710 = vector.extract %arg9[1, 3] : f32 from vector<2x8xf32>
      %2711 = vector.from_elements %2707, %2708, %2709, %2710 : vector<4xf32>
      %2712 = arith.mulf %2711, %2309 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2713 = vector.extract %2712[0] : f32 from vector<4xf32>
      %2714 = vector.extract %2712[1] : f32 from vector<4xf32>
      %2715 = vector.extract %2712[2] : f32 from vector<4xf32>
      %2716 = vector.extract %2712[3] : f32 from vector<4xf32>
      %2717 = vector.extract %arg9[1, 4] : f32 from vector<2x8xf32>
      %2718 = vector.extract %arg9[1, 5] : f32 from vector<2x8xf32>
      %2719 = vector.extract %arg9[1, 6] : f32 from vector<2x8xf32>
      %2720 = vector.extract %arg9[1, 7] : f32 from vector<2x8xf32>
      %2721 = vector.from_elements %2717, %2718, %2719, %2720 : vector<4xf32>
      %2722 = arith.mulf %2721, %2309 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2723 = vector.extract %2722[0] : f32 from vector<4xf32>
      %2724 = vector.extract %2722[1] : f32 from vector<4xf32>
      %2725 = vector.extract %2722[2] : f32 from vector<4xf32>
      %2726 = vector.extract %2722[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %2618 : !frisk.async_task
      %2727 = frisk.read_matrix_from_shm %view[%2325] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2728 = vector.bitcast %2727 : vector<4xi32> to vector<8xf16>
      %2729 = vector.extract %2728[0] : f16 from vector<8xf16>
      %2730 = vector.extract %2728[%199] : f16 from vector<8xf16>
      %2731 = vector.extract %2728[1] : f16 from vector<8xf16>
      %2732 = vector.extract %2728[%200] : f16 from vector<8xf16>
      %2733 = vector.extract %2728[2] : f16 from vector<8xf16>
      %2734 = vector.extract %2728[%201] : f16 from vector<8xf16>
      %2735 = vector.extract %2728[3] : f16 from vector<8xf16>
      %2736 = vector.extract %2728[%202] : f16 from vector<8xf16>
      %2737 = frisk.read_matrix_from_shm %view[%2336] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2738 = vector.bitcast %2737 : vector<4xi32> to vector<8xf16>
      %2739 = vector.extract %2738[0] : f16 from vector<8xf16>
      %2740 = vector.extract %2738[%199] : f16 from vector<8xf16>
      %2741 = vector.extract %2738[1] : f16 from vector<8xf16>
      %2742 = vector.extract %2738[%200] : f16 from vector<8xf16>
      %2743 = vector.extract %2738[2] : f16 from vector<8xf16>
      %2744 = vector.extract %2738[%201] : f16 from vector<8xf16>
      %2745 = vector.extract %2738[3] : f16 from vector<8xf16>
      %2746 = vector.extract %2738[%202] : f16 from vector<8xf16>
      %2747 = frisk.read_matrix_from_shm %view[%2347] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2748 = vector.bitcast %2747 : vector<4xi32> to vector<8xf16>
      %2749 = vector.extract %2748[0] : f16 from vector<8xf16>
      %2750 = vector.extract %2748[%199] : f16 from vector<8xf16>
      %2751 = vector.extract %2748[1] : f16 from vector<8xf16>
      %2752 = vector.extract %2748[%200] : f16 from vector<8xf16>
      %2753 = vector.extract %2748[2] : f16 from vector<8xf16>
      %2754 = vector.extract %2748[%201] : f16 from vector<8xf16>
      %2755 = vector.extract %2748[3] : f16 from vector<8xf16>
      %2756 = vector.extract %2748[%202] : f16 from vector<8xf16>
      %2757 = frisk.read_matrix_from_shm %view[%2358] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2758 = vector.bitcast %2757 : vector<4xi32> to vector<8xf16>
      %2759 = vector.extract %2758[0] : f16 from vector<8xf16>
      %2760 = vector.extract %2758[%199] : f16 from vector<8xf16>
      %2761 = vector.extract %2758[1] : f16 from vector<8xf16>
      %2762 = vector.extract %2758[%200] : f16 from vector<8xf16>
      %2763 = vector.extract %2758[2] : f16 from vector<8xf16>
      %2764 = vector.extract %2758[%201] : f16 from vector<8xf16>
      %2765 = vector.extract %2758[3] : f16 from vector<8xf16>
      %2766 = vector.extract %2758[%202] : f16 from vector<8xf16>
      %2767 = frisk.read_matrix_from_shm %view[%2369] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2768 = vector.bitcast %2767 : vector<4xi32> to vector<8xf16>
      %2769 = vector.extract %2768[0] : f16 from vector<8xf16>
      %2770 = vector.extract %2768[%199] : f16 from vector<8xf16>
      %2771 = vector.extract %2768[1] : f16 from vector<8xf16>
      %2772 = vector.extract %2768[%200] : f16 from vector<8xf16>
      %2773 = vector.extract %2768[2] : f16 from vector<8xf16>
      %2774 = vector.extract %2768[%201] : f16 from vector<8xf16>
      %2775 = vector.extract %2768[3] : f16 from vector<8xf16>
      %2776 = vector.extract %2768[%202] : f16 from vector<8xf16>
      %2777 = frisk.read_matrix_from_shm %view[%2380] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2778 = vector.bitcast %2777 : vector<4xi32> to vector<8xf16>
      %2779 = vector.extract %2778[0] : f16 from vector<8xf16>
      %2780 = vector.extract %2778[%199] : f16 from vector<8xf16>
      %2781 = vector.extract %2778[1] : f16 from vector<8xf16>
      %2782 = vector.extract %2778[%200] : f16 from vector<8xf16>
      %2783 = vector.extract %2778[2] : f16 from vector<8xf16>
      %2784 = vector.extract %2778[%201] : f16 from vector<8xf16>
      %2785 = vector.extract %2778[3] : f16 from vector<8xf16>
      %2786 = vector.extract %2778[%202] : f16 from vector<8xf16>
      %2787 = frisk.read_matrix_from_shm %view[%2391] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2788 = vector.bitcast %2787 : vector<4xi32> to vector<8xf16>
      %2789 = vector.extract %2788[0] : f16 from vector<8xf16>
      %2790 = vector.extract %2788[%199] : f16 from vector<8xf16>
      %2791 = vector.extract %2788[1] : f16 from vector<8xf16>
      %2792 = vector.extract %2788[%200] : f16 from vector<8xf16>
      %2793 = vector.extract %2788[2] : f16 from vector<8xf16>
      %2794 = vector.extract %2788[%201] : f16 from vector<8xf16>
      %2795 = vector.extract %2788[3] : f16 from vector<8xf16>
      %2796 = vector.extract %2788[%202] : f16 from vector<8xf16>
      %2797 = frisk.read_matrix_from_shm %view[%2402] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2798 = vector.bitcast %2797 : vector<4xi32> to vector<8xf16>
      %2799 = vector.extract %2798[0] : f16 from vector<8xf16>
      %2800 = vector.extract %2798[%199] : f16 from vector<8xf16>
      %2801 = vector.extract %2798[1] : f16 from vector<8xf16>
      %2802 = vector.extract %2798[%200] : f16 from vector<8xf16>
      %2803 = vector.extract %2798[2] : f16 from vector<8xf16>
      %2804 = vector.extract %2798[%201] : f16 from vector<8xf16>
      %2805 = vector.extract %2798[3] : f16 from vector<8xf16>
      %2806 = vector.extract %2798[%202] : f16 from vector<8xf16>
      %subview_160 = memref.subview %arg2[0, %198, %585, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      %2807 = frisk.async_copy_gm_to_shm %subview_160 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %2808 = vector.from_elements %2729, %2731, %2733, %2735 : vector<4xf16>
      %2809 = vector.from_elements %2693, %2694, %2695, %2696 : vector<4xf32>
      %2810 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2414, %2808, %2809 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2811 = vector.from_elements %2739, %2741, %2743, %2745 : vector<4xf16>
      %2812 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2418, %2811, %2810 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2813 = vector.from_elements %2749, %2751, %2753, %2755 : vector<4xf16>
      %2814 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2421, %2813, %2812 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2815 = vector.from_elements %2759, %2761, %2763, %2765 : vector<4xf16>
      %2816 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2424, %2815, %2814 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2817 = vector.from_elements %2769, %2771, %2773, %2775 : vector<4xf16>
      %2818 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2427, %2817, %2816 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2819 = vector.from_elements %2779, %2781, %2783, %2785 : vector<4xf16>
      %2820 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2430, %2819, %2818 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2821 = vector.from_elements %2789, %2791, %2793, %2795 : vector<4xf16>
      %2822 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2433, %2821, %2820 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2823 = vector.from_elements %2799, %2801, %2803, %2805 : vector<4xf16>
      %2824 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2436, %2823, %2822 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2825 = vector.extract %2824[0] : f32 from vector<4xf32>
      %2826 = vector.extract %2824[1] : f32 from vector<4xf32>
      %2827 = vector.extract %2824[2] : f32 from vector<4xf32>
      %2828 = vector.extract %2824[3] : f32 from vector<4xf32>
      %2829 = vector.from_elements %2730, %2732, %2734, %2736 : vector<4xf16>
      %2830 = vector.from_elements %2703, %2704, %2705, %2706 : vector<4xf32>
      %2831 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2414, %2829, %2830 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2832 = vector.from_elements %2740, %2742, %2744, %2746 : vector<4xf16>
      %2833 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2418, %2832, %2831 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2834 = vector.from_elements %2750, %2752, %2754, %2756 : vector<4xf16>
      %2835 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2421, %2834, %2833 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2836 = vector.from_elements %2760, %2762, %2764, %2766 : vector<4xf16>
      %2837 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2424, %2836, %2835 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2838 = vector.from_elements %2770, %2772, %2774, %2776 : vector<4xf16>
      %2839 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2427, %2838, %2837 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2840 = vector.from_elements %2780, %2782, %2784, %2786 : vector<4xf16>
      %2841 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2430, %2840, %2839 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2842 = vector.from_elements %2790, %2792, %2794, %2796 : vector<4xf16>
      %2843 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2433, %2842, %2841 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2844 = vector.from_elements %2800, %2802, %2804, %2806 : vector<4xf16>
      %2845 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2436, %2844, %2843 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2846 = vector.extract %2845[0] : f32 from vector<4xf32>
      %2847 = vector.extract %2845[1] : f32 from vector<4xf32>
      %2848 = vector.extract %2845[2] : f32 from vector<4xf32>
      %2849 = vector.extract %2845[3] : f32 from vector<4xf32>
      %2850 = vector.from_elements %2713, %2714, %2715, %2716 : vector<4xf32>
      %2851 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2464, %2808, %2850 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2852 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2811, %2851 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2853 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2813, %2852 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2854 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2815, %2853 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2855 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2817, %2854 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2856 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2819, %2855 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2857 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2821, %2856 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2858 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2479, %2823, %2857 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2859 = vector.extract %2858[0] : f32 from vector<4xf32>
      %2860 = vector.extract %2858[1] : f32 from vector<4xf32>
      %2861 = vector.extract %2858[2] : f32 from vector<4xf32>
      %2862 = vector.extract %2858[3] : f32 from vector<4xf32>
      %2863 = vector.from_elements %2723, %2724, %2725, %2726 : vector<4xf32>
      %2864 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2464, %2829, %2863 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2865 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2832, %2864 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2866 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2834, %2865 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2867 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2836, %2866 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2868 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2838, %2867 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2869 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2840, %2868 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2870 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2842, %2869 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2871 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2479, %2844, %2870 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2872 = vector.extract %2871[0] : f32 from vector<4xf32>
      %2873 = vector.extract %2871[1] : f32 from vector<4xf32>
      %2874 = vector.extract %2871[2] : f32 from vector<4xf32>
      %2875 = vector.extract %2871[3] : f32 from vector<4xf32>
      %2876 = vector.extract %arg10[0, 0] : f32 from vector<2x8xf32>
      %2877 = vector.extract %arg10[0, 1] : f32 from vector<2x8xf32>
      %2878 = vector.extract %arg10[0, 2] : f32 from vector<2x8xf32>
      %2879 = vector.extract %arg10[0, 3] : f32 from vector<2x8xf32>
      %2880 = vector.from_elements %2876, %2877, %2878, %2879 : vector<4xf32>
      %2881 = arith.mulf %2880, %2288 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2882 = vector.extract %2881[0] : f32 from vector<4xf32>
      %2883 = vector.extract %2881[1] : f32 from vector<4xf32>
      %2884 = vector.extract %2881[2] : f32 from vector<4xf32>
      %2885 = vector.extract %2881[3] : f32 from vector<4xf32>
      %2886 = vector.extract %arg10[0, 4] : f32 from vector<2x8xf32>
      %2887 = vector.extract %arg10[0, 5] : f32 from vector<2x8xf32>
      %2888 = vector.extract %arg10[0, 6] : f32 from vector<2x8xf32>
      %2889 = vector.extract %arg10[0, 7] : f32 from vector<2x8xf32>
      %2890 = vector.from_elements %2886, %2887, %2888, %2889 : vector<4xf32>
      %2891 = arith.mulf %2890, %2288 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2892 = vector.extract %2891[0] : f32 from vector<4xf32>
      %2893 = vector.extract %2891[1] : f32 from vector<4xf32>
      %2894 = vector.extract %2891[2] : f32 from vector<4xf32>
      %2895 = vector.extract %2891[3] : f32 from vector<4xf32>
      %2896 = vector.extract %arg10[1, 0] : f32 from vector<2x8xf32>
      %2897 = vector.extract %arg10[1, 1] : f32 from vector<2x8xf32>
      %2898 = vector.extract %arg10[1, 2] : f32 from vector<2x8xf32>
      %2899 = vector.extract %arg10[1, 3] : f32 from vector<2x8xf32>
      %2900 = vector.from_elements %2896, %2897, %2898, %2899 : vector<4xf32>
      %2901 = arith.mulf %2900, %2309 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2902 = vector.extract %2901[0] : f32 from vector<4xf32>
      %2903 = vector.extract %2901[1] : f32 from vector<4xf32>
      %2904 = vector.extract %2901[2] : f32 from vector<4xf32>
      %2905 = vector.extract %2901[3] : f32 from vector<4xf32>
      %2906 = vector.extract %arg10[1, 4] : f32 from vector<2x8xf32>
      %2907 = vector.extract %arg10[1, 5] : f32 from vector<2x8xf32>
      %2908 = vector.extract %arg10[1, 6] : f32 from vector<2x8xf32>
      %2909 = vector.extract %arg10[1, 7] : f32 from vector<2x8xf32>
      %2910 = vector.from_elements %2906, %2907, %2908, %2909 : vector<4xf32>
      %2911 = arith.mulf %2910, %2309 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2912 = vector.extract %2911[0] : f32 from vector<4xf32>
      %2913 = vector.extract %2911[1] : f32 from vector<4xf32>
      %2914 = vector.extract %2911[2] : f32 from vector<4xf32>
      %2915 = vector.extract %2911[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %2807 : !frisk.async_task
      %2916 = frisk.read_matrix_from_shm %view[%2325] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2917 = vector.bitcast %2916 : vector<4xi32> to vector<8xf16>
      %2918 = vector.extract %2917[0] : f16 from vector<8xf16>
      %2919 = vector.extract %2917[%199] : f16 from vector<8xf16>
      %2920 = vector.extract %2917[1] : f16 from vector<8xf16>
      %2921 = vector.extract %2917[%200] : f16 from vector<8xf16>
      %2922 = vector.extract %2917[2] : f16 from vector<8xf16>
      %2923 = vector.extract %2917[%201] : f16 from vector<8xf16>
      %2924 = vector.extract %2917[3] : f16 from vector<8xf16>
      %2925 = vector.extract %2917[%202] : f16 from vector<8xf16>
      %2926 = frisk.read_matrix_from_shm %view[%2336] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2927 = vector.bitcast %2926 : vector<4xi32> to vector<8xf16>
      %2928 = vector.extract %2927[0] : f16 from vector<8xf16>
      %2929 = vector.extract %2927[%199] : f16 from vector<8xf16>
      %2930 = vector.extract %2927[1] : f16 from vector<8xf16>
      %2931 = vector.extract %2927[%200] : f16 from vector<8xf16>
      %2932 = vector.extract %2927[2] : f16 from vector<8xf16>
      %2933 = vector.extract %2927[%201] : f16 from vector<8xf16>
      %2934 = vector.extract %2927[3] : f16 from vector<8xf16>
      %2935 = vector.extract %2927[%202] : f16 from vector<8xf16>
      %2936 = frisk.read_matrix_from_shm %view[%2347] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2937 = vector.bitcast %2936 : vector<4xi32> to vector<8xf16>
      %2938 = vector.extract %2937[0] : f16 from vector<8xf16>
      %2939 = vector.extract %2937[%199] : f16 from vector<8xf16>
      %2940 = vector.extract %2937[1] : f16 from vector<8xf16>
      %2941 = vector.extract %2937[%200] : f16 from vector<8xf16>
      %2942 = vector.extract %2937[2] : f16 from vector<8xf16>
      %2943 = vector.extract %2937[%201] : f16 from vector<8xf16>
      %2944 = vector.extract %2937[3] : f16 from vector<8xf16>
      %2945 = vector.extract %2937[%202] : f16 from vector<8xf16>
      %2946 = frisk.read_matrix_from_shm %view[%2358] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2947 = vector.bitcast %2946 : vector<4xi32> to vector<8xf16>
      %2948 = vector.extract %2947[0] : f16 from vector<8xf16>
      %2949 = vector.extract %2947[%199] : f16 from vector<8xf16>
      %2950 = vector.extract %2947[1] : f16 from vector<8xf16>
      %2951 = vector.extract %2947[%200] : f16 from vector<8xf16>
      %2952 = vector.extract %2947[2] : f16 from vector<8xf16>
      %2953 = vector.extract %2947[%201] : f16 from vector<8xf16>
      %2954 = vector.extract %2947[3] : f16 from vector<8xf16>
      %2955 = vector.extract %2947[%202] : f16 from vector<8xf16>
      %2956 = frisk.read_matrix_from_shm %view[%2369] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2957 = vector.bitcast %2956 : vector<4xi32> to vector<8xf16>
      %2958 = vector.extract %2957[0] : f16 from vector<8xf16>
      %2959 = vector.extract %2957[%199] : f16 from vector<8xf16>
      %2960 = vector.extract %2957[1] : f16 from vector<8xf16>
      %2961 = vector.extract %2957[%200] : f16 from vector<8xf16>
      %2962 = vector.extract %2957[2] : f16 from vector<8xf16>
      %2963 = vector.extract %2957[%201] : f16 from vector<8xf16>
      %2964 = vector.extract %2957[3] : f16 from vector<8xf16>
      %2965 = vector.extract %2957[%202] : f16 from vector<8xf16>
      %2966 = frisk.read_matrix_from_shm %view[%2380] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2967 = vector.bitcast %2966 : vector<4xi32> to vector<8xf16>
      %2968 = vector.extract %2967[0] : f16 from vector<8xf16>
      %2969 = vector.extract %2967[%199] : f16 from vector<8xf16>
      %2970 = vector.extract %2967[1] : f16 from vector<8xf16>
      %2971 = vector.extract %2967[%200] : f16 from vector<8xf16>
      %2972 = vector.extract %2967[2] : f16 from vector<8xf16>
      %2973 = vector.extract %2967[%201] : f16 from vector<8xf16>
      %2974 = vector.extract %2967[3] : f16 from vector<8xf16>
      %2975 = vector.extract %2967[%202] : f16 from vector<8xf16>
      %2976 = frisk.read_matrix_from_shm %view[%2391] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2977 = vector.bitcast %2976 : vector<4xi32> to vector<8xf16>
      %2978 = vector.extract %2977[0] : f16 from vector<8xf16>
      %2979 = vector.extract %2977[%199] : f16 from vector<8xf16>
      %2980 = vector.extract %2977[1] : f16 from vector<8xf16>
      %2981 = vector.extract %2977[%200] : f16 from vector<8xf16>
      %2982 = vector.extract %2977[2] : f16 from vector<8xf16>
      %2983 = vector.extract %2977[%201] : f16 from vector<8xf16>
      %2984 = vector.extract %2977[3] : f16 from vector<8xf16>
      %2985 = vector.extract %2977[%202] : f16 from vector<8xf16>
      %2986 = frisk.read_matrix_from_shm %view[%2402] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2987 = vector.bitcast %2986 : vector<4xi32> to vector<8xf16>
      %2988 = vector.extract %2987[0] : f16 from vector<8xf16>
      %2989 = vector.extract %2987[%199] : f16 from vector<8xf16>
      %2990 = vector.extract %2987[1] : f16 from vector<8xf16>
      %2991 = vector.extract %2987[%200] : f16 from vector<8xf16>
      %2992 = vector.extract %2987[2] : f16 from vector<8xf16>
      %2993 = vector.extract %2987[%201] : f16 from vector<8xf16>
      %2994 = vector.extract %2987[3] : f16 from vector<8xf16>
      %2995 = vector.extract %2987[%202] : f16 from vector<8xf16>
      %2996 = vector.from_elements %2918, %2920, %2922, %2924 : vector<4xf16>
      %2997 = vector.from_elements %2882, %2883, %2884, %2885 : vector<4xf32>
      %2998 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2414, %2996, %2997 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2999 = vector.from_elements %2928, %2930, %2932, %2934 : vector<4xf16>
      %3000 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2418, %2999, %2998 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3001 = vector.from_elements %2938, %2940, %2942, %2944 : vector<4xf16>
      %3002 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2421, %3001, %3000 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3003 = vector.from_elements %2948, %2950, %2952, %2954 : vector<4xf16>
      %3004 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2424, %3003, %3002 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3005 = vector.from_elements %2958, %2960, %2962, %2964 : vector<4xf16>
      %3006 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2427, %3005, %3004 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3007 = vector.from_elements %2968, %2970, %2972, %2974 : vector<4xf16>
      %3008 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2430, %3007, %3006 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3009 = vector.from_elements %2978, %2980, %2982, %2984 : vector<4xf16>
      %3010 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2433, %3009, %3008 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3011 = vector.from_elements %2988, %2990, %2992, %2994 : vector<4xf16>
      %3012 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2436, %3011, %3010 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3013 = vector.extract %3012[0] : f32 from vector<4xf32>
      %3014 = vector.extract %3012[1] : f32 from vector<4xf32>
      %3015 = vector.extract %3012[2] : f32 from vector<4xf32>
      %3016 = vector.extract %3012[3] : f32 from vector<4xf32>
      %3017 = vector.from_elements %2919, %2921, %2923, %2925 : vector<4xf16>
      %3018 = vector.from_elements %2892, %2893, %2894, %2895 : vector<4xf32>
      %3019 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2414, %3017, %3018 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3020 = vector.from_elements %2929, %2931, %2933, %2935 : vector<4xf16>
      %3021 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2418, %3020, %3019 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3022 = vector.from_elements %2939, %2941, %2943, %2945 : vector<4xf16>
      %3023 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2421, %3022, %3021 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3024 = vector.from_elements %2949, %2951, %2953, %2955 : vector<4xf16>
      %3025 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2424, %3024, %3023 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3026 = vector.from_elements %2959, %2961, %2963, %2965 : vector<4xf16>
      %3027 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2427, %3026, %3025 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3028 = vector.from_elements %2969, %2971, %2973, %2975 : vector<4xf16>
      %3029 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2430, %3028, %3027 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3030 = vector.from_elements %2979, %2981, %2983, %2985 : vector<4xf16>
      %3031 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2433, %3030, %3029 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3032 = vector.from_elements %2989, %2991, %2993, %2995 : vector<4xf16>
      %3033 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2436, %3032, %3031 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3034 = vector.extract %3033[0] : f32 from vector<4xf32>
      %3035 = vector.extract %3033[1] : f32 from vector<4xf32>
      %3036 = vector.extract %3033[2] : f32 from vector<4xf32>
      %3037 = vector.extract %3033[3] : f32 from vector<4xf32>
      %3038 = vector.from_elements %2902, %2903, %2904, %2905 : vector<4xf32>
      %3039 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2464, %2996, %3038 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3040 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2999, %3039 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3041 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %3001, %3040 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3042 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %3003, %3041 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3043 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %3005, %3042 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3044 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %3007, %3043 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3045 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %3009, %3044 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3046 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2479, %3011, %3045 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3047 = vector.extract %3046[0] : f32 from vector<4xf32>
      %3048 = vector.extract %3046[1] : f32 from vector<4xf32>
      %3049 = vector.extract %3046[2] : f32 from vector<4xf32>
      %3050 = vector.extract %3046[3] : f32 from vector<4xf32>
      %3051 = vector.from_elements %2912, %2913, %2914, %2915 : vector<4xf32>
      %3052 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2464, %3017, %3051 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3053 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %3020, %3052 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3054 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %3022, %3053 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3055 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %3024, %3054 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3056 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %3026, %3055 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3057 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %3028, %3056 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3058 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %3030, %3057 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3059 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2479, %3032, %3058 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3060 = vector.extract %3059[0] : f32 from vector<4xf32>
      %3061 = vector.extract %3059[1] : f32 from vector<4xf32>
      %3062 = vector.extract %3059[2] : f32 from vector<4xf32>
      %3063 = vector.extract %3059[3] : f32 from vector<4xf32>
      %3064 = vector.from_elements %2439, %2440, %2441, %2442, %2460, %2461, %2462, %2463, %2481, %2482, %2483, %2484, %2494, %2495, %2496, %2497 : vector<2x8xf32>
      %3065 = vector.from_elements %2182, %2186 : vector<2x1xf32>
      %3066 = vector.from_elements %1895, %1900 : vector<2x1xf32>
      %3067 = vector.from_elements %2636, %2637, %2638, %2639, %2657, %2658, %2659, %2660, %2670, %2671, %2672, %2673, %2683, %2684, %2685, %2686 : vector<2x8xf32>
      %3068 = vector.from_elements %2825, %2826, %2827, %2828, %2846, %2847, %2848, %2849, %2859, %2860, %2861, %2862, %2872, %2873, %2874, %2875 : vector<2x8xf32>
      %3069 = vector.from_elements %3013, %3014, %3015, %3016, %3034, %3035, %3036, %3037, %3047, %3048, %3049, %3050, %3060, %3061, %3062, %3063 : vector<2x8xf32>
      gpu.barrier
      affine.yield %3064, %3065, %3066, %3067, %3068, %3069 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %204 = arith.index_cast %thread_id_x : index to i32
    %205 = arith.andi %204, %c16_i32 : i32
    %206 = arith.cmpi ne, %205, %c0_i32 : i32
    %207 = arith.andi %204, %c32_i32 : i32
    %208 = arith.cmpi ne, %207, %c0_i32 : i32
    %209 = vector.extract %203#5[0, 0] : f32 from vector<2x8xf32>
    %210 = vector.extract %203#5[0, 1] : f32 from vector<2x8xf32>
    %211 = vector.extract %203#5[0, 2] : f32 from vector<2x8xf32>
    %212 = vector.extract %203#5[0, 3] : f32 from vector<2x8xf32>
    %213 = arith.select %206, %209, %210 : f32
    %shuffleResult, %valid = gpu.shuffle xor %213, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %214 = arith.select %206, %shuffleResult, %209 : f32
    %215 = arith.select %206, %210, %shuffleResult : f32
    %216 = arith.select %206, %211, %212 : f32
    %shuffleResult_11, %valid_12 = gpu.shuffle xor %216, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %217 = arith.select %206, %shuffleResult_11, %211 : f32
    %218 = arith.select %206, %212, %shuffleResult_11 : f32
    %219 = arith.select %208, %214, %217 : f32
    %shuffleResult_13, %valid_14 = gpu.shuffle xor %219, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %220 = arith.select %208, %shuffleResult_13, %214 : f32
    %221 = arith.select %208, %217, %shuffleResult_13 : f32
    %222 = arith.select %208, %215, %218 : f32
    %shuffleResult_15, %valid_16 = gpu.shuffle xor %222, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %223 = arith.select %208, %shuffleResult_15, %215 : f32
    %224 = arith.select %208, %218, %shuffleResult_15 : f32
    %225 = vector.extract %203#5[0, 4] : f32 from vector<2x8xf32>
    %226 = vector.extract %203#5[0, 5] : f32 from vector<2x8xf32>
    %227 = vector.extract %203#5[0, 6] : f32 from vector<2x8xf32>
    %228 = vector.extract %203#5[0, 7] : f32 from vector<2x8xf32>
    %229 = arith.select %206, %225, %226 : f32
    %shuffleResult_17, %valid_18 = gpu.shuffle xor %229, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %230 = arith.select %206, %shuffleResult_17, %225 : f32
    %231 = arith.select %206, %226, %shuffleResult_17 : f32
    %232 = arith.select %206, %227, %228 : f32
    %shuffleResult_19, %valid_20 = gpu.shuffle xor %232, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %233 = arith.select %206, %shuffleResult_19, %227 : f32
    %234 = arith.select %206, %228, %shuffleResult_19 : f32
    %235 = arith.select %208, %230, %233 : f32
    %shuffleResult_21, %valid_22 = gpu.shuffle xor %235, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %236 = arith.select %208, %shuffleResult_21, %230 : f32
    %237 = arith.select %208, %233, %shuffleResult_21 : f32
    %238 = arith.select %208, %231, %234 : f32
    %shuffleResult_23, %valid_24 = gpu.shuffle xor %238, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %239 = arith.select %208, %shuffleResult_23, %231 : f32
    %240 = arith.select %208, %234, %shuffleResult_23 : f32
    %241 = vector.extract %203#5[1, 0] : f32 from vector<2x8xf32>
    %242 = vector.extract %203#5[1, 1] : f32 from vector<2x8xf32>
    %243 = vector.extract %203#5[1, 2] : f32 from vector<2x8xf32>
    %244 = vector.extract %203#5[1, 3] : f32 from vector<2x8xf32>
    %245 = arith.select %206, %241, %242 : f32
    %shuffleResult_25, %valid_26 = gpu.shuffle xor %245, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %246 = arith.select %206, %shuffleResult_25, %241 : f32
    %247 = arith.select %206, %242, %shuffleResult_25 : f32
    %248 = arith.select %206, %243, %244 : f32
    %shuffleResult_27, %valid_28 = gpu.shuffle xor %248, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %249 = arith.select %206, %shuffleResult_27, %243 : f32
    %250 = arith.select %206, %244, %shuffleResult_27 : f32
    %251 = arith.select %208, %246, %249 : f32
    %shuffleResult_29, %valid_30 = gpu.shuffle xor %251, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %252 = arith.select %208, %shuffleResult_29, %246 : f32
    %253 = arith.select %208, %249, %shuffleResult_29 : f32
    %254 = arith.select %208, %247, %250 : f32
    %shuffleResult_31, %valid_32 = gpu.shuffle xor %254, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %255 = arith.select %208, %shuffleResult_31, %247 : f32
    %256 = arith.select %208, %250, %shuffleResult_31 : f32
    %257 = vector.extract %203#5[1, 4] : f32 from vector<2x8xf32>
    %258 = vector.extract %203#5[1, 5] : f32 from vector<2x8xf32>
    %259 = vector.extract %203#5[1, 6] : f32 from vector<2x8xf32>
    %260 = vector.extract %203#5[1, 7] : f32 from vector<2x8xf32>
    %261 = arith.select %206, %257, %258 : f32
    %shuffleResult_33, %valid_34 = gpu.shuffle xor %261, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %262 = arith.select %206, %shuffleResult_33, %257 : f32
    %263 = arith.select %206, %258, %shuffleResult_33 : f32
    %264 = arith.select %206, %259, %260 : f32
    %shuffleResult_35, %valid_36 = gpu.shuffle xor %264, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %265 = arith.select %206, %shuffleResult_35, %259 : f32
    %266 = arith.select %206, %260, %shuffleResult_35 : f32
    %267 = arith.select %208, %262, %265 : f32
    %shuffleResult_37, %valid_38 = gpu.shuffle xor %267, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %268 = arith.select %208, %shuffleResult_37, %262 : f32
    %269 = arith.select %208, %265, %shuffleResult_37 : f32
    %270 = arith.select %208, %263, %266 : f32
    %shuffleResult_39, %valid_40 = gpu.shuffle xor %270, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %271 = arith.select %208, %shuffleResult_39, %263 : f32
    %272 = arith.select %208, %266, %shuffleResult_39 : f32
    %273 = vector.extract %203#4[0, 0] : f32 from vector<2x8xf32>
    %274 = vector.extract %203#4[0, 1] : f32 from vector<2x8xf32>
    %275 = vector.extract %203#4[0, 2] : f32 from vector<2x8xf32>
    %276 = vector.extract %203#4[0, 3] : f32 from vector<2x8xf32>
    %277 = arith.select %206, %273, %274 : f32
    %shuffleResult_41, %valid_42 = gpu.shuffle xor %277, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %278 = arith.select %206, %shuffleResult_41, %273 : f32
    %279 = arith.select %206, %274, %shuffleResult_41 : f32
    %280 = arith.select %206, %275, %276 : f32
    %shuffleResult_43, %valid_44 = gpu.shuffle xor %280, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %281 = arith.select %206, %shuffleResult_43, %275 : f32
    %282 = arith.select %206, %276, %shuffleResult_43 : f32
    %283 = arith.select %208, %278, %281 : f32
    %shuffleResult_45, %valid_46 = gpu.shuffle xor %283, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %284 = arith.select %208, %shuffleResult_45, %278 : f32
    %285 = arith.select %208, %281, %shuffleResult_45 : f32
    %286 = arith.select %208, %279, %282 : f32
    %shuffleResult_47, %valid_48 = gpu.shuffle xor %286, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %287 = arith.select %208, %shuffleResult_47, %279 : f32
    %288 = arith.select %208, %282, %shuffleResult_47 : f32
    %289 = vector.extract %203#4[0, 4] : f32 from vector<2x8xf32>
    %290 = vector.extract %203#4[0, 5] : f32 from vector<2x8xf32>
    %291 = vector.extract %203#4[0, 6] : f32 from vector<2x8xf32>
    %292 = vector.extract %203#4[0, 7] : f32 from vector<2x8xf32>
    %293 = arith.select %206, %289, %290 : f32
    %shuffleResult_49, %valid_50 = gpu.shuffle xor %293, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %294 = arith.select %206, %shuffleResult_49, %289 : f32
    %295 = arith.select %206, %290, %shuffleResult_49 : f32
    %296 = arith.select %206, %291, %292 : f32
    %shuffleResult_51, %valid_52 = gpu.shuffle xor %296, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %297 = arith.select %206, %shuffleResult_51, %291 : f32
    %298 = arith.select %206, %292, %shuffleResult_51 : f32
    %299 = arith.select %208, %294, %297 : f32
    %shuffleResult_53, %valid_54 = gpu.shuffle xor %299, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %300 = arith.select %208, %shuffleResult_53, %294 : f32
    %301 = arith.select %208, %297, %shuffleResult_53 : f32
    %302 = arith.select %208, %295, %298 : f32
    %shuffleResult_55, %valid_56 = gpu.shuffle xor %302, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %303 = arith.select %208, %shuffleResult_55, %295 : f32
    %304 = arith.select %208, %298, %shuffleResult_55 : f32
    %305 = vector.extract %203#4[1, 0] : f32 from vector<2x8xf32>
    %306 = vector.extract %203#4[1, 1] : f32 from vector<2x8xf32>
    %307 = vector.extract %203#4[1, 2] : f32 from vector<2x8xf32>
    %308 = vector.extract %203#4[1, 3] : f32 from vector<2x8xf32>
    %309 = arith.select %206, %305, %306 : f32
    %shuffleResult_57, %valid_58 = gpu.shuffle xor %309, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %310 = arith.select %206, %shuffleResult_57, %305 : f32
    %311 = arith.select %206, %306, %shuffleResult_57 : f32
    %312 = arith.select %206, %307, %308 : f32
    %shuffleResult_59, %valid_60 = gpu.shuffle xor %312, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %313 = arith.select %206, %shuffleResult_59, %307 : f32
    %314 = arith.select %206, %308, %shuffleResult_59 : f32
    %315 = arith.select %208, %310, %313 : f32
    %shuffleResult_61, %valid_62 = gpu.shuffle xor %315, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %316 = arith.select %208, %shuffleResult_61, %310 : f32
    %317 = arith.select %208, %313, %shuffleResult_61 : f32
    %318 = arith.select %208, %311, %314 : f32
    %shuffleResult_63, %valid_64 = gpu.shuffle xor %318, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %319 = arith.select %208, %shuffleResult_63, %311 : f32
    %320 = arith.select %208, %314, %shuffleResult_63 : f32
    %321 = vector.extract %203#4[1, 4] : f32 from vector<2x8xf32>
    %322 = vector.extract %203#4[1, 5] : f32 from vector<2x8xf32>
    %323 = vector.extract %203#4[1, 6] : f32 from vector<2x8xf32>
    %324 = vector.extract %203#4[1, 7] : f32 from vector<2x8xf32>
    %325 = arith.select %206, %321, %322 : f32
    %shuffleResult_65, %valid_66 = gpu.shuffle xor %325, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %326 = arith.select %206, %shuffleResult_65, %321 : f32
    %327 = arith.select %206, %322, %shuffleResult_65 : f32
    %328 = arith.select %206, %323, %324 : f32
    %shuffleResult_67, %valid_68 = gpu.shuffle xor %328, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %329 = arith.select %206, %shuffleResult_67, %323 : f32
    %330 = arith.select %206, %324, %shuffleResult_67 : f32
    %331 = arith.select %208, %326, %329 : f32
    %shuffleResult_69, %valid_70 = gpu.shuffle xor %331, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %332 = arith.select %208, %shuffleResult_69, %326 : f32
    %333 = arith.select %208, %329, %shuffleResult_69 : f32
    %334 = arith.select %208, %327, %330 : f32
    %shuffleResult_71, %valid_72 = gpu.shuffle xor %334, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %335 = arith.select %208, %shuffleResult_71, %327 : f32
    %336 = arith.select %208, %330, %shuffleResult_71 : f32
    %337 = vector.extract %203#3[0, 0] : f32 from vector<2x8xf32>
    %338 = vector.extract %203#3[0, 1] : f32 from vector<2x8xf32>
    %339 = vector.extract %203#3[0, 2] : f32 from vector<2x8xf32>
    %340 = vector.extract %203#3[0, 3] : f32 from vector<2x8xf32>
    %341 = arith.select %206, %337, %338 : f32
    %shuffleResult_73, %valid_74 = gpu.shuffle xor %341, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %342 = arith.select %206, %shuffleResult_73, %337 : f32
    %343 = arith.select %206, %338, %shuffleResult_73 : f32
    %344 = arith.select %206, %339, %340 : f32
    %shuffleResult_75, %valid_76 = gpu.shuffle xor %344, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %345 = arith.select %206, %shuffleResult_75, %339 : f32
    %346 = arith.select %206, %340, %shuffleResult_75 : f32
    %347 = arith.select %208, %342, %345 : f32
    %shuffleResult_77, %valid_78 = gpu.shuffle xor %347, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %348 = arith.select %208, %shuffleResult_77, %342 : f32
    %349 = arith.select %208, %345, %shuffleResult_77 : f32
    %350 = arith.select %208, %343, %346 : f32
    %shuffleResult_79, %valid_80 = gpu.shuffle xor %350, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %351 = arith.select %208, %shuffleResult_79, %343 : f32
    %352 = arith.select %208, %346, %shuffleResult_79 : f32
    %353 = vector.extract %203#3[0, 4] : f32 from vector<2x8xf32>
    %354 = vector.extract %203#3[0, 5] : f32 from vector<2x8xf32>
    %355 = vector.extract %203#3[0, 6] : f32 from vector<2x8xf32>
    %356 = vector.extract %203#3[0, 7] : f32 from vector<2x8xf32>
    %357 = arith.select %206, %353, %354 : f32
    %shuffleResult_81, %valid_82 = gpu.shuffle xor %357, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %358 = arith.select %206, %shuffleResult_81, %353 : f32
    %359 = arith.select %206, %354, %shuffleResult_81 : f32
    %360 = arith.select %206, %355, %356 : f32
    %shuffleResult_83, %valid_84 = gpu.shuffle xor %360, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %361 = arith.select %206, %shuffleResult_83, %355 : f32
    %362 = arith.select %206, %356, %shuffleResult_83 : f32
    %363 = arith.select %208, %358, %361 : f32
    %shuffleResult_85, %valid_86 = gpu.shuffle xor %363, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %364 = arith.select %208, %shuffleResult_85, %358 : f32
    %365 = arith.select %208, %361, %shuffleResult_85 : f32
    %366 = arith.select %208, %359, %362 : f32
    %shuffleResult_87, %valid_88 = gpu.shuffle xor %366, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %367 = arith.select %208, %shuffleResult_87, %359 : f32
    %368 = arith.select %208, %362, %shuffleResult_87 : f32
    %369 = vector.extract %203#3[1, 0] : f32 from vector<2x8xf32>
    %370 = vector.extract %203#3[1, 1] : f32 from vector<2x8xf32>
    %371 = vector.extract %203#3[1, 2] : f32 from vector<2x8xf32>
    %372 = vector.extract %203#3[1, 3] : f32 from vector<2x8xf32>
    %373 = arith.select %206, %369, %370 : f32
    %shuffleResult_89, %valid_90 = gpu.shuffle xor %373, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %374 = arith.select %206, %shuffleResult_89, %369 : f32
    %375 = arith.select %206, %370, %shuffleResult_89 : f32
    %376 = arith.select %206, %371, %372 : f32
    %shuffleResult_91, %valid_92 = gpu.shuffle xor %376, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %377 = arith.select %206, %shuffleResult_91, %371 : f32
    %378 = arith.select %206, %372, %shuffleResult_91 : f32
    %379 = arith.select %208, %374, %377 : f32
    %shuffleResult_93, %valid_94 = gpu.shuffle xor %379, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %380 = arith.select %208, %shuffleResult_93, %374 : f32
    %381 = arith.select %208, %377, %shuffleResult_93 : f32
    %382 = arith.select %208, %375, %378 : f32
    %shuffleResult_95, %valid_96 = gpu.shuffle xor %382, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %383 = arith.select %208, %shuffleResult_95, %375 : f32
    %384 = arith.select %208, %378, %shuffleResult_95 : f32
    %385 = vector.extract %203#3[1, 4] : f32 from vector<2x8xf32>
    %386 = vector.extract %203#3[1, 5] : f32 from vector<2x8xf32>
    %387 = vector.extract %203#3[1, 6] : f32 from vector<2x8xf32>
    %388 = vector.extract %203#3[1, 7] : f32 from vector<2x8xf32>
    %389 = arith.select %206, %385, %386 : f32
    %shuffleResult_97, %valid_98 = gpu.shuffle xor %389, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %390 = arith.select %206, %shuffleResult_97, %385 : f32
    %391 = arith.select %206, %386, %shuffleResult_97 : f32
    %392 = arith.select %206, %387, %388 : f32
    %shuffleResult_99, %valid_100 = gpu.shuffle xor %392, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %393 = arith.select %206, %shuffleResult_99, %387 : f32
    %394 = arith.select %206, %388, %shuffleResult_99 : f32
    %395 = arith.select %208, %390, %393 : f32
    %shuffleResult_101, %valid_102 = gpu.shuffle xor %395, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %396 = arith.select %208, %shuffleResult_101, %390 : f32
    %397 = arith.select %208, %393, %shuffleResult_101 : f32
    %398 = arith.select %208, %391, %394 : f32
    %shuffleResult_103, %valid_104 = gpu.shuffle xor %398, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %399 = arith.select %208, %shuffleResult_103, %391 : f32
    %400 = arith.select %208, %394, %shuffleResult_103 : f32
    %401 = vector.extract %203#0[0, 0] : f32 from vector<2x8xf32>
    %402 = vector.extract %203#0[0, 1] : f32 from vector<2x8xf32>
    %403 = vector.extract %203#0[0, 2] : f32 from vector<2x8xf32>
    %404 = vector.extract %203#0[0, 3] : f32 from vector<2x8xf32>
    %405 = arith.select %206, %401, %402 : f32
    %shuffleResult_105, %valid_106 = gpu.shuffle xor %405, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %406 = arith.select %206, %shuffleResult_105, %401 : f32
    %407 = arith.select %206, %402, %shuffleResult_105 : f32
    %408 = arith.select %206, %403, %404 : f32
    %shuffleResult_107, %valid_108 = gpu.shuffle xor %408, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %409 = arith.select %206, %shuffleResult_107, %403 : f32
    %410 = arith.select %206, %404, %shuffleResult_107 : f32
    %411 = arith.select %208, %406, %409 : f32
    %shuffleResult_109, %valid_110 = gpu.shuffle xor %411, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %412 = arith.select %208, %shuffleResult_109, %406 : f32
    %413 = arith.select %208, %409, %shuffleResult_109 : f32
    %414 = arith.select %208, %407, %410 : f32
    %shuffleResult_111, %valid_112 = gpu.shuffle xor %414, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %415 = arith.select %208, %shuffleResult_111, %407 : f32
    %416 = arith.select %208, %410, %shuffleResult_111 : f32
    %417 = vector.extract %203#0[0, 4] : f32 from vector<2x8xf32>
    %418 = vector.extract %203#0[0, 5] : f32 from vector<2x8xf32>
    %419 = vector.extract %203#0[0, 6] : f32 from vector<2x8xf32>
    %420 = vector.extract %203#0[0, 7] : f32 from vector<2x8xf32>
    %421 = arith.select %206, %417, %418 : f32
    %shuffleResult_113, %valid_114 = gpu.shuffle xor %421, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %422 = arith.select %206, %shuffleResult_113, %417 : f32
    %423 = arith.select %206, %418, %shuffleResult_113 : f32
    %424 = arith.select %206, %419, %420 : f32
    %shuffleResult_115, %valid_116 = gpu.shuffle xor %424, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %425 = arith.select %206, %shuffleResult_115, %419 : f32
    %426 = arith.select %206, %420, %shuffleResult_115 : f32
    %427 = arith.select %208, %422, %425 : f32
    %shuffleResult_117, %valid_118 = gpu.shuffle xor %427, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %428 = arith.select %208, %shuffleResult_117, %422 : f32
    %429 = arith.select %208, %425, %shuffleResult_117 : f32
    %430 = arith.select %208, %423, %426 : f32
    %shuffleResult_119, %valid_120 = gpu.shuffle xor %430, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %431 = arith.select %208, %shuffleResult_119, %423 : f32
    %432 = arith.select %208, %426, %shuffleResult_119 : f32
    %433 = vector.extract %203#0[1, 0] : f32 from vector<2x8xf32>
    %434 = vector.extract %203#0[1, 1] : f32 from vector<2x8xf32>
    %435 = vector.extract %203#0[1, 2] : f32 from vector<2x8xf32>
    %436 = vector.extract %203#0[1, 3] : f32 from vector<2x8xf32>
    %437 = arith.select %206, %433, %434 : f32
    %shuffleResult_121, %valid_122 = gpu.shuffle xor %437, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %438 = arith.select %206, %shuffleResult_121, %433 : f32
    %439 = arith.select %206, %434, %shuffleResult_121 : f32
    %440 = arith.select %206, %435, %436 : f32
    %shuffleResult_123, %valid_124 = gpu.shuffle xor %440, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %441 = arith.select %206, %shuffleResult_123, %435 : f32
    %442 = arith.select %206, %436, %shuffleResult_123 : f32
    %443 = arith.select %208, %438, %441 : f32
    %shuffleResult_125, %valid_126 = gpu.shuffle xor %443, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %444 = arith.select %208, %shuffleResult_125, %438 : f32
    %445 = arith.select %208, %441, %shuffleResult_125 : f32
    %446 = arith.select %208, %439, %442 : f32
    %shuffleResult_127, %valid_128 = gpu.shuffle xor %446, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %447 = arith.select %208, %shuffleResult_127, %439 : f32
    %448 = arith.select %208, %442, %shuffleResult_127 : f32
    %449 = vector.extract %203#0[1, 4] : f32 from vector<2x8xf32>
    %450 = vector.extract %203#0[1, 5] : f32 from vector<2x8xf32>
    %451 = vector.extract %203#0[1, 6] : f32 from vector<2x8xf32>
    %452 = vector.extract %203#0[1, 7] : f32 from vector<2x8xf32>
    %453 = arith.select %206, %449, %450 : f32
    %shuffleResult_129, %valid_130 = gpu.shuffle xor %453, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %454 = arith.select %206, %shuffleResult_129, %449 : f32
    %455 = arith.select %206, %450, %shuffleResult_129 : f32
    %456 = arith.select %206, %451, %452 : f32
    %shuffleResult_131, %valid_132 = gpu.shuffle xor %456, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %457 = arith.select %206, %shuffleResult_131, %451 : f32
    %458 = arith.select %206, %452, %shuffleResult_131 : f32
    %459 = arith.select %208, %454, %457 : f32
    %shuffleResult_133, %valid_134 = gpu.shuffle xor %459, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %460 = arith.select %208, %shuffleResult_133, %454 : f32
    %461 = arith.select %208, %457, %shuffleResult_133 : f32
    %462 = arith.select %208, %455, %458 : f32
    %shuffleResult_135, %valid_136 = gpu.shuffle xor %462, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %463 = arith.select %208, %shuffleResult_135, %455 : f32
    %464 = arith.select %208, %458, %shuffleResult_135 : f32
    %465 = vector.extract %203#1[0, 0] : f32 from vector<2x1xf32>
    %466 = vector.from_elements %412, %415, %413, %416 : vector<4xf32>
    %467 = vector.broadcast %465 : f32 to vector<4xf32>
    %468 = arith.divf %cst, %467 fastmath<arcp> : vector<4xf32>
    %469 = arith.mulf %466, %468 : vector<4xf32>
    %470 = arith.truncf %469 {tiled = true} : vector<4xf32> to vector<4xf16>
    %471 = vector.extract %470[0] : f16 from vector<4xf16>
    %472 = vector.extract %470[1] : f16 from vector<4xf16>
    %473 = vector.extract %470[2] : f16 from vector<4xf16>
    %474 = vector.extract %470[3] : f16 from vector<4xf16>
    %475 = vector.from_elements %428, %431, %429, %432 : vector<4xf32>
    %476 = arith.mulf %475, %468 : vector<4xf32>
    %477 = arith.truncf %476 {tiled = true} : vector<4xf32> to vector<4xf16>
    %478 = vector.extract %477[0] : f16 from vector<4xf16>
    %479 = vector.extract %477[1] : f16 from vector<4xf16>
    %480 = vector.extract %477[2] : f16 from vector<4xf16>
    %481 = vector.extract %477[3] : f16 from vector<4xf16>
    %482 = vector.extract %203#1[1, 0] : f32 from vector<2x1xf32>
    %483 = vector.from_elements %444, %447, %445, %448 : vector<4xf32>
    %484 = vector.broadcast %482 : f32 to vector<4xf32>
    %485 = arith.divf %cst, %484 fastmath<arcp> : vector<4xf32>
    %486 = arith.mulf %483, %485 : vector<4xf32>
    %487 = arith.truncf %486 {tiled = true} : vector<4xf32> to vector<4xf16>
    %488 = vector.extract %487[0] : f16 from vector<4xf16>
    %489 = vector.extract %487[1] : f16 from vector<4xf16>
    %490 = vector.extract %487[2] : f16 from vector<4xf16>
    %491 = vector.extract %487[3] : f16 from vector<4xf16>
    %492 = vector.from_elements %460, %463, %461, %464 : vector<4xf32>
    %493 = arith.mulf %492, %485 : vector<4xf32>
    %494 = arith.truncf %493 {tiled = true} : vector<4xf32> to vector<4xf16>
    %495 = vector.extract %494[0] : f16 from vector<4xf16>
    %496 = vector.extract %494[1] : f16 from vector<4xf16>
    %497 = vector.extract %494[2] : f16 from vector<4xf16>
    %498 = vector.extract %494[3] : f16 from vector<4xf16>
    %499 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %500 = arith.cmpi ult, %499, %c4 : index
    scf.if %500 {
      affine.store %471, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
      affine.store %472, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
      affine.store %473, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
      affine.store %474, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      affine.store %478, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 16] : memref<1x32x4096x128xf16, 1>
      affine.store %479, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 17] : memref<1x32x4096x128xf16, 1>
      affine.store %480, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 18] : memref<1x32x4096x128xf16, 1>
      affine.store %481, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 19] : memref<1x32x4096x128xf16, 1>
      affine.store %488, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
      affine.store %489, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
      affine.store %490, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
      affine.store %491, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      affine.store %495, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 16] : memref<1x32x4096x128xf16, 1>
      affine.store %496, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 17] : memref<1x32x4096x128xf16, 1>
      affine.store %497, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 18] : memref<1x32x4096x128xf16, 1>
      affine.store %498, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 19] : memref<1x32x4096x128xf16, 1>
    }
    %501 = vector.from_elements %348, %351, %349, %352 : vector<4xf32>
    %502 = arith.mulf %501, %468 : vector<4xf32>
    %503 = arith.truncf %502 {tiled = true} : vector<4xf32> to vector<4xf16>
    %504 = vector.extract %503[0] : f16 from vector<4xf16>
    %505 = vector.extract %503[1] : f16 from vector<4xf16>
    %506 = vector.extract %503[2] : f16 from vector<4xf16>
    %507 = vector.extract %503[3] : f16 from vector<4xf16>
    %508 = vector.from_elements %364, %367, %365, %368 : vector<4xf32>
    %509 = arith.mulf %508, %468 : vector<4xf32>
    %510 = arith.truncf %509 {tiled = true} : vector<4xf32> to vector<4xf16>
    %511 = vector.extract %510[0] : f16 from vector<4xf16>
    %512 = vector.extract %510[1] : f16 from vector<4xf16>
    %513 = vector.extract %510[2] : f16 from vector<4xf16>
    %514 = vector.extract %510[3] : f16 from vector<4xf16>
    %515 = vector.from_elements %380, %383, %381, %384 : vector<4xf32>
    %516 = arith.mulf %515, %485 : vector<4xf32>
    %517 = arith.truncf %516 {tiled = true} : vector<4xf32> to vector<4xf16>
    %518 = vector.extract %517[0] : f16 from vector<4xf16>
    %519 = vector.extract %517[1] : f16 from vector<4xf16>
    %520 = vector.extract %517[2] : f16 from vector<4xf16>
    %521 = vector.extract %517[3] : f16 from vector<4xf16>
    %522 = vector.from_elements %396, %399, %397, %400 : vector<4xf32>
    %523 = arith.mulf %522, %485 : vector<4xf32>
    %524 = arith.truncf %523 {tiled = true} : vector<4xf32> to vector<4xf16>
    %525 = vector.extract %524[0] : f16 from vector<4xf16>
    %526 = vector.extract %524[1] : f16 from vector<4xf16>
    %527 = vector.extract %524[2] : f16 from vector<4xf16>
    %528 = vector.extract %524[3] : f16 from vector<4xf16>
    scf.if %500 {
      affine.store %504, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
      affine.store %505, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
      affine.store %506, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
      affine.store %507, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      affine.store %511, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 48] : memref<1x32x4096x128xf16, 1>
      affine.store %512, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 49] : memref<1x32x4096x128xf16, 1>
      affine.store %513, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 50] : memref<1x32x4096x128xf16, 1>
      affine.store %514, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 51] : memref<1x32x4096x128xf16, 1>
      affine.store %518, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
      affine.store %519, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
      affine.store %520, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
      affine.store %521, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      affine.store %525, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 48] : memref<1x32x4096x128xf16, 1>
      affine.store %526, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 49] : memref<1x32x4096x128xf16, 1>
      affine.store %527, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 50] : memref<1x32x4096x128xf16, 1>
      affine.store %528, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 51] : memref<1x32x4096x128xf16, 1>
    }
    %529 = vector.from_elements %284, %287, %285, %288 : vector<4xf32>
    %530 = arith.mulf %529, %468 : vector<4xf32>
    %531 = arith.truncf %530 {tiled = true} : vector<4xf32> to vector<4xf16>
    %532 = vector.extract %531[0] : f16 from vector<4xf16>
    %533 = vector.extract %531[1] : f16 from vector<4xf16>
    %534 = vector.extract %531[2] : f16 from vector<4xf16>
    %535 = vector.extract %531[3] : f16 from vector<4xf16>
    %536 = vector.from_elements %300, %303, %301, %304 : vector<4xf32>
    %537 = arith.mulf %536, %468 : vector<4xf32>
    %538 = arith.truncf %537 {tiled = true} : vector<4xf32> to vector<4xf16>
    %539 = vector.extract %538[0] : f16 from vector<4xf16>
    %540 = vector.extract %538[1] : f16 from vector<4xf16>
    %541 = vector.extract %538[2] : f16 from vector<4xf16>
    %542 = vector.extract %538[3] : f16 from vector<4xf16>
    %543 = vector.from_elements %316, %319, %317, %320 : vector<4xf32>
    %544 = arith.mulf %543, %485 : vector<4xf32>
    %545 = arith.truncf %544 {tiled = true} : vector<4xf32> to vector<4xf16>
    %546 = vector.extract %545[0] : f16 from vector<4xf16>
    %547 = vector.extract %545[1] : f16 from vector<4xf16>
    %548 = vector.extract %545[2] : f16 from vector<4xf16>
    %549 = vector.extract %545[3] : f16 from vector<4xf16>
    %550 = vector.from_elements %332, %335, %333, %336 : vector<4xf32>
    %551 = arith.mulf %550, %485 : vector<4xf32>
    %552 = arith.truncf %551 {tiled = true} : vector<4xf32> to vector<4xf16>
    %553 = vector.extract %552[0] : f16 from vector<4xf16>
    %554 = vector.extract %552[1] : f16 from vector<4xf16>
    %555 = vector.extract %552[2] : f16 from vector<4xf16>
    %556 = vector.extract %552[3] : f16 from vector<4xf16>
    scf.if %500 {
      affine.store %532, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
      affine.store %533, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
      affine.store %534, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
      affine.store %535, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      affine.store %539, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 80] : memref<1x32x4096x128xf16, 1>
      affine.store %540, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 81] : memref<1x32x4096x128xf16, 1>
      affine.store %541, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 82] : memref<1x32x4096x128xf16, 1>
      affine.store %542, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 83] : memref<1x32x4096x128xf16, 1>
      affine.store %546, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
      affine.store %547, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
      affine.store %548, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
      affine.store %549, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      affine.store %553, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 80] : memref<1x32x4096x128xf16, 1>
      affine.store %554, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 81] : memref<1x32x4096x128xf16, 1>
      affine.store %555, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 82] : memref<1x32x4096x128xf16, 1>
      affine.store %556, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 83] : memref<1x32x4096x128xf16, 1>
    }
    %557 = vector.from_elements %220, %223, %221, %224 : vector<4xf32>
    %558 = arith.mulf %557, %468 : vector<4xf32>
    %559 = arith.truncf %558 {tiled = true} : vector<4xf32> to vector<4xf16>
    %560 = vector.extract %559[0] : f16 from vector<4xf16>
    %561 = vector.extract %559[1] : f16 from vector<4xf16>
    %562 = vector.extract %559[2] : f16 from vector<4xf16>
    %563 = vector.extract %559[3] : f16 from vector<4xf16>
    %564 = vector.from_elements %236, %239, %237, %240 : vector<4xf32>
    %565 = arith.mulf %564, %468 : vector<4xf32>
    %566 = arith.truncf %565 {tiled = true} : vector<4xf32> to vector<4xf16>
    %567 = vector.extract %566[0] : f16 from vector<4xf16>
    %568 = vector.extract %566[1] : f16 from vector<4xf16>
    %569 = vector.extract %566[2] : f16 from vector<4xf16>
    %570 = vector.extract %566[3] : f16 from vector<4xf16>
    %571 = vector.from_elements %252, %255, %253, %256 : vector<4xf32>
    %572 = arith.mulf %571, %485 : vector<4xf32>
    %573 = arith.truncf %572 {tiled = true} : vector<4xf32> to vector<4xf16>
    %574 = vector.extract %573[0] : f16 from vector<4xf16>
    %575 = vector.extract %573[1] : f16 from vector<4xf16>
    %576 = vector.extract %573[2] : f16 from vector<4xf16>
    %577 = vector.extract %573[3] : f16 from vector<4xf16>
    %578 = vector.from_elements %268, %271, %269, %272 : vector<4xf32>
    %579 = arith.mulf %578, %485 : vector<4xf32>
    %580 = arith.truncf %579 {tiled = true} : vector<4xf32> to vector<4xf16>
    %581 = vector.extract %580[0] : f16 from vector<4xf16>
    %582 = vector.extract %580[1] : f16 from vector<4xf16>
    %583 = vector.extract %580[2] : f16 from vector<4xf16>
    %584 = vector.extract %580[3] : f16 from vector<4xf16>
    scf.if %500 {
      affine.store %560, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
      affine.store %561, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
      affine.store %562, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
      affine.store %563, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      affine.store %567, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 112] : memref<1x32x4096x128xf16, 1>
      affine.store %568, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 113] : memref<1x32x4096x128xf16, 1>
      affine.store %569, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 114] : memref<1x32x4096x128xf16, 1>
      affine.store %570, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 115] : memref<1x32x4096x128xf16, 1>
      affine.store %574, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
      affine.store %575, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
      affine.store %576, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
      affine.store %577, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      affine.store %581, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 112] : memref<1x32x4096x128xf16, 1>
      affine.store %582, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 113] : memref<1x32x4096x128xf16, 1>
      affine.store %583, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 114] : memref<1x32x4096x128xf16, 1>
      affine.store %584, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 115] : memref<1x32x4096x128xf16, 1>
    }
    return
  }
}
