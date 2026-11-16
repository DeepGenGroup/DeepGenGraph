module {
  memref.global "public" @shm_0 : memref<16384xi8, 3> = uninitialized alignment = 16
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 32, 0, 1, 0>, gridDim = [1 : i32, 32 : i32, 32 : i32], thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
    %cst_0 = arith.constant dense<0xFF800000> : vector<2x1xf32>
    %c8192 = arith.constant 8192 : index
    %cst_1 = arith.constant 0.127531052 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %cst_3 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %c4 = arith.constant 4 : index
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_5 = arith.constant 0.000000e+00 : f32
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
    %12 = arith.truncf %cst_1 : f32 to f16
    frisk.wait_buffer_ready %11 : !frisk.async_task
    %view_6 = memref.view %0[%c8192][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %13 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%7, %8]
    %subview_7 = memref.subview %arg0[0, %13, %10, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %14 = frisk.async_copy_gm_to_shm %subview_7 to %view_6 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
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
    %subview_8 = memref.subview %arg0[0, %13, %10, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %64 = frisk.async_copy_gm_to_shm %subview_8 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
    %65 = vector.load %view_6[%15] : memref<4096xf16, 3>, vector<4xf16>
    %66 = vector.load %view_6[%18] : memref<4096xf16, 3>, vector<4xf16>
    %67 = vector.load %view_6[%30] : memref<4096xf16, 3>, vector<4xf16>
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
    %78 = vector.load %view_6[%34] : memref<4096xf16, 3>, vector<4xf16>
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
    %subview_9 = memref.subview %arg0[0, %13, %10, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %109 = frisk.async_copy_gm_to_shm %subview_9 to %view_6 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((d0 floordiv 8) mod 128, (d0 floordiv 1024) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
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
    %154 = vector.load %view_6[%15] : memref<4096xf16, 3>, vector<4xf16>
    %155 = vector.load %view_6[%18] : memref<4096xf16, 3>, vector<4xf16>
    %156 = vector.load %view_6[%30] : memref<4096xf16, 3>, vector<4xf16>
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
    %167 = vector.load %view_6[%34] : memref<4096xf16, 3>, vector<4xf16>
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
    %203:6 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %cst_4, %arg6 = %cst_3, %arg7 = %cst_0, %arg8 = %cst_4, %arg9 = %cst_4, %arg10 = %cst_4) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %583 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %subview_136 = memref.subview %transpose[0, %198, 0, %583] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %584 = frisk.async_copy_gm_to_shm %subview_136 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      frisk.wait_buffer_ready %584 : !frisk.async_task
      %subview_137 = memref.subview %transpose[0, %198, 32, %583] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %585 = frisk.async_copy_gm_to_shm %subview_137 to %view_6 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %586 = affine.apply affine_map<()[s0] -> (s0 * 160 - (s0 floordiv 4) * 640 + ((s0 mod 16) floordiv 4) * 8 - ((s0 mod 4) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>()[%thread_id_x]
      %587 = vector.load %view[%586] : memref<4096xf16, 3>, vector<4xf16>
      %588 = vector.from_elements %36, %47, %48, %49 : vector<4xf16>
      %589 = vector.from_elements %50, %51, %52, %53 : vector<4xf16>
      %590 = vector.extract %587[0] : f16 from vector<4xf16>
      %591 = vector.extract %587[1] : f16 from vector<4xf16>
      %592 = vector.extract %587[2] : f16 from vector<4xf16>
      %593 = vector.extract %587[3] : f16 from vector<4xf16>
      %594 = vector.from_elements %590, %591, %592, %593 : vector<4xf16>
      %595 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %588, %594, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %596 = vector.from_elements %54, %55, %56, %57 : vector<4xf16>
      %597 = vector.from_elements %60, %61, %62, %63 : vector<4xf16>
      %598 = affine.apply affine_map<()[s0] -> (s0 * 160 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 - (s0 floordiv 2) * 64 + 2048)>()[%thread_id_x]
      %599 = vector.load %view[%598] : memref<4096xf16, 3>, vector<4xf16>
      %600 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 16) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 16) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 32)>()[%thread_id_x]
      %601 = vector.load %view[%600] : memref<4096xf16, 3>, vector<4xf16>
      %602 = vector.extract %599[0] : f16 from vector<4xf16>
      %603 = vector.extract %599[1] : f16 from vector<4xf16>
      %604 = vector.extract %599[2] : f16 from vector<4xf16>
      %605 = vector.extract %599[3] : f16 from vector<4xf16>
      %606 = vector.from_elements %602, %603, %604, %605 : vector<4xf16>
      %607 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %589, %606, %595 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %608 = vector.extract %601[0] : f16 from vector<4xf16>
      %609 = vector.extract %601[1] : f16 from vector<4xf16>
      %610 = vector.extract %601[2] : f16 from vector<4xf16>
      %611 = vector.extract %601[3] : f16 from vector<4xf16>
      %612 = vector.from_elements %608, %609, %610, %611 : vector<4xf16>
      %613 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %588, %612, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %614 = vector.extract %607[0] : f32 from vector<4xf32>
      %615 = vector.extract %607[1] : f32 from vector<4xf32>
      %616 = vector.extract %607[2] : f32 from vector<4xf32>
      %617 = vector.extract %607[3] : f32 from vector<4xf32>
      %618 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 16) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 16) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + 2080)>()[%thread_id_x]
      %619 = vector.load %view[%618] : memref<4096xf16, 3>, vector<4xf16>
      %620 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 32) floordiv 64) * 512 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 64)>()[%thread_id_x]
      %621 = vector.load %view[%620] : memref<4096xf16, 3>, vector<4xf16>
      %622 = vector.extract %619[0] : f16 from vector<4xf16>
      %623 = vector.extract %619[1] : f16 from vector<4xf16>
      %624 = vector.extract %619[2] : f16 from vector<4xf16>
      %625 = vector.extract %619[3] : f16 from vector<4xf16>
      %626 = vector.from_elements %622, %623, %624, %625 : vector<4xf16>
      %627 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %589, %626, %613 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %628 = vector.extract %621[0] : f16 from vector<4xf16>
      %629 = vector.extract %621[1] : f16 from vector<4xf16>
      %630 = vector.extract %621[2] : f16 from vector<4xf16>
      %631 = vector.extract %621[3] : f16 from vector<4xf16>
      %632 = vector.from_elements %628, %629, %630, %631 : vector<4xf16>
      %633 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %588, %632, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %634 = vector.extract %627[0] : f32 from vector<4xf32>
      %635 = vector.extract %627[1] : f32 from vector<4xf32>
      %636 = vector.extract %627[2] : f32 from vector<4xf32>
      %637 = vector.extract %627[3] : f32 from vector<4xf32>
      %638 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 32) floordiv 64) * 512 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + 2112)>()[%thread_id_x]
      %639 = vector.load %view[%638] : memref<4096xf16, 3>, vector<4xf16>
      %640 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 48) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 48) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 160)>()[%thread_id_x]
      %641 = vector.load %view[%640] : memref<4096xf16, 3>, vector<4xf16>
      %642 = vector.extract %639[0] : f16 from vector<4xf16>
      %643 = vector.extract %639[1] : f16 from vector<4xf16>
      %644 = vector.extract %639[2] : f16 from vector<4xf16>
      %645 = vector.extract %639[3] : f16 from vector<4xf16>
      %646 = vector.from_elements %642, %643, %644, %645 : vector<4xf16>
      %647 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %589, %646, %633 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %648 = vector.extract %641[0] : f16 from vector<4xf16>
      %649 = vector.extract %641[1] : f16 from vector<4xf16>
      %650 = vector.extract %641[2] : f16 from vector<4xf16>
      %651 = vector.extract %641[3] : f16 from vector<4xf16>
      %652 = vector.from_elements %648, %649, %650, %651 : vector<4xf16>
      %653 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %588, %652, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %654 = vector.extract %647[0] : f32 from vector<4xf32>
      %655 = vector.extract %647[1] : f32 from vector<4xf32>
      %656 = vector.extract %647[2] : f32 from vector<4xf32>
      %657 = vector.extract %647[3] : f32 from vector<4xf32>
      %658 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 48) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 48) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + 2208)>()[%thread_id_x]
      %659 = vector.load %view[%658] : memref<4096xf16, 3>, vector<4xf16>
      %660 = affine.apply affine_map<()[s0] -> (s0 * 160 - (s0 floordiv 4) * 640 + ((s0 mod 16) floordiv 4) * 8 - ((s0 mod 4) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 512)>()[%thread_id_x]
      %661 = vector.load %view[%660] : memref<4096xf16, 3>, vector<4xf16>
      %662 = vector.extract %659[0] : f16 from vector<4xf16>
      %663 = vector.extract %659[1] : f16 from vector<4xf16>
      %664 = vector.extract %659[2] : f16 from vector<4xf16>
      %665 = vector.extract %659[3] : f16 from vector<4xf16>
      %666 = vector.from_elements %662, %663, %664, %665 : vector<4xf16>
      %667 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %589, %666, %653 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %668 = vector.extract %661[0] : f16 from vector<4xf16>
      %669 = vector.extract %661[1] : f16 from vector<4xf16>
      %670 = vector.extract %661[2] : f16 from vector<4xf16>
      %671 = vector.extract %661[3] : f16 from vector<4xf16>
      %672 = vector.from_elements %668, %669, %670, %671 : vector<4xf16>
      %673 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %588, %672, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %674 = vector.extract %667[0] : f32 from vector<4xf32>
      %675 = vector.extract %667[1] : f32 from vector<4xf32>
      %676 = vector.extract %667[2] : f32 from vector<4xf32>
      %677 = vector.extract %667[3] : f32 from vector<4xf32>
      %678 = affine.apply affine_map<()[s0] -> (s0 * 160 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 - (s0 floordiv 2) * 64 + 2560)>()[%thread_id_x]
      %679 = vector.load %view[%678] : memref<4096xf16, 3>, vector<4xf16>
      %680 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 80) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 80) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 32)>()[%thread_id_x]
      %681 = vector.load %view[%680] : memref<4096xf16, 3>, vector<4xf16>
      %682 = vector.extract %679[0] : f16 from vector<4xf16>
      %683 = vector.extract %679[1] : f16 from vector<4xf16>
      %684 = vector.extract %679[2] : f16 from vector<4xf16>
      %685 = vector.extract %679[3] : f16 from vector<4xf16>
      %686 = vector.from_elements %682, %683, %684, %685 : vector<4xf16>
      %687 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %589, %686, %673 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %688 = vector.extract %681[0] : f16 from vector<4xf16>
      %689 = vector.extract %681[1] : f16 from vector<4xf16>
      %690 = vector.extract %681[2] : f16 from vector<4xf16>
      %691 = vector.extract %681[3] : f16 from vector<4xf16>
      %692 = vector.from_elements %688, %689, %690, %691 : vector<4xf16>
      %693 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %588, %692, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %694 = vector.extract %687[0] : f32 from vector<4xf32>
      %695 = vector.extract %687[1] : f32 from vector<4xf32>
      %696 = vector.extract %687[2] : f32 from vector<4xf32>
      %697 = vector.extract %687[3] : f32 from vector<4xf32>
      %698 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 80) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 80) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + 2080)>()[%thread_id_x]
      %699 = vector.load %view[%698] : memref<4096xf16, 3>, vector<4xf16>
      %700 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 96) floordiv 64) * 512 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 64)>()[%thread_id_x]
      %701 = vector.load %view[%700] : memref<4096xf16, 3>, vector<4xf16>
      %702 = vector.extract %699[0] : f16 from vector<4xf16>
      %703 = vector.extract %699[1] : f16 from vector<4xf16>
      %704 = vector.extract %699[2] : f16 from vector<4xf16>
      %705 = vector.extract %699[3] : f16 from vector<4xf16>
      %706 = vector.from_elements %702, %703, %704, %705 : vector<4xf16>
      %707 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %589, %706, %693 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %708 = vector.extract %701[0] : f16 from vector<4xf16>
      %709 = vector.extract %701[1] : f16 from vector<4xf16>
      %710 = vector.extract %701[2] : f16 from vector<4xf16>
      %711 = vector.extract %701[3] : f16 from vector<4xf16>
      %712 = vector.from_elements %708, %709, %710, %711 : vector<4xf16>
      %713 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %588, %712, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %714 = vector.extract %707[0] : f32 from vector<4xf32>
      %715 = vector.extract %707[1] : f32 from vector<4xf32>
      %716 = vector.extract %707[2] : f32 from vector<4xf32>
      %717 = vector.extract %707[3] : f32 from vector<4xf32>
      %718 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 96) floordiv 64) * 512 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + 2112)>()[%thread_id_x]
      %719 = vector.load %view[%718] : memref<4096xf16, 3>, vector<4xf16>
      %720 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 112) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 112) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 160)>()[%thread_id_x]
      %721 = vector.load %view[%720] : memref<4096xf16, 3>, vector<4xf16>
      %722 = vector.extract %719[0] : f16 from vector<4xf16>
      %723 = vector.extract %719[1] : f16 from vector<4xf16>
      %724 = vector.extract %719[2] : f16 from vector<4xf16>
      %725 = vector.extract %719[3] : f16 from vector<4xf16>
      %726 = vector.from_elements %722, %723, %724, %725 : vector<4xf16>
      %727 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %589, %726, %713 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %728 = vector.extract %721[0] : f16 from vector<4xf16>
      %729 = vector.extract %721[1] : f16 from vector<4xf16>
      %730 = vector.extract %721[2] : f16 from vector<4xf16>
      %731 = vector.extract %721[3] : f16 from vector<4xf16>
      %732 = vector.from_elements %728, %729, %730, %731 : vector<4xf16>
      %733 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %588, %732, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %734 = vector.extract %727[0] : f32 from vector<4xf32>
      %735 = vector.extract %727[1] : f32 from vector<4xf32>
      %736 = vector.extract %727[2] : f32 from vector<4xf32>
      %737 = vector.extract %727[3] : f32 from vector<4xf32>
      %738 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 112) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 112) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + 2208)>()[%thread_id_x]
      %739 = vector.load %view[%738] : memref<4096xf16, 3>, vector<4xf16>
      %740 = vector.load %view[%586] : memref<4096xf16, 3>, vector<4xf16>
      %741 = vector.extract %739[0] : f16 from vector<4xf16>
      %742 = vector.extract %739[1] : f16 from vector<4xf16>
      %743 = vector.extract %739[2] : f16 from vector<4xf16>
      %744 = vector.extract %739[3] : f16 from vector<4xf16>
      %745 = vector.from_elements %741, %742, %743, %744 : vector<4xf16>
      %746 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %589, %745, %733 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %747 = vector.extract %740[0] : f16 from vector<4xf16>
      %748 = vector.extract %740[1] : f16 from vector<4xf16>
      %749 = vector.extract %740[2] : f16 from vector<4xf16>
      %750 = vector.extract %740[3] : f16 from vector<4xf16>
      %751 = vector.from_elements %747, %748, %749, %750 : vector<4xf16>
      %752 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %596, %751, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %753 = vector.extract %746[0] : f32 from vector<4xf32>
      %754 = vector.extract %746[1] : f32 from vector<4xf32>
      %755 = vector.extract %746[2] : f32 from vector<4xf32>
      %756 = vector.extract %746[3] : f32 from vector<4xf32>
      %757 = vector.load %view[%598] : memref<4096xf16, 3>, vector<4xf16>
      %758 = vector.load %view[%600] : memref<4096xf16, 3>, vector<4xf16>
      %759 = vector.extract %757[0] : f16 from vector<4xf16>
      %760 = vector.extract %757[1] : f16 from vector<4xf16>
      %761 = vector.extract %757[2] : f16 from vector<4xf16>
      %762 = vector.extract %757[3] : f16 from vector<4xf16>
      %763 = vector.from_elements %759, %760, %761, %762 : vector<4xf16>
      %764 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %597, %763, %752 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %765 = vector.extract %758[0] : f16 from vector<4xf16>
      %766 = vector.extract %758[1] : f16 from vector<4xf16>
      %767 = vector.extract %758[2] : f16 from vector<4xf16>
      %768 = vector.extract %758[3] : f16 from vector<4xf16>
      %769 = vector.from_elements %765, %766, %767, %768 : vector<4xf16>
      %770 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %596, %769, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %771 = vector.extract %764[0] : f32 from vector<4xf32>
      %772 = vector.extract %764[1] : f32 from vector<4xf32>
      %773 = vector.extract %764[2] : f32 from vector<4xf32>
      %774 = vector.extract %764[3] : f32 from vector<4xf32>
      %775 = vector.load %view[%618] : memref<4096xf16, 3>, vector<4xf16>
      %776 = vector.load %view[%620] : memref<4096xf16, 3>, vector<4xf16>
      %777 = vector.extract %775[0] : f16 from vector<4xf16>
      %778 = vector.extract %775[1] : f16 from vector<4xf16>
      %779 = vector.extract %775[2] : f16 from vector<4xf16>
      %780 = vector.extract %775[3] : f16 from vector<4xf16>
      %781 = vector.from_elements %777, %778, %779, %780 : vector<4xf16>
      %782 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %597, %781, %770 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %783 = vector.extract %776[0] : f16 from vector<4xf16>
      %784 = vector.extract %776[1] : f16 from vector<4xf16>
      %785 = vector.extract %776[2] : f16 from vector<4xf16>
      %786 = vector.extract %776[3] : f16 from vector<4xf16>
      %787 = vector.from_elements %783, %784, %785, %786 : vector<4xf16>
      %788 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %596, %787, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %789 = vector.extract %782[0] : f32 from vector<4xf32>
      %790 = vector.extract %782[1] : f32 from vector<4xf32>
      %791 = vector.extract %782[2] : f32 from vector<4xf32>
      %792 = vector.extract %782[3] : f32 from vector<4xf32>
      %793 = vector.load %view[%638] : memref<4096xf16, 3>, vector<4xf16>
      %794 = vector.load %view[%640] : memref<4096xf16, 3>, vector<4xf16>
      %795 = vector.extract %793[0] : f16 from vector<4xf16>
      %796 = vector.extract %793[1] : f16 from vector<4xf16>
      %797 = vector.extract %793[2] : f16 from vector<4xf16>
      %798 = vector.extract %793[3] : f16 from vector<4xf16>
      %799 = vector.from_elements %795, %796, %797, %798 : vector<4xf16>
      %800 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %597, %799, %788 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %801 = vector.extract %794[0] : f16 from vector<4xf16>
      %802 = vector.extract %794[1] : f16 from vector<4xf16>
      %803 = vector.extract %794[2] : f16 from vector<4xf16>
      %804 = vector.extract %794[3] : f16 from vector<4xf16>
      %805 = vector.from_elements %801, %802, %803, %804 : vector<4xf16>
      %806 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %596, %805, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %807 = vector.extract %800[0] : f32 from vector<4xf32>
      %808 = vector.extract %800[1] : f32 from vector<4xf32>
      %809 = vector.extract %800[2] : f32 from vector<4xf32>
      %810 = vector.extract %800[3] : f32 from vector<4xf32>
      %811 = vector.load %view[%658] : memref<4096xf16, 3>, vector<4xf16>
      %812 = vector.load %view[%660] : memref<4096xf16, 3>, vector<4xf16>
      %813 = vector.extract %811[0] : f16 from vector<4xf16>
      %814 = vector.extract %811[1] : f16 from vector<4xf16>
      %815 = vector.extract %811[2] : f16 from vector<4xf16>
      %816 = vector.extract %811[3] : f16 from vector<4xf16>
      %817 = vector.from_elements %813, %814, %815, %816 : vector<4xf16>
      %818 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %597, %817, %806 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %819 = vector.extract %812[0] : f16 from vector<4xf16>
      %820 = vector.extract %812[1] : f16 from vector<4xf16>
      %821 = vector.extract %812[2] : f16 from vector<4xf16>
      %822 = vector.extract %812[3] : f16 from vector<4xf16>
      %823 = vector.from_elements %819, %820, %821, %822 : vector<4xf16>
      %824 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %596, %823, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %825 = vector.extract %818[0] : f32 from vector<4xf32>
      %826 = vector.extract %818[1] : f32 from vector<4xf32>
      %827 = vector.extract %818[2] : f32 from vector<4xf32>
      %828 = vector.extract %818[3] : f32 from vector<4xf32>
      %829 = vector.load %view[%678] : memref<4096xf16, 3>, vector<4xf16>
      %830 = vector.load %view[%680] : memref<4096xf16, 3>, vector<4xf16>
      %831 = vector.extract %829[0] : f16 from vector<4xf16>
      %832 = vector.extract %829[1] : f16 from vector<4xf16>
      %833 = vector.extract %829[2] : f16 from vector<4xf16>
      %834 = vector.extract %829[3] : f16 from vector<4xf16>
      %835 = vector.from_elements %831, %832, %833, %834 : vector<4xf16>
      %836 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %597, %835, %824 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %837 = vector.extract %830[0] : f16 from vector<4xf16>
      %838 = vector.extract %830[1] : f16 from vector<4xf16>
      %839 = vector.extract %830[2] : f16 from vector<4xf16>
      %840 = vector.extract %830[3] : f16 from vector<4xf16>
      %841 = vector.from_elements %837, %838, %839, %840 : vector<4xf16>
      %842 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %596, %841, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %843 = vector.extract %836[0] : f32 from vector<4xf32>
      %844 = vector.extract %836[1] : f32 from vector<4xf32>
      %845 = vector.extract %836[2] : f32 from vector<4xf32>
      %846 = vector.extract %836[3] : f32 from vector<4xf32>
      %847 = vector.load %view[%698] : memref<4096xf16, 3>, vector<4xf16>
      %848 = vector.load %view[%700] : memref<4096xf16, 3>, vector<4xf16>
      %849 = vector.extract %847[0] : f16 from vector<4xf16>
      %850 = vector.extract %847[1] : f16 from vector<4xf16>
      %851 = vector.extract %847[2] : f16 from vector<4xf16>
      %852 = vector.extract %847[3] : f16 from vector<4xf16>
      %853 = vector.from_elements %849, %850, %851, %852 : vector<4xf16>
      %854 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %597, %853, %842 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %855 = vector.extract %848[0] : f16 from vector<4xf16>
      %856 = vector.extract %848[1] : f16 from vector<4xf16>
      %857 = vector.extract %848[2] : f16 from vector<4xf16>
      %858 = vector.extract %848[3] : f16 from vector<4xf16>
      %859 = vector.from_elements %855, %856, %857, %858 : vector<4xf16>
      %860 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %596, %859, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %861 = vector.extract %854[0] : f32 from vector<4xf32>
      %862 = vector.extract %854[1] : f32 from vector<4xf32>
      %863 = vector.extract %854[2] : f32 from vector<4xf32>
      %864 = vector.extract %854[3] : f32 from vector<4xf32>
      %865 = vector.load %view[%718] : memref<4096xf16, 3>, vector<4xf16>
      %866 = vector.load %view[%720] : memref<4096xf16, 3>, vector<4xf16>
      %867 = vector.extract %865[0] : f16 from vector<4xf16>
      %868 = vector.extract %865[1] : f16 from vector<4xf16>
      %869 = vector.extract %865[2] : f16 from vector<4xf16>
      %870 = vector.extract %865[3] : f16 from vector<4xf16>
      %871 = vector.from_elements %867, %868, %869, %870 : vector<4xf16>
      %872 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %597, %871, %860 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %873 = vector.extract %866[0] : f16 from vector<4xf16>
      %874 = vector.extract %866[1] : f16 from vector<4xf16>
      %875 = vector.extract %866[2] : f16 from vector<4xf16>
      %876 = vector.extract %866[3] : f16 from vector<4xf16>
      %877 = vector.from_elements %873, %874, %875, %876 : vector<4xf16>
      %878 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %596, %877, %cst : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %879 = vector.extract %872[0] : f32 from vector<4xf32>
      %880 = vector.extract %872[1] : f32 from vector<4xf32>
      %881 = vector.extract %872[2] : f32 from vector<4xf32>
      %882 = vector.extract %872[3] : f32 from vector<4xf32>
      %883 = vector.load %view[%738] : memref<4096xf16, 3>, vector<4xf16>
      %884 = vector.extract %883[0] : f16 from vector<4xf16>
      %885 = vector.extract %883[1] : f16 from vector<4xf16>
      %886 = vector.extract %883[2] : f16 from vector<4xf16>
      %887 = vector.extract %883[3] : f16 from vector<4xf16>
      %888 = vector.from_elements %884, %885, %886, %887 : vector<4xf16>
      %889 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %597, %888, %878 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %890 = vector.extract %889[0] : f32 from vector<4xf32>
      %891 = vector.extract %889[1] : f32 from vector<4xf32>
      %892 = vector.extract %889[2] : f32 from vector<4xf32>
      %893 = vector.extract %889[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %585 : !frisk.async_task
      %subview_138 = memref.subview %transpose[0, %198, 64, %583] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %894 = frisk.async_copy_gm_to_shm %subview_138 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %895 = vector.load %view_6[%586] : memref<4096xf16, 3>, vector<4xf16>
      %896 = vector.from_elements %85, %86, %87, %88 : vector<4xf16>
      %897 = vector.from_elements %614, %615, %616, %617 : vector<4xf32>
      %898 = vector.from_elements %89, %90, %91, %92 : vector<4xf16>
      %899 = vector.from_elements %634, %635, %636, %637 : vector<4xf32>
      %900 = vector.extract %895[0] : f16 from vector<4xf16>
      %901 = vector.extract %895[1] : f16 from vector<4xf16>
      %902 = vector.extract %895[2] : f16 from vector<4xf16>
      %903 = vector.extract %895[3] : f16 from vector<4xf16>
      %904 = vector.from_elements %900, %901, %902, %903 : vector<4xf16>
      %905 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %904, %897 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %906 = vector.from_elements %654, %655, %656, %657 : vector<4xf32>
      %907 = vector.from_elements %674, %675, %676, %677 : vector<4xf32>
      %908 = vector.from_elements %694, %695, %696, %697 : vector<4xf32>
      %909 = vector.from_elements %714, %715, %716, %717 : vector<4xf32>
      %910 = vector.from_elements %734, %735, %736, %737 : vector<4xf32>
      %911 = vector.from_elements %753, %754, %755, %756 : vector<4xf32>
      %912 = vector.from_elements %101, %102, %103, %104 : vector<4xf16>
      %913 = vector.from_elements %771, %772, %773, %774 : vector<4xf32>
      %914 = vector.from_elements %105, %106, %107, %108 : vector<4xf16>
      %915 = vector.from_elements %789, %790, %791, %792 : vector<4xf32>
      %916 = vector.from_elements %807, %808, %809, %810 : vector<4xf32>
      %917 = vector.from_elements %825, %826, %827, %828 : vector<4xf32>
      %918 = vector.from_elements %843, %844, %845, %846 : vector<4xf32>
      %919 = vector.from_elements %861, %862, %863, %864 : vector<4xf32>
      %920 = vector.from_elements %879, %880, %881, %882 : vector<4xf32>
      %921 = vector.from_elements %890, %891, %892, %893 : vector<4xf32>
      %922 = vector.load %view_6[%598] : memref<4096xf16, 3>, vector<4xf16>
      %923 = vector.load %view_6[%600] : memref<4096xf16, 3>, vector<4xf16>
      %924 = vector.extract %922[0] : f16 from vector<4xf16>
      %925 = vector.extract %922[1] : f16 from vector<4xf16>
      %926 = vector.extract %922[2] : f16 from vector<4xf16>
      %927 = vector.extract %922[3] : f16 from vector<4xf16>
      %928 = vector.from_elements %924, %925, %926, %927 : vector<4xf16>
      %929 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %928, %905 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %930 = vector.extract %923[0] : f16 from vector<4xf16>
      %931 = vector.extract %923[1] : f16 from vector<4xf16>
      %932 = vector.extract %923[2] : f16 from vector<4xf16>
      %933 = vector.extract %923[3] : f16 from vector<4xf16>
      %934 = vector.from_elements %930, %931, %932, %933 : vector<4xf16>
      %935 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %934, %899 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %936 = vector.extract %929[0] : f32 from vector<4xf32>
      %937 = vector.extract %929[1] : f32 from vector<4xf32>
      %938 = vector.extract %929[2] : f32 from vector<4xf32>
      %939 = vector.extract %929[3] : f32 from vector<4xf32>
      %940 = vector.load %view_6[%618] : memref<4096xf16, 3>, vector<4xf16>
      %941 = vector.load %view_6[%620] : memref<4096xf16, 3>, vector<4xf16>
      %942 = vector.extract %940[0] : f16 from vector<4xf16>
      %943 = vector.extract %940[1] : f16 from vector<4xf16>
      %944 = vector.extract %940[2] : f16 from vector<4xf16>
      %945 = vector.extract %940[3] : f16 from vector<4xf16>
      %946 = vector.from_elements %942, %943, %944, %945 : vector<4xf16>
      %947 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %946, %935 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %948 = vector.extract %941[0] : f16 from vector<4xf16>
      %949 = vector.extract %941[1] : f16 from vector<4xf16>
      %950 = vector.extract %941[2] : f16 from vector<4xf16>
      %951 = vector.extract %941[3] : f16 from vector<4xf16>
      %952 = vector.from_elements %948, %949, %950, %951 : vector<4xf16>
      %953 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %952, %906 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %954 = vector.extract %947[0] : f32 from vector<4xf32>
      %955 = vector.extract %947[1] : f32 from vector<4xf32>
      %956 = vector.extract %947[2] : f32 from vector<4xf32>
      %957 = vector.extract %947[3] : f32 from vector<4xf32>
      %958 = vector.load %view_6[%638] : memref<4096xf16, 3>, vector<4xf16>
      %959 = vector.load %view_6[%640] : memref<4096xf16, 3>, vector<4xf16>
      %960 = vector.extract %958[0] : f16 from vector<4xf16>
      %961 = vector.extract %958[1] : f16 from vector<4xf16>
      %962 = vector.extract %958[2] : f16 from vector<4xf16>
      %963 = vector.extract %958[3] : f16 from vector<4xf16>
      %964 = vector.from_elements %960, %961, %962, %963 : vector<4xf16>
      %965 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %964, %953 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %966 = vector.extract %959[0] : f16 from vector<4xf16>
      %967 = vector.extract %959[1] : f16 from vector<4xf16>
      %968 = vector.extract %959[2] : f16 from vector<4xf16>
      %969 = vector.extract %959[3] : f16 from vector<4xf16>
      %970 = vector.from_elements %966, %967, %968, %969 : vector<4xf16>
      %971 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %970, %907 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %972 = vector.extract %965[0] : f32 from vector<4xf32>
      %973 = vector.extract %965[1] : f32 from vector<4xf32>
      %974 = vector.extract %965[2] : f32 from vector<4xf32>
      %975 = vector.extract %965[3] : f32 from vector<4xf32>
      %976 = vector.load %view_6[%658] : memref<4096xf16, 3>, vector<4xf16>
      %977 = vector.load %view_6[%660] : memref<4096xf16, 3>, vector<4xf16>
      %978 = vector.extract %976[0] : f16 from vector<4xf16>
      %979 = vector.extract %976[1] : f16 from vector<4xf16>
      %980 = vector.extract %976[2] : f16 from vector<4xf16>
      %981 = vector.extract %976[3] : f16 from vector<4xf16>
      %982 = vector.from_elements %978, %979, %980, %981 : vector<4xf16>
      %983 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %982, %971 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %984 = vector.extract %977[0] : f16 from vector<4xf16>
      %985 = vector.extract %977[1] : f16 from vector<4xf16>
      %986 = vector.extract %977[2] : f16 from vector<4xf16>
      %987 = vector.extract %977[3] : f16 from vector<4xf16>
      %988 = vector.from_elements %984, %985, %986, %987 : vector<4xf16>
      %989 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %988, %908 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %990 = vector.extract %983[0] : f32 from vector<4xf32>
      %991 = vector.extract %983[1] : f32 from vector<4xf32>
      %992 = vector.extract %983[2] : f32 from vector<4xf32>
      %993 = vector.extract %983[3] : f32 from vector<4xf32>
      %994 = vector.load %view_6[%678] : memref<4096xf16, 3>, vector<4xf16>
      %995 = vector.load %view_6[%680] : memref<4096xf16, 3>, vector<4xf16>
      %996 = vector.extract %994[0] : f16 from vector<4xf16>
      %997 = vector.extract %994[1] : f16 from vector<4xf16>
      %998 = vector.extract %994[2] : f16 from vector<4xf16>
      %999 = vector.extract %994[3] : f16 from vector<4xf16>
      %1000 = vector.from_elements %996, %997, %998, %999 : vector<4xf16>
      %1001 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %1000, %989 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1002 = vector.extract %995[0] : f16 from vector<4xf16>
      %1003 = vector.extract %995[1] : f16 from vector<4xf16>
      %1004 = vector.extract %995[2] : f16 from vector<4xf16>
      %1005 = vector.extract %995[3] : f16 from vector<4xf16>
      %1006 = vector.from_elements %1002, %1003, %1004, %1005 : vector<4xf16>
      %1007 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1006, %909 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1008 = vector.extract %1001[0] : f32 from vector<4xf32>
      %1009 = vector.extract %1001[1] : f32 from vector<4xf32>
      %1010 = vector.extract %1001[2] : f32 from vector<4xf32>
      %1011 = vector.extract %1001[3] : f32 from vector<4xf32>
      %1012 = vector.load %view_6[%698] : memref<4096xf16, 3>, vector<4xf16>
      %1013 = vector.load %view_6[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1014 = vector.extract %1012[0] : f16 from vector<4xf16>
      %1015 = vector.extract %1012[1] : f16 from vector<4xf16>
      %1016 = vector.extract %1012[2] : f16 from vector<4xf16>
      %1017 = vector.extract %1012[3] : f16 from vector<4xf16>
      %1018 = vector.from_elements %1014, %1015, %1016, %1017 : vector<4xf16>
      %1019 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %1018, %1007 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1020 = vector.extract %1013[0] : f16 from vector<4xf16>
      %1021 = vector.extract %1013[1] : f16 from vector<4xf16>
      %1022 = vector.extract %1013[2] : f16 from vector<4xf16>
      %1023 = vector.extract %1013[3] : f16 from vector<4xf16>
      %1024 = vector.from_elements %1020, %1021, %1022, %1023 : vector<4xf16>
      %1025 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1024, %910 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1026 = vector.extract %1019[0] : f32 from vector<4xf32>
      %1027 = vector.extract %1019[1] : f32 from vector<4xf32>
      %1028 = vector.extract %1019[2] : f32 from vector<4xf32>
      %1029 = vector.extract %1019[3] : f32 from vector<4xf32>
      %1030 = vector.load %view_6[%718] : memref<4096xf16, 3>, vector<4xf16>
      %1031 = vector.load %view_6[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1032 = vector.extract %1030[0] : f16 from vector<4xf16>
      %1033 = vector.extract %1030[1] : f16 from vector<4xf16>
      %1034 = vector.extract %1030[2] : f16 from vector<4xf16>
      %1035 = vector.extract %1030[3] : f16 from vector<4xf16>
      %1036 = vector.from_elements %1032, %1033, %1034, %1035 : vector<4xf16>
      %1037 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %1036, %1025 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1038 = vector.extract %1031[0] : f16 from vector<4xf16>
      %1039 = vector.extract %1031[1] : f16 from vector<4xf16>
      %1040 = vector.extract %1031[2] : f16 from vector<4xf16>
      %1041 = vector.extract %1031[3] : f16 from vector<4xf16>
      %1042 = vector.from_elements %1038, %1039, %1040, %1041 : vector<4xf16>
      %1043 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1042, %911 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1044 = vector.extract %1037[0] : f32 from vector<4xf32>
      %1045 = vector.extract %1037[1] : f32 from vector<4xf32>
      %1046 = vector.extract %1037[2] : f32 from vector<4xf32>
      %1047 = vector.extract %1037[3] : f32 from vector<4xf32>
      %1048 = vector.load %view_6[%738] : memref<4096xf16, 3>, vector<4xf16>
      %1049 = vector.load %view_6[%586] : memref<4096xf16, 3>, vector<4xf16>
      %1050 = vector.extract %1048[0] : f16 from vector<4xf16>
      %1051 = vector.extract %1048[1] : f16 from vector<4xf16>
      %1052 = vector.extract %1048[2] : f16 from vector<4xf16>
      %1053 = vector.extract %1048[3] : f16 from vector<4xf16>
      %1054 = vector.from_elements %1050, %1051, %1052, %1053 : vector<4xf16>
      %1055 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %898, %1054, %1043 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1056 = vector.extract %1049[0] : f16 from vector<4xf16>
      %1057 = vector.extract %1049[1] : f16 from vector<4xf16>
      %1058 = vector.extract %1049[2] : f16 from vector<4xf16>
      %1059 = vector.extract %1049[3] : f16 from vector<4xf16>
      %1060 = vector.from_elements %1056, %1057, %1058, %1059 : vector<4xf16>
      %1061 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %912, %1060, %913 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1062 = vector.extract %1055[0] : f32 from vector<4xf32>
      %1063 = vector.extract %1055[1] : f32 from vector<4xf32>
      %1064 = vector.extract %1055[2] : f32 from vector<4xf32>
      %1065 = vector.extract %1055[3] : f32 from vector<4xf32>
      %1066 = vector.load %view_6[%598] : memref<4096xf16, 3>, vector<4xf16>
      %1067 = vector.load %view_6[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1068 = vector.extract %1066[0] : f16 from vector<4xf16>
      %1069 = vector.extract %1066[1] : f16 from vector<4xf16>
      %1070 = vector.extract %1066[2] : f16 from vector<4xf16>
      %1071 = vector.extract %1066[3] : f16 from vector<4xf16>
      %1072 = vector.from_elements %1068, %1069, %1070, %1071 : vector<4xf16>
      %1073 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1072, %1061 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1074 = vector.extract %1067[0] : f16 from vector<4xf16>
      %1075 = vector.extract %1067[1] : f16 from vector<4xf16>
      %1076 = vector.extract %1067[2] : f16 from vector<4xf16>
      %1077 = vector.extract %1067[3] : f16 from vector<4xf16>
      %1078 = vector.from_elements %1074, %1075, %1076, %1077 : vector<4xf16>
      %1079 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %912, %1078, %915 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1080 = vector.extract %1073[0] : f32 from vector<4xf32>
      %1081 = vector.extract %1073[1] : f32 from vector<4xf32>
      %1082 = vector.extract %1073[2] : f32 from vector<4xf32>
      %1083 = vector.extract %1073[3] : f32 from vector<4xf32>
      %1084 = vector.load %view_6[%618] : memref<4096xf16, 3>, vector<4xf16>
      %1085 = vector.load %view_6[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1086 = vector.extract %1084[0] : f16 from vector<4xf16>
      %1087 = vector.extract %1084[1] : f16 from vector<4xf16>
      %1088 = vector.extract %1084[2] : f16 from vector<4xf16>
      %1089 = vector.extract %1084[3] : f16 from vector<4xf16>
      %1090 = vector.from_elements %1086, %1087, %1088, %1089 : vector<4xf16>
      %1091 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1090, %1079 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1092 = vector.extract %1085[0] : f16 from vector<4xf16>
      %1093 = vector.extract %1085[1] : f16 from vector<4xf16>
      %1094 = vector.extract %1085[2] : f16 from vector<4xf16>
      %1095 = vector.extract %1085[3] : f16 from vector<4xf16>
      %1096 = vector.from_elements %1092, %1093, %1094, %1095 : vector<4xf16>
      %1097 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %912, %1096, %916 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1098 = vector.extract %1091[0] : f32 from vector<4xf32>
      %1099 = vector.extract %1091[1] : f32 from vector<4xf32>
      %1100 = vector.extract %1091[2] : f32 from vector<4xf32>
      %1101 = vector.extract %1091[3] : f32 from vector<4xf32>
      %1102 = vector.load %view_6[%638] : memref<4096xf16, 3>, vector<4xf16>
      %1103 = vector.load %view_6[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1104 = vector.extract %1102[0] : f16 from vector<4xf16>
      %1105 = vector.extract %1102[1] : f16 from vector<4xf16>
      %1106 = vector.extract %1102[2] : f16 from vector<4xf16>
      %1107 = vector.extract %1102[3] : f16 from vector<4xf16>
      %1108 = vector.from_elements %1104, %1105, %1106, %1107 : vector<4xf16>
      %1109 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1108, %1097 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1110 = vector.extract %1103[0] : f16 from vector<4xf16>
      %1111 = vector.extract %1103[1] : f16 from vector<4xf16>
      %1112 = vector.extract %1103[2] : f16 from vector<4xf16>
      %1113 = vector.extract %1103[3] : f16 from vector<4xf16>
      %1114 = vector.from_elements %1110, %1111, %1112, %1113 : vector<4xf16>
      %1115 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %912, %1114, %917 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1116 = vector.extract %1109[0] : f32 from vector<4xf32>
      %1117 = vector.extract %1109[1] : f32 from vector<4xf32>
      %1118 = vector.extract %1109[2] : f32 from vector<4xf32>
      %1119 = vector.extract %1109[3] : f32 from vector<4xf32>
      %1120 = vector.load %view_6[%658] : memref<4096xf16, 3>, vector<4xf16>
      %1121 = vector.load %view_6[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1122 = vector.extract %1120[0] : f16 from vector<4xf16>
      %1123 = vector.extract %1120[1] : f16 from vector<4xf16>
      %1124 = vector.extract %1120[2] : f16 from vector<4xf16>
      %1125 = vector.extract %1120[3] : f16 from vector<4xf16>
      %1126 = vector.from_elements %1122, %1123, %1124, %1125 : vector<4xf16>
      %1127 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1126, %1115 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1128 = vector.extract %1121[0] : f16 from vector<4xf16>
      %1129 = vector.extract %1121[1] : f16 from vector<4xf16>
      %1130 = vector.extract %1121[2] : f16 from vector<4xf16>
      %1131 = vector.extract %1121[3] : f16 from vector<4xf16>
      %1132 = vector.from_elements %1128, %1129, %1130, %1131 : vector<4xf16>
      %1133 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %912, %1132, %918 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1134 = vector.extract %1127[0] : f32 from vector<4xf32>
      %1135 = vector.extract %1127[1] : f32 from vector<4xf32>
      %1136 = vector.extract %1127[2] : f32 from vector<4xf32>
      %1137 = vector.extract %1127[3] : f32 from vector<4xf32>
      %1138 = vector.load %view_6[%678] : memref<4096xf16, 3>, vector<4xf16>
      %1139 = vector.load %view_6[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1140 = vector.extract %1138[0] : f16 from vector<4xf16>
      %1141 = vector.extract %1138[1] : f16 from vector<4xf16>
      %1142 = vector.extract %1138[2] : f16 from vector<4xf16>
      %1143 = vector.extract %1138[3] : f16 from vector<4xf16>
      %1144 = vector.from_elements %1140, %1141, %1142, %1143 : vector<4xf16>
      %1145 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1144, %1133 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1146 = vector.extract %1139[0] : f16 from vector<4xf16>
      %1147 = vector.extract %1139[1] : f16 from vector<4xf16>
      %1148 = vector.extract %1139[2] : f16 from vector<4xf16>
      %1149 = vector.extract %1139[3] : f16 from vector<4xf16>
      %1150 = vector.from_elements %1146, %1147, %1148, %1149 : vector<4xf16>
      %1151 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %912, %1150, %919 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1152 = vector.extract %1145[0] : f32 from vector<4xf32>
      %1153 = vector.extract %1145[1] : f32 from vector<4xf32>
      %1154 = vector.extract %1145[2] : f32 from vector<4xf32>
      %1155 = vector.extract %1145[3] : f32 from vector<4xf32>
      %1156 = vector.load %view_6[%698] : memref<4096xf16, 3>, vector<4xf16>
      %1157 = vector.load %view_6[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1158 = vector.extract %1156[0] : f16 from vector<4xf16>
      %1159 = vector.extract %1156[1] : f16 from vector<4xf16>
      %1160 = vector.extract %1156[2] : f16 from vector<4xf16>
      %1161 = vector.extract %1156[3] : f16 from vector<4xf16>
      %1162 = vector.from_elements %1158, %1159, %1160, %1161 : vector<4xf16>
      %1163 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1162, %1151 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1164 = vector.extract %1157[0] : f16 from vector<4xf16>
      %1165 = vector.extract %1157[1] : f16 from vector<4xf16>
      %1166 = vector.extract %1157[2] : f16 from vector<4xf16>
      %1167 = vector.extract %1157[3] : f16 from vector<4xf16>
      %1168 = vector.from_elements %1164, %1165, %1166, %1167 : vector<4xf16>
      %1169 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %912, %1168, %920 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1170 = vector.extract %1163[0] : f32 from vector<4xf32>
      %1171 = vector.extract %1163[1] : f32 from vector<4xf32>
      %1172 = vector.extract %1163[2] : f32 from vector<4xf32>
      %1173 = vector.extract %1163[3] : f32 from vector<4xf32>
      %1174 = vector.load %view_6[%718] : memref<4096xf16, 3>, vector<4xf16>
      %1175 = vector.load %view_6[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1176 = vector.extract %1174[0] : f16 from vector<4xf16>
      %1177 = vector.extract %1174[1] : f16 from vector<4xf16>
      %1178 = vector.extract %1174[2] : f16 from vector<4xf16>
      %1179 = vector.extract %1174[3] : f16 from vector<4xf16>
      %1180 = vector.from_elements %1176, %1177, %1178, %1179 : vector<4xf16>
      %1181 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1180, %1169 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1182 = vector.extract %1175[0] : f16 from vector<4xf16>
      %1183 = vector.extract %1175[1] : f16 from vector<4xf16>
      %1184 = vector.extract %1175[2] : f16 from vector<4xf16>
      %1185 = vector.extract %1175[3] : f16 from vector<4xf16>
      %1186 = vector.from_elements %1182, %1183, %1184, %1185 : vector<4xf16>
      %1187 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %912, %1186, %921 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1188 = vector.extract %1181[0] : f32 from vector<4xf32>
      %1189 = vector.extract %1181[1] : f32 from vector<4xf32>
      %1190 = vector.extract %1181[2] : f32 from vector<4xf32>
      %1191 = vector.extract %1181[3] : f32 from vector<4xf32>
      %1192 = vector.load %view_6[%738] : memref<4096xf16, 3>, vector<4xf16>
      %1193 = vector.extract %1192[0] : f16 from vector<4xf16>
      %1194 = vector.extract %1192[1] : f16 from vector<4xf16>
      %1195 = vector.extract %1192[2] : f16 from vector<4xf16>
      %1196 = vector.extract %1192[3] : f16 from vector<4xf16>
      %1197 = vector.from_elements %1193, %1194, %1195, %1196 : vector<4xf16>
      %1198 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %914, %1197, %1187 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1199 = vector.extract %1198[0] : f32 from vector<4xf32>
      %1200 = vector.extract %1198[1] : f32 from vector<4xf32>
      %1201 = vector.extract %1198[2] : f32 from vector<4xf32>
      %1202 = vector.extract %1198[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %894 : !frisk.async_task
      %subview_139 = memref.subview %transpose[0, %198, 96, %583] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %1203 = frisk.async_copy_gm_to_shm %subview_139 to %view_6 {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> ((((d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4) * 8 + d0 mod 8) floordiv 1024) * 8 + d0 mod 8, (d0 floordiv 8 - (((d0 floordiv 8) floordiv 4) mod 2) * 4 + ((((d0 floordiv 8) floordiv 16) mod 2 + ((d0 floordiv 8) floordiv 4) mod 2) mod 2) * 4 + (d0 mod 8) floordiv 8) mod 128)>} : memref<32x128xf16, strided<[1, 128], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %1204 = vector.load %view[%586] : memref<4096xf16, 3>, vector<4xf16>
      %1205 = vector.from_elements %130, %131, %132, %133 : vector<4xf16>
      %1206 = vector.from_elements %936, %937, %938, %939 : vector<4xf32>
      %1207 = vector.from_elements %134, %135, %136, %137 : vector<4xf16>
      %1208 = vector.from_elements %954, %955, %956, %957 : vector<4xf32>
      %1209 = vector.extract %1204[0] : f16 from vector<4xf16>
      %1210 = vector.extract %1204[1] : f16 from vector<4xf16>
      %1211 = vector.extract %1204[2] : f16 from vector<4xf16>
      %1212 = vector.extract %1204[3] : f16 from vector<4xf16>
      %1213 = vector.from_elements %1209, %1210, %1211, %1212 : vector<4xf16>
      %1214 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1213, %1206 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1215 = vector.from_elements %972, %973, %974, %975 : vector<4xf32>
      %1216 = vector.from_elements %990, %991, %992, %993 : vector<4xf32>
      %1217 = vector.from_elements %1008, %1009, %1010, %1011 : vector<4xf32>
      %1218 = vector.from_elements %1026, %1027, %1028, %1029 : vector<4xf32>
      %1219 = vector.from_elements %1044, %1045, %1046, %1047 : vector<4xf32>
      %1220 = vector.from_elements %1062, %1063, %1064, %1065 : vector<4xf32>
      %1221 = vector.from_elements %146, %147, %148, %149 : vector<4xf16>
      %1222 = vector.from_elements %1080, %1081, %1082, %1083 : vector<4xf32>
      %1223 = vector.from_elements %150, %151, %152, %153 : vector<4xf16>
      %1224 = vector.from_elements %1098, %1099, %1100, %1101 : vector<4xf32>
      %1225 = vector.from_elements %1116, %1117, %1118, %1119 : vector<4xf32>
      %1226 = vector.from_elements %1134, %1135, %1136, %1137 : vector<4xf32>
      %1227 = vector.from_elements %1152, %1153, %1154, %1155 : vector<4xf32>
      %1228 = vector.from_elements %1170, %1171, %1172, %1173 : vector<4xf32>
      %1229 = vector.from_elements %1188, %1189, %1190, %1191 : vector<4xf32>
      %1230 = vector.from_elements %1199, %1200, %1201, %1202 : vector<4xf32>
      %1231 = vector.load %view[%598] : memref<4096xf16, 3>, vector<4xf16>
      %1232 = vector.load %view[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1233 = vector.extract %1231[0] : f16 from vector<4xf16>
      %1234 = vector.extract %1231[1] : f16 from vector<4xf16>
      %1235 = vector.extract %1231[2] : f16 from vector<4xf16>
      %1236 = vector.extract %1231[3] : f16 from vector<4xf16>
      %1237 = vector.from_elements %1233, %1234, %1235, %1236 : vector<4xf16>
      %1238 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1237, %1214 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1239 = vector.extract %1232[0] : f16 from vector<4xf16>
      %1240 = vector.extract %1232[1] : f16 from vector<4xf16>
      %1241 = vector.extract %1232[2] : f16 from vector<4xf16>
      %1242 = vector.extract %1232[3] : f16 from vector<4xf16>
      %1243 = vector.from_elements %1239, %1240, %1241, %1242 : vector<4xf16>
      %1244 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1243, %1208 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1245 = vector.extract %1238[0] : f32 from vector<4xf32>
      %1246 = vector.extract %1238[1] : f32 from vector<4xf32>
      %1247 = vector.extract %1238[2] : f32 from vector<4xf32>
      %1248 = vector.extract %1238[3] : f32 from vector<4xf32>
      %1249 = vector.load %view[%618] : memref<4096xf16, 3>, vector<4xf16>
      %1250 = vector.load %view[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1251 = vector.extract %1249[0] : f16 from vector<4xf16>
      %1252 = vector.extract %1249[1] : f16 from vector<4xf16>
      %1253 = vector.extract %1249[2] : f16 from vector<4xf16>
      %1254 = vector.extract %1249[3] : f16 from vector<4xf16>
      %1255 = vector.from_elements %1251, %1252, %1253, %1254 : vector<4xf16>
      %1256 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1255, %1244 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1257 = vector.extract %1250[0] : f16 from vector<4xf16>
      %1258 = vector.extract %1250[1] : f16 from vector<4xf16>
      %1259 = vector.extract %1250[2] : f16 from vector<4xf16>
      %1260 = vector.extract %1250[3] : f16 from vector<4xf16>
      %1261 = vector.from_elements %1257, %1258, %1259, %1260 : vector<4xf16>
      %1262 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1261, %1215 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1263 = vector.extract %1256[0] : f32 from vector<4xf32>
      %1264 = vector.extract %1256[1] : f32 from vector<4xf32>
      %1265 = vector.extract %1256[2] : f32 from vector<4xf32>
      %1266 = vector.extract %1256[3] : f32 from vector<4xf32>
      %1267 = vector.load %view[%638] : memref<4096xf16, 3>, vector<4xf16>
      %1268 = vector.load %view[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1269 = vector.extract %1267[0] : f16 from vector<4xf16>
      %1270 = vector.extract %1267[1] : f16 from vector<4xf16>
      %1271 = vector.extract %1267[2] : f16 from vector<4xf16>
      %1272 = vector.extract %1267[3] : f16 from vector<4xf16>
      %1273 = vector.from_elements %1269, %1270, %1271, %1272 : vector<4xf16>
      %1274 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1273, %1262 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1275 = vector.extract %1268[0] : f16 from vector<4xf16>
      %1276 = vector.extract %1268[1] : f16 from vector<4xf16>
      %1277 = vector.extract %1268[2] : f16 from vector<4xf16>
      %1278 = vector.extract %1268[3] : f16 from vector<4xf16>
      %1279 = vector.from_elements %1275, %1276, %1277, %1278 : vector<4xf16>
      %1280 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1279, %1216 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1281 = vector.extract %1274[0] : f32 from vector<4xf32>
      %1282 = vector.extract %1274[1] : f32 from vector<4xf32>
      %1283 = vector.extract %1274[2] : f32 from vector<4xf32>
      %1284 = vector.extract %1274[3] : f32 from vector<4xf32>
      %1285 = vector.load %view[%658] : memref<4096xf16, 3>, vector<4xf16>
      %1286 = vector.load %view[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1287 = vector.extract %1285[0] : f16 from vector<4xf16>
      %1288 = vector.extract %1285[1] : f16 from vector<4xf16>
      %1289 = vector.extract %1285[2] : f16 from vector<4xf16>
      %1290 = vector.extract %1285[3] : f16 from vector<4xf16>
      %1291 = vector.from_elements %1287, %1288, %1289, %1290 : vector<4xf16>
      %1292 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1291, %1280 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1293 = vector.extract %1286[0] : f16 from vector<4xf16>
      %1294 = vector.extract %1286[1] : f16 from vector<4xf16>
      %1295 = vector.extract %1286[2] : f16 from vector<4xf16>
      %1296 = vector.extract %1286[3] : f16 from vector<4xf16>
      %1297 = vector.from_elements %1293, %1294, %1295, %1296 : vector<4xf16>
      %1298 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1297, %1217 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1299 = vector.extract %1292[0] : f32 from vector<4xf32>
      %1300 = vector.extract %1292[1] : f32 from vector<4xf32>
      %1301 = vector.extract %1292[2] : f32 from vector<4xf32>
      %1302 = vector.extract %1292[3] : f32 from vector<4xf32>
      %1303 = vector.load %view[%678] : memref<4096xf16, 3>, vector<4xf16>
      %1304 = vector.load %view[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1305 = vector.extract %1303[0] : f16 from vector<4xf16>
      %1306 = vector.extract %1303[1] : f16 from vector<4xf16>
      %1307 = vector.extract %1303[2] : f16 from vector<4xf16>
      %1308 = vector.extract %1303[3] : f16 from vector<4xf16>
      %1309 = vector.from_elements %1305, %1306, %1307, %1308 : vector<4xf16>
      %1310 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1309, %1298 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1311 = vector.extract %1304[0] : f16 from vector<4xf16>
      %1312 = vector.extract %1304[1] : f16 from vector<4xf16>
      %1313 = vector.extract %1304[2] : f16 from vector<4xf16>
      %1314 = vector.extract %1304[3] : f16 from vector<4xf16>
      %1315 = vector.from_elements %1311, %1312, %1313, %1314 : vector<4xf16>
      %1316 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1315, %1218 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1317 = vector.extract %1310[0] : f32 from vector<4xf32>
      %1318 = vector.extract %1310[1] : f32 from vector<4xf32>
      %1319 = vector.extract %1310[2] : f32 from vector<4xf32>
      %1320 = vector.extract %1310[3] : f32 from vector<4xf32>
      %1321 = vector.load %view[%698] : memref<4096xf16, 3>, vector<4xf16>
      %1322 = vector.load %view[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1323 = vector.extract %1321[0] : f16 from vector<4xf16>
      %1324 = vector.extract %1321[1] : f16 from vector<4xf16>
      %1325 = vector.extract %1321[2] : f16 from vector<4xf16>
      %1326 = vector.extract %1321[3] : f16 from vector<4xf16>
      %1327 = vector.from_elements %1323, %1324, %1325, %1326 : vector<4xf16>
      %1328 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1327, %1316 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1329 = vector.extract %1322[0] : f16 from vector<4xf16>
      %1330 = vector.extract %1322[1] : f16 from vector<4xf16>
      %1331 = vector.extract %1322[2] : f16 from vector<4xf16>
      %1332 = vector.extract %1322[3] : f16 from vector<4xf16>
      %1333 = vector.from_elements %1329, %1330, %1331, %1332 : vector<4xf16>
      %1334 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1333, %1219 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1335 = vector.extract %1328[0] : f32 from vector<4xf32>
      %1336 = vector.extract %1328[1] : f32 from vector<4xf32>
      %1337 = vector.extract %1328[2] : f32 from vector<4xf32>
      %1338 = vector.extract %1328[3] : f32 from vector<4xf32>
      %1339 = vector.load %view[%718] : memref<4096xf16, 3>, vector<4xf16>
      %1340 = vector.load %view[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1341 = vector.extract %1339[0] : f16 from vector<4xf16>
      %1342 = vector.extract %1339[1] : f16 from vector<4xf16>
      %1343 = vector.extract %1339[2] : f16 from vector<4xf16>
      %1344 = vector.extract %1339[3] : f16 from vector<4xf16>
      %1345 = vector.from_elements %1341, %1342, %1343, %1344 : vector<4xf16>
      %1346 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1345, %1334 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1347 = vector.extract %1340[0] : f16 from vector<4xf16>
      %1348 = vector.extract %1340[1] : f16 from vector<4xf16>
      %1349 = vector.extract %1340[2] : f16 from vector<4xf16>
      %1350 = vector.extract %1340[3] : f16 from vector<4xf16>
      %1351 = vector.from_elements %1347, %1348, %1349, %1350 : vector<4xf16>
      %1352 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1351, %1220 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1353 = vector.extract %1346[0] : f32 from vector<4xf32>
      %1354 = vector.extract %1346[1] : f32 from vector<4xf32>
      %1355 = vector.extract %1346[2] : f32 from vector<4xf32>
      %1356 = vector.extract %1346[3] : f32 from vector<4xf32>
      %1357 = vector.load %view[%738] : memref<4096xf16, 3>, vector<4xf16>
      %1358 = vector.load %view[%586] : memref<4096xf16, 3>, vector<4xf16>
      %1359 = vector.extract %1357[0] : f16 from vector<4xf16>
      %1360 = vector.extract %1357[1] : f16 from vector<4xf16>
      %1361 = vector.extract %1357[2] : f16 from vector<4xf16>
      %1362 = vector.extract %1357[3] : f16 from vector<4xf16>
      %1363 = vector.from_elements %1359, %1360, %1361, %1362 : vector<4xf16>
      %1364 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1207, %1363, %1352 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1365 = vector.extract %1358[0] : f16 from vector<4xf16>
      %1366 = vector.extract %1358[1] : f16 from vector<4xf16>
      %1367 = vector.extract %1358[2] : f16 from vector<4xf16>
      %1368 = vector.extract %1358[3] : f16 from vector<4xf16>
      %1369 = vector.from_elements %1365, %1366, %1367, %1368 : vector<4xf16>
      %1370 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1369, %1222 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1371 = vector.extract %1364[0] : f32 from vector<4xf32>
      %1372 = vector.extract %1364[1] : f32 from vector<4xf32>
      %1373 = vector.extract %1364[2] : f32 from vector<4xf32>
      %1374 = vector.extract %1364[3] : f32 from vector<4xf32>
      %1375 = vector.load %view[%598] : memref<4096xf16, 3>, vector<4xf16>
      %1376 = vector.load %view[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1377 = vector.extract %1375[0] : f16 from vector<4xf16>
      %1378 = vector.extract %1375[1] : f16 from vector<4xf16>
      %1379 = vector.extract %1375[2] : f16 from vector<4xf16>
      %1380 = vector.extract %1375[3] : f16 from vector<4xf16>
      %1381 = vector.from_elements %1377, %1378, %1379, %1380 : vector<4xf16>
      %1382 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1381, %1370 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1383 = vector.extract %1376[0] : f16 from vector<4xf16>
      %1384 = vector.extract %1376[1] : f16 from vector<4xf16>
      %1385 = vector.extract %1376[2] : f16 from vector<4xf16>
      %1386 = vector.extract %1376[3] : f16 from vector<4xf16>
      %1387 = vector.from_elements %1383, %1384, %1385, %1386 : vector<4xf16>
      %1388 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1387, %1224 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1389 = vector.extract %1382[0] : f32 from vector<4xf32>
      %1390 = vector.extract %1382[1] : f32 from vector<4xf32>
      %1391 = vector.extract %1382[2] : f32 from vector<4xf32>
      %1392 = vector.extract %1382[3] : f32 from vector<4xf32>
      %1393 = vector.load %view[%618] : memref<4096xf16, 3>, vector<4xf16>
      %1394 = vector.load %view[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1395 = vector.extract %1393[0] : f16 from vector<4xf16>
      %1396 = vector.extract %1393[1] : f16 from vector<4xf16>
      %1397 = vector.extract %1393[2] : f16 from vector<4xf16>
      %1398 = vector.extract %1393[3] : f16 from vector<4xf16>
      %1399 = vector.from_elements %1395, %1396, %1397, %1398 : vector<4xf16>
      %1400 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1399, %1388 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1401 = vector.extract %1394[0] : f16 from vector<4xf16>
      %1402 = vector.extract %1394[1] : f16 from vector<4xf16>
      %1403 = vector.extract %1394[2] : f16 from vector<4xf16>
      %1404 = vector.extract %1394[3] : f16 from vector<4xf16>
      %1405 = vector.from_elements %1401, %1402, %1403, %1404 : vector<4xf16>
      %1406 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1405, %1225 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1407 = vector.extract %1400[0] : f32 from vector<4xf32>
      %1408 = vector.extract %1400[1] : f32 from vector<4xf32>
      %1409 = vector.extract %1400[2] : f32 from vector<4xf32>
      %1410 = vector.extract %1400[3] : f32 from vector<4xf32>
      %1411 = vector.load %view[%638] : memref<4096xf16, 3>, vector<4xf16>
      %1412 = vector.load %view[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1413 = vector.extract %1411[0] : f16 from vector<4xf16>
      %1414 = vector.extract %1411[1] : f16 from vector<4xf16>
      %1415 = vector.extract %1411[2] : f16 from vector<4xf16>
      %1416 = vector.extract %1411[3] : f16 from vector<4xf16>
      %1417 = vector.from_elements %1413, %1414, %1415, %1416 : vector<4xf16>
      %1418 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1417, %1406 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1419 = vector.extract %1412[0] : f16 from vector<4xf16>
      %1420 = vector.extract %1412[1] : f16 from vector<4xf16>
      %1421 = vector.extract %1412[2] : f16 from vector<4xf16>
      %1422 = vector.extract %1412[3] : f16 from vector<4xf16>
      %1423 = vector.from_elements %1419, %1420, %1421, %1422 : vector<4xf16>
      %1424 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1423, %1226 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1425 = vector.extract %1418[0] : f32 from vector<4xf32>
      %1426 = vector.extract %1418[1] : f32 from vector<4xf32>
      %1427 = vector.extract %1418[2] : f32 from vector<4xf32>
      %1428 = vector.extract %1418[3] : f32 from vector<4xf32>
      %1429 = vector.load %view[%658] : memref<4096xf16, 3>, vector<4xf16>
      %1430 = vector.load %view[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1431 = vector.extract %1429[0] : f16 from vector<4xf16>
      %1432 = vector.extract %1429[1] : f16 from vector<4xf16>
      %1433 = vector.extract %1429[2] : f16 from vector<4xf16>
      %1434 = vector.extract %1429[3] : f16 from vector<4xf16>
      %1435 = vector.from_elements %1431, %1432, %1433, %1434 : vector<4xf16>
      %1436 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1435, %1424 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1437 = vector.extract %1430[0] : f16 from vector<4xf16>
      %1438 = vector.extract %1430[1] : f16 from vector<4xf16>
      %1439 = vector.extract %1430[2] : f16 from vector<4xf16>
      %1440 = vector.extract %1430[3] : f16 from vector<4xf16>
      %1441 = vector.from_elements %1437, %1438, %1439, %1440 : vector<4xf16>
      %1442 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1441, %1227 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1443 = vector.extract %1436[0] : f32 from vector<4xf32>
      %1444 = vector.extract %1436[1] : f32 from vector<4xf32>
      %1445 = vector.extract %1436[2] : f32 from vector<4xf32>
      %1446 = vector.extract %1436[3] : f32 from vector<4xf32>
      %1447 = vector.load %view[%678] : memref<4096xf16, 3>, vector<4xf16>
      %1448 = vector.load %view[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1449 = vector.extract %1447[0] : f16 from vector<4xf16>
      %1450 = vector.extract %1447[1] : f16 from vector<4xf16>
      %1451 = vector.extract %1447[2] : f16 from vector<4xf16>
      %1452 = vector.extract %1447[3] : f16 from vector<4xf16>
      %1453 = vector.from_elements %1449, %1450, %1451, %1452 : vector<4xf16>
      %1454 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1453, %1442 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1455 = vector.extract %1448[0] : f16 from vector<4xf16>
      %1456 = vector.extract %1448[1] : f16 from vector<4xf16>
      %1457 = vector.extract %1448[2] : f16 from vector<4xf16>
      %1458 = vector.extract %1448[3] : f16 from vector<4xf16>
      %1459 = vector.from_elements %1455, %1456, %1457, %1458 : vector<4xf16>
      %1460 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1459, %1228 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1461 = vector.extract %1454[0] : f32 from vector<4xf32>
      %1462 = vector.extract %1454[1] : f32 from vector<4xf32>
      %1463 = vector.extract %1454[2] : f32 from vector<4xf32>
      %1464 = vector.extract %1454[3] : f32 from vector<4xf32>
      %1465 = vector.load %view[%698] : memref<4096xf16, 3>, vector<4xf16>
      %1466 = vector.load %view[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1467 = vector.extract %1465[0] : f16 from vector<4xf16>
      %1468 = vector.extract %1465[1] : f16 from vector<4xf16>
      %1469 = vector.extract %1465[2] : f16 from vector<4xf16>
      %1470 = vector.extract %1465[3] : f16 from vector<4xf16>
      %1471 = vector.from_elements %1467, %1468, %1469, %1470 : vector<4xf16>
      %1472 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1471, %1460 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1473 = vector.extract %1466[0] : f16 from vector<4xf16>
      %1474 = vector.extract %1466[1] : f16 from vector<4xf16>
      %1475 = vector.extract %1466[2] : f16 from vector<4xf16>
      %1476 = vector.extract %1466[3] : f16 from vector<4xf16>
      %1477 = vector.from_elements %1473, %1474, %1475, %1476 : vector<4xf16>
      %1478 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1477, %1229 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1479 = vector.extract %1472[0] : f32 from vector<4xf32>
      %1480 = vector.extract %1472[1] : f32 from vector<4xf32>
      %1481 = vector.extract %1472[2] : f32 from vector<4xf32>
      %1482 = vector.extract %1472[3] : f32 from vector<4xf32>
      %1483 = vector.load %view[%718] : memref<4096xf16, 3>, vector<4xf16>
      %1484 = vector.load %view[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1485 = vector.extract %1483[0] : f16 from vector<4xf16>
      %1486 = vector.extract %1483[1] : f16 from vector<4xf16>
      %1487 = vector.extract %1483[2] : f16 from vector<4xf16>
      %1488 = vector.extract %1483[3] : f16 from vector<4xf16>
      %1489 = vector.from_elements %1485, %1486, %1487, %1488 : vector<4xf16>
      %1490 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1489, %1478 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1491 = vector.extract %1484[0] : f16 from vector<4xf16>
      %1492 = vector.extract %1484[1] : f16 from vector<4xf16>
      %1493 = vector.extract %1484[2] : f16 from vector<4xf16>
      %1494 = vector.extract %1484[3] : f16 from vector<4xf16>
      %1495 = vector.from_elements %1491, %1492, %1493, %1494 : vector<4xf16>
      %1496 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1495, %1230 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1497 = vector.extract %1490[0] : f32 from vector<4xf32>
      %1498 = vector.extract %1490[1] : f32 from vector<4xf32>
      %1499 = vector.extract %1490[2] : f32 from vector<4xf32>
      %1500 = vector.extract %1490[3] : f32 from vector<4xf32>
      %1501 = vector.load %view[%738] : memref<4096xf16, 3>, vector<4xf16>
      %1502 = vector.extract %1501[0] : f16 from vector<4xf16>
      %1503 = vector.extract %1501[1] : f16 from vector<4xf16>
      %1504 = vector.extract %1501[2] : f16 from vector<4xf16>
      %1505 = vector.extract %1501[3] : f16 from vector<4xf16>
      %1506 = vector.from_elements %1502, %1503, %1504, %1505 : vector<4xf16>
      %1507 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1223, %1506, %1496 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1508 = vector.extract %1507[0] : f32 from vector<4xf32>
      %1509 = vector.extract %1507[1] : f32 from vector<4xf32>
      %1510 = vector.extract %1507[2] : f32 from vector<4xf32>
      %1511 = vector.extract %1507[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %1203 : !frisk.async_task
      %subview_140 = memref.subview %arg2[0, %198, %583, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %1512 = frisk.async_copy_gm_to_shm %subview_140 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %1513 = vector.load %view_6[%586] : memref<4096xf16, 3>, vector<4xf16>
      %1514 = vector.from_elements %174, %175, %176, %177 : vector<4xf16>
      %1515 = vector.from_elements %1245, %1246, %1247, %1248 : vector<4xf32>
      %1516 = vector.from_elements %178, %179, %180, %181 : vector<4xf16>
      %1517 = vector.from_elements %1263, %1264, %1265, %1266 : vector<4xf32>
      %1518 = vector.extract %1513[0] : f16 from vector<4xf16>
      %1519 = vector.extract %1513[1] : f16 from vector<4xf16>
      %1520 = vector.extract %1513[2] : f16 from vector<4xf16>
      %1521 = vector.extract %1513[3] : f16 from vector<4xf16>
      %1522 = vector.from_elements %1518, %1519, %1520, %1521 : vector<4xf16>
      %1523 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1514, %1522, %1515 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1524 = vector.from_elements %1281, %1282, %1283, %1284 : vector<4xf32>
      %1525 = vector.from_elements %1299, %1300, %1301, %1302 : vector<4xf32>
      %1526 = vector.from_elements %1317, %1318, %1319, %1320 : vector<4xf32>
      %1527 = vector.from_elements %1335, %1336, %1337, %1338 : vector<4xf32>
      %1528 = vector.from_elements %1353, %1354, %1355, %1356 : vector<4xf32>
      %1529 = vector.from_elements %1371, %1372, %1373, %1374 : vector<4xf32>
      %1530 = vector.from_elements %190, %191, %192, %193 : vector<4xf16>
      %1531 = vector.from_elements %1389, %1390, %1391, %1392 : vector<4xf32>
      %1532 = vector.from_elements %194, %195, %196, %197 : vector<4xf16>
      %1533 = vector.from_elements %1407, %1408, %1409, %1410 : vector<4xf32>
      %1534 = vector.from_elements %1425, %1426, %1427, %1428 : vector<4xf32>
      %1535 = vector.from_elements %1443, %1444, %1445, %1446 : vector<4xf32>
      %1536 = vector.from_elements %1461, %1462, %1463, %1464 : vector<4xf32>
      %1537 = vector.from_elements %1479, %1480, %1481, %1482 : vector<4xf32>
      %1538 = vector.from_elements %1497, %1498, %1499, %1500 : vector<4xf32>
      %1539 = vector.from_elements %1508, %1509, %1510, %1511 : vector<4xf32>
      %1540 = vector.load %view_6[%598] : memref<4096xf16, 3>, vector<4xf16>
      %1541 = vector.load %view_6[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1542 = vector.extract %1540[0] : f16 from vector<4xf16>
      %1543 = vector.extract %1540[1] : f16 from vector<4xf16>
      %1544 = vector.extract %1540[2] : f16 from vector<4xf16>
      %1545 = vector.extract %1540[3] : f16 from vector<4xf16>
      %1546 = vector.from_elements %1542, %1543, %1544, %1545 : vector<4xf16>
      %1547 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1546, %1523 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1548 = vector.extract %1541[0] : f16 from vector<4xf16>
      %1549 = vector.extract %1541[1] : f16 from vector<4xf16>
      %1550 = vector.extract %1541[2] : f16 from vector<4xf16>
      %1551 = vector.extract %1541[3] : f16 from vector<4xf16>
      %1552 = vector.from_elements %1548, %1549, %1550, %1551 : vector<4xf16>
      %1553 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1514, %1552, %1517 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1554 = vector.extract %1547[0] : f32 from vector<4xf32>
      %1555 = arith.maxnumf %1554, %cst_2 : f32
      %1556 = vector.extract %1547[1] : f32 from vector<4xf32>
      %1557 = arith.maxnumf %1555, %1556 : f32
      %1558 = vector.extract %1547[2] : f32 from vector<4xf32>
      %1559 = arith.maxnumf %1557, %1558 : f32
      %1560 = vector.extract %1547[3] : f32 from vector<4xf32>
      %1561 = arith.maxnumf %1559, %1560 : f32
      %1562 = vector.load %view_6[%618] : memref<4096xf16, 3>, vector<4xf16>
      %1563 = vector.load %view_6[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1564 = vector.extract %1562[0] : f16 from vector<4xf16>
      %1565 = vector.extract %1562[1] : f16 from vector<4xf16>
      %1566 = vector.extract %1562[2] : f16 from vector<4xf16>
      %1567 = vector.extract %1562[3] : f16 from vector<4xf16>
      %1568 = vector.from_elements %1564, %1565, %1566, %1567 : vector<4xf16>
      %1569 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1568, %1553 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1570 = vector.extract %1563[0] : f16 from vector<4xf16>
      %1571 = vector.extract %1563[1] : f16 from vector<4xf16>
      %1572 = vector.extract %1563[2] : f16 from vector<4xf16>
      %1573 = vector.extract %1563[3] : f16 from vector<4xf16>
      %1574 = vector.from_elements %1570, %1571, %1572, %1573 : vector<4xf16>
      %1575 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1514, %1574, %1524 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1576 = vector.extract %1569[0] : f32 from vector<4xf32>
      %1577 = arith.maxnumf %1561, %1576 : f32
      %1578 = vector.extract %1569[1] : f32 from vector<4xf32>
      %1579 = arith.maxnumf %1577, %1578 : f32
      %1580 = vector.extract %1569[2] : f32 from vector<4xf32>
      %1581 = arith.maxnumf %1579, %1580 : f32
      %1582 = vector.extract %1569[3] : f32 from vector<4xf32>
      %1583 = arith.maxnumf %1581, %1582 : f32
      %1584 = vector.load %view_6[%638] : memref<4096xf16, 3>, vector<4xf16>
      %1585 = vector.load %view_6[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1586 = vector.extract %1584[0] : f16 from vector<4xf16>
      %1587 = vector.extract %1584[1] : f16 from vector<4xf16>
      %1588 = vector.extract %1584[2] : f16 from vector<4xf16>
      %1589 = vector.extract %1584[3] : f16 from vector<4xf16>
      %1590 = vector.from_elements %1586, %1587, %1588, %1589 : vector<4xf16>
      %1591 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1590, %1575 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1592 = vector.extract %1585[0] : f16 from vector<4xf16>
      %1593 = vector.extract %1585[1] : f16 from vector<4xf16>
      %1594 = vector.extract %1585[2] : f16 from vector<4xf16>
      %1595 = vector.extract %1585[3] : f16 from vector<4xf16>
      %1596 = vector.from_elements %1592, %1593, %1594, %1595 : vector<4xf16>
      %1597 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1514, %1596, %1525 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1598 = vector.extract %1591[0] : f32 from vector<4xf32>
      %1599 = arith.maxnumf %1583, %1598 : f32
      %1600 = vector.extract %1591[1] : f32 from vector<4xf32>
      %1601 = arith.maxnumf %1599, %1600 : f32
      %1602 = vector.extract %1591[2] : f32 from vector<4xf32>
      %1603 = arith.maxnumf %1601, %1602 : f32
      %1604 = vector.extract %1591[3] : f32 from vector<4xf32>
      %1605 = arith.maxnumf %1603, %1604 : f32
      %1606 = vector.load %view_6[%658] : memref<4096xf16, 3>, vector<4xf16>
      %1607 = vector.load %view_6[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1608 = vector.extract %1606[0] : f16 from vector<4xf16>
      %1609 = vector.extract %1606[1] : f16 from vector<4xf16>
      %1610 = vector.extract %1606[2] : f16 from vector<4xf16>
      %1611 = vector.extract %1606[3] : f16 from vector<4xf16>
      %1612 = vector.from_elements %1608, %1609, %1610, %1611 : vector<4xf16>
      %1613 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1612, %1597 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1614 = vector.extract %1607[0] : f16 from vector<4xf16>
      %1615 = vector.extract %1607[1] : f16 from vector<4xf16>
      %1616 = vector.extract %1607[2] : f16 from vector<4xf16>
      %1617 = vector.extract %1607[3] : f16 from vector<4xf16>
      %1618 = vector.from_elements %1614, %1615, %1616, %1617 : vector<4xf16>
      %1619 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1514, %1618, %1526 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1620 = vector.extract %1613[0] : f32 from vector<4xf32>
      %1621 = arith.maxnumf %1605, %1620 : f32
      %1622 = vector.extract %1613[1] : f32 from vector<4xf32>
      %1623 = arith.maxnumf %1621, %1622 : f32
      %1624 = vector.extract %1613[2] : f32 from vector<4xf32>
      %1625 = arith.maxnumf %1623, %1624 : f32
      %1626 = vector.extract %1613[3] : f32 from vector<4xf32>
      %1627 = arith.maxnumf %1625, %1626 : f32
      %1628 = vector.load %view_6[%678] : memref<4096xf16, 3>, vector<4xf16>
      %1629 = vector.load %view_6[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1630 = vector.extract %1628[0] : f16 from vector<4xf16>
      %1631 = vector.extract %1628[1] : f16 from vector<4xf16>
      %1632 = vector.extract %1628[2] : f16 from vector<4xf16>
      %1633 = vector.extract %1628[3] : f16 from vector<4xf16>
      %1634 = vector.from_elements %1630, %1631, %1632, %1633 : vector<4xf16>
      %1635 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1634, %1619 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1636 = vector.extract %1629[0] : f16 from vector<4xf16>
      %1637 = vector.extract %1629[1] : f16 from vector<4xf16>
      %1638 = vector.extract %1629[2] : f16 from vector<4xf16>
      %1639 = vector.extract %1629[3] : f16 from vector<4xf16>
      %1640 = vector.from_elements %1636, %1637, %1638, %1639 : vector<4xf16>
      %1641 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1514, %1640, %1527 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1642 = vector.extract %1635[0] : f32 from vector<4xf32>
      %1643 = arith.maxnumf %1627, %1642 : f32
      %1644 = vector.extract %1635[1] : f32 from vector<4xf32>
      %1645 = arith.maxnumf %1643, %1644 : f32
      %1646 = vector.extract %1635[2] : f32 from vector<4xf32>
      %1647 = arith.maxnumf %1645, %1646 : f32
      %1648 = vector.extract %1635[3] : f32 from vector<4xf32>
      %1649 = arith.maxnumf %1647, %1648 : f32
      %1650 = vector.load %view_6[%698] : memref<4096xf16, 3>, vector<4xf16>
      %1651 = vector.load %view_6[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1652 = vector.extract %1650[0] : f16 from vector<4xf16>
      %1653 = vector.extract %1650[1] : f16 from vector<4xf16>
      %1654 = vector.extract %1650[2] : f16 from vector<4xf16>
      %1655 = vector.extract %1650[3] : f16 from vector<4xf16>
      %1656 = vector.from_elements %1652, %1653, %1654, %1655 : vector<4xf16>
      %1657 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1656, %1641 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1658 = vector.extract %1651[0] : f16 from vector<4xf16>
      %1659 = vector.extract %1651[1] : f16 from vector<4xf16>
      %1660 = vector.extract %1651[2] : f16 from vector<4xf16>
      %1661 = vector.extract %1651[3] : f16 from vector<4xf16>
      %1662 = vector.from_elements %1658, %1659, %1660, %1661 : vector<4xf16>
      %1663 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1514, %1662, %1528 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1664 = vector.extract %1657[0] : f32 from vector<4xf32>
      %1665 = arith.maxnumf %1649, %1664 : f32
      %1666 = vector.extract %1657[1] : f32 from vector<4xf32>
      %1667 = arith.maxnumf %1665, %1666 : f32
      %1668 = vector.extract %1657[2] : f32 from vector<4xf32>
      %1669 = arith.maxnumf %1667, %1668 : f32
      %1670 = vector.extract %1657[3] : f32 from vector<4xf32>
      %1671 = arith.maxnumf %1669, %1670 : f32
      %1672 = vector.load %view_6[%718] : memref<4096xf16, 3>, vector<4xf16>
      %1673 = vector.load %view_6[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1674 = vector.extract %1672[0] : f16 from vector<4xf16>
      %1675 = vector.extract %1672[1] : f16 from vector<4xf16>
      %1676 = vector.extract %1672[2] : f16 from vector<4xf16>
      %1677 = vector.extract %1672[3] : f16 from vector<4xf16>
      %1678 = vector.from_elements %1674, %1675, %1676, %1677 : vector<4xf16>
      %1679 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1678, %1663 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1680 = vector.extract %1673[0] : f16 from vector<4xf16>
      %1681 = vector.extract %1673[1] : f16 from vector<4xf16>
      %1682 = vector.extract %1673[2] : f16 from vector<4xf16>
      %1683 = vector.extract %1673[3] : f16 from vector<4xf16>
      %1684 = vector.from_elements %1680, %1681, %1682, %1683 : vector<4xf16>
      %1685 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1514, %1684, %1529 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1686 = vector.extract %1679[0] : f32 from vector<4xf32>
      %1687 = arith.maxnumf %1671, %1686 : f32
      %1688 = vector.extract %1679[1] : f32 from vector<4xf32>
      %1689 = arith.maxnumf %1687, %1688 : f32
      %1690 = vector.extract %1679[2] : f32 from vector<4xf32>
      %1691 = arith.maxnumf %1689, %1690 : f32
      %1692 = vector.extract %1679[3] : f32 from vector<4xf32>
      %1693 = arith.maxnumf %1691, %1692 : f32
      %1694 = vector.load %view_6[%738] : memref<4096xf16, 3>, vector<4xf16>
      %1695 = vector.load %view_6[%586] : memref<4096xf16, 3>, vector<4xf16>
      %1696 = vector.extract %1694[0] : f16 from vector<4xf16>
      %1697 = vector.extract %1694[1] : f16 from vector<4xf16>
      %1698 = vector.extract %1694[2] : f16 from vector<4xf16>
      %1699 = vector.extract %1694[3] : f16 from vector<4xf16>
      %1700 = vector.from_elements %1696, %1697, %1698, %1699 : vector<4xf16>
      %1701 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1516, %1700, %1685 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1702 = vector.extract %1695[0] : f16 from vector<4xf16>
      %1703 = vector.extract %1695[1] : f16 from vector<4xf16>
      %1704 = vector.extract %1695[2] : f16 from vector<4xf16>
      %1705 = vector.extract %1695[3] : f16 from vector<4xf16>
      %1706 = vector.from_elements %1702, %1703, %1704, %1705 : vector<4xf16>
      %1707 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1530, %1706, %1531 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1708 = vector.extract %1701[0] : f32 from vector<4xf32>
      %1709 = arith.maxnumf %1693, %1708 : f32
      %1710 = vector.extract %1701[1] : f32 from vector<4xf32>
      %1711 = arith.maxnumf %1709, %1710 : f32
      %1712 = vector.extract %1701[2] : f32 from vector<4xf32>
      %1713 = arith.maxnumf %1711, %1712 : f32
      %1714 = vector.extract %1701[3] : f32 from vector<4xf32>
      %1715 = arith.maxnumf %1713, %1714 : f32
      %1716 = vector.load %view_6[%598] : memref<4096xf16, 3>, vector<4xf16>
      %1717 = vector.load %view_6[%600] : memref<4096xf16, 3>, vector<4xf16>
      %1718 = vector.extract %1716[0] : f16 from vector<4xf16>
      %1719 = vector.extract %1716[1] : f16 from vector<4xf16>
      %1720 = vector.extract %1716[2] : f16 from vector<4xf16>
      %1721 = vector.extract %1716[3] : f16 from vector<4xf16>
      %1722 = vector.from_elements %1718, %1719, %1720, %1721 : vector<4xf16>
      %1723 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1722, %1707 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1724 = vector.extract %1717[0] : f16 from vector<4xf16>
      %1725 = vector.extract %1717[1] : f16 from vector<4xf16>
      %1726 = vector.extract %1717[2] : f16 from vector<4xf16>
      %1727 = vector.extract %1717[3] : f16 from vector<4xf16>
      %1728 = vector.from_elements %1724, %1725, %1726, %1727 : vector<4xf16>
      %1729 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1530, %1728, %1533 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1730 = vector.extract %1723[0] : f32 from vector<4xf32>
      %1731 = vector.extract %1723[1] : f32 from vector<4xf32>
      %1732 = vector.extract %1723[2] : f32 from vector<4xf32>
      %1733 = vector.extract %1723[3] : f32 from vector<4xf32>
      %1734 = vector.load %view_6[%618] : memref<4096xf16, 3>, vector<4xf16>
      %1735 = vector.load %view_6[%620] : memref<4096xf16, 3>, vector<4xf16>
      %1736 = vector.extract %1734[0] : f16 from vector<4xf16>
      %1737 = vector.extract %1734[1] : f16 from vector<4xf16>
      %1738 = vector.extract %1734[2] : f16 from vector<4xf16>
      %1739 = vector.extract %1734[3] : f16 from vector<4xf16>
      %1740 = vector.from_elements %1736, %1737, %1738, %1739 : vector<4xf16>
      %1741 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1740, %1729 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1742 = vector.extract %1735[0] : f16 from vector<4xf16>
      %1743 = vector.extract %1735[1] : f16 from vector<4xf16>
      %1744 = vector.extract %1735[2] : f16 from vector<4xf16>
      %1745 = vector.extract %1735[3] : f16 from vector<4xf16>
      %1746 = vector.from_elements %1742, %1743, %1744, %1745 : vector<4xf16>
      %1747 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1530, %1746, %1534 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1748 = vector.extract %1741[0] : f32 from vector<4xf32>
      %1749 = vector.extract %1741[1] : f32 from vector<4xf32>
      %1750 = vector.extract %1741[2] : f32 from vector<4xf32>
      %1751 = vector.extract %1741[3] : f32 from vector<4xf32>
      %1752 = vector.load %view_6[%638] : memref<4096xf16, 3>, vector<4xf16>
      %1753 = vector.load %view_6[%640] : memref<4096xf16, 3>, vector<4xf16>
      %1754 = vector.extract %1752[0] : f16 from vector<4xf16>
      %1755 = vector.extract %1752[1] : f16 from vector<4xf16>
      %1756 = vector.extract %1752[2] : f16 from vector<4xf16>
      %1757 = vector.extract %1752[3] : f16 from vector<4xf16>
      %1758 = vector.from_elements %1754, %1755, %1756, %1757 : vector<4xf16>
      %1759 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1758, %1747 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1760 = vector.extract %1753[0] : f16 from vector<4xf16>
      %1761 = vector.extract %1753[1] : f16 from vector<4xf16>
      %1762 = vector.extract %1753[2] : f16 from vector<4xf16>
      %1763 = vector.extract %1753[3] : f16 from vector<4xf16>
      %1764 = vector.from_elements %1760, %1761, %1762, %1763 : vector<4xf16>
      %1765 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1530, %1764, %1535 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1766 = vector.extract %1759[0] : f32 from vector<4xf32>
      %1767 = vector.extract %1759[1] : f32 from vector<4xf32>
      %1768 = vector.extract %1759[2] : f32 from vector<4xf32>
      %1769 = vector.extract %1759[3] : f32 from vector<4xf32>
      %1770 = vector.load %view_6[%658] : memref<4096xf16, 3>, vector<4xf16>
      %1771 = vector.load %view_6[%660] : memref<4096xf16, 3>, vector<4xf16>
      %1772 = vector.extract %1770[0] : f16 from vector<4xf16>
      %1773 = vector.extract %1770[1] : f16 from vector<4xf16>
      %1774 = vector.extract %1770[2] : f16 from vector<4xf16>
      %1775 = vector.extract %1770[3] : f16 from vector<4xf16>
      %1776 = vector.from_elements %1772, %1773, %1774, %1775 : vector<4xf16>
      %1777 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1776, %1765 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1778 = vector.extract %1771[0] : f16 from vector<4xf16>
      %1779 = vector.extract %1771[1] : f16 from vector<4xf16>
      %1780 = vector.extract %1771[2] : f16 from vector<4xf16>
      %1781 = vector.extract %1771[3] : f16 from vector<4xf16>
      %1782 = vector.from_elements %1778, %1779, %1780, %1781 : vector<4xf16>
      %1783 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1530, %1782, %1536 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1784 = vector.extract %1777[0] : f32 from vector<4xf32>
      %1785 = vector.extract %1777[1] : f32 from vector<4xf32>
      %1786 = vector.extract %1777[2] : f32 from vector<4xf32>
      %1787 = vector.extract %1777[3] : f32 from vector<4xf32>
      %1788 = vector.load %view_6[%678] : memref<4096xf16, 3>, vector<4xf16>
      %1789 = vector.load %view_6[%680] : memref<4096xf16, 3>, vector<4xf16>
      %1790 = vector.extract %1788[0] : f16 from vector<4xf16>
      %1791 = vector.extract %1788[1] : f16 from vector<4xf16>
      %1792 = vector.extract %1788[2] : f16 from vector<4xf16>
      %1793 = vector.extract %1788[3] : f16 from vector<4xf16>
      %1794 = vector.from_elements %1790, %1791, %1792, %1793 : vector<4xf16>
      %1795 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1794, %1783 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1796 = vector.extract %1789[0] : f16 from vector<4xf16>
      %1797 = vector.extract %1789[1] : f16 from vector<4xf16>
      %1798 = vector.extract %1789[2] : f16 from vector<4xf16>
      %1799 = vector.extract %1789[3] : f16 from vector<4xf16>
      %1800 = vector.from_elements %1796, %1797, %1798, %1799 : vector<4xf16>
      %1801 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1530, %1800, %1537 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1802 = vector.extract %1795[0] : f32 from vector<4xf32>
      %1803 = vector.extract %1795[1] : f32 from vector<4xf32>
      %1804 = vector.extract %1795[2] : f32 from vector<4xf32>
      %1805 = vector.extract %1795[3] : f32 from vector<4xf32>
      %1806 = vector.load %view_6[%698] : memref<4096xf16, 3>, vector<4xf16>
      %1807 = vector.load %view_6[%700] : memref<4096xf16, 3>, vector<4xf16>
      %1808 = vector.extract %1806[0] : f16 from vector<4xf16>
      %1809 = vector.extract %1806[1] : f16 from vector<4xf16>
      %1810 = vector.extract %1806[2] : f16 from vector<4xf16>
      %1811 = vector.extract %1806[3] : f16 from vector<4xf16>
      %1812 = vector.from_elements %1808, %1809, %1810, %1811 : vector<4xf16>
      %1813 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1812, %1801 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1814 = vector.extract %1807[0] : f16 from vector<4xf16>
      %1815 = vector.extract %1807[1] : f16 from vector<4xf16>
      %1816 = vector.extract %1807[2] : f16 from vector<4xf16>
      %1817 = vector.extract %1807[3] : f16 from vector<4xf16>
      %1818 = vector.from_elements %1814, %1815, %1816, %1817 : vector<4xf16>
      %1819 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1530, %1818, %1538 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1820 = vector.extract %1813[0] : f32 from vector<4xf32>
      %1821 = vector.extract %1813[1] : f32 from vector<4xf32>
      %1822 = vector.extract %1813[2] : f32 from vector<4xf32>
      %1823 = vector.extract %1813[3] : f32 from vector<4xf32>
      %1824 = vector.load %view_6[%718] : memref<4096xf16, 3>, vector<4xf16>
      %1825 = vector.load %view_6[%720] : memref<4096xf16, 3>, vector<4xf16>
      %1826 = vector.extract %1824[0] : f16 from vector<4xf16>
      %1827 = vector.extract %1824[1] : f16 from vector<4xf16>
      %1828 = vector.extract %1824[2] : f16 from vector<4xf16>
      %1829 = vector.extract %1824[3] : f16 from vector<4xf16>
      %1830 = vector.from_elements %1826, %1827, %1828, %1829 : vector<4xf16>
      %1831 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1830, %1819 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1832 = vector.extract %1825[0] : f16 from vector<4xf16>
      %1833 = vector.extract %1825[1] : f16 from vector<4xf16>
      %1834 = vector.extract %1825[2] : f16 from vector<4xf16>
      %1835 = vector.extract %1825[3] : f16 from vector<4xf16>
      %1836 = vector.from_elements %1832, %1833, %1834, %1835 : vector<4xf16>
      %1837 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1530, %1836, %1539 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1838 = vector.extract %1831[0] : f32 from vector<4xf32>
      %1839 = vector.extract %1831[1] : f32 from vector<4xf32>
      %1840 = vector.extract %1831[2] : f32 from vector<4xf32>
      %1841 = vector.extract %1831[3] : f32 from vector<4xf32>
      %1842 = vector.load %view_6[%738] : memref<4096xf16, 3>, vector<4xf16>
      %1843 = vector.extract %1842[0] : f16 from vector<4xf16>
      %1844 = vector.extract %1842[1] : f16 from vector<4xf16>
      %1845 = vector.extract %1842[2] : f16 from vector<4xf16>
      %1846 = vector.extract %1842[3] : f16 from vector<4xf16>
      %1847 = vector.from_elements %1843, %1844, %1845, %1846 : vector<4xf16>
      %1848 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1532, %1847, %1837 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1849 = vector.extract %1848[0] : f32 from vector<4xf32>
      %1850 = vector.extract %1848[1] : f32 from vector<4xf32>
      %1851 = vector.extract %1848[2] : f32 from vector<4xf32>
      %1852 = vector.extract %1848[3] : f32 from vector<4xf32>
      %shuffleResult_141, %valid_142 = gpu.shuffle xor %1715, %c16_i32, %c64_i32 : f32
      %1853 = arith.maxnumf %1715, %shuffleResult_141 : f32
      %shuffleResult_143, %valid_144 = gpu.shuffle xor %1853, %c32_i32, %c64_i32 : f32
      %1854 = arith.maxnumf %1853, %shuffleResult_143 : f32
      %1855 = arith.maxnumf %1730, %cst_2 : f32
      %1856 = arith.maxnumf %1855, %1731 : f32
      %1857 = arith.maxnumf %1856, %1732 : f32
      %1858 = arith.maxnumf %1857, %1733 : f32
      %1859 = arith.maxnumf %1858, %1748 : f32
      %1860 = arith.maxnumf %1859, %1749 : f32
      %1861 = arith.maxnumf %1860, %1750 : f32
      %1862 = arith.maxnumf %1861, %1751 : f32
      %1863 = arith.maxnumf %1862, %1766 : f32
      %1864 = arith.maxnumf %1863, %1767 : f32
      %1865 = arith.maxnumf %1864, %1768 : f32
      %1866 = arith.maxnumf %1865, %1769 : f32
      %1867 = arith.maxnumf %1866, %1784 : f32
      %1868 = arith.maxnumf %1867, %1785 : f32
      %1869 = arith.maxnumf %1868, %1786 : f32
      %1870 = arith.maxnumf %1869, %1787 : f32
      %1871 = arith.maxnumf %1870, %1802 : f32
      %1872 = arith.maxnumf %1871, %1803 : f32
      %1873 = arith.maxnumf %1872, %1804 : f32
      %1874 = arith.maxnumf %1873, %1805 : f32
      %1875 = arith.maxnumf %1874, %1820 : f32
      %1876 = arith.maxnumf %1875, %1821 : f32
      %1877 = arith.maxnumf %1876, %1822 : f32
      %1878 = arith.maxnumf %1877, %1823 : f32
      %1879 = arith.maxnumf %1878, %1838 : f32
      %1880 = arith.maxnumf %1879, %1839 : f32
      %1881 = arith.maxnumf %1880, %1840 : f32
      %1882 = arith.maxnumf %1881, %1841 : f32
      %1883 = arith.maxnumf %1882, %1849 : f32
      %1884 = arith.maxnumf %1883, %1850 : f32
      %1885 = arith.maxnumf %1884, %1851 : f32
      %1886 = arith.maxnumf %1885, %1852 : f32
      %shuffleResult_145, %valid_146 = gpu.shuffle xor %1886, %c16_i32, %c64_i32 : f32
      %1887 = arith.maxnumf %1886, %shuffleResult_145 : f32
      %shuffleResult_147, %valid_148 = gpu.shuffle xor %1887, %c32_i32, %c64_i32 : f32
      %1888 = arith.maxnumf %1887, %shuffleResult_147 : f32
      %1889 = vector.extract %arg7[0, 0] : f32 from vector<2x1xf32>
      %1890 = vector.broadcast %1854 : f32 to vector<1xf32>
      %1891 = vector.broadcast %1889 : f32 to vector<1xf32>
      %1892 = arith.maxnumf %1890, %1891 {origin = "frisk.max", tiled = true} : vector<1xf32>
      %1893 = vector.extract %1892[0] : f32 from vector<1xf32>
      %1894 = vector.extract %arg7[1, 0] : f32 from vector<2x1xf32>
      %1895 = vector.broadcast %1888 : f32 to vector<1xf32>
      %1896 = vector.broadcast %1894 : f32 to vector<1xf32>
      %1897 = arith.maxnumf %1895, %1896 {origin = "frisk.max", tiled = true} : vector<1xf32>
      %1898 = vector.extract %1897[0] : f32 from vector<1xf32>
      %1899 = vector.from_elements %1554, %1556, %1558, %1560 : vector<4xf32>
      %1900 = vector.broadcast %1893 : f32 to vector<4xf32>
      %1901 = arith.subf %1899, %1900 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1902 = vector.extract %1901[0] : f32 from vector<4xf32>
      %1903 = vector.extract %1901[1] : f32 from vector<4xf32>
      %1904 = vector.extract %1901[2] : f32 from vector<4xf32>
      %1905 = vector.extract %1901[3] : f32 from vector<4xf32>
      %1906 = vector.from_elements %1576, %1578, %1580, %1582 : vector<4xf32>
      %1907 = arith.subf %1906, %1900 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1908 = vector.extract %1907[0] : f32 from vector<4xf32>
      %1909 = vector.extract %1907[1] : f32 from vector<4xf32>
      %1910 = vector.extract %1907[2] : f32 from vector<4xf32>
      %1911 = vector.extract %1907[3] : f32 from vector<4xf32>
      %1912 = vector.from_elements %1598, %1600, %1602, %1604 : vector<4xf32>
      %1913 = arith.subf %1912, %1900 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1914 = vector.extract %1913[0] : f32 from vector<4xf32>
      %1915 = vector.extract %1913[1] : f32 from vector<4xf32>
      %1916 = vector.extract %1913[2] : f32 from vector<4xf32>
      %1917 = vector.extract %1913[3] : f32 from vector<4xf32>
      %1918 = vector.from_elements %1620, %1622, %1624, %1626 : vector<4xf32>
      %1919 = arith.subf %1918, %1900 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1920 = vector.extract %1919[0] : f32 from vector<4xf32>
      %1921 = vector.extract %1919[1] : f32 from vector<4xf32>
      %1922 = vector.extract %1919[2] : f32 from vector<4xf32>
      %1923 = vector.extract %1919[3] : f32 from vector<4xf32>
      %1924 = vector.from_elements %1642, %1644, %1646, %1648 : vector<4xf32>
      %1925 = arith.subf %1924, %1900 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1926 = vector.extract %1925[0] : f32 from vector<4xf32>
      %1927 = vector.extract %1925[1] : f32 from vector<4xf32>
      %1928 = vector.extract %1925[2] : f32 from vector<4xf32>
      %1929 = vector.extract %1925[3] : f32 from vector<4xf32>
      %1930 = vector.from_elements %1664, %1666, %1668, %1670 : vector<4xf32>
      %1931 = arith.subf %1930, %1900 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1932 = vector.extract %1931[0] : f32 from vector<4xf32>
      %1933 = vector.extract %1931[1] : f32 from vector<4xf32>
      %1934 = vector.extract %1931[2] : f32 from vector<4xf32>
      %1935 = vector.extract %1931[3] : f32 from vector<4xf32>
      %1936 = vector.from_elements %1686, %1688, %1690, %1692 : vector<4xf32>
      %1937 = arith.subf %1936, %1900 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1938 = vector.extract %1937[0] : f32 from vector<4xf32>
      %1939 = vector.extract %1937[1] : f32 from vector<4xf32>
      %1940 = vector.extract %1937[2] : f32 from vector<4xf32>
      %1941 = vector.extract %1937[3] : f32 from vector<4xf32>
      %1942 = vector.from_elements %1708, %1710, %1712, %1714 : vector<4xf32>
      %1943 = arith.subf %1942, %1900 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1944 = vector.extract %1943[0] : f32 from vector<4xf32>
      %1945 = vector.extract %1943[1] : f32 from vector<4xf32>
      %1946 = vector.extract %1943[2] : f32 from vector<4xf32>
      %1947 = vector.extract %1943[3] : f32 from vector<4xf32>
      %1948 = vector.from_elements %1730, %1731, %1732, %1733 : vector<4xf32>
      %1949 = vector.broadcast %1898 : f32 to vector<4xf32>
      %1950 = arith.subf %1948, %1949 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1951 = vector.extract %1950[0] : f32 from vector<4xf32>
      %1952 = vector.extract %1950[1] : f32 from vector<4xf32>
      %1953 = vector.extract %1950[2] : f32 from vector<4xf32>
      %1954 = vector.extract %1950[3] : f32 from vector<4xf32>
      %1955 = vector.from_elements %1748, %1749, %1750, %1751 : vector<4xf32>
      %1956 = arith.subf %1955, %1949 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1957 = vector.extract %1956[0] : f32 from vector<4xf32>
      %1958 = vector.extract %1956[1] : f32 from vector<4xf32>
      %1959 = vector.extract %1956[2] : f32 from vector<4xf32>
      %1960 = vector.extract %1956[3] : f32 from vector<4xf32>
      %1961 = vector.from_elements %1766, %1767, %1768, %1769 : vector<4xf32>
      %1962 = arith.subf %1961, %1949 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1963 = vector.extract %1962[0] : f32 from vector<4xf32>
      %1964 = vector.extract %1962[1] : f32 from vector<4xf32>
      %1965 = vector.extract %1962[2] : f32 from vector<4xf32>
      %1966 = vector.extract %1962[3] : f32 from vector<4xf32>
      %1967 = vector.from_elements %1784, %1785, %1786, %1787 : vector<4xf32>
      %1968 = arith.subf %1967, %1949 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1969 = vector.extract %1968[0] : f32 from vector<4xf32>
      %1970 = vector.extract %1968[1] : f32 from vector<4xf32>
      %1971 = vector.extract %1968[2] : f32 from vector<4xf32>
      %1972 = vector.extract %1968[3] : f32 from vector<4xf32>
      %1973 = vector.from_elements %1802, %1803, %1804, %1805 : vector<4xf32>
      %1974 = arith.subf %1973, %1949 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1975 = vector.extract %1974[0] : f32 from vector<4xf32>
      %1976 = vector.extract %1974[1] : f32 from vector<4xf32>
      %1977 = vector.extract %1974[2] : f32 from vector<4xf32>
      %1978 = vector.extract %1974[3] : f32 from vector<4xf32>
      %1979 = vector.from_elements %1820, %1821, %1822, %1823 : vector<4xf32>
      %1980 = arith.subf %1979, %1949 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1981 = vector.extract %1980[0] : f32 from vector<4xf32>
      %1982 = vector.extract %1980[1] : f32 from vector<4xf32>
      %1983 = vector.extract %1980[2] : f32 from vector<4xf32>
      %1984 = vector.extract %1980[3] : f32 from vector<4xf32>
      %1985 = vector.from_elements %1838, %1839, %1840, %1841 : vector<4xf32>
      %1986 = arith.subf %1985, %1949 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1987 = vector.extract %1986[0] : f32 from vector<4xf32>
      %1988 = vector.extract %1986[1] : f32 from vector<4xf32>
      %1989 = vector.extract %1986[2] : f32 from vector<4xf32>
      %1990 = vector.extract %1986[3] : f32 from vector<4xf32>
      %1991 = vector.from_elements %1849, %1850, %1851, %1852 : vector<4xf32>
      %1992 = arith.subf %1991, %1949 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %1993 = vector.extract %1992[0] : f32 from vector<4xf32>
      %1994 = vector.extract %1992[1] : f32 from vector<4xf32>
      %1995 = vector.extract %1992[2] : f32 from vector<4xf32>
      %1996 = vector.extract %1992[3] : f32 from vector<4xf32>
      %1997 = vector.broadcast %1893 : f32 to vector<1xf32>
      %1998 = arith.subf %1891, %1997 {origin = "frisk.sub", tiled = true} : vector<1xf32>
      %1999 = math.exp2 %1998 {tiled = true} : vector<1xf32>
      %2000 = vector.extract %1999[0] : f32 from vector<1xf32>
      %2001 = vector.extract %arg6[0, 0] : f32 from vector<2x1xf32>
      %2002 = vector.broadcast %2001 : f32 to vector<1xf32>
      %2003 = arith.mulf %2002, %1999 {origin = "frisk.mul", tiled = true} : vector<1xf32>
      %2004 = vector.extract %2003[0] : f32 from vector<1xf32>
      %2005 = vector.broadcast %1898 : f32 to vector<1xf32>
      %2006 = arith.subf %1896, %2005 {origin = "frisk.sub", tiled = true} : vector<1xf32>
      %2007 = math.exp2 %2006 {tiled = true} : vector<1xf32>
      %2008 = vector.extract %2007[0] : f32 from vector<1xf32>
      %2009 = vector.extract %arg6[1, 0] : f32 from vector<2x1xf32>
      %2010 = vector.broadcast %2009 : f32 to vector<1xf32>
      %2011 = arith.mulf %2010, %2007 {origin = "frisk.mul", tiled = true} : vector<1xf32>
      %2012 = vector.extract %2011[0] : f32 from vector<1xf32>
      %2013 = vector.from_elements %1902, %1903, %1904, %1905 : vector<4xf32>
      %2014 = math.exp2 %2013 {tiled = true} : vector<4xf32>
      %2015 = vector.extract %2014[0] : f32 from vector<4xf32>
      %2016 = vector.extract %2014[1] : f32 from vector<4xf32>
      %2017 = vector.extract %2014[2] : f32 from vector<4xf32>
      %2018 = vector.extract %2014[3] : f32 from vector<4xf32>
      %2019 = vector.from_elements %1908, %1909, %1910, %1911 : vector<4xf32>
      %2020 = math.exp2 %2019 {tiled = true} : vector<4xf32>
      %2021 = vector.extract %2020[0] : f32 from vector<4xf32>
      %2022 = vector.extract %2020[1] : f32 from vector<4xf32>
      %2023 = vector.extract %2020[2] : f32 from vector<4xf32>
      %2024 = vector.extract %2020[3] : f32 from vector<4xf32>
      %2025 = vector.from_elements %1914, %1915, %1916, %1917 : vector<4xf32>
      %2026 = math.exp2 %2025 {tiled = true} : vector<4xf32>
      %2027 = vector.extract %2026[0] : f32 from vector<4xf32>
      %2028 = vector.extract %2026[1] : f32 from vector<4xf32>
      %2029 = vector.extract %2026[2] : f32 from vector<4xf32>
      %2030 = vector.extract %2026[3] : f32 from vector<4xf32>
      %2031 = vector.from_elements %1920, %1921, %1922, %1923 : vector<4xf32>
      %2032 = math.exp2 %2031 {tiled = true} : vector<4xf32>
      %2033 = vector.extract %2032[0] : f32 from vector<4xf32>
      %2034 = vector.extract %2032[1] : f32 from vector<4xf32>
      %2035 = vector.extract %2032[2] : f32 from vector<4xf32>
      %2036 = vector.extract %2032[3] : f32 from vector<4xf32>
      %2037 = vector.from_elements %1926, %1927, %1928, %1929 : vector<4xf32>
      %2038 = math.exp2 %2037 {tiled = true} : vector<4xf32>
      %2039 = vector.extract %2038[0] : f32 from vector<4xf32>
      %2040 = vector.extract %2038[1] : f32 from vector<4xf32>
      %2041 = vector.extract %2038[2] : f32 from vector<4xf32>
      %2042 = vector.extract %2038[3] : f32 from vector<4xf32>
      %2043 = vector.from_elements %1932, %1933, %1934, %1935 : vector<4xf32>
      %2044 = math.exp2 %2043 {tiled = true} : vector<4xf32>
      %2045 = vector.extract %2044[0] : f32 from vector<4xf32>
      %2046 = vector.extract %2044[1] : f32 from vector<4xf32>
      %2047 = vector.extract %2044[2] : f32 from vector<4xf32>
      %2048 = vector.extract %2044[3] : f32 from vector<4xf32>
      %2049 = vector.from_elements %1938, %1939, %1940, %1941 : vector<4xf32>
      %2050 = math.exp2 %2049 {tiled = true} : vector<4xf32>
      %2051 = vector.extract %2050[0] : f32 from vector<4xf32>
      %2052 = vector.extract %2050[1] : f32 from vector<4xf32>
      %2053 = vector.extract %2050[2] : f32 from vector<4xf32>
      %2054 = vector.extract %2050[3] : f32 from vector<4xf32>
      %2055 = vector.from_elements %1944, %1945, %1946, %1947 : vector<4xf32>
      %2056 = math.exp2 %2055 {tiled = true} : vector<4xf32>
      %2057 = vector.extract %2056[0] : f32 from vector<4xf32>
      %2058 = vector.extract %2056[1] : f32 from vector<4xf32>
      %2059 = vector.extract %2056[2] : f32 from vector<4xf32>
      %2060 = vector.extract %2056[3] : f32 from vector<4xf32>
      %2061 = vector.from_elements %1951, %1952, %1953, %1954 : vector<4xf32>
      %2062 = math.exp2 %2061 {tiled = true} : vector<4xf32>
      %2063 = vector.extract %2062[0] : f32 from vector<4xf32>
      %2064 = vector.extract %2062[1] : f32 from vector<4xf32>
      %2065 = vector.extract %2062[2] : f32 from vector<4xf32>
      %2066 = vector.extract %2062[3] : f32 from vector<4xf32>
      %2067 = vector.from_elements %1957, %1958, %1959, %1960 : vector<4xf32>
      %2068 = math.exp2 %2067 {tiled = true} : vector<4xf32>
      %2069 = vector.extract %2068[0] : f32 from vector<4xf32>
      %2070 = vector.extract %2068[1] : f32 from vector<4xf32>
      %2071 = vector.extract %2068[2] : f32 from vector<4xf32>
      %2072 = vector.extract %2068[3] : f32 from vector<4xf32>
      %2073 = vector.from_elements %1963, %1964, %1965, %1966 : vector<4xf32>
      %2074 = math.exp2 %2073 {tiled = true} : vector<4xf32>
      %2075 = vector.extract %2074[0] : f32 from vector<4xf32>
      %2076 = vector.extract %2074[1] : f32 from vector<4xf32>
      %2077 = vector.extract %2074[2] : f32 from vector<4xf32>
      %2078 = vector.extract %2074[3] : f32 from vector<4xf32>
      %2079 = vector.from_elements %1969, %1970, %1971, %1972 : vector<4xf32>
      %2080 = math.exp2 %2079 {tiled = true} : vector<4xf32>
      %2081 = vector.extract %2080[0] : f32 from vector<4xf32>
      %2082 = vector.extract %2080[1] : f32 from vector<4xf32>
      %2083 = vector.extract %2080[2] : f32 from vector<4xf32>
      %2084 = vector.extract %2080[3] : f32 from vector<4xf32>
      %2085 = vector.from_elements %1975, %1976, %1977, %1978 : vector<4xf32>
      %2086 = math.exp2 %2085 {tiled = true} : vector<4xf32>
      %2087 = vector.extract %2086[0] : f32 from vector<4xf32>
      %2088 = vector.extract %2086[1] : f32 from vector<4xf32>
      %2089 = vector.extract %2086[2] : f32 from vector<4xf32>
      %2090 = vector.extract %2086[3] : f32 from vector<4xf32>
      %2091 = vector.from_elements %1981, %1982, %1983, %1984 : vector<4xf32>
      %2092 = math.exp2 %2091 {tiled = true} : vector<4xf32>
      %2093 = vector.extract %2092[0] : f32 from vector<4xf32>
      %2094 = vector.extract %2092[1] : f32 from vector<4xf32>
      %2095 = vector.extract %2092[2] : f32 from vector<4xf32>
      %2096 = vector.extract %2092[3] : f32 from vector<4xf32>
      %2097 = vector.from_elements %1987, %1988, %1989, %1990 : vector<4xf32>
      %2098 = math.exp2 %2097 {tiled = true} : vector<4xf32>
      %2099 = vector.extract %2098[0] : f32 from vector<4xf32>
      %2100 = vector.extract %2098[1] : f32 from vector<4xf32>
      %2101 = vector.extract %2098[2] : f32 from vector<4xf32>
      %2102 = vector.extract %2098[3] : f32 from vector<4xf32>
      %2103 = vector.from_elements %1993, %1994, %1995, %1996 : vector<4xf32>
      %2104 = math.exp2 %2103 {tiled = true} : vector<4xf32>
      %2105 = vector.extract %2104[0] : f32 from vector<4xf32>
      %2106 = vector.extract %2104[1] : f32 from vector<4xf32>
      %2107 = vector.extract %2104[2] : f32 from vector<4xf32>
      %2108 = vector.extract %2104[3] : f32 from vector<4xf32>
      %2109 = arith.addf %2015, %cst_5 : f32
      %2110 = arith.addf %2109, %2016 : f32
      %2111 = arith.addf %2110, %2017 : f32
      %2112 = arith.addf %2111, %2018 : f32
      %2113 = arith.addf %2112, %2021 : f32
      %2114 = arith.addf %2113, %2022 : f32
      %2115 = arith.addf %2114, %2023 : f32
      %2116 = arith.addf %2115, %2024 : f32
      %2117 = arith.addf %2116, %2027 : f32
      %2118 = arith.addf %2117, %2028 : f32
      %2119 = arith.addf %2118, %2029 : f32
      %2120 = arith.addf %2119, %2030 : f32
      %2121 = arith.addf %2120, %2033 : f32
      %2122 = arith.addf %2121, %2034 : f32
      %2123 = arith.addf %2122, %2035 : f32
      %2124 = arith.addf %2123, %2036 : f32
      %2125 = arith.addf %2124, %2039 : f32
      %2126 = arith.addf %2125, %2040 : f32
      %2127 = arith.addf %2126, %2041 : f32
      %2128 = arith.addf %2127, %2042 : f32
      %2129 = arith.addf %2128, %2045 : f32
      %2130 = arith.addf %2129, %2046 : f32
      %2131 = arith.addf %2130, %2047 : f32
      %2132 = arith.addf %2131, %2048 : f32
      %2133 = arith.addf %2132, %2051 : f32
      %2134 = arith.addf %2133, %2052 : f32
      %2135 = arith.addf %2134, %2053 : f32
      %2136 = arith.addf %2135, %2054 : f32
      %2137 = arith.addf %2136, %2057 : f32
      %2138 = arith.addf %2137, %2058 : f32
      %2139 = arith.addf %2138, %2059 : f32
      %2140 = arith.addf %2139, %2060 : f32
      %shuffleResult_149, %valid_150 = gpu.shuffle xor %2140, %c16_i32, %c64_i32 : f32
      %2141 = arith.addf %2140, %shuffleResult_149 : f32
      %shuffleResult_151, %valid_152 = gpu.shuffle xor %2141, %c32_i32, %c64_i32 : f32
      %2142 = arith.addf %2141, %shuffleResult_151 : f32
      %2143 = arith.addf %2063, %cst_5 : f32
      %2144 = arith.addf %2143, %2064 : f32
      %2145 = arith.addf %2144, %2065 : f32
      %2146 = arith.addf %2145, %2066 : f32
      %2147 = arith.addf %2146, %2069 : f32
      %2148 = arith.addf %2147, %2070 : f32
      %2149 = arith.addf %2148, %2071 : f32
      %2150 = arith.addf %2149, %2072 : f32
      %2151 = arith.addf %2150, %2075 : f32
      %2152 = arith.addf %2151, %2076 : f32
      %2153 = arith.addf %2152, %2077 : f32
      %2154 = arith.addf %2153, %2078 : f32
      %2155 = arith.addf %2154, %2081 : f32
      %2156 = arith.addf %2155, %2082 : f32
      %2157 = arith.addf %2156, %2083 : f32
      %2158 = arith.addf %2157, %2084 : f32
      %2159 = arith.addf %2158, %2087 : f32
      %2160 = arith.addf %2159, %2088 : f32
      %2161 = arith.addf %2160, %2089 : f32
      %2162 = arith.addf %2161, %2090 : f32
      %2163 = arith.addf %2162, %2093 : f32
      %2164 = arith.addf %2163, %2094 : f32
      %2165 = arith.addf %2164, %2095 : f32
      %2166 = arith.addf %2165, %2096 : f32
      %2167 = arith.addf %2166, %2099 : f32
      %2168 = arith.addf %2167, %2100 : f32
      %2169 = arith.addf %2168, %2101 : f32
      %2170 = arith.addf %2169, %2102 : f32
      %2171 = arith.addf %2170, %2105 : f32
      %2172 = arith.addf %2171, %2106 : f32
      %2173 = arith.addf %2172, %2107 : f32
      %2174 = arith.addf %2173, %2108 : f32
      %shuffleResult_153, %valid_154 = gpu.shuffle xor %2174, %c16_i32, %c64_i32 : f32
      %2175 = arith.addf %2174, %shuffleResult_153 : f32
      %shuffleResult_155, %valid_156 = gpu.shuffle xor %2175, %c32_i32, %c64_i32 : f32
      %2176 = arith.addf %2175, %shuffleResult_155 : f32
      %2177 = vector.broadcast %2142 : f32 to vector<1xf32>
      %2178 = vector.broadcast %2004 : f32 to vector<1xf32>
      %2179 = arith.addf %2177, %2178 {origin = "frisk.add", tiled = true} : vector<1xf32>
      %2180 = vector.extract %2179[0] : f32 from vector<1xf32>
      %2181 = vector.broadcast %2176 : f32 to vector<1xf32>
      %2182 = vector.broadcast %2012 : f32 to vector<1xf32>
      %2183 = arith.addf %2181, %2182 {origin = "frisk.add", tiled = true} : vector<1xf32>
      %2184 = vector.extract %2183[0] : f32 from vector<1xf32>
      %2185 = vector.from_elements %2015, %2016, %2017, %2018 : vector<4xf32>
      %2186 = arith.truncf %2185 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2187 = vector.extract %2186[0] : f16 from vector<4xf16>
      %2188 = vector.extract %2186[1] : f16 from vector<4xf16>
      %2189 = vector.extract %2186[2] : f16 from vector<4xf16>
      %2190 = vector.extract %2186[3] : f16 from vector<4xf16>
      %2191 = vector.from_elements %2021, %2022, %2023, %2024 : vector<4xf32>
      %2192 = arith.truncf %2191 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2193 = vector.extract %2192[0] : f16 from vector<4xf16>
      %2194 = vector.extract %2192[1] : f16 from vector<4xf16>
      %2195 = vector.extract %2192[2] : f16 from vector<4xf16>
      %2196 = vector.extract %2192[3] : f16 from vector<4xf16>
      %2197 = vector.from_elements %2027, %2028, %2029, %2030 : vector<4xf32>
      %2198 = arith.truncf %2197 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2199 = vector.extract %2198[0] : f16 from vector<4xf16>
      %2200 = vector.extract %2198[1] : f16 from vector<4xf16>
      %2201 = vector.extract %2198[2] : f16 from vector<4xf16>
      %2202 = vector.extract %2198[3] : f16 from vector<4xf16>
      %2203 = vector.from_elements %2033, %2034, %2035, %2036 : vector<4xf32>
      %2204 = arith.truncf %2203 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2205 = vector.extract %2204[0] : f16 from vector<4xf16>
      %2206 = vector.extract %2204[1] : f16 from vector<4xf16>
      %2207 = vector.extract %2204[2] : f16 from vector<4xf16>
      %2208 = vector.extract %2204[3] : f16 from vector<4xf16>
      %2209 = vector.from_elements %2039, %2040, %2041, %2042 : vector<4xf32>
      %2210 = arith.truncf %2209 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2211 = vector.extract %2210[0] : f16 from vector<4xf16>
      %2212 = vector.extract %2210[1] : f16 from vector<4xf16>
      %2213 = vector.extract %2210[2] : f16 from vector<4xf16>
      %2214 = vector.extract %2210[3] : f16 from vector<4xf16>
      %2215 = vector.from_elements %2045, %2046, %2047, %2048 : vector<4xf32>
      %2216 = arith.truncf %2215 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2217 = vector.extract %2216[0] : f16 from vector<4xf16>
      %2218 = vector.extract %2216[1] : f16 from vector<4xf16>
      %2219 = vector.extract %2216[2] : f16 from vector<4xf16>
      %2220 = vector.extract %2216[3] : f16 from vector<4xf16>
      %2221 = vector.from_elements %2051, %2052, %2053, %2054 : vector<4xf32>
      %2222 = arith.truncf %2221 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2223 = vector.extract %2222[0] : f16 from vector<4xf16>
      %2224 = vector.extract %2222[1] : f16 from vector<4xf16>
      %2225 = vector.extract %2222[2] : f16 from vector<4xf16>
      %2226 = vector.extract %2222[3] : f16 from vector<4xf16>
      %2227 = vector.from_elements %2057, %2058, %2059, %2060 : vector<4xf32>
      %2228 = arith.truncf %2227 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2229 = vector.extract %2228[0] : f16 from vector<4xf16>
      %2230 = vector.extract %2228[1] : f16 from vector<4xf16>
      %2231 = vector.extract %2228[2] : f16 from vector<4xf16>
      %2232 = vector.extract %2228[3] : f16 from vector<4xf16>
      %2233 = vector.from_elements %2063, %2064, %2065, %2066 : vector<4xf32>
      %2234 = arith.truncf %2233 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2235 = vector.extract %2234[0] : f16 from vector<4xf16>
      %2236 = vector.extract %2234[1] : f16 from vector<4xf16>
      %2237 = vector.extract %2234[2] : f16 from vector<4xf16>
      %2238 = vector.extract %2234[3] : f16 from vector<4xf16>
      %2239 = vector.from_elements %2069, %2070, %2071, %2072 : vector<4xf32>
      %2240 = arith.truncf %2239 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2241 = vector.extract %2240[0] : f16 from vector<4xf16>
      %2242 = vector.extract %2240[1] : f16 from vector<4xf16>
      %2243 = vector.extract %2240[2] : f16 from vector<4xf16>
      %2244 = vector.extract %2240[3] : f16 from vector<4xf16>
      %2245 = vector.from_elements %2075, %2076, %2077, %2078 : vector<4xf32>
      %2246 = arith.truncf %2245 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2247 = vector.extract %2246[0] : f16 from vector<4xf16>
      %2248 = vector.extract %2246[1] : f16 from vector<4xf16>
      %2249 = vector.extract %2246[2] : f16 from vector<4xf16>
      %2250 = vector.extract %2246[3] : f16 from vector<4xf16>
      %2251 = vector.from_elements %2081, %2082, %2083, %2084 : vector<4xf32>
      %2252 = arith.truncf %2251 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2253 = vector.extract %2252[0] : f16 from vector<4xf16>
      %2254 = vector.extract %2252[1] : f16 from vector<4xf16>
      %2255 = vector.extract %2252[2] : f16 from vector<4xf16>
      %2256 = vector.extract %2252[3] : f16 from vector<4xf16>
      %2257 = vector.from_elements %2087, %2088, %2089, %2090 : vector<4xf32>
      %2258 = arith.truncf %2257 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2259 = vector.extract %2258[0] : f16 from vector<4xf16>
      %2260 = vector.extract %2258[1] : f16 from vector<4xf16>
      %2261 = vector.extract %2258[2] : f16 from vector<4xf16>
      %2262 = vector.extract %2258[3] : f16 from vector<4xf16>
      %2263 = vector.from_elements %2093, %2094, %2095, %2096 : vector<4xf32>
      %2264 = arith.truncf %2263 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2265 = vector.extract %2264[0] : f16 from vector<4xf16>
      %2266 = vector.extract %2264[1] : f16 from vector<4xf16>
      %2267 = vector.extract %2264[2] : f16 from vector<4xf16>
      %2268 = vector.extract %2264[3] : f16 from vector<4xf16>
      %2269 = vector.from_elements %2099, %2100, %2101, %2102 : vector<4xf32>
      %2270 = arith.truncf %2269 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2271 = vector.extract %2270[0] : f16 from vector<4xf16>
      %2272 = vector.extract %2270[1] : f16 from vector<4xf16>
      %2273 = vector.extract %2270[2] : f16 from vector<4xf16>
      %2274 = vector.extract %2270[3] : f16 from vector<4xf16>
      %2275 = vector.from_elements %2105, %2106, %2107, %2108 : vector<4xf32>
      %2276 = arith.truncf %2275 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2277 = vector.extract %2276[0] : f16 from vector<4xf16>
      %2278 = vector.extract %2276[1] : f16 from vector<4xf16>
      %2279 = vector.extract %2276[2] : f16 from vector<4xf16>
      %2280 = vector.extract %2276[3] : f16 from vector<4xf16>
      %2281 = vector.extract %arg5[0, 0] : f32 from vector<2x8xf32>
      %2282 = vector.extract %arg5[0, 1] : f32 from vector<2x8xf32>
      %2283 = vector.extract %arg5[0, 2] : f32 from vector<2x8xf32>
      %2284 = vector.extract %arg5[0, 3] : f32 from vector<2x8xf32>
      %2285 = vector.from_elements %2281, %2282, %2283, %2284 : vector<4xf32>
      %2286 = vector.broadcast %2000 : f32 to vector<4xf32>
      %2287 = arith.mulf %2285, %2286 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2288 = vector.extract %2287[0] : f32 from vector<4xf32>
      %2289 = vector.extract %2287[1] : f32 from vector<4xf32>
      %2290 = vector.extract %2287[2] : f32 from vector<4xf32>
      %2291 = vector.extract %2287[3] : f32 from vector<4xf32>
      %2292 = vector.extract %arg5[0, 4] : f32 from vector<2x8xf32>
      %2293 = vector.extract %arg5[0, 5] : f32 from vector<2x8xf32>
      %2294 = vector.extract %arg5[0, 6] : f32 from vector<2x8xf32>
      %2295 = vector.extract %arg5[0, 7] : f32 from vector<2x8xf32>
      %2296 = vector.from_elements %2292, %2293, %2294, %2295 : vector<4xf32>
      %2297 = arith.mulf %2296, %2286 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2298 = vector.extract %2297[0] : f32 from vector<4xf32>
      %2299 = vector.extract %2297[1] : f32 from vector<4xf32>
      %2300 = vector.extract %2297[2] : f32 from vector<4xf32>
      %2301 = vector.extract %2297[3] : f32 from vector<4xf32>
      %2302 = vector.extract %arg5[1, 0] : f32 from vector<2x8xf32>
      %2303 = vector.extract %arg5[1, 1] : f32 from vector<2x8xf32>
      %2304 = vector.extract %arg5[1, 2] : f32 from vector<2x8xf32>
      %2305 = vector.extract %arg5[1, 3] : f32 from vector<2x8xf32>
      %2306 = vector.from_elements %2302, %2303, %2304, %2305 : vector<4xf32>
      %2307 = vector.broadcast %2008 : f32 to vector<4xf32>
      %2308 = arith.mulf %2306, %2307 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2309 = vector.extract %2308[0] : f32 from vector<4xf32>
      %2310 = vector.extract %2308[1] : f32 from vector<4xf32>
      %2311 = vector.extract %2308[2] : f32 from vector<4xf32>
      %2312 = vector.extract %2308[3] : f32 from vector<4xf32>
      %2313 = vector.extract %arg5[1, 4] : f32 from vector<2x8xf32>
      %2314 = vector.extract %arg5[1, 5] : f32 from vector<2x8xf32>
      %2315 = vector.extract %arg5[1, 6] : f32 from vector<2x8xf32>
      %2316 = vector.extract %arg5[1, 7] : f32 from vector<2x8xf32>
      %2317 = vector.from_elements %2313, %2314, %2315, %2316 : vector<4xf32>
      %2318 = arith.mulf %2317, %2307 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2319 = vector.extract %2318[0] : f32 from vector<4xf32>
      %2320 = vector.extract %2318[1] : f32 from vector<4xf32>
      %2321 = vector.extract %2318[2] : f32 from vector<4xf32>
      %2322 = vector.extract %2318[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %1512 : !frisk.async_task
      %2323 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>()[%thread_id_x]
      %2324 = frisk.read_matrix_from_shm %view[%2323] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2325 = vector.bitcast %2324 : vector<4xi32> to vector<8xf16>
      %2326 = vector.extract %2325[0] : f16 from vector<8xf16>
      %2327 = vector.extract %2325[%199] : f16 from vector<8xf16>
      %2328 = vector.extract %2325[1] : f16 from vector<8xf16>
      %2329 = vector.extract %2325[%200] : f16 from vector<8xf16>
      %2330 = vector.extract %2325[2] : f16 from vector<8xf16>
      %2331 = vector.extract %2325[%201] : f16 from vector<8xf16>
      %2332 = vector.extract %2325[3] : f16 from vector<8xf16>
      %2333 = vector.extract %2325[%202] : f16 from vector<8xf16>
      %2334 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 16) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 128)>()[%thread_id_x]
      %2335 = frisk.read_matrix_from_shm %view[%2334] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2336 = vector.bitcast %2335 : vector<4xi32> to vector<8xf16>
      %2337 = vector.extract %2336[0] : f16 from vector<8xf16>
      %2338 = vector.extract %2336[%199] : f16 from vector<8xf16>
      %2339 = vector.extract %2336[1] : f16 from vector<8xf16>
      %2340 = vector.extract %2336[%200] : f16 from vector<8xf16>
      %2341 = vector.extract %2336[2] : f16 from vector<8xf16>
      %2342 = vector.extract %2336[%201] : f16 from vector<8xf16>
      %2343 = vector.extract %2336[3] : f16 from vector<8xf16>
      %2344 = vector.extract %2336[%202] : f16 from vector<8xf16>
      %2345 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 32) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 256)>()[%thread_id_x]
      %2346 = frisk.read_matrix_from_shm %view[%2345] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2347 = vector.bitcast %2346 : vector<4xi32> to vector<8xf16>
      %2348 = vector.extract %2347[0] : f16 from vector<8xf16>
      %2349 = vector.extract %2347[%199] : f16 from vector<8xf16>
      %2350 = vector.extract %2347[1] : f16 from vector<8xf16>
      %2351 = vector.extract %2347[%200] : f16 from vector<8xf16>
      %2352 = vector.extract %2347[2] : f16 from vector<8xf16>
      %2353 = vector.extract %2347[%201] : f16 from vector<8xf16>
      %2354 = vector.extract %2347[3] : f16 from vector<8xf16>
      %2355 = vector.extract %2347[%202] : f16 from vector<8xf16>
      %2356 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 48) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 384)>()[%thread_id_x]
      %2357 = frisk.read_matrix_from_shm %view[%2356] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2358 = vector.bitcast %2357 : vector<4xi32> to vector<8xf16>
      %2359 = vector.extract %2358[0] : f16 from vector<8xf16>
      %2360 = vector.extract %2358[%199] : f16 from vector<8xf16>
      %2361 = vector.extract %2358[1] : f16 from vector<8xf16>
      %2362 = vector.extract %2358[%200] : f16 from vector<8xf16>
      %2363 = vector.extract %2358[2] : f16 from vector<8xf16>
      %2364 = vector.extract %2358[%201] : f16 from vector<8xf16>
      %2365 = vector.extract %2358[3] : f16 from vector<8xf16>
      %2366 = vector.extract %2358[%202] : f16 from vector<8xf16>
      %2367 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 2048)>()[%thread_id_x]
      %2368 = frisk.read_matrix_from_shm %view[%2367] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2369 = vector.bitcast %2368 : vector<4xi32> to vector<8xf16>
      %2370 = vector.extract %2369[0] : f16 from vector<8xf16>
      %2371 = vector.extract %2369[%199] : f16 from vector<8xf16>
      %2372 = vector.extract %2369[1] : f16 from vector<8xf16>
      %2373 = vector.extract %2369[%200] : f16 from vector<8xf16>
      %2374 = vector.extract %2369[2] : f16 from vector<8xf16>
      %2375 = vector.extract %2369[%201] : f16 from vector<8xf16>
      %2376 = vector.extract %2369[3] : f16 from vector<8xf16>
      %2377 = vector.extract %2369[%202] : f16 from vector<8xf16>
      %2378 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 80) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 128)>()[%thread_id_x]
      %2379 = frisk.read_matrix_from_shm %view[%2378] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2380 = vector.bitcast %2379 : vector<4xi32> to vector<8xf16>
      %2381 = vector.extract %2380[0] : f16 from vector<8xf16>
      %2382 = vector.extract %2380[%199] : f16 from vector<8xf16>
      %2383 = vector.extract %2380[1] : f16 from vector<8xf16>
      %2384 = vector.extract %2380[%200] : f16 from vector<8xf16>
      %2385 = vector.extract %2380[2] : f16 from vector<8xf16>
      %2386 = vector.extract %2380[%201] : f16 from vector<8xf16>
      %2387 = vector.extract %2380[3] : f16 from vector<8xf16>
      %2388 = vector.extract %2380[%202] : f16 from vector<8xf16>
      %2389 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 96) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 256)>()[%thread_id_x]
      %2390 = frisk.read_matrix_from_shm %view[%2389] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2391 = vector.bitcast %2390 : vector<4xi32> to vector<8xf16>
      %2392 = vector.extract %2391[0] : f16 from vector<8xf16>
      %2393 = vector.extract %2391[%199] : f16 from vector<8xf16>
      %2394 = vector.extract %2391[1] : f16 from vector<8xf16>
      %2395 = vector.extract %2391[%200] : f16 from vector<8xf16>
      %2396 = vector.extract %2391[2] : f16 from vector<8xf16>
      %2397 = vector.extract %2391[%201] : f16 from vector<8xf16>
      %2398 = vector.extract %2391[3] : f16 from vector<8xf16>
      %2399 = vector.extract %2391[%202] : f16 from vector<8xf16>
      %2400 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 112) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 384)>()[%thread_id_x]
      %2401 = frisk.read_matrix_from_shm %view[%2400] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2402 = vector.bitcast %2401 : vector<4xi32> to vector<8xf16>
      %2403 = vector.extract %2402[0] : f16 from vector<8xf16>
      %2404 = vector.extract %2402[%199] : f16 from vector<8xf16>
      %2405 = vector.extract %2402[1] : f16 from vector<8xf16>
      %2406 = vector.extract %2402[%200] : f16 from vector<8xf16>
      %2407 = vector.extract %2402[2] : f16 from vector<8xf16>
      %2408 = vector.extract %2402[%201] : f16 from vector<8xf16>
      %2409 = vector.extract %2402[3] : f16 from vector<8xf16>
      %2410 = vector.extract %2402[%202] : f16 from vector<8xf16>
      %subview_157 = memref.subview %arg2[0, %198, %583, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      %2411 = frisk.async_copy_gm_to_shm %subview_157 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %2412 = vector.from_elements %2187, %2188, %2189, %2190 : vector<4xf16>
      %2413 = vector.from_elements %2326, %2328, %2330, %2332 : vector<4xf16>
      %2414 = vector.from_elements %2288, %2289, %2290, %2291 : vector<4xf32>
      %2415 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2412, %2413, %2414 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2416 = vector.from_elements %2193, %2194, %2195, %2196 : vector<4xf16>
      %2417 = vector.from_elements %2337, %2339, %2341, %2343 : vector<4xf16>
      %2418 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2416, %2417, %2415 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2419 = vector.from_elements %2199, %2200, %2201, %2202 : vector<4xf16>
      %2420 = vector.from_elements %2348, %2350, %2352, %2354 : vector<4xf16>
      %2421 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2419, %2420, %2418 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2422 = vector.from_elements %2205, %2206, %2207, %2208 : vector<4xf16>
      %2423 = vector.from_elements %2359, %2361, %2363, %2365 : vector<4xf16>
      %2424 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2422, %2423, %2421 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2425 = vector.from_elements %2211, %2212, %2213, %2214 : vector<4xf16>
      %2426 = vector.from_elements %2370, %2372, %2374, %2376 : vector<4xf16>
      %2427 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2425, %2426, %2424 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2428 = vector.from_elements %2217, %2218, %2219, %2220 : vector<4xf16>
      %2429 = vector.from_elements %2381, %2383, %2385, %2387 : vector<4xf16>
      %2430 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2428, %2429, %2427 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2431 = vector.from_elements %2223, %2224, %2225, %2226 : vector<4xf16>
      %2432 = vector.from_elements %2392, %2394, %2396, %2398 : vector<4xf16>
      %2433 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2431, %2432, %2430 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2434 = vector.from_elements %2229, %2230, %2231, %2232 : vector<4xf16>
      %2435 = vector.from_elements %2403, %2405, %2407, %2409 : vector<4xf16>
      %2436 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2434, %2435, %2433 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2437 = vector.extract %2436[0] : f32 from vector<4xf32>
      %2438 = vector.extract %2436[1] : f32 from vector<4xf32>
      %2439 = vector.extract %2436[2] : f32 from vector<4xf32>
      %2440 = vector.extract %2436[3] : f32 from vector<4xf32>
      %2441 = vector.from_elements %2327, %2329, %2331, %2333 : vector<4xf16>
      %2442 = vector.from_elements %2298, %2299, %2300, %2301 : vector<4xf32>
      %2443 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2412, %2441, %2442 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2444 = vector.from_elements %2338, %2340, %2342, %2344 : vector<4xf16>
      %2445 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2416, %2444, %2443 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2446 = vector.from_elements %2349, %2351, %2353, %2355 : vector<4xf16>
      %2447 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2419, %2446, %2445 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2448 = vector.from_elements %2360, %2362, %2364, %2366 : vector<4xf16>
      %2449 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2422, %2448, %2447 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2450 = vector.from_elements %2371, %2373, %2375, %2377 : vector<4xf16>
      %2451 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2425, %2450, %2449 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2452 = vector.from_elements %2382, %2384, %2386, %2388 : vector<4xf16>
      %2453 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2428, %2452, %2451 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2454 = vector.from_elements %2393, %2395, %2397, %2399 : vector<4xf16>
      %2455 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2431, %2454, %2453 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2456 = vector.from_elements %2404, %2406, %2408, %2410 : vector<4xf16>
      %2457 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2434, %2456, %2455 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2458 = vector.extract %2457[0] : f32 from vector<4xf32>
      %2459 = vector.extract %2457[1] : f32 from vector<4xf32>
      %2460 = vector.extract %2457[2] : f32 from vector<4xf32>
      %2461 = vector.extract %2457[3] : f32 from vector<4xf32>
      %2462 = vector.from_elements %2235, %2236, %2237, %2238 : vector<4xf16>
      %2463 = vector.from_elements %2309, %2310, %2311, %2312 : vector<4xf32>
      %2464 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2462, %2413, %2463 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2465 = vector.from_elements %2241, %2242, %2243, %2244 : vector<4xf16>
      %2466 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2465, %2417, %2464 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2467 = vector.from_elements %2247, %2248, %2249, %2250 : vector<4xf16>
      %2468 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2420, %2466 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2469 = vector.from_elements %2253, %2254, %2255, %2256 : vector<4xf16>
      %2470 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2423, %2468 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2471 = vector.from_elements %2259, %2260, %2261, %2262 : vector<4xf16>
      %2472 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2426, %2470 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2473 = vector.from_elements %2265, %2266, %2267, %2268 : vector<4xf16>
      %2474 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2429, %2472 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2475 = vector.from_elements %2271, %2272, %2273, %2274 : vector<4xf16>
      %2476 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2432, %2474 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2477 = vector.from_elements %2277, %2278, %2279, %2280 : vector<4xf16>
      %2478 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2435, %2476 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2479 = vector.extract %2478[0] : f32 from vector<4xf32>
      %2480 = vector.extract %2478[1] : f32 from vector<4xf32>
      %2481 = vector.extract %2478[2] : f32 from vector<4xf32>
      %2482 = vector.extract %2478[3] : f32 from vector<4xf32>
      %2483 = vector.from_elements %2319, %2320, %2321, %2322 : vector<4xf32>
      %2484 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2462, %2441, %2483 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2485 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2465, %2444, %2484 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2486 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2446, %2485 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2487 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2448, %2486 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2488 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2450, %2487 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2489 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2452, %2488 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2490 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2454, %2489 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2491 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2456, %2490 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2492 = vector.extract %2491[0] : f32 from vector<4xf32>
      %2493 = vector.extract %2491[1] : f32 from vector<4xf32>
      %2494 = vector.extract %2491[2] : f32 from vector<4xf32>
      %2495 = vector.extract %2491[3] : f32 from vector<4xf32>
      %2496 = vector.extract %arg8[0, 0] : f32 from vector<2x8xf32>
      %2497 = vector.extract %arg8[0, 1] : f32 from vector<2x8xf32>
      %2498 = vector.extract %arg8[0, 2] : f32 from vector<2x8xf32>
      %2499 = vector.extract %arg8[0, 3] : f32 from vector<2x8xf32>
      %2500 = vector.from_elements %2496, %2497, %2498, %2499 : vector<4xf32>
      %2501 = arith.mulf %2500, %2286 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2502 = vector.extract %2501[0] : f32 from vector<4xf32>
      %2503 = vector.extract %2501[1] : f32 from vector<4xf32>
      %2504 = vector.extract %2501[2] : f32 from vector<4xf32>
      %2505 = vector.extract %2501[3] : f32 from vector<4xf32>
      %2506 = vector.extract %arg8[0, 4] : f32 from vector<2x8xf32>
      %2507 = vector.extract %arg8[0, 5] : f32 from vector<2x8xf32>
      %2508 = vector.extract %arg8[0, 6] : f32 from vector<2x8xf32>
      %2509 = vector.extract %arg8[0, 7] : f32 from vector<2x8xf32>
      %2510 = vector.from_elements %2506, %2507, %2508, %2509 : vector<4xf32>
      %2511 = arith.mulf %2510, %2286 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2512 = vector.extract %2511[0] : f32 from vector<4xf32>
      %2513 = vector.extract %2511[1] : f32 from vector<4xf32>
      %2514 = vector.extract %2511[2] : f32 from vector<4xf32>
      %2515 = vector.extract %2511[3] : f32 from vector<4xf32>
      %2516 = vector.extract %arg8[1, 0] : f32 from vector<2x8xf32>
      %2517 = vector.extract %arg8[1, 1] : f32 from vector<2x8xf32>
      %2518 = vector.extract %arg8[1, 2] : f32 from vector<2x8xf32>
      %2519 = vector.extract %arg8[1, 3] : f32 from vector<2x8xf32>
      %2520 = vector.from_elements %2516, %2517, %2518, %2519 : vector<4xf32>
      %2521 = arith.mulf %2520, %2307 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2522 = vector.extract %2521[0] : f32 from vector<4xf32>
      %2523 = vector.extract %2521[1] : f32 from vector<4xf32>
      %2524 = vector.extract %2521[2] : f32 from vector<4xf32>
      %2525 = vector.extract %2521[3] : f32 from vector<4xf32>
      %2526 = vector.extract %arg8[1, 4] : f32 from vector<2x8xf32>
      %2527 = vector.extract %arg8[1, 5] : f32 from vector<2x8xf32>
      %2528 = vector.extract %arg8[1, 6] : f32 from vector<2x8xf32>
      %2529 = vector.extract %arg8[1, 7] : f32 from vector<2x8xf32>
      %2530 = vector.from_elements %2526, %2527, %2528, %2529 : vector<4xf32>
      %2531 = arith.mulf %2530, %2307 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2532 = vector.extract %2531[0] : f32 from vector<4xf32>
      %2533 = vector.extract %2531[1] : f32 from vector<4xf32>
      %2534 = vector.extract %2531[2] : f32 from vector<4xf32>
      %2535 = vector.extract %2531[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %2411 : !frisk.async_task
      %2536 = frisk.read_matrix_from_shm %view[%2323] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2537 = vector.bitcast %2536 : vector<4xi32> to vector<8xf16>
      %2538 = vector.extract %2537[0] : f16 from vector<8xf16>
      %2539 = vector.extract %2537[%199] : f16 from vector<8xf16>
      %2540 = vector.extract %2537[1] : f16 from vector<8xf16>
      %2541 = vector.extract %2537[%200] : f16 from vector<8xf16>
      %2542 = vector.extract %2537[2] : f16 from vector<8xf16>
      %2543 = vector.extract %2537[%201] : f16 from vector<8xf16>
      %2544 = vector.extract %2537[3] : f16 from vector<8xf16>
      %2545 = vector.extract %2537[%202] : f16 from vector<8xf16>
      %2546 = frisk.read_matrix_from_shm %view[%2334] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2547 = vector.bitcast %2546 : vector<4xi32> to vector<8xf16>
      %2548 = vector.extract %2547[0] : f16 from vector<8xf16>
      %2549 = vector.extract %2547[%199] : f16 from vector<8xf16>
      %2550 = vector.extract %2547[1] : f16 from vector<8xf16>
      %2551 = vector.extract %2547[%200] : f16 from vector<8xf16>
      %2552 = vector.extract %2547[2] : f16 from vector<8xf16>
      %2553 = vector.extract %2547[%201] : f16 from vector<8xf16>
      %2554 = vector.extract %2547[3] : f16 from vector<8xf16>
      %2555 = vector.extract %2547[%202] : f16 from vector<8xf16>
      %2556 = frisk.read_matrix_from_shm %view[%2345] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2557 = vector.bitcast %2556 : vector<4xi32> to vector<8xf16>
      %2558 = vector.extract %2557[0] : f16 from vector<8xf16>
      %2559 = vector.extract %2557[%199] : f16 from vector<8xf16>
      %2560 = vector.extract %2557[1] : f16 from vector<8xf16>
      %2561 = vector.extract %2557[%200] : f16 from vector<8xf16>
      %2562 = vector.extract %2557[2] : f16 from vector<8xf16>
      %2563 = vector.extract %2557[%201] : f16 from vector<8xf16>
      %2564 = vector.extract %2557[3] : f16 from vector<8xf16>
      %2565 = vector.extract %2557[%202] : f16 from vector<8xf16>
      %2566 = frisk.read_matrix_from_shm %view[%2356] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2567 = vector.bitcast %2566 : vector<4xi32> to vector<8xf16>
      %2568 = vector.extract %2567[0] : f16 from vector<8xf16>
      %2569 = vector.extract %2567[%199] : f16 from vector<8xf16>
      %2570 = vector.extract %2567[1] : f16 from vector<8xf16>
      %2571 = vector.extract %2567[%200] : f16 from vector<8xf16>
      %2572 = vector.extract %2567[2] : f16 from vector<8xf16>
      %2573 = vector.extract %2567[%201] : f16 from vector<8xf16>
      %2574 = vector.extract %2567[3] : f16 from vector<8xf16>
      %2575 = vector.extract %2567[%202] : f16 from vector<8xf16>
      %2576 = frisk.read_matrix_from_shm %view[%2367] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2577 = vector.bitcast %2576 : vector<4xi32> to vector<8xf16>
      %2578 = vector.extract %2577[0] : f16 from vector<8xf16>
      %2579 = vector.extract %2577[%199] : f16 from vector<8xf16>
      %2580 = vector.extract %2577[1] : f16 from vector<8xf16>
      %2581 = vector.extract %2577[%200] : f16 from vector<8xf16>
      %2582 = vector.extract %2577[2] : f16 from vector<8xf16>
      %2583 = vector.extract %2577[%201] : f16 from vector<8xf16>
      %2584 = vector.extract %2577[3] : f16 from vector<8xf16>
      %2585 = vector.extract %2577[%202] : f16 from vector<8xf16>
      %2586 = frisk.read_matrix_from_shm %view[%2378] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2587 = vector.bitcast %2586 : vector<4xi32> to vector<8xf16>
      %2588 = vector.extract %2587[0] : f16 from vector<8xf16>
      %2589 = vector.extract %2587[%199] : f16 from vector<8xf16>
      %2590 = vector.extract %2587[1] : f16 from vector<8xf16>
      %2591 = vector.extract %2587[%200] : f16 from vector<8xf16>
      %2592 = vector.extract %2587[2] : f16 from vector<8xf16>
      %2593 = vector.extract %2587[%201] : f16 from vector<8xf16>
      %2594 = vector.extract %2587[3] : f16 from vector<8xf16>
      %2595 = vector.extract %2587[%202] : f16 from vector<8xf16>
      %2596 = frisk.read_matrix_from_shm %view[%2389] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2597 = vector.bitcast %2596 : vector<4xi32> to vector<8xf16>
      %2598 = vector.extract %2597[0] : f16 from vector<8xf16>
      %2599 = vector.extract %2597[%199] : f16 from vector<8xf16>
      %2600 = vector.extract %2597[1] : f16 from vector<8xf16>
      %2601 = vector.extract %2597[%200] : f16 from vector<8xf16>
      %2602 = vector.extract %2597[2] : f16 from vector<8xf16>
      %2603 = vector.extract %2597[%201] : f16 from vector<8xf16>
      %2604 = vector.extract %2597[3] : f16 from vector<8xf16>
      %2605 = vector.extract %2597[%202] : f16 from vector<8xf16>
      %2606 = frisk.read_matrix_from_shm %view[%2400] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2607 = vector.bitcast %2606 : vector<4xi32> to vector<8xf16>
      %2608 = vector.extract %2607[0] : f16 from vector<8xf16>
      %2609 = vector.extract %2607[%199] : f16 from vector<8xf16>
      %2610 = vector.extract %2607[1] : f16 from vector<8xf16>
      %2611 = vector.extract %2607[%200] : f16 from vector<8xf16>
      %2612 = vector.extract %2607[2] : f16 from vector<8xf16>
      %2613 = vector.extract %2607[%201] : f16 from vector<8xf16>
      %2614 = vector.extract %2607[3] : f16 from vector<8xf16>
      %2615 = vector.extract %2607[%202] : f16 from vector<8xf16>
      %subview_158 = memref.subview %arg2[0, %198, %583, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      %2616 = frisk.async_copy_gm_to_shm %subview_158 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %2617 = vector.from_elements %2538, %2540, %2542, %2544 : vector<4xf16>
      %2618 = vector.from_elements %2502, %2503, %2504, %2505 : vector<4xf32>
      %2619 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2412, %2617, %2618 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2620 = vector.from_elements %2548, %2550, %2552, %2554 : vector<4xf16>
      %2621 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2416, %2620, %2619 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2622 = vector.from_elements %2558, %2560, %2562, %2564 : vector<4xf16>
      %2623 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2419, %2622, %2621 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2624 = vector.from_elements %2568, %2570, %2572, %2574 : vector<4xf16>
      %2625 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2422, %2624, %2623 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2626 = vector.from_elements %2578, %2580, %2582, %2584 : vector<4xf16>
      %2627 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2425, %2626, %2625 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2628 = vector.from_elements %2588, %2590, %2592, %2594 : vector<4xf16>
      %2629 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2428, %2628, %2627 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2630 = vector.from_elements %2598, %2600, %2602, %2604 : vector<4xf16>
      %2631 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2431, %2630, %2629 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2632 = vector.from_elements %2608, %2610, %2612, %2614 : vector<4xf16>
      %2633 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2434, %2632, %2631 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2634 = vector.extract %2633[0] : f32 from vector<4xf32>
      %2635 = vector.extract %2633[1] : f32 from vector<4xf32>
      %2636 = vector.extract %2633[2] : f32 from vector<4xf32>
      %2637 = vector.extract %2633[3] : f32 from vector<4xf32>
      %2638 = vector.from_elements %2539, %2541, %2543, %2545 : vector<4xf16>
      %2639 = vector.from_elements %2512, %2513, %2514, %2515 : vector<4xf32>
      %2640 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2412, %2638, %2639 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2641 = vector.from_elements %2549, %2551, %2553, %2555 : vector<4xf16>
      %2642 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2416, %2641, %2640 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2643 = vector.from_elements %2559, %2561, %2563, %2565 : vector<4xf16>
      %2644 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2419, %2643, %2642 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2645 = vector.from_elements %2569, %2571, %2573, %2575 : vector<4xf16>
      %2646 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2422, %2645, %2644 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2647 = vector.from_elements %2579, %2581, %2583, %2585 : vector<4xf16>
      %2648 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2425, %2647, %2646 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2649 = vector.from_elements %2589, %2591, %2593, %2595 : vector<4xf16>
      %2650 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2428, %2649, %2648 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2651 = vector.from_elements %2599, %2601, %2603, %2605 : vector<4xf16>
      %2652 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2431, %2651, %2650 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2653 = vector.from_elements %2609, %2611, %2613, %2615 : vector<4xf16>
      %2654 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2434, %2653, %2652 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2655 = vector.extract %2654[0] : f32 from vector<4xf32>
      %2656 = vector.extract %2654[1] : f32 from vector<4xf32>
      %2657 = vector.extract %2654[2] : f32 from vector<4xf32>
      %2658 = vector.extract %2654[3] : f32 from vector<4xf32>
      %2659 = vector.from_elements %2522, %2523, %2524, %2525 : vector<4xf32>
      %2660 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2462, %2617, %2659 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2661 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2465, %2620, %2660 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2662 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2622, %2661 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2663 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2624, %2662 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2664 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2626, %2663 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2665 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2628, %2664 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2666 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2630, %2665 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2667 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2632, %2666 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2668 = vector.extract %2667[0] : f32 from vector<4xf32>
      %2669 = vector.extract %2667[1] : f32 from vector<4xf32>
      %2670 = vector.extract %2667[2] : f32 from vector<4xf32>
      %2671 = vector.extract %2667[3] : f32 from vector<4xf32>
      %2672 = vector.from_elements %2532, %2533, %2534, %2535 : vector<4xf32>
      %2673 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2462, %2638, %2672 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2674 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2465, %2641, %2673 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2675 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2643, %2674 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2676 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2645, %2675 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2677 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2647, %2676 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2678 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2649, %2677 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2679 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2651, %2678 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2680 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2653, %2679 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2681 = vector.extract %2680[0] : f32 from vector<4xf32>
      %2682 = vector.extract %2680[1] : f32 from vector<4xf32>
      %2683 = vector.extract %2680[2] : f32 from vector<4xf32>
      %2684 = vector.extract %2680[3] : f32 from vector<4xf32>
      %2685 = vector.extract %arg9[0, 0] : f32 from vector<2x8xf32>
      %2686 = vector.extract %arg9[0, 1] : f32 from vector<2x8xf32>
      %2687 = vector.extract %arg9[0, 2] : f32 from vector<2x8xf32>
      %2688 = vector.extract %arg9[0, 3] : f32 from vector<2x8xf32>
      %2689 = vector.from_elements %2685, %2686, %2687, %2688 : vector<4xf32>
      %2690 = arith.mulf %2689, %2286 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2691 = vector.extract %2690[0] : f32 from vector<4xf32>
      %2692 = vector.extract %2690[1] : f32 from vector<4xf32>
      %2693 = vector.extract %2690[2] : f32 from vector<4xf32>
      %2694 = vector.extract %2690[3] : f32 from vector<4xf32>
      %2695 = vector.extract %arg9[0, 4] : f32 from vector<2x8xf32>
      %2696 = vector.extract %arg9[0, 5] : f32 from vector<2x8xf32>
      %2697 = vector.extract %arg9[0, 6] : f32 from vector<2x8xf32>
      %2698 = vector.extract %arg9[0, 7] : f32 from vector<2x8xf32>
      %2699 = vector.from_elements %2695, %2696, %2697, %2698 : vector<4xf32>
      %2700 = arith.mulf %2699, %2286 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2701 = vector.extract %2700[0] : f32 from vector<4xf32>
      %2702 = vector.extract %2700[1] : f32 from vector<4xf32>
      %2703 = vector.extract %2700[2] : f32 from vector<4xf32>
      %2704 = vector.extract %2700[3] : f32 from vector<4xf32>
      %2705 = vector.extract %arg9[1, 0] : f32 from vector<2x8xf32>
      %2706 = vector.extract %arg9[1, 1] : f32 from vector<2x8xf32>
      %2707 = vector.extract %arg9[1, 2] : f32 from vector<2x8xf32>
      %2708 = vector.extract %arg9[1, 3] : f32 from vector<2x8xf32>
      %2709 = vector.from_elements %2705, %2706, %2707, %2708 : vector<4xf32>
      %2710 = arith.mulf %2709, %2307 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2711 = vector.extract %2710[0] : f32 from vector<4xf32>
      %2712 = vector.extract %2710[1] : f32 from vector<4xf32>
      %2713 = vector.extract %2710[2] : f32 from vector<4xf32>
      %2714 = vector.extract %2710[3] : f32 from vector<4xf32>
      %2715 = vector.extract %arg9[1, 4] : f32 from vector<2x8xf32>
      %2716 = vector.extract %arg9[1, 5] : f32 from vector<2x8xf32>
      %2717 = vector.extract %arg9[1, 6] : f32 from vector<2x8xf32>
      %2718 = vector.extract %arg9[1, 7] : f32 from vector<2x8xf32>
      %2719 = vector.from_elements %2715, %2716, %2717, %2718 : vector<4xf32>
      %2720 = arith.mulf %2719, %2307 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2721 = vector.extract %2720[0] : f32 from vector<4xf32>
      %2722 = vector.extract %2720[1] : f32 from vector<4xf32>
      %2723 = vector.extract %2720[2] : f32 from vector<4xf32>
      %2724 = vector.extract %2720[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %2616 : !frisk.async_task
      %2725 = frisk.read_matrix_from_shm %view[%2323] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2726 = vector.bitcast %2725 : vector<4xi32> to vector<8xf16>
      %2727 = vector.extract %2726[0] : f16 from vector<8xf16>
      %2728 = vector.extract %2726[%199] : f16 from vector<8xf16>
      %2729 = vector.extract %2726[1] : f16 from vector<8xf16>
      %2730 = vector.extract %2726[%200] : f16 from vector<8xf16>
      %2731 = vector.extract %2726[2] : f16 from vector<8xf16>
      %2732 = vector.extract %2726[%201] : f16 from vector<8xf16>
      %2733 = vector.extract %2726[3] : f16 from vector<8xf16>
      %2734 = vector.extract %2726[%202] : f16 from vector<8xf16>
      %2735 = frisk.read_matrix_from_shm %view[%2334] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2736 = vector.bitcast %2735 : vector<4xi32> to vector<8xf16>
      %2737 = vector.extract %2736[0] : f16 from vector<8xf16>
      %2738 = vector.extract %2736[%199] : f16 from vector<8xf16>
      %2739 = vector.extract %2736[1] : f16 from vector<8xf16>
      %2740 = vector.extract %2736[%200] : f16 from vector<8xf16>
      %2741 = vector.extract %2736[2] : f16 from vector<8xf16>
      %2742 = vector.extract %2736[%201] : f16 from vector<8xf16>
      %2743 = vector.extract %2736[3] : f16 from vector<8xf16>
      %2744 = vector.extract %2736[%202] : f16 from vector<8xf16>
      %2745 = frisk.read_matrix_from_shm %view[%2345] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2746 = vector.bitcast %2745 : vector<4xi32> to vector<8xf16>
      %2747 = vector.extract %2746[0] : f16 from vector<8xf16>
      %2748 = vector.extract %2746[%199] : f16 from vector<8xf16>
      %2749 = vector.extract %2746[1] : f16 from vector<8xf16>
      %2750 = vector.extract %2746[%200] : f16 from vector<8xf16>
      %2751 = vector.extract %2746[2] : f16 from vector<8xf16>
      %2752 = vector.extract %2746[%201] : f16 from vector<8xf16>
      %2753 = vector.extract %2746[3] : f16 from vector<8xf16>
      %2754 = vector.extract %2746[%202] : f16 from vector<8xf16>
      %2755 = frisk.read_matrix_from_shm %view[%2356] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2756 = vector.bitcast %2755 : vector<4xi32> to vector<8xf16>
      %2757 = vector.extract %2756[0] : f16 from vector<8xf16>
      %2758 = vector.extract %2756[%199] : f16 from vector<8xf16>
      %2759 = vector.extract %2756[1] : f16 from vector<8xf16>
      %2760 = vector.extract %2756[%200] : f16 from vector<8xf16>
      %2761 = vector.extract %2756[2] : f16 from vector<8xf16>
      %2762 = vector.extract %2756[%201] : f16 from vector<8xf16>
      %2763 = vector.extract %2756[3] : f16 from vector<8xf16>
      %2764 = vector.extract %2756[%202] : f16 from vector<8xf16>
      %2765 = frisk.read_matrix_from_shm %view[%2367] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2766 = vector.bitcast %2765 : vector<4xi32> to vector<8xf16>
      %2767 = vector.extract %2766[0] : f16 from vector<8xf16>
      %2768 = vector.extract %2766[%199] : f16 from vector<8xf16>
      %2769 = vector.extract %2766[1] : f16 from vector<8xf16>
      %2770 = vector.extract %2766[%200] : f16 from vector<8xf16>
      %2771 = vector.extract %2766[2] : f16 from vector<8xf16>
      %2772 = vector.extract %2766[%201] : f16 from vector<8xf16>
      %2773 = vector.extract %2766[3] : f16 from vector<8xf16>
      %2774 = vector.extract %2766[%202] : f16 from vector<8xf16>
      %2775 = frisk.read_matrix_from_shm %view[%2378] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2776 = vector.bitcast %2775 : vector<4xi32> to vector<8xf16>
      %2777 = vector.extract %2776[0] : f16 from vector<8xf16>
      %2778 = vector.extract %2776[%199] : f16 from vector<8xf16>
      %2779 = vector.extract %2776[1] : f16 from vector<8xf16>
      %2780 = vector.extract %2776[%200] : f16 from vector<8xf16>
      %2781 = vector.extract %2776[2] : f16 from vector<8xf16>
      %2782 = vector.extract %2776[%201] : f16 from vector<8xf16>
      %2783 = vector.extract %2776[3] : f16 from vector<8xf16>
      %2784 = vector.extract %2776[%202] : f16 from vector<8xf16>
      %2785 = frisk.read_matrix_from_shm %view[%2389] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2786 = vector.bitcast %2785 : vector<4xi32> to vector<8xf16>
      %2787 = vector.extract %2786[0] : f16 from vector<8xf16>
      %2788 = vector.extract %2786[%199] : f16 from vector<8xf16>
      %2789 = vector.extract %2786[1] : f16 from vector<8xf16>
      %2790 = vector.extract %2786[%200] : f16 from vector<8xf16>
      %2791 = vector.extract %2786[2] : f16 from vector<8xf16>
      %2792 = vector.extract %2786[%201] : f16 from vector<8xf16>
      %2793 = vector.extract %2786[3] : f16 from vector<8xf16>
      %2794 = vector.extract %2786[%202] : f16 from vector<8xf16>
      %2795 = frisk.read_matrix_from_shm %view[%2400] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2796 = vector.bitcast %2795 : vector<4xi32> to vector<8xf16>
      %2797 = vector.extract %2796[0] : f16 from vector<8xf16>
      %2798 = vector.extract %2796[%199] : f16 from vector<8xf16>
      %2799 = vector.extract %2796[1] : f16 from vector<8xf16>
      %2800 = vector.extract %2796[%200] : f16 from vector<8xf16>
      %2801 = vector.extract %2796[2] : f16 from vector<8xf16>
      %2802 = vector.extract %2796[%201] : f16 from vector<8xf16>
      %2803 = vector.extract %2796[3] : f16 from vector<8xf16>
      %2804 = vector.extract %2796[%202] : f16 from vector<8xf16>
      %subview_159 = memref.subview %arg2[0, %198, %583, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      gpu.barrier
      %2805 = frisk.async_copy_gm_to_shm %subview_159 to %view {alignment = 16 : i64, frisk.dma_bytes = 16 : i64, frisk.kpack = 2 : i64, frisk.source_map = affine_map<(d0) -> (d0 floordiv 32, ((d0 floordiv 8) mod 4) * 8 + d0 mod 8)>} : memref<128x32xf16, strided<[128, 1], offset: ?>, 1>, memref<4096xf16, 3> -> !frisk.async_task
      %2806 = vector.from_elements %2727, %2729, %2731, %2733 : vector<4xf16>
      %2807 = vector.from_elements %2691, %2692, %2693, %2694 : vector<4xf32>
      %2808 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2412, %2806, %2807 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2809 = vector.from_elements %2737, %2739, %2741, %2743 : vector<4xf16>
      %2810 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2416, %2809, %2808 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2811 = vector.from_elements %2747, %2749, %2751, %2753 : vector<4xf16>
      %2812 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2419, %2811, %2810 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2813 = vector.from_elements %2757, %2759, %2761, %2763 : vector<4xf16>
      %2814 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2422, %2813, %2812 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2815 = vector.from_elements %2767, %2769, %2771, %2773 : vector<4xf16>
      %2816 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2425, %2815, %2814 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2817 = vector.from_elements %2777, %2779, %2781, %2783 : vector<4xf16>
      %2818 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2428, %2817, %2816 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2819 = vector.from_elements %2787, %2789, %2791, %2793 : vector<4xf16>
      %2820 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2431, %2819, %2818 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2821 = vector.from_elements %2797, %2799, %2801, %2803 : vector<4xf16>
      %2822 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2434, %2821, %2820 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2823 = vector.extract %2822[0] : f32 from vector<4xf32>
      %2824 = vector.extract %2822[1] : f32 from vector<4xf32>
      %2825 = vector.extract %2822[2] : f32 from vector<4xf32>
      %2826 = vector.extract %2822[3] : f32 from vector<4xf32>
      %2827 = vector.from_elements %2728, %2730, %2732, %2734 : vector<4xf16>
      %2828 = vector.from_elements %2701, %2702, %2703, %2704 : vector<4xf32>
      %2829 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2412, %2827, %2828 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2830 = vector.from_elements %2738, %2740, %2742, %2744 : vector<4xf16>
      %2831 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2416, %2830, %2829 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2832 = vector.from_elements %2748, %2750, %2752, %2754 : vector<4xf16>
      %2833 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2419, %2832, %2831 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2834 = vector.from_elements %2758, %2760, %2762, %2764 : vector<4xf16>
      %2835 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2422, %2834, %2833 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2836 = vector.from_elements %2768, %2770, %2772, %2774 : vector<4xf16>
      %2837 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2425, %2836, %2835 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2838 = vector.from_elements %2778, %2780, %2782, %2784 : vector<4xf16>
      %2839 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2428, %2838, %2837 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2840 = vector.from_elements %2788, %2790, %2792, %2794 : vector<4xf16>
      %2841 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2431, %2840, %2839 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2842 = vector.from_elements %2798, %2800, %2802, %2804 : vector<4xf16>
      %2843 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2434, %2842, %2841 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2844 = vector.extract %2843[0] : f32 from vector<4xf32>
      %2845 = vector.extract %2843[1] : f32 from vector<4xf32>
      %2846 = vector.extract %2843[2] : f32 from vector<4xf32>
      %2847 = vector.extract %2843[3] : f32 from vector<4xf32>
      %2848 = vector.from_elements %2711, %2712, %2713, %2714 : vector<4xf32>
      %2849 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2462, %2806, %2848 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2850 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2465, %2809, %2849 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2851 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2811, %2850 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2852 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2813, %2851 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2853 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2815, %2852 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2854 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2817, %2853 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2855 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2819, %2854 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2856 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2821, %2855 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2857 = vector.extract %2856[0] : f32 from vector<4xf32>
      %2858 = vector.extract %2856[1] : f32 from vector<4xf32>
      %2859 = vector.extract %2856[2] : f32 from vector<4xf32>
      %2860 = vector.extract %2856[3] : f32 from vector<4xf32>
      %2861 = vector.from_elements %2721, %2722, %2723, %2724 : vector<4xf32>
      %2862 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2462, %2827, %2861 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2863 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2465, %2830, %2862 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2864 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2832, %2863 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2865 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %2834, %2864 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2866 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %2836, %2865 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2867 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %2838, %2866 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2868 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %2840, %2867 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2869 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %2842, %2868 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2870 = vector.extract %2869[0] : f32 from vector<4xf32>
      %2871 = vector.extract %2869[1] : f32 from vector<4xf32>
      %2872 = vector.extract %2869[2] : f32 from vector<4xf32>
      %2873 = vector.extract %2869[3] : f32 from vector<4xf32>
      %2874 = vector.extract %arg10[0, 0] : f32 from vector<2x8xf32>
      %2875 = vector.extract %arg10[0, 1] : f32 from vector<2x8xf32>
      %2876 = vector.extract %arg10[0, 2] : f32 from vector<2x8xf32>
      %2877 = vector.extract %arg10[0, 3] : f32 from vector<2x8xf32>
      %2878 = vector.from_elements %2874, %2875, %2876, %2877 : vector<4xf32>
      %2879 = arith.mulf %2878, %2286 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2880 = vector.extract %2879[0] : f32 from vector<4xf32>
      %2881 = vector.extract %2879[1] : f32 from vector<4xf32>
      %2882 = vector.extract %2879[2] : f32 from vector<4xf32>
      %2883 = vector.extract %2879[3] : f32 from vector<4xf32>
      %2884 = vector.extract %arg10[0, 4] : f32 from vector<2x8xf32>
      %2885 = vector.extract %arg10[0, 5] : f32 from vector<2x8xf32>
      %2886 = vector.extract %arg10[0, 6] : f32 from vector<2x8xf32>
      %2887 = vector.extract %arg10[0, 7] : f32 from vector<2x8xf32>
      %2888 = vector.from_elements %2884, %2885, %2886, %2887 : vector<4xf32>
      %2889 = arith.mulf %2888, %2286 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2890 = vector.extract %2889[0] : f32 from vector<4xf32>
      %2891 = vector.extract %2889[1] : f32 from vector<4xf32>
      %2892 = vector.extract %2889[2] : f32 from vector<4xf32>
      %2893 = vector.extract %2889[3] : f32 from vector<4xf32>
      %2894 = vector.extract %arg10[1, 0] : f32 from vector<2x8xf32>
      %2895 = vector.extract %arg10[1, 1] : f32 from vector<2x8xf32>
      %2896 = vector.extract %arg10[1, 2] : f32 from vector<2x8xf32>
      %2897 = vector.extract %arg10[1, 3] : f32 from vector<2x8xf32>
      %2898 = vector.from_elements %2894, %2895, %2896, %2897 : vector<4xf32>
      %2899 = arith.mulf %2898, %2307 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2900 = vector.extract %2899[0] : f32 from vector<4xf32>
      %2901 = vector.extract %2899[1] : f32 from vector<4xf32>
      %2902 = vector.extract %2899[2] : f32 from vector<4xf32>
      %2903 = vector.extract %2899[3] : f32 from vector<4xf32>
      %2904 = vector.extract %arg10[1, 4] : f32 from vector<2x8xf32>
      %2905 = vector.extract %arg10[1, 5] : f32 from vector<2x8xf32>
      %2906 = vector.extract %arg10[1, 6] : f32 from vector<2x8xf32>
      %2907 = vector.extract %arg10[1, 7] : f32 from vector<2x8xf32>
      %2908 = vector.from_elements %2904, %2905, %2906, %2907 : vector<4xf32>
      %2909 = arith.mulf %2908, %2307 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2910 = vector.extract %2909[0] : f32 from vector<4xf32>
      %2911 = vector.extract %2909[1] : f32 from vector<4xf32>
      %2912 = vector.extract %2909[2] : f32 from vector<4xf32>
      %2913 = vector.extract %2909[3] : f32 from vector<4xf32>
      frisk.wait_buffer_ready %2805 : !frisk.async_task
      %2914 = frisk.read_matrix_from_shm %view[%2323] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2915 = vector.bitcast %2914 : vector<4xi32> to vector<8xf16>
      %2916 = vector.extract %2915[0] : f16 from vector<8xf16>
      %2917 = vector.extract %2915[%199] : f16 from vector<8xf16>
      %2918 = vector.extract %2915[1] : f16 from vector<8xf16>
      %2919 = vector.extract %2915[%200] : f16 from vector<8xf16>
      %2920 = vector.extract %2915[2] : f16 from vector<8xf16>
      %2921 = vector.extract %2915[%201] : f16 from vector<8xf16>
      %2922 = vector.extract %2915[3] : f16 from vector<8xf16>
      %2923 = vector.extract %2915[%202] : f16 from vector<8xf16>
      %2924 = frisk.read_matrix_from_shm %view[%2334] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2925 = vector.bitcast %2924 : vector<4xi32> to vector<8xf16>
      %2926 = vector.extract %2925[0] : f16 from vector<8xf16>
      %2927 = vector.extract %2925[%199] : f16 from vector<8xf16>
      %2928 = vector.extract %2925[1] : f16 from vector<8xf16>
      %2929 = vector.extract %2925[%200] : f16 from vector<8xf16>
      %2930 = vector.extract %2925[2] : f16 from vector<8xf16>
      %2931 = vector.extract %2925[%201] : f16 from vector<8xf16>
      %2932 = vector.extract %2925[3] : f16 from vector<8xf16>
      %2933 = vector.extract %2925[%202] : f16 from vector<8xf16>
      %2934 = frisk.read_matrix_from_shm %view[%2345] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2935 = vector.bitcast %2934 : vector<4xi32> to vector<8xf16>
      %2936 = vector.extract %2935[0] : f16 from vector<8xf16>
      %2937 = vector.extract %2935[%199] : f16 from vector<8xf16>
      %2938 = vector.extract %2935[1] : f16 from vector<8xf16>
      %2939 = vector.extract %2935[%200] : f16 from vector<8xf16>
      %2940 = vector.extract %2935[2] : f16 from vector<8xf16>
      %2941 = vector.extract %2935[%201] : f16 from vector<8xf16>
      %2942 = vector.extract %2935[3] : f16 from vector<8xf16>
      %2943 = vector.extract %2935[%202] : f16 from vector<8xf16>
      %2944 = frisk.read_matrix_from_shm %view[%2356] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2945 = vector.bitcast %2944 : vector<4xi32> to vector<8xf16>
      %2946 = vector.extract %2945[0] : f16 from vector<8xf16>
      %2947 = vector.extract %2945[%199] : f16 from vector<8xf16>
      %2948 = vector.extract %2945[1] : f16 from vector<8xf16>
      %2949 = vector.extract %2945[%200] : f16 from vector<8xf16>
      %2950 = vector.extract %2945[2] : f16 from vector<8xf16>
      %2951 = vector.extract %2945[%201] : f16 from vector<8xf16>
      %2952 = vector.extract %2945[3] : f16 from vector<8xf16>
      %2953 = vector.extract %2945[%202] : f16 from vector<8xf16>
      %2954 = frisk.read_matrix_from_shm %view[%2367] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2955 = vector.bitcast %2954 : vector<4xi32> to vector<8xf16>
      %2956 = vector.extract %2955[0] : f16 from vector<8xf16>
      %2957 = vector.extract %2955[%199] : f16 from vector<8xf16>
      %2958 = vector.extract %2955[1] : f16 from vector<8xf16>
      %2959 = vector.extract %2955[%200] : f16 from vector<8xf16>
      %2960 = vector.extract %2955[2] : f16 from vector<8xf16>
      %2961 = vector.extract %2955[%201] : f16 from vector<8xf16>
      %2962 = vector.extract %2955[3] : f16 from vector<8xf16>
      %2963 = vector.extract %2955[%202] : f16 from vector<8xf16>
      %2964 = frisk.read_matrix_from_shm %view[%2378] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2965 = vector.bitcast %2964 : vector<4xi32> to vector<8xf16>
      %2966 = vector.extract %2965[0] : f16 from vector<8xf16>
      %2967 = vector.extract %2965[%199] : f16 from vector<8xf16>
      %2968 = vector.extract %2965[1] : f16 from vector<8xf16>
      %2969 = vector.extract %2965[%200] : f16 from vector<8xf16>
      %2970 = vector.extract %2965[2] : f16 from vector<8xf16>
      %2971 = vector.extract %2965[%201] : f16 from vector<8xf16>
      %2972 = vector.extract %2965[3] : f16 from vector<8xf16>
      %2973 = vector.extract %2965[%202] : f16 from vector<8xf16>
      %2974 = frisk.read_matrix_from_shm %view[%2389] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2975 = vector.bitcast %2974 : vector<4xi32> to vector<8xf16>
      %2976 = vector.extract %2975[0] : f16 from vector<8xf16>
      %2977 = vector.extract %2975[%199] : f16 from vector<8xf16>
      %2978 = vector.extract %2975[1] : f16 from vector<8xf16>
      %2979 = vector.extract %2975[%200] : f16 from vector<8xf16>
      %2980 = vector.extract %2975[2] : f16 from vector<8xf16>
      %2981 = vector.extract %2975[%201] : f16 from vector<8xf16>
      %2982 = vector.extract %2975[3] : f16 from vector<8xf16>
      %2983 = vector.extract %2975[%202] : f16 from vector<8xf16>
      %2984 = frisk.read_matrix_from_shm %view[%2400] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2985 = vector.bitcast %2984 : vector<4xi32> to vector<8xf16>
      %2986 = vector.extract %2985[0] : f16 from vector<8xf16>
      %2987 = vector.extract %2985[%199] : f16 from vector<8xf16>
      %2988 = vector.extract %2985[1] : f16 from vector<8xf16>
      %2989 = vector.extract %2985[%200] : f16 from vector<8xf16>
      %2990 = vector.extract %2985[2] : f16 from vector<8xf16>
      %2991 = vector.extract %2985[%201] : f16 from vector<8xf16>
      %2992 = vector.extract %2985[3] : f16 from vector<8xf16>
      %2993 = vector.extract %2985[%202] : f16 from vector<8xf16>
      %2994 = vector.from_elements %2916, %2918, %2920, %2922 : vector<4xf16>
      %2995 = vector.from_elements %2880, %2881, %2882, %2883 : vector<4xf32>
      %2996 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2412, %2994, %2995 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2997 = vector.from_elements %2926, %2928, %2930, %2932 : vector<4xf16>
      %2998 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2416, %2997, %2996 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2999 = vector.from_elements %2936, %2938, %2940, %2942 : vector<4xf16>
      %3000 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2419, %2999, %2998 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3001 = vector.from_elements %2946, %2948, %2950, %2952 : vector<4xf16>
      %3002 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2422, %3001, %3000 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3003 = vector.from_elements %2956, %2958, %2960, %2962 : vector<4xf16>
      %3004 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2425, %3003, %3002 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3005 = vector.from_elements %2966, %2968, %2970, %2972 : vector<4xf16>
      %3006 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2428, %3005, %3004 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3007 = vector.from_elements %2976, %2978, %2980, %2982 : vector<4xf16>
      %3008 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2431, %3007, %3006 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3009 = vector.from_elements %2986, %2988, %2990, %2992 : vector<4xf16>
      %3010 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2434, %3009, %3008 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3011 = vector.extract %3010[0] : f32 from vector<4xf32>
      %3012 = vector.extract %3010[1] : f32 from vector<4xf32>
      %3013 = vector.extract %3010[2] : f32 from vector<4xf32>
      %3014 = vector.extract %3010[3] : f32 from vector<4xf32>
      %3015 = vector.from_elements %2917, %2919, %2921, %2923 : vector<4xf16>
      %3016 = vector.from_elements %2890, %2891, %2892, %2893 : vector<4xf32>
      %3017 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2412, %3015, %3016 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3018 = vector.from_elements %2927, %2929, %2931, %2933 : vector<4xf16>
      %3019 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2416, %3018, %3017 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3020 = vector.from_elements %2937, %2939, %2941, %2943 : vector<4xf16>
      %3021 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2419, %3020, %3019 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3022 = vector.from_elements %2947, %2949, %2951, %2953 : vector<4xf16>
      %3023 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2422, %3022, %3021 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3024 = vector.from_elements %2957, %2959, %2961, %2963 : vector<4xf16>
      %3025 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2425, %3024, %3023 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3026 = vector.from_elements %2967, %2969, %2971, %2973 : vector<4xf16>
      %3027 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2428, %3026, %3025 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3028 = vector.from_elements %2977, %2979, %2981, %2983 : vector<4xf16>
      %3029 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2431, %3028, %3027 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3030 = vector.from_elements %2987, %2989, %2991, %2993 : vector<4xf16>
      %3031 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2434, %3030, %3029 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3032 = vector.extract %3031[0] : f32 from vector<4xf32>
      %3033 = vector.extract %3031[1] : f32 from vector<4xf32>
      %3034 = vector.extract %3031[2] : f32 from vector<4xf32>
      %3035 = vector.extract %3031[3] : f32 from vector<4xf32>
      %3036 = vector.from_elements %2900, %2901, %2902, %2903 : vector<4xf32>
      %3037 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2462, %2994, %3036 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3038 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2465, %2997, %3037 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3039 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %2999, %3038 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3040 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %3001, %3039 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3041 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %3003, %3040 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3042 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %3005, %3041 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3043 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %3007, %3042 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3044 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %3009, %3043 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3045 = vector.extract %3044[0] : f32 from vector<4xf32>
      %3046 = vector.extract %3044[1] : f32 from vector<4xf32>
      %3047 = vector.extract %3044[2] : f32 from vector<4xf32>
      %3048 = vector.extract %3044[3] : f32 from vector<4xf32>
      %3049 = vector.from_elements %2910, %2911, %2912, %2913 : vector<4xf32>
      %3050 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2462, %3015, %3049 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3051 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2465, %3018, %3050 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3052 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2467, %3020, %3051 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3053 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2469, %3022, %3052 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3054 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2471, %3024, %3053 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3055 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2473, %3026, %3054 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3056 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2475, %3028, %3055 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3057 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2477, %3030, %3056 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3058 = vector.extract %3057[0] : f32 from vector<4xf32>
      %3059 = vector.extract %3057[1] : f32 from vector<4xf32>
      %3060 = vector.extract %3057[2] : f32 from vector<4xf32>
      %3061 = vector.extract %3057[3] : f32 from vector<4xf32>
      %3062 = vector.from_elements %2437, %2438, %2439, %2440, %2458, %2459, %2460, %2461, %2479, %2480, %2481, %2482, %2492, %2493, %2494, %2495 : vector<2x8xf32>
      %3063 = vector.from_elements %2180, %2184 : vector<2x1xf32>
      %3064 = vector.from_elements %1893, %1898 : vector<2x1xf32>
      %3065 = vector.from_elements %2634, %2635, %2636, %2637, %2655, %2656, %2657, %2658, %2668, %2669, %2670, %2671, %2681, %2682, %2683, %2684 : vector<2x8xf32>
      %3066 = vector.from_elements %2823, %2824, %2825, %2826, %2844, %2845, %2846, %2847, %2857, %2858, %2859, %2860, %2870, %2871, %2872, %2873 : vector<2x8xf32>
      %3067 = vector.from_elements %3011, %3012, %3013, %3014, %3032, %3033, %3034, %3035, %3045, %3046, %3047, %3048, %3058, %3059, %3060, %3061 : vector<2x8xf32>
      gpu.barrier
      affine.yield %3062, %3063, %3064, %3065, %3066, %3067 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
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
    %shuffleResult_10, %valid_11 = gpu.shuffle xor %216, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %217 = arith.select %206, %shuffleResult_10, %211 : f32
    %218 = arith.select %206, %212, %shuffleResult_10 : f32
    %219 = arith.select %208, %214, %217 : f32
    %shuffleResult_12, %valid_13 = gpu.shuffle xor %219, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %220 = arith.select %208, %shuffleResult_12, %214 : f32
    %221 = arith.select %208, %217, %shuffleResult_12 : f32
    %222 = arith.select %208, %215, %218 : f32
    %shuffleResult_14, %valid_15 = gpu.shuffle xor %222, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %223 = arith.select %208, %shuffleResult_14, %215 : f32
    %224 = arith.select %208, %218, %shuffleResult_14 : f32
    %225 = vector.extract %203#5[0, 4] : f32 from vector<2x8xf32>
    %226 = vector.extract %203#5[0, 5] : f32 from vector<2x8xf32>
    %227 = vector.extract %203#5[0, 6] : f32 from vector<2x8xf32>
    %228 = vector.extract %203#5[0, 7] : f32 from vector<2x8xf32>
    %229 = arith.select %206, %225, %226 : f32
    %shuffleResult_16, %valid_17 = gpu.shuffle xor %229, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %230 = arith.select %206, %shuffleResult_16, %225 : f32
    %231 = arith.select %206, %226, %shuffleResult_16 : f32
    %232 = arith.select %206, %227, %228 : f32
    %shuffleResult_18, %valid_19 = gpu.shuffle xor %232, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %233 = arith.select %206, %shuffleResult_18, %227 : f32
    %234 = arith.select %206, %228, %shuffleResult_18 : f32
    %235 = arith.select %208, %230, %233 : f32
    %shuffleResult_20, %valid_21 = gpu.shuffle xor %235, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %236 = arith.select %208, %shuffleResult_20, %230 : f32
    %237 = arith.select %208, %233, %shuffleResult_20 : f32
    %238 = arith.select %208, %231, %234 : f32
    %shuffleResult_22, %valid_23 = gpu.shuffle xor %238, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %239 = arith.select %208, %shuffleResult_22, %231 : f32
    %240 = arith.select %208, %234, %shuffleResult_22 : f32
    %241 = vector.extract %203#5[1, 0] : f32 from vector<2x8xf32>
    %242 = vector.extract %203#5[1, 1] : f32 from vector<2x8xf32>
    %243 = vector.extract %203#5[1, 2] : f32 from vector<2x8xf32>
    %244 = vector.extract %203#5[1, 3] : f32 from vector<2x8xf32>
    %245 = arith.select %206, %241, %242 : f32
    %shuffleResult_24, %valid_25 = gpu.shuffle xor %245, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %246 = arith.select %206, %shuffleResult_24, %241 : f32
    %247 = arith.select %206, %242, %shuffleResult_24 : f32
    %248 = arith.select %206, %243, %244 : f32
    %shuffleResult_26, %valid_27 = gpu.shuffle xor %248, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %249 = arith.select %206, %shuffleResult_26, %243 : f32
    %250 = arith.select %206, %244, %shuffleResult_26 : f32
    %251 = arith.select %208, %246, %249 : f32
    %shuffleResult_28, %valid_29 = gpu.shuffle xor %251, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %252 = arith.select %208, %shuffleResult_28, %246 : f32
    %253 = arith.select %208, %249, %shuffleResult_28 : f32
    %254 = arith.select %208, %247, %250 : f32
    %shuffleResult_30, %valid_31 = gpu.shuffle xor %254, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %255 = arith.select %208, %shuffleResult_30, %247 : f32
    %256 = arith.select %208, %250, %shuffleResult_30 : f32
    %257 = vector.extract %203#5[1, 4] : f32 from vector<2x8xf32>
    %258 = vector.extract %203#5[1, 5] : f32 from vector<2x8xf32>
    %259 = vector.extract %203#5[1, 6] : f32 from vector<2x8xf32>
    %260 = vector.extract %203#5[1, 7] : f32 from vector<2x8xf32>
    %261 = arith.select %206, %257, %258 : f32
    %shuffleResult_32, %valid_33 = gpu.shuffle xor %261, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %262 = arith.select %206, %shuffleResult_32, %257 : f32
    %263 = arith.select %206, %258, %shuffleResult_32 : f32
    %264 = arith.select %206, %259, %260 : f32
    %shuffleResult_34, %valid_35 = gpu.shuffle xor %264, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %265 = arith.select %206, %shuffleResult_34, %259 : f32
    %266 = arith.select %206, %260, %shuffleResult_34 : f32
    %267 = arith.select %208, %262, %265 : f32
    %shuffleResult_36, %valid_37 = gpu.shuffle xor %267, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %268 = arith.select %208, %shuffleResult_36, %262 : f32
    %269 = arith.select %208, %265, %shuffleResult_36 : f32
    %270 = arith.select %208, %263, %266 : f32
    %shuffleResult_38, %valid_39 = gpu.shuffle xor %270, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %271 = arith.select %208, %shuffleResult_38, %263 : f32
    %272 = arith.select %208, %266, %shuffleResult_38 : f32
    %273 = vector.extract %203#4[0, 0] : f32 from vector<2x8xf32>
    %274 = vector.extract %203#4[0, 1] : f32 from vector<2x8xf32>
    %275 = vector.extract %203#4[0, 2] : f32 from vector<2x8xf32>
    %276 = vector.extract %203#4[0, 3] : f32 from vector<2x8xf32>
    %277 = arith.select %206, %273, %274 : f32
    %shuffleResult_40, %valid_41 = gpu.shuffle xor %277, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %278 = arith.select %206, %shuffleResult_40, %273 : f32
    %279 = arith.select %206, %274, %shuffleResult_40 : f32
    %280 = arith.select %206, %275, %276 : f32
    %shuffleResult_42, %valid_43 = gpu.shuffle xor %280, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %281 = arith.select %206, %shuffleResult_42, %275 : f32
    %282 = arith.select %206, %276, %shuffleResult_42 : f32
    %283 = arith.select %208, %278, %281 : f32
    %shuffleResult_44, %valid_45 = gpu.shuffle xor %283, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %284 = arith.select %208, %shuffleResult_44, %278 : f32
    %285 = arith.select %208, %281, %shuffleResult_44 : f32
    %286 = arith.select %208, %279, %282 : f32
    %shuffleResult_46, %valid_47 = gpu.shuffle xor %286, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %287 = arith.select %208, %shuffleResult_46, %279 : f32
    %288 = arith.select %208, %282, %shuffleResult_46 : f32
    %289 = vector.extract %203#4[0, 4] : f32 from vector<2x8xf32>
    %290 = vector.extract %203#4[0, 5] : f32 from vector<2x8xf32>
    %291 = vector.extract %203#4[0, 6] : f32 from vector<2x8xf32>
    %292 = vector.extract %203#4[0, 7] : f32 from vector<2x8xf32>
    %293 = arith.select %206, %289, %290 : f32
    %shuffleResult_48, %valid_49 = gpu.shuffle xor %293, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %294 = arith.select %206, %shuffleResult_48, %289 : f32
    %295 = arith.select %206, %290, %shuffleResult_48 : f32
    %296 = arith.select %206, %291, %292 : f32
    %shuffleResult_50, %valid_51 = gpu.shuffle xor %296, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %297 = arith.select %206, %shuffleResult_50, %291 : f32
    %298 = arith.select %206, %292, %shuffleResult_50 : f32
    %299 = arith.select %208, %294, %297 : f32
    %shuffleResult_52, %valid_53 = gpu.shuffle xor %299, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %300 = arith.select %208, %shuffleResult_52, %294 : f32
    %301 = arith.select %208, %297, %shuffleResult_52 : f32
    %302 = arith.select %208, %295, %298 : f32
    %shuffleResult_54, %valid_55 = gpu.shuffle xor %302, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %303 = arith.select %208, %shuffleResult_54, %295 : f32
    %304 = arith.select %208, %298, %shuffleResult_54 : f32
    %305 = vector.extract %203#4[1, 0] : f32 from vector<2x8xf32>
    %306 = vector.extract %203#4[1, 1] : f32 from vector<2x8xf32>
    %307 = vector.extract %203#4[1, 2] : f32 from vector<2x8xf32>
    %308 = vector.extract %203#4[1, 3] : f32 from vector<2x8xf32>
    %309 = arith.select %206, %305, %306 : f32
    %shuffleResult_56, %valid_57 = gpu.shuffle xor %309, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %310 = arith.select %206, %shuffleResult_56, %305 : f32
    %311 = arith.select %206, %306, %shuffleResult_56 : f32
    %312 = arith.select %206, %307, %308 : f32
    %shuffleResult_58, %valid_59 = gpu.shuffle xor %312, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %313 = arith.select %206, %shuffleResult_58, %307 : f32
    %314 = arith.select %206, %308, %shuffleResult_58 : f32
    %315 = arith.select %208, %310, %313 : f32
    %shuffleResult_60, %valid_61 = gpu.shuffle xor %315, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %316 = arith.select %208, %shuffleResult_60, %310 : f32
    %317 = arith.select %208, %313, %shuffleResult_60 : f32
    %318 = arith.select %208, %311, %314 : f32
    %shuffleResult_62, %valid_63 = gpu.shuffle xor %318, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %319 = arith.select %208, %shuffleResult_62, %311 : f32
    %320 = arith.select %208, %314, %shuffleResult_62 : f32
    %321 = vector.extract %203#4[1, 4] : f32 from vector<2x8xf32>
    %322 = vector.extract %203#4[1, 5] : f32 from vector<2x8xf32>
    %323 = vector.extract %203#4[1, 6] : f32 from vector<2x8xf32>
    %324 = vector.extract %203#4[1, 7] : f32 from vector<2x8xf32>
    %325 = arith.select %206, %321, %322 : f32
    %shuffleResult_64, %valid_65 = gpu.shuffle xor %325, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %326 = arith.select %206, %shuffleResult_64, %321 : f32
    %327 = arith.select %206, %322, %shuffleResult_64 : f32
    %328 = arith.select %206, %323, %324 : f32
    %shuffleResult_66, %valid_67 = gpu.shuffle xor %328, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %329 = arith.select %206, %shuffleResult_66, %323 : f32
    %330 = arith.select %206, %324, %shuffleResult_66 : f32
    %331 = arith.select %208, %326, %329 : f32
    %shuffleResult_68, %valid_69 = gpu.shuffle xor %331, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %332 = arith.select %208, %shuffleResult_68, %326 : f32
    %333 = arith.select %208, %329, %shuffleResult_68 : f32
    %334 = arith.select %208, %327, %330 : f32
    %shuffleResult_70, %valid_71 = gpu.shuffle xor %334, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %335 = arith.select %208, %shuffleResult_70, %327 : f32
    %336 = arith.select %208, %330, %shuffleResult_70 : f32
    %337 = vector.extract %203#3[0, 0] : f32 from vector<2x8xf32>
    %338 = vector.extract %203#3[0, 1] : f32 from vector<2x8xf32>
    %339 = vector.extract %203#3[0, 2] : f32 from vector<2x8xf32>
    %340 = vector.extract %203#3[0, 3] : f32 from vector<2x8xf32>
    %341 = arith.select %206, %337, %338 : f32
    %shuffleResult_72, %valid_73 = gpu.shuffle xor %341, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %342 = arith.select %206, %shuffleResult_72, %337 : f32
    %343 = arith.select %206, %338, %shuffleResult_72 : f32
    %344 = arith.select %206, %339, %340 : f32
    %shuffleResult_74, %valid_75 = gpu.shuffle xor %344, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %345 = arith.select %206, %shuffleResult_74, %339 : f32
    %346 = arith.select %206, %340, %shuffleResult_74 : f32
    %347 = arith.select %208, %342, %345 : f32
    %shuffleResult_76, %valid_77 = gpu.shuffle xor %347, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %348 = arith.select %208, %shuffleResult_76, %342 : f32
    %349 = arith.select %208, %345, %shuffleResult_76 : f32
    %350 = arith.select %208, %343, %346 : f32
    %shuffleResult_78, %valid_79 = gpu.shuffle xor %350, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %351 = arith.select %208, %shuffleResult_78, %343 : f32
    %352 = arith.select %208, %346, %shuffleResult_78 : f32
    %353 = vector.extract %203#3[0, 4] : f32 from vector<2x8xf32>
    %354 = vector.extract %203#3[0, 5] : f32 from vector<2x8xf32>
    %355 = vector.extract %203#3[0, 6] : f32 from vector<2x8xf32>
    %356 = vector.extract %203#3[0, 7] : f32 from vector<2x8xf32>
    %357 = arith.select %206, %353, %354 : f32
    %shuffleResult_80, %valid_81 = gpu.shuffle xor %357, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %358 = arith.select %206, %shuffleResult_80, %353 : f32
    %359 = arith.select %206, %354, %shuffleResult_80 : f32
    %360 = arith.select %206, %355, %356 : f32
    %shuffleResult_82, %valid_83 = gpu.shuffle xor %360, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %361 = arith.select %206, %shuffleResult_82, %355 : f32
    %362 = arith.select %206, %356, %shuffleResult_82 : f32
    %363 = arith.select %208, %358, %361 : f32
    %shuffleResult_84, %valid_85 = gpu.shuffle xor %363, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %364 = arith.select %208, %shuffleResult_84, %358 : f32
    %365 = arith.select %208, %361, %shuffleResult_84 : f32
    %366 = arith.select %208, %359, %362 : f32
    %shuffleResult_86, %valid_87 = gpu.shuffle xor %366, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %367 = arith.select %208, %shuffleResult_86, %359 : f32
    %368 = arith.select %208, %362, %shuffleResult_86 : f32
    %369 = vector.extract %203#3[1, 0] : f32 from vector<2x8xf32>
    %370 = vector.extract %203#3[1, 1] : f32 from vector<2x8xf32>
    %371 = vector.extract %203#3[1, 2] : f32 from vector<2x8xf32>
    %372 = vector.extract %203#3[1, 3] : f32 from vector<2x8xf32>
    %373 = arith.select %206, %369, %370 : f32
    %shuffleResult_88, %valid_89 = gpu.shuffle xor %373, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %374 = arith.select %206, %shuffleResult_88, %369 : f32
    %375 = arith.select %206, %370, %shuffleResult_88 : f32
    %376 = arith.select %206, %371, %372 : f32
    %shuffleResult_90, %valid_91 = gpu.shuffle xor %376, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %377 = arith.select %206, %shuffleResult_90, %371 : f32
    %378 = arith.select %206, %372, %shuffleResult_90 : f32
    %379 = arith.select %208, %374, %377 : f32
    %shuffleResult_92, %valid_93 = gpu.shuffle xor %379, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %380 = arith.select %208, %shuffleResult_92, %374 : f32
    %381 = arith.select %208, %377, %shuffleResult_92 : f32
    %382 = arith.select %208, %375, %378 : f32
    %shuffleResult_94, %valid_95 = gpu.shuffle xor %382, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %383 = arith.select %208, %shuffleResult_94, %375 : f32
    %384 = arith.select %208, %378, %shuffleResult_94 : f32
    %385 = vector.extract %203#3[1, 4] : f32 from vector<2x8xf32>
    %386 = vector.extract %203#3[1, 5] : f32 from vector<2x8xf32>
    %387 = vector.extract %203#3[1, 6] : f32 from vector<2x8xf32>
    %388 = vector.extract %203#3[1, 7] : f32 from vector<2x8xf32>
    %389 = arith.select %206, %385, %386 : f32
    %shuffleResult_96, %valid_97 = gpu.shuffle xor %389, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %390 = arith.select %206, %shuffleResult_96, %385 : f32
    %391 = arith.select %206, %386, %shuffleResult_96 : f32
    %392 = arith.select %206, %387, %388 : f32
    %shuffleResult_98, %valid_99 = gpu.shuffle xor %392, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %393 = arith.select %206, %shuffleResult_98, %387 : f32
    %394 = arith.select %206, %388, %shuffleResult_98 : f32
    %395 = arith.select %208, %390, %393 : f32
    %shuffleResult_100, %valid_101 = gpu.shuffle xor %395, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %396 = arith.select %208, %shuffleResult_100, %390 : f32
    %397 = arith.select %208, %393, %shuffleResult_100 : f32
    %398 = arith.select %208, %391, %394 : f32
    %shuffleResult_102, %valid_103 = gpu.shuffle xor %398, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %399 = arith.select %208, %shuffleResult_102, %391 : f32
    %400 = arith.select %208, %394, %shuffleResult_102 : f32
    %401 = vector.extract %203#0[0, 0] : f32 from vector<2x8xf32>
    %402 = vector.extract %203#0[0, 1] : f32 from vector<2x8xf32>
    %403 = vector.extract %203#0[0, 2] : f32 from vector<2x8xf32>
    %404 = vector.extract %203#0[0, 3] : f32 from vector<2x8xf32>
    %405 = arith.select %206, %401, %402 : f32
    %shuffleResult_104, %valid_105 = gpu.shuffle xor %405, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %406 = arith.select %206, %shuffleResult_104, %401 : f32
    %407 = arith.select %206, %402, %shuffleResult_104 : f32
    %408 = arith.select %206, %403, %404 : f32
    %shuffleResult_106, %valid_107 = gpu.shuffle xor %408, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %409 = arith.select %206, %shuffleResult_106, %403 : f32
    %410 = arith.select %206, %404, %shuffleResult_106 : f32
    %411 = arith.select %208, %406, %409 : f32
    %shuffleResult_108, %valid_109 = gpu.shuffle xor %411, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %412 = arith.select %208, %shuffleResult_108, %406 : f32
    %413 = arith.select %208, %409, %shuffleResult_108 : f32
    %414 = arith.select %208, %407, %410 : f32
    %shuffleResult_110, %valid_111 = gpu.shuffle xor %414, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %415 = arith.select %208, %shuffleResult_110, %407 : f32
    %416 = arith.select %208, %410, %shuffleResult_110 : f32
    %417 = vector.extract %203#0[0, 4] : f32 from vector<2x8xf32>
    %418 = vector.extract %203#0[0, 5] : f32 from vector<2x8xf32>
    %419 = vector.extract %203#0[0, 6] : f32 from vector<2x8xf32>
    %420 = vector.extract %203#0[0, 7] : f32 from vector<2x8xf32>
    %421 = arith.select %206, %417, %418 : f32
    %shuffleResult_112, %valid_113 = gpu.shuffle xor %421, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %422 = arith.select %206, %shuffleResult_112, %417 : f32
    %423 = arith.select %206, %418, %shuffleResult_112 : f32
    %424 = arith.select %206, %419, %420 : f32
    %shuffleResult_114, %valid_115 = gpu.shuffle xor %424, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %425 = arith.select %206, %shuffleResult_114, %419 : f32
    %426 = arith.select %206, %420, %shuffleResult_114 : f32
    %427 = arith.select %208, %422, %425 : f32
    %shuffleResult_116, %valid_117 = gpu.shuffle xor %427, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %428 = arith.select %208, %shuffleResult_116, %422 : f32
    %429 = arith.select %208, %425, %shuffleResult_116 : f32
    %430 = arith.select %208, %423, %426 : f32
    %shuffleResult_118, %valid_119 = gpu.shuffle xor %430, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %431 = arith.select %208, %shuffleResult_118, %423 : f32
    %432 = arith.select %208, %426, %shuffleResult_118 : f32
    %433 = vector.extract %203#0[1, 0] : f32 from vector<2x8xf32>
    %434 = vector.extract %203#0[1, 1] : f32 from vector<2x8xf32>
    %435 = vector.extract %203#0[1, 2] : f32 from vector<2x8xf32>
    %436 = vector.extract %203#0[1, 3] : f32 from vector<2x8xf32>
    %437 = arith.select %206, %433, %434 : f32
    %shuffleResult_120, %valid_121 = gpu.shuffle xor %437, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %438 = arith.select %206, %shuffleResult_120, %433 : f32
    %439 = arith.select %206, %434, %shuffleResult_120 : f32
    %440 = arith.select %206, %435, %436 : f32
    %shuffleResult_122, %valid_123 = gpu.shuffle xor %440, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %441 = arith.select %206, %shuffleResult_122, %435 : f32
    %442 = arith.select %206, %436, %shuffleResult_122 : f32
    %443 = arith.select %208, %438, %441 : f32
    %shuffleResult_124, %valid_125 = gpu.shuffle xor %443, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %444 = arith.select %208, %shuffleResult_124, %438 : f32
    %445 = arith.select %208, %441, %shuffleResult_124 : f32
    %446 = arith.select %208, %439, %442 : f32
    %shuffleResult_126, %valid_127 = gpu.shuffle xor %446, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %447 = arith.select %208, %shuffleResult_126, %439 : f32
    %448 = arith.select %208, %442, %shuffleResult_126 : f32
    %449 = vector.extract %203#0[1, 4] : f32 from vector<2x8xf32>
    %450 = vector.extract %203#0[1, 5] : f32 from vector<2x8xf32>
    %451 = vector.extract %203#0[1, 6] : f32 from vector<2x8xf32>
    %452 = vector.extract %203#0[1, 7] : f32 from vector<2x8xf32>
    %453 = arith.select %206, %449, %450 : f32
    %shuffleResult_128, %valid_129 = gpu.shuffle xor %453, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %454 = arith.select %206, %shuffleResult_128, %449 : f32
    %455 = arith.select %206, %450, %shuffleResult_128 : f32
    %456 = arith.select %206, %451, %452 : f32
    %shuffleResult_130, %valid_131 = gpu.shuffle xor %456, %c16_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %457 = arith.select %206, %shuffleResult_130, %451 : f32
    %458 = arith.select %206, %452, %shuffleResult_130 : f32
    %459 = arith.select %208, %454, %457 : f32
    %shuffleResult_132, %valid_133 = gpu.shuffle xor %459, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %460 = arith.select %208, %shuffleResult_132, %454 : f32
    %461 = arith.select %208, %457, %shuffleResult_132 : f32
    %462 = arith.select %208, %455, %458 : f32
    %shuffleResult_134, %valid_135 = gpu.shuffle xor %462, %c32_i32, %c64_i32 {frisk.layout_role = "wmmaC"} : f32
    %463 = arith.select %208, %shuffleResult_134, %455 : f32
    %464 = arith.select %208, %458, %shuffleResult_134 : f32
    %465 = vector.extract %203#1[0, 0] : f32 from vector<2x1xf32>
    %466 = vector.from_elements %412, %415, %413, %416 : vector<4xf32>
    %467 = vector.broadcast %465 : f32 to vector<4xf32>
    %468 = arith.divf %466, %467 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %469 = arith.truncf %468 {tiled = true} : vector<4xf32> to vector<4xf16>
    %470 = vector.extract %469[0] : f16 from vector<4xf16>
    %471 = vector.extract %469[1] : f16 from vector<4xf16>
    %472 = vector.extract %469[2] : f16 from vector<4xf16>
    %473 = vector.extract %469[3] : f16 from vector<4xf16>
    %474 = vector.from_elements %428, %431, %429, %432 : vector<4xf32>
    %475 = arith.divf %474, %467 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %476 = arith.truncf %475 {tiled = true} : vector<4xf32> to vector<4xf16>
    %477 = vector.extract %476[0] : f16 from vector<4xf16>
    %478 = vector.extract %476[1] : f16 from vector<4xf16>
    %479 = vector.extract %476[2] : f16 from vector<4xf16>
    %480 = vector.extract %476[3] : f16 from vector<4xf16>
    %481 = vector.extract %203#1[1, 0] : f32 from vector<2x1xf32>
    %482 = vector.from_elements %444, %447, %445, %448 : vector<4xf32>
    %483 = vector.broadcast %481 : f32 to vector<4xf32>
    %484 = arith.divf %482, %483 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %485 = arith.truncf %484 {tiled = true} : vector<4xf32> to vector<4xf16>
    %486 = vector.extract %485[0] : f16 from vector<4xf16>
    %487 = vector.extract %485[1] : f16 from vector<4xf16>
    %488 = vector.extract %485[2] : f16 from vector<4xf16>
    %489 = vector.extract %485[3] : f16 from vector<4xf16>
    %490 = vector.from_elements %460, %463, %461, %464 : vector<4xf32>
    %491 = arith.divf %490, %483 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %492 = arith.truncf %491 {tiled = true} : vector<4xf32> to vector<4xf16>
    %493 = vector.extract %492[0] : f16 from vector<4xf16>
    %494 = vector.extract %492[1] : f16 from vector<4xf16>
    %495 = vector.extract %492[2] : f16 from vector<4xf16>
    %496 = vector.extract %492[3] : f16 from vector<4xf16>
    %497 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %498 = arith.cmpi ult, %497, %c4 : index
    scf.if %498 {
      affine.store %470, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
      affine.store %471, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
      affine.store %472, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
      affine.store %473, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      affine.store %477, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 16] : memref<1x32x4096x128xf16, 1>
      affine.store %478, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 17] : memref<1x32x4096x128xf16, 1>
      affine.store %479, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 18] : memref<1x32x4096x128xf16, 1>
      affine.store %480, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 19] : memref<1x32x4096x128xf16, 1>
      affine.store %486, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
      affine.store %487, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
      affine.store %488, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
      affine.store %489, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      affine.store %493, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 16] : memref<1x32x4096x128xf16, 1>
      affine.store %494, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 17] : memref<1x32x4096x128xf16, 1>
      affine.store %495, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 18] : memref<1x32x4096x128xf16, 1>
      affine.store %496, %arg3[0, (symbol(%8) + symbol(%block_id_z) * 32 + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 19] : memref<1x32x4096x128xf16, 1>
    }
    %499 = vector.from_elements %348, %351, %349, %352 : vector<4xf32>
    %500 = arith.divf %499, %467 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %501 = arith.truncf %500 {tiled = true} : vector<4xf32> to vector<4xf16>
    %502 = vector.extract %501[0] : f16 from vector<4xf16>
    %503 = vector.extract %501[1] : f16 from vector<4xf16>
    %504 = vector.extract %501[2] : f16 from vector<4xf16>
    %505 = vector.extract %501[3] : f16 from vector<4xf16>
    %506 = vector.from_elements %364, %367, %365, %368 : vector<4xf32>
    %507 = arith.divf %506, %467 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %508 = arith.truncf %507 {tiled = true} : vector<4xf32> to vector<4xf16>
    %509 = vector.extract %508[0] : f16 from vector<4xf16>
    %510 = vector.extract %508[1] : f16 from vector<4xf16>
    %511 = vector.extract %508[2] : f16 from vector<4xf16>
    %512 = vector.extract %508[3] : f16 from vector<4xf16>
    %513 = vector.from_elements %380, %383, %381, %384 : vector<4xf32>
    %514 = arith.divf %513, %483 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %515 = arith.truncf %514 {tiled = true} : vector<4xf32> to vector<4xf16>
    %516 = vector.extract %515[0] : f16 from vector<4xf16>
    %517 = vector.extract %515[1] : f16 from vector<4xf16>
    %518 = vector.extract %515[2] : f16 from vector<4xf16>
    %519 = vector.extract %515[3] : f16 from vector<4xf16>
    %520 = vector.from_elements %396, %399, %397, %400 : vector<4xf32>
    %521 = arith.divf %520, %483 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %522 = arith.truncf %521 {tiled = true} : vector<4xf32> to vector<4xf16>
    %523 = vector.extract %522[0] : f16 from vector<4xf16>
    %524 = vector.extract %522[1] : f16 from vector<4xf16>
    %525 = vector.extract %522[2] : f16 from vector<4xf16>
    %526 = vector.extract %522[3] : f16 from vector<4xf16>
    scf.if %498 {
      affine.store %502, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
      affine.store %503, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
      affine.store %504, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
      affine.store %505, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      affine.store %509, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 48] : memref<1x32x4096x128xf16, 1>
      affine.store %510, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 49] : memref<1x32x4096x128xf16, 1>
      affine.store %511, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 50] : memref<1x32x4096x128xf16, 1>
      affine.store %512, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 51] : memref<1x32x4096x128xf16, 1>
      affine.store %516, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
      affine.store %517, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
      affine.store %518, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
      affine.store %519, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      affine.store %523, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 48] : memref<1x32x4096x128xf16, 1>
      affine.store %524, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 49] : memref<1x32x4096x128xf16, 1>
      affine.store %525, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 50] : memref<1x32x4096x128xf16, 1>
      affine.store %526, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 51] : memref<1x32x4096x128xf16, 1>
    }
    %527 = vector.from_elements %284, %287, %285, %288 : vector<4xf32>
    %528 = arith.divf %527, %467 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %529 = arith.truncf %528 {tiled = true} : vector<4xf32> to vector<4xf16>
    %530 = vector.extract %529[0] : f16 from vector<4xf16>
    %531 = vector.extract %529[1] : f16 from vector<4xf16>
    %532 = vector.extract %529[2] : f16 from vector<4xf16>
    %533 = vector.extract %529[3] : f16 from vector<4xf16>
    %534 = vector.from_elements %300, %303, %301, %304 : vector<4xf32>
    %535 = arith.divf %534, %467 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %536 = arith.truncf %535 {tiled = true} : vector<4xf32> to vector<4xf16>
    %537 = vector.extract %536[0] : f16 from vector<4xf16>
    %538 = vector.extract %536[1] : f16 from vector<4xf16>
    %539 = vector.extract %536[2] : f16 from vector<4xf16>
    %540 = vector.extract %536[3] : f16 from vector<4xf16>
    %541 = vector.from_elements %316, %319, %317, %320 : vector<4xf32>
    %542 = arith.divf %541, %483 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %543 = arith.truncf %542 {tiled = true} : vector<4xf32> to vector<4xf16>
    %544 = vector.extract %543[0] : f16 from vector<4xf16>
    %545 = vector.extract %543[1] : f16 from vector<4xf16>
    %546 = vector.extract %543[2] : f16 from vector<4xf16>
    %547 = vector.extract %543[3] : f16 from vector<4xf16>
    %548 = vector.from_elements %332, %335, %333, %336 : vector<4xf32>
    %549 = arith.divf %548, %483 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %550 = arith.truncf %549 {tiled = true} : vector<4xf32> to vector<4xf16>
    %551 = vector.extract %550[0] : f16 from vector<4xf16>
    %552 = vector.extract %550[1] : f16 from vector<4xf16>
    %553 = vector.extract %550[2] : f16 from vector<4xf16>
    %554 = vector.extract %550[3] : f16 from vector<4xf16>
    scf.if %498 {
      affine.store %530, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
      affine.store %531, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
      affine.store %532, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
      affine.store %533, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      affine.store %537, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 80] : memref<1x32x4096x128xf16, 1>
      affine.store %538, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 81] : memref<1x32x4096x128xf16, 1>
      affine.store %539, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 82] : memref<1x32x4096x128xf16, 1>
      affine.store %540, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 83] : memref<1x32x4096x128xf16, 1>
      affine.store %544, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
      affine.store %545, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
      affine.store %546, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
      affine.store %547, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      affine.store %551, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 80] : memref<1x32x4096x128xf16, 1>
      affine.store %552, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 81] : memref<1x32x4096x128xf16, 1>
      affine.store %553, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 82] : memref<1x32x4096x128xf16, 1>
      affine.store %554, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 83] : memref<1x32x4096x128xf16, 1>
    }
    %555 = vector.from_elements %220, %223, %221, %224 : vector<4xf32>
    %556 = arith.divf %555, %467 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %557 = arith.truncf %556 {tiled = true} : vector<4xf32> to vector<4xf16>
    %558 = vector.extract %557[0] : f16 from vector<4xf16>
    %559 = vector.extract %557[1] : f16 from vector<4xf16>
    %560 = vector.extract %557[2] : f16 from vector<4xf16>
    %561 = vector.extract %557[3] : f16 from vector<4xf16>
    %562 = vector.from_elements %236, %239, %237, %240 : vector<4xf32>
    %563 = arith.divf %562, %467 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %564 = arith.truncf %563 {tiled = true} : vector<4xf32> to vector<4xf16>
    %565 = vector.extract %564[0] : f16 from vector<4xf16>
    %566 = vector.extract %564[1] : f16 from vector<4xf16>
    %567 = vector.extract %564[2] : f16 from vector<4xf16>
    %568 = vector.extract %564[3] : f16 from vector<4xf16>
    %569 = vector.from_elements %252, %255, %253, %256 : vector<4xf32>
    %570 = arith.divf %569, %483 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %571 = arith.truncf %570 {tiled = true} : vector<4xf32> to vector<4xf16>
    %572 = vector.extract %571[0] : f16 from vector<4xf16>
    %573 = vector.extract %571[1] : f16 from vector<4xf16>
    %574 = vector.extract %571[2] : f16 from vector<4xf16>
    %575 = vector.extract %571[3] : f16 from vector<4xf16>
    %576 = vector.from_elements %268, %271, %269, %272 : vector<4xf32>
    %577 = arith.divf %576, %483 {origin = "frisk.div", tiled = true} : vector<4xf32>
    %578 = arith.truncf %577 {tiled = true} : vector<4xf32> to vector<4xf16>
    %579 = vector.extract %578[0] : f16 from vector<4xf16>
    %580 = vector.extract %578[1] : f16 from vector<4xf16>
    %581 = vector.extract %578[2] : f16 from vector<4xf16>
    %582 = vector.extract %578[3] : f16 from vector<4xf16>
    scf.if %498 {
      affine.store %558, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
      affine.store %559, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
      affine.store %560, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
      affine.store %561, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      affine.store %565, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 112] : memref<1x32x4096x128xf16, 1>
      affine.store %566, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 113] : memref<1x32x4096x128xf16, 1>
      affine.store %567, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 114] : memref<1x32x4096x128xf16, 1>
      affine.store %568, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 115] : memref<1x32x4096x128xf16, 1>
      affine.store %572, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
      affine.store %573, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
      affine.store %574, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
      affine.store %575, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      affine.store %579, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 112] : memref<1x32x4096x128xf16, 1>
      affine.store %580, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 113] : memref<1x32x4096x128xf16, 1>
      affine.store %581, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 114] : memref<1x32x4096x128xf16, 1>
      affine.store %582, %arg3[0, (symbol(%8) + symbol(%7) floordiv 32) mod 32, symbol(%7) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%7) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 115] : memref<1x32x4096x128xf16, 1>
    }
    return
  }
}
