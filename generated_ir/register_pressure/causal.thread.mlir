module {
  memref.global "public" @shm_0 : memref<16384xi8, 3> = uninitialized alignment = 16
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 4, 0, 1, 1>, gridDim = [1 : i32, 32 : i32, 32 : i32], thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %c-1_i32 = arith.constant -1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %2 = rocdl.mbcnt.lo %c-1_i32, %c0_i32 : (i32, i32) -> i32
    %3 = rocdl.mbcnt.hi %c-1_i32, %2 : (i32, i32) -> i32
    %4 = llvm.add %3, %c64_i32 : i32
    %5 = llvm.xor %3, %c16_i32 : i32
    %6 = llvm.sub %1, %c64_i32 : i32
    %7 = llvm.and %4, %6 : i32
    %8 = llvm.icmp "slt" %5, %7 : i32
    %9 = llvm.select %8, %5, %3 : i1, i32
    %10 = llvm.shl %9, %0 : i32
    %11 = llvm.xor %3, %c32_i32 : i32
    %12 = llvm.icmp "slt" %11, %7 : i32
    %13 = llvm.select %12, %11, %3 : i1, i32
    %14 = llvm.shl %13, %0 : i32
    %c256 = arith.constant 256 : index
    %15 = llvm.mlir.constant(128 : i64) : i64
    %c4096 = arith.constant 4096 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant dense<1.000000e+00> : vector<4xf32>
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<2x1xf32>
    %c8192 = arith.constant 8192 : index
    %cst_2 = arith.constant 0.127531052 : f32
    %cst_3 = arith.constant 0xFF800000 : f32
    %cst_4 = arith.constant dense<0.000000e+00> : vector<2x1xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<2x8xf32>
    %c1_i32 = arith.constant 1 : i32
    %cst_6 = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c31 = arith.constant 31 : index
    %16 = memref.get_global @shm_0 : memref<16384xi8, 3>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %17 = arith.muli %block_id_y, %c32 : index
    %18 = arith.addi %block_id_x, %17 : index
    %19 = arith.divui %18, %c128 : index
    %20 = arith.muli %19, %c4 : index
    %21 = arith.remui %18, %c128 : index
    %22 = arith.remui %21, %c4 : index
    %23 = arith.addi %20, %22 : index
    %24 = arith.subi %c31, %23 : index
    %25 = arith.divui %21, %c4 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %26 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 + s2 * 32 + s0 floordiv 32) mod 32)>()[%24, %25, %block_id_z]
    %27 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%24]
    %subview = memref.subview %arg0[0, %26, %27, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %28 = builtin.unrealized_conversion_cast %subview : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %view = memref.view %16[%c0][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %29 = builtin.unrealized_conversion_cast %view : memref<4096xf16, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %thread_id_x = gpu.thread_id x
    %30 = arith.muli %thread_id_x, %c8 : index
    %31 = affine.apply affine_map<()[s0] -> ((s0 floordiv 8) mod 128)>()[%30]
    %32 = builtin.unrealized_conversion_cast %31 : index to i64
    %33 = affine.apply affine_map<()[s0] -> (s0 + (s0 floordiv 1024) * 8 - (s0 floordiv 8) * 8)>()[%30]
    %34 = builtin.unrealized_conversion_cast %33 : index to i64
    %35 = arith.divui %thread_id_x, %c64 : index
    %36 = arith.muli %35, %c512 : index
    %37 = arith.remui %36, %c4096 : index
    %38 = builtin.unrealized_conversion_cast %37 : index to i64
    %39 = llvm.extractvalue %28[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.extractvalue %28[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.getelementptr %39[%40] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %42 = llvm.mul %32, %15 : i64
    %43 = llvm.add %42, %34 : i64
    %44 = llvm.getelementptr %41[%43] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %45 = llvm.extractvalue %29[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.getelementptr %45[%38] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %47 = llvm.ptrtoint %46 : !llvm.ptr<3> to i32
    %48 = rocdl.readfirstlane %47 : i32
    %49 = llvm.inttoptr %48 : i32 to !llvm.ptr<3>
    rocdl.global.load.async.lds %44, %49, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    %50 = arith.addi %thread_id_x, %c256 : index
    %51 = arith.muli %50, %c8 : index
    %52 = affine.apply affine_map<()[s0] -> ((s0 floordiv 8) mod 128)>()[%51]
    %53 = builtin.unrealized_conversion_cast %52 : index to i64
    %54 = affine.apply affine_map<()[s0] -> (s0 + (s0 floordiv 1024) * 8 - (s0 floordiv 8) * 8)>()[%51]
    %55 = builtin.unrealized_conversion_cast %54 : index to i64
    %56 = arith.divui %50, %c64 : index
    %57 = arith.muli %56, %c512 : index
    %58 = arith.remui %57, %c4096 : index
    %59 = builtin.unrealized_conversion_cast %58 : index to i64
    %60 = llvm.mul %53, %15 : i64
    %61 = llvm.add %60, %55 : i64
    %62 = llvm.getelementptr %41[%61] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %63 = llvm.getelementptr %45[%59] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %64 = llvm.ptrtoint %63 : !llvm.ptr<3> to i32
    %65 = rocdl.readfirstlane %64 : i32
    %66 = llvm.inttoptr %65 : i32 to !llvm.ptr<3>
    rocdl.global.load.async.lds %62, %66, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    rocdl.asyncmark
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %67 = arith.truncf %cst_2 : f32 to f16
    %68 = arith.muli %24, %c128 : index
    rocdl.wait.asyncmark 0
    gpu.barrier
    %view_7 = memref.view %16[%c8192][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %69 = builtin.unrealized_conversion_cast %view_7 : memref<4096xf16, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %70 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%24, %25]
    %subview_8 = memref.subview %arg0[0, %70, %27, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %71 = builtin.unrealized_conversion_cast %subview_8 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %72 = llvm.extractvalue %71[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.extractvalue %71[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.getelementptr %72[%73] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %75 = llvm.getelementptr %74[%43] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %76 = llvm.extractvalue %69[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.getelementptr %76[%38] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %78 = llvm.ptrtoint %77 : !llvm.ptr<3> to i32
    %79 = rocdl.readfirstlane %78 : i32
    %80 = llvm.inttoptr %79 : i32 to !llvm.ptr<3>
    rocdl.global.load.async.lds %75, %80, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    %81 = llvm.getelementptr %74[%61] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %82 = llvm.getelementptr %76[%59] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %83 = llvm.ptrtoint %82 : !llvm.ptr<3> to i32
    %84 = rocdl.readfirstlane %83 : i32
    %85 = llvm.inttoptr %84 : i32 to !llvm.ptr<3>
    rocdl.global.load.async.lds %81, %85, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    rocdl.asyncmark
    %86 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128)>()[%thread_id_x]
    %87 = vector.load %view[%86] : memref<4096xf16, 3>, vector<4xf16>
    %88 = vector.broadcast %67 : f16 to vector<4xf16>
    %89 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>()[%thread_id_x]
    %90 = vector.load %view[%89] : memref<4096xf16, 3>, vector<4xf16>
    %91 = vector.extract %87[0] : f16 from vector<4xf16>
    %92 = vector.extract %87[1] : f16 from vector<4xf16>
    %93 = vector.extract %87[2] : f16 from vector<4xf16>
    %94 = vector.extract %87[3] : f16 from vector<4xf16>
    %95 = vector.extract %90[0] : f16 from vector<4xf16>
    %96 = vector.extract %90[1] : f16 from vector<4xf16>
    %97 = vector.extract %90[2] : f16 from vector<4xf16>
    %98 = vector.extract %90[3] : f16 from vector<4xf16>
    %99 = vector.from_elements %91, %92, %93, %94 : vector<4xf16>
    %100 = arith.mulf %99, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %101 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 512)>()[%thread_id_x]
    %102 = vector.load %view[%101] : memref<4096xf16, 3>, vector<4xf16>
    %103 = vector.from_elements %95, %96, %97, %98 : vector<4xf16>
    %104 = arith.mulf %103, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %105 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2560)>()[%thread_id_x]
    %106 = vector.load %view[%105] : memref<4096xf16, 3>, vector<4xf16>
    %107 = vector.extract %100[0] : f16 from vector<4xf16>
    %108 = vector.extract %102[0] : f16 from vector<4xf16>
    %109 = vector.extract %102[1] : f16 from vector<4xf16>
    %110 = vector.extract %102[2] : f16 from vector<4xf16>
    %111 = vector.extract %102[3] : f16 from vector<4xf16>
    %112 = vector.extract %106[0] : f16 from vector<4xf16>
    %113 = vector.extract %106[1] : f16 from vector<4xf16>
    %114 = vector.extract %106[2] : f16 from vector<4xf16>
    %115 = vector.extract %106[3] : f16 from vector<4xf16>
    %116 = vector.from_elements %108, %109, %110, %111 : vector<4xf16>
    %117 = arith.mulf %116, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %118 = vector.extract %100[1] : f16 from vector<4xf16>
    %119 = vector.extract %100[2] : f16 from vector<4xf16>
    %120 = vector.extract %100[3] : f16 from vector<4xf16>
    %121 = vector.extract %104[0] : f16 from vector<4xf16>
    %122 = vector.extract %104[1] : f16 from vector<4xf16>
    %123 = vector.extract %104[2] : f16 from vector<4xf16>
    %124 = vector.extract %104[3] : f16 from vector<4xf16>
    %125 = vector.extract %117[0] : f16 from vector<4xf16>
    %126 = vector.extract %117[1] : f16 from vector<4xf16>
    %127 = vector.extract %117[2] : f16 from vector<4xf16>
    %128 = vector.extract %117[3] : f16 from vector<4xf16>
    %129 = vector.from_elements %112, %113, %114, %115 : vector<4xf16>
    %130 = arith.mulf %129, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %131 = vector.extract %130[0] : f16 from vector<4xf16>
    %132 = vector.extract %130[1] : f16 from vector<4xf16>
    %133 = vector.extract %130[2] : f16 from vector<4xf16>
    %134 = vector.extract %130[3] : f16 from vector<4xf16>
    rocdl.wait.asyncmark 0
    gpu.barrier
    %subview_9 = memref.subview %arg0[0, %70, %27, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %135 = builtin.unrealized_conversion_cast %subview_9 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %136 = llvm.extractvalue %135[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.extractvalue %135[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.getelementptr %136[%137] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %139 = llvm.getelementptr %138[%43] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    rocdl.global.load.async.lds %139, %49, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    %140 = llvm.getelementptr %138[%61] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    rocdl.global.load.async.lds %140, %66, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    rocdl.asyncmark
    %141 = vector.load %view_7[%86] : memref<4096xf16, 3>, vector<4xf16>
    %142 = vector.load %view_7[%89] : memref<4096xf16, 3>, vector<4xf16>
    %143 = vector.load %view_7[%101] : memref<4096xf16, 3>, vector<4xf16>
    %144 = vector.extract %141[0] : f16 from vector<4xf16>
    %145 = vector.extract %141[1] : f16 from vector<4xf16>
    %146 = vector.extract %141[2] : f16 from vector<4xf16>
    %147 = vector.extract %141[3] : f16 from vector<4xf16>
    %148 = vector.extract %142[0] : f16 from vector<4xf16>
    %149 = vector.extract %142[1] : f16 from vector<4xf16>
    %150 = vector.extract %142[2] : f16 from vector<4xf16>
    %151 = vector.extract %142[3] : f16 from vector<4xf16>
    %152 = vector.from_elements %144, %145, %146, %147 : vector<4xf16>
    %153 = arith.mulf %152, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %154 = vector.load %view_7[%105] : memref<4096xf16, 3>, vector<4xf16>
    %155 = vector.extract %143[0] : f16 from vector<4xf16>
    %156 = vector.extract %143[1] : f16 from vector<4xf16>
    %157 = vector.extract %143[2] : f16 from vector<4xf16>
    %158 = vector.extract %143[3] : f16 from vector<4xf16>
    %159 = vector.from_elements %148, %149, %150, %151 : vector<4xf16>
    %160 = arith.mulf %159, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %161 = vector.extract %153[0] : f16 from vector<4xf16>
    %162 = vector.extract %153[1] : f16 from vector<4xf16>
    %163 = vector.extract %153[2] : f16 from vector<4xf16>
    %164 = vector.extract %153[3] : f16 from vector<4xf16>
    %165 = vector.extract %160[0] : f16 from vector<4xf16>
    %166 = vector.extract %160[1] : f16 from vector<4xf16>
    %167 = vector.extract %160[2] : f16 from vector<4xf16>
    %168 = vector.extract %160[3] : f16 from vector<4xf16>
    %169 = vector.extract %154[0] : f16 from vector<4xf16>
    %170 = vector.extract %154[1] : f16 from vector<4xf16>
    %171 = vector.extract %154[2] : f16 from vector<4xf16>
    %172 = vector.extract %154[3] : f16 from vector<4xf16>
    %173 = vector.from_elements %155, %156, %157, %158 : vector<4xf16>
    %174 = arith.mulf %173, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %175 = vector.from_elements %169, %170, %171, %172 : vector<4xf16>
    %176 = arith.mulf %175, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %177 = vector.extract %174[0] : f16 from vector<4xf16>
    %178 = vector.extract %174[1] : f16 from vector<4xf16>
    %179 = vector.extract %174[2] : f16 from vector<4xf16>
    %180 = vector.extract %174[3] : f16 from vector<4xf16>
    %181 = vector.extract %176[0] : f16 from vector<4xf16>
    %182 = vector.extract %176[1] : f16 from vector<4xf16>
    %183 = vector.extract %176[2] : f16 from vector<4xf16>
    %184 = vector.extract %176[3] : f16 from vector<4xf16>
    rocdl.wait.asyncmark 0
    gpu.barrier
    %subview_10 = memref.subview %arg0[0, %70, %27, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %185 = builtin.unrealized_conversion_cast %subview_10 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %186 = llvm.extractvalue %185[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %187 = llvm.extractvalue %185[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %188 = llvm.getelementptr %186[%187] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %189 = llvm.getelementptr %188[%43] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    rocdl.global.load.async.lds %189, %80, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    %190 = llvm.getelementptr %188[%61] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    rocdl.global.load.async.lds %190, %85, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    rocdl.asyncmark
    %191 = vector.load %view[%86] : memref<4096xf16, 3>, vector<4xf16>
    %192 = vector.load %view[%89] : memref<4096xf16, 3>, vector<4xf16>
    %193 = vector.load %view[%101] : memref<4096xf16, 3>, vector<4xf16>
    %194 = vector.extract %191[0] : f16 from vector<4xf16>
    %195 = vector.extract %191[1] : f16 from vector<4xf16>
    %196 = vector.extract %191[2] : f16 from vector<4xf16>
    %197 = vector.extract %191[3] : f16 from vector<4xf16>
    %198 = vector.extract %192[0] : f16 from vector<4xf16>
    %199 = vector.extract %192[1] : f16 from vector<4xf16>
    %200 = vector.extract %192[2] : f16 from vector<4xf16>
    %201 = vector.extract %192[3] : f16 from vector<4xf16>
    %202 = vector.from_elements %194, %195, %196, %197 : vector<4xf16>
    %203 = arith.mulf %202, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %204 = vector.load %view[%105] : memref<4096xf16, 3>, vector<4xf16>
    %205 = vector.extract %193[0] : f16 from vector<4xf16>
    %206 = vector.extract %193[1] : f16 from vector<4xf16>
    %207 = vector.extract %193[2] : f16 from vector<4xf16>
    %208 = vector.extract %193[3] : f16 from vector<4xf16>
    %209 = vector.from_elements %198, %199, %200, %201 : vector<4xf16>
    %210 = arith.mulf %209, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %211 = vector.extract %203[0] : f16 from vector<4xf16>
    %212 = vector.extract %203[1] : f16 from vector<4xf16>
    %213 = vector.extract %203[2] : f16 from vector<4xf16>
    %214 = vector.extract %203[3] : f16 from vector<4xf16>
    %215 = vector.extract %210[0] : f16 from vector<4xf16>
    %216 = vector.extract %210[1] : f16 from vector<4xf16>
    %217 = vector.extract %210[2] : f16 from vector<4xf16>
    %218 = vector.extract %210[3] : f16 from vector<4xf16>
    %219 = vector.extract %204[0] : f16 from vector<4xf16>
    %220 = vector.extract %204[1] : f16 from vector<4xf16>
    %221 = vector.extract %204[2] : f16 from vector<4xf16>
    %222 = vector.extract %204[3] : f16 from vector<4xf16>
    %223 = vector.from_elements %205, %206, %207, %208 : vector<4xf16>
    %224 = arith.mulf %223, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %225 = vector.from_elements %219, %220, %221, %222 : vector<4xf16>
    %226 = arith.mulf %225, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %227 = vector.extract %224[0] : f16 from vector<4xf16>
    %228 = vector.extract %224[1] : f16 from vector<4xf16>
    %229 = vector.extract %224[2] : f16 from vector<4xf16>
    %230 = vector.extract %224[3] : f16 from vector<4xf16>
    %231 = vector.extract %226[0] : f16 from vector<4xf16>
    %232 = vector.extract %226[1] : f16 from vector<4xf16>
    %233 = vector.extract %226[2] : f16 from vector<4xf16>
    %234 = vector.extract %226[3] : f16 from vector<4xf16>
    rocdl.wait.asyncmark 0
    gpu.barrier
    %235 = vector.load %view_7[%86] : memref<4096xf16, 3>, vector<4xf16>
    %236 = vector.load %view_7[%89] : memref<4096xf16, 3>, vector<4xf16>
    %237 = vector.load %view_7[%101] : memref<4096xf16, 3>, vector<4xf16>
    %238 = vector.extract %235[0] : f16 from vector<4xf16>
    %239 = vector.extract %235[1] : f16 from vector<4xf16>
    %240 = vector.extract %235[2] : f16 from vector<4xf16>
    %241 = vector.extract %235[3] : f16 from vector<4xf16>
    %242 = vector.extract %236[0] : f16 from vector<4xf16>
    %243 = vector.extract %236[1] : f16 from vector<4xf16>
    %244 = vector.extract %236[2] : f16 from vector<4xf16>
    %245 = vector.extract %236[3] : f16 from vector<4xf16>
    %246 = vector.from_elements %238, %239, %240, %241 : vector<4xf16>
    %247 = arith.mulf %246, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %248 = vector.load %view_7[%105] : memref<4096xf16, 3>, vector<4xf16>
    %249 = vector.extract %237[0] : f16 from vector<4xf16>
    %250 = vector.extract %237[1] : f16 from vector<4xf16>
    %251 = vector.extract %237[2] : f16 from vector<4xf16>
    %252 = vector.extract %237[3] : f16 from vector<4xf16>
    %253 = vector.from_elements %242, %243, %244, %245 : vector<4xf16>
    %254 = arith.mulf %253, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %255 = vector.extract %247[0] : f16 from vector<4xf16>
    %256 = vector.extract %247[1] : f16 from vector<4xf16>
    %257 = vector.extract %247[2] : f16 from vector<4xf16>
    %258 = vector.extract %247[3] : f16 from vector<4xf16>
    %259 = vector.extract %254[0] : f16 from vector<4xf16>
    %260 = vector.extract %254[1] : f16 from vector<4xf16>
    %261 = vector.extract %254[2] : f16 from vector<4xf16>
    %262 = vector.extract %254[3] : f16 from vector<4xf16>
    %263 = vector.extract %248[0] : f16 from vector<4xf16>
    %264 = vector.extract %248[1] : f16 from vector<4xf16>
    %265 = vector.extract %248[2] : f16 from vector<4xf16>
    %266 = vector.extract %248[3] : f16 from vector<4xf16>
    %267 = vector.from_elements %249, %250, %251, %252 : vector<4xf16>
    %268 = arith.mulf %267, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %269 = vector.from_elements %263, %264, %265, %266 : vector<4xf16>
    %270 = arith.mulf %269, %88 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %271 = vector.extract %268[0] : f16 from vector<4xf16>
    %272 = vector.extract %268[1] : f16 from vector<4xf16>
    %273 = vector.extract %268[2] : f16 from vector<4xf16>
    %274 = vector.extract %268[3] : f16 from vector<4xf16>
    %275 = vector.extract %270[0] : f16 from vector<4xf16>
    %276 = vector.extract %270[1] : f16 from vector<4xf16>
    %277 = vector.extract %270[2] : f16 from vector<4xf16>
    %278 = vector.extract %270[3] : f16 from vector<4xf16>
    gpu.barrier
    %279 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
    %280 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
    %281 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
    %282 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
    %283:6 = affine.for %arg4 = 0 to affine_map<()[s0] -> (s0 + 1)>()[%24] iter_args(%arg5 = %cst_5, %arg6 = %cst_4, %arg7 = %cst_1, %arg8 = %cst_5, %arg9 = %cst_5, %arg10 = %cst_5) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %857 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 floordiv 4096 + s0 + s1 * 32) mod 32)>(%arg4)[%25, %block_id_z]
      %858 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 128)>(%arg4)
      %859 = affine.apply affine_map<(d0) -> (d0 * 128 - (d0 floordiv 32) * 4096)>(%arg4)
      %subview_11 = memref.subview %transpose[0, %857, %858, %859] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %860 = builtin.unrealized_conversion_cast %subview_11 : memref<32x128xf16, strided<[1, 128], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %861 = affine.apply affine_map<()[s0] -> (s0 mod 8 + ((s0 + ((s0 floordiv 8) floordiv 16) * 32 - (((s0 floordiv 8) floordiv 16) floordiv 2) * 64 - ((((s0 floordiv 8) floordiv 16) mod 2 + ((s0 floordiv 8) floordiv 4) mod 2) floordiv 2) * 64) floordiv 1024) * 8)>()[%30]
      %862 = builtin.unrealized_conversion_cast %861 : index to i64
      %863 = affine.apply affine_map<()[s0] -> ((s0 floordiv 8 + ((s0 floordiv 8) floordiv 16) * 4 - (((s0 floordiv 8) floordiv 16) floordiv 2) * 8 - ((((s0 floordiv 8) floordiv 16) mod 2 + ((s0 floordiv 8) floordiv 4) mod 2) floordiv 2) * 8) mod 128)>()[%30]
      %864 = builtin.unrealized_conversion_cast %863 : index to i64
      %865 = llvm.extractvalue %860[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %866 = llvm.extractvalue %860[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %867 = llvm.getelementptr %865[%866] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %868 = llvm.mul %864, %15 : i64
      %869 = llvm.add %862, %868 : i64
      %870 = llvm.getelementptr %867[%869] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %870, %49, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %871 = affine.apply affine_map<()[s0] -> (s0 mod 8 + ((s0 + ((s0 floordiv 8) floordiv 16) * 32 - (((s0 floordiv 8) floordiv 16) floordiv 2) * 64 - ((((s0 floordiv 8) floordiv 16) mod 2 + ((s0 floordiv 8) floordiv 4) mod 2) floordiv 2) * 64) floordiv 1024) * 8)>()[%51]
      %872 = builtin.unrealized_conversion_cast %871 : index to i64
      %873 = affine.apply affine_map<()[s0] -> ((s0 floordiv 8 + ((s0 floordiv 8) floordiv 16) * 4 - (((s0 floordiv 8) floordiv 16) floordiv 2) * 8 - ((((s0 floordiv 8) floordiv 16) mod 2 + ((s0 floordiv 8) floordiv 4) mod 2) floordiv 2) * 8) mod 128)>()[%51]
      %874 = builtin.unrealized_conversion_cast %873 : index to i64
      %875 = llvm.mul %874, %15 : i64
      %876 = llvm.add %872, %875 : i64
      %877 = llvm.getelementptr %867[%876] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %877, %66, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      rocdl.wait.asyncmark 0
      gpu.barrier
      %878 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32)>(%arg4)[%25, %block_id_z]
      %879 = affine.apply affine_map<(d0)[s0, s1] -> (d0 floordiv 32 + ((d0 floordiv 32 + s0 * 128 + s1 * 4096 + 32) floordiv 128) * -128 + s0 * 128 + s1 * 4096 + 32)>(%arg4)[%25, %block_id_z]
      %subview_12 = memref.subview %transpose[0, %878, %879, %859] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %880 = builtin.unrealized_conversion_cast %subview_12 : memref<32x128xf16, strided<[1, 128], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %881 = llvm.extractvalue %880[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %882 = llvm.extractvalue %880[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %883 = llvm.getelementptr %881[%882] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %884 = llvm.getelementptr %883[%869] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %884, %80, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %885 = llvm.getelementptr %883[%876] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %885, %85, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %886 = affine.apply affine_map<()[s0] -> (s0 * 160 - (s0 floordiv 4) * 640 + ((s0 mod 16) floordiv 4) * 8 - ((s0 mod 4) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>()[%thread_id_x]
      %887 = vector.load %view[%886] : memref<4096xf16, 3>, vector<4xf16>
      %888 = vector.from_elements %107, %118, %119, %120 : vector<4xf16>
      %889 = vector.from_elements %121, %122, %123, %124 : vector<4xf16>
      %890 = vector.extract %887[0] : f16 from vector<4xf16>
      %891 = vector.extract %887[1] : f16 from vector<4xf16>
      %892 = vector.extract %887[2] : f16 from vector<4xf16>
      %893 = vector.extract %887[3] : f16 from vector<4xf16>
      %894 = vector.from_elements %890, %891, %892, %893 : vector<4xf16>
      %895 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %888, %894, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %896 = vector.from_elements %125, %126, %127, %128 : vector<4xf16>
      %897 = vector.from_elements %131, %132, %133, %134 : vector<4xf16>
      %898 = affine.apply affine_map<()[s0] -> (s0 * 160 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 - (s0 floordiv 2) * 64 + 2048)>()[%thread_id_x]
      %899 = vector.load %view[%898] : memref<4096xf16, 3>, vector<4xf16>
      %900 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 16) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 16) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 32)>()[%thread_id_x]
      %901 = vector.load %view[%900] : memref<4096xf16, 3>, vector<4xf16>
      %902 = vector.extract %899[0] : f16 from vector<4xf16>
      %903 = vector.extract %899[1] : f16 from vector<4xf16>
      %904 = vector.extract %899[2] : f16 from vector<4xf16>
      %905 = vector.extract %899[3] : f16 from vector<4xf16>
      %906 = vector.from_elements %902, %903, %904, %905 : vector<4xf16>
      %907 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %889, %906, %895 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %908 = vector.extract %901[0] : f16 from vector<4xf16>
      %909 = vector.extract %901[1] : f16 from vector<4xf16>
      %910 = vector.extract %901[2] : f16 from vector<4xf16>
      %911 = vector.extract %901[3] : f16 from vector<4xf16>
      %912 = vector.from_elements %908, %909, %910, %911 : vector<4xf16>
      %913 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %888, %912, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %914 = vector.extract %907[0] : f32 from vector<4xf32>
      %915 = vector.extract %907[1] : f32 from vector<4xf32>
      %916 = vector.extract %907[2] : f32 from vector<4xf32>
      %917 = vector.extract %907[3] : f32 from vector<4xf32>
      %918 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 16) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 16) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + 2080)>()[%thread_id_x]
      %919 = vector.load %view[%918] : memref<4096xf16, 3>, vector<4xf16>
      %920 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 32) floordiv 64) * 512 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 64)>()[%thread_id_x]
      %921 = vector.load %view[%920] : memref<4096xf16, 3>, vector<4xf16>
      %922 = vector.extract %919[0] : f16 from vector<4xf16>
      %923 = vector.extract %919[1] : f16 from vector<4xf16>
      %924 = vector.extract %919[2] : f16 from vector<4xf16>
      %925 = vector.extract %919[3] : f16 from vector<4xf16>
      %926 = vector.from_elements %922, %923, %924, %925 : vector<4xf16>
      %927 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %889, %926, %913 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %928 = vector.extract %921[0] : f16 from vector<4xf16>
      %929 = vector.extract %921[1] : f16 from vector<4xf16>
      %930 = vector.extract %921[2] : f16 from vector<4xf16>
      %931 = vector.extract %921[3] : f16 from vector<4xf16>
      %932 = vector.from_elements %928, %929, %930, %931 : vector<4xf16>
      %933 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %888, %932, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %934 = vector.extract %927[0] : f32 from vector<4xf32>
      %935 = vector.extract %927[1] : f32 from vector<4xf32>
      %936 = vector.extract %927[2] : f32 from vector<4xf32>
      %937 = vector.extract %927[3] : f32 from vector<4xf32>
      %938 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 32) floordiv 64) * 512 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + 2112)>()[%thread_id_x]
      %939 = vector.load %view[%938] : memref<4096xf16, 3>, vector<4xf16>
      %940 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 48) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 48) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 160)>()[%thread_id_x]
      %941 = vector.load %view[%940] : memref<4096xf16, 3>, vector<4xf16>
      %942 = vector.extract %939[0] : f16 from vector<4xf16>
      %943 = vector.extract %939[1] : f16 from vector<4xf16>
      %944 = vector.extract %939[2] : f16 from vector<4xf16>
      %945 = vector.extract %939[3] : f16 from vector<4xf16>
      %946 = vector.from_elements %942, %943, %944, %945 : vector<4xf16>
      %947 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %889, %946, %933 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %948 = vector.extract %941[0] : f16 from vector<4xf16>
      %949 = vector.extract %941[1] : f16 from vector<4xf16>
      %950 = vector.extract %941[2] : f16 from vector<4xf16>
      %951 = vector.extract %941[3] : f16 from vector<4xf16>
      %952 = vector.from_elements %948, %949, %950, %951 : vector<4xf16>
      %953 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %888, %952, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %954 = vector.extract %947[0] : f32 from vector<4xf32>
      %955 = vector.extract %947[1] : f32 from vector<4xf32>
      %956 = vector.extract %947[2] : f32 from vector<4xf32>
      %957 = vector.extract %947[3] : f32 from vector<4xf32>
      %958 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 48) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 48) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + 2208)>()[%thread_id_x]
      %959 = vector.load %view[%958] : memref<4096xf16, 3>, vector<4xf16>
      %960 = affine.apply affine_map<()[s0] -> (s0 * 160 - (s0 floordiv 4) * 640 + ((s0 mod 16) floordiv 4) * 8 - ((s0 mod 4) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 512)>()[%thread_id_x]
      %961 = vector.load %view[%960] : memref<4096xf16, 3>, vector<4xf16>
      %962 = vector.extract %959[0] : f16 from vector<4xf16>
      %963 = vector.extract %959[1] : f16 from vector<4xf16>
      %964 = vector.extract %959[2] : f16 from vector<4xf16>
      %965 = vector.extract %959[3] : f16 from vector<4xf16>
      %966 = vector.from_elements %962, %963, %964, %965 : vector<4xf16>
      %967 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %889, %966, %953 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %968 = vector.extract %961[0] : f16 from vector<4xf16>
      %969 = vector.extract %961[1] : f16 from vector<4xf16>
      %970 = vector.extract %961[2] : f16 from vector<4xf16>
      %971 = vector.extract %961[3] : f16 from vector<4xf16>
      %972 = vector.from_elements %968, %969, %970, %971 : vector<4xf16>
      %973 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %888, %972, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %974 = vector.extract %967[0] : f32 from vector<4xf32>
      %975 = vector.extract %967[1] : f32 from vector<4xf32>
      %976 = vector.extract %967[2] : f32 from vector<4xf32>
      %977 = vector.extract %967[3] : f32 from vector<4xf32>
      %978 = affine.apply affine_map<()[s0] -> (s0 * 160 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 - (s0 floordiv 2) * 64 + 2560)>()[%thread_id_x]
      %979 = vector.load %view[%978] : memref<4096xf16, 3>, vector<4xf16>
      %980 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 80) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 80) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 32)>()[%thread_id_x]
      %981 = vector.load %view[%980] : memref<4096xf16, 3>, vector<4xf16>
      %982 = vector.extract %979[0] : f16 from vector<4xf16>
      %983 = vector.extract %979[1] : f16 from vector<4xf16>
      %984 = vector.extract %979[2] : f16 from vector<4xf16>
      %985 = vector.extract %979[3] : f16 from vector<4xf16>
      %986 = vector.from_elements %982, %983, %984, %985 : vector<4xf16>
      %987 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %889, %986, %973 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %988 = vector.extract %981[0] : f16 from vector<4xf16>
      %989 = vector.extract %981[1] : f16 from vector<4xf16>
      %990 = vector.extract %981[2] : f16 from vector<4xf16>
      %991 = vector.extract %981[3] : f16 from vector<4xf16>
      %992 = vector.from_elements %988, %989, %990, %991 : vector<4xf16>
      %993 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %888, %992, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %994 = vector.extract %987[0] : f32 from vector<4xf32>
      %995 = vector.extract %987[1] : f32 from vector<4xf32>
      %996 = vector.extract %987[2] : f32 from vector<4xf32>
      %997 = vector.extract %987[3] : f32 from vector<4xf32>
      %998 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 80) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 80) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + 2080)>()[%thread_id_x]
      %999 = vector.load %view[%998] : memref<4096xf16, 3>, vector<4xf16>
      %1000 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 96) floordiv 64) * 512 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 64)>()[%thread_id_x]
      %1001 = vector.load %view[%1000] : memref<4096xf16, 3>, vector<4xf16>
      %1002 = vector.extract %999[0] : f16 from vector<4xf16>
      %1003 = vector.extract %999[1] : f16 from vector<4xf16>
      %1004 = vector.extract %999[2] : f16 from vector<4xf16>
      %1005 = vector.extract %999[3] : f16 from vector<4xf16>
      %1006 = vector.from_elements %1002, %1003, %1004, %1005 : vector<4xf16>
      %1007 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %889, %1006, %993 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1008 = vector.extract %1001[0] : f16 from vector<4xf16>
      %1009 = vector.extract %1001[1] : f16 from vector<4xf16>
      %1010 = vector.extract %1001[2] : f16 from vector<4xf16>
      %1011 = vector.extract %1001[3] : f16 from vector<4xf16>
      %1012 = vector.from_elements %1008, %1009, %1010, %1011 : vector<4xf16>
      %1013 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %888, %1012, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1014 = vector.extract %1007[0] : f32 from vector<4xf32>
      %1015 = vector.extract %1007[1] : f32 from vector<4xf32>
      %1016 = vector.extract %1007[2] : f32 from vector<4xf32>
      %1017 = vector.extract %1007[3] : f32 from vector<4xf32>
      %1018 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 96) floordiv 64) * 512 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + 2112)>()[%thread_id_x]
      %1019 = vector.load %view[%1018] : memref<4096xf16, 3>, vector<4xf16>
      %1020 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 112) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 112) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 160)>()[%thread_id_x]
      %1021 = vector.load %view[%1020] : memref<4096xf16, 3>, vector<4xf16>
      %1022 = vector.extract %1019[0] : f16 from vector<4xf16>
      %1023 = vector.extract %1019[1] : f16 from vector<4xf16>
      %1024 = vector.extract %1019[2] : f16 from vector<4xf16>
      %1025 = vector.extract %1019[3] : f16 from vector<4xf16>
      %1026 = vector.from_elements %1022, %1023, %1024, %1025 : vector<4xf16>
      %1027 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %889, %1026, %1013 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1028 = vector.extract %1021[0] : f16 from vector<4xf16>
      %1029 = vector.extract %1021[1] : f16 from vector<4xf16>
      %1030 = vector.extract %1021[2] : f16 from vector<4xf16>
      %1031 = vector.extract %1021[3] : f16 from vector<4xf16>
      %1032 = vector.from_elements %1028, %1029, %1030, %1031 : vector<4xf16>
      %1033 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %888, %1032, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1034 = vector.extract %1027[0] : f32 from vector<4xf32>
      %1035 = vector.extract %1027[1] : f32 from vector<4xf32>
      %1036 = vector.extract %1027[2] : f32 from vector<4xf32>
      %1037 = vector.extract %1027[3] : f32 from vector<4xf32>
      %1038 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 112) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 112) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + 2208)>()[%thread_id_x]
      %1039 = vector.load %view[%1038] : memref<4096xf16, 3>, vector<4xf16>
      %1040 = vector.load %view[%886] : memref<4096xf16, 3>, vector<4xf16>
      %1041 = vector.extract %1039[0] : f16 from vector<4xf16>
      %1042 = vector.extract %1039[1] : f16 from vector<4xf16>
      %1043 = vector.extract %1039[2] : f16 from vector<4xf16>
      %1044 = vector.extract %1039[3] : f16 from vector<4xf16>
      %1045 = vector.from_elements %1041, %1042, %1043, %1044 : vector<4xf16>
      %1046 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %889, %1045, %1033 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1047 = vector.extract %1040[0] : f16 from vector<4xf16>
      %1048 = vector.extract %1040[1] : f16 from vector<4xf16>
      %1049 = vector.extract %1040[2] : f16 from vector<4xf16>
      %1050 = vector.extract %1040[3] : f16 from vector<4xf16>
      %1051 = vector.from_elements %1047, %1048, %1049, %1050 : vector<4xf16>
      %1052 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1051, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1053 = vector.extract %1046[0] : f32 from vector<4xf32>
      %1054 = vector.extract %1046[1] : f32 from vector<4xf32>
      %1055 = vector.extract %1046[2] : f32 from vector<4xf32>
      %1056 = vector.extract %1046[3] : f32 from vector<4xf32>
      %1057 = vector.load %view[%898] : memref<4096xf16, 3>, vector<4xf16>
      %1058 = vector.load %view[%900] : memref<4096xf16, 3>, vector<4xf16>
      %1059 = vector.extract %1057[0] : f16 from vector<4xf16>
      %1060 = vector.extract %1057[1] : f16 from vector<4xf16>
      %1061 = vector.extract %1057[2] : f16 from vector<4xf16>
      %1062 = vector.extract %1057[3] : f16 from vector<4xf16>
      %1063 = vector.from_elements %1059, %1060, %1061, %1062 : vector<4xf16>
      %1064 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %897, %1063, %1052 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1065 = vector.extract %1058[0] : f16 from vector<4xf16>
      %1066 = vector.extract %1058[1] : f16 from vector<4xf16>
      %1067 = vector.extract %1058[2] : f16 from vector<4xf16>
      %1068 = vector.extract %1058[3] : f16 from vector<4xf16>
      %1069 = vector.from_elements %1065, %1066, %1067, %1068 : vector<4xf16>
      %1070 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1069, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1071 = vector.extract %1064[0] : f32 from vector<4xf32>
      %1072 = vector.extract %1064[1] : f32 from vector<4xf32>
      %1073 = vector.extract %1064[2] : f32 from vector<4xf32>
      %1074 = vector.extract %1064[3] : f32 from vector<4xf32>
      %1075 = vector.load %view[%918] : memref<4096xf16, 3>, vector<4xf16>
      %1076 = vector.load %view[%920] : memref<4096xf16, 3>, vector<4xf16>
      %1077 = vector.extract %1075[0] : f16 from vector<4xf16>
      %1078 = vector.extract %1075[1] : f16 from vector<4xf16>
      %1079 = vector.extract %1075[2] : f16 from vector<4xf16>
      %1080 = vector.extract %1075[3] : f16 from vector<4xf16>
      %1081 = vector.from_elements %1077, %1078, %1079, %1080 : vector<4xf16>
      %1082 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %897, %1081, %1070 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1083 = vector.extract %1076[0] : f16 from vector<4xf16>
      %1084 = vector.extract %1076[1] : f16 from vector<4xf16>
      %1085 = vector.extract %1076[2] : f16 from vector<4xf16>
      %1086 = vector.extract %1076[3] : f16 from vector<4xf16>
      %1087 = vector.from_elements %1083, %1084, %1085, %1086 : vector<4xf16>
      %1088 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1087, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1089 = vector.extract %1082[0] : f32 from vector<4xf32>
      %1090 = vector.extract %1082[1] : f32 from vector<4xf32>
      %1091 = vector.extract %1082[2] : f32 from vector<4xf32>
      %1092 = vector.extract %1082[3] : f32 from vector<4xf32>
      %1093 = vector.load %view[%938] : memref<4096xf16, 3>, vector<4xf16>
      %1094 = vector.load %view[%940] : memref<4096xf16, 3>, vector<4xf16>
      %1095 = vector.extract %1093[0] : f16 from vector<4xf16>
      %1096 = vector.extract %1093[1] : f16 from vector<4xf16>
      %1097 = vector.extract %1093[2] : f16 from vector<4xf16>
      %1098 = vector.extract %1093[3] : f16 from vector<4xf16>
      %1099 = vector.from_elements %1095, %1096, %1097, %1098 : vector<4xf16>
      %1100 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %897, %1099, %1088 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1101 = vector.extract %1094[0] : f16 from vector<4xf16>
      %1102 = vector.extract %1094[1] : f16 from vector<4xf16>
      %1103 = vector.extract %1094[2] : f16 from vector<4xf16>
      %1104 = vector.extract %1094[3] : f16 from vector<4xf16>
      %1105 = vector.from_elements %1101, %1102, %1103, %1104 : vector<4xf16>
      %1106 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1105, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1107 = vector.extract %1100[0] : f32 from vector<4xf32>
      %1108 = vector.extract %1100[1] : f32 from vector<4xf32>
      %1109 = vector.extract %1100[2] : f32 from vector<4xf32>
      %1110 = vector.extract %1100[3] : f32 from vector<4xf32>
      %1111 = vector.load %view[%958] : memref<4096xf16, 3>, vector<4xf16>
      %1112 = vector.load %view[%960] : memref<4096xf16, 3>, vector<4xf16>
      %1113 = vector.extract %1111[0] : f16 from vector<4xf16>
      %1114 = vector.extract %1111[1] : f16 from vector<4xf16>
      %1115 = vector.extract %1111[2] : f16 from vector<4xf16>
      %1116 = vector.extract %1111[3] : f16 from vector<4xf16>
      %1117 = vector.from_elements %1113, %1114, %1115, %1116 : vector<4xf16>
      %1118 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %897, %1117, %1106 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1119 = vector.extract %1112[0] : f16 from vector<4xf16>
      %1120 = vector.extract %1112[1] : f16 from vector<4xf16>
      %1121 = vector.extract %1112[2] : f16 from vector<4xf16>
      %1122 = vector.extract %1112[3] : f16 from vector<4xf16>
      %1123 = vector.from_elements %1119, %1120, %1121, %1122 : vector<4xf16>
      %1124 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1123, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1125 = vector.extract %1118[0] : f32 from vector<4xf32>
      %1126 = vector.extract %1118[1] : f32 from vector<4xf32>
      %1127 = vector.extract %1118[2] : f32 from vector<4xf32>
      %1128 = vector.extract %1118[3] : f32 from vector<4xf32>
      %1129 = vector.load %view[%978] : memref<4096xf16, 3>, vector<4xf16>
      %1130 = vector.load %view[%980] : memref<4096xf16, 3>, vector<4xf16>
      %1131 = vector.extract %1129[0] : f16 from vector<4xf16>
      %1132 = vector.extract %1129[1] : f16 from vector<4xf16>
      %1133 = vector.extract %1129[2] : f16 from vector<4xf16>
      %1134 = vector.extract %1129[3] : f16 from vector<4xf16>
      %1135 = vector.from_elements %1131, %1132, %1133, %1134 : vector<4xf16>
      %1136 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %897, %1135, %1124 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1137 = vector.extract %1130[0] : f16 from vector<4xf16>
      %1138 = vector.extract %1130[1] : f16 from vector<4xf16>
      %1139 = vector.extract %1130[2] : f16 from vector<4xf16>
      %1140 = vector.extract %1130[3] : f16 from vector<4xf16>
      %1141 = vector.from_elements %1137, %1138, %1139, %1140 : vector<4xf16>
      %1142 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1141, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1143 = vector.extract %1136[0] : f32 from vector<4xf32>
      %1144 = vector.extract %1136[1] : f32 from vector<4xf32>
      %1145 = vector.extract %1136[2] : f32 from vector<4xf32>
      %1146 = vector.extract %1136[3] : f32 from vector<4xf32>
      %1147 = vector.load %view[%998] : memref<4096xf16, 3>, vector<4xf16>
      %1148 = vector.load %view[%1000] : memref<4096xf16, 3>, vector<4xf16>
      %1149 = vector.extract %1147[0] : f16 from vector<4xf16>
      %1150 = vector.extract %1147[1] : f16 from vector<4xf16>
      %1151 = vector.extract %1147[2] : f16 from vector<4xf16>
      %1152 = vector.extract %1147[3] : f16 from vector<4xf16>
      %1153 = vector.from_elements %1149, %1150, %1151, %1152 : vector<4xf16>
      %1154 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %897, %1153, %1142 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1155 = vector.extract %1148[0] : f16 from vector<4xf16>
      %1156 = vector.extract %1148[1] : f16 from vector<4xf16>
      %1157 = vector.extract %1148[2] : f16 from vector<4xf16>
      %1158 = vector.extract %1148[3] : f16 from vector<4xf16>
      %1159 = vector.from_elements %1155, %1156, %1157, %1158 : vector<4xf16>
      %1160 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1159, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1161 = vector.extract %1154[0] : f32 from vector<4xf32>
      %1162 = vector.extract %1154[1] : f32 from vector<4xf32>
      %1163 = vector.extract %1154[2] : f32 from vector<4xf32>
      %1164 = vector.extract %1154[3] : f32 from vector<4xf32>
      %1165 = vector.load %view[%1018] : memref<4096xf16, 3>, vector<4xf16>
      %1166 = vector.load %view[%1020] : memref<4096xf16, 3>, vector<4xf16>
      %1167 = vector.extract %1165[0] : f16 from vector<4xf16>
      %1168 = vector.extract %1165[1] : f16 from vector<4xf16>
      %1169 = vector.extract %1165[2] : f16 from vector<4xf16>
      %1170 = vector.extract %1165[3] : f16 from vector<4xf16>
      %1171 = vector.from_elements %1167, %1168, %1169, %1170 : vector<4xf16>
      %1172 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %897, %1171, %1160 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1173 = vector.extract %1166[0] : f16 from vector<4xf16>
      %1174 = vector.extract %1166[1] : f16 from vector<4xf16>
      %1175 = vector.extract %1166[2] : f16 from vector<4xf16>
      %1176 = vector.extract %1166[3] : f16 from vector<4xf16>
      %1177 = vector.from_elements %1173, %1174, %1175, %1176 : vector<4xf16>
      %1178 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %896, %1177, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1179 = vector.extract %1172[0] : f32 from vector<4xf32>
      %1180 = vector.extract %1172[1] : f32 from vector<4xf32>
      %1181 = vector.extract %1172[2] : f32 from vector<4xf32>
      %1182 = vector.extract %1172[3] : f32 from vector<4xf32>
      %1183 = vector.load %view[%1038] : memref<4096xf16, 3>, vector<4xf16>
      %1184 = vector.extract %1183[0] : f16 from vector<4xf16>
      %1185 = vector.extract %1183[1] : f16 from vector<4xf16>
      %1186 = vector.extract %1183[2] : f16 from vector<4xf16>
      %1187 = vector.extract %1183[3] : f16 from vector<4xf16>
      %1188 = vector.from_elements %1184, %1185, %1186, %1187 : vector<4xf16>
      %1189 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %897, %1188, %1178 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1190 = vector.extract %1189[0] : f32 from vector<4xf32>
      %1191 = vector.extract %1189[1] : f32 from vector<4xf32>
      %1192 = vector.extract %1189[2] : f32 from vector<4xf32>
      %1193 = vector.extract %1189[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %1194 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 2048) floordiv 4096) mod 32)>(%arg4)[%25, %block_id_z]
      %1195 = affine.apply affine_map<(d0)[s0, s1] -> (d0 floordiv 32 + ((d0 floordiv 32 + s0 * 128 + s1 * 4096 + 64) floordiv 128) * -128 + s0 * 128 + s1 * 4096 + 64)>(%arg4)[%25, %block_id_z]
      %subview_13 = memref.subview %transpose[0, %1194, %1195, %859] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %1196 = builtin.unrealized_conversion_cast %subview_13 : memref<32x128xf16, strided<[1, 128], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %1197 = llvm.extractvalue %1196[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1198 = llvm.extractvalue %1196[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1199 = llvm.getelementptr %1197[%1198] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %1200 = llvm.getelementptr %1199[%869] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1200, %49, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %1201 = llvm.getelementptr %1199[%876] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1201, %66, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %1202 = vector.load %view_7[%886] : memref<4096xf16, 3>, vector<4xf16>
      %1203 = vector.from_elements %161, %162, %163, %164 : vector<4xf16>
      %1204 = vector.from_elements %914, %915, %916, %917 : vector<4xf32>
      %1205 = vector.from_elements %165, %166, %167, %168 : vector<4xf16>
      %1206 = vector.from_elements %934, %935, %936, %937 : vector<4xf32>
      %1207 = vector.extract %1202[0] : f16 from vector<4xf16>
      %1208 = vector.extract %1202[1] : f16 from vector<4xf16>
      %1209 = vector.extract %1202[2] : f16 from vector<4xf16>
      %1210 = vector.extract %1202[3] : f16 from vector<4xf16>
      %1211 = vector.from_elements %1207, %1208, %1209, %1210 : vector<4xf16>
      %1212 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1203, %1211, %1204 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1213 = vector.from_elements %954, %955, %956, %957 : vector<4xf32>
      %1214 = vector.from_elements %974, %975, %976, %977 : vector<4xf32>
      %1215 = vector.from_elements %994, %995, %996, %997 : vector<4xf32>
      %1216 = vector.from_elements %1014, %1015, %1016, %1017 : vector<4xf32>
      %1217 = vector.from_elements %1034, %1035, %1036, %1037 : vector<4xf32>
      %1218 = vector.from_elements %1053, %1054, %1055, %1056 : vector<4xf32>
      %1219 = vector.from_elements %177, %178, %179, %180 : vector<4xf16>
      %1220 = vector.from_elements %1071, %1072, %1073, %1074 : vector<4xf32>
      %1221 = vector.from_elements %181, %182, %183, %184 : vector<4xf16>
      %1222 = vector.from_elements %1089, %1090, %1091, %1092 : vector<4xf32>
      %1223 = vector.from_elements %1107, %1108, %1109, %1110 : vector<4xf32>
      %1224 = vector.from_elements %1125, %1126, %1127, %1128 : vector<4xf32>
      %1225 = vector.from_elements %1143, %1144, %1145, %1146 : vector<4xf32>
      %1226 = vector.from_elements %1161, %1162, %1163, %1164 : vector<4xf32>
      %1227 = vector.from_elements %1179, %1180, %1181, %1182 : vector<4xf32>
      %1228 = vector.from_elements %1190, %1191, %1192, %1193 : vector<4xf32>
      %1229 = vector.load %view_7[%898] : memref<4096xf16, 3>, vector<4xf16>
      %1230 = vector.load %view_7[%900] : memref<4096xf16, 3>, vector<4xf16>
      %1231 = vector.extract %1229[0] : f16 from vector<4xf16>
      %1232 = vector.extract %1229[1] : f16 from vector<4xf16>
      %1233 = vector.extract %1229[2] : f16 from vector<4xf16>
      %1234 = vector.extract %1229[3] : f16 from vector<4xf16>
      %1235 = vector.from_elements %1231, %1232, %1233, %1234 : vector<4xf16>
      %1236 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1235, %1212 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1237 = vector.extract %1230[0] : f16 from vector<4xf16>
      %1238 = vector.extract %1230[1] : f16 from vector<4xf16>
      %1239 = vector.extract %1230[2] : f16 from vector<4xf16>
      %1240 = vector.extract %1230[3] : f16 from vector<4xf16>
      %1241 = vector.from_elements %1237, %1238, %1239, %1240 : vector<4xf16>
      %1242 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1203, %1241, %1206 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1243 = vector.extract %1236[0] : f32 from vector<4xf32>
      %1244 = vector.extract %1236[1] : f32 from vector<4xf32>
      %1245 = vector.extract %1236[2] : f32 from vector<4xf32>
      %1246 = vector.extract %1236[3] : f32 from vector<4xf32>
      %1247 = vector.load %view_7[%918] : memref<4096xf16, 3>, vector<4xf16>
      %1248 = vector.load %view_7[%920] : memref<4096xf16, 3>, vector<4xf16>
      %1249 = vector.extract %1247[0] : f16 from vector<4xf16>
      %1250 = vector.extract %1247[1] : f16 from vector<4xf16>
      %1251 = vector.extract %1247[2] : f16 from vector<4xf16>
      %1252 = vector.extract %1247[3] : f16 from vector<4xf16>
      %1253 = vector.from_elements %1249, %1250, %1251, %1252 : vector<4xf16>
      %1254 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1253, %1242 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1255 = vector.extract %1248[0] : f16 from vector<4xf16>
      %1256 = vector.extract %1248[1] : f16 from vector<4xf16>
      %1257 = vector.extract %1248[2] : f16 from vector<4xf16>
      %1258 = vector.extract %1248[3] : f16 from vector<4xf16>
      %1259 = vector.from_elements %1255, %1256, %1257, %1258 : vector<4xf16>
      %1260 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1203, %1259, %1213 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1261 = vector.extract %1254[0] : f32 from vector<4xf32>
      %1262 = vector.extract %1254[1] : f32 from vector<4xf32>
      %1263 = vector.extract %1254[2] : f32 from vector<4xf32>
      %1264 = vector.extract %1254[3] : f32 from vector<4xf32>
      %1265 = vector.load %view_7[%938] : memref<4096xf16, 3>, vector<4xf16>
      %1266 = vector.load %view_7[%940] : memref<4096xf16, 3>, vector<4xf16>
      %1267 = vector.extract %1265[0] : f16 from vector<4xf16>
      %1268 = vector.extract %1265[1] : f16 from vector<4xf16>
      %1269 = vector.extract %1265[2] : f16 from vector<4xf16>
      %1270 = vector.extract %1265[3] : f16 from vector<4xf16>
      %1271 = vector.from_elements %1267, %1268, %1269, %1270 : vector<4xf16>
      %1272 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1271, %1260 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1273 = vector.extract %1266[0] : f16 from vector<4xf16>
      %1274 = vector.extract %1266[1] : f16 from vector<4xf16>
      %1275 = vector.extract %1266[2] : f16 from vector<4xf16>
      %1276 = vector.extract %1266[3] : f16 from vector<4xf16>
      %1277 = vector.from_elements %1273, %1274, %1275, %1276 : vector<4xf16>
      %1278 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1203, %1277, %1214 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1279 = vector.extract %1272[0] : f32 from vector<4xf32>
      %1280 = vector.extract %1272[1] : f32 from vector<4xf32>
      %1281 = vector.extract %1272[2] : f32 from vector<4xf32>
      %1282 = vector.extract %1272[3] : f32 from vector<4xf32>
      %1283 = vector.load %view_7[%958] : memref<4096xf16, 3>, vector<4xf16>
      %1284 = vector.load %view_7[%960] : memref<4096xf16, 3>, vector<4xf16>
      %1285 = vector.extract %1283[0] : f16 from vector<4xf16>
      %1286 = vector.extract %1283[1] : f16 from vector<4xf16>
      %1287 = vector.extract %1283[2] : f16 from vector<4xf16>
      %1288 = vector.extract %1283[3] : f16 from vector<4xf16>
      %1289 = vector.from_elements %1285, %1286, %1287, %1288 : vector<4xf16>
      %1290 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1289, %1278 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1291 = vector.extract %1284[0] : f16 from vector<4xf16>
      %1292 = vector.extract %1284[1] : f16 from vector<4xf16>
      %1293 = vector.extract %1284[2] : f16 from vector<4xf16>
      %1294 = vector.extract %1284[3] : f16 from vector<4xf16>
      %1295 = vector.from_elements %1291, %1292, %1293, %1294 : vector<4xf16>
      %1296 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1203, %1295, %1215 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1297 = vector.extract %1290[0] : f32 from vector<4xf32>
      %1298 = vector.extract %1290[1] : f32 from vector<4xf32>
      %1299 = vector.extract %1290[2] : f32 from vector<4xf32>
      %1300 = vector.extract %1290[3] : f32 from vector<4xf32>
      %1301 = vector.load %view_7[%978] : memref<4096xf16, 3>, vector<4xf16>
      %1302 = vector.load %view_7[%980] : memref<4096xf16, 3>, vector<4xf16>
      %1303 = vector.extract %1301[0] : f16 from vector<4xf16>
      %1304 = vector.extract %1301[1] : f16 from vector<4xf16>
      %1305 = vector.extract %1301[2] : f16 from vector<4xf16>
      %1306 = vector.extract %1301[3] : f16 from vector<4xf16>
      %1307 = vector.from_elements %1303, %1304, %1305, %1306 : vector<4xf16>
      %1308 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1307, %1296 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1309 = vector.extract %1302[0] : f16 from vector<4xf16>
      %1310 = vector.extract %1302[1] : f16 from vector<4xf16>
      %1311 = vector.extract %1302[2] : f16 from vector<4xf16>
      %1312 = vector.extract %1302[3] : f16 from vector<4xf16>
      %1313 = vector.from_elements %1309, %1310, %1311, %1312 : vector<4xf16>
      %1314 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1203, %1313, %1216 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1315 = vector.extract %1308[0] : f32 from vector<4xf32>
      %1316 = vector.extract %1308[1] : f32 from vector<4xf32>
      %1317 = vector.extract %1308[2] : f32 from vector<4xf32>
      %1318 = vector.extract %1308[3] : f32 from vector<4xf32>
      %1319 = vector.load %view_7[%998] : memref<4096xf16, 3>, vector<4xf16>
      %1320 = vector.load %view_7[%1000] : memref<4096xf16, 3>, vector<4xf16>
      %1321 = vector.extract %1319[0] : f16 from vector<4xf16>
      %1322 = vector.extract %1319[1] : f16 from vector<4xf16>
      %1323 = vector.extract %1319[2] : f16 from vector<4xf16>
      %1324 = vector.extract %1319[3] : f16 from vector<4xf16>
      %1325 = vector.from_elements %1321, %1322, %1323, %1324 : vector<4xf16>
      %1326 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1325, %1314 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1327 = vector.extract %1320[0] : f16 from vector<4xf16>
      %1328 = vector.extract %1320[1] : f16 from vector<4xf16>
      %1329 = vector.extract %1320[2] : f16 from vector<4xf16>
      %1330 = vector.extract %1320[3] : f16 from vector<4xf16>
      %1331 = vector.from_elements %1327, %1328, %1329, %1330 : vector<4xf16>
      %1332 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1203, %1331, %1217 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1333 = vector.extract %1326[0] : f32 from vector<4xf32>
      %1334 = vector.extract %1326[1] : f32 from vector<4xf32>
      %1335 = vector.extract %1326[2] : f32 from vector<4xf32>
      %1336 = vector.extract %1326[3] : f32 from vector<4xf32>
      %1337 = vector.load %view_7[%1018] : memref<4096xf16, 3>, vector<4xf16>
      %1338 = vector.load %view_7[%1020] : memref<4096xf16, 3>, vector<4xf16>
      %1339 = vector.extract %1337[0] : f16 from vector<4xf16>
      %1340 = vector.extract %1337[1] : f16 from vector<4xf16>
      %1341 = vector.extract %1337[2] : f16 from vector<4xf16>
      %1342 = vector.extract %1337[3] : f16 from vector<4xf16>
      %1343 = vector.from_elements %1339, %1340, %1341, %1342 : vector<4xf16>
      %1344 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1343, %1332 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1345 = vector.extract %1338[0] : f16 from vector<4xf16>
      %1346 = vector.extract %1338[1] : f16 from vector<4xf16>
      %1347 = vector.extract %1338[2] : f16 from vector<4xf16>
      %1348 = vector.extract %1338[3] : f16 from vector<4xf16>
      %1349 = vector.from_elements %1345, %1346, %1347, %1348 : vector<4xf16>
      %1350 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1203, %1349, %1218 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1351 = vector.extract %1344[0] : f32 from vector<4xf32>
      %1352 = vector.extract %1344[1] : f32 from vector<4xf32>
      %1353 = vector.extract %1344[2] : f32 from vector<4xf32>
      %1354 = vector.extract %1344[3] : f32 from vector<4xf32>
      %1355 = vector.load %view_7[%1038] : memref<4096xf16, 3>, vector<4xf16>
      %1356 = vector.load %view_7[%886] : memref<4096xf16, 3>, vector<4xf16>
      %1357 = vector.extract %1355[0] : f16 from vector<4xf16>
      %1358 = vector.extract %1355[1] : f16 from vector<4xf16>
      %1359 = vector.extract %1355[2] : f16 from vector<4xf16>
      %1360 = vector.extract %1355[3] : f16 from vector<4xf16>
      %1361 = vector.from_elements %1357, %1358, %1359, %1360 : vector<4xf16>
      %1362 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1205, %1361, %1350 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1363 = vector.extract %1356[0] : f16 from vector<4xf16>
      %1364 = vector.extract %1356[1] : f16 from vector<4xf16>
      %1365 = vector.extract %1356[2] : f16 from vector<4xf16>
      %1366 = vector.extract %1356[3] : f16 from vector<4xf16>
      %1367 = vector.from_elements %1363, %1364, %1365, %1366 : vector<4xf16>
      %1368 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1219, %1367, %1220 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1369 = vector.extract %1362[0] : f32 from vector<4xf32>
      %1370 = vector.extract %1362[1] : f32 from vector<4xf32>
      %1371 = vector.extract %1362[2] : f32 from vector<4xf32>
      %1372 = vector.extract %1362[3] : f32 from vector<4xf32>
      %1373 = vector.load %view_7[%898] : memref<4096xf16, 3>, vector<4xf16>
      %1374 = vector.load %view_7[%900] : memref<4096xf16, 3>, vector<4xf16>
      %1375 = vector.extract %1373[0] : f16 from vector<4xf16>
      %1376 = vector.extract %1373[1] : f16 from vector<4xf16>
      %1377 = vector.extract %1373[2] : f16 from vector<4xf16>
      %1378 = vector.extract %1373[3] : f16 from vector<4xf16>
      %1379 = vector.from_elements %1375, %1376, %1377, %1378 : vector<4xf16>
      %1380 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1379, %1368 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1381 = vector.extract %1374[0] : f16 from vector<4xf16>
      %1382 = vector.extract %1374[1] : f16 from vector<4xf16>
      %1383 = vector.extract %1374[2] : f16 from vector<4xf16>
      %1384 = vector.extract %1374[3] : f16 from vector<4xf16>
      %1385 = vector.from_elements %1381, %1382, %1383, %1384 : vector<4xf16>
      %1386 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1219, %1385, %1222 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1387 = vector.extract %1380[0] : f32 from vector<4xf32>
      %1388 = vector.extract %1380[1] : f32 from vector<4xf32>
      %1389 = vector.extract %1380[2] : f32 from vector<4xf32>
      %1390 = vector.extract %1380[3] : f32 from vector<4xf32>
      %1391 = vector.load %view_7[%918] : memref<4096xf16, 3>, vector<4xf16>
      %1392 = vector.load %view_7[%920] : memref<4096xf16, 3>, vector<4xf16>
      %1393 = vector.extract %1391[0] : f16 from vector<4xf16>
      %1394 = vector.extract %1391[1] : f16 from vector<4xf16>
      %1395 = vector.extract %1391[2] : f16 from vector<4xf16>
      %1396 = vector.extract %1391[3] : f16 from vector<4xf16>
      %1397 = vector.from_elements %1393, %1394, %1395, %1396 : vector<4xf16>
      %1398 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1397, %1386 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1399 = vector.extract %1392[0] : f16 from vector<4xf16>
      %1400 = vector.extract %1392[1] : f16 from vector<4xf16>
      %1401 = vector.extract %1392[2] : f16 from vector<4xf16>
      %1402 = vector.extract %1392[3] : f16 from vector<4xf16>
      %1403 = vector.from_elements %1399, %1400, %1401, %1402 : vector<4xf16>
      %1404 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1219, %1403, %1223 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1405 = vector.extract %1398[0] : f32 from vector<4xf32>
      %1406 = vector.extract %1398[1] : f32 from vector<4xf32>
      %1407 = vector.extract %1398[2] : f32 from vector<4xf32>
      %1408 = vector.extract %1398[3] : f32 from vector<4xf32>
      %1409 = vector.load %view_7[%938] : memref<4096xf16, 3>, vector<4xf16>
      %1410 = vector.load %view_7[%940] : memref<4096xf16, 3>, vector<4xf16>
      %1411 = vector.extract %1409[0] : f16 from vector<4xf16>
      %1412 = vector.extract %1409[1] : f16 from vector<4xf16>
      %1413 = vector.extract %1409[2] : f16 from vector<4xf16>
      %1414 = vector.extract %1409[3] : f16 from vector<4xf16>
      %1415 = vector.from_elements %1411, %1412, %1413, %1414 : vector<4xf16>
      %1416 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1415, %1404 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1417 = vector.extract %1410[0] : f16 from vector<4xf16>
      %1418 = vector.extract %1410[1] : f16 from vector<4xf16>
      %1419 = vector.extract %1410[2] : f16 from vector<4xf16>
      %1420 = vector.extract %1410[3] : f16 from vector<4xf16>
      %1421 = vector.from_elements %1417, %1418, %1419, %1420 : vector<4xf16>
      %1422 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1219, %1421, %1224 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1423 = vector.extract %1416[0] : f32 from vector<4xf32>
      %1424 = vector.extract %1416[1] : f32 from vector<4xf32>
      %1425 = vector.extract %1416[2] : f32 from vector<4xf32>
      %1426 = vector.extract %1416[3] : f32 from vector<4xf32>
      %1427 = vector.load %view_7[%958] : memref<4096xf16, 3>, vector<4xf16>
      %1428 = vector.load %view_7[%960] : memref<4096xf16, 3>, vector<4xf16>
      %1429 = vector.extract %1427[0] : f16 from vector<4xf16>
      %1430 = vector.extract %1427[1] : f16 from vector<4xf16>
      %1431 = vector.extract %1427[2] : f16 from vector<4xf16>
      %1432 = vector.extract %1427[3] : f16 from vector<4xf16>
      %1433 = vector.from_elements %1429, %1430, %1431, %1432 : vector<4xf16>
      %1434 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1433, %1422 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1435 = vector.extract %1428[0] : f16 from vector<4xf16>
      %1436 = vector.extract %1428[1] : f16 from vector<4xf16>
      %1437 = vector.extract %1428[2] : f16 from vector<4xf16>
      %1438 = vector.extract %1428[3] : f16 from vector<4xf16>
      %1439 = vector.from_elements %1435, %1436, %1437, %1438 : vector<4xf16>
      %1440 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1219, %1439, %1225 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1441 = vector.extract %1434[0] : f32 from vector<4xf32>
      %1442 = vector.extract %1434[1] : f32 from vector<4xf32>
      %1443 = vector.extract %1434[2] : f32 from vector<4xf32>
      %1444 = vector.extract %1434[3] : f32 from vector<4xf32>
      %1445 = vector.load %view_7[%978] : memref<4096xf16, 3>, vector<4xf16>
      %1446 = vector.load %view_7[%980] : memref<4096xf16, 3>, vector<4xf16>
      %1447 = vector.extract %1445[0] : f16 from vector<4xf16>
      %1448 = vector.extract %1445[1] : f16 from vector<4xf16>
      %1449 = vector.extract %1445[2] : f16 from vector<4xf16>
      %1450 = vector.extract %1445[3] : f16 from vector<4xf16>
      %1451 = vector.from_elements %1447, %1448, %1449, %1450 : vector<4xf16>
      %1452 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1451, %1440 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1453 = vector.extract %1446[0] : f16 from vector<4xf16>
      %1454 = vector.extract %1446[1] : f16 from vector<4xf16>
      %1455 = vector.extract %1446[2] : f16 from vector<4xf16>
      %1456 = vector.extract %1446[3] : f16 from vector<4xf16>
      %1457 = vector.from_elements %1453, %1454, %1455, %1456 : vector<4xf16>
      %1458 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1219, %1457, %1226 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1459 = vector.extract %1452[0] : f32 from vector<4xf32>
      %1460 = vector.extract %1452[1] : f32 from vector<4xf32>
      %1461 = vector.extract %1452[2] : f32 from vector<4xf32>
      %1462 = vector.extract %1452[3] : f32 from vector<4xf32>
      %1463 = vector.load %view_7[%998] : memref<4096xf16, 3>, vector<4xf16>
      %1464 = vector.load %view_7[%1000] : memref<4096xf16, 3>, vector<4xf16>
      %1465 = vector.extract %1463[0] : f16 from vector<4xf16>
      %1466 = vector.extract %1463[1] : f16 from vector<4xf16>
      %1467 = vector.extract %1463[2] : f16 from vector<4xf16>
      %1468 = vector.extract %1463[3] : f16 from vector<4xf16>
      %1469 = vector.from_elements %1465, %1466, %1467, %1468 : vector<4xf16>
      %1470 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1469, %1458 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1471 = vector.extract %1464[0] : f16 from vector<4xf16>
      %1472 = vector.extract %1464[1] : f16 from vector<4xf16>
      %1473 = vector.extract %1464[2] : f16 from vector<4xf16>
      %1474 = vector.extract %1464[3] : f16 from vector<4xf16>
      %1475 = vector.from_elements %1471, %1472, %1473, %1474 : vector<4xf16>
      %1476 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1219, %1475, %1227 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1477 = vector.extract %1470[0] : f32 from vector<4xf32>
      %1478 = vector.extract %1470[1] : f32 from vector<4xf32>
      %1479 = vector.extract %1470[2] : f32 from vector<4xf32>
      %1480 = vector.extract %1470[3] : f32 from vector<4xf32>
      %1481 = vector.load %view_7[%1018] : memref<4096xf16, 3>, vector<4xf16>
      %1482 = vector.load %view_7[%1020] : memref<4096xf16, 3>, vector<4xf16>
      %1483 = vector.extract %1481[0] : f16 from vector<4xf16>
      %1484 = vector.extract %1481[1] : f16 from vector<4xf16>
      %1485 = vector.extract %1481[2] : f16 from vector<4xf16>
      %1486 = vector.extract %1481[3] : f16 from vector<4xf16>
      %1487 = vector.from_elements %1483, %1484, %1485, %1486 : vector<4xf16>
      %1488 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1487, %1476 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1489 = vector.extract %1482[0] : f16 from vector<4xf16>
      %1490 = vector.extract %1482[1] : f16 from vector<4xf16>
      %1491 = vector.extract %1482[2] : f16 from vector<4xf16>
      %1492 = vector.extract %1482[3] : f16 from vector<4xf16>
      %1493 = vector.from_elements %1489, %1490, %1491, %1492 : vector<4xf16>
      %1494 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1219, %1493, %1228 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1495 = vector.extract %1488[0] : f32 from vector<4xf32>
      %1496 = vector.extract %1488[1] : f32 from vector<4xf32>
      %1497 = vector.extract %1488[2] : f32 from vector<4xf32>
      %1498 = vector.extract %1488[3] : f32 from vector<4xf32>
      %1499 = vector.load %view_7[%1038] : memref<4096xf16, 3>, vector<4xf16>
      %1500 = vector.extract %1499[0] : f16 from vector<4xf16>
      %1501 = vector.extract %1499[1] : f16 from vector<4xf16>
      %1502 = vector.extract %1499[2] : f16 from vector<4xf16>
      %1503 = vector.extract %1499[3] : f16 from vector<4xf16>
      %1504 = vector.from_elements %1500, %1501, %1502, %1503 : vector<4xf16>
      %1505 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1221, %1504, %1494 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1506 = vector.extract %1505[0] : f32 from vector<4xf32>
      %1507 = vector.extract %1505[1] : f32 from vector<4xf32>
      %1508 = vector.extract %1505[2] : f32 from vector<4xf32>
      %1509 = vector.extract %1505[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %1510 = affine.apply affine_map<(d0)[s0, s1] -> (((d0 + s0 * 4096 + s1 * 131072 + 3072) floordiv 4096) mod 32)>(%arg4)[%25, %block_id_z]
      %1511 = affine.apply affine_map<(d0)[s0, s1] -> (d0 floordiv 32 + ((d0 floordiv 32 + s0 * 128 + s1 * 4096 + 96) floordiv 128) * -128 + s0 * 128 + s1 * 4096 + 96)>(%arg4)[%25, %block_id_z]
      %subview_14 = memref.subview %transpose[0, %1510, %1511, %859] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %1512 = builtin.unrealized_conversion_cast %subview_14 : memref<32x128xf16, strided<[1, 128], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %1513 = llvm.extractvalue %1512[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1514 = llvm.extractvalue %1512[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1515 = llvm.getelementptr %1513[%1514] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %1516 = llvm.getelementptr %1515[%869] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1516, %80, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %1517 = llvm.getelementptr %1515[%876] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1517, %85, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %1518 = vector.load %view[%886] : memref<4096xf16, 3>, vector<4xf16>
      %1519 = vector.from_elements %211, %212, %213, %214 : vector<4xf16>
      %1520 = vector.from_elements %1243, %1244, %1245, %1246 : vector<4xf32>
      %1521 = vector.from_elements %215, %216, %217, %218 : vector<4xf16>
      %1522 = vector.from_elements %1261, %1262, %1263, %1264 : vector<4xf32>
      %1523 = vector.extract %1518[0] : f16 from vector<4xf16>
      %1524 = vector.extract %1518[1] : f16 from vector<4xf16>
      %1525 = vector.extract %1518[2] : f16 from vector<4xf16>
      %1526 = vector.extract %1518[3] : f16 from vector<4xf16>
      %1527 = vector.from_elements %1523, %1524, %1525, %1526 : vector<4xf16>
      %1528 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1519, %1527, %1520 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1529 = vector.from_elements %1279, %1280, %1281, %1282 : vector<4xf32>
      %1530 = vector.from_elements %1297, %1298, %1299, %1300 : vector<4xf32>
      %1531 = vector.from_elements %1315, %1316, %1317, %1318 : vector<4xf32>
      %1532 = vector.from_elements %1333, %1334, %1335, %1336 : vector<4xf32>
      %1533 = vector.from_elements %1351, %1352, %1353, %1354 : vector<4xf32>
      %1534 = vector.from_elements %1369, %1370, %1371, %1372 : vector<4xf32>
      %1535 = vector.from_elements %227, %228, %229, %230 : vector<4xf16>
      %1536 = vector.from_elements %1387, %1388, %1389, %1390 : vector<4xf32>
      %1537 = vector.from_elements %231, %232, %233, %234 : vector<4xf16>
      %1538 = vector.from_elements %1405, %1406, %1407, %1408 : vector<4xf32>
      %1539 = vector.from_elements %1423, %1424, %1425, %1426 : vector<4xf32>
      %1540 = vector.from_elements %1441, %1442, %1443, %1444 : vector<4xf32>
      %1541 = vector.from_elements %1459, %1460, %1461, %1462 : vector<4xf32>
      %1542 = vector.from_elements %1477, %1478, %1479, %1480 : vector<4xf32>
      %1543 = vector.from_elements %1495, %1496, %1497, %1498 : vector<4xf32>
      %1544 = vector.from_elements %1506, %1507, %1508, %1509 : vector<4xf32>
      %1545 = vector.load %view[%898] : memref<4096xf16, 3>, vector<4xf16>
      %1546 = vector.load %view[%900] : memref<4096xf16, 3>, vector<4xf16>
      %1547 = vector.extract %1545[0] : f16 from vector<4xf16>
      %1548 = vector.extract %1545[1] : f16 from vector<4xf16>
      %1549 = vector.extract %1545[2] : f16 from vector<4xf16>
      %1550 = vector.extract %1545[3] : f16 from vector<4xf16>
      %1551 = vector.from_elements %1547, %1548, %1549, %1550 : vector<4xf16>
      %1552 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1521, %1551, %1528 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1553 = vector.extract %1546[0] : f16 from vector<4xf16>
      %1554 = vector.extract %1546[1] : f16 from vector<4xf16>
      %1555 = vector.extract %1546[2] : f16 from vector<4xf16>
      %1556 = vector.extract %1546[3] : f16 from vector<4xf16>
      %1557 = vector.from_elements %1553, %1554, %1555, %1556 : vector<4xf16>
      %1558 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1519, %1557, %1522 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1559 = vector.extract %1552[0] : f32 from vector<4xf32>
      %1560 = vector.extract %1552[1] : f32 from vector<4xf32>
      %1561 = vector.extract %1552[2] : f32 from vector<4xf32>
      %1562 = vector.extract %1552[3] : f32 from vector<4xf32>
      %1563 = vector.load %view[%918] : memref<4096xf16, 3>, vector<4xf16>
      %1564 = vector.load %view[%920] : memref<4096xf16, 3>, vector<4xf16>
      %1565 = vector.extract %1563[0] : f16 from vector<4xf16>
      %1566 = vector.extract %1563[1] : f16 from vector<4xf16>
      %1567 = vector.extract %1563[2] : f16 from vector<4xf16>
      %1568 = vector.extract %1563[3] : f16 from vector<4xf16>
      %1569 = vector.from_elements %1565, %1566, %1567, %1568 : vector<4xf16>
      %1570 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1521, %1569, %1558 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1571 = vector.extract %1564[0] : f16 from vector<4xf16>
      %1572 = vector.extract %1564[1] : f16 from vector<4xf16>
      %1573 = vector.extract %1564[2] : f16 from vector<4xf16>
      %1574 = vector.extract %1564[3] : f16 from vector<4xf16>
      %1575 = vector.from_elements %1571, %1572, %1573, %1574 : vector<4xf16>
      %1576 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1519, %1575, %1529 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1577 = vector.extract %1570[0] : f32 from vector<4xf32>
      %1578 = vector.extract %1570[1] : f32 from vector<4xf32>
      %1579 = vector.extract %1570[2] : f32 from vector<4xf32>
      %1580 = vector.extract %1570[3] : f32 from vector<4xf32>
      %1581 = vector.load %view[%938] : memref<4096xf16, 3>, vector<4xf16>
      %1582 = vector.load %view[%940] : memref<4096xf16, 3>, vector<4xf16>
      %1583 = vector.extract %1581[0] : f16 from vector<4xf16>
      %1584 = vector.extract %1581[1] : f16 from vector<4xf16>
      %1585 = vector.extract %1581[2] : f16 from vector<4xf16>
      %1586 = vector.extract %1581[3] : f16 from vector<4xf16>
      %1587 = vector.from_elements %1583, %1584, %1585, %1586 : vector<4xf16>
      %1588 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1521, %1587, %1576 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1589 = vector.extract %1582[0] : f16 from vector<4xf16>
      %1590 = vector.extract %1582[1] : f16 from vector<4xf16>
      %1591 = vector.extract %1582[2] : f16 from vector<4xf16>
      %1592 = vector.extract %1582[3] : f16 from vector<4xf16>
      %1593 = vector.from_elements %1589, %1590, %1591, %1592 : vector<4xf16>
      %1594 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1519, %1593, %1530 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1595 = vector.extract %1588[0] : f32 from vector<4xf32>
      %1596 = vector.extract %1588[1] : f32 from vector<4xf32>
      %1597 = vector.extract %1588[2] : f32 from vector<4xf32>
      %1598 = vector.extract %1588[3] : f32 from vector<4xf32>
      %1599 = vector.load %view[%958] : memref<4096xf16, 3>, vector<4xf16>
      %1600 = vector.load %view[%960] : memref<4096xf16, 3>, vector<4xf16>
      %1601 = vector.extract %1599[0] : f16 from vector<4xf16>
      %1602 = vector.extract %1599[1] : f16 from vector<4xf16>
      %1603 = vector.extract %1599[2] : f16 from vector<4xf16>
      %1604 = vector.extract %1599[3] : f16 from vector<4xf16>
      %1605 = vector.from_elements %1601, %1602, %1603, %1604 : vector<4xf16>
      %1606 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1521, %1605, %1594 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1607 = vector.extract %1600[0] : f16 from vector<4xf16>
      %1608 = vector.extract %1600[1] : f16 from vector<4xf16>
      %1609 = vector.extract %1600[2] : f16 from vector<4xf16>
      %1610 = vector.extract %1600[3] : f16 from vector<4xf16>
      %1611 = vector.from_elements %1607, %1608, %1609, %1610 : vector<4xf16>
      %1612 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1519, %1611, %1531 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1613 = vector.extract %1606[0] : f32 from vector<4xf32>
      %1614 = vector.extract %1606[1] : f32 from vector<4xf32>
      %1615 = vector.extract %1606[2] : f32 from vector<4xf32>
      %1616 = vector.extract %1606[3] : f32 from vector<4xf32>
      %1617 = vector.load %view[%978] : memref<4096xf16, 3>, vector<4xf16>
      %1618 = vector.load %view[%980] : memref<4096xf16, 3>, vector<4xf16>
      %1619 = vector.extract %1617[0] : f16 from vector<4xf16>
      %1620 = vector.extract %1617[1] : f16 from vector<4xf16>
      %1621 = vector.extract %1617[2] : f16 from vector<4xf16>
      %1622 = vector.extract %1617[3] : f16 from vector<4xf16>
      %1623 = vector.from_elements %1619, %1620, %1621, %1622 : vector<4xf16>
      %1624 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1521, %1623, %1612 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1625 = vector.extract %1618[0] : f16 from vector<4xf16>
      %1626 = vector.extract %1618[1] : f16 from vector<4xf16>
      %1627 = vector.extract %1618[2] : f16 from vector<4xf16>
      %1628 = vector.extract %1618[3] : f16 from vector<4xf16>
      %1629 = vector.from_elements %1625, %1626, %1627, %1628 : vector<4xf16>
      %1630 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1519, %1629, %1532 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1631 = vector.extract %1624[0] : f32 from vector<4xf32>
      %1632 = vector.extract %1624[1] : f32 from vector<4xf32>
      %1633 = vector.extract %1624[2] : f32 from vector<4xf32>
      %1634 = vector.extract %1624[3] : f32 from vector<4xf32>
      %1635 = vector.load %view[%998] : memref<4096xf16, 3>, vector<4xf16>
      %1636 = vector.load %view[%1000] : memref<4096xf16, 3>, vector<4xf16>
      %1637 = vector.extract %1635[0] : f16 from vector<4xf16>
      %1638 = vector.extract %1635[1] : f16 from vector<4xf16>
      %1639 = vector.extract %1635[2] : f16 from vector<4xf16>
      %1640 = vector.extract %1635[3] : f16 from vector<4xf16>
      %1641 = vector.from_elements %1637, %1638, %1639, %1640 : vector<4xf16>
      %1642 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1521, %1641, %1630 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1643 = vector.extract %1636[0] : f16 from vector<4xf16>
      %1644 = vector.extract %1636[1] : f16 from vector<4xf16>
      %1645 = vector.extract %1636[2] : f16 from vector<4xf16>
      %1646 = vector.extract %1636[3] : f16 from vector<4xf16>
      %1647 = vector.from_elements %1643, %1644, %1645, %1646 : vector<4xf16>
      %1648 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1519, %1647, %1533 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1649 = vector.extract %1642[0] : f32 from vector<4xf32>
      %1650 = vector.extract %1642[1] : f32 from vector<4xf32>
      %1651 = vector.extract %1642[2] : f32 from vector<4xf32>
      %1652 = vector.extract %1642[3] : f32 from vector<4xf32>
      %1653 = vector.load %view[%1018] : memref<4096xf16, 3>, vector<4xf16>
      %1654 = vector.load %view[%1020] : memref<4096xf16, 3>, vector<4xf16>
      %1655 = vector.extract %1653[0] : f16 from vector<4xf16>
      %1656 = vector.extract %1653[1] : f16 from vector<4xf16>
      %1657 = vector.extract %1653[2] : f16 from vector<4xf16>
      %1658 = vector.extract %1653[3] : f16 from vector<4xf16>
      %1659 = vector.from_elements %1655, %1656, %1657, %1658 : vector<4xf16>
      %1660 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1521, %1659, %1648 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1661 = vector.extract %1654[0] : f16 from vector<4xf16>
      %1662 = vector.extract %1654[1] : f16 from vector<4xf16>
      %1663 = vector.extract %1654[2] : f16 from vector<4xf16>
      %1664 = vector.extract %1654[3] : f16 from vector<4xf16>
      %1665 = vector.from_elements %1661, %1662, %1663, %1664 : vector<4xf16>
      %1666 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1519, %1665, %1534 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1667 = vector.extract %1660[0] : f32 from vector<4xf32>
      %1668 = vector.extract %1660[1] : f32 from vector<4xf32>
      %1669 = vector.extract %1660[2] : f32 from vector<4xf32>
      %1670 = vector.extract %1660[3] : f32 from vector<4xf32>
      %1671 = vector.load %view[%1038] : memref<4096xf16, 3>, vector<4xf16>
      %1672 = vector.load %view[%886] : memref<4096xf16, 3>, vector<4xf16>
      %1673 = vector.extract %1671[0] : f16 from vector<4xf16>
      %1674 = vector.extract %1671[1] : f16 from vector<4xf16>
      %1675 = vector.extract %1671[2] : f16 from vector<4xf16>
      %1676 = vector.extract %1671[3] : f16 from vector<4xf16>
      %1677 = vector.from_elements %1673, %1674, %1675, %1676 : vector<4xf16>
      %1678 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1521, %1677, %1666 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1679 = vector.extract %1672[0] : f16 from vector<4xf16>
      %1680 = vector.extract %1672[1] : f16 from vector<4xf16>
      %1681 = vector.extract %1672[2] : f16 from vector<4xf16>
      %1682 = vector.extract %1672[3] : f16 from vector<4xf16>
      %1683 = vector.from_elements %1679, %1680, %1681, %1682 : vector<4xf16>
      %1684 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1535, %1683, %1536 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1685 = vector.extract %1678[0] : f32 from vector<4xf32>
      %1686 = vector.extract %1678[1] : f32 from vector<4xf32>
      %1687 = vector.extract %1678[2] : f32 from vector<4xf32>
      %1688 = vector.extract %1678[3] : f32 from vector<4xf32>
      %1689 = vector.load %view[%898] : memref<4096xf16, 3>, vector<4xf16>
      %1690 = vector.load %view[%900] : memref<4096xf16, 3>, vector<4xf16>
      %1691 = vector.extract %1689[0] : f16 from vector<4xf16>
      %1692 = vector.extract %1689[1] : f16 from vector<4xf16>
      %1693 = vector.extract %1689[2] : f16 from vector<4xf16>
      %1694 = vector.extract %1689[3] : f16 from vector<4xf16>
      %1695 = vector.from_elements %1691, %1692, %1693, %1694 : vector<4xf16>
      %1696 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1537, %1695, %1684 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1697 = vector.extract %1690[0] : f16 from vector<4xf16>
      %1698 = vector.extract %1690[1] : f16 from vector<4xf16>
      %1699 = vector.extract %1690[2] : f16 from vector<4xf16>
      %1700 = vector.extract %1690[3] : f16 from vector<4xf16>
      %1701 = vector.from_elements %1697, %1698, %1699, %1700 : vector<4xf16>
      %1702 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1535, %1701, %1538 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1703 = vector.extract %1696[0] : f32 from vector<4xf32>
      %1704 = vector.extract %1696[1] : f32 from vector<4xf32>
      %1705 = vector.extract %1696[2] : f32 from vector<4xf32>
      %1706 = vector.extract %1696[3] : f32 from vector<4xf32>
      %1707 = vector.load %view[%918] : memref<4096xf16, 3>, vector<4xf16>
      %1708 = vector.load %view[%920] : memref<4096xf16, 3>, vector<4xf16>
      %1709 = vector.extract %1707[0] : f16 from vector<4xf16>
      %1710 = vector.extract %1707[1] : f16 from vector<4xf16>
      %1711 = vector.extract %1707[2] : f16 from vector<4xf16>
      %1712 = vector.extract %1707[3] : f16 from vector<4xf16>
      %1713 = vector.from_elements %1709, %1710, %1711, %1712 : vector<4xf16>
      %1714 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1537, %1713, %1702 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1715 = vector.extract %1708[0] : f16 from vector<4xf16>
      %1716 = vector.extract %1708[1] : f16 from vector<4xf16>
      %1717 = vector.extract %1708[2] : f16 from vector<4xf16>
      %1718 = vector.extract %1708[3] : f16 from vector<4xf16>
      %1719 = vector.from_elements %1715, %1716, %1717, %1718 : vector<4xf16>
      %1720 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1535, %1719, %1539 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1721 = vector.extract %1714[0] : f32 from vector<4xf32>
      %1722 = vector.extract %1714[1] : f32 from vector<4xf32>
      %1723 = vector.extract %1714[2] : f32 from vector<4xf32>
      %1724 = vector.extract %1714[3] : f32 from vector<4xf32>
      %1725 = vector.load %view[%938] : memref<4096xf16, 3>, vector<4xf16>
      %1726 = vector.load %view[%940] : memref<4096xf16, 3>, vector<4xf16>
      %1727 = vector.extract %1725[0] : f16 from vector<4xf16>
      %1728 = vector.extract %1725[1] : f16 from vector<4xf16>
      %1729 = vector.extract %1725[2] : f16 from vector<4xf16>
      %1730 = vector.extract %1725[3] : f16 from vector<4xf16>
      %1731 = vector.from_elements %1727, %1728, %1729, %1730 : vector<4xf16>
      %1732 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1537, %1731, %1720 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1733 = vector.extract %1726[0] : f16 from vector<4xf16>
      %1734 = vector.extract %1726[1] : f16 from vector<4xf16>
      %1735 = vector.extract %1726[2] : f16 from vector<4xf16>
      %1736 = vector.extract %1726[3] : f16 from vector<4xf16>
      %1737 = vector.from_elements %1733, %1734, %1735, %1736 : vector<4xf16>
      %1738 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1535, %1737, %1540 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1739 = vector.extract %1732[0] : f32 from vector<4xf32>
      %1740 = vector.extract %1732[1] : f32 from vector<4xf32>
      %1741 = vector.extract %1732[2] : f32 from vector<4xf32>
      %1742 = vector.extract %1732[3] : f32 from vector<4xf32>
      %1743 = vector.load %view[%958] : memref<4096xf16, 3>, vector<4xf16>
      %1744 = vector.load %view[%960] : memref<4096xf16, 3>, vector<4xf16>
      %1745 = vector.extract %1743[0] : f16 from vector<4xf16>
      %1746 = vector.extract %1743[1] : f16 from vector<4xf16>
      %1747 = vector.extract %1743[2] : f16 from vector<4xf16>
      %1748 = vector.extract %1743[3] : f16 from vector<4xf16>
      %1749 = vector.from_elements %1745, %1746, %1747, %1748 : vector<4xf16>
      %1750 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1537, %1749, %1738 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1751 = vector.extract %1744[0] : f16 from vector<4xf16>
      %1752 = vector.extract %1744[1] : f16 from vector<4xf16>
      %1753 = vector.extract %1744[2] : f16 from vector<4xf16>
      %1754 = vector.extract %1744[3] : f16 from vector<4xf16>
      %1755 = vector.from_elements %1751, %1752, %1753, %1754 : vector<4xf16>
      %1756 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1535, %1755, %1541 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1757 = vector.extract %1750[0] : f32 from vector<4xf32>
      %1758 = vector.extract %1750[1] : f32 from vector<4xf32>
      %1759 = vector.extract %1750[2] : f32 from vector<4xf32>
      %1760 = vector.extract %1750[3] : f32 from vector<4xf32>
      %1761 = vector.load %view[%978] : memref<4096xf16, 3>, vector<4xf16>
      %1762 = vector.load %view[%980] : memref<4096xf16, 3>, vector<4xf16>
      %1763 = vector.extract %1761[0] : f16 from vector<4xf16>
      %1764 = vector.extract %1761[1] : f16 from vector<4xf16>
      %1765 = vector.extract %1761[2] : f16 from vector<4xf16>
      %1766 = vector.extract %1761[3] : f16 from vector<4xf16>
      %1767 = vector.from_elements %1763, %1764, %1765, %1766 : vector<4xf16>
      %1768 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1537, %1767, %1756 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1769 = vector.extract %1762[0] : f16 from vector<4xf16>
      %1770 = vector.extract %1762[1] : f16 from vector<4xf16>
      %1771 = vector.extract %1762[2] : f16 from vector<4xf16>
      %1772 = vector.extract %1762[3] : f16 from vector<4xf16>
      %1773 = vector.from_elements %1769, %1770, %1771, %1772 : vector<4xf16>
      %1774 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1535, %1773, %1542 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1775 = vector.extract %1768[0] : f32 from vector<4xf32>
      %1776 = vector.extract %1768[1] : f32 from vector<4xf32>
      %1777 = vector.extract %1768[2] : f32 from vector<4xf32>
      %1778 = vector.extract %1768[3] : f32 from vector<4xf32>
      %1779 = vector.load %view[%998] : memref<4096xf16, 3>, vector<4xf16>
      %1780 = vector.load %view[%1000] : memref<4096xf16, 3>, vector<4xf16>
      %1781 = vector.extract %1779[0] : f16 from vector<4xf16>
      %1782 = vector.extract %1779[1] : f16 from vector<4xf16>
      %1783 = vector.extract %1779[2] : f16 from vector<4xf16>
      %1784 = vector.extract %1779[3] : f16 from vector<4xf16>
      %1785 = vector.from_elements %1781, %1782, %1783, %1784 : vector<4xf16>
      %1786 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1537, %1785, %1774 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1787 = vector.extract %1780[0] : f16 from vector<4xf16>
      %1788 = vector.extract %1780[1] : f16 from vector<4xf16>
      %1789 = vector.extract %1780[2] : f16 from vector<4xf16>
      %1790 = vector.extract %1780[3] : f16 from vector<4xf16>
      %1791 = vector.from_elements %1787, %1788, %1789, %1790 : vector<4xf16>
      %1792 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1535, %1791, %1543 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1793 = vector.extract %1786[0] : f32 from vector<4xf32>
      %1794 = vector.extract %1786[1] : f32 from vector<4xf32>
      %1795 = vector.extract %1786[2] : f32 from vector<4xf32>
      %1796 = vector.extract %1786[3] : f32 from vector<4xf32>
      %1797 = vector.load %view[%1018] : memref<4096xf16, 3>, vector<4xf16>
      %1798 = vector.load %view[%1020] : memref<4096xf16, 3>, vector<4xf16>
      %1799 = vector.extract %1797[0] : f16 from vector<4xf16>
      %1800 = vector.extract %1797[1] : f16 from vector<4xf16>
      %1801 = vector.extract %1797[2] : f16 from vector<4xf16>
      %1802 = vector.extract %1797[3] : f16 from vector<4xf16>
      %1803 = vector.from_elements %1799, %1800, %1801, %1802 : vector<4xf16>
      %1804 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1537, %1803, %1792 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1805 = vector.extract %1798[0] : f16 from vector<4xf16>
      %1806 = vector.extract %1798[1] : f16 from vector<4xf16>
      %1807 = vector.extract %1798[2] : f16 from vector<4xf16>
      %1808 = vector.extract %1798[3] : f16 from vector<4xf16>
      %1809 = vector.from_elements %1805, %1806, %1807, %1808 : vector<4xf16>
      %1810 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1535, %1809, %1544 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1811 = vector.extract %1804[0] : f32 from vector<4xf32>
      %1812 = vector.extract %1804[1] : f32 from vector<4xf32>
      %1813 = vector.extract %1804[2] : f32 from vector<4xf32>
      %1814 = vector.extract %1804[3] : f32 from vector<4xf32>
      %1815 = vector.load %view[%1038] : memref<4096xf16, 3>, vector<4xf16>
      %1816 = vector.extract %1815[0] : f16 from vector<4xf16>
      %1817 = vector.extract %1815[1] : f16 from vector<4xf16>
      %1818 = vector.extract %1815[2] : f16 from vector<4xf16>
      %1819 = vector.extract %1815[3] : f16 from vector<4xf16>
      %1820 = vector.from_elements %1816, %1817, %1818, %1819 : vector<4xf16>
      %1821 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1537, %1820, %1810 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1822 = vector.extract %1821[0] : f32 from vector<4xf32>
      %1823 = vector.extract %1821[1] : f32 from vector<4xf32>
      %1824 = vector.extract %1821[2] : f32 from vector<4xf32>
      %1825 = vector.extract %1821[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %1826 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 floordiv 32 + s0 + s1 * 32) mod 32)>(%arg4)[%25, %block_id_z]
      %subview_15 = memref.subview %arg2[0, %1826, %859, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %1827 = builtin.unrealized_conversion_cast %subview_15 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %1828 = affine.apply affine_map<()[s0] -> (s0 floordiv 32)>()[%30]
      %1829 = builtin.unrealized_conversion_cast %1828 : index to i64
      %1830 = affine.apply affine_map<()[s0] -> (s0 - ((s0 floordiv 8) floordiv 4) * 32)>()[%30]
      %1831 = builtin.unrealized_conversion_cast %1830 : index to i64
      %1832 = llvm.extractvalue %1827[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1833 = llvm.extractvalue %1827[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1834 = llvm.getelementptr %1832[%1833] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %1835 = llvm.mul %1829, %15 : i64
      %1836 = llvm.add %1835, %1831 : i64
      %1837 = llvm.getelementptr %1834[%1836] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1837, %49, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %1838 = affine.apply affine_map<()[s0] -> (s0 floordiv 32)>()[%51]
      %1839 = builtin.unrealized_conversion_cast %1838 : index to i64
      %1840 = affine.apply affine_map<()[s0] -> (s0 - ((s0 floordiv 8) floordiv 4) * 32)>()[%51]
      %1841 = builtin.unrealized_conversion_cast %1840 : index to i64
      %1842 = llvm.mul %1839, %15 : i64
      %1843 = llvm.add %1842, %1841 : i64
      %1844 = llvm.getelementptr %1834[%1843] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1844, %66, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %1845 = vector.load %view_7[%886] : memref<4096xf16, 3>, vector<4xf16>
      %1846 = vector.from_elements %255, %256, %257, %258 : vector<4xf16>
      %1847 = vector.from_elements %1559, %1560, %1561, %1562 : vector<4xf32>
      %1848 = vector.from_elements %259, %260, %261, %262 : vector<4xf16>
      %1849 = vector.from_elements %1577, %1578, %1579, %1580 : vector<4xf32>
      %1850 = vector.extract %1845[0] : f16 from vector<4xf16>
      %1851 = vector.extract %1845[1] : f16 from vector<4xf16>
      %1852 = vector.extract %1845[2] : f16 from vector<4xf16>
      %1853 = vector.extract %1845[3] : f16 from vector<4xf16>
      %1854 = vector.from_elements %1850, %1851, %1852, %1853 : vector<4xf16>
      %1855 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1846, %1854, %1847 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1856 = vector.from_elements %1595, %1596, %1597, %1598 : vector<4xf32>
      %1857 = vector.from_elements %1613, %1614, %1615, %1616 : vector<4xf32>
      %1858 = vector.from_elements %1631, %1632, %1633, %1634 : vector<4xf32>
      %1859 = vector.from_elements %1649, %1650, %1651, %1652 : vector<4xf32>
      %1860 = vector.from_elements %1667, %1668, %1669, %1670 : vector<4xf32>
      %1861 = vector.from_elements %1685, %1686, %1687, %1688 : vector<4xf32>
      %1862 = vector.from_elements %271, %272, %273, %274 : vector<4xf16>
      %1863 = vector.from_elements %1703, %1704, %1705, %1706 : vector<4xf32>
      %1864 = vector.from_elements %275, %276, %277, %278 : vector<4xf16>
      %1865 = vector.from_elements %1721, %1722, %1723, %1724 : vector<4xf32>
      %1866 = vector.from_elements %1739, %1740, %1741, %1742 : vector<4xf32>
      %1867 = vector.from_elements %1757, %1758, %1759, %1760 : vector<4xf32>
      %1868 = vector.from_elements %1775, %1776, %1777, %1778 : vector<4xf32>
      %1869 = vector.from_elements %1793, %1794, %1795, %1796 : vector<4xf32>
      %1870 = vector.from_elements %1811, %1812, %1813, %1814 : vector<4xf32>
      %1871 = vector.from_elements %1822, %1823, %1824, %1825 : vector<4xf32>
      %1872 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16)>()[%thread_id_x, %68]
      %1873 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16)>(%arg4)[%thread_id_x]
      %1874 = arith.index_cast %1872 : index to i32
      %1875 = arith.index_cast %1873 : index to i32
      %1876 = arith.addi %1874, %c1_i32 : i32
      %1877 = arith.cmpi ule, %1876, %1875 {frisk.causal_mask_i32} : i32
      %1878 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 4) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 4) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 4) floordiv 16) floordiv 4) * 16 + 1)>(%arg4)[%thread_id_x]
      %1879 = arith.index_cast %1878 : index to i32
      %1880 = arith.cmpi ule, %1876, %1879 {frisk.causal_mask_i32} : i32
      %1881 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 8) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 8) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 8) floordiv 16) floordiv 4) * 16 + 2)>(%arg4)[%thread_id_x]
      %1882 = arith.index_cast %1881 : index to i32
      %1883 = arith.cmpi ule, %1876, %1882 {frisk.causal_mask_i32} : i32
      %1884 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 12) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 12) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 12) floordiv 16) floordiv 4) * 16 + 3)>(%arg4)[%thread_id_x]
      %1885 = arith.index_cast %1884 : index to i32
      %1886 = arith.cmpi ule, %1876, %1885 {frisk.causal_mask_i32} : i32
      %1887 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 16) floordiv 64) * 64 + 4)>(%arg4)[%thread_id_x]
      %1888 = arith.index_cast %1887 : index to i32
      %1889 = arith.cmpi ule, %1876, %1888 {frisk.causal_mask_i32} : i32
      %1890 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 20) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 20) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 20) floordiv 16) floordiv 4) * 16 + 1)>(%arg4)[%thread_id_x]
      %1891 = arith.index_cast %1890 : index to i32
      %1892 = arith.cmpi ule, %1876, %1891 {frisk.causal_mask_i32} : i32
      %1893 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 24) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 24) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 24) floordiv 16) floordiv 4) * 16 + 2)>(%arg4)[%thread_id_x]
      %1894 = arith.index_cast %1893 : index to i32
      %1895 = arith.cmpi ule, %1876, %1894 {frisk.causal_mask_i32} : i32
      %1896 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 28) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 28) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 28) floordiv 16) floordiv 4) * 16 + 3)>(%arg4)[%thread_id_x]
      %1897 = arith.index_cast %1896 : index to i32
      %1898 = arith.cmpi ule, %1876, %1897 {frisk.causal_mask_i32} : i32
      %1899 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 32) floordiv 64) * 64 + 8)>(%arg4)[%thread_id_x]
      %1900 = arith.index_cast %1899 : index to i32
      %1901 = arith.cmpi ule, %1876, %1900 {frisk.causal_mask_i32} : i32
      %1902 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 36) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 36) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 36) floordiv 16) floordiv 4) * 16 + 1)>(%arg4)[%thread_id_x]
      %1903 = arith.index_cast %1902 : index to i32
      %1904 = arith.cmpi ule, %1876, %1903 {frisk.causal_mask_i32} : i32
      %1905 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 40) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 40) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 40) floordiv 16) floordiv 4) * 16 + 2)>(%arg4)[%thread_id_x]
      %1906 = arith.index_cast %1905 : index to i32
      %1907 = arith.cmpi ule, %1876, %1906 {frisk.causal_mask_i32} : i32
      %1908 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 44) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 44) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 44) floordiv 16) floordiv 4) * 16 + 3)>(%arg4)[%thread_id_x]
      %1909 = arith.index_cast %1908 : index to i32
      %1910 = arith.cmpi ule, %1876, %1909 {frisk.causal_mask_i32} : i32
      %1911 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 48) floordiv 64) * 64 + 12)>(%arg4)[%thread_id_x]
      %1912 = arith.index_cast %1911 : index to i32
      %1913 = arith.cmpi ule, %1876, %1912 {frisk.causal_mask_i32} : i32
      %1914 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 52) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 52) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 52) floordiv 16) floordiv 4) * 16 + 1)>(%arg4)[%thread_id_x]
      %1915 = arith.index_cast %1914 : index to i32
      %1916 = arith.cmpi ule, %1876, %1915 {frisk.causal_mask_i32} : i32
      %1917 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 56) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 56) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 56) floordiv 16) floordiv 4) * 16 + 2)>(%arg4)[%thread_id_x]
      %1918 = arith.index_cast %1917 : index to i32
      %1919 = arith.cmpi ule, %1876, %1918 {frisk.causal_mask_i32} : i32
      %1920 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 60) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 60) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 60) floordiv 16) floordiv 4) * 16 + 3)>(%arg4)[%thread_id_x]
      %1921 = arith.index_cast %1920 : index to i32
      %1922 = arith.cmpi ule, %1876, %1921 {frisk.causal_mask_i32} : i32
      %1923 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + 64)>(%arg4)[%thread_id_x]
      %1924 = arith.index_cast %1923 : index to i32
      %1925 = arith.cmpi ule, %1876, %1924 {frisk.causal_mask_i32} : i32
      %1926 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 68) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 68) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 68) floordiv 16) floordiv 4) * 16 + 1)>(%arg4)[%thread_id_x]
      %1927 = arith.index_cast %1926 : index to i32
      %1928 = arith.cmpi ule, %1876, %1927 {frisk.causal_mask_i32} : i32
      %1929 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 72) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 72) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 72) floordiv 16) floordiv 4) * 16 + 2)>(%arg4)[%thread_id_x]
      %1930 = arith.index_cast %1929 : index to i32
      %1931 = arith.cmpi ule, %1876, %1930 {frisk.causal_mask_i32} : i32
      %1932 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 76) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 76) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 76) floordiv 16) floordiv 4) * 16 + 3)>(%arg4)[%thread_id_x]
      %1933 = arith.index_cast %1932 : index to i32
      %1934 = arith.cmpi ule, %1876, %1933 {frisk.causal_mask_i32} : i32
      %1935 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 80) floordiv 64) * 64 + 4)>(%arg4)[%thread_id_x]
      %1936 = arith.index_cast %1935 : index to i32
      %1937 = arith.cmpi ule, %1876, %1936 {frisk.causal_mask_i32} : i32
      %1938 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 84) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 84) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 84) floordiv 16) floordiv 4) * 16 + 1)>(%arg4)[%thread_id_x]
      %1939 = arith.index_cast %1938 : index to i32
      %1940 = arith.cmpi ule, %1876, %1939 {frisk.causal_mask_i32} : i32
      %1941 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 88) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 88) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 88) floordiv 16) floordiv 4) * 16 + 2)>(%arg4)[%thread_id_x]
      %1942 = arith.index_cast %1941 : index to i32
      %1943 = arith.cmpi ule, %1876, %1942 {frisk.causal_mask_i32} : i32
      %1944 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 92) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 92) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 92) floordiv 16) floordiv 4) * 16 + 3)>(%arg4)[%thread_id_x]
      %1945 = arith.index_cast %1944 : index to i32
      %1946 = arith.cmpi ule, %1876, %1945 {frisk.causal_mask_i32} : i32
      %1947 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 96) floordiv 64) * 64 + 8)>(%arg4)[%thread_id_x]
      %1948 = arith.index_cast %1947 : index to i32
      %1949 = arith.cmpi ule, %1876, %1948 {frisk.causal_mask_i32} : i32
      %1950 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 100) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 100) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 100) floordiv 16) floordiv 4) * 16 + 1)>(%arg4)[%thread_id_x]
      %1951 = arith.index_cast %1950 : index to i32
      %1952 = arith.cmpi ule, %1876, %1951 {frisk.causal_mask_i32} : i32
      %1953 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 104) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 104) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 104) floordiv 16) floordiv 4) * 16 + 2)>(%arg4)[%thread_id_x]
      %1954 = arith.index_cast %1953 : index to i32
      %1955 = arith.cmpi ule, %1876, %1954 {frisk.causal_mask_i32} : i32
      %1956 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 108) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 108) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 108) floordiv 16) floordiv 4) * 16 + 3)>(%arg4)[%thread_id_x]
      %1957 = arith.index_cast %1956 : index to i32
      %1958 = arith.cmpi ule, %1876, %1957 {frisk.causal_mask_i32} : i32
      %1959 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 112) floordiv 64) * 64 + 12)>(%arg4)[%thread_id_x]
      %1960 = arith.index_cast %1959 : index to i32
      %1961 = arith.cmpi ule, %1876, %1960 {frisk.causal_mask_i32} : i32
      %1962 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 116) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 116) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 116) floordiv 16) floordiv 4) * 16 + 1)>(%arg4)[%thread_id_x]
      %1963 = arith.index_cast %1962 : index to i32
      %1964 = arith.cmpi ule, %1876, %1963 {frisk.causal_mask_i32} : i32
      %1965 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 120) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 120) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 120) floordiv 16) floordiv 4) * 16 + 2)>(%arg4)[%thread_id_x]
      %1966 = arith.index_cast %1965 : index to i32
      %1967 = arith.cmpi ule, %1876, %1966 {frisk.causal_mask_i32} : i32
      %1968 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + ((s0 mod 64) floordiv 16) * 16 + (((s0 mod 64) floordiv 16 + 124) floordiv 64) * 64 + (((s0 mod 64) floordiv 16 + 124) floordiv 16) * 4 - ((((s0 mod 64) floordiv 16 + 124) floordiv 16) floordiv 4) * 16 + 3)>(%arg4)[%thread_id_x]
      %1969 = arith.index_cast %1968 : index to i32
      %1970 = arith.cmpi ule, %1876, %1969 {frisk.causal_mask_i32} : i32
      %1971 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + (s0 floordiv 64) * 16 - ((s0 floordiv 64) floordiv 4) * 64 - (s0 floordiv 16) * 16 + 64)>()[%thread_id_x, %68]
      %1972 = arith.index_cast %1971 : index to i32
      %1973 = arith.addi %1972, %c1_i32 : i32
      %1974 = arith.cmpi ule, %1973, %1875 {frisk.causal_mask_i32} : i32
      %1975 = arith.cmpi ule, %1973, %1879 {frisk.causal_mask_i32} : i32
      %1976 = arith.cmpi ule, %1973, %1882 {frisk.causal_mask_i32} : i32
      %1977 = arith.cmpi ule, %1973, %1885 {frisk.causal_mask_i32} : i32
      %1978 = arith.cmpi ule, %1973, %1888 {frisk.causal_mask_i32} : i32
      %1979 = arith.cmpi ule, %1973, %1891 {frisk.causal_mask_i32} : i32
      %1980 = arith.cmpi ule, %1973, %1894 {frisk.causal_mask_i32} : i32
      %1981 = arith.cmpi ule, %1973, %1897 {frisk.causal_mask_i32} : i32
      %1982 = arith.cmpi ule, %1973, %1900 {frisk.causal_mask_i32} : i32
      %1983 = arith.cmpi ule, %1973, %1903 {frisk.causal_mask_i32} : i32
      %1984 = arith.cmpi ule, %1973, %1906 {frisk.causal_mask_i32} : i32
      %1985 = arith.cmpi ule, %1973, %1909 {frisk.causal_mask_i32} : i32
      %1986 = arith.cmpi ule, %1973, %1912 {frisk.causal_mask_i32} : i32
      %1987 = arith.cmpi ule, %1973, %1915 {frisk.causal_mask_i32} : i32
      %1988 = arith.cmpi ule, %1973, %1918 {frisk.causal_mask_i32} : i32
      %1989 = arith.cmpi ule, %1973, %1921 {frisk.causal_mask_i32} : i32
      %1990 = arith.cmpi ule, %1973, %1924 {frisk.causal_mask_i32} : i32
      %1991 = arith.cmpi ule, %1973, %1927 {frisk.causal_mask_i32} : i32
      %1992 = arith.cmpi ule, %1973, %1930 {frisk.causal_mask_i32} : i32
      %1993 = arith.cmpi ule, %1973, %1933 {frisk.causal_mask_i32} : i32
      %1994 = arith.cmpi ule, %1973, %1936 {frisk.causal_mask_i32} : i32
      %1995 = arith.cmpi ule, %1973, %1939 {frisk.causal_mask_i32} : i32
      %1996 = arith.cmpi ule, %1973, %1942 {frisk.causal_mask_i32} : i32
      %1997 = arith.cmpi ule, %1973, %1945 {frisk.causal_mask_i32} : i32
      %1998 = arith.cmpi ule, %1973, %1948 {frisk.causal_mask_i32} : i32
      %1999 = arith.cmpi ule, %1973, %1951 {frisk.causal_mask_i32} : i32
      %2000 = arith.cmpi ule, %1973, %1954 {frisk.causal_mask_i32} : i32
      %2001 = arith.cmpi ule, %1973, %1957 {frisk.causal_mask_i32} : i32
      %2002 = arith.cmpi ule, %1973, %1960 {frisk.causal_mask_i32} : i32
      %2003 = arith.cmpi ule, %1973, %1963 {frisk.causal_mask_i32} : i32
      %2004 = arith.cmpi ule, %1973, %1966 {frisk.causal_mask_i32} : i32
      %2005 = arith.cmpi ule, %1973, %1969 {frisk.causal_mask_i32} : i32
      %2006 = vector.load %view_7[%898] : memref<4096xf16, 3>, vector<4xf16>
      %2007 = vector.load %view_7[%900] : memref<4096xf16, 3>, vector<4xf16>
      %2008 = vector.extract %2006[0] : f16 from vector<4xf16>
      %2009 = vector.extract %2006[1] : f16 from vector<4xf16>
      %2010 = vector.extract %2006[2] : f16 from vector<4xf16>
      %2011 = vector.extract %2006[3] : f16 from vector<4xf16>
      %2012 = vector.from_elements %2008, %2009, %2010, %2011 : vector<4xf16>
      %2013 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1848, %2012, %1855 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2014 = vector.extract %2007[0] : f16 from vector<4xf16>
      %2015 = vector.extract %2007[1] : f16 from vector<4xf16>
      %2016 = vector.extract %2007[2] : f16 from vector<4xf16>
      %2017 = vector.extract %2007[3] : f16 from vector<4xf16>
      %2018 = vector.from_elements %2014, %2015, %2016, %2017 : vector<4xf16>
      %2019 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1846, %2018, %1849 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2020 = vector.extract %2013[0] : f32 from vector<4xf32>
      %2021 = arith.select %1877, %cst_3, %2020 {frisk.causal_mask_select} : f32
      %2022 = vector.extract %2013[1] : f32 from vector<4xf32>
      %2023 = arith.select %1880, %cst_3, %2022 {frisk.causal_mask_select} : f32
      %2024 = arith.maxnumf %2021, %cst_3 : f32
      %2025 = vector.extract %2013[2] : f32 from vector<4xf32>
      %2026 = arith.select %1883, %cst_3, %2025 {frisk.causal_mask_select} : f32
      %2027 = arith.maxnumf %2024, %2023 : f32
      %2028 = vector.extract %2013[3] : f32 from vector<4xf32>
      %2029 = arith.select %1886, %cst_3, %2028 {frisk.causal_mask_select} : f32
      %2030 = arith.maxnumf %2027, %2026 : f32
      %2031 = arith.maxnumf %2030, %2029 : f32
      %2032 = vector.load %view_7[%918] : memref<4096xf16, 3>, vector<4xf16>
      %2033 = vector.load %view_7[%920] : memref<4096xf16, 3>, vector<4xf16>
      %2034 = vector.extract %2032[0] : f16 from vector<4xf16>
      %2035 = vector.extract %2032[1] : f16 from vector<4xf16>
      %2036 = vector.extract %2032[2] : f16 from vector<4xf16>
      %2037 = vector.extract %2032[3] : f16 from vector<4xf16>
      %2038 = vector.from_elements %2034, %2035, %2036, %2037 : vector<4xf16>
      %2039 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1848, %2038, %2019 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2040 = vector.extract %2033[0] : f16 from vector<4xf16>
      %2041 = vector.extract %2033[1] : f16 from vector<4xf16>
      %2042 = vector.extract %2033[2] : f16 from vector<4xf16>
      %2043 = vector.extract %2033[3] : f16 from vector<4xf16>
      %2044 = vector.from_elements %2040, %2041, %2042, %2043 : vector<4xf16>
      %2045 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1846, %2044, %1856 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2046 = vector.extract %2039[0] : f32 from vector<4xf32>
      %2047 = arith.select %1889, %cst_3, %2046 {frisk.causal_mask_select} : f32
      %2048 = vector.extract %2039[1] : f32 from vector<4xf32>
      %2049 = arith.select %1892, %cst_3, %2048 {frisk.causal_mask_select} : f32
      %2050 = arith.maxnumf %2031, %2047 : f32
      %2051 = vector.extract %2039[2] : f32 from vector<4xf32>
      %2052 = arith.select %1895, %cst_3, %2051 {frisk.causal_mask_select} : f32
      %2053 = arith.maxnumf %2050, %2049 : f32
      %2054 = vector.extract %2039[3] : f32 from vector<4xf32>
      %2055 = arith.select %1898, %cst_3, %2054 {frisk.causal_mask_select} : f32
      %2056 = arith.maxnumf %2053, %2052 : f32
      %2057 = arith.maxnumf %2056, %2055 : f32
      %2058 = vector.load %view_7[%938] : memref<4096xf16, 3>, vector<4xf16>
      %2059 = vector.load %view_7[%940] : memref<4096xf16, 3>, vector<4xf16>
      %2060 = vector.extract %2058[0] : f16 from vector<4xf16>
      %2061 = vector.extract %2058[1] : f16 from vector<4xf16>
      %2062 = vector.extract %2058[2] : f16 from vector<4xf16>
      %2063 = vector.extract %2058[3] : f16 from vector<4xf16>
      %2064 = vector.from_elements %2060, %2061, %2062, %2063 : vector<4xf16>
      %2065 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1848, %2064, %2045 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2066 = vector.extract %2059[0] : f16 from vector<4xf16>
      %2067 = vector.extract %2059[1] : f16 from vector<4xf16>
      %2068 = vector.extract %2059[2] : f16 from vector<4xf16>
      %2069 = vector.extract %2059[3] : f16 from vector<4xf16>
      %2070 = vector.from_elements %2066, %2067, %2068, %2069 : vector<4xf16>
      %2071 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1846, %2070, %1857 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2072 = vector.extract %2065[0] : f32 from vector<4xf32>
      %2073 = arith.select %1901, %cst_3, %2072 {frisk.causal_mask_select} : f32
      %2074 = vector.extract %2065[1] : f32 from vector<4xf32>
      %2075 = arith.select %1904, %cst_3, %2074 {frisk.causal_mask_select} : f32
      %2076 = arith.maxnumf %2057, %2073 : f32
      %2077 = vector.extract %2065[2] : f32 from vector<4xf32>
      %2078 = arith.select %1907, %cst_3, %2077 {frisk.causal_mask_select} : f32
      %2079 = arith.maxnumf %2076, %2075 : f32
      %2080 = vector.extract %2065[3] : f32 from vector<4xf32>
      %2081 = arith.select %1910, %cst_3, %2080 {frisk.causal_mask_select} : f32
      %2082 = arith.maxnumf %2079, %2078 : f32
      %2083 = arith.maxnumf %2082, %2081 : f32
      %2084 = vector.load %view_7[%958] : memref<4096xf16, 3>, vector<4xf16>
      %2085 = vector.load %view_7[%960] : memref<4096xf16, 3>, vector<4xf16>
      %2086 = vector.extract %2084[0] : f16 from vector<4xf16>
      %2087 = vector.extract %2084[1] : f16 from vector<4xf16>
      %2088 = vector.extract %2084[2] : f16 from vector<4xf16>
      %2089 = vector.extract %2084[3] : f16 from vector<4xf16>
      %2090 = vector.from_elements %2086, %2087, %2088, %2089 : vector<4xf16>
      %2091 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1848, %2090, %2071 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2092 = vector.extract %2085[0] : f16 from vector<4xf16>
      %2093 = vector.extract %2085[1] : f16 from vector<4xf16>
      %2094 = vector.extract %2085[2] : f16 from vector<4xf16>
      %2095 = vector.extract %2085[3] : f16 from vector<4xf16>
      %2096 = vector.from_elements %2092, %2093, %2094, %2095 : vector<4xf16>
      %2097 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1846, %2096, %1858 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2098 = vector.extract %2091[0] : f32 from vector<4xf32>
      %2099 = arith.select %1913, %cst_3, %2098 {frisk.causal_mask_select} : f32
      %2100 = vector.extract %2091[1] : f32 from vector<4xf32>
      %2101 = arith.select %1916, %cst_3, %2100 {frisk.causal_mask_select} : f32
      %2102 = arith.maxnumf %2083, %2099 : f32
      %2103 = vector.extract %2091[2] : f32 from vector<4xf32>
      %2104 = arith.select %1919, %cst_3, %2103 {frisk.causal_mask_select} : f32
      %2105 = arith.maxnumf %2102, %2101 : f32
      %2106 = vector.extract %2091[3] : f32 from vector<4xf32>
      %2107 = arith.select %1922, %cst_3, %2106 {frisk.causal_mask_select} : f32
      %2108 = arith.maxnumf %2105, %2104 : f32
      %2109 = arith.maxnumf %2108, %2107 : f32
      %2110 = vector.load %view_7[%978] : memref<4096xf16, 3>, vector<4xf16>
      %2111 = vector.load %view_7[%980] : memref<4096xf16, 3>, vector<4xf16>
      %2112 = vector.extract %2110[0] : f16 from vector<4xf16>
      %2113 = vector.extract %2110[1] : f16 from vector<4xf16>
      %2114 = vector.extract %2110[2] : f16 from vector<4xf16>
      %2115 = vector.extract %2110[3] : f16 from vector<4xf16>
      %2116 = vector.from_elements %2112, %2113, %2114, %2115 : vector<4xf16>
      %2117 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1848, %2116, %2097 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2118 = vector.extract %2111[0] : f16 from vector<4xf16>
      %2119 = vector.extract %2111[1] : f16 from vector<4xf16>
      %2120 = vector.extract %2111[2] : f16 from vector<4xf16>
      %2121 = vector.extract %2111[3] : f16 from vector<4xf16>
      %2122 = vector.from_elements %2118, %2119, %2120, %2121 : vector<4xf16>
      %2123 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1846, %2122, %1859 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2124 = vector.extract %2117[0] : f32 from vector<4xf32>
      %2125 = arith.select %1925, %cst_3, %2124 {frisk.causal_mask_select} : f32
      %2126 = vector.extract %2117[1] : f32 from vector<4xf32>
      %2127 = arith.select %1928, %cst_3, %2126 {frisk.causal_mask_select} : f32
      %2128 = arith.maxnumf %2109, %2125 : f32
      %2129 = vector.extract %2117[2] : f32 from vector<4xf32>
      %2130 = arith.select %1931, %cst_3, %2129 {frisk.causal_mask_select} : f32
      %2131 = arith.maxnumf %2128, %2127 : f32
      %2132 = vector.extract %2117[3] : f32 from vector<4xf32>
      %2133 = arith.select %1934, %cst_3, %2132 {frisk.causal_mask_select} : f32
      %2134 = arith.maxnumf %2131, %2130 : f32
      %2135 = arith.maxnumf %2134, %2133 : f32
      %2136 = vector.load %view_7[%998] : memref<4096xf16, 3>, vector<4xf16>
      %2137 = vector.load %view_7[%1000] : memref<4096xf16, 3>, vector<4xf16>
      %2138 = vector.extract %2136[0] : f16 from vector<4xf16>
      %2139 = vector.extract %2136[1] : f16 from vector<4xf16>
      %2140 = vector.extract %2136[2] : f16 from vector<4xf16>
      %2141 = vector.extract %2136[3] : f16 from vector<4xf16>
      %2142 = vector.from_elements %2138, %2139, %2140, %2141 : vector<4xf16>
      %2143 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1848, %2142, %2123 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2144 = vector.extract %2137[0] : f16 from vector<4xf16>
      %2145 = vector.extract %2137[1] : f16 from vector<4xf16>
      %2146 = vector.extract %2137[2] : f16 from vector<4xf16>
      %2147 = vector.extract %2137[3] : f16 from vector<4xf16>
      %2148 = vector.from_elements %2144, %2145, %2146, %2147 : vector<4xf16>
      %2149 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1846, %2148, %1860 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2150 = vector.extract %2143[0] : f32 from vector<4xf32>
      %2151 = arith.select %1937, %cst_3, %2150 {frisk.causal_mask_select} : f32
      %2152 = vector.extract %2143[1] : f32 from vector<4xf32>
      %2153 = arith.select %1940, %cst_3, %2152 {frisk.causal_mask_select} : f32
      %2154 = arith.maxnumf %2135, %2151 : f32
      %2155 = vector.extract %2143[2] : f32 from vector<4xf32>
      %2156 = arith.select %1943, %cst_3, %2155 {frisk.causal_mask_select} : f32
      %2157 = arith.maxnumf %2154, %2153 : f32
      %2158 = vector.extract %2143[3] : f32 from vector<4xf32>
      %2159 = arith.select %1946, %cst_3, %2158 {frisk.causal_mask_select} : f32
      %2160 = arith.maxnumf %2157, %2156 : f32
      %2161 = arith.maxnumf %2160, %2159 : f32
      %2162 = vector.load %view_7[%1018] : memref<4096xf16, 3>, vector<4xf16>
      %2163 = vector.load %view_7[%1020] : memref<4096xf16, 3>, vector<4xf16>
      %2164 = vector.extract %2162[0] : f16 from vector<4xf16>
      %2165 = vector.extract %2162[1] : f16 from vector<4xf16>
      %2166 = vector.extract %2162[2] : f16 from vector<4xf16>
      %2167 = vector.extract %2162[3] : f16 from vector<4xf16>
      %2168 = vector.from_elements %2164, %2165, %2166, %2167 : vector<4xf16>
      %2169 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1848, %2168, %2149 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2170 = vector.extract %2163[0] : f16 from vector<4xf16>
      %2171 = vector.extract %2163[1] : f16 from vector<4xf16>
      %2172 = vector.extract %2163[2] : f16 from vector<4xf16>
      %2173 = vector.extract %2163[3] : f16 from vector<4xf16>
      %2174 = vector.from_elements %2170, %2171, %2172, %2173 : vector<4xf16>
      %2175 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1846, %2174, %1861 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2176 = vector.extract %2169[0] : f32 from vector<4xf32>
      %2177 = arith.select %1949, %cst_3, %2176 {frisk.causal_mask_select} : f32
      %2178 = vector.extract %2169[1] : f32 from vector<4xf32>
      %2179 = arith.select %1952, %cst_3, %2178 {frisk.causal_mask_select} : f32
      %2180 = arith.maxnumf %2161, %2177 : f32
      %2181 = vector.extract %2169[2] : f32 from vector<4xf32>
      %2182 = arith.select %1955, %cst_3, %2181 {frisk.causal_mask_select} : f32
      %2183 = arith.maxnumf %2180, %2179 : f32
      %2184 = vector.extract %2169[3] : f32 from vector<4xf32>
      %2185 = arith.select %1958, %cst_3, %2184 {frisk.causal_mask_select} : f32
      %2186 = arith.maxnumf %2183, %2182 : f32
      %2187 = arith.maxnumf %2186, %2185 : f32
      %2188 = vector.load %view_7[%1038] : memref<4096xf16, 3>, vector<4xf16>
      %2189 = vector.load %view_7[%886] : memref<4096xf16, 3>, vector<4xf16>
      %2190 = vector.extract %2188[0] : f16 from vector<4xf16>
      %2191 = vector.extract %2188[1] : f16 from vector<4xf16>
      %2192 = vector.extract %2188[2] : f16 from vector<4xf16>
      %2193 = vector.extract %2188[3] : f16 from vector<4xf16>
      %2194 = vector.from_elements %2190, %2191, %2192, %2193 : vector<4xf16>
      %2195 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1848, %2194, %2175 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2196 = vector.extract %2189[0] : f16 from vector<4xf16>
      %2197 = vector.extract %2189[1] : f16 from vector<4xf16>
      %2198 = vector.extract %2189[2] : f16 from vector<4xf16>
      %2199 = vector.extract %2189[3] : f16 from vector<4xf16>
      %2200 = vector.from_elements %2196, %2197, %2198, %2199 : vector<4xf16>
      %2201 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1862, %2200, %1863 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2202 = vector.extract %2195[0] : f32 from vector<4xf32>
      %2203 = arith.select %1961, %cst_3, %2202 {frisk.causal_mask_select} : f32
      %2204 = vector.extract %2195[1] : f32 from vector<4xf32>
      %2205 = arith.select %1964, %cst_3, %2204 {frisk.causal_mask_select} : f32
      %2206 = arith.maxnumf %2187, %2203 : f32
      %2207 = vector.extract %2195[2] : f32 from vector<4xf32>
      %2208 = arith.select %1967, %cst_3, %2207 {frisk.causal_mask_select} : f32
      %2209 = arith.maxnumf %2206, %2205 : f32
      %2210 = vector.extract %2195[3] : f32 from vector<4xf32>
      %2211 = arith.select %1970, %cst_3, %2210 {frisk.causal_mask_select} : f32
      %2212 = arith.maxnumf %2209, %2208 : f32
      %2213 = arith.maxnumf %2212, %2211 : f32
      %2214 = vector.load %view_7[%898] : memref<4096xf16, 3>, vector<4xf16>
      %2215 = vector.load %view_7[%900] : memref<4096xf16, 3>, vector<4xf16>
      %2216 = vector.extract %2214[0] : f16 from vector<4xf16>
      %2217 = vector.extract %2214[1] : f16 from vector<4xf16>
      %2218 = vector.extract %2214[2] : f16 from vector<4xf16>
      %2219 = vector.extract %2214[3] : f16 from vector<4xf16>
      %2220 = vector.from_elements %2216, %2217, %2218, %2219 : vector<4xf16>
      %2221 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1864, %2220, %2201 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2222 = vector.extract %2215[0] : f16 from vector<4xf16>
      %2223 = vector.extract %2215[1] : f16 from vector<4xf16>
      %2224 = vector.extract %2215[2] : f16 from vector<4xf16>
      %2225 = vector.extract %2215[3] : f16 from vector<4xf16>
      %2226 = vector.from_elements %2222, %2223, %2224, %2225 : vector<4xf16>
      %2227 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1862, %2226, %1865 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2228 = vector.extract %2221[0] : f32 from vector<4xf32>
      %2229 = arith.select %1974, %cst_3, %2228 {frisk.causal_mask_select} : f32
      %2230 = vector.extract %2221[1] : f32 from vector<4xf32>
      %2231 = arith.select %1975, %cst_3, %2230 {frisk.causal_mask_select} : f32
      %2232 = vector.extract %2221[2] : f32 from vector<4xf32>
      %2233 = arith.select %1976, %cst_3, %2232 {frisk.causal_mask_select} : f32
      %2234 = vector.extract %2221[3] : f32 from vector<4xf32>
      %2235 = arith.select %1977, %cst_3, %2234 {frisk.causal_mask_select} : f32
      %2236 = vector.load %view_7[%918] : memref<4096xf16, 3>, vector<4xf16>
      %2237 = vector.load %view_7[%920] : memref<4096xf16, 3>, vector<4xf16>
      %2238 = vector.extract %2236[0] : f16 from vector<4xf16>
      %2239 = vector.extract %2236[1] : f16 from vector<4xf16>
      %2240 = vector.extract %2236[2] : f16 from vector<4xf16>
      %2241 = vector.extract %2236[3] : f16 from vector<4xf16>
      %2242 = vector.from_elements %2238, %2239, %2240, %2241 : vector<4xf16>
      %2243 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1864, %2242, %2227 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2244 = vector.extract %2237[0] : f16 from vector<4xf16>
      %2245 = vector.extract %2237[1] : f16 from vector<4xf16>
      %2246 = vector.extract %2237[2] : f16 from vector<4xf16>
      %2247 = vector.extract %2237[3] : f16 from vector<4xf16>
      %2248 = vector.from_elements %2244, %2245, %2246, %2247 : vector<4xf16>
      %2249 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1862, %2248, %1866 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2250 = vector.extract %2243[0] : f32 from vector<4xf32>
      %2251 = arith.select %1978, %cst_3, %2250 {frisk.causal_mask_select} : f32
      %2252 = vector.extract %2243[1] : f32 from vector<4xf32>
      %2253 = arith.select %1979, %cst_3, %2252 {frisk.causal_mask_select} : f32
      %2254 = vector.extract %2243[2] : f32 from vector<4xf32>
      %2255 = arith.select %1980, %cst_3, %2254 {frisk.causal_mask_select} : f32
      %2256 = vector.extract %2243[3] : f32 from vector<4xf32>
      %2257 = arith.select %1981, %cst_3, %2256 {frisk.causal_mask_select} : f32
      %2258 = vector.load %view_7[%938] : memref<4096xf16, 3>, vector<4xf16>
      %2259 = vector.load %view_7[%940] : memref<4096xf16, 3>, vector<4xf16>
      %2260 = vector.extract %2258[0] : f16 from vector<4xf16>
      %2261 = vector.extract %2258[1] : f16 from vector<4xf16>
      %2262 = vector.extract %2258[2] : f16 from vector<4xf16>
      %2263 = vector.extract %2258[3] : f16 from vector<4xf16>
      %2264 = vector.from_elements %2260, %2261, %2262, %2263 : vector<4xf16>
      %2265 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1864, %2264, %2249 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2266 = vector.extract %2259[0] : f16 from vector<4xf16>
      %2267 = vector.extract %2259[1] : f16 from vector<4xf16>
      %2268 = vector.extract %2259[2] : f16 from vector<4xf16>
      %2269 = vector.extract %2259[3] : f16 from vector<4xf16>
      %2270 = vector.from_elements %2266, %2267, %2268, %2269 : vector<4xf16>
      %2271 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1862, %2270, %1867 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2272 = vector.extract %2265[0] : f32 from vector<4xf32>
      %2273 = arith.select %1982, %cst_3, %2272 {frisk.causal_mask_select} : f32
      %2274 = vector.extract %2265[1] : f32 from vector<4xf32>
      %2275 = arith.select %1983, %cst_3, %2274 {frisk.causal_mask_select} : f32
      %2276 = vector.extract %2265[2] : f32 from vector<4xf32>
      %2277 = arith.select %1984, %cst_3, %2276 {frisk.causal_mask_select} : f32
      %2278 = vector.extract %2265[3] : f32 from vector<4xf32>
      %2279 = arith.select %1985, %cst_3, %2278 {frisk.causal_mask_select} : f32
      %2280 = vector.load %view_7[%958] : memref<4096xf16, 3>, vector<4xf16>
      %2281 = vector.load %view_7[%960] : memref<4096xf16, 3>, vector<4xf16>
      %2282 = vector.extract %2280[0] : f16 from vector<4xf16>
      %2283 = vector.extract %2280[1] : f16 from vector<4xf16>
      %2284 = vector.extract %2280[2] : f16 from vector<4xf16>
      %2285 = vector.extract %2280[3] : f16 from vector<4xf16>
      %2286 = vector.from_elements %2282, %2283, %2284, %2285 : vector<4xf16>
      %2287 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1864, %2286, %2271 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2288 = vector.extract %2281[0] : f16 from vector<4xf16>
      %2289 = vector.extract %2281[1] : f16 from vector<4xf16>
      %2290 = vector.extract %2281[2] : f16 from vector<4xf16>
      %2291 = vector.extract %2281[3] : f16 from vector<4xf16>
      %2292 = vector.from_elements %2288, %2289, %2290, %2291 : vector<4xf16>
      %2293 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1862, %2292, %1868 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2294 = vector.extract %2287[0] : f32 from vector<4xf32>
      %2295 = arith.select %1986, %cst_3, %2294 {frisk.causal_mask_select} : f32
      %2296 = vector.extract %2287[1] : f32 from vector<4xf32>
      %2297 = arith.select %1987, %cst_3, %2296 {frisk.causal_mask_select} : f32
      %2298 = vector.extract %2287[2] : f32 from vector<4xf32>
      %2299 = arith.select %1988, %cst_3, %2298 {frisk.causal_mask_select} : f32
      %2300 = vector.extract %2287[3] : f32 from vector<4xf32>
      %2301 = arith.select %1989, %cst_3, %2300 {frisk.causal_mask_select} : f32
      %2302 = vector.load %view_7[%978] : memref<4096xf16, 3>, vector<4xf16>
      %2303 = vector.load %view_7[%980] : memref<4096xf16, 3>, vector<4xf16>
      %2304 = vector.extract %2302[0] : f16 from vector<4xf16>
      %2305 = vector.extract %2302[1] : f16 from vector<4xf16>
      %2306 = vector.extract %2302[2] : f16 from vector<4xf16>
      %2307 = vector.extract %2302[3] : f16 from vector<4xf16>
      %2308 = vector.from_elements %2304, %2305, %2306, %2307 : vector<4xf16>
      %2309 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1864, %2308, %2293 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2310 = vector.extract %2303[0] : f16 from vector<4xf16>
      %2311 = vector.extract %2303[1] : f16 from vector<4xf16>
      %2312 = vector.extract %2303[2] : f16 from vector<4xf16>
      %2313 = vector.extract %2303[3] : f16 from vector<4xf16>
      %2314 = vector.from_elements %2310, %2311, %2312, %2313 : vector<4xf16>
      %2315 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1862, %2314, %1869 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2316 = vector.extract %2309[0] : f32 from vector<4xf32>
      %2317 = arith.select %1990, %cst_3, %2316 {frisk.causal_mask_select} : f32
      %2318 = vector.extract %2309[1] : f32 from vector<4xf32>
      %2319 = arith.select %1991, %cst_3, %2318 {frisk.causal_mask_select} : f32
      %2320 = vector.extract %2309[2] : f32 from vector<4xf32>
      %2321 = arith.select %1992, %cst_3, %2320 {frisk.causal_mask_select} : f32
      %2322 = vector.extract %2309[3] : f32 from vector<4xf32>
      %2323 = arith.select %1993, %cst_3, %2322 {frisk.causal_mask_select} : f32
      %2324 = vector.load %view_7[%998] : memref<4096xf16, 3>, vector<4xf16>
      %2325 = vector.load %view_7[%1000] : memref<4096xf16, 3>, vector<4xf16>
      %2326 = vector.extract %2324[0] : f16 from vector<4xf16>
      %2327 = vector.extract %2324[1] : f16 from vector<4xf16>
      %2328 = vector.extract %2324[2] : f16 from vector<4xf16>
      %2329 = vector.extract %2324[3] : f16 from vector<4xf16>
      %2330 = vector.from_elements %2326, %2327, %2328, %2329 : vector<4xf16>
      %2331 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1864, %2330, %2315 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2332 = vector.extract %2325[0] : f16 from vector<4xf16>
      %2333 = vector.extract %2325[1] : f16 from vector<4xf16>
      %2334 = vector.extract %2325[2] : f16 from vector<4xf16>
      %2335 = vector.extract %2325[3] : f16 from vector<4xf16>
      %2336 = vector.from_elements %2332, %2333, %2334, %2335 : vector<4xf16>
      %2337 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1862, %2336, %1870 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2338 = vector.extract %2331[0] : f32 from vector<4xf32>
      %2339 = arith.select %1994, %cst_3, %2338 {frisk.causal_mask_select} : f32
      %2340 = vector.extract %2331[1] : f32 from vector<4xf32>
      %2341 = arith.select %1995, %cst_3, %2340 {frisk.causal_mask_select} : f32
      %2342 = vector.extract %2331[2] : f32 from vector<4xf32>
      %2343 = arith.select %1996, %cst_3, %2342 {frisk.causal_mask_select} : f32
      %2344 = vector.extract %2331[3] : f32 from vector<4xf32>
      %2345 = arith.select %1997, %cst_3, %2344 {frisk.causal_mask_select} : f32
      %2346 = vector.load %view_7[%1018] : memref<4096xf16, 3>, vector<4xf16>
      %2347 = vector.load %view_7[%1020] : memref<4096xf16, 3>, vector<4xf16>
      %2348 = vector.extract %2346[0] : f16 from vector<4xf16>
      %2349 = vector.extract %2346[1] : f16 from vector<4xf16>
      %2350 = vector.extract %2346[2] : f16 from vector<4xf16>
      %2351 = vector.extract %2346[3] : f16 from vector<4xf16>
      %2352 = vector.from_elements %2348, %2349, %2350, %2351 : vector<4xf16>
      %2353 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1864, %2352, %2337 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2354 = vector.extract %2347[0] : f16 from vector<4xf16>
      %2355 = vector.extract %2347[1] : f16 from vector<4xf16>
      %2356 = vector.extract %2347[2] : f16 from vector<4xf16>
      %2357 = vector.extract %2347[3] : f16 from vector<4xf16>
      %2358 = vector.from_elements %2354, %2355, %2356, %2357 : vector<4xf16>
      %2359 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1862, %2358, %1871 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2360 = vector.extract %2353[0] : f32 from vector<4xf32>
      %2361 = arith.select %1998, %cst_3, %2360 {frisk.causal_mask_select} : f32
      %2362 = vector.extract %2353[1] : f32 from vector<4xf32>
      %2363 = arith.select %1999, %cst_3, %2362 {frisk.causal_mask_select} : f32
      %2364 = vector.extract %2353[2] : f32 from vector<4xf32>
      %2365 = arith.select %2000, %cst_3, %2364 {frisk.causal_mask_select} : f32
      %2366 = vector.extract %2353[3] : f32 from vector<4xf32>
      %2367 = arith.select %2001, %cst_3, %2366 {frisk.causal_mask_select} : f32
      %2368 = vector.load %view_7[%1038] : memref<4096xf16, 3>, vector<4xf16>
      %2369 = vector.extract %2368[0] : f16 from vector<4xf16>
      %2370 = vector.extract %2368[1] : f16 from vector<4xf16>
      %2371 = vector.extract %2368[2] : f16 from vector<4xf16>
      %2372 = vector.extract %2368[3] : f16 from vector<4xf16>
      %2373 = vector.from_elements %2369, %2370, %2371, %2372 : vector<4xf16>
      %2374 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1864, %2373, %2359 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2375 = vector.extract %2374[0] : f32 from vector<4xf32>
      %2376 = arith.select %2002, %cst_3, %2375 {frisk.causal_mask_select} : f32
      %2377 = vector.extract %2374[1] : f32 from vector<4xf32>
      %2378 = arith.select %2003, %cst_3, %2377 {frisk.causal_mask_select} : f32
      %2379 = vector.extract %2374[2] : f32 from vector<4xf32>
      %2380 = arith.select %2004, %cst_3, %2379 {frisk.causal_mask_select} : f32
      %2381 = vector.extract %2374[3] : f32 from vector<4xf32>
      %2382 = arith.select %2005, %cst_3, %2381 {frisk.causal_mask_select} : f32
      %2383 = llvm.bitcast %2213 : f32 to i32
      %2384 = rocdl.ds_bpermute %10, %2383 : (i32, i32) -> i32
      %2385 = llvm.bitcast %2384 : i32 to f32
      %2386 = arith.maxnumf %2213, %2385 : f32
      %2387 = llvm.bitcast %2386 : f32 to i32
      %2388 = rocdl.ds_bpermute %14, %2387 : (i32, i32) -> i32
      %2389 = llvm.bitcast %2388 : i32 to f32
      %2390 = arith.maxnumf %2386, %2389 : f32
      %2391 = arith.maxnumf %2229, %cst_3 : f32
      %2392 = arith.maxnumf %2391, %2231 : f32
      %2393 = arith.maxnumf %2392, %2233 : f32
      %2394 = arith.maxnumf %2393, %2235 : f32
      %2395 = arith.maxnumf %2394, %2251 : f32
      %2396 = arith.maxnumf %2395, %2253 : f32
      %2397 = arith.maxnumf %2396, %2255 : f32
      %2398 = arith.maxnumf %2397, %2257 : f32
      %2399 = arith.maxnumf %2398, %2273 : f32
      %2400 = arith.maxnumf %2399, %2275 : f32
      %2401 = arith.maxnumf %2400, %2277 : f32
      %2402 = arith.maxnumf %2401, %2279 : f32
      %2403 = arith.maxnumf %2402, %2295 : f32
      %2404 = arith.maxnumf %2403, %2297 : f32
      %2405 = arith.maxnumf %2404, %2299 : f32
      %2406 = arith.maxnumf %2405, %2301 : f32
      %2407 = arith.maxnumf %2406, %2317 : f32
      %2408 = arith.maxnumf %2407, %2319 : f32
      %2409 = arith.maxnumf %2408, %2321 : f32
      %2410 = arith.maxnumf %2409, %2323 : f32
      %2411 = arith.maxnumf %2410, %2339 : f32
      %2412 = arith.maxnumf %2411, %2341 : f32
      %2413 = arith.maxnumf %2412, %2343 : f32
      %2414 = arith.maxnumf %2413, %2345 : f32
      %2415 = arith.maxnumf %2414, %2361 : f32
      %2416 = arith.maxnumf %2415, %2363 : f32
      %2417 = arith.maxnumf %2416, %2365 : f32
      %2418 = arith.maxnumf %2417, %2367 : f32
      %2419 = arith.maxnumf %2418, %2376 : f32
      %2420 = arith.maxnumf %2419, %2378 : f32
      %2421 = arith.maxnumf %2420, %2380 : f32
      %2422 = arith.maxnumf %2421, %2382 : f32
      %2423 = llvm.bitcast %2422 : f32 to i32
      %2424 = rocdl.ds_bpermute %10, %2423 : (i32, i32) -> i32
      %2425 = llvm.bitcast %2424 : i32 to f32
      %2426 = arith.maxnumf %2422, %2425 : f32
      %2427 = llvm.bitcast %2426 : f32 to i32
      %2428 = rocdl.ds_bpermute %14, %2427 : (i32, i32) -> i32
      %2429 = llvm.bitcast %2428 : i32 to f32
      %2430 = arith.maxnumf %2426, %2429 : f32
      %2431 = vector.extract %arg7[0, 0] : f32 from vector<2x1xf32>
      %2432 = vector.broadcast %2390 : f32 to vector<1xf32>
      %2433 = vector.broadcast %2431 : f32 to vector<1xf32>
      %2434 = arith.maxnumf %2432, %2433 {origin = "frisk.max", tiled = true} : vector<1xf32>
      %2435 = vector.extract %2434[0] : f32 from vector<1xf32>
      %2436 = vector.extract %arg7[1, 0] : f32 from vector<2x1xf32>
      %2437 = vector.broadcast %2430 : f32 to vector<1xf32>
      %2438 = vector.broadcast %2436 : f32 to vector<1xf32>
      %2439 = arith.maxnumf %2437, %2438 {origin = "frisk.max", tiled = true} : vector<1xf32>
      %2440 = vector.extract %2439[0] : f32 from vector<1xf32>
      %2441 = vector.from_elements %2021, %2023, %2026, %2029 : vector<4xf32>
      %2442 = vector.broadcast %2435 : f32 to vector<4xf32>
      %2443 = arith.subf %2441, %2442 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2444 = vector.extract %2443[0] : f32 from vector<4xf32>
      %2445 = vector.extract %2443[1] : f32 from vector<4xf32>
      %2446 = vector.extract %2443[2] : f32 from vector<4xf32>
      %2447 = vector.extract %2443[3] : f32 from vector<4xf32>
      %2448 = vector.from_elements %2047, %2049, %2052, %2055 : vector<4xf32>
      %2449 = arith.subf %2448, %2442 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2450 = vector.extract %2449[0] : f32 from vector<4xf32>
      %2451 = vector.extract %2449[1] : f32 from vector<4xf32>
      %2452 = vector.extract %2449[2] : f32 from vector<4xf32>
      %2453 = vector.extract %2449[3] : f32 from vector<4xf32>
      %2454 = vector.from_elements %2073, %2075, %2078, %2081 : vector<4xf32>
      %2455 = arith.subf %2454, %2442 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2456 = vector.extract %2455[0] : f32 from vector<4xf32>
      %2457 = vector.extract %2455[1] : f32 from vector<4xf32>
      %2458 = vector.extract %2455[2] : f32 from vector<4xf32>
      %2459 = vector.extract %2455[3] : f32 from vector<4xf32>
      %2460 = vector.from_elements %2099, %2101, %2104, %2107 : vector<4xf32>
      %2461 = arith.subf %2460, %2442 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2462 = vector.extract %2461[0] : f32 from vector<4xf32>
      %2463 = vector.extract %2461[1] : f32 from vector<4xf32>
      %2464 = vector.extract %2461[2] : f32 from vector<4xf32>
      %2465 = vector.extract %2461[3] : f32 from vector<4xf32>
      %2466 = vector.from_elements %2125, %2127, %2130, %2133 : vector<4xf32>
      %2467 = arith.subf %2466, %2442 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2468 = vector.extract %2467[0] : f32 from vector<4xf32>
      %2469 = vector.extract %2467[1] : f32 from vector<4xf32>
      %2470 = vector.extract %2467[2] : f32 from vector<4xf32>
      %2471 = vector.extract %2467[3] : f32 from vector<4xf32>
      %2472 = vector.from_elements %2151, %2153, %2156, %2159 : vector<4xf32>
      %2473 = arith.subf %2472, %2442 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2474 = vector.extract %2473[0] : f32 from vector<4xf32>
      %2475 = vector.extract %2473[1] : f32 from vector<4xf32>
      %2476 = vector.extract %2473[2] : f32 from vector<4xf32>
      %2477 = vector.extract %2473[3] : f32 from vector<4xf32>
      %2478 = vector.from_elements %2177, %2179, %2182, %2185 : vector<4xf32>
      %2479 = arith.subf %2478, %2442 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2480 = vector.extract %2479[0] : f32 from vector<4xf32>
      %2481 = vector.extract %2479[1] : f32 from vector<4xf32>
      %2482 = vector.extract %2479[2] : f32 from vector<4xf32>
      %2483 = vector.extract %2479[3] : f32 from vector<4xf32>
      %2484 = vector.from_elements %2203, %2205, %2208, %2211 : vector<4xf32>
      %2485 = arith.subf %2484, %2442 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2486 = vector.extract %2485[0] : f32 from vector<4xf32>
      %2487 = vector.extract %2485[1] : f32 from vector<4xf32>
      %2488 = vector.extract %2485[2] : f32 from vector<4xf32>
      %2489 = vector.extract %2485[3] : f32 from vector<4xf32>
      %2490 = vector.from_elements %2229, %2231, %2233, %2235 : vector<4xf32>
      %2491 = vector.broadcast %2440 : f32 to vector<4xf32>
      %2492 = arith.subf %2490, %2491 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2493 = vector.extract %2492[0] : f32 from vector<4xf32>
      %2494 = vector.extract %2492[1] : f32 from vector<4xf32>
      %2495 = vector.extract %2492[2] : f32 from vector<4xf32>
      %2496 = vector.extract %2492[3] : f32 from vector<4xf32>
      %2497 = vector.from_elements %2251, %2253, %2255, %2257 : vector<4xf32>
      %2498 = arith.subf %2497, %2491 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2499 = vector.extract %2498[0] : f32 from vector<4xf32>
      %2500 = vector.extract %2498[1] : f32 from vector<4xf32>
      %2501 = vector.extract %2498[2] : f32 from vector<4xf32>
      %2502 = vector.extract %2498[3] : f32 from vector<4xf32>
      %2503 = vector.from_elements %2273, %2275, %2277, %2279 : vector<4xf32>
      %2504 = arith.subf %2503, %2491 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2505 = vector.extract %2504[0] : f32 from vector<4xf32>
      %2506 = vector.extract %2504[1] : f32 from vector<4xf32>
      %2507 = vector.extract %2504[2] : f32 from vector<4xf32>
      %2508 = vector.extract %2504[3] : f32 from vector<4xf32>
      %2509 = vector.from_elements %2295, %2297, %2299, %2301 : vector<4xf32>
      %2510 = arith.subf %2509, %2491 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2511 = vector.extract %2510[0] : f32 from vector<4xf32>
      %2512 = vector.extract %2510[1] : f32 from vector<4xf32>
      %2513 = vector.extract %2510[2] : f32 from vector<4xf32>
      %2514 = vector.extract %2510[3] : f32 from vector<4xf32>
      %2515 = vector.from_elements %2317, %2319, %2321, %2323 : vector<4xf32>
      %2516 = arith.subf %2515, %2491 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2517 = vector.extract %2516[0] : f32 from vector<4xf32>
      %2518 = vector.extract %2516[1] : f32 from vector<4xf32>
      %2519 = vector.extract %2516[2] : f32 from vector<4xf32>
      %2520 = vector.extract %2516[3] : f32 from vector<4xf32>
      %2521 = vector.from_elements %2339, %2341, %2343, %2345 : vector<4xf32>
      %2522 = arith.subf %2521, %2491 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2523 = vector.extract %2522[0] : f32 from vector<4xf32>
      %2524 = vector.extract %2522[1] : f32 from vector<4xf32>
      %2525 = vector.extract %2522[2] : f32 from vector<4xf32>
      %2526 = vector.extract %2522[3] : f32 from vector<4xf32>
      %2527 = vector.from_elements %2361, %2363, %2365, %2367 : vector<4xf32>
      %2528 = arith.subf %2527, %2491 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2529 = vector.extract %2528[0] : f32 from vector<4xf32>
      %2530 = vector.extract %2528[1] : f32 from vector<4xf32>
      %2531 = vector.extract %2528[2] : f32 from vector<4xf32>
      %2532 = vector.extract %2528[3] : f32 from vector<4xf32>
      %2533 = vector.from_elements %2376, %2378, %2380, %2382 : vector<4xf32>
      %2534 = arith.subf %2533, %2491 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2535 = vector.extract %2534[0] : f32 from vector<4xf32>
      %2536 = vector.extract %2534[1] : f32 from vector<4xf32>
      %2537 = vector.extract %2534[2] : f32 from vector<4xf32>
      %2538 = vector.extract %2534[3] : f32 from vector<4xf32>
      %2539 = vector.broadcast %2435 : f32 to vector<1xf32>
      %2540 = arith.subf %2433, %2539 {origin = "frisk.sub", tiled = true} : vector<1xf32>
      %2541 = math.exp2 %2540 {tiled = true} : vector<1xf32>
      %2542 = vector.extract %2541[0] : f32 from vector<1xf32>
      %2543 = vector.extract %arg6[0, 0] : f32 from vector<2x1xf32>
      %2544 = vector.broadcast %2543 : f32 to vector<1xf32>
      %2545 = arith.mulf %2544, %2541 {origin = "frisk.mul", tiled = true} : vector<1xf32>
      %2546 = vector.extract %2545[0] : f32 from vector<1xf32>
      %2547 = vector.broadcast %2440 : f32 to vector<1xf32>
      %2548 = arith.subf %2438, %2547 {origin = "frisk.sub", tiled = true} : vector<1xf32>
      %2549 = math.exp2 %2548 {tiled = true} : vector<1xf32>
      %2550 = vector.extract %2549[0] : f32 from vector<1xf32>
      %2551 = vector.extract %arg6[1, 0] : f32 from vector<2x1xf32>
      %2552 = vector.broadcast %2551 : f32 to vector<1xf32>
      %2553 = arith.mulf %2552, %2549 {origin = "frisk.mul", tiled = true} : vector<1xf32>
      %2554 = vector.extract %2553[0] : f32 from vector<1xf32>
      %2555 = vector.from_elements %2444, %2445, %2446, %2447 : vector<4xf32>
      %2556 = math.exp2 %2555 {tiled = true} : vector<4xf32>
      %2557 = vector.extract %2556[0] : f32 from vector<4xf32>
      %2558 = vector.extract %2556[1] : f32 from vector<4xf32>
      %2559 = vector.extract %2556[2] : f32 from vector<4xf32>
      %2560 = vector.extract %2556[3] : f32 from vector<4xf32>
      %2561 = vector.from_elements %2450, %2451, %2452, %2453 : vector<4xf32>
      %2562 = math.exp2 %2561 {tiled = true} : vector<4xf32>
      %2563 = vector.extract %2562[0] : f32 from vector<4xf32>
      %2564 = vector.extract %2562[1] : f32 from vector<4xf32>
      %2565 = vector.extract %2562[2] : f32 from vector<4xf32>
      %2566 = vector.extract %2562[3] : f32 from vector<4xf32>
      %2567 = vector.from_elements %2456, %2457, %2458, %2459 : vector<4xf32>
      %2568 = math.exp2 %2567 {tiled = true} : vector<4xf32>
      %2569 = vector.extract %2568[0] : f32 from vector<4xf32>
      %2570 = vector.extract %2568[1] : f32 from vector<4xf32>
      %2571 = vector.extract %2568[2] : f32 from vector<4xf32>
      %2572 = vector.extract %2568[3] : f32 from vector<4xf32>
      %2573 = vector.from_elements %2462, %2463, %2464, %2465 : vector<4xf32>
      %2574 = math.exp2 %2573 {tiled = true} : vector<4xf32>
      %2575 = vector.extract %2574[0] : f32 from vector<4xf32>
      %2576 = vector.extract %2574[1] : f32 from vector<4xf32>
      %2577 = vector.extract %2574[2] : f32 from vector<4xf32>
      %2578 = vector.extract %2574[3] : f32 from vector<4xf32>
      %2579 = vector.from_elements %2468, %2469, %2470, %2471 : vector<4xf32>
      %2580 = math.exp2 %2579 {tiled = true} : vector<4xf32>
      %2581 = vector.extract %2580[0] : f32 from vector<4xf32>
      %2582 = vector.extract %2580[1] : f32 from vector<4xf32>
      %2583 = vector.extract %2580[2] : f32 from vector<4xf32>
      %2584 = vector.extract %2580[3] : f32 from vector<4xf32>
      %2585 = vector.from_elements %2474, %2475, %2476, %2477 : vector<4xf32>
      %2586 = math.exp2 %2585 {tiled = true} : vector<4xf32>
      %2587 = vector.extract %2586[0] : f32 from vector<4xf32>
      %2588 = vector.extract %2586[1] : f32 from vector<4xf32>
      %2589 = vector.extract %2586[2] : f32 from vector<4xf32>
      %2590 = vector.extract %2586[3] : f32 from vector<4xf32>
      %2591 = vector.from_elements %2480, %2481, %2482, %2483 : vector<4xf32>
      %2592 = math.exp2 %2591 {tiled = true} : vector<4xf32>
      %2593 = vector.extract %2592[0] : f32 from vector<4xf32>
      %2594 = vector.extract %2592[1] : f32 from vector<4xf32>
      %2595 = vector.extract %2592[2] : f32 from vector<4xf32>
      %2596 = vector.extract %2592[3] : f32 from vector<4xf32>
      %2597 = vector.from_elements %2486, %2487, %2488, %2489 : vector<4xf32>
      %2598 = math.exp2 %2597 {tiled = true} : vector<4xf32>
      %2599 = vector.extract %2598[0] : f32 from vector<4xf32>
      %2600 = vector.extract %2598[1] : f32 from vector<4xf32>
      %2601 = vector.extract %2598[2] : f32 from vector<4xf32>
      %2602 = vector.extract %2598[3] : f32 from vector<4xf32>
      %2603 = vector.from_elements %2493, %2494, %2495, %2496 : vector<4xf32>
      %2604 = math.exp2 %2603 {tiled = true} : vector<4xf32>
      %2605 = vector.extract %2604[0] : f32 from vector<4xf32>
      %2606 = vector.extract %2604[1] : f32 from vector<4xf32>
      %2607 = vector.extract %2604[2] : f32 from vector<4xf32>
      %2608 = vector.extract %2604[3] : f32 from vector<4xf32>
      %2609 = vector.from_elements %2499, %2500, %2501, %2502 : vector<4xf32>
      %2610 = math.exp2 %2609 {tiled = true} : vector<4xf32>
      %2611 = vector.extract %2610[0] : f32 from vector<4xf32>
      %2612 = vector.extract %2610[1] : f32 from vector<4xf32>
      %2613 = vector.extract %2610[2] : f32 from vector<4xf32>
      %2614 = vector.extract %2610[3] : f32 from vector<4xf32>
      %2615 = vector.from_elements %2505, %2506, %2507, %2508 : vector<4xf32>
      %2616 = math.exp2 %2615 {tiled = true} : vector<4xf32>
      %2617 = vector.extract %2616[0] : f32 from vector<4xf32>
      %2618 = vector.extract %2616[1] : f32 from vector<4xf32>
      %2619 = vector.extract %2616[2] : f32 from vector<4xf32>
      %2620 = vector.extract %2616[3] : f32 from vector<4xf32>
      %2621 = vector.from_elements %2511, %2512, %2513, %2514 : vector<4xf32>
      %2622 = math.exp2 %2621 {tiled = true} : vector<4xf32>
      %2623 = vector.extract %2622[0] : f32 from vector<4xf32>
      %2624 = vector.extract %2622[1] : f32 from vector<4xf32>
      %2625 = vector.extract %2622[2] : f32 from vector<4xf32>
      %2626 = vector.extract %2622[3] : f32 from vector<4xf32>
      %2627 = vector.from_elements %2517, %2518, %2519, %2520 : vector<4xf32>
      %2628 = math.exp2 %2627 {tiled = true} : vector<4xf32>
      %2629 = vector.extract %2628[0] : f32 from vector<4xf32>
      %2630 = vector.extract %2628[1] : f32 from vector<4xf32>
      %2631 = vector.extract %2628[2] : f32 from vector<4xf32>
      %2632 = vector.extract %2628[3] : f32 from vector<4xf32>
      %2633 = vector.from_elements %2523, %2524, %2525, %2526 : vector<4xf32>
      %2634 = math.exp2 %2633 {tiled = true} : vector<4xf32>
      %2635 = vector.extract %2634[0] : f32 from vector<4xf32>
      %2636 = vector.extract %2634[1] : f32 from vector<4xf32>
      %2637 = vector.extract %2634[2] : f32 from vector<4xf32>
      %2638 = vector.extract %2634[3] : f32 from vector<4xf32>
      %2639 = vector.from_elements %2529, %2530, %2531, %2532 : vector<4xf32>
      %2640 = math.exp2 %2639 {tiled = true} : vector<4xf32>
      %2641 = vector.extract %2640[0] : f32 from vector<4xf32>
      %2642 = vector.extract %2640[1] : f32 from vector<4xf32>
      %2643 = vector.extract %2640[2] : f32 from vector<4xf32>
      %2644 = vector.extract %2640[3] : f32 from vector<4xf32>
      %2645 = vector.from_elements %2535, %2536, %2537, %2538 : vector<4xf32>
      %2646 = math.exp2 %2645 {tiled = true} : vector<4xf32>
      %2647 = vector.extract %2646[0] : f32 from vector<4xf32>
      %2648 = vector.extract %2646[1] : f32 from vector<4xf32>
      %2649 = vector.extract %2646[2] : f32 from vector<4xf32>
      %2650 = vector.extract %2646[3] : f32 from vector<4xf32>
      %2651 = arith.addf %2557, %cst_6 : f32
      %2652 = arith.addf %2651, %2558 : f32
      %2653 = arith.addf %2652, %2559 : f32
      %2654 = arith.addf %2653, %2560 : f32
      %2655 = arith.addf %2654, %2563 : f32
      %2656 = arith.addf %2655, %2564 : f32
      %2657 = arith.addf %2656, %2565 : f32
      %2658 = arith.addf %2657, %2566 : f32
      %2659 = arith.addf %2658, %2569 : f32
      %2660 = arith.addf %2659, %2570 : f32
      %2661 = arith.addf %2660, %2571 : f32
      %2662 = arith.addf %2661, %2572 : f32
      %2663 = arith.addf %2662, %2575 : f32
      %2664 = arith.addf %2663, %2576 : f32
      %2665 = arith.addf %2664, %2577 : f32
      %2666 = arith.addf %2665, %2578 : f32
      %2667 = arith.addf %2666, %2581 : f32
      %2668 = arith.addf %2667, %2582 : f32
      %2669 = arith.addf %2668, %2583 : f32
      %2670 = arith.addf %2669, %2584 : f32
      %2671 = arith.addf %2670, %2587 : f32
      %2672 = arith.addf %2671, %2588 : f32
      %2673 = arith.addf %2672, %2589 : f32
      %2674 = arith.addf %2673, %2590 : f32
      %2675 = arith.addf %2674, %2593 : f32
      %2676 = arith.addf %2675, %2594 : f32
      %2677 = arith.addf %2676, %2595 : f32
      %2678 = arith.addf %2677, %2596 : f32
      %2679 = arith.addf %2678, %2599 : f32
      %2680 = arith.addf %2679, %2600 : f32
      %2681 = arith.addf %2680, %2601 : f32
      %2682 = arith.addf %2681, %2602 : f32
      %2683 = llvm.bitcast %2682 : f32 to i32
      %2684 = rocdl.ds_bpermute %10, %2683 : (i32, i32) -> i32
      %2685 = llvm.bitcast %2684 : i32 to f32
      %2686 = arith.addf %2682, %2685 : f32
      %2687 = llvm.bitcast %2686 : f32 to i32
      %2688 = rocdl.ds_bpermute %14, %2687 : (i32, i32) -> i32
      %2689 = llvm.bitcast %2688 : i32 to f32
      %2690 = arith.addf %2686, %2689 : f32
      %2691 = arith.addf %2605, %cst_6 : f32
      %2692 = arith.addf %2691, %2606 : f32
      %2693 = arith.addf %2692, %2607 : f32
      %2694 = arith.addf %2693, %2608 : f32
      %2695 = arith.addf %2694, %2611 : f32
      %2696 = arith.addf %2695, %2612 : f32
      %2697 = arith.addf %2696, %2613 : f32
      %2698 = arith.addf %2697, %2614 : f32
      %2699 = arith.addf %2698, %2617 : f32
      %2700 = arith.addf %2699, %2618 : f32
      %2701 = arith.addf %2700, %2619 : f32
      %2702 = arith.addf %2701, %2620 : f32
      %2703 = arith.addf %2702, %2623 : f32
      %2704 = arith.addf %2703, %2624 : f32
      %2705 = arith.addf %2704, %2625 : f32
      %2706 = arith.addf %2705, %2626 : f32
      %2707 = arith.addf %2706, %2629 : f32
      %2708 = arith.addf %2707, %2630 : f32
      %2709 = arith.addf %2708, %2631 : f32
      %2710 = arith.addf %2709, %2632 : f32
      %2711 = arith.addf %2710, %2635 : f32
      %2712 = arith.addf %2711, %2636 : f32
      %2713 = arith.addf %2712, %2637 : f32
      %2714 = arith.addf %2713, %2638 : f32
      %2715 = arith.addf %2714, %2641 : f32
      %2716 = arith.addf %2715, %2642 : f32
      %2717 = arith.addf %2716, %2643 : f32
      %2718 = arith.addf %2717, %2644 : f32
      %2719 = arith.addf %2718, %2647 : f32
      %2720 = arith.addf %2719, %2648 : f32
      %2721 = arith.addf %2720, %2649 : f32
      %2722 = arith.addf %2721, %2650 : f32
      %2723 = llvm.bitcast %2722 : f32 to i32
      %2724 = rocdl.ds_bpermute %10, %2723 : (i32, i32) -> i32
      %2725 = llvm.bitcast %2724 : i32 to f32
      %2726 = arith.addf %2722, %2725 : f32
      %2727 = llvm.bitcast %2726 : f32 to i32
      %2728 = rocdl.ds_bpermute %14, %2727 : (i32, i32) -> i32
      %2729 = llvm.bitcast %2728 : i32 to f32
      %2730 = arith.addf %2726, %2729 : f32
      %2731 = vector.broadcast %2690 : f32 to vector<1xf32>
      %2732 = vector.broadcast %2546 : f32 to vector<1xf32>
      %2733 = arith.addf %2731, %2732 {origin = "frisk.add", tiled = true} : vector<1xf32>
      %2734 = vector.extract %2733[0] : f32 from vector<1xf32>
      %2735 = vector.broadcast %2730 : f32 to vector<1xf32>
      %2736 = vector.broadcast %2554 : f32 to vector<1xf32>
      %2737 = arith.addf %2735, %2736 {origin = "frisk.add", tiled = true} : vector<1xf32>
      %2738 = vector.extract %2737[0] : f32 from vector<1xf32>
      %2739 = vector.from_elements %2557, %2558, %2559, %2560 : vector<4xf32>
      %2740 = arith.truncf %2739 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2741 = vector.extract %2740[0] : f16 from vector<4xf16>
      %2742 = vector.extract %2740[1] : f16 from vector<4xf16>
      %2743 = vector.extract %2740[2] : f16 from vector<4xf16>
      %2744 = vector.extract %2740[3] : f16 from vector<4xf16>
      %2745 = vector.from_elements %2563, %2564, %2565, %2566 : vector<4xf32>
      %2746 = arith.truncf %2745 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2747 = vector.extract %2746[0] : f16 from vector<4xf16>
      %2748 = vector.extract %2746[1] : f16 from vector<4xf16>
      %2749 = vector.extract %2746[2] : f16 from vector<4xf16>
      %2750 = vector.extract %2746[3] : f16 from vector<4xf16>
      %2751 = vector.from_elements %2569, %2570, %2571, %2572 : vector<4xf32>
      %2752 = arith.truncf %2751 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2753 = vector.extract %2752[0] : f16 from vector<4xf16>
      %2754 = vector.extract %2752[1] : f16 from vector<4xf16>
      %2755 = vector.extract %2752[2] : f16 from vector<4xf16>
      %2756 = vector.extract %2752[3] : f16 from vector<4xf16>
      %2757 = vector.from_elements %2575, %2576, %2577, %2578 : vector<4xf32>
      %2758 = arith.truncf %2757 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2759 = vector.extract %2758[0] : f16 from vector<4xf16>
      %2760 = vector.extract %2758[1] : f16 from vector<4xf16>
      %2761 = vector.extract %2758[2] : f16 from vector<4xf16>
      %2762 = vector.extract %2758[3] : f16 from vector<4xf16>
      %2763 = vector.from_elements %2581, %2582, %2583, %2584 : vector<4xf32>
      %2764 = arith.truncf %2763 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2765 = vector.extract %2764[0] : f16 from vector<4xf16>
      %2766 = vector.extract %2764[1] : f16 from vector<4xf16>
      %2767 = vector.extract %2764[2] : f16 from vector<4xf16>
      %2768 = vector.extract %2764[3] : f16 from vector<4xf16>
      %2769 = vector.from_elements %2587, %2588, %2589, %2590 : vector<4xf32>
      %2770 = arith.truncf %2769 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2771 = vector.extract %2770[0] : f16 from vector<4xf16>
      %2772 = vector.extract %2770[1] : f16 from vector<4xf16>
      %2773 = vector.extract %2770[2] : f16 from vector<4xf16>
      %2774 = vector.extract %2770[3] : f16 from vector<4xf16>
      %2775 = vector.from_elements %2593, %2594, %2595, %2596 : vector<4xf32>
      %2776 = arith.truncf %2775 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2777 = vector.extract %2776[0] : f16 from vector<4xf16>
      %2778 = vector.extract %2776[1] : f16 from vector<4xf16>
      %2779 = vector.extract %2776[2] : f16 from vector<4xf16>
      %2780 = vector.extract %2776[3] : f16 from vector<4xf16>
      %2781 = vector.from_elements %2599, %2600, %2601, %2602 : vector<4xf32>
      %2782 = arith.truncf %2781 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2783 = vector.extract %2782[0] : f16 from vector<4xf16>
      %2784 = vector.extract %2782[1] : f16 from vector<4xf16>
      %2785 = vector.extract %2782[2] : f16 from vector<4xf16>
      %2786 = vector.extract %2782[3] : f16 from vector<4xf16>
      %2787 = vector.from_elements %2605, %2606, %2607, %2608 : vector<4xf32>
      %2788 = arith.truncf %2787 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2789 = vector.extract %2788[0] : f16 from vector<4xf16>
      %2790 = vector.extract %2788[1] : f16 from vector<4xf16>
      %2791 = vector.extract %2788[2] : f16 from vector<4xf16>
      %2792 = vector.extract %2788[3] : f16 from vector<4xf16>
      %2793 = vector.from_elements %2611, %2612, %2613, %2614 : vector<4xf32>
      %2794 = arith.truncf %2793 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2795 = vector.extract %2794[0] : f16 from vector<4xf16>
      %2796 = vector.extract %2794[1] : f16 from vector<4xf16>
      %2797 = vector.extract %2794[2] : f16 from vector<4xf16>
      %2798 = vector.extract %2794[3] : f16 from vector<4xf16>
      %2799 = vector.from_elements %2617, %2618, %2619, %2620 : vector<4xf32>
      %2800 = arith.truncf %2799 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2801 = vector.extract %2800[0] : f16 from vector<4xf16>
      %2802 = vector.extract %2800[1] : f16 from vector<4xf16>
      %2803 = vector.extract %2800[2] : f16 from vector<4xf16>
      %2804 = vector.extract %2800[3] : f16 from vector<4xf16>
      %2805 = vector.from_elements %2623, %2624, %2625, %2626 : vector<4xf32>
      %2806 = arith.truncf %2805 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2807 = vector.extract %2806[0] : f16 from vector<4xf16>
      %2808 = vector.extract %2806[1] : f16 from vector<4xf16>
      %2809 = vector.extract %2806[2] : f16 from vector<4xf16>
      %2810 = vector.extract %2806[3] : f16 from vector<4xf16>
      %2811 = vector.from_elements %2629, %2630, %2631, %2632 : vector<4xf32>
      %2812 = arith.truncf %2811 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2813 = vector.extract %2812[0] : f16 from vector<4xf16>
      %2814 = vector.extract %2812[1] : f16 from vector<4xf16>
      %2815 = vector.extract %2812[2] : f16 from vector<4xf16>
      %2816 = vector.extract %2812[3] : f16 from vector<4xf16>
      %2817 = vector.from_elements %2635, %2636, %2637, %2638 : vector<4xf32>
      %2818 = arith.truncf %2817 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2819 = vector.extract %2818[0] : f16 from vector<4xf16>
      %2820 = vector.extract %2818[1] : f16 from vector<4xf16>
      %2821 = vector.extract %2818[2] : f16 from vector<4xf16>
      %2822 = vector.extract %2818[3] : f16 from vector<4xf16>
      %2823 = vector.from_elements %2641, %2642, %2643, %2644 : vector<4xf32>
      %2824 = arith.truncf %2823 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2825 = vector.extract %2824[0] : f16 from vector<4xf16>
      %2826 = vector.extract %2824[1] : f16 from vector<4xf16>
      %2827 = vector.extract %2824[2] : f16 from vector<4xf16>
      %2828 = vector.extract %2824[3] : f16 from vector<4xf16>
      %2829 = vector.from_elements %2647, %2648, %2649, %2650 : vector<4xf32>
      %2830 = arith.truncf %2829 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2831 = vector.extract %2830[0] : f16 from vector<4xf16>
      %2832 = vector.extract %2830[1] : f16 from vector<4xf16>
      %2833 = vector.extract %2830[2] : f16 from vector<4xf16>
      %2834 = vector.extract %2830[3] : f16 from vector<4xf16>
      %2835 = vector.extract %arg5[0, 0] : f32 from vector<2x8xf32>
      %2836 = vector.extract %arg5[0, 1] : f32 from vector<2x8xf32>
      %2837 = vector.extract %arg5[0, 2] : f32 from vector<2x8xf32>
      %2838 = vector.extract %arg5[0, 3] : f32 from vector<2x8xf32>
      %2839 = vector.from_elements %2835, %2836, %2837, %2838 : vector<4xf32>
      %2840 = vector.broadcast %2542 : f32 to vector<4xf32>
      %2841 = arith.mulf %2839, %2840 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2842 = vector.extract %2841[0] : f32 from vector<4xf32>
      %2843 = vector.extract %2841[1] : f32 from vector<4xf32>
      %2844 = vector.extract %2841[2] : f32 from vector<4xf32>
      %2845 = vector.extract %2841[3] : f32 from vector<4xf32>
      %2846 = vector.extract %arg5[0, 4] : f32 from vector<2x8xf32>
      %2847 = vector.extract %arg5[0, 5] : f32 from vector<2x8xf32>
      %2848 = vector.extract %arg5[0, 6] : f32 from vector<2x8xf32>
      %2849 = vector.extract %arg5[0, 7] : f32 from vector<2x8xf32>
      %2850 = vector.from_elements %2846, %2847, %2848, %2849 : vector<4xf32>
      %2851 = arith.mulf %2850, %2840 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2852 = vector.extract %2851[0] : f32 from vector<4xf32>
      %2853 = vector.extract %2851[1] : f32 from vector<4xf32>
      %2854 = vector.extract %2851[2] : f32 from vector<4xf32>
      %2855 = vector.extract %2851[3] : f32 from vector<4xf32>
      %2856 = vector.extract %arg5[1, 0] : f32 from vector<2x8xf32>
      %2857 = vector.extract %arg5[1, 1] : f32 from vector<2x8xf32>
      %2858 = vector.extract %arg5[1, 2] : f32 from vector<2x8xf32>
      %2859 = vector.extract %arg5[1, 3] : f32 from vector<2x8xf32>
      %2860 = vector.from_elements %2856, %2857, %2858, %2859 : vector<4xf32>
      %2861 = vector.broadcast %2550 : f32 to vector<4xf32>
      %2862 = arith.mulf %2860, %2861 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2863 = vector.extract %2862[0] : f32 from vector<4xf32>
      %2864 = vector.extract %2862[1] : f32 from vector<4xf32>
      %2865 = vector.extract %2862[2] : f32 from vector<4xf32>
      %2866 = vector.extract %2862[3] : f32 from vector<4xf32>
      %2867 = vector.extract %arg5[1, 4] : f32 from vector<2x8xf32>
      %2868 = vector.extract %arg5[1, 5] : f32 from vector<2x8xf32>
      %2869 = vector.extract %arg5[1, 6] : f32 from vector<2x8xf32>
      %2870 = vector.extract %arg5[1, 7] : f32 from vector<2x8xf32>
      %2871 = vector.from_elements %2867, %2868, %2869, %2870 : vector<4xf32>
      %2872 = arith.mulf %2871, %2861 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2873 = vector.extract %2872[0] : f32 from vector<4xf32>
      %2874 = vector.extract %2872[1] : f32 from vector<4xf32>
      %2875 = vector.extract %2872[2] : f32 from vector<4xf32>
      %2876 = vector.extract %2872[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %2877 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>()[%thread_id_x]
      %2878 = frisk.read_matrix_from_shm %view[%2877] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2879 = vector.bitcast %2878 : vector<4xi32> to vector<8xf16>
      %2880 = vector.extract %2879[0] : f16 from vector<8xf16>
      %2881 = vector.extract %2879[%279] : f16 from vector<8xf16>
      %2882 = vector.extract %2879[1] : f16 from vector<8xf16>
      %2883 = vector.extract %2879[%280] : f16 from vector<8xf16>
      %2884 = vector.extract %2879[2] : f16 from vector<8xf16>
      %2885 = vector.extract %2879[%281] : f16 from vector<8xf16>
      %2886 = vector.extract %2879[3] : f16 from vector<8xf16>
      %2887 = vector.extract %2879[%282] : f16 from vector<8xf16>
      %2888 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 16) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 128)>()[%thread_id_x]
      %2889 = frisk.read_matrix_from_shm %view[%2888] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2890 = vector.bitcast %2889 : vector<4xi32> to vector<8xf16>
      %2891 = vector.extract %2890[0] : f16 from vector<8xf16>
      %2892 = vector.extract %2890[%279] : f16 from vector<8xf16>
      %2893 = vector.extract %2890[1] : f16 from vector<8xf16>
      %2894 = vector.extract %2890[%280] : f16 from vector<8xf16>
      %2895 = vector.extract %2890[2] : f16 from vector<8xf16>
      %2896 = vector.extract %2890[%281] : f16 from vector<8xf16>
      %2897 = vector.extract %2890[3] : f16 from vector<8xf16>
      %2898 = vector.extract %2890[%282] : f16 from vector<8xf16>
      %2899 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 32) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 256)>()[%thread_id_x]
      %2900 = frisk.read_matrix_from_shm %view[%2899] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2901 = vector.bitcast %2900 : vector<4xi32> to vector<8xf16>
      %2902 = vector.extract %2901[0] : f16 from vector<8xf16>
      %2903 = vector.extract %2901[%279] : f16 from vector<8xf16>
      %2904 = vector.extract %2901[1] : f16 from vector<8xf16>
      %2905 = vector.extract %2901[%280] : f16 from vector<8xf16>
      %2906 = vector.extract %2901[2] : f16 from vector<8xf16>
      %2907 = vector.extract %2901[%281] : f16 from vector<8xf16>
      %2908 = vector.extract %2901[3] : f16 from vector<8xf16>
      %2909 = vector.extract %2901[%282] : f16 from vector<8xf16>
      %2910 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 48) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 384)>()[%thread_id_x]
      %2911 = frisk.read_matrix_from_shm %view[%2910] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2912 = vector.bitcast %2911 : vector<4xi32> to vector<8xf16>
      %2913 = vector.extract %2912[0] : f16 from vector<8xf16>
      %2914 = vector.extract %2912[%279] : f16 from vector<8xf16>
      %2915 = vector.extract %2912[1] : f16 from vector<8xf16>
      %2916 = vector.extract %2912[%280] : f16 from vector<8xf16>
      %2917 = vector.extract %2912[2] : f16 from vector<8xf16>
      %2918 = vector.extract %2912[%281] : f16 from vector<8xf16>
      %2919 = vector.extract %2912[3] : f16 from vector<8xf16>
      %2920 = vector.extract %2912[%282] : f16 from vector<8xf16>
      %2921 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 2048)>()[%thread_id_x]
      %2922 = frisk.read_matrix_from_shm %view[%2921] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2923 = vector.bitcast %2922 : vector<4xi32> to vector<8xf16>
      %2924 = vector.extract %2923[0] : f16 from vector<8xf16>
      %2925 = vector.extract %2923[%279] : f16 from vector<8xf16>
      %2926 = vector.extract %2923[1] : f16 from vector<8xf16>
      %2927 = vector.extract %2923[%280] : f16 from vector<8xf16>
      %2928 = vector.extract %2923[2] : f16 from vector<8xf16>
      %2929 = vector.extract %2923[%281] : f16 from vector<8xf16>
      %2930 = vector.extract %2923[3] : f16 from vector<8xf16>
      %2931 = vector.extract %2923[%282] : f16 from vector<8xf16>
      %2932 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 80) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 128)>()[%thread_id_x]
      %2933 = frisk.read_matrix_from_shm %view[%2932] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2934 = vector.bitcast %2933 : vector<4xi32> to vector<8xf16>
      %2935 = vector.extract %2934[0] : f16 from vector<8xf16>
      %2936 = vector.extract %2934[%279] : f16 from vector<8xf16>
      %2937 = vector.extract %2934[1] : f16 from vector<8xf16>
      %2938 = vector.extract %2934[%280] : f16 from vector<8xf16>
      %2939 = vector.extract %2934[2] : f16 from vector<8xf16>
      %2940 = vector.extract %2934[%281] : f16 from vector<8xf16>
      %2941 = vector.extract %2934[3] : f16 from vector<8xf16>
      %2942 = vector.extract %2934[%282] : f16 from vector<8xf16>
      %2943 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 96) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 256)>()[%thread_id_x]
      %2944 = frisk.read_matrix_from_shm %view[%2943] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2945 = vector.bitcast %2944 : vector<4xi32> to vector<8xf16>
      %2946 = vector.extract %2945[0] : f16 from vector<8xf16>
      %2947 = vector.extract %2945[%279] : f16 from vector<8xf16>
      %2948 = vector.extract %2945[1] : f16 from vector<8xf16>
      %2949 = vector.extract %2945[%280] : f16 from vector<8xf16>
      %2950 = vector.extract %2945[2] : f16 from vector<8xf16>
      %2951 = vector.extract %2945[%281] : f16 from vector<8xf16>
      %2952 = vector.extract %2945[3] : f16 from vector<8xf16>
      %2953 = vector.extract %2945[%282] : f16 from vector<8xf16>
      %2954 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 112) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 384)>()[%thread_id_x]
      %2955 = frisk.read_matrix_from_shm %view[%2954] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2956 = vector.bitcast %2955 : vector<4xi32> to vector<8xf16>
      %2957 = vector.extract %2956[0] : f16 from vector<8xf16>
      %2958 = vector.extract %2956[%279] : f16 from vector<8xf16>
      %2959 = vector.extract %2956[1] : f16 from vector<8xf16>
      %2960 = vector.extract %2956[%280] : f16 from vector<8xf16>
      %2961 = vector.extract %2956[2] : f16 from vector<8xf16>
      %2962 = vector.extract %2956[%281] : f16 from vector<8xf16>
      %2963 = vector.extract %2956[3] : f16 from vector<8xf16>
      %2964 = vector.extract %2956[%282] : f16 from vector<8xf16>
      %2965 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 32 + s0) mod 32)>(%arg4)[%25]
      %subview_16 = memref.subview %arg2[0, %2965, %859, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %2966 = builtin.unrealized_conversion_cast %subview_16 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      gpu.barrier
      %2967 = llvm.extractvalue %2966[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %2968 = llvm.extractvalue %2966[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %2969 = llvm.getelementptr %2967[%2968] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %2970 = llvm.getelementptr %2969[%1836] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %2970, %49, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %2971 = llvm.getelementptr %2969[%1843] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %2971, %66, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %2972 = vector.from_elements %2741, %2742, %2743, %2744 : vector<4xf16>
      %2973 = vector.from_elements %2880, %2882, %2884, %2886 : vector<4xf16>
      %2974 = vector.from_elements %2842, %2843, %2844, %2845 : vector<4xf32>
      %2975 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2972, %2973, %2974 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2976 = vector.from_elements %2747, %2748, %2749, %2750 : vector<4xf16>
      %2977 = vector.from_elements %2891, %2893, %2895, %2897 : vector<4xf16>
      %2978 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2976, %2977, %2975 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2979 = vector.from_elements %2753, %2754, %2755, %2756 : vector<4xf16>
      %2980 = vector.from_elements %2902, %2904, %2906, %2908 : vector<4xf16>
      %2981 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2979, %2980, %2978 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2982 = vector.from_elements %2759, %2760, %2761, %2762 : vector<4xf16>
      %2983 = vector.from_elements %2913, %2915, %2917, %2919 : vector<4xf16>
      %2984 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2982, %2983, %2981 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2985 = vector.from_elements %2765, %2766, %2767, %2768 : vector<4xf16>
      %2986 = vector.from_elements %2924, %2926, %2928, %2930 : vector<4xf16>
      %2987 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2985, %2986, %2984 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2988 = vector.from_elements %2771, %2772, %2773, %2774 : vector<4xf16>
      %2989 = vector.from_elements %2935, %2937, %2939, %2941 : vector<4xf16>
      %2990 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2988, %2989, %2987 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2991 = vector.from_elements %2777, %2778, %2779, %2780 : vector<4xf16>
      %2992 = vector.from_elements %2946, %2948, %2950, %2952 : vector<4xf16>
      %2993 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2991, %2992, %2990 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2994 = vector.from_elements %2783, %2784, %2785, %2786 : vector<4xf16>
      %2995 = vector.from_elements %2957, %2959, %2961, %2963 : vector<4xf16>
      %2996 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2994, %2995, %2993 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2997 = vector.extract %2996[0] : f32 from vector<4xf32>
      %2998 = vector.extract %2996[1] : f32 from vector<4xf32>
      %2999 = vector.extract %2996[2] : f32 from vector<4xf32>
      %3000 = vector.extract %2996[3] : f32 from vector<4xf32>
      %3001 = vector.from_elements %2881, %2883, %2885, %2887 : vector<4xf16>
      %3002 = vector.from_elements %2852, %2853, %2854, %2855 : vector<4xf32>
      %3003 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2972, %3001, %3002 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3004 = vector.from_elements %2892, %2894, %2896, %2898 : vector<4xf16>
      %3005 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2976, %3004, %3003 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3006 = vector.from_elements %2903, %2905, %2907, %2909 : vector<4xf16>
      %3007 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2979, %3006, %3005 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3008 = vector.from_elements %2914, %2916, %2918, %2920 : vector<4xf16>
      %3009 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2982, %3008, %3007 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3010 = vector.from_elements %2925, %2927, %2929, %2931 : vector<4xf16>
      %3011 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2985, %3010, %3009 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3012 = vector.from_elements %2936, %2938, %2940, %2942 : vector<4xf16>
      %3013 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2988, %3012, %3011 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3014 = vector.from_elements %2947, %2949, %2951, %2953 : vector<4xf16>
      %3015 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2991, %3014, %3013 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3016 = vector.from_elements %2958, %2960, %2962, %2964 : vector<4xf16>
      %3017 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2994, %3016, %3015 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3018 = vector.extract %3017[0] : f32 from vector<4xf32>
      %3019 = vector.extract %3017[1] : f32 from vector<4xf32>
      %3020 = vector.extract %3017[2] : f32 from vector<4xf32>
      %3021 = vector.extract %3017[3] : f32 from vector<4xf32>
      %3022 = vector.from_elements %2789, %2790, %2791, %2792 : vector<4xf16>
      %3023 = vector.from_elements %2863, %2864, %2865, %2866 : vector<4xf32>
      %3024 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3022, %2973, %3023 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3025 = vector.from_elements %2795, %2796, %2797, %2798 : vector<4xf16>
      %3026 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3025, %2977, %3024 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3027 = vector.from_elements %2801, %2802, %2803, %2804 : vector<4xf16>
      %3028 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3027, %2980, %3026 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3029 = vector.from_elements %2807, %2808, %2809, %2810 : vector<4xf16>
      %3030 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3029, %2983, %3028 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3031 = vector.from_elements %2813, %2814, %2815, %2816 : vector<4xf16>
      %3032 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3031, %2986, %3030 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3033 = vector.from_elements %2819, %2820, %2821, %2822 : vector<4xf16>
      %3034 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3033, %2989, %3032 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3035 = vector.from_elements %2825, %2826, %2827, %2828 : vector<4xf16>
      %3036 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3035, %2992, %3034 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3037 = vector.from_elements %2831, %2832, %2833, %2834 : vector<4xf16>
      %3038 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3037, %2995, %3036 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3039 = vector.extract %3038[0] : f32 from vector<4xf32>
      %3040 = vector.extract %3038[1] : f32 from vector<4xf32>
      %3041 = vector.extract %3038[2] : f32 from vector<4xf32>
      %3042 = vector.extract %3038[3] : f32 from vector<4xf32>
      %3043 = vector.from_elements %2873, %2874, %2875, %2876 : vector<4xf32>
      %3044 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3022, %3001, %3043 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3045 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3025, %3004, %3044 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3046 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3027, %3006, %3045 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3047 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3029, %3008, %3046 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3048 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3031, %3010, %3047 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3049 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3033, %3012, %3048 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3050 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3035, %3014, %3049 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3051 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3037, %3016, %3050 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3052 = vector.extract %3051[0] : f32 from vector<4xf32>
      %3053 = vector.extract %3051[1] : f32 from vector<4xf32>
      %3054 = vector.extract %3051[2] : f32 from vector<4xf32>
      %3055 = vector.extract %3051[3] : f32 from vector<4xf32>
      %3056 = vector.extract %arg8[0, 0] : f32 from vector<2x8xf32>
      %3057 = vector.extract %arg8[0, 1] : f32 from vector<2x8xf32>
      %3058 = vector.extract %arg8[0, 2] : f32 from vector<2x8xf32>
      %3059 = vector.extract %arg8[0, 3] : f32 from vector<2x8xf32>
      %3060 = vector.from_elements %3056, %3057, %3058, %3059 : vector<4xf32>
      %3061 = arith.mulf %3060, %2840 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3062 = vector.extract %3061[0] : f32 from vector<4xf32>
      %3063 = vector.extract %3061[1] : f32 from vector<4xf32>
      %3064 = vector.extract %3061[2] : f32 from vector<4xf32>
      %3065 = vector.extract %3061[3] : f32 from vector<4xf32>
      %3066 = vector.extract %arg8[0, 4] : f32 from vector<2x8xf32>
      %3067 = vector.extract %arg8[0, 5] : f32 from vector<2x8xf32>
      %3068 = vector.extract %arg8[0, 6] : f32 from vector<2x8xf32>
      %3069 = vector.extract %arg8[0, 7] : f32 from vector<2x8xf32>
      %3070 = vector.from_elements %3066, %3067, %3068, %3069 : vector<4xf32>
      %3071 = arith.mulf %3070, %2840 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3072 = vector.extract %3071[0] : f32 from vector<4xf32>
      %3073 = vector.extract %3071[1] : f32 from vector<4xf32>
      %3074 = vector.extract %3071[2] : f32 from vector<4xf32>
      %3075 = vector.extract %3071[3] : f32 from vector<4xf32>
      %3076 = vector.extract %arg8[1, 0] : f32 from vector<2x8xf32>
      %3077 = vector.extract %arg8[1, 1] : f32 from vector<2x8xf32>
      %3078 = vector.extract %arg8[1, 2] : f32 from vector<2x8xf32>
      %3079 = vector.extract %arg8[1, 3] : f32 from vector<2x8xf32>
      %3080 = vector.from_elements %3076, %3077, %3078, %3079 : vector<4xf32>
      %3081 = arith.mulf %3080, %2861 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3082 = vector.extract %3081[0] : f32 from vector<4xf32>
      %3083 = vector.extract %3081[1] : f32 from vector<4xf32>
      %3084 = vector.extract %3081[2] : f32 from vector<4xf32>
      %3085 = vector.extract %3081[3] : f32 from vector<4xf32>
      %3086 = vector.extract %arg8[1, 4] : f32 from vector<2x8xf32>
      %3087 = vector.extract %arg8[1, 5] : f32 from vector<2x8xf32>
      %3088 = vector.extract %arg8[1, 6] : f32 from vector<2x8xf32>
      %3089 = vector.extract %arg8[1, 7] : f32 from vector<2x8xf32>
      %3090 = vector.from_elements %3086, %3087, %3088, %3089 : vector<4xf32>
      %3091 = arith.mulf %3090, %2861 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3092 = vector.extract %3091[0] : f32 from vector<4xf32>
      %3093 = vector.extract %3091[1] : f32 from vector<4xf32>
      %3094 = vector.extract %3091[2] : f32 from vector<4xf32>
      %3095 = vector.extract %3091[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %3096 = frisk.read_matrix_from_shm %view[%2877] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3097 = vector.bitcast %3096 : vector<4xi32> to vector<8xf16>
      %3098 = vector.extract %3097[0] : f16 from vector<8xf16>
      %3099 = vector.extract %3097[%279] : f16 from vector<8xf16>
      %3100 = vector.extract %3097[1] : f16 from vector<8xf16>
      %3101 = vector.extract %3097[%280] : f16 from vector<8xf16>
      %3102 = vector.extract %3097[2] : f16 from vector<8xf16>
      %3103 = vector.extract %3097[%281] : f16 from vector<8xf16>
      %3104 = vector.extract %3097[3] : f16 from vector<8xf16>
      %3105 = vector.extract %3097[%282] : f16 from vector<8xf16>
      %3106 = frisk.read_matrix_from_shm %view[%2888] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3107 = vector.bitcast %3106 : vector<4xi32> to vector<8xf16>
      %3108 = vector.extract %3107[0] : f16 from vector<8xf16>
      %3109 = vector.extract %3107[%279] : f16 from vector<8xf16>
      %3110 = vector.extract %3107[1] : f16 from vector<8xf16>
      %3111 = vector.extract %3107[%280] : f16 from vector<8xf16>
      %3112 = vector.extract %3107[2] : f16 from vector<8xf16>
      %3113 = vector.extract %3107[%281] : f16 from vector<8xf16>
      %3114 = vector.extract %3107[3] : f16 from vector<8xf16>
      %3115 = vector.extract %3107[%282] : f16 from vector<8xf16>
      %3116 = frisk.read_matrix_from_shm %view[%2899] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3117 = vector.bitcast %3116 : vector<4xi32> to vector<8xf16>
      %3118 = vector.extract %3117[0] : f16 from vector<8xf16>
      %3119 = vector.extract %3117[%279] : f16 from vector<8xf16>
      %3120 = vector.extract %3117[1] : f16 from vector<8xf16>
      %3121 = vector.extract %3117[%280] : f16 from vector<8xf16>
      %3122 = vector.extract %3117[2] : f16 from vector<8xf16>
      %3123 = vector.extract %3117[%281] : f16 from vector<8xf16>
      %3124 = vector.extract %3117[3] : f16 from vector<8xf16>
      %3125 = vector.extract %3117[%282] : f16 from vector<8xf16>
      %3126 = frisk.read_matrix_from_shm %view[%2910] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3127 = vector.bitcast %3126 : vector<4xi32> to vector<8xf16>
      %3128 = vector.extract %3127[0] : f16 from vector<8xf16>
      %3129 = vector.extract %3127[%279] : f16 from vector<8xf16>
      %3130 = vector.extract %3127[1] : f16 from vector<8xf16>
      %3131 = vector.extract %3127[%280] : f16 from vector<8xf16>
      %3132 = vector.extract %3127[2] : f16 from vector<8xf16>
      %3133 = vector.extract %3127[%281] : f16 from vector<8xf16>
      %3134 = vector.extract %3127[3] : f16 from vector<8xf16>
      %3135 = vector.extract %3127[%282] : f16 from vector<8xf16>
      %3136 = frisk.read_matrix_from_shm %view[%2921] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3137 = vector.bitcast %3136 : vector<4xi32> to vector<8xf16>
      %3138 = vector.extract %3137[0] : f16 from vector<8xf16>
      %3139 = vector.extract %3137[%279] : f16 from vector<8xf16>
      %3140 = vector.extract %3137[1] : f16 from vector<8xf16>
      %3141 = vector.extract %3137[%280] : f16 from vector<8xf16>
      %3142 = vector.extract %3137[2] : f16 from vector<8xf16>
      %3143 = vector.extract %3137[%281] : f16 from vector<8xf16>
      %3144 = vector.extract %3137[3] : f16 from vector<8xf16>
      %3145 = vector.extract %3137[%282] : f16 from vector<8xf16>
      %3146 = frisk.read_matrix_from_shm %view[%2932] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3147 = vector.bitcast %3146 : vector<4xi32> to vector<8xf16>
      %3148 = vector.extract %3147[0] : f16 from vector<8xf16>
      %3149 = vector.extract %3147[%279] : f16 from vector<8xf16>
      %3150 = vector.extract %3147[1] : f16 from vector<8xf16>
      %3151 = vector.extract %3147[%280] : f16 from vector<8xf16>
      %3152 = vector.extract %3147[2] : f16 from vector<8xf16>
      %3153 = vector.extract %3147[%281] : f16 from vector<8xf16>
      %3154 = vector.extract %3147[3] : f16 from vector<8xf16>
      %3155 = vector.extract %3147[%282] : f16 from vector<8xf16>
      %3156 = frisk.read_matrix_from_shm %view[%2943] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3157 = vector.bitcast %3156 : vector<4xi32> to vector<8xf16>
      %3158 = vector.extract %3157[0] : f16 from vector<8xf16>
      %3159 = vector.extract %3157[%279] : f16 from vector<8xf16>
      %3160 = vector.extract %3157[1] : f16 from vector<8xf16>
      %3161 = vector.extract %3157[%280] : f16 from vector<8xf16>
      %3162 = vector.extract %3157[2] : f16 from vector<8xf16>
      %3163 = vector.extract %3157[%281] : f16 from vector<8xf16>
      %3164 = vector.extract %3157[3] : f16 from vector<8xf16>
      %3165 = vector.extract %3157[%282] : f16 from vector<8xf16>
      %3166 = frisk.read_matrix_from_shm %view[%2954] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3167 = vector.bitcast %3166 : vector<4xi32> to vector<8xf16>
      %3168 = vector.extract %3167[0] : f16 from vector<8xf16>
      %3169 = vector.extract %3167[%279] : f16 from vector<8xf16>
      %3170 = vector.extract %3167[1] : f16 from vector<8xf16>
      %3171 = vector.extract %3167[%280] : f16 from vector<8xf16>
      %3172 = vector.extract %3167[2] : f16 from vector<8xf16>
      %3173 = vector.extract %3167[%281] : f16 from vector<8xf16>
      %3174 = vector.extract %3167[3] : f16 from vector<8xf16>
      %3175 = vector.extract %3167[%282] : f16 from vector<8xf16>
      %subview_17 = memref.subview %arg2[0, %2965, %859, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %3176 = builtin.unrealized_conversion_cast %subview_17 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      gpu.barrier
      %3177 = llvm.extractvalue %3176[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %3178 = llvm.extractvalue %3176[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %3179 = llvm.getelementptr %3177[%3178] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %3180 = llvm.getelementptr %3179[%1836] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %3180, %49, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %3181 = llvm.getelementptr %3179[%1843] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %3181, %66, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %3182 = vector.from_elements %3098, %3100, %3102, %3104 : vector<4xf16>
      %3183 = vector.from_elements %3062, %3063, %3064, %3065 : vector<4xf32>
      %3184 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2972, %3182, %3183 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3185 = vector.from_elements %3108, %3110, %3112, %3114 : vector<4xf16>
      %3186 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2976, %3185, %3184 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3187 = vector.from_elements %3118, %3120, %3122, %3124 : vector<4xf16>
      %3188 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2979, %3187, %3186 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3189 = vector.from_elements %3128, %3130, %3132, %3134 : vector<4xf16>
      %3190 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2982, %3189, %3188 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3191 = vector.from_elements %3138, %3140, %3142, %3144 : vector<4xf16>
      %3192 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2985, %3191, %3190 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3193 = vector.from_elements %3148, %3150, %3152, %3154 : vector<4xf16>
      %3194 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2988, %3193, %3192 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3195 = vector.from_elements %3158, %3160, %3162, %3164 : vector<4xf16>
      %3196 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2991, %3195, %3194 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3197 = vector.from_elements %3168, %3170, %3172, %3174 : vector<4xf16>
      %3198 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2994, %3197, %3196 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3199 = vector.extract %3198[0] : f32 from vector<4xf32>
      %3200 = vector.extract %3198[1] : f32 from vector<4xf32>
      %3201 = vector.extract %3198[2] : f32 from vector<4xf32>
      %3202 = vector.extract %3198[3] : f32 from vector<4xf32>
      %3203 = vector.from_elements %3099, %3101, %3103, %3105 : vector<4xf16>
      %3204 = vector.from_elements %3072, %3073, %3074, %3075 : vector<4xf32>
      %3205 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2972, %3203, %3204 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3206 = vector.from_elements %3109, %3111, %3113, %3115 : vector<4xf16>
      %3207 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2976, %3206, %3205 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3208 = vector.from_elements %3119, %3121, %3123, %3125 : vector<4xf16>
      %3209 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2979, %3208, %3207 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3210 = vector.from_elements %3129, %3131, %3133, %3135 : vector<4xf16>
      %3211 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2982, %3210, %3209 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3212 = vector.from_elements %3139, %3141, %3143, %3145 : vector<4xf16>
      %3213 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2985, %3212, %3211 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3214 = vector.from_elements %3149, %3151, %3153, %3155 : vector<4xf16>
      %3215 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2988, %3214, %3213 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3216 = vector.from_elements %3159, %3161, %3163, %3165 : vector<4xf16>
      %3217 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2991, %3216, %3215 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3218 = vector.from_elements %3169, %3171, %3173, %3175 : vector<4xf16>
      %3219 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2994, %3218, %3217 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3220 = vector.extract %3219[0] : f32 from vector<4xf32>
      %3221 = vector.extract %3219[1] : f32 from vector<4xf32>
      %3222 = vector.extract %3219[2] : f32 from vector<4xf32>
      %3223 = vector.extract %3219[3] : f32 from vector<4xf32>
      %3224 = vector.from_elements %3082, %3083, %3084, %3085 : vector<4xf32>
      %3225 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3022, %3182, %3224 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3226 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3025, %3185, %3225 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3227 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3027, %3187, %3226 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3228 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3029, %3189, %3227 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3229 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3031, %3191, %3228 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3230 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3033, %3193, %3229 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3231 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3035, %3195, %3230 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3232 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3037, %3197, %3231 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3233 = vector.extract %3232[0] : f32 from vector<4xf32>
      %3234 = vector.extract %3232[1] : f32 from vector<4xf32>
      %3235 = vector.extract %3232[2] : f32 from vector<4xf32>
      %3236 = vector.extract %3232[3] : f32 from vector<4xf32>
      %3237 = vector.from_elements %3092, %3093, %3094, %3095 : vector<4xf32>
      %3238 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3022, %3203, %3237 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3239 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3025, %3206, %3238 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3240 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3027, %3208, %3239 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3241 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3029, %3210, %3240 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3242 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3031, %3212, %3241 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3243 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3033, %3214, %3242 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3244 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3035, %3216, %3243 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3245 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3037, %3218, %3244 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3246 = vector.extract %3245[0] : f32 from vector<4xf32>
      %3247 = vector.extract %3245[1] : f32 from vector<4xf32>
      %3248 = vector.extract %3245[2] : f32 from vector<4xf32>
      %3249 = vector.extract %3245[3] : f32 from vector<4xf32>
      %3250 = vector.extract %arg9[0, 0] : f32 from vector<2x8xf32>
      %3251 = vector.extract %arg9[0, 1] : f32 from vector<2x8xf32>
      %3252 = vector.extract %arg9[0, 2] : f32 from vector<2x8xf32>
      %3253 = vector.extract %arg9[0, 3] : f32 from vector<2x8xf32>
      %3254 = vector.from_elements %3250, %3251, %3252, %3253 : vector<4xf32>
      %3255 = arith.mulf %3254, %2840 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3256 = vector.extract %3255[0] : f32 from vector<4xf32>
      %3257 = vector.extract %3255[1] : f32 from vector<4xf32>
      %3258 = vector.extract %3255[2] : f32 from vector<4xf32>
      %3259 = vector.extract %3255[3] : f32 from vector<4xf32>
      %3260 = vector.extract %arg9[0, 4] : f32 from vector<2x8xf32>
      %3261 = vector.extract %arg9[0, 5] : f32 from vector<2x8xf32>
      %3262 = vector.extract %arg9[0, 6] : f32 from vector<2x8xf32>
      %3263 = vector.extract %arg9[0, 7] : f32 from vector<2x8xf32>
      %3264 = vector.from_elements %3260, %3261, %3262, %3263 : vector<4xf32>
      %3265 = arith.mulf %3264, %2840 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3266 = vector.extract %3265[0] : f32 from vector<4xf32>
      %3267 = vector.extract %3265[1] : f32 from vector<4xf32>
      %3268 = vector.extract %3265[2] : f32 from vector<4xf32>
      %3269 = vector.extract %3265[3] : f32 from vector<4xf32>
      %3270 = vector.extract %arg9[1, 0] : f32 from vector<2x8xf32>
      %3271 = vector.extract %arg9[1, 1] : f32 from vector<2x8xf32>
      %3272 = vector.extract %arg9[1, 2] : f32 from vector<2x8xf32>
      %3273 = vector.extract %arg9[1, 3] : f32 from vector<2x8xf32>
      %3274 = vector.from_elements %3270, %3271, %3272, %3273 : vector<4xf32>
      %3275 = arith.mulf %3274, %2861 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3276 = vector.extract %3275[0] : f32 from vector<4xf32>
      %3277 = vector.extract %3275[1] : f32 from vector<4xf32>
      %3278 = vector.extract %3275[2] : f32 from vector<4xf32>
      %3279 = vector.extract %3275[3] : f32 from vector<4xf32>
      %3280 = vector.extract %arg9[1, 4] : f32 from vector<2x8xf32>
      %3281 = vector.extract %arg9[1, 5] : f32 from vector<2x8xf32>
      %3282 = vector.extract %arg9[1, 6] : f32 from vector<2x8xf32>
      %3283 = vector.extract %arg9[1, 7] : f32 from vector<2x8xf32>
      %3284 = vector.from_elements %3280, %3281, %3282, %3283 : vector<4xf32>
      %3285 = arith.mulf %3284, %2861 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3286 = vector.extract %3285[0] : f32 from vector<4xf32>
      %3287 = vector.extract %3285[1] : f32 from vector<4xf32>
      %3288 = vector.extract %3285[2] : f32 from vector<4xf32>
      %3289 = vector.extract %3285[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %3290 = frisk.read_matrix_from_shm %view[%2877] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3291 = vector.bitcast %3290 : vector<4xi32> to vector<8xf16>
      %3292 = vector.extract %3291[0] : f16 from vector<8xf16>
      %3293 = vector.extract %3291[%279] : f16 from vector<8xf16>
      %3294 = vector.extract %3291[1] : f16 from vector<8xf16>
      %3295 = vector.extract %3291[%280] : f16 from vector<8xf16>
      %3296 = vector.extract %3291[2] : f16 from vector<8xf16>
      %3297 = vector.extract %3291[%281] : f16 from vector<8xf16>
      %3298 = vector.extract %3291[3] : f16 from vector<8xf16>
      %3299 = vector.extract %3291[%282] : f16 from vector<8xf16>
      %3300 = frisk.read_matrix_from_shm %view[%2888] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3301 = vector.bitcast %3300 : vector<4xi32> to vector<8xf16>
      %3302 = vector.extract %3301[0] : f16 from vector<8xf16>
      %3303 = vector.extract %3301[%279] : f16 from vector<8xf16>
      %3304 = vector.extract %3301[1] : f16 from vector<8xf16>
      %3305 = vector.extract %3301[%280] : f16 from vector<8xf16>
      %3306 = vector.extract %3301[2] : f16 from vector<8xf16>
      %3307 = vector.extract %3301[%281] : f16 from vector<8xf16>
      %3308 = vector.extract %3301[3] : f16 from vector<8xf16>
      %3309 = vector.extract %3301[%282] : f16 from vector<8xf16>
      %3310 = frisk.read_matrix_from_shm %view[%2899] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3311 = vector.bitcast %3310 : vector<4xi32> to vector<8xf16>
      %3312 = vector.extract %3311[0] : f16 from vector<8xf16>
      %3313 = vector.extract %3311[%279] : f16 from vector<8xf16>
      %3314 = vector.extract %3311[1] : f16 from vector<8xf16>
      %3315 = vector.extract %3311[%280] : f16 from vector<8xf16>
      %3316 = vector.extract %3311[2] : f16 from vector<8xf16>
      %3317 = vector.extract %3311[%281] : f16 from vector<8xf16>
      %3318 = vector.extract %3311[3] : f16 from vector<8xf16>
      %3319 = vector.extract %3311[%282] : f16 from vector<8xf16>
      %3320 = frisk.read_matrix_from_shm %view[%2910] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3321 = vector.bitcast %3320 : vector<4xi32> to vector<8xf16>
      %3322 = vector.extract %3321[0] : f16 from vector<8xf16>
      %3323 = vector.extract %3321[%279] : f16 from vector<8xf16>
      %3324 = vector.extract %3321[1] : f16 from vector<8xf16>
      %3325 = vector.extract %3321[%280] : f16 from vector<8xf16>
      %3326 = vector.extract %3321[2] : f16 from vector<8xf16>
      %3327 = vector.extract %3321[%281] : f16 from vector<8xf16>
      %3328 = vector.extract %3321[3] : f16 from vector<8xf16>
      %3329 = vector.extract %3321[%282] : f16 from vector<8xf16>
      %3330 = frisk.read_matrix_from_shm %view[%2921] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3331 = vector.bitcast %3330 : vector<4xi32> to vector<8xf16>
      %3332 = vector.extract %3331[0] : f16 from vector<8xf16>
      %3333 = vector.extract %3331[%279] : f16 from vector<8xf16>
      %3334 = vector.extract %3331[1] : f16 from vector<8xf16>
      %3335 = vector.extract %3331[%280] : f16 from vector<8xf16>
      %3336 = vector.extract %3331[2] : f16 from vector<8xf16>
      %3337 = vector.extract %3331[%281] : f16 from vector<8xf16>
      %3338 = vector.extract %3331[3] : f16 from vector<8xf16>
      %3339 = vector.extract %3331[%282] : f16 from vector<8xf16>
      %3340 = frisk.read_matrix_from_shm %view[%2932] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3341 = vector.bitcast %3340 : vector<4xi32> to vector<8xf16>
      %3342 = vector.extract %3341[0] : f16 from vector<8xf16>
      %3343 = vector.extract %3341[%279] : f16 from vector<8xf16>
      %3344 = vector.extract %3341[1] : f16 from vector<8xf16>
      %3345 = vector.extract %3341[%280] : f16 from vector<8xf16>
      %3346 = vector.extract %3341[2] : f16 from vector<8xf16>
      %3347 = vector.extract %3341[%281] : f16 from vector<8xf16>
      %3348 = vector.extract %3341[3] : f16 from vector<8xf16>
      %3349 = vector.extract %3341[%282] : f16 from vector<8xf16>
      %3350 = frisk.read_matrix_from_shm %view[%2943] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3351 = vector.bitcast %3350 : vector<4xi32> to vector<8xf16>
      %3352 = vector.extract %3351[0] : f16 from vector<8xf16>
      %3353 = vector.extract %3351[%279] : f16 from vector<8xf16>
      %3354 = vector.extract %3351[1] : f16 from vector<8xf16>
      %3355 = vector.extract %3351[%280] : f16 from vector<8xf16>
      %3356 = vector.extract %3351[2] : f16 from vector<8xf16>
      %3357 = vector.extract %3351[%281] : f16 from vector<8xf16>
      %3358 = vector.extract %3351[3] : f16 from vector<8xf16>
      %3359 = vector.extract %3351[%282] : f16 from vector<8xf16>
      %3360 = frisk.read_matrix_from_shm %view[%2954] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3361 = vector.bitcast %3360 : vector<4xi32> to vector<8xf16>
      %3362 = vector.extract %3361[0] : f16 from vector<8xf16>
      %3363 = vector.extract %3361[%279] : f16 from vector<8xf16>
      %3364 = vector.extract %3361[1] : f16 from vector<8xf16>
      %3365 = vector.extract %3361[%280] : f16 from vector<8xf16>
      %3366 = vector.extract %3361[2] : f16 from vector<8xf16>
      %3367 = vector.extract %3361[%281] : f16 from vector<8xf16>
      %3368 = vector.extract %3361[3] : f16 from vector<8xf16>
      %3369 = vector.extract %3361[%282] : f16 from vector<8xf16>
      %subview_18 = memref.subview %arg2[0, %2965, %859, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %3370 = builtin.unrealized_conversion_cast %subview_18 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      gpu.barrier
      %3371 = llvm.extractvalue %3370[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %3372 = llvm.extractvalue %3370[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %3373 = llvm.getelementptr %3371[%3372] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %3374 = llvm.getelementptr %3373[%1836] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %3374, %49, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %3375 = llvm.getelementptr %3373[%1843] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %3375, %66, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %3376 = vector.from_elements %3292, %3294, %3296, %3298 : vector<4xf16>
      %3377 = vector.from_elements %3256, %3257, %3258, %3259 : vector<4xf32>
      %3378 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2972, %3376, %3377 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3379 = vector.from_elements %3302, %3304, %3306, %3308 : vector<4xf16>
      %3380 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2976, %3379, %3378 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3381 = vector.from_elements %3312, %3314, %3316, %3318 : vector<4xf16>
      %3382 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2979, %3381, %3380 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3383 = vector.from_elements %3322, %3324, %3326, %3328 : vector<4xf16>
      %3384 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2982, %3383, %3382 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3385 = vector.from_elements %3332, %3334, %3336, %3338 : vector<4xf16>
      %3386 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2985, %3385, %3384 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3387 = vector.from_elements %3342, %3344, %3346, %3348 : vector<4xf16>
      %3388 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2988, %3387, %3386 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3389 = vector.from_elements %3352, %3354, %3356, %3358 : vector<4xf16>
      %3390 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2991, %3389, %3388 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3391 = vector.from_elements %3362, %3364, %3366, %3368 : vector<4xf16>
      %3392 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2994, %3391, %3390 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3393 = vector.extract %3392[0] : f32 from vector<4xf32>
      %3394 = vector.extract %3392[1] : f32 from vector<4xf32>
      %3395 = vector.extract %3392[2] : f32 from vector<4xf32>
      %3396 = vector.extract %3392[3] : f32 from vector<4xf32>
      %3397 = vector.from_elements %3293, %3295, %3297, %3299 : vector<4xf16>
      %3398 = vector.from_elements %3266, %3267, %3268, %3269 : vector<4xf32>
      %3399 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2972, %3397, %3398 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3400 = vector.from_elements %3303, %3305, %3307, %3309 : vector<4xf16>
      %3401 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2976, %3400, %3399 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3402 = vector.from_elements %3313, %3315, %3317, %3319 : vector<4xf16>
      %3403 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2979, %3402, %3401 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3404 = vector.from_elements %3323, %3325, %3327, %3329 : vector<4xf16>
      %3405 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2982, %3404, %3403 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3406 = vector.from_elements %3333, %3335, %3337, %3339 : vector<4xf16>
      %3407 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2985, %3406, %3405 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3408 = vector.from_elements %3343, %3345, %3347, %3349 : vector<4xf16>
      %3409 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2988, %3408, %3407 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3410 = vector.from_elements %3353, %3355, %3357, %3359 : vector<4xf16>
      %3411 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2991, %3410, %3409 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3412 = vector.from_elements %3363, %3365, %3367, %3369 : vector<4xf16>
      %3413 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2994, %3412, %3411 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3414 = vector.extract %3413[0] : f32 from vector<4xf32>
      %3415 = vector.extract %3413[1] : f32 from vector<4xf32>
      %3416 = vector.extract %3413[2] : f32 from vector<4xf32>
      %3417 = vector.extract %3413[3] : f32 from vector<4xf32>
      %3418 = vector.from_elements %3276, %3277, %3278, %3279 : vector<4xf32>
      %3419 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3022, %3376, %3418 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3420 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3025, %3379, %3419 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3421 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3027, %3381, %3420 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3422 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3029, %3383, %3421 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3423 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3031, %3385, %3422 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3424 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3033, %3387, %3423 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3425 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3035, %3389, %3424 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3426 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3037, %3391, %3425 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3427 = vector.extract %3426[0] : f32 from vector<4xf32>
      %3428 = vector.extract %3426[1] : f32 from vector<4xf32>
      %3429 = vector.extract %3426[2] : f32 from vector<4xf32>
      %3430 = vector.extract %3426[3] : f32 from vector<4xf32>
      %3431 = vector.from_elements %3286, %3287, %3288, %3289 : vector<4xf32>
      %3432 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3022, %3397, %3431 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3433 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3025, %3400, %3432 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3434 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3027, %3402, %3433 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3435 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3029, %3404, %3434 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3436 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3031, %3406, %3435 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3437 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3033, %3408, %3436 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3438 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3035, %3410, %3437 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3439 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3037, %3412, %3438 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3440 = vector.extract %3439[0] : f32 from vector<4xf32>
      %3441 = vector.extract %3439[1] : f32 from vector<4xf32>
      %3442 = vector.extract %3439[2] : f32 from vector<4xf32>
      %3443 = vector.extract %3439[3] : f32 from vector<4xf32>
      %3444 = vector.extract %arg10[0, 0] : f32 from vector<2x8xf32>
      %3445 = vector.extract %arg10[0, 1] : f32 from vector<2x8xf32>
      %3446 = vector.extract %arg10[0, 2] : f32 from vector<2x8xf32>
      %3447 = vector.extract %arg10[0, 3] : f32 from vector<2x8xf32>
      %3448 = vector.from_elements %3444, %3445, %3446, %3447 : vector<4xf32>
      %3449 = arith.mulf %3448, %2840 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3450 = vector.extract %3449[0] : f32 from vector<4xf32>
      %3451 = vector.extract %3449[1] : f32 from vector<4xf32>
      %3452 = vector.extract %3449[2] : f32 from vector<4xf32>
      %3453 = vector.extract %3449[3] : f32 from vector<4xf32>
      %3454 = vector.extract %arg10[0, 4] : f32 from vector<2x8xf32>
      %3455 = vector.extract %arg10[0, 5] : f32 from vector<2x8xf32>
      %3456 = vector.extract %arg10[0, 6] : f32 from vector<2x8xf32>
      %3457 = vector.extract %arg10[0, 7] : f32 from vector<2x8xf32>
      %3458 = vector.from_elements %3454, %3455, %3456, %3457 : vector<4xf32>
      %3459 = arith.mulf %3458, %2840 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3460 = vector.extract %3459[0] : f32 from vector<4xf32>
      %3461 = vector.extract %3459[1] : f32 from vector<4xf32>
      %3462 = vector.extract %3459[2] : f32 from vector<4xf32>
      %3463 = vector.extract %3459[3] : f32 from vector<4xf32>
      %3464 = vector.extract %arg10[1, 0] : f32 from vector<2x8xf32>
      %3465 = vector.extract %arg10[1, 1] : f32 from vector<2x8xf32>
      %3466 = vector.extract %arg10[1, 2] : f32 from vector<2x8xf32>
      %3467 = vector.extract %arg10[1, 3] : f32 from vector<2x8xf32>
      %3468 = vector.from_elements %3464, %3465, %3466, %3467 : vector<4xf32>
      %3469 = arith.mulf %3468, %2861 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3470 = vector.extract %3469[0] : f32 from vector<4xf32>
      %3471 = vector.extract %3469[1] : f32 from vector<4xf32>
      %3472 = vector.extract %3469[2] : f32 from vector<4xf32>
      %3473 = vector.extract %3469[3] : f32 from vector<4xf32>
      %3474 = vector.extract %arg10[1, 4] : f32 from vector<2x8xf32>
      %3475 = vector.extract %arg10[1, 5] : f32 from vector<2x8xf32>
      %3476 = vector.extract %arg10[1, 6] : f32 from vector<2x8xf32>
      %3477 = vector.extract %arg10[1, 7] : f32 from vector<2x8xf32>
      %3478 = vector.from_elements %3474, %3475, %3476, %3477 : vector<4xf32>
      %3479 = arith.mulf %3478, %2861 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3480 = vector.extract %3479[0] : f32 from vector<4xf32>
      %3481 = vector.extract %3479[1] : f32 from vector<4xf32>
      %3482 = vector.extract %3479[2] : f32 from vector<4xf32>
      %3483 = vector.extract %3479[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %3484 = frisk.read_matrix_from_shm %view[%2877] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3485 = vector.bitcast %3484 : vector<4xi32> to vector<8xf16>
      %3486 = vector.extract %3485[0] : f16 from vector<8xf16>
      %3487 = vector.extract %3485[%279] : f16 from vector<8xf16>
      %3488 = vector.extract %3485[1] : f16 from vector<8xf16>
      %3489 = vector.extract %3485[%280] : f16 from vector<8xf16>
      %3490 = vector.extract %3485[2] : f16 from vector<8xf16>
      %3491 = vector.extract %3485[%281] : f16 from vector<8xf16>
      %3492 = vector.extract %3485[3] : f16 from vector<8xf16>
      %3493 = vector.extract %3485[%282] : f16 from vector<8xf16>
      %3494 = frisk.read_matrix_from_shm %view[%2888] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3495 = vector.bitcast %3494 : vector<4xi32> to vector<8xf16>
      %3496 = vector.extract %3495[0] : f16 from vector<8xf16>
      %3497 = vector.extract %3495[%279] : f16 from vector<8xf16>
      %3498 = vector.extract %3495[1] : f16 from vector<8xf16>
      %3499 = vector.extract %3495[%280] : f16 from vector<8xf16>
      %3500 = vector.extract %3495[2] : f16 from vector<8xf16>
      %3501 = vector.extract %3495[%281] : f16 from vector<8xf16>
      %3502 = vector.extract %3495[3] : f16 from vector<8xf16>
      %3503 = vector.extract %3495[%282] : f16 from vector<8xf16>
      %3504 = frisk.read_matrix_from_shm %view[%2899] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3505 = vector.bitcast %3504 : vector<4xi32> to vector<8xf16>
      %3506 = vector.extract %3505[0] : f16 from vector<8xf16>
      %3507 = vector.extract %3505[%279] : f16 from vector<8xf16>
      %3508 = vector.extract %3505[1] : f16 from vector<8xf16>
      %3509 = vector.extract %3505[%280] : f16 from vector<8xf16>
      %3510 = vector.extract %3505[2] : f16 from vector<8xf16>
      %3511 = vector.extract %3505[%281] : f16 from vector<8xf16>
      %3512 = vector.extract %3505[3] : f16 from vector<8xf16>
      %3513 = vector.extract %3505[%282] : f16 from vector<8xf16>
      %3514 = frisk.read_matrix_from_shm %view[%2910] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3515 = vector.bitcast %3514 : vector<4xi32> to vector<8xf16>
      %3516 = vector.extract %3515[0] : f16 from vector<8xf16>
      %3517 = vector.extract %3515[%279] : f16 from vector<8xf16>
      %3518 = vector.extract %3515[1] : f16 from vector<8xf16>
      %3519 = vector.extract %3515[%280] : f16 from vector<8xf16>
      %3520 = vector.extract %3515[2] : f16 from vector<8xf16>
      %3521 = vector.extract %3515[%281] : f16 from vector<8xf16>
      %3522 = vector.extract %3515[3] : f16 from vector<8xf16>
      %3523 = vector.extract %3515[%282] : f16 from vector<8xf16>
      %3524 = frisk.read_matrix_from_shm %view[%2921] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3525 = vector.bitcast %3524 : vector<4xi32> to vector<8xf16>
      %3526 = vector.extract %3525[0] : f16 from vector<8xf16>
      %3527 = vector.extract %3525[%279] : f16 from vector<8xf16>
      %3528 = vector.extract %3525[1] : f16 from vector<8xf16>
      %3529 = vector.extract %3525[%280] : f16 from vector<8xf16>
      %3530 = vector.extract %3525[2] : f16 from vector<8xf16>
      %3531 = vector.extract %3525[%281] : f16 from vector<8xf16>
      %3532 = vector.extract %3525[3] : f16 from vector<8xf16>
      %3533 = vector.extract %3525[%282] : f16 from vector<8xf16>
      %3534 = frisk.read_matrix_from_shm %view[%2932] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3535 = vector.bitcast %3534 : vector<4xi32> to vector<8xf16>
      %3536 = vector.extract %3535[0] : f16 from vector<8xf16>
      %3537 = vector.extract %3535[%279] : f16 from vector<8xf16>
      %3538 = vector.extract %3535[1] : f16 from vector<8xf16>
      %3539 = vector.extract %3535[%280] : f16 from vector<8xf16>
      %3540 = vector.extract %3535[2] : f16 from vector<8xf16>
      %3541 = vector.extract %3535[%281] : f16 from vector<8xf16>
      %3542 = vector.extract %3535[3] : f16 from vector<8xf16>
      %3543 = vector.extract %3535[%282] : f16 from vector<8xf16>
      %3544 = frisk.read_matrix_from_shm %view[%2943] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3545 = vector.bitcast %3544 : vector<4xi32> to vector<8xf16>
      %3546 = vector.extract %3545[0] : f16 from vector<8xf16>
      %3547 = vector.extract %3545[%279] : f16 from vector<8xf16>
      %3548 = vector.extract %3545[1] : f16 from vector<8xf16>
      %3549 = vector.extract %3545[%280] : f16 from vector<8xf16>
      %3550 = vector.extract %3545[2] : f16 from vector<8xf16>
      %3551 = vector.extract %3545[%281] : f16 from vector<8xf16>
      %3552 = vector.extract %3545[3] : f16 from vector<8xf16>
      %3553 = vector.extract %3545[%282] : f16 from vector<8xf16>
      %3554 = frisk.read_matrix_from_shm %view[%2954] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3555 = vector.bitcast %3554 : vector<4xi32> to vector<8xf16>
      %3556 = vector.extract %3555[0] : f16 from vector<8xf16>
      %3557 = vector.extract %3555[%279] : f16 from vector<8xf16>
      %3558 = vector.extract %3555[1] : f16 from vector<8xf16>
      %3559 = vector.extract %3555[%280] : f16 from vector<8xf16>
      %3560 = vector.extract %3555[2] : f16 from vector<8xf16>
      %3561 = vector.extract %3555[%281] : f16 from vector<8xf16>
      %3562 = vector.extract %3555[3] : f16 from vector<8xf16>
      %3563 = vector.extract %3555[%282] : f16 from vector<8xf16>
      %3564 = vector.from_elements %3486, %3488, %3490, %3492 : vector<4xf16>
      %3565 = vector.from_elements %3450, %3451, %3452, %3453 : vector<4xf32>
      %3566 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2972, %3564, %3565 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3567 = vector.from_elements %3496, %3498, %3500, %3502 : vector<4xf16>
      %3568 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2976, %3567, %3566 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3569 = vector.from_elements %3506, %3508, %3510, %3512 : vector<4xf16>
      %3570 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2979, %3569, %3568 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3571 = vector.from_elements %3516, %3518, %3520, %3522 : vector<4xf16>
      %3572 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2982, %3571, %3570 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3573 = vector.from_elements %3526, %3528, %3530, %3532 : vector<4xf16>
      %3574 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2985, %3573, %3572 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3575 = vector.from_elements %3536, %3538, %3540, %3542 : vector<4xf16>
      %3576 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2988, %3575, %3574 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3577 = vector.from_elements %3546, %3548, %3550, %3552 : vector<4xf16>
      %3578 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2991, %3577, %3576 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3579 = vector.from_elements %3556, %3558, %3560, %3562 : vector<4xf16>
      %3580 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2994, %3579, %3578 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3581 = vector.extract %3580[0] : f32 from vector<4xf32>
      %3582 = vector.extract %3580[1] : f32 from vector<4xf32>
      %3583 = vector.extract %3580[2] : f32 from vector<4xf32>
      %3584 = vector.extract %3580[3] : f32 from vector<4xf32>
      %3585 = vector.from_elements %3487, %3489, %3491, %3493 : vector<4xf16>
      %3586 = vector.from_elements %3460, %3461, %3462, %3463 : vector<4xf32>
      %3587 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2972, %3585, %3586 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3588 = vector.from_elements %3497, %3499, %3501, %3503 : vector<4xf16>
      %3589 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2976, %3588, %3587 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3590 = vector.from_elements %3507, %3509, %3511, %3513 : vector<4xf16>
      %3591 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2979, %3590, %3589 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3592 = vector.from_elements %3517, %3519, %3521, %3523 : vector<4xf16>
      %3593 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2982, %3592, %3591 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3594 = vector.from_elements %3527, %3529, %3531, %3533 : vector<4xf16>
      %3595 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2985, %3594, %3593 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3596 = vector.from_elements %3537, %3539, %3541, %3543 : vector<4xf16>
      %3597 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2988, %3596, %3595 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3598 = vector.from_elements %3547, %3549, %3551, %3553 : vector<4xf16>
      %3599 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2991, %3598, %3597 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3600 = vector.from_elements %3557, %3559, %3561, %3563 : vector<4xf16>
      %3601 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2994, %3600, %3599 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3602 = vector.extract %3601[0] : f32 from vector<4xf32>
      %3603 = vector.extract %3601[1] : f32 from vector<4xf32>
      %3604 = vector.extract %3601[2] : f32 from vector<4xf32>
      %3605 = vector.extract %3601[3] : f32 from vector<4xf32>
      %3606 = vector.from_elements %3470, %3471, %3472, %3473 : vector<4xf32>
      %3607 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3022, %3564, %3606 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3608 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3025, %3567, %3607 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3609 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3027, %3569, %3608 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3610 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3029, %3571, %3609 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3611 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3031, %3573, %3610 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3612 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3033, %3575, %3611 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3613 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3035, %3577, %3612 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3614 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3037, %3579, %3613 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3615 = vector.extract %3614[0] : f32 from vector<4xf32>
      %3616 = vector.extract %3614[1] : f32 from vector<4xf32>
      %3617 = vector.extract %3614[2] : f32 from vector<4xf32>
      %3618 = vector.extract %3614[3] : f32 from vector<4xf32>
      %3619 = vector.from_elements %3480, %3481, %3482, %3483 : vector<4xf32>
      %3620 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3022, %3585, %3619 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3621 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3025, %3588, %3620 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3622 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3027, %3590, %3621 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3623 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3029, %3592, %3622 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3624 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3031, %3594, %3623 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3625 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3033, %3596, %3624 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3626 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3035, %3598, %3625 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3627 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %3037, %3600, %3626 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3628 = vector.extract %3627[0] : f32 from vector<4xf32>
      %3629 = vector.extract %3627[1] : f32 from vector<4xf32>
      %3630 = vector.extract %3627[2] : f32 from vector<4xf32>
      %3631 = vector.extract %3627[3] : f32 from vector<4xf32>
      %3632 = vector.from_elements %2997, %2998, %2999, %3000, %3018, %3019, %3020, %3021, %3039, %3040, %3041, %3042, %3052, %3053, %3054, %3055 : vector<2x8xf32>
      %3633 = vector.from_elements %2734, %2738 : vector<2x1xf32>
      %3634 = vector.from_elements %2435, %2440 : vector<2x1xf32>
      %3635 = vector.from_elements %3199, %3200, %3201, %3202, %3220, %3221, %3222, %3223, %3233, %3234, %3235, %3236, %3246, %3247, %3248, %3249 : vector<2x8xf32>
      %3636 = vector.from_elements %3393, %3394, %3395, %3396, %3414, %3415, %3416, %3417, %3427, %3428, %3429, %3430, %3440, %3441, %3442, %3443 : vector<2x8xf32>
      %3637 = vector.from_elements %3581, %3582, %3583, %3584, %3602, %3603, %3604, %3605, %3615, %3616, %3617, %3618, %3628, %3629, %3630, %3631 : vector<2x8xf32>
      gpu.barrier
      affine.yield %3632, %3633, %3634, %3635, %3636, %3637 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %284 = arith.index_cast %thread_id_x : index to i32
    %285 = arith.andi %284, %c16_i32 : i32
    %286 = arith.cmpi ne, %285, %c0_i32 : i32
    %287 = arith.andi %284, %c32_i32 : i32
    %288 = arith.cmpi ne, %287, %c0_i32 : i32
    %289 = vector.extract %283#5[0, 0] : f32 from vector<2x8xf32>
    %290 = vector.extract %283#5[0, 1] : f32 from vector<2x8xf32>
    %291 = vector.extract %283#5[0, 2] : f32 from vector<2x8xf32>
    %292 = vector.extract %283#5[0, 3] : f32 from vector<2x8xf32>
    %293 = arith.select %286, %289, %290 : f32
    %294 = llvm.bitcast %293 : f32 to i32
    %295 = rocdl.ds_bpermute %10, %294 : (i32, i32) -> i32
    %296 = llvm.bitcast %295 : i32 to f32
    %297 = arith.select %286, %296, %289 : f32
    %298 = arith.select %286, %290, %296 : f32
    %299 = arith.select %286, %291, %292 : f32
    %300 = llvm.bitcast %299 : f32 to i32
    %301 = rocdl.ds_bpermute %10, %300 : (i32, i32) -> i32
    %302 = llvm.bitcast %301 : i32 to f32
    %303 = arith.select %286, %302, %291 : f32
    %304 = arith.select %286, %292, %302 : f32
    %305 = arith.select %288, %297, %303 : f32
    %306 = llvm.bitcast %305 : f32 to i32
    %307 = rocdl.ds_bpermute %14, %306 : (i32, i32) -> i32
    %308 = llvm.bitcast %307 : i32 to f32
    %309 = arith.select %288, %308, %297 : f32
    %310 = arith.select %288, %303, %308 : f32
    %311 = arith.select %288, %298, %304 : f32
    %312 = llvm.bitcast %311 : f32 to i32
    %313 = rocdl.ds_bpermute %14, %312 : (i32, i32) -> i32
    %314 = llvm.bitcast %313 : i32 to f32
    %315 = arith.select %288, %314, %298 : f32
    %316 = arith.select %288, %304, %314 : f32
    %317 = vector.extract %283#5[0, 4] : f32 from vector<2x8xf32>
    %318 = vector.extract %283#5[0, 5] : f32 from vector<2x8xf32>
    %319 = vector.extract %283#5[0, 6] : f32 from vector<2x8xf32>
    %320 = vector.extract %283#5[0, 7] : f32 from vector<2x8xf32>
    %321 = arith.select %286, %317, %318 : f32
    %322 = llvm.bitcast %321 : f32 to i32
    %323 = rocdl.ds_bpermute %10, %322 : (i32, i32) -> i32
    %324 = llvm.bitcast %323 : i32 to f32
    %325 = arith.select %286, %324, %317 : f32
    %326 = arith.select %286, %318, %324 : f32
    %327 = arith.select %286, %319, %320 : f32
    %328 = llvm.bitcast %327 : f32 to i32
    %329 = rocdl.ds_bpermute %10, %328 : (i32, i32) -> i32
    %330 = llvm.bitcast %329 : i32 to f32
    %331 = arith.select %286, %330, %319 : f32
    %332 = arith.select %286, %320, %330 : f32
    %333 = arith.select %288, %325, %331 : f32
    %334 = llvm.bitcast %333 : f32 to i32
    %335 = rocdl.ds_bpermute %14, %334 : (i32, i32) -> i32
    %336 = llvm.bitcast %335 : i32 to f32
    %337 = arith.select %288, %336, %325 : f32
    %338 = arith.select %288, %331, %336 : f32
    %339 = arith.select %288, %326, %332 : f32
    %340 = llvm.bitcast %339 : f32 to i32
    %341 = rocdl.ds_bpermute %14, %340 : (i32, i32) -> i32
    %342 = llvm.bitcast %341 : i32 to f32
    %343 = arith.select %288, %342, %326 : f32
    %344 = arith.select %288, %332, %342 : f32
    %345 = vector.extract %283#5[1, 0] : f32 from vector<2x8xf32>
    %346 = vector.extract %283#5[1, 1] : f32 from vector<2x8xf32>
    %347 = vector.extract %283#5[1, 2] : f32 from vector<2x8xf32>
    %348 = vector.extract %283#5[1, 3] : f32 from vector<2x8xf32>
    %349 = arith.select %286, %345, %346 : f32
    %350 = llvm.bitcast %349 : f32 to i32
    %351 = rocdl.ds_bpermute %10, %350 : (i32, i32) -> i32
    %352 = llvm.bitcast %351 : i32 to f32
    %353 = arith.select %286, %352, %345 : f32
    %354 = arith.select %286, %346, %352 : f32
    %355 = arith.select %286, %347, %348 : f32
    %356 = llvm.bitcast %355 : f32 to i32
    %357 = rocdl.ds_bpermute %10, %356 : (i32, i32) -> i32
    %358 = llvm.bitcast %357 : i32 to f32
    %359 = arith.select %286, %358, %347 : f32
    %360 = arith.select %286, %348, %358 : f32
    %361 = arith.select %288, %353, %359 : f32
    %362 = llvm.bitcast %361 : f32 to i32
    %363 = rocdl.ds_bpermute %14, %362 : (i32, i32) -> i32
    %364 = llvm.bitcast %363 : i32 to f32
    %365 = arith.select %288, %364, %353 : f32
    %366 = arith.select %288, %359, %364 : f32
    %367 = arith.select %288, %354, %360 : f32
    %368 = llvm.bitcast %367 : f32 to i32
    %369 = rocdl.ds_bpermute %14, %368 : (i32, i32) -> i32
    %370 = llvm.bitcast %369 : i32 to f32
    %371 = arith.select %288, %370, %354 : f32
    %372 = arith.select %288, %360, %370 : f32
    %373 = vector.extract %283#5[1, 4] : f32 from vector<2x8xf32>
    %374 = vector.extract %283#5[1, 5] : f32 from vector<2x8xf32>
    %375 = vector.extract %283#5[1, 6] : f32 from vector<2x8xf32>
    %376 = vector.extract %283#5[1, 7] : f32 from vector<2x8xf32>
    %377 = arith.select %286, %373, %374 : f32
    %378 = llvm.bitcast %377 : f32 to i32
    %379 = rocdl.ds_bpermute %10, %378 : (i32, i32) -> i32
    %380 = llvm.bitcast %379 : i32 to f32
    %381 = arith.select %286, %380, %373 : f32
    %382 = arith.select %286, %374, %380 : f32
    %383 = arith.select %286, %375, %376 : f32
    %384 = llvm.bitcast %383 : f32 to i32
    %385 = rocdl.ds_bpermute %10, %384 : (i32, i32) -> i32
    %386 = llvm.bitcast %385 : i32 to f32
    %387 = arith.select %286, %386, %375 : f32
    %388 = arith.select %286, %376, %386 : f32
    %389 = arith.select %288, %381, %387 : f32
    %390 = llvm.bitcast %389 : f32 to i32
    %391 = rocdl.ds_bpermute %14, %390 : (i32, i32) -> i32
    %392 = llvm.bitcast %391 : i32 to f32
    %393 = arith.select %288, %392, %381 : f32
    %394 = arith.select %288, %387, %392 : f32
    %395 = arith.select %288, %382, %388 : f32
    %396 = llvm.bitcast %395 : f32 to i32
    %397 = rocdl.ds_bpermute %14, %396 : (i32, i32) -> i32
    %398 = llvm.bitcast %397 : i32 to f32
    %399 = arith.select %288, %398, %382 : f32
    %400 = arith.select %288, %388, %398 : f32
    %401 = vector.extract %283#4[0, 0] : f32 from vector<2x8xf32>
    %402 = vector.extract %283#4[0, 1] : f32 from vector<2x8xf32>
    %403 = vector.extract %283#4[0, 2] : f32 from vector<2x8xf32>
    %404 = vector.extract %283#4[0, 3] : f32 from vector<2x8xf32>
    %405 = arith.select %286, %401, %402 : f32
    %406 = llvm.bitcast %405 : f32 to i32
    %407 = rocdl.ds_bpermute %10, %406 : (i32, i32) -> i32
    %408 = llvm.bitcast %407 : i32 to f32
    %409 = arith.select %286, %408, %401 : f32
    %410 = arith.select %286, %402, %408 : f32
    %411 = arith.select %286, %403, %404 : f32
    %412 = llvm.bitcast %411 : f32 to i32
    %413 = rocdl.ds_bpermute %10, %412 : (i32, i32) -> i32
    %414 = llvm.bitcast %413 : i32 to f32
    %415 = arith.select %286, %414, %403 : f32
    %416 = arith.select %286, %404, %414 : f32
    %417 = arith.select %288, %409, %415 : f32
    %418 = llvm.bitcast %417 : f32 to i32
    %419 = rocdl.ds_bpermute %14, %418 : (i32, i32) -> i32
    %420 = llvm.bitcast %419 : i32 to f32
    %421 = arith.select %288, %420, %409 : f32
    %422 = arith.select %288, %415, %420 : f32
    %423 = arith.select %288, %410, %416 : f32
    %424 = llvm.bitcast %423 : f32 to i32
    %425 = rocdl.ds_bpermute %14, %424 : (i32, i32) -> i32
    %426 = llvm.bitcast %425 : i32 to f32
    %427 = arith.select %288, %426, %410 : f32
    %428 = arith.select %288, %416, %426 : f32
    %429 = vector.extract %283#4[0, 4] : f32 from vector<2x8xf32>
    %430 = vector.extract %283#4[0, 5] : f32 from vector<2x8xf32>
    %431 = vector.extract %283#4[0, 6] : f32 from vector<2x8xf32>
    %432 = vector.extract %283#4[0, 7] : f32 from vector<2x8xf32>
    %433 = arith.select %286, %429, %430 : f32
    %434 = llvm.bitcast %433 : f32 to i32
    %435 = rocdl.ds_bpermute %10, %434 : (i32, i32) -> i32
    %436 = llvm.bitcast %435 : i32 to f32
    %437 = arith.select %286, %436, %429 : f32
    %438 = arith.select %286, %430, %436 : f32
    %439 = arith.select %286, %431, %432 : f32
    %440 = llvm.bitcast %439 : f32 to i32
    %441 = rocdl.ds_bpermute %10, %440 : (i32, i32) -> i32
    %442 = llvm.bitcast %441 : i32 to f32
    %443 = arith.select %286, %442, %431 : f32
    %444 = arith.select %286, %432, %442 : f32
    %445 = arith.select %288, %437, %443 : f32
    %446 = llvm.bitcast %445 : f32 to i32
    %447 = rocdl.ds_bpermute %14, %446 : (i32, i32) -> i32
    %448 = llvm.bitcast %447 : i32 to f32
    %449 = arith.select %288, %448, %437 : f32
    %450 = arith.select %288, %443, %448 : f32
    %451 = arith.select %288, %438, %444 : f32
    %452 = llvm.bitcast %451 : f32 to i32
    %453 = rocdl.ds_bpermute %14, %452 : (i32, i32) -> i32
    %454 = llvm.bitcast %453 : i32 to f32
    %455 = arith.select %288, %454, %438 : f32
    %456 = arith.select %288, %444, %454 : f32
    %457 = vector.extract %283#4[1, 0] : f32 from vector<2x8xf32>
    %458 = vector.extract %283#4[1, 1] : f32 from vector<2x8xf32>
    %459 = vector.extract %283#4[1, 2] : f32 from vector<2x8xf32>
    %460 = vector.extract %283#4[1, 3] : f32 from vector<2x8xf32>
    %461 = arith.select %286, %457, %458 : f32
    %462 = llvm.bitcast %461 : f32 to i32
    %463 = rocdl.ds_bpermute %10, %462 : (i32, i32) -> i32
    %464 = llvm.bitcast %463 : i32 to f32
    %465 = arith.select %286, %464, %457 : f32
    %466 = arith.select %286, %458, %464 : f32
    %467 = arith.select %286, %459, %460 : f32
    %468 = llvm.bitcast %467 : f32 to i32
    %469 = rocdl.ds_bpermute %10, %468 : (i32, i32) -> i32
    %470 = llvm.bitcast %469 : i32 to f32
    %471 = arith.select %286, %470, %459 : f32
    %472 = arith.select %286, %460, %470 : f32
    %473 = arith.select %288, %465, %471 : f32
    %474 = llvm.bitcast %473 : f32 to i32
    %475 = rocdl.ds_bpermute %14, %474 : (i32, i32) -> i32
    %476 = llvm.bitcast %475 : i32 to f32
    %477 = arith.select %288, %476, %465 : f32
    %478 = arith.select %288, %471, %476 : f32
    %479 = arith.select %288, %466, %472 : f32
    %480 = llvm.bitcast %479 : f32 to i32
    %481 = rocdl.ds_bpermute %14, %480 : (i32, i32) -> i32
    %482 = llvm.bitcast %481 : i32 to f32
    %483 = arith.select %288, %482, %466 : f32
    %484 = arith.select %288, %472, %482 : f32
    %485 = vector.extract %283#4[1, 4] : f32 from vector<2x8xf32>
    %486 = vector.extract %283#4[1, 5] : f32 from vector<2x8xf32>
    %487 = vector.extract %283#4[1, 6] : f32 from vector<2x8xf32>
    %488 = vector.extract %283#4[1, 7] : f32 from vector<2x8xf32>
    %489 = arith.select %286, %485, %486 : f32
    %490 = llvm.bitcast %489 : f32 to i32
    %491 = rocdl.ds_bpermute %10, %490 : (i32, i32) -> i32
    %492 = llvm.bitcast %491 : i32 to f32
    %493 = arith.select %286, %492, %485 : f32
    %494 = arith.select %286, %486, %492 : f32
    %495 = arith.select %286, %487, %488 : f32
    %496 = llvm.bitcast %495 : f32 to i32
    %497 = rocdl.ds_bpermute %10, %496 : (i32, i32) -> i32
    %498 = llvm.bitcast %497 : i32 to f32
    %499 = arith.select %286, %498, %487 : f32
    %500 = arith.select %286, %488, %498 : f32
    %501 = arith.select %288, %493, %499 : f32
    %502 = llvm.bitcast %501 : f32 to i32
    %503 = rocdl.ds_bpermute %14, %502 : (i32, i32) -> i32
    %504 = llvm.bitcast %503 : i32 to f32
    %505 = arith.select %288, %504, %493 : f32
    %506 = arith.select %288, %499, %504 : f32
    %507 = arith.select %288, %494, %500 : f32
    %508 = llvm.bitcast %507 : f32 to i32
    %509 = rocdl.ds_bpermute %14, %508 : (i32, i32) -> i32
    %510 = llvm.bitcast %509 : i32 to f32
    %511 = arith.select %288, %510, %494 : f32
    %512 = arith.select %288, %500, %510 : f32
    %513 = vector.extract %283#3[0, 0] : f32 from vector<2x8xf32>
    %514 = vector.extract %283#3[0, 1] : f32 from vector<2x8xf32>
    %515 = vector.extract %283#3[0, 2] : f32 from vector<2x8xf32>
    %516 = vector.extract %283#3[0, 3] : f32 from vector<2x8xf32>
    %517 = arith.select %286, %513, %514 : f32
    %518 = llvm.bitcast %517 : f32 to i32
    %519 = rocdl.ds_bpermute %10, %518 : (i32, i32) -> i32
    %520 = llvm.bitcast %519 : i32 to f32
    %521 = arith.select %286, %520, %513 : f32
    %522 = arith.select %286, %514, %520 : f32
    %523 = arith.select %286, %515, %516 : f32
    %524 = llvm.bitcast %523 : f32 to i32
    %525 = rocdl.ds_bpermute %10, %524 : (i32, i32) -> i32
    %526 = llvm.bitcast %525 : i32 to f32
    %527 = arith.select %286, %526, %515 : f32
    %528 = arith.select %286, %516, %526 : f32
    %529 = arith.select %288, %521, %527 : f32
    %530 = llvm.bitcast %529 : f32 to i32
    %531 = rocdl.ds_bpermute %14, %530 : (i32, i32) -> i32
    %532 = llvm.bitcast %531 : i32 to f32
    %533 = arith.select %288, %532, %521 : f32
    %534 = arith.select %288, %527, %532 : f32
    %535 = arith.select %288, %522, %528 : f32
    %536 = llvm.bitcast %535 : f32 to i32
    %537 = rocdl.ds_bpermute %14, %536 : (i32, i32) -> i32
    %538 = llvm.bitcast %537 : i32 to f32
    %539 = arith.select %288, %538, %522 : f32
    %540 = arith.select %288, %528, %538 : f32
    %541 = vector.extract %283#3[0, 4] : f32 from vector<2x8xf32>
    %542 = vector.extract %283#3[0, 5] : f32 from vector<2x8xf32>
    %543 = vector.extract %283#3[0, 6] : f32 from vector<2x8xf32>
    %544 = vector.extract %283#3[0, 7] : f32 from vector<2x8xf32>
    %545 = arith.select %286, %541, %542 : f32
    %546 = llvm.bitcast %545 : f32 to i32
    %547 = rocdl.ds_bpermute %10, %546 : (i32, i32) -> i32
    %548 = llvm.bitcast %547 : i32 to f32
    %549 = arith.select %286, %548, %541 : f32
    %550 = arith.select %286, %542, %548 : f32
    %551 = arith.select %286, %543, %544 : f32
    %552 = llvm.bitcast %551 : f32 to i32
    %553 = rocdl.ds_bpermute %10, %552 : (i32, i32) -> i32
    %554 = llvm.bitcast %553 : i32 to f32
    %555 = arith.select %286, %554, %543 : f32
    %556 = arith.select %286, %544, %554 : f32
    %557 = arith.select %288, %549, %555 : f32
    %558 = llvm.bitcast %557 : f32 to i32
    %559 = rocdl.ds_bpermute %14, %558 : (i32, i32) -> i32
    %560 = llvm.bitcast %559 : i32 to f32
    %561 = arith.select %288, %560, %549 : f32
    %562 = arith.select %288, %555, %560 : f32
    %563 = arith.select %288, %550, %556 : f32
    %564 = llvm.bitcast %563 : f32 to i32
    %565 = rocdl.ds_bpermute %14, %564 : (i32, i32) -> i32
    %566 = llvm.bitcast %565 : i32 to f32
    %567 = arith.select %288, %566, %550 : f32
    %568 = arith.select %288, %556, %566 : f32
    %569 = vector.extract %283#3[1, 0] : f32 from vector<2x8xf32>
    %570 = vector.extract %283#3[1, 1] : f32 from vector<2x8xf32>
    %571 = vector.extract %283#3[1, 2] : f32 from vector<2x8xf32>
    %572 = vector.extract %283#3[1, 3] : f32 from vector<2x8xf32>
    %573 = arith.select %286, %569, %570 : f32
    %574 = llvm.bitcast %573 : f32 to i32
    %575 = rocdl.ds_bpermute %10, %574 : (i32, i32) -> i32
    %576 = llvm.bitcast %575 : i32 to f32
    %577 = arith.select %286, %576, %569 : f32
    %578 = arith.select %286, %570, %576 : f32
    %579 = arith.select %286, %571, %572 : f32
    %580 = llvm.bitcast %579 : f32 to i32
    %581 = rocdl.ds_bpermute %10, %580 : (i32, i32) -> i32
    %582 = llvm.bitcast %581 : i32 to f32
    %583 = arith.select %286, %582, %571 : f32
    %584 = arith.select %286, %572, %582 : f32
    %585 = arith.select %288, %577, %583 : f32
    %586 = llvm.bitcast %585 : f32 to i32
    %587 = rocdl.ds_bpermute %14, %586 : (i32, i32) -> i32
    %588 = llvm.bitcast %587 : i32 to f32
    %589 = arith.select %288, %588, %577 : f32
    %590 = arith.select %288, %583, %588 : f32
    %591 = arith.select %288, %578, %584 : f32
    %592 = llvm.bitcast %591 : f32 to i32
    %593 = rocdl.ds_bpermute %14, %592 : (i32, i32) -> i32
    %594 = llvm.bitcast %593 : i32 to f32
    %595 = arith.select %288, %594, %578 : f32
    %596 = arith.select %288, %584, %594 : f32
    %597 = vector.extract %283#3[1, 4] : f32 from vector<2x8xf32>
    %598 = vector.extract %283#3[1, 5] : f32 from vector<2x8xf32>
    %599 = vector.extract %283#3[1, 6] : f32 from vector<2x8xf32>
    %600 = vector.extract %283#3[1, 7] : f32 from vector<2x8xf32>
    %601 = arith.select %286, %597, %598 : f32
    %602 = llvm.bitcast %601 : f32 to i32
    %603 = rocdl.ds_bpermute %10, %602 : (i32, i32) -> i32
    %604 = llvm.bitcast %603 : i32 to f32
    %605 = arith.select %286, %604, %597 : f32
    %606 = arith.select %286, %598, %604 : f32
    %607 = arith.select %286, %599, %600 : f32
    %608 = llvm.bitcast %607 : f32 to i32
    %609 = rocdl.ds_bpermute %10, %608 : (i32, i32) -> i32
    %610 = llvm.bitcast %609 : i32 to f32
    %611 = arith.select %286, %610, %599 : f32
    %612 = arith.select %286, %600, %610 : f32
    %613 = arith.select %288, %605, %611 : f32
    %614 = llvm.bitcast %613 : f32 to i32
    %615 = rocdl.ds_bpermute %14, %614 : (i32, i32) -> i32
    %616 = llvm.bitcast %615 : i32 to f32
    %617 = arith.select %288, %616, %605 : f32
    %618 = arith.select %288, %611, %616 : f32
    %619 = arith.select %288, %606, %612 : f32
    %620 = llvm.bitcast %619 : f32 to i32
    %621 = rocdl.ds_bpermute %14, %620 : (i32, i32) -> i32
    %622 = llvm.bitcast %621 : i32 to f32
    %623 = arith.select %288, %622, %606 : f32
    %624 = arith.select %288, %612, %622 : f32
    %625 = vector.extract %283#0[0, 0] : f32 from vector<2x8xf32>
    %626 = vector.extract %283#0[0, 1] : f32 from vector<2x8xf32>
    %627 = vector.extract %283#0[0, 2] : f32 from vector<2x8xf32>
    %628 = vector.extract %283#0[0, 3] : f32 from vector<2x8xf32>
    %629 = arith.select %286, %625, %626 : f32
    %630 = llvm.bitcast %629 : f32 to i32
    %631 = rocdl.ds_bpermute %10, %630 : (i32, i32) -> i32
    %632 = llvm.bitcast %631 : i32 to f32
    %633 = arith.select %286, %632, %625 : f32
    %634 = arith.select %286, %626, %632 : f32
    %635 = arith.select %286, %627, %628 : f32
    %636 = llvm.bitcast %635 : f32 to i32
    %637 = rocdl.ds_bpermute %10, %636 : (i32, i32) -> i32
    %638 = llvm.bitcast %637 : i32 to f32
    %639 = arith.select %286, %638, %627 : f32
    %640 = arith.select %286, %628, %638 : f32
    %641 = arith.select %288, %633, %639 : f32
    %642 = llvm.bitcast %641 : f32 to i32
    %643 = rocdl.ds_bpermute %14, %642 : (i32, i32) -> i32
    %644 = llvm.bitcast %643 : i32 to f32
    %645 = arith.select %288, %644, %633 : f32
    %646 = arith.select %288, %639, %644 : f32
    %647 = arith.select %288, %634, %640 : f32
    %648 = llvm.bitcast %647 : f32 to i32
    %649 = rocdl.ds_bpermute %14, %648 : (i32, i32) -> i32
    %650 = llvm.bitcast %649 : i32 to f32
    %651 = arith.select %288, %650, %634 : f32
    %652 = arith.select %288, %640, %650 : f32
    %653 = vector.extract %283#0[0, 4] : f32 from vector<2x8xf32>
    %654 = vector.extract %283#0[0, 5] : f32 from vector<2x8xf32>
    %655 = vector.extract %283#0[0, 6] : f32 from vector<2x8xf32>
    %656 = vector.extract %283#0[0, 7] : f32 from vector<2x8xf32>
    %657 = arith.select %286, %653, %654 : f32
    %658 = llvm.bitcast %657 : f32 to i32
    %659 = rocdl.ds_bpermute %10, %658 : (i32, i32) -> i32
    %660 = llvm.bitcast %659 : i32 to f32
    %661 = arith.select %286, %660, %653 : f32
    %662 = arith.select %286, %654, %660 : f32
    %663 = arith.select %286, %655, %656 : f32
    %664 = llvm.bitcast %663 : f32 to i32
    %665 = rocdl.ds_bpermute %10, %664 : (i32, i32) -> i32
    %666 = llvm.bitcast %665 : i32 to f32
    %667 = arith.select %286, %666, %655 : f32
    %668 = arith.select %286, %656, %666 : f32
    %669 = arith.select %288, %661, %667 : f32
    %670 = llvm.bitcast %669 : f32 to i32
    %671 = rocdl.ds_bpermute %14, %670 : (i32, i32) -> i32
    %672 = llvm.bitcast %671 : i32 to f32
    %673 = arith.select %288, %672, %661 : f32
    %674 = arith.select %288, %667, %672 : f32
    %675 = arith.select %288, %662, %668 : f32
    %676 = llvm.bitcast %675 : f32 to i32
    %677 = rocdl.ds_bpermute %14, %676 : (i32, i32) -> i32
    %678 = llvm.bitcast %677 : i32 to f32
    %679 = arith.select %288, %678, %662 : f32
    %680 = arith.select %288, %668, %678 : f32
    %681 = vector.extract %283#0[1, 0] : f32 from vector<2x8xf32>
    %682 = vector.extract %283#0[1, 1] : f32 from vector<2x8xf32>
    %683 = vector.extract %283#0[1, 2] : f32 from vector<2x8xf32>
    %684 = vector.extract %283#0[1, 3] : f32 from vector<2x8xf32>
    %685 = arith.select %286, %681, %682 : f32
    %686 = llvm.bitcast %685 : f32 to i32
    %687 = rocdl.ds_bpermute %10, %686 : (i32, i32) -> i32
    %688 = llvm.bitcast %687 : i32 to f32
    %689 = arith.select %286, %688, %681 : f32
    %690 = arith.select %286, %682, %688 : f32
    %691 = arith.select %286, %683, %684 : f32
    %692 = llvm.bitcast %691 : f32 to i32
    %693 = rocdl.ds_bpermute %10, %692 : (i32, i32) -> i32
    %694 = llvm.bitcast %693 : i32 to f32
    %695 = arith.select %286, %694, %683 : f32
    %696 = arith.select %286, %684, %694 : f32
    %697 = arith.select %288, %689, %695 : f32
    %698 = llvm.bitcast %697 : f32 to i32
    %699 = rocdl.ds_bpermute %14, %698 : (i32, i32) -> i32
    %700 = llvm.bitcast %699 : i32 to f32
    %701 = arith.select %288, %700, %689 : f32
    %702 = arith.select %288, %695, %700 : f32
    %703 = arith.select %288, %690, %696 : f32
    %704 = llvm.bitcast %703 : f32 to i32
    %705 = rocdl.ds_bpermute %14, %704 : (i32, i32) -> i32
    %706 = llvm.bitcast %705 : i32 to f32
    %707 = arith.select %288, %706, %690 : f32
    %708 = arith.select %288, %696, %706 : f32
    %709 = vector.extract %283#0[1, 4] : f32 from vector<2x8xf32>
    %710 = vector.extract %283#0[1, 5] : f32 from vector<2x8xf32>
    %711 = vector.extract %283#0[1, 6] : f32 from vector<2x8xf32>
    %712 = vector.extract %283#0[1, 7] : f32 from vector<2x8xf32>
    %713 = arith.select %286, %709, %710 : f32
    %714 = llvm.bitcast %713 : f32 to i32
    %715 = rocdl.ds_bpermute %10, %714 : (i32, i32) -> i32
    %716 = llvm.bitcast %715 : i32 to f32
    %717 = arith.select %286, %716, %709 : f32
    %718 = arith.select %286, %710, %716 : f32
    %719 = arith.select %286, %711, %712 : f32
    %720 = llvm.bitcast %719 : f32 to i32
    %721 = rocdl.ds_bpermute %10, %720 : (i32, i32) -> i32
    %722 = llvm.bitcast %721 : i32 to f32
    %723 = arith.select %286, %722, %711 : f32
    %724 = arith.select %286, %712, %722 : f32
    %725 = arith.select %288, %717, %723 : f32
    %726 = llvm.bitcast %725 : f32 to i32
    %727 = rocdl.ds_bpermute %14, %726 : (i32, i32) -> i32
    %728 = llvm.bitcast %727 : i32 to f32
    %729 = arith.select %288, %728, %717 : f32
    %730 = arith.select %288, %723, %728 : f32
    %731 = arith.select %288, %718, %724 : f32
    %732 = llvm.bitcast %731 : f32 to i32
    %733 = rocdl.ds_bpermute %14, %732 : (i32, i32) -> i32
    %734 = llvm.bitcast %733 : i32 to f32
    %735 = arith.select %288, %734, %718 : f32
    %736 = arith.select %288, %724, %734 : f32
    %737 = vector.extract %283#1[0, 0] : f32 from vector<2x1xf32>
    %738 = vector.from_elements %645, %651, %646, %652 : vector<4xf32>
    %739 = vector.broadcast %737 : f32 to vector<4xf32>
    %740 = arith.divf %cst, %739 fastmath<arcp> : vector<4xf32>
    %741 = arith.mulf %738, %740 : vector<4xf32>
    %742 = arith.truncf %741 {tiled = true} : vector<4xf32> to vector<4xf16>
    %743 = vector.extract %742[0] : f16 from vector<4xf16>
    %744 = vector.extract %742[1] : f16 from vector<4xf16>
    %745 = vector.extract %742[2] : f16 from vector<4xf16>
    %746 = vector.extract %742[3] : f16 from vector<4xf16>
    %747 = vector.from_elements %673, %679, %674, %680 : vector<4xf32>
    %748 = arith.mulf %747, %740 : vector<4xf32>
    %749 = arith.truncf %748 {tiled = true} : vector<4xf32> to vector<4xf16>
    %750 = vector.extract %749[0] : f16 from vector<4xf16>
    %751 = vector.extract %749[1] : f16 from vector<4xf16>
    %752 = vector.extract %749[2] : f16 from vector<4xf16>
    %753 = vector.extract %749[3] : f16 from vector<4xf16>
    %754 = vector.extract %283#1[1, 0] : f32 from vector<2x1xf32>
    %755 = vector.from_elements %701, %707, %702, %708 : vector<4xf32>
    %756 = vector.broadcast %754 : f32 to vector<4xf32>
    %757 = arith.divf %cst, %756 fastmath<arcp> : vector<4xf32>
    %758 = arith.mulf %755, %757 : vector<4xf32>
    %759 = arith.truncf %758 {tiled = true} : vector<4xf32> to vector<4xf16>
    %760 = vector.extract %759[0] : f16 from vector<4xf16>
    %761 = vector.extract %759[1] : f16 from vector<4xf16>
    %762 = vector.extract %759[2] : f16 from vector<4xf16>
    %763 = vector.extract %759[3] : f16 from vector<4xf16>
    %764 = vector.from_elements %729, %735, %730, %736 : vector<4xf32>
    %765 = arith.mulf %764, %757 : vector<4xf32>
    %766 = arith.truncf %765 {tiled = true} : vector<4xf32> to vector<4xf16>
    %767 = vector.extract %766[0] : f16 from vector<4xf16>
    %768 = vector.extract %766[1] : f16 from vector<4xf16>
    %769 = vector.extract %766[2] : f16 from vector<4xf16>
    %770 = vector.extract %766[3] : f16 from vector<4xf16>
    %771 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %772 = arith.cmpi ult, %771, %c4 : index
    scf.if %772 {
      affine.store %743, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
      affine.store %744, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
      affine.store %745, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
      affine.store %746, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      affine.store %750, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 16] : memref<1x32x4096x128xf16, 1>
      affine.store %751, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 17] : memref<1x32x4096x128xf16, 1>
      affine.store %752, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 18] : memref<1x32x4096x128xf16, 1>
      affine.store %753, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 19] : memref<1x32x4096x128xf16, 1>
      affine.store %760, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
      affine.store %761, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
      affine.store %762, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
      affine.store %763, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      affine.store %767, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 16] : memref<1x32x4096x128xf16, 1>
      affine.store %768, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 17] : memref<1x32x4096x128xf16, 1>
      affine.store %769, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 18] : memref<1x32x4096x128xf16, 1>
      affine.store %770, %arg3[0, (symbol(%25) + symbol(%block_id_z) * 32 + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 19] : memref<1x32x4096x128xf16, 1>
    }
    %773 = vector.from_elements %533, %539, %534, %540 : vector<4xf32>
    %774 = arith.mulf %773, %740 : vector<4xf32>
    %775 = arith.truncf %774 {tiled = true} : vector<4xf32> to vector<4xf16>
    %776 = vector.extract %775[0] : f16 from vector<4xf16>
    %777 = vector.extract %775[1] : f16 from vector<4xf16>
    %778 = vector.extract %775[2] : f16 from vector<4xf16>
    %779 = vector.extract %775[3] : f16 from vector<4xf16>
    %780 = vector.from_elements %561, %567, %562, %568 : vector<4xf32>
    %781 = arith.mulf %780, %740 : vector<4xf32>
    %782 = arith.truncf %781 {tiled = true} : vector<4xf32> to vector<4xf16>
    %783 = vector.extract %782[0] : f16 from vector<4xf16>
    %784 = vector.extract %782[1] : f16 from vector<4xf16>
    %785 = vector.extract %782[2] : f16 from vector<4xf16>
    %786 = vector.extract %782[3] : f16 from vector<4xf16>
    %787 = vector.from_elements %589, %595, %590, %596 : vector<4xf32>
    %788 = arith.mulf %787, %757 : vector<4xf32>
    %789 = arith.truncf %788 {tiled = true} : vector<4xf32> to vector<4xf16>
    %790 = vector.extract %789[0] : f16 from vector<4xf16>
    %791 = vector.extract %789[1] : f16 from vector<4xf16>
    %792 = vector.extract %789[2] : f16 from vector<4xf16>
    %793 = vector.extract %789[3] : f16 from vector<4xf16>
    %794 = vector.from_elements %617, %623, %618, %624 : vector<4xf32>
    %795 = arith.mulf %794, %757 : vector<4xf32>
    %796 = arith.truncf %795 {tiled = true} : vector<4xf32> to vector<4xf16>
    %797 = vector.extract %796[0] : f16 from vector<4xf16>
    %798 = vector.extract %796[1] : f16 from vector<4xf16>
    %799 = vector.extract %796[2] : f16 from vector<4xf16>
    %800 = vector.extract %796[3] : f16 from vector<4xf16>
    scf.if %772 {
      affine.store %776, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
      affine.store %777, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
      affine.store %778, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
      affine.store %779, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      affine.store %783, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 48] : memref<1x32x4096x128xf16, 1>
      affine.store %784, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 49] : memref<1x32x4096x128xf16, 1>
      affine.store %785, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 50] : memref<1x32x4096x128xf16, 1>
      affine.store %786, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 51] : memref<1x32x4096x128xf16, 1>
      affine.store %790, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
      affine.store %791, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
      affine.store %792, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
      affine.store %793, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      affine.store %797, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 48] : memref<1x32x4096x128xf16, 1>
      affine.store %798, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 49] : memref<1x32x4096x128xf16, 1>
      affine.store %799, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 50] : memref<1x32x4096x128xf16, 1>
      affine.store %800, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 51] : memref<1x32x4096x128xf16, 1>
    }
    %801 = vector.from_elements %421, %427, %422, %428 : vector<4xf32>
    %802 = arith.mulf %801, %740 : vector<4xf32>
    %803 = arith.truncf %802 {tiled = true} : vector<4xf32> to vector<4xf16>
    %804 = vector.extract %803[0] : f16 from vector<4xf16>
    %805 = vector.extract %803[1] : f16 from vector<4xf16>
    %806 = vector.extract %803[2] : f16 from vector<4xf16>
    %807 = vector.extract %803[3] : f16 from vector<4xf16>
    %808 = vector.from_elements %449, %455, %450, %456 : vector<4xf32>
    %809 = arith.mulf %808, %740 : vector<4xf32>
    %810 = arith.truncf %809 {tiled = true} : vector<4xf32> to vector<4xf16>
    %811 = vector.extract %810[0] : f16 from vector<4xf16>
    %812 = vector.extract %810[1] : f16 from vector<4xf16>
    %813 = vector.extract %810[2] : f16 from vector<4xf16>
    %814 = vector.extract %810[3] : f16 from vector<4xf16>
    %815 = vector.from_elements %477, %483, %478, %484 : vector<4xf32>
    %816 = arith.mulf %815, %757 : vector<4xf32>
    %817 = arith.truncf %816 {tiled = true} : vector<4xf32> to vector<4xf16>
    %818 = vector.extract %817[0] : f16 from vector<4xf16>
    %819 = vector.extract %817[1] : f16 from vector<4xf16>
    %820 = vector.extract %817[2] : f16 from vector<4xf16>
    %821 = vector.extract %817[3] : f16 from vector<4xf16>
    %822 = vector.from_elements %505, %511, %506, %512 : vector<4xf32>
    %823 = arith.mulf %822, %757 : vector<4xf32>
    %824 = arith.truncf %823 {tiled = true} : vector<4xf32> to vector<4xf16>
    %825 = vector.extract %824[0] : f16 from vector<4xf16>
    %826 = vector.extract %824[1] : f16 from vector<4xf16>
    %827 = vector.extract %824[2] : f16 from vector<4xf16>
    %828 = vector.extract %824[3] : f16 from vector<4xf16>
    scf.if %772 {
      affine.store %804, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
      affine.store %805, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
      affine.store %806, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
      affine.store %807, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      affine.store %811, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 80] : memref<1x32x4096x128xf16, 1>
      affine.store %812, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 81] : memref<1x32x4096x128xf16, 1>
      affine.store %813, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 82] : memref<1x32x4096x128xf16, 1>
      affine.store %814, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 83] : memref<1x32x4096x128xf16, 1>
      affine.store %818, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
      affine.store %819, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
      affine.store %820, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
      affine.store %821, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      affine.store %825, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 80] : memref<1x32x4096x128xf16, 1>
      affine.store %826, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 81] : memref<1x32x4096x128xf16, 1>
      affine.store %827, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 82] : memref<1x32x4096x128xf16, 1>
      affine.store %828, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 83] : memref<1x32x4096x128xf16, 1>
    }
    %829 = vector.from_elements %309, %315, %310, %316 : vector<4xf32>
    %830 = arith.mulf %829, %740 : vector<4xf32>
    %831 = arith.truncf %830 {tiled = true} : vector<4xf32> to vector<4xf16>
    %832 = vector.extract %831[0] : f16 from vector<4xf16>
    %833 = vector.extract %831[1] : f16 from vector<4xf16>
    %834 = vector.extract %831[2] : f16 from vector<4xf16>
    %835 = vector.extract %831[3] : f16 from vector<4xf16>
    %836 = vector.from_elements %337, %343, %338, %344 : vector<4xf32>
    %837 = arith.mulf %836, %740 : vector<4xf32>
    %838 = arith.truncf %837 {tiled = true} : vector<4xf32> to vector<4xf16>
    %839 = vector.extract %838[0] : f16 from vector<4xf16>
    %840 = vector.extract %838[1] : f16 from vector<4xf16>
    %841 = vector.extract %838[2] : f16 from vector<4xf16>
    %842 = vector.extract %838[3] : f16 from vector<4xf16>
    %843 = vector.from_elements %365, %371, %366, %372 : vector<4xf32>
    %844 = arith.mulf %843, %757 : vector<4xf32>
    %845 = arith.truncf %844 {tiled = true} : vector<4xf32> to vector<4xf16>
    %846 = vector.extract %845[0] : f16 from vector<4xf16>
    %847 = vector.extract %845[1] : f16 from vector<4xf16>
    %848 = vector.extract %845[2] : f16 from vector<4xf16>
    %849 = vector.extract %845[3] : f16 from vector<4xf16>
    %850 = vector.from_elements %393, %399, %394, %400 : vector<4xf32>
    %851 = arith.mulf %850, %757 : vector<4xf32>
    %852 = arith.truncf %851 {tiled = true} : vector<4xf32> to vector<4xf16>
    %853 = vector.extract %852[0] : f16 from vector<4xf16>
    %854 = vector.extract %852[1] : f16 from vector<4xf16>
    %855 = vector.extract %852[2] : f16 from vector<4xf16>
    %856 = vector.extract %852[3] : f16 from vector<4xf16>
    scf.if %772 {
      affine.store %832, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
      affine.store %833, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
      affine.store %834, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
      affine.store %835, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      affine.store %839, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 112] : memref<1x32x4096x128xf16, 1>
      affine.store %840, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 113] : memref<1x32x4096x128xf16, 1>
      affine.store %841, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 114] : memref<1x32x4096x128xf16, 1>
      affine.store %842, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 115] : memref<1x32x4096x128xf16, 1>
      affine.store %846, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
      affine.store %847, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
      affine.store %848, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
      affine.store %849, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      affine.store %853, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 112] : memref<1x32x4096x128xf16, 1>
      affine.store %854, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 113] : memref<1x32x4096x128xf16, 1>
      affine.store %855, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 114] : memref<1x32x4096x128xf16, 1>
      affine.store %856, %arg3[0, (symbol(%25) + symbol(%24) floordiv 32) mod 32, symbol(%24) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%24) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 115] : memref<1x32x4096x128xf16, 1>
    }
    return
  }
}
