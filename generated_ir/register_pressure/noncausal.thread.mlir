module {
  memref.global "public" @shm_0 : memref<16384xi8, 3> = uninitialized alignment = 16
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "q"}, %arg1: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "k"}, %arg2: memref<1x32x4096x128xf16, 1> {frisk.shared_role = "v"}, %arg3: memref<1x32x4096x128xf16, 1>) attributes {block_layout = array<i64: 4, 1>, block_layout_order = array<i64: 0, 1>, frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, frisk.workgroup_order = array<i64: 32, 0, 1, 0>, gridDim = [1 : i32, 32 : i32, 32 : i32], thread_num = 256 : i32, warp_layout = array<i64: 16, 4>} {
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
    %c4 = arith.constant 4 : index
    %cst_6 = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %16 = memref.get_global @shm_0 : memref<16384xi8, 3>
    %block_id_x = gpu.block_id x {range = 32 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %17 = arith.muli %block_id_y, %c32 : index
    %18 = arith.addi %block_id_x, %17 : index
    %19 = arith.divui %18, %c1024 : index
    %20 = arith.muli %19, %c32 : index
    %21 = arith.remui %18, %c1024 : index
    %22 = arith.remui %21, %c32 : index
    %23 = arith.addi %20, %22 : index
    %24 = arith.divui %21, %c32 : index
    %block_id_z = gpu.block_id z {range = 1 : index}
    %25 = affine.apply affine_map<()[s0, s1, s2] -> ((s1 + s2 * 32 + s0 floordiv 32) mod 32)>()[%23, %24, %block_id_z]
    %26 = affine.apply affine_map<()[s0] -> (s0 * 128 - (s0 floordiv 32) * 4096)>()[%23]
    %subview = memref.subview %arg0[0, %25, %26, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %27 = builtin.unrealized_conversion_cast %subview : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %view = memref.view %16[%c0][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %28 = builtin.unrealized_conversion_cast %view : memref<4096xf16, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %thread_id_x = gpu.thread_id x
    %29 = arith.muli %thread_id_x, %c8 : index
    %30 = affine.apply affine_map<()[s0] -> ((s0 floordiv 8) mod 128)>()[%29]
    %31 = builtin.unrealized_conversion_cast %30 : index to i64
    %32 = affine.apply affine_map<()[s0] -> (s0 + (s0 floordiv 1024) * 8 - (s0 floordiv 8) * 8)>()[%29]
    %33 = builtin.unrealized_conversion_cast %32 : index to i64
    %34 = arith.divui %thread_id_x, %c64 : index
    %35 = arith.muli %34, %c512 : index
    %36 = arith.remui %35, %c4096 : index
    %37 = builtin.unrealized_conversion_cast %36 : index to i64
    %38 = llvm.extractvalue %27[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.extractvalue %27[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.getelementptr %38[%39] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %41 = llvm.mul %31, %15 : i64
    %42 = llvm.add %41, %33 : i64
    %43 = llvm.getelementptr %40[%42] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %44 = llvm.extractvalue %28[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.getelementptr %44[%37] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %46 = llvm.ptrtoint %45 : !llvm.ptr<3> to i32
    %47 = rocdl.readfirstlane %46 : i32
    %48 = llvm.inttoptr %47 : i32 to !llvm.ptr<3>
    rocdl.global.load.async.lds %43, %48, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    %49 = arith.addi %thread_id_x, %c256 : index
    %50 = arith.muli %49, %c8 : index
    %51 = affine.apply affine_map<()[s0] -> ((s0 floordiv 8) mod 128)>()[%50]
    %52 = builtin.unrealized_conversion_cast %51 : index to i64
    %53 = affine.apply affine_map<()[s0] -> (s0 + (s0 floordiv 1024) * 8 - (s0 floordiv 8) * 8)>()[%50]
    %54 = builtin.unrealized_conversion_cast %53 : index to i64
    %55 = arith.divui %49, %c64 : index
    %56 = arith.muli %55, %c512 : index
    %57 = arith.remui %56, %c4096 : index
    %58 = builtin.unrealized_conversion_cast %57 : index to i64
    %59 = llvm.mul %52, %15 : i64
    %60 = llvm.add %59, %54 : i64
    %61 = llvm.getelementptr %40[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %62 = llvm.getelementptr %44[%58] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %63 = llvm.ptrtoint %62 : !llvm.ptr<3> to i32
    %64 = rocdl.readfirstlane %63 : i32
    %65 = llvm.inttoptr %64 : i32 to !llvm.ptr<3>
    rocdl.global.load.async.lds %61, %65, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    rocdl.asyncmark
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %66 = arith.truncf %cst_2 : f32 to f16
    rocdl.wait.asyncmark 0
    gpu.barrier
    %view_7 = memref.view %16[%c8192][] : memref<16384xi8, 3> to memref<4096xf16, 3>
    %67 = builtin.unrealized_conversion_cast %view_7 : memref<4096xf16, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    %68 = affine.apply affine_map<()[s0, s1] -> ((s1 + s0 floordiv 32) mod 32)>()[%23, %24]
    %subview_8 = memref.subview %arg0[0, %68, %26, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %69 = builtin.unrealized_conversion_cast %subview_8 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.extractvalue %69[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.extractvalue %69[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.getelementptr %70[%71] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %73 = llvm.getelementptr %72[%42] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %74 = llvm.extractvalue %67[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.getelementptr %74[%37] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %76 = llvm.ptrtoint %75 : !llvm.ptr<3> to i32
    %77 = rocdl.readfirstlane %76 : i32
    %78 = llvm.inttoptr %77 : i32 to !llvm.ptr<3>
    rocdl.global.load.async.lds %73, %78, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    %79 = llvm.getelementptr %72[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %80 = llvm.getelementptr %74[%58] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    %81 = llvm.ptrtoint %80 : !llvm.ptr<3> to i32
    %82 = rocdl.readfirstlane %81 : i32
    %83 = llvm.inttoptr %82 : i32 to !llvm.ptr<3>
    rocdl.global.load.async.lds %79, %83, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    rocdl.asyncmark
    %84 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128)>()[%thread_id_x]
    %85 = vector.load %view[%84] : memref<4096xf16, 3>, vector<4xf16>
    %86 = vector.broadcast %66 : f16 to vector<4xf16>
    %87 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2048)>()[%thread_id_x]
    %88 = vector.load %view[%87] : memref<4096xf16, 3>, vector<4xf16>
    %89 = vector.extract %85[0] : f16 from vector<4xf16>
    %90 = vector.extract %85[1] : f16 from vector<4xf16>
    %91 = vector.extract %85[2] : f16 from vector<4xf16>
    %92 = vector.extract %85[3] : f16 from vector<4xf16>
    %93 = vector.extract %88[0] : f16 from vector<4xf16>
    %94 = vector.extract %88[1] : f16 from vector<4xf16>
    %95 = vector.extract %88[2] : f16 from vector<4xf16>
    %96 = vector.extract %88[3] : f16 from vector<4xf16>
    %97 = vector.from_elements %89, %90, %91, %92 : vector<4xf16>
    %98 = arith.mulf %97, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %99 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 512)>()[%thread_id_x]
    %100 = vector.load %view[%99] : memref<4096xf16, 3>, vector<4xf16>
    %101 = vector.from_elements %93, %94, %95, %96 : vector<4xf16>
    %102 = arith.mulf %101, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %103 = affine.apply affine_map<()[s0] -> (s0 * 8 + (s0 floordiv 64) * 128 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - ((s0 floordiv 64) floordiv 4) * 512 - (s0 floordiv 16) * 128 + 2560)>()[%thread_id_x]
    %104 = vector.load %view[%103] : memref<4096xf16, 3>, vector<4xf16>
    %105 = vector.extract %98[0] : f16 from vector<4xf16>
    %106 = vector.extract %100[0] : f16 from vector<4xf16>
    %107 = vector.extract %100[1] : f16 from vector<4xf16>
    %108 = vector.extract %100[2] : f16 from vector<4xf16>
    %109 = vector.extract %100[3] : f16 from vector<4xf16>
    %110 = vector.extract %104[0] : f16 from vector<4xf16>
    %111 = vector.extract %104[1] : f16 from vector<4xf16>
    %112 = vector.extract %104[2] : f16 from vector<4xf16>
    %113 = vector.extract %104[3] : f16 from vector<4xf16>
    %114 = vector.from_elements %106, %107, %108, %109 : vector<4xf16>
    %115 = arith.mulf %114, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %116 = vector.extract %98[1] : f16 from vector<4xf16>
    %117 = vector.extract %98[2] : f16 from vector<4xf16>
    %118 = vector.extract %98[3] : f16 from vector<4xf16>
    %119 = vector.extract %102[0] : f16 from vector<4xf16>
    %120 = vector.extract %102[1] : f16 from vector<4xf16>
    %121 = vector.extract %102[2] : f16 from vector<4xf16>
    %122 = vector.extract %102[3] : f16 from vector<4xf16>
    %123 = vector.extract %115[0] : f16 from vector<4xf16>
    %124 = vector.extract %115[1] : f16 from vector<4xf16>
    %125 = vector.extract %115[2] : f16 from vector<4xf16>
    %126 = vector.extract %115[3] : f16 from vector<4xf16>
    %127 = vector.from_elements %110, %111, %112, %113 : vector<4xf16>
    %128 = arith.mulf %127, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %129 = vector.extract %128[0] : f16 from vector<4xf16>
    %130 = vector.extract %128[1] : f16 from vector<4xf16>
    %131 = vector.extract %128[2] : f16 from vector<4xf16>
    %132 = vector.extract %128[3] : f16 from vector<4xf16>
    rocdl.wait.asyncmark 0
    gpu.barrier
    %subview_9 = memref.subview %arg0[0, %68, %26, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %133 = builtin.unrealized_conversion_cast %subview_9 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %134 = llvm.extractvalue %133[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.extractvalue %133[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.getelementptr %134[%135] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %137 = llvm.getelementptr %136[%42] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    rocdl.global.load.async.lds %137, %48, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    %138 = llvm.getelementptr %136[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    rocdl.global.load.async.lds %138, %65, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    rocdl.asyncmark
    %139 = vector.load %view_7[%84] : memref<4096xf16, 3>, vector<4xf16>
    %140 = vector.load %view_7[%87] : memref<4096xf16, 3>, vector<4xf16>
    %141 = vector.load %view_7[%99] : memref<4096xf16, 3>, vector<4xf16>
    %142 = vector.extract %139[0] : f16 from vector<4xf16>
    %143 = vector.extract %139[1] : f16 from vector<4xf16>
    %144 = vector.extract %139[2] : f16 from vector<4xf16>
    %145 = vector.extract %139[3] : f16 from vector<4xf16>
    %146 = vector.extract %140[0] : f16 from vector<4xf16>
    %147 = vector.extract %140[1] : f16 from vector<4xf16>
    %148 = vector.extract %140[2] : f16 from vector<4xf16>
    %149 = vector.extract %140[3] : f16 from vector<4xf16>
    %150 = vector.from_elements %142, %143, %144, %145 : vector<4xf16>
    %151 = arith.mulf %150, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %152 = vector.load %view_7[%103] : memref<4096xf16, 3>, vector<4xf16>
    %153 = vector.extract %141[0] : f16 from vector<4xf16>
    %154 = vector.extract %141[1] : f16 from vector<4xf16>
    %155 = vector.extract %141[2] : f16 from vector<4xf16>
    %156 = vector.extract %141[3] : f16 from vector<4xf16>
    %157 = vector.from_elements %146, %147, %148, %149 : vector<4xf16>
    %158 = arith.mulf %157, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %159 = vector.extract %151[0] : f16 from vector<4xf16>
    %160 = vector.extract %151[1] : f16 from vector<4xf16>
    %161 = vector.extract %151[2] : f16 from vector<4xf16>
    %162 = vector.extract %151[3] : f16 from vector<4xf16>
    %163 = vector.extract %158[0] : f16 from vector<4xf16>
    %164 = vector.extract %158[1] : f16 from vector<4xf16>
    %165 = vector.extract %158[2] : f16 from vector<4xf16>
    %166 = vector.extract %158[3] : f16 from vector<4xf16>
    %167 = vector.extract %152[0] : f16 from vector<4xf16>
    %168 = vector.extract %152[1] : f16 from vector<4xf16>
    %169 = vector.extract %152[2] : f16 from vector<4xf16>
    %170 = vector.extract %152[3] : f16 from vector<4xf16>
    %171 = vector.from_elements %153, %154, %155, %156 : vector<4xf16>
    %172 = arith.mulf %171, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %173 = vector.from_elements %167, %168, %169, %170 : vector<4xf16>
    %174 = arith.mulf %173, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %175 = vector.extract %172[0] : f16 from vector<4xf16>
    %176 = vector.extract %172[1] : f16 from vector<4xf16>
    %177 = vector.extract %172[2] : f16 from vector<4xf16>
    %178 = vector.extract %172[3] : f16 from vector<4xf16>
    %179 = vector.extract %174[0] : f16 from vector<4xf16>
    %180 = vector.extract %174[1] : f16 from vector<4xf16>
    %181 = vector.extract %174[2] : f16 from vector<4xf16>
    %182 = vector.extract %174[3] : f16 from vector<4xf16>
    rocdl.wait.asyncmark 0
    gpu.barrier
    %subview_10 = memref.subview %arg0[0, %68, %26, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
    %183 = builtin.unrealized_conversion_cast %subview_10 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    %184 = llvm.extractvalue %183[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.extractvalue %183[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
    %186 = llvm.getelementptr %184[%185] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %187 = llvm.getelementptr %186[%42] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    rocdl.global.load.async.lds %187, %78, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    %188 = llvm.getelementptr %186[%60] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    rocdl.global.load.async.lds %188, %83, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
    rocdl.asyncmark
    %189 = vector.load %view[%84] : memref<4096xf16, 3>, vector<4xf16>
    %190 = vector.load %view[%87] : memref<4096xf16, 3>, vector<4xf16>
    %191 = vector.load %view[%99] : memref<4096xf16, 3>, vector<4xf16>
    %192 = vector.extract %189[0] : f16 from vector<4xf16>
    %193 = vector.extract %189[1] : f16 from vector<4xf16>
    %194 = vector.extract %189[2] : f16 from vector<4xf16>
    %195 = vector.extract %189[3] : f16 from vector<4xf16>
    %196 = vector.extract %190[0] : f16 from vector<4xf16>
    %197 = vector.extract %190[1] : f16 from vector<4xf16>
    %198 = vector.extract %190[2] : f16 from vector<4xf16>
    %199 = vector.extract %190[3] : f16 from vector<4xf16>
    %200 = vector.from_elements %192, %193, %194, %195 : vector<4xf16>
    %201 = arith.mulf %200, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %202 = vector.load %view[%103] : memref<4096xf16, 3>, vector<4xf16>
    %203 = vector.extract %191[0] : f16 from vector<4xf16>
    %204 = vector.extract %191[1] : f16 from vector<4xf16>
    %205 = vector.extract %191[2] : f16 from vector<4xf16>
    %206 = vector.extract %191[3] : f16 from vector<4xf16>
    %207 = vector.from_elements %196, %197, %198, %199 : vector<4xf16>
    %208 = arith.mulf %207, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %209 = vector.extract %201[0] : f16 from vector<4xf16>
    %210 = vector.extract %201[1] : f16 from vector<4xf16>
    %211 = vector.extract %201[2] : f16 from vector<4xf16>
    %212 = vector.extract %201[3] : f16 from vector<4xf16>
    %213 = vector.extract %208[0] : f16 from vector<4xf16>
    %214 = vector.extract %208[1] : f16 from vector<4xf16>
    %215 = vector.extract %208[2] : f16 from vector<4xf16>
    %216 = vector.extract %208[3] : f16 from vector<4xf16>
    %217 = vector.extract %202[0] : f16 from vector<4xf16>
    %218 = vector.extract %202[1] : f16 from vector<4xf16>
    %219 = vector.extract %202[2] : f16 from vector<4xf16>
    %220 = vector.extract %202[3] : f16 from vector<4xf16>
    %221 = vector.from_elements %203, %204, %205, %206 : vector<4xf16>
    %222 = arith.mulf %221, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %223 = vector.from_elements %217, %218, %219, %220 : vector<4xf16>
    %224 = arith.mulf %223, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %225 = vector.extract %222[0] : f16 from vector<4xf16>
    %226 = vector.extract %222[1] : f16 from vector<4xf16>
    %227 = vector.extract %222[2] : f16 from vector<4xf16>
    %228 = vector.extract %222[3] : f16 from vector<4xf16>
    %229 = vector.extract %224[0] : f16 from vector<4xf16>
    %230 = vector.extract %224[1] : f16 from vector<4xf16>
    %231 = vector.extract %224[2] : f16 from vector<4xf16>
    %232 = vector.extract %224[3] : f16 from vector<4xf16>
    rocdl.wait.asyncmark 0
    gpu.barrier
    %233 = vector.load %view_7[%84] : memref<4096xf16, 3>, vector<4xf16>
    %234 = vector.load %view_7[%87] : memref<4096xf16, 3>, vector<4xf16>
    %235 = vector.load %view_7[%99] : memref<4096xf16, 3>, vector<4xf16>
    %236 = vector.extract %233[0] : f16 from vector<4xf16>
    %237 = vector.extract %233[1] : f16 from vector<4xf16>
    %238 = vector.extract %233[2] : f16 from vector<4xf16>
    %239 = vector.extract %233[3] : f16 from vector<4xf16>
    %240 = vector.extract %234[0] : f16 from vector<4xf16>
    %241 = vector.extract %234[1] : f16 from vector<4xf16>
    %242 = vector.extract %234[2] : f16 from vector<4xf16>
    %243 = vector.extract %234[3] : f16 from vector<4xf16>
    %244 = vector.from_elements %236, %237, %238, %239 : vector<4xf16>
    %245 = arith.mulf %244, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %246 = vector.load %view_7[%103] : memref<4096xf16, 3>, vector<4xf16>
    %247 = vector.extract %235[0] : f16 from vector<4xf16>
    %248 = vector.extract %235[1] : f16 from vector<4xf16>
    %249 = vector.extract %235[2] : f16 from vector<4xf16>
    %250 = vector.extract %235[3] : f16 from vector<4xf16>
    %251 = vector.from_elements %240, %241, %242, %243 : vector<4xf16>
    %252 = arith.mulf %251, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %253 = vector.extract %245[0] : f16 from vector<4xf16>
    %254 = vector.extract %245[1] : f16 from vector<4xf16>
    %255 = vector.extract %245[2] : f16 from vector<4xf16>
    %256 = vector.extract %245[3] : f16 from vector<4xf16>
    %257 = vector.extract %252[0] : f16 from vector<4xf16>
    %258 = vector.extract %252[1] : f16 from vector<4xf16>
    %259 = vector.extract %252[2] : f16 from vector<4xf16>
    %260 = vector.extract %252[3] : f16 from vector<4xf16>
    %261 = vector.extract %246[0] : f16 from vector<4xf16>
    %262 = vector.extract %246[1] : f16 from vector<4xf16>
    %263 = vector.extract %246[2] : f16 from vector<4xf16>
    %264 = vector.extract %246[3] : f16 from vector<4xf16>
    %265 = vector.from_elements %247, %248, %249, %250 : vector<4xf16>
    %266 = arith.mulf %265, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %267 = vector.from_elements %261, %262, %263, %264 : vector<4xf16>
    %268 = arith.mulf %267, %86 {origin = "frisk.mul", tiled = true} : vector<4xf16>
    %269 = vector.extract %266[0] : f16 from vector<4xf16>
    %270 = vector.extract %266[1] : f16 from vector<4xf16>
    %271 = vector.extract %266[2] : f16 from vector<4xf16>
    %272 = vector.extract %266[3] : f16 from vector<4xf16>
    %273 = vector.extract %268[0] : f16 from vector<4xf16>
    %274 = vector.extract %268[1] : f16 from vector<4xf16>
    %275 = vector.extract %268[2] : f16 from vector<4xf16>
    %276 = vector.extract %268[3] : f16 from vector<4xf16>
    gpu.barrier
    %277 = affine.apply affine_map<()[s0] -> (s0 mod 32)>()[%24]
    %278 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 4)>()[%thread_id_x]
    %279 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 5)>()[%thread_id_x]
    %280 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 6)>()[%thread_id_x]
    %281 = affine.apply affine_map<()[s0] -> (((s0 mod 16 + 16) floordiv 32) * -8 + 7)>()[%thread_id_x]
    %282:6 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %cst_5, %arg6 = %cst_4, %arg7 = %cst_1, %arg8 = %cst_5, %arg9 = %cst_5, %arg10 = %cst_5) -> (vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>) {
      %856 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      %subview_11 = memref.subview %transpose[0, %277, 0, %856] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %857 = builtin.unrealized_conversion_cast %subview_11 : memref<32x128xf16, strided<[1, 128], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %858 = affine.apply affine_map<()[s0] -> (s0 mod 8 + ((s0 + ((s0 floordiv 8) floordiv 16) * 32 - (((s0 floordiv 8) floordiv 16) floordiv 2) * 64 - ((((s0 floordiv 8) floordiv 16) mod 2 + ((s0 floordiv 8) floordiv 4) mod 2) floordiv 2) * 64) floordiv 1024) * 8)>()[%29]
      %859 = builtin.unrealized_conversion_cast %858 : index to i64
      %860 = affine.apply affine_map<()[s0] -> ((s0 floordiv 8 + ((s0 floordiv 8) floordiv 16) * 4 - (((s0 floordiv 8) floordiv 16) floordiv 2) * 8 - ((((s0 floordiv 8) floordiv 16) mod 2 + ((s0 floordiv 8) floordiv 4) mod 2) floordiv 2) * 8) mod 128)>()[%29]
      %861 = builtin.unrealized_conversion_cast %860 : index to i64
      %862 = llvm.extractvalue %857[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %863 = llvm.extractvalue %857[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %864 = llvm.getelementptr %862[%863] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %865 = llvm.mul %861, %15 : i64
      %866 = llvm.add %859, %865 : i64
      %867 = llvm.getelementptr %864[%866] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %867, %48, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %868 = affine.apply affine_map<()[s0] -> (s0 mod 8 + ((s0 + ((s0 floordiv 8) floordiv 16) * 32 - (((s0 floordiv 8) floordiv 16) floordiv 2) * 64 - ((((s0 floordiv 8) floordiv 16) mod 2 + ((s0 floordiv 8) floordiv 4) mod 2) floordiv 2) * 64) floordiv 1024) * 8)>()[%50]
      %869 = builtin.unrealized_conversion_cast %868 : index to i64
      %870 = affine.apply affine_map<()[s0] -> ((s0 floordiv 8 + ((s0 floordiv 8) floordiv 16) * 4 - (((s0 floordiv 8) floordiv 16) floordiv 2) * 8 - ((((s0 floordiv 8) floordiv 16) mod 2 + ((s0 floordiv 8) floordiv 4) mod 2) floordiv 2) * 8) mod 128)>()[%50]
      %871 = builtin.unrealized_conversion_cast %870 : index to i64
      %872 = llvm.mul %871, %15 : i64
      %873 = llvm.add %869, %872 : i64
      %874 = llvm.getelementptr %864[%873] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %874, %65, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      rocdl.wait.asyncmark 0
      gpu.barrier
      %subview_12 = memref.subview %transpose[0, %277, 32, %856] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %875 = builtin.unrealized_conversion_cast %subview_12 : memref<32x128xf16, strided<[1, 128], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %876 = llvm.extractvalue %875[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %877 = llvm.extractvalue %875[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %878 = llvm.getelementptr %876[%877] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %879 = llvm.getelementptr %878[%866] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %879, %78, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %880 = llvm.getelementptr %878[%873] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %880, %83, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %881 = affine.apply affine_map<()[s0] -> (s0 * 160 - (s0 floordiv 4) * 640 + ((s0 mod 16) floordiv 4) * 8 - ((s0 mod 4) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016)>()[%thread_id_x]
      %882 = vector.load %view[%881] : memref<4096xf16, 3>, vector<4xf16>
      %883 = vector.from_elements %105, %116, %117, %118 : vector<4xf16>
      %884 = vector.from_elements %119, %120, %121, %122 : vector<4xf16>
      %885 = vector.extract %882[0] : f16 from vector<4xf16>
      %886 = vector.extract %882[1] : f16 from vector<4xf16>
      %887 = vector.extract %882[2] : f16 from vector<4xf16>
      %888 = vector.extract %882[3] : f16 from vector<4xf16>
      %889 = vector.from_elements %885, %886, %887, %888 : vector<4xf16>
      %890 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %883, %889, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %891 = vector.from_elements %123, %124, %125, %126 : vector<4xf16>
      %892 = vector.from_elements %129, %130, %131, %132 : vector<4xf16>
      %893 = affine.apply affine_map<()[s0] -> (s0 * 160 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 - (s0 floordiv 2) * 64 + 2048)>()[%thread_id_x]
      %894 = vector.load %view[%893] : memref<4096xf16, 3>, vector<4xf16>
      %895 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 16) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 16) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 32)>()[%thread_id_x]
      %896 = vector.load %view[%895] : memref<4096xf16, 3>, vector<4xf16>
      %897 = vector.extract %894[0] : f16 from vector<4xf16>
      %898 = vector.extract %894[1] : f16 from vector<4xf16>
      %899 = vector.extract %894[2] : f16 from vector<4xf16>
      %900 = vector.extract %894[3] : f16 from vector<4xf16>
      %901 = vector.from_elements %897, %898, %899, %900 : vector<4xf16>
      %902 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %884, %901, %890 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %903 = vector.extract %896[0] : f16 from vector<4xf16>
      %904 = vector.extract %896[1] : f16 from vector<4xf16>
      %905 = vector.extract %896[2] : f16 from vector<4xf16>
      %906 = vector.extract %896[3] : f16 from vector<4xf16>
      %907 = vector.from_elements %903, %904, %905, %906 : vector<4xf16>
      %908 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %883, %907, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %909 = vector.extract %902[0] : f32 from vector<4xf32>
      %910 = vector.extract %902[1] : f32 from vector<4xf32>
      %911 = vector.extract %902[2] : f32 from vector<4xf32>
      %912 = vector.extract %902[3] : f32 from vector<4xf32>
      %913 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 16) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 16) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 16) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + 2080)>()[%thread_id_x]
      %914 = vector.load %view[%913] : memref<4096xf16, 3>, vector<4xf16>
      %915 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 32) floordiv 64) * 512 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 64)>()[%thread_id_x]
      %916 = vector.load %view[%915] : memref<4096xf16, 3>, vector<4xf16>
      %917 = vector.extract %914[0] : f16 from vector<4xf16>
      %918 = vector.extract %914[1] : f16 from vector<4xf16>
      %919 = vector.extract %914[2] : f16 from vector<4xf16>
      %920 = vector.extract %914[3] : f16 from vector<4xf16>
      %921 = vector.from_elements %917, %918, %919, %920 : vector<4xf16>
      %922 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %884, %921, %908 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %923 = vector.extract %916[0] : f16 from vector<4xf16>
      %924 = vector.extract %916[1] : f16 from vector<4xf16>
      %925 = vector.extract %916[2] : f16 from vector<4xf16>
      %926 = vector.extract %916[3] : f16 from vector<4xf16>
      %927 = vector.from_elements %923, %924, %925, %926 : vector<4xf16>
      %928 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %883, %927, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %929 = vector.extract %922[0] : f32 from vector<4xf32>
      %930 = vector.extract %922[1] : f32 from vector<4xf32>
      %931 = vector.extract %922[2] : f32 from vector<4xf32>
      %932 = vector.extract %922[3] : f32 from vector<4xf32>
      %933 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 32) floordiv 64) * 512 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 32) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + 2112)>()[%thread_id_x]
      %934 = vector.load %view[%933] : memref<4096xf16, 3>, vector<4xf16>
      %935 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 48) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 48) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 160)>()[%thread_id_x]
      %936 = vector.load %view[%935] : memref<4096xf16, 3>, vector<4xf16>
      %937 = vector.extract %934[0] : f16 from vector<4xf16>
      %938 = vector.extract %934[1] : f16 from vector<4xf16>
      %939 = vector.extract %934[2] : f16 from vector<4xf16>
      %940 = vector.extract %934[3] : f16 from vector<4xf16>
      %941 = vector.from_elements %937, %938, %939, %940 : vector<4xf16>
      %942 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %884, %941, %928 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %943 = vector.extract %936[0] : f16 from vector<4xf16>
      %944 = vector.extract %936[1] : f16 from vector<4xf16>
      %945 = vector.extract %936[2] : f16 from vector<4xf16>
      %946 = vector.extract %936[3] : f16 from vector<4xf16>
      %947 = vector.from_elements %943, %944, %945, %946 : vector<4xf16>
      %948 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %883, %947, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %949 = vector.extract %942[0] : f32 from vector<4xf32>
      %950 = vector.extract %942[1] : f32 from vector<4xf32>
      %951 = vector.extract %942[2] : f32 from vector<4xf32>
      %952 = vector.extract %942[3] : f32 from vector<4xf32>
      %953 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 48) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 48) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 48) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + 2208)>()[%thread_id_x]
      %954 = vector.load %view[%953] : memref<4096xf16, 3>, vector<4xf16>
      %955 = affine.apply affine_map<()[s0] -> (s0 * 160 - (s0 floordiv 4) * 640 + ((s0 mod 16) floordiv 4) * 8 - ((s0 mod 4) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 512)>()[%thread_id_x]
      %956 = vector.load %view[%955] : memref<4096xf16, 3>, vector<4xf16>
      %957 = vector.extract %954[0] : f16 from vector<4xf16>
      %958 = vector.extract %954[1] : f16 from vector<4xf16>
      %959 = vector.extract %954[2] : f16 from vector<4xf16>
      %960 = vector.extract %954[3] : f16 from vector<4xf16>
      %961 = vector.from_elements %957, %958, %959, %960 : vector<4xf16>
      %962 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %884, %961, %948 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %963 = vector.extract %956[0] : f16 from vector<4xf16>
      %964 = vector.extract %956[1] : f16 from vector<4xf16>
      %965 = vector.extract %956[2] : f16 from vector<4xf16>
      %966 = vector.extract %956[3] : f16 from vector<4xf16>
      %967 = vector.from_elements %963, %964, %965, %966 : vector<4xf16>
      %968 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %883, %967, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %969 = vector.extract %962[0] : f32 from vector<4xf32>
      %970 = vector.extract %962[1] : f32 from vector<4xf32>
      %971 = vector.extract %962[2] : f32 from vector<4xf32>
      %972 = vector.extract %962[3] : f32 from vector<4xf32>
      %973 = affine.apply affine_map<()[s0] -> (s0 * 160 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 - (s0 floordiv 2) * 64 + 2560)>()[%thread_id_x]
      %974 = vector.load %view[%973] : memref<4096xf16, 3>, vector<4xf16>
      %975 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 80) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 80) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 32)>()[%thread_id_x]
      %976 = vector.load %view[%975] : memref<4096xf16, 3>, vector<4xf16>
      %977 = vector.extract %974[0] : f16 from vector<4xf16>
      %978 = vector.extract %974[1] : f16 from vector<4xf16>
      %979 = vector.extract %974[2] : f16 from vector<4xf16>
      %980 = vector.extract %974[3] : f16 from vector<4xf16>
      %981 = vector.from_elements %977, %978, %979, %980 : vector<4xf16>
      %982 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %884, %981, %968 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %983 = vector.extract %976[0] : f16 from vector<4xf16>
      %984 = vector.extract %976[1] : f16 from vector<4xf16>
      %985 = vector.extract %976[2] : f16 from vector<4xf16>
      %986 = vector.extract %976[3] : f16 from vector<4xf16>
      %987 = vector.from_elements %983, %984, %985, %986 : vector<4xf16>
      %988 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %883, %987, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %989 = vector.extract %982[0] : f32 from vector<4xf32>
      %990 = vector.extract %982[1] : f32 from vector<4xf32>
      %991 = vector.extract %982[2] : f32 from vector<4xf32>
      %992 = vector.extract %982[3] : f32 from vector<4xf32>
      %993 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 80) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 80) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 80) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 4) floordiv 16) mod 2 + 1) floordiv 2) * 64 + 2080)>()[%thread_id_x]
      %994 = vector.load %view[%993] : memref<4096xf16, 3>, vector<4xf16>
      %995 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 96) floordiv 64) * 512 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 64)>()[%thread_id_x]
      %996 = vector.load %view[%995] : memref<4096xf16, 3>, vector<4xf16>
      %997 = vector.extract %994[0] : f16 from vector<4xf16>
      %998 = vector.extract %994[1] : f16 from vector<4xf16>
      %999 = vector.extract %994[2] : f16 from vector<4xf16>
      %1000 = vector.extract %994[3] : f16 from vector<4xf16>
      %1001 = vector.from_elements %997, %998, %999, %1000 : vector<4xf16>
      %1002 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %884, %1001, %988 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1003 = vector.extract %996[0] : f16 from vector<4xf16>
      %1004 = vector.extract %996[1] : f16 from vector<4xf16>
      %1005 = vector.extract %996[2] : f16 from vector<4xf16>
      %1006 = vector.extract %996[3] : f16 from vector<4xf16>
      %1007 = vector.from_elements %1003, %1004, %1005, %1006 : vector<4xf16>
      %1008 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %883, %1007, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1009 = vector.extract %1002[0] : f32 from vector<4xf32>
      %1010 = vector.extract %1002[1] : f32 from vector<4xf32>
      %1011 = vector.extract %1002[2] : f32 from vector<4xf32>
      %1012 = vector.extract %1002[3] : f32 from vector<4xf32>
      %1013 = affine.apply affine_map<()[s0] -> (s0 * 128 + ((s0 mod 16 + 96) floordiv 64) * 512 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 - (s0 floordiv 4) * 512 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 96) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 8) floordiv 16) floordiv 2) * 64 + 2112)>()[%thread_id_x]
      %1014 = vector.load %view[%1013] : memref<4096xf16, 3>, vector<4xf16>
      %1015 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 112) floordiv 64) * 1024 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 112) floordiv 64) * 16 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 1016 + 160)>()[%thread_id_x]
      %1016 = vector.load %view[%1015] : memref<4096xf16, 3>, vector<4xf16>
      %1017 = vector.extract %1014[0] : f16 from vector<4xf16>
      %1018 = vector.extract %1014[1] : f16 from vector<4xf16>
      %1019 = vector.extract %1014[2] : f16 from vector<4xf16>
      %1020 = vector.extract %1014[3] : f16 from vector<4xf16>
      %1021 = vector.from_elements %1017, %1018, %1019, %1020 : vector<4xf16>
      %1022 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %884, %1021, %1008 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1023 = vector.extract %1016[0] : f16 from vector<4xf16>
      %1024 = vector.extract %1016[1] : f16 from vector<4xf16>
      %1025 = vector.extract %1016[2] : f16 from vector<4xf16>
      %1026 = vector.extract %1016[3] : f16 from vector<4xf16>
      %1027 = vector.from_elements %1023, %1024, %1025, %1026 : vector<4xf16>
      %1028 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %883, %1027, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1029 = vector.extract %1022[0] : f32 from vector<4xf32>
      %1030 = vector.extract %1022[1] : f32 from vector<4xf32>
      %1031 = vector.extract %1022[2] : f32 from vector<4xf32>
      %1032 = vector.extract %1022[3] : f32 from vector<4xf32>
      %1033 = affine.apply affine_map<()[s0] -> (s0 * 256 + ((s0 mod 16 + 112) floordiv 64) * 1024 + ((s0 mod 64) floordiv 16) * 4 + (((s0 mod 64) floordiv 16) floordiv 2) * 2040 - (s0 floordiv 4) * 1024 + ((s0 mod 16) floordiv 4) * 8 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) * 32 - (((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) floordiv 2) * 64 - ((s0 * 4 + ((s0 mod 16 + 112) floordiv 64) * 16 + (((s0 mod 64) floordiv 16) floordiv 2) * 32 - (s0 floordiv 4) * 16 + ((s0 * 16 + ((s0 mod 16 + 112) floordiv 64) * 64 + (((s0 mod 64) floordiv 16) floordiv 2) * 128 - (s0 floordiv 4) * 64 + (s0 mod 16) floordiv 4 + 12) floordiv 16) mod 2 + 3) floordiv 2) * 64 + 2208)>()[%thread_id_x]
      %1034 = vector.load %view[%1033] : memref<4096xf16, 3>, vector<4xf16>
      %1035 = vector.load %view[%881] : memref<4096xf16, 3>, vector<4xf16>
      %1036 = vector.extract %1034[0] : f16 from vector<4xf16>
      %1037 = vector.extract %1034[1] : f16 from vector<4xf16>
      %1038 = vector.extract %1034[2] : f16 from vector<4xf16>
      %1039 = vector.extract %1034[3] : f16 from vector<4xf16>
      %1040 = vector.from_elements %1036, %1037, %1038, %1039 : vector<4xf16>
      %1041 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %884, %1040, %1028 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1042 = vector.extract %1035[0] : f16 from vector<4xf16>
      %1043 = vector.extract %1035[1] : f16 from vector<4xf16>
      %1044 = vector.extract %1035[2] : f16 from vector<4xf16>
      %1045 = vector.extract %1035[3] : f16 from vector<4xf16>
      %1046 = vector.from_elements %1042, %1043, %1044, %1045 : vector<4xf16>
      %1047 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %891, %1046, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1048 = vector.extract %1041[0] : f32 from vector<4xf32>
      %1049 = vector.extract %1041[1] : f32 from vector<4xf32>
      %1050 = vector.extract %1041[2] : f32 from vector<4xf32>
      %1051 = vector.extract %1041[3] : f32 from vector<4xf32>
      %1052 = vector.load %view[%893] : memref<4096xf16, 3>, vector<4xf16>
      %1053 = vector.load %view[%895] : memref<4096xf16, 3>, vector<4xf16>
      %1054 = vector.extract %1052[0] : f16 from vector<4xf16>
      %1055 = vector.extract %1052[1] : f16 from vector<4xf16>
      %1056 = vector.extract %1052[2] : f16 from vector<4xf16>
      %1057 = vector.extract %1052[3] : f16 from vector<4xf16>
      %1058 = vector.from_elements %1054, %1055, %1056, %1057 : vector<4xf16>
      %1059 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %892, %1058, %1047 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1060 = vector.extract %1053[0] : f16 from vector<4xf16>
      %1061 = vector.extract %1053[1] : f16 from vector<4xf16>
      %1062 = vector.extract %1053[2] : f16 from vector<4xf16>
      %1063 = vector.extract %1053[3] : f16 from vector<4xf16>
      %1064 = vector.from_elements %1060, %1061, %1062, %1063 : vector<4xf16>
      %1065 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %891, %1064, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1066 = vector.extract %1059[0] : f32 from vector<4xf32>
      %1067 = vector.extract %1059[1] : f32 from vector<4xf32>
      %1068 = vector.extract %1059[2] : f32 from vector<4xf32>
      %1069 = vector.extract %1059[3] : f32 from vector<4xf32>
      %1070 = vector.load %view[%913] : memref<4096xf16, 3>, vector<4xf16>
      %1071 = vector.load %view[%915] : memref<4096xf16, 3>, vector<4xf16>
      %1072 = vector.extract %1070[0] : f16 from vector<4xf16>
      %1073 = vector.extract %1070[1] : f16 from vector<4xf16>
      %1074 = vector.extract %1070[2] : f16 from vector<4xf16>
      %1075 = vector.extract %1070[3] : f16 from vector<4xf16>
      %1076 = vector.from_elements %1072, %1073, %1074, %1075 : vector<4xf16>
      %1077 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %892, %1076, %1065 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1078 = vector.extract %1071[0] : f16 from vector<4xf16>
      %1079 = vector.extract %1071[1] : f16 from vector<4xf16>
      %1080 = vector.extract %1071[2] : f16 from vector<4xf16>
      %1081 = vector.extract %1071[3] : f16 from vector<4xf16>
      %1082 = vector.from_elements %1078, %1079, %1080, %1081 : vector<4xf16>
      %1083 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %891, %1082, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1084 = vector.extract %1077[0] : f32 from vector<4xf32>
      %1085 = vector.extract %1077[1] : f32 from vector<4xf32>
      %1086 = vector.extract %1077[2] : f32 from vector<4xf32>
      %1087 = vector.extract %1077[3] : f32 from vector<4xf32>
      %1088 = vector.load %view[%933] : memref<4096xf16, 3>, vector<4xf16>
      %1089 = vector.load %view[%935] : memref<4096xf16, 3>, vector<4xf16>
      %1090 = vector.extract %1088[0] : f16 from vector<4xf16>
      %1091 = vector.extract %1088[1] : f16 from vector<4xf16>
      %1092 = vector.extract %1088[2] : f16 from vector<4xf16>
      %1093 = vector.extract %1088[3] : f16 from vector<4xf16>
      %1094 = vector.from_elements %1090, %1091, %1092, %1093 : vector<4xf16>
      %1095 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %892, %1094, %1083 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1096 = vector.extract %1089[0] : f16 from vector<4xf16>
      %1097 = vector.extract %1089[1] : f16 from vector<4xf16>
      %1098 = vector.extract %1089[2] : f16 from vector<4xf16>
      %1099 = vector.extract %1089[3] : f16 from vector<4xf16>
      %1100 = vector.from_elements %1096, %1097, %1098, %1099 : vector<4xf16>
      %1101 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %891, %1100, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1102 = vector.extract %1095[0] : f32 from vector<4xf32>
      %1103 = vector.extract %1095[1] : f32 from vector<4xf32>
      %1104 = vector.extract %1095[2] : f32 from vector<4xf32>
      %1105 = vector.extract %1095[3] : f32 from vector<4xf32>
      %1106 = vector.load %view[%953] : memref<4096xf16, 3>, vector<4xf16>
      %1107 = vector.load %view[%955] : memref<4096xf16, 3>, vector<4xf16>
      %1108 = vector.extract %1106[0] : f16 from vector<4xf16>
      %1109 = vector.extract %1106[1] : f16 from vector<4xf16>
      %1110 = vector.extract %1106[2] : f16 from vector<4xf16>
      %1111 = vector.extract %1106[3] : f16 from vector<4xf16>
      %1112 = vector.from_elements %1108, %1109, %1110, %1111 : vector<4xf16>
      %1113 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %892, %1112, %1101 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1114 = vector.extract %1107[0] : f16 from vector<4xf16>
      %1115 = vector.extract %1107[1] : f16 from vector<4xf16>
      %1116 = vector.extract %1107[2] : f16 from vector<4xf16>
      %1117 = vector.extract %1107[3] : f16 from vector<4xf16>
      %1118 = vector.from_elements %1114, %1115, %1116, %1117 : vector<4xf16>
      %1119 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %891, %1118, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1120 = vector.extract %1113[0] : f32 from vector<4xf32>
      %1121 = vector.extract %1113[1] : f32 from vector<4xf32>
      %1122 = vector.extract %1113[2] : f32 from vector<4xf32>
      %1123 = vector.extract %1113[3] : f32 from vector<4xf32>
      %1124 = vector.load %view[%973] : memref<4096xf16, 3>, vector<4xf16>
      %1125 = vector.load %view[%975] : memref<4096xf16, 3>, vector<4xf16>
      %1126 = vector.extract %1124[0] : f16 from vector<4xf16>
      %1127 = vector.extract %1124[1] : f16 from vector<4xf16>
      %1128 = vector.extract %1124[2] : f16 from vector<4xf16>
      %1129 = vector.extract %1124[3] : f16 from vector<4xf16>
      %1130 = vector.from_elements %1126, %1127, %1128, %1129 : vector<4xf16>
      %1131 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %892, %1130, %1119 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1132 = vector.extract %1125[0] : f16 from vector<4xf16>
      %1133 = vector.extract %1125[1] : f16 from vector<4xf16>
      %1134 = vector.extract %1125[2] : f16 from vector<4xf16>
      %1135 = vector.extract %1125[3] : f16 from vector<4xf16>
      %1136 = vector.from_elements %1132, %1133, %1134, %1135 : vector<4xf16>
      %1137 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %891, %1136, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1138 = vector.extract %1131[0] : f32 from vector<4xf32>
      %1139 = vector.extract %1131[1] : f32 from vector<4xf32>
      %1140 = vector.extract %1131[2] : f32 from vector<4xf32>
      %1141 = vector.extract %1131[3] : f32 from vector<4xf32>
      %1142 = vector.load %view[%993] : memref<4096xf16, 3>, vector<4xf16>
      %1143 = vector.load %view[%995] : memref<4096xf16, 3>, vector<4xf16>
      %1144 = vector.extract %1142[0] : f16 from vector<4xf16>
      %1145 = vector.extract %1142[1] : f16 from vector<4xf16>
      %1146 = vector.extract %1142[2] : f16 from vector<4xf16>
      %1147 = vector.extract %1142[3] : f16 from vector<4xf16>
      %1148 = vector.from_elements %1144, %1145, %1146, %1147 : vector<4xf16>
      %1149 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %892, %1148, %1137 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1150 = vector.extract %1143[0] : f16 from vector<4xf16>
      %1151 = vector.extract %1143[1] : f16 from vector<4xf16>
      %1152 = vector.extract %1143[2] : f16 from vector<4xf16>
      %1153 = vector.extract %1143[3] : f16 from vector<4xf16>
      %1154 = vector.from_elements %1150, %1151, %1152, %1153 : vector<4xf16>
      %1155 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %891, %1154, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1156 = vector.extract %1149[0] : f32 from vector<4xf32>
      %1157 = vector.extract %1149[1] : f32 from vector<4xf32>
      %1158 = vector.extract %1149[2] : f32 from vector<4xf32>
      %1159 = vector.extract %1149[3] : f32 from vector<4xf32>
      %1160 = vector.load %view[%1013] : memref<4096xf16, 3>, vector<4xf16>
      %1161 = vector.load %view[%1015] : memref<4096xf16, 3>, vector<4xf16>
      %1162 = vector.extract %1160[0] : f16 from vector<4xf16>
      %1163 = vector.extract %1160[1] : f16 from vector<4xf16>
      %1164 = vector.extract %1160[2] : f16 from vector<4xf16>
      %1165 = vector.extract %1160[3] : f16 from vector<4xf16>
      %1166 = vector.from_elements %1162, %1163, %1164, %1165 : vector<4xf16>
      %1167 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %892, %1166, %1155 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1168 = vector.extract %1161[0] : f16 from vector<4xf16>
      %1169 = vector.extract %1161[1] : f16 from vector<4xf16>
      %1170 = vector.extract %1161[2] : f16 from vector<4xf16>
      %1171 = vector.extract %1161[3] : f16 from vector<4xf16>
      %1172 = vector.from_elements %1168, %1169, %1170, %1171 : vector<4xf16>
      %1173 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %891, %1172, %cst_0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1174 = vector.extract %1167[0] : f32 from vector<4xf32>
      %1175 = vector.extract %1167[1] : f32 from vector<4xf32>
      %1176 = vector.extract %1167[2] : f32 from vector<4xf32>
      %1177 = vector.extract %1167[3] : f32 from vector<4xf32>
      %1178 = vector.load %view[%1033] : memref<4096xf16, 3>, vector<4xf16>
      %1179 = vector.extract %1178[0] : f16 from vector<4xf16>
      %1180 = vector.extract %1178[1] : f16 from vector<4xf16>
      %1181 = vector.extract %1178[2] : f16 from vector<4xf16>
      %1182 = vector.extract %1178[3] : f16 from vector<4xf16>
      %1183 = vector.from_elements %1179, %1180, %1181, %1182 : vector<4xf16>
      %1184 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %892, %1183, %1173 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1185 = vector.extract %1184[0] : f32 from vector<4xf32>
      %1186 = vector.extract %1184[1] : f32 from vector<4xf32>
      %1187 = vector.extract %1184[2] : f32 from vector<4xf32>
      %1188 = vector.extract %1184[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %subview_13 = memref.subview %transpose[0, %277, 64, %856] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %1189 = builtin.unrealized_conversion_cast %subview_13 : memref<32x128xf16, strided<[1, 128], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %1190 = llvm.extractvalue %1189[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1191 = llvm.extractvalue %1189[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1192 = llvm.getelementptr %1190[%1191] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %1193 = llvm.getelementptr %1192[%866] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1193, %48, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %1194 = llvm.getelementptr %1192[%873] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1194, %65, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %1195 = vector.load %view_7[%881] : memref<4096xf16, 3>, vector<4xf16>
      %1196 = vector.from_elements %159, %160, %161, %162 : vector<4xf16>
      %1197 = vector.from_elements %909, %910, %911, %912 : vector<4xf32>
      %1198 = vector.from_elements %163, %164, %165, %166 : vector<4xf16>
      %1199 = vector.from_elements %929, %930, %931, %932 : vector<4xf32>
      %1200 = vector.extract %1195[0] : f16 from vector<4xf16>
      %1201 = vector.extract %1195[1] : f16 from vector<4xf16>
      %1202 = vector.extract %1195[2] : f16 from vector<4xf16>
      %1203 = vector.extract %1195[3] : f16 from vector<4xf16>
      %1204 = vector.from_elements %1200, %1201, %1202, %1203 : vector<4xf16>
      %1205 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1196, %1204, %1197 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1206 = vector.from_elements %949, %950, %951, %952 : vector<4xf32>
      %1207 = vector.from_elements %969, %970, %971, %972 : vector<4xf32>
      %1208 = vector.from_elements %989, %990, %991, %992 : vector<4xf32>
      %1209 = vector.from_elements %1009, %1010, %1011, %1012 : vector<4xf32>
      %1210 = vector.from_elements %1029, %1030, %1031, %1032 : vector<4xf32>
      %1211 = vector.from_elements %1048, %1049, %1050, %1051 : vector<4xf32>
      %1212 = vector.from_elements %175, %176, %177, %178 : vector<4xf16>
      %1213 = vector.from_elements %1066, %1067, %1068, %1069 : vector<4xf32>
      %1214 = vector.from_elements %179, %180, %181, %182 : vector<4xf16>
      %1215 = vector.from_elements %1084, %1085, %1086, %1087 : vector<4xf32>
      %1216 = vector.from_elements %1102, %1103, %1104, %1105 : vector<4xf32>
      %1217 = vector.from_elements %1120, %1121, %1122, %1123 : vector<4xf32>
      %1218 = vector.from_elements %1138, %1139, %1140, %1141 : vector<4xf32>
      %1219 = vector.from_elements %1156, %1157, %1158, %1159 : vector<4xf32>
      %1220 = vector.from_elements %1174, %1175, %1176, %1177 : vector<4xf32>
      %1221 = vector.from_elements %1185, %1186, %1187, %1188 : vector<4xf32>
      %1222 = vector.load %view_7[%893] : memref<4096xf16, 3>, vector<4xf16>
      %1223 = vector.load %view_7[%895] : memref<4096xf16, 3>, vector<4xf16>
      %1224 = vector.extract %1222[0] : f16 from vector<4xf16>
      %1225 = vector.extract %1222[1] : f16 from vector<4xf16>
      %1226 = vector.extract %1222[2] : f16 from vector<4xf16>
      %1227 = vector.extract %1222[3] : f16 from vector<4xf16>
      %1228 = vector.from_elements %1224, %1225, %1226, %1227 : vector<4xf16>
      %1229 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1198, %1228, %1205 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1230 = vector.extract %1223[0] : f16 from vector<4xf16>
      %1231 = vector.extract %1223[1] : f16 from vector<4xf16>
      %1232 = vector.extract %1223[2] : f16 from vector<4xf16>
      %1233 = vector.extract %1223[3] : f16 from vector<4xf16>
      %1234 = vector.from_elements %1230, %1231, %1232, %1233 : vector<4xf16>
      %1235 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1196, %1234, %1199 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1236 = vector.extract %1229[0] : f32 from vector<4xf32>
      %1237 = vector.extract %1229[1] : f32 from vector<4xf32>
      %1238 = vector.extract %1229[2] : f32 from vector<4xf32>
      %1239 = vector.extract %1229[3] : f32 from vector<4xf32>
      %1240 = vector.load %view_7[%913] : memref<4096xf16, 3>, vector<4xf16>
      %1241 = vector.load %view_7[%915] : memref<4096xf16, 3>, vector<4xf16>
      %1242 = vector.extract %1240[0] : f16 from vector<4xf16>
      %1243 = vector.extract %1240[1] : f16 from vector<4xf16>
      %1244 = vector.extract %1240[2] : f16 from vector<4xf16>
      %1245 = vector.extract %1240[3] : f16 from vector<4xf16>
      %1246 = vector.from_elements %1242, %1243, %1244, %1245 : vector<4xf16>
      %1247 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1198, %1246, %1235 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1248 = vector.extract %1241[0] : f16 from vector<4xf16>
      %1249 = vector.extract %1241[1] : f16 from vector<4xf16>
      %1250 = vector.extract %1241[2] : f16 from vector<4xf16>
      %1251 = vector.extract %1241[3] : f16 from vector<4xf16>
      %1252 = vector.from_elements %1248, %1249, %1250, %1251 : vector<4xf16>
      %1253 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1196, %1252, %1206 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1254 = vector.extract %1247[0] : f32 from vector<4xf32>
      %1255 = vector.extract %1247[1] : f32 from vector<4xf32>
      %1256 = vector.extract %1247[2] : f32 from vector<4xf32>
      %1257 = vector.extract %1247[3] : f32 from vector<4xf32>
      %1258 = vector.load %view_7[%933] : memref<4096xf16, 3>, vector<4xf16>
      %1259 = vector.load %view_7[%935] : memref<4096xf16, 3>, vector<4xf16>
      %1260 = vector.extract %1258[0] : f16 from vector<4xf16>
      %1261 = vector.extract %1258[1] : f16 from vector<4xf16>
      %1262 = vector.extract %1258[2] : f16 from vector<4xf16>
      %1263 = vector.extract %1258[3] : f16 from vector<4xf16>
      %1264 = vector.from_elements %1260, %1261, %1262, %1263 : vector<4xf16>
      %1265 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1198, %1264, %1253 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1266 = vector.extract %1259[0] : f16 from vector<4xf16>
      %1267 = vector.extract %1259[1] : f16 from vector<4xf16>
      %1268 = vector.extract %1259[2] : f16 from vector<4xf16>
      %1269 = vector.extract %1259[3] : f16 from vector<4xf16>
      %1270 = vector.from_elements %1266, %1267, %1268, %1269 : vector<4xf16>
      %1271 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1196, %1270, %1207 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1272 = vector.extract %1265[0] : f32 from vector<4xf32>
      %1273 = vector.extract %1265[1] : f32 from vector<4xf32>
      %1274 = vector.extract %1265[2] : f32 from vector<4xf32>
      %1275 = vector.extract %1265[3] : f32 from vector<4xf32>
      %1276 = vector.load %view_7[%953] : memref<4096xf16, 3>, vector<4xf16>
      %1277 = vector.load %view_7[%955] : memref<4096xf16, 3>, vector<4xf16>
      %1278 = vector.extract %1276[0] : f16 from vector<4xf16>
      %1279 = vector.extract %1276[1] : f16 from vector<4xf16>
      %1280 = vector.extract %1276[2] : f16 from vector<4xf16>
      %1281 = vector.extract %1276[3] : f16 from vector<4xf16>
      %1282 = vector.from_elements %1278, %1279, %1280, %1281 : vector<4xf16>
      %1283 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1198, %1282, %1271 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1284 = vector.extract %1277[0] : f16 from vector<4xf16>
      %1285 = vector.extract %1277[1] : f16 from vector<4xf16>
      %1286 = vector.extract %1277[2] : f16 from vector<4xf16>
      %1287 = vector.extract %1277[3] : f16 from vector<4xf16>
      %1288 = vector.from_elements %1284, %1285, %1286, %1287 : vector<4xf16>
      %1289 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1196, %1288, %1208 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1290 = vector.extract %1283[0] : f32 from vector<4xf32>
      %1291 = vector.extract %1283[1] : f32 from vector<4xf32>
      %1292 = vector.extract %1283[2] : f32 from vector<4xf32>
      %1293 = vector.extract %1283[3] : f32 from vector<4xf32>
      %1294 = vector.load %view_7[%973] : memref<4096xf16, 3>, vector<4xf16>
      %1295 = vector.load %view_7[%975] : memref<4096xf16, 3>, vector<4xf16>
      %1296 = vector.extract %1294[0] : f16 from vector<4xf16>
      %1297 = vector.extract %1294[1] : f16 from vector<4xf16>
      %1298 = vector.extract %1294[2] : f16 from vector<4xf16>
      %1299 = vector.extract %1294[3] : f16 from vector<4xf16>
      %1300 = vector.from_elements %1296, %1297, %1298, %1299 : vector<4xf16>
      %1301 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1198, %1300, %1289 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1302 = vector.extract %1295[0] : f16 from vector<4xf16>
      %1303 = vector.extract %1295[1] : f16 from vector<4xf16>
      %1304 = vector.extract %1295[2] : f16 from vector<4xf16>
      %1305 = vector.extract %1295[3] : f16 from vector<4xf16>
      %1306 = vector.from_elements %1302, %1303, %1304, %1305 : vector<4xf16>
      %1307 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1196, %1306, %1209 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1308 = vector.extract %1301[0] : f32 from vector<4xf32>
      %1309 = vector.extract %1301[1] : f32 from vector<4xf32>
      %1310 = vector.extract %1301[2] : f32 from vector<4xf32>
      %1311 = vector.extract %1301[3] : f32 from vector<4xf32>
      %1312 = vector.load %view_7[%993] : memref<4096xf16, 3>, vector<4xf16>
      %1313 = vector.load %view_7[%995] : memref<4096xf16, 3>, vector<4xf16>
      %1314 = vector.extract %1312[0] : f16 from vector<4xf16>
      %1315 = vector.extract %1312[1] : f16 from vector<4xf16>
      %1316 = vector.extract %1312[2] : f16 from vector<4xf16>
      %1317 = vector.extract %1312[3] : f16 from vector<4xf16>
      %1318 = vector.from_elements %1314, %1315, %1316, %1317 : vector<4xf16>
      %1319 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1198, %1318, %1307 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1320 = vector.extract %1313[0] : f16 from vector<4xf16>
      %1321 = vector.extract %1313[1] : f16 from vector<4xf16>
      %1322 = vector.extract %1313[2] : f16 from vector<4xf16>
      %1323 = vector.extract %1313[3] : f16 from vector<4xf16>
      %1324 = vector.from_elements %1320, %1321, %1322, %1323 : vector<4xf16>
      %1325 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1196, %1324, %1210 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1326 = vector.extract %1319[0] : f32 from vector<4xf32>
      %1327 = vector.extract %1319[1] : f32 from vector<4xf32>
      %1328 = vector.extract %1319[2] : f32 from vector<4xf32>
      %1329 = vector.extract %1319[3] : f32 from vector<4xf32>
      %1330 = vector.load %view_7[%1013] : memref<4096xf16, 3>, vector<4xf16>
      %1331 = vector.load %view_7[%1015] : memref<4096xf16, 3>, vector<4xf16>
      %1332 = vector.extract %1330[0] : f16 from vector<4xf16>
      %1333 = vector.extract %1330[1] : f16 from vector<4xf16>
      %1334 = vector.extract %1330[2] : f16 from vector<4xf16>
      %1335 = vector.extract %1330[3] : f16 from vector<4xf16>
      %1336 = vector.from_elements %1332, %1333, %1334, %1335 : vector<4xf16>
      %1337 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1198, %1336, %1325 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1338 = vector.extract %1331[0] : f16 from vector<4xf16>
      %1339 = vector.extract %1331[1] : f16 from vector<4xf16>
      %1340 = vector.extract %1331[2] : f16 from vector<4xf16>
      %1341 = vector.extract %1331[3] : f16 from vector<4xf16>
      %1342 = vector.from_elements %1338, %1339, %1340, %1341 : vector<4xf16>
      %1343 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1196, %1342, %1211 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1344 = vector.extract %1337[0] : f32 from vector<4xf32>
      %1345 = vector.extract %1337[1] : f32 from vector<4xf32>
      %1346 = vector.extract %1337[2] : f32 from vector<4xf32>
      %1347 = vector.extract %1337[3] : f32 from vector<4xf32>
      %1348 = vector.load %view_7[%1033] : memref<4096xf16, 3>, vector<4xf16>
      %1349 = vector.load %view_7[%881] : memref<4096xf16, 3>, vector<4xf16>
      %1350 = vector.extract %1348[0] : f16 from vector<4xf16>
      %1351 = vector.extract %1348[1] : f16 from vector<4xf16>
      %1352 = vector.extract %1348[2] : f16 from vector<4xf16>
      %1353 = vector.extract %1348[3] : f16 from vector<4xf16>
      %1354 = vector.from_elements %1350, %1351, %1352, %1353 : vector<4xf16>
      %1355 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1198, %1354, %1343 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1356 = vector.extract %1349[0] : f16 from vector<4xf16>
      %1357 = vector.extract %1349[1] : f16 from vector<4xf16>
      %1358 = vector.extract %1349[2] : f16 from vector<4xf16>
      %1359 = vector.extract %1349[3] : f16 from vector<4xf16>
      %1360 = vector.from_elements %1356, %1357, %1358, %1359 : vector<4xf16>
      %1361 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1212, %1360, %1213 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1362 = vector.extract %1355[0] : f32 from vector<4xf32>
      %1363 = vector.extract %1355[1] : f32 from vector<4xf32>
      %1364 = vector.extract %1355[2] : f32 from vector<4xf32>
      %1365 = vector.extract %1355[3] : f32 from vector<4xf32>
      %1366 = vector.load %view_7[%893] : memref<4096xf16, 3>, vector<4xf16>
      %1367 = vector.load %view_7[%895] : memref<4096xf16, 3>, vector<4xf16>
      %1368 = vector.extract %1366[0] : f16 from vector<4xf16>
      %1369 = vector.extract %1366[1] : f16 from vector<4xf16>
      %1370 = vector.extract %1366[2] : f16 from vector<4xf16>
      %1371 = vector.extract %1366[3] : f16 from vector<4xf16>
      %1372 = vector.from_elements %1368, %1369, %1370, %1371 : vector<4xf16>
      %1373 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1214, %1372, %1361 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1374 = vector.extract %1367[0] : f16 from vector<4xf16>
      %1375 = vector.extract %1367[1] : f16 from vector<4xf16>
      %1376 = vector.extract %1367[2] : f16 from vector<4xf16>
      %1377 = vector.extract %1367[3] : f16 from vector<4xf16>
      %1378 = vector.from_elements %1374, %1375, %1376, %1377 : vector<4xf16>
      %1379 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1212, %1378, %1215 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1380 = vector.extract %1373[0] : f32 from vector<4xf32>
      %1381 = vector.extract %1373[1] : f32 from vector<4xf32>
      %1382 = vector.extract %1373[2] : f32 from vector<4xf32>
      %1383 = vector.extract %1373[3] : f32 from vector<4xf32>
      %1384 = vector.load %view_7[%913] : memref<4096xf16, 3>, vector<4xf16>
      %1385 = vector.load %view_7[%915] : memref<4096xf16, 3>, vector<4xf16>
      %1386 = vector.extract %1384[0] : f16 from vector<4xf16>
      %1387 = vector.extract %1384[1] : f16 from vector<4xf16>
      %1388 = vector.extract %1384[2] : f16 from vector<4xf16>
      %1389 = vector.extract %1384[3] : f16 from vector<4xf16>
      %1390 = vector.from_elements %1386, %1387, %1388, %1389 : vector<4xf16>
      %1391 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1214, %1390, %1379 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1392 = vector.extract %1385[0] : f16 from vector<4xf16>
      %1393 = vector.extract %1385[1] : f16 from vector<4xf16>
      %1394 = vector.extract %1385[2] : f16 from vector<4xf16>
      %1395 = vector.extract %1385[3] : f16 from vector<4xf16>
      %1396 = vector.from_elements %1392, %1393, %1394, %1395 : vector<4xf16>
      %1397 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1212, %1396, %1216 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1398 = vector.extract %1391[0] : f32 from vector<4xf32>
      %1399 = vector.extract %1391[1] : f32 from vector<4xf32>
      %1400 = vector.extract %1391[2] : f32 from vector<4xf32>
      %1401 = vector.extract %1391[3] : f32 from vector<4xf32>
      %1402 = vector.load %view_7[%933] : memref<4096xf16, 3>, vector<4xf16>
      %1403 = vector.load %view_7[%935] : memref<4096xf16, 3>, vector<4xf16>
      %1404 = vector.extract %1402[0] : f16 from vector<4xf16>
      %1405 = vector.extract %1402[1] : f16 from vector<4xf16>
      %1406 = vector.extract %1402[2] : f16 from vector<4xf16>
      %1407 = vector.extract %1402[3] : f16 from vector<4xf16>
      %1408 = vector.from_elements %1404, %1405, %1406, %1407 : vector<4xf16>
      %1409 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1214, %1408, %1397 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1410 = vector.extract %1403[0] : f16 from vector<4xf16>
      %1411 = vector.extract %1403[1] : f16 from vector<4xf16>
      %1412 = vector.extract %1403[2] : f16 from vector<4xf16>
      %1413 = vector.extract %1403[3] : f16 from vector<4xf16>
      %1414 = vector.from_elements %1410, %1411, %1412, %1413 : vector<4xf16>
      %1415 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1212, %1414, %1217 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1416 = vector.extract %1409[0] : f32 from vector<4xf32>
      %1417 = vector.extract %1409[1] : f32 from vector<4xf32>
      %1418 = vector.extract %1409[2] : f32 from vector<4xf32>
      %1419 = vector.extract %1409[3] : f32 from vector<4xf32>
      %1420 = vector.load %view_7[%953] : memref<4096xf16, 3>, vector<4xf16>
      %1421 = vector.load %view_7[%955] : memref<4096xf16, 3>, vector<4xf16>
      %1422 = vector.extract %1420[0] : f16 from vector<4xf16>
      %1423 = vector.extract %1420[1] : f16 from vector<4xf16>
      %1424 = vector.extract %1420[2] : f16 from vector<4xf16>
      %1425 = vector.extract %1420[3] : f16 from vector<4xf16>
      %1426 = vector.from_elements %1422, %1423, %1424, %1425 : vector<4xf16>
      %1427 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1214, %1426, %1415 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1428 = vector.extract %1421[0] : f16 from vector<4xf16>
      %1429 = vector.extract %1421[1] : f16 from vector<4xf16>
      %1430 = vector.extract %1421[2] : f16 from vector<4xf16>
      %1431 = vector.extract %1421[3] : f16 from vector<4xf16>
      %1432 = vector.from_elements %1428, %1429, %1430, %1431 : vector<4xf16>
      %1433 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1212, %1432, %1218 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1434 = vector.extract %1427[0] : f32 from vector<4xf32>
      %1435 = vector.extract %1427[1] : f32 from vector<4xf32>
      %1436 = vector.extract %1427[2] : f32 from vector<4xf32>
      %1437 = vector.extract %1427[3] : f32 from vector<4xf32>
      %1438 = vector.load %view_7[%973] : memref<4096xf16, 3>, vector<4xf16>
      %1439 = vector.load %view_7[%975] : memref<4096xf16, 3>, vector<4xf16>
      %1440 = vector.extract %1438[0] : f16 from vector<4xf16>
      %1441 = vector.extract %1438[1] : f16 from vector<4xf16>
      %1442 = vector.extract %1438[2] : f16 from vector<4xf16>
      %1443 = vector.extract %1438[3] : f16 from vector<4xf16>
      %1444 = vector.from_elements %1440, %1441, %1442, %1443 : vector<4xf16>
      %1445 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1214, %1444, %1433 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1446 = vector.extract %1439[0] : f16 from vector<4xf16>
      %1447 = vector.extract %1439[1] : f16 from vector<4xf16>
      %1448 = vector.extract %1439[2] : f16 from vector<4xf16>
      %1449 = vector.extract %1439[3] : f16 from vector<4xf16>
      %1450 = vector.from_elements %1446, %1447, %1448, %1449 : vector<4xf16>
      %1451 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1212, %1450, %1219 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1452 = vector.extract %1445[0] : f32 from vector<4xf32>
      %1453 = vector.extract %1445[1] : f32 from vector<4xf32>
      %1454 = vector.extract %1445[2] : f32 from vector<4xf32>
      %1455 = vector.extract %1445[3] : f32 from vector<4xf32>
      %1456 = vector.load %view_7[%993] : memref<4096xf16, 3>, vector<4xf16>
      %1457 = vector.load %view_7[%995] : memref<4096xf16, 3>, vector<4xf16>
      %1458 = vector.extract %1456[0] : f16 from vector<4xf16>
      %1459 = vector.extract %1456[1] : f16 from vector<4xf16>
      %1460 = vector.extract %1456[2] : f16 from vector<4xf16>
      %1461 = vector.extract %1456[3] : f16 from vector<4xf16>
      %1462 = vector.from_elements %1458, %1459, %1460, %1461 : vector<4xf16>
      %1463 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1214, %1462, %1451 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1464 = vector.extract %1457[0] : f16 from vector<4xf16>
      %1465 = vector.extract %1457[1] : f16 from vector<4xf16>
      %1466 = vector.extract %1457[2] : f16 from vector<4xf16>
      %1467 = vector.extract %1457[3] : f16 from vector<4xf16>
      %1468 = vector.from_elements %1464, %1465, %1466, %1467 : vector<4xf16>
      %1469 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1212, %1468, %1220 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1470 = vector.extract %1463[0] : f32 from vector<4xf32>
      %1471 = vector.extract %1463[1] : f32 from vector<4xf32>
      %1472 = vector.extract %1463[2] : f32 from vector<4xf32>
      %1473 = vector.extract %1463[3] : f32 from vector<4xf32>
      %1474 = vector.load %view_7[%1013] : memref<4096xf16, 3>, vector<4xf16>
      %1475 = vector.load %view_7[%1015] : memref<4096xf16, 3>, vector<4xf16>
      %1476 = vector.extract %1474[0] : f16 from vector<4xf16>
      %1477 = vector.extract %1474[1] : f16 from vector<4xf16>
      %1478 = vector.extract %1474[2] : f16 from vector<4xf16>
      %1479 = vector.extract %1474[3] : f16 from vector<4xf16>
      %1480 = vector.from_elements %1476, %1477, %1478, %1479 : vector<4xf16>
      %1481 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1214, %1480, %1469 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1482 = vector.extract %1475[0] : f16 from vector<4xf16>
      %1483 = vector.extract %1475[1] : f16 from vector<4xf16>
      %1484 = vector.extract %1475[2] : f16 from vector<4xf16>
      %1485 = vector.extract %1475[3] : f16 from vector<4xf16>
      %1486 = vector.from_elements %1482, %1483, %1484, %1485 : vector<4xf16>
      %1487 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1212, %1486, %1221 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1488 = vector.extract %1481[0] : f32 from vector<4xf32>
      %1489 = vector.extract %1481[1] : f32 from vector<4xf32>
      %1490 = vector.extract %1481[2] : f32 from vector<4xf32>
      %1491 = vector.extract %1481[3] : f32 from vector<4xf32>
      %1492 = vector.load %view_7[%1033] : memref<4096xf16, 3>, vector<4xf16>
      %1493 = vector.extract %1492[0] : f16 from vector<4xf16>
      %1494 = vector.extract %1492[1] : f16 from vector<4xf16>
      %1495 = vector.extract %1492[2] : f16 from vector<4xf16>
      %1496 = vector.extract %1492[3] : f16 from vector<4xf16>
      %1497 = vector.from_elements %1493, %1494, %1495, %1496 : vector<4xf16>
      %1498 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1214, %1497, %1487 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1499 = vector.extract %1498[0] : f32 from vector<4xf32>
      %1500 = vector.extract %1498[1] : f32 from vector<4xf32>
      %1501 = vector.extract %1498[2] : f32 from vector<4xf32>
      %1502 = vector.extract %1498[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %subview_14 = memref.subview %transpose[0, %277, 96, %856] [1, 1, 32, 128] [1, 1, 1, 1] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> to memref<32x128xf16, strided<[1, 128], offset: ?>, 1>
      %1503 = builtin.unrealized_conversion_cast %subview_14 : memref<32x128xf16, strided<[1, 128], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %1504 = llvm.extractvalue %1503[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1505 = llvm.extractvalue %1503[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1506 = llvm.getelementptr %1504[%1505] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %1507 = llvm.getelementptr %1506[%866] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1507, %78, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %1508 = llvm.getelementptr %1506[%873] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1508, %83, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %1509 = vector.load %view[%881] : memref<4096xf16, 3>, vector<4xf16>
      %1510 = vector.from_elements %209, %210, %211, %212 : vector<4xf16>
      %1511 = vector.from_elements %1236, %1237, %1238, %1239 : vector<4xf32>
      %1512 = vector.from_elements %213, %214, %215, %216 : vector<4xf16>
      %1513 = vector.from_elements %1254, %1255, %1256, %1257 : vector<4xf32>
      %1514 = vector.extract %1509[0] : f16 from vector<4xf16>
      %1515 = vector.extract %1509[1] : f16 from vector<4xf16>
      %1516 = vector.extract %1509[2] : f16 from vector<4xf16>
      %1517 = vector.extract %1509[3] : f16 from vector<4xf16>
      %1518 = vector.from_elements %1514, %1515, %1516, %1517 : vector<4xf16>
      %1519 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1510, %1518, %1511 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1520 = vector.from_elements %1272, %1273, %1274, %1275 : vector<4xf32>
      %1521 = vector.from_elements %1290, %1291, %1292, %1293 : vector<4xf32>
      %1522 = vector.from_elements %1308, %1309, %1310, %1311 : vector<4xf32>
      %1523 = vector.from_elements %1326, %1327, %1328, %1329 : vector<4xf32>
      %1524 = vector.from_elements %1344, %1345, %1346, %1347 : vector<4xf32>
      %1525 = vector.from_elements %1362, %1363, %1364, %1365 : vector<4xf32>
      %1526 = vector.from_elements %225, %226, %227, %228 : vector<4xf16>
      %1527 = vector.from_elements %1380, %1381, %1382, %1383 : vector<4xf32>
      %1528 = vector.from_elements %229, %230, %231, %232 : vector<4xf16>
      %1529 = vector.from_elements %1398, %1399, %1400, %1401 : vector<4xf32>
      %1530 = vector.from_elements %1416, %1417, %1418, %1419 : vector<4xf32>
      %1531 = vector.from_elements %1434, %1435, %1436, %1437 : vector<4xf32>
      %1532 = vector.from_elements %1452, %1453, %1454, %1455 : vector<4xf32>
      %1533 = vector.from_elements %1470, %1471, %1472, %1473 : vector<4xf32>
      %1534 = vector.from_elements %1488, %1489, %1490, %1491 : vector<4xf32>
      %1535 = vector.from_elements %1499, %1500, %1501, %1502 : vector<4xf32>
      %1536 = vector.load %view[%893] : memref<4096xf16, 3>, vector<4xf16>
      %1537 = vector.load %view[%895] : memref<4096xf16, 3>, vector<4xf16>
      %1538 = vector.extract %1536[0] : f16 from vector<4xf16>
      %1539 = vector.extract %1536[1] : f16 from vector<4xf16>
      %1540 = vector.extract %1536[2] : f16 from vector<4xf16>
      %1541 = vector.extract %1536[3] : f16 from vector<4xf16>
      %1542 = vector.from_elements %1538, %1539, %1540, %1541 : vector<4xf16>
      %1543 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1512, %1542, %1519 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1544 = vector.extract %1537[0] : f16 from vector<4xf16>
      %1545 = vector.extract %1537[1] : f16 from vector<4xf16>
      %1546 = vector.extract %1537[2] : f16 from vector<4xf16>
      %1547 = vector.extract %1537[3] : f16 from vector<4xf16>
      %1548 = vector.from_elements %1544, %1545, %1546, %1547 : vector<4xf16>
      %1549 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1510, %1548, %1513 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1550 = vector.extract %1543[0] : f32 from vector<4xf32>
      %1551 = vector.extract %1543[1] : f32 from vector<4xf32>
      %1552 = vector.extract %1543[2] : f32 from vector<4xf32>
      %1553 = vector.extract %1543[3] : f32 from vector<4xf32>
      %1554 = vector.load %view[%913] : memref<4096xf16, 3>, vector<4xf16>
      %1555 = vector.load %view[%915] : memref<4096xf16, 3>, vector<4xf16>
      %1556 = vector.extract %1554[0] : f16 from vector<4xf16>
      %1557 = vector.extract %1554[1] : f16 from vector<4xf16>
      %1558 = vector.extract %1554[2] : f16 from vector<4xf16>
      %1559 = vector.extract %1554[3] : f16 from vector<4xf16>
      %1560 = vector.from_elements %1556, %1557, %1558, %1559 : vector<4xf16>
      %1561 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1512, %1560, %1549 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1562 = vector.extract %1555[0] : f16 from vector<4xf16>
      %1563 = vector.extract %1555[1] : f16 from vector<4xf16>
      %1564 = vector.extract %1555[2] : f16 from vector<4xf16>
      %1565 = vector.extract %1555[3] : f16 from vector<4xf16>
      %1566 = vector.from_elements %1562, %1563, %1564, %1565 : vector<4xf16>
      %1567 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1510, %1566, %1520 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1568 = vector.extract %1561[0] : f32 from vector<4xf32>
      %1569 = vector.extract %1561[1] : f32 from vector<4xf32>
      %1570 = vector.extract %1561[2] : f32 from vector<4xf32>
      %1571 = vector.extract %1561[3] : f32 from vector<4xf32>
      %1572 = vector.load %view[%933] : memref<4096xf16, 3>, vector<4xf16>
      %1573 = vector.load %view[%935] : memref<4096xf16, 3>, vector<4xf16>
      %1574 = vector.extract %1572[0] : f16 from vector<4xf16>
      %1575 = vector.extract %1572[1] : f16 from vector<4xf16>
      %1576 = vector.extract %1572[2] : f16 from vector<4xf16>
      %1577 = vector.extract %1572[3] : f16 from vector<4xf16>
      %1578 = vector.from_elements %1574, %1575, %1576, %1577 : vector<4xf16>
      %1579 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1512, %1578, %1567 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1580 = vector.extract %1573[0] : f16 from vector<4xf16>
      %1581 = vector.extract %1573[1] : f16 from vector<4xf16>
      %1582 = vector.extract %1573[2] : f16 from vector<4xf16>
      %1583 = vector.extract %1573[3] : f16 from vector<4xf16>
      %1584 = vector.from_elements %1580, %1581, %1582, %1583 : vector<4xf16>
      %1585 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1510, %1584, %1521 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1586 = vector.extract %1579[0] : f32 from vector<4xf32>
      %1587 = vector.extract %1579[1] : f32 from vector<4xf32>
      %1588 = vector.extract %1579[2] : f32 from vector<4xf32>
      %1589 = vector.extract %1579[3] : f32 from vector<4xf32>
      %1590 = vector.load %view[%953] : memref<4096xf16, 3>, vector<4xf16>
      %1591 = vector.load %view[%955] : memref<4096xf16, 3>, vector<4xf16>
      %1592 = vector.extract %1590[0] : f16 from vector<4xf16>
      %1593 = vector.extract %1590[1] : f16 from vector<4xf16>
      %1594 = vector.extract %1590[2] : f16 from vector<4xf16>
      %1595 = vector.extract %1590[3] : f16 from vector<4xf16>
      %1596 = vector.from_elements %1592, %1593, %1594, %1595 : vector<4xf16>
      %1597 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1512, %1596, %1585 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1598 = vector.extract %1591[0] : f16 from vector<4xf16>
      %1599 = vector.extract %1591[1] : f16 from vector<4xf16>
      %1600 = vector.extract %1591[2] : f16 from vector<4xf16>
      %1601 = vector.extract %1591[3] : f16 from vector<4xf16>
      %1602 = vector.from_elements %1598, %1599, %1600, %1601 : vector<4xf16>
      %1603 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1510, %1602, %1522 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1604 = vector.extract %1597[0] : f32 from vector<4xf32>
      %1605 = vector.extract %1597[1] : f32 from vector<4xf32>
      %1606 = vector.extract %1597[2] : f32 from vector<4xf32>
      %1607 = vector.extract %1597[3] : f32 from vector<4xf32>
      %1608 = vector.load %view[%973] : memref<4096xf16, 3>, vector<4xf16>
      %1609 = vector.load %view[%975] : memref<4096xf16, 3>, vector<4xf16>
      %1610 = vector.extract %1608[0] : f16 from vector<4xf16>
      %1611 = vector.extract %1608[1] : f16 from vector<4xf16>
      %1612 = vector.extract %1608[2] : f16 from vector<4xf16>
      %1613 = vector.extract %1608[3] : f16 from vector<4xf16>
      %1614 = vector.from_elements %1610, %1611, %1612, %1613 : vector<4xf16>
      %1615 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1512, %1614, %1603 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1616 = vector.extract %1609[0] : f16 from vector<4xf16>
      %1617 = vector.extract %1609[1] : f16 from vector<4xf16>
      %1618 = vector.extract %1609[2] : f16 from vector<4xf16>
      %1619 = vector.extract %1609[3] : f16 from vector<4xf16>
      %1620 = vector.from_elements %1616, %1617, %1618, %1619 : vector<4xf16>
      %1621 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1510, %1620, %1523 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1622 = vector.extract %1615[0] : f32 from vector<4xf32>
      %1623 = vector.extract %1615[1] : f32 from vector<4xf32>
      %1624 = vector.extract %1615[2] : f32 from vector<4xf32>
      %1625 = vector.extract %1615[3] : f32 from vector<4xf32>
      %1626 = vector.load %view[%993] : memref<4096xf16, 3>, vector<4xf16>
      %1627 = vector.load %view[%995] : memref<4096xf16, 3>, vector<4xf16>
      %1628 = vector.extract %1626[0] : f16 from vector<4xf16>
      %1629 = vector.extract %1626[1] : f16 from vector<4xf16>
      %1630 = vector.extract %1626[2] : f16 from vector<4xf16>
      %1631 = vector.extract %1626[3] : f16 from vector<4xf16>
      %1632 = vector.from_elements %1628, %1629, %1630, %1631 : vector<4xf16>
      %1633 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1512, %1632, %1621 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1634 = vector.extract %1627[0] : f16 from vector<4xf16>
      %1635 = vector.extract %1627[1] : f16 from vector<4xf16>
      %1636 = vector.extract %1627[2] : f16 from vector<4xf16>
      %1637 = vector.extract %1627[3] : f16 from vector<4xf16>
      %1638 = vector.from_elements %1634, %1635, %1636, %1637 : vector<4xf16>
      %1639 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1510, %1638, %1524 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1640 = vector.extract %1633[0] : f32 from vector<4xf32>
      %1641 = vector.extract %1633[1] : f32 from vector<4xf32>
      %1642 = vector.extract %1633[2] : f32 from vector<4xf32>
      %1643 = vector.extract %1633[3] : f32 from vector<4xf32>
      %1644 = vector.load %view[%1013] : memref<4096xf16, 3>, vector<4xf16>
      %1645 = vector.load %view[%1015] : memref<4096xf16, 3>, vector<4xf16>
      %1646 = vector.extract %1644[0] : f16 from vector<4xf16>
      %1647 = vector.extract %1644[1] : f16 from vector<4xf16>
      %1648 = vector.extract %1644[2] : f16 from vector<4xf16>
      %1649 = vector.extract %1644[3] : f16 from vector<4xf16>
      %1650 = vector.from_elements %1646, %1647, %1648, %1649 : vector<4xf16>
      %1651 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1512, %1650, %1639 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1652 = vector.extract %1645[0] : f16 from vector<4xf16>
      %1653 = vector.extract %1645[1] : f16 from vector<4xf16>
      %1654 = vector.extract %1645[2] : f16 from vector<4xf16>
      %1655 = vector.extract %1645[3] : f16 from vector<4xf16>
      %1656 = vector.from_elements %1652, %1653, %1654, %1655 : vector<4xf16>
      %1657 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1510, %1656, %1525 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1658 = vector.extract %1651[0] : f32 from vector<4xf32>
      %1659 = vector.extract %1651[1] : f32 from vector<4xf32>
      %1660 = vector.extract %1651[2] : f32 from vector<4xf32>
      %1661 = vector.extract %1651[3] : f32 from vector<4xf32>
      %1662 = vector.load %view[%1033] : memref<4096xf16, 3>, vector<4xf16>
      %1663 = vector.load %view[%881] : memref<4096xf16, 3>, vector<4xf16>
      %1664 = vector.extract %1662[0] : f16 from vector<4xf16>
      %1665 = vector.extract %1662[1] : f16 from vector<4xf16>
      %1666 = vector.extract %1662[2] : f16 from vector<4xf16>
      %1667 = vector.extract %1662[3] : f16 from vector<4xf16>
      %1668 = vector.from_elements %1664, %1665, %1666, %1667 : vector<4xf16>
      %1669 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1512, %1668, %1657 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1670 = vector.extract %1663[0] : f16 from vector<4xf16>
      %1671 = vector.extract %1663[1] : f16 from vector<4xf16>
      %1672 = vector.extract %1663[2] : f16 from vector<4xf16>
      %1673 = vector.extract %1663[3] : f16 from vector<4xf16>
      %1674 = vector.from_elements %1670, %1671, %1672, %1673 : vector<4xf16>
      %1675 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1526, %1674, %1527 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1676 = vector.extract %1669[0] : f32 from vector<4xf32>
      %1677 = vector.extract %1669[1] : f32 from vector<4xf32>
      %1678 = vector.extract %1669[2] : f32 from vector<4xf32>
      %1679 = vector.extract %1669[3] : f32 from vector<4xf32>
      %1680 = vector.load %view[%893] : memref<4096xf16, 3>, vector<4xf16>
      %1681 = vector.load %view[%895] : memref<4096xf16, 3>, vector<4xf16>
      %1682 = vector.extract %1680[0] : f16 from vector<4xf16>
      %1683 = vector.extract %1680[1] : f16 from vector<4xf16>
      %1684 = vector.extract %1680[2] : f16 from vector<4xf16>
      %1685 = vector.extract %1680[3] : f16 from vector<4xf16>
      %1686 = vector.from_elements %1682, %1683, %1684, %1685 : vector<4xf16>
      %1687 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1528, %1686, %1675 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1688 = vector.extract %1681[0] : f16 from vector<4xf16>
      %1689 = vector.extract %1681[1] : f16 from vector<4xf16>
      %1690 = vector.extract %1681[2] : f16 from vector<4xf16>
      %1691 = vector.extract %1681[3] : f16 from vector<4xf16>
      %1692 = vector.from_elements %1688, %1689, %1690, %1691 : vector<4xf16>
      %1693 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1526, %1692, %1529 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1694 = vector.extract %1687[0] : f32 from vector<4xf32>
      %1695 = vector.extract %1687[1] : f32 from vector<4xf32>
      %1696 = vector.extract %1687[2] : f32 from vector<4xf32>
      %1697 = vector.extract %1687[3] : f32 from vector<4xf32>
      %1698 = vector.load %view[%913] : memref<4096xf16, 3>, vector<4xf16>
      %1699 = vector.load %view[%915] : memref<4096xf16, 3>, vector<4xf16>
      %1700 = vector.extract %1698[0] : f16 from vector<4xf16>
      %1701 = vector.extract %1698[1] : f16 from vector<4xf16>
      %1702 = vector.extract %1698[2] : f16 from vector<4xf16>
      %1703 = vector.extract %1698[3] : f16 from vector<4xf16>
      %1704 = vector.from_elements %1700, %1701, %1702, %1703 : vector<4xf16>
      %1705 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1528, %1704, %1693 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1706 = vector.extract %1699[0] : f16 from vector<4xf16>
      %1707 = vector.extract %1699[1] : f16 from vector<4xf16>
      %1708 = vector.extract %1699[2] : f16 from vector<4xf16>
      %1709 = vector.extract %1699[3] : f16 from vector<4xf16>
      %1710 = vector.from_elements %1706, %1707, %1708, %1709 : vector<4xf16>
      %1711 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1526, %1710, %1530 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1712 = vector.extract %1705[0] : f32 from vector<4xf32>
      %1713 = vector.extract %1705[1] : f32 from vector<4xf32>
      %1714 = vector.extract %1705[2] : f32 from vector<4xf32>
      %1715 = vector.extract %1705[3] : f32 from vector<4xf32>
      %1716 = vector.load %view[%933] : memref<4096xf16, 3>, vector<4xf16>
      %1717 = vector.load %view[%935] : memref<4096xf16, 3>, vector<4xf16>
      %1718 = vector.extract %1716[0] : f16 from vector<4xf16>
      %1719 = vector.extract %1716[1] : f16 from vector<4xf16>
      %1720 = vector.extract %1716[2] : f16 from vector<4xf16>
      %1721 = vector.extract %1716[3] : f16 from vector<4xf16>
      %1722 = vector.from_elements %1718, %1719, %1720, %1721 : vector<4xf16>
      %1723 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1528, %1722, %1711 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1724 = vector.extract %1717[0] : f16 from vector<4xf16>
      %1725 = vector.extract %1717[1] : f16 from vector<4xf16>
      %1726 = vector.extract %1717[2] : f16 from vector<4xf16>
      %1727 = vector.extract %1717[3] : f16 from vector<4xf16>
      %1728 = vector.from_elements %1724, %1725, %1726, %1727 : vector<4xf16>
      %1729 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1526, %1728, %1531 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1730 = vector.extract %1723[0] : f32 from vector<4xf32>
      %1731 = vector.extract %1723[1] : f32 from vector<4xf32>
      %1732 = vector.extract %1723[2] : f32 from vector<4xf32>
      %1733 = vector.extract %1723[3] : f32 from vector<4xf32>
      %1734 = vector.load %view[%953] : memref<4096xf16, 3>, vector<4xf16>
      %1735 = vector.load %view[%955] : memref<4096xf16, 3>, vector<4xf16>
      %1736 = vector.extract %1734[0] : f16 from vector<4xf16>
      %1737 = vector.extract %1734[1] : f16 from vector<4xf16>
      %1738 = vector.extract %1734[2] : f16 from vector<4xf16>
      %1739 = vector.extract %1734[3] : f16 from vector<4xf16>
      %1740 = vector.from_elements %1736, %1737, %1738, %1739 : vector<4xf16>
      %1741 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1528, %1740, %1729 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1742 = vector.extract %1735[0] : f16 from vector<4xf16>
      %1743 = vector.extract %1735[1] : f16 from vector<4xf16>
      %1744 = vector.extract %1735[2] : f16 from vector<4xf16>
      %1745 = vector.extract %1735[3] : f16 from vector<4xf16>
      %1746 = vector.from_elements %1742, %1743, %1744, %1745 : vector<4xf16>
      %1747 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1526, %1746, %1532 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1748 = vector.extract %1741[0] : f32 from vector<4xf32>
      %1749 = vector.extract %1741[1] : f32 from vector<4xf32>
      %1750 = vector.extract %1741[2] : f32 from vector<4xf32>
      %1751 = vector.extract %1741[3] : f32 from vector<4xf32>
      %1752 = vector.load %view[%973] : memref<4096xf16, 3>, vector<4xf16>
      %1753 = vector.load %view[%975] : memref<4096xf16, 3>, vector<4xf16>
      %1754 = vector.extract %1752[0] : f16 from vector<4xf16>
      %1755 = vector.extract %1752[1] : f16 from vector<4xf16>
      %1756 = vector.extract %1752[2] : f16 from vector<4xf16>
      %1757 = vector.extract %1752[3] : f16 from vector<4xf16>
      %1758 = vector.from_elements %1754, %1755, %1756, %1757 : vector<4xf16>
      %1759 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1528, %1758, %1747 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1760 = vector.extract %1753[0] : f16 from vector<4xf16>
      %1761 = vector.extract %1753[1] : f16 from vector<4xf16>
      %1762 = vector.extract %1753[2] : f16 from vector<4xf16>
      %1763 = vector.extract %1753[3] : f16 from vector<4xf16>
      %1764 = vector.from_elements %1760, %1761, %1762, %1763 : vector<4xf16>
      %1765 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1526, %1764, %1533 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1766 = vector.extract %1759[0] : f32 from vector<4xf32>
      %1767 = vector.extract %1759[1] : f32 from vector<4xf32>
      %1768 = vector.extract %1759[2] : f32 from vector<4xf32>
      %1769 = vector.extract %1759[3] : f32 from vector<4xf32>
      %1770 = vector.load %view[%993] : memref<4096xf16, 3>, vector<4xf16>
      %1771 = vector.load %view[%995] : memref<4096xf16, 3>, vector<4xf16>
      %1772 = vector.extract %1770[0] : f16 from vector<4xf16>
      %1773 = vector.extract %1770[1] : f16 from vector<4xf16>
      %1774 = vector.extract %1770[2] : f16 from vector<4xf16>
      %1775 = vector.extract %1770[3] : f16 from vector<4xf16>
      %1776 = vector.from_elements %1772, %1773, %1774, %1775 : vector<4xf16>
      %1777 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1528, %1776, %1765 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1778 = vector.extract %1771[0] : f16 from vector<4xf16>
      %1779 = vector.extract %1771[1] : f16 from vector<4xf16>
      %1780 = vector.extract %1771[2] : f16 from vector<4xf16>
      %1781 = vector.extract %1771[3] : f16 from vector<4xf16>
      %1782 = vector.from_elements %1778, %1779, %1780, %1781 : vector<4xf16>
      %1783 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1526, %1782, %1534 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1784 = vector.extract %1777[0] : f32 from vector<4xf32>
      %1785 = vector.extract %1777[1] : f32 from vector<4xf32>
      %1786 = vector.extract %1777[2] : f32 from vector<4xf32>
      %1787 = vector.extract %1777[3] : f32 from vector<4xf32>
      %1788 = vector.load %view[%1013] : memref<4096xf16, 3>, vector<4xf16>
      %1789 = vector.load %view[%1015] : memref<4096xf16, 3>, vector<4xf16>
      %1790 = vector.extract %1788[0] : f16 from vector<4xf16>
      %1791 = vector.extract %1788[1] : f16 from vector<4xf16>
      %1792 = vector.extract %1788[2] : f16 from vector<4xf16>
      %1793 = vector.extract %1788[3] : f16 from vector<4xf16>
      %1794 = vector.from_elements %1790, %1791, %1792, %1793 : vector<4xf16>
      %1795 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1528, %1794, %1783 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1796 = vector.extract %1789[0] : f16 from vector<4xf16>
      %1797 = vector.extract %1789[1] : f16 from vector<4xf16>
      %1798 = vector.extract %1789[2] : f16 from vector<4xf16>
      %1799 = vector.extract %1789[3] : f16 from vector<4xf16>
      %1800 = vector.from_elements %1796, %1797, %1798, %1799 : vector<4xf16>
      %1801 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1526, %1800, %1535 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1802 = vector.extract %1795[0] : f32 from vector<4xf32>
      %1803 = vector.extract %1795[1] : f32 from vector<4xf32>
      %1804 = vector.extract %1795[2] : f32 from vector<4xf32>
      %1805 = vector.extract %1795[3] : f32 from vector<4xf32>
      %1806 = vector.load %view[%1033] : memref<4096xf16, 3>, vector<4xf16>
      %1807 = vector.extract %1806[0] : f16 from vector<4xf16>
      %1808 = vector.extract %1806[1] : f16 from vector<4xf16>
      %1809 = vector.extract %1806[2] : f16 from vector<4xf16>
      %1810 = vector.extract %1806[3] : f16 from vector<4xf16>
      %1811 = vector.from_elements %1807, %1808, %1809, %1810 : vector<4xf16>
      %1812 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1528, %1811, %1801 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1813 = vector.extract %1812[0] : f32 from vector<4xf32>
      %1814 = vector.extract %1812[1] : f32 from vector<4xf32>
      %1815 = vector.extract %1812[2] : f32 from vector<4xf32>
      %1816 = vector.extract %1812[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %subview_15 = memref.subview %arg2[0, %277, %856, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %1817 = builtin.unrealized_conversion_cast %subview_15 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      %1818 = affine.apply affine_map<()[s0] -> (s0 floordiv 32)>()[%29]
      %1819 = builtin.unrealized_conversion_cast %1818 : index to i64
      %1820 = affine.apply affine_map<()[s0] -> (s0 - ((s0 floordiv 8) floordiv 4) * 32)>()[%29]
      %1821 = builtin.unrealized_conversion_cast %1820 : index to i64
      %1822 = llvm.extractvalue %1817[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1823 = llvm.extractvalue %1817[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %1824 = llvm.getelementptr %1822[%1823] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %1825 = llvm.mul %1819, %15 : i64
      %1826 = llvm.add %1825, %1821 : i64
      %1827 = llvm.getelementptr %1824[%1826] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1827, %48, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %1828 = affine.apply affine_map<()[s0] -> (s0 floordiv 32)>()[%50]
      %1829 = builtin.unrealized_conversion_cast %1828 : index to i64
      %1830 = affine.apply affine_map<()[s0] -> (s0 - ((s0 floordiv 8) floordiv 4) * 32)>()[%50]
      %1831 = builtin.unrealized_conversion_cast %1830 : index to i64
      %1832 = llvm.mul %1829, %15 : i64
      %1833 = llvm.add %1832, %1831 : i64
      %1834 = llvm.getelementptr %1824[%1833] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %1834, %65, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %1835 = vector.load %view_7[%881] : memref<4096xf16, 3>, vector<4xf16>
      %1836 = vector.from_elements %253, %254, %255, %256 : vector<4xf16>
      %1837 = vector.from_elements %1550, %1551, %1552, %1553 : vector<4xf32>
      %1838 = vector.from_elements %257, %258, %259, %260 : vector<4xf16>
      %1839 = vector.from_elements %1568, %1569, %1570, %1571 : vector<4xf32>
      %1840 = vector.extract %1835[0] : f16 from vector<4xf16>
      %1841 = vector.extract %1835[1] : f16 from vector<4xf16>
      %1842 = vector.extract %1835[2] : f16 from vector<4xf16>
      %1843 = vector.extract %1835[3] : f16 from vector<4xf16>
      %1844 = vector.from_elements %1840, %1841, %1842, %1843 : vector<4xf16>
      %1845 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1836, %1844, %1837 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1846 = vector.from_elements %1586, %1587, %1588, %1589 : vector<4xf32>
      %1847 = vector.from_elements %1604, %1605, %1606, %1607 : vector<4xf32>
      %1848 = vector.from_elements %1622, %1623, %1624, %1625 : vector<4xf32>
      %1849 = vector.from_elements %1640, %1641, %1642, %1643 : vector<4xf32>
      %1850 = vector.from_elements %1658, %1659, %1660, %1661 : vector<4xf32>
      %1851 = vector.from_elements %1676, %1677, %1678, %1679 : vector<4xf32>
      %1852 = vector.from_elements %269, %270, %271, %272 : vector<4xf16>
      %1853 = vector.from_elements %1694, %1695, %1696, %1697 : vector<4xf32>
      %1854 = vector.from_elements %273, %274, %275, %276 : vector<4xf16>
      %1855 = vector.from_elements %1712, %1713, %1714, %1715 : vector<4xf32>
      %1856 = vector.from_elements %1730, %1731, %1732, %1733 : vector<4xf32>
      %1857 = vector.from_elements %1748, %1749, %1750, %1751 : vector<4xf32>
      %1858 = vector.from_elements %1766, %1767, %1768, %1769 : vector<4xf32>
      %1859 = vector.from_elements %1784, %1785, %1786, %1787 : vector<4xf32>
      %1860 = vector.from_elements %1802, %1803, %1804, %1805 : vector<4xf32>
      %1861 = vector.from_elements %1813, %1814, %1815, %1816 : vector<4xf32>
      %1862 = vector.load %view_7[%893] : memref<4096xf16, 3>, vector<4xf16>
      %1863 = vector.load %view_7[%895] : memref<4096xf16, 3>, vector<4xf16>
      %1864 = vector.extract %1862[0] : f16 from vector<4xf16>
      %1865 = vector.extract %1862[1] : f16 from vector<4xf16>
      %1866 = vector.extract %1862[2] : f16 from vector<4xf16>
      %1867 = vector.extract %1862[3] : f16 from vector<4xf16>
      %1868 = vector.from_elements %1864, %1865, %1866, %1867 : vector<4xf16>
      %1869 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1838, %1868, %1845 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1870 = vector.extract %1863[0] : f16 from vector<4xf16>
      %1871 = vector.extract %1863[1] : f16 from vector<4xf16>
      %1872 = vector.extract %1863[2] : f16 from vector<4xf16>
      %1873 = vector.extract %1863[3] : f16 from vector<4xf16>
      %1874 = vector.from_elements %1870, %1871, %1872, %1873 : vector<4xf16>
      %1875 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1836, %1874, %1839 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1876 = vector.extract %1869[0] : f32 from vector<4xf32>
      %1877 = arith.maxnumf %1876, %cst_3 : f32
      %1878 = vector.extract %1869[1] : f32 from vector<4xf32>
      %1879 = arith.maxnumf %1877, %1878 : f32
      %1880 = vector.extract %1869[2] : f32 from vector<4xf32>
      %1881 = arith.maxnumf %1879, %1880 : f32
      %1882 = vector.extract %1869[3] : f32 from vector<4xf32>
      %1883 = arith.maxnumf %1881, %1882 : f32
      %1884 = vector.load %view_7[%913] : memref<4096xf16, 3>, vector<4xf16>
      %1885 = vector.load %view_7[%915] : memref<4096xf16, 3>, vector<4xf16>
      %1886 = vector.extract %1884[0] : f16 from vector<4xf16>
      %1887 = vector.extract %1884[1] : f16 from vector<4xf16>
      %1888 = vector.extract %1884[2] : f16 from vector<4xf16>
      %1889 = vector.extract %1884[3] : f16 from vector<4xf16>
      %1890 = vector.from_elements %1886, %1887, %1888, %1889 : vector<4xf16>
      %1891 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1838, %1890, %1875 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1892 = vector.extract %1885[0] : f16 from vector<4xf16>
      %1893 = vector.extract %1885[1] : f16 from vector<4xf16>
      %1894 = vector.extract %1885[2] : f16 from vector<4xf16>
      %1895 = vector.extract %1885[3] : f16 from vector<4xf16>
      %1896 = vector.from_elements %1892, %1893, %1894, %1895 : vector<4xf16>
      %1897 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1836, %1896, %1846 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1898 = vector.extract %1891[0] : f32 from vector<4xf32>
      %1899 = arith.maxnumf %1883, %1898 : f32
      %1900 = vector.extract %1891[1] : f32 from vector<4xf32>
      %1901 = arith.maxnumf %1899, %1900 : f32
      %1902 = vector.extract %1891[2] : f32 from vector<4xf32>
      %1903 = arith.maxnumf %1901, %1902 : f32
      %1904 = vector.extract %1891[3] : f32 from vector<4xf32>
      %1905 = arith.maxnumf %1903, %1904 : f32
      %1906 = vector.load %view_7[%933] : memref<4096xf16, 3>, vector<4xf16>
      %1907 = vector.load %view_7[%935] : memref<4096xf16, 3>, vector<4xf16>
      %1908 = vector.extract %1906[0] : f16 from vector<4xf16>
      %1909 = vector.extract %1906[1] : f16 from vector<4xf16>
      %1910 = vector.extract %1906[2] : f16 from vector<4xf16>
      %1911 = vector.extract %1906[3] : f16 from vector<4xf16>
      %1912 = vector.from_elements %1908, %1909, %1910, %1911 : vector<4xf16>
      %1913 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1838, %1912, %1897 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1914 = vector.extract %1907[0] : f16 from vector<4xf16>
      %1915 = vector.extract %1907[1] : f16 from vector<4xf16>
      %1916 = vector.extract %1907[2] : f16 from vector<4xf16>
      %1917 = vector.extract %1907[3] : f16 from vector<4xf16>
      %1918 = vector.from_elements %1914, %1915, %1916, %1917 : vector<4xf16>
      %1919 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1836, %1918, %1847 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1920 = vector.extract %1913[0] : f32 from vector<4xf32>
      %1921 = arith.maxnumf %1905, %1920 : f32
      %1922 = vector.extract %1913[1] : f32 from vector<4xf32>
      %1923 = arith.maxnumf %1921, %1922 : f32
      %1924 = vector.extract %1913[2] : f32 from vector<4xf32>
      %1925 = arith.maxnumf %1923, %1924 : f32
      %1926 = vector.extract %1913[3] : f32 from vector<4xf32>
      %1927 = arith.maxnumf %1925, %1926 : f32
      %1928 = vector.load %view_7[%953] : memref<4096xf16, 3>, vector<4xf16>
      %1929 = vector.load %view_7[%955] : memref<4096xf16, 3>, vector<4xf16>
      %1930 = vector.extract %1928[0] : f16 from vector<4xf16>
      %1931 = vector.extract %1928[1] : f16 from vector<4xf16>
      %1932 = vector.extract %1928[2] : f16 from vector<4xf16>
      %1933 = vector.extract %1928[3] : f16 from vector<4xf16>
      %1934 = vector.from_elements %1930, %1931, %1932, %1933 : vector<4xf16>
      %1935 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1838, %1934, %1919 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1936 = vector.extract %1929[0] : f16 from vector<4xf16>
      %1937 = vector.extract %1929[1] : f16 from vector<4xf16>
      %1938 = vector.extract %1929[2] : f16 from vector<4xf16>
      %1939 = vector.extract %1929[3] : f16 from vector<4xf16>
      %1940 = vector.from_elements %1936, %1937, %1938, %1939 : vector<4xf16>
      %1941 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1836, %1940, %1848 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1942 = vector.extract %1935[0] : f32 from vector<4xf32>
      %1943 = arith.maxnumf %1927, %1942 : f32
      %1944 = vector.extract %1935[1] : f32 from vector<4xf32>
      %1945 = arith.maxnumf %1943, %1944 : f32
      %1946 = vector.extract %1935[2] : f32 from vector<4xf32>
      %1947 = arith.maxnumf %1945, %1946 : f32
      %1948 = vector.extract %1935[3] : f32 from vector<4xf32>
      %1949 = arith.maxnumf %1947, %1948 : f32
      %1950 = vector.load %view_7[%973] : memref<4096xf16, 3>, vector<4xf16>
      %1951 = vector.load %view_7[%975] : memref<4096xf16, 3>, vector<4xf16>
      %1952 = vector.extract %1950[0] : f16 from vector<4xf16>
      %1953 = vector.extract %1950[1] : f16 from vector<4xf16>
      %1954 = vector.extract %1950[2] : f16 from vector<4xf16>
      %1955 = vector.extract %1950[3] : f16 from vector<4xf16>
      %1956 = vector.from_elements %1952, %1953, %1954, %1955 : vector<4xf16>
      %1957 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1838, %1956, %1941 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1958 = vector.extract %1951[0] : f16 from vector<4xf16>
      %1959 = vector.extract %1951[1] : f16 from vector<4xf16>
      %1960 = vector.extract %1951[2] : f16 from vector<4xf16>
      %1961 = vector.extract %1951[3] : f16 from vector<4xf16>
      %1962 = vector.from_elements %1958, %1959, %1960, %1961 : vector<4xf16>
      %1963 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1836, %1962, %1849 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1964 = vector.extract %1957[0] : f32 from vector<4xf32>
      %1965 = arith.maxnumf %1949, %1964 : f32
      %1966 = vector.extract %1957[1] : f32 from vector<4xf32>
      %1967 = arith.maxnumf %1965, %1966 : f32
      %1968 = vector.extract %1957[2] : f32 from vector<4xf32>
      %1969 = arith.maxnumf %1967, %1968 : f32
      %1970 = vector.extract %1957[3] : f32 from vector<4xf32>
      %1971 = arith.maxnumf %1969, %1970 : f32
      %1972 = vector.load %view_7[%993] : memref<4096xf16, 3>, vector<4xf16>
      %1973 = vector.load %view_7[%995] : memref<4096xf16, 3>, vector<4xf16>
      %1974 = vector.extract %1972[0] : f16 from vector<4xf16>
      %1975 = vector.extract %1972[1] : f16 from vector<4xf16>
      %1976 = vector.extract %1972[2] : f16 from vector<4xf16>
      %1977 = vector.extract %1972[3] : f16 from vector<4xf16>
      %1978 = vector.from_elements %1974, %1975, %1976, %1977 : vector<4xf16>
      %1979 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1838, %1978, %1963 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1980 = vector.extract %1973[0] : f16 from vector<4xf16>
      %1981 = vector.extract %1973[1] : f16 from vector<4xf16>
      %1982 = vector.extract %1973[2] : f16 from vector<4xf16>
      %1983 = vector.extract %1973[3] : f16 from vector<4xf16>
      %1984 = vector.from_elements %1980, %1981, %1982, %1983 : vector<4xf16>
      %1985 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1836, %1984, %1850 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %1986 = vector.extract %1979[0] : f32 from vector<4xf32>
      %1987 = arith.maxnumf %1971, %1986 : f32
      %1988 = vector.extract %1979[1] : f32 from vector<4xf32>
      %1989 = arith.maxnumf %1987, %1988 : f32
      %1990 = vector.extract %1979[2] : f32 from vector<4xf32>
      %1991 = arith.maxnumf %1989, %1990 : f32
      %1992 = vector.extract %1979[3] : f32 from vector<4xf32>
      %1993 = arith.maxnumf %1991, %1992 : f32
      %1994 = vector.load %view_7[%1013] : memref<4096xf16, 3>, vector<4xf16>
      %1995 = vector.load %view_7[%1015] : memref<4096xf16, 3>, vector<4xf16>
      %1996 = vector.extract %1994[0] : f16 from vector<4xf16>
      %1997 = vector.extract %1994[1] : f16 from vector<4xf16>
      %1998 = vector.extract %1994[2] : f16 from vector<4xf16>
      %1999 = vector.extract %1994[3] : f16 from vector<4xf16>
      %2000 = vector.from_elements %1996, %1997, %1998, %1999 : vector<4xf16>
      %2001 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1838, %2000, %1985 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2002 = vector.extract %1995[0] : f16 from vector<4xf16>
      %2003 = vector.extract %1995[1] : f16 from vector<4xf16>
      %2004 = vector.extract %1995[2] : f16 from vector<4xf16>
      %2005 = vector.extract %1995[3] : f16 from vector<4xf16>
      %2006 = vector.from_elements %2002, %2003, %2004, %2005 : vector<4xf16>
      %2007 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1836, %2006, %1851 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2008 = vector.extract %2001[0] : f32 from vector<4xf32>
      %2009 = arith.maxnumf %1993, %2008 : f32
      %2010 = vector.extract %2001[1] : f32 from vector<4xf32>
      %2011 = arith.maxnumf %2009, %2010 : f32
      %2012 = vector.extract %2001[2] : f32 from vector<4xf32>
      %2013 = arith.maxnumf %2011, %2012 : f32
      %2014 = vector.extract %2001[3] : f32 from vector<4xf32>
      %2015 = arith.maxnumf %2013, %2014 : f32
      %2016 = vector.load %view_7[%1033] : memref<4096xf16, 3>, vector<4xf16>
      %2017 = vector.load %view_7[%881] : memref<4096xf16, 3>, vector<4xf16>
      %2018 = vector.extract %2016[0] : f16 from vector<4xf16>
      %2019 = vector.extract %2016[1] : f16 from vector<4xf16>
      %2020 = vector.extract %2016[2] : f16 from vector<4xf16>
      %2021 = vector.extract %2016[3] : f16 from vector<4xf16>
      %2022 = vector.from_elements %2018, %2019, %2020, %2021 : vector<4xf16>
      %2023 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1838, %2022, %2007 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2024 = vector.extract %2017[0] : f16 from vector<4xf16>
      %2025 = vector.extract %2017[1] : f16 from vector<4xf16>
      %2026 = vector.extract %2017[2] : f16 from vector<4xf16>
      %2027 = vector.extract %2017[3] : f16 from vector<4xf16>
      %2028 = vector.from_elements %2024, %2025, %2026, %2027 : vector<4xf16>
      %2029 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1852, %2028, %1853 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2030 = vector.extract %2023[0] : f32 from vector<4xf32>
      %2031 = arith.maxnumf %2015, %2030 : f32
      %2032 = vector.extract %2023[1] : f32 from vector<4xf32>
      %2033 = arith.maxnumf %2031, %2032 : f32
      %2034 = vector.extract %2023[2] : f32 from vector<4xf32>
      %2035 = arith.maxnumf %2033, %2034 : f32
      %2036 = vector.extract %2023[3] : f32 from vector<4xf32>
      %2037 = arith.maxnumf %2035, %2036 : f32
      %2038 = vector.load %view_7[%893] : memref<4096xf16, 3>, vector<4xf16>
      %2039 = vector.load %view_7[%895] : memref<4096xf16, 3>, vector<4xf16>
      %2040 = vector.extract %2038[0] : f16 from vector<4xf16>
      %2041 = vector.extract %2038[1] : f16 from vector<4xf16>
      %2042 = vector.extract %2038[2] : f16 from vector<4xf16>
      %2043 = vector.extract %2038[3] : f16 from vector<4xf16>
      %2044 = vector.from_elements %2040, %2041, %2042, %2043 : vector<4xf16>
      %2045 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1854, %2044, %2029 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2046 = vector.extract %2039[0] : f16 from vector<4xf16>
      %2047 = vector.extract %2039[1] : f16 from vector<4xf16>
      %2048 = vector.extract %2039[2] : f16 from vector<4xf16>
      %2049 = vector.extract %2039[3] : f16 from vector<4xf16>
      %2050 = vector.from_elements %2046, %2047, %2048, %2049 : vector<4xf16>
      %2051 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1852, %2050, %1855 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2052 = vector.extract %2045[0] : f32 from vector<4xf32>
      %2053 = vector.extract %2045[1] : f32 from vector<4xf32>
      %2054 = vector.extract %2045[2] : f32 from vector<4xf32>
      %2055 = vector.extract %2045[3] : f32 from vector<4xf32>
      %2056 = vector.load %view_7[%913] : memref<4096xf16, 3>, vector<4xf16>
      %2057 = vector.load %view_7[%915] : memref<4096xf16, 3>, vector<4xf16>
      %2058 = vector.extract %2056[0] : f16 from vector<4xf16>
      %2059 = vector.extract %2056[1] : f16 from vector<4xf16>
      %2060 = vector.extract %2056[2] : f16 from vector<4xf16>
      %2061 = vector.extract %2056[3] : f16 from vector<4xf16>
      %2062 = vector.from_elements %2058, %2059, %2060, %2061 : vector<4xf16>
      %2063 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1854, %2062, %2051 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2064 = vector.extract %2057[0] : f16 from vector<4xf16>
      %2065 = vector.extract %2057[1] : f16 from vector<4xf16>
      %2066 = vector.extract %2057[2] : f16 from vector<4xf16>
      %2067 = vector.extract %2057[3] : f16 from vector<4xf16>
      %2068 = vector.from_elements %2064, %2065, %2066, %2067 : vector<4xf16>
      %2069 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1852, %2068, %1856 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2070 = vector.extract %2063[0] : f32 from vector<4xf32>
      %2071 = vector.extract %2063[1] : f32 from vector<4xf32>
      %2072 = vector.extract %2063[2] : f32 from vector<4xf32>
      %2073 = vector.extract %2063[3] : f32 from vector<4xf32>
      %2074 = vector.load %view_7[%933] : memref<4096xf16, 3>, vector<4xf16>
      %2075 = vector.load %view_7[%935] : memref<4096xf16, 3>, vector<4xf16>
      %2076 = vector.extract %2074[0] : f16 from vector<4xf16>
      %2077 = vector.extract %2074[1] : f16 from vector<4xf16>
      %2078 = vector.extract %2074[2] : f16 from vector<4xf16>
      %2079 = vector.extract %2074[3] : f16 from vector<4xf16>
      %2080 = vector.from_elements %2076, %2077, %2078, %2079 : vector<4xf16>
      %2081 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1854, %2080, %2069 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2082 = vector.extract %2075[0] : f16 from vector<4xf16>
      %2083 = vector.extract %2075[1] : f16 from vector<4xf16>
      %2084 = vector.extract %2075[2] : f16 from vector<4xf16>
      %2085 = vector.extract %2075[3] : f16 from vector<4xf16>
      %2086 = vector.from_elements %2082, %2083, %2084, %2085 : vector<4xf16>
      %2087 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1852, %2086, %1857 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2088 = vector.extract %2081[0] : f32 from vector<4xf32>
      %2089 = vector.extract %2081[1] : f32 from vector<4xf32>
      %2090 = vector.extract %2081[2] : f32 from vector<4xf32>
      %2091 = vector.extract %2081[3] : f32 from vector<4xf32>
      %2092 = vector.load %view_7[%953] : memref<4096xf16, 3>, vector<4xf16>
      %2093 = vector.load %view_7[%955] : memref<4096xf16, 3>, vector<4xf16>
      %2094 = vector.extract %2092[0] : f16 from vector<4xf16>
      %2095 = vector.extract %2092[1] : f16 from vector<4xf16>
      %2096 = vector.extract %2092[2] : f16 from vector<4xf16>
      %2097 = vector.extract %2092[3] : f16 from vector<4xf16>
      %2098 = vector.from_elements %2094, %2095, %2096, %2097 : vector<4xf16>
      %2099 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1854, %2098, %2087 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2100 = vector.extract %2093[0] : f16 from vector<4xf16>
      %2101 = vector.extract %2093[1] : f16 from vector<4xf16>
      %2102 = vector.extract %2093[2] : f16 from vector<4xf16>
      %2103 = vector.extract %2093[3] : f16 from vector<4xf16>
      %2104 = vector.from_elements %2100, %2101, %2102, %2103 : vector<4xf16>
      %2105 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1852, %2104, %1858 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2106 = vector.extract %2099[0] : f32 from vector<4xf32>
      %2107 = vector.extract %2099[1] : f32 from vector<4xf32>
      %2108 = vector.extract %2099[2] : f32 from vector<4xf32>
      %2109 = vector.extract %2099[3] : f32 from vector<4xf32>
      %2110 = vector.load %view_7[%973] : memref<4096xf16, 3>, vector<4xf16>
      %2111 = vector.load %view_7[%975] : memref<4096xf16, 3>, vector<4xf16>
      %2112 = vector.extract %2110[0] : f16 from vector<4xf16>
      %2113 = vector.extract %2110[1] : f16 from vector<4xf16>
      %2114 = vector.extract %2110[2] : f16 from vector<4xf16>
      %2115 = vector.extract %2110[3] : f16 from vector<4xf16>
      %2116 = vector.from_elements %2112, %2113, %2114, %2115 : vector<4xf16>
      %2117 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1854, %2116, %2105 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2118 = vector.extract %2111[0] : f16 from vector<4xf16>
      %2119 = vector.extract %2111[1] : f16 from vector<4xf16>
      %2120 = vector.extract %2111[2] : f16 from vector<4xf16>
      %2121 = vector.extract %2111[3] : f16 from vector<4xf16>
      %2122 = vector.from_elements %2118, %2119, %2120, %2121 : vector<4xf16>
      %2123 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1852, %2122, %1859 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2124 = vector.extract %2117[0] : f32 from vector<4xf32>
      %2125 = vector.extract %2117[1] : f32 from vector<4xf32>
      %2126 = vector.extract %2117[2] : f32 from vector<4xf32>
      %2127 = vector.extract %2117[3] : f32 from vector<4xf32>
      %2128 = vector.load %view_7[%993] : memref<4096xf16, 3>, vector<4xf16>
      %2129 = vector.load %view_7[%995] : memref<4096xf16, 3>, vector<4xf16>
      %2130 = vector.extract %2128[0] : f16 from vector<4xf16>
      %2131 = vector.extract %2128[1] : f16 from vector<4xf16>
      %2132 = vector.extract %2128[2] : f16 from vector<4xf16>
      %2133 = vector.extract %2128[3] : f16 from vector<4xf16>
      %2134 = vector.from_elements %2130, %2131, %2132, %2133 : vector<4xf16>
      %2135 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1854, %2134, %2123 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2136 = vector.extract %2129[0] : f16 from vector<4xf16>
      %2137 = vector.extract %2129[1] : f16 from vector<4xf16>
      %2138 = vector.extract %2129[2] : f16 from vector<4xf16>
      %2139 = vector.extract %2129[3] : f16 from vector<4xf16>
      %2140 = vector.from_elements %2136, %2137, %2138, %2139 : vector<4xf16>
      %2141 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1852, %2140, %1860 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2142 = vector.extract %2135[0] : f32 from vector<4xf32>
      %2143 = vector.extract %2135[1] : f32 from vector<4xf32>
      %2144 = vector.extract %2135[2] : f32 from vector<4xf32>
      %2145 = vector.extract %2135[3] : f32 from vector<4xf32>
      %2146 = vector.load %view_7[%1013] : memref<4096xf16, 3>, vector<4xf16>
      %2147 = vector.load %view_7[%1015] : memref<4096xf16, 3>, vector<4xf16>
      %2148 = vector.extract %2146[0] : f16 from vector<4xf16>
      %2149 = vector.extract %2146[1] : f16 from vector<4xf16>
      %2150 = vector.extract %2146[2] : f16 from vector<4xf16>
      %2151 = vector.extract %2146[3] : f16 from vector<4xf16>
      %2152 = vector.from_elements %2148, %2149, %2150, %2151 : vector<4xf16>
      %2153 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1854, %2152, %2141 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2154 = vector.extract %2147[0] : f16 from vector<4xf16>
      %2155 = vector.extract %2147[1] : f16 from vector<4xf16>
      %2156 = vector.extract %2147[2] : f16 from vector<4xf16>
      %2157 = vector.extract %2147[3] : f16 from vector<4xf16>
      %2158 = vector.from_elements %2154, %2155, %2156, %2157 : vector<4xf16>
      %2159 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1852, %2158, %1861 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2160 = vector.extract %2153[0] : f32 from vector<4xf32>
      %2161 = vector.extract %2153[1] : f32 from vector<4xf32>
      %2162 = vector.extract %2153[2] : f32 from vector<4xf32>
      %2163 = vector.extract %2153[3] : f32 from vector<4xf32>
      %2164 = vector.load %view_7[%1033] : memref<4096xf16, 3>, vector<4xf16>
      %2165 = vector.extract %2164[0] : f16 from vector<4xf16>
      %2166 = vector.extract %2164[1] : f16 from vector<4xf16>
      %2167 = vector.extract %2164[2] : f16 from vector<4xf16>
      %2168 = vector.extract %2164[3] : f16 from vector<4xf16>
      %2169 = vector.from_elements %2165, %2166, %2167, %2168 : vector<4xf16>
      %2170 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %1854, %2169, %2159 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2171 = vector.extract %2170[0] : f32 from vector<4xf32>
      %2172 = vector.extract %2170[1] : f32 from vector<4xf32>
      %2173 = vector.extract %2170[2] : f32 from vector<4xf32>
      %2174 = vector.extract %2170[3] : f32 from vector<4xf32>
      %2175 = llvm.bitcast %2037 : f32 to i32
      %2176 = rocdl.ds_bpermute %10, %2175 : (i32, i32) -> i32
      %2177 = llvm.bitcast %2176 : i32 to f32
      %2178 = arith.maxnumf %2037, %2177 : f32
      %2179 = llvm.bitcast %2178 : f32 to i32
      %2180 = rocdl.ds_bpermute %14, %2179 : (i32, i32) -> i32
      %2181 = llvm.bitcast %2180 : i32 to f32
      %2182 = arith.maxnumf %2178, %2181 : f32
      %2183 = arith.maxnumf %2052, %cst_3 : f32
      %2184 = arith.maxnumf %2183, %2053 : f32
      %2185 = arith.maxnumf %2184, %2054 : f32
      %2186 = arith.maxnumf %2185, %2055 : f32
      %2187 = arith.maxnumf %2186, %2070 : f32
      %2188 = arith.maxnumf %2187, %2071 : f32
      %2189 = arith.maxnumf %2188, %2072 : f32
      %2190 = arith.maxnumf %2189, %2073 : f32
      %2191 = arith.maxnumf %2190, %2088 : f32
      %2192 = arith.maxnumf %2191, %2089 : f32
      %2193 = arith.maxnumf %2192, %2090 : f32
      %2194 = arith.maxnumf %2193, %2091 : f32
      %2195 = arith.maxnumf %2194, %2106 : f32
      %2196 = arith.maxnumf %2195, %2107 : f32
      %2197 = arith.maxnumf %2196, %2108 : f32
      %2198 = arith.maxnumf %2197, %2109 : f32
      %2199 = arith.maxnumf %2198, %2124 : f32
      %2200 = arith.maxnumf %2199, %2125 : f32
      %2201 = arith.maxnumf %2200, %2126 : f32
      %2202 = arith.maxnumf %2201, %2127 : f32
      %2203 = arith.maxnumf %2202, %2142 : f32
      %2204 = arith.maxnumf %2203, %2143 : f32
      %2205 = arith.maxnumf %2204, %2144 : f32
      %2206 = arith.maxnumf %2205, %2145 : f32
      %2207 = arith.maxnumf %2206, %2160 : f32
      %2208 = arith.maxnumf %2207, %2161 : f32
      %2209 = arith.maxnumf %2208, %2162 : f32
      %2210 = arith.maxnumf %2209, %2163 : f32
      %2211 = arith.maxnumf %2210, %2171 : f32
      %2212 = arith.maxnumf %2211, %2172 : f32
      %2213 = arith.maxnumf %2212, %2173 : f32
      %2214 = arith.maxnumf %2213, %2174 : f32
      %2215 = llvm.bitcast %2214 : f32 to i32
      %2216 = rocdl.ds_bpermute %10, %2215 : (i32, i32) -> i32
      %2217 = llvm.bitcast %2216 : i32 to f32
      %2218 = arith.maxnumf %2214, %2217 : f32
      %2219 = llvm.bitcast %2218 : f32 to i32
      %2220 = rocdl.ds_bpermute %14, %2219 : (i32, i32) -> i32
      %2221 = llvm.bitcast %2220 : i32 to f32
      %2222 = arith.maxnumf %2218, %2221 : f32
      %2223 = vector.extract %arg7[0, 0] : f32 from vector<2x1xf32>
      %2224 = vector.broadcast %2182 : f32 to vector<1xf32>
      %2225 = vector.broadcast %2223 : f32 to vector<1xf32>
      %2226 = arith.maxnumf %2224, %2225 {origin = "frisk.max", tiled = true} : vector<1xf32>
      %2227 = vector.extract %2226[0] : f32 from vector<1xf32>
      %2228 = vector.extract %arg7[1, 0] : f32 from vector<2x1xf32>
      %2229 = vector.broadcast %2222 : f32 to vector<1xf32>
      %2230 = vector.broadcast %2228 : f32 to vector<1xf32>
      %2231 = arith.maxnumf %2229, %2230 {origin = "frisk.max", tiled = true} : vector<1xf32>
      %2232 = vector.extract %2231[0] : f32 from vector<1xf32>
      %2233 = vector.from_elements %1876, %1878, %1880, %1882 : vector<4xf32>
      %2234 = vector.broadcast %2227 : f32 to vector<4xf32>
      %2235 = arith.subf %2233, %2234 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2236 = vector.extract %2235[0] : f32 from vector<4xf32>
      %2237 = vector.extract %2235[1] : f32 from vector<4xf32>
      %2238 = vector.extract %2235[2] : f32 from vector<4xf32>
      %2239 = vector.extract %2235[3] : f32 from vector<4xf32>
      %2240 = vector.from_elements %1898, %1900, %1902, %1904 : vector<4xf32>
      %2241 = arith.subf %2240, %2234 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2242 = vector.extract %2241[0] : f32 from vector<4xf32>
      %2243 = vector.extract %2241[1] : f32 from vector<4xf32>
      %2244 = vector.extract %2241[2] : f32 from vector<4xf32>
      %2245 = vector.extract %2241[3] : f32 from vector<4xf32>
      %2246 = vector.from_elements %1920, %1922, %1924, %1926 : vector<4xf32>
      %2247 = arith.subf %2246, %2234 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2248 = vector.extract %2247[0] : f32 from vector<4xf32>
      %2249 = vector.extract %2247[1] : f32 from vector<4xf32>
      %2250 = vector.extract %2247[2] : f32 from vector<4xf32>
      %2251 = vector.extract %2247[3] : f32 from vector<4xf32>
      %2252 = vector.from_elements %1942, %1944, %1946, %1948 : vector<4xf32>
      %2253 = arith.subf %2252, %2234 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2254 = vector.extract %2253[0] : f32 from vector<4xf32>
      %2255 = vector.extract %2253[1] : f32 from vector<4xf32>
      %2256 = vector.extract %2253[2] : f32 from vector<4xf32>
      %2257 = vector.extract %2253[3] : f32 from vector<4xf32>
      %2258 = vector.from_elements %1964, %1966, %1968, %1970 : vector<4xf32>
      %2259 = arith.subf %2258, %2234 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2260 = vector.extract %2259[0] : f32 from vector<4xf32>
      %2261 = vector.extract %2259[1] : f32 from vector<4xf32>
      %2262 = vector.extract %2259[2] : f32 from vector<4xf32>
      %2263 = vector.extract %2259[3] : f32 from vector<4xf32>
      %2264 = vector.from_elements %1986, %1988, %1990, %1992 : vector<4xf32>
      %2265 = arith.subf %2264, %2234 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2266 = vector.extract %2265[0] : f32 from vector<4xf32>
      %2267 = vector.extract %2265[1] : f32 from vector<4xf32>
      %2268 = vector.extract %2265[2] : f32 from vector<4xf32>
      %2269 = vector.extract %2265[3] : f32 from vector<4xf32>
      %2270 = vector.from_elements %2008, %2010, %2012, %2014 : vector<4xf32>
      %2271 = arith.subf %2270, %2234 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2272 = vector.extract %2271[0] : f32 from vector<4xf32>
      %2273 = vector.extract %2271[1] : f32 from vector<4xf32>
      %2274 = vector.extract %2271[2] : f32 from vector<4xf32>
      %2275 = vector.extract %2271[3] : f32 from vector<4xf32>
      %2276 = vector.from_elements %2030, %2032, %2034, %2036 : vector<4xf32>
      %2277 = arith.subf %2276, %2234 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2278 = vector.extract %2277[0] : f32 from vector<4xf32>
      %2279 = vector.extract %2277[1] : f32 from vector<4xf32>
      %2280 = vector.extract %2277[2] : f32 from vector<4xf32>
      %2281 = vector.extract %2277[3] : f32 from vector<4xf32>
      %2282 = vector.from_elements %2052, %2053, %2054, %2055 : vector<4xf32>
      %2283 = vector.broadcast %2232 : f32 to vector<4xf32>
      %2284 = arith.subf %2282, %2283 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2285 = vector.extract %2284[0] : f32 from vector<4xf32>
      %2286 = vector.extract %2284[1] : f32 from vector<4xf32>
      %2287 = vector.extract %2284[2] : f32 from vector<4xf32>
      %2288 = vector.extract %2284[3] : f32 from vector<4xf32>
      %2289 = vector.from_elements %2070, %2071, %2072, %2073 : vector<4xf32>
      %2290 = arith.subf %2289, %2283 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2291 = vector.extract %2290[0] : f32 from vector<4xf32>
      %2292 = vector.extract %2290[1] : f32 from vector<4xf32>
      %2293 = vector.extract %2290[2] : f32 from vector<4xf32>
      %2294 = vector.extract %2290[3] : f32 from vector<4xf32>
      %2295 = vector.from_elements %2088, %2089, %2090, %2091 : vector<4xf32>
      %2296 = arith.subf %2295, %2283 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2297 = vector.extract %2296[0] : f32 from vector<4xf32>
      %2298 = vector.extract %2296[1] : f32 from vector<4xf32>
      %2299 = vector.extract %2296[2] : f32 from vector<4xf32>
      %2300 = vector.extract %2296[3] : f32 from vector<4xf32>
      %2301 = vector.from_elements %2106, %2107, %2108, %2109 : vector<4xf32>
      %2302 = arith.subf %2301, %2283 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2303 = vector.extract %2302[0] : f32 from vector<4xf32>
      %2304 = vector.extract %2302[1] : f32 from vector<4xf32>
      %2305 = vector.extract %2302[2] : f32 from vector<4xf32>
      %2306 = vector.extract %2302[3] : f32 from vector<4xf32>
      %2307 = vector.from_elements %2124, %2125, %2126, %2127 : vector<4xf32>
      %2308 = arith.subf %2307, %2283 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2309 = vector.extract %2308[0] : f32 from vector<4xf32>
      %2310 = vector.extract %2308[1] : f32 from vector<4xf32>
      %2311 = vector.extract %2308[2] : f32 from vector<4xf32>
      %2312 = vector.extract %2308[3] : f32 from vector<4xf32>
      %2313 = vector.from_elements %2142, %2143, %2144, %2145 : vector<4xf32>
      %2314 = arith.subf %2313, %2283 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2315 = vector.extract %2314[0] : f32 from vector<4xf32>
      %2316 = vector.extract %2314[1] : f32 from vector<4xf32>
      %2317 = vector.extract %2314[2] : f32 from vector<4xf32>
      %2318 = vector.extract %2314[3] : f32 from vector<4xf32>
      %2319 = vector.from_elements %2160, %2161, %2162, %2163 : vector<4xf32>
      %2320 = arith.subf %2319, %2283 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2321 = vector.extract %2320[0] : f32 from vector<4xf32>
      %2322 = vector.extract %2320[1] : f32 from vector<4xf32>
      %2323 = vector.extract %2320[2] : f32 from vector<4xf32>
      %2324 = vector.extract %2320[3] : f32 from vector<4xf32>
      %2325 = vector.from_elements %2171, %2172, %2173, %2174 : vector<4xf32>
      %2326 = arith.subf %2325, %2283 {origin = "frisk.sub", tiled = true} : vector<4xf32>
      %2327 = vector.extract %2326[0] : f32 from vector<4xf32>
      %2328 = vector.extract %2326[1] : f32 from vector<4xf32>
      %2329 = vector.extract %2326[2] : f32 from vector<4xf32>
      %2330 = vector.extract %2326[3] : f32 from vector<4xf32>
      %2331 = vector.broadcast %2227 : f32 to vector<1xf32>
      %2332 = arith.subf %2225, %2331 {origin = "frisk.sub", tiled = true} : vector<1xf32>
      %2333 = math.exp2 %2332 {tiled = true} : vector<1xf32>
      %2334 = vector.extract %2333[0] : f32 from vector<1xf32>
      %2335 = vector.extract %arg6[0, 0] : f32 from vector<2x1xf32>
      %2336 = vector.broadcast %2335 : f32 to vector<1xf32>
      %2337 = arith.mulf %2336, %2333 {origin = "frisk.mul", tiled = true} : vector<1xf32>
      %2338 = vector.extract %2337[0] : f32 from vector<1xf32>
      %2339 = vector.broadcast %2232 : f32 to vector<1xf32>
      %2340 = arith.subf %2230, %2339 {origin = "frisk.sub", tiled = true} : vector<1xf32>
      %2341 = math.exp2 %2340 {tiled = true} : vector<1xf32>
      %2342 = vector.extract %2341[0] : f32 from vector<1xf32>
      %2343 = vector.extract %arg6[1, 0] : f32 from vector<2x1xf32>
      %2344 = vector.broadcast %2343 : f32 to vector<1xf32>
      %2345 = arith.mulf %2344, %2341 {origin = "frisk.mul", tiled = true} : vector<1xf32>
      %2346 = vector.extract %2345[0] : f32 from vector<1xf32>
      %2347 = vector.from_elements %2236, %2237, %2238, %2239 : vector<4xf32>
      %2348 = math.exp2 %2347 {tiled = true} : vector<4xf32>
      %2349 = vector.extract %2348[0] : f32 from vector<4xf32>
      %2350 = vector.extract %2348[1] : f32 from vector<4xf32>
      %2351 = vector.extract %2348[2] : f32 from vector<4xf32>
      %2352 = vector.extract %2348[3] : f32 from vector<4xf32>
      %2353 = vector.from_elements %2242, %2243, %2244, %2245 : vector<4xf32>
      %2354 = math.exp2 %2353 {tiled = true} : vector<4xf32>
      %2355 = vector.extract %2354[0] : f32 from vector<4xf32>
      %2356 = vector.extract %2354[1] : f32 from vector<4xf32>
      %2357 = vector.extract %2354[2] : f32 from vector<4xf32>
      %2358 = vector.extract %2354[3] : f32 from vector<4xf32>
      %2359 = vector.from_elements %2248, %2249, %2250, %2251 : vector<4xf32>
      %2360 = math.exp2 %2359 {tiled = true} : vector<4xf32>
      %2361 = vector.extract %2360[0] : f32 from vector<4xf32>
      %2362 = vector.extract %2360[1] : f32 from vector<4xf32>
      %2363 = vector.extract %2360[2] : f32 from vector<4xf32>
      %2364 = vector.extract %2360[3] : f32 from vector<4xf32>
      %2365 = vector.from_elements %2254, %2255, %2256, %2257 : vector<4xf32>
      %2366 = math.exp2 %2365 {tiled = true} : vector<4xf32>
      %2367 = vector.extract %2366[0] : f32 from vector<4xf32>
      %2368 = vector.extract %2366[1] : f32 from vector<4xf32>
      %2369 = vector.extract %2366[2] : f32 from vector<4xf32>
      %2370 = vector.extract %2366[3] : f32 from vector<4xf32>
      %2371 = vector.from_elements %2260, %2261, %2262, %2263 : vector<4xf32>
      %2372 = math.exp2 %2371 {tiled = true} : vector<4xf32>
      %2373 = vector.extract %2372[0] : f32 from vector<4xf32>
      %2374 = vector.extract %2372[1] : f32 from vector<4xf32>
      %2375 = vector.extract %2372[2] : f32 from vector<4xf32>
      %2376 = vector.extract %2372[3] : f32 from vector<4xf32>
      %2377 = vector.from_elements %2266, %2267, %2268, %2269 : vector<4xf32>
      %2378 = math.exp2 %2377 {tiled = true} : vector<4xf32>
      %2379 = vector.extract %2378[0] : f32 from vector<4xf32>
      %2380 = vector.extract %2378[1] : f32 from vector<4xf32>
      %2381 = vector.extract %2378[2] : f32 from vector<4xf32>
      %2382 = vector.extract %2378[3] : f32 from vector<4xf32>
      %2383 = vector.from_elements %2272, %2273, %2274, %2275 : vector<4xf32>
      %2384 = math.exp2 %2383 {tiled = true} : vector<4xf32>
      %2385 = vector.extract %2384[0] : f32 from vector<4xf32>
      %2386 = vector.extract %2384[1] : f32 from vector<4xf32>
      %2387 = vector.extract %2384[2] : f32 from vector<4xf32>
      %2388 = vector.extract %2384[3] : f32 from vector<4xf32>
      %2389 = vector.from_elements %2278, %2279, %2280, %2281 : vector<4xf32>
      %2390 = math.exp2 %2389 {tiled = true} : vector<4xf32>
      %2391 = vector.extract %2390[0] : f32 from vector<4xf32>
      %2392 = vector.extract %2390[1] : f32 from vector<4xf32>
      %2393 = vector.extract %2390[2] : f32 from vector<4xf32>
      %2394 = vector.extract %2390[3] : f32 from vector<4xf32>
      %2395 = vector.from_elements %2285, %2286, %2287, %2288 : vector<4xf32>
      %2396 = math.exp2 %2395 {tiled = true} : vector<4xf32>
      %2397 = vector.extract %2396[0] : f32 from vector<4xf32>
      %2398 = vector.extract %2396[1] : f32 from vector<4xf32>
      %2399 = vector.extract %2396[2] : f32 from vector<4xf32>
      %2400 = vector.extract %2396[3] : f32 from vector<4xf32>
      %2401 = vector.from_elements %2291, %2292, %2293, %2294 : vector<4xf32>
      %2402 = math.exp2 %2401 {tiled = true} : vector<4xf32>
      %2403 = vector.extract %2402[0] : f32 from vector<4xf32>
      %2404 = vector.extract %2402[1] : f32 from vector<4xf32>
      %2405 = vector.extract %2402[2] : f32 from vector<4xf32>
      %2406 = vector.extract %2402[3] : f32 from vector<4xf32>
      %2407 = vector.from_elements %2297, %2298, %2299, %2300 : vector<4xf32>
      %2408 = math.exp2 %2407 {tiled = true} : vector<4xf32>
      %2409 = vector.extract %2408[0] : f32 from vector<4xf32>
      %2410 = vector.extract %2408[1] : f32 from vector<4xf32>
      %2411 = vector.extract %2408[2] : f32 from vector<4xf32>
      %2412 = vector.extract %2408[3] : f32 from vector<4xf32>
      %2413 = vector.from_elements %2303, %2304, %2305, %2306 : vector<4xf32>
      %2414 = math.exp2 %2413 {tiled = true} : vector<4xf32>
      %2415 = vector.extract %2414[0] : f32 from vector<4xf32>
      %2416 = vector.extract %2414[1] : f32 from vector<4xf32>
      %2417 = vector.extract %2414[2] : f32 from vector<4xf32>
      %2418 = vector.extract %2414[3] : f32 from vector<4xf32>
      %2419 = vector.from_elements %2309, %2310, %2311, %2312 : vector<4xf32>
      %2420 = math.exp2 %2419 {tiled = true} : vector<4xf32>
      %2421 = vector.extract %2420[0] : f32 from vector<4xf32>
      %2422 = vector.extract %2420[1] : f32 from vector<4xf32>
      %2423 = vector.extract %2420[2] : f32 from vector<4xf32>
      %2424 = vector.extract %2420[3] : f32 from vector<4xf32>
      %2425 = vector.from_elements %2315, %2316, %2317, %2318 : vector<4xf32>
      %2426 = math.exp2 %2425 {tiled = true} : vector<4xf32>
      %2427 = vector.extract %2426[0] : f32 from vector<4xf32>
      %2428 = vector.extract %2426[1] : f32 from vector<4xf32>
      %2429 = vector.extract %2426[2] : f32 from vector<4xf32>
      %2430 = vector.extract %2426[3] : f32 from vector<4xf32>
      %2431 = vector.from_elements %2321, %2322, %2323, %2324 : vector<4xf32>
      %2432 = math.exp2 %2431 {tiled = true} : vector<4xf32>
      %2433 = vector.extract %2432[0] : f32 from vector<4xf32>
      %2434 = vector.extract %2432[1] : f32 from vector<4xf32>
      %2435 = vector.extract %2432[2] : f32 from vector<4xf32>
      %2436 = vector.extract %2432[3] : f32 from vector<4xf32>
      %2437 = vector.from_elements %2327, %2328, %2329, %2330 : vector<4xf32>
      %2438 = math.exp2 %2437 {tiled = true} : vector<4xf32>
      %2439 = vector.extract %2438[0] : f32 from vector<4xf32>
      %2440 = vector.extract %2438[1] : f32 from vector<4xf32>
      %2441 = vector.extract %2438[2] : f32 from vector<4xf32>
      %2442 = vector.extract %2438[3] : f32 from vector<4xf32>
      %2443 = arith.addf %2349, %cst_6 : f32
      %2444 = arith.addf %2443, %2350 : f32
      %2445 = arith.addf %2444, %2351 : f32
      %2446 = arith.addf %2445, %2352 : f32
      %2447 = arith.addf %2446, %2355 : f32
      %2448 = arith.addf %2447, %2356 : f32
      %2449 = arith.addf %2448, %2357 : f32
      %2450 = arith.addf %2449, %2358 : f32
      %2451 = arith.addf %2450, %2361 : f32
      %2452 = arith.addf %2451, %2362 : f32
      %2453 = arith.addf %2452, %2363 : f32
      %2454 = arith.addf %2453, %2364 : f32
      %2455 = arith.addf %2454, %2367 : f32
      %2456 = arith.addf %2455, %2368 : f32
      %2457 = arith.addf %2456, %2369 : f32
      %2458 = arith.addf %2457, %2370 : f32
      %2459 = arith.addf %2458, %2373 : f32
      %2460 = arith.addf %2459, %2374 : f32
      %2461 = arith.addf %2460, %2375 : f32
      %2462 = arith.addf %2461, %2376 : f32
      %2463 = arith.addf %2462, %2379 : f32
      %2464 = arith.addf %2463, %2380 : f32
      %2465 = arith.addf %2464, %2381 : f32
      %2466 = arith.addf %2465, %2382 : f32
      %2467 = arith.addf %2466, %2385 : f32
      %2468 = arith.addf %2467, %2386 : f32
      %2469 = arith.addf %2468, %2387 : f32
      %2470 = arith.addf %2469, %2388 : f32
      %2471 = arith.addf %2470, %2391 : f32
      %2472 = arith.addf %2471, %2392 : f32
      %2473 = arith.addf %2472, %2393 : f32
      %2474 = arith.addf %2473, %2394 : f32
      %2475 = llvm.bitcast %2474 : f32 to i32
      %2476 = rocdl.ds_bpermute %10, %2475 : (i32, i32) -> i32
      %2477 = llvm.bitcast %2476 : i32 to f32
      %2478 = arith.addf %2474, %2477 : f32
      %2479 = llvm.bitcast %2478 : f32 to i32
      %2480 = rocdl.ds_bpermute %14, %2479 : (i32, i32) -> i32
      %2481 = llvm.bitcast %2480 : i32 to f32
      %2482 = arith.addf %2478, %2481 : f32
      %2483 = arith.addf %2397, %cst_6 : f32
      %2484 = arith.addf %2483, %2398 : f32
      %2485 = arith.addf %2484, %2399 : f32
      %2486 = arith.addf %2485, %2400 : f32
      %2487 = arith.addf %2486, %2403 : f32
      %2488 = arith.addf %2487, %2404 : f32
      %2489 = arith.addf %2488, %2405 : f32
      %2490 = arith.addf %2489, %2406 : f32
      %2491 = arith.addf %2490, %2409 : f32
      %2492 = arith.addf %2491, %2410 : f32
      %2493 = arith.addf %2492, %2411 : f32
      %2494 = arith.addf %2493, %2412 : f32
      %2495 = arith.addf %2494, %2415 : f32
      %2496 = arith.addf %2495, %2416 : f32
      %2497 = arith.addf %2496, %2417 : f32
      %2498 = arith.addf %2497, %2418 : f32
      %2499 = arith.addf %2498, %2421 : f32
      %2500 = arith.addf %2499, %2422 : f32
      %2501 = arith.addf %2500, %2423 : f32
      %2502 = arith.addf %2501, %2424 : f32
      %2503 = arith.addf %2502, %2427 : f32
      %2504 = arith.addf %2503, %2428 : f32
      %2505 = arith.addf %2504, %2429 : f32
      %2506 = arith.addf %2505, %2430 : f32
      %2507 = arith.addf %2506, %2433 : f32
      %2508 = arith.addf %2507, %2434 : f32
      %2509 = arith.addf %2508, %2435 : f32
      %2510 = arith.addf %2509, %2436 : f32
      %2511 = arith.addf %2510, %2439 : f32
      %2512 = arith.addf %2511, %2440 : f32
      %2513 = arith.addf %2512, %2441 : f32
      %2514 = arith.addf %2513, %2442 : f32
      %2515 = llvm.bitcast %2514 : f32 to i32
      %2516 = rocdl.ds_bpermute %10, %2515 : (i32, i32) -> i32
      %2517 = llvm.bitcast %2516 : i32 to f32
      %2518 = arith.addf %2514, %2517 : f32
      %2519 = llvm.bitcast %2518 : f32 to i32
      %2520 = rocdl.ds_bpermute %14, %2519 : (i32, i32) -> i32
      %2521 = llvm.bitcast %2520 : i32 to f32
      %2522 = arith.addf %2518, %2521 : f32
      %2523 = vector.broadcast %2482 : f32 to vector<1xf32>
      %2524 = vector.broadcast %2338 : f32 to vector<1xf32>
      %2525 = arith.addf %2523, %2524 {origin = "frisk.add", tiled = true} : vector<1xf32>
      %2526 = vector.extract %2525[0] : f32 from vector<1xf32>
      %2527 = vector.broadcast %2522 : f32 to vector<1xf32>
      %2528 = vector.broadcast %2346 : f32 to vector<1xf32>
      %2529 = arith.addf %2527, %2528 {origin = "frisk.add", tiled = true} : vector<1xf32>
      %2530 = vector.extract %2529[0] : f32 from vector<1xf32>
      %2531 = vector.from_elements %2349, %2350, %2351, %2352 : vector<4xf32>
      %2532 = arith.truncf %2531 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2533 = vector.extract %2532[0] : f16 from vector<4xf16>
      %2534 = vector.extract %2532[1] : f16 from vector<4xf16>
      %2535 = vector.extract %2532[2] : f16 from vector<4xf16>
      %2536 = vector.extract %2532[3] : f16 from vector<4xf16>
      %2537 = vector.from_elements %2355, %2356, %2357, %2358 : vector<4xf32>
      %2538 = arith.truncf %2537 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2539 = vector.extract %2538[0] : f16 from vector<4xf16>
      %2540 = vector.extract %2538[1] : f16 from vector<4xf16>
      %2541 = vector.extract %2538[2] : f16 from vector<4xf16>
      %2542 = vector.extract %2538[3] : f16 from vector<4xf16>
      %2543 = vector.from_elements %2361, %2362, %2363, %2364 : vector<4xf32>
      %2544 = arith.truncf %2543 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2545 = vector.extract %2544[0] : f16 from vector<4xf16>
      %2546 = vector.extract %2544[1] : f16 from vector<4xf16>
      %2547 = vector.extract %2544[2] : f16 from vector<4xf16>
      %2548 = vector.extract %2544[3] : f16 from vector<4xf16>
      %2549 = vector.from_elements %2367, %2368, %2369, %2370 : vector<4xf32>
      %2550 = arith.truncf %2549 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2551 = vector.extract %2550[0] : f16 from vector<4xf16>
      %2552 = vector.extract %2550[1] : f16 from vector<4xf16>
      %2553 = vector.extract %2550[2] : f16 from vector<4xf16>
      %2554 = vector.extract %2550[3] : f16 from vector<4xf16>
      %2555 = vector.from_elements %2373, %2374, %2375, %2376 : vector<4xf32>
      %2556 = arith.truncf %2555 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2557 = vector.extract %2556[0] : f16 from vector<4xf16>
      %2558 = vector.extract %2556[1] : f16 from vector<4xf16>
      %2559 = vector.extract %2556[2] : f16 from vector<4xf16>
      %2560 = vector.extract %2556[3] : f16 from vector<4xf16>
      %2561 = vector.from_elements %2379, %2380, %2381, %2382 : vector<4xf32>
      %2562 = arith.truncf %2561 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2563 = vector.extract %2562[0] : f16 from vector<4xf16>
      %2564 = vector.extract %2562[1] : f16 from vector<4xf16>
      %2565 = vector.extract %2562[2] : f16 from vector<4xf16>
      %2566 = vector.extract %2562[3] : f16 from vector<4xf16>
      %2567 = vector.from_elements %2385, %2386, %2387, %2388 : vector<4xf32>
      %2568 = arith.truncf %2567 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2569 = vector.extract %2568[0] : f16 from vector<4xf16>
      %2570 = vector.extract %2568[1] : f16 from vector<4xf16>
      %2571 = vector.extract %2568[2] : f16 from vector<4xf16>
      %2572 = vector.extract %2568[3] : f16 from vector<4xf16>
      %2573 = vector.from_elements %2391, %2392, %2393, %2394 : vector<4xf32>
      %2574 = arith.truncf %2573 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2575 = vector.extract %2574[0] : f16 from vector<4xf16>
      %2576 = vector.extract %2574[1] : f16 from vector<4xf16>
      %2577 = vector.extract %2574[2] : f16 from vector<4xf16>
      %2578 = vector.extract %2574[3] : f16 from vector<4xf16>
      %2579 = vector.from_elements %2397, %2398, %2399, %2400 : vector<4xf32>
      %2580 = arith.truncf %2579 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2581 = vector.extract %2580[0] : f16 from vector<4xf16>
      %2582 = vector.extract %2580[1] : f16 from vector<4xf16>
      %2583 = vector.extract %2580[2] : f16 from vector<4xf16>
      %2584 = vector.extract %2580[3] : f16 from vector<4xf16>
      %2585 = vector.from_elements %2403, %2404, %2405, %2406 : vector<4xf32>
      %2586 = arith.truncf %2585 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2587 = vector.extract %2586[0] : f16 from vector<4xf16>
      %2588 = vector.extract %2586[1] : f16 from vector<4xf16>
      %2589 = vector.extract %2586[2] : f16 from vector<4xf16>
      %2590 = vector.extract %2586[3] : f16 from vector<4xf16>
      %2591 = vector.from_elements %2409, %2410, %2411, %2412 : vector<4xf32>
      %2592 = arith.truncf %2591 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2593 = vector.extract %2592[0] : f16 from vector<4xf16>
      %2594 = vector.extract %2592[1] : f16 from vector<4xf16>
      %2595 = vector.extract %2592[2] : f16 from vector<4xf16>
      %2596 = vector.extract %2592[3] : f16 from vector<4xf16>
      %2597 = vector.from_elements %2415, %2416, %2417, %2418 : vector<4xf32>
      %2598 = arith.truncf %2597 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2599 = vector.extract %2598[0] : f16 from vector<4xf16>
      %2600 = vector.extract %2598[1] : f16 from vector<4xf16>
      %2601 = vector.extract %2598[2] : f16 from vector<4xf16>
      %2602 = vector.extract %2598[3] : f16 from vector<4xf16>
      %2603 = vector.from_elements %2421, %2422, %2423, %2424 : vector<4xf32>
      %2604 = arith.truncf %2603 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2605 = vector.extract %2604[0] : f16 from vector<4xf16>
      %2606 = vector.extract %2604[1] : f16 from vector<4xf16>
      %2607 = vector.extract %2604[2] : f16 from vector<4xf16>
      %2608 = vector.extract %2604[3] : f16 from vector<4xf16>
      %2609 = vector.from_elements %2427, %2428, %2429, %2430 : vector<4xf32>
      %2610 = arith.truncf %2609 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2611 = vector.extract %2610[0] : f16 from vector<4xf16>
      %2612 = vector.extract %2610[1] : f16 from vector<4xf16>
      %2613 = vector.extract %2610[2] : f16 from vector<4xf16>
      %2614 = vector.extract %2610[3] : f16 from vector<4xf16>
      %2615 = vector.from_elements %2433, %2434, %2435, %2436 : vector<4xf32>
      %2616 = arith.truncf %2615 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2617 = vector.extract %2616[0] : f16 from vector<4xf16>
      %2618 = vector.extract %2616[1] : f16 from vector<4xf16>
      %2619 = vector.extract %2616[2] : f16 from vector<4xf16>
      %2620 = vector.extract %2616[3] : f16 from vector<4xf16>
      %2621 = vector.from_elements %2439, %2440, %2441, %2442 : vector<4xf32>
      %2622 = arith.truncf %2621 {tiled = true} : vector<4xf32> to vector<4xf16>
      %2623 = vector.extract %2622[0] : f16 from vector<4xf16>
      %2624 = vector.extract %2622[1] : f16 from vector<4xf16>
      %2625 = vector.extract %2622[2] : f16 from vector<4xf16>
      %2626 = vector.extract %2622[3] : f16 from vector<4xf16>
      %2627 = vector.extract %arg5[0, 0] : f32 from vector<2x8xf32>
      %2628 = vector.extract %arg5[0, 1] : f32 from vector<2x8xf32>
      %2629 = vector.extract %arg5[0, 2] : f32 from vector<2x8xf32>
      %2630 = vector.extract %arg5[0, 3] : f32 from vector<2x8xf32>
      %2631 = vector.from_elements %2627, %2628, %2629, %2630 : vector<4xf32>
      %2632 = vector.broadcast %2334 : f32 to vector<4xf32>
      %2633 = arith.mulf %2631, %2632 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2634 = vector.extract %2633[0] : f32 from vector<4xf32>
      %2635 = vector.extract %2633[1] : f32 from vector<4xf32>
      %2636 = vector.extract %2633[2] : f32 from vector<4xf32>
      %2637 = vector.extract %2633[3] : f32 from vector<4xf32>
      %2638 = vector.extract %arg5[0, 4] : f32 from vector<2x8xf32>
      %2639 = vector.extract %arg5[0, 5] : f32 from vector<2x8xf32>
      %2640 = vector.extract %arg5[0, 6] : f32 from vector<2x8xf32>
      %2641 = vector.extract %arg5[0, 7] : f32 from vector<2x8xf32>
      %2642 = vector.from_elements %2638, %2639, %2640, %2641 : vector<4xf32>
      %2643 = arith.mulf %2642, %2632 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2644 = vector.extract %2643[0] : f32 from vector<4xf32>
      %2645 = vector.extract %2643[1] : f32 from vector<4xf32>
      %2646 = vector.extract %2643[2] : f32 from vector<4xf32>
      %2647 = vector.extract %2643[3] : f32 from vector<4xf32>
      %2648 = vector.extract %arg5[1, 0] : f32 from vector<2x8xf32>
      %2649 = vector.extract %arg5[1, 1] : f32 from vector<2x8xf32>
      %2650 = vector.extract %arg5[1, 2] : f32 from vector<2x8xf32>
      %2651 = vector.extract %arg5[1, 3] : f32 from vector<2x8xf32>
      %2652 = vector.from_elements %2648, %2649, %2650, %2651 : vector<4xf32>
      %2653 = vector.broadcast %2342 : f32 to vector<4xf32>
      %2654 = arith.mulf %2652, %2653 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2655 = vector.extract %2654[0] : f32 from vector<4xf32>
      %2656 = vector.extract %2654[1] : f32 from vector<4xf32>
      %2657 = vector.extract %2654[2] : f32 from vector<4xf32>
      %2658 = vector.extract %2654[3] : f32 from vector<4xf32>
      %2659 = vector.extract %arg5[1, 4] : f32 from vector<2x8xf32>
      %2660 = vector.extract %arg5[1, 5] : f32 from vector<2x8xf32>
      %2661 = vector.extract %arg5[1, 6] : f32 from vector<2x8xf32>
      %2662 = vector.extract %arg5[1, 7] : f32 from vector<2x8xf32>
      %2663 = vector.from_elements %2659, %2660, %2661, %2662 : vector<4xf32>
      %2664 = arith.mulf %2663, %2653 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2665 = vector.extract %2664[0] : f32 from vector<4xf32>
      %2666 = vector.extract %2664[1] : f32 from vector<4xf32>
      %2667 = vector.extract %2664[2] : f32 from vector<4xf32>
      %2668 = vector.extract %2664[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %2669 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32)>()[%thread_id_x]
      %2670 = frisk.read_matrix_from_shm %view[%2669] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2671 = vector.bitcast %2670 : vector<4xi32> to vector<8xf16>
      %2672 = vector.extract %2671[0] : f16 from vector<8xf16>
      %2673 = vector.extract %2671[%278] : f16 from vector<8xf16>
      %2674 = vector.extract %2671[1] : f16 from vector<8xf16>
      %2675 = vector.extract %2671[%279] : f16 from vector<8xf16>
      %2676 = vector.extract %2671[2] : f16 from vector<8xf16>
      %2677 = vector.extract %2671[%280] : f16 from vector<8xf16>
      %2678 = vector.extract %2671[3] : f16 from vector<8xf16>
      %2679 = vector.extract %2671[%281] : f16 from vector<8xf16>
      %2680 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 16) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 128)>()[%thread_id_x]
      %2681 = frisk.read_matrix_from_shm %view[%2680] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2682 = vector.bitcast %2681 : vector<4xi32> to vector<8xf16>
      %2683 = vector.extract %2682[0] : f16 from vector<8xf16>
      %2684 = vector.extract %2682[%278] : f16 from vector<8xf16>
      %2685 = vector.extract %2682[1] : f16 from vector<8xf16>
      %2686 = vector.extract %2682[%279] : f16 from vector<8xf16>
      %2687 = vector.extract %2682[2] : f16 from vector<8xf16>
      %2688 = vector.extract %2682[%280] : f16 from vector<8xf16>
      %2689 = vector.extract %2682[3] : f16 from vector<8xf16>
      %2690 = vector.extract %2682[%281] : f16 from vector<8xf16>
      %2691 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 32) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 256)>()[%thread_id_x]
      %2692 = frisk.read_matrix_from_shm %view[%2691] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2693 = vector.bitcast %2692 : vector<4xi32> to vector<8xf16>
      %2694 = vector.extract %2693[0] : f16 from vector<8xf16>
      %2695 = vector.extract %2693[%278] : f16 from vector<8xf16>
      %2696 = vector.extract %2693[1] : f16 from vector<8xf16>
      %2697 = vector.extract %2693[%279] : f16 from vector<8xf16>
      %2698 = vector.extract %2693[2] : f16 from vector<8xf16>
      %2699 = vector.extract %2693[%280] : f16 from vector<8xf16>
      %2700 = vector.extract %2693[3] : f16 from vector<8xf16>
      %2701 = vector.extract %2693[%281] : f16 from vector<8xf16>
      %2702 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 48) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 384)>()[%thread_id_x]
      %2703 = frisk.read_matrix_from_shm %view[%2702] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2704 = vector.bitcast %2703 : vector<4xi32> to vector<8xf16>
      %2705 = vector.extract %2704[0] : f16 from vector<8xf16>
      %2706 = vector.extract %2704[%278] : f16 from vector<8xf16>
      %2707 = vector.extract %2704[1] : f16 from vector<8xf16>
      %2708 = vector.extract %2704[%279] : f16 from vector<8xf16>
      %2709 = vector.extract %2704[2] : f16 from vector<8xf16>
      %2710 = vector.extract %2704[%280] : f16 from vector<8xf16>
      %2711 = vector.extract %2704[3] : f16 from vector<8xf16>
      %2712 = vector.extract %2704[%281] : f16 from vector<8xf16>
      %2713 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 2048)>()[%thread_id_x]
      %2714 = frisk.read_matrix_from_shm %view[%2713] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2715 = vector.bitcast %2714 : vector<4xi32> to vector<8xf16>
      %2716 = vector.extract %2715[0] : f16 from vector<8xf16>
      %2717 = vector.extract %2715[%278] : f16 from vector<8xf16>
      %2718 = vector.extract %2715[1] : f16 from vector<8xf16>
      %2719 = vector.extract %2715[%279] : f16 from vector<8xf16>
      %2720 = vector.extract %2715[2] : f16 from vector<8xf16>
      %2721 = vector.extract %2715[%280] : f16 from vector<8xf16>
      %2722 = vector.extract %2715[3] : f16 from vector<8xf16>
      %2723 = vector.extract %2715[%281] : f16 from vector<8xf16>
      %2724 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 80) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 128)>()[%thread_id_x]
      %2725 = frisk.read_matrix_from_shm %view[%2724] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2726 = vector.bitcast %2725 : vector<4xi32> to vector<8xf16>
      %2727 = vector.extract %2726[0] : f16 from vector<8xf16>
      %2728 = vector.extract %2726[%278] : f16 from vector<8xf16>
      %2729 = vector.extract %2726[1] : f16 from vector<8xf16>
      %2730 = vector.extract %2726[%279] : f16 from vector<8xf16>
      %2731 = vector.extract %2726[2] : f16 from vector<8xf16>
      %2732 = vector.extract %2726[%280] : f16 from vector<8xf16>
      %2733 = vector.extract %2726[3] : f16 from vector<8xf16>
      %2734 = vector.extract %2726[%281] : f16 from vector<8xf16>
      %2735 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 96) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 256)>()[%thread_id_x]
      %2736 = frisk.read_matrix_from_shm %view[%2735] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2737 = vector.bitcast %2736 : vector<4xi32> to vector<8xf16>
      %2738 = vector.extract %2737[0] : f16 from vector<8xf16>
      %2739 = vector.extract %2737[%278] : f16 from vector<8xf16>
      %2740 = vector.extract %2737[1] : f16 from vector<8xf16>
      %2741 = vector.extract %2737[%279] : f16 from vector<8xf16>
      %2742 = vector.extract %2737[2] : f16 from vector<8xf16>
      %2743 = vector.extract %2737[%280] : f16 from vector<8xf16>
      %2744 = vector.extract %2737[3] : f16 from vector<8xf16>
      %2745 = vector.extract %2737[%281] : f16 from vector<8xf16>
      %2746 = affine.apply affine_map<()[s0] -> (s0 * 8 + ((s0 mod 64) floordiv 4) * 32 + (((s0 mod 64) floordiv 4 + 112) floordiv 64) * 2048 + (((s0 mod 64) floordiv 4) floordiv 4) * 384 - (s0 floordiv 4) * 32 + 384)>()[%thread_id_x]
      %2747 = frisk.read_matrix_from_shm %view[%2746] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2748 = vector.bitcast %2747 : vector<4xi32> to vector<8xf16>
      %2749 = vector.extract %2748[0] : f16 from vector<8xf16>
      %2750 = vector.extract %2748[%278] : f16 from vector<8xf16>
      %2751 = vector.extract %2748[1] : f16 from vector<8xf16>
      %2752 = vector.extract %2748[%279] : f16 from vector<8xf16>
      %2753 = vector.extract %2748[2] : f16 from vector<8xf16>
      %2754 = vector.extract %2748[%280] : f16 from vector<8xf16>
      %2755 = vector.extract %2748[3] : f16 from vector<8xf16>
      %2756 = vector.extract %2748[%281] : f16 from vector<8xf16>
      %subview_16 = memref.subview %arg2[0, %277, %856, 32] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %2757 = builtin.unrealized_conversion_cast %subview_16 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      gpu.barrier
      %2758 = llvm.extractvalue %2757[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %2759 = llvm.extractvalue %2757[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %2760 = llvm.getelementptr %2758[%2759] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %2761 = llvm.getelementptr %2760[%1826] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %2761, %48, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %2762 = llvm.getelementptr %2760[%1833] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %2762, %65, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %2763 = vector.from_elements %2533, %2534, %2535, %2536 : vector<4xf16>
      %2764 = vector.from_elements %2672, %2674, %2676, %2678 : vector<4xf16>
      %2765 = vector.from_elements %2634, %2635, %2636, %2637 : vector<4xf32>
      %2766 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2763, %2764, %2765 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2767 = vector.from_elements %2539, %2540, %2541, %2542 : vector<4xf16>
      %2768 = vector.from_elements %2683, %2685, %2687, %2689 : vector<4xf16>
      %2769 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2767, %2768, %2766 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2770 = vector.from_elements %2545, %2546, %2547, %2548 : vector<4xf16>
      %2771 = vector.from_elements %2694, %2696, %2698, %2700 : vector<4xf16>
      %2772 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2770, %2771, %2769 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2773 = vector.from_elements %2551, %2552, %2553, %2554 : vector<4xf16>
      %2774 = vector.from_elements %2705, %2707, %2709, %2711 : vector<4xf16>
      %2775 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2773, %2774, %2772 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2776 = vector.from_elements %2557, %2558, %2559, %2560 : vector<4xf16>
      %2777 = vector.from_elements %2716, %2718, %2720, %2722 : vector<4xf16>
      %2778 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2776, %2777, %2775 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2779 = vector.from_elements %2563, %2564, %2565, %2566 : vector<4xf16>
      %2780 = vector.from_elements %2727, %2729, %2731, %2733 : vector<4xf16>
      %2781 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2779, %2780, %2778 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2782 = vector.from_elements %2569, %2570, %2571, %2572 : vector<4xf16>
      %2783 = vector.from_elements %2738, %2740, %2742, %2744 : vector<4xf16>
      %2784 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2782, %2783, %2781 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2785 = vector.from_elements %2575, %2576, %2577, %2578 : vector<4xf16>
      %2786 = vector.from_elements %2749, %2751, %2753, %2755 : vector<4xf16>
      %2787 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2785, %2786, %2784 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2788 = vector.extract %2787[0] : f32 from vector<4xf32>
      %2789 = vector.extract %2787[1] : f32 from vector<4xf32>
      %2790 = vector.extract %2787[2] : f32 from vector<4xf32>
      %2791 = vector.extract %2787[3] : f32 from vector<4xf32>
      %2792 = vector.from_elements %2673, %2675, %2677, %2679 : vector<4xf16>
      %2793 = vector.from_elements %2644, %2645, %2646, %2647 : vector<4xf32>
      %2794 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2763, %2792, %2793 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2795 = vector.from_elements %2684, %2686, %2688, %2690 : vector<4xf16>
      %2796 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2767, %2795, %2794 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2797 = vector.from_elements %2695, %2697, %2699, %2701 : vector<4xf16>
      %2798 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2770, %2797, %2796 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2799 = vector.from_elements %2706, %2708, %2710, %2712 : vector<4xf16>
      %2800 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2773, %2799, %2798 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2801 = vector.from_elements %2717, %2719, %2721, %2723 : vector<4xf16>
      %2802 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2776, %2801, %2800 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2803 = vector.from_elements %2728, %2730, %2732, %2734 : vector<4xf16>
      %2804 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2779, %2803, %2802 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2805 = vector.from_elements %2739, %2741, %2743, %2745 : vector<4xf16>
      %2806 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2782, %2805, %2804 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2807 = vector.from_elements %2750, %2752, %2754, %2756 : vector<4xf16>
      %2808 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2785, %2807, %2806 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2809 = vector.extract %2808[0] : f32 from vector<4xf32>
      %2810 = vector.extract %2808[1] : f32 from vector<4xf32>
      %2811 = vector.extract %2808[2] : f32 from vector<4xf32>
      %2812 = vector.extract %2808[3] : f32 from vector<4xf32>
      %2813 = vector.from_elements %2581, %2582, %2583, %2584 : vector<4xf16>
      %2814 = vector.from_elements %2655, %2656, %2657, %2658 : vector<4xf32>
      %2815 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2813, %2764, %2814 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2816 = vector.from_elements %2587, %2588, %2589, %2590 : vector<4xf16>
      %2817 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2816, %2768, %2815 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2818 = vector.from_elements %2593, %2594, %2595, %2596 : vector<4xf16>
      %2819 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2818, %2771, %2817 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2820 = vector.from_elements %2599, %2600, %2601, %2602 : vector<4xf16>
      %2821 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2820, %2774, %2819 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2822 = vector.from_elements %2605, %2606, %2607, %2608 : vector<4xf16>
      %2823 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2822, %2777, %2821 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2824 = vector.from_elements %2611, %2612, %2613, %2614 : vector<4xf16>
      %2825 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2824, %2780, %2823 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2826 = vector.from_elements %2617, %2618, %2619, %2620 : vector<4xf16>
      %2827 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2826, %2783, %2825 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2828 = vector.from_elements %2623, %2624, %2625, %2626 : vector<4xf16>
      %2829 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2828, %2786, %2827 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2830 = vector.extract %2829[0] : f32 from vector<4xf32>
      %2831 = vector.extract %2829[1] : f32 from vector<4xf32>
      %2832 = vector.extract %2829[2] : f32 from vector<4xf32>
      %2833 = vector.extract %2829[3] : f32 from vector<4xf32>
      %2834 = vector.from_elements %2665, %2666, %2667, %2668 : vector<4xf32>
      %2835 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2813, %2792, %2834 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2836 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2816, %2795, %2835 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2837 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2818, %2797, %2836 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2838 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2820, %2799, %2837 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2839 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2822, %2801, %2838 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2840 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2824, %2803, %2839 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2841 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2826, %2805, %2840 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2842 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2828, %2807, %2841 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2843 = vector.extract %2842[0] : f32 from vector<4xf32>
      %2844 = vector.extract %2842[1] : f32 from vector<4xf32>
      %2845 = vector.extract %2842[2] : f32 from vector<4xf32>
      %2846 = vector.extract %2842[3] : f32 from vector<4xf32>
      %2847 = vector.extract %arg8[0, 0] : f32 from vector<2x8xf32>
      %2848 = vector.extract %arg8[0, 1] : f32 from vector<2x8xf32>
      %2849 = vector.extract %arg8[0, 2] : f32 from vector<2x8xf32>
      %2850 = vector.extract %arg8[0, 3] : f32 from vector<2x8xf32>
      %2851 = vector.from_elements %2847, %2848, %2849, %2850 : vector<4xf32>
      %2852 = arith.mulf %2851, %2632 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2853 = vector.extract %2852[0] : f32 from vector<4xf32>
      %2854 = vector.extract %2852[1] : f32 from vector<4xf32>
      %2855 = vector.extract %2852[2] : f32 from vector<4xf32>
      %2856 = vector.extract %2852[3] : f32 from vector<4xf32>
      %2857 = vector.extract %arg8[0, 4] : f32 from vector<2x8xf32>
      %2858 = vector.extract %arg8[0, 5] : f32 from vector<2x8xf32>
      %2859 = vector.extract %arg8[0, 6] : f32 from vector<2x8xf32>
      %2860 = vector.extract %arg8[0, 7] : f32 from vector<2x8xf32>
      %2861 = vector.from_elements %2857, %2858, %2859, %2860 : vector<4xf32>
      %2862 = arith.mulf %2861, %2632 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2863 = vector.extract %2862[0] : f32 from vector<4xf32>
      %2864 = vector.extract %2862[1] : f32 from vector<4xf32>
      %2865 = vector.extract %2862[2] : f32 from vector<4xf32>
      %2866 = vector.extract %2862[3] : f32 from vector<4xf32>
      %2867 = vector.extract %arg8[1, 0] : f32 from vector<2x8xf32>
      %2868 = vector.extract %arg8[1, 1] : f32 from vector<2x8xf32>
      %2869 = vector.extract %arg8[1, 2] : f32 from vector<2x8xf32>
      %2870 = vector.extract %arg8[1, 3] : f32 from vector<2x8xf32>
      %2871 = vector.from_elements %2867, %2868, %2869, %2870 : vector<4xf32>
      %2872 = arith.mulf %2871, %2653 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2873 = vector.extract %2872[0] : f32 from vector<4xf32>
      %2874 = vector.extract %2872[1] : f32 from vector<4xf32>
      %2875 = vector.extract %2872[2] : f32 from vector<4xf32>
      %2876 = vector.extract %2872[3] : f32 from vector<4xf32>
      %2877 = vector.extract %arg8[1, 4] : f32 from vector<2x8xf32>
      %2878 = vector.extract %arg8[1, 5] : f32 from vector<2x8xf32>
      %2879 = vector.extract %arg8[1, 6] : f32 from vector<2x8xf32>
      %2880 = vector.extract %arg8[1, 7] : f32 from vector<2x8xf32>
      %2881 = vector.from_elements %2877, %2878, %2879, %2880 : vector<4xf32>
      %2882 = arith.mulf %2881, %2653 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %2883 = vector.extract %2882[0] : f32 from vector<4xf32>
      %2884 = vector.extract %2882[1] : f32 from vector<4xf32>
      %2885 = vector.extract %2882[2] : f32 from vector<4xf32>
      %2886 = vector.extract %2882[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %2887 = frisk.read_matrix_from_shm %view[%2669] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2888 = vector.bitcast %2887 : vector<4xi32> to vector<8xf16>
      %2889 = vector.extract %2888[0] : f16 from vector<8xf16>
      %2890 = vector.extract %2888[%278] : f16 from vector<8xf16>
      %2891 = vector.extract %2888[1] : f16 from vector<8xf16>
      %2892 = vector.extract %2888[%279] : f16 from vector<8xf16>
      %2893 = vector.extract %2888[2] : f16 from vector<8xf16>
      %2894 = vector.extract %2888[%280] : f16 from vector<8xf16>
      %2895 = vector.extract %2888[3] : f16 from vector<8xf16>
      %2896 = vector.extract %2888[%281] : f16 from vector<8xf16>
      %2897 = frisk.read_matrix_from_shm %view[%2680] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2898 = vector.bitcast %2897 : vector<4xi32> to vector<8xf16>
      %2899 = vector.extract %2898[0] : f16 from vector<8xf16>
      %2900 = vector.extract %2898[%278] : f16 from vector<8xf16>
      %2901 = vector.extract %2898[1] : f16 from vector<8xf16>
      %2902 = vector.extract %2898[%279] : f16 from vector<8xf16>
      %2903 = vector.extract %2898[2] : f16 from vector<8xf16>
      %2904 = vector.extract %2898[%280] : f16 from vector<8xf16>
      %2905 = vector.extract %2898[3] : f16 from vector<8xf16>
      %2906 = vector.extract %2898[%281] : f16 from vector<8xf16>
      %2907 = frisk.read_matrix_from_shm %view[%2691] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2908 = vector.bitcast %2907 : vector<4xi32> to vector<8xf16>
      %2909 = vector.extract %2908[0] : f16 from vector<8xf16>
      %2910 = vector.extract %2908[%278] : f16 from vector<8xf16>
      %2911 = vector.extract %2908[1] : f16 from vector<8xf16>
      %2912 = vector.extract %2908[%279] : f16 from vector<8xf16>
      %2913 = vector.extract %2908[2] : f16 from vector<8xf16>
      %2914 = vector.extract %2908[%280] : f16 from vector<8xf16>
      %2915 = vector.extract %2908[3] : f16 from vector<8xf16>
      %2916 = vector.extract %2908[%281] : f16 from vector<8xf16>
      %2917 = frisk.read_matrix_from_shm %view[%2702] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2918 = vector.bitcast %2917 : vector<4xi32> to vector<8xf16>
      %2919 = vector.extract %2918[0] : f16 from vector<8xf16>
      %2920 = vector.extract %2918[%278] : f16 from vector<8xf16>
      %2921 = vector.extract %2918[1] : f16 from vector<8xf16>
      %2922 = vector.extract %2918[%279] : f16 from vector<8xf16>
      %2923 = vector.extract %2918[2] : f16 from vector<8xf16>
      %2924 = vector.extract %2918[%280] : f16 from vector<8xf16>
      %2925 = vector.extract %2918[3] : f16 from vector<8xf16>
      %2926 = vector.extract %2918[%281] : f16 from vector<8xf16>
      %2927 = frisk.read_matrix_from_shm %view[%2713] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2928 = vector.bitcast %2927 : vector<4xi32> to vector<8xf16>
      %2929 = vector.extract %2928[0] : f16 from vector<8xf16>
      %2930 = vector.extract %2928[%278] : f16 from vector<8xf16>
      %2931 = vector.extract %2928[1] : f16 from vector<8xf16>
      %2932 = vector.extract %2928[%279] : f16 from vector<8xf16>
      %2933 = vector.extract %2928[2] : f16 from vector<8xf16>
      %2934 = vector.extract %2928[%280] : f16 from vector<8xf16>
      %2935 = vector.extract %2928[3] : f16 from vector<8xf16>
      %2936 = vector.extract %2928[%281] : f16 from vector<8xf16>
      %2937 = frisk.read_matrix_from_shm %view[%2724] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2938 = vector.bitcast %2937 : vector<4xi32> to vector<8xf16>
      %2939 = vector.extract %2938[0] : f16 from vector<8xf16>
      %2940 = vector.extract %2938[%278] : f16 from vector<8xf16>
      %2941 = vector.extract %2938[1] : f16 from vector<8xf16>
      %2942 = vector.extract %2938[%279] : f16 from vector<8xf16>
      %2943 = vector.extract %2938[2] : f16 from vector<8xf16>
      %2944 = vector.extract %2938[%280] : f16 from vector<8xf16>
      %2945 = vector.extract %2938[3] : f16 from vector<8xf16>
      %2946 = vector.extract %2938[%281] : f16 from vector<8xf16>
      %2947 = frisk.read_matrix_from_shm %view[%2735] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2948 = vector.bitcast %2947 : vector<4xi32> to vector<8xf16>
      %2949 = vector.extract %2948[0] : f16 from vector<8xf16>
      %2950 = vector.extract %2948[%278] : f16 from vector<8xf16>
      %2951 = vector.extract %2948[1] : f16 from vector<8xf16>
      %2952 = vector.extract %2948[%279] : f16 from vector<8xf16>
      %2953 = vector.extract %2948[2] : f16 from vector<8xf16>
      %2954 = vector.extract %2948[%280] : f16 from vector<8xf16>
      %2955 = vector.extract %2948[3] : f16 from vector<8xf16>
      %2956 = vector.extract %2948[%281] : f16 from vector<8xf16>
      %2957 = frisk.read_matrix_from_shm %view[%2746] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %2958 = vector.bitcast %2957 : vector<4xi32> to vector<8xf16>
      %2959 = vector.extract %2958[0] : f16 from vector<8xf16>
      %2960 = vector.extract %2958[%278] : f16 from vector<8xf16>
      %2961 = vector.extract %2958[1] : f16 from vector<8xf16>
      %2962 = vector.extract %2958[%279] : f16 from vector<8xf16>
      %2963 = vector.extract %2958[2] : f16 from vector<8xf16>
      %2964 = vector.extract %2958[%280] : f16 from vector<8xf16>
      %2965 = vector.extract %2958[3] : f16 from vector<8xf16>
      %2966 = vector.extract %2958[%281] : f16 from vector<8xf16>
      %subview_17 = memref.subview %arg2[0, %277, %856, 64] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %2967 = builtin.unrealized_conversion_cast %subview_17 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      gpu.barrier
      %2968 = llvm.extractvalue %2967[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %2969 = llvm.extractvalue %2967[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %2970 = llvm.getelementptr %2968[%2969] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %2971 = llvm.getelementptr %2970[%1826] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %2971, %48, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %2972 = llvm.getelementptr %2970[%1833] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %2972, %65, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %2973 = vector.from_elements %2889, %2891, %2893, %2895 : vector<4xf16>
      %2974 = vector.from_elements %2853, %2854, %2855, %2856 : vector<4xf32>
      %2975 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2763, %2973, %2974 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2976 = vector.from_elements %2899, %2901, %2903, %2905 : vector<4xf16>
      %2977 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2767, %2976, %2975 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2978 = vector.from_elements %2909, %2911, %2913, %2915 : vector<4xf16>
      %2979 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2770, %2978, %2977 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2980 = vector.from_elements %2919, %2921, %2923, %2925 : vector<4xf16>
      %2981 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2773, %2980, %2979 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2982 = vector.from_elements %2929, %2931, %2933, %2935 : vector<4xf16>
      %2983 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2776, %2982, %2981 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2984 = vector.from_elements %2939, %2941, %2943, %2945 : vector<4xf16>
      %2985 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2779, %2984, %2983 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2986 = vector.from_elements %2949, %2951, %2953, %2955 : vector<4xf16>
      %2987 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2782, %2986, %2985 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2988 = vector.from_elements %2959, %2961, %2963, %2965 : vector<4xf16>
      %2989 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2785, %2988, %2987 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2990 = vector.extract %2989[0] : f32 from vector<4xf32>
      %2991 = vector.extract %2989[1] : f32 from vector<4xf32>
      %2992 = vector.extract %2989[2] : f32 from vector<4xf32>
      %2993 = vector.extract %2989[3] : f32 from vector<4xf32>
      %2994 = vector.from_elements %2890, %2892, %2894, %2896 : vector<4xf16>
      %2995 = vector.from_elements %2863, %2864, %2865, %2866 : vector<4xf32>
      %2996 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2763, %2994, %2995 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2997 = vector.from_elements %2900, %2902, %2904, %2906 : vector<4xf16>
      %2998 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2767, %2997, %2996 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %2999 = vector.from_elements %2910, %2912, %2914, %2916 : vector<4xf16>
      %3000 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2770, %2999, %2998 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3001 = vector.from_elements %2920, %2922, %2924, %2926 : vector<4xf16>
      %3002 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2773, %3001, %3000 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3003 = vector.from_elements %2930, %2932, %2934, %2936 : vector<4xf16>
      %3004 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2776, %3003, %3002 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3005 = vector.from_elements %2940, %2942, %2944, %2946 : vector<4xf16>
      %3006 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2779, %3005, %3004 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3007 = vector.from_elements %2950, %2952, %2954, %2956 : vector<4xf16>
      %3008 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2782, %3007, %3006 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3009 = vector.from_elements %2960, %2962, %2964, %2966 : vector<4xf16>
      %3010 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2785, %3009, %3008 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3011 = vector.extract %3010[0] : f32 from vector<4xf32>
      %3012 = vector.extract %3010[1] : f32 from vector<4xf32>
      %3013 = vector.extract %3010[2] : f32 from vector<4xf32>
      %3014 = vector.extract %3010[3] : f32 from vector<4xf32>
      %3015 = vector.from_elements %2873, %2874, %2875, %2876 : vector<4xf32>
      %3016 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2813, %2973, %3015 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3017 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2816, %2976, %3016 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3018 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2818, %2978, %3017 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3019 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2820, %2980, %3018 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3020 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2822, %2982, %3019 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3021 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2824, %2984, %3020 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3022 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2826, %2986, %3021 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3023 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2828, %2988, %3022 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3024 = vector.extract %3023[0] : f32 from vector<4xf32>
      %3025 = vector.extract %3023[1] : f32 from vector<4xf32>
      %3026 = vector.extract %3023[2] : f32 from vector<4xf32>
      %3027 = vector.extract %3023[3] : f32 from vector<4xf32>
      %3028 = vector.from_elements %2883, %2884, %2885, %2886 : vector<4xf32>
      %3029 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2813, %2994, %3028 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3030 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2816, %2997, %3029 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3031 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2818, %2999, %3030 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3032 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2820, %3001, %3031 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3033 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2822, %3003, %3032 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3034 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2824, %3005, %3033 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3035 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2826, %3007, %3034 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3036 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2828, %3009, %3035 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3037 = vector.extract %3036[0] : f32 from vector<4xf32>
      %3038 = vector.extract %3036[1] : f32 from vector<4xf32>
      %3039 = vector.extract %3036[2] : f32 from vector<4xf32>
      %3040 = vector.extract %3036[3] : f32 from vector<4xf32>
      %3041 = vector.extract %arg9[0, 0] : f32 from vector<2x8xf32>
      %3042 = vector.extract %arg9[0, 1] : f32 from vector<2x8xf32>
      %3043 = vector.extract %arg9[0, 2] : f32 from vector<2x8xf32>
      %3044 = vector.extract %arg9[0, 3] : f32 from vector<2x8xf32>
      %3045 = vector.from_elements %3041, %3042, %3043, %3044 : vector<4xf32>
      %3046 = arith.mulf %3045, %2632 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3047 = vector.extract %3046[0] : f32 from vector<4xf32>
      %3048 = vector.extract %3046[1] : f32 from vector<4xf32>
      %3049 = vector.extract %3046[2] : f32 from vector<4xf32>
      %3050 = vector.extract %3046[3] : f32 from vector<4xf32>
      %3051 = vector.extract %arg9[0, 4] : f32 from vector<2x8xf32>
      %3052 = vector.extract %arg9[0, 5] : f32 from vector<2x8xf32>
      %3053 = vector.extract %arg9[0, 6] : f32 from vector<2x8xf32>
      %3054 = vector.extract %arg9[0, 7] : f32 from vector<2x8xf32>
      %3055 = vector.from_elements %3051, %3052, %3053, %3054 : vector<4xf32>
      %3056 = arith.mulf %3055, %2632 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3057 = vector.extract %3056[0] : f32 from vector<4xf32>
      %3058 = vector.extract %3056[1] : f32 from vector<4xf32>
      %3059 = vector.extract %3056[2] : f32 from vector<4xf32>
      %3060 = vector.extract %3056[3] : f32 from vector<4xf32>
      %3061 = vector.extract %arg9[1, 0] : f32 from vector<2x8xf32>
      %3062 = vector.extract %arg9[1, 1] : f32 from vector<2x8xf32>
      %3063 = vector.extract %arg9[1, 2] : f32 from vector<2x8xf32>
      %3064 = vector.extract %arg9[1, 3] : f32 from vector<2x8xf32>
      %3065 = vector.from_elements %3061, %3062, %3063, %3064 : vector<4xf32>
      %3066 = arith.mulf %3065, %2653 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3067 = vector.extract %3066[0] : f32 from vector<4xf32>
      %3068 = vector.extract %3066[1] : f32 from vector<4xf32>
      %3069 = vector.extract %3066[2] : f32 from vector<4xf32>
      %3070 = vector.extract %3066[3] : f32 from vector<4xf32>
      %3071 = vector.extract %arg9[1, 4] : f32 from vector<2x8xf32>
      %3072 = vector.extract %arg9[1, 5] : f32 from vector<2x8xf32>
      %3073 = vector.extract %arg9[1, 6] : f32 from vector<2x8xf32>
      %3074 = vector.extract %arg9[1, 7] : f32 from vector<2x8xf32>
      %3075 = vector.from_elements %3071, %3072, %3073, %3074 : vector<4xf32>
      %3076 = arith.mulf %3075, %2653 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3077 = vector.extract %3076[0] : f32 from vector<4xf32>
      %3078 = vector.extract %3076[1] : f32 from vector<4xf32>
      %3079 = vector.extract %3076[2] : f32 from vector<4xf32>
      %3080 = vector.extract %3076[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %3081 = frisk.read_matrix_from_shm %view[%2669] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3082 = vector.bitcast %3081 : vector<4xi32> to vector<8xf16>
      %3083 = vector.extract %3082[0] : f16 from vector<8xf16>
      %3084 = vector.extract %3082[%278] : f16 from vector<8xf16>
      %3085 = vector.extract %3082[1] : f16 from vector<8xf16>
      %3086 = vector.extract %3082[%279] : f16 from vector<8xf16>
      %3087 = vector.extract %3082[2] : f16 from vector<8xf16>
      %3088 = vector.extract %3082[%280] : f16 from vector<8xf16>
      %3089 = vector.extract %3082[3] : f16 from vector<8xf16>
      %3090 = vector.extract %3082[%281] : f16 from vector<8xf16>
      %3091 = frisk.read_matrix_from_shm %view[%2680] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3092 = vector.bitcast %3091 : vector<4xi32> to vector<8xf16>
      %3093 = vector.extract %3092[0] : f16 from vector<8xf16>
      %3094 = vector.extract %3092[%278] : f16 from vector<8xf16>
      %3095 = vector.extract %3092[1] : f16 from vector<8xf16>
      %3096 = vector.extract %3092[%279] : f16 from vector<8xf16>
      %3097 = vector.extract %3092[2] : f16 from vector<8xf16>
      %3098 = vector.extract %3092[%280] : f16 from vector<8xf16>
      %3099 = vector.extract %3092[3] : f16 from vector<8xf16>
      %3100 = vector.extract %3092[%281] : f16 from vector<8xf16>
      %3101 = frisk.read_matrix_from_shm %view[%2691] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3102 = vector.bitcast %3101 : vector<4xi32> to vector<8xf16>
      %3103 = vector.extract %3102[0] : f16 from vector<8xf16>
      %3104 = vector.extract %3102[%278] : f16 from vector<8xf16>
      %3105 = vector.extract %3102[1] : f16 from vector<8xf16>
      %3106 = vector.extract %3102[%279] : f16 from vector<8xf16>
      %3107 = vector.extract %3102[2] : f16 from vector<8xf16>
      %3108 = vector.extract %3102[%280] : f16 from vector<8xf16>
      %3109 = vector.extract %3102[3] : f16 from vector<8xf16>
      %3110 = vector.extract %3102[%281] : f16 from vector<8xf16>
      %3111 = frisk.read_matrix_from_shm %view[%2702] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3112 = vector.bitcast %3111 : vector<4xi32> to vector<8xf16>
      %3113 = vector.extract %3112[0] : f16 from vector<8xf16>
      %3114 = vector.extract %3112[%278] : f16 from vector<8xf16>
      %3115 = vector.extract %3112[1] : f16 from vector<8xf16>
      %3116 = vector.extract %3112[%279] : f16 from vector<8xf16>
      %3117 = vector.extract %3112[2] : f16 from vector<8xf16>
      %3118 = vector.extract %3112[%280] : f16 from vector<8xf16>
      %3119 = vector.extract %3112[3] : f16 from vector<8xf16>
      %3120 = vector.extract %3112[%281] : f16 from vector<8xf16>
      %3121 = frisk.read_matrix_from_shm %view[%2713] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3122 = vector.bitcast %3121 : vector<4xi32> to vector<8xf16>
      %3123 = vector.extract %3122[0] : f16 from vector<8xf16>
      %3124 = vector.extract %3122[%278] : f16 from vector<8xf16>
      %3125 = vector.extract %3122[1] : f16 from vector<8xf16>
      %3126 = vector.extract %3122[%279] : f16 from vector<8xf16>
      %3127 = vector.extract %3122[2] : f16 from vector<8xf16>
      %3128 = vector.extract %3122[%280] : f16 from vector<8xf16>
      %3129 = vector.extract %3122[3] : f16 from vector<8xf16>
      %3130 = vector.extract %3122[%281] : f16 from vector<8xf16>
      %3131 = frisk.read_matrix_from_shm %view[%2724] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3132 = vector.bitcast %3131 : vector<4xi32> to vector<8xf16>
      %3133 = vector.extract %3132[0] : f16 from vector<8xf16>
      %3134 = vector.extract %3132[%278] : f16 from vector<8xf16>
      %3135 = vector.extract %3132[1] : f16 from vector<8xf16>
      %3136 = vector.extract %3132[%279] : f16 from vector<8xf16>
      %3137 = vector.extract %3132[2] : f16 from vector<8xf16>
      %3138 = vector.extract %3132[%280] : f16 from vector<8xf16>
      %3139 = vector.extract %3132[3] : f16 from vector<8xf16>
      %3140 = vector.extract %3132[%281] : f16 from vector<8xf16>
      %3141 = frisk.read_matrix_from_shm %view[%2735] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3142 = vector.bitcast %3141 : vector<4xi32> to vector<8xf16>
      %3143 = vector.extract %3142[0] : f16 from vector<8xf16>
      %3144 = vector.extract %3142[%278] : f16 from vector<8xf16>
      %3145 = vector.extract %3142[1] : f16 from vector<8xf16>
      %3146 = vector.extract %3142[%279] : f16 from vector<8xf16>
      %3147 = vector.extract %3142[2] : f16 from vector<8xf16>
      %3148 = vector.extract %3142[%280] : f16 from vector<8xf16>
      %3149 = vector.extract %3142[3] : f16 from vector<8xf16>
      %3150 = vector.extract %3142[%281] : f16 from vector<8xf16>
      %3151 = frisk.read_matrix_from_shm %view[%2746] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3152 = vector.bitcast %3151 : vector<4xi32> to vector<8xf16>
      %3153 = vector.extract %3152[0] : f16 from vector<8xf16>
      %3154 = vector.extract %3152[%278] : f16 from vector<8xf16>
      %3155 = vector.extract %3152[1] : f16 from vector<8xf16>
      %3156 = vector.extract %3152[%279] : f16 from vector<8xf16>
      %3157 = vector.extract %3152[2] : f16 from vector<8xf16>
      %3158 = vector.extract %3152[%280] : f16 from vector<8xf16>
      %3159 = vector.extract %3152[3] : f16 from vector<8xf16>
      %3160 = vector.extract %3152[%281] : f16 from vector<8xf16>
      %subview_18 = memref.subview %arg2[0, %277, %856, 96] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x32x4096x128xf16, 1> to memref<128x32xf16, strided<[128, 1], offset: ?>, 1>
      %3161 = builtin.unrealized_conversion_cast %subview_18 : memref<128x32xf16, strided<[128, 1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
      gpu.barrier
      %3162 = llvm.extractvalue %3161[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %3163 = llvm.extractvalue %3161[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> 
      %3164 = llvm.getelementptr %3162[%3163] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %3165 = llvm.getelementptr %3164[%1826] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %3165, %48, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      %3166 = llvm.getelementptr %3164[%1833] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      rocdl.global.load.async.lds %3166, %65, 16, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
      rocdl.asyncmark
      %3167 = vector.from_elements %3083, %3085, %3087, %3089 : vector<4xf16>
      %3168 = vector.from_elements %3047, %3048, %3049, %3050 : vector<4xf32>
      %3169 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2763, %3167, %3168 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3170 = vector.from_elements %3093, %3095, %3097, %3099 : vector<4xf16>
      %3171 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2767, %3170, %3169 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3172 = vector.from_elements %3103, %3105, %3107, %3109 : vector<4xf16>
      %3173 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2770, %3172, %3171 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3174 = vector.from_elements %3113, %3115, %3117, %3119 : vector<4xf16>
      %3175 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2773, %3174, %3173 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3176 = vector.from_elements %3123, %3125, %3127, %3129 : vector<4xf16>
      %3177 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2776, %3176, %3175 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3178 = vector.from_elements %3133, %3135, %3137, %3139 : vector<4xf16>
      %3179 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2779, %3178, %3177 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3180 = vector.from_elements %3143, %3145, %3147, %3149 : vector<4xf16>
      %3181 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2782, %3180, %3179 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3182 = vector.from_elements %3153, %3155, %3157, %3159 : vector<4xf16>
      %3183 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2785, %3182, %3181 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3184 = vector.extract %3183[0] : f32 from vector<4xf32>
      %3185 = vector.extract %3183[1] : f32 from vector<4xf32>
      %3186 = vector.extract %3183[2] : f32 from vector<4xf32>
      %3187 = vector.extract %3183[3] : f32 from vector<4xf32>
      %3188 = vector.from_elements %3084, %3086, %3088, %3090 : vector<4xf16>
      %3189 = vector.from_elements %3057, %3058, %3059, %3060 : vector<4xf32>
      %3190 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2763, %3188, %3189 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3191 = vector.from_elements %3094, %3096, %3098, %3100 : vector<4xf16>
      %3192 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2767, %3191, %3190 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3193 = vector.from_elements %3104, %3106, %3108, %3110 : vector<4xf16>
      %3194 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2770, %3193, %3192 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3195 = vector.from_elements %3114, %3116, %3118, %3120 : vector<4xf16>
      %3196 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2773, %3195, %3194 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3197 = vector.from_elements %3124, %3126, %3128, %3130 : vector<4xf16>
      %3198 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2776, %3197, %3196 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3199 = vector.from_elements %3134, %3136, %3138, %3140 : vector<4xf16>
      %3200 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2779, %3199, %3198 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3201 = vector.from_elements %3144, %3146, %3148, %3150 : vector<4xf16>
      %3202 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2782, %3201, %3200 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3203 = vector.from_elements %3154, %3156, %3158, %3160 : vector<4xf16>
      %3204 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2785, %3203, %3202 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3205 = vector.extract %3204[0] : f32 from vector<4xf32>
      %3206 = vector.extract %3204[1] : f32 from vector<4xf32>
      %3207 = vector.extract %3204[2] : f32 from vector<4xf32>
      %3208 = vector.extract %3204[3] : f32 from vector<4xf32>
      %3209 = vector.from_elements %3067, %3068, %3069, %3070 : vector<4xf32>
      %3210 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2813, %3167, %3209 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3211 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2816, %3170, %3210 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3212 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2818, %3172, %3211 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3213 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2820, %3174, %3212 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3214 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2822, %3176, %3213 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3215 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2824, %3178, %3214 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3216 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2826, %3180, %3215 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3217 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2828, %3182, %3216 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3218 = vector.extract %3217[0] : f32 from vector<4xf32>
      %3219 = vector.extract %3217[1] : f32 from vector<4xf32>
      %3220 = vector.extract %3217[2] : f32 from vector<4xf32>
      %3221 = vector.extract %3217[3] : f32 from vector<4xf32>
      %3222 = vector.from_elements %3077, %3078, %3079, %3080 : vector<4xf32>
      %3223 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2813, %3188, %3222 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3224 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2816, %3191, %3223 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3225 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2818, %3193, %3224 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3226 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2820, %3195, %3225 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3227 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2822, %3197, %3226 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3228 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2824, %3199, %3227 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3229 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2826, %3201, %3228 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3230 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2828, %3203, %3229 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3231 = vector.extract %3230[0] : f32 from vector<4xf32>
      %3232 = vector.extract %3230[1] : f32 from vector<4xf32>
      %3233 = vector.extract %3230[2] : f32 from vector<4xf32>
      %3234 = vector.extract %3230[3] : f32 from vector<4xf32>
      %3235 = vector.extract %arg10[0, 0] : f32 from vector<2x8xf32>
      %3236 = vector.extract %arg10[0, 1] : f32 from vector<2x8xf32>
      %3237 = vector.extract %arg10[0, 2] : f32 from vector<2x8xf32>
      %3238 = vector.extract %arg10[0, 3] : f32 from vector<2x8xf32>
      %3239 = vector.from_elements %3235, %3236, %3237, %3238 : vector<4xf32>
      %3240 = arith.mulf %3239, %2632 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3241 = vector.extract %3240[0] : f32 from vector<4xf32>
      %3242 = vector.extract %3240[1] : f32 from vector<4xf32>
      %3243 = vector.extract %3240[2] : f32 from vector<4xf32>
      %3244 = vector.extract %3240[3] : f32 from vector<4xf32>
      %3245 = vector.extract %arg10[0, 4] : f32 from vector<2x8xf32>
      %3246 = vector.extract %arg10[0, 5] : f32 from vector<2x8xf32>
      %3247 = vector.extract %arg10[0, 6] : f32 from vector<2x8xf32>
      %3248 = vector.extract %arg10[0, 7] : f32 from vector<2x8xf32>
      %3249 = vector.from_elements %3245, %3246, %3247, %3248 : vector<4xf32>
      %3250 = arith.mulf %3249, %2632 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3251 = vector.extract %3250[0] : f32 from vector<4xf32>
      %3252 = vector.extract %3250[1] : f32 from vector<4xf32>
      %3253 = vector.extract %3250[2] : f32 from vector<4xf32>
      %3254 = vector.extract %3250[3] : f32 from vector<4xf32>
      %3255 = vector.extract %arg10[1, 0] : f32 from vector<2x8xf32>
      %3256 = vector.extract %arg10[1, 1] : f32 from vector<2x8xf32>
      %3257 = vector.extract %arg10[1, 2] : f32 from vector<2x8xf32>
      %3258 = vector.extract %arg10[1, 3] : f32 from vector<2x8xf32>
      %3259 = vector.from_elements %3255, %3256, %3257, %3258 : vector<4xf32>
      %3260 = arith.mulf %3259, %2653 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3261 = vector.extract %3260[0] : f32 from vector<4xf32>
      %3262 = vector.extract %3260[1] : f32 from vector<4xf32>
      %3263 = vector.extract %3260[2] : f32 from vector<4xf32>
      %3264 = vector.extract %3260[3] : f32 from vector<4xf32>
      %3265 = vector.extract %arg10[1, 4] : f32 from vector<2x8xf32>
      %3266 = vector.extract %arg10[1, 5] : f32 from vector<2x8xf32>
      %3267 = vector.extract %arg10[1, 6] : f32 from vector<2x8xf32>
      %3268 = vector.extract %arg10[1, 7] : f32 from vector<2x8xf32>
      %3269 = vector.from_elements %3265, %3266, %3267, %3268 : vector<4xf32>
      %3270 = arith.mulf %3269, %2653 {origin = "frisk.mul", tiled = true} : vector<4xf32>
      %3271 = vector.extract %3270[0] : f32 from vector<4xf32>
      %3272 = vector.extract %3270[1] : f32 from vector<4xf32>
      %3273 = vector.extract %3270[2] : f32 from vector<4xf32>
      %3274 = vector.extract %3270[3] : f32 from vector<4xf32>
      rocdl.wait.asyncmark 0
      gpu.barrier
      %3275 = frisk.read_matrix_from_shm %view[%2669] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3276 = vector.bitcast %3275 : vector<4xi32> to vector<8xf16>
      %3277 = vector.extract %3276[0] : f16 from vector<8xf16>
      %3278 = vector.extract %3276[%278] : f16 from vector<8xf16>
      %3279 = vector.extract %3276[1] : f16 from vector<8xf16>
      %3280 = vector.extract %3276[%279] : f16 from vector<8xf16>
      %3281 = vector.extract %3276[2] : f16 from vector<8xf16>
      %3282 = vector.extract %3276[%280] : f16 from vector<8xf16>
      %3283 = vector.extract %3276[3] : f16 from vector<8xf16>
      %3284 = vector.extract %3276[%281] : f16 from vector<8xf16>
      %3285 = frisk.read_matrix_from_shm %view[%2680] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3286 = vector.bitcast %3285 : vector<4xi32> to vector<8xf16>
      %3287 = vector.extract %3286[0] : f16 from vector<8xf16>
      %3288 = vector.extract %3286[%278] : f16 from vector<8xf16>
      %3289 = vector.extract %3286[1] : f16 from vector<8xf16>
      %3290 = vector.extract %3286[%279] : f16 from vector<8xf16>
      %3291 = vector.extract %3286[2] : f16 from vector<8xf16>
      %3292 = vector.extract %3286[%280] : f16 from vector<8xf16>
      %3293 = vector.extract %3286[3] : f16 from vector<8xf16>
      %3294 = vector.extract %3286[%281] : f16 from vector<8xf16>
      %3295 = frisk.read_matrix_from_shm %view[%2691] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3296 = vector.bitcast %3295 : vector<4xi32> to vector<8xf16>
      %3297 = vector.extract %3296[0] : f16 from vector<8xf16>
      %3298 = vector.extract %3296[%278] : f16 from vector<8xf16>
      %3299 = vector.extract %3296[1] : f16 from vector<8xf16>
      %3300 = vector.extract %3296[%279] : f16 from vector<8xf16>
      %3301 = vector.extract %3296[2] : f16 from vector<8xf16>
      %3302 = vector.extract %3296[%280] : f16 from vector<8xf16>
      %3303 = vector.extract %3296[3] : f16 from vector<8xf16>
      %3304 = vector.extract %3296[%281] : f16 from vector<8xf16>
      %3305 = frisk.read_matrix_from_shm %view[%2702] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3306 = vector.bitcast %3305 : vector<4xi32> to vector<8xf16>
      %3307 = vector.extract %3306[0] : f16 from vector<8xf16>
      %3308 = vector.extract %3306[%278] : f16 from vector<8xf16>
      %3309 = vector.extract %3306[1] : f16 from vector<8xf16>
      %3310 = vector.extract %3306[%279] : f16 from vector<8xf16>
      %3311 = vector.extract %3306[2] : f16 from vector<8xf16>
      %3312 = vector.extract %3306[%280] : f16 from vector<8xf16>
      %3313 = vector.extract %3306[3] : f16 from vector<8xf16>
      %3314 = vector.extract %3306[%281] : f16 from vector<8xf16>
      %3315 = frisk.read_matrix_from_shm %view[%2713] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3316 = vector.bitcast %3315 : vector<4xi32> to vector<8xf16>
      %3317 = vector.extract %3316[0] : f16 from vector<8xf16>
      %3318 = vector.extract %3316[%278] : f16 from vector<8xf16>
      %3319 = vector.extract %3316[1] : f16 from vector<8xf16>
      %3320 = vector.extract %3316[%279] : f16 from vector<8xf16>
      %3321 = vector.extract %3316[2] : f16 from vector<8xf16>
      %3322 = vector.extract %3316[%280] : f16 from vector<8xf16>
      %3323 = vector.extract %3316[3] : f16 from vector<8xf16>
      %3324 = vector.extract %3316[%281] : f16 from vector<8xf16>
      %3325 = frisk.read_matrix_from_shm %view[%2724] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3326 = vector.bitcast %3325 : vector<4xi32> to vector<8xf16>
      %3327 = vector.extract %3326[0] : f16 from vector<8xf16>
      %3328 = vector.extract %3326[%278] : f16 from vector<8xf16>
      %3329 = vector.extract %3326[1] : f16 from vector<8xf16>
      %3330 = vector.extract %3326[%279] : f16 from vector<8xf16>
      %3331 = vector.extract %3326[2] : f16 from vector<8xf16>
      %3332 = vector.extract %3326[%280] : f16 from vector<8xf16>
      %3333 = vector.extract %3326[3] : f16 from vector<8xf16>
      %3334 = vector.extract %3326[%281] : f16 from vector<8xf16>
      %3335 = frisk.read_matrix_from_shm %view[%2735] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3336 = vector.bitcast %3335 : vector<4xi32> to vector<8xf16>
      %3337 = vector.extract %3336[0] : f16 from vector<8xf16>
      %3338 = vector.extract %3336[%278] : f16 from vector<8xf16>
      %3339 = vector.extract %3336[1] : f16 from vector<8xf16>
      %3340 = vector.extract %3336[%279] : f16 from vector<8xf16>
      %3341 = vector.extract %3336[2] : f16 from vector<8xf16>
      %3342 = vector.extract %3336[%280] : f16 from vector<8xf16>
      %3343 = vector.extract %3336[3] : f16 from vector<8xf16>
      %3344 = vector.extract %3336[%281] : f16 from vector<8xf16>
      %3345 = frisk.read_matrix_from_shm %view[%2746] {element_bit_width = 16 : i64, m = 32 : i64, n = 16 : i64} : memref<4096xf16, 3> -> vector<4xi32>
      %3346 = vector.bitcast %3345 : vector<4xi32> to vector<8xf16>
      %3347 = vector.extract %3346[0] : f16 from vector<8xf16>
      %3348 = vector.extract %3346[%278] : f16 from vector<8xf16>
      %3349 = vector.extract %3346[1] : f16 from vector<8xf16>
      %3350 = vector.extract %3346[%279] : f16 from vector<8xf16>
      %3351 = vector.extract %3346[2] : f16 from vector<8xf16>
      %3352 = vector.extract %3346[%280] : f16 from vector<8xf16>
      %3353 = vector.extract %3346[3] : f16 from vector<8xf16>
      %3354 = vector.extract %3346[%281] : f16 from vector<8xf16>
      %3355 = vector.from_elements %3277, %3279, %3281, %3283 : vector<4xf16>
      %3356 = vector.from_elements %3241, %3242, %3243, %3244 : vector<4xf32>
      %3357 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2763, %3355, %3356 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3358 = vector.from_elements %3287, %3289, %3291, %3293 : vector<4xf16>
      %3359 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2767, %3358, %3357 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3360 = vector.from_elements %3297, %3299, %3301, %3303 : vector<4xf16>
      %3361 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2770, %3360, %3359 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3362 = vector.from_elements %3307, %3309, %3311, %3313 : vector<4xf16>
      %3363 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2773, %3362, %3361 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3364 = vector.from_elements %3317, %3319, %3321, %3323 : vector<4xf16>
      %3365 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2776, %3364, %3363 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3366 = vector.from_elements %3327, %3329, %3331, %3333 : vector<4xf16>
      %3367 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2779, %3366, %3365 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3368 = vector.from_elements %3337, %3339, %3341, %3343 : vector<4xf16>
      %3369 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2782, %3368, %3367 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3370 = vector.from_elements %3347, %3349, %3351, %3353 : vector<4xf16>
      %3371 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2785, %3370, %3369 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3372 = vector.extract %3371[0] : f32 from vector<4xf32>
      %3373 = vector.extract %3371[1] : f32 from vector<4xf32>
      %3374 = vector.extract %3371[2] : f32 from vector<4xf32>
      %3375 = vector.extract %3371[3] : f32 from vector<4xf32>
      %3376 = vector.from_elements %3278, %3280, %3282, %3284 : vector<4xf16>
      %3377 = vector.from_elements %3251, %3252, %3253, %3254 : vector<4xf32>
      %3378 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2763, %3376, %3377 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3379 = vector.from_elements %3288, %3290, %3292, %3294 : vector<4xf16>
      %3380 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2767, %3379, %3378 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3381 = vector.from_elements %3298, %3300, %3302, %3304 : vector<4xf16>
      %3382 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2770, %3381, %3380 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3383 = vector.from_elements %3308, %3310, %3312, %3314 : vector<4xf16>
      %3384 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2773, %3383, %3382 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3385 = vector.from_elements %3318, %3320, %3322, %3324 : vector<4xf16>
      %3386 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2776, %3385, %3384 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3387 = vector.from_elements %3328, %3330, %3332, %3334 : vector<4xf16>
      %3388 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2779, %3387, %3386 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3389 = vector.from_elements %3338, %3340, %3342, %3344 : vector<4xf16>
      %3390 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2782, %3389, %3388 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3391 = vector.from_elements %3348, %3350, %3352, %3354 : vector<4xf16>
      %3392 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2785, %3391, %3390 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3393 = vector.extract %3392[0] : f32 from vector<4xf32>
      %3394 = vector.extract %3392[1] : f32 from vector<4xf32>
      %3395 = vector.extract %3392[2] : f32 from vector<4xf32>
      %3396 = vector.extract %3392[3] : f32 from vector<4xf32>
      %3397 = vector.from_elements %3261, %3262, %3263, %3264 : vector<4xf32>
      %3398 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2813, %3355, %3397 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3399 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2816, %3358, %3398 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3400 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2818, %3360, %3399 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3401 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2820, %3362, %3400 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3402 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2822, %3364, %3401 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3403 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2824, %3366, %3402 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3404 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2826, %3368, %3403 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3405 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2828, %3370, %3404 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3406 = vector.extract %3405[0] : f32 from vector<4xf32>
      %3407 = vector.extract %3405[1] : f32 from vector<4xf32>
      %3408 = vector.extract %3405[2] : f32 from vector<4xf32>
      %3409 = vector.extract %3405[3] : f32 from vector<4xf32>
      %3410 = vector.from_elements %3271, %3272, %3273, %3274 : vector<4xf32>
      %3411 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2813, %3376, %3410 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3412 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2816, %3379, %3411 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3413 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2818, %3381, %3412 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3414 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2820, %3383, %3413 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3415 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2822, %3385, %3414 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3416 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2824, %3387, %3415 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3417 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2826, %3389, %3416 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3418 = llvm.inline_asm has_side_effects convergent "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0" %2828, %3391, %3417 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      %3419 = vector.extract %3418[0] : f32 from vector<4xf32>
      %3420 = vector.extract %3418[1] : f32 from vector<4xf32>
      %3421 = vector.extract %3418[2] : f32 from vector<4xf32>
      %3422 = vector.extract %3418[3] : f32 from vector<4xf32>
      %3423 = vector.from_elements %2788, %2789, %2790, %2791, %2809, %2810, %2811, %2812, %2830, %2831, %2832, %2833, %2843, %2844, %2845, %2846 : vector<2x8xf32>
      %3424 = vector.from_elements %2526, %2530 : vector<2x1xf32>
      %3425 = vector.from_elements %2227, %2232 : vector<2x1xf32>
      %3426 = vector.from_elements %2990, %2991, %2992, %2993, %3011, %3012, %3013, %3014, %3024, %3025, %3026, %3027, %3037, %3038, %3039, %3040 : vector<2x8xf32>
      %3427 = vector.from_elements %3184, %3185, %3186, %3187, %3205, %3206, %3207, %3208, %3218, %3219, %3220, %3221, %3231, %3232, %3233, %3234 : vector<2x8xf32>
      %3428 = vector.from_elements %3372, %3373, %3374, %3375, %3393, %3394, %3395, %3396, %3406, %3407, %3408, %3409, %3419, %3420, %3421, %3422 : vector<2x8xf32>
      gpu.barrier
      affine.yield %3423, %3424, %3425, %3426, %3427, %3428 : vector<2x8xf32>, vector<2x1xf32>, vector<2x1xf32>, vector<2x8xf32>, vector<2x8xf32>, vector<2x8xf32>
    } {frisk.retained_words = 32 : i32, local_yield_transformed = true}
    %283 = arith.index_cast %thread_id_x : index to i32
    %284 = arith.andi %283, %c16_i32 : i32
    %285 = arith.cmpi ne, %284, %c0_i32 : i32
    %286 = arith.andi %283, %c32_i32 : i32
    %287 = arith.cmpi ne, %286, %c0_i32 : i32
    %288 = vector.extract %282#5[0, 0] : f32 from vector<2x8xf32>
    %289 = vector.extract %282#5[0, 1] : f32 from vector<2x8xf32>
    %290 = vector.extract %282#5[0, 2] : f32 from vector<2x8xf32>
    %291 = vector.extract %282#5[0, 3] : f32 from vector<2x8xf32>
    %292 = arith.select %285, %288, %289 : f32
    %293 = llvm.bitcast %292 : f32 to i32
    %294 = rocdl.ds_bpermute %10, %293 : (i32, i32) -> i32
    %295 = llvm.bitcast %294 : i32 to f32
    %296 = arith.select %285, %295, %288 : f32
    %297 = arith.select %285, %289, %295 : f32
    %298 = arith.select %285, %290, %291 : f32
    %299 = llvm.bitcast %298 : f32 to i32
    %300 = rocdl.ds_bpermute %10, %299 : (i32, i32) -> i32
    %301 = llvm.bitcast %300 : i32 to f32
    %302 = arith.select %285, %301, %290 : f32
    %303 = arith.select %285, %291, %301 : f32
    %304 = arith.select %287, %296, %302 : f32
    %305 = llvm.bitcast %304 : f32 to i32
    %306 = rocdl.ds_bpermute %14, %305 : (i32, i32) -> i32
    %307 = llvm.bitcast %306 : i32 to f32
    %308 = arith.select %287, %307, %296 : f32
    %309 = arith.select %287, %302, %307 : f32
    %310 = arith.select %287, %297, %303 : f32
    %311 = llvm.bitcast %310 : f32 to i32
    %312 = rocdl.ds_bpermute %14, %311 : (i32, i32) -> i32
    %313 = llvm.bitcast %312 : i32 to f32
    %314 = arith.select %287, %313, %297 : f32
    %315 = arith.select %287, %303, %313 : f32
    %316 = vector.extract %282#5[0, 4] : f32 from vector<2x8xf32>
    %317 = vector.extract %282#5[0, 5] : f32 from vector<2x8xf32>
    %318 = vector.extract %282#5[0, 6] : f32 from vector<2x8xf32>
    %319 = vector.extract %282#5[0, 7] : f32 from vector<2x8xf32>
    %320 = arith.select %285, %316, %317 : f32
    %321 = llvm.bitcast %320 : f32 to i32
    %322 = rocdl.ds_bpermute %10, %321 : (i32, i32) -> i32
    %323 = llvm.bitcast %322 : i32 to f32
    %324 = arith.select %285, %323, %316 : f32
    %325 = arith.select %285, %317, %323 : f32
    %326 = arith.select %285, %318, %319 : f32
    %327 = llvm.bitcast %326 : f32 to i32
    %328 = rocdl.ds_bpermute %10, %327 : (i32, i32) -> i32
    %329 = llvm.bitcast %328 : i32 to f32
    %330 = arith.select %285, %329, %318 : f32
    %331 = arith.select %285, %319, %329 : f32
    %332 = arith.select %287, %324, %330 : f32
    %333 = llvm.bitcast %332 : f32 to i32
    %334 = rocdl.ds_bpermute %14, %333 : (i32, i32) -> i32
    %335 = llvm.bitcast %334 : i32 to f32
    %336 = arith.select %287, %335, %324 : f32
    %337 = arith.select %287, %330, %335 : f32
    %338 = arith.select %287, %325, %331 : f32
    %339 = llvm.bitcast %338 : f32 to i32
    %340 = rocdl.ds_bpermute %14, %339 : (i32, i32) -> i32
    %341 = llvm.bitcast %340 : i32 to f32
    %342 = arith.select %287, %341, %325 : f32
    %343 = arith.select %287, %331, %341 : f32
    %344 = vector.extract %282#5[1, 0] : f32 from vector<2x8xf32>
    %345 = vector.extract %282#5[1, 1] : f32 from vector<2x8xf32>
    %346 = vector.extract %282#5[1, 2] : f32 from vector<2x8xf32>
    %347 = vector.extract %282#5[1, 3] : f32 from vector<2x8xf32>
    %348 = arith.select %285, %344, %345 : f32
    %349 = llvm.bitcast %348 : f32 to i32
    %350 = rocdl.ds_bpermute %10, %349 : (i32, i32) -> i32
    %351 = llvm.bitcast %350 : i32 to f32
    %352 = arith.select %285, %351, %344 : f32
    %353 = arith.select %285, %345, %351 : f32
    %354 = arith.select %285, %346, %347 : f32
    %355 = llvm.bitcast %354 : f32 to i32
    %356 = rocdl.ds_bpermute %10, %355 : (i32, i32) -> i32
    %357 = llvm.bitcast %356 : i32 to f32
    %358 = arith.select %285, %357, %346 : f32
    %359 = arith.select %285, %347, %357 : f32
    %360 = arith.select %287, %352, %358 : f32
    %361 = llvm.bitcast %360 : f32 to i32
    %362 = rocdl.ds_bpermute %14, %361 : (i32, i32) -> i32
    %363 = llvm.bitcast %362 : i32 to f32
    %364 = arith.select %287, %363, %352 : f32
    %365 = arith.select %287, %358, %363 : f32
    %366 = arith.select %287, %353, %359 : f32
    %367 = llvm.bitcast %366 : f32 to i32
    %368 = rocdl.ds_bpermute %14, %367 : (i32, i32) -> i32
    %369 = llvm.bitcast %368 : i32 to f32
    %370 = arith.select %287, %369, %353 : f32
    %371 = arith.select %287, %359, %369 : f32
    %372 = vector.extract %282#5[1, 4] : f32 from vector<2x8xf32>
    %373 = vector.extract %282#5[1, 5] : f32 from vector<2x8xf32>
    %374 = vector.extract %282#5[1, 6] : f32 from vector<2x8xf32>
    %375 = vector.extract %282#5[1, 7] : f32 from vector<2x8xf32>
    %376 = arith.select %285, %372, %373 : f32
    %377 = llvm.bitcast %376 : f32 to i32
    %378 = rocdl.ds_bpermute %10, %377 : (i32, i32) -> i32
    %379 = llvm.bitcast %378 : i32 to f32
    %380 = arith.select %285, %379, %372 : f32
    %381 = arith.select %285, %373, %379 : f32
    %382 = arith.select %285, %374, %375 : f32
    %383 = llvm.bitcast %382 : f32 to i32
    %384 = rocdl.ds_bpermute %10, %383 : (i32, i32) -> i32
    %385 = llvm.bitcast %384 : i32 to f32
    %386 = arith.select %285, %385, %374 : f32
    %387 = arith.select %285, %375, %385 : f32
    %388 = arith.select %287, %380, %386 : f32
    %389 = llvm.bitcast %388 : f32 to i32
    %390 = rocdl.ds_bpermute %14, %389 : (i32, i32) -> i32
    %391 = llvm.bitcast %390 : i32 to f32
    %392 = arith.select %287, %391, %380 : f32
    %393 = arith.select %287, %386, %391 : f32
    %394 = arith.select %287, %381, %387 : f32
    %395 = llvm.bitcast %394 : f32 to i32
    %396 = rocdl.ds_bpermute %14, %395 : (i32, i32) -> i32
    %397 = llvm.bitcast %396 : i32 to f32
    %398 = arith.select %287, %397, %381 : f32
    %399 = arith.select %287, %387, %397 : f32
    %400 = vector.extract %282#4[0, 0] : f32 from vector<2x8xf32>
    %401 = vector.extract %282#4[0, 1] : f32 from vector<2x8xf32>
    %402 = vector.extract %282#4[0, 2] : f32 from vector<2x8xf32>
    %403 = vector.extract %282#4[0, 3] : f32 from vector<2x8xf32>
    %404 = arith.select %285, %400, %401 : f32
    %405 = llvm.bitcast %404 : f32 to i32
    %406 = rocdl.ds_bpermute %10, %405 : (i32, i32) -> i32
    %407 = llvm.bitcast %406 : i32 to f32
    %408 = arith.select %285, %407, %400 : f32
    %409 = arith.select %285, %401, %407 : f32
    %410 = arith.select %285, %402, %403 : f32
    %411 = llvm.bitcast %410 : f32 to i32
    %412 = rocdl.ds_bpermute %10, %411 : (i32, i32) -> i32
    %413 = llvm.bitcast %412 : i32 to f32
    %414 = arith.select %285, %413, %402 : f32
    %415 = arith.select %285, %403, %413 : f32
    %416 = arith.select %287, %408, %414 : f32
    %417 = llvm.bitcast %416 : f32 to i32
    %418 = rocdl.ds_bpermute %14, %417 : (i32, i32) -> i32
    %419 = llvm.bitcast %418 : i32 to f32
    %420 = arith.select %287, %419, %408 : f32
    %421 = arith.select %287, %414, %419 : f32
    %422 = arith.select %287, %409, %415 : f32
    %423 = llvm.bitcast %422 : f32 to i32
    %424 = rocdl.ds_bpermute %14, %423 : (i32, i32) -> i32
    %425 = llvm.bitcast %424 : i32 to f32
    %426 = arith.select %287, %425, %409 : f32
    %427 = arith.select %287, %415, %425 : f32
    %428 = vector.extract %282#4[0, 4] : f32 from vector<2x8xf32>
    %429 = vector.extract %282#4[0, 5] : f32 from vector<2x8xf32>
    %430 = vector.extract %282#4[0, 6] : f32 from vector<2x8xf32>
    %431 = vector.extract %282#4[0, 7] : f32 from vector<2x8xf32>
    %432 = arith.select %285, %428, %429 : f32
    %433 = llvm.bitcast %432 : f32 to i32
    %434 = rocdl.ds_bpermute %10, %433 : (i32, i32) -> i32
    %435 = llvm.bitcast %434 : i32 to f32
    %436 = arith.select %285, %435, %428 : f32
    %437 = arith.select %285, %429, %435 : f32
    %438 = arith.select %285, %430, %431 : f32
    %439 = llvm.bitcast %438 : f32 to i32
    %440 = rocdl.ds_bpermute %10, %439 : (i32, i32) -> i32
    %441 = llvm.bitcast %440 : i32 to f32
    %442 = arith.select %285, %441, %430 : f32
    %443 = arith.select %285, %431, %441 : f32
    %444 = arith.select %287, %436, %442 : f32
    %445 = llvm.bitcast %444 : f32 to i32
    %446 = rocdl.ds_bpermute %14, %445 : (i32, i32) -> i32
    %447 = llvm.bitcast %446 : i32 to f32
    %448 = arith.select %287, %447, %436 : f32
    %449 = arith.select %287, %442, %447 : f32
    %450 = arith.select %287, %437, %443 : f32
    %451 = llvm.bitcast %450 : f32 to i32
    %452 = rocdl.ds_bpermute %14, %451 : (i32, i32) -> i32
    %453 = llvm.bitcast %452 : i32 to f32
    %454 = arith.select %287, %453, %437 : f32
    %455 = arith.select %287, %443, %453 : f32
    %456 = vector.extract %282#4[1, 0] : f32 from vector<2x8xf32>
    %457 = vector.extract %282#4[1, 1] : f32 from vector<2x8xf32>
    %458 = vector.extract %282#4[1, 2] : f32 from vector<2x8xf32>
    %459 = vector.extract %282#4[1, 3] : f32 from vector<2x8xf32>
    %460 = arith.select %285, %456, %457 : f32
    %461 = llvm.bitcast %460 : f32 to i32
    %462 = rocdl.ds_bpermute %10, %461 : (i32, i32) -> i32
    %463 = llvm.bitcast %462 : i32 to f32
    %464 = arith.select %285, %463, %456 : f32
    %465 = arith.select %285, %457, %463 : f32
    %466 = arith.select %285, %458, %459 : f32
    %467 = llvm.bitcast %466 : f32 to i32
    %468 = rocdl.ds_bpermute %10, %467 : (i32, i32) -> i32
    %469 = llvm.bitcast %468 : i32 to f32
    %470 = arith.select %285, %469, %458 : f32
    %471 = arith.select %285, %459, %469 : f32
    %472 = arith.select %287, %464, %470 : f32
    %473 = llvm.bitcast %472 : f32 to i32
    %474 = rocdl.ds_bpermute %14, %473 : (i32, i32) -> i32
    %475 = llvm.bitcast %474 : i32 to f32
    %476 = arith.select %287, %475, %464 : f32
    %477 = arith.select %287, %470, %475 : f32
    %478 = arith.select %287, %465, %471 : f32
    %479 = llvm.bitcast %478 : f32 to i32
    %480 = rocdl.ds_bpermute %14, %479 : (i32, i32) -> i32
    %481 = llvm.bitcast %480 : i32 to f32
    %482 = arith.select %287, %481, %465 : f32
    %483 = arith.select %287, %471, %481 : f32
    %484 = vector.extract %282#4[1, 4] : f32 from vector<2x8xf32>
    %485 = vector.extract %282#4[1, 5] : f32 from vector<2x8xf32>
    %486 = vector.extract %282#4[1, 6] : f32 from vector<2x8xf32>
    %487 = vector.extract %282#4[1, 7] : f32 from vector<2x8xf32>
    %488 = arith.select %285, %484, %485 : f32
    %489 = llvm.bitcast %488 : f32 to i32
    %490 = rocdl.ds_bpermute %10, %489 : (i32, i32) -> i32
    %491 = llvm.bitcast %490 : i32 to f32
    %492 = arith.select %285, %491, %484 : f32
    %493 = arith.select %285, %485, %491 : f32
    %494 = arith.select %285, %486, %487 : f32
    %495 = llvm.bitcast %494 : f32 to i32
    %496 = rocdl.ds_bpermute %10, %495 : (i32, i32) -> i32
    %497 = llvm.bitcast %496 : i32 to f32
    %498 = arith.select %285, %497, %486 : f32
    %499 = arith.select %285, %487, %497 : f32
    %500 = arith.select %287, %492, %498 : f32
    %501 = llvm.bitcast %500 : f32 to i32
    %502 = rocdl.ds_bpermute %14, %501 : (i32, i32) -> i32
    %503 = llvm.bitcast %502 : i32 to f32
    %504 = arith.select %287, %503, %492 : f32
    %505 = arith.select %287, %498, %503 : f32
    %506 = arith.select %287, %493, %499 : f32
    %507 = llvm.bitcast %506 : f32 to i32
    %508 = rocdl.ds_bpermute %14, %507 : (i32, i32) -> i32
    %509 = llvm.bitcast %508 : i32 to f32
    %510 = arith.select %287, %509, %493 : f32
    %511 = arith.select %287, %499, %509 : f32
    %512 = vector.extract %282#3[0, 0] : f32 from vector<2x8xf32>
    %513 = vector.extract %282#3[0, 1] : f32 from vector<2x8xf32>
    %514 = vector.extract %282#3[0, 2] : f32 from vector<2x8xf32>
    %515 = vector.extract %282#3[0, 3] : f32 from vector<2x8xf32>
    %516 = arith.select %285, %512, %513 : f32
    %517 = llvm.bitcast %516 : f32 to i32
    %518 = rocdl.ds_bpermute %10, %517 : (i32, i32) -> i32
    %519 = llvm.bitcast %518 : i32 to f32
    %520 = arith.select %285, %519, %512 : f32
    %521 = arith.select %285, %513, %519 : f32
    %522 = arith.select %285, %514, %515 : f32
    %523 = llvm.bitcast %522 : f32 to i32
    %524 = rocdl.ds_bpermute %10, %523 : (i32, i32) -> i32
    %525 = llvm.bitcast %524 : i32 to f32
    %526 = arith.select %285, %525, %514 : f32
    %527 = arith.select %285, %515, %525 : f32
    %528 = arith.select %287, %520, %526 : f32
    %529 = llvm.bitcast %528 : f32 to i32
    %530 = rocdl.ds_bpermute %14, %529 : (i32, i32) -> i32
    %531 = llvm.bitcast %530 : i32 to f32
    %532 = arith.select %287, %531, %520 : f32
    %533 = arith.select %287, %526, %531 : f32
    %534 = arith.select %287, %521, %527 : f32
    %535 = llvm.bitcast %534 : f32 to i32
    %536 = rocdl.ds_bpermute %14, %535 : (i32, i32) -> i32
    %537 = llvm.bitcast %536 : i32 to f32
    %538 = arith.select %287, %537, %521 : f32
    %539 = arith.select %287, %527, %537 : f32
    %540 = vector.extract %282#3[0, 4] : f32 from vector<2x8xf32>
    %541 = vector.extract %282#3[0, 5] : f32 from vector<2x8xf32>
    %542 = vector.extract %282#3[0, 6] : f32 from vector<2x8xf32>
    %543 = vector.extract %282#3[0, 7] : f32 from vector<2x8xf32>
    %544 = arith.select %285, %540, %541 : f32
    %545 = llvm.bitcast %544 : f32 to i32
    %546 = rocdl.ds_bpermute %10, %545 : (i32, i32) -> i32
    %547 = llvm.bitcast %546 : i32 to f32
    %548 = arith.select %285, %547, %540 : f32
    %549 = arith.select %285, %541, %547 : f32
    %550 = arith.select %285, %542, %543 : f32
    %551 = llvm.bitcast %550 : f32 to i32
    %552 = rocdl.ds_bpermute %10, %551 : (i32, i32) -> i32
    %553 = llvm.bitcast %552 : i32 to f32
    %554 = arith.select %285, %553, %542 : f32
    %555 = arith.select %285, %543, %553 : f32
    %556 = arith.select %287, %548, %554 : f32
    %557 = llvm.bitcast %556 : f32 to i32
    %558 = rocdl.ds_bpermute %14, %557 : (i32, i32) -> i32
    %559 = llvm.bitcast %558 : i32 to f32
    %560 = arith.select %287, %559, %548 : f32
    %561 = arith.select %287, %554, %559 : f32
    %562 = arith.select %287, %549, %555 : f32
    %563 = llvm.bitcast %562 : f32 to i32
    %564 = rocdl.ds_bpermute %14, %563 : (i32, i32) -> i32
    %565 = llvm.bitcast %564 : i32 to f32
    %566 = arith.select %287, %565, %549 : f32
    %567 = arith.select %287, %555, %565 : f32
    %568 = vector.extract %282#3[1, 0] : f32 from vector<2x8xf32>
    %569 = vector.extract %282#3[1, 1] : f32 from vector<2x8xf32>
    %570 = vector.extract %282#3[1, 2] : f32 from vector<2x8xf32>
    %571 = vector.extract %282#3[1, 3] : f32 from vector<2x8xf32>
    %572 = arith.select %285, %568, %569 : f32
    %573 = llvm.bitcast %572 : f32 to i32
    %574 = rocdl.ds_bpermute %10, %573 : (i32, i32) -> i32
    %575 = llvm.bitcast %574 : i32 to f32
    %576 = arith.select %285, %575, %568 : f32
    %577 = arith.select %285, %569, %575 : f32
    %578 = arith.select %285, %570, %571 : f32
    %579 = llvm.bitcast %578 : f32 to i32
    %580 = rocdl.ds_bpermute %10, %579 : (i32, i32) -> i32
    %581 = llvm.bitcast %580 : i32 to f32
    %582 = arith.select %285, %581, %570 : f32
    %583 = arith.select %285, %571, %581 : f32
    %584 = arith.select %287, %576, %582 : f32
    %585 = llvm.bitcast %584 : f32 to i32
    %586 = rocdl.ds_bpermute %14, %585 : (i32, i32) -> i32
    %587 = llvm.bitcast %586 : i32 to f32
    %588 = arith.select %287, %587, %576 : f32
    %589 = arith.select %287, %582, %587 : f32
    %590 = arith.select %287, %577, %583 : f32
    %591 = llvm.bitcast %590 : f32 to i32
    %592 = rocdl.ds_bpermute %14, %591 : (i32, i32) -> i32
    %593 = llvm.bitcast %592 : i32 to f32
    %594 = arith.select %287, %593, %577 : f32
    %595 = arith.select %287, %583, %593 : f32
    %596 = vector.extract %282#3[1, 4] : f32 from vector<2x8xf32>
    %597 = vector.extract %282#3[1, 5] : f32 from vector<2x8xf32>
    %598 = vector.extract %282#3[1, 6] : f32 from vector<2x8xf32>
    %599 = vector.extract %282#3[1, 7] : f32 from vector<2x8xf32>
    %600 = arith.select %285, %596, %597 : f32
    %601 = llvm.bitcast %600 : f32 to i32
    %602 = rocdl.ds_bpermute %10, %601 : (i32, i32) -> i32
    %603 = llvm.bitcast %602 : i32 to f32
    %604 = arith.select %285, %603, %596 : f32
    %605 = arith.select %285, %597, %603 : f32
    %606 = arith.select %285, %598, %599 : f32
    %607 = llvm.bitcast %606 : f32 to i32
    %608 = rocdl.ds_bpermute %10, %607 : (i32, i32) -> i32
    %609 = llvm.bitcast %608 : i32 to f32
    %610 = arith.select %285, %609, %598 : f32
    %611 = arith.select %285, %599, %609 : f32
    %612 = arith.select %287, %604, %610 : f32
    %613 = llvm.bitcast %612 : f32 to i32
    %614 = rocdl.ds_bpermute %14, %613 : (i32, i32) -> i32
    %615 = llvm.bitcast %614 : i32 to f32
    %616 = arith.select %287, %615, %604 : f32
    %617 = arith.select %287, %610, %615 : f32
    %618 = arith.select %287, %605, %611 : f32
    %619 = llvm.bitcast %618 : f32 to i32
    %620 = rocdl.ds_bpermute %14, %619 : (i32, i32) -> i32
    %621 = llvm.bitcast %620 : i32 to f32
    %622 = arith.select %287, %621, %605 : f32
    %623 = arith.select %287, %611, %621 : f32
    %624 = vector.extract %282#0[0, 0] : f32 from vector<2x8xf32>
    %625 = vector.extract %282#0[0, 1] : f32 from vector<2x8xf32>
    %626 = vector.extract %282#0[0, 2] : f32 from vector<2x8xf32>
    %627 = vector.extract %282#0[0, 3] : f32 from vector<2x8xf32>
    %628 = arith.select %285, %624, %625 : f32
    %629 = llvm.bitcast %628 : f32 to i32
    %630 = rocdl.ds_bpermute %10, %629 : (i32, i32) -> i32
    %631 = llvm.bitcast %630 : i32 to f32
    %632 = arith.select %285, %631, %624 : f32
    %633 = arith.select %285, %625, %631 : f32
    %634 = arith.select %285, %626, %627 : f32
    %635 = llvm.bitcast %634 : f32 to i32
    %636 = rocdl.ds_bpermute %10, %635 : (i32, i32) -> i32
    %637 = llvm.bitcast %636 : i32 to f32
    %638 = arith.select %285, %637, %626 : f32
    %639 = arith.select %285, %627, %637 : f32
    %640 = arith.select %287, %632, %638 : f32
    %641 = llvm.bitcast %640 : f32 to i32
    %642 = rocdl.ds_bpermute %14, %641 : (i32, i32) -> i32
    %643 = llvm.bitcast %642 : i32 to f32
    %644 = arith.select %287, %643, %632 : f32
    %645 = arith.select %287, %638, %643 : f32
    %646 = arith.select %287, %633, %639 : f32
    %647 = llvm.bitcast %646 : f32 to i32
    %648 = rocdl.ds_bpermute %14, %647 : (i32, i32) -> i32
    %649 = llvm.bitcast %648 : i32 to f32
    %650 = arith.select %287, %649, %633 : f32
    %651 = arith.select %287, %639, %649 : f32
    %652 = vector.extract %282#0[0, 4] : f32 from vector<2x8xf32>
    %653 = vector.extract %282#0[0, 5] : f32 from vector<2x8xf32>
    %654 = vector.extract %282#0[0, 6] : f32 from vector<2x8xf32>
    %655 = vector.extract %282#0[0, 7] : f32 from vector<2x8xf32>
    %656 = arith.select %285, %652, %653 : f32
    %657 = llvm.bitcast %656 : f32 to i32
    %658 = rocdl.ds_bpermute %10, %657 : (i32, i32) -> i32
    %659 = llvm.bitcast %658 : i32 to f32
    %660 = arith.select %285, %659, %652 : f32
    %661 = arith.select %285, %653, %659 : f32
    %662 = arith.select %285, %654, %655 : f32
    %663 = llvm.bitcast %662 : f32 to i32
    %664 = rocdl.ds_bpermute %10, %663 : (i32, i32) -> i32
    %665 = llvm.bitcast %664 : i32 to f32
    %666 = arith.select %285, %665, %654 : f32
    %667 = arith.select %285, %655, %665 : f32
    %668 = arith.select %287, %660, %666 : f32
    %669 = llvm.bitcast %668 : f32 to i32
    %670 = rocdl.ds_bpermute %14, %669 : (i32, i32) -> i32
    %671 = llvm.bitcast %670 : i32 to f32
    %672 = arith.select %287, %671, %660 : f32
    %673 = arith.select %287, %666, %671 : f32
    %674 = arith.select %287, %661, %667 : f32
    %675 = llvm.bitcast %674 : f32 to i32
    %676 = rocdl.ds_bpermute %14, %675 : (i32, i32) -> i32
    %677 = llvm.bitcast %676 : i32 to f32
    %678 = arith.select %287, %677, %661 : f32
    %679 = arith.select %287, %667, %677 : f32
    %680 = vector.extract %282#0[1, 0] : f32 from vector<2x8xf32>
    %681 = vector.extract %282#0[1, 1] : f32 from vector<2x8xf32>
    %682 = vector.extract %282#0[1, 2] : f32 from vector<2x8xf32>
    %683 = vector.extract %282#0[1, 3] : f32 from vector<2x8xf32>
    %684 = arith.select %285, %680, %681 : f32
    %685 = llvm.bitcast %684 : f32 to i32
    %686 = rocdl.ds_bpermute %10, %685 : (i32, i32) -> i32
    %687 = llvm.bitcast %686 : i32 to f32
    %688 = arith.select %285, %687, %680 : f32
    %689 = arith.select %285, %681, %687 : f32
    %690 = arith.select %285, %682, %683 : f32
    %691 = llvm.bitcast %690 : f32 to i32
    %692 = rocdl.ds_bpermute %10, %691 : (i32, i32) -> i32
    %693 = llvm.bitcast %692 : i32 to f32
    %694 = arith.select %285, %693, %682 : f32
    %695 = arith.select %285, %683, %693 : f32
    %696 = arith.select %287, %688, %694 : f32
    %697 = llvm.bitcast %696 : f32 to i32
    %698 = rocdl.ds_bpermute %14, %697 : (i32, i32) -> i32
    %699 = llvm.bitcast %698 : i32 to f32
    %700 = arith.select %287, %699, %688 : f32
    %701 = arith.select %287, %694, %699 : f32
    %702 = arith.select %287, %689, %695 : f32
    %703 = llvm.bitcast %702 : f32 to i32
    %704 = rocdl.ds_bpermute %14, %703 : (i32, i32) -> i32
    %705 = llvm.bitcast %704 : i32 to f32
    %706 = arith.select %287, %705, %689 : f32
    %707 = arith.select %287, %695, %705 : f32
    %708 = vector.extract %282#0[1, 4] : f32 from vector<2x8xf32>
    %709 = vector.extract %282#0[1, 5] : f32 from vector<2x8xf32>
    %710 = vector.extract %282#0[1, 6] : f32 from vector<2x8xf32>
    %711 = vector.extract %282#0[1, 7] : f32 from vector<2x8xf32>
    %712 = arith.select %285, %708, %709 : f32
    %713 = llvm.bitcast %712 : f32 to i32
    %714 = rocdl.ds_bpermute %10, %713 : (i32, i32) -> i32
    %715 = llvm.bitcast %714 : i32 to f32
    %716 = arith.select %285, %715, %708 : f32
    %717 = arith.select %285, %709, %715 : f32
    %718 = arith.select %285, %710, %711 : f32
    %719 = llvm.bitcast %718 : f32 to i32
    %720 = rocdl.ds_bpermute %10, %719 : (i32, i32) -> i32
    %721 = llvm.bitcast %720 : i32 to f32
    %722 = arith.select %285, %721, %710 : f32
    %723 = arith.select %285, %711, %721 : f32
    %724 = arith.select %287, %716, %722 : f32
    %725 = llvm.bitcast %724 : f32 to i32
    %726 = rocdl.ds_bpermute %14, %725 : (i32, i32) -> i32
    %727 = llvm.bitcast %726 : i32 to f32
    %728 = arith.select %287, %727, %716 : f32
    %729 = arith.select %287, %722, %727 : f32
    %730 = arith.select %287, %717, %723 : f32
    %731 = llvm.bitcast %730 : f32 to i32
    %732 = rocdl.ds_bpermute %14, %731 : (i32, i32) -> i32
    %733 = llvm.bitcast %732 : i32 to f32
    %734 = arith.select %287, %733, %717 : f32
    %735 = arith.select %287, %723, %733 : f32
    %736 = vector.extract %282#1[0, 0] : f32 from vector<2x1xf32>
    %737 = vector.from_elements %644, %650, %645, %651 : vector<4xf32>
    %738 = vector.broadcast %736 : f32 to vector<4xf32>
    %739 = arith.divf %cst, %738 fastmath<arcp> : vector<4xf32>
    %740 = arith.mulf %737, %739 : vector<4xf32>
    %741 = arith.truncf %740 {tiled = true} : vector<4xf32> to vector<4xf16>
    %742 = vector.extract %741[0] : f16 from vector<4xf16>
    %743 = vector.extract %741[1] : f16 from vector<4xf16>
    %744 = vector.extract %741[2] : f16 from vector<4xf16>
    %745 = vector.extract %741[3] : f16 from vector<4xf16>
    %746 = vector.from_elements %672, %678, %673, %679 : vector<4xf32>
    %747 = arith.mulf %746, %739 : vector<4xf32>
    %748 = arith.truncf %747 {tiled = true} : vector<4xf32> to vector<4xf16>
    %749 = vector.extract %748[0] : f16 from vector<4xf16>
    %750 = vector.extract %748[1] : f16 from vector<4xf16>
    %751 = vector.extract %748[2] : f16 from vector<4xf16>
    %752 = vector.extract %748[3] : f16 from vector<4xf16>
    %753 = vector.extract %282#1[1, 0] : f32 from vector<2x1xf32>
    %754 = vector.from_elements %700, %706, %701, %707 : vector<4xf32>
    %755 = vector.broadcast %753 : f32 to vector<4xf32>
    %756 = arith.divf %cst, %755 fastmath<arcp> : vector<4xf32>
    %757 = arith.mulf %754, %756 : vector<4xf32>
    %758 = arith.truncf %757 {tiled = true} : vector<4xf32> to vector<4xf16>
    %759 = vector.extract %758[0] : f16 from vector<4xf16>
    %760 = vector.extract %758[1] : f16 from vector<4xf16>
    %761 = vector.extract %758[2] : f16 from vector<4xf16>
    %762 = vector.extract %758[3] : f16 from vector<4xf16>
    %763 = vector.from_elements %728, %734, %729, %735 : vector<4xf32>
    %764 = arith.mulf %763, %756 : vector<4xf32>
    %765 = arith.truncf %764 {tiled = true} : vector<4xf32> to vector<4xf16>
    %766 = vector.extract %765[0] : f16 from vector<4xf16>
    %767 = vector.extract %765[1] : f16 from vector<4xf16>
    %768 = vector.extract %765[2] : f16 from vector<4xf16>
    %769 = vector.extract %765[3] : f16 from vector<4xf16>
    %770 = affine.apply affine_map<()[s0] -> (s0 floordiv 64)>()[%thread_id_x]
    %771 = arith.cmpi ult, %770, %c4 : index
    scf.if %771 {
      affine.store %742, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
      affine.store %743, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
      affine.store %744, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
      affine.store %745, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      affine.store %749, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 16] : memref<1x32x4096x128xf16, 1>
      affine.store %750, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 17] : memref<1x32x4096x128xf16, 1>
      affine.store %751, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 18] : memref<1x32x4096x128xf16, 1>
      affine.store %752, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 19] : memref<1x32x4096x128xf16, 1>
      affine.store %759, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4] : memref<1x32x4096x128xf16, 1>
      affine.store %760, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 1] : memref<1x32x4096x128xf16, 1>
      affine.store %761, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 2] : memref<1x32x4096x128xf16, 1>
      affine.store %762, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 3] : memref<1x32x4096x128xf16, 1>
      affine.store %766, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 16] : memref<1x32x4096x128xf16, 1>
      affine.store %767, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 17] : memref<1x32x4096x128xf16, 1>
      affine.store %768, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 18] : memref<1x32x4096x128xf16, 1>
      affine.store %769, %arg3[0, (symbol(%24) + symbol(%block_id_z) * 32 + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 19] : memref<1x32x4096x128xf16, 1>
    }
    %772 = vector.from_elements %532, %538, %533, %539 : vector<4xf32>
    %773 = arith.mulf %772, %739 : vector<4xf32>
    %774 = arith.truncf %773 {tiled = true} : vector<4xf32> to vector<4xf16>
    %775 = vector.extract %774[0] : f16 from vector<4xf16>
    %776 = vector.extract %774[1] : f16 from vector<4xf16>
    %777 = vector.extract %774[2] : f16 from vector<4xf16>
    %778 = vector.extract %774[3] : f16 from vector<4xf16>
    %779 = vector.from_elements %560, %566, %561, %567 : vector<4xf32>
    %780 = arith.mulf %779, %739 : vector<4xf32>
    %781 = arith.truncf %780 {tiled = true} : vector<4xf32> to vector<4xf16>
    %782 = vector.extract %781[0] : f16 from vector<4xf16>
    %783 = vector.extract %781[1] : f16 from vector<4xf16>
    %784 = vector.extract %781[2] : f16 from vector<4xf16>
    %785 = vector.extract %781[3] : f16 from vector<4xf16>
    %786 = vector.from_elements %588, %594, %589, %595 : vector<4xf32>
    %787 = arith.mulf %786, %756 : vector<4xf32>
    %788 = arith.truncf %787 {tiled = true} : vector<4xf32> to vector<4xf16>
    %789 = vector.extract %788[0] : f16 from vector<4xf16>
    %790 = vector.extract %788[1] : f16 from vector<4xf16>
    %791 = vector.extract %788[2] : f16 from vector<4xf16>
    %792 = vector.extract %788[3] : f16 from vector<4xf16>
    %793 = vector.from_elements %616, %622, %617, %623 : vector<4xf32>
    %794 = arith.mulf %793, %756 : vector<4xf32>
    %795 = arith.truncf %794 {tiled = true} : vector<4xf32> to vector<4xf16>
    %796 = vector.extract %795[0] : f16 from vector<4xf16>
    %797 = vector.extract %795[1] : f16 from vector<4xf16>
    %798 = vector.extract %795[2] : f16 from vector<4xf16>
    %799 = vector.extract %795[3] : f16 from vector<4xf16>
    scf.if %771 {
      affine.store %775, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
      affine.store %776, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
      affine.store %777, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
      affine.store %778, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      affine.store %782, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 48] : memref<1x32x4096x128xf16, 1>
      affine.store %783, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 49] : memref<1x32x4096x128xf16, 1>
      affine.store %784, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 50] : memref<1x32x4096x128xf16, 1>
      affine.store %785, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 51] : memref<1x32x4096x128xf16, 1>
      affine.store %789, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 32] : memref<1x32x4096x128xf16, 1>
      affine.store %790, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 33] : memref<1x32x4096x128xf16, 1>
      affine.store %791, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 34] : memref<1x32x4096x128xf16, 1>
      affine.store %792, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 35] : memref<1x32x4096x128xf16, 1>
      affine.store %796, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 48] : memref<1x32x4096x128xf16, 1>
      affine.store %797, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 49] : memref<1x32x4096x128xf16, 1>
      affine.store %798, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 50] : memref<1x32x4096x128xf16, 1>
      affine.store %799, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 51] : memref<1x32x4096x128xf16, 1>
    }
    %800 = vector.from_elements %420, %426, %421, %427 : vector<4xf32>
    %801 = arith.mulf %800, %739 : vector<4xf32>
    %802 = arith.truncf %801 {tiled = true} : vector<4xf32> to vector<4xf16>
    %803 = vector.extract %802[0] : f16 from vector<4xf16>
    %804 = vector.extract %802[1] : f16 from vector<4xf16>
    %805 = vector.extract %802[2] : f16 from vector<4xf16>
    %806 = vector.extract %802[3] : f16 from vector<4xf16>
    %807 = vector.from_elements %448, %454, %449, %455 : vector<4xf32>
    %808 = arith.mulf %807, %739 : vector<4xf32>
    %809 = arith.truncf %808 {tiled = true} : vector<4xf32> to vector<4xf16>
    %810 = vector.extract %809[0] : f16 from vector<4xf16>
    %811 = vector.extract %809[1] : f16 from vector<4xf16>
    %812 = vector.extract %809[2] : f16 from vector<4xf16>
    %813 = vector.extract %809[3] : f16 from vector<4xf16>
    %814 = vector.from_elements %476, %482, %477, %483 : vector<4xf32>
    %815 = arith.mulf %814, %756 : vector<4xf32>
    %816 = arith.truncf %815 {tiled = true} : vector<4xf32> to vector<4xf16>
    %817 = vector.extract %816[0] : f16 from vector<4xf16>
    %818 = vector.extract %816[1] : f16 from vector<4xf16>
    %819 = vector.extract %816[2] : f16 from vector<4xf16>
    %820 = vector.extract %816[3] : f16 from vector<4xf16>
    %821 = vector.from_elements %504, %510, %505, %511 : vector<4xf32>
    %822 = arith.mulf %821, %756 : vector<4xf32>
    %823 = arith.truncf %822 {tiled = true} : vector<4xf32> to vector<4xf16>
    %824 = vector.extract %823[0] : f16 from vector<4xf16>
    %825 = vector.extract %823[1] : f16 from vector<4xf16>
    %826 = vector.extract %823[2] : f16 from vector<4xf16>
    %827 = vector.extract %823[3] : f16 from vector<4xf16>
    scf.if %771 {
      affine.store %803, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
      affine.store %804, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
      affine.store %805, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
      affine.store %806, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      affine.store %810, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 80] : memref<1x32x4096x128xf16, 1>
      affine.store %811, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 81] : memref<1x32x4096x128xf16, 1>
      affine.store %812, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 82] : memref<1x32x4096x128xf16, 1>
      affine.store %813, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 83] : memref<1x32x4096x128xf16, 1>
      affine.store %817, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 64] : memref<1x32x4096x128xf16, 1>
      affine.store %818, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 65] : memref<1x32x4096x128xf16, 1>
      affine.store %819, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 66] : memref<1x32x4096x128xf16, 1>
      affine.store %820, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 67] : memref<1x32x4096x128xf16, 1>
      affine.store %824, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 80] : memref<1x32x4096x128xf16, 1>
      affine.store %825, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 81] : memref<1x32x4096x128xf16, 1>
      affine.store %826, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 82] : memref<1x32x4096x128xf16, 1>
      affine.store %827, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 83] : memref<1x32x4096x128xf16, 1>
    }
    %828 = vector.from_elements %308, %314, %309, %315 : vector<4xf32>
    %829 = arith.mulf %828, %739 : vector<4xf32>
    %830 = arith.truncf %829 {tiled = true} : vector<4xf32> to vector<4xf16>
    %831 = vector.extract %830[0] : f16 from vector<4xf16>
    %832 = vector.extract %830[1] : f16 from vector<4xf16>
    %833 = vector.extract %830[2] : f16 from vector<4xf16>
    %834 = vector.extract %830[3] : f16 from vector<4xf16>
    %835 = vector.from_elements %336, %342, %337, %343 : vector<4xf32>
    %836 = arith.mulf %835, %739 : vector<4xf32>
    %837 = arith.truncf %836 {tiled = true} : vector<4xf32> to vector<4xf16>
    %838 = vector.extract %837[0] : f16 from vector<4xf16>
    %839 = vector.extract %837[1] : f16 from vector<4xf16>
    %840 = vector.extract %837[2] : f16 from vector<4xf16>
    %841 = vector.extract %837[3] : f16 from vector<4xf16>
    %842 = vector.from_elements %364, %370, %365, %371 : vector<4xf32>
    %843 = arith.mulf %842, %756 : vector<4xf32>
    %844 = arith.truncf %843 {tiled = true} : vector<4xf32> to vector<4xf16>
    %845 = vector.extract %844[0] : f16 from vector<4xf16>
    %846 = vector.extract %844[1] : f16 from vector<4xf16>
    %847 = vector.extract %844[2] : f16 from vector<4xf16>
    %848 = vector.extract %844[3] : f16 from vector<4xf16>
    %849 = vector.from_elements %392, %398, %393, %399 : vector<4xf32>
    %850 = arith.mulf %849, %756 : vector<4xf32>
    %851 = arith.truncf %850 {tiled = true} : vector<4xf32> to vector<4xf16>
    %852 = vector.extract %851[0] : f16 from vector<4xf16>
    %853 = vector.extract %851[1] : f16 from vector<4xf16>
    %854 = vector.extract %851[2] : f16 from vector<4xf16>
    %855 = vector.extract %851[3] : f16 from vector<4xf16>
    scf.if %771 {
      affine.store %831, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
      affine.store %832, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
      affine.store %833, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
      affine.store %834, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      affine.store %838, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 112] : memref<1x32x4096x128xf16, 1>
      affine.store %839, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 113] : memref<1x32x4096x128xf16, 1>
      affine.store %840, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 114] : memref<1x32x4096x128xf16, 1>
      affine.store %841, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 115] : memref<1x32x4096x128xf16, 1>
      affine.store %845, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 96] : memref<1x32x4096x128xf16, 1>
      affine.store %846, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 97] : memref<1x32x4096x128xf16, 1>
      affine.store %847, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 98] : memref<1x32x4096x128xf16, 1>
      affine.store %848, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 99] : memref<1x32x4096x128xf16, 1>
      affine.store %852, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 112] : memref<1x32x4096x128xf16, 1>
      affine.store %853, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 113] : memref<1x32x4096x128xf16, 1>
      affine.store %854, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 114] : memref<1x32x4096x128xf16, 1>
      affine.store %855, %arg3[0, (symbol(%24) + symbol(%23) floordiv 32) mod 32, symbol(%23) * 128 + symbol(%thread_id_x) + (symbol(%thread_id_x) floordiv 64) * 16 - ((symbol(%thread_id_x) floordiv 64) floordiv 4) * 64 - (symbol(%thread_id_x) floordiv 16) * 16 - (symbol(%23) floordiv 32) * 4096 + 64, ((symbol(%thread_id_x) mod 64) floordiv 16) * 4 + 115] : memref<1x32x4096x128xf16, 1>
    }
    return
  }
}
